"""Read-only web access to the accepted current claim/paper evidence.

The large graph stays on disk. Existing acceptance-bound query functions own
scientific grouping, publication versions and support counts. This adapter only
caches a validated revision and adds search/pagination for the web explorer.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from contextlib import closing
import json
import mmap
import os
from pathlib import Path
import sqlite3
import threading

from neurooracle.src.claim_evidence_query import paper_evidence, query_claim_evidence
from neurooracle.src.relation_evidence_dossier import find_relation_dossiers
from neurooracle.src.shared_relation_catalog import check_file


class EvidenceUnavailable(RuntimeError):
    pass


def configured_campaign(repo_root: Path) -> Path | None:
    configured = os.environ.get("NEUROCLAW_KG_CAMPAIGN")
    if not configured:
        config = repo_root / "neurooracle/configs/graph_explorer.json"
        if not config.exists():
            return None
        payload = json.loads(config.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("campaign"), str):
            raise ValueError("Invalid graph explorer configuration")
        configured = payload["campaign"]
    if not configured.strip():
        raise ValueError("Empty accepted graph campaign path")
    path = Path(configured)
    return (path if path.is_absolute() else repo_root / path).resolve()


class AcceptedClaimEvidence:
    INPUTS = (
        "current_graph", "current_acceptance", "current_paper_census",
        "current_shared_relations", "current_evidence_dossiers",
        "current_entity_terms", "current_paper_identities", "current_source_role_reviews",
    )
    COUNTS = (
        "article_count", "publication_record_count", "verified_versions_deduplicated",
        "verified_article_count", "default_counted_article_count",
        "reviewed_supporting_article_count", "own_result_article_count",
        "background_article_count", "review_article_count", "hypothesis_article_count",
    )

    def __init__(self, campaign_path: Path | None):
        self.path = campaign_path
        self._lock = threading.RLock()
        self._campaign = None
        self._database = None
        self._summaries = []
        self._member_relations = {}
        self._search = {}
        self._details = OrderedDict()
        self._snapshot_campaign = None
        self._global_index_fp = None
        self._global_index = None
        self._original_resources = None

    def _read_current(self):
        if self.path is None:
            raise EvidenceUnavailable("No accepted claim evidence source is configured")
        if not self.path.is_file():
            raise EvidenceUnavailable("The configured accepted claim evidence source is unavailable")
        campaign = json.loads(self.path.read_text(encoding="utf-8"))
        if campaign.get("status") != "COMPLETED" or campaign.get("active_process") is not None:
            raise EvidenceUnavailable("The graph is being updated; retry after its acceptance completes")
        return campaign

    def _check_current(self, campaign):
        for name in self.INPUTS:
            if campaign.get(name):
                check_file(campaign[name])
        if self._database:
            check_file(self._database)
        if self._global_index_fp:
            check_file(self._global_index_fp)
        if self._global_index:
            check_file(self._global_index["database"])
        if self._read_current() != campaign:
            raise EvidenceUnavailable("The graph revision changed during the request; retry")

    def _ensure_current(self):
        campaign = self._read_current()
        if campaign == self._campaign:
            self._check_current(campaign)
            return campaign
        self._campaign = None
        self._database = None
        self._details.clear()
        self._global_index_fp = None
        self._global_index = None
        self._original_resources = None
        receipt = json.loads(check_file(campaign["current_acceptance"], full_hash=True).read_text(encoding="utf-8"))
        census_fp = campaign["current_paper_census"]
        if (receipt.get("current_paper_census") != census_fp or
                not receipt.get("checks", {}).get("all_current_census_rows_independently_verified")):
            raise EvidenceUnavailable("The accepted claim census has not been validated")
        census = json.loads(check_file(census_fp, full_hash=True).read_text(encoding="utf-8"))
        if census["graph"] != campaign["current_graph"]:
            raise EvidenceUnavailable("The claim census belongs to another graph revision")
        check_file(census["database"])
        groups = find_relation_dossiers(self.path, minimum_papers=1,
                                       verified_papers_only=False, include_retracted=True)
        summaries, search = [], {}
        for item in groups:
            relation, dossier = item["relation"], item["evidence"]
            evidence = paper_evidence(dossier)
            row = dict(shared_claim_id=relation["id"],
                       claim={k: relation[k] for k in ("subject_id", "subject_name", "predicate", "object_id", "object_name")},
                       original_claim_ids=sorted(o["claim_id"] for o in dossier["observations"]),
                       observation_count=len(dossier["observations"]),
                       **{k: evidence[k] for k in self.COUNTS})
            summaries.append(row)
            terms = [relation["id"], *row["claim"].values(), *row["original_claim_ids"]]
            for paper in evidence["papers"]:
                for version in paper["publication_versions"]:
                    b = version["bibliography"]
                    terms.extend(str(b.get(k) or "") for k in ("pmid", "doi", "title"))
            search[relation["id"]] = " ".join(terms).casefold()
        self._database = census["database"]
        self._check_current(campaign)
        self._summaries = sorted(summaries, key=lambda r: (
            -r["reviewed_supporting_article_count"], -r["article_count"],
            r["claim"]["subject_name"].casefold(), r["shared_claim_id"]))
        self._search = search
        self._member_relations = {cid: row["shared_claim_id"] for row in summaries for cid in row["original_claim_ids"]}
        self._campaign = campaign
        return campaign

    @staticmethod
    def _revision(campaign):
        return dict(graph_revision=campaign["current_graph"]["sha256"],
                    source="accepted_current_graph", independence_established=False,
                    consensus_inferred=False)

    def status(self):
        if self.path is None:
            return dict(configured=False, available=False)
        with self._lock:
            campaign = self._ensure_current()
            return dict(configured=True, available=True,
                        shared_claim_count=len(self._summaries),
                        reviewed_multipaper_claim_count=sum(r["reviewed_supporting_article_count"] >= 2 for r in self._summaries),
                        **self._revision(campaign))

    def search(self, query="", *, minimum_papers=2, offset=0, limit=30):
        if not isinstance(query, str) or len(query) > 300:
            raise ValueError("Search must contain at most 300 characters")
        if type(minimum_papers) is not int or not 0 <= minimum_papers <= 10000:
            raise ValueError("Invalid minimum supporting paper count")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid search page")
        with self._lock:
            campaign = self._ensure_current()
            words = query.casefold().split()
            matches = [r for r in self._summaries if r["reviewed_supporting_article_count"] >= minimum_papers
                       and all(word in self._search[r["shared_claim_id"]] for word in words)]
            if words:
                def relevance(row):
                    claim = " ".join(row["claim"].values()).casefold()
                    return (query.casefold() not in claim, not all(word in claim for word in words))
                matches.sort(key=relevance)
            self._check_current(campaign)
            return dict(total=len(matches), offset=offset, limit=limit,
                        claims=deepcopy(matches[offset:offset + limit]),
                        minimum_reviewed_supporting_papers=minimum_papers, **self._revision(campaign))

    def claim_index(self):
        """The cached (summary, searchable text) pairs, under the current revision.

        Read-only view for topic aggregation; it never adds, renames or re-extracts
        a scientific claim, so callers must treat the text as retrieval keys only.
        """
        with self._lock:
            campaign = self._ensure_current()
            index = [(deepcopy(summary), self._search.get(summary["shared_claim_id"], ""))
                     for summary in self._summaries]
            self._check_current(campaign)
            return index

    @contextmanager
    def read_snapshot(self):
        """Keep retrieval and evidence expansion within one accepted revision."""
        with self._lock:
            campaign = self._ensure_current()
            revision = self._revision(campaign)
            previous = self._snapshot_campaign
            self._snapshot_campaign = campaign
            try:
                yield revision
            finally:
                self._snapshot_campaign = previous
            self._check_current(campaign)
            if self._revision(campaign) != revision:
                raise EvidenceUnavailable("Evidence dependencies changed during topic retrieval")

    def topic_search(self, **kwargs):
        from core.topic_evidence import topic_evidence
        return topic_evidence(self, **kwargs)

    def idea_hypotheses(self, **kwargs):
        from core.idea_hypotheses import generate_hypotheses
        return generate_hypotheses(self, **kwargs)

    def _original_index(self, campaign):
        """Use the existing sealed retrieval index; never create one on a read."""
        if self._global_index_fp is None:
            return None
        if self._global_index is None:
            manifest = json.loads(check_file(self._global_index_fp, full_hash=True).read_text(encoding="utf-8"))
            if (manifest.get("schema") != "kg.global_claim_index.acceptance.v1" or
                    manifest.get("status") != "COMPLETE_RETRIEVAL_INDEX_NOT_SCIENTIFIC_APPROVAL" or
                    manifest.get("graph") != campaign["current_graph"] or
                    manifest.get("all_claim_hashes_verified") is not True):
                raise EvidenceUnavailable("Original claim index does not cover this accepted graph")
            check_file(manifest["database"])
            self._global_index = manifest
        return self._global_index

    def original_claim_candidates(self, matchers, *, minimum_matches, limit):
        """Bounded single-paper recall from original endpoints and source IDs.

        Shared/projection members are excluded. One best observation per source
        keeps a prolific paper from consuming the evidence budget. Counts and
        scientific fields are only established by query_batch, never by this index.
        """
        with self._lock:
            campaign = self._ensure_current()
            manifest = self._original_index(campaign)
            if manifest is None:
                return dict(available=False, indexed=0, eligible=0, matched=0, candidates=[])
            import re
            any_term = re.compile("|".join(pattern.pattern for _, pattern in matchers), re.IGNORECASE)
            best, matched, scanned, covered = {}, 0, 0, 0
            path = check_file(manifest["database"])
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                rows = db.execute("SELECT cid,rid,subject_id,subject_name,predicate,object_id,object_name,"
                                  "pmid,doi,source_key,paper_sig FROM observations ORDER BY number")
                for cid, rid, sid, subject, predicate, oid, obj, pmid, doi, source, signature in rows:
                    scanned += 1
                    if cid in self._member_relations:
                        covered += 1
                        continue
                    text = " ".join(str(v or "") for v in (sid, subject, predicate, oid, obj, cid, pmid, doi))
                    if not any_term.search(text):
                        continue
                    hits = [term for term, pattern in matchers if pattern.search(text)]
                    if len(hits) < minimum_matches:
                        continue
                    matched += 1
                    key = source or signature or cid
                    rank = (-len(hits), subject.casefold(), cid)
                    if key not in best or rank < best[key][0]:
                        best[key] = (rank, dict(
                            shared_claim_id=rid, original_claim_ids=[cid], query_id=cid,
                            retrieval_origin="original_singleton", matched_terms=hits,
                            claim=dict(subject_id=sid, subject_name=subject, predicate=predicate,
                                       object_id=oid, object_name=obj), observation_count=1,
                            **{name: None for name in self.COUNTS}))
            candidates = [row for _, row in sorted(best.values(), key=lambda item: item[0])[:limit]]
            if scanned != manifest["all_current_claims_indexed"]:
                raise EvidenceUnavailable("Original index row count differs from its acceptance")
            self._check_current(campaign)
            return dict(available=True, indexed=scanned, eligible=scanned - covered,
                        matched=matched, matched_sources=len(best), candidates=candidates)

    def _indexed_singleton(self, campaign, cid):
        """Same census/identity/evidence owners as query_claim_evidence, by offset."""
        manifest = self._original_index(campaign)
        if manifest is None:
            return query_claim_evidence(self.path, claim_id=cid)
        from neurooracle.scripts.project_kg_systematic_review import record_at
        from neurooracle.src.kg_identity_pilot import digest
        from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
        from neurooracle.src.correlation_grouping import IndexTerms
        from neurooracle.src.verified_entity_terms import VerifiedEntityTerms
        from neurooracle.src.relation_evidence import summarize_relation
        from neurooracle.src.relation_evidence_dossier import observation, summarize

        with closing(sqlite3.connect(check_file(manifest["database"]).as_uri() + "?mode=ro", uri=True)) as db:
            witness = db.execute("SELECT node_sha,byte_offset,rid FROM observations WHERE cid=?", (cid,)).fetchone()
        if witness is None:
            raise KeyError("claim not found: " + cid)
        seal, offset, rid = witness
        with closing(sqlite3.connect(check_file(self._database).as_uri() + "?mode=ro", uri=True)) as db:
            members = db.execute("SELECT cid,node_sha,shared FROM claims WHERE relation_id=?", (rid,)).fetchall()
        if members != [(cid, seal, 0)]:
            raise EvidenceUnavailable("Original index and singleton census membership differ")
        with check_file(campaign["current_graph"]).open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            record = record_at(mm, offset)
        if record.get("id") != cid or digest(record) != seal:
            raise EvidenceUnavailable("Indexed observation does not match its current census seal")
        if self._original_resources is None:
            def read(name):
                return json.loads(check_file(campaign[name], full_hash=True).read_text(encoding="utf-8"))
            receipt = read("current_acceptance")
            for name, key in (("current_paper_identities", "paper_identities"),
                              ("current_entity_terms", "entity_terms"),
                              ("current_source_role_reviews", "source_role_reviews")):
                if receipt.get(key) != campaign.get(name):
                    raise EvidenceUnavailable("Unaccepted singleton identity or source review input")
            self._original_resources = (VerifiedPaperIdentities(read("current_paper_identities")),
                                        IndexTerms(VerifiedEntityTerms(read("current_entity_terms"))),
                                        read("current_source_role_reviews") if campaign.get("current_source_role_reviews") else {})
        registry, terms, reviews = self._original_resources
        group = summarize_relation(terms.relation_key(record["metadata"]), [record["metadata"]], identities=terms, papers=registry)
        if group["id"] != rid:
            raise EvidenceUnavailable("Singleton relation differs from current census")
        dossier = summarize(group, [observation(record, registry, reviews)])
        return dict(requested_claim_id=cid, shared_claim_id=rid,
                    claim={k: group[k] for k in ("subject_id", "subject_name", "predicate", "object_id", "object_name")},
                    original_claim_ids=[cid], observation_count=1, **paper_evidence(dossier),
                    scope="complete current graph membership of this reviewed fine relation; not complete literature recall")

    def adjacent_claim_ids(self, node_ids, *, predicates, limit=64):
        """One bounded original-index hop for a topic's typed chain candidates."""
        nodes = sorted(set(node_ids))
        relations = sorted(set(predicates))
        if not nodes or not relations:
            return dict(claim_ids=[], truncated=False)
        if len(nodes) > 128 or not 1 <= limit <= 96:
            raise ValueError("Too many chain anchors or adjacent claims")
        with self._lock:
            campaign = self._ensure_current()
            manifest = self._original_index(campaign)
            if manifest is None:
                return dict(claim_ids=[], truncated=False)
            marks = ','.join('?' for _ in nodes)
            predicates_sql = ','.join('?' for _ in relations)
            with closing(sqlite3.connect(check_file(manifest['database']).as_uri() + '?mode=ro', uri=True)) as db:
                rows = db.execute(f"SELECT cid FROM observations WHERE (subject_id IN ({marks}) OR object_id IN ({marks})) "
                                  f"AND predicate IN ({predicates_sql}) AND subject_type NOT IN ('null','', '\"\"') "
                                  "AND object_type NOT IN ('null','', '\"\"') ORDER BY number LIMIT ?",
                                  [*nodes, *nodes, *relations, limit + 1]).fetchall()
            self._check_current(campaign)
            return dict(claim_ids=[r[0] for r in rows[:limit]], truncated=len(rows) > limit)

    def chain_index(self, *, predicates):
        """Read lightweight typed edges from the existing sealed original index.

        This is a recall view, not trusted evidence. Retained paths must still
        resolve their CLM records through query_batch and validate actual nodes.
        No per-paper representative or arbitrary neighbor cutoff is applied.
        """
        with self._lock:
            campaign = self._ensure_current()
            manifest = self._original_index(campaign)
            if manifest is None:
                return dict(available=False, indexed=0, records=[])
            relations = sorted(set(predicates))
            records = []
            if relations:
                marks = ','.join('?' for _ in relations)
                with closing(sqlite3.connect(check_file(manifest['database']).as_uri() + '?mode=ro', uri=True)) as db:
                    rows = db.execute('SELECT cid,subject_id,subject_name,subject_type,predicate,object_id,object_name,object_type,'
                                      f'pmid,doi,source_key FROM observations WHERE predicate IN ({marks}) '
                                      "AND subject_type NOT IN ('null','', '\"\"') AND object_type NOT IN ('null','', '\"\"') ORDER BY number", relations)
                    for cid, sid, sn, st, predicate, oid, on, ot, pmid, doi, source in rows:
                        records.append(dict(id=cid, subject_id=sid, subject_name=sn, subject_type=json.loads(st),
                                            predicate=predicate, object_id=oid, object_name=on, object_type=json.loads(ot),
                                            source_key=source, pmid=pmid, doi=doi))
            self._check_current(campaign)
            return dict(available=True, indexed=manifest['all_current_claims_indexed'], records=records)

    def pair_support_counts(self, pairs):
        """Count distinct indexed sources joining each endpoint pair, in either direction.

        This is graph exposure, not a prior-art search or a whole-chain novelty verdict.
        Scan the existing sealed index once; never create an index or write the graph.
        """
        pairs = {tuple(sorted(pair)) for pair in pairs}
        with self._lock:
            campaign = self._ensure_current()
            manifest = self._original_index(campaign)
            if manifest is None:
                return None
            sources = {pair: set() for pair in pairs}
            if pairs:
                with closing(sqlite3.connect(check_file(manifest['database']).as_uri() + '?mode=ro', uri=True)) as db:
                    for s, t, work in db.execute('SELECT subject_id,object_id,source_key FROM observations'):
                        if not isinstance(s, str) or not isinstance(t, str):
                            continue
                        pair = tuple(sorted((s, t)))
                        if pair in sources and work:
                            sources[pair].add(work)
            self._check_current(campaign)
            return {pair: len(works) for pair, works in sources.items()}

    def graph_nodes(self, node_ids):
        """Read requested actual concept records; absent nodes stay absent."""
        from neurooracle.scripts.project_kg_systematic_review import record_at
        nodes = sorted(set(node_ids))
        if len(nodes) > 128:
            raise ValueError('Too many chain nodes')
        with self._lock:
            campaign = self._ensure_current()
            found = {}
            with check_file(campaign['current_graph']).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                start = mm.find(b'"concepts":{')
                if start < 0:
                    raise EvidenceUnavailable('Unsupported graph layout for concept lookup')
                for nid in nodes:
                    if not isinstance(nid, str) or nid.startswith('CLM:'):
                        continue
                    key = json.dumps(nid, ensure_ascii=False).encode('utf-8') + b':'
                    pos = mm.find(key, start)
                    if pos < 0:
                        continue
                    record = record_at(mm, pos + len(key))
                    if record.get('id') == nid and record.get('preferred_name'):
                        found[nid] = record
            self._check_current(campaign)
            return found

    def query(self, *, claim_id=None, relation_id=None):
        if bool(claim_id) == bool(relation_id):
            raise ValueError("Provide exactly one original claim ID or shared claim ID")
        value = claim_id or relation_id
        if not isinstance(value, str) or len(value) > 300 or not value.startswith("CLM:" if claim_id else "REL:"):
            raise ValueError("Invalid claim ID")
        with self._lock:
            campaign = self._ensure_current()
            shared_id = relation_id or self._member_relations.get(claim_id)
            key = (None, shared_id) if shared_id else (claim_id, None)
            if key not in self._details:
                result = (self._indexed_singleton(campaign, claim_id) if claim_id and not shared_id
                          else query_claim_evidence(self.path, claim_id=claim_id, relation_id=relation_id))
                self._check_current(campaign)
                self._details[key] = result
                if len(self._details) > 64:
                    self._details.popitem(last=False)
            self._details.move_to_end(key)
            result = deepcopy(self._details[key])
            if claim_id is not None and claim_id not in result["original_claim_ids"]:
                raise EvidenceUnavailable("The original claim is absent from its complete shared relation")
            result["requested_claim_id"] = claim_id
            self._check_current(campaign)
            return dict(result, **self._revision(campaign))
