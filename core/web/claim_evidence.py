"""Read-only web access to the accepted current claim/paper evidence.

The large graph stays on disk. Existing acceptance-bound query functions own
scientific grouping, publication versions and support counts. This adapter only
caches a validated revision and adds search/pagination for the web explorer.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
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
                result = query_claim_evidence(self.path, claim_id=claim_id, relation_id=relation_id)
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
