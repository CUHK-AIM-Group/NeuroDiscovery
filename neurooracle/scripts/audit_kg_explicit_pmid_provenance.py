"""R36 read-only provenance and edge-reference closure audit; no record copies.

Original curation artifacts are read-only evidence, never executed or replayed.
PMIDs come from explicit original fields, not synthetic queue-ID suffixes.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import hashed_reader, walk_graph, compact
from inspect_kg_bibliography_sources import load_public_xml
from kg_accepted_candidate_lineage import require
from prune_current_kg import rows, write_rows
from reclaim_kg_backup_storage import sha256
from neurooracle.src.kg_bibliography_repair import quote_key, primary_abstract
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities, normalize_pmid, normalize_doi
from neurooracle.src.pubmed_title_evidence import complete_title_evidence
from neurooracle.src.kg_explicit_pmid_repair import abstract_witness, reference_changes

OUTPUT = journal.OUTPUT / "round36_explicit_pmid_provenance"
ARCHIVE = journal.REPO / "neurooracle/data/archive/general_neuromed_manual_20260610_legacy_20260722/batch_reviews"
R35 = journal.OUTPUT / "round35_source_bibliography"


def progress(phase, **values):
    state = dict(status="READ_ONLY_AUDITING", phase=phase, at=journal.utc_now(), **values)
    journal.atomic_json(OUTPUT / "AUDIT_STATE.json", state); print(compact(state), flush=True)


class Archive:
    def __init__(self):
        self.batches, self.files = {}, {}

    def load(self, batch):
        require(type(batch) is int and 1 <= batch <= 9999, "invalid archive batch")
        if batch in self.batches: return self.batches[batch]
        paths = {"input": ARCHIVE / f"batch_{batch}_input.json", "claims": ARCHIVE / f"batch_{batch}_claims.jsonl",
                 "script": ARCHIVE / f"curate_batch_{batch}_manual.js"}
        for path in paths.values():
            require(path.resolve().parent == ARCHIVE.resolve() and path.is_file(), "missing/unsafe original curation artifact")
            self.files[str(path)] = journal.fingerprint(path)
        source = journal.read_json(paths["input"])
        inputs = {r["batch_index"]: r for r in source["items"]}
        claims = {}
        for row in rows(paths["claims"]):
            index = row["metadata"]["input_idx"]
            claims.setdefault(index, []).append(row)
        require(source["batch"] == batch and len(inputs) == len(source["items"]), "duplicate/wrong archive input")
        result = dict(inputs=inputs, claims=claims, paths=paths)
        self.batches[batch] = result
        return result


def review_pmid(row, expected, registry, articles, xml_witnesses, archive):
    md = row["metadata"]; p = md["source_paper"]; inner = md.get("metadata") or {}
    require(digest(row) == expected["claim_sha256"], "selected current claim changed")
    pmid = md.get("pmid")
    if not isinstance(pmid, str) or normalize_pmid(pmid) != pmid or pmid != expected["alternate_pmid"]:
        return None, "explicit_outer_pmid_not_exact_valid_identifier"
    if inner.get("dataset") != "general_neuromed_manual_20260610" or md.get("source") != "manual_abstract_review":
        return None, "unreviewed_import_family"
    batch, index = inner.get("batch"), inner.get("input_idx")
    if inner.get("batch_index") != index:
        return None, "ambiguous_import_row"
    originals = archive.load(batch)
    item = originals["inputs"].get(index)
    matches = [r for r in originals["claims"].get(index, []) if r.get("raw_sentence") == md.get("raw_text")]
    if item is None or len(matches) != 1:
        return None, "original_curated_claim_not_unique"
    claim = matches[0]
    if not (item["pmid"] == claim["pmid"] == pmid and claim["paper_id"] == item["queue_id"] == md.get("paper_id") == inner.get("queue_id") and
            item["source_snapshot_index"] == claim["metadata"]["source_snapshot_index"] == inner.get("source_snapshot_index") and
            inner.get("script") == originals["paths"]["script"].name):
        return None, "original_explicit_identifier_or_locator_disagreement"
    if not (md["raw_text"] == claim["evidence"] == claim["evidence_text"] == (md.get("evidence") or {}).get("legacy_value")):
        return None, "original_evidence_changed"
    # These are complete unchanged bibliography fields, not title similarity.
    if any(p.get(k) != claim.get(k) or claim.get(k) != item.get(k) for k in ("title", "year", "journal")):
        return None, "complete_original_bibliography_disagrees"
    if not (normalize_doi(p.get("doi")) == normalize_doi(claim.get("doi")) == normalize_doi(item.get("doi"))):
        return None, "original_doi_disagrees"
    # Accept either a missing PMID or the exact queue ID already in the proven
    # source records. Never interpret any substring of that queue ID as an ID.
    if p.get("pmid") not in ("", claim["paper_id"]):
        return None, "source_contains_another_identifier"
    if (inner.get("pmid") not in (None, "", pmid) or p.get("arxiv_id") or
        re.search(r"\b(?:biorxiv|medrxiv|arxiv|preprints?)\b", str(p.get("journal") or ""), re.I)):
        return None, "alternate_identifier_or_version_requires_review"
    authority = registry.export_payload()["records"].get(pmid)
    article = articles.get(pmid)
    if authority is None or article is None or article.tag != "PubmedArticle" or authority["preprint"]:
        return None, "complete_owning_article_evidence_missing"
    complete_title_evidence(article, authority, xml_witnesses[pmid])
    abstract = abstract_witness(article, item.get("abstract"), pmid)
    if abstract is None:
        return None, "original_full_abstract_differs_from_current_owning_xml"
    out = deepcopy(row); out["metadata"]["source_paper"]["pmid"] = pmid
    require(out["metadata"]["pmid"] == out["metadata"]["source_paper"]["pmid"], "duplicate PMID not exact")
    del out["metadata"]["pmid"]
    after = registry.resolve(out["metadata"])
    if after != dict(paper_key="pmid:" + pmid, status="verified", reasons=[]):
        return None, "full_identity_guard_still_fails"
    return dict(claim_id=row["id"], source_node_sha256=digest(row), current_node_sha256=digest(out),
        set_source_paper_pmid=pmid, remove_exact_duplicate_outer_pmid=True, canonical_paper_key=after["paper_key"],
        raw_text_sha256=hashlib.sha256(md["raw_text"].encode()).hexdigest(), original_curated_record_sha256=digest(claim),
        original_input_record_sha256=digest(item), abstract_witness=abstract,
        source_pmid_shape="missing" if p.get("pmid") == "" else "original_queue_id",
        archive_batch=batch, archive_input_index=index, archive_files={k:archive.files[str(v)] for k,v in originals["paths"].items()},
        public_xml_witness=xml_witnesses[pmid]), "explicit_original_pmid_and_complete_abstract_provenance"


def proposed_node(row, event):
    require(digest(row) == event["source_node_sha256"], "node changed after review")
    out = deepcopy(row); out["metadata"]["source_paper"]["pmid"] = event["set_source_paper_pmid"]
    require(out["metadata"].pop("pmid") == event["set_source_paper_pmid"], "outer PMID differs")
    require(digest(out) == event["current_node_sha256"], "proposed node differs")
    return out


def main(refine=False):
    OUTPUT.mkdir(exist_ok=True)
    previous = None
    if (OUTPUT / "PROVENANCE_AUDIT.json").exists():
        require(refine and not (OUTPUT / "BUILD_STATE.json").exists() and not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "audit exists; inspect/reuse")
        previous = journal.read_json(OUTPUT / "PROVENANCE_AUDIT.json")
    c = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(c["status"] == "COMPLETED" and c["active_process"] is None, "writer active")
    if previous: require(previous["graph"] == c["current_graph"], "refinement source advanced")
    census = journal.read_json(c["current_paper_census"]["path"])
    journal.guards([c["current_graph"], c["current_detail_store"], c["formal_sources"], census["database"]])
    targets = {r["claim_id"]: r for r in census["outer_pmid_conflicts"]}
    title_events = {r["claim_id"]: r for r in rows(R35 / "TITLE_REPAIRS.jsonl")}
    require(len(targets) == 219 and len(title_events) == 163 and not set(targets) & set(title_events), "unreviewed scope")
    registry = VerifiedPaperIdentities(journal.read_json(c["current_paper_identities"]["path"]))
    articles, witnesses = load_public_xml(); archive = Archive()
    node_events, held, originals, edge_events, edge_holds = {}, [], {}, [], []
    edge_counts, kinds, prior_reference_shapes = Counter(), Counter(), Counter()
    code_paths = [Path(__file__), journal.REPO / "neurooracle/src/kg_explicit_pmid_repair.py"]
    code = [journal.fingerprint(p) for p in code_paths]
    selected = set(targets) | set(title_events)
    progress("SOURCE_AND_REFERENCE_CLOSURE", selected_claims=len(selected))
    with hashed_reader(Path(c["current_graph"]["path"])) as (reader, source_sha):
        for kind, key, row in walk_graph(reader):
            kinds[kind] += 1
            if kind == "node" and key in selected:
                originals[key] = row
                if key in targets:
                    event, reason = review_pmid(row, targets[key], registry, articles, witnesses, archive)
                    if event: node_events[key] = event
                    else: held.append(dict(claim_id=key, claim_sha256=digest(row), reason=reason, pmid=targets[key]["alternate_pmid"]))
                else: require(digest(row) == title_events[key]["current_node_sha256"], "R35 current title record changed")
            if kind == "edge":
                owner = (row.get("metadata") or {}).get("claim_id")
                if owner in selected:
                    require(owner in originals, "owner record missing")
                    edge_counts[owner] += 1
                    md = originals[owner]["metadata"]
                    prior_reference_shapes[("pmid" if owner in targets else "title", row["relation_type"], str(row.get("source")), str(row.get("evidence_ref")) == str(md["source_paper"].get("title")))] += 1
                    changed, bases, reference_holds = reference_changes(row, originals[owner], node_events.get(owner), title_events.get(owner))
                    edge_holds.extend(dict(ordinal=int(key), claim_id=owner, edge_sha256=digest(row), field=field, reason=reason) for field, reason in reference_holds)
                    if changed:
                        out = deepcopy(row); out.update(changed)
                        edge_events.append(dict(ordinal=int(key), claim_id=owner, source_edge_sha256=digest(row), current_edge_sha256=digest(out), set_fields=changed, reference_basis=bases))
            if kinds[kind] % 500000 == 0:
                progress(kind, count=kinds[kind], selected_seen=len(originals), approved_nodes=len(node_events), approved_edges=len(edge_events))
        require(source_sha.hexdigest() == c["current_graph"]["sha256"], "full current graph SHA differs")
    require(set(originals) == selected and set(edge_counts) == selected, "selected node/owned-edge closure incomplete")
    require(sha256(Path(census["database"]["path"])) == census["database"]["sha256"], "current census SHA differs")
    for fp in archive.files.values(): require(journal.fingerprint(Path(fp["path"])) == fp, "original evidence changed during audit")
    require([journal.fingerprint(p) for p in code_paths] == code, "audit implementation changed")
    journal.guards([c["current_graph"], c["current_detail_store"], c["formal_sources"], census["database"]])
    require(journal.read_json(journal.OUTPUT / "CAMPAIGN.json") == c, "campaign advanced during audit")
    write_rows(OUTPUT / "PMID_REPAIRS.jsonl", list(node_events.values()))
    write_rows(OUTPUT / "EDGE_REFERENCE_REPAIRS.jsonl", edge_events)
    write_rows(OUTPUT / "PMID_HOLDS.jsonl", held); write_rows(OUTPUT / "EDGE_REFERENCE_HOLDS.jsonl", edge_holds)
    result = dict(status="AUDITED_NOT_APPLIED", at=journal.utc_now(), graph=c["current_graph"], source_acceptance=c["current_acceptance"],
        source_runtime=c["current_runtime_acceptance"], source_census=c["current_paper_census"], source_registry=c["current_paper_identities"],
        source_full_sha_verified=True, census_full_sha_verified=True, selected_nodes=len(selected), source_counts=dict(kinds),
        approved_pmid_nodes=len(node_events), pmid_holds=len(held), hold_reasons=dict(Counter(r["reason"] for r in held)),
        archive_batches=sorted(archive.batches), original_archive_witnesses=list(archive.files.values()),
        approved_edge_records=len(edge_events), edge_field_changes=dict(Counter(k for e in edge_events for k in e["set_fields"])),
        abstract_variants=dict(Counter(e["abstract_witness"]["mode"] for e in node_events.values())),
        reference_bases=dict(Counter(v for e in edge_events for v in e["reference_basis"].values())),
        edge_reference_holds=len(edge_holds), edge_hold_reasons=dict(Counter(r["reason"] for r in edge_holds)),
        selected_owned_edge_counts=dict(Counter(edge_counts.values())),
        repairs=journal.fingerprint(OUTPUT / "PMID_REPAIRS.jsonl"), edge_repairs=journal.fingerprint(OUTPUT / "EDGE_REFERENCE_REPAIRS.jsonl"),
        title_repair_predecessor=journal.fingerprint(R35 / "TITLE_REPAIRS.jsonl"), xml_fetch=journal.fingerprint(journal.OUTPUT / "round33_complete_title_evidence/XML_FETCH.json"),
        code=code, record_preimages_saved=False, graph_modified=False,
        original_curation_policy_observation="Inspected legacy curator explicitly stages one claim per paper; source scarcity is not solely duplicate-node identity. No extraction rule or model changed.")
    journal.atomic_json(OUTPUT / "PROVENANCE_AUDIT.json", result)
    journal.atomic_json(OUTPUT / "AUDIT_STATE.json", {k:v for k,v in result.items() if k in ("status","at","approved_pmid_nodes","pmid_holds","approved_edge_records","edge_field_changes","edge_reference_holds")})
    print(compact({k:v for k,v in result.items() if k in ("status","approved_pmid_nodes","pmid_holds","hold_reasons","archive_batches","approved_edge_records","edge_field_changes","edge_reference_holds","edge_hold_reasons","selected_owned_edge_counts")}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--refine", action="store_true")
    main(parser.parse_args().refine)
