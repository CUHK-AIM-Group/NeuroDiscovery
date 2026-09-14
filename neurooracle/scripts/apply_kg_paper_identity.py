"""R31: adopt authority-backed article counting with every KG node/edge intact.

Exactly one temporary KG for atomic replacement; no old KG or record preimages.
Frozen scripts, current-source SHA, independent candidate scan, and public
bibliography witnesses must all pass before the current work KG is replaced.
"""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from apply_kg_relation_identity import stream_patch, verify_catalog_and_graph
from build_umls_simplification_candidate import cheap, compact, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import require
from prune_current_kg import guard_native, rows, write_rows
from reclaim_kg_backup_storage import native_info, sha256
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_paper_identity import VERSION, VerifiedPaperIdentities, pubmed_record, census_identifier_projection
from neurooracle.src.case_study_membership_contract import strongest_paper_key
from neurooracle.src.verified_entity_terms import VerifiedEntityTerms

OUTPUT = journal.OUTPUT / "round31_paper_identity"
SOURCE = journal.OUTPUT / "round23_source_deletion_candidate/knowledge_graph.candidate.json"
TEMP = SOURCE.with_name(SOURCE.name + ".paper-identity.tmp")
CATALOG = OUTPUT / "CURRENT_SHARED_RELATIONS.jsonl"
PAPERS = OUTPUT / "CURRENT_PAPER_IDENTITIES.json"
FILES = [Path(__file__), *[journal.REPO / p for p in (
    "neurooracle/scripts/apply_kg_relation_identity.py", "neurooracle/scripts/audit_kg_paper_authorities.py",
    "neurooracle/scripts/build_umls_simplification_candidate.py", "neurooracle/scripts/prune_current_kg.py",
    "neurooracle/scripts/reclaim_kg_backup_storage.py", "neurooracle/scripts/kg_accepted_candidate_lineage.py",
    "neurooracle/src/kg_paper_identity.py", "neurooracle/src/verified_entity_terms.py", "neurooracle/src/kg_identity_pilot.py",
    "neurooracle/src/relation_evidence.py", "neurooracle/src/claim_evidence_identity.py", "neurooracle/src/claim_ingestion.py",
    "neurooracle/src/graph_manager.py", "neurooracle/src/storage.py", "neurooracle/src/shared_relation_catalog.py",
    "neurooracle/src/case_study_membership_contract.py", "neurooracle/src/schema.py", "neurooracle/src/kg_metadata_compaction.py",
    "neurooracle/tests/test_kg_paper_authorities.py", "neurooracle/tests/test_paper_identity.py")]]


def code_bindings(): return [journal.fingerprint(p) for p in FILES]


def progress(phase, **details):
    state = dict(status="RUNNING", phase=phase, pid=os.getpid(), at=journal.utc_now(), **details)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", state); print(compact(state), flush=True)


def verify_authorities(audit):
    require(journal.fingerprint(PAPERS) == audit["registry"], "paper registry changed")
    require(journal.fingerprint(OUTPUT / "AUTHORITY_FETCH.json") == audit["authority_fetch"], "authority manifest changed")
    for code in audit["code"]:
        require(journal.fingerprint(Path(code["path"])) == code, "audit implementation changed since adjudication")
    fetched = journal.read_json(OUTPUT / "AUTHORITY_FETCH.json")
    records = {}
    for witness in fetched["witnesses"]:
        require(journal.fingerprint(Path(witness["response"]["path"])) == witness["response"], "authority response changed")
        result = journal.read_json(witness["response"]["path"])["result"]
        for pmid in result["uids"]:
            if "error" in result[pmid]: continue
            require(pmid not in records, "duplicate authority PMID")
            records[pmid] = pubmed_record(pmid, result[pmid], dict(response_sha256=witness["response"]["sha256"],
                response_file=Path(witness["response"]["path"]).name, url=witness["url"]))
    payload = journal.read_json(PAPERS)
    require(payload["records"] == records, "registry not reproduced from authority responses")
    census = journal.read_json(OUTPUT / "CENSUS.json")
    require(journal.fingerprint(OUTPUT / "CENSUS.json") == audit["census"], "census changed")
    journal.guards(census["database"])
    require(sha256(Path(census["database"]["path"])) == census["database"]["sha256"], "census database full SHA mismatch")
    db = sqlite3.connect(Path(census["database"]["path"]).as_uri() + "?mode=ro", uri=True)
    projection = census_identifier_projection(db)
    db.close()
    require(journal.fingerprint(OUTPUT / "CENSUS_NORMALIZATION.json") == audit["census_normalization"], "census normalization evidence changed")
    require(payload["observed_collisions"] == projection["observed_collisions"], "global collision holds changed")
    require(journal.read_json(OUTPUT / "CENSUS_NORMALIZATION.json") == dict(**projection, census=audit["census"]), "normalized census does not reproduce")
    return VerifiedPaperIdentities(payload)


class SourceProof:
    def __init__(self, terms, papers):
        self.nodes = {}; self.refs = {}; self.seen_nodes = set(); self.seen_refs = set()
        for term in terms.entries.values():
            for field, h in (("target_id","target_sha256"), ("atom_id","atom_sha256"), ("parent_id","parent_sha256")):
                require(self.nodes.get(term[field], term[h]) == term[h], "conflicting entity witnesses")
                self.nodes[term[field]] = term[h]
            require(self.refs.get(term["mapping_ref"], term["mapping_sha256"]) == term["mapping_sha256"], "conflicting mapping witnesses")
            self.refs[term["mapping_ref"]] = term["mapping_sha256"]
        self.digests = {kind:hashlib.sha256() for kind in ("nodes","edges")}
        self.papers = papers; self.claim_status = Counter(); self.key_changes = 0

    def records(self, records):
        for kind, key, row in records:
            if kind == "node":
                if key in self.nodes:
                    require(digest(row) == self.nodes[key], "entity proof node differs"); self.seen_nodes.add(key)
                if key.startswith("CLM:"):
                    answer = self.papers.resolve(row["metadata"])
                    self.claim_status[answer["status"]] += 1
                    self.key_changes += answer["status"] == "verified" and answer["paper_key"] != strongest_paper_key(row["metadata"])
            if kind == "edge":
                ref = (row.get("metadata") or {}).get("audit_ref")
                if ref in self.refs:
                    require(digest(row) == self.refs[ref], "entity mapping witness differs"); self.seen_refs.add(ref)
            if kind != "metadata": self.digests[kind+"s"].update(compact(row).encode("utf-8") + b"\n")
            yield kind, key, row

    def verify(self, result, audit):
        require(self.seen_nodes == set(self.nodes) and self.seen_refs == set(self.refs), "entity identity witnesses incomplete")
        require({k:v.hexdigest() for k,v in self.digests.items()} == result["record_digests"], "source node/edge records changed")
        require(dict(self.claim_status) == audit["claim_status"] and self.key_changes == audit["verified_key_changes"], "full claim audit differs from census")


def build():
    require(not TEMP.exists() and not CATALOG.exists() and not (OUTPUT / "BUILD_STATE.json").exists(), "build exists; inspect/resume")
    baseline = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    audit = journal.read_json(OUTPUT / "PAPER_IDENTITY_AUDIT.json")
    require(baseline["status"] == "COMPLETED" and baseline["active_process"] is None, "writer active")
    require(baseline["current_graph"] == audit["graph"] and Path(audit["graph"]["path"]).resolve() == SOURCE.resolve(), "source advanced/wrong target")
    require(baseline["rollback_retention"] is False, "retention boundary changed")
    require(shutil.disk_usage(SOURCE).free > baseline["current_graph"]["bytes"] + 12*1024**3, "insufficient atomic replacement headroom")
    tests = OUTPUT / "TEST_RESULTS.xml"; counts = Counter()
    for suite in ElementTree.parse(tests).getroot().iter("testsuite"):
        counts.update({k:int(suite.get(k,0)) for k in ("tests","failures","errors","skipped")})
    require(counts["tests"] >= 290 and not any(counts[k] for k in ("failures","errors","skipped")), "complete passing regression suite required")
    journal.guards([baseline["current_graph"], baseline["current_detail_store"], baseline["formal_sources"]])
    for field in ("current_acceptance", "current_entity_terms", "current_coverage", "current_issues", "current_structure_holds", "current_shared_relations"):
        require(journal.fingerprint(Path(baseline[field]["path"])) == baseline[field], "current evidence artifact changed: " + field)
    guards = [native_info(fp["path"]) for fp in [baseline["current_graph"], baseline["current_detail_store"], *baseline["formal_sources"].values()]]
    papers = verify_authorities(audit); terms = VerifiedEntityTerms(journal.read_json(baseline["current_entity_terms"]["path"]))
    code = code_bindings(); audit_fp = journal.fingerprint(OUTPUT / "PAPER_IDENTITY_AUDIT.json")
    proof = SourceProof(terms, papers)
    campaign = dict(baseline, status="MANUAL_ACTIVE", phase="R31论文身份核验与证据计数整批应用", updated_at=journal.utc_now(),
        active_process=dict(kind="paper_identity", pid=os.getpid(), state=str(OUTPUT / "RUN_STATE.json")))
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json")
    night.update(active_process=campaign["active_process"]); journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    progress("BUILD_AUTHORITY_BACKED_PAPER_INDEX")
    with TEMP.open("xb", buffering=1024**2) as handle:
        with hashed_reader(SOURCE) as (reader, sha):
            result = stream_patch(proof.records(walk_graph(reader)), handle, [], {}, CATALOG, progress,
                relation_key_func=terms.relation_key, relation_version="kg.relation_evidence.v3", papers=papers,
                extra_graph_metadata=dict(paper_identity=dict(version=VERSION,
                    registry=os.path.relpath(PAPERS, SOURCE.parent).replace("\\", "/"), sha256=audit["registry"]["sha256"])))
            require(sha.hexdigest() == baseline["current_graph"]["sha256"], "full current source SHA mismatch")
        handle.flush(); os.fsync(handle.fileno())
    proof.verify(result, audit)
    require(result["counts"] == baseline["counts"] and not result["changed"], "node/edge mutation forbidden")
    for k in ("all_fine_grained_relation_groups", "all_claims", "shared_groups", "indexed_claims"):
        require(result["catalog_stats"][k] == baseline["relation_evidence_counts"][k], "relation or member grouping changed")
    require(result["catalog_stats"]["verified_multi_paper_shared_groups"] == audit["shared_group_counts"]["verified_multi_paper_groups"], "verified group count differs")
    require(result["catalog_stats"]["fully_verified_shared_groups"] == audit["shared_group_counts"]["fully_verified_groups"], "verified coverage differs")
    guard_native(guards); require(code_bindings() == code and journal.fingerprint(OUTPUT / "PAPER_IDENTITY_AUDIT.json") == audit_fp, "frozen code/audit advanced")
    state = dict(status="BUILT_NOT_ADOPTED", at=journal.utc_now(), baseline=baseline, protected_native=guards,
        temporary=cheap(TEMP), result=result, code=code, audit=audit_fp, catalog=journal.fingerprint(CATALOG),
        papers=audit["registry"], terms=baseline["current_entity_terms"], tests=journal.fingerprint(tests), test_counts=dict(counts),
        source_full_sha_verified=True, every_source_node_edge_identical=True, graph_backups=0, record_preimages_saved=False)
    journal.atomic_json(OUTPUT / "BUILD_STATE.json", state)
    progress("BUILT_NOT_ADOPTED", relations=result["catalog_stats"])


def validate():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json"); audit = journal.read_json(OUTPUT / "PAPER_IDENTITY_AUDIT.json")
    require(not (OUTPUT / "VALIDATED.json").exists(), "already validated")
    require(code_bindings() == state["code"] and cheap(TEMP) == state["temporary"], "build/code changed")
    require(journal.fingerprint(OUTPUT / "PAPER_IDENTITY_AUDIT.json") == state["audit"], "audit changed")
    guard_native(state["protected_native"])
    for fp in (state["catalog"], state["terms"], state["tests"]): require(journal.fingerprint(Path(fp["path"])) == fp, "frozen artifact changed")
    papers = verify_authorities(audit); terms = VerifiedEntityTerms(journal.read_json(state["terms"]["path"]))
    proof = SourceProof(terms, papers)
    with hashed_reader(TEMP) as (reader, sha):
        checks = verify_catalog_and_graph(proof.records(walk_graph(reader)), state["result"], CATALOG, progress,
            relation_key_func=terms.relation_key, papers=papers)
        candidate_sha = sha.hexdigest()
    proof.verify(state["result"], audit)
    progress("VERIFY_UNCHANGED_DETAIL_STORE_SHA")
    detail = state["baseline"]["current_detail_store"]
    require(sha256(Path(detail["path"])) == detail["sha256"], "detail store full SHA changed")
    checks.pop("records_match_approved_identity_only_transform", None)
    checks.update(all_node_and_edge_records_identical_to_source=True, verified_identity_proofs_complete=True,
        paper_identity_witnesses_validated=True, complete_title_and_identifier_conflicts_held=True,
        no_claim_or_edge_metadata_fields_added=True, metadata_coverage_unchanged_by_record_identity=True,
        authority_verified_claims=proof.claim_status["verified"], verified_paper_key_changes=proof.key_changes,
        distinct_articles_not_independent_cohorts=True)
    guard_native(state["protected_native"]); require(code_bindings() == state["code"], "code changed during validation")
    accepted = dict(status="VALIDATED_NOT_ADOPTED", at=journal.utc_now(), graph={**cheap(TEMP), "sha256":candidate_sha},
        temporary_native=native_info(TEMP), checks=checks, detail_sha_verified=True,
        build_state=journal.fingerprint(OUTPUT / "BUILD_STATE.json"), catalog=state["catalog"], papers=state["papers"], terms=state["terms"])
    journal.atomic_json(OUTPUT / "VALIDATED.json", accepted); progress("VALIDATED_NOT_ADOPTED", checks=checks)


def apply():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json"); accepted = journal.read_json(OUTPUT / "VALIDATED.json")
    require(not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "already applied")
    require(journal.fingerprint(OUTPUT / "BUILD_STATE.json") == accepted["build_state"] and code_bindings() == state["code"], "build/code advanced")
    guard_native([*state["protected_native"], accepted["temporary_native"]])
    for fp in (state["catalog"], state["papers"], state["terms"], state["audit"], state["tests"]):
        require(journal.fingerprint(Path(fp["path"])) == fp, "frozen acceptance artifact changed")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["current_graph"] == state["baseline"]["current_graph"] and campaign["active_process"]["kind"] == "paper_identity", "current source advanced")
    require(TEMP.resolve().parent == SOURCE.resolve().parent and SOURCE.resolve().is_relative_to(journal.OUTPUT.resolve()), "unsafe replacement target")
    os.replace(TEMP, SOURCE)
    graph = {**cheap(SOURCE), "sha256":accepted["graph"]["sha256"]}
    require(graph["bytes"] == accepted["graph"]["bytes"] and graph["mtime_ns"] == accepted["graph"]["mtime_ns"], "atomic replacement differs")
    baseline = state["baseline"]; result = state["result"]
    # Only graph-level binding changes. Held records/edge ordinals are identical.
    for field, filename in (("current_issues","CURRENT_REMAINING_ISSUES.jsonl"), ("current_structure_holds","CURRENT_STRUCTURE_HOLDS.jsonl")):
        write_rows(OUTPUT / filename, [dict(row, current_graph=graph, version_binding_status="FULL_CURRENT_GRAPH_VALIDATED") for row in rows(baseline[field]["path"])])
    audit = journal.read_json(OUTPUT / "PAPER_IDENTITY_AUDIT.json")
    receipt = dict(status="CURRENT_PAPER_IDENTITIES_APPLIED", at=journal.utc_now(), graph=graph, counts=result["counts"],
        changed=dict(claim_nodes=0, edge_records=0, verified_paper_keys=audit["verified_key_changes"]),
        connected_components=result["connected_components"], isolated_nodes=result["isolated_nodes"], self_loops=result["self_loops"],
        metadata_key_unions=result["metadata_key_unions"], metadata=result["metadata"], source_graph=baseline["current_graph"], source_kg_retained=False,
        shared_relations=state["catalog"], entity_terms=state["terms"], paper_identities=state["papers"], paper_issues=audit["paper_issues"],
        metadata_coverage=baseline["current_coverage"], relation_counts=result["catalog_stats"], checks=accepted["checks"],
        tests=state["tests"], test_counts=state["test_counts"], code=state["code"], detail_store=baseline["current_detail_store"],
        identity_audit=state["audit"], rollback_retention=False, record_preimages_saved=False, formal_apply_performed=False)
    journal.atomic_json(OUTPUT / "CURRENT_ACCEPTANCE.json", receipt)
    journal.atomic_json(OUTPUT / "CURRENT_RUNTIME_ACCEPTANCE.json", dict(status="CURRENT_RUNTIME_VALIDATED", at=receipt["at"],
        graph=graph, acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), code=state["code"], tests=state["tests"], test_counts=state["test_counts"]))
    campaign.update(status="COMPLETED", phase="R31论文编号统一、已核实论文计数和保守去重已应用", active_process=None, updated_at=receipt["at"],
        current_graph=graph, current_acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"),
        current_runtime_acceptance=journal.fingerprint(OUTPUT / "CURRENT_RUNTIME_ACCEPTANCE.json"),
        current_issues=journal.fingerprint(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"),
        current_structure_holds=journal.fingerprint(OUTPUT / "CURRENT_STRUCTURE_HOLDS.jsonl"),
        current_shared_relations=state["catalog"], current_paper_identities=state["papers"], current_paper_issues=audit["paper_issues"],
        relation_evidence_counts=result["catalog_stats"], last_deep_verification=accepted["at"],
        last_current_graph_content_verification=dict(graph=graph, independent_full_scan=True),
        next_steps=["复核新发现的书目矛盾，区分真实错配和标题/年代表示差异；未核实并不等于错误。",
            "继续复合表达、FANCE/VCP和其余语义/结构问题，不按相似名称强并。",
            "已核实不同论文数不表示独立样本或科学共识；正式full_v2不动，不运行模型/训练。"])
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    night = journal.read_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json")
    night.update(active_process=None, latest_acceptance=campaign["current_acceptance"], latest_completed_batch="R31")
    journal.atomic_json(journal.OUTPUT / "NIGHT_WINDOW_20260909.json", night)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", dict(status="COMPLETED", at=receipt["at"], graph=graph, relations=result["catalog_stats"]))
    print(compact(dict(status=receipt["status"], changes=receipt["changed"], relations=result["catalog_stats"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("phase", choices=("build","validate","apply","run"))
    args = parser.parse_args()
    for phase in ("build","validate","apply") if args.phase == "run" else (args.phase,): globals()[phase]()
