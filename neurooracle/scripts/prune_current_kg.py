"""Delete an explicitly approved claim set; atomic replacement, no rollback copy.

The temporary graph is not adopted until an independent complete read passes.
Only IDs/counts/hashes are logged, never new deleted-record preimages.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import Components, cheap, compact, hashed_reader, walk_graph
from inspect_kg_claim_deletion import exact_references
from kg_accepted_candidate_lineage import Bindings, require
from prepare_kg_simplified_reading import project_coverage
from reclaim_kg_backup_storage import native_info, sha256
from record_kg_source_findings import record_digest
from retire_kg_duplicate_references import same_record

R23 = journal.OUTPUT / "round23_source_deletion_candidate"
R24 = journal.OUTPUT / "round24_structure_metadata_simplification"
OUTPUT = journal.OUTPUT / "round26_confirmed_deletion"
SOURCE = R23 / "knowledge_graph.candidate.json"
TEMP = R23 / "knowledge_graph.candidate.json.rewrite.tmp"
SOURCE_SHA = "5c9e849af7e534167fff4d87cbedbf39fc6bb4882c21b9c487d9d0604c32ee43"
R23_SHA = "04c78fed8ba6173ca30f6dd47ad1874efca44330d7c858915b5b31554cebb01d"
R24_SHA = "45cbf84c4418f928580cd3180ca48cadc648ca1a7d17b833316db1ef92810a68"
HEADER_BYTES = 65536


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, values):
    journal.atomic_text(path, "".join(compact(v) + "\n" for v in values))


def select(issues, node_records, edge_records, expected=40):
    require(len(issues) == len({r["claim_id"] for r in issues}), "duplicate issue ID")
    require(all(r["classification"] in {"confirmed", "additional_review"} for r in issues), "unknown issue class")
    chosen = {r["claim_id"]: r for r in issues if r["classification"] == "confirmed"}
    require(len(chosen) == expected and all(k.startswith("CLM:") for k in chosen), "approved deletion count/IDs differ")
    nodes = {r["id"]: r for r in node_records}
    require(len(nodes) == len(node_records), "duplicate witness node")
    wanted = set()
    for cid, issue in chosen.items():
        require(cid in nodes and record_digest(nodes[cid]) == issue["current_node_sha256"], "selected claim witness changed")
        for key in ("related_edge_ordinals", "owned_science_ordinals", "about_ordinals"):
            require(all(type(o) is int and o > 0 for o in issue[key]), "invalid edge ordinal")
            require(len(issue[key]) == len(set(issue[key])), "duplicate related ordinal")
        require(set(issue["related_edge_ordinals"]) == set(issue["owned_science_ordinals"]) | set(issue["about_ordinals"]), "incomplete closure")
        wanted.update(issue["related_edge_ordinals"])
    edges = {r["candidate_edge_ordinal"]: r["edge"] for r in edge_records}
    require(len(edges) == len(edge_records) and wanted <= set(edges), "edge witness missing/duplicate")
    selected_edges = {o: edges[o] for o in sorted(wanted)}
    for ordinal, edge in edges.items():
        require(touches(edge, chosen) == (ordinal in wanted), "witness ownership closure mismatch")
    return {k: nodes[k] for k in chosen}, selected_edges


def touches(edge, ids):
    return bool({edge["source_id"], edge["target_id"], (edge.get("metadata") or {}).get("claim_id")} & set(ids))


def input_data():
    b = Bindings()
    accepted = journal.read_json(b.pin(R23 / "REPAIR_ACCEPTANCE.json", R23_SHA))
    plan = b.json(accepted["plan"])
    prior = journal.read_json(b.pin(R24 / "REVIEW_COMPLETE.json", R24_SHA))
    prior_build = b.json(prior["build"])
    issues = b.rows(accepted["artifacts"]["CURRENT_REMAINING_ISSUES.jsonl"])
    witnesses = b.rows(accepted["artifacts"]["CURRENT_PROTECTED_NODES.jsonl"])
    edge_witnesses = b.rows(accepted["artifacts"]["CURRENT_PROTECTED_EDGES.jsonl"])
    removed_nodes, removed_edges = select(issues, witnesses, edge_witnesses)
    require(len(removed_edges) == 120 and Counter(e["relation_type"] == "about" for e in removed_edges.values()) == {True: 80, False: 40}, "approved 40/120 scope changed")
    coverage = b.rows(prior_build["artifacts"]["CURRENT_METADATA_COVERAGE.jsonl"])
    return dict(accepted=accepted, metadata=plan["metadata_after"], issues=issues, witnesses=witnesses,
                edge_witnesses=edge_witnesses, removed_nodes=removed_nodes, removed_edges=removed_edges,
                coverage=coverage, denominators=prior["current_denominators"])


class Writer:
    """Reserve metadata space so accurate topology is written without a third pass."""
    def __init__(self, handle, header_bytes=HEADER_BYTES):
        self.handle, self.header_bytes = handle, header_bytes
        self.stage = "nodes"
        self.nodes = self.edges = 0
        handle.write(b" " * header_bytes)

    def node(self, key, payload):
        require(self.stage == "nodes", "node after edge")
        self.handle.write((b"," if self.nodes else b"") + compact(key).encode() + b":" + payload)
        self.nodes += 1

    def edge(self, payload):
        if self.stage == "nodes":
            self.handle.write(b'},"edges":[')
            self.stage = "edges"
        require(self.stage == "edges", "writer closed")
        self.handle.write((b"," if self.edges else b"") + payload)
        self.edges += 1

    def finish(self, metadata):
        require(self.stage != "closed", "writer already finished")
        if self.stage == "nodes":
            self.handle.write(b'},"edges":[')
        self.handle.write(b"]}\n")
        end = self.handle.tell()
        prefix = b'{"metadata":' + compact(metadata).encode("utf-8")
        suffix = b',"concepts":{'
        padding = self.header_bytes - len(prefix) - len(suffix)
        require(padding >= 0, "metadata exceeds reserved prefix")
        self.handle.seek(0)
        self.handle.write(prefix + b" " * padding + suffix)
        self.handle.seek(end)
        self.stage = "closed"


def no_references(record, payload, ids, needle):
    if needle is not None and needle.search(payload.decode("utf-8")):
        require(not list(exact_references(record, ids)), "retained exact reference to deleted claim")


def stream_delete(records, handle, deleted_nodes, deleted_edges, expected_metadata, progress=None):
    ids = set(deleted_nodes)
    needle = re.compile("|".join(re.escape(x) for x in sorted(ids))) if ids else None
    writer = Writer(handle)
    cc = Components()
    counts, before, domains, sources, relations = Counter(), Counter(), Counter(), Counter(), Counter()
    digests = {k: hashlib.sha256() for k in ("nodes", "edges")}
    unions = {k: set() for k in digests}
    seen_nodes, seen_edges = set(), set()
    degrees = previously_touched = None
    metadata = None
    for kind, key, record in records:
        if kind == "metadata":
            require(metadata is None and not cc.ids and same_record(record, expected_metadata), "source metadata changed/order")
            metadata = deepcopy(record)
            no_references(record, compact(record).encode(), ids, needle)
            continue
        require(metadata is not None, "missing metadata")
        payload = compact(record).encode("utf-8")
        before[kind + "s"] += 1
        if kind == "node":
            require(degrees is None and key == record["id"], "source node order/ID")
            before["claims"] += key.startswith("CLM:")
            if key in ids:
                require(key not in seen_nodes and same_record(record, deleted_nodes[key]), "selected node duplicate/changed")
                seen_nodes.add(key)
                continue
            cc.add(key)
            counts["claims"] += key.startswith("CLM:")
            domains.update(record.get("domain_tags") or [])
            sources[record.get("source_vocab") or ""] += 1
            writer.node(key, payload)
        else:
            require(kind == "edge" and int(key) == before["edges"], "source edge ordinal/order")
            if degrees is None:
                degrees, previously_touched = bytearray(len(cc.ids)), bytearray(len(cc.ids))
            sid, tid = record["source_id"], record["target_id"]
            for endpoint in (sid, tid):
                require(endpoint in ids or endpoint in cc.ids, "source dangling endpoint")
                if endpoint in cc.ids:
                    previously_touched[cc.ids[endpoint]] = 1
            ordinal = int(key)
            require(touches(record, ids) == (ordinal in deleted_edges), "unexpected incident/owned edge")
            if ordinal in deleted_edges:
                require(same_record(record, deleted_edges[ordinal]), "selected edge changed")
                seen_edges.add(ordinal)
                continue
            owner = (record.get("metadata") or {}).get("claim_id")
            require(owner is None or owner in cc.ids, "retained edge owner missing")
            cc.union(sid, tid)
            degrees[cc.ids[sid]] = degrees[cc.ids[tid]] = 1
            counts["self_loops"] += sid == tid
            relations[record["relation_type"]] += 1
            writer.edge(payload)
        no_references(record, payload, ids, needle)
        group = kind + "s"
        counts[group] += 1
        digests[group].update(payload + b"\n")
        unions[group].update(record.get("metadata") or {})
        if progress and before[group] % 250000 == 0:
            progress("BUILD_" + group.upper(), **dict(before))
    require(seen_nodes == ids and seen_edges == set(deleted_edges), "missing deletion target")
    require(metadata is not None, "empty graph")
    if degrees is None:
        degrees, previously_touched = bytearray(len(cc.ids)), bytearray(len(cc.ids))
    metadata["stats"] = dict(n_concepts=counts["nodes"], n_edges=counts["edges"], domains=dict(domains),
                             sources=dict(sources), relations=dict(relations), connected_components=cc.count)
    # Historical patch preimages are not a runtime dependency. The user has
    # explicitly discontinued rollback retention; do not leave a dangling link.
    metadata.get("umls_node_metadata_alias_repair", {}).pop("reversible_node_patch_log", None)
    writer.finish(metadata)
    return dict(counts={k: counts[k] for k in ("nodes", "claims", "edges")}, source_counts=dict(before),
                metadata=metadata, record_digests={k: v.hexdigest() for k, v in digests.items()},
                metadata_key_unions={k: sorted(v) for k, v in unions.items()},
                connected_components=cc.count, isolated_nodes=degrees.count(0), self_loops=counts["self_loops"],
                newly_isolated_nodes=sum(bool(previously_touched[i]) and not degrees[i] for i in range(len(degrees))),
                deleted_claim_ids=sorted(ids), deleted_edge_ordinals=sorted(deleted_edges),
                deleted_claims=len(ids), deleted_edges=len(deleted_edges), deletion_preimages_saved=False)


def verify_stream(records, expected, progress=None):
    """Independent retained-record, topology, identity and reference scanner."""
    cc = Components()
    counts, domains, sources, relations = Counter(), Counter(), Counter(), Counter()
    digests = {k: hashlib.sha256() for k in ("nodes", "edges")}
    unions = {k: set() for k in digests}
    ids = set(expected["deleted_claim_ids"])
    needle = re.compile("|".join(re.escape(x) for x in sorted(ids))) if ids else None
    metadata, degrees = None, None
    for kind, key, record in records:
        payload = compact(record).encode("utf-8")
        no_references(record, payload, ids, needle)
        if kind == "metadata":
            require(metadata is None and not cc.ids and same_record(record, expected["metadata"]), "output metadata differs")
            metadata = record
            continue
        require(metadata is not None, "output metadata missing")
        group = kind + "s"
        if kind == "node":
            require(degrees is None and key == record["id"] and key not in ids, "output node order/identity")
            cc.add(key)
            counts["claims"] += key.startswith("CLM:")
            domains.update(record.get("domain_tags") or [])
            sources[record.get("source_vocab") or ""] += 1
        else:
            require(kind == "edge" and int(key) == counts["edges"] + 1, "output edge ordinal gap")
            if degrees is None:
                degrees = bytearray(len(cc.ids))
            sid, tid = record["source_id"], record["target_id"]
            cc.union(sid, tid)
            degrees[cc.ids[sid]] = degrees[cc.ids[tid]] = 1
            owner = (record.get("metadata") or {}).get("claim_id")
            require(owner is None or owner in cc.ids, "output owner missing")
            counts["self_loops"] += sid == tid
            relations[record["relation_type"]] += 1
        counts[group] += 1
        digests[group].update(payload + b"\n")
        unions[group].update(record.get("metadata") or {})
        if progress and counts[group] % 250000 == 0:
            progress("VERIFY_" + group.upper(), **dict(counts))
    require(metadata is not None, "empty output")
    require({k: counts[k] for k in expected["counts"]} == expected["counts"], "output counts differ")
    require({k: d.hexdigest() for k, d in digests.items()} == expected["record_digests"], "retained record/order/type changed")
    require({k: sorted(v) for k, v in unions.items()} == expected["metadata_key_unions"], "metadata unions differ")
    require(cc.count == expected["connected_components"] and counts["self_loops"] == expected["self_loops"], "topology differs")
    isolates = (degrees if degrees is not None else bytearray(len(cc.ids))).count(0)
    require(isolates == expected["isolated_nodes"], "isolate count differs")
    require(metadata["stats"] == dict(n_concepts=counts["nodes"], n_edges=counts["edges"], domains=dict(domains),
                                     sources=dict(sources), relations=dict(relations), connected_components=cc.count), "metadata stats inaccurate")
    return dict(dangling_endpoints=0, duplicate_node_ids=0, missing_edge_owners=0,
                references_to_deleted_ids=0, retained_records_identical=True,
                independent_full_structure_scan=True, counts=expected["counts"])


def progress(phase, **details):
    value = dict(status="RUNNING", phase=phase, updated_at=journal.utc_now(), **details)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", value)
    print(compact(value), flush=True)


def guard_native(entries):
    for entry in entries:
        require(native_info(entry["path"]) == entry, "file changed after trusted boundary: " + entry["path"])


def build():
    require(not TEMP.exists() and not (OUTPUT / "BUILD_STATE.json").exists(), "rewrite already started")
    data = input_data()
    test_counts = Counter()
    for suite in ElementTree.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"):
        test_counts.update({k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
    require(test_counts["tests"] >= 20 and not any(test_counts[k] for k in ("failures", "errors", "skipped")), "passing deletion tests required")
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["active_process"] is None and campaign["current_acceptance"]["sha256"] == R23_SHA, "current state advanced/worker active")
    original = data["accepted"]["artifacts"]["knowledge_graph.candidate.json"]
    require(original["sha256"] == SOURCE_SHA and Path(original["path"]) == SOURCE, "unexpected current graph")
    journal.guards(original)
    journal.guards(campaign["formal_sources"])
    protected = [native_info(fp["path"]) for fp in campaign["formal_sources"].values()]
    source_native = native_info(SOURCE)
    require(shutil.disk_usage(SOURCE).free - original["bytes"] > 16 * 1024**3, "insufficient temporary space")
    OUTPUT.mkdir(exist_ok=True)
    progress("STARTING_SINGLE_VERSION_REWRITE")
    with TEMP.open("xb", buffering=1024**2) as handle:
        with hashed_reader(SOURCE) as (reader, digest):
            result = stream_delete(walk_graph(reader), handle, data["removed_nodes"], data["removed_edges"], data["metadata"], progress)
            require(digest.hexdigest() == SOURCE_SHA, "source full SHA mismatch")
        handle.flush()
        os.fsync(handle.fileno())
    guard_native([source_native, *protected])
    require(result["source_counts"] == data["accepted"]["counts"], "source counts differ")
    require(result["counts"] == dict(nodes=2587871, claims=905184, edges=3012015), "approved count delta differs")
    delta = [{"kind": "node", "scope": "node/claim", "record": r} for r in data["removed_nodes"].values()]
    delta += [{"kind": "edge", "scope": "edge/all", "record": r} for r in data["removed_edges"].values()]
    coverage, den, _ = project_coverage(data["coverage"], data["denominators"], delta, result["counts"])
    write_rows(OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl", coverage)
    for issue in data["issues"]:
        if issue["classification"] == "additional_review":
            for key in ("related_edge_ordinals", "owned_science_ordinals", "about_ordinals"):
                require(not set(issue[key]) & set(result["deleted_edge_ordinals"]), "held issue edge selected")
                issue[key] = [o - bisect_left(result["deleted_edge_ordinals"], o) for o in issue[key]]
            issue.pop("current_graph", None)
            issue["version_binding_status"] = "CURRENT_GRAPH_PENDING_VALIDATION"
    write_rows(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl", [r for r in data["issues"] if r["classification"] == "additional_review"])
    state = dict(status="BUILT_NOT_ADOPTED", at=journal.utc_now(), source=original, source_native=source_native,
                 protected_formal_native=protected, temporary=cheap(TEMP), result=result, metadata_denominators=den,
                 source_sha_verified=True, rollback_retention=False, code=journal.fingerprint(Path(__file__)))
    journal.atomic_json(OUTPUT / "BUILD_STATE.json", state)
    progress("BUILT_NOT_ADOPTED", counts=result["counts"])


def validate():
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    require(not (OUTPUT / "VALIDATED.json").exists(), "already validated")
    require(journal.fingerprint(Path(__file__)) == state["code"], "implementation changed during rewrite")
    guard_native([state["source_native"], *state["protected_formal_native"]])
    require(cheap(TEMP) == state["temporary"], "temporary file changed")
    temp_native = native_info(TEMP)
    with hashed_reader(TEMP) as (reader, digest):
        checks = verify_stream(walk_graph(reader), state["result"], progress)
        candidate_sha = digest.hexdigest()
    guard_native([temp_native, state["source_native"], *state["protected_formal_native"]])
    # Only the current detail store is needed; no copies or historical patch logs.
    detail = R23 / "umls_details.sqlite"
    before = native_info(detail)
    detail_sha = sha256(detail)
    expected_detail = input_data()["accepted"]["artifacts"]["umls_details.sqlite"]
    require(detail_sha == expected_detail["sha256"], "detail store SHA differs")
    guard_native([before])
    accepted = dict(status="VALIDATED_PENDING_ATOMIC_REPLACEMENT", at=journal.utc_now(),
                    graph={**cheap(TEMP), "sha256": candidate_sha}, temporary_native=temp_native,
                    detail_store={**cheap(detail), "sha256": detail_sha}, detail_native=before,
                    checks=checks, build_state=journal.fingerprint(OUTPUT / "BUILD_STATE.json"),
                    tests=journal.fingerprint(OUTPUT / "TEST_RESULTS.xml"), rollback_retention=False)
    journal.atomic_json(OUTPUT / "VALIDATED.json", accepted)
    progress("VALIDATED_PENDING_ATOMIC_REPLACEMENT", **checks)


def apply():
    accepted = journal.read_json(OUTPUT / "VALIDATED.json")
    require(not (OUTPUT / "CURRENT_ACCEPTANCE.json").exists(), "already replaced")
    require(journal.fingerprint(OUTPUT / "BUILD_STATE.json") == accepted["build_state"], "build state changed")
    state = journal.read_json(OUTPUT / "BUILD_STATE.json")
    require(journal.fingerprint(Path(__file__)) == state["code"], "implementation changed")
    guard_native([state["source_native"], accepted["temporary_native"], accepted["detail_native"], *state["protected_formal_native"]])
    require(SOURCE.resolve().parent == R23.resolve() and TEMP.resolve().parent == R23.resolve(), "replacement path escapes exact target")
    os.replace(TEMP, SOURCE)  # Explicit user authorization: no backup of old graph.
    current_graph = {**cheap(SOURCE), "sha256": accepted["graph"]["sha256"]}
    require(native_info(SOURCE)["file_id"] == accepted["temporary_native"]["file_id"], "replacement identity differs")
    issues = rows(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl")
    for issue in issues:
        issue["current_graph"] = current_graph
        issue["version_binding_status"] = "FULL_CURRENT_GRAPH_VALIDATED"
    write_rows(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl", issues)
    result = state["result"]
    receipt = dict(status="CURRENT_KG_40_CONFIRMED_DELETED_NO_ROLLBACK", at=journal.utc_now(),
                   graph=current_graph, detail_store=accepted["detail_store"], counts=result["counts"],
                   node_metadata_field_union=len(result["metadata_key_unions"]["nodes"]),
                   edge_metadata_field_union=len(result["metadata_key_unions"]["edges"]),
                   connected_components=result["connected_components"], isolated_nodes=result["isolated_nodes"],
                   newly_isolated_nodes=result["newly_isolated_nodes"], self_loops=result["self_loops"],
                   deleted_claim_ids=result["deleted_claim_ids"], deleted_claims=40, deleted_edges=120,
                   known_confirmed_remaining=0, additional_review=21, exhaustive_error_census=False,
                   rollback_retention=False, deleted_record_preimages_saved=False, formal_apply_performed=False,
                   validation=accepted["checks"], metadata_denominators=state["metadata_denominators"],
                   current_issues=journal.fingerprint(OUTPUT / "CURRENT_REMAINING_ISSUES.jsonl"),
                   current_coverage=journal.fingerprint(OUTPUT / "CURRENT_METADATA_COVERAGE.jsonl"))
    journal.atomic_json(OUTPUT / "CURRENT_ACCEPTANCE.json", receipt)
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    campaign.update(status="MANUAL_ACTIVE", phase="40条确认问题已删除；整理单版本保留策略与历史副本", active_process=None,
                    current_acceptance=journal.fingerprint(OUTPUT / "CURRENT_ACCEPTANCE.json"), counts=result["counts"],
                    confirmed_source_or_expression_issue_claims=0, confirmed_issue_claims_removed_from_current_candidate=90,
                    additional_semantic_or_detail_review_candidates=21, last_deep_verification=accepted["at"],
                    rollback_retention=False, current_graph_path=str(SOURCE),
                    retention_policy="User requests current KG only: atomic temporary replacement, no per-step backups or new preimages; remove superseded KG copies after dependency checks.")
    campaign["materialization_hold"] = "此前冻结及逐步回退保留规则已被本次用户单版本要求取代；正式应用另按用户回复决定。"
    campaign["boundaries"] = ["仅优化KG，不涉及模型或训练", "只保留当前工作KG，不新增回退副本/删除前像", "删除仅40条确认声明及120关联边；21复审与其他候选不擅自删除", "正式full_v2是否同步依用户明确回复；不提交Git或上传"]
    journal.atomic_json(journal.OUTPUT / "CAMPAIGN.json", campaign)
    progress("40_CLAIMS_120_EDGES_REPLACED_NO_ROLLBACK", graph=current_graph, counts=result["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "apply"))
    {"build": build, "validate": validate, "apply": apply}[parser.parse_args().command]()
