"""Retire only reviewed stale edges with an exact, independently retained witness.

All writes go to a new candidate directory. Nodes, claims, source graphs and
runtime code remain unchanged. Original edge ordinals belong to the source
candidate/audit index; candidate ordinals are explicitly translated in logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_kg_claim_compatibility import AUDIT, OUTPUT as SOURCE, fingerprint, guard
from build_umls_simplification_candidate import (
    REPO, UMLS_ROOT, Components, atomic_json, cheap, compact, digest_update,
    file_sha, hashed_reader, progress, read_json, utc_now, walk_graph,
)
from audit_kg_integrity import JsonlWriter, claim_summary, readonly_db
from analyze_kg_integrity import NodeLookup
from plan_kg_reference_repairs import propose_reference_patch
from reconcile_kg_reference_checks import about_edge_checks
from neurooracle.src.kg_quality_checks import edge_claim_agreement

OUTPUT = UMLS_ROOT / "reference_duplicate_retirement_v2_20260907"
CATEGORY = "exact_existing_record_after_redirect"
EXPECTED_RETIREMENTS = 92
NEW_CODE = (
    "neurooracle/scripts/retire_kg_duplicate_references.py",
    "neurooracle/scripts/validate_kg_reference_retirement.py",
    "neurooracle/tests/test_kg_reference_retirement.py",
    "neurooracle/scripts/build_umls_simplification_candidate.py",
    "neurooracle/scripts/streaming_graph_json.py",
)
AUDIT_FILES = (
    "audit_index.sqlite", "STRUCTURE_VALIDATED.json", "ISOLATED_NODES.jsonl",
    "REFERENCE_ISSUES.jsonl", "ABOUT_CLAIM_POINTER_ISSUES.jsonl",
    "EDGE_CLAIM_DISAGREEMENTS_CORRECTED.jsonl",
)


def same_record(left, right):
    # JSON types matter: Python's False == 0 must not certify equal evidence.
    def encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return encoded(left) == encoded(right)


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def fingerprints(value):
    if isinstance(value, dict):
        if {"path", "bytes", "mtime_ns", "sha256"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from fingerprints(child)
    elif isinstance(value, list):
        for child in value:
            yield from fingerprints(child)


def guards(value):
    for fp in fingerprints(value):
        guard(fp)


def deep_check(fp):
    guard(fp)
    if file_sha(Path(fp["path"])) != fp["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {fp['path']}")
    guard(fp)


def candidate_ordinal(original, retired):
    position = bisect_left(retired, original)
    if position < len(retired) and retired[position] == original:
        raise ValueError("retired record has no candidate ordinal")
    return original - position


def select_retirements(details):
    selected = [row for row in details if row["category"] == CATEGORY]
    retired = {row["edge_ordinal"] for row in selected}
    if len(retired) != len(selected):
        raise ValueError("duplicate retirement ordinal")
    result = []
    for row in selected:
        proposal = row["proposal"]
        if same_record(proposal["before"], proposal["after"]):
            raise ValueError("retirement is not a stale endpoint correction")
        witnesses = [item for item in row["existing_destination_edges"]
                     if item["ordinal"] not in retired and same_record(item["edge"], proposal["after"])]
        if not witnesses:
            raise ValueError("no exact, independently retained witness")
        witness = min(witnesses, key=lambda item: item["ordinal"])
        result.append({
            "original_edge_ordinal": row["edge_ordinal"],
            "retained_original_edge_ordinal": witness["ordinal"],
            "retained_candidate_edge_ordinal": candidate_ordinal(witness["ordinal"], sorted(retired)),
            "retired_edge": proposal["before"], "retained_edge": witness["edge"],
            "proposal": proposal, "claim_id": proposal["claim_id"],
            "action": "retire_stale_derived_edge_keep_existing_exact_corrected_record",
            "node_merge": False, "claim_mutation": False,
        })
    return sorted(result, key=lambda row: row["original_edge_ordinal"])


def check_proposal(row, claim, concepts):
    proposal, reason = propose_reference_patch(row["retired_edge"], claim, concepts)
    if proposal is None or not same_record(proposal, row["proposal"]):
        raise ValueError(f"reference guard no longer passes: {row['original_edge_ordinal']}: {reason}")
    if not same_record(proposal["after"], row["retained_edge"]):
        raise ValueError("retained edge is not the exact corrected record")
    checker = about_edge_checks if row["retained_edge"]["relation_type"] == "about" else edge_claim_agreement
    if checker(row["retained_edge"], claim):
        raise ValueError("retained witness does not agree with the current claim")


def metadata_after(before, topology):
    after = deepcopy(before)
    after["stats"]["n_edges"] = topology["after"]["edges"]
    after["stats"]["relations"] = topology["after"]["relations"]
    after["stats"]["connected_components"] = topology["after"]["connected_components"]
    return after


def restore_edges(candidate_edges, retirements, original_count):
    """Reinsert exact old records, failing on truncated or surplus candidates."""
    retired = {row["original_edge_ordinal"]: row["retired_edge"] for row in retirements}
    if len(retired) != len(retirements) or any(not 1 <= ordinal <= original_count for ordinal in retired):
        raise ValueError("invalid rollback ordinals")
    stream = iter(candidate_edges)
    for ordinal in range(1, original_count + 1):
        if ordinal in retired:
            yield ordinal, retired[ordinal], True
        else:
            try:
                edge = next(stream)
            except StopIteration as error:
                raise ValueError("candidate edge stream is truncated") from error
            yield ordinal, edge, False
    sentinel = object()
    if next(stream, sentinel) is not sentinel:
        raise ValueError("candidate has surplus edge records")


def compute_topology(db, retirements, output):
    components = Components()
    for (node_id,) in db.execute("SELECT id FROM nodes ORDER BY id"):
        components.add(node_id)
    touched = bytearray(len(components.ids))
    retired = {row["original_edge_ordinal"]: row for row in retirements}
    before, after, seen_retired = Counter(), Counter(), set()
    before_relations, after_relations = Counter(), Counter()
    progress(output, "CHECKING_ALL_INDEXED_EDGE_TOPOLOGY", nodes=len(components.ids))
    cursor = db.execute("SELECT s,t,r,ordinal FROM edges INDEXED BY edges_pair ORDER BY s,t,r,ordinal")
    for (source, target), group in groupby(cursor, key=lambda row: (row[0], row[1])):
        records = list(group)
        kept = [row for row in records if row[3] not in retired]
        before["edges"] += len(records)
        after["edges"] += len(kept)
        before_relations.update(row[2] for row in records)
        after_relations.update(row[2] for row in kept)
        seen_retired.update(row[3] for row in records if row[3] in retired)
        for counts, edges in ((before, records), (after, kept)):
            if not edges:
                continue
            if source == target:
                counts["self_loop_records"] += len(edges)
            else:
                counts["unique_nonself_pairs"] += 1
            if source != target and len(edges) > 1:
                counts["parallel_pair_groups"] += 1
                relations = {row[2] for row in edges}
                category = "different_relations_same_pair"
                if len(relations) == 1:
                    hashes = {db.execute("SELECT record_sha256 FROM edges WHERE ordinal=?", (row[3],)).fetchone()[0]
                              for row in edges}
                    category = "exact_duplicate_edge_records" if len(hashes) == 1 else "same_relation_different_records"
                counts[category] += 1
                counts["extra_records_in_parallel_pairs"] += len(edges) - 1
                counts["extra_relations_in_parallel_pairs"] += len(relations) - 1
        if kept:
            components.union(source, target)
            touched[components.ids[source]] = touched[components.ids[target]] = 1
        if before["edges"] // 500000 > (before["edges"] - len(records)) // 500000:
            progress(output, "CHECKING_ALL_INDEXED_EDGE_TOPOLOGY", edges=before["edges"])
    if seen_retired != set(retired):
        raise ValueError("index lacks a retired ordinal")
    after["connected_components"], after["isolated_nodes"] = components.count, touched.count(0)
    affected = {row["retired_edge"][key] for row in retirements for key in ("source_id", "target_id")}
    newly_isolated = sorted(node_id for node_id in affected if not touched[components.ids[node_id]])
    for row in retirements:
        source, target = row["retired_edge"]["source_id"], row["retired_edge"]["target_id"]
        components.union(source, target)
        touched[components.ids[source]] = touched[components.ids[target]] = 1
    before["connected_components"], before["isolated_nodes"] = components.count, touched.count(0)
    return {"before": {**dict(before), "relations": dict(sorted(before_relations.items()))},
            "after": {**dict(after), "relations": dict(sorted(after_relations.items()))},
            "newly_isolated_node_ids": newly_isolated, "nodes_removed": 0}


def claim_about_state(claim, edges):
    targets = [edge["target_id"] for edge in edges if edge["relation_type"] == "about"]
    s, t = claim["subject_id"], claim["object_id"]
    return {"code": "claim_about_links_disagree_with_endpoints", "claim_id": claim["id"],
            "subject_id": s, "object_id": t, "about_count": len(targets),
            "subject_links": targets.count(s), "object_links": targets.count(t),
            "extra_links": sum(target not in {s, t} for target in targets)}


def has_about_issue(state):
    return not state["subject_links"] or not state["object_links"] or bool(state["extra_links"])


def update_reference_reports(db, retirements, output):
    retired = sorted(row["original_edge_ordinal"] for row in retirements)
    retired_set = set(retired)
    artifacts, counts = {}, {}
    for name in ("ABOUT_CLAIM_POINTER_ISSUES.jsonl", "EDGE_CLAIM_DISAGREEMENTS_CORRECTED.jsonl"):
        writer, removed = JsonlWriter(output / name), 0
        for row in jsonl(AUDIT / name):
            ordinal = row["edge_ordinal"]
            if ordinal in retired_set:
                removed += 1
                continue
            writer.add({**row, "source_edge_ordinal": ordinal, "edge_ordinal": candidate_ordinal(ordinal, retired)})
        artifacts[name] = writer.close()
        counts[name] = {"remaining": artifacts[name]["rows"], "resolved_edge_records": removed}
    states = {row["claim_id"]: row for row in jsonl(AUDIT / "REFERENCE_ISSUES.jsonl")}
    affected = JsonlWriter(output / "AFFECTED_CLAIM_REFERENCES.jsonl")
    resolved = 0
    for claim_id in sorted({row["claim_id"] for row in retirements}):
        claim = json.loads(db.execute("SELECT summary_json FROM claims WHERE id=?", (claim_id,)).fetchone()[0])
        edges = [(ordinal, json.loads(payload)) for ordinal, payload in db.execute(
            "SELECT ordinal,payload_json FROM edges WHERE s=? AND r='about' ORDER BY ordinal", (claim_id,))]
        before = claim_about_state(claim, [edge for _, edge in edges])
        after = claim_about_state(claim, [edge for ordinal, edge in edges if ordinal not in retired_set])
        if has_about_issue(before) != (claim_id in states) or (claim_id in states and before != states[claim_id]):
            raise ValueError("affected claim reference baseline differs from frozen audit")
        if after["subject_links"] != before["subject_links"] or after["object_links"] != before["object_links"]:
            raise ValueError("retirement removes a correct claim endpoint link")
        affected.add({"claim_id": claim_id, "before": before, "after": after})
        if has_about_issue(after):
            states[claim_id] = after
        elif claim_id in states:
            del states[claim_id]
            resolved += 1
    artifacts["AFFECTED_CLAIM_REFERENCES.jsonl"] = affected.close()
    writer = JsonlWriter(output / "REFERENCE_ISSUES.jsonl")
    shapes = Counter()
    for row in sorted(states.values(), key=lambda row: row["claim_id"]):
        writer.add(row)
        missing = not row["subject_links"] or not row["object_links"]
        shapes["missing_and_extra" if missing and row["extra_links"] else "missing_only" if missing else "extra_only"] += 1
    artifacts["REFERENCE_ISSUES.jsonl"] = writer.close()
    counts["claim_about_disagreements"] = {"remaining": len(states), "resolved_claims": resolved, "shapes": dict(shapes)}
    return {"counts": counts, "artifacts": artifacts,
            "ordinal_policy": "edge_ordinal is candidate position; source_edge_ordinal refers to immutable source/audit index"}


def planning_bindings(accepted, audit_inputs):
    return [*audit_inputs.values(), *accepted["implementation"].values(),
            *[accepted["artifacts"][name] for name in
              ("REFERENCE_COLLISION_DETAILS.jsonl", "BUILD_COMPLETE.json", "VALIDATION_COMPLETE.json")]]


def prepare(output, reuse_planning_from=None):
    output = output.resolve()
    if output.parent != UMLS_ROOT.resolve() or not output.name.startswith("reference_duplicate_retirement_") or output.exists():
        raise ValueError("requires a fresh versioned retirement directory directly inside the UMLS workspace")
    accepted = read_json(SOURCE / "REPAIR_ACCEPTANCE.json")
    if accepted["status"] != "CLAIM_COMPATIBILITY_REPAIRED_REFERENCE_REVIEW_PENDING_NOT_APPLIED":
        raise ValueError("unexpected source acceptance")
    guards(accepted)
    deep_check(accepted["source_audit"])
    audit = read_json(Path(accepted["source_audit"]["path"]))
    audit_inputs = {name: audit["artifacts"][name] for name in AUDIT_FILES}
    guards(audit_inputs)
    reusable, reused_manifest = {}, None
    if reuse_planning_from is not None:
        previous_path = reuse_planning_from.resolve(strict=True) / "INPUTS.json"
        previous = read_json(previous_path)
        if not previous.get("planning_inputs_deep_verified_at"):
            raise ValueError("previous attempt has no completed deep-verification evidence")
        # A failed control-statistics comparison does not invalidate unchanged
        # immutable data. Never reuse validation for edited implementation files.
        guards(previous["source_bindings"])
        guard(previous["source_acceptance"])
        guards(previous["audit_inputs"])
        reusable = {fp["path"]: fp for fp in planning_bindings(previous["source_bindings"], previous["audit_inputs"])}
        reused_manifest = fingerprint(previous_path)
    output.mkdir(exist_ok=False)
    progress(output, "VERIFYING_FROZEN_PLANNING_INPUTS")
    # Deep-verify the authoritative index once at this mutation boundary.
    # The source graph is deep-verified while building; companion copies while validating.
    verified = set()
    for fp in planning_bindings(accepted, audit_inputs):
        if fp["path"] not in verified:
            if reusable.get(fp["path"]) == fp:
                guard(fp)
            else:
                deep_check(fp)
            verified.add(fp["path"])
    inputs = {"prepared_at": utc_now(), "source_acceptance": fingerprint(SOURCE / "REPAIR_ACCEPTANCE.json"),
              "source_bindings": accepted, "audit_inputs": audit_inputs,
              "implementation": {name: fingerprint(REPO / name) for name in NEW_CODE},
              "planning_inputs_deep_verified_at": utc_now(),
              "reused_planning_verification_from": reused_manifest,
              "integrity_policy": "frozen hashes plus unchanged size/mtime for progress; full source/candidate/companion verification at candidate boundary",
              "formal_apply_authorized": False, "node_merges_authorized": False}
    atomic_json(output / "INPUTS.json", inputs)
    details = list(jsonl(SOURCE / "REFERENCE_COLLISION_DETAILS.jsonl"))
    retirements = select_retirements(details)
    if len(retirements) != EXPECTED_RETIREMENTS:
        raise ValueError("reviewed retirement scope is not exactly 92")
    db = readonly_db(AUDIT)
    lookup = NodeLookup(db)
    try:
        for row in retirements:
            claim = json.loads(db.execute("SELECT summary_json FROM claims WHERE id=?", (row["claim_id"],)).fetchone()[0])
            check_proposal(row, claim, lookup)
            for ordinal, expected in ((row["original_edge_ordinal"], row["retired_edge"]),
                                      (row["retained_original_edge_ordinal"], row["retained_edge"])):
                actual = json.loads(db.execute("SELECT payload_json FROM edges WHERE ordinal=?", (ordinal,)).fetchone()[0])
                if not same_record(actual, expected):
                    raise ValueError("reviewed edge no longer matches the immutable index")
        topology = compute_topology(db, retirements, output)
        reference_report = update_reference_reports(db, retirements, output)
    finally:
        lookup.get.cache_clear()
        db.close()
    with hashed_reader(Path(accepted["artifacts"]["knowledge_graph.candidate.json"]["path"])) as (reader, _):
        kind, _, metadata = next(walk_graph(reader))
        if kind != "metadata":
            raise ValueError("source graph must have leading metadata")
    baseline = read_json(AUDIT / "STRUCTURE_VALIDATED.json")["counts"]
    for key, value in (("edges", metadata["stats"]["n_edges"]), ("relations", metadata["stats"]["relations"]),
                       ("connected_components", metadata["stats"]["connected_components"]),
                       ("unique_nonself_pairs", baseline["serialized_unique_nonself_pairs"]),
                       ("parallel_pair_groups", baseline["parallel_pair_groups"]),
                       ("isolated_nodes", audit_inputs["ISOLATED_NODES.jsonl"]["rows"])):
        if topology["before"][key] != value:
            raise ValueError(f"topology does not match source baseline: {key}")
    writer = JsonlWriter(output / "EDGE_RETIREMENTS.jsonl")
    for row in retirements:
        writer.add(row)
    result = {"status": "EXACT_WITNESS_RETIREMENT_PLANNED_NOT_APPLIED", "created_at": utc_now(),
              "retired_edges": len(retirements), "affected_claims": len({row["claim_id"] for row in retirements}),
              "retired_by_relation": dict(Counter(row["retired_edge"]["relation_type"] for row in retirements)),
              "unmodified_nonexact_collision_cases": [row for row in details if row["category"] != CATEGORY],
              "retirements": writer.close(), "topology": topology, "reference_report": reference_report,
              "source_metadata": metadata, "candidate_metadata": metadata_after(metadata, topology)}
    guards(inputs)
    atomic_json(output / "PLAN.json", result)
    atomic_json(output / "PLAN_BINDING.json", fingerprint(output / "PLAN.json"))
    progress(output, "RETIREMENT_PLAN_READY", retired_edges=len(retirements), topology=topology,
             references=reference_report["counts"])


def check_inputs(output):
    inputs = read_json(output / "INPUTS.json")
    guards(inputs)
    guard(read_json(output / "PLAN_BINDING.json"))
    plan = read_json(output / "PLAN.json")
    guard(plan["retirements"])
    guards(plan["reference_report"]["artifacts"])
    return inputs, plan


def build(output):
    inputs, plan = check_inputs(output)
    deep_check(plan["retirements"])
    retirements = list(jsonl(output / "EDGE_RETIREMENTS.jsonl"))
    retired = {row["original_edge_ordinal"]: row for row in retirements}
    witnesses = {row["retained_original_edge_ordinal"]: row["retained_edge"] for row in retirements}
    required_nodes = {row["claim_id"] for row in retirements}
    for row in retirements:
        for edge in (row["retired_edge"], row["retained_edge"]):
            required_nodes.update((edge["source_id"], edge["target_id"]))
    source = inputs["source_bindings"]["artifacts"]["knowledge_graph.candidate.json"]
    expected_digests = read_json(SOURCE / "BUILD_COMPLETE.json")["digests"]
    destination, partial = output / "knowledge_graph.candidate.json", output / "knowledge_graph.candidate.json.partial"
    if destination.exists() or partial.exists():
        raise ValueError("candidate output already exists")
    counts, scoped_nodes = Counter(), {}
    digests = {key: hashlib.sha256() for key in ("nodes", "source_edges", "candidate_edges", "candidate_file")}
    found_retired, found_witnesses = set(), set()
    progress(output, "COPYING_ALL_NODES_WITHOUT_CHANGES")
    with partial.open("xb", buffering=1024 * 1024) as handle:
        def emit(text):
            data = text.encode("utf-8")
            handle.write(data)
            digests["candidate_file"].update(data)
        first_node, first_edge, edges_started = True, True, False
        with hashed_reader(Path(source["path"])) as (reader, source_sha):
            for kind, record_id, record in walk_graph(reader):
                if kind == "metadata":
                    if not same_record(record, plan["source_metadata"]):
                        raise ValueError("source metadata changed")
                    emit('{"metadata":' + compact(plan["candidate_metadata"]) + ',"concepts":{')
                elif kind == "node":
                    counts["nodes"] += 1
                    counts["claims"] += record_id.startswith("CLM:")
                    digest_update(digests["nodes"], record)
                    if record_id in required_nodes:
                        scoped_nodes[record_id] = record
                    emit(("" if first_node else ",") + compact(record_id) + ":" + compact(record))
                    first_node = False
                    if counts["nodes"] % 250000 == 0:
                        progress(output, "COPYING_ALL_NODES_WITHOUT_CHANGES", nodes=counts["nodes"])
                else:
                    if not edges_started:
                        if set(scoped_nodes) != required_nodes:
                            raise ValueError("current source lacks a required witness/claim node")
                        for row in retirements:
                            check_proposal(row, scoped_nodes[row["claim_id"]]["metadata"], scoped_nodes)
                        emit('},"edges":[')
                        edges_started = True
                    ordinal = int(record_id)
                    counts["source_edges"] += 1
                    digest_update(digests["source_edges"], record)
                    if ordinal in witnesses:
                        if not same_record(record, witnesses[ordinal]):
                            raise ValueError("retained witness changed in current source")
                        found_witnesses.add(ordinal)
                    if ordinal in retired:
                        if not same_record(record, retired[ordinal]["retired_edge"]):
                            raise ValueError("retirement preimage changed in current source")
                        found_retired.add(ordinal)
                        continue
                    counts["edges"] += 1
                    digest_update(digests["candidate_edges"], record)
                    emit(("" if first_edge else ",") + compact(record))
                    first_edge = False
                    if counts["source_edges"] % 500000 == 0:
                        progress(output, "COPYING_EDGES_EXCEPT_REVIEWED_RETIREMENTS", **dict(counts))
            if not edges_started:
                raise ValueError("expected nonempty source edge stream")
            emit("]}\n")
            observed_source_sha = source_sha.hexdigest()
        handle.flush()
        os.fsync(handle.fileno())
    if observed_source_sha != source["sha256"]:
        raise ValueError("source graph SHA-256 differs from accepted source")
    if digests["nodes"].hexdigest() != expected_digests["candidate_nodes"] or digests["source_edges"].hexdigest() != expected_digests["edges"]:
        raise ValueError("source logical record streams differ from frozen compatibility candidate")
    expected_counts = inputs["source_bindings"]["counts"]
    if (counts["nodes"], counts["claims"], counts["source_edges"], counts["edges"]) != (
            expected_counts["nodes"], expected_counts["claims"], expected_counts["edges"], expected_counts["edges"] - len(retired)):
        raise ValueError("candidate counts differ from bounded retirement scope")
    if found_retired != set(retired) or found_witnesses != set(witnesses):
        raise ValueError("not all retirement preimages and independent witnesses were found")
    check_inputs(output)
    partial.rename(destination)
    # Keep relative detail-store and old node-repair references resolvable.
    companions = {}
    for name in ("umls_details.sqlite", "NODE_REPAIRS.jsonl"):
        source_fp = inputs["source_bindings"]["artifacts"][name]
        target = output / name
        if target.exists():
            raise ValueError("candidate companion already exists")
        progress(output, "COPYING_UNCHANGED_COMPANION", name=name)
        shutil.copyfile(Path(source_fp["path"]), target)
        companions[name] = {**cheap(target), "sha256": source_fp["sha256"], "deep_verification_pending": True}
    writer = JsonlWriter(output / "CURRENT_SOURCE_WITNESS_NODES.jsonl")
    for node_id in sorted(scoped_nodes):
        writer.add(scoped_nodes[node_id])
    result = {"status": "REFERENCE_RETIREMENT_CANDIDATE_BUILT_NOT_APPLIED", "completed_at": utc_now(),
              "counts": dict(counts), "retired_edges": len(retired),
              "all_retirement_preimages_and_retained_witnesses_verified": True,
              "all_92_reference_guards_rechecked_on_current_full_claims_and_nodes": True,
              "nodes_and_claims_unchanged": True, "source_graph_sha256": observed_source_sha,
              "source_graph_full_verified_at": utc_now(),
              "digests": {key: digest.hexdigest() for key, digest in digests.items()},
              "graph": {**cheap(destination), "sha256": digests["candidate_file"].hexdigest()},
              "companions": companions, "witness_nodes": writer.close(), "formal_apply_performed": False}
    check_inputs(output)
    atomic_json(output / "BUILD_COMPLETE.json", result)
    progress(output, "BUILT_AWAITING_INDEPENDENT_VALIDATION", counts=dict(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "build"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reuse-planning-from", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output.resolve(), args.reuse_planning_from)
    else:
        if args.reuse_planning_from is not None:
            raise ValueError("planning reuse is only valid for prepare")
        build(args.output.resolve())


if __name__ == "__main__":
    main()
