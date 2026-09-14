"""Independently stream the retirement candidate; prove exact logical rollback."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retire_kg_duplicate_references import (
    OUTPUT, SOURCE, Components, atomic_json, cheap, check_inputs, check_proposal,
    claim_about_state, claim_summary, compact, deep_check, digest_update, fingerprint,
    guard, guards, has_about_issue, hashed_reader, jsonl, progress, read_json,
    restore_edges, same_record, utc_now, walk_graph,
)
from reconcile_kg_reference_checks import about_edge_checks
from neurooracle.src.kg_quality_checks import edge_claim_agreement


def validate(output):
    if (output / "VALIDATION_COMPLETE.json").exists():
        raise ValueError("validation already complete; do not overwrite")
    inputs, plan = check_inputs(output)
    build = read_json(output / "BUILD_COMPLETE.json")
    guards(build)
    deep_check(read_json(output / "PLAN_BINDING.json"))
    retirements = list(jsonl(output / "EDGE_RETIREMENTS.jsonl"))
    witnesses = {row["retained_original_edge_ordinal"]: row["retained_edge"] for row in retirements}
    scoped_nodes = {node["id"]: node for node in jsonl(output / "CURRENT_SOURCE_WITNESS_NODES.jsonl")}
    ref_states = {row["claim_id"]: row for row in jsonl(output / "REFERENCE_ISSUES.jsonl")}
    affected_states = {row["claim_id"]: row["after"] for row in jsonl(output / "AFFECTED_CLAIM_REFERENCES.jsonl")}
    about_claim_ids = set(ref_states) | set(affected_states)
    issue_edges = {}
    for name in ("ABOUT_CLAIM_POINTER_ISSUES.jsonl", "EDGE_CLAIM_DISAGREEMENTS_CORRECTED.jsonl"):
        for row in jsonl(output / name):
            if row["edge_ordinal"] in issue_edges:
                raise ValueError("duplicate residual edge issue ordinal")
            issue_edges[row["edge_ordinal"]] = row
    tracked_claim_ids = set(ref_states) | set(affected_states) | {row["claim"]["id"] for row in issue_edges.values()}
    tracked_claims, actual_about = {}, defaultdict(list)
    found_nodes, found_witnesses, found_issues = set(), set(), set()
    components, touched = Components(), bytearray()
    counts, relations = Counter(), Counter()
    node_fields, edge_fields = set(), set()
    digests = {key: hashlib.sha256() for key in ("nodes", "edges", "restored_source_edges")}
    graph_metadata = None
    progress(output, "INDEPENDENT_FULL_CANDIDATE_READ_AND_ROLLBACK")
    with hashed_reader(Path(build["graph"]["path"])) as (reader, graph_sha):
        def candidate_edges():
            nonlocal graph_metadata
            edges_started = False
            for kind, record_id, record in walk_graph(reader):
                if kind == "metadata":
                    if graph_metadata is not None or not same_record(record, plan["candidate_metadata"]):
                        raise ValueError("candidate metadata differs from bounded stats patch")
                    graph_metadata = record
                elif kind == "node":
                    if edges_started:
                        raise ValueError("candidate has nodes after edge records")
                    components.add(record_id)
                    touched.append(0)
                    counts["nodes"] += 1
                    counts["claims"] += record_id.startswith("CLM:")
                    digest_update(digests["nodes"], record)
                    node_fields.update((record.get("metadata") or {}).keys())
                    if record_id in scoped_nodes:
                        if not same_record(record, scoped_nodes[record_id]):
                            raise ValueError("current-source witness node/claim changed")
                        found_nodes.add(record_id)
                    if record_id in tracked_claim_ids:
                        tracked_claims[record_id] = claim_summary(record_id, record)
                    if counts["nodes"] % 250000 == 0:
                        progress(output, "VERIFYING_UNCHANGED_NODE_AND_CLAIM_RECORDS", nodes=counts["nodes"])
                else:
                    edges_started = True
                    ordinal = int(record_id)
                    counts["edges"] += 1
                    digest_update(digests["edges"], record)
                    source, target = record["source_id"], record["target_id"]
                    components.union(source, target)
                    touched[components.ids[source]] = touched[components.ids[target]] = 1
                    relations[record["relation_type"]] += 1
                    counts["self_loop_records"] += source == target
                    edge_fields.update((record.get("metadata") or {}).keys())
                    if record["relation_type"] == "about" and source in about_claim_ids:
                        actual_about[source].append(record)
                    if ordinal in issue_edges:
                        expected = issue_edges[ordinal]
                        if not same_record(record, expected["edge"]):
                            raise ValueError("residual issue does not match translated candidate ordinal")
                        checker = about_edge_checks if record["relation_type"] == "about" else edge_claim_agreement
                        if checker(record, tracked_claims[expected["claim"]["id"]]) != expected["issues"]:
                            raise ValueError("residual issue no longer matches current claim")
                        found_issues.add(ordinal)
                    if counts["edges"] % 500000 == 0:
                        progress(output, "VERIFYING_EDGES_AND_EXACT_ROLLBACK", edges=counts["edges"])
                    yield record

        for original_ordinal, edge, restored in restore_edges(candidate_edges(), retirements, build["counts"]["source_edges"]):
            digest_update(digests["restored_source_edges"], edge)
            if restored:
                counts["retired_edges_reinserted"] += 1
            elif original_ordinal in witnesses:
                if not same_record(edge, witnesses[original_ordinal]):
                    raise ValueError("independently retained witness is missing or changed")
                found_witnesses.add(original_ordinal)
        observed_sha = graph_sha.hexdigest()
    if graph_metadata is None or observed_sha != build["graph"]["sha256"]:
        raise ValueError("candidate full-file verification failed")
    expected = read_json(SOURCE / "BUILD_COMPLETE.json")["digests"]
    if digests["nodes"].hexdigest() != expected["candidate_nodes"]:
        raise ValueError("not all source nodes/claims were preserved exactly")
    if digests["restored_source_edges"].hexdigest() != expected["edges"]:
        raise ValueError("source edge stream cannot be reconstructed exactly")
    if digests["edges"].hexdigest() != build["digests"]["candidate_edges"]:
        raise ValueError("candidate edge digest changed after build")
    if found_nodes != set(scoped_nodes) or found_witnesses != set(witnesses) or found_issues != set(issue_edges):
        raise ValueError("not all current-source witnesses and residual issues were found")
    for key in ("nodes", "claims", "edges"):
        if counts[key] != build["counts"][key]:
            raise ValueError(f"independent count mismatch: {key}")
    topology = {"connected_components": components.count, "isolated_nodes": touched.count(0),
                "relations": dict(sorted(relations.items())), "self_loop_records": counts["self_loop_records"]}
    for key, value in topology.items():
        if value != plan["topology"]["after"][key]:
            raise ValueError(f"candidate topology differs from plan: {key}")
    if graph_metadata["stats"]["n_edges"] != counts["edges"] or graph_metadata["stats"]["n_concepts"] != counts["nodes"]:
        raise ValueError("graph metadata counts are stale")
    for claim_id in about_claim_ids:
        actual = claim_about_state(tracked_claims[claim_id], actual_about[claim_id])
        expected_state = affected_states.get(claim_id, ref_states.get(claim_id))
        if actual != expected_state or has_about_issue(actual) != (claim_id in ref_states):
            raise ValueError("residual claim reference inventory is stale")
    for row in retirements:
        check_proposal(row, scoped_nodes[row["claim_id"]]["metadata"], scoped_nodes)
    # Newly isolated nodes are retained: this step does not authorize entity removal.
    affected_node_ids = {row["retired_edge"][key] for row in retirements for key in ("source_id", "target_id")}
    newly_isolated = sorted(node_id for node_id in affected_node_ids if not touched[components.ids[node_id]])
    if newly_isolated != plan["topology"]["newly_isolated_node_ids"]:
        raise ValueError("new-isolate inventory mismatch")
    for row in retirements:
        components.union(row["retired_edge"]["source_id"], row["retired_edge"]["target_id"])
    if components.count != plan["topology"]["before"]["connected_components"]:
        raise ValueError("restored topology does not match source")
    verified_companions = {}
    for name, fp in build["companions"].items():
        progress(output, "FULL_COMPANION_COPY_VERIFICATION", name=name)
        deep_check(fp)
        verified_companions[name] = {key: value for key, value in fp.items() if key != "deep_verification_pending"}
    guards(build)
    check_inputs(output)
    result = {"status": "FULL_CANDIDATE_AND_EXACT_LOGICAL_ROLLBACK_VERIFIED", "verified_at": utc_now(),
              "counts": dict(counts), "candidate_sha256": observed_sha,
              "digests": {key: digest.hexdigest() for key, digest in digests.items()},
              "all_nodes_and_claims_identical_to_accepted_source": True,
              "full_claim_compatibility_validation_reused": inputs["source_bindings"]["artifacts"]["VALIDATION_COMPLETE.json"],
              "all_unretired_edge_records_unchanged_in_order": True,
              "all_source_edge_records_exactly_recoverable": True,
              "source_metadata_recoverable_from_plan": True,
              "retained_witnesses_verified": len(found_witnesses),
              "residual_issue_edges_verified_in_current_candidate": len(found_issues),
              "topology": topology, "newly_isolated_node_ids": newly_isolated,
              "node_metadata_field_union": len(node_fields), "edge_metadata_field_union": len(edge_fields),
              "verified_companions": verified_companions, "formal_sources_guarded_unchanged": True,
              "formal_apply_performed": False, "node_merges": 0}
    atomic_json(output / "VALIDATION_COMPLETE.json", result)
    progress(output, "VALIDATED_AWAITING_FREEZE", counts=dict(counts), topology=topology)


def freeze(output):
    if (output / "RETIREMENT_ACCEPTANCE.json").exists():
        raise ValueError("candidate already frozen")
    inputs, plan = check_inputs(output)
    build = read_json(output / "BUILD_COMPLETE.json")
    validation = read_json(output / "VALIDATION_COMPLETE.json")
    guards(build)
    guards(validation)
    if validation["status"] != "FULL_CANDIDATE_AND_EXACT_LOGICAL_ROLLBACK_VERIFIED":
        raise ValueError("candidate not validated")
    suites = ElementTree.parse(output / "TEST_RESULTS.xml").getroot()
    totals = {key: sum(int(suite.get(key, 0)) for suite in suites.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    if totals["tests"] < 129 or any(totals[key] for key in ("failures", "errors", "skipped")):
        raise ValueError("regression tests did not all pass")
    artifacts = {}
    cached = {"knowledge_graph.candidate.json": build["graph"], **validation["verified_companions"]}
    for path in sorted(output.iterdir()):
        if path.name == "RUN_STATE.json" or not path.is_file():
            continue
        if path.suffix in {".partial", ".tmp"}:
            raise ValueError("unfinished artifact in candidate output")
        artifacts[path.name] = cached.get(path.name) or fingerprint(path)
    if "RETIREMENT_REPORT.md" not in artifacts:
        raise ValueError("candidate handoff report is missing")
    result = {"status": "EXACT_REFERENCE_DUPLICATES_RETIRED_CANDIDATE_ONLY_NOT_APPLIED",
              "frozen_at": utc_now(), "counts": {key: build["counts"][key] for key in ("nodes", "claims", "edges")},
              "retired_edges": plan["retired_edges"], "affected_claims": plan["affected_claims"],
              "retired_by_relation": plan["retired_by_relation"], "topology": plan["topology"],
              "remaining_reference_issues": plan["reference_report"]["counts"],
              "validation": validation, "tests": totals, "artifacts": artifacts,
              "implementation": inputs["implementation"], "source_acceptance": inputs["source_acceptance"],
              "source_graph": inputs["source_bindings"]["artifacts"]["knowledge_graph.candidate.json"],
              "formal_sources": inputs["source_bindings"]["formal_sources"],
              "formal_apply_performed": False, "source_candidates_modified": False,
              "remaining_nonexact_cases_modified": False, "metadata_fields_added": 0, "node_merges": 0}
    atomic_json(output / "RETIREMENT_ACCEPTANCE.json", result)
    atomic_json(output / "RUN_STATE.json", {"status": "FROZEN_NOT_APPLIED", "updated_at": utc_now(),
                                          "acceptance": fingerprint(output / "RETIREMENT_ACCEPTANCE.json")})
    print(compact({"status": result["status"], "counts": result["counts"], "retired_edges": result["retired_edges"],
                   "remaining_reference_issues": result["remaining_reference_issues"], "tests": totals}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "freeze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    {"validate": validate, "freeze": freeze}[args.phase](args.output.resolve())


if __name__ == "__main__":
    main()
