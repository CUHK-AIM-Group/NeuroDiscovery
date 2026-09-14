"""Read-only BASE-to-Round12 evidence adapter; never read/write whole graphs.

Uses the already completed full-file and inverse-stream boundary proofs.
Old source ordinals remain explicitly labeled; current ordinals are derived
from the complete accepted retirement list, never from a filtered cohort.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from pathlib import Path

import kg_overnight_report as journal
from record_kg_source_findings import record_digest
from retire_kg_duplicate_references import deep_check, guards, jsonl, read_json, same_record

R7 = journal.OUTPUT / "round07_negation_expression_census"
R9 = journal.OUTPUT / "round09_negation_source_groups"
R12 = journal.OUTPUT / "round12_conservative_candidate"
ACCEPTANCE_SHA = "ea4c879134fd013e1247ab721a9955f50a1e82c48508716cea1e97302cfa7118"
TRIAGE_SHA = "76eda6bb8959282c621259aa355d3f4c3d4ea1ca9c74c29e1fc48bc952f356cb"
GROUPS_BUILD_SHA = "c1fa3018d025253d4727b95f82ad3c824147af81dfd57a07e9d2b96d09295194"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, "duplicate identity: " + str(row[key]))
        result[row[key]] = row
    return result


def file_identity(fp):
    return {k: fp[k] for k in ("path", "bytes", "mtime_ns", "sha256")}


class Bindings:
    def __init__(self):
        self.files = {}

    def check(self, fp):
        require(fp["bytes"] < 256 * 1024**2, "whole-file scan outside small evidence boundary")
        path = Path(fp["path"])
        path.resolve(strict=True).relative_to(journal.REPO.resolve())
        if fp["path"] in self.files:
            require(same_record(file_identity(self.files[fp["path"]]), file_identity(fp)), "conflicting evidence binding")
            guards(fp)
        else:
            deep_check(fp)
            self.files[fp["path"]] = fp
        return path

    def pin(self, path, expected_sha=None):
        require(path.stat().st_size < 256 * 1024**2, "large file cannot be rebound as small evidence")
        fp = journal.fingerprint(path)
        require(expected_sha is None or fp["sha256"] == expected_sha, "frozen root digest differs")
        if fp["path"] in self.files:
            require(same_record(file_identity(self.files[fp["path"]]), fp), "root binding conflict")
        else:
            self.files[fp["path"]] = fp
        return path

    def json(self, fp):
        return read_json(self.check(fp))

    def rows(self, fp):
        result = list(jsonl(self.check(fp)))
        require("rows" not in fp or len(result) == fp["rows"], "frozen row count differs")
        return result


def candidate_ordinal(source_ordinal, retired_ordinals):
    require(type(source_ordinal) is int and source_ordinal > 0, "source ordinal must be a positive integer")
    require(all(type(o) is int and o > 0 for o in retired_ordinals), "invalid retired ordinal")
    require(list(retired_ordinals) == sorted(set(retired_ordinals)), "retired ordinals must be complete, sorted and unique")
    if source_ordinal in retired_ordinals:
        return None
    return source_ordinal - bisect_left(retired_ordinals, source_ordinal)


def touches(edge, ids):
    return bool({edge["source_id"], edge["target_id"], (edge.get("metadata") or {}).get("claim_id")} & ids)


def project_witnesses(nodes, edges, node_patches, edge_patches, retirements):
    """Project a bounded witness set, retaining both unambiguous versions."""
    nm = indexed(nodes, "id")
    em = indexed(edges, "source_candidate_edge_ordinal")
    np = indexed(node_patches, "node_id")
    ep = indexed(edge_patches, "source_candidate_edge_ordinal")
    retired = indexed(retirements, "source_candidate_edge_ordinal")
    require(not set(ep) & set(retired), "edge mutation/retirement overlap")
    order = sorted(retired)
    projected_nodes = deepcopy(nm)
    node_hits, patch_hits, retirement_hits = [], [], []
    for nid, row in np.items():
        require(row["before"]["id"] == nid == row["after"]["id"], "patch changes node identity")
        if nid in nm:
            require(same_record(nm[nid], row["before"]), "full node preimage mismatch")
            projected_nodes[nid] = deepcopy(row["after"])
            node_hits.append(nid)
    projected_edges = []
    for ordinal, row in sorted(em.items()):
        current_ordinal = candidate_ordinal(ordinal, order)
        if ordinal in retired:
            require(same_record(row["edge"], retired[ordinal]["before"]), "retirement full preimage mismatch")
            retirement_hits.append(ordinal)
            continue
        edge = row["edge"]
        if ordinal in ep:
            require(same_record(edge, ep[ordinal]["before"]), "edge full preimage mismatch")
            edge = ep[ordinal]["after"]
            patch_hits.append(ordinal)
        projected_edges.append({"source_candidate_edge_ordinal": ordinal,
                                "candidate_edge_ordinal": current_ordinal, "edge": deepcopy(edge)})
    require(len({r["candidate_edge_ordinal"] for r in projected_edges}) == len(projected_edges), "candidate ordinal collision")
    return projected_nodes, projected_edges, {"patched_node_ids_in_witnesses": sorted(node_hits),
        "patched_source_edge_ordinals_in_witnesses": patch_hits, "retired_source_edge_ordinals_in_witnesses": retirement_hits}


def verify_boundary(acceptance, inputs, plan, build, validation, census):
    require(acceptance["status"] == "OVERNIGHT_KG_CANDIDATE_ACCEPTED_NOT_FORMALLY_APPLIED", "candidate not accepted")
    require(validation["status"] == "CONSERVATIVE_CANDIDATE_FULLY_VERIFIED_WITH_ORDERED_SOURCE_ROLLBACK", "full candidate inverse proof missing")
    for key in ("ordered_source_records_recoverable", "all_other_records_identical", "claim_evidence_payloads_unchanged"):
        require(validation[key] is True, "full boundary proof missing: " + key)
    for key, expected in {"changed_nodes": 5, "changed_edges": 1, "retired_edges": 24,
                          "added_nodes": 0, "removed_nodes": 0, "added_edges": 0}.items():
        require(type(validation[key]) is int and validation[key] == expected, "unreviewed graph delta")
    require(same_record(acceptance["upstream_acceptance"], inputs["source_acceptance"])
            and same_record(census["source_acceptance"], inputs["source_acceptance"]), "different source acceptance")
    require(same_record(file_identity(census["source_graph"]), file_identity(inputs["source_graph"])), "census from another source graph")
    require(build["source_sha256"] == inputs["source_graph"]["sha256"], "build source SHA differs")
    require(same_record(census["counts"], inputs["source_counts"])
            and same_record(build["source_counts"], inputs["source_counts"]), "source counts differ")
    require(same_record(acceptance["counts"], validation["counts"]) and same_record(build["counts"], validation["counts"])
            and same_record(plan["counts"], validation["counts"]), "candidate counts differ")
    require(acceptance["counts"] == {**inputs["source_counts"], "edges": inputs["source_counts"]["edges"] - 24}, "candidate count delta differs")
    require(validation["candidate_sha256"] == build["graph"]["sha256"]
            == acceptance["artifacts"]["knowledge_graph.candidate.json"]["sha256"], "candidate full-file SHA differs")
    for kind in ("nodes", "edges"):
        require(validation["digests"]["restored_" + kind] == inputs["source_record_digests"][kind]
                == build["digests"]["source_" + kind], "incomplete ordered rollback proof")
        require(validation["digests"][kind] == plan["expected_record_digests"][kind]
                == build["digests"][kind], "candidate ordered digest differs")
    require(acceptance["formal_apply_performed"] is False and acceptance["models_called"] == 0
            and acceptance["training_jobs_started"] == 0, "candidate activity scope changed")


def load_current_census():
    bindings = Bindings()
    campaign = read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(Path(campaign["current_acceptance"]["path"]) == R12 / "REPAIR_ACCEPTANCE.json"
            and campaign["current_acceptance"]["sha256"] == ACCEPTANCE_SHA, "current candidate has advanced")
    acceptance = bindings.json(campaign["current_acceptance"])
    guards(acceptance)
    def accepted(name):
        fp = acceptance["artifacts"][name]
        require(Path(fp["path"]) == R12 / name, "accepted artifact path substituted")
        return bindings.json(fp)
    inputs, plan, build, validation = (accepted(name) for name in ("INPUTS.json", "PLAN.json", "BUILD_COMPLETE.json", "VALIDATION_COMPLETE.json"))
    for fp in acceptance["implementation"].values():
        bindings.check(fp)
    guards(inputs)
    guards(build)
    guards(campaign["formal_sources"])
    triage = read_json(bindings.pin(R7 / "TRIAGE_BUILD.json", TRIAGE_SHA))
    census = bindings.json(triage["census_receipt"])
    verify_boundary(acceptance, inputs, plan, build, validation, census)
    ops = [bindings.rows(acceptance["artifacts"][name]) for name in ("NODE_PATCHES.jsonl", "EDGE_PATCHES.jsonl", "EDGE_RETIREMENTS.jsonl")]
    require([len(rows) for rows in ops] == [5, 1, 24], "incomplete accepted operation list")
    for name in ("NODE_PATCHES.jsonl", "EDGE_PATCHES.jsonl", "EDGE_RETIREMENTS.jsonl"):
        require(same_record(plan["artifacts"][name], acceptance["artifacts"][name]), "accepted plan operation binding differs")
    queue = bindings.rows(triage["artifacts"]["SOURCE_EXPRESSION_REVIEW_QUEUE.jsonl"])
    ids = set(indexed(queue, "claim_id"))
    require(len(ids) == 2640 and not ids & {p["node_id"] for p in ops[0]}, "expression cohort includes changed claim")
    for row in ops[1]:
        require(not touches(row["before"], ids) and not touches(row["after"], ids), "expression cohort incident edge changed")
    require(not any(touches(row["before"], ids) for row in ops[2]), "expression cohort incident edge retired")
    source_nodes = bindings.rows(census["artifacts"]["SELECTED_CLAIMS.jsonl"])
    source_edges = bindings.rows(census["artifacts"]["CURRENT_RELATED_EDGES.jsonl"])
    nodes, edges, overlap = project_witnesses(source_nodes, source_edges, *ops)
    require(set(overlap["retired_source_edge_ordinals_in_witnesses"]) == {r["source_candidate_edge_ordinal"] for r in ops[2]}, "selected census retirement coverage differs")
    for r in queue:
        require(r["claim_id"] in nodes and record_digest(nodes[r["claim_id"]]) == r["current_record_sha256"], "queue full current claim differs")
    groups_build = read_json(bindings.pin(R9 / "BUILD_RECEIPT.json", GROUPS_BUILD_SHA))
    groups = bindings.rows(groups_build["artifacts"]["EXACT_PROVENANCE_GROUPS.jsonl"])
    prior_issues, prior_extra = set(), set()
    for key, fp in triage["queue_bindings"].items():
        rows = bindings.rows(fp)
        (prior_extra if key == "round06_review_candidates" else prior_issues).update(indexed(rows, "claim_id"))
    prior_issues.update(indexed(bindings.rows(groups_build["artifacts"]["CONFIRMED_EXPRESSION_ISSUES.jsonl"]), "claim_id"))
    prior_extra.update(indexed(bindings.rows(groups_build["artifacts"]["MIXED_SCOPE_REVIEW_CANDIDATES.jsonl"]), "claim_id"))
    require(prior_issues == set(inputs["protected_issue_ids"]) and len(prior_issues) == 72, "prior confirmed register differs")
    require(prior_extra == set(inputs["protected_extra_review_ids"]) and len(prior_extra) == 12 and not prior_issues & prior_extra, "prior review register differs")
    proof = {"method": "complete_frozen_R7_source_witnesses_plus_accepted_R12_exact_delta_and_full_inverse_proof",
        "source_acceptance": inputs["source_acceptance"], "current_acceptance": campaign["current_acceptance"],
        "source_graph": inputs["source_graph"], "current_graph": build["graph"], "full_current_verified_at": validation["completed_at"],
        "counts": acceptance["counts"], "source_census_claims": len(source_nodes), "source_census_edges": len(source_edges),
        "current_census_claims": len(nodes), "current_census_edges": len(edges), "complete_accepted_delta_counts": [5, 1, 24],
        "witness_operation_intersections": overlap, "source_expression_queue_claims": 2640,
        "queue_changed_claims": 0, "queue_changed_or_retired_incident_edges": 0,
        "prior_confirmed_queue_overlap": len(ids & prior_issues), "prior_extra_queue_overlap": len(ids & prior_extra),
        "old_source_ordinals_are_current_ordinals": False, "full_graph_rehashed_this_read_only_round": False,
        "formal_sources": campaign["formal_sources"], "graph_changes": 0}
    return {"bindings": bindings, "proof": proof, "current": {cid: nodes[cid] for cid in ids}, "all_edges": edges,
            "groups": groups, "prior_issues": prior_issues, "prior_extra": prior_extra, "queue": queue}
