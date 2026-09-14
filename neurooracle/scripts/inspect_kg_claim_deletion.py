"""Read-only full-graph deletion preflight for the 50 R21 source mismatches.

Writes small evidence artifacts only. Never modifies a graph, a runtime reader,
an audit seal, a source archive, or the active candidate pointer.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kg_overnight_report as journal
from build_umls_simplification_candidate import Components, compact, digest_update, hashed_reader, walk_graph
from kg_accepted_candidate_lineage import Bindings, require
from record_kg_source_findings import record_digest
from retire_kg_duplicate_references import guards, same_record

OUTPUT = journal.OUTPUT / "round22_source_deletion_preflight"
R21 = journal.OUTPUT / "round21_repair_readiness"
R21_SHA = "732c645c61034342bae43a43b57b557779aac75adf8dfe1305ffa3d27a152bd2"
FLOOR = 16 * 1024**3


def select_targets(queue, bundles):
    require(len(queue) == len({r["claim_id"] for r in queue}), "duplicate queue ID")
    require(len(bundles) == len({r["claim_id"] for r in bundles}), "duplicate preimage ID")
    lookup = {r["claim_id"]: r for r in bundles}
    require(set(lookup) == {r["claim_id"] for r in queue}, "preimage/queue scope differs")
    selected = [r for r in queue if r["route"] == "source_mismatch"]
    require(selected, "no source-mismatch targets")
    nodes, edges = {}, {}
    for row in selected:
        cid = row["claim_id"]
        require(row["classification"] == "confirmed" and row["source_round"] == 5
                and row["original_issue_category"] == "unsupported_by_bound_input_abstract",
                "unconfirmed or wrong-source target")
        bundle = lookup[cid]
        node = bundle["before_node"]
        require(cid.startswith("CLM:") and node["id"] == cid, "not a claim node")
        require(record_digest(node) == row["current_node_sha256"] == bundle["before_node_sha256"], "node preimage hash differs")
        require(not bundle["actual_applied_mutations"] and not row["actual_graph_mutations"], "already mutated target")
        nodes[cid] = node
        ordinals = [r["candidate_edge_ordinal"] for r in bundle["before_edges"]]
        require(len(set(ordinals)) == len(ordinals), "duplicate closure ordinal")
        require(sorted(ordinals) == sorted(row["related_edge_ordinals"]), "incomplete edge closure")
        about, science = [], []
        for item in bundle["before_edges"]:
            ordinal, edge = item["candidate_edge_ordinal"], item["edge"]
            require(type(ordinal) is int and ordinal > 0, "invalid edge ordinal")
            require(record_digest(edge) == bundle["before_edge_sha256"][str(ordinal)], "edge preimage hash differs")
            if edge["relation_type"] == "about":
                require(edge["source_id"] == cid, "about not owned by deleted claim")
                about.append(ordinal)
            else:
                require((edge.get("metadata") or {}).get("claim_id") == cid, "cannot borrow another claim's science")
                science.append(ordinal)
            require(ordinal not in edges, "edge shared between removal bundles")
            edges[ordinal] = edge
        require(len(about) == 2 and len(science) in (0, 1), "unexpected ownership profile")
        require(sorted(about) == sorted(row["about_ordinals"])
                and sorted(science) == sorted(row["owned_science_ordinals"]), "edge role classification differs")
    return nodes, edges


def exact_references(value, targets, path=()):
    """JSON paths to exact IDs; embedded prose/compound IDs are not references."""
    if isinstance(value, str):
        if value in targets:
            yield {"json_path": list(path), "claim_id": value}
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in targets:
                yield {"json_path": [*path, {"dictionary_key": key}], "claim_id": key}
            yield from exact_references(child, targets, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from exact_references(child, targets, (*path, index))


def scan(records, nodes, edges, progress=None):
    ids = set(nodes)
    needle = re.compile("|".join(re.escape(cid) for cid in sorted(ids)))
    old, new = Components(), Components()
    counts = {"before": Counter(), "if_deleted": Counter()}
    relations = {k: Counter() for k in counts}
    domains, sources = Counter(), Counter()
    unions = {k: {"nodes": set(), "edges": set()} for k in counts}
    digests = {k: hashlib.sha256() for k in ("before_nodes", "before_edges", "retained_nodes", "retained_edges")}
    seen_nodes, seen_edges, removals, references = set(), set(), [], []
    metadata, degrees, new_degrees, node_ordinal = None, None, None, 0

    def inspect_refs(kind, key, record, payload=None):
        text = payload if payload is not None else compact(record)
        if needle.search(text):
            hits = list(exact_references(record, ids))
            if hits:
                references.append({"kind": kind, "key": key, "record_sha256": record_digest(record), "references": hits, "record": record})

    for kind, key, record in records:
        if kind == "metadata":
            require(metadata is None and not old.ids, "unexpected metadata order")
            metadata = record
            inspect_refs(kind, key, record)
            continue
        require(metadata is not None, "metadata missing")
        if kind == "node":
            require(degrees is None and record["id"] == key, "node order/identity differs")
            old.add(key)
            node_ordinal += 1
            counts["before"]["nodes"] += 1
            counts["before"]["claims"] += key.startswith("CLM:")
            unions["before"]["nodes"].update(record.get("metadata") or {})
            payload = compact(record)
            digests["before_nodes"].update((payload + "\n").encode("utf-8"))
            if key in ids:
                require(same_record(record, nodes[key]), "actual node differs from exact preimage")
                seen_nodes.add(key)
                removals.append({"source_node_ordinal": node_ordinal, "node_id": key, "before": record,
                                 "before_sha256": record_digest(record), "after": None})
                continue
            new.add(key)
            counts["if_deleted"]["nodes"] += 1
            counts["if_deleted"]["claims"] += key.startswith("CLM:")
            unions["if_deleted"]["nodes"].update(record.get("metadata") or {})
            domains.update(record.get("domain_tags") or [])
            sources[record.get("source_vocab") or ""] += 1
            digests["retained_nodes"].update((payload + "\n").encode("utf-8"))
            inspect_refs(kind, key, record, payload)
            if progress and node_ordinal % 500000 == 0:
                progress("SCANNING_NODES", nodes=node_ordinal)
        else:
            require(kind == "edge", "unknown graph section")
            if degrees is None:
                degrees, new_degrees = bytearray(len(old.ids)), bytearray(len(new.ids))
            ordinal = int(key)
            require(ordinal == counts["before"]["edges"] + 1, "edge ordinal gap")
            sid, tid = record["source_id"], record["target_id"]
            old.union(sid, tid)
            degrees[old.ids[sid]] = degrees[old.ids[tid]] = 1
            counts["before"]["edges"] += 1
            counts["before"]["self_loops"] += sid == tid
            relations["before"][record["relation_type"]] += 1
            unions["before"]["edges"].update(record.get("metadata") or {})
            payload = compact(record)
            digests["before_edges"].update((payload + "\n").encode("utf-8"))
            direct = sid in ids or tid in ids or (record.get("metadata") or {}).get("claim_id") in ids
            require(direct == (ordinal in edges), "unexpected incident/owned edge outside deletion scope")
            if ordinal in edges:
                require(same_record(record, edges[ordinal]), "actual edge differs from exact preimage")
                seen_edges.add(ordinal)
            else:
                new.union(sid, tid)
                new_degrees[new.ids[sid]] = new_degrees[new.ids[tid]] = 1
                counts["if_deleted"]["edges"] += 1
                counts["if_deleted"]["self_loops"] += sid == tid
                relations["if_deleted"][record["relation_type"]] += 1
                unions["if_deleted"]["edges"].update(record.get("metadata") or {})
                digests["retained_edges"].update((payload + "\n").encode("utf-8"))
                inspect_refs(kind, ordinal, record, payload)
            if progress and ordinal % 500000 == 0:
                progress("SCANNING_EDGES", edges=ordinal)
    require(metadata is not None and degrees is not None, "incomplete graph")
    require(seen_nodes == ids and seen_edges == set(edges), "deletion target absent")
    topology = {}
    for label, cc, degree in (("before", old, degrees), ("if_deleted", new, new_degrees)):
        topology[label] = {**dict(counts[label]), "self_loops": counts[label]["self_loops"],
                           "connected_components": cc.count, "isolated_nodes": degree.count(0), "relations": dict(relations[label])}
    newly_isolated = sorted(nid for nid, index in new.ids.items() if degrees[old.ids[nid]] and not new_degrees[index])
    return {"topology": topology, "metadata_before": metadata,
            "metadata_stats_if_deleted": {"n_concepts": counts["if_deleted"]["nodes"], "n_edges": counts["if_deleted"]["edges"],
                 "domains": dict(domains), "sources": dict(sources), "relations": dict(relations["if_deleted"]), "connected_components": new.count},
            "ordered_record_digests": {k: d.hexdigest() for k, d in digests.items()},
            "metadata_key_unions": {k: {t: sorted(s) for t, s in v.items()} for k, v in unions.items()},
            "newly_isolated_retained_node_ids": newly_isolated, "retained_exact_references": references,
            "node_removal_preimages": removals,
            "edge_removal_preimages": [{"source_edge_ordinal": o, "before": e, "before_sha256": record_digest(e), "after": None} for o, e in sorted(edges.items())]}


def storage_check(free, graph_bytes):
    require(type(free) is int and free >= 0 and type(graph_bytes) is int and graph_bytes > 0, "invalid storage measurement")
    return {"available_bytes": free, "conservative_new_graph_bytes": graph_bytes, "required_reserve_bytes": FLOOR,
            "remaining_after_one_graph_bytes": free - graph_bytes,
            "one_graph_fits_with_reserve": free - graph_bytes >= FLOOR,
            "companion_copies_and_other_writes_not_included": True}


def write_once(name, value):
    path = OUTPUT / name
    require(not path.exists(), "preflight artifact exists: " + name)
    if name.endswith(".jsonl"):
        journal.atomic_text(path, "".join(compact(r) + "\n" for r in value))
    else:
        journal.atomic_json(path, value)
    return journal.fingerprint(path)


def main():
    require(not (OUTPUT / "PREFLIGHT_COMPLETE.json").exists(), "preflight already complete")
    tests = Counter()
    for suite in ElementTree.parse(OUTPUT / "TEST_RESULTS.xml").getroot().iter("testsuite"):
        tests.update({k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")})
    require(tests["tests"] >= 20 and not any(tests[k] for k in ("failures", "errors", "skipped")), "tests missing or failed")
    b = Bindings()
    complete_path = b.pin(R21 / "READINESS_COMPLETE.json", R21_SHA)
    complete = journal.read_json(complete_path)
    build = b.json(complete["build"])
    for fp in [*build["bindings"].values(), *build["artifacts"].values(), *build["implementation"].values(), build["test_results"]]:
        b.check(fp)
    campaign = journal.read_json(journal.OUTPUT / "CAMPAIGN.json")
    require(campaign["current_acceptance"] == build["current_acceptance"] and campaign["automation_id"] is None, "candidate/manual scope changed")
    accepted = b.json(build["current_acceptance"])
    plan = b.json(accepted["artifacts"]["PLAN.json"])
    guards(build); guards(campaign)
    nodes, edges = select_targets(b.rows(build["artifacts"]["REPAIR_READINESS_QUEUE.jsonl"]), b.rows(build["artifacts"]["ROLLBACK_PREIMAGES.jsonl"]))
    require(len(nodes) == 50 and len(edges) == 149, "authorized 50/149 scope differs")
    graph = build["current_graph"]
    def progress(phase, **counts):
        state = {"status": "RUNNING_READ_ONLY", "phase": phase, "updated_at": journal.utc_now(), **counts}
        journal.atomic_json(OUTPUT / "RUN_STATE.json", state)
        print(json.dumps(state), flush=True)
    progress("STARTING_FULL_SOURCE_HASH_AND_DELETION_IMPACT", targets=len(nodes))
    with hashed_reader(Path(graph["path"])) as (reader, sha):
        result = scan(walk_graph(reader), nodes, edges, progress)
        actual_sha = sha.hexdigest()
    require(actual_sha == graph["sha256"], "full current graph SHA changed")
    require(same_record(result["topology"]["before"], plan["candidate_topology"]), "before topology differs from accepted R12")
    require(same_record(result["metadata_before"], plan["metadata_after"]), "before metadata differs")
    for kind in ("nodes", "edges"):
        require(result["ordered_record_digests"]["before_" + kind] == plan["expected_record_digests"][kind], "source ordered digest differs")
    guards(build); guards(campaign); guards(b.files)
    artifacts = {}
    for name, key in (("DELETE_NODE_PREIMAGES.jsonl", "node_removal_preimages"), ("DELETE_EDGE_PREIMAGES.jsonl", "edge_removal_preimages"),
                      ("RETAINED_REFERENCES.jsonl", "retained_exact_references")):
        values = result.pop(key)
        artifacts[name] = {**write_once(name, values), "rows": len(values)}
    result.update(status="DELETION_IMPACT_VERIFIED_NOT_APPLIED", completed_at=journal.utc_now(),
                  source_graph=graph, full_graph_sha256_verified=actual_sha, source_review_receipt=journal.fingerprint(complete_path),
                  source_review_build=complete["build"], current_acceptance=build["current_acceptance"], bindings=b.files,
                  artifacts=artifacts, tests=dict(tests), test_results=journal.fingerprint(OUTPUT / "TEST_RESULTS.xml"),
                  implementation={name: journal.fingerprint(journal.REPO / name) for name in (
                      "neurooracle/scripts/inspect_kg_claim_deletion.py", "neurooracle/tests/test_kg_claim_deletion_preflight.py",
                      "neurooracle/scripts/build_umls_simplification_candidate.py", "neurooracle/scripts/streaming_graph_json.py")},
                  storage=storage_check(shutil.disk_usage(journal.REPO).free, graph["bytes"]),
                  graph_mutations=0, candidate_copies=0, formal_data_mutations=0, models_called=0, training_jobs_started=0,
                  original_audits_preserved_in_preimages=True, active_candidate_advanced=False,
                  authorizing_user_message="可以，如果数量不多，我们直接去掉这些数据也行",
                  pending_decisions=["deletion_candidate_destination_and_full_candidate_budget"],
                  limits=["Projected counts/topology are a full-stream calculation, not an applied deletion.",
                          "Exact ID reference check covers graph JSON only; external archived/source files remain historical and read-only.",
                          "Newly isolated shared concept nodes are retained, not automatically removed.",
                          "Copied preimages are complete restoration inputs, not an accepted apply/rollback execution."])
    if artifacts["RETAINED_REFERENCES.jsonl"]["rows"]:
        result["pending_decisions"].append("retained_reference_disposition")
    write_once("PREFLIGHT_COMPLETE.json", result)
    journal.atomic_json(OUTPUT / "RUN_STATE.json", {"status": result["status"], "completed_at": result["completed_at"], "graph_mutations": 0})
    print(json.dumps({k: result[k] for k in ("status", "completed_at", "topology", "storage", "pending_decisions", "newly_isolated_retained_node_ids", "tests")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
