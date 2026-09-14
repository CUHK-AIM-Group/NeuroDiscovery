"""Read-only full candidate audit: strict mapping admission and metadata coverage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_umls_node_repair_candidate import DEFAULT_OUTPUT as CANDIDATE
from build_umls_simplification_candidate import (
    REPO, UMLS_ROOT, atomic_json, cheap, compact, digest_update, file_sha,
    hashed_reader, progress, read_json, summarize_triples, utc_now, walk_graph,
)
from finalize_umls_node_repair_runtime import check_frozen_data
from neurooracle.src.kge.experiment_admission import (
    AdmissionView, POLICY_VERSION, is_umls_mapping, mapping_admission_reason,
)
from neurooracle.src.kge.triple_loader import load_triples_from_kg, split_triples
from neurooracle.src.metadata_field_audit import Coverage, node_class, static_key_reads


DEFAULT_OUTPUT = UMLS_ROOT / "experiment_admission_metadata_audit_v1_20260907"


def check_candidate() -> dict:
    pointer_path = CANDIDATE / "CURRENT_ACCEPTANCE.json"
    pointer = read_json(pointer_path)
    current_path = Path(pointer["acceptance"]["path"])
    if file_sha(current_path) != pointer["acceptance"]["sha256"]:
        raise ValueError("current acceptance binding failed")
    current = read_json(current_path)
    if current["status"] != "REPAIRED_CANDIDATE_RUNTIME_VERIFIED_NOT_APPLIED":
        raise ValueError("candidate runtime is not accepted")
    data_path = Path(current["data_freeze"]["path"])
    if file_sha(data_path) != current["data_freeze"]["sha256"]:
        raise ValueError("candidate data freeze binding failed")
    check_frozen_data(read_json(data_path))
    for group in ("data_artifacts", "artifacts", "implementation"):
        for fp in current[group].values():
            if any(cheap(Path(fp["path"]))[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
                raise ValueError(f"accepted input changed: {fp['path']}")
    return {"pointer": {**cheap(pointer_path), "sha256": file_sha(pointer_path)},
            "acceptance": {**cheap(current_path), "sha256": file_sha(current_path)},
            "graph": current["data_artifacts"]["knowledge_graph.candidate.json"],
            "details": current["data_artifacts"]["umls_details.sqlite"],
            "counts": current["counts"], "last_deep_data_verification": read_json(data_path)["frozen_at"]}


def write_jsonl(path: Path, rows) -> dict:
    digest, count = hashlib.sha256(), 0
    with path.open("xb") as handle:
        for row in rows:
            data = (compact(row) + "\n").encode("utf-8")
            handle.write(data)
            digest.update(data)
            count += 1
    return {**cheap(path), "rows": count, "sha256": digest.hexdigest()}


def snapshot_code() -> dict:
    result = {}
    for root in (REPO / "neurooracle/src", REPO / "neurooracle/scripts", REPO / "core"):
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(REPO).as_posix()
            if any(part in {"__pycache__", ".venv", "tests", "test"} for part in path.parts) or path.name.startswith("test_"):
                continue
            result[relative] = {**cheap(path), "sha256": file_sha(path)}
    return result


def scan_graph(output: Path, baseline: dict) -> tuple[dict, list[dict]]:
    coverage, view = Coverage(), AdmissionView()
    counts, classes, full_drops = Counter(), Counter(), Counter()
    raw_active, strict_active = set(), set()
    dropped_rows = []
    full_review_edges = 0
    progress(output, "SCAN_GRAPH_COVERAGE_AND_ADMISSION")
    with hashed_reader(Path(baseline["graph"]["path"])) as (reader, digest):
        for kind, record_id, record in walk_graph(reader):
            if kind == "metadata":
                continue
            counts[kind] += 1
            if kind == "node":
                category = node_class(record_id, record)
                classes[category] += 1
                coverage.add("node/all", record)
                coverage.add("node/" + category, record)
                view.add_node(record_id, record)
                if counts[kind] % 250000 == 0:
                    progress(output, "SCAN_NODES", nodes=counts[kind])
            else:
                s, t = record["source_id"], record["target_id"]
                if s not in view.domains or t not in view.domains:
                    raise ValueError("dangling graph edge")
                raw_active.update((s, t))
                reason = mapping_admission_reason(record)
                if reason:
                    full_drops[reason] += 1
                else:
                    full_review_edges += 1
                    strict_active.update((s, t))
                coverage.add("edge/all", record)
                category = "umls_mapping" if is_umls_mapping(record) else "non_umls_mapping"
                coverage.add("edge/" + category, record)
                decision, reason = view.add_edge(record)
                if decision == "admitted":
                    coverage.add("edge/kge_admitted", record)
                elif decision == "admission_excluded":
                    dropped_rows.append({"edge_ordinal": int(record_id), "edge_id": record.get("id"),
                                         "source_id": s, "target_id": t, "relation_type": record["relation_type"],
                                         "review_status": (record.get("metadata") or {}).get("review_status"),
                                         "reason": reason, "audit_ref": (record.get("metadata") or {}).get("audit_ref")})
                if counts[kind] % 500000 == 0:
                    progress(output, "SCAN_EDGES", edges=counts[kind])
        graph_sha = digest.hexdigest()
    if graph_sha != baseline["graph"]["sha256"]:
        raise ValueError("full candidate graph hash differs from accepted snapshot")
    if counts != {"node": baseline["counts"]["nodes"], "edge": baseline["counts"]["edges"]}:
        raise ValueError("source count changed")
    rows = coverage.rows()
    coverage_fp = write_jsonl(output / "METADATA_COVERAGE.jsonl", rows)
    excluded_fp = write_jsonl(output / "EXCLUDED_KGE_MAPPING_EDGES.jsonl", dropped_rows)
    triples_fp = write_jsonl(output / "ADMITTED_KGE_TRIPLES.jsonl", (t.as_tuple() for t in view.admitted))
    old_entities = {e for t in view.legacy for e in (t.source_id, t.target_id)}
    new_entities = {e for t in view.admitted for e in (t.source_id, t.target_id)}
    domains_fp = write_jsonl(output / "ADMITTED_NODE_DOMAINS.jsonl", (
        {"id": node_id, "domain": view.domains[node_id]} for node_id in sorted(new_entities)))
    legacy_summary = summarize_triples(view.legacy)
    prior_kge = read_json(CANDIDATE / "KGE_REPAIRED_INPUTS.json")
    if legacy_summary != prior_kge["all_triples"]:
        raise ValueError("legacy predicates differ from prior actual KGE input digest")
    strict_summary = summarize_triples(view.admitted)
    split_results = {}
    for seed in (0, 42):
        progress(output, "STRICT_VIEW_FIXED_SPLITS", seed=seed)
        train, val, test = split_triples(view.admitted, view.domains, seed=seed)
        train_entities = {e for t in train for e in (t.source_id, t.target_id)}
        train_relations = {t.relation_type for t in train}
        if any(t.source_id not in train_entities or t.target_id not in train_entities or t.relation_type not in train_relations for t in (*val, *test)):
            raise ValueError("strict split is not transductive")
        split_results[str(seed)] = {
            "train": summarize_triples(train), "validation": summarize_triples(val),
            "test": summarize_triples(test), "all_test_validation_vocabulary_in_train": True,
        }
        if len(train) + len(val) + len(test) != len(view.admitted):
            raise ValueError("split loses or duplicates records")
        del train, val, test, train_entities
    result = {"status": "SCANNED_AWAITING_FINAL_VALIDATION", "policy_version": POLICY_VERSION,
              "graph_sha256": graph_sha, "counts": dict(counts), "node_classes": dict(classes),
              "coverage_denominators": dict(coverage.denominators),
              "full_graph_active_nodes": len(raw_active),
              "review_filtered_graph_edges": full_review_edges,
              "review_filtered_graph_active_nodes": len(strict_active),
              "review_filtered_graph_exclusions": dict(full_drops),
              "legacy_kge_triples": legacy_summary, "admitted_kge_triples": strict_summary,
              "legacy_kge_active_nodes": len(old_entities), "admitted_kge_active_nodes": len(new_entities),
              "legacy_drop_reasons": dict(view.legacy_drops), "admission_drop_reasons": dict(view.admission_drops),
              "all_umls_mapping_review_states": dict(view.all_mapping_statuses),
              "admitted_umls_mapping_review_states": dict(view.admitted_mapping_statuses),
              "retained_relations": dict(view.retained_relations), "excluded_relations": dict(view.excluded_relations),
              "admitted_fixed_splits": split_results,
              "artifacts": {"coverage": coverage_fp, "excluded_edges": excluded_fp,
                            "admitted_triples": triples_fp, "admitted_node_domains": domains_fp},
              "formal_files_modified": False, "candidate_files_modified": False,
              "legacy_training_entrypoint_changed": False, "full_training_performed": False,
              "mapping_admission_is_not_full_claim_quality_or_temporal_certification": True}
    del view, old_entities, new_entities, raw_active, strict_active, coverage
    gc.collect()
    progress(output, "ACTUAL_LEGACY_READER_PARITY_CHECK")
    actual, domains = load_triples_from_kg(Path(baseline["graph"]["path"]))
    if summarize_triples(actual) != legacy_summary or len(domains) != baseline["counts"]["nodes"]:
        raise ValueError("current actual legacy reader differs from audit predicate clone")
    result["actual_legacy_reader_parity"] = True
    atomic_json(output / "SCAN_COMPLETE.json", result)
    return result, rows


def code_usage(output: Path, snapshot: dict, rows: list[dict]) -> dict:
    keys = {row["field"].split(".")[-1] for row in rows if row["field"].startswith("metadata.")}
    reads, dynamic, parse_failures = [], [], []
    for relative, fp in snapshot.items():
        if relative in {"neurooracle/src/metadata_field_audit.py", "neurooracle/scripts/audit_umls_experiment_metadata.py"}:
            continue
        try:
            found, indirect = static_key_reads(Path(fp["path"]), keys)
            for row in [*found, *indirect]:
                row["relative_path"] = relative
            reads.extend(found)
            dynamic.extend(indirect)
        except (SyntaxError, UnicodeError) as error:
            parse_failures.append({"path": fp["path"], "error": str(error)})
    return {"read_candidates": write_jsonl(output / "CODE_KEY_READ_CANDIDATES.jsonl", reads),
            "dynamic_reads": write_jsonl(output / "DYNAMIC_KEY_READS.jsonl", dynamic),
            "source_files_scanned": len(snapshot), "parse_failures": parse_failures,
            "limitation": "Dictionary key reads are not proven graph-field lineage. Dynamic/external consumers may be missed. No field is automatically safe to delete."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and (output / "INPUT_SNAPSHOT.json").exists():
        raise ValueError("audit already started; do not overwrite its evidence")
    output.mkdir(parents=True, exist_ok=True)
    baseline, code = check_candidate(), snapshot_code()
    atomic_json(output / "INPUT_SNAPSHOT.json", {"baseline": baseline, "code": code, "started_at": utc_now()})
    result, rows = scan_graph(output, baseline)
    progress(output, "STATIC_CODE_USAGE_INDEX")
    usage = code_usage(output, code, rows)
    atomic_json(output / "CODE_USAGE_SUMMARY.json", usage)
    if check_candidate() != baseline:
        raise ValueError("candidate changed during audit")
    for fp in code.values():
        if any(cheap(Path(fp["path"]))[key] != fp[key] for key in ("path", "bytes", "mtime_ns")):
            raise ValueError(f"source code changed during audit: {fp['path']}")
    progress(output, "SCAN_COMPLETE_AWAITING_FIELD_TRIAGE", triples=result["admitted_kge_triples"]["count"])


if __name__ == "__main__":
    main()
