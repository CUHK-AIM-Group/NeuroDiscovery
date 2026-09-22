"""Rebase a collected literature queue after a sealed subset was injected.

This utility is deliberately read-only with respect to the formal KG.  It
builds an isolated identity index from the canonical extracted-claim store,
verifies the sealed paper inventory against the original queue, and emits a
new queue containing only papers that are both unreviewed and absent from the
current formal KG.

The original JSONL bytes are preserved for every retained paper.  Search
provenance is not used as Case Study membership evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.src.paper_identity import (  # noqa: E402
    GlobalPaperIdentityIndex,
    paper_identity_aliases,
)


DATA_ROOT = REPO_ROOT / "neurooracle" / "data"
COLLECTION_ROOT = (
    DATA_ROOT
    / "case_study_staging"
    / "kg_v3_sparse_case_studies_100k_20260811"
)
REVIEW_ROOT = (
    COLLECTION_ROOT
    / "claim_extraction_luna_max_24tasks_12p12q_20260811"
)
DEFAULT_QUEUE = COLLECTION_ROOT / "abstracts_ready_for_extraction.jsonl"
DEFAULT_RESULTS = REVIEW_ROOT / "postreview" / "final" / "paper_results.jsonl"
DEFAULT_SUMMARY = REVIEW_ROOT / "postreview" / "final" / "FINAL_SUMMARY.json"
DEFAULT_INJECTION = REVIEW_ROOT / "KG_INJECTION_REPORT.json"
DEFAULT_FORMAL_CLAIMS = DATA_ROOT / "full_v2" / "extracted_claims.jsonl"
DEFAULT_FORMAL_GRAPH = DATA_ROOT / "full_v2" / "knowledge_graph.json"
DEFAULT_CURRENT_STATE = DATA_ROOT / "full_v2" / "CURRENT_STATE.json"
DEFAULT_OUTPUT = COLLECTION_ROOT / "rebase_after_20k_injection_20260814"

SCHEMA_VERSION = "neurooracle.ready_queue_rebase.v1"
TARGET_NET_CLAIM_BEARING_PAPERS = 100_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        result["sha256"] = file_hash(path)
    return result


def iter_jsonl(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSON row at {path}:{line_number}")
            yield line_number, raw, row


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def append_compact_json(handle: Any, value: object) -> None:
    handle.write(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _paper_key(row: dict[str, Any]) -> str:
    return str(row.get("paper_id") or row.get("paper_key") or "").strip()


def load_sealed_results(
    results_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    results: dict[int, dict[str, Any]] = {}
    paper_keys: set[str] = set()
    zero_claim = 0
    claim_bearing = 0
    claims = 0
    for line_number, _raw, row in iter_jsonl(results_path):
        queue_index = int(row.get("queue_index") or 0)
        paper_key = str(row.get("paper_key") or "").strip()
        claim_count = int(row.get("claim_count") or 0)
        is_zero = bool(row.get("zero_claim"))
        if queue_index <= 0 or not paper_key:
            raise RuntimeError(f"invalid sealed result identity at line {line_number}")
        if queue_index in results:
            raise RuntimeError(f"duplicate sealed queue_index: {queue_index}")
        if paper_key in paper_keys:
            raise RuntimeError(f"duplicate sealed paper_key: {paper_key}")
        if is_zero != (claim_count == 0):
            raise RuntimeError(f"zero-claim inconsistency for {paper_key}")
        if row.get("validation_status") != "final_complete":
            raise RuntimeError(f"unsealed paper result: {paper_key}")
        results[queue_index] = row
        paper_keys.add(paper_key)
        zero_claim += int(is_zero)
        claim_bearing += int(not is_zero)
        claims += claim_count
    return results, {
        "papers": len(results),
        "zero_claim_papers": zero_claim,
        "claim_bearing_papers": claim_bearing,
        "claims": claims,
    }


def validate_seals(
    *,
    summary_path: Path,
    injection_path: Path,
    sealed_stats: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    injection = json.loads(injection_path.read_text(encoding="utf-8"))
    if summary.get("complete") is not True:
        raise RuntimeError("final review summary is not complete")
    summary_papers = int(
        summary.get("final_complete_papers")
        or summary.get("target_papers")
        or summary.get("papers")
        or 0
    )
    expected_summary = {
        "papers": summary_papers,
        "zero_claim_papers": int(summary.get("zero_claim_papers") or 0),
        "claim_bearing_papers": summary_papers
        - int(summary.get("zero_claim_papers") or 0),
        "claims": int(
            summary.get("final_claims")
            or summary.get("claims")
            or summary.get("claim_count")
            or 0
        ),
    }
    # Older final summaries expose claim_count under a nested artifact summary.
    if expected_summary["claims"] == 0:
        expected_summary["claims"] = int(injection.get("sealed_claims") or 0)
    if expected_summary != sealed_stats:
        raise RuntimeError(
            f"sealed paper-results statistics differ from FINAL_SUMMARY: "
            f"{sealed_stats!r} != {expected_summary!r}"
        )
    expected_injection = {
        "papers": int(injection.get("sealed_target_papers") or 0),
        "zero_claim_papers": int(injection.get("sealed_zero_claim_papers") or 0),
        "claim_bearing_papers": int(injection.get("sealed_papers_with_claims") or 0),
        "claims": int(injection.get("sealed_claims") or 0),
    }
    if expected_injection != sealed_stats:
        raise RuntimeError(
            f"sealed paper-results statistics differ from injection report: "
            f"{sealed_stats!r} != {expected_injection!r}"
        )
    if injection.get("status") != "injected_and_validated":
        raise RuntimeError("20k campaign was not injected and validated")
    return summary, injection


def run(
    *,
    queue_path: Path,
    results_path: Path,
    summary_path: Path,
    injection_path: Path,
    formal_claims_path: Path,
    formal_graph_path: Path,
    current_state_path: Path,
    output_dir: Path,
    target_net_papers: int,
) -> dict[str, Any]:
    inputs = (
        queue_path,
        results_path,
        summary_path,
        injection_path,
        formal_claims_path,
        formal_graph_path,
        current_state_path,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    formal_before = {
        "knowledge_graph": file_snapshot(formal_graph_path),
        "extracted_claims": file_snapshot(formal_claims_path),
        "current_state": file_snapshot(current_state_path),
    }
    source_files = {
        "ready_queue": file_snapshot(queue_path),
        "paper_results": file_snapshot(results_path),
        "final_summary": file_snapshot(summary_path),
        "injection_report": file_snapshot(injection_path),
    }
    sealed_results, sealed_stats = load_sealed_results(results_path)
    summary, injection = validate_seals(
        summary_path=summary_path,
        injection_path=injection_path,
        sealed_stats=sealed_stats,
    )

    identity_db = output_dir / "formal_identity.sqlite3"
    with GlobalPaperIdentityIndex(identity_db) as identity_index:
        sync_stats = identity_index.sync(
            formal_claim_store=formal_claims_path,
            staging_roots=(),
        )

        temp_dir = Path(tempfile.mkdtemp(prefix="rebase_", dir=output_dir))
        temp_remaining = temp_dir / "remaining_ready_for_extraction.jsonl"
        temp_map = temp_dir / "remaining_queue_map.jsonl"
        temp_zero = temp_dir / "sealed_zero_claim_inventory.jsonl"
        temp_overlap = temp_dir / "formal_overlap_audit.jsonl"

        queue_alias_owner: dict[str, str] = {}
        seen_sealed_indices: set[int] = set()
        queue_papers = 0
        retained = 0
        formal_overlap = 0
        overlap_by_status: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        primary_by_status: dict[str, Counter[str]] = defaultdict(Counter)
        retained_primary_counts: Counter[str] = Counter()
        sealed_primary_total: Counter[str] = Counter()
        sealed_primary_positive: Counter[str] = Counter()
        sealed_primary_zero: Counter[str] = Counter()

        with (
            temp_remaining.open("wb") as remaining_handle,
            temp_map.open("wb") as map_handle,
            temp_zero.open("wb") as zero_handle,
            temp_overlap.open("wb") as overlap_handle,
        ):
            for queue_index, raw, row in iter_jsonl(queue_path):
                queue_papers += 1
                paper_key = _paper_key(row)
                aliases = paper_identity_aliases(row)
                if not paper_key or not aliases:
                    raise RuntimeError(f"queue row {queue_index} lacks paper identity")
                for alias in aliases:
                    prior = queue_alias_owner.setdefault(alias, paper_key)
                    if prior != paper_key:
                        raise RuntimeError(
                            f"within-queue paper identity collision: {alias}: "
                            f"{prior} != {paper_key}"
                        )

                sealed = sealed_results.get(queue_index)
                if sealed is not None:
                    if str(sealed["paper_key"]) != paper_key:
                        raise RuntimeError(
                            f"sealed result/queue identity mismatch at {queue_index}: "
                            f"{sealed['paper_key']} != {paper_key}"
                        )
                    seen_sealed_indices.add(queue_index)
                    status = "sealed_zero_claim" if sealed["zero_claim"] else "sealed_claim_bearing"
                else:
                    status = "unreviewed"

                primary = str(row.get("primary_search_case_study_id") or "unassigned")
                status_counts[status] += 1
                primary_by_status[status][primary] += 1
                if sealed is not None:
                    sealed_primary_total[primary] += 1
                    if sealed["zero_claim"]:
                        sealed_primary_zero[primary] += 1
                        append_compact_json(
                            zero_handle,
                            {
                                "original_queue_index": queue_index,
                                "paper_key": paper_key,
                                "pmid": row.get("pmid"),
                                "doi": row.get("doi"),
                                "title": row.get("title"),
                                "year": row.get("year"),
                                "primary_search_case_study_id": primary,
                                "source_context_sha256": sealed.get("source_context_sha256"),
                                "validation_status": sealed.get("validation_status"),
                                "zero_claim": True,
                            },
                        )
                    else:
                        sealed_primary_positive[primary] += 1

                match = identity_index.match(row)
                if match is not None:
                    formal_overlap += 1
                    overlap_by_status[status] += 1
                    append_compact_json(
                        overlap_handle,
                        {
                            "original_queue_index": queue_index,
                            "paper_key": paper_key,
                            "status": status,
                            "matched_alias": match.alias,
                            "matched_origin": match.origin,
                            "matched_source_path": match.source_path,
                        },
                    )

                if sealed is None and match is None:
                    retained += 1
                    retained_primary_counts[primary] += 1
                    remaining_handle.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    append_compact_json(
                        map_handle,
                        {
                            "rebased_queue_index": retained,
                            "original_queue_index": queue_index,
                            "paper_key": paper_key,
                            "primary_search_case_study_id": primary,
                        },
                    )

                if queue_index % 10_000 == 0:
                    print(
                        f"queue scan: {queue_index:,} rows; retained={retained:,}; "
                        f"formal_overlap={formal_overlap:,}",
                        flush=True,
                    )

        missing_sealed = sorted(set(sealed_results) - seen_sealed_indices)
        if missing_sealed:
            raise RuntimeError(
                f"sealed inventory contains {len(missing_sealed)} queue indices absent from source"
            )

        remaining_path = output_dir / "remaining_ready_for_extraction.jsonl"
        map_path = output_dir / "remaining_queue_map.jsonl"
        zero_path = output_dir / "sealed_zero_claim_inventory.jsonl"
        overlap_path = output_dir / "formal_overlap_audit.jsonl"
        for source, destination in (
            (temp_remaining, remaining_path),
            (temp_map, map_path),
            (temp_zero, zero_path),
            (temp_overlap, overlap_path),
        ):
            os.replace(source, destination)
        shutil.rmtree(temp_dir)

    current_state = json.loads(current_state_path.read_text(encoding="utf-8"))
    formal_after_stats = {
        key: {
            "bytes": Path(value["path"]).stat().st_size,
            "mtime_ns": Path(value["path"]).stat().st_mtime_ns,
        }
        for key, value in formal_before.items()
    }
    for key, before in formal_before.items():
        after = formal_after_stats[key]
        if after["bytes"] != before["bytes"] or after["mtime_ns"] != before["mtime_ns"]:
            raise RuntimeError(f"formal KG changed during rebase audit: {key}")

    observed_rate = sealed_stats["claim_bearing_papers"] / sealed_stats["papers"]
    wilson_low, wilson_high = wilson_interval(
        sealed_stats["claim_bearing_papers"], sealed_stats["papers"]
    )
    strata: dict[str, Any] = {}
    stratified_expected = 0.0
    fallback_strata: list[str] = []
    all_primary = sorted(set(retained_primary_counts) | set(sealed_primary_total))
    for primary in all_primary:
        trials = sealed_primary_total[primary]
        if trials:
            rate = sealed_primary_positive[primary] / trials
        else:
            rate = observed_rate
            fallback_strata.append(primary)
        expected = retained_primary_counts[primary] * rate
        stratified_expected += expected
        strata[primary] = {
            "sealed_papers": trials,
            "sealed_claim_bearing_papers": sealed_primary_positive[primary],
            "sealed_zero_claim_papers": sealed_primary_zero[primary],
            "observed_claim_bearing_rate": rate,
            "remaining_unreviewed_papers": retained_primary_counts[primary],
            "expected_claim_bearing_from_remaining": round(expected, 3),
        }

    expected_from_remaining_overall = retained * observed_rate
    deficit_after_remaining = max(0.0, target_net_papers - stratified_expected)
    extra_search_point = math.ceil(deficit_after_remaining / observed_rate)
    extra_search_conservative_95 = math.ceil(
        max(0.0, target_net_papers - retained * wilson_low) / wilson_low
    )
    extra_search_with_tail_buffer = math.ceil(extra_search_point * 1.10)

    invariants = {
        "queue_papers_equal_100000": queue_papers == 100_000,
        "sealed_papers_equal_20000": sealed_stats["papers"] == 20_000,
        "sealed_inventory_fully_found": len(seen_sealed_indices) == sealed_stats["papers"],
        "partition_complete": sum(status_counts.values()) == queue_papers,
        "retained_are_unreviewed_and_not_formal": retained
        == status_counts["unreviewed"] - overlap_by_status["unreviewed"],
        "claim_bearing_sealed_are_formal": overlap_by_status["sealed_claim_bearing"]
        == sealed_stats["claim_bearing_papers"],
        "zero_claim_sealed_are_not_formal": overlap_by_status["sealed_zero_claim"] == 0,
        "no_unreviewed_formal_overlap": overlap_by_status["unreviewed"] == 0,
        "formal_current_state_matches_injection": int(
            ((current_state.get("formal_kg_statistics") or {}).get("general") or {}).get("papers")
            or 0
        )
        == int(((injection.get("final_coverage") or {}).get("general") or {}).get("papers") or 0),
    }
    if not all(invariants.values()):
        failed = [key for key, value in invariants.items() if not value]
        raise RuntimeError(f"rebase invariants failed: {failed}")

    outputs = {
        "remaining_ready_queue": file_snapshot(remaining_path),
        "remaining_queue_map": file_snapshot(map_path),
        "sealed_zero_claim_inventory": file_snapshot(zero_path),
        "formal_overlap_audit": file_snapshot(overlap_path),
        "formal_identity_index": file_snapshot(identity_db),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "started_at": started_at,
        "status": "complete",
        "mode": "read_formal_kg_write_isolated_rebased_queue",
        "formal_kg_mutated": False,
        "inputs": source_files,
        "formal_baseline": formal_before,
        "formal_after_stats": formal_after_stats,
        "identity_index_sync": sync_stats,
        "source_queue": {
            "papers": queue_papers,
            "unique_identity_aliases": len(queue_alias_owner),
        },
        "sealed_subset": {
            **sealed_stats,
            "observed_claim_bearing_rate": observed_rate,
            "claim_bearing_rate_wilson_95": {
                "low": wilson_low,
                "high": wilson_high,
            },
            "final_summary_sha256": source_files["final_summary"]["sha256"],
            "injection_report_sha256": source_files["injection_report"]["sha256"],
        },
        "partition": dict(sorted(status_counts.items())),
        "formal_overlap": {
            "papers": formal_overlap,
            "by_status": dict(sorted(overlap_by_status.items())),
        },
        "rebased_queue": {
            "papers": retained,
            "batches_of_100": math.ceil(retained / 100),
            "source_rows_preserved_byte_for_byte": True,
            "excluded_sealed_zero_claim_as_reviewed_negative_evidence": sealed_stats[
                "zero_claim_papers"
            ],
        },
        "yield_model": {
            "target_net_new_claim_bearing_papers": target_net_papers,
            "overall_observed_rate": observed_rate,
            "expected_claim_bearing_from_remaining_overall": round(
                expected_from_remaining_overall, 3
            ),
            "expected_claim_bearing_from_remaining_stratified": round(
                stratified_expected, 3
            ),
            "strata_without_sealed_sample_using_overall_rate": fallback_strata,
            "additional_search_candidates_point_estimate": extra_search_point,
            "additional_search_candidates_conservative_wilson_95": extra_search_conservative_95,
            "additional_search_candidates_recommended_10pct_tail_buffer": extra_search_with_tail_buffer,
            "note": (
                "The estimate assumes future candidates retain the sealed sample's "
                "claim-bearing yield. Search-tail drift can lower yield; the 10% buffer "
                "is the operational recommendation, not a statistical guarantee."
            ),
            "by_primary_search_provenance": strata,
        },
        "invariants": invariants,
        "outputs": outputs,
    }
    report["report_sha256"] = canonical_hash(report)
    report_path = output_dir / "REBASE_AUDIT.json"
    write_json(report_path, report)
    seal = {
        "schema_version": f"{SCHEMA_VERSION}.seal",
        "created_at": utc_now(),
        "report_sha256": report["report_sha256"],
        "report_file_sha256": file_hash(report_path),
        "remaining_ready_queue_sha256": outputs["remaining_ready_queue"]["sha256"],
        "remaining_papers": retained,
        "sealed_papers": sealed_stats["papers"],
        "sealed_zero_claim_papers": sealed_stats["zero_claim_papers"],
        "formal_overlap_papers": formal_overlap,
        "formal_kg_mutated": False,
        "complete": True,
    }
    seal["seal_sha256"] = canonical_hash(seal)
    write_json(output_dir / "REBASE_COMPLETE.json", seal)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--paper-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--final-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--injection-report", type=Path, default=DEFAULT_INJECTION)
    parser.add_argument("--formal-claims", type=Path, default=DEFAULT_FORMAL_CLAIMS)
    parser.add_argument("--formal-graph", type=Path, default=DEFAULT_FORMAL_GRAPH)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_CURRENT_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-net-claim-bearing-papers",
        type=int,
        default=TARGET_NET_CLAIM_BEARING_PAPERS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_net_claim_bearing_papers <= 0:
        raise SystemExit("--target-net-claim-bearing-papers must be positive")
    report = run(
        queue_path=args.queue.resolve(),
        results_path=args.paper_results.resolve(),
        summary_path=args.final_summary.resolve(),
        injection_path=args.injection_report.resolve(),
        formal_claims_path=args.formal_claims.resolve(),
        formal_graph_path=args.formal_graph.resolve(),
        current_state_path=args.current_state.resolve(),
        output_dir=args.output_dir.resolve(),
        target_net_papers=args.target_net_claim_bearing_papers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
