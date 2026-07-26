from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_WINDOWS = (
    (2016, 2017, 2021),
    (2017, 2018, 2022),
    (2018, 2019, 2023),
    (2019, 2020, 2024),
    (2020, 2021, 2025),
)
EXPECTED_KS = (10, 20, 50, 100, 200, 500, 1000)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _audit_snapshot(snapshot_root: Path, freeze_year: int, errors: list[str]) -> dict[str, Any]:
    snapshot_dir = snapshot_root / f"kg_{freeze_year}"
    manifest_path = snapshot_dir / "manifest.json"
    claims_path = snapshot_dir / "extracted_claims.jsonl"
    if not manifest_path.is_file() or not claims_path.is_file():
        errors.append(f"missing snapshot artifacts for {freeze_year}: {snapshot_dir}")
        return {}
    manifest = _read_json(manifest_path)
    if int(manifest.get("cutoff_year", -1)) != freeze_year:
        errors.append(f"snapshot {freeze_year} has cutoff {manifest.get('cutoff_year')}")

    claim_count = 0
    max_claim_year: int | None = None
    with claims_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            claim_count += 1
            claim = json.loads(line)
            claim_year = _year((claim.get("source_paper") or {}).get("year") or claim.get("year"))
            if claim_year is None:
                errors.append(f"snapshot {freeze_year} claim line {line_number} has no year")
                continue
            max_claim_year = claim_year if max_claim_year is None else max(max_claim_year, claim_year)
            if claim_year > freeze_year:
                errors.append(
                    f"snapshot {freeze_year} contains future claim {claim.get('id')} from {claim_year}"
                )
    expected_count = int(manifest.get("claims_kept_year_le_cutoff", -1))
    if claim_count != expected_count:
        errors.append(f"snapshot {freeze_year} claim count {claim_count} != manifest {expected_count}")
    return {
        "freeze_year": freeze_year,
        "claim_count": claim_count,
        "max_claim_year": max_claim_year,
        "concepts": manifest.get("output_concepts"),
        "edges": manifest.get("output_edges"),
        "future_claim_edges_removed": manifest.get("removed_future_claim_edges"),
        "future_curated_edges_removed": manifest.get("removed_future_dated_non_claim_edges"),
        "future_curated_nodes_removed": manifest.get("removed_future_dated_curated_nodes"),
    }


def _audit_hypotheses(path: Path, freeze_year: int, errors: list[str]) -> tuple[int, int]:
    payload = _read_json(path)
    hypotheses = payload.get("hypotheses") or []
    dated_edges = 0
    for hypothesis in hypotheses:
        for edge in hypothesis.get("path") or []:
            edge_year = _year((edge.get("source_paper") or {}).get("year"))
            if edge_year is None:
                continue
            dated_edges += 1
            if edge_year > freeze_year:
                errors.append(
                    f"{path}: hypothesis {hypothesis.get('id')} uses path evidence from {edge_year} "
                    f"after freeze {freeze_year}"
                )
    return len(hypotheses), dated_edges


def _audit_window(
    task_dir: Path,
    freeze_year: int,
    future_start: int,
    future_end: int,
    errors: list[str],
) -> dict[str, Any]:
    label = f"kg{freeze_year}_to_{future_start}_{future_end}"
    window_dir = task_dir / label
    hypotheses_path = window_dir / "hypotheses_general_raw.json"
    metrics_dir = window_dir / "general_hindcasting"
    metrics_path = metrics_dir / "metrics.json"
    scored_path = metrics_dir / "scored_hypotheses.csv"
    for required in (hypotheses_path, metrics_path, scored_path):
        if not required.is_file():
            errors.append(f"missing {required}")
            return {"window": label, "complete": False}

    metrics = _read_json(metrics_path)
    observed = (
        int(metrics.get("freeze_year", -1)),
        int(metrics.get("future_start_year", -1)),
        int(metrics.get("future_end_year", -1)),
    )
    if observed != (freeze_year, future_start, future_end):
        errors.append(f"{metrics_path}: window {observed} does not match expected")

    topk = {int(k) for k in (metrics.get("topk") or {})}
    if topk != set(EXPECTED_KS):
        errors.append(f"{metrics_path}: K values {sorted(topk)} != {list(EXPECTED_KS)}")

    n_hypotheses, dated_path_edges = _audit_hypotheses(hypotheses_path, freeze_year, errors)
    hit_count = 0
    with scored_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("any_future_hit") or "").lower() != "true":
                continue
            hit_count += 1
            first_year = _year(row.get("first_future_year"))
            if first_year is None or not (future_start <= first_year <= future_end):
                errors.append(
                    f"{scored_path}: hit {row.get('id')} has first future year {first_year}, "
                    f"expected {future_start}-{future_end}"
                )
    return {
        "window": label,
        "complete": True,
        "hypotheses": n_hypotheses,
        "dated_path_edges": dated_path_edges,
        "future_hits": hit_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete five-year Case Study 3 task-wise run.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    run_manifest_path = args.root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest = _read_json(run_manifest_path)
    manifest_windows = tuple(
        (int(row["freeze_year"]), int(row["future_start_year"]), int(row["future_end_year"]))
        for row in run_manifest.get("windows") or []
    )
    if manifest_windows != EXPECTED_WINDOWS:
        errors.append(f"run manifest windows {manifest_windows} != {EXPECTED_WINDOWS}")

    tasks = list(run_manifest.get("tasks") or [])
    task_runs = {row.get("task"): row for row in run_manifest.get("task_runs") or []}
    if len(task_runs) != len(tasks):
        errors.append(f"completed task runs {len(task_runs)} != configured tasks {len(tasks)}")
    for task, row in task_runs.items():
        if int(row.get("returncode", -1)) != 0:
            errors.append(f"task {task} returned {row.get('returncode')}")

    snapshots = [_audit_snapshot(args.snapshot_root, freeze, errors) for freeze, _, _ in EXPECTED_WINDOWS]
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        task_dir = args.root / task
        windows = [
            _audit_window(task_dir, freeze, start, end, errors)
            for freeze, start, end in EXPECTED_WINDOWS
        ]
        task_summaries.append({"task": task, "windows": windows})

    aggregate_files = (
        "taskwise_metrics_by_k.csv",
        "taskwise_random_distribution.csv",
        "taskwise_recovered_examples.csv",
        "taskwise_lead_time_summary.csv",
        "taskwise_aggregate_p_values.csv",
    )
    missing_aggregates = [name for name in aggregate_files if not (args.root / name).is_file()]
    if missing_aggregates:
        errors.append(f"missing aggregate files: {', '.join(missing_aggregates)}")

    report = {
        "ok": not errors,
        "root": str(args.root),
        "tasks": len(tasks),
        "windows_per_task": len(EXPECTED_WINDOWS),
        "expected_task_windows": len(tasks) * len(EXPECTED_WINDOWS),
        "snapshots": snapshots,
        "task_summaries": task_summaries,
        "errors": errors,
    }
    report_path = args.root / "five_year_run_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "task_summaries"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
