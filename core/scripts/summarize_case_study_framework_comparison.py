"""Build cross-task audit tables for the native framework comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop_specs import (
    EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES,
    EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES,
    TASK_PROTOCOLS,
)
from core.scripts.case_study_framework_runtime import (
    METHODS as FRAMEWORK_METHODS,
    _ensure_directory_with_retry,
    _write_text_with_retry,
)


DEFAULT_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v1")
DEFAULT_OUTPUT_NAME = "framework_comparison_sub2api_gpt55_high"
DEFAULT_TASKS = tuple(task for task in TASK_PROTOCOLS if task != "biomarker_discovery")
BENCHMARK_METHODS = ("random_walk", *FRAMEWORK_METHODS, "neurodiscovery")
SUCCESS_STATUSES = frozenset({"complete", "reused"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _variance(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0


def summarize_status_records(
    records: list[dict[str, Any]],
    *,
    task: str,
) -> pd.DataFrame:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    metric_sources: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        method = str(record.get("method", ""))
        trial = int(record.get("trial", -1))
        if method and trial >= 0:
            key = (method, trial)
            latest[key] = record
            if all(
                field in record
                for field in (
                    "duration_seconds",
                    "valid_anchors",
                    "requested_anchors",
                )
            ):
                metric_sources[key] = record

    rows: list[dict[str, Any]] = []
    methods = sorted({key[0] for key in latest})
    for method in methods:
        group = {
            key: row for key, row in latest.items() if key[0] == method
        }
        successful_keys = [
            key
            for key, row in group.items()
            if row.get("status") in SUCCESS_STATUSES
        ]
        completed = [
            metric_sources[key]
            for key in successful_keys
            if key in metric_sources
        ]
        durations = [float(row["duration_seconds"]) for row in completed]
        valid = [float(row["valid_anchors"]) for row in completed]
        fractions = [
            float(row["valid_anchors"]) / max(1.0, float(row["requested_anchors"]))
            for row in completed
        ]
        rows.append(
            {
                "task": task,
                "method": method,
                "completed_trials": len(successful_keys),
                "runtime_observed_trials": len(completed),
                "reused_trials": sum(
                    row.get("status") == "reused" for row in group.values()
                ),
                "failed_trials": sum(
                    row.get("status") == "failed" for row in group.values()
                ),
                "duration_seconds_mean": float(np.mean(durations)) if durations else np.nan,
                "duration_seconds_variance": _variance(durations),
                "valid_anchors_mean": float(np.mean(valid)) if valid else np.nan,
                "valid_anchors_variance": _variance(valid),
                "valid_anchor_fraction_mean": float(np.mean(fractions)) if fractions else np.nan,
                "valid_anchor_fraction_variance": _variance(fractions),
            }
        )
    return pd.DataFrame(rows)


def _load_table(path: Path, *, task: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"empty framework summary table: {path}")
    frame.insert(0, "task", task)
    return frame


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _external_assignment(task: str) -> str:
    if task in EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES:
        return "required"
    if task in EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES:
        return "registered_exempt"
    raise ValueError(f"task has no registered external-validation assignment: {task}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_task_manifests(
    *,
    task: str,
    task_dir: Path,
    manifest: dict[str, Any],
    benchmark_manifest: dict[str, Any],
) -> tuple[str, int]:
    assignment = _external_assignment(task)
    trials = int(manifest.get("trials") or 0)
    expected_jobs = len(FRAMEWORK_METHODS) * trials
    expected_trial_counts = {
        method: trials for method in BENCHMARK_METHODS
    }

    _require(manifest.get("task") == task, f"comparison task mismatch for {task}")
    _require(manifest.get("status") == "complete", f"comparison incomplete for {task}")
    _require(trials >= 2, f"comparison trials are invalid for {task}: {trials}")
    _require(
        tuple(manifest.get("methods") or ()) == tuple(FRAMEWORK_METHODS),
        f"comparison methods are incomplete or out of order for {task}",
    )
    _require(
        int(manifest.get("completed_jobs") or 0) == expected_jobs,
        f"comparison completed-job count mismatch for {task}",
    )
    _require(
        int(manifest.get("failed_jobs") or 0) == 0,
        f"comparison contains failed jobs for {task}",
    )
    _require(
        manifest.get("external_validation_assignment") == assignment,
        f"external-validation assignment mismatch for {task}",
    )
    _require(
        manifest.get("code_lock_drift_detected") is False,
        f"code-lock drift was recorded for {task}",
    )

    _require(
        benchmark_manifest.get("task") == task,
        f"benchmark task mismatch for {task}",
    )
    _require(
        benchmark_manifest.get("status") == "complete",
        f"benchmark incomplete for {task}",
    )
    _require(
        int(benchmark_manifest.get("trials") or 0) == trials,
        f"benchmark trial count mismatch for {task}",
    )
    _require(
        tuple(benchmark_manifest.get("methods") or ()) == BENCHMARK_METHODS,
        f"benchmark methods are incomplete or out of order for {task}",
    )
    observed_trial_counts = {
        str(method): int(count)
        for method, count in (
            benchmark_manifest.get("method_trial_counts") or {}
        ).items()
    }
    _require(
        observed_trial_counts == expected_trial_counts,
        f"benchmark method/trial coverage mismatch for {task}",
    )
    frozen = benchmark_manifest.get("frozen_discovery") or {}
    _require(
        frozen.get("external_data_read_before_freeze") is False,
        f"rankings were not outcome-blind before freeze for {task}",
    )

    external = benchmark_manifest.get("external")
    external_files = tuple(
        task_dir / "benchmark" / f"external_{suffix}.csv"
        for suffix in (
            "metrics_summary",
            "recall_cost_summary",
        )
    )
    if assignment == "required":
        _require(isinstance(external, dict), f"external audit is missing for {task}")
        _require(
            external.get("loaded_after_freeze") is True,
            f"external outcomes were not loaded after freeze for {task}",
        )
        _require(
            int(external.get("executable_candidates") or 0) > 0,
            f"external validation has no executable candidates for {task}",
        )
        _require(
            all(path.is_file() for path in external_files),
            f"external framework tables are missing for {task}",
        )
    else:
        _require(external is None, f"exempt task has an external audit for {task}")
        _require(
            not any(path.exists() for path in external_files),
            f"exempt task has stale external framework tables for {task}",
        )
    return assignment, trials


def summarize(
    *,
    root: Path,
    output_name: str,
    tasks: list[str],
    allow_incomplete: bool = False,
) -> Path:
    destination = root / output_name / "summary"
    _ensure_directory_with_retry(destination)
    inventories: list[dict[str, Any]] = []
    runtimes: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []
    cost_tables: list[pd.DataFrame] = []
    p_value_tables: list[pd.DataFrame] = []
    incomplete: list[str] = []
    incomplete_details: dict[str, str] = {}
    source_manifests: dict[str, dict[str, Any]] = {}

    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be non-empty and unique")

    for task in tasks:
        task_dir = root / task / output_name
        manifest_path = task_dir / "manifest.json"
        benchmark_dir = task_dir / "benchmark"
        benchmark_manifest_path = benchmark_dir / "run_manifest.json"
        try:
            if not manifest_path.is_file() or not benchmark_manifest_path.is_file():
                raise FileNotFoundError(
                    f"incomplete comparison output for {task}: {task_dir}"
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            benchmark_manifest = json.loads(
                benchmark_manifest_path.read_text(encoding="utf-8")
            )
            assignment, trials = _validate_task_manifests(
                task=task,
                task_dir=task_dir,
                manifest=manifest,
                benchmark_manifest=benchmark_manifest,
            )

            embedded_benchmark = manifest.get("benchmark_manifest") or {}
            _require(
                isinstance(embedded_benchmark, dict)
                and embedded_benchmark.get("status") == "complete",
                f"comparison manifest lacks a complete embedded benchmark for {task}",
            )
            embedded_hash = str(embedded_benchmark.get("manifest_sha256") or "")
            if embedded_hash:
                _require(
                    embedded_hash.lower() == sha256_file(benchmark_manifest_path).lower(),
                    f"embedded benchmark hash mismatch for {task}",
                )

            status_records = read_jsonl(task_dir / "job_status.jsonl")
            runtime = summarize_status_records(status_records, task=task)
            _require(
                tuple(runtime["method"].astype(str)) == tuple(sorted(FRAMEWORK_METHODS)),
                f"runtime method coverage mismatch for {task}",
            )
            _require(
                bool((runtime["completed_trials"] == trials).all()),
                f"runtime successful-trial coverage mismatch for {task}",
            )
            _require(
                bool((runtime["runtime_observed_trials"] == trials).all()),
                f"runtime evidence is missing for one or more trials in {task}",
            )
            _require(
                bool((runtime["failed_trials"] == 0).all()),
                f"latest runtime state contains failures for {task}",
            )

            external = benchmark_manifest.get("external")
            required = assignment == "required"
            inventories.append(
                {
                    "task": task,
                    "external_validation_assignment": assignment,
                    "external_required": required,
                    "candidate_count": int(manifest["candidate_count"]),
                    "anchors_per_trial": int(manifest["n_anchors"]),
                    "methods": len(manifest["methods"]),
                    "trials": trials,
                    "completed_jobs": int(manifest["completed_jobs"]),
                    "failed_jobs": int(manifest["failed_jobs"]),
                    "internal_gt_total": int(benchmark_manifest["internal_gt_total"]),
                    "external_executable": (
                        int(external.get("executable_candidates") or 0)
                        if required and isinstance(external, dict)
                        else np.nan
                    ),
                    "external_validated": (
                        int(external.get("validated_candidates") or 0)
                        if required and isinstance(external, dict)
                        else np.nan
                    ),
                    "rankings_frozen_before_external": (
                        True if required else np.nan
                    ),
                    "external_validation_not_applicable": not required,
                }
            )
            runtimes.append(runtime)

            expected_scopes = (
                ("internal", "external") if required else ("internal",)
            )
            for scope in expected_scopes:
                metrics = _load_table(
                    benchmark_dir / f"{scope}_metrics_summary.csv", task=task
                )
                costs = _load_table(
                    benchmark_dir / f"{scope}_recall_cost_summary.csv", task=task
                )
                _require(
                    set(metrics["scope"].astype(str)) == {scope},
                    f"metric scope mismatch for {task}/{scope}",
                )
                _require(
                    set(costs["scope"].astype(str)) == {scope},
                    f"recall-cost scope mismatch for {task}/{scope}",
                )
                _require(
                    set(metrics["method"].astype(str)) == set(BENCHMARK_METHODS),
                    f"metric method coverage mismatch for {task}/{scope}",
                )
                _require(
                    set(costs["method"].astype(str)) == set(BENCHMARK_METHODS),
                    f"recall-cost method coverage mismatch for {task}/{scope}",
                )
                metric_tables.append(metrics)
                cost_tables.append(costs)

            p_values = _load_table(
                benchmark_dir / "paired_p_values.csv", task=task
            )
            _require(
                set(p_values["scope"].astype(str)) == set(expected_scopes),
                f"paired-p-value scope coverage mismatch for {task}",
            )
            p_value_tables.append(p_values)
            source_manifests[task] = {
                "comparison": _source_descriptor(manifest_path),
                "benchmark": _source_descriptor(benchmark_manifest_path),
                "job_status": _source_descriptor(task_dir / "job_status.jsonl"),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            incomplete.append(task)
            incomplete_details[task] = f"{type(exc).__name__}: {exc}"
            if allow_incomplete:
                continue
            raise

    tables = {
        "task_inventory.csv": pd.DataFrame(inventories),
        "framework_runtime_summary.csv": _concat(runtimes),
        "performance_summary.csv": _concat(metric_tables),
        "recall_cost_summary.csv": _concat(cost_tables),
        "paired_p_values.csv": _concat(p_value_tables),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        path = destination / name
        _write_text_with_retry(path, frame.to_csv(index=False))
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }

    summarizer_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "case-study-framework-summary.v1",
        "created_at": utc_now(),
        "summarizer": _source_descriptor(summarizer_path),
        "source_root": str(root.resolve()),
        "source_output_name": output_name,
        "requested_tasks": tasks,
        "included_tasks": [row["task"] for row in inventories],
        "incomplete_tasks": incomplete,
        "incomplete_details": incomplete_details,
        "complete": not incomplete,
        "framework_methods": list(FRAMEWORK_METHODS),
        "benchmark_methods": list(BENCHMARK_METHODS),
        "external_validation": {
            "required": list(EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES),
            "exempt": list(EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES),
        },
        "source_manifests": source_manifests,
        "artifacts": artifacts,
    }
    path = destination / "summary_manifest.json"
    _write_text_with_retry(
        path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--tasks", nargs="+", choices=[*TASK_PROTOCOLS, "all"], default=list(DEFAULT_TASKS)
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = list(DEFAULT_TASKS) if "all" in args.tasks else list(args.tasks)
    path = summarize(
        root=args.root,
        output_name=args.output_name,
        tasks=tasks,
        allow_incomplete=args.allow_incomplete,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
