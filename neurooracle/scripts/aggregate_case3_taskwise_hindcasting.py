from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from neurooracle.src.case3_subtasks import CASE3_SUBTASK_BY_NAME


DEFAULT_ROOT = Path(
    "neurooracle/data/experiments/case3/case3_taskwise_phase3_seed10_20260708"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _window_label(manifest: dict[str, Any]) -> str:
    return f"{manifest['future_start_year']}-{manifest['future_end_year']}"


def _subtask_fields(name: str) -> dict[str, str]:
    subtask = CASE3_SUBTASK_BY_NAME.get(name)
    return {
        "subtask_kind": subtask.kind if subtask else "unknown",
        "signature": subtask.signature if subtask else "",
    }


def _empirical_p_ge(values: list[int], observed: int) -> float | None:
    if not values:
        return None
    return (1 + sum(1 for value in values if value >= observed)) / (len(values) + 1)


def collect_window_outputs(root: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metrics_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    recovered_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []

    for metrics_path in sorted(root.glob("*/kg*_to_*_*/general_hindcasting/metrics_by_k.csv")):
        task = metrics_path.relative_to(root).parts[0]
        out_dir = metrics_path.parent
        manifest_path = out_dir / "metrics.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        freeze_year = int(manifest["freeze_year"])
        future_window = _window_label(manifest)
        future_stats = manifest.get("future_stats") or {}
        metric_rows_for_window: list[dict[str, Any]] = []

        for row in _read_csv(metrics_path):
            random_mean = _float(row.get("random_any_future_hits_mean"))
            observed = _int(row.get("any_future_hits"))
            metric_row = {
                "task": task,
                **_subtask_fields(task),
                "freeze_year": freeze_year,
                "future_window": future_window,
                "n_hypotheses": manifest["n_hypotheses"],
                "future_claims_total": future_stats.get("future_claims_total", 0),
                "future_evaluable_claims": future_stats.get("future_evaluable_claims", 0),
                "future_unique_pairs": future_stats.get("future_unique_pairs", 0),
                "k": _int(row.get("k")),
                "n": _int(row.get("n")),
                "endpoint_hits": _int(row.get("endpoint_hits")),
                "any_future_hits": observed,
                "any_future_hit_rate": _float(row.get("any_future_hit_rate")),
                "random_any_future_hits_mean": random_mean,
                "random_any_future_hits_sd": _float(row.get("random_any_future_hits_sd")),
                "random_any_future_hits_ci95_low": _float(row.get("random_any_future_hits_ci95_low")),
                "random_any_future_hits_ci95_high": _float(row.get("random_any_future_hits_ci95_high")),
                "p_any_future_hits_ge_observed": row.get("p_any_future_hits_ge_observed"),
                "lift_vs_random_any": (observed / random_mean) if random_mean else None,
                "mean_lead_time": row.get("mean_lead_time"),
                "random_endpoint_hits_mean": row.get("random_endpoint_hits_mean"),
                "p_endpoint_hits_ge_observed": row.get("p_endpoint_hits_ge_observed"),
            }
            metrics_rows.append(metric_row)
            metric_rows_for_window.append(metric_row)

        random_path = out_dir / "random_trials.csv"
        if random_path.is_file():
            budgets_by_n: dict[int, list[int]] = defaultdict(list)
            for metric_row in metric_rows_for_window:
                budgets_by_n[int(metric_row["n"])].append(int(metric_row["k"]))
            occurrence_by_trial_n: dict[tuple[int, int], int] = defaultdict(int)
            for row in _read_csv(random_path):
                actual_n = _int(row.get("k"))
                candidate_budgets = sorted(budgets_by_n.get(actual_n) or [actual_n])
                occurrence_key = (_int(row.get("trial")), actual_n)
                occurrence = occurrence_by_trial_n[occurrence_key]
                budget_k = candidate_budgets[min(occurrence, len(candidate_budgets) - 1)]
                occurrence_by_trial_n[occurrence_key] += 1
                random_rows.append({
                    "task": task,
                    **_subtask_fields(task),
                    "freeze_year": freeze_year,
                    "future_window": future_window,
                    "k": budget_k,
                    "n": actual_n,
                    "trial": _int(row.get("trial")),
                    "any_future_hits": _int(row.get("any_future_hits")),
                    "endpoint_hits": _int(row.get("endpoint_hits")),
                })

        recovered_path = out_dir / "recovered_examples.csv"
        if recovered_path.is_file():
            for row in _read_csv(recovered_path):
                row = dict(row)
                row.update({
                    "task": task,
                    **_subtask_fields(task),
                    "freeze_year": freeze_year,
                    "future_window": future_window,
                })
                recovered_rows.append(row)

        scored_path = out_dir / "scored_hypotheses.csv"
        if scored_path.is_file():
            scored = _read_csv(scored_path)
            for k in sorted({m["k"] for m in metrics_rows if m["task"] == task and m["freeze_year"] == freeze_year}):
                top = scored[: int(k)]
                lead_times = [_int(row.get("lead_time")) for row in top if str(row.get("any_future_hit")).lower() == "true"]
                lead_rows.append({
                    "task": task,
                    **_subtask_fields(task),
                    "freeze_year": freeze_year,
                    "future_window": future_window,
                    "k": int(k),
                    "n_hits": len(lead_times),
                    "mean_lead_time": mean(lead_times) if lead_times else None,
                    "median_lead_time": median(lead_times) if lead_times else None,
                    "min_lead_time": min(lead_times) if lead_times else None,
                    "max_lead_time": max(lead_times) if lead_times else None,
                })

    return metrics_rows, random_rows, recovered_rows, lead_rows


def aggregate_p_values(metrics_rows: list[dict[str, Any]], random_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed_by_key: dict[tuple[str, int], int] = defaultdict(int)
    random_by_window: dict[tuple[str, int, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    random_by_key_trial: dict[tuple[str, int, int], int] = defaultdict(int)
    random_trial_counts: dict[tuple[str, int], int] = defaultdict(int)

    for row in random_rows:
        random_by_window[(
            str(row["task"]),
            int(row["freeze_year"]),
            str(row["future_window"]),
            int(row["k"]),
        )].setdefault(int(row["trial"]), row)

    for row in metrics_rows:
        task = str(row["task"])
        budget_k = int(row["k"])
        sample_n = int(row["n"])
        freeze_year = int(row["freeze_year"])
        future_window = str(row["future_window"])
        observed_by_key[(task, budget_k)] += int(row["any_future_hits"])
        observed_by_key[("overall", budget_k)] += int(row["any_future_hits"])
        random_window_rows = list(random_by_window.get((task, freeze_year, future_window, sample_n), {}).values())
        if not random_window_rows:
            random_window_rows = list(random_by_window.get((task, freeze_year, future_window, budget_k), {}).values())
        for random_row in random_window_rows:
            trial = int(random_row["trial"])
            random_by_key_trial[(task, budget_k, trial)] += int(random_row["any_future_hits"])
            random_by_key_trial[("overall", budget_k, trial)] += int(random_row["any_future_hits"])
            random_trial_counts[(task, budget_k)] = max(random_trial_counts[(task, budget_k)], trial)
            random_trial_counts[("overall", budget_k)] = max(random_trial_counts[("overall", budget_k)], trial)

    out: list[dict[str, Any]] = []
    for task, k in sorted(observed_by_key):
        values = [
            value
            for (task2, k2, _trial), value in random_by_key_trial.items()
            if task2 == task and k2 == k
        ]
        if not values:
            continue
        observed = observed_by_key[(task, k)]
        rand_mean = mean(values)
        out.append({
            "task": task,
            **_subtask_fields(task),
            "k": k,
            "observed_any_future_hits": observed,
            "random_any_future_hits_mean": rand_mean,
            "random_any_future_hits_sd": stdev(values) if len(values) > 1 else 0.0,
            "p_any_future_hits_ge_observed": _empirical_p_ge(values, observed),
            "lift_vs_random_any": (observed / rand_mean) if rand_mean else None,
            "random_trials": random_trial_counts.get((task, k), len(values)),
        })
    return out


def build_coverage_rows(root: Path, metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    expected = manifest.get("subtasks") or []
    if not expected:
        names = sorted({str(row["task"]) for row in metrics_rows})
        expected = [CASE3_SUBTASK_BY_NAME[name].to_dict() for name in names if name in CASE3_SUBTASK_BY_NAME]

    runs = manifest.get("subtask_runs") or manifest.get("task_runs") or []
    run_by_name = {str(row.get("subtask") or row.get("task")): row for row in runs}
    windows_by_name: dict[str, set[int]] = defaultdict(set)
    for row in metrics_rows:
        windows_by_name[str(row["task"])].add(int(row["freeze_year"]))
    expected_windows = len(manifest.get("windows") or [])

    rows: list[dict[str, Any]] = []
    for item in expected:
        name = str(item["name"])
        run = run_by_name.get(name, {})
        completed_windows = len(windows_by_name.get(name, set()))
        if run.get("status") == "failed" or int(run.get("returncode", 0) or 0) != 0:
            status = "failed"
        elif expected_windows and completed_windows >= expected_windows:
            status = "completed"
        elif completed_windows:
            status = "partial"
        else:
            status = str(run.get("status") or "missing")
        rows.append({
            "task": name,
            "subtask_kind": item.get("kind") or _subtask_fields(name)["subtask_kind"],
            "signature": item.get("signature") or _subtask_fields(name)["signature"],
            "status": status,
            "completed_windows": completed_windows,
            "expected_windows": expected_windows,
            "returncode": run.get("returncode"),
            "log": run.get("log", ""),
            "output": run.get("output", str(root / name)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Case Study 3 task-wise hindcasting tables.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    metrics_rows, random_rows, recovered_rows, lead_rows = collect_window_outputs(args.root)
    aggregate_rows = aggregate_p_values(metrics_rows, random_rows)
    coverage_rows = build_coverage_rows(args.root, metrics_rows)

    _write_csv(
        args.root / "taskwise_metrics_by_k.csv",
        metrics_rows,
        [
            "task", "subtask_kind", "signature", "freeze_year", "future_window", "n_hypotheses",
            "future_claims_total", "future_evaluable_claims", "future_unique_pairs",
            "k", "n", "endpoint_hits", "any_future_hits", "any_future_hit_rate",
            "random_any_future_hits_mean", "random_any_future_hits_sd",
            "random_any_future_hits_ci95_low", "random_any_future_hits_ci95_high",
            "p_any_future_hits_ge_observed", "lift_vs_random_any", "mean_lead_time",
            "random_endpoint_hits_mean", "p_endpoint_hits_ge_observed",
        ],
    )
    _write_csv(
        args.root / "taskwise_random_distribution.csv",
        random_rows,
        [
            "task", "subtask_kind", "signature", "freeze_year", "future_window",
            "k", "n", "trial", "any_future_hits", "endpoint_hits",
        ],
    )
    _write_csv(
        args.root / "taskwise_recovered_examples.csv",
        recovered_rows,
        [
            "task", "subtask_kind", "signature", "freeze_year", "future_window",
            "rank", "id", "hypothesis_type",
            "task_name", "signature", "source_name", "target_name", "composite_score",
            "endpoint_hit", "any_future_hit", "first_future_year", "lead_time",
            "support_kind", "support_pmid", "support_doi", "support_title",
            "support_journal", "support_year", "support_predicate",
            "support_subject_name", "support_object_name", "support_raw_text",
        ],
    )
    _write_csv(
        args.root / "taskwise_lead_time_summary.csv",
        lead_rows,
        [
            "task", "subtask_kind", "signature", "freeze_year", "future_window", "k", "n_hits",
            "mean_lead_time", "median_lead_time", "min_lead_time", "max_lead_time",
        ],
    )
    _write_csv(
        args.root / "taskwise_aggregate_p_values.csv",
        aggregate_rows,
        [
            "task", "subtask_kind", "signature", "k", "observed_any_future_hits", "random_any_future_hits_mean",
            "random_any_future_hits_sd", "p_any_future_hits_ge_observed",
            "lift_vs_random_any", "random_trials",
        ],
    )
    _write_csv(
        args.root / "taskwise_coverage.csv",
        coverage_rows,
        [
            "task", "subtask_kind", "signature", "status", "completed_windows",
            "expected_windows", "returncode", "log", "output",
        ],
    )
    summary = {
        "root": str(args.root),
        "expected_subtasks": len(coverage_rows),
        "completed_subtasks": sum(row["status"] == "completed" for row in coverage_rows),
        "partial_subtasks": [row["task"] for row in coverage_rows if row["status"] == "partial"],
        "failed_subtasks": [row["task"] for row in coverage_rows if row["status"] == "failed"],
        "planned_subtasks": [row["task"] for row in coverage_rows if row["status"] == "planned"],
        "missing_subtasks": [row["task"] for row in coverage_rows if row["status"] == "missing"],
        "metrics_rows": len(metrics_rows),
        "random_rows": len(random_rows),
        "recovered_rows": len(recovered_rows),
        "lead_rows": len(lead_rows),
        "aggregate_rows": len(aggregate_rows),
    }
    (args.root / "taskwise_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        **summary,
    }, indent=2))


if __name__ == "__main__":
    main()
