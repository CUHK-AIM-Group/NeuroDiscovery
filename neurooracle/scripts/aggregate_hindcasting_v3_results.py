"""Aggregate a completed Hindcasting v3 cohort under its locked analysis plan.

The v3 NeuroDiscovery runner and the frozen-baseline evaluator intentionally
write different raw formats.  This read-only post-processing step normalizes
both formats, verifies the complete paired matrix, reconstructs terminal-only
baseline discoveries from scored hypotheses, and implements the analysis plan
stored in ``formal_hindcasting_design_v3.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CANONICAL_NEURODISCOVERY_LABELS = {
    "neurodiscovery",
    "neurodiscovery_dynamic_closed_loop",
}
METRIC_COLUMNS = (
    "unique_future_discoveries",
    "terminal_unique_future_discoveries",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _resolve(path_value: str | Path, anchor: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _exact_sign_flip_greater(differences: Sequence[float]) -> float:
    """Exact one-sided P value for H1: mean paired difference > 0."""

    values = [float(value) for value in differences]
    if not values:
        raise ValueError("exact sign-flip test requires at least one task")
    if len(values) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 tasks")
    observed = fmean(values)
    extreme = 0
    assignments = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        assignments += 1
        permuted = fmean(sign * value for sign, value in zip(signs, values))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / assignments


def _holm_adjust(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda index: float(values[index]))
    adjusted = [1.0] * len(values)
    running = 0.0
    family_size = len(values)
    for rank, index in enumerate(ordered):
        candidate = (family_size - rank) * float(values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def _window_label(window: Mapping[str, Any]) -> str:
    return (
        f"kg{int(window['freeze_year'])}_to_"
        f"{int(window['future_start_year'])}_{int(window['future_end_year'])}"
    )


def _validate_locked_sources(
    formal_root: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _read_json(plan_path)
    expected_plan_hash = str(plan.get("plan_sha256") or "")
    actual_plan_hash = _sha256(plan_path)
    # Execution plans keep their hash in the lock/execution manifest rather
    # than self-embedding it, so only compare when present.
    if expected_plan_hash and actual_plan_hash != expected_plan_hash:
        raise ValueError("execution-plan SHA-256 mismatch")

    protocol_record = plan["protocol"]
    protocol_path = _resolve(protocol_record["path"], plan_path.parent)
    protocol = _read_json(protocol_path)
    if _sha256(protocol_path) != str(protocol_record["sha256"]).upper():
        raise ValueError("protocol SHA-256 mismatch")

    execution_path = formal_root / "execution_manifest.json"
    verification_path = formal_root / "output_verification.json"
    execution = _read_json(execution_path)
    verification = _read_json(verification_path)
    if execution.get("status") != "complete":
        raise ValueError("formal execution manifest is not complete")
    if int(execution.get("llm_api_requests", -1)) != 0:
        raise ValueError("unexpected LLM API requests in deterministic formal run")
    if execution.get("plan_sha256") != actual_plan_hash:
        raise ValueError("formal execution is bound to a different plan")
    if verification.get("status") != "passed":
        raise ValueError("formal output verification did not pass")
    expected_runs = int(plan["matrix"]["total_method_runs"])
    if (
        int(verification.get("expected_cells", -1)) != expected_runs
        or int(verification.get("verified_cells", -1)) != expected_runs
    ):
        raise ValueError("verified formal-cell count does not match the plan")
    if verification.get("protocol_sha256") != protocol_record["sha256"]:
        raise ValueError("output verification is bound to a different protocol")
    if verification.get("cohort_sha256") != plan["cohort"]["sha256"]:
        raise ValueError("output verification is bound to a different cohort")
    if (
        verification.get("source_bundle_manifest_sha256")
        != execution["source_bundle"]["manifest_sha256"]
    ):
        raise ValueError("output verification is bound to a different source bundle")
    return plan, protocol, execution, verification


def _collect_neurodiscovery(
    formal_root: Path,
    windows: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Path]:
    source = formal_root / "neurodiscovery" / "metrics_by_run.csv"
    frame = pd.read_csv(source)
    required = {
        "method",
        "case_study_id",
        "seed",
        "freeze_year",
        "future_start_year",
        "future_end_year",
        "requested_k",
        "executed_hypotheses",
        "generation_failure_slots",
        "unique_primary_discoveries",
        "terminal_unique_primary_discoveries",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"NeuroDiscovery metrics are missing columns: {missing_columns}")

    labels = set(frame["method"].dropna().astype(str))
    if not labels or not labels.issubset(CANONICAL_NEURODISCOVERY_LABELS):
        raise ValueError(f"unexpected NeuroDiscovery labels: {sorted(labels)}")

    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        freeze = int(raw["freeze_year"])
        window = windows.get(freeze)
        if window is None:
            raise ValueError(f"NeuroDiscovery row has an unexpected freeze year: {freeze}")
        start = int(raw["future_start_year"])
        end = int(raw["future_end_year"])
        if start != int(window["future_start_year"]) or end != int(window["future_end_year"]):
            raise ValueError("NeuroDiscovery row has a mismatched temporal window")
        k = int(raw["requested_k"])
        if int(raw["executed_hypotheses"]) != k:
            raise ValueError("NeuroDiscovery did not execute the requested budget")
        run_manifest = (
            formal_root
            / "neurodiscovery"
            / f"seed_{int(raw['seed']):02d}"
            / str(raw["case_study_id"])
            / _window_label(window)
            / "run_manifest.json"
        )
        if not run_manifest.is_file():
            raise FileNotFoundError(run_manifest)
        rows.append(
            {
                "method": "neurodiscovery",
                "case_study_id": str(raw["case_study_id"]),
                "seed": int(raw["seed"]),
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
                "terminal_start_year": int(window["terminal_start_year"]),
                "terminal_end_year": int(window["terminal_end_year"]),
                "k": k,
                "unique_future_discoveries": int(raw["unique_primary_discoveries"]),
                "terminal_unique_future_discoveries": int(
                    raw["terminal_unique_primary_discoveries"]
                ),
                "generation_failure_slots": int(raw["generation_failure_slots"]),
                "source_artifact": str(run_manifest),
            }
        )
    return rows, source


def _nonempty_keys(frame: pd.DataFrame) -> pd.Series:
    values = frame["primary_discovery_key"]
    return values.notna() & values.astype(str).str.strip().ne("")


def _collect_baselines(
    formal_root: Path,
    windows: Mapping[int, Mapping[str, Any]],
    budgets: Sequence[int],
) -> tuple[list[dict[str, Any]], Path, int]:
    evaluation_root = formal_root / "baseline_evaluation"
    manifest_path = evaluation_root / "evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    reconstruction_checks = 0

    for run in manifest.get("runs") or ():
        method = str(run["method"])
        if method not in {"sciagents", "openscholar_rag"}:
            raise ValueError(f"unexpected baseline method: {method}")
        freeze = int(run["freeze_year"])
        window = windows.get(freeze)
        if window is None:
            raise ValueError(f"baseline row has an unexpected freeze year: {freeze}")
        if (
            int(run["future_start_year"]) != int(window["future_start_year"])
            or int(run["future_end_year"]) != int(window["future_end_year"])
        ):
            raise ValueError("baseline row has a mismatched temporal window")
        metrics_path = _resolve(run["metrics_path"], evaluation_root)
        metrics = _read_json(metrics_path)
        scored_path = metrics_path.parent / "scored_hypotheses.csv"
        scored = pd.read_csv(scored_path)
        required = {"rank", "primary_hit", "primary_year", "primary_discovery_key"}
        missing_columns = sorted(required - set(scored.columns))
        if missing_columns:
            raise ValueError(f"{scored_path} is missing columns: {missing_columns}")
        if metrics.get("benchmark_status") != "executable":
            raise ValueError(f"non-executable baseline result: {metrics_path}")
        if int(metrics.get("n_hypotheses", -1)) != len(scored):
            raise ValueError(f"scored-hypothesis count mismatch: {metrics_path}")
        if scored["rank"].astype(int).nunique() != len(scored):
            raise ValueError(f"duplicate baseline ranks: {scored_path}")

        scored = scored.copy()
        scored["rank"] = scored["rank"].astype(int)
        scored["_primary_hit"] = scored["primary_hit"].map(_as_bool)
        scored["_primary_year"] = pd.to_numeric(scored["primary_year"], errors="coerce")
        hit_rows = scored[scored["_primary_hit"]]
        if not hit_rows.empty:
            if hit_rows["_primary_year"].isna().any() or not _nonempty_keys(hit_rows).all():
                raise ValueError(f"primary hit lacks year or discovery key: {scored_path}")
            if (
                hit_rows["_primary_year"].lt(int(window["future_start_year"])).any()
                or hit_rows["_primary_year"].gt(int(window["future_end_year"])).any()
            ):
                raise ValueError(f"primary hit falls outside its future window: {scored_path}")

        topk = metrics.get("topk") or {}
        for k in budgets:
            selected = scored[scored["rank"] <= int(k)]
            hits = selected[selected["_primary_hit"] & _nonempty_keys(selected)]
            reconstructed_full = int(hits["primary_discovery_key"].astype(str).nunique())
            observed = (topk.get(str(k)) or {}).get("observed") or {}
            declared_full = int(observed.get("unique_primary_discoveries", -1))
            if reconstructed_full != declared_full:
                raise ValueError(
                    f"baseline unique-discovery reconstruction mismatch at {metrics_path} K={k}: "
                    f"declared={declared_full} reconstructed={reconstructed_full}"
                )
            terminal = hits[
                hits["_primary_year"].between(
                    int(window["terminal_start_year"]),
                    int(window["terminal_end_year"]),
                    inclusive="both",
                )
            ]
            terminal_unique = int(
                terminal["primary_discovery_key"].astype(str).nunique()
            )
            reconstruction_checks += 1
            rows.append(
                {
                    "method": method,
                    "case_study_id": str(run["case_study_id"]),
                    "seed": int(run["seed"]),
                    "freeze_year": freeze,
                    "future_start_year": int(run["future_start_year"]),
                    "future_end_year": int(run["future_end_year"]),
                    "terminal_start_year": int(window["terminal_start_year"]),
                    "terminal_end_year": int(window["terminal_end_year"]),
                    "k": int(k),
                    "unique_future_discoveries": declared_full,
                    "terminal_unique_future_discoveries": terminal_unique,
                    "generation_failure_slots": 0,
                    "source_artifact": str(metrics_path),
                }
            )
    return rows, manifest_path, reconstruction_checks


def _assert_complete_matrix(
    observations: pd.DataFrame,
    plan: Mapping[str, Any],
) -> dict[str, int]:
    methods = [str(value) for value in plan["matrix"]["methods"]]
    tasks = [str(value) for value in plan["cohort"]["task_ids"]]
    seeds = [int(value) for value in plan["matrix"]["seeds"]]
    freezes = [int(value["freeze_year"]) for value in plan["matrix"]["windows"]]
    budgets = [int(value) for value in plan["matrix"]["experiment_counts"]]
    key_columns = ["method", "case_study_id", "seed", "freeze_year", "k"]
    actual = {
        tuple(value)
        for value in observations[key_columns].itertuples(index=False, name=None)
    }
    expected = set(itertools.product(methods, tasks, seeds, freezes, budgets))
    duplicate_rows = int(observations.duplicated(key_columns, keep=False).sum())
    missing = expected - actual
    extra = actual - expected
    if duplicate_rows or missing or extra or len(observations) != len(expected):
        raise ValueError(
            "incomplete normalized metric matrix: "
            f"rows={len(observations)} expected={len(expected)} "
            f"duplicates={duplicate_rows} missing={list(sorted(missing))[:3]} "
            f"extra={list(sorted(extra))[:3]}"
        )
    method_runs = observations.drop_duplicates(
        ["method", "case_study_id", "seed", "freeze_year"]
    )
    expected_runs = int(plan["matrix"]["total_method_runs"])
    if len(method_runs) != expected_runs:
        raise ValueError("normalized method-run count does not match the plan")
    return {
        "expected_method_runs": expected_runs,
        "observed_method_runs": len(method_runs),
        "expected_metric_rows": len(expected),
        "observed_metric_rows": len(observations),
        "duplicate_metric_rows": duplicate_rows,
        "missing_metric_rows": len(missing),
        "extra_metric_rows": len(extra),
    }


def _balanced_tables(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell = (
        observations.groupby(
            [
                "method",
                "case_study_id",
                "freeze_year",
                "future_start_year",
                "future_end_year",
                "terminal_start_year",
                "terminal_end_year",
                "k",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            unique_future_discoveries_mean=("unique_future_discoveries", "mean"),
            unique_future_discoveries_sd=("unique_future_discoveries", "std"),
            terminal_unique_future_discoveries_mean=(
                "terminal_unique_future_discoveries",
                "mean",
            ),
            terminal_unique_future_discoveries_sd=(
                "terminal_unique_future_discoveries",
                "std",
            ),
            generation_failure_slots_sum=("generation_failure_slots", "sum"),
        )
        .sort_values(["method", "case_study_id", "freeze_year", "k"], kind="stable")
        .reset_index(drop=True)
    )
    task = (
        cell.groupby(["method", "case_study_id", "k"], as_index=False)
        .agg(
            n_windows=("freeze_year", "nunique"),
            unique_future_discoveries=("unique_future_discoveries_mean", "mean"),
            terminal_unique_future_discoveries=(
                "terminal_unique_future_discoveries_mean",
                "mean",
            ),
        )
        .sort_values(["method", "case_study_id", "k"], kind="stable")
        .reset_index(drop=True)
    )
    overall = (
        task.groupby(["method", "k"], as_index=False)
        .agg(
            n_tasks=("case_study_id", "nunique"),
            unique_future_discoveries=("unique_future_discoveries", "mean"),
            terminal_unique_future_discoveries=(
                "terminal_unique_future_discoveries",
                "mean",
            ),
        )
        .sort_values(["method", "k"], kind="stable")
        .reset_index(drop=True)
    )
    return cell, task, overall


def _comparison_family(
    task_summary: pd.DataFrame,
    endpoint: str,
    budgets: Sequence[int],
    baselines: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        for k in budgets:
            block = task_summary[task_summary["k"] == int(k)]
            reference = block[block["method"] == "neurodiscovery"].set_index(
                "case_study_id"
            )[endpoint]
            comparator = block[block["method"] == baseline].set_index(
                "case_study_id"
            )[endpoint]
            if set(reference.index) != set(comparator.index):
                raise ValueError(f"unpaired task-level comparison: {baseline} K={k}")
            tasks = sorted(reference.index)
            differences = [float(reference[task] - comparator[task]) for task in tasks]
            for task, difference in zip(tasks, differences):
                task_rows.append(
                    {
                        "endpoint": endpoint,
                        "comparison_method": baseline,
                        "k": int(k),
                        "case_study_id": task,
                        "neurodiscovery_value": float(reference[task]),
                        "comparison_value": float(comparator[task]),
                        "paired_difference": difference,
                    }
                )
            rows.append(
                {
                    "endpoint": endpoint,
                    "reference_method": "neurodiscovery",
                    "comparison_method": baseline,
                    "k": int(k),
                    "n_tasks": len(tasks),
                    "neurodiscovery_balanced_mean": float(fmean(reference[task] for task in tasks)),
                    "comparison_balanced_mean": float(fmean(comparator[task] for task in tasks)),
                    "mean_paired_task_difference": float(fmean(differences)),
                    "median_paired_task_difference": float(median(differences)),
                    "reference_wins": sum(value > 0 for value in differences),
                    "ties": sum(abs(value) <= 1e-15 for value in differences),
                    "reference_losses": sum(value < 0 for value in differences),
                    "p_one_sided_exact_sign_flip": _exact_sign_flip_greater(differences),
                }
            )
    adjusted = _holm_adjust([float(row["p_one_sided_exact_sign_flip"]) for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["p_holm_family"] = p_holm
        row["holm_family_size"] = len(rows)
        row["reject_fwer_0_05"] = bool(p_holm < 0.05)
    return pd.DataFrame(rows), pd.DataFrame(task_rows)


def _temporal_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    frame = cell_summary.copy()
    frame["temporal_stratum"] = frame["freeze_year"].map(
        lambda year: "2016-2018_retrospective" if int(year) <= 2018 else "2019-2020_locked_retrospective"
    )
    per_task = (
        frame.groupby(["method", "case_study_id", "k", "temporal_stratum"], as_index=False)
        .agg(
            n_windows=("freeze_year", "nunique"),
            unique_future_discoveries=("unique_future_discoveries_mean", "mean"),
            terminal_unique_future_discoveries=(
                "terminal_unique_future_discoveries_mean",
                "mean",
            ),
        )
    )
    return (
        per_task.groupby(["method", "k", "temporal_stratum"], as_index=False)
        .agg(
            n_tasks=("case_study_id", "nunique"),
            windows_per_task=("n_windows", "mean"),
            unique_future_discoveries=("unique_future_discoveries", "mean"),
            terminal_unique_future_discoveries=(
                "terminal_unique_future_discoveries",
                "mean",
            ),
        )
        .sort_values(["method", "k", "temporal_stratum"], kind="stable")
        .reset_index(drop=True)
    )


def _display_method(method: str) -> str:
    return {
        "neurodiscovery": "NeuroDiscovery",
        "sciagents": "SciAgents",
        "openscholar_rag": "OpenScholar-RAG",
    }.get(method, method)


def _markdown_report(
    plan: Mapping[str, Any],
    overall: pd.DataFrame,
    primary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    temporal: pd.DataFrame,
    matrix: Mapping[str, int],
) -> str:
    primary_budgets = [100, 1000]
    lines = [
        "# Hindcasting v3 locked-cohort results",
        "",
        f"Cohort: `{plan['cohort']['cohort_id']}`",
        "",
        (
            f"Formal matrix complete: {matrix['observed_method_runs']}/"
            f"{matrix['expected_method_runs']} method-runs and "
            f"{matrix['observed_metric_rows']}/{matrix['expected_metric_rows']} "
            "method-run-budget observations."
        ),
        "",
        "The experiment is a locked retrospective rolling-window evaluation. It is not an untouched prospective or confirmatory study.",
        "",
        "## Balanced five-window summary",
        "",
        "Values are unique future-supported discoveries. Seeds are averaged within each task-window, windows within task, then tasks equally.",
        "",
        "| Method | Full 5y @100 | Terminal 3y @100 | Full 5y @1000 | Terminal 3y @1000 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in plan["matrix"]["methods"]:
        row100 = overall[(overall["method"] == method) & (overall["k"] == 100)].iloc[0]
        row1000 = overall[(overall["method"] == method) & (overall["k"] == 1000)].iloc[0]
        lines.append(
            f"| {_display_method(method)} | {row100['unique_future_discoveries']:.3f} | "
            f"{row100['terminal_unique_future_discoveries']:.3f} | "
            f"{row1000['unique_future_discoveries']:.3f} | "
            f"{row1000['terminal_unique_future_discoveries']:.3f} |"
        )

    def family_table(title: str, frame: pd.DataFrame) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Comparator | K | NeuroDiscovery | Comparator | Mean task difference | Exact P | Holm P | FWER 0.05 |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in frame.to_dict(orient="records"):
            lines.append(
                f"| {_display_method(str(row['comparison_method']))} | {int(row['k'])} | "
                f"{float(row['neurodiscovery_balanced_mean']):.3f} | "
                f"{float(row['comparison_balanced_mean']):.3f} | "
                f"{float(row['mean_paired_task_difference']):.3f} | "
                f"{float(row['p_one_sided_exact_sign_flip']):.6f} | "
                f"{float(row['p_holm_family']):.6f} | "
                f"{'reject' if bool(row['reject_fwer_0_05']) else 'not reject'} |"
            )

    family_table("Primary family: complete five-year discoveries", primary)
    family_table("Strict sensitivity: terminal three-year discoveries", sensitivity)

    lines.extend(
        [
            "",
            "## Temporal strata (descriptive)",
            "",
            "| Stratum | Method | K | Full 5y | Terminal 3y |",
            "|---|---|---:|---:|---:|",
        ]
    )
    selected = temporal[temporal["k"].isin(primary_budgets)]
    for row in selected.to_dict(orient="records"):
        lines.append(
            f"| {row['temporal_stratum']} | {_display_method(str(row['method']))} | "
            f"{int(row['k'])} | {float(row['unique_future_discoveries']):.3f} | "
            f"{float(row['terminal_unique_future_discoveries']):.3f} |"
        )

    n_tasks = len(plan["cohort"]["task_ids"])
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            f"- The locked inference unit is the task; this cohort has only {n_tasks} tasks. The smallest attainable one-sided exact sign-flip P value is {1 / (2 ** n_tasks):.3f} before multiplicity correction.",
            "- Statistical non-rejection therefore does not erase descriptive effect sizes, but it does preclude a formal superiority claim from this two-task cohort.",
            "- The terminal-three-year endpoint was never exposed to NeuroDiscovery as feedback and is reported as strict sensitivity.",
            "- No LLM API was called in this deterministic formal execution.",
            "- Task-matched random resampling is diagnostic only and is not treated as a fourth method.",
            "",
        ]
    )
    return "\n".join(lines)


def run(formal_root: Path, plan_path: Path, output_dir: Path) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    plan_path = plan_path.resolve()
    output_dir = output_dir.resolve()
    plan, protocol, execution, verification = _validate_locked_sources(formal_root, plan_path)

    methods = [str(value) for value in plan["matrix"]["methods"]]
    if methods != [str(value) for value in protocol["fixed_methods"]["primary"]]:
        raise ValueError("plan methods differ from the locked protocol")
    budgets = [int(value) for value in plan["matrix"]["experiment_counts"]]
    primary_budgets = [int(value) for value in protocol["fixed_outcome_contract"]["primary_summary_budgets"]]
    if not set(primary_budgets).issubset(budgets):
        raise ValueError("primary summary budgets are absent from the plan")
    windows = {
        int(window["freeze_year"]): window for window in plan["matrix"]["windows"]
    }

    nd_rows, nd_source = _collect_neurodiscovery(formal_root, windows)
    baseline_rows, baseline_manifest, reconstruction_checks = _collect_baselines(
        formal_root, windows, budgets
    )
    observations = pd.DataFrame([*nd_rows, *baseline_rows]).sort_values(
        ["method", "case_study_id", "seed", "freeze_year", "k"], kind="stable"
    ).reset_index(drop=True)
    matrix = _assert_complete_matrix(observations, plan)

    expected_seeds = len(plan["matrix"]["seeds"])
    if (observations["generation_failure_slots"] != 0).any():
        raise ValueError("formal matrix contains generation-failure slots")
    cell, task, overall = _balanced_tables(observations)
    if set(cell["n_seeds"].astype(int)) != {expected_seeds}:
        raise ValueError("a task-window cell does not contain every locked seed")
    if set(task["n_windows"].astype(int)) != {len(windows)}:
        raise ValueError("a task summary does not contain every locked window")

    baselines = [method for method in methods if method != "neurodiscovery"]
    primary, primary_task_values = _comparison_family(
        task,
        "unique_future_discoveries",
        primary_budgets,
        baselines,
    )
    sensitivity, sensitivity_task_values = _comparison_family(
        task,
        "terminal_unique_future_discoveries",
        primary_budgets,
        baselines,
    )
    comparison_task_values = pd.concat(
        [primary_task_values, sensitivity_task_values], ignore_index=True
    )
    temporal = _temporal_summary(cell)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "normalized_method_budget_observations.csv": observations,
        "task_window_seed_summary.csv": cell,
        "task_balanced_summary.csv": task,
        "overall_balanced_summary.csv": overall,
        "primary_comparison_family.csv": primary,
        "strict_sensitivity_family.csv": sensitivity,
        "comparison_task_values.csv": comparison_task_values,
        "temporal_strata_summary.csv": temporal,
    }
    for name, frame in tables.items():
        _atomic_csv(output_dir / name, frame)

    report_path = output_dir / "RESULTS.md"
    _atomic_text(
        report_path,
        _markdown_report(plan, overall, primary, sensitivity, temporal, matrix),
    )

    source_paths = {
        "execution_plan": plan_path,
        "protocol": _resolve(plan["protocol"]["path"], plan_path.parent),
        "run_matrix": _resolve(plan["matrix"]["path"], plan_path.parent),
        "execution_manifest": formal_root / "execution_manifest.json",
        "output_verification": formal_root / "output_verification.json",
        "neurodiscovery_metrics": nd_source,
        "baseline_evaluation_manifest": baseline_manifest,
        "analysis_script": Path(__file__).resolve(),
    }
    table_records = {
        name: {
            "path": str(output_dir / name),
            "rows": len(frame),
            "sha256": _sha256(output_dir / name),
        }
        for name, frame in tables.items()
    }
    table_records[report_path.name] = {
        "path": str(report_path),
        "rows": len(report_path.read_text(encoding="utf-8").splitlines()),
        "sha256": _sha256(report_path),
    }
    manifest = {
        "schema_version": "hindcasting-v3-locked-result-summary.v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort_id": plan["cohort"]["cohort_id"],
        "retrospective_interpretation": protocol["retrospective_interpretation"],
        "matrix": matrix,
        "methods": methods,
        "case_study_ids": [str(value) for value in plan["cohort"]["task_ids"]],
        "seeds": [int(value) for value in plan["matrix"]["seeds"]],
        "freeze_years": sorted(windows),
        "experiment_counts": budgets,
        "primary_summary_budgets": primary_budgets,
        "primary_metric": "unique_future_discoveries",
        "strict_sensitivity_metric": "terminal_unique_future_discoveries",
        "balanced_summary_rule": protocol["analysis_plan"]["balanced_summary"],
        "primary_inference_unit": protocol["analysis_plan"]["primary_inference_unit"],
        "primary_test": protocol["analysis_plan"]["primary_test"],
        "primary_multiplicity": protocol["analysis_plan"]["primary_multiplicity"],
        "baseline_full_metric_reconstruction_checks": reconstruction_checks,
        "formal_output_verification": {
            "status": verification["status"],
            "verified_cells": verification["verified_cells"],
        },
        "execution": {
            "status": execution["status"],
            "llm_api_requests": execution["llm_api_requests"],
        },
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "tables": table_records,
        "primary_family": primary.to_dict(orient="records"),
        "strict_sensitivity_family": sensitivity.to_dict(orient="records"),
        "legacy_aggregate_note": (
            "aggregate_hindcasting_method_replicates.py predates v3 and hard-codes "
            "a 17-task/3-budget template; it is not used for v3 inference."
        ),
    }
    manifest_path = output_dir / "summary_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "cohort_id": manifest["cohort_id"],
        **matrix,
        "output_dir": str(output_dir),
        "summary_manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(args.formal_root, args.plan, args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
