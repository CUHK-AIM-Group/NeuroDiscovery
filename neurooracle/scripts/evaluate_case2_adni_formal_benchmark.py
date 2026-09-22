"""Evaluate frozen Case Study 2 generator rankings on the formal 168-candidate endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.scripts.prepare_case2_adni_formal_benchmark import (
    EXPECTED_METHODS,
    load_json,
    sha256_file,
    verify_freeze,
    write_json,
)


HKT = timezone(timedelta(hours=8))
REFERENCE_COLUMNS = (
    "candidate_id",
    "bootstrap_weakest_link_evidence",
    "supplemental_family_fdr_hit",
    "global_fdr_hit",
    "nominal_bootstrap_hit",
)
INVALID_SLOT_PREFIX = "__INVALID_GENERATION_SLOT_"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bool_series(values: pd.Series, name: str) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.casefold()
    _require(normalized.isin({"true", "false"}).all(), f"Invalid Boolean values in {name}")
    return normalized.eq("true").to_numpy(bool)


def normalized_discounted_cumulative_gain(
    ordered_relevance: Sequence[float],
    *,
    k: int | None = None,
) -> float:
    relevance = np.asarray(ordered_relevance, dtype=float)
    relevance = np.where(np.isfinite(relevance), np.maximum(relevance, 0.0), 0.0)
    limit = len(relevance) if k is None else min(len(relevance), max(0, int(k)))
    if limit == 0:
        return float("nan")
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))
    observed = float(np.sum(relevance[:limit] / discounts))
    ideal = np.sort(relevance)[::-1][:limit]
    denominator = float(np.sum(ideal / discounts))
    return observed / denominator if denominator > 0 else float("nan")


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float(
        (ranks[y].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranked = y[order]
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def first_hit_rank(labels: Sequence[bool]) -> int | None:
    positions = np.flatnonzero(np.asarray(labels, dtype=bool))
    return int(positions[0] + 1) if len(positions) else None


def validate_reference(reference: pd.DataFrame) -> pd.DataFrame:
    _require(tuple(reference.columns) == REFERENCE_COLUMNS, "Evaluator reference schema changed")
    _require(len(reference) == 168, "Evaluator reference must contain 168 candidates")
    _require(reference["candidate_id"].is_unique, "Evaluator candidate IDs are not unique")
    relevance = pd.to_numeric(reference["bootstrap_weakest_link_evidence"], errors="raise")
    _require(np.isfinite(relevance).all() and relevance.ge(0).all(), "Invalid evidence relevance")
    family = _bool_series(reference["supplemental_family_fdr_hit"], "supplemental_family_fdr_hit")
    global_hits = _bool_series(reference["global_fdr_hit"], "global_fdr_hit")
    nominal = _bool_series(reference["nominal_bootstrap_hit"], "nominal_bootstrap_hit")
    _require(int(family.sum()) == 6, "Supplemental family-FDR label count changed")
    _require(int(global_hits.sum()) == 0, "Global-FDR endpoint is no longer structurally empty")
    _require(int(nominal.sum()) == 15, "Nominal label count changed")
    return reference.assign(
        bootstrap_weakest_link_evidence=relevance.to_numpy(float),
        supplemental_family_fdr_hit=family,
        global_fdr_hit=global_hits,
        nominal_bootstrap_hit=nominal,
    )


def evaluate_order(
    candidate_ids: Sequence[str],
    reference: pd.DataFrame,
    *,
    budgets: Iterable[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    indexed = reference.set_index("candidate_id", drop=False)
    order = list(map(str, candidate_ids))
    sentinels = [value for value in order if value.startswith(INVALID_SLOT_PREFIX)]
    real_ids = [value for value in order if not value.startswith(INVALID_SLOT_PREFIX)]
    _require(len(sentinels) <= 80, "More than 80 failed generation slots")
    _require(len(set(order)) == len(order), "Ranking stream contains duplicate IDs or sentinels")
    _require(len(real_ids) == len(indexed), "Ranking stream does not contain all 168 real candidates")
    _require(set(real_ids) == set(indexed.index), "Ranking stream differs from the frozen candidate universe")
    _require(len(order) == 168 + len(sentinels), "Ranking stream length does not match failed slots")
    relevance_lookup = indexed["bootstrap_weakest_link_evidence"].to_dict()
    family_lookup = indexed["supplemental_family_fdr_hit"].to_dict()
    global_lookup = indexed["global_fdr_hit"].to_dict()
    nominal_lookup = indexed["nominal_bootstrap_hit"].to_dict()
    relevance = np.asarray([float(relevance_lookup.get(value, 0.0)) for value in order], dtype=float)
    family = np.asarray([bool(family_lookup.get(value, False)) for value in order], dtype=bool)
    global_hits = np.asarray([bool(global_lookup.get(value, False)) for value in order], dtype=bool)
    nominal = np.asarray([bool(nominal_lookup.get(value, False)) for value in order], dtype=bool)
    rank_scores = np.arange(len(order), 0, -1, dtype=float)
    summary = {
        "candidate_count": len(real_ids),
        "failed_generation_slots": len(sentinels),
        "experiment_stream_length": len(order),
        "bootstrap_weakest_link_evidence_ndcg": normalized_discounted_cumulative_gain(relevance),
        "supplemental_family_fdr_positive_count": int(family.sum()),
        "supplemental_family_fdr_auroc": binary_auroc(family, rank_scores),
        "supplemental_family_fdr_auprc": average_precision(family, rank_scores),
        "first_supplemental_family_fdr_hit_rank": first_hit_rank(family),
        "nominal_bootstrap_positive_count": int(nominal.sum()),
        "global_fdr_positive_count": int(global_hits.sum()),
        "global_fdr_auroc": binary_auroc(global_hits, rank_scores),
        "global_fdr_auprc": average_precision(global_hits, rank_scores),
        "global_fdr_metrics_identifiable": bool(global_hits.any() and (~global_hits).any()),
    }
    budget_rows: list[dict[str, Any]] = []
    for raw_budget in sorted(set(map(int, budgets))):
        k = min(raw_budget, len(order))
        _require(k > 0, "Budgets must be positive")
        family_hits = int(family[:k].sum())
        budget_rows.append(
            {
                "k": k,
                "bootstrap_weakest_link_evidence_ndcg_at_k": normalized_discounted_cumulative_gain(relevance, k=k),
                "supplemental_family_fdr_hits_at_k": family_hits,
                "supplemental_family_fdr_recall_at_k": family_hits / int(family.sum()),
                "nominal_bootstrap_hits_at_k_exploratory": int(nominal[:k].sum()),
                "global_fdr_hits_at_k": int(global_hits[:k].sum()),
                "global_fdr_recall_at_k": float("nan"),
            }
        )
    return summary, budget_rows


def exact_paired_sign_flip_greater(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = float(values.mean())
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        signs = np.fromiter(
            (1.0 if mask & (1 << index) else -1.0 for index in range(len(values))),
            dtype=float,
            count=len(values),
        )
        exceed += float((values * signs).mean()) >= observed - 1e-15
    return float(exceed / total)


def percentile_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(int(resamples), len(array)))]
    return tuple(map(float, np.percentile(samples.mean(axis=1), [2.5, 97.5])))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(values[finite], kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_primary_comparisons(
    trial_summary: pd.DataFrame,
    *,
    metric: str,
    target: str,
    baselines: Sequence[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    target_frame = trial_summary.loc[
        trial_summary["method"].eq(target), ["trial", metric]
    ].rename(columns={metric: "target_value"})
    rows: list[dict[str, Any]] = []
    for offset, baseline in enumerate(baselines):
        baseline_frame = trial_summary.loc[
            trial_summary["method"].eq(baseline), ["trial", metric]
        ].rename(columns={metric: "baseline_value"})
        paired = target_frame.merge(baseline_frame, on="trial", validate="one_to_one")
        _require(len(paired) == 10, f"Expected ten paired trials for {baseline}")
        differences = paired["target_value"].to_numpy(float) - paired["baseline_value"].to_numpy(float)
        ci_low, ci_high = percentile_bootstrap_mean_ci(
            differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + offset,
        )
        rows.append(
            {
                "metric": metric,
                "target": target,
                "baseline": baseline,
                "n_pairs": len(differences),
                "target_mean": float(paired["target_value"].mean()),
                "baseline_mean": float(paired["baseline_value"].mean()),
                "paired_mean_difference": float(differences.mean()),
                "paired_difference_ci95_low": ci_low,
                "paired_difference_ci95_high": ci_high,
                "p_one_sided_exact_sign_flip": exact_paired_sign_flip_greater(differences),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm_six_baselines"] = holm_adjust(frame["p_one_sided_exact_sign_flip"].to_numpy(float))
    frame["supplemental_superiority_criterion_met"] = (
        frame["paired_mean_difference"].gt(0)
        & frame["paired_difference_ci95_low"].gt(0)
        & frame["p_holm_six_baselines"].lt(0.05)
    )
    return frame


def aggregate_trials(trial_summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "bootstrap_weakest_link_evidence_ndcg",
        "supplemental_family_fdr_auroc",
        "supplemental_family_fdr_auprc",
        "first_supplemental_family_fdr_hit_rank",
    ]
    rows: list[dict[str, Any]] = []
    for method, group in trial_summary.groupby("method", sort=False):
        _require(len(group) == 10, f"Expected ten trials for {method}")
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_finite_trials": len(values),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "trial_percentile_2_5": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
                    "trial_percentile_97_5": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def load_frozen_rankings(benchmark_root: Path) -> dict[tuple[str, int], list[str]]:
    ranking_lock_path = benchmark_root / "rankings" / "RANKINGS.lock.json"
    lock = load_json(ranking_lock_path)
    _require(lock.get("status") == "locked_complete_70_rankings", "Ranking set is not complete")
    _require(lock.get("ranking_count") == 70, "Ranking count is not 70")
    orders: dict[tuple[str, int], list[str]] = {}
    for record in lock["rankings"]:
        method = str(record["method"])
        trial = int(record["trial"])
        relative = Path(record["relative_path"])
        path = benchmark_root / relative
        _require(sha256_file(path) == record["sha256"], f"Frozen ranking changed: {method} trial {trial}")
        frame = pd.read_csv(path)
        _require(tuple(frame.columns) == ("rank", "candidate_id"), f"Ranking schema changed: {path}")
        frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
        _require(frame["rank"].tolist() == list(range(1, len(frame) + 1)), f"Rank values changed: {path}")
        key = (method, trial)
        _require(key not in orders, f"Duplicate ranking: {key}")
        orders[key] = frame["candidate_id"].astype(str).tolist()
    expected = set(itertools.product(EXPECTED_METHODS, range(10)))
    _require(set(orders) == expected, "Ranking method/trial grid is incomplete")
    return orders


def evaluate_benchmark(benchmark_root: Path, *, fast_large_files: bool = False) -> dict[str, Any]:
    verification = verify_freeze(benchmark_root, fast_large_files=fast_large_files)
    config = load_json(benchmark_root / "PROTOCOL.json")
    reference = validate_reference(pd.read_csv(benchmark_root / "evaluator_only" / "REFERENCE_LABELS.csv"))
    orders = load_frozen_rankings(benchmark_root)
    output_root = benchmark_root / "evaluation"
    if output_root.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_root}")
    output_root.mkdir(parents=True)
    trial_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for (method, trial), order in sorted(orders.items()):
        summary, budgets = evaluate_order(order, reference, budgets=config["evaluation"]["budgets"])
        trial_rows.append({"method": method, "trial": trial, **summary})
        budget_rows.extend({"method": method, "trial": trial, **row} for row in budgets)
        indexed = reference.set_index("candidate_id")
        for rank, candidate_id in enumerate(order, start=1):
            invalid = candidate_id.startswith(INVALID_SLOT_PREFIX)
            row = None if invalid else indexed.loc[candidate_id]
            ranking_rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "failed_generation_slot": invalid,
                    "bootstrap_weakest_link_evidence": 0.0 if invalid else row["bootstrap_weakest_link_evidence"],
                    "supplemental_family_fdr_hit": False if invalid else row["supplemental_family_fdr_hit"],
                    "global_fdr_hit": False if invalid else row["global_fdr_hit"],
                    "nominal_bootstrap_hit": False if invalid else row["nominal_bootstrap_hit"],
                }
            )
    trial_summary = pd.DataFrame(trial_rows)
    budget_summary = pd.DataFrame(budget_rows)
    ranked_labels = pd.DataFrame(ranking_rows)
    aggregate = aggregate_trials(trial_summary)
    comparison_config = config["evaluation"]["comparison"]
    comparisons = paired_primary_comparisons(
        trial_summary,
        metric=config["evaluation"]["primary_metric"]["name"],
        target=config["methods"]["target"],
        baselines=config["methods"]["baselines"],
        bootstrap_resamples=comparison_config["bootstrap_resamples"],
        bootstrap_seed=comparison_config["bootstrap_seed"],
    )
    files = {
        "TRIAL_METRICS.csv": trial_summary,
        "BUDGET_METRICS.csv": budget_summary,
        "RANKED_REFERENCE_LABELS.csv": ranked_labels,
        "AGGREGATE_METRICS.csv": aggregate,
        "PAIRED_PRIMARY_COMPARISONS.csv": comparisons,
    }
    for name, frame in files.items():
        frame.to_csv(output_root / name, index=False)
    all_six = bool(comparisons["supplemental_superiority_criterion_met"].all())
    metadata = {
        "schema_version": "neurooracle.case2_formal_benchmark_evaluation.v1",
        "benchmark_id": config["benchmark_id"],
        "freeze_id": verification["freeze_id"],
        "generated_at_hkt": datetime.now(HKT).isoformat(timespec="seconds"),
        "analysis_role": config["analysis_role"],
        "method_count": 7,
        "trials_per_method": 10,
        "candidate_count": 168,
        "primary_metric": config["evaluation"]["primary_metric"],
        "all_six_supplemental_superiority_criteria_met": all_six,
        "clear_sota_claim_allowed": False,
        "clear_sota_claim_forbidden_reason": "The generator-comparison design was frozen after formal results had already been accessed.",
        "global_binary_metrics_identifiable": False,
        "external_validation_used": False,
    }
    write_json(output_root / "EVALUATION_METADATA.json", metadata)
    output_pins = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    result_lock = {
        "schema_version": "neurooracle.case2_formal_benchmark_result_lock.v1",
        "benchmark_id": config["benchmark_id"],
        "freeze_id": verification["freeze_id"],
        "status": "locked_complete_evaluation",
        "post_result_design_freeze": True,
        "files": output_pins,
        "result_id": hashlib.sha256(json.dumps(output_pins, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    write_json(output_root / "EVALUATION_RESULTS.lock.json", result_lock)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--fast-large-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_benchmark(args.benchmark_root, fast_large_files=args.fast_large_files)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
