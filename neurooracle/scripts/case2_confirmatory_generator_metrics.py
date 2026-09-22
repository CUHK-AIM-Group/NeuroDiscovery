"""Prespecified ranking metrics for the formal Case Study 2 confirmation."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def normalized_recall_auc(
    labels: np.ndarray,
    budgets: Iterable[int],
) -> float:
    labels = np.asarray(labels, dtype=bool)
    total = int(labels.sum())
    if total == 0:
        return float("nan")
    n = len(labels)
    unique = sorted({int(value) for value in budgets if 0 < int(value) <= n})
    x = np.asarray([0.0, *(value / n for value in unique)], dtype=float)
    y = np.asarray(
        [0.0, *(labels[:value].sum() / total for value in unique)], dtype=float
    )
    return float(np.trapezoid(y, x))


def normalized_discounted_cumulative_gain(
    relevance: np.ndarray,
    *,
    k: int | None = None,
) -> float:
    values = np.asarray(relevance, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    limit = len(values) if k is None else min(len(values), max(0, int(k)))
    if limit == 0:
        return float("nan")
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))
    dcg = float(np.sum(values[:limit] / discounts))
    ideal = np.sort(values)[::-1][:limit]
    ideal_dcg = float(np.sum(ideal / discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else float("nan")


def experiments_to_recall(
    labels: np.ndarray,
    targets: Iterable[float],
) -> dict[float, int | None]:
    labels = np.asarray(labels, dtype=bool)
    total = int(labels.sum())
    result: dict[float, int | None] = {}
    cumulative = np.cumsum(labels)
    for raw_target in targets:
        target = float(raw_target)
        if not 0 < target <= 1:
            raise ValueError("recall targets must be in (0, 1]")
        if total == 0:
            result[target] = None
            continue
        required = max(1, int(math.ceil(total * target)))
        positions = np.flatnonzero(cumulative >= required)
        result[target] = int(positions[0] + 1) if len(positions) else None
    return result


def evaluate_orders(
    orders: Mapping[tuple[str, int], list[str]],
    results: pd.DataFrame,
    *,
    budgets: Iterable[int],
    recall_targets: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "candidate_id",
        "family_fdr_chain_hit",
        "global_fdr_chain_hit",
        "nominal_chain_hit",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Case 2 result table is missing ranking labels: {missing}")
    indexed = results.set_index("candidate_id", drop=False)
    population = len(indexed)
    budget_values = sorted(
        {int(value) for value in budgets if 0 < int(value) <= population}
    )
    summary_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for (method, trial), candidate_ids in sorted(orders.items()):
        if len(candidate_ids) != population or set(candidate_ids) != set(indexed.index):
            raise ValueError(f"{method} trial {trial} is not a full candidate permutation")
        ordered = indexed.loc[candidate_ids]
        family = ordered["family_fdr_chain_hit"].astype(bool).to_numpy()
        global_hits = ordered["global_fdr_chain_hit"].astype(bool).to_numpy()
        nominal = ordered["nominal_chain_hit"].astype(bool).to_numpy()
        replication = (
            ordered["directional_replication_hit"].astype(bool).to_numpy()
            if "directional_replication_hit" in ordered.columns
            else np.zeros(population, dtype=bool)
        )
        evidence = (
            pd.to_numeric(
                ordered["heldout_chain_evidence_score"], errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(float)
            if "heldout_chain_evidence_score" in ordered.columns
            else np.zeros(population, dtype=float)
        )
        rank_scores = np.arange(population, 0, -1, dtype=float)
        recall_positions = experiments_to_recall(family, recall_targets)
        replication_positions = experiments_to_recall(replication, recall_targets)
        row: dict[str, Any] = {
            "method": method,
            "trial": int(trial),
            "candidate_count": population,
            "family_positive_count": int(family.sum()),
            "global_positive_count": int(global_hits.sum()),
            "replication_positive_count": int(replication.sum()),
            "normalized_family_discovery_recall_auc": normalized_recall_auc(
                family, budget_values
            ),
            "normalized_global_discovery_recall_auc": normalized_recall_auc(
                global_hits, budget_values
            ),
            "normalized_replication_recall_auc": normalized_recall_auc(
                replication, budget_values
            ),
            "heldout_chain_evidence_ndcg": normalized_discounted_cumulative_gain(
                evidence
            ),
            "family_fdr_auroc": binary_auroc(family, rank_scores),
            "family_fdr_auprc": average_precision(family, rank_scores),
            "global_fdr_auroc": binary_auroc(global_hits, rank_scores),
            "global_fdr_auprc": average_precision(global_hits, rank_scores),
            "replication_auroc": binary_auroc(replication, rank_scores),
            "replication_auprc": average_precision(replication, rank_scores),
            "first_family_hit_rank": (
                int(np.flatnonzero(family)[0] + 1) if family.any() else None
            ),
            "first_replication_hit_rank": (
                int(np.flatnonzero(replication)[0] + 1)
                if replication.any()
                else None
            ),
        }
        for target, rank in recall_positions.items():
            row[f"experiments_to_recall_{target:g}"] = rank
        for target, rank in replication_positions.items():
            row[f"experiments_to_replication_recall_{target:g}"] = rank
        summary_rows.append(row)
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            ranking_rows.append(
                {
                    "method": method,
                    "trial": int(trial),
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "family_fdr_chain_hit": bool(family[rank - 1]),
                    "global_fdr_chain_hit": bool(global_hits[rank - 1]),
                    "nominal_chain_hit": bool(nominal[rank - 1]),
                    "directional_replication_hit": bool(replication[rank - 1]),
                    "heldout_chain_evidence_score": float(evidence[rank - 1]),
                }
            )
        for budget in budget_values:
            family_hits = int(family[:budget].sum())
            scientific_hits = int(global_hits[:budget].sum())
            budget_rows.append(
                {
                    "method": method,
                    "trial": int(trial),
                    "k": budget,
                    "family_fdr_chain_hits": family_hits,
                    "family_fdr_chain_recall": (
                        family_hits / int(family.sum()) if family.any() else np.nan
                    ),
                    "global_fdr_chain_hits": scientific_hits,
                    "global_fdr_chain_recall": (
                        scientific_hits / int(global_hits.sum())
                        if global_hits.any()
                        else np.nan
                    ),
                    "nominal_chain_hits": int(nominal[:budget].sum()),
                    "directional_replication_hits": int(
                        replication[:budget].sum()
                    ),
                    "directional_replication_recall": (
                        int(replication[:budget].sum()) / int(replication.sum())
                        if replication.any()
                        else np.nan
                    ),
                    "heldout_chain_evidence_ndcg_at_k": (
                        normalized_discounted_cumulative_gain(evidence, k=budget)
                    ),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(budget_rows),
        pd.DataFrame(ranking_rows),
    )


def exact_paired_sign_flip_greater(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = float(values.mean())
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        signed = np.fromiter(
            (
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            ),
            dtype=float,
            count=len(values),
        )
        exceed += float(signed.mean()) >= observed - 1e-12
    return float(exceed / total)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices], kind="stable")]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def percentile_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(int(resamples), len(values)))]
    means = samples.mean(axis=1)
    return tuple(map(float, np.percentile(means, [2.5, 97.5])))


def paired_primary_comparisons(
    trial_summary: pd.DataFrame,
    *,
    metric: str,
    target_method: str,
    baseline_methods: Iterable[str],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    target = trial_summary.loc[
        trial_summary["method"].eq(target_method), ["trial", metric]
    ].rename(columns={metric: "target_value"})
    rows: list[dict[str, Any]] = []
    for offset, method in enumerate(baseline_methods):
        baseline = trial_summary.loc[
            trial_summary["method"].eq(method), ["trial", metric]
        ].rename(columns={metric: "baseline_value"})
        paired = target.merge(baseline, on="trial", validate="one_to_one")
        differences = paired["target_value"].to_numpy(float) - paired[
            "baseline_value"
        ].to_numpy(float)
        finite_differences = differences[np.isfinite(differences)]
        ci_low, ci_high = percentile_bootstrap_mean_ci(
            differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + offset,
        )
        rows.append(
            {
                "metric": metric,
                "target": target_method,
                "baseline": method,
                "n_pairs": int(len(paired)),
                "n_finite_pairs": int(len(finite_differences)),
                "target_mean": float(paired["target_value"].mean()),
                "baseline_mean": float(paired["baseline_value"].mean()),
                "paired_mean_difference": (
                    float(finite_differences.mean())
                    if len(finite_differences)
                    else float("nan")
                ),
                "paired_difference_ci95_low": ci_low,
                "paired_difference_ci95_high": ci_high,
                "p_one_sided_exact_sign_flip": exact_paired_sign_flip_greater(
                    differences
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm_six_baselines"] = holm_adjust(
        frame["p_one_sided_exact_sign_flip"]
    )
    frame["target_superior"] = (
        frame["paired_mean_difference"].gt(0)
        & frame["paired_difference_ci95_low"].gt(0)
        & frame["p_holm_six_baselines"].lt(0.05)
    )
    return frame


def aggregate_trials(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "trial"
    ]
    rows: list[dict[str, Any]] = []
    for method, group in frame.groupby("method", sort=False):
        row: dict[str, Any] = {
            "method": method,
            "n_trials": int(group["trial"].nunique()),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_sd"] = float("nan")
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


# Last Updated At: 2026-08-16 13:49 HKT
