"""Create the authoritative summary for a completed Case Study 1 rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


TARGET_METHOD = "neurodiscovery"
EXPECTED_TRIALS = 10
EXPECTED_EXTERNAL_DATASETS = {"pooled", "adhd200", "cobre", "hcpep", "ucla"}
PRIMARY_BASELINES = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
)
SUPPLEMENTARY_BASELINES = ("data_to_paper", "openscholar_rag")
PRIMARY_METHODS = (*PRIMARY_BASELINES, TARGET_METHOD)

METHOD_LABELS = {
    "ai_scientist_v2": "AI Scientist-v2",
    "open_coscientist": "Open Co-Scientist",
    "data_to_paper": "Data-to-Paper",
    "sciagents": "SciAgents",
    "virtual_lab": "Virtual Lab",
    "brainpilot_native": "BrainPilot",
    "biomni_native": "Biomni",
    "openscholar_rag": "OpenScholar + fixed generator",
    "neurodiscovery": "NeuroDiscovery",
}
DATASET_LABELS = {
    "pooled": "Pooled external validation",
    "adhd200": "ADHD-200",
    "cobre": "COBRE",
    "hcpep": "HCP-EP",
    "ucla": "UCLA",
}

INTERNAL_BUDGETS = (100, 500, 1000, 5000, 10000, 50000, 100000, 200000)
RECALL_TARGETS_PCT = (1, 5, 10, 20, 30, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--expected-trials",
        type=int,
        default=EXPECTED_TRIALS,
        help=(
            "Required independent trials per primary method. Defaults to the "
            "historical 10-trial publication protocol."
        ),
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=None,
        help=(
            "External-validation result directory. Defaults to "
            "<run-root>/external_method_comparison."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sample_variance(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.var(array, ddof=1)) if len(array) > 1 else 0.0


def summarize_values(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_cols, sort=False, dropna=False)
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values, strict=True))
        row["n_seeds"] = int(group["trial"].nunique())
        for column in value_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(finite.mean()) if len(finite) else math.nan
            row[f"{column}_variance"] = sample_variance(finite)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon(
    frame: pd.DataFrame,
    value_col: str,
    baseline: str,
    alternative: str,
    extra_filters: dict[str, object],
) -> dict[str, object]:
    subset = frame.copy()
    for column, value in extra_filters.items():
        subset = subset[subset[column] == value]
    pivot = subset.pivot_table(
        index="trial",
        columns="method",
        values=value_col,
        aggfunc="first",
    )
    if TARGET_METHOD not in pivot or baseline not in pivot:
        return {
            "n_paired_seeds": 0,
            "neurodiscovery_mean": math.nan,
            "baseline_mean": math.nan,
            "mean_paired_difference": math.nan,
            "wilcoxon_statistic": math.nan,
            "p_value_one_sided": math.nan,
            "alternative": alternative,
        }
    pair = pivot[[TARGET_METHOD, baseline]].dropna()
    if pair.empty:
        return {
            "n_paired_seeds": 0,
            "neurodiscovery_mean": math.nan,
            "baseline_mean": math.nan,
            "mean_paired_difference": math.nan,
            "wilcoxon_statistic": math.nan,
            "p_value_one_sided": math.nan,
            "alternative": alternative,
        }
    nd = pair[TARGET_METHOD].to_numpy(float)
    base = pair[baseline].to_numpy(float)
    differences = nd - base
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(
            nd,
            base,
            alternative=alternative,
            zero_method="wilcox",
        )
    return {
        "n_paired_seeds": int(len(pair)),
        "neurodiscovery_mean": float(np.mean(nd)),
        "baseline_mean": float(np.mean(base)),
        "mean_paired_difference": float(np.mean(differences)),
        "wilcoxon_statistic": float(statistic),
        "p_value_one_sided": float(p_value),
        "alternative": alternative,
    }


def select_budget_rows(
    frame: pd.DataFrame,
    budgets: Iterable[int],
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for (method, trial), group in frame.groupby(["method", "trial"], sort=False):
        group = group.sort_values("budget")
        for budget in budgets:
            exact = group[group["budget"] == budget]
            if not exact.empty:
                selected = exact.iloc[-1].copy()
            else:
                eligible = group[group["budget"] <= budget]
                if eligible.empty:
                    continue
                selected = eligible.iloc[-1].copy()
            selected["requested_budget"] = int(budget)
            selected["evaluated_budget"] = int(selected["budget"])
            selected["method"] = method
            selected["trial"] = int(trial)
            rows.append(selected)
    return pd.DataFrame(rows)


def load_internal_trials(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal_dir = run_root / "internal_method_comparison"
    native_dir = run_root / "native_baselines"

    api_curves = pd.read_csv(
        require_file(internal_dir / "case1_discovery_curves_by_trial.csv")
    )
    api_curves = api_curves.rename(columns={"recall": "gt_recall"})
    api_curves["trial"] = api_curves["trial"].astype(int)
    api_curves = api_curves[
        ["method", "trial", "budget", "gt_hits", "gt_recall", "strict_fdr_hits"]
    ]

    curves = api_curves[api_curves["method"].isin(PRIMARY_METHODS)].copy()
    missing_curve_methods = set(PRIMARY_METHODS) - set(curves["method"].unique())
    if missing_curve_methods:
        native_curves = pd.read_csv(
            require_file(native_dir / "native_baselines_full_curve_seed.csv")
        ).rename(columns={"seed": "trial"})
        native_curves["trial"] = native_curves["trial"].astype(int)
        native_curves = native_curves[
            [
                "method",
                "trial",
                "budget",
                "gt_hits",
                "gt_recall",
                "strict_fdr_hits",
            ]
        ]
        curves = pd.concat(
            [
                curves,
                native_curves[
                    native_curves["method"].isin(missing_curve_methods)
                ],
            ],
            ignore_index=True,
        )
    curves = curves.drop_duplicates(["method", "trial", "budget"])
    curves["label"] = curves["method"].map(METHOD_LABELS)

    api_recall = pd.read_csv(
        require_file(internal_dir / "case1_method_summary_by_trial.csv")
    )
    recall_columns = [
        f"experiments_for_recall_{target}pct" for target in RECALL_TARGETS_PCT
    ]
    api_recall = api_recall[["method", "trial", *recall_columns]]
    api_recall["trial"] = api_recall["trial"].astype(int)

    recall = api_recall[api_recall["method"].isin(PRIMARY_METHODS)].copy()
    missing_recall_methods = set(PRIMARY_METHODS) - set(recall["method"].unique())
    if missing_recall_methods:
        native_recall = pd.read_csv(
            require_file(native_dir / "native_baselines_recall_cost_seed.csv")
        ).rename(columns={"seed": "trial"})
        native_recall["trial"] = native_recall["trial"].astype(int)
        native_recall = native_recall[["method", "trial", *recall_columns]]
        recall = pd.concat(
            [
                recall,
                native_recall[
                    native_recall["method"].isin(missing_recall_methods)
                ],
            ],
            ignore_index=True,
        )
    recall = recall.drop_duplicates(["method", "trial"])
    recall["label"] = recall["method"].map(METHOD_LABELS)
    return curves, recall


def build_generation_quality_output(
    run_root: Path,
    out_dir: Path,
) -> tuple[Path, dict[str, object]]:
    official_dir = run_root / "official_baselines"
    official_summary_path = official_dir / "official_native_proposal_summary.csv"
    if official_summary_path.is_file():
        api = pd.read_csv(official_summary_path)
        api = api[api["method"].isin(PRIMARY_BASELINES)].copy()
        api = api.rename(
            columns={
                "requested_anchors": "generated_slots",
                "valid_anchors": "mapped",
            }
        )
        api["schema_valid"] = api["mapped"]
        api["mapping_rate"] = api["mapped"] / api["generated_slots"]

        ranked_gt = pd.read_csv(
            require_file(
                run_root
                / "internal_method_comparison"
                / "ranked_candidates_exhaustive_gt.csv"
            ),
            usecols=["candidate_id", "is_gt_top"],
        )
        gt_ids = set(
            ranked_gt.loc[
                ranked_gt["is_gt_top"].astype(str).str.casefold().eq("true"),
                "candidate_id",
            ].astype(str)
        )
        mapped = pd.read_csv(
            require_file(official_dir / "generation_first_mapped_hypotheses.csv"),
            usecols=["method", "seed", "mapping_status", "mapped_candidate_id"],
        )
        mapped = mapped[
            mapped["mapping_status"].astype(str).str.casefold().eq("mapped")
        ].copy()
        mapped["is_gt"] = mapped["mapped_candidate_id"].astype(str).isin(gt_ids)
        gt_hits = (
            mapped.groupby(["method", "seed"], as_index=False)["is_gt"]
            .sum()
            .rename(columns={"seed": "trial", "is_gt": "gt_hits"})
        )
        api = api.merge(gt_hits, on=["method", "trial"], how="left")
        api["gt_hits"] = api["gt_hits"].fillna(0).astype(int)
    else:
        api = pd.read_csv(
            require_file(
                run_root
                / "generation_first"
                / "generation_first_summary_by_seed.csv"
            )
        )
        api = api[
            (api["budget_format_correct"] == 80)
            & api["method"].isin(PRIMARY_BASELINES)
        ].copy()
        api = api.rename(
            columns={
                "seed": "trial",
                "generated_total": "generated_slots",
                "schema_format_correct_total": "schema_valid",
                "valid_mapped_total": "mapped",
                "valid_mapping_rate": "mapping_rate",
            }
        )

    native = pd.read_csv(
        require_file(
            run_root
            / "native_baselines"
            / "native_baselines_direct_seed_summary.csv"
        )
    )
    native = native[
        (native["budget"] == 80) & native["method"].isin(PRIMARY_BASELINES)
    ].copy()
    native = native.rename(columns={"seed": "trial"})
    native["generated_slots"] = native["budget"]

    columns = [
        "method",
        "trial",
        "generated_slots",
        "schema_valid",
        "mapped",
        "mapping_rate",
        "gt_hits",
    ]
    combined = pd.concat([api[columns], native[columns]], ignore_index=True)
    combined["label"] = combined["method"].map(METHOD_LABELS)
    if set(combined["method"].unique()) != set(PRIMARY_BASELINES):
        raise ValueError(
            "Generation-quality primary method set mismatch: "
            f"{sorted(combined['method'].unique())}"
        )
    counts = combined.groupby("method")["trial"].nunique()
    if not (counts == EXPECTED_TRIALS).all():
        raise ValueError(
            f"Generation-quality trial counts mismatch: {counts.to_dict()}"
        )
    summary = summarize_values(
        combined,
        ["method", "label"],
        [
            "generated_slots",
            "schema_valid",
            "mapped",
            "mapping_rate",
            "gt_hits",
        ],
    ).sort_values("method")
    path = out_dir / "generation_quality_primary_summary.csv"
    summary.to_csv(path, index=False)
    audit = {
        "methods": sorted(combined["method"].unique().tolist()),
        "n_seeds_by_method": counts.astype(int).to_dict(),
        "generated_slots_per_seed": 80,
        "failure_policy": (
            "Invalid, duplicate, missing, and unmappable outputs consume a "
            "generation slot and receive zero hit; no manual repair."
        ),
    }
    return path, audit


def build_internal_outputs(
    run_root: Path,
    out_dir: Path,
) -> tuple[list[Path], dict[str, object]]:
    curves, recall = load_internal_trials(run_root)
    if set(curves["method"].unique()) != set(PRIMARY_METHODS):
        raise ValueError(
            "Internal primary method set mismatch: "
            f"{sorted(curves['method'].unique())}"
        )
    curve_counts = curves.groupby("method")["trial"].nunique()
    recall_counts = recall.groupby("method")["trial"].nunique()
    if not (curve_counts == EXPECTED_TRIALS).all():
        raise ValueError(f"Internal curve trial counts mismatch: {curve_counts.to_dict()}")
    if not (recall_counts == EXPECTED_TRIALS).all():
        raise ValueError(
            f"Internal recall trial counts mismatch: {recall_counts.to_dict()}"
        )
    selected = select_budget_rows(curves, INTERNAL_BUDGETS)
    if not (
        selected["requested_budget"].astype(int)
        == selected["evaluated_budget"].astype(int)
    ).all():
        mismatch = selected[
            selected["requested_budget"].astype(int)
            != selected["evaluated_budget"].astype(int)
        ]
        raise ValueError(
            "Internal budget grid lacks exact requested points: "
            f"{mismatch[['method', 'trial', 'requested_budget', 'evaluated_budget']].head().to_dict('records')}"
        )
    selected["label"] = selected["method"].map(METHOD_LABELS)

    budget_summary = summarize_values(
        selected,
        ["method", "label", "requested_budget"],
        ["evaluated_budget", "gt_hits", "gt_recall", "strict_fdr_hits"],
    ).sort_values(["requested_budget", "method"])
    budget_path = out_dir / "internal_primary_budget_summary.csv"
    budget_summary.to_csv(budget_path, index=False)

    recall_long = recall.melt(
        id_vars=["method", "label", "trial"],
        value_vars=[
            f"experiments_for_recall_{target}pct" for target in RECALL_TARGETS_PCT
        ],
        var_name="recall_target_label",
        value_name="experiments_required",
    )
    recall_long["recall_target_pct"] = (
        recall_long["recall_target_label"]
        .str.extract(r"(\d+)pct", expand=False)
        .astype(int)
    )
    recall_summary = summarize_values(
        recall_long,
        ["method", "label", "recall_target_pct"],
        ["experiments_required"],
    ).sort_values(["recall_target_pct", "method"])
    recall_path = out_dir / "internal_primary_recall_cost_summary.csv"
    recall_summary.to_csv(recall_path, index=False)

    p_rows: list[dict[str, object]] = []
    for budget in INTERNAL_BUDGETS:
        for baseline in PRIMARY_BASELINES:
            p_rows.append(
                {
                    "comparison_type": "same_experiments_gt_hits",
                    "budget_or_recall_target": budget,
                    "baseline_method": baseline,
                    **paired_wilcoxon(
                        selected,
                        "gt_hits",
                        baseline,
                        "greater",
                        {"requested_budget": budget},
                    ),
                }
            )
    for target in RECALL_TARGETS_PCT:
        for baseline in PRIMARY_BASELINES:
            p_rows.append(
                {
                    "comparison_type": "same_recall_experiments",
                    "budget_or_recall_target": target,
                    "baseline_method": baseline,
                    **paired_wilcoxon(
                        recall_long,
                        "experiments_required",
                        baseline,
                        "less",
                        {"recall_target_pct": target},
                    ),
                }
            )
    p_values = pd.DataFrame(p_rows)
    p_values["baseline_label"] = p_values["baseline_method"].map(METHOD_LABELS)
    p_path = out_dir / "internal_primary_p_values.csv"
    p_values.to_csv(p_path, index=False)

    budget_headline_rows: list[dict[str, object]] = []
    for budget in INTERNAL_BUDGETS:
        group = budget_summary[budget_summary["requested_budget"] == budget]
        nd = group[group["method"] == TARGET_METHOD].iloc[0]
        baselines = group[group["method"].isin(PRIMARY_BASELINES)]
        best = baselines.sort_values("gt_hits_mean", ascending=False).iloc[0]
        budget_headline_rows.append(
            {
                "budget": budget,
                "neurodiscovery_gt_hits_mean": nd["gt_hits_mean"],
                "neurodiscovery_gt_hits_variance": nd["gt_hits_variance"],
                "best_baseline_method": best["method"],
                "best_baseline_label": best["label"],
                "best_baseline_gt_hits_mean": best["gt_hits_mean"],
                "best_baseline_gt_hits_variance": best["gt_hits_variance"],
                "additional_discoveries": nd["gt_hits_mean"] - best["gt_hits_mean"],
                "fold_more_discoveries": (
                    nd["gt_hits_mean"] / best["gt_hits_mean"]
                    if best["gt_hits_mean"] > 0
                    else math.inf
                ),
            }
        )
    budget_headline = pd.DataFrame(budget_headline_rows)
    budget_headline_path = out_dir / "internal_same_experiments_headline.csv"
    budget_headline.to_csv(budget_headline_path, index=False)

    recall_headline_rows: list[dict[str, object]] = []
    for target in RECALL_TARGETS_PCT:
        group = recall_summary[recall_summary["recall_target_pct"] == target]
        nd = group[group["method"] == TARGET_METHOD].iloc[0]
        baselines = group[group["method"].isin(PRIMARY_BASELINES)]
        best = baselines.sort_values("experiments_required_mean").iloc[0]
        nd_mean = float(nd["experiments_required_mean"])
        best_mean = float(best["experiments_required_mean"])
        recall_headline_rows.append(
            {
                "recall_target_pct": target,
                "neurodiscovery_experiments_mean": nd_mean,
                "neurodiscovery_experiments_variance": nd[
                    "experiments_required_variance"
                ],
                "best_baseline_method": best["method"],
                "best_baseline_label": best["label"],
                "best_baseline_experiments_mean": best_mean,
                "best_baseline_experiments_variance": best[
                    "experiments_required_variance"
                ],
                "fewer_experiments_pct": 100.0 * (1.0 - nd_mean / best_mean),
                "speedup": best_mean / nd_mean,
            }
        )
    recall_headline = pd.DataFrame(recall_headline_rows)
    recall_headline_path = out_dir / "internal_same_recall_headline.csv"
    recall_headline.to_csv(recall_headline_path, index=False)

    outputs = [
        budget_path,
        recall_path,
        p_path,
        budget_headline_path,
        recall_headline_path,
    ]
    audit = {
        "methods": sorted(curves["method"].unique().tolist()),
        "n_curve_trials_by_method": curves.groupby("method")["trial"]
        .nunique()
        .astype(int)
        .to_dict(),
        "n_recall_trials_by_method": recall.groupby("method")["trial"]
        .nunique()
        .astype(int)
        .to_dict(),
        "budgets": list(INTERNAL_BUDGETS),
        "recall_targets_pct": list(RECALL_TARGETS_PCT),
    }
    return outputs, audit


def build_external_outputs(
    run_root: Path,
    out_dir: Path,
    external_dir: Path | None = None,
) -> tuple[list[Path], dict[str, object]]:
    external_dir = external_dir or run_root / "external_method_comparison"
    metrics = pd.read_csv(
        require_file(external_dir / "case1_external_metrics_by_trial.csv")
    )
    metrics = metrics[
        metrics["method"].isin(PRIMARY_METHODS) & (metrics["scope"] == "tcp_budget")
    ].copy()
    metrics["trial"] = metrics["trial"].astype(int)
    metrics["label"] = metrics["method"].map(METHOD_LABELS)
    metrics["dataset_label"] = metrics["dataset"].map(DATASET_LABELS)
    if set(metrics["dataset"].unique()) != EXPECTED_EXTERNAL_DATASETS:
        raise ValueError(
            "External dataset set mismatch: "
            f"{sorted(metrics['dataset'].unique())}"
        )
    if set(metrics["method"].unique()) != set(PRIMARY_METHODS):
        raise ValueError(
            "External primary method set mismatch: "
            f"{sorted(metrics['method'].unique())}"
        )
    external_counts = metrics.groupby(["dataset", "method"])["trial"].nunique()
    if not (external_counts == EXPECTED_TRIALS).all():
        raise ValueError(
            "External metric trial counts mismatch: "
            f"{external_counts[external_counts != EXPECTED_TRIALS].to_dict()}"
        )

    metric_summary = summarize_values(
        metrics,
        ["dataset", "dataset_label", "method", "label", "scope", "budget"],
        [
            "tcp_rank_reached",
            "n_confirmed",
            "n_falsified",
            "n_conflicting",
            "n_not_confirmed",
            "n_executable",
            "external_precision",
            "external_recall",
            "external_gt_total",
            "executable_fraction",
        ],
    ).sort_values(["dataset", "budget", "method"])
    metric_path = out_dir / "external_primary_metrics_summary.csv"
    metric_summary.to_csv(metric_path, index=False)

    recall = pd.read_csv(
        require_file(external_dir / "case1_external_recall_cost_by_trial.csv")
    )
    recall = recall[recall["method"].isin(PRIMARY_METHODS)].copy()
    recall["trial"] = recall["trial"].astype(int)
    recall["label"] = recall["method"].map(METHOD_LABELS)
    recall["dataset_label"] = recall["dataset"].map(DATASET_LABELS)
    recall_summary = summarize_values(
        recall,
        ["dataset", "dataset_label", "method", "label", "recall_target"],
        [
            "confirmed_needed",
            "external_gt_total",
            "tcp_experiments_required",
            "executable_experiments_required",
        ],
    ).sort_values(["dataset", "recall_target", "method"])
    recall_path = out_dir / "external_primary_recall_cost_summary.csv"
    recall_summary.to_csv(recall_path, index=False)

    p_rows: list[dict[str, object]] = []
    for dataset in sorted(metrics["dataset"].unique()):
        for budget in sorted(metrics.loc[metrics["dataset"] == dataset, "budget"].unique()):
            for baseline in PRIMARY_BASELINES:
                p_rows.append(
                    {
                        "dataset": dataset,
                        "comparison_type": "same_experiments_confirmed",
                        "budget_or_recall_target": float(budget),
                        "baseline_method": baseline,
                        **paired_wilcoxon(
                            metrics,
                            "n_confirmed",
                            baseline,
                            "greater",
                            {"dataset": dataset, "budget": budget},
                        ),
                    }
                )
    for dataset in sorted(recall["dataset"].unique()):
        targets = sorted(
            recall.loc[recall["dataset"] == dataset, "recall_target"].unique()
        )
        for target in targets:
            for baseline in PRIMARY_BASELINES:
                p_rows.append(
                    {
                        "dataset": dataset,
                        "comparison_type": "same_recall_tcp_experiments",
                        "budget_or_recall_target": float(target),
                        "baseline_method": baseline,
                        **paired_wilcoxon(
                            recall,
                            "tcp_experiments_required",
                            baseline,
                            "less",
                            {"dataset": dataset, "recall_target": target},
                        ),
                    }
                )
    p_values = pd.DataFrame(p_rows)
    p_values["baseline_label"] = p_values["baseline_method"].map(METHOD_LABELS)
    p_path = out_dir / "external_primary_p_values.csv"
    p_values.to_csv(p_path, index=False)

    pooled_budget_rows: list[dict[str, object]] = []
    pooled_metrics = metric_summary[metric_summary["dataset"] == "pooled"]
    for budget in sorted(pooled_metrics["budget"].unique()):
        group = pooled_metrics[pooled_metrics["budget"] == budget]
        nd = group[group["method"] == TARGET_METHOD].iloc[0]
        baselines = group[group["method"].isin(PRIMARY_BASELINES)]
        best = baselines.sort_values("n_confirmed_mean", ascending=False).iloc[0]
        pooled_budget_rows.append(
            {
                "budget": int(budget),
                "neurodiscovery_confirmed_mean": nd["n_confirmed_mean"],
                "neurodiscovery_confirmed_variance": nd["n_confirmed_variance"],
                "best_baseline_method": best["method"],
                "best_baseline_label": best["label"],
                "best_baseline_confirmed_mean": best["n_confirmed_mean"],
                "best_baseline_confirmed_variance": best["n_confirmed_variance"],
                "additional_confirmations": (
                    nd["n_confirmed_mean"] - best["n_confirmed_mean"]
                ),
                "fold_more_confirmations": (
                    nd["n_confirmed_mean"] / best["n_confirmed_mean"]
                    if best["n_confirmed_mean"] > 0
                    else math.inf
                ),
            }
        )
    pooled_budget = pd.DataFrame(pooled_budget_rows)
    pooled_budget_path = out_dir / "external_pooled_same_experiments_headline.csv"
    pooled_budget.to_csv(pooled_budget_path, index=False)

    pooled_recall_rows: list[dict[str, object]] = []
    pooled_recall = recall_summary[recall_summary["dataset"] == "pooled"]
    for target in sorted(pooled_recall["recall_target"].unique()):
        group = pooled_recall[pooled_recall["recall_target"] == target]
        nd = group[group["method"] == TARGET_METHOD].iloc[0]
        baselines = group[group["method"].isin(PRIMARY_BASELINES)]
        best = baselines.sort_values("tcp_experiments_required_mean").iloc[0]
        nd_mean = float(nd["tcp_experiments_required_mean"])
        best_mean = float(best["tcp_experiments_required_mean"])
        pooled_recall_rows.append(
            {
                "recall_target_pct": 100.0 * float(target),
                "neurodiscovery_tcp_experiments_mean": nd_mean,
                "neurodiscovery_tcp_experiments_variance": nd[
                    "tcp_experiments_required_variance"
                ],
                "best_baseline_method": best["method"],
                "best_baseline_label": best["label"],
                "best_baseline_tcp_experiments_mean": best_mean,
                "best_baseline_tcp_experiments_variance": best[
                    "tcp_experiments_required_variance"
                ],
                "fewer_experiments_pct": 100.0 * (1.0 - nd_mean / best_mean),
                "speedup": best_mean / nd_mean,
            }
        )
    pooled_recall_headline = pd.DataFrame(pooled_recall_rows)
    pooled_recall_path = out_dir / "external_pooled_same_recall_headline.csv"
    pooled_recall_headline.to_csv(pooled_recall_path, index=False)

    outputs = [
        metric_path,
        recall_path,
        p_path,
        pooled_budget_path,
        pooled_recall_path,
    ]
    audit = {
        "datasets": sorted(metrics["dataset"].unique().tolist()),
        "methods": sorted(metrics["method"].unique().tolist()),
        "n_trials_by_dataset_method": {
            f"{dataset}|{method}": int(group["trial"].nunique())
            for (dataset, method), group in metrics.groupby(["dataset", "method"])
        },
        "external_gt_total_by_dataset": {
            dataset: int(group["external_gt_total"].iloc[0])
            for dataset, group in metrics.groupby("dataset")
        },
    }
    return outputs, audit


def write_method_scope(out_dir: Path) -> Path:
    path = out_dir / "method_scope.json"
    payload = {
        "target_method": TARGET_METHOD,
        "primary_baselines": list(PRIMARY_BASELINES),
        "supplementary_baselines": list(SUPPLEMENTARY_BASELINES),
        "labels": METHOD_LABELS,
        "policy": {
            "data_to_paper": (
                "Supplementary end-to-end workflow because its manuscript-oriented "
                "objective is not a direct hypothesis-search baseline."
            ),
            "openscholar_rag": (
                "Supplementary retrieval baseline. It executes OpenScholar retrieval, "
                "freezes the evidence, and applies the common fixed projection generator; "
                "OpenScholar itself is not treated as a hypothesis generator."
            )
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_manifest(
    run_root: Path,
    out_dir: Path,
    outputs: list[Path],
    generation_audit: dict[str, object],
    internal_audit: dict[str, object],
    external_audit: dict[str, object],
    external_dir: Path | None = None,
) -> Path:
    external_dir = external_dir or run_root / "external_method_comparison"
    internal_manifest_path = (
        run_root
        / "internal_method_comparison"
        / "case1_method_comparison_manifest.json"
    )
    internal_manifest = json.loads(
        require_file(internal_manifest_path).read_text(encoding="utf-8")
    )
    exhaustive_path = Path(internal_manifest["all_tests"])
    kg_path = Path(internal_manifest["kg_path"])
    input_paths = [
        exhaustive_path,
        kg_path,
        run_root / "official_baselines" / "case1_search_policies.jsonl",
        run_root
        / "official_baselines"
        / "generation_first_mapped_hypotheses.csv",
        run_root
        / "official_baselines"
        / "official_native_proposal_summary.csv",
        run_root
        / "native_baselines"
        / "native_baselines_direct_seed_summary.csv",
        run_root
        / "internal_method_comparison"
        / "case1_discovery_curves_by_trial.csv",
        run_root
        / "internal_method_comparison"
        / "case1_method_summary_by_trial.csv",
        external_dir / "case1_external_metrics_by_trial.csv",
        external_dir / "case1_external_recall_cost_by_trial.csv",
    ]
    artifact_paths = [
        run_root / "official_baselines" / "official_adapter_manifest.json",
        run_root / "official_baselines" / "case1_policy_independence_audit.json",
        run_root / "native_baselines" / "native_baselines_manifest.json",
        internal_manifest_path,
        run_root
        / "internal_method_comparison"
        / "case1_generator_comparison_main.png",
        run_root
        / "internal_method_comparison"
        / "case1_generator_comparison_main.pdf",
        external_dir / "case1_external_method_comparison_manifest.json",
        external_dir / "case1_external_fixed_effect_meta_analysis.csv",
        external_dir / "case1_external_compatibility_audit.csv",
    ]
    input_paths = [path for path in input_paths if path.is_file()]
    artifact_paths = [path for path in artifact_paths if path.is_file()]
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("case1_method_comparison.py"),
        Path(__file__).resolve().with_name("case1_external_method_comparison.py"),
        Path(__file__).resolve().with_name(
            "plot_case1_generator_surface_comparison.py"
        ),
    ]
    manifest_path = out_dir / "case1_final_manifest.json"
    manifest = {
        "authoritative": True,
        "run_root": str(run_root),
        "model": {
            "name": "gpt-5.5",
            "reasoning_effort": "high",
            "wire_api": "responses",
            "api_key_persisted": False,
        },
        "method_scope": {
            "target": TARGET_METHOD,
            "primary_baselines": list(PRIMARY_BASELINES),
            "supplementary_baselines": list(SUPPLEMENTARY_BASELINES),
        },
        "statistics": {
            "stochastic_repetitions": EXPECTED_TRIALS,
            "dispersion": "sample variance across seeds (ddof=1)",
            "paired_test": "one-sided paired Wilcoxon signed-rank test",
            "internal_gt": "top 1% by absolute adjusted residual Cohen d",
            "external_confirmation": (
                "dataset-specific disease contrast x feature-family BH-FDR q<0.05, "
                "|d|>0.15, and direction concordant with TCP"
            ),
        },
        "generation_audit": generation_audit,
        "internal_audit": internal_audit,
        "external_audit": external_audit,
        "inputs": {
            (
                str(path.relative_to(run_root))
                if path.is_relative_to(run_root)
                else path.name
            ): {
                "path": str(path),
                "sha256": sha256_file(require_file(path)),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        },
        "sources": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(require_file(path)),
            }
            for path in source_paths
        },
        "outputs": {
            str(path.relative_to(out_dir)): {
                "path": str(path),
                "sha256": sha256_file(require_file(path)),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        },
        "artifacts": {
            str(path.relative_to(run_root)): {
                "path": str(path),
                "sha256": sha256_file(require_file(path)),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    global EXPECTED_TRIALS
    args = parse_args()
    if args.expected_trials < 2:
        raise ValueError("--expected-trials must be at least 2")
    EXPECTED_TRIALS = int(args.expected_trials)
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(run_root)
    out_dir = run_root / "final_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    external_dir = (
        args.external_dir.resolve()
        if args.external_dir is not None
        else run_root / "external_method_comparison"
    )

    method_scope_path = write_method_scope(out_dir)
    generation_path, generation_audit = build_generation_quality_output(
        run_root, out_dir
    )
    internal_outputs, internal_audit = build_internal_outputs(run_root, out_dir)
    external_outputs, external_audit = build_external_outputs(
        run_root,
        out_dir,
        external_dir,
    )
    outputs = [
        method_scope_path,
        generation_path,
        *internal_outputs,
        *external_outputs,
    ]
    manifest_path = write_manifest(
        run_root,
        out_dir,
        outputs,
        generation_audit,
        internal_audit,
        external_audit,
        external_dir,
    )

    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "final_summary": str(out_dir),
                "manifest": str(manifest_path),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
