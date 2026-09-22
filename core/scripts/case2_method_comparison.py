"""Compare Case Study 2 NeuroDiscovery with six native autoresearch frameworks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from core.scripts.case2_search_policy import (
    COMPILER_VERSION,
    SearchPolicy,
    build_public_registry,
    candidate_id_from_fields,
    compile_policy_order,
    policy_from_payload,
)
from core.scripts.case_study_policy_audit import audit_policy_independence
from core.scripts.case_study_closed_loop import ClosedLoopConfig, write_kg_delta
from core.scripts.case_study_closed_loop_engine import run_closed_loop_order
from core.scripts.case_study_feedback_adapters import adapter_for
from core.scripts.case_study_score_components import (
    embedded_score_component_audit,
    load_score_component_bundle,
)
from core.scripts.canonical_kg_release import (
    validate_canonical_kg_release,
    write_release_manifest,
)

DEFAULT_RESULTS_PARENT = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc")
    / "case2_adni_genetics_v1"
    / "experiments"
    / "case2_adni_kg_guided_longitudinal_v1"
)
DEFAULT_K = (5, 10, 20, 50, 100, 210)
NEURODISCOVERY_PRIORS = ("kg_rank_v1", "evidence_consensus_v2")
EVIDENCE_CONSENSUS_WEIGHT = 0.25


def _latest(parent: Path, relative: str) -> Path:
    direct = parent / relative
    candidates = [direct] if direct.is_file() else []
    candidates.extend(path for path in parent.glob(f"*/{relative}") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No {relative} below {parent}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--policies", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--current-state", type=Path, required=True)
    parser.add_argument("--score-components", type=Path)
    parser.add_argument("--score-components-manifest", type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--nd-trials", type=int, default=10)
    parser.add_argument("--nd-batch-size", type=int, default=5)
    parser.add_argument(
        "--nd-prior",
        choices=NEURODISCOVERY_PRIORS,
        default="evidence_consensus_v2",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--fail-on-collapse", action="store_true")
    return parser.parse_args()


def _attach_candidate_ids(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["candidate_id"] = [
        candidate_id_from_fields(*values)
        for values in frame.loc[
            :, ["exposure", "modality", "marker", "outcome"]
        ].itertuples(index=False, name=None)
    ]
    if frame["candidate_id"].duplicated().any():
        raise ValueError("experimental Case 2 results contain duplicate candidates")
    return frame


def load_policies(path: Path) -> list[SearchPolicy]:
    policies: list[SearchPolicy] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                policies.append(policy_from_payload(json.loads(line)))
    if not policies:
        raise ValueError(f"No search policies in {path}")
    return policies


def _minmax(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must be finite for every Case 2 candidate")
    lower = float(numeric.min())
    upper = float(numeric.max())
    if math.isclose(lower, upper):
        return np.full(len(numeric), 0.5, dtype=float)
    return (numeric - lower) / (upper - lower)


def _build_neurodiscovery_prior(
    registry: pd.DataFrame,
    indexed_results: pd.DataFrame,
    *,
    prior_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a frozen KG-only prior without reading experimental statistics."""

    if prior_name not in NEURODISCOVERY_PRIORS:
        raise ValueError(f"unknown NeuroDiscovery prior: {prior_name!r}")
    candidate_ids = registry["candidate_id"].astype(str)
    if prior_name == "kg_rank_v1":
        kg_rank = candidate_ids.map(indexed_results["kg_rank"])
        score = 1.0 - _minmax(kg_rank, name="kg_rank")
        return score, {
            "name": prior_name,
            "outcome_blind": True,
            "frozen_before_experiment": True,
            "source_fields": ["kg_rank"],
            "formula": "1 - minmax(kg_rank)",
        }

    mapping = _minmax(
        candidate_ids.map(indexed_results["mapping_score"]),
        name="mapping_score",
    )
    support_count = pd.to_numeric(
        candidate_ids.map(indexed_results["meaningful_support_count"]),
        errors="coerce",
    )
    if support_count.isna().any() or support_count.lt(0).any():
        raise ValueError(
            "meaningful_support_count must be finite and non-negative for every "
            "Case 2 candidate"
        )
    consensus = _minmax(
        np.log1p(support_count).rename("log1p_meaningful_support_count"),
        name="log1p_meaningful_support_count",
    )
    weight = EVIDENCE_CONSENSUS_WEIGHT
    score = (mapping + weight * consensus) / (1.0 + weight)
    return score, {
        "name": prior_name,
        "outcome_blind": True,
        "frozen_before_experiment": True,
        "source_fields": ["mapping_score", "meaningful_support_count"],
        "formula": (
            "(minmax(mapping_score) + 0.25 * "
            "minmax(log1p(meaningful_support_count))) / 1.25"
        ),
        "evidence_consensus_weight": weight,
        "support_transform": "log1p_then_minmax",
        "experimental_columns_used": [],
    }


def compile_blinded_orders(
    registry: pd.DataFrame,
    policies: Iterable[SearchPolicy],
    neurodiscovery_ids: list[str],
    *,
    nd_trials: int,
) -> dict[tuple[str, int], list[str]]:
    """Freeze all candidate orders before experimental statistics are accessed."""

    orders: dict[tuple[str, int], list[str]] = {}
    for policy in policies:
        index = compile_policy_order(registry, policy)
        orders[(policy.method, policy.trial)] = (
            registry.iloc[index]["candidate_id"].astype(str).tolist()
        )
    for trial in range(nd_trials):
        orders[("neurodiscovery", trial)] = list(neurodiscovery_ids)
    return orders


def compile_closed_loop_orders(
    registry: pd.DataFrame,
    policies: Iterable[SearchPolicy],
    results: pd.DataFrame,
    *,
    nd_trials: int,
    batch_size: int,
    seed: int,
    alpha: float,
    overlay_dir: Path,
    prior_name: str = "kg_rank_v1",
) -> tuple[
    dict[tuple[str, int], list[str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    """Compile baselines, then run CS2 NeuroDiscovery as a real batch loop."""

    # Baselines remain fully outcome-blind and are frozen from public policies.
    orders: dict[tuple[str, int], list[str]] = {}
    for policy in policies:
        index = compile_policy_order(registry, policy)
        orders[(policy.method, policy.trial)] = (
            registry.iloc[index]["candidate_id"].astype(str).tolist()
        )

    indexed_results = results.set_index("candidate_id", drop=False)
    public = registry.copy()
    prior_score, _ = _build_neurodiscovery_prior(
        public,
        indexed_results,
        prior_name=prior_name,
    )
    public["score_neurodiscovery"] = prior_score

    outcomes = indexed_results.loc[public["candidate_id"]].reset_index(drop=True).copy()
    outcomes["validated"] = _family_chain_labels(outcomes, alpha)
    outcomes["feedback_status"] = np.where(
        outcomes["validated"], "supported", "inconclusive"
    )
    adapter = adapter_for("case2_pathway_mediation")
    config = ClosedLoopConfig(
        batch_size=max(1, int(batch_size)),
        warmup_batches=1,
        sampling_temperature=(
            0.005 if prior_name == "evidence_consensus_v2" else 0.01
        ),
        warmup_diversity_penalty=(
            1.0 if prior_name == "evidence_consensus_v2" else None
        ),
        max_feedback_rounds=max(1, int(math.ceil(len(public) / max(1, batch_size)))),
        feedback_horizon=len(public),
    )
    traces: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for trial in range(nd_trials):
        rng = np.random.default_rng(seed + 1009 * trial)
        order, trial_trace, overlay = run_closed_loop_order(
            public,
            outcomes,
            adapter=adapter,
            factor_fields=adapter.factor_fields,
            rng=rng,
            config=config,
            seed=seed,
            trial=trial,
            overlay_path=overlay_dir / f"seed_{seed}_trial_{trial:02d}.jsonl",
        )
        orders[("neurodiscovery", trial)] = (
            public.iloc[order]["candidate_id"].astype(str).tolist()
        )
        traces.extend(
            {"method": "neurodiscovery", "trial": trial, **row}
            for row in trial_trace
        )
        overlays.append(overlay)
    return orders, traces, overlays, outcomes


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels.astype(bool)
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(bool)
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def _global_chain_labels(results: pd.DataFrame, alpha: float) -> np.ndarray:
    if "sobel_q_global" in results:
        q_values = pd.to_numeric(results["sobel_q_global"], errors="coerce").to_numpy(
            float
        )
    else:
        p_values = pd.to_numeric(results["sobel_p"], errors="coerce").to_numpy(float)
        q_values = np.full(len(results), np.nan)
        finite = np.isfinite(p_values)
        q_values[finite] = multipletests(p_values[finite], method="fdr_bh")[1]
    return (
        (q_values < alpha)
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
    )


def _family_chain_labels(results: pd.DataFrame, alpha: float) -> np.ndarray:
    """Return complete mediation chains passing the prespecified family FDR."""

    if "sobel_q_family" not in results:
        return _global_chain_labels(results, alpha)
    return (
        pd.to_numeric(results["sobel_q_family"], errors="coerce")
        .lt(alpha)
        .to_numpy()
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy()
    )


def evaluate_orders(
    orders: dict[tuple[str, int], list[str]],
    results: pd.DataFrame,
    *,
    k_values: Iterable[int],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = results.set_index("candidate_id", drop=False)
    global_labels = pd.Series(
        _global_chain_labels(results, alpha), index=results["candidate_id"]
    )
    family_labels = pd.Series(
        _family_chain_labels(results, alpha), index=results["candidate_id"]
    )
    nominal_labels = pd.Series(
        pd.to_numeric(results["sobel_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["a_path_p"], errors="coerce").lt(alpha).to_numpy()
        & pd.to_numeric(results["b_path_p"], errors="coerce").lt(alpha).to_numpy(),
        index=results["candidate_id"],
    )
    metric_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    population = len(results)
    for (method, trial), candidate_ids in sorted(orders.items()):
        if len(candidate_ids) != population or len(set(candidate_ids)) != population:
            raise ValueError(
                f"{method} trial {trial} is not a full candidate permutation"
            )
        ordered = lookup.loc[candidate_ids]
        rank_scores = np.arange(population, 0, -1, dtype=float)
        labels = global_labels.loc[candidate_ids].to_numpy(bool)
        family = family_labels.loc[candidate_ids].to_numpy(bool)
        nominal = nominal_labels.loc[candidate_ids].to_numpy(bool)
        auroc = _binary_auroc(labels, rank_scores)
        auprc = _average_precision(labels, rank_scores)
        family_auroc = _binary_auroc(family, rank_scores)
        family_auprc = _average_precision(family, rank_scores)
        nominal_auroc = _binary_auroc(nominal, rank_scores)
        nominal_auprc = _average_precision(nominal, rank_scores)
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            ranking_rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "global_chain_hit": bool(labels[rank - 1]),
                    "family_fdr_chain_hit": bool(family[rank - 1]),
                    "nominal_chain_hit": bool(nominal[rank - 1]),
                }
            )
        for k in sorted(set(int(value) for value in k_values)):
            if k <= 0 or k > population:
                continue
            top = ordered.head(k)
            p_values = pd.to_numeric(top["sobel_p"], errors="coerce").to_numpy(float)
            q_values = np.full(k, np.nan)
            finite = np.isfinite(p_values)
            q_values[finite] = multipletests(p_values[finite], method="fdr_bh")[1]
            path_ok = (
                pd.to_numeric(top["a_path_p"], errors="coerce").lt(alpha).to_numpy()
                & pd.to_numeric(top["b_path_p"], errors="coerce").lt(alpha).to_numpy()
            )
            metric_rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "k": k,
                    "topk_fdr_chain_hits": int(((q_values < alpha) & path_ok).sum()),
                    "nominal_chain_hits": int(((p_values < alpha) & path_ok).sum()),
                    "family_fdr_chain_hits": int(family[:k].sum()),
                    "family_fdr_chain_recall": float(
                        family[:k].sum() / max(1, family.sum())
                    ),
                    "global_chain_hits": int(labels[:k].sum()),
                    "global_chain_recall": float(
                        labels[:k].sum() / max(1, labels.sum())
                    ),
                    "auroc_global_chain": auroc,
                    "auprc_global_chain": auprc,
                    "auroc_family_fdr_chain": family_auroc,
                    "auprc_family_fdr_chain": family_auprc,
                    "auroc_nominal_chain": nominal_auroc,
                    "auprc_nominal_chain": nominal_auprc,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(ranking_rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_columns = [
        "topk_fdr_chain_hits",
        "nominal_chain_hits",
        "family_fdr_chain_hits",
        "family_fdr_chain_recall",
        "global_chain_hits",
        "global_chain_recall",
        "auroc_global_chain",
        "auprc_global_chain",
        "auroc_family_fdr_chain",
        "auprc_family_fdr_chain",
        "auroc_nominal_chain",
        "auprc_nominal_chain",
    ]
    for (method, k), group in metrics.groupby(["method", "k"], sort=False):
        row: dict[str, Any] = {
            "method": method,
            "k": int(k),
            "n_trials": int(group["trial"].nunique()),
        }
        for column in value_columns:
            values = (
                pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            )
            if not len(values):
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_variance"] = (
                float(values.var(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _exact_sign_flip_p(differences: np.ndarray, *, greater: bool) -> float:
    """Exact paired randomization P value for a mean difference."""

    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = float(values.mean())
    null = np.empty(1 << len(values), dtype=float)
    for mask in range(len(null)):
        signs = np.fromiter(
            (1.0 if mask & (1 << bit) else -1.0 for bit in range(len(values))),
            dtype=float,
            count=len(values),
        )
        null[mask] = float(np.mean(signs * values))
    tolerance = 1e-12
    if greater:
        return float(np.mean(null >= observed - tolerance))
    return float(np.mean(null <= observed + tolerance))


def _holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_method_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare cumulative family-FDR hits using paired trial identities."""

    required = {"method", "trial", "k", "family_fdr_chain_hits"}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"missing paired-comparison columns: {sorted(missing)}")
    target = metrics.loc[metrics["method"].eq("neurodiscovery")].copy()
    if target.empty:
        raise ValueError("NeuroDiscovery rows are required for paired comparisons")
    methods = sorted(set(metrics["method"].astype(str)) - {"neurodiscovery"})
    max_k = int(pd.to_numeric(metrics["k"], errors="raise").max())
    rows: list[dict[str, Any]] = []
    for k in sorted(pd.to_numeric(target["k"], errors="raise").astype(int).unique()):
        target_k = target.loc[target["k"].astype(int).eq(k), ["trial", "family_fdr_chain_hits"]]
        target_k = target_k.rename(columns={"family_fdr_chain_hits": "target_hits"})
        endpoint_rows: list[dict[str, Any]] = []
        for method in methods:
            baseline = metrics.loc[
                metrics["method"].eq(method) & metrics["k"].astype(int).eq(k),
                ["trial", "family_fdr_chain_hits"],
            ].rename(columns={"family_fdr_chain_hits": "baseline_hits"})
            paired = target_k.merge(baseline, on="trial", validate="one_to_one")
            differences = (
                paired["target_hits"].to_numpy(float)
                - paired["baseline_hits"].to_numpy(float)
            )
            endpoint_rows.append(
                {
                    "metric": "family_fdr_chain_hits",
                    "k": int(k),
                    "target": "neurodiscovery",
                    "baseline": method,
                    "n_pairs": int(len(differences)),
                    "target_mean": float(paired["target_hits"].mean()),
                    "baseline_mean": float(paired["baseline_hits"].mean()),
                    "mean_difference": float(differences.mean()),
                    "difference_variance": (
                        float(differences.var(ddof=1)) if len(differences) > 1 else 0.0
                    ),
                    "p_neurodiscovery_greater_exact_sign_flip": _exact_sign_flip_p(
                        differences, greater=True
                    ),
                    "p_baseline_greater_exact_sign_flip": _exact_sign_flip_p(
                        differences, greater=False
                    ),
                    "full_pool_sanity": bool(k >= max_k),
                }
            )
        forward = _holm_adjust(
            [row["p_neurodiscovery_greater_exact_sign_flip"] for row in endpoint_rows]
        )
        reverse = _holm_adjust(
            [row["p_baseline_greater_exact_sign_flip"] for row in endpoint_rows]
        )
        for row, forward_p, reverse_p in zip(endpoint_rows, forward, reverse):
            row["p_neurodiscovery_greater_holm_within_k"] = forward_p
            row["p_baseline_greater_holm_within_k"] = reverse_p
            rows.append(row)
    return pd.DataFrame(rows)


def audit_all_policy_trials(
    registry: pd.DataFrame,
    policies: Iterable[SearchPolicy],
    *,
    top_ks: tuple[int, ...] = (10, 50, 100),
    collapse_threshold: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit every trial shared by all adapted methods."""

    policies = list(policies)
    trials_by_method: dict[str, set[int]] = {}
    for policy in policies:
        trials_by_method.setdefault(policy.method, set()).add(policy.trial)
    if len(trials_by_method) < 2:
        raise ValueError("policy-independence audit requires at least two methods")
    common_trials = sorted(set.intersection(*trials_by_method.values()))
    if not common_trials:
        raise ValueError("adapted methods have no common trial")

    frames: list[pd.DataFrame] = []
    collapsed_pairs: list[dict[str, Any]] = []
    for trial in common_trials:
        frame, summary = audit_policy_independence(
            registry,
            policies,
            compile_order=compile_policy_order,
            compiler_version=COMPILER_VERSION,
            trial=trial,
            top_ks=top_ks,
            collapse_threshold=collapse_threshold,
        )
        frames.append(frame)
        collapsed_pairs.extend(
            {"trial": trial, **pair} for pair in summary["collapsed_pairs"]
        )

    return pd.concat(frames, ignore_index=True), {
        "compiler_version": COMPILER_VERSION,
        "candidate_count": len(registry),
        "methods": sorted(trials_by_method),
        "trials": common_trials,
        "top_ks": sorted(
            {min(int(value), len(registry)) for value in top_ks if int(value) > 0}
        ),
        "collapse_threshold": collapse_threshold,
        "collapsed_pairs": collapsed_pairs,
        "passed": not collapsed_pairs,
    }


def main() -> int:
    args = parse_args()
    canonical_release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id="case2_pathway_mediation",
    )
    results_path = args.results or _latest(
        DEFAULT_RESULTS_PARENT, "kg_ranked_longitudinal_results.parquet"
    )
    policies_path = args.policies or _latest(
        DEFAULT_RESULTS_PARENT,
        "official_baselines_gpt55_high/case2_search_policies.jsonl",
    )
    out_dir = args.out_dir or policies_path.parent / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_release_manifest(out_dir / "canonical_kg_release.json", canonical_release)

    results = _attach_candidate_ids(pd.read_parquet(results_path))
    registry = build_public_registry(results)
    if bool(args.score_components) != bool(args.score_components_manifest):
        raise ValueError(
            "--score-components and --score-components-manifest must be provided together"
        )
    if args.score_components:
        registry, score_component_audit = load_score_component_bundle(
            registry,
            table_path=args.score_components,
            manifest_path=args.score_components_manifest,
        )
    else:
        score_component_audit = embedded_score_component_audit(registry)
    policies = load_policies(policies_path)
    indexed_results = results.set_index("candidate_id", drop=False)
    nd_prior_score, nd_prior_audit = _build_neurodiscovery_prior(
        registry,
        indexed_results,
        prior_name=args.nd_prior,
    )
    orders, nd_traces, nd_overlays, nd_outcomes = compile_closed_loop_orders(
        registry,
        policies,
        results,
        nd_trials=args.nd_trials,
        batch_size=args.nd_batch_size,
        seed=args.seed,
        alpha=args.alpha,
        overlay_dir=out_dir / "experimental_overlays",
        prior_name=args.nd_prior,
    )
    metrics, rankings = evaluate_orders(
        orders,
        results,
        k_values=args.k,
        alpha=args.alpha,
    )
    aggregate = aggregate_metrics(metrics)
    paired = paired_method_comparisons(metrics)
    metrics.to_csv(out_dir / "case2_method_metrics_by_trial.csv", index=False)
    aggregate.to_csv(out_dir / "case2_method_metrics_aggregate.csv", index=False)
    paired.to_csv(out_dir / "case2_paired_comparisons.csv", index=False)
    rankings.to_parquet(
        out_dir / "case2_compiled_rankings.parquet", index=False, compression="zstd"
    )
    with (out_dir / "neurodiscovery_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in nd_traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    experimental_delta = write_kg_delta(
        registry.assign(
            score_neurodiscovery=nd_prior_score
        ),
        nd_outcomes,
        nd_traces,
        path=out_dir / "experimental_kg_delta.jsonl",
        task="case2_pathway_mediation",
        overlay_manifests=nd_overlays,
        factor_fields=adapter_for("case2_pathway_mediation").factor_fields,
    )

    policy_methods = sorted({policy.method for policy in policies})
    if len(policy_methods) >= 2:
        audit_frame, audit_summary = audit_all_policy_trials(registry, policies)
    else:
        audit_frame = pd.DataFrame()
        audit_summary = {
            "compiler_version": COMPILER_VERSION,
            "methods": policy_methods,
            "passed": True,
            "not_applicable": True,
            "collapsed_pairs": [],
        }
    audit_frame.to_csv(out_dir / "case2_policy_independence_audit.csv", index=False)
    (out_dir / "case2_policy_independence_audit.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )
    if args.fail_on_collapse and not audit_summary["passed"]:
        raise ValueError(
            "Case 2 baseline policies collapsed to indistinguishable outputs"
        )

    manifest = {
        "results": str(results_path),
        "policies": str(policies_path),
        "canonical_kg_release": canonical_release,
        "case_study_membership": {
            "schema_version": "case_study_membership.v2",
            "case_study_id": "case2_pathway_mediation",
            "canonical_claim_field": "claim_case_study_ids",
        },
        "candidate_count": len(registry),
        "methods": sorted(metrics["method"].unique()),
        "trials_by_method": metrics.groupby("method")["trial"].nunique().to_dict(),
        "k_values": sorted(metrics["k"].unique().astype(int).tolist()),
        "compiler_version": COMPILER_VERSION,
        "ranking_frozen_before_result_merge": False,
        "baseline_rankings_frozen_before_result_access": True,
        "neurodiscovery_outcomes_revealed_batchwise_only": True,
        "neurodiscovery_closed_loop": {
            "batch_size": args.nd_batch_size,
            "prior": nd_prior_audit,
            "seed": args.seed,
            "trials": args.nd_trials,
            "experimental_kg_delta": experimental_delta,
            "complete_chain_required": True,
            "unsupported_chain_status": "inconclusive",
            "feedback_definition": (
                "Sobel family-level BH q<0.05 and both mediation paths P<0.05"
            ),
        },
        "score_component_bundle": score_component_audit,
        "topk_fdr_definition": (
            "BH q<0.05 within frozen Top-K and both mediation paths P<0.05"
        ),
        "family_fdr_definition": (
            "Prespecified exposure-outcome imaging-family BH q<0.05 and both "
            "mediation paths P<0.05"
        ),
        "primary_ranking_endpoint": {
            "metric": "family_fdr_chain_hits",
            "recall_metric": "family_fdr_chain_recall",
            "cumulative_over_frozen_labels": True,
            "topk_fdr_chain_hits_role": (
                "sensitivity analysis only; BH is recomputed inside each Top-K "
                "prefix and the count is therefore not a cumulative discovery curve"
            ),
        },
        "paired_comparisons": {
            "path": "case2_paired_comparisons.csv",
            "test": "exact paired sign-flip over matched trial identities",
            "multiplicity": "Holm correction across baselines within each K",
            "primary_metric": "family_fdr_chain_hits",
        },
        "nominal_rank_metric_definition": (
            "Exploratory label: Sobel P<0.05 and both mediation paths P<0.05"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-12 02:45 HKT
