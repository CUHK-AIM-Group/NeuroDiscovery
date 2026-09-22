"""Tune generic NeuroDiscovery policies without opening external outcomes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import (
    ClosedLoopConfig,
    align_outcome_array,
    closed_loop_neurodiscovery_order,
    validate_hidden_outcomes,
    validate_public_registry,
)
from core.scripts.case_study_neurodiscovery_policy import (
    DESIGN_PRIOR_COLUMN,
    KG_SCORE_COLUMNS,
    NeuroDiscoveryPolicy,
    apply_neurodiscovery_policy,
    write_neurodiscovery_policy,
)
from core.scripts.case_study_candidate_tables import score_public_candidates
from core.scripts.case_study_closed_loop_specs import TASK_PROTOCOLS


DEFAULT_TASKS = (
    "biomarker_discovery",
    "differential_diagnosis",
    "disease_subtyping",
    "progression_prediction",
    "connectome_behavior",
    "brain_age",
    "imaging_genetics",
    "prognosis",
)
DEFAULT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v2"
    r"\20260810_kg737C884E_seeded10_r2"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_release_fingerprints(kg_path: Path | None) -> dict[str, Any] | None:
    if kg_path is None:
        return None
    files = {
        "knowledge_graph": kg_path,
        "extracted_claims": kg_path.parent / "extracted_claims.jsonl",
        "current_state": kg_path.parent / "CURRENT_STATE.json",
    }
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in files.items()
        if path.is_file()
    }


def stable_group_folds(
    public: pd.DataFrame,
    factor_fields: Sequence[str],
    *,
    n_folds: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    excluded = {"model"}
    if "roi_index" in factor_fields:
        excluded.update({"feature", "feature_family"})
    group_fields = tuple(field for field in factor_fields if field not in excluded)
    if not group_fields:
        group_fields = tuple(factor_fields)
    folds = []
    for values in (
        public.loc[:, group_fields]
        .fillna("")
        .astype(str)
        .itertuples(index=False, name=None)
    ):
        key = "\x1f".join(values)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        folds.append(int.from_bytes(digest[:8], "big") % n_folds)
    return np.asarray(folds, dtype=np.int64), group_fields


def task_budgets(
    manifest: dict[str, Any],
    n: int,
    max_horizon: int,
    *,
    include_early_budget: bool = False,
) -> tuple[int, ...]:
    registered = manifest.get("provenance", {}).get("protocol", {}).get("budgets", [])
    if not registered:
        task = str(manifest.get("task", ""))
        protocol = TASK_PROTOCOLS.get(task)
        registered = protocol.budgets if protocol is not None else ()
    values = {int(value) for value in registered if int(value) > 0}
    if not values:
        values = {5, 10, 25, 50}
    horizon = min(n, max_horizon)
    clipped = sorted({min(value, horizon) for value in values if value <= horizon})
    if not clipped:
        clipped = [horizon]
    if include_early_budget:
        clipped = sorted({*clipped, min(10, horizon)})
    return tuple(clipped)


def recovery_objective(
    order: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    budgets: Sequence[int],
    *,
    filter_order: bool,
) -> tuple[float, dict[str, int | float]]:
    ranked = np.asarray(order, dtype=np.int64)
    if filter_order:
        ranked = ranked[mask[ranked]]
    target = labels & mask
    total = int(target.sum())
    row: dict[str, int | float] = {"target_total": total}
    if total == 0:
        for budget in budgets:
            row[f"hits_at_{budget}"] = 0
            row[f"oracle_fraction_at_{budget}"] = 0.0
        return 0.0, row
    cumulative = np.cumsum(target[ranked])
    fractions = []
    for budget in budgets:
        effective = min(int(budget), len(ranked))
        hits = int(cumulative[effective - 1]) if effective else 0
        oracle = min(effective, total)
        fraction = float(hits / max(oracle, 1))
        row[f"hits_at_{budget}"] = hits
        row[f"oracle_fraction_at_{budget}"] = fraction
        fractions.append(fraction)
    weights = np.asarray([1.0 / np.sqrt(value) for value in budgets], dtype=float)
    weights /= weights.sum()
    objective = float(np.dot(weights, np.asarray(fractions, dtype=float)))
    row["objective"] = objective
    for recall_target in (0.1, 0.2, 0.5, 0.8, 1.0):
        hits_needed = max(1, int(np.ceil(total * recall_target)))
        reached = np.flatnonzero(cumulative >= hits_needed)
        experiments_required = int(reached[0] + 1) if len(reached) else len(ranked)
        suffix = int(round(100 * recall_target))
        row[f"experiments_for_recall_{suffix}"] = experiments_required
        row[f"recall_efficiency_{suffix}"] = float(
            hits_needed / max(experiments_required, 1)
        )
    return objective, row


def policy_regularization(
    policy: NeuroDiscoveryPolicy,
) -> tuple[int, int, float, int]:
    """Apply outcome-independent preferences only after exact CV ties."""

    semantic_projection = {
        "all_factors": 0,
        "relation_endpoints": 1,
        "generalizable_factors": 2,
    }[policy.feedback_projection]
    reference = {
        "feedback_weight": 2.00,
        "pair_feedback_weight": 0.15,
        "exploration_weight": 0.03,
        "inconclusive_search_failure_weight": 0.00,
        "diversity_penalty": 0.08,
    }
    distance = sum(
        abs(float(getattr(policy, name)) - target) / max(target, 0.03)
        for name, target in reference.items()
    )
    relation_aware_score = int(policy.score_family == "relation_aware")
    return (
        semantic_projection,
        relation_aware_score,
        -float(distance),
        -int(policy.batch_size),
    )


def select_top_static_policies(
    static_screen: pd.DataFrame,
    limit: int,
) -> list[NeuroDiscoveryPolicy]:
    """Retain the best policy from every score family before filling the slate."""

    if limit < 1:
        raise ValueError("top-static limit must be positive")
    parsed = [
        NeuroDiscoveryPolicy.from_dict(json.loads(value))
        for value in static_screen["policy_json"]
    ]
    selected: list[NeuroDiscoveryPolicy] = []
    seen: set[str] = set()
    for family in ("relation_aware", "legacy"):
        policy = next((item for item in parsed if item.score_family == family), None)
        if policy is not None and policy.policy_id() not in seen:
            selected.append(policy)
            seen.add(policy.policy_id())
    for policy in parsed:
        if policy.policy_id() in seen:
            continue
        selected.append(policy)
        seen.add(policy.policy_id())
        if len(selected) >= limit:
            break
    return selected[:limit]


def static_policy_grid(
    task: str,
    *,
    design_prior_available: bool = False,
) -> list[NeuroDiscoveryPolicy]:
    design_weights = (
        (0.0, 0.05, 0.10, 0.15)
        if task in {"brain_age", "connectome_behavior"} and design_prior_available
        else (0.0,)
    )
    policies = [
        replace(NeuroDiscoveryPolicy(task=task), design_prior_weight=weight)
        for weight in design_weights
    ]
    for score_family in ("legacy", "relation_aware"):
        for scoped_fraction in (0.00, 0.50, 0.70, 0.90, 1.00):
            for node_fraction in (0.00, 0.20, 0.40, 0.60, 0.80, 1.00):
                for design_weight in design_weights:
                    policies.append(
                        NeuroDiscoveryPolicy(
                            task=task,
                            score_family=score_family,
                            global_node_weight=(1.0 - scoped_fraction) * node_fraction,
                            global_pair_weight=(1.0 - scoped_fraction)
                            * (1.0 - node_fraction),
                            scoped_node_weight=scoped_fraction * node_fraction,
                            scoped_pair_weight=scoped_fraction * (1.0 - node_fraction),
                            design_prior_weight=design_weight,
                            tie_break_weight=1e-9,
                        )
                    )
    unique = {policy.policy_id(): policy for policy in policies}
    return list(unique.values())


def static_control_policy(
    policy: NeuroDiscoveryPolicy,
) -> NeuroDiscoveryPolicy:
    """Encode the exact static ranking as a valid closed-loop control."""

    return replace(
        policy,
        batch_size=1,
        warmup_batches=1,
        sampling_temperature=0.0,
        feedback_weight=0.0,
        pair_feedback_weight=0.0,
        exploration_weight=0.0,
        max_feedback_rounds=1,
        feedback_horizon=1,
        inconclusive_search_failure_weight=0.0,
        feedback_projection="relation_endpoints",
        feedback_batch_schedule="fixed",
    )


def front_loaded_warmup_batches(
    *,
    horizon: int,
    batch_size: int,
    max_feedback_rounds: int,
    feedback_start_budget: int,
) -> int:
    """Translate an interpretable feedback-start rank to scheduled batches."""

    planned_rounds = min(
        int(max_feedback_rounds),
        int(np.ceil(int(horizon) / int(batch_size))),
    )
    target_budget = min(int(horizon), max(1, int(feedback_start_budget)))
    for completed_rounds in range(1, planned_rounds + 1):
        progress = float(completed_rounds) / planned_rounds
        selected = min(
            int(horizon),
            max(
                completed_rounds * int(batch_size),
                int(np.ceil(int(horizon) * progress**4)),
            ),
        )
        if selected >= target_budget:
            return completed_rounds
    return planned_rounds


def dynamic_policy_grid(
    static_policies: Iterable[NeuroDiscoveryPolicy],
    *,
    smallest_budget: int,
    horizon: int,
    quality_endpoint: bool = False,
) -> list[NeuroDiscoveryPolicy]:
    if horizon >= 50_000:
        batch_size = 2
        max_rounds = 128
        profiles = (
            (0.005, 0.60, 0.35, 0.08),
            (0.025, 2.00, 0.15, 0.03),
            (0.050, 2.00, 0.15, 0.03),
            (0.125, 0.60, 0.35, 0.08),
        )
        policies = []
        for static in static_policies:
            policies.append(static_control_policy(static))
            for start_fraction, feedback, pair_feedback, exploration in profiles:
                start_budget = max(
                    int(smallest_budget),
                    int(round(float(horizon) * start_fraction)),
                )
                policies.append(
                    replace(
                        static,
                        batch_size=batch_size,
                        warmup_batches=front_loaded_warmup_batches(
                            horizon=horizon,
                            batch_size=batch_size,
                            max_feedback_rounds=max_rounds,
                            feedback_start_budget=start_budget,
                        ),
                        sampling_temperature=1e-6,
                        feedback_weight=feedback,
                        pair_feedback_weight=pair_feedback,
                        exploration_weight=exploration,
                        inconclusive_search_failure_weight=0.0,
                        diversity_penalty=0.08,
                        warmup_diversity_penalty=10.0,
                        max_feedback_rounds=max_rounds,
                        feedback_horizon=horizon,
                        feedback_projection="relation_endpoints",
                        feedback_batch_schedule="front_loaded",
                    )
                )
        return list({policy.policy_id(): policy for policy in policies}.values())

    if quality_endpoint:
        policies = []
        for static in static_policies:
            default_feedback_model = (
                "ridge_surrogate"
                if static.task == "brain_age"
                else "hierarchical_utility"
            )
            profiles = (
                (
                    (0.00025, 4.0, 0.60, 0.02, 0, False),
                    (0.00050, 2.0, 0.60, 0.02, 0, False),
                    (0.00100, 2.0, 0.60, 0.03, 0, False),
                    (0.00200, 1.0, 0.60, 0.03, 0, False),
                    (0.00025, 4.0, 0.30, 0.08, 0, False),
                    (0.00050, 2.0, 0.30, 0.08, 0, False),
                    (0.00050, 2.0, 0.60, 0.08, 0, False),
                    (0.00050, 2.0, 1.20, 0.05, 0, False),
                    (0.00050, 2.0, 1.20, 0.10, 0, False),
                    (0.00100, 1.0, 1.20, 0.05, 0, False),
                    (0.00200, 1.0, 1.20, 0.05, 0, False),
                    (0.00050, 4.0, 0.30, 0.15, 0, False),
                    (0.00010, 2.0, 0.60, 0.03, 1, True),
                    (0.00025, 2.0, 1.20, 0.05, 1, True),
                    (0.00050, 2.0, 1.20, 0.05, 1, True),
                    (0.00100, 1.0, 1.20, 0.05, 1, True),
                    # Factor-UCB retains the ridge quality predictor while giving
                    # candidates with an under-sampled factor level an uncertainty bonus.
                    (
                        0.00025,
                        2.0,
                        0.60,
                        0.05,
                        0,
                        False,
                        1,
                        "factor_ucb_ridge_surrogate",
                    ),
                    (
                        0.00050,
                        2.0,
                        1.20,
                        0.10,
                        0,
                        False,
                        1,
                        "factor_ucb_ridge_surrogate",
                    ),
                    (
                        0.00050,
                        1.0,
                        1.20,
                        0.20,
                        0,
                        False,
                        1,
                        "factor_ucb_ridge_surrogate",
                    ),
                    (
                        0.00100,
                        1.0,
                        1.20,
                        0.30,
                        0,
                        False,
                        1,
                        "factor_ucb_ridge_surrogate",
                    ),
                    (
                        0.00100,
                        1.0,
                        1.20,
                        0.10,
                        0,
                        False,
                        1,
                        "factor_ucb_ridge_surrogate",
                    ),
                    (
                        0.00200,
                        1.0,
                        1.20,
                        0.20,
                        0,
                        False,
                        1,
                        "factor_ucb_ridge_surrogate",
                    ),
                )
                if static.task == "brain_age"
                else (
                    (0.00050, 1.0, 0.60, 0.03, 0, False),
                    (0.00100, 1.0, 0.60, 0.03, 0, False),
                    (0.00200, 1.0, 0.60, 0.03, 0, False),
                )
            )
            for profile in profiles:
                (
                    temperature,
                    ridge_alpha,
                    feedback,
                    exploration,
                    min_supported,
                    preserve_static,
                ) = profile[:6]
                warmup_batches = int(profile[6]) if len(profile) > 6 else 1
                feedback_model = (
                    str(profile[7]) if len(profile) > 7 else default_feedback_model
                )
                policies.append(
                    replace(
                        static,
                        batch_size=1,
                        warmup_batches=warmup_batches,
                        sampling_temperature=temperature,
                        feedback_weight=feedback,
                        pair_feedback_weight=0.15,
                        exploration_weight=exploration,
                        inconclusive_search_failure_weight=0.0,
                        diversity_penalty=0.08,
                        max_feedback_rounds=max(128, horizon),
                        feedback_horizon=horizon,
                        min_supported_before_feedback=min_supported,
                        feedback_projection="all_factors",
                        feedback_model=feedback_model,
                        surrogate_ridge_alpha=ridge_alpha,
                        preserve_static_until_informative_feedback=preserve_static,
                    )
                )
        return list({policy.policy_id(): policy for policy in policies}.values())

    base_batch = max(2, min(64, int(np.ceil(smallest_budget / 4))))
    batches = sorted({max(2, base_batch // 2), base_batch, min(64, base_batch * 2)})
    profiles = (
        (0.30, 0.15, 0.03, 0.00),
        (0.60, 0.35, 0.08, 0.00),
        (2.00, 0.15, 0.03, 0.00),
        (8.00, 0.25, 0.01, 0.00),
    )
    policies = [static_control_policy(static) for static in static_policies]
    failure_weights = (0.0,)
    diversity_values = (0.08, 0.50, 2.00, 10.00)
    for static in static_policies:
        for batch_size in batches:
            for diversity_relaxation in diversity_values:
                for (
                    feedback,
                    pair_feedback,
                    exploration,
                    inconclusive_failure,
                ) in profiles:
                    for inconclusive_failure in failure_weights:
                        for feedback_projection in (
                            "generalizable_factors",
                            "all_factors",
                            "relation_endpoints",
                        ):
                            policies.append(
                                replace(
                                    static,
                                    batch_size=batch_size,
                                    sampling_temperature=0.002,
                                    feedback_weight=feedback,
                                    pair_feedback_weight=pair_feedback,
                                    exploration_weight=exploration,
                                    inconclusive_search_failure_weight=(
                                        inconclusive_failure
                                    ),
                                    diversity_penalty=diversity_relaxation,
                                    max_feedback_rounds=max(
                                        128, int(np.ceil(horizon / batch_size))
                                    ),
                                    feedback_horizon=horizon,
                                    feedback_projection=feedback_projection,
                                )
                            )
    return list({policy.policy_id(): policy for policy in policies}.values())


def load_task_bundle(
    root: Path, task: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], tuple[str, ...]]:
    table_dir = root / task / "tables"
    manifest_path = table_dir / "table_manifest.json"
    public_path = table_dir / "public_candidates.csv"
    internal_path = table_dir / "internal_outcomes.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    factor_fields = tuple(manifest["factor_fields"])
    public = validate_public_registry(
        pd.read_csv(public_path, low_memory=False), factor_fields=factor_fields
    )
    internal = validate_hidden_outcomes(
        public, pd.read_csv(internal_path, low_memory=False)
    )
    manifest["tuning_inputs"] = {
        "public_candidates": {
            "path": str(public_path.resolve()),
            "sha256": sha256_file(public_path),
        },
        "internal_outcomes": {
            "path": str(internal_path.resolve()),
            "sha256": sha256_file(internal_path),
        },
        "external_outcomes_opened": False,
    }
    return public, internal, manifest, factor_fields


def tune_task(args: argparse.Namespace, task: str) -> dict[str, Any]:
    public, internal, table_manifest, factor_fields = load_task_bundle(
        args.run_root, task
    )
    labels = align_outcome_array(public, internal, "validated")
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    rescoring_audit: dict[str, Any] | None = None
    if args.kg is not None:
        semantic_fields = tuple(
            table_manifest.get("kg_scoring", {}).get("semantic_fields") or factor_fields
        )
        score_inputs = public.drop(
            columns=[
                column
                for column in (*KG_SCORE_COLUMNS, "score_neurodiscovery")
                if column in public
            ]
        )
        public, rescoring_audit = score_public_candidates(
            score_inputs,
            semantic_fields=semantic_fields,
            case_study_id=task,
            kg_path=args.kg,
            seed=args.seed,
        )
        public = validate_public_registry(public, factor_fields=factor_fields)
        rescored_path = task_dir / "rescored_public_candidates.csv"
        public.to_csv(rescored_path, index=False)
        rescoring_audit["output_path"] = str(rescored_path.resolve())
        rescoring_audit["output_sha256"] = sha256_file(rescored_path)
    quality_endpoint = bool(table_manifest.get("quality_endpoint"))
    budgets = task_budgets(
        table_manifest,
        len(public),
        args.max_horizon,
        include_early_budget=quality_endpoint,
    )
    folds, group_fields = stable_group_folds(public, factor_fields, n_folds=args.folds)
    split = (
        pd.DataFrame(
            {
                "fold": folds,
                "validated": labels,
            }
        )
        .groupby("fold")["validated"]
        .agg(["count", "sum"])
    )
    split.to_csv(task_dir / "split_summary.csv")
    base_manifest = {
        "schema_version": "case-study-neurodiscovery-tuning.v1",
        "created_at": utc_now(),
        "task": task,
        "candidate_count": int(len(public)),
        "validated_count": int(labels.sum()),
        "factor_fields": list(factor_fields),
        "group_fields": list(group_fields),
        "folds": int(args.folds),
        "holdout_fold": int(args.holdout_fold),
        "search_seed_base": int(args.seed),
        "budgets": list(budgets),
        "external_outcomes_opened": False,
        "holdout_outcomes_evaluated_during_selection": False,
        "holdout_feedback_available_during_selection": False,
        "table_bundle": table_manifest["tuning_inputs"],
        "outcome_blind_rescoring": rescoring_audit,
    }
    if labels.sum() == 0 or labels.sum() == len(labels):
        reason = (
            "no_positive_internal_outcomes"
            if labels.sum() == 0
            else "all_positive_internal_outcomes"
        )
        manifest = {**base_manifest, "status": "not_tunable", "reason": reason}
        (task_dir / "tuning_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    development_folds = [
        fold for fold in range(args.folds) if fold != args.holdout_fold
    ]
    static_rows: list[dict[str, Any]] = []
    policies = static_policy_grid(
        task,
        design_prior_available=DESIGN_PRIOR_COLUMN in public,
    )
    for policy in policies:
        scored, _ = apply_neurodiscovery_policy(public, policy)
        kg_only, _ = apply_neurodiscovery_policy(
            public, replace(policy, tie_break_weight=0.0)
        )
        kg_signal_std = float(kg_only["score_neurodiscovery"].std(ddof=0))
        if kg_signal_std <= 1e-12:
            continue
        scores = scored["score_neurodiscovery"].to_numpy(float)
        order = np.lexsort((np.arange(len(scored)), -scores))
        fold_scores = []
        row: dict[str, Any] = {
            "policy_id": policy.policy_id(),
            "policy_json": json.dumps(policy.to_dict(), sort_keys=True),
            "kg_signal_std": kg_signal_std,
        }
        for fold in development_folds:
            mask = folds == fold
            objective, metrics = recovery_objective(
                order, labels, mask, budgets, filter_order=False
            )
            fold_scores.append(objective)
            row[f"fold_{fold}_objective"] = objective
            for key, value in metrics.items():
                row[f"fold_{fold}_{key}"] = value
        development_objective, development_metrics = recovery_objective(
            order,
            labels,
            folds != args.holdout_fold,
            budgets,
            filter_order=False,
        )
        row["development_mean"] = development_objective
        row["development_fold_mean"] = float(np.mean(fold_scores))
        row["development_min"] = float(np.min(fold_scores))
        row["robust_objective"] = (
            0.75 * row["development_mean"] + 0.25 * row["development_min"]
        )
        for key, value in development_metrics.items():
            row[f"development_{key}"] = value
        static_rows.append(row)
    if not static_rows:
        raise ValueError(f"{task} has no non-degenerate static KG policy")
    static_screen = pd.DataFrame(static_rows)
    static_screen["relation_aware_preference"] = static_screen["policy_json"].map(
        lambda value: int(
            NeuroDiscoveryPolicy.from_dict(json.loads(value)).score_family
            == "relation_aware"
        )
    )
    static_screen = static_screen.sort_values(
        ["robust_objective", "development_mean", "relation_aware_preference"],
        ascending=False,
    )
    static_screen.to_csv(task_dir / "static_screen.csv", index=False)
    top_static = select_top_static_policies(static_screen, args.top_static)
    if args.static_only:
        selected = top_static[0]
        write_neurodiscovery_policy(task_dir / "selected_policy.json", selected)
        manifest = {
            **base_manifest,
            "status": "static_complete",
            "selected_policy_id": selected.policy_id(),
            "selected_policy": selected.to_dict(),
        }
        (task_dir / "tuning_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    horizon = min(
        len(public) if quality_endpoint else max(budgets),
        len(public),
        args.max_horizon,
    )
    dynamic_policies = dynamic_policy_grid(
        top_static,
        smallest_budget=min(budgets),
        horizon=horizon,
        quality_endpoint=quality_endpoint,
    )
    feedback = internal.copy()
    holdout_ids = set(
        public.loc[folds == args.holdout_fold, "candidate_id"].astype(str)
    )
    holdout_rows = feedback["candidate_id"].astype(str).isin(holdout_ids)
    feedback["feedback_available"] = True
    feedback.loc[holdout_rows, "feedback_available"] = False
    feedback.loc[holdout_rows, "validated"] = False
    feedback.loc[holdout_rows, "feedback_status"] = "inconclusive"
    if "feedback_utility" in feedback:
        feedback.loc[holdout_rows, "feedback_utility"] = np.nan
    development_mask = folds != args.holdout_fold
    dynamic_rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(dynamic_policies, 1):
        scored, _ = apply_neurodiscovery_policy(public, policy)
        config = ClosedLoopConfig(**policy.closed_loop_kwargs(default_horizon=horizon))
        for seed_index in range(args.search_seeds):
            seed = args.seed + 1009 * seed_index
            order, _trace = closed_loop_neurodiscovery_order(
                scored,
                labels,
                factor_fields=factor_fields,
                rng=np.random.default_rng(seed),
                config=config,
                task=task,
                seed=seed,
                trial=seed_index,
                outcomes=feedback,
                audit_records=False,
                collect_trace=False,
            )
            dev_objective, dev_metrics = recovery_objective(
                order,
                labels,
                development_mask,
                budgets,
                filter_order=False,
            )
            dynamic_rows.append(
                {
                    "policy_id": policy.policy_id(),
                    "policy_json": json.dumps(policy.to_dict(), sort_keys=True),
                    "seed": seed,
                    "development_objective": dev_objective,
                    **{
                        f"development_{key}": value
                        for key, value in dev_metrics.items()
                    },
                }
            )
        print(
            f"[{task}] dynamic {policy_index}/{len(dynamic_policies)}",
            flush=True,
        )
    dynamic_trials = pd.DataFrame(dynamic_rows)
    dynamic_trials.to_csv(task_dir / "dynamic_trials.csv", index=False)
    summary_rows = []
    early_budget = min(budgets)
    early_column = f"development_oracle_fraction_at_{early_budget}"
    recall_efficiency_columns = [
        f"development_recall_efficiency_{suffix}"
        for suffix in (10, 20, 50, 80, 100)
    ]
    for policy_id, group in dynamic_trials.groupby("policy_id", sort=False):
        values = group["development_objective"].to_numpy(float)
        early_values = group[early_column].to_numpy(float)
        recall_efficiency = group[recall_efficiency_columns].mean(axis=1).to_numpy(
            float
        )
        summary_rows.append(
            {
                "policy_id": policy_id,
                "policy_json": group.iloc[0]["policy_json"],
                "development_mean": float(values.mean()),
                "development_variance": float(values.var(ddof=1))
                if len(values) > 1
                else 0.0,
                "development_std": float(values.std(ddof=1))
                if len(values) > 1
                else 0.0,
                "development_min": float(values.min()),
                "robust_objective": float(0.75 * values.mean() + 0.25 * values.min()),
                "early_budget": int(early_budget),
                "early_recovery_mean": float(early_values.mean()),
                "early_recovery_std": float(early_values.std(ddof=1))
                if len(early_values) > 1
                else 0.0,
                "recall_efficiency_mean": float(recall_efficiency.mean()),
                "recall_efficiency_std": float(recall_efficiency.std(ddof=1))
                if len(recall_efficiency) > 1
                else 0.0,
            }
        )
    dynamic_summary = pd.DataFrame(summary_rows)
    dynamic_summary["selection_objective"] = dynamic_summary["robust_objective"]
    if quality_endpoint:
        curve_lcb = dynamic_summary["robust_objective"] - (
            0.25 * dynamic_summary["development_std"]
        )
        early_lcb = dynamic_summary["early_recovery_mean"] - (
            0.25 * dynamic_summary["early_recovery_std"]
        )
        recall_lcb = dynamic_summary["recall_efficiency_mean"] - (
            0.25 * dynamic_summary["recall_efficiency_std"]
        )
        dynamic_summary["selection_objective"] = (
            0.70 * curve_lcb + 0.15 * early_lcb + 0.15 * recall_lcb
        )
    regularization = dynamic_summary["policy_json"].map(
        lambda value: policy_regularization(
            NeuroDiscoveryPolicy.from_dict(json.loads(value))
        )
    )
    dynamic_summary["semantic_projection_preference"] = regularization.map(
        lambda value: value[0]
    )
    dynamic_summary["relation_aware_score_preference"] = regularization.map(
        lambda value: value[1]
    )
    dynamic_summary["shared_profile_regularization"] = regularization.map(
        lambda value: value[2]
    )
    dynamic_summary["feedback_latency_preference"] = regularization.map(
        lambda value: value[3]
    )
    dynamic_summary = dynamic_summary.sort_values(
        [
            "selection_objective",
            "robust_objective",
            "development_mean",
            "semantic_projection_preference",
            "relation_aware_score_preference",
            "shared_profile_regularization",
            "feedback_latency_preference",
        ],
        ascending=False,
    )
    dynamic_summary.to_csv(task_dir / "dynamic_summary.csv", index=False)
    selected = NeuroDiscoveryPolicy.from_dict(
        json.loads(dynamic_summary.iloc[0]["policy_json"])
    )
    write_neurodiscovery_policy(task_dir / "selected_policy.json", selected)
    manifest = {
        **base_manifest,
        "status": "complete",
        "static_policy_count": len(policies),
        "dynamic_policy_count": len(dynamic_policies),
        "search_seeds": args.search_seeds,
        "quality_endpoint": table_manifest.get("quality_endpoint"),
        "selected_policy_id": selected.policy_id(),
        "selected_policy": selected.to_dict(),
        "selection_uses_external_outcomes": False,
        "selection_uses_holdout_outcomes": False,
        "selection_tie_break": (
            "quality endpoints use 70% multi-budget robust recovery, 15% "
            "first-budget recovery, and 15% mean registered recall-cost "
            "efficiency; each is penalized by 0.25 development standard "
            "deviations. Remaining ties use semantic feedback "
            "projection, relation-aware static score, distance to the shared "
            "feedback profile, then shorter feedback latency"
        ),
    }
    (task_dir / "tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--kg",
        type=Path,
        help="Optionally rebuild outcome-blind KG score components before tuning.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=4)
    parser.add_argument("--top-static", type=int, default=2)
    parser.add_argument("--search-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-horizon", type=int, default=50_000)
    parser.add_argument("--static-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.folds < 3:
        raise ValueError("at least three folds are required")
    if not 0 <= args.holdout_fold < args.folds:
        raise ValueError("holdout_fold is outside the fold range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for task in args.tasks:
        print(f"Tuning {task}...", flush=True)
        manifests.append(tune_task(args, task))
    summary = {
        "schema_version": "case-study-neurodiscovery-tuning-matrix.v1",
        "created_at": utc_now(),
        "run_root": str(args.run_root.resolve()),
        "canonical_release": canonical_release_fingerprints(args.kg),
        "external_outcomes_opened": False,
        "tasks": manifests,
    }
    path = args.output_dir / "tuning_matrix_manifest.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
