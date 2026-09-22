"""Evaluate frozen case-study rankings on a deterministic group holdout.

Hyperparameter search and group-holdout evaluation are intentionally separate:
the tuner writes a frozen policy without computing holdout metrics, while this
module opens holdout labels only after full ranking permutations are frozen.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import (
    aggregate_with_variance,
    align_outcome_array,
    evaluate_rankings,
    load_frozen_rankings,
    paired_comparisons,
    recovery_curve_auc_by_trial,
    sha256_file,
    validate_hidden_outcomes,
    validate_public_registry,
)
from core.scripts.tune_case_study_neurodiscovery import stable_group_folds


SCHEMA_VERSION = "case-study-group-holdout-evaluation.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate_group_holdout(
    public: pd.DataFrame,
    outcomes: pd.DataFrame,
    records: Sequence[Any],
    *,
    factor_fields: Sequence[str],
    n_folds: int,
    holdout_fold: int,
    budgets: Sequence[int],
    recall_targets: Sequence[float],
) -> dict[str, Any]:
    """Score frozen full-pool orders using only positives in one group fold."""

    if n_folds < 3:
        raise ValueError("at least three folds are required")
    if not 0 <= holdout_fold < n_folds:
        raise ValueError("holdout_fold is outside the fold range")
    folds, group_fields = stable_group_folds(
        public,
        factor_fields,
        n_folds=n_folds,
    )
    all_labels = align_outcome_array(public, outcomes, "validated")
    holdout_mask = folds == holdout_fold
    holdout_labels = all_labels & holdout_mask
    if not holdout_labels.any():
        raise ValueError("selected group holdout contains no validated candidates")

    metrics, recall_costs = evaluate_rankings(
        records,
        holdout_labels,
        budgets=budgets,
        recall_targets=recall_targets,
        scope="group_holdout",
    )
    auc = recovery_curve_auc_by_trial(metrics)
    return {
        "folds": folds,
        "group_fields": tuple(group_fields),
        "holdout_mask": holdout_mask,
        "holdout_labels": holdout_labels,
        "metrics": metrics,
        "recall_costs": recall_costs,
        "recovery_auc": auc,
        "paired_p_values": paired_comparisons(metrics, recall_costs),
    }


def audit_tuning_selection(tuning_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify that the selected dynamic policy is ranked by development columns."""

    summary_path = tuning_dir / "dynamic_summary.csv"
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError("dynamic_summary.csv is empty")
    quality_endpoint = bool(manifest.get("quality_endpoint"))
    if quality_endpoint:
        expected = (
            0.70
            * (summary["robust_objective"] - 0.25 * summary["development_std"])
            + 0.15
            * (summary["early_recovery_mean"] - 0.25 * summary["early_recovery_std"])
            + 0.15
            * (
                summary["recall_efficiency_mean"]
                - 0.25 * summary["recall_efficiency_std"]
            )
        )
    else:
        expected = summary["robust_objective"]
    if not np.allclose(
        summary["selection_objective"].to_numpy(float),
        expected.to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("selection_objective is not development-only")
    selected_id = str(manifest["selected_policy_id"])
    if str(summary.iloc[0]["policy_id"]) != selected_id:
        raise ValueError("selected policy does not match the top development row")
    return {
        "dynamic_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
        },
        "selected_policy_id": selected_id,
        "selection_objective_recomputed_from_development_only": True,
        "holdout_columns_used_for_selection": [],
    }


def _write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    frame.to_csv(path, index=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    tuning_manifest_path = args.tuning_dir / "tuning_manifest.json"
    tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    task = str(benchmark_manifest["task"])
    if str(tuning_manifest["task"]) != task:
        raise ValueError("benchmark and tuning task IDs do not match")

    factor_fields = tuple(map(str, benchmark_manifest["factor_fields"]))
    public = validate_public_registry(
        pd.read_csv(args.public_candidates, low_memory=False),
        factor_fields=factor_fields,
    )
    outcomes = validate_hidden_outcomes(
        public,
        pd.read_csv(args.internal_outcomes),
    )
    records, frozen_manifest = load_frozen_rankings(args.frozen_rankings, public)
    expected_methods = set(map(str, benchmark_manifest["methods"]))
    observed_methods = {record.method for record in records}
    if observed_methods != expected_methods:
        raise ValueError("frozen ranking methods do not match benchmark manifest")

    tuning_audit = audit_tuning_selection(args.tuning_dir, tuning_manifest)
    selected_policy_path = args.tuning_dir / "selected_policy.json"
    selected_policy_sha256 = sha256_file(selected_policy_path)
    frozen_policy = (frozen_manifest.get("inputs") or {}).get(
        "neurodiscovery_config"
    ) or {}
    if str(frozen_policy.get("sha256") or "") != selected_policy_sha256:
        raise ValueError("strict tuning policy does not match the frozen ranking policy")
    benchmark_policy_id = str(
        (benchmark_manifest.get("neurodiscovery_policy") or {}).get("policy_id") or ""
    )
    if benchmark_policy_id != str(tuning_manifest["selected_policy_id"]):
        raise ValueError("benchmark and strict tuning policy IDs do not match")
    result = evaluate_group_holdout(
        public,
        outcomes,
        records,
        factor_fields=factor_fields,
        n_folds=int(tuning_manifest["folds"]),
        holdout_fold=int(tuning_manifest["holdout_fold"]),
        budgets=tuple(map(int, benchmark_manifest["budgets"])),
        recall_targets=tuple(map(float, benchmark_manifest["recall_targets"])),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "metrics_by_trial": _write_csv(
            args.output_dir / "holdout_metrics_by_trial.csv", result["metrics"]
        ),
        "metrics_summary": _write_csv(
            args.output_dir / "holdout_metrics_summary.csv",
            aggregate_with_variance(
                result["metrics"], ("scope", "method", "experiments")
            ),
        ),
        "recall_cost_by_trial": _write_csv(
            args.output_dir / "holdout_recall_cost_by_trial.csv",
            result["recall_costs"],
        ),
        "recall_cost_summary": _write_csv(
            args.output_dir / "holdout_recall_cost_summary.csv",
            aggregate_with_variance(
                result["recall_costs"], ("scope", "method", "recall_target")
            ),
        ),
        "recovery_auc_by_trial": _write_csv(
            args.output_dir / "holdout_recovery_auc_by_trial.csv",
            result["recovery_auc"],
        ),
        "recovery_auc_summary": _write_csv(
            args.output_dir / "holdout_recovery_auc_summary.csv",
            aggregate_with_variance(
                result["recovery_auc"], ("scope", "method")
            ),
        ),
        "paired_p_values": _write_csv(
            args.output_dir / "holdout_paired_p_values.csv",
            result["paired_p_values"],
        ),
    }
    holdout_mask = np.asarray(result["holdout_mask"], dtype=bool)
    holdout_labels = np.asarray(result["holdout_labels"], dtype=bool)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "task": task,
        "status": "complete",
        "evaluation_role": "post-selection deterministic group holdout",
        "candidate_count": int(len(public)),
        "holdout_candidate_count": int(holdout_mask.sum()),
        "holdout_gt_total": int(holdout_labels.sum()),
        "folds": int(tuning_manifest["folds"]),
        "holdout_fold": int(tuning_manifest["holdout_fold"]),
        "group_fields": list(result["group_fields"]),
        "factor_fields": list(factor_fields),
        "methods": sorted(observed_methods),
        "trials": int(benchmark_manifest["trials"]),
        "budgets": list(map(int, benchmark_manifest["budgets"])),
        "recall_targets": list(map(float, benchmark_manifest["recall_targets"])),
        "selection_audit": tuning_audit,
        "evaluation_semantics": {
            "rankings_frozen_before_holdout_metrics": True,
            "policy_selected_from_development_folds_only": True,
            "holdout_feedback_available_during_tuning": False,
            "formal_ranking_is_prequential": True,
            "executed_candidate_outcomes_revealed_only_after_selection": True,
            "holdout_labels_count_only_toward_holdout_metrics": True,
        },
        "inputs": {
            "public_candidates": {
                "path": str(args.public_candidates.resolve()),
                "sha256": sha256_file(args.public_candidates),
            },
            "internal_outcomes": {
                "path": str(args.internal_outcomes.resolve()),
                "sha256": sha256_file(args.internal_outcomes),
            },
            "benchmark_manifest": {
                "path": str(args.benchmark_manifest.resolve()),
                "sha256": sha256_file(args.benchmark_manifest),
            },
            "tuning_manifest": {
                "path": str(tuning_manifest_path.resolve()),
                "sha256": sha256_file(tuning_manifest_path),
            },
            "selected_policy": {
                "path": str(selected_policy_path.resolve()),
                "sha256": selected_policy_sha256,
                "matches_frozen_ranking_policy": True,
            },
            "frozen_rankings_manifest": {
                "path": str(
                    (args.frozen_rankings / "frozen_rankings_manifest.json").resolve()
                ),
                "sha256": sha256_file(
                    args.frozen_rankings / "frozen_rankings_manifest.json"
                ),
                "orders_sha256": frozen_manifest["orders"]["sha256"],
            },
        },
        "artifacts": artifacts,
    }
    manifest_path = args.output_dir / "holdout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-candidates", type=Path, required=True)
    parser.add_argument("--internal-outcomes", type=Path, required=True)
    parser.add_argument("--frozen-rankings", type=Path, required=True)
    parser.add_argument("--tuning-dir", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    manifest = run(build_parser().parse_args())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
