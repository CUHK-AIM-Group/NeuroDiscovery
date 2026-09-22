"""Evaluate the single-seed Case Study 2 closed-loop trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.scripts.prepare_case2_adni_closed_loop_benchmark import (
    EXPECTED_METHODS,
    load_json,
    sha256_file,
    verify_freeze,
    write_json,
)
from neurooracle.scripts.run_case2_adni_closed_loop_benchmark import INVALID_PREFIX


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def discrete_cumulative_gain_auc(values: Sequence[float], *, horizon: int) -> float:
    utility = np.asarray(values, dtype=float)[:horizon]
    if len(utility) < horizon:
        utility = np.pad(utility, (0, horizon - len(utility)))
    utility = np.where(np.isfinite(utility), np.maximum(utility, 0.0), 0.0)
    return float(np.cumsum(utility).sum())


def normalized_auc(values: Sequence[float], population_utility: Sequence[float], *, horizon: int) -> dict[str, float]:
    population = np.asarray(population_utility, dtype=float)
    _require(len(population) == horizon, "The current normalization requires horizon=population")
    observed = discrete_cumulative_gain_auc(values, horizon=horizon)
    oracle_values = np.sort(population)[::-1]
    oracle = discrete_cumulative_gain_auc(oracle_values, horizon=horizon)
    random_expected = float(population.mean() * horizon * (horizon + 1) / 2.0)
    denominator = oracle - random_expected
    return {
        "raw_auc": observed,
        "oracle_auc": oracle,
        "random_expected_auc": random_expected,
        "oracle_normalized_auc": observed / oracle if oracle > 0 else float("nan"),
        "random_oracle_normalized_auc": (
            (observed - random_expected) / denominator if denominator > 0 else float("nan")
        ),
    }


def ndcg_at_k(values: Sequence[float], population_utility: Sequence[float], k: int) -> float:
    observed = np.asarray(values, dtype=float)[:k]
    if len(observed) < k:
        observed = np.pad(observed, (0, k - len(observed)))
    ideal = np.sort(np.asarray(population_utility, dtype=float))[::-1][:k]
    discounts = np.log2(np.arange(2, k + 2, dtype=float))
    denominator = float(np.sum(ideal / discounts))
    return float(np.sum(observed / discounts) / denominator) if denominator > 0 else float("nan")


def _bool_series(values: pd.Series) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.casefold()
    _require(normalized.isin({"true", "false"}).all(), "Invalid Boolean reference")
    return normalized.eq("true").to_numpy(bool)


def load_reference(benchmark_root: Path) -> pd.DataFrame:
    reference = pd.read_csv(benchmark_root / "evaluator_only/REFERENCE_LABELS.csv", dtype=str)
    _require(len(reference) == 168 and reference["candidate_id"].is_unique, "Reference changed")
    for column in ("bootstrap_weakest_link_evidence", "absolute_indirect_effect"):
        reference[column] = pd.to_numeric(reference[column], errors="raise")
    for column in ("supplemental_family_fdr_hit", "global_fdr_hit", "nominal_bootstrap_hit"):
        reference[column] = _bool_series(reference[column])
    _require(int(reference["supplemental_family_fdr_hit"].sum()) == 6, "Family labels changed")
    _require(int(reference["global_fdr_hit"].sum()) == 0, "Global labels changed")
    _require(int(reference["nominal_bootstrap_hit"].sum()) == 15, "Nominal labels changed")
    return reference


def load_trajectory(benchmark_root: Path, method: str) -> pd.DataFrame:
    path = benchmark_root / "trajectories" / method / "trial_00/trajectory.csv"
    lock_path = path.with_name("TRAJECTORY.lock.json")
    lock = load_json(lock_path)
    _require(sha256_file(path) == lock["trajectory_sha256"], f"Trajectory drift: {method}")
    frame = pd.read_csv(path, dtype={"candidate_id": str})
    _require(frame["action_slot"].tolist() == list(range(1, len(frame) + 1)), "Action slots changed")
    real = frame.loc[~frame["candidate_id"].str.startswith(INVALID_PREFIX), "candidate_id"]
    _require(len(real) == len(set(real)) == 168, f"Incomplete method-selected trajectory: {method}")
    return frame


def static_neurodiscovery_trajectory(benchmark_root: Path) -> pd.DataFrame:
    path = benchmark_root / "runs/neurodiscovery/trial_00/static_initial_ranking.csv"
    static = pd.read_csv(path, dtype={"candidate_id": str})
    _require(static["static_rank"].tolist() == list(range(1, 169)), "Static ND ranking changed")
    return pd.DataFrame(
        {
            "action_slot": range(1, 169), "round": -1, "phase": "static_ablation",
            "within_round_action": range(1, 169), "candidate_id": static["candidate_id"],
            "valid_selected": True,
        }
    )


def _actions_to_target(values: np.ndarray, target: float) -> int | None:
    positions = np.flatnonzero(np.cumsum(values) >= target - 1e-15)
    return int(positions[0] + 1) if len(positions) else None


def evaluate_method(
    method: str,
    trajectory: pd.DataFrame,
    reference: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    indexed = reference.set_index("candidate_id", drop=False)
    ids = trajectory["candidate_id"].astype(str).tolist()
    evidence_lookup = indexed["bootstrap_weakest_link_evidence"].to_dict()
    effect_lookup = indexed["absolute_indirect_effect"].to_dict()
    family_lookup = indexed["supplemental_family_fdr_hit"].to_dict()
    nominal_lookup = indexed["nominal_bootstrap_hit"].to_dict()
    evidence = np.asarray([float(evidence_lookup.get(value, 0.0)) for value in ids])
    effects = np.asarray([float(effect_lookup.get(value, 0.0)) for value in ids])
    family = np.asarray([bool(family_lookup.get(value, False)) for value in ids])
    nominal = np.asarray([bool(nominal_lookup.get(value, False)) for value in ids])
    horizon = int(config["sequential_design"]["primary_action_horizon"])
    evidence_norm = normalized_auc(
        evidence, reference["bootstrap_weakest_link_evidence"], horizon=horizon,
    )
    effect_norm = normalized_auc(effects, reference["absolute_indirect_effect"], horizon=horizon)
    prefix_ids = ids[:horizon]
    summary: dict[str, Any] = {
        "method": method,
        "trial": 0,
        "primary_random_oracle_normalized_evidence_auc": evidence_norm["random_oracle_normalized_auc"],
        "evidence_auc_raw": evidence_norm["raw_auc"],
        "evidence_auc_oracle_normalized": evidence_norm["oracle_normalized_auc"],
        "effect_auc_random_oracle_normalized": effect_norm["random_oracle_normalized_auc"],
        "main_action_horizon": horizon,
        "total_action_slots_including_recovery": len(ids),
        "invalid_actions_within_horizon": sum(value.startswith(INVALID_PREFIX) for value in prefix_ids),
        "unique_real_candidates_within_horizon": len({value for value in prefix_ids if not value.startswith(INVALID_PREFIX)}),
        "recovery_action_slots": max(0, len(ids) - horizon),
        "family_hits_within_horizon": int(family[:horizon].sum()),
        "nominal_hits_within_horizon": int(nominal[:horizon].sum()),
        "single_seed_descriptive_only": True,
    }
    total_evidence = float(reference["bootstrap_weakest_link_evidence"].sum())
    for target in map(float, config["evaluation"]["graded_evidence_recovery_targets"]):
        summary[f"actions_to_{int(target * 100)}pct_graded_evidence"] = _actions_to_target(
            evidence, target * total_evidence,
        )
    for target in map(int, config["evaluation"]["family_hit_targets"]):
        summary[f"actions_to_{target}_family_hits"] = _actions_to_target(family.astype(float), float(target))

    budgets: list[dict[str, Any]] = []
    for k in map(int, config["evaluation"]["fixed_budgets"]):
        gain = float(evidence[:k].sum())
        budgets.append(
            {
                "method": method, "trial": 0, "k": k,
                "cumulative_weakest_link_gain": gain,
                "graded_evidence_recall": gain / total_evidence if total_evidence > 0 else float("nan"),
                "ndcg": ndcg_at_k(evidence, reference["bootstrap_weakest_link_evidence"], k),
                "supplemental_family_fdr_hits": int(family[:k].sum()),
                "supplemental_family_fdr_recall": float(family[:k].sum() / 6.0),
                "nominal_bootstrap_hits": int(nominal[:k].sum()),
                "invalid_actions": sum(value.startswith(INVALID_PREFIX) for value in ids[:k]),
                "unique_real_candidates": len({value for value in ids[:k] if not value.startswith(INVALID_PREFIX)}),
            }
        )
    labeled = trajectory.copy()
    labeled["bootstrap_weakest_link_evidence"] = evidence
    labeled["absolute_indirect_effect"] = effects
    labeled["supplemental_family_fdr_hit"] = family
    labeled["nominal_bootstrap_hit"] = nominal
    labeled["cumulative_weakest_link_gain"] = np.cumsum(evidence)
    return summary, budgets, labeled


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.benchmark_root.resolve()
    frozen = verify_freeze(root)
    trajectories_lock = load_json(root / "TRAJECTORIES.lock.json")
    _require(trajectories_lock["status"] == "locked_complete_single_seed_trajectories", "Run incomplete")
    config = load_json(root / "PROTOCOL.json")
    migration = config.get("provider_migration") or {}
    old_thread_id = migration.get("old_thread_id")
    new_thread_id = config["model_backend"]["codex_thread_id"]
    additional_reused = set(config["hybrid_resume"].get("additional_reused_methods") or [])
    reference = load_reference(root)
    summaries: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    labeled_frames: list[pd.DataFrame] = []
    for method in EXPECTED_METHODS:
        summary, budgets, labeled = evaluate_method(method, load_trajectory(root, method), reference, config)
        if method == config["hybrid_resume"]["reused_method"]:
            provider = "none (hash-verified NeuroDiscovery import)"
            provenance = "reused_from_v2_freeze"
        elif method in additional_reused:
            provider = f"gpt-5.6-luna/max via source Codex CLI thread {old_thread_id}"
            provenance = "completed_trajectory_hash_verified_import_from_superseded_v3e_freeze"
        elif method == "open_coscientist" and migration:
            provider = "gpt-5.6-luna/max via audited mixed-thread Codex CLI replay"
            provenance = (
                f"exact_sealed_cache_from_{old_thread_id}_plus_new_responses_from_{new_thread_id}"
            )
        else:
            provider = f"gpt-5.6-luna/max via destination Codex CLI thread {new_thread_id}"
            provenance = "executed_in_v3f_provider_migration_resume"
        summary["model_provider"] = provider
        summary["execution_provenance"] = provenance
        for row in budgets:
            row["model_provider"] = provider
            row["execution_provenance"] = provenance
        summaries.append(summary)
        budget_rows.extend(budgets)
        labeled.insert(0, "method", method)
        labeled.insert(1, "model_provider", provider)
        labeled.insert(2, "execution_provenance", provenance)
        labeled_frames.append(labeled)
    static_method = "neurodiscovery_static_ablation"
    summary, budgets, labeled = evaluate_method(
        static_method, static_neurodiscovery_trajectory(root), reference, config,
    )
    summary["model_provider"] = "none (deterministic static ablation)"
    summary["execution_provenance"] = "derived_from_reused_neurodiscovery_static_ranking"
    for row in budgets:
        row["model_provider"] = summary["model_provider"]
        row["execution_provenance"] = summary["execution_provenance"]
    summaries.append(summary)
    budget_rows.extend(budgets)
    labeled.insert(0, "method", static_method)
    labeled.insert(1, "model_provider", summary["model_provider"])
    labeled.insert(2, "execution_provenance", summary["execution_provenance"])
    labeled_frames.append(labeled)

    summary_frame = pd.DataFrame(summaries).sort_values(
        "primary_random_oracle_normalized_evidence_auc", ascending=False, kind="stable"
    )
    budget_frame = pd.DataFrame(budget_rows)
    labeled_frame = pd.concat(labeled_frames, ignore_index=True)
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "METHOD_SUMMARY.csv"
    budget_path = output / "BUDGET_CURVES.csv"
    labeled_path = output / "LABELED_TRAJECTORIES.csv"
    summary_frame.to_csv(summary_path, index=False)
    budget_frame.to_csv(budget_path, index=False)
    labeled_frame.to_csv(labeled_path, index=False)

    by_method = summary_frame.set_index("method")
    dynamic = float(by_method.loc["neurodiscovery", "primary_random_oracle_normalized_evidence_auc"])
    static_value = float(by_method.loc[static_method, "primary_random_oracle_normalized_evidence_auc"])
    lift = {
        "schema_version": "neurooracle.case2_closed_loop_lift.v3",
        "dynamic_method": "neurodiscovery",
        "static_ablation": static_method,
        "metric": config["evaluation"]["primary_metric"]["name"],
        "dynamic_value": dynamic,
        "static_value": static_value,
        "absolute_lift": dynamic - static_value,
        "feedback_improved_primary_metric": dynamic > static_value,
        "single_seed_descriptive_only": True,
    }
    write_json(output / "NEURODISCOVERY_CLOSED_LOOP_LIFT.json", lift)
    manifest = {
        "schema_version": "neurooracle.case2_closed_loop_evaluation_manifest.v3",
        "benchmark_id": config["benchmark_id"],
        "status": "single_seed_evaluation_complete",
        "freeze_id": frozen["freeze_id"],
        "method_count": 7,
        "static_ablation_count": 1,
        "candidate_count": 168,
        "primary_metric": config["evaluation"]["primary_metric"],
        "single_seed_descriptive_only": True,
        "paired_significance_tests_run": False,
        "confidence_intervals_across_seeds_run": False,
        "holm_superiority_claim_run": False,
        "clear_sota_claim_allowed": False,
        "hybrid_resume": True,
        "reused_methods": [
            config["hybrid_resume"]["reused_method"],
            *sorted(additional_reused),
        ],
        "model_backed_methods_provider": "gpt-5.6-luna/max via audited mixed-thread Codex CLI replay",
        "provider_migration": migration,
        "deepseek_results_claimed": False,
        "files": {
            "method_summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)},
            "budget_curves": {"path": budget_path.name, "sha256": sha256_file(budget_path)},
            "labeled_trajectories": {"path": labeled_path.name, "sha256": sha256_file(labeled_path)},
            "closed_loop_lift": {
                "path": "NEURODISCOVERY_CLOSED_LOOP_LIFT.json",
                "sha256": sha256_file(output / "NEURODISCOVERY_CLOSED_LOOP_LIFT.json"),
            },
        },
    }
    write_json(output / "EVALUATION.lock.json", manifest)
    print(summary_frame.to_string(index=False))
    print(json.dumps(lift, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
