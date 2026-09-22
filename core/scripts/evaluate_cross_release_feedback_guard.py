"""Select the informative-feedback guard on sealed pre-update KG tasks.

This script deliberately reuses the public candidate caches and internal outcomes
from an older, sealed KG release. It never opens external outcomes or the current
target KG. The resulting decision can therefore be frozen before a new KG release
is evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    NeuroDiscoveryPolicy,
    apply_neurodiscovery_policy,
)
from core.scripts.tune_case_study_neurodiscovery import (
    recovery_objective,
    stable_group_folds,
)
from core.scripts.tune_shared_neurodiscovery_policy import (
    SharedProfile,
    crossfit_feedback,
    materialize_policy,
    profiles_for_slate,
    sha256_file,
)


SCHEMA = "cross-release-informative-feedback-guard.v1"
IMPLEMENTATION_REVISION = "static-until-feedback-gate-and-informative-result.v1"
SELECTION_RULE = {
    "uses_external_outcomes": False,
    "uses_outer_holdout": False,
    "required_robust_cross_task_delta": 0.0,
    "required_worst_task_relative_delta": 0.0,
    "maximum_single_task_objective_drop": 0.02,
    "minimum_task_win_or_tie_fraction": 0.5,
    "semantics": (
        "Accept the guard only when inner cross-fit performance is at least as "
        "strong overall and on the worst normalized task, no task loses more "
        "than 0.02 objective, and at least half of selection tasks win or tie."
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ranking_sha256(order: np.ndarray) -> str:
    compact = np.asarray(order, dtype=np.int32)
    return hashlib.sha256(compact.tobytes()).hexdigest()


def implementation_record() -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    files = {
        "guard_evaluator": Path(__file__).resolve(),
        "closed_loop_engine": script_dir / "case_study_closed_loop_engine.py",
        "policy_contract": script_dir / "case_study_neurodiscovery_policy.py",
        "policy_materializer": script_dir / "tune_shared_neurodiscovery_policy.py",
        "benchmark_contract": script_dir / "case_study_closed_loop.py",
    }
    return {
        "revision": IMPLEMENTATION_REVISION,
        "outcome_labels_used_for_revision": False,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }


def _artifact_path(source_root: Path, descriptor: object, fallback: str) -> Path:
    if isinstance(descriptor, Mapping) and descriptor.get("path"):
        return Path(str(descriptor["path"]))
    return source_root / fallback


def _verify_source_artifacts(
    source_root: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for name, descriptor in dict(source_manifest.get("artifacts") or {}).items():
        path = _artifact_path(source_root, descriptor, name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = str(dict(descriptor).get("sha256") or "")
        if expected and actual != expected:
            raise ValueError(f"sealed source artifact failed verification: {path}")
        verified[str(name)] = {"path": str(path), "sha256": actual}
    return verified


def select_source_profiles(
    source_root: Path,
    source_manifest: Mapping[str, Any],
    *,
    static_name: str | None = None,
    dynamic_name: str | None = None,
) -> tuple[SharedProfile, SharedProfile, dict[str, Any]]:
    slate = str(source_manifest.get("profile_slate") or "v3")
    profiles = {profile.name: profile for profile in profiles_for_slate(slate)}
    if static_name is None:
        static_name = str(
            dict(source_manifest.get("selected_shared_profile") or {}).get("name")
            or ""
        )
    if static_name not in profiles or not profiles[static_name].static_only:
        raise ValueError(f"invalid sealed static profile: {static_name!r}")

    summary_descriptor = dict(source_manifest.get("artifacts") or {}).get(
        "shared_profile_summary.csv"
    )
    summary_path = _artifact_path(
        source_root,
        summary_descriptor,
        "shared_profile_summary.csv",
    )
    summary = pd.read_csv(summary_path)
    if dynamic_name is None:
        dynamic_rows = summary.loc[
            summary["profile"].isin(
                [name for name, profile in profiles.items() if not profile.static_only]
            )
        ].sort_values(
            ["robust_cross_task_score", "crossfit_relative_mean"],
            ascending=False,
        )
        if dynamic_rows.empty:
            raise ValueError("sealed source summary has no dynamic profile")
        dynamic_name = str(dynamic_rows.iloc[0]["profile"])
    if dynamic_name not in profiles or profiles[dynamic_name].static_only:
        raise ValueError(f"invalid sealed dynamic profile: {dynamic_name!r}")
    audit = {
        "profile_slate": slate,
        "static_profile": asdict(profiles[static_name]),
        "dynamic_profile": asdict(profiles[dynamic_name]),
        "dynamic_selection_source": str(summary_path),
        "dynamic_selection_source_sha256": sha256_file(summary_path),
        "selection_used_target_release": False,
    }
    return profiles[static_name], profiles[dynamic_name], audit


def load_sealed_task(
    task: str,
    bundle_descriptor: Mapping[str, Any],
    *,
    source_root: Path,
    source_kg_sha256: str,
    folds: int,
) -> dict[str, Any]:
    table_dir = Path(str(bundle_descriptor["table_directory"]))
    table_manifest_path = table_dir / "table_manifest.json"
    source_public_path = table_dir / "public_candidates.csv"
    internal_path = table_dir / "internal_outcomes.csv"
    cache_path = source_root / task / "rescored_public_candidates.csv.gz"
    audit_path = source_root / task / "rescoring_audit.json"
    for path in (
        table_manifest_path,
        source_public_path,
        internal_path,
        cache_path,
        audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    table_manifest = read_json(table_manifest_path)
    cache_audit = read_json(audit_path)
    if bool(cache_audit.get("external_outcomes_opened")):
        raise ValueError(f"{task}: sealed cache opened external outcomes")
    if str(cache_audit.get("kg_sha256") or "") != source_kg_sha256:
        raise ValueError(f"{task}: sealed cache KG hash mismatch")
    if sha256_file(source_public_path) != str(
        cache_audit.get("source_public_sha256") or ""
    ):
        raise ValueError(f"{task}: source candidate table hash mismatch")

    factor_fields = tuple(map(str, bundle_descriptor["factor_fields"]))
    if tuple(map(str, table_manifest.get("factor_fields") or ())) != factor_fields:
        raise ValueError(f"{task}: factor fields changed since the sealed run")
    source_public = validate_public_registry(
        pd.read_csv(source_public_path, low_memory=False),
        factor_fields=factor_fields,
    )
    public = validate_public_registry(
        pd.read_csv(cache_path, low_memory=False),
        factor_fields=factor_fields,
    )
    if public["candidate_id"].astype(str).tolist() != source_public[
        "candidate_id"
    ].astype(str).tolist():
        raise ValueError(f"{task}: rescored cache changed candidate identity or order")
    internal = validate_hidden_outcomes(
        public,
        pd.read_csv(internal_path, low_memory=False),
    )
    labels = align_outcome_array(public, internal, "validated")
    fold_ids, group_fields = stable_group_folds(
        public,
        factor_fields,
        n_folds=folds,
    )
    expected_groups = tuple(map(str, bundle_descriptor.get("group_fields") or ()))
    if expected_groups and group_fields != expected_groups:
        raise ValueError(f"{task}: grouped fold definition changed")
    if len(public) != int(bundle_descriptor["candidate_count"]):
        raise ValueError(f"{task}: candidate count changed")
    if int(labels.sum()) != int(bundle_descriptor["validated_count"]):
        raise ValueError(f"{task}: internal positive count changed")

    return {
        "task": task,
        "public": public,
        "internal": internal,
        "labels": labels,
        "factor_fields": factor_fields,
        "group_fields": group_fields,
        "fold_ids": fold_ids,
        "budgets": tuple(map(int, bundle_descriptor["budgets"])),
        "horizon": int(bundle_descriptor["horizon"]),
        "input_audit": {
            "table_manifest": {
                "path": str(table_manifest_path),
                "sha256": sha256_file(table_manifest_path),
            },
            "source_public": {
                "path": str(source_public_path),
                "sha256": sha256_file(source_public_path),
            },
            "internal_outcomes": {
                "path": str(internal_path),
                "sha256": sha256_file(internal_path),
            },
            "rescored_public_cache": {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
            },
            "rescoring_audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
            },
            "external_outcomes_opened": False,
        },
    }


def _policy_for_variant(
    variant: str,
    profile: SharedProfile,
    bundle: Mapping[str, Any],
) -> NeuroDiscoveryPolicy:
    policy = materialize_policy(
        profile,
        task=str(bundle["task"]),
        candidate_count=len(bundle["public"]),
        smallest_budget=min(bundle["budgets"]),
        horizon=int(bundle["horizon"]),
    )
    if variant == "dynamic_original":
        return replace(policy, preserve_static_until_informative_feedback=False)
    if variant == "dynamic_guarded":
        return replace(policy, preserve_static_until_informative_feedback=True)
    if variant == "static_control":
        return policy
    raise ValueError(f"unknown guard variant: {variant!r}")


def evaluate_variant(
    variant: str,
    profile: SharedProfile,
    bundle: Mapping[str, Any],
    *,
    evaluation_folds: Sequence[int],
    outer_holdout_fold: int,
    trials: int,
    seed: int,
    completed: set[tuple[str, str, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task = str(bundle["task"])
    public = bundle["public"]
    labels = np.asarray(bundle["labels"], dtype=bool)
    fold_ids = np.asarray(bundle["fold_ids"], dtype=int)
    policy = _policy_for_variant(variant, profile, bundle)
    scored, score_audit = apply_neurodiscovery_policy(public, policy)
    static_scores = scored["score_neurodiscovery"].to_numpy(dtype=float)
    static_order = np.lexsort((np.arange(len(scored)), -static_scores))
    rows: list[dict[str, Any]] = []
    for evaluation_fold in evaluation_folds:
        feedback, masked_folds = crossfit_feedback(
            bundle["internal"],
            fold_ids,
            evaluation_fold=evaluation_fold,
            outer_holdout_fold=outer_holdout_fold,
        )
        evaluation_mask = fold_ids == int(evaluation_fold)
        role = (
            "outer_holdout"
            if int(evaluation_fold) == int(outer_holdout_fold)
            else "inner_crossfit"
        )
        for trial in range(trials):
            key = (variant, task, int(evaluation_fold), int(trial))
            if key in completed:
                continue
            trial_seed = seed + 1009 * trial + 104729 * int(evaluation_fold)
            overlay: Mapping[str, Any] = {}
            if variant == "static_control":
                order = static_order
            else:
                config = ClosedLoopConfig(
                    **policy.closed_loop_kwargs(default_horizon=int(bundle["horizon"]))
                )
                order, _trace, overlay = closed_loop_neurodiscovery_order(
                    scored,
                    labels,
                    factor_fields=bundle["factor_fields"],
                    rng=np.random.default_rng(trial_seed),
                    config=config,
                    task=task,
                    seed=trial_seed,
                    trial=trial,
                    outcomes=feedback,
                    return_overlay_manifest=True,
                    audit_records=False,
                    collect_trace=False,
                )
            objective, metrics = recovery_objective(
                order,
                labels,
                evaluation_mask,
                bundle["budgets"],
                filter_order=False,
            )
            guard_manifest = dict(overlay.get("static_order_guard") or {})
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "profile": profile.name,
                    "profile_id": profile.profile_id(),
                    "evaluation_role": role,
                    "evaluation_fold": int(evaluation_fold),
                    "masked_feedback_folds": ";".join(map(str, masked_folds)),
                    "trial": int(trial),
                    "seed": int(trial_seed),
                    "policy_id": policy.policy_id(),
                    "guard_enabled": bool(
                        policy.preserve_static_until_informative_feedback
                    ),
                    "guard_activated": bool(
                        guard_manifest.get("adaptive_ranking_activated", False)
                    ),
                    "ranking_reads_after_feedback": int(
                        overlay.get("ranking_reads_after_feedback", 0)
                    ),
                    "nonzero_feedback_reads": int(
                        overlay.get("nonzero_feedback_reads", 0)
                    ),
                    "ranking_sha256": ranking_sha256(order),
                    "objective": float(objective),
                    **metrics,
                }
            )
    policy_audit = {
        "variant": variant,
        "profile": asdict(profile),
        "policy": policy.to_dict(),
        "policy_id": policy.policy_id(),
        "score_audit": score_audit,
    }
    return rows, policy_audit


def aggregate_trials(
    trials: pd.DataFrame,
    selection_tasks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner = trials.loc[trials["evaluation_role"].eq("inner_crossfit")].copy()
    outer = trials.loc[trials["evaluation_role"].eq("outer_holdout")].copy()
    folds = (
        inner.groupby(["variant", "task", "evaluation_fold"], as_index=False)[
            "objective"
        ]
        .mean()
        .rename(columns={"objective": "fold_objective"})
    )
    task_summary = (
        folds.groupby(["variant", "task"], as_index=False)
        .agg(
            crossfit_mean=("fold_objective", "mean"),
            crossfit_min_fold=("fold_objective", "min"),
            crossfit_variance=("fold_objective", "var"),
        )
        .fillna(0.0)
    )
    outer_summary = (
        outer.groupby(["variant", "task"], as_index=False)
        .agg(
            outer_holdout_mean=("objective", "mean"),
            outer_holdout_variance=("objective", "var"),
        )
        .fillna(0.0)
    )
    task_summary = task_summary.merge(
        outer_summary,
        on=["variant", "task"],
        how="left",
        validate="one_to_one",
    ).fillna(0.0)
    task_summary["crossfit_robust"] = (
        0.75 * task_summary["crossfit_mean"]
        + 0.25 * task_summary["crossfit_min_fold"]
    )
    best = task_summary.groupby("task")["crossfit_robust"].transform("max")
    task_summary["crossfit_relative"] = np.divide(
        task_summary["crossfit_robust"],
        best,
        out=np.ones(len(task_summary), dtype=float),
        where=best.to_numpy(dtype=float) > 0,
    )

    selection = task_summary.loc[
        task_summary["task"].isin(selection_tasks)
    ].copy()
    rows: list[dict[str, Any]] = []
    for variant, group in selection.groupby("variant", sort=False):
        by_task = group.set_index("task")
        relative = np.asarray(
            [float(by_task.loc[task, "crossfit_relative"]) for task in selection_tasks]
        )
        rows.append(
            {
                "variant": variant,
                "crossfit_relative_mean": float(relative.mean()),
                "crossfit_relative_min": float(relative.min()),
                "robust_cross_task_score": float(
                    0.75 * relative.mean() + 0.25 * relative.min()
                ),
                "outer_holdout_mean": float(group["outer_holdout_mean"].mean()),
                "selection_task_count": len(relative),
            }
        )
    variant_summary = pd.DataFrame(rows).sort_values(
        ["robust_cross_task_score", "crossfit_relative_mean"],
        ascending=False,
    )
    return task_summary, variant_summary


def exact_sign_flip_p_value(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = float(values.mean())
    if len(values) > 20:
        raise ValueError("exact sign-flip test is limited to 20 paired tasks")
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = float(np.mean(values * np.asarray(signs)))
        if statistic >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def guard_decision(
    task_summary: pd.DataFrame,
    variant_summary: pd.DataFrame,
    selection_tasks: Sequence[str],
) -> dict[str, Any]:
    summaries = variant_summary.set_index("variant")
    original = summaries.loc["dynamic_original"]
    guarded = summaries.loc["dynamic_guarded"]
    pairs = task_summary.loc[
        task_summary["task"].isin(selection_tasks)
        & task_summary["variant"].isin(["dynamic_original", "dynamic_guarded"])
    ].pivot(index="task", columns="variant", values="crossfit_robust")
    differences = pairs["dynamic_guarded"] - pairs["dynamic_original"]
    wins_or_ties = int((differences >= -1e-12).sum())
    required_wins = int(
        math.ceil(
            len(selection_tasks)
            * float(SELECTION_RULE["minimum_task_win_or_tie_fraction"])
        )
    )
    maximum_drop = float((-differences).clip(lower=0.0).max())
    checks = {
        "robust_cross_task_not_worse": bool(
            guarded["robust_cross_task_score"]
            >= original["robust_cross_task_score"]
            + float(SELECTION_RULE["required_robust_cross_task_delta"])
            - 1e-12
        ),
        "worst_task_relative_not_worse": bool(
            guarded["crossfit_relative_min"]
            >= original["crossfit_relative_min"]
            + float(SELECTION_RULE["required_worst_task_relative_delta"])
            - 1e-12
        ),
        "single_task_drop_within_limit": bool(
            maximum_drop
            <= float(SELECTION_RULE["maximum_single_task_objective_drop"])
            + 1e-12
        ),
        "task_win_or_tie_count_met": bool(wins_or_ties >= required_wins),
    }
    return {
        "schema_version": SCHEMA,
        "accepted": bool(all(checks.values())),
        "selection_rule": SELECTION_RULE,
        "checks": checks,
        "selection_tasks": list(selection_tasks),
        "dynamic_original": original.to_dict(),
        "dynamic_guarded": guarded.to_dict(),
        "paired_task_differences": {
            str(task): float(value) for task, value in differences.items()
        },
        "wins_or_ties": wins_or_ties,
        "required_wins_or_ties": required_wins,
        "maximum_single_task_drop": maximum_drop,
        "mean_task_delta": float(differences.mean()),
        "one_sided_exact_sign_flip_p": exact_sign_flip_p_value(differences),
        "outer_holdout_used_for_selection": False,
        "external_outcomes_used_for_selection": False,
    }


def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_trial_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source_root / "shared_tuning_manifest.json"
    source_manifest = read_json(source_manifest_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    source_kg = dict(source_manifest.get("kg") or {})
    source_kg_sha = str(source_kg.get("sha256") or "").lower()
    if len(source_kg_sha) != 64:
        raise ValueError("sealed source manifest has no valid KG SHA-256")
    if bool(source_manifest.get("selection_uses_external_outcomes")):
        raise ValueError("source tuning selected a policy using external outcomes")
    if bool(source_manifest.get("selection_uses_outer_holdout")):
        raise ValueError("source tuning selected a policy using outer holdout")
    source_artifacts = _verify_source_artifacts(source_root, source_manifest)
    static_profile, dynamic_profile, profile_audit = select_source_profiles(
        source_root,
        source_manifest,
        static_name=args.static_profile,
        dynamic_name=args.dynamic_profile,
    )

    folds = int(source_manifest["folds"])
    trials = int(source_manifest["trials"])
    outer_holdout = int(source_manifest["outer_holdout_fold"])
    if args.folds is not None and int(args.folds) != folds:
        raise ValueError("fold override does not match the sealed source protocol")
    if args.trials is not None and int(args.trials) != trials:
        raise ValueError("trial override does not match the sealed source protocol")
    selection_tasks = tuple(map(str, source_manifest["selection_tasks"]))
    tasks = tuple(map(str, source_manifest["tasks"]))
    bundle_descriptors = dict(source_manifest["bundles"])
    task_order = sorted(tasks, key=lambda task: int(bundle_descriptors[task]["candidate_count"]))

    trials_jsonl_path = output_dir / "guard_trials.jsonl"
    trials_path = output_dir / "guard_trials.csv"
    completed: set[tuple[str, str, int, int]] = set()
    if trials_jsonl_path.is_file():
        existing = _read_trial_jsonl(trials_jsonl_path)
        completed = {
            (
                str(row.variant),
                str(row.task),
                int(row.evaluation_fold),
                int(row.trial),
            )
            for row in existing.itertuples(index=False)
        }

    run_manifest_path = output_dir / "guard_evaluation_manifest.json"
    initial_manifest = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "status": "running",
        "source_root": str(source_root),
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": source_manifest_sha,
        },
        "source_kg": {
            "recorded_path": source_kg.get("path"),
            "sha256": source_kg_sha,
            "target_release_opened": False,
        },
        "source_artifacts": source_artifacts,
        "implementation": implementation_record(),
        "profile_selection": profile_audit,
        "selection_rule": SELECTION_RULE,
        "tasks": list(tasks),
        "selection_tasks": list(selection_tasks),
        "audit_only_tasks": list(source_manifest.get("audit_only_tasks") or []),
        "folds": folds,
        "outer_holdout_fold": outer_holdout,
        "trials": trials,
        "seed": int(args.seed),
        "external_outcomes_opened": False,
        "outer_holdout_used_for_selection": False,
    }
    if run_manifest_path.is_file():
        previous = read_json(run_manifest_path)
        if str(dict(previous.get("source_manifest") or {}).get("sha256")) != source_manifest_sha:
            raise ValueError("output directory belongs to a different source manifest")
        initial_manifest["created_at"] = previous.get("created_at", initial_manifest["created_at"])
    write_json(run_manifest_path, initial_manifest)

    input_audits: dict[str, Any] = {}
    policy_audits: dict[str, Any] = {}
    evaluation_folds = tuple(range(folds))
    variants = (
        ("static_control", static_profile),
        ("dynamic_original", dynamic_profile),
        ("dynamic_guarded", dynamic_profile),
    )
    for task in task_order:
        bundle = load_sealed_task(
            task,
            bundle_descriptors[task],
            source_root=source_root,
            source_kg_sha256=source_kg_sha,
            folds=folds,
        )
        input_audits[task] = bundle["input_audit"]
        for variant, profile in variants:
            rows, policy_audit = evaluate_variant(
                variant,
                profile,
                bundle,
                evaluation_folds=evaluation_folds,
                outer_holdout_fold=outer_holdout,
                trials=trials,
                seed=int(args.seed),
                completed=completed,
            )
            _append_rows(trials_jsonl_path, rows)
            completed.update(
                (
                    str(row["variant"]),
                    str(row["task"]),
                    int(row["evaluation_fold"]),
                    int(row["trial"]),
                )
                for row in rows
            )
            policy_audits[f"{task}:{variant}"] = policy_audit
            print(
                f"[{utc_now()}] task={task} variant={variant} "
                f"new_rows={len(rows)} completed={len(completed)}",
                flush=True,
            )
        del bundle

    trials_frame = _read_trial_jsonl(trials_jsonl_path)
    expected_rows = len(tasks) * len(variants) * folds * trials
    if len(trials_frame) != expected_rows:
        raise RuntimeError(
            f"guard evaluation is incomplete: {len(trials_frame)}/{expected_rows} rows"
        )
    trials_frame.to_csv(trials_path, index=False)
    task_summary, variant_summary = aggregate_trials(trials_frame, selection_tasks)
    task_summary_path = output_dir / "guard_task_summary.csv"
    variant_summary_path = output_dir / "guard_variant_summary.csv"
    decision_path = output_dir / "guard_decision.json"
    task_summary.to_csv(task_summary_path, index=False)
    variant_summary.to_csv(variant_summary_path, index=False)
    decision = guard_decision(task_summary, variant_summary, selection_tasks)
    write_json(decision_path, decision)

    complete_manifest = {
        **initial_manifest,
        "completed_at": utc_now(),
        "status": "complete",
        "external_outcomes_opened": False,
        "target_release_opened": False,
        "input_audits": input_audits,
        "policy_audits": policy_audits,
        "decision": decision,
        "artifacts": {
            "guard_trials.jsonl": {
                "path": str(trials_jsonl_path),
                "sha256": sha256_file(trials_jsonl_path),
            },
            "guard_trials.csv": {
                "path": str(trials_path),
                "sha256": sha256_file(trials_path),
            },
            "guard_task_summary.csv": {
                "path": str(task_summary_path),
                "sha256": sha256_file(task_summary_path),
            },
            "guard_variant_summary.csv": {
                "path": str(variant_summary_path),
                "sha256": sha256_file(variant_summary_path),
            },
            "guard_decision.json": {
                "path": str(decision_path),
                "sha256": sha256_file(decision_path),
            },
        },
    }
    write_json(run_manifest_path, complete_manifest)
    complete_manifest["manifest_path"] = str(run_manifest_path)
    complete_manifest["manifest_sha256"] = sha256_file(run_manifest_path)
    return complete_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--static-profile")
    parser.add_argument("--dynamic-profile")
    return parser


def main() -> None:
    manifest = run(build_parser().parse_args())
    print(json.dumps({
        "status": manifest["status"],
        "accepted": manifest["decision"]["accepted"],
        "manifest": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
