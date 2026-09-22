"""Select one compact NeuroDiscovery policy family across multiple case studies.

The tuner reads only public candidate registries and internal outcomes. External
outcomes are never opened. Candidate-group holdouts remain hidden from online
feedback, and every task receives equal weight when selecting the shared
profile. Leave-one-task-out results quantify transfer to a task that did not
participate in profile selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_candidate_tables import score_public_candidates
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
    write_neurodiscovery_policy,
)
from core.scripts.tune_case_study_neurodiscovery import (
    front_loaded_warmup_batches,
    recovery_objective,
    stable_group_folds,
    task_budgets,
)


SCHEMA = "shared-neurodiscovery-tuning.v1"


@dataclass(frozen=True)
class SharedProfile:
    """Small task-invariant surface mapped deterministically to task scale."""

    name: str
    score_family: str
    scoped_fraction: float
    pair_fraction: float
    feedback_weight: float
    pair_feedback_weight: float
    exploration_weight: float
    diversity_penalty: float = 0.08
    feedback_start_fraction: float = 0.025
    static_only: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError("profile name cannot be empty")
        if self.score_family not in {"legacy", "relation_aware"}:
            raise ValueError(f"unknown score family: {self.score_family!r}")
        for field in ("scoped_fraction", "pair_fraction", "feedback_start_fraction"):
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        for field in (
            "feedback_weight",
            "pair_feedback_weight",
            "exploration_weight",
            "diversity_penalty",
        ):
            if float(getattr(self, field)) < 0:
                raise ValueError(f"{field} must be non-negative")

    def profile_id(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_profiles() -> tuple[SharedProfile, ...]:
    """The original compact v2 slate retained for exact reproducibility."""

    return (
        SharedProfile("relation_scoped_pair_static", "relation_aware", 0.90, 0.60, 0, 0, 0, static_only=True),
        SharedProfile("relation_scoped_pair", "relation_aware", 0.90, 0.60, 2.00, 0.15, 0.03),
        SharedProfile("relation_scoped_node", "relation_aware", 0.90, 0.20, 0.60, 0.35, 0.08),
        SharedProfile("relation_balanced_pair", "relation_aware", 0.50, 0.80, 2.00, 0.15, 0.03),
        SharedProfile("relation_balanced", "relation_aware", 0.50, 0.40, 2.00, 0.15, 0.03),
        SharedProfile("relation_balanced_strong", "relation_aware", 0.50, 0.40, 8.00, 0.25, 0.01),
        SharedProfile("relation_mid_scoped", "relation_aware", 0.70, 0.60, 0.60, 0.35, 0.08),
        SharedProfile("legacy_moderate", "legacy", 0.60, 0.60, 0.60, 0.35, 0.08),
    )


def delayed_feedback_profiles() -> tuple[SharedProfile, ...]:
    """Compact v3 slate separating static evidence from weak late feedback."""

    return (
        SharedProfile("relation_scoped_pair_static", "relation_aware", 0.90, 0.60, 0, 0, 0, static_only=True),
        SharedProfile("relation_scoped_high_pair_static", "relation_aware", 0.90, 0.80, 0, 0, 0, static_only=True),
        SharedProfile("relation_balanced_pair_static", "relation_aware", 0.50, 0.80, 0, 0, 0, static_only=True),
        SharedProfile("relation_balanced_static", "relation_aware", 0.50, 0.40, 0, 0, 0, static_only=True),
        SharedProfile("relation_scoped_node_static", "relation_aware", 0.90, 0.20, 0, 0, 0, static_only=True),
        SharedProfile("legacy_balanced_static", "legacy", 0.60, 0.60, 0, 0, 0, static_only=True),
        SharedProfile(
            "relation_scoped_pair_weak_late",
            "relation_aware",
            0.90,
            0.60,
            0.10,
            0.03,
            0.01,
            diversity_penalty=0.03,
            feedback_start_fraction=0.20,
        ),
        SharedProfile(
            "relation_balanced_pair_weak_late",
            "relation_aware",
            0.50,
            0.80,
            0.10,
            0.03,
            0.01,
            diversity_penalty=0.03,
            feedback_start_fraction=0.20,
        ),
    )


def profiles_for_slate(name: str) -> tuple[SharedProfile, ...]:
    if name == "v2":
        return default_profiles()
    if name == "v3":
        return delayed_feedback_profiles()
    raise ValueError(f"unknown profile slate: {name!r}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_task_paths(values: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        task, separator, raw_path = str(value).partition("=")
        if not separator or not task.strip() or not raw_path.strip():
            raise ValueError(f"expected TASK=PATH, received {value!r}")
        task = task.strip()
        if task in parsed:
            raise ValueError(f"duplicate bundle for {task!r}")
        path = Path(raw_path.strip())
        table_dir = path / "tables" if (path / "tables").is_dir() else path
        if not (table_dir / "table_manifest.json").is_file():
            raise FileNotFoundError(f"table bundle is incomplete: {table_dir}")
        parsed[task] = table_dir
    if len(parsed) < 2:
        raise ValueError("shared tuning requires at least two task bundles")
    return parsed


def materialize_policy(
    profile: SharedProfile,
    *,
    task: str,
    candidate_count: int,
    smallest_budget: int,
    horizon: int,
) -> NeuroDiscoveryPolicy:
    profile.validate()
    global_fraction = 1.0 - profile.scoped_fraction
    node_fraction = 1.0 - profile.pair_fraction
    weights = {
        "global_node_weight": global_fraction * node_fraction,
        "global_pair_weight": global_fraction * profile.pair_fraction,
        "scoped_node_weight": profile.scoped_fraction * node_fraction,
        "scoped_pair_weight": profile.scoped_fraction * profile.pair_fraction,
    }
    if profile.static_only:
        return NeuroDiscoveryPolicy(
            task=task,
            score_family=profile.score_family,
            **weights,
            batch_size=1,
            warmup_batches=1,
            sampling_temperature=0.0,
            feedback_weight=0.0,
            pair_feedback_weight=0.0,
            exploration_weight=0.0,
            diversity_penalty=0.0,
            max_feedback_rounds=1,
            feedback_horizon=1,
            feedback_projection="relation_endpoints",
            preserve_static_until_informative_feedback=False,
        )

    if candidate_count >= 50_000:
        batch_size = 2
        max_rounds = 128
        warmup_batches = front_loaded_warmup_batches(
            horizon=horizon,
            batch_size=batch_size,
            max_feedback_rounds=max_rounds,
            feedback_start_budget=max(
                smallest_budget,
                int(round(horizon * profile.feedback_start_fraction)),
            ),
        )
        schedule = "front_loaded"
        temperature = 1e-6
        warmup_diversity = 10.0
    else:
        batch_size = max(2, min(64, int(math.ceil(smallest_budget / 4))))
        max_rounds = max(128, int(math.ceil(horizon / batch_size)))
        feedback_start_budget = max(
            smallest_budget,
            int(round(horizon * profile.feedback_start_fraction)),
        )
        warmup_batches = max(
            1,
            min(max_rounds, int(math.ceil(feedback_start_budget / batch_size))),
        )
        schedule = "fixed"
        temperature = 0.002
        warmup_diversity = None

    if schedule == "fixed":
        effective_batch_size = max(
            batch_size,
            int(math.ceil(horizon / max_rounds)),
        )
        planned_rounds = int(math.ceil(horizon / effective_batch_size))
    else:
        planned_rounds = min(max_rounds, int(math.ceil(horizon / batch_size)))
    if planned_rounds >= 2:
        # A dynamic policy must leave at least one ranking round after warm-up
        # so that completed experiments can actually affect a later decision.
        warmup_batches = min(warmup_batches, planned_rounds - 1)

    return NeuroDiscoveryPolicy(
        task=task,
        score_family=profile.score_family,
        **weights,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
        sampling_temperature=temperature,
        feedback_weight=profile.feedback_weight,
        pair_feedback_weight=profile.pair_feedback_weight,
        exploration_weight=profile.exploration_weight,
        diversity_penalty=profile.diversity_penalty,
        warmup_diversity_penalty=warmup_diversity,
        max_feedback_rounds=max_rounds,
        feedback_horizon=horizon,
        feedback_projection="relation_endpoints",
        feedback_batch_schedule=schedule,
        preserve_static_until_informative_feedback=True,
    )


def load_task(
    task: str,
    table_dir: Path,
    *,
    kg_path: Path,
    output_dir: Path,
    folds: int,
    seed: int,
    max_horizon: int,
) -> dict[str, Any]:
    manifest_path = table_dir / "table_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("task")) != task:
        raise ValueError(f"bundle task mismatch: expected {task}, found {manifest.get('task')}")
    factor_fields = tuple(manifest["factor_fields"])
    source_public = table_dir / "public_candidates.csv"
    source_internal = table_dir / "internal_outcomes.csv"
    public_raw = validate_public_registry(
        pd.read_csv(source_public, low_memory=False), factor_fields=factor_fields
    )
    internal = validate_hidden_outcomes(
        public_raw, pd.read_csv(source_internal, low_memory=False)
    )
    labels = align_outcome_array(public_raw, internal, "validated")
    if not 0 < int(labels.sum()) < len(labels):
        raise ValueError(
            f"{task} is not tunable: validated={int(labels.sum())}/{len(labels)}"
        )

    semantic_fields = tuple(
        manifest.get("kg_scoring", {}).get("semantic_fields") or factor_fields
    )
    cache_dir = output_dir / task
    cache_dir.mkdir(parents=True, exist_ok=True)
    rescored_path = cache_dir / "rescored_public_candidates.csv.gz"
    audit_path = cache_dir / "rescoring_audit.json"
    source_sha = sha256_file(source_public)
    kg_sha = sha256_file(kg_path)
    use_cache = False
    if rescored_path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        use_cache = (
            audit.get("source_public_sha256") == source_sha
            and audit.get("kg_sha256") == kg_sha
        )
    if use_cache:
        public = validate_public_registry(
            pd.read_csv(rescored_path, low_memory=False), factor_fields=factor_fields
        )
        rescoring_audit = audit
    else:
        score_inputs = public_raw.drop(
            columns=[
                column
                for column in public_raw.columns
                if column.startswith("kg_") or column == "score_neurodiscovery"
            ]
        )
        public, rescoring_audit = score_public_candidates(
            score_inputs,
            semantic_fields=semantic_fields,
            case_study_id=task,
            kg_path=kg_path,
            seed=seed,
        )
        public.to_csv(rescored_path, index=False, compression="gzip")
        rescoring_audit = {
            **rescoring_audit,
            "source_public": str(source_public.resolve()),
            "source_public_sha256": source_sha,
            "kg_sha256": kg_sha,
            "external_outcomes_opened": False,
        }
        audit_path.write_text(
            json.dumps(rescoring_audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if public["candidate_id"].astype(str).tolist() != public_raw["candidate_id"].astype(str).tolist():
        raise ValueError(f"rescoring changed candidate identity or order for {task}")

    fold_ids, group_fields = stable_group_folds(public, factor_fields, n_folds=folds)
    budgets = task_budgets(manifest, len(public), max_horizon)
    horizon = min(max(budgets), len(public), max_horizon)
    return {
        "task": task,
        "public": public,
        "internal": internal,
        "labels": labels,
        "factor_fields": factor_fields,
        "group_fields": group_fields,
        "fold_ids": fold_ids,
        "budgets": budgets,
        "horizon": horizon,
        "manifest": manifest,
        "rescoring_audit": rescoring_audit,
    }


def crossfit_feedback(
    internal: pd.DataFrame,
    fold_ids: np.ndarray,
    *,
    evaluation_fold: int,
    outer_holdout_fold: int,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Mask every fold whose labels cannot be consumed during one ranking run."""
    masked_folds = {int(evaluation_fold)}
    if evaluation_fold != outer_holdout_fold:
        masked_folds.add(int(outer_holdout_fold))
    masked = np.isin(np.asarray(fold_ids, dtype=int), sorted(masked_folds))
    feedback = internal.copy()
    feedback.loc[masked, "validated"] = False
    feedback.loc[masked, "feedback_status"] = "inconclusive"
    return feedback, tuple(sorted(masked_folds))


def evaluate_profile(
    profile: SharedProfile,
    bundle: dict[str, Any],
    *,
    evaluation_folds: Sequence[int],
    outer_holdout_fold: int,
    trials: int,
    seed: int,
    completed_keys: set[tuple[str, str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    task = str(bundle["task"])
    public = bundle["public"]
    labels = np.asarray(bundle["labels"], dtype=bool)
    folds = np.asarray(bundle["fold_ids"], dtype=int)
    policy = materialize_policy(
        profile,
        task=task,
        candidate_count=len(public),
        smallest_budget=min(bundle["budgets"]),
        horizon=int(bundle["horizon"]),
    )
    scored, _audit = apply_neurodiscovery_policy(public, policy)
    completed_keys = completed_keys or set()
    rows: list[dict[str, Any]] = []
    profile_id = profile.profile_id()
    for evaluation_fold in evaluation_folds:
        feedback, masked_folds = crossfit_feedback(
            bundle["internal"],
            folds,
            evaluation_fold=evaluation_fold,
            outer_holdout_fold=outer_holdout_fold,
        )
        evaluation_mask = folds == evaluation_fold
        role = (
            "outer_holdout"
            if evaluation_fold == outer_holdout_fold
            else "inner_crossfit"
        )
        for trial in range(trials):
            key = (profile_id, task, int(evaluation_fold), int(trial))
            if key in completed_keys:
                continue
            trial_seed = seed + 1009 * trial + 104729 * int(evaluation_fold)
            if profile.static_only:
                scores = scored["score_neurodiscovery"].to_numpy(float)
                order = np.lexsort((np.arange(len(scored)), -scores))
            else:
                config = ClosedLoopConfig(
                    **policy.closed_loop_kwargs(default_horizon=int(bundle["horizon"]))
                )
                order, _trace = closed_loop_neurodiscovery_order(
                    scored,
                    labels,
                    factor_fields=bundle["factor_fields"],
                    rng=np.random.default_rng(trial_seed),
                    config=config,
                    task=task,
                    seed=trial_seed,
                    trial=trial,
                    outcomes=feedback,
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
            rows.append(
                {
                    "task": task,
                    "profile": profile.name,
                    "profile_id": profile_id,
                    "evaluation_role": role,
                    "evaluation_fold": int(evaluation_fold),
                    "masked_feedback_folds": ";".join(map(str, masked_folds)),
                    "trial": trial,
                    "seed": trial_seed,
                    "policy_id": policy.policy_id(),
                    "objective": objective,
                    **metrics,
                }
            )
    return rows


def aggregate_profiles(
    trials: pd.DataFrame,
    selection_tasks: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner = trials.loc[trials["evaluation_role"].eq("inner_crossfit")].copy()
    outer = trials.loc[trials["evaluation_role"].eq("outer_holdout")].copy()
    fold_summary = (
        inner.groupby(
            ["profile", "profile_id", "task", "evaluation_fold"],
            as_index=False,
        )["objective"]
        .mean()
    )
    task_summary = (
        fold_summary.groupby(["profile", "profile_id", "task"], as_index=False)
        .agg(
            crossfit_mean=("objective", "mean"),
            crossfit_min_fold=("objective", "min"),
            crossfit_variance=("objective", "var"),
        )
        .fillna(0.0)
    )
    outer_summary = (
        outer.groupby(["profile", "profile_id", "task"], as_index=False)
        .agg(
            outer_holdout_mean=("objective", "mean"),
            outer_holdout_variance=("objective", "var"),
        )
        .fillna(0.0)
    )
    task_summary = task_summary.merge(
        outer_summary,
        on=["profile", "profile_id", "task"],
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
        where=best.to_numpy(float) > 0,
    )
    selection = task_summary.loc[task_summary["task"].isin(selection_tasks)].copy()
    profile_rows: list[dict[str, Any]] = []
    for (profile, profile_id), group in selection.groupby(
        ["profile", "profile_id"], sort=False
    ):
        by_task = group.set_index("task")
        relative = np.asarray(
            [float(by_task.loc[task, "crossfit_relative"]) for task in selection_tasks],
            dtype=float,
        )
        profile_rows.append(
            {
                "profile": profile,
                "profile_id": profile_id,
                "crossfit_relative_mean": float(relative.mean()),
                "crossfit_relative_min": float(relative.min()),
                "robust_cross_task_score": float(0.75 * relative.mean() + 0.25 * relative.min()),
                "outer_holdout_mean": float(group["outer_holdout_mean"].mean()),
            }
        )
    profiles = pd.DataFrame(profile_rows).sort_values(
        ["robust_cross_task_score", "crossfit_relative_mean"], ascending=False
    )
    return task_summary, profiles


def leave_one_task_out(
    task_summary: pd.DataFrame,
    tasks: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for held_task in tasks:
        training = task_summary.loc[task_summary["task"] != held_task]
        candidates: list[dict[str, Any]] = []
        for (profile, profile_id), group in training.groupby(
            ["profile", "profile_id"], sort=False
        ):
            values = group["crossfit_relative"].to_numpy(float)
            candidates.append(
                {
                    "profile": profile,
                    "profile_id": profile_id,
                    "selection_score": float(0.75 * values.mean() + 0.25 * values.min()),
                }
            )
        selected = max(candidates, key=lambda row: (row["selection_score"], row["profile"]))
        held = task_summary.loc[
            (task_summary["task"] == held_task)
            & (task_summary["profile_id"] == selected["profile_id"])
        ].iloc[0]
        rows.append(
            {
                "held_out_task": held_task,
                "selected_profile": selected["profile"],
                "selected_profile_id": selected["profile_id"],
                "training_selection_score": selected["selection_score"],
                "held_out_crossfit_relative": held["crossfit_relative"],
                "outer_candidate_group_holdout_objective": held["outer_holdout_mean"],
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True, help="TASK=table-directory")
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-horizon", type=int, default=200_000)
    parser.add_argument("--min-selection-positives", type=int, default=5)
    parser.add_argument(
        "--profile-slate",
        choices=("v2", "v3"),
        default="v2",
        help="Predeclared shared-policy slate; v2 is retained for reproducibility.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.folds < 3 or not 0 <= args.holdout_fold < args.folds:
        raise ValueError("invalid grouped-fold configuration")
    if args.trials < 1:
        raise ValueError("trials must be positive")
    if args.min_selection_positives < 1:
        raise ValueError("min-selection-positives must be positive")
    args.kg = args.kg.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = parse_task_paths(args.bundle)
    tasks = tuple(paths)
    profiles = profiles_for_slate(args.profile_slate)
    bundles: dict[str, dict[str, Any]] = {}
    for task, table_dir in paths.items():
        print(f"[{task}] loading and outcome-blind rescoring", flush=True)
        bundles[task] = load_task(
            task,
            table_dir,
            kg_path=args.kg,
            output_dir=args.output_dir,
            folds=args.folds,
            seed=args.seed,
            max_horizon=args.max_horizon,
        )

    selection_tasks = tuple(
        task
        for task, bundle in bundles.items()
        if int(np.asarray(bundle["labels"], dtype=bool).sum())
        >= args.min_selection_positives
    )
    audit_only_tasks = tuple(task for task in tasks if task not in selection_tasks)
    if len(selection_tasks) < 2:
        raise ValueError("fewer than two tasks have enough positives for shared selection")

    partial_path = args.output_dir / "profile_trials.partial.csv"
    if partial_path.exists() and not args.resume:
        raise FileExistsError(
            f"Partial tuning state exists: {partial_path}; pass --resume to continue"
        )
    trials_frame = pd.read_csv(partial_path) if partial_path.exists() else pd.DataFrame()
    completed_keys: set[tuple[str, str, int, int]] = set()
    if not trials_frame.empty:
        completed_keys = {
            (
                str(row.profile_id),
                str(row.task),
                int(row.evaluation_fold),
                int(row.trial),
            )
            for row in trials_frame.itertuples(index=False)
        }
    evaluation_folds = tuple(range(args.folds))
    for profile_index, profile in enumerate(profiles, 1):
        for task in tasks:
            print(
                f"[{profile_index}/{len(profiles)} {profile.name}] evaluating {task}",
                flush=True,
            )
            new_rows = evaluate_profile(
                profile,
                bundles[task],
                evaluation_folds=evaluation_folds,
                outer_holdout_fold=args.holdout_fold,
                trials=args.trials,
                seed=args.seed,
                completed_keys=completed_keys,
            )
            if new_rows:
                trials_frame = pd.concat(
                    [trials_frame, pd.DataFrame(new_rows)], ignore_index=True
                )
                trials_frame.to_csv(partial_path, index=False)
                completed_keys.update(
                    (
                        str(row["profile_id"]),
                        str(row["task"]),
                        int(row["evaluation_fold"]),
                        int(row["trial"]),
                    )
                    for row in new_rows
                )
    expected_rows = len(profiles) * len(tasks) * args.folds * args.trials
    if len(trials_frame) != expected_rows:
        raise RuntimeError(
            f"incomplete cross-fit matrix: {len(trials_frame)}/{expected_rows} rows"
        )
    trials = trials_frame.sort_values(
        ["profile", "task", "evaluation_fold", "trial"]
    ).reset_index(drop=True)
    trials.to_csv(args.output_dir / "profile_trials.csv", index=False)
    task_summary, profile_summary = aggregate_profiles(trials, selection_tasks)
    task_summary.to_csv(args.output_dir / "task_profile_summary.csv", index=False)
    profile_summary.to_csv(args.output_dir / "shared_profile_summary.csv", index=False)
    loto = leave_one_task_out(task_summary, selection_tasks)
    loto.to_csv(args.output_dir / "leave_one_task_out.csv", index=False)

    selected_id = str(profile_summary.iloc[0]["profile_id"])
    selected = next(profile for profile in profiles if profile.profile_id() == selected_id)
    (args.output_dir / "selected_shared_profile.json").write_text(
        json.dumps(
            {"schema_version": SCHEMA, "profile_id": selected_id, **asdict(selected)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    policies: dict[str, Any] = {}
    for task, bundle in bundles.items():
        policy = materialize_policy(
            selected,
            task=task,
            candidate_count=len(bundle["public"]),
            smallest_budget=min(bundle["budgets"]),
            horizon=int(bundle["horizon"]),
        )
        policy_path = args.output_dir / "policies" / f"{task}.json"
        write_neurodiscovery_policy(policy_path, policy)
        policies[task] = {
            "path": str(policy_path.resolve()),
            "sha256": sha256_file(policy_path),
            "policy_id": policy.policy_id(),
        }
    manifest = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "status": "complete",
        "tasks": list(tasks),
        "profile_count": len(profiles),
        "profile_slate": args.profile_slate,
        "trials": args.trials,
        "folds": args.folds,
        "outer_holdout_fold": args.holdout_fold,
        "selection_tasks": list(selection_tasks),
        "audit_only_tasks": list(audit_only_tasks),
        "minimum_positives_for_selection": args.min_selection_positives,
        "selection_uses_external_outcomes": False,
        "selection_uses_outer_holdout": False,
        "task_weighting": "equal after within-task normalization",
        "selection_rule": (
            "Each inner fold is masked from feedback and scored out of fold; "
            "within-task score is 0.75 cross-fit mean plus 0.25 worst fold, then "
            "shared selection is 0.75 mean plus 0.25 worst-task relative score"
        ),
        "selected_shared_profile": {
            "profile_id": selected_id,
            **asdict(selected),
        },
        "kg": {"path": str(args.kg), "sha256": sha256_file(args.kg)},
        "bundles": {
            task: {
                "table_directory": str(paths[task].resolve()),
                "candidate_count": len(bundle["public"]),
                "validated_count": int(np.asarray(bundle["labels"], dtype=bool).sum()),
                "factor_fields": list(bundle["factor_fields"]),
                "group_fields": list(bundle["group_fields"]),
                "budgets": list(bundle["budgets"]),
                "horizon": int(bundle["horizon"]),
            }
            for task, bundle in bundles.items()
        },
        "policies": policies,
        "artifacts": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in (
                "profile_trials.csv",
                "task_profile_summary.csv",
                "shared_profile_summary.csv",
                "leave_one_task_out.csv",
                "selected_shared_profile.json",
            )
        },
    }
    manifest_path = args.output_dir / "shared_tuning_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
