"""Freeze fixed NeuroDiscovery policies for a new KG benchmark release.

The source profiles must have been selected without external outcomes. This
script never opens target internal or external outcome tables. It materializes
the best source-selected static profile and the best source-selected non-static
profile for every target task, then seals all inputs and policies by SHA-256.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.scripts.case_study_neurodiscovery_policy import (
    write_neurodiscovery_policy,
)
from core.scripts.tune_shared_neurodiscovery_policy import (
    SharedProfile,
    materialize_policy,
    profiles_for_slate,
)


SCHEMA = "cross-release-neurodiscovery-policy-bundle.v1"
MATERIALIZATION_REVISION = (
    "reserve-one-feedback-active-round-and-static-until-informative.v2"
)
REQUIRED_RELATION_COLUMNS = {
    "kg_relation_global_node_support",
    "kg_relation_global_pair_support",
    "kg_relation_scoped_node_support",
    "kg_relation_scoped_pair_support",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _implementation_record() -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    files = {
        "freezer": Path(__file__).resolve(),
        "shared_policy_materializer": script_dir
        / "tune_shared_neurodiscovery_policy.py",
        "policy_contract": script_dir / "case_study_neurodiscovery_policy.py",
        "closed_loop_engine": script_dir / "case_study_closed_loop_engine.py",
        "guard_evaluator": script_dir
        / "evaluate_cross_release_feedback_guard.py",
        "benchmark_runner": script_dir
        / "run_cross_release_case_study_benchmarks.py",
    }
    return {
        "materialization_revision": MATERIALIZATION_REVISION,
        "materialization_rule": (
            "For every non-static policy with at least two scheduled ranking "
            "rounds, warm-up is capped at planned_rounds - 1. Seeded "
            "perturbation and diversity remain inactive until the feedback gate "
            "is open and at least one supported or contradicted result exists."
        ),
        "outcome_labels_used_for_revision": False,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }


def _source_profiles(
    source_manifest: dict[str, Any],
    summary: pd.DataFrame,
) -> tuple[SharedProfile, SharedProfile, dict[str, dict[str, Any]]]:
    if bool(source_manifest.get("selection_uses_external_outcomes")):
        raise ValueError("source policy selection used external outcomes")
    if bool(source_manifest.get("selection_uses_outer_holdout")):
        raise ValueError("source policy selection used the outer holdout")
    slate = str(source_manifest.get("profile_slate") or "")
    profiles = {profile.name: profile for profile in profiles_for_slate(slate)}
    selected = source_manifest.get("selected_shared_profile") or {}
    static_name = str(selected.get("name") or "")
    if static_name not in profiles:
        raise ValueError(f"selected source profile is absent from slate: {static_name}")
    static_profile = profiles[static_name]
    if not static_profile.static_only:
        raise ValueError("source-selected primary profile is not static")
    if str(selected.get("profile_id")) != static_profile.profile_id():
        raise ValueError("source-selected profile ID does not match its definition")

    required = {"profile", "profile_id", "robust_cross_task_score"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"source profile summary lacks columns: {missing}")
    candidates = summary.loc[
        summary["profile"].isin(
            [name for name, profile in profiles.items() if not profile.static_only]
        )
    ].copy()
    if candidates.empty:
        raise ValueError("source slate contains no evaluated non-static profile")
    candidates["robust_cross_task_score"] = pd.to_numeric(
        candidates["robust_cross_task_score"], errors="raise"
    )
    candidates = candidates.sort_values(
        ["robust_cross_task_score", "profile"], ascending=[False, True]
    )
    closed_name = str(candidates.iloc[0]["profile"])
    closed_profile = profiles[closed_name]
    if str(candidates.iloc[0]["profile_id"]) != closed_profile.profile_id():
        raise ValueError("best non-static profile ID does not match its definition")

    metrics: dict[str, dict[str, Any]] = {}
    for profile in (static_profile, closed_profile):
        row = summary.loc[summary["profile"].eq(profile.name)]
        if len(row) != 1:
            raise ValueError(f"expected one summary row for {profile.name}")
        metrics[profile.name] = row.iloc[0].to_dict()
    return static_profile, closed_profile, metrics


def _task_dirs(root: Path, requested: Iterable[str] | None) -> list[Path]:
    wanted = set(requested or [])
    task_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "tables" / "table_manifest.json").is_file()
        and (path / "benchmark" / "run_manifest.json").is_file()
    ]
    if wanted:
        found = {path.name for path in task_dirs}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"target benchmark lacks tasks: {missing}")
        task_dirs = [path for path in task_dirs if path.name in wanted]
    return sorted(task_dirs, key=lambda path: path.name)


def _guard_evaluation(
    path: Path | None,
    *,
    source_manifest_path: Path,
    closed_profile: SharedProfile,
) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    payload = _load_json(path)
    if str(payload.get("status")) != "complete":
        raise ValueError("informative-feedback guard evaluation is incomplete")
    decision = dict(payload.get("decision") or {})
    if not bool(decision.get("accepted")):
        raise ValueError("informative-feedback guard failed its frozen selection gate")
    if bool(payload.get("external_outcomes_opened")):
        raise ValueError("guard evaluation opened external outcomes")
    if bool(payload.get("target_release_opened")):
        raise ValueError("guard evaluation opened the target release")
    recorded_source = dict(payload.get("source_manifest") or {})
    if str(recorded_source.get("sha256") or "") != sha256_file(
        source_manifest_path
    ):
        raise ValueError("guard evaluation used a different source tuning manifest")
    selected_dynamic = str(
        dict(payload.get("profile_selection") or {})
        .get("dynamic_profile", {})
        .get("name", "")
    )
    if selected_dynamic != closed_profile.name:
        raise ValueError("guard evaluation used a different dynamic profile")
    artifacts: dict[str, Any] = {}
    for name, descriptor in dict(payload.get("artifacts") or {}).items():
        descriptor = dict(descriptor)
        artifact_path = Path(str(descriptor.get("path") or ""))
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual = sha256_file(artifact_path)
        if actual != str(descriptor.get("sha256") or ""):
            raise ValueError(f"guard artifact failed SHA-256 verification: {name}")
        artifacts[str(name)] = {"path": str(artifact_path), "sha256": actual}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "accepted": True,
        "decision": decision,
        "artifacts": artifacts,
        "external_outcomes_opened": False,
        "target_release_opened": False,
    }


def freeze_bundle(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_tuning_root.resolve()
    benchmark_root = args.target_benchmark_root.resolve()
    target_kg = args.target_kg.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_root / "shared_tuning_manifest.json"
    source_summary_path = source_root / "shared_profile_summary.csv"
    source_trials_path = source_root / "profile_trials.csv"
    source_manifest = _load_json(source_manifest_path)
    source_summary = pd.read_csv(source_summary_path)
    static_profile, closed_profile, source_metrics = _source_profiles(
        source_manifest, source_summary
    )
    guard_evaluation = _guard_evaluation(
        getattr(args, "guard_evaluation_manifest", None),
        source_manifest_path=source_manifest_path,
        closed_profile=closed_profile,
    )

    target_kg_sha = sha256_file(target_kg)
    if args.expected_target_kg_sha256:
        expected = args.expected_target_kg_sha256.lower()
        if target_kg_sha.lower() != expected:
            raise ValueError(
                f"target KG SHA mismatch: expected {expected}, found {target_kg_sha}"
            )

    task_records: dict[str, Any] = {}
    for task_dir in _task_dirs(benchmark_root, args.tasks):
        table_dir = task_dir / "tables"
        table_manifest_path = table_dir / "table_manifest.json"
        run_manifest_path = task_dir / "benchmark" / "run_manifest.json"
        table_manifest = _load_json(table_manifest_path)
        run_manifest = _load_json(run_manifest_path)
        task = str(table_manifest.get("task") or task_dir.name)
        if task != task_dir.name or str(run_manifest.get("task")) != task:
            raise ValueError(f"task identity mismatch in {task_dir}")
        table_kg_sha = str(
            (table_manifest.get("kg_scoring") or {}).get("kg_sha256") or ""
        ).lower()
        if table_kg_sha != target_kg_sha.lower():
            raise ValueError(
                f"{task} table KG SHA {table_kg_sha} does not match target KG"
            )

        public_path = table_dir / "public_candidates.csv"
        header = set(pd.read_csv(public_path, nrows=0).columns)
        missing_columns = sorted(REQUIRED_RELATION_COLUMNS - header)
        if missing_columns:
            raise ValueError(
                f"{task} public candidates lack relation-aware columns: "
                f"{missing_columns}"
            )
        candidate_count = int(run_manifest["candidate_count"])
        budgets = sorted({int(value) for value in run_manifest["budgets"]})
        non_full = [value for value in budgets if value < candidate_count]
        horizon = max(non_full or budgets)
        smallest_budget = min(budgets)

        policies: dict[str, Any] = {}
        for track, profile in (
            ("static_ablation", static_profile),
            ("closed_loop", closed_profile),
        ):
            policy = materialize_policy(
                profile,
                task=task,
                candidate_count=candidate_count,
                smallest_budget=smallest_budget,
                horizon=horizon,
            )
            policy_path = output_dir / "policies" / track / f"{task}.json"
            write_neurodiscovery_policy(policy_path, policy)
            policies[track] = {
                "profile": profile.name,
                "profile_id": profile.profile_id(),
                "policy_id": policy.policy_id(),
                "path": str(policy_path),
                "sha256": sha256_file(policy_path),
                "feedback_horizon": int(policy.feedback_horizon or horizon),
                "static_only": bool(profile.static_only),
                "preserve_static_until_informative_feedback": bool(
                    policy.preserve_static_until_informative_feedback
                ),
            }
        task_records[task] = {
            "candidate_count": candidate_count,
            "factor_fields": list(run_manifest["factor_fields"]),
            "budgets": budgets,
            "recall_targets": list(run_manifest["recall_targets"]),
            "smallest_budget": smallest_budget,
            "comparison_horizon": horizon,
            "table_manifest": {
                "path": str(table_manifest_path),
                "sha256": sha256_file(table_manifest_path),
            },
            "source_run_manifest": {
                "path": str(run_manifest_path),
                "sha256": sha256_file(run_manifest_path),
            },
            "public_candidates": {
                "path": str(public_path),
                "sha256": sha256_file(public_path),
            },
            "policies": policies,
        }

    source_kg = source_manifest.get("kg") or {}
    manifest = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "status": "frozen",
        "selection_protocol": {
            "source_profile_slate": source_manifest.get("profile_slate"),
            "source_folds": source_manifest.get("folds"),
            "source_trials": source_manifest.get("trials"),
            "source_task_weighting": source_manifest.get("task_weighting"),
            "source_selection_rule": source_manifest.get("selection_rule"),
            "external_outcomes_used_for_selection": False,
            "outer_holdout_used_for_selection": False,
            "target_internal_outcome_tables_opened": False,
            "target_external_outcome_tables_opened": False,
            "informative_feedback_guard_selected_on_source_release": bool(
                guard_evaluation is not None
            ),
            "investigators_had_seen_prior_target_release_results": True,
            "evidence_tier": (
                "fixed cross-release reapplication; target labels were not used to "
                "select or materialize policies, but prior target-release results "
                "were historically visible to investigators"
            ),
        },
        "implementation": _implementation_record(),
        "informative_feedback_guard_evaluation": guard_evaluation,
        "tracks": {
            "static_ablation": {
                "role": "relation-aware static ranking ablation",
                "profile": static_profile.name,
                "profile_id": static_profile.profile_id(),
                "source_metrics": source_metrics[static_profile.name],
            },
            "closed_loop": {
                "role": "best source-selected non-static closed-loop policy",
                "profile": closed_profile.name,
                "profile_id": closed_profile.profile_id(),
                "source_metrics": source_metrics[closed_profile.name],
            },
        },
        "source_release": {
            "kg_path_recorded_at_selection": source_kg.get("path"),
            "kg_sha256": source_kg.get("sha256"),
            "tuning_manifest": {
                "path": str(source_manifest_path),
                "sha256": sha256_file(source_manifest_path),
            },
            "profile_summary": {
                "path": str(source_summary_path),
                "sha256": sha256_file(source_summary_path),
            },
            "profile_trials": {
                "path": str(source_trials_path),
                "sha256": sha256_file(source_trials_path),
            },
        },
        "target_release": {
            "kg_path": str(target_kg),
            "kg_sha256": target_kg_sha,
            "benchmark_root": str(benchmark_root),
        },
        "tasks": task_records,
    }
    manifest_path = output_dir / "cross_release_policy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_sha = sha256_file(manifest_path)
    (output_dir / "cross_release_policy_manifest.sha256").write_text(
        f"{manifest_sha}  {manifest_path.name}\n", encoding="ascii"
    )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tuning-root", type=Path, required=True)
    parser.add_argument("--target-benchmark-root", type=Path, required=True)
    parser.add_argument("--target-kg", type=Path, required=True)
    parser.add_argument("--expected-target-kg-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--guard-evaluation-manifest", type=Path)
    return parser


def main() -> None:
    result = freeze_bundle(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
