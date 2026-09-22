"""Recover a completed guard evaluation from its legacy variable-width CSV."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from core.scripts.case_study_neurodiscovery_policy import (
    apply_neurodiscovery_policy,
)
from core.scripts.evaluate_cross_release_feedback_guard import (
    SCHEMA,
    _policy_for_variant,
    aggregate_trials,
    guard_decision,
    load_sealed_task,
    read_json,
    select_source_profiles,
    sha256_file,
    write_json,
)


RECOVERY_SCHEMA = "cross-release-feedback-guard-recovery.v1"
COMMON_COLUMNS = (
    "variant",
    "task",
    "profile",
    "profile_id",
    "evaluation_role",
    "evaluation_fold",
    "masked_feedback_folds",
    "trial",
    "seed",
    "policy_id",
    "guard_enabled",
    "guard_activated",
    "ranking_reads_after_feedback",
    "nonzero_feedback_reads",
    "ranking_sha256",
    "objective",
    "target_total",
)
INTEGER_COLUMNS = {
    "evaluation_fold",
    "trial",
    "seed",
    "ranking_reads_after_feedback",
    "nonzero_feedback_reads",
    "target_total",
}
BOOLEAN_COLUMNS = {"guard_enabled", "guard_activated"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid serialized boolean: {value!r}")


def recover_trial_rows(
    path: Path,
    bundles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.reader(handle))
    if not raw or tuple(raw[0][: len(COMMON_COLUMNS)]) != COMMON_COLUMNS:
        raise ValueError("legacy trial table has an unexpected common header")
    rows: list[dict[str, Any]] = []
    for line_number, values in enumerate(raw[1:], start=2):
        if len(values) < len(COMMON_COLUMNS):
            raise ValueError(f"line {line_number} is shorter than the common schema")
        task = values[1]
        if task not in bundles:
            raise ValueError(f"line {line_number} has unknown task {task!r}")
        budgets = tuple(map(int, bundles[task]["budgets"]))
        columns = list(COMMON_COLUMNS)
        for budget in budgets:
            columns.extend((f"hits_at_{budget}", f"oracle_fraction_at_{budget}"))
        if len(values) != len(columns):
            raise ValueError(
                f"line {line_number} has {len(values)} values; expected {len(columns)}"
            )
        row: dict[str, Any] = dict(zip(columns, values, strict=True))
        for column in INTEGER_COLUMNS | {
            name for name in columns if name.startswith("hits_at_")
        }:
            row[column] = int(row[column])
        for column in BOOLEAN_COLUMNS:
            row[column] = _parse_bool(str(row[column]))
        row["objective"] = float(row["objective"])
        for column in columns:
            if column.startswith("oracle_fraction_at_"):
                row[column] = float(row[column])
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def recover(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_root = Path(args.evaluation_root).resolve()
    source_root = Path(args.source_root).resolve()
    manifest_path = evaluation_root / "guard_evaluation_manifest.json"
    legacy_path = evaluation_root / "guard_trials.csv"
    previous = read_json(manifest_path)
    if str(previous.get("schema_version")) != SCHEMA:
        raise ValueError("guard manifest schema mismatch")
    if str(previous.get("status")) not in {"running", "failed"}:
        raise ValueError("recovery is only valid for an incomplete guard manifest")

    source_manifest_path = source_root / "shared_tuning_manifest.json"
    source_manifest = read_json(source_manifest_path)
    if str(dict(previous.get("source_manifest") or {}).get("sha256")) != sha256_file(
        source_manifest_path
    ):
        raise ValueError("guard output and source tuning manifest differ")
    bundles = dict(source_manifest["bundles"])
    rows = recover_trial_rows(legacy_path, bundles)
    tasks = tuple(map(str, source_manifest["tasks"]))
    folds = int(source_manifest["folds"])
    trials = int(source_manifest["trials"])
    variants = ("static_control", "dynamic_original", "dynamic_guarded")
    expected = len(tasks) * len(variants) * folds * trials
    keys = {
        (
            str(row["variant"]),
            str(row["task"]),
            int(row["evaluation_fold"]),
            int(row["trial"]),
        )
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(
            f"legacy trial recovery is incomplete or duplicated: rows={len(rows)} "
            f"keys={len(keys)} expected={expected}"
        )

    frame = pd.DataFrame(rows)
    jsonl_path = evaluation_root / "guard_trials.jsonl"
    proper_csv_path = evaluation_root / "guard_trials.recovered.csv"
    _write_jsonl(jsonl_path, rows)
    frame.to_csv(proper_csv_path, index=False)
    selection_tasks = tuple(map(str, source_manifest["selection_tasks"]))
    task_summary, variant_summary = aggregate_trials(frame, selection_tasks)
    task_summary_path = evaluation_root / "guard_task_summary.csv"
    variant_summary_path = evaluation_root / "guard_variant_summary.csv"
    decision_path = evaluation_root / "guard_decision.json"
    task_summary.to_csv(task_summary_path, index=False)
    variant_summary.to_csv(variant_summary_path, index=False)
    decision = guard_decision(task_summary, variant_summary, selection_tasks)
    write_json(decision_path, decision)

    static_profile, dynamic_profile, profile_audit = select_source_profiles(
        source_root,
        source_manifest,
    )
    source_kg_sha = str(dict(source_manifest["kg"])["sha256"]).lower()
    input_audits: dict[str, Any] = {}
    policy_audits: dict[str, Any] = {}
    for task in tasks:
        bundle = load_sealed_task(
            task,
            bundles[task],
            source_root=source_root,
            source_kg_sha256=source_kg_sha,
            folds=folds,
        )
        input_audits[task] = bundle["input_audit"]
        for variant, profile in (
            ("static_control", static_profile),
            ("dynamic_original", dynamic_profile),
            ("dynamic_guarded", dynamic_profile),
        ):
            policy = _policy_for_variant(variant, profile, bundle)
            _, score_audit = apply_neurodiscovery_policy(bundle["public"], policy)
            policy_audits[f"{task}:{variant}"] = {
                "variant": variant,
                "profile": asdict(profile),
                "policy": policy.to_dict(),
                "policy_id": policy.policy_id(),
                "score_audit": score_audit,
            }
        del bundle

    recovery_script = Path(__file__).resolve()
    current_evaluator = (
        recovery_script.parent / "evaluate_cross_release_feedback_guard.py"
    )
    recovery_manifest_path = evaluation_root / "guard_recovery_manifest.json"
    recovery_manifest = {
        "schema_version": RECOVERY_SCHEMA,
        "created_at": utc_now(),
        "reason": "legacy append-only CSV used task-dependent budget columns",
        "ranking_recomputed": False,
        "recovered_rows": len(rows),
        "unique_evaluation_keys": len(keys),
        "source_legacy_csv": {
            "path": str(legacy_path),
            "sha256": sha256_file(legacy_path),
        },
        "recovery_script": {
            "path": str(recovery_script),
            "sha256": sha256_file(recovery_script),
        },
        "current_evaluator_after_output_fix": {
            "path": str(current_evaluator),
            "sha256": sha256_file(current_evaluator),
        },
        "original_ranking_implementation": previous.get("implementation"),
    }
    write_json(recovery_manifest_path, recovery_manifest)

    artifacts = {
        "guard_trials.jsonl": {
            "path": str(jsonl_path),
            "sha256": sha256_file(jsonl_path),
        },
        "guard_trials.recovered.csv": {
            "path": str(proper_csv_path),
            "sha256": sha256_file(proper_csv_path),
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
        "guard_recovery_manifest.json": {
            "path": str(recovery_manifest_path),
            "sha256": sha256_file(recovery_manifest_path),
        },
    }
    complete = {
        **previous,
        "status": "complete",
        "completed_at": utc_now(),
        "profile_selection": profile_audit,
        "external_outcomes_opened": False,
        "target_release_opened": False,
        "outer_holdout_used_for_selection": False,
        "input_audits": input_audits,
        "policy_audits": policy_audits,
        "decision": decision,
        "postprocessing_recovery": recovery_manifest,
        "artifacts": artifacts,
    }
    write_json(manifest_path, complete)
    return {
        "status": "complete",
        "accepted": bool(decision["accepted"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "recovery_manifest": str(recovery_manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(recover(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

