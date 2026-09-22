"""Run the locked v3 smoke or formal matrix from an immutable source bundle."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from neurooracle.scripts.prepare_hindcasting_v3_execution import (
    V3_ROOT,
    verify_execution_plan,
)
from neurooracle.src.experiment_source_bundle import (
    sha256_file,
    verify_bundle_member,
    verify_source_bundle,
)
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    V3_ROOT
    / "execution/locked_r1_two_tasks_20260826/execution_plan.json"
)
STAGES = (
    "neurodiscovery",
    "baseline_generation",
    "baseline_evaluation",
    "verify",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_bundle_identity(record: Mapping[str, Any] | None) -> tuple[str, str]:
    if not record:
        return "", ""
    return (
        str(record.get("manifest_path") or ""),
        str(record.get("manifest_sha256") or "").upper(),
    )


def _derive_smoke_eligibility(
    *, plan: Mapping[str, Any], output_root: Path
) -> Path:
    formal_path = Path(plan["inputs"]["dynamic_eligibility"]["path"]).resolve()
    formal = load_locked_hindcasting_eligibility(formal_path)
    smoke = plan["smoke"]
    window = smoke["window"]
    selected = (
        str(smoke["task_id"]),
        int(window["freeze_year"]),
        int(window["future_start_year"]),
        int(window["future_end_year"]),
    )
    if selected not in formal.primary_windows:
        raise ValueError(f"smoke cell is not dynamically eligible: {selected}")

    lock_root = output_root / "smoke_eligibility"
    matrix_path = lock_root / "dynamic_eligibility_matrix_locked.csv"
    manifest_path = lock_root / "dynamic_eligibility_manifest.json"
    lock_root.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_study_id",
                "freeze_year",
                "future_start_year",
                "future_end_year",
                "analysis_tier",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_study_id": selected[0],
                "freeze_year": selected[1],
                "future_start_year": selected[2],
                "future_end_year": selected[3],
                "analysis_tier": "primary",
            }
        )
    payload = {
        "schema_version": "neurodiscovery-dynamic-hindcasting-eligibility.v1",
        "registered_at": str(plan["created_at"]),
        "status": "derived_smoke_only_not_formal_analysis",
        "feedback_years": 2,
        "primary_case_study_ids": [selected[0]],
        "primary_windows": 1,
        "locked_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": sha256_file(matrix_path),
            "rows": 1,
        },
        "source_formal_eligibility_manifest": str(formal_path),
        "source_formal_eligibility_manifest_sha256": sha256_file(formal_path),
        "selection_contract": {
            "performance_columns_consumed": False,
            "task_or_window_selection_after_method_scoring_permitted": False,
        },
    }
    _atomic_json(manifest_path, payload)
    return manifest_path


def _option(command: list[str], name: str, value: Any) -> None:
    command.extend((name, str(value)))


def build_commands(
    *,
    plan: Mapping[str, Any],
    bundle_manifest: Path,
    python: Path,
    code_root: Path,
    mode: str,
    output_root: Path,
    eligibility_manifest: Path,
    force: bool,
) -> dict[str, list[str]]:
    matrix = plan["matrix"]
    config = dict(plan["configuration"]["neurodiscovery"])
    task_ids = list(plan["cohort"]["task_ids"])
    windows = list(matrix["windows"])
    seeds = list(matrix["seeds"])
    budgets = list(matrix["experiment_counts"])
    if mode == "smoke":
        smoke = plan["smoke"]
        task_ids = [str(smoke["task_id"])]
        windows = [dict(smoke["window"])]
        seeds = [int(smoke["seed"])]
        budgets = [int(value) for value in smoke["experiment_counts"]]
        config.update(
            {
                "proposal_batch_size": min(int(config["proposal_batch_size"]), 120),
                "max_unique_proposals": min(int(config["max_unique_proposals"]), 240),
                "max_generation_rounds": min(int(config["max_generation_rounds"]), 2),
                "max_executions": int(smoke["maximum_executions"]),
                "refill_below": min(int(config["refill_below"]), 100),
            }
        )
    window_tokens = [
        f"{row['freeze_year']}:{row['future_start_year']}:{row['future_end_year']}"
        for row in windows
    ]
    common_matrix = [
        "--snapshot-root",
        str(Path(plan["inputs"]["snapshot_root"])),
        "--eligibility-manifest",
        str(eligibility_manifest),
        "--case-study-ids",
        *task_ids,
        "--windows",
        *window_tokens,
        "--seeds",
        *(str(value) for value in seeds),
    ]
    source_guard = [
        "--source-bundle-manifest",
        str(bundle_manifest),
        "--skip-source-bundle-reference-rehash",
    ]
    force_args = ["--force"] if force else []

    neuro = [
        str(python),
        str(
            code_root
            / "neurooracle/scripts/run_neurodiscovery_dynamic_closed_loop_hindcasting.py"
        ),
        *common_matrix,
        *source_guard,
        "--future-claims",
        str(Path(plan["inputs"]["future_claims"]["path"])),
        "--output-root",
        str(output_root / "neurodiscovery"),
        "--index-cache-root",
        str(output_root / "index_cache"),
        "--budgets",
        *(str(value) for value in budgets),
        "--profile",
        str(config["profile"]),
        "--feedback-enabled",
    ]
    for option_name, config_name in (
        ("--proposal-batch-size", "proposal_batch_size"),
        ("--max-unique-proposals", "max_unique_proposals"),
        ("--max-generation-rounds", "max_generation_rounds"),
        ("--max-stagnant-generation-rounds", "max_stagnant_generation_rounds"),
        ("--max-executions", "max_executions"),
        ("--execution-batch-size", "execution_batch_size"),
        ("--refresh-every-executions", "refresh_every_executions"),
        ("--refill-below", "refill_below"),
        ("--feedback-years", "feedback_years"),
        ("--feedback-start-budget", "feedback_start_budget"),
        ("--min-supported-before-feedback", "min_supported_before_feedback"),
        ("--seed-diversity-fraction", "seed_diversity_fraction"),
        ("--candidate-pool-mode", "candidate_pool_mode"),
        ("--task-scope-fraction", "task_scope_fraction"),
        ("--evidence-frontier-fraction", "evidence_frontier_fraction"),
        ("--protect-general-top-k", "protect_general_top_k"),
        ("--path-variants-per-endpoint", "path_variants_per_endpoint"),
        ("--feedback-mutation-fraction", "feedback_mutation_fraction"),
        ("--max-paths-per-endpoint", "max_paths_per_endpoint"),
        ("--endpoint-canonical-quality-weight", "endpoint_canonical_quality_weight"),
        ("--kge-weight", "kge_weight"),
    ):
        _option(neuro, option_name, config[config_name])
    neuro.extend(
        (
            "--kge-root",
            str(Path(plan["inputs"]["kge_root"])),
            "--kge-checkpoint-pattern",
            str(config["kge_checkpoint_pattern"]),
            "--kge-device",
            "cuda",
            *force_args,
        )
    )

    baseline_target = 100 if mode == "smoke" else int(
        plan["configuration"]["baselines"]["target_per_case_study"]
    )
    generation_root = output_root / "baseline_generation"
    baseline_generation = [
        str(python),
        str(code_root / "neurooracle/scripts/generate_case_study_frozen_baselines.py"),
        *common_matrix,
        *source_guard,
        "--output-root",
        str(generation_root),
        "--methods",
        "sciagents",
        "openscholar_rag",
        "--target-per-case-study",
        str(baseline_target),
        *force_args,
    ]
    baseline_evaluation = [
        str(python),
        str(code_root / "neurooracle/scripts/evaluate_case_study_frozen_baselines.py"),
        "--generation-root",
        str(generation_root),
        "--output-root",
        str(output_root / "baseline_evaluation"),
        "--snapshot-root",
        str(Path(plan["inputs"]["snapshot_root"])),
        "--future-claims",
        str(Path(plan["inputs"]["future_claims"]["path"])),
        *source_guard,
        "--case-study-ids",
        *task_ids,
        "--methods",
        "sciagents",
        "openscholar_rag",
        "--seeds",
        *(str(value) for value in seeds),
        "--windows",
        *window_tokens,
        "--top-k",
        *(str(value) for value in budgets),
        "--random-trials",
        "50"
        if mode == "smoke"
        else str(plan["configuration"]["baselines"]["random_trials"]),
        *force_args,
    ]
    return {
        "neurodiscovery": neuro,
        "baseline_generation": baseline_generation,
        "baseline_evaluation": baseline_evaluation,
    }


def _run_subprocess(
    *,
    stage: str,
    command: Sequence[str],
    output_root: Path,
    environment: Mapping[str, str],
    code_root: Path,
) -> dict[str, Any]:
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{stage}.stdout.log"
    stderr_path = logs / f"{stage}.stderr.log"
    started = _utc_now()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            list(command),
            cwd=code_root,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    record = {
        "stage": stage,
        "started_at": started,
        "finished_at": _utc_now(),
        "command": list(command),
        "returncode": result.returncode,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {result.returncode}; see {stderr_path}"
        )
    return record


def _expected_cells(plan: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    matrix = _read_json(Path(plan["matrix"]["path"]))
    rows = list(matrix["rows"])
    if mode == "formal":
        return rows
    smoke = plan["smoke"]
    window = smoke["window"]
    return [
        row
        for row in rows
        if row["task_id"] == smoke["task_id"]
        and int(row["freeze_year"]) == int(window["freeze_year"])
        and int(row["seed"]) == int(smoke["seed"])
    ]


def _adaptive_feedback_start_budget(
    *, configured_budget: int, execution_batch_size: int, initial_valid: int
) -> int:
    """Mirror the dynamic runner's locked sparse-task feedback gate."""

    return min(
        configured_budget,
        max(execution_batch_size, (max(1, initial_valid) + 3) // 4),
    )


def verify_outputs(
    *,
    plan: Mapping[str, Any],
    mode: str,
    output_root: Path,
    bundle_manifest: Path,
) -> dict[str, Any]:
    expected = _expected_cells(plan, mode)
    expected_bundle_hash = sha256_file(bundle_manifest)
    observed: list[dict[str, Any]] = []
    for row in expected:
        method = str(row["method"])
        task = str(row["task_id"])
        freeze = int(row["freeze_year"])
        start = int(row["future_start_year"])
        end = int(row["future_end_year"])
        seed = int(row["seed"])
        label = f"kg{freeze}_to_{start}_{end}"
        if method == "neurodiscovery":
            artifact = (
                output_root
                / "neurodiscovery"
                / f"seed_{seed:02d}"
                / task
                / label
                / "run_manifest.json"
            )
            payload = _read_json(artifact)
            if payload.get("status") != "complete":
                raise ValueError(f"incomplete NeuroDiscovery cell: {artifact}")
            if (
                str(payload.get("case_study_id")) != task
                or int(payload.get("seed", -1)) != seed
                or int(payload.get("freeze_year", -1)) != freeze
            ):
                raise ValueError(f"NeuroDiscovery cell identity mismatch: {artifact}")
            bundle = payload.get("source_bundle") or {}
            if str(bundle.get("manifest_sha256") or "").upper() != expected_bundle_hash:
                raise ValueError(f"NeuroDiscovery source bundle mismatch: {artifact}")
            isolation = payload.get("temporal_isolation") or {}
            if (
                isolation.get("terminal_labels_available_to_generator") is not False
                or isolation.get("formal_kg_mutated") is not False
            ):
                raise ValueError(f"temporal isolation failed: {artifact}")
            if not (payload.get("kge_prior") or {}).get("enabled"):
                raise ValueError(f"NeuroDiscovery KGE prior is disabled: {artifact}")
        else:
            artifact = (
                output_root
                / "baseline_evaluation"
                / method
                / f"seed_{seed:02d}"
                / task
                / label
                / "hindcasting/metrics.json"
            )
            payload = _read_json(artifact)
            identity = payload.get("execution_identity") or {}
            if (
                identity.get("method") != method
                or int(identity.get("seed", -1)) != seed
                or identity.get("case_study_id") != task
                or int(identity.get("freeze_year", -1)) != freeze
            ):
                raise ValueError(f"baseline cell identity mismatch: {artifact}")
            bundle = identity.get("source_bundle") or {}
            if str(bundle.get("manifest_sha256") or "").upper() != expected_bundle_hash:
                raise ValueError(f"baseline source bundle mismatch: {artifact}")
        observed.append(
            {
                "cell_id": row["cell_id"],
                "artifact": str(artifact.resolve()),
                "sha256": sha256_file(artifact),
            }
        )

    if mode == "smoke":
        neuro_artifact = Path(observed[0]["artifact"])
        neuro_payload = _read_json(neuro_artifact)
        if int(neuro_payload.get("executed_hypotheses") or 0) != int(
            plan["smoke"]["maximum_executions"]
        ):
            raise ValueError("smoke did not execute the locked diagnostic budget")
        neuro_config = neuro_payload.get("config") or {}
        configured_budget = int(neuro_config.get("feedback_start_budget") or -1)
        locked_budget = int(
            plan["configuration"]["neurodiscovery"]["feedback_start_budget"]
        )
        if configured_budget != locked_budget:
            raise ValueError("smoke configured feedback boundary changed")
        generation_path = neuro_artifact.parent / "generation_rounds.csv"
        with generation_path.open("r", encoding="utf-8-sig", newline="") as handle:
            generation_rows = list(csv.DictReader(handle))
        initial_row = next(
            (row for row in generation_rows if int(row.get("round") or -1) == 0),
            None,
        )
        if initial_row is None:
            raise ValueError("smoke initial generation round is missing")
        expected_feedback_start = _adaptive_feedback_start_budget(
            configured_budget=configured_budget,
            execution_batch_size=int(neuro_config.get("execution_batch_size") or 0),
            initial_valid=int(initial_row.get("new_unique_candidates") or 0),
        )
        if int(neuro_payload.get("feedback_start_budget") or -1) != expected_feedback_start:
            raise ValueError("smoke adaptive feedback boundary mismatch")

    report = {
        "schema_version": "neurodiscovery-hindcasting-v3-output-verification.v1",
        "status": "passed",
        "mode": mode,
        "verified_at": _utc_now(),
        "protocol_sha256": str(plan["protocol"]["sha256"]),
        "cohort_sha256": plan["cohort"]["sha256"],
        "source_bundle_manifest_sha256": expected_bundle_hash,
        "expected_cells": len(expected),
        "verified_cells": len(observed),
        "cells": observed,
    }
    _atomic_json(output_root / "output_verification.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    verified_plan = verify_execution_plan(plan_path)
    plan = _read_json(plan_path)
    bundle_manifest = (
        args.source_bundle_manifest.resolve()
        if args.source_bundle_manifest is not None
        else Path(verified_plan["source_bundle_manifest"]).resolve()
    )
    if bundle_manifest != Path(plan["outputs"]["source_bundle_manifest"]).resolve():
        raise ValueError("runner was given a different source bundle")
    verify_bundle_member(bundle_manifest, plan_path)

    archive_root = (bundle_manifest.parent / "files").resolve()
    code_root = Path(__file__).resolve().parents[2]
    executing_from_archive = Path(__file__).resolve().is_relative_to(archive_root)
    if not executing_from_archive and not args.allow_live_source:
        raise RuntimeError(
            "formal v3 runner must execute from its frozen archive; "
            "--allow-live-source is diagnostic only"
        )
    if executing_from_archive and code_root != archive_root:
        raise RuntimeError("archive code root does not match bundle files root")

    output_root = Path(
        plan["smoke"]["output_root"]
        if args.mode == "smoke"
        else plan["outputs"]["formal_root"]
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_manifest = (
        _derive_smoke_eligibility(plan=plan, output_root=output_root)
        if args.mode == "smoke"
        else Path(plan["inputs"]["dynamic_eligibility"]["path"]).resolve()
    )
    bundle_at_start = verify_source_bundle(
        bundle_manifest,
        require_live_source=not executing_from_archive,
        verify_references=True,
    )
    commands = build_commands(
        plan=plan,
        bundle_manifest=bundle_manifest,
        python=args.python.resolve(),
        code_root=code_root,
        mode=args.mode,
        output_root=output_root,
        eligibility_manifest=eligibility_manifest,
        force=args.force,
    )
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(code_root), environment.get("PYTHONPATH", ""))
        if value
    )
    records: list[dict[str, Any]] = []
    execution_manifest_path = output_root / "execution_manifest.json"
    base_manifest = {
        "schema_version": "neurodiscovery-hindcasting-v3-execution.v1",
        "mode": args.mode,
        "plan": str(plan_path),
        "plan_sha256": verified_plan["plan_sha256"],
        "cohort_id": verified_plan["cohort_id"],
        "source_bundle": bundle_at_start,
        "eligibility_manifest": str(eligibility_manifest),
        "llm_api_requests": 0,
        "llm_api_routing_policy_sha256": plan["llm_api"][
            "routing_policy_if_required"
        ]["sha256"],
    }
    _atomic_json(
        execution_manifest_path,
        {**base_manifest, "status": "running", "records": records},
    )
    try:
        for stage in selected:
            shallow_before = verify_source_bundle(
                bundle_manifest,
                require_live_source=not executing_from_archive,
                verify_references=False,
            )
            if stage == "verify":
                started = _utc_now()
                verification = verify_outputs(
                    plan=plan,
                    mode=args.mode,
                    output_root=output_root,
                    bundle_manifest=bundle_manifest,
                )
                record = {
                    "stage": stage,
                    "started_at": started,
                    "finished_at": _utc_now(),
                    "returncode": 0,
                    "verified_cells": verification["verified_cells"],
                }
            else:
                record = _run_subprocess(
                    stage=stage,
                    command=commands[stage],
                    output_root=output_root,
                    environment=environment,
                    code_root=code_root,
                )
            shallow_after = verify_source_bundle(
                bundle_manifest,
                require_live_source=not executing_from_archive,
                verify_references=False,
            )
            if shallow_before != shallow_after:
                raise ValueError(f"source bundle changed during stage {stage}")
            record["source_bundle_before"] = shallow_before
            record["source_bundle_after"] = shallow_after
            records.append(record)
            _atomic_json(
                execution_manifest_path,
                {**base_manifest, "status": "running", "records": records},
            )
    except BaseException as exc:
        _atomic_json(
            execution_manifest_path,
            {
                **base_manifest,
                "status": "failed",
                "records": records,
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": _utc_now(),
            },
        )
        raise

    bundle_at_completion = verify_source_bundle(
        bundle_manifest,
        require_live_source=not executing_from_archive,
        verify_references=True,
    )
    if bundle_at_completion != bundle_at_start:
        raise ValueError("deep source-bundle verification changed during execution")
    result = {
        **base_manifest,
        "status": "complete",
        "completed_at": _utc_now(),
        "records": records,
        "source_bundle_at_completion": bundle_at_completion,
    }
    _atomic_json(execution_manifest_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--source-bundle-manifest", type=Path)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--python", type=Path, default=Path(os.sys.executable))
    parser.add_argument("--allow-live-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": result["mode"],
                "cohort_id": result["cohort_id"],
                "stages": [record["stage"] for record in result["records"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
