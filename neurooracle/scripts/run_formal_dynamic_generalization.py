"""Run the frozen cross-task dynamic closed-loop generalization benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from neurooracle.src.experiment_source_bundle import (
    sha256_file,
    verify_bundle_member,
    verify_source_bundle,
)
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
)
DEFAULT_DESIGN = (
    DEFAULT_PROTOCOL_ROOT / "formal_dynamic_generalization_design_20260813.json"
)
DEFAULT_BUNDLE = (
    ROOT
    / "neurooracle/.frozen/formal_dynamic_generalization_v1/bundle_manifest.json"
)
STAGES = ("closed", "open", "compare")
POLICY_METADATA_FIELDS = frozenset({"feedback_enabled", "selection_reason"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path, workspace_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _assert_hash(record: Mapping[str, Any], workspace_root: Path, label: str) -> Path:
    path = _resolve(str(record.get("path") or ""), workspace_root)
    expected = str(record.get("sha256") or "").upper()
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")
    return path


def _policy_runtime_configuration(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return only parameters that must be identical in the paired arms."""

    selected = dict(policy.get("selected_policy") or {})
    missing = POLICY_METADATA_FIELDS - set(selected)
    if missing:
        raise ValueError(f"frozen dynamic policy is missing fields: {sorted(missing)}")
    for field in POLICY_METADATA_FIELDS:
        selected.pop(field, None)
    return selected


def _verify_design(
    design_path: Path,
    workspace_root: Path,
) -> tuple[dict[str, Any], Path]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("status") != "frozen_before_dynamic_generalization":
        raise ValueError("dynamic generalization design is not frozen")

    canonical = design.get("canonical_release") or {}
    for label in ("knowledge_graph", "extracted_claims", "current_state"):
        _assert_hash(canonical[label], workspace_root, f"canonical {label}")
    policy_path = _assert_hash(
        design["frozen_policy"], workspace_root, "frozen dynamic policy"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "frozen_for_confirmatory_application":
        raise ValueError("dynamic policy is not frozen")
    if dict(design["shared_configuration"]) != _policy_runtime_configuration(policy):
        raise ValueError("dynamic design differs from the frozen selected policy")
    policy_development = {
        str(value)
        for value in (
            (policy.get("development_protocol") or {}).get("case_study_ids") or ()
        )
    }
    if set(design.get("development_case_study_ids") or ()) != policy_development:
        raise ValueError("dynamic design declares a different development task set")

    budget_contract = design.get("fixed_budget_contract") or {}
    budgets = [int(value) for value in budget_contract.get("rank_points") or ()]
    if not budgets or budgets != sorted(set(budgets)):
        raise ValueError("dynamic rank points must be unique and increasing")
    if budgets[-1] != int(design["shared_configuration"]["max_executions"]):
        raise ValueError("largest dynamic rank point must equal max_executions")
    primary_endpoint = design.get("endpoints", {}).get("primary") or {}
    if int(primary_endpoint.get("k") or 0) != budgets[-1]:
        raise ValueError("dynamic primary endpoint must use the maximum rank point")

    eligibility_path = _assert_hash(
        design["dynamic_eligibility"]["manifest"],
        workspace_root,
        "dynamic eligibility manifest",
    )
    eligibility = load_locked_hindcasting_eligibility(eligibility_path)
    matrix_path = _assert_hash(
        design["dynamic_eligibility"]["matrix"],
        workspace_root,
        "dynamic eligibility matrix",
    )
    if eligibility.matrix_path != matrix_path:
        raise ValueError("dynamic eligibility manifest declares a different matrix")
    expected_matrix = str(
        design["dynamic_eligibility"]["matrix"]["sha256"]
    ).upper()
    if eligibility.matrix_sha256 != expected_matrix:
        raise ValueError("dynamic eligibility matrix hash mismatch")
    primary = design["primary_matrix"]
    if len(eligibility.primary_windows) != int(primary["case_study_windows"]):
        raise ValueError("dynamic primary-window count differs from design")
    if list(eligibility.primary_case_study_ids) != list(primary["case_study_ids"]):
        raise ValueError("dynamic primary Case Studies differ from design")
    development = set(design.get("development_case_study_ids") or ())
    overlap = development & set(eligibility.primary_case_study_ids)
    if overlap:
        raise ValueError(
            f"development Case Studies entered dynamic primary matrix: {sorted(overlap)}"
        )
    if int(primary.get("paired_runs_per_arm") or 0) != len(
        eligibility.primary_windows
    ) * len(primary.get("seeds") or ()):
        raise ValueError("dynamic paired-run count differs from design")
    seeds = [int(value) for value in primary.get("seeds") or ()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("dynamic seeds are missing or duplicated")
    if int(eligibility.manifest.get("feedback_years") or 0) != int(
        design["shared_configuration"]["feedback_years"]
    ):
        raise ValueError("eligibility feedback interval differs from frozen policy")
    selection = eligibility.manifest.get("selection_contract") or {}
    if selection.get("performance_columns_consumed") is not False:
        raise ValueError("dynamic eligibility must be selected without method outcomes")
    if selection.get("task_or_window_selection_after_method_scoring_permitted") is not False:
        raise ValueError("dynamic eligibility permits post-outcome selection")
    if primary.get("previously_exposed_historical_years") is not True:
        raise ValueError("dynamic evidence tier must acknowledge historical exposure")
    if primary.get("claim_as_untouched_confirmatory_permitted") is not False:
        raise ValueError("historical dynamic benchmark cannot be called untouched")
    return design, eligibility_path


def _derive_smoke_lock(formal_manifest_path: Path, output_root: Path) -> Path:
    """Derive one latest primary window without changing the formal lock."""

    formal = load_locked_hindcasting_eligibility(formal_manifest_path)
    selected = max(formal.primary_windows, key=lambda key: (key[1], key))
    smoke_root = output_root / "smoke_lock"
    matrix_path = smoke_root / "dynamic_eligibility_matrix_locked.csv"
    manifest_path = smoke_root / "dynamic_eligibility_manifest.json"
    smoke_root.mkdir(parents=True, exist_ok=True)
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
        "registered_at": _utc_now(),
        "status": "derived_smoke_only_not_formal_analysis",
        "feedback_years": int(formal.manifest.get("feedback_years") or 0),
        "primary_case_study_ids": [selected[0]],
        "primary_windows": 1,
        "locked_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": sha256_file(matrix_path),
            "rows": 1,
        },
        "source_formal_eligibility_manifest": str(formal.manifest_path),
        "source_formal_eligibility_manifest_sha256": sha256_file(
            formal.manifest_path
        ),
        "selection_contract": {
            "performance_columns_consumed": False,
            "task_or_window_selection_after_method_scoring_permitted": False,
        },
    }
    _atomic_json(manifest_path, payload)
    return manifest_path


def _dynamic_command(
    *,
    python: Path,
    code_root: Path,
    design: Mapping[str, Any],
    workspace_root: Path,
    eligibility_manifest: Path,
    bundle_manifest: Path,
    output_root: Path,
    feedback_enabled: bool,
    force: bool,
    smoke: bool = False,
) -> list[str]:
    config = dict(design["shared_configuration"])
    matrix = design["primary_matrix"]
    budgets = [
        int(value)
        for value in design["fixed_budget_contract"]["rank_points"]
    ]
    case_ids = [str(value) for value in matrix["case_study_ids"]]
    seeds = [int(value) for value in matrix["seeds"]]
    if smoke:
        lock = load_locked_hindcasting_eligibility(eligibility_manifest)
        case_ids = list(lock.primary_case_study_ids)
        seeds = [0]
        budgets = [value for value in budgets if value <= 100]
        if not budgets or budgets[-1] != 100:
            budgets = sorted(set([*budgets, 100]))
        config.update(
            {
                "proposal_batch_size": min(int(config["proposal_batch_size"]), 120),
                "max_unique_proposals": min(int(config["max_unique_proposals"]), 240),
                "max_generation_rounds": min(int(config["max_generation_rounds"]), 2),
                "max_executions": 100,
                "refill_below": min(int(config["refill_below"]), 100),
            }
        )
    command = [
        str(python),
        str(
            code_root
            / "neurooracle/scripts/run_neurodiscovery_dynamic_closed_loop_hindcasting.py"
        ),
        "--snapshot-root",
        str(_resolve(design["temporal_inputs"]["snapshot_root"], workspace_root)),
        "--future-claims",
        str(
            _resolve(
                design["canonical_release"]["extracted_claims"]["path"],
                workspace_root,
            )
        ),
        "--eligibility-manifest",
        str(eligibility_manifest),
        "--source-bundle-manifest",
        str(bundle_manifest),
        "--output-root",
        str(output_root),
        "--case-study-ids",
        *case_ids,
        "--seeds",
        *(str(value) for value in seeds),
        "--budgets",
        *(str(value) for value in budgets),
        "--profile",
        str(config["profile"]),
        "--feedback-enabled" if feedback_enabled else "--no-feedback-enabled",
        "--proposal-batch-size",
        str(config["proposal_batch_size"]),
        "--max-unique-proposals",
        str(config["max_unique_proposals"]),
        "--max-generation-rounds",
        str(config["max_generation_rounds"]),
        "--max-stagnant-generation-rounds",
        str(config["max_stagnant_generation_rounds"]),
        "--max-executions",
        str(config["max_executions"]),
        "--execution-batch-size",
        str(config["execution_batch_size"]),
        "--refresh-every-executions",
        str(config["refresh_every_executions"]),
        "--refill-below",
        str(config["refill_below"]),
        "--feedback-years",
        str(config["feedback_years"]),
        "--feedback-start-budget",
        str(config["feedback_start_budget"]),
        "--min-supported-before-feedback",
        str(config["min_supported_before_feedback"]),
        "--seed-diversity-fraction",
        str(config["seed_diversity_fraction"]),
        "--candidate-pool-mode",
        str(config["candidate_pool_mode"]),
        "--task-scope-fraction",
        str(config["task_scope_fraction"]),
        "--evidence-frontier-fraction",
        str(config["evidence_frontier_fraction"]),
        "--protect-general-top-k",
        str(config["protect_general_top_k"]),
        "--path-variants-per-endpoint",
        str(config["path_variants_per_endpoint"]),
        "--feedback-mutation-fraction",
        str(config["feedback_mutation_fraction"]),
        "--max-paths-per-endpoint",
        str(config["max_paths_per_endpoint"]),
        "--endpoint-canonical-quality-weight",
        str(config["endpoint_canonical_quality_weight"]),
        "--kge-weight",
        str(config["kge_weight"]),
    ]
    if force:
        command.append("--force")
    return command


def _normalized_arm_command(command: Sequence[str]) -> tuple[str, ...]:
    """Remove only the two fields allowed to differ between paired arms."""

    normalized: list[str] = []
    skip_next = False
    for value in command:
        if skip_next:
            skip_next = False
            continue
        if value == "--output-root":
            skip_next = True
            continue
        if value in {"--feedback-enabled", "--no-feedback-enabled"}:
            continue
        normalized.append(value)
    return tuple(normalized)


def build_commands(
    *,
    python: Path,
    code_root: Path,
    design: Mapping[str, Any],
    workspace_root: Path,
    eligibility_manifest: Path,
    bundle_manifest: Path,
    output_root: Path,
    force: bool,
    smoke: bool = False,
) -> dict[str, list[str]]:
    closed_root = output_root / "closed"
    open_root = output_root / "open"
    comparison_root = output_root / "comparison"
    closed = _dynamic_command(
        python=python,
        code_root=code_root,
        design=design,
        workspace_root=workspace_root,
        eligibility_manifest=eligibility_manifest,
        bundle_manifest=bundle_manifest,
        output_root=closed_root,
        feedback_enabled=True,
        force=force,
        smoke=smoke,
    )
    opened = _dynamic_command(
        python=python,
        code_root=code_root,
        design=design,
        workspace_root=workspace_root,
        eligibility_manifest=eligibility_manifest,
        bundle_manifest=bundle_manifest,
        output_root=open_root,
        feedback_enabled=False,
        force=force,
        smoke=smoke,
    )
    if _normalized_arm_command(closed) != _normalized_arm_command(opened):
        raise ValueError("closed/open commands differ beyond feedback and output root")
    compare = [
        str(python),
        str(code_root / "neurooracle/scripts/compare_dynamic_closed_open_loop.py"),
        "--closed-root",
        str(closed_root),
        "--open-root",
        str(open_root),
        "--output-root",
        str(comparison_root),
        "--metrics",
        *(str(value) for value in design["endpoints"]["metrics"]),
    ]
    return {"closed": closed, "open": opened, "compare": compare}


def _run_command(
    *,
    stage: str,
    command: Sequence[str],
    output_root: Path,
    bundle_manifest: Path,
    environment: Mapping[str, str],
    code_root: Path,
    verify_references: bool,
) -> dict[str, Any]:
    before = verify_source_bundle(
        bundle_manifest,
        require_live_source=False,
        verify_references=verify_references,
    )
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
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {result.returncode}; see {stderr_path}"
        )
    after = verify_source_bundle(
        bundle_manifest,
        require_live_source=False,
        verify_references=verify_references,
    )
    return {
        "stage": stage,
        "started_at": started,
        "finished_at": _utc_now(),
        "command": list(command),
        "returncode": result.returncode,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "bundle_before": before,
        "bundle_after": after,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    bundle_manifest = args.source_bundle_manifest.resolve()
    archive_root = (bundle_manifest.parent / "files").resolve()
    code_root = Path(__file__).resolve().parents[2]
    executing_from_archive = Path(__file__).resolve().is_relative_to(archive_root)
    if not executing_from_archive and not args.allow_live_source:
        raise RuntimeError(
            "formal dynamic runner must execute from the frozen archive; use "
            "--allow-live-source only for diagnostics"
        )
    if executing_from_archive and code_root != archive_root:
        raise RuntimeError("dynamic archive code root mismatch")
    verify_bundle_member(bundle_manifest, args.design.resolve())
    design, eligibility_manifest = _verify_design(
        args.design.resolve(), workspace_root
    )
    declared_bundle = _resolve(
        design["reproducibility"]["source_bundle_manifest"], workspace_root
    )
    if declared_bundle != bundle_manifest:
        raise ValueError("dynamic design declares a different source bundle")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else _resolve(design["output_root"], workspace_root)
    )
    if args.smoke:
        output_root = output_root.parent / f"{output_root.name}_smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        eligibility_manifest = _derive_smoke_lock(
            eligibility_manifest, output_root
        )
    commands = build_commands(
        python=args.python.resolve(),
        code_root=code_root,
        design=design,
        workspace_root=workspace_root,
        eligibility_manifest=eligibility_manifest,
        bundle_manifest=bundle_manifest,
        output_root=output_root,
        force=args.force,
        smoke=args.smoke,
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
    for stage in selected:
        record = _run_command(
            stage=stage,
            command=commands[stage],
            output_root=output_root,
            bundle_manifest=bundle_manifest,
            environment=environment,
            code_root=code_root,
            verify_references=not args.skip_reference_rehash,
        )
        records.append(record)
        _atomic_json(
            output_root / "execution_manifest.json",
            {
                "schema_version": "formal-dynamic-generalization-execution.v1",
                "status": "running",
                "mode": "smoke" if args.smoke else "formal",
                "design_path": str(args.design.resolve()),
                "design_sha256": sha256_file(args.design.resolve()),
                "source_bundle_manifest": str(bundle_manifest),
                "source_bundle_manifest_sha256": sha256_file(bundle_manifest),
                "records": records,
            },
        )
    result = {
        "schema_version": "formal-dynamic-generalization-execution.v1",
        "status": "complete",
        "mode": "smoke" if args.smoke else "formal",
        "design_path": str(args.design.resolve()),
        "design_sha256": sha256_file(args.design.resolve()),
        "source_bundle_manifest": str(bundle_manifest),
        "source_bundle_manifest_sha256": sha256_file(bundle_manifest),
        "records": records,
    }
    _atomic_json(output_root / "execution_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--source-bundle-manifest", type=Path, default=DEFAULT_BUNDLE
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-live-source", action="store_true")
    parser.add_argument(
        "--skip-reference-rehash",
        action="store_true",
        help="Diagnostic only; formal runs must rehash all referenced inputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
