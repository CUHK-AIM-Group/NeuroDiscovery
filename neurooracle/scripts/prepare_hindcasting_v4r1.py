"""Prepare and lock the leakage-free Hindcasting v4r1 execution release.

Preparation deliberately does not receive, inspect, hash, or record the later
publication corpus.  That corpus is supplied only to the separate evaluation
command after every discovery cell has a valid seal.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurooracle.src.hindcasting_v4_feedback_boundary import (
    DISCOVERY_INPUT_SCHEMA,
    canonical_sha256,
    scan_discovery_source,
    sha256_file,
    validate_discovery_input_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
V4_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v4_computational_feedback_20260916"
)
DEFAULT_RELEASE = V4_ROOT / "execution/v4r1_two_tasks_teas5_20260916"
DEFAULT_OUTPUT = ROOT / "neurooracle/data/hv4_runs/r1_computational_feedback_20260916/discovery"
SNAPSHOT_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v3_expandable_20260826/eligibility"
    / "kg_20260825_2c02732582da_705b0799/snapshots"
)
KGE_ASSET_ROOT = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v3_expandable_20260826/execution_assets"
    / "kg_20260825_2c02732582da_705b0799/kge"
)
CASE1_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_autoresearch_comparison"
    r"\20260824_kg89e40d_neurodiscovery_true_closed_loop_3seeds_v1\inputs"
)
BIOMARKER_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v8_biomarker_independent"
    r"\biomarker_discovery\tables"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_record(
    path: Path,
    *,
    digest: str | None = None,
    deep_verify_at_start: bool = False,
    columns_read: Sequence[str] = (),
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path) if digest is None else str(digest).casefold()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
        "deep_verify_at_start": bool(deep_verify_at_start),
        **({"columns_read": list(columns_read)} if columns_read else {}),
    }


def _snapshot_records(year: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    asset_path = KGE_ASSET_ROOT / f"kg_{year}_complex_dim64_ep10_asset.json"
    asset = read_json(asset_path)
    if int(asset.get("freeze_year", -1)) != year or asset.get("status") != "locked":
        raise ValueError(f"invalid locked temporal asset: {asset_path}")
    snapshot = asset["snapshot"]
    graph = file_record(
        Path(snapshot["knowledge_graph"]["path"]),
        digest=snapshot["knowledge_graph"]["sha256"],
    )
    claims = file_record(
        Path(snapshot["extracted_claims"]["path"]),
        digest=snapshot["extracted_claims"]["sha256"],
    )
    manifest = file_record(
        Path(snapshot["manifest"]["path"]),
        digest=snapshot["manifest"]["sha256"],
        deep_verify_at_start=True,
    )
    for record, declared in (
        (graph, snapshot["knowledge_graph"]),
        (claims, snapshot["extracted_claims"]),
        (manifest, snapshot["manifest"]),
    ):
        if int(record["bytes"]) != int(declared["bytes"]):
            raise ValueError(f"locked snapshot size mismatch: {record['path']}")
    return graph, claims, manifest


def _write_locked(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    atomic_json(path, payload)
    digest = sha256_file(path)
    lock = {
        "schema_version": "hindcasting-v4r1-file-lock.v1",
        "status": "locked",
        "locked_at": utc_now(),
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "immutable": True,
    }
    atomic_json(path.with_name(path.name + ".lock.json"), lock)
    return file_record(path, digest=digest, deep_verify_at_start=True)


def _source_bundle(release: Path) -> dict[str, Any]:
    explicit = [
        ROOT / "neurooracle/scripts/run_hindcasting_v4_discovery.py",
        ROOT / "neurooracle/src/hindcasting_v4_computational_executor.py",
        ROOT / "neurooracle/src/hindcasting_v4_feedback_boundary.py",
        ROOT / "neurooracle/scripts/generate_case_study_frozen_baselines.py",
        ROOT / "neurooracle/scripts/run_case_study_hindcasting.py",
    ]
    for path in explicit[:4]:
        scan_discovery_source(path)
    dependency_files = sorted((ROOT / "neurooracle/src").glob("*.py"))
    files = sorted(set(explicit + dependency_files))
    manifest = {
        "schema_version": "hindcasting-v4r1-discovery-source-bundle.v1",
        "status": "hash_locked_before_discovery",
        "created_at": utc_now(),
        "files": [file_record(path, deep_verify_at_start=True) for path in files],
        "source_scan": [scan_discovery_source(path) for path in explicit[:4]],
        "retrospective_evaluator_in_bundle": False,
    }
    path = release / "source_bundle_manifest.json"
    atomic_json(path, manifest)
    return file_record(path, deep_verify_at_start=True)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    release = args.release.resolve()
    plan_path = release / "execution_plan.json"
    lock_path = release / "execution_plan.lock.json"
    if plan_path.exists() or lock_path.exists():
        raise FileExistsError(
            f"v4r1 release already exists and is immutable: {release}"
        )
    release.mkdir(parents=True, exist_ok=False)
    created_at = utc_now()

    task_schema = _write_locked(
        release / "task_cohort.json",
        {
            "schema_version": "hindcasting-v4r1-task-cohort.v1",
            "status": "locked_without_later_literature",
            "created_at": created_at,
            "tasks": ["case1_transdiagnostic", "biomarker_discovery"],
            "admission_basis": [
                "user-selected comparison scope",
                "registered frozen-KG task schema",
                "registered TCP computational executor",
                "at least 1000 deterministic frozen-KG proposal slots",
            ],
            "later_literature_used_for_admission": False,
            "expandability": (
                "Additional tasks require a separately locked successor cohort; "
                "this release is immutable."
            ),
        },
    )
    seed_schedule = _write_locked(
        release / "seed_schedule.json",
        {
            "schema_version": "hindcasting-v4r1-seed-schedule.v1",
            "seeds": list(range(10)),
            "paired_across_methods": True,
            "rng_rule": "20260916 + 1009 * seed + freeze_year",
        },
    )
    method_config = _write_locked(
        release / "method_config.json",
        {
            "schema_version": "hindcasting-v4r1-method-config.v1",
            "methods": ["neurodiscovery", "sciagents", "openscholar_rag"],
            "display_names": {
                "neurodiscovery": "NeuroDiscovery",
                "sciagents": "SciAgents",
                "openscholar_rag": "OpenScholar-RAG",
            },
            "maximum_rank": 1000,
            "budgets": [10, 20, 50, 100, 200, 500, 1000],
            "batch_size": 10,
            "warmup_budget": 50,
            "feedback_weight": 0.10,
            "pair_feedback_weight": 0.08,
            "exploration_weight": 0.015,
            "neurodiscovery_endpoint_quality_weight": 0.20,
            "neurodiscovery_evidence_frontier_fraction": 0.35,
            "baseline_feedback_enabled": False,
            "llm_api_enabled": False,
            "current_kg_scores_in_tcp_public_tables_used": False,
        },
    )
    analysis_plan = _write_locked(
        release / "computational_analysis_plan.json",
        {
            "schema_version": "hindcasting-v4r1-computational-analysis-plan.v1",
            "status": "locked_before_discovery",
            "case1_transdiagnostic": {
                "support": "execution succeeds, p < 0.01, and absolute adjusted Cohen d >= 0.15",
                "contradiction": "the same threshold is met in the opposite direction when direction is explicit in the frozen hypothesis",
                "nonsignificance": "inconclusive",
            },
            "biomarker_discovery": {
                "support": "registered biomarker TCP feedback_status from its frozen analysis plan",
                "validation_rule": (
                    "registered-family q <= 0.05, absolute adjusted Cohen d >= 0.15, "
                    "held-out directional AUC >= 0.55, and direction concordance >= 0.70"
                ),
                "nonsignificance": "inconclusive",
            },
            "mapping_gate": (
                "Disease cohort, imaging measurement, and anatomical/network tag must all "
                "match public TCP metadata; otherwise retain execution_failed."
            ),
            "selection_commitment_precedes_outcome_reveal": True,
            "publication_support_is_not_computational_feedback": True,
        },
    )
    metric_contract = _write_locked(
        release / "metric_contract.json",
        {
            "schema_version": "hindcasting-v4r1-teas5-metric.v1",
            "status": "locked_before_discovery",
            "primary_metric": "TEAS-5",
            "definition": "For each task x cutoff x seed cell and rank K, TEAS-5 = 20 * min(D_K, 5), where D_K is the number of unique exact primary future-supported discoveries among the first K sealed hypotheses.",
            "aggregation": "Arithmetic mean over all 2 tasks x 5 cutoffs x 10 paired seeds.",
            "range": [0, 100],
            "primary_display_ranks": [100, 1000],
            "retained_audit_metrics": [
                "unique exact primary discoveries",
                "raw primary-hit slots",
                "future-pair recall",
                "lead time to first supporting publication",
            ],
            "missing_or_zero_support_rule": "Retain the locked cell with D_K=0; never remove it after inspection.",
            "metric_changes_after_discovery": "forbidden for confirmatory results; successors are exploratory until rerun",
        },
    )
    protocol = _write_locked(
        release / "formal_hindcasting_design_v4r1.json",
        {
            "schema_version": "neurodiscovery-formal-hindcasting-design.v4r1",
            "protocol_id": "formal_hindcasting_v4r1_two_tasks_teas5_20260916",
            "status": "locked_before_discovery",
            "created_at": created_at,
            "supersedes_boundary_protocol": str(
                V4_ROOT / "protocol/formal_hindcasting_design_v4.json"
            ),
            "scientific_question": (
                "Can NeuroDiscovery use only results from its own registered computational "
                "experiments to improve later-paper recovery relative to two frozen baselines?"
            ),
            "methods": ["neurodiscovery", "sciagents", "openscholar_rag"],
            "neurodiscovery_definition": "full computational-feedback closed loop",
            "tasks": ["case1_transdiagnostic", "biomarker_discovery"],
            "freeze_years": [2016, 2017, 2018, 2019, 2020],
            "evaluation_horizon_years": 5,
            "seeds": list(range(10)),
            "maximum_rank": 1000,
            "metric_contract": metric_contract,
            "task_cohort": task_schema,
            "method_config": method_config,
            "computational_analysis_plan": analysis_plan,
            "discovery_knowledge": "freeze-year KG and claims only",
            "neurodiscovery_feedback": "registered TCP computational results only",
            "later_publications_available_during_discovery": False,
            "later_publications_used_for_task_admission": False,
            "llm_api_calls_permitted": False,
            "discovery_and_evaluation_are_separate_processes": True,
            "interpretation": "algorithmically blinded retrospective hindcasting; computational datasets postdate the historical cutoffs",
            "v3_orders_or_scores_reused": False,
        },
    )
    source_bundle = _source_bundle(release)

    case1_public_columns = (
        "candidate_id",
        "disease",
        "modality",
        "source",
        "roi_index",
        "roi_name",
        "anatomy_full",
        "map_group",
        "network",
        "structure_class",
        "feature",
        "feature_family",
    )
    biomarker_public_columns = (
        "candidate_id",
        "disease",
        "modality",
        "atlas",
        "source",
        "roi_index",
        "roi_name",
        "anatomy",
        "anatomy_full",
        "network",
        "structure_class",
        "feature",
        "feature_family",
    )
    resources = {
        "case1_transdiagnostic": {
            "dataset_id": "TCP_CASE1_20260616_FULL_MAIN_NOBOOT",
            "executor": "case1_adjusted_group_contrast.v1",
            "release_date": "2026-06-16",
            "strict_historical_resource": False,
            "public_candidates": file_record(
                CASE1_ROOT / "public_candidates.csv.gz",
                columns_read=case1_public_columns,
            ),
            "outcomes": file_record(
                CASE1_ROOT / "feedback_outcomes.csv.gz",
                columns_read=(
                    "candidate_id",
                    "execution_succeeded",
                    "adjusted_residual_d",
                    "p_value",
                    "expected_direction",
                ),
            ),
        },
        "biomarker_discovery": {
            "dataset_id": "TCP_BIOMARKER_INDEPENDENT_V2_20260816",
            "executor": "biomarker_discovery_independent_tcp.v2",
            "release_date": "2026-08-16",
            "strict_historical_resource": False,
            "public_candidates": file_record(
                BIOMARKER_ROOT / "public_candidates.csv",
                columns_read=biomarker_public_columns,
            ),
            "outcomes": file_record(
                BIOMARKER_ROOT / "internal_outcomes.csv",
                columns_read=(
                    "candidate_id",
                    "validated",
                    "execution_succeeded",
                    "feedback_status",
                    "feedback_available",
                    "feedback_utility",
                    "adjusted_residual_d",
                    "p_value",
                    "q_fdr_registered_family",
                    "cv_auc_mean",
                    "cv_direction_stability",
                    "direction",
                ),
            ),
        },
    }

    windows: list[dict[str, Any]] = []
    for year in range(2016, 2021):
        graph, claims, snapshot_manifest = _snapshot_records(year)
        manifest = {
            "schema_version": DISCOVERY_INPUT_SCHEMA,
            "status": "locked_before_discovery",
            "freeze_year": year,
            "generator_mode": "deterministic_frozen",
            "llm_api_enabled": False,
            "evaluation_data_mounted": False,
            "evaluation_module_imported": False,
            "inputs": [
                {"role": "historical_kg", "max_publication_year": year, **graph},
                {"role": "historical_claims", "max_publication_year": year, **claims},
                {"role": "task_schema", **task_schema},
                {
                    "role": "deterministic_generator",
                    **file_record(
                        ROOT / "neurooracle/scripts/generate_case_study_frozen_baselines.py"
                    ),
                },
                {
                    "role": "computational_dataset",
                    "task_id": "case1_transdiagnostic",
                    "selected_without_retrospective_evaluation": True,
                    **resources["case1_transdiagnostic"]["public_candidates"],
                },
                {
                    "role": "computational_dataset",
                    "task_id": "biomarker_discovery",
                    "selected_without_retrospective_evaluation": True,
                    **resources["biomarker_discovery"]["public_candidates"],
                },
                {
                    "role": "computational_pipeline",
                    "selected_without_retrospective_evaluation": True,
                    **file_record(
                        ROOT / "neurooracle/src/hindcasting_v4_computational_executor.py"
                    ),
                },
                {
                    "role": "computational_analysis_plan",
                    "selected_without_retrospective_evaluation": True,
                    **analysis_plan,
                },
                {"role": "method_config", **method_config},
                {"role": "seed_schedule", **seed_schedule},
            ],
            "snapshot_manifest": snapshot_manifest,
            "computational_outcome_vaults_are_access_controlled": True,
            "current_kg_candidate_score_columns_read": False,
        }
        validate_discovery_input_manifest(manifest)
        manifest_path = release / "discovery_inputs" / f"kg_{year}.json"
        atomic_json(manifest_path, manifest)
        windows.append(
            {
                "freeze_year": year,
                "future_start_year": year + 1,
                "future_end_year": year + 5,
                "snapshot_root": str((SNAPSHOT_ROOT / f"kg_{year}").resolve()),
                "discovery_input_manifest": str(manifest_path.resolve()),
                "discovery_input_manifest_sha256": sha256_file(manifest_path),
            }
        )

    plan = {
        "schema_version": "neurodiscovery-hindcasting-v4r1-execution-plan.v1",
        "status": "locked_before_discovery",
        "created_at": created_at,
        "experiment_id": "hindcasting_v4r1_two_tasks_teas5_20260916",
        "protocol": protocol,
        "task_cohort": task_schema,
        "metric_contract": metric_contract,
        "analysis_plan": analysis_plan,
        "pipeline": file_record(
            ROOT / "neurooracle/src/hindcasting_v4_computational_executor.py"
        ),
        "discovery_source_bundle": source_bundle,
        "computational_resources": resources,
        "matrix": {
            "methods": ["neurodiscovery", "sciagents", "openscholar_rag"],
            "tasks": ["case1_transdiagnostic", "biomarker_discovery"],
            "seeds": list(range(10)),
            "budgets": [10, 20, 50, 100, 200, 500, 1000],
            "maximum_rank": 1000,
            "windows": windows,
            "total_cells": 300,
        },
        "configuration": read_json(Path(method_config["path"])),
        "output_root": str(args.output.resolve()),
        "future_publication_corpus_recorded_in_discovery_plan": False,
        "llm_api_calls": 0,
        "execution_command": [
            str(Path(__import__("sys").executable).resolve()),
            "-m",
            "neurooracle.scripts.run_hindcasting_v4_discovery",
            "--plan",
            str(plan_path.resolve()),
        ],
    }
    atomic_json(plan_path, plan)
    atomic_json(
        lock_path,
        {
            "schema_version": "neurodiscovery-hindcasting-v4r1-execution-plan-lock.v1",
            "status": "locked",
            "locked_at": utc_now(),
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
            "plan_bytes": plan_path.stat().st_size,
            "immutable": True,
            "later_publication_corpus_opened": False,
        },
    )
    report = {
        "status": "prepared_and_locked",
        "release": str(release),
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "source_bundle_sha256": source_bundle["sha256"],
        "protocol_sha256": protocol["sha256"],
        "metric_sha256": metric_contract["sha256"],
        "cells": 300,
        "later_publication_corpus_opened": False,
        "configuration_sha256": canonical_sha256(plan["configuration"]),
    }
    atomic_json(release / "PREEXECUTION_AUDIT.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(prepare(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
