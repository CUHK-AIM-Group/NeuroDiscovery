"""Freeze the outcome-blind readiness registry for formal supplementary CS2.

The command reads only outcome identifiers and dates from the longitudinal
outcome table.  It cannot fit models and it rejects observed outcome-value
columns before constructing the readiness artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from neurooracle.scripts.case2_formal_supplementary import (
    CATEGORICAL_COMMON_COVARIATES,
    assert_outcome_blind_frame,
    build_master_registry,
    build_outcome_blind_analysis_keys,
    build_readiness_audit,
    canonical_sha256,
    file_pin,
    file_sha256,
    genetic_pc_eligibility,
    load_protocol,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASE2_ROOT = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc")
    / "case2_adni_genetics_v1"
)
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "neurooracle"
    / "configs"
    / "case2_adni_formal_supplementary_v2.json"
)
DEFAULT_DATASET_ROOT = CASE2_ROOT / "experiment_tables" / "case2_adni_multimodal_v1"
DEFAULT_PATHWAY_ROOT = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
    / "features"
    / "ad_pathway_prs_v1"
)
DEFAULT_OUTPUT_ROOT = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_formal_supplementary_v2"
    / "readiness_freeze_code_pinned_r2"
)
HKT = timezone(timedelta(hours=8))
SOURCE_PATHS = {
    "shared_analysis_source": Path(__file__).with_name("case2_formal_supplementary.py"),
    "readiness_source": Path(__file__).resolve(),
    "formal_runner_source": Path(__file__).with_name(
        "run_case2_adni_formal_supplementary.py"
    ),
    "formal_test_source": (
        REPOSITORY_ROOT / "neurooracle" / "tests" / "test_case2_formal_supplementary.py"
    ),
}


def _artifact_pin(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _validate_pathway_inputs(
    protocol: dict[str, Any],
    subjects: pd.DataFrame,
    score_manifest: pd.DataFrame,
) -> None:
    required_manifest = {
        "score_name",
        "pathway_id",
        "pathway_name",
        "threshold_label",
        "pathway_source",
    }
    missing = required_manifest - set(score_manifest)
    if missing:
        raise ValueError(f"Score manifest is missing columns: {sorted(missing)}")
    manifest = score_manifest[
        score_manifest["pathway_source"].astype(str).eq("NeuroOracle curated")
    ].copy()
    manifest = manifest.set_index("score_name", drop=False)
    for pathway in protocol["pathway_exposures"]:
        score_name = pathway["score_name"]
        if score_name not in subjects:
            raise ValueError(f"Frozen subject table lacks score: {score_name}")
        if score_name not in manifest.index:
            raise ValueError(f"Score manifest lacks frozen score: {score_name}")
        row = manifest.loc[score_name]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Score manifest duplicates frozen score: {score_name}")
        for column in ("pathway_id", "pathway_name", "threshold_label"):
            if str(row[column]) != str(pathway[column]):
                raise ValueError(f"Score manifest disagrees for {score_name}: {column}")


def prepare_readiness(args: argparse.Namespace) -> dict[str, Any]:
    """Create an immutable, result-free readiness freeze."""

    protocol = load_protocol(args.protocol)
    output_root = args.output_root
    if output_root.exists():
        raise FileExistsError(
            f"Readiness output already exists and is immutable: {output_root}"
        )

    dataset_paths = {
        "dataset_manifest": args.dataset_root / "manifest.json",
        "subjects": args.dataset_root
        / "case2_adni_subject_genetics_covariates.parquet",
        "clinical_dates": args.dataset_root / "case2_adni_clinical_visits.parquet",
        "outcome_dates": args.dataset_root / "case2_adni_outcomes_long.parquet",
        "imaging_markers": args.dataset_root
        / "case2_adni_primary_imaging_markers_long.parquet",
        "score_manifest": args.pathway_root / "score_manifest.csv",
        "pathway_manifest": args.pathway_root / "manifest.json",
    }
    for path in [args.protocol, *dataset_paths.values(), *SOURCE_PATHS.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset_manifest = json.loads(
        dataset_paths["dataset_manifest"].read_text(encoding="utf-8")
    )
    pathway_manifest = json.loads(
        dataset_paths["pathway_manifest"].read_text(encoding="utf-8")
    )
    expected_subjects = int(protocol["cohort"]["primary_genetic_subject_count"])
    if int(dataset_manifest.get("genetic_subjects", -1)) != expected_subjects:
        raise ValueError("Dataset manifest is not the frozen 691-subject cohort")
    if int(pathway_manifest.get("samples", -1)) != expected_subjects:
        raise ValueError("Pathway manifest is not the frozen 691-subject cohort")
    if pathway_manifest.get("selection_is_outcome_blind") is not True:
        raise ValueError(
            "Pathway PRS manifest does not certify outcome-blind selection"
        )

    exposure_names = [row["score_name"] for row in protocol["pathway_exposures"]]
    subject_columns = [
        "subject_id",
        "AGE",
        "sex_binary",
        "PTEDUCAT",
        *[f"PC{index}" for index in range(1, 11)],
        *CATEGORICAL_COMMON_COVARIATES,
        *exposure_names,
    ]
    subjects = pd.read_parquet(dataset_paths["subjects"], columns=subject_columns)
    if (
        len(subjects) != expected_subjects
        or subjects["subject_id"].nunique() != expected_subjects
    ):
        raise ValueError(
            "Observed subject table does not contain exactly 691 unique subjects"
        )
    clinical_dates = pd.read_parquet(
        dataset_paths["clinical_dates"], columns=["subject_id", "clinical_date"]
    )
    outcome_dates = pd.read_parquet(
        dataset_paths["outcome_dates"],
        columns=["subject_id", "outcome", "outcome_date"],
    )
    markers = pd.read_parquet(
        dataset_paths["imaging_markers"],
        columns=[
            "visit_id",
            "subject_id",
            "modality",
            "imaging_date",
            "qc_primary_eligible",
            "marker",
        ],
    )
    score_manifest = pd.read_csv(dataset_paths["score_manifest"])
    _validate_pathway_inputs(protocol, subjects, score_manifest)
    assert_outcome_blind_frame(outcome_dates, label="projected outcome dates")
    pc_eligible = genetic_pc_eligibility(
        subjects,
        absolute_z_threshold=float(
            protocol["readiness_gate"]["combined_pca_absolute_z_threshold"]
        ),
    )

    registry = build_master_registry(protocol)
    row_keys = build_outcome_blind_analysis_keys(markers, outcome_dates, protocol)
    family_audit, candidate_audit, executable_registry, executable_keys = (
        build_readiness_audit(
            row_keys,
            markers,
            subjects,
            clinical_dates,
            registry,
            protocol,
        )
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        protocol_snapshot = staging / "PROTOCOL.json"
        shutil.copyfile(args.protocol, protocol_snapshot)
        artifact_frames = {
            "master_candidate_registry": (
                staging / "MASTER_CANDIDATE_REGISTRY.csv",
                registry,
            ),
            "readiness_family_audit": (
                staging / "READINESS_FAMILY_AUDIT.csv",
                family_audit,
            ),
            "candidate_readiness_audit": (
                staging / "CANDIDATE_READINESS_AUDIT.csv",
                candidate_audit,
            ),
            "executable_candidate_registry": (
                staging / "EXECUTABLE_CANDIDATE_REGISTRY.csv",
                executable_registry,
            ),
        }
        for _, (path, frame) in artifact_frames.items():
            assert_outcome_blind_frame(frame, label=path.name)
            frame.to_csv(path, index=False)
        row_keys_path = staging / "EXECUTABLE_ANALYSIS_ROW_KEYS.parquet"
        assert_outcome_blind_frame(executable_keys, label=row_keys_path.name)
        executable_keys.to_parquet(row_keys_path, index=False)

        executable_families = int(family_audit["executable"].sum())
        executable_candidates = int(len(executable_registry))
        generated_at = datetime.now(HKT).isoformat()
        summary = {
            "schema_version": "neurooracle.case2_readiness_summary.v1",
            "protocol_id": protocol["protocol_id"],
            "generated_at_hkt": generated_at,
            "status": "locked_ready_for_formal_execution",
            "cohort": {
                "expected_source_subjects": expected_subjects,
                "observed_unique_source_subjects": int(
                    subjects["subject_id"].nunique()
                ),
                "combined_pca_eligible_subjects": int(pc_eligible.sum()),
                "combined_pca_outliers_excluded": int((~pc_eligible).sum()),
                "combined_pca_absolute_z_threshold": float(
                    protocol["readiness_gate"]["combined_pca_absolute_z_threshold"]
                ),
            },
            "registry": {
                "master_candidates": int(len(registry)),
                "marker_outcome_families": int(len(family_audit)),
                "executable_families": executable_families,
                "excluded_families": int(len(family_audit) - executable_families),
                "executable_candidates": executable_candidates,
                "retained_pathways_per_family": 7,
            },
            "complete_case_n": {
                "minimum_executable_candidate": (
                    int(executable_registry["complete_case_n"].min())
                    if executable_candidates
                    else None
                ),
                "median_executable_candidate": (
                    float(executable_registry["complete_case_n"].median())
                    if executable_candidates
                    else None
                ),
                "maximum_executable_candidate": (
                    int(executable_registry["complete_case_n"].max())
                    if executable_candidates
                    else None
                ),
            },
            "guardrails": {
                "outcome_columns_read": ["subject_id", "outcome", "outcome_date"],
                "outcome_value_column_read": False,
                "association_model_fit": False,
                "effect_or_p_value_exported": False,
                "historical_registry_modified": False,
                "external_validation_queue_created": False,
            },
        }
        summary_path = staging / "READINESS_SUMMARY.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        artifact_pins = {
            "protocol_snapshot": _artifact_pin(protocol_snapshot),
            **{
                label: _artifact_pin(path)
                for label, (path, _) in artifact_frames.items()
            },
            "executable_analysis_row_keys": _artifact_pin(row_keys_path),
            "readiness_summary": _artifact_pin(summary_path),
        }
        input_pins = {
            "protocol_source": file_pin(args.protocol),
            **{label: file_pin(path) for label, path in dataset_paths.items()},
            **{label: file_pin(path) for label, path in SOURCE_PATHS.items()},
        }
        lock_material = {
            "protocol_sha256": artifact_pins["protocol_snapshot"]["sha256"],
            "input_pins": input_pins,
            "artifact_pins": artifact_pins,
            "master_candidate_count": int(len(registry)),
            "executable_candidate_count": executable_candidates,
            "outcome_value_column_read": False,
            "association_model_fit": False,
        }
        freeze_id = canonical_sha256(lock_material)
        lock = {
            "schema_version": "neurooracle.case2_formal_supplementary_readiness_lock.v1",
            "protocol_id": protocol["protocol_id"],
            "freeze_id": freeze_id,
            "generated_at_hkt": generated_at,
            "output_root": str(output_root),
            "status": "locked_ready_for_formal_execution",
            "lock_material": lock_material,
        }
        lock_path = staging / "READINESS.lock.json"
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "output_root": str(output_root),
        "freeze_id": freeze_id,
        "master_candidates": int(len(registry)),
        "executable_families": executable_families,
        "executable_candidates": executable_candidates,
        "outcome_value_column_read": False,
        "association_model_fit": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    summary = prepare_readiness(parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
