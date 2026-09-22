"""Build the outcome-blind Case Study 2 readiness and design lock.

The script never fits an association or mediation model. It inventories the
available cohorts, fixes a master candidate registry, and records the gates
that must pass before an unseen cohort may be opened for final evaluation.
Historical Case Study 2 results are treated as method-development evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    CASE2_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PATHWAY_ROOT,
    _complete_case_mask,
    build_index_analysis_rows,
)


CASE_STUDY_ID = "case2"
PUBLIC_NAME = "Case Study 2"
BASELINES = (
    "AI Scientist-v2",
    "Open Co-Scientist",
    "SciAgents",
    "Virtual Lab",
    "BrainPilot",
    "Biomni",
)
MASTER_MARKER_SPECS = (
    "amyloid_pet::CENTILOIDS",
    "amyloid_pet::SUMMARY_SUVR",
    "fdg_pet::FDG_META_ROI_SUVR",
    "tau_pet::META_TEMPORAL_SUVR",
    "tau_pet::CTX_ENTORHINAL_SUVR",
    "tau_pvc_pet::META_TEMPORAL_SUVR",
    "tau_pvc_pet::CTX_ENTORHINAL_SUVR",
    "smri_adnimerge::Hippocampus",
    "smri_adnimerge::Entorhinal",
    "smri_adnimerge::Fusiform",
    "smri_adnimerge::MidTemp",
    "smri_adnimerge::WholeBrain",
    "smri_adnimerge::Ventricles",
)
MASTER_OUTCOMES = (
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "ADASQ4",
    "MMSE",
    "RAVLT_immediate",
    "RAVLT_learning",
    "RAVLT_forgetting",
    "RAVLT_perc_forgetting",
    "LDELTOTAL",
    "TRABSCOR",
    "FAQ",
    "MOCA",
    "mPACCdigit",
    "mPACCtrailsB",
)
THRESHOLD_PRIORITY = {"p1em03": 0, "p5em02": 1}
REQUIRED_UNSEEN_MANIFEST_FIELDS = {
    "cohort_id",
    "independent_of_development",
    "case2_outcomes_previously_accessed",
    "genetics_available",
    "imaging_available",
    "longitudinal_outcomes_available",
    "minimum_complete_case_n",
    "executable_candidate_count",
    "distinct_pathways",
    "distinct_markers",
    "distinct_outcomes",
    "data_manifest_sha256",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_master_pathways(
    subjects: pd.DataFrame,
    score_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Select one deterministic, outcome-blind PRS threshold per pathway."""

    required = {
        "score_name",
        "pathway_id",
        "pathway_source",
        "pathway_name",
        "threshold_label",
    }
    missing = required - set(score_manifest.columns)
    if missing:
        raise ValueError(f"Score manifest is missing columns: {sorted(missing)}")
    available_rows: list[pd.Series] = []
    for _, row in score_manifest.iterrows():
        score_name = str(row["score_name"])
        if score_name not in subjects.columns:
            continue
        if pd.to_numeric(subjects[score_name], errors="coerce").notna().sum() <= 0:
            continue
        available_rows.append(row)
    if not available_rows:
        raise ValueError("No pathway PRS is available in the subject table")
    available = pd.DataFrame(available_rows).copy()
    available["threshold_priority"] = (
        available["threshold_label"]
        .map(THRESHOLD_PRIORITY)
        .fillna(99)
        .astype(int)
    )
    selected = (
        available.sort_values(
            ["pathway_id", "threshold_priority", "score_name"],
            kind="stable",
        )
        .drop_duplicates("pathway_id")
        .sort_values("pathway_id", kind="stable")
        .reset_index(drop=True)
    )
    return selected.drop(columns="threshold_priority")


def build_master_candidate_registry(
    pathways: pd.DataFrame,
    *,
    marker_specs: Iterable[str] = MASTER_MARKER_SPECS,
    outcomes: Iterable[str] = MASTER_OUTCOMES,
) -> pd.DataFrame:
    """Enumerate the fixed pathway-imaging-outcome master universe."""

    markers = tuple(marker_specs)
    outcome_names = tuple(outcomes)
    rows: list[dict[str, Any]] = []
    for pathway_row, marker_spec, outcome in product(
        pathways.to_dict(orient="records"), markers, outcome_names
    ):
        modality, marker = marker_spec.split("::", 1)
        pathway_id = str(pathway_row["pathway_id"])
        candidate_id = "|".join((pathway_id, modality, marker, str(outcome)))
        rows.append(
            {
                "candidate_id": candidate_id,
                "pathway_id": pathway_id,
                "pathway_name": str(pathway_row["pathway_name"]),
                "pathway_source": str(pathway_row["pathway_source"]),
                "exposure": str(pathway_row["score_name"]),
                "prs_threshold": str(pathway_row["threshold_label"]),
                "modality": modality,
                "imaging_marker": marker,
                "outcome": str(outcome),
                "family_id": f"{modality}::{marker}::{outcome}",
                "scientific_chain": "GENE_OR_PATHWAY->IMAGING_MARKER->OUTCOME",
            }
        )
    registry = pd.DataFrame(rows).sort_values("candidate_id", kind="stable")
    registry.insert(0, "registry_order", np.arange(1, len(registry) + 1))
    if registry["candidate_id"].duplicated().any():
        raise ValueError("Master candidate IDs are not unique")
    return registry.reset_index(drop=True)


def development_availability(
    registry: pd.DataFrame,
    *,
    dataset_root: Path,
    marker_specs: Iterable[str],
    outcomes: Iterable[str],
    minimum_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach development complete-case counts without estimating effects."""

    subjects = pd.read_parquet(
        dataset_root / "case2_adni_subject_genetics_covariates.parquet"
    )
    clinical = pd.read_parquet(dataset_root / "case2_adni_clinical_visits.parquet")
    markers = pd.read_parquet(
        dataset_root / "case2_adni_primary_imaging_markers_long.parquet"
    )
    longitudinal = pd.read_parquet(
        dataset_root / "case2_adni_longitudinal_outcome_pairs.parquet"
    )
    analysis_rows = build_index_analysis_rows(
        markers,
        longitudinal,
        subjects,
        clinical,
        marker_specs=tuple(marker_specs),
        outcomes=tuple(outcomes),
        target_followup_days=730,
        min_followup_days=365,
        max_followup_days=1095,
    )
    lookup = {
        tuple(map(str, key)): frame
        for key, frame in analysis_rows.groupby(
            ["modality", "marker", "outcome"], sort=False
        )
    }
    complete_counts: list[int] = []
    raw_counts: list[int] = []
    for row in registry.itertuples(index=False):
        family = lookup.get((row.modality, row.imaging_marker, row.outcome))
        raw_n = int(len(family)) if family is not None else 0
        if family is None:
            complete_n = 0
        else:
            complete_n = int(
                _complete_case_mask(
                    family,
                    row.exposure,
                    include_icv=row.modality == "smri_adnimerge",
                ).sum()
            )
        raw_counts.append(raw_n)
        complete_counts.append(complete_n)
    result = registry.copy()
    result["development_raw_family_n"] = raw_counts
    result["development_complete_case_n"] = complete_counts
    result["development_meets_minimum_n"] = (
        result["development_complete_case_n"] >= minimum_n
    )
    phase_counts = (
        subjects.groupby("COLPROT", dropna=False)["subject_id"]
        .nunique()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    metadata = {
        "development_subjects": int(subjects["subject_id"].nunique()),
        "development_phases": {str(key): value for key, value in phase_counts.items()},
        "subjects_with_analysis_rows": int(analysis_rows["subject_id"].nunique()),
        "candidate_count": int(len(result)),
        "candidates_meeting_minimum_n": int(
            result["development_meets_minimum_n"].sum()
        ),
        "minimum_complete_case_n": int(result["development_complete_case_n"].min()),
        "median_complete_case_n": float(
            result["development_complete_case_n"].median()
        ),
        "maximum_complete_case_n": int(result["development_complete_case_n"].max()),
        "association_or_mediation_model_fitted": False,
        "effect_estimates_computed": False,
        "outcome_values_exported": False,
        "allowed_information_used": (
            "phase membership, variable availability, missingness, and complete-case counts"
        ),
    }
    return result, metadata


def approximate_joint_path_power(
    sample_size: int,
    standardized_path: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Conservative normal approximation for detecting both mediation legs."""

    if sample_size <= 0 or standardized_path < 0:
        raise ValueError("sample_size must be positive and standardized_path non-negative")
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    noncentrality = float(standardized_path * math.sqrt(sample_size))
    single_path_power = float(
        norm.sf(critical - noncentrality) + norm.cdf(-critical - noncentrality)
    )
    return float(single_path_power**2)


def build_power_grid() -> pd.DataFrame:
    rows = []
    for sample_size, path_effect in product(
        (200, 300, 400, 500, 600, 800), (0.10, 0.15, 0.20)
    ):
        rows.append(
            {
                "complete_case_n": sample_size,
                "standardized_a_and_b_path": path_effect,
                "approximate_joint_path_power": approximate_joint_path_power(
                    sample_size, path_effect
                ),
                "interpretation": "planning approximation; not a mediation-model result",
            }
        )
    return pd.DataFrame(rows)


def _matching_files(roots: Iterable[Path], token: str) -> list[str]:
    matches: list[str] = []
    lowered = token.lower()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and lowered in path.name.lower():
                matches.append(str(path))
    return sorted(set(matches))


def cohort_readiness_inventory(
    *,
    controlled_root: Path,
    local_download_root: Path,
    unseen_manifest: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Inventory final-cohort inputs without reading biological outcomes."""

    adni4_files = _matching_files(
        (controlled_root / "raw", local_download_root), "adni4"
    )
    external_locations = {
        "AIBL": (
            controlled_root.parent / "AIBL",
            controlled_root.parents[2] / "Public Dataset" / "AIBL",
        ),
        "OASIS": (
            controlled_root.parent / "OASIS",
            controlled_root.parents[2] / "Public Dataset" / "OASIS",
        ),
    }
    rows = [
        {
            "cohort": "current ADNI1/GO/2/3 tables",
            "data_present": True,
            "independent_of_development": False,
            "genetics_present": True,
            "imaging_present": True,
            "longitudinal_outcomes_present": True,
            "eligible_as_final_unseen_cohort": False,
            "reason": "already used for Case Study 2 method development",
        },
        {
            "cohort": "ADNI4",
            "data_present": bool(adni4_files),
            "independent_of_development": True,
            "genetics_present": bool(adni4_files),
            "imaging_present": False,
            "longitudinal_outcomes_present": False,
            "eligible_as_final_unseen_cohort": False,
            "reason": (
                "ADNI4-named files found but a complete locked manifest is absent"
                if adni4_files
                else "ADNI4 genotype/pathway-PRS input is absent"
            ),
        },
    ]
    for cohort, locations in external_locations.items():
        present = any(path.exists() for path in locations)
        rows.append(
            {
                "cohort": cohort,
                "data_present": present,
                "independent_of_development": True,
                "genetics_present": False,
                "imaging_present": False,
                "longitudinal_outcomes_present": False,
                "eligible_as_final_unseen_cohort": False,
                "reason": (
                    "directory found but no validated multimodal manifest was supplied"
                    if present
                    else "dataset is not present at the registered NAS locations"
                ),
            }
        )

    manifest: dict[str, Any] | None = None
    if unseen_manifest is not None:
        manifest = json.loads(unseen_manifest.read_text(encoding="utf-8"))
        missing = REQUIRED_UNSEEN_MANIFEST_FIELDS - set(manifest)
        if missing:
            raise ValueError(
                f"Unseen cohort manifest is missing fields: {sorted(missing)}"
            )
        eligible = unseen_manifest_is_eligible(manifest)
        rows.append(
            {
                "cohort": str(manifest["cohort_id"]),
                "data_present": True,
                "independent_of_development": bool(
                    manifest["independent_of_development"]
                ),
                "genetics_present": bool(manifest["genetics_available"]),
                "imaging_present": bool(manifest["imaging_available"]),
                "longitudinal_outcomes_present": bool(
                    manifest["longitudinal_outcomes_available"]
                ),
                "eligible_as_final_unseen_cohort": eligible,
                "reason": (
                    "validated unseen-cohort manifest passes all readiness gates"
                    if eligible
                    else "supplied unseen-cohort manifest fails one or more readiness gates"
                ),
            }
        )
    return pd.DataFrame(rows), manifest


def unseen_manifest_is_eligible(manifest: Mapping[str, Any]) -> bool:
    return bool(
        manifest.get("independent_of_development") is True
        and manifest.get("case2_outcomes_previously_accessed") is False
        and manifest.get("genetics_available") is True
        and manifest.get("imaging_available") is True
        and manifest.get("longitudinal_outcomes_available") is True
        and int(manifest.get("minimum_complete_case_n", 0)) >= 400
        and int(manifest.get("executable_candidate_count", 0)) >= 1000
        and int(manifest.get("distinct_pathways", 0)) >= 15
        and int(manifest.get("distinct_markers", 0)) >= 8
        and int(manifest.get("distinct_outcomes", 0)) >= 6
        and len(str(manifest.get("data_manifest_sha256", ""))) == 64
    )


def build_protocol_lock(
    *,
    registry_sha256: str,
    registry_count: int,
    development_metadata: Mapping[str, Any],
    final_holdout_ready: bool,
    unseen_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create the immutable design lock and fail closed before final data exist."""

    payload: dict[str, Any] = {
        "schema_version": "neurooracle.case2.protocol-lock.v1",
        "case_study_id": CASE_STUDY_ID,
        "public_name": PUBLIC_NAME,
        "status": (
            "ready_for_outcome_blind_policy_freeze"
            if final_holdout_ready
            else "design_locked_pending_unseen_cohort"
        ),
        "created_at_utc": utc_now(),
        "public_version_label": None,
        "historical_runs_classification": "method_development_only",
        "scientific_chain": "GENE_OR_PATHWAY->IMAGING_MARKER->LONGITUDINAL_OUTCOME",
        "master_candidate_registry": {
            "sha256": registry_sha256,
            "candidate_count": int(registry_count),
            "threshold_rule": "prefer p1em03, otherwise p5em02, then lexical score_name",
            "candidate_inclusion_uses_effects": False,
            "final_executable_filter": (
                "availability and complete-case N only, applied before any effect estimate"
            ),
        },
        "development_data": dict(development_metadata),
        "unseen_cohort": dict(unseen_manifest) if unseen_manifest else None,
        "unseen_cohort_gates": {
            "independent_of_development": True,
            "case2_outcomes_previously_accessed": False,
            "minimum_complete_case_n": 400,
            "target_complete_case_n": 600,
            "minimum_executable_candidates": 1000,
            "minimum_distinct_pathways": 15,
            "minimum_distinct_markers": 8,
            "minimum_distinct_outcomes": 6,
        },
        "ranking_freeze": {
            "definition": (
                "all method policies, seeds, code hashes, metrics, and thresholds are "
                "committed before unseen-cohort biological effects are computed"
            ),
            "is_temporal_kg_freeze": False,
            "hindcasting_freeze_year_used": False,
        },
        "methods": {
            "neurodiscovery_runs": 10,
            "baseline_runs_each": 10,
            "baselines": list(BASELINES),
            "common_candidate_registry": True,
            "common_experiment_feedback": True,
            "invalid_duplicate_or_missing_outputs_consume_slots": True,
        },
        "evaluation": {
            "primary_metric": "heldout_graded_evidence_ndcg_at_100",
            "secondary_metrics": [
                "cumulative_evidence_gain_auc_through_200",
                "strict_support_average_precision",
                "strict_support_recall_at_k",
                "experiments_to_fixed_recall",
            ],
            "budgets": [10, 20, 50, 100, 200, 500, 1000],
            "multiplicity": "Holm across six NeuroDiscovery-vs-baseline tests",
            "clear_sota_gate": {
                "neurodiscovery_highest_primary_mean": True,
                "holm_adjusted_p_below": 0.05,
                "minimum_absolute_ndcg_gain_over_best_baseline": 0.02,
                "minimum_relative_cumulative_evidence_gain": 0.10,
                "minimum_budget_wins_out_of_seven": 5,
                "minimum_strict_positive_labels": 5,
            },
        },
        "outcome_access": {
            "permitted": bool(final_holdout_ready),
            "fail_closed": True,
            "same_holdout_retuning_permitted": False,
        },
    }
    payload["protocol_hash"] = canonical_sha256(payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_parquet(
        args.dataset_root / "case2_adni_subject_genetics_covariates.parquet"
    )
    score_manifest = pd.read_csv(args.pathway_root / "score_manifest.csv")
    pathways = select_master_pathways(subjects, score_manifest)
    registry = build_master_candidate_registry(
        pathways,
        marker_specs=args.marker_specs,
        outcomes=args.outcomes,
    )
    registry, development_metadata = development_availability(
        registry,
        dataset_root=args.dataset_root,
        marker_specs=args.marker_specs,
        outcomes=args.outcomes,
        minimum_n=args.minimum_n,
    )
    inventory, unseen_manifest = cohort_readiness_inventory(
        controlled_root=args.controlled_root,
        local_download_root=args.local_download_root,
        unseen_manifest=args.unseen_cohort_manifest,
    )
    final_holdout_ready = bool(
        unseen_manifest is not None and unseen_manifest_is_eligible(unseen_manifest)
    )
    power = build_power_grid()

    registry_path = args.output_root / "candidate_registry.csv"
    inventory_path = args.output_root / "dataset_readiness.csv"
    power_path = args.output_root / "power_planning.csv"
    registry.to_csv(registry_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    power.to_csv(power_path, index=False)

    protocol = build_protocol_lock(
        registry_sha256=sha256_file(registry_path),
        registry_count=len(registry),
        development_metadata=development_metadata,
        final_holdout_ready=final_holdout_ready,
        unseen_manifest=unseen_manifest,
    )
    protocol["artifact_sha256"] = {
        registry_path.name: sha256_file(registry_path),
        inventory_path.name: sha256_file(inventory_path),
        power_path.name: sha256_file(power_path),
        Path(__file__).name: sha256_file(Path(__file__)),
    }
    protocol_without_hash = dict(protocol)
    protocol_without_hash.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_sha256(protocol_without_hash)
    protocol_path = args.output_root / "CS2_PROTOCOL.lock.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit = {
        "schema_version": "neurooracle.case2.readiness-audit.v1",
        "created_at_utc": utc_now(),
        "case_study_id": CASE_STUDY_ID,
        "public_name": PUBLIC_NAME,
        "status": protocol["status"],
        "final_holdout_ready": final_holdout_ready,
        "outcome_access_permitted": protocol["outcome_access"]["permitted"],
        "master_candidate_count": int(len(registry)),
        "pathway_count": int(registry["pathway_id"].nunique()),
        "marker_count": int(
            registry[["modality", "imaging_marker"]].drop_duplicates().shape[0]
        ),
        "outcome_count": int(registry["outcome"].nunique()),
        "development": development_metadata,
        "blocking_reasons": (
            []
            if final_holdout_ready
            else [
                "no validated genuinely unseen multimodal cohort manifest",
                "ADNI4 pathway-genetics input is not present",
                "no independent AIBL/OASIS multimodal genetics cohort is present",
            ]
        ),
        "protocol_hash": protocol["protocol_hash"],
        "outputs": {
            "candidate_registry": str(registry_path),
            "dataset_readiness": str(inventory_path),
            "power_planning": str(power_path),
            "protocol_lock": str(protocol_path),
        },
        "output_sha256": {
            registry_path.name: sha256_file(registry_path),
            inventory_path.name: sha256_file(inventory_path),
            power_path.name: sha256_file(power_path),
            protocol_path.name: sha256_file(protocol_path),
        },
    }
    audit_path = args.output_root / "READINESS_AUDIT.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(inventory.to_string(index=False))
    return audit


def parse_args() -> argparse.Namespace:
    controlled_root = Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-root", type=Path, default=controlled_root)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument("--local-download-root", type=Path, default=Path(r"D:\ADNI"))
    parser.add_argument("--unseen-cohort-manifest", type=Path)
    parser.add_argument("--marker-specs", nargs="+", default=list(MASTER_MARKER_SPECS))
    parser.add_argument("--outcomes", nargs="+", default=list(MASTER_OUTCOMES))
    parser.add_argument("--minimum-n", type=int, default=400)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 17:53 HKT
