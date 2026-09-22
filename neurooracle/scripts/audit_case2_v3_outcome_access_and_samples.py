"""Audit Case Study 2 outcome access and phase-specific sample feasibility.

This audit never fits an association or mediation model. It inspects historical
result schemas for evidence of prior endpoint access and computes only
phase-specific availability and complete-case counts for a possible new protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    CASE2_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MARKER_SPECS,
    DEFAULT_PATHWAY_ROOT,
    _complete_case_mask,
    build_index_analysis_rows,
    select_curated_pathway_exposures,
)


TARGET_OUTCOMES = ("mPACCdigit", "FAQ", "LDELTOTAL")
TARGET_PHASES = ("ADNI1", "ADNIGO", "ADNI2")
REFERENCE_PHASE = "ADNI3"
MEDIATION_RESULT_COLUMNS = {
    "a_path_p",
    "b_path_p",
    "sobel_p",
    "sobel_q_family",
    "sobel_q_global",
    "indirect_effect_std",
    "family_fdr_chain_hit",
    "global_fdr_chain_hit",
    "nominal_chain_hit",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def audit_historical_access(
    experiments_root: Path,
    access_record_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Find target-outcome mediation statistics created before formal access."""

    access_record = json.loads(access_record_path.read_text(encoding="utf-8"))
    formal_access_started = _utc_from_iso(str(access_record["started_at_utc"]))
    records: list[dict[str, Any]] = []
    for path in sorted(experiments_root.rglob("*.parquet")):
        columns = _parquet_columns(path)
        if "outcome" not in columns:
            continue
        frame = pd.read_parquet(
            path,
            columns=[name for name in ("outcome", "COLPROT") if name in columns],
        )
        outcomes = sorted(
            set(frame["outcome"].dropna().astype(str)).intersection(TARGET_OUTCOMES)
        )
        if not outcomes:
            continue
        modified_utc = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        result_columns = sorted(columns.intersection(MEDIATION_RESULT_COLUMNS))
        phases = (
            sorted(frame["COLPROT"].dropna().astype(str).unique().tolist())
            if "COLPROT" in frame.columns
            else []
        )
        records.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(experiments_root)),
                "modified_at_utc": modified_utc.isoformat(timespec="seconds"),
                "before_formal_access_record": modified_utc < formal_access_started,
                "target_outcomes": ";".join(outcomes),
                "phases_if_recorded": ";".join(phases),
                "contains_mediation_statistics": bool(result_columns),
                "mediation_statistic_columns": ";".join(result_columns),
            }
        )
    table = pd.DataFrame(records)
    if table.empty:
        pre_access_statistics = 0
        post_access_statistics = 0
    else:
        pre_access_statistics = int(
            (
                table["before_formal_access_record"]
                & table["contains_mediation_statistics"]
            ).sum()
        )
        post_access_statistics = int(
            (
                ~table["before_formal_access_record"]
                & table["contains_mediation_statistics"]
            ).sum()
        )
    summary = {
        "formal_access_started_at_utc": formal_access_started.isoformat(
            timespec="seconds"
        ),
        "target_outcome_artifacts": int(len(table)),
        "pre_access_target_outcome_mediation_artifacts": pre_access_statistics,
        "post_access_target_outcome_mediation_artifacts": post_access_statistics,
        "pre_access_leak_detected": bool(pre_access_statistics),
    }
    return table, summary


def _phase_sets(subjects: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    observed = set(subjects["COLPROT"].dropna().astype(str))
    requested = {
        phase: (phase,) for phase in (*TARGET_PHASES, REFERENCE_PHASE) if phase in observed
    }
    available_target = tuple(phase for phase in TARGET_PHASES if phase in observed)
    if available_target:
        requested["ADNI1_GO_2_combined"] = available_target
    return requested


def audit_phase_samples(
    dataset_root: Path,
    pathway_root: Path,
    *,
    marker_specs: Iterable[str],
    outcomes: Iterable[str],
    minimum_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Count candidate availability without estimating any biological effect."""

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
    score_manifest = pd.read_csv(pathway_root / "score_manifest.csv")
    exposures = select_curated_pathway_exposures(subjects, score_manifest)

    detail_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for phase_group, phases in _phase_sets(subjects).items():
        phase_subjects = subjects.loc[
            subjects["COLPROT"].astype(str).isin(phases)
        ].copy()
        subject_ids = set(phase_subjects["subject_id"].astype(str))
        phase_clinical = clinical.loc[
            clinical["subject_id"].astype(str).isin(subject_ids)
        ].copy()
        phase_markers = markers.loc[
            markers["subject_id"].astype(str).isin(subject_ids)
        ].copy()
        phase_longitudinal = longitudinal.loc[
            longitudinal["subject_id"].astype(str).isin(subject_ids)
        ].copy()
        try:
            analysis_rows = build_index_analysis_rows(
                phase_markers,
                phase_longitudinal,
                phase_subjects,
                phase_clinical,
                marker_specs=marker_specs,
                outcomes=outcomes,
                target_followup_days=730,
                min_followup_days=365,
                max_followup_days=1095,
            )
        except ValueError:
            analysis_rows = pd.DataFrame()

        if analysis_rows.empty:
            phase_rows.append(
                {
                    "phase_group": phase_group,
                    "included_COLPROT": ";".join(phases),
                    "genotyped_subjects": int(len(phase_subjects)),
                    "subjects_with_analysis_rows": 0,
                    "candidate_cells": int(len(exposures) * len(tuple(marker_specs)) * len(tuple(outcomes))),
                    "estimable_at_minimum_n": 0,
                    "minimum_complete_case_n": 0,
                    "median_complete_case_n": 0.0,
                    "maximum_complete_case_n": 0,
                }
            )
            continue

        lookup = {
            tuple(map(str, key)): frame
            for key, frame in analysis_rows.groupby(
                ["modality", "marker", "outcome"], sort=False
            )
        }
        counts: list[int] = []
        for exposure_row in exposures.itertuples(index=False):
            exposure = str(exposure_row.score_name)
            for marker_spec in marker_specs:
                modality, marker = str(marker_spec).split("::", 1)
                for outcome in outcomes:
                    family = lookup.get((modality, marker, str(outcome)))
                    raw_n = int(len(family)) if family is not None else 0
                    if family is None:
                        complete_n = 0
                    else:
                        complete_n = int(
                            _complete_case_mask(
                                family,
                                exposure,
                                include_icv=modality == "smri_adnimerge",
                            ).sum()
                        )
                    counts.append(complete_n)
                    detail_rows.append(
                        {
                            "phase_group": phase_group,
                            "included_COLPROT": ";".join(phases),
                            "exposure": exposure,
                            "pathway_id": str(exposure_row.pathway_id),
                            "modality": modality,
                            "marker": marker,
                            "outcome": str(outcome),
                            "raw_family_n": raw_n,
                            "complete_case_n": complete_n,
                            "meets_minimum_n": complete_n >= minimum_n,
                        }
                    )
        count_array = np.asarray(counts, dtype=int)
        phase_rows.append(
            {
                "phase_group": phase_group,
                "included_COLPROT": ";".join(phases),
                "genotyped_subjects": int(len(phase_subjects)),
                "subjects_with_analysis_rows": int(
                    analysis_rows["subject_id"].nunique()
                ),
                "candidate_cells": int(len(count_array)),
                "estimable_at_minimum_n": int((count_array >= minimum_n).sum()),
                "minimum_complete_case_n": int(count_array.min()),
                "median_complete_case_n": float(np.median(count_array)),
                "maximum_complete_case_n": int(count_array.max()),
            }
        )

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(phase_rows)
    metadata = {
        "association_or_mediation_model_fitted": False,
        "outcome_effect_estimates_computed": False,
        "outcome_values_exported": False,
        "allowed_information_used": (
            "phase membership, longitudinal availability, missingness, and "
            "complete-case sample counts only"
        ),
        "target_outcomes": list(outcomes),
        "marker_specs": list(marker_specs),
        "selected_pathway_count": int(len(exposures)),
        "minimum_n": int(minimum_n),
        "observed_COLPROT": sorted(
            subjects["COLPROT"].dropna().astype(str).unique().tolist()
        ),
    }
    return detail, summary, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    history, history_summary = audit_historical_access(
        args.experiments_root,
        args.access_record,
    )
    detail, phase_summary, sample_metadata = audit_phase_samples(
        args.dataset_root,
        args.pathway_root,
        marker_specs=args.marker_specs,
        outcomes=args.outcomes,
        minimum_n=args.minimum_n,
    )

    history_path = args.output_root / "historical_outcome_access_audit.csv"
    detail_path = args.output_root / "phase_candidate_sample_counts.csv"
    phase_path = args.output_root / "phase_sample_summary.csv"
    history.to_csv(history_path, index=False)
    detail.to_csv(detail_path, index=False)
    phase_summary.to_csv(phase_path, index=False)

    combined = phase_summary.loc[
        phase_summary["phase_group"].eq("ADNI1_GO_2_combined")
    ]
    combined_complete = bool(
        len(combined)
        and int(combined.iloc[0]["estimable_at_minimum_n"])
        == int(combined.iloc[0]["candidate_cells"])
    )
    verdict = {
        "pre_access_target_outcome_leak_detected": bool(
            history_summary["pre_access_leak_detected"]
        ),
        "adni1_go_2_all_candidates_meet_minimum_n": combined_complete,
        "adni1_go_2_status": (
            "eligible_as_phase-held-out confirmation values, not as a new endpoint-held-out set"
            if not history_summary["pre_access_leak_detected"] and combined_complete
            else "not eligible as the sole formal confirmation set under the current minimum-N rule"
        ),
        "reason": (
            "The endpoint definitions and ADNI3 results are already known. This audit found no "
            "pre-access target-outcome mediation artifact, so untouched ADNI1/GO/2 values may "
            "serve only as an independently phase-held-out validation if sample feasibility passes."
        ),
    }
    payload = {
        "schema_version": "neurooracle.case2_v3_outcome_access_sample_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "outcome-access provenance and sample feasibility only",
        "history": history_summary,
        "sample_audit": sample_metadata,
        "verdict": verdict,
        "outputs": {
            "historical_outcome_access_audit": str(history_path),
            "phase_candidate_sample_counts": str(detail_path),
            "phase_sample_summary": str(phase_path),
        },
        "output_sha256": {
            history_path.name: _sha256(history_path),
            detail_path.name: _sha256(detail_path),
            phase_path.name: _sha256(phase_path),
        },
        "script_sha256": _sha256(Path(__file__)),
    }
    audit_path = args.output_root / "AUDIT.json"
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(phase_summary.to_string(index=False))
    return payload


def parse_args() -> argparse.Namespace:
    experiments_root = CASE2_ROOT / "experiments"
    access_record = (
        experiments_root
        / "case2_adni_confirmatory_v1"
        / "a0bdc6a564ba"
        / "hidden_confirmation_results"
        / "CONFIRMATION_OUTCOME_ACCESS_STARTED.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=experiments_root)
    parser.add_argument("--access-record", type=Path, default=access_record)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CASE2_ROOT / "audits" / "case2_v3_outcome_access_and_sample_audit",
    )
    parser.add_argument("--outcomes", nargs="+", default=list(TARGET_OUTCOMES))
    parser.add_argument("--marker-specs", nargs="+", default=list(DEFAULT_MARKER_SPECS))
    parser.add_argument("--minimum-n", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 12:40 HKT
