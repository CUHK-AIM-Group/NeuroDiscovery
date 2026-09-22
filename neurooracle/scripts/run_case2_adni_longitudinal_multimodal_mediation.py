"""Run the longitudinal multimodal ADNI mediation screen for Case Study 2.

The analysis tests pathway PRS -> imaging marker -> subsequent clinical decline.
It is an association-based mediation screen and does not establish causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

from neurooracle.scripts.run_case2_adni_mediation_smoke import mediation_screen


CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_DATASET_ROOT = (
    CASE2_ROOT
    / "experiment_tables"
    / "case2_adni_multimodal_v1"
)
DEFAULT_PATHWAY_ROOT = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
    / "features"
    / "ad_pathway_prs_v1"
)
DEFAULT_OUTPUT_BASE = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_longitudinal_multimodal_smoke_v1"
)
DEFAULT_OUTCOMES = ("MMSE", "ADAS13", "CDRSB")
DEFAULT_MARKER_SPECS = (
    "amyloid_pet::CENTILOIDS",
    "fdg_pet::FDG_META_ROI_SUVR",
    "tau_pet::META_TEMPORAL_SUVR",
    "tau_pet::CTX_ENTORHINAL_SUVR",
    "smri_adnimerge::Hippocampus",
    "smri_adnimerge::Entorhinal",
    "smri_adnimerge::Fusiform",
    "smri_adnimerge::MidTemp",
    "smri_adnimerge::WholeBrain",
    "smri_adnimerge::Ventricles",
)
BASE_NUMERIC_COVARIATES = (
    "age_at_imaging",
    "sex_binary",
    "PTEDUCAT",
    "baseline_value",
    "followup_years",
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "PC7",
    "PC8",
    "PC9",
    "PC10",
)
CATEGORICAL_COVARIATES = ("SITE", "COLPROT", "batch")


def select_curated_pathway_exposures(
    subjects: pd.DataFrame,
    score_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Select one outcome-blind PRS threshold per curated pathway."""

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
    available = score_manifest[
        score_manifest["score_name"].astype(str).isin(subjects.columns)
        & score_manifest["pathway_source"].astype(str).eq("NeuroOracle curated")
    ].copy()
    if available.empty:
        raise ValueError("No curated pathway PRS columns are available")
    priority = {"p1em03": 0, "p5em02": 1}
    available["threshold_priority"] = (
        available["threshold_label"].map(priority).fillna(99).astype(int)
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


def build_index_analysis_rows(
    markers: pd.DataFrame,
    longitudinal_pairs: pd.DataFrame,
    subjects: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    marker_specs: Iterable[str],
    outcomes: Iterable[str],
    target_followup_days: int = 730,
    min_followup_days: int = 365,
    max_followup_days: int = 1095,
) -> pd.DataFrame:
    """Build one prespecified index-imaging row per subject and test family."""

    markers = markers.copy()
    markers["marker_spec"] = (
        markers["modality"].astype(str) + "::" + markers["marker"].astype(str)
    )
    markers["value"] = pd.to_numeric(markers["value"], errors="coerce")
    selected_markers = markers[
        markers["qc_primary_eligible"].fillna(False).astype(bool)
        & markers["marker_spec"].isin(tuple(marker_specs))
        & markers["value"].notna()
    ].copy()
    if selected_markers.empty:
        raise ValueError("No eligible imaging marker rows match the requested specs")

    icv = (
        markers[
            markers["qc_primary_eligible"].fillna(False).astype(bool)
            & markers["modality"].eq("smri_adnimerge")
            & markers["marker"].eq("ICV")
        ][["visit_id", "value"]]
        .drop_duplicates("visit_id")
        .rename(columns={"value": "icv"})
    )
    pairs = longitudinal_pairs[
        longitudinal_pairs["outcome"].isin(tuple(outcomes))
        & longitudinal_pairs["followup_days"].between(
            min_followup_days,
            max_followup_days,
        )
    ].copy()
    join_keys = ["visit_id", "subject_id", "modality", "imaging_date"]
    candidates = selected_markers.merge(
        pairs,
        on=join_keys,
        how="inner",
        validate="many_to_many",
    )
    candidates["horizon_distance_days"] = (
        candidates["followup_days"] - target_followup_days
    ).abs()
    family_keys = ["subject_id", "modality", "marker", "outcome"]
    candidates = (
        candidates.sort_values(
            [
                *family_keys,
                "imaging_date",
                "horizon_distance_days",
                "future_outcome_date",
            ],
            kind="stable",
        )
        .drop_duplicates(family_keys)
        .reset_index(drop=True)
    )
    candidates = candidates.merge(icv, on="visit_id", how="left")

    subject_columns = [
        column
        for column in subjects.columns
        if column == "subject_id"
        or column.startswith("pathway_prs__")
        or column in {
            "AGE",
            "sex_binary",
            "PTEDUCAT",
            "SITE",
            "COLPROT",
            "batch",
            *(f"PC{index}" for index in range(1, 11)),
        }
    ]
    candidates = candidates.merge(
        subjects.loc[:, subject_columns],
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )
    baseline_dates = (
        clinical.groupby("subject_id", as_index=False)["clinical_date"]
        .min()
        .rename(columns={"clinical_date": "baseline_clinical_date"})
    )
    candidates = candidates.merge(
        baseline_dates,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    candidates["age_at_imaging"] = pd.to_numeric(
        candidates["AGE"], errors="coerce"
    ) + (
        candidates["imaging_date"] - candidates["baseline_clinical_date"]
    ).dt.days / 365.25
    candidates["family_id"] = (
        candidates["modality"].astype(str)
        + "::"
        + candidates["marker"].astype(str)
        + "::"
        + candidates["outcome"].astype(str)
    )
    if candidates.duplicated(family_keys).any():
        raise ValueError("Analysis rows are not unique per subject and family")
    return candidates


def _independent_columns(matrix: np.ndarray) -> list[int]:
    selected: list[int] = []
    current = np.empty((matrix.shape[0], 0), dtype=float)
    rank = 0
    for index in range(matrix.shape[1]):
        candidate = np.column_stack([current, matrix[:, index]])
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            selected.append(index)
            current = candidate
            rank = candidate_rank
    return selected


def build_longitudinal_covariate_matrix(
    rows: pd.DataFrame,
    *,
    include_icv: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build a standardized, full-rank covariate matrix for one family."""

    numeric_columns = list(BASE_NUMERIC_COVARIATES)
    if include_icv:
        numeric_columns.append("icv")
    numeric = rows.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_parts: list[np.ndarray] = []
    numeric_names: list[str] = []
    for column in numeric:
        values = numeric[column].to_numpy(float)
        scale = float(values.std(ddof=0))
        if np.isfinite(scale) and scale > np.finfo(float).eps:
            numeric_parts.append((values - float(values.mean())) / scale)
            numeric_names.append(column)

    categorical = pd.get_dummies(
        rows.loc[:, list(CATEGORICAL_COVARIATES)].astype(str),
        prefix=list(CATEGORICAL_COVARIATES),
        drop_first=True,
        dtype=float,
    )
    parts: list[np.ndarray] = [np.ones((len(rows), 1), dtype=float)]
    names = ["intercept"]
    if numeric_parts:
        parts.append(np.column_stack(numeric_parts))
        names.extend(numeric_names)
    if not categorical.empty:
        parts.append(categorical.to_numpy(float))
        names.extend(map(str, categorical.columns))
    matrix = np.column_stack(parts)
    selected = _independent_columns(matrix)
    return matrix[:, selected], [names[index] for index in selected]


def _complete_case_mask(
    rows: pd.DataFrame,
    exposure: str,
    *,
    include_icv: bool,
) -> pd.Series:
    numeric = [
        exposure,
        "value",
        "annualized_decline_score",
        *BASE_NUMERIC_COVARIATES,
    ]
    if include_icv:
        numeric.append("icv")
    mask = rows.loc[:, numeric].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    for column in CATEGORICAL_COVARIATES:
        mask &= rows[column].notna()
    return mask


def _fdr(values: pd.Series) -> np.ndarray:
    adjusted = np.full(len(values), np.nan)
    numeric = pd.to_numeric(values, errors="coerce")
    finite = np.isfinite(numeric)
    if finite.any():
        adjusted[finite] = multipletests(numeric[finite], method="fdr_bh")[1]
    return adjusted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Execute the prespecified smoke screen and save reproducibility artifacts."""

    output_root = args.output_root
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        if not args.force:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_parquet(
        args.dataset_root / "case2_adni_subject_genetics_covariates.parquet"
    )
    clinical = pd.read_parquet(
        args.dataset_root / "case2_adni_clinical_visits.parquet"
    )
    markers = pd.read_parquet(
        args.dataset_root / "case2_adni_primary_imaging_markers_long.parquet"
    )
    longitudinal = pd.read_parquet(
        args.dataset_root / "case2_adni_longitudinal_outcome_pairs.parquet"
    )
    score_manifest = pd.read_csv(args.pathway_root / "score_manifest.csv")
    exposures = select_curated_pathway_exposures(subjects, score_manifest)
    analysis_rows = build_index_analysis_rows(
        markers,
        longitudinal,
        subjects,
        clinical,
        marker_specs=args.marker_specs,
        outcomes=args.outcomes,
        target_followup_days=args.target_followup_days,
        min_followup_days=args.min_followup_days,
        max_followup_days=args.max_followup_days,
    )

    results: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    grouped = analysis_rows.groupby(["modality", "marker", "outcome"], sort=True)
    for (modality, marker, outcome), family in grouped:
        include_icv = modality == "smri_adnimerge"
        for exposure_row in exposures.itertuples(index=False):
            exposure = str(exposure_row.score_name)
            mask = _complete_case_mask(
                family,
                exposure,
                include_icv=include_icv,
            )
            complete = family.loc[mask].copy()
            if len(complete) < args.min_n:
                skipped.append(
                    {
                        "exposure": exposure,
                        "modality": modality,
                        "marker": marker,
                        "outcome": outcome,
                        "n": int(len(complete)),
                        "reason": "below_min_n",
                    }
                )
                continue
            covariates, covariate_names = build_longitudinal_covariate_matrix(
                complete,
                include_icv=include_icv,
            )
            fitted, marker_mask = mediation_screen(
                pd.to_numeric(complete[exposure], errors="coerce").to_numpy(float),
                pd.to_numeric(
                    complete["annualized_decline_score"], errors="coerce"
                ).to_numpy(float),
                pd.to_numeric(complete["value"], errors="coerce")
                .to_numpy(float)
                .reshape(-1, 1),
                covariates,
            )
            if not marker_mask[0] or fitted.empty:
                skipped.append(
                    {
                        "exposure": exposure,
                        "modality": modality,
                        "marker": marker,
                        "outcome": outcome,
                        "n": int(len(complete)),
                        "reason": "non_estimable_marker",
                    }
                )
                continue
            record = fitted.iloc[0].to_dict()
            record.update(
                {
                    "exposure": exposure,
                    "pathway_id": exposure_row.pathway_id,
                    "pathway_name": exposure_row.pathway_name,
                    "threshold_label": exposure_row.threshold_label,
                    "gene_count": exposure_row.gene_count,
                    "modality": modality,
                    "marker": marker,
                    "outcome": outcome,
                    "outcome_scale": "positive values indicate annualized worsening",
                    "n_subjects": int(len(complete)),
                    "median_followup_years": float(complete["followup_years"].median()),
                    "covariates": ";".join(covariate_names),
                    "abs_indirect_effect_std": abs(float(record["indirect_effect_std"])),
                }
            )
            results.append(record)

    result_table = pd.DataFrame(results)
    if result_table.empty:
        raise RuntimeError("No mediation family met the minimum sample size")
    result_table["sobel_q_global"] = _fdr(result_table["sobel_p"])
    result_table["a_path_q_global"] = _fdr(result_table["a_path_p"])
    result_table["b_path_q_global"] = _fdr(result_table["b_path_p"])
    result_table["sobel_q_outcome"] = np.nan
    for _, index in result_table.groupby("outcome", sort=False).groups.items():
        result_table.loc[index, "sobel_q_outcome"] = _fdr(
            result_table.loc[index, "sobel_p"]
        )
    result_table["fdr_significant_global"] = (
        result_table["sobel_q_global"] < args.fdr_alpha
    )
    result_table = result_table.sort_values(
        ["sobel_q_global", "sobel_p", "abs_indirect_effect_std"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    result_table.insert(0, "rank", np.arange(1, len(result_table) + 1))

    result_path = output_root / "all_longitudinal_mediation_results.parquet"
    top_path = output_root / "top_longitudinal_mediation_results.csv"
    skipped_path = output_root / "skipped_families.csv"
    selection_path = output_root / "analysis_row_selection.parquet"
    exposure_path = output_root / "selected_pathway_exposures.csv"
    result_table.to_parquet(result_path, index=False, compression="zstd")
    result_table.head(args.top_n).to_csv(top_path, index=False)
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    analysis_rows.to_parquet(selection_path, index=False, compression="zstd")
    exposures.to_csv(exposure_path, index=False)

    now = datetime.now(timezone(timedelta(hours=8)))
    task_manifest = {
        "session_id": now.strftime("case2-longitudinal-%Y%m%d-%H%M%S"),
        "tasks": [
            "load controlled ADNI genetics, imaging, and outcome tables",
            "select outcome-blind curated pathway PRS exposures",
            "select one index imaging visit and approximately two-year outcome",
            "run covariate-adjusted linear indirect-effect screens",
            "apply global and outcome-wise Benjamini-Hochberg correction",
            "verify time ordering, uniqueness, and output integrity",
        ],
        "success_criteria": {
            "minimum_family_n": args.min_n,
            "unique_subject_family_rows": True,
            "future_outcome_after_imaging": True,
            "finite_model_inputs": True,
        },
    }
    environment = {
        "created_at_hkt": now.isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "random_seed": args.seed,
    }
    (output_root / "experiment_task_manifest.json").write_text(
        json.dumps(task_manifest, indent=2), encoding="utf-8"
    )
    (output_root / "environment_manifest.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    unique_ok = not analysis_rows.duplicated(
        ["subject_id", "modality", "marker", "outcome"]
    ).any()
    temporal_ok = analysis_rows["future_outcome_date"].gt(
        analysis_rows["imaging_date"]
    ).all()
    audit = {
        "status": "passed" if unique_ok and temporal_ok else "failed",
        "unique_subject_family_rows": bool(unique_ok),
        "future_outcome_after_imaging": bool(temporal_ok),
        "analysis_rows": int(len(analysis_rows)),
        "subjects": int(analysis_rows["subject_id"].nunique()),
        "tests_completed": int(len(result_table)),
        "tests_skipped": int(len(skipped)),
    }
    (output_root / "experiment_audit_report.md").write_text(
        "# Case Study 2 Longitudinal Smoke Audit\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in audit.items())
        + "\n",
        encoding="utf-8",
    )
    experiment_summary = (
        "# Case Study 2 Experiment\n\n"
        f"Generated: {now.isoformat(timespec='seconds')}\n\n"
        "Association-based pathway PRS to imaging marker to subsequent clinical "
        "decline smoke screen. This analysis is not causal.\n\n"
        f"- Subjects represented: {analysis_rows['subject_id'].nunique()}\n"
        f"- Tests completed: {len(result_table)}\n"
        f"- Global FDR hits: {int(result_table['fdr_significant_global'].sum())}\n"
        f"- Minimum Sobel P: {result_table['sobel_p'].min():.6g}\n"
    )
    (output_root / "EXPERIMENT.md").write_text(experiment_summary, encoding="utf-8")

    hash_targets = [
        result_path,
        top_path,
        skipped_path,
        selection_path,
        exposure_path,
        output_root / "experiment_task_manifest.json",
        output_root / "environment_manifest.json",
        output_root / "experiment_audit_report.md",
        output_root / "EXPERIMENT.md",
    ]
    hashes = {path.name: _sha256(path) for path in hash_targets}
    (output_root / "output_sha256.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    payload: dict[str, object] = {
        "created_at_hkt": now.isoformat(timespec="seconds"),
        "analysis_type": "longitudinal multimodal association-based mediation smoke screen",
        "causal_interpretation": False,
        "dataset_root": str(args.dataset_root),
        "output_root": str(output_root),
        "exposures": exposures["score_name"].astype(str).tolist(),
        "marker_specs": list(args.marker_specs),
        "outcomes": list(args.outcomes),
        "target_followup_days": args.target_followup_days,
        "followup_window_days": [args.min_followup_days, args.max_followup_days],
        "minimum_n": args.min_n,
        "analysis_rows": int(len(analysis_rows)),
        "subjects": int(analysis_rows["subject_id"].nunique()),
        "tests_completed": int(len(result_table)),
        "tests_skipped": int(len(skipped)),
        "global_fdr_hits": int(result_table["fdr_significant_global"].sum()),
        "nominal_sobel_hits": int((result_table["sobel_p"] < 0.05).sum()),
        "minimum_sobel_p": float(result_table["sobel_p"].min()),
        "audit": audit,
        "outputs": {path.stem: str(path) for path in hash_targets},
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_BASE / timestamp,
    )
    parser.add_argument("--outcomes", nargs="+", default=list(DEFAULT_OUTCOMES))
    parser.add_argument(
        "--marker-specs", nargs="+", default=list(DEFAULT_MARKER_SPECS)
    )
    parser.add_argument("--target-followup-days", type=int, default=730)
    parser.add_argument("--min-followup-days", type=int, default=365)
    parser.add_argument("--max-followup-days", type=int, default=1095)
    parser.add_argument("--min-n", type=int, default=120)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    run_analysis(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-07-31 19:54 HKT
