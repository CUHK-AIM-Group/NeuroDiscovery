"""Run an association-based ADNI mediation screen for Case Study 2.

This smoke test validates the executable path from genetic exposure through
an imaging marker to a clinical outcome. It is a cross-sectional linear
indirect-effect screen, not a causal mediation analysis. The final Case Study
2 experiment should replace genome-wide PRS exposures with pathway-level PRS.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_DATASET_ROOT = (
    CASE2_ROOT
    / "experiment_tables"
    / "case2_adni_fmri_v1"
)
DEFAULT_OUTPUT_ROOT = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_mediation_smoke_v1"
)
DEFAULT_PATHWAY_PRS = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
    / "features"
    / "ad_pathway_prs_v1"
    / "case2_adni_pathway_prs_wide.parquet"
)
DEFAULT_SOURCE = "schaefer_400_7net"
DEFAULT_EXPOSURES = (
    "apoe_e4_dosage",
    "ad_prs_p5em08_avg",
    "ad_prs_p1em05_avg",
    "ad_prs_p1em03_avg",
    "ad_prs_p5em02_avg",
    "ad_prs_p1e00_avg",
)
DEFAULT_OUTCOMES = (
    "diagnosis_code",
    "ADAS13",
    "MMSE",
    "CDRSB",
)
NUMERIC_COVARIATES = (
    "AGE",
    "sex_binary",
    "PTEDUCAT",
    "imaging_mean_fd",
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
OUTCOME_HIGHER_IS = {
    "diagnosis_code": "greater diagnostic severity (CN=0, MCI=1, AD=2)",
    "ADAS11": "worse cognition",
    "ADAS13": "worse cognition",
    "MMSE": "better cognition",
    "MOCA": "better cognition",
    "CDRSB": "greater clinical impairment",
}
MARKER_METADATA_COLUMNS = (
    "marker_id",
    "source",
    "atlas",
    "roi_index",
    "roi_id",
    "roi_name",
    "parcel_name",
    "hemisphere",
    "network",
    "structure_class",
    "feature",
)


def _numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def complete_case_mask(
    subjects: pd.DataFrame,
    exposure: str,
    outcome: str,
    *,
    fd_max: float,
) -> pd.Series:
    """Return complete cases for one exposure-outcome family."""

    required_numeric = [
        exposure,
        outcome,
        *NUMERIC_COVARIATES,
    ]
    missing = [column for column in required_numeric if column not in subjects]
    missing.extend(
        column for column in CATEGORICAL_COVARIATES if column not in subjects
    )
    if missing:
        raise ValueError(f"Subject table is missing columns: {sorted(set(missing))}")

    numeric = _numeric_frame(subjects, required_numeric)
    mask = numeric.notna().all(axis=1)
    mask &= numeric["imaging_mean_fd"].le(fd_max)
    for column in CATEGORICAL_COVARIATES:
        mask &= subjects[column].notna()
    if exposure == "apoe_e4_dosage" and "apoe_complete" in subjects:
        mask &= subjects["apoe_complete"].fillna(False).astype(bool)
    return mask


def build_covariate_matrix(subjects: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build a full-rank numeric covariate matrix with an intercept."""

    numeric = _numeric_frame(subjects, NUMERIC_COVARIATES)
    numeric_values = []
    numeric_names = []
    for column in numeric:
        values = numeric[column].to_numpy(dtype=float)
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            continue
        numeric_values.append((values - float(values.mean())) / scale)
        numeric_names.append(column)

    categorical = pd.get_dummies(
        subjects.loc[:, list(CATEGORICAL_COVARIATES)].astype(str),
        prefix=list(CATEGORICAL_COVARIATES),
        drop_first=True,
        dtype=float,
    )
    parts = [np.ones((len(subjects), 1), dtype=float)]
    names = ["intercept"]
    if numeric_values:
        parts.append(np.column_stack(numeric_values))
        names.extend(numeric_names)
    if not categorical.empty:
        parts.append(categorical.to_numpy(dtype=float))
        names.extend(map(str, categorical.columns))

    matrix = np.column_stack(parts)
    _, independent = np.linalg.qr(matrix.T, mode="reduced")
    del independent
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1]:
        # SVD-based selection is stable for the small ADNI covariate matrix.
        _, _, pivot = _independent_columns(matrix)
        matrix = matrix[:, pivot]
        names = [names[index] for index in pivot]
    return matrix, names


def _independent_columns(matrix: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    """Select deterministic linearly independent columns."""

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
    return current, rank, selected


def _residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(covariates, values, rcond=None)[0]
    return values - covariates @ coefficients


def _standardize_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0)
    valid = np.isfinite(scale) & (scale > np.finfo(float).eps)
    standardized = centered[:, valid] / scale[valid]
    return standardized, valid


def mediation_screen(
    exposure: np.ndarray,
    outcome: np.ndarray,
    mediators: np.ndarray,
    covariates: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Vectorize standardized linear indirect-effect tests across mediators."""

    exposure = np.asarray(exposure, dtype=float).reshape(-1, 1)
    outcome = np.asarray(outcome, dtype=float).reshape(-1, 1)
    mediators = np.asarray(mediators, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if mediators.ndim != 2:
        raise ValueError("mediators must be a two-dimensional array")
    if not (
        len(exposure)
        == len(outcome)
        == len(mediators)
        == len(covariates)
    ):
        raise ValueError("All model arrays must contain the same number of rows")
    if not np.isfinite(
        np.column_stack([exposure, outcome, covariates])
    ).all():
        raise ValueError("Exposure, outcome, and covariates must be finite")

    complete_mediators = np.isfinite(mediators).all(axis=0)
    mediator_subset = mediators[:, complete_mediators]
    residuals = _residualize(
        np.column_stack([exposure, outcome, mediator_subset]),
        covariates,
    )
    standardized, nonconstant = _standardize_columns(residuals)
    if len(nonconstant) < 2 or not nonconstant[0] or not nonconstant[1]:
        raise ValueError("Exposure or outcome has no residual variance")

    mediator_nonconstant = nonconstant[2:]
    marker_mask = complete_mediators.copy()
    marker_mask[complete_mediators] = mediator_nonconstant
    x = standardized[:, 0]
    y = standardized[:, 1]
    marker_start = 2
    marker_stop = marker_start + int(mediator_nonconstant.sum())
    m = standardized[:, marker_start:marker_stop]

    n = len(x)
    covariate_rank = int(np.linalg.matrix_rank(covariates))
    df_a = n - covariate_rank - 1
    df_b = n - covariate_rank - 2
    if df_b <= 0:
        raise ValueError(
            f"Insufficient residual degrees of freedom: n={n}, "
            f"covariate_rank={covariate_rank}"
        )

    x_x = float(x @ x)
    y_y = float(y @ y)
    x_y = float(x @ y)
    m_m = np.einsum("ij,ij->j", m, m)
    x_m = x @ m
    m_y = m.T @ y
    denominator = x_x * m_m - x_m**2
    estimable = denominator > np.finfo(float).eps

    a = np.full(m.shape[1], np.nan)
    b = np.full(m.shape[1], np.nan)
    direct = np.full(m.shape[1], np.nan)
    se_a = np.full(m.shape[1], np.nan)
    se_b = np.full(m.shape[1], np.nan)
    p_a = np.full(m.shape[1], np.nan)
    p_b = np.full(m.shape[1], np.nan)

    a[estimable] = x_m[estimable] / x_x
    direct[estimable] = (
        x_y * m_m[estimable] - m_y[estimable] * x_m[estimable]
    ) / denominator[estimable]
    b[estimable] = (
        m_y[estimable] * x_x - x_y * x_m[estimable]
    ) / denominator[estimable]

    residual_m_ss = m_m - (x_m**2 / x_x)
    sigma2_a = residual_m_ss[estimable] / df_a
    se_a[estimable] = np.sqrt(sigma2_a / x_x)
    t_a = a[estimable] / se_a[estimable]
    p_a[estimable] = 2 * stats.t.sf(np.abs(t_a), df=df_a)

    residual_y_ss = (
        y_y
        - direct[estimable] * x_y
        - b[estimable] * m_y[estimable]
    )
    residual_y_ss = np.maximum(residual_y_ss, 0)
    sigma2_b = residual_y_ss / df_b
    se_b[estimable] = np.sqrt(sigma2_b * x_x / denominator[estimable])
    t_b = b[estimable] / se_b[estimable]
    p_b[estimable] = 2 * stats.t.sf(np.abs(t_b), df=df_b)

    indirect = a * b
    sobel_se = np.sqrt((b**2 * se_a**2) + (a**2 * se_b**2))
    sobel_z = indirect / sobel_se
    p_sobel = 2 * stats.norm.sf(np.abs(sobel_z))
    q_sobel = np.full(len(p_sobel), np.nan)
    finite_p = np.isfinite(p_sobel)
    if finite_p.any():
        q_sobel[finite_p] = multipletests(
            p_sobel[finite_p],
            method="fdr_bh",
        )[1]

    result = pd.DataFrame(
        {
            "a_path_std": a,
            "a_path_se": se_a,
            "a_path_p": p_a,
            "b_path_std": b,
            "b_path_se": se_b,
            "b_path_p": p_b,
            "total_effect_std": x_y / x_x,
            "direct_effect_std": direct,
            "indirect_effect_std": indirect,
            "sobel_se": sobel_se,
            "sobel_z": sobel_z,
            "sobel_p": p_sobel,
            "sobel_q_family": q_sobel,
            "n": n,
            "covariate_rank": covariate_rank,
            "df_a": df_a,
            "df_b": df_b,
        }
    )
    return result, marker_mask


def load_imaging_matrix(
    imaging_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one atlas and return subject-by-marker values plus marker metadata."""

    if not imaging_path.is_file():
        raise FileNotFoundError(imaging_path)
    imaging = pd.read_parquet(imaging_path)
    required = {"subject_id", "marker_id", "value", *MARKER_METADATA_COLUMNS}
    missing = required - set(imaging.columns)
    if missing:
        raise ValueError(f"Imaging table is missing columns: {sorted(missing)}")
    duplicated = imaging.duplicated(["subject_id", "marker_id"])
    if duplicated.any():
        raise ValueError(
            "Imaging table has duplicate subject-marker rows: "
            f"{int(duplicated.sum())}"
        )

    metadata = (
        imaging.loc[:, list(MARKER_METADATA_COLUMNS)]
        .drop_duplicates("marker_id")
        .set_index("marker_id")
    )
    matrix = imaging.pivot(
        index="subject_id",
        columns="marker_id",
        values="value",
    )
    metadata = metadata.loc[matrix.columns].reset_index()
    return matrix, metadata


def load_pathway_exposures(
    pathway_prs_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Load pathway PRS columns and their auditable pathway metadata."""

    if not pathway_prs_path.is_file():
        raise FileNotFoundError(pathway_prs_path)
    pathway = pd.read_parquet(pathway_prs_path)
    if "subject_id" not in pathway:
        raise ValueError("Pathway PRS table has no subject_id column")
    if pathway["subject_id"].duplicated().any():
        raise ValueError("Pathway PRS table has duplicate subject_id values")
    score_columns = [
        column
        for column in pathway
        if str(column).startswith("pathway_prs__")
    ]
    if not score_columns:
        raise ValueError("Pathway PRS table has no pathway_prs__ columns")

    manifest_path = pathway_prs_path.parent / "score_manifest.csv"
    metadata: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        metadata = {
            str(row.score_name): row._asdict()
            for row in manifest.itertuples(index=False)
        }
    return pathway.loc[:, ["subject_id", *score_columns]], metadata


def _family_summary(
    results: pd.DataFrame,
    *,
    exposure: str,
    outcome: str,
    n_complete: int,
    markers_available: int,
) -> dict[str, object]:
    ranked = results.sort_values(
        ["sobel_p", "abs_indirect_effect_std"],
        ascending=[True, False],
        kind="stable",
    )
    top = ranked.iloc[0] if not ranked.empty else None
    return {
        "exposure": exposure,
        "outcome": outcome,
        "outcome_higher_is": OUTCOME_HIGHER_IS.get(outcome, "not specified"),
        "n_complete": n_complete,
        "markers_available": markers_available,
        "markers_tested": int(len(results)),
        "fdr_hits_q_lt_0p05": int(
            (results["sobel_q_family"] < 0.05).fillna(False).sum()
        ),
        "nominal_hits_p_lt_0p05": int(
            (results["sobel_p"] < 0.05).fillna(False).sum()
        ),
        "minimum_sobel_p": (
            float(results["sobel_p"].min()) if not results.empty else None
        ),
        "minimum_sobel_q": (
            float(results["sobel_q_family"].min()) if not results.empty else None
        ),
        "top_marker_id": str(top["marker_id"]) if top is not None else "",
        "top_indirect_effect_std": (
            float(top["indirect_effect_std"]) if top is not None else None
        ),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Run every configured exposure-outcome family and write audit outputs."""

    subject_path = args.dataset_root / "case2_adni_subject_table.parquet"
    imaging_path = (
        args.dataset_root
        / "imaging_features"
        / f"{args.source.removesuffix('_multiatlas')}.parquet"
    )
    if not subject_path.is_file():
        raise FileNotFoundError(subject_path)
    existing = (
        any(path.is_file() for path in args.output_root.rglob("*"))
        if args.output_root.exists()
        else False
    )
    if existing and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {args.output_root}. "
            "Use a new version or --force."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_parquet(subject_path).set_index("subject_id", drop=False)
    exposure_metadata: dict[str, dict[str, object]] = {}
    pathway_exposures: tuple[str, ...] = ()
    if args.exposure_set in {"pathway", "all"} or any(
        str(exposure).startswith("pathway_prs__")
        for exposure in (args.exposures or ())
    ):
        pathway, exposure_metadata = load_pathway_exposures(args.pathway_prs)
        pathway = pathway.set_index("subject_id")
        pathway_exposures = tuple(
            column
            for column in pathway
            if str(column).startswith("pathway_prs__")
        )
        subjects = subjects.join(
            pathway.loc[:, list(pathway_exposures)],
            how="left",
            validate="one_to_one",
        )
    if args.exposures:
        exposures = tuple(args.exposures)
    elif args.exposure_set == "pathway":
        exposures = pathway_exposures
    elif args.exposure_set == "all":
        exposures = (*DEFAULT_EXPOSURES, *pathway_exposures)
    else:
        exposures = DEFAULT_EXPOSURES

    imaging_matrix, marker_metadata = load_imaging_matrix(imaging_path)
    common_ids = subjects.index.intersection(imaging_matrix.index, sort=False)
    subjects = subjects.loc[common_ids].copy()
    imaging_matrix = imaging_matrix.loc[common_ids]

    all_results = []
    summaries = []
    for exposure in exposures:
        if exposure not in subjects:
            raise ValueError(f"Unknown exposure column: {exposure}")
        for outcome in args.outcomes:
            if outcome not in subjects:
                raise ValueError(f"Unknown outcome column: {outcome}")
            family_mask = complete_case_mask(
                subjects,
                exposure,
                outcome,
                fd_max=args.fd_max,
            )
            family_subjects = subjects.loc[family_mask].copy()
            if len(family_subjects) < args.min_n:
                summaries.append(
                    {
                        "exposure": exposure,
                        "outcome": outcome,
                        "outcome_higher_is": OUTCOME_HIGHER_IS.get(
                            outcome,
                            "not specified",
                        ),
                        "n_complete": int(len(family_subjects)),
                        "markers_available": int(imaging_matrix.shape[1]),
                        "markers_tested": 0,
                        "status": "skipped_below_min_n",
                    }
                )
                continue

            family_imaging = imaging_matrix.loc[family_subjects.index]
            covariates, covariate_names = build_covariate_matrix(family_subjects)
            result, marker_mask = mediation_screen(
                pd.to_numeric(
                    family_subjects[exposure],
                    errors="coerce",
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    family_subjects[outcome],
                    errors="coerce",
                ).to_numpy(dtype=float),
                family_imaging.to_numpy(dtype=float),
                covariates,
            )
            family_metadata = marker_metadata.loc[marker_mask].reset_index(drop=True)
            result = pd.concat([family_metadata, result], axis=1)
            result.insert(0, "outcome", outcome)
            result.insert(0, "exposure", exposure)
            result["outcome_higher_is"] = OUTCOME_HIGHER_IS.get(
                outcome,
                "not specified",
            )
            result["abs_indirect_effect_std"] = result[
                "indirect_effect_std"
            ].abs()
            result["fdr_significant"] = result["sobel_q_family"] < args.fdr_alpha
            result["covariates"] = ";".join(covariate_names)
            for key, value in exposure_metadata.get(exposure, {}).items():
                if key not in result:
                    result[f"exposure_{key}"] = value
            all_results.append(result)

            summary = _family_summary(
                result,
                exposure=exposure,
                outcome=outcome,
                n_complete=len(family_subjects),
                markers_available=imaging_matrix.shape[1],
            )
            summary["status"] = "completed"
            summary["covariates"] = ";".join(covariate_names)
            for key, value in exposure_metadata.get(exposure, {}).items():
                if key not in summary:
                    summary[f"exposure_{key}"] = value
            summaries.append(summary)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    summary_table = pd.DataFrame(summaries)
    if not combined.empty:
        combined.to_parquet(
            args.output_root / "all_marker_results.parquet",
            index=False,
            compression="zstd",
        )
        top = (
            combined.sort_values(
                ["exposure", "outcome", "sobel_p", "abs_indirect_effect_std"],
                ascending=[True, True, True, False],
                kind="stable",
            )
            .groupby(["exposure", "outcome"], sort=False, group_keys=False)
            .head(args.top_n)
        )
        top.to_csv(args.output_root / "top_marker_results.csv", index=False)
    summary_table.to_csv(args.output_root / "family_summary.csv", index=False)

    now_hkt = datetime.now(timezone(timedelta(hours=8))).isoformat()
    payload: dict[str, object] = {
        "created_at_hkt": now_hkt,
        "analysis_type": "cross-sectional association-based linear mediation screen",
        "causal_interpretation": False,
        "genetic_exposure_scope": (
            "Outcome-blind pathway-partitioned AD PRS"
            if args.exposure_set == "pathway"
            else (
                "APOE and genome-wide AD PRS smoke exposures; pathway-level "
                "PRS is required for the final Case Study 2 analysis"
            )
        ),
        "dataset_root": str(args.dataset_root),
        "subject_table": str(subject_path),
        "imaging_table": str(imaging_path),
        "output_root": str(args.output_root),
        "source": args.source,
        "subjects_with_genetics_and_imaging": int(len(subjects)),
        "fd_max": args.fd_max,
        "min_n": args.min_n,
        "fdr_alpha": args.fdr_alpha,
        "exposure_set": args.exposure_set,
        "pathway_prs": (
            str(args.pathway_prs)
            if args.exposure_set in {"pathway", "all"}
            else None
        ),
        "exposures": list(exposures),
        "outcomes": list(args.outcomes),
        "numeric_covariates": list(NUMERIC_COVARIATES),
        "categorical_covariates": list(CATEGORICAL_COVARIATES),
        "families_completed": int(
            (summary_table.get("status") == "completed").sum()
            if not summary_table.empty
            else 0
        ),
        "marker_tests": int(len(combined)),
        "fdr_hits": int(
            combined["fdr_significant"].fillna(False).sum()
            if not combined.empty
            else 0
        ),
        "outputs": {
            "all_marker_results": str(
                args.output_root / "all_marker_results.parquet"
            ),
            "top_marker_results": str(
                args.output_root / "top_marker_results.csv"
            ),
            "family_summary": str(args.output_root / "family_summary.csv"),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--exposure-set",
        choices=("smoke", "pathway", "all"),
        default="smoke",
    )
    parser.add_argument(
        "--pathway-prs",
        type=Path,
        default=DEFAULT_PATHWAY_PRS,
    )
    parser.add_argument(
        "--exposure",
        dest="exposures",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--outcome",
        dest="outcomes",
        action="append",
        default=None,
    )
    parser.add_argument("--fd-max", type=float, default=0.5)
    parser.add_argument("--min-n", type=int, default=80)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.exposures = tuple(args.exposures or ())
    args.outcomes = tuple(args.outcomes or DEFAULT_OUTCOMES)
    return args


def main() -> int:
    run_analysis(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Last Updated At: 2026-07-31 02:55 HKT
