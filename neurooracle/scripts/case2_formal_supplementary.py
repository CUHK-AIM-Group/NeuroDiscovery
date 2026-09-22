"""Shared machinery for the formal supplementary Case Study 2 analysis.

This module is intentionally separate from the historical 210- and
5,265-candidate pathways.  It supports an outcome-blind readiness freeze and,
only after that freeze has been verified, the association-based indirect-
effect model specified in ``case2_adni_formal_supplementary_v1.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


SCHEMA_VERSION = "neurooracle.case2_formal_supplementary_protocol.v1"
MASTER_REGISTRY_COLUMNS = (
    "candidate_id",
    "score_name",
    "pathway_id",
    "pathway_name",
    "threshold_label",
    "modality",
    "marker",
    "marker_spec",
    "outcome",
    "outcome_domain",
    "higher_is_worse_multiplier",
    "readiness_family_id",
    "supplemental_fdr_family_id",
)
ROW_KEY_COLUMNS = (
    "subject_id",
    "visit_id",
    "modality",
    "marker",
    "marker_spec",
    "outcome",
    "readiness_family_id",
    "imaging_date",
    "baseline_outcome_date",
    "future_outcome_date",
    "baseline_distance_days",
    "exact_followup_days",
    "target_distance_days",
)
CONTINUOUS_COMMON_COVARIATES = (
    "age_at_imaging",
    "PTEDUCAT",
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
BINARY_COMMON_COVARIATES = ("sex_binary",)
CATEGORICAL_COMMON_COVARIATES = ("SITE", "COLPROT", "batch")
DEFAULT_MINIMUM_CATEGORY_N = 10
READINESS_FORBIDDEN_COLUMN_TOKENS = (
    "outcome_value",
    "baseline_value",
    "future_value",
    "effect",
    "coefficient",
    "estimate",
    "statistic",
    "p_value",
    "pvalue",
    "q_value",
    "qvalue",
    "beta",
    "sobel",
    "ci_lower",
    "ci_upper",
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with deterministic serialization."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_pin(path: Path) -> dict[str, Any]:
    """Build a reproducibility pin for one input or artifact."""

    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and strictly validate the supplementary protocol."""

    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected protocol schema {SCHEMA_VERSION!r}, found "
            f"{protocol.get('schema_version')!r}"
        )
    required_sections = {
        "cohort",
        "pathway_exposures",
        "imaging_markers",
        "clinical_outcomes",
        "timing",
        "candidate_universe",
        "readiness_gate",
        "model",
        "multiplicity",
        "freeze_semantics",
    }
    missing = required_sections - set(protocol)
    if missing:
        raise ValueError(f"Protocol is missing sections: {sorted(missing)}")

    pathways = protocol["pathway_exposures"]
    markers = protocol["imaging_markers"]
    outcomes = protocol["clinical_outcomes"]
    if len(pathways) != 7 or len(markers) != 8 or len(outcomes) != 3:
        raise ValueError("Formal protocol must contain exactly 7 x 8 x 3 entries")
    expected = int(protocol["candidate_universe"]["expected_count"])
    if expected != len(pathways) * len(markers) * len(outcomes) or expected != 168:
        raise ValueError("Formal candidate universe must be fixed at 168")
    if int(protocol["cohort"]["primary_genetic_subject_count"]) != 691:
        raise ValueError("Formal primary cohort must be fixed at 691 subjects")
    if protocol["cohort"].get("external_validation_required") is not False:
        raise ValueError(
            "Supplementary Case Study 2 must not require external validation"
        )
    if protocol["readiness_gate"].get("audit_is_outcome_blind") is not True:
        raise ValueError("Readiness audit must be outcome-blind")
    pc_threshold = float(
        protocol["readiness_gate"].get("combined_pca_absolute_z_threshold", np.nan)
    )
    if not np.isfinite(pc_threshold) or pc_threshold <= 0:
        raise ValueError("Protocol must freeze a positive combined-PCA z threshold")
    if protocol["model"].get("bootstrap_replicates") != 5000:
        raise ValueError("Primary indirect inference must use 5,000 bootstraps")
    if protocol["model"].get("master_seed_count") != 1:
        raise ValueError("Current formal execution must use exactly one master seed")
    minimum_category_n = int(
        protocol["model"].get("minimum_categorical_level_n", DEFAULT_MINIMUM_CATEGORY_N)
    )
    if minimum_category_n < 2:
        raise ValueError(
            "HC3 requires sparse categorical pooling with a minimum of at least two"
        )
    if protocol["multiplicity"].get("global_primary_family_size") != 168:
        raise ValueError("Global multiplicity denominator must remain 168")
    if protocol["multiplicity"].get("supplemental_family_size") != 8:
        raise ValueError("Supplemental pathway-outcome denominator must remain 8")

    uniqueness_constraints = (
        ("pathways", pathways, ("pathway_id",)),
        ("pathways", pathways, ("score_name",)),
        ("markers", markers, ("modality", "marker")),
        ("outcomes", outcomes, ("outcome",)),
    )
    for label, rows, columns in uniqueness_constraints:
        keys = [tuple(row[column] for column in columns) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Protocol contains duplicate {label}: {columns}")
    return protocol


def build_master_registry(protocol: Mapping[str, Any]) -> pd.DataFrame:
    """Construct the fixed, result-free 168-candidate public registry."""

    rows: list[dict[str, Any]] = []
    for pathway in protocol["pathway_exposures"]:
        for marker in protocol["imaging_markers"]:
            marker_spec = f"{marker['modality']}::{marker['marker']}"
            for outcome in protocol["clinical_outcomes"]:
                candidate_id = "|".join(
                    [
                        str(pathway["score_name"]),
                        str(marker["modality"]),
                        str(marker["marker"]),
                        str(outcome["outcome"]),
                    ]
                )
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        **pathway,
                        **marker,
                        "marker_spec": marker_spec,
                        "outcome": outcome["outcome"],
                        "outcome_domain": outcome["domain"],
                        "higher_is_worse_multiplier": float(
                            outcome["higher_is_worse_multiplier"]
                        ),
                        "readiness_family_id": (f"{marker_spec}::{outcome['outcome']}"),
                        "supplemental_fdr_family_id": (
                            f"{pathway['pathway_id']}::{outcome['outcome']}"
                        ),
                    }
                )
    registry = pd.DataFrame(rows).loc[:, list(MASTER_REGISTRY_COLUMNS)]
    expected = int(protocol["candidate_universe"]["expected_count"])
    if len(registry) != expected or not registry["candidate_id"].is_unique:
        raise ValueError("Master candidate registry is not the fixed unique universe")
    assert_outcome_blind_frame(registry, label="master registry")
    return registry


def assert_outcome_blind_frame(frame: pd.DataFrame, *, label: str) -> None:
    """Reject result-bearing columns from readiness artifacts."""

    leaked = sorted(
        column
        for column in frame.columns
        if any(
            token in str(column).lower() for token in READINESS_FORBIDDEN_COLUMN_TOKENS
        )
    )
    if leaked:
        raise ValueError(f"{label} contains forbidden result columns: {leaked}")


def build_outcome_blind_analysis_keys(
    markers: pd.DataFrame,
    outcomes: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Select one date-only imaging/baseline/future row per subject-family.

    The caller must project the outcome table to subject, outcome, and date.
    Observed outcome values are deliberately rejected here.
    """

    if "value" in outcomes.columns:
        raise ValueError("Outcome values must not be loaded during readiness audit")
    marker_required = {
        "visit_id",
        "subject_id",
        "modality",
        "marker",
        "imaging_date",
        "qc_primary_eligible",
    }
    outcome_required = {"subject_id", "outcome", "outcome_date"}
    missing_marker = marker_required - set(markers)
    missing_outcome = outcome_required - set(outcomes)
    if missing_marker or missing_outcome:
        raise ValueError(
            "Readiness inputs are missing columns: "
            f"markers={sorted(missing_marker)}, outcomes={sorted(missing_outcome)}"
        )

    marker_specs = {
        f"{row['modality']}::{row['marker']}" for row in protocol["imaging_markers"]
    }
    outcome_names = {row["outcome"] for row in protocol["clinical_outcomes"]}
    marker_dates = markers.loc[:, list(marker_required)].copy()
    marker_dates["imaging_date"] = pd.to_datetime(
        marker_dates["imaging_date"], errors="coerce"
    )
    marker_dates["marker_spec"] = (
        marker_dates["modality"].astype(str) + "::" + marker_dates["marker"].astype(str)
    )
    marker_dates = marker_dates[
        marker_dates["qc_primary_eligible"].fillna(False).astype(bool)
        & marker_dates["marker_spec"].isin(marker_specs)
        & marker_dates["imaging_date"].notna()
    ].drop_duplicates(["visit_id", "subject_id", "modality", "marker", "imaging_date"])
    if marker_dates.empty:
        raise ValueError("No primary-QC imaging rows match the formal marker set")

    outcome_dates = outcomes.loc[:, list(outcome_required)].copy()
    outcome_dates["outcome_date"] = pd.to_datetime(
        outcome_dates["outcome_date"], errors="coerce"
    )
    outcome_dates = outcome_dates[
        outcome_dates["outcome"].isin(outcome_names)
        & outcome_dates["outcome_date"].notna()
    ].drop_duplicates(["subject_id", "outcome", "outcome_date"])

    baseline_window = int(protocol["timing"]["baseline_outcome_window_days"])
    baseline_candidates = marker_dates.merge(
        outcome_dates,
        on="subject_id",
        how="inner",
        validate="many_to_many",
    )
    baseline_candidates["baseline_signed_distance_days"] = (
        baseline_candidates["outcome_date"] - baseline_candidates["imaging_date"]
    ).dt.days
    baseline_candidates["baseline_distance_days"] = baseline_candidates[
        "baseline_signed_distance_days"
    ].abs()
    baseline_candidates = baseline_candidates[
        baseline_candidates["baseline_distance_days"].le(baseline_window)
    ]
    baseline_keys = [
        "visit_id",
        "subject_id",
        "modality",
        "marker",
        "outcome",
    ]
    baselines = (
        baseline_candidates.sort_values(
            [*baseline_keys, "baseline_distance_days", "outcome_date"],
            kind="stable",
        )
        .drop_duplicates(baseline_keys)
        .rename(columns={"outcome_date": "baseline_outcome_date"})
    )

    future_dates = outcome_dates.rename(columns={"outcome_date": "future_outcome_date"})
    candidates = baselines.merge(
        future_dates,
        on=["subject_id", "outcome"],
        how="inner",
        validate="many_to_many",
    )
    candidates["exact_followup_days"] = (
        candidates["future_outcome_date"] - candidates["baseline_outcome_date"]
    ).dt.days
    min_days = int(protocol["timing"]["future_min_days_from_baseline"])
    max_days = int(protocol["timing"]["future_max_days_from_baseline"])
    target_days = int(protocol["timing"]["target_followup_days_from_baseline"])
    candidates = candidates[
        candidates["exact_followup_days"].between(min_days, max_days)
    ].copy()
    candidates["target_distance_days"] = (
        candidates["exact_followup_days"] - target_days
    ).abs()

    family_keys = ["subject_id", "modality", "marker", "outcome"]
    selected = (
        candidates.sort_values(
            [
                *family_keys,
                "target_distance_days",
                "baseline_distance_days",
                "imaging_date",
                "future_outcome_date",
                "baseline_outcome_date",
                "visit_id",
            ],
            kind="stable",
        )
        .drop_duplicates(family_keys)
        .reset_index(drop=True)
    )
    selected["readiness_family_id"] = (
        selected["marker_spec"].astype(str) + "::" + selected["outcome"].astype(str)
    )
    if selected.duplicated(family_keys).any():
        raise ValueError("Date-only analysis keys are not unique per subject-family")
    keys = selected.loc[:, list(ROW_KEY_COLUMNS)].copy()
    assert_outcome_blind_frame(keys, label="analysis row keys")
    return keys.sort_values(family_keys, kind="stable").reset_index(drop=True)


def _numeric_complete(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
    )


def genetic_pc_eligibility(
    subjects: pd.DataFrame,
    *,
    absolute_z_threshold: float,
) -> pd.Series:
    """Flag gross combined-cohort PC outliers without using outcome information."""

    if not np.isfinite(absolute_z_threshold) or absolute_z_threshold <= 0:
        raise ValueError("absolute_z_threshold must be positive and finite")
    pc_columns = [f"PC{index}" for index in range(1, 11)]
    missing = set(pc_columns) - set(subjects)
    if missing:
        raise ValueError(f"Subject table is missing PCs: {sorted(missing)}")
    pcs = subjects.loc[:, pc_columns].apply(pd.to_numeric, errors="coerce")
    scales = pcs.std(axis=0, ddof=0)
    if (~np.isfinite(scales) | scales.le(np.finfo(float).eps)).any():
        raise ValueError("A combined-cohort genetic PC has no usable variance")
    standardized = (pcs - pcs.mean(axis=0)) / scales
    return pcs.notna().all(axis=1) & standardized.abs().max(axis=1).le(
        absolute_z_threshold
    )


def build_readiness_audit(
    row_keys: pd.DataFrame,
    markers: pd.DataFrame,
    subjects: pd.DataFrame,
    clinical_dates: pd.DataFrame,
    registry: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count outcome-blind complete cases and apply the whole-family N gate."""

    exposure_names = registry["score_name"].drop_duplicates().tolist()
    subject_columns = [
        "subject_id",
        "AGE",
        *BINARY_COMMON_COVARIATES,
        "PTEDUCAT",
        *[f"PC{index}" for index in range(1, 11)],
        *CATEGORICAL_COMMON_COVARIATES,
        *exposure_names,
    ]
    missing_subject = set(subject_columns) - set(subjects)
    if missing_subject:
        raise ValueError(f"Subject table is missing columns: {sorted(missing_subject)}")
    if subjects["subject_id"].duplicated().any():
        raise ValueError("Subject table contains duplicate subject IDs")
    clinical_required = {"subject_id", "clinical_date"}
    if not clinical_required.issubset(clinical_dates):
        raise ValueError("Clinical date table is missing subject_id or clinical_date")

    icv_visits = set(
        markers.loc[
            markers["qc_primary_eligible"].fillna(False).astype(bool)
            & markers["modality"].eq("smri_adnimerge")
            & markers["marker"].eq("ICV"),
            "visit_id",
        ].astype(str)
    )
    baseline_dates = clinical_dates.loc[:, ["subject_id", "clinical_date"]].copy()
    baseline_dates["clinical_date"] = pd.to_datetime(
        baseline_dates["clinical_date"], errors="coerce"
    )
    baseline_dates = (
        baseline_dates.dropna(subset=["clinical_date"])
        .groupby("subject_id", as_index=False)["clinical_date"]
        .min()
        .rename(columns={"clinical_date": "cohort_baseline_date"})
    )

    subject_readiness = subjects.loc[:, subject_columns].copy()
    pc_threshold = float(
        protocol["readiness_gate"]["combined_pca_absolute_z_threshold"]
    )
    subject_readiness["genetic_pc_eligible"] = genetic_pc_eligibility(
        subject_readiness,
        absolute_z_threshold=pc_threshold,
    ).to_numpy(bool)
    rows = row_keys.merge(
        subject_readiness,
        on="subject_id",
        how="left",
        validate="many_to_one",
    ).merge(
        baseline_dates,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    rows["age_at_imaging"] = (
        pd.to_numeric(rows["AGE"], errors="coerce")
        + (rows["imaging_date"] - rows["cohort_baseline_date"]).dt.days / 365.25
    )
    rows["icv_available"] = ~rows["modality"].eq("smri_adnimerge") | rows[
        "visit_id"
    ].astype(str).isin(icv_visits)
    numeric_covariates = [
        "age_at_imaging",
        *BINARY_COMMON_COVARIATES,
        "PTEDUCAT",
        *[f"PC{index}" for index in range(1, 11)],
    ]
    rows["base_covariates_complete"] = _numeric_complete(rows, numeric_covariates)
    for column in CATEGORICAL_COMMON_COVARIATES:
        rows["base_covariates_complete"] &= rows[column].notna()
    rows["base_covariates_complete"] &= rows["icv_available"]
    rows["base_covariates_complete"] &= rows["genetic_pc_eligible"]

    candidate_rows: list[dict[str, Any]] = []
    for candidate in registry.itertuples(index=False):
        family = rows[rows["readiness_family_id"].eq(candidate.readiness_family_id)]
        exposure_complete = pd.to_numeric(
            family[candidate.score_name], errors="coerce"
        ).notna()
        complete_n = int((family["base_covariates_complete"] & exposure_complete).sum())
        candidate_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "readiness_family_id": candidate.readiness_family_id,
                "complete_case_n": complete_n,
            }
        )
    candidate_audit = registry.merge(
        pd.DataFrame(candidate_rows),
        on=["candidate_id", "readiness_family_id"],
        how="left",
        validate="one_to_one",
    )

    family_frame = registry[
        ["readiness_family_id", "modality", "marker", "marker_spec", "outcome"]
    ].drop_duplicates()
    time_counts = (
        rows.groupby("readiness_family_id")["subject_id"]
        .nunique()
        .rename("time_eligible_n")
    )
    base_counts = (
        rows.loc[rows["base_covariates_complete"]]
        .groupby("readiness_family_id")["subject_id"]
        .nunique()
        .rename("base_covariate_complete_n")
    )
    pc_counts = (
        rows.loc[rows["genetic_pc_eligible"]]
        .groupby("readiness_family_id")["subject_id"]
        .nunique()
        .rename("genetic_pc_eligible_n")
    )
    candidate_ranges = candidate_audit.groupby("readiness_family_id")[
        "complete_case_n"
    ].agg(
        min_complete_case_n="min",
        max_complete_case_n="max",
        pathway_candidate_count="size",
    )
    family_audit = (
        family_frame.join(time_counts, on="readiness_family_id")
        .join(pc_counts, on="readiness_family_id")
        .join(base_counts, on="readiness_family_id")
        .join(candidate_ranges, on="readiness_family_id")
    )
    count_columns = [
        "time_eligible_n",
        "genetic_pc_eligible_n",
        "base_covariate_complete_n",
        "min_complete_case_n",
        "max_complete_case_n",
        "pathway_candidate_count",
    ]
    family_audit[count_columns] = family_audit[count_columns].fillna(0).astype(int)
    threshold = int(protocol["readiness_gate"]["minimum_complete_case_n"])
    family_audit["minimum_required_n"] = threshold
    family_audit["executable"] = family_audit["min_complete_case_n"].ge(threshold)
    family_audit["readiness_status"] = np.where(
        family_audit["executable"],
        "ready",
        "excluded_whole_family_below_n",
    )

    candidate_audit = candidate_audit.merge(
        family_audit[["readiness_family_id", "executable", "readiness_status"]],
        on="readiness_family_id",
        how="left",
        validate="many_to_one",
    )
    executable_registry = candidate_audit[candidate_audit["executable"]].reset_index(
        drop=True
    )
    executable_family_ids = set(
        family_audit.loc[family_audit["executable"], "readiness_family_id"]
    )
    executable_row_keys = row_keys[
        row_keys["readiness_family_id"].isin(executable_family_ids)
        & row_keys["subject_id"].isin(
            set(
                subject_readiness.loc[
                    subject_readiness["genetic_pc_eligible"], "subject_id"
                ]
            )
        )
    ].reset_index(drop=True)

    if not family_audit["pathway_candidate_count"].eq(7).all():
        raise ValueError(
            "Readiness gate must operate on complete seven-pathway families"
        )
    if len(candidate_audit) != 168:
        raise ValueError("Candidate audit must preserve all 168 master candidates")
    for label, frame in (
        ("family audit", family_audit),
        ("candidate audit", candidate_audit),
        ("executable registry", executable_registry),
        ("executable row keys", executable_row_keys),
    ):
        assert_outcome_blind_frame(frame, label=label)
    return (
        family_audit.sort_values(
            ["modality", "marker", "outcome"], kind="stable"
        ).reset_index(drop=True),
        candidate_audit.reset_index(drop=True),
        executable_registry,
        executable_row_keys.loc[:, list(ROW_KEY_COLUMNS)],
    )


def orient_outcome(values: pd.Series, multiplier: float) -> pd.Series:
    """Orient an outcome so larger values consistently indicate worse status."""

    if multiplier not in (-1.0, 1.0):
        raise ValueError("Outcome orientation multiplier must be -1 or 1")
    return pd.to_numeric(values, errors="coerce") * float(multiplier)


def complete_candidate_frame(
    frame: pd.DataFrame,
    *,
    include_icv: bool,
) -> pd.DataFrame:
    """Return the exact complete-case rows required by both component paths."""

    numeric = [
        "exposure",
        "mediator",
        "future_outcome",
        "baseline_outcome",
        "exact_followup_days",
        *CONTINUOUS_COMMON_COVARIATES,
        *BINARY_COMMON_COVARIATES,
    ]
    if include_icv:
        numeric.append("icv")
    missing = set(numeric) - set(frame)
    missing.update(set(CATEGORICAL_COMMON_COVARIATES) - set(frame))
    if missing:
        raise ValueError(f"Candidate model frame is missing columns: {sorted(missing)}")
    mask = _numeric_complete(frame, numeric)
    for column in CATEGORICAL_COMMON_COVARIATES:
        mask &= frame[column].notna()
    return frame.loc[mask].reset_index(drop=True)


def collapse_sparse_sites(
    frame: pd.DataFrame,
    *,
    minimum_site_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> pd.DataFrame:
    """Pool sparse complete-case SITE levels without using observed outcomes."""

    if minimum_site_n < 1:
        raise ValueError("minimum_site_n must be at least one")
    result = frame.copy()
    site = result["SITE"].astype(str)
    counts = site.value_counts(dropna=False)
    sparse = set(counts[counts < minimum_site_n].index)
    result["SITE"] = site.where(~site.isin(sparse), "__POOLED_SPARSE_SITE__")
    return result


def collapse_sparse_categories(
    frame: pd.DataFrame,
    *,
    minimum_category_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> pd.DataFrame:
    """Pool sparse SITE, phase, and batch levels using complete-case counts only."""

    if minimum_category_n < 2:
        raise ValueError("minimum_category_n must be at least two")
    result = frame.copy()
    for column in CATEGORICAL_COMMON_COVARIATES:
        values = result[column].astype(str)
        counts = values.value_counts(dropna=False)
        sparse = set(counts[counts < minimum_category_n].index)
        if not sparse:
            result[column] = values
            continue
        pooled_count = int(values.isin(sparse).sum())
        if pooled_count >= minimum_category_n:
            replacement = f"__POOLED_SPARSE_{column}__"
        else:
            replacement = str(counts.drop(labels=list(sparse)).idxmax())
        result[column] = values.where(~values.isin(sparse), replacement)
    return result


def _zscore(
    values: pd.Series, *, label: str, required: bool = True
) -> np.ndarray | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    scale = float(numeric.std(ddof=0))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        if required:
            raise ValueError(f"{label} has no variance")
        return None
    return (numeric - float(numeric.mean())) / scale


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


def build_component_path_matrices(
    frame: pd.DataFrame,
    *,
    include_icv: bool,
    minimum_category_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Build separate standardized path-a and path-b ANCOVA matrices."""

    complete = collapse_sparse_categories(
        complete_candidate_frame(frame, include_icv=include_icv),
        minimum_category_n=minimum_category_n,
    )
    if len(complete) < 10:
        raise ValueError("Fewer than 10 complete rows are available")
    x = _zscore(complete["exposure"], label="exposure")
    mediator = _zscore(complete["mediator"], label="mediator")
    future = _zscore(complete["future_outcome"], label="future outcome")
    baseline = _zscore(complete["baseline_outcome"], label="baseline outcome")
    followup = _zscore(
        complete["exact_followup_days"],
        label="exact follow-up days",
        required=False,
    )
    assert x is not None and mediator is not None and future is not None
    assert baseline is not None

    common_parts: list[np.ndarray] = [np.ones((len(complete), 1), dtype=float)]
    common_names = ["intercept"]
    for column in CONTINUOUS_COMMON_COVARIATES:
        values = _zscore(complete[column], label=column, required=False)
        if values is not None:
            common_parts.append(values.reshape(-1, 1))
            common_names.append(f"{column}_z")
    if include_icv:
        values = _zscore(complete["icv"], label="icv", required=False)
        if values is not None:
            common_parts.append(values.reshape(-1, 1))
            common_names.append("icv_z")
    for column in BINARY_COMMON_COVARIATES:
        values = pd.to_numeric(complete[column], errors="coerce").to_numpy(float)
        common_parts.append(values.reshape(-1, 1))
        common_names.append(column)
    categorical = pd.get_dummies(
        complete.loc[:, list(CATEGORICAL_COMMON_COVARIATES)].astype(str),
        prefix=list(CATEGORICAL_COMMON_COVARIATES),
        drop_first=True,
        dtype=float,
    )
    if not categorical.empty:
        common_parts.append(categorical.to_numpy(float))
        common_names.extend(map(str, categorical.columns))
    common = np.column_stack(common_parts)
    common_selected = _independent_columns(common)
    common = common[:, common_selected]
    common_names = [common_names[index] for index in common_selected]

    path_a = np.column_stack([common, x])
    path_a_names = [*common_names, "exposure_z"]
    path_b_parts = [
        common,
        mediator.reshape(-1, 1),
        x.reshape(-1, 1),
        baseline.reshape(-1, 1),
    ]
    path_b_names = [*common_names, "mediator_z", "exposure_z", "baseline_outcome_z"]
    if followup is not None:
        path_b_parts.append(followup.reshape(-1, 1))
        path_b_names.append("exact_followup_days_z")
    path_b = np.column_stack(path_b_parts)
    path_b_selected = _independent_columns(path_b)
    path_b = path_b[:, path_b_selected]
    path_b_names = [path_b_names[index] for index in path_b_selected]
    required_b = {"mediator_z", "exposure_z", "baseline_outcome_z"}
    if not required_b.issubset(path_b_names):
        raise ValueError("Required path-b term is not independently estimable")
    if "exposure_z" not in path_a_names:
        raise ValueError("Exposure is not independently estimable in path a")
    return mediator, future, path_a, path_b, path_a_names, path_b_names


def _point_path_fit(
    frame: pd.DataFrame,
    *,
    include_icv: bool,
    robust: bool,
    minimum_category_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> dict[str, Any]:
    mediator, future, path_a, path_b, names_a, names_b = build_component_path_matrices(
        frame,
        include_icv=include_icv,
        minimum_category_n=minimum_category_n,
    )
    base_a = sm.OLS(mediator, path_a).fit()
    base_b = sm.OLS(future, path_b).fit()
    leverage_a = np.asarray(base_a.get_influence().hat_matrix_diag, dtype=float)
    leverage_b = np.asarray(base_b.get_influence().hat_matrix_diag, dtype=float)
    max_leverage_a = float(np.max(leverage_a))
    max_leverage_b = float(np.max(leverage_b))
    if max(max_leverage_a, max_leverage_b) >= 1.0 - 1e-10:
        raise ValueError(
            "HC3 is undefined because a component path has leverage near one"
        )
    if robust:
        fit_a = base_a.get_robustcov_results(cov_type="HC3")
        fit_b = base_b.get_robustcov_results(cov_type="HC3")
        if not np.isfinite(np.concatenate([fit_a.bse, fit_b.bse])).all():
            raise ValueError("HC3 produced a non-finite component-path standard error")
    else:
        fit_a = base_a
        fit_b = base_b
    index_a = names_a.index("exposure_z")
    index_b = names_b.index("mediator_z")
    index_direct = names_b.index("exposure_z")
    return {
        "a": float(fit_a.params[index_a]),
        "a_se": float(fit_a.bse[index_a]),
        "a_p": float(fit_a.pvalues[index_a]),
        "b": float(fit_b.params[index_b]),
        "b_se": float(fit_b.bse[index_b]),
        "b_p": float(fit_b.pvalues[index_b]),
        "direct": float(fit_b.params[index_direct]),
        "direct_se": float(fit_b.bse[index_direct]),
        "direct_p": float(fit_b.pvalues[index_direct]),
        "max_leverage_path_a": max_leverage_a,
        "max_leverage_path_b": max_leverage_b,
        "path_a_design_columns": names_a,
        "path_b_design_columns": names_b,
    }


def _bootstrap_seed(base_seed: int, candidate_id: str) -> int:
    suffix = hashlib.sha256(candidate_id.encode("utf-8")).digest()[:8]
    return (int(base_seed) + int.from_bytes(suffix, "big")) % (2**32)


def _lstsq_path_coefficients(
    frame: pd.DataFrame,
    *,
    include_icv: bool,
    minimum_category_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> tuple[float, float]:
    mediator, future, path_a, path_b, names_a, names_b = build_component_path_matrices(
        frame,
        include_icv=include_icv,
        minimum_category_n=minimum_category_n,
    )
    coefficients_a = np.linalg.lstsq(path_a, mediator, rcond=None)[0]
    coefficients_b = np.linalg.lstsq(path_b, future, rcond=None)[0]
    return (
        float(coefficients_a[names_a.index("exposure_z")]),
        float(coefficients_b[names_b.index("mediator_z")]),
    )


def _prepare_bootstrap_arrays(
    frame: pd.DataFrame,
    *,
    include_icv: bool,
    minimum_category_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> dict[str, np.ndarray | int]:
    """Prepare a low-dimensional representation with SITE fixed effects absorbed.

    Scaling nuisance covariates does not change the exposure or mediator
    coefficients. SITE is the high-dimensional categorical term, so absorbing
    it before each resampled fit preserves the specified fixed-effect column
    space while avoiding repeated 70-plus-column decompositions.
    """

    frame = collapse_sparse_categories(frame, minimum_category_n=minimum_category_n)
    continuous = [*CONTINUOUS_COMMON_COVARIATES]
    if include_icv:
        continuous.append("icv")
    covariate_parts: list[np.ndarray] = []
    for column in continuous:
        values = _zscore(frame[column], label=column, required=False)
        if values is not None:
            covariate_parts.append(values.reshape(-1, 1))
    for column in BINARY_COMMON_COVARIATES:
        covariate_parts.append(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(float).reshape(-1, 1)
        )
    categorical = pd.get_dummies(
        frame.loc[:, ["COLPROT", "batch"]].astype(str),
        prefix=["COLPROT", "batch"],
        drop_first=True,
        dtype=float,
    )
    if not categorical.empty:
        covariate_parts.append(categorical.to_numpy(float))
    covariates = (
        np.column_stack(covariate_parts)
        if covariate_parts
        else np.empty((len(frame), 0), dtype=float)
    )
    primary = frame.loc[
        :,
        [
            "exposure",
            "mediator",
            "future_outcome",
            "baseline_outcome",
            "exact_followup_days",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    site_codes, sites = pd.factorize(frame["SITE"].astype(str), sort=True)
    return {
        "values": np.column_stack([primary.to_numpy(float), covariates]),
        "site_codes": site_codes.astype(np.int64),
        "site_count": int(len(sites)),
        "covariate_start": 5,
    }


def _within_site(
    values: np.ndarray,
    site_codes: np.ndarray,
    site_count: int,
) -> np.ndarray:
    counts = np.bincount(site_codes, minlength=site_count).astype(float)
    means = np.zeros((site_count, values.shape[1]), dtype=float)
    present = counts > 0
    for column in range(values.shape[1]):
        sums = np.bincount(
            site_codes,
            weights=values[:, column],
            minlength=site_count,
        )
        means[present, column] = sums[present] / counts[present]
    return values - means[site_codes]


def _fast_bootstrap_path_coefficients(
    prepared: Mapping[str, np.ndarray | int],
    indices: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(prepared["values"])[indices]
    site_codes = np.asarray(prepared["site_codes"], dtype=np.int64)[indices]
    exposure_scale = float(values[:, 0].std(ddof=0))
    mediator_scale = float(values[:, 1].std(ddof=0))
    future_scale = float(values[:, 2].std(ddof=0))
    within = _within_site(values, site_codes, int(prepared["site_count"]))
    exposure = within[:, 0]
    mediator = within[:, 1]
    future = within[:, 2]
    baseline = within[:, 3]
    followup = within[:, 4]
    covariates = within[:, int(prepared["covariate_start"]) :]

    if min(exposure_scale, mediator_scale, future_scale) <= np.finfo(float).eps:
        raise ValueError("Bootstrap exposure, mediator, or outcome has no variance")
    path_a_matrix = np.column_stack([covariates, exposure])
    path_b_matrix = np.column_stack(
        [covariates, mediator, exposure, baseline, followup]
    )
    path_a_raw = float(np.linalg.lstsq(path_a_matrix, mediator, rcond=None)[0][-1])
    path_b_raw = float(
        np.linalg.lstsq(path_b_matrix, future, rcond=None)[0][covariates.shape[1]]
    )
    return (
        path_a_raw * exposure_scale / mediator_scale,
        path_b_raw * mediator_scale / future_scale,
    )


def fit_candidate_model(
    frame: pd.DataFrame,
    *,
    include_icv: bool,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    candidate_id: str,
    confidence_level: float = 0.95,
    minimum_category_n: int = DEFAULT_MINIMUM_CATEGORY_N,
) -> dict[str, Any]:
    """Fit the corrected ANCOVA indirect-effect model for one candidate."""

    if bootstrap_replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    complete = complete_candidate_frame(frame, include_icv=include_icv)
    point = _point_path_fit(
        complete,
        include_icv=include_icv,
        robust=True,
        minimum_category_n=minimum_category_n,
    )
    indirect = point["a"] * point["b"]
    candidate_bootstrap_seed = _bootstrap_seed(bootstrap_seed, candidate_id)
    rng = np.random.default_rng(candidate_bootstrap_seed)
    bootstrap = np.full(bootstrap_replicates, np.nan, dtype=float)
    prepared = _prepare_bootstrap_arrays(
        complete,
        include_icv=include_icv,
        minimum_category_n=minimum_category_n,
    )
    for replicate in range(bootstrap_replicates):
        indices = rng.integers(0, len(complete), size=len(complete))
        try:
            path_a, path_b = _fast_bootstrap_path_coefficients(prepared, indices)
        except (ValueError, np.linalg.LinAlgError):
            continue
        bootstrap[replicate] = path_a * path_b
    finite = bootstrap[np.isfinite(bootstrap)]
    if len(finite) < max(1, int(0.95 * bootstrap_replicates)):
        raise ValueError(
            f"Only {len(finite)} of {bootstrap_replicates} bootstrap replicates were estimable"
        )
    alpha = 1.0 - confidence_level
    ci_lower, ci_upper = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower_tail = (int(np.count_nonzero(finite <= 0)) + 1) / (len(finite) + 1)
    upper_tail = (int(np.count_nonzero(finite >= 0)) + 1) / (len(finite) + 1)
    bootstrap_p = min(1.0, 2.0 * min(lower_tail, upper_tail))
    sobel_se = float(
        np.sqrt(
            (point["b"] ** 2 * point["a_se"] ** 2)
            + (point["a"] ** 2 * point["b_se"] ** 2)
        )
    )
    sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
    sobel_p = float(2 * stats.norm.sf(abs(sobel_z))) if np.isfinite(sobel_z) else np.nan
    return {
        "n": int(len(complete)),
        "a_path_std": point["a"],
        "a_path_hc3_se": point["a_se"],
        "a_path_hc3_p": point["a_p"],
        "b_path_std": point["b"],
        "b_path_hc3_se": point["b_se"],
        "b_path_hc3_p": point["b_p"],
        "direct_effect_std": point["direct"],
        "direct_effect_hc3_se": point["direct_se"],
        "direct_effect_hc3_p": point["direct_p"],
        "max_leverage_path_a": point["max_leverage_path_a"],
        "max_leverage_path_b": point["max_leverage_path_b"],
        "indirect_effect_std": indirect,
        "indirect_bootstrap_ci_lower": float(ci_lower),
        "indirect_bootstrap_ci_upper": float(ci_upper),
        "indirect_bootstrap_p": float(bootstrap_p),
        "bootstrap_valid_replicates": int(len(finite)),
        "master_seed_count": 1,
        "master_bootstrap_seed": int(bootstrap_seed),
        "candidate_bootstrap_seed": int(candidate_bootstrap_seed),
        "sobel_se_sensitivity": sobel_se,
        "sobel_z_sensitivity": float(sobel_z),
        "sobel_p_sensitivity": sobel_p,
        "path_a_design_columns": point["path_a_design_columns"],
        "path_b_design_columns": point["path_b_design_columns"],
    }


def bh_adjust_fixed(
    p_values: pd.Series | Iterable[float],
    *,
    family_size: int,
) -> np.ndarray:
    """Benjamini-Hochberg adjustment with a prespecified total denominator."""

    numeric = pd.to_numeric(pd.Series(p_values), errors="coerce").to_numpy(float)
    adjusted = np.full(len(numeric), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(numeric))
    if family_size < len(finite_indices):
        raise ValueError("Fixed BH family size is smaller than observed finite tests")
    if not len(finite_indices):
        return adjusted
    observed = numeric[finite_indices]
    if ((observed < 0) | (observed > 1)).any():
        raise ValueError("P values must lie between zero and one")
    order = np.argsort(observed, kind="stable")
    ranked = observed[order]
    raw_q = ranked * float(family_size) / np.arange(1, len(ranked) + 1)
    monotone = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty(len(monotone), dtype=float)
    restored[order] = monotone
    adjusted[finite_indices] = restored
    return adjusted
