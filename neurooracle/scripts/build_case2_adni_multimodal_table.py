"""Build the controlled multimodal ADNI analysis tables for Case Study 2.

The builder keeps licensed records on the configured NAS root and creates a
lossless set of linked tables instead of one extremely wide Cartesian product:

* subject-level genetic exposures and baseline covariates;
* longitudinal clinical outcomes;
* modality-specific imaging tables and a common visit index;
* same-visit and longitudinal imaging-outcome pair indexes.

The resulting tables support pathway PRS -> imaging marker -> outcome
mediation analyses without silently treating unassessed or failed scans as
fully quality-controlled observations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from neurooracle.scripts.build_case2_adni_experiment_table import (
        DEFAULT_POSTIMPUTATION_ROOT,
        load_genetics,
        normalize_subject_id,
    )
except ModuleNotFoundError:
    from build_case2_adni_experiment_table import (  # type: ignore[no-redef]
        DEFAULT_POSTIMPUTATION_ROOT,
        load_genetics,
        normalize_subject_id,
    )


CONTROLLED_ROOT = Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI")
PHENOTYPE_ROOT = CONTROLLED_ROOT / "raw" / "phenotype"
DEFAULT_ADNIMERGE = (
    PHENOTYPE_ROOT
    / "clinical"
    / "ADNIMERGE_20250103"
    / "ADNIMERGE_03Jan2025.csv"
)
DEFAULT_PACC_DIR = (
    PHENOTYPE_ROOT
    / "clinical"
    / "ADNIMERGE2_package"
    / "ADNIMERGE2"
    / "inst"
    / "extradata"
    / "pacc-raw-input"
)
DEFAULT_PATHWAY_PRS = (
    DEFAULT_POSTIMPUTATION_ROOT
    / "features"
    / "ad_pathway_prs_v1"
    / "case2_adni_pathway_prs_wide.parquet"
)
DEFAULT_FMRI_SUBJECT_TABLE = (
    CONTROLLED_ROOT
    / "derived"
    / "qc"
    / "case2_adni_genetics_v1"
    / "experiment_tables"
    / "case2_adni_fmri_v1"
    / "case2_adni_subject_table.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    CONTROLLED_ROOT
    / "derived"
    / "qc"
    / "case2_adni_genetics_v1"
    / "experiment_tables"
    / "case2_adni_multimodal_v1"
)

DEFAULT_FS_CROSS = (
    PHENOTYPE_ROOT
    / "smri"
    / "cross_sectional_freesurfer7"
    / "UCSFFSX7_31Jul2026.csv"
)
DEFAULT_FS_LONG = (
    PHENOTYPE_ROOT
    / "smri"
    / "longitudinal_freesurfer51"
    / "UCSFFSL51_03_01_22_31Jul2026.csv"
)
DEFAULT_AMYLOID = (
    PHENOTYPE_ROOT
    / "pet"
    / "tables"
    / "UCBERKELEY_AMY_6MM_31Jul2026.csv"
)
DEFAULT_TAU = (
    PHENOTYPE_ROOT
    / "pet"
    / "tables"
    / "UCBERKELEY_TAU_6MM_31Jul2026.csv"
)
DEFAULT_TAU_PVC = (
    PHENOTYPE_ROOT
    / "pet"
    / "tables"
    / "UCBERKELEY_TAUPVC_6MM_31Jul2026.csv"
)
DEFAULT_FDG = (
    PHENOTYPE_ROOT
    / "pet"
    / "tables"
    / "UCBERKELEYFDG_8mm_02_17_23_31Jul2026.csv"
)

CLINICAL_ID_COLUMNS = (
    "RID",
    "COLPROT",
    "ORIGPROT",
    "PTID",
    "SITE",
    "VISCODE",
    "EXAMDATE",
)
BASELINE_COVARIATES = (
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "PTETHCAT",
    "PTRACCAT",
    "PTMARRY",
    "APOE4",
    "DX_bl",
)
CLINICAL_OUTCOMES = (
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
    "DIGITSCOR",
    "TRABSCOR",
    "FAQ",
    "MOCA",
    "mPACCdigit",
    "mPACCtrailsB",
)
ADNIMERGE_SMRI_MARKERS = (
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "ICV",
)
PACC_NAME_MAP = {
    "ADASQ4SCORE": "ADASQ4",
    "MMSE": "MMSE",
    "LDELTOTL": "LDELTOTAL",
    "DIGITSCR": "DIGITSCOR",
    "TRABSCOR": "TRABSCOR",
}
OUTCOME_WORSE_DIRECTION = {
    "CDRSB": 1.0,
    "ADAS11": 1.0,
    "ADAS13": 1.0,
    "ADASQ4": 1.0,
    "MMSE": -1.0,
    "RAVLT_immediate": -1.0,
    "RAVLT_learning": -1.0,
    "RAVLT_forgetting": 1.0,
    "RAVLT_perc_forgetting": 1.0,
    "LDELTOTAL": -1.0,
    "DIGITSCOR": -1.0,
    "TRABSCOR": 1.0,
    "FAQ": 1.0,
    "MOCA": -1.0,
    "mPACCdigit": -1.0,
    "mPACCtrailsB": -1.0,
}
CORE_MARKERS = {
    "amyloid_pet": (
        "CENTILOIDS",
        "SUMMARY_SUVR",
        "AMYLOID_STATUS",
    ),
    "tau_pet": (
        "META_TEMPORAL_SUVR",
        "CTX_ENTORHINAL_SUVR",
    ),
    "tau_pvc_pet": (
        "META_TEMPORAL_SUVR",
        "CTX_ENTORHINAL_SUVR",
    ),
}


def normalize_rid(value: object) -> str:
    """Normalize an ADNI RID without losing integer identity."""

    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def build_rid_subject_map(frames: Iterable[pd.DataFrame]) -> dict[str, str]:
    """Build a validated one-to-one RID -> PTID map."""

    rows = []
    for frame in frames:
        missing = {"RID", "PTID"} - set(frame.columns)
        if missing:
            raise ValueError(f"RID map frame is missing columns: {sorted(missing)}")
        part = frame.loc[:, ["RID", "PTID"]].copy()
        part["rid_key"] = part["RID"].map(normalize_rid)
        part["subject_id"] = part["PTID"].map(normalize_subject_id)
        rows.append(
            part[
                part["rid_key"].ne("")
                & part["subject_id"].ne("")
            ].loc[:, ["rid_key", "subject_id"]]
        )
    mapping = pd.concat(rows, ignore_index=True).drop_duplicates()
    conflicts = mapping.groupby("rid_key")["subject_id"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        examples = ", ".join(conflicts.index[:5])
        raise ValueError(f"RID values map to multiple PTIDs: {examples}")
    return dict(zip(mapping["rid_key"], mapping["subject_id"], strict=False))


def _require_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def load_genetic_exposures(
    postimputation_root: Path,
    pathway_prs_path: Path,
) -> pd.DataFrame:
    """Load global genetic features and pathway PRS into one table."""

    genetics = load_genetics(postimputation_root)
    pathway = pd.read_parquet(pathway_prs_path)
    pathway["subject_id"] = pathway["subject_id"].map(normalize_subject_id)
    pathway = pathway[pathway["subject_id"].ne("")].drop(columns=["iid"], errors="ignore")
    if pathway["subject_id"].duplicated().any():
        raise ValueError("Pathway PRS table contains duplicate subjects")
    result = genetics.merge(
        pathway,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    return result.sort_values("subject_id", kind="stable").reset_index(drop=True)


def load_adnimerge(
    path: Path,
    cohort_ids: set[str],
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load the licensed ADNIMERGE visit table for the genetic cohort."""

    available = set(pd.read_csv(path, nrows=0).columns)
    requested = (
        list(CLINICAL_ID_COLUMNS)
        + list(BASELINE_COVARIATES)
        + list(CLINICAL_OUTCOMES)
        + list(ADNIMERGE_SMRI_MARKERS)
        + [
            "DX",
            "Years_bl",
            "Month_bl",
            "FSVERSION",
            "IMAGEUID",
            "update_stamp",
        ]
    )
    usecols = [column for column in requested if column in available]
    clinical = pd.read_csv(path, usecols=usecols, low_memory=False)
    clinical["subject_id"] = clinical["PTID"].map(normalize_subject_id)
    clinical["RID"] = clinical["RID"].map(normalize_rid)
    clinical["clinical_date"] = pd.to_datetime(
        clinical["EXAMDATE"], errors="coerce"
    )
    clinical = clinical[
        clinical["subject_id"].isin(cohort_ids)
        & clinical["clinical_date"].notna()
        & clinical["clinical_date"].le(cutoff_date)
    ].copy()
    if "Years_bl" in clinical:
        clinical["age_at_visit"] = pd.to_numeric(
            clinical["AGE"], errors="coerce"
        ) + pd.to_numeric(clinical["Years_bl"], errors="coerce")
    else:
        clinical["age_at_visit"] = pd.to_numeric(
            clinical["AGE"], errors="coerce"
        )
    return clinical.sort_values(
        ["subject_id", "clinical_date", "VISCODE"], kind="stable"
    ).reset_index(drop=True)


def build_subject_covariates(clinical: pd.DataFrame) -> pd.DataFrame:
    """Select stable baseline covariates from the earliest visit per subject."""

    columns = [
        "subject_id",
        "RID",
        "SITE",
        "COLPROT",
        "ORIGPROT",
        *BASELINE_COVARIATES,
    ]
    columns = [column for column in columns if column in clinical]
    baseline = (
        clinical.sort_values(["subject_id", "clinical_date"], kind="stable")
        .drop_duplicates("subject_id")
        .loc[:, columns]
        .copy()
    )
    baseline["sex_binary"] = baseline.get("PTGENDER", pd.Series(dtype=object)).map(
        {"Female": 0, "Male": 1}
    )
    return baseline.reset_index(drop=True)


def load_outcomes(
    clinical: pd.DataFrame,
    pacc_dir: Path,
    rid_map: dict[str, str],
    cohort_ids: set[str],
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Combine legacy ADNIMERGE outcomes with newer PACC source tables."""

    id_vars = ["subject_id", "RID", "VISCODE", "clinical_date"]
    available = [column for column in CLINICAL_OUTCOMES if column in clinical]
    legacy = clinical.loc[:, id_vars + available].melt(
        id_vars=id_vars,
        value_vars=available,
        var_name="outcome",
        value_name="value",
    )
    legacy = legacy.rename(columns={"clinical_date": "outcome_date"})
    legacy["source"] = "ADNIMERGE_20250103"
    legacy["source_priority"] = 1

    pacc_parts = []
    for path in sorted(pacc_dir.glob("*.csv")):
        part = pd.read_csv(path, low_memory=False)
        required = {"RID", "VISCODE", "VISDATE", "SCORE_SOURCE", "SCORE"}
        if not required.issubset(part.columns):
            continue
        part["RID"] = part["RID"].map(normalize_rid)
        part["subject_id"] = part["RID"].map(rid_map).fillna("")
        part["outcome_date"] = pd.to_datetime(part["VISDATE"], errors="coerce")
        part["outcome"] = part["SCORE_SOURCE"].map(PACC_NAME_MAP)
        part["value"] = pd.to_numeric(part["SCORE"], errors="coerce")
        part = part[
            part["subject_id"].isin(cohort_ids)
            & part["outcome"].notna()
            & part["outcome_date"].notna()
            & part["outcome_date"].le(cutoff_date)
        ].copy()
        part["source"] = f"ADNIMERGE2_PACC:{path.name}"
        part["source_priority"] = 2
        pacc_parts.append(
            part[
                [
                    "subject_id",
                    "RID",
                    "VISCODE",
                    "outcome_date",
                    "outcome",
                    "value",
                    "source",
                    "source_priority",
                ]
            ]
        )

    outcomes = pd.concat([legacy, *pacc_parts], ignore_index=True)
    outcomes["value"] = pd.to_numeric(outcomes["value"], errors="coerce")
    outcomes = outcomes[outcomes["value"].notna()].copy()
    outcomes["worse_direction"] = outcomes["outcome"].map(
        OUTCOME_WORSE_DIRECTION
    )
    outcomes = outcomes.sort_values(
        [
            "subject_id",
            "outcome",
            "outcome_date",
            "source_priority",
        ],
        kind="stable",
    )
    outcomes = outcomes.drop_duplicates(
        ["subject_id", "outcome", "outcome_date"],
        keep="last",
    )
    return outcomes.drop(columns=["source_priority"]).reset_index(drop=True)


def _attach_identity(
    frame: pd.DataFrame,
    rid_map: dict[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    if "PTID" in result:
        result["subject_id"] = result["PTID"].map(normalize_subject_id)
    elif "RID" in result:
        result["subject_id"] = (
            result["RID"].map(normalize_rid).map(rid_map).fillna("")
        )
    else:
        raise ValueError("Imaging table has neither PTID nor RID")
    if "RID" in result:
        result["RID"] = result["RID"].map(normalize_rid)
    return result


def _visit_id(modality: str, frame: pd.DataFrame) -> pd.Series:
    date_text = frame["imaging_date"].dt.strftime("%Y%m%d").fillna("unknown")
    return modality + "|" + frame["subject_id"].astype(str) + "|" + date_text


def _deduplicate_visits(
    frame: pd.DataFrame,
    rank_columns: list[str],
) -> pd.DataFrame:
    ascending = [True, True] + [False] * len(rank_columns)
    return (
        frame.sort_values(
            ["subject_id", "imaging_date", *rank_columns],
            ascending=ascending,
            kind="stable",
        )
        .drop_duplicates(["subject_id", "imaging_date"])
        .reset_index(drop=True)
    )


def load_freesurfer_table(
    path: Path,
    modality: str,
    rid_map: dict[str, str],
    cohort_ids: set[str],
    cutoff_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and QC a cross-sectional or longitudinal FreeSurfer table."""

    table = _attach_identity(pd.read_csv(path, low_memory=False), rid_map)
    table["imaging_date"] = pd.to_datetime(table["EXAMDATE"], errors="coerce")
    table = table[
        table["subject_id"].isin(cohort_ids)
        & table["imaging_date"].notna()
        & table["imaging_date"].le(cutoff_date)
    ].copy()
    marker_columns = [column for column in table if column.startswith("ST")]
    table["marker_count"] = table[marker_columns].notna().sum(axis=1)
    table["qc_status"] = table["OVERALLQC"].fillna("Not assessed").astype(str)
    table["qc_rank"] = table["qc_status"].map(
        {
            "Pass": 5,
            "Partial": 4,
            "Hippocampus Only": 3,
            "Hippoca": 3,
            "Not assessed": 2,
            "Fail": 0,
        }
    ).fillna(1)
    table["qc_primary_eligible"] = (
        table["qc_status"].eq("Pass") & table["marker_count"].ge(300)
    )
    table["qc_whole_brain_eligible"] = table["qc_primary_eligible"]
    table["qc_relaxed_eligible"] = (
        ~table["qc_status"].eq("Fail") & table["marker_count"].ge(100)
    )
    table = _deduplicate_visits(table, ["qc_rank", "marker_count"])
    table["modality"] = modality
    table["visit_id"] = _visit_id(modality, table)
    index = table[
        [
            "visit_id",
            "subject_id",
            "RID",
            "modality",
            "imaging_date",
            "VISCODE",
            "VISCODE2",
            "qc_status",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
            "marker_count",
        ]
    ].copy()
    return table, index


def load_pet_table(
    path: Path,
    modality: str,
    rid_map: dict[str, str],
    cohort_ids: set[str],
    cutoff_date: pd.Timestamp,
    *,
    qc_lookup: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load an amyloid or tau PET table and create its core-marker rows."""

    table = _attach_identity(pd.read_csv(path, low_memory=False), rid_map)
    table["imaging_date"] = pd.to_datetime(table["SCANDATE"], errors="coerce")
    table = table[
        table["subject_id"].isin(cohort_ids)
        & table["imaging_date"].notna()
        & table["imaging_date"].le(cutoff_date)
    ].copy()
    if "qc_flag" not in table and qc_lookup is not None:
        lookup = qc_lookup[
            ["subject_id", "imaging_date", "qc_flag"]
        ].drop_duplicates(["subject_id", "imaging_date"])
        table = table.merge(
            lookup,
            on=["subject_id", "imaging_date"],
            how="left",
            validate="many_to_one",
        )
    table["qc_flag"] = pd.to_numeric(
        table.get("qc_flag", pd.Series(index=table.index, dtype=float)),
        errors="coerce",
    )
    table["qc_status"] = table["qc_flag"].map(
        {
            2.0: "Pass",
            1.0: "Partial pass",
            0.0: "Fail",
            -1.0: "Not assessed",
            -2.0: "Cannot be processed",
        }
    ).fillna("Not assessed")
    marker_columns = [
        column for column in CORE_MARKERS[modality] if column in table
    ]
    table["marker_count"] = table[marker_columns].notna().sum(axis=1)
    table["qc_primary_eligible"] = table["qc_flag"].ge(1)
    table["qc_whole_brain_eligible"] = table["qc_flag"].eq(2)
    table["qc_relaxed_eligible"] = (
        table["qc_flag"].isin([-1, 1, 2]) & table["marker_count"].gt(0)
    )
    table["qc_rank"] = table["qc_flag"].map(
        {2.0: 5, 1.0: 4, -1.0: 2, 0.0: 1, -2.0: 0}
    ).fillna(2)
    table = _deduplicate_visits(table, ["qc_rank", "marker_count"])
    table["modality"] = modality
    table["visit_id"] = _visit_id(modality, table)

    index_columns = [
        "visit_id",
        "subject_id",
        "RID",
        "modality",
        "imaging_date",
        "VISCODE",
        "VISCODE2",
        "qc_flag",
        "qc_status",
        "qc_primary_eligible",
        "qc_whole_brain_eligible",
        "qc_relaxed_eligible",
        "marker_count",
    ]
    index = table[index_columns].copy()
    marker_id_vars = [
        "visit_id",
        "subject_id",
        "modality",
        "imaging_date",
        "qc_primary_eligible",
        "qc_whole_brain_eligible",
        "qc_relaxed_eligible",
    ]
    markers = table[marker_id_vars + marker_columns].melt(
        id_vars=marker_id_vars,
        value_vars=marker_columns,
        var_name="marker",
        value_name="value",
    )
    markers["value"] = pd.to_numeric(markers["value"], errors="coerce")
    markers = markers[markers["value"].notna()].reset_index(drop=True)
    return table, index, markers


def load_fdg_table(
    path: Path,
    rid_map: dict[str, str],
    cohort_ids: set[str],
    cutoff_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FDG MetaROI data and compute the prescribed reference ratio."""

    raw = _attach_identity(pd.read_csv(path, low_memory=False), rid_map)
    raw["imaging_date"] = pd.to_datetime(raw["EXAMDATE"], errors="coerce")
    raw = raw[
        raw["subject_id"].isin(cohort_ids)
        & raw["imaging_date"].notna()
        & raw["imaging_date"].le(cutoff_date)
    ].copy()
    identifiers = [
        "subject_id",
        "RID",
        "VISCODE",
        "VISCODE2",
        "imaging_date",
    ]
    table = raw.pivot_table(
        index=identifiers,
        columns="ROINAME",
        values="MEAN",
        aggfunc="first",
    ).reset_index()
    table.columns.name = None
    numerator = pd.to_numeric(table.get("MetaROI"), errors="coerce")
    denominator = pd.to_numeric(table.get("Top50PonsVermis"), errors="coerce")
    table["FDG_META_ROI_SUVR"] = numerator / denominator.replace(0, np.nan)
    table["modality"] = "fdg_pet"
    table["qc_status"] = "Not reported"
    table["marker_count"] = table["FDG_META_ROI_SUVR"].notna().astype(int)
    table["qc_primary_eligible"] = table["FDG_META_ROI_SUVR"].notna()
    table["qc_whole_brain_eligible"] = False
    table["qc_relaxed_eligible"] = table["qc_primary_eligible"]
    table["visit_id"] = _visit_id("fdg_pet", table)

    index = table[
        [
            "visit_id",
            "subject_id",
            "RID",
            "modality",
            "imaging_date",
            "VISCODE",
            "VISCODE2",
            "qc_status",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
            "marker_count",
        ]
    ].copy()
    markers = table[
        [
            "visit_id",
            "subject_id",
            "modality",
            "imaging_date",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
            "FDG_META_ROI_SUVR",
        ]
    ].rename(columns={"FDG_META_ROI_SUVR": "value"})
    markers["marker"] = "FDG_META_ROI_SUVR"
    return table, index, markers


def build_adnimerge_smri(
    clinical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create named structural markers from ADNIMERGE's FreeSurfer fields."""

    marker_columns = [
        column for column in ADNIMERGE_SMRI_MARKERS if column in clinical
    ]
    table = clinical[
        [
            "subject_id",
            "RID",
            "VISCODE",
            "clinical_date",
            "FSVERSION",
            "IMAGEUID",
            *marker_columns,
        ]
    ].copy()
    table = table.rename(columns={"clinical_date": "imaging_date"})
    table["marker_count"] = table[marker_columns].notna().sum(axis=1)
    table = table[table["marker_count"].gt(0)].copy()
    table["modality"] = "smri_adnimerge"
    table["qc_status"] = "Legacy derived table"
    table["qc_primary_eligible"] = table["marker_count"].ge(4)
    table["qc_whole_brain_eligible"] = table["qc_primary_eligible"]
    table["qc_relaxed_eligible"] = table["marker_count"].gt(0)
    table["visit_id"] = _visit_id("smri_adnimerge", table)
    table = table.drop_duplicates(["subject_id", "imaging_date"])

    index = table[
        [
            "visit_id",
            "subject_id",
            "RID",
            "modality",
            "imaging_date",
            "VISCODE",
            "qc_status",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
            "marker_count",
        ]
    ].copy()
    markers = table[
        [
            "visit_id",
            "subject_id",
            "modality",
            "imaging_date",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
            *marker_columns,
        ]
    ].melt(
        id_vars=[
            "visit_id",
            "subject_id",
            "modality",
            "imaging_date",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
        ],
        value_vars=marker_columns,
        var_name="marker",
        value_name="value",
    )
    markers["value"] = pd.to_numeric(markers["value"], errors="coerce")
    return index, markers[markers["value"].notna()].reset_index(drop=True)


def load_fmri_visit_index(path: Path) -> pd.DataFrame:
    """Expose the existing multi-atlas fMRI cohort through the common index."""

    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_parquet(path)
    table["imaging_date"] = pd.to_datetime(table["EXAMDATE"], errors="coerce")
    table["modality"] = "fmri_multiatlas"
    table["qc_status"] = "Passed existing fMRI pipeline"
    table["qc_primary_eligible"] = table["imaging_source_count"].gt(0)
    table["qc_whole_brain_eligible"] = table["qc_primary_eligible"]
    table["qc_relaxed_eligible"] = table["qc_primary_eligible"]
    table["marker_count"] = table["imaging_source_count"]
    table["visit_id"] = _visit_id("fmri_multiatlas", table)
    return table[
        [
            "visit_id",
            "subject_id",
            "RID",
            "modality",
            "imaging_date",
            "VISCODE",
            "qc_status",
            "qc_primary_eligible",
            "qc_whole_brain_eligible",
            "qc_relaxed_eligible",
            "marker_count",
        ]
    ].copy()


def build_outcome_pairs(
    visits: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    same_visit_days: int = 180,
    future_min_days: int = 270,
    future_max_days: int = 2190,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match imaging visits to concurrent and future clinical outcomes."""

    pair_columns = [
        "visit_id",
        "subject_id",
        "modality",
        "imaging_date",
        "outcome",
        "outcome_date",
        "outcome_value",
        "days_from_imaging",
        "worse_direction",
    ]
    longitudinal_columns = [
        "visit_id",
        "subject_id",
        "modality",
        "imaging_date",
        "outcome",
        "baseline_outcome_date",
        "baseline_value",
        "future_outcome_date",
        "future_value",
        "followup_days",
        "followup_years",
        "delta_value",
        "decline_score",
        "annualized_decline_score",
        "worse_direction",
    ]
    eligible = visits[visits["qc_primary_eligible"].fillna(False)].copy()
    grouped_outcomes = {
        (subject_id, outcome): group.sort_values("outcome_date", kind="stable")
        for (subject_id, outcome), group in outcomes.groupby(
            ["subject_id", "outcome"], sort=False
        )
    }
    subject_outcomes: dict[str, list[str]] = {}
    for subject_id, outcome in grouped_outcomes:
        subject_outcomes.setdefault(subject_id, []).append(outcome)

    same_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    for visit in eligible.itertuples(index=False):
        for outcome_name in subject_outcomes.get(visit.subject_id, []):
            outcome_group = grouped_outcomes[(visit.subject_id, outcome_name)]
            day_delta = (
                outcome_group["outcome_date"] - visit.imaging_date
            ).dt.days
            near_mask = day_delta.abs().le(same_visit_days)
            if not near_mask.any():
                continue
            near = outcome_group.loc[near_mask].copy()
            near["abs_days"] = (
                near["outcome_date"] - visit.imaging_date
            ).dt.days.abs()
            baseline = near.sort_values(
                ["abs_days", "outcome_date"], kind="stable"
            ).iloc[0]
            baseline_days = int(
                (baseline["outcome_date"] - visit.imaging_date).days
            )
            same_rows.append(
                {
                    "visit_id": visit.visit_id,
                    "subject_id": visit.subject_id,
                    "modality": visit.modality,
                    "imaging_date": visit.imaging_date,
                    "outcome": outcome_name,
                    "outcome_date": baseline["outcome_date"],
                    "outcome_value": float(baseline["value"]),
                    "days_from_imaging": baseline_days,
                    "worse_direction": float(baseline["worse_direction"]),
                }
            )

            future_mask = day_delta.between(future_min_days, future_max_days)
            for future in outcome_group.loc[future_mask].itertuples(index=False):
                followup_days = int(
                    (future.outcome_date - visit.imaging_date).days
                )
                delta_value = float(future.value) - float(baseline["value"])
                direction = float(baseline["worse_direction"])
                decline_score = delta_value * direction
                years = followup_days / 365.25
                future_rows.append(
                    {
                        "visit_id": visit.visit_id,
                        "subject_id": visit.subject_id,
                        "modality": visit.modality,
                        "imaging_date": visit.imaging_date,
                        "outcome": outcome_name,
                        "baseline_outcome_date": baseline["outcome_date"],
                        "baseline_value": float(baseline["value"]),
                        "future_outcome_date": future.outcome_date,
                        "future_value": float(future.value),
                        "followup_days": followup_days,
                        "followup_years": years,
                        "delta_value": delta_value,
                        "decline_score": decline_score,
                        "annualized_decline_score": decline_score / years,
                        "worse_direction": direction,
                    }
                )
    return (
        pd.DataFrame(same_rows, columns=pair_columns),
        pd.DataFrame(future_rows, columns=longitudinal_columns),
    )


def summarize_coverage(
    visits: pd.DataFrame,
    same_pairs: pd.DataFrame,
    longitudinal_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize subject and pair availability by modality and outcome."""

    rows = []
    modalities = sorted(visits["modality"].dropna().unique())
    outcomes = sorted(same_pairs["outcome"].dropna().unique())
    for modality in modalities:
        modality_visits = visits[visits["modality"].eq(modality)]
        for outcome in outcomes:
            same = same_pairs[
                same_pairs["modality"].eq(modality)
                & same_pairs["outcome"].eq(outcome)
            ]
            longitudinal = longitudinal_pairs[
                longitudinal_pairs["modality"].eq(modality)
                & longitudinal_pairs["outcome"].eq(outcome)
            ]
            rows.append(
                {
                    "modality": modality,
                    "outcome": outcome,
                    "primary_qc_visits": int(
                        modality_visits["qc_primary_eligible"].sum()
                    ),
                    "primary_qc_subjects": int(
                        modality_visits.loc[
                            modality_visits["qc_primary_eligible"],
                            "subject_id",
                        ].nunique()
                    ),
                    "same_visit_pairs": int(len(same)),
                    "same_visit_subjects": int(same["subject_id"].nunique()),
                    "longitudinal_pairs": int(len(longitudinal)),
                    "longitudinal_subjects": int(
                        longitudinal["subject_id"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed identifier/object columns before Arrow serialization."""

    result = frame.copy()
    identifier_columns = {
        "RID",
        "PTID",
        "subject_id",
        "visit_id",
        "VISCODE",
        "VISCODE2",
        "IMAGEUID",
    }
    for column in result.columns:
        series = result[column]
        if column in identifier_columns:
            if column == "RID":
                series = series.map(normalize_rid).replace("", pd.NA)
            else:
                series = series.map(
                    lambda value: pd.NA if pd.isna(value) else str(value).strip()
                )
            result[column] = series.astype("string")
            continue
        if not pd.api.types.is_object_dtype(series.dtype):
            continue
        observed_types = series.dropna().map(type).unique()
        if len(observed_types) > 1:
            result[column] = series.map(
                lambda value: pd.NA if pd.isna(value) else str(value)
            ).astype("string")
    return result


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe_frame(frame).to_parquet(
        path,
        index=False,
        compression="zstd",
    )


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    """Build every linked Case Study 2 multimodal table."""

    source_paths = [
        args.adnimerge,
        args.pathway_prs,
        args.fs_cross,
        args.fs_long,
        args.amyloid,
        args.tau,
        args.tau_pvc,
        args.fdg,
    ]
    _require_files(source_paths)
    if not args.pacc_dir.is_dir():
        raise FileNotFoundError(args.pacc_dir)
    if args.output_root.exists() and any(args.output_root.rglob("*")) and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {args.output_root}. Use --force to overwrite."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    cutoff_date = pd.Timestamp(args.cutoff_date)
    genetics = load_genetic_exposures(
        args.postimputation_root,
        args.pathway_prs,
    )
    cohort_ids = set(genetics["subject_id"])

    map_sources = []
    for path in (args.fs_cross, args.amyloid, args.tau, args.tau_pvc):
        columns = pd.read_csv(path, nrows=0).columns
        if {"RID", "PTID"}.issubset(columns):
            map_sources.append(
                pd.read_csv(path, usecols=["RID", "PTID"], dtype=str)
            )
    rid_map = build_rid_subject_map(map_sources)

    clinical = load_adnimerge(args.adnimerge, cohort_ids, cutoff_date)
    covariates = build_subject_covariates(clinical)
    subject_table = genetics.merge(
        covariates,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    outcomes = load_outcomes(
        clinical,
        args.pacc_dir,
        rid_map,
        cohort_ids,
        cutoff_date,
    )

    imaging_tables: dict[str, pd.DataFrame] = {}
    indexes = []
    marker_tables = []

    fs_cross, fs_cross_index = load_freesurfer_table(
        args.fs_cross,
        "smri_freesurfer7_cross_sectional",
        rid_map,
        cohort_ids,
        cutoff_date,
    )
    fs_long, fs_long_index = load_freesurfer_table(
        args.fs_long,
        "smri_freesurfer51_longitudinal",
        rid_map,
        cohort_ids,
        cutoff_date,
    )
    imaging_tables["smri_freesurfer7_cross_sectional"] = fs_cross
    imaging_tables["smri_freesurfer51_longitudinal"] = fs_long
    indexes.extend([fs_cross_index, fs_long_index])

    amyloid, amyloid_index, amyloid_markers = load_pet_table(
        args.amyloid,
        "amyloid_pet",
        rid_map,
        cohort_ids,
        cutoff_date,
    )
    tau, tau_index, tau_markers = load_pet_table(
        args.tau,
        "tau_pet",
        rid_map,
        cohort_ids,
        cutoff_date,
    )
    tau_pvc, tau_pvc_index, tau_pvc_markers = load_pet_table(
        args.tau_pvc,
        "tau_pvc_pet",
        rid_map,
        cohort_ids,
        cutoff_date,
        qc_lookup=tau,
    )
    fdg, fdg_index, fdg_markers = load_fdg_table(
        args.fdg,
        rid_map,
        cohort_ids,
        cutoff_date,
    )
    imaging_tables.update(
        {
            "amyloid_pet": amyloid,
            "tau_pet": tau,
            "tau_pvc_pet": tau_pvc,
            "fdg_pet": fdg,
        }
    )
    indexes.extend([amyloid_index, tau_index, tau_pvc_index, fdg_index])
    marker_tables.extend(
        [amyloid_markers, tau_markers, tau_pvc_markers, fdg_markers]
    )

    smri_index, smri_markers = build_adnimerge_smri(clinical)
    indexes.append(smri_index)
    marker_tables.append(smri_markers)
    fmri_index = load_fmri_visit_index(args.fmri_subject_table)
    if not fmri_index.empty:
        indexes.append(fmri_index)

    visit_index = pd.concat(indexes, ignore_index=True, sort=False)
    visit_index = visit_index.sort_values(
        ["subject_id", "imaging_date", "modality"], kind="stable"
    ).reset_index(drop=True)
    imaging_markers = pd.concat(marker_tables, ignore_index=True, sort=False)
    imaging_markers = imaging_markers.sort_values(
        ["subject_id", "imaging_date", "modality", "marker"],
        kind="stable",
    ).reset_index(drop=True)
    same_pairs, longitudinal_pairs = build_outcome_pairs(
        visit_index,
        outcomes,
        same_visit_days=args.same_visit_days,
        future_min_days=args.future_min_days,
        future_max_days=args.future_max_days,
    )
    coverage = summarize_coverage(visit_index, same_pairs, longitudinal_pairs)

    _write_parquet(
        subject_table,
        args.output_root / "case2_adni_subject_genetics_covariates.parquet",
    )
    _write_parquet(
        clinical,
        args.output_root / "case2_adni_clinical_visits.parquet",
    )
    _write_parquet(
        outcomes,
        args.output_root / "case2_adni_outcomes_long.parquet",
    )
    _write_parquet(
        visit_index,
        args.output_root / "case2_adni_imaging_visit_index.parquet",
    )
    _write_parquet(
        imaging_markers,
        args.output_root / "case2_adni_primary_imaging_markers_long.parquet",
    )
    _write_parquet(
        same_pairs,
        args.output_root / "case2_adni_same_visit_outcome_pairs.parquet",
    )
    _write_parquet(
        longitudinal_pairs,
        args.output_root / "case2_adni_longitudinal_outcome_pairs.parquet",
    )
    for modality, table in imaging_tables.items():
        _write_parquet(
            table,
            args.output_root / "imaging_features" / f"{modality}.parquet",
        )
    coverage.to_csv(
        args.output_root / "case2_adni_modality_outcome_coverage.csv",
        index=False,
    )

    modality_summary = []
    for modality, group in visit_index.groupby("modality", sort=True):
        modality_summary.append(
            {
                "modality": modality,
                "visits": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                "primary_qc_visits": int(group["qc_primary_eligible"].sum()),
                "primary_qc_subjects": int(
                    group.loc[
                        group["qc_primary_eligible"], "subject_id"
                    ].nunique()
                ),
                "relaxed_qc_visits": int(group["qc_relaxed_eligible"].sum()),
                "relaxed_qc_subjects": int(
                    group.loc[
                        group["qc_relaxed_eligible"], "subject_id"
                    ].nunique()
                ),
            }
        )
    pd.DataFrame(modality_summary).to_csv(
        args.output_root / "case2_adni_modality_summary.csv",
        index=False,
    )

    hkt = timezone(timedelta(hours=8))
    manifest = {
        "generated_at": datetime.now(hkt).isoformat(timespec="seconds"),
        "case_study": "case2_pathway_mediation",
        "controlled_data": True,
        "cutoff_date": str(cutoff_date.date()),
        "genetic_subjects": int(len(genetics)),
        "subjects_with_baseline_covariates": int(
            subject_table["RID"].notna().sum()
        ),
        "pathway_prs_features": int(
            sum(column.startswith("pathway_prs__") for column in genetics)
        ),
        "outcome_rows": int(len(outcomes)),
        "outcome_subjects": int(outcomes["subject_id"].nunique()),
        "imaging_visits": int(len(visit_index)),
        "imaging_subjects": int(visit_index["subject_id"].nunique()),
        "same_visit_pairs": int(len(same_pairs)),
        "same_visit_subjects": int(same_pairs["subject_id"].nunique()),
        "longitudinal_pairs": int(len(longitudinal_pairs)),
        "longitudinal_subjects": int(
            longitudinal_pairs["subject_id"].nunique()
        ),
        "modalities": modality_summary,
        "limitations": [
            "FreeSurfer 7 ST-code anatomical labels remain pending the current ADNI data dictionary.",
            "ADNIMERGE demographics and diagnosis are the 03-Jan-2025 snapshot; newer ADNIMERGE2 PACC outcomes are merged through the configured cutoff.",
            "FDG has no row-level QC flag in the downloaded summary table.",
        ],
        "paths": {
            "output_root": str(args.output_root),
            "adnimerge": str(args.adnimerge),
            "pacc_dir": str(args.pacc_dir),
            "pathway_prs": str(args.pathway_prs),
            "fmri_subject_table": str(args.fmri_subject_table),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postimputation-root",
        type=Path,
        default=DEFAULT_POSTIMPUTATION_ROOT,
    )
    parser.add_argument("--pathway-prs", type=Path, default=DEFAULT_PATHWAY_PRS)
    parser.add_argument("--adnimerge", type=Path, default=DEFAULT_ADNIMERGE)
    parser.add_argument("--pacc-dir", type=Path, default=DEFAULT_PACC_DIR)
    parser.add_argument(
        "--fmri-subject-table",
        type=Path,
        default=DEFAULT_FMRI_SUBJECT_TABLE,
    )
    parser.add_argument("--fs-cross", type=Path, default=DEFAULT_FS_CROSS)
    parser.add_argument("--fs-long", type=Path, default=DEFAULT_FS_LONG)
    parser.add_argument("--amyloid", type=Path, default=DEFAULT_AMYLOID)
    parser.add_argument("--tau", type=Path, default=DEFAULT_TAU)
    parser.add_argument("--tau-pvc", type=Path, default=DEFAULT_TAU_PVC)
    parser.add_argument("--fdg", type=Path, default=DEFAULT_FDG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cutoff-date", default="2026-07-31")
    parser.add_argument("--same-visit-days", type=int, default=180)
    parser.add_argument("--future-min-days", type=int, default=270)
    parser.add_argument("--future-max-days", type=int, default=2190)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = build_dataset(parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Last Updated At: 2026-07-31 17:52 HKT
