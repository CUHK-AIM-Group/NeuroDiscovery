"""Build an experiment-ready ADNI table for Case Study 2.

The output joins post-imputation genetic features to the first testable ADNI
fMRI visit and writes the corresponding ROI-level imaging features as
compressed, atlas-partitioned Parquet files. Controlled ADNI records remain
inside the configured local/NAS output root.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE2_GENETICS_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_POSTIMPUTATION_ROOT = (
    CASE2_GENETICS_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
)
DEFAULT_PHENOTYPE_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "case1_adni_validation"
    / "fmri_multiatlas_full"
    / "adni_fmri_first_testable_runs.csv"
)
DEFAULT_IMAGING_DIR = (
    REPO_ROOT
    / "outputs"
    / "case1_adni_validation"
    / "fmri_multiatlas_full"
    / "atlas_features"
)
DEFAULT_OUTPUT_ROOT = (
    CASE2_GENETICS_ROOT
    / "experiment_tables"
    / "case2_adni_fmri_v1"
)

PTID_PATTERN = re.compile(r"(?:ADNI)?(\d{3})[_-]?S[_-]?(\d{4})", re.IGNORECASE)
DIAGNOSIS_CODES = {"CN": 0, "MCI": 1, "AD": 2}
CLINICAL_COLUMNS = (
    "phase",
    "subject_bids",
    "session_bids",
    "PTID",
    "VISCODE",
    "RID",
    "COLPROT",
    "SITE",
    "EXAMDATE",
    "DX_bl",
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "APOE4",
    "CDRSB",
    "ADAS11",
    "ADAS13",
    "MMSE",
    "MOCA",
    "ICV",
    "DX",
    "diagnosis",
    "adni_testable",
    "session_month",
)
IMAGING_COLUMNS = (
    "PTID",
    "subject_bids",
    "session_bids",
    "VISCODE",
    "diagnosis",
    "AGE",
    "PTGENDER",
    "SITE",
    "mean_fd",
    "n_timepoints",
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
    "value",
)


def normalize_subject_id(value: object) -> str:
    """Normalize ADNI/BIDS/PLINK subject labels to ``000_S_0000``."""

    match = PTID_PATTERN.search(str(value or "").strip())
    if not match:
        return ""
    return f"{match.group(1)}_S_{match.group(2)}"


def _require_unique(df: pd.DataFrame, column: str, label: str) -> None:
    duplicates = df.loc[df[column].duplicated(keep=False), column].dropna().unique()
    if len(duplicates):
        examples = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"{label} has duplicate {column} values: {examples}")


def load_genetics(postimputation_root: Path) -> pd.DataFrame:
    """Load APOE, AD PRS, and ancestry PCs into one subject-level table."""

    apoe_path = postimputation_root / "features" / "apoe_dosage.csv"
    prs_path = (
        postimputation_root
        / "features"
        / "ad_prs_bellenguez"
        / "adni_ad_prs_bellenguez.csv"
    )
    pca_path = (
        postimputation_root
        / "features"
        / "pca"
        / "case2_adni_common_pca20.eigenvec"
    )
    for path in (apoe_path, prs_path, pca_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    apoe = pd.read_csv(apoe_path)
    prs = pd.read_csv(prs_path)
    pca = pd.read_csv(pca_path, sep=r"\s+").rename(columns={"#IID": "iid"})

    apoe["subject_id"] = apoe["subject_id"].map(normalize_subject_id)
    prs["subject_id"] = prs["subject_id"].map(normalize_subject_id)
    pca["subject_id"] = pca["iid"].map(normalize_subject_id)
    for label, frame in (("APOE", apoe), ("PRS", prs), ("PCA", pca)):
        frame.drop(frame.index[frame["subject_id"].eq("")], inplace=True)
        _require_unique(frame, "subject_id", label)

    prs = prs.drop(columns=["iid"], errors="ignore")
    pca = pca.drop(columns=["iid"], errors="ignore")
    genetics = apoe.merge(prs, on="subject_id", how="outer", validate="one_to_one")
    genetics = genetics.merge(pca, on="subject_id", how="outer", validate="one_to_one")
    return genetics.sort_values("subject_id", kind="stable").reset_index(drop=True)


def load_phenotypes(manifest_path: Path) -> pd.DataFrame:
    """Load one imaging-aligned ADNI visit per subject."""

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    missing = [column for column in CLINICAL_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"Phenotype manifest is missing columns: {missing}")

    phenotypes = manifest.loc[:, CLINICAL_COLUMNS].copy()
    phenotypes["subject_id"] = phenotypes["PTID"].map(normalize_subject_id)
    phenotypes = phenotypes[
        phenotypes["subject_id"].ne("")
        & phenotypes["diagnosis"].isin(DIAGNOSIS_CODES)
    ].copy()
    _require_unique(phenotypes, "subject_id", "phenotype manifest")
    return phenotypes


def build_subject_table(
    genetics: pd.DataFrame,
    phenotypes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join genetics and phenotypes and return the cohort plus match audit."""

    genetic_ids = set(genetics["subject_id"])
    phenotype_ids = set(phenotypes["subject_id"])
    all_ids = sorted(genetic_ids | phenotype_ids)
    audit = pd.DataFrame({"subject_id": all_ids})
    audit["has_genetics"] = audit["subject_id"].isin(genetic_ids)
    audit["has_imaging_phenotype"] = audit["subject_id"].isin(phenotype_ids)
    audit["included"] = audit["has_genetics"] & audit["has_imaging_phenotype"]

    cohort = phenotypes.merge(
        genetics,
        on="subject_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_clinical", "_genetic"),
    )
    cohort["diagnosis_code"] = cohort["diagnosis"].map(DIAGNOSIS_CODES)
    cohort["sex_binary"] = cohort["PTGENDER"].map({"Female": 0, "Male": 1})

    clinical_apoe = pd.to_numeric(cohort["APOE4"], errors="coerce")
    genetic_apoe = pd.to_numeric(cohort["apoe_e4_dosage"], errors="coerce")
    complete = cohort["apoe_complete"].fillna(False).astype(bool)
    concordant = clinical_apoe.round().eq(genetic_apoe.round())
    cohort["apoe4_concordant"] = concordant.where(
        complete & clinical_apoe.notna() & genetic_apoe.notna(),
        pd.NA,
    ).astype("boolean")

    preferred = [
        "subject_id",
        "phase",
        "PTID",
        "RID",
        "VISCODE",
        "session_month",
        "EXAMDATE",
        "subject_bids",
        "session_bids",
        "diagnosis",
        "diagnosis_code",
        "DX_bl",
        "DX",
        "AGE",
        "PTGENDER",
        "sex_binary",
        "PTEDUCAT",
        "SITE",
        "COLPROT",
        "CDRSB",
        "ADAS11",
        "ADAS13",
        "MMSE",
        "MOCA",
        "ICV",
        "APOE4",
        "apoe_e4_dosage",
        "apoe_e2_dosage",
        "apoe_genotype",
        "apoe_complete",
        "apoe4_concordant",
        "batch",
    ]
    remaining = [column for column in cohort.columns if column not in preferred]
    cohort = cohort.loc[:, [column for column in preferred if column in cohort] + remaining]
    return cohort.sort_values("subject_id", kind="stable").reset_index(drop=True), audit


def _source_from_path(path: Path) -> str:
    prefix = "adni_"
    suffix = "_run_features"
    stem = path.stem
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def discover_imaging_files(
    imaging_dir: Path,
    sources: Iterable[str] | None = None,
) -> list[Path]:
    """Return deterministic per-atlas feature CSV paths."""

    requested = {
        str(source).removesuffix("_multiatlas")
        for source in (sources or [])
    }
    paths = sorted(imaging_dir.glob("adni_*_run_features.csv"))
    if requested:
        paths = [path for path in paths if _source_from_path(path) in requested]
        missing = requested - {_source_from_path(path) for path in paths}
        if missing:
            raise ValueError(f"Requested imaging sources not found: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No per-atlas ADNI feature files found in {imaging_dir}")
    return paths


def write_imaging_partitions(
    imaging_dir: Path,
    output_dir: Path,
    cohort_ids: set[str],
    *,
    sources: Iterable[str] | None = None,
    chunksize: int = 250_000,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter imaging rows to the cohort and write one Parquet file per atlas."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    subject_qc: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "imaging_sources": set(),
            "mean_fd": [],
            "n_timepoints": [],
        }
    )

    for input_path in discover_imaging_files(imaging_dir, sources):
        source_alias = _source_from_path(input_path)
        output_path = output_dir / f"{source_alias}.parquet"
        temporary_path = output_dir / f"{source_alias}.parquet.tmp"
        if output_path.exists() and not force:
            raise FileExistsError(output_path)
        if temporary_path.exists():
            temporary_path.unlink()

        writer: pq.ParquetWriter | None = None
        row_count = 0
        source_name = ""
        source_subjects: set[str] = set()
        markers: set[str] = set()
        roi_ids: set[str] = set()
        features: set[str] = set()
        try:
            for chunk in pd.read_csv(
                input_path,
                usecols=IMAGING_COLUMNS,
                chunksize=chunksize,
                low_memory=False,
            ):
                chunk["subject_id"] = chunk["PTID"].map(normalize_subject_id)
                chunk = chunk[chunk["subject_id"].isin(cohort_ids)].copy()
                if chunk.empty:
                    continue

                chunk["source"] = chunk["source"].astype(str)
                observed_sources = set(chunk["source"].dropna().unique())
                allowed_sources = {source_alias, f"{source_alias}_multiatlas"}
                if len(observed_sources) != 1 or not observed_sources.issubset(
                    allowed_sources
                ):
                    raise ValueError(
                        f"{input_path} source mismatch: expected one of "
                        f"{sorted(allowed_sources)}, "
                        f"observed {sorted(observed_sources)}"
                    )
                observed_source = next(iter(observed_sources))
                if source_name and observed_source != source_name:
                    raise ValueError(
                        f"{input_path} changes source within the file: "
                        f"{source_name} -> {observed_source}"
                    )
                source_name = observed_source
                chunk["marker_id"] = (
                    chunk["source"].astype(str)
                    + "|"
                    + chunk["roi_index"].astype(str)
                    + "|"
                    + chunk["feature"].astype(str)
                )

                source_subjects.update(chunk["subject_id"].unique())
                markers.update(chunk["marker_id"].unique())
                roi_ids.update(chunk["roi_id"].astype(str).unique())
                features.update(chunk["feature"].astype(str).unique())
                row_count += len(chunk)

                qc = chunk[
                    ["subject_id", "mean_fd", "n_timepoints"]
                ].drop_duplicates("subject_id")
                for row in qc.itertuples(index=False):
                    record = subject_qc[row.subject_id]
                    record["imaging_sources"].add(observed_source)
                    if pd.notna(row.mean_fd):
                        record["mean_fd"].append(float(row.mean_fd))
                    if pd.notna(row.n_timepoints):
                        record["n_timepoints"].append(float(row.n_timepoints))

                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            summaries.append(
                {
                    "source": source_name,
                    "source_alias": source_alias,
                    "input_path": str(input_path),
                    "output_path": "",
                    "rows": 0,
                    "subjects": 0,
                    "rois": 0,
                    "features": 0,
                    "markers": 0,
                }
            )
            continue
        temporary_path.replace(output_path)
        summaries.append(
            {
                "source": source_name,
                "source_alias": source_alias,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "rows": row_count,
                "subjects": len(source_subjects),
                "rois": len(roi_ids),
                "features": len(features),
                "markers": len(markers),
            }
        )

    qc_rows = []
    for subject_id in sorted(cohort_ids):
        record = subject_qc.get(subject_id, {})
        mean_fd_values = record.get("mean_fd", [])
        n_timepoint_values = record.get("n_timepoints", [])
        qc_rows.append(
            {
                "subject_id": subject_id,
                "imaging_source_count": len(record.get("imaging_sources", set())),
                "imaging_mean_fd": (
                    float(pd.Series(mean_fd_values).median())
                    if mean_fd_values
                    else pd.NA
                ),
                "imaging_n_timepoints": (
                    int(round(float(pd.Series(n_timepoint_values).median())))
                    if n_timepoint_values
                    else pd.NA
                ),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(qc_rows)


def _outcome_availability(cohort: pd.DataFrame) -> dict[str, int]:
    outcomes = ("diagnosis", "CDRSB", "ADAS11", "ADAS13", "MMSE", "MOCA")
    return {
        outcome: int(cohort[outcome].notna().sum())
        for outcome in outcomes
        if outcome in cohort.columns
    }


def _write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, compression="zstd")


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root
    existing_files = (
        any(path.is_file() for path in output_root.rglob("*"))
        if output_root.exists()
        else False
    )
    if existing_files and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {output_root}. Use a new version or --force."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    genetics = load_genetics(args.postimputation_root)
    phenotypes = load_phenotypes(args.phenotype_manifest)
    cohort, match_audit = build_subject_table(genetics, phenotypes)

    imaging_summary = pd.DataFrame()
    imaging_qc = pd.DataFrame({"subject_id": cohort["subject_id"]})
    if not args.subject_only:
        imaging_summary, imaging_qc = write_imaging_partitions(
            args.imaging_dir,
            output_root / "imaging_features",
            set(cohort["subject_id"]),
            sources=args.sources,
            chunksize=args.chunksize,
            force=args.force,
        )
        cohort = cohort.merge(
            imaging_qc,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )

    _write_table(
        cohort,
        output_root / "case2_adni_subject_table.csv",
        output_root / "case2_adni_subject_table.parquet",
    )
    match_audit.to_csv(output_root / "case2_adni_subject_match_audit.csv", index=False)
    imaging_summary.to_csv(
        output_root / "case2_adni_imaging_partition_summary.csv",
        index=False,
    )

    apoe_observed = cohort["apoe4_concordant"].dropna()
    hkt = timezone(timedelta(hours=8))
    summary = {
        "generated_at": datetime.now(hkt).isoformat(timespec="seconds"),
        "case_study": "case2_pathway_mediation",
        "controlled_data": True,
        "subjects": int(len(cohort)),
        "diagnosis_counts": {
            str(key): int(value)
            for key, value in cohort["diagnosis"].value_counts().items()
        },
        "genetics_subjects": int(len(genetics)),
        "imaging_phenotype_subjects": int(len(phenotypes)),
        "matched_subjects": int(match_audit["included"].sum()),
        "complete_apoe_subjects": int(
            cohort["apoe_complete"].fillna(False).astype(bool).sum()
        ),
        "apoe4_concordance": {
            "comparable_subjects": int(len(apoe_observed)),
            "concordant_subjects": int(apoe_observed.sum()),
            "fraction": (
                float(apoe_observed.mean()) if len(apoe_observed) else None
            ),
        },
        "outcome_availability": _outcome_availability(cohort),
        "imaging_sources": (
            int((imaging_summary["rows"] > 0).sum())
            if not imaging_summary.empty
            else 0
        ),
        "imaging_rows": (
            int(imaging_summary["rows"].sum())
            if not imaging_summary.empty
            else 0
        ),
        "paths": {
            "postimputation_root": str(args.postimputation_root),
            "phenotype_manifest": str(args.phenotype_manifest),
            "imaging_dir": str(args.imaging_dir),
            "output_root": str(output_root),
            "subject_table_csv": str(
                output_root / "case2_adni_subject_table.csv"
            ),
            "subject_table_parquet": str(
                output_root / "case2_adni_subject_table.parquet"
            ),
            "imaging_feature_dir": str(output_root / "imaging_features"),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postimputation-root",
        type=Path,
        default=DEFAULT_POSTIMPUTATION_ROOT,
    )
    parser.add_argument(
        "--phenotype-manifest",
        type=Path,
        default=DEFAULT_PHENOTYPE_MANIFEST,
    )
    parser.add_argument("--imaging-dir", type=Path, default=DEFAULT_IMAGING_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional exact atlas/source names; default writes every available source.",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--subject-only",
        action="store_true",
        help="Write only the genetics/clinical subject table.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite known output files in an existing output version.",
    )
    return parser.parse_args()


def main() -> int:
    summary = build_dataset(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-07-31 02:44 HKT
