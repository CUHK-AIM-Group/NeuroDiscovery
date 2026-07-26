"""Prepare an ADNI independent-validation manifest for Case Study 1.

The script does not run preprocessing. It scans local ADNI fMRIPrep derivatives
and ADNIMERGE, joins subject/session diagnosis and covariates, and writes a
manifest that downstream validation scripts can consume.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_ADNI_ROOT = Path(r"Z:\Dataset\fMRI\ADNI_fmriprep_all")
DEFAULT_OUT_DIR = Path("outputs/case1_adni_validation")
BOLD_PATTERN = "*_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
CONFOUND_PATTERN = "*_task-rest_desc-confounds_timeseries.tsv"


def parse_bids_name(path: Path) -> dict[str, str] | None:
    match = re.match(r"(?P<sub>sub-[^_]+)_(?P<ses>ses-[^_]+)_task-rest_", path.name)
    if not match:
        return None
    return match.groupdict()


def bids_to_ptid(sub: str) -> str:
    label = sub.removeprefix("sub-")
    if label.startswith("ADNI"):
        label = label[4:]
    if len(label) == 8 and label[3] == "S":
        return f"{label[:3]}_S_{label[-4:]}"
    if len(label) == 9 and label[3] == "_" and label[5] == "_":
        return f"{label[:3]}_S_{label[-4:]}"
    return label


def ses_to_viscode(ses: str) -> str:
    label = ses.removeprefix("ses-")
    if label.upper() in {"M00", "M0"}:
        return "bl"
    match = re.fullmatch(r"M0*(\d+)", label.upper())
    if match:
        month = int(match.group(1))
        return "bl" if month == 0 else f"m{month:02d}"
    return label.lower()


def load_adnimerge(path: Path) -> pd.DataFrame:
    usecols = [
        "RID",
        "PTID",
        "COLPROT",
        "SITE",
        "VISCODE",
        "EXAMDATE",
        "DX_bl",
        "DX",
        "AGE",
        "PTGENDER",
        "PTEDUCAT",
        "APOE4",
        "MMSE",
        "CDRSB",
        "ADAS11",
        "ADAS13",
        "MOCA",
        "ICV",
        "IMAGEUID",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df["PTID"] = df["PTID"].astype(str)
    df["VISCODE"] = df["VISCODE"].astype(str).str.lower()
    return df


def scan_fmriprep(root: Path) -> pd.DataFrame:
    rows = []
    for phase_dir in sorted(p for p in root.glob("ADNI*_fmriprep") if p.is_dir()):
        bold_files = sorted(phase_dir.glob(f"sub-*/{BOLD_PATTERN}"))
        confounds_by_key = {}
        for confound in phase_dir.glob(f"sub-*/{CONFOUND_PATTERN}"):
            parsed = parse_bids_name(confound)
            if parsed:
                confounds_by_key[(parsed["sub"], parsed["ses"])] = confound
        for bold in bold_files:
            parsed = parse_bids_name(bold)
            if not parsed:
                continue
            sub = parsed["sub"]
            ses = parsed["ses"]
            rows.append(
                {
                    "phase": phase_dir.name.replace("_fmriprep", ""),
                    "subject_bids": sub,
                    "session_bids": ses,
                    "PTID": bids_to_ptid(sub),
                    "VISCODE": ses_to_viscode(ses),
                    "bold_path": str(bold),
                    "confounds_path": str(confounds_by_key.get((sub, ses), "")),
                    "has_confounds": (sub, ses) in confounds_by_key,
                }
            )
    return pd.DataFrame(rows)


def normalize_dx(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    upper = text.upper()
    if upper in {"CN", "NL"}:
        return "CN"
    if "MCI" in upper:
        return "MCI"
    if upper in {"AD", "DEMENTIA"} or "DEMENTIA" in upper:
        return "AD"
    return text


def build_manifest(adni_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    adnimerge_path = adni_root / "ADNIMERGE_03Jan2025.csv"
    if not adnimerge_path.exists():
        raise FileNotFoundError(f"Missing ADNIMERGE file: {adnimerge_path}")
    scans = scan_fmriprep(adni_root)
    clinical = load_adnimerge(adnimerge_path)
    merged = scans.merge(clinical, how="left", on=["PTID", "VISCODE"], validate="many_to_one")
    merged["diagnosis"] = merged["DX"].map(normalize_dx)
    missing_dx = merged["diagnosis"].eq("")
    merged.loc[missing_dx, "diagnosis"] = merged.loc[missing_dx, "DX_bl"].map(normalize_dx)
    merged["adni_testable"] = merged["diagnosis"].isin(["CN", "MCI", "AD"]) & merged["has_confounds"]
    merged["contrast_ad_vs_cn"] = merged["diagnosis"].isin(["AD", "CN"])
    merged["contrast_mci_vs_cn"] = merged["diagnosis"].isin(["MCI", "CN"])
    merged["contrast_ad_mci_vs_cn"] = merged["diagnosis"].isin(["AD", "MCI", "CN"])
    testable = merged[merged["adni_testable"]].copy()
    subject_dx = (
        testable.sort_values(["PTID", "VISCODE"])
        .drop_duplicates("PTID", keep="first")
        .groupby("diagnosis")["PTID"]
        .nunique()
        .to_dict()
    )

    summary = {
        "adni_root": str(adni_root),
        "adnimerge": str(adnimerge_path),
        "n_bold_runs": int(len(merged)),
        "n_subjects": int(merged["PTID"].nunique()),
        "n_runs_with_confounds": int(merged["has_confounds"].sum()),
        "n_adni_testable_runs": int(merged["adni_testable"].sum()),
        "diagnosis_counts_by_run": testable["diagnosis"].value_counts().to_dict(),
        "diagnosis_counts_by_subject_first_testable_run": subject_dx,
        "design": {
            "case_study_link": "Independent validation of AD/MCI-testable Case Study 1 disease-region-feature hypotheses.",
            "primary_unit": "disease - brain region - imaging feature",
            "first_pass_modality": "rs-fMRI derivatives already present under ADNI_fmriprep_all",
            "planned_t1_extension": "Add FreeSurfer/DK68 cortical thickness and volume after local T1 download completes.",
            "leakage_control": "Use ADNI-masked NeuroOracle ranking for the primary validation and full ranking as sensitivity analysis.",
        },
    }
    return merged, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adni-root", type=Path, default=DEFAULT_ADNI_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest, summary = build_manifest(args.adni_root)
    manifest.to_csv(args.out_dir / "adni_validation_manifest.csv", index=False)
    (args.out_dir / "adni_validation_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
