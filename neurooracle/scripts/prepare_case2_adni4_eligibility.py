"""Build an auditable Case Study 2 eligibility increment for ADNI4.

The script never edits the downloaded ADNI files. It joins the post-call-rate
ADNI4 genotype cohort to current ADNIMERGE2 outcomes and released imaging
derivatives, then writes both an ADNI4-only eligibility table and a combined
table suitable for ``finalize_case2_adni_preimputation.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from neurooracle.src.adni_genetics_qc import normalize_adni_subject_id


DERIVED_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived"
    r"\qc\case2_adni_genetics_v1"
)
DEFAULT_SAMPLE_QC = DERIVED_ROOT / "preimputation" / "ADNI4_GSA_v3" / "sample_qc.csv"
DEFAULT_EXISTING_ELIGIBLE = (
    DERIVED_ROOT / "post_qc_audit" / "eligible_case2_samples.csv"
)
DEFAULT_ADNIMERGE2 = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\raw\phenotype"
    r"\clinical\ADNIMERGE2_package\ADNIMERGE2\data"
)
DEFAULT_OUTPUT = DERIVED_ROOT / "adni4_intake_20260816" / "eligibility"
DEFAULT_FREESURFER = Path(r"D:\ADNI\UCSFFSX7_31Jul2026.csv")
DEFAULT_AMYLOID = Path(
    r"D:\ADNI\PET_Image_Analysis\UCBERKELEY_AMY_6MM_31Jul2026.csv"
)
DEFAULT_TAU = Path(
    r"D:\ADNI\PET_Image_Analysis\UCBERKELEY_TAU_6MM_31Jul2026.csv"
)
DEFAULT_FDG = Path(
    r"D:\ADNI\PET_Image_Analysis\UCBERKELEYFDG_8mm_02_17_23_31Jul2026.csv"
)

OUTCOME_SPECS = {
    "MMSE": ("MMSE.rda", "MMSE", "MMSCORE", ("VISDATE", "VISCODE2", "VISCODE")),
    "MOCA": ("MOCA.rda", "MOCA", "MOCA", ("VISDATE", "VISCODE2", "VISCODE")),
    "FAQ": ("FAQ.rda", "FAQ", "FAQTOTAL", ("VISDATE", "VISCODE2", "VISCODE")),
    "CDRSB": ("CDR.rda", "CDR", "CDRSB", ("VISDATE", "VISCODE2", "VISCODE")),
    "ADAS13": ("ADAS.rda", "ADAS", "TOTAL13", ("VISDATE", "VISCODE2", "VISCODE")),
}

LEGACY_IMAGING_FIELDS = (
    "WholeBrain",
    "Ventricles",
    "Hippocampus",
    "Entorhinal",
    "FDG",
    "AV45",
    "FBB",
    "PIB",
    "ABETA",
    "TAU",
    "PTAU",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_valid_nonnegative(value: object) -> bool:
    """Return true for a finite, non-negative released measurement."""

    if value is None:
        return False
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none"}:
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number) and number >= 0


def canonical_rid(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def visit_key(row: Mapping[str, object], date_fields: Sequence[str], fallback: int) -> str:
    values = [str(row.get(field) or "").strip() for field in date_fields]
    values = [value for value in values if value and value.lower() != "nan"]
    return "|".join(values) if values else f"row:{fallback}"


def measurement_sets(
    rows: Iterable[Mapping[str, object]],
    *,
    eligible_subjects: set[str],
    value_field: str,
    date_fields: Sequence[str],
    subject_field: str = "PTID",
    rid_to_subject: Mapping[str, str] | None = None,
    reject_field: str | None = None,
    reject_values: set[str] | None = None,
) -> dict[str, set[str]]:
    """Collect unique valid visits for each genotyped participant."""

    visits: dict[str, set[str]] = defaultdict(set)
    rejected = {value.lower() for value in (reject_values or set())}
    for index, row in enumerate(rows):
        if subject_field == "RID":
            subject = (rid_to_subject or {}).get(canonical_rid(row.get(subject_field)), "")
        else:
            subject = normalize_adni_subject_id(row.get(subject_field))
        if not subject or subject not in eligible_subjects:
            continue
        if reject_field and str(row.get(reject_field) or "").strip().lower() in rejected:
            continue
        if value_field and not is_valid_nonnegative(row.get(value_field)):
            continue
        visits[subject].add(visit_key(row, date_fields, index))
    return visits


def load_rda_rows(path: Path, object_name: str) -> list[dict[str, object]]:
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise RuntimeError("pyreadr is required to read ADNIMERGE2 .rda files") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        objects = pyreadr.read_r(str(path))
    if object_name not in objects:
        raise KeyError(f"{object_name!r} not found in {path}")
    return objects[object_name].to_dict(orient="records")


def build_rid_map(tables: Iterable[Iterable[Mapping[str, object]]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for rows in tables:
        for row in rows:
            subject = normalize_adni_subject_id(row.get("PTID"))
            rid = canonical_rid(row.get("RID"))
            if subject and rid:
                mapping.setdefault(rid, subject)
    return mapping


def latest_values(
    rows: Iterable[Mapping[str, object]],
    *,
    eligible_subjects: set[str],
    value_field: str,
    date_fields: Sequence[str],
) -> dict[str, str]:
    selected: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        subject = normalize_adni_subject_id(row.get("PTID"))
        if subject not in eligible_subjects:
            continue
        value = str(row.get(value_field) or "").strip()
        if not value or value.lower() == "nan":
            continue
        key = visit_key(row, date_fields, index)
        if subject not in selected or key > selected[subject][0]:
            selected[subject] = (key, value)
    return {subject: value for subject, (_, value) in selected.items()}


def merge_eligibility_rows(
    existing: Sequence[Mapping[str, object]],
    increment: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_subject = {str(row["subject_id"]): dict(row) for row in existing}
    for row in increment:
        by_subject[str(row["subject_id"])] = dict(row)
    return [by_subject[subject] for subject in sorted(by_subject)]


def build_adni4_eligibility(
    *,
    sample_qc: Path,
    adnimerge2: Path,
    freesurfer: Path,
    amyloid: Path,
    tau: Path,
    fdg: Path,
    output_dir: Path,
    existing_eligible: Path | None,
) -> dict[str, object]:
    sample_rows = read_csv(sample_qc)
    subject_ids = {
        normalize_adni_subject_id(row.get("IID"))
        for row in sample_rows
        if normalize_adni_subject_id(row.get("IID"))
    }

    outcome_rows: dict[str, list[dict[str, object]]] = {}
    for outcome, (filename, object_name, _, _) in OUTCOME_SPECS.items():
        outcome_rows[outcome] = load_rda_rows(adnimerge2 / filename, object_name)
    dx_rows = load_rda_rows(adnimerge2 / "DXSUM.rda", "DXSUM")
    apoe_rows = load_rda_rows(adnimerge2 / "APOERES.rda", "APOERES")
    rid_map = build_rid_map([*outcome_rows.values(), dx_rows, apoe_rows])

    outcome_visits: dict[str, dict[str, set[str]]] = {}
    all_clinical_visits: dict[str, set[str]] = defaultdict(set)
    for outcome, (_, _, value_field, date_fields) in OUTCOME_SPECS.items():
        visits = measurement_sets(
            outcome_rows[outcome],
            eligible_subjects=subject_ids,
            value_field=value_field,
            date_fields=date_fields,
        )
        outcome_visits[outcome] = visits
        for subject, keys in visits.items():
            all_clinical_visits[subject].update(keys)

    diagnoses = latest_values(
        dx_rows,
        eligible_subjects=subject_ids,
        value_field="DIAGNOSIS",
        date_fields=("EXAMDATE", "VISCODE2", "VISCODE"),
    )
    apoe_subjects = {
        normalize_adni_subject_id(row.get("PTID"))
        for row in apoe_rows
        if normalize_adni_subject_id(row.get("PTID")) in subject_ids
        and str(row.get("GENOTYPE") or "").strip().lower() not in {"", "nan"}
    }

    smri_rows = read_csv(freesurfer)
    amyloid_rows = read_csv(amyloid)
    tau_rows = read_csv(tau)
    fdg_rows = read_csv(fdg)
    imaging_visits = {
        "sMRI": measurement_sets(
            smri_rows,
            eligible_subjects=subject_ids,
            value_field="",
            date_fields=("EXAMDATE", "VISCODE2", "VISCODE"),
            reject_field="OVERALLQC",
            reject_values={"Fail"},
        ),
        "AmyloidPET": measurement_sets(
            amyloid_rows,
            eligible_subjects=subject_ids,
            value_field="SUMMARY_SUVR",
            date_fields=("SCANDATE", "VISCODE2", "VISCODE"),
        ),
        "TauPET": measurement_sets(
            tau_rows,
            eligible_subjects=subject_ids,
            value_field="META_TEMPORAL_SUVR",
            date_fields=("SCANDATE", "VISCODE2", "VISCODE"),
        ),
        "FDG": measurement_sets(
            fdg_rows,
            eligible_subjects=subject_ids,
            value_field="MEAN",
            date_fields=("EXAMDATE", "VISCODE2", "VISCODE"),
            subject_field="RID",
            rid_to_subject=rid_map,
        ),
    }

    rows: list[dict[str, object]] = []
    for subject in sorted(subject_ids):
        outcome_counts = {
            outcome: len(visits.get(subject, set()))
            for outcome, visits in outcome_visits.items()
        }
        imaging_counts = {
            modality: len(visits.get(subject, set()))
            for modality, visits in imaging_visits.items()
        }
        repeated_outcome = any(count >= 2 for count in outcome_counts.values())
        imaging_available = any(count >= 1 for count in imaging_counts.values())
        row: dict[str, object] = {
            "subject_id": subject,
            "batches": "ADNI4_GSA_v3",
            "batch_count": 1,
            "in_adnimerge": bool(all_clinical_visits.get(subject) or subject in diagnoses),
            "fmri_available": False,
            "visits": len(all_clinical_visits.get(subject, set())),
            "apoe4_available": subject in apoe_subjects,
            "diagnosis": diagnoses.get(subject, ""),
            "repeated_outcome_available": repeated_outcome,
            "imaging_marker_available": imaging_available,
            "case2_core_eligible": repeated_outcome and imaging_available,
        }
        for outcome in OUTCOME_SPECS:
            row[f"{outcome}_count"] = outcome_counts[outcome]
        for field in LEGACY_IMAGING_FIELDS:
            row[f"{field}_count"] = 0
        row["WholeBrain_count"] = imaging_counts["sMRI"]
        row["FDG_count"] = imaging_counts["FDG"]
        for modality, count in imaging_counts.items():
            row[f"{modality}_count"] = count
        rows.append(row)

    eligible_rows = [row for row in rows if row["case2_core_eligible"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = output_dir / "adni4_subject_summary.csv"
    eligible_path = output_dir / "adni4_eligible_case2_samples.csv"
    write_csv(subject_path, rows)
    write_csv(eligible_path, eligible_rows)

    combined_path = output_dir / "combined_eligible_case2_samples.csv"
    existing_rows = read_csv(existing_eligible) if existing_eligible else []
    combined_rows = merge_eligibility_rows(existing_rows, eligible_rows)
    write_csv(combined_path, combined_rows)

    summary: dict[str, object] = {
        "genotyped_post_callrate_qc": len(subject_ids),
        "in_current_adnimerge2": sum(bool(row["in_adnimerge"]) for row in rows),
        "with_repeated_outcome": sum(
            bool(row["repeated_outcome_available"]) for row in rows
        ),
        "with_imaging_marker": sum(bool(row["imaging_marker_available"]) for row in rows),
        "case2_core_eligible": len(eligible_rows),
        "eligible_outcome_counts": {
            outcome: sum(int(row[f"{outcome}_count"]) >= 2 for row in eligible_rows)
            for outcome in OUTCOME_SPECS
        },
        "eligible_imaging_counts": {
            modality: sum(int(row[f"{modality}_count"]) >= 1 for row in eligible_rows)
            for modality in imaging_visits
        },
        "existing_eligible_rows": len(existing_rows),
        "combined_eligible_rows": len(combined_rows),
        "eligibility_rule": (
            "post-call-rate genotype plus >=1 released imaging derivative plus "
            ">=2 valid observations for at least one longitudinal clinical outcome"
        ),
        "apoe_required": False,
        "outputs": {
            "subject_summary": str(subject_path),
            "adni4_eligible": str(eligible_path),
            "combined_eligible": str(combined_path),
        },
    }
    (output_dir / "adni4_eligibility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-qc", type=Path, default=DEFAULT_SAMPLE_QC)
    parser.add_argument("--adnimerge2", type=Path, default=DEFAULT_ADNIMERGE2)
    parser.add_argument("--freesurfer", type=Path, default=DEFAULT_FREESURFER)
    parser.add_argument("--amyloid", type=Path, default=DEFAULT_AMYLOID)
    parser.add_argument("--tau", type=Path, default=DEFAULT_TAU)
    parser.add_argument("--fdg", type=Path, default=DEFAULT_FDG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--existing-eligible", type=Path, default=DEFAULT_EXISTING_ELIGIBLE
    )
    args = parser.parse_args()
    summary = build_adni4_eligibility(
        sample_qc=args.sample_qc,
        adnimerge2=args.adnimerge2,
        freesurfer=args.freesurfer,
        amyloid=args.amyloid,
        tau=args.tau,
        fdg=args.fdg,
        output_dir=args.output_dir,
        existing_eligible=args.existing_eligible,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
