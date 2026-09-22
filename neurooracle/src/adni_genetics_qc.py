"""ADNI genetics inventory, sample matching, and conservative PLINK QC."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PTID_PATTERN = re.compile(r"(?<!\d)(\d{3})[_-]?S[_-]?(\d{4,5})(?!\d)", re.IGNORECASE)
ADNI4_REMOTE_PATTERN = re.compile(
    r"(?<![A-Z0-9])ADNI4[-_]?([A-Z0-9]{3})[-_]?([A-Z0-9]{3})(?![A-Z0-9])",
    re.IGNORECASE,
)

OUTCOME_FIELDS = ("CDRSB", "ADAS13", "MMSE", "MOCA", "FAQ")
IMAGING_FIELDS = (
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

BUILD_ANCHORS = {
    "rs3094315": {742429: "GRCh36", 752566: "GRCh37", 817186: "GRCh38"},
    "rs12562034": {758311: "GRCh36", 768448: "GRCh37", 833068: "GRCh38"},
    "rs9442372": {1008567: "GRCh36", 1018704: "GRCh37", 1084846: "GRCh38"},
    "rs429358": {45411941: "GRCh37", 44908684: "GRCh38"},
    "rs7412": {45412079: "GRCh37", 44908822: "GRCh38"},
}


@dataclass(frozen=True)
class BatchSpec:
    """One complete PLINK 1 binary batch."""

    name: str
    prefix: Path
    bed: Path
    bim: Path
    fam: Path


def normalize_adni_subject_id(value: object) -> str:
    """Normalize clinic and ADNI4 remote participant identifiers."""

    text = str(value or "").strip()
    match = PTID_PATTERN.search(text)
    if match:
        return f"{match.group(1)}_S_{match.group(2)}"
    remote_match = ADNI4_REMOTE_PATTERN.search(text)
    if remote_match:
        return f"ADNI4-{remote_match.group(1).upper()}-{remote_match.group(2).upper()}"
    return ""


def classify_batch(path: Path) -> str:
    """Return a stable batch label from the ADNI archive layout."""

    text = str(path).replace("\\", "/").casefold()
    if "adni1_gwas" in text:
        return "ADNI1_Human610_Quad"
    if "adnigo2_gwas" in text and "second_batch" in text:
        return "ADNIGO2_second_batch"
    if "adnigo2_gwas" in text:
        return "ADNIGO2_OmniExpress"
    if "adni3_gwas" in text and "/set2/" in text:
        return "ADNI3_GSA_set2"
    if "adni3_gwas" in text:
        return "ADNI3_GSA_set1"
    if "adni4_gwas" in text:
        return "ADNI4_GSA_v3"
    return path.parent.name


def discover_batches(raw_root: Path) -> list[BatchSpec]:
    """Discover complete BED/BIM/FAM triplets below ``raw_root``."""

    batches: list[BatchSpec] = []
    names: set[str] = set()
    for bed in sorted(raw_root.rglob("*.bed")):
        prefix = bed.with_suffix("")
        bim = prefix.with_suffix(".bim")
        fam = prefix.with_suffix(".fam")
        if not bim.exists() or not fam.exists():
            continue
        name = classify_batch(bed)
        if name in names:
            raise ValueError(f"Duplicate batch label {name}: {bed}")
        names.add(name)
        batches.append(BatchSpec(name=name, prefix=prefix, bed=bed, bim=bim, fam=fam))
    if not batches:
        raise FileNotFoundError(f"No complete PLINK batches found below {raw_root}")
    return batches


def count_lines(path: Path, block_size: int = 1024 * 1024) -> int:
    """Count lines without decoding a potentially large BIM file."""

    count = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            count += block.count(b"\n")
            last = block[-1:]
    return count + int(bool(last and last != b"\n"))


def read_fam_subjects(path: Path) -> list[str]:
    """Read canonical participant IDs from a PLINK FAM file."""

    subjects: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed FAM row at {path}:{line_number}")
            subject = normalize_adni_subject_id(fields[1])
            if not subject:
                raise ValueError(f"Unrecognized ADNI IID {fields[1]!r} at {path}:{line_number}")
            subjects.append(subject)
    return subjects


def detect_genome_build(path: Path) -> tuple[str, str]:
    """Infer the coordinate build from a small set of stable dbSNP anchors."""

    votes: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4 or fields[1] not in BUILD_ANCHORS:
                continue
            try:
                position = int(fields[3])
            except ValueError:
                continue
            build = BUILD_ANCHORS[fields[1]].get(position)
            if build:
                votes[build].append(f"{fields[1]}:{position}")
    if not votes:
        return "unknown", ""
    build, evidence = max(votes.items(), key=lambda item: len(item[1]))
    conflicting = sum(len(items) for key, items in votes.items() if key != build)
    if conflicting:
        return "conflicting", ";".join(
            f"{key}={','.join(items)}" for key, items in sorted(votes.items())
        )
    return build, ",".join(evidence)


def _present(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return text not in {"", "na", "nan", "n/a", "none", "-4"}


def aggregate_adnimerge(path: Path) -> dict[str, dict[str, object]]:
    """Aggregate visit counts and nonmissing endpoint counts by PTID."""

    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PTID" not in reader.fieldnames:
            raise ValueError(f"ADNIMERGE file lacks PTID: {path}")
        for row in reader:
            subject = normalize_adni_subject_id(row.get("PTID"))
            if not subject:
                continue
            record = records.setdefault(
                subject,
                {
                    "visits": 0,
                    "apoe4": "",
                    "diagnosis": "",
                    "outcome_counts": {field: 0 for field in OUTCOME_FIELDS},
                    "imaging_counts": {field: 0 for field in IMAGING_FIELDS},
                },
            )
            record["visits"] = int(record["visits"]) + 1
            if _present(row.get("APOE4")):
                record["apoe4"] = str(row["APOE4"]).strip()
            for diagnosis_field in ("DX", "DX_bl"):
                if _present(row.get(diagnosis_field)):
                    record["diagnosis"] = str(row[diagnosis_field]).strip()
                    break
            for field in OUTCOME_FIELDS:
                if _present(row.get(field)):
                    record["outcome_counts"][field] += 1
            for field in IMAGING_FIELDS:
                if _present(row.get(field)):
                    record["imaging_counts"][field] += 1
    return records


def discover_fmri_subjects(roots: Sequence[Path]) -> set[str]:
    """Collect ADNI participant IDs from one or more FMRIPrep roots."""

    subjects: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("sub-ADNI*"):
            if not path.is_dir():
                continue
            subject = normalize_adni_subject_id(path.name)
            if subject:
                subjects.add(subject)
    return subjects


def overlap_matrix(batch_subjects: dict[str, set[str]]) -> list[dict[str, object]]:
    """Return a long-form symmetric batch overlap matrix."""

    rows: list[dict[str, object]] = []
    for left in sorted(batch_subjects):
        for right in sorted(batch_subjects):
            rows.append(
                {
                    "batch_left": left,
                    "batch_right": right,
                    "shared_subjects": len(batch_subjects[left] & batch_subjects[right]),
                }
            )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sample_audit(
    batches: Sequence[BatchSpec],
    adnimerge_path: Path,
    fmri_roots: Sequence[Path],
    output_dir: Path,
) -> dict[str, object]:
    """Create restricted sample-level audit tables and an aggregate summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_subject_lists = {batch.name: read_fam_subjects(batch.fam) for batch in batches}
    batch_subjects = {name: set(subjects) for name, subjects in batch_subject_lists.items()}
    clinical = aggregate_adnimerge(adnimerge_path)
    fmri_subjects = discover_fmri_subjects(fmri_roots)

    memberships: dict[str, list[str]] = defaultdict(list)
    for batch_name, subjects in batch_subjects.items():
        for subject in subjects:
            memberships[subject].append(batch_name)

    inventory_rows = []
    for batch in batches:
        subjects = batch_subject_lists[batch.name]
        genome_build, build_evidence = detect_genome_build(batch.bim)
        inventory_rows.append(
            {
                "batch": batch.name,
                "prefix": str(batch.prefix),
                "samples": len(subjects),
                "unique_samples": len(set(subjects)),
                "variants": count_lines(batch.bim),
                "genome_build": genome_build,
                "build_evidence": build_evidence,
                "bed_bytes": batch.bed.stat().st_size,
                "bim_bytes": batch.bim.stat().st_size,
                "fam_bytes": batch.fam.stat().st_size,
            }
        )
    _write_csv(
        output_dir / "batch_inventory.csv",
        inventory_rows,
        (
            "batch",
            "prefix",
            "samples",
            "unique_samples",
            "variants",
            "genome_build",
            "build_evidence",
            "bed_bytes",
            "bim_bytes",
            "fam_bytes",
        ),
    )

    membership_rows = [
        {"subject_id": subject, "batch": batch}
        for subject in sorted(memberships)
        for batch in sorted(memberships[subject])
    ]
    _write_csv(output_dir / "sample_membership.csv", membership_rows, ("subject_id", "batch"))
    _write_csv(
        output_dir / "batch_overlap_matrix.csv",
        overlap_matrix(batch_subjects),
        ("batch_left", "batch_right", "shared_subjects"),
    )

    subject_rows: list[dict[str, object]] = []
    eligible_rows: list[dict[str, object]] = []
    for subject in sorted(memberships):
        record = clinical.get(subject)
        outcome_counts = record["outcome_counts"] if record else {}
        imaging_counts = record["imaging_counts"] if record else {}
        visits = int(record["visits"]) if record else 0
        has_apoe = bool(record and _present(record.get("apoe4")))
        outcome_repeated = any(int(outcome_counts.get(field, 0)) >= 2 for field in OUTCOME_FIELDS)
        imaging_available = any(int(imaging_counts.get(field, 0)) >= 1 for field in IMAGING_FIELDS)
        has_fmri = subject in fmri_subjects
        eligible = bool(record and has_fmri and has_apoe and visits >= 2 and outcome_repeated)
        row: dict[str, object] = {
            "subject_id": subject,
            "batches": ";".join(sorted(memberships[subject])),
            "batch_count": len(memberships[subject]),
            "in_adnimerge": bool(record),
            "fmri_available": has_fmri,
            "visits": visits,
            "apoe4_available": has_apoe,
            "diagnosis": str(record.get("diagnosis") or "") if record else "",
            "repeated_outcome_available": outcome_repeated,
            "imaging_marker_available": imaging_available,
            "case2_core_eligible": eligible,
        }
        for field in OUTCOME_FIELDS:
            row[f"{field}_count"] = int(outcome_counts.get(field, 0))
        for field in IMAGING_FIELDS:
            row[f"{field}_count"] = int(imaging_counts.get(field, 0))
        subject_rows.append(row)
        if eligible:
            eligible_rows.append(row)

    subject_fields = (
        "subject_id",
        "batches",
        "batch_count",
        "in_adnimerge",
        "fmri_available",
        "visits",
        "apoe4_available",
        "diagnosis",
        "repeated_outcome_available",
        "imaging_marker_available",
        "case2_core_eligible",
        *(f"{field}_count" for field in OUTCOME_FIELDS),
        *(f"{field}_count" for field in IMAGING_FIELDS),
    )
    _write_csv(output_dir / "subject_summary.csv", subject_rows, subject_fields)
    _write_csv(output_dir / "eligible_case2_samples.csv", eligible_rows, subject_fields)

    all_subjects = set(memberships)
    summary: dict[str, object] = {
        "adnimerge": str(adnimerge_path),
        "fmri_roots": [str(path) for path in fmri_roots],
        "output_dir": str(output_dir),
        "batches": inventory_rows,
        "genotype_rows": sum(len(subjects) for subjects in batch_subject_lists.values()),
        "genotype_unique_subjects": len(all_subjects),
        "subjects_in_multiple_batches": sum(len(names) > 1 for names in memberships.values()),
        "matched_adnimerge": len(all_subjects & set(clinical)),
        "matched_fmri": len(all_subjects & fmri_subjects),
        "matched_genotype_adnimerge_fmri": len(all_subjects & set(clinical) & fmri_subjects),
        "case2_core_eligible": len(eligible_rows),
        "eligible_outcome_counts": {
            field: sum(int(row[f"{field}_count"]) >= 2 for row in eligible_rows)
            for field in OUTCOME_FIELDS
        },
        "eligible_imaging_counts": {
            field: sum(int(row[f"{field}_count"]) >= 1 for row in eligible_rows)
            for field in IMAGING_FIELDS
        },
    }
    (output_dir / "sample_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _run(command: Sequence[str], log_path: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def run_batch_qc(
    batches: Sequence[BatchSpec],
    plink2: Path,
    output_dir: Path,
    mind: float = 0.02,
    geno: float = 0.02,
    maf: float = 0.01,
) -> list[dict[str, object]]:
    """Run report-only QC and conservative per-array filtering."""

    if not plink2.exists():
        raise FileNotFoundError(plink2)
    rows: list[dict[str, object]] = []
    for batch in batches:
        batch_dir = output_dir / batch.name
        batch_dir.mkdir(parents=True, exist_ok=True)
        report_prefix = batch_dir / "raw_report"
        variant_filtered_prefix = batch_dir / "variant_callrate98_maf01"
        filtered_prefix = batch_dir / "filtered_callrate98_maf01"
        _run(
            [
                str(plink2),
                "--bfile",
                str(batch.prefix),
                "--freq",
                "counts",
                "--missing",
                "--het",
                "--out",
                str(report_prefix),
            ],
            batch_dir / "raw_report.command.log",
        )
        _run(
            [
                str(plink2),
                "--bfile",
                str(batch.prefix),
                "--geno",
                str(geno),
                "--maf",
                str(maf),
                "--make-bed",
                "--out",
                str(variant_filtered_prefix),
            ],
            batch_dir / "variant_filter.command.log",
        )
        # Sample missingness must be calculated after bad probes are removed.
        # Otherwise older arrays with many globally missing probes can make every
        # participant fail an otherwise reasonable 98% sample call-rate threshold.
        _run(
            [
                str(plink2),
                "--bfile",
                str(variant_filtered_prefix),
                "--mind",
                str(mind),
                "--make-bed",
                "--out",
                str(filtered_prefix),
            ],
            batch_dir / "sample_filter.command.log",
        )
        filtered_fam = filtered_prefix.with_suffix(".fam")
        filtered_bim = filtered_prefix.with_suffix(".bim")
        rows.append(
            {
                "batch": batch.name,
                "input_samples": count_lines(batch.fam),
                "input_variants": count_lines(batch.bim),
                "filtered_samples": count_lines(filtered_fam),
                "filtered_variants": count_lines(filtered_bim),
                "mind": mind,
                "geno": geno,
                "maf": maf,
                "hwe_filter_applied": False,
                "hwe_status": "deferred_to_ancestry_matched_controls",
                "filtered_prefix": str(filtered_prefix),
            }
        )
    _write_csv(
        output_dir / "qc_summary.csv",
        rows,
        (
            "batch",
            "input_samples",
            "input_variants",
            "filtered_samples",
            "filtered_variants",
            "mind",
            "geno",
            "maf",
            "hwe_filter_applied",
            "hwe_status",
            "filtered_prefix",
        ),
    )
    (output_dir / "qc_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows
