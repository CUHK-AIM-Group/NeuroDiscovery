"""Create final Case Study 2 ADNI pre-imputation sample manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from neurooracle.src.adni_genetics_manifest import (
    as_bool,
    build_preimputation_manifest,
)


ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived"
    r"\qc\case2_adni_genetics_v1"
)
DEFAULT_PREIMPUTATION = ROOT / "preimputation"
DEFAULT_ELIGIBLE = ROOT / "post_qc_audit" / "eligible_case2_samples.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_related_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline().lstrip("#").split()
        rows = [dict(zip(header, line.split())) for line in handle if line.strip()]
    return [(row["IID1"], row["IID2"]) for row in rows]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalize(preimputation: Path, eligible_path: Path) -> dict[str, object]:
    batch_dirs = sorted(
        path
        for path in preimputation.iterdir()
        if path.is_dir() and (path / "sample_qc.summary.json").exists()
    )
    if not batch_dirs:
        raise RuntimeError(f"No completed batches found under {preimputation}")

    qc_rows: list[dict[str, object]] = []
    exact_matches: dict[str, int] = {}
    related_pairs: list[tuple[str, str]] = []
    batch_summaries: dict[str, dict[str, object]] = {}
    for batch_dir in batch_dirs:
        batch = batch_dir.name
        rows = read_csv(batch_dir / "sample_qc.csv")
        for row in rows:
            row["batch"] = batch
        qc_rows.extend(rows)
        match_summary = json.loads(
            (batch_dir / "reference_match_summary.json").read_text(encoding="utf-8")
        )
        sample_summary = json.loads(
            (batch_dir / "sample_qc.summary.json").read_text(encoding="utf-8")
        )
        ancestry_summary = json.loads(
            (batch_dir / "ancestry_assignment.summary.json").read_text(encoding="utf-8")
        )
        exact_matches[batch] = int(match_summary["exact_matches"])
        related_pairs.extend(read_related_pairs(batch_dir / "adni_kinship.kin0"))
        batch_summaries[batch] = {
            "samples": int(sample_summary["samples"]),
            "exact_reference_matches": exact_matches[batch],
            "eur_compatible": int(sample_summary["eur_compatible"]),
            "heterozygosity_outliers": int(sample_summary["heterozygosity_outliers"]),
            "pedigree_sex_missing": int(sample_summary["pedigree_sex_missing"]),
            "sexcheck_problems": int(sample_summary["sexcheck_problems"]),
            "related_pairs": int(sample_summary["related_pairs_kinship_ge_0.0884"]),
            "ancestry_assignment_counts": ancestry_summary["assignment_counts"],
        }

    eligible_rows = read_csv(eligible_path)
    manifest = build_preimputation_manifest(
        qc_rows,
        eligible_rows,
        exact_matches,
        related_pairs,
    )
    manifest_path = preimputation / "preimputation_sample_manifest.csv"
    write_csv(manifest_path, manifest)

    keep_root = preimputation / "case2_eur_keep_by_batch"
    keep_root.mkdir(parents=True, exist_ok=True)
    kept_by_batch: Counter[str] = Counter()
    for batch_dir in batch_dirs:
        batch = batch_dir.name
        kept = [
            row
            for row in manifest
            if row["batch"] == batch and row["final_preimputation_keep"]
        ]
        kept.sort(key=lambda row: str(row["IID"]))
        (keep_root / f"{batch}.keep").write_text(
            "".join(f"{row['FID']}\t{row['IID']}\n" for row in kept),
            encoding="utf-8",
        )
        kept_by_batch[batch] = len(kept)

    base_failures: Counter[str] = Counter()
    for row in manifest:
        for reason in str(row["base_qc_fail_reasons"]).split(";"):
            if reason:
                base_failures[reason] += 1
    summary: dict[str, object] = {
        "completed_batches": len(batch_dirs),
        "genotype_records": len(manifest),
        "unique_genotyped_subjects": len({str(row["subject_id"]) for row in manifest}),
        "case2_core_eligible_subjects": sum(
            as_bool(row.get("case2_core_eligible")) for row in eligible_rows
        ),
        "base_qc_pass_records": sum(bool(row["base_qc_pass"]) for row in manifest),
        "selected_unique_arrays": sum(bool(row["selected_array"]) for row in manifest),
        "duplicate_array_records_removed": sum(
            bool(row["duplicate_array_removed"]) for row in manifest
        ),
        "related_subjects_removed": sum(
            bool(row["relatedness_removed"]) for row in manifest
        ),
        "final_preimputation_subjects": sum(
            bool(row["final_preimputation_keep"]) for row in manifest
        ),
        "final_keep_by_batch": dict(sorted(kept_by_batch.items())),
        "base_qc_failure_counts": dict(sorted(base_failures.items())),
        "batch_qc": batch_summaries,
        "manifest": str(manifest_path),
        "keep_directory": str(keep_root),
        "notes": [
            "Pedigree sex missingness is reported but is not an exclusion.",
            "One array per subject is selected by exact reference matches, then phenotype completeness.",
            "One subject per related component is retained by phenotype completeness, then reference matches.",
            "These lists are pre-imputation inputs; post-imputation variant QC remains required.",
        ],
    }
    summary_path = preimputation / "preimputation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Case Study 2 ADNI pre-imputation QC",
        "",
        f"- Completed array batches: {summary['completed_batches']}",
        f"- Genotype records: {summary['genotype_records']}",
        f"- Unique genotyped subjects: {summary['unique_genotyped_subjects']}",
        f"- Strict CS2 phenotype-eligible subjects: {summary['case2_core_eligible_subjects']}",
        f"- Final unrelated EUR-compatible subjects: {summary['final_preimputation_subjects']}",
        "",
        "## Final keep counts",
        "",
        "| Batch | Subjects |",
        "|---|---:|",
        *[
            f"| {batch} | {count} |"
            for batch, count in sorted(kept_by_batch.items())
        ],
        "",
        "Missing pedigree sex is not an exclusion; it will be supplemented from ADNIMERGE.",
    ]
    (preimputation / "PREIMPUTATION_QC_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preimputation", type=Path, default=DEFAULT_PREIMPUTATION)
    parser.add_argument("--eligible", type=Path, default=DEFAULT_ELIGIBLE)
    args = parser.parse_args()
    summary = finalize(args.preimputation, args.eligible)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
