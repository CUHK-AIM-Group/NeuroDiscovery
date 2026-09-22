"""Harmonize each ADNI array with 1000 Genomes and run ancestry/sample QC."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from neurooracle.src.adni_genetics_harmonization import (
    assign_ancestry,
    build_reference_match_set,
    prepare_autosomal_candidates,
    summarize_sample_qc,
)


ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived"
    r"\qc\case2_adni_genetics_v1"
)
REFERENCE = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
    r"\1000G\phase3_GRCh37\all_phase3_ns"
)
REFERENCE_PSAM = REFERENCE.with_suffix(".psam")
PLINK2 = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
    r"\tools\plink2_20260504\plink2.exe"
)
PLINK1 = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
    r"\tools\plink1_20250819\plink.exe"
)
BATCHES = {
    "ADNI1_Human610_Quad": ROOT / "harmonized" / "ADNI1_GRCh37" / "ADNI1_GRCh37",
    "ADNI3_GSA_set1": ROOT / "batch_qc" / "ADNI3_GSA_set1" / "filtered_callrate98_maf01",
    "ADNI3_GSA_set2": ROOT / "batch_qc" / "ADNI3_GSA_set2" / "filtered_callrate98_maf01",
    "ADNI4_GSA_v3": ROOT / "batch_qc" / "ADNI4_GSA_v3" / "filtered_callrate98_maf01",
    "ADNIGO2_OmniExpress": ROOT / "batch_qc" / "ADNIGO2_OmniExpress" / "filtered_callrate98_maf01",
    "ADNIGO2_second_batch": ROOT / "batch_qc" / "ADNIGO2_second_batch" / "filtered_callrate98_maf01",
}


def run(command: list[str], log_path: Path) -> None:
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


def process_batch(name: str, input_prefix: Path, output_root: Path) -> dict[str, object]:
    batch_dir = output_root / name
    batch_dir.mkdir(parents=True, exist_ok=True)
    completed_files = {
        "candidate_summary": batch_dir / "candidate_summary.json",
        "reference_match_summary": batch_dir / "reference_match_summary.json",
        "ancestry_summary": batch_dir / "ancestry_assignment.summary.json",
        "sample_qc_summary": batch_dir / "sample_qc.summary.json",
    }
    if all(path.exists() for path in completed_files.values()):
        return {
            "batch": name,
            "input_prefix": str(input_prefix),
            **{
                key: json.loads(path.read_text(encoding="utf-8"))
                for key, path in completed_files.items()
            },
        }
    candidate_summary = prepare_autosomal_candidates(
        input_prefix.with_suffix(".bim"),
        batch_dir,
    )
    candidate_ids = Path(str(candidate_summary["candidate_ids"]))
    reference_candidate = batch_dir / "reference_candidate"
    run(
        [
            str(PLINK2),
            "--pfile",
            str(REFERENCE),
            "vzs",
            "--extract",
            str(candidate_ids),
            "--rm-dup",
            "force-first",
            "--snps-only",
            "just-acgt",
            "--max-alleles",
            "2",
            "--make-pgen",
            "--sort-vars",
            "--out",
            str(reference_candidate),
        ],
        batch_dir / "reference_candidate.command.log",
    )
    match_summary = build_reference_match_set(
        input_prefix.with_suffix(".bim"),
        reference_candidate.with_suffix(".pvar"),
        batch_dir,
    )
    matched_ids = Path(str(match_summary["matched_ids"]))
    adni_matched = batch_dir / "adni_matched"
    reference_matched = batch_dir / "reference_matched"
    adni_bed = batch_dir / "adni_matched_bed"
    reference_bed = batch_dir / "reference_matched_bed"
    merged = batch_dir / "merged_1000G_ADNI_bed"
    run(
        [
            str(PLINK2),
            "--bfile",
            str(input_prefix),
            "--extract",
            str(matched_ids),
            "--rm-dup",
            "force-first",
            "--make-pgen",
            "--sort-vars",
            "--out",
            str(adni_matched),
        ],
        batch_dir / "adni_matched.command.log",
    )
    run(
        [
            str(PLINK2),
            "--pfile",
            str(reference_candidate),
            "--extract",
            str(matched_ids),
            "--make-pgen",
            "--sort-vars",
            "--out",
            str(reference_matched),
        ],
        batch_dir / "reference_matched.command.log",
    )
    for source, destination, label in (
        (adni_matched, adni_bed, "adni_to_bed"),
        (reference_matched, reference_bed, "reference_to_bed"),
    ):
        run(
            [
                str(PLINK2),
                "--pfile",
                str(source),
                "--make-bed",
                "--out",
                str(destination),
            ],
            batch_dir / f"{label}.command.log",
        )
    run(
        [
            str(PLINK1),
            "--bfile",
            str(reference_bed),
            "--bmerge",
            str(adni_bed),
            "--make-bed",
            "--out",
            str(merged),
        ],
        batch_dir / "merge.command.log",
    )
    prune = batch_dir / "reference_ld"
    run(
        [
            str(PLINK2),
            "--pfile",
            str(reference_matched),
            "--keep-founders",
            "--maf",
            "0.05",
            "--geno",
            "0.02",
            "--indep-pairwise",
            "200",
            "50",
            "0.2",
            "--out",
            str(prune),
        ],
        batch_dir / "ld_prune.command.log",
    )
    pca = batch_dir / "ancestry_pca"
    run(
        [
            str(PLINK2),
            "--bfile",
            str(merged),
            "--extract",
            str(prune.with_suffix(".prune.in")),
            "--keep-founders",
            "--pca",
            "20",
            "--out",
            str(pca),
        ],
        batch_dir / "pca.command.log",
    )
    ancestry_summary = assign_ancestry(
        pca.with_suffix(".eigenvec"),
        REFERENCE_PSAM,
        batch_dir / "ancestry_assignment.csv",
    )
    het = batch_dir / "adni_het"
    kinship = batch_dir / "adni_kinship"
    run(
        [
            str(PLINK2),
            "--pfile",
            str(adni_matched),
            "--extract",
            str(prune.with_suffix(".prune.in")),
            "--het",
            "--out",
            str(het),
        ],
        batch_dir / "het.command.log",
    )
    run(
        [
            str(PLINK2),
            "--pfile",
            str(adni_matched),
            "--extract",
            str(prune.with_suffix(".prune.in")),
            "--make-king-table",
            "--king-table-filter",
            "0.0884",
            "--out",
            str(kinship),
        ],
        batch_dir / "kinship.command.log",
    )
    sexcheck = batch_dir / "sexcheck"
    run(
        [
            str(PLINK2),
            "--bfile",
            str(input_prefix),
            "--split-par",
            "hg19",
            "--check-sex",
            "max-female-xf=0.2",
            "min-male-xf=0.8",
            "--out",
            str(sexcheck),
        ],
        batch_dir / "sexcheck.command.log",
    )
    sample_qc_summary = summarize_sample_qc(
        batch_dir / "ancestry_assignment.csv",
        het.with_suffix(".het"),
        sexcheck.with_suffix(".sexcheck"),
        kinship.with_suffix(".kin0"),
        batch_dir / "sample_qc.csv",
    )
    return {
        "batch": name,
        "input_prefix": str(input_prefix),
        "candidate_summary": candidate_summary,
        "reference_match_summary": match_summary,
        "ancestry_summary": ancestry_summary,
        "sample_qc_summary": sample_qc_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=sorted(BATCHES), action="append")
    parser.add_argument("--output-root", type=Path, default=ROOT / "preimputation")
    args = parser.parse_args()
    selected = args.batch or list(BATCHES)
    summaries = [
        process_batch(name, BATCHES[name], args.output_root)
        for name in selected
    ]
    summary = {"batches": summaries}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
