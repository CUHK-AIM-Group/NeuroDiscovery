"""Run local Beagle phasing/imputation for registered Case Study 2 ADNI arrays."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

from neurooracle.scripts.prepare_case2_adni_ancestry import (
    BATCHES,
    PLINK2,
    REFERENCE,
    ROOT,
)


REFERENCE_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
)
DEFAULT_JAVA = Path(
    r"C:\Program Files\MATLAB\R2025a\sys\java\jre\win64\jre\bin\java.exe"
)
DEFAULT_BEAGLE = (
    REFERENCE_ROOT / "tools" / "beagle_5.5_20250227" / "beagle.27Feb25.75f.jar"
)
DEFAULT_MAP_ROOT = REFERENCE_ROOT / "genetic_maps" / "plink_GRCh37"
DEFAULT_BREF3_ROOT = REFERENCE_ROOT / "1000G" / "phase3_GRCh37_bref3"
READY_ROOT = ROOT / "preimputation" / "ready"


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


def region_args(chrom: int, start_bp: int | None, end_bp: int | None) -> list[str]:
    args = ["--chr", str(chrom)]
    if start_bp is not None:
        args.extend(["--from-bp", str(start_bp)])
    if end_bp is not None:
        args.extend(["--to-bp", str(end_bp)])
    return args


def find_map(map_root: Path, chrom: int) -> Path:
    matches = sorted(map_root.rglob(f"plink.chr{chrom}.GRCh37.map"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one GRCh37 map for chromosome {chrom} under {map_root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def beagle_chrom_arg(
    chrom: int,
    start_bp: int | None,
    end_bp: int | None,
) -> str:
    if start_bp is None and end_bp is None:
        return str(chrom)
    return f"{chrom}:{start_bp or ''}-{end_bp or ''}"


def find_bref3(reference_root: Path, chrom: int) -> Path:
    matches = sorted(reference_root.rglob(f"chr{chrom}.1kg.phase3.v5a.b37.bref3"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one chromosome {chrom} bref3 under {reference_root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def export_vcf(
    input_prefix: Path,
    output_prefix: Path,
    chrom: int,
    start_bp: int | None,
    end_bp: int | None,
    phased_reference: bool,
) -> Path:
    output_vcf = output_prefix.with_suffix(".vcf.gz")
    if output_vcf.exists() and output_vcf.stat().st_size > 0:
        return output_vcf
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PLINK2),
        "--pfile",
        str(input_prefix),
    ]
    if phased_reference:
        command.append("vzs")
    command.extend(region_args(chrom, start_bp, end_bp))
    if phased_reference:
        command.extend(
            [
                "--set-missing-var-ids",
                "@:#:$r:$a",
                "--rm-dup",
                "force-first",
            ]
        )
    command.extend(
        [
            "--snps-only",
            "just-acgt",
            "--max-alleles",
            "2",
            "--export",
            "vcf",
            "bgz",
            "--out",
            str(output_prefix),
        ]
    )
    run(command, output_prefix.parent / f"{output_prefix.name}.export.log")
    return output_vcf


def count_vcf(path: Path) -> tuple[int, int]:
    variants = samples = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = max(0, len(line.rstrip().split("\t")) - 9)
            elif not line.startswith("#"):
                variants += 1
    return samples, variants


def beagle_output_complete(vcf: Path, log: Path) -> bool:
    return (
        vcf.exists()
        and vcf.stat().st_size > 1000
        and log.exists()
        and "beagle.27Feb25.75f.jar finished" in log.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )


def merge_chromosome_summaries(
    output_root: Path,
    new_summaries: list[dict[str, object]],
    *,
    chrom: int,
    start_bp: int | None,
    end_bp: int | None,
) -> dict[str, object]:
    """Merge incremental batch runs without dropping prior batch summaries."""

    by_batch: dict[str, dict[str, object]] = {}
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        by_batch.update(
            {
                str(row["batch"]): row
                for row in existing.get("batches", [])
                if isinstance(row, dict) and row.get("batch")
            }
        )
    for batch in BATCHES:
        batch_summary = output_root / batch / f"chr{chrom}" / "summary.json"
        if batch_summary.exists():
            row = json.loads(batch_summary.read_text(encoding="utf-8"))
            if isinstance(row, dict) and row.get("batch"):
                by_batch[str(row["batch"])] = row
    by_batch.update({str(row["batch"]): row for row in new_summaries})
    return {
        "chromosome": chrom,
        "start_bp": start_bp,
        "end_bp": end_bp,
        "batches": [by_batch[batch] for batch in BATCHES if batch in by_batch],
    }


def run_batch(
    batch: str,
    chrom: int,
    reference_panel: Path,
    genetic_map: Path,
    output_root: Path,
    java: Path,
    beagle: Path,
    threads: int,
    memory_gb: int,
    seed: int,
    dr2: float,
    start_bp: int | None,
    end_bp: int | None,
) -> dict[str, object]:
    batch_dir = output_root / batch / f"chr{chrom}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    target_prefix = (
        READY_ROOT / batch / f"{batch}_GRCh37_preimputation"
    )
    target_vcf = export_vcf(
        target_prefix,
        batch_dir / "target",
        chrom,
        start_bp,
        end_bp,
        phased_reference=False,
    )
    imputed_prefix = batch_dir / "beagle_imputed"
    imputed_vcf = imputed_prefix.with_suffix(".vcf.gz")
    beagle_log = imputed_prefix.with_suffix(".log")
    if not beagle_output_complete(imputed_vcf, beagle_log):
        run(
            [
                str(java),
                f"-Xmx{memory_gb}g",
                "-jar",
                str(beagle),
                f"gt={target_vcf}",
                f"ref={reference_panel}",
                f"map={genetic_map}",
                f"out={imputed_prefix}",
                f"chrom={beagle_chrom_arg(chrom, start_bp, end_bp)}",
                f"nthreads={threads}",
                f"seed={seed}",
                "impute=true",
                "gp=false",
                "ap=false",
            ],
            batch_dir / "beagle.command.log",
        )
    postqc_prefix = batch_dir / f"imputed_dr2_{str(dr2).replace('.', 'p')}"
    run(
        [
            str(PLINK2),
            "--vcf",
            str(imputed_vcf),
            "dosage=DS",
            "--extract-if-info",
            f"DR2 >= {dr2}",
            "--snps-only",
            "just-acgt",
            "--max-alleles",
            "2",
            "--threads",
            str(threads),
            "--make-pgen",
            "--sort-vars",
            "--out",
            str(postqc_prefix),
        ],
        batch_dir / "post_imputation_qc.command.log",
    )
    target_samples, target_variants = count_vcf(target_vcf)
    imputed_samples, imputed_variants = count_vcf(imputed_vcf)
    postqc_variants = sum(
        1
        for line in postqc_prefix.with_suffix(".pvar").open(
            "r", encoding="utf-8", errors="replace"
        )
        if line.strip() and not line.startswith("#")
    )
    summary: dict[str, object] = {
        "batch": batch,
        "chromosome": chrom,
        "start_bp": start_bp,
        "end_bp": end_bp,
        "target_samples": target_samples,
        "target_variants": target_variants,
        "imputed_samples": imputed_samples,
        "imputed_variants": imputed_variants,
        "postqc_variants": postqc_variants,
        "dr2_min": dr2,
        "reference_panel": str(reference_panel),
        "genetic_map": str(genetic_map),
        "imputed_vcf": str(imputed_vcf),
        "postqc_prefix": str(postqc_prefix),
    }
    (batch_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom", type=int, choices=range(1, 23), required=True)
    parser.add_argument("--start-bp", type=int)
    parser.add_argument("--end-bp", type=int)
    parser.add_argument("--batch", choices=sorted(BATCHES), action="append")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-gb", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--dr2", type=float, default=0.8)
    parser.add_argument("--java", type=Path, default=DEFAULT_JAVA)
    parser.add_argument("--beagle", type=Path, default=DEFAULT_BEAGLE)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--bref3-root", type=Path, default=DEFAULT_BREF3_ROOT)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    for required in (args.java, args.beagle):
        if not required.exists():
            raise FileNotFoundError(required)
    genetic_map = find_map(args.map_root, args.chrom)
    region_name = (
        f"chr{args.chrom}_{args.start_bp or 'start'}_{args.end_bp or 'end'}"
        if args.start_bp is not None or args.end_bp is not None
        else f"chr{args.chrom}"
    )
    output_root = args.output_root or (
        ROOT / "imputation" / ("smoke" if args.start_bp or args.end_bp else "full")
        / region_name
    )
    try:
        reference_panel = find_bref3(args.bref3_root, args.chrom)
    except FileNotFoundError:
        reference_panel = export_vcf(
            REFERENCE,
            output_root / "reference" / f"1000G_GRCh37_{region_name}",
            args.chrom,
            args.start_bp,
            args.end_bp,
            phased_reference=True,
        )
    batches = args.batch or list(BATCHES)
    summaries = [
        run_batch(
            batch,
            args.chrom,
            reference_panel,
            genetic_map,
            output_root,
            args.java,
            args.beagle,
            args.threads,
            args.memory_gb,
            args.seed,
            args.dr2,
            args.start_bp,
            args.end_bp,
        )
        for batch in batches
    ]
    payload = merge_chromosome_summaries(
        output_root,
        summaries,
        chrom=args.chrom,
        start_bp=args.start_bp,
        end_bp=args.end_bp,
    )
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
