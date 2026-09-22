"""Build per-chromosome Beagle bref3 panels from the local phased 1000G PGEN."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from neurooracle.scripts.prepare_case2_adni_ancestry import REFERENCE
from neurooracle.scripts.run_case2_adni_imputation import (
    DEFAULT_JAVA,
    DEFAULT_BREF3_ROOT,
    REFERENCE_ROOT,
    count_vcf,
    export_vcf,
)


DEFAULT_CONVERTER = (
    REFERENCE_ROOT
    / "tools"
    / "beagle_5.5_20250227"
    / "bref3.22Jul22.46e.jar"
)
DEFAULT_WORK_ROOT = Path(tempfile.gettempdir()) / "case2_adni_bref3"


def reference_names(
    output_root: Path,
    work_root: Path,
    chrom: int,
) -> tuple[Path, Path, Path]:
    vcf_prefix = work_root / f"chr{chrom}_1kg_phase3_v5a_b37"
    filename = f"chr{chrom}.1kg.phase3.v5a.b37.bref3"
    return vcf_prefix, work_root / filename, output_root / filename


def convert_bref3(
    java: Path,
    converter: Path,
    vcf: Path,
    output: Path,
    force: bool = False,
) -> None:
    if output.exists() and output.stat().st_size > 0 and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    command = [str(java), "-jar", str(converter), str(vcf)]
    started = time.perf_counter()
    with partial.open("wb") as stdout_handle:
        result = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    log_path = output.with_suffix(".convert.log")
    log_path.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDERR\n"
        + result.stderr
        + f"\n\nELAPSED_SECONDS\n{time.perf_counter() - started:.3f}\n",
        encoding="utf-8",
    )
    if result.returncode != 0 or partial.stat().st_size == 0:
        raise RuntimeError(
            f"bref3 conversion failed ({result.returncode}); see {log_path}"
        )
    partial.replace(output)


def export_counts(log_path: Path, vcf: Path) -> tuple[int, int]:
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        sample_match = re.search(r"(?m)^(\d+) samples .* loaded from", text)
        variant_matches = re.findall(
            r"(?m)^(\d+) variants remaining after main filters\.$",
            text,
        )
        if sample_match and variant_matches:
            return int(sample_match.group(1)), int(variant_matches[-1])
    return count_vcf(vcf)


def build_chromosome(
    chrom: int,
    output_root: Path,
    work_root: Path,
    java: Path,
    converter: Path,
    keep_vcf: bool,
    force: bool,
) -> dict[str, object]:
    vcf_stem, local_bref3, bref3 = reference_names(
        output_root,
        work_root,
        chrom,
    )
    summary_path = bref3.with_suffix(".summary.json")
    if bref3.exists() and summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    vcf = export_vcf(
        REFERENCE,
        vcf_stem,
        chrom,
        None,
        None,
        phased_reference=True,
    )
    samples, variants = export_counts(
        vcf_stem.parent / f"{vcf_stem.name}.export.log",
        vcf,
    )
    convert_bref3(java, converter, vcf, local_bref3, force=force)
    bref3.parent.mkdir(parents=True, exist_ok=True)
    copy_partial = bref3.with_suffix(bref3.suffix + ".copying")
    shutil.copy2(local_bref3, copy_partial)
    if copy_partial.stat().st_size != local_bref3.stat().st_size:
        raise RuntimeError(f"Incomplete bref3 copy: {copy_partial}")
    copy_partial.replace(bref3)
    summary: dict[str, object] = {
        "chromosome": chrom,
        "samples": samples,
        "variants": variants,
        "source_prefix": str(REFERENCE),
        "bref3": str(bref3),
        "bref3_bytes": bref3.stat().st_size,
        "converter": str(converter),
        "work_root": str(work_root),
        "temporary_vcf": str(vcf) if keep_vcf else None,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not keep_vcf:
        vcf.unlink(missing_ok=True)
        vcf_stem.with_suffix(".log").unlink(missing_ok=True)
        (
            vcf_stem.parent / f"{vcf_stem.name}.export.log"
        ).unlink(missing_ok=True)
        local_bref3.unlink(missing_ok=True)
        local_bref3.with_suffix(".convert.log").unlink(missing_ok=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chrom",
        type=int,
        choices=range(1, 23),
        action="append",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BREF3_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--java", type=Path, default=DEFAULT_JAVA)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--keep-vcf", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for required in (args.java, args.converter):
        if not required.exists():
            raise FileNotFoundError(required)
    summaries = [
        build_chromosome(
            chrom,
            args.output_root,
            args.work_root,
            args.java,
            args.converter,
            args.keep_vcf,
            args.force,
        )
        for chrom in (args.chrom or list(range(1, 23)))
    ]
    summary_path = args.output_root / "summary.json"
    by_chromosome: dict[int, dict[str, object]] = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        by_chromosome = {
            int(row["chromosome"]): row
            for row in existing.get("chromosomes", [])
            if isinstance(row, dict) and row.get("chromosome")
        }
    by_chromosome.update(
        {int(row["chromosome"]): row for row in summaries}
    )
    payload = {
        "chromosomes": [
            by_chromosome[chrom]
            for chrom in sorted(by_chromosome)
        ]
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
