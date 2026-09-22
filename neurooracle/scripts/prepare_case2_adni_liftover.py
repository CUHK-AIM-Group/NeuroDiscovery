"""Lift the QC-filtered ADNI1 genotype batch from GRCh36 to GRCh37."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurooracle.src.adni_genetics_harmonization import (
    build_liftover_map,
    run_plink_liftover,
)
from neurooracle.src.adni_genetics_qc import count_lines, detect_genome_build


DEFAULT_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived"
    r"\qc\case2_adni_genetics_v1"
)
DEFAULT_INPUT = (
    DEFAULT_ROOT
    / "batch_qc"
    / "ADNI1_Human610_Quad"
    / "filtered_callrate98_maf01"
)
DEFAULT_CHAIN = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
    r"\liftover\hg18ToHg19.over.chain.gz"
)
DEFAULT_PLINK2 = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
    r"\tools\plink2_20260504\plink2.exe"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "harmonized" / "ADNI1_GRCh37"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-prefix", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--plink2", type=Path, default=DEFAULT_PLINK2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    map_summary = build_liftover_map(
        args.input_prefix.with_suffix(".bim"),
        args.chain,
        args.output_dir,
    )
    output_prefix = args.output_dir / "ADNI1_GRCh37"
    run_plink_liftover(
        args.input_prefix,
        Path(str(map_summary["update_map"])),
        args.plink2,
        output_prefix,
    )
    build, evidence = detect_genome_build(output_prefix.with_suffix(".bim"))
    summary = {
        **map_summary,
        "output_prefix": str(output_prefix),
        "output_samples": count_lines(output_prefix.with_suffix(".fam")),
        "output_variants": count_lines(output_prefix.with_suffix(".bim")),
        "detected_output_build": build,
        "detected_output_build_evidence": evidence,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
