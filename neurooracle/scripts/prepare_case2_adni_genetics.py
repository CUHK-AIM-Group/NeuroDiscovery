"""Audit and conservatively QC ADNI genotype batches for Case Study 2.

The script never modifies raw data. Subject-level outputs remain in the supplied
restricted output directory; only aggregate code and tests belong in Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurooracle.src.adni_genetics_qc import (
    BatchSpec,
    build_sample_audit,
    discover_batches,
    run_batch_qc,
)


DEFAULT_GENETICS_ROOT = Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI")
DEFAULT_FMRI_ROOT = Path(r"\\192.168.3.61\data\Dataset\fMRI\ADNI_fmriprep_all")
DEFAULT_PLINK2 = Path(
    r"\\192.168.3.61\data\Dataset\genetics\references"
    r"\tools\plink2_20260504\plink2.exe"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genetics-root", type=Path, default=DEFAULT_GENETICS_ROOT)
    parser.add_argument(
        "--adnimerge",
        type=Path,
        default=DEFAULT_FMRI_ROOT / "ADNIMERGE_03Jan2025.csv",
    )
    parser.add_argument(
        "--fmri-root",
        type=Path,
        action="append",
        dest="fmri_roots",
        help="Repeat for multiple FMRIPrep roots.",
    )
    parser.add_argument("--plink2", type=Path, default=DEFAULT_PLINK2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENETICS_ROOT / "derived" / "qc" / "case2_adni_genetics_v1",
    )
    parser.add_argument("--run-qc", action="store_true")
    args = parser.parse_args()

    fmri_roots = args.fmri_roots or [
        DEFAULT_FMRI_ROOT / "ADNI2_fmriprep",
        DEFAULT_FMRI_ROOT / "ADNI3_fmriprep",
    ]
    batches = discover_batches(args.genetics_root / "raw")
    audit_dir = args.output_dir / "pre_qc_audit"
    summary = build_sample_audit(batches, args.adnimerge, fmri_roots, audit_dir)
    result: dict[str, object] = {"sample_audit": summary}
    if args.run_qc:
        qc_rows = run_batch_qc(
            batches,
            args.plink2,
            args.output_dir / "batch_qc",
        )
        result["batch_qc"] = qc_rows
        filtered_batches = []
        for row in qc_rows:
            prefix = Path(str(row["filtered_prefix"]))
            filtered_batches.append(
                BatchSpec(
                    name=str(row["batch"]),
                    prefix=prefix,
                    bed=prefix.with_suffix(".bed"),
                    bim=prefix.with_suffix(".bim"),
                    fam=prefix.with_suffix(".fam"),
                )
            )
        result["post_qc_sample_audit"] = build_sample_audit(
            filtered_batches,
            args.adnimerge,
            fmri_roots,
            args.output_dir / "post_qc_audit",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
