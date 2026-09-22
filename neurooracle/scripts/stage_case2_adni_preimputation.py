"""Stage final GRCh37 ADNI array files for Case Study 2 imputation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from neurooracle.scripts.prepare_case2_adni_ancestry import BATCHES, PLINK2, ROOT


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


def data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def stage_batch(batch: str, preimputation: Path, output_root: Path) -> dict[str, object]:
    output_dir = output_root / batch
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / f"{batch}_GRCh37_preimputation"
    keep_path = preimputation / "case2_eur_keep_by_batch" / f"{batch}.keep"
    extract_path = preimputation / batch / "reference_matched_ids.txt"
    reference_allele_path = preimputation / batch / "reference_ref_alleles.tsv"
    run(
        [
            str(PLINK2),
            "--bfile",
            str(BATCHES[batch]),
            "--keep",
            str(keep_path),
            "--extract",
            str(extract_path),
            "--ref-allele",
            "force",
            str(reference_allele_path),
            "2",
            "1",
            "--rm-dup",
            "force-first",
            "--mind",
            "0.02",
            "--geno",
            "0.02",
            "--maf",
            "0.01",
            "--sort-vars",
            "--make-pgen",
            "--out",
            str(output_prefix),
        ],
        output_dir / "stage.command.log",
    )
    summary: dict[str, object] = {
        "batch": batch,
        "source_prefix": str(BATCHES[batch]),
        "keep_file": str(keep_path),
        "extract_file": str(extract_path),
        "reference_allele_file": str(reference_allele_path),
        "output_prefix": str(output_prefix),
        "genome_build": "GRCh37",
        "samples": data_rows(output_prefix.with_suffix(".psam")),
        "variants": data_rows(output_prefix.with_suffix(".pvar")),
        "sample_missingness_max": 0.02,
        "variant_missingness_max": 0.02,
        "minor_allele_frequency_min": 0.01,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preimputation",
        type=Path,
        default=ROOT / "preimputation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "preimputation" / "ready",
    )
    parser.add_argument("--batch", choices=sorted(BATCHES), action="append")
    args = parser.parse_args()
    batches = args.batch or list(BATCHES)
    new_summaries = [
        stage_batch(batch, args.preimputation, args.output_root)
        for batch in batches
    ]
    summary_path = args.output_root / "summary.json"
    by_batch: dict[str, dict[str, object]] = {}
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        by_batch = {
            str(row["batch"]): row
            for row in existing.get("batches", [])
            if isinstance(row, dict) and row.get("batch")
        }
    by_batch.update({str(row["batch"]): row for row in new_summaries})
    payload = {
        "batches": [
            by_batch[batch]
            for batch in BATCHES
            if batch in by_batch
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
