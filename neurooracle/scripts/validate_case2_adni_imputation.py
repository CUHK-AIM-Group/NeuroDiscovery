"""Validate sample retention, typed-site concordance, and DR2 for ADNI imputation."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from neurooracle.scripts.prepare_case2_adni_ancestry import BATCHES, ROOT


def genotype_dosage(sample_field: str) -> int | None:
    genotype = sample_field.split(":", 1)[0]
    if genotype in {".", "./.", ".|."}:
        return None
    alleles = genotype.replace("|", "/").split("/")
    if len(alleles) != 2 or any(allele == "." for allele in alleles):
        return None
    return sum(int(allele) for allele in alleles)


def read_vcf_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                return line.rstrip().split("\t")[9:]
    raise ValueError(f"Missing #CHROM header in {path}")


def target_genotypes(path: Path) -> dict[tuple[str, str, str, str], list[int | None]]:
    genotypes: dict[tuple[str, str, str, str], list[int | None]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            key = (fields[0], fields[1], fields[3], fields[4])
            genotypes[key] = [genotype_dosage(value) for value in fields[9:]]
    return genotypes


def typed_site_concordance(target: Path, imputed: Path) -> dict[str, int | float]:
    target_samples = read_vcf_header(target)
    imputed_samples = read_vcf_header(imputed)
    if target_samples != imputed_samples:
        raise ValueError("Target and imputed VCF sample order differs")
    expected = target_genotypes(target)
    matched_sites = compared = concordant = 0
    with gzip.open(imputed, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            key = (fields[0], fields[1], fields[3], fields[4])
            target_calls = expected.get(key)
            if target_calls is None:
                continue
            matched_sites += 1
            imputed_calls = [genotype_dosage(value) for value in fields[9:]]
            for target_call, imputed_call in zip(target_calls, imputed_calls):
                if target_call is None or imputed_call is None:
                    continue
                compared += 1
                concordant += target_call == imputed_call
    return {
        "target_sites": len(expected),
        "matched_typed_sites": matched_sites,
        "compared_genotypes": compared,
        "concordant_genotypes": concordant,
        "concordance": concordant / compared if compared else 0.0,
    }


def postqc_metrics(pvar: Path) -> dict[str, int | float]:
    seen: set[tuple[str, str, str, str]] = set()
    variants = duplicates = 0
    min_dr2 = 1.0
    with pvar.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            key = (fields[0], fields[1], fields[3], fields[4])
            duplicates += key in seen
            seen.add(key)
            variants += 1
            info = dict(
                item.split("=", 1)
                for item in fields[6].split(";")
                if "=" in item
            )
            if "DR2" in info:
                min_dr2 = min(min_dr2, float(info["DR2"]))
    return {
        "postqc_variants": variants,
        "duplicate_variant_keys": duplicates,
        "minimum_dr2": min_dr2,
    }


def validate_batch(batch_dir: Path) -> dict[str, object]:
    summary = json.loads(
        (batch_dir / "summary.json").read_text(encoding="utf-8")
    )
    target = batch_dir / "target.vcf.gz"
    imputed = Path(str(summary["imputed_vcf"]))
    postqc = Path(str(summary["postqc_prefix"]) + ".pvar")
    result: dict[str, object] = {
        "batch": summary["batch"],
        "target_samples": summary["target_samples"],
        "imputed_samples": summary["imputed_samples"],
        "samples_retained": (
            summary["target_samples"] == summary["imputed_samples"]
        ),
        **typed_site_concordance(target, imputed),
        **postqc_metrics(postqc),
    }
    (batch_dir / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def merge_validation_results(
    root: Path, new_results: list[dict[str, object]], *, chrom: int
) -> dict[str, object]:
    """Merge incremental validation without dropping prior batch results."""

    by_batch: dict[str, dict[str, object]] = {}
    root_validation = root / "validation.json"
    if root_validation.exists():
        existing = json.loads(root_validation.read_text(encoding="utf-8"))
        by_batch.update(
            {
                str(row["batch"]): row
                for row in existing.get("batches", [])
                if isinstance(row, dict) and row.get("batch")
            }
        )
    for batch in BATCHES:
        batch_validation = root / batch / f"chr{chrom}" / "validation.json"
        if batch_validation.exists():
            row = json.loads(batch_validation.read_text(encoding="utf-8"))
            if isinstance(row, dict) and row.get("batch"):
                by_batch[str(row["batch"])] = row
    by_batch.update({str(row["batch"]): row for row in new_results})
    return {
        "batches": [by_batch[batch] for batch in BATCHES if batch in by_batch]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "imputation" / "full" / "chr22",
    )
    parser.add_argument("--chrom", type=int, default=22)
    parser.add_argument("--batch", choices=sorted(BATCHES), action="append")
    args = parser.parse_args()
    summary = json.loads(
        (args.root / "summary.json").read_text(encoding="utf-8")
    )
    selected = set(args.batch or BATCHES)
    results = [
        validate_batch(args.root / str(row["batch"]) / f"chr{args.chrom}")
        for row in summary["batches"]
        if str(row["batch"]) in selected
    ]
    payload = merge_validation_results(args.root, results, chrom=args.chrom)
    (args.root / "validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
