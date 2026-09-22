"""Post-process ADNI Case Study 2 imputed genetics into experiment features."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from neurooracle.scripts.prepare_case2_adni_ancestry import PLINK1, PLINK2, ROOT


DEFAULT_FULL_ROOT = ROOT / "imputation" / "full"
DEFAULT_OUTPUT_ROOT = ROOT / "postimputation" / "case2_adni_common_dr2_0p8_v1"
DEFAULT_KEEP_ROOT = ROOT / "preimputation" / "case2_eur_keep_by_batch"
DEFAULT_GWAS = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\references")
    / "gwas"
    / "GCST90027158"
    / "35379992-GCST90027158-MONDO_0004975.h.tsv.gz"
)
APOE_ALLELES = {
    "rs429358": "C",
    "rs7412": "T",
}
PRS_THRESHOLDS = (5e-8, 1e-5, 1e-3, 0.05, 1.0)


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    dr2: float | None
    af: float | None

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.chrom, self.pos, self.ref, self.alt)

    @property
    def coordinate_id(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


def run(
    command: list[str],
    log_path: Path,
    *,
    dry_run: bool = False,
    allow_no_valid_score: bool = False,
    allow_merge_missnp: bool = False,
) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_line = subprocess.list2cmdline(command)
    if dry_run:
        log_path.write_text(f"DRY RUN\n{command_line}\n", encoding="utf-8")
        return True
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.write_text(
        "COMMAND\n"
        + command_line
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if (
        result.returncode != 0
        and allow_no_valid_score
        and "--score: No valid variants" in result.stderr
    ):
        return False
    if (
        result.returncode != 0
        and allow_merge_missnp
        and "variant with 3+ alleles present" in result.stderr
    ):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")
    return True


def prefix_complete(prefix: Path, suffixes: tuple[str, ...]) -> bool:
    return all(prefix.with_suffix(suffix).exists() for suffix in suffixes)


def merge_missnp_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(output_prefix.name + "-merge.missnp")


def data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def parse_info(info: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in info.split(";") if "=" in item)


def parse_float(value: str | None) -> float | None:
    if value in {None, "", ".", "NA", "nan"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def read_pvar(path: Path) -> dict[tuple[str, int, str, str], Variant]:
    variants: dict[tuple[str, int, str, str], Variant] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 5:
                continue
            info = parse_info(fields[6]) if len(fields) > 6 else {}
            variant = Variant(
                chrom=fields[0],
                pos=int(fields[1]),
                variant_id=fields[2],
                ref=fields[3],
                alt=fields[4],
                dr2=parse_float(info.get("DR2")),
                af=parse_float(info.get("AF")),
            )
            variants[variant.key] = variant
    return variants


def read_full_summary(full_root: Path, chrom: int) -> list[dict[str, object]]:
    summary_path = full_root / f"chr{chrom}" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload["batches"])


def postqc_prefixes(full_root: Path, chrom: int) -> dict[str, Path]:
    return {
        str(row["batch"]): Path(str(row["postqc_prefix"]))
        for row in read_full_summary(full_root, chrom)
    }


def common_variant_records(
    variants_by_batch: dict[str, dict[tuple[str, int, str, str], Variant]],
) -> list[dict[str, object]]:
    common_keys: set[tuple[str, int, str, str]] | None = None
    for variants in variants_by_batch.values():
        keys = {
            key
            for key, variant in variants.items()
            if variant.variant_id and variant.variant_id != "."
        }
        common_keys = keys if common_keys is None else common_keys & keys
    if common_keys is None:
        return []

    records: list[dict[str, object]] = []
    batches = sorted(variants_by_batch)
    for key in sorted(common_keys, key=lambda item: (int(item[0]), item[1], item[2], item[3])):
        variants = [variants_by_batch[batch][key] for batch in batches]
        ids = [variant.variant_id for variant in variants]
        dr2_values = [variant.dr2 for variant in variants if variant.dr2 is not None]
        af_values = [variant.af for variant in variants if variant.af is not None]
        stable_ids = {variant_id for variant_id in ids if variant_id != "."}
        records.append(
            {
                "chrom": key[0],
                "pos": key[1],
                "ref": key[2],
                "alt": key[3],
                "canonical_id": ids[0] if len(stable_ids) == 1 else variants[0].coordinate_id,
                "ids_by_batch": dict(zip(batches, ids, strict=True)),
                "min_dr2": min(dr2_values) if dr2_values else None,
                "mean_dr2": mean(dr2_values) if dr2_values else None,
                "mean_af": mean(af_values) if af_values else None,
            }
        )
    return records


def write_common_variant_files(
    records: list[dict[str, object]],
    output_dir: Path,
    batches: Iterable[str],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "common_variants.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chrom",
                "pos",
                "canonical_id",
                "ref",
                "alt",
                "min_dr2",
                "mean_dr2",
                "mean_af",
                "ids_by_batch",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["ids_by_batch"] = json.dumps(row["ids_by_batch"], sort_keys=True)
            writer.writerow(row)

    id_files: dict[str, Path] = {}
    for batch in batches:
        id_path = output_dir / f"{batch}.common_ids.txt"
        with id_path.open("w", encoding="utf-8") as handle:
            for record in records:
                ids_by_batch = record["ids_by_batch"]
                assert isinstance(ids_by_batch, dict)
                handle.write(str(ids_by_batch[batch]) + "\n")
        id_files[batch] = id_path
    return id_files


def stage_chromosome_batches(
    full_root: Path,
    output_root: Path,
    keep_root: Path,
    chrom: int,
    *,
    threads: int,
    dry_run: bool = False,
) -> dict[str, object]:
    prefixes = postqc_prefixes(full_root, chrom)
    chromosome_dir = output_root / "chromosomes" / f"chr{chrom}"
    merged_bed_prefix = chromosome_dir / f"chr{chrom}_common_dr2_0p8_merged_bed"
    merged_prefix = chromosome_dir / f"chr{chrom}_common_dr2_0p8_merged"
    summary_path = chromosome_dir / "summary.json"
    if summary_path.exists() and prefix_complete(
        merged_prefix, (".pgen", ".pvar", ".psam")
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    variants_by_batch = {
        batch: read_pvar(prefix.with_suffix(".pvar"))
        for batch, prefix in prefixes.items()
    }
    records = common_variant_records(variants_by_batch)
    id_files = write_common_variant_files(records, chromosome_dir, prefixes)
    staged_prefixes: dict[str, Path] = {}
    for batch, prefix in prefixes.items():
        keep_path = keep_root / f"{batch}.keep"
        if not keep_path.exists():
            raise FileNotFoundError(keep_path)
        staged_prefix = (
            chromosome_dir
            / "staging"
            / batch
            / f"{batch}_chr{chrom}_common_dr2_0p8"
        )
        staged_prefix.parent.mkdir(parents=True, exist_ok=True)
        if not prefix_complete(staged_prefix, (".pgen", ".pvar", ".psam")):
            run(
                [
                    str(PLINK2),
                    "--pfile",
                    str(prefix),
                    "--keep",
                    str(keep_path),
                    "--extract",
                    str(id_files[batch]),
                    "--set-missing-var-ids",
                    "@:#:$r:$a",
                    "--rm-dup",
                    "force-first",
                    "--threads",
                    str(threads),
                    "--make-pgen",
                    "--sort-vars",
                    "--out",
                    str(staged_prefix),
                ],
                staged_prefix.parent / "stage_common.command.log",
                dry_run=dry_run,
            )
        staged_prefixes[batch] = staged_prefix

    bed_prefixes: dict[str, Path] = {}
    for batch, staged_prefix in staged_prefixes.items():
        bed_prefix = staged_prefix.parent / f"{staged_prefix.name}_bed"
        if not prefix_complete(bed_prefix, (".bed", ".bim", ".fam")):
            run(
                [
                    str(PLINK2),
                    "--pfile",
                    str(staged_prefix),
                    "--threads",
                    str(threads),
                    "--make-bed",
                    "--out",
                    str(bed_prefix),
                ],
                staged_prefix.parent / "common_to_bed.command.log",
                dry_run=dry_run,
            )
        bed_prefixes[batch] = bed_prefix

    merge_list = chromosome_dir / "bed_merge_list.txt"
    merge_list.write_text(
        "\n".join(str(bed_prefixes[batch]) for batch in sorted(bed_prefixes))
        + "\n",
        encoding="utf-8",
    )
    merge_excluded_variants: list[str] = []
    merge_excluded_path: str | None = None
    if not prefix_complete(merged_bed_prefix, (".bed", ".bim", ".fam")):
        merge_ok = run(
            [
                str(PLINK1),
                "--merge-list",
                str(merge_list),
                "--keep-allele-order",
                "--make-bed",
                "--out",
                str(merged_bed_prefix),
            ],
            chromosome_dir / "merge_chromosome_bed.command.log",
            dry_run=dry_run,
            allow_merge_missnp=True,
        )
        if not merge_ok:
            missnp_path = merge_missnp_path(merged_bed_prefix)
            if not missnp_path.exists():
                raise RuntimeError(
                    "PLINK merge reported a multiallelic conflict, "
                    f"but no missnp file was written: {missnp_path}"
                )
            merge_excluded_variants = [
                line.strip()
                for line in missnp_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            merge_excluded_path = str(missnp_path)
            clean_bed_prefixes: dict[str, Path] = {}
            for batch, bed_prefix in bed_prefixes.items():
                clean_prefix = bed_prefix.parent / f"{bed_prefix.name}_mergeclean"
                if not prefix_complete(clean_prefix, (".bed", ".bim", ".fam")):
                    run(
                        [
                            str(PLINK1),
                            "--bfile",
                            str(bed_prefix),
                            "--exclude",
                            str(missnp_path),
                            "--keep-allele-order",
                            "--make-bed",
                            "--out",
                            str(clean_prefix),
                        ],
                        clean_prefix.parent / "exclude_merge_missnp.command.log",
                        dry_run=dry_run,
                    )
                clean_bed_prefixes[batch] = clean_prefix

            clean_merge_list = chromosome_dir / "bed_merge_list.mergeclean.txt"
            clean_merge_list.write_text(
                "\n".join(
                    str(clean_bed_prefixes[batch])
                    for batch in sorted(clean_bed_prefixes)
                )
                + "\n",
                encoding="utf-8",
            )
            run(
                [
                    str(PLINK1),
                    "--merge-list",
                    str(clean_merge_list),
                    "--keep-allele-order",
                    "--make-bed",
                    "--out",
                    str(merged_bed_prefix),
                ],
                chromosome_dir / "merge_chromosome_bed_retry.command.log",
                dry_run=dry_run,
            )

    if not prefix_complete(merged_prefix, (".pgen", ".pvar", ".psam")):
        run(
            [
                str(PLINK2),
                "--bfile",
                str(merged_bed_prefix),
                "--threads",
                str(threads),
                "--make-pgen",
                "--sort-vars",
                "--out",
                str(merged_prefix),
            ],
            chromosome_dir / "merge_chromosome.command.log",
            dry_run=dry_run,
        )
    summary = {
        "chromosome": chrom,
        "batches": sorted(prefixes),
        "common_variants": len(records),
        "merge_representation": "PLINK1 hardcall BED merge converted back to PGEN",
        "merge_excluded_variants": len(merge_excluded_variants),
        "merge_excluded_variant_ids": merge_excluded_variants[:20],
        "merge_excluded_path": merge_excluded_path,
        "merged_prefix": str(merged_prefix),
        "sample_count": data_rows(merged_prefix.with_suffix(".psam"))
        if merged_prefix.with_suffix(".psam").exists()
        else None,
        "variant_count": data_rows(merged_prefix.with_suffix(".pvar"))
        if merged_prefix.with_suffix(".pvar").exists()
        else None,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def merge_autosomes(
    output_root: Path,
    chromosome_summaries: list[dict[str, object]],
    *,
    threads: int,
    dry_run: bool = False,
) -> Path:
    merged_dir = output_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merge_list = merged_dir / "autosome_pmerge_list.txt"
    prefixes = [str(row["merged_prefix"]) for row in chromosome_summaries]
    merge_list.write_text("\n".join(prefixes) + "\n", encoding="utf-8")
    final_prefix = merged_dir / "case2_adni_common_dr2_0p8_autosomes"
    if not final_prefix.with_suffix(".pgen").exists():
        run(
            [
                str(PLINK2),
                "--pmerge-list",
                str(merge_list),
                "pfile",
                "--merge-info-mode",
                "erase",
                "--merge-mode",
                "nm-first",
                "--threads",
                str(threads),
                "--make-pgen",
                "--sort-vars",
                "--out",
                str(final_prefix),
            ],
            merged_dir / "merge_autosomes.command.log",
            dry_run=dry_run,
        )
    return final_prefix


def extract_subject_id(iid: str) -> str:
    match = re.search(r"\d{3}_S_\d{4}", iid)
    return match.group(0) if match else iid


def hard_count(value: str | float | None) -> int | None:
    parsed = parse_float(str(value) if value is not None else None)
    if parsed is None:
        return None
    rounded = round(parsed)
    return int(rounded) if abs(parsed - rounded) <= 0.05 else None


def apoe_label(rs429358_c: object, rs7412_t: object) -> str:
    e4 = hard_count(rs429358_c)
    e2 = hard_count(rs7412_t)
    if e4 is None or e2 is None:
        return "incomplete"
    labels = {
        (0, 0): "e3/e3",
        (1, 0): "e3/e4",
        (2, 0): "e4/e4",
        (0, 1): "e2/e3",
        (0, 2): "e2/e2",
        (1, 1): "e2/e4",
    }
    return labels.get((e4, e2), "ambiguous")


def find_variant_ids(prefix: Path, ids: Iterable[str]) -> set[str]:
    wanted = set(ids)
    found: set[str] = set()
    with prefix.with_suffix(".pvar").open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            variant_id = line.rstrip().split("\t", 4)[2]
            if variant_id in wanted:
                found.add(variant_id)
    return found


def raw_value(row: dict[str, str], variant_id: str, allele: str) -> str | None:
    candidates = (
        f"{variant_id}_{allele}",
        f"{variant_id}_{allele}_DOSAGE",
        variant_id,
    )
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
    for key, value in row.items():
        if key.startswith(f"{variant_id}_"):
            return value
    return None


def extract_apoe_features(
    full_root: Path,
    output_root: Path,
    *,
    threads: int,
    dry_run: bool = False,
) -> dict[str, object]:
    feature_dir = output_root / "features" / "apoe"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prefixes = postqc_prefixes(full_root, 19)
    allele_file = feature_dir / "apoe_count_alleles.tsv"
    allele_file.write_text(
        "".join(f"{variant_id}\t{allele}\n" for variant_id, allele in APOE_ALLELES.items()),
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    available_by_batch: dict[str, list[str]] = {}
    for batch, prefix in prefixes.items():
        found = sorted(find_variant_ids(prefix, APOE_ALLELES))
        available_by_batch[batch] = found
        if not found:
            continue
        extract_file = feature_dir / f"{batch}.apoe_ids.txt"
        extract_file.write_text("\n".join(found) + "\n", encoding="utf-8")
        out_prefix = feature_dir / batch / f"{batch}_apoe"
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        if not out_prefix.with_suffix(".raw").exists():
            run(
                [
                    str(PLINK2),
                    "--pfile",
                    str(prefix),
                    "--extract",
                    str(extract_file),
                    "--export-allele",
                    str(allele_file),
                    "--threads",
                    str(threads),
                    "--export",
                    "A",
                    "--out",
                    str(out_prefix),
                ],
                out_prefix.parent / "export_apoe.command.log",
                dry_run=dry_run,
            )
        raw_path = out_prefix.with_suffix(".raw")
        if not raw_path.exists():
            continue
        with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                iid = row.get("#IID") or row.get("IID") or ""
                rs429358_c = raw_value(row, "rs429358", "C")
                rs7412_t = raw_value(row, "rs7412", "T")
                label = apoe_label(rs429358_c, rs7412_t)
                rows.append(
                    {
                        "iid": iid,
                        "subject_id": extract_subject_id(iid),
                        "batch": batch,
                        "rs429358_C_dosage": rs429358_c,
                        "rs7412_T_dosage": rs7412_t,
                        "apoe_e4_dosage": rs429358_c,
                        "apoe_e2_dosage": rs7412_t,
                        "apoe_genotype": label,
                        "apoe_complete": label != "incomplete",
                    }
                )

    table_path = output_root / "features" / "apoe_dosage.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iid",
                "subject_id",
                "batch",
                "rs429358_C_dosage",
                "rs7412_T_dosage",
                "apoe_e4_dosage",
                "apoe_e2_dosage",
                "apoe_genotype",
                "apoe_complete",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(1 for row in rows if row["apoe_complete"])
    return {
        "feature_table": str(table_path),
        "samples": len(rows),
        "complete_apoe_samples": complete,
        "available_variants_by_batch": available_by_batch,
    }


def threshold_label(threshold: float) -> str:
    return f"p{threshold:.0e}".replace("+", "").replace("-", "m")


def prepare_prs_score_files(
    gwas_path: Path,
    output_dir: Path,
    thresholds: Iterable[float],
) -> dict[float, Path]:
    thresholds = tuple(sorted(thresholds))
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[float, object] = {}
    writers: dict[float, csv.writer] = {}
    seen: dict[float, set[str]] = {threshold: set() for threshold in thresholds}
    paths = {
        threshold: output_dir / f"bellenguez_ad_{threshold_label(threshold)}.score.tsv"
        for threshold in thresholds
    }
    try:
        for threshold, path in paths.items():
            handle = path.open("w", encoding="utf-8", newline="")
            handles[threshold] = handle
            writer = csv.writer(handle, delimiter="\t")
            writers[threshold] = writer
            writer.writerow(["ID", "A1", "BETA"])
        with gzip.open(gwas_path, "rt", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                rsid = (row.get("hm_rsid") or "").strip()
                allele = (row.get("hm_effect_allele") or "").strip().upper()
                beta = parse_float(row.get("hm_beta"))
                p_value = parse_float(row.get("p_value"))
                if not rsid or rsid == "." or not allele or beta is None or p_value is None:
                    continue
                for threshold in thresholds:
                    if p_value <= threshold and rsid not in seen[threshold]:
                        writers[threshold].writerow([rsid, allele, beta])
                        seen[threshold].add(rsid)
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        threshold_label(threshold): {
            "path": str(paths[threshold]),
            "variants": len(seen[threshold]),
            "p_value_max": threshold,
        }
        for threshold in thresholds
    }
    (output_dir / "score_file_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def read_score_output(path: Path, label: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            iid = row.get("#IID") or row.get("IID")
            if not iid:
                continue
            score_avg = next((value for key, value in row.items() if key.endswith("_AVG")), "")
            score_sum = next((value for key, value in row.items() if key.endswith("_SUM")), "")
            rows[iid] = {
                f"ad_prs_{label}_avg": score_avg,
                f"ad_prs_{label}_sum": score_sum,
            }
    return rows


def run_prs(
    merged_prefix: Path,
    gwas_path: Path,
    output_root: Path,
    *,
    thresholds: Iterable[float],
    threads: int,
    dry_run: bool = False,
) -> dict[str, object]:
    prs_dir = output_root / "features" / "ad_prs_bellenguez"
    score_dir = prs_dir / "score_files"
    score_files = prepare_prs_score_files(gwas_path, score_dir, thresholds)
    merged_rows: dict[str, dict[str, str]] = {}
    score_summaries: dict[str, dict[str, object]] = {}
    for threshold, score_file in score_files.items():
        label = threshold_label(threshold)
        out_prefix = prs_dir / label / f"case2_adni_bellenguez_{label}"
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        if not out_prefix.with_suffix(".sscore").exists():
            scored = run(
                [
                    str(PLINK2),
                    "--pfile",
                    str(merged_prefix),
                    "--score",
                    str(score_file),
                    "1",
                    "2",
                    "3",
                    "header-read",
                    "ignore-dup-ids",
                    "list-variants",
                    "cols=+scoresums",
                    "--threads",
                    str(threads),
                    "--out",
                    str(out_prefix),
                ],
                out_prefix.parent / "score.command.log",
                dry_run=dry_run,
                allow_no_valid_score=True,
            )
            if not scored:
                score_summaries[label] = {
                    "score_file": str(score_file),
                    "sscore": None,
                    "variants_available_for_scoring": 0,
                    "status": "no_valid_variants",
                }
                continue
        sscore_path = out_prefix.with_suffix(".sscore")
        if sscore_path.exists():
            for iid, values in read_score_output(sscore_path, label).items():
                row = merged_rows.setdefault(
                    iid,
                    {"iid": iid, "subject_id": extract_subject_id(iid)},
                )
                row.update(values)
        vars_path = out_prefix.with_suffix(".sscore.vars")
        score_summaries[label] = {
            "score_file": str(score_file),
            "sscore": str(sscore_path),
            "variants_available_for_scoring": data_rows(vars_path)
            if vars_path.exists()
            else None,
            "status": "scored" if sscore_path.exists() else "missing_output",
        }

    table_path = prs_dir / "adni_ad_prs_bellenguez.csv"
    fieldnames = ["iid", "subject_id"]
    for threshold in sorted(score_files):
        label = threshold_label(threshold)
        fieldnames.extend([f"ad_prs_{label}_avg", f"ad_prs_{label}_sum"])
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for iid in sorted(merged_rows):
            writer.writerow(merged_rows[iid])
    return {
        "feature_table": str(table_path),
        "samples": len(merged_rows),
        "scores": score_summaries,
        "gwas": str(gwas_path),
    }


def compute_pca(
    merged_prefix: Path,
    output_root: Path,
    *,
    threads: int,
    dry_run: bool = False,
) -> dict[str, object]:
    pca_dir = output_root / "features" / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    prune_prefix = pca_dir / "case2_common_ld"
    if not prune_prefix.with_suffix(".prune.in").exists():
        run(
            [
                str(PLINK2),
                "--pfile",
                str(merged_prefix),
                "--maf",
                "0.05",
                "--indep-pairwise",
                "200",
                "50",
                "0.2",
                "--threads",
                str(threads),
                "--out",
                str(prune_prefix),
            ],
            pca_dir / "ld_prune.command.log",
            dry_run=dry_run,
        )
    pca_prefix = pca_dir / "case2_adni_common_pca20"
    if not pca_prefix.with_suffix(".eigenvec").exists():
        run(
            [
                str(PLINK2),
                "--pfile",
                str(merged_prefix),
                "--extract",
                str(prune_prefix.with_suffix(".prune.in")),
                "--pca",
                "20",
                "approx",
                "--threads",
                str(threads),
                "--out",
                str(pca_prefix),
            ],
            pca_dir / "pca.command.log",
            dry_run=dry_run,
        )
    return {
        "pruned_variants": data_rows(prune_prefix.with_suffix(".prune.in"))
        if prune_prefix.with_suffix(".prune.in").exists()
        else None,
        "eigenvec": str(pca_prefix.with_suffix(".eigenvec")),
        "eigenval": str(pca_prefix.with_suffix(".eigenval")),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    chromosomes = args.chromosome or list(range(1, 23))
    chromosome_summaries = [
        stage_chromosome_batches(
            args.full_root,
            args.output_root,
            args.keep_root,
            chrom,
            threads=args.threads,
            dry_run=args.dry_run,
        )
        for chrom in chromosomes
    ]
    merged_prefix = merge_autosomes(
        args.output_root,
        chromosome_summaries,
        threads=args.threads,
        dry_run=args.dry_run,
    )
    apoe = extract_apoe_features(
        args.full_root,
        args.output_root,
        threads=args.threads,
        dry_run=args.dry_run,
    )
    pca = compute_pca(
        merged_prefix,
        args.output_root,
        threads=args.threads,
        dry_run=args.dry_run,
    )
    prs = run_prs(
        merged_prefix,
        args.gwas,
        args.output_root,
        thresholds=args.prs_threshold,
        threads=args.threads,
        dry_run=args.dry_run,
    )
    summary = {
        "full_root": str(args.full_root),
        "output_root": str(args.output_root),
        "keep_root": str(args.keep_root),
        "chromosomes": chromosomes,
        "merged_prefix": str(merged_prefix),
        "samples": data_rows(merged_prefix.with_suffix(".psam"))
        if merged_prefix.with_suffix(".psam").exists()
        else None,
        "variants": data_rows(merged_prefix.with_suffix(".pvar"))
        if merged_prefix.with_suffix(".pvar").exists()
        else None,
        "chromosome_summaries": chromosome_summaries,
        "apoe": apoe,
        "pca": pca,
        "prs": prs,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--keep-root", type=Path, default=DEFAULT_KEEP_ROOT)
    parser.add_argument("--gwas", type=Path, default=DEFAULT_GWAS)
    parser.add_argument("--chromosome", type=int, action="append")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prs-threshold", type=float, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.prs_threshold is None:
        args.prs_threshold = list(PRS_THRESHOLDS)
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Last Updated At: 2026-07-31 01:48 HKT
