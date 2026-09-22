"""Coordinate and allele harmonization helpers for ADNI genotype arrays."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


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


def _ucsc_chromosome(chromosome: str) -> str:
    text = chromosome.strip()
    if text.casefold().startswith("chr"):
        return text
    if text in {"M", "MT", "26"}:
        return "chrM"
    if text == "23":
        return "chrX"
    if text == "24":
        return "chrY"
    return f"chr{text}"


def _plink_chromosome(chromosome: str) -> str:
    text = chromosome.removeprefix("chr")
    return "MT" if text == "M" else text


def build_liftover_map(
    bim_path: Path,
    chain_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create a conservative PLINK update map from a UCSC chain file."""

    from pyliftover import LiftOver

    output_dir.mkdir(parents=True, exist_ok=True)
    lifter = LiftOver(str(chain_path))
    source_rows: list[tuple[str, str, int, str, str]] = []
    id_counts: Counter[str] = Counter()
    with bim_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f"Malformed BIM row at {bim_path}:{line_number}")
            chromosome, variant_id, _, position, allele1, allele2 = fields[:6]
            source_rows.append((chromosome, variant_id, int(position), allele1, allele2))
            id_counts[variant_id] += 1

    candidates: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for chromosome, variant_id, position, allele1, allele2 in source_rows:
        base = {
            "variant_id": variant_id,
            "old_chrom": chromosome,
            "old_pos": position,
            "allele1": allele1,
            "allele2": allele2,
        }
        if not variant_id or variant_id == ".":
            dropped.append({**base, "reason": "missing_variant_id"})
            continue
        if id_counts[variant_id] > 1:
            dropped.append({**base, "reason": "duplicate_source_id"})
            continue
        mappings = lifter.convert_coordinate(_ucsc_chromosome(chromosome), position - 1)
        if not mappings:
            dropped.append({**base, "reason": "unmapped"})
            continue
        if len(mappings) != 1:
            dropped.append({**base, "reason": "multiple_mappings"})
            continue
        new_chromosome, new_position_zero, strand, score = mappings[0]
        if strand != "+":
            dropped.append({**base, "reason": "reverse_strand_mapping"})
            continue
        candidates.append(
            {
                **base,
                "new_chrom": _plink_chromosome(new_chromosome),
                "new_pos": int(new_position_zero) + 1,
                "strand": strand,
                "chain_score": score,
            }
        )

    target_counts: Counter[tuple[str, int]] = Counter(
        (str(row["new_chrom"]), int(row["new_pos"])) for row in candidates
    )
    mapped: list[dict[str, object]] = []
    for row in candidates:
        target = (str(row["new_chrom"]), int(row["new_pos"]))
        if target_counts[target] > 1:
            dropped.append(
                {
                    **{key: row[key] for key in ("variant_id", "old_chrom", "old_pos", "allele1", "allele2")},
                    "reason": "duplicate_target_coordinate",
                }
            )
            continue
        mapped.append(row)

    map_path = output_dir / "liftover_update.tsv"
    with map_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in mapped:
            writer.writerow((row["variant_id"], row["new_chrom"], row["new_pos"]))

    mapped_audit_path = output_dir / "liftover_mapped.tsv"
    mapped_fields = (
        "variant_id",
        "old_chrom",
        "old_pos",
        "new_chrom",
        "new_pos",
        "strand",
        "chain_score",
        "allele1",
        "allele2",
    )
    with mapped_audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapped_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(mapped)

    dropped_path = output_dir / "liftover_dropped.tsv"
    dropped_fields = ("variant_id", "old_chrom", "old_pos", "allele1", "allele2", "reason")
    with dropped_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dropped_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(dropped)

    reason_counts = Counter(str(row["reason"]) for row in dropped)
    summary: dict[str, object] = {
        "input_bim": str(bim_path),
        "chain": str(chain_path),
        "input_variants": len(source_rows),
        "mapped_variants": len(mapped),
        "dropped_variants": len(dropped),
        "mapped_fraction": len(mapped) / len(source_rows) if source_rows else 0.0,
        "drop_reasons": dict(sorted(reason_counts.items())),
        "update_map": str(map_path),
        "mapped_audit": str(mapped_audit_path),
        "dropped_audit": str(dropped_path),
    }
    (output_dir / "liftover_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def run_plink_liftover(
    input_prefix: Path,
    update_map: Path,
    plink2: Path,
    output_prefix: Path,
) -> None:
    """Apply a generated chromosome/position map to a PLINK fileset."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sorted_prefix = output_prefix.parent / f"{output_prefix.name}_sorted"
    _run(
        [
            str(plink2),
            "--bfile",
            str(input_prefix),
            "--extract",
            str(update_map),
            "--update-chr",
            str(update_map),
            "2",
            "1",
            "--update-map",
            str(update_map),
            "3",
            "1",
            "--sort-vars",
            "--make-pgen",
            "--out",
            str(sorted_prefix),
        ],
        output_prefix.parent / "plink_liftover_to_pgen.command.log",
    )
    _run(
        [
            str(plink2),
            "--pfile",
            str(sorted_prefix),
            "--make-bed",
            "--out",
            str(output_prefix),
        ],
        output_prefix.parent / "plink_liftover_to_bed.command.log",
    )


def _read_bim(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    duplicate_ids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            chromosome, variant_id, _, position, allele1, allele2 = line.split()[:6]
            if variant_id in rows:
                duplicate_ids.add(variant_id)
            rows[variant_id] = {
                "chrom": chromosome,
                "pos": int(position),
                "allele1": allele1.upper(),
                "allele2": allele2.upper(),
            }
    for variant_id in duplicate_ids:
        rows.pop(variant_id, None)
    return rows


def prepare_autosomal_candidates(bim_path: Path, output_dir: Path) -> dict[str, object]:
    """Select unique, non-palindromic autosomal SNP IDs for ancestry analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_bim(bim_path)
    coordinate_counts: Counter[tuple[str, int]] = Counter(
        (str(row["chrom"]), int(row["pos"])) for row in rows.values()
    )
    keep: list[str] = []
    reasons: Counter[str] = Counter()
    for variant_id, row in rows.items():
        chrom = str(row["chrom"])
        alleles = {str(row["allele1"]), str(row["allele2"])}
        try:
            autosomal = 1 <= int(chrom) <= 22
        except ValueError:
            autosomal = False
        if not autosomal:
            reasons["non_autosomal"] += 1
        elif not variant_id.startswith("rs"):
            reasons["non_rs_id"] += 1
        elif any(len(allele) != 1 or allele not in "ACGT" for allele in alleles):
            reasons["non_snp"] += 1
        elif alleles in ({"A", "T"}, {"C", "G"}):
            reasons["palindromic"] += 1
        elif coordinate_counts[(chrom, int(row["pos"]))] > 1:
            reasons["duplicate_coordinate"] += 1
        else:
            keep.append(variant_id)
    keep.sort()
    candidate_path = output_dir / "candidate_ids.txt"
    candidate_path.write_text("".join(f"{variant_id}\n" for variant_id in keep), encoding="utf-8")
    summary: dict[str, object] = {
        "input_variants_with_unique_ids": len(rows),
        "candidate_variants": len(keep),
        "excluded": dict(sorted(reasons.items())),
        "candidate_ids": str(candidate_path),
    }
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _read_pvar(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0] == "#CHROM":
                continue
            if len(fields) < 5:
                continue
            rows[fields[2]] = {
                "chrom": fields[0],
                "pos": int(fields[1]),
                "ref": fields[3].upper(),
                "alt": fields[4].upper(),
            }
    return rows


def build_reference_match_set(
    adni_bim: Path,
    reference_pvar: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Require exact ID, position, and unordered allele-set agreement."""

    output_dir.mkdir(parents=True, exist_ok=True)
    adni = _read_bim(adni_bim)
    reference = _read_pvar(reference_pvar)
    matched: list[str] = []
    reference_alleles: list[tuple[str, str]] = []
    reasons: Counter[str] = Counter()
    for variant_id, adni_row in adni.items():
        ref_row = reference.get(variant_id)
        if ref_row is None:
            reasons["missing_from_reference_subset"] += 1
            continue
        if str(adni_row["chrom"]) != str(ref_row["chrom"]) or int(adni_row["pos"]) != int(ref_row["pos"]):
            reasons["position_mismatch"] += 1
            continue
        adni_alleles = {str(adni_row["allele1"]), str(adni_row["allele2"])}
        ref_alleles = {str(ref_row["ref"]), str(ref_row["alt"])}
        if adni_alleles != ref_alleles:
            reasons["allele_mismatch_or_complement"] += 1
            continue
        matched.append(variant_id)
        reference_alleles.append((variant_id, str(ref_row["ref"])))
    matched.sort()
    reference_alleles.sort()
    match_path = output_dir / "reference_matched_ids.txt"
    match_path.write_text("".join(f"{variant_id}\n" for variant_id in matched), encoding="utf-8")
    reference_allele_path = output_dir / "reference_ref_alleles.tsv"
    reference_allele_path.write_text(
        "".join(f"{variant_id}\t{allele}\n" for variant_id, allele in reference_alleles),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "adni_variants": len(adni),
        "reference_subset_variants": len(reference),
        "exact_matches": len(matched),
        "excluded": dict(sorted(reasons.items())),
        "matched_ids": str(match_path),
        "reference_alleles": str(reference_allele_path),
    }
    (output_dir / "reference_match_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _read_whitespace_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline().lstrip("#").split()
        return [dict(zip(header, line.split())) for line in handle if line.strip()]


def _read_csv_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assign_ancestry(
    eigenvec_path: Path,
    reference_psam: Path,
    output_path: Path,
    pc_count: int = 4,
) -> dict[str, object]:
    """Assign broad ancestry by nearest standardized 1000G superpopulation centroid."""

    psam_rows = _read_whitespace_table(reference_psam)
    superpop = {row["IID"]: row["SuperPop"] for row in psam_rows}
    eigen_rows = _read_whitespace_table(eigenvec_path)
    pc_columns = [f"PC{index}" for index in range(1, pc_count + 1)]
    reference_rows = [row for row in eigen_rows if row.get("IID") in superpop]
    if not reference_rows:
        raise ValueError(f"No 1000 Genomes samples found in {eigenvec_path}")
    reference_matrix = np.asarray(
        [[float(row[column]) for column in pc_columns] for row in reference_rows],
        dtype=float,
    )
    mean = reference_matrix.mean(axis=0)
    scale = reference_matrix.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    standardized = (reference_matrix - mean) / scale
    populations = sorted(set(superpop.values()))
    centroids = {
        population: standardized[
            [superpop[row["IID"]] == population for row in reference_rows]
        ].mean(axis=0)
        for population in populations
    }
    eur_distances = [
        float(np.linalg.norm(vector - centroids["EUR"]))
        for vector, row in zip(standardized, reference_rows)
        if superpop[row["IID"]] == "EUR"
    ]
    eur_threshold = float(np.quantile(eur_distances, 0.99))

    output_rows: list[dict[str, object]] = []
    assignment_counts: Counter[str] = Counter()
    for row in eigen_rows:
        if row.get("IID") in superpop:
            continue
        vector = (np.asarray([float(row[column]) for column in pc_columns]) - mean) / scale
        distances = {
            population: float(np.linalg.norm(vector - centroid))
            for population, centroid in centroids.items()
        }
        assigned = min(distances, key=distances.get)
        assignment_counts[assigned] += 1
        output_rows.append(
            {
                "FID": row.get("FID", "0"),
                "IID": row["IID"],
                "assigned_superpop": assigned,
                "distance_to_assigned": distances[assigned],
                "distance_to_EUR": distances["EUR"],
                "EUR_compatible": assigned == "EUR" and distances["EUR"] <= eur_threshold,
                **{column: row[column] for column in pc_columns},
            }
        )
    fields = (
        "FID",
        "IID",
        "assigned_superpop",
        "distance_to_assigned",
        "distance_to_EUR",
        "EUR_compatible",
        *pc_columns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    summary: dict[str, object] = {
        "adni_samples": len(output_rows),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "eur_compatible": sum(bool(row["EUR_compatible"]) for row in output_rows),
        "eur_reference_distance_99pct": eur_threshold,
        "pc_count": pc_count,
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def summarize_sample_qc(
    ancestry_path: Path,
    het_path: Path,
    sexcheck_path: Path,
    kinship_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Join ancestry, heterozygosity, sex-check, and relatedness flags."""

    ancestry = {row["IID"]: row for row in _read_csv_table(ancestry_path)}
    het = {row["IID"]: row for row in _read_whitespace_table(het_path)}
    sex = {row["IID"]: row for row in _read_whitespace_table(sexcheck_path)}
    f_values = np.asarray(
        [float(row["F"]) for row in het.values() if row.get("F") not in {None, "NA", "nan"}],
        dtype=float,
    )
    f_mean = float(f_values.mean()) if len(f_values) else math.nan
    f_sd = float(f_values.std(ddof=1)) if len(f_values) > 1 else math.nan
    related_ids: set[str] = set()
    related_pairs = 0
    if kinship_path.exists():
        for row in _read_whitespace_table(kinship_path):
            related_pairs += 1
            related_ids.update((row["IID1"], row["IID2"]))

    rows: list[dict[str, object]] = []
    for iid, ancestry_row in ancestry.items():
        het_row = het.get(iid, {})
        sex_row = sex.get(iid, {})
        f_value = float(het_row["F"]) if het_row.get("F") not in {None, "NA", "nan"} else math.nan
        f_z = (f_value - f_mean) / f_sd if math.isfinite(f_value) and f_sd > 0 else math.nan
        pedigree_sex = sex_row.get("PEDSEX", "NA")
        sex_missing = pedigree_sex not in {"1", "2"}
        sex_problem = not sex_missing and sex_row.get("STATUS", "NA") == "PROBLEM"
        rows.append(
            {
                "FID": ancestry_row.get("FID", "0"),
                "IID": iid,
                "assigned_superpop": ancestry_row["assigned_superpop"],
                "EUR_compatible": ancestry_row["EUR_compatible"],
                "heterozygosity_F": f_value,
                "heterozygosity_z": f_z,
                "heterozygosity_outlier": math.isfinite(f_z) and abs(f_z) > 3,
                "pedigree_sex_missing": sex_missing,
                "sexcheck_status": sex_row.get("STATUS", "NA"),
                "sexcheck_problem": sex_problem,
                "related_pair_member": iid in related_ids,
            }
        )
    fields = tuple(rows[0]) if rows else (
        "FID",
        "IID",
        "assigned_superpop",
        "EUR_compatible",
        "heterozygosity_F",
        "heterozygosity_z",
        "heterozygosity_outlier",
        "pedigree_sex_missing",
        "sexcheck_status",
        "sexcheck_problem",
        "related_pair_member",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {
        "samples": len(rows),
        "eur_compatible": sum(str(row["EUR_compatible"]).casefold() == "true" for row in rows),
        "heterozygosity_outliers": sum(bool(row["heterozygosity_outlier"]) for row in rows),
        "pedigree_sex_missing": sum(bool(row["pedigree_sex_missing"]) for row in rows),
        "sexcheck_problems": sum(bool(row["sexcheck_problem"]) for row in rows),
        "related_pairs_kinship_ge_0.0884": related_pairs,
        "related_pair_members": len(related_ids),
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
