"""Compute pathway-partitioned Alzheimer disease PRS for ADNI Case Study 2."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from neurooracle.scripts.build_case2_adni_experiment_table import (
    normalize_subject_id,
)
from neurooracle.scripts.prepare_case2_adni_ancestry import PLINK2
from neurooracle.src.recipe.genetic_marker import CURATED_GENE_SETS


CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
POSTIMPUTATION_ROOT = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
)
DEFAULT_MERGED_PREFIX = (
    POSTIMPUTATION_ROOT
    / "merged"
    / "case2_adni_common_dr2_0p8_autosomes"
)
DEFAULT_GTF = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\references")
    / "ensembl"
    / "GRCh37.87"
    / "Homo_sapiens.GRCh37.87.chr.gtf.gz"
)
DEFAULT_REACTOME = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\references")
    / "reactome"
    / "v97"
    / "ReactomePathways.gmt"
)
DEFAULT_OUTPUT_ROOT = (
    POSTIMPUTATION_ROOT
    / "features"
    / "ad_pathway_prs_v1"
)
GLOBAL_SCORE_FILES = {
    "p1em03": (
        POSTIMPUTATION_ROOT
        / "features"
        / "ad_prs_bellenguez"
        / "score_files"
        / "bellenguez_ad_p1em03.score.tsv"
    ),
    "p5em02": (
        POSTIMPUTATION_ROOT
        / "features"
        / "ad_prs_bellenguez"
        / "score_files"
        / "bellenguez_ad_p5em02.score.tsv"
    ),
}
CURATED_PATHWAYS = (
    "AD_risk_GWAS",
    "Mendelian_AD",
    "Synaptic",
    "Cholinergic",
    "Microglia_immune",
    "Myelin",
    "GABA_glutamate",
)
GTF_ATTRIBUTE_PATTERN = re.compile(r'(\S+)\s+"([^"]*)"')


def stable_slug(value: str, max_length: int = 48) -> str:
    """Return a deterministic filesystem- and header-safe identifier."""

    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{normalized[: max_length - 9]}_{digest}"


def parse_gtf_attributes(value: str) -> dict[str, str]:
    return dict(GTF_ATTRIBUTE_PATTERN.findall(value))


def load_gene_coordinates(gtf_path: Path) -> pd.DataFrame:
    """Read autosomal gene spans and HGNC-like symbols from Ensembl GTF."""

    if not gtf_path.is_file():
        raise FileNotFoundError(gtf_path)
    rows: list[dict[str, object]] = []
    with gzip.open(gtf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chrom = fields[0].removeprefix("chr")
            if not chrom.isdigit() or not 1 <= int(chrom) <= 22:
                continue
            attributes = parse_gtf_attributes(fields[8])
            gene_name = (attributes.get("gene_name") or "").strip()
            if not gene_name:
                continue
            rows.append(
                {
                    "chrom": str(int(chrom)),
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "gene_symbol": gene_name.upper(),
                    "gene_id": attributes.get("gene_id", ""),
                    "gene_biotype": (
                        attributes.get("gene_biotype")
                        or attributes.get("gene_type")
                        or ""
                    ),
                }
            )
    coordinates = pd.DataFrame(rows)
    if coordinates.empty:
        raise ValueError(f"No autosomal genes parsed from {gtf_path}")
    return (
        coordinates.sort_values(
            ["chrom", "start", "end", "gene_symbol"],
            kind="stable",
        )
        .drop_duplicates(["chrom", "start", "end", "gene_symbol"])
        .reset_index(drop=True)
    )


def load_reactome_pathways(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    pathways = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            pathways.append(
                {
                    "pathway_name": fields[0],
                    "reference_id": fields[1],
                    "genes": {
                        value.strip().upper()
                        for value in fields[2:]
                        if value.strip()
                    },
                }
            )
    return pathways


def selected_seed_genes() -> set[str]:
    genes: set[str] = set()
    for name in CURATED_PATHWAYS:
        genes.update(
            str(gene).upper()
            for gene in CURATED_GENE_SETS[name]["members"]
        )
    return genes


def select_reactome_pathways(
    pathways: Iterable[dict[str, object]],
    seed_genes: set[str],
    *,
    top_n: int,
    min_overlap: int = 2,
    min_genes: int = 10,
    max_genes: int = 500,
) -> pd.DataFrame:
    """Select Reactome pathways by seed-gene enrichment without outcome data."""

    pathways = list(pathways)
    universe = set().union(
        *(set(pathway["genes"]) for pathway in pathways)
    )
    seeds = seed_genes & universe
    rows = []
    for pathway in pathways:
        genes = set(pathway["genes"]) & universe
        overlap = genes & seeds
        if (
            len(overlap) < min_overlap
            or len(genes) < min_genes
            or len(genes) > max_genes
        ):
            continue
        p_value = float(
            hypergeom.sf(
                len(overlap) - 1,
                len(universe),
                len(seeds),
                len(genes),
            )
        )
        rows.append(
            {
                "pathway_name": pathway["pathway_name"],
                "reference_id": pathway["reference_id"],
                "gene_count": len(genes),
                "seed_overlap_count": len(overlap),
                "seed_overlap": ";".join(sorted(overlap)),
                "enrichment_p": p_value,
                "genes": genes,
            }
        )
    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected["enrichment_q"] = multipletests(
        selected["enrichment_p"],
        method="fdr_bh",
    )[1]
    return (
        selected.sort_values(
            [
                "enrichment_p",
                "seed_overlap_count",
                "gene_count",
                "pathway_name",
            ],
            ascending=[True, False, True, True],
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def build_pathway_catalog(
    reactome_selection: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Combine explicit curated sets with enriched Reactome pathways."""

    rows = []
    gene_sets: dict[str, set[str]] = {}
    for name in CURATED_PATHWAYS:
        pathway_id = f"curated__{stable_slug(name)}"
        genes = {
            str(gene).upper()
            for gene in CURATED_GENE_SETS[name]["members"]
        }
        gene_sets[pathway_id] = genes
        rows.append(
            {
                "pathway_id": pathway_id,
                "pathway_source": "NeuroOracle curated",
                "pathway_name": name,
                "reference_id": "",
                "gene_count": len(genes),
                "seed_overlap_count": len(genes & selected_seed_genes()),
                "seed_overlap": ";".join(sorted(genes & selected_seed_genes())),
                "enrichment_p": np.nan,
                "enrichment_q": np.nan,
                "genes": ";".join(sorted(genes)),
            }
        )
    for row in reactome_selection.itertuples(index=False):
        pathway_id = f"reactome__{stable_slug(str(row.reference_id))}"
        genes = set(row.genes)
        gene_sets[pathway_id] = genes
        rows.append(
            {
                "pathway_id": pathway_id,
                "pathway_source": "Reactome v97",
                "pathway_name": row.pathway_name,
                "reference_id": row.reference_id,
                "gene_count": len(genes),
                "seed_overlap_count": row.seed_overlap_count,
                "seed_overlap": row.seed_overlap,
                "enrichment_p": row.enrichment_p,
                "enrichment_q": row.enrichment_q,
                "genes": ";".join(sorted(genes)),
            }
        )
    return pd.DataFrame(rows), gene_sets


def map_variants_to_pathways(
    pvar_path: Path,
    coordinates: pd.DataFrame,
    gene_sets: dict[str, set[str]],
    *,
    window_bp: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Map PGEN variants to pathways through nearby gene spans."""

    gene_to_pathways: dict[str, set[str]] = defaultdict(set)
    for pathway_id, genes in gene_sets.items():
        for gene in genes:
            gene_to_pathways[gene].add(pathway_id)
    relevant_genes = set(gene_to_pathways)
    intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    relevant_coordinates = coordinates[
        coordinates["gene_symbol"].isin(relevant_genes)
    ]
    for row in relevant_coordinates.itertuples(index=False):
        intervals[str(row.chrom)].append(
            (
                max(1, int(row.start) - window_bp),
                int(row.end) + window_bp,
                str(row.gene_symbol),
            )
        )
    for chrom in intervals:
        intervals[chrom].sort()

    variant_to_pathways: dict[str, set[str]] = {}
    variant_to_genes: dict[str, set[str]] = {}
    current_chrom = ""
    chrom_intervals: list[tuple[int, int, str]] = []
    active: list[tuple[int, str]] = []
    interval_index = 0
    with pvar_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0].removeprefix("chr")
            if chrom != current_chrom:
                current_chrom = chrom
                chrom_intervals = intervals.get(chrom, [])
                active = []
                interval_index = 0
            if not chrom_intervals:
                continue
            position = int(fields[1])
            variant_id = fields[2]
            if not variant_id or variant_id == ".":
                continue
            while (
                interval_index < len(chrom_intervals)
                and chrom_intervals[interval_index][0] <= position
            ):
                _, end, gene = chrom_intervals[interval_index]
                active.append((end, gene))
                interval_index += 1
            active = [
                (end, gene)
                for end, gene in active
                if end >= position
            ]
            if not active:
                continue
            genes = {gene for _, gene in active}
            pathway_ids = set().union(
                *(gene_to_pathways[gene] for gene in genes)
            )
            if pathway_ids:
                variant_to_pathways[variant_id] = pathway_ids
                variant_to_genes[variant_id] = genes
    return variant_to_pathways, variant_to_genes


def write_variant_map(
    output_path: Path,
    variant_to_pathways: dict[str, set[str]],
    variant_to_genes: dict[str, set[str]],
) -> None:
    with gzip.open(
        output_path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["variant_id", "genes", "pathway_ids"])
        for variant_id in sorted(variant_to_pathways):
            writer.writerow(
                [
                    variant_id,
                    ";".join(sorted(variant_to_genes[variant_id])),
                    ";".join(sorted(variant_to_pathways[variant_id])),
                ]
            )


def prepare_partition_score_files(
    global_score_files: dict[str, Path],
    variant_to_pathways: dict[str, set[str]],
    pathway_catalog: pd.DataFrame,
    output_dir: Path,
    *,
    min_variants: int,
) -> pd.DataFrame:
    """Route existing Bellenguez score rows into pathway score files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pathway_lookup = pathway_catalog.set_index("pathway_id").to_dict("index")
    rows = []
    for threshold_label, global_path in global_score_files.items():
        if not global_path.is_file():
            raise FileNotFoundError(global_path)
        handles: dict[str, object] = {}
        writers: dict[str, csv.writer] = {}
        paths: dict[str, Path] = {}
        score_names: dict[str, str] = {}
        counts: dict[str, int] = defaultdict(int)
        try:
            for pathway_id in sorted(pathway_lookup):
                score_name = f"pathway_prs__{pathway_id}__{threshold_label}"
                path = output_dir / f"{score_name}.score.tsv"
                handle = path.open("w", encoding="utf-8", newline="")
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["ID", "A1", score_name])
                handles[pathway_id] = handle
                writers[pathway_id] = writer
                paths[pathway_id] = path
                score_names[pathway_id] = score_name
            seen: set[str] = set()
            with global_path.open(
                "r",
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    variant_id = (row.get("ID") or "").strip()
                    if not variant_id or variant_id in seen:
                        continue
                    seen.add(variant_id)
                    pathway_ids = variant_to_pathways.get(variant_id)
                    if not pathway_ids:
                        continue
                    allele = row.get("A1")
                    beta = row.get("BETA")
                    for pathway_id in pathway_ids:
                        writers[pathway_id].writerow(
                            [variant_id, allele, beta]
                        )
                        counts[pathway_id] += 1
        finally:
            for handle in handles.values():
                handle.close()

        for pathway_id in sorted(pathway_lookup):
            count = counts[pathway_id]
            path = paths[pathway_id]
            if count < min_variants:
                path.unlink(missing_ok=True)
                continue
            metadata = pathway_lookup[pathway_id]
            rows.append(
                {
                    "score_name": score_names[pathway_id],
                    "pathway_id": pathway_id,
                    "pathway_source": metadata["pathway_source"],
                    "pathway_name": metadata["pathway_name"],
                    "reference_id": metadata["reference_id"],
                    "threshold_label": threshold_label,
                    "gene_count": metadata["gene_count"],
                    "variants_in_score_file": count,
                    "score_file": str(path),
                }
            )
    return pd.DataFrame(rows)


def run_plink_score_list(
    merged_prefix: Path,
    score_manifest: pd.DataFrame,
    output_root: Path,
    *,
    threads: int,
) -> Path:
    score_list = output_root / "score_file_list.txt"
    score_list.write_text(
        "\n".join(score_manifest["score_file"].astype(str)) + "\n",
        encoding="utf-8",
    )
    output_prefix = output_root / "case2_adni_pathway_prs"
    command = [
        str(PLINK2),
        "--pfile",
        str(merged_prefix),
        "--score-list",
        str(score_list),
        "1",
        "2",
        "3",
        "header-read",
        "ignore-dup-ids",
        "cols=+scoresums",
        "--threads",
        str(threads),
        "--out",
        str(output_prefix),
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    (output_root / "plink_score.command.log").write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PLINK pathway scoring failed ({result.returncode}); "
            f"see {output_root / 'plink_score.command.log'}"
        )
    return output_prefix.with_suffix(".sscore")


def parse_pathway_scores(
    sscore_path: Path,
    score_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(sscore_path, sep="\t")
    iid_column = "#IID" if "#IID" in scores else "IID"
    score_names = score_manifest["score_name"].astype(str).tolist()
    expected_avg = [f"{name}_AVG" for name in score_names]
    missing = [column for column in expected_avg if column not in scores]
    if missing:
        raise ValueError(
            f"PLINK score output is missing columns: {missing[:5]}"
        )
    wide = scores.loc[:, [iid_column, *expected_avg]].copy()
    wide = wide.rename(columns={iid_column: "iid"})
    wide["subject_id"] = wide["iid"].map(normalize_subject_id)
    wide = wide.rename(
        columns={
            f"{name}_AVG": name
            for name in score_names
        }
    )
    wide = wide.loc[:, ["iid", "subject_id", *score_names]]

    long = wide.melt(
        id_vars=["iid", "subject_id"],
        value_vars=score_names,
        var_name="score_name",
        value_name="pathway_prs_avg",
    )
    long = long.merge(
        score_manifest,
        on="score_name",
        how="left",
        validate="many_to_one",
    )
    return wide, long


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    existing = (
        any(path.is_file() for path in args.output_root.rglob("*"))
        if args.output_root.exists()
        else False
    )
    if existing and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {args.output_root}. "
            "Use a new version or --force."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    coordinates = load_gene_coordinates(args.gtf)
    reactome = load_reactome_pathways(args.reactome)
    selection = select_reactome_pathways(
        reactome,
        selected_seed_genes(),
        top_n=args.reactome_top_n,
        min_overlap=args.reactome_min_overlap,
    )
    pathway_catalog, gene_sets = build_pathway_catalog(selection)
    pathway_catalog.to_csv(
        args.output_root / "pathway_catalog.csv",
        index=False,
    )

    pvar_path = args.merged_prefix.with_suffix(".pvar")
    variant_to_pathways, variant_to_genes = map_variants_to_pathways(
        pvar_path,
        coordinates,
        gene_sets,
        window_bp=args.gene_window_bp,
    )
    write_variant_map(
        args.output_root / "variant_pathway_map.tsv.gz",
        variant_to_pathways,
        variant_to_genes,
    )

    score_manifest = prepare_partition_score_files(
        {
            label: GLOBAL_SCORE_FILES[label]
            for label in args.threshold_label
        },
        variant_to_pathways,
        pathway_catalog,
        args.output_root / "score_files",
        min_variants=args.min_variants,
    )
    if score_manifest.empty:
        raise ValueError("No pathway score files passed the variant minimum")
    score_manifest.to_csv(
        args.output_root / "score_manifest.csv",
        index=False,
    )
    sscore_path = run_plink_score_list(
        args.merged_prefix,
        score_manifest,
        args.output_root,
        threads=args.threads,
    )
    wide, long = parse_pathway_scores(sscore_path, score_manifest)
    wide.to_parquet(
        args.output_root / "case2_adni_pathway_prs_wide.parquet",
        index=False,
        compression="zstd",
    )
    wide.to_csv(
        args.output_root / "case2_adni_pathway_prs_wide.csv",
        index=False,
    )
    long.to_parquet(
        args.output_root / "case2_adni_pathway_prs_long.parquet",
        index=False,
        compression="zstd",
    )

    payload: dict[str, object] = {
        "created_at_hkt": datetime.now(
            timezone(timedelta(hours=8))
        ).isoformat(),
        "method": (
            "Bellenguez 2022 AD GWAS beta-weighted pathway-partitioned PRS; "
            "variants assigned to GRCh37 Ensembl gene spans plus window"
        ),
        "selection_is_outcome_blind": True,
        "merged_prefix": str(args.merged_prefix),
        "samples": int(len(wide)),
        "gtf": str(args.gtf),
        "gtf_sha256": sha256_file(args.gtf),
        "reactome": str(args.reactome),
        "reactome_sha256": sha256_file(args.reactome),
        "gene_window_bp": args.gene_window_bp,
        "curated_pathways": list(CURATED_PATHWAYS),
        "reactome_pathways_selected": int(len(selection)),
        "pathways_total": int(len(pathway_catalog)),
        "mapped_pgen_variants": int(len(variant_to_pathways)),
        "threshold_labels": list(args.threshold_label),
        "score_features": int(len(score_manifest)),
        "minimum_variants_per_score": args.min_variants,
        "outputs": {
            "pathway_catalog": str(
                args.output_root / "pathway_catalog.csv"
            ),
            "variant_pathway_map": str(
                args.output_root / "variant_pathway_map.tsv.gz"
            ),
            "score_manifest": str(
                args.output_root / "score_manifest.csv"
            ),
            "pathway_prs_wide": str(
                args.output_root / "case2_adni_pathway_prs_wide.parquet"
            ),
            "pathway_prs_long": str(
                args.output_root / "case2_adni_pathway_prs_long.parquet"
            ),
            "plink_sscore": str(sscore_path),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-prefix",
        type=Path,
        default=DEFAULT_MERGED_PREFIX,
    )
    parser.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    parser.add_argument(
        "--reactome",
        type=Path,
        default=DEFAULT_REACTOME,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--threshold-label", action="append")
    parser.add_argument("--reactome-top-n", type=int, default=20)
    parser.add_argument("--reactome-min-overlap", type=int, default=2)
    parser.add_argument("--gene-window-bp", type=int, default=10_000)
    parser.add_argument("--min-variants", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.threshold_label = tuple(
        args.threshold_label or ("p1em03", "p5em02")
    )
    unknown = set(args.threshold_label) - set(GLOBAL_SCORE_FILES)
    if unknown:
        parser.error(f"Unknown threshold labels: {sorted(unknown)}")
    return args


def main() -> int:
    run_pipeline(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Last Updated At: 2026-07-31 02:48 HKT
