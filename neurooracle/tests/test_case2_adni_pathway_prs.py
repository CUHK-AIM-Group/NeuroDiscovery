from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from neurooracle.scripts.compute_case2_adni_pathway_prs import (
    build_pathway_catalog,
    load_gene_coordinates,
    map_variants_to_pathways,
    parse_gtf_attributes,
    select_reactome_pathways,
    stable_slug,
)


def test_gtf_parser_and_slug(tmp_path: Path) -> None:
    gtf = tmp_path / "genes.gtf.gz"
    with gzip.open(gtf, "wt", encoding="utf-8") as handle:
        handle.write(
            '1\ttest\tgene\t100\t200\t.\t+\t.\t'
            'gene_id "ENSG1"; gene_name "APOE"; '
            'gene_biotype "protein_coding";\n'
        )
        handle.write(
            'X\ttest\tgene\t300\t400\t.\t+\t.\t'
            'gene_id "ENSG2"; gene_name "XGENE";\n'
        )

    attributes = parse_gtf_attributes(
        'gene_id "ENSG1"; gene_name "APOE";'
    )
    coordinates = load_gene_coordinates(gtf)

    assert attributes["gene_name"] == "APOE"
    assert coordinates["gene_symbol"].tolist() == ["APOE"]
    assert stable_slug("R-HSA: Amyloid / processing") == (
        "r_hsa_amyloid_processing"
    )


def test_reactome_selection_and_catalog() -> None:
    pathways = [
        {
            "pathway_name": "AD mechanism",
            "reference_id": "R-HSA-1",
            "genes": {
                "APOE",
                "BIN1",
                "CR1",
                "GENE4",
                "GENE5",
                "GENE6",
                "GENE7",
                "GENE8",
                "GENE9",
                "GENE10",
            },
        },
        {
            "pathway_name": "Unrelated",
            "reference_id": "R-HSA-2",
            "genes": {f"OTHER{index}" for index in range(20)},
        },
    ]
    selected = select_reactome_pathways(
        pathways,
        {"APOE", "BIN1", "CR1"},
        top_n=5,
    )
    catalog, gene_sets = build_pathway_catalog(selected)

    assert selected["reference_id"].tolist() == ["R-HSA-1"]
    assert "reactome__r_hsa_1" in gene_sets
    assert "curated__ad_risk_gwas" in set(catalog["pathway_id"])


def test_variant_to_pathway_mapping_uses_gene_window(
    tmp_path: Path,
) -> None:
    pvar = tmp_path / "tiny.pvar"
    pvar.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n"
        "1\t85\trsNear\tA\tG\n"
        "1\t150\trsInside\tC\tT\n"
        "1\t500\trsFar\tA\tC\n",
        encoding="utf-8",
    )
    coordinates = pd.DataFrame(
        {
            "chrom": ["1"],
            "start": [100],
            "end": [200],
            "gene_symbol": ["APOE"],
            "gene_id": ["ENSG1"],
            "gene_biotype": ["protein_coding"],
        }
    )

    pathway_map, gene_map = map_variants_to_pathways(
        pvar,
        coordinates,
        {"pathway_a": {"APOE"}},
        window_bp=20,
    )

    assert set(pathway_map) == {"rsNear", "rsInside"}
    assert pathway_map["rsNear"] == {"pathway_a"}
    assert gene_map["rsInside"] == {"APOE"}

# Last Updated At: 2026-07-31 02:48 HKT
