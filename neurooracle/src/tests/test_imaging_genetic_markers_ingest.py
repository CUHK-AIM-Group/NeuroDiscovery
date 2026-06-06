from __future__ import annotations

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.ingestion.imaging_genetic_markers import (
    _build_name_index,
    _inject_gms_and_genesets,
)
from neurooracle.src.schema import ConceptNode


def _node(
    cid: str,
    name: str,
    domain: str,
    source_vocab: str = "",
) -> ConceptNode:
    return ConceptNode(
        id=cid,
        preferred_name=name,
        domain_tags=[domain],
        source_vocab=source_vocab,
    )


def test_polygenic_risk_trait_falls_back_to_prs_anchor():
    kg = KnowledgeGraph()
    kg.add_concept(_node("CUI:PRS", "Polygenic Risk Score", "individual_data_anchor", "IndividualDataAnchor"))

    gms = [{
        "id": "gm_0014",
        "name": "cp_prs_geno",
        "family": "polygenic_risk",
        "operation": "polygenic_risk",
        "data_type": "genotype_array",
        "gene_symbols": [],
        "gene_ids": [],
        "gene_set": None,
        "tissue": None,
        "tissues": [],
        "clock": None,
        "disease": "Cognitive Performance",
        "formula": "polygenic_risk on genotype_array using Cognitive Performance GWAS weights",
        "rationale": "Summarizes common-variant propensity for cognitive performance.",
        "atoms": ["GENE_TARGET", "DISEASE"],
        "llm_model": "gpt-5.4",
    }]

    counts = _inject_gms_and_genesets(kg, gms, _build_name_index(kg))

    assert counts["gm_nodes_added"] == 1
    assert counts["gm_to_prs_anchor"] == 1
    assert kg.G.has_edge("GM:gm_0014", "CUI:PRS")
    assert kg.G["GM:gm_0014"]["CUI:PRS"]["relation_type"] == "is_a"


def test_polygenic_risk_prefers_trait_anchor_when_available():
    kg = KnowledgeGraph()
    kg.add_concept(_node("CUI:PRS", "Polygenic Risk Score", "individual_data_anchor", "IndividualDataAnchor"))
    kg.add_concept(_node("CUI:EDU", "Educational Attainment", "individual_data_anchor", "IndividualDataAnchor"))

    gms = [{
        "id": "gm_0099",
        "name": "ea_prs_geno",
        "family": "polygenic_risk",
        "operation": "polygenic_risk",
        "data_type": "genotype_array",
        "gene_symbols": [],
        "gene_ids": [],
        "gene_set": None,
        "tissue": None,
        "tissues": [],
        "clock": None,
        "disease": "Educational Attainment",
        "formula": "polygenic_risk on genotype_array using Educational Attainment GWAS weights",
        "rationale": "Summarizes common-variant propensity for educational attainment.",
        "atoms": ["GENE_TARGET", "DISEASE"],
        "llm_model": "gpt-5.4",
    }]

    counts = _inject_gms_and_genesets(kg, gms, _build_name_index(kg))

    assert counts["gm_to_trait"] == 1
    assert counts["gm_to_prs_anchor"] == 1
    assert kg.G.has_edge("GM:gm_0099", "CUI:EDU")
    assert kg.G["GM:gm_0099"]["CUI:EDU"]["relation_type"] == "is_associated_with"
