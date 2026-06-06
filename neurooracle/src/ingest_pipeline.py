"""Unified ingestion pipeline for building the neuroscience knowledge graph.

Usage:
    python -m neurooracle.phase1 --data-dir ./data/raw --output ./data/full_snapshot_v2/knowledge_graph.json

    Or programmatically:
        from neurooracle.phase1 import run_full_ingestion
        kg = run_full_ingestion(data_dir=Path("./data/raw"))
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .graph_manager import KnowledgeGraph
from .ingestion.ahba_gene_expression import ingest_ahba_gene_expression
from .ingestion.atc_drugs import ingest_atc_drugs
from .ingestion.atlas_roi_modality import ingest_atlas_roi_modality
from .ingestion.brainmap import ingest_brainmap
from .ingestion.clinical_outcomes import ingest_clinical_outcomes
from .ingestion.cognitive_atlas import ingest_cognitive_atlas
from .ingestion.dataset_variables import ingest_dataset_variables
from .ingestion.disgenet import ingest_disgenet
from .ingestion.drug_receptor_binding import ingest_drug_receptor_binding
from .ingestion.enigma_disease_im import ingest_enigma_disease_im
from .ingestion.experiment_infra import ingest_experiment_infrastructure
from .ingestion.hansen_receptor_density import ingest_hansen_receptor_density
from .ingestion.hpo_gene_anatomy import ingest_hpo_gene_anatomy
from .ingestion.imaging_genetic_markers import ingest_imaging_genetic_markers
from .ingestion.individual_data_anchors import ingest_individual_data_anchors
from .ingestion.individual_data_bridges import ingest_individual_data_bridges
from .ingestion.mesh import ingest_mesh
from .ingestion.neuronames import ingest_neuronames
from .ingestion.neurosynth_task_im import ingest_neurosynth_task_im
from .ingestion.outcome_bridges import ingest_outcome_bridges
from .ingestion.outcome_im_bridges import ingest_outcome_im_bridges
from .ingestion.visual_functional_roi import ingest_visual_functional_roi
from .storage import save_graph
from .umls_canonicalize import canonicalize_kg

logger = logging.getLogger(__name__)


def run_full_ingestion(
    data_dir: Path,
    output_path: Path | None = None,
    sources: list[str] | None = None,
) -> KnowledgeGraph:
    """Run the full ingestion pipeline.

    Args:
        data_dir: Directory containing raw data files.
        output_path: Where to save the graph JSON. Defaults to data/knowledge_graph.json.
        sources: Which sources to ingest. None = all available.
                 Options: 'neuronames', 'mesh', 'disgenet', 'brainmap'.

    Returns:
        Populated KnowledgeGraph.
    """
    data_dir = Path(data_dir)
    kg = KnowledgeGraph()
    results = {}

    all_sources = ["neuronames", "mesh", "disgenet", "brainmap", "cognitive_atlas",
                   "experiment_infra", "visual_functional_roi",
                   "clinical_outcomes", "dataset_variables", "atc_drugs",
                   "outcome_bridges", "individual_data_anchors",
                   "individual_data_bridges", "outcome_im_bridges",
                   "hpo_gene_anatomy", "ahba_gene_expression",
                   "enigma_disease_im", "atlas_roi_modality",
                   "hansen_receptor_density", "drug_receptor_binding",
                   "neurosynth_task_im", "imaging_genetic_markers"]
    if sources is None:
        sources = all_sources

    if "neuronames" in sources:
        logger.info("=== Ingesting NeuroNames ===")
        results["neuronames"] = ingest_neuronames(kg, data_dir / "neuronames.tsv")

    if "mesh" in sources:
        logger.info("=== Ingesting MeSH ===")
        results["mesh"] = ingest_mesh(kg, data_dir)

    if "disgenet" in sources:
        logger.info("=== Ingesting DisGeNET ===")
        results["disgenet"] = ingest_disgenet(kg, data_dir / "disgenet_gene_disease.tsv")

    if "brainmap" in sources:
        logger.info("=== Ingesting BrainMap ===")
        results["brainmap"] = ingest_brainmap(
            kg,
            sleuth_file=data_dir / "brainmap_sleuth.txt",
            taxonomy_file=data_dir / "brainmap_taxonomy.csv",
        )

    if "cognitive_atlas" in sources:
        logger.info("=== Ingesting Cognitive Atlas ===")
        results["cognitive_atlas"] = ingest_cognitive_atlas(
            kg, data_dir, download=True,
        )

    if "experiment_infra" in sources:
        logger.info("=== Ingesting Experiment Infrastructure (atlases/modalities/models/datasets) ===")
        results["experiment_infra"] = ingest_experiment_infrastructure(kg)

    if "visual_functional_roi" in sources:
        logger.info("=== Ingesting Visual Functional ROIs (FFA/PPA/EBA/VWFA/LOC/MT+/V3/V4) ===")
        results["visual_functional_roi"] = ingest_visual_functional_roi(kg)

    if "clinical_outcomes" in sources:
        logger.info("=== Ingesting Clinical Outcomes (rating scales + MedDRA SOC) ===")
        results["clinical_outcomes"] = ingest_clinical_outcomes(kg)

    if "dataset_variables" in sources:
        logger.info("=== Ingesting Dataset Variables (UKB/ADNI/HCP-YA categories) ===")
        results["dataset_variables"] = ingest_dataset_variables(kg)

    if "atc_drugs" in sources:
        logger.info("=== Ingesting ATC Neuropsychiatric Drugs (N03-N07) ===")
        results["atc_drugs"] = ingest_atc_drugs(kg)

    if "outcome_bridges" in sources:
        logger.info("=== Ingesting Outcome / Dataset-Variable Bridges ===")
        results["outcome_bridges"] = ingest_outcome_bridges(kg)

    if "individual_data_anchors" in sources:
        logger.info("=== Ingesting INDIVIDUAL_DATA Anchors (Aging / APOE / Big-5 / lifestyle) ===")
        results["individual_data_anchors"] = ingest_individual_data_anchors(kg)

    if "individual_data_bridges" in sources:
        logger.info("=== Ingesting INDIVIDUAL_DATA Bridges (anchor↔dataset, IM↔anchor) ===")
        results["individual_data_bridges"] = ingest_individual_data_bridges(kg)

    if "outcome_im_bridges" in sources:
        logger.info("=== Ingesting OUTCOME-IM Bridges (IM→scale, disease→scale, drug→AE) ===")
        results["outcome_im_bridges"] = ingest_outcome_im_bridges(kg)

    if "hpo_gene_anatomy" in sources:
        logger.info("=== Ingesting HPO Gene -> Brain-Region Edges (GENE -> IM layer 1) ===")
        results["hpo_gene_anatomy"] = ingest_hpo_gene_anatomy(
            kg, data_dir / "hpo_genes_to_phenotype.txt",
        )

    if "ahba_gene_expression" in sources:
        logger.info("=== Ingesting AHBA Gene Expression -> Brain-Region Edges (GENE -> IM layer 2) ===")
        results["ahba_gene_expression"] = ingest_ahba_gene_expression(
            kg,
            ahba_dir=data_dir / "abagen-data" / "microarray",
            hgnc_file=data_dir / "hgnc_complete_set.txt",
        )

    if "enigma_disease_im" in sources:
        logger.info("=== Ingesting ENIGMA Toolbox Disease -> Brain-Region Edges (DISEASE -> IM layer 1) ===")
        results["enigma_disease_im"] = ingest_enigma_disease_im(kg)

    if "atlas_roi_modality" in sources:
        logger.info("=== Ingesting Atlas -> ROI and Imaging-Feature -> Modality bridges ===")
        results["atlas_roi_modality"] = ingest_atlas_roi_modality(kg)

    if "hansen_receptor_density" in sources:
        logger.info("=== Ingesting Hansen 2022 Receptor Density (GENE -> NN region; PET-derived) ===")
        results["hansen_receptor_density"] = ingest_hansen_receptor_density(
            kg, raw_dir=data_dir / "hansen_receptors",
        )

    if "drug_receptor_binding" in sources:
        logger.info("=== Ingesting Curated Drug -> Receptor Pharmacology (DRUG -> GENE binds_to) ===")
        results["drug_receptor_binding"] = ingest_drug_receptor_binding(kg)

    if "neurosynth_task_im" in sources:
        logger.info("=== Ingesting Neurosynth v0.7 task -> region forward inference (COGAT -> NN activates) ===")
        results["neurosynth_task_im"] = ingest_neurosynth_task_im(
            kg,
            dataset_path=data_dir / "neurosynth" / "neurosynth_dataset.pkl.gz",
        )

    if "imaging_genetic_markers" in sources:
        logger.info("=== Ingesting IM/GM/GENESET nodes (NeuroClaw marker registries) ===")
        results["imaging_genetic_markers"] = ingest_imaging_genetic_markers(kg)

    stats = kg.stats()
    logger.info(f"\n{'='*50}")
    logger.info(f"INGESTION COMPLETE (pre-UMLS)")
    logger.info(f"  Total concepts: {stats['n_concepts']}")
    logger.info(f"  Total edges: {stats['n_edges']}")

    # Final step: canonicalize node ids to UMLS CUIs where possible.
    # Cross-vocab duplicates (e.g. NN:1 + MSH:D009420 + DISGENET:Cxxx for
    # the same biological entity) collapse into a single CUI:Cxxxxxxx node;
    # research-only entities (atlas/modality/model/dataset/IF/VROI/COGAT
    # task/concept/dataset-variable hosts) keep their typed prefixes.
    logger.info(f"\n=== UMLS canonicalization ===")
    canon_summary = canonicalize_kg(kg)
    results["umls_canonicalize"] = canon_summary

    stats = kg.stats()
    logger.info(f"\n{'='*50}")
    logger.info(f"INGESTION COMPLETE (post-UMLS)")
    logger.info(f"  Total concepts: {stats['n_concepts']}")
    logger.info(f"  Total edges: {stats['n_edges']}")
    logger.info(f"  Domains: {stats['domains']}")
    logger.info(f"  Sources: {stats['sources']}")
    logger.info(f"  Relations: {stats['relations']}")
    logger.info(f"  Connected components: {stats['connected_components']}")

    if output_path:
        save_graph(kg, output_path)
    else:
        save_graph(kg)

    return kg


def main():
    parser = argparse.ArgumentParser(
        description="Build neuroscience knowledge graph from raw data sources"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/knowledge_graph.json)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["neuronames", "mesh", "disgenet", "brainmap", "cognitive_atlas",
                 "experiment_infra", "visual_functional_roi",
                 "clinical_outcomes", "dataset_variables", "atc_drugs",
                 "outcome_bridges", "individual_data_anchors",
                 "individual_data_bridges", "outcome_im_bridges",
                 "hpo_gene_anatomy", "ahba_gene_expression",
                 "enigma_disease_im", "atlas_roi_modality",
                 "hansen_receptor_density", "drug_receptor_binding",
                 "neurosynth_task_im", "imaging_genetic_markers"],
        default=None,
        help="Which sources to ingest (default: all available)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_full_ingestion(args.data_dir, args.output, args.sources)


if __name__ == "__main__":
    main()
