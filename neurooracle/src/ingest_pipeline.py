"""Unified ingestion pipeline for building the neuroscience knowledge graph.

Usage:
    python -m neurooracle.phase1 --data-dir ./data/raw --output ./data/knowledge_graph.json

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
from .ingestion.atc_drugs import ingest_atc_drugs
from .ingestion.brainmap import ingest_brainmap
from .ingestion.clinical_outcomes import ingest_clinical_outcomes
from .ingestion.cognitive_atlas import ingest_cognitive_atlas
from .ingestion.dataset_variables import ingest_dataset_variables
from .ingestion.disgenet import ingest_disgenet
from .ingestion.experiment_infra import ingest_experiment_infrastructure
from .ingestion.individual_data_anchors import ingest_individual_data_anchors
from .ingestion.individual_data_bridges import ingest_individual_data_bridges
from .ingestion.mesh import ingest_mesh
from .ingestion.neuronames import ingest_neuronames
from .ingestion.outcome_bridges import ingest_outcome_bridges
from .ingestion.outcome_im_bridges import ingest_outcome_im_bridges
from .ingestion.visual_functional_roi import ingest_visual_functional_roi
from .ingestion.visual_stimulus import ingest_visual_stimuli
from .storage import save_graph

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
                   "experiment_infra", "visual_functional_roi", "visual_stimulus",
                   "clinical_outcomes", "dataset_variables", "atc_drugs",
                   "outcome_bridges", "individual_data_anchors",
                   "individual_data_bridges", "outcome_im_bridges"]
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

    if "visual_stimulus" in sources:
        logger.info("=== Ingesting Visual Stimulus Taxonomy (COCO/Places/SEED-DV) ===")
        results["visual_stimulus"] = ingest_visual_stimuli(kg)

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

    stats = kg.stats()
    logger.info(f"\n{'='*50}")
    logger.info(f"INGESTION COMPLETE")
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
                 "experiment_infra", "visual_functional_roi", "visual_stimulus",
                 "clinical_outcomes", "dataset_variables", "atc_drugs",
                 "outcome_bridges", "individual_data_anchors",
                 "individual_data_bridges", "outcome_im_bridges"],
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
