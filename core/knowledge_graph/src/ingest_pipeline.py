"""Unified ingestion pipeline for building the neuroscience knowledge graph.

Usage:
    python -m core.knowledge_graph.phase1 --data-dir ./data/raw --output ./data/knowledge_graph.json

    Or programmatically:
        from core.knowledge_graph.phase1 import run_full_ingestion
        kg = run_full_ingestion(data_dir=Path("./data/raw"))
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .graph_manager import KnowledgeGraph
from .ingestion.brainmap import ingest_brainmap
from .ingestion.cognitive_atlas import ingest_cognitive_atlas
from .ingestion.disgenet import ingest_disgenet
from .ingestion.experiment_infra import ingest_experiment_infrastructure
from .ingestion.mesh import ingest_mesh
from .ingestion.neuronames import ingest_neuronames
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
                   "experiment_infra", "visual_functional_roi", "visual_stimulus"]
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
                 "experiment_infra", "visual_functional_roi", "visual_stimulus"],
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
