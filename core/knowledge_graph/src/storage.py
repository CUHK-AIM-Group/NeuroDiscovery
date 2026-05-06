"""JSON serialization and deserialization for the knowledge graph."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode, Edge

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "knowledge_graph.json"


def save_graph(kg: KnowledgeGraph, path: Optional[Path] = None) -> Path:
    """Save knowledge graph to JSON file."""
    path = Path(path) if path else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    edges = []
    for src, tgt, edata in kg.G.edges(data=True):
        # Ensure source_id and target_id are always present
        edge_dict = dict(edata)
        edge_dict["source_id"] = src
        edge_dict["target_id"] = tgt
        edges.append(edge_dict)

    data = {
        "metadata": {
            "version": "0.1",
            "created": datetime.now().isoformat(),
            "stats": kg.stats(),
        },
        "concepts": {nid: node.to_dict() for nid, node in kg._index.items()},
        "edges": edges,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"saved graph to {path}: {kg.stats()['n_concepts']} concepts, {kg.stats()['n_edges']} edges")
    return path


def load_graph(path: Optional[Path] = None) -> KnowledgeGraph:
    """Load knowledge graph from JSON file."""
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        logger.info(f"no graph file at {path}, returning empty graph")
        return KnowledgeGraph()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kg = KnowledgeGraph()

    for nid, ndata in data.get("concepts", {}).items():
        node = ConceptNode.from_dict(ndata)
        kg.add_concept(node)

    for edata in data.get("edges", []):
        try:
            edge = Edge.from_dict(edata)
            kg.add_edge(edge)
        except (TypeError, KeyError) as e:
            logger.warning(f"skipping malformed edge: {e}")
            continue

    stats = kg.stats()
    logger.info(f"loaded graph from {path}: {stats['n_concepts']} concepts, {stats['n_edges']} edges")
    return kg
