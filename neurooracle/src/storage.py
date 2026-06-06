"""JSON serialization and deserialization for the knowledge graph."""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode, DISPLAY_TIERS_DEFAULT, Edge

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "full_snapshot_v2" / "knowledge_graph.json"


def _resolve_read_path(path: Path) -> Path:
    """If `path` doesn't exist but `path.gz` does, return the gz variant."""
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gz
    return path


def _open_for_read(path: Path):
    """Open a JSON file for text read, transparently handling .gz."""
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _open_for_write(path: Path):
    """Open a JSON file for text write, transparently handling .gz (level 9)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "wb", compresslevel=9), encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def save_graph(kg: KnowledgeGraph, path: Optional[Path] = None) -> Path:
    """Save knowledge graph to JSON file. Compresses transparently if path ends with .gz.

    Atomic: writes to ``<path>.tmp`` then os.replace()s into place. A SIGTERM
    or crash mid-write leaves the previous good file untouched.
    """
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

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with _open_for_write(tmp_path) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    logger.info(f"saved graph to {path}: {kg.stats()['n_concepts']} concepts, {kg.stats()['n_edges']} edges")
    return path


def load_graph(path: Optional[Path] = None) -> KnowledgeGraph:
    """Load knowledge graph from JSON file. Auto-detects .gz fallback."""
    path = Path(path) if path else DEFAULT_PATH
    path = _resolve_read_path(path)
    if not path.exists():
        logger.info(f"no graph file at {path}, returning empty graph")
        return KnowledgeGraph()

    with _open_for_read(path) as f:
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


def save_display_graph(
    kg: KnowledgeGraph,
    path: Path,
    tiers: Optional[set[str]] = None,
) -> Path:
    """Save the display-tier subgraph to JSON, for HF Space / public consumption.

    Drops provenance / inverse / bridge edges and orphaned nodes — see
    `KnowledgeGraph.export_display_subgraph`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sub = kg.export_display_subgraph(tiers=tiers)
    keep_ids = set(sub.nodes())

    edges = []
    for src, tgt, edata in sub.edges(data=True):
        edge_dict = dict(edata)
        edge_dict["source_id"] = src
        edge_dict["target_id"] = tgt
        edges.append(edge_dict)

    data = {
        "metadata": {
            "version": "0.1-display",
            "created": datetime.now().isoformat(),
            "tiers": sorted(tiers if tiers is not None else DISPLAY_TIERS_DEFAULT),
            "n_concepts": len(keep_ids),
            "n_edges": len(edges),
        },
        "concepts": {
            nid: node.to_dict()
            for nid, node in kg._index.items()
            if nid in keep_ids
        },
        "edges": edges,
    }

    with _open_for_write(path) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"saved display graph to {path}: {len(keep_ids)} concepts, {len(edges)} edges")
    return path
