"""JSON serialization and deserialization for the knowledge graph."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode, DISPLAY_TIERS_DEFAULT, Edge
from .kg_metadata_compaction import compact_layout_enabled, compact_record
from .correlation_grouping import enabled as correlation_grouping_enabled

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
    graph_metadata = deepcopy(kg.serialization_metadata)
    # This export is not the independently validated campaign snapshot. A
    # relative, hash-bound sidecar from the input must never look current here.
    graph_metadata.pop("relation_evidence", None)
    graph_metadata.pop("entity_identity", None)
    graph_metadata.pop("paper_identity", None)
    use_compact = compact_layout_enabled(graph_metadata)
    correlation_grouping_enabled(graph_metadata)

    edges = []
    for edge_dict in kg.iter_edge_records():
        edges.append(compact_record("edge", edge_dict) if use_compact else edge_dict)

    graph_metadata.update(version="0.1", created=datetime.now().isoformat(), stats=kg.stats())
    graph_metadata["stats"]["n_stored_edge_records"] = len(edges)
    data = {
        "metadata": graph_metadata,
        "concepts": {nid: compact_record("node", node.to_dict()) if use_compact else node.to_dict()
                     for nid, node in kg._index.items()},
        "edges": edges,
    }

    if kg.relation_identities:
        from .verified_entity_terms import VERSION
        registry = kg.relation_identities.export_payload(kg, data["concepts"], edges)
        raw = json.dumps(registry, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        signature = hashlib.sha256(raw).hexdigest()
        # Content-addressed sidecar never overwrites the registry of a still
        # current graph, and repeated ordinary ingestions reuse the same file.
        sidecar = path.with_name(path.name + ".entity_terms." + signature[:16] + ".json")
        if sidecar.exists():
            if sidecar.read_bytes() != raw: raise ValueError("identity sidecar content conflict")
        else:
            with sidecar.open("xb") as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        graph_metadata["entity_identity"] = dict(version=VERSION, registry=sidecar.name, sha256=signature)

    if kg.paper_identities:
        from .kg_paper_identity import VERSION as PAPER_VERSION
        raw = json.dumps(kg.paper_identities.export_payload(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        signature = hashlib.sha256(raw).hexdigest()
        sidecar = path.with_name(path.name + ".paper_identities." + signature[:16] + ".json")
        if sidecar.exists():
            if sidecar.read_bytes() != raw: raise ValueError("paper identity sidecar content conflict")
        else:
            with sidecar.open("xb") as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        graph_metadata["paper_identity"] = dict(version=PAPER_VERSION, registry=sidecar.name, sha256=signature)

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

    from .verified_entity_terms import load_graph_terms
    identities = load_graph_terms(path, data)
    from .kg_paper_identity import load_graph_papers
    paper_identities = load_graph_papers(path, data.get("metadata") or {})

    kg = KnowledgeGraph()
    kg.paper_identities = paper_identities
    kg.serialization_metadata = deepcopy(data.get("metadata") or {})
    compact_layout_enabled(kg.serialization_metadata)  # fail closed on unknown layouts
    correlation_grouping_enabled(kg.serialization_metadata)

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

    if identities:
        identities.bind_runtime(kg)
        kg.relation_identities = identities
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
    graph_metadata = deepcopy(kg.serialization_metadata)
    graph_metadata.pop("relation_evidence", None)
    graph_metadata.pop("entity_identity", None)  # display projection omits identity proof nodes/edges
    graph_metadata.pop("paper_identity", None)
    use_compact = compact_layout_enabled(graph_metadata)
    correlation_grouping_enabled(graph_metadata)

    sub = kg.export_display_subgraph(tiers=tiers)
    keep_ids = set(sub.nodes())

    edges = []
    for src, tgt, edata in sub.edges(data=True):
        edge_dict = dict(edata)
        edge_dict["source_id"] = src
        edge_dict["target_id"] = tgt
        edges.append(compact_record("edge", edge_dict) if use_compact else edge_dict)

    graph_metadata.update(version="0.1-display", created=datetime.now().isoformat(),
                          tiers=sorted(tiers if tiers is not None else DISPLAY_TIERS_DEFAULT),
                          n_concepts=len(keep_ids), n_edges=len(edges))
    data = {
        "metadata": graph_metadata,
        "concepts": {
            nid: compact_record("node", node.to_dict()) if use_compact else node.to_dict()
            for nid, node in kg._index.items()
            if nid in keep_ids
        },
        "edges": edges,
    }

    with _open_for_write(path) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"saved display graph to {path}: {len(keep_ids)} concepts, {len(edges)} edges")
    return path
