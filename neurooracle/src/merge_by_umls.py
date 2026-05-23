"""Merge synonymous nodes across vocabularies via UMLS CUI.

Steps:
1. Pre-populate `metadata.umls_cui` for nodes that already encode CUI in their ID
   (e.g. `DISGENET:C0005695`).
2. Stream MRCONSO.RRF once to assign CUIs to remaining concepts (MeSH, NeuroNames,
   Cognitive Atlas, claim-derived) by exact name/alias match.
3. Group nodes sharing the same CUI within compatible domains (disease/gene/etc.)
   and merge each group into a single canonical node:
   - Pick canonical by source-vocab priority (MSH > COGAT > NeuroNames > DisGeNET > CLM)
   - Move aliases / external_ids / domain_tags from duplicates onto canonical
   - Rewrite all incoming/outgoing edges to point at canonical id
   - Drop duplicate nodes

CLI:
    python -m neurooracle.src.merge_by_umls \\
        --graph neurooracle/data/full_snapshot_v1/knowledge_graph.json \\
        --mrconso neurooracle/data/raw/MRCONSO.RRF \\
        --output neurooracle/data/full_snapshot_v1/knowledge_graph.json \\
        --backup
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode, Edge
from .storage import load_graph, save_graph
from .umls_integration import _collect_graph_names, _parse_mrconso_streaming

logger = logging.getLogger(__name__)

# Source-vocab priority for picking the canonical node when merging.
# Higher = preferred. Curated vocabularies win over downloads / claim mints.
SOURCE_PRIORITY = {
    "NeuroNames": 100,
    "MeSH": 95,
    "MSH": 95,
    "Cognitive Atlas": 90,
    "CognitiveAtlas": 90,
    "COGAT": 90,
    "DisGeNET": 80,
    "BrainMap": 75,
    "ExperimentInfra": 70,
    "DOI": 50,
    "PMID": 40,
    "CLM_CONCEPT": 10,
}

# Domains where merging by CUI is safe — nodes in different atom domains
# should never collapse into one even if they share a CUI.
COMPATIBLE_DOMAINS = [
    {"disease"},
    {"gene"},
    {"neuroanatomy"},
    {"cognitive_function", "paradigm"},
    {"drug"},
    {"neurotransmitter"},
    {"biomarker"},
]

CUI_RE = re.compile(r"^C\d{7}$")


def _extract_cui_from_id(node_id: str) -> str | None:
    """If node_id encodes a UMLS CUI (DISGENET:Cxxxxxxx), return it."""
    if ":" not in node_id:
        return None
    suffix = node_id.split(":", 1)[1]
    if CUI_RE.match(suffix):
        return suffix
    return None


def _seed_cuis_from_ids(kg: KnowledgeGraph) -> int:
    """Pre-populate metadata.umls_cui for nodes whose ID already encodes a CUI."""
    seeded = 0
    for nid, node in kg._index.items():
        if node.metadata.get("umls_cui"):
            continue
        cui = _extract_cui_from_id(nid)
        if cui:
            node.metadata["umls_cui"] = cui
            seeded += 1
    logger.info(f"seeded {seeded} CUIs directly from node IDs")
    return seeded


def _align_via_mrconso(kg: KnowledgeGraph, mrconso_path: Path) -> int:
    """Stream MRCONSO and assign metadata.umls_cui to nodes still missing one."""
    # Build name lookup limited to nodes WITHOUT an existing CUI to keep the work scoped.
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for nid, node in kg._index.items():
        if "claim" in node.domain_tags:
            continue
        if node.metadata.get("umls_cui"):
            continue
        for label in [node.preferred_name, *node.aliases]:
            key = (label or "").lower().strip()
            if key:
                name_to_ids[key].append(nid)
    if not name_to_ids:
        logger.info("no nodes need MRCONSO alignment")
        return 0

    matches = _parse_mrconso_streaming(mrconso_path, dict(name_to_ids))
    applied = 0
    for nid, info in matches.items():
        node = kg._index.get(nid)
        if node is None:
            continue
        node.metadata["umls_cui"] = info["cui"]
        applied += 1
    logger.info(f"assigned {applied} CUIs via MRCONSO scan")
    return applied


def _domain_signature(node: ConceptNode) -> frozenset[str]:
    """Pick the compatibility bucket for a node (used to gate merges)."""
    tags = set(node.domain_tags or [])
    for bucket in COMPATIBLE_DOMAINS:
        if tags & bucket:
            return frozenset(bucket)
    return frozenset(tags) or frozenset({"_other"})


def _node_priority(node: ConceptNode) -> tuple:
    """Higher = better candidate to be the canonical node in a merge group."""
    src_score = SOURCE_PRIORITY.get(node.source_vocab, 0)
    has_def = 1 if node.definition else 0
    n_aliases = len(node.aliases)
    return (src_score, has_def, n_aliases, node.id)


def _group_by_cui(kg: KnowledgeGraph) -> dict[tuple[str, frozenset[str]], list[str]]:
    groups: dict[tuple[str, frozenset[str]], list[str]] = defaultdict(list)
    for nid, node in kg._index.items():
        cui = node.metadata.get("umls_cui")
        if not cui:
            continue
        if "claim" in node.domain_tags:
            continue
        groups[(cui, _domain_signature(node))].append(nid)
    return {k: v for k, v in groups.items() if len(v) > 1}


def _merge_groups(kg: KnowledgeGraph, groups: dict) -> dict:
    """Merge each duplicate group into a canonical node, rewriting edges."""
    merged_nodes = 0
    rewritten_edges = 0
    dropped_self_loops = 0
    redirect: dict[str, str] = {}

    for (cui, _bucket), members in groups.items():
        nodes = [kg._index[nid] for nid in members]
        canonical = max(nodes, key=_node_priority)
        canonical_id = canonical.id

        for node in nodes:
            if node.id == canonical_id:
                continue
            redirect[node.id] = canonical_id
            # absorb metadata
            for alias in [node.preferred_name, *node.aliases]:
                if alias and alias not in canonical.aliases and alias != canonical.preferred_name:
                    canonical.aliases.append(alias)
            for tag in node.domain_tags:
                if tag not in canonical.domain_tags:
                    canonical.domain_tags.append(tag)
            for st in node.semantic_types:
                if st not in canonical.semantic_types:
                    canonical.semantic_types.append(st)
            for k, v in node.external_ids.items():
                canonical.external_ids.setdefault(k, v)
            if not canonical.definition and node.definition:
                canonical.definition = node.definition
            # record provenance of the merged id so downstream tools can track it
            canonical.external_ids.setdefault(node.id, node.id)
            merged_nodes += 1

    if not redirect:
        return {"merged_nodes": 0, "rewritten_edges": 0, "dropped_self_loops": 0}

    # rebuild graph with redirected edges
    new_g = type(kg.G)()
    # add all surviving nodes
    surviving_ids = set(kg._index.keys()) - set(redirect.keys())
    for nid in surviving_ids:
        new_g.add_node(nid, **kg._index[nid].to_dict())

    for src, tgt, data in kg.G.edges(data=True):
        new_src = redirect.get(src, src)
        new_tgt = redirect.get(tgt, tgt)
        if new_src == new_tgt:
            dropped_self_loops += 1
            continue
        if new_src not in surviving_ids or new_tgt not in surviving_ids:
            continue
        if new_g.has_edge(new_src, new_tgt):
            existing = new_g.edges[new_src, new_tgt]
            if data.get("confidence", 0) > existing.get("confidence", 0):
                edata = dict(data)
                edata["source_id"] = new_src
                edata["target_id"] = new_tgt
                new_g.edges[new_src, new_tgt].update(edata)
        else:
            edata = dict(data)
            edata["source_id"] = new_src
            edata["target_id"] = new_tgt
            new_g.add_edge(new_src, new_tgt, **edata)
            rewritten_edges += 1

    kg.G = new_g
    for nid in redirect:
        kg._index.pop(nid, None)

    # store redirect map for traceability
    return {
        "merged_nodes": merged_nodes,
        "rewritten_edges": rewritten_edges,
        "dropped_self_loops": dropped_self_loops,
        "redirect": redirect,
    }


def merge_kg_by_umls(
    graph_path: Path,
    mrconso_path: Path,
    output_path: Path | None = None,
    backup: bool = True,
    redirect_save_path: Path | None = None,
) -> dict:
    """Top-level entrypoint: align + merge a KG file in-place."""
    graph_path = Path(graph_path)
    mrconso_path = Path(mrconso_path)
    output_path = Path(output_path) if output_path else graph_path

    if backup and output_path == graph_path:
        backup_path = graph_path.with_suffix(graph_path.suffix + f".pre_umls_merge")
        if not backup_path.exists():
            logger.info(f"backing up {graph_path} -> {backup_path}")
            shutil.copy2(graph_path, backup_path)

    logger.info(f"loading {graph_path}")
    kg = load_graph(graph_path)
    before = kg.stats()
    logger.info(f"before: {before['n_concepts']} concepts, {before['n_edges']} edges")

    seeded = _seed_cuis_from_ids(kg)
    aligned = _align_via_mrconso(kg, mrconso_path)

    groups = _group_by_cui(kg)
    n_groups = len(groups)
    n_dup_nodes = sum(len(v) - 1 for v in groups.values())
    logger.info(f"found {n_groups} merge groups covering {n_dup_nodes} duplicate nodes")

    merge_result = _merge_groups(kg, groups)
    redirect = merge_result.pop("redirect", {})

    after = kg.stats()
    logger.info(f"after: {after['n_concepts']} concepts, {after['n_edges']} edges")

    save_graph(kg, output_path)

    if redirect_save_path and redirect:
        redirect_save_path = Path(redirect_save_path)
        redirect_save_path.parent.mkdir(parents=True, exist_ok=True)
        redirect_save_path.write_text(
            json.dumps(redirect, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"saved redirect map to {redirect_save_path}")

    summary = {
        "graph_path": str(graph_path),
        "output_path": str(output_path),
        "before": {k: before[k] for k in ("n_concepts", "n_edges")},
        "after": {k: after[k] for k in ("n_concepts", "n_edges")},
        "cuis_seeded_from_ids": seeded,
        "cuis_aligned_via_mrconso": aligned,
        "merge_groups": n_groups,
        "duplicate_nodes": n_dup_nodes,
        **merge_result,
        "timestamp": datetime.now().isoformat(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge synonymous KG nodes via UMLS CUI")
    parser.add_argument("--graph", required=True, type=Path, help="KG JSON path")
    parser.add_argument("--mrconso", required=True, type=Path, help="MRCONSO.RRF path")
    parser.add_argument("--output", type=Path, default=None, help="Output KG path (default: overwrite input)")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup of input file")
    parser.add_argument("--redirect-out", type=Path, default=None, help="Save redirect map (old_id -> canonical_id) to JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = merge_kg_by_umls(
        graph_path=args.graph,
        mrconso_path=args.mrconso,
        output_path=args.output,
        backup=not args.no_backup,
        redirect_save_path=args.redirect_out,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
