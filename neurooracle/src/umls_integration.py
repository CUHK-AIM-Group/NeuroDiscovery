"""UMLS entity alignment: map knowledge graph concepts to CUIs.

Streaming approach: scan MRCONSO.RRF once, matching against graph concept names.
No full in-memory index needed.

Usage:
    from neurooracle import load_graph
    from neurooracle.phase1 import align_graph_to_umls

    kg = load_graph()
    results = align_graph_to_umls(kg, "neurooracle/data/raw/MRCONSO.RRF")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MRCONSO.RRF columns (0-indexed)
COL_CUI = 0
COL_LAT = 1      # Language
COL_TS = 2       # Term Status: P=Preferred
COL_ISPREF = 6   # Is Preferred in source (Y/N)
COL_SAB = 11     # Source abbreviation
COL_STR = 14     # String (concept name)
COL_SUPPRESS = 16  # Suppressible flag


def _collect_graph_names(kg) -> dict[str, list[str]]:
    """Collect all concept names from graph, grouped by lowercase name.

    Returns: {lowercase_name: [concept_id, ...]}
    """
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for nid, node in kg._index.items():
        if "claim" in node.domain_tags:
            continue
        key = node.preferred_name.lower().strip()
        if key:
            name_to_ids[key].append(nid)
        for alias in node.aliases:
            key = alias.lower().strip()
            if key:
                name_to_ids[key].append(nid)
    return dict(name_to_ids)


def _parse_mrconso_streaming(
    mrconso_path: str | Path,
    graph_names: dict[str, list[str]],
) -> dict[str, dict]:
    """Scan MRCONSO.RRF once, matching against graph names.

    Returns: {concept_id: {"cui": str, "preferred_name": str, "sources": set}}
    """
    # Build set of all lowercase names we're looking for
    target_names = set(graph_names.keys())
    logger.info(f"scanning MRCONSO for {len(target_names):,} unique graph names...")

    # Results: concept_id → best match info
    matches: dict[str, dict] = {}

    line_count = 0
    match_count = 0

    with open(mrconso_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            if line_count % 2_000_000 == 0:
                logger.info(f"  scanned {line_count/1e6:.0f}M lines, {match_count} matches")

            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue

            # Filter: English only
            if parts[COL_LAT] != "ENG":
                continue

            # Filter: not suppressed
            if parts[COL_SUPPRESS] in ("O", "Y"):
                continue

            name = parts[COL_STR].strip()
            if not name:
                continue

            key = name.lower()
            if key not in target_names:
                continue

            cui = parts[COL_CUI]
            is_preferred = parts[COL_TS] == "P" or parts[COL_ISPREF] == "Y"
            sab = parts[COL_SAB]

            # Match this UMLS entry to all graph concepts with this name
            for concept_id in graph_names[key]:
                existing = matches.get(concept_id)
                if existing is None:
                    matches[concept_id] = {
                        "cui": cui,
                        "name": name,
                        "is_preferred": is_preferred,
                        "sources": {sab},
                    }
                    match_count += 1
                else:
                    # Upgrade: prefer the entry where name is preferred
                    if is_preferred and not existing["is_preferred"]:
                        existing["cui"] = cui
                        existing["name"] = name
                        existing["is_preferred"] = True
                    existing["sources"].add(sab)

    logger.info(f"scan complete: {line_count:,} lines, {match_count:,} concept matches")
    return matches


def align_graph_to_umls(
    kg,
    mrconso_path: str | Path,
    save_path: Optional[str | Path] = None,
) -> dict:
    """Align knowledge graph concepts to UMLS CUIs via streaming MRCONSO scan.

    Modifies kg in-place: adds 'umls_cui' to node metadata.
    Optionally saves alignment results to JSON.

    Returns summary dict.
    """
    # Step 1: collect all graph concept names
    graph_names = _collect_graph_names(kg)
    logger.info(f"graph has {len(graph_names):,} unique concept names")

    # Step 2: scan MRCONSO for matches
    matches = _parse_mrconso_streaming(mrconso_path, graph_names)

    # Step 3: apply matches to graph
    from .ingestion.experiment_infra import should_skip_umls_alignment
    matched = 0
    unmatched = 0
    skipped = 0
    for nid, node in kg._index.items():
        if "claim" in node.domain_tags:
            continue
        if should_skip_umls_alignment(node):
            skipped += 1
            continue
        if nid in matches:
            node.metadata["umls_cui"] = matches[nid]["cui"]
            matched += 1
        else:
            unmatched += 1

    total = matched + unmatched
    logger.info(
        f"UMLS alignment: {matched}/{total} concepts matched "
        f"({100*matched/total:.1f}%); {skipped} experiment-infra nodes skipped"
    )

    # Step 4: optionally save results
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            cid: {
                "cui": m["cui"],
                "name": m["name"],
                "is_preferred": m["is_preferred"],
                "sources": sorted(m["sources"]),
            }
            for cid, m in matches.items()
        }
        save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"saved alignment to {save_path}")

    return {
        "total": total,
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": matched / total if total else 0,
    }
