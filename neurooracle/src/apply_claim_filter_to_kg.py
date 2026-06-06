"""P0 step 1b: delete Tier A bad claim nodes from the KG.

Takes bad_claim_ids.json from verbatim_validator and removes those
claim nodes from the knowledge graph (and, transitively, all edges
referencing them).

Usage:
    python -m neurooracle.src.apply_claim_filter_to_kg \
        --graph neurooracle/data/full_snapshot_v2/knowledge_graph.json \
        --bad-ids neurooracle/data/full_snapshot_v2/bad_claim_ids.json \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from .storage import load_graph, save_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--bad-ids", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("apply_claim_filter")

    bad = json.loads(args.bad_ids.read_text(encoding="utf-8"))
    bad_ids = set(bad.get("tier_a_ids") or [])
    log.info(f"tier A claims to delete: {len(bad_ids)}")
    log.info(f"breakdown: {bad.get('by_reason')}")

    log.info(f"loading {args.graph}")
    kg = load_graph(args.graph)
    n_before = kg.stats()["n_concepts"]
    e_before = kg.stats()["n_edges"]
    log.info(f"before: {n_before} concepts, {e_before} edges")

    # Claim nodes in the KG are stored with id = f"CLM:{hash}" (the 'id' on
    # the claim record IS its node id in our schema). Drop those nodes.
    missing = 0
    deleted = 0
    for cid in bad_ids:
        if cid in kg._index:
            kg.G.remove_node(cid)
            del kg._index[cid]
            deleted += 1
        else:
            missing += 1

    n_after = kg.stats()["n_concepts"]
    e_after = kg.stats()["n_edges"]
    log.info(f"deleted {deleted} claim nodes (missing in graph: {missing})")
    log.info(f"after : {n_after} concepts ({n_before - n_after:+d}), "
             f"{e_after} edges ({e_before - e_after:+d})")

    if args.dry_run:
        log.info("dry-run: not writing back")
        return

    backup = args.graph.with_suffix(args.graph.suffix + ".pre_claim_filter")
    if not backup.exists():
        shutil.copy2(args.graph, backup)
        log.info(f"backup -> {backup}")
    else:
        log.info(f"backup already exists: {backup}")

    save_graph(kg, args.graph)
    log.info(f"saved {args.graph}")


if __name__ == "__main__":
    main()
