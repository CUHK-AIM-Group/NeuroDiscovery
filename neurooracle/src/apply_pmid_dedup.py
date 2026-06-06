"""P0 step 3: apply only the same-PMID dedup to the full KG.

Why a standalone runner instead of `run_phase4`:
    run_phase4 also does claim-merge + bridge edges + evidence weighting,
    which would overwrite existing Phase 4 state. Here we do only the
    cross-predicate same-PMID dedup (which you already previewed in a
    small test removing 3,791 claims), write back to the same graph
    file after backing it up.

Usage:
    python -m neurooracle.src.apply_pmid_dedup \
        --graph neurooracle/data/full_snapshot_v2/knowledge_graph.json \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from .phase4_optimize import dedupe_same_pmid_claims
from .storage import load_graph, save_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Load + count only; don't write back.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("apply_pmid_dedup")

    log.info(f"loading {args.graph}")
    kg = load_graph(args.graph)
    n_before = kg.stats()["n_concepts"]
    e_before = kg.stats()["n_edges"]
    log.info(f"before: {n_before} concepts, {e_before} edges")

    deleted = dedupe_same_pmid_claims(kg)

    n_after = kg.stats()["n_concepts"]
    e_after = kg.stats()["n_edges"]
    log.info(f"after : {n_after} concepts ({n_before - n_after:+d}), "
             f"{e_after} edges ({e_before - e_after:+d})")
    log.info(f"deleted {deleted} cross-predicate duplicate claims")

    if args.dry_run:
        log.info("dry-run: not writing back")
        return

    backup = args.graph.with_suffix(args.graph.suffix + ".pre_pmid_dedup")
    if not backup.exists():
        shutil.copy2(args.graph, backup)
        log.info(f"backup -> {backup}")
    else:
        log.info(f"backup already exists: {backup}")

    save_graph(kg, args.graph)
    log.info(f"saved {args.graph}")


if __name__ == "__main__":
    main()
