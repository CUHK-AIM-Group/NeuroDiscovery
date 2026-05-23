"""Incremental ingest: add OUTCOME-IM bridges to an existing KG.

Usage:
    python -m neurooracle.src.apply_outcome_im_bridges \
        --graph neurooracle/data/full_snapshot_v1/knowledge_graph.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .ingestion.outcome_im_bridges import ingest_outcome_im_bridges
from .storage import load_graph, save_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("apply_outcome_im_bridges")

    log.info(f"Loading KG from {args.graph}")
    kg = load_graph(args.graph)
    pre = kg.stats()
    log.info(f"  before: {pre['n_concepts']} concepts / {pre['n_edges']} edges")

    counts = ingest_outcome_im_bridges(kg)
    log.info(f"  bridges: {counts}")

    post = kg.stats()
    log.info(f"  after:  {post['n_concepts']} concepts / {post['n_edges']} edges "
             f"(+{post['n_edges']-pre['n_edges']} edges)")

    save_graph(kg, args.graph)
    log.info(f"Saved to {args.graph}")


if __name__ == "__main__":
    main()
