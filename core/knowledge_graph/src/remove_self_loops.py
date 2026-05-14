"""Remove self-loop edges (source_id == target_id) from a KG JSON file.

Usage:
    python -m core.knowledge_graph.src.remove_self_loops \
        --graph core/knowledge_graph/data/quick/knowledge_graph.json \
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("remove_self_loops")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info(f"loading {args.graph}")
    kg = json.loads(args.graph.read_text(encoding="utf-8"))
    edges = kg.get("edges", [])
    before = len(edges)

    kept = [e for e in edges if e.get("source_id") != e.get("target_id")]
    removed = before - len(kept)
    log.info(f"self-loop edges: {removed} / {before} total")

    if args.dry_run:
        log.info("dry-run: no files written")
        return

    if removed == 0:
        log.info("nothing to remove")
        return

    backup = args.graph.with_name(args.graph.stem + ".pre_selfloop_fix.json")
    if not backup.exists():
        shutil.copy2(args.graph, backup)
        log.info(f"backup -> {backup}")

    kg["edges"] = kept
    args.graph.write_text(
        json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"saved {args.graph} ({before} -> {len(kept)} edges)")


if __name__ == "__main__":
    main()
