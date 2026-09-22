"""Synchronize canonical case-study membership from formal KG to claim store."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neurooracle.scripts.migrate_case_study_taxonomy_v2 import rewrite_extracted
from neurooracle.scripts.streaming_graph_json import is_claim_node, iter_concepts
from neurooracle.src.case_study_scope import (
    claim_case_study_ids_from_dict,
    paper_case_study_ids_from_dict,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_EXTRACTED = REPO / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"


def load_graph_routing(
    graph_path: Path,
) -> dict[str, tuple[list[str], list[str]]]:
    routing: dict[str, tuple[list[str], list[str]]] = {}
    for node_id, node in iter_concepts(graph_path):
        if not is_claim_node(node_id, node):
            continue
        claim = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        claim.setdefault("id", node_id)
        routing[node_id] = (
            paper_case_study_ids_from_dict(claim),
            claim_case_study_ids_from_dict(claim),
        )
    return routing


def synchronize(
    graph_path: Path,
    extracted_path: Path,
    *,
    dry_run: bool,
) -> dict[str, object]:
    routing = load_graph_routing(graph_path)
    temp = extracted_path.with_suffix(extracted_path.suffix + ".case-study-sync.tmp")
    if temp.exists():
        temp.unlink()
    try:
        summary = rewrite_extracted(extracted_path, temp, routing)
        summary["dry_run"] = dry_run
        summary["graph_claims"] = len(routing)
        if dry_run:
            summary["projected_bytes"] = temp.stat().st_size
        else:
            os.replace(temp, extracted_path)
        return summary
    finally:
        if temp.exists():
            temp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = synchronize(
        args.graph.resolve(),
        args.extracted.resolve(),
        dry_run=args.dry_run,
    )
    report["graph"] = str(args.graph.resolve())
    report["extracted"] = str(args.extracted.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
