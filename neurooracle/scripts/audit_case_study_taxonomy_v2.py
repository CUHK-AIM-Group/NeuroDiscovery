"""Audit structured legacy Case 1/2/3 routing fields across the entire KG.

Unlike claim coverage statistics, this audit also inspects non-claim concepts
and edges.  Historical free-text review explanations are intentionally left
untouched; only JSON keys that formerly carried routing state are counted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from neurooracle.scripts.streaming_graph_json import (
    is_claim_node,
    iter_concepts,
    iter_edges,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_REPORT = (
    REPO
    / "neurooracle"
    / "data"
    / "migration_reports"
    / "case_study_taxonomy_v2_full_audit.json"
)

LEGACY_SCOPE_KEYS = frozenset(
    {
        "paper_scope",
        "case3_tasks",
        "case3_subtasks",
        "case1_eligible",
        "case2_eligible",
        "case_study",
        "case_study_ids",
    }
)


def iter_legacy_fields(
    value: object,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[str, str, object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            if key in LEGACY_SCOPE_KEYS:
                yield key, ".".join(child_path), child
            yield from iter_legacy_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_legacy_fields(child, (*path, f"[{index}]"))


def _brief(value: object) -> object:
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if isinstance(value, list):
        return [_brief(item) for item in value[:8]]
    if isinstance(value, dict):
        return {str(key): _brief(item) for key, item in list(value.items())[:8]}
    return value


def audit_graph(graph_path: Path, *, example_limit: int = 3) -> dict[str, Any]:
    occurrences: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    concept_nodes = 0
    claim_nodes = 0
    edges = 0

    def record(
        *,
        entity_type: str,
        entity_id: str,
        fields: list[tuple[str, str, object]],
    ) -> None:
        if not fields:
            return
        entity_counts[entity_type] += 1
        for key, path, value in fields:
            occurrences[key] += 1
            bucket = examples.setdefault(f"{entity_type}:{key}", [])
            if len(bucket) < example_limit:
                bucket.append(
                    {
                        "entity_id": entity_id,
                        "path": path,
                        "value": _brief(value),
                    }
                )

    for node_id, node in iter_concepts(graph_path):
        concept_nodes += 1
        claim = is_claim_node(node_id, node)
        if claim:
            claim_nodes += 1
        record(
            entity_type="claim_concept" if claim else "non_claim_concept",
            entity_id=node_id,
            fields=list(iter_legacy_fields(node)),
        )

    for index, edge in enumerate(iter_edges(graph_path)):
        edges += 1
        source = str(edge.get("source") or edge.get("source_id") or "?")
        target = str(edge.get("target") or edge.get("target_id") or "?")
        record(
            entity_type="edge",
            entity_id=f"{index}:{source}->{target}",
            fields=list(iter_legacy_fields(edge)),
        )

    return {
        "schema_version": "case_study_membership.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(graph_path),
        "source_bytes": graph_path.stat().st_size,
        "entities_scanned": {
            "concept_nodes": concept_nodes,
            "claim_nodes": claim_nodes,
            "non_claim_concept_nodes": concept_nodes - claim_nodes,
            "edges": edges,
        },
        "structured_legacy_field_occurrences": dict(sorted(occurrences.items())),
        "entities_with_structured_legacy_fields": dict(sorted(entity_counts.items())),
        "examples": examples,
        "is_canonical": not occurrences,
        "note": (
            "Free-text historical audit explanations may mention Case 3 or "
            "case3_tasks; those strings are evidence, not routing fields."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--example-limit", type=int, default=3)
    args = parser.parse_args()
    report = audit_graph(args.graph.resolve(), example_limit=args.example_limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
