"""Finish taxonomy-v2 migration for non-claim concepts and provenance edges.

The primary migration canonically labels papers and claims.  Legacy graphs also
duplicated ``paper_scope`` onto generated anchor concepts and relation edges.
This repair removes that obsolete anchor routing state and replaces edge scope
with the canonical membership of its referenced claim.  It writes and audits a
temporary graph before atomically replacing the formal source.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.audit_case_study_taxonomy_v2 import (
    LEGACY_SCOPE_KEYS,
    audit_graph,
    iter_legacy_fields,
)
from neurooracle.scripts.streaming_graph_json import (
    is_claim_node,
    iter_concepts,
    rewrite_concepts,
    rewrite_edges,
)
from neurooracle.src.case_study_scope import (
    apply_case_study_membership,
    claim_case_study_ids_from_dict,
    paper_case_study_ids_from_dict,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_REPORT = (
    REPO
    / "neurooracle"
    / "data"
    / "migration_reports"
    / "case_study_taxonomy_v2_provenance_applied.json"
)
SCHEMA_VERSION = "case_study_membership.v2"


def remove_structured_legacy_fields(value: object) -> int:
    """Remove old routing keys recursively while preserving text values."""

    removed = 0
    if isinstance(value, dict):
        for key in list(value):
            if key in LEGACY_SCOPE_KEYS:
                value.pop(key, None)
                removed += 1
            else:
                removed += remove_structured_legacy_fields(value[key])
    elif isinstance(value, list):
        for child in value:
            removed += remove_structured_legacy_fields(child)
    return removed


def build_claim_routing(
    graph_path: Path,
) -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, int]]:
    routing: dict[str, tuple[list[str], list[str]]] = {}
    concepts = 0
    claims = 0
    aliases = 0
    for node_id, node in iter_concepts(graph_path):
        concepts += 1
        if not is_claim_node(node_id, node):
            continue
        claims += 1
        claim = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        paper_ids = paper_case_study_ids_from_dict(claim)
        claim_ids = claim_case_study_ids_from_dict(claim)
        route = (paper_ids, claim_ids)
        for claim_id in {
            node_id,
            str(claim.get("id") or ""),
            str(claim.get("claim_id") or ""),
        }:
            if claim_id:
                if claim_id in routing and routing[claim_id] != route:
                    raise ValueError(f"conflicting membership for claim ID {claim_id!r}")
                if claim_id not in routing and claim_id != node_id:
                    aliases += 1
                routing[claim_id] = route
    return routing, {
        "concept_nodes": concepts,
        "claim_nodes": claims,
        "claim_id_aliases": aliases,
        "routing_keys": len(routing),
    }


def rewrite_anchor_concepts(graph_path: Path, output_path: Path) -> dict[str, int]:
    counters: Counter[str] = Counter()

    def transform(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
        counters["concept_nodes"] += 1
        claim = is_claim_node(node_id, node)
        if claim:
            counters["claim_nodes"] += 1
        before = list(iter_legacy_fields(node))
        removed = remove_structured_legacy_fields(node)
        counters["structured_legacy_fields_removed"] += removed
        if removed:
            counters[
                "claim_concepts_cleaned" if claim else "non_claim_concepts_cleaned"
            ] += 1
        if len(before) != removed:
            raise ValueError(
                f"legacy-field removal mismatch on {node_id}: {len(before)} != {removed}"
            )
        return node

    rewritten = rewrite_concepts(graph_path, output_path, transform)
    if rewritten != counters["concept_nodes"]:
        raise ValueError("concept rewrite count mismatch")
    return dict(counters)


def rewrite_provenance_edges(
    graph_path: Path,
    output_path: Path,
    routing: dict[str, tuple[list[str], list[str]]],
) -> dict[str, int]:
    counters: Counter[str] = Counter()

    def transform(edge: dict[str, Any]) -> dict[str, Any]:
        counters["edges"] += 1
        metadata = edge.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            edge["metadata"] = metadata
        legacy = list(iter_legacy_fields(edge))
        if legacy:
            counters["edges_with_legacy_fields"] += 1
            counters["structured_legacy_fields_removed"] += len(legacy)
        claim_id = str(metadata.get("claim_id") or "")
        route = routing.get(claim_id)
        if route is not None:
            paper_ids, claim_ids = route
            apply_case_study_membership(
                metadata,
                paper_case_study_ids=paper_ids,
                claim_case_study_ids=claim_ids,
            )
            metadata["case_study_membership_schema_version"] = SCHEMA_VERSION
            counters["edges_routed_from_claim"] += 1
        elif legacy:
            paper_ids = paper_case_study_ids_from_dict(metadata)
            claim_ids = claim_case_study_ids_from_dict(metadata)
            apply_case_study_membership(
                metadata,
                paper_case_study_ids=paper_ids,
                claim_case_study_ids=claim_ids,
            )
            metadata["case_study_membership_schema_version"] = SCHEMA_VERSION
            counters["legacy_edges_using_local_fallback"] += 1
            if claim_id:
                counters["legacy_edges_with_unmatched_claim_id"] += 1
        # ``apply_case_study_membership`` removes direct legacy keys.  The
        # recursive cleanup catches any deeper routing state without counting
        # those direct keys a second time.
        counters["nested_structured_legacy_fields_removed"] += (
            remove_structured_legacy_fields(edge)
        )
        return edge

    rewritten = rewrite_edges(graph_path, output_path, transform)
    if rewritten != counters["edges"]:
        raise ValueError("edge rewrite count mismatch")
    return dict(counters)


def apply_provenance_migration(graph_path: Path) -> dict[str, Any]:
    concepts_temp = graph_path.with_suffix(graph_path.suffix + ".anchors-v2.tmp")
    final_temp = graph_path.with_suffix(graph_path.suffix + ".provenance-v2.tmp")
    for temp in (concepts_temp, final_temp):
        if temp.exists():
            temp.unlink()

    source_bytes = graph_path.stat().st_size
    routing, routing_stats = build_claim_routing(graph_path)
    try:
        concept_rewrite = rewrite_anchor_concepts(graph_path, concepts_temp)
        edge_rewrite = rewrite_provenance_edges(concepts_temp, final_temp, routing)
        validation = audit_graph(final_temp, example_limit=1)
        if not validation["is_canonical"]:
            raise ValueError(
                "temporary graph retains structured legacy fields: "
                f"{validation['structured_legacy_field_occurrences']}"
            )
        if validation["entities_scanned"]["concept_nodes"] != routing_stats["concept_nodes"]:
            raise ValueError("validated concept count differs from source")
        if validation["entities_scanned"]["edges"] != edge_rewrite["edges"]:
            raise ValueError("validated edge count differs from source")
        os.replace(final_temp, graph_path)
    finally:
        for temp in (concepts_temp, final_temp):
            if temp.exists():
                temp.unlink()

    return {
        "source_bytes": source_bytes,
        "result_bytes": graph_path.stat().st_size,
        "claim_routing": routing_stats,
        "concept_rewrite": concept_rewrite,
        "edge_rewrite": edge_rewrite,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    graph = args.graph.resolve()
    before = audit_graph(graph, example_limit=3)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "graph": str(graph),
        "before": before,
    }
    if args.apply:
        report["application"] = apply_provenance_migration(graph)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
