"""Normalize residual MANUAL provenance and placeholder types in active KG files."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from neurooracle.scripts.canonicalize_manual_claim_ids import (
        DEFAULT_EXTRACTED,
        DEFAULT_GRAPH,
        atomic_swap_with_rollback,
        canonical_claim_id,
        is_legacy_manual_claim_id,
    )
except ModuleNotFoundError:
    from canonicalize_manual_claim_ids import (  # type: ignore[no-redef]
        DEFAULT_EXTRACTED,
        DEFAULT_GRAPH,
        atomic_swap_with_rollback,
        canonical_claim_id,
        is_legacy_manual_claim_id,
    )


MIGRATION_NAME = "normalize_manual_residuals_20260722"
LEGACY_ID_KEYS = {
    "legacy_claim_id",
    "legacy_concept_id",
    "legacy_subject_id",
    "legacy_object_id",
}
TYPE_KEYS = {"atom_type", "subject_type", "object_type"}
TYPE_RENAMES = {
    "MANUAL_TOPIC": "TOPIC",
    "MANUAL_FINDING": "FINDING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_GRAPH.parent / "manual_residual_cleanup_20260722.json",
    )
    return parser.parse_args()


def _legacy_claim_source(value: Any) -> bool:
    text = str(value or "")
    return text.startswith("claim:") and is_legacy_manual_claim_id(text[6:])


def normalize_manual_residuals(value: Any, counts: Counter[str] | None = None) -> Counter[str]:
    """Mutate a JSON-compatible value while preserving curation provenance text."""
    counts = counts if counts is not None else Counter()
    if isinstance(value, dict):
        metadata = value.get("metadata")
        claim_id = metadata.get("claim_id") if isinstance(metadata, dict) else ""
        source = value.get("source")
        if _legacy_claim_source(source):
            legacy_id = str(source)[6:]
            replacement_id = (
                str(claim_id)
                if str(claim_id).startswith("CLM:")
                else canonical_claim_id(legacy_id)
            )
            value["source"] = f"claim:{replacement_id}"
            counts["claim_sources_normalized"] += 1

        for key in list(value):
            item = value[key]
            if key in LEGACY_ID_KEYS and is_legacy_manual_claim_id(item):
                del value[key]
                counts["legacy_id_fields_removed"] += 1
                continue
            if key in TYPE_KEYS and item in TYPE_RENAMES:
                value[key] = TYPE_RENAMES[item]
                counts[f"type_{item.lower()}_normalized"] += 1
                continue
            normalize_manual_residuals(item, counts)
    elif isinstance(value, list):
        for item in value:
            normalize_manual_residuals(item, counts)
    return counts


def count_manual_residuals(value: Any, counts: Counter[str] | None = None) -> Counter[str]:
    counts = counts if counts is not None else Counter()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LEGACY_ID_KEYS and is_legacy_manual_claim_id(item):
                counts["legacy_id_fields"] += 1
            if key in TYPE_KEYS and item in TYPE_RENAMES:
                counts["legacy_placeholder_types"] += 1
            if key == "source" and _legacy_claim_source(item):
                counts["legacy_claim_sources"] += 1
            count_manual_residuals(item, counts)
    elif isinstance(value, list):
        for item in value:
            count_manual_residuals(item, counts)
    return counts


def validate_graph(graph: dict[str, Any], before: dict[str, int]) -> dict[str, int]:
    concepts = graph.get("concepts")
    edges = graph.get("edges")
    if not isinstance(concepts, dict) or not isinstance(edges, list):
        raise ValueError("KG must contain object concepts and array edges")
    if len(concepts) != before["concepts"] or len(edges) != before["edges"]:
        raise ValueError("KG node or edge count changed during residual normalization")
    dangling = sum(
        1
        for edge in edges
        if edge.get("source_id") not in concepts or edge.get("target_id") not in concepts
    )
    if dangling:
        raise ValueError(f"KG contains {dangling} dangling edge endpoints")
    residuals = count_manual_residuals(graph)
    if residuals:
        raise ValueError(f"MANUAL residuals remain in staged KG: {dict(residuals)}")
    return {
        "concepts": len(concepts),
        "edges": len(edges),
        "dangling_edge_endpoints": dangling,
        "manual_residuals": 0,
    }


def stage_extracted(source: Path, target: Path) -> tuple[int, Counter[str]]:
    rows = 0
    counts: Counter[str] = Counter()
    with source.open("r", encoding="utf-8") as src, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id"):
                raise ValueError(f"Missing claim id in {source} line {number}")
            rows += 1
            normalize_manual_residuals(row, counts)
            residuals = count_manual_residuals(row)
            if residuals:
                raise ValueError(
                    f"MANUAL residuals remain in staged archive line {number}: "
                    f"{dict(residuals)}"
                )
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return rows, counts


def main() -> None:
    args = parse_args()
    with args.graph.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    before = {
        "concepts": len(graph.get("concepts") or {}),
        "edges": len(graph.get("edges") or []),
    }
    graph_counts = normalize_manual_residuals(graph)
    graph_validation = validate_graph(graph, before)

    report: dict[str, Any] = {
        "migration": MIGRATION_NAME,
        "mode": "apply" if args.apply else "dry_run",
        "graph": str(args.graph.resolve()),
        "extracted": str(args.extracted.resolve()),
        "graph_changes": dict(sorted(graph_counts.items())),
        "graph_validation": graph_validation,
    }
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    graph_temp = args.graph.with_suffix(args.graph.suffix + f".{MIGRATION_NAME}.tmp")
    extracted_temp = args.extracted.with_suffix(
        args.extracted.suffix + f".{MIGRATION_NAME}.tmp"
    )
    graph_backup = args.graph.with_suffix(args.graph.suffix + f".{MIGRATION_NAME}.bak")
    extracted_backup = args.extracted.with_suffix(
        args.extracted.suffix + f".{MIGRATION_NAME}.bak"
    )
    for path in (graph_temp, extracted_temp):
        if path.exists():
            path.unlink()
    for path in (graph_backup, extracted_backup):
        if path.exists():
            raise ValueError(f"Rollback file already exists: {path}")

    extracted_rows, extracted_counts = stage_extracted(args.extracted, extracted_temp)
    with graph_temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    swapped_graph = False
    try:
        atomic_swap_with_rollback(graph_temp, args.graph, graph_backup)
        swapped_graph = True
        atomic_swap_with_rollback(extracted_temp, args.extracted, extracted_backup)
    except Exception:
        if swapped_graph and graph_backup.exists():
            if args.graph.exists():
                args.graph.unlink()
            os.replace(graph_backup, args.graph)
        raise

    report.update(
        {
            "extracted_rows": extracted_rows,
            "extracted_changes": dict(sorted(extracted_counts.items())),
            "validation": {
                "legacy_claim_sources_remaining": 0,
                "legacy_placeholder_types_remaining": 0,
                "legacy_id_fields_remaining": 0,
            },
            "rollback_files": [str(graph_backup), str(extracted_backup)],
        }
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
