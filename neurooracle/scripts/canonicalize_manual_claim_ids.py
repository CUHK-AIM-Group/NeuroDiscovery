"""Canonicalize legacy MANUAL IDs in a KG and its claim archive.

The migration is dry-run by default. With ``--apply`` it stages both large
files, validates the staged representation, and swaps them into place while
retaining rollback copies until the caller finishes external verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_EXTRACTED = ROOT / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
MIGRATION_NAME = "canonicalize_manual_claim_ids_20260722"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--apply", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all-non-clm",
        action="store_true",
        help="Canonicalize every claim-domain node whose ID does not start with CLM:.",
    )
    mode.add_argument(
        "--all-manual-concepts",
        action="store_true",
        help="Canonicalize every MANUAL-prefixed claim or endpoint concept and archive reference.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_GRAPH.parent / "claim_id_migration_20260722.json",
    )
    return parser.parse_args()


def is_legacy_manual_claim_id(value: Any) -> bool:
    text = str(value or "")
    return text.upper().startswith("MANUAL") and ":" in text


def canonical_claim_id(legacy_id: str) -> str:
    digest = hashlib.sha256(legacy_id.encode("utf-8")).hexdigest()[:16]
    return f"CLM:{digest}"


def canonical_concept_id(legacy_id: str) -> str:
    digest = hashlib.sha256(legacy_id.encode("utf-8")).hexdigest()[:16]
    return f"CLM_CONCEPT:{digest}"


def is_claim_node(node: dict[str, Any]) -> bool:
    return "claim" in (node.get("domain_tags") or [])


def replace_exact_ids(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [replace_exact_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: replace_exact_ids(item, mapping) for key, item in value.items()}
    return value


def read_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            if not claim_id:
                raise ValueError(f"Missing claim id in {path} line {number}")
            if claim_id in ids:
                raise ValueError(f"Duplicate claim id in {path}: {claim_id}")
            ids.add(claim_id)
    return ids


def build_mapping(
    concepts: dict[str, dict[str, Any]],
    *,
    all_non_clm: bool = False,
    all_manual_concepts: bool = False,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node_id, node in concepts.items():
        claim_node = is_claim_node(node)
        if all_non_clm and claim_node and not node_id.startswith("CLM:"):
            mapping[node_id] = canonical_claim_id(node_id)
        elif all_manual_concepts and is_legacy_manual_claim_id(node_id):
            mapping[node_id] = (
                canonical_claim_id(node_id)
                if claim_node
                else canonical_concept_id(node_id)
            )
        elif claim_node and is_legacy_manual_claim_id(node_id):
            mapping[node_id] = canonical_claim_id(node_id)
    validate_mapping(mapping, concepts)
    return mapping


def validate_mapping(
    mapping: dict[str, str], concepts: dict[str, dict[str, Any]]
) -> None:
    targets = list(mapping.values())
    if len(targets) != len(set(targets)):
        raise ValueError("Canonical claim ID collision within MANUAL migration set")
    occupied = set(concepts) - set(mapping)
    collisions = sorted(set(targets) & occupied)
    if collisions:
        raise ValueError(f"Canonical claim IDs collide with existing nodes: {collisions[:5]}")


def extend_mapping_from_extracted(
    path: Path,
    mapping: dict[str, str],
    concepts: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Include legacy IDs found only in the extracted candidate archive."""
    added_claim_ids = 0
    added_concept_ids = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key, factory in (
                ("id", canonical_claim_id),
                ("subject_id", canonical_concept_id),
                ("object_id", canonical_concept_id),
            ):
                legacy_id = str(row.get(key) or "")
                if not is_legacy_manual_claim_id(legacy_id):
                    continue
                canonical_id = factory(legacy_id)
                previous = mapping.get(legacy_id)
                if previous is not None and previous != canonical_id:
                    raise ValueError(
                        f"Legacy ID used as both claim and endpoint at line {number}: {legacy_id}"
                    )
                if previous is None:
                    mapping[legacy_id] = canonical_id
                    if key == "id":
                        added_claim_ids += 1
                    else:
                        added_concept_ids += 1
    validate_mapping(mapping, concepts)
    return {
        "extracted_only_manual_claim_ids_selected": added_claim_ids,
        "extracted_only_manual_concept_ids_selected": added_concept_ids,
    }


def migrate_graph(
    graph: dict[str, Any], mapping: dict[str, str], migration_name: str = MIGRATION_NAME
) -> tuple[dict[str, Any], dict[str, int]]:
    concepts = graph.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError("knowledge_graph.json concepts must be an object")

    migrated_concepts: dict[str, dict[str, Any]] = {}
    node_ids_changed = 0
    for old_id, node in concepts.items():
        new_id = mapping.get(old_id, old_id)
        migrated = replace_exact_ids(node, mapping)
        if old_id in mapping:
            node_ids_changed += 1
            migrated["id"] = new_id
            metadata = migrated.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                migrated["metadata"] = metadata
            if is_claim_node(node):
                metadata["id"] = new_id
        if new_id in migrated_concepts:
            raise ValueError(f"Duplicate concept after migration: {new_id}")
        migrated_concepts[new_id] = migrated

    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise ValueError("knowledge_graph.json edges must be an array")
    migrated_edges = []
    edge_endpoint_refs_changed = 0
    edge_claim_refs_changed = 0
    for edge in edges:
        if edge.get("source_id") in mapping:
            edge_endpoint_refs_changed += 1
        if edge.get("target_id") in mapping:
            edge_endpoint_refs_changed += 1
        metadata = edge.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("claim_id") in mapping:
            edge_claim_refs_changed += 1
        migrated_edges.append(replace_exact_ids(edge, mapping))

    migrated_graph = dict(graph)
    migrated_graph["concepts"] = migrated_concepts
    migrated_graph["edges"] = migrated_edges
    metadata = dict(migrated_graph.get("metadata") or {})
    migrations = list(metadata.get("migrations") or [])
    migrations.append(
        {
            "name": migration_name,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "concept_ids_changed": len(mapping),
            "claim_ids_changed": sum(
                1
                for old_id in mapping
                if old_id in concepts and is_claim_node(concepts[old_id])
            ),
            "endpoint_ids_changed": sum(
                1
                for old_id in mapping
                if old_id in concepts and not is_claim_node(concepts[old_id])
            ),
            "policy": "claims use CLM:<sha256-16>; endpoints use CLM_CONCEPT:<sha256-16>",
        }
    )
    metadata["migrations"] = migrations
    migrated_graph["metadata"] = metadata
    return migrated_graph, {
        "graph_concept_nodes_changed": node_ids_changed,
        "edge_endpoint_references_changed": edge_endpoint_refs_changed,
        "edge_claim_id_references_changed": edge_claim_refs_changed,
    }


def graph_claim_rows(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    concepts = graph.get("concepts") or {}
    for node_id, node in concepts.items():
        if not isinstance(node, dict) or not is_claim_node(node):
            continue
        metadata = node.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Claim node has no claim metadata: {node_id}")
        row = json.loads(json.dumps(metadata, ensure_ascii=False))
        row["id"] = node_id
        rows[node_id] = row
    return rows


def stage_extracted(
    source: Path,
    target: Path,
    mapping: dict[str, str],
    graph_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seen: set[str] = set()
    legacy_rows_changed = 0
    with source.open("r", encoding="utf-8") as src, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            old_id = str(row.get("id") or "")
            if not old_id:
                raise ValueError(f"Missing claim id in {source} line {number}")
            new_id = mapping.get(old_id, old_id)
            row = replace_exact_ids(row, mapping)
            if old_id in mapping:
                legacy_rows_changed += 1
                row["id"] = new_id
            if new_id in seen:
                raise ValueError(f"Duplicate extracted claim after migration: {new_id}")
            seen.add(new_id)
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

        graph_only_ids = sorted(set(graph_rows) - seen)
        graph_only_count = len(graph_only_ids)
        for claim_id in graph_only_ids:
            dst.write(
                json.dumps(graph_rows[claim_id], ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            seen.add(claim_id)

    extracted_only = seen - set(graph_rows)
    return {
        "extracted_claim_rows_changed": legacy_rows_changed,
        "graph_only_claims_appended_to_extracted": graph_only_count,
        "final_extracted_claims": len(seen),
        "extracted_not_in_graph": len(extracted_only),
    }


def validate_staged(
    graph: dict[str, Any],
    extracted_path: Path,
    *,
    all_non_clm: bool = False,
    all_manual_concepts: bool = False,
) -> dict[str, Any]:
    concepts = graph.get("concepts") or {}
    claim_ids = {
        node_id
        for node_id, node in concepts.items()
        if isinstance(node, dict) and is_claim_node(node)
    }
    legacy_nodes = sorted(node_id for node_id in claim_ids if is_legacy_manual_claim_id(node_id))
    if legacy_nodes:
        raise ValueError(f"Legacy MANUAL claim nodes remain: {legacy_nodes[:5]}")
    if all_non_clm:
        non_clm_nodes = sorted(node_id for node_id in claim_ids if not node_id.startswith("CLM:"))
        if non_clm_nodes:
            raise ValueError(f"Non-CLM claim nodes remain: {non_clm_nodes[:5]}")
    legacy_concepts = sorted(
        node_id for node_id in concepts if is_legacy_manual_claim_id(node_id)
    )
    if all_manual_concepts and legacy_concepts:
        raise ValueError(f"Legacy MANUAL concepts remain: {legacy_concepts[:5]}")
    dangling = []
    for edge in graph.get("edges") or []:
        for key in ("source_id", "target_id"):
            node_id = edge.get(key)
            if node_id and node_id not in concepts:
                dangling.append((key, node_id))
                if len(dangling) >= 5:
                    break
        if len(dangling) >= 5:
            break
    if dangling:
        raise ValueError(f"Dangling graph edge endpoints after migration: {dangling}")
    extracted_ids: set[str] = set()
    legacy_extracted_refs: list[tuple[int, str, str]] = []
    with extracted_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            claim_id = str(row.get("id") or "")
            if not claim_id:
                raise ValueError(f"Missing claim id in {extracted_path} line {number}")
            if claim_id in extracted_ids:
                raise ValueError(f"Duplicate claim id in {extracted_path}: {claim_id}")
            extracted_ids.add(claim_id)
            if all_manual_concepts:
                for key in ("id", "subject_id", "object_id"):
                    value = str(row.get(key) or "")
                    if is_legacy_manual_claim_id(value):
                        legacy_extracted_refs.append((number, key, value))
                        if len(legacy_extracted_refs) >= 5:
                            break
            if len(legacy_extracted_refs) >= 5:
                break
    if legacy_extracted_refs:
        raise ValueError(
            f"Legacy MANUAL archive references remain: {legacy_extracted_refs}"
        )
    graph_only = claim_ids - extracted_ids
    if graph_only:
        raise ValueError(f"Graph claims missing from extracted archive: {sorted(graph_only)[:5]}")
    return {
        "graph_claim_nodes": len(claim_ids),
        "extracted_claims": len(extracted_ids),
        "graph_claims_missing_from_extracted": 0,
        "extracted_candidates_not_in_graph": len(extracted_ids - claim_ids),
        "legacy_manual_claim_nodes_remaining": 0,
        "legacy_manual_concepts_remaining": len(legacy_concepts),
        "legacy_manual_archive_references_remaining": 0,
        "dangling_edge_endpoints": 0,
    }


def atomic_swap_with_rollback(staged: Path, live: Path, backup: Path) -> None:
    if backup.exists():
        raise ValueError(f"Rollback file already exists: {backup}")
    os.replace(live, backup)
    try:
        os.replace(staged, live)
    except Exception:
        os.replace(backup, live)
        raise


def main() -> None:
    args = parse_args()
    migration_name = (
        "canonicalize_non_clm_claim_ids_20260722"
        if args.all_non_clm
        else "canonicalize_manual_concept_ids_20260722"
        if args.all_manual_concepts
        else MIGRATION_NAME
    )
    with args.graph.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    concepts = graph.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError("knowledge_graph.json concepts must be an object")
    mapping = build_mapping(
        concepts,
        all_non_clm=args.all_non_clm,
        all_manual_concepts=args.all_manual_concepts,
    )
    extracted_mapping_counts = {
        "extracted_only_manual_claim_ids_selected": 0,
        "extracted_only_manual_concept_ids_selected": 0,
    }
    if args.all_manual_concepts:
        extracted_mapping_counts = extend_mapping_from_extracted(
            args.extracted, mapping, concepts
        )
    extracted_ids_before = read_jsonl_ids(args.extracted)
    graph_claim_ids_before = {
        node_id for node_id, node in concepts.items() if isinstance(node, dict) and is_claim_node(node)
    }
    report: dict[str, Any] = {
        "migration": migration_name,
        "mode": "apply" if args.apply else "dry_run",
        "graph": str(args.graph.resolve()),
        "extracted": str(args.extracted.resolve()),
        "selected_ids": len(mapping),
        "selected_graph_claim_nodes": sum(
            1 for node_id in mapping if node_id in concepts and is_claim_node(concepts[node_id])
        ),
        "selected_graph_endpoint_concepts": sum(
            1 for node_id in mapping if node_id in concepts and not is_claim_node(concepts[node_id])
        ),
        "legacy_prefix_counts": dict(
            sorted(
                Counter(
                    old_id.split(":", 1)[0]
                    if ":" in old_id
                    else old_id.split("-", 1)[0]
                    for old_id in mapping
                ).items()
            )
        ),
        "mapping_collisions": 0,
        "graph_claim_nodes_before": len(graph_claim_ids_before),
        "extracted_claims_before": len(extracted_ids_before),
        "graph_only_before": len(graph_claim_ids_before - extracted_ids_before),
        "extracted_only_before": len(extracted_ids_before - graph_claim_ids_before),
        "mapping_sample": dict(list(sorted(mapping.items()))[:10]),
    }
    report.update(extracted_mapping_counts)
    if not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    migrated_graph, graph_counts = migrate_graph(graph, mapping, migration_name)
    graph_rows = graph_claim_rows(migrated_graph)
    graph_temp = args.graph.with_suffix(args.graph.suffix + f".{migration_name}.tmp")
    extracted_temp = args.extracted.with_suffix(args.extracted.suffix + f".{migration_name}.tmp")
    graph_backup = args.graph.with_suffix(args.graph.suffix + f".{migration_name}.bak")
    extracted_backup = args.extracted.with_suffix(args.extracted.suffix + f".{migration_name}.bak")
    for path in (graph_temp, extracted_temp):
        if path.exists():
            path.unlink()

    extracted_counts = stage_extracted(args.extracted, extracted_temp, mapping, graph_rows)
    with graph_temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(migrated_graph, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    validation = validate_staged(
        migrated_graph,
        extracted_temp,
        all_non_clm=args.all_non_clm,
        all_manual_concepts=args.all_manual_concepts,
    )

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

    report.update(graph_counts)
    report.update(extracted_counts)
    report["validation"] = validation
    report["rollback_files"] = [str(graph_backup), str(extracted_backup)]
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
