from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

import ijson


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_YEAR_KEYS = {
    "year",
    "publication_year",
    "published_year",
    "release_year",
    "source_year",
    "reference_year",
    "ref_year",
}
_DATED_TEXT_KEY_PARTS = ("ref", "reference", "citation", "source", "publication")
_SNAPSHOT_SOURCE_FILES = ("knowledge_graph.json", "extracted_claims.jsonl", "papers_metadata.csv")
TEMPORAL_SNAPSHOT_SCHEMA_VERSION = "temporal_kg_snapshot.v2"

_CLAIM_ENDPOINT_SOURCE_VOCABS = {
    "claim_extraction",
    "claim_extraction_anchor",
    "manual_claim_anchor",
    "manual_claim_anchor_repair",
    "manual_general_claim_anchor",
    "replay_anchor_mint",
}


def source_fingerprint(input_dir: Path) -> dict[str, dict[str, int]]:
    """Cheap identity check used to invalidate temporal snapshots after KG updates."""

    fingerprint: dict[str, dict[str, int]] = {}
    for name in _SNAPSHOT_SOURCE_FILES:
        path = input_dir / name
        if not path.is_file():
            continue
        stat = path.stat()
        fingerprint[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return fingerprint


def snapshot_matches_input(
    manifest: dict[str, Any],
    input_dir: Path,
    cutoff_year: int,
) -> bool:
    """Return True only when a snapshot was built from the current formal KG."""

    return (
        int(manifest.get("cutoff_year") or -1) == int(cutoff_year)
        and manifest.get("snapshot_schema_version") == TEMPORAL_SNAPSHOT_SCHEMA_VERSION
        and manifest.get("source_fingerprint") == source_fingerprint(input_dir)
    )


def _claim_year(claim: dict[str, Any]) -> int | None:
    year = (claim.get("source_paper") or {}).get("year") or claim.get("year")
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def _claim_id_from_edge(edge: dict[str, Any]) -> str:
    return str((edge.get("metadata") or {}).get("claim_id") or "")


def _is_claim_endpoint_node(node: dict[str, Any]) -> bool:
    """Return whether a concept was minted from a paper claim endpoint."""

    source_vocab = str(node.get("source_vocab") or "")
    domain_tags = {str(tag) for tag in node.get("domain_tags") or []}
    node_id = str(node.get("id") or "")
    return (
        source_vocab in _CLAIM_ENDPOINT_SOURCE_VOCABS
        or "claim_concept" in domain_tags
        or node_id.startswith("CLM_CONCEPT:")
    )


def _is_claim_node(node: dict[str, Any]) -> bool:
    return "claim" in (node.get("domain_tags") or [])


def _years_from_value(value: Any) -> list[int]:
    if isinstance(value, int) and 1900 <= value <= 2099:
        return [value]
    if isinstance(value, str):
        return [int(match) for match in _YEAR_RE.findall(value)]
    return []


def _explicit_evidence_year(record: dict[str, Any]) -> int | None:
    """Return the earliest explicit provenance year attached to a node/edge.

    Numeric scientific values are deliberately ignored unless their key is
    year-like. Free text is inspected only for provenance-like keys, avoiding
    false years from sample sizes or effect metadata.
    """
    years: list[int] = []
    for key, value in record.items():
        key_lower = str(key).lower()
        if key_lower in _YEAR_KEYS:
            years.extend(_years_from_value(value))
        elif isinstance(value, str) and any(part in key_lower for part in _DATED_TEXT_KEY_PARTS):
            years.extend(_years_from_value(value))
        elif key_lower == "metadata" and isinstance(value, dict):
            nested = _explicit_evidence_year(value)
            if nested is not None:
                years.append(nested)
        elif key_lower == "source_paper" and isinstance(value, dict):
            years.extend(_years_from_value(value.get("year")))
    return min(years) if years else None


def _component_count(node_ids: set[str], edges: list[dict[str, Any]]) -> int:
    parent = {nid: nid for nid in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        union(str(edge.get("source_id") or ""), str(edge.get("target_id") or ""))
    return len({find(nid) for nid in node_ids})


def _stats(concepts: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    for node in concepts.values():
        for tag in node.get("domain_tags") or []:
            domains[str(tag)] += 1
        sources[str(node.get("source_vocab") or "")] += 1
    for edge in edges:
        relations[str(edge.get("relation_type") or "unknown")] += 1
    return {
        "n_concepts": len(concepts),
        "n_edges": len(edges),
        "domains": dict(domains),
        "sources": dict(sources),
        "relations": dict(relations),
        "connected_components": _component_count(set(concepts), edges),
    }


def _stream_metadata(graph_path: Path) -> dict[str, Any]:
    with graph_path.open("rb") as handle:
        metadata = next(ijson.items(handle, "metadata", use_float=True), None)
    if not isinstance(metadata, dict):
        raise ValueError(f"knowledge graph has no metadata object: {graph_path}")
    return metadata


def _stream_concepts(graph_path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with graph_path.open("rb") as handle:
        for node_id, node in ijson.kvitems(handle, "concepts", use_float=True):
            if not isinstance(node, dict):
                raise ValueError(f"concept {node_id!r} is not an object")
            yield str(node_id), node


def _stream_edges(graph_path: Path) -> Iterator[dict[str, Any]]:
    with graph_path.open("rb") as handle:
        for edge in ijson.items(handle, "edges.item", use_float=True):
            if not isinstance(edge, dict):
                raise ValueError("knowledge graph edge is not an object")
            yield edge


def _write_fragment_item(handle: TextIO, item: Any, *, first: bool) -> bool:
    if not first:
        handle.write(",")
    json.dump(item, handle, ensure_ascii=False, separators=(",", ":"))
    return False


def _write_concept_fragment_item(
    handle: TextIO,
    node_id: str,
    node: dict[str, Any],
    *,
    first: bool,
) -> bool:
    if not first:
        handle.write(",")
    json.dump(node_id, handle, ensure_ascii=False)
    handle.write(":")
    json.dump(node, handle, ensure_ascii=False, separators=(",", ":"))
    return False


def _sync_text_file(handle: TextIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _copy_text_fragment(source: Path, destination: TextIO) -> None:
    with source.open("r", encoding="utf-8") as handle:
        shutil.copyfileobj(handle, destination, length=16 * 1024 * 1024)


def _progress(label: str, count: int, *, every: int = 250_000) -> None:
    if count and count % every == 0:
        print(f"[{label}] {count:,}", flush=True)


def _find_root(parent: dict[str, str], node_id: str) -> str:
    while parent[node_id] != node_id:
        parent[node_id] = parent[parent[node_id]]
        node_id = parent[node_id]
    return node_id


def _union_roots(parent: dict[str, str], source_id: str, target_id: str) -> None:
    source_root = _find_root(parent, source_id)
    target_root = _find_root(parent, target_id)
    if source_root != target_root:
        parent[target_root] = source_root


def _build_snapshot_in_memory_legacy(
    input_dir: Path, output_dir: Path, cutoff_year: int
) -> dict[str, Any]:
    graph_path = input_dir / "knowledge_graph.json"
    claims_path = input_dir / "extracted_claims.jsonl"
    papers_path = input_dir / "papers_metadata.csv"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if not claims_path.is_file():
        raise FileNotFoundError(claims_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_fingerprint = source_fingerprint(input_dir)
    graph = json.load(graph_path.open("r", encoding="utf-8"))
    concepts_in: dict[str, dict[str, Any]] = graph["concepts"]
    edges_in: list[dict[str, Any]] = graph["edges"]
    graph_claim_ids = {
        node_id for node_id, node in concepts_in.items() if _is_claim_node(node)
    }

    historical_claim_ids: set[str] = set()
    future_claim_ids: set[str] = set()
    historical_pmids: set[str] = set()
    historical_claim_endpoint_ids: set[str] = set()
    claims_before = 0
    claims_after = 0
    missing_year_claims = 0
    excluded_non_kg_claims = 0

    out_claims = output_dir / "extracted_claims.jsonl"
    with claims_path.open("r", encoding="utf-8") as src, out_claims.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            claim = json.loads(line)
            claim_id = str(claim.get("id") or "")
            if claim_id not in graph_claim_ids:
                excluded_non_kg_claims += 1
                continue
            year = _claim_year(claim)
            if year is None:
                missing_year_claims += 1
                future_claim_ids.add(claim_id)
                continue
            if year <= cutoff_year:
                historical_claim_ids.add(claim_id)
                paper = claim.get("source_paper") or {}
                if paper.get("pmid"):
                    historical_pmids.add(str(paper["pmid"]))
                for key in ("subject_id", "object_id"):
                    if claim.get(key):
                        historical_claim_endpoint_ids.add(str(claim[key]))
                dst.write(json.dumps(claim, ensure_ascii=False) + "\n")
                claims_before += 1
            else:
                future_claim_ids.add(claim_id)
                claims_after += 1

    if papers_path.is_file():
        out_papers = output_dir / "papers_metadata.csv"
        with papers_path.open("r", encoding="utf-8", newline="") as src, out_papers.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                try:
                    year = int(row.get("year") or 0)
                except ValueError:
                    continue
                if year <= cutoff_year:
                    writer.writerow(row)

    kept_edges: list[dict[str, Any]] = []
    removed_future_claim_edges = 0
    kept_historical_claim_edges = 0
    kept_non_claim_edges = 0
    kept_dated_non_claim_edges = 0
    kept_undated_non_claim_edges = 0
    removed_future_dated_non_claim_edges = 0

    for edge in edges_in:
        claim_id = _claim_id_from_edge(edge)
        if claim_id:
            if claim_id not in historical_claim_ids:
                removed_future_claim_edges += 1
                continue
            kept_historical_claim_edges += 1
        else:
            evidence_year = _explicit_evidence_year(edge)
            if evidence_year is not None and evidence_year > cutoff_year:
                removed_future_dated_non_claim_edges += 1
                continue
            if evidence_year is None:
                kept_undated_non_claim_edges += 1
            else:
                kept_dated_non_claim_edges += 1
            kept_non_claim_edges += 1
        kept_edges.append(edge)

    kept_concepts: dict[str, dict[str, Any]] = {}
    removed_future_claim_nodes = 0
    removed_orphan_claim_concepts = 0
    removed_future_dated_curated_nodes = 0
    for nid, node in concepts_in.items():
        if _is_claim_node(node):
            if nid in historical_claim_ids:
                kept_concepts[nid] = node
            else:
                removed_future_claim_nodes += 1
            continue
        # Claim-derived endpoint concepts are dated by the retained claim
        # edges that use them. Curated resources may carry their own release
        # year (atlas/ref/source metadata) and must obey the temporal freeze.
        if not _is_claim_endpoint_node(node):
            evidence_year = _explicit_evidence_year(node)
            if evidence_year is not None and evidence_year > cutoff_year:
                removed_future_dated_curated_nodes += 1
                continue
        # A claim-minted endpoint is temporally available only if it appears in
        # an extracted claim at or before the cutoff. Undated bridge/ontology
        # edges must not make a future-minted phrase available retrospectively.
        if _is_claim_endpoint_node(node) and nid not in historical_claim_endpoint_ids:
            removed_orphan_claim_concepts += 1
            continue
        kept_concepts[nid] = node

    edges_before_node_filter = len(kept_edges)
    kept_edges = [
        edge for edge in kept_edges
        if str(edge.get("source_id") or "") in kept_concepts
        and str(edge.get("target_id") or "") in kept_concepts
    ]
    removed_edges_to_future_dated_nodes = edges_before_node_filter - len(kept_edges)

    # An endpoint may look used before an edge is removed because its opposite
    # endpoint is future-dated. Recompute availability from the final edge set.
    post_filter_orphans = {
        nid
        for nid, node in kept_concepts.items()
        if not _is_claim_node(node)
        and _is_claim_endpoint_node(node)
        and nid not in historical_claim_endpoint_ids
    }
    for nid in post_filter_orphans:
        kept_concepts.pop(nid, None)
    removed_orphan_claim_concepts += len(post_filter_orphans)

    metadata = dict(graph.get("metadata") or {})
    metadata["temporal_snapshot"] = {
        "schema_version": TEMPORAL_SNAPSHOT_SCHEMA_VERSION,
        "source_dir": str(input_dir),
        "cutoff_year": cutoff_year,
        "cutoff_policy": f"claim source_paper.year <= {cutoff_year}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Claim-backed edges and claim-domain nodes after the cutoff were removed. "
            "Dated curated nodes and non-claim edges after the cutoff were also removed. "
            "Undated ontology/schema/infrastructure edges were retained and counted in the manifest."
        ),
    }
    metadata["stats"] = _stats(kept_concepts, kept_edges)

    out_graph = {
        "metadata": metadata,
        "concepts": kept_concepts,
        "edges": kept_edges,
    }
    out_graph_path = output_dir / "knowledge_graph.json"
    temp_graph_path = out_graph_path.with_suffix(".json.tmp")
    with temp_graph_path.open("w", encoding="utf-8") as f:
        json.dump(out_graph, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp_graph_path.replace(out_graph_path)

    manifest = {
        "snapshot_schema_version": TEMPORAL_SNAPSHOT_SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cutoff_year": cutoff_year,
        "source_fingerprint": input_fingerprint,
        "claims_kept_year_le_cutoff": claims_before,
        "claims_removed_year_gt_cutoff": claims_after,
        "claims_missing_year_removed": missing_year_claims,
        "excluded_non_kg_extracted_claims": excluded_non_kg_claims,
        "historical_claim_ids": len(historical_claim_ids),
        "future_claim_ids": len(future_claim_ids),
        "historical_pmids": len(historical_pmids),
        "input_concepts": len(concepts_in),
        "output_concepts": len(kept_concepts),
        "removed_future_claim_nodes": removed_future_claim_nodes,
        "removed_future_dated_curated_nodes": removed_future_dated_curated_nodes,
        "removed_orphan_claim_endpoint_concepts": removed_orphan_claim_concepts,
        "removed_orphan_claim_endpoint_concepts_after_edge_filter": len(post_filter_orphans),
        "input_edges": len(edges_in),
        "output_edges": len(kept_edges),
        "kept_historical_claim_edges": kept_historical_claim_edges,
        "kept_non_claim_edges": kept_non_claim_edges,
        "kept_dated_non_claim_edges_year_le_cutoff": kept_dated_non_claim_edges,
        "kept_undated_non_claim_edges": kept_undated_non_claim_edges,
        "removed_future_dated_non_claim_edges": removed_future_dated_non_claim_edges,
        "removed_edges_to_future_dated_nodes": removed_edges_to_future_dated_nodes,
        "removed_future_claim_edges": removed_future_claim_edges,
        "output_stats": metadata["stats"],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _build_snapshot_streaming(
    input_dir: Path, output_dir: Path, cutoff_year: int
) -> dict[str, Any]:
    graph_path = input_dir / "knowledge_graph.json"
    claims_path = input_dir / "extracted_claims.jsonl"
    papers_path = input_dir / "papers_metadata.csv"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if not claims_path.is_file():
        raise FileNotFoundError(claims_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_fingerprint = source_fingerprint(input_dir)
    metadata = _stream_metadata(graph_path)

    # A canonical release can be many gigabytes. json.load() first expands the
    # complete UTF-8 file and then materializes every node and edge. Streaming
    # retains only identifiers and counters required by the temporal filter.
    graph_claim_ids: set[str] = set()
    input_concepts = 0
    print("[stream] index claim nodes", flush=True)
    for node_id, node in _stream_concepts(graph_path):
        input_concepts += 1
        if _is_claim_node(node):
            graph_claim_ids.add(node_id)
        _progress("claim-node-index", input_concepts)

    historical_claim_ids: set[str] = set()
    future_claim_ids: set[str] = set()
    historical_pmids: set[str] = set()
    historical_claim_endpoint_ids: set[str] = set()
    claims_before = 0
    claims_after = 0
    missing_year_claims = 0
    excluded_non_kg_claims = 0
    claims_seen = 0

    out_claims = output_dir / "extracted_claims.jsonl"
    temp_claims_path = output_dir / ".extracted_claims.jsonl.tmp"
    print("[stream] filter extracted claims", flush=True)
    with claims_path.open("r", encoding="utf-8") as src, temp_claims_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            claims_seen += 1
            claim = json.loads(line)
            claim_id = str(claim.get("id") or "")
            if claim_id not in graph_claim_ids:
                excluded_non_kg_claims += 1
                _progress("claims", claims_seen)
                continue
            year = _claim_year(claim)
            if year is None:
                missing_year_claims += 1
                future_claim_ids.add(claim_id)
                _progress("claims", claims_seen)
                continue
            if year <= cutoff_year:
                historical_claim_ids.add(claim_id)
                paper = claim.get("source_paper") or {}
                if paper.get("pmid"):
                    historical_pmids.add(str(paper["pmid"]))
                for key in ("subject_id", "object_id"):
                    if claim.get(key):
                        historical_claim_endpoint_ids.add(str(claim[key]))
                dst.write(json.dumps(claim, ensure_ascii=False) + "\n")
                claims_before += 1
            else:
                future_claim_ids.add(claim_id)
                claims_after += 1
            _progress("claims", claims_seen)
        _sync_text_file(dst)

    future_claim_ids_count = len(future_claim_ids)
    historical_pmids_count = len(historical_pmids)
    del graph_claim_ids, future_claim_ids, historical_pmids
    gc.collect()

    temp_papers_path: Path | None = None
    if papers_path.is_file():
        temp_papers_path = output_dir / ".papers_metadata.csv.tmp"
        with papers_path.open("r", encoding="utf-8", newline="") as src, temp_papers_path.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                try:
                    year = int(row.get("year") or 0)
                except ValueError:
                    continue
                if year <= cutoff_year:
                    writer.writerow(row)
            _sync_text_file(dst)

    concepts_fragment_path = output_dir / ".concepts.json.tmp"
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    component_parent: dict[str, str] = {}
    output_concepts = 0
    removed_future_claim_nodes = 0
    removed_orphan_claim_concepts = 0
    removed_future_dated_curated_nodes = 0
    concepts_seen = 0
    first_concept = True

    print("[stream] filter concepts", flush=True)
    with concepts_fragment_path.open("w", encoding="utf-8") as concepts_out:
        for node_id, node in _stream_concepts(graph_path):
            concepts_seen += 1
            keep = True
            if _is_claim_node(node):
                if node_id not in historical_claim_ids:
                    removed_future_claim_nodes += 1
                    keep = False
            else:
                if not _is_claim_endpoint_node(node):
                    evidence_year = _explicit_evidence_year(node)
                    if evidence_year is not None and evidence_year > cutoff_year:
                        removed_future_dated_curated_nodes += 1
                        keep = False
                if (
                    keep
                    and _is_claim_endpoint_node(node)
                    and node_id not in historical_claim_endpoint_ids
                ):
                    removed_orphan_claim_concepts += 1
                    keep = False

            if keep:
                first_concept = _write_concept_fragment_item(
                    concepts_out,
                    node_id,
                    node,
                    first=first_concept,
                )
                component_parent[node_id] = node_id
                output_concepts += 1
                for tag in node.get("domain_tags") or []:
                    domains[str(tag)] += 1
                sources[str(node.get("source_vocab") or "")] += 1
            _progress("concepts", concepts_seen)
        _sync_text_file(concepts_out)

    if concepts_seen != input_concepts:
        raise RuntimeError(
            "knowledge graph changed while streaming concepts: "
            f"first_pass={input_concepts}, second_pass={concepts_seen}"
        )
    del historical_claim_endpoint_ids
    gc.collect()

    edges_fragment_path = output_dir / ".edges.json.tmp"
    removed_future_claim_edges = 0
    kept_historical_claim_edges = 0
    kept_non_claim_edges = 0
    kept_dated_non_claim_edges = 0
    kept_undated_non_claim_edges = 0
    removed_future_dated_non_claim_edges = 0
    removed_edges_to_future_dated_nodes = 0
    relations: Counter[str] = Counter()
    input_edges = 0
    output_edges = 0
    first_edge = True

    print("[stream] filter edges and compute components", flush=True)
    with edges_fragment_path.open("w", encoding="utf-8") as edges_out:
        for edge in _stream_edges(graph_path):
            input_edges += 1
            claim_id = _claim_id_from_edge(edge)
            if claim_id:
                if claim_id not in historical_claim_ids:
                    removed_future_claim_edges += 1
                    _progress("edges", input_edges)
                    continue
                kept_historical_claim_edges += 1
            else:
                evidence_year = _explicit_evidence_year(edge)
                if evidence_year is not None and evidence_year > cutoff_year:
                    removed_future_dated_non_claim_edges += 1
                    _progress("edges", input_edges)
                    continue
                if evidence_year is None:
                    kept_undated_non_claim_edges += 1
                else:
                    kept_dated_non_claim_edges += 1
                kept_non_claim_edges += 1

            source_id = str(edge.get("source_id") or "")
            target_id = str(edge.get("target_id") or "")
            if source_id not in component_parent or target_id not in component_parent:
                removed_edges_to_future_dated_nodes += 1
                _progress("edges", input_edges)
                continue

            first_edge = _write_fragment_item(edges_out, edge, first=first_edge)
            _union_roots(component_parent, source_id, target_id)
            relations[str(edge.get("relation_type") or "unknown")] += 1
            output_edges += 1
            _progress("edges", input_edges)
        _sync_text_file(edges_out)

    historical_claim_ids_count = len(historical_claim_ids)
    del historical_claim_ids
    gc.collect()
    print("[stream] finalize component count", flush=True)
    connected_components = sum(
        1
        for node_id in component_parent
        if _find_root(component_parent, node_id) == node_id
    )
    del component_parent
    gc.collect()

    metadata = dict(metadata)
    metadata["temporal_snapshot"] = {
        "schema_version": TEMPORAL_SNAPSHOT_SCHEMA_VERSION,
        "source_dir": str(input_dir),
        "cutoff_year": cutoff_year,
        "cutoff_policy": f"claim source_paper.year <= {cutoff_year}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Claim-backed edges and claim-domain nodes after the cutoff were removed. "
            "Dated curated nodes and non-claim edges after the cutoff were also removed. "
            "Undated ontology/schema/infrastructure edges were retained and counted in the manifest."
        ),
    }
    metadata["stats"] = {
        "n_concepts": output_concepts,
        "n_edges": output_edges,
        "domains": dict(domains),
        "sources": dict(sources),
        "relations": dict(relations),
        "connected_components": connected_components,
    }

    out_graph_path = output_dir / "knowledge_graph.json"
    temp_graph_path = output_dir / ".knowledge_graph.json.tmp"
    print("[stream] assemble snapshot graph", flush=True)
    with temp_graph_path.open("w", encoding="utf-8") as handle:
        handle.write('{"metadata":')
        json.dump(metadata, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write(',"concepts":{')
        _copy_text_fragment(concepts_fragment_path, handle)
        handle.write('},"edges":[')
        _copy_text_fragment(edges_fragment_path, handle)
        handle.write("]}")
        _sync_text_file(handle)

    if source_fingerprint(input_dir) != input_fingerprint:
        raise RuntimeError("snapshot source changed during streaming build")

    manifest = {
        "snapshot_schema_version": TEMPORAL_SNAPSHOT_SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cutoff_year": cutoff_year,
        "source_fingerprint": input_fingerprint,
        "claims_kept_year_le_cutoff": claims_before,
        "claims_removed_year_gt_cutoff": claims_after,
        "claims_missing_year_removed": missing_year_claims,
        "excluded_non_kg_extracted_claims": excluded_non_kg_claims,
        "historical_claim_ids": historical_claim_ids_count,
        "future_claim_ids": future_claim_ids_count,
        "historical_pmids": historical_pmids_count,
        "input_concepts": input_concepts,
        "output_concepts": output_concepts,
        "removed_future_claim_nodes": removed_future_claim_nodes,
        "removed_future_dated_curated_nodes": removed_future_dated_curated_nodes,
        "removed_orphan_claim_endpoint_concepts": removed_orphan_claim_concepts,
        "removed_orphan_claim_endpoint_concepts_after_edge_filter": 0,
        "input_edges": input_edges,
        "output_edges": output_edges,
        "kept_historical_claim_edges": kept_historical_claim_edges,
        "kept_non_claim_edges": kept_non_claim_edges,
        "kept_dated_non_claim_edges_year_le_cutoff": kept_dated_non_claim_edges,
        "kept_undated_non_claim_edges": kept_undated_non_claim_edges,
        "removed_future_dated_non_claim_edges": removed_future_dated_non_claim_edges,
        "removed_edges_to_future_dated_nodes": removed_edges_to_future_dated_nodes,
        "removed_future_claim_edges": removed_future_claim_edges,
        "output_stats": metadata["stats"],
    }
    temp_manifest_path = output_dir / ".manifest.json.tmp"
    with temp_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        _sync_text_file(handle)

    # The manifest is the completion marker. Publish it last so an interrupted
    # directory is never accepted as a reusable snapshot.
    temp_graph_path.replace(out_graph_path)
    temp_claims_path.replace(out_claims)
    if temp_papers_path is not None:
        temp_papers_path.replace(output_dir / "papers_metadata.csv")
    temp_manifest_path.replace(output_dir / "manifest.json")
    concepts_fragment_path.unlink()
    edges_fragment_path.unlink()
    return manifest


def build_snapshot(input_dir: Path, output_dir: Path, cutoff_year: int) -> dict[str, Any]:
    """Build a temporal snapshot without materializing the canonical KG."""

    output_dir = Path(output_dir)
    final_paths = (
        output_dir / "knowledge_graph.json",
        output_dir / "extracted_claims.jsonl",
        output_dir / "papers_metadata.csv",
        output_dir / "manifest.json",
    )
    temporary_paths = (
        output_dir / ".knowledge_graph.json.tmp",
        output_dir / ".extracted_claims.jsonl.tmp",
        output_dir / ".papers_metadata.csv.tmp",
        output_dir / ".manifest.json.tmp",
        output_dir / ".concepts.json.tmp",
        output_dir / ".edges.json.tmp",
    )
    preexisting = {path: path.exists() for path in final_paths}
    for path in temporary_paths:
        path.unlink(missing_ok=True)
    try:
        return _build_snapshot_streaming(input_dir, output_dir, cutoff_year)
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        for path, existed in preexisting.items():
            if not existed:
                path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a year-truncated NeuroOracle KG snapshot.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("neurooracle/data/full_v2"),
        help="Directory containing knowledge_graph.json and extracted_claims.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neurooracle/data/experiments/hindcasting/snapshots/kg_2020"),
        help="Directory for the temporal snapshot.",
    )
    parser.add_argument("--cutoff-year", type=int, default=2020)
    args = parser.parse_args()
    manifest = build_snapshot(args.input_dir, args.output_dir, args.cutoff_year)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Last updated: 2026-08-12 17:54 HKT - enforce claim-anchor temporal availability and version snapshots.
