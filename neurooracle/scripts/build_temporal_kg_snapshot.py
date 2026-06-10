from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _claim_year(claim: dict[str, Any]) -> int | None:
    year = (claim.get("source_paper") or {}).get("year") or claim.get("year")
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def _claim_id_from_edge(edge: dict[str, Any]) -> str:
    return str((edge.get("metadata") or {}).get("claim_id") or "")


def _is_claim_extraction_node(node: dict[str, Any]) -> bool:
    return str(node.get("source_vocab") or "") == "claim_extraction"


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


def build_snapshot(input_dir: Path, output_dir: Path, cutoff_year: int) -> dict[str, Any]:
    graph_path = input_dir / "knowledge_graph.json"
    claims_path = input_dir / "extracted_claims.jsonl"
    papers_path = input_dir / "papers_metadata.csv"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if not claims_path.is_file():
        raise FileNotFoundError(claims_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    historical_claim_ids: set[str] = set()
    future_claim_ids: set[str] = set()
    historical_pmids: set[str] = set()
    historical_claim_endpoint_ids: set[str] = set()
    claims_before = 0
    claims_after = 0
    missing_year_claims = 0

    out_claims = output_dir / "extracted_claims.jsonl"
    with claims_path.open("r", encoding="utf-8") as src, out_claims.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            claim = json.loads(line)
            year = _claim_year(claim)
            if year is None:
                missing_year_claims += 1
                future_claim_ids.add(str(claim.get("id") or ""))
                continue
            claim_id = str(claim.get("id") or "")
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

    graph = json.load(graph_path.open("r", encoding="utf-8"))
    concepts_in: dict[str, dict[str, Any]] = graph["concepts"]
    edges_in: list[dict[str, Any]] = graph["edges"]

    kept_edges: list[dict[str, Any]] = []
    used_node_ids: set[str] = set(historical_claim_endpoint_ids)
    removed_future_claim_edges = 0
    kept_historical_claim_edges = 0
    kept_non_claim_edges = 0

    for edge in edges_in:
        claim_id = _claim_id_from_edge(edge)
        if claim_id:
            if claim_id not in historical_claim_ids:
                removed_future_claim_edges += 1
                continue
            kept_historical_claim_edges += 1
        else:
            kept_non_claim_edges += 1
        kept_edges.append(edge)
        if edge.get("source_id"):
            used_node_ids.add(str(edge["source_id"]))
        if edge.get("target_id"):
            used_node_ids.add(str(edge["target_id"]))

    kept_concepts: dict[str, dict[str, Any]] = {}
    removed_future_claim_nodes = 0
    removed_orphan_claim_concepts = 0
    for nid, node in concepts_in.items():
        if nid.startswith("CLM:"):
            if nid in historical_claim_ids:
                kept_concepts[nid] = node
            else:
                removed_future_claim_nodes += 1
            continue
        if _is_claim_extraction_node(node) and nid not in used_node_ids:
            removed_orphan_claim_concepts += 1
            continue
        kept_concepts[nid] = node

    kept_edges = [
        edge for edge in kept_edges
        if str(edge.get("source_id") or "") in kept_concepts
        and str(edge.get("target_id") or "") in kept_concepts
    ]

    metadata = dict(graph.get("metadata") or {})
    metadata["temporal_snapshot"] = {
        "source_dir": str(input_dir),
        "cutoff_year": cutoff_year,
        "cutoff_policy": f"claim source_paper.year <= {cutoff_year}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Claim-backed edges and CLM:* nodes after the cutoff were removed. "
            "Non-claim ontology, dataset, atlas, modality, and curated infrastructure edges were retained."
        ),
    }
    metadata["stats"] = _stats(kept_concepts, kept_edges)

    out_graph = {
        "metadata": metadata,
        "concepts": kept_concepts,
        "edges": kept_edges,
    }
    out_graph_path = output_dir / "knowledge_graph.json"
    with out_graph_path.open("w", encoding="utf-8") as f:
        json.dump(out_graph, f, ensure_ascii=False)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "cutoff_year": cutoff_year,
        "claims_kept_year_le_cutoff": claims_before,
        "claims_removed_year_gt_cutoff": claims_after,
        "claims_missing_year_removed": missing_year_claims,
        "historical_claim_ids": len(historical_claim_ids),
        "future_claim_ids": len(future_claim_ids),
        "historical_pmids": len(historical_pmids),
        "input_concepts": len(concepts_in),
        "output_concepts": len(kept_concepts),
        "removed_future_claim_nodes": removed_future_claim_nodes,
        "removed_orphan_claim_extraction_concepts": removed_orphan_claim_concepts,
        "input_edges": len(edges_in),
        "output_edges": len(kept_edges),
        "kept_historical_claim_edges": kept_historical_claim_edges,
        "kept_non_claim_edges": kept_non_claim_edges,
        "removed_future_claim_edges": removed_future_claim_edges,
        "output_stats": metadata["stats"],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a year-truncated NeuroOracle KG snapshot.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("neurooracle/data/full_snapshot_v1"),
        help="Directory containing knowledge_graph.json and extracted_claims.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neurooracle/data/snapshots/kg_2020_from_full_snapshot_v1"),
        help="Directory for the temporal snapshot.",
    )
    parser.add_argument("--cutoff-year", type=int, default=2020)
    args = parser.parse_args()
    manifest = build_snapshot(args.input_dir, args.output_dir, args.cutoff_year)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
