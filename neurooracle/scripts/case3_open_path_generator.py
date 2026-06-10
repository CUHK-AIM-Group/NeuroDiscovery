from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from case3_hindcasting_eval import _is_negative_relation, _role_for_node, load_kg_index


TREE_RELATIONS = {"is_a", "part_of", "about", "supported_by"}
SKIP_DOMAINS = {"claim", "dataset", "atlas", "modality", "ml_model", "recipe"}


@dataclass(order=True)
class Candidate:
    sort_score: float
    source_id: str = field(compare=False)
    mediator_id: str = field(compare=False)
    target_id: str = field(compare=False)
    left_edge: dict[str, Any] = field(compare=False)
    right_edge: dict[str, Any] = field(compare=False)


def _domains(node: dict[str, Any] | None) -> set[str]:
    if not node:
        return set()
    return {str(tag) for tag in (node.get("domain_tags") or [])}


def _usable_node(node: dict[str, Any] | None) -> bool:
    if not node:
        return False
    domains = _domains(node)
    if domains & SKIP_DOMAINS:
        return False
    name = str(node.get("preferred_name") or "").strip()
    return bool(name) and len(name) > 2


def _node_role(node: dict[str, Any] | None) -> str:
    role = _role_for_node(node)
    if role:
        return role
    domains = sorted(_domains(node))
    return domains[0] if domains else "unknown"


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _edge_score(edge: dict[str, Any]) -> float:
    confidence = max(0.01, float(edge.get("confidence") or 0.0))
    claim_bonus = 0.12 if (edge.get("metadata") or {}).get("claim_id") else 0.0
    return min(1.0, confidence + claim_bonus)


def _source_paper(edge: dict[str, Any]) -> dict[str, Any]:
    return dict((edge.get("metadata") or {}).get("source_paper") or {})


def generate(
    kg_path: Path,
    output_path: Path,
    max_candidates: int,
    per_mediator_neighbor_limit: int,
    min_score: float,
) -> dict[str, Any]:
    concepts, edges, names = load_kg_index(kg_path)
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    direct_pairs: set[tuple[str, str]] = set()
    dropped = Counter()
    for edge in edges:
        rel = str(edge.get("relation_type") or "")
        if rel in TREE_RELATIONS or _is_negative_relation(rel):
            dropped["tree_or_negative"] += 1
            continue
        src = str(edge.get("source_id") or "")
        dst = str(edge.get("target_id") or "")
        if not src or not dst or src == dst:
            dropped["bad_edge"] += 1
            continue
        if not _usable_node(concepts.get(src)) or not _usable_node(concepts.get(dst)):
            dropped["unusable_node"] += 1
            continue
        direct_pairs.add(_edge_key(src, dst))
        adjacency[src].append((dst, edge))
        adjacency[dst].append((src, edge))

    heap: list[Candidate] = []
    for mediator_id, nbrs in adjacency.items():
        mediator_node = concepts.get(mediator_id)
        if not _usable_node(mediator_node):
            continue
        ranked = sorted(nbrs, key=lambda item: _edge_score(item[1]), reverse=True)[:per_mediator_neighbor_limit]
        for i, (left_id, left_edge) in enumerate(ranked):
            left_role = _node_role(concepts.get(left_id))
            for right_id, right_edge in ranked[i + 1:]:
                if left_id == right_id:
                    continue
                if _edge_key(left_id, right_id) in direct_pairs:
                    continue
                right_role = _node_role(concepts.get(right_id))
                if left_role == right_role == "unknown":
                    continue
                path_score = math.sqrt(_edge_score(left_edge) * _edge_score(right_edge))
                role_bonus = 0.08 if left_role != right_role else 0.0
                score = min(1.0, path_score + role_bonus)
                if score < min_score:
                    continue
                cand = Candidate(
                    sort_score=score,
                    source_id=left_id,
                    mediator_id=mediator_id,
                    target_id=right_id,
                    left_edge=left_edge,
                    right_edge=right_edge,
                )
                if len(heap) < max_candidates:
                    heapq.heappush(heap, cand)
                elif cand.sort_score > heap[0].sort_score:
                    heapq.heapreplace(heap, cand)

    candidates = sorted(heap, key=lambda c: c.sort_score, reverse=True)
    hypotheses = [_to_hypothesis(i + 1, c, concepts, names) for i, c in enumerate(candidates)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_hypotheses": len(hypotheses),
        "hypotheses": hypotheses,
        "metadata": {
            "generator": "case3_open_path_2hop",
            "kg_path": str(kg_path),
            "max_candidates": max_candidates,
            "per_mediator_neighbor_limit": per_mediator_neighbor_limit,
            "min_score": min_score,
            "dropped": dict(dropped),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _to_hypothesis(index: int, cand: Candidate, concepts: dict[str, dict[str, Any]], names: dict[str, str]) -> dict[str, Any]:
    left_conf = _edge_score(cand.left_edge)
    right_conf = _edge_score(cand.right_edge)
    score = cand.sort_score
    source_name = names.get(cand.source_id, cand.source_id)
    mediator_name = names.get(cand.mediator_id, cand.mediator_id)
    target_name = names.get(cand.target_id, cand.target_id)
    return {
        "id": f"CS3GEN:{index:06d}",
        "hypothesis_type": "open_path",
        "source_id": cand.source_id,
        "source_name": source_name,
        "target_id": cand.target_id,
        "target_name": target_name,
        "path": [
            _link(cand.source_id, cand.mediator_id, cand.left_edge, names, left_conf),
            _link(cand.mediator_id, cand.target_id, cand.right_edge, names, right_conf),
        ],
        "confidence_score": math.sqrt(max(left_conf, 1e-6) * max(right_conf, 1e-6)),
        "novelty_score": 0.65,
        "evidence_score": min(1.0, (left_conf + right_conf) / 2.0),
        "testability_score": 0.5,
        "composite_score": score,
        "supporting_claims": [
            str((cand.left_edge.get("metadata") or {}).get("claim_id") or ""),
            str((cand.right_edge.get("metadata") or {}).get("claim_id") or ""),
        ],
        "explanation": f"{source_name} may relate to {target_name} through {mediator_name}.",
        "metadata": {
            "task_kind": "general_hindcasting",
            "generator": "case3_open_path_2hop",
            "source_role": _node_role(concepts.get(cand.source_id)),
            "mediator_role": _node_role(concepts.get(cand.mediator_id)),
            "target_role": _node_role(concepts.get(cand.target_id)),
        },
    }


def _link(src: str, dst: str, edge: dict[str, Any], names: dict[str, str], confidence: float) -> dict[str, Any]:
    return {
        "from_id": src,
        "from_name": names.get(src, src),
        "to_id": dst,
        "to_name": names.get(dst, dst),
        "relation_type": str(edge.get("relation_type") or "is_associated_with"),
        "confidence": confidence,
        "claim_id": str((edge.get("metadata") or {}).get("claim_id") or ""),
        "raw_text": str((edge.get("metadata") or {}).get("raw_text") or ""),
        "evidence": dict((edge.get("metadata") or {}).get("evidence") or {}),
        "source_paper": _source_paper(edge),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast open-path candidate generator for general Case Study 3 hindcasting.")
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--per-mediator-neighbor-limit", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.20)
    args = parser.parse_args()
    payload = generate(
        kg_path=args.kg,
        output_path=args.output,
        max_candidates=args.max_candidates,
        per_mediator_neighbor_limit=args.per_mediator_neighbor_limit,
        min_score=args.min_score,
    )
    print(json.dumps({"output": str(args.output), "n_hypotheses": payload["n_hypotheses"]}, indent=2))


if __name__ == "__main__":
    main()
