from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from case3_hindcasting_eval import (
    TREE_RELATIONS,
    _is_negative_relation,
    _is_specific_imaging_node,
    _is_stable_disease_node,
    _role_for_node,
    build_pair_evidence,
    load_kg_index,
)
from generate_ai_scientist_gid_hypotheses import (
    failure_placeholder,
    load_env_keys,
    normalize_hypothesis,
)
from generate_coscientist_style_gid_hypotheses import call_with_system, rows_from_payload


SYSTEM_PROMPT = """You are reproducing the hypothesis-generation component of a
paper-grounded SciAgents adaptation, based on the Advanced Materials paper
"SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent
Intelligent Graph Reasoning".

Use graph reasoning rather than free brainstorming:

1. Graph ontologist: inspect the supplied KG path/neighborhood cards.
2. Scientist agents: propose mechanistic links suggested by the graph context.
3. Critic agent: reject vague modality-only markers and weak generic claims.
4. Ranker: return the strongest fixed-budget hypotheses first.

You may use only the supplied frozen historical graph context and general
biomedical knowledge. Do not use NeuroDiscovery scores, KGE scores, future
labels, graph-degree-only ranking, or any closed-loop feedback. JSON only."""


CANONICAL_MARKER_TEXT = """Canonical imaging-variable vocabulary:
- cortical thickness
- cortical surface area
- regional volume
- gray matter density
- fractional anisotropy
- mean diffusivity
- radial diffusivity
- axial diffusivity
- functional connectivity
- amplitude of low-frequency fluctuation
- regional homogeneity
- task BOLD amplitude
- amyloid SUVR
- tau SUVR
- FDG uptake

Write markers as either a canonical variable or
"<canonical variable> in <brain region>" /
"functional connectivity between <region/network> and <region/network>".
Do not write modality-only markers such as MRI, fMRI, PET, SPECT, CT, scan, or
bare anatomical regions."""


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def node_name(concepts: dict[str, dict[str, Any]], node_id: str) -> str:
    node = concepts.get(node_id) or {}
    return str(node.get("preferred_name") or node_id)


def edge_label(edge: dict[str, Any]) -> str:
    rel = str(edge.get("relation_type") or "related_to")
    src = str(edge.get("source") or "")
    if src:
        return f"{rel} [{src}]"
    return rel


def build_graph(
    concepts: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, list[tuple[str, str]]], Counter[str], dict[tuple[str, str], str]]:
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    degrees: Counter[str] = Counter()
    edge_names: dict[tuple[str, str], str] = {}
    for edge in edges:
        rel = str(edge.get("relation_type") or "")
        if rel in TREE_RELATIONS or _is_negative_relation(rel):
            continue
        source = str(edge.get("source_id") or "")
        target = str(edge.get("target_id") or "")
        if not source or not target or source == target:
            continue
        if source not in concepts or target not in concepts:
            continue
        source_domains = set(concepts[source].get("domain_tags") or [])
        target_domains = set(concepts[target].get("domain_tags") or [])
        if "claim" in source_domains or "claim" in target_domains:
            continue
        label = edge_label(edge)
        adj[source].append((target, label))
        adj[target].append((source, label))
        degrees[source] += 1
        degrees[target] += 1
        edge_names[(source, target)] = label
        edge_names[(target, source)] = label
    return dict(adj), degrees, edge_names


def shortest_path(
    adj: dict[str, list[tuple[str, str]]],
    start: str,
    target: str,
    max_depth: int = 4,
    max_neighbors: int = 80,
) -> list[str]:
    if start == target:
        return [start]
    seen = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for nxt, _label in adj.get(node, [])[:max_neighbors]:
            if nxt in seen:
                continue
            next_path = [*path, nxt]
            if nxt == target:
                return next_path
            seen.add(nxt)
            queue.append((nxt, next_path))
    return []


def path_to_text(
    concepts: dict[str, dict[str, Any]],
    edge_names: dict[tuple[str, str], str],
    path: list[str],
) -> str:
    if not path:
        return ""
    parts = [node_name(concepts, path[0])]
    for a, b in zip(path[:-1], path[1:], strict=True):
        parts.append(f"--{edge_names.get((a, b), 'related_to')}-->")
        parts.append(node_name(concepts, b))
    return " ".join(parts)


def weighted_choice(items: list[str], degrees: Counter[str], rng: random.Random) -> str:
    weights = [max(1, min(80, degrees.get(item, 1))) for item in items]
    return rng.choices(items, weights=weights, k=1)[0]


def clean_marker_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180]


def sample_graph_cards(
    *,
    concepts: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    n_cards: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    adj, degrees, edge_names = build_graph(concepts, edges)
    gene_imaging, imaging_disease, gene_disease, _direct = build_pair_evidence(
        concepts, edges, strict_candidate_anchors=True
    )
    gi_by_gene: dict[str, list[str]] = defaultdict(list)
    id_by_disease: dict[str, list[str]] = defaultdict(list)
    gd_by_gene: dict[str, list[str]] = defaultdict(list)
    for gene, imaging in gene_imaging:
        gi_by_gene[gene].append(imaging)
    for imaging, disease in imaging_disease:
        id_by_disease[disease].append(imaging)
    for gene, disease in gene_disease:
        gd_by_gene[gene].append(disease)

    genes = [gene for gene in gi_by_gene if concepts.get(gene)]
    diseases = [
        disease
        for disease in id_by_disease
        if _is_stable_disease_node(disease, concepts.get(disease))
    ]
    if not genes:
        genes = [nid for nid, node in concepts.items() if _role_for_node(node) == "gene"]
    if not diseases:
        diseases = [nid for nid, node in concepts.items() if _role_for_node(node) == "disease"]

    cards: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    attempts = 0
    while len(cards) < n_cards and attempts < n_cards * 30:
        attempts += 1
        gene = weighted_choice(genes, degrees, rng)
        # Mix graph-near diseases with broad disease samples.
        near_diseases = [d for d in gd_by_gene.get(gene, []) if d in diseases]
        disease = rng.choice(near_diseases) if near_diseases and rng.random() < 0.45 else weighted_choice(diseases, degrees, rng)
        if (gene, disease) in used_pairs:
            continue
        used_pairs.add((gene, disease))
        gene_markers = sorted(
            set(gi_by_gene.get(gene, [])),
            key=lambda marker: degrees.get(marker, 0),
            reverse=True,
        )[:8]
        disease_markers = sorted(
            set(id_by_disease.get(disease, [])),
            key=lambda marker: degrees.get(marker, 0),
            reverse=True,
        )[:8]
        common = sorted(set(gene_markers) & set(disease_markers), key=lambda marker: degrees.get(marker, 0), reverse=True)
        path = shortest_path(adj, gene, disease, max_depth=4)
        if not (gene_markers or disease_markers or path):
            continue
        cards.append(
            {
                "gene": node_name(concepts, gene),
                "disease": node_name(concepts, disease),
                "gene_imaging_neighbors": [
                    clean_marker_name(node_name(concepts, marker)) for marker in gene_markers[:5]
                ],
                "disease_imaging_neighbors": [
                    clean_marker_name(node_name(concepts, marker)) for marker in disease_markers[:5]
                ],
                "shared_imaging_neighbors": [
                    clean_marker_name(node_name(concepts, marker)) for marker in common[:5]
                ],
                "short_graph_path": path_to_text(concepts, edge_names, path),
            }
        )
    return cards


def prompt_for_batch(batch_n: int, start_rank: int, previous_names: list[str], cards: list[dict[str, Any]]) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    return f"""Generate exactly {batch_n} new GENE-IMAGING-DISEASE hypotheses for ranks {start_rank} to {start_rank + batch_n - 1}.

{CANONICAL_MARKER_TEXT}

Already generated hypothesis names:
{previous}

Frozen historical KG path/neighborhood cards:
{compact_json({"cards": cards})}

Follow a SciAgents graph-reasoning procedure:
1. Use the graph cards as the starting point.
2. Prefer hypotheses that bridge a gene and disease through a plausible imaging marker.
3. A marker can come from gene_imaging_neighbors, disease_imaging_neighbors, shared_imaging_neighbors, or a cautious graph-path extrapolation.
4. Avoid repeating the same gene/disease/marker pattern across ranks.
5. Rank stronger, more graph-grounded hypotheses first.

Return one JSON object:
{{
  "graph_reasoning_summary": "one concise paragraph",
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence grounded in the graph card",
      "expected_direction": "optional concise direction",
      "graph_card_used": "gene/disease or short card reference",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Every object must attempt GENE_TARGET -> IMAGING_MARKER -> DISEASE.
- Use concrete measured imaging markers, not modality names.
- Confidence should be a number from 0 to 1.
- JSON only; no markdown."""


def normalize_sciagents_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    hyp = normalize_hypothesis(row, rank)
    hyp["graph_card_used"] = str(row.get("graph_card_used") or "")
    hyp["baseline"] = "sciagents"
    return hyp


def generate_batch(
    args: argparse.Namespace,
    batch_n: int,
    start_rank: int,
    previous_names: list[str],
    cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            payload = call_with_system(
                model=args.model,
                system=SYSTEM_PROMPT,
                prompt=prompt_for_batch(batch_n, start_rank, previous_names, cards),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rows = rows_from_payload(payload)
            if rows:
                return rows[:batch_n], payload, ""
            raise ValueError("No hypotheses returned.")
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
    return [], {}, last_error


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_env_keys()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set and no usable key was found in .env.keys")

    concepts, edges, _names = load_kg_index(args.kg)
    hypotheses: list[dict[str, Any]] = []
    names: list[str] = []
    reasoning_logs: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            hypotheses = [row for row in existing if isinstance(row, dict)]
            names = [str(row.get("Name") or row.get("name") or "") for row in hypotheses]
            print(f"resuming from {len(hypotheses)}/{args.n} hypotheses -> {args.output}")

    while len(hypotheses) < args.n:
        start_rank = len(hypotheses) + 1
        batch_n = min(args.batch_size, args.n - len(hypotheses))
        cards = sample_graph_cards(
            concepts=concepts,
            edges=edges,
            n_cards=max(args.cards_per_batch, batch_n),
            seed=args.seed + start_rank,
        )
        rows, payload, error = generate_batch(args, batch_n, start_rank, names, cards)
        if len(rows) < batch_n:
            rows.extend({"generation_failure": error or "model returned too few hypotheses"} for _ in range(batch_n - len(rows)))
        for row in rows[:batch_n]:
            rank = len(hypotheses) + 1
            if row.get("generation_failure"):
                hyp = failure_placeholder(rank, str(row["generation_failure"]))
                hyp["baseline"] = "sciagents"
            else:
                hyp = normalize_sciagents_row(row, rank)
            hypotheses.append(hyp)
            names.append(str(hyp["Name"]))
        reasoning_logs.append(
            {
                "start_rank": start_rank,
                "batch_n": batch_n,
                "cards": cards,
                "graph_reasoning_summary": payload.get("graph_reasoning_summary", "") if payload else "",
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output.parent / "graph_reasoning_logs.json").write_text(
            json.dumps(reasoning_logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"generated {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-budget SciAgents GID hypotheses.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--cards-per-batch", type=int, default=30)
    parser.add_argument("--model", default=os.environ.get("SCIAGENTS_STYLE_MODEL", "gpt-5.4"))
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--kg",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/snapshots_full_v2_5year_2016_2020/kg_2020/knowledge_graph.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/sciagents_fixed_1000/hypotheses.json"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
