from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from case3_gid_baseline_comparison import make_negatives, write_csv
from case3_hindcasting_eval import (
    _auc,
    _auprc,
    _role_for_node,
    load_future_claim_pairs,
    load_kg_index,
)


TOP_KS = (1, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000)


MANUAL_EXTRACTION_RULES = (
    {
        "name_contains": "gba",
        "gene_terms": ("GBA", "glucocerebrosidase"),
        "imaging_terms": (
            "posterior putamen dopamine transporter specific binding ratio",
            "posterior putamen DAT",
            "dopamine transporter binding",
            "DaTSCAN SPECT",
        ),
        "disease_terms": ("Parkinson Disease", "Parkinson's disease", "Parkinsons disease"),
    },
)


def norm(text: str) -> str:
    text = text.casefold()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_score(query: str, candidate: str) -> float:
    q = norm(query)
    c = norm(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if q in c or c in q:
        return min(0.98, 0.78 + 0.20 * min(len(q), len(c)) / max(len(q), len(c)))
    q_tokens = set(q.split())
    c_tokens = set(c.split())
    jaccard = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
    seq = SequenceMatcher(None, q, c).ratio()
    return 0.55 * jaccard + 0.45 * seq


def role_index(concepts: dict[str, dict[str, Any]], role: str) -> list[dict[str, Any]]:
    rows = []
    for node_id, node in concepts.items():
        if _role_for_node(node) != role:
            continue
        name = str(node.get("preferred_name") or node_id)
        aliases = [str(x) for x in (node.get("aliases") or [])]
        candidates = [name, *aliases, node_id]
        normalized = [norm(candidate) for candidate in candidates if norm(candidate)]
        tokens = set()
        for normalized_candidate in normalized:
            tokens.update(normalized_candidate.split())
        rows.append({
            "id": node_id,
            "name": name,
            "aliases": aliases,
            "candidates": candidates,
            "normalized": normalized,
            "tokens": tokens,
        })
    return rows


def best_match(
    index: list[dict[str, Any]],
    terms: tuple[str, ...],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    query_norms = [norm(term) for term in terms if norm(term)]
    if not query_norms:
        return None, []
    for row in index:
        for query in query_norms:
            if query in row["normalized"]:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "score": 1.0,
                    "term": query,
                    "matched_text": query,
                }, [{
                    "id": row["id"],
                    "name": row["name"],
                    "score": 1.0,
                    "term": query,
                    "matched_text": query,
                }]
    query_tokens = set()
    for query in query_norms:
        query_tokens.update(query.split())
    pool = [row for row in index if row["tokens"] & query_tokens]
    if not pool:
        pool = index
    scored = []
    for row in pool:
        candidates = row["candidates"]
        best = 0.0
        best_term = ""
        best_text = ""
        for term in terms:
            for candidate in candidates:
                score = token_score(term, candidate)
                if score > best:
                    best = score
                    best_term = term
                    best_text = candidate
        scored.append({
            "id": row["id"],
            "name": row["name"],
            "score": best,
            "term": best_term,
            "matched_text": best_text,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return (scored[0] if scored else None), scored[:10]


def load_external_ideas(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "hypotheses" in payload:
        return list(payload["hypotheses"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported external hypothesis format: {path}")


def idea_text(idea: dict[str, Any]) -> str:
    parts = [
        str(idea.get("Name") or idea.get("name") or ""),
        str(idea.get("Title") or idea.get("title") or ""),
        str(idea.get("Short Hypothesis") or idea.get("hypothesis") or ""),
        str(idea.get("Abstract") or ""),
        " ".join(str(x) for x in idea.get("Experiments", []) if isinstance(x, str)),
    ]
    return "\n".join(parts)


def extract_terms(idea: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    explicit = {
        "gene": field_terms(idea, ("gene_target", "gene", "gene_name", "pathway", "target")),
        "imaging": field_terms(idea, ("imaging_marker", "imaging", "marker", "neuroimaging_marker")),
        "disease": field_terms(idea, ("disease", "disease_name", "outcome")),
    }
    if any(explicit.values()):
        return explicit
    text = idea_text(idea)
    low = norm(text)
    for rule in MANUAL_EXTRACTION_RULES:
        if rule["name_contains"] in low:
            return {
                "gene": tuple(rule["gene_terms"]),
                "imaging": tuple(rule["imaging_terms"]),
                "disease": tuple(rule["disease_terms"]),
            }
    return {"gene": (), "imaging": (), "disease": ()}


def field_terms(idea: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    terms = []
    for key in keys:
        value = idea.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            terms.append(value)
        elif isinstance(value, list):
            terms.extend(str(x) for x in value if str(x).strip())
        elif isinstance(value, dict):
            terms.extend(str(x) for x in value.values() if str(x).strip())
        else:
            terms.append(str(value))
    return tuple(dict.fromkeys(term.strip() for term in terms if term.strip()))


def ground_ideas(
    ideas: list[dict[str, Any]],
    concepts: dict[str, dict[str, Any]],
    min_match: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexes = {
        "gene": role_index(concepts, "gene"),
        "imaging": role_index(concepts, "imaging"),
        "disease": role_index(concepts, "disease"),
    }
    rows = []
    diagnostics = []
    for rank, idea in enumerate(ideas, start=1):
        terms = extract_terms(idea)
        matches = {}
        candidate_matches = {}
        for role in ("gene", "imaging", "disease"):
            best, top = best_match(indexes[role], terms[role]) if terms[role] else (None, [])
            matches[role] = best
            candidate_matches[role] = top
        valid = all(matches[role] and matches[role]["score"] >= min_match for role in ("gene", "imaging", "disease"))
        score = 1.0 / rank
        row = {
            "rank": rank,
            "score": score,
            "name": idea.get("Name") or idea.get("name") or f"external_{rank}",
            "title": idea.get("Title") or idea.get("title") or "",
            "valid_gid": bool(valid),
            "gene_id": matches["gene"]["id"] if matches["gene"] else "",
            "gene_name": matches["gene"]["name"] if matches["gene"] else "",
            "gene_match": matches["gene"]["score"] if matches["gene"] else 0.0,
            "imaging_id": matches["imaging"]["id"] if matches["imaging"] else "",
            "imaging_name": matches["imaging"]["name"] if matches["imaging"] else "",
            "imaging_match": matches["imaging"]["score"] if matches["imaging"] else 0.0,
            "disease_id": matches["disease"]["id"] if matches["disease"] else "",
            "disease_name": matches["disease"]["name"] if matches["disease"] else "",
            "disease_match": matches["disease"]["score"] if matches["disease"] else 0.0,
        }
        rows.append(row)
        diagnostics.append({
            "rank": rank,
            "name": row["name"],
            "terms": terms,
            "valid_gid": bool(valid),
            "matches": candidate_matches,
        })
    return rows, diagnostics


def proposed_pair_scores(grounded: list[dict[str, Any]]) -> dict[tuple[str, tuple[str, str]], float]:
    out: dict[tuple[str, tuple[str, str]], float] = {}
    for row in grounded:
        if not row["valid_gid"]:
            continue
        score = float(row["score"])
        pairs = (
            ("gene_imaging", (row["gene_id"], row["imaging_id"])),
            ("imaging_disease", (row["imaging_id"], row["disease_id"])),
            ("gene_disease", (row["gene_id"], row["disease_id"])),
        )
        for key, pair in pairs:
            out[(key, pair)] = max(out.get((key, pair), 0.0), score)
    return out


def score_pair_rows(rows: list[dict[str, Any]], scores: dict[tuple[str, tuple[str, str]], float]) -> list[float]:
    return [scores.get((row["key"], tuple(row["pair"])), 0.0) for row in rows]


def safe_auprc(labels: list[int], scores: list[float]) -> float:
    if not labels or not any(labels):
        return 0.0
    if len({float(score) for score in scores}) <= 1:
        return sum(labels) / len(labels)
    return _auprc(labels, scores) or 0.0


def recall_at_k_external(
    future: list[dict[str, Any]],
    grounded: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    future_keys = {(row["key"], tuple(row["pair"])) for row in future}
    rows = []
    for k in TOP_KS:
        # Invalid hypotheses still consume one executed slot in the top-k prefix.
        # They simply add no proposed pair and therefore contribute zero hits.
        proposed_keys = set()
        for row in grounded[:k]:
            if not row["valid_gid"]:
                continue
            for key, pair in (
                ("gene_imaging", (row["gene_id"], row["imaging_id"])),
                ("imaging_disease", (row["imaging_id"], row["disease_id"])),
                ("gene_disease", (row["gene_id"], row["disease_id"])),
            ):
                proposed_keys.add((key, pair))
        hits = len(proposed_keys & future_keys)
        rows.append({
            "k": k,
            "hits": hits,
            "recall": hits / max(1, len(future_keys)),
            "future_evaluable_novel": len(future_keys),
        })
    return rows


def evaluate_window(
    *,
    kg_path: Path,
    claims_path: Path,
    ideas_path: Path,
    output_dir: Path,
    freeze_year: int,
    future_start: int,
    future_end: int,
    min_match: float,
    seed: int,
) -> dict[str, Any]:
    concepts, edges, _names = load_kg_index(kg_path)
    from case3_hindcasting_eval import build_pair_evidence

    _gi, _imd, _gd, historical_direct_pairs = build_pair_evidence(
        concepts, edges, strict_candidate_anchors=True
    )
    future, future_stats = load_future_claim_pairs(
        claims_path=claims_path,
        concepts=concepts,
        start_year=future_start,
        end_year=future_end,
        historical_direct_pairs=historical_direct_pairs,
    )
    ideas = load_external_ideas(ideas_path)
    grounded, diagnostics = ground_ideas(ideas, concepts, min_match)
    negatives = make_negatives(future, concepts, historical_direct_pairs, seed + freeze_year)
    paired = future + negatives
    labels = [1] * len(future) + [0] * len(negatives)
    scores = score_pair_rows(paired, proposed_pair_scores(grounded))
    auc = _auc(labels, scores) or 0.0
    auprc = safe_auprc(labels, scores)
    topk = recall_at_k_external(future, grounded)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "grounded_hypotheses.csv", grounded)
    write_csv(output_dir / "topk_external_hits.csv", topk)
    (output_dir / "grounding_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "ideas_path": str(ideas_path),
        "kg_path": str(kg_path),
        "claims_path": str(claims_path),
        "freeze_year": freeze_year,
        "future_window": f"{future_start}-{future_end}",
        "n_ideas": len(ideas),
        "valid_gid": sum(1 for row in grounded if row["valid_gid"]),
        "auc": auc,
        "auprc": auprc,
        "topk": topk,
        "future_stats": future_stats,
        "note": (
            "External free-text hypothesis evaluation. Invalid non-GID hypotheses "
            "consume rank positions and count as unsupported in top-k evaluation."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate external GENE-IMAGING-DISEASE hypotheses on Case Study 3.")
    parser.add_argument("--ideas", type=Path, required=True)
    parser.add_argument("--kg", type=Path, default=Path("neurooracle/data/experiments/case3/snapshots_full_v2_5year_2016_2020/kg_2020/knowledge_graph.json"))
    parser.add_argument("--future-claims", type=Path, default=Path("neurooracle/data/full_v2/extracted_claims.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/ai_scientist_v2"))
    parser.add_argument("--freeze-year", type=int, default=2020)
    parser.add_argument("--future-start", type=int, default=2021)
    parser.add_argument("--future-end", type=int, default=2025)
    parser.add_argument("--min-match", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=260619)
    args = parser.parse_args()
    result = evaluate_window(
        kg_path=args.kg,
        claims_path=args.future_claims,
        ideas_path=args.ideas,
        output_dir=args.output_dir,
        freeze_year=args.freeze_year,
        future_start=args.future_start,
        future_end=args.future_end,
        min_match=args.min_match,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
