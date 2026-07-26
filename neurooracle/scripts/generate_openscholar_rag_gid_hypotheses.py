from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from generate_ai_scientist_gid_hypotheses import (
    failure_placeholder,
    load_env_keys,
    normalize_hypothesis,
)
from generate_coscientist_style_gid_hypotheses import call_with_system, rows_from_payload
from generate_data_to_paper_style_gid_hypotheses import TASK_CARD, compact_json


SYSTEM_PROMPT = """You are reproducing the open-source retrieval-augmented
literature-synthesis component of OpenScholar, based on the Nature paper
"Synthesizing scientific literature with retrieval-augmented language models".

This is a fair frozen-corpus benchmark adaptation. Use only the retrieved
pre-freeze literature snippets supplied in the user prompt plus general
biomedical language understanding. Do not use post-freeze literature, future
labels, NeuroOracle, NeuroDiscovery, curated KG traversal, or task-specific
scores.

Generate citation-backed ranked hypotheses for:

GENE_TARGET -> IMAGING_MARKER -> DISEASE

JSON only."""


def claim_year(claim: dict[str, Any]) -> int:
    source = claim.get("source_paper") or {}
    for value in (source.get("year"), claim.get("year")):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 9999


def card_from_claim(claim: dict[str, Any]) -> dict[str, Any]:
    source = claim.get("source_paper") or {}
    return {
        "claim_id": str(claim.get("id") or ""),
        "year": claim_year(claim),
        "pmid": str(source.get("pmid") or claim.get("pmid") or ""),
        "doi": str(source.get("doi") or ""),
        "journal": str(source.get("journal") or ""),
        "title": str(source.get("title") or ""),
        "subject": str(claim.get("subject_name") or ""),
        "predicate": str(claim.get("predicate") or ""),
        "object": str(claim.get("object_name") or ""),
        "disease": str(claim.get("disease") or ""),
        "passage": " ".join(str(claim.get("raw_text") or "").split())[:700],
    }


def load_literature_cards(path: Path, freeze_year: int, max_cards: int) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            claim = json.loads(line)
            if claim.get("negated"):
                continue
            if claim_year(claim) > freeze_year:
                continue
            card = card_from_claim(claim)
            if not card["passage"]:
                continue
            if not (card["subject"] or card["object"] or card["disease"]):
                continue
            cards.append(card)
            if len(cards) >= max_cards:
                break
    return cards


def sample_cards(cards: list[dict[str, Any]], n_cards: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(cards) <= n_cards:
        return cards

    # Mix high-information records with random coverage so independent
    # replicates do not retrieve the same narrow literature slice.
    scored = []
    for card in cards:
        score = 0
        if card.get("pmid"):
            score += 2
        if card.get("doi"):
            score += 2
        if card.get("disease"):
            score += 1
        if len(str(card.get("passage") or "")) > 120:
            score += 1
        scored.append((score, rng.random(), card))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    top_n = max(1, n_cards // 3)
    selected = [card for _score, _rand, card in scored[:top_n]]
    remaining = [card for _score, _rand, card in scored[top_n:]]
    selected.extend(rng.sample(remaining, k=min(n_cards - len(selected), len(remaining))))
    rng.shuffle(selected)
    return selected


def prompt_for_batch(
    batch_n: int,
    start_rank: int,
    previous_names: list[str],
    cards: list[dict[str, Any]],
    freeze_year: int,
    seed: int,
) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    return f"""{TASK_CARD}

OpenScholar-style retrieval setting:
- The retrieved passages below are from papers published on or before {freeze_year}.
- Treat them as the only explicit literature corpus available to this baseline.
- Use citations to support synthesis, but do not simply copy a retrieved claim.
- Do not infer from any post-freeze literature.

Replicate seed: {seed}. Use it only to diversify independent benchmark
repetitions.

Already generated hypothesis names:
{previous}

Retrieved pre-freeze passages:
{compact_json({"retrieved_passages": cards})}

Generate exactly {batch_n} new citation-backed hypotheses for ranks {start_rank}
to {start_rank + batch_n - 1}.

Return one JSON object:
{{
  "retrieval_summary": "one concise paragraph",
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence grounded in retrieved literature",
      "expected_direction": "optional concise direction",
      "supporting_pmids": ["PMID strings from retrieved passages, if available"],
      "retrieved_passages_used": ["claim_id strings"],
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Rank strongest and best-supported hypotheses first.
- Use concrete measured imaging markers, never modality-only markers.
- Prefer hypotheses that combine at least two retrieved passages when possible.
- Confidence should be a number from 0 to 1.
- JSON only."""


def normalize_openscholar_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    hyp = normalize_hypothesis(row, rank)
    hyp["supporting_pmids"] = ";".join(str(x) for x in row.get("supporting_pmids", []) if str(x).strip())
    hyp["retrieved_passages_used"] = ";".join(
        str(x) for x in row.get("retrieved_passages_used", []) if str(x).strip()
    )
    hyp["baseline"] = "openscholar_rag"
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
            retrieved = sample_cards(cards, args.passages_per_batch, args.seed + start_rank + attempt)
            payload = call_with_system(
                model=args.model,
                system=SYSTEM_PROMPT,
                prompt=prompt_for_batch(batch_n, start_rank, previous_names, retrieved, args.freeze_year, args.seed),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rows = rows_from_payload(payload)
            if rows:
                return rows[:batch_n], {"retrieved_passages": retrieved, "retrieval_summary": payload.get("retrieval_summary", "")}, ""
            raise ValueError("No hypotheses returned by OpenScholar-RAG generator.")
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
    return [], {}, last_error


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_env_keys()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set and no usable key was found in .env.keys")

    cards = load_literature_cards(args.claims, args.freeze_year, args.max_corpus_cards)
    if not cards:
        raise RuntimeError(f"No pre-freeze literature cards found in {args.claims}")

    hypotheses: list[dict[str, Any]] = []
    names: list[str] = []
    retrieval_logs: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            hypotheses = [row for row in existing if isinstance(row, dict)]
            names = [str(row.get("Name") or row.get("name") or "") for row in hypotheses]
            print(f"resuming from {len(hypotheses)}/{args.n} hypotheses -> {args.output}")

    while len(hypotheses) < args.n:
        start_rank = len(hypotheses) + 1
        batch_n = min(args.batch_size, args.n - len(hypotheses))
        rows, log, error = generate_batch(args, batch_n, start_rank, names, cards)
        if len(rows) < batch_n:
            raise RuntimeError(error or "model returned too few hypotheses")
        for row in rows[:batch_n]:
            rank = len(hypotheses) + 1
            if row.get("generation_failure"):
                hyp = failure_placeholder(rank, str(row["generation_failure"]))
                hyp["baseline"] = "openscholar_rag"
            else:
                hyp = normalize_openscholar_row(row, rank)
            hypotheses.append(hyp)
            names.append(str(hyp["Name"]))
        retrieval_logs.append({"start_rank": start_rank, "batch_n": batch_n, **log})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output.parent / "retrieval_logs.json").write_text(
            json.dumps(retrieval_logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"generated {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-budget OpenScholar-RAG GID hypotheses.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default=os.environ.get("OPENSCHOLAR_RAG_MODEL", "gpt-5.4"))
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--freeze-year", type=int, default=2020)
    parser.add_argument("--passages-per-batch", type=int, default=24)
    parser.add_argument("--max-corpus-cards", type=int, default=25000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("neurooracle/data/full_v2/extracted_claims.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/openscholar_rag_fixed_1000/hypotheses.json"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
