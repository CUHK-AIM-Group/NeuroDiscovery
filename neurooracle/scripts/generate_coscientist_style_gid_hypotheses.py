from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from generate_ai_scientist_gid_hypotheses import (
    call_model,
    failure_placeholder,
    http_chat_completion,
    load_env_keys,
    normalize_hypothesis,
    parse_json_object,
)


GENERATOR_SYSTEM = """You are the hypothesis-generation agent in a
paper-grounded adaptation of Co-Scientist, the multi-agent scientific discovery
system described in the published Nature paper "Accelerating scientific
discovery with Co-Scientist".

Do not use NeuroOracle, NeuroDiscovery, curated KG traversal, or any
domain-specific graph scoring. Use only general biomedical and neuroscience
knowledge.

Generate ranked hypotheses for a neuroscience hindcasting benchmark. Each
hypothesis should attempt:

GENE_TARGET -> IMAGING_MARKER -> DISEASE

The imaging marker must be a concrete measured neuroimaging marker, not a
modality name. JSON only."""


CRITIC_SYSTEM = """You are the critic agent in a paper-grounded Co-Scientist
adaptation. Score each proposed hypothesis for:

1. valid GENE_TARGET -> IMAGING_MARKER -> DISEASE structure,
2. concrete imaging-marker specificity,
3. disease specificity,
4. mechanistic plausibility,
5. likelihood that future literature could support the proposed relation.

You must not use NeuroOracle, NeuroDiscovery, curated KG traversal, or
future-paper labels. JSON only."""


REFINER_SYSTEM = """You are the refiner/ranker agent in a paper-grounded
Co-Scientist adaptation.

Given candidate hypotheses and critic feedback, produce the final fixed-budget
ranked hypothesis list. Keep exactly the requested number of hypotheses. Do not
drop hard cases: if uncertain, provide the best structured attempt. Invalid or
vague outputs will be counted as failed hypotheses by the evaluator.

Do not use NeuroOracle, NeuroDiscovery, curated KG traversal, or future-paper
labels. JSON only."""


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def generation_prompt(batch_n: int, start_rank: int, previous_names: list[str], seed: int) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    return f"""Generate exactly {batch_n} new hypotheses for ranks {start_rank} to {start_rank + batch_n - 1}.

Replicate seed: {seed}. Use it only to diversify hypothesis selection and
ordering across independent benchmark repetitions.

Avoid duplicating these already generated names:
{previous}

Return one JSON object:
{{
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Use concrete disease names, not broad phrases.
- Use concrete imaging measurements. Invalid: MRI, fMRI, PET, SPECT, CT,
  neuroimaging, brain scan, or a bare anatomical region without a measured
  feature.
- Confidence should be 0 to 1.
- JSON only."""


def critic_prompt(hypotheses: list[dict[str, Any]]) -> str:
    return f"""Critique and score these candidate GENE-IMAGING-DISEASE hypotheses.

Candidates:
{compact_json({"hypotheses": hypotheses})}

Return one JSON object:
{{
  "critiques": [
    {{
      "name": "candidate name",
      "valid_gid": true,
      "marker_specificity": 0.0,
      "mechanistic_plausibility": 0.0,
      "future_support_plausibility": 0.0,
      "overall_score": 0.0,
      "issues": ["short issue strings"],
      "suggested_fix": "short fix if needed"
    }}
  ]
}}

JSON only."""


def refiner_prompt(batch_n: int, generated: list[dict[str, Any]], critiques: list[dict[str, Any]]) -> str:
    return f"""Refine and rank the candidate hypotheses using the critic feedback.

Generated candidates:
{compact_json({"hypotheses": generated})}

Critic feedback:
{compact_json({"critiques": critiques})}

Return exactly {batch_n} final hypotheses as one JSON object:
{{
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Keep exactly {batch_n}; no markdown.
- Rank the strongest hypotheses first.
- Fix vague modality-only imaging markers when possible.
- If a hypothesis cannot be made valid, still keep a best attempt; the evaluator
  will count invalid items as failed hypotheses.
"""


def call_with_system(*, model: str, system: str, prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    wire_api = os.environ.get("OPENAI_WIRE_API", "").strip().lower()
    if wire_api in {"chat_completions", "http_chat_completions"}:
        return parse_json_object(
            http_chat_completion(
                model=model,
                system=system,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": os.environ.get("OPENAI_API_KEY")}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_TIMEOUT"):
        kwargs["timeout"] = float(os.environ["OPENAI_TIMEOUT"])
    client = OpenAI(**kwargs)
    if wire_api == "responses":
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_tokens,
            text={"format": {"type": "json_object"}},
        )
        return parse_json_object(response.output_text or "{}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return parse_json_object(response.choices[0].message.content or "{}")


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("hypotheses")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def critiques_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("critiques")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def generate_batch(args: argparse.Namespace, batch_n: int, start_rank: int, previous_names: list[str]) -> tuple[list[dict[str, Any]], str]:
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            generated = rows_from_payload(
                call_with_system(
                    model=args.model,
                    system=GENERATOR_SYSTEM,
                    prompt=generation_prompt(batch_n, start_rank, previous_names, args.seed),
                    temperature=args.generator_temperature,
                    max_tokens=args.max_tokens,
                )
            )
            critiques = critiques_from_payload(
                call_with_system(
                    model=args.model,
                    system=CRITIC_SYSTEM,
                    prompt=critic_prompt(generated[:batch_n]),
                    temperature=args.critic_temperature,
                    max_tokens=args.max_tokens,
                )
            )
            refined = rows_from_payload(
                call_with_system(
                    model=args.model,
                    system=REFINER_SYSTEM,
                    prompt=refiner_prompt(batch_n, generated[:batch_n], critiques),
                    temperature=args.refiner_temperature,
                    max_tokens=args.max_tokens,
                )
            )
            rows = refined or generated
            if rows:
                return rows[:batch_n], ""
            raise ValueError("No hypotheses returned by generator/refiner.")
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
    return [], last_error


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_env_keys()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set and no usable key was found in .env.keys")

    hypotheses: list[dict[str, Any]] = []
    names: list[str] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            hypotheses = [row for row in existing if isinstance(row, dict)]
            names = [str(row.get("Name") or row.get("name") or "") for row in hypotheses]
            print(f"resuming from {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    while len(hypotheses) < args.n:
        start_rank = len(hypotheses) + 1
        batch_n = min(args.batch_size, args.n - len(hypotheses))
        rows, error = generate_batch(args, batch_n, start_rank, names)
        if len(rows) < batch_n:
            rows.extend({"generation_failure": error or "model returned too few hypotheses"} for _ in range(batch_n - len(rows)))
        for row in rows[:batch_n]:
            rank = len(hypotheses) + 1
            if row.get("generation_failure"):
                hyp = failure_placeholder(rank, str(row["generation_failure"]))
            else:
                hyp = normalize_hypothesis(row, rank)
                hyp["baseline"] = "co_scientist"
            hypotheses.append(hyp)
            names.append(str(hyp["Name"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"generated {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-budget Co-Scientist GID hypotheses.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("COSCIENTIST_STYLE_MODEL", "gpt-5.4"))
    parser.add_argument("--generator-temperature", type=float, default=0.9)
    parser.add_argument("--critic-temperature", type=float, default=0.2)
    parser.add_argument("--refiner-temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/co_scientist_fixed_1000/hypotheses.json"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
