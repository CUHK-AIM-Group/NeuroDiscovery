from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from generate_ai_scientist_gid_hypotheses import (
    failure_placeholder,
    load_env_keys,
    normalize_hypothesis,
)
from generate_coscientist_style_gid_hypotheses import call_with_system, rows_from_payload


SYSTEM_PROMPT = """You are reproducing the hypothesis-generation component of a
paper-grounded data-to-paper adaptation, based on the NEJM AI paper
"Autonomous LLM-Driven Research - from Data to Human-Verifiable Research
Papers".

Your workflow should imitate a system that converts a documented research task
into a human-verifiable scientific manuscript plan:

1. understand the research question and data schema,
2. define analyzable variables,
3. propose statistical/biological claims,
4. turn the strongest claims into concise testable hypotheses.

Do not use NeuroOracle, NeuroDiscovery, curated KG traversal, graph degree,
future labels, or any task-specific NeuroClaw scoring. Use only general
biomedical/neuroscience knowledge plus the task description supplied by the
user. JSON only."""


TASK_CARD = """Research task card
==================

Benchmark:
- Predict future-supported neuroscience hypotheses from a frozen knowledge
  state.
- Each candidate consumes one experiment/hypothesis rank even if it is invalid.

Required hypothesis schema:
- GENE_TARGET -> IMAGING_MARKER -> DISEASE

Valid GENE_TARGET examples:
- APOE, MAPT, SNCA, GBA, LRRK2, COMT, BDNF, NLRP3, DRD2, HTR2A
- A pathway is allowed if it is specific, such as neuroinflammation/NLRP3,
  dopamine signaling, amyloid processing, tau pathology, synaptic plasticity.
- Prefer specific gene symbols over broad pathway names because downstream
  evaluation grounds hypotheses to gene/pathway concept nodes.

Valid IMAGING_MARKER examples:
- Use one of these canonical measured variables whenever possible:
  cortical thickness, cortical surface area, regional volume,
  gray matter density, fractional anisotropy, mean diffusivity,
  radial diffusivity, axial diffusivity, functional connectivity,
  amplitude of low-frequency fluctuation, regional homogeneity,
  task BOLD amplitude, amyloid SUVR, tau SUVR, FDG uptake.
- Region-specific markers are allowed when written as
  "<canonical variable> in <brain region>", for example:
  regional volume in Hippocampus, cortical thickness in Entorhinal Cortex,
  functional connectivity between Default Mode Network and Anterior Cingulate
  Cortex, amyloid SUVR in Default Mode Network, tau SUVR in Entorhinal Cortex,
  FDG uptake in Hippocampus.

Invalid IMAGING_MARKER examples:
- MRI, fMRI, PET, SPECT, CT, neuroimaging, brain scan
- a bare anatomical region without a measured quantity, such as "hippocampus"
- informal marker names that do not contain a canonical variable, such as
  "dopamine transporter binding", "D2 receptor availability", or
  "hippocampal volume"; write "regional volume in Hippocampus" instead.

Valid DISEASE examples:
- Alzheimer's disease, Parkinson's disease, schizophrenia, major depressive
  disorder, bipolar disorder, ADHD, autism spectrum disorder, obsessive
  compulsive disorder, PTSD, anxiety disorder, substance use disorder.

The output is not a paper. It is the ranked hypothesis table that a
data-to-paper system would pass to downstream experiments."""


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def prompt_for_batch(batch_n: int, start_rank: int, previous_names: list[str], seed: int) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    return f"""{TASK_CARD}

Replicate seed: {seed}. Use it only to diversify hypothesis selection and
ordering across independent benchmark repetitions.

Already generated hypothesis names:
{previous}

Generate exactly {batch_n} new ranked hypotheses for ranks {start_rank} to {start_rank + batch_n - 1}.

First internally perform the data-to-paper steps:
1. write a compact research objective,
2. define the data variables implied by GENE_TARGET, IMAGING_MARKER, DISEASE,
3. identify plausible analyzable associations,
4. rank the best hypotheses by expected future support and testability.

Return one JSON object with this exact shape:
{{
  "research_plan": {{
    "objective": "one sentence",
    "data_schema": ["short variable descriptions"],
    "analysis_plan": ["short analysis steps"]
  }},
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "analysis_test": "short statistical test or validation design",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Rank strongest hypotheses first.
- Do not duplicate names listed above.
- Use concrete measured imaging markers, never modality-only markers.
- Confidence should be a number from 0 to 1.
- JSON only; no markdown."""


def normalize_data_to_paper_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    hyp = normalize_hypothesis(row, rank)
    hyp["analysis_test"] = str(row.get("analysis_test") or "")
    hyp["baseline"] = "data_to_paper"
    return hyp


def generate_batch(
    args: argparse.Namespace,
    batch_n: int,
    start_rank: int,
    previous_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            payload = call_with_system(
                model=args.model,
                system=SYSTEM_PROMPT,
                prompt=prompt_for_batch(batch_n, start_rank, previous_names, args.seed),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            rows = rows_from_payload(payload)
            if rows:
                return rows[:batch_n], payload.get("research_plan") or {}, ""
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

    hypotheses: list[dict[str, Any]] = []
    names: list[str] = []
    research_plans: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            hypotheses = [row for row in existing if isinstance(row, dict)]
            names = [str(row.get("Name") or row.get("name") or "") for row in hypotheses]
            print(f"resuming from {len(hypotheses)}/{args.n} hypotheses -> {args.output}")

    while len(hypotheses) < args.n:
        start_rank = len(hypotheses) + 1
        batch_n = min(args.batch_size, args.n - len(hypotheses))
        rows, plan, error = generate_batch(args, batch_n, start_rank, names)
        if plan:
            research_plans.append({"start_rank": start_rank, "batch_n": batch_n, **plan})
        if len(rows) < batch_n:
            rows.extend({"generation_failure": error or "model returned too few hypotheses"} for _ in range(batch_n - len(rows)))
        for row in rows[:batch_n]:
            rank = len(hypotheses) + 1
            if row.get("generation_failure"):
                hyp = failure_placeholder(rank, str(row["generation_failure"]))
                hyp["baseline"] = "data_to_paper"
            else:
                hyp = normalize_data_to_paper_row(row, rank)
            hypotheses.append(hyp)
            names.append(str(hyp["Name"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output.parent / "research_plans.json").write_text(
            json.dumps(research_plans, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"generated {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-budget data-to-paper GID hypotheses.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("DATA_TO_PAPER_STYLE_MODEL", "gpt-5.4"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/data_to_paper_fixed_1000/hypotheses.json"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
