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
from generate_data_to_paper_style_gid_hypotheses import TASK_CARD, compact_json


PI_SYSTEM = """You are reproducing only the open-source agent-workflow component
of Virtual Lab, based on the Nature paper "The Virtual Lab of AI agents designs
new SARS-CoV-2 nanobodies".

This is a fair text-only benchmark adaptation. Do not use wet-lab validation,
nanobody-specific tools, AlphaFold, Rosetta, ESM, docking, future labels,
NeuroOracle, NeuroDiscovery, curated KG traversal, or any task-specific scoring.

Act as the principal investigator agent: define the meeting agenda, assign
scientist-agent perspectives, and specify how the team should converge on
ranked GENE_TARGET -> IMAGING_MARKER -> DISEASE hypotheses. JSON only."""

SCIENTIST_SYSTEM = """You are a scientist agent in a Virtual Lab-style team.
Use your assigned perspective to propose concrete neuroscience hypotheses for
the fixed-budget hindcasting benchmark.

Do not use wet-lab validation, structural-biology tools, future labels,
NeuroOracle, NeuroDiscovery, curated KG traversal, or task-specific scores.
Every proposal must attempt GENE_TARGET -> IMAGING_MARKER -> DISEASE. JSON only."""

CRITIC_SYSTEM = """You are the PI/ranker in a Virtual Lab-style meeting.
Integrate proposals from several scientist agents, remove vague modality-only
markers, and return a final ranked fixed-budget hypothesis table.

Do not use wet-lab validation, nanobody-specific tools, future labels,
NeuroOracle, NeuroDiscovery, curated KG traversal, or task-specific scores.
JSON only."""

COMPACT_SYSTEM = """You are reproducing a compact, open-source workflow
adaptation of Virtual Lab, based on the Nature paper "The Virtual Lab of AI
agents designs new SARS-CoV-2 nanobodies".

For this benchmark, simulate the full virtual team meeting inside one response:
1. PI agenda setting,
2. molecular neuroscience scientist proposals,
3. clinical neuroimaging scientist proposals,
4. translational psychiatry scientist proposals,
5. critic/ranker synthesis into a fixed-budget hypothesis table.

This is a fair text-only adaptation. Do not use wet-lab validation,
nanobody-specific tools, AlphaFold, Rosetta, ESM, docking, future labels,
NeuroOracle, NeuroDiscovery, curated KG traversal, or task-specific scores.
JSON only."""


PERSPECTIVES = [
    (
        "molecular_neuroscience",
        "Prioritize gene/pathway mechanisms that plausibly alter neuroimaging phenotypes.",
    ),
    (
        "clinical_neuroimaging",
        "Prioritize concrete measured imaging markers and disease-specific clinical phenotypes.",
    ),
    (
        "translational_psychiatry",
        "Prioritize psychiatric and neurodegenerative diseases where future literature could test the link.",
    ),
]


def pi_prompt(batch_n: int, start_rank: int, previous_names: list[str], seed: int) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    return f"""{TASK_CARD}

Replicate seed: {seed}. Use it only to diversify independent benchmark
repetitions.

Already generated hypothesis names:
{previous}

Plan a Virtual Lab-style team meeting that will generate exactly {batch_n}
hypotheses for ranks {start_rank} to {start_rank + batch_n - 1}.

Return one JSON object:
{{
  "meeting_agenda": ["short agenda items"],
  "agent_assignments": [
    {{"agent": "molecular_neuroscience", "task": "short task"}},
    {{"agent": "clinical_neuroimaging", "task": "short task"}},
    {{"agent": "translational_psychiatry", "task": "short task"}}
  ],
  "ranking_criteria": ["short criteria"]
}}

JSON only."""


def scientist_prompt(
    *,
    batch_n: int,
    start_rank: int,
    previous_names: list[str],
    meeting_plan: dict[str, Any],
    perspective: str,
    guidance: str,
) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    proposal_n = max(batch_n, 8)
    return f"""{TASK_CARD}

Virtual Lab meeting plan:
{compact_json(meeting_plan)}

Scientist-agent perspective: {perspective}
Guidance: {guidance}

Already generated hypothesis names:
{previous}

Generate {proposal_n} candidate hypotheses that could contribute to final ranks
{start_rank} to {start_rank + batch_n - 1}.

Return one JSON object:
{{
  "agent": "{perspective}",
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "agent_rationale": "why this agent proposed it",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Use concrete measured imaging markers, never modality-only markers.
- Use concrete disease names.
- Confidence should be a number from 0 to 1.
- JSON only."""


def ranker_prompt(batch_n: int, meeting_plan: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    return f"""Integrate this Virtual Lab-style meeting into exactly {batch_n}
final ranked GENE-IMAGING-DISEASE hypotheses.

Meeting plan:
{compact_json(meeting_plan)}

Scientist-agent proposals:
{compact_json({"proposals": proposals})}

Return one JSON object:
{{
  "meeting_summary": "one concise paragraph",
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "team_consensus": "short note on why the team selected it",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Rank strongest hypotheses first.
- Fix vague modality-only markers when possible.
- Do not add wet-lab, structural-biology, docking, or nanobody-specific claims.
- JSON only."""


def compact_meeting_prompt(batch_n: int, start_rank: int, previous_names: list[str], seed: int) -> str:
    previous = "\n".join(f"- {name}" for name in previous_names[-120:]) or "(none)"
    return f"""{TASK_CARD}

Replicate seed: {seed}. Use it only to diversify independent benchmark
repetitions.

Already generated hypothesis names:
{previous}

Generate exactly {batch_n} final ranked hypotheses for ranks {start_rank} to
{start_rank + batch_n - 1} by internally simulating a Virtual Lab team meeting:
- PI: defines the research agenda and ranking criteria.
- Molecular neuroscience scientist: proposes gene/pathway mechanisms.
- Clinical neuroimaging scientist: enforces concrete measured imaging markers.
- Translational psychiatry scientist: checks disease specificity and future-testability.
- Critic/ranker: removes vague modality-only markers and ranks the final list.

Return one JSON object:
{{
  "meeting_summary": "one concise paragraph describing the simulated team consensus",
  "hypotheses": [
    {{
      "name": "short_snake_case_name",
      "title": "short human-readable title",
      "gene_target": "specific gene or pathway",
      "imaging_marker": "specific measured neuroimaging marker",
      "disease": "specific disease",
      "mechanistic_rationale": "one concise sentence",
      "expected_direction": "optional concise direction",
      "team_consensus": "short note on the PI/scientist-agent rationale",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- The hypotheses list must contain exactly {batch_n} objects.
- Rank strongest hypotheses first.
- Use concrete measured imaging markers, never modality-only markers.
- Use concrete disease names.
- Confidence should be a number from 0 to 1.
- JSON only."""


def generate_batch(
    args: argparse.Namespace,
    batch_n: int,
    start_rank: int,
    previous_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if args.compact_meeting:
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                payload = call_with_system(
                    model=args.model,
                    system=COMPACT_SYSTEM,
                    prompt=compact_meeting_prompt(batch_n, start_rank, previous_names, args.seed),
                    temperature=args.scientist_temperature,
                    max_tokens=args.max_tokens,
                )
                rows = rows_from_payload(payload)
                if rows:
                    return rows[:batch_n], {"meeting_summary": payload.get("meeting_summary", "")}, ""
                raise ValueError("No hypotheses returned by compact Virtual Lab-style meeting.")
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)
        return [], {}, last_error

    last_error = ""
    for attempt in range(1, args.retries + 1):
        try:
            meeting_plan = call_with_system(
                model=args.model,
                system=PI_SYSTEM,
                prompt=pi_prompt(batch_n, start_rank, previous_names, args.seed),
                temperature=args.pi_temperature,
                max_tokens=args.max_tokens,
            )
            proposals: list[dict[str, Any]] = []
            for perspective, guidance in PERSPECTIVES:
                payload = call_with_system(
                    model=args.model,
                    system=SCIENTIST_SYSTEM,
                    prompt=scientist_prompt(
                        batch_n=batch_n,
                        start_rank=start_rank,
                        previous_names=previous_names,
                        meeting_plan=meeting_plan,
                        perspective=perspective,
                        guidance=guidance,
                    ),
                    temperature=args.scientist_temperature,
                    max_tokens=args.max_tokens,
                )
                proposals.append({"agent": perspective, "hypotheses": rows_from_payload(payload)})
            final_payload = call_with_system(
                model=args.model,
                system=CRITIC_SYSTEM,
                prompt=ranker_prompt(batch_n, meeting_plan, proposals),
                temperature=args.ranker_temperature,
                max_tokens=args.max_tokens,
            )
            rows = rows_from_payload(final_payload)
            if rows:
                log = {
                    "meeting_plan": meeting_plan,
                    "proposals": proposals,
                    "meeting_summary": final_payload.get("meeting_summary", ""),
                }
                return rows[:batch_n], log, ""
            raise ValueError("No hypotheses returned by Virtual Lab-style ranker.")
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
    return [], {}, last_error


def normalize_virtual_lab_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    hyp = normalize_hypothesis(row, rank)
    hyp["team_consensus"] = str(row.get("team_consensus") or row.get("agent_rationale") or "")
    hyp["baseline"] = "virtual_lab_style"
    return hyp


def generate(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_env_keys()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set and no usable key was found in .env.keys")

    hypotheses: list[dict[str, Any]] = []
    names: list[str] = []
    meeting_logs: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            hypotheses = [row for row in existing if isinstance(row, dict)]
            names = [str(row.get("Name") or row.get("name") or "") for row in hypotheses]
            print(f"resuming from {len(hypotheses)}/{args.n} hypotheses -> {args.output}")

    while len(hypotheses) < args.n:
        start_rank = len(hypotheses) + 1
        batch_n = min(args.batch_size, args.n - len(hypotheses))
        rows, log, error = generate_batch(args, batch_n, start_rank, names)
        if len(rows) < batch_n:
            raise RuntimeError(error or "model returned too few hypotheses")
        for row in rows[:batch_n]:
            rank = len(hypotheses) + 1
            if row.get("generation_failure"):
                hyp = failure_placeholder(rank, str(row["generation_failure"]))
                hyp["baseline"] = "virtual_lab_style"
            else:
                hyp = normalize_virtual_lab_row(row, rank)
            hypotheses.append(hyp)
            names.append(str(hyp["Name"]))
        meeting_logs.append({"start_rank": start_rank, "batch_n": batch_n, **log})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8")
        (args.output.parent / "virtual_lab_meeting_logs.json").write_text(
            json.dumps(meeting_logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"generated {len(hypotheses)}/{args.n} hypotheses -> {args.output}")
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-budget Virtual Lab-style GID hypotheses.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default=os.environ.get("VIRTUAL_LAB_STYLE_MODEL", "gpt-5.4"))
    parser.add_argument("--pi-temperature", type=float, default=0.35)
    parser.add_argument("--scientist-temperature", type=float, default=0.8)
    parser.add_argument("--ranker-temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--compact-meeting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurooracle/data/experiments/case3/external_hypothesis_generators/virtual_lab_style_fixed_1000/hypotheses.json"),
    )
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
