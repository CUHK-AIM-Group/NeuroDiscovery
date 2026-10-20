"""Prompt assembly and skill routing shared by every driver.

The baseline imports the published runner's ``NO_SKILLS_CONTRACT`` unchanged;
the complete prompt has new version metadata and is frozen per campaign. The
harness arm permits two skill tools. Neither arm injects skill summaries or
uses the old preselection router. Models discover methodology through tools.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
from neurobench.runner import codex500




PROMPT_VERSION = "neurobench-harness-10000-v2"


HARNESS_CONTRACT = """You are generating one isolated NeuroBench benchmark response in NeuroRuntime harness mode.

Non-negotiable experiment controls:
- Do not execute the benchmark workflow. Do not access the network, browser, APIs, datasets, containers, package managers, training jobs, or task output paths.
- You may call the NeuroRuntime skill tools (`search_skills` and `read_skill`) to discover and load reusable methodology, then adapt it to this task. Those are the only permitted tools; do not run shell commands.
- Skills are reusable guidance, not a rigid pipeline. Adapt their methodology, validation logic, parameters and file conventions to the task; do not copy a generic example when the task needs different inputs or outputs.
- Missing real inputs do not stop this benchmark response: state the missing-input assumption briefly, then still provide complete proposed commands or code. The proposed code may include validation and failure handling but must not be run now.
- Return a final answer with exactly two top-level Markdown sections, in this order: `## Solution Thinking` and `## Commands Or Code`. Include no preamble and no other level-2 headings.
- Keep the solution deterministic, task-directed, and sufficiently complete for a scoring agent to judge correctness.
"""


def baseline_contract() -> str:
    """The published no-skills contract, imported unchanged."""
    return codex500.NO_SKILLS_CONTRACT


def discover_tasks(benchmark_root: Path | None = None):
    module = codex500
    return module.discover_tasks(benchmark_root or (REPO_ROOT / "neurobench" / "tasks"))


def load_skills(skills_root: Path | None = None):
    module = codex500
    return module.load_local_skills(skills_root or (REPO_ROOT / "skills"))


def route_skills(task, skills, *, limit: int = 12):
    module = codex500
    return module.select_candidate_skills(task, skills, limit=limit)


def validate_answer_contract(text: str) -> str:
    return codex500.validate_answer_contract(text)


def normalize_answer_for_contract(text: str) -> tuple[str, str]:
    """Remove only a leading prose preamble before the required answer.

    Some CLI/model combinations return a short acknowledgement before the two
    required sections even though the answer body itself is complete.  The
    benchmark must score the contract-shaped answer while retaining the exact
    provider text as a separate immutable artifact.  This function therefore
    performs one narrow, structural repair: it drops bytes before the first
    ``## Solution Thinking`` heading when that prefix contains no level-2
    heading of its own.  It never edits section content, headings, or code.
    """
    if not isinstance(text, str):
        return text, "none"
    match = re.search(r"^##\s+Solution\s+Thinking\s*$", text,
                      flags=re.IGNORECASE | re.MULTILINE)
    if not match or not text[:match.start()].strip():
        return text, "none"
    prefix = text[:match.start()]
    # A competing H2 means this is not a harmless preamble; leave it for the
    # strict validator to reject rather than silently changing the answer.
    if re.search(r"^##\s+\S", prefix, flags=re.MULTILINE):
        return text, "none"
    normalized = text[match.start():].lstrip()
    return normalized, "leading_preamble_stripped"


def split_answer(text: str):
    return codex500.split_answer(text)


def render_prompt(task, *, harness_enabled: bool) -> str:
    """Assemble the prompt for one arm.

    Neither arm preloads skill files. The harness arm discovers skills through the
    tools; the baseline arm receives no skill material at all, exactly like the
    published no-skills condition. That is what makes this a capability
    comparison rather than a prompt-text comparison.
    """
    prefix = HARNESS_CONTRACT if harness_enabled else baseline_contract()
    return (
        f"{prefix}\n\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Case ID: {task.case_id}\n"
        f"Source task SHA-256: {task.source_sha256}\n\n"
        "--- BEGIN BENCHMARK TASK ---\n"
        f"{task.text.rstrip()}\n"
        "--- END BENCHMARK TASK ---\n"
    )


def report_metadata_header(
    task,
    *,
    report_model: str,
    actual_model: str,
    harness_enabled: bool,
    driver: str,
    session_id: str,
    prompt_sha256: str,
    response_sha256: str,
    elapsed_seconds: float,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    skill_names: list[str],
    tool_lines: list[str],
    reasoning_effort: str | None = None,
    timestamp: str | None = None,
) -> str:
    """Render a report in the shape the shipped scorer parses.

    The scorer reads the model name from the filename, not the header, but it
    reads ``Skills used (unique)`` for the usage-efficiency column and the level-2
    heading pair for the answer contract. Both are emitted here so the report is
    parseable without any scorer change.
    """
    import datetime as _dt
    import json as _json

    unique_skills = sorted(set(skill_names), key=str.lower)
    total_tokens = input_tokens + output_tokens
    return "\n".join(
        [
            f"# Benchmark Report: {task.case_id}_{report_model}",
            "",
            f"- Timestamp: {timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}",
            "- Mode: benchmark",
            f"- Benchmark variant: {'harness' if harness_enabled else 'baseline'}",
            f"- Case ID: {task.case_id}",
            f"- Model: {report_model}",
            f"- Actual model: {actual_model}",
            f"- Reasoning effort: {reasoning_effort or 'default'}",
            f"- Driver: {driver}",
            f"- Session isolation: {session_id}",
            f"- Prompt version: {PROMPT_VERSION}",
            f"- Source task SHA-256: {task.source_sha256}",
            f"- Prompt SHA-256: {prompt_sha256}",
            f"- Response SHA-256: {response_sha256}",
            f"- Task: {task.title}",
            f"- Elapsed seconds: {elapsed_seconds:.3f}",
            f"- Token usage: prompt={input_tokens}, completion={output_tokens}, total={total_tokens}",
            f"- Cached input tokens: {cached_input_tokens}",
            f"- Skills used (unique): {len(unique_skills)}",
            f"- Skill names used: {_json.dumps(unique_skills, ensure_ascii=False)}",
        ]
    )
