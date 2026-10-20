#!/usr/bin/env python3
"""Run the 500-task NeuroBench comparison through fresh Codex CLI sessions.

The runner intentionally does not execute benchmark solutions.  It invokes one
ephemeral Codex session per task, validates the tool policy for the selected
condition, wraps the final answer in the repository's benchmark-report format,
and persists resumable state in SQLite.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "neurobench" / "tasks"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dev" / "outputs" / "neurobench_codex_500_20260831"
PROMPT_VERSION = "neurobench-codex-500-v4"
REPORT_SCHEMA_VERSION = "neuroclaw-benchmark-report-v1"
EXPECTED_TASK_COUNT = 500
INJECTED_SKILL_COUNT = 3


@dataclasses.dataclass(frozen=True)
class ConditionSpec:
    key: str
    model: str
    reasoning_effort: str
    variant: str
    report_model: str
    skills_enabled: bool


CONDITIONS: dict[str, ConditionSpec] = {
    "neuroclaw_gpt_5_6_sol_high": ConditionSpec(
        key="neuroclaw_gpt_5_6_sol_high",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        variant="with-skills",
        report_model="gpt-5.6-sol_withskills",
        skills_enabled=True,
    ),
    "codex_gpt_5_6_sol_high": ConditionSpec(
        key="codex_gpt_5_6_sol_high",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        variant="no-skills",
        report_model="gpt-5.6-sol_noskills",
        skills_enabled=False,
    ),
    "neuroclaw_gpt_5_5_high": ConditionSpec(
        key="neuroclaw_gpt_5_5_high",
        model="gpt-5.5",
        reasoning_effort="high",
        variant="with-skills",
        report_model="gpt-5.5_withskills",
        skills_enabled=True,
    ),
    "codex_gpt_5_5_high": ConditionSpec(
        key="codex_gpt_5_5_high",
        model="gpt-5.5",
        reasoning_effort="high",
        variant="no-skills",
        report_model="gpt-5.5_noskills",
        skills_enabled=False,
    ),
}


@dataclasses.dataclass(frozen=True)
class BenchmarkTask:
    number: int
    case_id: str
    directory_name: str
    path: Path
    title: str
    text: str
    source_sha256: str


@dataclasses.dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    summary: str
    skill_md: Path


@dataclasses.dataclass
class AttemptResult:
    ok: bool
    error: str
    retry_class: str
    session_id: str
    response: str
    response_sha256: str
    report_path: str
    report_sha256: str
    elapsed_seconds: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    tool_event_count: int
    commands: list[str]
    skill_files_used: list[str]
    events_path: str
    stderr_path: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text_file(path: Path) -> str:
    """Hash the canonical text actually supplied to the benchmark model."""
    return sha256_text(path.read_text(encoding="utf-8"))


def combined_digest(pairs: Iterable[tuple[str, str]]) -> str:
    material = "".join(f"{name}\0{digest}\n" for name, digest in sorted(pairs))
    return sha256_text(material)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def first_markdown_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def discover_tasks(benchmark_root: Path) -> list[BenchmarkTask]:
    found: list[BenchmarkTask] = []
    for directory in benchmark_root.iterdir():
        if not directory.is_dir():
            continue
        match = re.match(r"^T(\d+)_", directory.name, flags=re.IGNORECASE)
        if not match:
            continue
        task_path = directory / "task.md"
        if not task_path.is_file():
            raise RuntimeError(f"Missing task.md in {directory}")
        number_text = match.group(1)
        text = task_path.read_text(encoding="utf-8")
        found.append(
            BenchmarkTask(
                number=int(number_text),
                case_id=f"T{number_text}",
                directory_name=directory.name,
                path=task_path.resolve(),
                title=first_markdown_heading(text, directory.name),
                text=text,
                source_sha256=sha256_text(text),
            )
        )
    found.sort(key=lambda item: item.number)
    numbers = [item.number for item in found]
    if len(found) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} benchmark tasks, discovered {len(found)}"
        )
    expected = list(range(1, EXPECTED_TASK_COUNT + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        duplicates = sorted(number for number, count in Counter(numbers).items() if count > 1)
        raise RuntimeError(f"Task numbering is not exactly 1..500; missing={missing}, duplicates={duplicates}")
    return found


def _frontmatter_value(text: str, key: str) -> str:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if not match:
        return ""
    key_match = re.search(rf"^{re.escape(key)}:\s*[\"']?(.*?)[\"']?\s*$", match.group(1), flags=re.MULTILINE)
    return key_match.group(1).strip() if key_match else ""


def _first_body_paragraph(text: str) -> str:
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)
    in_fence = False
    paragraph: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith(("#", "-", "*", "|", ">")):
            if paragraph:
                break
            continue
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(re.sub(r"\s+", " ", stripped))
    return " ".join(paragraph)[:500]


def load_local_skills(skills_root: Path) -> list[SkillInfo]:
    skills: list[SkillInfo] = []
    for skill_md in sorted(skills_root.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        name = _frontmatter_value(text, "name") or skill_md.parent.name
        description = _frontmatter_value(text, "description")
        summary = _first_body_paragraph(text) or description
        skills.append(
            SkillInfo(
                name=name,
                description=description,
                summary=summary,
                skill_md=skill_md.resolve(),
            )
        )
    if not skills:
        raise RuntimeError(f"No local NeuroRuntime skills found under {skills_root}")
    return skills


def materialize_skill_snapshot(
    source_skills: Sequence[SkillInfo],
    output_root: Path,
    *,
    write_files: bool,
) -> list[SkillInfo]:
    """Copy the frozen skill library for deterministic prompt preloading and audit."""
    snapshot_root = output_root / "control" / "skill_snapshot" / "skills"
    mapped: list[SkillInfo] = []
    seen_destinations: set[Path] = set()
    for skill in source_skills:
        destination = snapshot_root / skill.skill_md.parent.name / "SKILL.md"
        resolved_destination = destination.resolve()
        if resolved_destination in seen_destinations:
            raise RuntimeError(f"Duplicate skill snapshot destination: {resolved_destination}")
        seen_destinations.add(resolved_destination)
        source_text = skill.skill_md.read_text(encoding="utf-8")
        if write_files:
            if destination.exists():
                if sha256_file(destination) != sha256_text(source_text):
                    raise RuntimeError(f"Frozen skill snapshot changed: {destination}")
            else:
                atomic_write_text(destination, source_text)
            os.chmod(destination, stat.S_IREAD)
        mapped.append(
            SkillInfo(
                name=skill.name,
                description=skill.description,
                summary=skill.summary,
                skill_md=resolved_destination,
            )
        )
    return mapped


def skill_snapshot_hashes(skills: Sequence[SkillInfo]) -> dict[str, str]:
    return {
        f"skills/{skill.skill_md.parent.name}/SKILL.md": sha256_file(skill.skill_md)
        for skill in skills
    }


def cheap_workspace_fingerprint(root: Path) -> str:
    """Fingerprint names and cheap stat fields, without repeatedly hashing content."""
    entries: list[tuple[str, str, int, int, int]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        info = path.stat()
        entries.append(
            (
                path.relative_to(root).as_posix(),
                "d" if path.is_dir() else "f",
                info.st_size,
                info.st_mtime_ns,
                int(getattr(info, "st_file_attributes", 0)),
            )
        )
    return sha256_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")))


STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "task", "test",
    "case", "required", "input", "output", "expected", "save", "results", "result",
    "using", "use", "run", "create", "generate", "report", "file", "files", "data",
    "model", "models", "pipeline", "analysis", "check", "evaluation", "benchmark",
    "should", "must", "under", "all", "each", "per", "full", "new", "based",
}


# Deterministic high-precision routing comes before the lexical fallback.  The
# benchmark task names deliberately expose the canonical tool/modality family,
# while long task bodies contain many generic words that otherwise over-rank a
# broad skill such as ICA or logging.
TASK_SKILL_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"academic|arxiv|pubmed|openreview|semantic scholar|literature search", ("academic-research-hub", "multi-search-engine")),
    (r"bids|participants\.tsv|scans\.tsv|dataset_description", ("bids-organizer",)),
    (r"dicom|dcm2nii|dcm2bids|heudiconv", ("dcm2nii", "nii2dcm", "bids-organizer")),
    (r"nifti|\.nii|nibabel|header audit|compress", ("nibabel-skill", "bids-organizer")),
    (r"\bconda\b|dependency conflict|requirements(?:\.txt)?|lockfile|pip constraints|support env|repro env capture|virtual environment", ("conda-env-manager", "dependency-planner")),
    (r"docker|container|apptainer|singularity|neurodocker", ("docker-env-manager", "dependency-planner")),
    (r"\bgit\b|github|datalad|annex|dvc|versioning", ("git-essentials", "git-workflows")),
    (r"overleaf|latex|bibtex|zotero|citation", ("overleaf-skill", "paper-writing")),
    (r"\bdipy\b|diffusion|\bdwi\b|\bdti\b|\bdki\b|\btract(?:ography)?\b|mrtrix|qsiprep|qsirecon", ("dipy-tool", "dwi-skill", "qsiprep-tool")),
    (r"freesurfer|recon-all|hippo|subfield", ("freesurfer-tool", "smri-skill")),
    (r"\bfsl\b|fslmaths|feat|bedpostx|probtrackx|flame", ("fsl-tool",)),
    (r"hcp pipeline|hcppipeline|cifti|ciftify|msm", ("hcppipeline-tool", "hcpya-skill")),
    (r"nilearn|fmri|bold|connectivity|confound|glm|xcp|fmriprep|cpac", ("nilearn-tool", "fmri-skill")),
    (r"\bica\b|canica|aroma", ("ica",)),
    (r"eeg|\bmne\b|erp", ("mne-eeg-tool", "eeg-skill")),
    (r"\bmeg\b|maxfilter", ("meg-skill",)),
    (r"wmh|white matter hyperintens", ("wmh-segmentation",)),
    (r"\basl\b|arterial spin", ("asl-skill",)),
    (r"\bpet\b", ("pet-skill",)),
    (r"adni", ("adni-skill",)),
    (r"abide", ("abide-skill",)),
    (r"oasis", ("oasis-skill",)),
    (r"camcan", ("camcan-skill",)),
    (r"cobre", ("cobre-skill",)),
    (r"pnc", ("pnc-skill",)),
    (r"hbn", ("hbn-skill",)),
    (r"brain.?gnn", ("brain_gnn", "braingnn")),
    (r"brainnetcnn", ("brainnetcnn",)),
    (r"\bbnt\b|transformer", ("bnt", "temporal-models")),
    (r"ibgnn", ("ibgnn",)),
    (r"lggnn", ("lggnn",)),
    (r"combraintf", ("combraintf",)),
    (r"neurostorm", ("neurostorm",)),
    (r"harmoni|combat|site shift|domain adaptation", ("harmonization-tool",)),
    (r"survival|cox|time.to.event", ("survival-models",)),
    (r"causal|treatment", ("causal-treatment-models",)),
    (r"brain age|age prediction", ("brain-age-modeling",)),
    (r"subtyp|cluster|kmeans", ("subject-subtyping", "kmeans")),
    (r"statistic|bootstrap|confidence interval|anova|friedman|mcnemar|delong|reliability|icc|variance", ("statistical-ml",)),
    (r"evaluation protocol|scorecard|leaderboard|metric definition|reproducibility checklist|ablation|sweep", ("experiment-controller", "statistical-ml", "beautiful-log")),
    (r"document|report|table|plain language abstract|reviewer|manuscript", ("paper-writing", "beautiful-log")),
    (r"knowledge graph|citation network|cocitation|bibliographic|graph mapping", ("knowledge-graph-builder",)),
    (r"visuali|figure|plot|graphical abstract|dashboard", ("brain-visualization",)),
)


def tokens(text: str) -> list[str]:
    values = re.findall(r"[a-z][a-z0-9]+", text.lower())
    return [value for value in values if len(value) >= 3 and value not in STOPWORDS]


def select_candidate_skills(
    task: BenchmarkTask,
    skills: Sequence[SkillInfo],
    *,
    limit: int = 12,
) -> list[SkillInfo]:
    task_blob = f"{task.directory_name}\n{task.title}\n{task.text}".lower()
    task_tokens = Counter(tokens(task_blob))
    document_frequency: Counter[str] = Counter()
    skill_token_sets: list[set[str]] = []
    for skill in skills:
        token_set = set(tokens(f"{skill.name} {skill.description} {skill.summary}"))
        skill_token_sets.append(token_set)
        document_frequency.update(token_set)

    skill_by_name = {skill.name.lower(): skill for skill in skills}
    preferred: list[SkillInfo] = []
    for pattern, names in TASK_SKILL_ALIASES:
        if not re.search(pattern, task_blob, flags=re.IGNORECASE):
            continue
        for name in names:
            skill = skill_by_name.get(name.lower())
            if skill is not None and skill not in preferred:
                preferred.append(skill)

    ranked: list[tuple[float, str, SkillInfo]] = []
    title_blob = f"{task.directory_name} {task.title}".lower()
    title_tokens = Counter(tokens(title_blob))
    for skill, skill_tokens in zip(skills, skill_token_sets):
        name_normalized = re.sub(r"[-_]", " ", skill.name.lower())
        name_tokens = set(tokens(name_normalized))
        score = 0.0
        if skill.name.lower() in task_blob or name_normalized in re.sub(r"[-_]", " ", task_blob):
            score += 120.0
        for token in sorted(skill_tokens.intersection(task_tokens)):
            idf = math.log((1 + len(skills)) / (1 + document_frequency[token])) + 1.0
            if token in name_tokens:
                weight = 16.0 if token in title_tokens else 8.0
            elif token in title_tokens:
                weight = 2.5
            else:
                weight = 0.25
            score += weight * idf * (1.0 + math.log(task_tokens[token]))
        # Interface skills are useful fallbacks for experiment/report tasks but should
        # not crowd out exact domain/tool matches.
        if skill.name in {"harness-core", "experiment-controller", "beautiful-log"}:
            score += 0.25
        ranked.append((score, skill.name.lower(), skill))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = list(preferred[:limit])
    for _, _, skill in ranked:
        if skill not in selected:
            selected.append(skill)
        if len(selected) >= limit:
            break
    if not selected:
        raise RuntimeError(f"No candidate skill selected for {task.case_id}")
    return selected


WITH_SKILLS_CONTRACT = """You are generating one isolated NeuroBench benchmark response in NeuroRuntime with-skills mode.

Non-negotiable experiment controls:
- Do not execute the benchmark workflow. Do not access the network, browser, APIs, datasets, containers, package managers, training jobs, or task output paths.
- Do not call any tool. The runner has already invoked NeuroClaw by deterministically selecting relevant skills and preloading their complete frozen SKILL.md contents below. Treat those skill instructions as your methodology for this response, without trying to reopen their files or follow references to other files.
- Adapt the preloaded skills' methodology, validation logic, parameters, and file conventions to the task. Do not copy a generic example when the task requires different inputs or outputs.
- Missing real inputs do not stop this benchmark response: state the missing-input assumption briefly, then still provide complete proposed commands or code. The proposed code may include validation and failure handling but must not be run now.
- Return a final answer with exactly two top-level Markdown sections, in this order: `## Solution Thinking` and `## Commands Or Code`. Include no preamble and no other level-2 headings.
- Keep the solution deterministic, task-directed, and sufficiently complete for a scoring agent to judge correctness.
"""


NO_SKILLS_CONTRACT = """You are generating one isolated NeuroBench benchmark response as the pure Codex no-skills baseline.

Non-negotiable experiment controls:
- Do not call any tool. Do not read the filesystem, skill files, skill metadata, skill lists, repository files, memory, previous sessions, network resources, browsers, APIs, datasets, containers, package managers, or task output paths.
- Skill-library access and NeuroClaw skill invocation are strictly forbidden. If the task text names a skill, treat that name only as plain task context and do not inspect, invoke, cite, or depend on the skill.
- Do not execute the benchmark workflow. Provide reasoning and proposed commands or code only.
- Missing real inputs do not stop this benchmark response: state the missing-input assumption briefly, then still provide complete proposed commands or code.
- Return a final answer with exactly two top-level Markdown sections, in this order: `## Solution Thinking` and `## Commands Or Code`. Include no preamble and no other level-2 headings.
- Keep the solution deterministic, task-directed, and sufficiently complete for a scoring agent to judge correctness.
"""


def render_prompt(
    task: BenchmarkTask,
    condition: ConditionSpec,
    candidate_skills: Sequence[SkillInfo],
) -> str:
    if condition.skills_enabled:
        injected_skills = list(candidate_skills[:INJECTED_SKILL_COUNT])
        if not injected_skills:
            raise RuntimeError(f"No NeuroClaw skill selected for {task.case_id}")
        skill_blocks: list[str] = []
        for skill in injected_skills:
            content = skill.skill_md.read_text(encoding="utf-8")
            skill_blocks.append(
                "\n".join(
                    [
                        f"--- BEGIN PRELOADED NEUROCLAW SKILL: {skill.name} ---",
                        f"Frozen SKILL.md SHA-256: {sha256_text(content)}",
                        content.rstrip(),
                        f"--- END PRELOADED NEUROCLAW SKILL: {skill.name} ---",
                    ]
                )
            )
        prefix = (
            WITH_SKILLS_CONTRACT
            + "\nPreloaded NeuroRuntime skills (complete frozen contents):\n\n"
            + "\n\n".join(skill_blocks)
        )
    else:
        prefix = NO_SKILLS_CONTRACT
    return (
        f"{prefix}\n\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Case ID: {task.case_id}\n"
        f"Source task SHA-256: {task.source_sha256}\n\n"
        "--- BEGIN BENCHMARK TASK ---\n"
        f"{task.text.rstrip()}\n"
        "--- END BENCHMARK TASK ---\n"
    )


def top_level_h2_headings(text: str) -> list[str]:
    headings: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            match = re.match(r"^##\s+(.+?)\s*$", stripped)
            if match:
                headings.append(match.group(1).strip())
    return headings


def split_answer(text: str) -> tuple[str, str]:
    thinking_match = re.search(r"^##\s+Solution\s+Thinking\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    commands_match = re.search(r"^##\s+Commands\s+Or\s+Code\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not thinking_match or not commands_match or commands_match.start() <= thinking_match.start():
        return "", ""
    thinking = text[thinking_match.end() : commands_match.start()].strip()
    commands = text[commands_match.end() :].strip()
    return thinking, commands


def validate_answer_contract(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("## Solution Thinking"):
        return "Answer must begin with ## Solution Thinking"
    headings = [heading.lower() for heading in top_level_h2_headings(stripped)]
    if headings != ["solution thinking", "commands or code"]:
        return f"Expected exactly two level-2 headings, found {headings}"
    thinking, commands = split_answer(stripped)
    if len(thinking) < 40:
        return "Solution Thinking section is empty or too short"
    if len(commands) < 40:
        return "Commands Or Code section is empty or too short"
    return ""


def _walk_values(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)


def parse_codex_events(raw_stdout: str) -> dict[str, Any]:
    session_id = ""
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    tool_item_types: list[str] = []
    commands: list[str] = []
    parse_errors = 0
    events: list[dict[str, Any]] = []
    for line in raw_stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        event_type = str(event.get("type") or "")
        if event_type == "thread.started":
            session_id = str(event.get("thread_id") or event.get("threadId") or "")
        if event_type in {"turn.completed", "turn.finished"}:
            usage = event.get("usage") or {}
            if isinstance(usage, dict):
                input_tokens = max(input_tokens, int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0))
                cached_input_tokens = max(cached_input_tokens, int(usage.get("cached_input_tokens") or 0))
                output_tokens = max(output_tokens, int(usage.get("output_tokens") or usage.get("completion_tokens") or 0))
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type") or "")
            if item_type and item_type not in {"agent_message", "reasoning", "plan", "todo_list", "error"}:
                tool_item_types.append(item_type)
            command = item.get("command")
            if isinstance(command, str) and command.strip():
                commands.append(command.strip())
            elif isinstance(command, list):
                commands.extend(str(part).strip() for part in command if str(part).strip())
        for key, value in _walk_values(event):
            if key in {"command", "cmd", "script"} and isinstance(value, str) and value.strip():
                commands.append(value.strip())

    commands = list(dict.fromkeys(commands))
    skill_files: list[str] = []
    skill_pattern = re.compile(r"skills[\\/]+([^\\/\s'\"`]+)[\\/]+SKILL\.md", flags=re.IGNORECASE)
    for command in commands:
        normalized = command.replace("\\\\", "\\")
        for match in skill_pattern.finditer(normalized):
            skill_files.append(match.group(1))
    skill_files = sorted(set(skill_files), key=str.lower)
    return {
        "session_id": session_id,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "tool_item_types": tool_item_types,
        "tool_event_count": len(tool_item_types),
        "commands": commands,
        "skill_files_read": skill_files,
        "parse_errors": parse_errors,
        "event_count": len(events),
    }


def validate_tool_policy(condition: ConditionSpec, event_summary: dict[str, Any]) -> str:
    tool_types = list(event_summary.get("tool_item_types") or [])
    commands = list(event_summary.get("commands") or [])
    if tool_types or commands:
        label = "With-skills" if condition.skills_enabled else "No-skills"
        return f"{label} condition observed forbidden tool activity: types={tool_types}, commands={commands[:2]}"
    return ""


def classify_retry(error_text: str) -> str:
    lowered = error_text.lower()
    if any(token in lowered for token in ("429", "rate limit", "too many requests", "usage limit", "quota", "capacity")):
        return "rate_limit"
    if any(token in lowered for token in ("timed out", "timeout")):
        return "timeout"
    if any(token in lowered for token in ("model", "reasoning effort", "unsupported")):
        return "configuration"
    if any(token in lowered for token in ("tool activity", "disallowed tool", "non-inspection command", "mutated isolated")):
        return "policy"
    if any(token in lowered for token in ("answer must", "level-2 headings", "too short")):
        return "contract"
    return "transient"


def codex_feature_flags(condition: ConditionSpec) -> list[str]:
    disabled = [
        "apps", "browser_use", "browser_use_external", "in_app_browser", "computer_use",
        "image_generation", "multi_agent", "multi_agent_v2", "plugins", "skill_search",
        "hooks", "standalone_web_search",
    ]
    disabled.extend(["shell_tool", "unified_exec"])
    args: list[str] = ["--enable", "skip_host_skill_discovery"]
    for feature in disabled:
        args.extend(["--disable", feature])
    return args


def build_codex_command(
    codex_path: str,
    condition: ConditionSpec,
    workspace: Path,
    last_message_path: Path,
) -> list[str]:
    return [
        codex_path,
        "exec",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--model",
        condition.model,
        "--config",
        f'model_reasoning_effort="{condition.reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--config",
        "sandbox_workspace_write.network_access=false",
        "--config",
        "suppress_unstable_features_warning=true",
        "--cd",
        str(workspace),
        "--output-last-message",
        str(last_message_path),
        *codex_feature_flags(condition),
        "-",
    ]


def escape_inline_code(text: str) -> str:
    return text.replace("`", "'").replace("\r", " ").replace("\n", " ")


def render_report(
    task: BenchmarkTask,
    condition: ConditionSpec,
    *,
    response: str,
    session_id: str,
    prompt_sha256: str,
    response_sha256: str,
    elapsed_seconds: float,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    commands: Sequence[str],
    skill_files_used: Sequence[str],
) -> str:
    thinking, code = split_answer(response.strip())
    skill_names = sorted(set(skill_files_used), key=str.lower) if condition.skills_enabled else []
    skill_delivery = "frozen SKILL.md contents preloaded by runner" if skill_names else "none"
    tool_lines = [
        f"{index}. [forbidden-tool-observation] `{escape_inline_code(command)}`"
        for index, command in enumerate(commands, start=1)
    ]
    if not tool_lines:
        tool_lines = ["1. No tool command was used."]
    report_stem = f"{task.case_id}_{condition.report_model}"
    total_tokens = input_tokens + output_tokens
    return "\n".join(
        [
            f"# Benchmark Report: {report_stem}",
            "",
            f"- Timestamp: {utc_now()}",
            "- Mode: benchmark",
            f"- Benchmark variant: {condition.variant}",
            f"- Case ID: {task.case_id}",
            f"- Model: {condition.report_model}",
            f"- Actual model: {condition.model}",
            f"- Reasoning effort: {condition.reasoning_effort}",
            "- Session isolation: fresh process; codex exec --ephemeral; no resume",
            f"- Session ID: {session_id}",
            f"- Prompt version: {PROMPT_VERSION}",
            f"- Source task SHA-256: {task.source_sha256}",
            f"- Prompt SHA-256: {prompt_sha256}",
            f"- Response SHA-256: {response_sha256}",
            f"- Task: {task.title}",
            f"- Elapsed seconds: {elapsed_seconds:.3f}",
            f"- Token usage: prompt={input_tokens}, completion={output_tokens}, total={total_tokens}",
            f"- Cached input tokens: {cached_input_tokens}",
            f"- Skill delivery: {skill_delivery}",
            f"- Skills used (unique): {len(skill_names)}",
            f"- Skill names used: {json.dumps(skill_names, ensure_ascii=False)}",
            f"- Skill files preloaded by runner (unique): {len(skill_names)}",
            f"- Skill files preloaded by runner: {json.dumps(skill_names, ensure_ascii=False)}",
            "- Skill files read via tool (unique): 0",
            "- Skill files read via tool: []",
            "",
            "## Solution Thinking",
            thinking,
            "",
            "## Commands Or Code",
            code,
            "",
            "## Tool Events",
            *tool_lines,
            "",
        ]
    )


def run_attempt(
    *,
    codex_path: str,
    output_root: Path,
    isolated_workspace: Path,
    task: BenchmarkTask,
    condition: ConditionSpec,
    candidate_skills: Sequence[SkillInfo],
    attempt: int,
    timeout_seconds: int,
) -> AttemptResult:
    prompt = render_prompt(task, condition, candidate_skills)
    prompt_sha = sha256_text(prompt)
    prompt_path = output_root / "prompts" / condition.key / f"{task.case_id}.txt"
    if prompt_path.exists():
        existing = prompt_path.read_text(encoding="utf-8")
        if sha256_text(existing) != prompt_sha:
            raise RuntimeError(f"Frozen prompt changed for {condition.key}/{task.case_id}")
    else:
        atomic_write_text(prompt_path, prompt)

    events_path = output_root / "events" / condition.key / f"{task.case_id}.attempt{attempt}.jsonl"
    stderr_path = output_root / "stderr" / condition.key / f"{task.case_id}.attempt{attempt}.txt"
    attempt_tmp_dir = output_root / ".attempt_tmp" / condition.key / task.case_id
    attempt_tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, raw_last_path = tempfile.mkstemp(prefix=f"attempt{attempt}.", suffix=".last.md", dir=attempt_tmp_dir)
    os.close(fd)
    last_path = Path(raw_last_path)
    workspace = (
        output_root / "control" / "with_skills_session_workspace"
        if condition.skills_enabled
        else isolated_workspace
    )
    workspace_fingerprint_before = cheap_workspace_fingerprint(workspace)
    command = build_codex_command(codex_path, condition, workspace, last_path)
    started = time.monotonic()
    raw_stdout = ""
    raw_stderr = ""
    response = ""
    return_code = -1
    error = ""
    try:
        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        kwargs: dict[str, Any] = {
            "input": prompt,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "capture_output": True,
            "timeout": timeout_seconds,
            "cwd": str(workspace),
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        completed = subprocess.run(command, **kwargs)
        return_code = completed.returncode
        raw_stdout = completed.stdout or ""
        raw_stderr = completed.stderr or ""
        if last_path.exists():
            response = last_path.read_text(encoding="utf-8", errors="replace").strip()
    except subprocess.TimeoutExpired as exc:
        raw_stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", errors="replace")
        raw_stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", errors="replace")
        error = f"Codex invocation timed out after {timeout_seconds}s"
    except Exception as exc:  # noqa: BLE001 - persisted as task failure evidence
        error = f"Codex invocation failed: {type(exc).__name__}: {exc}"
    finally:
        elapsed = time.monotonic() - started
        with contextlib.suppress(OSError):
            last_path.unlink()

    atomic_write_text(events_path, raw_stdout)
    atomic_write_text(stderr_path, raw_stderr)
    event_summary = parse_codex_events(raw_stdout)
    workspace_fingerprint_after = cheap_workspace_fingerprint(workspace)

    if not error and return_code != 0:
        tail = (raw_stderr.strip() or raw_stdout.strip())[-2000:]
        error = f"Codex exited with code {return_code}: {tail}"
    if not error and not event_summary.get("session_id"):
        error = "Codex JSON stream did not provide a fresh session ID"
    if not error:
        error = validate_answer_contract(response)
    if not error:
        error = validate_tool_policy(condition, event_summary)
    if not error and workspace_fingerprint_after != workspace_fingerprint_before:
        error = "Codex mutated isolated session workspace"

    response_sha = sha256_text(response) if response else ""
    skill_files_used = (
        [skill.name for skill in candidate_skills[:INJECTED_SKILL_COUNT]]
        if condition.skills_enabled
        else []
    )
    report_path = output_root / "reports" / condition.report_model / f"{task.case_id}_{condition.report_model}.md"
    report_sha = ""
    if not error:
        report = render_report(
            task,
            condition,
            response=response,
            session_id=str(event_summary["session_id"]),
            prompt_sha256=prompt_sha,
            response_sha256=response_sha,
            elapsed_seconds=elapsed,
            input_tokens=int(event_summary["input_tokens"]),
            cached_input_tokens=int(event_summary["cached_input_tokens"]),
            output_tokens=int(event_summary["output_tokens"]),
            commands=list(event_summary["commands"]),
            skill_files_used=skill_files_used,
        )
        atomic_write_text(report_path, report)
        report_sha = sha256_text(report)

    return AttemptResult(
        ok=not bool(error),
        error=error,
        retry_class=classify_retry(error) if error else "",
        session_id=str(event_summary.get("session_id") or ""),
        response=response,
        response_sha256=response_sha,
        report_path=str(report_path) if not error else "",
        report_sha256=report_sha,
        elapsed_seconds=elapsed,
        input_tokens=int(event_summary.get("input_tokens") or 0),
        cached_input_tokens=int(event_summary.get("cached_input_tokens") or 0),
        output_tokens=int(event_summary.get("output_tokens") or 0),
        tool_event_count=int(event_summary.get("tool_event_count") or 0),
        commands=list(event_summary.get("commands") or []),
        skill_files_used=skill_files_used,
        events_path=str(events_path),
        stderr_path=str(stderr_path),
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS task_runs (
    condition TEXT NOT NULL,
    case_id TEXT NOT NULL,
    task_number INTEGER NOT NULL,
    task_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL DEFAULT '',
    requested_model TEXT NOT NULL,
    reasoning_effort TEXT NOT NULL,
    variant TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    response_sha256 TEXT NOT NULL DEFAULT '',
    report_path TEXT NOT NULL DEFAULT '',
    report_sha256 TEXT NOT NULL DEFAULT '',
    elapsed_seconds REAL NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    tool_event_count INTEGER NOT NULL DEFAULT 0,
    skill_files_used_json TEXT NOT NULL DEFAULT '[]',
    events_path TEXT NOT NULL DEFAULT '',
    stderr_path TEXT NOT NULL DEFAULT '',
    last_error TEXT NOT NULL DEFAULT '',
    retry_class TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (condition, case_id)
);
CREATE INDEX IF NOT EXISTS idx_task_runs_status ON task_runs(condition, status, task_number);
CREATE TABLE IF NOT EXISTS campaign_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    condition TEXT NOT NULL DEFAULT '',
    case_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}'
);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(SCHEMA)
    return connection


def connect_db_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise RuntimeError(f"Campaign ledger does not exist: {db_path}")
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=60000")
    return connection


def record_event(
    connection: sqlite3.Connection,
    event_type: str,
    *,
    condition: str = "",
    case_id: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT INTO campaign_events(timestamp,event_type,condition,case_id,payload_json) VALUES(?,?,?,?,?)",
        (utc_now(), event_type, condition, case_id, json.dumps(payload or {}, ensure_ascii=False)),
    )


def git_snapshot() -> dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    status = run_git("status", "--porcelain=v1")
    return {
        "head": run_git("rev-parse", "HEAD"),
        "dirty_entry_count": len(status.splitlines()) if status else 0,
        "status_sha256": sha256_text(status),
    }


def codex_version(codex_path: str) -> str:
    completed = subprocess.run(
        [codex_path, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=True,
    )
    return completed.stdout.strip()


def codex_model_evidence(codex_path: str) -> dict[str, Any]:
    completed = subprocess.run(
        [codex_path, "debug", "models"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=True,
    )
    payload = json.loads(completed.stdout)
    models = payload.get("models") if isinstance(payload, dict) else []
    evidence: dict[str, Any] = {}
    for condition in CONDITIONS.values():
        match = next((item for item in models if item.get("slug") == condition.model), None)
        if not match:
            raise RuntimeError(f"Requested model is absent from local Codex catalog: {condition.model}")
        efforts = [item.get("effort") for item in match.get("supported_reasoning_levels", [])]
        if condition.reasoning_effort not in efforts:
            raise RuntimeError(
                f"{condition.model} does not support reasoning effort {condition.reasoning_effort}; supported={efforts}"
            )
        evidence[condition.model] = {
            "display_name": match.get("display_name"),
            "supported_reasoning_levels": efforts,
            "verified_requested_effort": condition.reasoning_effort,
        }
    return evidence


def validate_additive_condition_extension(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    """Allow a frozen campaign to gain conditions without changing old ones."""
    removed = [key for key in existing if key not in current]
    if removed:
        raise RuntimeError(
            f"Frozen experiment conditions are absent from the runner: {removed}"
        )
    changed = [key for key, value in existing.items() if current.get(key) != value]
    if changed:
        raise RuntimeError(
            f"Frozen experiment condition definitions changed: {changed}"
        )
    return [key for key in current if key not in existing]


def initialize_campaign(
    output_root: Path,
    tasks: Sequence[BenchmarkTask],
    skills: Sequence[SkillInfo],
    codex_path: str,
) -> sqlite3.Connection:
    output_root.mkdir(parents=True, exist_ok=True)
    isolated_workspace = output_root / "control" / "no_skill_workspace"
    isolated_workspace.mkdir(parents=True, exist_ok=True)
    with_skills_session_workspace = output_root / "control" / "with_skills_session_workspace"
    with_skills_session_workspace.mkdir(parents=True, exist_ok=True)
    source_pairs = [(task.case_id, task.source_sha256) for task in tasks]
    snapshot_files = skill_snapshot_hashes(skills)
    snapshot_root_sha256 = combined_digest(snapshot_files.items())
    manifest_path = output_root / "experiment_manifest.json"
    current_manifest_core = {
        "experiment_id": output_root.name,
        "prompt_version": PROMPT_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "benchmark_root": str(DEFAULT_BENCHMARK_ROOT.resolve()),
        "task_count": len(tasks),
        "task_source_root_sha256": combined_digest(source_pairs),
        "skill_snapshot_root": str((output_root / "control" / "skill_snapshot").resolve()),
        "skill_snapshot_root_sha256": snapshot_root_sha256,
        "skill_snapshot_files": snapshot_files,
        "conditions": {key: dataclasses.asdict(value) for key, value in CONDITIONS.items()},
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in current_manifest_core.items():
            if key == "conditions":
                continue
            if manifest.get(key) != value:
                raise RuntimeError(f"Frozen experiment manifest mismatch for {key}")
        existing_conditions = manifest.get("conditions")
        if not isinstance(existing_conditions, dict):
            raise RuntimeError("Frozen experiment manifest conditions are missing or invalid")
        added_conditions = validate_additive_condition_extension(
            existing_conditions,
            current_manifest_core["conditions"],
        )
        if added_conditions:
            expansion_history = list(manifest.get("condition_expansion_history") or [])
            expansion_history.append(
                {
                    "expanded_at": utc_now(),
                    "previous_condition_keys": list(existing_conditions),
                    "added_condition_keys": added_conditions,
                    "resulting_condition_keys": list(current_manifest_core["conditions"]),
                    "reason": "expanded diagonal comparison to the complete model-by-skill 2x2 matrix",
                }
            )
            manifest["conditions"] = current_manifest_core["conditions"]
            manifest["condition_count"] = len(current_manifest_core["conditions"])
            manifest["factorial_design"] = {
                "model": ["gpt-5.6-sol", "gpt-5.5"],
                "skill_delivery": ["with-skills", "no-skills"],
            }
            manifest["condition_expansion_history"] = expansion_history
            manifest["execution_contract"] = (
                "the runner preloads complete immutable SKILL.md contents into with-skills prompts; "
                "all four conditions use isolated read-only empty session workspaces with all tools "
                "disabled; no benchmark workflow execution"
            )
            atomic_write_json(manifest_path, manifest)
    else:
        manifest = {
            **current_manifest_core,
            "created_at": utc_now(),
            "codex_cli": codex_version(codex_path),
            "codex_path": str(Path(codex_path).resolve()),
            "model_catalog_evidence": codex_model_evidence(codex_path),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256_at_creation": sha256_file(Path(__file__).resolve()),
            "git_snapshot": git_snapshot(),
            "fresh_session_contract": "one codex exec --ephemeral process per task; never resume",
            "execution_contract": (
                "the runner preloads complete immutable SKILL.md contents into with-skills prompts; "
                "all four conditions use isolated read-only empty session workspaces with all tools "
                "disabled; no benchmark workflow execution"
            ),
            "condition_count": len(CONDITIONS),
            "factorial_design": {
                "model": ["gpt-5.6-sol", "gpt-5.5"],
                "skill_delivery": ["with-skills", "no-skills"],
            },
            "skill_library_count": len(skills),
            "tasks": [
                {
                    "number": task.number,
                    "case_id": task.case_id,
                    "directory": task.directory_name,
                    "path": str(task.path),
                    "source_sha256": task.source_sha256,
                    "candidate_skills": [
                        skill.name for skill in select_candidate_skills(task, skills)
                    ],
                    "injected_skills": [
                        skill.name
                        for skill in select_candidate_skills(task, skills)[:INJECTED_SKILL_COUNT]
                    ],
                }
                for task in tasks
            ],
        }
        atomic_write_json(manifest_path, manifest)

    connection = connect_db(output_root / "state.sqlite3")
    with connection:
        for task in tasks:
            for condition in CONDITIONS.values():
                prompt = render_prompt(
                    task,
                    condition,
                    select_candidate_skills(task, skills),
                )
                prompt_sha = sha256_text(prompt)
                existing = connection.execute(
                    "SELECT source_sha256,prompt_sha256,requested_model,reasoning_effort,variant FROM task_runs WHERE condition=? AND case_id=?",
                    (condition.key, task.case_id),
                ).fetchone()
                expected = (
                    task.source_sha256,
                    prompt_sha,
                    condition.model,
                    condition.reasoning_effort,
                    condition.variant,
                )
                if existing:
                    actual = tuple(existing[key] for key in existing.keys())
                    if actual != expected:
                        raise RuntimeError(
                            f"Frozen ledger mismatch for {condition.key}/{task.case_id}: {actual} != {expected}"
                        )
                else:
                    connection.execute(
                        """INSERT INTO task_runs(
                            condition,case_id,task_number,task_path,source_sha256,prompt_sha256,
                            requested_model,reasoning_effort,variant,status
                        ) VALUES(?,?,?,?,?,?,?,?,?,'pending')""",
                        (
                            condition.key,
                            task.case_id,
                            task.number,
                            str(task.path),
                            task.source_sha256,
                            prompt_sha,
                            condition.model,
                            condition.reasoning_effort,
                            condition.variant,
                        ),
                    )
        record_event(
            connection,
            "campaign_initialized",
            payload={"task_count": len(tasks), "condition_keys": list(CONDITIONS)},
        )
    return connection


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def manager_lock(output_root: Path) -> Iterator[None]:
    lock_path = output_root / "manager.lock"
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
        except Exception:
            pid = 0
        if process_is_alive(pid):
            raise RuntimeError(f"Another campaign manager is active with PID {pid}")
        stale_path = output_root / f"manager.lock.stale.{int(time.time())}"
        os.replace(lock_path, stale_path)
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at": utc_now()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def select_tasks(
    tasks: Sequence[BenchmarkTask],
    *,
    task_ids: str,
    task_start: int,
    task_end: int,
) -> list[BenchmarkTask]:
    wanted_ids = {value.strip().upper() for value in task_ids.split(",") if value.strip()}
    selected = [
        task
        for task in tasks
        if task_start <= task.number <= task_end
        and (not wanted_ids or task.case_id.upper() in wanted_ids)
    ]
    if wanted_ids:
        found_ids = {task.case_id.upper() for task in selected}
        if found_ids != wanted_ids:
            raise RuntimeError(f"Unknown or out-of-range task IDs: {sorted(wanted_ids - found_ids)}")
    if not selected:
        raise RuntimeError("Task selection is empty")
    return selected


def update_started(connection: sqlite3.Connection, condition: ConditionSpec, task: BenchmarkTask) -> int:
    row = connection.execute(
        "SELECT attempts FROM task_runs WHERE condition=? AND case_id=?",
        (condition.key, task.case_id),
    ).fetchone()
    attempt = int(row["attempts"]) + 1
    with connection:
        connection.execute(
            """UPDATE task_runs SET status='running',attempts=?,started_at=?,finished_at='',
               last_error='',retry_class='' WHERE condition=? AND case_id=?""",
            (attempt, utc_now(), condition.key, task.case_id),
        )
        record_event(
            connection,
            "attempt_started",
            condition=condition.key,
            case_id=task.case_id,
            payload={"attempt": attempt},
        )
    return attempt


def update_finished(
    connection: sqlite3.Connection,
    condition: ConditionSpec,
    task: BenchmarkTask,
    result: AttemptResult,
    *,
    max_attempts: int,
) -> str:
    row = connection.execute(
        "SELECT attempts FROM task_runs WHERE condition=? AND case_id=?",
        (condition.key, task.case_id),
    ).fetchone()
    attempts = int(row["attempts"])
    status = "success" if result.ok else ("pending" if attempts < max_attempts else "failed")
    with connection:
        connection.execute(
            """UPDATE task_runs SET status=?,finished_at=?,session_id=?,response_sha256=?,
               report_path=?,report_sha256=?,elapsed_seconds=?,input_tokens=?,cached_input_tokens=?,
               output_tokens=?,tool_event_count=?,skill_files_used_json=?,events_path=?,stderr_path=?,
               last_error=?,retry_class=? WHERE condition=? AND case_id=?""",
            (
                status,
                utc_now(),
                result.session_id,
                result.response_sha256,
                result.report_path,
                result.report_sha256,
                result.elapsed_seconds,
                result.input_tokens,
                result.cached_input_tokens,
                result.output_tokens,
                result.tool_event_count,
                json.dumps(result.skill_files_used, ensure_ascii=False),
                result.events_path,
                result.stderr_path,
                result.error,
                result.retry_class,
                condition.key,
                task.case_id,
            ),
        )
        record_event(
            connection,
            "attempt_succeeded" if result.ok else "attempt_failed",
            condition=condition.key,
            case_id=task.case_id,
            payload={
                "status": status,
                "attempt": attempts,
                "retry_class": result.retry_class,
                "error": result.error[:1000],
            },
        )
    return status


def condition_counts(connection: sqlite3.Connection, condition_key: str) -> dict[str, int]:
    rows = connection.execute(
        "SELECT status,COUNT(*) AS n FROM task_runs WHERE condition=? GROUP BY status",
        (condition_key,),
    ).fetchall()
    counts = {"pending": 0, "running": 0, "success": 0, "failed": 0}
    counts.update({str(row["status"]): int(row["n"]) for row in rows})
    return counts


def run_condition(
    *,
    connection: sqlite3.Connection,
    output_root: Path,
    codex_path: str,
    tasks: Sequence[BenchmarkTask],
    skills: Sequence[SkillInfo],
    condition: ConditionSpec,
    workers: int,
    timeout_seconds: int,
    max_attempts: int,
    retry_failed: bool,
) -> None:
    if retry_failed:
        with connection:
            connection.execute(
                "UPDATE task_runs SET status='pending' WHERE condition=? AND status='failed' AND attempts<?",
                (condition.key, max_attempts),
            )
    with connection:
        connection.execute(
            "UPDATE task_runs SET status='pending',last_error='Recovered interrupted attempt' WHERE condition=? AND status='running'",
            (condition.key,),
        )

    task_by_id = {task.case_id: task for task in tasks}
    rows = connection.execute(
        """SELECT case_id,status,attempts FROM task_runs
           WHERE condition=? ORDER BY task_number""",
        (condition.key,),
    ).fetchall()
    queue: deque[tuple[BenchmarkTask, float]] = deque()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in task_by_id:
            continue
        status = str(row["status"])
        attempts = int(row["attempts"])
        if status == "success":
            continue
        if status == "failed" and (not retry_failed or attempts >= max_attempts):
            continue
        if attempts < max_attempts:
            queue.append((task_by_id[case_id], 0.0))

    isolated_workspace = output_root / "control" / "no_skill_workspace"
    selected_skill_map = {
        task.case_id: select_candidate_skills(task, skills) for task in tasks
    }
    in_flight: dict[concurrent.futures.Future[AttemptResult], tuple[BenchmarkTask, int]] = {}
    completed_this_run = 0
    rate_limit_until = 0.0
    print(
        f"[{condition.key}] queued={len(queue)} workers={workers} model={condition.model} effort={condition.reasoning_effort}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while queue or in_flight:
            now = time.monotonic()
            while queue and len(in_flight) < workers and now >= rate_limit_until:
                task, ready_at = queue[0]
                if ready_at > now:
                    break
                queue.popleft()
                attempt = update_started(connection, condition, task)
                future = executor.submit(
                    run_attempt,
                    codex_path=codex_path,
                    output_root=output_root,
                    isolated_workspace=isolated_workspace,
                    task=task,
                    condition=condition,
                    candidate_skills=selected_skill_map[task.case_id],
                    attempt=attempt,
                    timeout_seconds=timeout_seconds,
                )
                in_flight[future] = (task, attempt)

            if not in_flight:
                wait_until = rate_limit_until
                if queue:
                    wait_until = max(wait_until, queue[0][1])
                delay = max(0.2, min(10.0, wait_until - time.monotonic()))
                time.sleep(delay)
                continue

            done, _ = concurrent.futures.wait(
                in_flight,
                timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                task, attempt = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = AttemptResult(
                        ok=False,
                        error=f"Runner worker crashed: {type(exc).__name__}: {exc}",
                        retry_class="runner",
                        session_id="",
                        response="",
                        response_sha256="",
                        report_path="",
                        report_sha256="",
                        elapsed_seconds=0.0,
                        input_tokens=0,
                        cached_input_tokens=0,
                        output_tokens=0,
                        tool_event_count=0,
                        commands=[],
                        skill_files_used=[],
                        events_path="",
                        stderr_path="",
                    )
                status = update_finished(
                    connection,
                    condition,
                    task,
                    result,
                    max_attempts=max_attempts,
                )
                completed_this_run += 1
                counts = condition_counts(connection, condition.key)
                if result.ok:
                    print(
                        f"[{condition.key}] {task.case_id} success attempt={attempt} "
                        f"session={result.session_id} totals={counts}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{condition.key}] {task.case_id} {status} attempt={attempt} "
                        f"class={result.retry_class} error={result.error[:240]} totals={counts}",
                        flush=True,
                    )
                    if result.retry_class == "rate_limit":
                        rate_limit_until = max(
                            rate_limit_until,
                            time.monotonic() + min(300.0, 30.0 * (2 ** max(0, attempt - 1))),
                        )
                    if status == "pending":
                        delay = min(120.0, 5.0 * (2 ** max(0, attempt - 1)))
                        queue.append((task, time.monotonic() + delay))

    counts = condition_counts(connection, condition.key)
    print(f"[{condition.key}] run finished: {counts}; attempts completed this run={completed_this_run}", flush=True)


def parse_report_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("## "):
            break
        match = re.match(r"^- ([^:]+):\s*(.*)$", line)
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()
    return metadata


def audit_campaign(
    *,
    connection: sqlite3.Connection,
    output_root: Path,
    tasks: Sequence[BenchmarkTask],
    condition_keys: Sequence[str],
    final: bool,
) -> tuple[bool, dict[str, Any]]:
    task_map = {task.case_id: task for task in tasks}
    failures: list[str] = []
    report_hash_pairs: list[tuple[str, str]] = []
    source_hash_pairs = [
        (task.case_id, sha256_normalized_text_file(task.path)) for task in tasks
    ]
    manifest_path = output_root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_injected_skills = {
        str(item["case_id"]): list(item.get("injected_skills") or [])
        for item in manifest.get("tasks", [])
    }
    expected_snapshot_files = dict(manifest.get("skill_snapshot_files") or {})
    snapshot_workspace = output_root / "control" / "skill_snapshot"
    actual_snapshot_files = {
        path.relative_to(snapshot_workspace).as_posix(): sha256_file(path)
        for path in sorted(snapshot_workspace.rglob("*"), key=lambda item: item.as_posix().lower())
        if path.is_file()
    }
    snapshot_root_sha256 = combined_digest(actual_snapshot_files.items())
    if actual_snapshot_files != expected_snapshot_files:
        failures.append("Frozen NeuroClaw skill snapshot file set or content changed")
    if snapshot_root_sha256 != str(manifest.get("skill_snapshot_root_sha256") or ""):
        failures.append("Frozen NeuroClaw skill snapshot root SHA-256 mismatch")
    conditions_payload: dict[str, Any] = {}
    all_session_ids: list[str] = []

    for case_id, digest in source_hash_pairs:
        if digest != task_map[case_id].source_sha256:
            failures.append(f"Source task changed: {case_id}")

    for condition_key in condition_keys:
        condition = CONDITIONS[condition_key]
        rows = connection.execute(
            "SELECT * FROM task_runs WHERE condition=? ORDER BY task_number",
            (condition_key,),
        ).fetchall()
        selected_rows = [row for row in rows if str(row["case_id"]) in task_map]
        success_rows = [row for row in selected_rows if row["status"] == "success"]
        session_ids: list[str] = []
        condition_failures: list[str] = []
        for row in selected_rows:
            case_id = str(row["case_id"])
            if row["status"] != "success":
                condition_failures.append(f"{case_id}: status={row['status']}")
                continue
            session_id = str(row["session_id"])
            if not session_id:
                condition_failures.append(f"{case_id}: missing session ID")
            session_ids.append(session_id)
            all_session_ids.append(session_id)
            report_path = Path(str(row["report_path"]))
            if not report_path.is_file():
                condition_failures.append(f"{case_id}: missing report {report_path}")
                continue
            current_hash = sha256_file(report_path)
            if current_hash != str(row["report_sha256"]):
                condition_failures.append(f"{case_id}: report SHA-256 mismatch")
            report_hash_pairs.append((f"{condition_key}/{case_id}", current_hash))
            text = report_path.read_text(encoding="utf-8")
            metadata = parse_report_metadata(text)
            expected_metadata = {
                "Benchmark variant": condition.variant,
                "Case ID": case_id,
                "Model": condition.report_model,
                "Actual model": condition.model,
                "Reasoning effort": condition.reasoning_effort,
                "Session ID": session_id,
                "Prompt version": PROMPT_VERSION,
                "Source task SHA-256": task_map[case_id].source_sha256,
                "Prompt SHA-256": str(row["prompt_sha256"]),
                "Response SHA-256": str(row["response_sha256"]),
            }
            for key, value in expected_metadata.items():
                if metadata.get(key) != value:
                    condition_failures.append(
                        f"{case_id}: metadata {key!r}={metadata.get(key)!r}, expected {value!r}"
                    )
            if "## Solution Thinking" not in text or "## Commands Or Code" not in text:
                condition_failures.append(f"{case_id}: missing scorer sections")
            skills_used = json.loads(str(row["skill_files_used_json"] or "[]"))
            if condition.skills_enabled:
                expected_skills = expected_injected_skills.get(case_id, [])
                if skills_used != expected_skills:
                    condition_failures.append(
                        f"{case_id}: preloaded skill evidence={skills_used}, expected={expected_skills}"
                    )
            if int(row["tool_event_count"]) != 0:
                condition_failures.append(f"{case_id}: forbidden tool event count is nonzero")
            if not condition.skills_enabled:
                if skills_used:
                    condition_failures.append(f"{case_id}: no-skills tool/skill evidence is nonzero")

        if len(selected_rows) != len(task_map):
            condition_failures.append(
                f"ledger coverage={len(selected_rows)}, expected={len(task_map)}"
            )
        if len(success_rows) != len(task_map):
            condition_failures.append(
                f"success coverage={len(success_rows)}, expected={len(task_map)}"
            )
        duplicate_sessions = sorted(
            session for session, count in Counter(session_ids).items() if session and count > 1
        )
        if duplicate_sessions:
            condition_failures.append(f"duplicate session IDs: {duplicate_sessions[:5]}")
        conditions_payload[condition_key] = {
            "expected_tasks": len(task_map),
            "ledger_rows": len(selected_rows),
            "success_rows": len(success_rows),
            "unique_nonempty_session_ids": len(set(item for item in session_ids if item)),
            "report_model": condition.report_model,
            "actual_model": condition.model,
            "reasoning_effort": condition.reasoning_effort,
            "variant": condition.variant,
            "failure_count": len(condition_failures),
            "failures": condition_failures,
        }
        failures.extend(f"{condition_key}: {item}" for item in condition_failures)

    if len(set(item for item in all_session_ids if item)) != len([item for item in all_session_ids if item]):
        failures.append("Session ID reused across conditions")

    payload = {
        "audit_timestamp": utc_now(),
        "audit_kind": "final" if final else "condition-boundary",
        "ok": not failures,
        "prompt_version": PROMPT_VERSION,
        "source_task_count": len(tasks),
        "source_root_sha256": combined_digest(source_hash_pairs),
        "skill_snapshot_count": len(actual_snapshot_files),
        "skill_snapshot_root_sha256": snapshot_root_sha256,
        "report_count": len(report_hash_pairs),
        "report_root_sha256": combined_digest(report_hash_pairs),
        "conditions": conditions_payload,
        "failure_count": len(failures),
        "failures": failures,
        "files": {
            "sources": {name: digest for name, digest in source_hash_pairs},
            "skill_snapshot": actual_snapshot_files,
            "reports": {name: digest for name, digest in report_hash_pairs},
        },
    }
    if final:
        json_path = output_root / "final_audit.json"
        md_path = output_root / "final_audit.md"
    else:
        suffix = "_".join(condition_keys)
        json_path = output_root / f"integrity_{suffix}.json"
        md_path = output_root / f"integrity_{suffix}.md"
    atomic_write_json(json_path, payload)
    lines = [
        f"# {'Final' if final else 'Condition-boundary'} NeuroBench audit",
        "",
        f"- Timestamp: {payload['audit_timestamp']}",
        f"- Status: {'PASS' if payload['ok'] else 'FAIL'}",
        f"- Source tasks: {payload['source_task_count']}",
        f"- Frozen skill files: {payload['skill_snapshot_count']}",
        f"- Reports: {payload['report_count']}",
        f"- Source root SHA-256: `{payload['source_root_sha256']}`",
        f"- Skill snapshot root SHA-256: `{payload['skill_snapshot_root_sha256']}`",
        f"- Report root SHA-256: `{payload['report_root_sha256']}`",
        "",
        "## Conditions",
        "",
    ]
    for key, data in conditions_payload.items():
        lines.append(
            f"- `{key}`: success={data['success_rows']}/{data['expected_tasks']}, "
            f"unique_sessions={data['unique_nonempty_session_ids']}, failures={data['failure_count']}"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in failures[:200])
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return not failures, payload


def print_status(connection: sqlite3.Connection, output_root: Path) -> None:
    print(f"Experiment root: {output_root}")
    for condition in CONDITIONS.values():
        counts = condition_counts(connection, condition.key)
        usage = connection.execute(
            """SELECT COALESCE(SUM(input_tokens),0) AS input_tokens,
                      COALESCE(SUM(cached_input_tokens),0) AS cached_input_tokens,
                      COALESCE(SUM(output_tokens),0) AS output_tokens,
                      COALESCE(SUM(elapsed_seconds),0) AS elapsed_seconds
               FROM task_runs WHERE condition=? AND status='success'""",
            (condition.key,),
        ).fetchone()
        print(
            f"{condition.key}: {counts}; tokens=input {usage['input_tokens']}, "
            f"cached {usage['cached_input_tokens']}, output {usage['output_tokens']}; "
            f"task_elapsed={float(usage['elapsed_seconds']):.1f}s"
        )
    manifest = output_root / "experiment_manifest.json"
    if manifest.exists():
        stat = manifest.stat()
        print(
            f"Manifest fingerprint: bytes={stat.st_size}, mtime_ns={stat.st_mtime_ns}; "
            "full hashes are reserved for condition/final audit boundaries"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("init", "run", "status", "audit", "dry-run"),
        help="Campaign operation",
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--condition",
        choices=("all", *CONDITIONS.keys()),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--task-start", type=int, default=1)
    parser.add_argument("--task-end", type=int, default=EXPECTED_TASK_COUNT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    tasks = discover_tasks(benchmark_root)
    source_skills = load_local_skills(REPO_ROOT / "skills")
    if args.command in {"init", "run"}:
        skills = materialize_skill_snapshot(source_skills, output_root, write_files=True)
    elif args.command == "dry-run":
        skills = source_skills
    else:
        skills = source_skills
    selected_tasks = select_tasks(
        tasks,
        task_ids=args.task_ids,
        task_start=max(1, args.task_start),
        task_end=min(EXPECTED_TASK_COUNT, args.task_end),
    )
    condition_keys = list(CONDITIONS) if args.condition == "all" else [args.condition]

    if args.command == "dry-run":
        print(f"tasks={len(selected_tasks)} skills={len(skills)} conditions={condition_keys}")
        for task in selected_tasks[:20]:
            candidates = select_candidate_skills(task, skills)
            print(f"{task.case_id}: {', '.join(skill.name for skill in candidates)}")
            for key in condition_keys:
                prompt = render_prompt(task, CONDITIONS[key], candidates)
                print(f"  {key}: prompt_sha256={sha256_text(prompt)} bytes={len(prompt.encode('utf-8'))}")
        return 0

    if args.command in {"status", "audit"}:
        connection = connect_db_readonly(output_root / "state.sqlite3")
    else:
        codex_path = shutil.which("codex")
        if not codex_path:
            raise RuntimeError("codex executable was not found on PATH")
        connection = initialize_campaign(output_root, tasks, skills, codex_path)
    try:
        if args.command == "init":
            print_status(connection, output_root)
            return 0
        if args.command == "status":
            print_status(connection, output_root)
            return 0
        if args.command == "audit":
            ok, payload = audit_campaign(
                connection=connection,
                output_root=output_root,
                tasks=tasks,
                condition_keys=condition_keys,
                final=args.condition == "all",
            )
            print(json.dumps({"ok": ok, "report_count": payload["report_count"], "failure_count": payload["failure_count"]}))
            return 0 if ok else 2
        if args.command == "run":
            codex_path = shutil.which("codex")
            if not codex_path:
                raise RuntimeError("codex executable was not found on PATH")
            if args.workers < 1 or args.workers > 16:
                raise RuntimeError("--workers must be between 1 and 16")
            if args.max_attempts < 1:
                raise RuntimeError("--max-attempts must be positive")
            with manager_lock(output_root):
                for key in condition_keys:
                    run_condition(
                        connection=connection,
                        output_root=output_root,
                        codex_path=codex_path,
                        tasks=selected_tasks,
                        skills=skills,
                        condition=CONDITIONS[key],
                        workers=args.workers,
                        timeout_seconds=args.timeout_seconds,
                        max_attempts=args.max_attempts,
                        retry_failed=args.retry_failed,
                    )
                    # Deep verification is a phase boundary only when the full
                    # 500-task condition was selected.
                    if len(selected_tasks) == EXPECTED_TASK_COUNT:
                        ok, _ = audit_campaign(
                            connection=connection,
                            output_root=output_root,
                            tasks=tasks,
                            condition_keys=[key],
                            final=False,
                        )
                        if not ok:
                            raise RuntimeError(f"Condition-boundary audit failed for {key}")
                if args.condition == "all" and len(selected_tasks) == EXPECTED_TASK_COUNT:
                    ok, _ = audit_campaign(
                        connection=connection,
                        output_root=output_root,
                        tasks=tasks,
                        condition_keys=list(CONDITIONS),
                        final=True,
                    )
                    if not ok:
                        raise RuntimeError("Final campaign audit failed")
            print_status(connection, output_root)
            return 0
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed task state is preserved in SQLite.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
