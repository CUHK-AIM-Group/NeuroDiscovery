"""NeuroBench report parsing, scoring rubric and leaderboard generation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_SCORER_MODEL = "gpt-5.4"

BENCHMARK_SCORE_WEIGHTS = {
    "planning_completeness": 0.30,
    "tool_reasonableness": 0.40,
    "code_command_correctness": 0.30,
}

BENCHMARK_WITH_SKILLS_SUFFIX = "_withskills"

BENCHMARK_NO_SKILLS_SUFFIX = "_noskills"

_BENCHMARK_SKILLS_CACHE: list[dict[str, Any]] | None = None

_BENCHMARK_SCORER_CLIENT_CACHE: dict[str, Any] = {}

_BENCHMARK_SCORER_CLIENT_CACHE_LOCK = threading.Lock()

def _slugify_filename_part(text: str, fallback: str, max_len: int = 80) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        raw = fallback
    raw = raw.replace("/", "_").replace("\\", "_")
    raw = re.sub(r"[^a-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("._-")
    if not raw:
        raw = fallback
    return raw[:max_len].rstrip("._-") or fallback

def _extract_case_id(task_name: str, fallback_source: str = "") -> str:
    text = str(task_name or "").strip()
    fallback_text = str(fallback_source or "").strip()
    if not text and fallback_text:
        text = fallback_text
    if not text:
        return "unknown"

    patterns = [
        r"\bcase[_\s-]*id\s*[:=]\s*([a-zA-Z0-9._-]+)",
        r"\bbenchmark\s*test\s*case\s*[:#-]?\s*([a-zA-Z0-9._-]+)",
        r"\btest\s*case\s*[:#-]?\s*([a-zA-Z0-9._-]+)",
        r"\bcase\s*[:#-]?\s*([a-zA-Z0-9._-]+)",
        r"\bT(\d{1,3})(?=\b|[_-])",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            token = m.group(1)
            if token:
                return _slugify_filename_part(token, "unknown")

    if fallback_text and fallback_text != text:
        for p in patterns:
            m = re.search(p, fallback_text, flags=re.IGNORECASE)
            if m:
                token = m.group(1)
                if token:
                    return _slugify_filename_part(token, "unknown")

    return "unknown"

def _benchmark_report_stem(task_name: str, model_name: str, fallback_source: str = "") -> str:
    case_id = _extract_case_id(task_name, fallback_source=fallback_source)
    model_part = _slugify_filename_part(model_name, "model_unknown")
    return f"{case_id}_{model_part}"

def _benchmark_report_filename(
    task_name: str,
    model_name: str,
    run_index: int | None = None,
    fallback_source: str = "",
) -> str:
    stem = _benchmark_report_stem(task_name, model_name, fallback_source=fallback_source)
    if isinstance(run_index, int) and run_index > 0:
        return f"{stem}_run{run_index}"
    return stem

def _build_skill_hint_summary(skills: list[dict[str, Any]], max_skills: int = 24) -> str:
    """Build short skill introduction phrases for benchmark model and scorer prompts."""
    if not skills:
        return ""

    lines = [
        "[Skill Hints]",
        "Use the following short descriptions as valid capability hints for this benchmark run.",
        "Only count a skill as used when the task solution actually relies on it.",
    ]

    for skill in skills[:max_skills]:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        summary = str(skill.get("summary_en", "")).strip() or str(skill.get("description", "")).strip()
        summary = re.sub(r"\s+", " ", summary).strip()
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "..."
        if not summary:
            summary = f"{name} handles a specialized NeuroRuntime workflow."
        lines.append(f"- {name}: {summary}")

    return "\n".join(lines)

def _build_compact_skill_hint_summary(skills: list[dict[str, Any]]) -> str:
    """Build a single-line skill hint for routes that need minimal prompt overhead."""
    if not skills:
        return ""
    skill = skills[0]
    name = str(skill.get("name", "")).strip()
    if not name:
        return ""
    summary = str(skill.get("summary_en", "")).strip() or str(skill.get("description", "")).strip()
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > 90:
        summary = summary[:87].rstrip() + "..."
    if not summary:
        summary = "Use only if it clearly improves the task solution."
    return f"[Primary Skill Hint] {name}: {summary}"

def _benchmark_scoring_expectation_for_model(model_name: str) -> str:
    """Return variant-aware scoring expectation text."""
    normalized = str(model_name or "").strip().lower()
    if normalized.endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
        return (
            "with-skills expectation: treat cited skills as valid when they match the provided skill hints and the task intent. "
            "Do not require a literal runnable skill entry point just to credit skill usage. "
            "Judge whether the chosen skill is reasonable for the task, whether the plan appears grounded in the skill's described capability, "
            "and whether the final code/commands are adapted to the benchmark task's required inputs and outputs rather than copied verbatim from generic skill text. "
            "The model may internally compare task-required inputs/outputs against a skill's default example inputs/outputs before deciding whether and how to adapt the skill, but that intermediate deliberation does not need to appear in the final report. "
            "Give positive credit when the report shows evidence that the model inspected local skill content and then rewrote the solution around the benchmark task contract. "
            "Also give positive credit when the final solution is clearly skill-informed, meaning it borrows methodology, sequencing, validation ideas, parameter choices, file conventions, or implementation lessons from a relevant skill even if it does not present itself as a full adopted skill workflow. "
            "If no local skill file was actually inspected, skill hints should only refine the task's canonical solution family rather than replacing it with a different task framing, different starting data representation, different main toolchain, or a broader generic workflow. "
            "Do not penalize a with-skills answer merely because it customizes or replaces the skill's default example I/O with task-specific filenames, paths, inputs, or outputs. "
            "Do not penalize a with-skills answer merely because the model likely reasoned about whether to use a skill before choosing the final workflow; judge the final task-directed answer rather than requiring the intermediate comparison to be shown. "
            "Give extra credit when the report explicitly identifies missing task-critical inputs such as atlas/parcellation files, label tables, registration/alignment prerequisites, or other required artifacts, and then asks the user to choose the key analysis options or hyperparameters that materially affect the result. "
            "Penalize hallucinated capabilities only when the claimed skill use clearly contradicts the provided skill hints or task contract. "
            "Penalize reports when the final commands still follow the skill's default interface even though the benchmark task contract is more specific or clearly different. "
            "Also penalize reports when, without inspecting local skill content, the final answer changes the task into a different canonical family than the benchmark asked for, such as switching from tensor-derived metrics to raw-DWI tensor fitting, from HCP Pipelines to a generic fMRIPrep substitute, or from reuse of existing preprocessing transforms to de novo registration without task evidence. "
            "If required inputs are missing, reward answers that say 'Missing required input' but still provide the most task-directed executable fallback they can; penalize answers that retreat into generic orchestration placeholders, wrapper commands, or high-level skill chaining that are less concrete than a no-skills baseline. "
            "Also penalize answers that create an extra orchestrator layer around a skill, such as subprocess wrappers, temporary intermediate artifacts, or shelling out to a skill CLI, when the task can be solved more directly with task-level code or commands."
        )
    if normalized.endswith(BENCHMARK_NO_SKILLS_SUFFIX):
        return (
            "no-skills baseline expectation: do not penalize for not using skills. "
            "Judge only planning quality, tool/code reasonableness, and execution correctness under no-skill constraints."
        )
    return (
        "standard expectation: judge planning quality, tool/code reasonableness, and execution correctness based on the task contract."
    )

def _discover_benchmark_task_files(benchmark_root: Path) -> list[Path]:
    task_files = [p for p in benchmark_root.rglob("task.md") if p.is_file()]
    return sorted(task_files, key=lambda p: str(p.parent.relative_to(benchmark_root)).lower())

def _discover_task_contracts(benchmark_root: Path) -> dict[str, Path]:
    tasks: dict[str, Path] = {}
    for task_file in _discover_benchmark_task_files(benchmark_root):
        task_text = ""
        try:
            task_text = task_file.read_text(encoding="utf-8")
        except Exception:
            task_text = ""
        # The NeuroBench layout names each directory ``T<number>_<slug>``. That
        # number is the authoritative case id, but ``_extract_case_id`` prefers
        # prose patterns, so a task body containing the words "case is" or
        # "test case" collides with unrelated tasks and the 500-task set collapses
        # to about 150 ids. The directory number is read first; the earlier
        # text-based extraction remains the fallback for any other layout.
        directory_match = re.match(r"^T(\d{1,3})", task_file.parent.name, flags=re.IGNORECASE)
        if directory_match:
            case_id = str(int(directory_match.group(1)))
        else:
            case_id = _extract_case_id(task_text, fallback_source=task_file.parent.name)
        tasks[case_id] = task_file
    return tasks

def _parse_benchmark_report_filename(path: Path) -> tuple[str, str, int] | None:
    stem = path.stem
    if not re.match(r"^\d+_.+_run\d+$", stem, flags=re.IGNORECASE):
        return None
    if "_" not in stem:
        return None
    run_index = 1
    run_match = re.search(r"_run(\d+)$", stem, flags=re.IGNORECASE)
    if run_match:
        try:
            run_index = int(run_match.group(1))
        except Exception:
            run_index = 1
        stem = stem[: run_match.start()]
    case_id, model_name = stem.split("_", 1)
    case_id = _slugify_filename_part(case_id, "unknown")
    model_name = _slugify_filename_part(model_name, "model_unknown", max_len=120)
    if not case_id or not model_name:
        return None
    return case_id, model_name, max(1, run_index)

def _discover_benchmark_reports(output_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    reports: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for file_path in sorted(output_dir.rglob("*.md")):
        stem_lower = file_path.stem.lower()
        if stem_lower.startswith("benchmark_leaderboard_") or stem_lower.startswith("benchmark_scores_"):
            continue
        parsed = _parse_benchmark_report_filename(file_path)
        if not parsed:
            continue
        case_id, model_name, run_index = parsed
        reports.setdefault(model_name, {}).setdefault(case_id, []).append(
            {
                "run_index": run_index,
                "report_file": file_path,
            }
        )

    for case_map in reports.values():
        for run_entries in case_map.values():
            run_entries.sort(key=lambda item: (int(item.get("run_index", 1)), str(item.get("report_file", ""))))
    return reports

def _mean_and_variance(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return round(mean, 4), round(variance, 4)

def _model_uses_gemini_compat(model_name: str) -> bool:
    """Detect Gemini models for provider-specific compatibility handling."""
    name = str(model_name or "").strip().lower()
    return "gemini" in name

def _model_uses_grok_compat(model_name: str) -> bool:
    """Detect Grok models for benchmark-specific prompt shaping."""
    name = str(model_name or "").strip().lower()
    return "grok" in name

def _model_is_withskills_variant(model_name: str) -> bool:
    """Return True when the benchmark model name is the with-skills variant."""
    name = _slugify_filename_part(model_name, "model_unknown", max_len=120)
    return name.endswith(BENCHMARK_WITH_SKILLS_SUFFIX)

def _split_skill_condition(model_name: str) -> tuple[str, str]:
    name = _slugify_filename_part(model_name, "model_unknown", max_len=120)
    if name.endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
        return name[: -len(BENCHMARK_WITH_SKILLS_SUFFIX)], "with_skills"
    if name.endswith(BENCHMARK_NO_SKILLS_SUFFIX):
        return name[: -len(BENCHMARK_NO_SKILLS_SUFFIX)], "no_skills"
    return name, "unknown"

def _compute_normalized_gain(with_skills_score: float, no_skills_score: float) -> float:
    delta = float(with_skills_score) - float(no_skills_score)
    if delta >= 0:
        denom = max(1e-6, 100.0 - float(no_skills_score))
    else:
        denom = max(1e-6, float(no_skills_score))
    gain = delta / denom
    if gain > 1.0:
        gain = 1.0
    if gain < -1.0:
        gain = -1.0
    return round(gain, 4)

def _interpret_normalized_gain(avg_abs_improvement: float, avg_gain: float) -> str:
    if avg_gain >= 0.5 and avg_abs_improvement < 5.0:
        return (
            "High normalized gain with low absolute improvement suggests ceiling effects; "
            "proportional benefit exists but raw improvement is limited."
        )
    if avg_gain >= 0.5 and avg_abs_improvement >= 5.0:
        return (
            "High normalized gain with high absolute improvement suggests substantial "
            "scaffolding benefit."
        )
    if avg_gain > 0:
        return (
            "Positive normalized gain indicates proportional benefit from skills; "
            "consistency refers to similar proportion, not identical absolute improvement."
        )
    if avg_gain == 0:
        return "No measurable proportional gain between with-skills and no-skills conditions."
    return "Negative gain indicates skills condition underperformed the no-skills baseline."

def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def _clamp_score_1_10(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 1.0
    if num < 1:
        return 1.0
    if num > 10:
        return 10.0
    return round(num, 2)

def _extract_token_usage_from_response(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _read_int(field: str) -> int:
        value = 0
        try:
            if isinstance(usage, dict):
                value = int(usage.get(field, 0) or 0)
            else:
                value = int(getattr(usage, field, 0) or 0)
        except Exception:
            value = 0
        return max(0, value)

    return {
        "prompt_tokens": _read_int("prompt_tokens"),
        "completion_tokens": _read_int("completion_tokens"),
        "total_tokens": _read_int("total_tokens"),
    }

def _classify_shell_command_tool_kinds(command: str) -> set[str]:
    cmd = str(command or "")
    kinds: set[str] = {"shell"}
    if re.search(r"\bpython(\d+(\.\d+)*)?\b", cmd, flags=re.IGNORECASE):
        kinds.add("python")
    if re.search(r"\b(docker|podman)\b", cmd, flags=re.IGNORECASE):
        kinds.add("docker")
    if re.search(
        r"\b(fsl|fslmaths|fslstats|bet|flirt|fnirt|eddy|topup|fast|melodic|feat)\b",
        cmd,
        flags=re.IGNORECASE,
    ):
        kinds.add("fsl")
    if re.search(r"\b(freesurfer|recon-all|mri_\w+)\b", cmd, flags=re.IGNORECASE):
        kinds.add("freesurfer")
    if re.search(r"\b(dcm2niix|dcm2nii)\b", cmd, flags=re.IGNORECASE):
        kinds.add("dcm2niix")
    return kinds

def _normalize_skill_token(text: str) -> str:
    """Normalize skill tokens for robust command/text matching."""
    token = str(text or "").lower()
    return re.sub(r"[^a-z0-9]+", "", token)

def _infer_skills_from_text(text: str, skills: list[dict[str, Any]]) -> list[str]:
    """Infer referenced skills from command or answer text using skill and folder names."""
    norm_text = _normalize_skill_token(text)
    if not norm_text or not skills:
        return []

    inferred: list[str] = []
    for skill in skills:
        skill_name = str(skill.get("name", "")).strip()
        if not skill_name:
            continue

        dir_name = ""
        try:
            dir_name = Path(skill.get("path")).name
        except Exception:
            dir_name = ""

        for candidate in (skill_name, dir_name):
            norm_candidate = _normalize_skill_token(candidate)
            if norm_candidate and norm_candidate in norm_text:
                inferred.append(skill_name)
                break

    seen: set[str] = set()
    unique_names: list[str] = []
    for name in inferred:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    return unique_names

def _extract_skill_contract_candidates(skill_md_path: Any) -> list[str]:
    """Read static contract-related skill references from SKILL.md.

    This is only used as a candidate source for expansion. Actual counting still
    requires relevance support from the current task plan/commands.
    """
    try:
        skill_md = Path(skill_md_path)
    except Exception:
        return []
    if not skill_md.exists():
        return []

    try:
        text = skill_md.read_text(encoding="utf-8")
    except Exception:
        return []

    candidates: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", text):
        token = str(match.group(1) or "").strip()
        key = _normalize_skill_token(token)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(token)
    return candidates

def _build_skill_context_text(skill: dict[str, Any]) -> str:
    parts = [
        str(skill.get("name", "")).strip(),
        str(skill.get("summary_en", "")).strip(),
        str(skill.get("description", "")).strip(),
    ]
    return " ".join(part for part in parts if part).strip()

def _extract_specialized_skill_signals(skill: dict[str, Any]) -> list[str]:
    """Extract high-specificity execution signals for a skill from its metadata/docs."""
    context = _build_skill_context_text(skill)
    signals: list[str] = []
    seen: set[str] = set()

    for token in re.findall(r"[A-Za-z][A-Za-z0-9._/-]{3,}", context):
        raw = str(token or "").strip()
        norm = _normalize_skill_token(raw)
        if len(norm) < 5 or norm in seen:
            continue
        if not any(ch.isdigit() for ch in raw) and raw.lower() == raw and "-" not in raw and "_" not in raw and "/" not in raw and "." not in raw:
            # Skip broad plain words like 'install', 'workflow', 'execution'.
            continue
        seen.add(norm)
        signals.append(raw)

    return signals

def _expand_skill_contract_usage(
    used_skills: list[str],
    skills: list[dict[str, Any]],
    assistant_response: str,
    tool_events: list[dict[str, Any]],
) -> list[str]:
    """Expand counted skills using skill contracts, filtered by task-plan relevance.

    A contract candidate is counted only when:
    - it is referenced by a top-level used skill's SKILL.md, and
    - the current plan/commands mention evidence aligned with that candidate's
      capability hint or explicit skill name.
    """
    if not used_skills or not skills:
        return []

    skill_by_key: dict[str, dict[str, Any]] = {}
    for skill in skills:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        key = _normalize_skill_token(name)
        if key:
            skill_by_key[key] = skill
        try:
            dir_name = Path(skill.get("path")).name
        except Exception:
            dir_name = ""
        dir_key = _normalize_skill_token(dir_name)
        if dir_key and dir_key not in skill_by_key:
            skill_by_key[dir_key] = skill

    plan_parts: list[str] = []
    thinking_body, commands_body = _split_benchmark_answer_sections(assistant_response)
    if thinking_body:
        plan_parts.append(thinking_body)
    if commands_body:
        plan_parts.append(commands_body)
    for event in tool_events:
        command = str(event.get("command", "")).strip()
        if command:
            plan_parts.append(command)
    plan_text = "\n".join(part for part in plan_parts if part).strip()
    norm_plan_text = _normalize_skill_token(plan_text)
    if not norm_plan_text:
        return []

    expanded: list[str] = []
    expanded_seen: set[str] = set()
    already_used = {_normalize_skill_token(name) for name in used_skills}

    for used_name in used_skills:
        used_key = _normalize_skill_token(used_name)
        used_skill = skill_by_key.get(used_key)
        if not used_skill:
            continue

        for candidate_name in _extract_skill_contract_candidates(used_skill.get("skill_md")):
            candidate_key = _normalize_skill_token(candidate_name)
            candidate_skill = skill_by_key.get(candidate_key)
            if not candidate_key or not candidate_skill:
                continue
            canonical_name = str(candidate_skill.get("name", "")).strip()
            canonical_key = _normalize_skill_token(canonical_name)
            if not canonical_key or canonical_key in already_used or canonical_key in expanded_seen:
                continue

            candidate_signal = False
            if candidate_key in norm_plan_text:
                candidate_signal = True
            else:
                for signal in _extract_specialized_skill_signals(candidate_skill):
                    norm_signal = _normalize_skill_token(signal)
                    if norm_signal and norm_signal in norm_plan_text:
                        candidate_signal = True
                        break

            if candidate_signal:
                expanded_seen.add(canonical_key)
                expanded.append(canonical_name)

    return expanded

def _summarize_skill_usage(
    tool_events: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    assistant_response: str = "",
) -> dict[str, Any]:
    """Summarize unique skill usage from explicit runtime signals only.

    Benchmark reports should distinguish between skills that actually influenced
    execution and skills that were only mentioned during contract audit or plan
    narration. Counting raw answer-text mentions as usage creates false positives
    in with-skills runs where the model discusses a skill but ultimately solves
    the task directly without adopting it.
    """
    seen: set[str] = set()
    used_skills: list[str] = []
    read_seen: set[str] = set()
    read_skills: list[str] = []
    catalog: dict[str, str] = {}

    for skill in skills:
        skill_name = str(skill.get("name", "")).strip()
        if not skill_name:
            continue
        catalog[_normalize_skill_token(skill_name)] = skill_name
        try:
            dir_name = Path(skill.get("path")).name
        except Exception:
            dir_name = ""
        if dir_name:
            catalog.setdefault(_normalize_skill_token(dir_name), skill_name)

    def record(names: list[str]) -> None:
        for name in names:
            key = _normalize_skill_token(str(name).strip())
            canonical = catalog.get(key)
            if not key or not canonical or key in seen:
                continue
            seen.add(key)
            used_skills.append(canonical)

    def record_reads(names: list[str]) -> None:
        for name in names:
            key = _normalize_skill_token(str(name).strip())
            canonical = catalog.get(key)
            if not key or not canonical or key in read_seen:
                continue
            read_seen.add(key)
            read_skills.append(canonical)

    for event in tool_events:
        tool = str(event.get("tool", "")).strip()
        explicit = event.get("skills_used")
        if isinstance(explicit, list):
            record([str(name).strip() for name in explicit if str(name).strip()])

        command = str(event.get("command", "")).strip()
        if command:
            record(_infer_skills_from_text(command, skills))
            if tool == "read_workspace_file":
                record_reads(_infer_skills_from_text(command, skills))

        payload_skills = _extract_skills_from_result_payload(event.get("result"))
        if payload_skills:
            record(payload_skills)
            if tool == "read_workspace_file":
                record_reads(payload_skills)

    # Assistant-text mentions are not runtime evidence. In benchmark reports we
    # only count explicit tool/runtime signals as skill usage, otherwise direct
    # answers that merely cite or imitate a skill inflate usage statistics.
    response_text = str(assistant_response or "")
    normalized_response = response_text.lower()
    contract_not_adopted = (
        "contract decision: not adopted" in normalized_response
        or "contract decision: skill not adopted" in normalized_response
        or "skill not adopted" in normalized_response
    )

    # Only expand from already-confirmed runtime usage. This preserves canonical
    # aliases without treating answer-text narration as evidence of actual use.
    record(_expand_skill_contract_usage(used_skills, skills, "", tool_events))

    return {
        "total_unique_skills": len(used_skills),
        "skills": used_skills,
        "read_unique_skills": len(read_skills),
        "read_skills": read_skills,
    }

def _extract_skills_from_result_payload(payload: Any) -> list[str]:
    """Collect explicitly reported skill usage from nested handler results.

    Skills may invoke other lower-level skills for dependency installation or other
    atomic work. Handlers must surface that chain through explicit skill-oriented
    fields such as skills_used / used_skills / invoked_skills; this extractor
    flattens those runtime traces only.
    """
    seen: set[str] = set()
    collected: list[str] = []
    candidate_keys = {
        "skill",
        "skill_name",
        "skills",
        "skills_used",
        "used_skills",
        "invoked_skills",
        "called_skills",
        "dependent_skills",
    }

    def record(value: Any) -> None:
        name = str(value or "").strip()
        if not name:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        collected.append(name)

    def walk(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key or "").strip().lower()
                if key_text in candidate_keys:
                    if isinstance(item, list):
                        for entry in item:
                            if isinstance(entry, (dict, list)):
                                walk(entry, key_text)
                            else:
                                record(entry)
                        continue
                    if isinstance(item, dict):
                        nested_name = item.get("name") or item.get("skill") or item.get("skill_name")
                        if nested_name:
                            record(nested_name)
                        walk(item, key_text)
                        continue
                    record(item)
                    continue
                walk(item, key_text)
            return

        if isinstance(value, list):
            for item in value:
                walk(item, parent_key)
            return

        if parent_key in candidate_keys:
            record(value)

    walk(payload)
    return collected

def _extract_usage_count_from_report(report_text: str) -> int | None:
    text = str(report_text or "")
    m = re.search(r"-\s*Skills used \(unique\):\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        try:
            return max(0, int(m.group(1)))
        except Exception:
            return None

    count = len(re.findall(r"^\d+\.\s+\[[^\]]+\]", text, flags=re.MULTILINE))
    return count if count > 0 else None

def _extract_elapsed_seconds_from_report(report_text: str) -> float | None:
    text = str(report_text or "")
    m = re.search(r"-\s*Elapsed seconds:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return max(0.0, float(m.group(1)))
    except Exception:
        return None

def _extract_token_usage_from_report_text(report_text: str) -> dict[str, int]:
    text = str(report_text or "")
    m = re.search(
        r"-\s*Token usage:\s*prompt=(\d+),\s*completion=(\d+),\s*total=(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        return {
            "prompt_tokens": max(0, int(m.group(1))),
            "completion_tokens": max(0, int(m.group(2))),
            "total_tokens": max(0, int(m.group(3))),
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _build_skill_catalog_summary(skills: list[dict[str, Any]], max_skills: int = 80) -> str:
    """Build a concise skill catalog for benchmark evaluator/model context."""
    if not skills:
        return ""

    lines = [
        "[Available Skill Library]",
        "The following NeuroRuntime skills are available in this benchmark run.",
        "Treat these skills as valid in-repo capabilities when judging feasibility or completeness.",
    ]
    for skill in skills[:max_skills]:
        name = str(skill.get("name", "")).strip()
        if not name:
            continue
        summary = str(skill.get("summary_en", "")).strip() or str(skill.get("description", "")).strip()
        if not summary:
            summary = f"{name} provides specialized workflow support."
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines)

def _select_benchmark_skill_hints(skills: list[dict[str, Any]], model_name: str = "") -> list[dict[str, Any]]:
    """Choose a smaller skill-hint set for models with weak tool-call reliability."""
    if not skills:
        return []
    if _model_uses_grok_compat(model_name):
        return skills[:1]
    return skills

def _format_duration(seconds: float | int | None) -> str:
    """Format duration as H:MM:SS (or M:SS for short spans)."""
    if seconds is None:
        return "-"
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return "-"

    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

def _split_benchmark_answer_sections(answer_text: str) -> tuple[str, str]:
    """Extract Solution Thinking / Commands Or Code bodies from benchmark answer text.

    This prevents duplicated section headers in saved benchmark reports when the model
    already follows the required two-section output format.
    """
    text = str(answer_text or "").strip()
    if not text:
        return "(empty)", ""

    m_thinking = re.search(r"^##\s+Solution\s+Thinking\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    m_commands = re.search(r"^##\s+Commands\s+Or\s+Code\s*$", text, flags=re.IGNORECASE | re.MULTILINE)

    if not m_thinking or not m_commands:
        return text, ""

    if m_thinking.start() > m_commands.start():
        # Unexpected order; keep raw text as thinking and leave commands empty.
        return text, ""

    thinking = text[m_thinking.end(): m_commands.start()].strip()
    commands = text[m_commands.end():].strip()
    return (thinking or "(empty)"), commands

def _is_incomplete_benchmark_response(answer_text: str) -> bool:
    """Return True when a benchmark answer stops at tool traces or inspection intent."""
    text = str(answer_text or "").strip()
    if not text:
        return True

    lower = text.lower()
    if "<tool_call>" in lower or "<tool_result>" in lower:
        return True

    thinking_body, commands_body = _split_benchmark_answer_sections(text)
    commands_lower = str(commands_body or "").strip().lower()

    if not commands_lower or commands_lower == "1. no tool command was used.":
        return True

    inspect_markers = (
        "let me inspect",
        "i'll inspect",
        "i will inspect",
        "need to inspect",
        "inspect the ",
    )
    if any(marker in lower for marker in inspect_markers):
        meaningful_commands = any(
            token in commands_lower
            for token in ("```", "conda ", "python ", "bash", "#!/", "docker ", "export ")
        )
        if not meaningful_commands:
            return True

    return thinking_body.strip() == "(empty)" and not commands_lower

def _tool_events_show_inspection_only(tool_events: list[dict[str, Any]]) -> bool:
    """Return True when the trace only inspects skill files without task execution."""
    if not tool_events:
        return False

    saw_skill_read = False
    for event in tool_events:
        tool = str(event.get("tool") or "").strip()
        command = str(event.get("command") or "").strip().lower()
        executed = bool(event.get("executed", False))

        if tool == "read_workspace_file":
            if "skills/" in command or command.endswith("skill.md"):
                saw_skill_read = True
            continue

        if tool == "run_shell_command":
            # In benchmark mode, suggested-only shell commands are inspection intent, not execution.
            if not executed:
                continue
            return False

        if executed:
            return False

    return saw_skill_read

def _score_single_benchmark_case(
    llm_client: Any,
    task_text: str,
    report_text: str,
    model_name: str,
    scorer_model: str,
    reference_examples: list[dict[str, Any]] | None = None,
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from core.agent.main import _retry_api_call

    scoring_system = (
        "You are a strict benchmark evaluator for neuroscience-agent outputs. "
        "Score only from the provided task and report. "
        "Return JSON only."
    )
    reference_examples = reference_examples or []
    skills = skills or []
    reference_section = ""
    skill_hint_section = ""
    if str(model_name or "").strip().lower().endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
        skill_hint_section = _build_skill_hint_summary(skills)
    if reference_examples:
        lines: list[str] = [
            "Reference scored examples for calibration (use them to keep scoring criteria consistent):",
        ]
        for idx, ref in enumerate(reference_examples, start=1):
            ref_model = str(ref.get("model_name", ""))
            ref_run = max(1, int(ref.get("run_index", 1) or 1))
            ref_report = str(ref.get("report_text", ""))
            ref_report_preview = ref_report[:1200] if len(ref_report) > 1200 else ref_report
            lines.extend(
                [
                    f"[{idx}] model={ref_model} run={ref_run}",
                    "Scored dimensions:",
                    f"- planning_completeness: {ref.get('planning_completeness')}",
                    f"- action_reasonableness: {ref.get('tool_reasonableness')}",
                    f"- code_command_correctness: {ref.get('code_command_correctness')}",
                    f"- brief_justification: {ref.get('brief_justification', '')}",
                    "Report log (truncated if long):",
                    ref_report_preview,
                    "",
                ]
            )
        reference_section = "\n".join(lines)

    expectation_hint = _benchmark_scoring_expectation_for_model(model_name)
    scoring_user = (
        "Evaluate the benchmark report on a 1-10 scale for three dimensions:\n"
        "1) planning_completeness\n"
        "2) action_reasonableness\n"
        "3) code_command_correctness\n"
        "Do not score usage_efficiency here; it is derived separately from skill usage counts across models for the same case.\n"
        "Important: action selection quality is still very important.\n"
        "For with-skills reports, do not require the final answer to expose the model's internal skill-vs-task comparison or any explicit contract-audit lines.\n"
        "For with-skills reports, reward solutions that inspect or rely on local skill content and then adapt that material to the benchmark task's required inputs, outputs, filenames, and validation rules.\n"
        "For with-skills reports, also reward skill-informed answers that clearly borrow relevant methodology, sequencing, validation logic, parameterization, or implementation lessons from a hinted skill even when the final answer does not present a full explicit skill workflow.\n"
        "Do NOT reduce scores merely because a with-skills report reads or inspects a relevant local skill before adapting it to the task; that behavior is positive evidence when the final workflow remains task-directed.\n"
        "Do NOT reduce scores merely because the model appears to have thought about whether to use a skill, compared task-required I/O against skill-default I/O, or revised away an intermediate skill-based plan before presenting the final answer.\n"
        "If no local skill file was inspected, do NOT reward skill hints being used to replace the benchmark's canonical task family, starting representation, or main toolchain; in that situation, skill hints should only sharpen the direct task solution.\n"
        "Do not treat task-specific rewrites of a skill's default example I/O as a flaw by themselves.\n"
        "When required artifacts or decisions are missing, give higher scores to reports that explicitly ask the user for task-critical inputs such as atlas/parcellation files, label tables, alignment/registration choices, or other result-shaping hyperparameters, especially when those choices materially change the downstream analysis.\n"
        "Also give higher scores when the report makes the reuse-versus-recompute decision explicit: if required derived maps or intermediates already exist and satisfy the task contract, reusing them should be rewarded; if they are missing, the report should justify deriving them from the nearest skill-compatible or pipeline-compatible path instead of silently assuming either branch.\n"
        "For tasks involving atlas/parcellation summaries or derived imaging maps, reward reports that explicitly validate spatial consistency, state atlas/label/LUT provenance, and distinguish outputs summarized from pre-existing maps versus outputs that depend on newly generated maps in the current workflow.\n"
        "For with-skills reports, reduce action_reasonableness and code_command_correctness only when the final commands or code still keep the skill's default interface instead of the task-required interface.\n"
        "For with-skills reports, also reduce action_reasonableness and code_command_correctness when the answer drifts into a different canonical workflow than the task requested, especially when no local skill file was inspected and the drift appears to come from over-generalizing the skill hint.\n"
        "If the report says 'Missing required input', do not reward generic skill orchestration alone; prefer the answer that still gives the most task-specific executable fallback, concrete file expectations, and validation logic under the missing-input constraint.\n"
        "Do NOT penalize solutions purely for being longer when the added content improves coverage, safety, validation, or clarifies key assumptions/choices.\n"
        "Prefer solutions that comprehensively consider the problem (inputs, outputs, edge cases, validation, reproducibility, failure modes, and clinically/analytically meaningful QC) even if the plan/code is not the shortest possible.\n"
        "Do NOT reduce scores merely because the report includes environment setup or dependency installation steps. Only reduce scores if those steps are clearly irrelevant, replace required task execution, or crowd out task-specific commands/validation.\n"
        "Do NOT reduce scores merely because the report includes a small extra validation script or post-run verification block. Reward it when it concretely checks the benchmark contract, output artifacts, schema, numerical integrity, or reproducibility assumptions.\n"
        "Do NOT reduce scores merely because the report includes a thin task-specific adapter layer that rewrites paths, normalizes fields, enforces the benchmark export schema, or preserves degraded partial results around a skill or native tool; reward that when it improves contract compliance.\n"
        "If the report adds wrappers/orchestration, only reduce action_reasonableness when those wrappers are clearly redundant, create a second workflow or broad generic scaffolding, AND do not provide additional validation, traceability, contract compliance, or necessary interface adaptation.\n"
        "Output strict JSON with keys exactly:\n"
        "planning_completeness, tool_reasonableness, code_command_correctness, "
        "brief_justification.\n"
        "Variant-aware expectation:\n"
        f"- {expectation_hint}\n"
        "\n"
        + (f"{skill_hint_section}\n\n" if skill_hint_section else "")
        + f"Task contract:\n{task_text}\n\n"
        + (f"{reference_section}\n\n" if reference_section else "")
        + f"Target model: {model_name}\n"
        + f"Model report:\n{report_text}\n"
    )
    resp = _retry_api_call(
        "Benchmark scoring request",
        lambda: llm_client.chat.completions.create(
            model=scorer_model,
            messages=[
                {"role": "system", "content": scoring_system},
                {"role": "user", "content": scoring_user},
            ],
        ),
        retries=10,
    )
    content = ""
    try:
        content = str(resp.choices[0].message.content or "")
    except Exception:
        content = ""

    parsed = _extract_json_object(content) or {}
    planning = _clamp_score_1_10(parsed.get("planning_completeness", 1))
    tool = _clamp_score_1_10(parsed.get("tool_reasonableness", 1))
    code = _clamp_score_1_10(parsed.get("code_command_correctness", 1))
    return {
        "planning_completeness": planning,
        "tool_reasonableness": tool,
        "code_command_correctness": code,
        "usage_count": _extract_usage_count_from_report(report_text),
        "brief_justification": str(parsed.get("brief_justification", "")).strip(),
        "raw_model_output": content,
    }

def _get_benchmark_scorer_client(scorer_model: str = BENCHMARK_SCORER_MODEL) -> Any:
    from core.agent.main import load_environment, build_llm_client

    global _BENCHMARK_SCORER_CLIENT_CACHE
    resolved_scorer_model = str(scorer_model or "").strip() or BENCHMARK_SCORER_MODEL
    with _BENCHMARK_SCORER_CLIENT_CACHE_LOCK:
        cached = _BENCHMARK_SCORER_CLIENT_CACHE.get(resolved_scorer_model)
        if cached is not None:
            return cached

        scorer_env = load_environment()
        llm_cfg = scorer_env.setdefault("llm_backend", {})
        llm_cfg["provider"] = "openai"
        llm_cfg["model"] = resolved_scorer_model
        _BENCHMARK_SCORER_CLIENT_CACHE[resolved_scorer_model] = build_llm_client(scorer_env)
        return _BENCHMARK_SCORER_CLIENT_CACHE[resolved_scorer_model]

def _score_batch_benchmark_case(
    llm_client: Any,
    task_text: str,
    runs_by_model: dict[str, list[dict[str, Any]]],
    scorer_model: str,
    case_id: str = "",
    skills: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    from core.agent.main import _retry_api_call

    """Score multiple model-runs for the same task in a single LLM call."""
    skills = skills or []
    run_blocks: list[str] = []
    expectation_lines: list[str] = []
    seen_models: set[str] = set()
    include_skill_hints = False
    for model_name in sorted(runs_by_model.keys()):
        if model_name not in seen_models:
            seen_models.add(model_name)
            expectation_lines.append(
                f"- {model_name}: {_benchmark_scoring_expectation_for_model(model_name)}"
            )
            if str(model_name or "").strip().lower().endswith(BENCHMARK_WITH_SKILLS_SUFFIX):
                include_skill_hints = True
        for run_data in runs_by_model[model_name]:
            run_index = int(run_data.get("run_index", 1) or 1)
            report_text = str(run_data.get("report_text", ""))
            run_blocks.append(
                f"=== MODEL: {model_name} | RUN: {run_index} ===\n"
                f"{report_text}\n"
                "=== END REPORT ==="
            )
    skill_hint_section = _build_skill_hint_summary(skills) if include_skill_hints else ""

    scoring_system = (
        "You are a strict benchmark evaluator for neuroscience-agent outputs. "
        "You evaluate multiple model-runs for the SAME task to keep scoring criteria consistent. "
        "Return JSON only."
    )
    scoring_user = (
        "Evaluate EACH report on a 1-10 scale for three dimensions:\n"
        "1) planning_completeness\n"
        "2) action_reasonableness\n"
        "3) code_command_correctness\n"
        "Do not score usage_efficiency here; it is computed separately from skill usage counts.\n"
        "For with-skills reports, do not require the final answer to expose the model's internal skill-vs-task comparison or any explicit contract-audit lines.\n"
        "For with-skills reports, reward solutions that inspect or rely on local skill content and then adapt that material to the benchmark task's required inputs, outputs, filenames, and validation rules.\n"
        "For with-skills reports, also reward skill-informed answers that clearly borrow relevant methodology, sequencing, validation logic, parameterization, or implementation lessons from a hinted skill even when the final answer does not present a full explicit skill workflow.\n"
        "Do NOT reduce scores merely because a with-skills report reads or inspects a relevant local skill before adapting it to the task; that behavior is positive evidence when the final workflow remains task-directed.\n"
        "Do NOT reduce scores merely because the model appears to have thought about whether to use a skill, compared task-required I/O against skill-default I/O, or revised away an intermediate skill-based plan before presenting the final answer.\n"
        "If no local skill file was inspected, do NOT reward skill hints being used to replace the benchmark's canonical task family, starting representation, or main toolchain; in that situation, skill hints should only sharpen the direct task solution.\n"
        "Do not treat task-specific rewrites of a skill's default example I/O as a flaw by themselves.\n"
        "When required artifacts or decisions are missing, give higher scores to reports that explicitly ask the user for task-critical inputs such as atlas/parcellation files, label tables, alignment/registration choices, or other result-shaping hyperparameters, especially when those choices materially change the downstream analysis.\n"
        "Also give higher scores when the report makes the reuse-versus-recompute decision explicit: if required derived maps or intermediates already exist and satisfy the task contract, reusing them should be rewarded; if they are missing, the report should justify deriving them from the nearest skill-compatible or pipeline-compatible path instead of silently assuming either branch.\n"
        "For tasks involving atlas/parcellation summaries or derived imaging maps, reward reports that explicitly validate spatial consistency, state atlas/label/LUT provenance, and distinguish outputs summarized from pre-existing maps versus outputs that depend on newly generated maps in the current workflow.\n"
        "For with-skills reports, reduce action_reasonableness and code_command_correctness only when the final commands or code still keep the skill's default interface instead of the task-required interface.\n"
        "For with-skills reports, also reduce action_reasonableness and code_command_correctness when the answer drifts into a different canonical workflow than the task requested, especially when no local skill file was inspected and the drift appears to come from over-generalizing the skill hint.\n"
        "If a report says 'Missing required input', do not reward generic skill orchestration alone; prefer the report that still gives the most task-specific executable fallback, concrete file expectations, and validation logic under the missing-input constraint.\n"
        "Do NOT penalize solutions purely for being longer when the added content improves coverage, safety, validation, or clarifies key assumptions/choices.\n"
        "Prefer solutions that comprehensively consider the problem (inputs, outputs, edge cases, validation, reproducibility, failure modes, and clinically/analytically meaningful QC) even if the plan/code is not the shortest possible.\n"
        "Do NOT reduce scores merely because the report includes environment setup or dependency installation steps. Only reduce scores if those steps are clearly irrelevant, replace required task execution, or crowd out task-specific commands/validation.\n"
        "Do NOT reduce scores merely because the report includes a small extra validation script or post-run verification block. Reward it when it concretely checks the benchmark contract, output artifacts, schema, numerical integrity, or reproducibility assumptions.\n"
        "Do NOT reduce scores merely because the report includes a thin task-specific adapter layer that rewrites paths, normalizes fields, enforces the benchmark export schema, or preserves degraded partial results around a skill or native tool; reward that when it improves contract compliance.\n"
        "For with-skills reports, reduce action_reasonableness only when wrappers/orchestration around the skill are clearly redundant, create a second workflow or broad generic scaffolding, AND do not provide additional validation, traceability, contract compliance, or necessary interface adaptation.\n"
        "\n"
        "Return a strict JSON ARRAY. Each item must contain keys exactly:\n"
        "model, run_index, planning_completeness, tool_reasonableness, code_command_correctness, brief_justification\n"
        "\n"
        "Variant-aware expectations (apply per model):\n"
        + ("\n".join(expectation_lines) if expectation_lines else "- (none)")
        + "\n\n"
        + (f"{skill_hint_section}\n\n" if skill_hint_section else "")
        + f"Task contract:\n{task_text}\n\n"
        "Reports:\n"
        + "\n\n".join(run_blocks)
    )

    resp = _retry_api_call(
        "Benchmark batch scoring request",
        lambda: llm_client.chat.completions.create(
            model=scorer_model,
            messages=[
                {"role": "system", "content": scoring_system},
                {"role": "user", "content": scoring_user},
            ],
        ),
        retries=10,
    )
    content = ""
    try:
        content = str(resp.choices[0].message.content or "")
    except Exception:
        content = ""

    parsed_list: list[Any] = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            parsed_list = parsed
        elif isinstance(parsed, dict):
            parsed_list = [parsed]
    except Exception:
        parsed_list = []

    def _normalize_model_name_key(name: str) -> str:
        # Normalize model names to tolerate case/punctuation/style variations
        # from the scorer model output (e.g., "MiniMax-M2.7" vs "minimax-m2.7").
        return re.sub(r"[^a-z0-9]", "", str(name or "").strip().lower())

    results_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in runs_by_model.keys()}
    normalized_model_lookup: dict[str, str] = {}
    for canonical_model in runs_by_model.keys():
        normalized_model_lookup.setdefault(
            _normalize_model_name_key(canonical_model),
            canonical_model,
        )
    expected_pairs: set[tuple[str, int]] = set()
    for model_name, run_entries in runs_by_model.items():
        for run_data in run_entries:
            expected_pairs.add((model_name, max(1, int(run_data.get("run_index", 1) or 1))))

    for item in parsed_list:
        if not isinstance(item, dict):
            continue
        raw_model = str(item.get("model", "")).strip()
        model = raw_model
        if model not in runs_by_model:
            model = normalized_model_lookup.get(_normalize_model_name_key(raw_model), "")
        if model not in runs_by_model:
            continue
        run_index = max(1, int(item.get("run_index", 1) or 1))

        report_text = ""
        for run_data in runs_by_model[model]:
            if int(run_data.get("run_index", 1) or 1) == run_index:
                report_text = str(run_data.get("report_text", ""))
                break

        results_by_model[model].append(
            {
                "run_index": run_index,
                "planning_completeness": _clamp_score_1_10(item.get("planning_completeness", 1)),
                "tool_reasonableness": _clamp_score_1_10(item.get("tool_reasonableness", 1)),
                "code_command_correctness": _clamp_score_1_10(item.get("code_command_correctness", 1)),
                "usage_count": _extract_usage_count_from_report(report_text),
                "brief_justification": str(item.get("brief_justification", "")).strip(),
                "raw_model_output": content,
            }
        )

    scored_pairs: set[tuple[str, int]] = set()
    for model_name, run_entries in results_by_model.items():
        for run in run_entries:
            scored_pairs.add((model_name, max(1, int(run.get("run_index", 1) or 1))))

    missing_pairs = sorted(expected_pairs - scored_pairs)

    # If all model-runs failed in this case, retry batch scoring first.
    if missing_pairs and not scored_pairs:
        print(
            f"[batch scorer] all model-runs missing for case={case_id}; retrying batch scoring",
            flush=True,
        )
        retry_limit = 2
        for retry_idx in range(1, retry_limit + 1):
            rescored = _score_batch_benchmark_case(
                llm_client=llm_client,
                task_text=task_text,
                runs_by_model=runs_by_model,
                scorer_model=scorer_model,
                case_id=case_id,
            )
            rescored_pairs: set[tuple[str, int]] = set()
            for model_name, runs in rescored.items():
                for run in runs:
                    rescored_pairs.add((model_name, max(1, int(run.get("run_index", 1) or 1))))

            rescored_missing = sorted(expected_pairs - rescored_pairs)
            if rescored_pairs:
                scored = rescored
                scored_pairs = rescored_pairs
                missing_pairs = rescored_missing
                print(
                    f"[batch scorer] retry {retry_idx}/{retry_limit} recovered "
                    f"{len(scored_pairs)}/{len(expected_pairs)} model-runs for case={case_id}",
                    flush=True,
                )
                break
    if missing_pairs:
        preview = ", ".join(f"{name}#run{idx}" for name, idx in missing_pairs[:6])
        if len(missing_pairs) > 6:
            preview = f"{preview}, ..."
        print(
            "[batch scorer] warning: incomplete batch output "
            f"(missing {len(missing_pairs)}/{len(expected_pairs)}): {preview}",
            flush=True,
        )

    return results_by_model

def _score_benchmark_batch_job(job: dict[str, Any]) -> dict[str, Any]:
    """Score all model runs for one case in one batch call."""
    task_file = Path(str(job.get("task_file", "")))
    case_id = str(job.get("case_id", ""))
    runs_by_model_raw = job.get("runs_by_model", {})
    scorer_model = str(job.get("scorer_model", BENCHMARK_SCORER_MODEL))

    task_text = task_file.read_text(encoding="utf-8")
    scorer_client = _get_benchmark_scorer_client(scorer_model)
    skills = _get_benchmark_skills()

    runs_by_model: dict[str, list[dict[str, Any]]] = {}
    run_meta: dict[tuple[str, int], dict[str, Any]] = {}
    for model_name, run_entries in runs_by_model_raw.items():
        runs_by_model[model_name] = []
        for entry in run_entries:
            report_file = Path(str(entry.get("report_file", "")))
            run_index = max(1, int(entry.get("run_index", 1) or 1))
            report_text = report_file.read_text(encoding="utf-8")

            runs_by_model[model_name].append({"run_index": run_index, "report_text": report_text})
            token_usage = _extract_token_usage_from_report_text(report_text)
            run_meta[(model_name, run_index)] = {
                "report_file": str(report_file),
                "elapsed_seconds": _extract_elapsed_seconds_from_report(report_text),
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
            }

    scored = _score_batch_benchmark_case(
        llm_client=scorer_client,
        task_text=task_text,
        runs_by_model=runs_by_model,
        scorer_model=scorer_model,
        case_id=case_id,
        skills=skills,
    )

    # Fallback: if batch JSON omits some model-run pairs, score them one-by-one.
    expected_pairs: set[tuple[str, int]] = set(run_meta.keys())
    scored_pairs: set[tuple[str, int]] = set()
    for model_name, runs in scored.items():
        for run in runs:
            scored_pairs.add((model_name, max(1, int(run.get("run_index", 1) or 1))))

    missing_pairs = sorted(expected_pairs - scored_pairs)
    if missing_pairs:
        print(
            f"[batch scorer] fallback to single-run scoring for case={case_id}, "
            f"missing={len(missing_pairs)}/{len(expected_pairs)}",
            flush=True,
        )

        runs_by_model_lookup: dict[tuple[str, int], str] = {}
        for model_name, run_entries in runs_by_model.items():
            for run_data in run_entries:
                run_index = max(1, int(run_data.get("run_index", 1) or 1))
                runs_by_model_lookup[(model_name, run_index)] = str(run_data.get("report_text", ""))

        # Build calibration references from successfully scored model-runs.
        reference_examples: list[dict[str, Any]] = []
        for ref_model, ref_runs in scored.items():
            for ref_run in ref_runs:
                ref_run_index = max(1, int(ref_run.get("run_index", 1) or 1))
                reference_examples.append(
                    {
                        "model_name": ref_model,
                        "run_index": ref_run_index,
                        "report_text": runs_by_model_lookup.get((ref_model, ref_run_index), ""),
                        "planning_completeness": ref_run.get("planning_completeness"),
                        "tool_reasonableness": ref_run.get("tool_reasonableness"),
                        "code_command_correctness": ref_run.get("code_command_correctness"),
                        "brief_justification": ref_run.get("brief_justification", ""),
                    }
                )

        for model_name, run_index in missing_pairs:
            report_text = runs_by_model_lookup.get((model_name, run_index), "")
            single = _score_single_benchmark_case(
                llm_client=scorer_client,
                task_text=task_text,
                report_text=report_text,
                model_name=model_name,
                scorer_model=scorer_model,
                reference_examples=reference_examples,
                skills=skills,
            )
            scored.setdefault(model_name, []).append(
                {
                    "run_index": run_index,
                    **single,
                }
            )
            reference_examples.append(
                {
                    "model_name": model_name,
                    "run_index": run_index,
                    "report_text": report_text,
                    "planning_completeness": single.get("planning_completeness"),
                    "tool_reasonableness": single.get("tool_reasonableness"),
                    "code_command_correctness": single.get("code_command_correctness"),
                    "brief_justification": single.get("brief_justification", ""),
                }
            )

    packed_results: list[dict[str, Any]] = []
    for model_name, runs in scored.items():
        for run in runs:
            run_index = max(1, int(run.get("run_index", 1) or 1))
            packed_results.append(
                {
                    "model_name": model_name,
                    "case_id": case_id,
                    "run_index": run_index,
                    "task_file": str(task_file),
                    **run_meta.get((model_name, run_index), {}),
                    **run,
                }
            )

    return {"case_id": case_id, "results": packed_results}

def _score_benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    task_file = Path(str(job.get("task_file", "")))
    report_file = Path(str(job.get("report_file", "")))
    model_name = str(job.get("model_name", ""))
    case_id = str(job.get("case_id", ""))
    run_index = max(1, int(job.get("run_index", 1) or 1))
    scorer_model = str(job.get("scorer_model", BENCHMARK_SCORER_MODEL))

    task_text = task_file.read_text(encoding="utf-8")
    report_text = report_file.read_text(encoding="utf-8")
    elapsed_seconds = _extract_elapsed_seconds_from_report(report_text)
    token_usage = _extract_token_usage_from_report_text(report_text)
    scorer_client = _get_benchmark_scorer_client(scorer_model)
    skills = _get_benchmark_skills()
    case_score = _score_single_benchmark_case(
        llm_client=scorer_client,
        task_text=task_text,
        report_text=report_text,
        model_name=model_name,
        scorer_model=scorer_model,
        skills=skills,
    )
    return {
        "model_name": model_name,
        "case_id": case_id,
        "run_index": run_index,
        "task_file": str(task_file),
        "report_file": str(report_file),
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
        **case_score,
    }

def _render_benchmark_leaderboard_markdown(results: dict[str, Any]) -> str:
    meta = results.get("metadata", {}) if isinstance(results, dict) else {}
    ranking = results.get("ranking", []) if isinstance(results, dict) else []

    lines: list[str] = [
        "# Benchmark Leaderboard",
        "",
        f"- Timestamp: {meta.get('timestamp', '')}",
        f"- Scorer model: {meta.get('scorer_model', '')}",
        f"- Benchmark root: {meta.get('benchmark_root', '')}",
        f"- Output dir: {meta.get('output_dir', '')}",
        f"- Scored case count: {meta.get('scored_case_count', '')}",
        "",
        "## Ranking",
        "",
        "| Rank | Model | Average Score (%) | Avg Skill Usage | Avg Tokens | Avg Time (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for idx, item in enumerate(ranking, start=1):
        model = str(item.get("model", ""))
        avg = item.get("average_weighted_score", "")
        avg_calls = item.get("average_usage_count", "-")
        avg_tokens = item.get("average_total_tokens", "-")
        avg_time = item.get("average_elapsed_seconds", "-")
        lines.append(f"| {idx} | {model} | {avg} | {avg_calls} | {avg_tokens} | {avg_time} |")

    if not ranking:
        lines.append("| - | (no complete models) | - | - | - | - |")

    gain = results.get("skill_gain_analysis") if isinstance(results, dict) else None
    comparisons = gain.get("comparisons", []) if isinstance(gain, dict) else []
    if comparisons:
        lines.extend([
            "",
            "## Skill Gain (With Skills vs No Skills)",
            "",
            "| Base Model | With Skills (%) | No Skills (%) | A_abs (%) | g | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for item in comparisons:
            lines.append(
                "| "
                f"{item.get('base_model', '')} | "
                f"{item.get('with_skills_average', '')} | "
                f"{item.get('no_skills_average', '')} | "
                f"{item.get('absolute_improvement', '')} | "
                f"{item.get('normalized_gain', '')} | "
                f"{item.get('interpretation', '')} |"
            )

    return "\n".join(lines) + "\n"

def _score_benchmark_reports(
    benchmark_root: Path,
    output_dir: Path,
    score_workers: int = 8,
    scorer_model: str = BENCHMARK_SCORER_MODEL,
    compare_base_model: str | None = None,
) -> tuple[Path, Path]:
    from core.agent.main import _retry_api_call

    if not benchmark_root.exists() or not benchmark_root.is_dir():
        raise RuntimeError(f"Benchmark directory not found: {benchmark_root}")
    if not output_dir.exists() or not output_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_dir}")

    task_contracts = _discover_task_contracts(benchmark_root)
    expected_case_ids = sorted(task_contracts.keys())
    if not expected_case_ids:
        raise RuntimeError("No benchmark task.md files discovered.")

    report_index = _discover_benchmark_reports(output_dir)
    if not report_index:
        raise RuntimeError("No benchmark report files found in output directory.")

    print(f"Expected benchmark cases: {len(expected_case_ids)}")
    print(f"Discovered benchmark report models in output: {len(report_index)}")

    complete_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, case_map in sorted(report_index.items()):
        if case_map:
            complete_models[model_name] = case_map

    if not complete_models:
        raise RuntimeError("No models with benchmark reports found.")

    case_sets = [set(case_map.keys()) for case_map in complete_models.values()]
    common_case_ids = sorted(set.intersection(*case_sets) if case_sets else set())
    common_case_ids = [case_id for case_id in common_case_ids if case_id in task_contracts]
    if not common_case_ids:
        raise RuntimeError("No shared case ids found across benchmark reports.")

    dropped_models: list[dict[str, Any]] = []
    kept_models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    comparable_runs_per_model: dict[str, int] = {}
    for model_name, case_map in sorted(complete_models.items()):
        case_ids = set(case_map.keys())
        missing_shared = sorted(set(common_case_ids) - case_ids)
        extra = sorted(case_ids - set(common_case_ids))
        run_counts = {case_id: len(case_map.get(case_id, [])) for case_id in common_case_ids}
        comparable_run_count = min(run_counts.values()) if run_counts else 0
        if missing_shared or comparable_run_count <= 0:
            dropped_models.append(
                {
                    "model": model_name,
                    "reason": "missing_shared_case_ids" if missing_shared else "no_runs_for_shared_cases",
                    "found_count": sum(len(case_map.get(case_id, [])) for case_id in common_case_ids),
                    "expected_count": len(common_case_ids),
                    "missing_case_ids": missing_shared,
                    "run_counts": run_counts,
                    "comparable_runs_per_case": comparable_run_count,
                    "extra_case_ids": extra,
                }
            )
            continue
        kept_models[model_name] = case_map
        comparable_runs_per_model[model_name] = comparable_run_count

    complete_models = kept_models
    if not complete_models:
        raise RuntimeError("No models with benchmark reports covering the shared cases found.")

    required_runs_per_case = min(comparable_runs_per_model.values()) if comparable_runs_per_model else 0
    if required_runs_per_case <= 0:
        raise RuntimeError("Unable to infer a comparable benchmark run count from the discovered reports.")

    # If caller requested comparison for a specific base model, restrict to
    # that base model's with_skills and no_skills variants only.
    if compare_base_model:
        want_with = f"{compare_base_model}{BENCHMARK_WITH_SKILLS_SUFFIX}"
        want_no = f"{compare_base_model}{BENCHMARK_NO_SKILLS_SUFFIX}"
        filtered = {k: v for k, v in complete_models.items() if k in {want_with, want_no}}
        if not filtered:
            raise RuntimeError(f"No reports found for specified base model variants: {compare_base_model}")
        complete_models = filtered

    print(f"Comparable models kept for scoring: {len(complete_models)}")
    print(f"Shared case ids selected for scoring: {len(common_case_ids)}")
    print(f"Comparable runs per case used for scoring: {required_runs_per_case}")

    workers = max(1, int(score_workers or 1))
    print(f"Score workers: {workers}")
    resolved_scorer_model = str(scorer_model or "").strip() or BENCHMARK_SCORER_MODEL
    print(f"Scorer model: {resolved_scorer_model}")

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "scorer_model": resolved_scorer_model,
            "scoring_mode": "per_task_joint_scoring_all_models",
            "fairness_policy": (
                "For each task case, all participating models are scored in a single batch call "
                "to minimize rubric drift across models."
            ),
            "score_scale": "0-100",
            "llm_score_components": [
                "planning_completeness",
                "tool_reasonableness",
                "code_command_correctness",
            ],
            "usage_efficiency_note": "Computed separately from skill usage counts within the same case across models; not included in the LLM weighted score.",
            "weights": BENCHMARK_SCORE_WEIGHTS,
            "benchmark_root": str(benchmark_root),
            "output_dir": str(output_dir),
            "runs_per_case": required_runs_per_case,
            "comparable_models": sorted(complete_models.keys()),
            "comparable_model_count": len(complete_models),
            "expected_case_ids": expected_case_ids,
            "scored_case_ids": common_case_ids,
            "scored_case_count": len(common_case_ids),
        },
        "dropped_models": dropped_models,
        "models": {},
        "ranking": [],
        "skill_gain_analysis": {
            "note": "Interpreting Normalized Gain: report both absolute improvement (A_abs) and normalized gain (g). Similar g means similar proportional benefit, not identical absolute gains.",
            "comparisons": [],
        },
    }

    jobs: list[dict[str, Any]] = []
    for case_id in common_case_ids:
        runs_by_model: dict[str, list[dict[str, Any]]] = {}
        for model_name in sorted(complete_models.keys()):
            case_map = complete_models[model_name]
            run_entries = case_map[case_id][:required_runs_per_case]
            runs_by_model[model_name] = [
                {
                    "run_index": int(entry.get("run_index", 1) or 1),
                    "report_file": str(entry["report_file"]),
                }
                for entry in run_entries
            ]
        jobs.append(
            {
                "task_file": str(task_contracts[case_id]),
                "case_id": case_id,
                "runs_by_model": runs_by_model,
                "scorer_model": resolved_scorer_model,
            }
        )

    total_jobs = len(jobs)
    print(f"Total scoring jobs: {total_jobs}", flush=True)

    print(
        "Note: Fair scoring mode enabled - all comparable models are jointly scored per task case",
        flush=True,
    )

    score_start_time = time.time()
    per_case_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_case_id: dict[concurrent.futures.Future[Any], str] = {}
        for job in jobs:
            case_id = str(job.get("case_id", ""))
            future = executor.submit(
                _retry_api_call,
                f"Benchmark scoring job {case_id}",
                lambda job=job: _score_benchmark_batch_job(job),
                10,
            )
            future_to_case_id[future] = case_id

        for done_jobs, future in enumerate(concurrent.futures.as_completed(future_to_case_id), start=1):
            elapsed = time.time() - score_start_time
            avg_per_job = elapsed / done_jobs if done_jobs > 0 else 0.0
            remaining_jobs = max(0, total_jobs - done_jobs)
            eta_seconds = avg_per_job * remaining_jobs

            case_id = future_to_case_id.get(future, f"case-{done_jobs}")
            try:
                batch_result = future.result()
            except Exception as exc:
                print(
                    f"[score job {done_jobs}/{total_jobs}] case={case_id} ERROR: batch scoring failed: {exc} "
                    f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                    flush=True,
                )
                continue

            case_id = str(batch_result.get("case_id", case_id))
            results_list = batch_result.get("results", [])
            unique_result_pairs = {
                (
                    str(item.get("model_name", "")),
                    max(1, int(item.get("run_index", 1) or 1)),
                )
                for item in results_list
                if isinstance(item, dict)
            }
            unique_count = len(unique_result_pairs)
            for result in results_list:
                model_name = str(result.get("model_name", ""))
                run_index = max(1, int(result.get("run_index", 1) or 1))
                result["run_index"] = run_index
                per_case_results.setdefault((model_name, case_id), []).append(result)

            print(
                f"Scoring [{done_jobs}/{total_jobs}] case={case_id} "
                f"(batch scored unique_count={unique_count} model-runs, "
                f"elapsed={_format_duration(elapsed)}, eta={_format_duration(eta_seconds)})",
                flush=True,
            )

    for model_name, case_map in sorted(complete_models.items()):
        per_case: dict[str, Any] = {}
        for case_id in common_case_ids:
            run_results = sorted(
                per_case_results.get((model_name, case_id), []),
                key=lambda item: int(item.get("run_index", 1) or 1),
            )
            if len(run_results) < required_runs_per_case:
                raise RuntimeError(f"Missing scoring results for model={model_name} case={case_id}")
            run_results = run_results[:required_runs_per_case]
            weighted_distribution: list[float] = []
            usage_distribution: list[float] = []
            elapsed_distribution: list[float] = []
            prompt_distribution: list[float] = []
            completion_distribution: list[float] = []
            total_distribution: list[float] = []
            run_records: list[dict[str, Any]] = []
            for result in run_results:
                planning = _clamp_score_1_10(result.get("planning_completeness", 1))
                tool = _clamp_score_1_10(result.get("tool_reasonableness", 1))
                code = _clamp_score_1_10(result.get("code_command_correctness", 1))
                weighted_10 = round(
                    planning * BENCHMARK_SCORE_WEIGHTS["planning_completeness"]
                    + tool * BENCHMARK_SCORE_WEIGHTS["tool_reasonableness"]
                    + code * BENCHMARK_SCORE_WEIGHTS["code_command_correctness"],
                    3,
                )
                weighted_score = round(weighted_10 * 10.0, 2)
                weighted_distribution.append(weighted_score)
                usage_count = result.get("usage_count")
                if isinstance(usage_count, int):
                    usage_distribution.append(float(usage_count))
                if isinstance(result.get("elapsed_seconds"), (int, float)):
                    elapsed_distribution.append(float(result["elapsed_seconds"]))
                if isinstance(result.get("prompt_tokens"), int):
                    prompt_distribution.append(float(result["prompt_tokens"]))
                if isinstance(result.get("completion_tokens"), int):
                    completion_distribution.append(float(result["completion_tokens"]))
                if isinstance(result.get("total_tokens"), int):
                    total_distribution.append(float(result["total_tokens"]))
                run_records.append(
                    {
                        "run_index": int(result.get("run_index", 1) or 1),
                        "task_file": result["task_file"],
                        "report_file": result["report_file"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"],
                        "total_tokens": result["total_tokens"],
                        "planning_completeness": result["planning_completeness"],
                        "tool_reasonableness": result["tool_reasonableness"],
                        "code_command_correctness": result["code_command_correctness"],
                        "usage_count": result.get("usage_count"),
                        "brief_justification": result["brief_justification"],
                        "raw_model_output": result["raw_model_output"],
                        "weighted_score_10": weighted_10,
                        "weighted_score": weighted_score,
                    }
                )

            weighted_mean, weighted_variance = _mean_and_variance(weighted_distribution)
            usage_mean, usage_variance = _mean_and_variance(usage_distribution)
            elapsed_mean, elapsed_variance = _mean_and_variance(elapsed_distribution)
            prompt_mean, prompt_variance = _mean_and_variance(prompt_distribution)
            completion_mean, completion_variance = _mean_and_variance(completion_distribution)
            total_mean, total_variance = _mean_and_variance(total_distribution)
            per_case[case_id] = {
                "task_file": run_records[0]["task_file"],
                "runs": run_records,
                "run_count": len(run_records),
                "score_distribution": weighted_distribution,
                "average_weighted_score": weighted_mean,
                "variance_weighted_score": weighted_variance,
                "average_usage_count": usage_mean,
                "variance_usage_count": usage_variance,
                "average_elapsed_seconds": elapsed_mean,
                "variance_elapsed_seconds": elapsed_variance,
                "average_prompt_tokens": prompt_mean,
                "variance_prompt_tokens": prompt_variance,
                "average_completion_tokens": completion_mean,
                "variance_completion_tokens": completion_variance,
                "average_total_tokens": total_mean,
                "variance_total_tokens": total_variance,
            }
        results["models"][model_name] = {
            "case_count": len(common_case_ids),
            "average_weighted_score": 0.0,
            "variance_weighted_score": None,
            "average_usage_count": None,
            "variance_usage_count": None,
            "average_elapsed_seconds": None,
            "variance_elapsed_seconds": None,
            "average_prompt_tokens": None,
            "variance_prompt_tokens": None,
            "average_completion_tokens": None,
            "variance_completion_tokens": None,
            "average_total_tokens": None,
            "variance_total_tokens": None,
            "cases": per_case,
        }

    for data in results["models"].values():
        score_values: list[float] = []
        usage_values: list[float] = []
        elapsed_values: list[float] = []
        prompt_values: list[float] = []
        completion_values: list[float] = []
        total_values: list[float] = []
        for case_id in common_case_ids:
            case_data = data.get("cases", {}).get(case_id, {})
            if isinstance(case_data.get("average_weighted_score"), (int, float)):
                score_values.append(float(case_data["average_weighted_score"]))
            usage_value = case_data.get("average_usage_count")
            if isinstance(usage_value, (int, float)):
                usage_values.append(float(usage_value))
            if isinstance(case_data.get("average_elapsed_seconds"), (int, float)):
                elapsed_values.append(float(case_data["average_elapsed_seconds"]))
            if isinstance(case_data.get("average_prompt_tokens"), (int, float)):
                prompt_values.append(float(case_data["average_prompt_tokens"]))
            if isinstance(case_data.get("average_completion_tokens"), (int, float)):
                completion_values.append(float(case_data["average_completion_tokens"]))
            if isinstance(case_data.get("average_total_tokens"), (int, float)):
                total_values.append(float(case_data["average_total_tokens"]))

        score_mean, score_variance = _mean_and_variance(score_values)
        usage_mean, usage_variance = _mean_and_variance(usage_values)
        elapsed_mean, elapsed_variance = _mean_and_variance(elapsed_values)
        prompt_mean, prompt_variance = _mean_and_variance(prompt_values)
        completion_mean, completion_variance = _mean_and_variance(completion_values)
        total_mean, total_variance = _mean_and_variance(total_values)
        data["average_weighted_score"] = round(score_mean, 2) if score_mean is not None else None
        data["variance_weighted_score"] = score_variance
        data["average_usage_count"] = round(usage_mean, 2) if usage_mean is not None else None
        data["variance_usage_count"] = usage_variance
        data["average_elapsed_seconds"] = round(elapsed_mean, 3) if elapsed_mean is not None else None
        data["variance_elapsed_seconds"] = elapsed_variance
        data["average_prompt_tokens"] = round(prompt_mean, 2) if prompt_mean is not None else None
        data["variance_prompt_tokens"] = prompt_variance
        data["average_completion_tokens"] = round(completion_mean, 2) if completion_mean is not None else None
        data["variance_completion_tokens"] = completion_variance
        data["average_total_tokens"] = round(total_mean, 2) if total_mean is not None else None
        data["variance_total_tokens"] = total_variance

    ranking = sorted(
        (
            {
                "model": model,
                "average_weighted_score": data["average_weighted_score"],
                "average_usage_count": data.get("average_usage_count"),
                "average_total_tokens": data.get("average_total_tokens"),
                "average_elapsed_seconds": data.get("average_elapsed_seconds"),
            }
            for model, data in results["models"].items()
        ),
        key=lambda x: (
            -float(x.get("average_weighted_score", 0.0) or 0.0),
            float(x.get("average_usage_count", 1e9) or 1e9),
            float(x.get("average_total_tokens", 1e18) or 1e18),
            float(x.get("average_elapsed_seconds", 1e18) or 1e18),
        ),
    )
    results["ranking"] = ranking

    condition_groups: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, data in results["models"].items():
        base_model, cond = _split_skill_condition(model_name)
        condition_groups.setdefault(base_model, {})[cond] = data

    comparisons: list[dict[str, Any]] = []
    for base_model, cond_map in sorted(condition_groups.items()):
        with_data = cond_map.get("with_skills")
        no_data = cond_map.get("no_skills")
        if not with_data or not no_data:
            continue

        with_avg = float(with_data.get("average_weighted_score", 0.0) or 0.0)
        no_avg = float(no_data.get("average_weighted_score", 0.0) or 0.0)
        abs_improvement = round(with_avg - no_avg, 2)
        norm_gain = _compute_normalized_gain(with_avg, no_avg)

        per_case: list[dict[str, Any]] = []
        for case_id in common_case_ids:
            with_case = with_data.get("cases", {}).get(case_id, {})
            no_case = no_data.get("cases", {}).get(case_id, {})
            with_case_score = float(with_case.get("average_weighted_score", 0.0) or 0.0)
            no_case_score = float(no_case.get("average_weighted_score", 0.0) or 0.0)
            case_abs = round(with_case_score - no_case_score, 2)
            case_g = _compute_normalized_gain(with_case_score, no_case_score)
            per_case.append(
                {
                    "case_id": case_id,
                    "with_skills_score": with_case_score,
                    "no_skills_score": no_case_score,
                    "with_skills_variance": with_case.get("variance_weighted_score"),
                    "no_skills_variance": no_case.get("variance_weighted_score"),
                    "absolute_improvement": case_abs,
                    "normalized_gain": case_g,
                }
            )

        comparisons.append(
            {
                "base_model": base_model,
                "with_skills_model": f"{base_model}{BENCHMARK_WITH_SKILLS_SUFFIX}",
                "no_skills_model": f"{base_model}{BENCHMARK_NO_SKILLS_SUFFIX}",
                "with_skills_average": round(with_avg, 2),
                "no_skills_average": round(no_avg, 2),
                "absolute_improvement": abs_improvement,
                "normalized_gain": norm_gain,
                "interpretation": _interpret_normalized_gain(abs_improvement, norm_gain),
                "per_case": per_case,
            }
        )

    results["skill_gain_analysis"]["comparisons"] = comparisons

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    scorer_slug = _slugify_filename_part(resolved_scorer_model, "scorer", max_len=80)
    out_file = output_dir / f"benchmark_scores_{scorer_slug}_{timestamp}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    leaderboard_file = output_dir / f"benchmark_leaderboard_{scorer_slug}_{timestamp}.md"
    leaderboard_file.write_text(_render_benchmark_leaderboard_markdown(results), encoding="utf-8")
    return out_file, leaderboard_file

def _get_benchmark_skills() -> list[dict[str, Any]]:
    global _BENCHMARK_SKILLS_CACHE
    if _BENCHMARK_SKILLS_CACHE is not None:
        return _BENCHMARK_SKILLS_CACHE

    from core.skill_loader.loader import SkillLoader
    loader = SkillLoader(REPO_ROOT / "skills")
    _BENCHMARK_SKILLS_CACHE = loader.load_all()
    return _BENCHMARK_SKILLS_CACHE


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score NeuroBench reports and generate a leaderboard")
    parser.add_argument("--benchmark-root", type=Path, default=REPO_ROOT / "neurobench/tasks")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=BENCHMARK_SCORER_MODEL)
    parser.add_argument("--compare-model")
    args = parser.parse_args(argv)
    scores, leaderboard = _score_benchmark_reports(
        args.benchmark_root, args.output_root, score_workers=args.workers,
        scorer_model=args.model, compare_base_model=args.compare_model)
    print(f"Benchmark scoring completed: {scores}")
    print(f"Leaderboard generated: {leaderboard}")


if __name__ == "__main__":
    main()
