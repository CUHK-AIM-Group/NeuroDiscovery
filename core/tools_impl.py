"""Execution bodies for the tools registered in ``core/tool_registry``.

These are the same operations the agent loop previously kept inline in its
dispatch chain. Moving them here does not change behavior: each returns the same
result dictionary the loop built before, including ``benchmark_mode`` short
circuits and the ``tool_events`` bookkeeping each branch used to do.

Handlers receive ``(session, arguments, workspace)``. They may touch session
state, because the loop's own branches did: benchmark mode, the checkpoint
manager, the tool-event log and the workspace root are all session attributes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.tool_registry import ToolSpec, register


def _record(session: Any, event: dict[str, Any]) -> None:
    session._tool_events.append(event)


def _skill_capability(session: Any):
    """Lazily build the skill capability for this session's workspace."""
    capability = getattr(session, "_skill_capability", None)
    if capability is None:
        from core.harness_skills import default_capability

        capability = default_capability(session.workspace / "skills")
        session._skill_capability = capability
    return capability


def _search_skills(session: Any, args: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    query = str(args.get("query", ""))
    limit = int(args.get("limit", 8))
    capability = _skill_capability(session)
    matches = capability.search(query, limit=limit)
    if not matches:
        return {
            "success": False,
            "executed": False,
            "error_type": "no_skill_match",
            "error": f"No skill matched '{query}'. Try broader domain words such as fMRI, EEG, connectivity or dataset names.",
            "skill_count": len(capability.catalog()),
        }
    result = {
        "success": True,
        "query": query,
        "matches": [
            {"name": item.name, "summary": capability.summary_line(item), "layer": item.layer, "type": item.skill_type}
            for item in matches
        ],
        "skill_count": len(capability.catalog()),
    }
    _record(session, {
        "tool": "search_skills",
        "command": query,
        "executed": True,
        "success": True,
        "skills_used": [],
        "result": result,
    })
    return result


def _read_skill(session: Any, args: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    name = str(args.get("name", ""))
    result = _skill_capability(session).load(name, offset=int(args.get("offset", 0)))
    _record(session, {
        "tool": "read_skill",
        "command": name,
        "executed": bool(result.get("success")),
        "success": bool(result.get("success")),
        "skills_used": [str(result.get("name"))] if result.get("success") else [],
        "result": result,
    })
    return result


def _run_shell(session: Any, args: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    from core.agent.main import _looks_file_io_shell_command, _run_shell_command
    from core.shell_safety import is_readonly_probe

    command = str(args.get("command", ""))
    timeout_sec = int(args.get("timeout_sec", 180))

    if session._checkpoint_mgr is not None and not session.benchmark_mode and not is_readonly_probe(command):
        try:
            session._checkpoint_mgr.checkpoint(
                workspace, label=f"before: {command[:80]}", timeout=15, cancel_event=session._cancel_event
            )
        except Exception as exc:  # noqa: BLE001 - checkpointing is best effort
            session._emit_execution_event({"type": "checkpoint", "status": "skipped", "error": str(exc)[:500]})

    if session.benchmark_mode and _looks_file_io_shell_command(command):
        result = {
            "success": True,
            "benchmark_mode": True,
            "executed": False,
            "message": "Benchmark mode skipped real execution for file/dataset I/O task. Return command/code only.",
            "suggested_command": command,
        }
    else:
        result = _run_shell_command(
            command=command,
            cwd=workspace,
            timeout_sec=timeout_sec,
            cancel_event=session._cancel_event,
        )
        if session.benchmark_mode:
            result["benchmark_mode"] = True
            result["executed"] = True

    _record(session, {
        "tool": "run_shell_command",
        "command": command,
        "executed": bool(
            result.get("executed")
            if "executed" in result
            else result.get("failure_stage") not in {"validation", "process_start"}
        ),
        "success": bool(result.get("success", False)),
        "skills_used": _extract_skills(result),
        "result": result,
    })
    return result


def _inspect_path(session: Any, args: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    from core.agent.main import _inspect_local_path

    path = str(args.get("path", ""))
    if session.benchmark_mode:
        result = {
            "success": True,
            "benchmark_mode": True,
            "executed": False,
            "message": "Benchmark mode skipped real local path inspection.",
            "suggested_path": path,
        }
    else:
        result = _inspect_local_path(path, workspace)
    _record(session, {
        "tool": "inspect_local_path",
        "command": path,
        "executed": bool(result.get("executed", True)),
        "success": bool(result.get("success", False)),
        "skills_used": [],
        "result": result,
    })
    return result


def _read_file(session: Any, args: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    from core.agent.main import _read_workspace_file

    path = str(args.get("path", ""))
    max_chars = args.get("max_chars", 12000)
    autoresearch = getattr(session, "_autoresearch_run", None)
    result = _read_workspace_file(
        path,
        workspace,
        max_chars=max_chars,
        offset=args.get("offset"),
        paper_start=args.get("paper_start"),
        paper_count=args.get("paper_count", 5),
        ledger=autoresearch.state.setdefault("reading_ledger", {}) if autoresearch else None,
    )
    if autoresearch:
        autoresearch.save()
    if result.get("success") and not session.benchmark_mode:
        from core.instructions import scoped_instructions

        target = Path(path)
        if not target.is_absolute():
            target = workspace / target
        result["scoped_instructions"] = scoped_instructions(workspace, target)
    _record(session, {
        "tool": "read_workspace_file",
        "command": path,
        "executed": bool(result.get("success", False)),
        "success": bool(result.get("success", False)),
        "skills_used": _extract_skills(result),
        "result": result,
    })
    return result


def _extract_skills(result: dict[str, Any]) -> list[str]:
    from core.agent.main import _extract_skills_from_result_payload

    return _extract_skills_from_result_payload(result)


def register_core_tools() -> None:
    """Register the four inline tools, matching their original schemas exactly."""
    from core.research_reading import PAPER_PAGE_MAX, READ_MAX_CHARS

    register(ToolSpec(
        name="run_shell_command",
        description=(
            "Run a shell command in the local workspace using the platform shell "
            "and inherited environment variables. On Windows, use cmd.exe syntax "
            "(for example: dir, type, and if exist), not bash syntax."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Exact shell command to execute."},
                "timeout_sec": {"type": "integer", "description": "Timeout in seconds (default 180)."},
            },
            "required": ["command"],
        },
        handler=_run_shell,
    ))
    register(ToolSpec(
        name="inspect_local_path",
        description=(
            "Inspect a local file or directory without using a shell. Returns existence, resolved path, "
            "kind, and exact file size. Prefer this for attachment existence/size questions and as a "
            "read-only fallback when shell execution fails."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Exact local path supplied by the user or recorded for an attachment."},
            },
            "required": ["path"],
        },
        handler=_inspect_path,
    ))
    register(ToolSpec(
        name="read_workspace_file",
        description=(
            "Read a UTF-8 text file inside the current workspace, such as a SKILL.md, script, or config file. "
            "Use this for JSON and Chinese text; never use type | more, which can corrupt Unicode. "
            "In AutoResearch literature JSON is read as complete papers, and omitted offsets continue at the first "
            "unread range. Use explicit offset or paper_start for intentional rereading. EOF means synthesize, not restart."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative or absolute path to a text file inside the workspace."},
                "max_chars": {"type": "integer", "description": f"Maximum characters to return (default 12000, maximum {READ_MAX_CHARS})."},
                "offset": {"type": "integer", "minimum": 0, "description": "Zero-based character start; forces text mode."},
                "paper_start": {"type": "integer", "minimum": 0, "description": "Zero-based paper index in literature JSON; excludes offset."},
                "paper_count": {"type": "integer", "minimum": 1, "maximum": PAPER_PAGE_MAX, "description": "Maximum complete papers per page (default 5)."},
            },
            "required": ["path"],
        },
        handler=_read_file,
    ))
    register(ToolSpec(
        name="spawn_subagent",
        description=(
            "Spawn an independent subagent session to execute a subagent-layer "
            "skill or a specialized task autonomously. The subagent has its own "
            "conversation history, session ID, and can run tool calls independently."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task description for the subagent to execute."},
                "persona": {"type": "string", "description": "Optional expert persona: 'biostatistician', 'clinical_neuroscientist', 'methodology_expert'."},
                "skills_filter": {"type": "array", "items": {"type": "string"}, "description": "Optional list of skill names to restrict the subagent to."},
                "mode": {"type": "string", "enum": ["run_and_return", "fire_and_forget"], "description": "'run_and_return' waits for result, 'fire_and_forget' returns immediately."},
            },
            "required": ["task"],
        },
        handler=_spawn_subagent,
    ))
    register(ToolSpec(
        name="search_skills",
        description=(
            "Search the NeuroRuntime skill library for skills relevant to a task. "
            "Returns ranked skill names with short summaries. Call this before proposing new code when a task "
            "involves programming, data processing, model inference or training, file I/O, visualization or a "
            "named dataset or modality, then load the best match with read_skill."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Task keywords, for example 'fMRI connectivity' or 'ADNI BIDS'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Maximum matches to return (default 8)."},
            },
            "required": ["query"],
        },
        handler=_search_skills,
    ))
    register(ToolSpec(
        name="read_skill",
        description=(
            "Load the full instructions for one skill named in the session skill catalog. "
            "If truncated, call again with the returned next_offset to read the remaining instructions. "
            "Call it before acting on a task that names or clearly matches a listed skill. "
            "The skill is reusable guidance, not a rigid pipeline: reuse the parts that fit the task contract."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The exact skill name from the skill catalog."},
                "offset": {"type": "integer", "minimum": 0,
                           "description": "Zero-based character offset; use next_offset when a prior page was truncated."},
            },
            "required": ["name"],
        },
        handler=_read_skill,
    ))


SKILL_TOOL_NAMES = ("search_skills", "read_skill")


def skill_tools_enabled() -> bool:
    """Whether the on-demand skill tools are part of the visible surface.

    Disabled by default so existing paths, including the legacy benchmark suite
    and its published 500-task results, keep the exact tool surface they had.
    DeepSeek Harness uses the same posture for its optional skill providers: the
    plugin ships disabled and enabling its composition row is an explicit opt-in.
    The benchmark runner turns this on for the harness arm and leaves it off for
    the baseline arm, which is what makes the two arms differ in more than prompt
    text.
    """
    from core.config_flags import feature_enabled

    return feature_enabled("NEUROCLAW_HARNESS_SKILL_TOOLS")


def visible_schemas(**kwargs) -> list[dict[str, Any]]:
    """Provider-facing schemas, hiding the skill tools unless the seam is on."""
    from core import tool_registry

    exclude = list(kwargs.pop("exclude", []) or [])
    if not skill_tools_enabled():
        exclude.extend(SKILL_TOOL_NAMES)
    return tool_registry.schemas(exclude=exclude, **kwargs)


def _spawn_subagent(session: Any, args: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    task = str(args.get("task", ""))
    persona = str(args.get("persona", ""))
    skills_filter = args.get("skills_filter")
    mode = str(args.get("mode", "run_and_return"))
    autoresearch = getattr(session, "_autoresearch_run", None)
    try:
        if autoresearch and mode == "fire_and_forget":
            raise ValueError(
                "AutoResearch must wait for delegated deliverables; use run_and_return or execute the work directly."
            )
        manager = session._get_or_create_subagent_manager()
        session_id = manager.spawn(
            task,
            persona=persona,
            skills_filter=skills_filter,
            mode=mode,
        )
        if mode == "run_and_return":
            sub_result = manager.get_result(session_id, timeout=180.0)
            result = {
                "success": sub_result.status == "completed",
                "session_id": session_id,
                "response": sub_result.response,
                "error": sub_result.error,
            }
        else:
            result = {
                "success": True,
                "session_id": session_id,
                "message": f"Subagent {session_id} spawned in fire-and-forget mode.",
            }
    except Exception as exc:  # noqa: BLE001 - reported to the model
        result = {"success": False, "error": str(exc)}
    _record(session, {
        "tool": "spawn_subagent",
        "command": task[:100],
        "executed": True,
        "success": bool(result.get("success", False)),
        "skills_used": [],
        "result": result,
    })
    return result
