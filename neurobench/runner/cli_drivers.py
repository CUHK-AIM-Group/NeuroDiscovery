"""External CLI drivers with isolated settings and durable event streams."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from neurobench.runner.artifacts import freeze_text, json_text
from neurobench.runner.drivers import AttemptOutcome, SKILL_TOOLS, classify_retry, normalize_usage, resolve_api_key

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ALLOWED = tuple("mcp__neuro_runtime__" + name for name in SKILL_TOOLS)


def find_codex_cli() -> str:
    return _find_cli("NEUROCLAW_CODEX_CLI", "codex", "@openai/codex/bin/codex.js")


def find_claude_cli() -> str:
    return _find_cli("NEUROCLAW_CLAUDE_CLI", "claude", "@anthropic-ai/claude-code/bin/claude.exe")


def _find_cli(variable, runtime, relative) -> str:
    override = (os.environ.get(variable) or "").strip()
    candidate = Path(override) if override else (
        Path(os.environ.get("APPDATA", "")) / "dsh-desktop/runtimes" / runtime / "node_modules" / relative)
    if not candidate.is_file():
        raise RuntimeError("CLI not found; set " + variable)
    return str(candidate)


def write_codex_home(codex_home: Path, *, base_url: str, harness_enabled: bool,
                     skills_dir: Path | None = None) -> Path:
    lines = ['model_provider = "nvidia"', 'approval_policy = "never"', "",
             "[model_providers.nvidia]", 'name = "nvidia"',
             'base_url = "%s"' % base_url, 'env_key = "NK"',
             "request_max_retries = 0", "stream_max_retries = 0"]
    if harness_enabled:
        from core.mcp_stdio import mount_config
        mount = mount_config(python_executable=sys.executable, workspace=str(REPO_ROOT))
        lines += ["", "[mcp_servers.neuro_runtime]",
                  "command = " + json.dumps(mount["command"].replace("\\", "/")),
                  "args = " + json.dumps([a.replace("\\", "/") for a in mount["args"]]),
                  'default_tools_approval_mode = "approve"',
                  "startup_timeout_sec = 30", "tool_timeout_sec = 120",
                  "[mcp_servers.neuro_runtime.env]",
                  'NEURORUNTIME_MCP_TOOLS = "search_skills,read_skill"',
                  "NEURORUNTIME_MCP_WORKSPACE = " + json.dumps(str(REPO_ROOT).replace("\\", "/"))]
        if skills_dir is not None:
            lines += ["NEURORUNTIME_MCP_SKILLS_DIR = " + json.dumps(str(skills_dir).replace("\\", "/"))]
    freeze_text(codex_home / "config.toml", "\n".join(lines) + "\n")
    return codex_home


def write_claude_mcp_config(path: Path, *, workspace: Path, harness_enabled: bool = True,
                            skills_dir: Path | None = None) -> Path:
    from core.mcp_stdio import mount_config
    servers = {}
    if harness_enabled:
        mount = mount_config(python_executable=sys.executable, workspace=str(workspace))
        mount["env"]["NEURORUNTIME_MCP_TOOLS"] = ",".join(SKILL_TOOLS)
        if skills_dir is not None:
            mount["env"]["NEURORUNTIME_MCP_SKILLS_DIR"] = str(skills_dir)
        servers["neuro_runtime"] = mount
    freeze_text(path, json_text({"mcpServers": servers}))
    return path


CODEX_DISABLED_FEATURES = (
    "apps", "browser_use", "browser_use_external", "in_app_browser", "computer_use",
    "image_generation", "multi_agent", "multi_agent_v2", "plugins", "skill_search",
    "hooks", "standalone_web_search", "shell_tool", "unified_exec",
)


def _paths(output_root, slug, case_id, driver, attempt):
    stem = "%s.%s.attempt%d" % (case_id, driver, attempt)
    return (output_root / "events" / slug / (stem + ".jsonl"),
            output_root / "stderr" / slug / (stem + ".txt"))


def _capture(command, *, prompt, workspace, env, events_path, stderr_path, timeout):
    """Write stdout as it arrives; a lost parent never discards completed events."""
    for path in (events_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    freeze_text(events_path.with_suffix(".launch.json"), json_text({"command": command, "cwd": str(workspace),
                                                                   "timeout_seconds": timeout}))
    error, code = "", None
    with events_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
            cwd=str(workspace), env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            process.communicate(prompt.encode("utf-8"), timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.kill()
            process.communicate(timeout=30)
            code, error = process.returncode, "CLI timed out after %ss" % timeout
    return code, error, events_path.read_text(encoding="utf-8", errors="replace"), stderr_path.read_text(encoding="utf-8", errors="replace")


def _make_outcome(summary, *, error, raw_stdout, raw_stderr, elapsed, settings, response=None):
    response = summary.get("response", "") if response is None else response
    error = error or summary.get("error", "")
    if not error and not summary["terminal"]:
        error = "CLI stream ended without a terminal result"
    if not error and not response:
        error = "CLI result was empty"
    unknown = (not summary["terminal"]) or "timed out" in error
    return AttemptOutcome(
        ok=not error, error=error, retry_class=classify_retry(error) if error else "",
        session_id=summary["session_id"], response=response,
        input_tokens=summary["input_tokens"], cached_input_tokens=summary["cached_input_tokens"],
        output_tokens=summary["output_tokens"], elapsed_seconds=elapsed,
        tool_events=summary["tool_events"], skills_loaded=summary["skills_loaded"],
        raw_stdout=raw_stdout, raw_stderr=raw_stderr,
        uncached_input_tokens=summary["uncached_input_tokens"],
        cache_creation_input_tokens=summary["cache_creation_input_tokens"],
        usage_records=summary["usage_records"], usage_complete=summary["usage_complete"] and not unknown,
        remote_outcome_unknown=unknown, reasoning_effort=settings["reasoning_effort"],
        model_ids=summary.get("model_ids", []), runtime_settings=settings)


def run_codex_cli(*, prompt, model, base_url, harness_enabled, workspace, codex_home,
                  output_root, case_id, attempt, timeout_seconds, reasoning_effort="high", artifact_slug=""):
    events, stderr = _paths(output_root, artifact_slug, case_id, "codex", attempt)
    last = events.with_suffix(".last.md")
    command = [os.environ.get("NEUROCLAW_NODE") or "node", find_codex_cli(), "exec",
        "--ephemeral", "--json", "--color", "never", "--ignore-rules", "--skip-git-repo-check",
        "--model", model, "--config", 'model_reasoning_effort="%s"' % reasoning_effort,
        "--sandbox", "read-only", "--config", "sandbox_workspace_write.network_access=false",
        "--config", "suppress_unstable_features_warning=true", "--cd", str(workspace),
        "--output-last-message", str(last), "--enable", "skip_host_skill_discovery"]
    for feature in CODEX_DISABLED_FEATURES:
        command += ["--disable", feature]
    command.append("-")
    env = os.environ.copy()
    env.update(CODEX_HOME=str(codex_home), NK=resolve_api_key("nvidia"), NO_COLOR="1")
    started = time.monotonic()
    code, error, raw, errors = _capture(command, prompt=prompt, workspace=workspace, env=env,
                                      events_path=events, stderr_path=stderr, timeout=timeout_seconds)
    summary = parse_cli_events(raw)
    if code and not error:
        error = "codex exited %s: %s" % (code, errors[-1000:])
    response = last.read_text(encoding="utf-8") if last.exists() else summary["response"]
    settings = {"reasoning_effort": reasoning_effort, "base_url": base_url,
                "harness_enabled": harness_enabled, "timeout_seconds": timeout_seconds,
                "disabled_features": list(CODEX_DISABLED_FEATURES), "internal_retries": 0}
    return _make_outcome(summary, error=error, raw_stdout=raw, raw_stderr=errors,
                         elapsed=time.monotonic()-started, settings=settings, response=response)


def run_claude_code(*, prompt, model, base_url, harness_enabled, workspace, output_root,
                    case_id, attempt, timeout_seconds, mcp_config_path, artifact_slug="",
                    claude_config_dir=None, reasoning_effort="high"):
    events, stderr = _paths(output_root, artifact_slug, case_id, "claude", attempt)
    if claude_config_dir is None:
        claude_config_dir = output_root / "control" / artifact_slug / case_id / ("attempt%d" % attempt)
    freeze_text(claude_config_dir / "settings.json", "{}\n")
    if mcp_config_path is None:
        mcp_config_path = write_claude_mcp_config(claude_config_dir / "mcp.json",
                                                workspace=REPO_ROOT, harness_enabled=harness_enabled)
    command = [find_claude_cli(), "-p", "--bare", "--output-format", "stream-json", "--verbose", "--model", model,
               "--tools", "", "--disable-slash-commands", "--setting-sources", "",
               "--strict-mcp-config", "--mcp-config", str(mcp_config_path), "--no-session-persistence",
               "--effort", reasoning_effort]
    if harness_enabled:
        command += ["--allowedTools", ",".join(MCP_ALLOWED)]
    # Anthropic client appends /v1/messages. Its base is the host, unlike OpenAI's.
    anthropic_base = base_url.rstrip("/")
    if anthropic_base.endswith("/v1"):
        anthropic_base = anthropic_base[:-3]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env.pop("MAX_THINKING_TOKENS", None)
    # Bare mode explicitly suppresses CLAUDE.md, memory, plugins and background
    # context discovery. Its documented auth source is ANTHROPIC_API_KEY.
    env.update(CLAUDE_CONFIG_DIR=str(claude_config_dir), ANTHROPIC_API_KEY=resolve_api_key("nvidia"),
               ANTHROPIC_BASE_URL=anthropic_base, ANTHROPIC_MODEL=model,
               ANTHROPIC_DEFAULT_OPUS_MODEL=model, ANTHROPIC_DEFAULT_SONNET_MODEL=model,
               ANTHROPIC_DEFAULT_HAIKU_MODEL=model, CLAUDE_CODE_EFFORT_LEVEL=reasoning_effort,
               CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT="1")
    started = time.monotonic()
    code, error, raw, errors = _capture(command, prompt=prompt, workspace=workspace, env=env,
                                      events_path=events, stderr_path=stderr, timeout=timeout_seconds)
    summary = parse_claude_result(raw)
    if code and not error:
        error = "claude exited %s: %s" % (code, summary["error"] or errors[-1000:])
    settings = {"reasoning_effort": reasoning_effort, "MAX_THINKING_TOKENS": "unset", "base_url": anthropic_base,
                "harness_enabled": harness_enabled, "timeout_seconds": timeout_seconds,
                "builtin_tools": [], "bare": True,
                "allowed_mcp_tools": list(MCP_ALLOWED) if harness_enabled else []}
    return _make_outcome(summary, error=error, raw_stdout=raw, raw_stderr=errors,
                         elapsed=time.monotonic()-started, settings=settings)


def _json_events(text):
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def _tool_result_ok(result):
    if isinstance(result, dict):
        if result.get("isError") or result.get("is_error") or result.get("success") is False:
            return False
        if result.get("success") is True:
            return True
        result = result.get("content", [])
    if isinstance(result, str):
        try:
            return _tool_result_ok(json.loads(result))
        except ValueError:
            return False
    if isinstance(result, list):
        return any(_tool_result_ok(block.get("text", "")) for block in result if isinstance(block, dict))
    return False


def _summary():
    return dict(session_id="", response="", terminal=False, error="", input_tokens=0,
                cached_input_tokens=0, uncached_input_tokens=0, cache_creation_input_tokens=0,
                output_tokens=0, usage_records=[], usage_complete=False,
                tool_events=[], skills_loaded=[], model_ids=[])


def _loaded(summary):
    for event in summary["tool_events"]:
        name = event["name"]
        if event["ok"] and name in ("neuro_runtime/read_skill", "mcp__neuro_runtime__read_skill"):
            skill = (event.get("arguments") or {}).get("name")
            if skill and skill not in summary["skills_loaded"]:
                summary["skills_loaded"].append(skill)


def parse_cli_events(raw_stdout: str) -> dict:
    summary = _summary()
    events = {}
    for row in _json_events(raw_stdout):
        kind = row.get("type", "")
        if kind == "thread.started":
            summary["session_id"] = row.get("thread_id") or row.get("threadId") or ""
        if kind in ("turn.completed", "turn.finished"):
            summary["terminal"] = True
            usage = row.get("usage")
            if usage is not None:
                summary["usage_records"].append(usage)
            normalized = normalize_usage(usage, "codex")
            summary.update(normalized)
        if kind == "turn.failed":
            summary["terminal"] = True
            summary["error"] = json.dumps(row.get("error") or row, ensure_ascii=False)
        item = row.get("item") or {}
        item_type = item.get("type", "")
        if item_type == "agent_message" and kind == "item.completed":
            summary["response"] = item.get("text") or summary["response"]
        if not any(t in item_type for t in ("tool", "command", "mcp", "web_search", "file_change")):
            continue
        name = (str(item.get("server")) + "/" + str(item["tool"])) if item.get("tool") else str(item.get("name") or item_type)
        item_id = item.get("id") or "unidentified_%d" % len(events)
        completed = kind == "item.completed" and item.get("status") != "in_progress" and not item.get("error")
        # A native shell item has exit_code/aggregated_output, not an MCP result.
        # Preserve failed and incomplete commands rather than inferring success
        # from the model's narration or from a turn's terminal marker.
        command = item_type == "command_execution"
        ok = completed and (type(item.get("exit_code")) is int and item["exit_code"] == 0 and item.get("status") == "completed"
                            if command else _tool_result_ok(item.get("result")))
        events[item_id] = {"id": item_id, "name": name,
                           "arguments": {"command": item.get("command")} if command else item.get("arguments") or {},
                           "ok": bool(ok)}
        if command:
            events[item_id]["exit_code"] = item.get("exit_code")
    summary["tool_events"] = list(events.values())
    _loaded(summary)
    return summary


def parse_claude_result(raw_stdout: str) -> dict:
    """Prefer all-model billing usage; preserve both provider usage representations."""
    summary = _summary()
    payload = {}
    calls, results = {}, {}
    for row in _json_events(raw_stdout):
        if row.get("session_id"):
            summary["session_id"] = row["session_id"]
        if row.get("type") == "result":
            payload = row
            summary["terminal"] = True
        message = row.get("message") or {}
        blocks = message.get("content") or []
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if block.get("type") == "tool_use":
                calls[block["id"]] = {"id": block["id"], "name": block.get("name", ""),
                                       "arguments": block.get("input") or {}, "ok": False}
            if block.get("type") == "tool_result":
                results[block.get("tool_use_id")] = not block.get("is_error") and _tool_result_ok(block.get("content"))
    for tool_id, call in calls.items():
        call["ok"] = results.get(tool_id, False)
    summary["tool_events"] = list(calls.values())
    summary["response"] = str(payload.get("result") or "")
    summary["is_error"] = bool(payload.get("is_error"))
    summary["subtype"] = payload.get("subtype", "")
    if summary["is_error"]:
        summary["error"] = "%s %s %s" % (summary["subtype"], payload.get("api_error_status", ""),
                                          payload.get("result") or payload.get("errors") or "")
    usage = payload.get("usage")
    models = payload.get("modelUsage") or {}
    summary["model_ids"] = list(models)
    summary["usage_records"] = [{"usage": usage, "modelUsage": models}] if payload else []
    if models:
        normalized = [normalize_usage(u, "claude_model") for u in models.values()]
        for field in normalized[0]:
            summary[field] = (all(u[field] for u in normalized) if field == "usage_complete"
                              else sum(u[field] for u in normalized))
    else:
        summary.update(normalize_usage(usage, "claude"))
    _loaded(summary)
    return summary
