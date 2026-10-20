"""Driver outcome and direct HTTP loop, with durable per-call evidence."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from pathlib import Path

from neurobench.runner.artifacts import EventLog
from core.tools_impl import SKILL_TOOL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_TOOLS = SKILL_TOOL_NAMES

# The benchmark requires high reasoning in both arms. Output exhaustion must
# be handled through disclosed budgets and retries, never by lowering effort.
# Historical frozen runtimes retain their actual settings independently.
MODEL_DEFAULT_REQUEST_OVERRIDES = {
    "xai/xai/grok-4.7": {"reasoning_effort": "high"},
    "deepseek-v4-pro": {"reasoning_effort": "high"},
    "glm-5.3": {"reasoning_effort": "high"},
    "kimi-k3": {"reasoning_effort": "high"},
}


def _merge_length_responses(previous: str, continuation: str) -> str:
    """Join a bounded length recovery without losing a usable partial answer.

    Providers differ in whether the continuation contains the complete answer,
    only the missing second section, or plain text that follows the cut-off.
    Prefer a complete contract-shaped continuation, otherwise join the pieces
    while avoiding a duplicated first heading.
    """
    previous = previous or ""
    continuation = continuation or ""
    if not previous:
        return continuation
    if not continuation:
        return previous

    h2 = re.compile(r"^##\s+([^\n#]+?)\s*$", re.IGNORECASE | re.MULTILINE)
    continuation_headings = [m.group(1).strip().lower() for m in h2.finditer(continuation)]
    previous_headings = [m.group(1).strip().lower() for m in h2.finditer(previous)]
    required = {"solution thinking", "commands or code"}
    if required.issubset(set(continuation_headings)):
        return continuation

    # A provider may return only the second section after the first response.
    # In that case the two fragments form the contract-shaped answer directly.
    if "commands or code" in continuation_headings and "solution thinking" in previous_headings:
        return previous.rstrip() + "\n\n" + continuation.lstrip()

    # If the continuation repeats the first heading, retain the original
    # section and append the continuation body once.
    if continuation_headings and continuation_headings[0] == "solution thinking" \
            and "solution thinking" in previous_headings:
        first = re.match(r"^##\s+Solution\s+Thinking\s*\n?", continuation,
                         flags=re.IGNORECASE)
        body = continuation[first.end():] if first else continuation
        return previous.rstrip() + "\n\n" + body.lstrip()
    return previous.rstrip() + "\n\n" + continuation.lstrip()


@dataclasses.dataclass
class AttemptOutcome:
    ok: bool
    error: str
    retry_class: str
    session_id: str
    response: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    tool_events: list = dataclasses.field(default_factory=list)
    skills_loaded: list = dataclasses.field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    uncached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    usage_records: list = dataclasses.field(default_factory=list)
    usage_complete: bool = True
    remote_outcome_unknown: bool = False
    request_count: int = 0
    reasoning_effort: str = "provider_default"
    model_ids: list = dataclasses.field(default_factory=list)
    runtime_settings: dict = dataclasses.field(default_factory=dict)
    # Exact provider final text before any contract-only normalization.  This
    # is deliberately excluded from ledger JSON and written as a separate
    # immutable artifact by the campaign runner.
    raw_response: str = ""

    @property
    def tool_event_count(self) -> int:
        return len(self.tool_events)


def normalize_usage(usage, dialect: str) -> dict:
    """Preserve reported counts; absent/invalid required counts are not zero usage.

    OpenAI/Codex input includes cached tokens. Anthropic input excludes cache
    reads and writes, which must be added for comparable total-input accounting.
    """
    complete = isinstance(usage, dict)
    usage = usage if complete else {}

    def count(data, key, *, required=False):
        nonlocal complete
        value = data.get(key, None if required else 0)
        if type(value) is not int or value < 0:
            complete = False
            return 0
        return value

    if dialect == "chat":
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            complete, details = False, {}
        total = count(usage, "prompt_tokens", required=True)
        output = count(usage, "completion_tokens", required=True)
        cached = count({"cached": details.get("cached_tokens", usage.get("cache_read_input_tokens", 0))}, "cached")
        creation = count({"creation": details.get("cache_creation_tokens", details.get("cache_write_tokens",
                                  usage.get("cache_creation_input_tokens", 0)))}, "creation")
        uncached = max(0, total - cached - creation)
        complete = complete and cached + creation <= total
    elif dialect == "codex":
        total = count(usage, "input_tokens", required=True)
        cached = count(usage, "cached_input_tokens")
        creation = 0
        output = count(usage, "output_tokens", required=True)
        uncached = max(0, total - cached)
        complete = complete and cached <= total
    elif dialect in ("claude", "claude_model"):
        fields = (("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens")
                  if dialect == "claude_model" else
                  ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))
        uncached = count(usage, fields[0], required=True)
        cached, creation = count(usage, fields[1]), count(usage, fields[2])
        output = count(usage, fields[3], required=True)
        total = uncached + cached + creation
    else:
        raise ValueError("Unknown usage dialect: " + dialect)
    return dict(input_tokens=total, cached_input_tokens=cached, cache_creation_input_tokens=creation,
                uncached_input_tokens=uncached, output_tokens=output, usage_complete=complete)


def classify_retry(error_text: str, status_code: int | None = None) -> str:
    lowered = (error_text or "").lower()
    if status_code in (401, 403) or re.search(r"\b(401|403)\b|unauthorized|authentication", lowered):
        return "authentication"
    if status_code == 429 or any(t in lowered for t in ("429", "rate limit", "too many requests", "quota")):
        return "rate_limit"
    if any(t in lowered for t in ("timed out", "timeout")):
        return "timeout"
    if status_code in (400, 404, 422) or any(t in lowered for t in ("unsupported", "reasoning effort", "missing credential")):
        return "configuration"
    if any(t in lowered for t in ("answer must", "level-2 headings", "too short", "exactly two")):
        return "contract"
    if status_code in (500, 502, 503, 504) or re.search(r"\b(500|502|503|504)\b|overloaded", lowered):
        return "transient"
    return "unknown"


def resolve_api_key(provider: str) -> str:
    names = {"nvidia": ("NK", "NVIDIA_API_KEY", "NEUROCLAW_NVIDIA_KEY"),
             "deepseek": ("DEEPSEEK_API_KEY", "NEUROCLAW_DEEPSEEK_KEY"),
             "ollama": ("OLLAMA_API_KEY", "NEUROCLAW_OLLAMA_KEY")}[provider]
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    raise RuntimeError("Missing credential for %s; set %s" % (provider, ", ".join(names)))


def _tool_schemas() -> list:
    from core import tool_registry
    from core.tools_impl import register_core_tools
    register_core_tools()
    return tool_registry.schemas(include=SKILL_TOOLS)


def _dispatch_tool(name: str, arguments: dict, workspace: Path, *, skills_dir: Path | None = None, host=None) -> dict:
    if name not in SKILL_TOOLS:
        return {"success": False, "error": "Tool is outside the benchmark allowlist"}
    from core import tool_registry
    from core.mcp_stdio import HarnessHost
    from core.tools_impl import register_core_tools
    register_core_tools()
    host = host if host is not None else HarnessHost(workspace, skills_dir=skills_dir)
    return tool_registry.dispatch(host, name, arguments, host.workspace)


def run_http_chat(*, prompt: str, model: str, base_url: str, provider: str,
                  harness_enabled: bool, workspace: Path, timeout_seconds: int,
                  events_path: Path, max_tool_iterations: int = 12,
                  max_output_tokens: int = 12000, request_overrides: dict | None = None,
                  skills_dir: Path | None = None) -> AttemptOutcome:
    from openai import OpenAI
    import httpx

    key = resolve_api_key(provider)
    transcript = EventLog(events_path, secrets=(key,))
    started = time.monotonic()
    outcome = AttemptOutcome(False, "", "", "", "", 0, 0, 0, 0)
    tools = _tool_schemas() if harness_enabled else None
    from core.mcp_stdio import HarnessHost
    tool_host = HarnessHost(workspace, skills_dir=skills_dir or REPO_ROOT / 'skills') if harness_enabled else None
    messages = [{"role": "user", "content": prompt}]
    overrides = {"reasoning_effort": "high", **MODEL_DEFAULT_REQUEST_OVERRIDES.get(model, {})}
    overrides.update(request_overrides or {})
    omit_temperature = bool(overrides.pop("omit_temperature", False))
    forbidden = set(overrides) & {"messages", "model", "tools", "tool_choice", "stream"}
    if forbidden:
        transcript.close()
        raise ValueError("Request overrides cannot change %s" % sorted(forbidden))
    settings = {"max_tool_iterations": max_tool_iterations, "max_output_tokens": max_output_tokens,
                "temperature": "omitted" if omit_temperature else 0.0, "request_overrides": overrides,
                "automatic_retries": 0, "timeout_seconds": timeout_seconds, "trust_env": False}
    outcome.runtime_settings = settings
    outcome.reasoning_effort = str(overrides.get("reasoning_effort", "provider_default"))
    client = None
    try:
        client = OpenAI(api_key=key, base_url=base_url, max_retries=0,
                        http_client=httpx.Client(trust_env=False, timeout=timeout_seconds))
        transcript.write("configuration", model=model, endpoint=base_url, settings=settings)
        contract_recovery_used = False
        recovery_tools_disabled = False
        partial_response = ""
        tool_iterations = 0
        iteration = 0
        while True:
            if tool_iterations >= max_tool_iterations:
                outcome.error, outcome.retry_class = "HTTP tool loop iteration cap reached", "budget"
                break
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                outcome.error, outcome.retry_class = "Attempt deadline reached between calls", "budget"
                break
            request = {"model": model, "messages": messages}
            request["max_completion_tokens" if model.endswith("gpt-5.6-sol") else "max_tokens"] = max_output_tokens
            if tools and not recovery_tools_disabled:
                request.update(tools=tools, tool_choice="auto")
            if not omit_temperature:
                request["temperature"] = 0.0
            request.update(overrides)
            transcript.write("request", iteration=iteration, payload=request)
            outcome.request_count += 1
            try:
                completion = client.chat.completions.create(**request, timeout=remaining)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                outcome.error = ("%s: %s" % (type(exc).__name__, exc)).replace(key, "[REDACTED]")
                outcome.retry_class = classify_retry(outcome.error, status)
                outcome.remote_outcome_unknown = status is None
                outcome.usage_complete = False
                transcript.write("request_error", iteration=iteration, status_code=status,
                                 error=outcome.error, remote_outcome_unknown=outcome.remote_outcome_unknown)
                break
            payload = completion.model_dump(mode="json", exclude_none=True)
            transcript.write("response", iteration=iteration, payload=payload)
            iteration += 1
            outcome.session_id = payload.get("id") or outcome.session_id
            returned_model = payload.get("model")
            if returned_model and returned_model not in outcome.model_ids:
                outcome.model_ids.append(returned_model)
            usage = payload.get("usage")
            if usage is not None:
                outcome.usage_records.append(usage)
            for field, value in normalize_usage(usage, "chat").items():
                if field == "usage_complete":
                    outcome.usage_complete = outcome.usage_complete and value
                else:
                    setattr(outcome, field, getattr(outcome, field) + value)
            choices = payload.get("choices") or []
            if not choices:
                outcome.error, outcome.retry_class = "Provider returned no choices", "protocol"
                break
            choice = choices[0]
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            if not calls:
                response_text = message.get("content") or ""
                outcome.response = _merge_length_responses(partial_response, response_text) \
                    if partial_response else response_text
                finish_reason = choice.get("finish_reason")
                if finish_reason == "length" and not contract_recovery_used:
                    # A length stop is recoverable within the same paid attempt:
                    # preserve the partial assistant message, ask for only the
                    # missing contract-shaped continuation, and log the turn.
                    contract_recovery_used = True
                    outcome.runtime_settings["contract_recovery_attempts"] = 1
                    recovery_tools_disabled = True
                    outcome.runtime_settings["contract_recovery_tools"] = "disabled"
                    partial_response = outcome.response
                    transcript.write("contract_recovery", iteration=iteration - 1,
                                     reason="finish_reason=length",
                                     continuation="same_attempt_single_bounded_continuation")
                    messages.append({"role": "assistant", "content": outcome.response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "The previous answer hit the output limit before it finished. "
                            "Continue the same answer in this turn. Return only one complete "
                            "final answer starting exactly with `## Solution Thinking` and "
                            "then `## Commands Or Code`; do not add a preamble, repeat the "
                            "task, or add any other level-2 heading. Keep both sections "
                            "substantive and concise."
                        ),
                    })
                    continue
                if finish_reason == "content_filter":
                    outcome.error, outcome.retry_class = "Incomplete response: content_filter", "contract"
                elif finish_reason == "length":
                    outcome.error, outcome.retry_class = "Incomplete response: length after bounded continuation", "contract"
                elif not outcome.response:
                    outcome.error, outcome.retry_class = "HTTP response was empty", "protocol"
                break
            tool_iterations += 1
            # Preserve provider-specific assistant fields, including reasoning_content.
            messages.append(message)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                if not harness_enabled or name not in SKILL_TOOLS:
                    outcome.tool_events.append({"name": name, "ok": False, "blocked": True})
                    outcome.error, outcome.retry_class = "Forbidden tool call: " + name, "policy"
                    break
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Arguments must be an object")
                    result = _dispatch_tool(name, arguments, workspace, skills_dir=skills_dir, host=tool_host)
                except (ValueError, TypeError) as exc:
                    result = {"success": False, "error": str(exc)}
                    arguments = {}
                event = {"name": name, "id": call.get("id"), "arguments": arguments,
                         "ok": result.get("success") is True}
                outcome.tool_events.append(event)
                if name == "read_skill" and event["ok"]:
                    skill = str(result.get("name") or arguments.get("name") or "")
                    if skill and skill not in outcome.skills_loaded:
                        outcome.skills_loaded.append(skill)
                tool_message = {"role": "tool", "tool_call_id": call.get("id"),
                                "content": json.dumps(result, ensure_ascii=False, default=str)}
                transcript.write("tool_result", call=call, message=tool_message)
                messages.append(tool_message)
            if outcome.error:
                break
        outcome.ok = not outcome.error and bool(outcome.response)
    except Exception as exc:
        outcome.error = ("%s: %s" % (type(exc).__name__, exc)).replace(key, "[REDACTED]")
        outcome.retry_class = "runner"
        transcript.write("driver_error", error=outcome.error)
    finally:
        outcome.elapsed_seconds = time.monotonic() - started
        transcript.write("attempt_finished", ok=outcome.ok, error=outcome.error,
                         usage_complete=outcome.usage_complete, request_count=outcome.request_count)
        transcript.close()
        if client is not None:
            client.close()
    return outcome
