"""Single source of truth for the tools the agent may call.

Before this module the tool surface lived in three places that had to be kept
in sync by hand: the JSON schemas written inline in
``core/agent/main.py`` (``_chat_openai_with_tools``), the dispatch ``elif``
chain further down the same method, and the allow/deny policy in
``core/permissions.py``. Adding a tool meant editing all three, and a mismatch
failed silently at call time.

This module mirrors the registry seam DeepSeek Harness establishes with
``ctx.tools`` / ``defineTool`` (packages/core/tools/src/schema.ts): a tool
declares its model-facing name, description and parameter schema, plus the
canonical execution body that is dispatched by name. The system-prompt schema
list reads from here. Cooperative permission decisions remain in permissions.py
and are supplied by each consumer as a pipeline guard.

The registry is deliberately plain data plus one function per entry. It does not
own scheduling, budgets, transcripts or retries; those stay in the agent loop,
exactly as DSH keeps policy and sandboxing in the ``tools/*`` waterfalls rather
than inside each tool body.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.tool_pipeline import DEFAULT_PIPELINE


ToolHandler = Callable[[Any, Mapping[str, Any], Path], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """One registered tool: its model-facing schema and its execution body.

    ``handler`` receives the live agent session, the already-parsed argument
    mapping, and the workspace root, and returns the result dict appended to the
    transcript. Keeping the signature uniform is what lets the loop dispatch by
    name instead of by an ``elif`` ladder.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler


REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    """Register one tool, refusing a conflicting redefinition.

    Re-registering the identical spec is allowed so module-level registration is
    idempotent under repeated imports and test setup; a different spec under an
    existing name is still an error.
    """
    existing = REGISTRY.get(spec.name)
    if existing is not None:
        if existing is spec or (
            existing.description == spec.description and existing.parameters == spec.parameters
        ):
            return existing
        raise ValueError(f"tool already registered: {spec.name}")
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ToolSpec | None:
    return REGISTRY.get(name)


def names() -> list[str]:
    return list(REGISTRY)


def schemas(*, include: Sequence[str] | None = None, exclude: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Return the provider-facing tool schemas in registration order.

    ``include``/``exclude`` let the AutoResearch loop add or hide tools without a
    second schema literal; the loop still decides which tools a given run may see.
    """
    selected = list(REGISTRY.values())
    if include is not None:
        wanted = set(include)
        selected = [spec for spec in selected if spec.name in wanted]
    if exclude:
        hidden = set(exclude)
        selected = [spec for spec in selected if spec.name not in hidden]
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in selected
    ]


def dispatch(session: Any, name: str, arguments: Mapping[str, Any], workspace: Path,
             *, guard: Callable[[], dict | None] | None = None) -> dict[str, Any]:
    """Run the shared stages with the caller's existing approval/control guard."""
    spec = REGISTRY.get(name)
    events = getattr(session, "_tool_events", [])
    first_event = len(events)

    def owner_guard(call):
        cancel = getattr(session, "_cancel_event", None)
        if cancel is not None and cancel.is_set():
            return {"success": False, "executed": False, "error_type": "cancelled", "error": "Execution cancelled."}
        if spec is None:
            return {"success": False, "executed": False, "error_type": "unknown_tool", "error": f"unknown tool: {name}"}
        host_guard = getattr(session, "_tool_guard", None)
        rejection = host_guard(call) if host_guard is not None else None
        return rejection if rejection is not None else (guard() if guard is not None else None)

    pipeline = getattr(session, "_tool_pipeline", DEFAULT_PIPELINE)
    result = pipeline.run(name, arguments, workspace,
                          lambda args: spec.handler(session, args, workspace), guards=(owner_guard,))
    # Handlers keep their existing evidence bookkeeping; publish the final
    # normalized result there as well as to the model/MCP caller.
    for event in events[first_event:]:
        if event.get("tool") == name:
            event["result"] = result
            event["success"] = bool(result.get("success"))
    return result


def canonical_schema_digest() -> str:
    """Stable digest of the registered surface, for drift checks and receipts."""
    import hashlib

    blob = json.dumps(schemas(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
