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
canonical execution body that is dispatched by name. Everything else, the
system-prompt schema list and the permission lookup included, reads from here
instead of keeping a second copy.

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


def dispatch(session: Any, name: str, arguments: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    """Run one tool body, converting an unexpected exception into a tool error.

    A missing tool and a crash inside one tool must both surface as an ordinary
    result the model can read, never as an exception that ends the turn. That
    matches DSH's registry-level normalization, where a throwing pipeline becomes
    ``isError`` rather than an aborted turn.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return {"success": False, "executed": False, "error_type": "unknown_tool", "error": f"unknown tool: {name}"}
    try:
        return spec.handler(session, arguments, workspace)
    except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
        return {
            "success": False,
            "executed": False,
            "error_type": "tool_execution_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def canonical_schema_digest() -> str:
    """Stable digest of the registered surface, for drift checks and receipts."""
    import hashlib

    blob = json.dumps(schemas(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
