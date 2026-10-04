"""Synchronous tool stages shared by the agent, direct harness and MCP.

Definitions: immutable ToolCall and callback contracts. Provider: ToolPipeline.
Consumer: tool_registry.dispatch. Policy remains owned by the caller's guard;
hooks cannot change the identity/arguments of a call after it was authorized.
This is a cooperative in-process boundary, not an OS sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.tool_outcomes import normalize_shell_outcome


def _snapshot(value):
    # Reject unsupported objects and NaN instead of silently stringifying them.
    def check(item):
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise TypeError("Tool data must be lossless JSON with string object keys")
    check(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    workspace: Path


Check = Callable[[ToolCall], dict | None]
Around = Callable[[ToolCall, Callable[[], dict]], dict]
Post = Callable[[ToolCall, dict], dict | None]
Observer = Callable[[ToolCall, Mapping[str, Any]], None]


def normalize_result(call: ToolCall, result: dict) -> dict:
    if call.name == "run_shell_command" and result.get("executed", True) is not False:
        return normalize_shell_outcome(str(call.arguments.get("command") or ""), result)
    return result


@dataclass(frozen=True)
class ToolPipeline:
    """Five stages, configured per host/session rather than by global hooks.

    Pre-execute checks and guards may return a denial or abstain with None.
    Execute wrappers can measure/short-circuit one dispatch, never replay it.
    Post-execute transforms a detached result; denial stays authoritative.
    Result observers receive a recursively immutable final snapshot. Observer
    failures are logged and cannot change a result already published to peers.
    """

    pre_execute: tuple[Check, ...] = ()
    guards: tuple[Check, ...] = ()
    execute: tuple[Around, ...] = ()
    post_execute: tuple[Post, ...] = ()
    result: tuple[Observer, ...] = ()

    def run(self, name: str, arguments: Mapping, workspace: Path,
            body: Callable[[dict], dict], *, guards: tuple[Check, ...] = ()) -> dict:
        called = False
        denied = None
        stage = "pre_execute"
        call = ToolCall(name, MappingProxyType({}), Path(workspace))
        try:
            if not isinstance(arguments, Mapping):
                raise ValueError("Tool arguments must be an object")
            original = _snapshot(dict(arguments))
            call = ToolCall(name, _freeze(original), Path(workspace))
            for stage, checks in (("pre_execute", self.pre_execute), ("guards", self.guards + guards)):
                for check in checks:
                    rejection = check(call)
                    if rejection is not None:
                        if not isinstance(rejection, dict) or rejection.get("success") is not False:
                            raise ValueError("A guard may only return a failure or None")
                        denied = _snapshot(rejection)
                        denied["executed"] = False
                        break
                if denied is not None:
                    break
            stage = "execute"
            if denied is None:
                def dispatch_once():
                    nonlocal called
                    if called:
                        raise RuntimeError("A tool body cannot be replayed inside a pipeline")
                    called = True
                    try:
                        return body(_snapshot(original))
                    except Exception as exc:
                        return {"success": False, "executed": True,
                                "error_type": "tool_execution_error", "outcome_unknown": True,
                                "error": f"{type(exc).__name__}: {exc}"}

                invoke = dispatch_once
                for wrapper in reversed(self.execute):
                    previous = invoke
                    invoke = lambda wrapper=wrapper, previous=previous: wrapper(call, previous)
                candidate = invoke()
            else:
                candidate = denied
            stage = "post_execute"
            candidate = _snapshot(candidate)
            if not isinstance(candidate, dict):
                raise TypeError("Tool result must be an object")
            for transform in self.post_execute:
                updated = transform(call, candidate)
                if updated is not None:
                    candidate = updated
                candidate = _snapshot(candidate)
                if not isinstance(candidate, dict):
                    raise TypeError("Post-execute result must be an object")
            # The existing shell business-failure check is a final invariant.
            candidate = normalize_result(call, candidate)
            if denied is not None:
                candidate.update(denied)
            final = _snapshot(candidate)
        except Exception as exc:
            final = {"success": False, "executed": called, "error_type": "tool_pipeline_error",
                     "failure_stage": stage, "error": f"{type(exc).__name__}: {exc}"}
            if called:
                final["outcome_unknown"] = True
            if denied is not None:
                final.update(denied)
        frozen = _freeze(final)
        for observe in self.result:
            try:
                observe(call, frozen)
            except Exception:
                logging.getLogger(__name__).exception("Tool result observer failed for %s", name)
        return _snapshot(final)


DEFAULT_PIPELINE = ToolPipeline()
