"""Serve audited OpenAI Chat Completions using fresh Luna Max Codex sessions.

Every cache miss starts an independent ``codex exec`` session.  The exact
request is supplied over stdin, the session's final answer is bound through
the local Codex JSONL log, and tool calls or structured content are validated
before an OpenAI-compatible response is returned.  Authorization headers,
prompts, and responses are never printed.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import jsonschema

from core.scripts.case1_codex_session_mixed_import import (
    _load_bound_final_answer,
)
from core.scripts.case1_codex_session_mixed_runner import (
    _find_bound_turn,
    _parse_cli_events,
    _write_json_atomic,
)


MODEL = "gpt-5.6-luna"
EFFORT = "max"
GATEWAY_SCHEMA = "case1-codex-session-chat-gateway.v1"
REQUEST_SCHEMA = "case1-codex-session-chat-request.v1"
RECEIPT_SCHEMA = "case1-codex-session-chat-receipt.v1"
CHAT_PATHS = {"/v1/chat/completions", "/chat/completions"}
MAX_BODY_BYTES = 32 * 1024 * 1024
ENVELOPE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "content": {"type": ["string", "null"]},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    # Codex output schemas are checked with OpenAI's strict
                    # schema validator, which requires an explicit JSON type
                    # even when ``const`` already fixes the only valid value.
                    "type": {"type": "string", "const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "arguments": {"type": "string"},
                        },
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                    },
                },
                "required": ["id", "type", "function"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["content", "tool_calls"],
    "additionalProperties": False,
}


class RequestContractError(ValueError):
    """The caller supplied an unsupported or invalid API request."""


class BackendFailure(RuntimeError):
    """No validated Luna answer was available after bounded attempts."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _available_tools(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(payload.get("tools") or [])
    for function in payload.get("functions") or []:
        rows.append({"type": "function", "function": function})
    available: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "function":
            raise RequestContractError("only function tools are supported")
        function = row.get("function") or {}
        name = str(function.get("name") or "")
        if not name or name in available:
            raise RequestContractError("function tool names must be unique")
        parameters = function.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        if not isinstance(parameters, dict):
            raise RequestContractError(f"invalid parameters schema for tool {name}")
        available[name] = parameters
    return available


def validate_chat_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestContractError("request body must be one JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestContractError("messages must be one non-empty list")
    if any(not isinstance(message, dict) for message in messages):
        raise RequestContractError("every message must be one object")
    if payload.get("stream") not in (None, False, True):
        raise RequestContractError("stream must be boolean when supplied")
    if int(payload.get("n") or 1) != 1:
        raise RequestContractError("only n=1 is supported")
    _available_tools(payload)
    return payload


def _tool_choice(payload: dict[str, Any]) -> Any:
    return payload.get("tool_choice", payload.get("function_call"))


def validate_envelope(
    envelope: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    jsonschema.validate(instance=envelope, schema=ENVELOPE_SCHEMA)
    assert isinstance(envelope, dict)
    tool_calls = envelope["tool_calls"]
    content = envelope["content"]
    available = _available_tools(payload)
    choice = _tool_choice(payload)
    if choice in ("none", {"type": "none"}) and tool_calls:
        raise RequestContractError("tool calls were returned when tools are disabled")
    required_tool = ""
    if choice in ("required", "any") and not tool_calls:
        raise RequestContractError("the request requires at least one tool call")
    if isinstance(choice, dict):
        selected = choice.get("function") or choice
        required_tool = str(selected.get("name") or "")
    seen_ids: set[str] = set()
    for call in tool_calls:
        call_id = str(call["id"])
        if call_id in seen_ids:
            raise RequestContractError("tool call ids must be unique")
        seen_ids.add(call_id)
        function = call["function"]
        name = str(function["name"])
        if name not in available:
            raise RequestContractError(f"unknown tool call: {name}")
        if required_tool and name != required_tool:
            raise RequestContractError(
                f"tool choice requires {required_tool}, got {name}"
            )
        try:
            arguments = json.loads(function["arguments"])
        except json.JSONDecodeError as exc:
            raise RequestContractError(
                f"tool arguments are not JSON for {name}"
            ) from exc
        jsonschema.validate(instance=arguments, schema=available[name])
    if not tool_calls and (not isinstance(content, str) or not content.strip()):
        raise RequestContractError("a non-tool response must have non-empty content")
    response_format = payload.get("response_format") or {}
    response_type = response_format.get("type")
    if response_type in {"json_object", "json_schema"} and not tool_calls:
        try:
            structured = json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise RequestContractError("structured response content is not JSON") from exc
        if response_type == "json_object" and not isinstance(structured, dict):
            raise RequestContractError("json_object response is not one object")
        if response_type == "json_schema":
            wrapper = response_format.get("json_schema") or {}
            schema = wrapper.get("schema", wrapper)
            jsonschema.validate(instance=structured, schema=schema)
    return envelope


def controlled_prompt(payload: dict[str, Any], request_sha256: str) -> str:
    request_text = _canonical_json(payload)
    tool_names = sorted(_available_tools(payload))
    if tool_names:
        tool_contract = (
            "本请求实际注册的 function tool 名称仅限以下 JSON 数组："
            f"{_canonical_json(tool_names)}。tool_calls 中的 name 必须逐字匹配其中一项；"
            "messages 正文里提到但未出现在该数组中的工具名不属于可调用工具。"
        )
    else:
        tool_contract = (
            "本请求没有注册任何 function tool（available_tool_names 为空）。因此最终"
            "tool_calls 必须严格为 []，content 必须是非空普通助手正文。即使 messages "
            "正文提到 SearchSemanticScholar、检索、浏览或其他工具，也不得把它们输出为"
            "函数调用；请仅根据 messages 内已有信息直接作答。"
        )
    metadata = {
        "request_sha256": request_sha256,
        "actual_model": MODEL,
        "reasoning_effort": EFFORT,
        "available_tool_names": tool_names,
        "response_format": payload.get("response_format"),
        "max_tokens": payload.get("max_tokens"),
    }
    return (
        "你是 Case Study 1 的受控 OpenAI Chat Completions 后端。实际模型固定为 "
        "gpt-5.6-luna，reasoning effort=max。下面是一个独立 API 请求；只根据请求中"
        "提供的 messages 和虚拟 function tools 作答。不得使用 Codex 的 shell、文件、"
        "浏览器、网络或任何外部工具；不得读取或推断实验结果、ground truth、effect "
        "size、p/FDR、NeuroDiscovery score 或外部验证结果。请求里的 function tools 只"
        "能通过最终 envelope 的 tool_calls 表示，不能由你实际执行。\n\n"
        f"{tool_contract}\n\n"
        "最终只返回一个 JSON 对象，严格符合 output schema：content 是普通助手正文"
        "（仅在实际提供 tool_calls 时才可为 null），tool_calls 是 OpenAI 格式的函数调用"
        "数组。若 tool_calls 为空，content 必须是非空字符串；不得用 null 或空字符串"
        "表示跳过。若调用函数，"
        "arguments 必须是 JSON 字符串且满足请求给出的参数 schema；不得输出分析、"
        "Markdown 围栏、请求哈希或额外字段。\n\n"
        f"REQUEST_METADATA_JSON\n{_canonical_json(metadata)}\n\n"
        f"BEGIN_OPENAI_CHAT_REQUEST_JSON\n{request_text}\n"
        "END_OPENAI_CHAT_REQUEST_JSON"
    )


def build_chat_response(
    *,
    payload: dict[str, Any],
    envelope: dict[str, Any],
    request_sha256: str,
    created: int,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": envelope["content"],
    }
    if envelope["tool_calls"]:
        message["tool_calls"] = envelope["tool_calls"]
    return {
        "id": f"chatcmpl-case1-{request_sha256[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": str(payload.get("model") or MODEL),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": (
                    "tool_calls" if envelope["tool_calls"] else "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_chat_stream(response: dict[str, Any], payload: dict[str, Any]) -> bytes:
    """Encode one validated completion as OpenAI-compatible SSE chunks.

    The Luna session itself remains non-streaming and is fully validated before
    any bytes are returned.  Native clients such as BrainPilot still receive the
    streaming wire contract they expect, with the complete delta in one event.
    """

    choice = response["choices"][0]
    message = choice["message"]
    common = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": index, **tool_call}
            for index, tool_call in enumerate(message["tool_calls"])
        ]
    chunks: list[dict[str, Any]] = [
        {
            **common,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": None}
            ],
        },
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice["finish_reason"],
                }
            ],
        },
    ]
    stream_options = payload.get("stream_options") or {}
    if isinstance(stream_options, dict) and stream_options.get("include_usage"):
        chunks.append({**common, "choices": [], "usage": response["usage"]})
    records = [
        "data: " + json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        for chunk in chunks
    ]
    records.append("data: [DONE]")
    return ("\n\n".join(records) + "\n\n").encode("utf-8")


def assert_no_codex_tool_use(session_log: Path, turn_id: str) -> None:
    """Reject a backend turn that used Codex-side tools or file operations."""

    allowed = {"UserMessage", "AgentMessage", "Reasoning"}
    disallowed: list[str] = []
    with session_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload") or {}
            if (
                record.get("type") != "event_msg"
                or payload.get("turn_id") != turn_id
                or payload.get("type") != "item_completed"
            ):
                continue
            item_type = str((payload.get("item") or {}).get("type") or "")
            if item_type and item_type not in allowed:
                disallowed.append(item_type)
    if disallowed:
        raise BackendFailure(
            "Codex-side tools are forbidden for API emulation: "
            + ",".join(sorted(set(disallowed)))
        )


class SessionChatGateway:
    def __init__(
        self,
        *,
        state_dir: Path,
        codex_executable: Path,
        provider_workspace: Path,
        session_log_root: Path,
        timeout_seconds: float,
        max_attempts: int,
        max_concurrent_sessions: int,
    ) -> None:
        self.state_dir = state_dir
        self.codex_executable = codex_executable
        self.provider_workspace = provider_workspace
        self.session_log_root = session_log_root
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.max_concurrent_sessions = max(1, max_concurrent_sessions)
        self._request_locks_guard = threading.Lock()
        self._request_locks: dict[str, threading.Lock] = {}
        self._backend_slots = threading.BoundedSemaphore(
            self.max_concurrent_sessions
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.provider_workspace.mkdir(parents=True, exist_ok=True)
        self.envelope_schema_path = self.state_dir / "assistant_envelope.schema.json"
        _write_json_atomic(self.envelope_schema_path, ENVELOPE_SCHEMA)

    def _request_lock(self, request_sha256: str) -> threading.Lock:
        """Serialize only duplicate requests while allowing independent calls."""

        with self._request_locks_guard:
            return self._request_locks.setdefault(request_sha256, threading.Lock())

    def _response_path(self, request_sha256: str) -> Path:
        return self.state_dir / "responses" / f"{request_sha256}.json"

    def _request_path(self, request_sha256: str) -> Path:
        return self.state_dir / "requests" / f"{request_sha256}.json"

    def _receipt_path(self, request_sha256: str) -> Path:
        return self.state_dir / "receipts" / f"{request_sha256}.json"

    def _load_cached(
        self,
        payload: dict[str, Any],
        request_sha256: str,
    ) -> dict[str, Any] | None:
        path = self._response_path(request_sha256)
        if not path.is_file():
            return None
        response = json.loads(path.read_text(encoding="utf-8"))
        message = response["choices"][0]["message"]
        envelope = {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
        }
        validate_envelope(envelope, payload)
        return response

    def _record_request(
        self,
        payload: dict[str, Any],
        request_sha256: str,
    ) -> None:
        path = self._request_path(request_sha256)
        if path.is_file():
            return
        _write_json_atomic(
            path,
            {
                "schema_version": REQUEST_SCHEMA,
                "created_at": time.time(),
                "request_sha256": request_sha256,
                "actual_backend": {"model": MODEL, "reasoning_effort": EFFORT},
                "authorization_header_persisted": False,
                "request": payload,
            },
        )

    def _run_attempt(
        self,
        *,
        payload: dict[str, Any],
        request_sha256: str,
        attempt: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        staging = (
            self.state_dir
            / "staging"
            / f"{request_sha256}.attempt_{attempt:02d}.txt"
        )
        staging.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.codex_executable),
            "exec",
            "--json",
            "-m",
            MODEL,
            "-c",
            'model_reasoning_effort="max"',
            "-s",
            "read-only",
            "--skip-git-repo-check",
            "-C",
            str(self.provider_workspace),
            "--output-schema",
            str(self.envelope_schema_path),
            "-o",
            str(staging),
            "-",
        ]
        started = time.time()
        result = subprocess.run(
            command,
            input=controlled_prompt(payload, request_sha256),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=self.provider_workspace,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise BackendFailure(
                f"Codex CLI exit={result.returncode}; "
                f"stderr_sha256={_sha256_bytes(result.stderr.encode('utf-8'))}"
            )
        cli_audit = _parse_cli_events(result.stdout, None)
        thread_id = str(cli_audit["thread_id"])
        session_log, turn_id = _find_bound_turn(
            log_root=self.session_log_root,
            thread_id=thread_id,
            request_sha256=request_sha256,
        )
        content, message_id, turn_audit = _load_bound_final_answer(
            session_log=session_log,
            thread_id=thread_id,
            turn_id=turn_id,
            request_sha256=request_sha256,
            expected_model=MODEL,
            expected_effort=EFFORT,
        )
        assert_no_codex_tool_use(session_log, turn_id)
        if not staging.is_file():
            raise BackendFailure("Codex CLI did not write its final response")
        if _sha256_bytes(staging.read_bytes()) != _sha256_bytes(
            content.encode("utf-8")
        ):
            raise BackendFailure("session answer differs from CLI output file")
        envelope = validate_envelope(json.loads(content), payload)
        audit = {
            "attempt": attempt,
            "duration_seconds": time.time() - started,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "message_id": message_id,
            "session_log_path": str(session_log),
            "response_content_sha256": _sha256_bytes(content.encode("utf-8")),
            "cli_audit": cli_audit,
            "turn_audit": turn_audit,
        }
        return envelope, audit

    def complete(self, raw_payload: Any) -> tuple[dict[str, Any], bool]:
        payload = validate_chat_request(raw_payload)
        request_sha256 = _sha256_bytes(
            ("chat.completions\n" + _canonical_json(payload)).encode("utf-8")
        )
        with self._request_lock(request_sha256):
            cached = self._load_cached(payload, request_sha256)
            if cached is not None:
                return cached, True
            self._record_request(payload, request_sha256)
            errors: list[dict[str, Any]] = []
            for attempt in range(1, self.max_attempts + 1):
                try:
                    with self._backend_slots:
                        envelope, audit = self._run_attempt(
                            payload=payload,
                            request_sha256=request_sha256,
                            attempt=attempt,
                        )
                    response = build_chat_response(
                        payload=payload,
                        envelope=envelope,
                        request_sha256=request_sha256,
                        created=int(time.time()),
                    )
                    _write_json_atomic(self._response_path(request_sha256), response)
                    _write_json_atomic(
                        self._receipt_path(request_sha256),
                        {
                            "schema_version": RECEIPT_SCHEMA,
                            "status": "validated_and_cached",
                            "request_sha256": request_sha256,
                            "actual_model": MODEL,
                            "reasoning_effort": EFFORT,
                            "logical_request_model": payload.get("model"),
                            "authorization_header_persisted": False,
                            "prompts_or_responses_printed": False,
                            **audit,
                        },
                    )
                    return response, False
                except BaseException as exc:
                    failure = {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                    errors.append(failure)
                    _write_json_atomic(
                        self.state_dir
                        / "rejected"
                        / f"{request_sha256}.attempt_{attempt:02d}.json",
                        {
                            "schema_version": RECEIPT_SCHEMA,
                            "status": "rejected_before_cache_write",
                            "request_sha256": request_sha256,
                            "actual_model": MODEL,
                            "reasoning_effort": EFFORT,
                            "formal_response_cached": False,
                            "prompts_or_responses_printed": False,
                            **failure,
                        },
                    )
            raise BackendFailure(
                f"no validated Luna answer after {self.max_attempts} attempts; "
                f"last={errors[-1]['error_type']}: {errors[-1]['error']}"
            )


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "Case1LunaSessionGateway/1"

    @property
    def gateway(self) -> SessionChatGateway:
        return self.server.gateway  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/health", "/v1/health"}:
            self._send(404, {"error": {"message": "not found"}})
            return
        state = self.gateway.state_dir
        self._send(
            200,
            {
                "status": "ok",
                "model": MODEL,
                "reasoning_effort": EFFORT,
                "cached_responses": len(list((state / "responses").glob("*.json"))),
                "rejected_attempts": len(list((state / "rejected").glob("*.json"))),
                "prompts_or_responses_printed": False,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in CHAT_PATHS:
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > MAX_BODY_BYTES:
                raise RequestContractError("invalid Content-Length")
            payload = json.loads(self.rfile.read(length))
            response, cache_hit = self.gateway.complete(payload)
            self.send_response(200)
            if payload.get("stream") is True:
                body = build_chat_stream(response, payload)
                content_type = "text/event-stream; charset=utf-8"
                self.send_header("Cache-Control", "no-cache")
            else:
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Case1-Actual-Model", MODEL)
            self.send_header("X-Case1-Cache", "hit" if cache_hit else "miss")
            self.end_headers()
            self.wfile.write(body)
        except RequestContractError as exc:
            self._send(400, {"error": {"message": str(exc), "type": "invalid_request"}})
        except BaseException as exc:
            self._send(
                502,
                {
                    "error": {
                        "message": str(exc)[:500],
                        "type": "luna_session_backend_failure",
                    }
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--provider-workspace", type=Path, required=True)
    parser.add_argument(
        "--session-log-root",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--canonical-release", type=Path, required=True)
    parser.add_argument("--sealed-adapter", type=Path, required=True)
    parser.add_argument("--sealed-adapter-sha256", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-concurrent-sessions", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_adapter_sha = args.sealed_adapter_sha256.casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_adapter_sha):
        raise RuntimeError("invalid sealed adapter SHA-256")
    actual_adapter_sha = _sha256_file(args.sealed_adapter)
    if actual_adapter_sha != expected_adapter_sha:
        raise RuntimeError(
            f"sealed adapter hash mismatch: {actual_adapter_sha}"
        )
    task = json.loads(args.task.read_text(encoding="utf-8"))
    registry = Path(str(task["public_registry_path"]))
    release = json.loads(args.canonical_release.read_text(encoding="utf-8"))
    gateway = SessionChatGateway(
        state_dir=args.state_dir,
        codex_executable=args.codex,
        provider_workspace=args.provider_workspace,
        session_log_root=args.session_log_root,
        timeout_seconds=max(30.0, args.timeout_seconds),
        max_attempts=max(1, args.max_attempts),
        max_concurrent_sessions=max(1, args.max_concurrent_sessions),
    )
    _write_json_atomic(
        args.state_dir / "gateway_manifest.json",
        {
            "schema_version": GATEWAY_SCHEMA,
            "created_at": time.time(),
            "actual_model": MODEL,
            "reasoning_effort": EFFORT,
            "transport": "fresh_codex_session_per_api_request",
            "max_concurrent_sessions": gateway.max_concurrent_sessions,
            "task_path": str(args.task),
            "task_sha256": _sha256_file(args.task),
            "registry_path": str(registry),
            "registry_sha256": _sha256_file(registry),
            "canonical_release_path": str(args.canonical_release),
            "canonical_release": release,
            "gateway_source_path": str(Path(__file__).resolve()),
            "gateway_source_sha256": _sha256_file(Path(__file__).resolve()),
            "tool_contract": "exact_registered_names_or_empty_v1",
            "sealed_adapter_path": str(args.sealed_adapter),
            "sealed_adapter_sha256": actual_adapter_sha,
            "authorization_headers_persisted": False,
            "prompts_or_responses_printed": False,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    server.gateway = gateway  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "status": "listening",
                "host": args.host,
                "port": args.port,
                "model": MODEL,
                "reasoning_effort": EFFORT,
                "prompts_or_responses_printed": False,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
