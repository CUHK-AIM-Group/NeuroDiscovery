"""Audited OpenAI-compatible replay gateway for a Luna Max Codex thread.

The gateway never calls a model or reads provider credentials.  It freezes each
incoming request, renders a deterministic dispatch packet for the designated
Codex thread, and waits for the thread's raw answer to be sealed locally.  A
sealed answer is then wrapped as an OpenAI Chat Completions or Responses result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import os
import sys
import threading
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BENCHMARK_ID = "case2_adni_closed_loop_benchmark_v3_luna_seed1"
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "max"
TRANSPORT = "codex_thread_replay"
DELIVERY_MODE = "message-specified single read-only dispatch file"
RESPONSE_CAPTURE_MODE = "codex-cli output-last-message direct capture"
REQUEST_SCHEMA = "neurooracle.codex_thread_replay_request.v1"
RESPONSE_META_SCHEMA = "neurooracle.codex_thread_replay_response_meta.v1"
RECEIPT_SCHEMA = "neurooracle.codex_thread_replay_receipt.v1"
SUPPORTED_PATHS = {
    "/v1/responses",
    "/responses",
    "/v1/chat/completions",
    "/chat/completions",
}


class ReplayPending(TimeoutError):
    def __init__(self, request_id: str, request_path: Path, dispatch_path: Path) -> None:
        super().__init__(f"Luna thread response is pending for {request_id}")
        self.request_id = request_id
        self.request_path = request_path
        self.dispatch_path = dispatch_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp_{os.getpid()}_{threading.get_ident()}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n")


def normalize_endpoint(path: str) -> str:
    normalized = path.rstrip("/")
    if normalized not in SUPPORTED_PATHS:
        raise ValueError(f"Unsupported endpoint: {normalized}")
    return normalized


def request_identity(endpoint: str, body: Mapping[str, Any]) -> tuple[str, str]:
    material = {"endpoint": normalize_endpoint(endpoint), "body": dict(body)}
    request_sha = sha256_bytes(canonical_json(material).encode("utf-8"))
    return f"luna_{request_sha[:32]}", request_sha


def validate_request_body(body: Mapping[str, Any]) -> None:
    declared = str(body.get("model") or MODEL)
    if declared not in {MODEL, f"openai/{MODEL}"}:
        raise ValueError(f"Request model must be {MODEL}; received {declared}")
    effort: Any = body.get("reasoning_effort")
    reasoning = body.get("reasoning")
    if isinstance(reasoning, Mapping) and reasoning.get("effort") is not None:
        effort = reasoning.get("effort")
    if effort is not None and str(effort) != REASONING_EFFORT:
        raise ValueError(
            f"Request reasoning effort must be {REASONING_EFFORT}; received {effort}"
        )
    if not any(key in body for key in ("messages", "input")):
        raise ValueError("Request contains neither messages nor Responses input")


def render_dispatch(record: Mapping[str, Any]) -> str:
    request_body = json.dumps(record["request_body"], indent=2, ensure_ascii=False)
    return (
        "CS2_LUNA_THREAD_REPLAY_REQUEST_V1\n"
        f"request_id: {record['request_id']}\n"
        f"request_sha256: {record['request_sha256']}\n"
        f"endpoint: {record['endpoint']}\n\n"
        "Treat this turn as one stateless model invocation. Use only the original "
        "API request below and the provider boundary established at thread creation; "
        "do not use facts, candidates, feedback, or answers from any earlier replay turn.\n\n"
        "Return exactly one JSON object with no Markdown fence and no surrounding text:\n"
        "{\n"
        "  \"message\": {\n"
        "    \"role\": \"assistant\",\n"
        "    \"content\": \"string result, or null when making tool calls\",\n"
        "    \"tool_calls\": [\"OpenAI-compatible tool-call objects, or an empty array\"]\n"
        "  },\n"
        "  \"finish_reason\": \"stop or tool_calls\"\n"
        "}\n"
        "Set message.tool_calls to an empty array when no tool is called. Preserve every structured-output "
        "constraint in the original request. Never claim to be DeepSeek.\n\n"
        "ORIGINAL_API_REQUEST_JSON\n"
        f"{request_body}\n"
        "END_ORIGINAL_API_REQUEST_JSON\n"
    )


def validate_envelope(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Thread response must be one JSON object")
    if set(payload) - {"message", "finish_reason"}:
        raise ValueError("Thread response envelope contains unsupported top-level fields")
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Thread response envelope is missing message")
    if str(message.get("role") or "") != "assistant":
        raise ValueError("Thread response message role must be assistant")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("Thread response content must be a string or null")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise ValueError("message.tool_calls must be a list")
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise ValueError("Each tool call must be an object")
            function = call.get("function")
            if str(call.get("type") or "") != "function" or not isinstance(function, Mapping):
                raise ValueError("Only OpenAI-compatible function tool calls are supported")
            if not str(call.get("id") or "") or not str(function.get("name") or ""):
                raise ValueError("Tool call id and function name are required")
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                raise ValueError("Tool-call arguments must be a JSON string")
    finish_reason = str(payload.get("finish_reason") or "")
    if finish_reason not in {"stop", "tool_calls", "length", "content_filter"}:
        raise ValueError("Unsupported finish_reason")
    if tool_calls and finish_reason != "tool_calls":
        raise ValueError("Tool calls require finish_reason=tool_calls")
    normalized_message = dict(message)
    normalized_message.setdefault("content", None if tool_calls else "")
    return {"message": normalized_message, "finish_reason": finish_reason}


def seal_response(
    data_dir: Path,
    request_id: str,
    *,
    thread_id: str,
    host_id: str,
    provider_turn_id: str,
    provider_message_id: str,
) -> dict[str, Any]:
    request_path = data_dir / "requests" / f"{request_id}.json"
    raw_path = data_dir / "responses" / f"{request_id}.txt"
    meta_path = data_dir / "responses" / f"{request_id}.meta.json"
    request = read_json(request_path)
    if request.get("request_id") != request_id:
        raise ValueError("Request ID mismatch")
    if request.get("thread_id") != thread_id or request.get("host_id") != host_id:
        raise ValueError("Provider thread identity differs from the frozen request")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw thread response is missing: {raw_path}")
    raw = raw_path.read_text(encoding="utf-8-sig")
    envelope = validate_envelope(json.loads(raw))
    meta = {
        "schema_version": RESPONSE_META_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "request_id": request_id,
        "request_sha256": request["request_sha256"],
        "thread_id": thread_id,
        "host_id": host_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "dispatch_delivery_mode": DELIVERY_MODE,
        "response_capture_mode": RESPONSE_CAPTURE_MODE,
        "provider_turn_id": provider_turn_id,
        "provider_message_id": provider_message_id,
        "raw_response_relative_path": raw_path.relative_to(data_dir).as_posix(),
        "raw_response_sha256": sha256_file(raw_path),
        "envelope_sha256": sha256_bytes(canonical_json(envelope).encode("utf-8")),
        "sealed_at_utc": utc_now(),
        "manual_content_edit_after_provider_response": False,
    }
    if meta_path.exists():
        existing = read_json(meta_path)
        stable = {key: value for key, value in existing.items() if key != "sealed_at_utc"}
        expected = {key: value for key, value in meta.items() if key != "sealed_at_utc"}
        if stable != expected:
            raise ValueError("Existing response seal differs from the supplied provider response")
        return existing
    atomic_write_json(meta_path, meta)
    return meta


def _response_artifacts(data_dir: Path, record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    request_id = str(record["request_id"])
    raw_path = data_dir / "responses" / f"{request_id}.txt"
    meta_path = data_dir / "responses" / f"{request_id}.meta.json"
    meta = read_json(meta_path)
    checks = {
        "schema_version": RESPONSE_META_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "request_id": request_id,
        "request_sha256": record["request_sha256"],
        "thread_id": record["thread_id"],
        "host_id": record["host_id"],
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "dispatch_delivery_mode": DELIVERY_MODE,
        "response_capture_mode": RESPONSE_CAPTURE_MODE,
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise ValueError(f"Sealed response metadata mismatch: {key}")
    if not raw_path.is_file() or sha256_file(raw_path) != meta.get("raw_response_sha256"):
        raise ValueError("Sealed raw thread response hash mismatch")
    envelope = validate_envelope(json.loads(raw_path.read_text(encoding="utf-8-sig")))
    if sha256_bytes(canonical_json(envelope).encode("utf-8")) != meta.get("envelope_sha256"):
        raise ValueError("Sealed response envelope hash mismatch")
    return envelope, meta, raw_path, meta_path


def chat_response(record: Mapping[str, Any], envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl_{str(record['request_sha256'])[:32]}",
        "object": "chat.completion",
        "created": int(record["created_unix"]),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": dict(envelope["message"]),
                "finish_reason": envelope["finish_reason"],
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "system_fingerprint": f"{TRANSPORT}:{str(record['thread_id'])[:8]}",
    }


def responses_response(record: Mapping[str, Any], envelope: Mapping[str, Any]) -> dict[str, Any]:
    message = envelope["message"]
    output: list[dict[str, Any]] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call["function"]
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{str(record['request_sha256'])[:20]}_{index}",
                "call_id": str(call["id"]),
                "name": str(function["name"]),
                "arguments": str(function["arguments"]),
                "status": "completed",
            }
        )
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{str(record['request_sha256'])[:24]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    return {
        "id": f"resp_{str(record['request_sha256'])[:32]}",
        "object": "response",
        "created_at": int(record["created_unix"]),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": MODEL,
        "output": output,
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "metadata": {
            "transport": TRANSPORT,
            "dispatch_delivery_mode": DELIVERY_MODE,
            "response_capture_mode": RESPONSE_CAPTURE_MODE,
            "request_id": record["request_id"],
            "token_usage_available": False,
        },
    }


def responses_stream_events(
    response: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    sequence = 0
    events: list[tuple[str, dict[str, Any]]] = []

    def append(event_type: str, **payload: Any) -> None:
        nonlocal sequence
        events.append((event_type, {"type": event_type, "sequence_number": sequence, **payload}))
        sequence += 1

    created = dict(response)
    created["status"] = "in_progress"
    created["output"] = []
    append("response.created", response=created)
    for output_index, raw_item in enumerate(response.get("output") or []):
        item = dict(raw_item)
        added = dict(item)
        added["status"] = "in_progress"
        if item.get("type") == "message":
            added["content"] = []
        elif item.get("type") == "function_call":
            added["arguments"] = ""
        append("response.output_item.added", output_index=output_index, item=added)
        if item.get("type") == "message":
            for content_index, part in enumerate(item.get("content") or []):
                if part.get("type") != "output_text":
                    continue
                text = str(part.get("text") or "")
                append(
                    "response.output_text.delta",
                    item_id=item.get("id"),
                    output_index=output_index,
                    content_index=content_index,
                    delta=text,
                    logprobs=[],
                )
                append(
                    "response.output_text.done",
                    item_id=item.get("id"),
                    output_index=output_index,
                    content_index=content_index,
                    text=text,
                    logprobs=[],
                )
        elif item.get("type") == "function_call":
            arguments = str(item.get("arguments") or "{}")
            append(
                "response.function_call_arguments.delta",
                item_id=item.get("id"),
                output_index=output_index,
                delta=arguments,
            )
            append(
                "response.function_call_arguments.done",
                item_id=item.get("id"),
                output_index=output_index,
                arguments=arguments,
            )
        append("response.output_item.done", output_index=output_index, item=item)
    append("response.completed", response=dict(response))
    return tuple(events)


class ReplayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        data_dir: Path,
        thread_id: str,
        host_id: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(address, ReplayHandler)
        self.data_dir = data_dir.resolve()
        self.thread_id = thread_id
        self.host_id = host_id
        self.timeout_seconds = float(timeout_seconds)
        self._active_request = threading.Lock()
        for name in ("requests", "dispatches", "responses", "receipts"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)

    def materialize_request(self, endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
        validate_request_body(body)
        request_id, request_sha = request_identity(endpoint, body)
        request_path = self.data_dir / "requests" / f"{request_id}.json"
        dispatch_path = self.data_dir / "dispatches" / f"{request_id}.txt"
        if request_path.exists():
            record = read_json(request_path)
            if record.get("request_sha256") != request_sha or record.get("request_body") != dict(body):
                raise ValueError("Existing replay request does not match its content hash")
        else:
            record = {
                "schema_version": REQUEST_SCHEMA,
                "benchmark_id": BENCHMARK_ID,
                "request_id": request_id,
                "request_sha256": request_sha,
                "endpoint": normalize_endpoint(endpoint),
                "request_body": dict(body),
                "thread_id": self.thread_id,
                "host_id": self.host_id,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "transport": TRANSPORT,
                "dispatch_delivery_mode": DELIVERY_MODE,
                "response_capture_mode": RESPONSE_CAPTURE_MODE,
                "created_at_utc": utc_now(),
                "created_unix": int(time.time()),
                "authorization_header_persisted": False,
                "provider_keys_used": False,
            }
            atomic_write_json(request_path, record)
        dispatch = render_dispatch(record)
        if dispatch_path.exists():
            if dispatch_path.read_text(encoding="utf-8-sig") != dispatch:
                raise ValueError("Existing dispatch packet differs from the frozen request")
        else:
            atomic_write_text(dispatch_path, dispatch)
        return record

    def wait_for_response(self, record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
        request_id = str(record["request_id"])
        raw_path = self.data_dir / "responses" / f"{request_id}.txt"
        meta_path = self.data_dir / "responses" / f"{request_id}.meta.json"
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if raw_path.is_file() and meta_path.is_file():
                return _response_artifacts(self.data_dir, record)
            time.sleep(0.25)
        raise ReplayPending(
            request_id,
            self.data_dir / "requests" / f"{request_id}.json",
            self.data_dir / "dispatches" / f"{request_id}.txt",
        )

    def write_receipt(
        self,
        record: Mapping[str, Any],
        response: Mapping[str, Any],
        raw_path: Path,
        meta_path: Path,
    ) -> None:
        receipt_path = self.data_dir / "receipts" / f"{record['request_id']}.json"
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "request_id": record["request_id"],
            "request_sha256": record["request_sha256"],
            "raw_response_sha256": sha256_file(raw_path),
            "response_meta_sha256": sha256_file(meta_path),
            "wire_response_sha256": sha256_bytes(canonical_json(response).encode("utf-8")),
            # Imported, already-sealed responses retain the exact provider
            # thread recorded in their frozen request.  Fresh requests use the
            # current gateway thread, so this remains identical in the common
            # case while preserving honest cross-thread migration provenance.
            "thread_id": record["thread_id"],
            "host_id": self.host_id,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "transport": TRANSPORT,
            "response_capture_mode": RESPONSE_CAPTURE_MODE,
            "served_at_utc": utc_now(),
        }
        if receipt_path.exists():
            existing = read_json(receipt_path)
            stable_keys = set(payload) - {"served_at_utc"}
            if any(existing.get(key) != payload.get(key) for key in stable_keys):
                raise ValueError("Existing replay receipt differs from the sealed response")
            return
        atomic_write_json(receipt_path, payload)

    def pending(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for request_path in sorted((self.data_dir / "requests").glob("luna_*.json")):
            request_id = request_path.stem
            meta = self.data_dir / "responses" / f"{request_id}.meta.json"
            if not meta.is_file():
                rows.append(
                    {
                        "request_id": request_id,
                        "request": str(request_path),
                        "dispatch": str(self.data_dir / "dispatches" / f"{request_id}.txt"),
                    }
                )
        return rows


class ReplayHandler(BaseHTTPRequestHandler):
    server: ReplayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        stamp = utc_now()
        print(f"[{stamp}] {self.client_address[0]} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, events: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
        chunks = [
            f"event: {event}\ndata: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"
            for event, payload in events
        ]
        chunks.append("data: [DONE]\n\n")
        body = "".join(chunks).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in {"", "/health", "/v1/health"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "benchmark_id": BENCHMARK_ID,
                    "model": MODEL,
                    "forced_reasoning_effort": REASONING_EFFORT,
                    "execution_channel": TRANSPORT,
                    "max_concurrent_upstream_requests": 1,
                    "provider_api_calls": False,
                    "provider_keys_required": False,
                    "temperature_control": "not_exposed_by_codex_thread",
                    "thread_id": self.server.thread_id,
                    "host_id": self.server.host_id,
                    "request_hash_lock": True,
                    "response_hash_lock": True,
                    "dispatch_delivery_mode": DELIVERY_MODE,
                    "response_capture_mode": RESPONSE_CAPTURE_MODE,
                    "pending_count": len(self.server.pending()),
                    "secrets_exposed": False,
                },
            )
            return
        if path in {"/pending", "/v1/pending"}:
            rows = self.server.pending()
            self._send_json(HTTPStatus.OK, {"object": "list", "data": rows})
            return
        if path in {"/models", "/v1/models"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "codex-thread-replay",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path not in SUPPORTED_PATHS:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        acquired = self.server._active_request.acquire(blocking=False)
        if not acquired:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": {"type": "concurrent_request", "message": "Only one replay request is allowed"}},
            )
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 16 * 1024 * 1024:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("JSON body must be an object")
            record = self.server.materialize_request(path, body)
            envelope, _meta, raw_path, meta_path = self.server.wait_for_response(record)
            is_responses = path.endswith("responses")
            response = (
                responses_response(record, envelope)
                if is_responses
                else chat_response(record, envelope)
            )
            self.server.write_receipt(record, response, raw_path, meta_path)
            if not bool(body.get("stream")):
                self._send_json(HTTPStatus.OK, response)
            elif is_responses:
                self._send_sse(responses_stream_events(response))
            else:
                choice = response["choices"][0]
                self._send_sse(
                    (
                        (
                            "message",
                            {
                                "id": response["id"],
                                "object": "chat.completion.chunk",
                                "created": response["created"],
                                "model": response["model"],
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": choice["message"],
                                        "finish_reason": choice["finish_reason"],
                                    }
                                ],
                            },
                        ),
                    )
                )
        except ReplayPending as exc:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {
                    "error": {
                        "type": "thread_response_pending",
                        "message": str(exc),
                        "request_id": exc.request_id,
                        "request_path": str(exc.request_path),
                        "dispatch_path": str(exc.dispatch_path),
                    }
                },
            )
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"type": type(exc).__name__, "message": str(exc)[:1000]}},
            )
        finally:
            self.server._active_request.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--host-id", default="local")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--seal-response")
    parser.add_argument("--provider-turn-id")
    parser.add_argument("--provider-message-id")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The replay gateway may bind only to loopback")
    data_dir = args.data_dir.resolve()
    if args.seal_response:
        if not args.provider_turn_id or not args.provider_message_id:
            raise ValueError("Provider turn and message IDs are required when sealing a response")
        result = seal_response(
            data_dir,
            args.seal_response,
            thread_id=args.thread_id,
            host_id=args.host_id,
            provider_turn_id=args.provider_turn_id,
            provider_message_id=args.provider_message_id,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    status = {
        "status": "ok" if args.preflight_only else "listening",
        "benchmark_id": BENCHMARK_ID,
        "base_url": f"http://{args.host}:{args.port}/v1",
        "model": MODEL,
        "forced_reasoning_effort": REASONING_EFFORT,
        "execution_channel": TRANSPORT,
        "dispatch_delivery_mode": DELIVERY_MODE,
        "response_capture_mode": RESPONSE_CAPTURE_MODE,
        "thread_id": args.thread_id,
        "host_id": args.host_id,
        "provider_api_calls": False,
        "provider_keys_required": False,
        "temperature_control": "not_exposed_by_codex_thread",
        "max_concurrent_upstream_requests": 1,
        "request_hash_lock": True,
        "response_hash_lock": True,
        "secrets_printed_or_persisted": False,
    }
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0
    server = ReplayServer(
        (args.host, args.port),
        data_dir=data_dir,
        thread_id=args.thread_id,
        host_id=args.host_id,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
