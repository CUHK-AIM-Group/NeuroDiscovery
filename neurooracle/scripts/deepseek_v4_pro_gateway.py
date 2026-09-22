"""Local OpenAI-compatible gateway backed by the finite DeepSeek V4 Pro router."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

from neurooracle.scripts.deepseek_v4_pro_router import (
    DEFAULT_KEYS,
    DeepSeekV4ProRouter,
    RouterPaused,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18082


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping):
            text = item.get("text") or item.get("input_text") or item.get("output_text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def normalize_provider_messages(value: Any) -> list[dict[str, Any]]:
    """Translate OpenAI-only message roles to the provider's accepted roles."""

    if not isinstance(value, list):
        raise ValueError("request messages must be a list")
    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        message = dict(item)
        if str(message.get("role") or "user") == "developer":
            message["role"] = "system"
        messages.append(message)
    return messages


def responses_input_to_messages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    value = payload.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return messages
    if not isinstance(value, list):
        raise ValueError("Responses input must be a string or list")
    for item in value:
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "message")
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": _text_content(item.get("output")),
                }
            )
            continue
        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": str(item.get("call_id") or item.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(item.get("name") or ""),
                                "arguments": str(item.get("arguments") or "{}"),
                            },
                        }
                    ],
                }
            )
            continue
        role = str(item.get("role") or "user")
        content = _text_content(item.get("content"))
        message: dict[str, Any] = {"role": role, "content": content}
        if item.get("tool_calls"):
            message["tool_calls"] = item["tool_calls"]
        messages.append(message)
    return messages


def responses_tools_to_chat(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or item.get("type") != "function":
            continue
        if isinstance(item.get("function"), Mapping):
            tools.append(dict(item))
            continue
        function = {
            "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""),
            "parameters": item.get("parameters") or {},
        }
        if "strict" in item:
            function["strict"] = bool(item["strict"])
        tools.append({"type": "function", "function": function})
    return tools or None


def _usage(payload: Mapping[str, Any]) -> dict[str, int]:
    prompt = int(payload.get("prompt_tokens") or payload.get("input_tokens") or 0)
    completion = int(
        payload.get("completion_tokens") or payload.get("output_tokens") or 0
    )
    total = int(payload.get("total_tokens") or prompt + completion)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
    }


def result_to_responses(result: Any) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    tool_calls = result.message.get("tool_calls") or []
    for index, call in enumerate(tool_calls):
        function = call.get("function") or {}
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": str(call.get("id") or f"call_{index}"),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
                "status": "completed",
            }
        )
    if result.content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.content,
                        "annotations": [],
                    }
                ],
            }
        )
    usage = _usage(result.usage)
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "model": "deepseek-v4-pro",
        "output": output,
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
        "metadata": {
            "upstream_channel": result.channel,
            "upstream_key_label": result.key_label,
        },
    }


def result_to_chat(result: Any) -> dict[str, Any]:
    message = dict(result.message)
    if result.thinking:
        message["reasoning_content"] = result.thinking
    usage = _usage(result.usage)
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason or (
                    "tool_calls" if message.get("tool_calls") else "stop"
                ),
            }
        ],
        "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        },
        "system_fingerprint": f"route:{result.channel}:{result.key_label}",
    }


def responses_stream_events(
    response: Mapping[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Build a complete Responses SSE transcript from a buffered result.

    The upstream router deliberately buffers each request so that retry and
    quota decisions remain deterministic.  Frameworks such as BrainPilot
    still consume the OpenAI Responses streaming API, so the gateway replays
    that buffered result as standards-shaped item/delta/done events.
    """

    sequence_number = 0
    events: list[tuple[str, dict[str, Any]]] = []

    def append(event_type: str, **payload: Any) -> None:
        nonlocal sequence_number
        events.append(
            (
                event_type,
                {
                    "type": event_type,
                    "sequence_number": sequence_number,
                    **payload,
                },
            )
        )
        sequence_number += 1

    created_response = dict(response)
    created_response["status"] = "in_progress"
    created_response["output"] = []
    append("response.created", response=created_response)

    for output_index, raw_item in enumerate(response.get("output") or []):
        item = dict(raw_item)
        item_type = item.get("type")
        added_item = dict(item)
        added_item["status"] = "in_progress"
        if item_type == "message":
            added_item["content"] = []
        elif item_type == "function_call":
            added_item["arguments"] = ""
        append(
            "response.output_item.added",
            output_index=output_index,
            item=added_item,
        )

        if item_type == "message":
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
        elif item_type == "function_call":
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

        append(
            "response.output_item.done",
            output_index=output_index,
            item=item,
        )

    append("response.completed", response=dict(response))
    return tuple(events)


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        keys_path: Path,
        checkpoint_dir: Path,
        timeout_seconds: float,
        allow_official_deepseek: bool,
        route_mode: str = "automatic",
        router_pause_http_status: int = HTTPStatus.SERVICE_UNAVAILABLE,
    ) -> None:
        super().__init__(address, GatewayHandler)
        if router_pause_http_status not in {
            HTTPStatus.UNPROCESSABLE_ENTITY,
            HTTPStatus.SERVICE_UNAVAILABLE,
        }:
            raise ValueError("router pause status must be 422 or 503")
        self.keys_path = keys_path
        self.checkpoint_dir = checkpoint_dir
        self.timeout_seconds = timeout_seconds
        self.allow_official_deepseek = allow_official_deepseek
        self.route_mode = route_mode
        self.router_pause_http_status = HTTPStatus(router_pause_http_status)
        self._quota_exhausted_keys: set[str] = set()
        self._quota_lock = threading.Lock()

    def is_quota_exhausted(self, key_label: str) -> bool:
        with self._quota_lock:
            return key_label in self._quota_exhausted_keys

    def mark_quota_exhausted(self, key_label: str) -> None:
        with self._quota_lock:
            self._quota_exhausted_keys.add(key_label)

    def clear_quota_exhausted(self, key_label: str) -> None:
        with self._quota_lock:
            self._quota_exhausted_keys.discard(key_label)

    def router(self, request_id: str) -> DeepSeekV4ProRouter:
        return DeepSeekV4ProRouter(
            keys_path=self.keys_path,
            checkpoint_path=self.checkpoint_dir / f"{request_id}.json",
            timeout_seconds=self.timeout_seconds,
            allow_official_deepseek=self.allow_official_deepseek,
            route_mode=self.route_mode,
            is_quota_exhausted=self.is_quota_exhausted,
            mark_quota_exhausted=self.mark_quota_exhausted,
            clear_quota_exhausted=self.clear_quota_exhausted,
        )

    def automatic_route(self) -> list[str]:
        """Return the finite route plan without exposing credential values."""

        routes = DeepSeekV4ProRouter(
            keys_path=self.keys_path,
            checkpoint_path=None,
            timeout_seconds=self.timeout_seconds,
            allow_official_deepseek=self.allow_official_deepseek,
            route_mode=self.route_mode,
        ).routes()
        return [
            (
                f"{route.key_label}_final_pass"
                if route.pass_number == 2
                else route.key_label
            )
            for route in routes
        ]


class GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Do not log request bodies, authorization headers, or upstream details.
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{stamp}] {self.client_address[0]} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, events: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
        chunks = []
        for event, payload in events:
            chunks.append(
                f"event: {event}\ndata: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"
            )
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
                    "model": "deepseek-v4-pro",
                    "automatic_route": self.server.automatic_route(),
                    "route_mode": self.server.route_mode,
                    "upstream_timeout_seconds": self.server.timeout_seconds,
                    "router_pause_http_status": int(
                        self.server.router_pause_http_status
                    ),
                    "official_deepseek_enabled": self.server.allow_official_deepseek,
                    "secrets_exposed": False,
                },
            )
            return
        if path in {"/models", "/v1/models"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "deepseek-v4-pro",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "neuroclaw-router",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path not in {"/v1/responses", "/responses", "/v1/chat/completions", "/chat/completions"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 16 * 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            request_id = f"gw_{uuid.uuid4().hex}"
            is_responses = path.endswith("responses")
            raw_messages = (
                responses_input_to_messages(payload)
                if is_responses
                else payload.get("messages")
            )
            messages = normalize_provider_messages(raw_messages)
            if not isinstance(messages, list) or not messages:
                raise ValueError("request contains no messages")
            reasoning = payload.get("reasoning") or {}
            effort = str(
                reasoning.get("effort")
                if isinstance(reasoning, Mapping)
                else payload.get("reasoning_effort")
                or "high"
            )
            if effort in {"None", ""}:
                effort = str(payload.get("reasoning_effort") or "high")
            tools = (
                responses_tools_to_chat(payload.get("tools"))
                if is_responses
                else payload.get("tools")
            )
            result = self.server.router(request_id).complete(
                messages=messages,
                reasoning_effort=effort,
                temperature=payload.get("temperature"),
                max_output_tokens=int(
                    payload.get("max_output_tokens")
                    or payload.get("max_tokens")
                    or 8192
                ),
                tools=tools,
                tool_choice=payload.get("tool_choice"),
                response_format=(
                    payload.get("response_format")
                    if isinstance(payload.get("response_format"), dict)
                    else None
                ),
                request_id=request_id,
            )
            response = result_to_responses(result) if is_responses else result_to_chat(result)
            if not bool(payload.get("stream")):
                self._send_json(HTTPStatus.OK, response)
                return
            if is_responses:
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
        except RouterPaused as exc:
            self._send_json(
                self.server.router_pause_http_status,
                {
                    "error": {
                        "type": "router_paused",
                        "message": str(exc),
                        "checkpoint": str(exc.checkpoint) if exc.checkpoint else None,
                    }
                },
            )
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                },
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--keys", type=Path, default=DEFAULT_KEYS)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Safe per-request routing checkpoints; prompts and secrets are excluded.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--router-pause-status",
        type=int,
        choices=(422, 503),
        default=503,
        help=(
            "HTTP status returned after a safely checkpointed finite-route pause; "
            "422 prevents framework-level transient retries."
        ),
    )
    parser.add_argument(
        "--route-mode",
        choices=("automatic", "ollama_only"),
        default="automatic",
        help="Finite upstream route selection; Ollama-only never calls Go.",
    )
    parser.add_argument(
        "--allow-official-deepseek",
        action="store_true",
        help="Explicit opt-in only; disabled in the supplemental protocol.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the credential-bearing gateway may bind only to loopback")
    # Parse and count keys without printing or persisting their values.
    router = DeepSeekV4ProRouter(
        keys_path=args.keys.resolve(),
        checkpoint_path=None,
        timeout_seconds=args.timeout_seconds,
        allow_official_deepseek=args.allow_official_deepseek,
        route_mode=args.route_mode,
    )
    args.checkpoint_dir.resolve().mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "route_count": len(router.routes()),
                    "automatic_route": [
                        route.key_label for route in router.routes()
                    ],
                    "route_mode": args.route_mode,
                    "router_pause_http_status": args.router_pause_status,
                    "official_deepseek_enabled": args.allow_official_deepseek,
                    "secrets_printed_or_persisted": False,
                },
                indent=2,
            )
        )
        return 0
    server = GatewayServer(
        (args.host, args.port),
        keys_path=args.keys.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        timeout_seconds=args.timeout_seconds,
        allow_official_deepseek=args.allow_official_deepseek,
        route_mode=args.route_mode,
        router_pause_http_status=args.router_pause_status,
    )
    print(
        json.dumps(
            {
                "status": "listening",
                "base_url": f"http://{args.host}:{args.port}/v1",
                "model": "deepseek-v4-pro",
                "route_mode": args.route_mode,
                "upstream_timeout_seconds": args.timeout_seconds,
                "official_deepseek_enabled": args.allow_official_deepseek,
                "secrets_printed_or_persisted": False,
            }
        ),
        flush=True,
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
