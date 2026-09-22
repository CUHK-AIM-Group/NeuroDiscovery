"""Production DeepSeek V4 Pro routing for supplemental Case Study jobs.

Secrets are read from ``Downloads/keys.txt`` for each router process and are
never included in checkpoints, exceptions, logs, or command-line arguments.
The automatic route is deliberately finite:

OpenCode Go key 1 -> key 2 -> key 3 -> Ollama Cloud -> one final Go pass.

A route change is allowed only after confirmed quota exhaustion. Network
errors and HTTP 500/502/503/504 stay on the same route and retry after
2, 8, and 30 seconds. DeepSeek's official endpoint is opt-in and is never an
automatic fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from neurooracle.scripts.probe_deepseek_v4_pro_channels import load_channel_keys


DEFAULT_KEYS = Path.home() / "Downloads" / "keys.txt"
ROUTE_MODES = ("automatic", "ollama_only")
RETRY_DELAYS_SECONDS = (2.0, 8.0, 30.0)
TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})
USER_AGENT = "NeuroClaw-DeepSeek-V4-Pro-Router/1.0"

OPENCODE_GO_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
OLLAMA_CLOUD_ENDPOINT = "https://ollama.com/api/chat"
DEEPSEEK_OFFICIAL_ENDPOINT = "https://api.deepseek.com/chat/completions"

_QUOTA_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"insufficient[_ -]?quota",
        r"quota (?:is )?(?:exceeded|exhausted|depleted)",
        r"insufficient (?:balance|credits?)",
        r"(?:credit|account) balance",
        r"billing quota",
        r"out of credits?",
        r"\busage limit (?:has been )?reached\b",
        r"\breached (?:your |the )?(?:(?:session|weekly|daily|monthly|\d+[- ]?hour) )?usage limit\b",
        r"额度(?:不足|耗尽)",
        r"配额(?:不足|耗尽|已用完)",
        r"余额不足",
    )
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _safe_error_text(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("message") or value.get("error") or json.dumps(
            dict(value), ensure_ascii=False
        )
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", text)
    text = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]+", "<redacted>", text)
    text = re.sub(r"\bwrk_[A-Za-z0-9]+\b", "wrk_<redacted>", text)
    return text[:1000]


def _error_detail(payload: Mapping[str, Any] | None, fallback: str = "") -> str:
    if payload:
        error = payload.get("error")
        if isinstance(error, Mapping):
            return _safe_error_text(error.get("message") or error)
        if error:
            return _safe_error_text(error)
        message = payload.get("message")
        if isinstance(message, str):
            return _safe_error_text(message)
    return _safe_error_text(fallback)


def _safe_response_shape(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Describe a provider response without persisting generated text."""

    if not isinstance(payload, Mapping):
        return {"payload_type": type(payload).__name__}
    shape: dict[str, Any] = {
        "top_level_keys": sorted(str(key) for key in payload),
        "error_present": bool(payload.get("error")),
    }
    choices = payload.get("choices")
    shape["choices_type"] = type(choices).__name__
    shape["choices_count"] = len(choices) if isinstance(choices, list) else 0
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        shape["wire_format"] = "openai_chat"
        shape["choice_keys"] = sorted(str(key) for key in choice)
        shape["finish_reason"] = choice.get("finish_reason")
        message = choice.get("message")
        shape["message_type"] = type(message).__name__
        if isinstance(message, Mapping):
            shape["message_keys"] = sorted(str(key) for key in message)
            for field in ("content", "reasoning_content", "thinking", "reasoning"):
                value = message.get(field)
                shape[f"{field}_type"] = type(value).__name__
                shape[f"{field}_characters"] = len(_content_text(value))
            tool_calls = message.get("tool_calls")
            shape["tool_calls_count"] = (
                len(tool_calls) if isinstance(tool_calls, list) else 0
            )
    elif isinstance(payload.get("message"), Mapping):
        message = payload["message"]
        shape["wire_format"] = "ollama_chat"
        shape["done"] = payload.get("done")
        shape["done_reason"] = payload.get("done_reason")
        shape["prompt_eval_count"] = payload.get("prompt_eval_count")
        shape["eval_count"] = payload.get("eval_count")
        shape["message_type"] = type(message).__name__
        shape["message_keys"] = sorted(str(key) for key in message)
        for field in ("content", "reasoning_content", "thinking", "reasoning"):
            value = message.get(field)
            shape[f"{field}_type"] = type(value).__name__
            shape[f"{field}_characters"] = len(_content_text(value))
        tool_calls = message.get("tool_calls")
        shape["tool_calls_count"] = (
            len(tool_calls) if isinstance(tool_calls, list) else 0
        )
    return shape


def _messages_for_ollama(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI tool-call history to Ollama's native message shape."""

    converted: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    for raw_message in messages:
        message = dict(raw_message)
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            calls: list[dict[str, Any]] = []
            for raw_call in raw_calls:
                if not isinstance(raw_call, Mapping):
                    continue
                call = dict(raw_call)
                raw_function = call.get("function")
                if isinstance(raw_function, Mapping):
                    function = dict(raw_function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            decoded = json.loads(arguments)
                        except json.JSONDecodeError:
                            decoded = arguments
                        if isinstance(decoded, Mapping):
                            function["arguments"] = dict(decoded)
                    call["function"] = function
                    call_id = str(call.get("id") or "")
                    name = str(function.get("name") or "")
                    if call_id and name:
                        tool_names[call_id] = name
                calls.append(call)
            message["tool_calls"] = calls
        if str(message.get("role") or "") == "tool" and not message.get("tool_name"):
            call_id = str(message.get("tool_call_id") or "")
            if call_id and call_id in tool_names:
                message["tool_name"] = tool_names[call_id]
        converted.append(message)
    return converted


def _tool_calls_for_openai(value: Any, request_id: str) -> list[dict[str, Any]]:
    """Convert Ollama tool calls back to the OpenAI-compatible wire shape."""

    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(value):
        if not isinstance(raw_call, Mapping):
            continue
        call = dict(raw_call)
        call.setdefault("id", f"call_{request_id}_{index}")
        call.setdefault("type", "function")
        raw_function = call.get("function")
        if isinstance(raw_function, Mapping):
            function = dict(raw_function)
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(
                    arguments if arguments is not None else {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            call["function"] = function
        calls.append(call)
    return calls


def _increase_output_budget_after_length(
    route: Route,
    provider_payload: dict[str, Any],
    response_payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Increase only an exhausted output budget, without changing channels."""

    if not isinstance(response_payload, Mapping):
        return None
    if route.channel == "ollama_cloud":
        finish_reason = response_payload.get("done_reason")
        field = "num_predict"
        options = dict(provider_payload.get("options") or {})
        previous = int(options.get(field) or 8192)
        target = min(65536, max(previous + 1024, previous * 2))
        if finish_reason != "length" or target <= previous:
            return None
        options[field] = target
        provider_payload["options"] = options
    else:
        choices = response_payload.get("choices")
        finish_reason = (
            choices[0].get("finish_reason")
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], Mapping)
            else None
        )
        field = "max_tokens"
        previous = int(provider_payload.get(field) or 8192)
        target = min(65536, max(previous + 1024, previous * 2))
        if finish_reason != "length" or target <= previous:
            return None
        provider_payload[field] = target
    return {"field": field, "previous": previous, "next": target}


def confirmed_quota_exhaustion(
    status: int | None,
    payload: Mapping[str, Any] | None,
    detail: str,
) -> bool:
    """Return true only for a billing/quota exhaustion signal, not rate limiting."""

    if status == 402:
        return True
    if status not in {403, 429}:
        return False
    combined = " ".join(
        part
        for part in (
            detail,
            _error_detail(payload),
            str((payload or {}).get("code") or ""),
            str(((payload or {}).get("error") or {}).get("code") or "")
            if isinstance((payload or {}).get("error"), Mapping)
            else "",
        )
        if part
    )
    return any(pattern.search(combined) for pattern in _QUOTA_PATTERNS)


@dataclass(frozen=True)
class Route:
    channel: str
    key_group: str
    key_index: int
    key_label: str
    pass_number: int
    endpoint: str
    model: str


@dataclass(frozen=True)
class TransportResult:
    status: int | None
    payload: dict[str, Any] | None
    error_kind: str | None = None
    error_detail: str = ""


@dataclass(frozen=True)
class ChatResult:
    channel: str
    key_label: str
    model: str
    message: dict[str, Any]
    finish_reason: str | None
    usage: dict[str, Any]
    thinking: str
    request_id: str

    @property
    def content(self) -> str:
        return _content_text(self.message.get("content"))


class RouterPaused(RuntimeError):
    """The finite route paused and persisted a safe checkpoint."""

    def __init__(self, reason: str, checkpoint: Path | None = None):
        super().__init__(reason)
        self.checkpoint = checkpoint


Transport = Callable[[str, str, Mapping[str, Any], float], TransportResult]


def urllib_transport(
    endpoint: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> TransportResult:
    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(16 * 1024 * 1024).decode("utf-8", "replace")
            status = int(response.status)
    except HTTPError as exc:
        raw = exc.read(2 * 1024 * 1024).decode("utf-8", "replace")
        status = int(exc.code)
        error_kind = "http_error"
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        return TransportResult(
            status=None,
            payload=None,
            error_kind="network_error",
            error_detail=_safe_error_text(getattr(exc, "reason", exc)),
        )
    else:
        error_kind = None

    try:
        parsed = json.loads(raw)
        payload_dict = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        payload_dict = None
    return TransportResult(
        status=status,
        payload=payload_dict,
        error_kind=error_kind,
        error_detail="" if payload_dict is not None else _safe_error_text(raw),
    )


class DeepSeekV4ProRouter:
    def __init__(
        self,
        *,
        keys_path: Path = DEFAULT_KEYS,
        checkpoint_path: Path | None = None,
        timeout_seconds: float = 300.0,
        retry_delays: Sequence[float] = RETRY_DELAYS_SECONDS,
        allow_official_deepseek: bool = False,
        route_mode: str = "automatic",
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
        is_quota_exhausted: Callable[[str], bool] | None = None,
        mark_quota_exhausted: Callable[[str], None] | None = None,
        clear_quota_exhausted: Callable[[str], None] | None = None,
    ) -> None:
        keys = load_channel_keys(keys_path)
        if len(keys["opencode"]) != 3:
            raise ValueError("expected exactly three OpenCode Go keys")
        if route_mode not in ROUTE_MODES:
            raise ValueError(f"unsupported route mode: {route_mode}")
        if route_mode == "ollama_only" and allow_official_deepseek:
            raise ValueError(
                "official DeepSeek cannot be enabled in Ollama-only mode"
            )
        self._keys = keys
        self._all_secrets = tuple(
            secret for values in keys.values() for secret in values
        )
        self.checkpoint_path = checkpoint_path
        self.timeout_seconds = float(timeout_seconds)
        self.retry_delays = tuple(float(value) for value in retry_delays)
        self.allow_official_deepseek = bool(allow_official_deepseek)
        self.route_mode = route_mode
        self.transport = transport
        self.sleep = sleep
        self.is_quota_exhausted = is_quota_exhausted or (lambda _key_label: False)
        self.mark_quota_exhausted = mark_quota_exhausted or (lambda _key_label: None)
        self.clear_quota_exhausted = clear_quota_exhausted or (lambda _key_label: None)
        self._checkpoint_lock = threading.Lock()

    def _scrub_secrets(self, value: str) -> str:
        safe = _safe_error_text(value)
        for secret in self._all_secrets:
            if secret:
                safe = safe.replace(secret, "<redacted>")
        return safe

    def routes(self) -> tuple[Route, ...]:
        go_first = tuple(
            Route(
                channel="opencode_go",
                key_group="opencode",
                key_index=index,
                key_label=f"opencode_go_key_{index + 1}",
                pass_number=1,
                endpoint=OPENCODE_GO_ENDPOINT,
                model="deepseek-v4-pro",
            )
            for index in range(3)
        )
        ollama = tuple(
            Route(
                channel="ollama_cloud",
                key_group="ollama",
                key_index=index,
                key_label=f"ollama_cloud_key_{index + 1}",
                pass_number=1,
                endpoint=OLLAMA_CLOUD_ENDPOINT,
                model="deepseek-v4-pro:cloud",
            )
            for index in range(len(self._keys["ollama"]))
        )
        go_final = tuple(
            Route(
                channel="opencode_go",
                key_group="opencode",
                key_index=index,
                key_label=f"opencode_go_key_{index + 1}",
                pass_number=2,
                endpoint=OPENCODE_GO_ENDPOINT,
                model="deepseek-v4-pro",
            )
            for index in range(3)
        )
        if self.route_mode == "ollama_only":
            return ollama
        routes: tuple[Route, ...] = (*go_first, *ollama, *go_final)
        if self.allow_official_deepseek:
            routes = (
                *routes,
                Route(
                    channel="deepseek_official",
                    key_group="deepseek",
                    key_index=0,
                    key_label="deepseek_official_key_1",
                    pass_number=1,
                    endpoint=DEEPSEEK_OFFICIAL_ENDPOINT,
                    model="deepseek-v4-pro",
                ),
            )
        return routes

    @staticmethod
    def _payload(
        route: Route,
        *,
        messages: Sequence[Mapping[str, Any]],
        reasoning_effort: str,
        temperature: float | None,
        max_output_tokens: int,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: Any,
        response_format: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        provider_messages = [dict(message) for message in messages]
        requested_response_type = (
            str(response_format.get("type") or "") if response_format else ""
        )
        if route.channel == "opencode_go" and requested_response_type == "json_schema":
            json_schema = response_format.get("json_schema") if response_format else None
            schema = (
                json_schema.get("schema")
                if isinstance(json_schema, Mapping)
                else None
            )
            if isinstance(schema, Mapping):
                schema_instruction = (
                    "Return one JSON object only. It must conform exactly to this "
                    "JSON Schema; deterministic validation will reject violations:\n"
                    + json.dumps(schema, ensure_ascii=False, sort_keys=True)
                )
                if provider_messages and provider_messages[0].get("role") == "system":
                    provider_messages[0]["content"] = (
                        _content_text(provider_messages[0].get("content"))
                        + "\n\n"
                        + schema_instruction
                    )
                else:
                    provider_messages.insert(
                        0,
                        {"role": "system", "content": schema_instruction},
                    )
        elif (
            route.channel == "opencode_go"
            and requested_response_type == "json_object"
            and not any(
                re.search(r"\bjson\b", _content_text(message.get("content")), re.IGNORECASE)
                for message in provider_messages
            )
        ):
            provider_messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Return one valid JSON object only.",
                },
            )
        if route.channel == "ollama_cloud":
            provider_messages = _messages_for_ollama(provider_messages)
            payload: dict[str, Any] = {
                "model": route.model,
                "messages": provider_messages,
                "think": reasoning_effort,
                "stream": False,
                "options": {"num_predict": int(max_output_tokens)},
            }
            if temperature is not None:
                payload["options"]["temperature"] = float(temperature)
            if response_format:
                response_type = str(response_format.get("type") or "")
                if response_type == "json_schema":
                    json_schema = response_format.get("json_schema")
                    if isinstance(json_schema, Mapping) and isinstance(
                        json_schema.get("schema"), Mapping
                    ):
                        payload["format"] = dict(json_schema["schema"])
                elif response_type == "json_object":
                    payload["format"] = "json"
        else:
            payload = {
                "model": route.model,
                "messages": provider_messages,
                "thinking": {"type": "enabled"},
                "reasoning_effort": reasoning_effort,
                "stream": False,
                "max_tokens": int(max_output_tokens),
            }
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if response_format:
                payload["response_format"] = (
                    {"type": "json_object"}
                    if route.channel == "opencode_go"
                    and requested_response_type == "json_schema"
                    else dict(response_format)
                )
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    @staticmethod
    def _parse_success(route: Route, payload: Mapping[str, Any], request_id: str) -> ChatResult:
        if route.channel == "ollama_cloud":
            raw_message = payload.get("message") or {}
            if not isinstance(raw_message, Mapping):
                raise ValueError("Ollama Cloud returned no message object")
            message = dict(raw_message)
            thinking = _content_text(message.pop("thinking", ""))
            if message.get("tool_calls"):
                message["tool_calls"] = _tool_calls_for_openai(
                    message.get("tool_calls"), request_id
                )
            finish_reason = str(payload.get("done_reason") or "stop")
            usage = {
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count"),
                "total_tokens": (
                    int(payload.get("prompt_eval_count") or 0)
                    + int(payload.get("eval_count") or 0)
                ),
            }
        else:
            choices = payload.get("choices") or []
            if not isinstance(choices, list) or not choices:
                raise ValueError("chat completion returned no choices")
            choice = choices[0]
            if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
                raise ValueError("chat completion returned no message object")
            message = dict(choice["message"])
            thinking = _content_text(
                message.pop("reasoning_content", message.pop("thinking", ""))
            )
            finish_reason = (
                str(choice.get("finish_reason"))
                if choice.get("finish_reason") is not None
                else None
            )
            usage = dict(payload.get("usage") or {})
        if not _content_text(message.get("content")) and not message.get("tool_calls"):
            raise ValueError("provider returned neither content nor tool calls")
        message.setdefault("role", "assistant")
        return ChatResult(
            channel=route.channel,
            key_label=route.key_label,
            model=str(payload.get("model") or route.model),
            message=message,
            finish_reason=finish_reason,
            usage=usage,
            thinking=thinking,
            request_id=request_id,
        )

    def _write_checkpoint(self, payload: Mapping[str, Any]) -> None:
        if self.checkpoint_path is None:
            return
        path = self.checkpoint_path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._checkpoint_lock:
            temporary.write_text(
                json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        reasoning_effort: str = "high",
        temperature: float | None = None,
        max_output_tokens: int = 8192,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> ChatResult:
        if not messages:
            raise ValueError("messages cannot be empty")
        request_id = request_id or f"ds4p_{time.time_ns():x}"
        request_fingerprint = _sha256_text(
            json.dumps(list(messages), sort_keys=True, ensure_ascii=False, default=str)
        )
        attempts: list[dict[str, Any]] = []
        started_at = utc_now()

        for route in self.routes():
            if route.pass_number == 1 and self.is_quota_exhausted(route.key_label):
                attempts.append(
                    {
                        "channel": route.channel,
                        "key_label": route.key_label,
                        "pass_number": route.pass_number,
                        "retry_index": None,
                        "http_status": None,
                        "elapsed_ms": 0.0,
                        "error_kind": "cached_quota_exhaustion",
                        "confirmed_quota_exhaustion": True,
                        "error_detail": None,
                    }
                )
                continue
            key = self._keys[route.key_group][route.key_index]
            provider_payload = self._payload(
                route,
                messages=messages,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
            retries = (0.0, *self.retry_delays)
            for retry_index, delay in enumerate(retries):
                if delay:
                    self.sleep(delay)
                began = time.perf_counter()
                result = self.transport(
                    route.endpoint,
                    key,
                    provider_payload,
                    self.timeout_seconds,
                )
                elapsed_ms = round((time.perf_counter() - began) * 1000, 1)
                detail = self._scrub_secrets(
                    _error_detail(result.payload, result.error_detail)
                )
                quota = confirmed_quota_exhaustion(
                    result.status,
                    result.payload,
                    detail,
                )
                attempt = {
                    "channel": route.channel,
                    "key_label": route.key_label,
                    "pass_number": route.pass_number,
                    "retry_index": retry_index,
                    "http_status": result.status,
                    "elapsed_ms": elapsed_ms,
                    "error_kind": result.error_kind,
                    "confirmed_quota_exhaustion": quota,
                    "error_detail": detail or None,
                }
                if response_format:
                    requested_type = str(response_format.get("type") or "")
                    attempt["response_format_requested"] = requested_type
                    attempt["response_format_transmitted"] = (
                        "json_object"
                        if route.channel == "opencode_go"
                        and requested_type == "json_schema"
                        else requested_type
                    )
                    attempt["json_schema_embedded_in_prompt"] = bool(
                        route.channel == "opencode_go"
                        and requested_type == "json_schema"
                    )
                attempts.append(attempt)

                if result.status is not None and 200 <= result.status < 300:
                    try:
                        parsed = self._parse_success(
                            route,
                            result.payload or {},
                            request_id,
                        )
                    except Exception as exc:
                        attempt["error_kind"] = "invalid_response"
                        attempt["error_detail"] = self._scrub_secrets(str(exc))
                        attempt["response_shape"] = _safe_response_shape(result.payload)
                        if retry_index + 1 < len(retries):
                            adjustment = _increase_output_budget_after_length(
                                route, provider_payload, result.payload
                            )
                            if adjustment is not None:
                                attempt["retry_adjustment"] = adjustment
                            continue
                        reason = (
                            f"invalid provider response on {route.key_label}: "
                            f"{type(exc).__name__}"
                        )
                        self._write_checkpoint(
                            {
                                "schema_version": "neuroclaw.deepseek-v4-pro-router-checkpoint.v1",
                                "status": "paused_invalid_response",
                                "started_at": started_at,
                                "paused_at": utc_now(),
                                "request_id": request_id,
                                "request_sha256": request_fingerprint,
                                "attempts": attempts,
                                "reason": reason,
                                "secrets_persisted": False,
                            }
                        )
                        raise RouterPaused(reason, self.checkpoint_path) from exc
                    self._write_checkpoint(
                        {
                            "schema_version": "neuroclaw.deepseek-v4-pro-router-checkpoint.v1",
                            "status": "complete",
                            "started_at": started_at,
                            "completed_at": utc_now(),
                            "request_id": request_id,
                            "request_sha256": request_fingerprint,
                            "selected_channel": parsed.channel,
                            "selected_key_label": parsed.key_label,
                            "response_sha256": _sha256_text(parsed.content),
                            "attempts": attempts,
                            "secrets_persisted": False,
                        }
                    )
                    self.clear_quota_exhausted(route.key_label)
                    return parsed

                if quota:
                    self.mark_quota_exhausted(route.key_label)
                    break

                retryable = (
                    result.status is None
                    or result.status in TRANSIENT_HTTP_STATUSES
                )
                if retryable and retry_index + 1 < len(retries):
                    continue

                status = "paused_transient_failure" if retryable else "paused_nonquota_error"
                reason = (
                    f"{route.key_label} halted after retries"
                    if retryable
                    else f"{route.key_label} returned a non-quota error"
                )
                self._write_checkpoint(
                    {
                        "schema_version": "neuroclaw.deepseek-v4-pro-router-checkpoint.v1",
                        "status": status,
                        "started_at": started_at,
                        "paused_at": utc_now(),
                        "request_id": request_id,
                        "request_sha256": request_fingerprint,
                        "attempts": attempts,
                        "reason": reason,
                        "secrets_persisted": False,
                    }
                )
                raise RouterPaused(reason, self.checkpoint_path)

        reason = "all permitted routes confirmed quota exhaustion"
        self._write_checkpoint(
            {
                "schema_version": "neuroclaw.deepseek-v4-pro-router-checkpoint.v1",
                "status": "paused_quota_exhausted",
                "started_at": started_at,
                "paused_at": utc_now(),
                "request_id": request_id,
                "request_sha256": request_fingerprint,
                "attempts": attempts,
                "reason": reason,
                "secrets_persisted": False,
            }
        )
        raise RouterPaused(reason, self.checkpoint_path)


def result_metadata(result: ChatResult) -> dict[str, Any]:
    """Return safe result metadata suitable for a manifest."""

    payload = asdict(result)
    payload.pop("message", None)
    payload.pop("thinking", None)
    payload["content_sha256"] = _sha256_text(result.content)
    payload["thinking_returned"] = bool(result.thinking)
    return payload
