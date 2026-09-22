from __future__ import annotations

import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai import OpenAI
import pytest

from neurooracle.scripts.deepseek_v4_pro_gateway import (
    GatewayServer,
    normalize_provider_messages,
    responses_input_to_messages,
    responses_stream_events,
    responses_tools_to_chat,
    result_to_responses,
)
from neurooracle.scripts.deepseek_v4_pro_router import (
    ChatResult,
    DeepSeekV4ProRouter,
    RouterPaused,
    TransportResult,
    confirmed_quota_exhaustion,
)


def keys_file(
    tmp_path: Path,
    *,
    ollama_keys: tuple[str, ...] = ("ollama-secret",),
) -> Path:
    path = tmp_path / "keys.txt"
    ollama_section = "\n".join(ollama_keys)
    path.write_text(
        "OpenCode API Key\n"
        "go-secret-1\n"
        "go-secret-2\n"
        "go-secret-3\n\n"
        "Ollama API Key\n"
        f"{ollama_section}\n\n"
        "DeepSeek API Key\n"
        "deepseek-secret\n",
        encoding="utf-8",
    )
    return path


def test_multiple_ollama_keys_are_tried_in_file_order(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(endpoint, key, payload, timeout):
        calls.append(key)
        if key != "ollama-secret-2":
            return quota()
        return TransportResult(
            status=200,
            payload={
                "model": "deepseek-v4-pro:cloud",
                "message": {"role": "assistant", "content": "OLLAMA_2_OK"},
                "done": True,
                "done_reason": "stop",
            },
        )

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(
            tmp_path,
            ollama_keys=("ollama-secret-1", "ollama-secret-2"),
        ),
        checkpoint_path=tmp_path / "checkpoint.json",
        retry_delays=(),
        transport=transport,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert calls == [
        "go-secret-1",
        "go-secret-2",
        "go-secret-3",
        "ollama-secret-1",
        "ollama-secret-2",
    ]
    assert result.channel == "ollama_cloud"
    assert result.content == "OLLAMA_2_OK"
    assert [route.key_label for route in router.routes()] == [
        "opencode_go_key_1",
        "opencode_go_key_2",
        "opencode_go_key_3",
        "ollama_cloud_key_1",
        "ollama_cloud_key_2",
        "opencode_go_key_1",
        "opencode_go_key_2",
        "opencode_go_key_3",
    ]


def test_ollama_only_mode_never_calls_opencode(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(endpoint, key, payload, timeout):
        calls.append(key)
        if key == "ollama-secret-1":
            return quota()
        return TransportResult(
            status=200,
            payload={
                "model": "deepseek-v4-pro:cloud",
                "message": {"role": "assistant", "content": "OLLAMA_ONLY_OK"},
                "done": True,
                "done_reason": "stop",
            },
        )

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(
            tmp_path,
            ollama_keys=("ollama-secret-1", "ollama-secret-2"),
        ),
        checkpoint_path=tmp_path / "checkpoint.json",
        retry_delays=(),
        route_mode="ollama_only",
        transport=transport,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert calls == ["ollama-secret-1", "ollama-secret-2"]
    assert result.channel == "ollama_cloud"
    assert result.content == "OLLAMA_ONLY_OK"
    assert [route.key_label for route in router.routes()] == [
        "ollama_cloud_key_1",
        "ollama_cloud_key_2",
    ]


def quota() -> TransportResult:
    return TransportResult(
        status=429,
        payload={"error": {"code": "insufficient_quota", "message": "quota exhausted"}},
        error_kind="http_error",
    )


def success(content: str = "OK") -> TransportResult:
    return TransportResult(
        status=200,
        payload={
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


def test_route_switches_only_after_confirmed_quota(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(endpoint, key, payload, timeout):
        calls.append(key)
        if key.startswith("go-secret"):
            return quota()
        return TransportResult(
            status=200,
            payload={
                "model": "deepseek-v4-pro:cloud",
                "message": {"role": "assistant", "content": "OLLAMA_OK"},
                "done": True,
                "done_reason": "stop",
            },
        )

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.json",
        retry_delays=(),
        transport=transport,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert calls == ["go-secret-1", "go-secret-2", "go-secret-3", "ollama-secret"]
    assert result.channel == "ollama_cloud"
    assert result.content == "OLLAMA_OK"
    checkpoint = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")
    assert "go-secret" not in checkpoint
    assert "ollama-secret" not in checkpoint


def test_transient_errors_retry_same_route_with_2_8_30_policy(tmp_path: Path) -> None:
    responses = [
        TransportResult(status=503, payload={"error": {"message": "temporary"}}),
        TransportResult(status=None, payload=None, error_kind="network_error"),
        success(),
    ]
    keys: list[str] = []
    delays: list[float] = []

    def transport(endpoint, key, payload, timeout):
        keys.append(key)
        return responses.pop(0)

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(2, 8, 30),
        transport=transport,
        sleep=delays.append,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert result.content == "OK"
    assert keys == ["go-secret-1"] * 3
    assert delays == [2.0, 8.0]


def test_invalid_success_payload_retries_same_route(tmp_path: Path) -> None:
    responses = [
        TransportResult(
            status=200,
            payload={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "unfinished reasoning",
                        },
                        "finish_reason": "length",
                    }
                ]
            },
        ),
        success("RECOVERED"),
    ]
    keys: list[str] = []
    delays: list[float] = []
    budgets: list[int] = []

    def transport(endpoint, key, payload, timeout):
        keys.append(key)
        budgets.append(int(payload["max_tokens"]))
        return responses.pop(0)

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(2, 8, 30),
        transport=transport,
        sleep=delays.append,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert result.content == "RECOVERED"
    assert keys == ["go-secret-1", "go-secret-1"]
    assert delays == [2.0]
    assert budgets == [8192, 16384]


def test_json_schema_is_embedded_for_go_and_translated_for_ollama(
    tmp_path: Path,
) -> None:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    seen: list[tuple[str, dict]] = []

    def transport(endpoint, key, payload, timeout):
        seen.append((key, payload))
        if key.startswith("go-secret"):
            return quota()
        return TransportResult(
            status=200,
            payload={
                "model": "deepseek-v4-pro:cloud",
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                "done": True,
                "done_reason": "stop",
            },
        )

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(),
        transport=transport,
    )
    router.complete(
        messages=[{"role": "user", "content": "hello"}],
        response_format=response_format,
    )

    for key, payload in seen[:3]:
        assert key.startswith("go-secret")
        assert payload["response_format"] == {"type": "json_object"}
        assert "JSON Schema" in payload["messages"][0]["content"]
        assert '"required": ["answer"]' in payload["messages"][0]["content"]
    assert seen[3][0] == "ollama-secret"
    assert seen[3][1]["format"] == response_format["json_schema"]["schema"]


def test_ollama_tool_call_arguments_are_converted_in_both_directions(
    tmp_path: Path,
) -> None:
    seen_ollama_payload: dict = {}

    def transport(endpoint, key, payload, timeout):
        if key.startswith("go-secret"):
            return quota()
        seen_ollama_payload.update(payload)
        return TransportResult(
            status=200,
            payload={
                "model": "deepseek-v4-pro:cloud",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "brain age"},
                            }
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
            },
        )

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(),
        transport=transport,
    )
    result = router.complete(
        messages=[
            {"role": "user", "content": "use the tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"brain age"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
    )

    sent_call = seen_ollama_payload["messages"][1]["tool_calls"][0]
    assert sent_call["function"]["arguments"] == {"query": "brain age"}
    assert seen_ollama_payload["messages"][2]["tool_name"] == "lookup"
    returned_call = result.message["tool_calls"][0]
    assert returned_call["type"] == "function"
    assert returned_call["id"].startswith("call_")
    assert json.loads(returned_call["function"]["arguments"]) == {
        "query": "brain age"
    }


def test_ollama_length_retry_increases_num_predict_on_same_route(
    tmp_path: Path,
) -> None:
    budgets: list[int] = []
    delays: list[float] = []

    def transport(endpoint, key, payload, timeout):
        if key.startswith("go-secret"):
            return quota()
        budgets.append(int(payload["options"]["num_predict"]))
        if len(budgets) == 1:
            return TransportResult(
                status=200,
                payload={
                    "model": "deepseek-v4-pro:cloud",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "private reasoning",
                    },
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 100,
                    "eval_count": 8192,
                },
            )
        return TransportResult(
            status=200,
            payload={
                "model": "deepseek-v4-pro:cloud",
                "message": {"role": "assistant", "content": "RECOVERED"},
                "done": True,
                "done_reason": "stop",
            },
        )

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(2,),
        transport=transport,
        sleep=delays.append,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert result.content == "RECOVERED"
    assert budgets == [8192, 16384]
    assert delays == [2.0]


def test_json_object_adds_required_go_prompt_marker(tmp_path: Path) -> None:
    seen_payload: dict = {}

    def transport(endpoint, key, payload, timeout):
        seen_payload.update(payload)
        return success('{"answer":"ok"}')

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(),
        transport=transport,
    )
    router.complete(
        messages=[{"role": "user", "content": "Return the answer object."}],
        response_format={"type": "json_object"},
    )

    assert seen_payload["response_format"] == {"type": "json_object"}
    assert seen_payload["messages"][0] == {
        "role": "system",
        "content": "Return one valid JSON object only.",
    }


def test_generic_rate_limit_does_not_switch_channel(tmp_path: Path) -> None:
    calls = 0

    def transport(endpoint, key, payload, timeout):
        nonlocal calls
        calls += 1
        return TransportResult(
            status=429,
            payload={
                "error": {
                    "message": "rate limit: retry later; token=go-secret-1"
                }
            },
            error_kind="http_error",
        )

    checkpoint = tmp_path / "checkpoint.json"
    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        checkpoint_path=checkpoint,
        retry_delays=(),
        transport=transport,
    )
    with pytest.raises(RouterPaused, match="non-quota"):
        router.complete(messages=[{"role": "user", "content": "hello"}])
    assert calls == 1
    checkpoint_text = checkpoint.read_text(encoding="utf-8")
    assert "go-secret-1" not in checkpoint_text
    assert json.loads(checkpoint_text)["status"] == "paused_nonquota_error"


def test_invalid_success_pauses_once_with_content_free_response_shape(
    tmp_path: Path,
) -> None:
    calls = 0

    def transport(endpoint, key, payload, timeout):
        nonlocal calls
        calls += 1
        return TransportResult(
            status=200,
            payload={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "private reasoning",
                        },
                    }
                ]
            },
        )

    checkpoint = tmp_path / "checkpoint.json"
    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        checkpoint_path=checkpoint,
        retry_delays=(),
        transport=transport,
    )

    with pytest.raises(RouterPaused, match="invalid provider response"):
        router.complete(messages=[{"role": "user", "content": "hello"}])

    assert calls == 1
    checkpoint_text = checkpoint.read_text(encoding="utf-8")
    payload = json.loads(checkpoint_text)
    assert payload["status"] == "paused_invalid_response"
    attempt = payload["attempts"][0]
    assert attempt["error_detail"] == "provider returned neither content nor tool calls"
    assert attempt["response_shape"]["finish_reason"] == "length"
    assert attempt["response_shape"]["content_characters"] == 0
    assert attempt["response_shape"]["reasoning_content_characters"] == 17
    assert "private reasoning" not in checkpoint_text


def test_final_go_pass_occurs_once_then_pauses(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(endpoint, key, payload, timeout):
        calls.append(key)
        return quota()

    checkpoint = tmp_path / "checkpoint.json"
    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        checkpoint_path=checkpoint,
        retry_delays=(),
        transport=transport,
    )
    with pytest.raises(RouterPaused, match="all permitted routes"):
        router.complete(messages=[{"role": "user", "content": "hello"}])

    assert calls == [
        "go-secret-1",
        "go-secret-2",
        "go-secret-3",
        "ollama-secret",
        "go-secret-1",
        "go-secret-2",
        "go-secret-3",
    ]
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "paused_quota_exhausted"
    assert len(payload["attempts"]) == 7


def test_quota_classifier_rejects_plain_rate_limit() -> None:
    assert confirmed_quota_exhaustion(
        429,
        {"error": {"message": "insufficient quota"}},
        "",
    )
    assert not confirmed_quota_exhaustion(
        429,
        {"error": {"message": "requests per minute rate limit"}},
        "",
    )


@pytest.mark.parametrize(
    "message",
    [
        "Weekly usage limit reached. Resets in 4 days.",
        "5-hour usage limit reached. Resets in 2hr 47min.",
        "You have reached your session usage limit; upgrade for higher limits.",
        "You have reached your weekly usage limit; upgrade for higher limits.",
    ],
)
def test_quota_classifier_accepts_explicit_usage_limit(message: str) -> None:
    assert confirmed_quota_exhaustion(
        429,
        {"error": {"message": message}},
        message,
    )


def test_explicit_usage_limit_switches_to_next_go_key(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(endpoint, key, payload, timeout):
        calls.append(key)
        if key == "go-secret-1":
            return TransportResult(
                status=429,
                payload={
                    "error": {
                        "message": "Weekly usage limit reached. Resets in 4 days."
                    }
                },
                error_kind="http_error",
            )
        return success("KEY_2_OK")

    router = DeepSeekV4ProRouter(
        keys_path=keys_file(tmp_path),
        retry_delays=(),
        transport=transport,
    )
    result = router.complete(messages=[{"role": "user", "content": "hello"}])

    assert calls == ["go-secret-1", "go-secret-2"]
    assert result.key_label == "opencode_go_key_2"
    assert result.content == "KEY_2_OK"


def test_confirmed_quota_is_cached_across_gateway_style_requests(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    exhausted: set[str] = set()

    def transport(endpoint, key, payload, timeout):
        calls.append(key)
        if key == "go-secret-1":
            return TransportResult(
                status=429,
                payload={"error": {"message": "Weekly usage limit reached."}},
            )
        return success("OK")

    def router(checkpoint: str) -> DeepSeekV4ProRouter:
        return DeepSeekV4ProRouter(
            keys_path=keys_file(tmp_path),
            checkpoint_path=tmp_path / checkpoint,
            retry_delays=(),
            transport=transport,
            is_quota_exhausted=exhausted.__contains__,
            mark_quota_exhausted=exhausted.add,
            clear_quota_exhausted=exhausted.discard,
        )

    router("first.json").complete(messages=[{"role": "user", "content": "one"}])
    router("second.json").complete(messages=[{"role": "user", "content": "two"}])

    assert calls == ["go-secret-1", "go-secret-2", "go-secret-2"]
    second = json.loads((tmp_path / "second.json").read_text(encoding="utf-8"))
    assert second["attempts"][0]["error_kind"] == "cached_quota_exhaustion"
    assert second["selected_key_label"] == "opencode_go_key_2"


def test_responses_translation_supports_tools() -> None:
    messages = responses_input_to_messages(
        {
            "instructions": "system",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            ],
        }
    )
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]
    tools = responses_tools_to_chat(
        [
            {
                "type": "function",
                "name": "lookup",
                "description": "look up",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look up",
                "parameters": {"type": "object"},
            },
        }
    ]

    result = ChatResult(
        channel="opencode_go",
        key_label="opencode_go_key_1",
        model="deepseek-v4-pro",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        finish_reason="tool_calls",
        usage={},
        thinking="",
        request_id="request",
    )
    response = result_to_responses(result)
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["name"] == "lookup"


def test_provider_messages_translate_developer_role_without_mutating_input() -> None:
    source = [
        {"role": "developer", "content": "follow the protocol"},
        {"role": "user", "content": "hello"},
    ]

    assert normalize_provider_messages(source) == [
        {"role": "system", "content": "follow the protocol"},
        {"role": "user", "content": "hello"},
    ]
    assert source[0]["role"] == "developer"


def test_gateway_is_compatible_with_openai_responses_and_chat_clients(
    tmp_path: Path,
) -> None:
    result = ChatResult(
        channel="opencode_go",
        key_label="opencode_go_key_1",
        model="deepseek-v4-pro",
        message={"role": "assistant", "content": "CLIENT_OK"},
        finish_reason="stop",
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        thinking="",
        request_id="request",
    )

    class FakeRouter:
        def complete(self, **kwargs):
            return result

    server = GatewayServer(
        ("127.0.0.1", 0),
        keys_path=keys_file(tmp_path),
        checkpoint_dir=tmp_path,
        timeout_seconds=1,
        allow_official_deepseek=False,
    )
    server.router = lambda request_id: FakeRouter()  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/health", timeout=5) as health_response:
            health = json.loads(health_response.read().decode("utf-8"))
        assert health["upstream_timeout_seconds"] == 1
        client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="neuroclaw-local-router",
        )
        response = client.responses.create(model="deepseek-v4-pro", input="hello")
        assert response.output_text == "CLIENT_OK"
        stream_events = list(
            client.responses.create(
                model="deepseek-v4-pro",
                input="hello",
                stream=True,
            )
        )
        assert [event.type for event in stream_events] == [
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert stream_events[2].delta == "CLIENT_OK"
        chat = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert chat.choices[0].message.content == "CLIENT_OK"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_gateway_can_return_nonretryable_status_after_router_pause(
    tmp_path: Path,
) -> None:
    class PausedRouter:
        def complete(self, **kwargs):
            raise RouterPaused("finite route paused", tmp_path / "pause.json")

    server = GatewayServer(
        ("127.0.0.1", 0),
        keys_path=keys_file(tmp_path),
        checkpoint_dir=tmp_path,
        timeout_seconds=1,
        allow_official_deepseek=False,
        router_pause_http_status=422,
    )
    server.router = lambda request_id: PausedRouter()  # type: ignore[method-assign]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["router_pause_http_status"] == 422
        request = Request(
            f"http://{host}:{port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as captured:
            urlopen(request, timeout=5)
        assert captured.value.code == 422
        payload = json.loads(captured.value.read().decode("utf-8"))
        assert payload["error"]["type"] == "router_paused"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_responses_stream_replays_tool_calls_for_brainpilot() -> None:
    result = ChatResult(
        channel="opencode_go",
        key_label="opencode_go_key_1",
        model="deepseek-v4-pro",
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"x":1}'},
                }
            ],
        },
        finish_reason="tool_calls",
        usage={},
        thinking="",
        request_id="request",
    )
    events = responses_stream_events(result_to_responses(result))
    types = [event_type for event_type, _ in events]
    assert types == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = events[1][1]["item"]
    assert added["type"] == "function_call"
    assert added["arguments"] == ""
    assert events[2][1]["delta"] == '{"x":1}'
