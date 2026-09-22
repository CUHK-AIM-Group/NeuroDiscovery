from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.scripts.case1_codex_session_chat_gateway import (
    ENVELOPE_SCHEMA,
    RequestContractError,
    assert_no_codex_tool_use,
    build_chat_response,
    build_chat_stream,
    controlled_prompt,
    validate_chat_request,
    validate_envelope,
)


def test_envelope_const_property_declares_json_type() -> None:
    tool_type_schema = ENVELOPE_SCHEMA["properties"]["tool_calls"]["items"][
        "properties"
    ]["type"]

    assert tool_type_schema == {"type": "string", "const": "function"}


def tool_request() -> dict[str, object]:
    return {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Find a path"}],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "generate_path",
                    "parameters": {
                        "type": "object",
                        "properties": {"keyword": {"type": "string"}},
                        "required": ["keyword"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
    }


def test_validate_envelope_accepts_schema_valid_tool_call() -> None:
    payload = validate_chat_request(tool_request())
    envelope = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "generate_path",
                    "arguments": json.dumps({"keyword": "insula"}),
                },
            }
        ],
    }

    assert validate_envelope(envelope, payload) == envelope


def test_validate_envelope_rejects_invalid_tool_arguments() -> None:
    payload = validate_chat_request(tool_request())
    envelope = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "generate_path",
                    "arguments": json.dumps({}),
                },
            }
        ],
    }

    with pytest.raises(Exception):
        validate_envelope(envelope, payload)


def test_validate_chat_request_accepts_streaming() -> None:
    payload = tool_request()
    payload["stream"] = True

    assert validate_chat_request(payload)["stream"] is True


def test_build_chat_stream_emits_tool_delta_and_done() -> None:
    payload = validate_chat_request({**tool_request(), "stream": True})
    envelope = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "generate_path",
                    "arguments": json.dumps({"keyword": "insula"}),
                },
            }
        ],
    }
    response = build_chat_response(
        payload=payload,
        envelope=envelope,
        request_sha256="c" * 64,
        created=123,
    )

    stream = build_chat_stream(response, payload).decode("utf-8")

    assert '"object":"chat.completion.chunk"' in stream
    assert '"finish_reason":"tool_calls"' in stream
    assert '"index":0,"id":"call_1"' in stream
    assert stream.endswith("data: [DONE]\n\n")


def test_structured_content_and_response_shape() -> None:
    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Return JSON"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    envelope = {"content": '{"answer":"ok"}', "tool_calls": []}

    validate_envelope(envelope, validate_chat_request(payload))
    response = build_chat_response(
        payload=payload,
        envelope=envelope,
        request_sha256="a" * 64,
        created=123,
    )

    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == envelope["content"]


def test_controlled_prompt_binds_hash_and_forbids_external_access() -> None:
    payload = validate_chat_request(tool_request())
    prompt = controlled_prompt(payload, "b" * 64)

    assert "b" * 64 in prompt
    assert "不得使用 Codex 的 shell" in prompt
    assert "BEGIN_OPENAI_CHAT_REQUEST_JSON" in prompt


def test_codex_side_tool_use_is_rejected(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"
    rows = [
        {
            "type": "event_msg",
            "payload": {
                "turn_id": "turn-1",
                "type": "item_completed",
                "item": {"type": "CommandExecution"},
            },
        }
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(Exception, match="forbidden"):
        assert_no_codex_tool_use(log, "turn-1")
