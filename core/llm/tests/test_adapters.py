"""Deterministic wire-contract tests. No keys, accounts, or upstream calls."""
from copy import deepcopy
from importlib import import_module
import json
from types import SimpleNamespace as NS

import pytest

from core.llm.adapters import (ProviderClient, IncompleteModelResponse, assistant_message,
                               plain, objects)
from core.llm.model_capabilities import PRESETS, api_mode, request_options, validate_config
from core.llm.provider_profiles import apply_openai_compatible_profile_defaults, canonical_provider


TOOL = {"type": "function", "function": {"name": "inspect_local_path", "description": "Inspect a path",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
MESSAGES = [{"role": "system", "content": "Synthetic system"}, {"role": "user", "content": "Inspect the fixture"}]


def sdk_http_module(default_client):
    """Match mock request/response types to the SDK's actual HTTP dependency."""
    # Both libraries can be installed together. Import availability alone does
    # not identify the transport used by this SDK version.
    for base in default_client.__mro__:
        module = base.__module__.partition(".")[0]
        if module in {"httpx", "httpx2"}:
            return import_module(module)
    raise AssertionError("SDK default client has an unsupported HTTP transport")


@pytest.mark.parametrize("module_name", ["httpx", "httpx2"])
def test_sdk_http_module_matches_client_inheritance(module_name):
    http = pytest.importorskip(module_name)

    class WrappedClient(http.Client):
        pass

    assert sdk_http_module(WrappedClient) is http


def test_sdk_http_module_rejects_unknown_transport():
    with pytest.raises(AssertionError, match="unsupported HTTP transport"):
        sdk_http_module(object)


def config(provider):
    _, url, _, models = PRESETS[provider]
    return {"provider": provider, "model": models[0], "base_url": url, "max_output_tokens": 1024}


def response(provider, text="done", calls=(), *, finish=None):
    if provider == "openai":
        items = [{"type": "reasoning", "id": "rs_test", "summary": [], "encrypted_content": "opaque-encrypted-fixture"}]
        if text:
            items.append({"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                          "content": [{"type": "output_text", "text": text, "annotations": []}]})
        items.extend({"type": "function_call", "id": f"fc_{i}", "call_id": c["id"], "status": "completed", **c["function"]} for i, c in enumerate(calls))
        return objects({"status": finish or "completed", "output": items, "usage": {"input_tokens": 11, "output_tokens": 7}})
    if provider == "anthropic":
        blocks = [{"type": "thinking", "thinking": "synthetic reasoning", "signature": "opaque-signature-fixture"},
                  {"type": "redacted_thinking", "data": "opaque-redacted-fixture"}]
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.extend({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": json.loads(c["function"]["arguments"])} for c in calls)
        return objects({"content": blocks, "stop_reason": finish or ("tool_use" if calls else "end_turn"),
                        "usage": {"input_tokens": 5, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2, "output_tokens": 7}})
    return objects({"choices": [{"message": {"role": "assistant", "content": text, "reasoning_content": "synthetic reasoning",
                      "tool_calls": list(calls)}, "finish_reason": finish or ("tool_calls" if calls else "stop")}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})


def call(name="inspect_local_path", args=None, id="call_fixture"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {"path": "."})},
            "extra_content": {"google": {"thought_signature": "opaque-google-fixture"}}}


def fake(provider, replies):
    requests = []
    replies = iter(replies)
    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return item() if callable(item) else item
    raw = NS(chat=NS(completions=NS(create=create)), responses=NS(create=create), messages=NS(create=create))
    return ProviderClient(raw, config(provider)), requests


@pytest.mark.parametrize("provider", list(PRESETS))
def test_eight_providers_roundtrip_two_tools_and_usage(provider):
    client, requests = fake(provider, [response(provider, "", [call(), call(id="call_second")]), response(provider)])
    history = deepcopy(MESSAGES)
    result = client.create(model=config(provider)["model"], messages=history, tools=[TOOL], tool_choice="auto")
    history.append(assistant_message(result.choices[0].message))
    history.extend({"role": "tool", "tool_call_id": id, "content": '{"success": true}'} for id in ("call_fixture", "call_second"))
    done = client.create(model=config(provider)["model"], messages=history, tools=[TOOL], tool_choice="auto")
    assert done.choices[0].message.content == "done"
    assert plain(done.usage)["total_tokens"] == 18
    assert len(requests) == 2
    first, second = requests
    if provider == "openai":
        assert first["max_output_tokens"] == 1024 and first["store"] is False
        assert first["tools"][0]["strict"] is False
        assert "messages" not in first
        assert second["input"][2]["encrypted_content"] == "opaque-encrypted-fixture"
        assert [m["call_id"] for m in second["input"] if m.get("type") == "function_call_output"] == ["call_fixture", "call_second"]
    elif provider == "anthropic":
        assert first["max_tokens"] == 1024 and "input_schema" in first["tools"][0]
        assert first["system"] == [{"type": "text", "text": "Synthetic system"}]
        blocks = second["messages"][1]["content"]
        assert blocks[0]["signature"] == "opaque-signature-fixture" and blocks[1]["type"] == "redacted_thinking"
        assert len(second["messages"][-1]["content"]) == 2
    else:
        assert first["max_tokens"] == 1024
        saved = second["messages"][2]
        assert saved["reasoning_content"] == "synthetic reasoning"
        assert saved["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "opaque-google-fixture"
    assert MESSAGES[0]["content"] == "Synthetic system"  # no input mutation


@pytest.mark.parametrize("provider, control, expected", [
    ("openai", {"reasoning_effort": "max"}, {"reasoning": {"effort": "max"}}),
    ("anthropic", {"reasoning_effort": "high", "thinking_mode": "adaptive"}, {"output_config": {"effort": "high"}, "thinking": {"type": "adaptive"}}),
    ("gemini", {"reasoning_effort": "medium"}, {"reasoning_effort": "medium"}),
    ("grok", {"reasoning_effort": "xhigh"}, {"reasoning_effort": "xhigh"}),
    ("deepseek", {"thinking_mode": "enabled", "reasoning_effort": "high"}, {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "high"}),
    ("kimi", {"thinking_mode": "disabled"}, {"extra_body": {"thinking": {"type": "disabled"}}}),
    ("glm", {"thinking": {"type": "enabled", "clear_thinking": False}}, {"extra_body": {"thinking": {"type": "enabled", "clear_thinking": False}}}),
    ("qwen", {"thinking_mode": "enabled"}, {"extra_body": {"enable_thinking": True}}),
])
def test_controls_map_to_provider_wire(provider, control, expected):
    client, requests = fake(provider, [response(provider)])
    client.cfg.update(control)
    client.create(model=client.cfg["model"], messages=MESSAGES)
    for key, value in expected.items():
        assert requests[0][key] == value
    assert "temperature" not in requests[0]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "kimi"])
def test_fixed_sampling_omits_generic_probe_but_rejects_explicit_setting(provider):
    cfg = config(provider)
    assert "temperature" not in request_options(cfg, cfg["model"], {"temperature": 0})
    cfg["temperature"] = 0
    with pytest.raises(ValueError, match="provider-default"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [0, -1, True, "1.2", "nan", "inf", "bad"])
def test_invalid_caps_fail_before_dispatch(value):
    client, requests = fake("openai", [])
    client.cfg["max_output_tokens"] = value
    with pytest.raises((ValueError, OverflowError)):
        client.create(model="gpt-6-astra", messages=MESSAGES)
    assert requests == []


@pytest.mark.parametrize("requested, expected", [(512, 512), (2048, 1024)])
def test_output_budget_is_never_increased(requested, expected):
    client, requests = fake("openai", [response("openai")])
    client.create(model="gpt-6-astra", messages=MESSAGES, max_tokens=requested)
    assert requests[0]["max_output_tokens"] == expected


def test_conflicting_request_aliases_fail_without_dispatch():
    client, requests = fake("openai", [])
    with pytest.raises(ValueError, match="Conflicting"):
        client.create(model="gpt-6-astra", messages=MESSAGES, max_tokens=512, max_completion_tokens=1024)
    assert not requests


@pytest.mark.parametrize("effort", ["none", "minimal", "ultra"])
def test_gpt6_effort_validation(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        validate_config({**config("openai"), "reasoning_effort": effort})


def test_custom_routes_are_not_guessed_or_switched():
    cfg = {**config("openai"), "base_url": "https://proxy.example/v1"}
    assert api_mode(cfg) == "chat_completions"
    with pytest.raises(ValueError, match="requires api_mode=responses"):
        request_options(cfg, cfg["model"], {"tools": [TOOL]})
    assert api_mode({**cfg, "api_mode": "responses"}) == "responses"
    assert api_mode({**cfg, "base_url": "https://api.openai.com.proxy.example/v1"}) == "chat_completions"
    assert validate_config({**cfg, "model": "future-custom-model"})["custom_model_allowed"]


def test_ollama_experiment_options_are_not_reinterpreted():
    cfg = {"provider": "ollama", "model": "deepseek-v4.1-flash:cloud", "temperature": 0,
           "extra_body": {"options": {"num_predict": 65536}, "think": "high"}}
    previous = deepcopy(cfg)
    result = request_options(cfg, cfg["model"], {})
    assert result == {"temperature": 0, "extra_body": cfg["extra_body"]}
    assert cfg == previous


def test_imported_text_history_does_not_invent_deepseek_reasoning():
    client, requests = fake("deepseek", [response("deepseek", "", [call()]), response("deepseek")])
    original = MESSAGES + [{"role": "assistant", "content": "Earlier visible answer"}, {"role": "user", "content": "Continue"}]
    first = client.create(model=client.cfg["model"], messages=original, tools=[TOOL])
    second_history = original + [assistant_message(first.choices[0].message), {"role": "tool", "tool_call_id": "call_fixture", "content": "Actual fixture result"}]
    client.create(model=client.cfg["model"], messages=second_history, tools=[TOOL])
    assert "Imported conversation reference" in requests[0]["messages"][1]["content"]
    assert "Earlier visible answer" in requests[0]["messages"][1]["content"]
    assert "reasoning_content" not in original[2]
    assert requests[1]["messages"][-2]["reasoning_content"] == "synthetic reasoning"
    assert requests[1]["messages"][-1]["role"] == "tool"


def test_interrupted_autoresearch_counts_actual_usage(tmp_path):
    from core.agent.test_autoresearch_execution import session
    agent, _ = session(tmp_path, [])
    client, requests = fake("openai", [response("openai", "unfinished", finish="incomplete")])
    agent.env["llm_backend"] = config("openai")
    agent._llm = client
    assert "interrupted" in agent._chat()
    assert agent._last_token_usage["total_tokens"] == 18
    assert len(requests) == 1


@pytest.mark.parametrize("provider, reason", [("openai", "incomplete"), ("anthropic", "max_tokens"), ("qwen", "length")])
def test_truncated_tool_call_is_never_executed_or_replayed(provider, reason):
    from core.agent.main import _retry_api_call
    client, requests = fake(provider, [response(provider, "partial", [call()], finish=reason)])
    with pytest.raises(IncompleteModelResponse):
        _retry_api_call("offline", lambda: client.create(model=client.cfg["model"], messages=MESSAGES), retries=10)
    assert len(requests) == 1


def test_native_protocol_metadata_cannot_cross_models():
    client, requests = fake("openai", [response("openai", "", [call()])])
    result = client.create(model="gpt-6-astra", messages=MESSAGES)
    history = MESSAGES + [assistant_message(result.choices[0].message)]
    with pytest.raises(ValueError, match="changed"):
        client.create(model="gpt-6-astra-other", messages=history)
    assert len(requests) == 1


@pytest.mark.parametrize("alias, provider", [("claude", "anthropic"), ("xai", "grok"), ("google", "gemini"), ("zhipuai", "zhipu")])
def test_aliases(alias, provider):
    assert canonical_provider(alias) == provider


@pytest.mark.parametrize("provider", ["deepseek", "kimi", "qwen", "glm", "gemini", "grok"])
def test_profiles_preserve_existing_endpoint_model_and_identity(provider):
    cfg = {"provider": provider, "model": "pinned-original", "base_url": "https://private.example/v1", "api_key_env": "TEST_ENV_NAME"}
    apply_openai_compatible_profile_defaults(cfg)
    assert cfg["model"] == "pinned-original" and cfg["base_url"] == "https://private.example/v1"
    assert cfg["api_key_env"] == "TEST_ENV_NAME"


class Stream:
    def __init__(self, events):
        self.events, self.closed = events, False
    def __iter__(self):
        yield from map(objects, self.events)
    def close(self):
        self.closed = True


def test_responses_stream_and_usage():
    stream = Stream([{"type": "response.output_text.delta", "delta": "done"},
                     {"type": "response.completed", "response": plain(response("openai"))}])
    client, _ = fake("openai", [stream])
    chunks = list(client.create(model="gpt-6-astra", messages=MESSAGES, stream=True))
    assert chunks[0].choices[0].delta.content == "done"
    assert stream.closed and client.last_response.usage.total_tokens == 18


def test_anthropic_stream_keeps_complete_thinking_signature():
    stream = Stream([
        {"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "private fixture"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "opaque"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "done"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ])
    client, _ = fake("anthropic", [stream])
    chunks = list(client.create(model=client.cfg["model"], messages=MESSAGES, stream=True))
    deltas = [c.choices[0].delta for c in chunks if c.choices]
    assert [d.content for d in deltas if hasattr(d, "content")] == ["done"]
    assert [d.reasoning_content for d in deltas if hasattr(d, "reasoning_content")] == ["private fixture"]
    assert all(not hasattr(d, "signature") for d in deltas)
    saved = assistant_message(client.last_response.choices[0].message)
    assert saved["_nd_protocol"]["output"][0]["signature"] == "opaque"
    assert stream.closed and client.last_response.usage.total_tokens == 18


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_interrupted_stream_is_closed_and_not_reported_complete(provider):
    stream = Stream([])
    client, requests = fake(provider, [stream])
    with pytest.raises(IncompleteModelResponse):
        list(client.create(model=client.cfg["model"], messages=MESSAGES, stream=True))
    assert stream.closed and client.last_response is None and len(requests) == 1


def test_real_openai_sdk_serializes_responses_contract_without_network():
    from openai import OpenAI, DefaultHttpxClient
    httpx = sdk_http_module(DefaultHttpxClient)
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        data = plain(response("openai"))
        return httpx.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 1,
                                       "model": "gpt-6-astra", **data})
    with OpenAI(api_key="offline-fixture", http_client=DefaultHttpxClient(transport=httpx.MockTransport(handler), trust_env=False)) as raw:
        client = ProviderClient(raw, {**config("openai"), "reasoning_effort": "high"})
        result = client.create(model="gpt-6-astra", messages=MESSAGES, tools=[TOOL])
    assert result.choices[0].message.content == "done"
    assert captured[0]["reasoning"] == {"effort": "high"}
    assert captured[0]["max_output_tokens"] == 1024


@pytest.mark.parametrize("provider", list(PRESETS))
def test_autoresearch_runs_past_eight_rounds_until_valid_delivery(tmp_path, monkeypatch, provider):
    from core.agent.test_autoresearch_execution import session, completion
    monkeypatch.delenv("NEUROCLAW_MAX_TOOL_ITERATIONS", raising=False)
    (tmp_path / "result.md").write_text("Synthetic fixture, not scientific evidence.", encoding="utf-8")
    replies = [response(provider, "I will start.")]
    replies += [response(provider, "", [call(args={"path": "result.md"}, id=f"inspect_{i}")]) for i in range(10)]
    replies += [response(provider, "", [call("finish_autoresearch", completion())])]
    client, requests = fake(provider, replies)
    agent, _ = session(tmp_path, [])
    agent.env["llm_backend"] = config(provider)
    agent._llm = client
    result = agent._chat()
    assert "completed" in result, result
    assert len(requests) == 12 and agent.autoresearch_state["status"] == "completed"
    assert agent._last_token_usage["total_tokens"] == 12 * 18
    assert len(agent.history) == 2  # do not persist synthetic continuation prompts as user input


@pytest.mark.parametrize("provider", list(PRESETS))
def test_cancellation_after_response_does_not_execute_tools_or_next_request(tmp_path, provider):
    from core.agent.test_autoresearch_execution import session
    agent, _ = session(tmp_path, [])
    def cancel_inflight():
        agent.request_cancel()
        return response(provider, "", [call()])
    client, requests = fake(provider, [cancel_inflight])
    agent.env["llm_backend"] = config(provider)
    agent._llm = client
    assert "cancelled" in agent._chat()
    assert len(requests) == 1 and agent.autoresearch_state["evidence"] == []


def test_real_anthropic_sdk_wire_and_thinking_roundtrip():
    from anthropic import Anthropic, DefaultHttpxClient
    httpx = sdk_http_module(DefaultHttpxClient)
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        data = plain(response("anthropic", "", [call()]) if len(captured) == 1 else response("anthropic"))
        return httpx.Response(200, json={"id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-opus-5", **data})
    with Anthropic(api_key="offline-fixture", http_client=DefaultHttpxClient(transport=httpx.MockTransport(handler), trust_env=False)) as raw:
        client = ProviderClient(raw, {**config("anthropic"), "reasoning_effort": "high", "thinking_mode": "adaptive"})
        first = client.create(model="claude-opus-5", messages=MESSAGES, tools=[TOOL], tool_choice="auto")
        history = MESSAGES + [assistant_message(first.choices[0].message), {"role": "tool", "tool_call_id": "call_fixture", "content": "Synthetic result"}]
        result = client.create(model="claude-opus-5", messages=history, tools=[TOOL], tool_choice="auto")
    assert result.choices[0].message.content == "done"
    assert captured[0]["output_config"] == {"effort": "high"}
    assert captured[1]["messages"][1]["content"][0]["signature"] == "opaque-signature-fixture"


@pytest.mark.parametrize("provider", ["gemini", "grok", "deepseek", "kimi", "glm", "qwen"])
def test_real_sdk_chat_compatible_wire(provider):
    from openai import OpenAI, DefaultHttpxClient
    httpx = sdk_http_module(DefaultHttpxClient)
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        data = plain(response(provider, "", [call()]) if len(captured) == 1 else response(provider))
        return httpx.Response(200, json={"id": "chatcmpl_fixture", "object": "chat.completion", "created": 1, "model": config(provider)["model"], **data})
    with OpenAI(api_key="offline-fixture", base_url=config(provider)["base_url"], http_client=DefaultHttpxClient(transport=httpx.MockTransport(handler), trust_env=False)) as raw:
        client = ProviderClient(raw, config(provider))
        first = client.create(model=client.cfg["model"], messages=MESSAGES, tools=[TOOL])
        history = MESSAGES + [assistant_message(first.choices[0].message), {"role": "tool", "tool_call_id": "call_fixture", "content": "Synthetic result"}]
        client.create(model=client.cfg["model"], messages=history, tools=[TOOL])
    assert captured[1]["messages"][2]["reasoning_content"] == "synthetic reasoning"
    assert captured[1]["messages"][2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "opaque-google-fixture"


def test_auxiliary_model_controls_do_not_inherit_flagship_reasoning_or_budget():
    client, requests = fake("openai", [response("gemini", '{}')])
    client.cfg.update(reasoning_effort="max", max_output_tokens=65536)
    client.create(model="gpt-4o-mini", messages=MESSAGES, temperature=0, max_tokens=600, response_format={"type": "json_object"})
    assert requests[0]["model"] == "gpt-4o-mini" and requests[0]["max_tokens"] == 600
    assert requests[0]["temperature"] == 0 and "reasoning_effort" not in requests[0]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("kind", ["json_object", "json_schema"])
def test_json_output_contract_translation(provider, kind):
    client, requests = fake(provider, [response(provider, '{"ok": true}')])
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    fmt = {"type": kind}
    if kind == "json_schema":
        fmt["json_schema"] = {"name": "result", "schema": schema, "strict": True}
    client.cfg["reasoning_effort"] = "high"
    result = client.create(model=client.cfg["model"], messages=MESSAGES, response_format=fmt)
    assert json.loads(result.choices[0].message.content) == {"ok": True}
    assert "response_format" not in requests[0]
    if provider == "openai":
        assert requests[0]["text"]["format"]["type"] == kind
    elif kind == "json_schema":
        assert requests[0]["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": schema}}
    else:
        assert "valid JSON object" in requests[0]["system"][-1]["text"]


def test_native_json_object_rejects_malformed_answer_without_replay():
    client, requests = fake("anthropic", [response("anthropic", 'not JSON')])
    with pytest.raises(IncompleteModelResponse, match="JSON object"):
        client.create(model=client.cfg["model"], messages=MESSAGES, response_format={"type": "json_object"})
    assert client.last_response is None and len(requests) == 1


@pytest.mark.parametrize("provider", list(PRESETS))
def test_background_models_are_not_silently_upgraded(provider):
    from core.llm.adapters import auxiliary_model
    client, requests = fake(provider, [])
    if provider == "openai":
        assert auxiliary_model(client) == "gpt-4o-mini"
    else:
        with pytest.raises(ValueError, match="auxiliary_model"):
            auxiliary_model(client)
    client.cfg["auxiliary_model"] = "explicit-small-model"
    assert auxiliary_model(client) == "explicit-small-model"
    assert requests == []


@pytest.mark.parametrize("provider", list(PRESETS))
def test_explicit_tool_probe_uses_the_same_provider_adapter(tmp_path, monkeypatch, provider):
    from core.agent import main
    client, requests = fake(provider, [response(provider, "", [call("read_workspace_file", {"path": "fixture.txt"})]), response(provider, "OK")])
    (tmp_path / "fixture.txt").write_text("Synthetic fixture.", encoding="utf-8")
    monkeypatch.setattr(main, "AgentSession", lambda **kwargs: NS())
    monkeypatch.setattr(main, "build_llm_client", lambda env: client)
    assert main._run_openai_tool_loop_probe({"llm_backend": config(provider)}, client.cfg["model"], tmp_path) == 0
    assert len(requests) == 2


def test_one_shot_probe_does_not_claim_tool_support_for_text_only(tmp_path, monkeypatch):
    from core.agent import main
    client, requests = fake("openai", [response("openai", "No tool call")])
    monkeypatch.setattr(main, "build_llm_client", lambda env: client)
    assert main._run_openai_tool_probe({"llm_backend": config("openai")}, "gpt-6-astra", tmp_path) == 2
