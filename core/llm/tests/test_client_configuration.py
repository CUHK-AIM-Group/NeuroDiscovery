"""Client integration and config validation with no network/model requests."""
import asyncio
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from core.agent import main
from core.web import server
from core.llm.model_capabilities import PRESETS


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(server, "load_environment", lambda: {"llm_backend": {}}, raising=False)
    monkeypatch.setattr(server, "save_environment", lambda value: None, raising=False)
    monkeypatch.setattr(main, "load_environment", lambda: server.load_environment())
    monkeypatch.setattr(main, "save_environment", lambda value: server.save_environment(value))
    def offline(*args, **kwargs):
        raise server.urllib.error.URLError("Synthetic offline endpoint")
    monkeypatch.setattr(server.urllib.request, "urlopen", offline)
    with TestClient(server.create_app()) as client:
        yield client


def test_provider_catalog_and_preflight_do_not_probe_accounts(api, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pure configuration preflight must not inspect accounts or call providers")
    monkeypatch.setattr(server, "load_environment", forbidden)
    monkeypatch.setattr(server.urllib.request, "urlopen", forbidden)
    catalog = api.get("/api/llm/providers").json()
    assert len(catalog["providers"]) == len(PRESETS) and catalog["live_verified"] is False
    for preset in catalog["providers"]:
        result = api.post("/api/llm/validate", json={"provider": preset["provider"], "model": preset["default_model"],
                         "api_key": "synthetic-private-value", "headers": {"Authorization": "synthetic-private-value"}})
        assert result.status_code == 200
        assert "synthetic-private-value" not in result.text
        assert result.json()["live_verified"] is False


@pytest.mark.parametrize("controls", [{"reasoning_effort": "none"}, {"temperature": 0}, {"max_output_tokens": -1}, {"api_mode": "invalid"}])
def test_invalid_controls_are_actionable_errors(api, controls):
    result = api.post("/api/llm/validate", json={"provider": "openai", "model": "gpt-6-astra", **controls})
    assert result.status_code == 400 and result.json()["message"]


def test_anthropic_model_discovery_uses_native_path_and_auth(api, monkeypatch):
    captured = []
    cfg = {"provider": "anthropic", "model": "claude-opus-5", "base_url": "https://api.anthropic.com",
           "api_key": "offline-native-fixture", "api_key_env": "UNSET_TEST_KEY"}
    monkeypatch.setattr(server, "load_environment", lambda: {"llm_backend": cfg})
    class Reply(io.BytesIO):
        status = 200
    def mock(request, **kwargs):
        captured.append(request)
        return Reply(b'{"data":[{"id":"claude-opus-5"}]}')
    monkeypatch.setattr(server.urllib.request, "urlopen", mock)
    result = api.get("/api/env/models")
    assert result.status_code == 200
    assert captured[0].full_url == "https://api.anthropic.com/v1/models"
    headers = {key.lower(): value for key, value in captured[0].header_items()}
    assert headers["x-api-key"] == "offline-native-fixture" and "authorization" not in headers
    assert "offline-native-fixture" not in result.text
    assert result.json()["available_models"][0]["model"] == "claude-opus-5"


def test_cross_provider_selection_drops_previous_credentials_and_controls(api, monkeypatch):
    cfg = {"provider": "openai", "model": "gpt-6-astra", "api_key": "offline-old-fixture", "headers": {"Authorization": "offline-old-fixture"},
           "reasoning_effort": "max", "extra_body": {"old": True}, "api_mode": "responses",
           "available_models": [{"provider": "gemini", "model": "gemini-3.8-flash"}]}
    env, saved = {"llm_backend": cfg}, []
    monkeypatch.setattr(server, "load_environment", lambda: deepcopy(env))
    monkeypatch.setattr(server, "save_environment", lambda value: saved.append(deepcopy(value)))
    result = api.post("/api/env/model", json={"provider": "gemini", "model": "gemini-3.8-flash"})
    assert result.status_code == 200
    llm = saved[0]["llm_backend"]
    assert llm["api_key_env"] == "GEMINI_API_KEY"
    assert llm["base_url"].startswith("https://generativelanguage.googleapis.com/")
    for key in ("api_key", "headers", "reasoning_effort", "extra_body", "api_mode"):
        assert key not in llm


def test_desktop_numeric_controls_preserve_zero_and_never_enlarge_limit():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for desktop configuration tests")
    root = Path(__file__).resolve().parents[3]
    script = r'''
const assert = require('node:assert/strict');
const {applyModelControls} = require('./desktop/llm-settings');
assert.deepEqual(applyModelControls({llmMaxOutputTokens: '65536', llmTemperature: '0', llmReasoningEffort: 'high'}, {}),
  {max_output_tokens: 65536, temperature: 0, reasoning_effort: 'high'});
assert.deepEqual(applyModelControls({llmMaxOutputTokens: ''}, {max_output_tokens: 500}), {});
assert.deepEqual(applyModelControls({}, {max_output_tokens: 500}), {max_output_tokens: 500});
for (const value of ['0', '-1', '1.5', 'Infinity', '9007199254740992'])
  assert.throws(() => applyModelControls({llmMaxOutputTokens: value}, {}));
assert.throws(() => applyModelControls({llmApiMode: 'invalid'}, {}));
'''
    subprocess.run([node, "-e", script], cwd=root, check=True, capture_output=True, text=True)


def test_frontend_model_controls_and_scripts_parse():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for JavaScript syntax validation")
    root = Path(__file__).resolve().parents[3]
    script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('core/web/static/index.html', 'utf8');
for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
for (const file of ['desktop/main.js', 'desktop/llm-settings.js']) new vm.Script(fs.readFileSync(file, 'utf8'));
for (const key of ['llmApiMode', 'llmThinkingMode', 'llmReasoningEffort', 'llmMaxOutputTokens', 'llmTemperature']) assert.ok(html.includes("key: '" + key + "'"));
assert.ok(JSON.parse(fs.readFileSync('desktop/package.json', 'utf8')).build.files.includes('llm-settings.js'));
'''
    subprocess.run([node, "-e", script], cwd=root, check=True, capture_output=True, text=True)


def test_stream_disconnect_stops_producer_and_closes_iterator():
    emitted, closed = threading.Event(), threading.Event()
    class Stream:
        def __iter__(self):
            yield NS(choices=[NS(delta=NS(content="fixture"))])
            assert emitted.wait(2)
            yield NS(choices=[NS(delta=NS(content="must not be sent"))])
        def close(self):
            closed.set()
    class Socket:
        async def send_text(self, data):
            emitted.set()
            raise RuntimeError("synthetic disconnect")
    client = NS(chat=NS(completions=NS(create=lambda **kwargs: Stream())))
    with pytest.raises(RuntimeError, match="disconnect"):
        asyncio.run(server._stream_openai(Socket(), client, "fixture", []))
    assert closed.wait(2)


def test_factory_honors_native_direct_key_and_custom_endpoint(monkeypatch):
    import anthropic
    captured = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: captured.append(kwargs) or NS())
    monkeypatch.setattr(main, "is_feature_enabled", lambda *args: True)
    cfg = {"provider": "anthropic", "model": "claude-opus-5", "api_key": "offline-fixture", "base_url": "https://fixture.example"}
    main.build_llm_client({"llm_backend": cfg})
    assert captured == [{"api_key": "offline-fixture", "base_url": "https://fixture.example"}]


@pytest.mark.parametrize("provider", list(PRESETS))
def test_websocket_stream_routes_and_usage_for_eight_providers(provider):
    from core.llm.tests.test_adapters import Stream, fake, response
    from core.llm.adapters import plain
    if provider == "openai":
        events = [{"type": "response.output_text.delta", "delta": "done"}, {"type": "response.completed", "response": plain(response(provider))}]
    elif provider == "anthropic":
        events = [{"type": "message_start", "message": {"usage": {"input_tokens": 11}}},
                  {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                  {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                  {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}, {"type": "message_stop"}]
    else:
        events = [{"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}]
    stream = Stream(events)
    client, requests = fake(provider, [stream])
    session = NS(env={"llm_backend": client.cfg}, autoresearch_mode="off", _llm=client,
                 history=[{"role": "user", "content": "Fixture"}], _cancel_event=threading.Event())
    session.request_cancel = session._cancel_event.set
    sent = []
    class Socket:
        async def send_text(self, data):
            sent.append(json.loads(data))
        async def receive_text(self):
            await asyncio.Event().wait()
    assert asyncio.run(server._respond(Socket(), session)) == "done"
    assert sent[-1]["type"] == "done" and sent[-1]["token_usage"]["total_tokens"] == 18
    assert len(requests) == 1 and stream.closed
    if provider not in {"openai", "anthropic"}:
        assert requests[0]["stream_options"] == {"include_usage": True}
