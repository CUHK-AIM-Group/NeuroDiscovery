"""Ollama Cloud routing checks; synthetic credentials and offline SDK transport."""
import json

import httpx
import pytest

from core.agent import main
from core.llm.adapters import ProviderClient
from core.llm.model_capabilities import request_options
from core.llm.provider_profiles import apply_openai_compatible_profile_defaults


def test_cloud_defaults_require_cloud_key_and_local_stays_unchanged(monkeypatch):
    local, cloud = {'provider': 'ollama'}, {'provider': 'ollama_cloud', 'no_api_key_required': True}
    apply_openai_compatible_profile_defaults(local)
    apply_openai_compatible_profile_defaults(cloud)
    assert local['no_api_key_required'] and local['base_url'] == 'http://localhost:11434/v1'
    assert cloud['base_url'] == 'https://ollama.com/v1'
    assert cloud['api_key_env'] == 'OLLAMA_API_KEY' and not cloud.get('no_api_key_required')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-wrong-provider')
    monkeypatch.delenv('OLLAMA_API_KEY', raising=False)
    assert main._resolve_openai_api_key(cloud) == ('', 'OLLAMA_API_KEY')
    monkeypatch.setenv('OLLAMA_API_KEY', 'synthetic-cloud-fixture')
    assert main._resolve_openai_api_key(cloud) == ('synthetic-cloud-fixture', 'OLLAMA_API_KEY')


@pytest.mark.parametrize('endpoint', ['http://ollama.com/v1', 'https://ollama.com.other.example/v1',
                                    'https://other.example/v1', 'https://ollama.com/v1?other=1'])
def test_cloud_key_never_reaches_a_different_host(endpoint, monkeypatch):
    monkeypatch.setenv('OLLAMA_API_KEY', 'synthetic-cloud-fixture')
    with pytest.raises(ValueError, match='require https://ollama.com/v1'):
        main._resolve_openai_api_key({'provider': 'ollama_cloud', 'base_url': endpoint})


def test_cloud_sdk_request_preserves_reasoning_temperature_budget_and_auth():
    from openai import OpenAI
    cfg = {'provider': 'ollama_cloud', 'model': 'deepseek-v4.1-flash',
           'base_url': 'https://ollama.com/v1', 'temperature': 0,
           'reasoning_effort': 'high', 'max_output_tokens': 65536}
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': 'OK'},
                                                      'finish_reason': 'stop'}]})
    with OpenAI(api_key='synthetic-cloud-fixture', base_url=cfg['base_url'],
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as raw:
        reply = ProviderClient(raw, cfg).create(model=cfg['model'], messages=[{'role': 'user', 'content': 'OK'}])
    assert reply.choices[0].message.content == 'OK'
    assert str(captured[0].url) == 'https://ollama.com/v1/chat/completions'
    assert captured[0].headers['authorization'] == 'Bearer synthetic-cloud-fixture'
    body = json.loads(captured[0].content)
    assert body['max_tokens'] == 65536 and body['temperature'] == 0 and body['reasoning_effort'] == 'high'
    assert request_options(cfg, cfg['model'], {'max_tokens': 128})['max_tokens'] == 128
