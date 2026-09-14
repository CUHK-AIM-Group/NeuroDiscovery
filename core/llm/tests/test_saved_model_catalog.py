"""Reading/switching saved models never performs discovery."""
from copy import deepcopy
import io

from core.llm.tests.test_client_configuration import api  # shared offline fixture
from core.llm.provider_profiles import apply_openai_compatible_profile_defaults
from core.web import server
from core.agent.main import _normalize_llm_backend, build_llm_client
import pytest


def test_get_env_and_switch_use_only_saved_models(api, monkeypatch):
    env = {'llm_backend': {'provider': 'openai', 'model': 'saved-one', 'api_key_env': 'UNSET_TEST',
           'base_url': 'https://fixture.example/v1', 'model_selection_managed': True,
           'available_models': [{'provider': 'openai', 'model': 'saved-one'}, {'provider': 'openai', 'model': 'saved-two'}]}}
    monkeypatch.setattr(server, 'load_environment', lambda: deepcopy(env))
    saved = []
    monkeypatch.setattr(server, 'save_environment', lambda value: saved.append(deepcopy(value)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Chat menus must not contact the provider')
    monkeypatch.setattr(server.urllib.request, 'urlopen', forbidden)
    data = api.get('/api/env').json()
    assert [item['model'] for item in data['available_models']] == ['saved-one', 'saved-two']
    assert not data['model_probe']['attempted']
    assert api.post('/api/env/model', json={'provider': 'openai', 'model': 'saved-two'}).status_code == 200
    assert api.post('/api/env/model', json={'provider': 'openai', 'model': 'not-added'}).status_code == 400
    assert len(saved) == 1 and saved[0]['llm_backend']['model'] == 'saved-two'


def test_explicit_legacy_discovery_does_not_expand_saved_models(api, monkeypatch):
    env = {'llm_backend': {'provider': 'openai', 'model': 'saved', 'base_url': 'https://fixture.example/v1',
           'available_models': [{'provider': 'openai', 'model': 'saved'}]}}
    monkeypatch.setattr(server, 'load_environment', lambda: deepcopy(env))
    monkeypatch.setattr(server, 'save_environment', lambda value: (_ for _ in ()).throw(AssertionError('Discovery must not save')))
    monkeypatch.setattr(server.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'{"data":[{"id":"unselected"}]}'))
    assert api.get('/api/env/models').json()['available_models'][0]['model'] == 'unselected'
    assert api.get('/api/env').json()['available_models'][0]['model'] == 'saved'


def test_explicitly_empty_library_is_not_repopulated_from_provider_presets(api, monkeypatch):
    cfg = {'provider': 'ollama_cloud', 'model': '', 'model_selection_managed': True, 'available_models': []}
    apply_openai_compatible_profile_defaults(cfg)
    assert cfg['model'] == '' and cfg['available_models'] == []
    monkeypatch.setattr(server, 'load_environment', lambda: {'llm_backend': cfg})
    assert api.get('/api/env').json()['available_models'] == []


@pytest.mark.parametrize('selected', [[], ['chosen-one', 'chosen-two']])
def test_full_environment_normalization_respects_desktop_selection(selected):
    env = {'llm_backend': {'provider': 'ollama_cloud', 'model': 'old-unselected',
           'base_url': 'https://ollama.com/v1', 'model_selection_managed': True,
           'available_models': selected},
           'legacy': {'api': 'openai-completions', 'baseUrl': 'https://fixture.example/v1',
                      'models': ['not-chosen'], 'apiKey': 'synthetic-legacy-key'}}
    _normalize_llm_backend(env)
    llm = env['llm_backend']
    assert [item['model'] for item in llm['available_models']] == selected
    assert llm['model'] == (selected[0] if selected else '')
    assert not llm.get('api_key')
    _normalize_llm_backend(env)
    assert [item['model'] for item in llm['available_models']] == selected
    if not selected:
        with pytest.raises(RuntimeError, match='Add a model in Settings'):
            build_llm_client(env)
