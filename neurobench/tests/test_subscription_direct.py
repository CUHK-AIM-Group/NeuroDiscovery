import copy
import contextlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobench.runner.subscription_direct import completed_response, response_body


def events(item, **response_fields):
    return [{'type': 'response.output_item.done', 'output_index': 0, 'item': item},
            {'type': 'response.completed', 'response': {'status': 'completed', 'output': [], **response_fields}}]


def test_actual_saved_native_smoke_without_another_request():
    p = Path(__file__).resolve().parents[2] / 'tmp/chatgpt_plan_direct_20261009/GPT_5_6_SOL_HIGH_SMOKE.json'
    if not p.is_file():
        pytest.skip('Requires the retained local native smoke receipt; no request is replayed')
    r = completed_response(json.loads(p.read_text(encoding='utf-8'))['events'])
    assert r['text'] == 'NEUROCLAW_DIRECT_OK' and not r['calls']


def test_function_only_completed_output():
    c = {'type': 'function_call', 'id': 'f1', 'call_id': 'c1', 'name': 'describe_task_contract', 'namespace': 'neuro_runtime', 'arguments': '{}'}
    r = completed_response(events(c))
    assert r['calls'] == [c] and not r['text']


def test_stateless_reasoning_item_is_not_discarded():
    e = events({'type': 'reasoning', 'id': 'r1', 'encrypted_content': 'opaque'})
    e.insert(1, {'type': 'response.output_item.done', 'output_index': 1, 'item': {'type': 'message', 'role': 'assistant', 'id': 'm1', 'content': [{'type': 'output_text', 'text': 'OK'}]}})
    assert completed_response(e)['output'][0]['encrypted_content'] == 'opaque'


def test_argument_delta_must_agree():
    c = {'type': 'function_call', 'id': 'f1', 'call_id': 'c1', 'name': 'validate_proposal', 'arguments': '{}'}
    e = events(c)
    e.insert(0, {'type': 'response.function_call_arguments.done', 'item_id': 'f1', 'output_index': 0, 'arguments': '{"wrong":1}'})
    with pytest.raises(ValueError, match='arguments_done_mismatch'):
        completed_response(e)


def test_wrong_namespace_rejected():
    with pytest.raises(ValueError, match='unexpected_function_namespace'):
        completed_response(events({'type': 'function_call', 'id': 'f1', 'call_id': 'c1', 'name': 'f', 'namespace': 'other', 'arguments': '{}'}))


def test_function_arguments_array_rejected():
    with pytest.raises(ValueError, match='arguments_not_object'):
        completed_response(events({'type': 'function_call', 'id': 'f1', 'call_id': 'c1', 'name': 'f', 'arguments': '[]'}))


def test_high_subscription_request_has_no_unsupported_output_cap():
    r = response_body('gpt-5.6-sol', [{'role': 'user', 'content': 'task'}], [{'name': 'check', 'parameters': {'type': 'object', 'properties': {}}, 'description': 'test'}])
    assert r['reasoning'] == {'effort': 'high'} and r['store'] is False and r['stream'] is True
    assert 'max_output_tokens' not in r and r['tools'][0]['type'] == 'namespace'
    assert r['tools'][0]['tools'][0]['type'] == 'function'


@pytest.mark.parametrize('kind', ['response.failed', 'response.incomplete', 'error'])
def test_non_completed_terminal_not_scored(kind):
    with pytest.raises(Exception):
        completed_response([{'type': kind, 'response': {'status': 'failed'}}])


def test_no_terminal_is_unknown():
    with pytest.raises(Exception):
        completed_response([{'type': 'response.output_text.delta', 'delta': 'partial'}])


def test_duplicate_terminal_rejected():
    e = events({'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'ok'}]})
    with pytest.raises(Exception):
        completed_response(e + [copy.deepcopy(e[-1])])


@pytest.fixture
def refresh_fixture(monkeypatch, tmp_path):
    from neurobench.runner import subscription_direct as s
    previous = {'issuer': 'https://auth.openai.com', 'client_id': 'oaiapp_test', 'subject': 'test_subject',
                'ext_agent_host_id': 'urn:uuid:test', 'refresh_token': 'fake_refresh',
                'scopes': ['resource.invoke', 'chatgpt.tokens.use.direct'], 'expires_at': 0}
    saved = []
    class PlanError(Exception):
        pass
    def atomic(path, value, **kwargs):
        Path(path).write_text(json.dumps(value), encoding='utf-8')
    def registration(tokens, jwks, client_id, host_id, nonce, prior):
        assert client_id == prior['client_id'] and host_id == prior['ext_agent_host_id'] and nonce is None
        assert prior['subject'] == 'test_subject'
        return {**prior, 'scopes': tokens['scope'].split(), 'expires_at': time.time() + 3600}
    auth = SimpleNamespace(PlanError=PlanError, ISSUER=previous['issuer'], DIRECT_SCOPE='chatgpt.tokens.use.direct',
        RESOURCE='https://api.openai.com/v1', TOKEN='https://auth.openai.com/api/accounts/oauth/token',
        session_lock=lambda root: contextlib.nullcontext(), active_record=lambda root: previous,
        discovery=lambda client: {'jwks_uri': 'https://auth.openai.com/.well-known/jwks.json'},
        checked_json=lambda response, op: {}, utc=lambda: 'time', atomic_json=atomic,
        registration=registration, save_record=lambda root,r: saved.append(r))
    class Client:
        def __init__(self): self.posts = []; self.tokens = {'access_token': 'fake_access'}
        def get(self, url): return None
        def post(self, url, data):
            self.posts.append((url, data))
            return SimpleNamespace(status_code=200, json=lambda: self.tokens)
    client = Client()
    monkeypatch.setattr(s, 'auth_helper', lambda: auth)
    return s, tmp_path, previous, saved, client


def test_refresh_uses_same_issued_client_and_retained_scope(refresh_fixture):
    s, root, old, saved, client = refresh_fixture
    result = s.get_access_record(root=root, client=client)
    assert len(client.posts) == len(saved) == 1
    assert client.posts[0][1]['client_id'] == old['client_id']
    assert 'scope' not in client.posts[0][1]
    assert result['scopes'] == old['scopes']
    journal = json.loads((root/'refresh_transaction.json').read_text())
    assert journal['state'] == 'COMPLETE' and 'fake_refresh' not in json.dumps(journal)


def test_unexpired_access_does_not_refresh(refresh_fixture):
    s, root, old, saved, client = refresh_fixture
    old['expires_at'] = time.time() + 3600
    assert s.get_access_record(root=root, client=client) is old
    assert not client.posts and not saved


def test_scope_escalation_is_not_stored(refresh_fixture):
    s, root, old, saved, client = refresh_fixture
    client.tokens['scope'] = 'resource.invoke chatgpt.tokens.use.direct new_scope'
    with pytest.raises(Exception, match='validation_failed'):
        s.get_access_record(root=root, client=client)
    assert not saved
    assert json.loads((root/'refresh_transaction.json').read_text())['state'] == 'UNKNOWN'


def test_unknown_refresh_is_not_dispatched_again(refresh_fixture):
    s, root, old, saved, client = refresh_fixture
    (root/'refresh_transaction.json').write_text('{"state":"UNKNOWN"}')
    with pytest.raises(Exception, match='remote_unknown'):
        s.get_access_record(root=root, client=client)
    assert not client.posts
