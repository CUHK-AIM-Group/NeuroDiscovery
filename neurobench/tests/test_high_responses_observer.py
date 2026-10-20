import json
import time

import httpx
import pytest

from neurobench.runner.high_responses_observer import ResponsesObserver


def fixture_response(model='gpt-5.6-sol', effort='high'):
    return {'type': 'response.completed', 'response': {
        'id': 'native-fixture', 'model': model, 'reasoning': {'effort': effort},
        'usage': {'input_tokens': 19, 'output_tokens': 7}, 'output': []}}


def test_native_request_bytes_and_response_evidence_are_preserved(tmp_path):
    sent = []
    event = fixture_response()

    def upstream(request):
        sent.append((str(request.url), request.content))
        assert request.headers['authorization'] == 'Bearer test-only-credential'
        return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n')

    observer = ResponsesObserver(tmp_path, 'test-only-credential', time.monotonic() + 10)
    observer.client.close()
    observer.client = httpx.Client(transport=httpx.MockTransport(upstream))
    raw = b'{"model":"gpt-5.6-sol", "reasoning":{"effort":"high"},"input":[],"stream":true}'
    with observer:
        with httpx.Client(trust_env=False) as client:
            response = client.post(observer.base_url + '/responses', content=raw,
                                   headers={'authorization': 'Bearer test-only-credential'})
        assert response.status_code == 200 and json.dumps(event) in response.text
    assert sent == [('https://api.openai.com/v1/responses', raw)]
    result = observer.summary()
    assert result['all_requests_completed'] and result['native_high_echo']
    text = (tmp_path / 'native_responses_events.jsonl').read_text()
    assert 'test-only-credential' not in text and 'Authorization' not in text
    assert '"input_tokens": 19' in text


@pytest.mark.parametrize('change,status', [('credential', 401), ('model', 400), ('effort', 400), ('path', 404)])
def test_invalid_auth_or_settings_never_reach_upstream(tmp_path, change, status):
    def forbidden(_):
        pytest.fail('invalid request reached upstream')
    observer = ResponsesObserver(tmp_path, 'test-only-credential', time.monotonic() + 10)
    observer.client.close()
    observer.client = httpx.Client(transport=httpx.MockTransport(forbidden))
    body = {'model': 'gpt-5.6-sol', 'reasoning': {'effort': 'high'}}
    if change == 'model': body['model'] = 'different-model'
    if change == 'effort': body['reasoning']['effort'] = 'low'
    key = 'wrong' if change == 'credential' else 'test-only-credential'
    path = '/elsewhere' if change == 'path' else '/responses'
    with observer, httpx.Client(trust_env=False) as client:
        response = client.post(observer.base_url + path, json=body, headers={'authorization': 'Bearer ' + key})
        assert response.status_code == status
    assert observer.summary()['native_request_count'] == 0


def test_upstream_error_and_missing_terminal_are_not_complete(tmp_path):
    observer = ResponsesObserver(tmp_path, 'test-only-credential', time.monotonic() + 10)
    observer.client.close()
    observer.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(429, text='quota fixture')))
    with observer, httpx.Client(trust_env=False) as client:
        response = client.post(observer.base_url + '/responses', json={'model': 'gpt-5.6-sol', 'reasoning': {'effort': 'high'}},
                               headers={'authorization': 'Bearer test-only-credential'})
        assert response.status_code == 429
    value = observer.summary()
    assert value['native_request_count'] == 1 and value['native_completed_response_count'] == 0
    assert not value['all_requests_completed'] and not value['native_high_echo']
