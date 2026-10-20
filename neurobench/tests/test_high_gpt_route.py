import json
import time

import httpx
import pytest

from neurobench.runner import high_channels, high_gpt_route
from neurobench.runner.artifacts import EventLog
from neurobench.runner.high_responses_observer import ResponsesObserver


def terminal(model='gpt-5.6-sol', effort='high'):
    return {'type': 'response.completed', 'response': {
        'id': 'offline-response', 'status': 'completed', 'model': model,
        'reasoning': {'effort': effort}, 'usage': {'input_tokens': 11, 'output_tokens': 5},
        'output': [{'id': 'offline-message', 'type': 'message', 'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'Offline fixture.'}]}]}}


def test_xnova_route_cannot_fall_back_to_subscription_or_send_to_another_host(tmp_path, monkeypatch):
    monkeypatch.setattr(high_gpt_route, 'route_key', lambda _: 'offline-key')
    monkeypatch.setattr(high_channels, 'get_access_record', lambda: pytest.fail('subscription credentials accessed'))
    sent = []

    def post(body, url, key, log, deadline, *, provider):
        sent.append((body, url, key, provider))
        return iter([terminal()])

    monkeypatch.setattr(high_channels, '_http', post)
    with_log = EventLog(tmp_path / 'events.jsonl')
    try:
        result, usage = high_channels.responses_round(
            [{'role': 'user', 'content': 'Offline.'}], [], with_log,
            time.monotonic() + 10, route=high_gpt_route.XNOVA_ROUTE)
        assert result['text'] == 'Offline fixture.' and usage['total_tokens'] == 16
        assert sent[0][1:] == ('https://api.xnova.online/v1/responses', 'offline-key', 'xnova_responses')
        assert sent[0][0]['reasoning'] == {'effort': 'high'}
        assert 'temperature' not in sent[0][0]
        wrong = dict(high_gpt_route.XNOVA_ROUTE, endpoint='https://wrong.example/v1/responses')
        with pytest.raises(ValueError, match='unapproved_gpt_experiment_route'):
            high_channels.responses_round([], [], with_log, time.monotonic() + 10, route=wrong)
        assert len(sent) == 1
    finally:
        with_log.close()


@pytest.mark.parametrize('model,effort', [('gpt-6-sol', 'high'), ('gpt-5.6-sol', 'medium')])
def test_provider_model_or_effort_substitution_is_held(tmp_path, monkeypatch, model, effort):
    monkeypatch.setattr(high_gpt_route, 'route_key', lambda _: 'offline-key')
    monkeypatch.setattr(high_channels, '_http', lambda *args, **kwargs: iter([terminal(model, effort)]))
    log = EventLog(tmp_path / 'events.jsonl')
    try:
        with pytest.raises(ValueError, match='native_model_or_high_echo_mismatch'):
            high_channels.responses_round([], [], log, time.monotonic() + 10, route=high_gpt_route.XNOVA_ROUTE)
    finally:
        log.close()


def test_cli_gateway_preserves_payload_and_records_third_party_route_without_credentials(tmp_path):
    sent = []
    event = terminal()

    def upstream(request):
        sent.append((str(request.url), request.content))
        assert request.headers['authorization'] == 'Bearer offline-upstream-key'
        return httpx.Response(200, text='data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n')

    observer = ResponsesObserver(tmp_path, 'offline-key', time.monotonic() + 10,
                                 upstream_url=high_gpt_route.XNOVA_ROUTE['endpoint'], upstream_token='offline-upstream-key')
    observer.client.close()
    observer.client = httpx.Client(transport=httpx.MockTransport(upstream))
    raw = b'{"model":"gpt-5.6-sol", "reasoning":{"effort":"high"},"input":[],"stream":true}'
    with observer, httpx.Client(trust_env=False) as client:
        response = client.post(observer.base_url + '/responses', content=raw,
                               headers={'authorization': 'Bearer offline-key'})
    assert response.status_code == 200
    assert sent == [(high_gpt_route.XNOVA_ROUTE['endpoint'], raw)]
    value = observer.summary()
    assert value['all_requests_completed'] and value['native_high_echo']
    assert not value['official_openai_endpoint']
    assert 'offline-key' not in (tmp_path / 'native_responses_events.jsonl').read_text()
    assert 'offline-upstream-key' not in (tmp_path / 'native_responses_events.jsonl').read_text()


@pytest.mark.parametrize('content', ['key1\nkey2', 'Bearer key1', ''])
def test_ambiguous_credential_content_is_rejected_without_echo(monkeypatch, content):
    monkeypatch.setattr(high_gpt_route.Path, 'read_text', lambda *_args, **_kwargs: content)
    with pytest.raises(ValueError) as error:
        high_gpt_route.route_key(high_gpt_route.XNOVA_ROUTE)
    assert str(error.value) == 'expected_one_plain_gpt_api_key'
