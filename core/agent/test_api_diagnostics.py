import json
import socket
import ssl

import httpx

from core.api_diagnostics import exception_diagnostic
from core.agent.main import _retry_api_call
from core.agent.test_autoresearch_execution import session


class APIConnectionError(Exception):
    def __init__(self, message, request):
        super().__init__(message)
        self.request = request


def connection_error():
    request = httpx.Request('POST', 'https://user:private-secret@example.test/v1?api_key=private-secret',
                            headers={'Authorization': 'Bearer private-secret'})
    tls = ssl.SSLEOFError(8, 'private-secret prompt and server response')
    transport = httpx.ConnectError('private-secret', request=request)
    transport.__cause__ = tls
    error = APIConnectionError(message='private-secret', request=request)
    error.__cause__ = transport
    return error


def test_tls_chain_without_any_payload_or_credentials():
    result = exception_diagnostic(connection_error())
    assert [item['type'] for item in result['exception_chain']] == ['APIConnectionError', 'ConnectError', 'SSLEOFError']
    assert result['category'] == 'tls' and result['http_status'] is None
    assert result['exception_chain'][-1]['tls_reason'] == 'UNEXPECTED_EOF_WHILE_READING'
    assert 'private-secret' not in json.dumps(result)
    assert 'example.test' not in json.dumps(result)


def test_context_cycle_depth_and_suppression():
    first, second = ValueError('private-secret'), RuntimeError('private-secret')
    first.__context__ = second
    second.__cause__ = first
    assert len(exception_diagnostic(first)['exception_chain']) == 2
    first.__suppress_context__ = True
    assert len(exception_diagnostic(first)['exception_chain']) == 1
    current = first
    for index in range(12):
        parent = RuntimeError('private-secret')
        parent.__cause__ = current
        current = parent
    assert len(exception_diagnostic(current)['exception_chain']) == 6


def test_http_dns_timeout_and_unknown_diagnostics():
    error = RuntimeError('private-secret')
    error.status_code = 429
    assert exception_diagnostic(error)['http_status'] == 429
    assert exception_diagnostic(socket.gaierror(-2, 'private-secret'))['category'] == 'dns'
    assert exception_diagnostic(httpx.ReadTimeout('private-secret'))['category'] == 'timeout'
    result = exception_diagnostic(ValueError('private-secret'))
    assert result['category'] == 'unknown' and 'private-secret' not in json.dumps(result)


def test_autoresearch_saves_chain_and_emits_event_without_replay(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS', '1')
    agent, calls = session(tmp_path, [connection_error()])
    events = []
    agent._execution_observer = events.append
    assert 'interrupted' in agent._chat()
    assert len(calls) == 1
    state = json.loads(agent._autoresearch_run.path.read_text(encoding='utf-8'))
    assert state['last_error']['category'] == 'tls'
    assert state['last_error']['exception_chain'][0]['type'] == 'APIConnectionError'
    assert any(event['type'] == 'api_diagnostic' for event in events)
    output = capsys.readouterr().out
    assert '[api_diagnostic]' in output
    assert 'private-secret' not in output + json.dumps(state) + json.dumps(events)


def test_retry_logging_does_not_change_attempt_policy(capsys):
    attempts = []
    def fail():
        attempts.append(1)
        raise connection_error()
    try:
        _retry_api_call('test', fail, retries=1)
    except APIConnectionError:
        pass
    assert len(attempts) == 1
    output = capsys.readouterr().out
    assert '"attempt": 1' in output and '"max_attempts": 1' in output
    assert 'SSLEOFError' in output and 'private-secret' not in output
