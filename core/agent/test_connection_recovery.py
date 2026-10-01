import json
import ssl
import sys
from types import SimpleNamespace as NS

import httpx
import pytest

from core.agent import main
from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.autoresearch_runtime import AutoResearchRun
from core.llm.adapters import IncompleteModelResponse
from core.llm.connection_policy import retryable_connection_failure


class Cancel:
    def __init__(self, cancel_on_wait=False):
        self.cancelled = False
        self.cancel_on_wait = cancel_on_wait
        self.waits = []

    def is_set(self):
        return self.cancelled

    def wait(self, seconds):
        self.waits.append(seconds)
        self.cancelled = self.cancel_on_wait
        return self.cancelled


def connection():
    error = RuntimeError('Synthetic SDK wrapper')
    error.__cause__ = httpx.ConnectTimeout('Synthetic handshake timeout')
    return error


def test_connection_retry_continues_tools_once_and_counts_budget(tmp_path):
    (tmp_path / 'result.md').write_text('Synthetic draft')
    agent, calls = session(tmp_path, [reply(calls=[tool('read_workspace_file', path='result.md')]),
                                    connection(), reply(calls=[tool('finish_autoresearch', **completion())])])
    agent._cancel_event = Cancel()
    assert 'completed' in agent._chat()
    assert len(calls) == 3 and agent.autoresearch_state['iterations'] == 3
    assert agent.autoresearch_state['completed_rounds'] == 1
    assert len(agent.autoresearch_state['evidence']) == 1
    assert calls[1]['messages'] == calls[2]['messages']
    assert agent._cancel_event.waits == [1]


def test_three_attempt_ceiling_and_no_tool_dispatch(tmp_path):
    agent, calls = session(tmp_path, [connection(), connection(), connection()])
    agent._cancel_event = Cancel()
    assert 'interrupted' in agent._chat()
    assert len(calls) == 3 and agent.autoresearch_state['iterations'] == 3
    assert agent._cancel_event.waits == [1, 2]
    assert agent.autoresearch_state['completed_rounds'] == 0
    assert not agent.autoresearch_state['connection_failures'][-1]['retry_scheduled']
    assert not agent.autoresearch_state['evidence']


def test_retry_cannot_exceed_remaining_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS', '2')
    agent, calls = session(tmp_path, [connection(), connection()])
    agent._cancel_event = Cancel()
    agent._chat()
    assert len(calls) == agent.autoresearch_state['iteration_limit'] == 2
    assert agent.autoresearch_state['iterations'] == 2
    assert agent._cancel_event.waits == [1]


def test_cancellation_during_retry_wait_never_dispatches_again(tmp_path):
    agent, calls = session(tmp_path, [connection()])
    agent._cancel_event = Cancel(cancel_on_wait=True)
    assert 'cancelled' in agent._chat()
    assert len(calls) == 1 and agent.autoresearch_state['iterations'] == 1


@pytest.mark.parametrize('error', [httpx.ReadTimeout('unknown delivery'), httpx.WriteTimeout('partial write'),
                                 httpx.ReadError('partial stream'), httpx.RemoteProtocolError('EOF'),
                                 RuntimeError('Connection error'), IncompleteModelResponse('partial model response')])
def test_ambiguous_failures_never_retry(tmp_path, error):
    error.__cause__ = httpx.ConnectError('Misleading nested error')
    if type(error) is RuntimeError:
        error.__cause__ = None
    agent, calls = session(tmp_path, [error])
    agent._cancel_event = Cancel()
    agent._chat()
    assert len(calls) == 1 and not agent._cancel_event.waits


def test_http_and_certificate_failures_are_not_retried():
    error = httpx.ConnectError('bad certificate')
    error.__cause__ = ssl.SSLCertVerificationError(1, 'certificate failure')
    assert not retryable_connection_failure(error)
    error = connection()
    error.status_code = 429
    assert not retryable_connection_failure(error)
    cycle = RuntimeError()
    cycle.__cause__ = cycle
    assert not retryable_connection_failure(cycle)


def test_ollama_factory_uses_longer_connect_timeout_without_sdk_retries(monkeypatch):
    captured = []
    monkeypatch.setitem(sys.modules, 'openai', NS(OpenAI=lambda **kwargs: captured.append(kwargs) or NS()))
    main._build_openai_client({'provider': 'ollama_cloud', 'api_key': 'offline-fixture', 'base_url': 'https://ollama.com/v1'})
    assert captured[0]['timeout'].connect == 20
    assert captured[0]['timeout'].read == 600
    assert captured[0]['max_retries'] == 0


def test_observed_client_accounts_each_connection_attempt(tmp_path):
    from core.web.workbench import ObservedClient, WorkbenchStore
    agent, calls = session(tmp_path, [connection(), reply('OK')])
    store = WorkbenchStore(tmp_path / 'usage.sqlite')
    events = []
    agent._llm = ObservedClient(agent._llm, store, {'request_id': 'retry-test', 'chat_id': 'chat',
        'provider': 'offline', 'model': 'offline-test', 'source': 'main'}, events.append)
    agent._cancel_event = Cancel()
    agent._autoresearch_run = AutoResearchRun(tmp_path, 'idea')
    agent._autoresearch_run.begin_iteration()
    agent._context_chat_create('offline-test', [{'role': 'user', 'content': 'Synthetic'}], [])
    assert len(calls) == 2
    assert len([event for event in events if event['type'] == 'model_start']) == 2
    assert [event['status'] for event in events if event['type'] == 'model_end'] == ['failed', 'completed']
    assert agent._autoresearch_run.state['iterations'] == 2
