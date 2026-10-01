from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from core.runtime_store import RuntimeStore
from core.agent.test_autoresearch_execution import session, reply, tool, completion
from core.session.context import compact_context


def payload(request='request_first_123456'):
    return {'request_id': request, 'chat_id': 'chat', 'message': 'test', 'workspace_path': 'workspace'}


def test_queue_idempotency_fifo_ownership_and_pause(tmp_path):
    store = RuntimeStore(tmp_path / 'runtime.db')
    first = store.enqueue(payload())
    assert store.enqueue(payload()) == first
    with pytest.raises(ValueError):
        store.enqueue({**payload(), 'message': 'changed'})
    store.enqueue(payload('request_second_123456'))
    with ThreadPoolExecutor(4) as pool:
        claims = list(pool.map(lambda owner: store.claim('chat', str(owner)), range(4)))
    assert sum(row is not None for row in claims) == 1
    running = store.get(first['request_id'])
    with pytest.raises(ValueError):
        store.finish(first['request_id'], 'not-owner', {}, 'completed')
    store.finish(first['request_id'], running['owner'], {}, 'failed')
    assert store.claim('chat', 'next') is None
    assert store.list('chat')[-1]['status'] == 'paused'
    store.continue_queue('chat')
    assert store.claim('chat', 'next')['request_id'] == 'request_second_123456'


def test_crash_uncertainty_and_atomic_safe_checkpoint(tmp_path):
    store = RuntimeStore(tmp_path / 'runtime.db')
    request = payload()['request_id']
    store.enqueue(payload())
    store.claim('chat', 'owner')
    store.begin_step(request, 'tool:1', 'tool', {'command': 'synthetic'})
    store.recover_owner('owner')
    assert store.recovery(request)['uncertain_steps'] == ['tool:1']
    with pytest.raises(ValueError, match='Uncertain'):
        store.resumable(request, 'chat')
    checkpoint = {'messages': [{'role': 'user', 'content': 'test'}], 'workspace': str(tmp_path)}
    store.end_step(request, 'tool:1', {'success': True}, checkpoint, 'chat')
    assert store.resumable(request, 'chat') == checkpoint
    store.enqueue(payload('request_second_123456'))
    resumed = {**payload('request_recovery_123456'), 'recovery_from': request}
    store.enqueue_recovery(resumed, request, checkpoint)
    with pytest.raises(ValueError):
        store.enqueue_recovery({**resumed, 'request_id': 'duplicate_recovery_123456'}, request, checkpoint)
    assert store.claim('chat', 'new')['request_id'] == resumed['request_id']
    store.finish(resumed['request_id'], 'new', {'content': 'recovered result'}, 'completed')
    assert store.history_for('request_second_123456')[-1]['content'] == 'recovered result'


def test_partial_tool_transaction_is_not_replayed(tmp_path):
    store = RuntimeStore(tmp_path / 'runtime.db')
    request = payload()['request_id']
    store.enqueue(payload()); store.claim('chat', 'owner')
    store.checkpoint(request, 'chat', {'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'unexecuted'}]}]})
    store.recover_owner('owner')
    with pytest.raises(ValueError, match='Partial tool transaction'):
        store.resumable(request, 'chat')


def test_permissions_are_exact_one_time_and_fail_closed(tmp_path):
    agent, _ = session(tmp_path, [], mode='off')
    store = RuntimeStore(tmp_path / 'runtime.db')
    agent._runtime_store = store; agent._runtime_request_id = 'request'; agent.permission_mode = 'ask'
    events = []; agent._execution_observer = events.append
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(agent._tool_authorized, 'run_shell_command', {'command': 'synthetic'}, 'step')
        for _ in range(100):
            if store.approval('request', 'step') == 'pending':
                break
            threading.Event().wait(.01)
        assert not store.decide('other', 'step', True)
        assert store.decide('request', 'step', True)
        assert not store.decide('request', 'step', False)
        assert pending.result(2)
    agent.permission_mode = 'read_only'
    assert not agent._tool_authorized('run_shell_command', {}, 'denied')
    assert not agent._tool_authorized('read_workspace_file', {'path': '../outside'}, 'outside')
    assert not agent._tool_authorized('spawn_subagent', {}, 'child')


def test_tool_outputs_and_model_messages_are_durable(tmp_path):
    (tmp_path / 'result.md').write_text('synthetic')
    agent, _ = session(tmp_path, [reply(calls=[tool('read_workspace_file', path='result.md')]), reply('finished')], mode='off')
    agent._runtime_store = RuntimeStore(tmp_path / 'runtime.db')
    agent._runtime_request_id = 'request'; agent._runtime_chat_id = 'chat'
    assert agent._chat() == 'finished'
    recovery = agent._runtime_store.recovery('request')
    assert not recovery['uncertain_steps']
    assert recovery['checkpoint']['messages'][-1]['content'] == 'finished'
    assert len(recovery['steps']) == 3


def test_semantic_summary_fallback_and_archive(tmp_path):
    original = [{'role': 'system', 'content': 'protected'}]
    original += [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'x' * 8000}, {'role': 'user', 'content': 'current'}]
    messages = list(original)
    result = compact_context(messages, tmp_path, 2048, summarizer=lambda raw: '')
    assert result['method'] == 'extractive'
    messages = list(original)
    messages[1]['content'] = 'x' * 5000
    messages[2] = {'role': 'assistant', 'content': 'y' * 2000}
    messages.append({'role': 'assistant', 'content': 'z' * 4000})
    messages.append({'role': 'user', 'content': 'next'})
    result = compact_context(messages, tmp_path, 3000, summarizer=lambda raw: 'Unresolved evidence, retain uncertainty.')
    assert result['method'] == 'semantic'
    assert messages[0]['content'] == 'protected' and messages[-1]['content'] == 'next'


def test_context_overflow_retries_smaller_payload_once(tmp_path):
    error = RuntimeError('synthetic too long'); error.code = 'context_length_exceeded'
    agent, calls = session(tmp_path, [error, reply('ok')], mode='off')
    messages = [{'role': 'system', 'content': 'protected'}, {'role': 'user', 'content': 'old'},
                {'role': 'assistant', 'content': 'x' * 20000}, {'role': 'user', 'content': 'current'}]
    assert agent._context_chat_create('offline', messages, []).choices[0].message.content == 'ok'
    assert len(calls) == 2
    assert len(json.dumps(calls[1]['messages'])) < len(json.dumps(calls[0]['messages']))


def test_bounded_independent_review_correction(tmp_path):
    (tmp_path / 'result.md').write_text('synthetic report')
    agent, calls = session(tmp_path, [
        reply(calls=[tool('read_workspace_file', path='result.md')]),
        reply(calls=[tool('finish_autoresearch', **completion())]),
        reply(json.dumps({'accepted': False, 'reason': 'Check missing scope'})),
        reply(calls=[tool('finish_autoresearch', **completion())]),
        reply(json.dumps({'accepted': True, 'reason': 'Separate acceptance passed'})),
    ])
    agent.env.update(autoresearch_independent_review=True, autoresearch_repair_limit=1)
    agent._chat()
    assert agent.autoresearch_state['status'] == 'completed'
    assert agent.autoresearch_state['repairs'] == 1
    assert len(agent.autoresearch_state['review_history']) == 1
    assert len(calls) == 5


def test_recovery_preserves_research_evidence_and_budget(tmp_path):
    (tmp_path / 'result.md').write_text('synthetic report')
    agent, _ = session(tmp_path, [reply(calls=[tool('read_workspace_file', path='result.md')]), reply(calls=[tool('finish_autoresearch', **completion())])])
    agent._runtime_store = RuntimeStore(tmp_path / 'runtime.db')
    agent._runtime_request_id = 'request'; agent._runtime_chat_id = 'chat'
    agent._chat()
    checkpoint = agent._runtime_store.recovery('request')['checkpoint']
    assert checkpoint['autoresearch']['evidence'][0]['id'] == 1
    recovered, calls = session(tmp_path, [])
    checkpoint['autoresearch']['iteration_limit'] = checkpoint['autoresearch']['iterations']
    recovered._recovery_state = checkpoint['autoresearch']
    recovered._chat()
    assert recovered.autoresearch_state['status'] == 'budget_exhausted'
    assert not calls


def test_summary_fragments_are_tool_free_and_budgeted(tmp_path):
    from core.autoresearch_runtime import AutoResearchRun
    agent, calls = session(tmp_path, [reply('Compact unresolved evidence')])
    agent._autoresearch_run = AutoResearchRun(tmp_path, 'data')
    agent._runtime_store = RuntimeStore(tmp_path / 'runtime.db')
    agent._runtime_request_id = 'summary-request'
    assert agent._summarize_context('synthetic transcript') == 'Compact unresolved evidence'
    assert 'tools' not in calls[0]
    assert agent.autoresearch_state['iterations'] == 1
    assert not agent._runtime_store.recovery('summary-request')['uncertain_steps']
