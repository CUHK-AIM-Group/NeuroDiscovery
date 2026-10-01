"""Offline workbench evidence: synthetic wire events, temporary SQLite, no keys."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient
import pytest

from core.agent import main
from core.llm.adapters import ProviderClient, IncompleteModelResponse, _usage, objects
from core.web import server
from core.web.workbench import WorkbenchStore, WorkbenchRun, ObservedClient, StateConflict, normalize_usage, observe_local_call


@pytest.mark.parametrize("raw,protocol,expected", [
    (None, "chat_completions", (None, None, None)),
    ({"prompt_tokens": 0, "completion_tokens": 0}, "chat_completions", (0, 0, None)),
    ({"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 80}}, "chat_completions", (100,20,80)),
    ({"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 80}}, "responses", (100,20,80)),
    ({"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 10}, "anthropic", (100,20,80)),
    ({"input_tokens": 5}, "responses", (5,None,None)),
    ({"prompt_tokens": True, "completion_tokens": -1}, "chat_completions", (None,None,None)),
])
def test_usage_accounting(raw, protocol, expected):
    usage = normalize_usage(raw, protocol)
    assert (usage["input_tokens"],usage["output_tokens"],usage["cache_read_tokens"]) == expected
    assert usage["reported"] == (expected[0] is not None and expected[1] is not None)


def test_adapter_compatibility_zero_is_not_reported_zero():
    assert normalize_usage(_usage(None, "responses"))["input_tokens"] is None
    assert normalize_usage(_usage({"input_tokens":7}, "responses"))["output_tokens"] is None
    raw = {"input_tokens": 10, "output_tokens":20,"cache_read_input_tokens":80,"cache_creation_input_tokens":10}
    assert normalize_usage(_usage(raw,"anthropic"))["input_tokens"] == 100


def test_durable_state_cas_and_ledger_independent_of_transcripts(tmp_path):
    store = WorkbenchStore(tmp_path / "client.db")
    state = {"sessions":[{"id":"s","draft":"unsent"}],"projects":[]}
    assert store.state() == {"revision":0,"state":None}
    assert store.save_state(0,state) == {"revision":1}
    with pytest.raises(StateConflict) as conflict: store.save_state(0,state)
    assert store.recovery(conflict.value.recovery_id) == state
    store.start_call("c",{"chat_id":"s","model":"m","provider":"p","source":"main"})
    store.finish_call("c","completed",normalize_usage({"prompt_tokens":100,"completion_tokens":20}))
    store.finish_call("c","completed",normalize_usage({"prompt_tokens":100,"completion_tokens":20}))
    store.save_state(1,{"sessions":[],"projects":[]})
    restored = WorkbenchStore(store.path)
    assert restored.usage()["totals"]["input_tokens"] == 100
    assert restored.usage()["totals"]["requests"] == 1
    assert restored.state()["state"]["sessions"] == []


def test_concurrent_windows_cannot_overwrite_each_other(tmp_path):
    store = WorkbenchStore(tmp_path / "client.db")
    def write():
        try: return store.save_state(0,{"sessions":[],"projects":[]})
        except StateConflict: return "conflict"
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda _:write(),range(2)))
    assert results.count("conflict") == 1
    assert store.state()["revision"] == 1


class Stream:
    def __init__(self, events): self.events,self.closed = events,False
    def __iter__(self):
        for event in self.events:
            if isinstance(event, Exception): raise event
            yield objects(event)
    def close(self): self.closed=True


def chat_stream(*, incomplete=False, usage=True):
    events=[{"choices":[{"delta":{"reasoning_content":"Public provider summary"}}]},
            {"choices":[{"delta":{"content":"Synthetic output"},"finish_reason":"length" if incomplete else "stop"}]}]
    if usage: events.append({"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":80}}})
    return Stream(events)


def observed(tmp_path, stream, cancel=None):
    calls=[]
    def create(**kwargs): calls.append(kwargs); return stream
    client=ProviderClient(NS(chat=NS(completions=NS(create=create))),{"provider":"openai","model":"gpt-test","api_mode":"chat_completions"})
    store=WorkbenchStore(tmp_path / "client.db")
    events=[]
    wrapper=ObservedClient(client,store,{"request_id":"request","chat_id":"chat","model":"gpt-test","provider":"openai","source":"main"},events.append,cancel)
    return wrapper,store,events,calls


def test_stream_preserves_protocol_and_records_once(tmp_path):
    stream=chat_stream()
    wrapper,store,events,calls=observed(tmp_path,stream)
    result=wrapper.chat.completions.create(model="gpt-test",messages=[{"role":"user","content":"Synthetic"}])
    assert result.choices[0].message.content == "Synthetic output"
    assert result.choices[0].message.reasoning_content == "Public provider summary"
    assert calls[0]["stream"] is True and stream.closed
    assert {e["type"] for e in events} == {"model_start","text","reasoning","model_end","usage"}
    totals=store.usage()["totals"]
    assert (totals["requests"],totals["input_tokens"],totals["cache_read_tokens"],totals["completed"]) == (1,100,80,1)
    assert "base_url" not in store.usage()["calls"][0]


def test_incomplete_stream_keeps_reported_usage_without_tool_execution(tmp_path):
    wrapper,store,events,calls=observed(tmp_path,chat_stream(incomplete=True))
    with pytest.raises(IncompleteModelResponse): wrapper.create(model="gpt-test",messages=[])
    assert len(calls)==1
    assert store.usage()["totals"]["failed"]==1
    assert store.usage()["totals"]["input_tokens"]==100


def test_transport_failure_after_partial_is_not_replayed(tmp_path):
    stream=Stream([{"choices":[{"delta":{"content":"partial"}}]}, TimeoutError("fixture")])
    wrapper,store,events,calls=observed(tmp_path,stream)
    with pytest.raises(IncompleteModelResponse):
        main._retry_api_call("test",lambda:wrapper.create(model="gpt-test",messages=[]),retries=3)
    assert len(calls)==1 and stream.closed
    assert store.usage()["totals"]["unreported"]==1
    assert next(e for e in events if e["type"]=="text")["text"]=="partial"


def test_cancel_before_dispatch_costs_nothing(tmp_path):
    cancel=threading.Event(); cancel.set()
    wrapper,store,events,calls=observed(tmp_path,chat_stream(),cancel)
    with pytest.raises(RuntimeError): wrapper.create(model="gpt-test",messages=[])
    assert not calls and store.usage()["totals"]["requests"]==0


def test_auxiliary_identity_and_unreported_usage(tmp_path):
    wrapper,store,events,calls=observed(tmp_path,chat_stream(usage=False))
    wrapper.create(model="small-model",messages=[])
    data=store.usage(model="small-model")
    assert data["calls"][0]["source"]=="auxiliary"
    assert data["totals"]["unreported"]==1
    assert data["calls"][0]["input_tokens"] is None
    assert store.usage(model="gpt-test")["totals"]["requests"]==0


def test_sdk_retries_are_disabled_only_on_the_observed_copy(tmp_path):
    configurations=[]
    class Raw:
        def __init__(self): self.chat=NS(completions=NS(create=lambda **kwargs:chat_stream()))
        def with_options(self,**kwargs): configurations.append(kwargs); return Raw()
    raw=Raw(); client=ProviderClient(raw,{"provider":"openai","model":"gpt-test","api_mode":"chat_completions"})
    wrapped=ObservedClient(client,WorkbenchStore(tmp_path/'client.db'),{"model":"gpt-test"},lambda event:None)
    assert configurations==[{"max_retries":0}]
    assert wrapped.client is not client and client.raw is raw


def test_cancellation_during_model_preserves_usage_already_received(tmp_path):
    cancellation=threading.Event()
    class CancelStream(Stream):
        def __iter__(self):
            cancellation.set()
            yield objects({"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}})
    wrapper,store,events,calls=observed(tmp_path,CancelStream([]),cancellation)
    with pytest.raises(IncompleteModelResponse): wrapper.create(model='gpt-test',messages=[])
    data=store.usage()
    assert data['totals']['cancelled']==1
    assert data['totals']['input_tokens']==12


def test_visible_reasoning_never_exports_signature_or_encrypted_blocks(tmp_path):
    from core.llm.tests.test_adapters import fake, MESSAGES
    events=[{"type":"response.reasoning_summary_text.delta","delta":"Visible summary"},
            {"type":"response.output_item.added","item":{"type":"reasoning","encrypted_content":"OPAQUE_SECRET"}},
            {"type":"response.output_text.delta","delta":"done"},
            {"type":"response.completed","response":{"status":"completed","output":[
                {"type":"reasoning","summary":[{"type":"summary_text","text":"Visible summary"}],"encrypted_content":"OPAQUE_SECRET"},
                {"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":11,"output_tokens":7}}}]
    native,_=fake('openai',[Stream(events)])
    public=[]
    wrapper=ObservedClient(native,WorkbenchStore(tmp_path/'client.db'),{"model":native.cfg['model']},public.append)
    wrapper.create(model=native.cfg['model'],messages=MESSAGES)
    assert 'Visible summary' in json.dumps(public)
    assert 'OPAQUE_SECRET' not in json.dumps(public)


def test_delegated_client_keeps_subagent_source(tmp_path):
    wrapper,store,events,calls=observed(tmp_path,chat_stream())
    wrapper.meta['source']='subagent:fixture-child'
    wrapper.create(model='gpt-test',messages=[])
    assert store.usage()['calls'][0]['source']=='subagent:fixture-child'


def test_legacy_local_response_is_observed_without_changing_wire_payload(tmp_path):
    data={"message":{"content":"Synthetic local output","thinking":"Public local summary"},
          "prompt_eval_count":12,"eval_count":4}
    calls=[]; events=[]; store=WorkbenchStore(tmp_path/'client.db')
    def request(): calls.append(True); return data
    result=observe_local_call(request,store,{"model":"local-test","provider":"local"},events.append)
    assert result is data and len(calls)==1
    assert store.usage()['totals']['input_tokens']==12
    assert store.usage()['totals']['output_tokens']==4
    assert next(e for e in events if e['type']=='reasoning')['text']=='Public local summary'


def test_replay_compaction_and_restart_do_not_dispatch(tmp_path):
    store=WorkbenchStore(tmp_path / "client.db")
    run=WorkbenchRun("r","s",store)
    for _ in range(1030): run.emit({"type":"text","call_id":"c","text":"a"})
    snapshot=run.poll(1)["snapshot"]
    assert snapshot["blocks"][0]["text"]=="a"*1030
    store.start_call("c",{"request_id":"r"})
    run.store.save_run(run)
    assert store.saved_run("r")["status"]=="interrupted"
    assert store.usage()['totals']['running']==0
    assert store.usage()['totals']['interrupted']==1
    assert store.saved_run('r')['result']['usage']['totals']['unreported']==1
    run.finish({"type":"done","content":"complete"},"completed")
    assert store.saved_run("r")["result"]["content"]=="complete"
    assert run.poll(run.seq)["events"]==[]


def test_real_executor_emits_tool_lifecycle_without_changing_science(tmp_path):
    from core.agent.test_autoresearch_execution import session, reply, tool, completion
    (tmp_path / "result.md").write_text("Synthetic output",encoding="utf-8")
    agent,calls=session(tmp_path,[reply(calls=[tool("inspect_local_path",path="result.md")]),reply(calls=[tool("finish_autoresearch",**completion())])])
    events=[]; agent._execution_observer=events.append
    assert "completed" in agent._chat()
    assert [e["type"] for e in events]==["tool_start","tool_end","tool_start","tool_end"]
    assert events[0]["tool_id"]==events[1]["tool_id"]
    assert events[1]["status"]=="completed"


@pytest.fixture
def api(monkeypatch,tmp_path):
    instances=[]
    class SyntheticSession:
        def __init__(self,**kwargs):
            self.constructor_options=kwargs
            self.env={"llm_backend":{"provider":"openai","model":"gpt-test"}}
            self.history=[]; self._llm=None; self._cancel_event=threading.Event()
            self.started=threading.Event(); self.release=threading.Event(); self.autoresearch_state=None
            instances.append(self)
        def set_llm_client(self,client): self._llm=client
        def request_cancel(self): self._cancel_event.set(); self.release.set()
        def steer(self, message): self.steering = message
        def _chat(self):
            self._llm.chat.completions.create(model="gpt-test",messages=self.history)
            self.started.set()
            if self.history[-1]["content"].startswith("wait"): assert self.release.wait(5)
            return "Synthetic final result"
    monkeypatch.setattr(main,"AgentSession",SyntheticSession)
    monkeypatch.setattr(main,"build_llm_client",lambda env: observed(tmp_path,chat_stream())[0].client)
    monkeypatch.setattr(server,"_workspace_change_snapshot",lambda *args:{})
    monkeypatch.setattr(server,"_workspace_change_summary",lambda *args:[])
    with TestClient(server.create_app()) as client: yield client,instances


def poll_complete(client,request_id):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        response=client.get(f"/api/chat/runs/{request_id}")
        data=response.json()
        if data["status"] not in {"running","stopping"}: return data
        time.sleep(.01)
    pytest.fail("Synthetic request did not finish")


def test_http_returns_before_completion_and_refresh_never_replays(api):
    client,sessions=api; rid="request_fixture_123456"
    response=client.post('/api/chat',json={"message":"wait for release","stream_events":True,"request_id":rid,"chat_id":"chat"})
    assert response.status_code==202
    assert sessions[0].started.wait(3)
    for _ in range(3):
        data=client.get(f'/api/chat/runs/{rid}').json()
        assert data["status"]=="running"
        assert data["snapshot"]["blocks"][0]["text"]=="Synthetic output"
    assert client.post('/api/chat',json={"message":"duplicate","request_id":rid,"chat_id":"chat"}).status_code==409
    assert client.post('/api/chat',json={"message":"competing","request_id":"another_request_123456","chat_id":"chat"}).status_code==409
    assert client.post('/api/chat/steer',json={"request_id":rid,"chat_id":"other","message":"No"}).status_code==409
    assert client.post('/api/chat/steer',json={"request_id":rid,"chat_id":"chat","message":"Updated task"}).json()["accepted"]
    assert sessions[0].steering == "Updated task"
    assert client.post('/api/chat/cancel',json={"request_id":rid}).json()=={"cancelled":True}
    result=poll_complete(client,rid)
    assert result["status"]=="cancelled"
    assert result["result"]["usage"]["totals"]["requests"]==1
    assert len(sessions)==1
    assert client.post('/api/chat',json={"message":"same id","request_id":rid,"chat_id":"chat"}).status_code==202
    assert len(sessions)==1


def test_workspace_canonical_validation_and_history_revision(api,tmp_path):
    client,_=api
    assert client.post('/api/workbench/validate-workspace',json={"path":"relative"}).status_code==400
    data=client.post('/api/workbench/validate-workspace',json={"path":str(tmp_path / '..' / tmp_path.name)}).json()
    assert Path(data["path"])==tmp_path.resolve()
    snapshot={"revision":0,"state":{"sessions":[{"id":"s","draft":"preserve"}],"projects":[]}}
    assert client.put('/api/workbench/state',json=snapshot).status_code==200
    assert client.put('/api/workbench/state',json=snapshot).status_code==409
    assert client.get('/api/workbench/state').json()["state"]["sessions"][0]["draft"]=="preserve"


def test_title_calls_are_separate_and_counted(api):
    client,sessions=api
    assert client.post('/api/chat/title',json={"user":"Synthetic title","chat_id":"s"}).status_code==200
    assert sessions[0].constructor_options['no_skill_mode'] is True
    data=client.get('/api/workbench/usage?chat_id=s').json()
    assert data["totals"]["requests"]==1
    assert data["calls"][0]["source"]=="title"


def test_node_workbench_behaviors():
    script=Path(__file__).parent / "static" / "tests" / "client-workbench.test.cjs"
    result=subprocess.run(["node","--test",str(script)],capture_output=True,text=True,encoding="utf-8")
    assert result.returncode==0,result.stdout+result.stderr


def test_server_queue_survives_observer_and_orders_history_once(api):
    client, sessions = api
    first = {'message': 'wait for release', 'server_queue': True, 'request_id': 'queue_first_123456', 'chat_id': 'queued-chat',
             'history': [{'role': 'user', 'content': 'original context'}]}
    second = {**first, 'message': 'second prompt', 'request_id': 'queue_second_123456',
              'history': [{'role': 'user', 'content': 'must not duplicate browser history'}]}
    assert client.post('/api/chat', json=first).status_code == 202
    deadline = time.monotonic() + 3
    while not sessions and time.monotonic() < deadline:
        time.sleep(.01)
    assert sessions[0].started.wait(3)
    assert client.post('/api/chat', json=second).status_code == 202
    assert client.post('/api/chat', json=second).status_code == 202
    assert client.post('/api/chat', json={**second, 'message': 'changed'}).status_code == 409
    assert client.get('/api/chat/runs/queue_second_123456').json()['status'] == 'queued'
    assert len(sessions) == 1
    sessions[0].release.set()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        rows = client.get('/api/chat/queue/queued-chat').json()['items']
        if all(row['status'] == 'completed' for row in rows):
            break
        time.sleep(.01)
    assert [row['status'] for row in rows] == ['completed', 'completed']
    assert len(sessions) == 2
    contents = [message['content'] for message in sessions[1].history]
    assert contents.count('original context') == 1
    assert contents.count('wait for release') == 1
    assert 'must not duplicate browser history' not in contents


def test_queue_cancel_pauses_and_requires_explicit_continue(api):
    client, sessions = api
    first = {'message': 'wait', 'server_queue': True, 'request_id': 'queue_cancel_123456', 'chat_id': 'cancel-chat'}
    assert client.post('/api/chat', json=first).status_code == 202
    deadline = time.monotonic() + 3
    while not sessions and time.monotonic() < deadline:
        time.sleep(.01)
    assert sessions[0].started.wait(3)
    assert client.post('/api/chat', json={**first, 'message': 'after cancel', 'request_id': 'queue_after_123456'}).status_code == 202
    assert client.post('/api/chat/cancel', json={'request_id': first['request_id']}).json()['cancelled']
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        rows = client.get('/api/chat/queue/cancel-chat').json()['items']
        if rows[-1]['status'] == 'paused':
            break
        time.sleep(.01)
    assert rows[-1]['status'] == 'paused'
    assert len(sessions) == 1
    assert client.post('/api/chat/queue/cancel-chat/continue').status_code == 200
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = client.get('/api/chat/runs/queue_after_123456').json()['status']
        if status == 'completed':
            break
        time.sleep(.01)
    assert status == 'completed'


def test_recovery_cannot_be_injected_through_normal_chat(api):
    client, sessions = api
    for queued in [False, True]:
        response = client.post('/api/chat', json={'message': 'spoof', 'request_id': 'recovery_spoof_123456',
                                                 'recovery_from': 'someone-else', 'server_queue': queued})
        assert response.status_code == 400
    assert not sessions


def test_cross_origin_cannot_approve_tools(api):
    client, _ = api
    assert client.post('/api/chat/approval', headers={'Origin': 'https://unrelated.example'}, json={}).status_code == 403


@pytest.mark.parametrize('mode', ['ask', 'risk', 'never', 'read_only'])
def test_permission_mode_is_bound_to_http_session(api, mode):
    client, sessions = api
    response = client.post('/api/chat', json={'message': 'synthetic permission check', 'permission_mode': mode})
    assert response.status_code == 200
    assert sessions[-1].permission_mode == mode


def test_tasks_listing_never_schedules_and_creation_requires_confirmation(api, tmp_path):
    client, sessions = api
    assert client.get('/api/chat/tasks/chat').json() == {'items': []}
    payload = {'chat_id': 'chat', 'workspace_path': str(tmp_path), 'path': 'status.json'}
    assert client.post('/api/chat/tasks', json=payload).status_code == 400
    assert client.post('/api/chat/tasks', json={**payload, 'confirmed': True, 'max_repairs': 1}).status_code == 400
    response = client.post('/api/chat/tasks', json={**payload, 'confirmed': True})
    assert response.status_code == 200
    task = response.json()
    assert task['checks'] == 0 and task['max_repairs'] == 0
    assert client.post('/api/chat/tasks/' + task['id'], json={'chat_id': 'chat', 'action': 'pause'}).status_code == 409
    assert client.post('/api/chat/tasks/' + task['id'], json={'chat_id': 'chat', 'action': 'pause', 'confirmed': True}).status_code == 200
    assert client.get('/api/chat/tasks/chat').json()['items'][0]['status'] == 'paused'
    assert not sessions


def test_compact_command_archives_history_and_changes_next_context_without_model_call(api, tmp_path):
    client, sessions = api
    history = [{'role': 'user', 'content': 'earlier request'}, {'role': 'assistant', 'content': 'old ' * 6000},
               {'role': 'user', 'content': 'current request'}, {'role': 'assistant', 'content': 'current reply'}]
    response = client.post('/api/chat/compact', json={'chat_id': 'compact-chat', 'workspace_path': str(tmp_path), 'history': history})
    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'compacted' and result['after'] < result['before']
    assert Path(result['archive']).is_file()
    assert not sessions
    response = client.post('/api/chat', json={'message': 'next', 'server_queue': True, 'chat_id': 'compact-chat',
                          'request_id': 'compact_next_123456', 'workspace_path': str(tmp_path), 'history': history})
    assert response.status_code == 202
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        data = client.get('/api/chat/runs/compact_next_123456').json()
        if data['status'] == 'completed':
            break
        time.sleep(.01)
    assert data['status'] == 'completed'
    assert len(json.dumps(sessions[0].history)) < len(json.dumps(history))
    assert any(result['sha256'] in message['content'] for message in sessions[0].history)


def test_compact_refuses_pending_work(api, tmp_path):
    import os
    from core.runtime_store import RuntimeStore
    client, sessions = api
    store = RuntimeStore(Path(os.environ['NEURODISCOVERY_WORKBENCH_DB']))
    store.enqueue({'chat_id': 'busy-compact', 'request_id': 'busy_compact_123456', 'message': 'waiting', 'workspace_path': str(tmp_path)})
    response = client.post('/api/chat/compact', json={'chat_id': 'busy-compact', 'workspace_path': str(tmp_path)})
    assert response.status_code == 409
    assert not sessions


def test_reasoning_effort_bound_to_request_not_global_settings(api, monkeypatch):
    client, sessions = api
    session_type = main.AgentSession
    original_init = session_type.__init__
    def initialize(instance, **kwargs):
        original_init(instance, **kwargs)
        instance.env['llm_backend']['model'] = 'gpt-5.5'
    monkeypatch.setattr(session_type, '__init__', initialize)
    response = client.post('/api/chat', json={'message': 'synthetic', 'reasoning_effort': 'high'})
    assert response.status_code == 200
    assert sessions[0].env['llm_backend']['reasoning_effort'] == 'high'
    response = client.post('/api/chat', json={'message': 'synthetic', 'reasoning_effort': 'invalid'})
    assert response.status_code == 400


def test_recovery_endpoint_restores_saved_tool_transaction_without_replaying(api, tmp_path, monkeypatch):
    import os
    from core.runtime_store import RuntimeStore
    client, sessions = api
    store = RuntimeStore(Path(os.environ['NEURODISCOVERY_WORKBENCH_DB']))
    original = {'message': 'original objective', 'server_queue': True, 'request_id': 'recovery_original_123456',
                'chat_id': 'recovery-chat', 'workspace_path': str(tmp_path)}
    store.enqueue(original); store.claim('recovery-chat', 'dead-owner')
    messages = [{'role': 'user', 'content': 'original objective'},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'already-executed', 'type': 'function',
                    'function': {'name': 'read_workspace_file', 'arguments': '{"path":"result.md"}'}}]},
                {'role': 'tool', 'tool_call_id': 'already-executed', 'content': '{"success":true}'}]
    store.checkpoint(original['request_id'], 'recovery-chat', {'workspace': str(tmp_path), 'messages': messages, 'autoresearch': None})
    store.recover_owner('dead-owner')
    response = client.post('/api/chat/recover', json={'request_id': original['request_id'], 'chat_id': 'wrong', 'new_request_id': 'recovery_new_123456'})
    assert response.status_code == 409
    response = client.post('/api/chat/recover', json={'request_id': original['request_id'], 'chat_id': 'recovery-chat', 'new_request_id': 'recovery_new_123456'})
    assert response.status_code == 200
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        data = client.get('/api/chat/runs/recovery_new_123456').json()
        if data['status'] == 'completed':
            break
        time.sleep(.01)
    assert data['status'] == 'completed'
    assert sessions[0].history[1:] == messages
    assert len(sessions) == 1
