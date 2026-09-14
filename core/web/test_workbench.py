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
