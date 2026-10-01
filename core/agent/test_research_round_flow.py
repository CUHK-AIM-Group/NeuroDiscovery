import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from core.agent import main
from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.autoresearch_runtime import AutoResearchRun
from core.research_reading import read_workspace_page
from core import pubmed_recovery
from core.checkpoint.manager import ShadowCheckpointManager
from core.shell_safety import is_readonly_probe


@pytest.mark.parametrize('command', ['dir /b', 'cd /d C:\\work && dir /b && echo SKILLS && dir /b skills 2>nul | findstr /i "idea paper"', 'python -V', 'pwd'])
def test_readonly_probes_skip_checkpoint(command):
    assert is_readonly_probe(command)


@pytest.mark.parametrize('command', ['dir > report.txt', 'python mutate.py', 'echo hi & del file', 'dir %COMMAND%', 'cat a; touch b'])
def test_unknown_commands_still_checkpoint(command):
    assert not is_readonly_probe(command)


def test_batch_finishes_and_counts_once_even_at_threshold(tmp_path):
    run = AutoResearchRun(tmp_path,'idea')
    run.state.update(no_progress_count=11, correction_sent=True)
    run.start_round()
    for index in range(20):
        assert not run.observe('inspect_local_path', {}, {'success':True})
    assert run.state['no_progress_count'] == 11
    (tmp_path/'IDEA.md').write_text('Synthetic evidence and candidate limitations')
    assert not run.end_round()
    assert run.state['no_progress_count'] == 0
    assert len(run.state['evidence']) == 20


def test_preparation_then_correction_before_stall(tmp_path, monkeypatch):
    executed = []
    monkeypatch.setattr(main, '_run_shell_command', lambda **kwargs: executed.append(kwargs) or {'success':True})
    agent, calls = session(tmp_path, [reply(calls=[tool('run_shell_command',command='dir')])] * 12)
    assert 'stalled' in agent._chat()
    assert len(executed) == 2
    assert any('corrective action' in entry.get('content','') for entry in calls[2]['messages'])
    assert len(calls) == 12
    assert agent.autoresearch_state['completed_rounds'] == 12


def test_recovery_grace_is_bounded_and_persisted(tmp_path):
    run = AutoResearchRun(tmp_path,'idea')
    for index in range(3):
        run.start_round()
        run.observe('run_shell_command', {}, {'success':False,'error_type':'missing_dependency'})
        run.end_round()
    assert run.state['recovery_rounds'] == 2
    assert run.state['no_progress_count'] == 1
    run.halt('interrupted','synthetic pause')
    restored = AutoResearchRun(tmp_path,'idea')
    restored.inherit_checkpoint(run.path.parent.name,'continue')
    assert restored.state['recovery_rounds'] == 2
    assert restored.state['no_progress_count'] == 1
    assert restored.state['completed_rounds'] == 3


def test_valid_source_read_counts_without_registration(tmp_path):
    (tmp_path/'papers.json').write_text(json.dumps([{'pmid':'123','title':'Synthetic source'}]))
    run = AutoResearchRun(tmp_path,'idea')
    run.start_round()
    run.observe('read_workspace_file', {'path':'papers.json'}, {'success':True})
    run.end_round()
    assert run.state['no_progress_count'] == 0
    run.start_round()
    run.observe('read_workspace_file', {'path':'papers.json'}, {'success':True})
    run.end_round()
    assert run.state['no_progress_count'] == 1


def test_existing_delivery_not_credited_until_read_and_receipts_not_scanned(tmp_path):
    (tmp_path/'IDEA.md').write_text('Historical material')
    run = AutoResearchRun(tmp_path,'idea')
    run.start_round()
    run.observe('read_workspace_file', {'path':str(run.path)}, {'success':True})
    run.end_round()
    assert not run.state['research_progress']
    assert run.state['no_progress_count'] == 1
    run.start_round()
    run.observe('read_workspace_file', {'path':'IDEA.md'}, {'success':True})
    run.end_round()
    assert run.state['no_progress_count'] == 0


def test_actual_agent_batch_is_not_cut_off_mid_round(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','1')
    sequence = [tool('read_workspace_file',path='missing.json') for index in range(15)]
    (tmp_path/'papers.json').write_text(json.dumps([{'pmid':'123','title':'Evidence'}]))
    sequence.append(tool('read_workspace_file',path='papers.json'))
    agent, calls = session(tmp_path,[reply(calls=sequence)])
    agent._chat()
    assert len(agent.autoresearch_state['evidence']) == 16
    assert agent.autoresearch_state['no_progress_count'] == 0
    assert agent.autoresearch_state['completed_rounds'] == 1
    assert agent.autoresearch_state['status'] == 'budget_exhausted'


def test_readonly_agent_never_calls_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','1')
    monkeypatch.setattr(main,'_run_shell_command',lambda **kwargs: {'success':True})
    agent, calls = session(tmp_path,[reply(calls=[tool('run_shell_command',command='dir /b')])])
    agent._checkpoint_mgr = SimpleNamespace(begin_turn=lambda:None,
        checkpoint=lambda *args,**kwargs:pytest.fail('Readonly command must not checkpoint'))
    agent._chat()
    assert agent.autoresearch_state['evidence'][0]['success']


def test_crash_recovery_preserves_round_counters_and_material(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','4')
    (tmp_path/'papers.json').write_text(json.dumps([{'pmid':'123','title':'Evidence'}]))
    run = AutoResearchRun(tmp_path,'idea')
    page = read_workspace_page(tmp_path, 'papers.json', ledger=run.state.setdefault('reading_ledger', {}))
    run.observe('read_workspace_file', {'path':'papers.json'}, page)
    run.state.update(iterations=3, no_progress_count=7, completed_rounds=9, correction_sent=True, recovery_rounds=2)
    agent, calls = session(tmp_path,[reply(calls=[tool('read_workspace_file',path='papers.json')])])
    agent._recovery_state = run.state
    agent._chat()
    assert agent.autoresearch_state['completed_rounds'] == 10
    assert agent.autoresearch_state['no_progress_count'] == 8
    assert agent.autoresearch_state['recovery_rounds'] == 2
    assert len(calls) == 1


@pytest.mark.skipif(os.name != 'nt', reason='Windows cmd transport regression')
def test_actual_multiline_python_and_reject_partial_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(main,'AGENT_SHELL_STATUS_FILE',tmp_path/'status.json')
    command = f'cd /d "{tmp_path}" && "{sys.executable}" -c "import json\nfor value in [1,2]:\n    print(value)\n"'
    result = main._run_shell_command(command,tmp_path,10)
    assert result['success'] and result['stdout'].split() == ['1','2']
    assert not main._run_shell_command('echo first\necho second',tmp_path,10)['success']
    assert main._run_shell_command('echo first\necho second',tmp_path,10)['executed'] is False


def test_checkpoint_timeout_and_cancellation(tmp_path, monkeypatch):
    from core.checkpoint import manager as checkpoint_module
    real_popen = checkpoint_module.subprocess.Popen
    commands = []
    def spawn(command, **kwargs):
        if command[0] == 'git':
            commands.append(command)
            return real_popen([sys.executable,'-c','import time; time.sleep(30)'], **kwargs)
        return real_popen(command, **kwargs)
    monkeypatch.setattr(checkpoint_module.subprocess,'Popen',spawn)
    manager = ShadowCheckpointManager(tmp_path)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        manager.checkpoint(tmp_path,timeout=0.1)
    assert time.monotonic()-started < 6
    assert 'gc.auto=0' in commands[0] and 'maintenance.auto=false' in commands[0]
    cancel = threading.Event(); cancel.set()
    with pytest.raises(TimeoutError):
        manager.checkpoint(tmp_path,cancel_event=cancel)
    assert len(commands) == 1


def test_checkpoint_timeout_does_not_block_mutating_tool(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','1')
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == 15
        raise TimeoutError('synthetic timeout')
    executed = []
    monkeypatch.setattr(main,'_run_shell_command',lambda **kwargs: executed.append(kwargs) or {'success':True})
    agent, calls = session(tmp_path,[reply(calls=[tool('run_shell_command',command='python mutate.py')])])
    agent._checkpoint_mgr = SimpleNamespace(checkpoint=timeout, begin_turn=lambda: None)
    events = []
    agent._emit_execution_event = events.append
    agent._chat()
    assert len(executed) == 1
    assert any(event['type'] == 'checkpoint' for event in events)


@pytest.mark.parametrize('accepted',[True,False])
def test_offline_full_flow_recovery_sources_hypotheses_delivery_review(tmp_path, monkeypatch, accepted):
    monkeypatch.setattr(main,'AGENT_SHELL_STATUS_FILE',tmp_path/'shell.json')
    monkeypatch.setattr(pubmed_recovery,'_process',lambda *args: (1,'','synthetic dependency failure'))
    monkeypatch.setattr(pubmed_recovery,'pubmed_http',lambda *args: [{'pmid':'123','title':'Synthetic source','abstract':'Synthetic observation'}])
    candidate = json.dumps([{'hypothesis':'Synthetic testable claim','rationale':'Source-linked rationale','source_ids':['PMID:123']}])
    command = "python -c \"from pathlib import Path; Path('candidates.json').write_bytes(bytes.fromhex('" + candidate.encode().hex() + "')); Path('IDEA.md').write_text('Synthetic candidate PMID:123; limitations; falsifiable prediction.')\""
    responses = [reply(calls=[tool('run_shell_command',command='dir' if os.name=='nt' else 'pwd')]),
                 reply(calls=[tool('search_pubmed',query='synthetic',output='papers.json')]),
                 reply(calls=[tool('read_workspace_file',path='papers.json')]),
                 reply(calls=[tool('run_shell_command',command=command),tool('read_workspace_file',path='candidates.json')]),
                 reply(calls=[tool('read_workspace_file',path='IDEA.md')]),
                 reply(calls=[tool('finish_autoresearch',**completion(artifacts=['IDEA.md'],evidence_ids=[2,3,4,5,6]))]),
                 reply(json.dumps({'accepted':accepted,'reason':'Synthetic independent verdict'}))]
    agent, calls = session(tmp_path,responses,mode='idea')
    agent.env['autoresearch_independent_review'] = True
    agent._chat()
    assert agent.autoresearch_state['status'] == ('completed' if accepted else 'review_required')
    assert set(agent.autoresearch_state['research_progress']) == {'literature','hypotheses','deliverable'}
    assert agent.autoresearch_state['dependency_recovery']['install_attempted']
    assert len(calls) == 7
    assert agent.autoresearch_state['iterations'] == 7
    assert all(event['tool'] != 'inspect_research_progress' for event in agent.autoresearch_state['evidence'])
    assert 'tools' not in calls[-1]
