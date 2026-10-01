import json

import pytest

from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.agent.test_research_reading import papers, note, read_round, note_round
from core.autoresearch_runtime import AutoResearchRun


def tool_names(call):
    return {item['function']['name'] for item in call['tools']}


def test_batch_gate_unlocks_only_after_valid_current_note(tmp_path):
    papers(tmp_path, 8)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run, paper_count=1)
    assert not run.synthesis_control()
    read_round(run, paper_count=1)
    assert run.synthesis_control() == 'note'
    for name in ('read_workspace_file', 'search_pubmed', 'run_shell_command', 'spawn_subagent'):
        assert run.control_rejection(name, {})['executed'] is False
    assert run.control_rejection('write_research_file', {'kind': 'literature'})
    assert run.control_rejection('write_research_file', {'kind': 'deliverable'}) is None
    result, _ = note_round(run, 7)
    assert not result['success'] and run.synthesis_control() == 'note'
    result, _ = note_round(run, 1)
    assert result['success'] and not run.synthesis_control()
    assert run.state['reading_batch_calls'] == 0
    assert not run.state['research_progress'].get('hypotheses')


def test_batch_gate_persists_resume_and_context_without_changing_budget(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run, paper_count=1)
    read_round(run, paper_count=1)
    run.start_round()
    run.end_round()
    run.state['iterations'] = 3
    run.halt('interrupted', 'synthetic')
    resumed = AutoResearchRun(tmp_path, 'idea')
    resumed.inherit_checkpoint(run.path.parent.name, 'continue')
    assert resumed.limit == 37
    assert resumed.synthesis_control() == 'note'
    assert resumed.state['controlled_rounds'] == 1
    assert 'Observed association 1' in resumed.synthesis_prompt()
    assert 'PMID:101' in resumed.synthesis_prompt()


def test_old_stalled_run_gets_only_three_consolidation_rounds(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run)
    run.state.update(iterations=13, synthesis_idle_rounds=12, correction_sent=True)
    run.halt('stalled', 'synthetic')
    resumed = AutoResearchRun(tmp_path, 'idea')
    resumed.inherit_checkpoint(run.path.parent.name, 'continue')
    assert resumed.limit == 27
    for index in range(3):
        resumed.begin_iteration()
        resumed.start_round()
        assert resumed.synthesis_control() == 'consolidate'
        assert resumed.control_rejection('record_research_note', {})
        assert resumed.end_round() is (index == 2)
    resumed.halt('stalled', 'synthetic')
    assert 'synthesis_control_exhausted' in resumed.state['stall_diagnostics']['triggers']
    again = AutoResearchRun(tmp_path, 'idea')
    again.inherit_checkpoint(resumed.path.parent.name, 'continue')
    assert again.limit == 24 and again.state['controlled_rounds'] == 3


@pytest.mark.parametrize('accepted', [True, False])
def test_real_dispatch_batch_gate_then_note_delivery_and_review(tmp_path, accepted):
    papers(tmp_path, 5)
    sequence = [reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('record_research_note', **note(1))]),
                reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable', content='Evidence gap PMID:101. Synthetic incomplete novelty evidence.')]),
                reply(calls=[tool('read_workspace_file', path='IDEA.md')]),
                reply(calls=[tool('finish_autoresearch', **completion(artifacts=['IDEA.md'], evidence_ids=[6, 7]))]),
                reply(json.dumps({'accepted': accepted, 'reason': 'Synthetic review'}))]
    agent, calls = session(tmp_path, sequence, mode='idea')
    agent.env['autoresearch_independent_review'] = True
    agent._chat()
    assert 'read_workspace_file' not in tool_names(calls[2])
    assert 'record_research_note' in tool_names(calls[2])
    assert not agent.autoresearch_state['evidence'][2]['executed']
    assert 'read_workspace_file' in tool_names(calls[4])
    assert agent.autoresearch_state['reading_ledger']['papers.json']['paper_ranges'] == [[0, 3]]
    assert agent.autoresearch_state['status'] == ('completed' if accepted else 'review_required')
    assert 'tools' not in calls[-1]


def test_denied_controlled_writes_stop_without_artifact_or_budget_increase(tmp_path, monkeypatch):
    papers(tmp_path)
    previous = AutoResearchRun(tmp_path, 'idea')
    read_round(previous)
    previous.state.update(iterations=10, synthesis_idle_rounds=8, correction_sent=True)
    previous.halt('interrupted', 'synthetic')
    sequence = [reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable', content='Synthetic')])] * 3
    agent, calls = session(tmp_path, sequence, mode='idea')
    agent._recovery_state = previous.state
    monkeypatch.setattr(agent, '_tool_authorized', lambda *args: False)
    agent._chat()
    assert agent.autoresearch_state['status'] == 'stalled'
    assert agent.autoresearch_state['iterations'] == 13
    assert agent.autoresearch_state['iteration_limit'] == 40
    assert len(calls) == 3 and not (tmp_path / 'IDEA.md').exists()


def test_data_mode_keeps_tools(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'data')
    read_round(run, paper_count=1)
    read_round(run, paper_count=1)
    assert not run.synthesis_control()
    assert run.control_rejection('search_pubmed', {}) is None


def test_same_response_cannot_read_past_batch_gate(tmp_path, monkeypatch):
    papers(tmp_path, 8)
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS', '10')
    sequence = [reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1) for _ in range(4)]),
                reply(calls=[tool('record_research_note', **note(1))]),
                reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable', content='Synthetic evidence gap PMID:101.')]),
                reply(calls=[tool('finish_autoresearch', **completion(artifacts=['IDEA.md'], evidence_ids=[6]))])]
    agent, calls = session(tmp_path, sequence, mode='idea')
    agent._chat()
    assert agent.autoresearch_state['reading_ledger']['papers.json']['paper_ranges'] == [[0, 2]]
    assert [item['executed'] for item in agent.autoresearch_state['evidence'][:4]] == [True, True, False, False]
    assert agent.autoresearch_state['iterations'] == 4


def test_consolidation_can_deliver_then_validate_with_original_budget(tmp_path):
    papers(tmp_path)
    previous = AutoResearchRun(tmp_path, 'idea')
    read_round(previous)
    previous.state.update(iterations=13, synthesis_idle_rounds=12, correction_sent=True)
    previous.halt('stalled', 'synthetic')
    sequence = [reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable', content='Synthetic evidence-gap report PMID:100; novelty unverified.')]),
                reply(calls=[tool('read_workspace_file', path='IDEA.md')]),
                reply(calls=[tool('finish_autoresearch', **completion(artifacts=['IDEA.md'], evidence_ids=[1, 2]))]),
                reply(json.dumps({'accepted': True, 'reason': 'Synthetic review'}))]
    agent, calls = session(tmp_path, sequence, mode='idea')
    agent._recovery_state = previous.state
    agent.env['autoresearch_independent_review'] = True
    agent._chat()
    assert 'read_workspace_file' not in tool_names(calls[0])
    assert 'record_research_note' not in tool_names(calls[0])
    assert 'read_workspace_file' in tool_names(calls[1])
    assert agent.autoresearch_state['iterations'] == 17
    assert agent.autoresearch_state['status'] == 'completed'
    assert agent.autoresearch_state['iteration_limit'] == 40


def test_repeated_eof_read_is_refused_and_run_can_still_finish(tmp_path, monkeypatch):
    # Signature of the flash loop: after delivering, the model keeps rereading an
    # already-covered file at EOF instead of calling finish_autoresearch. The
    # runtime must notice, expose a repeat_read refusal to the model, and still
    # allow a real finish inside the same budget.
    papers(tmp_path, 4)
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS', '10')
    sequence = [reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=4)]),
                reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable',
                                  content='Evidence-gap report: PMID:100 read; novelty unverified.')])]
    sequence += [reply(calls=[tool('read_workspace_file', path='IDEA.md')])] * 4
    sequence += [reply(calls=[tool('finish_autoresearch', **completion(artifacts=['IDEA.md'], evidence_ids=[1]))])]
    agent, calls = session(tmp_path, sequence, mode='idea')
    result = agent._chat()
    assert agent.autoresearch_state['status'] == 'completed'
    executed = [item['executed'] for item in agent.autoresearch_state['evidence'] if item['tool'] == 'read_workspace_file']
    # papers read + first (content-bearing) deliverable read, then two empty
    # rereads are tolerated with a notice before the next plain reread is refused.
    assert executed == [True, True, True, True, False]
    assert agent.autoresearch_state['repeat_read_paths'] == {'IDEA.md': 2}
    assert any('repeat_read' in json.dumps(call['messages']) for call in calls)
