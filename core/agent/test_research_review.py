import json

import pytest

from core.agent.test_autoresearch_execution import reply, session
from core.autoresearch_runtime import AutoResearchRun, FINISH_TOOL
from core.research_review import MAX_REVIEW_CONTENT_BYTES, build_review_packet


def review_setup(tmp_path, responses=()):
    path = tmp_path / 'report.md'
    path.write_text('Synthetic scoped report', encoding='utf-8')
    run = AutoResearchRun(tmp_path, 'idea')
    run.state['artifacts'] = [str(path)]
    agent, calls = session(tmp_path, list(responses), mode='idea')
    return run, agent, calls, path


@pytest.mark.parametrize('failure', ['too_large', 'too_many', 'invalid_utf8', 'metadata_too_large', 'empty'])
def test_invalid_packet_fails_before_provider_call_and_budget_charge(tmp_path, failure):
    run, agent, calls, path = review_setup(tmp_path)
    if failure == 'too_large':
        path.write_bytes(b'x' * (MAX_REVIEW_CONTENT_BYTES + 1))
    elif failure == 'too_many':
        run.state['artifacts'] *= 13
    elif failure == 'invalid_utf8':
        path.write_bytes(b'bad\xff')
    elif failure == 'metadata_too_large':
        run.state['summary'] = 'x' * (1024 * 1024)
    else:
        path.write_bytes(b'')
    verdict = agent._review_autoresearch(run, 'offline-test')
    assert not verdict['accepted']
    assert verdict['error_type'] == 'review_packet_invalid'
    assert calls == [] and run.state['iterations'] == 0


def test_combined_limit_not_per_file_and_boundary_is_complete(tmp_path):
    run, agent, calls, path = review_setup(tmp_path, [reply('{"accepted": true, "reason": "Synthetic pass"}')])
    path.write_bytes(b'x' * (MAX_REVIEW_CONTENT_BYTES // 2))
    other = tmp_path / 'sources.json'
    other.write_bytes(b'y' * (MAX_REVIEW_CONTENT_BYTES // 2))
    run.state['artifacts'].append(str(other))
    serialized, bindings = build_review_packet(run)
    assert sum(item['bytes'] for item in bindings) == MAX_REVIEW_CONTENT_BYTES
    assert all(not item['partial'] for item in json.loads(serialized)['artifacts'])
    other.write_bytes(other.read_bytes() + b'z')
    assert not agent._review_autoresearch(run, 'offline-test')['accepted']
    assert not calls


def test_review_roles_notes_and_coverage_are_explicit(tmp_path):
    run, agent, calls, path = review_setup(tmp_path, [reply('{"accepted": false, "reason": "Unsupported inference"}')])
    run.state['reading_ledger'] = {'sources.json': {'sha256': 'synthetic', 'paper_ranges': [[0, 2]], 'total_papers': 50}}
    run.state['analysis_notes'] = [{'summary': 'Synthetic unvalidated note', 'citations': []}]
    run.state['validation'] = 'Delegation denied; no prior review'
    verdict = agent._review_autoresearch(run, 'offline-test')
    assert not verdict['accepted'] and verdict['reason'] == 'Unsupported inference'
    assert 'tools' not in calls[0]
    system = calls[0]['messages'][0]['content']
    assert 'This call IS' in system and 'Do not require an earlier acceptance' in system
    assert 'never waive such requirements' in system
    packet = json.loads(calls[0]['messages'][1]['content'])
    assert packet['reading_ledger'] == run.state['reading_ledger']
    assert packet['analysis_notes'] == run.state['analysis_notes']
    assert packet['review_contract']['scientific_validation'] is False
    assert 'do not spawn a reviewer' in FINISH_TOOL['function']['description']
    assert 'Do not call spawn_subagent for acceptance' in run.prompt()


@pytest.mark.parametrize('content', ['not JSON', '{"accepted":"true","reason":"bad"}', '{"accepted":true,"reason":""}'])
def test_invalid_verdict_cannot_accept_complete_packet(tmp_path, content):
    run, agent, calls, path = review_setup(tmp_path, [reply(content)])
    assert not agent._review_autoresearch(run, 'offline-test')['accepted']
    assert run.state['iterations'] == 1 and len(calls) == 1


def test_artifact_changed_during_review_is_rejected(tmp_path):
    run, agent, calls, path = review_setup(tmp_path)
    def mutate():
        path.write_text('Changed after packet', encoding='utf-8')
        return reply('{"accepted":true,"reason":"Synthetic pass"}')
    agent, calls = session(tmp_path, [mutate])
    assert not agent._review_autoresearch(run, 'offline-test')['accepted']
    assert len(calls) == 1


def test_cancel_and_budget_exhaustion_never_dispatch_review(tmp_path):
    run, agent, calls, path = review_setup(tmp_path)
    agent.request_cancel()
    assert not agent._review_autoresearch(run, 'offline-test')['accepted']
    agent._cancel_event.clear()
    run.state['iterations'] = run.limit
    assert not agent._review_autoresearch(run, 'offline-test')['accepted']
    assert not calls
