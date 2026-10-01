import json

import pytest

from core.agent import main
from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.autoresearch_runtime import AutoResearchRun
from core.permissions import tool_permission
from core.research_reading import read_workspace_page, reading_memory, record_note


def papers(tmp_path, count=8):
    rows = [{'pmid': str(100 + index), 'title': f'论文 {index}',
             'abstract': f'Observed association {index}, not causality. ' + '中文证据。' * 40} for index in range(count)]
    (tmp_path / 'papers.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    return rows


def note(index=0):
    return {'source_path': 'papers.json', 'summary': '关联不代表因果；需控制混杂并验证。',
            'citations': [{'source_id': f'PMID:{100 + index}', 'paper_index': index,
                           'quote': f'Observed association {index}, not causality.'}]}


def test_unicode_character_pages_ranges_reread_and_eof(tmp_path):
    text = '中文分页🙂abc' * 4
    (tmp_path / '中文.txt').write_text(text, encoding='utf-8-sig')
    ledger = {}
    pages = [read_workspace_page(tmp_path, '中文.txt', max_chars=10, ledger=ledger) for _ in range(4)]
    assert ''.join(page['content'] for page in pages) == text
    assert pages[-1]['all_returned']
    assert ledger['中文.txt']['char_ranges'] == [[0, len(text)]]
    assert read_workspace_page(tmp_path, '中文.txt', ledger=ledger)['content'] == ''
    assert read_workspace_page(tmp_path, '中文.txt', max_chars=3, offset=2, ledger=ledger)['content'] == text[2:5]


def test_paper_pages_never_split_records_and_skip_no_unread(tmp_path):
    rows = papers(tmp_path)
    ledger = {}
    first = read_workspace_page(tmp_path, 'papers.json', paper_start=4, paper_count=2, ledger=ledger)
    assert first['paper_end'] == 6 and first['next_paper_start'] == 0
    second = read_workspace_page(tmp_path, 'papers.json', paper_count=4, ledger=ledger)
    assert second['paper_start'] == 0 and second['next_paper_start'] == 6
    third = read_workspace_page(tmp_path, 'papers.json', paper_count=4, ledger=ledger)
    assert json.loads(third['content'])[-1]['paper'] == rows[-1]
    assert third['all_returned'] and ledger['papers.json']['paper_ranges'] == [[0, 8]]
    assert read_workspace_page(tmp_path, 'papers.json', ledger=ledger)['papers'] == []


def test_partial_paper_not_marked_returned_and_version_reset(tmp_path):
    papers(tmp_path)
    ledger = {}
    assert not read_workspace_page(tmp_path, 'papers.json', max_chars=20, ledger=ledger)['success']
    assert ledger == {}
    read_workspace_page(tmp_path, 'papers.json', ledger=ledger)
    papers(tmp_path, 9)
    result = read_workspace_page(tmp_path, 'papers.json', ledger=ledger)
    assert result['version_changed'] and result['paper_start'] == 0


@pytest.mark.parametrize('args', [{'offset': -1}, {'paper_start': -1}, {'paper_count': 0},
                                 {'offset': 1, 'paper_start': 0}, {'max_chars': 90000}, {'offset': 900000}])
def test_invalid_read_inputs(tmp_path, args):
    papers(tmp_path)
    assert not read_workspace_page(tmp_path, 'papers.json', **args)['success']


def test_reader_containment_and_no_cross_file_cursor(tmp_path):
    papers(tmp_path)
    assert not read_workspace_page(tmp_path, '../outside.json')['success']
    ledger = {}
    read_workspace_page(tmp_path, 'papers.json', ledger=ledger)
    (tmp_path / 'other.json').write_bytes((tmp_path / 'papers.json').read_bytes())
    assert read_workspace_page(tmp_path, 'other.json', ledger=ledger)['paper_start'] == 0


def test_notes_validate_returned_source_quote_and_version(tmp_path):
    papers(tmp_path)
    state = {'reading_ledger': {}}
    assert not record_note(tmp_path, state, note())['success']
    read_workspace_page(tmp_path, 'papers.json', paper_count=1, ledger=state['reading_ledger'])
    assert not record_note(tmp_path, state, note(1))['success']
    bad = note()
    bad['citations'][0]['quote'] = 'fabricated evidence'
    assert not record_note(tmp_path, state, bad)['success']
    assert record_note(tmp_path, state, note())['new_note']
    assert not record_note(tmp_path, state, note())['new_note']
    memory = reading_memory(tmp_path, state)
    assert 'PMID:100' in memory and '关联不代表因果' in memory
    papers(tmp_path, 9)
    assert not record_note(tmp_path, state, note())['success']
    assert '"current_version": false' in reading_memory(tmp_path, state)


def test_character_fragment_does_not_support_complete_paper_note(tmp_path):
    papers(tmp_path)
    state = {'reading_ledger': {}}
    read_workspace_page(tmp_path, 'papers.json', offset=0, ledger=state['reading_ledger'])
    assert not record_note(tmp_path, state, note())['success']


def test_resume_keeps_version_ranges_notes_and_original_budget(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_workspace_page(tmp_path, 'papers.json', paper_count=2, ledger=run.state.setdefault('reading_ledger', {}))
    record_note(tmp_path, run.state, note())
    run.state['iterations'] = 3
    run.halt('interrupted', 'Synthetic')
    frozen = run.path.read_bytes()
    resumed = AutoResearchRun(tmp_path, 'idea')
    resumed.inherit_checkpoint(run.path.parent.name, 'continue')
    assert resumed.limit == 37 and len(resumed.state['analysis_notes']) == 1
    page = read_workspace_page(tmp_path, 'papers.json', ledger=resumed.state['reading_ledger'])
    assert page['paper_start'] == 2 and run.path.read_bytes() == frozen


@pytest.mark.parametrize('mode', ['never', 'risk', 'ask', 'read_only'])
def test_notes_only_write_runtime_memory_not_user_files(tmp_path, mode):
    assert tool_permission(mode, 'record_research_note', {'source_path': 'papers.json'}, tmp_path) == 'allow'
    assert tool_permission(mode, 'record_research_note', {'source_path': '../outside'}, tmp_path) == 'deny'


@pytest.mark.parametrize('accepted', [True, False])
def test_full_flow_pages_notes_survive_compaction_and_review(tmp_path, monkeypatch, accepted):
    papers(tmp_path, 3)
    sequence = [reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('record_research_note', **note())]),
                reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=2)]),
                reply(calls=[tool('record_research_note', **note(2))]),
                reply(calls=[tool('write_research_file', path='candidates.json', kind='hypotheses', content=json.dumps([
                    {'hypothesis': 'Synthetic testable association', 'rationale': 'Source-limited draft', 'source_ids': ['PMID:100', 'PMID:102']}]))]),
                reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable', content='# Draft\nPMID:100, PMID:102. Association only; novelty unverified.')]),
                reply(calls=[tool('read_workspace_file', path='IDEA.md')]),
                reply(calls=[tool('finish_autoresearch', **completion(artifacts=['IDEA.md'], evidence_ids=[7]))]),
                reply(json.dumps({'accepted': accepted, 'reason': 'Synthetic independent review'}))]
    agent, calls = session(tmp_path, sequence, mode='idea')
    agent.env['autoresearch_independent_review'] = True
    def compact(messages):
        messages[:] = [messages[0], messages[1]]
    monkeypatch.setattr(agent, '_compact_context', compact)
    agent._chat()
    assert agent.autoresearch_state['status'] == ('completed' if accepted else 'review_required')
    assert agent.autoresearch_state['reading_ledger']['papers.json']['paper_ranges'] == [[0, 3]]
    assert 'PMID:100' in calls[2]['messages'][-1]['content']
    assert 'PMID:102' in calls[4]['messages'][-1]['content']
    assert '关联不代表因果' in calls[4]['messages'][-1]['content']
    assert 'tools' not in calls[-1] and len(calls) == 9


def test_paraphrased_notes_cannot_postpone_synthesis_forever(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    run.start_round()
    page = read_workspace_page(tmp_path, 'papers.json', ledger=run.state.setdefault('reading_ledger', {}))
    run.observe('read_workspace_file', {'path': 'papers.json'}, page)
    run.end_round()
    run.closeout_prompt()
    for index in range(13):
        run.start_round()
        result = record_note(tmp_path, run.state, dict(note(), summary=f'Observation {index}, not validated.'))
        run.observe('record_research_note', {}, result)
        stalled = run.end_round()
    assert stalled and run.state['synthesis_idle_rounds'] == 12
    assert run.state['note_relief_rounds'] == 1
    assert 'hypotheses' not in run.state['research_progress']


def read_round(run, **kwargs):
    run.start_round()
    result = read_workspace_page(run.workspace, 'papers.json', ledger=run.state.setdefault('reading_ledger', {}), **kwargs)
    run.observe('read_workspace_file', {'path': 'papers.json'}, result)
    return result, run.end_round()


def note_round(run, index=0):
    run.start_round()
    result = record_note(run.workspace, run.state, note(index))
    run.observe('record_research_note', {}, result)
    return result, run.end_round()


def test_new_pages_reset_activity_not_synthesis_and_reread_eof_do_not(tmp_path):
    papers(tmp_path, 4)
    run = AutoResearchRun(tmp_path, 'idea')
    page, _ = read_round(run, paper_count=2)
    assert page['new_papers_returned'] == 2
    page, _ = read_round(run, paper_start=1, paper_count=2)
    assert page['new_papers_returned'] == 1
    assert run.state['no_progress_count'] == 0
    assert run.state['synthesis_idle_rounds'] == 2
    page, _ = read_round(run, paper_start=1, paper_count=2)
    assert page['new_papers_returned'] == 0
    assert run.state['no_progress_count'] == 1
    page, _ = read_round(run)
    assert page['new_papers_returned'] == 1 and page['all_returned']
    assert run.state['no_progress_count'] == 0
    page, _ = read_round(run)
    assert page['new_papers_returned'] == 0 and page['all_returned']
    assert run.state['no_progress_count'] == 1


def test_character_pages_and_instructions_are_distinguished(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run, offset=0, max_chars=20)
    page, _ = read_round(run, offset=10, max_chars=20)
    assert page['new_chars_returned'] == 10
    assert run.state['no_progress_count'] == 0
    read_round(run, offset=10, max_chars=20)
    assert run.state['no_progress_count'] == 1
    (tmp_path / 'AGENTS.md').write_text('Instructions only')
    run.start_round()
    page = read_workspace_page(tmp_path, 'AGENTS.md', ledger=run.state['reading_ledger'])
    run.observe('read_workspace_file', {'path': 'AGENTS.md'}, page)
    run.end_round()
    assert run.state['no_progress_count'] == 2


def test_note_relief_is_bounded_and_preserved_across_resume(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run, paper_count=8)
    for index in range(4):
        run.state['synthesis_idle_rounds'] = 7
        result, stalled = note_round(run, index)
        assert result['success'] and not stalled
        assert run.state['synthesis_idle_rounds'] == 0
    assert run.state['note_relief_rounds'] == 4
    run.state['iterations'] = 5
    run.halt('interrupted', 'Synthetic pause')
    resumed = AutoResearchRun(tmp_path, 'idea')
    resumed.inherit_checkpoint(run.path.parent.name, 'continue')
    assert resumed.limit == 35
    assert resumed.state['note_relief_rounds'] == 4
    assert resumed.state['note_relief_sources'] == run.state['note_relief_sources']
    resumed.state.update(synthesis_idle_rounds=11, correction_sent=True)
    result, stalled = note_round(resumed, 4)
    assert result['success'] and not stalled
    assert resumed.state['synthesis_idle_rounds'] == 12
    for _ in range(2):
        resumed.start_round()
        stalled = resumed.end_round()
    assert stalled


def test_duplicate_and_invalid_notes_do_not_grant_relief(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run, paper_count=1)
    note_round(run)
    result, _ = note_round(run)
    assert not result['new_note']
    assert run.state['synthesis_idle_rounds'] == 1
    result, _ = note_round(run, 1)
    assert not result['success']
    assert run.state['note_relief_rounds'] == 1
    assert run.state['synthesis_idle_rounds'] == 2


def test_note_batch_consumes_one_relief_round(tmp_path):
    papers(tmp_path)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run)
    run.start_round()
    for index in range(3):
        result = record_note(tmp_path, run.state, note(index))
        run.observe('record_research_note', {}, result)
    run.end_round()
    assert run.state['note_relief_rounds'] == 1
    assert len(run.state['note_relief_sources']) == 3


@pytest.mark.parametrize('language', ['en', 'zh'])
def test_stall_report_names_guard_and_returned_coverage(tmp_path, language):
    papers(tmp_path, 25)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run, paper_count=20, max_chars=40000)
    read_round(run, paper_count=3)
    run.state.update(language=language, synthesis_idle_rounds=12, no_progress_count=0)
    report = run.halt('stalled', 'Synthetic synthesis deadline')
    assert run.state['stall_diagnostics']['triggers'] == ['no_synthesis']
    assert '23/25' in report and 'no_synthesis=12' in report
    assert 'notes=0' in report and 'note_relief=0/4' in report
    assert not run.state['closeout']['accepted']


def test_actual_compaction_retains_cited_analysis_on_next_dispatch(tmp_path):
    papers(tmp_path, 3)
    sequence = [reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('record_research_note', **note())]),
                reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('read_workspace_file', path='papers.json', paper_count=1)]),
                reply(calls=[tool('write_research_file', path='IDEA.md', kind='deliverable', content='Evidence gap, PMID:100. Not accepted.')]),
                reply(calls=[tool('finish_autoresearch', **completion(artifacts=['IDEA.md'], evidence_ids=[5]))])]
    agent, calls = session(tmp_path, sequence, mode='idea')
    agent.env['context_input_budget'] = 2048
    agent.history[0]['content'] += 'Synthetic context pressure. ' * 4000
    agent._chat()
    assert list((tmp_path / '.neurodiscovery/context').glob('*.json'))
    last_memory = calls[4]['messages'][-1]['content']
    assert 'PMID:100' in last_memory and 'Observed association 0' in last_memory
    assert '关联不代表因果' in last_memory and '"next_paper_start": 3' in last_memory


def test_stateless_explicit_character_cursor_and_json_wrapper(tmp_path):
    rows = papers(tmp_path)
    (tmp_path / 'wrapped.json').write_text(json.dumps({'articles': rows}), encoding='utf-8')
    page = read_workspace_page(tmp_path, 'wrapped.json', paper_start=2, paper_count=2)
    assert page['next_paper_start'] == 4
    text = read_workspace_page(tmp_path, 'wrapped.json', offset=10, max_chars=20)
    assert text['next_offset'] == 30


def test_same_identifier_wrong_quote_and_invalid_note_never_persist(tmp_path):
    papers(tmp_path)
    state = {'reading_ledger': {}}
    read_workspace_page(tmp_path, 'papers.json', ledger=state['reading_ledger'])
    wrong = note()
    wrong['citations'][0]['source_id'] = 'PMID:101'
    assert not record_note(tmp_path, state, wrong)['success']
    assert not record_note(tmp_path, state, dict(note(), source_path='../outside.json'))['success']
    assert not state.get('analysis_notes')


def test_long_note_with_many_citations_is_accepted(tmp_path):
    papers(tmp_path, 10)
    state = {'reading_ledger': {}}
    read_workspace_page(tmp_path, 'papers.json', paper_count=10, ledger=state['reading_ledger'])
    long_note = {
        'source_path': 'papers.json',
        'summary': '分析摘要。' * 400,
        'citations': [{'source_id': f'PMID:{100 + index}', 'paper_index': index,
                       'quote': f'Observed association {index}, not causality.'} for index in range(8)],
    }
    assert len(long_note['summary']) > 1500 and len(long_note['citations']) > 5
    assert record_note(tmp_path, state, long_note)['success']


def test_large_paper_page_is_accepted(tmp_path):
    papers(tmp_path, 25)
    page = read_workspace_page(tmp_path, 'papers.json', paper_count=25, max_chars=40000, ledger={})
    assert page['success'] and page['paper_end'] == 25


def test_quote_ignores_unicode_whitespace_style(tmp_path):
    (tmp_path / 'papers.json').write_text(json.dumps([
        {'pmid': '500', 'title': 'Thin space',
         'abstract': 'The result held (p\u2009>\u20090.05) across cohorts.'}]), encoding='utf-8')
    state = {'reading_ledger': {}}
    read_workspace_page(tmp_path, 'papers.json', ledger=state['reading_ledger'])
    tolerant = {'source_path': 'papers.json', 'summary': 'Whitespace style differs; content verbatim.',
                'citations': [{'source_id': 'PMID:500', 'paper_index': 0,
                               'quote': 'held (p > 0.05) across cohorts'}]}
    assert record_note(tmp_path, state, tolerant)['success']
    fabricated = dict(tolerant, citations=[{'source_id': 'PMID:500', 'paper_index': 0,
                                            'quote': 'held (p > 0.01) across cohorts'}])
    assert not record_note(tmp_path, state, fabricated)['success']


def test_empty_eof_read_is_labeled_not_an_error(tmp_path):
    # A plain reread at EOF returns empty content; it must say so explicitly
    # instead of looking like a failed read that a model would retry.
    (tmp_path / 'candidates.json').write_text(json.dumps([{'candidate': 'x'}]), encoding='utf-8')
    ledger = {}
    first = read_workspace_page(tmp_path, 'candidates.json', ledger=ledger)
    assert first['content'] and first['all_returned'] and not first.get('empty_content')
    eof = read_workspace_page(tmp_path, 'candidates.json', ledger=ledger)
    assert eof['success'] and eof['content'] == ''
    assert eof['empty_content'] is True and eof['eof'] is True
    assert eof['empty_reason'] == 'already_returned' and 'not an error' in eof['notice']
    # An intentional explicit reread still returns the real content.
    reread = read_workspace_page(tmp_path, 'candidates.json', offset=0, ledger=ledger)
    assert reread['content'] and not reread.get('empty_content')


def test_repeat_plain_read_is_noticed_then_refused(tmp_path):
    papers(tmp_path, 4)
    run = AutoResearchRun(tmp_path, 'idea')
    read_round(run)
    assert not run.repeat_read_rejection('read_workspace_file', {'path': 'papers.json'})
    for _ in range(2):
        result, _ = read_round(run)
        assert result['success'] and 'repeat_read_notice' in result
    read_round(run)
    rejection = run.repeat_read_rejection('read_workspace_file', {'path': 'papers.json'})
    assert rejection and rejection['error_type'] == 'repeat_read' and rejection['executed'] is False
    # Explicit offset/paper_start rereads and not-yet-covered files are never blocked.
    assert run.repeat_read_rejection('read_workspace_file', {'path': 'papers.json', 'offset': 0}) is None
    assert run.repeat_read_rejection('read_workspace_file', {'path': 'unread.json'}) is None
