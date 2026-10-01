import json
import threading

import pytest

from core.agent import main
from core.agent.test_autoresearch_execution import completion, reply, session, tool
from core.autoresearch_runtime import AutoResearchRun
from core.permissions import tool_permission
from core.research_writer import write_research_file
from core.shell_safety import is_python_listing_probe


PAPERS = [{'pmid': '123', 'title': 'Synthetic source', 'abstract': 'Synthetic observation, not real science.'}]
CANDIDATES = [{'hypothesis': 'Synthetic falsifiable hypothesis', 'rationale': 'Source 123 suggests an association, not causality.',
               'source_ids': ['PMID:123'], 'prediction': 'Predefined held-out prediction', 'limitations': 'Unverified novelty.'}]


def literature(run, filename='papers.json'):
    (run.workspace / filename).write_text(json.dumps(PAPERS))
    run.start_round()
    run.observe('search_pubmed', {}, {'success': True, 'path': filename, 'count': 1})
    run.end_round()


def test_literature_does_not_reenable_preparation(tmp_path):
    run = AutoResearchRun(tmp_path, 'idea')
    literature(run)
    run.start_round(); run.end_round()
    assert run.preparation_exhausted()
    assert run.idea_stage() == 'analysis'
    prompt = run.closeout_prompt()
    assert 'WRITE' in prompt and 'papers.json' in prompt and 'source_ids' in prompt
    assert 'Do not read IDEA.md before creating it' in prompt


def test_more_literature_cannot_postpone_synthesis_forever(tmp_path):
    run = AutoResearchRun(tmp_path, 'idea')
    for index in range(12):
        run.start_round()
        path = f'papers-{index}.json'
        (tmp_path/path).write_text(json.dumps([{'pmid':str(index+1),'title':'Synthetic'}]))
        run.observe('search_pubmed', {}, {'success':True,'path':path,'count':1})
        stalled = run.end_round()
        run.closeout_prompt()
    assert stalled
    assert run.state['no_progress_count'] == 0
    assert run.state['synthesis_idle_rounds'] == 12
    assert not run.state['artifacts']


def test_synthesis_counter_survives_resume(tmp_path):
    run = AutoResearchRun(tmp_path,'idea')
    literature(run)
    run.state['synthesis_idle_rounds'] = 7
    run.state['iterations'] = 5
    run.halt('interrupted','Synthetic')
    restored = AutoResearchRun(tmp_path,'idea')
    restored.inherit_checkpoint(run.path.parent.name,'continue')
    assert restored.state['synthesis_idle_rounds'] == 7
    assert restored.limit == 35
    assert restored.idea_stage() == 'analysis'


def test_nested_attempted_delivery_is_watched_without_recursive_scan(tmp_path):
    run = AutoResearchRun(tmp_path,'idea')
    literature(run)
    run.start_round()
    run.observe('read_workspace_file', {'path':'study/IDEA.md'}, {'success':False,'error':'file_not_found'})
    (tmp_path/'study').mkdir()
    (tmp_path/'study/IDEA.md').write_text('Synthetic evidence-gap report, incomplete.')
    run.end_round()
    assert run.idea_stage() == 'validation'
    assert 'study' in run.closeout_prompt()
    assert run.state['synthesis_idle_rounds'] == 0
    assert run.state['status'] == 'running'


@pytest.mark.parametrize('mode,expected',[('ask','ask'),('risk','ask'),('never','allow'),('read_only','deny')])
def test_writer_permissions(tmp_path,mode,expected):
    assert tool_permission(mode,'write_research_file',{},tmp_path) == expected


@pytest.mark.parametrize('path',['../escape.md','.neurodiscovery/report.md','core/report.md','AGENTS.md','result.md:secret'])
def test_writer_rejects_unsafe_paths(tmp_path,path):
    result = write_research_file(tmp_path,{'path':path,'kind':'deliverable','content':'Do not write'},threading.Event())
    assert not result['success']


def test_writer_unicode_and_no_overwrite(tmp_path):
    args = {'path':'study/IDEA.md','kind':'deliverable','content':'# 中文假设\n证据尚不足，不宣称新发现。'}
    assert write_research_file(tmp_path,args,threading.Event())['success']
    assert not write_research_file(tmp_path,args|{'content':'overwrite'},threading.Event())['success']
    assert (tmp_path/'study/IDEA.md').read_text(encoding='utf-8') == args['content']
    cancel = threading.Event(); cancel.set()
    assert not write_research_file(tmp_path,args|{'path':'cancel.md'},cancel)['success']
    assert not (tmp_path/'cancel.md').exists()


def test_invalid_candidates_never_write(tmp_path):
    result = write_research_file(tmp_path,{'path':'candidates.json','kind':'hypotheses','content':'[{"hypothesis":"unsupported"}]'},threading.Event())
    assert not result['success']
    assert not (tmp_path/'candidates.json').exists()


@pytest.mark.parametrize('accepted',[True,False])
@pytest.mark.parametrize('probe', ['dir /b', 'python -c "import glob; print(glob.glob(\'*\'))"'])
def test_post_retrieval_regression_then_real_writing_and_review(tmp_path,monkeypatch,accepted,probe):
    from core.test_idea_hypotheses import synthetic_candidate, synthetic_batch
    from core import novelty_gate
    monkeypatch.setattr(novelty_gate, 'resolve_graph_evidence', synthetic_batch)
    candidates = [synthetic_candidate()]
    def search(workspace, dependencies, args, cancel, state):
        (workspace/'papers.json').write_text(json.dumps(PAPERS))
        return {'success':True,'path':'papers.json','count':1}
    monkeypatch.setattr(main,'search_with_recovery',search)
    executed = []
    monkeypatch.setattr(main,'_run_shell_command',lambda **kwargs:executed.append(kwargs) or {'success':True})
    sequence = [reply(calls=[tool('search_pubmed',query='synthetic',output='papers.json')]),
                reply(calls=[tool('read_workspace_file',path='papers.json')]),
                reply(calls=[tool('run_shell_command',command=probe),tool('inspect_local_path',path='.')]),
                reply(calls=[tool('write_research_file',path='study/candidates.json',kind='hypotheses',content=json.dumps(candidates))]),
                reply(calls=[tool('write_research_file',path='study/IDEA.md',kind='deliverable',content='# Synthetic idea\nPMID:123. Prediction, contrary evidence, limitations; unverified novelty.')]),
                reply(calls=[tool('read_workspace_file',path='study/IDEA.md')]),
                reply(calls=[tool('finish_autoresearch',**completion(artifacts=['study/IDEA.md'],evidence_ids=[1,2,5,6,7]))]),
                reply(json.dumps({'accepted':accepted,'reason':'Synthetic independent verdict'}))]
    agent,calls = session(tmp_path,sequence,mode='idea')
    agent.env['autoresearch_independent_review'] = True
    agent._chat()
    assert agent.autoresearch_state['status'] == ('completed' if accepted else 'review_required')
    assert executed == []
    assert not agent.autoresearch_state['evidence'][2]['executed']
    assert not agent.autoresearch_state['evidence'][3]['executed']
    assert 'synthesis: analysis' in calls[1]['messages'][-1]['content']
    assert 'synthesis: delivery' in calls[4]['messages'][-1]['content']
    assert 'synthesis: validation' in calls[5]['messages'][-1]['content']
    assert len(calls) == 8
    assert 'tools' not in calls[-1]


def test_denied_writer_does_not_create_or_progress(tmp_path,monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','1')
    agent,calls = session(tmp_path,[reply(calls=[tool('write_research_file',path='IDEA.md',kind='deliverable',content='Draft')])],mode='idea')
    monkeypatch.setattr(agent,'_tool_authorized',lambda *args:False)
    agent._chat()
    assert not (tmp_path/'IDEA.md').exists()
    assert not agent.autoresearch_state['research_progress']


def test_data_mode_does_not_force_idea_synthesis(tmp_path):
    run = AutoResearchRun(tmp_path,'data')
    literature(run)
    assert not run.idea_stage()
    assert not run.synthesis_prompt()


def test_listing_disguised_as_python_is_not_analysis():
    assert is_python_listing_probe('python -c "import glob,os;[print(f,os.path.getsize(f)) for f in glob.glob(\'study/**/*\',recursive=True)]"')
    assert not is_python_listing_probe('python -c "import json; print(json.load(open(\'papers.json\')))"')
    assert not is_python_listing_probe('python -c "from pathlib import Path; Path(\'IDEA.md\').write_text(\'analysis\')"')
    assert not is_python_listing_probe('python -c "import glob,os;os.system(\'echo test\');print(glob.glob(\'*\'))"')


def test_negative_report_does_not_require_fabricated_candidate(tmp_path):
    run = AutoResearchRun(tmp_path,'idea')
    literature(run)
    assert 'no defensible candidate' in run.synthesis_prompt()
    run.start_round()
    result = write_research_file(tmp_path,{'path':'gap.md','kind':'deliverable','content':'No defensible novelty established; scope limitations and missing evidence.'},threading.Event())
    run.observe('write_research_file',{},result)
    run.end_round()
    assert run.idea_stage() == 'validation'
    assert 'hypotheses' not in run.state['research_progress']
    assert run.state['status'] == 'running'


def test_actual_writer_approval_prompt_and_denial(tmp_path):
    from core.runtime_store import RuntimeStore
    agent,calls = session(tmp_path,[])
    agent._runtime_store = RuntimeStore(tmp_path/'runtime.sqlite')
    agent._runtime_request_id = 'synthetic'
    agent.permission_mode = 'ask'
    events = []
    def emit(event):
        events.append(event)
        if event.get('status') == 'pending':
            agent._runtime_store.decide('synthetic','write',False)
    agent._emit_execution_event = emit
    assert not agent._tool_authorized('write_research_file',{'path':'IDEA.md','kind':'deliverable','content':'Draft'},'write')
    assert events[0]['type'] == 'approval'
    assert events[0]['arguments']['path'] == 'IDEA.md'
