import json
import sys
import threading
from io import BytesIO

import pytest

from core import pubmed_recovery as recovery
from core.agent import main
from core.agent.test_autoresearch_execution import session, reply, tool
from core.autoresearch_runtime import AutoResearchRun
from core.permissions import tool_permission
from core.tool_outcomes import normalize_shell_outcome


PAPERS = [{"pmid": "123", "title": "Synthetic title", "abstract": "Synthetic abstract"}]


@pytest.mark.parametrize('mode,expected', [('ask','ask'),('risk','ask'),('never','allow'),('read_only','deny')])
def test_pubmed_requires_execution_approval(tmp_path, mode, expected):
    assert tool_permission(mode, recovery.PUBMED_TOOL_NAME, {"query":"test","output":"papers.json"}, tmp_path) == expected


def test_install_verify_retry_same_interpreter(tmp_path, monkeypatch):
    calls = []
    responses = iter([(1,'','missing'), (0,'installed',''), (0,'import_verified',''), (0,json.dumps(PAPERS),'')])
    def execute(arguments, env, cancel):
        calls.append((arguments, env))
        return next(responses)
    monkeypatch.setattr(recovery, '_process', execute)
    monkeypatch.setattr(recovery, 'pubmed_http', lambda *args: pytest.fail('No fallback after successful retry'))
    state = {}
    result = recovery.search_with_recovery(tmp_path, tmp_path/'deps', {'query':'test','output':'papers.json'}, threading.Event(), state)
    assert result['success']
    assert result['interpreter'] == sys.executable
    assert state['install_attempted']
    assert calls[1][0][-1] == 'biopython'
    assert '--target' in calls[1][0]
    assert calls[0][0] == calls[2][0]
    assert all(call[1]['PYTHONPATH'] == str(tmp_path/'deps') for call in calls)
    assert calls[3][0][1:3] == ['pubmed', 'test']
    assert json.loads((tmp_path/'papers.json').read_text()) == PAPERS


@pytest.mark.parametrize('failure', ['install','verify','search'])
def test_failure_falls_back_to_stdlib(tmp_path, monkeypatch, failure):
    outputs = [(1,'','missing'), (0 if failure != 'install' else 1,'',''),
               (0 if failure != 'verify' else 1,'',''), (1,'','search error')]
    calls = []
    def execute(*args):
        calls.append(args[0])
        return outputs[len(calls)-1]
    monkeypatch.setattr(recovery, '_process', execute)
    fallback = []
    monkeypatch.setattr(recovery, 'pubmed_http', lambda *args: fallback.append(args) or PAPERS)
    result = recovery.search_with_recovery(tmp_path, tmp_path/'deps', {'query':'test','output':'papers.json'}, threading.Event(), {})
    assert result['success']
    assert result['recovery_steps'][-1] == 'stdlib_http_fallback'
    assert len(fallback) == 1


def test_failed_install_not_repeated_and_cancellation_has_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, '_process', lambda args, *rest: (1,'','') if args[0] == '-c' else pytest.fail('Repeated pip'))
    monkeypatch.setattr(recovery, 'pubmed_http', lambda *args: [])
    result = recovery.search_with_recovery(tmp_path, tmp_path/'deps', {'query':'test','output':'empty.json'}, threading.Event(), {'install_attempted':True})
    assert result['success'] and result['count'] == 0
    cancel = threading.Event(); cancel.set()
    monkeypatch.setattr(recovery, 'pubmed_http', lambda *args: pytest.fail('No network after cancel'))
    result = recovery.search_with_recovery(tmp_path, tmp_path/'deps', {'query':'test','output':'cancel.json'}, cancel, {'install_attempted':True})
    assert result['error_type'] == 'cancelled'
    assert not (tmp_path/'cancel.json').exists()


@pytest.mark.parametrize('output', ['../outside.json', '.neurodiscovery/autoresearch/papers.json', 'AGENTS.md'])
def test_output_boundary_checked_before_install(tmp_path, monkeypatch, output):
    monkeypatch.setattr(recovery, '_process', lambda *args: pytest.fail('No execution'))
    result = recovery.search_with_recovery(tmp_path, tmp_path/'deps', {'query':'test','output':output}, threading.Event(), {})
    assert not result['success']


def test_pubmed_http_uses_source_bound_records_without_packages(monkeypatch):
    responses = iter([b'{"esearchresult":{"idlist":["123"]}}', b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>A <i>real</i> title</ArticleTitle><Abstract><AbstractText>First</AbstractText><AbstractText>Second</AbstractText></Abstract></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10/test</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'])
    urls = []
    monkeypatch.setattr(recovery, 'urlopen', lambda url, timeout: urls.append(url) or BytesIO(next(responses)))
    papers = recovery.pubmed_http('cerebellum ADHD', 10, threading.Event())
    assert papers[0]['title'] == 'A real title'
    assert papers[0]['pmid'] == '123'
    assert papers[0]['doi'] == '10/test'
    assert papers[0]['abstract'] == 'First\nSecond'
    assert 'term=cerebellum+ADHD' in urls[0]


@pytest.mark.parametrize('payload', [b'{"error":"rate limit"}', b'{"esearchresult":{"ERROR":"failed"}}', b'{}'])
def test_pubmed_http_errors_not_empty_success(monkeypatch, payload):
    monkeypatch.setattr(recovery, 'urlopen', lambda *args, **kwargs: BytesIO(payload))
    with pytest.raises(ValueError):
        recovery.pubmed_http('test', 10, threading.Event())


def test_actual_subprocess_uses_active_python(tmp_path):
    import os
    code, stdout, stderr = recovery._process(['-c','import sys; print(sys.executable)'], dict(os.environ), threading.Event())
    assert code == 0
    assert stdout.strip() == sys.executable


def test_explicit_package_path_works_even_if_pythonpath_is_ignored(tmp_path):
    import os
    (tmp_path/'synthetic_package.py').write_text('value = "loaded"')
    code, stdout, stderr = recovery._process(['-c','import synthetic_package; print(synthetic_package.value)'],
                                             dict(os.environ, PYTHONPATH=str(tmp_path)), threading.Event())
    assert code == 0 and stdout.strip() == 'loaded'


@pytest.mark.parametrize('approved', [True, False])
def test_recovery_approval_emits_ui_request(tmp_path, approved):
    from core.runtime_store import RuntimeStore
    agent, calls = session(tmp_path, [])
    agent._runtime_store = RuntimeStore(tmp_path/'runtime.db')
    agent._runtime_request_id = 'request'
    agent.permission_mode = 'ask'
    events = []
    def emit(event):
        events.append(event)
        if event.get('status') == 'pending':
            agent._runtime_store.decide('request', 'step', approved)
    agent._emit_execution_event = emit
    assert agent._tool_authorized('search_pubmed', {'query':'test','output':'papers.json'}, 'step') is approved
    assert events[0]['type'] == 'approval'
    assert events[0]['tool'] == 'search_pubmed'
    assert events[-1]['status'] == ('approved' if approved else 'denied')


def test_tool_denial_prevents_install_and_network(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','1')
    agent, calls = session(tmp_path, [reply(calls=[tool('search_pubmed', query='test', output='papers.json')])])
    monkeypatch.setattr(agent, '_tool_authorized', lambda *args: False)
    monkeypatch.setattr(main, 'search_with_recovery', lambda *args: pytest.fail('Denied tool must not run'))
    agent._chat()
    assert not agent.autoresearch_state['evidence'][0]['executed']


def test_tool_dispatch_records_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROCLAW_MAX_TOOL_ITERATIONS','1')
    agent, calls = session(tmp_path, [reply(calls=[tool('search_pubmed', query='test', output='papers.json')])])
    monkeypatch.setattr(main, 'search_with_recovery', lambda *args: {'success':True,'count':0,'recovery_steps':['stdlib_http_fallback']})
    agent._chat()
    assert agent.autoresearch_state['evidence'][0]['success']
    assert agent._tool_events[0]['tool'] == 'search_pubmed'


def test_json_and_chinese_use_file_reader(tmp_path):
    (tmp_path/'中文.json').write_text('{"结果":"成功"}', encoding='utf-8-sig')
    result = main._run_shell_command('type "中文.json" | more', tmp_path)
    assert result['error_type'] == 'use_file_reader' and result['executed'] is False
    assert main._read_workspace_file('中文.json', tmp_path)['content'] == '{"结果":"成功"}'


def test_empty_graph_query_has_actionable_recovery():
    result = normalize_shell_outcome('curl "http://127.0.0.1:4444/api/kg/search?q=cerebellum+ADHD"', {'success':True,'stdout':'{"results":[],"query":"cerebellum ADHD"}'})
    assert result['success']
    assert result['diagnostic_code'] == 'empty_graph_search'
    assert 'Split' in result['recovery_hint'] and 'search_pubmed' in result['recovery_hint']


@pytest.mark.parametrize('language,chinese', [('zh',True),('en',False),(None,True)])
def test_stall_report_language_and_evidence(tmp_path, language, chinese):
    run = AutoResearchRun(tmp_path,'idea')
    run.state.update(language=language, objective='关于小脑的新想法', iterations=6)
    run.observe('run_shell_command', {}, {'success':False,'error_type':'missing_dependency'})
    run.observe('run_shell_command', {}, {'success':True,'diagnostic_code':'empty_graph_search'})
    run.observe('search_pubmed', {}, {'success':True,'path':'papers.json','count':3,'recovery_steps':['stdlib_http_fallback']})
    report = run.halt('stalled','generic English reason')
    assert '34' in report
    assert 'generic English reason' not in report
    assert ('缺少依赖' if chinese else 'Missing dependency') in report
    assert ('尚缺' if chinese else 'Missing:') in report
    assert ('未登记' if chinese else 'unregistered') in report
    assert 'search_pubmed' in report
    assert 'papers.json' in report
    assert run.state['closeout']['accepted'] is False
