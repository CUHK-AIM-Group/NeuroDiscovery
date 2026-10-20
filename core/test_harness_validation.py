import hashlib
import json

import pytest

from core.harness_validation import code_blocks, task_contract, validate_proposal


@pytest.fixture(autouse=True)
def isolate_opt_in_registration():
    from core import tool_registry
    original = dict(tool_registry.REGISTRY)
    yield
    tool_registry.REGISTRY.clear()
    tool_registry.REGISTRY.update(original)


def answer(code, language='python', fence='```'):
    return f'## Solution Thinking\nUse the real inputs.\n\n## Commands Or Code\n{fence}{language}\n{code}\n{fence}\n'


def test_full_python_is_checked_without_execution(tmp_path):
    target = tmp_path / 'not_created'
    result = validate_proposal('Create a reproducible environment', answer(f"open({str(target)!r}, 'w').write('x')"))
    assert not target.exists()
    assert result['syntax_blocks'][0]['status'] == 'parse_pass'
    assert result['workflow_executed'] is False
    assert result['scientific_correctness_verified'] is False


def test_python_error_reports_original_line():
    result = validate_proposal('Write code', answer('def f():\nprint(1)'))
    issue = next(i for i in result['findings'] if i['code'] == 'python_syntax')
    assert issue['line'] == 7 and issue['severity'] == 'error'


def test_long_fence_preserves_embedded_markdown():
    text = answer('s = """\n```python\nprint(1)\n```\n"""', fence='````')
    result = validate_proposal('Write code', text)
    assert result['blocking_error_count'] == 0
    assert result['syntax_blocks'][0]['status'] == 'parse_pass'


def test_unclosed_fence_is_not_accepted():
    blocks, errors = code_blocks('```python\nx=1')
    assert blocks == [] and errors[0]['code'] == 'unclosed_code_fence'


def test_tilde_fences_and_crlf():
    blocks, errors = code_blocks('~~~python\r\nx=1\r\n~~~\r\n')
    assert len(blocks) == 1 and not errors


def test_extra_heading_remains_error():
    r = validate_proposal('Write code', answer('x=1') + '\n## Appendix\ntext')
    assert any(i['code'] == 'answer_sections' for i in r['findings'])


def test_shell_is_not_claimed_parsed():
    r = validate_proposal('Write a shell command', answer('false', 'bash'))
    assert r['syntax_blocks'][0]['status'] == 'not_parsed'


def test_task_hash_and_no_per_case_oracle():
    task = 'Validate BIDS subjects and train with independent site splits.'
    c = task_contract(task)
    assert c['task_sha256'] == hashlib.sha256(task.encode()).hexdigest()
    assert set(c['domains']) == {'bids', 'learning'}
    assert not c['workflow_executed']


@pytest.mark.parametrize('task', ['', None, 'x' * 500001], ids=['empty', 'not-text', 'oversize'])
def test_invalid_task_rejected(task):
    with pytest.raises(ValueError):
        task_contract(task)


def test_print_only_reconstruction_does_not_pass():
    r = validate_proposal('Run FreeSurfer recon-all', answer("print('recon-all -all -s subject')"))
    assert any(i['code'] == 'print_only_workflow' for i in r['findings'])


def test_real_launcher_and_explicit_freshness():
    r = validate_proposal('Run FreeSurfer recon-all', answer("import subprocess\nstart_time = 1\nsubprocess.run(['recon-all', '-all'], check=True)"))
    assert not any(i['code'] in ('print_only_workflow', 'recon_exit_evidence', 'fresh_output_evidence') for i in r['findings'])


def test_tracking_missing_samples_is_review_not_unsupported_proof():
    r = validate_proposal('Run probtrackx from bedpostx samples', answer('x=1'))
    issues = [i for i in r['findings'] if i['code'].startswith('sampling_')]
    assert len(issues) == 3 and all(i['severity'] == 'review' for i in issues)


def test_environment_which_needs_ownership_check():
    r = validate_proposal('Check a conda environment', answer("import shutil\np = shutil.which('htop')"))
    assert any(i['code'] == 'executable_prefix' for i in r['findings'])


def test_bound_mcp_task_cannot_be_replaced(monkeypatch, tmp_path):
    from core.mcp_stdio import call_tool, tool_definitions
    task = tmp_path / 'task.txt'; task.write_text('Run FreeSurfer recon-all', encoding='utf-8')
    monkeypatch.setenv('NEURORUNTIME_MCP_TOOLS', 'describe_task_contract,validate_proposal')
    monkeypatch.setenv('NEURORUNTIME_MCP_TASK_FILE', str(task))
    monkeypatch.setenv('NEURORUNTIME_MCP_TASK_SHA256', hashlib.sha256(task.read_bytes()).hexdigest())
    assert {r['name'] for r in tool_definitions()} == {'describe_task_contract', 'validate_proposal'}
    r = call_tool('describe_task_contract', {'task_text': 'Other task'}, tmp_path)
    payload = json.loads(r['content'][0]['text'])
    assert 'freesurfer' in payload['domains']
    task.write_text('tampered', encoding='utf-8')
    r = call_tool('describe_task_contract', {}, tmp_path)
    assert r['isError']


def test_default_mcp_unified_surface_keeps_proposal_checks_opt_in(monkeypatch):
    from core.mcp_stdio import tool_definitions
    monkeypatch.delenv('NEURORUNTIME_MCP_TOOLS', raising=False)
    assert {r['name'] for r in tool_definitions()} == {'search_skills', 'read_skill', 'compile_task', 'read_skill_section'}
