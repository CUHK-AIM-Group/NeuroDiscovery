"""Offline checks for the standalone scoring module."""
import json
import subprocess
import sys
from types import SimpleNamespace

from neurobench.runner import scoring


def test_parsing_and_prompt_imports_do_not_load_agent():
    code = ("import sys; from neurobench.runner import scoring, prompts; "
            "assert 'core.agent.main' not in sys.modules; "
            "assert len(prompts.discover_tasks()) == 500; "
            "assert len(scoring._discover_task_contracts(prompts.REPO_ROOT / 'neurobench/tasks')) == 500")
    subprocess.run([sys.executable, '-I', '-c', code], check=True)


def test_single_report_grading_uses_supplied_backend():
    grades = dict(planning_completeness=8, tool_reasonableness=6,
                  code_command_correctness=7, brief_justification='Offline fixture')
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(grades)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = scoring._score_single_benchmark_case(client, 'Synthetic task', 'Synthetic report',
                                                 'fixture_noskills', 'offline-judge')
    assert all(result[key] == value for key, value in grades.items())
    assert len(calls) == 1 and calls[0]['model'] == 'offline-judge'


def test_core_cli_has_no_old_batch_or_scoring_flags():
    result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'core.agent.main', '--help'],
                            capture_output=True, encoding='utf-8', check=True)
    for flag in ['--score-benchmark', '--benchmark-workers', '--benchmark-repeats', '--benchmark-compare-skills']:
        assert flag not in result.stdout
    assert '--benchmark' in result.stdout
