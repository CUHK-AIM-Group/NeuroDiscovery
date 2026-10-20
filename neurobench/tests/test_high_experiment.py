import json
from pathlib import Path

import pytest

from neurobench.runner.high_tools import TaskTools, prompt
from neurobench.runner.high_channels import native_usage


def test_shared_context_and_arm_specific_tools(tmp_path):
    a=TaskTools(tmp_path,'Conda environment',enabled=False)
    b=TaskTools(tmp_path,'Conda environment',enabled=True)
    na={s['name'] for s in a.schemas()}; nb={s['name'] for s in b.schemas()}
    assert na < nb and nb-na == {'search_skills','read_skill','compile_task','read_skill_section',
                                 'describe_task_contract','validate_proposal'}
    assert a.call('validate_proposal',{'proposal':'x'})['success'] is False
    assert a.call('compile_task',{})['success'] is False
    assert a.call('read_skill_section',{'name':'x','query':'x'})['success'] is False
    assert b.call('compile_task',{})['task_text'] == 'Conda environment'


def test_generic_reader_confines_paths_and_preserves_pages(tmp_path):
    context=tmp_path/'context';context.mkdir();(context/'ref.txt').write_text('a'*21000,encoding='utf-8')
    t=TaskTools(tmp_path,'Task',enabled=False)
    assert not t.call('read_context_file',{'path':'../secret'})['success']
    one=t.call('read_context_file',{'path':'context/ref.txt'})
    two=t.call('read_context_file',{'path':'context/ref.txt','offset':one['next_offset']})
    assert one['content']+two['content']=='a'*21000


def test_proposed_write_has_history_and_cannot_escape(tmp_path):
    t=TaskTools(tmp_path,'Task',enabled=False)
    assert not t.call('save_proposed_file',{'path':'../../escape','content':'x'})['success']
    t.call('save_proposed_file',{'path':'script.py','content':'first'})
    t.call('save_proposed_file',{'path':'script.py','content':'second'})
    assert (tmp_path/'proposals/script.py').read_text()=='second'
    assert len(list((tmp_path/'proposal_history').iterdir()))==1


def test_generic_ast_never_executes(tmp_path):
    target=tmp_path/'forbidden'
    t=TaskTools(tmp_path,'Task',enabled=False)
    assert t.call('check_python',{'code':f'open({str(target)!r},"w").write("x")'})['syntax']=='parse_pass'
    assert not target.exists()


def test_anthropic_cache_components_counted_once():
    r=native_usage('anthropic',{'input_tokens':10,'output_tokens':7,'cache_read_input_tokens':20,'cache_creation_input_tokens':5})
    assert r['input_tokens']==35 and r['total_tokens']==42


def test_reasoning_tokens_not_added_again():
    r=native_usage('responses',{'input_tokens':10,'output_tokens':50,'output_tokens_details':{'reasoning_tokens':40}})
    assert r['total_tokens']==60 and r['reasoning_tokens']==40


def test_missing_usage_is_not_complete():
    assert not native_usage('ollama',{})['usage_complete']


@pytest.mark.parametrize('raw',[{'input_tokens':-1,'output_tokens':1},{'input_tokens':True,'output_tokens':1}])
def test_invalid_native_counters_rejected(raw):
    with pytest.raises(ValueError): native_usage('responses',raw)


def test_prompt_requires_proposal_not_false_execution():
    assert 'Do not execute' in prompt('Task',False)
    assert 'validate_proposal' in prompt('Task',True)
