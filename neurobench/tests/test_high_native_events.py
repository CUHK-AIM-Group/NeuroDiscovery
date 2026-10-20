import json

import pytest

from neurobench.runner.cli_drivers import parse_cli_events


@pytest.mark.parametrize('code,status,expected', [(0,'completed',True), (1,'failed',False), (None,'in_progress',False)])
def test_native_shell_exit_is_separate_from_mcp_result(code,status,expected):
    # Real native wire shape: command items have no "result" member.
    item={'id':'item_1','type':'command_execution','command':'read context/task.md',
          'aggregated_output':'supplied task text','exit_code':code,'status':status}
    raw='\n'.join(json.dumps(r) for r in [
        {'type':'item.started','item':dict(item,exit_code=None,status='in_progress')},
        {'type':'item.completed','item':item},
        {'type':'turn.completed','usage':{'input_tokens':11,'output_tokens':3}}])
    result=parse_cli_events(raw)
    assert len(result['tool_events'])==1
    assert result['tool_events'][0]['ok']==expected
    assert result['tool_events'][0]['exit_code']==code
    assert result['tool_events'][0]['arguments']=={'command':'read context/task.md'}
    assert result['input_tokens']==11 and result['output_tokens']==3


def test_failed_mcp_is_still_failed_and_deduplicated():
    item={'id':'m1','type':'mcp_tool_call','server':'neuro_runtime','tool':'read_skill',
          'arguments':{'name':'python'},'result':{'isError':True}}
    raw='\n'.join(json.dumps({'type':kind,'item':item}) for kind in ('item.started','item.completed'))
    result=parse_cli_events(raw)
    assert len(result['tool_events'])==1 and not result['tool_events'][0]['ok']
    assert result['skills_loaded']==[]
