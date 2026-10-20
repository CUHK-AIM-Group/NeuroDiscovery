"""Driver regressions: native errors never resend; both protocols reach round 48."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from neurobench.runner.exec_hard.common import ROOT,sha
from neurobench.runner.exec_hard.trial import attempt


@pytest.fixture
def hard_run(tmp_path):
    original=ROOT/'dev/runs/benchmark_hard_trial_20261020_r1/packet'
    if not (original/'INDEX.json').is_file():
        pytest.skip('Requires the retained local hard-trial packet')
    run=tmp_path/'run';packet=run/'packet'
    (packet/'public/cases').mkdir(parents=True)
    shutil.copy2(original/'INDEX.json',packet/'INDEX.json')
    shutil.copytree(original/'public/cases/H07',packet/'public/cases/H07')
    shutil.copytree(original/'public/library',packet/'public/library')
    return run,{'packet_index_sha256':sha(packet/'INDEX.json')}


@pytest.mark.parametrize('model',['gpt-6.1-sol','deepseek-v4.1-flash'])
def test_native_error_is_retained_without_resend(hard_run,model):
    run,manifest=hard_run
    slot={'id':'native-error','case_id':'H07','model':model,'arm':'NR_OFF'}
    counter=[]
    def fail(history,schemas,log,deadline,**kwargs):
        counter.append(1);log.write('model_request',body={'model':model},endpoint='mock://offline')
        raise RuntimeError('native_http_503')
    with patch('neurobench.runner.exec_brief_gpt61_channel.responses_round',side_effect=fail),patch('neurobench.runner.exec_brief.channel.flash_round',side_effect=fail):
        result=attempt(run,manifest,slot)
    assert len(counter)==1
    assert result['status']=='FAILED_KNOWN'
    assert result['native_request_count']==1 and result['submissions']==[]
    assert not result['final_pass']
    assert (run/'attempts/native-error/ALLOCATED.json').is_file()
    assert (run/'attempts/native-error/RESULT.json').is_file()


@pytest.mark.parametrize('model',['gpt-6.1-sol','deepseek-v4.1-flash'])
def test_both_protocols_deliver_at_round_48_with_shared_complex_budget(hard_run,model):
    run,manifest=hard_run;rounds=[]
    slot={'id':'last-round','case_id':'H07','model':model,'arm':'NR_ON'}
    def respond(history,schemas,log,deadline,**kwargs):
        n=len(rounds)+1;rounds.append(n)
        names={s['name'] for s in schemas}
        assert {'run_project','finish_result','compile_task','read_skill_section'}<=names
        if n>1:
            assert any(v.get('type')=='function_call_output' or v.get('role')=='tool' for v in history)
        name='read_file' if n<48 else 'finish_result'
        args={'path':'inputs/protocol.json'} if n<48 else {'status':'blocked','reason':'Offline protocol test; no scientific artifacts generated.'}
        call={'name':name,'arguments':args,'call_id':'call_'+str(n)}
        usage={'total_tokens':1,'input_tokens':1,'output_tokens':0,'usage_complete':True}
        if model=='gpt-6.1-sol':
            return {'output':[{'type':'function_call','name':name,'call_id':call['call_id'],'arguments':json.dumps(args)}],
                'response':{'model':model},'calls':[call],'text':''},usage
        message={'role':'assistant','content':'','tool_calls':[{'id':call['call_id'],'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
        return {'message':message,'model':'deepseek-flash','calls':[call],'text':''},usage
    with patch('neurobench.runner.exec_brief_gpt61_channel.responses_round',side_effect=respond),patch('neurobench.runner.exec_brief.channel.flash_round',side_effect=respond):
        result=attempt(run,manifest,slot)
    assert rounds==list(range(1,49))
    assert result['native_request_count']==48 and result['tool_calls']==48
    assert result['status']=='SUBMITTED' and len(result['submissions'])==1
    assert result['submissions'][0]['model_round']==48
    assert result['submissions'][0]['elapsed_seconds']<900
    assert not result['final_pass']  # Whole-task blocking cannot pass this supported case.
