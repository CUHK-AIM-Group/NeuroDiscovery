"""Predeclare and qualify the GPT-only rerun; no case dispatch in this module."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time

from neurobench.runner.artifacts import EventLog
from neurobench.runner.exec_dev.common import read_json,sha,write_json
from neurobench.runner.exec_budget_gpt61_trial import prepare,REPO


def preflight(root):
    prepare(root,'那如果题目没问题，gpt 也重测一下')
    m=read_json(root/'MANIFEST.json')
    flash_root=REPO/'tmp/benchmark_exec_budget_flash_trial_20261019_r1'
    flash=read_json(flash_root/'MANIFEST.json')
    same_fields=('packet_index_sha256','qualification_sha256','budget','methods','common_prompt','arm_prompts')
    identical={k:m[k]==flash[k] for k in same_fields}
    assert all(identical.values()),'protocol_or_scientific_input_changed'
    runtime={p:h for p,h in flash['source_bindings'].items()
             if '/core/' in p.replace('\\','/') or '/neurobench/runner/exec_brief/' in p.replace('\\','/')
             or '/neurobench/runner/exec_dev/' in p.replace('\\','/') or Path(p).name in ('exec_budget_v3.py','exec_protocol_v2.py','exec_trial.py','high_tools.py','exec_brief_acceptance_r2.py')}
    assert all(sha(p)==h for p,h in runtime.items()),'flash_shared_runtime_changed'
    write_json(root/'SAME_PROTOCOL_AS_FLASH.json',{'passed':True,'fields':identical,'runtime_bindings':runtime,
        'flash_manifest':str(flash_root/'MANIFEST.json'),'flash_manifest_sha256':sha(flash_root/'MANIFEST.json')})
    # Reuse the recent full common-tool qualification only with identical sources.
    qualification=read_json(flash_root/'qualification_budget/UNIT_TEST_RESULT.json')
    assert qualification['passed'] and qualification['tests']==22
    for name in ('exec_budget_v3.py','exec_protocol_v2.py','test_exec_budget_v3.py','test_exec_protocol_v2.py'):
        p=str(REPO/'neurobench'/('tests' if name.startswith('test_') else 'runner')/name)
        if p in flash['source_bindings']:
            assert sha(p)==flash['source_bindings'][p],p
    shutil.copytree(flash_root/'qualification_budget',root/'qualification_common_budget')
    command=[sys.executable,'-X','utf8','-m','unittest','neurobench.tests.test_exec_budget_gpt61.GptBudgetTests','-v']
    test=subprocess.run(command,capture_output=True)
    (root/'GPT_UNIT_STDOUT.txt').write_bytes(test.stdout);(root/'GPT_UNIT_STDERR.txt').write_bytes(test.stderr)
    write_json(root/'GPT_UNIT_RESULT.json',{'passed':test.returncode==0,'tests':2,'returncode':test.returncode,'command':command})
    assert test.returncode==0,test.stderr.decode('utf-8',errors='replace')[-3000:]
    shutil.copytree(flash_root/'acceptance_roles_r2',root/'acceptance_roles_r2')
    snapshots=[]
    for p,h in m['source_bindings'].items():
        source=Path(p);destination=root/'source_snapshot/files'/source.relative_to(REPO)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        assert sha(destination)==h
        snapshots.append({'source':p,'snapshot':str(destination),'sha256':h})
    write_json(root/'source_snapshot/SOURCE_MAP.json',{'files':snapshots})
    write_json(root/'source_snapshot/EXTRA_SOURCE_MAP.json',{'files':[]})
    packages=[]
    for name in ('benchmark_exec_brief_gpt61_trial_20261019_r1','benchmark_exec_budget_flash_trial_20261019_r1'):
        base=REPO/'tmp'/name;binding=base/'DELIVERY_BINDINGS.json';files=read_json(binding)['files']
        assert all((base/p).is_file() and sha(base/p)==h for p,h in files.items()),name
        packages.append({'root':str(base),'binding':str(binding),'binding_sha256':sha(binding),'files':len(files)})
    write_json(root/'HISTORICAL_BINDINGS.json',{'packages':packages})
    from neurobench.runner.exec_brief_gpt61_channel import responses_round,GPT61_ROUTE
    from neurobench.runner.high_gpt_route import route_key,XNOVA_ROUTE
    from neurobench.runner.exec_protocol_v2 import FINISH_SCHEMA
    import httpx
    with httpx.Client(timeout=30) as client:
        response=client.get('https://api.xnova.online/v1/models',headers={'Authorization':'Bearer '+route_key(XNOVA_ROUTE)})
        response.raise_for_status();catalogue=response.json()
    write_json(root/'preflight_native/MODELS.json',catalogue)
    ids=[item['id'] for item in catalogue.get('data',[])]
    assert 'gpt-6.1-sol' in ids,'gpt61_missing_from_authorized_catalogue'
    log=EventLog(root/'preflight_native/SYNTHETIC_EVENTS.jsonl')
    history=[{'role':'user','content':'Synthetic protocol integration check. No dataset or research task. Call finish_result exactly once with status="blocked" and reason="synthetic check". After the tool result reply DONE without further tools.'}]
    deadline=time.monotonic()+120
    try:
        first,usage1=responses_round(history,[FINISH_SCHEMA],log,deadline,route=GPT61_ROUTE)
        assert len(first['calls'])==1 and first['calls'][0]['name']=='finish_result'
        call=first['calls'][0]
        assert call['arguments']['status']=='blocked' and call['arguments'].get('reason')
        history.extend(first['output'])
        history.append({'type':'function_call_output','call_id':call['call_id'],'output':json.dumps({'success':True,'terminal':True,'synthetic':True})})
        second,usage2=responses_round(history,[FINISH_SCHEMA],log,deadline,route=GPT61_ROUTE)
        assert not second['calls'] and usage1['usage_complete'] and usage2['usage_complete']
        write_json(root/'preflight_native/SYNTHETIC_RESULT.json',{'passed':True,'native_requests':2,
            'first_model':first['response']['model'],'second_model':second['response']['model'],
            'first_reasoning':first['response']['reasoning'],'second_reasoning':second['response']['reasoning'],
            'usage':[usage1,usage2],'case_data_exposed':False,'text':second['text']})
    finally:
        log.close()
    result={'passed':True,'manifest_sha256':sha(root/'MANIFEST.json'),'new_gpt_tests':2,
        'unchanged_common_budget_tests_reused':22,'snapshot_sources':len(snapshots),
        'historical_files_preserved':sum(p['files'] for p in packages),'catalogue_gpt61_available':True,
        'synthetic_roundtrip_requests':2,'task_attempts_started':0,'same_protocol_as_flash':True,
        'preexisting_documentation_drift':m['preexisting_documentation_drift']}
    write_json(root/'PREFLIGHT.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    preflight(parser.parse_args().root.resolve())
