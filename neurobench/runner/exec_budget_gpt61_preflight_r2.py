"""Finish an unexposed preflight after correcting its catalogue-only proxy use.

The benchmark channel already uses direct TLS for xn ova_responses. No request
body, inference transport, manifest, or original source is changed here.
"""
from pathlib import Path
import argparse
import json
import shutil
import time

from neurobench.runner.artifacts import EventLog
from neurobench.runner.exec_dev.common import read_json,sha,write_json
from neurobench.runner.exec_trial import utc


def finish(root):
    assert not (root/'RUN_START.json').exists() and not (root/'PREFLIGHT.json').exists()
    assert not (root/'preflight_native/SYNTHETIC_EVENTS.jsonl').exists()
    m=read_json(root/'MANIFEST.json')
    assert sha(root/'MANIFEST.json')==read_json(root/'GATE.json')['manifest_sha256']
    assert all(Path(p).is_file() and sha(p)==h for p,h in m['source_bindings'].items())
    assert read_json(root/'GPT_UNIT_RESULT.json')['passed']
    assert read_json(root/'SAME_PROTOCOL_AS_FLASH.json')['passed']
    packages=read_json(root/'HISTORICAL_BINDINGS.json')['packages']
    write_json(root/'PREFLIGHT_DIRECTORY_FAILURE.json',{'recorded_at_utc':utc(),
        'phase':'catalogue_only_get','endpoint':'https://api.xnova.online/v1/models',
        'error_type':'ConnectError','error':'SSL UNEXPECTED_EOF_WHILE_READING using environment proxy',
        'native_inference_calls':0,'task_attempts':0,'source':m['source_bindings'].get(str(Path(__file__).with_name('exec_budget_gpt61_preflight.py'))),
        'diagnosis':'Original directory GET inherited environment proxy. Existing inference channel uses trust_env=False for xn ova_responses. Correct this GET to match that channel; leave inference source and transport unchanged.'})
    snapshot=root/'source_snapshot/preflight_directory_r2.py'
    shutil.copy2(Path(__file__),snapshot)
    oldmap=root/'source_snapshot/EXTRA_SOURCE_MAP.json'
    shutil.copy2(oldmap,root/'source_snapshot/EXTRA_SOURCE_MAP.before_directory_fix.json')
    entries=read_json(oldmap)['files']
    entries.append({'source':str(Path(__file__).resolve()),'snapshot':str(snapshot),'sha256':sha(snapshot)})
    write_json(oldmap,{'files':entries})
    import httpx
    from neurobench.runner.high_gpt_route import route_key,XNOVA_ROUTE
    from neurobench.runner.exec_brief_gpt61_channel import responses_round,GPT61_ROUTE
    from neurobench.runner.exec_protocol_v2 import FINISH_SCHEMA
    with httpx.Client(timeout=30,trust_env=False) as client:
        response=client.get('https://api.xnova.online/v1/models',headers={'Authorization':'Bearer '+route_key(XNOVA_ROUTE)})
        response.raise_for_status();catalogue=response.json()
    write_json(root/'preflight_native/MODELS.json',catalogue)
    assert 'gpt-6.1-sol' in [r['id'] for r in catalogue.get('data',[])]
    log=EventLog(root/'preflight_native/SYNTHETIC_EVENTS.jsonl')
    history=[{'role':'user','content':'Synthetic protocol integration check. No dataset or research task. Call finish_result exactly once with status="blocked" and reason="synthetic check". After the tool result reply DONE without further tools.'}]
    deadline=time.monotonic()+120
    try:
        first,u1=responses_round(history,[FINISH_SCHEMA],log,deadline,route=GPT61_ROUTE)
        assert len(first['calls'])==1 and first['calls'][0]['name']=='finish_result'
        call=first['calls'][0]
        assert call['arguments']['status']=='blocked' and call['arguments'].get('reason')
        history.extend(first['output'])
        history.append({'type':'function_call_output','call_id':call['call_id'],'output':json.dumps({'success':True,'terminal':True,'synthetic':True})})
        second,u2=responses_round(history,[FINISH_SCHEMA],log,deadline,route=GPT61_ROUTE)
        assert not second['calls'] and u1['usage_complete'] and u2['usage_complete']
        write_json(root/'preflight_native/SYNTHETIC_RESULT.json',{'passed':True,'native_requests':2,
            'first_model':first['response']['model'],'second_model':second['response']['model'],
            'first_reasoning':first['response']['reasoning'],'second_reasoning':second['response']['reasoning'],
            'usage':[u1,u2],'case_data_exposed':False,'text':second['text']})
    finally:
        log.close()
    result={'passed':True,'manifest_sha256':sha(root/'MANIFEST.json'),'new_gpt_tests':2,
        'unchanged_common_budget_tests_reused':22,'snapshot_sources':len(m['source_bindings'])+1,
        'historical_files_preserved':sum(p['files'] for p in packages),'catalogue_gpt61_available':True,
        'synthetic_roundtrip_requests':2,'task_attempts_started':0,'same_protocol_as_flash':True,
        'catalogue_transport_correction_only':True,'inference_channel_changed':False,
        'preexisting_documentation_drift':m['preexisting_documentation_drift']}
    write_json(root/'PREFLIGHT.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    finish(parser.parse_args().root.resolve())
