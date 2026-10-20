"""Complete the second half of a saved, successful synthetic tool request.

Never resend the first request. Model cases remain entirely unexposed.
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
    m=read_json(root/'MANIFEST.json')
    assert all(Path(p).is_file() and sha(p)==h for p,h in m['source_bindings'].items())
    path=root/'preflight_native/SYNTHETIC_EVENTS.jsonl'
    rows=[json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]
    assert sum(e['event']=='model_request' for e in rows)==1
    from neurobench.runner.subscription_direct import completed_response
    from neurobench.runner.high_channels import native_usage
    from neurobench.runner.exec_brief_gpt61_channel import responses_round,GPT61_ROUTE
    from neurobench.runner.exec_protocol_v2 import FINISH_SCHEMA
    first=completed_response([e['payload'] for e in rows if e['event']=='native_stream_event'])
    assert first['response']['model']=='gpt-6.1-sol' and first['response']['reasoning']['effort']=='high'
    assert len(first['calls'])==1
    call=first['calls'][0]
    args=call['arguments'] if isinstance(call['arguments'],dict) else json.loads(call['arguments'])
    assert call['name']=='finish_result' and args['status']=='blocked' and args.get('reason')
    write_json(root/'PREFLIGHT_ARGUMENT_READER_NOTE.json',{'recorded_at_utc':utc(),
        'phase':'synthetic_arguments_read','error_type':'TypeError',
        'diagnosis':'Preflight assumed parsed arguments. Responses correctly returned a JSON string; actual benchmark driver already parses both forms.',
        'first_request_completed':True,'first_request_resent':False,'first_wire_sha256':sha(path),
        'case_attempts':0,'inference_transport_or_runtime_changed':False})
    snapshot=root/'source_snapshot/preflight_resume_synthetic_r3.py';shutil.copy2(Path(__file__),snapshot)
    map_path=root/'source_snapshot/EXTRA_SOURCE_MAP.json'
    shutil.copy2(map_path,root/'source_snapshot/EXTRA_SOURCE_MAP.before_synthetic_continuation.json')
    entries=read_json(map_path)['files'];entries.append({'source':str(Path(__file__).resolve()),'snapshot':str(snapshot),'sha256':sha(snapshot)})
    write_json(map_path,{'files':entries})
    history=next(e['body']['input'] for e in rows if e['event']=='model_request')
    history.extend(first['output'])
    history.append({'type':'function_call_output','call_id':call['call_id'],'output':json.dumps({'success':True,'terminal':True,'synthetic':True})})
    log=EventLog(root/'preflight_native/SYNTHETIC_EVENTS_part2.jsonl')
    try:
        second,u2=responses_round(history,[FINISH_SCHEMA],log,time.monotonic()+120,route=GPT61_ROUTE)
        u1=native_usage('responses',first['response'].get('usage') or {})
        assert not second['calls'] and u1['usage_complete'] and u2['usage_complete']
        write_json(root/'preflight_native/SYNTHETIC_RESULT.json',{'passed':True,'native_requests':2,
            'first_request_resent':False,'first_model':first['response']['model'],'second_model':second['response']['model'],
            'first_reasoning':first['response']['reasoning'],'second_reasoning':second['response']['reasoning'],
            'usage':[u1,u2],'case_data_exposed':False,'text':second['text']})
    finally:
        log.close()
    packages=read_json(root/'HISTORICAL_BINDINGS.json')['packages']
    result={'passed':True,'manifest_sha256':sha(root/'MANIFEST.json'),'new_gpt_tests':2,
        'unchanged_common_budget_tests_reused':22,'snapshot_sources':len(m['source_bindings'])+2,
        'historical_files_preserved':sum(p['files'] for p in packages),'catalogue_gpt61_available':True,
        'synthetic_roundtrip_requests':2,'task_attempts_started':0,'same_protocol_as_flash':True,
        'preflight_fixes':['directory GET uses existing inference direct transport','parse saved Responses JSON string; continue only second request'],
        'inference_channel_changed':False,'preexisting_documentation_drift':m['preexisting_documentation_drift']}
    write_json(root/'PREFLIGHT.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    finish(parser.parse_args().root.resolve())
