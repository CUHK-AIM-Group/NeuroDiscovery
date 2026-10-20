"""Exactly one fresh attempt per model/case/NR condition, no inference retries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import psutil

from neurobench.runner.artifacts import EventLog
from neurobench.runner.exec_budget_v3 import COMPLEX
from .common import *
from .host import BASE_PROMPT, HardTools,prepare_workspace,replay,public_checks
from .oracle import evaluate

PRINT_LOCK=threading.Lock()
USER_REQUEST='可以，准备首批测试题，然后我们对 gpt 和 deepseek 做一次测试'


def bind_channels():
    # Package consolidation moved bench/ under neurobench/runner/. The legacy
    # OAuth parser source loader still calculates parents[2]. Bind its original
    # trusted helper path in this process, without changing any legacy source.
    from neurobench.runner import subscription_direct
    subscription_direct.ROOT=ROOT
    helper=subscription_direct.auth_helper()
    return {'legacy_parser_source_path_rebound':True,'helper_source_sha256':subscription_direct.AUTH_SOURCE_SHA256,
            'scope':'Only local source lookup; inference endpoint/authentication/parser contracts unchanged.'}


def initialize(run,qualification):
    run=Path(run).resolve();packet=run/'packet';qualification=Path(qualification).resolve()
    q=read_json(qualification)
    if q['status']!='PASS' or sha(packet/'INDEX.json')!=q['packet_index_sha256'] or file_map(packet/'public')!=q['packet_public_hashes']:
        raise ValueError('qualification_or_packet_binding_failed')
    import xml.etree.ElementTree as ET
    tests=run/'DRIVER_TESTS.xml';suites=ET.parse(tests).getroot().iter('testsuite')
    checks=list(suites)
    if sum(int(s.get('tests','0')) for s in checks)!=4 or any(int(s.get(k,'0')) for s in checks for k in ('failures','errors')):raise ValueError('driver_regressions_not_passed')
    if (run/'MANIFEST.json').exists():raise FileExistsError('manifest_already_exists_no_reinitialization')
    binding=bind_channels()
    # Read credentials for availability only, never log or copy their payloads.
    from neurobench.runner.high_gpt_route import XNOVA_ROUTE,route_key
    from neurobench.runner.high_channels import credential
    if not route_key(XNOVA_ROUTE) or not credential('deepseek'):raise ValueError('credential_unavailable')
    snapshot=run/'source_snapshot';snapshot.mkdir()
    for package in ('core','neurobench'):
        source=ROOT/package
        for path in source.rglob('*.py'):
            if package=='neurobench' and any(p in ('tasks','tests','__pycache__') for p in path.relative_to(source).parts):continue
            if '__pycache__' in path.parts:continue
            destination=snapshot/path.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    slots=[]
    for i,case_id in enumerate(CASE_IDS):
        for j,model in enumerate(MODELS):
            order=ARMS if (i+j)%2==0 else ARMS[::-1]
            for arm in order:slots.append({'id':f'{case_id}__{model}__{arm}','case_id':case_id,'model':model,'arm':arm})
    from neurobench.runner.exec_dev.common import environment
    env=environment()
    for name in ('sklearn','statsmodels','pandas','networkx'):
        module=__import__(name);env[name]=module.__version__
    env.update(cpu_logical=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total)
    original=read_json(packet/'private/PROVENANCE.json')
    manifest={'revision':REVISION,'user_request':USER_REQUEST,'created_at_utc':utc(),'scope':'Authored development comparison; eight cases in six related capability families, not independent held-out evidence.',
        'models':list(MODELS),'arms':list(ARMS),'slots':slots,'budget':asdict(COMPLEX),
        'max_managed_estimator_fits':64,'maximum_native_requests':32*48,'stage_seconds':14400,'stage_observed_token_ceiling':20000000,
        'parallelism':4,'dispatch_order':'Four conditions for the same case run concurrently; cases follow the frozen order. Arm order counterbalances model/case parity.',
        'routes':{'gpt-6.1-sol':{'provider':'xnova_responses','endpoint':XNOVA_ROUTE['endpoint'],'native_model_required':'gpt-6.1-sol','reasoning':'high required in native echo','backend_identity':'Provider reported; third-party endpoint, not independent OpenAI backend attestation.'},
                  'deepseek-v4.1-flash':{'provider':'deepseek','endpoint':'https://api.deepseek.com/chat/completions','wire_model':'deepseek-flash','accepted_native_echoes':['deepseek-flash','deepseek-v4.1-flash','deepseek-v4.1-flash-1017'],'reasoning':'high requested with thinking enabled; effort is not independently echoed by this channel.'}},
        'one_nr_implementation':True,'nr_tools':['search_skills','read_skill','compile_task','read_skill_section'],
        'same_public_knowledge_and_common_primitives':True,'no_private_feedback_or_tool_count_bonus':True,
        'acceptance':'Prequalified scientific endpoints, on-time submission, bound source/context and clean final-entrypoint replay. H03 additionally checks fit-state invariance to external-feature values. Private numerical results never permit an in-session repair.',
        'attempt_policy':'One native attempt per slot. At most two public structural submissions inside budget. No automatic model retries, post-terminal recoveries, hidden-score-based changes or resend of UNKNOWN.',
        'qualification':{'path':str(qualification.relative_to(run)),'sha256':sha(qualification),'calibration':q['calibration']},
        'offline_driver_tests_sha256':sha(tests),'offline_driver_tests':4,
        'packet_index_sha256':sha(packet/'INDEX.json'),'public_packet_hashes':file_map(packet/'public'),'private_packet_hashes':file_map(packet/'private'),
        'snapshot_hashes':file_map(snapshot),'source_bindings':original['source_bindings'],'environment':env,'channel_path_binding':binding,
        'model_requests_before_manifest':0,'scientific_rules_frozen_before_inference':True}
    write_json(run/'MANIFEST.json',manifest)
    launch=f'''import os,sys\nfrom pathlib import Path\nos.environ['NEUROCLAW_HARD_REPO']={str(ROOT)!r}\nsys.path.insert(0,{str(snapshot)!r})\nfrom neurobench.runner.exec_hard.trial import main\nmain()\n'''
    (run/'RUN_FROZEN.py').write_text(launch,encoding='utf-8')
    print(json.dumps({'ready':True,'slots':len(slots),'source_files':len(manifest['snapshot_hashes']),'budget':manifest['budget'],'calibration':q['calibration']},ensure_ascii=False),flush=True)
    return manifest


def attempt(run,manifest,slot):
    run=Path(run);target=run/'attempts'/slot['id'];target.mkdir(parents=True,exist_ok=False)
    write_json(target/'ALLOCATED.json',{'slot':slot,'at_utc':utc(),'policy':'Exclusive reservation; no automatic resend even if result is absent.'})
    case=load_case(run/'packet',slot['case_id'],manifest['packet_index_sha256'])
    work=prepare_workspace(case,run/'workspaces'/slot['id'],run/'packet/public/library')
    input_before=file_map(work/'inputs');context_before=file_map(work/'context')
    common=BASE_PROMPT+'\nPUBLIC TASK:\n'+(case.root/'task.md').read_text(encoding='utf-8')+'\nINPUT DIRECTORY ENTRIES:\n'+', '.join(sorted(p.name for p in (case.root/'inputs').iterdir()))
    instruction=common+('\nCurrent NeuroRuntime read-only tools are available optionally: search_skills, read_skill, compile_task, read_skill_section. They do not establish scientific correctness.\n' if slot['arm']=='NR_ON' else '')
    (target/'PROMPT.txt').write_text(instruction,encoding='utf-8')
    started=time.monotonic();deadline=started+COMPLEX.task_seconds
    tools=HardTools(case,work,target,slot['arm'],deadline,started);schemas=tools.schemas();write_json(target/'SCHEMAS.json',schemas)
    write_json(target/'START.json',{'slot':slot,'at_utc':utc(),'input_hashes':input_before,'context_hashes':context_before,'prompt_sha256':sha(target/'PROMPT.txt')})
    history=[{'role':'user','content':instruction}];native_models=[];usage=[];rounds=0;status='UNKNOWN';error='';final=''
    log=EventLog(target/'events.jsonl')
    from neurobench.runner.exec_brief_gpt61_channel import responses_round,GPT61_ROUTE
    from neurobench.runner.exec_brief.channel import flash_round
    try:
        for index in range(1,COMPLEX.model_rounds+1):
            if time.monotonic()>=deadline:status,error='BUDGET_STOP','task_wall_deadline';break
            if tools.tool_count>=COMPLEX.tool_calls:status,error='BUDGET_STOP','tool_call_budget';break
            feedback=tools.begin_round(index);history.append({'role':'user','content':'Common remaining budget: '+json.dumps(feedback)})
            log.write('budget_feedback',payload=feedback);rounds+=1
            if slot['model']=='gpt-6.1-sol':
                result,counts=responses_round(history,schemas,log,deadline,route=GPT61_ROUTE)
                history.extend(result['output']);native_models.append(result['response'].get('model'))
            else:
                result,counts=flash_round(history,schemas,log,deadline);history.append(result['message']);native_models.append(result['model'])
            usage.append(counts);log.write('native_usage',payload=counts)
            if not counts['usage_complete']:raise RuntimeError('native_usage_incomplete')
            for call in result['calls']:
                args=call['arguments'] if isinstance(call['arguments'],dict) else json.loads(call['arguments'])
                log.write('tool_request',name=call['name'],arguments=args,call_id=call['call_id'])
                value=tools.call(call['name'],args);log.write('tool_result',name=call['name'],result=value,call_id=call['call_id'])
                payload=json.dumps(value,ensure_ascii=False)
                history.append({'type':'function_call_output','call_id':call['call_id'],'output':payload} if slot['model']=='gpt-6.1-sol' else {'role':'tool','tool_call_id':call['call_id'],'name':call['name'],'content':payload})
            if not tools.terminal and (tools.output/'result.json').is_file() and public_checks(case.contract,tools.output)['public_check_pass']:
                log.write('automatic_submission_after_helper_manifest',result=tools.submit())
            if tools.terminal:status='SUBMITTED';break
            if not result['calls']:
                final=result['text'];diagnostic=tools.submit();log.write('implicit_submission_after_final_text',result=diagnostic)
                if tools.terminal:status='SUBMITTED';break
                if time.monotonic()>=deadline:status,error='BUDGET_STOP','task_wall_deadline';break
                history.append({'role':'user','content':'Public structural checks failed; one repair is available within the original budget. '+json.dumps(diagnostic,ensure_ascii=False)})
        else:status,error='BUDGET_STOP','model_round_budget'
    except Exception as exc:
        error=str(exc)[:1200] if isinstance(exc,(RuntimeError,ValueError,TimeoutError)) else type(exc).__name__
        status='FAILED_KNOWN' if error.startswith(('native_http_','native_model_','native_usage_','deepseek_noncomplete_')) else 'UNKNOWN'
        log.write('attempt_error',error_type=type(exc).__name__,error=error,status=status)
    finally:
        elapsed=time.monotonic()-started;log.close()
    input_unchanged=file_map(work/'inputs')==input_before;context_unchanged=file_map(work/'context')==context_before
    verdicts=[]
    for s in tools.submissions:
        v=evaluate(case,s['output'],s['processes']);write_json(Path(s['output']).parent/'VERDICT.json',v);verdicts.append(v)
    clean={'passed':False,'applicable':False};invariance={'passed':True,'applicable':False}
    if tools.submissions and (Path(tools.submissions[-1]['output'])/'project/main.py').is_file():
        try:
            clean=replay(case,tools.submissions[-1]['output'],target/'clean_replay',run/'packet/public/library')
            if case.id=='H03' and clean['passed']:
                invariance=replay(case,tools.submissions[-1]['output'],target/'external_values_replay',run/'packet/public/library',external_perturbation=True)
        except Exception as exc:
            clean={'passed':False,'applicable':True,'error_type':type(exc).__name__,'error':str(exc)[:600]}
    if not tools.submissions:shutil.copytree(tools.output,target/'unfinished_output')
    final_pass=bool(verdicts) and verdicts[-1]['passed'] and clean['passed'] and invariance['passed'] and input_unchanged and context_unchanged and tools.submissions[-1]['elapsed_seconds']<=900
    first_pass=bool(verdicts) and verdicts[0]['passed'] and clean['passed'] and invariance['passed'] and (len(verdicts)==1 or tools.submissions[0]['output_hashes']==tools.submissions[-1]['output_hashes'])
    record={**slot,'started_at_utc':read_json(target/'START.json')['at_utc'],'finished_at_utc':utc(),'status':status,'error':error,'elapsed_seconds':elapsed,
        'native_request_count':rounds,'native_models':native_models,'reasoning_requested':'high','native_usage':usage,
        'observed_tokens':sum(u['total_tokens'] for u in usage),'usage_complete':len(usage)==rounds and all(u['usage_complete'] for u in usage),
        'tool_calls':tools.tool_count,'submissions':tools.submissions,'verdicts':verdicts,'first_pass':first_pass,'final_pass':final_pass,
        'clean_replay':clean,'external_fit_invariance':invariance,'inputs_unchanged':input_unchanged,'context_unchanged':context_unchanged,'final_text':final,
        'no_post_terminal_model_resend':True,'delivery_protocol':REVISION}
    write_json(target/'RESULT.json',record)
    with PRINT_LOCK:print(json.dumps({k:record[k] for k in ('id','status','final_pass','elapsed_seconds','native_request_count','error')},ensure_ascii=False),flush=True)
    return record


def execute(run):
    run=Path(run).resolve();m=read_json(run/'MANIFEST.json')
    if file_map(run/'source_snapshot')!=m['snapshot_hashes']:raise ValueError('frozen_source_changed')
    if file_map(run/'packet/public')!=m['public_packet_hashes'] or file_map(run/'packet/private')!=m['private_packet_hashes']:raise ValueError('packet_changed')
    if any((run/'attempts'/s['id']).exists() for s in m['slots']):raise RuntimeError('allocated_slot_exists_no_automatic_resume')
    with (run/'CONTROLLER.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'at_utc':utc()},f)
    bind_channels();start=time.monotonic();results=[]
    write_json(run/'PROGRESS.json',{'status':'RUNNING','completed':0,'total':len(m['slots']),'at_utc':utc()})
    with ThreadPoolExecutor(max_workers=4) as executor:
        for case_id in CASE_IDS:
            if time.monotonic()-start+900>m['stage_seconds'] or sum(r['observed_tokens'] for r in results)>=m['stage_observed_token_ceiling']:break
            batch=[s for s in m['slots'] if s['case_id']==case_id]
            futures={executor.submit(attempt,run,m,slot):slot for slot in batch}
            for future in as_completed(futures):
                slot=futures[future]
                try:results.append(future.result())
                except Exception as exc:
                    write_json(run/'attempts'/slot['id']/'CONTROLLER_ERROR.json',{'error_type':type(exc).__name__,'error':str(exc)[:700],'at_utc':utc(),'no_resend':True})
                    print(json.dumps({'controller_error':slot['id'],'error_type':type(exc).__name__}),flush=True)
                write_json(run/'PROGRESS.json',{'status':'RUNNING','completed':len(results),'total':len(m['slots']),'at_utc':utc(),'active_case':case_id,
                    'outcomes':[{k:r[k] for k in ('id','status','final_pass','elapsed_seconds')} for r in results]})
    write_json(run/'PROGRESS.json',{'status':'FINISHED' if len(results)==len(m['slots']) else 'STOPPED_PARTIAL','completed':len(results),'total':len(m['slots']),'at_utc':utc()})
    from .report import deliver
    deliver(run)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','run','report']);p.add_argument('run');p.add_argument('--qualification');a=p.parse_args()
    if a.mode=='init':initialize(a.run,a.qualification)
    elif a.mode=='run':execute(a.run)
    else:
        from .report import deliver
        deliver(a.run)


if __name__=='__main__':main()
