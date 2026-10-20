"""Explicit adapter over the existing exactly-once attempt implementation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import html
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from neurobench.runner.exec_budget_v3 import COMPLEX
from neurobench.runner.exec_dev.common import environment
from .common import *
from .oracle import evaluate
from .execution import replay


def bind_shared():
    # Process-local dependency binding, fully included in the source capsule.
    # Common inference, receipts, budgets, cooperative host and tool gates are unchanged.
    from neurobench.runner.exec_hard import trial as shared
    shared.BASE_PROMPT=PROMPT;shared.REVISION=REVISION
    shared.evaluate=evaluate;shared.replay=replay
    return shared


def public_prompt(case,arm):
    common=PROMPT+'\nPUBLIC TASK:\n'+(case.root/'task.md').read_text(encoding='utf-8')+'\nINPUT DIRECTORY ENTRIES:\n'+', '.join(sorted(p.name for p in (case.root/'inputs').iterdir()))
    return common+('\nCurrent NeuroRuntime read-only tools are available optionally: search_skills, read_skill, compile_task, read_skill_section. They do not establish scientific correctness.\n' if arm=='NR_ON' else '')


def frozen_launcher(snapshot):
    return f"import os,sys\nsys.dont_write_bytecode=True\nos.environ['NEUROCLAW_HARD_REPO']={str(ROOT)!r}\nsys.path.insert(0,{str(snapshot)!r})\nfrom neurobench.runner.exec_open.trial import main\nmain()\n"


def initialize(run,qualification):
    run=Path(run).resolve();packet=run/'packet';qualification=Path(qualification).resolve()
    if (run/'MANIFEST.json').exists() or (run/'source_snapshot').exists():raise FileExistsError('cannot_overwrite_or_reinitialize')
    q=read_json(qualification)
    if q['status']!='PASS' or q['packet_index_sha256']!=sha(packet/'INDEX.json') or q['packet_public_hashes']!=file_map(packet/'public') or q['packet_private_hashes']!=file_map(packet/'private'):raise ValueError('qualification_binding_failed')
    suites=list(ET.parse(run/'OFFLINE_TESTS.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.get(k,'0')) for s in suites for k in ('failures','errors')):raise ValueError('offline_tests_failed')
    shared=bind_shared();binding=shared.bind_channels()
    from neurobench.runner.high_gpt_route import XNOVA_ROUTE,route_key
    from neurobench.runner.high_channels import credential
    if not route_key(XNOVA_ROUTE) or not credential('deepseek'):raise ValueError('credential_unavailable')
    snapshot=run/'source_snapshot';snapshot.mkdir()
    for package in ('core','neurobench'):
        source=ROOT/package
        for path in source.rglob('*.py'):
            if '__pycache__' in path.parts or (package=='neurobench' and any(p in ('tasks','tests') for p in path.relative_to(source).parts)):continue
            dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    prompts=run/'prompts';prompts.mkdir();slots=[]
    for i,cid in enumerate(CASE_IDS):
        case=load_case(packet,cid)
        for j,model in enumerate(MODELS):
            for arm in (ARMS if (i+j)%2==0 else ARMS[::-1]):
                sid=f'{cid}__{model}__{arm}';slots.append({'id':sid,'case_id':cid,'model':model,'arm':arm})
                (prompts/(sid+'.txt')).write_text(public_prompt(case,arm),encoding='utf-8')
    # New synthetic fixtures and task text must not be any previously exposed old hard inputs.
    old=ROOT/'dev/runs/benchmark_hard_trial_20261020_r1'
    historical=read_json(old/'MANIFEST.json');old_values=set(historical['public_packet_hashes'].values())
    novel={}
    for cid in CASE_IDS:
        case=load_case(packet,cid);paths=[p for p in (case.root/'inputs').rglob('*') if p.is_file() and p.name!='protocol.json']
        novel[cid]={'all_nonprotocol_input_payloads_new':all(sha(p) not in old_values for p in paths),'task_text_new':sha(case.root/'task.md') not in old_values,'payloads':len(paths)}
    if not all(v['all_nonprotocol_input_payloads_new'] and v['task_text_new'] for v in novel.values()):raise ValueError('historically_exposed_payload')
    env=environment()
    for name in ('sklearn','pandas','statsmodels'):env[name]=__import__(name).__version__
    manifest={'revision':REVISION,'created_at_utc':utc(),'user_request':USER_REQUEST,
        'scope':'Four new authored synthetic calibration cases; not real clinical validation, independent tasks or preregistered superiority evidence.',
        'slots':slots,'models':list(MODELS),'arms':list(ARMS),'budget':asdict(COMPLEX),'parallelism':4,
        'stage_seconds':7200,'stage_observed_token_ceiling':10000000,'maximum_native_requests':len(slots)*48,
        'attempt_policy':'Exclusive allocation before dispatch, exactly one attempt per frozen slot, no automatic retries/recovery/resend. Two public structural submissions only; no score-based repair.',
        'source_adapter':'exec_open binds only public prompt/revision/private evaluate/private clean replay into exec_hard.trial.attempt; actual tool host/native channels/receipts/budgets unchanged.',
        'routes':historical['routes'],'same_public_knowledge_and_common_primitives':True,
        'same_optional_library_hashes':file_map(packet/'public/library'),'nr_schema_delta':['search_skills','read_skill','compile_task','read_skill_section'],
        'scoring':read_json(packet/'public/cases/O01/inputs/protocol.json')['acceptance_and_scoring'],
        'packet_index_sha256':sha(packet/'INDEX.json'),'public_packet_hashes':file_map(packet/'public'),'private_packet_hashes':file_map(packet/'private'),
        'snapshot_hashes':file_map(snapshot),'prompt_hashes':file_map(prompts),'channel_path_binding':binding,
        'qualification':{'path':str(qualification.relative_to(run)),'sha256':sha(qualification),'calibration':q['calibration']},
        'offline_tests_sha256':sha(run/'OFFLINE_TESTS.xml'),'offline_tests':sum(int(s.get('tests','0')) for s in suites),
        'novelty_against_previous_exposed_hard_packet':novel,'historical_preservation':{f:sha(old/f) for f in ['MANIFEST.json','FINAL_READBACK.json','FINAL_SEAL.json','CURRENT_REVIEW_ENTRYPOINTS.json']},
        'environment':env,'scientific_rules_frozen_before_inference':True,'model_requests_before_manifest':0,
        'continuation':{'path':'CONTINUATION.json','sha256':sha(run/'CONTINUATION.json')} if (run/'CONTINUATION.json').exists() else None}
    write_json(run/'MANIFEST.json',manifest)
    (run/'RUN_FROZEN.py').write_text(frozen_launcher(snapshot),encoding='utf-8')
    write_json(run/'BEFORE_DISPATCH.json',{'at_utc':utc(),'manifest_sha256':sha(run/'MANIFEST.json'),'launcher_sha256':sha(run/'RUN_FROZEN.py'),'prompts':file_map(prompts),'packet_index_sha256':manifest['packet_index_sha256'],'request_exposures':0,'status':'READY'})
    print(json.dumps({'ready':True,'slots':len(slots),'snapshot_files':len(manifest['snapshot_hashes']),'calibration':q['calibration']}),flush=True)
    return manifest


def verify_frozen(run,manifest):
    for name,key in [('source_snapshot','snapshot_hashes'),('packet/public','public_packet_hashes'),('packet/private','private_packet_hashes'),('prompts','prompt_hashes')]:
        if file_map(Path(run)/name)!=manifest[key]:raise ValueError('frozen_binding_changed:'+name)
    before=read_json(Path(run)/'BEFORE_DISPATCH.json')
    if sha(Path(run)/'MANIFEST.json')!=before['manifest_sha256'] or sha(Path(run)/'RUN_FROZEN.py')!=before['launcher_sha256']:raise ValueError('manifest_or_launcher_changed')
    q=Path(run)/manifest['qualification']['path']
    if sha(q)!=manifest['qualification']['sha256']:raise ValueError('qualification_changed')
    if manifest.get('continuation') and sha(Path(run)/manifest['continuation']['path'])!=manifest['continuation']['sha256']:raise ValueError('continuation_binding_changed')


def deliver(run):
    run=Path(run);manifest=read_json(run/'MANIFEST.json');rows=[]
    for slot in manifest['slots']:
        path=run/'attempts'/slot['id']/'RESULT.json'
        if path.exists():
            r=read_json(path);endpoints=r['verdicts'][-1].get('endpoints',{}) if r['verdicts'] else {}
            rows.append({**slot,**{k:r[k] for k in ('status','final_pass','first_pass','elapsed_seconds','native_request_count','tool_calls','error','usage_complete')},'endpoints':endpoints,'result_path':str(path.relative_to(run))})
        else:rows.append({**slot,'status':'UNRESOLVED' if (run/'attempts'/slot['id']).exists() else 'UNATTEMPTED','final_pass':False})
    totals=[]
    for model in MODELS:
        for arm in ARMS:
            selected=[r for r in rows if r['model']==model and r['arm']==arm]
            totals.append({'model':model,'arm':arm,'valid':sum(r['final_pass'] for r in selected),'denominator':len(selected),'submitted':sum(r['status']=='SUBMITTED' for r in selected),'total_seconds':sum(r.get('elapsed_seconds',0) for r in selected)})
    paired=[]
    for model in MODELS:
        successful=[]
        for cid in CASE_IDS:
            pair={r['arm']:r for r in rows if r['model']==model and r['case_id']==cid}
            if all(pair[a]['final_pass'] for a in ARMS):successful.append(pair)
        off=sum(p['NR_OFF']['elapsed_seconds'] for p in successful);on=sum(p['NR_ON']['elapsed_seconds'] for p in successful)
        paired.append({'model':model,'common_valid_pairs':len(successful),'off_seconds':off,'on_seconds':on,'on_change_pct':(on/off-1)*100 if off else None})
    report={'revision':REVISION,'at_utc':utc(),'slots':rows,'summary':totals,'matched_valid_time':paired,'primary':'Validity and task endpoints are descriptive; no pooled scientific superiority score.','limitations':['Four synthetic authored development cases, one realization per slot.','No independent expert certification of narrative or scientific validity.','Hidden O01 accuracy is a continuous endpoint, not a validity gate.','No historical task or failed/UNKNOWN input was resent.']}
    delivery=run/'delivery';delivery.mkdir(exist_ok=True);write_json(delivery/'SUMMARY.json',report)
    write_csv(delivery/'RESULTS.csv',[{k:r.get(k,'') for k in ('case_id','model','arm','status','final_pass','elapsed_seconds','native_request_count','tool_calls','error')} for r in rows],['case_id','model','arm','status','final_pass','elapsed_seconds','native_request_count','tool_calls','error'])
    content=['<!doctype html><meta charset="utf-8"><title>Open-method calibration</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>Open-method synthetic development calibration</h1><p>One frozen attempt per condition. Descriptive results; no clinical or independent expert acceptance.</p><table><tr><th>Case</th><th>Model</th><th>Arm</th><th>Status</th><th>Valid</th><th>Seconds</th><th>Details</th></tr>']
    for r in rows:
        path=r.get('result_path');link=f'<a href="../{html.escape(path)}">original result</a>' if path else ''
        content.append('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in ('case_id','model','arm','status','final_pass','elapsed_seconds'))+'<td>'+link+'</td></tr>')
    content.append('</table><h2>Summary and task endpoints</h2><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>')
    (delivery/'INDEX.html').write_text('\n'.join(content),encoding='utf-8')
    write_json(run/'CURRENT_DELIVERY_ENTRYPOINTS.json',{'index':'delivery/INDEX.html','summary':'delivery/SUMMARY.json','results':'delivery/RESULTS.csv','manifest':'MANIFEST.json','public_protocols':[f'packet/public/cases/{c}/inputs/protocol.json' for c in CASE_IDS]})
    return report


def execute(run):
    run=Path(run).resolve();m=read_json(run/'MANIFEST.json');verify_frozen(run,m)
    if any((run/'attempts'/s['id']).exists() for s in m['slots']):raise RuntimeError('prior_allocation_no_automatic_resume_or_resend')
    with (run/'CONTROLLER.json').open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'started_at_utc':utc(),'status':'RUNNING'},f)
    shared=bind_shared();shared.bind_channels();started=time.monotonic();results=[];outcomes=[]
    write_json(run/'PROGRESS.json',{'status':'RUNNING','completed':0,'total':len(m['slots']),'at_utc':utc()})
    with ThreadPoolExecutor(max_workers=4) as executor:
        for cid in CASE_IDS:
            if time.monotonic()-started+900>m['stage_seconds'] or sum(r['observed_tokens'] for r in results)>=m['stage_observed_token_ceiling']:break
            batch=[s for s in m['slots'] if s['case_id']==cid]
            pending={executor.submit(shared.attempt,run,m,s):s for s in batch}
            for future in as_completed(pending):
                slot=pending[future]
                try:
                    r=future.result()
                    if sha(run/'attempts'/slot['id']/'PROMPT.txt')!=m['prompt_hashes'][slot['id']+'.txt']:raise ValueError('actual_prompt_changed')
                    results.append(r);outcomes.append({k:r[k] for k in ('id','status','final_pass','elapsed_seconds')})
                except Exception as exc:
                    write_json(run/'attempts'/slot['id']/'CONTROLLER_ERROR.json',{'error_type':type(exc).__name__,'error':str(exc)[:800],'no_resend':True,'at_utc':utc()})
                    outcomes.append({**slot,'status':'CONTROLLER_ERROR','final_pass':False})
                write_json(run/'PROGRESS.json',{'status':'RUNNING','completed':len(outcomes),'total':len(m['slots']),'active_case':cid,'outcomes':outcomes,'at_utc':utc()})
    verify_frozen(run,m)
    write_json(run/'PROGRESS.json',{'status':'FINISHED' if len(outcomes)==len(m['slots']) else 'STOPPED_PARTIAL','completed':len(outcomes),'total':len(m['slots']),'outcomes':outcomes,'at_utc':utc()})
    controller=read_json(run/'CONTROLLER.json');controller.update(status='ENDED',finished_at_utc=utc());write_json(run/'CONTROLLER.json',controller)
    report=deliver(run);print(json.dumps({'summary':report['summary'],'matched_valid_time':report['matched_valid_time']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','qualify','init','run','report']);p.add_argument('run');p.add_argument('--qualification');p.add_argument('--name',default='qualification');p.add_argument('--previous');a=p.parse_args();run=Path(a.run)
    if a.mode=='build':
        from .build import build
        build(run/'packet')
    elif a.mode=='qualify':
        from .qualify import qualify
        qualify(run/'packet',a.name,a.previous)
    elif a.mode=='init':initialize(run,a.qualification)
    elif a.mode=='run':execute(run)
    else:deliver(run)


if __name__=='__main__':main()
