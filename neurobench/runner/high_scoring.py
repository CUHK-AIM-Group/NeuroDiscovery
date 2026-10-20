"""Finite Astra/high judging for complete model/task bundles only.

Uses the unchanged historical 30/40/30 rubric, captures its prompt offline,
and sends no identity, timings, tokens, or harness usage counts to the judge.
Importing this module never dispatches a model.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from neurobench.runner.artifacts import EventLog,file_sha256,freeze_text,json_text
from neurobench.runner.subscription_direct import completed_response,get_access_record,response_body
from neurobench.runner.high_channels import _http,native_usage

MODEL='gpt-6-astra'
DIMS=('planning_completeness','tool_reasonableness','code_command_correctness')
WEIGHTS=(.30,.40,.30)
SEED='neuroruntime-high-comparison-20261013-judge1'


class Captured(BaseException):
    def __init__(self,request): self.request=request


def capture_rubric(rubric_path,task,reports):
    def capture(**request): raise Captured(request)
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
    scope={'Any':Any,'json':json,'re':re,
           'BENCHMARK_WITH_SKILLS_SUFFIX':'_with_skills','BENCHMARK_NO_SKILLS_SUFFIX':'_no_skills',
           '_build_skill_hint_summary':lambda _: (_ for _ in ()).throw(RuntimeError('anonymous_judge_has_no_skill_hints')),
           '_retry_api_call':lambda label,fn,**_:fn()}
    exec(compile(Path(rubric_path).read_text(encoding='utf-8'),str(rubric_path),'exec'),scope)
    try: scope['_score_batch_benchmark_case'](client,task,reports,MODEL,skills=[])
    except Captured as c: return c.request['messages']
    raise RuntimeError('rubric_did_not_capture_request')


def blind_reports(case,model,rows):
    ordered=sorted(rows,key=lambda r:r['arm'])
    random.Random(int(hashlib.sha256((SEED+'|'+case+'|'+model).encode()).hexdigest(),16)).shuffle(ordered)
    mapping=[];reports={}
    for i,r in enumerate(ordered,1):
        alias=f'Candidate{i:02d}'
        reports[alias]=[{'run_index':1,'report_text':r['text']}]
        mapping.append({'alias':alias,'arm':r['arm'],'condition_id':r['condition_id'],'answer_sha256':r['answer_sha256']})
    return reports,mapping


def parse_scores(text,aliases):
    rows=json.loads(text)
    if not isinstance(rows,list) or len(rows)!=len(aliases): raise ValueError('judge_bundle_shape_mismatch')
    seen=set();result=[]
    for r in rows:
        name=r.get('model')
        if name not in aliases or name in seen or type(r.get('run_index')) is not int or r['run_index']!=1:
            raise ValueError('judge_alias_identity_mismatch')
        seen.add(name)
        for key in DIMS:
            value=r.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not 1<=value<=10:
                raise ValueError('judge_score_out_of_range')
        result.append(dict(r,quality_100=round(10*sum(r[k]*w for k,w in zip(DIMS,WEIGHTS)),6)))
    return result


def prepare(run,rubric_path):
    from neurobench.runner.high_reporting import summarize
    metrics,coverage=summarize(run)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    out=run/'judging';out.mkdir(exist_ok=False)
    from neurobench.runner.artifacts import freeze_bytes
    freeze_bytes(out/'rubric_functions.py',Path(rubric_path).read_bytes())
    jobs=[]
    for model,block in coverage['complete_bundles'].items():
        for task in manifest['tasks']:
            selected=[r for r in metrics if r['model_block']==model and r['case_id']==task['case_id']]
            if not all(r['status']=='COMPLETE' for r in selected): continue
            rows=[]
            for row in selected:
                receipt=json.loads(Path(row['receipt_path']).read_text(encoding='utf-8'))
                path=Path(receipt['native_final_path'])
                assert file_sha256(path)==receipt['native_final_sha256']
                rows.append(dict(row,text=path.read_text(encoding='utf-8'),answer_sha256=receipt['native_final_sha256']))
            reports,mapping=blind_reports(task['case_id'],model,rows)
            task_text=(run/'tasks'/(task['case_id']+'.md')).read_text(encoding='utf-8')
            messages=capture_rubric(out/'rubric_functions.py',task_text,reports)
            body=response_body(MODEL,messages,[])
            ident=model+'__'+task['case_id']
            freeze_text(out/'jobs'/(ident+'.json'),json_text(body))
            freeze_text(out/'blind_maps'/(ident+'.json'),json_text(mapping))
            jobs.append({'job_id':ident,'aliases':sorted(reports),'request_path':str(out/'jobs'/(ident+'.json')),
                         'request_sha256':file_sha256(out/'jobs'/(ident+'.json')),
                         'blind_map_path':str(out/'blind_maps'/(ident+'.json')),
                         'blind_map_sha256':file_sha256(out/'blind_maps'/(ident+'.json')),
                         'case_id':task['case_id'],'model_block':model})
    assert len(jobs)<=80
    freeze_text(out/'JOBS.json',json_text(jobs))
    freeze_text(out/'JUDGE_GATE.json',json_text({
        'model':MODEL,'reasoning_effort':'high','endpoint':'https://api.openai.com/v1/responses',
        'route_amendment':'Same authorized application subscription grant; exact Astra listed in current public model catalog. Historical r38 judge used NVIDIA.',
        'rubric_sha256':file_sha256(out/'rubric_functions.py'),'weights':dict(zip(DIMS,WEIGHTS)),
        'max_bundle_attempts':80,'max_attempts_per_bundle':1,'bundle_deadline_seconds':600,'stage_wall_seconds':7200,
        'recovery_budget':0,'unknowns_resent':False,'strict_JSON_scores_no_clamping':True,
        'jobs_sha256':file_sha256(out/'JOBS.json'),'complete_group_only':True,'planned_slots_denominator':240,
        'missing_anchors_prevent_superiority_claim':True,'billing_equals_tokens_claim':False}))
    return jobs


def run_judge(run):
    out=run/'judging'
    if (out/'JUDGE_START.json').exists(): raise RuntimeError('judge_already_started_inspect_do_not_retry')
    gate=json.loads((out/'JUDGE_GATE.json').read_text(encoding='utf-8'))
    assert gate['jobs_sha256']==file_sha256(out/'JOBS.json')
    assert gate['model']==MODEL and gate['recovery_budget']==0
    jobs=json.loads((out/'JOBS.json').read_text(encoding='utf-8'))
    freeze_text(out/'JUDGE_START.json',json_text({'planned_bundles':len(jobs),'wall_seconds':7200,'no_generation_in_this_phase':True}))
    stage=time.monotonic()+7200
    settled=0
    for job in jobs:
        if time.monotonic()+600>=stage: break
        deadline=time.monotonic()+600
        attempt=out/'attempts'/job['job_id'];attempt.mkdir(parents=True,exist_ok=False)
        freeze_text(attempt/'ALLOCATION.json',json_text({'job':job,'automatic_retries':0}))
        log=EventLog(attempt/'events.jsonl')
        value={'status':'UNKNOWN','usage_complete':False}
        try:
            assert file_sha256(job['request_path'])==job['request_sha256']
            assert file_sha256(job['blind_map_path'])==job['blind_map_sha256']
            body=json.loads(Path(job['request_path']).read_text(encoding='utf-8'))
            key=get_access_record(minimum_valid_seconds=720)['access_token']
            events=list(_http(body,gate['endpoint'],key,log,deadline,provider='responses'))
            result=completed_response(events);native=result['response']
            if native.get('model')!=MODEL or (native.get('reasoning') or {}).get('effort')!='high':
                raise ValueError('native_judge_model_or_high_echo_mismatch')
            usage=native_usage('responses',native.get('usage') or {})
            freeze_text(attempt/'native_answer.txt',result['text'])
            assert usage['usage_complete']
            scores=parse_scores(result['text'],set(job['aliases']))
            freeze_text(attempt/'scores.json',json_text(scores))
            value={'status':'COMPLETE','usage':usage,'native_model':native['model'],'native_effort':'high',
                   'native_answer_sha256':file_sha256(attempt/'native_answer.txt'),'scores_sha256':file_sha256(attempt/'scores.json')}
        except Exception as error:
            value.update(error_type=type(error).__name__,error=str(error) if isinstance(error,(ValueError,RuntimeError)) else type(error).__name__)
        finally: log.close()
        freeze_text(attempt/'RECEIPT.json',json_text(value))
        settled+=1
        if value['status']!='COMPLETE': break
    freeze_text(out/'JUDGE_TERMINAL.json',json_text({'attempted_bundles':settled,'planned_bundles':len(jobs),'all_scores_complete':settled==len(jobs) and all(json.loads(p.read_text(encoding='utf-8')).get('status')=='COMPLETE' for p in (out/'attempts').rglob('RECEIPT.json'))}))
