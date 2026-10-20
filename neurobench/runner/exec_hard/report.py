"""Cold readback of all attempts and an inspectable, descriptive delivery."""
from __future__ import annotations

import html
import json
from pathlib import Path
import statistics

import numpy as np

from .common import *
from .host import public_checks
from .oracle import evaluate


def deliver(run):
    run=Path(run).resolve();manifest=read_json(run/'MANIFEST.json');records=[];checks=[]
    def check(name,passed):checks.append({'name':name,'passed':bool(passed)})
    check('frozen_source_hashes',file_map(run/'source_snapshot')==manifest['snapshot_hashes'])
    check('public_packet_hashes',file_map(run/'packet/public')==manifest['public_packet_hashes'])
    check('private_packet_hashes',file_map(run/'packet/private')==manifest['private_packet_hashes'])
    check('qualification_binding',sha(run/manifest['qualification']['path'])==manifest['qualification']['sha256'])
    for p,h in manifest['source_bindings'].items():check('source:'+p,sha(ROOT/p)==h)
    cold=[]
    for slot in manifest['slots']:
        target=run/'attempts'/slot['id'];path=target/'RESULT.json'
        if not path.exists():
            records.append({**slot,'status':'UNFINISHED','first_pass':False,'final_pass':False,'elapsed_seconds':None,'native_request_count':0,'tool_calls':0,'observed_tokens':0,'error':'Allocated or pending without a terminal result; no resend.'})
            check('terminal_result:'+slot['id'],False);continue
        r=read_json(path);records.append(r)
        check('slot_identity:'+slot['id'],all(r[k]==slot[k] for k in slot))
        events=[json.loads(line) for line in (target/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        requests=[e for e in events if e['event']=='model_request'];tools=[e for e in events if e['event']=='tool_request']
        check('request_count:'+slot['id'],len(requests)==r['native_request_count']<=48)
        check('tool_count_cap:'+slot['id'],r['tool_calls']<=192)
        check('no_post_terminal_resend:'+slot['id'],r['no_post_terminal_model_resend'])
        for i,req in enumerate(requests):
            body=req['body'];gpt=slot['model']=='gpt-6.1-sol'
            check(f'wire_model:{slot["id"]}:{i}',body['model']==('gpt-6.1-sol' if gpt else 'deepseek-flash'))
            check(f'wire_high:{slot["id"]}:{i}',(body.get('reasoning') or {}).get('effort')=='high' if gpt else body.get('reasoning_effort')=='high' and body.get('thinking')=={'type':'enabled'})
        check('native_model_echo:'+slot['id'],all(m=='gpt-6.1-sol' for m in r['native_models']) if slot['model']=='gpt-6.1-sol' else all(m in manifest['routes']['deepseek-v4.1-flash']['accepted_native_echoes'] for m in r['native_models']))
        case=load_case(run/'packet',slot['case_id'],manifest['packet_index_sha256'])
        check('workspace_input_binding:'+slot['id'],file_map(run/'workspaces'/slot['id']/'inputs')==case.contract['input_sha256'])
        start=read_json(target/'START.json');check('workspace_context_binding:'+slot['id'],file_map(run/'workspaces'/slot['id']/'context')==start['context_hashes'])
        check('prompt_binding:'+slot['id'],sha(target/'PROMPT.txt')==start['prompt_sha256'])
        schema_names={s['name'] for s in read_json(target/'SCHEMAS.json')}
        check('nr_visibility:'+slot['id'],set(manifest['nr_tools']).issubset(schema_names) if slot['arm']=='NR_ON' else not (set(manifest['nr_tools'])&schema_names))
        verdicts=[]
        for s in r['submissions']:
            check('submission_hash:'+slot['id']+':'+str(s['number']),file_map(s['output'])==s['output_hashes'] and file_map(s['processes'])==s['process_hashes'])
            verdicts.append(evaluate(case,s['output'],s['processes']))
        check('cold_verdicts:'+slot['id'],[v['passed'] for v in verdicts]==[v['passed'] for v in r['verdicts']])
        expected=bool(verdicts) and verdicts[-1]['passed'] and r['clean_replay']['passed'] and r['external_fit_invariance']['passed'] and r['inputs_unchanged'] and r['context_unchanged'] and r['submissions'][-1]['elapsed_seconds']<=900
        check('final_acceptance:'+slot['id'],bool(expected)==r['final_pass'])
        # Recompute each already-executed clean result, without another execution.
        replay_path=target/'clean_replay/REPLAY.json'
        if replay_path.exists():
            clean=read_json(replay_path)
            if clean['run']['success']:
                v=evaluate(case,target/'clean_replay/workspace/output',target/'clean_replay/processes')
                check('cold_clean_replay:'+slot['id'],v['passed']==clean['passed'])
        nr_calls=[e for e in tools if e['name'] in manifest['nr_tools']]
        successful_nr=[e for e in events if e['event']=='tool_result' and e['name'] in manifest['nr_tools'] and e['result'].get('success')]
        r['nr_usage']={'calls':len(nr_calls),'successful_results':len(successful_nr),'by_tool':{name:sum(e['name']==name for e in nr_calls) for name in manifest['nr_tools']},
                      'section_cache_hits':sum(e['result'].get('cache_hit',False) for e in successful_nr if e['name']=='read_skill_section')}
        if case.id=='H03' and r['submissions']:
            try:
                labels=read_json(run/'packet/private/H03_EXTERNAL_LABELS.json');pred=read_csv(Path(r['submissions'][-1]['output'])/'predictions.csv')
                external={p['participant_id']:p for p in pred if p['split']=='external'}
                if set(external)==set(labels):
                    y=np.array([labels[s] for s in sorted(labels)]);p=np.array([int(external[s]['prediction']) for s in sorted(labels)])
                    from sklearn.metrics import roc_auc_score
                    r['host_external_metrics']={'n':len(y),'balanced_accuracy':float(np.mean([(p[y==k]==k).mean() for k in (0,1)])),
                        'roc_auc':float(roc_auc_score(y,[float(external[s]['score']) for s in sorted(labels)])),
                        'scope':'Host calculation after frozen submission; not model-visible feedback or an accuracy threshold for correctness.'}
            except (ValueError,KeyError,OSError):pass
        cold.append({'id':r['id'],'verdicts':verdicts})
    summary=[]
    for model in MODELS:
        for arm in ARMS:
            rows=[r for r in records if r['model']==model and r['arm']==arm];times=[r['elapsed_seconds'] for r in rows if r['elapsed_seconds'] is not None]
            summary.append({'model':model,'arm':arm,'tasks':len(rows),'first_pass':sum(r['first_pass'] for r in rows),'final_pass':sum(r['final_pass'] for r in rows),
                'median_attempt_seconds':statistics.median(times) if times else None,'total_tool_calls':sum(r['tool_calls'] for r in rows),
                'native_requests':sum(r['native_request_count'] for r in rows),'observed_tokens':sum(r['observed_tokens'] for r in rows),
                'unknown_or_unfinished':sum(r['status'] in ('UNKNOWN','UNFINISHED') for r in rows),'budget_stops':sum(r['status']=='BUDGET_STOP' for r in rows),
                'nr_calls':sum(r.get('nr_usage',{}).get('calls',0) for r in rows),'nr_cache_hits':sum(r.get('nr_usage',{}).get('section_cache_hits',0) for r in rows)})
    pairs=[]
    for model in MODELS:
        for case_id in CASE_IDS:
            a=next(r for r in records if r['model']==model and r['case_id']==case_id and r['arm']=='NR_OFF');b=next(r for r in records if r['model']==model and r['case_id']==case_id and r['arm']=='NR_ON')
            pairs.append({'model':model,'case_id':case_id,'off_pass':a['final_pass'],'on_pass':b['final_pass'],'pass_delta':int(b['final_pass'])-int(a['final_pass']),
                'off_seconds':a['elapsed_seconds'],'on_seconds':b['elapsed_seconds'],'off_status':a['status'],'on_status':b['status']})
    result={'revision':REVISION,'scope':manifest['scope'],'models':list(MODELS),'arms':list(ARMS),'summary':summary,'paired_tasks':pairs,'records':records,
        'limitations':['Authored development tasks over shared materials; eight cases in six related families, one attempt per condition. No population, independent holdout or superiority inference.',
            'OFF retains the same optional public library and common execution/evidence primitives; this estimates the additional NR interfaces.',
            'GPT uses the pre-existing XNova Responses route with provider-reported model/high echo. DeepSeek uses its official endpoint and accepted Flash alias; high requested, effort not independently echoed.',
            'Reference/offline qualification is engineering evidence, not independent scientific expert acceptance. Controlled data/layout/project perturbations are not new clinical acquisitions.',
            'Timeout, UNKNOWN, failed and unsubmitted attempts stay in the denominator. No post-terminal model resend. Timing includes reasoning/tools and is a single-run descriptive comparison.'],
        'model_requests':sum(r['native_request_count'] for r in records),'all_slots_terminal':all(r['status']!='UNFINISHED' for r in records)}
    delivery=run/'delivery';delivery.mkdir(exist_ok=False)
    write_json(delivery/'RESULTS.json',result);write_csv(delivery/'SUMMARY.csv',summary,list(summary[0]));write_csv(delivery/'PAIRED_TASKS.csv',pairs,list(pairs[0]))
    flat=[]
    for r in records:
        failures=r.get('verdicts',[{}])[-1].get('errors',[]) if r.get('verdicts') else [r.get('error','no submission')]
        flat.append({k:r[k] for k in ('id','case_id','model','arm','status','first_pass','final_pass','elapsed_seconds','native_request_count','tool_calls')}|{'failure_details':'; '.join(failures[:8]),'nr_calls':r.get('nr_usage',{}).get('calls',0)})
    write_csv(delivery/'TASK_RESULTS.csv',flat,list(flat[0]))
    def esc(value):return html.escape(str(value))
    rows=''.join('<tr>'+''.join(f'<td>{esc(r[k])}</td>' for k in ('model','arm','first_pass','final_pass','tasks','median_attempt_seconds','total_tool_calls','nr_calls'))+'</tr>' for r in summary)
    detail=''.join(f'<tr><td>{esc(r["case_id"])}</td><td>{esc(r["model"])}</td><td>{esc(r["arm"])}</td><td>{"PASS" if r["final_pass"] else "FAIL"}</td><td>{esc(r["status"])}</td><td>{esc(r["failure_details"])}</td><td><a href="../attempts/{r["id"]}/RESULT.json">原始结果</a></td></tr>' for r in flat)
    document=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>困难执行题开发对比</title><style>body{{font:15px system-ui;margin:32px;max-width:1500px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px;text-align:left}}p{{line-height:1.7}}</style><h1>8 道困难执行题：NR 开关对比</h1><p>GPT-6.1-Sol 与 DeepSeek V4.1 Flash，各 8 题 × NR OFF/ON，共 32 个固定槽位。每题 900 秒、48 轮、192 次工具调用、每次 Python 120 秒。原始失败、超时和未知状态保留。</p><p>质量计分要求科学产物、输入身份、真实运行和干净目录复现全部通过。两组共享数据、公开协议、可选方法库和执行接口，只比较同一 NR 的专用接口增量。此处是开发样例的一次描述性对比。</p><table><tr><th>模型</th><th>NR</th><th>首次提交通过</th><th>最终通过</th><th>分母</th><th>尝试耗时中位秒</th><th>工具调用总数</th><th>NR 调用</th></tr>{rows}</table><p><a href="RESULTS.json">完整结果</a> · <a href="SUMMARY.csv">汇总 CSV</a> · <a href="PAIRED_TASKS.csv">逐题配对</a> · <a href="TASK_RESULTS.csv">失败明细</a> · <a href="VERIFIED.json">冷读核验</a></p><table><tr><th>题目</th><th>模型</th><th>NR</th><th>科学交付</th><th>原始状态</th><th>诊断</th><th>证据</th></tr>{detail}</table><p>GPT 来自既有 XNova Responses 通道，原生响应验证模型名和 high；后端身份属于服务商报告。DeepSeek 使用官方接口并记录实际返回的 Flash 别名，显式请求 high。未进行模型补跑或自动恢复。</p></html>'''
    (delivery/'INDEX.html').write_text(document,encoding='utf-8')
    verification={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,'cold_verdicts':cold,
        'scope':'Cold result/source/request/submission binding and numerical readback; not a second model run or independent expert review.',
        'delivery_hashes':file_map(delivery),'total_slots':len(records),'native_requests':result['model_requests'],'at_utc':utc()}
    write_json(delivery/'VERIFIED.json',verification)
    write_json(run/'CURRENT_DELIVERY_ENTRYPOINTS.json',{'status':verification['status'],'index':'delivery/INDEX.html','results':'delivery/RESULTS.json','verification':'delivery/VERIFIED.json'})
    print(json.dumps({'delivery':str(delivery),'verification':verification['status'],'summary':summary,'failed_checks':[c['name'] for c in checks if not c['passed']]},ensure_ascii=False),flush=True)
    return result
