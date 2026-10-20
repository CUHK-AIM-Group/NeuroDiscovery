"""Qualify multiple valid methods, representation equivalents and real errors."""
from __future__ import annotations

import itertools
import shutil
import time
import numpy as np
from neurobench.runner.exec_hard.host import prepare_workspace,run_process,HardTools
from .common import *
from .reference import write_reference
from .oracle import evaluate,inference_expected
from .execution import replay


def mutate(case_id,output,variant):
    output=Path(output)
    if case_id=='O01':
        if variant==0:
            rows=read_csv(output/'predictions.csv');rows[0]['score']=1.2;write_csv(output/'predictions.csv',rows,list(rows[0]))
        elif variant==1:
            rows=read_csv(output/'validation.csv');rows[0]['fold']='E';write_csv(output/'validation.csv',rows,list(rows[0]))
        elif variant==2:
            data=read_json(output/'metrics.json');data['external_metrics']={'balanced_accuracy':1};write_json(output/'metrics.json',data)
        else:
            rows=read_csv(output/'predictions.csv');rows.append(rows[0]);write_csv(output/'predictions.csv',rows,list(rows[0]))
    elif case_id=='O02':
        if variant==3:
            m=read_json(output/'method.json');m['claim_type']='causation';write_json(output/'method.json',m)
        else:
            rows=read_csv(output/'estimates.csv');key=('beta','se','q')[variant];rows[0][key]=float(rows[0][key])+.2;write_csv(output/'estimates.csv',rows,list(rows[0]))
    elif case_id=='O03':
        if variant==0:
            rows=read_csv(output/'status.csv');next(r for r in rows if r['record_id']=='I3')['status']='complete';write_csv(output/'status.csv',rows,list(rows[0]))
        else:
            rows=read_csv(output/'measurements.csv')
            if variant==1:rows[0]['volume_mm3']=rows[0]['voxel_count']
            elif variant==2:rows[0]['mean_value']=float(rows[0]['mean_value'])+.2
            else:rows.pop()
            write_csv(output/'measurements.csv',rows,list(rows[0]))
    elif case_id=='O04':
        if variant==0:
            path=next((output/'connectivity').glob('*.npz'))
            with np.load(path,allow_pickle=False) as d:v={k:d[k] for k in d.files}
            v['roi_ids']=np.roll(v['roi_ids'],1);np.savez(path,**v)
        elif variant==1:
            path=next((output/'connectivity').glob('*.npz'))
            with np.load(path,allow_pickle=False) as d:v={k:d[k] for k in d.files}
            v['fc'][0,1]+=.2;np.savez(path,**v)
        else:
            rows=read_csv(output/'summary.csv')
            if variant==2:rows[0]['n_frames']=int(rows[0]['n_frames'])+1
            else:rows.pop()
            write_csv(output/'summary.csv',rows,list(rows[0]))


def qualify(packet,name='qualification',previous=None):
    packet=Path(packet).resolve();root=packet.parent/name;root.mkdir(parents=True,exist_ok=False)
    library=packet/'public/library';checks=[];runs=[];selected=CASE_IDS;prior_binding=None
    if previous:
        previous=Path(previous).resolve();prior=read_json(previous)
        if prior['status']!='PASS' or prior['packet_index_sha256']!=sha(packet/'INDEX.json') or prior['packet_public_hashes']!=file_map(packet/'public') or prior['packet_private_hashes']!=file_map(packet/'private'):raise ValueError('prior_qualification_binding_failed')
        checks=list(prior['checks']);runs=list(prior['reference_runs']);selected=()
        prior_binding={'path':str(previous),'sha256':sha(previous),'retained':'Unchanged valid implementations, equivalent representations, wrong outputs and their private original/clean evidence; new invalid-learner and four-tool checks added before any native dispatch.'}
    def check(name,value,**more):checks.append({'name':name,'passed':bool(value),**more})
    for cid in selected:
        case=load_case(packet,cid)
        for variant in range(2):
            target=root/f'{cid}_valid_{variant}';work=prepare_workspace(case,target/'workspace',library);write_reference(case,work/'output',variant)
            run=run_process(work,target/'processes');verdict={'passed':False,'errors':['reference_process_failed']}
            clean={'passed':False}
            if run['success']:
                finalize(case,work/'output');verdict=evaluate(case,work/'output',target/'processes')
                clean=replay(case,work/'output',target/'clean',library)
            check(f'{cid}:valid:{variant}',run['success'] and verdict['passed'] and clean['passed'])
            record={'case_id':cid,'variant':variant,'run':run,'evaluation':verdict,'clean_replay':clean};write_json(target/'REFERENCE.json',record);runs.append(record)
            if not verdict['passed']:continue
            if variant==0:
                for j in range(4):
                    bad=root/f'{cid}_wrong_{j}';shutil.copytree(work/'output',bad/'output');mutate(cid,bad/'output',j);finalize(case,bad/'output')
                    result=evaluate(case,bad/'output',target/'processes');write_json(bad/'VERDICT.json',result);check(f'{cid}:wrong:{j}',not result['passed'])
            equivalent=root/f'{cid}_equivalent_{variant}';shutil.copytree(work/'output',equivalent/'output')
            for path in (equivalent/'output').glob('*.csv'):
                rows=read_csv(path)
                for row in rows:
                    for k in ('region_id','voxel_count','n_frames','n_valid_rois','prediction'):
                        if k in row:row[k]=str(float(row[k]))
                write_csv(path,rows[::-1],list(rows[0]))
            for path in (equivalent/'output').glob('*.json'):
                if path.name in ('method.json','metrics.json'):
                    data=read_json(path);data['descriptive_metadata']={'note':'valid alternative representation'};write_json(path,data)
            finalize(case,equivalent/'output');result=evaluate(case,equivalent/'output',target/'processes');write_json(equivalent/'VERDICT.json',result);check(f'{cid}:equivalent:{variant}',result['passed'])
    # Bad learners can produce internally consistent CSVs; independent calls must reject them.
    case=load_case(packet,'O01')
    invalid_learners={
        'reads_other_training_rows':'''def fit_predict(train,test):
    import pandas as pd
    from pathlib import Path
    leaked=pd.read_csv(Path.cwd()/'inputs/train.csv')
    return {str(k):float(leaked.label.mean()) for k in test.participant_id.unique()}
''',
        'fits_test_distribution':'''def fit_predict(train,test):
    import numpy as np
    x=test.groupby('participant_id').f00.mean()
    p=1/(1+np.exp(-(x-x.mean())/(x.std(ddof=0)+1e-6)))
    return {str(k):float(v) for k,v in p.items()}
''',
    }
    for label,source in invalid_learners.items():
        target=root/('O01_invalid_learner_'+label);work=prepare_workspace(case,target/'workspace',library);write_reference(case,work/'output')
        (work/'output/project/learner.py').write_text(source,encoding='utf-8');run=run_process(work,target/'processes')
        if run['success']:
            finalize(case,work/'output');basic=evaluate(case,work/'output',target/'processes');clean=replay(case,work/'output',target/'clean',library)
        else:basic={'passed':False};clean={'passed':True}
        write_json(target/'REJECTED.json',{'run':run,'basic':basic,'clean':clean})
        check('O01:reject_invalid_learner:'+label,run['success'] and basic['passed'] and not clean['passed'])
    # Every disclosed O02 choice is checked against an independent matrix implementation.
    case=load_case(packet,'O02')
    from statsmodels.stats.multitest import multipletests
    for unit,inference,adjustment in itertools.product(('participant_mean','records'),('normal','t'),('fdr_bh','holm','bonferroni')):
        method={'unit':unit,'covariance':'HC3' if unit=='participant_mean' else 'cluster','inference':inference,'adjustment':adjustment,'claim_type':'association'}
        expected=inference_expected(case.root/'inputs',method);p=[r['p'] for r in expected];q=multipletests(p,method=adjustment)[1]
        check('O02:all_choices:'+unit+':'+inference+':'+adjustment,arrays_close(q,[r['q'] for r in expected]))
    # The optional library and common tools remain identical; only four NR schemas differ.
    case=load_case(packet,'O03');base=root/'schema_probe';work=prepare_workspace(case,base/'workspace',library)
    (base/'off').mkdir();(base/'on').mkdir()
    started=time.monotonic();off=HardTools(case,work,base/'off','NR_OFF',started+900,started);on=HardTools(case,work,base/'on','NR_ON',started+900,started)
    off_names={s['name'] for s in off.schemas()};on_names={s['name'] for s in on.schemas()}
    check('shared_interface_and_nr_only_delta',on_names-off_names==set(['search_skills','read_skill','compile_task','read_skill_section']) and off_names<=on_names)
    # Smoke actual NR search/load/compile/section handlers, without exposing private data.
    search=on.call('search_skills',{'query':'statistical machine learning','limit':3})
    check('actual_nr_search',search.get('success') is not False)
    for name,args in [('read_skill',{'name':'statistical-ml'}),('compile_task',{'input_paths':['inputs/records.csv']}),('read_skill_section',{'name':'statistical-ml','query':'validation','max_chars':2500})]:
        answer=on.call(name,args)
        check('actual_nr_'+name,answer.get('success') is not False and not answer.get('error'))
    check('off_does_not_expose_nr',off.call('search_skills',{'query':'statistics'}).get('success') is False)
    record={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','revision':REVISION,'created_at_utc':utc(),'checks':checks,'reference_runs':runs,
        'packet_index_sha256':sha(packet/'INDEX.json'),'packet_public_hashes':file_map(packet/'public'),'packet_private_hashes':file_map(packet/'private'),
        'calibration':{'max_reference_seconds':max(r['run']['elapsed_seconds'] for r in runs),'max_reference_peak_rss_bytes_sampled':max(r['run']['peak_rss_bytes_sampled'] for r in runs),'valid_reference_runs':len(runs),'wrong_deliveries':16,'invalid_learners':len(invalid_learners),'equivalent_deliveries':8},
        'prior_qualification':prior_binding,'model_requests':0,'expert_scientific_acceptance':False}
    write_json(root/'QUALIFICATION.json',record)
    print(json.dumps({'status':record['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['passed']],'calibration':record['calibration']},ensure_ascii=False),flush=True)
    return record
