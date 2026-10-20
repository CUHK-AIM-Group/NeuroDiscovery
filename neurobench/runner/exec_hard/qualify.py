"""Offline references, clean replay, metaproperties and targeted wrong deliveries."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np

from .common import *
from .host import HardTools,prepare_workspace,replay,run_process
from .oracle import evaluate
from .reference import program


def mutate(case_id,output,variant):
    output=Path(output)
    if case_id=='H01':
        if variant==0:
            rows=read_csv(output/'cohort.csv');rows.append(rows[0]);write_csv(output/'cohort.csv',rows,list(rows[0]))
        elif variant==1:
            rows=read_csv(output/'cohort.csv');rows[0]['aal']=rows[1]['aal'];write_csv(output/'cohort.csv',rows,list(rows[0]))
        else:
            data=read_json(output/'audit.json');data['records'].pop();write_json(output/'audit.json',data)
    elif case_id in ('H02','H09'):
        folder='connectivity'
        path=sorted((output/folder).glob('*.npz'))[variant]
        with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        if variant==0:data['fc'][0,1]+=.2;data['fc'][1,0]+=.2
        elif variant==1:data['roi_ids']=np.roll(data['roi_ids'],1)
        else:data['valid']=~data['valid']
        np.savez(path,**data)
    elif case_id=='H11':
        rows=read_csv(output/'status.csv')
        if variant==0:
            for r in rows:r['status']='unsupported';r['artifact']='';r['reason']='entire batch blocked'
        elif variant==1:rows.pop()
        else:
            source=next(r for r in rows if r['status']=='complete')
            target=next(r for r in rows if r['status']=='unsupported')
            target['artifact']=f'outputs/{target["record_id"]}.npz';target['status']='complete'
            shutil.copy2(output/source['artifact'],output/target['artifact'])
        write_csv(output/'status.csv',rows,['record_id','participant_id','status','reason','artifact'])
    elif case_id=='H03':
        if variant==0:
            path=output/'model/state.npz'
            with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
            data['mean']+=.1;np.savez(path,**data)
        elif variant==1:
            rows=read_csv(output/'predictions.csv');rows[0]['score']=float(rows[0]['score'])+1;write_csv(output/'predictions.csv',rows,list(rows[0]))
        else:
            data=read_json(output/'evaluation.json');data['external_metrics']={'accuracy':1.0};write_json(output/'evaluation.json',data)
    elif case_id=='H05':
        if variant<2:
            rows=read_csv(output/'estimates.csv');key='beta' if variant==0 else 'q';rows[0][key]=float(rows[0][key])+.2;write_csv(output/'estimates.csv',rows,list(rows[0]))
        else:
            data=read_json(output/'design.json');data['n']+=1;write_json(output/'design.json',data)
    elif case_id=='H07':
        if variant==0:
            path=output/'images/labels.nii.gz';a=nib.load(path);data=np.asarray(a.dataobj).astype(float);data.flat[50000]+=.5;nib.save(nib.Nifti1Image(data,a.affine),path)
        elif variant==1:
            rows=read_csv(output/'measurements.csv');rows[0]['volume_mm3']=rows[0]['voxel_count'];write_csv(output/'measurements.csv',rows,list(rows[0]))
        else:
            path=output/'images/values.nii.gz';a=nib.load(path);nib.save(nib.Nifti1Image(np.asarray(a.dataobj)[::-1].copy(),a.affine),path)
    elif case_id=='H10':
        if variant==0:
            path=sorted((output/'results').glob('*_metric.json'))[0];data=read_json(path);data['mean_fc']+=.3;write_json(path,data)
        elif variant==1:
            data=read_json(output/'dependencies.json');name=next(k for k in data if k.startswith('metric:'));data[name]['inputs']=[];write_json(output/'dependencies.json',data)
        else:
            data=read_json(output/'execution.json');data['steps'][0]['action']='reuse' if data['steps'][0]['action']=='execute' else 'execute';write_json(output/'execution.json',data)
    else:raise ValueError(case_id)


def qualify(packet,*,name='qualification',only=None,previous=None):
    packet=Path(packet).resolve();root=packet.parent/name;root.mkdir(exist_ok=False)
    index_sha=sha(packet/'INDEX.json');library=packet/'public/library';results=[];checks=[]
    selected=tuple(only or CASE_IDS)
    if previous:
        prior=read_json(previous)
        current=file_map(packet/'public');before=prior['packet_public_hashes']
        changed={p for p in set(current)|set(before) if current.get(p)!=before.get(p)}
        if any(not any(p.startswith('cases/'+c+'/') for c in selected) for p in changed):raise ValueError('retained_case_or_library_changed')
        retained=set(CASE_IDS)-set(selected)
        results=[r for r in prior['reference_runs'] if r['case_id'] in retained]
        checks=[c for c in prior['checks'] if c['name'].split(':')[0] in retained]
        if not all(r['verdict']['passed'] for r in results):raise ValueError('cannot_reuse_failed_reference')
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)})
    for case_id in selected:
        case=load_case(packet,case_id,index_sha);outputs=[]
        for variant in (0,1):
            target=root/case_id/f'valid_{variant}';work=prepare_workspace(case,target/'workspace',library)
            (work/'output/project').mkdir();(work/'output/project/main.py').write_text(program(case_id,bool(variant)),encoding='utf-8')
            start=time.monotonic();run=run_process(work,target/'processes')
            if run['success']:finalize(case,work/'output')
            verdict=evaluate(case,work/'output',target/'processes')
            record={'case_id':case_id,'variant':variant,'run':run,'verdict':verdict}
            write_json(target/'RESULT.json',record);results.append(record)
            check(f'{case_id}:valid_{variant}',run['success'] and verdict['passed'])
            print(json.dumps({'qualification':case_id,'variant':variant,'passed':verdict['passed'],'seconds':run['elapsed_seconds'],'errors':verdict['errors'][:4]},ensure_ascii=False),flush=True)
            outputs.append((work/'output',target/'processes'))
        if not all(r['verdict']['passed'] for r in results if r['case_id']==case_id):continue
        output,processes=outputs[0]
        clean=replay(case,output,root/case_id/'clean_replay',library)
        check(case_id+':clean_replay',clean['passed'])
        if case_id=='H03':
            changed=replay(case,output,root/case_id/'external_values_replay',library,external_perturbation=True)
            check('H03:external_fit_invariance',changed['passed'])
        for variant in range(3):
            target=root/case_id/f'wrong_{variant}';shutil.copytree(output,target/'output')
            mutate(case_id,target/'output',variant);finalize(case,target/'output')
            verdict=evaluate(case,target/'output',processes)
            write_json(target/'VERDICT.json',verdict);check(f'{case_id}:wrong_{variant}_rejected',not verdict['passed'])
    # Actual common guard behavior and real production NR endpoints, without inference.
    case=load_case(packet,'H07',index_sha)
    for arm in ARMS:
        target=root/'host_checks'/arm;work=prepare_workspace(case,target/'workspace',library);target.mkdir(exist_ok=True)
        tools=HardTools(case,work,target,arm,time.monotonic()+900,time.monotonic())
        check(arm+':last_round',tools.begin_round(48)['model_round']==48)
        tools.tool_count=191;check(arm+':last_tool',tools.call('read_file',{'path':'inputs/protocol.json'})['success'])
        check(arm+':over_tool_limit',not tools.call('read_file',{'path':'inputs/protocol.json'})['success'])
        write_json(target/'SCHEMAS.json',tools.schemas())
    # Same common schemas; ON contains exactly the four production NR tools.
    a=read_json(root/'host_checks/NR_OFF/SCHEMAS.json');b=read_json(root/'host_checks/NR_ON/SCHEMAS.json')
    check('common_schemas_identical',a==[s for s in b if s['name'] in {s['name'] for s in a}])
    check('on_exact_production_tools',{s['name'] for s in b}-{s['name'] for s in a}=={'compile_task','read_skill_section','search_skills','read_skill'})
    # Exercise the real production registry, bounded reads and persistent cache.
    tools.tool_count=0
    compiled=tools.call('compile_task',{'input_paths':['inputs/reference.nii.gz'],'public_contract':{'title':'reference space region measurements'}})
    search=tools.call('search_skills',{'query':'functional connectivity','limit':2})
    write_json(root/'host_checks/PRODUCTION_TOOLS.json',{'compiled':compiled,'search':search})
    check('production_compile',compiled.get('success') and 'inputs/reference.nii.gz' in compiled.get('input_metadata',{}))
    check('production_search',search.get('success'))
    rows=search.get('skills') or search.get('results') or search.get('matches') or []
    if rows:
        name=rows[0]['name'];full=tools.call('read_skill',{'name':name,'max_chars':1000})
        first=tools.call('read_skill_section',{'name':name,'query':'connectivity'});second=tools.call('read_skill_section',{'name':name,'query':'connectivity'})
        write_json(root/'host_checks/PRODUCTION_READS.json',{'full':full,'first':first,'second':second})
        check('production_full_read',full.get('success'))
        check('production_section_cache',first.get('success') and second.get('success') and not first.get('cache_hit') and second.get('cache_hit') and first.get('content')==second.get('content'))
    else:check('production_search_results',False)
    # Source, every public file, and private hidden-label custody are rechecked.
    provenance=read_json(packet/'private/PROVENANCE.json')
    check('all_original_source_hashes_unchanged',all(sha(ROOT/p)==h for p,h in provenance['source_bindings'].items()))
    check('packet_inputs_unchanged',all(load_case(packet,c,index_sha).id==c for c in CASE_IDS))
    check('external_labels_not_public',not any(p.name=='H03_EXTERNAL_LABELS.json' for p in (packet/'public').rglob('*')))
    record={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,
        'reference_runs':[{k:r[k] for k in ('case_id','variant','run','verdict')} for r in results],
        'calibration':{'max_reference_seconds':max((r['run']['elapsed_seconds'] for r in results),default=0),
            'max_sampled_peak_rss_bytes':max((r['run']['peak_rss_bytes_sampled'] for r in results),default=0)},
        'scope':'Offline development engineering qualification; two valid implementations/order variants, targeted wrong deliveries, clean replay and held-out-feature invariance. Not clinical/expert acceptance or demonstrated NR benefit.',
        'model_requests':0,'packet_index_sha256':index_sha,'packet_public_hashes':file_map(packet/'public')}
    if previous:record['retained_qualified_runs']={'source':str(previous),'sha256':sha(previous),'cases':sorted(set(CASE_IDS)-set(selected)),'changed_public_paths':sorted(changed),'reason':'Only selected cases or their preparation programs changed. Every retained case input, contract and task file plus the entire common library matches its previous qualification; all previous executions and failures remain.'}
    write_json(root/'QUALIFICATION.json',record)
    print(json.dumps({'status':record['status'],'failed':[c['name'] for c in checks if not c['passed']],'calibration':record['calibration']},ensure_ascii=False),flush=True)
    return record


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('packet');parser.add_argument('--name',default='qualification');parser.add_argument('--only',nargs='+');parser.add_argument('--previous');args=parser.parse_args()
    raise SystemExit(0 if qualify(args.packet,name=args.name,only=args.only,previous=args.previous)['status']=='PASS' else 1)
