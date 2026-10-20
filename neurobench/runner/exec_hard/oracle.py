"""Private scientific checks, independent library calculations and host evidence.

Checks never enter model prompts or tool results. A scientific failure is terminal.
"""
from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np

from neurobench.runner.exec_protocol_v2 import public_checks
from .common import *
from .host import trace_events
from .reference import namespace
from .semantic import participant_id,participant_ids,qc_records,covariates_equivalent,external_metadata_valid,step_records,dependency_records,default_group_config


def evaluate(case,output,processes,*,require_execution=True):
    output=Path(output);inp=case.root/'inputs';errors=[];checks=[];scope=namespace()
    def check(name,valid):
        checks.append(name)
        if not bool(valid):errors.append(name)
    def numeric(name,a,b,atol=5e-6,rtol=5e-5):check(name,arrays_close(a,b,atol,rtol))
    def rows_by(rows,key):
        result={key(r):r for r in rows}
        if len(result)!=len(rows):raise ValueError('duplicate_output_identity')
        return result
    def csv_compare(name,expected,keys,numerics):
        rows=read_csv(output/name);a=rows_by(rows,lambda r:tuple(str(r[k]) for k in keys));b=rows_by(expected,lambda r:tuple(str(r[k]) for k in keys))
        check(name+':identity_set',set(a)==set(b))
        for key in set(a)&set(b):
            for field in numerics:numeric(name+':'+str(key)+':'+field,float(a[key][field]),float(b[key][field]))
        return a
    def fc_check(name,expected):
        with np.load(contained(output,name),allow_pickle=False) as data:
            ids=[str(v) for v in data['roi_ids']];expected_ids=[str(v) for v in expected[1]]
            check(name+':roi_identity',len(ids)==len(set(ids)) and set(ids)==set(expected_ids))
            if set(ids)!=set(expected_ids):return
            order=[ids.index(v) for v in expected_ids]
            numeric(name+':fc',data['fc'][np.ix_(order,order)],expected[0])
            check(name+':valid',np.array_equal(data['valid'][order].astype(bool),expected[2]))
    try:
        public=public_checks(case.contract,output)
        check('public_artifacts',public['public_check_pass'])
        manifest=read_json(output/'result.json');check('supported_task_delivered',manifest.get('status')=='complete')
        events=trace_events(Path(processes)/'trusted')
        successful={e['run_id'] for e in events if e['type']=='process_end' and e.get('success')}
        starts=[e for e in events if e['type']=='process_start' and e['run_id'] in successful and e.get('kind')=='project']
        if require_execution:
            check('actual_final_entrypoint_execution',bool(starts) and starts[-1]['entrypoint_sha256']==sha(output/'project/main.py'))
        check('managed_fit_budget',sum(e['type']=='managed_fit' for e in events)<=64)
        if case.id=='H01':
            cohort,audit=scope['cohort_solution'](inp,True)
            found=csv_compare('cohort.csv',cohort,['participant_id'],['label','age','sex','mean_fd'])
            for row in cohort:
                item=found.get((row['participant_id'],),{})
                check('cohort_paths:'+row['participant_id'],all(item.get(k)==row[k] for k in ('site','aal','cc200')))
            observed=read_json(output/'audit.json')['records']
            a=rows_by(observed,lambda r:(r['source'],r['record_id']));b=rows_by(audit,lambda r:(r['source'],r['record_id']))
            check('audit_record_conservation',set(a)==set(b))
            for key in set(a)&set(b):check('audit:'+str(key),participant_id(a[key]['participant_id'])==b[key]['participant_id'] and a[key]['status']==b[key]['status'] and bool(a[key].get('reason')))
        elif case.id=='H02':
            comparison=[];invalid=[]
            for row in read_csv(inp/'cohort.csv'):
                sid=row['participant_id']
                for atlas in ('aal','cc200'):
                    path=inp/f'series/{sid}_{atlas}.npy';values=scope['connectivity'](path,path.with_suffix('.json'),True)
                    fc_check(f'connectivity/{sid}_{atlas}.npz',values)
                    comparison.append(dict(participant_id=sid,atlas=atlas,**scope['network_metrics'](values,True)))
                    invalid.append({'participant_id':sid,'atlas':atlas,'invalid_roi_ids':values[1][~values[2]].tolist()})
            csv_compare('comparison.csv',comparison,['participant_id','atlas'],['valid_rois','n_edges','mean_fc','global_transitivity'])
            observed=rows_by(qc_records(read_json(output/'qc.json')),lambda r:(participant_id(r['participant_id']),r['atlas']))
            check('quality_record_conservation',set(observed)=={(r['participant_id'],r['atlas']) for r in invalid})
            for r in invalid:check('invalid_rois:'+r['participant_id']+r['atlas'],set(map(str,observed[r['participant_id'],r['atlas']]['invalid_roi_ids']))==set(r['invalid_roi_ids']))
        elif case.id=='H03':
            rows,ids,x,y,dev,best,state,pred,candidates,folds=scope['prediction_solution'](inp,True)
            selected=read_json(output/'evaluation.json');family=selected['family'];hp=float(selected['hyperparameter'])
            candidate=next(c for c in candidates if c['family']==family and c['hyperparameter']==hp)
            check('validation_based_selection',candidate['mean_site_balanced_accuracy']>=best['mean_site_balanced_accuracy']-1e-12)
            numeric('visible_validation_metric',selected['mean_site_balanced_accuracy'],candidate['mean_site_balanced_accuracy'])
            check('external_label_custody',external_metadata_valid(selected,rows))
            state=scope['fitted'](x[dev],y[dev],family,hp,True)
            expected_score=scope['scores'](state,x);sites=np.array([r['site'] for r in rows])
            for site in sorted(set(sites[dev])):
                fit=dev&(sites!=site);val=dev&(sites==site)
                expected_score[val]=scope['scores'](scope['fitted'](x[fit],y[fit],family,hp,True),x[val])
            observed=rows_by(read_csv(output/'predictions.csv'),lambda r:r['participant_id'])
            check('prediction_identity_set',set(observed)=={r['participant_id'] for r in rows})
            for i,row in enumerate(rows):
                r=observed[row['participant_id']];check('prediction_split:'+row['participant_id'],r['split']==row['split'])
                numeric('prediction_score:'+row['participant_id'],float(r['score']),expected_score[i],5e-4,2e-3)
                check('prediction_decision:'+row['participant_id'],int(r['prediction'])==int(expected_score[i]>=0))
            with np.load(output/'model/state.npz',allow_pickle=False) as model:
                model_ids=list(map(str,model['feature_ids']));check('model_feature_identity',len(model_ids)==len(set(model_ids)) and set(model_ids)==set(ids))
                order=[model_ids.index(k) for k in ids]
                for key,expected in zip(('mean','scale','coef'),state[:3]):numeric('model_'+key,model[key][order],expected,5e-4 if key=='coef' else 5e-6,2e-3 if key=='coef' else 5e-5)
                numeric('model_intercept',model['intercept'],state[3],5e-4,2e-3)
            evidence=read_json(output/'fit_evidence.json')
            check('final_fit_membership',participant_ids(evidence['final_fit_participant_ids'])=={r['participant_id'] for i,r in enumerate(rows) if dev[i]} and len(evidence['final_fit_participant_ids'])==sum(dev))
            observed_folds=rows_by(evidence['validation_folds'],lambda r:r['site'])
            check('fold_site_set',set(observed_folds)=={r['site'] for r in folds})
            for fold in folds:
                for k in ('fit_ids','validation_ids'):check('fold:'+fold['site']+k,participant_ids(observed_folds[fold['site']][k])==set(fold[k]) and len(observed_folds[fold['site']][k])==len(fold[k]))
            actual=rows_by(evidence['candidate_scores'],lambda r:(r['family'],float(r['hyperparameter'])))
            check('candidate_search_conservation',set(actual)=={(r['family'],r['hyperparameter']) for r in candidates})
            for c in candidates:numeric('candidate:'+str((c['family'],c['hyperparameter'])),actual[c['family'],c['hyperparameter']]['mean_site_balanced_accuracy'],c['mean_site_balanced_accuracy'])
        elif case.id=='H05':
            included,columns,x,audit,values=scope['inference_solution'](inp,True)
            csv_compare('estimates.csv',values,['feature_id'],['beta','se','ci_low','ci_high','p','q'])
            design=read_json(output/'design.json')
            check('analysis_sample',participant_ids(design['participant_ids'])=={r['participant_id'] for r in included} and len(design['participant_ids'])==len(included))
            check('design_identity',covariates_equivalent(design['columns'],columns) and design['rank']==int(np.linalg.matrix_rank(x)) and design['n']==len(x))
            observed=rows_by(read_csv(output/'sample_audit.csv'),lambda r:r['participant_id'])
            check('sample_audit_conservation',set(observed)=={r['participant_id'] for r in audit})
            for r in audit:check('sample_status:'+r['participant_id'],observed[r['participant_id']]['status']==r['status'] and bool(observed[r['participant_id']].get('reason')))
            report=read_json(output/'report.json')
            check('inference_report',report['n']==len(x) and report['feature_count']==len(values) and set(report['significant_feature_ids'])=={r['feature_id'] for r in values if r['q']<.05})
        elif case.id=='H07':
            ref,labels,values=scope['aligned_images'](inp,True)
            for name,expected in (('labels',labels),('values',values)):
                image=nib.load(output/f'images/{name}.nii.gz');numeric('grid_affine:'+name,image.affine,ref.affine)
                data=np.asarray(image.dataobj);numeric('aligned_data:'+name,data,expected)
                if name=='labels':check('categorical_labels',np.equal(data,np.floor(data)).all() and set(np.unique(data)).issubset(set(np.unique(labels))))
            volume=abs(np.linalg.det(ref.affine[:3,:3]))
            expected=[{'roi_id':int(k),'voxel_count':int((labels==k).sum()),'volume_mm3':float((labels==k).sum()*volume),'mean_value':float(values[labels==k].mean())} for k in np.unique(labels) if k>0]
            csv_compare('measurements.csv',expected,['roi_id'],['voxel_count','volume_mm3','mean_value'])
            audit=read_json(output/'spatial_audit.json');numeric('audit_affine',audit['reference_affine'],ref.affine);numeric('audit_voxel_volume',audit['voxel_volume_mm3'],volume)
            check('audit_space_units',audit['reference_shape']==list(ref.shape) and audit['space']=='MNI' and audit['units']=='mm')
        elif case.id=='H09':
            groups={};ids=[]
            for r in read_csv(inp/'cohort.csv'):
                sid=r['participant_id'];ids.append(sid);p=inp/f'series/{sid}_aal.npy';values=scope['connectivity'](p,p.with_suffix('.json'),True)
                fc_check(f'connectivity/{sid}.npz',values);groups.setdefault((r['site'],int(r['label'])),[]).append(scope['fc_mean'](values))
            expected=[dict(site=k[0],label=k[1],n=len(v),mean_fc=np.mean(v)) for k,v in groups.items()]
            csv_compare('summary.csv',expected,['site','label'],['n','mean_fc'])
            record=read_json(output/'execution.json');check('project_analysis_identity',participant_ids(record['analyzed_participant_ids'])==set(ids) and len(record['analyzed_participant_ids'])==len(ids))
        elif case.id=='H10':
            ids=sorted(r['participant_id'] for r in read_csv(inp/'cohort.csv'));config=read_json(inp/'config.json');prior=read_json(inp/'prior/records.json')['steps']
            metrics=[];expected_actions={};specs=scope['cache_specs'](inp,output,ids,config)
            for sid in ids:
                path=inp/f'series/{sid}_aal.npy';values=scope['connectivity'](path,path.with_suffix('.json'),True)
                fc_check(f'results/{sid}_fc.npz',values)
                a=values[0][np.ix_(values[2],values[2])];upper=a[np.triu_indices(len(a),1)]
                metric={'mean_fc':float(upper.mean()),'above_threshold':int((upper>config[sid]['threshold']).sum())};metrics.append(metric)
                actual=read_json(output/f'results/{sid}_metric.json')
                numeric('metric_mean:'+sid,actual['mean_fc'],metric['mean_fc']);check('metric_count:'+sid,actual['above_threshold']==metric['above_threshold'])
                current={f'inputs/series/{sid}_aal.npy':sha(path),f'inputs/series/{sid}_aal.json':sha(path.with_suffix('.json'))}
                changed=current!=prior['fc:'+sid]['input_sha256'];expected_actions['fc:'+sid]='execute' if changed else 'reuse'
                metric_changed=changed or config[sid]!=prior['metric:'+sid]['config'];expected_actions['metric:'+sid]='execute' if metric_changed else 'reuse'
            expected_actions['group']='execute' if any(a=='execute' for a in expected_actions.values()) else 'reuse'
            group=read_json(output/'results/group.json');check('group_n',group['n']==len(ids));numeric('group_mean',group['mean_fc'],np.mean([r['mean_fc'] for r in metrics]));check('group_threshold_count',group['above_threshold_sum']==sum(r['above_threshold'] for r in metrics))
            deps=dependency_records(read_json(output/'dependencies.json'));check('dependency_step_set',set(deps)==set(expected_actions))
            last=starts[-1]['run_id'] if starts else None;steps=[e for e in events if e['type']=='step' and e['run_id']==last]
            actual_steps=rows_by(steps,lambda r:r['name']);check('trusted_step_conservation',set(actual_steps)==set(expected_actions))
            declared=rows_by(step_records(read_json(output/'execution.json')),lambda r:r['name']);check('declared_step_conservation',set(declared)==set(expected_actions))
            for name,inputs,outputs,settings,_ in specs:
                record=actual_steps[name]
                check('minimal_valid_action:'+name,record['success'] and record['action']==expected_actions[name] and declared[name]['action']==record['action'])
                check('step_dependency_identity:'+name,set(deps[name]['inputs'])==set(inputs) and set(deps[name]['outputs'])==set(outputs) and deps[name]['config']==settings)
                expected_inputs={p:sha(inp/p.removeprefix('inputs/')) if p.startswith('inputs/') else sha(output/p.removeprefix('output/')) for p in inputs}
                config_equivalent=record['config']==settings or name=='group' and settings=={} and default_group_config(record['config'])
                check('trusted_input_binding:'+name,record['input_sha256']==expected_inputs and config_equivalent)
                check('trusted_output_binding:'+name,record['output_sha256']=={p:sha(output/p.removeprefix('output/')) for p in outputs})
                if expected_actions[name]=='reuse':check('prior_artifact_preserved:'+name,all(sha(output/p.removeprefix('output/'))==prior[name]['output_sha256'][p] for p in outputs))
        elif case.id=='H11':
            records=read_csv(inp/'records.csv');observed=rows_by(read_csv(output/'status.csv'),lambda r:r['record_id'])
            check('batch_record_conservation',set(observed)=={r['record_id'] for r in records})
            counts={'total':len(records),'complete':0,'unsupported':0,'uncertain':0}
            for row in records:
                status,reason=scope['record_status'](inp,row);r=observed[row['record_id']];counts[status]+=1
                check('record_status:'+row['record_id'],r['participant_id']==row['participant_id'] and r['status']==status and bool(r.get('reason')))
                if status=='complete':
                    path=inp/row['path'];fc_check(r['artifact'],scope['connectivity'](path,path.with_suffix('.json'),True))
                else:check('unsupported_not_fabricated:'+row['record_id'],not r['artifact'] and not (output/f'outputs/{row["record_id"]}.npz').exists())
            summary=read_json(output/'summary.json')
            check('batch_totals',all(summary.get(k)==v for k,v in counts.items()))
        else:raise ValueError('unknown_case')
    except Exception as e:
        errors.append('oracle_input_or_output_error:'+type(e).__name__+':'+str(e)[:300])
    return {'case_id':case.id,'verdict':'PASS' if not errors else 'FAIL','passed':not errors,'checks_evaluated':len(checks),'errors':errors,
        'scope':'Public protocol scientific endpoints plus actual artifacts and privately retained execution evidence; no clinical efficacy or expert acceptance claim.'}
