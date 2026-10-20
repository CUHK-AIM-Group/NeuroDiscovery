"""Private numerical checks, independently expressed from preparation programs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from nibabel.processing import resample_from_to
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from neurobench.runner.exec_hard.host import trace_events
from neurobench.runner.exec_protocol_v2 import public_checks
from .common import *


def keyed(rows, keys):
    result={}
    for row in rows:
        key=tuple(str(row[k]) for k in keys)
        if key in result:raise ValueError('duplicate_identity:'+str(key))
        result[key]=row
    return result


def integer(value):
    v=float(value)
    if not np.isfinite(v) or v!=int(v):raise ValueError('non_integer_identity_or_count')
    return int(v)


def inference_expected(inp, method):
    unit=method['unit'];reference=method['inference'];adjustment=method['adjustment']
    if unit not in ('participant_mean','records') or reference not in ('normal','t') or adjustment not in ('fdr_bh','holm','bonferroni'):raise ValueError('unpublished_analysis_choice')
    if method['covariance']!=('HC3' if unit=='participant_mean' else 'cluster') or method['claim_type']!='association':raise ValueError('incompatible_covariance_or_claim')
    data=pd.read_csv(inp/'observations.csv',dtype={'participant_id':str})
    if unit=='participant_mean':
        numeric=['label','age','sex','mean_fd']+[f'm{i}' for i in range(6)]
        data=data.groupby('participant_id',sort=True).agg({**{k:'mean' for k in numeric},'site':'first'}).reset_index()
    sites=sorted(data.site.unique());x=np.column_stack([np.ones(len(data)),data[['label','age','sex','mean_fd']].to_numpy(float)]+[(data.site==s).to_numpy(float) for s in sites[1:]])
    n,k=x.shape;bread=np.linalg.inv(x.T@x);projection=x@bread;groups=data.participant_id.to_numpy();g=len(set(groups));rows=[]
    for marker in [f'm{i}' for i in range(6)]:
        y=data[marker].to_numpy(float);coeff=np.linalg.lstsq(x,y,rcond=None)[0];resid=y-x@coeff
        if unit=='participant_mean':
            leverage=np.sum(projection*x,axis=1);influence=x*(resid/(1-leverage))[:,None];meat=influence.T@influence;df=n-k
        else:
            cluster_scores=np.array([x[groups==group].T@resid[groups==group] for group in sorted(set(groups))]);meat=cluster_scores.T@cluster_scores*g/(g-1)*(n-1)/(n-k);df=g-1
        variance=bread@meat@bread;beta=float(coeff[1]);se=float(np.sqrt(variance[1,1]));dist=stats.norm if reference=='normal' else stats.t(df)
        p=float(2*dist.sf(abs(beta/se)));width=float(dist.ppf(.975)*se)
        rows.append(dict(marker=marker,beta=beta,se=se,p=p,ci_low=beta-width,ci_high=beta+width))
    raw=np.array([r['p'] for r in rows]);order=np.argsort(raw);sorted_p=raw[order];m=len(rows)
    if adjustment=='bonferroni':q=np.minimum(raw*m,1)
    else:
        adjusted=np.minimum.accumulate((sorted_p*m/np.arange(1,m+1))[::-1])[::-1] if adjustment=='fdr_bh' else np.maximum.accumulate(sorted_p*np.arange(m,0,-1))
        q=np.empty(m);q[order]=np.clip(adjusted,0,1)
    for row,value in zip(rows,q):row['q']=float(value);row['decision']='supported' if value<.05 else 'inconclusive'
    return rows


def spatial_expected(inp, interpolation):
    if interpolation not in ('linear','nearest'):raise ValueError('unpublished_interpolation')
    atlas=nib.load(inp/'atlas.nii.gz');labels=np.asarray(atlas.dataobj).astype(int);frame=read_json(inp/'atlas.json')['coordinate_frame']
    scales={'mm':1.,'meter':1000.,'micron':.001};target=atlas.affine.copy();target[:3,:]*=scales[atlas.header.get_xyzt_units()[0]]
    status={};measurements=[]
    for row in read_csv(inp/'records.csv'):
        try:
            image=nib.load(inp/row['path']);data=np.asanyarray(image.dataobj);units=image.header.get_xyzt_units()[0]
            if data.ndim!=3 or not np.isfinite(data).all():status[row['record_id']]='unsupported';continue
            if units not in scales or row['coordinate_frame']!=frame:status[row['record_id']]='uncertain';continue
            source=image.affine.copy();source[:3,:]*=scales[units]
            coordinates=nib.affines.apply_affine(np.linalg.inv(source)@target,np.argwhere(labels>0))
            if (coordinates<0).any() or (coordinates>np.asarray(data.shape)-1).any():status[row['record_id']]='uncertain';continue
            resampled=resample_from_to(nib.Nifti1Image(data,source),(labels.shape,target),order=1 if interpolation=='linear' else 0,mode='constant',cval=np.nan)
            aligned=np.asanyarray(resampled.dataobj);status[row['record_id']]='complete'
            for label in np.unique(labels[labels>0]):
                mask=labels==label;n=int(mask.sum());measurements.append(dict(record_id=row['record_id'],region_id=int(label),voxel_count=n,volume_mm3=n*abs(np.linalg.det(target[:3,:3])),mean_value=float(aligned[mask].mean())))
        except (OSError,ValueError):status[row['record_id']]='unsupported'
    return status,measurements


def connectivity_expected(inp):
    result={}
    for record in read_csv(inp/'records.csv'):
        with np.load(inp/record['series'],allow_pickle=False) as data:
            x=data['signal'].copy();frames=data['frame_ids'];roi=data['roi_ids'].copy()
        conf=pd.read_csv(inp/record['confounds']).set_index('frame_id').loc[frames]
        keep=np.isfinite(conf.fd.to_numpy())&(conf.fd.to_numpy()<.35);z=np.column_stack([np.ones(keep.sum()),conf.loc[keep,[f'm{k}' for k in range(6)]].to_numpy()])
        residual=x[keep]-(z@np.linalg.pinv(z))@x[keep];residual-=residual.mean(axis=0);length=np.sqrt(np.sum(residual**2,axis=0));valid=length>1e-8
        normalized=np.zeros_like(residual);normalized[:,valid]=residual[:,valid]/length[valid];fc=np.clip(normalized.T@normalized,-1,1);fc[valid,valid]=1
        upper=np.triu(np.outer(valid,valid),1)
        result[record['participant_id']]={'fc':fc,'roi_ids':roi,'valid':valid,'n_frames':int(keep.sum()),'n_valid_rois':int(valid.sum()),'mean_fc':float(fc[upper].mean())}
    return result


def evaluate(case, output, processes, *, require_execution=True):
    output=Path(output);inp=case.root/'inputs';checks=[];errors=[];endpoints={}
    def check(name,value):
        ok=bool(value);checks.append({'name':name,'passed':ok})
        if not ok:errors.append(name)
    def numeric(name,a,b):check(name,arrays_close(np.atleast_1d(np.asarray(a,dtype=float)),np.atleast_1d(np.asarray(b,dtype=float))))
    try:
        check('public_artifacts',public_checks(case.contract,output)['public_check_pass'])
        check('complete_status',read_json(output/'result.json').get('status')=='complete')
        events=trace_events(Path(processes)/'trusted');successful={e['run_id'] for e in events if e['type']=='process_end' and e.get('success')}
        starts=[e for e in events if e['type']=='process_start' and e['run_id'] in successful and e.get('kind')=='project']
        if require_execution:check('actual_final_entrypoint_execution',bool(starts) and starts[-1]['entrypoint_sha256']==sha(output/'project/main.py'))
        check('managed_fit_budget',sum(e['type']=='managed_fit' for e in events)<=64)
        check('report_nonempty',bool((output/'report.md').read_text(encoding='utf-8').strip()))
        if case.id=='O01':
            train=pd.read_csv(inp/'train.csv',dtype={'participant_id':str});test=pd.read_csv(inp/'external.csv',dtype={'participant_id':str})
            validation=keyed(read_csv(output/'validation.csv'),['participant_id']);pred=keyed(read_csv(output/'predictions.csv'),['participant_id'])
            truth=train.groupby('participant_id',sort=True).agg({'label':'first','site':'first'});external=set(test.participant_id)
            check('validation_participant_conservation',{k[0] for k in validation}==set(truth.index));check('external_participant_conservation',{k[0] for k in pred}==external)
            for group in (validation,pred):
                for key,row in group.items():check('probability_and_threshold:'+key[0],np.isfinite(float(row['score'])) and 0<=float(row['score'])<=1 and integer(row['prediction'])==int(float(row['score'])>=.5))
            for key,row in validation.items():check('site_holdout_identity:'+key[0],str(row['fold'])==str(truth.loc[key[0],'site']))
            order=list(truth.index);actual=balanced_accuracy_score(truth.label.to_numpy(),[integer(validation[(sid,)]['prediction']) for sid in order]);metrics=read_json(output/'metrics.json')
            numeric('reported_validation_balanced_accuracy',metrics['validation_balanced_accuracy'],actual);check('external_label_custody',metrics['external_metrics'] is None)
            labels=read_json(case.root.parents[2]/'private/O01_LABELS.json');labels={r['participant_id']:r['label'] for r in labels if r['site']=='E'}
            ids=sorted(external);y=[labels[sid] for sid in ids];scores=[float(pred[(sid,)]['score']) for sid in ids]
            endpoints={'validation_balanced_accuracy':float(actual),'external_balanced_accuracy':float(balanced_accuracy_score(y,np.array(scores)>=.5)),'external_roc_auc':float(roc_auc_score(y,scores)),'external_participants':len(ids)}
        elif case.id=='O02':
            method=read_json(output/'method.json');expected=inference_expected(inp,method);found=keyed(read_csv(output/'estimates.csv'),['marker'])
            check('marker_conservation',{k[0] for k in found}=={r['marker'] for r in expected})
            for row in expected:
                actual=found[(row['marker'],)]
                for key in ('beta','se','p','q','ci_low','ci_high'):numeric(row['marker']+':'+key,actual[key],row[key])
                check(row['marker']+':evidence_decision',actual['decision']==row['decision'])
            endpoints={'method':method,'supported_markers':[r['marker'] for r in expected if r['decision']=='supported']}
        elif case.id=='O03':
            method=read_json(output/'method.json');statuses,expected=spatial_expected(inp,method['interpolation']);found=keyed(read_csv(output/'status.csv'),['record_id'])
            check('status_conservation',{k[0] for k in found}==set(statuses))
            for key,value in statuses.items():check('status:'+key,found[(key,)]['status']==value and bool(str(found[(key,)]['reason']).strip()))
            observed=read_csv(output/'measurements.csv')
            for row in observed:row['region_id']=str(integer(row['region_id']))
            observed=keyed(observed,['record_id','region_id']);expected_map={(r['record_id'],str(r['region_id'])):r for r in expected}
            check('measurement_identity_conservation',set(observed)==set(expected_map))
            for key,row in expected_map.items():
                for field in ('voxel_count','volume_mm3','mean_value'):numeric(str(key)+':'+field,observed[key][field],row[field])
            endpoints={'statuses':statuses,'interpolation':method['interpolation'],'measurements':len(expected)}
        elif case.id=='O04':
            expected=connectivity_expected(inp);observed=keyed(read_csv(output/'summary.csv'),['participant_id']);check('participant_conservation',{k[0] for k in observed}==set(expected))
            for sid,row in expected.items():
                actual=observed[(sid,)]
                for field in ('n_frames','n_valid_rois','mean_fc'):numeric(sid+':'+field,actual[field],row[field])
                with np.load(output/f'connectivity/{sid}.npz',allow_pickle=False) as data:
                    ids=[integer(v) for v in data['roi_ids']];check(sid+':roi_identity',len(ids)==len(set(ids)) and set(ids)==set(map(int,row['roi_ids'])))
                    order=[ids.index(integer(v)) for v in row['roi_ids']];numeric(sid+':connectivity',data['fc'][np.ix_(order,order)],row['fc']);check(sid+':valid_mask',np.array_equal(data['valid'][order],row['valid']))
            diagnoses=read_json(output/'diagnosis.json');check('diagnostic_evidence_present',isinstance(diagnoses,list) and bool(diagnoses) and all(isinstance(r,dict) and str(r.get('evidence','')).strip() and str(r.get('explanation','')).strip() for r in diagnoses))
            endpoints={'participants':len(expected),'diagnosis_narrative_machine_certified':False}
        else:raise ValueError('unknown_case')
    except Exception as exc:
        check('evaluator_exception:'+type(exc).__name__,False);errors.append(str(exc)[:700])
    return {'passed':not errors,'verdict':'PASS' if not errors else 'FAIL','case_id':case.id,'checks':checks,'errors':errors,'endpoints':endpoints,'expert_scientific_acceptance':False}
