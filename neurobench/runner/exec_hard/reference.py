"""Private preparation programs. None of this module is mounted for candidates."""
from __future__ import annotations

import inspect
from pathlib import Path

HEADER = '''import argparse, csv, hashlib, json, math, shutil
from pathlib import Path
import numpy as np
from scipy import stats, ndimage, optimize
import nibabel as nib
from nibabel.processing import resample_from_to

def rd(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def jr(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def jw(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
def cw(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def hs(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
'''


def connectivity(path, sidecar, alternate=False):
    x=np.load(path,allow_pickle=False)
    ids=jr(sidecar)['roi_ids']
    order=np.argsort([int(i) for i in ids]);x=x[:,order];ids=np.array(ids,dtype='U')[order]
    valid=np.ptp(x,axis=0)>1e-12
    fc=np.zeros((x.shape[1],x.shape[1]))
    if alternate:
        fc[np.ix_(valid,valid)]=np.corrcoef(x[:,valid],rowvar=False)
    else:
        z=x[:,valid]-x[:,valid].mean(axis=0);n=np.sqrt((z*z).sum(axis=0))
        fc[np.ix_(valid,valid)]=(z.T@z)/np.outer(n,n)
    fc=np.clip(fc,-1,1);fc=(fc+fc.T)/2
    fc[valid,valid]=1
    return fc,ids,valid


def fc_save(path, values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,fc=values[0],roi_ids=values[1],valid=values[2])


def fc_mean(values):
    a,_,valid=values
    b=a[np.ix_(valid,valid)]
    return float(b[np.triu_indices(len(b),1)].mean())


def cohort_solution(inp, alternate=False):
    phen=rd(inp/'phenotype.csv');acq=rd(inp/'acquisitions.csv')
    groups={}
    for r in phen: groups.setdefault(r['participant_id'],[]).append(r)
    decisions={};cohort=[]
    for sid, rows in sorted(groups.items(),reverse=alternate):
        signatures={tuple(r[k] for k in ('site','label','age','sex','mean_fd')) for r in rows}
        status,reason='included','consistent paired baseline'
        if len(signatures)!=1: status,reason='uncertain','conflicting phenotype'
        else:
            r=rows[0]
            try: ok=float(r['age'])>0 and int(r['sex']) in (1,2) and float(r['mean_fd'])>=0 and all(np.isfinite(float(r[k])) for k in ('age','mean_fd'))
            except (ValueError,TypeError): ok=False
            if not ok: status,reason='excluded','invalid phenotype fields'
        found={}
        if status=='included':
            for atlas in ('aal','cc200'):
                options={r['path'] for r in acq if r['participant_id']==sid and r['atlas']==atlas}
                validpaths=[]
                for rel in options:
                    try:
                        a=np.load(inp/rel,allow_pickle=False)
                        if a.ndim==2 and np.isfinite(a).all(): validpaths.append(rel)
                    except (OSError,ValueError): pass
                if len(validpaths)!=1: status,reason='excluded','missing or ambiguous paired acquisition';break
                found[atlas]=validpaths[0]
        decisions[sid]=(status,reason)
        if status=='included': cohort.append({k:rows[0][k] for k in ('participant_id','site','label','age','sex','mean_fd')} | found)
    audit=[]
    for source,rows in (('phenotype.csv',phen),('acquisitions.csv',acq)):
        for r in rows:
            status,reason=decisions.get(r['participant_id'],('uncertain','no matching phenotype'))
            audit.append(dict(source=source,record_id=r['record_id'],participant_id=r['participant_id'],status=status,reason=reason))
    return cohort,audit


def h01(inp,out,alternate=False):
    cohort,audit=cohort_solution(inp,alternate)
    cw(out/'cohort.csv',cohort,['participant_id','site','label','age','sex','mean_fd','aal','cc200'])
    jw(out/'audit.json',{'records':audit})


def network_metrics(values, alternate=False):
    fc,ids,valid=values;indices=np.flatnonzero(valid);n=len(indices)
    pairs=[(float(fc[i,j]),int(ids[i]),int(ids[j]),a,b) for a,i in enumerate(indices) for b,j in enumerate(indices) if a<b and fc[i,j]>0]
    pairs.sort(key=lambda r:(-r[0],r[1],r[2]))
    pairs=pairs[:((n*(n-1)//2)+9)//10]
    adj=np.zeros((n,n),dtype=np.int64)
    for _,_,_,a,b in pairs: adj[a,b]=adj[b,a]=1
    if alternate:
        import networkx as nx
        trans=nx.transitivity(nx.from_numpy_array(adj))
    else:
        degree=adj.sum(axis=0);den=float((degree*(degree-1)/2).sum())
        trans=float(np.trace(adj@adj@adj)/2/den) if den else 0.
    return {'valid_rois':n,'n_edges':len(pairs),'mean_fc':fc_mean(values),'global_transitivity':trans}


def h02(inp,out,alternate=False):
    rows=[];invalid=[]
    for r in rd(inp/'cohort.csv'):
        sid=r['participant_id']
        for atlas in ('aal','cc200'):
            p=inp/f'series/{sid}_{atlas}.npy'
            values=connectivity(p,p.with_suffix('.json'),alternate)
            fc_save(out/f'connectivity/{sid}_{atlas}.npz',values)
            rows.append(dict(participant_id=sid,atlas=atlas,**network_metrics(values,alternate)))
            invalid.append({'participant_id':sid,'atlas':atlas,'invalid_roi_ids':values[1][~values[2]].tolist()})
    cw(out/'comparison.csv',rows,['participant_id','atlas','valid_rois','n_edges','mean_fc','global_transitivity'])
    jw(out/'qc.json',{'records':invalid})


def prediction_data(inp):
    rows=sorted(rd(inp/'cohort.csv'),key=lambda r:r['participant_id'])
    features=rd(inp/'features.csv');lookup={r['participant_id']:r for r in features}
    ids=sorted(k for k in features[0] if k!='participant_id')
    x=np.array([[float(lookup[r['participant_id']][k]) for k in ids] for r in rows])
    y=np.array([int(r['label']) if r['label']!='' else -1 for r in rows])
    return rows,ids,x,y


def fitted(x,y,family,hp,alternate=False):
    mean=np.nanmean(np.where(np.isfinite(x),x,np.nan),axis=0)
    mean=np.nan_to_num(mean);z=np.where(np.isfinite(x),x,mean)
    scale=z.std(axis=0);scale[scale==0]=1;z=(z-mean)/scale
    if alternate:
        design=np.column_stack([z,np.ones(len(z))])
        if family=='ridge':
            reg=np.eye(design.shape[1])*hp;reg[-1,-1]=0
            coef=np.linalg.solve(design.T@design+reg,design.T@(2*y-1))
        else:
            from scipy.special import expit
            def objective(b):
                t=design@b;loss=np.logaddexp(0,t).sum()-y@t+.5/hp*(b[:-1]@b[:-1])
                g=design.T@(expit(t)-y);g[:-1]+=b[:-1]/hp
                return loss,g
            result=optimize.minimize(objective,np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-9,'maxiter':2000})
            coef=result.x
        w,b=coef[:-1],float(coef[-1])
    else:
        from sklearn.linear_model import LogisticRegression,RidgeClassifier
        model=LogisticRegression(C=hp,solver='lbfgs',tol=1e-10,max_iter=2000) if family=='logistic_l2' else RidgeClassifier(alpha=hp,solver='svd')
        model.fit(z,y);w=np.asarray(model.coef_).reshape(-1);b=float(np.asarray(model.intercept_).reshape(-1)[0])
    return mean,scale,w,b


def scores(state,x):
    mean,scale,w,b=state
    return ((np.where(np.isfinite(x),x,mean)-mean)/scale)@w+b


def balanced(y,p):
    return float(np.mean([(p[y==k]==k).mean() for k in (0,1)]))


def prediction_solution(inp,alternate=False):
    rows,ids,x,y=prediction_data(inp)
    dev=np.array([r['split']=='development' for r in rows]);sites=np.array([r['site'] for r in rows])
    devsites=sorted(set(sites[dev]));candidates=[];fold_predictions={}
    for family in ('logistic_l2','ridge'):
        for hp in (.1,1.,10.):
            allscores=np.zeros(len(rows));acc=[]
            for site in devsites:
                train=dev&(sites!=site);val=dev&(sites==site)
                state=fitted(x[train],y[train],family,hp,alternate)
                allscores[val]=scores(state,x[val]);acc.append(balanced(y[val],(allscores[val]>=0).astype(int)))
            result={'family':family,'hyperparameter':hp,'mean_site_balanced_accuracy':float(np.mean(acc))}
            candidates.append(result);fold_predictions[family,hp]=allscores
    selected=max(candidates,key=lambda r:r['mean_site_balanced_accuracy'])
    state=fitted(x[dev],y[dev],selected['family'],selected['hyperparameter'],alternate)
    pred=scores(state,x);pred[dev]=fold_predictions[selected['family'],selected['hyperparameter']][dev]
    folds=[{'site':site,'fit_ids':[r['participant_id'] for i,r in enumerate(rows) if dev[i] and r['site']!=site],
            'validation_ids':[r['participant_id'] for i,r in enumerate(rows) if dev[i] and r['site']==site]} for site in devsites]
    return rows,ids,x,y,dev,selected,state,pred,candidates,folds


def h03(inp,out,alternate=False):
    rows,ids,x,y,dev,selected,state,pred,candidates,folds=prediction_solution(inp,alternate)
    cw(out/'predictions.csv',[{'participant_id':r['participant_id'],'split':r['split'],'score':float(pred[i]),'prediction':int(pred[i]>=0)} for i,r in enumerate(rows)],['participant_id','split','score','prediction'])
    jw(out/'evaluation.json',dict(selected,external_metrics='pending_host_evaluation'))
    jw(out/'fit_evidence.json',{'final_fit_participant_ids':[r['participant_id'] for i,r in enumerate(rows) if dev[i]],'validation_folds':folds,'candidate_scores':candidates})
    (out/'model').mkdir(exist_ok=True)
    np.savez(out/'model/state.npz',feature_ids=np.array(ids,dtype='U'),mean=state[0],scale=state[1],coef=state[2],intercept=state[3])


def inference_data(inp):
    cohort=rd(inp/'cohort.csv');f=rd(inp/'connectivity.csv');lookup={r['participant_id']:r for r in f}
    features=sorted(k for k in f[0] if k!='participant_id');included=[];audit=[]
    for r in cohort:
        try:
            values=[float(r[k]) for k in ('label','age','sex','mean_fd')]+[float(lookup[r['participant_id']][k]) for k in features]
            valid=np.isfinite(values).all() and int(r['sex']) in (1,2)
        except (ValueError,KeyError): valid=False
        audit.append({'participant_id':r['participant_id'],'status':'included' if valid else 'excluded','reason':'complete analysis row' if valid else 'missing or invalid covariate/outcome'})
        if valid: included.append(r)
    included.sort(key=lambda r:r['participant_id']);sites=sorted({r['site'] for r in included})
    columns=['intercept','label','age','sex2','mean_fd']+['site:'+s for s in sites[1:]]
    design=np.array([[1,float(r['label']),float(r['age']),int(r['sex'])==2,float(r['mean_fd'])]+[r['site']==s for s in sites[1:]] for r in included],float)
    y=np.array([[float(lookup[r['participant_id']][k]) for k in features] for r in included])
    return included,features,columns,design,y,audit


def inference_solution(inp,alternate=False):
    included,features,columns,x,y,audit=inference_data(inp)
    if np.linalg.matrix_rank(x)!=x.shape[1]: raise ValueError('rank deficient')
    if alternate:
        import statsmodels.api as sm
        beta=[];se=[];pv=[]
        for j in range(y.shape[1]):
            model=sm.OLS(y[:,j],x).fit(cov_type='HC3',use_t=False)
            beta.append(float(model.params[1]));se.append(float(model.bse[1]));pv.append(float(model.pvalues[1]))
        beta,se,pv=map(np.asarray,(beta,se,pv))
    else:
        pinv=np.linalg.pinv(x);coef=pinv@y;residual=y-x@coef
        leverage=(x*pinv.T).sum(axis=1)
        se=np.sqrt(((pinv[1,:,None]*residual/(1-leverage[:,None]))**2).sum(axis=0))
        beta=coef[1];pv=2*stats.norm.sf(np.abs(beta/se))
    order=np.argsort(pv);adj=np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1]
    q=np.zeros(len(pv));q[order]=np.minimum(adj,1)
    values=[dict(feature_id=name,beta=float(beta[i]),se=float(se[i]),ci_low=float(beta[i]-stats.norm.ppf(.975)*se[i]),ci_high=float(beta[i]+stats.norm.ppf(.975)*se[i]),p=float(pv[i]),q=float(q[i])) for i,name in enumerate(features)]
    return included,columns,x,audit,values


def h05(inp,out,alternate=False):
    included,columns,x,audit,values=inference_solution(inp,alternate)
    cw(out/'estimates.csv',values,['feature_id','beta','se','ci_low','ci_high','p','q'])
    jw(out/'design.json',{'participant_ids':[r['participant_id'] for r in included],'columns':columns,'n':len(x),'rank':int(np.linalg.matrix_rank(x))})
    cw(out/'sample_audit.csv',audit,['participant_id','status','reason'])
    jw(out/'report.json',{'n':len(x),'feature_count':len(values),'significant_feature_ids':[r['feature_id'] for r in values if r['q']<.05]})


def aligned_images(inp,alternate=False):
    ref=nib.load(inp/'reference.nii.gz');result=[]
    for name,order in (('labels',0),('values',1)):
        image=nib.load(inp/f'images/{name}.nii.gz')
        if alternate:
            data=np.asarray(resample_from_to(image,(ref.shape,ref.affine),order=order,mode='constant',cval=0).dataobj)
        else:
            transform=np.linalg.solve(image.affine,ref.affine)
            data=ndimage.affine_transform(np.asarray(image.dataobj),transform[:3,:3],transform[:3,3],output_shape=ref.shape,order=order,mode='constant',cval=0,prefilter=False)
        result.append(data)
    return ref,result[0].astype(np.int32),result[1]


def h07(inp,out,alternate=False):
    ref,labels,values=aligned_images(inp,alternate);(out/'images').mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(labels,ref.affine),out/'images/labels.nii.gz')
    nib.save(nib.Nifti1Image(values.astype(np.float32),ref.affine),out/'images/values.nii.gz')
    vol=float(abs(np.linalg.det(ref.affine[:3,:3])))
    rows=[{'roi_id':int(k),'voxel_count':int((labels==k).sum()),'volume_mm3':float((labels==k).sum()*vol),'mean_value':float(values[labels==k].mean())} for k in np.unique(labels) if k>0]
    cw(out/'measurements.csv',rows,['roi_id','voxel_count','volume_mm3','mean_value'])
    jw(out/'spatial_audit.json',{'reference_shape':list(ref.shape),'reference_affine':ref.affine.tolist(),'voxel_volume_mm3':vol,'space':'MNI','units':'mm'})


def h09(inp,out,alternate=False):
    groups={};ids=[]
    for row in rd(inp/'cohort.csv'):
        sid=row['participant_id'];ids.append(sid)
        p=inp/f'series/{sid}_aal.npy';values=connectivity(p,p.with_suffix('.json'),alternate)
        fc_save(out/f'connectivity/{sid}.npz',values)
        groups.setdefault((row['site'],int(row['label'])),[]).append(fc_mean(values))
    cw(out/'summary.csv',[{'site':k[0],'label':k[1],'n':len(v),'mean_fc':float(np.mean(v))} for k,v in sorted(groups.items())],['site','label','n','mean_fc'])
    jw(out/'execution.json',{'entrypoint':'project/main.py','analyzed_participant_ids':sorted(ids)})


def cache_fc(inp,out,sid,alternate=False):
    p=inp/f'series/{sid}_aal.npy';fc_save(out/f'results/{sid}_fc.npz',connectivity(p,p.with_suffix('.json'),alternate))


def cache_metric(out,sid,config):
    with np.load(out/f'results/{sid}_fc.npz',allow_pickle=False) as data:
        values=data['fc'],data['roi_ids'],data['valid']
    a=values[0][np.ix_(values[2],values[2])];upper=a[np.triu_indices(len(a),1)]
    jw(out/f'results/{sid}_metric.json',{'mean_fc':float(upper.mean()),'above_threshold':int((upper>config['threshold']).sum())})


def cache_group(out,ids):
    rows=[jr(out/f'results/{sid}_metric.json') for sid in ids]
    jw(out/'results/group.json',{'n':len(rows),'mean_fc':float(np.mean([r['mean_fc'] for r in rows])),'above_threshold_sum':sum(r['above_threshold'] for r in rows)})


def cache_specs(inp,out,ids,config):
    result=[]
    for sid in ids:
        result.append((f'fc:{sid}',[f'inputs/series/{sid}_aal.npy',f'inputs/series/{sid}_aal.json'],[f'output/results/{sid}_fc.npz'],{},f'cache_fc(inp,out,"{sid}",ALTERNATE)'))
        result.append((f'metric:{sid}',[f'output/results/{sid}_fc.npz'],[f'output/results/{sid}_metric.json'],config[sid],f'cache_metric(out,"{sid}",{config[sid]!r})'))
    result.append(('group',[f'output/results/{sid}_metric.json' for sid in ids],['output/results/group.json'],{},'cache_group(out,ids)'))
    return result


def h10(inp,out,alternate=False):
    from runtime import run_step
    ids=sorted(r['participant_id'] for r in rd(inp/'cohort.csv'));config=jr(inp/'config.json')
    shutil.copytree(inp/'prior/results',out/'results',dirs_exist_ok=True)
    prior=jr(inp/'prior/records.json')['steps'];base=out.parent
    specs=cache_specs(inp,out,ids,config);actions=[];deps={}
    if alternate:
        specs=[s for s in specs if s[0].startswith('fc:')][::-1]+[s for s in specs if s[0].startswith('metric:')][::-1]+[s for s in specs if s[0]=='group']
    for name,inputs,outputs,settings,code in specs:
        old=prior[name]
        same=old['input_sha256']=={p:hs(base/p) for p in inputs} and old['config']==settings and all(hs(base/p)==old['output_sha256'][p] for p in outputs)
        run_step(name,inputs,outputs,settings,code,reuse=bool(same))
        actions.append({'name':name,'action':'reuse' if same else 'execute'})
        deps[name]={'inputs':inputs,'outputs':outputs,'config':settings}
    jw(out/'dependencies.json',deps);jw(out/'execution.json',{'steps':actions})


def record_status(inp,row):
    matching=[r for r in rd(inp/'cohort.csv') if r['participant_id']==row['participant_id']]
    if len(matching)!=1: return 'uncertain','no unique matching metadata'
    path=inp/row['path']
    if not path.is_file() or not path.with_suffix('.json').is_file(): return 'unsupported','missing series or ROI definitions'
    try:
        x=np.load(path,allow_pickle=False);ids=jr(path.with_suffix('.json'))['roi_ids']
        ok=x.ndim==2 and x.shape[0]>=100 and len(ids)==x.shape[1] and len(set(ids))==len(ids) and np.isfinite(x).all() and (np.ptp(x,axis=0)>1e-12).sum()>=2
    except (ValueError,KeyError,OSError): ok=False
    return ('complete','supported finite series') if ok else ('unsupported','invalid shape, duration, numeric data or ROI identity')


def h11(inp,out,alternate=False):
    results=[]
    for row in rd(inp/'records.csv'):
        status,reason=record_status(inp,row);artifact=''
        if status=='complete':
            p=inp/row['path'];artifact=f'outputs/{row["record_id"]}.npz'
            fc_save(out/artifact,connectivity(p,p.with_suffix('.json'),alternate))
        results.append(dict(record_id=row['record_id'],participant_id=row['participant_id'],status=status,reason=reason,artifact=artifact))
    cw(out/'status.csv',results,['record_id','participant_id','status','reason','artifact'])
    jw(out/'summary.json',{'total':len(results),**{s:sum(r['status']==s for r in results) for s in ('complete','unsupported','uncertain')}})


def namespace():
    scope={}
    exec(HEADER,scope)
    for function in FUNCTIONS:
        exec(inspect.getsource(function),scope)
    scope['ALTERNATE']=False
    return scope


def create_prior(inp,selected):
    scope=namespace();inp=Path(inp);out=inp/'prior';(out/'results').mkdir(parents=True)
    ids=sorted(r['participant_id'] for r in selected);config=scope['jr'](inp/'prior_config.json');steps={}
    for sid in ids:
        scope['cache_fc'](inp,out,sid);scope['cache_metric'](out,sid,config[sid])
    scope['cache_group'](out,ids)
    for name,inputs,outputs,settings,_ in scope['cache_specs'](inp,out,ids,config):
        def resolve(p):
            return inp/p.removeprefix('inputs/') if p.startswith('inputs/') else out/p.removeprefix('output/')
        steps[name]={'input_sha256':{p:scope['hs'](resolve(p)) for p in inputs},'output_sha256':{p:scope['hs'](resolve(p)) for p in outputs},'config':settings}
    scope['jw'](out/'records.json',{'steps':steps})


FUNCTIONS = (connectivity,fc_save,fc_mean,cohort_solution,h01,network_metrics,h02,
    prediction_data,fitted,scores,balanced,prediction_solution,h03,inference_data,
    inference_solution,h05,aligned_images,h07,h09,cache_fc,cache_metric,cache_group,
    cache_specs,h10,record_status,h11)


def program(case_id,alternate=False):
    return HEADER+'\nALTERNATE='+repr(alternate)+'\n\n'+'\n\n'.join(inspect.getsource(f) for f in FUNCTIONS)+f'''\n
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();inp=Path(args.input).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    {case_id.lower()}(inp,out,ALTERNATE)
'''
