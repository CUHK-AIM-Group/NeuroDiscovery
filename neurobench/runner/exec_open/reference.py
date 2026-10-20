"""Private preparation implementations; never mounted in model workspaces."""
from __future__ import annotations

import inspect
from .common import *

HEADER = '''import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage,stats

def rd(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def jr(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def jw(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\\n',encoding='utf-8')
def cw(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
'''


def learner_poly(train, test):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures,StandardScaler
    from sklearn.linear_model import LogisticRegression
    features=[f'f{k:02}' for k in range(16)]
    a=train.groupby('participant_id',sort=True)[features+['label']].mean()
    b=test.groupby('participant_id',sort=True)[features].mean()
    model=make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=.08,max_iter=2000,random_state=17))
    model.fit(a[features].to_numpy(),a['label'].to_numpy().astype(int))
    return {str(k):float(v) for k,v in zip(b.index,model.predict_proba(b.to_numpy())[:,1])}


def learner_svc(train, test):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    features=[f'f{k:02}' for k in range(16)]
    a=train.groupby('participant_id',sort=True)[features+['label']].mean()
    b=test.groupby('participant_id',sort=True)[features].mean()
    model=make_pipeline(StandardScaler(),SVC(C=2,gamma='scale',probability=True,random_state=17))
    model.fit(a[features].to_numpy(),a['label'].to_numpy().astype(int))
    return {str(k):float(v) for k,v in zip(b.index,model.predict_proba(b.to_numpy())[:,1])}


def solve_o01(inp,out,variant=0):
    from learner import fit_predict
    from sklearn.metrics import balanced_accuracy_score
    train=pd.read_csv(inp/'train.csv',dtype={'participant_id':str});test=pd.read_csv(inp/'external.csv',dtype={'participant_id':str})
    validation=[]
    for site in sorted(train.site.unique()):
        held=train[train.site==site]
        scores=fit_predict(train[train.site!=site].copy(),held.drop(columns='label').copy())
        validation.extend(dict(participant_id=str(k),fold=str(site),score=float(v),prediction=int(v>=.5)) for k,v in scores.items())
    scores=fit_predict(train.copy(),test.copy())
    predictions=[dict(participant_id=str(k),score=float(v),prediction=int(v>=.5)) for k,v in scores.items()]
    labels=train.groupby('participant_id').label.first().to_dict()
    metric=balanced_accuracy_score([labels[r['participant_id']] for r in validation],[r['prediction'] for r in validation])
    cw(out/'validation.csv',validation);cw(out/'predictions.csv',predictions)
    jw(out/'metrics.json',{'validation_balanced_accuracy':float(metric),'external_metrics':None})


def solve_o02(inp,out,variant=0):
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    method={'unit':'participant_mean','covariance':'HC3','inference':'normal','adjustment':'fdr_bh','claim_type':'association'} if variant==0 else {'unit':'records','covariance':'cluster','inference':'t','adjustment':'holm','claim_type':'association'}
    d=pd.read_csv(inp/'observations.csv',dtype={'participant_id':str})
    if method['unit']=='participant_mean':
        numeric=['label','age','sex','mean_fd']+[f'm{k}' for k in range(6)]
        agg={k:'mean' for k in numeric};agg['site']='first';d=d.groupby('participant_id',sort=True).agg(agg).reset_index()
    x=pd.concat([pd.DataFrame({'intercept':np.ones(len(d))}),d[['label','age','sex','mean_fd']].reset_index(drop=True),pd.get_dummies(d.site,drop_first=True,dtype=float).reset_index(drop=True)],axis=1).to_numpy(float)
    rows=[]
    for marker in [f'm{k}' for k in range(6)]:
        fit=sm.OLS(d[marker].to_numpy(float),x).fit()
        robust=fit.get_robustcov_results(cov_type='HC3') if method['unit']=='participant_mean' else fit.get_robustcov_results(cov_type='cluster',groups=d.participant_id.to_numpy(),use_correction=True)
        b,se=float(fit.params[1]),float(robust.bse[1]);df=len(d)-np.linalg.matrix_rank(x) if method['unit']=='participant_mean' else d.participant_id.nunique()-1
        distribution=stats.norm if method['inference']=='normal' else stats.t(df)
        p=float(2*distribution.sf(abs(b/se)));width=float(distribution.ppf(.975)*se)
        rows.append(dict(marker=marker,beta=b,se=se,p=p,q=0.,ci_low=b-width,ci_high=b+width,decision=''))
    q=multipletests([r['p'] for r in rows],method=method['adjustment'])[1]
    for r,v in zip(rows,q):r['q']=float(v);r['decision']='supported' if v<.05 else 'inconclusive'
    cw(out/'estimates.csv',rows);jw(out/'method.json',method)


def solve_o03(inp,out,variant=0):
    method={'interpolation':'linear' if variant==0 else 'nearest'};order=1 if variant==0 else 0
    atlas=nib.load(inp/'atlas.nii.gz');labels=np.asarray(atlas.dataobj).astype(int);atlas_frame=jr(inp/'atlas.json')['coordinate_frame']
    atlas_aff=atlas.affine.copy();atlas_aff[:3,:]*={'mm':1,'meter':1000,'micron':.001}[atlas.header.get_xyzt_units()[0]]
    positions=np.argwhere(labels>0);world=nib.affines.apply_affine(atlas_aff,positions);status=[];results=[]
    for row in rd(inp/'records.csv'):
        state='complete';reason='Established frame and finite covered field'
        try:
            image=nib.load(inp/row['path']);data=np.asarray(image.dataobj)
            if data.ndim!=3 or not np.isfinite(data).all():state='unsupported';reason='Nonfinite or non-3D data'
            elif image.header.get_xyzt_units()[0]=='unknown':state='uncertain';reason='Physical spatial units are unknown'
            elif row['coordinate_frame']!=atlas_frame:state='uncertain';reason='Different coordinate frame without a transform'
            else:
                affine=image.affine.copy();affine[:3,:]*={'mm':1,'meter':1000,'micron':.001}[image.header.get_xyzt_units()[0]]
                coords=nib.affines.apply_affine(np.linalg.inv(affine),world)
                if (coords<0).any() or (coords>np.array(data.shape)-1).any():state='uncertain';reason='Atlas centers outside supplied field'
                else:
                    values=ndimage.map_coordinates(data,coords.T,order=order,mode='constant',cval=np.nan,prefilter=False)
                    poslabels=labels[tuple(positions.T)]
                    for region in sorted(set(poslabels)):
                        v=values[poslabels==region];n=len(v)
                        results.append(dict(record_id=row['record_id'],region_id=int(region),voxel_count=n,volume_mm3=float(n*abs(np.linalg.det(atlas_aff[:3,:3]))),mean_value=float(v.mean())))
        except (OSError,ValueError):state='unsupported';reason='Unreadable image'
        status.append(dict(record_id=row['record_id'],status=state,reason=reason))
    cw(out/'status.csv',status);cw(out/'measurements.csv',results);jw(out/'method.json',method)


def solve_o04(inp,out,variant=0):
    summary=[];(out/'connectivity').mkdir(parents=True,exist_ok=True)
    for row in rd(inp/'records.csv'):
        with np.load(inp/row['series'],allow_pickle=False) as data:
            x=data['signal'];frames=data['frame_ids'];ids=data['roi_ids']
        conf={int(r['frame_id']):r for r in rd(inp/row['confounds'])}
        keep=np.array([np.isfinite(float(conf[int(f)]['fd'])) and float(conf[int(f)]['fd'])<.35 for f in frames])
        z=np.array([[1]+[float(conf[int(f)]['m'+str(k)]) for k in range(6)] for f in frames[keep]])
        values=x[keep]-z@np.linalg.lstsq(z,x[keep],rcond=None)[0];values-=values.mean(axis=0)
        norm=np.linalg.norm(values,axis=0);valid=norm>1e-8;fc=np.zeros((len(ids),len(ids)))
        if variant==0:fc[np.ix_(valid,valid)]=np.corrcoef(values[:,valid],rowvar=False)
        else:
            standardized=values[:,valid]/norm[valid]
            fc[np.ix_(valid,valid)]=standardized.T@standardized
        fc=np.clip(fc,-1,1);fc[valid,valid]=1
        if variant:
            order=np.argsort(ids);ids=ids[order];valid=valid[order];fc=fc[np.ix_(order,order)]
        np.savez(out/'connectivity'/(row['participant_id']+'.npz'),fc=fc,roi_ids=ids,valid=valid)
        upper=np.triu(np.outer(valid,valid),1)
        summary.append(dict(participant_id=row['participant_id'],n_frames=int(keep.sum()),n_valid_rois=int(valid.sum()),mean_fc=float(fc[upper].mean())))
    cw(out/'summary.csv',summary)
    jw(out/'diagnosis.json',[{'evidence':'Confound frame IDs differ from stored row positions; the repaired program aligns IDs before selection and regression.','explanation':'Positional selection/regression changes the measured signals.'},{'evidence':'Legacy code correlates rows and sorts ROI labels without applying a corresponding column permutation.','explanation':'Matrix dimension and ROI identities fail the public scientific target.'},{'evidence':'Residual constant ROIs and upper triangle differ from the legacy whole-matrix average.','explanation':'The repaired program preserves invalid masks and defines mean connectivity on valid pairs only.'}])


def write_reference(case, output, variant=0):
    output=Path(output);project=output/'project';project.mkdir(parents=True,exist_ok=True)
    name='solve_'+case.id.lower();function=globals()[name]
    source=HEADER+'\n'+inspect.getsource(function)+f'''\np=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()\n{name}(Path(a.input),Path(a.output),{variant})\n(Path(a.output)/'report.md').write_text('Synthetic development analysis. Methods and numerical evidence are recorded in the delivered artifacts. Association and transfer estimates do not establish clinical validity or causality.\\n',encoding='utf-8')\n'''
    (project/'main.py').write_text(source,encoding='utf-8')
    if case.id=='O01':
        f=learner_poly if variant==0 else learner_svc
        (project/'learner.py').write_text(inspect.getsource(f).replace('def '+f.__name__+'(', 'def fit_predict('),encoding='utf-8')
