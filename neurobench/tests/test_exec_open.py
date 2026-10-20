"""Independent scientific-oracle and immutable-dispatch checks for open tasks."""
import itertools
import numpy as np
import pandas as pd
import pytest
stats = pytest.importorskip('scipy.stats')
sm = pytest.importorskip('statsmodels.api')
multipletests = pytest.importorskip('statsmodels.stats.multitest').multipletests
pytest.importorskip('sklearn.metrics')

from neurobench.runner.exec_open.common import file_map,sha,write_json
from neurobench.runner.exec_open.oracle import inference_expected
from neurobench.runner.exec_open.trial import verify_frozen,frozen_launcher


@pytest.mark.parametrize('unit,inference,adjustment',list(itertools.product(
    ['participant_mean','records'],['normal','t'],['fdr_bh','holm','bonferroni'])))
def test_all_public_inference_choices_match_independent_statsmodels(tmp_path,unit,inference,adjustment):
    rng=np.random.default_rng(93841);rows=[]
    for subject in range(48):
        site=subject%3;label=int(rng.random()<.3+.15*site);age=rng.uniform(15,50);sex=subject%2;base=rng.normal(size=6)
        for visit in range(3):
            fd=rng.uniform(.02,.25);y=base+label*np.arange(6)*.11+site*.3+age*.02+fd+rng.normal(scale=.3,size=6)
            rows.append(dict(participant_id=f'P{subject}',visit=visit,site=f'S{site}',label=label,age=age,sex=sex,mean_fd=fd,**{f'm{i}':v for i,v in enumerate(y)}))
    d=pd.DataFrame(rows);d.to_csv(tmp_path/'observations.csv',index=False)
    method=dict(unit=unit,covariance='HC3' if unit=='participant_mean' else 'cluster',inference=inference,adjustment=adjustment,claim_type='association')
    actual=inference_expected(tmp_path,method)
    if unit=='participant_mean':d=d.groupby('participant_id').agg({**{k:'mean' for k in ['label','age','sex','mean_fd']+[f'm{i}' for i in range(6)]},'site':'first'}).reset_index()
    x=sm.add_constant(pd.concat([d[['label','age','sex','mean_fd']],pd.get_dummies(d.site,drop_first=True,dtype=float)],axis=1)).to_numpy(float)
    expected=[]
    for i in range(6):
        fit=sm.OLS(d[f'm{i}'].to_numpy(float),x).fit()
        robust=fit.get_robustcov_results(cov_type='HC3') if unit=='participant_mean' else fit.get_robustcov_results(cov_type='cluster',groups=d.participant_id.to_numpy(),use_correction=True)
        b=float(fit.params[1]);se=float(robust.bse[1]);df=len(d)-x.shape[1] if unit=='participant_mean' else d.participant_id.nunique()-1;dist=stats.norm if inference=='normal' else stats.t(df)
        p=float(2*dist.sf(abs(b/se)));width=float(dist.ppf(.975)*se);expected.append(dict(beta=b,se=se,p=p,ci_low=b-width,ci_high=b+width))
    q=multipletests([r['p'] for r in expected],method=adjustment)[1]
    for a,e,adjusted in zip(actual,expected,q):
        for key in e:assert a[key]==pytest.approx(e[key],rel=1e-8,abs=1e-10)
        assert a['q']==pytest.approx(adjusted,rel=1e-8,abs=1e-10)


def test_public_scoring_and_source_mutations_are_rejected_before_dispatch(tmp_path):
    mapping={}
    for directory,key in [('source_snapshot','snapshot_hashes'),('packet/public','public_packet_hashes'),('packet/private','private_packet_hashes'),('prompts','prompt_hashes')]:
        root=tmp_path/directory;root.mkdir(parents=True);(root/'binding.txt').write_text('frozen',encoding='utf-8');mapping[key]=file_map(root)
    q=tmp_path/'QUALIFICATION.json';write_json(q,{'status':'PASS'})
    mapping['qualification']={'path':q.name,'sha256':sha(q)}
    write_json(tmp_path/'MANIFEST.json',mapping);(tmp_path/'RUN_FROZEN.py').write_text('pass\n',encoding='utf-8')
    write_json(tmp_path/'BEFORE_DISPATCH.json',{'manifest_sha256':sha(tmp_path/'MANIFEST.json'),'launcher_sha256':sha(tmp_path/'RUN_FROZEN.py')})
    verify_frozen(tmp_path,mapping)
    (tmp_path/'packet/public/binding.txt').write_text('changed criterion',encoding='utf-8')
    with pytest.raises(ValueError,match='frozen_binding_changed:packet/public'):verify_frozen(tmp_path,mapping)


def test_real_frozen_launcher_preserves_source_tree_without_external_B_flag(tmp_path):
    import subprocess
    import sys
    snapshot=tmp_path/'snapshot'
    for name in ['neurobench','neurobench/runner','neurobench/runner/exec_open']:
        package=snapshot/name;package.mkdir(parents=True,exist_ok=True);(package/'__init__.py').write_text('',encoding='utf-8')
    (snapshot/'neurobench/runner/exec_open/trial.py').write_text("def main():\n    print('frozen-main-ran')\n",encoding='utf-8')
    before=file_map(snapshot);launcher=tmp_path/'RUN_FROZEN.py';launcher.write_text(frozen_launcher(snapshot),encoding='utf-8')
    child=subprocess.run([sys.executable,str(launcher)],cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert child.returncode==0 and 'frozen-main-ran' in child.stdout
    assert file_map(snapshot)==before
