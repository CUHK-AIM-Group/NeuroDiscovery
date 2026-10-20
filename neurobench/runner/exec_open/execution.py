"""Reuse the common controlled host with fresh clean and metamorphic checks."""
from __future__ import annotations

import shutil
import numpy as np
import pandas as pd
from neurobench.runner.exec_hard.host import prepare_workspace,run_process
from .common import *
from .oracle import evaluate


def probe(case,output,target,library,train,test):
    """Supply only passed train/test data, with no withheld labels on disk."""
    target=Path(target);clone=target/'case';clone.parent.mkdir(parents=True,exist_ok=False)
    shutil.copytree(case.root,clone)
    train.to_csv(clone/'inputs/train.csv',index=False);test.to_csv(clone/'inputs/external.csv',index=False)
    contract={**case.contract,'input_sha256':file_map(clone/'inputs')};write_json(clone/'CONTRACT.json',contract)
    work=prepare_workspace(Case(clone,contract),target/'workspace',library)
    shutil.copytree(Path(output)/'project',work/'output/project')
    (work/'scratch/probe.py').write_text('''import sys,json
sys.path.insert(0,str(OUTPUT_DIR/'project'))
import pandas as pd
from learner import fit_predict
train=pd.read_csv(INPUT_DIR/'train.csv',dtype={'participant_id':str})
test=pd.read_csv(INPUT_DIR/'external.csv',dtype={'participant_id':str})
# A learner is a pure train/test function: input/output and other files are not data sources.
def no_io(event,args):
    if event=='open' and not isinstance(args[0],int):
        from pathlib import Path
        p=Path(args[0]).resolve()
        code_import=p.is_relative_to(OUTPUT_DIR/'project') and p.suffix in ('.py','.pyc') and not (isinstance(args[1],str) and any(c in args[1] for c in 'wax+'))
        if not code_import and (p.is_relative_to(INPUT_DIR) or p.is_relative_to(OUTPUT_DIR) or p.is_relative_to(Path.cwd()/'scratch')):
            raise PermissionError('fit_predict_must_use_passed_frames_only')
import numpy as np
sys.addaudithook(no_io)
result=fit_predict(train,test)
if not isinstance(result,dict):raise ValueError('fit_predict_requires_dict')
value={str(k):float(v) for k,v in result.items()}
assert set(value)==set(test.participant_id.astype(str))
assert all(np.isfinite(v) and 0<=v<=1 for v in value.values())
# Return through stdout so no learner-time data/output-file access is needed.
print('PROBE_RESULT:'+json.dumps(value,allow_nan=False,sort_keys=True))
''',encoding='utf-8')
    run=run_process(work,target/'processes',kind='python',target='scratch/probe.py')
    value=None
    if run['success']:
        line=next((s for s in run['stdout'].splitlines() if s.startswith('PROBE_RESULT:')),None)
        if line is not None:value=json.loads(line[len('PROBE_RESULT:'):])
    write_json(target/'PROBE.json',{'run':run,'scores':value})
    if value is None:raise ValueError('learner_probe_failed:'+run['stderr'][-1000:])
    return value


def prediction_checks(case,output,target,library):
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    inp=case.root/'inputs';train=pd.read_csv(inp/'train.csv',dtype={'participant_id':str});test=pd.read_csv(inp/'external.csv',dtype={'participant_id':str});checks=[]
    observed={r['participant_id']:float(r['score']) for r in read_csv(Path(output)/'validation.csv')}
    for site in sorted(train.site.unique()):
        fit=train[train.site!=site].copy();held=train[train.site==site].drop(columns='label').copy()
        scores=probe(case,output,target/f'fold_{site}',library,fit,held)
        checks.append({'name':'independent_site_validation:'+str(site),'passed':set(scores)==set(held.participant_id) and all(arrays_close([v],[observed[k]]) for k,v in scores.items())})
    expected={r['participant_id']:float(r['score']) for r in read_csv(Path(output)/'predictions.csv')}
    for name,data in [('all',test),('test_reversed',test.iloc[::-1].copy()),('single_participant',test[test.participant_id==sorted(test.participant_id.unique())[0]].copy())]:
        scores=probe(case,output,target/name,library,train.copy(),data)
        checks.append({'name':'pure_train_test_prediction:'+name,'passed':all(arrays_close([v],[expected[k]]) for k,v in scores.items())})
    record={'passed':all(c['passed'] for c in checks),'checks':checks,'public_rule':'same passed training data, no file I/O or fitting on test values; held-site and reordered/batch-independent predictions'}
    write_json(target/'CHECKS.json',record);return record


def replay(case,output,target,library,**kwargs):
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    work=prepare_workspace(case,target/'workspace',library)
    shutil.copytree(Path(output)/'project',work/'output/project')
    run=run_process(work,target/'processes')
    verdict={'passed':False,'errors':['clean_execution_failed']};extra={'passed':True,'applicable':False}
    if run['success']:
        finalize(case,work/'output');verdict=evaluate(case,work/'output',target/'processes')
        if verdict['passed'] and case.id=='O01':
            try:extra=prediction_checks(case,work/'output',target/'learner_checks',library)
            except Exception as exc:extra={'passed':False,'error_type':type(exc).__name__,'error':str(exc)[:1000]}
        if verdict['passed'] and case.id=='O04':
            clone=target/'reordered_case';shutil.copytree(case.root,clone)
            for row in read_csv(clone/'inputs/records.csv'):
                path=clone/'inputs'/row['confounds'];rows=read_csv(path);write_csv(path,rows[::-1],list(rows[0]))
                path=clone/'inputs'/row['series']
                with np.load(path,allow_pickle=False) as d:values={k:d[k] for k in d.files}
                values['signal']=values['signal'][:,::-1];values['roi_ids']=values['roi_ids'][::-1];np.savez(path,**values)
            contract={**case.contract,'input_sha256':file_map(clone/'inputs')};write_json(clone/'CONTRACT.json',contract);altered=Case(clone,contract)
            altered_work=prepare_workspace(altered,target/'reordered_workspace',library);shutil.copytree(Path(output)/'project',altered_work/'output/project')
            altered_run=run_process(altered_work,target/'reordered_processes')
            if altered_run['success']:
                finalize(altered,altered_work/'output');altered_verdict=evaluate(altered,altered_work/'output',target/'reordered_processes')
            else:altered_verdict={'passed':False,'errors':['metamorphic_execution_failed']}
            extra={'passed':altered_run['success'] and altered_verdict['passed'],'run':altered_run,'evaluation':altered_verdict,'rule':'Confound-row and ROI-column order must follow identities.'}
    record={'passed':run['success'] and verdict['passed'] and extra['passed'],'run':run,'evaluation':verdict,'metamorphic':extra}
    write_json(target/'REPLAY.json',record);return record
