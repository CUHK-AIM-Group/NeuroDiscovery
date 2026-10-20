"""Common cooperative execution host and private actual-step evidence.

No OS sandbox claim. This interface is identical in NR OFF and ON.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import psutil

from core.tools_impl import SKILL_TOOL_NAMES
from neurobench.runner.exec_budget_v3 import COMPLEX
from neurobench.runner.exec_protocol_v2 import FINISH_SCHEMA, HOST_FIELDS, public_checks
from neurobench.runner.exec_trial import schema
from .common import *

RUNTIME_TEXT = '''Common runtime for every comparison condition.
run_python starts a fresh process; files persist, Python variables do not.
run_project executes output/project/main.py --input inputs --output output in the same controlled Python environment. Deliver project/main.py with these command-line arguments. Only project files are carried into an initially empty output directory for final clean replay.
The runtime module is supplied by this host, including during replay. Its optional run_step(name, inputs, outputs, config, code, reuse=False) accepts a unique step name, lists of workspace-relative input/output paths, JSON-compatible settings, and Python source text for one step. code executes with the caller's variables available. reuse=True records the existing files without executing code. This records actual input/output hashes, settings, execution action and exceptions outside the candidate workspace; it does not infer dependencies or cache validity. Both conditions have this same API.
numpy, scipy, nibabel, pandas, networkx, sklearn and statsmodels are available. CPU numerical/statistical fits within protocol scope are allowed; at most 64 managed LogisticRegression/RidgeClassifier fits per attempt. No neural training, downloads, package installs, external data or network. All source inputs and context are read-only. This is a cooperative execution boundary.
'''

BASE_PROMPT = '''Complete the supplied research task by delivering actual files under output/. The short task is in context/task.md and the public scientific definitions and output schemas are in inputs/protocol.json. context/CONTRACT.json lists required files and input bindings. The optional method library in context/skills/ and its catalog are identical for both conditions.
Use the common run_python and run_project tools to inspect data, create a reusable program and execute it. Each Python call is a fresh process; files persist, variables do not. INPUT_DIR and OUTPUT_DIR are pathlib Paths. Only output/ and scratch/ are writable. The final entrypoint is output/project/main.py --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY and will be replayed in a clean directory. context/RUNTIME.txt documents the shared execution and step-evidence API.
Limited CPU statistical and linear-baseline fits specified by the protocol are allowed. Do not access external data, credentials, other runs, evaluator code or the network, and do not change inputs. Use supported evidence for unsupported records; do not invent data or positive scientific findings.
finish_result(status="complete", summary=...) finalizes and submits existing files. The host fills case_id, input_sha256 and artifact hashes. inspect_outputs shows structural checks, not numerical answers. Passing structural checks ends the attempt and is followed by private evaluation. At most two submissions are allowed; one further submission is available only after structural failure. There are no post-terminal retries.
Budget: 900 seconds, 48 model rounds, 192 tool calls, at most 120 seconds per Python/project execution. Both conditions receive the same limits and remaining-budget feedback.
'''

# Only generic execution mechanics, never task-specific procedures or private expectations.
PYTHON_HOST = r'''
import sys,os,json,hashlib,pathlib,time,types,inspect,runpy,math,csv,shutil
from pathlib import Path
import numpy as np
import scipy
from scipy import ndimage,stats,optimize
import nibabel as nib
from nibabel.processing import resample_from_to
import pandas,networkx,sklearn,statsmodels.api
from sklearn.linear_model import LogisticRegression,RidgeClassifier
WORKSPACE=Path.cwd().resolve();INPUT_DIR=WORKSPACE/'inputs';OUTPUT_DIR=WORKSPACE/'output'
kind,target,trace_path,run_id,fit_remaining=sys.argv[1:6]
fit_remaining=int(fit_remaining);_fits=0
_trace=open(trace_path,'x',encoding='utf-8');_trace_seq=0
_reads=[WORKSPACE,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
_writes=[OUTPUT_DIR,WORKSPACE/'scratch']
def _inside(p,roots): return any(p.is_relative_to(r) for r in roots)
def _audit(event,args):
    if event=='open' and not isinstance(args[0],int):
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1],args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if not _inside(p,_writes if writing else _reads): raise PermissionError('cooperative_workspace_file_boundary')
    elif event in ('os.listdir','os.scandir','os.chdir') and args and args[0] is not None and not isinstance(args[0],int):
        if not _inside(Path(os.fsdecode(args[0])).resolve(),_reads): raise PermissionError('cooperative_workspace_directory_boundary')
    elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.utime'):
        if not _inside(Path(os.fsdecode(args[0])).resolve(),_writes): raise PermissionError('cooperative_workspace_write_boundary')
    elif event in ('os.rename','os.link','os.symlink'):
        if not all(_inside(Path(os.fsdecode(v)).resolve(),_writes) for v in args[:2]): raise PermissionError('cooperative_workspace_move_boundary')
    elif event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.exec','os.spawn','ctypes.dlopen','ctypes.dlsym'):
        raise PermissionError('external_execution_or_network_disabled')
sys.addaudithook(_audit)
def _hash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _event(type,**values):
    global _trace_seq
    _trace_seq+=1
    _trace.write(json.dumps(dict(type=type,run_id=run_id,sequence=_trace_seq,**values),allow_nan=False)+'\n');_trace.flush()
def _relative(p,output=False):
    q=(WORKSPACE/str(p)).resolve()
    if not _inside(q,_writes if output else _reads) or not q.is_relative_to(WORKSPACE): raise PermissionError('step_path_outside_workspace')
    return q.relative_to(WORKSPACE).as_posix(),q
def run_step(name,inputs,outputs,config,code,reuse=False):
    if not isinstance(name,str) or not name or not isinstance(reuse,bool): raise ValueError('invalid_step_arguments')
    ip=[_relative(p) for p in inputs];op=[_relative(p,True) for p in outputs]
    before={s:_hash(p) for s,p in ip};old={s:_hash(p) if p.is_file() else None for s,p in op}
    settings=json.loads(json.dumps(config,allow_nan=False));start=time.monotonic()
    try:
        if not reuse:
            frame=inspect.currentframe().f_back
            scope=dict(frame.f_globals);scope.update(frame.f_locals)
            exec(compile(code,'<recorded-step:'+name+'>','exec'),scope,scope)
        after={s:_hash(p) for s,p in op};unchanged=before=={s:_hash(p) for s,p in ip}
        if not unchanged: raise ValueError('step_modified_its_inputs')
        _event('step',name=name,action='reuse' if reuse else 'execute',input_sha256=before,output_sha256_before=old,output_sha256=after,config=settings,code_sha256=hashlib.sha256(code.encode()).hexdigest(),elapsed_seconds=time.monotonic()-start,success=True)
        return {'success':True,'name':name,'action':'reuse' if reuse else 'execute','output_sha256':after}
    except Exception as e:
        _event('step',name=name,action='reuse' if reuse else 'execute',input_sha256=before,config=settings,success=False,error_type=type(e).__name__)
        raise
runtime=types.ModuleType('runtime');runtime.run_step=run_step;sys.modules['runtime']=runtime
def _wrap_fit(cls):
    original=cls.fit
    def fit(self,*args,**kwargs):
        global _fits
        if _fits>=fit_remaining: raise RuntimeError('managed_estimator_fit_budget_exhausted')
        _fits+=1;_event('managed_fit',estimator=cls.__name__,number=_fits)
        return original(self,*args,**kwargs)
    cls.fit=fit
_wrap_fit(LogisticRegression);_wrap_fit(RidgeClassifier)
def finish_result(status='complete',**extra):
    if set(extra)&{'case_id','input_sha256','artifacts'}: raise ValueError('host_owned_fields')
    c=json.loads((WORKSPACE/'context/CONTRACT.json').read_text(encoding='utf-8'))
    artifacts={p.relative_to(OUTPUT_DIR).as_posix():_hash(p) for p in sorted(OUTPUT_DIR.rglob('*')) if p.is_file() and p!=OUTPUT_DIR/'result.json'}
    value=dict(case_id=c['case_id'],status=status,input_sha256=c['input_sha256'],artifacts=artifacts,**extra)
    (OUTPUT_DIR/'result.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
try:
    script=(WORKSPACE/target).resolve()
    if not script.is_relative_to(WORKSPACE): raise PermissionError('entrypoint_outside_workspace')
    _event('process_start',kind=kind,entrypoint=target,entrypoint_sha256=_hash(script))
    if kind=='project':
        sys.argv=[str(script),'--input',str(INPUT_DIR),'--output',str(OUTPUT_DIR)]
        sys.path.insert(0,str(script.parent));runpy.run_path(str(script),run_name='__main__')
    elif kind=='python':
        exec(compile(script.read_text(encoding='utf-8'),str(script),'exec'),globals(),globals())
    else: raise ValueError('unknown_process_kind')
    _event('process_end',success=True,managed_fits=_fits)
except SystemExit as e:
    _event('process_end',success=e.code in (None,0),error_type=type(e).__name__,managed_fits=_fits)
    if e.code not in (None,0):raise
except BaseException as e:
    _event('process_end',success=False,error_type=type(e).__name__,managed_fits=_fits)
    raise
finally: _trace.close()
'''


def trace_events(directory):
    result=[]
    for path in sorted(Path(directory).glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip(): result.append(json.loads(line))
    return result


def prepare_workspace(case,workspace,library):
    workspace=Path(workspace);workspace.mkdir(parents=True,exist_ok=False)
    shutil.copytree(case.root/'inputs',workspace/'inputs')
    shutil.copytree(library,workspace/'context')
    shutil.copy2(case.root/'task.md',workspace/'context/task.md')
    shutil.copy2(case.root/'CONTRACT.json',workspace/'context/CONTRACT.json')
    (workspace/'context/RUNTIME.txt').write_text(RUNTIME_TEXT,encoding='utf-8')
    (workspace/'output').mkdir();(workspace/'scratch').mkdir()
    (workspace/'python_host.py').write_text(PYTHON_HOST,encoding='utf-8')
    return workspace


def replay(case,output,target,library,*,external_perturbation=False):
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    replay_case=case
    if external_perturbation:
        clone=target/'case';shutil.copytree(case.root,clone)
        rows=read_csv(clone/'inputs/cohort.csv');external={r['participant_id'] for r in rows if r['split']=='external'}
        features=read_csv(clone/'inputs/features.csv')
        for r in features:
            if r['participant_id'] in external:
                for k in r:
                    if k!='participant_id':r[k]=float(r[k])*3+7
        write_csv(clone/'inputs/features.csv',features,list(features[0]))
        contract={**case.contract,'input_sha256':file_map(clone/'inputs')};write_json(clone/'CONTRACT.json',contract)
        replay_case=Case(clone,contract)
    work=prepare_workspace(replay_case,target/'workspace',library)
    shutil.copytree(Path(output)/'project',work/'output/project')
    run=run_process(work,target/'processes')
    if run['success']:finalize(replay_case,work/'output')
    from .oracle import evaluate
    result=evaluate(replay_case,work/'output',target/'processes') if run['success'] else {'passed':False,'verdict':'FAIL','errors':['entrypoint_execution_failed']}
    if external_perturbation and run['success']:
        import numpy as np
        from .semantic import model_state_equivalent
        tolerance=read_json(replay_case.root/'inputs/protocol.json')['numerical_tolerance']
        invariant=model_state_equivalent(Path(output)/'model/state.npz',work/'output/model/state.npz',tolerance['absolute'],tolerance['relative'])
        result['fit_state_invariant_to_external_values']=invariant
        result['passed']=result['passed'] and invariant
        if not invariant:result['errors'].append('external_values_changed_fit_state')
    record={'run':run,'evaluation':result,'passed':result['passed'],'external_perturbation':external_perturbation}
    write_json(target/'REPLAY.json',record)
    return record


def run_process(workspace, private, *, kind='project', target='output/project/main.py', timeout=120):
    workspace,private=Path(workspace).resolve(),Path(private).resolve()
    private.mkdir(parents=True,exist_ok=True);traces=private/'trusted';traces.mkdir(exist_ok=True)
    completed_fits=sum(e['type']=='managed_fit' for e in trace_events(traces))
    sequence=len(list(traces.glob('*.jsonl')))+1;run_id=uuid.uuid4().hex
    trace=traces/f'{sequence:04d}.jsonl';host=workspace/'python_host.py'
    host.write_text(PYTHON_HOST,encoding='utf-8')
    clean={k:v for k,v in os.environ.items() if k.upper() in ('WINDIR','SYSTEMROOT','SYSTEMDRIVE','TEMP','TMP','COMSPEC')}
    clean.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(workspace/'scratch/matplotlib'))
    command=[sys.executable,'-I','-B','-X','utf8',str(host),kind,target,str(trace),run_id,str(max(0,64-completed_fits))]
    start=time.monotonic();timed_out=False;peak_rss=0
    stdout_path=private/f'{sequence:04d}.stdout.bin';stderr_path=private/f'{sequence:04d}.stderr.bin'
    with stdout_path.open('wb') as out,stderr_path.open('wb') as err:
        child=subprocess.Popen(command,cwd=workspace,env=clean,stdout=out,stderr=err)
        while child.poll() is None:
            try:peak_rss=max(peak_rss,psutil.Process(child.pid).memory_info().rss)
            except psutil.Error:pass
            if time.monotonic()-start>=max(.01,timeout):
                child.kill();child.wait();timed_out=True;break
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        returncode=None if timed_out else child.returncode
    stdout,stderr=stdout_path.read_bytes(),stderr_path.read_bytes()
    record={'run_id':run_id,'kind':kind,'target':target,'target_sha256':sha(workspace/target),'host_sha256':sha(host),'timeout_seconds':timeout,
        'returncode':returncode,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes_sampled':peak_rss,'trace_sha256':sha(trace) if trace.exists() else None}
    write_json(private/f'{sequence:04d}.json',record)
    return {**record,'success':returncode==0,'stdout':stdout.decode('utf-8',errors='replace')[-16000:],'stderr':stderr.decode('utf-8',errors='replace')[-16000:]}


COMMON_SCHEMAS = [
    schema('read_file','Read a public UTF-8 file from inputs/ or context/. Binary files can be read through Python.',{'path':{'type':'string'},'offset':{'type':'integer','minimum':0}},('path',)),
    schema('run_python','Execute Python in a fresh controlled process. INPUT_DIR/OUTPUT_DIR are Paths; only output/ and scratch/ are writable. CPU numerical/statistical work is allowed within protocol scope.',{'code':{'type':'string'}},('code',)),
    schema('run_project','Execute delivered output/project/main.py with --input and --output arguments. Records actual execution and any runtime.run_step calls.'),
    schema('inspect_outputs','Inspect actual output files, missing required paths and manifest structure. No scientific expected values.'),
    FINISH_SCHEMA,
    schema('submit','Freeze current output. Structural failure permits one more submission within the original budget.'),
]


class HardTools:
    def __init__(self,case,workspace,attempt,arm,deadline,started):
        self.case,self.workspace,self.attempt,self.arm=case,Path(workspace),Path(attempt),arm
        self.deadline,self.started=deadline,started;self.budget=COMPLEX
        self.output=self.workspace/'output';self.submissions=[];self.terminal=False;self.tool_count=0;self.round_index=0
        self.processes=self.attempt/'processes';self.processes.mkdir()
        self.nr=None
        if arm=='NR_ON':
            from neurobench.runner.high_tools import TaskTools
            self.nr=TaskTools(self.workspace,(case.root/'task.md').read_text(encoding='utf-8')+'\n'+json.dumps(case.contract,ensure_ascii=False),enabled=True)

    def schemas(self):
        return list(COMMON_SCHEMAS)+([s for s in self.nr.schemas() if s['name'] in SKILL_TOOL_NAMES] if self.nr else [])

    def feedback(self):
        return {'model_round':self.round_index,'model_rounds_remaining_after_current':48-self.round_index,
            'tool_calls_remaining':192-self.tool_count,'seconds_remaining':round(max(0,self.deadline-time.monotonic()),3),
            'submissions_remaining':2-len(self.submissions),'closeout_recommended':self.round_index>=46}

    def begin_round(self,index):
        if isinstance(index,bool) or not self.round_index<index<=48: raise ValueError('round_budget_exceeded')
        self.round_index=index;return self.feedback()

    def submit(self):
        if self.terminal or len(self.submissions)>=2: return {'success':False,'error':'submission_budget_exhausted'}
        if time.monotonic()>=self.deadline: return {'success':False,'error':'wall_budget_reached'}
        number=len(self.submissions)+1;target=self.attempt/'submissions'/str(number)
        shutil.copytree(self.output,target/'output');shutil.copytree(self.processes,target/'processes')
        public=public_checks(self.case.contract,target/'output')
        record={'number':number,'elapsed_seconds':time.monotonic()-self.started,'at_utc':utc(),'model_round':self.round_index,'tool_calls':self.tool_count,
            'output':str(target/'output'),'processes':str(target/'processes'),'output_hashes':file_map(target/'output'),'process_hashes':file_map(target/'processes'),'public_diagnostics':public}
        write_json(target/'SUBMISSION.json',record);self.submissions.append(record)
        self.terminal=public['public_check_pass'] or number==2
        return {'submission_number':number,'terminal':self.terminal,**public}

    def call(self,name,args):
        if self.terminal:return {'success':False,'error':'already_terminal'}
        if self.tool_count>=192 or time.monotonic()>=self.deadline:return {'success':False,'error':'tool_or_wall_budget_reached'}
        self.tool_count+=1
        try:
            if not isinstance(args,dict): raise ValueError('arguments_not_object')
            if name=='read_file':
                rel=args['path']
                if not Path(rel).parts or Path(rel).parts[0] not in ('context','inputs'): raise ValueError('not_public_context')
                path=contained(self.workspace,rel);offset=max(0,int(args.get('offset',0)));body=path.read_text(encoding='utf-8-sig')
                value={'success':True,'content':body[offset:offset+18000],'total_chars':len(body),'next_offset':offset+18000 if offset+18000<len(body) else None,'sha256':sha(path)}
            elif name in ('run_python','run_project'):
                kind='project' if name=='run_project' else 'python';target='output/project/main.py'
                if name=='run_python':
                    code=args['code']
                    if not isinstance(code,str) or len(code)>150000:raise ValueError('invalid_code_length')
                    ast.parse(code);target=f'scratch/call_{self.tool_count:03d}.py';(self.workspace/target).write_text(code,encoding='utf-8')
                value=run_process(self.workspace,self.processes,kind=kind,target=target,timeout=min(120,max(.01,self.deadline-time.monotonic())))
                value['public_diagnostics']=public_checks(self.case.contract,self.output)
            elif name=='inspect_outputs':value=public_checks(self.case.contract,self.output)
            elif name=='finish_result':
                if set(args)&HOST_FIELDS:raise ValueError('host_owned_fields')
                if set(args)-{'status','summary','reason','evidence','metadata'}:raise ValueError('unknown_finish_fields')
                if args.get('status') not in ('complete','blocked'):raise ValueError('invalid_finish_status')
                artifacts=file_map(self.output);artifacts.pop('result.json',None)
                write_json(self.output/'result.json',{'case_id':self.case.id,'input_sha256':self.case.contract['input_sha256'],'artifacts':artifacts,**args})
                value=self.submit()
            elif name=='submit':value=self.submit()
            elif self.nr and name in SKILL_TOOL_NAMES:value=self.nr.call(name,args)
            else:raise ValueError('tool_not_exposed')
        except Exception as e:
            value={'success':False,'error_type':type(e).__name__,'error':str(e)[:1600]}
        return {**value,'budget':self.feedback()}
