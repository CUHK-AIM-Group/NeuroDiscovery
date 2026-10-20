"""Finite versioned high pilot: prepare, compatibility, run, inspect.

No generation on import. One initial attempt per immutable slot, no UNKNOWN
resend, one shared-account queue per model route, and a six-hour run window.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import queue
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from neurobench.runner.artifacts import EventLog, file_sha256, freeze_bytes, freeze_text, json_text
from neurobench.runner.high_tools import TaskTools, NR_NAMES, prompt
from neurobench.runner.high_channels import anthropic_round, chat_round, credential, responses_round
from neurobench.runner.subscription_direct import get_access_record
from neurobench.runner.high_responses_observer import ResponsesObserver
from core.harness_validation import validate_proposal

REPO = Path(os.environ.get('NEUROCLAW_BENCH_REPO') or Path(__file__).resolve().parents[2])
RUN = Path(os.environ.get('NEURORUNTIME_HIGH_RUN') or REPO / 'tmp/benchmark_high_comparison_20261013_r1')
RUNTIME = RUN / 'runtime'
DEADLINE_SECONDS = 1500
STAGE_SECONDS = 21600
TOKEN_STOP = 30000000
USER_REQUEST = '可以，开始做吧'


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def atomic(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.writing')
    temporary.write_text(json_text(obj), encoding='utf-8')
    os.replace(temporary, p)


def _con():
    c = sqlite3.connect(RUN / 'state.sqlite3', timeout=30)
    c.row_factory = sqlite3.Row
    return c


def _sources():
    old = read(REPO / 'tmp/benchmark_high_pilot_20261008/baseline_v1/manifest.json')['source_sha256']
    # Historical manifests retain their original paths; bind current code at its new location.
    names = {('neurobench/' + ('tests/' if Path(n).name.startswith('test_') else 'runner/') + n[len('bench/'):]) if n.startswith('bench/') else n for n in old} | {
        'neurobench/__init__.py', 'neurobench/runner/__init__.py',
        'core/harness_validation.py', 'neurobench/runner/subscription_direct.py', 'neurobench/runner/high_tools.py', 'neurobench/runner/high_mcp.py',
        'neurobench/runner/high_channels.py', 'neurobench/runner/high_experiment.py', 'core/test_harness_validation.py', 'neurobench/tests/test_subscription_direct.py',
        'neurobench/runner/high_responses_observer.py', 'neurobench/tests/test_high_responses_observer.py', 'neurobench/tests/test_high_experiment.py',
    }
    return sorted(n for n in names if (REPO / n).is_file())


def prepare():
    if (RUN / 'EXECUTION_GATE.json').exists():
        raise RuntimeError('already_prepared_inspect_existing_revision')
    plan_path = REPO / 'docs/NEURORUNTIME_HIGH_COMPARISON_PLAN_20261013.json'
    plan = read(plan_path)
    conditions = [c for c in plan['condition_matrix'] if c['core']]
    sources = {}
    for name in _sources():
        freeze_bytes(RUNTIME / name, (REPO / name).read_bytes())
        sources[name] = file_sha256(RUNTIME / name)
    auth = REPO / 'tmp/chatgpt_plan_direct_20261009'
    freeze_bytes(RUNTIME / 'tmp/chatgpt_plan_direct_20261009/chatgpt_plan.py', (auth / 'chatgpt_plan.py').read_bytes())
    shutil.copytree(auth / '.dependencies', RUNTIME / 'tmp/chatgpt_plan_direct_20261009/.dependencies', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=False)
    material = RUN / 'materials/skills'
    shutil.copytree(REPO / 'skills', material, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'node_modules', '.git'), dirs_exist_ok=False)
    material_hashes = {str(p.relative_to(RUN)).replace('\\', '/'): file_sha256(p) for p in material.rglob('*') if p.is_file()}
    from core.harness_skills import default_capability
    capability=default_capability(material,include_cwd=False)
    freeze_text(RUN/'materials/skill_catalog.json',json_text([{'name':r.name,'description':r.description,'path':'context/skills/'+str(r.skill_md.relative_to(material)).replace('\\','/')} for r in capability.catalog()]))
    material_hashes['materials/skill_catalog.json']=file_sha256(RUN/'materials/skill_catalog.json')
    tasks = plan['selection']['tasks']
    for row in tasks:
        task_path = Path(row['task_path'])
        if task_path.parts[0] == 'neurobench' and task_path.parts[1].startswith('T'):
            task_path = Path('neurobench/tasks').joinpath(*task_path.parts[1:])
        text = (REPO / task_path).read_text(encoding='utf-8')
        assert hashlib.sha256(text.encode()).hexdigest() == row['source_sha256']
        freeze_text(RUN / 'tasks' / (row['case_id'] + '.md'), text)
    manifest = {'revision': RUN.name, 'created_at_utc': now(),
                'tasks': tasks, 'conditions': conditions, 'source_sha256': sources, 'material_sha256': material_hashes,
                'proposal_only': True, 'response_contract': ['Solution Thinking', 'Commands Or Code'],
                'requested_high_all_arms': True, 'primary_outputs': 240, 'compatibility_extra_outputs_max': 6,
                'compatibility_initial_attempts_per_route': 1, 'automatic_recovery': 0,
                'per_task_total_wall_seconds': DEADLINE_SECONDS, 'stage_wall_seconds': STAGE_SECONDS,
                'observed_generation_token_stop': TOKEN_STOP, 'initial_inflight_per_account': 1,
                'direct_tool_round_limit': 12, 'feedback_repairs_after_final_max': 1,
                'unsupported_subscription_output_cap_disclosed': True, 'strict_token_matching_claim': False,
                'task_order': 'three mandatory smoke witnesses first, then preselected tasks; arms rotate by task hash',
                'same_model_backend_claim': 'pending actual compatibility receipts',
                'unknown_usage_stops_account': True, 'unknown_retries': False}
    freeze_text(RUN / 'manifest.json', json_text(manifest))
    con = _con()
    con.execute('CREATE TABLE slots(case_id TEXT, condition_id TEXT, model_block TEXT, arm TEXT, status TEXT NOT NULL, receipt_path TEXT, PRIMARY KEY(case_id,condition_id))')
    con.execute('CREATE TABLE compatibility(model_block TEXT,driver TEXT,status TEXT,receipt_path TEXT,PRIMARY KEY(model_block,driver))')
    con.executemany('INSERT INTO slots VALUES(?,?,?,?,?,NULL)', [(t['case_id'], c['condition_id'], c['model_block'], c['arm'], 'PENDING') for t in tasks for c in conditions])
    con.commit(); con.close()
    gate = {'created_at_utc': now(), 'user_request': USER_REQUEST, 'current_user_run_token': 'e3f40c00-187a-4cde-adec-077e17630361',
            'manifest_sha256': file_sha256(RUN / 'manifest.json'), 'plan_sha256': file_sha256(plan_path),
            'source_snapshot_sha256': sources, 'generation_max_primary_attempts': 240, 'compatibility_max_attempts': 6,
            'recovery_budget': 0, 'default_primary_state': 'PENDING_NOT_DISPATCHED'}
    freeze_text(RUN / 'EXECUTION_GATE.json', json_text(gate))
    bootstrap = "import os,sys\nfrom pathlib import Path\np=Path(__file__).resolve().parent\nos.environ['NEUROCLAW_BENCH_REPO']=" + repr(str(REPO)) + "\nos.environ['NEURORUNTIME_HIGH_RUN']=str(p)\nsys.path.insert(0,str(p/'runtime'))\nfrom neurobench.runner.high_experiment import main\nmain()\n"
    freeze_text(RUN / 'run.py', bootstrap)
    print(json.dumps({'prepared': True, 'slots': 240, 'source_files': len(sources), 'method_files': len(material_hashes), 'api_calls': 0}), flush=True)


def check():
    g = read(RUN / 'EXECUTION_GATE.json'); m = read(RUN / 'manifest.json')
    assert file_sha256(RUN / 'manifest.json') == g['manifest_sha256']
    for name, digest in g['source_snapshot_sha256'].items():
        assert file_sha256(RUNTIME / name) == digest, name
    return m


def workspace(case, condition, task):
    root = RUN / 'workspaces' / condition / case
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(RUN / 'materials/skills', root / 'context/skills')
    freeze_bytes(root/'context/skill_catalog.json',(RUN/'materials/skill_catalog.json').read_bytes())
    freeze_text(root / 'context/task.md', task)
    freeze_text(root / 'context/README.md', 'These are generic method documents shared by all conditions. Task outputs belong in proposals/. They contain no benchmark answers or grader oracle. Static checks do not execute the research workflow.\n')
    return root


def _sum_usage(rows):
    return {'input_tokens': sum(r['input_tokens'] for r in rows), 'output_tokens': sum(r['output_tokens'] for r in rows),
            'total_tokens': sum(r['total_tokens'] for r in rows), 'usage_complete': bool(rows) and all(r['usage_complete'] for r in rows), 'native_calls_with_usage': len(rows)}


def direct(task, model, enabled, work, attempt, *, compatibility=False, timeout_seconds=DEADLINE_SECONDS, route=None):
    deadline = time.monotonic() + timeout_seconds
    provider = {'gpt-5.6-sol': 'responses', 'claude-opus-4-8': 'anthropic', 'deepseek-v4-pro': 'deepseek', 'glm-5.3': 'ollama'}[model]
    tools = TaskTools(work, task, enabled=enabled)
    instruction = prompt(task, enabled)
    if compatibility:
        tool = 'describe_task_contract' if enabled else 'read_context_file with path context/task.md'
        instruction += '\nFor this simple compatibility fixture, call ' + tool + ' exactly once. Do not search or read method documents in this fixture. Then immediately return the two required sections with NEUROCLAW_TOOL_ROUNDTRIP_OK in the final answer. This special fixture instruction takes precedence over generic method-loading guidance.\n'
    history = [{'role': 'user', 'content': instruction}]
    usage_rows, tool_calls, drafts = [], [], []
    status, text, error, repair = 'UNKNOWN', '', '', False
    calls = 0
    log = EventLog(attempt / 'events.jsonl')
    try:
        for step in range(12):
            calls += 1
            if provider == 'responses':
                result, usage = responses_round(history, tools.schemas(), log, deadline, route=route)
                history.extend(result['output'])
            elif provider == 'anthropic':
                result, usage = anthropic_round(history, tools.schemas(), log, deadline)
                history.append(result['message'])
            else:
                result, usage = chat_round(provider, history, tools.schemas(), log, deadline)
                history.append(result['message'])
            usage_rows.append(usage)
            log.write('native_usage', payload=usage)
            if not usage['usage_complete']:
                raise RuntimeError('native_usage_incomplete_stop_account')
            if result['calls']:
                outputs = []
                for call in result['calls']:
                    args = call['arguments'] if isinstance(call['arguments'], dict) else json.loads(call['arguments'])
                    log.write('tool_request', name=call['name'], arguments=args, call_id=call['call_id'])
                    output = tools.call(call['name'], args)
                    log.write('tool_result', name=call['name'], result=output, call_id=call['call_id'])
                    tool_calls.append({'name': call['name'], 'arguments': args, 'result': output})
                    payload = json.dumps(output, ensure_ascii=False)
                    if provider == 'responses': history.append({'type': 'function_call_output', 'call_id': call['call_id'], 'output': payload})
                    elif provider == 'anthropic': outputs.append({'type': 'tool_result', 'tool_use_id': call['call_id'], 'content': payload})
                    else:
                        history.append({'role': 'tool', 'tool_call_id': call['call_id'], 'name': call['name'], 'tool_name': call['name'], 'content': payload} if provider=='ollama' else {'role': 'tool', 'tool_call_id': call['call_id'], 'name': call['name'], 'content': payload})
                if outputs: history.append({'role': 'user', 'content': outputs})
                continue
            text = result['text']; drafts.append(text)
            validation = validate_proposal(task, text)
            log.write('offline_proposal_diagnostics', result=validation, model_visible=enabled)
            if enabled and not repair and validation['blocking_error_count']:
                repair = True
                history.append({'role': 'user', 'content': 'One bounded diagnostic feedback repair is available inside the original deadline. Correct the detected static errors, inspect the review hints, and return the entire final answer with the two required sections. Do not claim workflow execution. Diagnostics: ' + json.dumps(validation, ensure_ascii=False)})
                continue
            status = 'COMPLETE'; break
        else:
            status, error = 'BUDGET_STOP', 'direct_tool_round_limit'
    except Exception as e:
        error = str(e) if isinstance(e, (RuntimeError, ValueError, TimeoutError)) else type(e).__name__
        status = 'FAILED_KNOWN' if error.startswith(('native_http_', 'anthropic_native_stream_error', 'deepseek_noncomplete_', 'ollama_noncomplete_', 'native_usage_incomplete')) else 'UNKNOWN'
        log.write('attempt_error', error_type=type(e).__name__, error_code=error, status=status)
    finally:
        log.close()
    return {'status': status, 'error': error, 'response': text, 'native_request_count': calls,
            'usage': _sum_usage(usage_rows), 'native_usage_records': usage_rows, 'tool_calls': tool_calls,
            'drafts': drafts, 'feedback_repair_used': repair, 'reasoning_requested': 'high',
            'provider': route['provider'] if route else provider, 'responses_route': route,
            'native_high_echo': provider == 'responses' and status == 'COMPLETE',
            'workflow_executed': False}


def _capture(command, text, env, cwd, directory, timeout, secrets):
    outputs = {'stdout': [], 'stderr': []}
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env, text=True, encoding='utf-8', errors='replace')
    freeze_text(directory / 'launch.json', json_text({'command': command, 'cwd': str(cwd), 'pid': process.pid, 'timeout_seconds': timeout, 'started_at_utc': now(), 'environment_values_not_logged': True}))
    def collect(pipe, name):
        with (directory / (name + '.jsonl')).open('x', encoding='utf-8') as stream:
            for line in pipe:
                for secret in secrets:
                    if secret: line = line.replace(secret, '[REDACTED]')
                outputs[name].append(line); stream.write(line); stream.flush()
    threads = [threading.Thread(target=collect, args=(getattr(process,n),n), daemon=True) for n in outputs]
    for t in threads: t.start()
    process.stdin.write(text); process.stdin.close()
    unknown = False
    try: process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        unknown = True
        if os.name == 'nt': subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else: process.kill()
        process.wait(timeout=10)
    for t in threads: t.join(timeout=5)
    return process.returncode, ''.join(outputs['stdout']), ''.join(outputs['stderr']), unknown, time.monotonic()-started


def native(task, model, enabled, work, attempt, *, compatibility=False, timeout_seconds=DEADLINE_SECONDS, route=None):
    from neurobench.runner.cli_drivers import find_codex_cli, find_claude_cli, parse_cli_events, parse_claude_result
    deadline=time.monotonic()+timeout_seconds
    env = {k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('TOKEN', 'API_KEY', 'AUTHORIZATION', 'ANTHROPIC_', 'MAX_THINKING', 'RUST_LOG'))}
    env['PYTHONPATH'] = str(RUNTIME); env['NO_COLOR'] = '1'
    mcp = {'mcpServers': {}}
    if enabled:
        mcp['mcpServers']['neuro_runtime'] = {'command': sys.executable, 'args': ['-X','utf8','-m','neurobench.runner.high_mcp'], 'env': {
            'PYTHONPATH': str(RUNTIME), 'NEURORUNTIME_MCP_WORKSPACE': str(work), 'NEURORUNTIME_MCP_SKILLS_DIR': str(work/'context/skills'),
            'NEURORUNTIME_MCP_TOOLS': ','.join(NR_NAMES), 'NEURORUNTIME_MCP_TASK_FILE': str(work/'context/task.md'),
            'NEURORUNTIME_MCP_TASK_SHA256': hashlib.sha256(task.encode()).hexdigest(), 'NEURORUNTIME_MCP_EVENT_LOG': str(attempt/'mcp_events.jsonl')}}
    freeze_text(attempt/'mcp.json', json_text(mcp))
    instruction = prompt(task, enabled)
    instruction += '\nThe absolute task workspace is ' + str(work) + '. Use this exact working directory for native commands, or read context files with absolute paths below it. The supplied task is ' + str(work/'context/task.md') + '.\n'
    if compatibility:
        instruction += '\nThis is a pure transport fixture. This instruction overrides generic method loading, proposal drafting, static validation and research instructions above. Read only the absolute supplied task path with a native file tool exactly once. Do not read the skill catalog or any method document, inspect other paths, validate proposals, write files or execute the proposed code.\n'
        if enabled:
            instruction += '\nCall the NeuroRuntime describe_task_contract MCP tool exactly once and inspect its returned contract.\n'
        instruction += '\nImmediately finish with exactly two sections: ## Solution Thinking containing only NEUROCLAW_NATIVE_TOOL_OK, then ## Commands Or Code containing only a Python fence with print("NEUROCLAW_NATIVE_TOOL_OK"). Do not run that print command. No additional explanation.\n'
    if model == 'gpt-5.6-sol':
        if route is None:
            key = get_access_record(minimum_valid_seconds=timeout_seconds+120)['access_token']
            upstream, credential_env = 'https://api.openai.com/v1/responses', 'NEUROCLAW_PLAN_ACCESS_TOKEN'
            provider_name = 'NeuroClaw authorized ChatGPT plan'
        else:
            from neurobench.runner.high_gpt_route import checked_route, route_key
            route = checked_route(route)
            key = route_key(route)
            upstream, credential_env = route['endpoint'], 'NEUROCLAW_GPT_ROUTE_API_KEY'
            provider_name = 'NeuroClaw user-authorized Xnova API'
        local_token = secrets.token_urlsafe(32) if route else key
        observer=ResponsesObserver(attempt,local_token,deadline,upstream_url=upstream,upstream_token=key)
        home = attempt/'codex_home'; home.mkdir()
        config = ['model_provider = "neuroclaw_plan"', 'model = "gpt-5.6-sol"', 'model_reasoning_effort = "high"', 'approval_policy = "never"',
                  '[windows]', 'sandbox = "unelevated"',
                  '[model_providers.neuroclaw_plan]', 'name = '+json.dumps(provider_name), 'base_url = '+json.dumps(observer.base_url),
                  'wire_api = "responses"', 'env_key = '+json.dumps(credential_env), 'request_max_retries = 0', 'stream_max_retries = 0']
        if enabled:
            row = mcp['mcpServers']['neuro_runtime']; config += ['', '[mcp_servers.neuro_runtime]', 'command = '+json.dumps(row['command']), 'args = '+json.dumps(row['args']), 'default_tools_approval_mode = "approve"', '[mcp_servers.neuro_runtime.env]']
            config += [k+' = '+json.dumps(v) for k,v in row['env'].items()]
        freeze_text(home/'config.toml', '\n'.join(config)+'\n')
        env['CODEX_HOME'] = str(home); env[credential_env] = local_token
        env['NO_PROXY']=','.join(filter(None,[env.get('NO_PROXY',''),'127.0.0.1','localhost']))
        command = [os.environ.get('NEUROCLAW_NODE') or 'node', find_codex_cli(), '--no-daemon','exec', '--ephemeral','--json','--color','never','--ignore-rules','--skip-git-repo-check','--model','gpt-5.6-sol','--config','model_reasoning_effort="high"','--sandbox','workspace-write','--disable','multi_agent','--disable','unbounded_connection_retries','--disable','enable_request_compression','--cd',str(work),'-']
        parser = parse_cli_events
    else:
        observer=None
        key = credential('nvidia'); home = attempt/'claude_home'; home.mkdir()
        for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
            env.pop(name,None)
        settings = {'effortLevel':'high','alwaysThinkingEnabled':True,'permissions':{'allow':['Read','Write','Edit','Glob','Grep','mcp__neuro_runtime__*','Bash(python *)','Bash("'+sys.executable+'" *)']}}
        freeze_text(home/'settings.json', json_text(settings))
        env.update(CLAUDE_CONFIG_DIR=str(home), ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL='https://inference-api.nvidia.com', ANTHROPIC_MODEL='azure/anthropic/claude-opus-4-8', CLAUDE_CODE_EFFORT_LEVEL='high', CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT='1')
        command = [find_claude_cli(),'-p','--output-format','stream-json','--verbose','--model','azure/anthropic/claude-opus-4-8','--effort','high','--setting-sources','','--settings',str(home/'settings.json'),'--strict-mcp-config','--mcp-config',str(attempt/'mcp.json'),'--no-session-persistence','--disallowedTools','Agent,Task']
        parser = parse_claude_result
    with observer if observer is not None else contextlib.nullcontext():
        code, raw, stderr, unknown, elapsed = _capture(command, instruction, env, work, attempt, max(1,deadline-time.monotonic()),
                                                      [key, local_token] if model == 'gpt-5.6-sol' else [key])
    summary = parser(raw)
    wire=observer.summary() if observer is not None else None
    complete = bool(not unknown and code == 0 and summary.get('terminal') and not summary.get('error') and summary.get('response','').strip())
    usage = {'input_tokens':summary.get('input_tokens',0),'output_tokens':summary.get('output_tokens',0),'cached_input_tokens':summary.get('cached_input_tokens'),
             'total_tokens':summary.get('input_tokens',0)+summary.get('output_tokens',0),'usage_complete':summary.get('usage_complete',False)}
    network_unknown=any(t in str(summary.get('error','')).lower() for t in ('connection','econnreset','timed out','stream','timeout'))
    if wire is not None and not wire['all_requests_completed']:
        complete=False;network_unknown=True
    return {'status':'COMPLETE' if complete else 'UNKNOWN' if unknown or network_unknown else 'FAILED_KNOWN','error':summary.get('error') or ('native_cli_exit_'+str(code) if code else ''),
            'response':summary.get('response',''),'usage':usage,'native_usage_records':summary.get('usage_records'), 'tool_calls':summary.get('tool_events'),
            'reasoning_requested':'high','native_high_echo':wire['native_high_echo'] if wire else False,'native_call_count_unavailable':wire is None,'native_wire':wire,'model_ids':wire['native_model_ids'] if wire else summary.get('model_ids'),
            'responses_route':route,
            'workflow_executed':False,'elapsed_seconds':elapsed,'native_file_shell_tools_enabled':True,'native_delegation_disabled_in_primary':True}


def execute(case, condition, task, *, compatibility=False, timeout_seconds=DEADLINE_SECONDS):
    identity = condition['condition_id']; attempt = RUN/('compatibility' if compatibility else 'attempts')/identity/case
    attempt.mkdir(parents=True,exist_ok=False)
    freeze_text(attempt/'ALLOCATION.json', json_text({'case_id':case,'condition':identity,'started_at_utc':now(),'user_request':USER_REQUEST,'automatic_retries':0}))
    work = workspace(case,identity,task)
    start = time.monotonic()
    try:
        f = native if condition['arm'] in 'CD' else direct
        result = f(task,condition['model_block'],condition['arm'] in 'BD',work,attempt,compatibility=compatibility,timeout_seconds=timeout_seconds,route=condition.get('responses_route'))
    except Exception as e:
        result = {'status':'UNKNOWN','error_type':type(e).__name__,'error':str(e) if isinstance(e,(ValueError,RuntimeError)) else type(e).__name__,'response':'','usage':{'usage_complete':False,'total_tokens':0}}
    result['elapsed_seconds'] = time.monotonic()-start
    result['finished_at_utc'] = now(); result['condition'] = condition
    result['task_sha256'] = hashlib.sha256(task.encode()).hexdigest()
    result['work_directory'] = str(work)
    if result.get('response'):
        freeze_text(attempt/'native_final.md', result.pop('response'))
        result['native_final_path'] = str(attempt/'native_final.md'); result['native_final_sha256'] = file_sha256(attempt/'native_final.md')
        freeze_text(attempt/'DIAGNOSTICS.json', json_text(validate_proposal(task,(attempt/'native_final.md').read_text(encoding='utf-8'))))
    freeze_text(attempt/'RECEIPT.json',json_text(result))
    return result, str(attempt/'RECEIPT.json')


def compatibility():
    m=check(); con=_con()
    if con.execute('SELECT count(*) FROM compatibility').fetchone()[0]: raise RuntimeError('compatibility_already_allocated_inspect_instead')
    jobs=[c for c in m['conditions'] if c['arm']=='B' or c['arm']=='C']
    assert len(jobs)==6
    for c in jobs: con.execute('INSERT INTO compatibility VALUES(?,?,?,NULL)',(c['model_block'],c['driver'],'ALLOCATED'))
    con.commit(); con.close()
    task='Describe a minimal BIDS filename-validation proposal for one supplied subject. This new compatibility fixture tests one local tool result and a final answer, not scientific workflow execution. Do not search methods, execute research commands or access datasets.'
    for c in jobs:
        if c['model_block']=='glm-5.3' and not read(RUN/'OLLAMA_MODEL_METADATA.json').get('named_high_verified'):
            with _con() as con: con.execute('UPDATE compatibility SET status=? WHERE model_block=? AND driver=?',('HELD_HIGH_METADATA_UNVERIFIED',c['model_block'],c['driver']))
            print(json.dumps({'compatibility':c['condition_id'],'status':'HELD_HIGH_METADATA_UNVERIFIED','inference_dispatched':False}),flush=True)
            continue
        if c['model_block']=='claude-opus-4-8' and not read(REPO/'tmp/benchmark_high_comparison_20261013_r1/NVIDIA_FILE_MODEL_CHECK.json').get('claude_4_8_available'):
            with _con() as con: con.execute('UPDATE compatibility SET status=? WHERE model_block=? AND driver=?',('HELD_CREDENTIAL_HTTP401',c['model_block'],c['driver']))
            print(json.dumps({'compatibility':c['condition_id'],'status':'HELD_CREDENTIAL_HTTP401','inference_dispatched':False}),flush=True)
            continue
        result,path=execute('COMPAT2',c,task,compatibility=True,timeout_seconds=300)
        final=Path(result['native_final_path']).read_text(encoding='utf-8') if result.get('native_final_path') else ''
        tool_used=(any(t.get('name')=='describe_task_contract' for t in result.get('tool_calls',[])) if c['arm']=='B' else bool(result.get('tool_calls')))
        ready=result['status']=='COMPLETE' and result['usage']['usage_complete'] and tool_used and ('NEUROCLAW_TOOL_ROUNDTRIP_OK' in final if c['arm']=='B' else 'NEUROCLAW_NATIVE_TOOL_OK' in final)
        with _con() as con: con.execute('UPDATE compatibility SET status=?,receipt_path=? WHERE model_block=? AND driver=?',('READY' if ready else 'HELD_NOT_READY',path,c['model_block'],c['driver']))
        print(json.dumps({'compatibility':c['condition_id'],'status':'READY' if ready else 'HELD_NOT_READY','receipt':path}),flush=True)
    inspect()


def _run_account(model, jobs, deadline, budget):
    for case,c in jobs:
        # Do not launch a new task without its full common time allowance.
        if time.monotonic()+DEADLINE_SECONDS>=deadline: break
        with budget['lock']:
            if budget['tokens']>=TOKEN_STOP: break
        with _con() as con:
            changed=con.execute("UPDATE slots SET status='ALLOCATED' WHERE case_id=? AND condition_id=? AND status='PENDING'",(case,c['condition_id'])).rowcount
        if changed!=1: continue
        text=(RUN/'tasks'/(case+'.md')).read_text(encoding='utf-8')
        result,path=execute(case,c,text)
        with _con() as con: con.execute('UPDATE slots SET status=?,receipt_path=? WHERE case_id=? AND condition_id=?',(result['status'],path,case,c['condition_id']))
        with budget['lock']:
            budget['tokens']+=result.get('usage',{}).get('total_tokens',0)
            atomic(RUN/'OBSERVED_GENERATION_TOKENS.json',{'observed_native_tokens':budget['tokens'],'stop':TOKEN_STOP,'unknown_usage_is_not_zero':True,'hard_remote_token_cap_claim':False})
        print(json.dumps({'case':case,'condition':c['condition_id'],'status':result['status'],'seconds':round(result['elapsed_seconds'],2)}),flush=True)
        if result['status']!='COMPLETE' or not result.get('usage',{}).get('usage_complete'): break


def run():
    m=check(); con=_con()
    existing=con.execute("SELECT count(*) FROM slots WHERE status!='PENDING'").fetchone()[0]
    if existing: raise RuntimeError('primary_already_started_no_automatic_resume')
    readiness={(r['model_block'],r['driver']):r['status']=='READY' for r in con.execute('SELECT * FROM compatibility')}
    pools={}
    ids=['T03','T15','T42']+[t['case_id'] for t in m['tasks'] if t['case_id'] not in ('T03','T15','T42')]
    for model in sorted({c['model_block'] for c in m['conditions']}):
        cs=[c for c in m['conditions'] if c['model_block']==model]
        if not all(readiness.get((model,'native_cli' if c['arm'] in 'CD' else 'direct_neuroruntime'),False) for c in cs):
            continue
        jobs=[]
        for case in ids:
            ordered=sorted(cs,key=lambda c:hashlib.sha256((case+'|'+c['arm']).encode()).hexdigest())
            jobs.extend((case,c) for c in ordered)
        pools[model]=jobs
    con.close()
    if not pools:
        raise RuntimeError('no_eligible_model_blocks_no_primary_dispatch')
    freeze_text(RUN/'PRIMARY_START.json',json_text({'started_at_utc':now(),'controller_pid':os.getpid(),'eligible_models':list(pools),'wall_seconds':STAGE_SECONDS,'recovery':0}))
    deadline=time.monotonic()+STAGE_SECONDS
    budget={'tokens':0,'lock':threading.Lock()}
    with cf.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(_run_account,model,jobs,deadline,budget) for model,jobs in pools.items()]
        for f in futures: f.result()
    freeze_text(RUN/'GENERATION_TERMINAL.json',json_text({'finished_at_utc':now(),'status':'FINITE_PHASE_SETTLED_NOT_QUALITY_ACCEPTANCE'}))
    inspect()


def inspect():
    with _con() as con:
        counts={r[0]:r[1] for r in con.execute('SELECT status,count(*) FROM slots GROUP BY status')}
        compatibility=[dict(r) for r in con.execute('SELECT * FROM compatibility')]
    value={'recorded_at_utc':now(),'slots':counts,'compatibility':compatibility,'complete_primary_outputs':counts.get('COMPLETE',0),'planned_primary_outputs':240,'superiority_established':False}
    atomic(RUN/'PROGRESS.json',value); print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','compatibility','run','inspect']);a=p.parse_args()
    {'prepare':prepare,'compatibility':compatibility,'run':run,'inspect':inspect}[a.action]()


if __name__=='__main__': main()
