"""Finite, append-only executable development comparison, separate from legacy runs.

All arms share execution, public checks, methods, inputs and submission rules.
No hidden numerical checks or reference implementations enter model workspaces.
The Python audit hook is a cooperative boundary, not an OS sandbox.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from neurobench.runner.artifacts import EventLog, freeze_text, json_text
from neurobench.runner.exec_dev.common import (Case, contained, environment, file_map, load_case,
                                   read_json, receipt_digest, sha, write_json)
from neurobench.runner.exec_dev.host import run_tool

REPO = Path(__file__).resolve().parents[2]
P0 = REPO / 'tmp/benchmark_exec_dev_20261019_r1'
PACKET = P0 / 'packet'
INDEX_SHA = '464915eced7fd850cda0c7a796c1d4c14206cae822fa2ebe83cc685e4f535e58'
METHODS = REPO / 'tmp/benchmark_high_continuation_20261016_r5/materials'
MODELS = ('gpt-5.6-sol', 'deepseek-v4-pro')
ARMS = ('A', 'B0', 'B1')
NR_NAMES = ('search_skills', 'read_skill', 'describe_task_contract', 'validate_proposal')
PRINT_LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def schema(name, description, properties=None, required=()):
    return {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties or {}, 'required': list(required)}}


COMMON_SCHEMAS = [
    schema('read_file', 'Read a public UTF-8 file under context/ or inputs/. Binary inputs are read with run_python.',
           {'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}}, ('path',)),
    schema('run_python', 'Actually execute Python in this workspace, with numpy, scipy and nibabel available. INPUT_DIR, OUTPUT_DIR are pathlib Paths. finish_result(status="complete", **fields) writes result.json with actual hashes. Only output/ and scratch/ may be written. No network, subprocess, package installation or external data access.',
           {'code': {'type': 'string'}}, ('code',)),
    schema('run_qc_tool', 'Actually run the supplied QC executable through the trusted recorded host. Writes output/stage.json; execution receipts are independently retained.',
           {'mode': {'type': 'string', 'enum': ['normal', 'probe', 'repair']}}, ('mode',)),
    schema('inspect_outputs', 'Public checks only: file inventory, required paths, result.json schema and SHA-256 bindings. Does not check numerical answers.'),
    schema('submit', 'Freeze the current output and return public diagnostics. At most two submissions, with one repair only when public diagnostics fail. Passing public checks ends the run; hidden numerical evaluation occurs later.')
]

BASE_PROMPT = '''Execute this engineering development task using the supplied real inputs. Deliver actual files in output/, then call submit. Prose alone is not delivery. Do not access external files, credentials, other runs, reference solutions, evaluator code or the network. Do not train models or modify inputs. Missing required inputs must be reported as blocked with an explicit reason, never invented.
All arms have the same public inputs, task/CONTRACT.json, method library and execution/check tools. context/skill_catalog.json lists generic methods, readable with read_file. Methods are optional guidance; adapt them to the exact task, without executing their unrelated full pipelines.
Use run_python to inspect binary inputs and generate outputs. Python has INPUT_DIR=Path('inputs'), OUTPUT_DIR=Path('output'), and finish_result(status='complete', **extra_fields), which computes actual hashes and writes the required result.json. File paths inside code are relative to this workspace. Write supporting scripts to scratch/, not output/, unless they are intended delivery artifacts. For recorded QC tasks use run_qc_tool, not an unrecorded process or a handwritten QC result.
You have 120 seconds including model reasoning and tools, at most 12 model rounds and 48 tool calls. You may self-check with inspect_outputs, which reveals no numerical expected answers. First submit is saved immutably. Only a failing public check permits one further submission in the original budget. No post-terminal retries are scheduled. Call submit once ready; no follow-up final explanation is needed.
'''
ARM_PROMPTS = {
    'A': '',
    'B0': '''Current NeuroRuntime tools are enabled: search_skills, read_skill, describe_task_contract, validate_proposal. Establish the public contract and load relevant guidance. validate_proposal is the existing proposal-format/static checker; its proposal headings do not change this executable task's file contract. It does not establish numerical correctness.\n''',
    'B1': '''Candidate NeuroRuntime adds compile_task and read_skill_section. Start with compile_task to organize the public requirements, actual input metadata and suggested skill names. Load relevant guidance with bounded sections when useful. Before submitting, inspect_outputs to review actual artifacts and their provenance. All these checks are public engineering diagnostics, not scientific or numerical correctness.\n'''
}


PYTHON_HOST = r'''
import sys, os, json, hashlib, csv, math, shutil, pathlib, time
import numpy as np
import scipy
from scipy import ndimage, stats
import nibabel as nib
from nibabel.processing import resample_from_to
from pathlib import Path
WORKSPACE = Path.cwd().resolve()
INPUT_DIR = WORKSPACE / 'inputs'
OUTPUT_DIR = WORKSPACE / 'output'
_read_roots = [WORKSPACE, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
_write_roots = [OUTPUT_DIR, WORKSPACE / 'scratch']
def _inside(p, roots):
    return any(p.is_relative_to(r) for r in roots)
def _audit(event, args):
    if event == 'open' and not isinstance(args[0], int):
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if not _inside(p, _write_roots if writing else _read_roots):
            raise PermissionError('cooperative_workspace_file_boundary')
    elif event in ('os.listdir', 'os.scandir') and args and args[0] is not None and not isinstance(args[0], int):
        if not _inside(Path(os.fsdecode(args[0])).resolve(), _read_roots):
            raise PermissionError('cooperative_workspace_directory_boundary')
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime'):
        if not _inside(Path(os.fsdecode(args[0])).resolve(), _write_roots):
            raise PermissionError('cooperative_workspace_write_boundary')
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        if not all(_inside(Path(os.fsdecode(v)).resolve(), _write_roots) for v in args[:2]):
            raise PermissionError('cooperative_workspace_move_boundary')
    elif event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.spawn', 'ctypes.dlopen', 'ctypes.dlsym'):
        raise PermissionError('external_execution_or_network_disabled')
sys.addaudithook(_audit)
def finish_result(status='complete', **extra):
    contract = json.loads((WORKSPACE / 'context/CONTRACT.json').read_text(encoding='utf-8'))
    artifacts = {}
    for p in sorted(OUTPUT_DIR.rglob('*')):
        if p.is_file() and p != OUTPUT_DIR / 'result.json':
            artifacts[p.relative_to(OUTPUT_DIR).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(case_id=contract['case_id'], status=status, input_sha256=contract['input_sha256'], artifacts=artifacts)
    result.update(extra)
    (OUTPUT_DIR / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
_source = (WORKSPACE / sys.argv[1]).read_text(encoding='utf-8')
exec(compile(_source, sys.argv[1], 'exec'), globals(), globals())
'''


def public_checks(contract, output):
    """Structural public checks, intentionally no per-case numerical logic."""
    errors = []
    try:
        files = file_map(output)
        result = read_json(output / 'result.json')
        if result.get('case_id') != contract['case_id']:
            errors.append('case_identity')
        if result.get('status') not in ('complete', 'blocked'):
            errors.append('status_enum')
        if result.get('input_sha256') != contract['input_sha256']:
            errors.append('input_hash_manifest')
        artifacts = dict(files); artifacts.pop('result.json', None)
        if result.get('artifacts') != artifacts:
            errors.append('output_hash_manifest')
        if result.get('status') == 'complete':
            errors += ['missing_required:' + p for p in contract['required_outputs'] if p not in artifacts]
    except (OSError, ValueError, TypeError) as exc:
        files, result = {}, {}
        errors.append('manifest_unreadable:' + type(exc).__name__)
    return {'success': True, 'public_check_pass': not errors, 'errors': errors,
            'files': files, 'claimed_status': result.get('status'),
            'scope': 'public_schema_and_hash_checks_only_no_expected_numerical_values'}


class TrialTools:
    def __init__(self, case, workspace, attempt, arm, deadline):
        self.case, self.workspace, self.attempt, self.arm, self.deadline = case, workspace, attempt, arm, deadline
        self.output = workspace / 'output'
        self.receipts = attempt / 'trusted_process_receipts'
        self.receipts.mkdir()
        self.submissions = []
        self.tool_count = 0
        self.cache = {}
        self.terminal = False
        self.current = None
        if arm != 'A':
            from neurobench.runner.high_tools import TaskTools
            task = (case.root / 'task.md').read_text(encoding='utf-8') + '\n' + json_text(case.contract)
            self.current = TaskTools(workspace, task, enabled=True)

    def schemas(self):
        result = list(COMMON_SCHEMAS)
        if self.current:
            result += [s for s in self.current.schemas() if s['name'] in NR_NAMES]
        if self.arm == 'B1':
            result += [schema('compile_task', 'Compile only the public task contract plus actual file/header metadata, grouped requirements and suggested existing method names.'),
                       schema('read_skill_section', 'Read relevant headings/paragraphs from an existing public skill, with source hash and session cache.',
                              {'name': {'type': 'string'}, 'query': {'type': 'string'}}, ('name', 'query'))]
        return result

    def submit(self):
        if len(self.submissions) >= 2 or self.terminal:
            return {'success': False, 'error': 'submission_budget_exhausted'}
        number = len(self.submissions) + 1
        target = self.attempt / 'submissions' / str(number)
        shutil.copytree(self.output, target / 'output')
        shutil.copytree(self.receipts, target / 'receipts')
        diagnostic = public_checks(self.case.contract, target / 'output')
        record = {'number': number, 'at_utc': utc(), 'elapsed_seconds': time.monotonic() - (self.deadline - 120),
                  'output': str(target / 'output'), 'receipts': str(target / 'receipts'),
                  'output_hashes': file_map(target / 'output'), 'receipts_sha256': receipt_digest(target / 'receipts'),
                  'public_diagnostics': diagnostic}
        write_json(target / 'SUBMISSION.json', record)
        self.submissions.append(record)
        self.terminal = diagnostic['public_check_pass'] or number == 2
        return {'submission_number': number, 'terminal': self.terminal, **diagnostic}

    def compile_task(self):
        metadata = {}
        import numpy as np
        import nibabel as nib
        for name in self.case.contract['input_sha256']:
            p = self.workspace / 'inputs' / name
            detail = {'sha256': sha(p), 'bytes': p.stat().st_size}
            if name.endswith('.npy'):
                x = np.load(p, allow_pickle=False, mmap_mode='r')
                detail.update(shape=list(x.shape), dtype=str(x.dtype))
            elif name.endswith(('.nii', '.nii.gz')):
                img = nib.load(p)
                detail.update(shape=list(img.shape), dtype=str(img.get_data_dtype()), affine=img.affine.tolist())
            metadata[name] = detail
        task = self.case.contract['title'] + ' ' + self.case.contract['family_id'].replace('_', ' ')
        hints = self.current.call('search_skills', {'query': task, 'limit': 4})
        return {'success': True, 'provenance': 'same_public_CONTRACT_and_input_headers',
                'requirements': self.case.contract, 'input_metadata': metadata, 'method_candidates': hints,
                'delivery_evidence': {'required_artifacts': self.case.contract['required_outputs'],
                                      'process_host_available': (self.workspace / 'inputs/qc_tool.py').is_file()},
                'numerical_expectations_included': False}

    def section(self, name, query):
        key = (name, query)
        if key in self.cache:
            return {**self.cache[key], 'cache_hit': True}
        raw = self.current.call('read_skill', {'name': name, 'max_chars': 200000, 'offset': 0})
        if not raw.get('success'):
            return raw
        body = raw.get('content', '')
        words = [s.casefold() for s in query.split() if len(s) > 2]
        paragraphs = body.split('\n\n')
        ranked = sorted(enumerate(paragraphs), key=lambda t: (-sum(w in t[1].casefold() for w in words), t[0]))
        selected, length = [], 0
        for i, para in ranked:
            if length >= 4800:
                break
            selected.append((i, para[:4800 - length])); length += len(selected[-1][1])
        value = {'success': True, 'name': name, 'source_sha256': raw.get('source_sha256'),
                 'content_kind': 'method_guidance', 'content': '\n\n'.join(p for _, p in sorted(selected)),
                 'source_total_chars': raw.get('total_chars'), 'cached': True, 'cache_hit': False,
                 'workflow_executed': False}
        self.cache[key] = value
        return value

    def call(self, name, args):
        self.tool_count += 1
        if self.tool_count > 48 or time.monotonic() >= self.deadline:
            return {'success': False, 'error': 'tool_or_wall_budget_reached'}
        try:
            if not isinstance(args, dict):
                raise ValueError('arguments_not_object')
            if name == 'read_file':
                rel = args['path']
                if Path(rel).parts[0] not in ('context', 'inputs'):
                    raise ValueError('not_public_context')
                path = contained(self.workspace, rel)
                offset = max(0, int(args.get('offset', 0)))
                body = path.read_text(encoding='utf-8-sig')
                return {'success': True, 'path': rel, 'source_sha256': sha(path),
                        'content': body[offset:offset + 16000], 'total_chars': len(body),
                        'next_offset': offset + 16000 if offset + 16000 < len(body) else None}
            if name == 'run_python':
                code = args['code']
                if not isinstance(code, str) or len(code) > 100000:
                    raise ValueError('code_too_large_or_not_text')
                ast.parse(code)
                sequence = len(list((self.workspace / 'scratch').glob('call_*.py'))) + 1
                script = self.workspace / 'scratch' / f'call_{sequence:03d}.py'
                script.write_text(code, encoding='utf-8')
                seconds = max(.01, min(30, self.deadline - time.monotonic()))
                clean = {k: v for k, v in os.environ.items() if k.upper() in ('WINDIR', 'SYSTEMROOT', 'SYSTEMDRIVE', 'TEMP', 'TMP', 'COMSPEC')}
                clean.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                started = time.monotonic()
                r = subprocess.run([sys.executable, '-I', str(self.workspace / 'python_host.py'), str(script.relative_to(self.workspace))], cwd=self.workspace,
                                   env=clean, capture_output=True, text=True, encoding='utf-8', timeout=seconds)
                return {'success': r.returncode == 0, 'returncode': r.returncode,
                        'stdout': r.stdout[-16000:], 'stderr': r.stderr[-16000:], 'elapsed_seconds': time.monotonic() - started,
                        'script_sha256': sha(script), 'output_files': file_map(self.output)}
            if name == 'run_qc_tool':
                if not (self.workspace / 'inputs/qc_tool.py').is_file():
                    raise ValueError('no_supplied_qc_tool')
                r = run_tool(self.case, self.output, self.receipts, args['mode'])
                return {'success': r['returncode'] == 0, 'returncode': r['returncode'], 'stdout': r['stdout'],
                        'stderr': r['stderr'], 'output_path': 'output/stage.json', 'output_sha256': r['output_sha256'],
                        'trusted_receipt_retained': True, 'elapsed_seconds': r['elapsed_seconds']}
            if name == 'inspect_outputs':
                return public_checks(self.case.contract, self.output)
            if name == 'submit':
                return self.submit()
            if self.current and name in NR_NAMES:
                return self.current.call(name, args)
            if self.arm == 'B1' and name == 'compile_task':
                return self.compile_task()
            if self.arm == 'B1' and name == 'read_skill_section':
                return self.section(args['name'], args['query'])
            raise ValueError('tool_not_exposed')
        except Exception as exc:
            return {'success': False, 'error_type': type(exc).__name__, 'error': str(exc)[:1500]}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    p0_bindings = read_json(P0 / 'DELIVERY_BINDINGS.json')['files']
    assert all(sha(p) == h for p, h in p0_bindings.items()), 'p0_frozen_file_changed'
    shutil.copytree(METHODS / 'skills', root / 'methods/skills')
    shutil.copy2(METHODS / 'skill_catalog.json', root / 'methods/skill_catalog.json')
    cases = read_json(PACKET / 'INDEX.json')['cases']
    slots = []
    # Rotate three arms across successive cases, separately for each model.
    for model in MODELS:
        for i, c in enumerate(cases):
            order = ARMS[i % 3:] + ARMS[:i % 3]
            for arm in order:
                slots.append({'id': model + '__' + c['case_id'] + '__' + arm,
                              'model': model, 'case_id': c['case_id'], 'arm': arm,
                              'family_id': c['family_id'], 'input_kind': c['input_kind']})
    sources = [Path(__file__), REPO / 'neurobench/runner/high_channels.py', REPO / 'neurobench/runner/high_tools.py',
               REPO / 'neurobench/runner/high_gpt_route.py', REPO / 'neurobench/runner/subscription_direct.py', REPO / 'neurobench/runner/artifacts.py']
    sources += sorted((REPO / 'core').glob('harness*.py'))
    sources += [REPO / 'core/tool_registry.py', REPO / 'core/tool_pipeline.py', REPO / 'core/tools_impl.py', REPO / 'core/mcp_stdio.py',
                REPO / 'core/permissions.py', REPO / 'core/skill_loader/loader.py']
    sources += sorted((REPO / 'neurobench/runner/exec_dev').glob('*.py'))
    bindings = {str(p): sha(p) for p in sources if p.is_file()}
    manifest = {'revision': 'exec-trial-20261019-r1', 'created_at_utc': utc(),
                'user_request': '先测试 12 例，看看对比结果', 'case_count': 12, 'planned_task_runs': 72,
                'models': MODELS, 'arms': {'A': 'common_execution_only', 'B0': 'current_NeuroRuntime', 'B1': 'candidate_public_contract_metadata_sections'},
                'slots': slots, 'packet': str(PACKET), 'packet_index_sha256': INDEX_SHA,
                'deadline_seconds_per_task': 120, 'max_model_rounds': 12, 'max_tool_calls': 48,
                'max_native_requests': 864, 'stage_deadline_seconds': 4800, 'observed_total_token_stop': 3000000,
                'account_concurrency': 1, 'post_terminal_retries': 0, 'max_public_submissions': 2,
                'same_common_execution_tools_and_public_context': True, 'same_wall_clock_not_equal_token_budget': True,
                'gpt_route': 'existing_user_authorized_xnova_responses',
                'gpt_subscription_preflight': 'TLS timeout through inherited proxy; direct discovery HTTP403 before any refresh exchange or inference',
                'deepseek_route': 'official_api_deepseek_com_chat_completions',
                'model_identity': 'native/provider reported; no independent backend attestation', 'reasoning_requested': 'high',
                'python_execution': 'CPU; cooperative audit boundary; no OS sandbox claim',
                'methods': file_map(root / 'methods'), 'source_bindings': bindings, 'p0_bindings': p0_bindings,
                'environment': environment(), 'scope': '12 exposed engineering development cases; no blind accuracy or superiority claim; CLI deferred',
                'oracle_feedback_to_model': False, 'common_prompt': BASE_PROMPT, 'arm_prompts': ARM_PROMPTS}
    write_json(root / 'MANIFEST.json', manifest)
    write_json(root / 'GATE.json', {'manifest_sha256': sha(root / 'MANIFEST.json'), 'p0_preserved': len(p0_bindings),
                                   'source_files': len(bindings), 'created_at_utc': utc()})
    print(json.dumps({'prepared': True, 'planned_task_runs': 72, 'root': str(root)}), flush=True)


def check(root):
    assert sha(root / 'MANIFEST.json') == read_json(root / 'GATE.json')['manifest_sha256']
    m = read_json(root / 'MANIFEST.json')
    assert sha(PACKET / 'INDEX.json') == m['packet_index_sha256']
    for p, h in m['source_bindings'].items():
        assert sha(p) == h, p
    assert file_map(root / 'methods') == m['methods']
    return m


def evaluate_snapshots(case, toolset):
    # Separate subprocess, never imported into the model-facing Python tool.
    verdicts = []
    for s in toolset.submissions:
        target = Path(s['output']).parent
        verdict = target / 'VERDICT.json'
        cmd = [sys.executable, '-X', 'utf8', '-m', 'neurobench.runner.exec_dev.oracle', '--packet', str(PACKET), '--case', case.id,
               '--output', s['output'], '--index-sha', INDEX_SHA, '--receipts', s['receipts'],
               '--receipts-sha', s['receipts_sha256'], '--verdict', str(verdict)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding='utf-8', timeout=90)
        if not verdict.is_file():
            write_json(verdict, {'case_id': case.id, 'verdict': 'ORACLE_ERROR', 'reason': 'verifier_process_failure', 'stderr': r.stderr[-2000:]})
        verdicts.append(read_json(verdict))
    return verdicts


def attempt(root, slot):
    from neurobench.runner.high_channels import responses_round, chat_round
    from neurobench.runner.high_gpt_route import XNOVA_ROUTE
    case = load_case(PACKET, slot['case_id'], INDEX_SHA)
    private = root / 'attempts' / slot['id']
    private.mkdir(parents=True, exist_ok=False)
    work = root / 'workspaces' / slot['id']
    shutil.copytree(case.root / 'inputs', work / 'inputs')
    shutil.copytree(root / 'methods/skills', work / 'context/skills')
    shutil.copy2(root / 'methods/skill_catalog.json', work / 'context/skill_catalog.json')
    shutil.copy2(case.root / 'CONTRACT.json', work / 'context/CONTRACT.json')
    shutil.copy2(case.root / 'task.md', work / 'context/task.md')
    for name in ('output', 'scratch'):
        (work / name).mkdir()
    (work / 'python_host.py').write_text(PYTHON_HOST, encoding='utf-8')
    public_before = file_map(work / 'inputs')
    context_before = file_map(work / 'context')
    instruction = BASE_PROMPT + ARM_PROMPTS[slot['arm']] + '\nPUBLIC TASK:\n' + (case.root / 'task.md').read_text(encoding='utf-8') + '\nPUBLIC CONTRACT:\n' + json_text(case.contract)
    freeze_text(private / 'PROMPT.txt', instruction)
    write_json(private / 'START.json', {'slot': slot, 'at_utc': utc(), 'public_inputs': public_before, 'context': context_before})
    started = time.monotonic(); deadline = started + 120
    log = EventLog(private / 'events.jsonl')
    toolset = TrialTools(case, work, private, slot['arm'], deadline)
    schemas = toolset.schemas()
    write_json(private / 'SCHEMAS.json', schemas)
    history = [{'role': 'user', 'content': instruction}]
    provider = 'responses' if slot['model'] == 'gpt-5.6-sol' else 'deepseek'
    status, error, rounds, usage, native_models, final = 'UNKNOWN', '', 0, [], [], ''
    try:
        for _ in range(12):
            if time.monotonic() >= deadline:
                status, error = 'BUDGET_STOP', 'task_wall_deadline'; break
            rounds += 1
            if provider == 'responses':
                result, counts = responses_round(history, schemas, log, deadline, route=XNOVA_ROUTE)
                history.extend(result['output'])
                native_models.append(result['response'].get('model'))
            else:
                result, counts = chat_round('deepseek', history, schemas, log, deadline)
                history.append(result['message'])
                native_models.append(result.get('model'))
                if result.get('model') != slot['model']:
                    raise RuntimeError('native_model_echo_mismatch')
            usage.append(counts); log.write('native_usage', payload=counts)
            if not counts['usage_complete']:
                raise RuntimeError('native_usage_incomplete')
            for call in result['calls']:
                args = call['arguments'] if isinstance(call['arguments'], dict) else json.loads(call['arguments'])
                log.write('tool_request', name=call['name'], arguments=args, call_id=call['call_id'])
                output = toolset.call(call['name'], args) if not toolset.terminal else {'success': False, 'error': 'already_submitted_terminal'}
                log.write('tool_result', name=call['name'], result=output, call_id=call['call_id'])
                payload = json.dumps(output, ensure_ascii=False)
                if provider == 'responses':
                    history.append({'type': 'function_call_output', 'call_id': call['call_id'], 'output': payload})
                else:
                    history.append({'role': 'tool', 'tool_call_id': call['call_id'], 'name': call['name'], 'content': payload})
            if toolset.terminal:
                status = 'SUBMITTED'; break
            if not result['calls']:
                final = result['text']
                if time.monotonic() >= deadline:
                    status, error = 'BUDGET_STOP', 'text_arrived_after_deadline'; break
                public = toolset.submit()
                log.write('implicit_submission_after_final_text', result=public)
                if toolset.terminal:
                    status = 'SUBMITTED'; break
                history.append({'role': 'user', 'content': 'Your first submission failed public structural checks. One repair is available inside the original budget. ' + json.dumps(public)})
        else:
            status, error = 'BUDGET_STOP', 'max_model_rounds'
    except Exception as exc:
        error = str(exc)[:1200] if isinstance(exc, (RuntimeError, ValueError, TimeoutError)) else type(exc).__name__
        status = 'FAILED_KNOWN' if error.startswith(('native_http_', 'native_usage_incomplete', 'deepseek_noncomplete_', 'native_model_echo_mismatch')) else 'UNKNOWN'
        log.write('attempt_error', status=status, error_type=type(exc).__name__, error=error)
    finally:
        elapsed = time.monotonic() - started
        log.close()
    # Preserve an unfinished output snapshot without disguising it as an on-time submission.
    if not toolset.submissions:
        shutil.copytree(toolset.output, private / 'unfinished_output')
    inputs_unchanged = file_map(work / 'inputs') == public_before
    context_unchanged = file_map(work / 'context') == context_before
    verdicts = evaluate_snapshots(case, toolset)
    result = {**slot, 'started_at_utc': read_json(private / 'START.json')['at_utc'], 'finished_at_utc': utc(),
              'status': status, 'error': error, 'elapsed_seconds': elapsed, 'native_request_count': rounds,
              'native_models': native_models, 'reasoning_requested': 'high', 'native_usage': usage,
              'tokens_observed': sum(u['total_tokens'] for u in usage), 'usage_complete': len(usage) == rounds and all(u['usage_complete'] for u in usage),
              'tool_calls': toolset.tool_count, 'submissions': toolset.submissions, 'verdicts': verdicts,
              'first_pass': bool(verdicts) and verdicts[0]['verdict'] == 'PASS',
              'final_pass': bool(verdicts) and verdicts[-1]['verdict'] == 'PASS' and toolset.submissions[-1]['elapsed_seconds'] <= 120 and inputs_unchanged and context_unchanged,
              'inputs_unchanged': inputs_unchanged, 'context_unchanged': context_unchanged, 'final_text': final,
              'no_post_terminal_resend': True}
    write_json(private / 'RESULT.json', result)
    with PRINT_LOCK:
        print(json.dumps({k: result[k] for k in ('id', 'status', 'final_pass', 'elapsed_seconds', 'native_request_count', 'error')}, ensure_ascii=False), flush=True)
    return result


def run(root):
    m = check(root)
    started_file = root / 'RUN_START.json'
    # Exclusive once-only launch; never automatically resume old or ambiguous slots.
    with started_file.open('x', encoding='utf-8') as f:
        json.dump({'at_utc': utc(), 'pid': os.getpid(), 'manifest_sha256': sha(root / 'MANIFEST.json')}, f)
    stage_deadline = time.monotonic() + m['stage_deadline_seconds']
    def worker(model):
        used = 0
        for slot in (s for s in m['slots'] if s['model'] == model):
            if time.monotonic() + 120 > stage_deadline or used >= m['observed_total_token_stop'] // 2:
                with PRINT_LOCK:
                    print(json.dumps({'stopped_before_dispatch': slot['id'], 'reason': 'finite_stage_budget'}), flush=True)
                break
            result = attempt(root, slot)
            used += result['tokens_observed']
            if result['error'].startswith(('native_http_401', 'native_http_403', 'native_http_429')):
                break
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, MODELS))
    write_json(root / 'RUN_END.json', {'at_utc': utc(), 'finished_attempts': len(list((root / 'attempts').glob('*/RESULT.json'))),
                                      'note': 'all original results retained; no terminal retries'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    (prepare if a.action == 'prepare' else run)(a.root.resolve())
