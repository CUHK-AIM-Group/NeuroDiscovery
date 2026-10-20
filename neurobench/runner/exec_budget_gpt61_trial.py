"""Finite GPT-6.1-Sol-only comparison using the same common protocol and budget.

Keeps original packets, implementations, terminal outcomes and scores frozen.
"""
from __future__ import annotations
import argparse, concurrent.futures, json, os, shutil, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
from neurobench.runner.artifacts import EventLog, freeze_text, json_text
from neurobench.runner.exec_dev.common import (Case, environment, file_map, load_case, read_json,
                                   receipt_digest, sha, write_json)
from neurobench.runner.exec_trial import ARMS, REPO, METHODS, PRINT_LOCK, utc
from neurobench.runner.exec_budget_v3 import BudgetTools as TrialTools, PYTHON_HOST, BASE_PROMPT, ARM_PROMPTS, public_checks, REVISION, STANDARD
P0 = REPO / 'tmp/benchmark_exec_brief_dev_20261019_r1'
PACKET = P0 / 'packet'
def index_sha():
    return sha(PACKET / 'INDEX.json')
MODELS = ('gpt-6.1-sol',)
READY = REPO / 'tmp/benchmark_exec_protocol_v2_20261019_r1/READY.json'

def prepare(root, user_request):
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError('user_request_required_for_new_manifest')
    ready = read_json(READY)
    assert ready['passed'] and all(sha(p) == h for p, h in ready['source_bindings'].items()), 'qualified_v2_source_changed'
    root.mkdir(parents=True, exist_ok=False)
    original_p0_bindings = read_json(PACKET / 'PRESERVED.json')['files']
    known_drift = read_json(REPO / 'tmp/benchmark_exec_budget_flash_trial_20261019_r1/DOCUMENT_DRIFT.json')
    p0_bindings, preexisting_drift = {}, []
    for p, h in original_p0_bindings.items():
        if Path(p).is_file() and sha(p) == h:
            p0_bindings[p] = h
        elif p == known_drift['path'] and h == known_drift['expected_sha256'] and not Path(p).exists():
            preexisting_drift.append({'path': p, 'expected_sha256': h, 'state': 'missing_before_this_run',
                                     'execution_or_model_input': False})
        else:
            raise RuntimeError('unexpected_prior_binding_drift:' + p)
    assert len(preexisting_drift) <= 1

    shutil.copytree(METHODS / 'skills', root / 'methods/skills')
    shutil.copy2(METHODS / 'skill_catalog.json', root / 'methods/skill_catalog.json')
    q = read_json(P0 / 'qualification_r1/QUALIFICATION.json')
    assert q['passed'] and q['packet_index_sha256'] == index_sha(), 'qualification_required'
    cases = read_json(PACKET / 'INDEX.json')['cases']
    private_lineage = {c['case_id']: c for c in read_json(PACKET / 'PRIVATE_LINEAGE.json')['cases']}
    slots = []
    # Rotate three arms across successive cases, separately for each model.
    for model in MODELS:
        for i, c in enumerate(cases):
            order = ARMS[i % 3:] + ARMS[:i % 3]
            for arm in order:
                slots.append({'id': model + '__' + c['case_id'] + '__' + arm,
                              'model': model, 'case_id': c['case_id'], 'arm': arm,
                              'family_id': private_lineage[c['case_id']]['family_id'], 'input_kind': private_lineage[c['case_id']]['input_kind']})
    sources = [Path(__file__), REPO / 'neurobench/runner/exec_budget_v3.py',
               REPO / 'neurobench/runner/exec_protocol_v2_trial.py', REPO / 'neurobench/tests/test_exec_budget_v3.py', REPO / 'neurobench/tests/test_exec_budget_gpt61.py',
               REPO / 'neurobench/runner/exec_budget_gpt61_preflight.py',
               REPO / 'neurobench/runner/exec_budget_gpt61_report.py', REPO / 'neurobench/runner/exec_budget_gpt61_readback.py',
               REPO / 'neurobench/runner/exec_budget_flash_report.py', REPO / 'neurobench/runner/exec_brief_gpt61_report.py',
               REPO / 'neurobench/runner/exec_brief_report.py', REPO / 'neurobench/runner/high_channels.py', REPO / 'neurobench/runner/high_tools.py',
               REPO / 'neurobench/runner/high_gpt_route.py', REPO / 'neurobench/runner/subscription_direct.py', REPO / 'neurobench/runner/artifacts.py',
               REPO / 'neurobench/runner/exec_brief_trial.py', REPO / 'neurobench/runner/exec_brief_gpt61_channel.py',
               REPO / 'neurobench/runner/exec_brief_acceptance_r2.py', REPO / 'neurobench/runner/exec_protocol_v2.py',
               REPO / 'neurobench/runner/exec_brief_gpt61_trial.py']
    sources += sorted((REPO / 'core').glob('harness*.py'))
    sources += [REPO / 'core/tool_registry.py', REPO / 'core/tool_pipeline.py', REPO / 'core/tools_impl.py', REPO / 'core/mcp_stdio.py',
                REPO / 'core/permissions.py', REPO / 'core/skill_loader/loader.py']
    sources += sorted((REPO / 'neurobench/runner/exec_dev').glob('*.py'))
    sources += sorted((REPO / 'neurobench/runner/exec_brief').glob('*.py')) + [REPO / 'neurobench/runner/exec_trial.py']
    bindings = {str(p): sha(p) for p in sources if p.is_file()}
    manifest = {'revision': 'exec-budget-gpt61-trial-20261019-r1', 'created_at_utc': utc(),
                'user_request': user_request, 'case_count': 12, 'planned_task_runs': 36,
                'budget_profile': 'standard', 'budget': asdict(STANDARD),
                'future_complex_profile_not_dispatched': {'task_seconds': 900, 'model_rounds': 48, 'tool_calls': 192, 'python_call_seconds': 120},
                'baseline_budget': {'task_seconds': 120, 'model_rounds': 12, 'tool_calls': 48},
                'analysis_predeclared': 'Report actual delivery and observed submissions within all old caps separately. The latter is descriptive under a changed prompt, not a counterfactual old-budget run. Both common protocol and budget change; before/after is not isolated NR causality.',
                'qualified_common_protocol_ready': str(READY), 'qualified_ready_sha256': sha(READY),
                'models': MODELS, 'arms': {'A': 'common_execution_only', 'B0': 'current_NeuroRuntime', 'B1': 'candidate_public_contract_metadata_sections'},
                'slots': slots, 'packet': str(PACKET), 'packet_index_sha256': index_sha(),
                'deadline_seconds_per_task': 300, 'max_model_rounds': 24, 'max_tool_calls': 96,
                'max_native_requests': 864, 'stage_deadline_seconds': 14400, 'observed_total_token_stop': 3000000,
                'account_concurrency': 1, 'post_terminal_retries': 0, 'max_public_submissions': 2,
                'same_common_execution_tools_and_public_context': True, 'same_wall_clock_not_equal_token_budget': True,
                'gpt_dispatch_this_revision': True, 'gpt_route': 'existing_user_authorized_xnova_responses', 'requested_gpt_model': 'gpt-6.1-sol',
                'acceptance_revision': 'exec-brief-roles-r2',
                'acceptance_policy': 'Predeclared existing qualified role-alias correction; raw r1 oracle results retained.',
                'comparison_reference': str(REPO / 'tmp/benchmark_exec_brief_gpt61_trial_20261019_r1/delivery_r1/RESULTS.json'),
                'comparison_reference_sha256': sha(REPO / 'tmp/benchmark_exec_brief_gpt61_trial_20261019_r1/delivery_r1/RESULTS.json'),
                'same_protocol_flash_reference': str(REPO / 'tmp/benchmark_exec_budget_flash_trial_20261019_r1/delivery_r2/RESULTS.json'),
                'same_protocol_flash_reference_sha256': sha(REPO / 'tmp/benchmark_exec_budget_flash_trial_20261019_r1/delivery_r2/RESULTS.json'),
                'official_model_docs': 'https://developers.openai.com/api/docs/models/gpt-6.1-sol',
                'deepseek_dispatch_this_revision': False,
                'model_change_only': False,
                'common_delivery_protocol': REVISION,
                'registered_finish_tool_all_arms': True,
                'same_round_and_tool_budget_feedback_all_arms': True,
                'domain_workflow_hints_added_to_tasks': False,
                'model_identity': 'native/provider reported; no independent backend attestation', 'reasoning_requested': 'high',
                'python_execution': 'CPU; cooperative audit boundary; no OS sandbox claim',
                'methods': file_map(root / 'methods'), 'source_bindings': bindings, 'p0_bindings': p0_bindings, 'original_p0_binding_count': len(original_p0_bindings), 'preexisting_documentation_drift': preexisting_drift,
                'environment': environment(), 'scope': '12 exposed engineering development cases; no blind accuracy or superiority claim; CLI deferred',
                'qualification_sha256': sha(P0 / 'qualification_r1/QUALIFICATION.json'), 'auto_submit_after_valid_final_manifest': True, 'public_diagnostic_and_fault_flags_removed': True, 'oracle_feedback_to_model': False, 'common_prompt': BASE_PROMPT, 'arm_prompts': ARM_PROMPTS}
    write_json(root / 'MANIFEST.json', manifest)
    write_json(root / 'GATE.json', {'manifest_sha256': sha(root / 'MANIFEST.json'), 'p0_preserved': len(p0_bindings),
                                   'source_files': len(bindings), 'created_at_utc': utc()})
    print(json.dumps({'prepared': True, 'planned_task_runs': 36, 'root': str(root)}), flush=True)


def check(root):
    assert sha(root / 'MANIFEST.json') == read_json(root / 'GATE.json')['manifest_sha256']
    m = read_json(root / 'MANIFEST.json')
    assert sha(PACKET / 'INDEX.json') == m['packet_index_sha256']
    for p, h in m['source_bindings'].items():
        assert sha(p) == h, p
    assert all(Path(p).is_file() and sha(p) == h for p, h in m['p0_bindings'].items()), 'prior_binding_changed_since_preflight'
    assert file_map(root / 'methods') == m['methods']
    assert m['budget'] == asdict(STANDARD)
    assert (m['deadline_seconds_per_task'], m['max_model_rounds'], m['max_tool_calls']) == (300, 24, 96)
    assert m['models'] == list(MODELS) and len(m['slots']) == 36
    preflight = read_json(root / 'PREFLIGHT.json')
    assert preflight['passed'] and preflight['manifest_sha256'] == sha(root / 'MANIFEST.json')
    return m


def evaluate_snapshots(case, toolset):
    # Separate subprocess, never imported into the model-facing Python tool.
    verdicts = []
    for s in toolset.submissions:
        target = Path(s['output']).parent
        verdict = target / 'VERDICT.json'
        cmd = [sys.executable, '-X', 'utf8', '-m', 'neurobench.runner.exec_brief.oracle', '--packet', str(PACKET), '--case', case.id,
               '--output', s['output'], '--index-sha', index_sha(), '--receipts', s['receipts'],
               '--receipts-sha', s['receipts_sha256'], '--verdict', str(verdict)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding='utf-8', timeout=90)
        if not verdict.is_file():
            write_json(verdict, {'case_id': case.id, 'verdict': 'ORACLE_ERROR', 'reason': 'verifier_process_failure', 'stderr': r.stderr[-2000:]})
        verdicts.append(read_json(verdict))
    return verdicts


def replay_split(work, private):
    """Repeat recorded Python calls in a clean workspace, without inference."""
    target = private / 'split_reproducibility'
    target.mkdir()
    for name in ('inputs', 'context'):
        shutil.copytree(work / name, target / name)
    for name in ('output', 'scratch'):
        (target / name).mkdir()
    shutil.copy2(work / 'python_host.py', target / 'python_host.py')
    events = []
    started = time.monotonic()
    try:
        for script in sorted((work / 'scratch').glob('call_*.py')):
            relative = script.relative_to(work)
            shutil.copy2(script, target / relative)
            r = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-B', str(target / 'python_host.py'), str(relative)],
                               cwd=target, capture_output=True, timeout=max(.01, 30 - (time.monotonic() - started)))
            events.append({'script_sha256': sha(script), 'returncode': r.returncode,
                           'stdout': r.stdout.decode('utf-8', errors='replace'), 'stderr': r.stderr.decode('utf-8', errors='replace')})
            if r.returncode:
                raise ValueError('replayed_python_call_failed')
        from neurobench.runner.exec_dev.common import read_csv
        read = lambda p: {r['subject_id']: r['split'] for r in read_csv(p)}
        equal = read(work / 'output/split.csv') == read(target / 'output/split.csv')
        result = {'passed': equal, 'method': 'clean_process_replay_of_all_recorded_python_calls', 'events': events}
    except Exception as exc:
        result = {'passed': False, 'error_type': type(exc).__name__, 'error': str(exc), 'events': events}
    write_json(target / 'REPLAY.json', result)
    return {'passed': result['passed'], 'evidence': str(target / 'REPLAY.json')}


def attempt(root, slot):
    from neurobench.runner.exec_brief_gpt61_channel import responses_round
    from neurobench.runner.exec_brief.channel import flash_round
    from neurobench.runner.exec_brief_gpt61_channel import GPT61_ROUTE as XNOVA_ROUTE
    case = load_case(PACKET, slot['case_id'], index_sha())
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
    started = time.monotonic(); deadline = started + STANDARD.task_seconds
    log = EventLog(private / 'events.jsonl')
    toolset = TrialTools(case, work, private, slot['arm'], deadline, budget=STANDARD, started=started)
    schemas = toolset.schemas()
    write_json(private / 'SCHEMAS.json', schemas)
    history = [{'role': 'user', 'content': instruction}]
    provider = 'responses' if slot['model'] == 'gpt-6.1-sol' else 'deepseek'
    status, error, rounds, usage, native_models, final = 'UNKNOWN', '', 0, [], [], ''
    try:
        for _ in range(STANDARD.model_rounds):
            if time.monotonic() >= deadline:
                status, error = 'BUDGET_STOP', 'task_wall_deadline'; break
            rounds += 1
            feedback = toolset.begin_round(rounds)
            history.append({'role': 'user', 'content': 'Common execution budget state: ' + json.dumps(feedback)})
            log.write('budget_feedback_before_model', payload=feedback)
            if provider == 'responses':
                result, counts = responses_round(history, schemas, log, deadline, route=XNOVA_ROUTE)
                history.extend(result['output'])
                native_models.append(result['response'].get('model'))
            else:
                result, counts = flash_round(history, schemas, log, deadline)
                history.append(result['message'])
                native_models.append(result.get('model'))
                if (result.get('model') or '').casefold() not in {'deepseek-flash', 'deepseek-v4.1-flash', 'deepseek-v4.1-flash-1017'}:
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
            if not toolset.terminal and (toolset.output / 'result.json').is_file() and public_checks(case.contract, toolset.output)['public_check_pass']:
                automatic = toolset.submit()
                log.write('automatic_submission_after_final_manifest', result=automatic)
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
    reproducibility = replay_split(work, private) if case.id == 'N07' and toolset.submissions else {'passed': True, 'applicable': False}
    result = {**slot, 'started_at_utc': read_json(private / 'START.json')['at_utc'], 'finished_at_utc': utc(),
              'status': status, 'error': error, 'elapsed_seconds': elapsed, 'native_request_count': rounds,
              'native_models': native_models, 'reasoning_requested': 'high', 'native_usage': usage,
              'tokens_observed': sum(u['total_tokens'] for u in usage), 'usage_complete': len(usage) == rounds and all(u['usage_complete'] for u in usage),
              'tool_calls': toolset.tool_count, 'submissions': toolset.submissions, 'verdicts': verdicts,
              'first_pass': bool(verdicts) and verdicts[0]['verdict'] == 'PASS' and reproducibility['passed'],
              'final_pass': bool(verdicts) and verdicts[-1]['verdict'] == 'PASS' and toolset.submissions[-1]['elapsed_seconds'] <= STANDARD.task_seconds and inputs_unchanged and context_unchanged and reproducibility['passed'],
              'split_reproducibility': reproducibility,
              'inputs_unchanged': inputs_unchanged, 'context_unchanged': context_unchanged, 'final_text': final,
              'no_post_terminal_resend': True, 'delivery_protocol': REVISION}
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
            if time.monotonic() + STANDARD.task_seconds > stage_deadline or used >= m['observed_total_token_stop'] // len(MODELS):
                with PRINT_LOCK:
                    print(json.dumps({'stopped_before_dispatch': slot['id'], 'reason': 'finite_stage_budget'}), flush=True)
                break
            result = attempt(root, slot)
            used += result['tokens_observed']
            if result['error'].startswith(('native_http_401', 'native_http_403', 'native_http_429')):
                break
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(worker, MODELS))
    write_json(root / 'RUN_END.json', {'at_utc': utc(), 'finished_attempts': len(list((root / 'attempts').glob('*/RESULT.json'))),
                                      'note': 'all original results retained; no terminal retries'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--user-request', help='Current request authorizing the separately scoped experiment')
    a = p.parse_args()
    if a.action == 'prepare':
        if not a.user_request:
            p.error('--user-request is required to bind a new experiment manifest')
        prepare(a.root.resolve(), a.user_request)
    else:
        run(a.root.resolve())
