"""Versioned common delivery protocol for executable development comparisons.

All arms share this interface. Public checks inspect structure and provenance;
the existing private numerical oracle remains separate. Frozen v1 code is not
modified, and importing this module never starts a model request.
"""
from __future__ import annotations

import copy
import json
import shutil
import time

from neurobench.runner.exec_brief.tools import BriefTools, COMMON_SCHEMAS as V1_SCHEMAS, PYTHON_HOST, ARM_PROMPTS
from neurobench.runner.exec_dev.common import file_map, read_json, receipt_digest, sha, write_json
from neurobench.runner.exec_trial import schema, utc

REVISION = 'exec-delivery-protocol-20261019-v2'
HOST_FIELDS = frozenset(('case_id', 'input_sha256', 'artifacts'))
FINISH_FIELDS = frozenset(('status', 'summary', 'reason', 'evidence', 'metadata'))

FINISH_SCHEMA = schema(
    'finish_result',
    'Finalize delivery from the actual output files and freeze a submission. '
    'The host fills case_id, input_sha256 and artifact hashes; do not pass these fields. '
    'Use blocked with an observable reason if the requested result cannot be supported. '
    'This checks structure and provenance, not numerical correctness.',
    {'status': {'type': 'string', 'enum': ['complete', 'blocked']},
     'summary': {'type': 'string'}, 'reason': {'type': 'string'},
     'evidence': {'type': 'array', 'items': {'type': 'string'}},
     'metadata': {'type': 'object'}}, ('status',))
FINISH_SCHEMA['parameters']['additionalProperties'] = False

BASE_PROMPT = '''Complete the task with the supplied read-only inputs. Deliver actual artifacts under output/. Do not access external data, other runs, credentials, evaluator code or the network, and do not train models or modify inputs.
The public task and contract are in context/task.md and context/CONTRACT.json. context/skill_catalog.json and context/skills/ contain a common optional method library. All comparison conditions have the same input files, library and common execution interface.
run_python executes a fresh Python process on every call, with numpy, scipy and nibabel. Python variables do not persist between calls; files under output/ and scratch/ persist. INPUT_DIR and OUTPUT_DIR are pathlib Paths. Only output/ and scratch/ are writable.
The registered finish_result tool finalizes and submits the existing files. Pass status="complete", or status="blocked" with an observable reason. Optional summary, evidence and metadata describe your delivery. The host owns case_id, input_sha256 and artifacts; do not pass or author these fields. The Python finish_result helper also remains available. A valid helper-generated manifest submits automatically after the current tool batch. inspect_outputs reports actual files, missing paths and manifest issues without numerical answers. run_qc_tool executes a supplied analysis.py and records input/output provenance outside the candidate output.
Budget: 120 seconds including reasoning and tools, at most 12 model rounds and 48 tool calls. Common feedback reports the remaining budget. When model rounds are nearly exhausted, inspect the delivery state and finalize the available supported result within the original budget. Explicit submissions are immutable; at most one additional submission is allowed after public structural failure. There are no post-terminal retries.
'''


def public_checks(contract, output):
    """Retain inventory even when the delivery manifest is absent or invalid."""
    issues, files, state = [], {}, {}
    try:
        files = file_map(output)
    except (OSError, ValueError, TypeError) as exc:
        issues.append({'code': 'inventory_unreadable', 'error_type': type(exc).__name__})
    artifacts = {k: v for k, v in files.items() if k != 'result.json'}
    required = list(contract['required_outputs'])
    missing = [p for p in required if p not in artifacts]
    if 'result.json' not in files:
        issues.append({'code': 'manifest_missing', 'path': 'result.json'})
    else:
        try:
            state = read_json(output / 'result.json')
            if not isinstance(state, dict):
                raise ValueError('manifest_not_object')
        except (OSError, ValueError, TypeError) as exc:
            state = {}
            issues.append({'code': 'manifest_invalid', 'error_type': type(exc).__name__})
        else:
            for code, valid in (
                ('case_identity', state.get('case_id') == contract['case_id']),
                ('status_enum', state.get('status') in ('complete', 'blocked')),
                ('input_hash_manifest', state.get('input_sha256') == contract['input_sha256']),
                ('output_hash_manifest', state.get('artifacts') == artifacts),
            ):
                if not valid:
                    issues.append({'code': code})
            if state.get('status') == 'complete':
                issues += [{'code': 'missing_required_output', 'path': p} for p in missing]
            if state.get('status') == 'blocked' and not str(state.get('reason') or '').strip():
                issues.append({'code': 'blocked_reason_missing'})
    if not issues:
        delivery_state = 'ready_to_submit'
    elif any(i['code'] == 'inventory_unreadable' for i in issues):
        delivery_state = 'invalid_inventory'
    elif 'result.json' in files:
        delivery_state = 'invalid_manifest'
    elif not missing:
        delivery_state = 'awaiting_finalization'
    else:
        delivery_state = 'artifacts_incomplete'
    next_action = None
    if delivery_state == 'awaiting_finalization':
        next_action = {'tool': 'finish_result', 'arguments': {'status': 'complete'}}
    elif delivery_state == 'ready_to_submit':
        next_action = {'tool': 'submit', 'arguments': {}}
    elif delivery_state == 'invalid_manifest':
        status = state.get('status', 'complete')
        if status not in ('complete', 'blocked'):
            status = 'complete'
        args = {'status': status}
        if status == 'blocked' and state.get('reason'):
            args['reason'] = state['reason']
        if not missing or status == 'blocked':
            next_action = {'tool': 'finish_result', 'arguments': args}
    return {'success': True, 'public_check_pass': not issues, 'state': delivery_state,
            'files': files, 'required_outputs': required, 'missing_required_outputs': missing,
            'manifest_present': 'result.json' in files, 'claimed_status': state.get('status'),
            'issues': issues, 'errors': [i['code'] for i in issues], 'next_action': next_action,
            'scope': 'public_structure_and_hash_checks_only_no_numerical_answers'}


class ProtocolTools(BriefTools):
    def __init__(self, *args, max_model_rounds=12, max_tool_calls=48, **kwargs):
        if not 1 <= max_model_rounds <= 12 or not 1 <= max_tool_calls <= 48:
            raise ValueError('protocol_budget_outside_original_limits')
        super().__init__(*args, **kwargs)
        self.max_model_rounds = max_model_rounds
        self.max_tool_calls = max_tool_calls
        self.round_index = 0

    def schemas(self):
        result = copy.deepcopy(super().schemas())
        for item in result:
            if item['name'] == 'run_python':
                item['description'] += ' Each call starts a fresh process; variables reset and files persist.'
            elif item['name'] == 'inspect_outputs':
                item['description'] = 'Inspect actual output files and hashes, missing required paths and manifest issues. No numerical expectations.'
        result.insert(len(V1_SCHEMAS), copy.deepcopy(FINISH_SCHEMA))
        return result

    def begin_round(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not self.round_index < index <= self.max_model_rounds:
            raise ValueError('model_round_outside_budget_or_not_increasing')
        self.round_index = index
        return self.budget_status()

    def budget_status(self):
        remaining = self.max_model_rounds - self.round_index
        return {'model_round': self.round_index, 'max_model_rounds': self.max_model_rounds,
                'model_rounds_remaining_after_current': remaining,
                'tool_calls_remaining': max(0, self.max_tool_calls - self.tool_count),
                'seconds_remaining': round(max(0, self.deadline - time.monotonic()), 3),
                'closeout_recommended': remaining <= 2 and not self.terminal,
                'finalization_tool': 'finish_result'}

    def _feedback(self, result):
        value = dict(result)
        value['budget'] = self.budget_status()
        value['python_process'] = 'fresh_per_call_files_persist_variables_reset'
        return value

    def _finish(self, args):
        reserved = sorted(HOST_FIELDS & args.keys())
        if reserved:
            return {'success': False, 'error': 'host_owned_fields', 'fields': reserved,
                    'next_action': {'tool': 'finish_result', 'arguments': {
                        k: v for k, v in args.items() if k in FINISH_FIELDS}},
                    'message': 'Omit host-owned fields; the host computes their values.'}
        if set(args) - FINISH_FIELDS:
            return {'success': False, 'error': 'unknown_finish_fields',
                    'fields': sorted(set(args) - FINISH_FIELDS), 'allowed_fields': sorted(FINISH_FIELDS)}
        if args.get('status') not in ('complete', 'blocked'):
            return {'success': False, 'error': 'status_required', 'allowed_statuses': ['complete', 'blocked']}
        for key in ('summary', 'reason'):
            if key in args and not isinstance(args[key], str):
                return {'success': False, 'error': 'invalid_field_type', 'field': key}
        if 'evidence' in args and (not isinstance(args['evidence'], list) or not all(isinstance(v, str) for v in args['evidence'])):
            return {'success': False, 'error': 'invalid_field_type', 'field': 'evidence'}
        if 'metadata' in args and (not isinstance(args['metadata'], dict) or HOST_FIELDS & args['metadata'].keys()):
            return {'success': False, 'error': 'invalid_metadata_or_host_owned_fields'}
        if args['status'] == 'blocked' and not args.get('reason', '').strip():
            return {'success': False, 'error': 'blocked_reason_required'}
        # Validate serializability before any output or immutable submission write.
        json.dumps(args, allow_nan=False)
        diagnostics = public_checks(self.case.contract, self.output)
        if diagnostics['state'] == 'invalid_inventory':
            return {'success': False, 'error': 'invalid_inventory', 'public_diagnostics': diagnostics}
        if args['status'] == 'complete' and diagnostics['missing_required_outputs']:
            return {'success': False, 'error': 'required_outputs_missing', 'public_diagnostics': diagnostics}
        if file_map(self.workspace / 'inputs') != self.case.contract['input_sha256']:
            return {'success': False, 'error': 'input_integrity_lost'}
        artifacts = {p: h for p, h in diagnostics['files'].items() if p != 'result.json'}
        if time.monotonic() >= self.deadline:
            return {'success': False, 'error': 'wall_budget_reached'}
        previous = sha(self.output / 'result.json') if (self.output / 'result.json').is_file() else None
        state = {'case_id': self.case.id, 'input_sha256': self.case.contract['input_sha256'],
                 'artifacts': artifacts, **args}
        write_json(self.output / 'result.json', state)
        audit = self.attempt / 'protocol_finalizations'
        write_json(audit / f'{len(list(audit.glob("*.json"))) + 1:03d}.json', {
            'at_utc': utc(), 'round': self.round_index, 'requested_fields': args,
            'previous_manifest_sha256': previous, 'manifest_sha256': sha(self.output / 'result.json'),
            'provenance_source': 'host_bound_public_contract_and_actual_file_hashes'})
        return self.submit()

    def submit(self):
        if self.terminal or len(self.submissions) >= 2:
            return {'success': False, 'error': 'submission_budget_exhausted'}
        if time.monotonic() >= self.deadline:
            return {'success': False, 'error': 'wall_budget_reached'}
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

    def call(self, name, args):
        if self.terminal:
            return self._feedback({'success': False, 'error': 'already_submitted_terminal'})
        if self.tool_count >= self.max_tool_calls or time.monotonic() >= self.deadline:
            return self._feedback({'success': False, 'error': 'tool_or_wall_budget_reached'})
        if name in ('finish_result', 'inspect_outputs', 'submit'):
            self.tool_count += 1
            try:
                if not isinstance(args, dict):
                    raise ValueError('arguments_not_object')
                if name == 'finish_result':
                    result = self._finish(args)
                elif name == 'inspect_outputs':
                    result = public_checks(self.case.contract, self.output)
                else:
                    result = self.submit()
            except (ValueError, TypeError, OSError) as exc:
                result = {'success': False, 'error_type': type(exc).__name__, 'error': str(exc)[:1500]}
        else:
            result = super().call(name, args)
        if name == 'run_python':
            # Replace the legacy diagnostic which loses inventory on manifest errors.
            result['public_diagnostics'] = public_checks(self.case.contract, self.output)
            if 'reserved_result_manifest_field' in result.get('stderr', ''):
                result['error'] = 'host_owned_fields'
                result['reserved_fields'] = sorted(HOST_FIELDS)
                result['next_action'] = {'tool': 'finish_result', 'arguments': {'status': 'complete'}}
        return self._feedback(result)
