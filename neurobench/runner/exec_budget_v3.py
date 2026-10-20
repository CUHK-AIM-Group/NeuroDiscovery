"""Explicit common budget profiles over the frozen v2 delivery protocol.

Numerical acceptance, schemas and NR adapters are inherited unchanged. The
dispatcher applies the selected limit to every tool, including calls after 48.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from neurobench.runner.exec_protocol_v2 import (
    ProtocolTools as V2Tools, BASE_PROMPT as V2_PROMPT, PYTHON_HOST,
    ARM_PROMPTS, HOST_FIELDS, public_checks,
)
from neurobench.runner.exec_brief.host import run_tool
from neurobench.runner.exec_dev.common import contained, file_map, receipt_digest, sha, write_json
from neurobench.runner.exec_trial import NR_NAMES, utc

REVISION = 'exec-common-budget-20261019-v3'


@dataclass(frozen=True)
class Budget:
    task_seconds: int
    model_rounds: int
    tool_calls: int
    python_call_seconds: int
    max_submissions: int = 2

    def __post_init__(self):
        values = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values.values()):
            raise ValueError('budget_fields_must_be_positive_integers')
        if self.python_call_seconds > self.task_seconds or self.max_submissions != 2:
            raise ValueError('invalid_process_or_submission_budget')


STANDARD = Budget(300, 24, 96, 30)
COMPLEX = Budget(900, 48, 192, 120)
PROFILES = {'standard': STANDARD, 'complex': COMPLEX}


def base_prompt(budget):
    original = 'Budget: 120 seconds including reasoning and tools, at most 12 model rounds and 48 tool calls.'
    if V2_PROMPT.count(original) != 1:
        raise RuntimeError('frozen_prompt_budget_anchor_changed')
    return V2_PROMPT.replace(original,
        f'Budget: {budget.task_seconds} seconds including reasoning and tools, '
        f'at most {budget.model_rounds} model rounds and {budget.tool_calls} tool calls.')


BASE_PROMPT = base_prompt(STANDARD)


class BudgetTools(V2Tools):
    def __init__(self, case, workspace, attempt, arm, deadline, *, budget=STANDARD, started=None):
        super().__init__(case, workspace, attempt, arm, deadline)
        self.budget = budget
        self.max_model_rounds = budget.model_rounds
        self.max_tool_calls = budget.tool_calls
        self.started = deadline - budget.task_seconds if started is None else started
        if abs(deadline - self.started - budget.task_seconds) > .001:
            raise ValueError('deadline_must_match_selected_budget')

    def submit(self):
        if self.terminal or len(self.submissions) >= self.budget.max_submissions:
            return {'success': False, 'error': 'submission_budget_exhausted'}
        if time.monotonic() >= self.deadline:
            return {'success': False, 'error': 'wall_budget_reached'}
        number = len(self.submissions) + 1
        target = self.attempt / 'submissions' / str(number)
        shutil.copytree(self.output, target / 'output')
        shutil.copytree(self.receipts, target / 'receipts')
        diagnostic = public_checks(self.case.contract, target / 'output')
        record = {'number': number, 'at_utc': utc(),
                  'elapsed_seconds': time.monotonic() - self.started,
                  'model_round': self.round_index, 'tool_calls': self.tool_count,
                  'output': str(target / 'output'), 'receipts': str(target / 'receipts'),
                  'output_hashes': file_map(target / 'output'),
                  'receipts_sha256': receipt_digest(target / 'receipts'),
                  'public_diagnostics': diagnostic}
        write_json(target / 'SUBMISSION.json', record)
        self.submissions.append(record)
        self.terminal = diagnostic['public_check_pass'] or number == self.budget.max_submissions
        return {'submission_number': number, 'terminal': self.terminal, **diagnostic}

    def _dispatch(self, name, args):
        if name == 'finish_result':
            return self._finish(args)
        if name == 'inspect_outputs':
            return public_checks(self.case.contract, self.output)
        if name == 'submit':
            return self.submit()
        if name == 'read_file':
            relative = args['path']
            if Path(relative).parts[0] not in ('context', 'inputs'):
                raise ValueError('not_public_context')
            path = contained(self.workspace, relative)
            offset = max(0, int(args.get('offset', 0)))
            body = path.read_text(encoding='utf-8-sig')
            return {'success': True, 'path': relative, 'source_sha256': sha(path),
                    'content': body[offset:offset + 16000], 'total_chars': len(body),
                    'next_offset': offset + 16000 if offset + 16000 < len(body) else None}
        if self.current and name in NR_NAMES:
            return self.current.call(name, args)
        if self.arm == 'B1' and name == 'compile_task':
            return self.compile_task()
        if self.arm == 'B1' and name == 'read_skill_section':
            return self.section(args['name'], args['query'])
        if name not in ('run_python', 'run_qc_tool'):
            raise ValueError('tool_not_exposed')
        remaining = max(.01, min(self.budget.python_call_seconds, self.deadline - time.monotonic()))
        if name == 'run_qc_tool':
            result = run_tool(self.case, self.output, self.receipts, remaining)
            return {'success': result['returncode'] == 0, 'returncode': result['returncode'],
                    'stdout': result['stdout'], 'stderr': result['stderr'],
                    'output_path': 'output/stage.json', 'output_sha256': result['output_sha256'],
                    'trusted_receipt_retained': True, 'elapsed_seconds': result['elapsed_seconds']}
        code = args['code']
        if not isinstance(code, str) or len(code) > 100000:
            raise ValueError('invalid_code_length_or_type')
        ast.parse(code)
        sequence = len(list((self.workspace / 'scratch').glob('call_*.py'))) + 1
        script = self.workspace / 'scratch' / f'call_{sequence:03d}.py'
        script.write_text(code, encoding='utf-8')
        clean = {k: v for k, v in os.environ.items() if k.upper() in
                 ('WINDIR', 'SYSTEMROOT', 'SYSTEMDRIVE', 'TEMP', 'TMP', 'COMSPEC')}
        clean.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, '-I', '-X', 'utf8', '-B', str(self.workspace / 'python_host.py'),
             str(script.relative_to(self.workspace))], cwd=self.workspace, env=clean,
            capture_output=True, timeout=remaining)
        raw = self.attempt / 'python_receipts'
        raw.mkdir(exist_ok=True)
        (raw / f'{sequence:03d}.stdout.bin').write_bytes(result.stdout)
        (raw / f'{sequence:03d}.stderr.bin').write_bytes(result.stderr)
        return {'success': result.returncode == 0, 'returncode': result.returncode,
                'stdout': result.stdout.decode('utf-8', errors='replace')[-16000:],
                'stderr': result.stderr.decode('utf-8', errors='replace')[-16000:],
                'elapsed_seconds': time.monotonic() - start,
                'script_sha256': sha(script), 'output_files': file_map(self.output)}

    def call(self, name, args):
        if self.terminal:
            return self._feedback({'success': False, 'error': 'already_submitted_terminal'})
        if self.tool_count >= self.max_tool_calls or time.monotonic() >= self.deadline:
            return self._feedback({'success': False, 'error': 'tool_or_wall_budget_reached'})
        self.tool_count += 1
        try:
            if not isinstance(args, dict):
                raise ValueError('arguments_not_object')
            result = self._dispatch(name, args)
        except Exception as exc:
            result = {'success': False, 'error_type': type(exc).__name__, 'error': str(exc)[:1500]}
        if name == 'run_python':
            result['public_diagnostics'] = public_checks(self.case.contract, self.output)
            if 'reserved_result_manifest_field' in result.get('stderr', ''):
                result['error'] = 'host_owned_fields'
                result['reserved_fields'] = sorted(HOST_FIELDS)
                result['next_action'] = {'tool': 'finish_result', 'arguments': {'status': 'complete'}}
        return self._feedback(result)
