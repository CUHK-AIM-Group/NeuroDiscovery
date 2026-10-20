"""Common execution interface and public-only runtime adapters."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

from neurobench.runner.exec_trial import (TrialTools as OriginalTools, PYTHON_HOST as ORIGINAL_HOST,
                              NR_NAMES, public_checks, schema)
from neurobench.runner.exec_dev.common import file_map, sha
from .host import run_tool

PYTHON_HOST = ORIGINAL_HOST.replace(
    '    result.update(extra)',
    "    if set(extra) & {'case_id', 'input_sha256', 'artifacts', 'status'}:\n"
    "        raise ValueError('reserved_result_manifest_field')\n    result.update(extra)")

BASE_PROMPT = '''Complete the task with the supplied read-only inputs. Deliver actual artifacts under output/. Do not access external data, other runs, credentials, evaluator code or the network, and do not train models or modify inputs.
The public task and contract are in context/task.md and context/CONTRACT.json. context/skill_catalog.json and context/skills/ contain a common optional method library. All comparison conditions have the same input files, library and common execution interface.
run_python executes Python with numpy, scipy and nibabel. INPUT_DIR and OUTPUT_DIR are pathlib Paths for inputs/ and output/. Only output/ and scratch/ are writable. finish_result(status='complete', **extra_fields) finalizes the delivery manifest with actual hashes; case_id, input_sha256 and artifacts are host-owned fields. Use status='blocked' with observable reasons when a requested result is unsupported. A valid final manifest ends the task automatically after the current tool batch; a separate submit call is optional. inspect_outputs provides public structural checks, without numerical answers. run_qc_tool executes a supplied analysis.py and records input/output provenance outside the candidate output.
Budget: 120 seconds including reasoning and tools, at most 12 model rounds and 48 tool calls. Explicit submissions are immutable; at most one additional submission is allowed after public structural failure, within the same budget. There are no post-terminal retries.
'''
ARM_PROMPTS = {
    'A': '',
    'B0': 'Optional current NeuroRuntime tools: search_skills, read_skill, describe_task_contract, validate_proposal. Proposal-format checks do not establish artifact or numerical correctness.\n',
    'B1': 'Optional candidate NeuroRuntime tools additionally include compile_task (public requirements and observed input headers) and read_skill_section (relevant method sections with a session cache).\n',
}
COMMON_SCHEMAS = [
    schema('read_file', 'Read a UTF-8 text file under inputs/ or context/. Binary inputs are accessible through Python.', {'path': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}}, ('path',)),
    schema('run_python', 'Execute Python in this workspace. INPUT_DIR and OUTPUT_DIR are pathlib Paths. finish_result(status="complete", **fields) finalizes actual output hashes and ends the task after this tool batch. Use scratch/ for temporary files. No external processes or network.', {'code': {'type': 'string'}}, ('code',)),
    schema('run_qc_tool', 'Execute a supplied inputs/analysis.py on inputs/values.npy. Writes output/stage.json and independently retained execution evidence.'),
    schema('inspect_outputs', 'Inspect output inventory, declared completion, required paths and actual hashes. No numerical expectations.'),
    schema('submit', 'Freeze an explicit submission and return public structural diagnostics. A valid finish_result manifest also submits automatically.'),
]


class BriefTools(OriginalTools):
    def schemas(self):
        result = list(COMMON_SCHEMAS)
        if self.current:
            result += [s for s in self.current.schemas() if s['name'] in NR_NAMES]
        if self.arm == 'B1':
            result += [schema('compile_task', 'Organize the same public contract and observed input headers; find methods from the public task title.'),
                       schema('read_skill_section', 'Read relevant public method sections with source hash and session cache.', {'name': {'type': 'string'}, 'query': {'type': 'string'}}, ('name', 'query'))]
        return result

    def compile_task(self):
        metadata = {}
        for name in self.case.contract['input_sha256']:
            p = self.workspace / 'inputs' / name
            item = {'sha256': sha(p), 'bytes': p.stat().st_size}
            if name.endswith('.npy'):
                a = np.load(p, allow_pickle=False, mmap_mode='r')
                item.update(shape=list(a.shape), dtype=str(a.dtype))
            elif name.endswith(('.nii', '.nii.gz')):
                img = nib.load(p)
                item.update(shape=list(img.shape), dtype=str(img.get_data_dtype()), affine=img.affine.tolist())
            metadata[name] = item
        hints = self.current.call('search_skills', {'query': self.case.contract['title'], 'limit': 4})
        return {'success': True, 'provenance': 'public_contract_and_actual_headers',
                'requirements': self.case.contract, 'input_metadata': metadata, 'method_candidates': hints}

    def call(self, name, args):
        if name not in ('run_python', 'run_qc_tool'):
            return super().call(name, args)
        self.tool_count += 1
        if self.tool_count > 48 or time.monotonic() >= self.deadline:
            return {'success': False, 'error': 'tool_or_wall_budget_reached'}
        try:
            if not isinstance(args, dict):
                raise ValueError('arguments_not_object')
            remaining = max(.01, min(30, self.deadline - time.monotonic()))
            if name == 'run_qc_tool':
                r = run_tool(self.case, self.output, self.receipts, remaining)
                return {'success': r['returncode'] == 0, 'returncode': r['returncode'],
                        'stdout': r['stdout'], 'stderr': r['stderr'], 'output_path': 'output/stage.json',
                        'output_sha256': r['output_sha256'], 'trusted_receipt_retained': True,
                        'elapsed_seconds': r['elapsed_seconds']}
            code = args['code']
            if not isinstance(code, str) or len(code) > 100000:
                raise ValueError('invalid_code_length_or_type')
            ast.parse(code)
            sequence = len(list((self.workspace / 'scratch').glob('call_*.py'))) + 1
            script = self.workspace / 'scratch' / f'call_{sequence:03d}.py'
            script.write_text(code, encoding='utf-8')
            clean = {k: v for k, v in os.environ.items() if k.upper() in ('WINDIR', 'SYSTEMROOT', 'SYSTEMDRIVE', 'TEMP', 'TMP', 'COMSPEC')}
            clean.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            start = time.monotonic()
            r = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-B', str(self.workspace / 'python_host.py'), str(script.relative_to(self.workspace))],
                               cwd=self.workspace, env=clean, capture_output=True, timeout=remaining)
            raw = self.attempt / 'python_receipts'; raw.mkdir(exist_ok=True)
            (raw / f'{sequence:03d}.stdout.bin').write_bytes(r.stdout)
            (raw / f'{sequence:03d}.stderr.bin').write_bytes(r.stderr)
            result = {'success': r.returncode == 0, 'returncode': r.returncode,
                      'stdout': r.stdout.decode('utf-8', errors='replace')[-16000:],
                      'stderr': r.stderr.decode('utf-8', errors='replace')[-16000:],
                      'elapsed_seconds': time.monotonic() - start, 'script_sha256': sha(script),
                      'output_files': file_map(self.output)}
            if (self.output / 'result.json').is_file():
                result['public_diagnostics'] = public_checks(self.case.contract, self.output)
            return result
        except Exception as exc:
            return {'success': False, 'error_type': type(exc).__name__, 'error': str(exc)[:1500]}
