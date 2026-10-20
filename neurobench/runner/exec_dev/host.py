"""Common finite QC-process primitive, with receipts outside submitted outputs.

Future model drivers must own this host and pass receipts directly to the
evaluator. A model-authored receipt is not trusted evidence. This development
host does not provide OS isolation or adversarial filesystem protection.
"""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .common import sha, write_json


def run_tool(case, output, receipts, mode):
    output, receipts = Path(output).resolve(), Path(receipts).resolve()
    if receipts.is_relative_to(output):
        raise ValueError('trusted_receipts_must_be_outside_candidate_output')
    output.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    if mode not in ('normal', 'probe', 'repair'):
        raise ValueError('unsupported_tool_mode')
    tool, inp = case.input('qc_tool.py'), case.input('values.npy')
    nonce = uuid.uuid4().hex
    command = [sys.executable, '-I', str(tool), '--input', str(inp),
               '--output', str(output / 'stage.json'), '--mode', mode, '--nonce', nonce]
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    before = sha(inp)
    completed = subprocess.run(command, cwd=output, capture_output=True, text=True,
                               encoding='utf-8', timeout=30, check=False)
    event = {'command': command, 'python_executable': sys.executable,
             'python_prefix': sys.prefix, 'tool_sha256': sha(tool),
             'input_sha256_before': before, 'input_sha256_after': sha(inp),
             'nonce': nonce, 'mode': mode, 'returncode': completed.returncode,
             'stdout': completed.stdout, 'stderr': completed.stderr,
             'output_path': str(output / 'stage.json'),
             'output_sha256': sha(output / 'stage.json') if (output / 'stage.json').is_file() else None,
             'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
             'elapsed_seconds': time.monotonic() - start}
    number = len(list(receipts.glob('*.json'))) + 1
    path = receipts / f'{number:03d}.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, event)
    return event
