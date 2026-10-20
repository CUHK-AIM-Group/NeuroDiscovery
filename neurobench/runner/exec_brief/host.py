"""Neutral, recorded process primitive shared by all trial conditions."""
from pathlib import Path
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

from neurobench.runner.exec_dev.common import sha, write_json


def run_tool(case, output, receipts, timeout=30):
    output, receipts = Path(output).resolve(), Path(receipts).resolve()
    if receipts.is_relative_to(output):
        raise ValueError('receipts_inside_candidate_output')
    output.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    tool, source = case.input('analysis.py'), case.input('values.npy')
    nonce = uuid.uuid4().hex
    cmd = [sys.executable, '-I', '-X', 'utf8', str(tool), '--input', str(source),
           '--output', str(output / 'stage.json'), '--nonce', nonce]
    before, start = sha(source), time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    r = subprocess.run(cmd, cwd=output, capture_output=True, timeout=max(.01, min(30, timeout)))
    event = {'command': cmd, 'python_executable': sys.executable, 'python_prefix': sys.prefix,
             'tool_sha256': sha(tool), 'input_sha256_before': before, 'input_sha256_after': sha(source),
             'nonce': nonce, 'returncode': r.returncode,
             'stdout': r.stdout.decode('utf-8', errors='replace'),
             'stderr': r.stderr.decode('utf-8', errors='replace'),
             'stdout_raw_hex': r.stdout.hex(), 'stderr_raw_hex': r.stderr.hex(),
             'output_path': str(output / 'stage.json'),
             'output_sha256': sha(output / 'stage.json') if (output / 'stage.json').is_file() else None,
             'started_at_utc': started, 'elapsed_seconds': time.monotonic() - start}
    p = receipts / f'{len(list(receipts.glob("*.json"))) + 1:03d}.json'
    if p.exists():
        raise FileExistsError(p)
    write_json(p, event)
    return event
