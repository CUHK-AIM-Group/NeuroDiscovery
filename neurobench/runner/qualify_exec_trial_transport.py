"""Synthetic UTF-8 transport regression; does not alter or rerun model cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from neurobench.runner.exec_dev.common import write_json, sha


def qualify(destination):
    destination.mkdir(parents=True, exist_ok=False)
    code = "import sys,json; print(json.dumps({'encoding':sys.stdout.encoding})); print(chr(20013)+chr(25991)+chr(215)+' '+chr(10003))"
    env = {k: v for k, v in os.environ.items() if k.upper() in ('WINDIR', 'SYSTEMROOT', 'SYSTEMDRIVE', 'TEMP', 'TMP', 'COMSPEC')}
    env['PYTHONIOENCODING'] = 'utf-8'
    rows = []
    for name, command in [('original', [sys.executable, '-I', '-c', code]),
                          ('explicit_utf8', [sys.executable, '-I', '-X', 'utf8', '-c', code])]:
        result = subprocess.run(command, env=env, capture_output=True, timeout=10)
        (destination / (name + '.stdout.bin')).write_bytes(result.stdout)
        (destination / (name + '.stderr.bin')).write_bytes(result.stderr)
        try:
            decoded = result.stdout.decode('utf-8')
            roundtrip = chr(20013) + chr(25991) + chr(215) + ' ' + chr(10003) in decoded
        except UnicodeDecodeError:
            decoded, roundtrip = None, False
        rows.append({'mode': name, 'command': command, 'returncode': result.returncode,
                     'utf8_decodes_and_roundtrips': roundtrip, 'decoded_stdout': decoded,
                     'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
                     'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()})
    write_json(destination / 'RESULT.json', {
        'observed_trial_error': 'UTF8 UnicodeDecodeError in subprocess reader thread, followed by caught tool TypeError when stdout absent',
        'root_cause': 'isolated -I ignores PYTHONIOENCODING; use explicit -X utf8 and preserve raw bytes',
        'fixtures': rows, 'explicit_utf8_passed': rows[1]['utf8_decodes_and_roundtrips'] and rows[1]['returncode'] == 0,
        'no_model_requests': True, 'original_trial_source_and_outcomes_unchanged': True,
        'qualifier_sha256': sha(__file__)})
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True, type=Path)
    qualify(p.parse_args().destination)
