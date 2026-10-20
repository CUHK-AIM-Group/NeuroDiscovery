"""Finite offline qualification: two legal solutions and >=3 bad variants/case.

Reference and oracle run in different Python processes. All source snapshots,
failed verdicts and process logs are retained in a new output directory.
This command makes no model requests and does not run a scientific experiment.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from .common import contained, file_map, finish, load_case, read_csv, read_json, receipt_digest, sha, write_csv, write_json


def corrupt_semantics(case, out, specific=None):
    cid = case.id
    if cid == 'E01':
        p = out / 'matched.csv'; rows = read_csv(p); rows[0]['label'] = float(rows[0]['label']) + 1
        write_csv(p, rows, ['subject_id', 'label'])
    elif cid == 'E02':
        p = out / 'sub-case001/anat/sub-case001_T1w.nii.gz'
        b = bytearray(p.read_bytes()); b[-1] ^= 1; p.write_bytes(b)
    elif cid in ('E03', 'E04'):
        p = out / 'mask.nii.gz'; img = nib.load(p)
        if specific == 'affine_copy_without_resampling':
            data = np.asarray(nib.load(case.input('source_mask.nii.gz')).dataobj).astype(np.uint8)
        else:
            data = np.asarray(img.dataobj).copy(); data.flat[0] = 1 - data.flat[0]
        nib.save(nib.Nifti1Image(data, img.affine), p)
    elif cid == 'E05':
        p = out / 'fc.npy'; data = np.load(p, allow_pickle=False); data[0, 1] += .1; data[1, 0] += .1
        np.save(p, data, allow_pickle=False)
    elif cid == 'E06':
        p = out / 'nodes.csv'; rows = read_csv(p); rows[0]['degree'] = int(rows[0]['degree']) + 1
        write_csv(p, rows, ['node', 'degree', 'strength'])
    elif cid == 'E07':
        p = out / 'split.csv'; rows = read_csv(p)
        test = next(r for r in rows if r['split'] == 'test'); train = next(r for r in rows if r['split'] == 'train')
        test['split'], train['split'] = 'train', 'test'
        write_csv(p, rows, ['subject_id', 'split'])
    elif cid == 'E08':
        p = out / 'scaled.npy'
        if specific == 'all_data_fit_with_false_train_metadata':
            x = np.load(case.input('features.npy'), allow_pickle=False)
            mean, scale = x.mean(axis=0), x.std(axis=0); scale[scale == 0] = 1
            scaler = read_json(out / 'scaler.json'); scaler.update(mean=mean.tolist(), scale=scale.tolist())
            write_json(out / 'scaler.json', scaler); data = (x - mean) / scale
        else:
            data = np.load(p, allow_pickle=False); data[0, 0] += .1
        np.save(p, data, allow_pickle=False)
    elif cid in ('E09', 'E10'):
        p = out / 'stage.json'; stage = read_json(p); stage['mean'] += 1; write_json(p, stage)
    elif cid == 'E11':
        p = out / 'qc.json'; qc = read_json(p); qc['frames'] += 1; write_json(p, qc)
    elif cid == 'E12':
        result = read_json(out / 'result.json'); result['status'] = 'complete'; write_json(out / 'result.json', result)
        return
    finish(case, out)


def qualify(packet, destination):
    packet, destination = Path(packet).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new qualification directory; old failures remain.')
    destination.mkdir(parents=True)
    code_root = Path(__file__).parent
    code_before = file_map(code_root)
    # Exclude interpreter bytecode from scientific/code bindings.
    code_before = {k: v for k, v in code_before.items() if k.endswith('.py')}
    for name in code_before:
        path = contained(destination / 'code', name); path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code_root / name, path)
    index_sha = sha(packet / 'INDEX.json')
    index = read_json(packet / 'INDEX.json')
    write_json(destination / 'PLAN.json', {'index_sha256': index_sha,
               'case_count': len(index['cases']), 'positive_variants': ['reference', 'alternative'],
               'base_mutants': ['missing_output', 'semantic_error', 'wrong_input_binding'],
               'code_sha256': code_before, 'model_calls': 0, 'judge_calls': 0})
    rows = []
    started = time.monotonic()

    def invoke(module, args, log):
        command = [sys.executable, '-X', 'utf8', '-m', module, *map(str, args)]
        r = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=120)
        write_json(log, {'command': command, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr})
        return r.returncode

    def judge(case, label, output, events, expected, trust=True):
        verdict_path = destination / 'verdicts' / f'{case.id}_{label}.json'
        args = ['--packet', packet, '--case', case.id, '--output', output,
                '--index-sha', index_sha, '--verdict', verdict_path]
        if trust and events.is_dir():
            args += ['--receipts', events, '--receipts-sha', receipt_digest(events)]
        invoke('neurobench.runner.exec_dev.oracle', args, destination / 'logs' / f'{case.id}_{label}_oracle.json')
        verdict = read_json(verdict_path)
        rows.append({'case_id': case.id, 'label': label, 'expected': expected,
                     'actual': verdict['verdict'], 'ok': verdict['verdict'] == expected,
                     'reason': verdict.get('reason')})

    for entry in index['cases']:
        case = load_case(packet, entry['case_id'], index_sha)
        for variant in ('reference', 'alternative'):
            output = destination / 'outputs' / case.id / variant
            events = destination / 'trusted_events' / case.id / variant
            args = ['--packet', packet, '--case', case.id, '--output', output,
                    '--receipts', events, '--index-sha', index_sha, '--variant', variant]
            rc = invoke('neurobench.runner.exec_dev.reference', args, destination / 'logs' / f'{case.id}_{variant}_solver.json')
            if rc != 0:
                rows.append({'case_id': case.id, 'label': variant, 'expected': 'PASS',
                             'actual': 'SOLVER_ERROR', 'ok': False})
            else:
                judge(case, variant, output, events, 'PASS')
        source = destination / 'outputs' / case.id / 'reference'
        events = destination / 'trusted_events' / case.id / 'reference'
        mutants = ['missing_output', 'semantic_error', 'wrong_input_binding']
        if case.id in ('E03', 'E04'): mutants += ['malformed_candidate_nifti']
        if case.id == 'E04': mutants += ['affine_copy_without_resampling']
        if case.id == 'E08': mutants += ['all_data_fit_with_false_train_metadata']
        if case.id in ('E09', 'E10'): mutants += ['self_report_without_host_receipts']
        for name in mutants:
            out = destination / 'outputs' / case.id / name
            shutil.copytree(source, out)
            if name == 'missing_output':
                target = contained(out, case.contract['required_outputs'][0] if case.contract['required_outputs'] else 'result.json')
                # One verified file in this new work directory, never a recursive deletion.
                target.unlink()
                if case.id != 'E12': finish(case, out)
            elif name == 'wrong_input_binding':
                r = read_json(out / 'result.json'); key = next(iter(r['input_sha256']))
                r['input_sha256'][key] = '0' * 64; write_json(out / 'result.json', r)
            elif name == 'malformed_candidate_nifti':
                (out / 'mask.nii.gz').write_bytes(b'not a NIfTI payload')
                finish(case, out)
            elif name != 'self_report_without_host_receipts':
                corrupt_semantics(case, out, name)
            judge(case, name, out, events, 'FAIL', trust=name != 'self_report_without_host_receipts')
        print('QUALIFIED', case.id, 'checks', sum(r['case_id'] == case.id for r in rows),
              'ok', all(r['ok'] for r in rows if r['case_id'] == case.id), flush=True)
    code_after = {k: v for k, v in file_map(code_root).items() if k.endswith('.py')}
    stable = code_after == code_before and sha(packet / 'INDEX.json') == index_sha
    all_ok = stable and all(r['ok'] for r in rows)
    report = {'status': 'PASS' if all_ok else 'FAIL', 'scope': 'P0 offline development-oracle qualification; no model performance or scientific acceptance',
              'case_count': len(index['cases']), 'families': dict(Counter(x['family_id'] for x in index['cases'])),
              'input_mix': dict(Counter(x['input_kind'] for x in index['cases'])),
              'positive_checks': sum(r['expected'] == 'PASS' for r in rows),
              'negative_checks': sum(r['expected'] == 'FAIL' for r in rows),
              'all_checks': len(rows), 'source_and_index_stable': stable,
              'elapsed_seconds': time.monotonic() - started, 'index_sha256': index_sha,
              'checks': rows, 'code_sha256': code_before,
              'source_snapshot': 'code/', 'new_model_calls': 0, 'new_judge_calls': 0,
              'limitations': ['All cases and references are exposed development material.',
                             'Host receipt ownership is a driver contract, not an OS sandbox.',
                             'No model-backed A/B trial or scientific model training was performed.']}
    write_json(destination / 'QUALIFICATION.json', report)
    print('QUALIFICATION', report['status'], report['all_checks'], 'checks', flush=True)
    return all_ok


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--destination', required=True, type=Path)
    a = p.parse_args()
    raise SystemExit(0 if qualify(a.packet, a.destination) else 1)
