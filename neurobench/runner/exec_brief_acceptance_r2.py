"""Forward-only acceptance correction for public train/validation/test roles.

Original inputs, submissions, r1 verifier and raw verdicts remain untouched.
This review changes role spelling only, without changing row assignments.
"""
from pathlib import Path
import argparse
import shutil
import tempfile

from neurobench.runner.exec_dev.common import (file_map, finish, load_case, read_csv, read_json,
                                   sha, write_csv, write_json)
from neurobench.runner.exec_brief.oracle import evaluate as original_evaluate

ROLE_ALIASES = {'train': 'train', 'training': 'train', 'val': 'val',
                'validation': 'val', 'test': 'test', 'testing': 'test'}


def evaluate(case, output, receipts=None, receipts_sha=None):
    output = Path(output)
    before = file_map(output)
    raw = original_evaluate(case, output, receipts, receipts_sha)
    result = dict(raw)
    if case.id == 'N07' and raw.get('reason') == 'split_labels':
        rows = read_csv(output / 'split.csv')
        # All original identity/provenance checks have already passed. Unknown
        # roles remain failures; every scientific split constraint is rerun.
        if all(r['split'].strip().casefold() in ROLE_ALIASES for r in rows):
            with tempfile.TemporaryDirectory(prefix='brief_roles_r2_') as temporary:
                target = Path(temporary) / 'output'
                shutil.copytree(output, target)
                fields = list(rows[0])
                for row in rows:
                    row['split'] = ROLE_ALIASES[row['split'].strip().casefold()]
                write_csv(target / 'split.csv', rows, fields)
                state = read_json(output / 'result.json')
                extra = {k: v for k, v in state.items() if k not in ('case_id', 'status', 'input_sha256', 'artifacts')}
                finish(case, target, state['status'], **extra)
                result = original_evaluate(case, target, receipts, receipts_sha)
                result['review_normalized_split_sha256'] = sha(target / 'split.csv')
                result['semantic_alias_review'] = True
    if file_map(output) != before:
        raise RuntimeError('original_submission_modified')
    return {**result, 'acceptance_revision': 'exec-brief-roles-r2', 'raw_verdict': raw,
            'original_output_hashes': before}


def qualify(packet, destination):
    from neurobench.runner.exec_brief.qualify import reference
    case = load_case(packet, 'N07', sha(packet / 'INDEX.json'))
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for name, labels, fault, expected in [
        ('short_names', {'train': 'train', 'val': 'val', 'test': 'test'}, None, 'PASS'),
        ('public_names', {'train': 'train', 'val': 'validation', 'test': 'test'}, None, 'PASS'),
        ('full_names', {'train': 'training', 'val': 'validation', 'test': 'testing'}, None, 'PASS'),
        ('case_whitespace', {'train': ' TRAIN ', 'val': ' Validation ', 'test': ' TEST '}, None, 'PASS'),
        ('unknown_role', {'train': 'train', 'val': 'calibration', 'test': 'test'}, None, 'FAIL'),
        ('held_site_leak', {'train': 'training', 'val': 'validation', 'test': 'testing'}, 'site', 'FAIL'),
        ('duplicate_subject', {'train': 'train', 'val': 'validation', 'test': 'test'}, 'duplicate', 'FAIL'),
        ('wrong_proportion', {'train': 'train', 'val': 'validation', 'test': 'test'}, 'proportion', 'FAIL'),
        ('wrong_input_hash', {'train': 'train', 'val': 'validation', 'test': 'test'}, 'hash', 'FAIL'),
    ]:
        root = destination / name
        reference(case, root / 'output', root / 'receipts', 0)
        rows = read_csv(root / 'output/split.csv')
        for row in rows:
            row['split'] = labels[row['split']]
        if fault == 'site':
            next(r for r in rows if r['split'] == labels['test'])['split'] = labels['train']
        elif fault == 'duplicate':
            rows.append(dict(rows[0]))
        elif fault == 'proportion':
            for row in rows:
                if row['split'] != labels['test']:
                    row['split'] = labels['train']
        write_csv(root / 'output/split.csv', rows, ['subject_id', 'split'])
        finish(case, root / 'output')
        if fault == 'hash':
            state = read_json(root / 'output/result.json'); state['input_sha256'] = {}
            write_json(root / 'output/result.json', state)
        actual = evaluate(case, root / 'output')
        records.append({'test': name, 'expected': expected, 'actual': actual['verdict'], 'passed': actual['verdict'] == expected})
    result = {'passed': all(r['passed'] for r in records), 'checks': len(records), 'records': records,
              'scope': 'Role spelling aliases only; public scientific requirements and original evidence unchanged.'}
    write_json(destination / 'QUALIFIED.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--qualify', type=Path)
    args = parser.parse_args()
    if args.qualify:
        verdict = qualify(args.packet.resolve(), args.qualify.resolve())
        print(verdict)
        raise SystemExit(0 if verdict['passed'] else 1)
