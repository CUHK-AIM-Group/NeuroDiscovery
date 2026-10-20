"""Finite references, legitimate alternatives and semantic counterexamples."""
from pathlib import Path
import argparse
import hashlib
import math
import shutil

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from neurobench.runner.exec_dev.common import (file_map, finish, load_case, read_csv, read_json,
                                   receipt_digest, sha, write_csv, write_json)
from .host import run_tool
from .oracle import evaluate


def reference(case, output, receipts, variant):
    output.mkdir(parents=True)
    cid = case.id
    if cid == 'N01':
        labels = {r['subject_id']: r['label'] for r in read_csv(case.input('labels.csv'))}
        ids = read_json(case.input('subjects.json'))[::(-1 if variant else 1)]
        write_csv(output / 'matched.csv', [{'subject_id': s, 'label': labels[s]} for s in ids], ['subject_id', 'label'])
    elif cid == 'N02':
        p = output / 'sub-case001/anat/sub-case001_T1w.nii.gz'; p.parent.mkdir(parents=True)
        shutil.copyfile(case.input('anatomical.nii.gz'), p)
        write_json(output / 'staging.json', {'source_sha256': sha(p), 'shape': list(nib.load(p).shape)})
    elif cid in ('N03', 'N04'):
        if cid == 'N03':
            img = nib.load(case.input('map.nii.gz')); a = np.asarray(img.dataobj) >= read_json(case.input('parameters.json'))['threshold']
            write_json(output / 'stats.json', {'voxel_count': int(a.sum()), 'volume_ml': float(a.sum() * abs(np.linalg.det(img.affine[:3, :3])) / 1000)})
        else:
            source, img = nib.load(case.input('region.nii.gz')), nib.load(case.input('reference.nii.gz'))
            a = np.asarray(resample_from_to(source, (img.shape, img.affine), order=0).dataobj) > 0
        nib.save(nib.Nifti1Image(a.astype(np.int16 if variant else np.uint8), img.affine), output / 'mask.nii.gz')
    elif cid in ('N05', 'N11'):
        x = np.load(case.input('timeseries.npy'))
        np.save(output / 'fc.npy', np.corrcoef(x.T))
        write_json(output / 'qc.json', {'frames': len(x), 'rois': x.shape[1], 'undefined': [], 'ddof': variant})
        if cid == 'N11':
            scale = x.std(axis=0, ddof=variant); scale[scale == 0] = 1
            np.save(output / 'timeseries_z.npy', (x - x.mean(axis=0)) / scale)
    elif cid == 'N06':
        fc = np.load(case.input('fc.npy')); n = len(fc)
        density = read_json(case.input('parameters.json'))['density']
        edges = sorted([(i, j, float(fc[i, j])) for i in range(n) for j in range(i + 1, n)], key=lambda t: (-abs(t[2]), t[0], t[1]))
        # Both exact-integer arithmetic and the old floating ceil are legal now.
        count = math.ceil(density * n * (n - 1) / 2) if variant else round(density * (n * (n - 1) // 2))
        selected = edges[:count]
        write_csv(output / 'edges.csv', [dict(i=i, j=j, weight=w) for i, j, w in selected], ['i', 'j', 'weight'])
        write_csv(output / 'nodes.csv', [{'node': i, 'degree': sum(i in (a, b) for a, b, _ in selected), 'strength': sum(abs(w) for a, b, w in selected if i in (a, b))} for i in range(n)], ['node', 'degree', 'strength'])
        write_json(output / 'qc.json', {'density': count / len(edges), 'rounding': 'ceil' if variant else 'nearest', 'ties': 'index'})
    elif cid == 'N07':
        params, rows = read_json(case.input('parameters.json')), read_csv(case.input('subjects.csv'))
        groups, mapping = {}, {}
        for r in rows:
            if r['site'] == params['held_out_site']:
                mapping[r['subject_id']] = 'test'
            else:
                groups.setdefault((r['site'], r['label']), []).append(r['subject_id'])
        for group in groups.values():
            ordered = sorted(group, key=lambda s: hashlib.sha256((params['seed'] + str(variant) + s).encode()).digest())
            k = (math.floor if variant else math.ceil)(len(group) * params['validation_fraction'])
            mapping.update({s: ('val' if i < k else 'train') for i, s in enumerate(ordered)})
        write_csv(output / 'split.csv', [{'subject_id': s, 'split': t} for s, t in sorted(mapping.items())], ['subject_id', 'split'])
        write_json(output / 'design.json', {'seed': params['seed'], 'method': 'seeded hash, per site/label', 'rounding': 'floor' if variant else 'ceil'})
    elif cid == 'N08':
        x = np.load(case.input('features.npy')); train = [i for i, s in enumerate(read_json(case.input('split.json'))) if s == 'train']
        mean = x[train].mean(axis=0); scale = x[train].std(axis=0, ddof=variant); scale[scale == 0] = 1
        np.save(output / 'scaled.npy', (x - mean) / scale)
        write_json(output / 'scaler.json', {'mean': mean.tolist(), 'scale': scale.tolist(), 'fit_rows': train, 'statistical_conventions': {'ddof': variant}})
    elif cid in ('N09', 'N10'):
        run_tool(case, output, receipts)
    elif cid == 'N12':
        if variant:
            write_json(output / 'qc.json', {'missing': ['timeseries.npy'], 'analysis_performed': False})
        finish(case, output, 'blocked', reason='Input timeseries.npy is unavailable', missing_inputs=['timeseries.npy'])
        return
    finish(case, output)


def mutate(case, output, kind):
    cid = case.id
    if kind == 'wrong_identity':
        p = output / 'result.json'; state = read_json(p); state['case_id'] = 'OTHER'; write_json(p, state)
        return
    if cid == 'N01':
        rows = read_csv(output / 'matched.csv'); rows[0]['label'] = -999
        write_csv(output / 'matched.csv', rows, ['subject_id', 'label'])
    elif cid == 'N02':
        write_json(output / 'staging.json', {'source_sha256': '0' * 64, 'shape': [1, 1, 1]})
    elif cid in ('N03', 'N04'):
        p = output / 'mask.nii.gz'; img = nib.load(p); a = np.asarray(img.dataobj).copy(); a.flat[0] = 1 - a.flat[0]
        nib.save(nib.Nifti1Image(a, img.affine), p)
    elif cid in ('N05', 'N11'):
        p = output / 'fc.npy'; x = np.load(p); x[0, 1] = x[1, 0] = 5; np.save(p, x)
    elif cid == 'N06':
        rows = read_csv(output / 'edges.csv'); rows[0]['weight'] = float(rows[0]['weight']) + 1
        write_csv(output / 'edges.csv', rows, ['i', 'j', 'weight'])
    elif cid == 'N07':
        rows = read_csv(output / 'split.csv'); site = read_json(case.input('parameters.json'))['held_out_site']
        held = {r['subject_id'] for r in read_csv(case.input('subjects.csv')) if r['site'] == site}
        next(r for r in rows if r['subject_id'] in held)['split'] = 'train'
        write_csv(output / 'split.csv', rows, ['subject_id', 'split'])
    elif cid == 'N08':
        p = output / 'scaler.json'; state = read_json(p)
        state['fit_rows'] = list(range(len(np.load(case.input('features.npy'))))); write_json(p, state)
    elif cid in ('N09', 'N10'):
        p = output / 'stage.json'; state = read_json(p); state['input_sha256'] = '0' * 64; write_json(p, state)
    elif cid == 'N12':
        np.save(output / 'fc.npy', np.eye(116)); finish(case, output, 'complete')
        return
    state = read_json(output / 'result.json')
    extra = {k: v for k, v in state.items() if k not in ('case_id', 'status', 'input_sha256', 'artifacts')}
    finish(case, output, state['status'], **extra)


def qualify(packet, destination):
    destination.mkdir(parents=True, exist_ok=False)
    digest = sha(packet / 'INDEX.json'); records = []
    for entry in read_json(packet / 'INDEX.json')['cases']:
        case = load_case(packet, entry['case_id'], digest)
        for variant in (0, 1):
            root = destination / case.id / f'legal_{variant}'
            reference(case, root / 'output', root / 'receipts', variant)
            receipt = root / 'receipts'
            actual = evaluate(case, root / 'output', receipt, receipt_digest(receipt))
            records.append({'case': case.id, 'test': f'legal_{variant}', 'expected': 'PASS', 'actual': actual})
        for kind in ('wrong_identity', 'wrong_semantics'):
            root = destination / case.id / kind
            shutil.copytree(destination / case.id / 'legal_0', root)
            mutate(case, root / 'output', kind)
            actual = evaluate(case, root / 'output', root / 'receipts', receipt_digest(root / 'receipts'))
            records.append({'case': case.id, 'test': kind, 'expected': 'FAIL', 'actual': actual})
    errors = [r for r in records if r['actual']['verdict'] != r['expected']]
    result = {'passed': not errors, 'checks': len(records), 'errors': errors, 'records': records,
              'packet_index_sha256': digest, 'scope': 'engineering reference/alternative/counterexample qualification; no scientific expert acceptance'}
    write_json(destination / 'QUALIFICATION.json', result)
    print({'passed': result['passed'], 'checks': len(records), 'errors': errors})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--packet', type=Path, required=True); p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args(); result = qualify(a.packet.resolve(), a.destination.resolve())
    raise SystemExit(0 if result['passed'] else 1)
