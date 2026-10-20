"""Two legal development implementations. Never imported by the oracle."""
from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .common import finish, load_case, read_csv, read_json, sha, write_csv, write_json
from .host import run_tool


def correlation(x, alternative):
    if alternative:
        centered = x - x.mean(axis=0)
        norm = np.sqrt(np.sum(centered * centered, axis=0))
        normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
        result = normalized.T @ normalized
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            result = np.corrcoef(x, rowvar=False)
        result = np.nan_to_num(result, nan=0.0)
    constant = x.std(axis=0) == 0
    result[constant, :] = 0; result[:, constant] = 0
    np.fill_diagonal(result, (~constant).astype(float))
    return np.clip(result, -1, 1)


def solve(case, output, receipts, variant):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    alt = variant == 'alternative'
    cid = case.id
    if cid == 'E01':
        labels = {r['subject_id']: r['label'] for r in read_csv(case.input('labels.csv'))}
        subjects = read_json(case.input('subjects.json'))
        rows = [{'subject_id': s, 'label': labels[s]} for s in subjects]
        write_csv(output / 'matched.csv', rows[::-1] if alt else rows, ['subject_id', 'label'])
    elif cid == 'E02':
        target = output / 'sub-case001/anat/sub-case001_T1w.nii.gz'
        target.parent.mkdir(parents=True)
        if alt:
            with case.input('anatomical.nii.gz').open('rb') as a, target.open('wb') as b:
                shutil.copyfileobj(a, b, 65536)
        else:
            shutil.copyfile(case.input('anatomical.nii.gz'), target)
        write_json(output / 'staging.json', {'source_sha256': sha(case.input('anatomical.nii.gz')),
                                            'shape': list(nib.load(target).shape)})
    elif cid == 'E03':
        img = nib.load(case.input('map.nii.gz'))
        threshold = read_json(case.input('parameters.json'))['threshold']
        data = np.asarray(img.dataobj)
        mask = np.where(data >= threshold, 1, 0).astype(np.uint8) if alt else (data >= threshold).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, img.affine), output / 'mask.nii.gz')
        volume = np.prod(img.header.get_zooms()[:3]) if alt else abs(np.linalg.det(img.affine[:3, :3]))
        write_json(output / 'stats.json', {'voxel_count': int(mask.sum()), 'volume_ml': float(mask.sum() * volume / 1000)})
    elif cid == 'E04':
        source = nib.load(case.input('source_mask.nii.gz'))
        target = nib.load(case.input('target_grid.nii.gz'))
        if alt:
            coords = np.indices(target.shape).reshape(3, -1)
            affine = np.linalg.solve(source.affine, target.affine)
            where = np.rint(affine[:3, :3] @ coords + affine[:3, 3, None]).astype(int)
            data = np.asarray(source.dataobj)[tuple(where)].reshape(target.shape).astype(np.uint8)
            result = nib.Nifti1Image(data, target.affine)
        else:
            result = resample_from_to(source, target, order=0, mode='constant', cval=0)
            result = nib.Nifti1Image(np.asarray(result.dataobj).astype(np.uint8), target.affine)
        nib.save(result, output / 'mask.nii.gz')
    elif cid in ('E05', 'E11'):
        x = np.load(case.input('timeseries.npy'), allow_pickle=False)
        if cid == 'E11':
            scale = x.std(axis=0); scale[scale == 0] = 1
            z = (x - x.mean(axis=0)) / scale
            np.save(output / 'timeseries_z.npy', z, allow_pickle=False)
            write_json(output / 'qc.json', {'frames': x.shape[0], 'rois': x.shape[1], 'finite': True})
            x = z
        np.save(output / 'fc.npy', correlation(x, alt), allow_pickle=False)
    elif cid == 'E06':
        fc = np.load(case.input('fc.npy'), allow_pickle=False)
        i, j = np.triu_indices(len(fc), 1)
        count = math.ceil(len(i) * read_json(case.input('parameters.json'))['density'])
        if alt:
            order = np.lexsort((j, i, -abs(fc[i, j])))[:count]
        else:
            order = sorted(range(len(i)), key=lambda k: (-abs(fc[i[k], j[k]]), i[k], j[k]))[:count]
        edges = [{'i': int(i[k]), 'j': int(j[k]), 'weight': float(fc[i[k], j[k]])} for k in order]
        degrees = np.zeros(len(fc), dtype=int); strengths = np.zeros(len(fc))
        for row in edges:
            for node in (row['i'], row['j']):
                degrees[node] += 1; strengths[node] += abs(row['weight'])
        write_csv(output / 'edges.csv', edges[::-1] if alt else edges, ['i', 'j', 'weight'])
        write_csv(output / 'nodes.csv', [{'node': n, 'degree': int(degrees[n]), 'strength': strengths[n]} for n in range(len(fc))], ['node', 'degree', 'strength'])
    elif cid == 'E07':
        params = read_json(case.input('parameters.json'))
        groups, assignments = {}, []
        for row in read_csv(case.input('subjects.csv')):
            if row['site'] == params['held_out_site']:
                assignments.append({'subject_id': row['subject_id'], 'split': 'test'})
            else:
                groups.setdefault((row['site'], row['label']), []).append(row['subject_id'])
        for group in groups.values():
            ranked = sorted(group, key=lambda s: hashlib.sha256((params['seed'] + '|' + s).encode()).hexdigest())
            nval = math.ceil(len(ranked) * params['validation_fraction'])
            assignments += [{'subject_id': s, 'split': 'val' if n < nval else 'train'} for n, s in enumerate(ranked)]
        write_csv(output / 'split.csv', assignments[::-1] if alt else assignments, ['subject_id', 'split'])
    elif cid == 'E08':
        x = np.load(case.input('features.npy'), allow_pickle=False)
        train = [i for i, s in enumerate(read_json(case.input('split.json'))) if s == 'train']
        if alt:
            mean = np.array([math.fsum(float(x[i, j]) for i in train) / len(train) for j in range(x.shape[1])])
            scale = np.array([math.sqrt(math.fsum((float(x[i, j]) - mean[j]) ** 2 for i in train) / len(train)) for j in range(x.shape[1])])
        else:
            mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
        scale[scale == 0] = 1
        np.save(output / 'scaled.npy', (x - mean) / scale, allow_pickle=False)
        write_json(output / 'scaler.json', {'mean': mean.tolist(), 'scale': scale.tolist(), 'fit_rows': train})
    elif cid in ('E09', 'E10'):
        if cid == 'E10' and not alt:
            failed = run_tool(case, output, receipts, 'probe')
            if failed['returncode'] != 7:
                raise RuntimeError('expected_known_probe_failure')
        event = run_tool(case, output, receipts, 'repair' if cid == 'E10' else 'normal')
        if event['returncode'] != 0:
            raise RuntimeError('tool_failed')
    elif cid == 'E12':
        finish(case, output, status='blocked', reason_code='MISSING_REQUIRED_INPUT', missing_inputs=['timeseries.npy'])
        return
    else:
        raise ValueError(cid)
    finish(case, output, implementation_variant=variant)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--case', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--receipts', required=True, type=Path)
    p.add_argument('--index-sha', required=True)
    p.add_argument('--variant', choices=['reference', 'alternative'], default='reference')
    a = p.parse_args()
    solve(load_case(a.packet, a.case, a.index_sha), a.output, a.receipts, a.variant)
