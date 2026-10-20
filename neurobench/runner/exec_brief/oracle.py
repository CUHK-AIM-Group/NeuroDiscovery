"""Private property-based acceptance, never copied into model workspaces."""
from pathlib import Path
import argparse
import math
import sys

import nibabel as nib
import numpy as np

from neurobench.runner.exec_dev.common import (contained, file_map, load_case, read_csv, read_json,
                                   receipt_digest, sha, write_json)
from neurobench.runner.exec_dev.oracle import Rejected, close, pearson_expected, require


def array(path):
    a = np.load(path, allow_pickle=False)
    require(isinstance(a, np.ndarray), 'not_an_array')
    return a


def binary_image(path, expected, affine):
    img = nib.load(path)
    a = np.asarray(img.dataobj)
    require(np.isin(a, (0, 1)).all(), 'nonbinary_mask')
    require(np.array_equal(a, expected), 'incorrect_region')
    close(img.affine, affine, 'incorrect_space', 1e-5)


def pearson(case, output):
    x, actual = array(case.input('timeseries.npy')), array(output / 'fc.npy')
    expected = pearson_expected(x)
    good = np.std(x, axis=0) > 0
    valid = good[:, None] & good[None, :]
    require(actual.shape == expected.shape, 'fc_shape')
    close(actual[valid], expected[valid], 'incorrect_pearson')
    require(np.all(np.isnan(actual[~valid]) | (np.isfinite(actual[~valid]) & (np.abs(actual[~valid]) <= 1))), 'undefined_representation')
    qc = read_json(output / 'qc.json')
    require(isinstance(qc, dict) and bool(qc), 'quality_record_missing')
    if not good.all():
        require(any(s in str(qc).lower() for s in ('constant', 'undefined', 'variance', 'degenerate')), 'undefined_estimates_undocumented')


def process(case, output, receipts, receipts_sha):
    require(receipts is not None and receipt_digest(receipts) == receipts_sha, 'unbound_process_receipts')
    events = [read_json(p) for p in sorted(Path(receipts).glob('*.json'))]
    require(bool(events), 'actual_execution_missing')
    e = events[-1]
    require(e['returncode'] == 0 and 'QC_COMPLETE' in e['stdout'], 'process_did_not_complete')
    require(e['tool_sha256'] == sha(case.input('analysis.py')), 'wrong_program')
    require(e['input_sha256_before'] == e['input_sha256_after'] == sha(case.input('values.npy')), 'wrong_process_input')
    require(e['python_executable'] == sys.executable and e['python_prefix'] == sys.prefix, 'wrong_environment')
    require(e['command'] == [sys.executable, '-I', '-X', 'utf8', str(case.input('analysis.py')),
                            '--input', str(case.input('values.npy')), '--output', e['output_path'], '--nonce', e['nonce']], 'wrong_process_invocation')
    require(e['output_sha256'] == sha(output / 'stage.json'), 'process_output_not_bound')
    state = read_json(output / 'stage.json')
    require(state['qualified'] is True and state['nonce'] == e['nonce']
            and state['input_sha256'] == sha(case.input('values.npy')), 'incorrect_output_provenance')
    x = array(case.input('values.npy')).ravel()
    mean = math.fsum(map(float, x)) / len(x)
    variance = math.fsum((float(v) - mean) ** 2 for v in x) / len(x)
    require(state['count'] == x.size, 'wrong_count')
    close(state['mean'], mean, 'wrong_mean')
    close(state['std'], math.sqrt(variance), 'wrong_std')


def semantics(case, output, receipts, receipts_sha):
    cid = case.id
    if cid == 'N01':
        subjects, rows = read_json(case.input('subjects.json')), read_csv(output / 'matched.csv')
        expected = {r['subject_id']: float(r['label']) for r in read_csv(case.input('labels.csv'))}
        require(len(rows) == len(subjects) and {r['subject_id'] for r in rows} == set(subjects), 'subject_conservation')
        for row in rows:
            close(float(row['label']), expected[row['subject_id']], 'wrong_age')
    elif cid == 'N02':
        target = output / 'sub-case001/anat/sub-case001_T1w.nii.gz'
        require(sha(target) == sha(case.input('anatomical.nii.gz')), 'image_bytes_changed')
        state = read_json(output / 'staging.json')
        require(state['source_sha256'] == sha(target) and state['shape'] == list(nib.load(target).shape), 'inventory_provenance')
    elif cid == 'N03':
        img = nib.load(case.input('map.nii.gz'))
        expected = np.asarray(img.dataobj) >= read_json(case.input('parameters.json'))['threshold']
        binary_image(output / 'mask.nii.gz', expected, img.affine)
        state = read_json(output / 'stats.json')
        require(state['voxel_count'] == int(expected.sum()), 'voxel_count')
        voxel_volume = abs(np.dot(img.affine[:3, 0], np.cross(img.affine[:3, 1], img.affine[:3, 2])))
        close(state['volume_ml'], expected.sum() * voxel_volume / 1000, 'physical_volume')
    elif cid == 'N04':
        img, target = nib.load(case.input('region.nii.gz')), nib.load(case.input('reference.nii.gz'))
        grid = np.indices(target.shape).reshape(3, -1)
        xyz = target.affine[:3, :3] @ grid + target.affine[:3, 3, None]
        source = np.linalg.inv(img.affine) @ np.vstack((xyz, np.ones(xyz.shape[1])))
        idx = np.floor(source[:3] + .5).astype(int)
        valid = np.all((idx >= 0) & (idx < np.array(img.shape)[:, None]), axis=0)
        expected = np.zeros(grid.shape[1], dtype=np.uint8)
        expected[valid] = np.asarray(img.dataobj)[tuple(idx[:, valid])]
        binary_image(output / 'mask.nii.gz', expected.reshape(target.shape), target.affine)
    elif cid in ('N05', 'N11'):
        pearson(case, output)
        if cid == 'N11':
            x, z = array(case.input('timeseries.npy')), array(output / 'timeseries_z.npy')
            possibilities = []
            for ddof in (0, 1):
                scale = x.std(axis=0, ddof=ddof); scale[scale == 0] = 1
                possibilities.append(np.allclose(z, (x - x.mean(axis=0)) / scale, rtol=0, atol=1e-6))
            require(z.shape == x.shape and np.isfinite(z).all() and any(possibilities), 'incorrect_standardized_series')
    elif cid == 'N06':
        fc = array(case.input('fc.npy')); n = len(fc); pairs = n * (n - 1) // 2
        desired = pairs * read_json(case.input('parameters.json'))['density']
        edges = read_csv(output / 'edges.csv'); actual = {}
        for row in edges:
            i, j, w = int(row['i']), int(row['j']), float(row['weight'])
            require(0 <= i < n and 0 <= j < n and i != j, 'invalid_edge')
            key = tuple(sorted((i, j)))
            require(key not in actual, 'duplicate_edge')
            close(w, fc[key], 'wrong_signed_weight')
            actual[key] = w
        require(abs(len(actual) - desired) <= 1 + 1e-9, 'density_mismatch')
        excluded = [abs(fc[i, j]) for i in range(n) for j in range(i + 1, n) if (i, j) not in actual]
        if actual and excluded:
            require(min(abs(w) for w in actual.values()) + 1e-9 >= max(excluded), 'not_strongest_edges')
        nodes = read_csv(output / 'nodes.csv')
        require(len(nodes) == n and {int(r['node']) for r in nodes} == set(range(n)), 'node_conservation')
        for row in nodes:
            weights = [abs(w) for edge, w in actual.items() if int(row['node']) in edge]
            require(int(row['degree']) == len(weights), 'degree')
            close(float(row['strength']), math.fsum(weights), 'strength')
        require(bool(read_json(output / 'qc.json')), 'conventions_missing')
    elif cid == 'N07':
        subjects, rows = read_csv(case.input('subjects.csv')), read_csv(output / 'split.csv')
        params, design = read_json(case.input('parameters.json')), read_json(output / 'design.json')
        mapping = {r['subject_id']: r['split'] for r in rows}
        require(len(rows) == len(subjects) == len(mapping) and set(mapping) == {r['subject_id'] for r in subjects}, 'split_identity')
        require(all(s in ('train', 'val', 'test') for s in mapping.values()), 'split_labels')
        groups = {}
        for row in subjects:
            split = mapping[row['subject_id']]
            if row['site'] == params['held_out_site']:
                require(split == 'test', 'held_out_site_leakage')
            else:
                require(split != 'test', 'wrong_test_population')
                groups.setdefault((row['site'], row['label']), []).append(split)
        for group in groups.values():
            count = group.count('val')
            require(0 < count < len(group) and abs(count - len(group) * params['validation_fraction']) < 1 + 1e-9, 'invalid_validation_proportion')
        require(params['seed'] in str(design) and bool(design.get('method')) and bool(design.get('rounding')), 'split_design_not_reproducible')
    elif cid == 'N08':
        x, z, state = array(case.input('features.npy')), array(output / 'scaled.npy'), read_json(output / 'scaler.json')
        train = [i for i, s in enumerate(read_json(case.input('split.json'))) if s == 'train']
        require(sorted(state['fit_rows']) == train and len(state['fit_rows']) == len(train), 'fitting_data_leakage')
        close(state['mean'], x[train].mean(axis=0), 'incorrect_fitted_mean')
        scales = np.asarray(state['scale'], dtype=float)
        require(scales.shape == (x.shape[1],) and np.isfinite(scales).all() and (scales > 0).all(), 'invalid_scale')
        constant = x[train].std(axis=0) == 0
        acceptable = any(np.allclose(scales[~constant], x[train].std(axis=0, ddof=d)[~constant], atol=1e-6, rtol=0) for d in (0, 1))
        require(acceptable, 'incorrect_fitted_scale')
        close(z, (x - np.asarray(state['mean'])) / scales, 'transform_not_replayable')
    elif cid in ('N09', 'N10'):
        process(case, output, receipts, receipts_sha)
    elif cid == 'N12':
        result = read_json(output / 'result.json')
        missing = [n for n in read_json(case.input('dataset.json')) if not case.input(n).is_file()]
        require(bool(missing) and all(n in str(result) for n in missing), 'unsupported_block_reason')
        require(not (output / 'fc.npy').exists() and not list(output.glob('*timeseries*')), 'fabricated_analysis')


def evaluate(case, output, receipts=None, receipts_sha=None):
    output = Path(output)
    try:
        require(not output.is_symlink(), 'linked_output')
        state = read_json(output / 'result.json')
        require(state['case_id'] == case.id, 'case_identity')
        expected = 'blocked' if case.id == 'N12' else 'complete'
        require(state['status'] == expected, 'completion_status')
        require(state['input_sha256'] == case.contract['input_sha256'], 'input_provenance')
        actual = file_map(output); actual.pop('result.json', None)
        for name in state['artifacts']:
            contained(output, name)
        require(actual == state['artifacts'], 'artifact_provenance')
        if expected == 'complete':
            require(all(n in actual for n in case.contract['required_outputs']), 'required_output_missing')
        semantics(case, output, receipts, receipts_sha)
        return {'case_id': case.id, 'verdict': 'PASS', 'status': expected}
    except Exception as exc:
        return {'case_id': case.id, 'verdict': 'FAIL', 'reason': str(exc), 'error_type': type(exc).__name__}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--packet', required=True, type=Path); p.add_argument('--case', required=True)
    p.add_argument('--output', required=True, type=Path); p.add_argument('--index-sha', required=True)
    p.add_argument('--receipts', type=Path); p.add_argument('--receipts-sha'); p.add_argument('--verdict', required=True, type=Path)
    a = p.parse_args()
    try:
        verdict = evaluate(load_case(a.packet, a.case, a.index_sha), a.output, a.receipts, a.receipts_sha)
    except Exception as exc:
        verdict = {'case_id': a.case, 'verdict': 'ORACLE_ERROR', 'reason': str(exc), 'error_type': type(exc).__name__}
    write_json(a.verdict, verdict)
    raise SystemExit(0 if verdict['verdict'] == 'PASS' else 1)
