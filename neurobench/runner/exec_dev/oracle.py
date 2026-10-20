"""Independent artifact oracle. No reference/solver or model imports.

The caller pins INDEX.json and owns any trusted host receipts. Public specs
define every acceptance requirement; evaluator-private arithmetic stays here.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from .common import contained, file_map, load_case, read_csv, read_json, receipt_digest, sha, write_json


class Rejected(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise Rejected(reason)


def close(actual, expected, reason, tolerance=1e-6):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=0, atol=tolerance), reason)


def array(path):
    x = np.load(path, allow_pickle=False)
    require(isinstance(x, np.ndarray) and np.isfinite(x).all(), 'nonfinite_or_invalid_array')
    return x


def pearson_expected(x):
    # Pairwise scalar arithmetic deliberately differs from both solvers.
    means = [math.fsum(map(float, x[:, j])) / len(x) for j in range(x.shape[1])]
    centered = [x[:, j].astype(float) - means[j] for j in range(x.shape[1])]
    energy = [math.fsum(float(v) ** 2 for v in col) for col in centered]
    result = np.zeros((x.shape[1], x.shape[1]))
    for i in range(len(centered)):
        if energy[i] == 0:
            continue
        result[i, i] = 1
        for j in range(i + 1, len(centered)):
            if energy[j] > 0:
                v = math.fsum(float(a) * float(b) for a, b in zip(centered[i], centered[j]))
                result[i, j] = result[j, i] = max(-1, min(1, v / math.sqrt(energy[i] * energy[j])))
    return result


def check_image(output_path, expected, affine):
    try:
        img = nib.load(output_path)
        values = np.asarray(img.dataobj)
    except (nib.filebasedimages.ImageFileError, OSError, ValueError, EOFError) as e:
        raise Rejected('invalid_candidate_nifti:' + type(e).__name__) from e
    require(img.get_data_dtype() == np.dtype('uint8'), 'mask_not_uint8')
    require(np.isin(values, (0, 1)).all(), 'mask_not_binary')
    close(img.affine, affine, 'wrong_output_affine', 1e-5)
    require(np.array_equal(values, expected), 'wrong_mask_voxels')


def check_process(case, output, receipts, receipts_sha):
    require(receipts is not None and receipts_sha is not None, 'missing_trusted_host_receipts')
    require(receipt_digest(receipts) == receipts_sha, 'trusted_host_receipts_changed')
    events = [read_json(p) for p in sorted(Path(receipts).glob('*.json'))]
    require(bool(events), 'no_actual_process_event')
    e = events[-1]
    mode = 'normal' if case.id == 'E09' else 'repair'
    require(e['returncode'] == 0 and e['mode'] == mode, 'no_successful_required_process')
    require(e['tool_sha256'] == sha(case.input('qc_tool.py')), 'wrong_executed_tool')
    expected_input = sha(case.input('values.npy'))
    require(e['input_sha256_before'] == e['input_sha256_after'] == expected_input, 'wrong_process_input')
    require(Path(e['python_executable']).resolve() == Path(sys.executable).resolve()
            and e['python_prefix'] == sys.prefix, 'wrong_python_environment')
    command = [e['python_executable'], '-I', str(case.input('qc_tool.py')),
               '--input', str(case.input('values.npy')), '--output', e['output_path'],
               '--mode', mode, '--nonce', e['nonce']]
    require(e['command'] == command and 'QC_COMPLETE' in e['stdout'], 'wrong_process_command_or_completion')
    require(e['output_sha256'] == sha(output / 'stage.json'), 'process_output_not_bound')
    stage = read_json(output / 'stage.json')
    require(stage['qualified'] is True and stage['nonce'] == e['nonce']
            and stage['input_sha256'] == expected_input, 'stale_or_unqualified_process_output')
    values = array(case.input('values.npy')).reshape(-1)
    mean = math.fsum(map(float, values)) / len(values)
    variance = math.fsum((float(v) - mean) ** 2 for v in values) / len(values)
    require(stage['count'] == len(values), 'qc_count')
    close(stage['mean'], mean, 'qc_mean')
    close(stage['std'], math.sqrt(variance), 'qc_std')


def check_semantics(case, output, receipts=None, receipts_sha=None):
    cid = case.id
    if cid == 'E01':
        subjects = read_json(case.input('subjects.json'))
        rows = read_csv(output / 'matched.csv')
        expected = {r['subject_id']: float(r['label']) for r in read_csv(case.input('labels.csv'))}
        require(len(rows) == len(subjects) and len({r['subject_id'] for r in rows}) == len(rows), 'subject_conservation')
        require({r['subject_id'] for r in rows} == set(subjects), 'subject_identity')
        for r in rows:
            close(float(r['label']), expected[r['subject_id']], 'wrong_joined_label')
    elif cid == 'E02':
        target = output / 'sub-case001/anat/sub-case001_T1w.nii.gz'
        require(sha(target) == sha(case.input('anatomical.nii.gz')), 'image_bytes_changed')
        stage = read_json(output / 'staging.json')
        require(stage['source_sha256'] == sha(case.input('anatomical.nii.gz'))
                and stage['shape'] == list(nib.load(target).shape), 'staging_identity_or_shape')
    elif cid == 'E03':
        source = nib.load(case.input('map.nii.gz'))
        threshold = read_json(case.input('parameters.json'))['threshold']
        expected = (np.asarray(source.dataobj) >= threshold).astype(np.uint8)
        check_image(output / 'mask.nii.gz', expected, source.affine)
        stats = read_json(output / 'stats.json')
        require(stats['voxel_count'] == int(np.count_nonzero(expected)), 'mask_count')
        volume = abs(np.dot(source.affine[:3, 0], np.cross(source.affine[:3, 1], source.affine[:3, 2])))
        close(stats['volume_ml'], np.count_nonzero(expected) * volume / 1000, 'mask_volume_ml')
    elif cid == 'E04':
        source = nib.load(case.input('source_mask.nii.gz'))
        target = nib.load(case.input('target_grid.nii.gz'))
        grid = np.indices(target.shape).reshape(3, -1)
        world = target.affine[:3, :3] @ grid + target.affine[:3, 3, None]
        where = np.linalg.inv(source.affine) @ np.vstack((world, np.ones(world.shape[1])))
        where = np.floor(where[:3] + .5).astype(int)
        valid = np.all((where >= 0) & (where < np.array(source.shape)[:, None]), axis=0)
        flat = np.zeros(grid.shape[1], dtype=np.uint8)
        flat[valid] = np.asarray(source.dataobj)[tuple(where[:, valid])]
        check_image(output / 'mask.nii.gz', flat.reshape(target.shape), target.affine)
    elif cid in ('E05', 'E11'):
        source = array(case.input('timeseries.npy'))
        close(array(output / 'fc.npy'), pearson_expected(source), 'incorrect_pearson_fc')
        if cid == 'E11':
            means = np.array([math.fsum(map(float, col)) / len(col) for col in source.T])
            var = np.array([math.fsum((float(v) - m) ** 2 for v in col) / len(col)
                            for col, m in zip(source.T, means)])
            scales = np.sqrt(var); scales[scales == 0] = 1
            close(array(output / 'timeseries_z.npy'), (source - means) / scales, 'incorrect_normalized_series')
            qc = read_json(output / 'qc.json')
            require(all(qc.get(k) == v for k, v in {'frames': source.shape[0], 'rois': source.shape[1], 'finite': True}.items()), 'inconsistent_workflow_qc')
    elif cid == 'E06':
        fc = array(case.input('fc.npy'))
        density = read_json(case.input('parameters.json'))['density']
        candidates = [(i, j, float(fc[i, j])) for i in range(len(fc)) for j in range(i + 1, len(fc))]
        ranked = sorted(candidates, key=lambda t: (-abs(t[2]), t[0], t[1]))
        expected = {(i, j): w for i, j, w in ranked[:math.ceil(len(ranked) * density)]}
        rows = read_csv(output / 'edges.csv')
        actual = {(int(r['i']), int(r['j'])): float(r['weight']) for r in rows}
        require(len(actual) == len(rows) and actual.keys() == expected.keys(), 'edge_set_or_duplicates')
        for key in expected:
            close(actual[key], expected[key], 'incorrect_edge_weight')
        nodes = read_csv(output / 'nodes.csv')
        require(len(nodes) == len(fc) and {int(r['node']) for r in nodes} == set(range(len(fc))), 'node_conservation')
        for r in nodes:
            node = int(r['node'])
            weights = [abs(w) for (i, j), w in expected.items() if node in (i, j)]
            require(int(r['degree']) == len(weights), 'incorrect_degree')
            close(float(r['strength']), math.fsum(weights), 'incorrect_strength')
    elif cid == 'E07':
        params, subjects = read_json(case.input('parameters.json')), read_csv(case.input('subjects.csv'))
        actual_rows = read_csv(output / 'split.csv')
        actual = {r['subject_id']: r['split'] for r in actual_rows}
        require(len(actual_rows) == len(subjects) == len(actual)
                and set(actual) == {r['subject_id'] for r in subjects}, 'split_subject_conservation')
        for row in subjects:
            if row['site'] == params['held_out_site']:
                require(actual[row['subject_id']] == 'test', 'held_out_site_leakage')
                continue
            group = [r['subject_id'] for r in subjects if (r['site'], r['label']) == (row['site'], row['label'])]
            hashed = [(hashlib.sha256((params['seed'] + '|' + sid).encode('utf-8')).digest(), sid) for sid in group]
            val = {sid for _, sid in sorted(hashed)[:math.ceil(len(group) * params['validation_fraction'])]}
            require(actual[row['subject_id']] == ('val' if row['subject_id'] in val else 'train'), 'split_protocol_or_leakage')
    elif cid == 'E08':
        x = array(case.input('features.npy'))
        train = [i for i, s in enumerate(read_json(case.input('split.json'))) if s == 'train']
        # Welford population moments, independent of reference numpy std/fsum.
        mean, m2 = np.zeros(x.shape[1]), np.zeros(x.shape[1])
        for n, i in enumerate(train, 1):
            delta = x[i] - mean; mean += delta / n; m2 += delta * (x[i] - mean)
        scale = np.sqrt(m2 / len(train)); scale[scale == 0] = 1
        scaler = read_json(output / 'scaler.json')
        require(scaler['fit_rows'] == train, 'scaler_fit_includes_nontrain_rows')
        close(scaler['mean'], mean, 'wrong_train_mean')
        close(scaler['scale'], scale, 'wrong_train_scale')
        close(array(output / 'scaled.npy'), (x - mean) / scale, 'wrong_transformed_values')
    elif cid in ('E09', 'E10'):
        check_process(case, output, receipts, receipts_sha)
    elif cid == 'E12':
        result = read_json(output / 'result.json')
        require(result.get('reason_code') == 'MISSING_REQUIRED_INPUT'
                and result.get('missing_inputs') == ['timeseries.npy'], 'incorrect_block_reason')
        require(set(p.name for p in output.iterdir()) == {'result.json'}, 'fabricated_outputs_for_missing_input')


def evaluate(case, output, receipts=None, receipts_sha=None):
    output = Path(output)
    if output.is_symlink() or (hasattr(output, 'is_junction') and output.is_junction()):
        return {'case_id': case.id, 'verdict': 'FAIL', 'reason': 'linked_output_root'}
    output = output.resolve()
    try:
        result = read_json(output / 'result.json')
        require(result['case_id'] == case.id, 'wrong_case_identity')
        expected_status = 'blocked' if case.id == 'E12' else 'complete'
        require(result['status'] == expected_status, 'wrong_completion_status')
        require(result['input_sha256'] == case.contract['input_sha256'], 'wrong_input_provenance')
        actual = file_map(output); actual.pop('result.json', None)
        for rel in result['artifacts']:
            contained(output, rel)
        require(actual == result['artifacts'], 'output_manifest_or_hash_mismatch')
        require(all(name in actual for name in case.contract['required_outputs']), 'missing_required_output')
        check_semantics(case, output, receipts, receipts_sha)
        return {'case_id': case.id, 'verdict': 'PASS', 'status': expected_status}
    except (Rejected, ValueError, KeyError, TypeError, OSError, EOFError, IndexError) as e:
        return {'case_id': case.id, 'verdict': 'FAIL', 'reason': str(e), 'error_type': type(e).__name__}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--case', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--index-sha', required=True)
    p.add_argument('--receipts', type=Path)
    p.add_argument('--receipts-sha')
    p.add_argument('--verdict', required=True, type=Path)
    a = p.parse_args()
    try:
        case = load_case(a.packet, a.case, a.index_sha)
        verdict = evaluate(case, a.output, a.receipts, a.receipts_sha)
    except Exception as e:
        verdict = {'case_id': a.case, 'verdict': 'ORACLE_ERROR', 'error_type': type(e).__name__, 'reason': str(e)}
    write_json(a.verdict, verdict)
    raise SystemExit(0 if verdict['verdict'] == 'PASS' else 1)
