"""Materialize twelve finite development cases from existing local/NAS data.

No original task, source data, accepted split or frozen campaign is modified.
Raw data is kept in the local ignored packet, never in the source repository.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from .common import REVISION, environment, file_map, read_csv, read_json, sha, write_csv, write_json

SPECS = [
    ('E01', 'data_identity', 'normal', 'Subject and age-label join', ['matched.csv'],
     'Join every manifest subject exactly once to its age label. Preserve subject IDs and numeric ages; no silent omission, duplicate or extra subject. matched.csv columns: subject_id,label.'),
    ('E02', 'data_identity', 'normal', 'Anatomical image staging', ['sub-case001/anat/sub-case001_T1w.nii.gz', 'staging.json'],
     'Copy anatomical.nii.gz byte-for-byte to sub-case001/anat/sub-case001_T1w.nii.gz. staging.json must give source_sha256 and shape from the NIfTI. This tests a controlled BIDS-style layout, not full BIDS validation or reconstruction.'),
    ('E03', 'image_space_math', 'normal', 'Atlas map threshold and volume', ['mask.nii.gz', 'stats.json'],
     'Threshold map.nii.gz with >= the threshold in parameters.json. Save an exactly binary uint8 mask in the same affine/grid. stats.json: voxel_count and volume_ml = count * abs(det(affine[:3,:3])) / 1000. Map weights are not asserted to be probabilities.'),
    ('E04', 'image_space_math', 'recoverable_fault', 'Resample a flipped voxel grid', ['mask.nii.gz'],
     'source_mask.nii.gz and target_grid.nii.gz describe different voxel grids. Resample the source into the target world-coordinate grid with nearest-neighbor interpolation and zero outside. Copying only the affine is incorrect. Save a binary uint8 mask.'),
    ('E05', 'fc_graphs', 'normal', 'Pearson FC from ABIDE ROI series', ['fc.npy'],
     'Compute Pearson correlations between columns of timeseries.npy, using all rows without filtering. Nonconstant ROI diagonal is 1; constant ROI rows/columns are 0. Output float FC, symmetric, finite, with absolute error <= 1e-6.'),
    ('E06', 'fc_graphs', 'normal', 'Deterministic connectivity graph', ['edges.csv', 'nodes.csv'],
     'From fc.npy use i<j, rank by descending absolute weight, then ascending i,j. Keep ceil(density*n*(n-1)/2) edges. edges.csv columns i,j,weight (original signed weight). nodes.csv columns node,degree,strength; strength sums absolute incident weights. Density is in parameters.json.'),
    ('E07', 'split_statistics', 'normal', 'Site-held-out subject split', ['split.csv'],
     'Use every subjects.csv row exactly once. All rows at held_out_site in parameters.json are test. For each other (site,label) group, rank subject IDs by SHA256(seed+"|"+subject_id), ascending; first ceil(group_size*validation_fraction) are val, others train. split.csv columns subject_id,split. No subject overlap or site leakage.'),
    ('E08', 'split_statistics', 'recoverable_fault', 'Repair an all-data scaler', ['scaled.npy', 'scaler.json'],
     'provided_scaler.json was intentionally fitted on all rows and must not be reused. Fit mean and population standard deviation (ddof=0) on indices marked train in split.json only; use scale=1 for exact zero std. Transform all features.npy rows. scaler.json contains mean,scale,fit_rows; scaled.npy preserves input shape. Absolute error <= 1e-6.'),
    ('E09', 'environment_process', 'normal', 'Execute a bounded QC tool', ['stage.json'],
     'Run the supplied qc_tool.py on values.npy through the common recorded process host with mode=normal. Submit stage.json and result.json. The evaluator requires a real successful host receipt bound to the Python executable, supplied tool, exact input/output and per-run nonce. A handwritten stage.json alone does not pass.'),
    ('E10', 'environment_process', 'recoverable_fault', 'Recover from stale process output', ['stage.json'],
     'The supplied stale_result.json belongs to another input. Execute qc_tool.py through the common recorded host using mode=repair, and use its new stage.json. mode=probe intentionally exits 7 after writing an unqualified partial output; a probe is optional. Only a receipt-bound successful repair counts.'),
    ('E11', 'workflow_delivery', 'normal', 'ROI normalization, FC and QC delivery', ['timeseries_z.npy', 'fc.npy', 'qc.json'],
     'Normalize each timeseries.npy column with population std (zero std becomes scale=1). Save timeseries_z.npy, then compute Pearson FC with the E05 constant-ROI policy. qc.json: frames,rois,finite=true. All three artifacts must agree with the supplied input; absolute error <= 1e-6.'),
    ('E12', 'workflow_delivery', 'must_block', 'Stop on a missing required series', [],
     'required_inputs.json requires timeseries.npy, which is intentionally absent. Return result.json with status=blocked, reason_code=MISSING_REQUIRED_INPUT, missing_inputs=["timeseries.npy"], and artifacts={}. Do not invent series, FC or success outputs.'),
]

TOOL = '''import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--input',required=True); p.add_argument('--output',required=True)
p.add_argument('--mode',choices=['normal','probe','repair'],required=True)
p.add_argument('--nonce',required=True)
a=p.parse_args(); x=np.load(a.input,allow_pickle=False)
if a.mode=='probe':
 Path(a.output).write_text(json.dumps({'qualified':False,'nonce':a.nonce}),encoding='utf-8')
 raise SystemExit(7)
r={'qualified':True,'nonce':a.nonce,'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
 'count':int(x.size),'mean':float(x.mean()),'std':float(x.std(ddof=0)),'pid':os.getpid()}
Path(a.output).write_text(json.dumps(r,allow_nan=False)+'\\n',encoding='utf-8')
print('QC_COMPLETE')
'''


def build(destination, repo, nas):
    destination, repo, nas = Path(destination).resolve(), Path(repo).resolve(), Path(nas)
    if destination.exists():
        raise FileExistsError('Use a new packet directory; partial packets are retained.')
    destination.mkdir(parents=True)
    sources = {}

    def bind(path):
        path = Path(path).resolve()
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('source_changed_while_reading:' + str(path))
        sources[str(path)] = {'path': str(path), 'bytes': after.st_size,
                              'mtime_ns': after.st_mtime_ns, 'sha256': digest}
        return path

    def image(path, data, affine):
        img = nib.Nifti1Image(data, affine)
        img.set_qform(affine, 1); img.set_sform(affine, 1)
        nib.save(img, path)

    labels_path = bind(repo / 'data/hcp_age_labels.csv')
    labels = {r['subject_id']: float(r['label']) for r in read_csv(labels_path)}
    volumes = repo / 'data/t1_volume/aal_116'
    graphs = repo / 'data/braingnn_input/aal_116'
    ids = sorted(s for s in labels if (volumes / ('sub-' + s + '.npz')).is_file()
                 and (graphs / ('sub-' + s + '.pt')).is_file())[:12]
    if len(ids) != 12:
        raise ValueError('insufficient_joined_hcp_inputs')
    features = []
    for sid in ids:
        with np.load(bind(volumes / ('sub-' + sid + '.npz')), allow_pickle=False) as z:
            if str(z['subject_id']) != sid or not np.isfinite(z['gm_fraction']).all():
                raise ValueError('volume_identity_or_values')
            features.append(z['gm_fraction'][:8].astype(np.float64))
    features = np.array(features)
    import torch
    graph_path = bind(graphs / ('sub-' + ids[0] + '.pt'))
    graph = torch.load(graph_path, map_location='cpu', weights_only=True)
    if graph['subject_id'] != ids[0]:
        raise ValueError('graph_identity')
    fc = graph['fc_matrix'].numpy().astype(np.float64)
    if not np.isfinite(fc).all() or not np.allclose(fc, fc.T, atol=1e-6):
        raise ValueError('graph_not_finite_symmetric')

    phenotype_path = bind(repo / 'data/abide/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
    roi_root = repo / 'data/abide/ABIDE_pcp/cpac/filt_noglobal/rois_aal'
    phenotype = [r for r in read_csv(phenotype_path)
                 if (roi_root / (r['FILE_ID'] + '_rois_aal.1D')).is_file()]
    groups = {}
    for r in phenotype:
        if r['DX_GROUP'] in ('1', '2'):
            groups.setdefault((r['SITE_ID'], r['DX_GROUP']), []).append(r)
    sites = sorted(s for s in {k[0] for k in groups}
                   if all(len(groups.get((s, d), [])) >= 6 for d in ('1', '2')))[:4]
    if len(sites) != 4:
        raise ValueError('insufficient_site_label_groups')
    cohort = [r for s in sites for d in ('1', '2')
              for r in sorted(groups[s, d], key=lambda r: r['FILE_ID'])[:6]]
    series_path = bind(roi_root / (cohort[0]['FILE_ID'] + '_rois_aal.1D'))
    series = np.loadtxt(series_path)
    if series.ndim != 2 or not np.isfinite(series).all():
        raise ValueError('invalid_series')

    atlas_path = bind(repo / 'data/atlas/msdl_39/atlas.nii.gz')
    atlas = nib.load(atlas_path)
    atlas_map = np.asarray(atlas.dataobj[..., 0], dtype=np.float32)
    positive = atlas_map[atlas_map > 0]
    threshold = float(np.quantile(positive, .6))
    if not np.isfinite(atlas_map).all() or not 0 < (atlas_map >= threshold).sum() < atlas_map.size:
        raise ValueError('unusable_map_threshold')
    aal_path = bind(repo / 'data/atlas/aal_116/atlas.nii.gz')
    aal = nib.load(aal_path)
    aal_values = np.asarray(aal.dataobj)
    mask = (aal_values == np.unique(aal_values[aal_values > 0])[0]).astype(np.uint8)
    flip = np.eye(4); flip[0, 0] = -1; flip[0, 3] = mask.shape[0] - 1
    anatomical_path = bind(nas / 'ucla_cnp_smri/sub-10159/anat/sub-10159_T1w.nii.gz')
    # Full selected source payload, not merely a header, must decode here.
    anat = nib.load(anatomical_path)
    if len(anat.shape) != 3 or not np.isfinite(np.asarray(anat.dataobj)).all():
        raise ValueError('anatomical_payload_invalid')

    entries = []
    for cid, family, kind, title, outputs, instructions in SPECS:
        case_root = destination / 'public/cases' / cid
        inp = case_root / 'inputs'; inp.mkdir(parents=True)
        if cid == 'E01':
            write_json(inp / 'subjects.json', ids)
            write_csv(inp / 'labels.csv', [{'subject_id': s, 'label': labels[s]} for s in reversed(ids)], ['subject_id', 'label'])
        elif cid == 'E02':
            shutil.copy2(anatomical_path, inp / 'anatomical.nii.gz')
        elif cid == 'E03':
            image(inp / 'map.nii.gz', atlas_map, atlas.affine)
            write_json(inp / 'parameters.json', {'threshold': threshold, 'source_map_index': 0, 'units': 'map_weight'})
        elif cid == 'E04':
            image(inp / 'source_mask.nii.gz', mask[::-1].copy(), aal.affine @ flip)
            image(inp / 'target_grid.nii.gz', np.zeros_like(mask), aal.affine)
        elif cid in ('E05', 'E11'):
            np.save(inp / 'timeseries.npy', series, allow_pickle=False)
        elif cid == 'E06':
            np.save(inp / 'fc.npy', fc, allow_pickle=False)
            write_json(inp / 'parameters.json', {'density': .1})
        elif cid == 'E07':
            write_csv(inp / 'subjects.csv', [{'subject_id': r['FILE_ID'], 'site': r['SITE_ID'], 'label': r['DX_GROUP']} for r in cohort], ['subject_id', 'site', 'label'])
            write_json(inp / 'parameters.json', {'held_out_site': sites[0], 'validation_fraction': .2, 'seed': 'exec-dev-split-20261019'})
        elif cid == 'E08':
            np.save(inp / 'features.npy', features, allow_pickle=False)
            write_json(inp / 'split.json', ['train'] * 8 + ['val'] * 2 + ['test'] * 2)
            write_json(inp / 'provided_scaler.json', {'mean': features.mean(0).tolist(), 'scale': features.std(0).tolist(), 'fit_rows': list(range(12)), 'injected_fault': 'all_data_fit'})
        elif cid in ('E09', 'E10'):
            np.save(inp / 'values.npy', features[:, 0], allow_pickle=False)
            (inp / 'qc_tool.py').write_text(TOOL, encoding='utf-8')
            if cid == 'E10':
                write_json(inp / 'stale_result.json', {'qualified': True, 'input_sha256': '0' * 64, 'mean': -1, 'nonce': 'old_run'})
        else:
            write_json(inp / 'required_inputs.json', ['timeseries.npy'])
        contract = {'revision': REVISION, 'case_id': cid, 'family_id': family,
                    'input_kind': kind, 'title': title, 'instructions': instructions,
                    'required_outputs': outputs, 'input_sha256': file_map(inp),
                    'float_absolute_tolerance': 1e-6,
                    'exposure': 'development_exposed_not_holdout',
                    'profile': 'python_numpy_nibabel_scipy_cpu_no_network',
                    'time_limit_seconds': 120,
                    'scientific_scope': 'engineering_contract_only_no_training_or_discovery'}
        write_json(case_root / 'CONTRACT.json', contract)
        text = (f'# {cid}: {title}\n\n{instructions}\n\n'
                'Read CONTRACT.json for exact input hashes and required outputs. Write only to the supplied output directory. '
                'Also write result.json with case_id, status (complete or blocked), input_sha256 copied from CONTRACT.json, '
                'and artifacts mapping every output file except result.json to its SHA-256. '
                'Artifact paths are relative to the output directory. Extra explanatory files are allowed if listed. '
                'Inputs are engineering development samples, not a new scientific acceptance cohort.\n')
        (case_root / 'task.md').write_text(text, encoding='utf-8')
        entries.append({'case_id': cid, 'family_id': family, 'input_kind': kind,
                        'title': title, 'contract_sha256': sha(case_root / 'CONTRACT.json'),
                        'task_sha256': sha(case_root / 'task.md'), 'required_outputs': outputs})
        print('BUILT', cid, title, flush=True)
    for row in sources.values():
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('selected_source_changed:' + row['path'])
    write_json(destination / 'SOURCE_BINDINGS.json', {
        'selected_sources': list(sources.values()),
        'cohort_selection': {'hcp_ids': ids, 'abide_sites': sites,
                             'abide_subjects': len(cohort), 'series_subject': cohort[0]['FILE_ID']},
        'derivations': {'map': 'MSDL map0, original weights/affine',
                        'space_fault': 'AAL label mask, x-voxel reversal with affine composed to preserve world coordinates',
                        'features': 'First8 gm_fraction features for12 existing HCP subjects',
                        'all_data_scaler': 'Injected fixture error, original sources unchanged'},
        'qualification': 'local engineering inputs; source authority/license and scientific cohort acceptance not newly established',
        'data_publication_authorized': False})
    write_json(destination / 'AMENDMENT.json', {
        'user_request': '可以', 'work': 'P0 development case inputs, reference solutions and independent oracle qualification',
        'prior_plan': 'docs/NEURORUNTIME_BENCHMARK_REDESIGN_20261019.json',
        'selection_change': 'DWI sampling category deferred because complete upstream inputs are not qualified; actual FC/graph cases replace it in this development packet.',
        'input_mix': {'normal': 8, 'recoverable_fault': 3, 'must_block': 1},
        'model_calls': 0, 'judge_calls': 0, 'training': False,
        'not_a_blind_holdout': True})
    write_json(destination / 'ENVIRONMENT.json', environment())
    write_json(destination / 'INDEX.json', {'revision': REVISION, 'cases': entries,
                                          'case_count': 12, 'family_count': 6})
    print('PACKET', str(destination), 'INDEX_SHA256', sha(destination / 'INDEX.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True, type=Path)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--nas', type=Path, default=Path(r'\\192.168.3.61\data\Public Dataset'))
    a = p.parse_args()
    build(a.destination, a.repo, a.nas)
