"""Cold readback of selected sources, materialized cases and oracle evidence.

Runs separately from builder, references and oracle. It reconstructs derivation
bindings from existing source payloads; it does not import expected-answer code.
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from neurobench.runner.exec_dev.common import file_map, load_case, read_csv, read_json, sha, write_json


def verify(packet, qualification, before, output):
    packet, qualification, output = Path(packet).resolve(), Path(qualification).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Verification outputs are append-only; choose a new path.')
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'pass': bool(condition)})
        if not condition:
            raise AssertionError(name)

    index_sha = sha(packet / 'INDEX.json')
    index = read_json(packet / 'INDEX.json')
    sources = read_json(packet / 'SOURCE_BINDINGS.json')
    source_rows = sources['selected_sources']
    by_name = {Path(r['path']).name: Path(r['path']) for r in source_rows}
    cases = {e['case_id']: load_case(packet, e['case_id'], index_sha) for e in index['cases']}
    check('exactly_twelve_unique_cases', set(cases) == {f'E{i:02d}' for i in range(1, 13)})
    check('six_two_case_families', sorted(Counter(e['family_id'] for e in index['cases']).values()) == [2] * 6)
    check('fixed_input_mix', Counter(e['input_kind'] for e in index['cases']) == {'normal': 8, 'recoverable_fault': 3, 'must_block': 1})
    for row in source_rows:
        path = Path(row['path'])
        check('selected_source:' + row['path'], path.stat().st_size == row['bytes'] and sha(path) == row['sha256'])
    for row in read_json(before)['files']:
        check('preserved_frozen:' + row['path'], sha(row['path']) == row['sha256'])

    ids = sources['cohort_selection']['hcp_ids']
    labels = read_csv(by_name['hcp_age_labels.csv'])
    check('source_age_ids_unique', len({r['subject_id'] for r in labels}) == len(labels))
    labels = {r['subject_id']: float(r['label']) for r in labels}
    selected = read_json(cases['E01'].input('subjects.json'))
    joined = read_csv(cases['E01'].input('labels.csv'))
    check('identity_input_matches_age_source', selected == ids and len(joined) == 12
          and {r['subject_id']: float(r['label']) for r in joined} == {s: labels[s] for s in ids})

    anatomical_source = next(Path(r['path']) for r in source_rows if 'ucla_cnp_smri' in r['path'])
    anatomical_input = cases['E02'].input('anatomical.nii.gz')
    check('anatomical_copy_exact_bytes', sha(anatomical_source) == sha(anatomical_input))
    anatomy = nib.load(anatomical_input)
    check('anatomical_full_payload_decodes_and_is_finite', anatomy.shape == (176, 256, 256)
          and np.isfinite(np.asarray(anatomy.dataobj)).all())

    msdl = next(Path(r['path']) for r in source_rows if 'msdl_39' in r['path'])
    raw_map = nib.load(msdl); fixture_map = nib.load(cases['E03'].input('map.nii.gz'))
    check('map_values_and_world_grid_preserved', np.array_equal(np.asarray(raw_map.dataobj[..., 0], dtype=np.float32), np.asarray(fixture_map.dataobj))
          and np.allclose(raw_map.affine, fixture_map.affine))
    threshold = read_json(cases['E03'].input('parameters.json'))['threshold']
    count = np.count_nonzero(np.asarray(fixture_map.dataobj) >= threshold)
    check('nontrivial_mask_task', 0 < count < np.prod(fixture_map.shape))

    aal = next(Path(r['path']) for r in source_rows if 'aal_116' in r['path'] and r['path'].endswith('.nii.gz'))
    raw_atlas = nib.load(aal); atlas_values = np.asarray(raw_atlas.dataobj)
    label = np.unique(atlas_values[atlas_values > 0])[0]
    mask = (atlas_values == label).astype(np.uint8)
    flipped = nib.load(cases['E04'].input('source_mask.nii.gz'))
    target = nib.load(cases['E04'].input('target_grid.nii.gz'))
    transform = np.eye(4); transform[0, 0] = -1; transform[0, 3] = mask.shape[0] - 1
    check('space_fault_derivation_is_world_preserving_flip', np.array_equal(np.asarray(flipped.dataobj)[::-1], mask)
          and np.allclose(flipped.affine, raw_atlas.affine @ transform) and np.allclose(target.affine, raw_atlas.affine))

    series_path = next(Path(r['path']) for r in source_rows if r['path'].endswith('.1D'))
    raw_series = np.loadtxt(series_path)
    for cid in ('E05', 'E11'):
        check(cid + '_real_series_exact', np.array_equal(np.load(cases[cid].input('timeseries.npy'), allow_pickle=False), raw_series))
    import torch
    graph_path = next(Path(r['path']) for r in source_rows if r['path'].endswith('.pt'))
    graph = torch.load(graph_path, map_location='cpu', weights_only=True)
    check('connectivity_input_matches_bound_graph', graph['subject_id'] == ids[0]
          and np.array_equal(np.load(cases['E06'].input('fc.npy'), allow_pickle=False), graph['fc_matrix'].numpy()))

    phenotype = read_csv(by_name['Phenotypic_V1_0b_preprocessed1.csv'])
    mapping = {r['FILE_ID']: r for r in phenotype if r['FILE_ID'] != 'no_filename'}
    cohort = read_csv(cases['E07'].input('subjects.csv'))
    check('site_cohort_subject_conservation', len(cohort) == len({r['subject_id'] for r in cohort}) == 48)
    for row in cohort:
        source = mapping[row['subject_id']]
        check('site_label_source:' + row['subject_id'], row['site'] == source['SITE_ID'] and row['label'] == source['DX_GROUP'])
    check('balanced_site_label_cells', sorted(Counter((r['site'], r['label']) for r in cohort).values()) == [6] * 8)

    expected_features = []
    for sid in ids:
        with np.load(by_name['sub-' + sid + '.npz'], allow_pickle=False) as z:
            check('volume_subject_identity:' + sid, str(z['subject_id']) == sid and int(z['n_rois']) == 116)
            expected_features.append(z['gm_fraction'][:8].astype(np.float64))
    expected_features = np.array(expected_features)
    x = np.load(cases['E08'].input('features.npy'), allow_pickle=False)
    check('volume_feature_rows_exact', np.array_equal(x, expected_features))
    scaler = read_json(cases['E08'].input('provided_scaler.json'))
    check('leakage_fault_is_present_and_detectable', scaler['fit_rows'] == list(range(12))
          and np.allclose(scaler['mean'], x.mean(0)) and not np.allclose(x.mean(0), x[:8].mean(0), rtol=0, atol=1e-6))
    for cid in ('E09', 'E10'):
        check(cid + '_tool_input_real_values', np.array_equal(np.load(cases[cid].input('values.npy'), allow_pickle=False), x[:, 0]))
    check('missing_input_case_is_explicit', read_json(cases['E12'].input('required_inputs.json')) == ['timeseries.npy']
          and not cases['E12'].input('timeseries.npy').exists())

    report = read_json(qualification / 'QUALIFICATION.json')
    check('qualified_case_index_matches_current', report['index_sha256'] == index_sha and report['source_and_index_stable'])
    check('all_24_legal_deliveries_and_42_bad_variants', report['status'] == 'PASS'
          and report['positive_checks'] == 24 and report['negative_checks'] == 42
          and len(report['checks']) == 66 and all(r['ok'] for r in report['checks']))
    for row in report['checks']:
        verdict = read_json(qualification / 'verdicts' / f"{row['case_id']}_{row['label']}.json")
        check('retained_verdict:' + row['case_id'] + '_' + row['label'], verdict['verdict'] == row['expected'])
        log = read_json(qualification / 'logs' / f"{row['case_id']}_{row['label']}_oracle.json")
        check('oracle_exit:' + row['case_id'] + '_' + row['label'], log['returncode'] == (0 if row['expected'] == 'PASS' else 1))
        if row['expected'] == 'PASS':
            artifacts_root = qualification / 'outputs' / row['case_id'] / row['label']
            result_manifest = read_json(artifacts_root / 'result.json')
            actual_files = file_map(artifacts_root); actual_files.pop('result.json')
            check('positive_artifacts_readback:' + row['case_id'] + '_' + row['label'],
                  actual_files == result_manifest['artifacts']
                  and result_manifest['input_sha256'] == cases[row['case_id']].contract['input_sha256'])
    for rel, digest in report['code_sha256'].items():
        check('qualified_code_unchanged:' + rel, sha(Path('neurobench/runner/exec_dev') / rel) == digest
              and sha(qualification / 'code' / rel) == digest)
    for p in [*Path('neurobench/runner/exec_dev').glob('*.py'), Path('neurobench/tests/test_exec_dev.py'), Path(__file__)]:
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'VERIFIED_P0_DEVELOPMENT_CASES',
              'scope': 'independent program readback of local development preparation; not independent expert/scientific acceptance or model benefit evidence',
              'checks': checks, 'check_count': len(checks), 'packet': str(packet),
              'index_sha256': index_sha, 'source_count': len(source_rows),
              'qualified_case_count': 12, 'legal_deliveries_passed': 24, 'invalid_variants_rejected': 42,
              'frozen_old_bindings_checked': len(read_json(before)['files']),
              'qualification': str(qualification), 'qualification_sha256': sha(qualification / 'QUALIFICATION.json'),
              'new_model_calls': 0, 'new_judge_calls': 0, 'new_scientific_training_runs': 0,
              'limitations': ['Development cases are exposed and share datasets; not twelve independent scientific experiments.',
                             'Only selected source/delivery hashes were checked; no full historical or full-NAS census.',
                             'These checks do not establish model/harness superiority or scientific cohort/license acceptance.']}
    write_json(output, result)
    print('VERIFIED', len(checks), 'checks', str(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--qualification', required=True, type=Path)
    p.add_argument('--frozen-before', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    verify(a.packet, a.qualification, a.frozen_before, a.output)
