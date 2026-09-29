"""Independent NumPy reads and numerical/identity checks of Node-built fixtures."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    assert manifest['synthetic'] and not manifest['downloaded']
    for entry in manifest['files']:
        data = (root / entry['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert len(data) == entry['bytes']
        if 'shape' in entry:
            x = np.load(root / entry['path'], allow_pickle=False)
            assert list(x.shape) == entry['shape']
            assert np.isfinite(x).all()
    for dataset in ['adhd200', 'abide', 'adni', 'seediv']:
        ready = root / dataset / 'model_ready'
        with (ready / 'subjects.csv').open(newline='') as handle:
            subjects = list(csv.DictReader(handle))
        splits = json.loads((ready / 'splits.json').read_text())
        groups = [{subjects[i]['subject_id'] for i in splits[k]} for k in ['train', 'validation', 'test']]
        assert [len(x) for x in groups] == [8, 2, 2]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert sorted(sum(splits.values(), [])) == list(range(len(subjects)))
        assert {subjects[i]['site'] for i in splits['test']} == {'C'}
        labels = np.load(ready / 'labels.npy', allow_pickle=False)
        assert labels.dtype == np.int64
        assert labels.tolist() == [int(row['label']) for row in subjects]
        if dataset != 'seediv':
            ts = np.load(root / dataset / 'source/roi_timeseries.npy')
            fc = np.load(ready / 'fc_z.npy')
            for i in range(12):
                r = np.clip(np.corrcoef(ts[i].T), -.99999, .99999)
                expected = np.arctanh(r)
                np.fill_diagonal(expected, 0)
                np.testing.assert_allclose(fc[i], expected, atol=1e-6)
            t1 = np.load(ready / 't1.npy').reshape(12, -1)
            np.testing.assert_allclose(t1.mean(axis=1), 0, atol=1e-5)
            np.testing.assert_allclose(t1.std(axis=1), 1, atol=1e-5)
        else:
            raw = np.load(root / dataset / 'source/epochs.npy')
            x = np.load(ready / 'epochs.npy')
            train = raw[splits['train']].astype('float64')
            mean, std = train.mean(axis=(0, 2)), train.std(axis=(0, 2))
            np.testing.assert_allclose(x, (raw - mean[None, :, None]) / std[None, :, None], atol=1e-6)
    ast.parse((root / 'load_demo.py').read_text(encoding='utf-8'))
    return dict(ok=True, files=len(manifest['files']), datasets=4, hash_checks=True,
                numpy_load=True, subject_disjoint=True, fc_recomputed=True,
                training_only_eeg_normalization=True, scientific_validation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2] / 'data/demo')
    print(json.dumps(verify(parser.parse_args().root)))
