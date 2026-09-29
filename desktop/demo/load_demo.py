"""Load the synthetic demo fixtures. No processing of real images or training."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(root, dataset):
    root = Path(root) / dataset / 'model_ready'
    feature = 'epochs' if dataset == 'seediv' else 'fc_z'
    x = np.load(root / (feature + '.npy'), allow_pickle=False)
    y = np.load(root / 'labels.npy', allow_pickle=False)
    splits = json.loads((root / 'splits.json').read_text())
    assert x.shape[0] == len(y) and np.isfinite(x).all()
    return x, y, splits


def export_pt(root, dataset, destination):
    """Map demo12 FC into the project's BrainGNN/BNT/BrainNetCNN contract."""
    import torch
    if dataset == 'seediv':
        raise ValueError('EEG uses epochs, not the MRI graph adapter')
    destination = Path(destination)
    # Preserve existing exported files rather than overwriting a real dataset.
    if destination.exists():
        raise FileExistsError('Use a new export directory')
    x, _, _ = load(root, dataset)
    ts = np.load(Path(root) / dataset / 'source/roi_timeseries.npy', allow_pickle=False)
    destination.mkdir(parents=True)
    for i, fc in enumerate(x):
        fc_z = torch.from_numpy(fc.copy())
        r = torch.tanh(fc_z)
        r.fill_diagonal_(0)
        edge_index = (~torch.eye(len(r), dtype=torch.bool)).nonzero().T.contiguous()
        record = dict(subject_id=f'demo-{i+1:03d}', atlas='demo12', n_rois=len(r),
                      time_series=torch.from_numpy(ts[i].copy()), fc_matrix=fc_z,
                      node_features=fc_z.clone(), edge_index=edge_index,
                      edge_attr=r[edge_index[0], edge_index[1]].abs().unsqueeze(1),
                      roi_names=[f'demo-roi-{j:02d}' for j in range(len(r))],
                      synthetic=True)
        torch.save(record, destination / f'sub-demo-{i+1:03d}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent)
    parser.add_argument('--dataset', choices=['adhd200', 'abide', 'adni', 'seediv'], default='adhd200')
    parser.add_argument('--torch', action='store_true')
    parser.add_argument('--export-pt', type=Path)
    args = parser.parse_args()
    x, y, splits = load(args.root, args.dataset)
    if args.torch:
        import torch
        x, y = torch.from_numpy(x), torch.from_numpy(y)
    if args.export_pt:
        export_pt(args.root, args.dataset, args.export_pt)
    print(json.dumps(dict(synthetic=True, x_shape=list(x.shape), y_shape=list(y.shape),
                          dtype=str(x.dtype), splits={k: len(v) for k, v in splits.items()})))
