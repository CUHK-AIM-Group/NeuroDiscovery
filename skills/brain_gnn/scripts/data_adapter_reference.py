"""BrainGNN data adapter reference — loading NeuroClaw .pt files into PyG format.

Key design decisions (learned from debugging):
1. Fisher-z -> Pearson r via torch.tanh(), diagonal zeroed
2. Full graph (all i!=j pairs) as input — TopKPooling does the selection
3. Edge weights = |Pearson r| (non-negative, required for softmax in MyNNConv)
4. Optional T1 GM volume as z-scored extra node feature column
5. PyG InMemoryDataset with auto-caching (delete cache dir when data changes)

Data path convention:
    data/braingnn_input/<atlas>/sub-<id>.pt   (fMRI connectivity)
    data/t1_volume/<atlas>/sub-<id>.npz       (optional T1 GM volume)

Full implementation: models/braingnn/scripts/data_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_single_subject_as_pyg(atlas: str, subject_id: str,
                                include_t1: bool = False) -> Data:
    """Minimal example: load one subject and build a PyG Data object.

    This demonstrates the exact transformation pipeline that
    NeuroClawFCDataset applies internally.
    """
    fmri_root = ROOT / "data" / "braingnn_input"
    t1_root = ROOT / "data" / "t1_volume"

    pt_path = fmri_root / atlas / f"sub-{subject_id}.pt"
    sub_pt = torch.load(pt_path, weights_only=False)

    # Step 1: Get FC matrix (stored as Fisher-z)
    fc_z = sub_pt.get("fc_matrix", sub_pt.get("node_features"))
    fc_z = torch.as_tensor(fc_z).float()
    n_roi = fc_z.size(0)

    # Step 2: Invert Fisher-z -> Pearson r, zero diagonal
    corr = torch.tanh(fc_z)
    corr.fill_diagonal_(0.0)

    # Step 3: Optional T1 GM volume as extra feature
    x = corr
    if include_t1:
        t1_path = t1_root / atlas / f"sub-{subject_id}.npz"
        if t1_path.exists():
            t1 = np.load(t1_path, allow_pickle=True)
            gm = torch.as_tensor(t1["gm_volume_mm3"]).float()
            if gm.numel() == n_roi:
                mu, sd = gm.mean(), gm.std().clamp(min=1e-6)
                gm_z = ((gm - mu) / sd).view(-1, 1)
                x = torch.cat([corr, gm_z], dim=1)

    # Step 4: Build FULL graph with |corr| as edge weight
    abs_corr = corr.abs()
    mask = torch.ones(n_roi, n_roi, dtype=torch.bool)
    mask.fill_diagonal_(False)
    mask &= abs_corr > 0
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_attr = abs_corr[src, dst].view(-1, 1).float()

    # Step 5: pos = ROI identity (one-hot pseudo-coord for MyNNConv)
    pos = torch.eye(n_roi)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


if __name__ == "__main__":
    atlas = sys.argv[1] if len(sys.argv) > 1 else "aal_116"
    from models.braingnn.scripts.data_adapter import scan_available_subjects
    subs = scan_available_subjects(atlas)
    if not subs:
        print(f"No subjects found for atlas '{atlas}'")
        sys.exit(1)
    data = load_single_subject_as_pyg(atlas, subs[0])
    print(f"Subject: {subs[0]}")
    print(f"  x shape: {data.x.shape}  (n_roi x indim)")
    print(f"  edge_index shape: {data.edge_index.shape}")
    print(f"  edge_attr shape: {data.edge_attr.shape}")
    print(f"  pos shape: {data.pos.shape}")
