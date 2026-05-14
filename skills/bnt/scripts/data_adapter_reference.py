"""BNT data adapter reference — loading NeuroClaw .pt files into dense FC format.

Key design decisions (learned from debugging):
1. Fisher-z -> Pearson r via torch.tanh(), diagonal zeroed
2. Dense FC matrix [N, N] as node features (NOT graph/edge format)
3. Optional T1 GM volume as z-scored extra column (dim N -> N+1)
4. Pure PyTorch Dataset — no PyG dependency
5. Custom bnt_collate returns (fc_batch, y_batch, sid_list)

Data path convention:
    data/braingnn_input/<atlas>/sub-<id>.pt   (fMRI connectivity, shared with BrainGNN)
    data/t1_volume/<atlas>/sub-<id>.npz       (optional T1 GM volume)

Full implementation: models/bnt/scripts/data_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_single_subject_as_dense(atlas: str, subject_id: str,
                                  include_t1: bool = False) -> torch.Tensor:
    """Minimal example: load one subject and build a dense FC matrix for BNT.

    This demonstrates the exact transformation pipeline that
    BNTDataset applies internally.
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

    # Step 3: Optional T1 GM volume as extra column
    if include_t1:
        t1_path = t1_root / atlas / f"sub-{subject_id}.npz"
        if t1_path.exists():
            t1 = np.load(t1_path, allow_pickle=True)
            gm = torch.as_tensor(t1["gm_volume_mm3"]).float()
            if gm.numel() == n_roi:
                mu, sd = gm.mean(), gm.std().clamp(min=1e-6)
                gm_z = ((gm - mu) / sd).view(-1, 1)
                corr = torch.cat([corr, gm_z], dim=1)  # [N, N+1]

    # BNT expects [N, F] where F = N (or N+1 with T1)
    # In batched form: [B, N, F]
    return corr


if __name__ == "__main__":
    atlas = sys.argv[1] if len(sys.argv) > 1 else "aal_116"
    from models.bnt.scripts.data_adapter import scan_available_subjects
    subs = scan_available_subjects(atlas)
    if not subs:
        print(f"No subjects found for atlas '{atlas}'")
        sys.exit(1)
    fc = load_single_subject_as_dense(atlas, subs[0])
    print(f"Subject: {subs[0]}")
    print(f"  FC shape: {fc.shape}  (n_roi x feature_dim)")
    print(f"  Value range: [{fc.min():.3f}, {fc.max():.3f}]")
    print(f"  Diagonal (should be 0): {fc.diagonal()[:5].tolist()}")
