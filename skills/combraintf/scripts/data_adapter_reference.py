"""Com-BrainTF data adapter reference — dense FC + community_ids.

Key design decisions:
1. Com-BrainTF takes the same dense FC matrix as BNT ([B, N, N]).
2. Additionally requires `community_ids: list[int]` of length n_roi to
   partition ROIs into K functional communities before the local transformer.
3. `build_community_ids(roi_names, atlas)` auto-derives the partition:
   - schaefer_*_7net: Yeo 7-network from ROI names + Unknown bucket (K=8)
   - aal_*, destrieux, dk_*, harvard_oxford_*: lobe-based (K=8)
   - other: MD5 hash round-robin (K=user-specified, default 8)
4. `get_community_ids(atlas)` loads one .pt to read roi_names then calls
   build_community_ids — convenience for training scripts.

Data path convention:
    data/braingnn_input/<atlas>/sub-<id>.pt   (fMRI connectivity, shared)

Full implementation: models/combraintf/scripts/data_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_subject_and_community(atlas: str, subject_id: str):
    """Minimal example: load one subject's dense FC and the community_ids list.

    The dense FC pipeline is identical to BNT — see
    skills/bnt/scripts/data_adapter_reference.py.
    """
    from models.combraintf.scripts.data_adapter import get_community_ids

    pt_path = ROOT / "data" / "braingnn_input" / atlas / f"sub-{subject_id}.pt"
    sub_pt = torch.load(pt_path, weights_only=False)

    # Step 1: FC (Fisher-z) -> Pearson r, zero diagonal
    fc_z = sub_pt.get("fc_matrix", sub_pt.get("node_features"))
    fc_z = torch.as_tensor(fc_z).float()
    corr = torch.tanh(fc_z)
    corr.fill_diagonal_(0.0)

    # Step 2: derive community ids from atlas (Yeo 7-net / lobe / hash fallback)
    community_ids = get_community_ids(atlas)

    return corr, community_ids


if __name__ == "__main__":
    atlas = sys.argv[1] if len(sys.argv) > 1 else "schaefer_200_7net"
    from models.bnt.scripts.data_adapter import scan_available_subjects
    subs = scan_available_subjects(atlas)
    if not subs:
        print(f"No subjects found for atlas '{atlas}'")
        sys.exit(1)
    fc, cids = load_subject_and_community(atlas, subs[0])
    counts = {c: cids.count(c) for c in sorted(set(cids))}
    print(f"Subject: {subs[0]}")
    print(f"  FC shape: {fc.shape}")
    print(f"  Communities (K={len(set(cids))}): {counts}")
