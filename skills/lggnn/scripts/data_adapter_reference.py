"""LG-GNN data adapter reference — reuses BrainGNN's PyG format.

Key design decisions:
1. LG-GNN's LocalGNN takes the same PyG Data graph as BrainGNN (x=[N,N] FC
   rows, edge_index, edge_attr).
2. No new data adapter needed — `models/lggnn/scripts/data_adapter.py` is a
   thin re-export of `models.braingnn.scripts.data_adapter`.
3. SABP pooling operates on the graph internally; the data layer doesn't need
   to know about ROI selection.

Data path convention:
    data/braingnn_input/<atlas>/sub-<id>.pt   (fMRI connectivity, shared with BrainGNN)

Full implementation: models/lggnn/scripts/data_adapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_single_subject_as_pyg(atlas: str, subject_id: str) -> Data:
    """Minimal example: load one subject as a PyG Data object for LG-GNN.

    Identical pipeline to BrainGNN — see skills/brain_gnn/scripts/data_adapter_reference.py
    for the full Fisher-z inversion + full-graph construction logic.
    """
    from models.braingnn.scripts.data_adapter import _pyg_data_from_subject

    pt_path = ROOT / "data" / "braingnn_input" / atlas / f"sub-{subject_id}.pt"
    sub_pt = torch.load(pt_path, weights_only=False)
    return _pyg_data_from_subject(sub_pt)


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
