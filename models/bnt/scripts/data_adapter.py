"""Dataset adapter for BrainNetworkTransformer.

BNT takes a dense FC matrix [N_roi, N_roi] per subject, NOT a PyG graph.
Labels are scalar (regression) or class index (classification).

This is deliberately kept simple — pure PyTorch Dataset + DataLoader,
no PyG dependency. Optionally can append T1 GM volume as an extra column
to each row of the FC matrix (increasing node feature dim from N to N+1),
mirroring the BrainGNN data_adapter option.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
FMRI_ROOT = ROOT / "data" / "braingnn_input"
T1_ROOT = ROOT / "data" / "t1_volume"


def _load_subject(atlas: str, sid: str, include_t1: bool,
                   roi_mask: list[bool] | None = None,
                   edge_boost: np.ndarray | None = None):
    pt_path = FMRI_ROOT / atlas / f"sub-{sid}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)
    sub_pt = torch.load(pt_path, weights_only=False)

    fc_z = sub_pt.get("fc_matrix")
    if fc_z is None:
        fc_z = sub_pt["node_features"]
    if not torch.is_tensor(fc_z):
        fc_z = torch.as_tensor(fc_z)
    fc_z = fc_z.float()
    # Invert Fisher-z -> Pearson r, zero diagonal (BNT paper uses Pearson corr)
    corr = torch.tanh(fc_z)
    corr.fill_diagonal_(0.0)
    n_roi = corr.size(0)

    # Apply edge boost before ROI mask (boost is in full-ROI space)
    if edge_boost is not None:
        boost_t = torch.as_tensor(edge_boost, dtype=torch.float32)
        corr = corr * boost_t[:n_roi, :n_roi]
        corr.fill_diagonal_(0.0)

    # Apply ROI mask (subgraph selection)
    if roi_mask is not None:
        mask_idx = torch.tensor([i for i, m in enumerate(roi_mask) if m], dtype=torch.long)
        corr = corr[mask_idx][:, mask_idx]
        n_roi = corr.size(0)

    if include_t1:
        t1_path = T1_ROOT / atlas / f"sub-{sid}.npz"
        if t1_path.exists():
            t1 = np.load(t1_path, allow_pickle=True)
            gm = torch.as_tensor(t1["gm_volume_mm3"]).float()
            if roi_mask is not None:
                gm = gm[mask_idx]
            if gm.numel() == n_roi:
                mu, sd = gm.mean(), gm.std().clamp(min=1e-6)
                gm_z = ((gm - mu) / sd).view(-1, 1)
                corr = torch.cat([corr, gm_z], dim=1)
            else:
                warnings.warn(
                    f"t1 gm_volume len {gm.numel()} != n_roi {n_roi} for {sid}; "
                    f"skipping t1 for this subject"
                )
    return corr, n_roi


def scan_available_subjects(atlas: str) -> list[str]:
    atlas_dir = FMRI_ROOT / atlas
    if not atlas_dir.exists():
        return []
    return sorted(p.stem.replace("sub-", "") for p in atlas_dir.glob("sub-*.pt"))


def build_labels_from_csv(csv_path: Path | str, subject_col: str = "subject_id",
                          label_col: str = "label",
                          label_dtype: str = "long") -> dict[str, float]:
    df = pd.read_csv(csv_path, dtype={subject_col: str})
    if subject_col not in df or label_col not in df:
        raise KeyError(f"CSV must have columns '{subject_col}' and '{label_col}'")
    out = {}
    for _, row in df.iterrows():
        sid = str(row[subject_col]).strip()
        if sid.endswith(".0"):
            sid = sid[:-2]
        if label_dtype == "float":
            out[sid] = float(row[label_col])
        else:
            out[sid] = int(row[label_col])
    return out


class BNTDataset(Dataset):
    """Subjects -> (FC matrix, label)."""

    def __init__(self, atlas: str,
                 labels: dict[str, float] | None = None,
                 include_t1: bool = False,
                 subjects: Iterable[str] | None = None,
                 label_dtype: str = "long",
                 roi_mask: list[bool] | None = None,
                 edge_boost: np.ndarray | None = None):
        self.atlas = atlas
        self.include_t1 = include_t1
        self.label_dtype = label_dtype
        self.roi_mask = roi_mask
        self.edge_boost = edge_boost
        labels = labels or {}

        available = set(scan_available_subjects(atlas))
        if subjects is not None:
            subs = [s for s in subjects if s in available]
        else:
            subs = sorted(available)

        self.samples: list[tuple[str, torch.Tensor, float]] = []
        n_roi_seen = None
        for sid in subs:
            if sid not in labels and label_dtype == "float":
                continue
            try:
                corr, n_roi = _load_subject(atlas, sid, include_t1,
                                            roi_mask=roi_mask,
                                            edge_boost=edge_boost)
            except FileNotFoundError:
                continue
            if n_roi_seen is None:
                n_roi_seen = corr.size(0)
            elif corr.size(0) != n_roi_seen:
                warnings.warn(f"subject {sid} has inconsistent n_roi; skipping")
                continue
            label = labels.get(sid, int(sid) % 2)
            self.samples.append((sid, corr, label))
        self.n_roi = n_roi_seen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, fc, label = self.samples[idx]
        if self.label_dtype == "float":
            y = torch.tensor(float(label), dtype=torch.float)
        else:
            y = torch.tensor(int(label), dtype=torch.long)
        return fc, y, sid


def bnt_collate(batch):
    fcs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    sids = [b[2] for b in batch]
    return fcs, ys, sids
