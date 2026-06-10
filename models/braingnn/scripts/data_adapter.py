"""Adapter: load NeuroClaw fMRI .pt files (+ optional T1 volume .npz) into
PyTorch Geometric `Data` objects compatible with BrainGNN.

NeuroClaw fMRI output (per subject, per atlas):
    data/braingnn_input/<atlas>/sub-<id>.pt
    {
      subject_id, atlas, n_rois,
      time_series:    [T, n_roi],
      fc_matrix:      [n_roi, n_roi]  (Fisher-z),
      node_features:  [n_roi, n_roi]  (FC rows),
      edge_index:     [2, n_edge],
      edge_attr:      [n_edge, 1],
      roi_names: [n_roi],
    }

NeuroClaw T1 volume output (optional, per subject, per atlas):
    data/t1_volume/<atlas>/sub-<id>.npz
    {
      subject_id, atlas, n_rois, roi_names,
      gm_volume_mm3 [n_roi], gm_fraction [n_roi],
    }

Usage:
    from models.braingnn.scripts.data_adapter import NeuroClawFCDataset
    ds = NeuroClawFCDataset(
        atlas='aal_116',
        labels_csv='path/to/labels.csv',   # columns: subject_id, label
        include_t1=True,
    )
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

ROOT = Path(__file__).resolve().parents[3]  # NeuroClaw/
FMRI_ROOT = ROOT / "data" / "braingnn_input"
T1_ROOT = ROOT / "data" / "t1_volume"


def _pyg_data_from_subject(sub_pt: dict, t1_npz: Optional[dict] = None,
                            label: Optional[float] = None,
                            n_roi_expected: Optional[int] = None,
                            label_dtype: str = "long",
                            roi_mask: Optional[list[bool]] = None,
                            edge_boost: Optional[np.ndarray] = None) -> Data:
    """Convert a NeuroClaw subject dict to a PyG Data object.

    Rebuilds the graph at load time to match the original BrainGNN
    (Li 2020, read_abide_stats_parall.read_sigle_data):

      - Node features x: raw Pearson correlation matrix (n_roi rows, n_roi dims).
        Stored in sub_pt["fc_matrix"] as Fisher-z; we invert with tanh to get
        back Pearson r and zero the diagonal. Optionally append 1-d GM volume.
      - Edge graph: FULL (all i!=j pairs), NOT the 10%-density sparsified one
        we saved. This is because TopKPooling downstream is what does the
        selection; edge sparsification at input-time hurts learning.
      - Edge weights: abs(Pearson r), matching BrainGNN's use of |pcorr| as
        non-negative edge attention weights (softmax in MyNNConv.message
        only makes sense on non-negative values).
    """
    # --- node features: raw Pearson r (no Fisher-z), diag=0 ---
    fc_z = sub_pt.get("fc_matrix")
    if fc_z is None:
        fc_z = sub_pt["node_features"]
    if not torch.is_tensor(fc_z):
        fc_z = torch.as_tensor(fc_z)
    fc_z = fc_z.float()
    n_roi = fc_z.size(0)

    # Invert Fisher-z -> Pearson r; zero diagonal.
    corr = torch.tanh(fc_z)
    corr.fill_diagonal_(0.0)

    # Apply ROI mask (subgraph selection)
    if roi_mask is not None:
        mask_idx = torch.tensor([i for i, m in enumerate(roi_mask) if m], dtype=torch.long)
        corr = corr[mask_idx][:, mask_idx]
        n_roi = corr.size(0)

    if n_roi_expected is not None and n_roi != n_roi_expected:
        raise ValueError(f"subject {sub_pt.get('subject_id')} n_roi={n_roi} vs expected {n_roi_expected}")

    # Optional T1 GM volume as extra node feature
    if t1_npz is not None:
        gm_vol = torch.as_tensor(t1_npz["gm_volume_mm3"]).float()
        if roi_mask is not None:
            gm_vol = gm_vol[mask_idx]
        if gm_vol.numel() == n_roi:
            mu, sd = gm_vol.mean(), gm_vol.std().clamp(min=1e-6)
            gm_vol_z = ((gm_vol - mu) / sd).view(-1, 1)
            x = torch.cat([corr, gm_vol_z], dim=1)
        else:
            warnings.warn(
                f"t1 gm_volume len {gm_vol.numel()} != n_roi {n_roi} for "
                f"{sub_pt.get('subject_id')}; skipping t1 feature"
            )
            x = corr
    else:
        x = corr

    # --- full graph with |corr| as edge weight ---
    abs_corr = corr.abs()

    # Apply edge boost if provided (KG-guided edge weighting)
    if edge_boost is not None:
        boost_t = torch.as_tensor(edge_boost, dtype=torch.float32)
        if roi_mask is not None:
            boost_t = boost_t[mask_idx][:, mask_idx]
        abs_corr = abs_corr * boost_t[:n_roi, :n_roi]

    # remove self-loops by masking diagonal (already 0) and also drop
    # edges with weight exactly 0 to keep the tensor a bit smaller.
    mask = torch.ones(n_roi, n_roi, dtype=torch.bool)
    mask.fill_diagonal_(False)
    # additionally drop numerically-zero edges (e.g. from zero-variance ROIs)
    mask &= abs_corr > 0
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()  # [2, E]
    edge_attr = abs_corr[src, dst].view(-1, 1).float()  # [E, 1]

    # pos = identity (ROI one-hot as pseudo-coord for MyNNConv)
    pos = torch.eye(n_roi)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
    )
    if label is not None:
        if label_dtype == "float":
            data.y = torch.tensor([float(label)], dtype=torch.float)
        else:
            data.y = torch.tensor([int(label)], dtype=torch.long)
    data.subject_id = sub_pt.get("subject_id", "")
    return data


def scan_available_subjects(atlas: str) -> list[str]:
    """List subjects with .pt available for this atlas."""
    atlas_dir = FMRI_ROOT / atlas
    if not atlas_dir.exists():
        return []
    subs = []
    for p in sorted(atlas_dir.glob("sub-*.pt")):
        subs.append(p.stem.replace("sub-", ""))
    return subs


class NeuroClawFCDataset(InMemoryDataset):
    """In-memory PyG dataset built from NeuroClaw preprocessed files.

    Args:
        atlas: atlas name (must match subdir in data/braingnn_input/<atlas>/)
        labels: mapping subject_id -> label (int). If None, uses synthetic
            binary labels based on subject_id parity for smoke testing.
        include_t1: if True, append z-scored GM volume as 1 extra node feature.
        subjects: subset of subject_ids; if None use all available.
        processed_suffix: appended to `processed_file_names` to allow multiple
            variants (e.g. t1=True vs False) to coexist in cache.
    """

    def __init__(self, atlas: str,
                 labels: dict[str, float] | None = None,
                 include_t1: bool = False,
                 subjects: Iterable[str] | None = None,
                 processed_suffix: str = "",
                 label_dtype: str = "long",
                 roi_mask: Optional[list[bool]] = None,
                 edge_boost: Optional[np.ndarray] = None):
        self.atlas = atlas
        self.include_t1 = include_t1
        self.explicit_subjects = list(subjects) if subjects is not None else None
        self.labels = labels or {}
        self.label_dtype = label_dtype
        self.roi_mask = roi_mask
        self.edge_boost = edge_boost
        self._processed_suffix = processed_suffix
        import hashlib
        # Include label_dtype and label content hash to avoid cache collisions
        if label_dtype == "long":
            self._processed_suffix += "_cls"
        labels_key = sorted(self.labels.items())[:5]
        lbl_hash = hashlib.md5(str(labels_key).encode()).hexdigest()[:6]
        self._processed_suffix += f"_l{lbl_hash}"
        if roi_mask is not None:
            mask_hash = hashlib.md5(str(roi_mask).encode()).hexdigest()[:8]
            self._processed_suffix += f"_mask{sum(roi_mask)}_{mask_hash}"
        if edge_boost is not None:
            boost_hash = hashlib.md5(edge_boost.tobytes()).hexdigest()[:8]
            self._processed_suffix += f"_boost{boost_hash}"
        root = ROOT / "data" / "braingnn_cache" / atlas
        root.mkdir(parents=True, exist_ok=True)
        super().__init__(str(root))
        loaded = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices = loaded[0], loaded[1]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        suffix = ""
        if self.include_t1:
            suffix += "_t1"
        suffix += self._processed_suffix
        return [f"pyg_data{suffix}.pt"]

    def download(self):
        pass

    def process(self):
        available = set(scan_available_subjects(self.atlas))
        if self.explicit_subjects is not None:
            subs = [s for s in self.explicit_subjects if s in available]
        else:
            subs = sorted(available)

        data_list: list[Data] = []
        n_roi_seen = None
        for sid in subs:
            pt_path = FMRI_ROOT / self.atlas / f"sub-{sid}.pt"
            if not pt_path.exists():
                continue
            sub_pt = torch.load(pt_path, weights_only=False)

            t1_npz = None
            if self.include_t1:
                t1_path = T1_ROOT / self.atlas / f"sub-{sid}.npz"
                if t1_path.exists():
                    t1_npz = np.load(t1_path, allow_pickle=True)

            label = self.labels.get(sid)
            if label is None:
                if self.label_dtype == "float":
                    # no fallback for regression — skip subjects without label
                    continue
                # smoke-test fallback for numeric subject IDs.
                label = int(sid) % 2

            try:
                d = _pyg_data_from_subject(sub_pt, t1_npz, label, n_roi_seen,
                                           label_dtype=self.label_dtype,
                                           roi_mask=self.roi_mask,
                                           edge_boost=self.edge_boost)
            except ValueError as e:
                warnings.warn(str(e))
                continue
            if n_roi_seen is None:
                n_roi_seen = d.x.size(0)
            data_list.append(d)

        if not data_list:
            raise RuntimeError(
                f"No data found for atlas '{self.atlas}'. "
                f"Expected files under {FMRI_ROOT / self.atlas}/sub-*.pt"
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def build_labels_from_csv(csv_path: Path | str, subject_col: str = "subject_id",
                          label_col: str = "label",
                          label_dtype: str = "long") -> dict[str, float]:
    """Helper: load a mapping subject_id -> label from CSV.

    label_dtype='long' casts labels to int (classification).
    label_dtype='float' keeps labels as float (regression).
    """
    df = pd.read_csv(csv_path, dtype={subject_col: str})
    if subject_col not in df or label_col not in df:
        raise KeyError(f"CSV must have columns '{subject_col}' and '{label_col}'")
    out = {}
    for _, row in df.iterrows():
        sid = str(row[subject_col]).strip()
        # Normalize numeric-looking IDs: "100206.0" -> "100206"
        if sid.endswith(".0"):
            sid = sid[:-2]
        if label_dtype == "float":
            out[sid] = float(row[label_col])
        else:
            out[sid] = int(row[label_col])
    return out
