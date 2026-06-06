"""Data adapter for Com-BrainTF.

Reuses BNT's dense FC dataset and adds a `build_community_ids(atlas)` helper
that derives a coarse community partition from ROI names. The resulting
list[int] is passed to the model so it can rearrange ROIs by community before
the local transformers.

Supported atlases (community partition):
  schaefer_*_7net : Yeo 7 networks parsed from "7Networks_<HEMI>_<NET>_<i>".
  aal_116, aal3_166: 8 lobe-like groups (Frontal/Parietal/Temporal/Occipital/
                     Limbic/Subcortical/Cerebellum/Other).
  destrieux_148, dk_112, harvard_oxford_*: lobe heuristic from Destrieux/DK
                     prefixes (frontal, parietal, temporal, occipital, etc.).
  basc_122, cc200, glasser_360, power_264, msdl_39: no name-based partition;
                     we fall back to a deterministic round-robin into
                     `n_communities` groups (set via `n_communities` kwarg).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

import torch

from models.bnt.scripts.data_adapter import (
    BNTDataset,
    bnt_collate,
    build_labels_from_csv,
)


SCHAEFER_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                     "Limbic", "Cont", "Default"]


def _schaefer_community(name: str) -> int:
    # "7Networks_LH_Vis_1" -> "Vis"
    parts = name.split("_")
    for i, p in enumerate(parts):
        if p in SCHAEFER_NETWORKS:
            return SCHAEFER_NETWORKS.index(p)
    return len(SCHAEFER_NETWORKS)  # unknown bucket


_AAL_LOBES = {
    "Frontal": 0, "Precentral": 0, "Rolandic": 0, "Supp_Motor": 0, "Olfactory": 0,
    "Rectus": 0, "OFC": 0,
    "Parietal": 1, "Postcentral": 1, "Precuneus": 1, "SupraMarginal": 1, "Angular": 1,
    "Temporal": 2, "Heschl": 2, "Fusiform": 2,
    "Occipital": 3, "Calcarine": 3, "Cuneus": 3, "Lingual": 3,
    "Cingulum": 4, "Hippocampus": 4, "ParaHippocampal": 4, "Amygdala": 4,
    "Insula": 4,
    "Thalamus": 5, "Caudate": 5, "Putamen": 5, "Pallidum": 5,
    "Cerebelum": 6, "Cerebellum": 6, "Vermis": 6,
}


def _aal_community(name: str) -> int:
    for prefix, cid in _AAL_LOBES.items():
        if prefix in name:
            return cid
    return 7  # other


def _hash_community(name: str, k: int) -> int:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % k


def build_community_ids(roi_names: List[str], atlas: str,
                        n_communities: Optional[int] = None) -> List[int]:
    """Return list[int] of community ids per ROI, dropping background labels."""
    if atlas.startswith("schaefer_") and "_7net" in atlas:
        return [_schaefer_community(n) for n in roi_names]
    if atlas.startswith("aal"):
        return [_aal_community(n) for n in roi_names]
    if atlas.startswith("destrieux") or atlas.startswith("dk_") or \
            atlas.startswith("harvard_oxford"):
        return [_aal_community(n) for n in roi_names]
    # Fallback for atlases without semantic labels
    k = n_communities or 8
    return [_hash_community(n, k) for n in roi_names]


def get_community_ids(atlas: str, n_communities: Optional[int] = None,
                      data_root: Optional[Path] = None) -> List[int]:
    """Load one subject from data/braingnn_input/<atlas>/, derive community ids."""
    root = data_root or Path(__file__).resolve().parents[3] / "data" / "braingnn_input"
    atlas_dir = root / atlas
    pt_files = sorted(atlas_dir.glob("sub-*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"no .pt files in {atlas_dir}")
    sub = torch.load(pt_files[0], weights_only=False)
    roi_names = sub.get("roi_names", [])
    if not roi_names:
        n = int(sub.get("n_rois", 0))
        roi_names = [f"ROI_{i}" for i in range(n)]
    return build_community_ids(list(roi_names), atlas, n_communities)


__all__ = [
    "BNTDataset", "bnt_collate", "build_labels_from_csv",
    "build_community_ids", "get_community_ids",
]
