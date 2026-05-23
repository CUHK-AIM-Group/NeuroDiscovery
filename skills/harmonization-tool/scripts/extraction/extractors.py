"""ROI / FC extractors.

Builds nilearn maskers from AtlasSpec (cached per atlas+target_affine) and
returns (T, R) ROI time series + (R, R) Pearson FC.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.maskers import (NiftiLabelsMasker, NiftiMapsMasker,
                             NiftiSpheresMasker)
import nibabel as nib

from atlas_registry import AtlasSpec


def _load_power_coords(csv_path: Path) -> list[tuple[float, float, float]]:
    df = pd.read_csv(csv_path)
    return [(float(r.x), float(r.y), float(r.z)) for r in df.itertuples()]


@lru_cache(maxsize=64)
def _resampled_atlas_path(
    atlas_path_str: str, target_signature: str
) -> str:
    """resample atlas to target shape+affine; keyed by (atlas, target_signature)
    so we hit the cache when the BOLD grid is consistent across subjects."""
    raise NotImplementedError("use build_masker which handles caching directly")


class AtlasMasker:
    """Wraps a nilearn masker pre-fitted to a reference BOLD image.

    Use one instance per (atlas, dataset). The first call to fit_to_reference()
    aligns the atlas to the BOLD grid; subsequent extract() calls reuse it.
    """

    def __init__(self, spec: AtlasSpec):
        self.spec = spec
        self._masker = None
        self._n_rois_realized = None

    def fit_to_reference(self, ref_img: nib.Nifti1Image) -> "AtlasMasker":
        spec = self.spec
        if spec.kind == "labels":
            atlas_img = nib.load(str(spec.image_path))
            atlas_resampled = resample_to_img(
                atlas_img, ref_img, interpolation="nearest"
            )
            masker = NiftiLabelsMasker(
                labels_img=atlas_resampled,
                standardize=False,
                resampling_target=None,
                strategy="mean",
            )
            masker.fit()
            self._masker = masker
            unique = np.unique(atlas_resampled.get_fdata().astype(np.int32))
            self._n_rois_realized = int(len(unique) - 1)
        elif spec.kind == "maps":
            atlas_img = nib.load(str(spec.image_path))
            atlas_resampled = resample_to_img(
                atlas_img, ref_img, interpolation="continuous"
            )
            masker = NiftiMapsMasker(
                maps_img=atlas_resampled,
                standardize=False,
                resampling_target=None,
            )
            masker.fit()
            self._masker = masker
            self._n_rois_realized = atlas_resampled.shape[-1]
        elif spec.kind == "spheres":
            coords = _load_power_coords(spec.coords_npy)
            masker = NiftiSpheresMasker(
                seeds=coords,
                radius=spec.radius_mm,
                standardize=False,
                allow_overlap=True,
            )
            masker.fit()
            self._masker = masker
            self._n_rois_realized = len(coords)
        else:
            raise ValueError(f"unknown kind: {spec.kind}")
        return self

    @property
    def n_rois(self) -> int:
        return self._n_rois_realized or self.spec.n_rois

    def extract(self, bold_img: nib.Nifti1Image) -> np.ndarray:
        """Return (T, R) ROI time series, float32."""
        if self._masker is None:
            raise RuntimeError("call fit_to_reference() before extract()")
        ts = self._masker.transform(bold_img)
        return np.asarray(ts, dtype=np.float32)


def fc_from_roi(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation FC; (T, R) -> (R, R) float32, zero-diag, NaN-safe."""
    if ts.shape[0] < 2:
        r = ts.shape[1]
        return np.zeros((r, r), dtype=np.float32)
    x = ts - ts.mean(axis=0, keepdims=True)
    s = x.std(axis=0, ddof=0, keepdims=True)
    s[s == 0] = 1.0
    x = x / s
    fc = (x.T @ x) / float(ts.shape[0])
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    return fc
