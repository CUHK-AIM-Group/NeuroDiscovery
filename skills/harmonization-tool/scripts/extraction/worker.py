"""Streaming per-subject worker.

End-to-end per subject:
  1. download raw .nii.gz to tmp
  2. nibabel load
  3. spatial resample to 2mm iso (skip if already 2mm)
  4. z-norm on foreground
  5. fill background with foreground min
  6. symmetric int8 quantization (NeuroSTORM convention):
       scale = abs_max / 127
       frames = (data / scale).round().clip(-127, 127).astype(int8)
       permute to [T, H, W, D]
     save data.pt = {frames, scale, num_frames, affine}
  7. for each of 17 atlases:
       resample atlas to BOLD grid (cached per dataset)
       extract ROI time series (T, R) float32 -> save .npy
       compute Pearson FC (R, R) -> save .npy
  8. delete raw

Important: ROI/FC are computed BEFORE quantization, on the resampled float32
volume. The int8 frames are only for the foundation-model 4D path.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from atlas_registry import AtlasSpec, get_registry  # noqa: E402
from extractors import AtlasMasker, fc_from_roi  # noqa: E402


@dataclass
class WorkerConfig:
    out_root: Path
    target_voxel_size: tuple[float, float, float] = (2.0, 2.0, 2.0)
    quantize: bool = True
    save_q8: bool = True
    save_roi: bool = True
    save_fc: bool = True
    fill_zeroback: bool = False
    overwrite: bool = False
    raw_tmp: Optional[Path] = None  # default: out_root/raw_tmp

    def resolved_tmp(self) -> Path:
        return self.raw_tmp if self.raw_tmp is not None else (self.out_root / "raw_tmp")


@dataclass
class SubjectTask:
    subject_id: str
    raw_url_or_path: str  # http(s):// or local path
    dataset_tag: str  # e.g. "abide1", "abide2"


class _ResampleCache:
    """Holds 17 maskers fitted to the first subject's BOLD grid."""

    def __init__(self, atlases: list[AtlasSpec]):
        self.atlases = atlases
        self.maskers: dict[str, AtlasMasker] = {}
        self._fitted = False

    def fit_if_needed(self, ref_img: nib.Nifti1Image) -> None:
        if self._fitted:
            return
        for spec in self.atlases:
            m = AtlasMasker(spec).fit_to_reference(ref_img)
            self.maskers[spec.name] = m
        self._fitted = True


def _download(url: str, dst: Path, retries: int = 5,
              backoff: float = 2.0) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 256)
            tmp.rename(dst)
            return dst
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"download failed after {retries} retries: {url} ({last})")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _spatial_resample_to_iso(
    data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header,
    target_vox: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear resample (X,Y,Z,T) to target voxel size.

    Returns (resampled_data, new_affine). Skips if already at target (within 1%).
    """
    cur_vox = header.get_zooms()[:3]
    if all(abs(c - t) / t < 0.01 for c, t in zip(cur_vox, target_vox)):
        return data, affine

    import torch.nn.functional as F  # local import; torch is heavy
    scale = [c / t for c, t in zip(cur_vox, target_vox)]
    new_dims = [int(round(d * s)) for d, s in zip(data.shape[:3], scale)]

    if data.ndim == 4:
        x = torch.from_numpy(data.astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(1)
    else:
        x = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    y = F.interpolate(x, size=new_dims, mode="trilinear", align_corners=False)
    if data.ndim == 4:
        out = y.squeeze(1).permute(1, 2, 3, 0).numpy()
    else:
        out = y.squeeze(0).squeeze(0).numpy()

    new_affine = affine.copy()
    for i in range(3):
        new_affine[:, i] *= cur_vox[i] / target_vox[i]
    return out, new_affine


def _quantize_int8_neurostorm(
    data: np.ndarray, fill_zeroback: bool = False
) -> tuple[np.ndarray, float]:
    """NeuroSTORM-style symmetric int8 quantization.

    Replicates upstream preprocessing_volume._process_task_cpu:
      - foreground from data == 0
      - data[bg] = 0; data[data<0] = 0
      - z-norm on foreground
      - fill background with foreground-min (or 0 if fill_zeroback)
      - scale = abs_max / 127; clamp to [-127, 127]
    Returns (int8 array shape [T,H,W,D], scale).
    """
    d = torch.from_numpy(data.astype(np.float32))
    background = d == 0
    d[d < 0] = 0
    fg = d[~background]
    mean = fg.mean()
    std = fg.std()
    if std == 0:
        std = torch.tensor(1.0)
    d_norm = (d - mean) / std
    fg_norm = d_norm[~background]
    fill = (fg_norm.min() if not fill_zeroback else torch.tensor(0.0))
    out = torch.empty_like(d)
    out[background] = fill
    out[~background] = fg_norm

    abs_max = float(out.abs().max().item())
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    q = (out / scale).round().clamp_(-127, 127).to(torch.int8)
    if q.ndim == 4:
        q = q.permute(3, 0, 1, 2).contiguous()  # [T, H, W, D]
    return q.numpy(), scale


def process_subject(
    task: SubjectTask, cfg: WorkerConfig, masker_cache: _ResampleCache,
) -> dict:
    sid = task.subject_id
    out_root = cfg.out_root
    out_q8 = out_root / "q8" / sid
    out_roi = out_root / "roi"
    out_fc = out_root / "fc"
    out_meta = out_root / "meta" / f"{sid}.json"

    if not cfg.overwrite and out_meta.exists():
        return {"subject_id": sid, "status": "skip"}

    out_q8.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if _is_url(task.raw_url_or_path):
        tmp_dir = cfg.resolved_tmp()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / f"{sid}.nii.gz"
        _download(task.raw_url_or_path, tmp_file)
        local_path = tmp_file
        delete_after = True
    else:
        local_path = Path(task.raw_url_or_path)
        delete_after = False

    img = nib.load(str(local_path))
    data = img.get_fdata().astype(np.float32)
    if data.ndim != 4:
        if delete_after:
            local_path.unlink(missing_ok=True)
        return {"subject_id": sid, "status": "fail",
                "reason": f"not 4D: shape={data.shape}"}

    affine = img.affine
    header = img.header

    data, new_affine = _spatial_resample_to_iso(
        data, affine, header, cfg.target_voxel_size
    )
    new_img = nib.Nifti1Image(data, new_affine)
    masker_cache.fit_if_needed(new_img)

    # ROI extraction (per atlas) on resampled float32
    roi_shapes = {}
    fc_shapes = {}
    for spec in masker_cache.atlases:
        m = masker_cache.maskers[spec.name]
        try:
            ts = m.extract(new_img)  # (T, R) float32
        except Exception as e:
            roi_shapes[spec.name] = f"fail:{type(e).__name__}"
            continue
        if cfg.save_roi:
            d = out_roi / spec.name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{sid}.npy", ts)
        roi_shapes[spec.name] = list(ts.shape)
        if cfg.save_fc:
            fc = fc_from_roi(ts)
            d = out_fc / spec.name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{sid}.npy", fc)
            fc_shapes[spec.name] = list(fc.shape)

    # Quantization 4D
    quant_info = {}
    if cfg.save_q8:
        q, scale = _quantize_int8_neurostorm(data, cfg.fill_zeroback)
        torch.save({
            "frames": torch.from_numpy(q),
            "scale": float(scale),
            "num_frames": int(q.shape[0]),
            "affine": new_affine.astype(np.float32),
            "voxel_size": list(cfg.target_voxel_size),
        }, str(out_q8 / "data.pt"))
        quant_info = {
            "scale": float(scale), "shape_thwd": list(q.shape),
            "dtype": "int8",
        }

    if delete_after:
        local_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    meta = {
        "subject_id": sid, "dataset_tag": task.dataset_tag,
        "status": "ok",
        "elapsed_s": round(elapsed, 2),
        "shape_resampled": list(data.shape),
        "voxel_size": list(cfg.target_voxel_size),
        "quant": quant_info,
        "roi_shapes": roi_shapes,
        "fc_shapes": fc_shapes,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def make_masker_cache() -> _ResampleCache:
    return _ResampleCache(get_registry())


if __name__ == "__main__":
    # Tiny smoke test on one subject
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-id", required=True)
    ap.add_argument("--src", required=True, help="local path or http(s) URL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-tag", default="abide1")
    ap.add_argument("--no-q8", action="store_true")
    args = ap.parse_args()

    cfg = WorkerConfig(out_root=Path(args.out), save_q8=not args.no_q8)
    cache = make_masker_cache()
    task = SubjectTask(args.subject_id, args.src, args.dataset_tag)
    print(json.dumps(process_subject(task, cfg, cache), indent=2))
