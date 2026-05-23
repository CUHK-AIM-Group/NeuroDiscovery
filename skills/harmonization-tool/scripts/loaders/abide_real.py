"""Load real ABIDE I connectomes from data/abide/ABIDE_pcp.

ABIDE I CPAC ROI time series live as `.1D` files (T, R) under
data/abide/ABIDE_pcp/cpac/filt_noglobal/<atlas>/<FILE_ID>_<atlas>.1D.
This loader builds a connectome (R, R) per subject via Pearson correlation
and joins phenotype rows on FILE_ID.

Atlases supported: rois_aal (R=116), rois_cc200 (R=200), rois_cc400 (R=392),
rois_dosenbach160 (R=161), rois_ez (R=116), rois_ho (R=111), rois_tt (R=97).
Most ABIDE papers use cc200; aal_116 is the closest match to the ADHD-200
extraction in this repo, useful for cross-cohort comparisons.

DX_GROUP encoding (from ABIDE phenotype): 1 = autism, 2 = control. We
remap to 1 = autism, 0 = control to keep dx as a {0,1} target consistent
with the rest of the harmonization pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO / "data" / "abide" / "ABIDE_pcp" / "cpac" / "filt_noglobal"
DEFAULT_PHENO = REPO / "data" / "abide" / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"


def _ts_to_fc(ts: np.ndarray) -> np.ndarray:
    if ts.shape[0] < ts.shape[1]:
        ts = ts.T
    fc = np.corrcoef(ts.T).astype(np.float32)
    np.fill_diagonal(fc, 0.0)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    return fc


def load_abide_connectomes(
    atlas: str = "rois_aal",
    roi_root: Path | str = DEFAULT_ROOT,
    phenotype_csv: Path | str = DEFAULT_PHENO,
    min_site_size: int = 30,
    quality_checked: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (features, meta) for ABIDE I.

    Parameters
    ----------
    atlas : one of rois_aal / rois_cc200 / rois_cc400 / rois_dosenbach160
        / rois_ez / rois_ho / rois_tt.
    min_site_size : drop sites with fewer subjects after QC + load.
    quality_checked : keep only subjects who passed every available QC rater.

    Returns
    -------
    features : (N, R, R) float32 connectome matrices.
    meta : DataFrame with subject_id, dataset, site, age, sex, dx.
    """
    atlas_dir = Path(roi_root) / atlas
    if not atlas_dir.is_dir():
        raise FileNotFoundError(
            f"{atlas_dir} not found; run fetch_abide_rois_robust.py first."
        )

    pheno = pd.read_csv(phenotype_csv)
    if quality_checked:
        for c in ("qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2",
                  "qc_anat_rater_3", "qc_func_rater_3"):
            if c in pheno.columns:
                pheno = pheno[
                    pheno[c].fillna("OK").astype(str).str.upper() != "FAIL"
                ]
    pheno = pheno[pheno["FILE_ID"].notna() & (pheno["FILE_ID"] != "no_filename")]
    pheno = pheno.copy()

    rows: list[dict] = []
    fcs: list[np.ndarray] = []
    for _, r in pheno.iterrows():
        fid = str(r["FILE_ID"])
        f = atlas_dir / f"{fid}_{atlas}.1D"
        if not f.exists():
            continue
        try:
            ts = np.loadtxt(f, dtype=np.float32)
        except Exception:
            continue
        if ts.size == 0 or ts.ndim != 2:
            continue
        fc = _ts_to_fc(ts)
        if fc.shape[0] != fc.shape[1]:
            continue
        fcs.append(fc)
        sex = "M" if int(r["SEX"]) == 1 else "F"
        dx_binary = 1 if int(r["DX_GROUP"]) == 1 else 0
        rows.append({
            "subject_id": f"abide-{int(r['SUB_ID'])}",
            "dataset": "ABIDE-I",
            "site": str(r["SITE_ID"]),
            "age": float(r.get("AGE_AT_SCAN", np.nan)),
            "sex": sex,
            "dx": dx_binary,
        })

    if not fcs:
        raise RuntimeError(
            f"No usable ABIDE connectomes built from {atlas_dir}."
        )

    shapes = {fc.shape for fc in fcs}
    if len(shapes) != 1:
        target = max(shapes, key=lambda s: sum(1 for fc in fcs if fc.shape == s))
        keep = [i for i, fc in enumerate(fcs) if fc.shape == target]
        fcs = [fcs[i] for i in keep]
        rows = [rows[i] for i in keep]

    features = np.stack(fcs, axis=0)
    meta = pd.DataFrame(rows)

    site_counts = meta["site"].value_counts()
    big_sites = site_counts[site_counts >= min_site_size].index.tolist()
    keep_mask = meta["site"].isin(big_sites).to_numpy()
    features = features[keep_mask]
    meta = meta[keep_mask].reset_index(drop=True)

    if "age" in meta:
        meta["age"] = meta["age"].fillna(meta["age"].median())

    return features, meta
