"""Load real ADHD-200 connectomes from data/braingnn_input.

ADHD-200 is the closest in-repo proxy to ABIDE for a multi-site harmonization
pilot: 695 subjects across 8 acquisition sites with ADHD vs control labels.
ABIDE features are not extracted in this repo yet; the harmonization plumbing
is data-source agnostic, so we validate it on ADHD-200 first and swap in ABIDE
later when fc_matrix extraction completes.

The aal_116 atlas is preferred over cc200 because every ADHD-200 subject has
exactly 116 ROIs (cc200 has a length-mismatch outlier). Site information lives
in the official ADHD-200 phenotype CSV (`data/labels/adhd200_phenotype_with_site.csv`),
which encodes 8 numeric site IDs.

Diagnosis collapsing: the official label set has 4 levels (0=control,
1=ADHD-Combined, 2=ADHD-Hyper, 3=ADHD-Inattentive). For binary harmonization
benchmarking we collapse all ADHD subtypes to 1 and drop "pending" rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
DEFAULT_FEATURE_DIR = REPO / "data" / "braingnn_input" / "aal_116"
DEFAULT_PHENO = REPO / "data" / "labels" / "adhd200_phenotype_with_site.csv"

SITE_ID_TO_NAME = {
    1: "Peking",
    2: "Bradley",
    3: "KKI",
    4: "NeuroIMAGE",
    5: "NYU",
    6: "OHSU",
    7: "Pittsburgh",
    8: "WashU",
}


def load_adhd200_connectomes(
    atlas_dir: Path | str = DEFAULT_FEATURE_DIR,
    phenotype_csv: Path | str = DEFAULT_PHENO,
    min_site_size: int = 30,
    drop_pending_dx: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (features, meta) for the ADHD-200 cohort.

    Parameters
    ----------
    atlas_dir : ROI-extracted .pt files; expects sub-adhd200_<id>.pt entries.
    phenotype_csv : ADHD-200 phenotype with `subject_id, Site, Gender, Age, DX`.
    min_site_size : drop sites with fewer than this many usable subjects
        (e.g. Bradley has only 26 subjects and tends to be unstable for
        per-site harmonization).
    drop_pending_dx : drop rows whose DX is "pending".

    Returns
    -------
    features : (N, R, R) float32 connectome matrices, R=116.
    meta : DataFrame with columns
        subject_id, dataset, site, age, sex, dx
    """
    atlas_dir = Path(atlas_dir)
    pheno = pd.read_csv(phenotype_csv)

    pheno["subject_id"] = pheno["subject_id"].astype(str).str.zfill(7)
    if drop_pending_dx:
        pheno = pheno[pheno["DX"].astype(str).str.lower() != "pending"].copy()
    pheno["DX"] = pheno["DX"].astype(int)
    pheno["dx_binary"] = (pheno["DX"] > 0).astype(int)
    pheno["site"] = pheno["Site"].map(SITE_ID_TO_NAME).fillna("UnknownSite")

    rows: list[dict] = []
    fcs: list[np.ndarray] = []
    for _, r in pheno.iterrows():
        sid = r["subject_id"]
        f = atlas_dir / f"sub-adhd200_{sid}.pt"
        if not f.exists():
            continue
        try:
            d = torch.load(f, weights_only=False)
        except Exception:
            continue
        fc = np.asarray(d["fc_matrix"], dtype=np.float32)
        if fc.shape[0] != fc.shape[1]:
            continue
        np.fill_diagonal(fc, 0.0)
        fcs.append(fc)
        gender_raw = r.get("Gender")
        if pd.isna(gender_raw):
            sex = "U"
        else:
            sex = "M" if int(gender_raw) == 1 else "F"
        rows.append({
            "subject_id": f"adhd200-{sid}",
            "dataset": "ADHD-200",
            "site": r["site"],
            "age": float(r.get("Age", np.nan)),
            "sex": sex,
            "dx": int(r["dx_binary"]),
        })

    if not fcs:
        raise FileNotFoundError(
            f"No ADHD-200 connectomes found in {atlas_dir}; "
            "extract aal_116 features first via brain_gnn pipeline."
        )
        raise FileNotFoundError(
            f"No ADHD-200 connectomes found in {atlas_dir}; "
            "extract aal_116 features first via brain_gnn pipeline."
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
