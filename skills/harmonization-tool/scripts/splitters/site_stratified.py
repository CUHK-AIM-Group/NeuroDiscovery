"""Site-stratified 80/10/10 splitter.

Compatible with the project default split (see feedback-cv-protocol.md):
80% train / 10% val / 10% test, but stratified per site so each site appears
in all three splits in proportion. This prevents accidental site holdout.

If a strat label (e.g. 'dx') is provided, additionally stratify within
each site by that label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def site_stratified_split(
    meta: pd.DataFrame,
    site_col: str = "site",
    label_col: str | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train+val+test fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    sites = meta[site_col].astype(str)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for s in sorted(sites.unique()):
        site_mask = (sites == s).to_numpy()
        site_idx = np.where(site_mask)[0]
        if site_idx.size < 3:
            # Tiny sites: assign all to train; the splitter cannot meaningfully
            # split fewer than 3 subjects across train/val/test.
            train_idx.extend(site_idx.tolist())
            continue

        if label_col is not None and label_col in meta.columns:
            labels = meta.iloc[site_idx][label_col].astype(str).to_numpy()
            for lab in np.unique(labels):
                sub = site_idx[labels == lab]
                _split_one(sub, rng, train_frac, val_frac, train_idx, val_idx, test_idx)
        else:
            _split_one(site_idx, rng, train_frac, val_frac, train_idx, val_idx, test_idx)

    return (
        np.array(sorted(train_idx), dtype=int),
        np.array(sorted(val_idx), dtype=int),
        np.array(sorted(test_idx), dtype=int),
    )


def _split_one(
    idx: np.ndarray,
    rng: np.random.Generator,
    train_frac: float,
    val_frac: float,
    train_out: list[int],
    val_out: list[int],
    test_out: list[int],
) -> None:
    n = idx.size
    perm = rng.permutation(idx)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n))) if n - n_train >= 2 else 0
    n_test = n - n_train - n_val
    if n_test < 1 and n - n_train >= 1:
        n_val = max(0, n_val - 1)
        n_test = n - n_train - n_val
    train_out.extend(perm[:n_train].tolist())
    val_out.extend(perm[n_train : n_train + n_val].tolist())
    test_out.extend(perm[n_train + n_val :].tolist())
