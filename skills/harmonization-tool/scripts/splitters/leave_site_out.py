"""Leave-site-out (LOSO) splitter.

Yields (train_idx, val_idx, test_idx) triples where one site at a time is
held out as the test set. The val set is carved from the remaining sites
in a site-stratified way so it stays representative of the train sites.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def leave_site_out_splits(
    meta: pd.DataFrame,
    site_col: str = "site",
    val_frac: float = 0.1,
    seed: int = 42,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Iterate (held_out_site, train_idx, val_idx, test_idx).

    val_idx is sampled from the non-held-out sites, stratified by site,
    so that train and val come from the same distribution.
    """
    rng = np.random.default_rng(seed)
    sites = meta[site_col].astype(str)
    unique_sites = sorted(sites.unique())

    for held in unique_sites:
        test_idx = np.where(sites == held)[0]
        remaining = np.where(sites != held)[0]

        val_idx_parts: list[np.ndarray] = []
        for s in unique_sites:
            if s == held:
                continue
            site_idx = np.where(sites == s)[0]
            if site_idx.size == 0:
                continue
            n_val = max(1, int(round(val_frac * site_idx.size)))
            n_val = min(n_val, site_idx.size - 1)  # leave at least 1 for train
            chosen = rng.choice(site_idx, size=n_val, replace=False)
            val_idx_parts.append(chosen)

        val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)
        train_mask = np.ones(len(meta), dtype=bool)
        train_mask[test_idx] = False
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        yield held, train_idx, val_idx, test_idx


def first_split(
    meta: pd.DataFrame, site_col: str = "site", val_frac: float = 0.1, seed: int = 42
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: return only the first LOSO fold."""
    return next(leave_site_out_splits(meta, site_col, val_frac, seed))
