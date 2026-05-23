"""Diagnostics: quantify how much variance is explained by the batch variable.

Per-feature R^2 of `site` (or whatever batch column is chosen) regressed
against the feature value. Run before AND after harmonization — the drop
is the headline metric for whether harmonization actually worked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SiteEffectReport:
    per_feature_r2: np.ndarray
    mean_r2: float
    median_r2: float
    p95_r2: float
    n_features: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"r2_site": self.per_feature_r2})

    def summary(self) -> dict:
        return {
            "mean_r2": float(self.mean_r2),
            "median_r2": float(self.median_r2),
            "p95_r2": float(self.p95_r2),
            "n_features": int(self.n_features),
        }


def _flatten_features(features: np.ndarray) -> np.ndarray:
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        n, r, _ = features.shape
        iu = np.triu_indices(r, k=1)
        return features[:, iu[0], iu[1]]
    raise ValueError(f"unsupported feature shape {features.shape}")


def _one_hot(labels: pd.Series) -> np.ndarray:
    return pd.get_dummies(labels.astype("category"), drop_first=True).to_numpy(
        dtype=np.float64
    )


def site_effect_r2(
    features: np.ndarray,
    meta: pd.DataFrame,
    batch: str = "site",
) -> SiteEffectReport:
    """Per-feature R^2 of feature ~ batch via one-hot OLS.

    Mathematically this is 1 - SS_res / SS_tot when the only predictor is
    the batch indicator. High R^2 = feature is dominated by batch.
    """
    X = _flatten_features(features).astype(np.float64)
    if X.shape[0] != len(meta):
        raise ValueError("features and meta length mismatch")

    B = _one_hot(meta[batch])
    if B.shape[1] == 0:
        raise ValueError(f"batch '{batch}' has no contrast (singleton)")

    # Center features and design once
    X_c = X - X.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)

    # Solve B_c @ beta ~ X_c column-wise; broadcast across all features at once
    beta, *_ = np.linalg.lstsq(B_c, X_c, rcond=None)
    X_hat = B_c @ beta

    ss_tot = (X_c ** 2).sum(axis=0)
    ss_res = ((X_c - X_hat) ** 2).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    r2 = np.clip(r2, 0.0, 1.0)

    return SiteEffectReport(
        per_feature_r2=r2,
        mean_r2=float(r2.mean()),
        median_r2=float(np.median(r2)),
        p95_r2=float(np.quantile(r2, 0.95)),
        n_features=int(r2.size),
    )


def compare_reports(before: SiteEffectReport, after: SiteEffectReport) -> dict:
    return {
        "before": before.summary(),
        "after": after.summary(),
        "delta_mean_r2": before.mean_r2 - after.mean_r2,
        "delta_median_r2": before.median_r2 - after.median_r2,
        "delta_p95_r2": before.p95_r2 - after.p95_r2,
    }
