"""Site-as-covariate adapter (no external deps; sanity-check baseline).

Solves: feature ~ intercept + batch + protected. Subtract only the batch
contribution at transform time, keeping the protected covariates intact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import HarmonizerBase


def _build_design(
    meta: pd.DataFrame,
    batch: str,
    protected: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    parts = [pd.Series(1.0, index=meta.index, name="intercept").to_frame()]
    batch_d = pd.get_dummies(
        meta[batch].astype("category"), prefix=batch, drop_first=True
    ).astype(np.float64)
    parts.append(batch_d)

    for p in protected:
        if p not in meta.columns:
            continue
        s = meta[p]
        if not pd.api.types.is_numeric_dtype(s):
            d = pd.get_dummies(s.astype("category"), prefix=p, drop_first=True).astype(
                np.float64
            )
            parts.append(d)
        else:
            parts.append(s.astype(np.float64).rename(p).to_frame())

    D = pd.concat(parts, axis=1)
    return D, batch_d.columns.tolist()


@dataclass
class SiteCovarHarmonizer(HarmonizerBase):
    def fit(self, features: np.ndarray, meta: pd.DataFrame) -> "SiteCovarHarmonizer":
        if features.ndim == 3:
            n, r, _ = features.shape
            iu = np.triu_indices(r, k=1)
            X = features[:, iu[0], iu[1]].astype(np.float64)
            self.fitted_state["feature_kind"] = "connectome"
            self.fitted_state["roi_dim"] = r
        else:
            X = features.astype(np.float64)
            self.fitted_state["feature_kind"] = "roi"

        D, batch_cols = _build_design(meta, self.batch, self.protected)
        beta, *_ = np.linalg.lstsq(D.to_numpy(), X, rcond=None)

        self.fitted_state["fit_columns"] = D.columns.tolist()
        self.fitted_state["batch_cols"] = batch_cols
        self.fitted_state["beta"] = beta
        return self

    def transform(self, features: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted():
            raise RuntimeError("call fit() before transform()")

        kind = self.fitted_state["feature_kind"]
        if kind == "connectome":
            r = self.fitted_state["roi_dim"]
            iu = np.triu_indices(r, k=1)
            X = features[:, iu[0], iu[1]].astype(np.float64)
        else:
            X = features.astype(np.float64)

        D, _ = _build_design(meta, self.batch, self.protected)
        D = D.reindex(columns=self.fitted_state["fit_columns"], fill_value=0.0)

        beta = self.fitted_state["beta"]
        batch_cols = self.fitted_state["batch_cols"]
        batch_idx = [self.fitted_state["fit_columns"].index(c) for c in batch_cols]
        batch_effect = D.iloc[:, batch_idx].to_numpy() @ beta[batch_idx, :]
        X_h = X - batch_effect

        if kind == "connectome":
            r = self.fitted_state["roi_dim"]
            iu = np.triu_indices(r, k=1)
            out = np.zeros((X_h.shape[0], r, r), dtype=np.float64)
            out[:, iu[0], iu[1]] = X_h
            out = out + np.transpose(out, (0, 2, 1))
            return out
        return X_h
