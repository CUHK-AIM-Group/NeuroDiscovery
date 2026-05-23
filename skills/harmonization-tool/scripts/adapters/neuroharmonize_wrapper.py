"""neuroHarmonize wrapper: ComBat and ComBat-GAM.

Lazy-imports neuroHarmonize so the rest of the skill works (and tests) when
the optional dep is not installed. Install via:

    pip install neuroHarmonize
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import HarmonizerBase


def _to_2d(features: np.ndarray) -> tuple[np.ndarray, dict]:
    if features.ndim == 3:
        n, r, _ = features.shape
        iu = np.triu_indices(r, k=1)
        return features[:, iu[0], iu[1]].astype(np.float64), {
            "kind": "connectome",
            "roi_dim": r,
        }
    return features.astype(np.float64), {"kind": "roi"}


def _from_2d(X: np.ndarray, info: dict) -> np.ndarray:
    if info["kind"] == "connectome":
        r = info["roi_dim"]
        iu = np.triu_indices(r, k=1)
        out = np.zeros((X.shape[0], r, r), dtype=np.float64)
        out[:, iu[0], iu[1]] = X
        out = out + np.transpose(out, (0, 2, 1))
        return out
    return X


def _build_covars(
    meta: pd.DataFrame,
    batch: str,
    protected: tuple[str, ...],
    smooth_terms: tuple[str, ...] = (),
) -> pd.DataFrame:
    """neuroHarmonize expects a DataFrame with 'SITE' (str) plus covariates."""
    cov = pd.DataFrame({"SITE": meta[batch].astype(str).to_numpy()})
    for p in protected:
        if p not in meta.columns:
            continue
        s = meta[p]
        if not pd.api.types.is_numeric_dtype(s):
            d = pd.get_dummies(s.astype("category"), prefix=p, drop_first=True).astype(
                np.float64
            )
            for c in d.columns:
                cov[c] = d[c].to_numpy()
        else:
            cov[p] = s.astype(np.float64).to_numpy()
    return cov


@dataclass
class NeuroHarmonizeAdapter(HarmonizerBase):
    """Wraps neuroHarmonize.harmonizationLearn / harmonizationApply.

    method = "combat" (linear) or "combat-gam" (smooth on `smooth_terms`).
    """

    method: str = "combat"
    smooth_terms: tuple[str, ...] = ("age",)

    def fit(self, features: np.ndarray, meta: pd.DataFrame) -> "NeuroHarmonizeAdapter":
        try:
            from neuroHarmonize import harmonizationLearn  # type: ignore
        except ImportError as e:
            raise ImportError(
                "neuroHarmonize is required. Install with: pip install neuroHarmonize"
            ) from e

        X, info = _to_2d(features)
        cov = _build_covars(meta, self.batch, self.protected, self.smooth_terms)

        kwargs: dict = {}
        if self.method == "combat-gam":
            present = [t for t in self.smooth_terms if t in cov.columns]
            if present:
                kwargs["smooth_terms"] = present

        model, X_h = harmonizationLearn(X, cov, **kwargs)

        self.fitted_state["model"] = model
        self.fitted_state["info"] = info
        self.fitted_state["smooth_terms"] = kwargs.get("smooth_terms", [])
        # Cache training transform so fit_transform can short-circuit
        self.fitted_state["train_features_harmonized"] = _from_2d(X_h, info)
        return self

    def transform(self, features: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted():
            raise RuntimeError("call fit() before transform()")

        try:
            from neuroHarmonize import harmonizationApply  # type: ignore
        except ImportError as e:
            raise ImportError(
                "neuroHarmonize is required. Install with: pip install neuroHarmonize"
            ) from e

        info = self.fitted_state["info"]
        X, _ = _to_2d(features)
        cov = _build_covars(meta, self.batch, self.protected, self.smooth_terms)

        X_h = harmonizationApply(X, cov, self.fitted_state["model"])
        return _from_2d(X_h, info)

    def fit_transform(
        self, features: np.ndarray, meta: pd.DataFrame
    ) -> np.ndarray:
        self.fit(features, meta)
        cached = self.fitted_state.pop("train_features_harmonized", None)
        if cached is not None:
            return cached
        return self.transform(features, meta)

    def method_name(self) -> str:
        return self.method
