"""CovBat wrapper: harmonizes mean, variance, AND covariance.

Use for connectome / FC features where second-order structure matters.
Falls back to a clean error when the optional dep is missing.

Reference: Chen et al., 2022, *Mitigating site effects in covariance for
machine learning in neuroimaging data*, NeuroImage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import HarmonizerBase
from .neuroharmonize_wrapper import _to_2d, _from_2d


@dataclass
class CovBatAdapter(HarmonizerBase):
    """Wraps CovBat (Python port).

    Recommended for connectome inputs. For ROI scalars, prefer ComBat-GAM.
    """

    n_pc: int | None = None  # number of PCs to harmonize; None = auto

    def fit(self, features: np.ndarray, meta: pd.DataFrame) -> "CovBatAdapter":
        try:
            from covbat import covbat  # type: ignore
        except ImportError as e:
            raise ImportError(
                "CovBat is required. Install from "
                "https://github.com/andy1764/CovBat_Harmonization "
                "or pip install covbat (if a published mirror is available)."
            ) from e

        X, info = _to_2d(features)

        covars = pd.DataFrame({self.batch: meta[self.batch].astype(str).to_numpy()})
        for p in self.protected:
            if p not in meta.columns:
                continue
            s = meta[p]
            if not pd.api.types.is_numeric_dtype(s):
                covars[p] = s.astype(str).to_numpy()
            else:
                covars[p] = s.astype(np.float64).to_numpy()

        result = covbat(
            data=X.T,
            batch=covars[self.batch].to_numpy(),
            mod=covars.drop(columns=[self.batch]) if covars.shape[1] > 1 else None,
            n_pc=self.n_pc,
        )
        # covbat returns (F, N); transpose back
        X_h = np.asarray(result).T

        self.fitted_state["info"] = info
        self.fitted_state["train_features_harmonized"] = _from_2d(X_h, info)
        return self

    def transform(self, features: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted():
            raise RuntimeError("call fit() before transform()")
        raise NotImplementedError(
            "CovBatAdapter does not support split fit/transform yet. "
            "Use full-cohort fit_transform, or fall back to ComBat-GAM "
            "via NeuroHarmonizeAdapter for train/val/test workflows."
        )

    def fit_transform(
        self, features: np.ndarray, meta: pd.DataFrame
    ) -> np.ndarray:
        self.fit(features, meta)
        return self.fitted_state.pop("train_features_harmonized")

    def method_name(self) -> str:
        return "covbat"
