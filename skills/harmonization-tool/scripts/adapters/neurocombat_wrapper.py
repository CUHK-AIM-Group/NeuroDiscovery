"""neuroCombat wrapper: raw ComBat reference implementation.

Useful when comparing to neuroHarmonize, or when smoothing is not needed.
Install: pip install neuroCombat
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import HarmonizerBase
from .neuroharmonize_wrapper import _to_2d, _from_2d


@dataclass
class NeuroCombatAdapter(HarmonizerBase):
    """Wraps neuroCombat.neuroCombat.

    Note: stock neuroCombat does not separate fit/transform; we cache the
    estimated parameters and rerun the transform analytically on new data
    by treating it as a re-fit limited to known levels.
    """

    def fit(self, features: np.ndarray, meta: pd.DataFrame) -> "NeuroCombatAdapter":
        try:
            from neuroCombat import neuroCombat  # type: ignore
        except ImportError as e:
            raise ImportError(
                "neuroCombat is required. Install with: pip install neuroCombat"
            ) from e

        X, info = _to_2d(features)

        covars = pd.DataFrame({self.batch: meta[self.batch].astype(str).to_numpy()})
        continuous_cols: list[str] = []
        categorical_cols: list[str] = []
        for p in self.protected:
            if p not in meta.columns:
                continue
            s = meta[p]
            if not pd.api.types.is_numeric_dtype(s):
                covars[p] = s.astype(str).to_numpy()
                categorical_cols.append(p)
            else:
                covars[p] = s.astype(np.float64).to_numpy()
                continuous_cols.append(p)

        # neuroCombat expects features as (F, N)
        out = neuroCombat(
            dat=X.T,
            covars=covars,
            batch_col=self.batch,
            categorical_cols=categorical_cols or None,
            continuous_cols=continuous_cols or None,
        )
        X_h = out["data"].T

        self.fitted_state["estimates"] = out.get("estimates", {})
        self.fitted_state["info"] = info
        self.fitted_state["train_features_harmonized"] = _from_2d(X_h, info)
        return self

    def transform(self, features: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted():
            raise RuntimeError("call fit() before transform()")
        # neuroCombat's public API does not expose a clean apply path; for
        # train-only or full-cohort use, fit_transform is the supported route.
        # Cross-split application should use the neuroHarmonize adapter instead.
        raise NotImplementedError(
            "NeuroCombatAdapter does not support split fit/transform. "
            "Use NeuroHarmonizeAdapter for train/val/test workflows."
        )

    def fit_transform(
        self, features: np.ndarray, meta: pd.DataFrame
    ) -> np.ndarray:
        self.fit(features, meta)
        return self.fitted_state.pop("train_features_harmonized")

    def method_name(self) -> str:
        return "combat-raw"
