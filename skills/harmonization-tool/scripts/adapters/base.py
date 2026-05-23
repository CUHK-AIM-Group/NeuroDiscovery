"""Adapter base class: enforces fit-on-train, transform-on-val/test.

Fitting harmonization on the full dataset before splitting leaks site-level
information into the test set. All adapters subclass HarmonizerBase and
expose explicit fit / transform / fit_transform entry points.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HarmonizerBase(ABC):
    batch: str = "site"
    protected: tuple[str, ...] = ("age", "sex", "dx")
    fitted_state: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def fit(self, features: np.ndarray, meta: pd.DataFrame) -> "HarmonizerBase":
        ...

    @abstractmethod
    def transform(self, features: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        ...

    def fit_transform(
        self, features: np.ndarray, meta: pd.DataFrame
    ) -> np.ndarray:
        self.fit(features, meta)
        return self.transform(features, meta)

    def is_fitted(self) -> bool:
        return bool(self.fitted_state)

    def method_name(self) -> str:
        return self.__class__.__name__
