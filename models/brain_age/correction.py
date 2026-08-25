"""Brain-age bias correction fitted only on training controls."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class BrainAgeBiasCorrector:
    """Correct regression-to-the-mean bias in predicted brain age."""

    def fit(self, chronological_age: np.ndarray, predicted_age: np.ndarray):
        age = np.asarray(chronological_age, dtype=float).reshape(-1, 1)
        pred = np.asarray(predicted_age, dtype=float)
        self.model_ = LinearRegression().fit(age, pred - age.ravel())
        return self

    def transform(
        self,
        chronological_age: np.ndarray,
        predicted_age: np.ndarray,
    ) -> np.ndarray:
        age = np.asarray(chronological_age, dtype=float)
        pred = np.asarray(predicted_age, dtype=float)
        expected_bias = self.model_.predict(age.reshape(-1, 1))
        return pred - expected_bias

    def brain_pad(
        self,
        chronological_age: np.ndarray,
        predicted_age: np.ndarray,
    ) -> np.ndarray:
        corrected = self.transform(chronological_age, predicted_age)
        return corrected - np.asarray(chronological_age, dtype=float)
