"""Cox, survival forest, DeepSurv, and XGBoost-survival estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class CoxPHRegressor(BaseEstimator):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray):
        try:
            from statsmodels.duration.hazard_regression import PHReg
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for Cox PH") from exc
        X = self.scaler.fit_transform(self.imputer.fit_transform(X))
        self.model_ = PHReg(duration, X, status=np.asarray(event, dtype=int))
        self.result_ = self.model_.fit(disp=0)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(self.imputer.transform(X))
        return X @ np.asarray(self.result_.params)


class DeepSurvEstimator(BaseEstimator):
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 123,
        device: str = "cpu",
    ):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    @staticmethod
    def _cox_loss(
        risk,
        duration,
        event,
    ):
        import torch

        order = torch.argsort(duration, descending=True)
        ordered_risk = risk[order]
        ordered_event = event[order]
        log_cumulative_hazard = torch.logcumsumexp(ordered_risk, dim=0)
        contributions = ordered_risk - log_cumulative_hazard
        return -(contributions * ordered_event).sum() / ordered_event.sum().clamp_min(1.0)

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        device = torch.device(self.device)
        X = self.scaler.fit_transform(self.imputer.fit_transform(X)).astype(np.float32)
        layers: list[nn.Module] = []
        in_dim = X.shape[1]
        for hidden in self.hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.network_ = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        x_tensor = torch.from_numpy(X).to(device)
        duration_tensor = torch.as_tensor(duration, dtype=torch.float32, device=device)
        event_tensor = torch.as_tensor(event, dtype=torch.float32, device=device)
        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            risk = self.network_(x_tensor).squeeze(-1)
            loss = self._cox_loss(risk, duration_tensor, event_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = self.scaler.transform(self.imputer.transform(X)).astype(np.float32)
        self.network_.eval()
        with torch.no_grad():
            device = next(self.network_.parameters()).device
            return (
                self.network_(torch.from_numpy(X).to(device))
                .squeeze(-1)
                .cpu()
                .numpy()
            )


class SkSurvWrapper(BaseEstimator):
    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray):
        X = self.imputer.fit_transform(X)
        outcome = np.zeros(
            len(duration), dtype=[("event", "?"), ("duration", "<f8")]
        )
        outcome["event"] = np.asarray(event, dtype=bool)
        outcome["duration"] = np.asarray(duration, dtype=float)
        self.estimator.fit(X, outcome)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.imputer.transform(X)))


class XGBoostCox(BaseEstimator):
    def __init__(self, seed: int = 123, **params: Any):
        self.seed = seed
        self.params = params
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is required for XGBoost survival") from exc
        defaults = {
            "objective": "survival:cox",
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.seed,
            "n_jobs": 1,
        }
        defaults.update(self.params)
        self.model_ = XGBRegressor(**defaults)
        signed_duration = np.asarray(duration, dtype=float).copy()
        signed_duration[~np.asarray(event, dtype=bool)] *= -1
        self.model_.fit(self.imputer.fit_transform(X), signed_duration)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.imputer.transform(X))


def make_survival_estimator(model: str, seed: int = 123, **params: Any):
    name = model.lower().replace("-", "_")
    if name in {"cox", "coxph", "cox_ph"}:
        return CoxPHRegressor()
    if name in {"deepsurv", "deep_surv"}:
        return DeepSurvEstimator(seed=seed, **params)
    if name in {"random_survival_forest", "rsf"}:
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError as exc:
            raise RuntimeError(
                "scikit-survival is required for Random Survival Forest"
            ) from exc
        estimator = RandomSurvivalForest(
            n_estimators=int(params.get("n_estimators", 300)),
            min_samples_split=int(params.get("min_samples_split", 10)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=1,
            random_state=seed,
        )
        return SkSurvWrapper(estimator)
    if name in {"xgboost_survival", "xgb_survival", "xgb_cox"}:
        return XGBoostCox(seed=seed, **params)
    raise ValueError(f"Unknown survival model: {model}")
