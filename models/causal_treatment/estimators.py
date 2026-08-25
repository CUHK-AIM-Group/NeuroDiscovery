"""Meta-learners, doubly robust learner, causal forest, TARNet, DragonNet."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _outcome_model(seed: int):
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def _propensity_model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )


class BaseCATELearner:
    def _prepare(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.imputer_ = SimpleImputer(strategy="median")
            return self.imputer_.fit_transform(X)
        return self.imputer_.transform(X)

    def predict_policy(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return (self.predict_cate(X) > threshold).astype(int)


class IPWLearner(BaseCATELearner):
    """Propensity-weighted ATE represented as a constant CATE."""

    def __init__(self, seed: int = 123):
        self.seed = seed

    def fit(self, X, treatment, outcome):
        X = self._prepare(X, fit=True)
        t = np.asarray(treatment, dtype=float)
        y = np.asarray(outcome, dtype=float)
        self.propensity_ = _propensity_model(self.seed).fit(X, t)
        p = np.clip(self.propensity_.predict_proba(X)[:, 1], 0.02, 0.98)
        self.ate_ = float(np.mean(t * y / p - (1 - t) * y / (1 - p)))
        return self

    def predict_cate(self, X):
        return np.full(len(self._prepare(X)), self.ate_)


class SLearner(BaseCATELearner):
    def __init__(self, seed: int = 123):
        self.seed = seed

    def fit(self, X, treatment, outcome):
        X = self._prepare(X, fit=True)
        self.model_ = _outcome_model(self.seed)
        self.model_.fit(np.column_stack([X, treatment]), outcome)
        return self

    def predict_cate(self, X):
        X = self._prepare(X)
        return self.model_.predict(np.column_stack([X, np.ones(len(X))])) - self.model_.predict(
            np.column_stack([X, np.zeros(len(X))])
        )


class TLearner(BaseCATELearner):
    def __init__(self, seed: int = 123):
        self.seed = seed

    def fit(self, X, treatment, outcome):
        X = self._prepare(X, fit=True)
        treatment = np.asarray(treatment, dtype=int)
        self.control_ = _outcome_model(self.seed)
        self.treated_ = _outcome_model(self.seed + 1)
        self.control_.fit(X[treatment == 0], np.asarray(outcome)[treatment == 0])
        self.treated_.fit(X[treatment == 1], np.asarray(outcome)[treatment == 1])
        return self

    def potential_outcomes(self, X):
        X = self._prepare(X)
        return self.control_.predict(X), self.treated_.predict(X)

    def predict_cate(self, X):
        y0, y1 = self.potential_outcomes(X)
        return y1 - y0


class XLearner(TLearner):
    def fit(self, X, treatment, outcome):
        super().fit(X, treatment, outcome)
        X_clean = self._prepare(X)
        treatment = np.asarray(treatment, dtype=int)
        outcome = np.asarray(outcome, dtype=float)
        d_treated = outcome[treatment == 1] - self.control_.predict(X_clean[treatment == 1])
        d_control = self.treated_.predict(X_clean[treatment == 0]) - outcome[treatment == 0]
        self.tau_treated_ = _outcome_model(self.seed + 2).fit(
            X_clean[treatment == 1], d_treated
        )
        self.tau_control_ = _outcome_model(self.seed + 3).fit(
            X_clean[treatment == 0], d_control
        )
        self.propensity_ = _propensity_model(self.seed).fit(X_clean, treatment)
        return self

    def predict_cate(self, X):
        X_clean = self._prepare(X)
        propensity = self.propensity_.predict_proba(X_clean)[:, 1]
        tau_control = self.tau_control_.predict(X_clean)
        tau_treated = self.tau_treated_.predict(X_clean)
        return propensity * tau_control + (1 - propensity) * tau_treated


class DoublyRobustLearner(TLearner):
    def fit(self, X, treatment, outcome):
        super().fit(X, treatment, outcome)
        X_clean = self._prepare(X)
        treatment = np.asarray(treatment, dtype=float)
        outcome = np.asarray(outcome, dtype=float)
        self.propensity_ = _propensity_model(self.seed).fit(X_clean, treatment)
        p = np.clip(self.propensity_.predict_proba(X_clean)[:, 1], 0.02, 0.98)
        mu0, mu1 = self.control_.predict(X_clean), self.treated_.predict(X_clean)
        pseudo = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / p
            - (1 - treatment) * (outcome - mu0) / (1 - p)
        )
        self.effect_model_ = _outcome_model(self.seed + 4).fit(X_clean, pseudo)
        return self

    def predict_cate(self, X):
        return self.effect_model_.predict(self._prepare(X))


class PolicyLearner(DoublyRobustLearner):
    """Interpretable plug-in policy learned from doubly robust effects."""

    def fit(self, X, treatment, outcome):
        from sklearn.tree import DecisionTreeClassifier

        super().fit(X, treatment, outcome)
        X_clean = self._prepare(X)
        effect = self.effect_model_.predict(X_clean)
        self.policy_model_ = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=max(5, len(X_clean) // 50),
            random_state=self.seed,
        ).fit(X_clean, effect > 0, sample_weight=np.abs(effect) + 1e-6)
        return self

    def predict_policy(self, X, threshold: float = 0.0):
        return self.policy_model_.predict(self._prepare(X)).astype(int)


class CausalForestLearner(BaseCATELearner):
    def __init__(self, seed: int = 123, **params: Any):
        self.seed = seed
        self.params = params

    def fit(self, X, treatment, outcome):
        try:
            from econml.dml import CausalForestDML
        except ImportError as exc:
            raise RuntimeError("econml is required for CausalForestDML") from exc
        X = self._prepare(X, fit=True)
        self.model_ = CausalForestDML(
            n_estimators=int(self.params.get("n_estimators", 500)),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 5)),
            max_depth=self.params.get("max_depth"),
            discrete_treatment=True,
            cv=int(self.params.get("cv", 3)),
            random_state=self.seed,
            n_jobs=1,
        )
        self.model_.fit(np.asarray(outcome), np.asarray(treatment), X=X)
        return self

    def predict_cate(self, X):
        return np.asarray(self.model_.effect(self._prepare(X)))


class TreatmentNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dragon: bool):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.y0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1)
        )
        self.y1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1)
        )
        self.propensity = nn.Linear(hidden_dim, 1) if dragon else None

    def forward(self, values):
        representation = self.trunk(values)
        y0 = self.y0(representation).squeeze(-1)
        y1 = self.y1(representation).squeeze(-1)
        propensity = (
            self.propensity(representation).squeeze(-1)
            if self.propensity is not None
            else None
        )
        return y0, y1, propensity


class NeuralTreatmentLearner(BaseCATELearner):
    def __init__(
        self,
        architecture: str,
        seed: int = 123,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.architecture = architecture
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def fit(self, X, treatment, outcome):
        torch.manual_seed(self.seed)
        X = self._prepare(X, fit=True)
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X).astype(np.float32)
        input_dim = X.shape[1]

        device = torch.device(self.device)
        self.network_ = TreatmentNetwork(
            input_dim, self.hidden_dim, self.architecture == "dragonnet"
        ).to(device)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.lr, weight_decay=1e-4)
        x_tensor = torch.from_numpy(X).to(device)
        t_tensor = torch.as_tensor(treatment, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(outcome, dtype=torch.float32, device=device)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            y0, y1, propensity_logit = self.network_(x_tensor)
            factual = t_tensor * y1 + (1 - t_tensor) * y0
            loss = nn.functional.mse_loss(factual, y_tensor)
            if propensity_logit is not None:
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    propensity_logit, t_tensor
                )
            loss.backward()
            optimizer.step()
        return self

    def predict_cate(self, X):
        X = self.scaler_.transform(self._prepare(X)).astype(np.float32)
        self.network_.eval()
        with torch.no_grad():
            device = next(self.network_.parameters()).device
            y0, y1, _ = self.network_(torch.from_numpy(X).to(device))
        return (y1 - y0).cpu().numpy()


def make_cate_estimator(model: str, seed: int = 123, **params: Any):
    name = model.lower().replace("-", "_")
    if name in {"ipw", "propensity_weighting"}:
        return IPWLearner(seed)
    if name in {"s_learner", "slearner"}:
        return SLearner(seed)
    if name in {"t_learner", "tlearner"}:
        return TLearner(seed)
    if name in {"x_learner", "xlearner"}:
        return XLearner(seed)
    if name in {"doubly_robust", "dr_learner", "drlearner"}:
        return DoublyRobustLearner(seed)
    if name in {"policy_learner", "policy"}:
        return PolicyLearner(seed)
    if name in {"causal_forest", "causalforestdml"}:
        return CausalForestLearner(seed, **params)
    if name in {"tarnet", "dragonnet"}:
        return NeuralTreatmentLearner(name, seed=seed, **params)
    raise ValueError(f"Unknown treatment-effect model: {model}")
