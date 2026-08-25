"""Cross-fitted treatment-effect and policy evaluation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models.causal_treatment.estimators import make_cate_estimator
from models.causal_treatment.metrics import policy_value, standardized_mean_difference
from models.common.artifacts import RunArtifacts
from models.common.data import make_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--treatment-col", required=True)
    parser.add_argument("--outcome-col", required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.features)
    excluded = {args.subject_col, args.treatment_col, args.outcome_col}
    features = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    clean = frame.dropna(
        subset=[args.treatment_col, args.outcome_col, *features]
    ).reset_index(drop=True)
    X = clean[features].to_numpy(dtype=float)
    treatment = clean[args.treatment_col].to_numpy(dtype=int)
    outcome = clean[args.outcome_col].to_numpy(dtype=float)
    cate = np.empty(len(clean))
    propensity = np.empty(len(clean))
    policy = np.empty(len(clean), dtype=int)
    fold_ids = np.full(len(clean), -1)
    checkpoints = []
    for fold, (train, test) in enumerate(
        make_splits(treatment, "classification", args.folds, args.seed)
    ):
        params = (
            {"epochs": args.epochs, "device": args.device}
            if args.model.lower() in {"tarnet", "dragonnet"}
            else {}
        )
        learner = make_cate_estimator(args.model, args.seed + fold, **params)
        learner.fit(X[train], treatment[train], outcome[train])
        cate[test] = learner.predict_cate(X[test])
        policy[test] = learner.predict_policy(X[test])
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        propensity_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=args.seed + fold),
        ).fit(X[train], treatment[train])
        propensity[test] = propensity_model.predict_proba(X[test])[:, 1]
        fold_ids[test] = fold
        if hasattr(learner, "network_"):
            learner.network_.cpu()
        checkpoints.append((learner, propensity_model))
    metrics = {
        "ate": float(np.mean(cate)),
        "cate_sd": float(np.std(cate, ddof=1)),
        "policy_treatment_rate": float(np.mean(policy)),
        "policy_value_ipw": policy_value(outcome, treatment, propensity, policy),
        "max_abs_smd_unweighted": float(
            np.max(np.abs(standardized_mean_difference(X, treatment)))
        ),
    }
    weights = np.where(
        treatment == 1,
        1 / np.clip(propensity, 0.02, 0.98),
        1 / np.clip(1 - propensity, 0.02, 0.98),
    )
    metrics["max_abs_smd_ipw"] = float(
        np.max(np.abs(standardized_mean_difference(X, treatment, weights)))
    )
    artifacts = RunArtifacts(args.output_dir, vars(args))
    artifacts.write_config()
    artifacts.write_metrics(metrics)
    artifacts.write_predictions(
        {
            "subject_id": clean[args.subject_col].astype(str),
            "fold": fold_ids,
            "treatment": treatment,
            "outcome": outcome,
            "propensity": propensity,
            "cate": cate,
            "recommended_treatment": policy,
        }
    )
    artifacts.write_folds(
        {"subject_id": clean[args.subject_col].astype(str), "fold": fold_ids}
    )
    joblib.dump(checkpoints, Path(args.output_dir) / "checkpoint.joblib")
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
