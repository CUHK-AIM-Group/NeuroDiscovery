"""Cross-validated CLI for censored outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models.common.artifacts import RunArtifacts
from models.common.data import make_splits
from models.survival_models.estimators import make_survival_estimator
from models.survival_models.metrics import concordance_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--duration-col", required=True)
    parser.add_argument("--event-col", required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--group-col")
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
    excluded = {
        args.subject_col,
        args.duration_col,
        args.event_col,
        args.group_col,
    }
    features = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    clean = frame.dropna(
        subset=[args.duration_col, args.event_col, *features]
    ).reset_index(drop=True)
    X = clean[features].to_numpy(dtype=np.float32)
    duration = clean[args.duration_col].to_numpy(dtype=float)
    event = clean[args.event_col].to_numpy(dtype=int)
    groups = clean[args.group_col].to_numpy() if args.group_col else None
    risk = np.empty(len(clean))
    fold_ids = np.full(len(clean), -1)
    models = []
    for fold, (train, test) in enumerate(
        make_splits(event, "classification", args.folds, args.seed, groups)
    ):
        params = (
            {"epochs": args.epochs, "device": args.device}
            if "deep" in args.model.lower()
            else {}
        )
        estimator = make_survival_estimator(args.model, args.seed + fold, **params)
        estimator.fit(X[train], duration[train], event[train])
        risk[test] = estimator.predict_risk(X[test])
        fold_ids[test] = fold
        if hasattr(estimator, "network_"):
            estimator.network_.cpu()
        models.append(estimator)
    metrics = {"concordance_index": concordance_index(duration, event, risk)}
    artifacts = RunArtifacts(args.output_dir, vars(args))
    artifacts.write_config()
    artifacts.write_metrics(metrics)
    artifacts.write_predictions(
        {
            "subject_id": clean[args.subject_col].astype(str),
            "fold": fold_ids,
            "duration": duration,
            "event": event,
            "risk": risk,
        }
    )
    artifacts.write_folds(
        {"subject_id": clean[args.subject_col].astype(str), "fold": fold_ids}
    )
    joblib.dump(models, Path(args.output_dir) / "checkpoint.joblib")
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
