"""Canonical CLI for CPM using NPZ connectomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.common.artifacts import RunArtifacts
from models.common.data import make_splits
from models.common.metrics import classification_metrics, regression_metrics
from models.cpm.cpm import CPM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectomes", required=True, help="NPZ with X and subject_id")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--p-threshold", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    blob = np.load(args.connectomes, allow_pickle=True)
    ids = blob["subject_id"].astype(str)
    X = blob["X"]
    labels = pd.read_csv(args.labels)
    target_by_id = dict(
        zip(labels[args.subject_col].astype(str), labels[args.target])
    )
    keep = np.array([sid in target_by_id for sid in ids])
    ids, X = ids[keep], X[keep]
    y = np.asarray([target_by_id[sid] for sid in ids])
    pred = np.empty(len(y), dtype=float)
    score = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    models = []
    for fold, (train, test) in enumerate(
        make_splits(y, args.task, args.folds, args.seed)
    ):
        model = CPM(args.task, args.p_threshold).fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        if args.task == "classification":
            score[test] = model.predict_proba(X[test])[:, -1]
        fold_ids[test] = fold
        models.append(model)
    metrics = (
        classification_metrics(y, pred, score)
        if args.task == "classification"
        else regression_metrics(y, pred)
    )
    artifacts = RunArtifacts(args.output_dir, vars(args))
    artifacts.write_config()
    artifacts.write_metrics(metrics)
    artifacts.write_predictions(
        {
            "subject_id": ids,
            "fold": fold_ids,
            "target": y,
            "prediction": pred,
            "score": score,
        }
    )
    artifacts.write_folds({"subject_id": ids, "fold": fold_ids})
    joblib.dump(models, Path(args.output_dir) / "checkpoint.joblib")
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
