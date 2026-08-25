"""Cross-validated brain-age training with fold-local bias correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from models.brain_age.correction import BrainAgeBiasCorrector
from models.common.artifacts import RunArtifacts
from models.common.data import load_tabular_csv, make_splits
from models.common.metrics import regression_metrics
from models.statistical_ml.estimators import make_estimator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--age-col", default="age")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--group-col")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    data = load_tabular_csv(
        args.features, args.age_col, args.subject_col, args.group_col
    )
    raw = np.empty(len(data.y))
    corrected = np.empty(len(data.y))
    fold_ids = np.full(len(data.y), -1)
    checkpoints = []
    for fold, (train, test) in enumerate(
        make_splits(data.y, "regression", args.folds, args.seed, data.groups)
    ):
        predictor = make_estimator(args.model, "regression", args.seed + fold)
        predictor.fit(data.X[train], data.y[train])
        train_pred = predictor.predict(data.X[train])
        test_pred = predictor.predict(data.X[test])
        corrector = BrainAgeBiasCorrector().fit(data.y[train], train_pred)
        raw[test] = test_pred
        corrected[test] = corrector.transform(data.y[test], test_pred)
        fold_ids[test] = fold
        checkpoints.append((predictor, corrector))
    metrics = {
        "raw": regression_metrics(data.y, raw),
        "bias_corrected": regression_metrics(data.y, corrected),
        "brain_pad_mean": float(np.mean(corrected - data.y.astype(float))),
        "brain_pad_sd": float(np.std(corrected - data.y.astype(float), ddof=1)),
    }
    artifacts = RunArtifacts(args.output_dir, vars(args))
    artifacts.write_config()
    artifacts.write_metrics(metrics)
    artifacts.write_predictions(
        {
            "subject_id": data.subject_ids,
            "fold": fold_ids,
            "chronological_age": data.y,
            "predicted_age_raw": raw,
            "predicted_age_corrected": corrected,
            "brain_pad": corrected - data.y.astype(float),
        }
    )
    artifacts.write_folds({"subject_id": data.subject_ids, "fold": fold_ids})
    joblib.dump(checkpoints, Path(args.output_dir) / "checkpoint.joblib")
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
