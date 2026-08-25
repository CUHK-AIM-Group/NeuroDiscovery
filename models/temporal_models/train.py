"""Cross-validated sequence trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.common.artifacts import RunArtifacts
from models.common.data import make_splits
from models.common.metrics import classification_metrics, regression_metrics
from models.temporal_models.net import TemporalPredictor


def _fit_fold(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    valid_train = np.concatenate(
        [X[index, : lengths[index]] for index in train_idx], axis=0
    )
    mean = valid_train.mean(axis=0, keepdims=True)
    std = valid_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X = (X - mean[None, ...]) / std[None, ...]
    for index, length in enumerate(lengths):
        if length < X.shape[1]:
            X[index, length:] = 0
    output_dim = len(np.unique(y)) if args.task == "classification" else 1
    model = TemporalPredictor(
        X.shape[2],
        output_dim,
        args.model,
        args.hidden_dim,
        args.layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    x_tensor = torch.as_tensor(X, dtype=torch.float32)
    length_tensor = torch.as_tensor(lengths, dtype=torch.long)
    y_tensor = torch.as_tensor(
        y,
        dtype=torch.long if args.task == "classification" else torch.float32,
    )
    loader = DataLoader(
        TensorDataset(
            x_tensor[train_idx], length_tensor[train_idx], y_tensor[train_idx]
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        for xb, lb, yb in loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb, lb)
            loss = (
                loss_fn(output, yb)
                if args.task == "classification"
                else loss_fn(output.squeeze(-1), yb)
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(
            x_tensor[test_idx].to(device),
            length_tensor[test_idx].to(device),
        )
        if args.task == "classification":
            probabilities = output.softmax(dim=-1).cpu().numpy()
            prediction = probabilities.argmax(axis=1)
            score = probabilities[:, -1]
        else:
            prediction = output.squeeze(-1).cpu().numpy()
            score = None
    return model, prediction, score, mean.squeeze(0), std.squeeze(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with X, y, optional lengths/id")
    parser.add_argument("--model", choices=["lstm", "gru", "tcn", "transformer"], required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    blob = np.load(args.input, allow_pickle=True)
    X, y = blob["X"].astype(np.float32), blob["y"]
    lengths = (
        blob["lengths"].astype(int)
        if "lengths" in blob
        else np.full(len(X), X.shape[1], dtype=int)
    )
    ids = (
        blob["subject_id"].astype(str)
        if "subject_id" in blob
        else np.arange(len(X)).astype(str)
    )
    pred = np.empty(len(y), dtype=float)
    scores = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    checkpoints = []
    for fold, (train_idx, test_idx) in enumerate(
        make_splits(y, args.task, args.folds, args.seed)
    ):
        model, fold_pred, fold_score, mean, std = _fit_fold(
            X, y, lengths, train_idx, test_idx, args
        )
        pred[test_idx] = fold_pred
        if fold_score is not None:
            scores[test_idx] = fold_score
        fold_ids[test_idx] = fold
        checkpoints.append(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "feature_mean": mean,
                "feature_std": std,
            }
        )
    metrics = (
        classification_metrics(y, pred.astype(y.dtype), scores)
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
            "score": scores,
        }
    )
    artifacts.write_folds({"subject_id": ids, "fold": fold_ids})
    torch.save(checkpoints, Path(args.output_dir) / "checkpoint.pt")
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
