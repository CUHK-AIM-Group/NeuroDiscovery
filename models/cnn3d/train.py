"""Cross-validated trainer for the canonical CNN3D model."""

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
from models.cnn3d.net import VoxelCNN3D


def _fit_fold(X, y, train_idx, test_idx, args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    channel_mean = X[train_idx].mean(axis=(0, 2, 3, 4), keepdims=True)
    channel_std = X[train_idx].std(axis=(0, 2, 3, 4), keepdims=True)
    channel_std[channel_std < 1e-6] = 1.0
    X = (X - channel_mean) / channel_std
    output_dim = len(np.unique(y)) if args.task == "classification" else 1
    model = VoxelCNN3D(
        in_channels=X.shape[1],
        output_dim=output_dim,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    inputs = torch.as_tensor(X, dtype=torch.float32)
    targets = torch.as_tensor(
        y,
        dtype=torch.long if args.task == "classification" else torch.float32,
    )
    loader = DataLoader(
        TensorDataset(inputs[train_idx], targets[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = (
                loss_fn(output, yb)
                if args.task == "classification"
                else loss_fn(output.squeeze(-1), yb)
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(inputs[test_idx].to(device))
        if args.task == "classification":
            probability = output.softmax(dim=-1).cpu().numpy()
            return (
                model,
                probability.argmax(axis=1),
                probability[:, -1],
                channel_mean,
                channel_std,
            )
        return (
            model,
            output.squeeze(-1).cpu().numpy(),
            None,
            channel_mean,
            channel_std,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with X and y")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
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
    if X.ndim == 4:
        X = X[:, None]
    if X.ndim != 5:
        raise ValueError("X must have shape [N, D, H, W] or [N, C, D, H, W]")
    ids = (
        blob["subject_id"].astype(str)
        if "subject_id" in blob
        else np.arange(len(X)).astype(str)
    )
    pred, score = np.empty(len(y)), np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    checkpoints = []
    for fold, (train_idx, test_idx) in enumerate(
        make_splits(y, args.task, args.folds, args.seed)
    ):
        model, fold_pred, fold_score, mean, std = _fit_fold(
            X, y, train_idx, test_idx, args
        )
        pred[test_idx] = fold_pred
        if fold_score is not None:
            score[test_idx] = fold_score
        fold_ids[test_idx] = fold
        checkpoints.append(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "channel_mean": mean,
                "channel_std": std,
            }
        )
    metrics = (
        classification_metrics(y, pred.astype(y.dtype), score)
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
    torch.save(checkpoints, Path(args.output_dir) / "checkpoint.pt")
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
