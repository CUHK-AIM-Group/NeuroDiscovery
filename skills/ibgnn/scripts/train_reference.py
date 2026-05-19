"""IBGNN training reference — NeuroClaw edition.

Demonstrates the complete training loop for IBGNN with NeuroClaw data.
Supports both classification and regression tasks with stratified K-fold CV.

Usage (classification, smoke test):
    python skills/ibgnn/scripts/train_reference.py \
        --atlas aal_116 --fold 0 --n-epochs 10 --batch-size 16

Usage (regression):
    python skills/ibgnn/scripts/train_reference.py \
        --atlas schaefer_100_7net \
        --labels-csv data/hcp_age_labels.csv \
        --subject-col subject_id --label-col age \
        --task regression --fold 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.loader import DataLoader as PyGLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.ibgnn.net.ibgnn import IBGNN
from models.ibgnn.scripts.data_adapter import (
    NeuroClawFCDataset,
    build_labels_from_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--nclass", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-gnn-layers", type=int, default=2)
    p.add_argument("--n-mlp-layers", type=int, default=1)
    p.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    label_dtype = "float" if args.task == "regression" else "long"
    labels_map = None
    if args.labels_csv:
        labels_map = build_labels_from_csv(
            args.labels_csv, args.subject_col, args.label_col, label_dtype=label_dtype
        )

    ds = NeuroClawFCDataset(atlas=args.atlas, labels=labels_map, label_dtype=label_dtype)
    n_samples = len(ds)
    n_roi = int(ds[0].x.size(0))
    indim = int(ds[0].x.size(1))
    print(f"Dataset: {n_samples} subjects, {n_roi} ROIs, indim={indim}")

    y_all = np.array([float(ds[i].y.item()) for i in range(n_samples)])
    if args.task == "classification":
        splitter = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    else:
        args.nclass = 1
        splitter = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.arange(n_samples),
                                 y_all if args.task == "classification" else None))
    train_idx, test_idx = splits[args.fold]

    train_loader = PyGLoader(ds[train_idx], batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = PyGLoader(ds[test_idx], batch_size=args.batch_size)

    model = IBGNN(
        n_roi=indim, nclass=args.nclass, hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers, n_mlp_layers=args.n_mlp_layers,
        pooling=args.pooling, task=args.task,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            if args.task == "regression":
                loss = F.mse_loss(logits.squeeze(-1), batch.y.float())
            else:
                loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if args.task == "classification":
                correct = total = 0
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                    correct += (out.argmax(1) == batch.y).sum().item()
                    total += batch.y.size(0)
                metric = correct / max(total, 1)
                tag = "acc"
            else:
                preds, trues = [], []
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                    preds.append(out.squeeze(-1).cpu())
                    trues.append(batch.y.cpu())
                metric = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
                tag = "mae"
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:3d} | loss {total_loss/len(train_idx):.4f} | test_{tag}={metric:.3f}")


if __name__ == "__main__":
    main()
