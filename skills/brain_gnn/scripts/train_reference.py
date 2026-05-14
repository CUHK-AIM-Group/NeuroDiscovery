"""BrainGNN training reference — NeuroClaw edition.

Demonstrates the complete training loop for BrainGNN with NeuroClaw data format.
Supports both classification and regression tasks with stratified K-fold CV.

Usage (classification, smoke test):
    python skills/brain_gnn/scripts/train_reference.py \
        --atlas aal_116 --fold 0 --n-epochs 10 --batch-size 4

Usage (regression):
    python skills/brain_gnn/scripts/train_reference.py \
        --atlas schaefer_100_7net \
        --labels-csv data/hcp_age_labels.csv \
        --subject-col subject_id --label-col age \
        --task regression --fold 0

Full implementation: models/braingnn/scripts/train.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.braingnn.net.braingnn import (
    BrainGNN,
    consist_loss,
    topk_loss,
    unit_loss,
)
from models.braingnn.scripts.data_adapter import (
    NeuroClawFCDataset,
    build_labels_from_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--subjects-file", default=None)
    p.add_argument("--include-t1", action="store_true")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--nclass", type=int, default=2)
    p.add_argument("--lamb3", type=float, default=0.1)
    p.add_argument("--lamb4", type=float, default=0.1)
    p.add_argument("--lamb5", type=float, default=0.1)
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

    subjects = None
    if args.subjects_file:
        subjects = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]

    # 1. Build dataset — PyG InMemoryDataset with auto-caching
    ds = NeuroClawFCDataset(
        atlas=args.atlas, labels=labels_map,
        include_t1=args.include_t1, subjects=subjects, label_dtype=label_dtype,
    )
    n_roi = int(ds[0].x.size(0))
    indim = int(ds[0].x.size(1))
    n_samples = len(ds)
    print(f"Dataset: {n_samples} subjects, {n_roi} ROIs, indim={indim}")

    # 2. K-fold split
    y_all = np.array([float(ds[i].y.item()) for i in range(n_samples)])
    if args.task == "classification":
        kfold = min(args.kfold, int(np.unique(y_all, return_counts=True)[1].min()))
        splitter = StratifiedKFold(n_splits=max(2, kfold), shuffle=True, random_state=args.seed)
    else:
        args.nclass = 1
        kfold = min(args.kfold, n_samples // 2)
        splitter = KFold(n_splits=max(2, kfold), shuffle=True, random_state=args.seed)

    splits = list(splitter.split(np.arange(n_samples), y_all if args.task == "classification" else None))
    train_val_idx, test_idx = splits[args.fold]
    cut = int(0.8 * len(train_val_idx))
    train_idx, val_idx = train_val_idx[:cut], train_val_idx[cut:]

    train_loader = DataLoader(ds[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds[val_idx], batch_size=args.batch_size)
    test_loader = DataLoader(ds[test_idx], batch_size=args.batch_size)

    # 3. Model
    model = BrainGNN(
        indim=indim, ratio=args.ratio, nclass=args.nclass,
        n_roi=n_roi, task=args.task,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 4. Training loop with 6-term loss
    best_val = -1.0 if args.task == "classification" else float("inf")
    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, w1, w2, s1, s2 = model(
                batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos
            )
            if args.task == "regression":
                loss_c = F.mse_loss(output.squeeze(-1), batch.y.float())
                loss_gc = torch.tensor(0.0, device=device)
            else:
                loss_c = F.nll_loss(output, batch.y)
                loss_gc = sum(
                    consist_loss(s1[batch.y == c])
                    for c in range(args.nclass) if (batch.y == c).sum() >= 2
                ) or torch.tensor(0.0, device=device)

            loss = (loss_c
                    + args.lamb3 * topk_loss(s1, args.ratio)
                    + args.lamb4 * topk_loss(s2, args.ratio)
                    + args.lamb5 * loss_gc
                    + unit_loss(w1) * 0.0
                    + unit_loss(w2) * 0.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if args.task == "classification":
                correct = sum(
                    (model(b.to(device).x, b.to(device).edge_index, b.to(device).batch,
                           b.to(device).edge_attr, b.to(device).pos)[0].argmax(1) == b.to(device).y).sum().item()
                    for b in val_loader
                )
                val_metric = correct / len(val_idx)
                improved = val_metric > best_val
            else:
                preds, trues = [], []
                for b in val_loader:
                    b = b.to(device)
                    out = model(b.x, b.edge_index, b.batch, b.edge_attr, b.pos)[0]
                    preds.append(out.squeeze(-1).cpu())
                    trues.append(b.y.cpu())
                val_metric = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
                improved = val_metric < best_val

        if improved:
            best_val = val_metric
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:3d} | loss {total_loss/len(train_idx):.4f} | val={val_metric:.3f}")

    print(f"\nBest val metric: {best_val:.4f}")


if __name__ == "__main__":
    main()
