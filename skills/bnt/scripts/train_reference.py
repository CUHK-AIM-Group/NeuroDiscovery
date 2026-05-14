"""BNT training reference — NeuroClaw edition.

Demonstrates the complete training loop for BrainNetworkTransformer with NeuroClaw data.
Supports both classification and regression tasks with stratified K-fold CV.

Usage (classification, smoke test):
    python skills/bnt/scripts/train_reference.py \
        --atlas aal_116 --fold 0 --n-epochs 10 --batch-size 8

Usage (regression):
    python skills/bnt/scripts/train_reference.py \
        --atlas schaefer_100_7net \
        --labels-csv data/hcp_age_labels.csv \
        --subject-col subject_id --label-col age \
        --task regression --fold 0

Full implementation: models/bnt/scripts/train.py (to be created)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.bnt.net.bnt import BrainNetworkTransformer
from models.bnt.scripts.data_adapter import (
    BNTDataset,
    bnt_collate,
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--nclass", type=int, default=2)
    p.add_argument("--dec-weight", type=float, default=0.1)
    p.add_argument("--pos-embed-dim", type=int, default=8)
    p.add_argument("--nhead", type=int, default=4)
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

    # 1. Build dataset
    ds = BNTDataset(
        atlas=args.atlas, labels=labels_map,
        include_t1=args.include_t1, subjects=subjects, label_dtype=label_dtype,
    )
    n_roi = ds.n_roi
    sample_fc = ds[0][0]
    feature_dim = sample_fc.size(1)
    n_samples = len(ds)
    print(f"Dataset: {n_samples} subjects, {n_roi} ROIs, feature_dim={feature_dim}")

    # 2. K-fold split
    y_all = np.array([float(ds[i][1].item()) for i in range(n_samples)])
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

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=bnt_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=bnt_collate)

    # 3. Model
    model = BrainNetworkTransformer(
        n_roi=n_roi, nclass=args.nclass, task=args.task,
        pos_embed_dim=args.pos_embed_dim, nhead=args.nhead,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 4. Training loop: task loss + DEC KL loss
    best_val = -1.0 if args.task == "classification" else float("inf")
    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        for fc_batch, y_batch, _ in train_loader:
            fc_batch, y_batch = fc_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, assignments = model(fc_batch)

            if args.task == "regression":
                loss_task = F.mse_loss(logits.squeeze(-1), y_batch.float())
            else:
                loss_task = F.cross_entropy(logits, y_batch)

            loss_dec = model.dec_loss(assignments)
            loss = loss_task + args.dec_weight * loss_dec
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * fc_batch.size(0)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if args.task == "classification":
                correct = 0
                for fc_b, y_b, _ in val_loader:
                    fc_b, y_b = fc_b.to(device), y_b.to(device)
                    out, _ = model(fc_b)
                    correct += (out.argmax(1) == y_b).sum().item()
                val_metric = correct / len(val_idx)
                improved = val_metric > best_val
            else:
                preds, trues = [], []
                for fc_b, y_b, _ in val_loader:
                    fc_b = fc_b.to(device)
                    out, _ = model(fc_b)
                    preds.append(out.squeeze(-1).cpu())
                    trues.append(y_b)
                val_metric = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
                improved = val_metric < best_val

        if improved:
            best_val = val_metric
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:3d} | loss {total_loss/len(train_idx):.4f} | val={val_metric:.3f}")

    print(f"\nBest val metric: {best_val:.4f}")


if __name__ == "__main__":
    main()
