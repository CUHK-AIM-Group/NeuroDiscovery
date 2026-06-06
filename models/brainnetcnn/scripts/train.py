"""BrainNetCNN training entry — NeuroClaw edition.

Usage:
    python models/brainnetcnn/scripts/train.py \
        --atlas schaefer_100_7net --n-epochs 50 --batch-size 16

    python models/brainnetcnn/scripts/train.py \
        --atlas aal_116 --labels-csv data/hcp_gender_labels.csv \
        --label-col label --n-epochs 100 --task classification
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.brainnetcnn.net.brainnetcnn import BrainNetCNN
from models.bnt.scripts.data_adapter import BNTDataset, bnt_collate, build_labels_from_csv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True)
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--include-t1", action="store_true")

    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--nclass", type=int, default=2)
    p.add_argument("--e2e-channels", type=int, default=32)
    p.add_argument("--e2n-channels", type=int, default=64)
    p.add_argument("--n2g-channels", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--device", default=None)

    p.add_argument("--task", choices=["classification", "regression"],
                   default="classification")
    p.add_argument("--save-dir", default="models/brainnetcnn/checkpoints")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    label_dtype = "float" if args.task == "regression" else "long"
    labels_map = None
    if args.labels_csv:
        labels_map = build_labels_from_csv(
            args.labels_csv, args.subject_col, args.label_col,
            label_dtype=label_dtype)

    ds = BNTDataset(atlas=args.atlas, labels=labels_map, label_dtype=label_dtype,
                    include_t1=args.include_t1)
    n_samples = len(ds)
    n_roi = ds[0][0].size(0)

    y_all = np.array([float(ds[i][1].item()) for i in range(n_samples)])
    print(f"BrainNetCNN | atlas={args.atlas} | n_roi={n_roi} | n={n_samples} | task={args.task}")

    if args.task == "classification":
        y_int = y_all.astype(int)
        uniq, counts = np.unique(y_int, return_counts=True)
        print(f"  classes: {dict(zip(uniq.tolist(), counts.tolist()))}")
        kfold = min(args.kfold, int(counts.min()))
        splitter = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.arange(n_samples), y_int))
    else:
        if args.nclass != 1:
            args.nclass = 1
        kfold = min(args.kfold, n_samples // 5)
        splitter = KFold(n_splits=kfold, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.arange(n_samples)))

    if args.dry_run:
        print("--dry-run: dataset OK, exiting.")
        return

    fold = min(args.fold, len(splits) - 1)
    train_idx, test_idx = splits[fold]
    print(f"  fold {fold}: train={len(train_idx)} test={len(test_idx)}")

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=bnt_collate, drop_last=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=args.batch_size,
                             collate_fn=bnt_collate)

    model = BrainNetCNN(
        n_roi=n_roi, nclass=args.nclass,
        e2e_channels=args.e2e_channels, e2n_channels=args.e2n_channels,
        n2g_channels=args.n2g_channels, dropout=args.dropout,
        task=args.task,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    best_val = -1.0 if args.task == "classification" else float("inf")
    best_state = None

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0
        n_batch = 0
        for x_batch, y_batch, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            if args.task == "regression":
                loss = F.mse_loss(logits.squeeze(-1), y_batch.float())
            else:
                loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == args.n_epochs - 1:
            model.eval()
            with torch.no_grad():
                if args.task == "classification":
                    correct = total = 0
                    for x_b, y_b, _ in test_loader:
                        x_b, y_b = x_b.to(device), y_b.to(device)
                        out = model(x_b)
                        correct += (out.argmax(1) == y_b).sum().item()
                        total += y_b.size(0)
                    metric = correct / max(total, 1)
                    improved = metric > best_val
                    if improved:
                        best_val = metric
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"  epoch {epoch+1:3d} | loss {total_loss/n_batch:.4f} | "
                          f"test_acc {metric:.4f} {'*' if improved else ''}")
                else:
                    preds, trues = [], []
                    for x_b, y_b, _ in test_loader:
                        x_b = x_b.to(device)
                        out = model(x_b)
                        preds.append(out.squeeze(-1).cpu())
                        trues.append(y_b)
                    mae = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
                    improved = mae < best_val
                    if improved:
                        best_val = mae
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"  epoch {epoch+1:3d} | loss {total_loss/n_batch:.4f} | "
                          f"test_MAE {mae:.4f} {'*' if improved else ''}")

    metric_name = "accuracy" if args.task == "classification" else "MAE"
    print(f"\n  Best test {metric_name}: {best_val:.4f}")

    save_dir = ROOT / args.save_dir / args.atlas
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / f"fold{fold}.pt"
    torch.save({"state_dict": best_state, "args": vars(args),
                f"best_{metric_name}": best_val, "n_roi": n_roi}, ckpt)
    print(f"  saved -> {ckpt}")


if __name__ == "__main__":
    main()
