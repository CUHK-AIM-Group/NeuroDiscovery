"""Com-BrainTF training reference — NeuroClaw edition.

Demonstrates the complete training loop for Com-BrainTF with NeuroClaw data.
Auto-derives community partitions from atlas (Yeo 7-net for Schaefer, lobe for
AAL, hash fallback otherwise) and supports classification + regression with
stratified K-fold CV.

Usage (classification, smoke test):
    python skills/combraintf/scripts/train_reference.py \
        --atlas schaefer_200_7net --fold 0 --n-epochs 10 --batch-size 8

Usage (regression):
    python skills/combraintf/scripts/train_reference.py \
        --atlas aal_116 \
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.combraintf.net.combraintf import ComBrainTF
from models.combraintf.scripts.data_adapter import (
    BNTDataset,
    bnt_collate,
    build_labels_from_csv,
    get_community_ids,
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--nclass", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--n-communities", type=int, default=None,
                   help="override community count (used only for hash-fallback atlases)")
    p.add_argument("--dec-weight", type=float, default=0.1)
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

    ds = BNTDataset(atlas=args.atlas, labels=labels_map, label_dtype=label_dtype)
    n_samples = len(ds)
    n_roi = ds[0][0].size(0)
    community_ids = get_community_ids(args.atlas, n_communities=args.n_communities)
    n_communities = len(set(community_ids))
    print(f"Dataset: {n_samples} subjects, {n_roi} ROIs, K={n_communities} communities")

    y_all = np.array([float(ds[i][1].item()) for i in range(n_samples)])
    if args.task == "classification":
        splitter = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    else:
        args.nclass = 1
        splitter = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.arange(n_samples),
                                 y_all if args.task == "classification" else None))
    train_idx, test_idx = splits[args.fold]

    train_sub = torch.utils.data.Subset(ds, train_idx)
    test_sub = torch.utils.data.Subset(ds, test_idx)
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True,
                               collate_fn=bnt_collate, drop_last=True)
    test_loader = DataLoader(test_sub, batch_size=args.batch_size, collate_fn=bnt_collate)

    # nhead must divide d_model (= n_roi)
    nhead = args.nhead
    while n_roi % nhead != 0 and nhead > 1:
        nhead -= 1
    if nhead != args.nhead:
        print(f"  nhead adjusted: {args.nhead} -> {nhead} (must divide n_roi={n_roi})")

    model = ComBrainTF(
        n_roi=n_roi, nclass=args.nclass, community_ids=community_ids,
        n_communities=n_communities, n_clusters=args.n_clusters,
        hidden_size=args.hidden_size, nhead=nhead, task=args.task,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0.0
        for fc_batch, y_batch, _ in train_loader:
            fc_batch, y_batch = fc_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits, assignment = model(fc_batch)
            if args.task == "regression":
                loss_task = F.mse_loss(logits.squeeze(-1), y_batch.float())
            else:
                loss_task = F.cross_entropy(logits, y_batch)
            loss_dec = model.dec_loss(assignment)
            loss = loss_task + args.dec_weight * loss_dec
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * fc_batch.size(0)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if args.task == "classification":
                correct = total = 0
                for fc_b, y_b, _ in test_loader:
                    fc_b, y_b = fc_b.to(device), y_b.to(device)
                    out, _ = model(fc_b)
                    correct += (out.argmax(1) == y_b).sum().item()
                    total += y_b.size(0)
                metric = correct / max(total, 1)
                tag = "acc"
            else:
                preds, trues = [], []
                for fc_b, y_b, _ in test_loader:
                    fc_b = fc_b.to(device)
                    out, _ = model(fc_b)
                    preds.append(out.squeeze(-1).cpu())
                    trues.append(y_b)
                metric = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
                tag = "mae"
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:3d} | loss {total_loss/len(train_idx):.4f} | test_{tag}={metric:.3f}")


if __name__ == "__main__":
    main()
