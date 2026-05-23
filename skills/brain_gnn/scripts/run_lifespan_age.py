"""Lifespan age regression — 80/10/10 split, z-score y, multi-model multi-atlas.

Combines HCP-YA + HCP-A (or any other dataset listed in the lifespan label CSV)
into a single training run. Uses an 80/10/10 deterministic split, z-scores age
labels using train-set statistics, picks the best model on val MAE, reports
test MAE in both years and z-units.

Targets:
  - Match NeuroStorm-style numbers (BrainNetCNN cc200 = 0.81 z, BNT basc_122 = 0.82 z)
  - Validate the protocol switch (5-fold CV -> 80/10/10) consistently

Usage:
    python skills/brain_gnn/scripts/run_lifespan_age.py --atlases cc200 basc_122 schaefer_100_7net
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.braingnn.net.braingnn import BrainGNN, topk_loss
from models.braingnn.scripts.data_adapter import _pyg_data_from_subject
from models.bnt.net.bnt import BrainNetworkTransformer
from models.brainnetcnn.net.brainnetcnn import BrainNetCNN
from models.lggnn.net.lggnn import LGGNN
from models.ibgnn.net.ibgnn import IBGNN
from models.combraintf.net.combraintf import ComBrainTF
from models.combraintf.scripts.data_adapter import get_community_ids


def _build_dense_fc(sub_pt) -> torch.Tensor:
    """Same transform as BNTDataset: Fisher-z -> Pearson r, diag=0."""
    fc_z = sub_pt.get("fc_matrix", sub_pt.get("node_features"))
    if not torch.is_tensor(fc_z):
        fc_z = torch.as_tensor(fc_z)
    fc_z = fc_z.float()
    corr = torch.tanh(fc_z)
    corr.fill_diagonal_(0.0)
    return corr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_CSV = ROOT / "data" / "labels" / "lifespan_age_labels.csv"


def pt_path_for(dataset: str, sid: str, atlas: str) -> Path:
    if dataset == "hcpya":
        return ROOT / "data" / "braingnn_input" / atlas / f"sub-{sid}.pt"
    return ROOT / "data" / "braingnn_input" / atlas / f"sub-{dataset}_{sid}.pt"


# ---------------- Dataset wrappers ----------------

class PyGLifespanDataset:
    """Build a list[Data] for BrainGNN from lifespan labels CSV."""

    def __init__(self, atlas: str, df: pd.DataFrame, y_mean: float, y_std: float):
        self.data_list = []
        for _, row in df.iterrows():
            p = pt_path_for(row["dataset"], row["subject_id"], atlas)
            if not p.exists():
                continue
            sub_pt = torch.load(p, weights_only=False)
            y_z = (float(row["label"]) - y_mean) / y_std
            try:
                d = _pyg_data_from_subject(sub_pt, None, y_z, None,
                                           label_dtype="float")
                self.data_list.append(d)
            except Exception:
                continue

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class DenseFCLifespanDataset(Dataset):
    """Dense FC matrices for BNT/BrainNetCNN."""

    def __init__(self, atlas: str, df: pd.DataFrame, y_mean: float, y_std: float):
        self.items = []
        for _, row in df.iterrows():
            p = pt_path_for(row["dataset"], row["subject_id"], atlas)
            if not p.exists():
                continue
            sub_pt = torch.load(p, weights_only=False)
            try:
                fc = _build_dense_fc(sub_pt)
            except Exception:
                continue
            y_z = (float(row["label"]) - y_mean) / y_std
            self.items.append((fc, torch.tensor(y_z, dtype=torch.float32),
                                row["subject_id"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def dense_collate(batch):
    fcs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    sids = [b[2] for b in batch]
    return fcs, ys, sids


# ---------------- Split helpers ----------------

def make_split(df: pd.DataFrame, seed: int = 42, val_frac=0.1, test_frac=0.1):
    """Deterministic 80/10/10 stratified by dataset (so each split has both)."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for ds, sub in df.groupby("dataset"):
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    return df.loc[train_idx].reset_index(drop=True), \
           df.loc[val_idx].reset_index(drop=True), \
           df.loc[test_idx].reset_index(drop=True)


# ---------------- Training fns ----------------

def train_braingnn(atlas: str, train_df, val_df, test_df, y_mean, y_std,
                    n_epochs=80, batch_size=16, lr=1e-3, wd=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = PyGLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = PyGLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = PyGLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.data_list:
        return {"error": "empty train"}
    n_roi = int(train_ds.data_list[0].x.size(0))
    indim = int(train_ds.data_list[0].x.size(1))
    train_loader = PyGLoader(train_ds.data_list, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = PyGLoader(val_ds.data_list, batch_size=batch_size)
    test_loader = PyGLoader(test_ds.data_list, batch_size=batch_size)

    model = BrainGNN(indim=indim, ratio=0.5, nclass=1, n_roi=n_roi,
                      n_communities=16, task="regression").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out, _, _, s1, s2 = model(batch.x, batch.edge_index, batch.batch,
                                       batch.edge_attr, batch.pos)
            loss = F.mse_loss(out.squeeze(-1), batch.y.float())
            loss = loss + 0.1 * topk_loss(s1, 0.5) + 0.1 * topk_loss(s2, 0.5)
            loss.backward(); opt.step()
        sched.step()
        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch,
                            batch.edge_attr, batch.pos)[0]
                preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Test using best model
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch,
                        batch.edge_attr, batch.pos)[0]
            preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
    test_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    test_mae_yr = test_mae_z * y_std
    return {"val_mae_z": best_val, "test_mae_z": test_mae_z,
            "test_mae_yr": test_mae_yr, "n_train": len(train_ds),
            "n_val": len(val_ds), "n_test": len(test_ds)}


def train_bnt(atlas, train_df, val_df, test_df, y_mean, y_std,
               n_epochs=80, batch_size=16, lr=5e-4, wd=1e-4, hidden=1024,
               nhead=4, dropout=0.3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = DenseFCLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = DenseFCLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = DenseFCLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.items:
        return {"error": "empty train"}
    n_roi = train_ds.items[0][0].size(0)
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=dense_collate, drop_last=True)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, collate_fn=dense_collate)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, collate_fn=dense_collate)

    model = BrainNetworkTransformer(n_roi=n_roi, nclass=1, task="regression",
                                     hidden_size=hidden, nhead=nhead,
                                     dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None

    for epoch in range(n_epochs):
        model.train()
        for fc, y, _ in train_loader:
            fc, y = fc.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, assign = model(fc)
            loss = F.mse_loss(logits.squeeze(-1), y) + 0.1 * model.dec_loss(assign)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for fc, y, _ in val_loader:
                fc = fc.to(DEVICE)
                out, _ = model(fc)
                preds.append(out.squeeze(-1).cpu()); trues.append(y)
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for fc, y, _ in test_loader:
            fc = fc.to(DEVICE)
            out, _ = model(fc)
            preds.append(out.squeeze(-1).cpu()); trues.append(y)
    test_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    return {"val_mae_z": best_val, "test_mae_z": test_mae_z,
            "test_mae_yr": test_mae_z * y_std, "n_train": len(train_ds),
            "n_val": len(val_ds), "n_test": len(test_ds)}


def train_brainnetcnn(atlas, train_df, val_df, test_df, y_mean, y_std,
                       n_epochs=80, batch_size=16, lr=1e-4, wd=5e-4,
                       dropout=0.5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = DenseFCLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = DenseFCLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = DenseFCLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.items:
        return {"error": "empty train"}
    n_roi = train_ds.items[0][0].size(0)
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=dense_collate, drop_last=True)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, collate_fn=dense_collate)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, collate_fn=dense_collate)

    model = BrainNetCNN(n_roi=n_roi, nclass=1, e2e_channels=32, e2n_channels=64,
                         n2g_channels=256, dropout=dropout, task="regression").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None

    for epoch in range(n_epochs):
        model.train()
        for fc, y, _ in train_loader:
            fc, y = fc.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(fc)
            loss = F.mse_loss(out.squeeze(-1), y)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for fc, y, _ in val_loader:
                fc = fc.to(DEVICE)
                out = model(fc)
                preds.append(out.squeeze(-1).cpu()); trues.append(y)
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for fc, y, _ in test_loader:
            fc = fc.to(DEVICE)
            out = model(fc)
            preds.append(out.squeeze(-1).cpu()); trues.append(y)
    test_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    return {"val_mae_z": best_val, "test_mae_z": test_mae_z,
            "test_mae_yr": test_mae_z * y_std, "n_train": len(train_ds),
            "n_val": len(val_ds), "n_test": len(test_ds)}


def train_lggnn(atlas, train_df, val_df, test_df, y_mean, y_std,
                n_epochs=80, batch_size=16, lr=1e-3, wd=5e-4,
                hidden_dim=64, embed_dim=20, ratio=0.5, dropout=0.3,
                mi_weight=0.1, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = PyGLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = PyGLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = PyGLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.data_list:
        return {"error": "empty train"}
    indim = int(train_ds.data_list[0].x.size(1))
    train_loader = PyGLoader(train_ds.data_list, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = PyGLoader(val_ds.data_list, batch_size=batch_size)
    test_loader = PyGLoader(test_ds.data_list, batch_size=batch_size)

    model = LGGNN(n_roi=indim, nclass=1, hidden_dim=hidden_dim,
                   embed_dim=embed_dim, ratio=ratio, dropout=dropout,
                   task="regression").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits, mi = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = F.mse_loss(logits.squeeze(-1), batch.y.float()) - mi_weight * mi
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
    test_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    return {"val_mae_z": best_val, "test_mae_z": test_mae_z,
            "test_mae_yr": test_mae_z * y_std, "n_train": len(train_ds),
            "n_val": len(val_ds), "n_test": len(test_ds)}


def train_ibgnn(atlas, train_df, val_df, test_df, y_mean, y_std,
                n_epochs=80, batch_size=16, lr=1e-3, wd=5e-4,
                hidden_dim=128, n_gnn_layers=2, n_mlp_layers=1,
                pooling="mean", seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = PyGLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = PyGLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = PyGLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.data_list:
        return {"error": "empty train"}
    indim = int(train_ds.data_list[0].x.size(1))
    train_loader = PyGLoader(train_ds.data_list, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = PyGLoader(val_ds.data_list, batch_size=batch_size)
    test_loader = PyGLoader(test_ds.data_list, batch_size=batch_size)

    model = IBGNN(n_roi=indim, nclass=1, hidden_dim=hidden_dim,
                   n_gnn_layers=n_gnn_layers, n_mlp_layers=n_mlp_layers,
                   pooling=pooling, task="regression").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = F.mse_loss(logits.squeeze(-1), batch.y.float())
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
    test_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    return {"val_mae_z": best_val, "test_mae_z": test_mae_z,
            "test_mae_yr": test_mae_z * y_std, "n_train": len(train_ds),
            "n_val": len(val_ds), "n_test": len(test_ds)}


def train_combraintf(atlas, train_df, val_df, test_df, y_mean, y_std,
                      n_epochs=80, batch_size=8, lr=1e-4, wd=5e-4,
                      hidden_size=512, nhead=4, n_clusters=8,
                      dec_weight=0.1, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = DenseFCLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = DenseFCLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = DenseFCLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.items:
        return {"error": "empty train"}
    n_roi = train_ds.items[0][0].size(0)
    community_ids = get_community_ids(atlas)
    n_communities = len(set(community_ids))
    # nhead must divide n_roi
    while n_roi % nhead != 0 and nhead > 1:
        nhead -= 1
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=dense_collate, drop_last=True)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, collate_fn=dense_collate)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, collate_fn=dense_collate)

    model = ComBrainTF(n_roi=n_roi, nclass=1, community_ids=community_ids,
                        n_communities=n_communities, n_clusters=n_clusters,
                        hidden_size=hidden_size, nhead=nhead,
                        task="regression").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None

    for epoch in range(n_epochs):
        model.train()
        for fc, y, _ in train_loader:
            fc, y = fc.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, assign = model(fc)
            loss = F.mse_loss(logits.squeeze(-1), y) + dec_weight * model.dec_loss(assign)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for fc, y, _ in val_loader:
                fc = fc.to(DEVICE)
                out, _ = model(fc)
                preds.append(out.squeeze(-1).cpu()); trues.append(y)
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for fc, y, _ in test_loader:
            fc = fc.to(DEVICE)
            out, _ = model(fc)
            preds.append(out.squeeze(-1).cpu()); trues.append(y)
    test_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    return {"val_mae_z": best_val, "test_mae_z": test_mae_z,
            "test_mae_yr": test_mae_z * y_std, "n_train": len(train_ds),
            "n_val": len(val_ds), "n_test": len(test_ds), "nhead_used": nhead}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlases", nargs="+",
                   default=["cc200", "basc_122", "schaefer_100_7net",
                            "schaefer_200_7net", "aal_116", "glasser_360"])
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+",
                   default=["braingnn", "bnt", "brainnetcnn", "lggnn", "ibgnn", "combraintf"])
    p.add_argument("--output", default="lifespan_age_results.json")
    args = p.parse_args()

    df = pd.read_csv(LABEL_CSV, dtype={"subject_id": str})
    train_df, val_df, test_df = make_split(df, seed=args.seed)
    y_mean = float(train_df["label"].mean())
    y_std = float(train_df["label"].std())
    print(f"Lifespan age | n={len(df)} | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"y stats (train): mean={y_mean:.2f} std={y_std:.2f}")
    print(f"datasets in train: {train_df['dataset'].value_counts().to_dict()}")
    print(f"DEVICE: {DEVICE} | atlases: {args.atlases} | models: {args.models}")
    print(f"epochs={args.n_epochs} batch={args.batch_size}\n", flush=True)

    results = []
    for atlas in args.atlases:
        print(f"\n{'='*70}\nAtlas: {atlas}\n{'='*70}", flush=True)
        for model_name in args.models:
            t0 = time.time()
            try:
                if model_name == "braingnn":
                    r = train_braingnn(atlas, train_df, val_df, test_df,
                                        y_mean, y_std,
                                        n_epochs=args.n_epochs,
                                        batch_size=args.batch_size, seed=args.seed)
                elif model_name == "bnt":
                    r = train_bnt(atlas, train_df, val_df, test_df,
                                   y_mean, y_std, n_epochs=args.n_epochs,
                                   batch_size=args.batch_size, seed=args.seed)
                elif model_name == "brainnetcnn":
                    r = train_brainnetcnn(atlas, train_df, val_df, test_df,
                                           y_mean, y_std, n_epochs=args.n_epochs,
                                           batch_size=args.batch_size, seed=args.seed)
                elif model_name == "lggnn":
                    r = train_lggnn(atlas, train_df, val_df, test_df,
                                     y_mean, y_std, n_epochs=args.n_epochs,
                                     batch_size=args.batch_size, seed=args.seed)
                elif model_name == "ibgnn":
                    r = train_ibgnn(atlas, train_df, val_df, test_df,
                                     y_mean, y_std, n_epochs=args.n_epochs,
                                     batch_size=args.batch_size, seed=args.seed)
                elif model_name == "combraintf":
                    r = train_combraintf(atlas, train_df, val_df, test_df,
                                          y_mean, y_std, n_epochs=args.n_epochs,
                                          batch_size=min(args.batch_size, 8),
                                          seed=args.seed)
                else:
                    r = {"error": f"unknown model {model_name}"}
            except Exception as e:
                r = {"error": str(e)[:200]}
            r["model"] = model_name
            r["atlas"] = atlas
            r["elapsed_sec"] = round(time.time() - t0, 1)
            results.append(r)
            if "error" in r:
                print(f"  {model_name:12s}: ERROR {r['error']}", flush=True)
            else:
                print(f"  {model_name:12s}: val_z={r['val_mae_z']:.3f} | "
                      f"test_z={r['test_mae_z']:.3f} | test_yr={r['test_mae_yr']:.2f} "
                      f"({r['elapsed_sec']:.0f}s)", flush=True)

    out = ROOT / "skills" / "brain_gnn" / "scripts" / args.output
    out.write_text(json.dumps({"y_mean": y_mean, "y_std": y_std,
                               "n_train": len(train_df), "n_val": len(val_df),
                               "n_test": len(test_df), "results": results},
                              indent=2))
    print(f"\nSaved -> {out}", flush=True)

    # Final table
    print(f"\n{'Atlas':<22} {'Model':<14} {'val_z':>8} {'test_z':>8} {'test_yr':>9}")
    print("-" * 65)
    for r in sorted(results, key=lambda r: (r['atlas'], r['model'])):
        if 'error' in r:
            print(f"{r['atlas']:<22} {r['model']:<14} ERROR")
        else:
            print(f"{r['atlas']:<22} {r['model']:<14} "
                  f"{r['val_mae_z']:>8.3f} {r['test_mae_z']:>8.3f} {r['test_mae_yr']:>9.2f}")


if __name__ == "__main__":
    main()
