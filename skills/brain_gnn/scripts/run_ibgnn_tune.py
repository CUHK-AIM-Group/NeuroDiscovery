"""IBGNN debug + retrain on lifespan age.

Diagnosis of prior divergence: IBGNN's IBGConv has no LayerNorm/BatchNorm,
and our node features are FC rows (Fisher-z, range ~[-7,7]) of dimension n_roi.
Even with grad_clip + low lr, the first batch's predictions explode because
input scale × untouched MPConv stack produces huge logits, MSE loss explodes,
gradients blow up, and the model never recovers.

Key fixes:
1. **Input z-score per node (per-subject)** — node features standardized to
   mean=0, std=1 per ROI. Brings input scale to model-friendly range.
2. **Wrapper LayerNorm** — `IBGNNStable` adds LN to GNN output before MLP.
3. Gradient clipping (clip_grad_norm_=1.0).
4. NaN/Inf guard with early-abort if val_z > 5 mid-training.
5. Conservative LR grid + mean pooling + 2 GNN layers only.

Output: skills/brain_gnn/scripts/ibgnn_tuned_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from run_lifespan_age import (
    LABEL_CSV, make_split, PyGLifespanDataset, DEVICE,
)
from models.ibgnn.net.ibgnn import IBGNN
import torch.nn as nn


class IBGNNStable(nn.Module):
    """IBGNN wrapped with input z-score + post-GNN LayerNorm.

    Fixes the divergence problem on lifespan age (z-scored target) by
    normalizing both inputs and intermediate representations.
    """
    def __init__(self, n_roi, hidden_dim, n_gnn_layers, normalize_input=True):
        super().__init__()
        self.normalize_input = normalize_input
        self.input_norm = nn.LayerNorm(n_roi) if normalize_input else nn.Identity()
        self.ibgnn = IBGNN(n_roi=n_roi, nclass=1, hidden_dim=hidden_dim,
                            n_gnn_layers=n_gnn_layers, n_mlp_layers=1,
                            pooling="mean", task="regression")
        # Patch: replace MLP input with LN-wrapped to stabilize after GNN pool
        # by hooking forward; instead, simpler: insert LN after GNN encoder.
        self.gnn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.input_norm(x)
        # Manually call gnn encoder + apply LN, then MLP head
        g = self.ibgnn.gnn(x, edge_index, edge_attr, batch)
        g = self.gnn_norm(g)
        return self.ibgnn.mlp(g)


# Conservative grid: only stable region, vary regularization knobs
GRID = [
    # (lr, wd, hidden_dim, n_gnn_layers, normalize_input, grad_clip)
    (1e-4, 1e-5, 128, 2, True,  1.0),  # paper default + LN + clip
    (5e-4, 1e-5, 128, 2, True,  1.0),  # higher lr (now safe with LN)
    (1e-3, 5e-4, 128, 2, True,  1.0),  # aggressive lr
    (1e-4, 1e-5,  64, 2, True,  1.0),  # smaller hidden
    (1e-4, 1e-5, 256, 2, True,  1.0),  # larger hidden
    (5e-4, 5e-4, 128, 3, True,  1.0),  # 3 layers (now safe)
    (1e-4, 1e-5, 128, 2, False, 1.0),  # no input norm (control)
    (5e-4, 1e-5, 128, 2, True,  0.5),  # higher lr + tighter clip
]


def train_ibgnn_safe(atlas, train_df, val_df, test_df, y_mean, y_std,
                     n_epochs, batch_size, lr, wd, hidden_dim, n_gnn_layers,
                     normalize_input, grad_clip, seed=42):
    """Train IBGNN with grad clip + NaN guard + early-abort on divergence."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = PyGLifespanDataset(atlas, train_df, y_mean, y_std)
    val_ds = PyGLifespanDataset(atlas, val_df, y_mean, y_std)
    test_ds = PyGLifespanDataset(atlas, test_df, y_mean, y_std)
    if not train_ds.data_list:
        return {"error": "empty train"}
    indim = int(train_ds.data_list[0].x.size(1))
    train_loader = PyGLoader(train_ds.data_list, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = PyGLoader(val_ds.data_list, batch_size=batch_size)
    test_loader = PyGLoader(test_ds.data_list, batch_size=batch_size)

    model = IBGNNStable(n_roi=indim, hidden_dim=hidden_dim,
                         n_gnn_layers=n_gnn_layers,
                         normalize_input=normalize_input).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = float("inf"); best_state = None
    diverged = False
    epochs_run = 0

    for epoch in range(n_epochs):
        epochs_run = epoch + 1
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = F.mse_loss(logits.squeeze(-1), batch.y.float())
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if diverged:
            break
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                preds.append(out.squeeze(-1).cpu()); trues.append(batch.y.cpu())
        val_mae_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if not np.isfinite(val_mae_z) or val_mae_z > 5.0:
            # Diverged: bail out early
            diverged = True
            break
        if val_mae_z < best_val:
            best_val = val_mae_z
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if diverged or best_state is None:
        return {"error": "diverged",
                "epochs_run": epochs_run,
                "best_val": best_val if np.isfinite(best_val) else None}

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
            "test_mae_yr": test_mae_z * y_std,
            "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
            "epochs_run": epochs_run}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlases", nargs="+",
                   default=["cc200", "basc_122", "aal_116"])
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output",
                   default="skills/brain_gnn/scripts/ibgnn_tuned_results.json")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(LABEL_CSV, dtype={"subject_id": str})
    train_df, val_df, test_df = make_split(df, seed=args.seed)
    y_mean = float(train_df["label"].mean())
    y_std = float(train_df["label"].std())
    print(f"Lifespan age | n={len(df)} | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"y stats: mean={y_mean:.2f} std={y_std:.2f}", flush=True)
    print(f"Grad clip + NaN guard + conservative grid", flush=True)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text()).get("results", [])
            print(f"Resuming with {len(existing)} prior results", flush=True)
        except json.JSONDecodeError:
            existing = []
    done_keys = {(r["atlas"], r["cfg_idx"]) for r in existing}

    jobs = []
    for atlas in args.atlases:
        for cfg_i, cfg in enumerate(GRID):
            if (atlas, cfg_i) not in done_keys:
                jobs.append((atlas, cfg_i, cfg))

    total = len(args.atlases) * len(GRID)
    print(f"Total jobs: {total} | pending: {len(jobs)}\n", flush=True)

    results = list(existing)
    t_start = time.time()

    for i, (atlas, cfg_i, cfg) in enumerate(jobs, 1):
        lr, wd, hidden_dim, n_gnn, normalize_input, grad_clip = cfg
        t0 = time.time()
        try:
            r = train_ibgnn_safe(atlas, train_df, val_df, test_df, y_mean, y_std,
                                  n_epochs=args.n_epochs, batch_size=args.batch_size,
                                  lr=lr, wd=wd, hidden_dim=hidden_dim,
                                  n_gnn_layers=n_gnn, normalize_input=normalize_input,
                                  grad_clip=grad_clip, seed=args.seed)
        except Exception as e:
            r = {"error": str(e)[:200]}
        r["atlas"] = atlas
        r["cfg_idx"] = cfg_i
        r["cfg"] = list(cfg)
        r["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(r)

        if "error" in r:
            ep = r.get("epochs_run", "?")
            bv = r.get("best_val")
            bv_s = f"{bv:.3f}" if bv is not None else "n/a"
            print(f"[{i}/{len(jobs)}] {atlas:18s} cfg{cfg_i}: {r['error']:12s} "
                  f"epoch={ep} best_val={bv_s} ({r['elapsed_sec']:.0f}s)",
                  flush=True)
        else:
            elapsed = (time.time() - t_start) / 60.0
            eta = elapsed / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {atlas:18s} cfg{cfg_i}: "
                  f"val_z={r['val_mae_z']:.3f} test_z={r['test_mae_z']:.3f} "
                  f"yr={r['test_mae_yr']:.2f} ({r['elapsed_sec']:.0f}s) | "
                  f"elapsed {elapsed:.1f}m ETA {eta:.1f}m",
                  flush=True)

        # atomic save
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "y_mean": y_mean, "y_std": y_std,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "results": results,
        }, indent=2))
        tmp.replace(out_path)

    print(f"\nSaved {len(results)} results -> {out_path}", flush=True)

    # Summary
    print(f"\n{'Atlas':<18}{'cfg':>4}{'val_z':>9}{'test_z':>9}{'test_yr':>10}{'epochs':>8}")
    print("-" * 60)
    rows = [r for r in results if "error" not in r]
    rows.sort(key=lambda r: r["val_mae_z"])
    for r in rows[:15]:
        print(f"{r['atlas']:<18}{r['cfg_idx']:>4}{r['val_mae_z']:>9.3f}"
              f"{r['test_mae_z']:>9.3f}{r['test_mae_yr']:>10.2f}"
              f"{r.get('epochs_run', 0):>8}")
    n_div = sum(1 for r in results if "error" in r)
    n_ok = len(rows)
    print(f"\n{n_ok} converged | {n_div} diverged out of {len(results)} total")


if __name__ == "__main__":
    main()
