"""Lifespan age hyperparam search for new models (LG-GNN, IBGNN, Com-BrainTF).

Strategy:
  1. Stage A — paper-default + a few sensible perturbations on top atlases.
  2. For each (model, atlas) pair, run 4-6 configs and pick the best by val MAE.
  3. Save results incrementally so we can stop/resume.

Atlas selection rationale:
  - cc200 (190 ROI): top performer for BNT/BrainNetCNN in lifespan baseline
  - basc_122 (122): top BNT on lifespan
  - aal_116 (116): consistent across tasks
  - schaefer_100_7net (100): smaller, tests low-ROI regime

Output: skills/brain_gnn/scripts/lifespan_new_models_search.json (incremental).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from run_lifespan_age import (
    LABEL_CSV, make_split, train_lggnn, train_ibgnn, train_combraintf,
)


# Hyperparam grids (kept small: 4-6 configs per model)
LGGNN_GRID = [
    # (lr, wd, hidden_dim, embed_dim, ratio, dropout, mi_weight)
    (1e-3, 5e-4, 64,  20, 0.5, 0.3, 0.1),    # paper default
    (1e-3, 5e-4, 64,  20, 0.7, 0.3, 0.1),    # higher keep ratio
    (5e-4, 1e-3, 128, 32, 0.5, 0.3, 0.1),    # bigger model, lower lr
    (1e-3, 5e-4, 64,  20, 0.5, 0.5, 0.1),    # higher dropout
    (5e-4, 1e-3, 128, 32, 0.7, 0.5, 0.0),    # no MI loss
    (1e-3, 1e-3, 64,  20, 0.5, 0.3, 0.5),    # heavier MI weight
]

IBGNN_GRID = [
    # (lr, wd, hidden_dim, n_gnn_layers, n_mlp_layers, pooling)
    (1e-4, 1e-5, 128, 2, 1, "mean"),         # paper default
    (1e-4, 1e-5, 256, 2, 1, "mean"),         # bigger hidden
    (5e-4, 5e-4, 128, 3, 1, "mean"),         # 3-layer GNN
    (1e-3, 5e-4, 128, 2, 2, "mean"),         # higher lr + 2 mlp
    (5e-4, 5e-4, 256, 2, 1, "sum"),          # sum pooling
    (1e-3, 5e-4, 64,  2, 1, "mean"),         # smaller hidden, faster
]

COMBRAINTF_GRID = [
    # (lr, wd, hidden_size, nhead, n_clusters, dec_weight)
    (1e-4, 5e-4, 1024, 8, 8, 0.1),           # paper default
    (1e-4, 5e-4,  512, 4, 8, 0.1),           # smaller
    (5e-5, 1e-4, 1024, 8, 8, 0.5),           # lower lr, heavier dec
    (5e-4, 5e-4,  512, 4, 8, 0.1),           # higher lr
    (1e-4, 5e-4, 1024, 8, 4, 0.1),           # fewer clusters
    (1e-4, 5e-4,  256, 4, 8, 0.1),           # tiny model
]


def build_jobs(args):
    grids = {
        "lggnn": LGGNN_GRID,
        "ibgnn": IBGNN_GRID,
        "combraintf": COMBRAINTF_GRID,
    }
    jobs = []
    for atlas in args.atlases:
        for model in args.models:
            grid = grids.get(model)
            if grid is None:
                continue
            for cfg_i, cfg in enumerate(grid):
                jobs.append((atlas, model, cfg_i, cfg))
    return jobs


def run_one(atlas, model, cfg, train_df, val_df, test_df, y_mean, y_std,
             n_epochs, batch_size, seed):
    if model == "lggnn":
        lr, wd, hidden_dim, embed_dim, ratio, dropout, mi_weight = cfg
        return train_lggnn(atlas, train_df, val_df, test_df, y_mean, y_std,
                            n_epochs=n_epochs, batch_size=batch_size,
                            lr=lr, wd=wd, hidden_dim=hidden_dim,
                            embed_dim=embed_dim, ratio=ratio, dropout=dropout,
                            mi_weight=mi_weight, seed=seed)
    if model == "ibgnn":
        lr, wd, hidden_dim, n_gnn, n_mlp, pooling = cfg
        return train_ibgnn(atlas, train_df, val_df, test_df, y_mean, y_std,
                            n_epochs=n_epochs, batch_size=batch_size,
                            lr=lr, wd=wd, hidden_dim=hidden_dim,
                            n_gnn_layers=n_gnn, n_mlp_layers=n_mlp,
                            pooling=pooling, seed=seed)
    if model == "combraintf":
        lr, wd, hidden_size, nhead, n_clusters, dec_weight = cfg
        return train_combraintf(atlas, train_df, val_df, test_df, y_mean, y_std,
                                 n_epochs=n_epochs, batch_size=batch_size,
                                 lr=lr, wd=wd, hidden_size=hidden_size,
                                 nhead=nhead, n_clusters=n_clusters,
                                 dec_weight=dec_weight, seed=seed)
    return {"error": f"unknown {model}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlases", nargs="+",
                   default=["cc200", "basc_122", "aal_116", "schaefer_100_7net"])
    p.add_argument("--models", nargs="+",
                   default=["lggnn", "ibgnn", "combraintf"])
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output",
                   default="skills/brain_gnn/scripts/lifespan_new_models_search.json")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(LABEL_CSV, dtype={"subject_id": str})
    train_df, val_df, test_df = make_split(df, seed=args.seed)
    y_mean = float(train_df["label"].mean())
    y_std = float(train_df["label"].std())
    print(f"Lifespan age | n={len(df)} | train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"y stats: mean={y_mean:.2f} std={y_std:.2f}", flush=True)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text()).get("results", [])
        print(f"Resuming with {len(existing)} prior results", flush=True)
    done_keys = {(r["atlas"], r["model"], r["cfg_idx"]) for r in existing}

    jobs = build_jobs(args)
    pending = [j for j in jobs if (j[0], j[1], j[2]) not in done_keys]
    print(f"Total jobs: {len(jobs)} | pending: {len(pending)}\n", flush=True)

    results = list(existing)
    t_start = time.time()

    for i, (atlas, model, cfg_i, cfg) in enumerate(pending, 1):
        t0 = time.time()
        try:
            r = run_one(atlas, model, cfg, train_df, val_df, test_df, y_mean, y_std,
                        args.n_epochs, args.batch_size, args.seed)
        except Exception as e:
            r = {"error": str(e)[:200]}
        r["atlas"] = atlas
        r["model"] = model
        r["cfg_idx"] = cfg_i
        r["cfg"] = list(cfg)
        r["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(r)

        if "error" in r:
            print(f"[{i}/{len(pending)}] {atlas}/{model}/cfg{cfg_i}: ERROR {r['error']}",
                  flush=True)
        else:
            elapsed_total = (time.time() - t_start) / 60.0
            eta = elapsed_total / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {atlas:25s} {model:11s} cfg{cfg_i}: "
                  f"val_z={r['val_mae_z']:.3f} test_z={r['test_mae_z']:.3f} "
                  f"yr={r['test_mae_yr']:.2f} ({r['elapsed_sec']:.0f}s) | "
                  f"elapsed {elapsed_total:.1f}m ETA {eta:.1f}m",
                  flush=True)

        # incremental save (atomic: write to .tmp then rename)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({
            "y_mean": y_mean, "y_std": y_std,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "results": results,
        }, indent=2))
        tmp_path.replace(out_path)

    print(f"\nSaved {len(results)} results -> {out_path}", flush=True)

    # Top-3 per (atlas, model)
    print(f"\n{'Atlas':<22} {'Model':<11} {'Top val_z':>10} {'cfg':>4} {'test_z':>8} {'test_yr':>8}")
    print("-" * 75)
    rows = [r for r in results if "error" not in r]
    by_key = {}
    for r in rows:
        key = (r["atlas"], r["model"])
        by_key.setdefault(key, []).append(r)
    for (atlas, model), lst in sorted(by_key.items()):
        lst.sort(key=lambda r: r["val_mae_z"])
        b = lst[0]
        print(f"{atlas:<22} {model:<11} {b['val_mae_z']:>10.3f} {b['cfg_idx']:>4} "
              f"{b['test_mae_z']:>8.3f} {b['test_mae_yr']:>8.2f}")


if __name__ == "__main__":
    main()
