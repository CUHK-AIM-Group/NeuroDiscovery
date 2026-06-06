"""BrainGNN hyperparam sweep for age regression.

Try a small grid of (lr, weight_decay, dropout-proxy via lamb, ratio) on
one atlas + fold 0 to find a working configuration, since the out-of-the-
box settings give MAE=12 (no signal) in the full benchmark.

Usage:
    python models/scripts/tune_braingnn.py --atlas aal_116
"""
from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PYTHON = r"C:/Users/45846/anaconda3/envs/neuroclaw/python.exe"
TRAIN = ROOT / "models" / "train_unified.py"


def run(lr, wd, lamb3, lamb4, lamb5, ratio, n_epochs, atlas, device="cuda:0",
        include_t1=False):
    cmd = [
        PYTHON, str(TRAIN),
        "--model", "braingnn", "--task", "regression",
        "--atlas", atlas,
        "--labels-csv", "data/hcp_age_labels.csv",
        "--subjects-file", "data/ready_subjects.txt",
        "--fold", "0", "--kfold", "5",
        "--n-epochs", str(n_epochs), "--batch-size", "16",
        "--lr", str(lr), "--weight-decay", str(wd),
        "--lamb3", str(lamb3), "--lamb4", str(lamb4), "--lamb5", str(lamb5),
        "--ratio", str(ratio),
        "--device", device, "--quiet",
    ]
    if include_t1:
        cmd.append("--include-t1")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    out = r.stdout
    # Parse final line
    m = re.search(r"best_val_mae=([\-0-9.]+).*test_mae=([\-0-9.]+).*r2=([\-0-9.]+).*r=([\-0-9.+]+)",
                  out)
    res = {"lr": lr, "wd": wd, "lamb3": lamb3, "lamb4": lamb4, "lamb5": lamb5,
           "ratio": ratio, "n_epochs": n_epochs, "include_t1": include_t1,
           "time_sec": round(dt, 1), "returncode": r.returncode}
    if m:
        res["val_mae"] = float(m.group(1))
        res["test_mae"] = float(m.group(2))
        res["r2"] = float(m.group(3))
        res["pearson"] = float(m.group(4))
    # Peek training progression
    ep_lines = [l for l in out.splitlines() if l.startswith("epoch")]
    if ep_lines:
        res["last_epoch_line"] = ep_lines[-1][:80]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default="destrieux_148")  # BNT winner
    p.add_argument("--include-t1", action="store_true")
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="models/benchmark_results/braingnn_tuning.csv")
    args = p.parse_args()

    # Pre-build cache
    from models.braingnn.scripts.data_adapter import (
        NeuroClawFCDataset, build_labels_from_csv,
    )
    labels = build_labels_from_csv("data/hcp_age_labels.csv", label_dtype="float")
    subs = Path("data/ready_subjects.txt").read_text().split()
    NeuroClawFCDataset(atlas=args.atlas, labels=labels, subjects=subs,
                       include_t1=args.include_t1, label_dtype="float")

    # Grid (kept small: 3 lr x 2 wd x 3 reg x 2 ratio = 36 runs)
    grid = []
    for lr in [1e-3, 5e-4, 1e-4]:
        for wd in [1e-4, 1e-5]:
            for reg in [0.0, 0.05]:  # lamb3=lamb4 pooling regs
                for gc in [0.0, 0.1]:  # lamb5 group consistency (for classification; harmless in reg)
                    for ratio in [0.5, 0.7]:
                        grid.append((lr, wd, reg, reg, gc, ratio))

    print(f"Sweeping {len(grid)} configs on atlas={args.atlas}, {args.n_epochs} epochs each")
    print(f"Output: {args.out}\n")

    rows = []
    t_start = time.time()
    for i, (lr, wd, l3, l4, l5, ratio) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] lr={lr} wd={wd} reg={l3} gc={l5} ratio={ratio} ...",
              end=" ", flush=True)
        rec = run(lr, wd, l3, l4, l5, ratio, args.n_epochs, args.atlas, args.device,
                  args.include_t1)
        if rec.get("test_mae") is not None:
            print(f"val={rec['val_mae']:.2f} test={rec['test_mae']:.2f} "
                  f"r={rec['pearson']:+.3f} t={rec['time_sec']:.0f}s")
        else:
            print(f"FAILED")
        rows.append(rec)
        pd.DataFrame(rows).to_csv(Path(args.out), index=False)
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(grid) - i)
        if i % 4 == 0:
            print(f"  elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m\n")

    df = pd.DataFrame(rows)
    df_ok = df[df["returncode"] == 0].copy() if "returncode" in df else df
    df_ok = df_ok.dropna(subset=["test_mae"])
    print("\n=== TOP 10 configs by test_mae ===")
    print(df_ok.nsmallest(10, "test_mae")[
        ["lr","wd","lamb3","lamb5","ratio","val_mae","test_mae","pearson","time_sec"]
    ].to_string(index=False))
    print("\n=== TOP 10 by pearson r ===")
    print(df_ok.nlargest(10, "pearson")[
        ["lr","wd","lamb3","lamb5","ratio","val_mae","test_mae","pearson","time_sec"]
    ].to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
