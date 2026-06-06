"""Sweep BrainGNN across all atlases, fold 0 only.

Quick comparison to find which atlas gives the best signal. Use this to
decide which atlas to run full 5-fold CV on later.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PYTHON = r"C:/Users/45846/anaconda3/envs/neuroclaw/python.exe"
TRAIN_SCRIPT = ROOT / "models" / "braingnn" / "scripts" / "train.py"


def get_atlases():
    atlas_root = ROOT / "data" / "atlas"
    return sorted([d.name for d in atlas_root.iterdir()
                   if d.is_dir() and (d / "atlas.nii.gz").exists()])


def run_one(atlas: str, include_t1: bool = True, n_epochs: int = 40) -> dict:
    t0 = time.time()
    # Clear atlas-specific PyG cache so each run starts fresh
    cache = ROOT / "data" / "braingnn_cache" / atlas
    if cache.exists():
        import shutil
        shutil.rmtree(cache, ignore_errors=True)

    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "--atlas", atlas,
        "--labels-csv", "data/hcp_gender_labels.csv",
        "--subjects-file", "data/ready_subjects.txt",
        "--fold", "0", "--kfold", "5",
        "--n-epochs", str(n_epochs),
        "--batch-size", "16",
        "--lr", "0.005",
        "--weight-decay", "5e-3",
        "--nclass", "2",
    ]
    if include_t1:
        cmd.append("--include-t1")

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0

    # Parse last line: "[fold 0] best_val_acc=X  test_acc=Y"
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    summary = next((l for l in reversed(lines) if "best_val_acc" in l), "")
    best_val = test_acc = None
    n_samples = n_roi = indim = None
    for l in lines:
        if "n_samples=" in l:
            parts = l.strip().split()
            for p in parts:
                if p.startswith("n_samples="): n_samples = int(p.split("=")[1])
                if p.startswith("n_roi="):     n_roi = int(p.split("=")[1])
                if p.startswith("indim="):     indim = int(p.split("=")[1])
        if "best_val_acc=" in l:
            import re
            m = re.search(r"best_val_acc=([\d.]+)\s+test_acc=([\d.]+)", l)
            if m:
                best_val = float(m.group(1))
                test_acc = float(m.group(2))

    # Last 3 train epochs for a quick overfitting sanity check
    tr_final = val_final = None
    for l in reversed(lines):
        if l.startswith("epoch"):
            parts = l.split("|")
            try:
                tr_final = float(parts[2].split()[1])
                val_final = float(parts[3].split()[1])
            except Exception:
                pass
            break

    return {
        "atlas": atlas,
        "n_samples": n_samples, "n_roi": n_roi, "indim": indim,
        "best_val_acc": best_val, "test_acc": test_acc,
        "final_tr_acc": tr_final, "final_val_acc": val_final,
        "time_sec": round(dt, 1),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-3:]) if result.returncode else "",
        "returncode": result.returncode,
    }


def main():
    atlases = get_atlases()
    print(f"Sweeping {len(atlases)} atlases, 130 subjects, fold 0/5, 40 epochs each\n")

    rows = []
    for i, a in enumerate(atlases, 1):
        print(f"[{i}/{len(atlases)}] {a} ...", flush=True)
        r = run_one(a)
        print(f"   -> val={r['best_val_acc']} test={r['test_acc']}  "
              f"tr_final={r['final_tr_acc']} val_final={r['final_val_acc']}  "
              f"time={r['time_sec']}s  indim={r['indim']}", flush=True)
        rows.append(r)
        # Save incremental results
        pd.DataFrame(rows).to_csv(
            ROOT / "models" / "braingnn" / "atlas_sweep_results.csv", index=False
        )

    print("\n" + "=" * 70)
    print("ATLAS SWEEP SUMMARY (fold 0/5, 130 subjects, T1 fused)")
    print("=" * 70)
    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    print(df[["atlas", "n_roi", "indim", "best_val_acc", "test_acc",
              "final_tr_acc", "final_val_acc", "time_sec"]].to_string(index=False))

    out_csv = ROOT / "models" / "braingnn" / "atlas_sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
