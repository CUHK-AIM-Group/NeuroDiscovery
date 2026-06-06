"""Unified benchmark sweep: models x atlases x folds.

Runs the full combinatorial grid for HCP age regression (or gender
classification), aggregates per-fold test metrics into a single CSV, and
prints a leaderboard.

Usage (age regression, all 10 atlases, both models, 5 folds):
    python models/scripts/run_benchmark.py \
        --task regression \
        --labels-csv data/hcp_age_labels.csv \
        --subjects-file data/ready_subjects.txt \
        --models braingnn bnt --atlases all \
        --n-epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PYTHON = r"C:/Users/45846/anaconda3/envs/neuroclaw/python.exe"
TRAIN_SCRIPT = ROOT / "models" / "train_unified.py"
RIDGE_SCRIPT = ROOT / "models" / "ridge" / "train.py"


def get_atlases():
    atlas_root = ROOT / "data" / "atlas"
    return sorted([d.name for d in atlas_root.iterdir()
                   if d.is_dir() and (d / "atlas.nii.gz").exists()])


def parse_final_line(stdout: str, task: str) -> dict:
    """Parse the '[model fold N] ...' line at end of train script output."""
    line = ""
    for l in reversed(stdout.splitlines()):
        if "test_mae=" in l or "test_acc=" in l:
            line = l
            break
    out = {}
    if task == "regression":
        m = re.search(r"best_val_mae=([\-0-9.]+).*test_mae=([\-0-9.]+).*r2=([\-0-9.]+).*r=([\-0-9.+]+)", line)
        if m:
            out["best_val_mae"] = float(m.group(1))
            out["test_mae"] = float(m.group(2))
            out["test_r2"] = float(m.group(3))
            out["test_r"] = float(m.group(4))
    else:
        m = re.search(r"best_val_acc=([\-0-9.]+)\s+test_acc=([\-0-9.]+)", line)
        if m:
            out["best_val_acc"] = float(m.group(1))
            out["test_acc"] = float(m.group(2))
    return out


def run_one(model: str, atlas: str, fold: int, args_dict: dict, log_dir: Path,
            device: str = "cpu") -> dict:
    # Re-hydrate args from dict (needed when called via ProcessPoolExecutor)
    class _Args: pass
    args = _Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # NOTE: We DO NOT clear BrainGNN PyG cache here. Multiple workers running
    # different (model, atlas, fold) combinations could race on the same atlas
    # cache dir. Instead, the user should remove data/braingnn_cache once
    # before starting the benchmark. Each atlas is cached under a subdir and
    # shared across folds (same PyG processed file for the same dataset),
    # which is actually what we want.

    if model == "ridge":
        cmd = [
            PYTHON, str(RIDGE_SCRIPT),
            "--atlas", atlas,
            "--labels-csv", args.labels_csv,
            "--subjects-file", args.subjects_file,
            "--task", args.task,
            "--fold", str(fold),
            "--kfold", str(args.kfold),
            "--seed", str(args.seed),
            "--quiet",
        ]
        if args.task == "classification":
            cmd += ["--nclass", str(args.nclass)]
        if args.include_t1:
            cmd.append("--include-t1")
    else:
        cmd = [
            PYTHON, str(TRAIN_SCRIPT),
            "--model", model,
            "--task", args.task,
            "--atlas", atlas,
            "--labels-csv", args.labels_csv,
            "--subjects-file", args.subjects_file,
            "--fold", str(fold),
            "--kfold", str(args.kfold),
            "--n-epochs", str(args.n_epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--device", device,
            "--quiet",
        ]
        if args.task == "classification":
            cmd += ["--nclass", str(args.nclass)]
        if args.include_t1:
            cmd.append("--include-t1")
        # Model-specific lr
        if model == "bnt":
            cmd += ["--lr", str(args.bnt_lr), "--weight-decay", str(args.bnt_wd)]
        else:
            cmd += ["--lr", str(args.braingnn_lr),
                    "--weight-decay", str(args.braingnn_wd),
                    "--lamb3", str(args.braingnn_lamb3),
                    "--lamb4", str(args.braingnn_lamb4),
                    "--lamb5", str(args.braingnn_lamb5)]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0

    # Save the per-run log
    log_path = log_dir / f"{model}_{atlas}_fold{fold}.log"
    log_path.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)

    rec = {
        "model": model, "atlas": atlas, "fold": fold,
        "time_sec": round(dt, 1),
        "returncode": result.returncode,
    }
    rec.update(parse_final_line(result.stdout, args.task))
    if result.returncode != 0:
        rec["error"] = "\n".join(result.stderr.splitlines()[-5:])
    return rec


def leaderboard(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "regression":
        agg = (df[df["returncode"] == 0]
               .groupby(["model", "atlas"])
               .agg(mae_mean=("test_mae", "mean"),
                    mae_std=("test_mae", "std"),
                    r_mean=("test_r", "mean"),
                    r_std=("test_r", "std"),
                    r2_mean=("test_r2", "mean"),
                    n_folds=("fold", "count"))
               .reset_index()
               .sort_values("mae_mean"))
    else:
        agg = (df[df["returncode"] == 0]
               .groupby(["model", "atlas"])
               .agg(acc_mean=("test_acc", "mean"),
                    acc_std=("test_acc", "std"),
                    n_folds=("fold", "count"))
               .reset_index()
               .sort_values("acc_mean", ascending=False))
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--subjects-file", required=True)
    p.add_argument("--models", nargs="+", default=["braingnn", "bnt"])
    p.add_argument("--atlases", nargs="+", default=["all"])
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--include-t1", action="store_true")
    p.add_argument("--nclass", type=int, default=2)

    # Model-specific
    p.add_argument("--braingnn-lr", type=float, default=0.005)
    p.add_argument("--braingnn-wd", type=float, default=5e-3)
    p.add_argument("--braingnn-lamb3", type=float, default=0.1)
    p.add_argument("--braingnn-lamb4", type=float, default=0.1)
    p.add_argument("--braingnn-lamb5", type=float, default=0.1)
    p.add_argument("--bnt-lr", type=float, default=1e-4)
    p.add_argument("--bnt-wd", type=float, default=1e-4)

    p.add_argument("--out-dir", default="models/benchmark_results")
    p.add_argument("--workers", type=int, default=1,
                   help="number of parallel worker processes (each runs on one device)")
    p.add_argument("--devices", nargs="+", default=["cpu"],
                   help="device pool, e.g. --devices cuda:0 cuda:0 cuda:0 cpu "
                        "(workers round-robin through this list)")
    args = p.parse_args()

    atlases = get_atlases() if "all" in args.atlases else args.atlases
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Pre-warm BrainGNN PyG cache serially (one atlas at a time) to avoid
    # multi-worker write races on the same cache file. BNT has no cache.
    if "braingnn" in args.models:
        print("Pre-building BrainGNN PyG cache for each atlas (serial) ...")
        import shutil, importlib
        sys.path.insert(0, str(ROOT))
        from models.braingnn.scripts.data_adapter import (
            NeuroClawFCDataset, build_labels_from_csv,
        )
        label_dtype = "float" if args.task == "regression" else "long"
        labels = build_labels_from_csv(args.labels_csv, label_dtype=label_dtype)
        subs = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]
        for atlas in atlases:
            cache = ROOT / "data" / "braingnn_cache" / atlas
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
            NeuroClawFCDataset(
                atlas=atlas, labels=labels, include_t1=args.include_t1,
                subjects=subs, label_dtype=label_dtype,
            )
            print(f"  cached {atlas}")
        print()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"benchmark_{args.task}_{timestamp}.csv"
    print(f"Running {len(args.models)} models x {len(atlases)} atlases x "
          f"{args.kfold} folds = {len(args.models)*len(atlases)*args.kfold} jobs")
    print(f"Results -> {out_csv}")
    print(f"Logs    -> {log_dir}\n")

    rows = []
    total = len(args.models) * len(atlases) * args.kfold
    done = 0
    t_start = time.time()

    # Build job list + device round-robin assignment
    jobs = []
    for model in args.models:
        for atlas in atlases:
            for fold in range(args.kfold):
                jobs.append((model, atlas, fold))
    args_dict = vars(args)

    if args.workers <= 1:
        for i, (model, atlas, fold) in enumerate(jobs):
            done += 1
            device = args.devices[i % len(args.devices)]
            print(f"[{done}/{total}] {model} / {atlas} / fold {fold} ({device}) ...",
                  end=" ", flush=True)
            rec = run_one(model, atlas, fold, args_dict, log_dir, device=device)
            if rec.get("returncode") == 0:
                if args.task == "regression":
                    msg = (f"val_mae={rec.get('best_val_mae','?'):.2f} "
                           f"test_mae={rec.get('test_mae','?'):.2f} "
                           f"r={rec.get('test_r','?'):+.3f} "
                           f"t={rec['time_sec']:.0f}s")
                else:
                    msg = (f"val_acc={rec.get('best_val_acc','?'):.3f} "
                           f"test_acc={rec.get('test_acc','?'):.3f} "
                           f"t={rec['time_sec']:.0f}s")
                print(msg, flush=True)
            else:
                print(f"FAILED ({rec.get('error','?')[:80]})", flush=True)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        # Parallel: assign device per job via round-robin, submit all to pool
        print(f"Parallel mode: {args.workers} workers, devices={args.devices}\n")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            future_map = {}
            for i, (model, atlas, fold) in enumerate(jobs):
                device = args.devices[i % len(args.devices)]
                fut = ex.submit(run_one, model, atlas, fold, args_dict, log_dir, device)
                future_map[fut] = (model, atlas, fold, device)

            for fut in as_completed(future_map):
                done += 1
                model, atlas, fold, device = future_map[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"model": model, "atlas": atlas, "fold": fold,
                           "returncode": -1, "error": str(e)[:200]}
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done)
                if rec.get("returncode") == 0:
                    if args.task == "regression":
                        msg = (f"val_mae={rec.get('best_val_mae','?'):.2f} "
                               f"test_mae={rec.get('test_mae','?'):.2f} "
                               f"r={rec.get('test_r','?'):+.3f}")
                    else:
                        msg = (f"val_acc={rec.get('best_val_acc','?'):.3f} "
                               f"test_acc={rec.get('test_acc','?'):.3f}")
                    print(f"[{done}/{total}] {model}/{atlas}/f{fold} ({device}) "
                          f"{msg} t={rec.get('time_sec',0):.0f}s | "
                          f"ETA {eta/60:.1f}m", flush=True)
                else:
                    print(f"[{done}/{total}] {model}/{atlas}/f{fold} ({device}) "
                          f"FAILED: {rec.get('error','?')[:100]}", flush=True)
                rows.append(rec)
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print(f"LEADERBOARD ({args.task}, kfold={args.kfold})")
    print("=" * 78)
    lb = leaderboard(df, args.task)
    print(lb.to_string(index=False))
    lb_csv = out_dir / f"leaderboard_{args.task}_{timestamp}.csv"
    lb.to_csv(lb_csv, index=False)
    print(f"\nSaved leaderboard -> {lb_csv}")


if __name__ == "__main__":
    main()
