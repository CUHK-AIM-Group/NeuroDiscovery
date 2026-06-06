"""KG-guided vs Vanilla experiment.

Compares three conditions on HCP-YA age regression:
  1. Vanilla: full FC matrix (AAL-116, 116 ROIs)
  2. KG-guided: only ROIs referenced by KG hypotheses (~48 ROIs)
  3. Random-subset: same number of ROIs as KG-guided, randomly selected (N trials)

Models: BrainGNN + BNT
Evaluation: 5-fold CV, reports MAE / R^2 / Pearson r

Usage:
    python models/scripts/kg_guided_experiment.py \
        --task regression \
        --labels-csv data/hcp_age_labels.csv \
        --subjects-file data/ready_subjects.txt \
        --atlas aal_116 \
        --hypotheses neurooracle/data/quick/hypotheses_imaging_hcp.json \
        --models braingnn bnt \
        --kfold 5 --n-epochs 50 --n-random-trials 5
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PYTHON = r"C:/Users/45846/anaconda3/envs/neuroclaw/python.exe"
TRAIN_SCRIPT = ROOT / "models" / "train_unified.py"

from models.scripts.hypothesis_roi_mapper import build_kg_roi_mask, get_roi_names


def run_condition(model: str, atlas: str, fold: int, task: str,
                  labels_csv: str, subjects_file: str, n_epochs: int,
                  batch_size: int, seed: int, device: str,
                  roi_mask: list[bool] | None, condition_name: str,
                  log_dir: Path) -> dict:
    """Run a single (model, fold, condition) training job."""
    import json
    import tempfile

    mask_file = None
    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "--model", model,
        "--task", task,
        "--atlas", atlas,
        "--labels-csv", labels_csv,
        "--subjects-file", subjects_file,
        "--fold", str(fold),
        "--kfold", "5",
        "--n-epochs", str(n_epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--device", device,
        "--quiet",
    ]
    if model == "bnt":
        cmd += ["--lr", "1e-4", "--weight-decay", "1e-4"]
    else:
        cmd += ["--lr", "0.005", "--weight-decay", "5e-3"]

    if roi_mask is not None:
        mask_file = log_dir / f"mask_{condition_name}_{model}_f{fold}.json"
        mask_file.write_text(json.dumps(roi_mask))
        cmd += ["--roi-mask-file", str(mask_file)]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0

    log_path = log_dir / f"{condition_name}_{model}_fold{fold}.log"
    log_path.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)

    rec = {
        "condition": condition_name,
        "model": model,
        "atlas": atlas,
        "fold": fold,
        "n_roi": sum(roi_mask) if roi_mask else 116,
        "time_sec": round(dt, 1),
        "returncode": result.returncode,
    }

    for line in reversed(result.stdout.splitlines()):
        if "test_mae=" in line:
            import re
            m = re.search(r"test_mae=([\-0-9.]+).*r2=([\-0-9.]+).*r=([\-0-9.+]+)", line)
            if m:
                rec["test_mae"] = float(m.group(1))
                rec["test_r2"] = float(m.group(2))
                rec["test_r"] = float(m.group(3))
            break
        if "test_acc=" in line:
            import re
            m = re.search(r"test_acc=([\-0-9.]+)", line)
            if m:
                rec["test_acc"] = float(m.group(1))
            break

    if result.returncode != 0:
        rec["error"] = "\n".join(result.stderr.splitlines()[-3:])
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="regression")
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--subjects-file", required=True)
    p.add_argument("--atlas", default="aal_116")
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--models", nargs="+", default=["braingnn", "bnt"])
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-random-trials", type=int, default=5)
    p.add_argument("--out-dir", default="models/experiment_results/kg_guided")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    kg_mask = build_kg_roi_mask(ROOT / args.hypotheses, args.atlas)
    n_kg_rois = sum(kg_mask)
    n_total_rois = len(kg_mask)
    roi_names = get_roi_names(args.atlas)

    print(f"KG-guided mask: {n_kg_rois}/{n_total_rois} ROIs")
    print(f"Models: {args.models}")
    print(f"Conditions: vanilla, kg_guided, random x{args.n_random_trials}")
    print(f"Total jobs: {len(args.models)} x {args.kfold} x (1 + 1 + {args.n_random_trials})")
    print()

    # Generate random masks
    rng = np.random.default_rng(args.seed)
    random_masks = []
    for trial in range(args.n_random_trials):
        indices = rng.choice(n_total_rois, size=n_kg_rois, replace=False)
        mask = [False] * n_total_rois
        for idx in indices:
            mask[idx] = True
        random_masks.append(mask)

    # Clear PyG cache for this atlas to avoid stale cached data
    cache_dir = ROOT / "data" / "braingnn_cache" / args.atlas
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    rows = []
    total_jobs = len(args.models) * args.kfold * (2 + args.n_random_trials)
    done = 0
    t_start = time.time()

    for model in args.models:
        for fold in range(args.kfold):
            # Vanilla (full graph)
            done += 1
            print(f"[{done}/{total_jobs}] {model}/vanilla/fold{fold} ...", end=" ", flush=True)
            rec = run_condition(model, args.atlas, fold, args.task,
                               args.labels_csv, args.subjects_file,
                               args.n_epochs, args.batch_size, args.seed,
                               args.device, None, "vanilla", log_dir)
            print(f"MAE={rec.get('test_mae', '?'):.2f} r={rec.get('test_r', '?')}" if rec.get("returncode") == 0 else f"FAILED", flush=True)
            rows.append(rec)

            # KG-guided
            done += 1
            print(f"[{done}/{total_jobs}] {model}/kg_guided/fold{fold} ...", end=" ", flush=True)
            rec = run_condition(model, args.atlas, fold, args.task,
                               args.labels_csv, args.subjects_file,
                               args.n_epochs, args.batch_size, args.seed,
                               args.device, kg_mask, "kg_guided", log_dir)
            print(f"MAE={rec.get('test_mae', '?'):.2f} r={rec.get('test_r', '?')}" if rec.get("returncode") == 0 else f"FAILED", flush=True)
            rows.append(rec)

            # Random subsets
            for trial, rmask in enumerate(random_masks):
                done += 1
                cond_name = f"random_{trial}"
                print(f"[{done}/{total_jobs}] {model}/{cond_name}/fold{fold} ...", end=" ", flush=True)
                rec = run_condition(model, args.atlas, fold, args.task,
                                   args.labels_csv, args.subjects_file,
                                   args.n_epochs, args.batch_size, args.seed,
                                   args.device, rmask, cond_name, log_dir)
                print(f"MAE={rec.get('test_mae', '?'):.2f} r={rec.get('test_r', '?')}" if rec.get("returncode") == 0 else f"FAILED", flush=True)
                rows.append(rec)

            # Save intermediate results
            pd.DataFrame(rows).to_csv(out_dir / "results_raw.csv", index=False)

    total_time = time.time() - t_start
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_raw.csv", index=False)

    # Aggregate results
    print(f"\n{'='*70}")
    print(f"RESULTS (total time: {total_time/60:.1f} min)")
    print(f"{'='*70}\n")

    ok = df[df["returncode"] == 0].copy()
    if args.task == "regression" and "test_mae" in ok.columns:
        # Group random trials together
        ok["condition_group"] = ok["condition"].apply(
            lambda c: "random" if c.startswith("random_") else c)
        summary = (ok.groupby(["model", "condition_group"])
                   .agg(mae_mean=("test_mae", "mean"),
                        mae_std=("test_mae", "std"),
                        r_mean=("test_r", "mean"),
                        r_std=("test_r", "std"),
                        r2_mean=("test_r2", "mean"),
                        n_runs=("fold", "count"))
                   .reset_index()
                   .sort_values(["model", "mae_mean"]))
        print(summary.to_string(index=False))
        summary.to_csv(out_dir / "summary.csv", index=False)

        # Statistical test: KG-guided vs Random (paired by fold)
        print(f"\n--- Statistical Comparison ---")
        from scipy import stats
        for model in args.models:
            kg_maes = ok[(ok["model"] == model) & (ok["condition"] == "kg_guided")]["test_mae"].values
            rand_maes = ok[(ok["model"] == model) & (ok["condition_group"] == "random")]["test_mae"].values
            if len(kg_maes) > 1 and len(rand_maes) > 1:
                t_stat, p_val = stats.ttest_ind(kg_maes, rand_maes)
                print(f"  {model}: KG-guided MAE={kg_maes.mean():.2f} vs Random MAE={rand_maes.mean():.2f} "
                      f"(t={t_stat:.2f}, p={p_val:.4f})")
    else:
        print(ok.groupby(["model", "condition"]).agg(
            acc_mean=("test_acc", "mean"), n=("fold", "count")).to_string())

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
