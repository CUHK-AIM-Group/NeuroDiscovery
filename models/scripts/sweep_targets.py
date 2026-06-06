"""Multi-target cognitive prediction sweep on HCP-YA.

Runs BNT (and optionally BrainGNN/Ridge) across many HCP cognitive/behavioral
targets to find where rsfMRI connectivity has real signal.

Usage:
    python models/scripts/sweep_targets.py \
        --targets cogfluidcomp cogcrystal picvocab readeng listsort \
                  procspeed flanker cardsort \
                  neuroticism extraversion openness \
                  tom emotion_acc neg_affect pos_affect \
        --models bnt \
        --atlas destrieux_148 \
        --kfold 5 --n-epochs 50
"""
from __future__ import annotations

import argparse
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


def run_one(model: str, target: str, fold: int, args_dict: dict, device: str):
    class _A: pass
    args = _A()
    for k, v in args_dict.items(): setattr(args, k, v)
    labels_csv = f"data/hcp_{target}_labels.csv"

    if model == "ridge":
        cmd = [PYTHON, str(RIDGE_SCRIPT),
               "--atlas", args.atlas, "--labels-csv", labels_csv,
               "--subjects-file", args.subjects_file,
               "--task", "regression",
               "--fold", str(fold), "--kfold", str(args.kfold),
               "--seed", str(args.seed), "--quiet"]
    else:
        cmd = [PYTHON, str(TRAIN_SCRIPT),
               "--model", model, "--task", "regression",
               "--atlas", args.atlas, "--labels-csv", labels_csv,
               "--subjects-file", args.subjects_file,
               "--fold", str(fold), "--kfold", str(args.kfold),
               "--n-epochs", str(args.n_epochs),
               "--batch-size", str(args.batch_size),
               "--seed", str(args.seed),
               "--device", device, "--quiet"]
        if model == "bnt":
            cmd += ["--lr", "1e-4", "--weight-decay", "1e-4"]
        else:
            cmd += ["--lr", "1e-3", "--weight-decay", "1e-4",
                    "--lamb3", "0", "--lamb4", "0", "--lamb5", "0.1"]
    if args.include_t1:
        cmd.append("--include-t1")

    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    out = r.stdout
    m = re.search(r"best_val_mae=([\-0-9.]+).*test_mae=([\-0-9.]+).*r2=([\-0-9.]+).*r=([\-0-9.+]+)", out)
    rec = {"model": model, "target": target, "fold": fold,
           "time_sec": round(dt, 1), "returncode": r.returncode}
    if m:
        rec["val_mae"] = float(m.group(1))
        rec["test_mae"] = float(m.group(2))
        rec["test_r2"] = float(m.group(3))
        rec["test_r"] = float(m.group(4))
    else:
        rec["error"] = "\n".join(r.stderr.splitlines()[-3:])
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["bnt"])
    p.add_argument("--atlas", default="destrieux_148")
    p.add_argument("--subjects-file", default="data/ready_subjects.txt")
    p.add_argument("--include-t1", action="store_true")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--devices", nargs="+", default=["cuda:0","cuda:0","cuda:0","cuda:0"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.out is None:
        args.out = f"models/benchmark_results/sweep_targets_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_csv = ROOT / args.out

    # Pre-build cache for BrainGNN if requested
    if "braingnn" in args.models:
        print(f"Pre-building BrainGNN cache for {args.atlas} ...")
        sys.path.insert(0, str(ROOT))
        import shutil
        from models.braingnn.scripts.data_adapter import (
            NeuroClawFCDataset, build_labels_from_csv,
        )
        cache = ROOT / "data" / "braingnn_cache" / args.atlas
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        labels = build_labels_from_csv(f"data/hcp_{args.targets[0]}_labels.csv",
                                        label_dtype="float")
        subs = Path(args.subjects_file).read_text().split()
        NeuroClawFCDataset(atlas=args.atlas, labels=labels, include_t1=args.include_t1,
                           subjects=subs, label_dtype="float")
        print("  cached.\n")

    jobs = []
    for m in args.models:
        for t in args.targets:
            for f in range(args.kfold):
                jobs.append((m, t, f))
    total = len(jobs)
    print(f"Running {len(args.models)} models x {len(args.targets)} targets x "
          f"{args.kfold} folds = {total} jobs")
    print(f"Atlas: {args.atlas}, workers={args.workers}")
    print(f"Output -> {out_csv}\n")

    rows = []
    args_dict = vars(args)
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut2job = {ex.submit(run_one, m, t, f, args_dict, args.devices[i % len(args.devices)]): (m,t,f)
                   for i, (m,t,f) in enumerate(jobs)}
        done = 0
        for fut in as_completed(fut2job):
            done += 1
            m, t, f = fut2job[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"model": m, "target": t, "fold": f, "returncode": -1, "error": str(e)[:200]}
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done)
            if rec.get("returncode") == 0:
                print(f"[{done}/{total}] {m}/{t}/f{f} val={rec.get('val_mae','?'):.2f} "
                      f"test={rec.get('test_mae','?'):.2f} r={rec.get('test_r','?'):+.3f} "
                      f"t={rec.get('time_sec',0):.0f}s | ETA {eta/60:.1f}m", flush=True)
            else:
                print(f"[{done}/{total}] {m}/{t}/f{f} FAILED", flush=True)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    done = df[df["returncode"] == 0].dropna(subset=["test_mae"])
    print("\n" + "=" * 70)
    print("TARGET SWEEP LEADERBOARD")
    print("=" * 70)
    agg = (done.groupby(["model", "target"])
           .agg(mae=("test_mae","mean"), mae_std=("test_mae","std"),
                r=("test_r","mean"), r_std=("test_r","std"),
                r2=("test_r2","mean"), n=("fold","count"))
           .reset_index().sort_values("r", ascending=False))
    print(agg.to_string(index=False))
    lb_csv = out_csv.with_name(out_csv.stem.replace("sweep_targets", "leaderboard_targets") + ".csv")
    agg.to_csv(lb_csv, index=False)
    print(f"\nLeaderboard -> {lb_csv}")


if __name__ == "__main__":
    main()
