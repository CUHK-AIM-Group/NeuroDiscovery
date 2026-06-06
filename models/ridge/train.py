"""Ridge regression baseline for FC-based prediction.

Uses the upper-triangle of the Pearson correlation matrix as a feature
vector per subject, applies PCA (for dimensionality) + RidgeCV. This is
the Finn 2015-style baseline used to sanity-check whether a deep model
(BrainGNN, BNT) adds value over a well-tuned linear model.

Output format matches the unified benchmark (prints a line that
run_benchmark.py can parse).

Usage:
    python models/ridge/train.py \
        --atlas destrieux_148 \
        --labels-csv data/hcp_pmat24_labels.csv \
        --subjects-file data/ready_subjects.txt \
        --task regression \
        --fold 0 --kfold 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FMRI_ROOT = ROOT / "data" / "braingnn_input"
T1_ROOT = ROOT / "data" / "t1_volume"


def load_fc_upper(atlas: str, sid: str, include_t1: bool = False):
    """Load a subject's FC matrix, return upper-triangle vector.
    If include_t1, concatenate z-scored GM volume vector.
    """
    pt = FMRI_ROOT / atlas / f"sub-{sid}.pt"
    if not pt.exists():
        return None
    data = torch.load(pt, weights_only=False)
    fc_z = data.get("fc_matrix", data.get("node_features"))
    if not torch.is_tensor(fc_z):
        fc_z = torch.as_tensor(fc_z)
    corr = torch.tanh(fc_z.float())
    corr.fill_diagonal_(0.0)
    n_roi = corr.size(0)
    iu = np.triu_indices(n_roi, k=1)
    feat = corr.numpy()[iu]  # [n_roi*(n_roi-1)/2]
    if include_t1:
        t1_path = T1_ROOT / atlas / f"sub-{sid}.npz"
        if t1_path.exists():
            t1 = np.load(t1_path, allow_pickle=True)
            gm = t1["gm_volume_mm3"].astype(float)
            if gm.size == n_roi:
                mu, sd = gm.mean(), gm.std()
                if sd < 1e-6: sd = 1.0
                gm_z = (gm - mu) / sd
                feat = np.concatenate([feat, gm_z])
    return feat


def build_labels(csv_path, dtype="float"):
    df = pd.read_csv(csv_path, dtype={"subject_id": str})
    out = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"]).strip()
        if sid.endswith(".0"): sid = sid[:-2]
        if dtype == "float":
            out[sid] = float(row["label"])
        else:
            out[sid] = int(row["label"])
    return out


def make_folds(y, kfold, seed, fold, task):
    if task == "classification":
        sk = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = list(sk.split(np.arange(len(y)), y))
    else:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(len(y))))
    tv, te = splits[fold]
    if task == "classification":
        skv = StratifiedKFold(n_splits=max(2, kfold - 1), shuffle=True, random_state=seed + 1)
        tr_rel, val_rel = next(iter(skv.split(np.arange(len(tv)), y[tv])))
    else:
        kfv = KFold(n_splits=max(2, kfold - 1), shuffle=True, random_state=seed + 1)
        tr_rel, val_rel = next(iter(kfv.split(np.arange(len(tv)))))
    return tv[tr_rel], tv[val_rel], te


def reg_metrics(y_pred, y_true):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    if y_pred.std() < 1e-9 or y_true.std() < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(y_pred, y_true)[0, 1])
    return mae, r2, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ridge")  # ignored, for CLI compat
    p.add_argument("--atlas", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--subjects-file", required=True)
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--include-t1", action="store_true")
    p.add_argument("--task", choices=["classification", "regression"], default="regression")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-components", type=int, default=100,
                   help="PCA components (0=no PCA)")
    p.add_argument("--nclass", type=int, default=2)
    # ignored (for CLI parity)
    for k in ["n-epochs","batch-size","lr","weight-decay","step-size","gamma",
              "device","lamb0","lamb1","lamb2","lamb3","lamb4","lamb5",
              "ratio","n-communities","label-scaling",
              "bnt-sizes","bnt-no-pooling","bnt-no-pos","bnt-pos-dim",
              "bnt-nhead","bnt-hidden","bnt-dec-weight",
              "save-dir"]:
        p.add_argument(f"--{k}", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    dtype = "float" if args.task == "regression" else "long"
    labels = build_labels(args.labels_csv, dtype=dtype)
    subs = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]
    subs = [s for s in subs if s in labels]

    X_list = []; y_list = []
    for sid in subs:
        feat = load_fc_upper(args.atlas, sid, include_t1=args.include_t1)
        if feat is None:
            continue
        X_list.append(feat)
        y_list.append(labels[sid])
    X = np.stack(X_list); y = np.array(y_list, dtype=float if dtype == "float" else int)
    n_samples, n_feat = X.shape
    print(f"[ridge] n_samples={n_samples} n_feat={n_feat} atlas={args.atlas}")
    if args.task == "regression":
        print(f"  y: mean={y.mean():.2f} std={y.std():.2f} range=[{y.min():.1f}, {y.max():.1f}]")

    kfold = min(args.kfold, n_samples // 2)
    tr, val, te = make_folds(y, kfold, args.seed, args.fold, args.task)
    print(f"  fold {args.fold}: train={len(tr)} val={len(val)} test={len(te)}")

    # Normalize + PCA on train only, transform all
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_val = scaler.transform(X[val])
    X_te = scaler.transform(X[te])

    n_components = int(args.n_components) if args.n_components else 0
    if n_components > 0 and n_components < min(X_tr.shape):
        pca = PCA(n_components=n_components, random_state=args.seed)
        X_tr = pca.fit_transform(X_tr)
        X_val = pca.transform(X_val)
        X_te = pca.transform(X_te)

    if args.task == "regression":
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        model.fit(X_tr, y[tr])
        y_val_pred = model.predict(X_val)
        y_te_pred = model.predict(X_te)
        val_mae, val_r2, val_r = reg_metrics(y_val_pred, y[val])
        te_mae, te_r2, te_r = reg_metrics(y_te_pred, y[te])
        dt = time.time() - t0
        if not args.quiet:
            print(f"  alpha={model.alpha_}")
            print(f"  val MAE {val_mae:.2f} r {val_r:+.3f} r2 {val_r2:+.3f}")
            print(f"  test MAE {te_mae:.2f} r {te_r:+.3f} r2 {te_r2:+.3f}")
        # Print line matching the benchmark parser format
        print(f"\n[ridge fold {args.fold}] best_val_mae={val_mae:.3f}  "
              f"test_mae={te_mae:.3f} r2={te_r2:.3f} r={te_r:+.3f}")
        print(f"saved -> (ridge has no checkpoint; t={dt:.1f}s)")
    else:
        model = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, max_iter=2000,
                                     random_state=args.seed)
        model.fit(X_tr, y[tr])
        val_acc = float(model.score(X_val, y[val]))
        te_acc = float(model.score(X_te, y[te]))
        dt = time.time() - t0
        if not args.quiet:
            print(f"  val_acc {val_acc:.3f}  test_acc {te_acc:.3f}")
        print(f"\n[ridge fold {args.fold}] best_val_acc={val_acc:.3f}  test_acc={te_acc:.3f}")
        print(f"saved -> (ridge has no checkpoint; t={dt:.1f}s)")


if __name__ == "__main__":
    main()
