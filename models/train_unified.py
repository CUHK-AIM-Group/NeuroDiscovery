"""Unified trainer for BrainGNN and BrainNetworkTransformer (BNT).

Both models are trained on the same NeuroClaw preprocessed FC data.
Supports classification and regression (with z-score label standardization).

Usage:
    # BrainGNN, age regression
    python models/train_unified.py --model braingnn --task regression \
        --atlas aal_116 --labels-csv data/hcp_age_labels.csv \
        --subjects-file data/ready_subjects.txt --include-t1 \
        --fold 0 --kfold 5 --n-epochs 50

    # BNT, sex classification
    python models/train_unified.py --model bnt --task classification \
        --atlas schaefer_200_7net --labels-csv data/hcp_gender_labels.csv \
        --subjects-file data/ready_subjects.txt \
        --fold 0 --kfold 5 --n-epochs 50

Folds are deterministic given --seed, so fold 0..4 can be run sequentially
(or in a sweep loop) and their test sets are disjoint.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# BrainGNN imports (PyG)
from models.braingnn.net.braingnn import (
    BrainGNN, consist_loss, topk_loss, unit_loss,
)
from models.braingnn.scripts.data_adapter import (
    NeuroClawFCDataset, build_labels_from_csv as build_labels_pyg,
)
from torch_geometric.loader import DataLoader as PyGLoader

# BNT imports (plain PyTorch)
from models.bnt.net.bnt import BrainNetworkTransformer
from models.bnt.scripts.data_adapter import (
    BNTDataset, bnt_collate, build_labels_from_csv as build_labels_plain,
)
from torch.utils.data import DataLoader as PlainLoader


# -------------------- args --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["braingnn", "bnt"])
    p.add_argument("--task", choices=["classification", "regression"],
                   default="classification")
    p.add_argument("--atlas", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--subjects-file")
    p.add_argument("--include-t1", action="store_true")

    # CV
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)

    # Classification
    p.add_argument("--nclass", type=int, default=2)

    # Regression
    p.add_argument("--label-scaling", choices=["none", "standardization"],
                   default="standardization")

    # BrainGNN-specific
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--n-communities", type=int, default=8)
    p.add_argument("--lamb0", type=float, default=1.0)
    p.add_argument("--lamb1", type=float, default=0.0)
    p.add_argument("--lamb2", type=float, default=0.0)
    p.add_argument("--lamb3", type=float, default=0.1)
    p.add_argument("--lamb4", type=float, default=0.1)
    p.add_argument("--lamb5", type=float, default=0.1)

    # BNT-specific
    p.add_argument("--bnt-sizes", nargs="+", type=int, default=None,
                   help="e.g. --bnt-sizes 100 20 (default: auto from n_roi)")
    p.add_argument("--bnt-no-pooling", action="store_true",
                   help="disable DEC pooling in all layers")
    p.add_argument("--bnt-no-pos", action="store_true",
                   help="disable identity positional embedding")
    p.add_argument("--bnt-pos-dim", type=int, default=8)
    p.add_argument("--bnt-nhead", type=int, default=4)
    p.add_argument("--bnt-hidden", type=int, default=1024)
    p.add_argument("--bnt-dec-weight", type=float, default=1.0,
                   help="weight for DEC KL auxiliary loss")

    # Training
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--step-size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--device", default=None)

    p.add_argument("--save-dir", default="models/checkpoints")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--roi-mask-file", default=None,
                   help="JSON file with boolean list for ROI subgraph selection")
    return p.parse_args()


def resolve_device(spec):
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- fold split --------------------
def make_folds(y: np.ndarray, kfold: int, seed: int, fold: int, task: str):
    from sklearn.model_selection import KFold, StratifiedKFold

    if task == "classification":
        sk = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = list(sk.split(np.arange(len(y)), y))
    else:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(len(y))))
    tv_idx, te_idx = splits[fold]

    if task == "classification":
        y_tv = y[tv_idx]
        if len(np.unique(y_tv)) > 1 and len(tv_idx) >= kfold:
            skv = StratifiedKFold(n_splits=max(2, kfold - 1), shuffle=True,
                                  random_state=seed + 1)
            tr_rel, val_rel = next(iter(skv.split(np.arange(len(tv_idx)), y_tv)))
            return tv_idx[tr_rel], tv_idx[val_rel], te_idx
    else:
        kfv = KFold(n_splits=max(2, kfold - 1), shuffle=True, random_state=seed + 1)
        tr_rel, val_rel = next(iter(kfv.split(np.arange(len(tv_idx)))))
        return tv_idx[tr_rel], tv_idx[val_rel], te_idx

    rng = np.random.default_rng(seed + 1)
    shuf = rng.permutation(tv_idx)
    cut = max(1, int(0.8 * len(shuf)))
    return shuf[:cut], shuf[cut:] if cut < len(shuf) else shuf[-1:], te_idx


# -------------------- metric helpers --------------------
def cls_metrics(y_pred: np.ndarray, y_true: np.ndarray):
    acc = float((y_pred == y_true).mean())
    return {"acc": acc}


def reg_metrics(y_pred: np.ndarray, y_true: np.ndarray):
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    if y_pred.std() < 1e-9 or y_true.std() < 1e-9:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_pred, y_true)[0, 1])
    return {"mae": mae, "r2": r2, "pearson": pearson}


# -------------------- training loops --------------------
def train_braingnn(args, device, save_dir: Path, roi_mask=None):
    label_dtype = "float" if args.task == "regression" else "long"
    labels = build_labels_pyg(args.labels_csv, args.subject_col, args.label_col,
                              label_dtype=label_dtype)
    subjects = args.subjects
    if subjects is None and args.subjects_file:
        subjects = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]

    ds = NeuroClawFCDataset(
        atlas=args.atlas, labels=labels,
        include_t1=args.include_t1, subjects=subjects,
        label_dtype=label_dtype, roi_mask=roi_mask,
    )
    n_roi = int(ds[0].x.size(0))
    indim = int(ds[0].x.size(1))
    n_samples = len(ds)

    if args.task == "regression":
        y_all = np.array([float(ds[i].y.item()) for i in range(n_samples)])
        args.nclass = 1
        print(f"[braingnn] n_samples={n_samples} n_roi={n_roi} indim={indim}")
        print(f"  y: mean={y_all.mean():.2f} std={y_all.std():.2f} "
              f"range=[{y_all.min():.1f}, {y_all.max():.1f}]")
    else:
        y_all = np.array([int(ds[i].y.item()) for i in range(n_samples)])
        uniq, counts = np.unique(y_all, return_counts=True)
        print(f"[braingnn] n_samples={n_samples} n_roi={n_roi} indim={indim}")
        print(f"  class dist: {dict(zip(uniq.tolist(), counts.tolist()))}")

    kfold = min(args.kfold, n_samples // 2)
    tr_idx, val_idx, te_idx = make_folds(y_all, kfold, args.seed, args.fold, args.task)
    print(f"  fold {args.fold}: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

    y_mean, y_std = 0.0, 1.0
    if args.task == "regression" and args.label_scaling == "standardization":
        y_mean = float(y_all[tr_idx].mean())
        y_std = max(1e-6, float(y_all[tr_idx].std()))
        # PyG InMemoryDataset shares a single y tensor + caches _data_list.
        # Mutate the underlying tensor and invalidate the cache so ds[i].y
        # actually picks up the change.
        ds._data.y = ((ds._data.y.float() - y_mean) / y_std).view(-1)
        if hasattr(ds, "_data_list") and ds._data_list is not None:
            ds._data_list = [None] * len(ds)

    train_loader = PyGLoader(ds[tr_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = PyGLoader(ds[val_idx], batch_size=args.batch_size)
    test_loader = PyGLoader(ds[te_idx], batch_size=args.batch_size)

    model = BrainGNN(
        indim=indim, ratio=args.ratio, nclass=args.nclass,
        n_roi=n_roi, n_communities=args.n_communities, task=args.task,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params = {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    def eval_one(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out, *_ = model(batch.x, batch.edge_index, batch.batch,
                                batch.edge_attr, batch.pos)
                if args.task == "regression":
                    preds.append(out.squeeze(-1).cpu().numpy())
                    trues.append(batch.y.cpu().numpy())
                else:
                    preds.append(out.argmax(dim=1).cpu().numpy())
                    trues.append(batch.y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        if args.task == "regression":
            preds = preds * y_std + y_mean
            trues = trues * y_std + y_mean
            return reg_metrics(preds, trues)
        return cls_metrics(preds, trues)

    def train_one():
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out, w1, w2, s1, s2 = model(batch.x, batch.edge_index, batch.batch,
                                         batch.edge_attr, batch.pos)
            if args.task == "regression":
                loss_c = F.mse_loss(out.squeeze(-1), batch.y.float())
                loss_gc = torch.tensor(0.0, device=device)
            else:
                loss_c = F.nll_loss(out, batch.y)
                loss_gc = 0.0
                for c in range(args.nclass):
                    m = batch.y == c
                    if m.sum() >= 2:
                        loss_gc = loss_gc + consist_loss(s1[m])
                if not torch.is_tensor(loss_gc):
                    loss_gc = torch.tensor(0.0, device=device)
            loss = (args.lamb0 * loss_c + args.lamb1 * unit_loss(w1)
                    + args.lamb2 * unit_loss(w2)
                    + args.lamb3 * topk_loss(s1, args.ratio)
                    + args.lamb4 * topk_loss(s2, args.ratio)
                    + args.lamb5 * loss_gc)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        return total / max(1, n)

    return model, opt, sched, train_loader, val_loader, test_loader, \
           eval_one, train_one, y_mean, y_std


def train_bnt(args, device, save_dir: Path, roi_mask=None):
    label_dtype = "float" if args.task == "regression" else "long"
    labels = build_labels_plain(args.labels_csv, args.subject_col, args.label_col,
                                label_dtype=label_dtype)
    subjects = args.subjects
    if subjects is None and args.subjects_file:
        subjects = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]

    ds = BNTDataset(
        atlas=args.atlas, labels=labels, include_t1=args.include_t1,
        subjects=subjects, label_dtype=label_dtype, roi_mask=roi_mask,
    )
    n_samples = len(ds)
    n_roi = ds.n_roi
    # BNT: d_model = n_roi (+ include_t1 col). It must match transformer input dim.
    # But n_roi here is corr.size(0) which is the number of ROIs regardless of
    # whether we appended GM. The transformer d_model = corr.size(1) = n_roi or n_roi+1.
    # Peek at first sample to get the actual feature dim.
    first_fc, _, _ = ds[0]
    feat_dim = int(first_fc.size(1))
    print(f"[bnt] n_samples={n_samples} n_roi={n_roi} feat_dim={feat_dim}")

    if args.task == "regression":
        y_all = np.array([float(s[2]) for s in ds.samples])
        args.nclass = 1
        print(f"  y: mean={y_all.mean():.2f} std={y_all.std():.2f} "
              f"range=[{y_all.min():.1f}, {y_all.max():.1f}]")
    else:
        y_all = np.array([int(s[2]) for s in ds.samples])
        uniq, counts = np.unique(y_all, return_counts=True)
        print(f"  class dist: {dict(zip(uniq.tolist(), counts.tolist()))}")

    kfold = min(args.kfold, n_samples // 2)
    tr_idx, val_idx, te_idx = make_folds(y_all, kfold, args.seed, args.fold, args.task)
    print(f"  fold {args.fold}: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

    y_mean, y_std = 0.0, 1.0
    if args.task == "regression" and args.label_scaling == "standardization":
        y_mean = float(y_all[tr_idx].mean())
        y_std = max(1e-6, float(y_all[tr_idx].std()))
        # Overwrite labels in-place in samples (store scaled for training)
        for i in range(len(ds.samples)):
            sid, fc, raw = ds.samples[i]
            ds.samples[i] = (sid, fc, (float(raw) - y_mean) / y_std)

    from torch.utils.data import Subset
    tr_loader = PlainLoader(Subset(ds, tr_idx.tolist()), batch_size=args.batch_size,
                            shuffle=True, collate_fn=bnt_collate)
    val_loader = PlainLoader(Subset(ds, val_idx.tolist()), batch_size=args.batch_size,
                             collate_fn=bnt_collate)
    test_loader = PlainLoader(Subset(ds, te_idx.tolist()), batch_size=args.batch_size,
                              collate_fn=bnt_collate)

    # When include_t1 is True, feat_dim = n_roi + 1; BNT needs the transformer
    # input dim to account for this. We pass feat_dim as the "n_roi" arg of BNT
    # but it's really the feature dimension. Pool sizes use n_roi (number of nodes).
    model = BrainNetworkTransformer(
        n_roi=n_roi,
        sizes=args.bnt_sizes if args.bnt_sizes else None,
        do_pooling=[False, False] if args.bnt_no_pooling else None,
        pos_encoding=None if args.bnt_no_pos else "identity",
        pos_embed_dim=args.bnt_pos_dim,
        nhead=args.bnt_nhead, hidden_size=args.bnt_hidden,
        nclass=args.nclass, task=args.task,
    )
    # Override forward_dim if include_t1 added a column — we need to override
    # the node_identity dim and re-init layers. Simplest: if feat_dim != n_roi,
    # set pos_encoding=None and n_roi=feat_dim so d_model matches.
    if feat_dim != n_roi:
        print(f"  (include_t1 added {feat_dim - n_roi} feat col; rebuilding BNT)")
        model = BrainNetworkTransformer(
            n_roi=feat_dim,  # use feat_dim as n_roi so forward_dim matches
            sizes=args.bnt_sizes if args.bnt_sizes else None,
            do_pooling=[False, False] if args.bnt_no_pooling else None,
            pos_encoding=None,  # disable pos enc since feat_dim != n_roi
            pos_embed_dim=args.bnt_pos_dim,
            nhead=args.bnt_nhead, hidden_size=args.bnt_hidden,
            nclass=args.nclass, task=args.task,
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params = {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)

    def eval_one(loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for fcs, ys, _ in loader:
                fcs = fcs.to(device)
                logits, _ = model(fcs)
                if args.task == "regression":
                    preds.append(logits.squeeze(-1).cpu().numpy())
                    trues.append(ys.numpy())
                else:
                    preds.append(logits.argmax(dim=1).cpu().numpy())
                    trues.append(ys.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        if args.task == "regression":
            preds = preds * y_std + y_mean
            trues = trues * y_std + y_mean
            return reg_metrics(preds, trues)
        return cls_metrics(preds, trues)

    def train_one():
        model.train()
        total, n = 0.0, 0
        for fcs, ys, _ in tr_loader:
            fcs = fcs.to(device); ys = ys.to(device)
            opt.zero_grad()
            logits, assignments = model(fcs)
            if args.task == "regression":
                loss_main = F.mse_loss(logits.squeeze(-1), ys.float())
            else:
                loss_main = F.cross_entropy(logits, ys.long())
            loss_dec = model.dec_loss(assignments)
            loss = loss_main + args.bnt_dec_weight * loss_dec
            loss.backward()
            opt.step()
            total += loss.item() * fcs.size(0)
            n += fcs.size(0)
        return total / max(1, n)

    return model, opt, sched, tr_loader, val_loader, test_loader, \
           eval_one, train_one, y_mean, y_std


# -------------------- main --------------------
def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = ROOT / args.save_dir / args.model / args.atlas
    save_dir.mkdir(parents=True, exist_ok=True)

    roi_mask = None
    if args.roi_mask_file:
        import json
        roi_mask = json.loads(Path(args.roi_mask_file).read_text())

    if args.model == "braingnn":
        setup = train_braingnn(args, device, save_dir, roi_mask=roi_mask)
    else:
        setup = train_bnt(args, device, save_dir, roi_mask=roi_mask)
    model, opt, sched, tr_loader, val_loader, te_loader, eval_one, train_one, y_mean, y_std = setup

    if args.dry_run:
        return

    best_primary = float("inf") if args.task == "regression" else -1.0
    best_state = None
    for epoch in range(args.n_epochs):
        t0 = time.time()
        tr_loss = train_one()
        sched.step()
        tr_m = eval_one(tr_loader)
        val_m = eval_one(val_loader)
        dt = time.time() - t0
        if args.task == "regression":
            msg = (f"epoch {epoch:3d} | loss {tr_loss:.4f} | "
                   f"tr MAE {tr_m['mae']:.2f} r {tr_m['pearson']:+.3f} | "
                   f"val MAE {val_m['mae']:.2f} r {val_m['pearson']:+.3f} | {dt:.1f}s")
            primary = val_m["mae"]
            is_best = primary < best_primary
        else:
            msg = (f"epoch {epoch:3d} | loss {tr_loss:.4f} | "
                   f"tr_acc {tr_m['acc']:.3f} | val_acc {val_m['acc']:.3f} | {dt:.1f}s")
            primary = val_m["acc"]
            is_best = primary > best_primary
        if not args.quiet or epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(msg)
        if is_best:
            best_primary = primary
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = eval_one(te_loader)
    if args.task == "regression":
        print(f"\n[{args.model} fold {args.fold}] best_val_mae={best_primary:.3f}  "
              f"test_mae={test_m['mae']:.3f} r2={test_m['r2']:.3f} r={test_m['pearson']:+.3f}")
    else:
        print(f"\n[{args.model} fold {args.fold}] best_val_acc={best_primary:.3f}  "
              f"test_acc={test_m['acc']:.3f}")

    ckpt = save_dir / f"fold{args.fold}.pt"
    torch.save({
        "state_dict": best_state, "args": vars(args),
        "best_val_primary": best_primary, "test_metrics": test_m,
        "y_mean": y_mean, "y_std": y_std,
    }, ckpt)
    print(f"saved -> {ckpt}")


if __name__ == "__main__":
    main()
