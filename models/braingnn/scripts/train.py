"""BrainGNN training entry — NeuroClaw edition.

Loads atlas-processed NeuroClaw data, runs stratified K-fold CV,
and trains BrainGNN with the original 6 loss terms:

  loss = lamb0*NLL + lamb1*unit(w1) + lamb2*unit(w2)
       + lamb3*topk(s1) + lamb4*topk(s2) + lamb5*consistency(s1)

Usage (smoke test, 10 subjects, AAL 116, 3 folds, 10 epochs):
    python models/braingnn/scripts/train.py \
        --atlas aal_116 --fold 0 --n-epochs 10 --batch-size 4

Usage with real labels:
    python models/braingnn/scripts/train.py \
        --atlas schaefer_100_7net \
        --labels-csv path/to/labels.csv \
        --subject-col subject_id --label-col sex \
        --include-t1 --n-epochs 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

# Make models.braingnn importable when run as a script
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.braingnn.net.braingnn import (
    BrainGNN,
    consist_loss,
    topk_loss,
    unit_loss,
)
from models.braingnn.scripts.data_adapter import (
    NeuroClawFCDataset,
    build_labels_from_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", required=True,
                   help="atlas name, must match data/braingnn_input/<atlas>/")
    p.add_argument("--labels-csv", default=None,
                   help="CSV with subject_id + label columns; default = synthetic parity label")
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="explicit list of subject IDs; default = all available")
    p.add_argument("--subjects-file", default=None,
                   help="path to text file with one subject ID per line (used if --subjects not given)")
    p.add_argument("--include-t1", action="store_true",
                   help="append GM volume as 1 extra node feature (requires data/t1_volume/<atlas>/)")

    # CV
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)

    # Model
    p.add_argument("--n-communities", type=int, default=8)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--nclass", type=int, default=2)

    # Training
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-3)
    p.add_argument("--step-size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--device", default=None,
                   help="cpu | cuda | cuda:0 ; default auto-detect")

    # Task type
    p.add_argument("--task", choices=["classification", "regression"],
                   default="classification")
    p.add_argument("--label-scaling", choices=["none", "standardization"],
                   default="standardization",
                   help="(regression only) normalize y to zero-mean/unit-std on training split")

    # Loss weights (paper defaults)
    p.add_argument("--lamb0", type=float, default=1.0)
    p.add_argument("--lamb1", type=float, default=0.0)
    p.add_argument("--lamb2", type=float, default=0.0)
    p.add_argument("--lamb3", type=float, default=0.1)
    p.add_argument("--lamb4", type=float, default=0.1)
    p.add_argument("--lamb5", type=float, default=0.1)

    p.add_argument("--save-dir", default="models/braingnn/checkpoints")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_folds(labels: np.ndarray, kfold: int, seed: int, fold: int,
               task: str = "classification"):
    """K-fold split. Stratified for classification, plain KFold for regression.
    Returns (train_idx, val_idx, test_idx) for the given fold.
    """
    from sklearn.model_selection import KFold, StratifiedKFold as SKF

    if task == "classification":
        skf_test = SKF(n_splits=kfold, shuffle=True, random_state=seed)
        test_splits = list(skf_test.split(np.arange(len(labels)), labels))
    else:
        kf_test = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        test_splits = list(kf_test.split(np.arange(len(labels))))

    train_val_idx, test_idx = test_splits[fold]

    if task == "classification":
        y_tv = labels[train_val_idx]
        if len(np.unique(y_tv)) > 1 and len(train_val_idx) >= kfold:
            skf_val = SKF(n_splits=max(2, kfold - 1), shuffle=True,
                          random_state=seed + 1)
            tr_rel, val_rel = next(iter(skf_val.split(np.arange(len(train_val_idx)), y_tv)))
            train_idx = train_val_idx[tr_rel]
            val_idx = train_val_idx[val_rel]
            return train_idx, val_idx, test_idx
    else:
        kf_val = KFold(n_splits=max(2, kfold - 1), shuffle=True,
                       random_state=seed + 1)
        tr_rel, val_rel = next(iter(kf_val.split(np.arange(len(train_val_idx)))))
        train_idx = train_val_idx[tr_rel]
        val_idx = train_val_idx[val_rel]
        return train_idx, val_idx, test_idx

    # Classification fallback
    rng = np.random.default_rng(seed + 1)
    shuffled = rng.permutation(train_val_idx)
    cut = max(1, int(0.8 * len(shuffled)))
    train_idx = shuffled[:cut]
    val_idx = shuffled[cut:] if cut < len(shuffled) else shuffled[-1:]
    return train_idx, val_idx, test_idx


def evaluate_classification(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logp, *_ = model(batch.x, batch.edge_index, batch.batch,
                             batch.edge_attr, batch.pos)
            pred = logp.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.numel()
    return correct / max(1, total)


def evaluate_regression(model, loader, device, y_mean: float = 0.0, y_std: float = 1.0):
    """Return (MAE in original units, R^2 in original units, Pearson r)."""
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, *_ = model(batch.x, batch.edge_index, batch.batch,
                            batch.edge_attr, batch.pos)
            preds.append(out.squeeze(-1).cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    import numpy as _np
    y_pred = _np.concatenate(preds) * y_std + y_mean
    y_true = _np.concatenate(trues) * y_std + y_mean
    mae = float(_np.mean(_np.abs(y_pred - y_true)))
    ss_res = float(_np.sum((y_true - y_pred) ** 2))
    ss_tot = float(_np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    if y_pred.std() < 1e-9 or y_true.std() < 1e-9:
        pearson = 0.0
    else:
        pearson = float(_np.corrcoef(y_pred, y_true)[0, 1])
    return mae, r2, pearson


# Backwards-compatible alias used elsewhere
def evaluate(model, loader, device):
    return evaluate_classification(model, loader, device)


def train_one_epoch(model, loader, optimizer, args, device):
    model.train()
    total_loss = 0.0
    n_total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output, w1, w2, s1, s2 = model(
            batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos
        )

        if args.task == "regression":
            # output: [B, 1], y: [B] float
            loss_c = F.mse_loss(output.squeeze(-1), batch.y.float())
            loss_gc = torch.tensor(0.0, device=device)  # no per-class consist loss in regression
        else:
            loss_c = F.nll_loss(output, batch.y)
            loss_gc = 0.0
            for c in range(args.nclass):
                mask = batch.y == c
                if mask.sum() >= 2:
                    loss_gc = loss_gc + consist_loss(s1[mask])
            if not torch.is_tensor(loss_gc):
                loss_gc = torch.tensor(0.0, device=device)

        loss_p1 = unit_loss(w1)
        loss_p2 = unit_loss(w2)
        loss_tpk1 = topk_loss(s1, args.ratio)
        loss_tpk2 = topk_loss(s2, args.ratio)

        loss = (args.lamb0 * loss_c
                + args.lamb1 * loss_p1
                + args.lamb2 * loss_p2
                + args.lamb3 * loss_tpk1
                + args.lamb4 * loss_tpk2
                + args.lamb5 * loss_gc)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        n_total += batch.num_graphs
    return total_loss / max(1, n_total)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = ROOT / args.save_dir / args.atlas
    save_dir.mkdir(parents=True, exist_ok=True)

    labels_map = None
    label_dtype = "float" if args.task == "regression" else "long"
    if args.labels_csv:
        labels_map = build_labels_from_csv(
            args.labels_csv, args.subject_col, args.label_col,
            label_dtype=label_dtype,
        )

    subjects = args.subjects
    if subjects is None and args.subjects_file:
        subjects = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]

    print(f"Loading dataset: atlas={args.atlas}, include_t1={args.include_t1}, task={args.task}")
    ds = NeuroClawFCDataset(
        atlas=args.atlas,
        labels=labels_map,
        include_t1=args.include_t1,
        subjects=subjects,
        label_dtype=label_dtype,
    )
    n_roi = int(ds[0].x.size(0))
    indim = int(ds[0].x.size(1))
    n_samples = len(ds)

    if args.task == "regression":
        if args.nclass != 1:
            print(f"  note: regression task forces nclass=1 (was {args.nclass})")
            args.nclass = 1
        y_all = np.array([float(ds[i].y.item()) for i in range(n_samples)])
        print(f"  n_samples={n_samples} n_roi={n_roi} indim={indim}")
        print(f"  y stats: mean={y_all.mean():.2f} std={y_all.std():.2f} "
              f"min={y_all.min():.1f} max={y_all.max():.1f}")
        kfold = min(args.kfold, n_samples // 2)
        if kfold != args.kfold:
            print(f"  kfold auto-adjusted {args.kfold} -> {kfold}")
    else:
        y_all = np.array([int(ds[i].y.item()) for i in range(n_samples)])
        uniq, counts = np.unique(y_all, return_counts=True)
        print(f"  n_samples={n_samples} n_roi={n_roi} indim={indim}")
        print(f"  class distribution: {dict(zip(uniq.tolist(), counts.tolist()))}")
        min_class = int(counts.min()) if len(counts) > 0 else n_samples
        kfold = min(args.kfold, min_class) if min_class >= 2 else 2
        if kfold != args.kfold:
            print(f"  kfold auto-adjusted {args.kfold} -> {kfold} (smallest class = {min_class})")

    if args.dry_run:
        print("--dry-run: dataset built, exiting.")
        return

    tr_idx, val_idx, te_idx = make_folds(y_all, kfold, args.seed, args.fold, task=args.task)
    print(f"  fold {args.fold}: train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}")

    # For regression with standardization: z-score y using training set stats,
    # then overwrite .y in the dataset tensors per split. Keep original for eval.
    y_mean = 0.0
    y_std = 1.0
    if args.task == "regression" and args.label_scaling == "standardization":
        y_mean = float(y_all[tr_idx].mean())
        y_std = float(y_all[tr_idx].std())
        if y_std < 1e-6:
            y_std = 1.0
        print(f"  label standardization: mean={y_mean:.3f} std={y_std:.3f}")
        # Apply to every sample's .y in place
        for i in range(n_samples):
            raw = float(ds[i].y.item())
            ds[i].y = torch.tensor([(raw - y_mean) / y_std], dtype=torch.float)

    train_loader = DataLoader(ds[tr_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds[val_idx], batch_size=args.batch_size)
    test_loader = DataLoader(ds[te_idx], batch_size=args.batch_size)

    model = BrainGNN(
        indim=indim, ratio=args.ratio, nclass=args.nclass,
        n_roi=n_roi, n_communities=args.n_communities,
        task=args.task,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params = {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    if args.task == "regression":
        best_val = float("inf")  # minimize MAE
        best_state = None
        for epoch in range(args.n_epochs):
            t0 = time.time()
            tr_loss = train_one_epoch(model, train_loader, optimizer, args, device)
            scheduler.step()
            tr_mae, tr_r2, tr_r = evaluate_regression(model, train_loader, device, y_mean, y_std)
            val_mae, val_r2, val_r = evaluate_regression(model, val_loader, device, y_mean, y_std)
            dt = time.time() - t0
            print(f"epoch {epoch:3d} | loss {tr_loss:.4f} | "
                  f"tr MAE {tr_mae:.2f} r {tr_r:+.3f} | "
                  f"val MAE {val_mae:.2f} r {val_r:+.3f} | {dt:.1f}s")
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        test_mae, test_r2, test_r = evaluate_regression(model, test_loader, device, y_mean, y_std)
        print(f"\n[fold {args.fold}] best_val_mae={best_val:.3f}  "
              f"test_mae={test_mae:.3f} test_r2={test_r2:.3f} test_r={test_r:+.3f}")

        ckpt = save_dir / f"fold{args.fold}_reg.pt"
        torch.save({
            "state_dict": best_state, "args": vars(args),
            "best_val_mae": best_val,
            "test_mae": test_mae, "test_r2": test_r2, "test_r": test_r,
            "y_mean": y_mean, "y_std": y_std,
            "n_roi": n_roi, "indim": indim,
        }, ckpt)
        print(f"saved -> {ckpt}")
    else:
        best_val = -1.0
        best_state = None
        for epoch in range(args.n_epochs):
            t0 = time.time()
            tr_loss = train_one_epoch(model, train_loader, optimizer, args, device)
            scheduler.step()
            tr_acc = evaluate_classification(model, train_loader, device)
            val_acc = evaluate_classification(model, val_loader, device)
            dt = time.time() - t0
            print(f"epoch {epoch:3d} | loss {tr_loss:.4f} | "
                  f"tr_acc {tr_acc:.3f} | val_acc {val_acc:.3f} | {dt:.1f}s")
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc = evaluate_classification(model, test_loader, device)
        print(f"\n[fold {args.fold}] best_val_acc={best_val:.3f}  test_acc={test_acc:.3f}")

        ckpt = save_dir / f"fold{args.fold}.pt"
        torch.save({
            "state_dict": best_state, "args": vars(args),
            "best_val_acc": best_val, "test_acc": test_acc,
            "n_roi": n_roi, "indim": indim,
        }, ckpt)
        print(f"saved -> {ckpt}")


if __name__ == "__main__":
    main()
