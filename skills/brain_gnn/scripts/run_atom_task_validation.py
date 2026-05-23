"""Atom-task downstream validation on HCP-YA with ComBrainTF.

For each selected hypothesis (Tier 1: top-K mappable hypotheses ranked by
confidence; Tier 2: task-aware connectome_behavior targets without KG region
prior), train ComBrainTF (our lifespan-age SOTA) under two conditions:

  - vanilla     : full AAL-116 FC
  - roi_mask    : subgraph restricted to ROIs implied by hypothesis.input_region

Protocol: 80/10/10 deterministic split (per user preference, not 5-fold CV).
Atlas: aal_116 (best ROI-name match for region mapper).
Target: regression MAE on z-scored label, also reported in raw units.

Output: skills/brain_gnn/scripts/atom_task_validation_results.json
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.combraintf.net.combraintf import ComBrainTF
from models.combraintf.scripts.data_adapter import (
    BNTDataset, bnt_collate, build_labels_from_csv, get_community_ids,
)
from skills.brain_gnn.scripts.hypothesis_label_mapper import (
    get_all_mappable_hypotheses, HCP_LABEL_CATEGORIES,
)
from skills.brain_gnn.scripts.region_roi_mapper import build_roi_mask

ATLAS = "aal_116"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HYP_JSON = ROOT / "neurooracle" / "data" / "full" / "hypotheses_imaging_hcp_ya.json"
DATA_DIR = ROOT / "data"


# Tier 2: task-aware connectome_behavior (no input_region; whole-brain only)
TIER2_TARGETS = [
    ("connectome_behavior:pmat24",      "pmat24",      "cognitive_executive"),
    ("connectome_behavior:neuroticism", "neuroticism", "personality"),
    ("connectome_behavior:psqi",        "psqi",        "sleep"),
]


def make_split(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    return idx[: n - n_test - n_val], idx[n - n_test - n_val: n - n_test], idx[n - n_test:]


def train_combraintf(labels_z: dict, y_mean: float, y_std: float,
                     community_ids, roi_mask, n_epochs: int, batch_size: int,
                     lr: float, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = BNTDataset(atlas=ATLAS, labels=labels_z, label_dtype="float",
                    roi_mask=roi_mask)
    n = len(ds)
    if n < 30:
        return {"error": f"too few samples ({n})"}
    n_roi = ds.n_roi

    # Community ids may need masking too
    if roi_mask is not None:
        cids = [c for c, m in zip(community_ids, roi_mask) if m]
    else:
        cids = list(community_ids)
    # Remap to contiguous 0..K-1
    uniq = sorted(set(cids))
    remap = {c: i for i, c in enumerate(uniq)}
    cids = [remap[c] for c in cids]

    tr_i, va_i, te_i = make_split(n, seed=seed)
    tr_loader = DataLoader(Subset(ds, tr_i), batch_size=batch_size, shuffle=True,
                           collate_fn=bnt_collate, drop_last=True)
    va_loader = DataLoader(Subset(ds, va_i), batch_size=batch_size, collate_fn=bnt_collate)
    te_loader = DataLoader(Subset(ds, te_i), batch_size=batch_size, collate_fn=bnt_collate)

    nhead = next((h for h in (4, 2, 1) if n_roi % h == 0), 1)
    n_clusters = min(8, n_roi)
    model = ComBrainTF(n_roi=n_roi, nclass=1, community_ids=cids,
                       n_clusters=n_clusters, hidden_size=256, nhead=nhead,
                       task="regression").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val = float("inf"); best_state = None
    for epoch in range(n_epochs):
        model.train()
        for fcs, ys, _ in tr_loader:
            fcs, ys = fcs.to(DEVICE), ys.to(DEVICE)
            opt.zero_grad()
            logits, assign = model(fcs)
            loss = F.mse_loss(logits.squeeze(-1), ys.float())
            loss = loss + 0.1 * model.dec_loss(assign)
            if not torch.isfinite(loss):
                return {"error": "nan_loss", "epoch": epoch}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for fcs, ys, _ in va_loader:
                fcs = fcs.to(DEVICE)
                out, _ = model(fcs)
                preds.append(out.squeeze(-1).cpu()); trues.append(ys)
        v = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
        if v < best_val:
            best_val = v
            best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}

    if best_state is None:
        return {"error": "no_best_state"}
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for fcs, ys, _ in te_loader:
            fcs = fcs.to(DEVICE)
            out, _ = model(fcs)
            preds.append(out.squeeze(-1).cpu()); trues.append(ys)
    test_z = F.l1_loss(torch.cat(preds), torch.cat(trues)).item()
    return {
        "val_mae_z": best_val,
        "test_mae_z": test_z,
        "test_mae_raw": test_z * y_std,
        "n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
        "n_roi_effective": n_roi,
    }


def load_zscored_labels(csv_path: Path) -> tuple[dict, float, float]:
    raw = build_labels_from_csv(csv_path, label_dtype="float")
    arr = np.array(list(raw.values()), dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    mu, sd = float(arr.mean()), float(arr.std() or 1.0)
    z = {sid: (v - mu) / sd for sid, v in raw.items() if np.isfinite(v)}
    return z, mu, sd


def get_csv_for(label_key: str, category: str) -> Path | None:
    entry = HCP_LABEL_CATEGORIES.get(category, {}).get(label_key)
    if entry is None:
        return None
    csv = DATA_DIR / entry[0]
    return csv if csv.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=8,
                    help="Top-K mappable hypotheses by confidence_score")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output",
                    default="skills/brain_gnn/scripts/atom_task_validation_results.json")
    args = ap.parse_args()

    with open(HYP_JSON, encoding="utf-8") as f:
        hyps_all = json.load(f).get("hypotheses", [])
    mappable = get_all_mappable_hypotheses(hyps_all)
    mappable.sort(key=lambda h: -float(h.get("confidence_score", 0)))
    tier1 = mappable[: args.top_k]

    print(f"Atom-task validation | atlas={ATLAS} | device={DEVICE}")
    print(f"Tier 1 (KG region-prior): top-{args.top_k} hypotheses by confidence")
    for h in tier1:
        print(f"  {h['id']} conf={h['confidence_score']:.2f}  "
              f"{h['target_name']:30s} -> {h['label_key']:18s}  "
              f"region: {h.get('metadata',{}).get('input_region','?')}")
    print(f"\nTier 2 (task-aware connectome_behavior, whole-brain only):")
    for tid, lk, cat in TIER2_TARGETS:
        csv = get_csv_for(lk, cat)
        print(f"  {tid:38s} csv={'OK' if csv else 'MISSING'}")

    if args.dry_run:
        return

    community_ids = get_community_ids(ATLAS)
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()

    # Tier 1: vanilla + roi_mask per hypothesis
    for i, h in enumerate(tier1, 1):
        label_csv = get_csv_for(h["label_key"], h["label_category"])
        if label_csv is None:
            print(f"[T1 {i}/{len(tier1)}] {h['id']} SKIP: no csv")
            continue
        labels_z, y_mu, y_sd = load_zscored_labels(label_csv)
        mask = build_roi_mask(h, n_roi=116)
        n_mask = sum(mask)

        print(f"\n[T1 {i}/{len(tier1)}] {h['id']} | {h['target_name']} -> "
              f"{h['label_key']} | mask={n_mask}/116", flush=True)

        ts = time.time()
        v = train_combraintf(labels_z, y_mu, y_sd, community_ids, None,
                              args.n_epochs, args.batch_size, args.lr, args.seed)
        print(f"  vanilla  : {v}", flush=True)
        m = train_combraintf(labels_z, y_mu, y_sd, community_ids, mask,
                              args.n_epochs, args.batch_size, args.lr, args.seed)
        print(f"  roi_mask : {m}", flush=True)

        results.append({
            "tier": 1, "hypothesis_id": h["id"],
            "target_name": h["target_name"], "label_key": h["label_key"],
            "label_category": h["label_category"],
            "confidence_score": h.get("confidence_score"),
            "input_region": h.get("metadata", {}).get("input_region"),
            "roi_mask_size": n_mask,
            "y_mean": y_mu, "y_std": y_sd,
            "vanilla": v, "roi_mask": m,
            "elapsed_sec": round(time.time() - ts, 1),
        })
        # Save partial
        out_path.write_text(json.dumps({"results": results}, indent=2,
                                        ensure_ascii=False))

    # Tier 2: vanilla only (task-aware, no region prior)
    for tid, lk, cat in TIER2_TARGETS:
        csv = get_csv_for(lk, cat)
        if csv is None:
            continue
        labels_z, y_mu, y_sd = load_zscored_labels(csv)
        print(f"\n[T2] {tid}", flush=True)
        ts = time.time()
        v = train_combraintf(labels_z, y_mu, y_sd, community_ids, None,
                              args.n_epochs, args.batch_size, args.lr, args.seed)
        print(f"  vanilla  : {v}", flush=True)
        results.append({
            "tier": 2, "task_id": tid, "label_key": lk, "label_category": cat,
            "y_mean": y_mu, "y_std": y_sd, "vanilla": v,
            "elapsed_sec": round(time.time() - ts, 1),
        })
        out_path.write_text(json.dumps({"results": results}, indent=2,
                                        ensure_ascii=False))

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f} min. Saved -> {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'ID':18s} {'target':22s} {'label':12s} {'V_yr':>8s} {'M_yr':>8s} {'Δ':>8s}")
    print("-" * 80)
    for r in results:
        if r["tier"] != 1: continue
        v = r["vanilla"].get("test_mae_raw")
        m = r["roi_mask"].get("test_mae_raw")
        v_s = f"{v:.3f}" if v is not None else "n/a"
        m_s = f"{m:.3f}" if m is not None else "n/a"
        d_s = f"{m-v:+.3f}" if (v and m) else "n/a"
        print(f"{r['hypothesis_id']:18s} {r['target_name'][:22]:22s} "
              f"{r['label_key']:12s} {v_s:>8s} {m_s:>8s} {d_s:>8s}")
    for r in results:
        if r["tier"] != 2: continue
        v = r["vanilla"].get("test_mae_raw")
        v_s = f"{v:.3f}" if v is not None else "n/a"
        print(f"{r['task_id']:30s} {r['label_key']:12s} {v_s:>8s} (T2 vanilla-only)")


if __name__ == "__main__":
    main()
