"""Hypothesis-driven prediction tasks.

Maps KG hypothesis targets to HCP behavioral measures, then tests whether
the hypothesis-specified ROI connections predict the target better than
random ROI connections.

For each hypothesis-target mapping:
  1. KG-specific: extract FC values between hypothesis-specified ROI pairs
  2. Random-pair: same number of random ROI pairs as control
  3. Full-graph: full FC matrix (BrainGNN/BNT baseline)

Uses Ridge regression for ROI-pair features (low-dimensional),
BrainGNN/BNT for subgraph features.

Usage:
    python models/scripts/kg_prediction_tasks.py \
        --subjects-file data/ready_subjects.txt \
        --atlas aal_116 \
        --hypotheses neurooracle/data/quick/hypotheses_imaging_hcp.json \
        --n-random-trials 10 --kfold 5
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
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.scripts.hypothesis_roi_mapper import build_hypothesis_roi_pairs, get_roi_names
from neurooracle.src.outcome_grounding import hcp_label_for_target, target_to_hcp_label_mapping

FMRI_ROOT = ROOT / "data" / "braingnn_input"

TARGET_TO_HCP_LABEL = {
    "impulsivity": "ddisc_auc_40k",
    "Psychological Distress": "percstress",
    "Social Interaction": "social_tom_perc_tom",
    "Shyness": "extraversion",
    "insomnia": "psqi",
    "Apathy": "neg_affect",
    "aggressive subgroup": "anghostil",
    "anxiety sensitivity": "fearsomat",
    "substance abuse": "angaggr",
    "suicidality": "percstress",
    "conduct disorder": "anghostil",
    "obsessive-compulsive disorder": "neuroticism",
    "bipolar disorder": "neg_affect",
    "Attention Deficit Disorder with Hyperactivity": "flanker",
    "Depressive Disorder": "neg_affect",
    "social phobia": "extraversion",
    "Hallucinations": "neuroticism",
    "Social Skills": "friendship",
    "personal relationships": "friendship",
}
TARGET_TO_HCP_LABEL.update(target_to_hcp_label_mapping())


def load_fc_matrix(atlas: str, sid: str) -> np.ndarray | None:
    pt_path = FMRI_ROOT / atlas / f"sub-{sid}.pt"
    if not pt_path.exists():
        return None
    data = torch.load(pt_path, weights_only=False)
    fc_z = data.get("fc_matrix", data.get("node_features"))
    if torch.is_tensor(fc_z):
        fc_z = fc_z.numpy()
    corr = np.tanh(fc_z)
    np.fill_diagonal(corr, 0.0)
    return corr


def extract_pair_features(fc: np.ndarray, roi_pairs: list[tuple[list[int], list[int]]]) -> np.ndarray:
    """Extract FC values between specified ROI pairs."""
    features = []
    for idx_a, idx_b in roi_pairs:
        for a in idx_a:
            for b in idx_b:
                if a != b:
                    features.append(fc[a, b])
    return np.array(features, dtype=np.float32)


def run_ridge_cv(X: np.ndarray, y: np.ndarray, kfold: int, seed: int) -> dict:
    """Run Ridge regression with K-fold CV, return aggregate metrics."""
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    maes, r2s, rs = [], [], []

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_tr, y_tr)
        y_pred = ridge.predict(X_te)

        maes.append(mean_absolute_error(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
        if y_pred.std() > 1e-9 and y_te.std() > 1e-9:
            rs.append(np.corrcoef(y_pred, y_te)[0, 1])
        else:
            rs.append(0.0)

    return {
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
        "r_mean": np.mean(rs), "r_std": np.std(rs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects-file", required=True)
    p.add_argument("--atlas", default="aal_116")
    p.add_argument("--hypotheses", required=True)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-random-trials", type=int, default=10)
    p.add_argument("--out-dir", default="models/experiment_results/kg_prediction")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in Path(args.subjects_file).read_text().splitlines() if s.strip()]
    roi_names = get_roi_names(args.atlas)
    n_roi = len(roi_names)
    hyp_pairs = build_hypothesis_roi_pairs(ROOT / args.hypotheses, args.atlas)

    print(f"Subjects: {len(subjects)}")
    print(f"Atlas: {args.atlas} ({n_roi} ROIs)")
    print(f"Hypothesis ROI pairs: {len(hyp_pairs)}")
    print()

    # Group hypotheses by target
    target_groups: dict[str, list[dict]] = {}
    for pair in hyp_pairs:
        target = pair["target_name"]
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(pair)

    # Filter to targets that have HCP label mappings
    valid_tasks = []
    for target, pairs in target_groups.items():
        hcp_label = TARGET_TO_HCP_LABEL.get(target) or hcp_label_for_target(target)
        if hcp_label:
            label_file = ROOT / "data" / f"hcp_{hcp_label}_labels.csv"
            if label_file.exists():
                valid_tasks.append({
                    "target": target,
                    "hcp_label": hcp_label,
                    "label_file": label_file,
                    "pairs": pairs,
                })

    print(f"Valid prediction tasks: {len(valid_tasks)}")
    for t in valid_tasks:
        print(f"  {t['target']} -> {t['hcp_label']} ({len(t['pairs'])} ROI pairs)")
    print()

    # Load all FC matrices
    print("Loading FC matrices...", flush=True)
    fc_cache = {}
    for sid in subjects:
        fc = load_fc_matrix(args.atlas, sid)
        if fc is not None:
            fc_cache[sid] = fc
    print(f"  Loaded {len(fc_cache)} subjects\n")

    all_results = []
    rng = np.random.default_rng(args.seed)

    for task in valid_tasks:
        print(f"--- Task: {task['target']} -> {task['hcp_label']} ---")

        # Load labels
        df_labels = pd.read_csv(task["label_file"], dtype={"subject_id": str})
        label_map = dict(zip(df_labels["subject_id"].astype(str), df_labels["label"].astype(float)))

        # Get subjects with both FC and label
        valid_subs = [s for s in subjects if s in fc_cache and s in label_map]
        if len(valid_subs) < 20:
            print(f"  Skipping: only {len(valid_subs)} valid subjects\n")
            continue

        y = np.array([label_map[s] for s in valid_subs])
        print(f"  Subjects: {len(valid_subs)}, y: mean={y.mean():.2f} std={y.std():.2f}")

        # Extract KG-specified ROI pair features
        kg_roi_pairs = [(p["roi_indices_a"], p["roi_indices_b"]) for p in task["pairs"]]
        X_kg = np.array([extract_pair_features(fc_cache[s], kg_roi_pairs) for s in valid_subs])
        n_features = X_kg.shape[1]
        print(f"  KG features: {n_features} FC values from {len(kg_roi_pairs)} ROI pairs")

        # Run KG-guided Ridge
        kg_result = run_ridge_cv(X_kg, y, args.kfold, args.seed)
        print(f"  KG-guided:  MAE={kg_result['mae_mean']:.3f}±{kg_result['mae_std']:.3f} "
              f"r={kg_result['r_mean']:+.3f}±{kg_result['r_std']:.3f}")

        # Run random ROI pair controls
        rand_results = []
        for trial in range(args.n_random_trials):
            rand_pairs = []
            for _ in range(len(kg_roi_pairs)):
                a_idx = rng.choice(n_roi, size=len(kg_roi_pairs[0][0]), replace=False).tolist()
                b_idx = rng.choice(n_roi, size=len(kg_roi_pairs[0][1]), replace=False).tolist()
                rand_pairs.append((a_idx, b_idx))
            X_rand = np.array([extract_pair_features(fc_cache[s], rand_pairs) for s in valid_subs])
            rand_result = run_ridge_cv(X_rand, y, args.kfold, args.seed)
            rand_results.append(rand_result)

        rand_maes = [r["mae_mean"] for r in rand_results]
        rand_rs = [r["r_mean"] for r in rand_results]
        print(f"  Random:     MAE={np.mean(rand_maes):.3f}±{np.std(rand_maes):.3f} "
              f"r={np.mean(rand_rs):+.3f}±{np.std(rand_rs):.3f}")

        # Statistical test
        t_stat, p_val = stats.ttest_1samp(rand_maes, kg_result["mae_mean"])
        print(f"  KG vs Random MAE: t={t_stat:.2f}, p={p_val:.4f} "
              f"({'KG better' if kg_result['mae_mean'] < np.mean(rand_maes) else 'Random better'})")
        print()

        all_results.append({
            "target": task["target"],
            "hcp_label": task["hcp_label"],
            "n_subjects": len(valid_subs),
            "n_features": n_features,
            "n_roi_pairs": len(kg_roi_pairs),
            "kg_mae": kg_result["mae_mean"],
            "kg_mae_std": kg_result["mae_std"],
            "kg_r": kg_result["r_mean"],
            "kg_r_std": kg_result["r_std"],
            "rand_mae_mean": np.mean(rand_maes),
            "rand_mae_std": np.std(rand_maes),
            "rand_r_mean": np.mean(rand_rs),
            "rand_r_std": np.std(rand_rs),
            "t_stat": t_stat,
            "p_value": p_val,
            "kg_better": kg_result["mae_mean"] < np.mean(rand_maes),
        })

    # Summary
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "results.csv", index=False)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if len(df) > 0:
        n_better = df["kg_better"].sum()
        n_sig = (df["p_value"] < 0.05).sum()
        print(f"Tasks evaluated: {len(df)}")
        print(f"KG better than random: {n_better}/{len(df)}")
        print(f"Statistically significant (p<0.05): {n_sig}/{len(df)}")
        print()
        print(df[["target", "hcp_label", "kg_mae", "rand_mae_mean", "kg_r", "rand_r_mean", "p_value", "kg_better"]]
              .to_string(index=False))
    else:
        print("No valid tasks found.")

    print(f"\nResults saved to: {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
