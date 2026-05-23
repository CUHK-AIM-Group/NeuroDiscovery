"""Multi-site harmonization pilot.

Three-way comparison protocol:
    (a) no harmonization + random 80/10/10
    (b) no harmonization + site-stratified 80/10/10
    (c) ComBat-GAM   + site-stratified 80/10/10

Read (a) - (b) as the site-leakage budget under random splitting.
Read (c) - (b) as the harmonization gain on a site-aware split.

Why site-stratified, not leave-site-out?
ComBat-GAM (and CovBat) cannot harmonize a site it did not see during
fit — applying the model to an unseen site produces NaNs. Site-stratified
splits keep every site present in train/val/test in proportion, which
matches the project default 80/10/10 protocol and lets ComBat-GAM run
cleanly. Leave-site-out is a separate, stricter cross-site generalization
check that requires either site-as-covariate harmonization (weaker) or
test-time domain adaptation (more involved); see SKILL.md.

Two data sources are supported:
  --source synthetic : injects bio + site-offset/scale/prevalence signals.
                       Used to validate the harmonization plumbing.
  --source adhd200   : real ADHD-200 cohort, 669 subjects across 6 sites,
                       aal_116 atlas. Closest in-repo proxy to ABIDE; ABIDE
                       features are not extracted here yet.

Model: logistic regression on connectome upper triangle. Acts as a
neutral, fast stand-in for any feature-level deep model (BrainGNN, BNT,
IBGNN). Harmonization is feature-level so the choice of classifier does
not matter for testing whether harmonization works.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "skills" / "harmonization-tool" / "scripts"))

from io_schema import HarmonizationInputs, validate_inputs  # noqa: E402
from diagnostics import site_effect_r2, compare_reports  # noqa: E402
from adapters import build as build_adapter  # noqa: E402
from splitters import leave_site_out_splits, site_stratified_split  # noqa: E402
from loaders import load_adhd200_connectomes, load_abide_connectomes  # noqa: E402


@dataclass
class RunResult:
    name: str
    site_r2_before: float
    site_r2_after: float
    accuracy: float
    balanced_accuracy: float
    auc: float
    n_train: int
    n_val: int
    n_test: int
    note: str = ""


def generate_synthetic(
    n_subjects_per_site: dict[str, int],
    n_roi: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    sids: list[str] = []
    fcs: list[np.ndarray] = []

    site_offsets = {
        s: float(rng.normal(scale=0.6)) for s in n_subjects_per_site
    }
    site_scales = {
        s: float(np.exp(rng.normal(scale=0.25))) for s in n_subjects_per_site
    }
    # Site-specific *prevalence imbalance*: each site has a different
    # case/control ratio. This is what makes site information *predictive*
    # of dx in the (a) random-split run, leaking across splits.
    # Use a wide spread to make the leakage budget visible.
    site_dx_prior = {
        s: float(np.clip(rng.normal(0.5, 0.3), 0.1, 0.9))
        for s in n_subjects_per_site
    }

    network_a = list(range(0, n_roi // 2))
    network_b = list(range(n_roi // 2, n_roi))

    for site, n_sub in n_subjects_per_site.items():
        for i in range(n_sub):
            sid = f"{site}-{i:04d}"
            age = float(rng.integers(18, 60))
            sex = rng.choice(["M", "F"])
            p_case = site_dx_prior[site]
            dx = int(rng.choice([0, 1], p=[1 - p_case, p_case]))

            # Biological signal: very subtle within-network coupling shift.
            # Kept small on purpose so the site effect (and its potential
            # leakage benefit under random splitting) becomes visible.
            ts = rng.standard_normal((200, n_roi))
            common_a = rng.standard_normal(200) * (0.30 if dx == 1 else 0.18)
            common_b = rng.standard_normal(200) * (0.18 if dx == 1 else 0.30)
            for r in network_a:
                ts[:, r] += common_a
            for r in network_b:
                ts[:, r] += common_b

            fc = np.corrcoef(ts.T)
            np.fill_diagonal(fc, 0.0)

            # Site effect: additive offset + multiplicative scaling on edges
            fc = fc * site_scales[site] + site_offsets[site]
            np.fill_diagonal(fc, 0.0)

            rows.append({
                "subject_id": sid,
                "dataset": "SYNTH-ABIDE",
                "site": site,
                "age": age,
                "sex": sex,
                "dx": dx,
            })
            sids.append(sid)
            fcs.append(fc)

    features = np.stack(fcs, axis=0)
    meta = pd.DataFrame(rows)
    return features, meta


def upper_tri(features: np.ndarray) -> np.ndarray:
    n, r, _ = features.shape
    iu = np.triu_indices(r, k=1)
    return features[:, iu[0], iu[1]]


def train_eval(
    features: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> dict:
    X = upper_tri(features)
    y = meta["dx"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx]) if len(val_idx) else None
    X_test = scaler.transform(X[test_idx])

    # Strong L2 to prevent LR from memorizing training set;
    # we want to see signal on held-out data, not training accuracy.
    model = LogisticRegression(
        max_iter=2000, C=0.05, solver="lbfgs", random_state=seed
    )
    model.fit(X_train, y[train_idx])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test = y[test_idx]

    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": (
            float(roc_auc_score(y_test, y_prob))
            if len(np.unique(y_test)) > 1
            else float("nan")
        ),
    }
    return out


def run_a_random(features, meta, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    N = len(meta)
    perm = rng.permutation(N)
    n_tr = int(0.8 * N)
    n_va = int(0.1 * N)
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr : n_tr + n_va]
    test_idx = perm[n_tr + n_va :]

    before = site_effect_r2(features, meta).mean_r2
    metrics = train_eval(features, meta, train_idx, val_idx, test_idx, seed)
    return RunResult(
        name="(a) none + random-8/1/1",
        site_r2_before=before,
        site_r2_after=before,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        auc=metrics["auc"],
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        note="optimistic baseline; site information leaks across splits",
    )


def run_b_stratified_no_harm(features, meta, seed: int) -> RunResult:
    train_idx, val_idx, test_idx = site_stratified_split(
        meta, site_col="site", label_col="dx", seed=seed
    )
    before = site_effect_r2(features, meta).mean_r2
    metrics = train_eval(features, meta, train_idx, val_idx, test_idx, seed)
    return RunResult(
        name="(b) none + site-stratified-8/1/1",
        site_r2_before=before,
        site_r2_after=before,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        auc=metrics["auc"],
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        note="site-aware split, no harmonization",
    )


def run_c_stratified_combat_gam(features, meta, seed: int, method: str = "combat-gam") -> RunResult:
    train_idx, val_idx, test_idx = site_stratified_split(
        meta, site_col="site", label_col="dx", seed=seed
    )
    before_report = site_effect_r2(features, meta)

    adapter = build_adapter(
        method, batch="site", protected=("age", "sex", "dx")
    )
    note = ""
    try:
        adapter.fit(features[train_idx], meta.iloc[train_idx])
        try:
            harmonized = adapter.transform(features, meta)
            note = "fit-train, transform-all (clean)"
        except NotImplementedError:
            harmonized = adapter.fit_transform(features, meta)
            note = "fallback to full-cohort fit"
    except ImportError:
        adapter = build_adapter(
            "site-covar", batch="site", protected=("age", "sex", "dx")
        )
        adapter.fit(features[train_idx], meta.iloc[train_idx])
        harmonized = adapter.transform(features, meta)
        note = "neuroHarmonize missing; used site-as-covariate fallback"

    after_report = site_effect_r2(harmonized, meta)
    metrics = train_eval(harmonized, meta, train_idx, val_idx, test_idx, seed)
    return RunResult(
        name=f"(c) {adapter.method_name()} + site-stratified-8/1/1",
        site_r2_before=before_report.mean_r2,
        site_r2_after=after_report.mean_r2,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        auc=metrics["auc"],
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        note=note,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        choices=["synthetic", "adhd200", "abide"],
        default="synthetic",
        help="synthetic = injected multi-site signal (validation); "
             "adhd200 = real ADHD-200 cohort, 669 subjects, aal_116; "
             "abide = real ABIDE I cohort (atlas via --abide-atlas)",
    )
    p.add_argument(
        "--abide-atlas",
        default="rois_aal",
        choices=["rois_aal", "rois_cc200", "rois_cc400",
                 "rois_dosenbach160", "rois_ez", "rois_ho", "rois_tt"],
        help="ABIDE parcellation; only used by --source abide",
    )
    p.add_argument("--n-roi", type=int, default=30,
                   help="only used by --source synthetic")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument(
        "--method",
        default="combat-gam",
        choices=["combat", "combat-gam", "site-covar"],
        help="harmonization method for run (c). combat-gam protects age "
             "non-linearly but is slower (per-edge GAM); combat is much "
             "faster on large cohorts; site-covar is the LOSO-safe fallback.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="default: runs/harmonization_pilot_<source>",
    )
    args = p.parse_args()

    if args.out is None:
        args.out = str(REPO / "runs" / f"harmonization_pilot_{args.source}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "adhd200":
        # real cohort is fixed; no per-seed regeneration
        features_fixed, meta_fixed = load_adhd200_connectomes()
        n_roi_actual = features_fixed.shape[1]
        site_summary = (
            meta_fixed.groupby(["site", "dx"]).size().unstack(fill_value=0)
        )
        print(f"[pilot] source=adhd200  N={len(meta_fixed)}  "
              f"ROIs={n_roi_actual}  sites={sorted(meta_fixed['site'].unique())}  "
              f"seeds={args.seeds}")
        print("\n[pilot] site x dx counts:")
        print(site_summary.to_string())
        print()
    elif args.source == "abide":
        features_fixed, meta_fixed = load_abide_connectomes(atlas=args.abide_atlas)
        n_roi_actual = features_fixed.shape[1]
        site_summary = (
            meta_fixed.groupby(["site", "dx"]).size().unstack(fill_value=0)
        )
        print(f"[pilot] source=abide  atlas={args.abide_atlas}  "
              f"N={len(meta_fixed)}  ROIs={n_roi_actual}  "
              f"sites={sorted(meta_fixed['site'].unique())}  seeds={args.seeds}")
        print("\n[pilot] site x dx counts:")
        print(site_summary.to_string())
        print()
    else:
        site_sizes = {"NYU": 80, "UM": 70, "USM": 60, "Stanford": 50, "OHSU": 40}
        features_fixed = None  # regenerated per seed below

    all_rows = []
    per_seed_summaries = []
    for seed in args.seeds:
        if args.source in ("adhd200", "abide"):
            features, meta = features_fixed, meta_fixed
        else:
            features, meta = generate_synthetic(
                site_sizes, n_roi=args.n_roi, seed=seed
            )

        inputs = HarmonizationInputs(
            features=features,
            meta=meta,
            batch="site",
            protected=("age", "sex", "dx"),
            feature_kind="connectome",
        )
        validate_inputs(inputs)

        results = [
            run_a_random(features, meta, seed=seed),
            run_b_stratified_no_harm(features, meta, seed=seed),
            run_c_stratified_combat_gam(features, meta, seed=seed, method=args.method),
        ]

        for r in results:
            all_rows.append({
                "seed": seed,
                "run": r.name,
                "site_R2_before": round(r.site_r2_before, 4),
                "site_R2_after": round(r.site_r2_after, 4),
                "acc": round(r.accuracy, 4),
                "bal_acc": round(r.balanced_accuracy, 4),
                "auc": round(r.auc, 4),
                "n_train": r.n_train,
                "n_val": r.n_val,
                "n_test": r.n_test,
                "note": r.note,
            })

        per_seed_summaries.append({
            "seed": seed,
            "leakage_budget_acc": results[0].accuracy - results[1].accuracy,
            "harmonization_gain_acc": results[2].accuracy - results[1].accuracy,
            "site_r2_after": results[2].site_r2_after,
            "site_r2_before": results[2].site_r2_before,
        })

    df = pd.DataFrame(all_rows)
    if args.source == "synthetic":
        print(f"[pilot] source=synthetic  N={len(meta)}  ROIs={args.n_roi}  "
              f"sites={list(site_sizes.keys())}  seeds={args.seeds}")
    print()
    print(df.to_string(index=False))
    df.to_csv(out / "results.csv", index=False)

    # Aggregate over seeds
    sums = pd.DataFrame(per_seed_summaries)
    agg = {
        "site_leakage_budget_acc_mean": float(sums["leakage_budget_acc"].mean()),
        "site_leakage_budget_acc_std": float(sums["leakage_budget_acc"].std(ddof=0)),
        "harmonization_gain_acc_mean": float(sums["harmonization_gain_acc"].mean()),
        "harmonization_gain_acc_std": float(sums["harmonization_gain_acc"].std(ddof=0)),
        "site_R2_before_mean": float(sums["site_r2_before"].mean()),
        "site_R2_after_mean": float(sums["site_r2_after"].mean()),
        "site_R2_reduction_mean": float(
            sums["site_r2_before"].mean() - sums["site_r2_after"].mean()
        ),
        "n_seeds": len(args.seeds),
    }

    print()
    print("[pilot] interpretation (mean over seeds)")
    print(f"  site-leakage budget   (a)-(b) acc: "
          f"{agg['site_leakage_budget_acc_mean']:+.4f}  ± {agg['site_leakage_budget_acc_std']:.4f}")
    print(f"  harmonization gain    (c)-(b) acc: "
          f"{agg['harmonization_gain_acc_mean']:+.4f}  ± {agg['harmonization_gain_acc_std']:.4f}")
    print(f"  site R^2 reduction:   {agg['site_R2_before_mean']:.4f} -> "
          f"{agg['site_R2_after_mean']:.4f}  (-{agg['site_R2_reduction_mean']:.4f})")

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"per_seed": per_seed_summaries, "aggregate": agg, "rows": all_rows}, f, indent=2)

    print(f"\n[pilot] outputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
