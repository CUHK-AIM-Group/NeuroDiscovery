"""harmonize.py — CLI dispatcher for the harmonization-tool skill.

Usage example:
    python harmonize.py \
        --features features.npy \
        --meta meta.csv \
        --method combat-gam \
        --batch site \
        --protected age sex dx \
        --feature-kind roi \
        --split site-stratified \
        --out runs/abide_combat_gam/

Outputs (under --out):
    harmonized_features.npy
    meta.csv
    manifest.json
    site_effect_before.csv
    site_effect_after.csv
    splits.json   (if --split is set)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running both as a module and as a plain script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_schema import load_inputs, validate_inputs, HarmonizationInputs
    from diagnostics import site_effect_r2, compare_reports
    from adapters import build as build_adapter
    from splitters import leave_site_out_splits, site_stratified_split
else:
    from .io_schema import load_inputs, validate_inputs, HarmonizationInputs
    from .diagnostics import site_effect_r2, compare_reports
    from .adapters import build as build_adapter
    from .splitters import leave_site_out_splits, site_stratified_split


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Harmonize neuroimaging features across sites/scanners."
    )
    p.add_argument("--features", required=True, help="path to features .npy")
    p.add_argument("--meta", required=True, help="path to meta .csv")
    p.add_argument(
        "--method",
        default="combat-gam",
        choices=["none", "site-covar", "combat", "combat-gam", "combat-raw", "covbat"],
    )
    p.add_argument("--batch", default="site", help="batch column to harmonize away")
    p.add_argument(
        "--protected",
        nargs="+",
        default=["age", "sex", "dx"],
        help="protected covariates",
    )
    p.add_argument(
        "--feature-kind",
        default="roi",
        choices=["roi", "connectome", "voxel"],
    )
    p.add_argument(
        "--split",
        default="none",
        choices=["none", "site-stratified", "leave-site-out"],
        help="optional split to fit harmonization on train only",
    )
    p.add_argument("--label-col", default=None, help="optional stratify label")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--run-id",
        default=None,
        help="manifest run_id; auto-generated if omitted",
    )
    return p.parse_args()


def _harmonize_no_split(
    inputs: HarmonizationInputs, method: str
) -> tuple[np.ndarray, dict]:
    adapter = build_adapter(method, batch=inputs.batch, protected=inputs.protected)
    harmonized = adapter.fit_transform(inputs.features, inputs.meta)
    return harmonized, {
        "split_protocol": "none",
        "method_name": adapter.method_name(),
    }


def _harmonize_with_site_stratified(
    inputs: HarmonizationInputs, method: str, label_col: str | None, seed: int
) -> tuple[np.ndarray, dict]:
    train_idx, val_idx, test_idx = site_stratified_split(
        inputs.meta,
        site_col=inputs.batch,
        label_col=label_col,
        seed=seed,
    )
    adapter = build_adapter(method, batch=inputs.batch, protected=inputs.protected)
    adapter.fit(inputs.features[train_idx], inputs.meta.iloc[train_idx])

    # Site-covar / neuroHarmonize support transform on new data; combat-raw / covbat do not.
    try:
        harmonized = adapter.transform(inputs.features, inputs.meta)
    except NotImplementedError:
        # Fall back: refit on full cohort. Documented limitation.
        harmonized = adapter.fit_transform(inputs.features, inputs.meta)
        split_note = "fallback_full_cohort_fit"
    else:
        split_note = "fit_train_transform_all"

    return harmonized, {
        "split_protocol": "site-stratified-80-10-10",
        "method_name": adapter.method_name(),
        "split_note": split_note,
        "splits": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        },
    }


def main() -> int:
    args = _parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(
        features_path=args.features,
        meta_path=args.meta,
        batch=args.batch,
        protected=tuple(args.protected),
        feature_kind=args.feature_kind,
    )

    before = site_effect_r2(inputs.features, inputs.meta, batch=inputs.batch)

    if args.split == "leave-site-out":
        # LOSO is a per-fold protocol; this CLI runs only the first fold here
        # for a quick demo. Full LOSO sweeps belong in an experiment script.
        held, train_idx, val_idx, test_idx = next(
            leave_site_out_splits(inputs.meta, site_col=inputs.batch, seed=args.seed)
        )
        adapter = build_adapter(
            args.method, batch=inputs.batch, protected=inputs.protected
        )
        adapter.fit(inputs.features[train_idx], inputs.meta.iloc[train_idx])
        try:
            harmonized = adapter.transform(inputs.features, inputs.meta)
            split_note = "fit_train_transform_all"
        except NotImplementedError:
            harmonized = adapter.fit_transform(inputs.features, inputs.meta)
            split_note = "fallback_full_cohort_fit"
        meta_extra = {
            "split_protocol": f"leave-site-out:held={held}",
            "method_name": adapter.method_name(),
            "split_note": split_note,
            "splits": {
                "held_out_site": held,
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            },
        }
    elif args.split == "site-stratified":
        harmonized, meta_extra = _harmonize_with_site_stratified(
            inputs, args.method, args.label_col, args.seed
        )
    else:
        harmonized, meta_extra = _harmonize_no_split(inputs, args.method)

    after = site_effect_r2(harmonized, inputs.meta, batch=inputs.batch)

    np.save(out / "harmonized_features.npy", harmonized)
    inputs.meta.to_csv(out / "meta.csv", index=False)
    before.to_frame().to_csv(out / "site_effect_before.csv", index_label="feature")
    after.to_frame().to_csv(out / "site_effect_after.csv", index_label="feature")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "batch": args.batch,
        "protected": list(args.protected),
        "feature_kind": args.feature_kind,
        "feature_shape": list(inputs.features.shape),
        "n_subjects": int(inputs.n_subjects()),
        "site_effect": compare_reports(before, after),
        "inputs": {
            "features": str(args.features),
            "meta": str(args.meta),
        },
        **meta_extra,
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = manifest["site_effect"]
    print(
        f"[harmonize] method={args.method} split={args.split} "
        f"site_R2 mean: {summary['before']['mean_r2']:.4f} -> "
        f"{summary['after']['mean_r2']:.4f} "
        f"(delta {summary['delta_mean_r2']:.4f})"
    )
    print(f"[harmonize] outputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
