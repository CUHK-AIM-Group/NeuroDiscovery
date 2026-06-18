"""Add atlas ROI labels to an existing Case Study 1 exhaustive full run.

This is a post-hoc metadata enrichment step. It does not recompute statistics;
it replaces placeholder ROI names with atlas-derived labels and regenerates the
derived top/significant/shared CSVs from the labeled all-tests table.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from case1_exhaustive_full import (
    DEFAULT_ATLAS_ROOT,
    DEFAULT_TRANSDIAG_ROOT,
    atlas_roi_meta,
    infer_target_shape,
)
from case1_exhaustive_v2 import summarize_shared_v2


STEM = "case1_exhaustive_full"
LABEL_COLUMNS = [
    "roi_id",
    "roi_name",
    "anatomy_key",
    "anatomy_full",
    "hemisphere",
    "network",
    "structure_class",
    "atlas_label_source",
    "atlas_label_weight",
    "center_of_mass",
]


def source_to_atlas(source: object) -> str:
    text = str(source)
    suffix = "_multiatlas"
    return text[: -len(suffix)] if text.endswith(suffix) else ""


def build_label_catalog(run_dir: Path, atlas_root: Path, transdiag_root: Path) -> pd.DataFrame:
    atlas_summary = pd.read_csv(run_dir / f"{STEM}_atlas_summary.csv")
    target_shape = infer_target_shape(transdiag_root)
    frames = []
    for row in atlas_summary.to_dict("records"):
        atlas = str(row["atlas"])
        n_roi = int(row["n_roi"])
        meta = atlas_roi_meta(atlas, n_roi, atlas_root=atlas_root, target_shape=target_shape).copy()
        meta.insert(0, "source", f"{atlas}_multiatlas")
        frames.append(meta)
    return pd.concat(frames, ignore_index=True)


def enrich_all_tests(all_df: pd.DataFrame, label_catalog: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["source", "roi_index", *LABEL_COLUMNS]
    labels = label_catalog[label_cols].drop_duplicates(["source", "roi_index"])
    merged = all_df.merge(labels, on=["source", "roi_index"], how="left", suffixes=("", "_label"))
    is_fmri = merged["source"].astype(str).str.endswith("_multiatlas")
    for col in LABEL_COLUMNS:
        mapped = f"{col}_label"
        if mapped not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = pd.NA
        merged[col] = merged[col].astype("object")
        use = is_fmri & merged[mapped].notna()
        merged.loc[use, col] = merged.loc[use, mapped]
        merged = merged.drop(columns=[mapped])
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--transdiag-root", type=Path, default=DEFAULT_TRANSDIAG_ROOT)
    args = parser.parse_args()

    run_dir = args.run_dir
    all_path = run_dir / f"{STEM}_all_tests.csv"
    if not all_path.exists():
        raise FileNotFoundError(all_path)

    label_catalog = build_label_catalog(run_dir, args.atlas_root, args.transdiag_root)
    label_catalog_path = run_dir / f"{STEM}_roi_label_catalog.csv"
    label_catalog.to_csv(label_catalog_path, index=False)

    all_df = pd.read_csv(all_path, low_memory=False)
    labeled = enrich_all_tests(all_df, label_catalog)

    labeled_all_path = run_dir / f"{STEM}_all_tests_labeled.csv"
    labeled_top_path = run_dir / f"{STEM}_top_hits_labeled.csv"
    labeled_sig_global_path = run_dir / f"{STEM}_significant_global_q05_labeled.csv"
    labeled_sig_modality_path = run_dir / f"{STEM}_significant_modality_q05_labeled.csv"
    labeled_ci_path = run_dir / f"{STEM}_bootstrap_ci_nonzero_labeled.csv"
    labeled_shared_path = run_dir / f"{STEM}_shared_roi_feature_summary_labeled.csv"

    labeled.to_csv(labeled_all_path, index=False)
    labeled.sort_values(["abs_adjusted_residual_d", "q_fdr_modality"], ascending=[False, True]).head(5000).to_csv(
        labeled_top_path, index=False
    )
    labeled[labeled["q_fdr_global"] < 0.05].to_csv(labeled_sig_global_path, index=False)
    labeled[labeled["q_fdr_modality"] < 0.05].to_csv(labeled_sig_modality_path, index=False)
    labeled[labeled["bootstrap_ci_excludes_zero"].astype(bool)].to_csv(labeled_ci_path, index=False)
    summarize_shared_v2(labeled).to_csv(labeled_shared_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "atlas_root": str(args.atlas_root),
        "transdiag_root": str(args.transdiag_root),
        "n_label_catalog_rows": int(label_catalog.shape[0]),
        "n_labeled_tests": int(labeled.shape[0]),
        "roi_label_catalog": str(label_catalog_path),
        "all_tests_labeled": str(labeled_all_path),
        "top_hits_labeled": str(labeled_top_path),
        "significant_global_q05_labeled": str(labeled_sig_global_path),
        "significant_modality_q05_labeled": str(labeled_sig_modality_path),
        "bootstrap_ci_nonzero_labeled": str(labeled_ci_path),
        "shared_summary_labeled": str(labeled_shared_path),
    }
    manifest_path = run_dir / f"{STEM}_label_enrichment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
