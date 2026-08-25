"""CLI for ROI MVPA, ROI GLM, and SearchLight."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.artifacts import RunArtifacts
from models.common.data import load_tabular_csv
from models.common.training import cross_validate_estimator
from models.neuroimaging_decoding.roi_glm import fit_roi_glm
from models.neuroimaging_decoding.searchlight import fit_searchlight
from models.statistical_ml.estimators import make_estimator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mvpa", "roi-glm", "searchlight"], required=True)
    parser.add_argument("--features")
    parser.add_argument("--target")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--model", default="svm")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--images-list")
    parser.add_argument("--mask")
    parser.add_argument("--design")
    parser.add_argument("--contrast-index", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    artifacts = RunArtifacts(output, vars(args))
    artifacts.write_config()
    if args.mode == "mvpa":
        if not all([args.features, args.target, args.task]):
            parser.error("mvpa requires --features, --target, and --task")
        data = load_tabular_csv(args.features, args.target, args.subject_col)
        metrics = cross_validate_estimator(
            data,
            lambda: make_estimator(args.model, args.task, args.seed),
            args.task,
            output,
            vars(args),
            args.folds,
            args.seed,
        )
        print(metrics)
        return 0
    if args.mode == "roi-glm":
        if not args.features or not args.design:
            parser.error("roi-glm requires --features and --design")
        roi_frame = pd.read_csv(args.features)
        design_frame = pd.read_csv(args.design)
        roi_columns = [c for c in roi_frame if c != args.subject_col]
        design_columns = [c for c in design_frame if c != args.subject_col]
        merged = roi_frame.merge(design_frame, on=args.subject_col, how="inner")
        result = fit_roi_glm(
            merged[roi_columns].to_numpy(),
            merged[design_columns].to_numpy(),
            roi_columns,
            args.contrast_index,
        )
        result.to_csv(output / "roi_glm_results.csv", index=False)
        metrics = {
            "rois_tested": len(result),
            "fdr_significant": int((result["q_value"] < 0.05).sum()),
        }
        artifacts.write_metrics(metrics)
        artifacts.write_manifest()
        print(metrics)
        return 0
    if not all([args.images_list, args.features, args.target, args.mask]):
        parser.error(
            "searchlight requires --images-list, --features labels, --target, and --mask"
        )
    images = Path(args.images_list).read_text(encoding="utf-8").splitlines()
    labels = pd.read_csv(args.features)[args.target].to_numpy()
    fit_searchlight(
        images,
        labels,
        args.mask,
        output / "searchlight_scores.nii.gz",
        cv=args.folds,
    )
    metrics = {"samples": len(images)}
    artifacts.write_metrics(metrics)
    artifacts.write_manifest()
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
