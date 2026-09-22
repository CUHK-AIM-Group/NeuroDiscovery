"""Five-fold deep-model robustness for supplemental differential diagnosis.

The discovery cohort is UCLA schizophrenia versus bipolar disorder. Outer
folds are created once per atlas and seed and reused unchanged by BrainNetCNN,
BNT, and BrainGNN. External transfer remains owned by the frozen closed-loop
runner, which retrains its selected conventional pipeline on full UCLA data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from core.scripts.case1_multimodel_pilot import (
    FoldData,
    fold_adjust_standardize,
    make_dense_connectome_model,
    make_folds,
    resolve_device,
    set_seed,
    train_dense_model,
    train_graph_model,
    write_csv_with_retry,
)
from core.scripts.run_psychiatric_closed_loop_experiments import (
    cohort_metadata,
    differential_labels,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "data/braingnn_input"
DEFAULT_OUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\supplemental_case_studies_formal"
)
DEFAULT_MODELS = ("brainnetcnn", "bnt", "braingnn")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cache_path(cache_root: Path, atlas: str, subject_id: str) -> Path:
    subject = str(subject_id).strip().removeprefix("sub-")
    return cache_root / atlas / f"sub-ucla_{subject}.pt"


def discover_atlases(cache_root: Path, subject_ids: Sequence[str]) -> list[str]:
    atlases: list[str] = []
    sample = [str(value) for value in subject_ids[: min(10, len(subject_ids))]]
    for directory in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        if sample and any(cache_path(cache_root, directory.name, value).is_file() for value in sample):
            atlases.append(directory.name)
    return atlases


def _fc_and_names(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    matrix = torch.as_tensor(blob.get("fc_matrix", blob.get("node_features"))).numpy()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"non-square FC cache: {path}")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    names = tuple(str(value) for value in blob.get("roi_names", ()))
    if len(names) != matrix.shape[0]:
        names = tuple(f"ROI_{index + 1}" for index in range(matrix.shape[0]))
    return matrix, names


def load_ucla_atlas(
    cache_root: Path,
    atlas: str,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    covariates: list[list[float]] = []
    sites: list[str] = []
    subjects: list[str] = []
    expected_names: tuple[str, ...] | None = None
    excluded: dict[str, int] = {}
    for row in metadata.itertuples(index=False):
        path = cache_path(cache_root, atlas, str(row.subject_id))
        if not path.is_file():
            excluded["missing_cache"] = excluded.get("missing_cache", 0) + 1
            continue
        try:
            matrix, names = _fc_and_names(path)
        except Exception:
            excluded["unreadable_cache"] = excluded.get("unreadable_cache", 0) + 1
            continue
        if expected_names is None:
            expected_names = names
        if names != expected_names:
            excluded["roi_contract_mismatch"] = excluded.get("roi_contract_mismatch", 0) + 1
            continue
        age = float(row.age)
        sex = float(row.sex)
        if not np.isfinite(age) or not np.isfinite(sex):
            excluded["missing_covariate"] = excluded.get("missing_covariate", 0) + 1
            continue
        matrices.append(matrix)
        labels.append(int(row.label))
        covariates.append([age, sex])
        sites.append(str(row.site))
        subjects.append(str(row.subject_id))
    if not matrices or expected_names is None:
        raise RuntimeError(f"no usable UCLA caches for {atlas}")
    roi_meta = pd.DataFrame(
        {
            "roi_index": np.arange(len(expected_names)),
            "roi_name": expected_names,
            "parcel_name": expected_names,
            "network": "",
        }
    )
    audit = {
        "atlas": atlas,
        "requested": int(len(metadata)),
        "usable": int(len(matrices)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in pd.Series(labels).value_counts().sort_index().items()
        },
        "excluded_by_reason": excluded,
        "n_rois": len(expected_names),
    }
    return (
        np.stack(matrices),
        np.asarray(labels, dtype=np.int64),
        np.asarray(covariates, dtype=np.float32),
        np.asarray(sites, dtype=str),
        subjects,
        roi_meta,
        audit,
    )


def summarize(performance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if performance.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_seed = (
        performance.groupby(["atlas", "model", "seed"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
    overall = (
        by_seed.groupby(["atlas", "model"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_variance=("auroc", "var"),
            auprc_mean=("auprc", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    return by_seed, overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--atlases", default="auto")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="20260826")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--graph-density", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.folds != 5 and not args.smoke:
        raise ValueError("formal supplemental differential diagnosis requires five folds")
    models = parse_csv(args.models)
    unknown = set(models) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"unsupported models: {sorted(unknown)}")
    seeds = [int(value) for value in parse_csv(args.seeds)]
    metadata = differential_labels(cohort_metadata("UCLA"), "UCLA")
    atlases = (
        discover_atlases(args.cache_root, metadata["subject_id"].astype(str).tolist())
        if args.atlases == "auto"
        else parse_csv(args.atlases)
    )
    if args.smoke:
        atlases = atlases[:1]
        models = models[:1]
        seeds = seeds[:1]
        args.folds = min(2, args.folds)
        args.epochs = min(2, args.epochs)
        args.patience = 1

    out_dir = args.out_root / args.run_name
    performance_path = out_dir / "performance_folds.csv"
    failure_path = out_dir / "failed_model_folds.csv"
    split_path = out_dir / "fold_assignments.csv"
    if performance_path.exists() and not args.resume:
        raise FileExistsError(f"existing run: {out_dir}; pass --resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        performance_rows = pd.read_csv(performance_path).to_dict("records")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        performance_rows = []
    completed = {
        (str(row["atlas"]), str(row["model"]), int(row["seed"]), int(row["fold"]))
        for row in performance_rows
    }
    failure_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    device = resolve_device(args.device)
    started = time.perf_counter()

    for atlas in atlases:
        raw, y, covariates, sites, subjects, roi_meta, audit = load_ucla_atlas(
            args.cache_root,
            atlas,
            metadata,
        )
        audits[atlas] = audit
        for seed in seeds:
            folds = make_folds(y, args.folds, seed, strata=y)
            for fold_index, fold in enumerate(folds):
                for split_name, indices in (
                    ("train", fold.train),
                    ("validation", fold.val),
                    ("test", fold.test),
                ):
                    split_rows.extend(
                        {
                            "atlas": atlas,
                            "seed": seed,
                            "fold": fold_index,
                            "split": split_name,
                            "subject_id": subjects[index],
                            "label": int(y[index]),
                            "site": str(sites[index]),
                        }
                        for index in indices
                    )
                adjusted = fold_adjust_standardize(raw, covariates, fold.train)
                for model_index, model in enumerate(models):
                    key = (atlas, model, seed, fold_index)
                    if key in completed:
                        continue
                    set_seed(seed + fold_index * 101 + model_index * 1009)
                    began = time.perf_counter()
                    try:
                        if model in {"brainnetcnn", "bnt"}:
                            fitted = make_dense_connectome_model(
                                model,
                                adjusted.shape[1],
                                atlas,
                                roi_meta,
                            )
                            fitted, metrics, epochs_run = train_dense_model(
                                model_name=model,
                                model=fitted,
                                x=adjusted,
                                y=y,
                                fold=fold,
                                device=device,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.lr,
                                weight_decay=args.weight_decay,
                                patience=args.patience,
                            )
                        else:
                            fitted, _dataset, metrics, epochs_run = train_graph_model(
                                model_name=model,
                                x=adjusted,
                                y=y,
                                subjects=subjects,
                                fold=fold,
                                device=device,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                learning_rate=args.lr,
                                weight_decay=args.weight_decay,
                                patience=args.patience,
                                edge_density=args.graph_density,
                            )
                        row = {
                            "atlas": atlas,
                            "model": model,
                            "seed": seed,
                            "fold": fold_index,
                            "n_train": len(fold.train),
                            "n_validation": len(fold.val),
                            "n_test": len(fold.test),
                            "epochs": epochs_run,
                            "elapsed_seconds": round(time.perf_counter() - began, 3),
                            **metrics,
                        }
                        performance_rows.append(row)
                        write_csv_with_retry(
                            pd.DataFrame([row]),
                            performance_path,
                            mode="a",
                            header=not performance_path.exists(),
                        )
                        completed.add(key)
                        del fitted
                    except Exception as exc:
                        failure = {
                            "atlas": atlas,
                            "model": model,
                            "seed": seed,
                            "fold": fold_index,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        failure_rows.append(failure)
                        write_csv_with_retry(
                            pd.DataFrame([failure]),
                            failure_path,
                            mode="a",
                            header=not failure_path.exists(),
                        )
                        if args.fail_fast:
                            raise
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    performance = pd.DataFrame(performance_rows).drop_duplicates(
        ["atlas", "model", "seed", "fold"], keep="last"
    )
    write_csv_with_retry(performance, performance_path)
    splits = pd.DataFrame(split_rows).drop_duplicates(
        ["atlas", "seed", "fold", "split", "subject_id"], keep="last"
    )
    write_csv_with_retry(splits, split_path)
    by_seed, summary = summarize(performance)
    write_csv_with_retry(by_seed, out_dir / "performance_by_seed.csv")
    write_csv_with_retry(summary, out_dir / "performance_summary.csv")
    failures = pd.DataFrame(failure_rows)
    if not failures.empty:
        failures = failures.drop_duplicates(
            ["atlas", "model", "seed", "fold"], keep="last"
        )
        write_csv_with_retry(failures, failure_path)
    elif failure_path.exists():
        failure_path.unlink()
    expected = len(atlases) * len(models) * len(seeds) * args.folds
    manifest = {
        "schema_version": "supplemental-differential-deep-models.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "differential_diagnosis",
        "discovery": "UCLA schizophrenia versus bipolar disorder",
        "models": models,
        "atlases": atlases,
        "seeds": seeds,
        "folds": args.folds,
        "same_splits_across_methods": True,
        "split_file": str(split_path),
        "cache_audits": audits,
        "external_validation_role": "owned by the closed-loop task runner",
        "n_model_folds_expected": expected,
        "n_model_folds": len(performance),
        "n_failed_model_folds": len(failures),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if len(performance) == expected and failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
