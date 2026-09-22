"""Run five-fold lightweight NeuroRuntime models on HCP lifespan cohorts.

The input is the existing leakage-safe NeuroRuntime FC cache under
``data/braingnn_input``. Outer folds are frozen once per atlas and seed and
reused across all six deep models. Results are appended incrementally so long
GPU sweeps can be resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPT_DIR = ROOT / "skills" / "brain_gnn" / "scripts"
for path in (ROOT, SKILL_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_lifespan_age import (  # noqa: E402
    LABEL_CSV,
    pt_path_for,
    train_braingnn,
    train_bnt,
    train_brainnetcnn,
    train_combraintf,
    train_lggnn,
)
from run_ibgnn_tune import train_ibgnn_safe  # noqa: E402


DEFAULT_OUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\lifespan_multimodel_expansion"
)
DEFAULT_MODELS = (
    "braingnn",
    "bnt",
    "brainnetcnn",
    "lggnn",
    "ibgnn",
    "combraintf",
)
DEFAULT_COMPATIBLE_ATLASES = (
    "aal3_166",
    "aal_116",
    "basc_122",
    "cc200",
    "cc400",
    "destrieux_148",
    "dk_112",
    "dosenbach_160",
    "eickhoff_zilles",
    "glasser_360",
    "harvard_oxford_cort",
    "harvard_oxford_merged",
    "harvard_oxford_sub",
    "msdl_39",
    "power_264",
    "schaefer_100_7net",
    "schaefer_200_7net",
    "schaefer_400_7net",
    "talairach_tournoux",
)
INCOMPATIBLE_CACHE_ATLASES: dict[str, str] = {}
HIGH_RESOLUTION_IBGNN_ATLASES = frozenset(
    {"cc400", "glasser_360", "power_264", "schaefer_400_7net"}
)
HIGH_RESOLUTION_IBGNN_MAX_EDGES = 4096
MODEL_PROFILES = {
    "braingnn": "native regression head",
    "bnt": "native regression head",
    "brainnetcnn": "compact channels 8/16/64; maximum 3 screening epochs",
    "lggnn": "native regression head",
    "ibgnn": (
        "LayerNorm + gradient clipping stable regression adapter; "
        "maximum 3 screening epochs and 4096 directed top-|FC| edges "
        "for declared high-edge-count atlases"
    ),
    "combraintf": "native regression head, batch size capped at 8",
}


def continuous_strata(values: pd.Series, n_splits: int) -> np.ndarray | None:
    numeric = pd.to_numeric(values, errors="coerce")
    for bins in range(min(10, max(2, len(numeric) // n_splits)), 1, -1):
        try:
            strata = pd.qcut(numeric, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        array = np.asarray(strata, dtype=int)
        counts = np.bincount(array)
        if len(counts) >= 2 and int(counts.min()) >= n_splits:
            return array
    return None


def make_fold_splits(
    frame: pd.DataFrame,
    *,
    folds: int,
    seed: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Create one shared train/validation/test split per outer fold."""

    strata = continuous_strata(frame["label"], folds)
    if strata is None:
        outer = KFold(n_splits=folds, shuffle=True, random_state=seed)
        raw_splits = outer.split(np.arange(len(frame)))
    else:
        outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        raw_splits = outer.split(np.arange(len(frame)), strata)
    splits: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for fold, (train_validation, test) in enumerate(raw_splits):
        inner_strata = continuous_strata(frame.iloc[train_validation]["label"], 2)
        train, validation = train_test_split(
            train_validation,
            test_size=0.20,
            random_state=seed + fold + 1,
            shuffle=True,
            stratify=inner_strata,
        )
        splits.append(
            (
                frame.iloc[train].reset_index(drop=True),
                frame.iloc[validation].reset_index(drop=True),
                frame.iloc[test].reset_index(drop=True),
            )
        )
    return splits


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_lifespan_atlases(input_root: Path) -> list[str]:
    atlases = []
    for directory in sorted(path for path in input_root.iterdir() if path.is_dir()):
        has_hcpa = next(directory.glob("sub-hcpa_*.pt"), None) is not None
        has_hcpya = next(
            (
                path
                for path in directory.glob("sub-*.pt")
                if "_" not in path.stem.removeprefix("sub-")
            ),
            None,
        ) is not None
        if has_hcpa and has_hcpya:
            atlases.append(directory.name)
    return atlases


def discover_atlases_for_datasets(
    input_root: Path,
    datasets: list[str] | tuple[str, ...],
) -> list[str]:
    """Return atlases containing at least one cache for every selected dataset."""
    selected = tuple(dict.fromkeys(str(value) for value in datasets))
    if not selected:
        raise ValueError("At least one dataset is required")

    atlases: list[str] = []
    for directory in sorted(path for path in input_root.iterdir() if path.is_dir()):
        available = True
        for dataset in selected:
            if dataset == "hcpya":
                found = next(
                    (
                        path
                        for path in directory.glob("sub-*.pt")
                        if "_" not in path.stem.removeprefix("sub-")
                    ),
                    None,
                )
            else:
                found = next(directory.glob(f"sub-{dataset}_*.pt"), None)
            if found is None:
                available = False
                break
        if available:
            atlases.append(directory.name)
    return atlases


def _roi_signature(path: Path) -> tuple[int, tuple[str, ...]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    matrix = torch.as_tensor(blob.get("fc_matrix", blob.get("node_features")))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"non-square FC matrix: {tuple(matrix.shape)}")
    n_roi = int(matrix.shape[0])
    names = tuple(str(value) for value in blob.get("roi_names", []))
    if len(names) != n_roi:
        raise ValueError(f"{len(names)} ROI names for {n_roi} rows")
    return n_roi, names


def filter_labels_for_atlas_contract(
    labels: pd.DataFrame,
    atlas: str,
    *,
    reference_dataset: str | None = None,
    path_resolver=pt_path_for,
) -> tuple[pd.DataFrame, dict[str, object]]:
    datasets = labels["dataset"].astype(str).drop_duplicates().tolist()
    if not datasets:
        raise ValueError("No labels available for atlas-contract validation")
    if reference_dataset is None:
        reference_dataset = "hcpya" if "hcpya" in datasets else datasets[0]
    if reference_dataset not in datasets:
        raise ValueError(
            f"Reference dataset {reference_dataset!r} is absent from labels"
        )

    reference = None
    for row in labels.loc[
        labels["dataset"].astype(str).eq(reference_dataset)
    ].itertuples(index=False):
        candidate = path_resolver(str(row.dataset), str(row.subject_id), atlas)
        if candidate.is_file():
            reference = candidate
            break
    if reference is None:
        raise RuntimeError(
            f"No {reference_dataset} contract reference for {atlas}"
        )
    expected_n_roi, expected_names = _roi_signature(reference)

    keep_indices: list[int] = []
    excluded: Counter[str] = Counter()
    before = labels["dataset"].value_counts().to_dict()
    for index, row in labels.iterrows():
        path = path_resolver(str(row["dataset"]), str(row["subject_id"]), atlas)
        if not path.is_file():
            excluded["missing_cache"] += 1
            continue
        try:
            n_roi, names = _roi_signature(path)
        except Exception:
            excluded["unreadable_cache"] += 1
            continue
        if n_roi != expected_n_roi:
            excluded["roi_dimension_mismatch"] += 1
            continue
        if names != expected_names:
            excluded["roi_name_or_order_mismatch"] += 1
            continue
        keep_indices.append(index)

    filtered = labels.loc[keep_indices].reset_index(drop=True)
    name_hash = hashlib.sha256("\n".join(expected_names).encode("utf-8")).hexdigest()
    audit = {
        "reference_dataset": reference_dataset,
        "reference_cache": str(reference),
        "expected_n_rois": expected_n_roi,
        "roi_names_sha256": name_hash,
        "before_by_dataset": before,
        "after_by_dataset": filtered["dataset"].value_counts().to_dict(),
        "excluded_by_reason": dict(excluded),
    }
    return filtered, audit


def write_csv_with_retry(
    frame: pd.DataFrame,
    path: Path,
    *,
    mode: str = "w",
    header: bool = True,
    attempts: int = 8,
) -> None:
    for attempt in range(attempts):
        try:
            frame.to_csv(path, mode=mode, header=header, index=False)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.25 * (2**attempt), 4.0))


def run_model(
    model: str,
    atlas: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_mean: float,
    y_std: float,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    common = dict(
        atlas=atlas,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        y_mean=y_mean,
        y_std=y_std,
        n_epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    if model == "braingnn":
        return train_braingnn(**common)
    if model == "bnt":
        return train_bnt(**common)
    if model == "brainnetcnn":
        return train_brainnetcnn(
            **{**common, "n_epochs": min(epochs, 3)},
            e2e_channels=8,
            e2n_channels=16,
            n2g_channels=64,
        )
    if model == "lggnn":
        return train_lggnn(**common)
    if model == "ibgnn":
        return train_ibgnn_safe(
            atlas=atlas,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            y_mean=y_mean,
            y_std=y_std,
            n_epochs=(
                min(epochs, 3)
                if atlas in HIGH_RESOLUTION_IBGNN_ATLASES
                else epochs
            ),
            batch_size=batch_size,
            lr=1e-4,
            wd=1e-5,
            hidden_dim=128,
            n_gnn_layers=2,
            normalize_input=True,
            grad_clip=1.0,
            seed=seed,
            max_edges_per_graph=(
                HIGH_RESOLUTION_IBGNN_MAX_EDGES
                if atlas in HIGH_RESOLUTION_IBGNN_ATLASES
                else None
            ),
        )
    if model == "combraintf":
        common["batch_size"] = min(batch_size, 8)
        return train_combraintf(**common)
    raise ValueError(f"Unsupported model: {model}")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "status" not in results.columns:
        return pd.DataFrame()
    successful = results[results["status"] == "complete"].copy()
    if successful.empty:
        return pd.DataFrame()
    if "fold" not in successful:
        successful["fold"] = 0
    if "test_mae_target_units" not in successful:
        successful["test_mae_target_units"] = successful["test_mae_yr"]
    else:
        successful["test_mae_target_units"] = pd.to_numeric(
            successful["test_mae_target_units"], errors="coerce"
        ).fillna(pd.to_numeric(successful["test_mae_yr"], errors="coerce"))
    by_seed = (
        successful.groupby(["atlas", "model", "seed"], as_index=False)
        .agg(
            test_mae_target_units=("test_mae_target_units", "mean"),
            test_mae_yr=("test_mae_yr", "mean"),
            test_mae_z=("test_mae_z", "mean"),
            n_folds=("fold", "nunique"),
            elapsed_sec=("elapsed_sec", "sum"),
        )
    )
    return (
        by_seed.groupby(["atlas", "model"], as_index=False)
        .agg(
            test_mae_target_units_mean=("test_mae_target_units", "mean"),
            test_mae_target_units_variance=("test_mae_target_units", "var"),
            test_mae_years_mean=("test_mae_yr", "mean"),
            test_mae_years_variance=("test_mae_yr", "var"),
            test_mae_z_mean=("test_mae_z", "mean"),
            test_mae_z_variance=("test_mae_z", "var"),
            n_seeds=("seed", "nunique"),
            elapsed_sec=("elapsed_sec", "sum"),
        )
        .sort_values(["test_mae_years_mean", "atlas", "model"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv", type=Path, default=LABEL_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--atlases", default="auto")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset IDs from the label CSV, or 'all'",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="20260805")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cohort-name", default="HCP-YA + HCP-Aging")
    parser.add_argument("--target-name", default="age_years")
    parser.add_argument(
        "--atlas-contract",
        default="identical ROI count, names, and order within each selected atlas",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    labels = pd.read_csv(args.label_csv, dtype={"subject_id": str})
    available_datasets = labels["dataset"].astype(str).drop_duplicates().tolist()
    if args.datasets.casefold() == "all":
        selected_datasets = available_datasets
    else:
        selected_datasets = parse_csv_list(args.datasets)
        unknown_datasets = set(selected_datasets) - set(available_datasets)
        if unknown_datasets:
            raise ValueError(
                f"Datasets absent from label CSV: {sorted(unknown_datasets)}"
            )
        labels = labels.loc[
            labels["dataset"].astype(str).isin(selected_datasets)
        ].reset_index(drop=True)
    if labels.empty:
        raise ValueError("No labels remain after dataset selection")

    if args.atlases == "auto":
        available = set(
            discover_atlases_for_datasets(
                ROOT / "data" / "braingnn_input",
                selected_datasets,
            )
        )
        atlases = [atlas for atlas in DEFAULT_COMPATIBLE_ATLASES if atlas in available]
    else:
        atlases = parse_csv_list(args.atlases)
    models = parse_csv_list(args.models)
    seeds = [int(seed) for seed in parse_csv_list(args.seeds)]
    unknown = set(models) - set(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    if args.smoke:
        atlases = atlases[:1]
        seeds = seeds[:1]
        models = models[:1]
        args.folds = min(args.folds, 2)
        args.epochs = min(args.epochs, 1)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "model_results.csv"
    failure_path = out_dir / "failed_jobs.csv"
    split_path = out_dir / "fold_assignments.csv"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Existing run: {out_dir}; pass --resume or use a new name")

    try:
        results = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    except pd.errors.EmptyDataError:
        results = []
    try:
        failures = pd.read_csv(failure_path).to_dict("records") if failure_path.exists() else []
    except pd.errors.EmptyDataError:
        failures = []
    done = {
        (str(row["atlas"]), str(row["model"]), int(row["seed"]), int(row["fold"]))
        for row in results
        if row.get("status") == "complete"
    }
    dataset_counts = labels["dataset"].value_counts().to_dict()
    excluded_cache_atlases = INCOMPATIBLE_CACHE_ATLASES if len(dataset_counts) > 1 else {}
    labels_by_atlas = {}
    atlas_contract_audits = {}
    for atlas in atlases:
        filtered, audit = filter_labels_for_atlas_contract(labels, atlas)
        labels_by_atlas[atlas] = filtered
        atlas_contract_audits[atlas] = audit
    started = time.perf_counter()
    split_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for atlas in atlases:
            fold_splits = make_fold_splits(
                labels_by_atlas[atlas],
                folds=args.folds,
                seed=seed,
            )
            for fold, (train_df, val_df, test_df) in enumerate(fold_splits):
                for split_name, frame in (
                    ("train", train_df),
                    ("validation", val_df),
                    ("test", test_df),
                ):
                    split_rows.extend(
                        {
                            "atlas": atlas,
                            "seed": seed,
                            "fold": fold,
                            "split": split_name,
                            "dataset": str(row.dataset),
                            "subject_id": str(row.subject_id),
                            "label": float(row.label),
                        }
                        for row in frame.itertuples(index=False)
                    )
                y_mean = float(train_df["label"].mean())
                y_std = float(train_df["label"].std())
                for model_index, model in enumerate(models):
                    key = (atlas, model, seed, fold)
                    if key in done:
                        print(
                            f"{atlas}/{model}/seed={seed}/fold={fold} already complete",
                            flush=True,
                        )
                        continue
                    job_started = time.perf_counter()
                    try:
                        output = run_model(
                            model,
                            atlas,
                            train_df,
                            val_df,
                            test_df,
                            y_mean,
                            y_std,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            seed=seed + fold * 101 + model_index * 1009,
                        )
                        if "error" in output:
                            raise RuntimeError(str(output["error"]))
                        row = {
                            "atlas": atlas,
                            "model": model,
                            "model_profile": MODEL_PROFILES[model],
                            "seed": seed,
                            "fold": fold,
                            "status": "complete",
                            "n_train": output["n_train"],
                            "n_validation": output["n_val"],
                            "n_test": output["n_test"],
                            "train_target_mean": y_mean,
                            "train_target_sd": y_std,
                            "val_mae_z": output["val_mae_z"],
                            "test_mae_z": output["test_mae_z"],
                            "test_mae_yr": output["test_mae_yr"],
                            "test_mae_target_units": output["test_mae_yr"],
                            "target_name": args.target_name,
                            "elapsed_sec": round(time.perf_counter() - job_started, 3),
                        }
                        if args.target_name == "age_years":
                            row["train_age_mean"] = y_mean
                            row["train_age_sd"] = y_std
                        results.append(row)
                        write_csv_with_retry(
                            pd.DataFrame(results).drop_duplicates(
                                ["atlas", "model", "seed", "fold"], keep="last"
                            ),
                            result_path,
                        )
                        done.add(key)
                        print(
                            f"{atlas}/{model}/seed={seed}/fold={fold}: "
                            f"MAE={row['test_mae_target_units']:.3f} "
                            f"{args.target_name} time={row['elapsed_sec']:.1f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        failure = {
                            "atlas": atlas,
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "elapsed_sec": round(time.perf_counter() - job_started, 3),
                        }
                        failures.append(failure)
                        write_csv_with_retry(
                            pd.DataFrame(failures).drop_duplicates(
                                ["atlas", "model", "seed", "fold"], keep="last"
                            ),
                            failure_path,
                        )
                        print(
                            f"{atlas}/{model}/seed={seed}/fold={fold} FAILED: {exc}",
                            flush=True,
                        )
                        if args.fail_fast:
                            raise

    result_frame = pd.DataFrame(results).drop_duplicates(
        ["atlas", "model", "seed", "fold"], keep="last"
    )
    write_csv_with_retry(result_frame, result_path)
    split_frame = pd.DataFrame(split_rows).drop_duplicates(
        ["atlas", "seed", "fold", "split", "dataset", "subject_id"],
        keep="last",
    )
    write_csv_with_retry(split_frame, split_path)
    summary = summarize(result_frame)
    if args.target_name != "age_years":
        summary = summary.drop(
            columns=[
                "test_mae_years_mean",
                "test_mae_years_variance",
            ],
            errors="ignore",
        )
    write_csv_with_retry(summary, out_dir / "model_summary.csv")
    failure_frame = pd.DataFrame(failures)
    if not failure_frame.empty:
        failure_frame = failure_frame.drop_duplicates(
            ["atlas", "model", "seed", "fold"], keep="last"
        )
        unresolved = [
            (str(row.atlas), str(row.model), int(row.seed), int(row.fold)) not in done
            for row in failure_frame.itertuples(index=False)
        ]
        failure_frame = failure_frame.loc[unresolved].reset_index(drop=True)
    if failure_frame.empty:
        failure_path.unlink(missing_ok=True)
    else:
        write_csv_with_retry(failure_frame, failure_path)
    expected_jobs = len(atlases) * len(models) * len(seeds) * args.folds
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cohort": args.cohort_name,
        "target": args.target_name,
        "selected_datasets": selected_datasets,
        "dataset_counts": dataset_counts,
        "atlases": atlases,
        "atlas_contract": args.atlas_contract,
        "atlas_contract_filtering": (
            "strict FC dimension and ordered ROI-name equality to one observed "
            "reference cache per atlas; applied to single- and multi-cohort runs"
        ),
        "atlas_contract_audits": atlas_contract_audits,
        "excluded_incompatible_cache_atlases": excluded_cache_atlases,
        "models": models,
        "model_profiles": {model: MODEL_PROFILES[model] for model in models},
        "seeds": seeds,
        "folds": args.folds,
        "epochs": args.epochs,
        "split": "five outer folds per seed with a training-only validation split",
        "same_splits_across_methods": True,
        "fold_assignments": str(split_path),
        "target_scaling": "training-split z score",
        "large_models_excluded": ["SwiFT", "NeuroSTORM", "CNN3D", "FM-APP"],
        "n_jobs_expected": expected_jobs,
        "n_jobs_complete": len(done),
        "n_jobs_failed": len(failure_frame),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if len(done) == expected_jobs and failure_frame.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
