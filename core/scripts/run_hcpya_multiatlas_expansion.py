"""Run a resumable HCP-YA multi-atlas cognition and brain-age sweep."""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from models.brain_age.correction import BrainAgeBiasCorrector
from models.common.data import make_splits
from models.common.metrics import regression_metrics
from models.cpm.cpm import CPM
from models.statistical_ml.estimators import make_estimator


DEFAULT_HCP_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\hcpya_preprocessed"
)
DEFAULT_PHENOTYPE = Path(r"\\192.168.3.61\data\Public Dataset\csv\hcp.csv")
DEFAULT_AGE = Path(
    r"\\192.168.3.61\data\Public Dataset\csv\HCP_1200_precise_age.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"\\192.168.3.61\data\Public Dataset\hcpya_multiatlas_model_expansion"
)
DEFAULT_MODELS = ("ols", "ridge", "elastic_net", "svm")
DEFAULT_CPM_THRESHOLDS = (0.001, 0.005, 0.01, 0.05)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def normalize_subject_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def discover_atlases(hcp_root: Path) -> list[str]:
    fc_root = hcp_root / "fc"
    return sorted(
        path.name
        for path in fc_root.iterdir()
        if path.is_dir() and (path / "correlation").is_dir()
    )


def load_target_maps(
    phenotype_path: Path,
    age_path: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    phenotype = pd.read_csv(
        phenotype_path,
        usecols=["Subject", "CogTotalComp_Unadj"],
        low_memory=False,
    ).copy()
    age = pd.read_csv(age_path, usecols=["subject", "age"], low_memory=False).copy()
    required_phenotype = {"Subject", "CogTotalComp_Unadj"}
    required_age = {"subject", "age"}
    if missing := required_phenotype - set(phenotype.columns):
        raise ValueError(f"Missing HCP phenotype columns: {sorted(missing)}")
    if missing := required_age - set(age.columns):
        raise ValueError(f"Missing HCP age columns: {sorted(missing)}")

    phenotype["subject_id"] = phenotype["Subject"].map(normalize_subject_id)
    phenotype["target"] = pd.to_numeric(
        phenotype["CogTotalComp_Unadj"], errors="coerce"
    )
    age["subject_id"] = age["subject"].map(normalize_subject_id)
    age["target"] = pd.to_numeric(age["age"], errors="coerce")
    behavior_map = dict(
        phenotype.dropna(subset=["target"])[["subject_id", "target"]].itertuples(
            index=False, name=None
        )
    )
    age_map = dict(
        age.dropna(subset=["target"])[["subject_id", "target"]].itertuples(
            index=False, name=None
        )
    )
    return behavior_map, age_map


def compact_fc_features(matrix: np.ndarray) -> np.ndarray:
    """Summarize global and ROI-wise FC while retaining atlas resolution."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"FC matrix must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("FC matrix contains NaN or Inf")
    n_roi = matrix.shape[0]
    if n_roi < 2:
        raise ValueError("FC matrix needs at least two ROIs")
    upper = matrix[np.triu_indices(n_roi, 1)]
    diagonal = np.diag(matrix)
    signed = (matrix.sum(axis=1) - diagonal) / (n_roi - 1)
    absolute = (np.abs(matrix).sum(axis=1) - np.abs(diagonal)) / (n_roi - 1)
    positive = np.where(matrix > 0, matrix, 0.0)
    negative = np.where(matrix < 0, matrix, 0.0)
    positive_count = np.maximum((matrix > 0).sum(axis=1) - (diagonal > 0), 1)
    negative_count = np.maximum((matrix < 0).sum(axis=1) - (diagonal < 0), 1)
    positive_mean = (positive.sum(axis=1) - np.maximum(diagonal, 0.0)) / positive_count
    negative_mean = (negative.sum(axis=1) - np.minimum(diagonal, 0.0)) / negative_count
    global_features = np.asarray(
        [
            np.mean(upper),
            np.mean(np.abs(upper)),
            np.std(upper),
            np.mean(upper > 0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [global_features, signed, absolute, positive_mean, negative_mean]
    ).astype(np.float32, copy=False)


def load_atlas_data(
    hcp_root: Path,
    atlas: str,
    target_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fc_dir = hcp_root / "fc" / atlas / "correlation"
    files = sorted(fc_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No correlation FC matrices found for {atlas}")
    suffix = f"_{atlas}_correlation.npy"
    subject_ids: list[str] = []
    compact: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    skipped: list[str] = []
    n_roi: int | None = None
    triu: tuple[np.ndarray, np.ndarray] | None = None
    for path in files:
        subject_id = normalize_subject_id(
            path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
        )
        if subject_id not in target_ids:
            continue
        try:
            matrix = np.load(path, allow_pickle=False)
            if n_roi is None:
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                    raise ValueError(f"FC matrix must be square, got {matrix.shape}")
                n_roi = int(matrix.shape[0])
                triu = np.triu_indices(n_roi, 1)
            if matrix.shape != (n_roi, n_roi):
                raise ValueError(f"Expected {(n_roi, n_roi)}, got {matrix.shape}")
            features = compact_fc_features(matrix)
            assert triu is not None
            edge_vector = np.asarray(matrix[triu], dtype=np.float32)
            subject_ids.append(subject_id)
            compact.append(features)
            edges.append(edge_vector)
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")
    if not subject_ids or n_roi is None:
        raise RuntimeError(f"No usable HCP-YA subjects for atlas {atlas}")
    return (
        np.asarray(subject_ids, dtype=str),
        np.stack(compact),
        np.stack(edges),
        {
            "atlas": atlas,
            "fc_dir": str(fc_dir),
            "files_seen": len(files),
            "subjects_loaded": len(subject_ids),
            "n_rois": n_roi,
            "n_compact_features": int(compact[0].shape[0]),
            "n_edges": int(edges[0].shape[0]),
            "skipped_files": skipped,
        },
    )


def align_target(
    subject_ids: np.ndarray,
    X_compact: np.ndarray,
    X_edges: np.ndarray,
    target_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray([subject_id in target_map for subject_id in subject_ids])
    ids = subject_ids[keep]
    y = np.asarray([target_map[subject_id] for subject_id in ids], dtype=np.float64)
    return ids, X_compact[keep], X_edges[keep], y


def metric_moments(rows: Iterable[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    frame = pd.DataFrame(list(rows))
    means = {column: float(frame[column].mean()) for column in frame}
    variances = {
        column: float(frame[column].var(ddof=1)) if len(frame) > 1 else 0.0
        for column in frame
    }
    return means, variances


def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    model_name: str,
    seed: int,
    folds: int,
    brain_age: bool,
) -> tuple[dict[str, Any], pd.DataFrame, list[Any]]:
    splits = make_splits(y, "regression", folds, seed)
    raw = np.empty(len(y), dtype=np.float64)
    corrected = np.empty(len(y), dtype=np.float64) if brain_age else None
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows: list[dict[str, float]] = []
    checkpoints: list[Any] = []
    for fold, (train, test) in enumerate(splits):
        estimator_params: dict[str, Any] = {}
        if model_name == "elastic_net":
            estimator_params = {
                "scale_target": True,
                "alpha": 0.05,
                "max_iter": 20000,
            }
        estimator = make_estimator(
            model_name,
            "regression",
            seed + fold,
            **estimator_params,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(X[train], y[train])
        convergence_messages = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        if convergence_messages:
            raise RuntimeError(
                f"{model_name} did not converge in fold {fold}: "
                f"{convergence_messages[-1]}"
            )
        train_pred = estimator.predict(X[train])
        test_pred = estimator.predict(X[test])
        raw[test] = test_pred
        fold_ids[test] = fold
        row = {"fold": fold, **regression_metrics(y[test], test_pred)}
        if brain_age:
            corrector = BrainAgeBiasCorrector().fit(y[train], train_pred)
            corrected_pred = corrector.transform(y[test], test_pred)
            assert corrected is not None
            corrected[test] = corrected_pred
            row.update(
                {
                    f"bias_corrected_{key}": value
                    for key, value in regression_metrics(y[test], corrected_pred).items()
                }
            )
            checkpoints.append((estimator, corrector))
        else:
            checkpoints.append(estimator)
        fold_rows.append(row)

    aggregate = {f"raw_{key}": value for key, value in regression_metrics(y, raw).items()}
    predictions = {
        "subject_id": subject_ids,
        "fold": fold_ids,
        "target": y,
        "prediction_raw": raw,
    }
    if brain_age:
        assert corrected is not None
        aggregate.update(
            {
                f"bias_corrected_{key}": value
                for key, value in regression_metrics(y, corrected).items()
            }
        )
        aggregate["brain_pad_mean"] = float(np.mean(corrected - y))
        aggregate["brain_pad_sd"] = float(np.std(corrected - y, ddof=1))
        predictions["prediction_bias_corrected"] = corrected
        predictions["brain_pad"] = corrected - y
    fold_mean, fold_variance = metric_moments(fold_rows)
    metrics = {
        "aggregate": aggregate,
        "fold_mean": fold_mean,
        "fold_variance": fold_variance,
        "n_samples": len(y),
        "n_features": int(X.shape[1]),
        "n_folds": len(splits),
    }
    return metrics, pd.DataFrame(predictions), checkpoints


def run_cpm_cv(
    X_edges: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    p_threshold: float,
    seed: int,
    folds: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[CPM]]:
    splits = make_splits(y, "regression", folds, seed)
    predictions = np.empty(len(y), dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows: list[dict[str, float]] = []
    checkpoints: list[CPM] = []
    selected_counts: list[int] = []
    for fold, (train, test) in enumerate(splits):
        model = CPM("regression", p_threshold=p_threshold).fit(X_edges[train], y[train])
        fold_prediction = model.predict(X_edges[test])
        predictions[test] = fold_prediction
        fold_ids[test] = fold
        fold_rows.append({"fold": fold, **regression_metrics(y[test], fold_prediction)})
        selected_counts.append(
            int(model.positive_mask_.sum() + model.negative_mask_.sum())
        )
        checkpoints.append(model)
    aggregate = {
        f"raw_{key}": value for key, value in regression_metrics(y, predictions).items()
    }
    fold_mean, fold_variance = metric_moments(fold_rows)
    metrics = {
        "aggregate": aggregate,
        "fold_mean": fold_mean,
        "fold_variance": fold_variance,
        "selected_edges_mean": float(np.mean(selected_counts)),
        "selected_edges_variance": float(np.var(selected_counts, ddof=1))
        if len(selected_counts) > 1
        else 0.0,
        "n_samples": len(y),
        "n_features": int(X_edges.shape[1]),
        "n_folds": len(splits),
    }
    frame = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "fold": fold_ids,
            "target": y,
            "prediction_raw": predictions,
        }
    )
    return metrics, frame, checkpoints


def task_record(
    *,
    atlas: str,
    target: str,
    model: str,
    seed: int,
    metrics: dict[str, Any],
    elapsed_sec: float,
    resumed: bool,
    output_dir: Path,
) -> dict[str, Any]:
    aggregate = metrics["aggregate"]
    primary_name = "raw_mae" if target == "brain_age" else "raw_pearson_r"
    return {
        "atlas": atlas,
        "target": target,
        "model": model,
        "seed": seed,
        "status": "completed",
        "resumed": resumed,
        "elapsed_sec": round(elapsed_sec, 3),
        "n_samples": metrics["n_samples"],
        "n_features": metrics["n_features"],
        "primary_metric": primary_name,
        "primary_value": aggregate[primary_name],
        "metric_direction": "lower" if target == "brain_age" else "higher",
        **aggregate,
        "output_dir": str(output_dir),
    }


def persist_task(
    output_dir: Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    checkpoints: list[Any],
    save_checkpoints: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "metrics.json", metrics)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    if save_checkpoints:
        joblib.dump(checkpoints, output_dir / "checkpoint.joblib", compress=3)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    completed = records[records["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    return (
        completed.groupby(
            ["atlas", "target", "model", "primary_metric", "metric_direction"]
        )["primary_value"]
        .agg(mean="mean", variance="var", n_runs="count")
        .reset_index()
        .sort_values(["target", "atlas", "mean"])
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hcp-root", type=Path, default=DEFAULT_HCP_ROOT)
    parser.add_argument("--phenotype", type=Path, default=DEFAULT_PHENOTYPE)
    parser.add_argument("--age", type=Path, default=DEFAULT_AGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--atlases", default="all")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--cpm-thresholds",
        default=",".join(str(value) for value in DEFAULT_CPM_THRESHOLDS),
    )
    parser.add_argument("--seeds", default="20260805,20260806,20260807")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    models = parse_csv(args.models)
    unknown_models = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown_models:
        raise ValueError(f"Unknown regression models: {unknown_models}")
    thresholds = [float(value) for value in parse_csv(args.cpm_thresholds)]
    seeds = [int(value) for value in parse_csv(args.seeds)]
    available_atlases = discover_atlases(args.hcp_root)
    atlases = available_atlases if args.atlases == "all" else parse_csv(args.atlases)
    missing_atlases = sorted(set(atlases) - set(available_atlases))
    if missing_atlases:
        raise ValueError(f"Unavailable HCP-YA atlases: {missing_atlases}")
    if args.smoke:
        atlases = atlases[:1]
        models = ["ridge"]
        thresholds = [0.05]
        seeds = seeds[:1]

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    behavior_map, age_map = load_target_maps(args.phenotype, args.age)
    target_ids = set(behavior_map) | set(age_map)
    n_tasks = len(atlases) * len(seeds) * (2 * len(models) + len(thresholds))
    write_json(
        output_dir / "sweep_plan.json",
        {
            "created_at": utc_now(),
            "hcp_root": str(args.hcp_root),
            "phenotype": str(args.phenotype),
            "age": str(args.age),
            "atlases": atlases,
            "models": models,
            "cpm_thresholds": thresholds,
            "seeds": seeds,
            "folds": args.folds,
            "n_tasks": n_tasks,
            "representation": {
                "statistical_models": "global FC plus ROI signed/absolute/positive/negative strength",
                "cpm": "all unique undirected FC edges with fold-local edge selection",
            },
            "excluded_large_models": ["SwiFT", "NeuroSTORM", "FM-APP"],
        },
    )

    records: list[dict[str, Any]] = []
    status_path = output_dir / "task_status.jsonl"
    atlas_audits: list[dict[str, Any]] = []
    task_index = 0
    for atlas_index, atlas in enumerate(atlases, 1):
        print(f"[atlas {atlas_index}/{len(atlases)}] loading {atlas}", flush=True)
        try:
            subject_ids, X_compact, X_edges, atlas_audit = load_atlas_data(
                args.hcp_root, atlas, target_ids
            )
            atlas_audits.append(atlas_audit)
            write_json(output_dir / "atlas_manifests" / f"{atlas}.json", atlas_audit)
            target_specs = {
                "cognition": (*align_target(subject_ids, X_compact, X_edges, behavior_map), False),
                "brain_age": (*align_target(subject_ids, X_compact, X_edges, age_map), True),
            }
            for target, (ids, compact, edges, y, is_brain_age) in target_specs.items():
                for seed in seeds:
                    for model in models:
                        task_index += 1
                        task_dir = output_dir / atlas / target / model / f"seed_{seed}"
                        metrics_path = task_dir / "metrics.json"
                        print(
                            f"[{task_index}/{n_tasks}] {atlas} {target} {model} seed={seed}",
                            flush=True,
                        )
                        try:
                            if args.resume and metrics_path.is_file():
                                metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                                record = task_record(
                                    atlas=atlas,
                                    target=target,
                                    model=model,
                                    seed=seed,
                                    metrics=metrics,
                                    elapsed_sec=0.0,
                                    resumed=True,
                                    output_dir=task_dir,
                                )
                            else:
                                started = time.perf_counter()
                                metrics, predictions, checkpoints = run_regression_cv(
                                    compact,
                                    y,
                                    ids,
                                    model_name=model,
                                    seed=seed,
                                    folds=args.folds,
                                    brain_age=is_brain_age,
                                )
                                elapsed = time.perf_counter() - started
                                persist_task(
                                    task_dir,
                                    config={
                                        "atlas": atlas,
                                        "target": target,
                                        "model": model,
                                        "seed": seed,
                                        "folds": args.folds,
                                        "representation": "compact_roi_strength",
                                    },
                                    metrics=metrics,
                                    predictions=predictions,
                                    checkpoints=checkpoints,
                                    save_checkpoints=args.save_checkpoints,
                                )
                                record = task_record(
                                    atlas=atlas,
                                    target=target,
                                    model=model,
                                    seed=seed,
                                    metrics=metrics,
                                    elapsed_sec=elapsed,
                                    resumed=False,
                                    output_dir=task_dir,
                                )
                            records.append(record)
                        except Exception as exc:
                            record = {
                                "atlas": atlas,
                                "target": target,
                                "model": model,
                                "seed": seed,
                                "status": "failed",
                                "error": str(exc),
                                "output_dir": str(task_dir),
                            }
                            records.append(record)
                            if args.fail_fast:
                                raise
                        with status_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if target == "cognition":
                        for threshold in thresholds:
                            task_index += 1
                            model = f"cpm_p{threshold:g}"
                            task_dir = output_dir / atlas / target / model / f"seed_{seed}"
                            metrics_path = task_dir / "metrics.json"
                            print(
                                f"[{task_index}/{n_tasks}] {atlas} {target} {model} seed={seed}",
                                flush=True,
                            )
                            try:
                                if args.resume and metrics_path.is_file():
                                    metrics = json.loads(
                                        metrics_path.read_text(encoding="utf-8-sig")
                                    )
                                    record = task_record(
                                        atlas=atlas,
                                        target=target,
                                        model=model,
                                        seed=seed,
                                        metrics=metrics,
                                        elapsed_sec=0.0,
                                        resumed=True,
                                        output_dir=task_dir,
                                    )
                                else:
                                    started = time.perf_counter()
                                    metrics, predictions, checkpoints = run_cpm_cv(
                                        edges,
                                        y,
                                        ids,
                                        p_threshold=threshold,
                                        seed=seed,
                                        folds=args.folds,
                                    )
                                    elapsed = time.perf_counter() - started
                                    persist_task(
                                        task_dir,
                                        config={
                                            "atlas": atlas,
                                            "target": target,
                                            "model": model,
                                            "seed": seed,
                                            "folds": args.folds,
                                            "p_threshold": threshold,
                                            "representation": "all_unique_fc_edges",
                                        },
                                        metrics=metrics,
                                        predictions=predictions,
                                        checkpoints=checkpoints,
                                        save_checkpoints=args.save_checkpoints,
                                    )
                                    record = task_record(
                                        atlas=atlas,
                                        target=target,
                                        model=model,
                                        seed=seed,
                                        metrics=metrics,
                                        elapsed_sec=elapsed,
                                        resumed=False,
                                        output_dir=task_dir,
                                    )
                                records.append(record)
                            except Exception as exc:
                                record = {
                                    "atlas": atlas,
                                    "target": target,
                                    "model": model,
                                    "seed": seed,
                                    "status": "failed",
                                    "error": str(exc),
                                    "output_dir": str(task_dir),
                                }
                                records.append(record)
                                if args.fail_fast:
                                    raise
                            with status_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            atlas_audits.append({"atlas": atlas, "status": "failed", "error": str(exc)})
            if args.fail_fast:
                raise
        finally:
            subject_ids = X_compact = X_edges = target_specs = None
            gc.collect()

    result_frame = pd.DataFrame(records)
    result_frame.to_csv(output_dir / "model_sweep_results.csv", index=False)
    summary = summarize(result_frame)
    summary.to_csv(output_dir / "model_sweep_summary.csv", index=False)
    audit = {
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "n_atlases_requested": len(atlases),
        "n_atlases_loaded": sum("n_rois" in row for row in atlas_audits),
        "n_tasks_planned": n_tasks,
        "n_tasks_recorded": len(records),
        "n_completed": sum(row["status"] == "completed" for row in records),
        "n_failed": sum(row["status"] != "completed" for row in records),
        "atlas_audits": atlas_audits,
    }
    write_json(output_dir / "model_sweep_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0 if audit["n_failed"] == 0 and len(records) == n_tasks else 1


if __name__ == "__main__":
    raise SystemExit(main())
