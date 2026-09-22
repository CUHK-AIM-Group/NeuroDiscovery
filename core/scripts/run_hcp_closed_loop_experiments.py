"""Run the HCP connectome-behavior and brain-age closed-loop studies.

The primary search space uses compact, deterministic summaries of the same
19-atlas NeuroRuntime FC caches for every cohort.  Outcome-blind candidate
tables are physically separated from internal and external outcomes before the
shared random-walk and NeuroDiscovery benchmark is run.

Connectome-behavior discovery uses the age-adjusted HCP-YA total cognition
composite.  ADHD200 typically-developing participants provide an independent
 IQ replication cohort. Brain-age discovery uses HCP-YA alone. Frozen models are transferred only to
 healthy controls within the discovery age-support interval. Large voxel
 foundation models are intentionally out of scope for this closure pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, t
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.multitest import multipletests
import torch

from core.scripts.case_study_candidate_tables import (
    attach_candidate_ids,
    export_table_bundle,
    score_public_candidates,
)
from core.scripts.case_study_closed_loop import run_benchmark
from core.scripts.case_study_closed_loop_specs import protocol_for


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "data" / "braingnn_input"
DEFAULT_DATA_ROOT = Path(r"\\192.168.3.61\data\Public Dataset")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "case_study_closed_loop_v1"
DEFAULT_KG = ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"
DEFAULT_DEEP_SWEEP = (
    DEFAULT_DATA_ROOT
    / "lifespan_multimodel_expansion"
    / "full19x6_contract_seed20260805"
)

DEFAULT_ATLASES = (
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
FEATURE_FAMILIES = (
    "fc_edge_projection",
    "node_connectivity_profile",
    "graph_topology_summary",
)
REGRESSION_MODELS = ("ridge", "elastic_net", "svm")
EXTERNAL_AGE_COHORTS = ("ucla", "cobre", "hcpep", "adhd200")
BRAIN_AGE_MIN_R_CI_LOW = 0.40
BRAIN_AGE_MIN_NORMALIZED_MAE_REDUCTION = 0.10
BRAIN_AGE_FDR_P_COLUMN = "permutation_p_exact"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _cache_path(cache_root: Path, dataset: str, subject_id: str, atlas: str) -> Path:
    subject = str(subject_id).strip().removeprefix("sub-")
    if dataset == "hcpya":
        return cache_root / atlas / f"sub-{subject}.pt"
    direct = cache_root / atlas / f"sub-{dataset}_{subject}.pt"
    if direct.is_file() or dataset != "adhd200":
        return direct
    return cache_root / atlas / f"sub-{dataset}_{subject.zfill(7)}.pt"


def _edge_projection(values: np.ndarray, *, seed: int, width: int = 256) -> np.ndarray:
    """Deterministic signed CountSketch of upper-triangle FC edges."""

    values = np.asarray(values, dtype=np.float64)
    index = np.arange(values.size, dtype=np.uint64)
    mixed = index * np.uint64(11400714819323198485) + np.uint64(seed)
    buckets = np.asarray(mixed % np.uint64(width), dtype=np.int64)
    signs = np.where(((mixed >> np.uint64(32)) & np.uint64(1)) == 0, 1.0, -1.0)
    projected = np.zeros(width, dtype=np.float64)
    counts = np.zeros(width, dtype=np.float64)
    np.add.at(projected, buckets, values * signs)
    np.add.at(counts, buckets, 1.0)
    return projected / np.sqrt(np.maximum(counts, 1.0))


def extract_fc_views(fc_matrix: np.ndarray, *, atlas: str) -> dict[str, np.ndarray]:
    """Return three compact, outcome-blind whole-connectome representations."""

    matrix = np.asarray(fc_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"FC matrix must be square, got {matrix.shape}")
    matrix = np.tanh(matrix)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    projection = _edge_projection(
        upper,
        seed=stable_seed("fc-edge-projection", atlas),
    )

    signed_strength = matrix.mean(axis=1)
    absolute_strength = np.abs(matrix).mean(axis=1)
    variability = matrix.std(axis=1)
    node_profile = np.concatenate(
        [signed_strength, absolute_strength, variability]
    )

    summaries: list[float] = []
    summaries.extend(
        np.quantile(
            upper,
            (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
        ).tolist()
    )
    summaries.extend(
        [
            float(upper.mean()),
            float(upper.std()),
            float(np.abs(upper).mean()),
            float(np.mean(upper > 0)),
            float(np.mean(upper < 0)),
        ]
    )
    absolute = np.abs(matrix)
    for threshold in (0.10, 0.20, 0.30, 0.40, 0.50):
        degree = (absolute >= threshold).sum(axis=1) / max(matrix.shape[0] - 1, 1)
        summaries.extend(
            [
                float(degree.mean()),
                float(degree.std()),
                *np.quantile(degree, (0.10, 0.25, 0.50, 0.75, 0.90)).tolist(),
            ]
        )
    return {
        "fc_edge_projection": projection.astype(np.float32),
        "node_connectivity_profile": node_profile.astype(np.float32),
        "graph_topology_summary": np.asarray(summaries, dtype=np.float32),
    }


def _load_feature_cache(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    payload = np.load(path, allow_pickle=False)
    subjects = [str(value) for value in payload["subject_ids"].tolist()]
    views = {name: np.asarray(payload[name]) for name in FEATURE_FAMILIES}
    return subjects, views


def load_feature_views(
    *,
    cache_root: Path,
    work_dir: Path,
    dataset: str,
    atlas: str,
    subject_ids: Sequence[str],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    """Load FC summaries, caching only outcome-blind arrays on disk."""

    feature_dir = work_dir / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{dataset}_{atlas}.npz"
    requested = [str(value).strip().removeprefix("sub-") for value in subject_ids]
    requested_hash = hashlib.sha256("\n".join(requested).encode("utf-8")).hexdigest()
    audit_path = path.with_suffix(".json")
    if path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("requested_subjects_sha256") == requested_hash:
            subjects, views = _load_feature_cache(path)
            return subjects, views, audit

    subjects: list[str] = []
    collected: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_FAMILIES}
    excluded: dict[str, int] = {}
    roi_names_hash = ""
    expected_rois: int | None = None
    for subject in requested:
        source = _cache_path(cache_root, dataset, subject, atlas)
        if not source.is_file():
            excluded["missing_cache"] = excluded.get("missing_cache", 0) + 1
            continue
        try:
            blob = torch.load(source, map_location="cpu", weights_only=False)
            fc = torch.as_tensor(blob.get("fc_matrix", blob.get("node_features"))).numpy()
            names = tuple(str(value) for value in blob.get("roi_names", []))
            if expected_rois is None:
                expected_rois = int(fc.shape[0])
                roi_names_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
            if fc.shape != (expected_rois, expected_rois) or len(names) != expected_rois:
                excluded["atlas_contract_mismatch"] = excluded.get(
                    "atlas_contract_mismatch", 0
                ) + 1
                continue
            views = extract_fc_views(fc, atlas=atlas)
        except Exception:
            excluded["unreadable_cache"] = excluded.get("unreadable_cache", 0) + 1
            continue
        subjects.append(subject)
        for name in FEATURE_FAMILIES:
            collected[name].append(views[name])
    if not subjects:
        raise RuntimeError(f"No usable {dataset}/{atlas} FC caches")
    arrays = {name: np.stack(values) for name, values in collected.items()}
    np.savez_compressed(
        path,
        subject_ids=np.asarray(subjects),
        **arrays,
    )
    audit = {
        "created_at": utc_now(),
        "dataset": dataset,
        "atlas": atlas,
        "requested_subjects": len(requested),
        "usable_subjects": len(subjects),
        "requested_subjects_sha256": requested_hash,
        "expected_n_rois": expected_rois,
        "roi_names_sha256": roi_names_hash,
        "excluded_by_reason": excluded,
        "feature_dimensions": {name: int(value.shape[1]) for name, value in arrays.items()},
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return subjects, arrays, audit


def _label_table(path: Path, *, dataset: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    if dataset is not None and "dataset" in frame:
        frame = frame.loc[frame["dataset"].astype(str).eq(dataset)].copy()
    frame["subject_id"] = frame["subject_id"].astype(str).str.removeprefix("sub-")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    return frame.dropna(subset=["label"]).drop_duplicates("subject_id")


def hcpya_cognition_metadata(data_root: Path) -> pd.DataFrame:
    source = pd.read_csv(data_root / "csv" / "hcp.csv", dtype={"Subject": str})
    age = _label_table(ROOT / "data" / "labels" / "lifespan_age_labels.csv", dataset="hcpya")
    sex = _label_table(ROOT / "data" / "labels" / "lifespan_sex_labels.csv", dataset="hcpya")
    frame = pd.DataFrame(
        {
            "subject_id": source["Subject"].astype(str),
            "label": pd.to_numeric(source["CogTotalComp_AgeAdj"], errors="coerce"),
        }
    )
    frame = frame.merge(age.rename(columns={"label": "age"}), on="subject_id", how="inner")
    frame = frame.merge(sex.rename(columns={"label": "sex"}), on="subject_id", how="inner")
    return frame.dropna().drop_duplicates("subject_id").reset_index(drop=True)


def adhd200_iq_controls(data_root: Path) -> pd.DataFrame:
    source = pd.read_csv(
        data_root / "adhd200_preprocessed" / "metadata" / "adhd200-rest.csv",
        dtype={"subject_id": str},
    )
    full4 = pd.to_numeric(source["Full4 IQ"], errors="coerce")
    full2 = pd.to_numeric(source["Full2 IQ"], errors="coerce")
    verbal = pd.to_numeric(source["Verbal IQ"], errors="coerce")
    performance = pd.to_numeric(source["Performance IQ"], errors="coerce")
    label = full4.fillna(full2).fillna((verbal + performance) / 2.0)
    qc = (
        pd.to_numeric(source["QC_Athena"], errors="coerce").eq(1)
        | pd.to_numeric(source["QC_NIAK"], errors="coerce").eq(1)
    )
    controls = source["DX"].astype(str).str.strip().eq("0")
    frame = pd.DataFrame(
        {
            "subject_id": source["subject_id"].astype(str),
            "label": label,
            "age": pd.to_numeric(source["Age"], errors="coerce"),
            "sex": pd.to_numeric(source["Gender"], errors="coerce"),
            "site": source["Site"].astype(str),
        }
    )
    return frame.loc[qc & controls].dropna().drop_duplicates("subject_id").reset_index(drop=True)


def lifespan_age_metadata() -> pd.DataFrame:
    age = _label_table(ROOT / "data" / "labels" / "lifespan_age_labels.csv")
    sex = _label_table(ROOT / "data" / "labels" / "lifespan_sex_labels.csv")
    age = age.rename(columns={"label": "age"})
    sex = sex.rename(columns={"label": "sex"})
    frame = age.merge(sex[["subject_id", "dataset", "sex"]], on=["subject_id", "dataset"])
    frame["label"] = frame["age"]
    return frame.reset_index(drop=True)


def _healthy_subjects(data_root: Path, dataset: str) -> set[str]:
    if dataset == "ucla":
        frame = pd.read_csv(data_root / "ucla_preprocessed" / "metadata" / "ucla-rest.csv")
        mask = frame["diagnosis"].astype(str).str.strip().str.casefold().eq("control")
    elif dataset == "cobre":
        frame = pd.read_csv(data_root / "cobre_preprocessed" / "metadata" / "cobre-rest.csv")
        mask = (
            frame["dx"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .str.replace("_", " ", regex=False)
            .eq("no known disorder")
        )
    elif dataset == "hcpep":
        frame = pd.read_csv(data_root / "hcpep_preprocessed" / "metadata" / "hcpep-rest.csv")
        mask = (
            frame["phenotype_description"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("in good health")
        )
    elif dataset == "adhd200":
        frame = pd.read_csv(data_root / "adhd200_preprocessed" / "metadata" / "adhd200-rest.csv")
        mask = frame["DX"].astype(str).str.strip().eq("0")
    else:
        raise KeyError(dataset)
    return set(frame.loc[mask, "subject_id"].astype(str).str.removeprefix("sub-"))


def external_age_metadata(data_root: Path, dataset: str) -> pd.DataFrame:
    age = _label_table(ROOT / "data" / "labels" / f"{dataset}_age_labels.csv")
    sex = _label_table(ROOT / "data" / "labels" / f"{dataset}_sex_labels.csv")
    frame = age.rename(columns={"label": "age"}).merge(
        sex.rename(columns={"label": "sex"}), on="subject_id", how="inner"
    )
    frame = frame.loc[frame["subject_id"].isin(_healthy_subjects(data_root, dataset))]
    frame["label"] = frame["age"]
    return frame.dropna().drop_duplicates("subject_id").reset_index(drop=True)


def align_features(
    subjects: Sequence[str],
    views: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    feature_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    index = {str(subject): idx for idx, subject in enumerate(subjects)}
    rows = metadata.loc[metadata["subject_id"].astype(str).isin(index)].copy()
    rows["_index"] = rows["subject_id"].astype(str).map(index)
    rows = rows.sort_values("_index")
    idx = rows["_index"].astype(int).to_numpy()
    X = np.asarray(views[feature_family][idx], dtype=np.float64)
    y = rows["label"].to_numpy(dtype=float)
    covariates = rows[["age", "sex"]].to_numpy(dtype=float)
    return X, y, covariates, rows["subject_id"].astype(str).tolist()


def _regressor_pipeline(
    model: str,
    *,
    n_features: int,
    n_train: int,
    seed: int,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    n_components = max(1, min(64, n_features, n_train - 2))
    if model == "ridge":
        regressor = Ridge()
    elif model == "elastic_net":
        regressor = ElasticNet(max_iter=5000, random_state=seed)
    elif model == "svm":
        regressor = SVR(kernel="linear")
    else:
        raise ValueError(f"Unknown regression model: {model}")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("model", regressor),
        ]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline


def _parameter_grid(model: str) -> dict[str, list[Any]]:
    if model == "ridge":
        return {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    if model == "elastic_net":
        return {
            "model__alpha": [0.001, 0.01, 0.1],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        }
    if model == "svm":
        return {"model__C": [0.1, 1.0, 10.0]}
    raise ValueError(model)


def _mode_params(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    serialized = [json.dumps(value, sort_keys=True) for value in values]
    return json.loads(max(set(serialized), key=serialized.count)) if serialized else {}


def _mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1:
        return mean, mean, mean
    half = float(t.ppf(0.975, array.size - 1) * array.std(ddof=1) / math.sqrt(array.size))
    return mean, mean - half, mean + half


def _correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def continuous_strata(
    values: np.ndarray,
    *,
    n_splits: int,
    max_bins: int = 10,
) -> np.ndarray | None:
    """Create quantile strata with enough observations for every fold."""

    values = np.asarray(values, dtype=float)
    for bins in range(min(max_bins, max(2, len(values) // n_splits)), 1, -1):
        try:
            strata = pd.qcut(values, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        strata = np.asarray(strata, dtype=int)
        counts = np.bincount(strata)
        if len(counts) >= 2 and int(counts.min()) >= n_splits:
            return strata
    return None


def age_stratified_splits(
    y: np.ndarray,
    *,
    n_splits: int,
    repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = continuous_strata(y, n_splits=n_splits)
    if strata is None:
        return list(
            RepeatedKFold(
                n_splits=n_splits,
                n_repeats=repeats,
                random_state=seed,
            ).split(np.zeros(len(y)))
        )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        splits.extend(splitter.split(np.zeros(len(y)), strata))
    return splits


def repeated_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    seed: int,
    repeats: int,
    stratify_continuous: bool = False,
    split_seed: int | None = None,
) -> dict[str, Any]:
    frozen_split_seed = seed if split_seed is None else int(split_seed)
    splits = (
        age_stratified_splits(
            y,
            n_splits=5,
            repeats=repeats,
            seed=frozen_split_seed,
        )
        if stratify_continuous
        else list(
            RepeatedKFold(
                n_splits=5,
                n_repeats=repeats,
                random_state=frozen_split_seed,
            ).split(X)
        )
    )
    fold_rows: list[dict[str, float]] = []
    params: list[dict[str, Any]] = []
    prediction_sum = np.zeros(len(y), dtype=np.float64)
    prediction_count = np.zeros(len(y), dtype=np.int64)
    for fold, (train, test) in enumerate(splits, start=1):
        pipeline = _regressor_pipeline(
            model,
            n_features=X.shape[1],
            n_train=len(train),
            seed=seed + fold,
        )
        inner_strata = (
            continuous_strata(y[train], n_splits=3, max_bins=5)
            if stratify_continuous
            else None
        )
        if inner_strata is None:
            inner_cv: Any = KFold(
                n_splits=3,
                shuffle=True,
                random_state=frozen_split_seed + fold,
            )
        else:
            inner_cv = list(
                StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=frozen_split_seed + fold,
                ).split(np.zeros(len(train)), inner_strata)
            )
        search = GridSearchCV(
            pipeline,
            _parameter_grid(model),
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=1,
        )
        search.fit(X[train], y[train])
        prediction = search.best_estimator_.predict(X[test])
        baseline = np.full(len(test), float(np.mean(y[train])))
        fold_rows.append(
            {
                "r": _correlation(y[test], prediction),
                "mae": float(np.mean(np.abs(y[test] - prediction))),
                "baseline_mae": float(np.mean(np.abs(y[test] - baseline))),
            }
        )
        params.append(dict(search.best_params_))
        prediction_sum[test] += prediction
        prediction_count[test] += 1
    frame = pd.DataFrame(fold_rows)
    r_mean, r_low, r_high = _mean_ci(frame["r"])
    improvement = frame["baseline_mae"] - frame["mae"]
    imp_mean, imp_low, imp_high = _mean_ci(improvement)
    normalized_improvement = improvement / frame["baseline_mae"].replace(0, np.nan)
    norm_imp_mean, norm_imp_low, norm_imp_high = _mean_ci(normalized_improvement)
    mae_mean, mae_low, mae_high = _mean_ci(frame["mae"])
    oof_prediction = np.divide(
        prediction_sum,
        prediction_count,
        out=np.full(len(y), np.nan, dtype=np.float64),
        where=prediction_count > 0,
    )
    return {
        "r": r_mean,
        "r_ci_low": r_low,
        "r_ci_high": r_high,
        "mae": mae_mean,
        "mae_ci_low": mae_low,
        "mae_ci_high": mae_high,
        "mae_improvement": imp_mean,
        "mae_improvement_ci_low": imp_low,
        "mae_improvement_ci_high": imp_high,
        "normalized_mae_improvement": norm_imp_mean,
        "normalized_mae_improvement_ci_low": norm_imp_low,
        "normalized_mae_improvement_ci_high": norm_imp_high,
        "selected_params": _mode_params(params),
        "oof_true": np.asarray(y, dtype=np.float64),
        "oof_prediction": oof_prediction,
        "n_folds": len(frame),
    }


def prediction_permutation_p(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    observed = _correlation(y_true, y_pred)
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [_correlation(rng.permutation(y_true), y_pred) for _ in range(permutations)],
        dtype=float,
    )
    exact = float((1 + np.sum(null >= observed)) / (permutations + 1))
    sd = float(np.nanstd(null, ddof=1))
    tail = float(norm.sf((observed - float(np.nanmean(null))) / sd)) if sd > 0 else 1.0
    return {
        "permutation_p_exact": exact,
        "permutation_p_tail": tail,
        "permutation_null_mean": float(np.nanmean(null)),
        "permutation_null_sd": sd,
    }


def fit_final_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    params: dict[str, Any],
    seed: int,
) -> Pipeline:
    pipeline = _regressor_pipeline(
        model,
        n_features=X.shape[1],
        n_train=len(y),
        seed=seed,
        params=params,
    )
    return pipeline.fit(X, y)


def bootstrap_external_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    frozen_baseline: float,
    seed: int,
    samples: int = 500,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    r = _correlation(y, prediction)
    mae = float(np.mean(np.abs(y - prediction)))
    baseline_mae = float(np.mean(np.abs(y - frozen_baseline)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), size=(samples, len(y)))
    sampled_y = y[indices]
    sampled_prediction = prediction[indices]
    centered_y = sampled_y - sampled_y.mean(axis=1, keepdims=True)
    centered_prediction = sampled_prediction - sampled_prediction.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(centered_y**2, axis=1) * np.sum(centered_prediction**2, axis=1)
    )
    correlations = np.divide(
        np.sum(centered_y * centered_prediction, axis=1),
        denominator,
        out=np.full(samples, np.nan, dtype=float),
        where=denominator > 0,
    )
    improvements = np.mean(np.abs(sampled_y - frozen_baseline), axis=1) - np.mean(
        np.abs(sampled_y - sampled_prediction), axis=1
    )
    sampled_baseline_mae = np.mean(np.abs(sampled_y - frozen_baseline), axis=1)
    normalized_improvements = np.divide(
        improvements,
        sampled_baseline_mae,
        out=np.full(samples, np.nan, dtype=float),
        where=sampled_baseline_mae > 0,
    )
    finite_r = np.asarray(correlations, dtype=float)
    finite_r = finite_r[np.isfinite(finite_r)]
    p_value = float(pearsonr(y, prediction).pvalue) if np.isfinite(r) else 1.0
    return {
        "n": int(len(y)),
        "r": r,
        "r_ci_low": float(np.quantile(finite_r, 0.025)) if finite_r.size else float("nan"),
        "r_ci_high": float(np.quantile(finite_r, 0.975)) if finite_r.size else float("nan"),
        "r_p": p_value,
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": baseline_mae - mae,
        "mae_improvement_ci_low": float(np.quantile(improvements, 0.025)),
        "mae_improvement_ci_high": float(np.quantile(improvements, 0.975)),
        "normalized_mae_improvement": (
            (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else float("nan")
        ),
        "normalized_mae_improvement_ci_low": float(
            np.nanquantile(normalized_improvements, 0.025)
        ),
        "normalized_mae_improvement_ci_high": float(
            np.nanquantile(normalized_improvements, 0.975)
        ),
    }


def _fdr(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    array[~np.isfinite(array)] = 1.0
    return multipletests(array, method="fdr_bh")[1]


def family_fdr(
    frame: pd.DataFrame,
    *,
    p_column: str,
    family_fields: Sequence[str],
) -> np.ndarray:
    output = np.ones(len(frame), dtype=float)
    for _, indices in frame.groupby(list(family_fields), sort=False).groups.items():
        position = np.asarray(list(indices), dtype=int)
        output[position] = _fdr(frame.loc[position, p_column])
    return output


def _identity_key(identity: dict[str, Any]) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        completed[_identity_key(payload["identity"])] = payload
    return completed


def append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def compact_checkpoint(path: Path) -> dict[str, int]:
    """Atomically retain the latest record for each candidate identity."""

    if not path.is_file():
        return {"rows_before": 0, "rows_after": 0, "duplicates_removed": 0}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest: dict[str, dict[str, Any]] = {}
    for line in lines:
        payload = json.loads(line)
        latest[_identity_key(payload["identity"])] = payload
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in latest.values():
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return {
        "rows_before": len(lines),
        "rows_after": len(latest),
        "duplicates_removed": len(lines) - len(latest),
    }


def _augment(X: np.ndarray, covariates: np.ndarray, *, include_age: bool) -> np.ndarray:
    selected = covariates if include_age else covariates[:, 1:2]
    return np.concatenate([X, selected], axis=1)


def _prepare_dataset_views(
    args: argparse.Namespace,
    *,
    work_dir: Path,
    dataset: str,
    atlas: str,
    metadata: pd.DataFrame,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    return load_feature_views(
        cache_root=args.cache_root,
        work_dir=work_dir,
        dataset=dataset,
        atlas=atlas,
        subject_ids=metadata["subject_id"].astype(str).tolist(),
    )


def _export_and_benchmark(
    args: argparse.Namespace,
    *,
    task: str,
    public: pd.DataFrame,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    output_dir: Path,
    kg_audit: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    protocol = protocol_for(task)
    tables_dir = output_dir / "tables"
    table_manifest = export_table_bundle(
        task=task,
        public=public,
        internal=internal,
        external=external,
        factor_fields=protocol.factor_fields,
        output_dir=tables_dir,
        provenance={**provenance, "created_at": utc_now(), "protocol": asdict(protocol)},
        kg_audit=kg_audit,
    )
    budgets = sorted({value for value in protocol.budgets if value <= len(public)} | {len(public)})
    benchmark_args = argparse.Namespace(
        task=task,
        public_candidates=tables_dir / "public_candidates.csv",
        internal_outcomes=tables_dir / "internal_outcomes.csv",
        external_outcomes=tables_dir / "external_outcomes.csv",
        search_policies=None,
        policy_schema="case-study-search-policy.v1",
        output_dir=output_dir / "benchmark",
        factor_fields=list(protocol.factor_fields),
        methods=["random_walk", "neurodiscovery"],
        trials=int(getattr(args, "benchmark_trials", protocol.minimum_trials)),
        seed=args.seed,
        batch_size=min(64, max(8, len(public) // 10)),
        warmup_batches=1,
        max_feedback_rounds=128,
        feedback_horizon=max(budgets),
        budgets=budgets,
        recall_targets=list(protocol.recall_targets),
    )
    benchmark_manifest = run_benchmark(benchmark_args)
    closure = {
        "schema_version": "hcp-case-study-closure.v1",
        "created_at": utc_now(),
        "task": task,
        "status": "complete",
        "table_manifest": table_manifest,
        "benchmark_manifest": benchmark_manifest,
    }
    (output_dir / "closure_manifest.json").write_text(
        json.dumps(closure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return closure


def build_connectome_behavior(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    work_dir = output_dir / "work"
    discovery_meta = hcpya_cognition_metadata(args.data_root)
    external_meta = adhd200_iq_controls(args.data_root)
    checkpoint_path = work_dir / f"connectome_behavior_paired_v4_cv{args.cv_repeats}_perm{args.permutations}.jsonl"
    compact_checkpoint(checkpoint_path)
    completed = load_checkpoint(checkpoint_path)
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    feature_audits: dict[str, Any] = {}
    total = len(args.atlases) * len(FEATURE_FAMILIES) * len(REGRESSION_MODELS)
    progress = 0

    def empty_external_metric() -> dict[str, Any]:
        metric = {
            name: float("nan")
            for name in (
                "r",
                "r_ci_low",
                "r_ci_high",
                "r_p",
                "mae",
                "baseline_mae",
                "mae_improvement",
                "mae_improvement_ci_low",
                "mae_improvement_ci_high",
            )
        }
        metric["n"] = 0
        return metric

    for atlas in args.atlases:
        subjects, views, audit = _prepare_dataset_views(
            args, work_dir=work_dir, dataset="hcpya", atlas=atlas, metadata=discovery_meta
        )
        try:
            ext_subjects, ext_views, ext_audit = _prepare_dataset_views(
                args, work_dir=work_dir, dataset="adhd200", atlas=atlas, metadata=external_meta
            )
            external_unavailable = ""
        except RuntimeError as exc:
            if not str(exc).startswith("No usable adhd200/"):
                raise
            ext_subjects, ext_views = [], {}
            external_unavailable = str(exc)
            ext_audit = {
                "status": "unavailable",
                "reason": external_unavailable,
                "executable": False,
            }
        feature_audits[atlas] = {"HCP-YA": audit, "ADHD200": ext_audit}
        for feature_family in FEATURE_FAMILIES:
            X_raw, y, cov, _ = align_features(subjects, views, discovery_meta, feature_family)
            X = _augment(X_raw, cov, include_age=True)
            if ext_views:
                X_ext_raw, y_ext, cov_ext, _ = align_features(
                    ext_subjects, ext_views, external_meta, feature_family
                )
                X_ext: np.ndarray | None = _augment(X_ext_raw, cov_ext, include_age=True)
            else:
                X_ext = None
                y_ext = np.asarray([], dtype=float)
            split_seed = stable_seed(
                args.seed,
                "connectome-paired-splits",
                atlas,
                feature_family,
            )
            for model in REGRESSION_MODELS:
                identity = {
                    "phenotype": "age_adjusted_total_cognition",
                    "atlas": atlas,
                    "feature_family": feature_family,
                    "model": model,
                }
                progress += 1
                key = _identity_key(identity)
                if key in completed:
                    payload = completed[key]
                    rows.append(identity)
                    internal_rows.append(payload["internal"])
                    external_rows.append(payload["external"])
                    print(f"[connectome {progress}/{total}] resume {atlas} {feature_family} {model}", flush=True)
                    continue
                seed = stable_seed(args.seed, "connectome", atlas, feature_family, model)
                internal_error = ""
                try:
                    cv = repeated_regression_cv(
                        X,
                        y,
                        model=model,
                        seed=seed,
                        repeats=args.cv_repeats,
                        split_seed=split_seed,
                    )
                    permutation = prediction_permutation_p(
                        cv["oof_true"],
                        cv["oof_prediction"],
                        permutations=args.permutations,
                        seed=seed + 17,
                    )
                    final = fit_final_regressor(
                        X,
                        y,
                        model=model,
                        params=cv["selected_params"],
                        seed=seed + 31,
                    )
                except Exception as exc:
                    cv = {name: float("nan") for name in (
                        "r", "r_ci_low", "r_ci_high", "mae", "mae_ci_low", "mae_ci_high",
                        "mae_improvement", "mae_improvement_ci_low", "mae_improvement_ci_high"
                    )}
                    cv.update({"selected_params": {}, "n_folds": 0})
                    permutation = {
                        "permutation_p_exact": 1.0,
                        "permutation_p_tail": 1.0,
                        "permutation_null_mean": float("nan"),
                        "permutation_null_sd": float("nan"),
                    }
                    final = None
                    internal_error = f"{type(exc).__name__}: {exc}"

                external_metric = empty_external_metric()
                external_error = ""
                if internal_error:
                    external_error = "internal model unavailable"
                elif X_ext is None:
                    external_error = external_unavailable
                else:
                    try:
                        prediction = final.predict(X_ext)
                        external_metric = bootstrap_external_metrics(
                            y_ext,
                            prediction,
                            frozen_baseline=float(np.mean(y)),
                            seed=seed + 43,
                        )
                    except Exception as exc:
                        external_error = f"{type(exc).__name__}: {exc}"
                internal_payload = {
                    **{name: cv[name] for name in (
                        "r", "r_ci_low", "r_ci_high", "mae", "mae_ci_low", "mae_ci_high",
                        "mae_improvement", "mae_improvement_ci_low", "mae_improvement_ci_high",
                    )},
                    **permutation,
                    "selected_params": json.dumps(cv["selected_params"], sort_keys=True),
                    "split_seed": split_seed,
                    "n_discovery": int(len(y)),
                    "n_folds": int(cv["n_folds"]),
                    "error": internal_error,
                }
                external_payload = {
                    "executable": bool(external_metric["n"] >= 50),
                    "adhd200_n": int(external_metric["n"]),
                    "adhd200_r": external_metric["r"],
                    "adhd200_r_ci_low": external_metric["r_ci_low"],
                    "adhd200_r_ci_high": external_metric["r_ci_high"],
                    "adhd200_r_p": external_metric["r_p"],
                    "adhd200_mae": external_metric["mae"],
                    "adhd200_baseline_mae": external_metric["baseline_mae"],
                    "adhd200_mae_improvement": external_metric["mae_improvement"],
                    "error": external_error,
                }
                rows.append(identity)
                internal_rows.append(internal_payload)
                external_rows.append(external_payload)
                append_checkpoint(
                    checkpoint_path,
                    {"identity": identity, "internal": internal_payload, "external": external_payload},
                )
                print(f"[connectome {progress}/{total}] done {atlas} {feature_family} {model}", flush=True)

    compact_checkpoint(checkpoint_path)
    protocol = protocol_for("connectome_behavior")
    factors = attach_candidate_ids(
        pd.DataFrame(rows), task="connectome_behavior", identity_fields=protocol.factor_fields
    )
    internal = pd.DataFrame(internal_rows)
    combined_internal = pd.concat([factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1)
    internal["q_value"] = family_fdr(
        combined_internal,
        p_column="permutation_p_tail",
        family_fields=("feature_family", "model"),
    )
    internal["validated"] = internal["r_ci_low"].gt(0) & internal["q_value"].lt(0.05)
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])

    external = pd.DataFrame(external_rows)
    combined_external = pd.concat([factors.reset_index(drop=True), external.reset_index(drop=True)], axis=1)
    external["adhd200_q"] = family_fdr(
        combined_external,
        p_column="adhd200_r_p",
        family_fields=("feature_family", "model"),
    )
    external["validated"] = (
        external["executable"].astype(bool)
        & external["adhd200_r_ci_low"].gt(0)
        & external["adhd200_q"].lt(0.05)
    )
    external.insert(0, "candidate_id", factors["candidate_id"])
    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=("phenotype", "atlas", "feature_family", "model"),
        case_study_id="connectome_behavior",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    return _export_and_benchmark(
        args,
        task="connectome_behavior",
        public=public,
        internal=internal,
        external=external,
        output_dir=output_dir,
        kg_audit=kg_audit,
        provenance={
            "discovery": "HCP-YA CogTotalComp_AgeAdj",
            "external": "ADHD200 typically-developing controls with full-scale IQ",
            "external_age_domain_shift": "preserved and reported; no external tuning",
            "family_structure": "restricted HCP family identifiers unavailable locally; subject-level folds",
            "feature_cache_audits": feature_audits,
            "cv_repeats": args.cv_repeats,
            "permutations": args.permutations,
        },
    )


def _copy_deep_robustness(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    robustness_dir = output_dir / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.deep_sweep / "model_results.csv"
    manifest_path = args.deep_sweep / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        payload = {"status": "missing", "source": str(args.deep_sweep)}
    else:
        results = pd.read_csv(result_path)
        results.to_csv(robustness_dir / "neuro_runtime_small_models.csv", index=False)
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = {
            "status": "complete",
            "source": str(args.deep_sweep),
            "models": sorted(results["model"].dropna().astype(str).unique().tolist()),
            "atlases": sorted(results["atlas"].dropna().astype(str).unique().tolist()),
            "completed_jobs": int(results["status"].eq("complete").sum()),
            "expected_jobs": int(source_manifest.get("n_jobs_expected", len(results))),
            "seeds": source_manifest.get("seeds", []),
            "role": "internal model-family robustness; not used to define primary GT",
        }
    (robustness_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def build_brain_age(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    work_dir = output_dir / "work"
    discovery_meta = lifespan_age_metadata()
    discovery_meta = discovery_meta.loc[discovery_meta["dataset"].eq("hcpya")].copy()
    age_support = (
        float(discovery_meta["age"].min()),
        float(discovery_meta["age"].max()),
    )
    external_meta: dict[str, pd.DataFrame] = {}
    external_age_audit: dict[str, Any] = {}
    for dataset in EXTERNAL_AGE_COHORTS:
        metadata = external_age_metadata(args.data_root, dataset)
        before = int(len(metadata))
        metadata = metadata.loc[metadata["age"].between(*age_support)].copy()
        external_meta[dataset] = metadata
        external_age_audit[dataset] = {
            "healthy_controls_before_age_support_filter": before,
            "within_discovery_age_support": int(len(metadata)),
        }
    checkpoint_path = work_dir / (
        f"brain_age_hcpya_stratified_v4_cv{args.cv_repeats}_perm{args.permutations}.jsonl"
    )
    compact_checkpoint(checkpoint_path)
    completed = load_checkpoint(checkpoint_path)
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    feature_audits: dict[str, Any] = {}
    total = len(args.atlases) * len(FEATURE_FAMILIES) * len(REGRESSION_MODELS)
    progress = 0

    def empty_external_metric() -> dict[str, Any]:
        return {
            "n": 0,
            **{
                name: float("nan")
                for name in (
                    "r",
                    "r_ci_low",
                    "r_ci_high",
                    "r_p",
                    "mae",
                    "baseline_mae",
                    "mae_improvement",
                    "mae_improvement_ci_low",
                    "mae_improvement_ci_high",
                    "normalized_mae_improvement",
                    "normalized_mae_improvement_ci_low",
                    "normalized_mae_improvement_ci_high",
                )
            },
        }

    for atlas in args.atlases:
        dataset_views: dict[str, tuple[list[str], dict[str, np.ndarray]]] = {}
        atlas_audit: dict[str, Any] = {}
        subjects, views, audit = _prepare_dataset_views(
            args,
            work_dir=work_dir,
            dataset="hcpya",
            atlas=atlas,
            metadata=discovery_meta,
        )
        dataset_views["hcpya"] = (subjects, views)
        atlas_audit["hcpya"] = audit
        for dataset, metadata in external_meta.items():
            try:
                subjects, views, audit = _prepare_dataset_views(
                    args, work_dir=work_dir, dataset=dataset, atlas=atlas, metadata=metadata
                )
                dataset_views[dataset] = (subjects, views)
                atlas_audit[dataset] = audit
            except RuntimeError as exc:
                if not str(exc).startswith(f"No usable {dataset}/"):
                    raise
                dataset_views[dataset] = ([], {})
                atlas_audit[dataset] = {
                    "status": "unavailable",
                    "reason": str(exc),
                    "executable": False,
                }
        feature_audits[atlas] = atlas_audit

        for feature_family in FEATURE_FAMILIES:
            subjects, views = dataset_views["hcpya"]
            X_raw, y, cov, _ = align_features(
                subjects,
                views,
                discovery_meta,
                feature_family,
            )
            X = _augment(X_raw, cov, include_age=False)
            external_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for dataset, metadata in external_meta.items():
                subjects, views = dataset_views[dataset]
                if not views:
                    continue
                X_ext_raw, y_ext, cov_ext, _ = align_features(
                    subjects, views, metadata, feature_family
                )
                external_arrays[dataset] = (
                    _augment(X_ext_raw, cov_ext, include_age=False),
                    y_ext,
                )
            split_seed = stable_seed(
                args.seed,
                "brain-age-paired-splits",
                atlas,
                feature_family,
            )
            for model in REGRESSION_MODELS:
                identity = {
                    "modality": "resting_state_fmri",
                    "atlas": atlas,
                    "feature_family": feature_family,
                    "model": model,
                }
                progress += 1
                key = _identity_key(identity)
                if key in completed:
                    payload = completed[key]
                    rows.append(identity)
                    internal_rows.append(payload["internal"])
                    external_rows.append(payload["external"])
                    print(f"[brain-age {progress}/{total}] resume {atlas} {feature_family} {model}", flush=True)
                    continue
                seed = stable_seed(args.seed, "brain-age", atlas, feature_family, model)
                error = ""
                try:
                    cv = repeated_regression_cv(
                        X,
                        y,
                        model=model,
                        seed=seed,
                        repeats=args.cv_repeats,
                        stratify_continuous=True,
                        split_seed=split_seed,
                    )
                    permutation = prediction_permutation_p(
                        cv["oof_true"],
                        cv["oof_prediction"],
                        permutations=args.permutations,
                        seed=seed + 19,
                    )
                    final = fit_final_regressor(
                        X,
                        y,
                        model=model,
                        params=cv["selected_params"],
                        seed=seed + 31,
                    )
                except Exception as exc:
                    cv = {name: float("nan") for name in (
                        "r", "r_ci_low", "r_ci_high", "mae", "mae_ci_low", "mae_ci_high",
                        "mae_improvement", "mae_improvement_ci_low", "mae_improvement_ci_high",
                        "normalized_mae_improvement", "normalized_mae_improvement_ci_low",
                        "normalized_mae_improvement_ci_high",
                    )}
                    cv.update({"selected_params": {}, "n_folds": 0})
                    permutation = {
                        "permutation_p_exact": 1.0,
                        "permutation_p_tail": 1.0,
                        "permutation_null_mean": float("nan"),
                        "permutation_null_sd": float("nan"),
                    }
                    final = None
                    error = f"{type(exc).__name__}: {exc}"

                metrics = {dataset: empty_external_metric() for dataset in EXTERNAL_AGE_COHORTS}
                external_errors = {
                    dataset: str(atlas_audit[dataset].get("reason") or "")
                    for dataset in EXTERNAL_AGE_COHORTS
                }
                if final is not None:
                    for index, (dataset, (X_ext, y_ext)) in enumerate(external_arrays.items()):
                        try:
                            metrics[dataset] = bootstrap_external_metrics(
                                y_ext,
                                final.predict(X_ext),
                                frozen_baseline=float(np.mean(y)),
                                seed=seed + 43 + index,
                            )
                            external_errors[dataset] = ""
                        except Exception as exc:
                            external_errors[dataset] = f"{type(exc).__name__}: {exc}"
                elif error:
                    for dataset in external_arrays:
                        external_errors[dataset] = "internal model unavailable"
                internal_payload = {
                    **{name: cv[name] for name in (
                        "r", "r_ci_low", "r_ci_high", "mae", "mae_ci_low", "mae_ci_high",
                        "mae_improvement", "mae_improvement_ci_low", "mae_improvement_ci_high",
                        "normalized_mae_improvement", "normalized_mae_improvement_ci_low",
                        "normalized_mae_improvement_ci_high",
                    )},
                    **permutation,
                    "selected_params": json.dumps(cv["selected_params"], sort_keys=True),
                    "split_seed": split_seed,
                    "n_discovery": int(len(y)),
                    "n_folds": int(cv["n_folds"]),
                    "error": error,
                }
                external_payload: dict[str, Any] = {"executable": False}
                executable_cohorts = 0
                for dataset, metric in metrics.items():
                    prefix = dataset.replace("-", "_")
                    for name, value in metric.items():
                        external_payload[f"{prefix}_{name}"] = value
                    external_payload[f"{prefix}_error"] = external_errors[dataset]
                    executable_cohorts += int(int(metric["n"]) >= 30)
                external_payload["executable_cohorts"] = executable_cohorts
                external_payload["executable"] = executable_cohorts >= 2
                rows.append(identity)
                internal_rows.append(internal_payload)
                external_rows.append(external_payload)
                append_checkpoint(
                    checkpoint_path,
                    {"identity": identity, "internal": internal_payload, "external": external_payload},
                )
                print(f"[brain-age {progress}/{total}] done {atlas} {feature_family} {model}", flush=True)

    compact_checkpoint(checkpoint_path)
    protocol = protocol_for("brain_age")
    factors = attach_candidate_ids(
        pd.DataFrame(rows), task="brain_age", identity_fields=protocol.factor_fields
    )
    internal = pd.DataFrame(internal_rows)
    combined_internal = pd.concat(
        [factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1
    )
    internal["q_value"] = family_fdr(
        combined_internal,
        p_column=BRAIN_AGE_FDR_P_COLUMN,
        family_fields=("feature_family", "model"),
    )
    internal["validated"] = (
        internal["r_ci_low"].ge(BRAIN_AGE_MIN_R_CI_LOW)
        & internal["normalized_mae_improvement_ci_low"].ge(
            BRAIN_AGE_MIN_NORMALIZED_MAE_REDUCTION
        )
        & internal["q_value"].lt(0.05)
    )
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])

    external = pd.DataFrame(external_rows)
    support_columns: list[str] = []
    for dataset in EXTERNAL_AGE_COHORTS:
        prefix = dataset.replace("-", "_")
        q_column = f"{prefix}_r_q"
        external[q_column] = family_fdr(
            pd.concat([factors.reset_index(drop=True), external.reset_index(drop=True)], axis=1),
            p_column=f"{prefix}_r_p",
            family_fields=("feature_family", "model"),
        )
        support_column = f"{prefix}_supported"
        external[support_column] = (
            external[f"{prefix}_n"].ge(30)
            & external[f"{prefix}_r_ci_low"].gt(0)
            & external[q_column].lt(0.05)
            & external[f"{prefix}_normalized_mae_improvement_ci_low"].gt(0)
        )
        support_columns.append(support_column)
    external["supporting_cohorts"] = external[support_columns].sum(axis=1)
    external["validated"] = external["supporting_cohorts"].ge(2)
    external.insert(0, "candidate_id", factors["candidate_id"])
    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=("modality", "atlas", "feature_family", "model"),
        case_study_id="brain_age",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    robustness = {
        "status": "excluded",
        "source": str(args.deep_sweep),
        "reason": (
            "the existing sweep mixes HCP-YA and HCP-Aging, so cohort identity can proxy age; "
            "an HCP-YA-only five-fold rerun is required"
        ),
    }
    robustness_dir = output_dir / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    (robustness_dir / "manifest.json").write_text(
        json.dumps(robustness, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return _export_and_benchmark(
        args,
        task="brain_age",
        public=public,
        internal=internal,
        external=external,
        output_dir=output_dir,
        kg_audit=kg_audit,
        provenance={
            "discovery": "HCP-YA only",
            "cohort_confound_control": (
                "HCP-Aging is excluded because combining non-overlapping cohorts makes cohort "
                "identity a proxy for age"
            ),
            "internal_cv": "nested age-stratified repeated five-fold cross-validation",
            "internal_thresholds": {
                "r_ci_low": BRAIN_AGE_MIN_R_CI_LOW,
                "normalized_mae_reduction_ci_low": BRAIN_AGE_MIN_NORMALIZED_MAE_REDUCTION,
                "fdr_p_column": BRAIN_AGE_FDR_P_COLUMN,
                "family_fdr_q": 0.05,
            },
            "external": list(EXTERNAL_AGE_COHORTS),
            "external_age_support": {
                "minimum": age_support[0],
                "maximum": age_support[1],
            },
            "external_age_audit": external_age_audit,
            "external_rule": (
                "at least two in-support cohorts with q<0.05, positive r CI, and positive "
                "normalized frozen-baseline MAE improvement CI"
            ),
            "feature_cache_audits": feature_audits,
            "cv_repeats": args.cv_repeats,
            "permutations": args.permutations,
            "deep_model_robustness": robustness,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("connectome_behavior", "brain_age", "both"), default="both")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--deep-sweep", type=Path, default=DEFAULT_DEEP_SWEEP)
    parser.add_argument("--atlases", nargs="+", default=list(DEFAULT_ATLASES))
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument(
        "--benchmark-trials",
        type=int,
        default=10,
        help="Outcome-blind generator trials (5 development; 10 final).",
    )
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.atlases = args.atlases[:1]
        args.cv_repeats = 1
        args.permutations = min(args.permutations, 9)
    tasks = ("connectome_behavior", "brain_age") if args.task == "both" else (args.task,)
    manifests: dict[str, Any] = {}
    for task in tasks:
        output_dir = args.output_root / task
        output_dir.mkdir(parents=True, exist_ok=True)
        if task == "connectome_behavior":
            manifests[task] = build_connectome_behavior(args, output_dir)
        else:
            manifests[task] = build_brain_age(args, output_dir)
    print(json.dumps(manifests, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
