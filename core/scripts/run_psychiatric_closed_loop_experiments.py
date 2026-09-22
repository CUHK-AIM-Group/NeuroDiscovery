"""Build and run the formal psychiatric closed-loop case studies.

Discovery is restricted to UCLA. Differential diagnosis transfers a frozen
schizophrenia-versus-bipolar model to the non-affective-versus-affective
psychosis contrast in HCP-EP; COBRE is retained as a small supplementary
cohort. Disease subtyping is outcome-blind: imaging-only clusters are tested
against continuous symptom dimensions that were never used to form clusters.

The script writes outcome-blind candidates, internal outcomes, and external
outcomes to physically separate files, then invokes the shared closed-loop
benchmark. Deep NeuroRuntime models are handled by the separate robustness
sweep because their whole-connectome outputs are not ROI-level hypotheses.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, norm, spearmanr, t
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

# Historical CS1 feature helpers still import their siblings as top-level
# modules. Keep that compatibility local to this runner.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core.scripts.case1_exhaustive_full import (
    CORR_FEATURES,
    PARTIAL_FEATURES,
    ROI_FEATURES,
    build_atlas_feature_matrices,
)
from core.scripts.case_study_candidate_tables import (
    attach_candidate_ids,
    export_table_bundle,
    score_public_candidates,
)
from core.scripts.case_study_closed_loop import run_benchmark
from core.scripts.case_study_closed_loop_specs import protocol_for
from models.subtyping import fit_subtypes


DEFAULT_DATA_ROOT = Path(r"\\192.168.3.61\data\Public Dataset")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "case_study_closed_loop_v1"
DEFAULT_KG = Path("neurooracle/data/full_v2/knowledge_graph.json")
DEFAULT_UCLA_PHENOTYPE_ROOT = (
    DEFAULT_DATA_ROOT
    / "ucla_preprocessed"
    / "metadata"
    / "openneuro_ds000030_1.0.0_4070b6e"
    / "phenotype"
)
UCLA_PHENOTYPE_SOURCE = {
    "dataset": "OpenNeuro ds000030",
    "version": "1.0.0",
    "git_commit": "4070b6eea231517bfeff42527a46d9d166ac4e13",
}
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
VIEW_FEATURES = {
    "roi_dynamics": tuple(ROI_FEATURES),
    "correlation_topology": tuple(CORR_FEATURES),
    "partial_correlation_topology": tuple(PARTIAL_FEATURES),
}
DIFFERENTIAL_MODELS = ("logistic", "elastic_net", "svm")
SUBTYPING_MODELS = ("kmeans", "gmm", "spectral", "nmf", "consensus", "autoencoder")
SUBTYPING_CLINICAL_DIMENSIONS = (
    "positive_psychosis",
    "negative_symptoms",
    "mania",
    "depression_anxiety",
    "attention_inattention",
)


@dataclass(frozen=True)
class Cohort:
    name: str
    root: Path
    metadata: Path


COHORTS = {
    "UCLA": Cohort(
        "UCLA",
        DEFAULT_DATA_ROOT / "ucla_preprocessed",
        DEFAULT_DATA_ROOT / "ucla_preprocessed" / "metadata" / "ucla-rest.csv",
    ),
    "HCP-EP": Cohort(
        "HCP-EP",
        DEFAULT_DATA_ROOT / "hcpep_preprocessed",
        DEFAULT_DATA_ROOT / "hcpep_preprocessed" / "metadata" / "hcpep-rest.csv",
    ),
    "COBRE": Cohort(
        "COBRE",
        DEFAULT_DATA_ROOT / "cobre_preprocessed",
        DEFAULT_DATA_ROOT / "cobre_preprocessed" / "metadata" / "cobre-rest.csv",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def _sex_binary(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.casefold()
    return normalized.map(
        {
            "m": 1.0,
            "male": 1.0,
            "1": 1.0,
            "f": 0.0,
            "female": 0.0,
            "0": 0.0,
        }
    )


def cohort_metadata(name: str) -> pd.DataFrame:
    cohort = COHORTS[name]
    raw = pd.read_csv(cohort.metadata)
    if name == "UCLA":
        out = pd.DataFrame(
            {
                "subject_id": raw["subject_id"].astype(str),
                "age": pd.to_numeric(raw["age"], errors="coerce"),
                "sex": _sex_binary(raw["gender"]),
                "phenotype": raw["diagnosis"].astype(str),
                "site": "UCLA",
            }
        )
    elif name == "HCP-EP":
        out = pd.DataFrame(
            {
                "subject_id": raw["subject_id"].astype(str).map(
                    lambda value: value if value.startswith("sub-") else f"sub-{value}"
                ),
                "age": pd.to_numeric(raw["interview_age"], errors="coerce") / 12.0,
                "sex": _sex_binary(raw["sex"]),
                "phenotype": raw["phenotype_description"].astype(str),
                "site": raw["site"].fillna("HCP-EP").astype(str),
            }
        )
    elif name == "COBRE":
        out = pd.DataFrame(
            {
                "subject_id": raw["subject_id"].astype(str),
                "age": pd.to_numeric(raw["age"], errors="coerce"),
                "sex": _sex_binary(raw["sex"]),
                "phenotype": raw["dx"].astype(str),
                "site": "COBRE",
            }
        )
    else:  # pragma: no cover - guarded by the registry
        raise KeyError(name)
    return out.drop_duplicates("subject_id").reset_index(drop=True)


def differential_labels(metadata: pd.DataFrame, cohort: str) -> pd.DataFrame:
    values = (
        metadata["phenotype"]
        .astype(str)
        .str.casefold()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if cohort == "UCLA":
        keep = values.isin(("schz", "bipolar"))
        label = values.map({"bipolar": 0, "schz": 1})
    elif cohort == "HCP-EP":
        keep = values.isin(("non-affective psychosis", "affective psychosis"))
        label = values.map({"affective psychosis": 0, "non-affective psychosis": 1})
    elif cohort == "COBRE":
        affective = values.eq("bipolar disorder")
        non_affective = values.isin(("schizophrenia strict", "schizoaffective"))
        keep = affective | non_affective
        label = pd.Series(np.where(non_affective, 1, 0), index=metadata.index)
    else:  # pragma: no cover
        raise KeyError(cohort)
    out = metadata.loc[keep].copy()
    out["label"] = label.loc[keep].astype(int)
    return out.reset_index(drop=True)


def _z_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    scale = float(numeric.std(ddof=0))
    if not math.isfinite(scale) or scale < 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - float(numeric.mean())) / scale


def load_ucla_subtyping_metadata(
    metadata: pd.DataFrame,
    phenotype_root: Path,
) -> pd.DataFrame:
    """Return patient imaging metadata plus outcome-blind symptom dimensions."""

    required = {
        "bprs": ("bprs_positive", "bprs_negative", "bprs_mania", "bprs_depanx"),
        "hamilton": ("hamd_28",),
        "ymrs": ("ymrs_score",),
        "asrs": ("asrs_score",),
    }
    merged = metadata.copy()
    normalized = (
        merged["phenotype"]
        .astype(str)
        .str.casefold()
        .str.replace("_", " ", regex=False)
        .str.strip()
    )
    merged = merged.loc[normalized.isin(("adhd", "bipolar", "schz"))].copy()
    merged["diagnosis"] = normalized.loc[merged.index].to_numpy()
    for stem, columns in required.items():
        path = phenotype_root / f"{stem}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing UCLA phenotype table: {path}")
        table = pd.read_csv(path, sep="\t", na_values=["n/a"])
        missing = [column for column in ("participant_id", *columns) if column not in table]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        table = table.loc[:, ["participant_id", *columns]].rename(
            columns={"participant_id": "subject_id"}
        )
        merged = merged.merge(table, on="subject_id", how="left", validate="one_to_one")

    source_scores = {
        column: _z_score(merged[column])
        for columns in required.values()
        for column in columns
    }
    merged["positive_psychosis"] = source_scores["bprs_positive"]
    merged["negative_symptoms"] = source_scores["bprs_negative"]
    merged["mania"] = (
        source_scores["bprs_mania"] + source_scores["ymrs_score"]
    ) / 2.0
    merged["depression_anxiety"] = (
        source_scores["bprs_depanx"] + source_scores["hamd_28"]
    ) / 2.0
    merged["attention_inattention"] = source_scores["asrs_score"]
    merged = merged.dropna(subset=list(SUBTYPING_CLINICAL_DIMENSIONS))
    return merged.reset_index(drop=True)


_ATLAS_CACHE: dict[tuple[str, str], tuple[list[str], dict[str, np.ndarray]]] = {}


def atlas_feature_views(cohort: str, atlas: str) -> tuple[list[str], dict[str, np.ndarray]]:
    key = (cohort, atlas)
    if key in _ATLAS_CACHE:
        return _ATLAS_CACHE[key]
    meta = cohort_metadata(cohort)
    subjects, _roi_meta, features = build_atlas_feature_matrices(
        COHORTS[cohort].root,
        atlas,
        meta["subject_id"].astype(str).tolist(),
    )
    views = {
        view: np.concatenate([features[name] for name in names], axis=1)
        for view, names in VIEW_FEATURES.items()
    }
    _ATLAS_CACHE[key] = (subjects, views)
    return _ATLAS_CACHE[key]


def aligned_view(
    cohort: str,
    atlas: str,
    feature_family: str,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subjects, views = atlas_feature_views(cohort, atlas)
    index = {subject: idx for idx, subject in enumerate(subjects)}
    rows = labels.loc[labels["subject_id"].isin(index)].copy()
    rows["_index"] = rows["subject_id"].map(index)
    rows = rows.sort_values("_index")
    indices = rows["_index"].astype(int).to_numpy()
    X = np.asarray(views[feature_family][indices], dtype=np.float64)
    covariates = rows[["age", "sex"]].to_numpy(dtype=np.float64)
    y = rows["label"].to_numpy(dtype=int)
    return X, covariates, y, rows["subject_id"].astype(str).tolist()


def aligned_subtyping_view(
    atlas: str,
    feature_family: str,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subjects, views = atlas_feature_views("UCLA", atlas)
    index = {subject: idx for idx, subject in enumerate(subjects)}
    rows = metadata.loc[metadata["subject_id"].isin(index)].copy()
    rows["_index"] = rows["subject_id"].map(index)
    rows = rows.sort_values("_index")
    indices = rows["_index"].astype(int).to_numpy()
    X = np.asarray(views[feature_family][indices], dtype=np.float64)
    covariates = rows[["age", "sex"]].to_numpy(dtype=np.float64)
    symptoms = rows[list(SUBTYPING_CLINICAL_DIMENSIONS)].to_numpy(dtype=np.float64)
    return X, covariates, symptoms, rows["subject_id"].astype(str).tolist()


class CovariateResidualizedPCA(BaseEstimator, TransformerMixin):
    """Fold-local feature residualization followed by scaling and PCA."""

    def __init__(
        self,
        n_covariates: int = 2,
        max_components: int = 20,
        robust: bool = False,
        clip: float = 5.0,
    ):
        self.n_covariates = n_covariates
        self.max_components = max_components
        self.robust = robust
        self.clip = clip

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        values = np.asarray(X, dtype=np.float64)
        split = values.shape[1] - int(self.n_covariates)
        if split < 1:
            raise ValueError("residualizer requires imaging features and covariates")
        imaging, covariates = values[:, :split], values[:, split:]
        self.feature_imputer_ = SimpleImputer(strategy="median").fit(imaging)
        self.covariate_imputer_ = SimpleImputer(strategy="median").fit(covariates)
        imaging = self.feature_imputer_.transform(imaging)
        covariates = self.covariate_imputer_.transform(covariates)
        design = np.column_stack((np.ones(len(covariates)), covariates))
        self.beta_, *_ = np.linalg.lstsq(design, imaging, rcond=None)
        residual = imaging - design @ self.beta_
        self.scaler_ = (
            RobustScaler(quantile_range=(10.0, 90.0))
            if self.robust
            else StandardScaler()
        ).fit(residual)
        scaled = self.scaler_.transform(residual)
        if self.robust:
            scaled = np.clip(scaled, -float(self.clip), float(self.clip))
        components = max(
            1,
            min(int(self.max_components), scaled.shape[0] - 1, scaled.shape[1]),
        )
        self.pca_ = PCA(n_components=components, svd_solver="full").fit(scaled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        split = values.shape[1] - int(self.n_covariates)
        imaging = self.feature_imputer_.transform(values[:, :split])
        covariates = self.covariate_imputer_.transform(values[:, split:])
        design = np.column_stack((np.ones(len(covariates)), covariates))
        residual = imaging - design @ self.beta_
        scaled = self.scaler_.transform(residual)
        if self.robust:
            scaled = np.clip(scaled, -float(self.clip), float(self.clip))
        return self.pca_.transform(scaled)


def append_covariates(X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    return np.column_stack((X, covariates))


def align_external_features(reference: np.ndarray, external: np.ndarray) -> np.ndarray:
    """Label-blind location/scale alignment fixed before external evaluation."""

    reference = np.asarray(reference, dtype=np.float64)
    external = np.asarray(external, dtype=np.float64)
    ref_median = np.nanmedian(reference, axis=0)
    ext_median = np.nanmedian(external, axis=0)
    ref_scale = np.nanstd(reference, axis=0)
    ext_scale = np.nanstd(external, axis=0)
    ref_scale[~np.isfinite(ref_scale) | (ref_scale < 1e-8)] = 1.0
    ext_scale[~np.isfinite(ext_scale) | (ext_scale < 1e-8)] = 1.0
    aligned = (external - ext_median) / ext_scale * ref_scale + ref_median
    return np.asarray(aligned, dtype=np.float64)


def classifier_pipeline(model: str, *, params: dict[str, Any] | None = None) -> Pipeline:
    if model == "logistic":
        classifier = LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0,
        )
        grid = {"model__C": (0.1, 1.0, 10.0)}
    elif model == "elastic_net":
        classifier = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            l1_ratio=0.5,
            random_state=0,
        )
        grid = {
            "model__C": (0.1, 1.0, 10.0),
            "model__l1_ratio": (0.25, 0.5, 0.75),
        }
    elif model == "svm":
        classifier = SVC(
            kernel="linear",
            class_weight="balanced",
            probability=False,
            random_state=0,
        )
        grid = {"model__C": (0.1, 1.0, 10.0)}
    else:
        raise ValueError(f"unsupported differential model: {model}")
    pipeline = Pipeline(
        (
            ("representation", CovariateResidualizedPCA()),
            ("model", classifier),
        )
    )
    pipeline._registered_grid = grid  # type: ignore[attr-defined]
    if params:
        pipeline.set_params(**params)
    return pipeline


def classifier_core_pipeline(
    model: str,
    *,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Classifier used after a frozen, outcome-blind representation."""

    full = classifier_pipeline(model)
    classifier = clone(full.named_steps["model"])
    pipeline = Pipeline((("scale", StandardScaler()), ("model", classifier)))
    if params:
        pipeline.set_params(**params)
    return pipeline


def decision_scores(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    return np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)


def _mode_params(params: Sequence[dict[str, Any]]) -> dict[str, Any]:
    serialized = [json.dumps(value, sort_keys=True) for value in params]
    return json.loads(Counter(serialized).most_common(1)[0][0])


def nested_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    seed: int,
    repeats: int,
    split_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    frozen_split_seed = seed if split_seed is None else int(split_seed)
    outer = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=repeats,
        random_state=frozen_split_seed,
    )
    scores: list[float] = []
    best_params: list[dict[str, Any]] = []
    base = classifier_pipeline(model)
    grid = dict(base._registered_grid)  # type: ignore[attr-defined]
    for fold, (train, test) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=frozen_split_seed + fold + 1,
        )
        search = GridSearchCV(
            clone(base),
            grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X[train], y[train])
        scores.append(roc_auc_score(y[test], decision_scores(search.best_estimator_, X[test])))
        best_params.append(dict(search.best_params_))
    return np.asarray(scores, dtype=float), _mode_params(best_params)


def permutation_p_values(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    params: dict[str, Any],
    observed: float,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    if permutations <= 0:
        return {
            "exact": 1.0,
            "tail": 1.0,
            "null_mean": float("nan"),
            "null_sd": float("nan"),
        }
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    representation = CovariateResidualizedPCA().fit(X)
    frozen_X = representation.transform(X)
    null: list[float] = []
    for _ in range(permutations):
        permuted = rng.permutation(y)
        fold_scores: list[float] = []
        for train, test in cv.split(frozen_X, permuted):
            estimator = classifier_core_pipeline(model, params=params)
            estimator.fit(frozen_X[train], permuted[train])
            fold_scores.append(
                roc_auc_score(
                    permuted[test],
                    decision_scores(estimator, frozen_X[test]),
                )
            )
        null.append(float(np.mean(fold_scores)))
    null_values = np.asarray(null, dtype=float)
    exact = float(
        (1 + np.count_nonzero(null_values >= observed)) / (permutations + 1)
    )
    null_mean = float(np.mean(null_values))
    null_sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
    if null_sd <= 1e-12:
        tail = exact
    else:
        tail = float(norm.sf((observed - null_mean) / null_sd))
        tail = min(1.0, max(np.finfo(float).tiny, tail))
    return {
        "exact": exact,
        "tail": tail,
        "null_mean": null_mean,
        "null_sd": null_sd,
    }


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    delta = float(t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - delta, mean + delta


def bootstrap_auc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    repeats: int = 1000,
) -> tuple[float, float, float]:
    observed = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    boot: list[float] = []
    for _ in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in class_indices]
        )
        boot.append(float(roc_auc_score(y[sampled], scores[sampled])))
    low, high = np.quantile(boot, (0.025, 0.975))
    return observed, float(low), float(high)


def external_auc(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    scores = decision_scores(estimator, X)
    auc, low, high = bootstrap_auc_ci(y, scores, seed=seed)
    positive = scores[y == 1]
    negative = scores[y == 0]
    p_value = float(mannwhitneyu(positive, negative, alternative="greater").pvalue)
    return {
        "n": int(len(y)),
        "n_negative": int(np.count_nonzero(y == 0)),
        "n_positive": int(np.count_nonzero(y == 1)),
        "auroc": auc,
        "auroc_ci_low": low,
        "auroc_ci_high": high,
        "p_value": p_value,
    }


def _fdr(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    array[~np.isfinite(array)] = 1.0
    return multipletests(array, alpha=0.05, method="fdr_bh")[1]


def family_fdr(
    frame: pd.DataFrame,
    *,
    p_column: str,
    family_fields: Sequence[str],
) -> np.ndarray:
    q_values = np.ones(len(frame), dtype=float)
    for _family, indices in frame.groupby(list(family_fields), sort=False).groups.items():
        local = np.asarray(list(indices), dtype=int)
        q_values[local] = _fdr(frame.iloc[local][p_column])
    return q_values


def _identity_key(identity: dict[str, Any]) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows[_identity_key(payload["identity"])] = payload
    return rows


def append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def compact_checkpoint(path: Path) -> dict[str, int]:
    """Atomically retain one last-write-wins record per candidate identity."""

    if not path.exists():
        return {"rows_before": 0, "rows_after": 0, "duplicates_removed": 0}
    rows_before = 0
    unique: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows_before += 1
            payload = json.loads(line)
            unique[_identity_key(payload["identity"])] = payload
    temporary = path.with_suffix(path.suffix + ".compact.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in unique.values():
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)
    return {
        "rows_before": rows_before,
        "rows_after": len(unique),
        "duplicates_removed": rows_before - len(unique),
    }


def fit_final_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    seed: int,
    split_seed: int | None = None,
) -> GridSearchCV:
    base = classifier_pipeline(model)
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed if split_seed is None else int(split_seed),
    )
    search = GridSearchCV(
        base,
        dict(base._registered_grid),  # type: ignore[attr-defined]
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def build_differential_tables(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    discovery_labels = differential_labels(cohort_metadata("UCLA"), "UCLA")
    hcp_labels = differential_labels(cohort_metadata("HCP-EP"), "HCP-EP")
    cobre_labels = differential_labels(cohort_metadata("COBRE"), "COBRE")
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    checkpoint_path = (
        output_dir
        / "work"
        / f"differential_candidates_paired_v4_cv{args.cv_repeats}_perm{args.permutations}.jsonl"
    )
    completed = load_checkpoint(checkpoint_path)
    total = len(args.atlases) * len(VIEW_FEATURES) * len(DIFFERENTIAL_MODELS)
    progress = 0

    for atlas in args.atlases:
        for feature_family in VIEW_FEATURES:
            X_raw, cov, y, _ = aligned_view(
                "UCLA", atlas, feature_family, discovery_labels
            )
            X = append_covariates(X_raw, cov)
            X_hcp_raw, cov_hcp, y_hcp, _ = aligned_view(
                "HCP-EP", atlas, feature_family, hcp_labels
            )
            X_cobre_raw, cov_cobre, y_cobre, _ = aligned_view(
                "COBRE", atlas, feature_family, cobre_labels
            )
            X_hcp = append_covariates(
                align_external_features(X_raw, X_hcp_raw), cov_hcp
            )
            X_cobre = append_covariates(
                align_external_features(X_raw, X_cobre_raw), cov_cobre
            )
            split_seed = stable_seed(
                args.seed,
                "differential-paired-splits",
                atlas,
                feature_family,
            )
            for model in DIFFERENTIAL_MODELS:
                identity = {
                    "diagnostic_contrast": "non-affective_vs_affective_psychosis",
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
                    print(f"[differential {progress}/{total}] resume {atlas} {feature_family} {model}", flush=True)
                    continue
                seed = stable_seed(args.seed, "differential", atlas, feature_family, model)
                try:
                    scores, params = nested_auc(
                        X,
                        y,
                        model=model,
                        seed=seed,
                        repeats=args.cv_repeats,
                        split_seed=split_seed,
                    )
                    mean, low, high = mean_ci(scores)
                    permutation = permutation_p_values(
                        X,
                        y,
                        model=model,
                        params=params,
                        observed=mean,
                        permutations=args.permutations,
                        seed=seed + 17,
                    )
                    final = fit_final_classifier(
                        X,
                        y,
                        model=model,
                        seed=seed + 31,
                        split_seed=split_seed + 31,
                    )
                    hcp = external_auc(final.best_estimator_, X_hcp, y_hcp, seed=seed + 43)
                    cobre = external_auc(
                        final.best_estimator_, X_cobre, y_cobre, seed=seed + 59
                    )
                    error = ""
                except Exception as exc:  # preserve failed candidates for audit
                    mean = low = high = float("nan")
                    permutation = {
                        "exact": 1.0,
                        "tail": 1.0,
                        "null_mean": float("nan"),
                        "null_sd": float("nan"),
                    }
                    params = {}
                    hcp = {key: float("nan") for key in ("auroc", "auroc_ci_low", "auroc_ci_high", "p_value")}
                    hcp.update({"n": 0, "n_negative": 0, "n_positive": 0})
                    cobre = dict(hcp)
                    error = f"{type(exc).__name__}: {exc}"
                internal_payload = {
                        "mean_auroc": mean,
                        "auroc_ci_low": low,
                        "auroc_ci_high": high,
                        "permutation_p_exact": permutation["exact"],
                        "permutation_p_tail": permutation["tail"],
                        "permutation_null_mean": permutation["null_mean"],
                        "permutation_null_sd": permutation["null_sd"],
                        "selected_params": json.dumps(params, sort_keys=True),
                        "split_seed": split_seed,
                        "n_folds": int(5 * args.cv_repeats),
                        "n_discovery": int(len(y)),
                        "error": error,
                    }
                external_payload = {
                        "executable": bool(hcp["n_negative"] >= 10 and hcp["n_positive"] >= 10),
                        "hcp_ep_auroc": hcp["auroc"],
                        "hcp_ep_ci_low": hcp["auroc_ci_low"],
                        "hcp_ep_ci_high": hcp["auroc_ci_high"],
                        "hcp_ep_p": hcp["p_value"],
                        "hcp_ep_n": hcp["n"],
                        "cobre_auroc": cobre["auroc"],
                        "cobre_ci_low": cobre["auroc_ci_low"],
                        "cobre_ci_high": cobre["auroc_ci_high"],
                        "cobre_p": cobre["p_value"],
                        "cobre_n": cobre["n"],
                    }
                rows.append(identity)
                internal_rows.append(internal_payload)
                external_rows.append(external_payload)
                append_checkpoint(
                    checkpoint_path,
                    {
                        "identity": identity,
                        "internal": internal_payload,
                        "external": external_payload,
                    },
                )
                print(f"[differential {progress}/{total}] done {atlas} {feature_family} {model}", flush=True)

    compact_checkpoint(checkpoint_path)
    factors = attach_candidate_ids(
        pd.DataFrame(rows),
        task="differential_diagnosis",
        identity_fields=protocol_for("differential_diagnosis").factor_fields,
    )
    internal = pd.DataFrame(internal_rows)
    combined_internal = pd.concat(
        [factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1
    )
    internal["q_value"] = family_fdr(
        combined_internal,
        p_column="permutation_p_tail",
        family_fields=("feature_family", "model"),
    )
    internal["validated"] = (
        internal["mean_auroc"].ge(0.5)
        & internal["auroc_ci_low"].gt(0.5)
        & internal["q_value"].lt(0.05)
    )
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])

    external = pd.DataFrame(external_rows)
    combined_external = pd.concat(
        [factors.reset_index(drop=True), external.reset_index(drop=True)], axis=1
    )
    external["hcp_ep_q"] = family_fdr(
        combined_external,
        p_column="hcp_ep_p",
        family_fields=("feature_family", "model"),
    )
    external["cobre_q"] = family_fdr(
        combined_external,
        p_column="cobre_p",
        family_fields=("feature_family", "model"),
    )
    external["validated"] = (
        external["executable"].astype(bool)
        & external["hcp_ep_ci_low"].gt(0.5)
        & external["hcp_ep_q"].lt(0.05)
    )
    external.insert(0, "candidate_id", factors["candidate_id"])

    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=("diagnostic_contrast", "atlas", "feature_family", "model"),
        case_study_id="differential_diagnosis",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    return export_and_benchmark(
        args,
        task="differential_diagnosis",
        public=public,
        internal=internal,
        external=external,
        output_dir=output_dir,
        kg_audit=kg_audit,
        provenance={
            "discovery": "UCLA SCHZ vs BIPOLAR",
            "external_primary": "HCP-EP non-affective vs affective psychosis",
            "external_supplementary": "COBRE SZ/SZA vs bipolar",
            "covariate_adjustment": "fold-local age and sex residualization",
            "permutations": args.permutations,
            "cv_repeats": args.cv_repeats,
            "permutation_representation": (
                "single outcome-blind residualized PCA frozen before label permutations"
            ),
        },
    )


def cluster_solution(
    Z: np.ndarray,
    *,
    model: str,
    n_clusters: int,
    seed: int,
    bootstraps: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    # The same metric space must be used for fitting and nearest-centroid
    # bootstrap assignment; otherwise high-variance PCA axes dominate only the
    # latter and make stability estimates meaningless.
    Z = StandardScaler().fit_transform(np.asarray(Z, dtype=np.float64))
    base = fit_subtypes(
        Z,
        model=model,
        n_clusters=n_clusters,
        seed=seed,
        latent_dim=min(8, Z.shape[1]),
        epochs=25,
        n_bootstraps=max(10, bootstraps),
    )
    labels = np.asarray(base.labels, dtype=int)
    centroids = np.vstack([Z[labels == group].mean(axis=0) for group in range(n_clusters)])
    rng = np.random.default_rng(seed + 1)
    stability: list[float] = []
    sample_size = max(n_clusters * 5, int(math.ceil(0.8 * len(Z))))
    for repeat in range(bootstraps):
        indices = np.sort(rng.choice(len(Z), size=sample_size, replace=False))
        fitted = fit_subtypes(
            Z[indices],
            model=model,
            n_clusters=n_clusters,
            seed=seed + 100 + repeat,
            latent_dim=min(8, Z.shape[1]),
            epochs=15,
            n_bootstraps=max(10, bootstraps),
        )
        sample_labels = np.asarray(fitted.labels, dtype=int)
        sample_centroids = np.vstack(
            [Z[indices][sample_labels == group].mean(axis=0) for group in range(n_clusters)]
        )
        assigned = np.argmin(
            ((Z[:, None, :] - sample_centroids[None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        stability.append(float(adjusted_rand_score(labels, assigned)))
    return labels, centroids, float(np.mean(stability)), float(base.metrics["silhouette"])


def clinical_separation(labels: np.ndarray, phenotype: np.ndarray) -> tuple[float, float, int]:
    table = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(phenotype, name="phenotype"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 1.0, 0.0, int(table.sum(axis=1).min()) if len(table) else 0
    chi2, p_value, _dof, _expected = chi2_contingency(table.to_numpy())
    n = float(table.to_numpy().sum())
    denom = max(1.0, min(table.shape[0] - 1, table.shape[1] - 1))
    cramer_v = math.sqrt(max(0.0, float(chi2)) / (n * denom))
    return float(p_value), float(cramer_v), int(table.sum(axis=1).min())


def multivariate_clinical_separation(
    labels: np.ndarray,
    symptoms: np.ndarray,
    *,
    dimension_names: Sequence[str] = SUBTYPING_CLINICAL_DIMENSIONS,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test imaging-only clusters against withheld continuous symptoms."""

    labels = np.asarray(labels, dtype=int)
    symptoms = np.asarray(symptoms, dtype=np.float64)
    finite = np.isfinite(symptoms).all(axis=1)
    labels = labels[finite]
    symptoms = symptoms[finite]
    groups = np.unique(labels)
    minimum_cluster = min((int(np.sum(labels == group)) for group in groups), default=0)
    if len(groups) < 2 or len(symptoms) <= len(groups):
        return {
            "p_value": 1.0,
            "pseudo_f": 0.0,
            "multivariate_r2": 0.0,
            "minimum_cluster": minimum_cluster,
            "strongest_dimension": "",
            "strongest_dimension_eta2": 0.0,
            "n": int(len(symptoms)),
        }

    center = symptoms.mean(axis=0)
    scale = symptoms.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    standardized = (symptoms - center) / scale

    def statistic(candidate_labels: np.ndarray) -> tuple[float, float, np.ndarray]:
        overall = standardized.mean(axis=0)
        total_by_dimension = np.sum((standardized - overall) ** 2, axis=0)
        between_by_dimension = np.zeros(standardized.shape[1], dtype=np.float64)
        for group in groups:
            subset = standardized[candidate_labels == group]
            if len(subset):
                between_by_dimension += len(subset) * (subset.mean(axis=0) - overall) ** 2
        total = float(total_by_dimension.sum())
        between = float(between_by_dimension.sum())
        residual = max(0.0, total - between)
        numerator_df = max(1, len(groups) - 1)
        denominator_df = max(1, len(standardized) - len(groups))
        pseudo_f = (between / numerator_df) / max(residual / denominator_df, 1e-12)
        r2 = between / total if total > 0 else 0.0
        eta2 = np.divide(
            between_by_dimension,
            total_by_dimension,
            out=np.zeros_like(between_by_dimension),
            where=total_by_dimension > 0,
        )
        return float(pseudo_f), float(r2), eta2

    observed_f, observed_r2, eta2 = statistic(labels)
    rng = np.random.default_rng(seed)
    null = np.empty(max(0, int(permutations)), dtype=np.float64)
    for index in range(len(null)):
        null[index] = statistic(rng.permutation(labels))[0]
    p_value = (
        float((1 + np.sum(null >= observed_f)) / (len(null) + 1))
        if len(null)
        else 1.0
    )
    strongest = int(np.argmax(eta2)) if len(eta2) else 0
    names = tuple(dimension_names)
    return {
        "p_value": p_value,
        "pseudo_f": observed_f,
        "multivariate_r2": observed_r2,
        "minimum_cluster": minimum_cluster,
        "strongest_dimension": names[strongest] if strongest < len(names) else "",
        "strongest_dimension_eta2": float(eta2[strongest]) if len(eta2) else 0.0,
        "n": int(len(symptoms)),
    }


def external_cluster_metrics(
    Z: np.ndarray,
    phenotype: np.ndarray,
    *,
    centroids: np.ndarray,
    discovery_rates: np.ndarray,
) -> dict[str, Any]:
    labels = np.argmin(
        ((Z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    p_value, cramer_v, minimum_cluster = clinical_separation(labels, phenotype)
    external_rates = np.asarray(
        [phenotype[labels == group].mean() if np.any(labels == group) else np.nan for group in range(len(centroids))]
    )
    finite = np.isfinite(discovery_rates) & np.isfinite(external_rates)
    concordance = (
        float(spearmanr(discovery_rates[finite], external_rates[finite]).statistic)
        if finite.sum() >= 2
        else float("nan")
    )
    return {
        "n": int(len(phenotype)),
        "p_value": p_value,
        "cramer_v": cramer_v,
        "minimum_cluster": minimum_cluster,
        "rate_concordance": concordance,
    }


def build_subtyping_tables(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    discovery_metadata = load_ucla_subtyping_metadata(
        cohort_metadata("UCLA"), args.ucla_phenotype_root
    )
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    cluster_token = "-".join(map(str, args.cluster_counts))
    checkpoint_path = (
        output_dir
        / "work"
        / (
            "subtyping_symptom_v2_"
            f"boot{args.subtype_bootstraps}_perm{args.subtype_permutations}_k{cluster_token}.jsonl"
        )
    )
    completed = load_checkpoint(checkpoint_path)
    total = (
        len(args.atlases)
        * len(VIEW_FEATURES)
        * len(SUBTYPING_MODELS)
        * len(args.cluster_counts)
    )
    progress = 0

    for atlas in args.atlases:
        for feature_family in VIEW_FEATURES:
            X_raw, cov, symptoms, _ = aligned_subtyping_view(
                atlas, feature_family, discovery_metadata
            )
            representation = CovariateResidualizedPCA(
                max_components=10,
                robust=True,
            ).fit(
                append_covariates(X_raw, cov)
            )
            Z = representation.transform(append_covariates(X_raw, cov))
            for model in SUBTYPING_MODELS:
                for n_clusters in args.cluster_counts:
                    identity = {
                        "disease": "adhd_bipolar_schizophrenia_spectrum",
                        "atlas": atlas,
                        "feature_family": feature_family,
                        "model": model,
                        "cluster_count": str(n_clusters),
                    }
                    progress += 1
                    key = _identity_key(identity)
                    if key in completed:
                        payload = completed[key]
                        rows.append(identity)
                        internal_rows.append(payload["internal"])
                        external_rows.append(payload["external"])
                        print(
                            f"[subtyping {progress}/{total}] resume {atlas} {feature_family} {model} k={n_clusters}",
                            flush=True,
                        )
                        continue
                    seed = stable_seed(args.seed, "subtyping", atlas, feature_family, model, n_clusters)
                    try:
                        labels, centroids, stability, silhouette = cluster_solution(
                            Z,
                            model=model,
                            n_clusters=n_clusters,
                            seed=seed,
                            bootstraps=args.subtype_bootstraps,
                        )
                        clinical = multivariate_clinical_separation(
                            labels,
                            symptoms,
                            permutations=args.subtype_permutations,
                            seed=seed + 7,
                        )
                        error = ""
                    except Exception as exc:
                        stability = silhouette = float("nan")
                        clinical = {
                            "p_value": 1.0,
                            "pseudo_f": float("nan"),
                            "multivariate_r2": float("nan"),
                            "minimum_cluster": 0,
                            "strongest_dimension": "",
                            "strongest_dimension_eta2": float("nan"),
                            "n": 0,
                        }
                        error = f"{type(exc).__name__}: {exc}"
                    internal_payload = {
                            "stability_ari": stability,
                            "silhouette": silhouette,
                            "clinical_permutation_p": clinical["p_value"],
                            "clinical_pseudo_f": clinical["pseudo_f"],
                            "clinical_multivariate_r2": clinical["multivariate_r2"],
                            "strongest_clinical_dimension": clinical["strongest_dimension"],
                            "strongest_dimension_eta2": clinical["strongest_dimension_eta2"],
                            "minimum_cluster_size": clinical["minimum_cluster"],
                            "n_discovery": int(len(symptoms)),
                            "error": error,
                        }
                    external_payload = {
                            "executable": False,
                            "validated": False,
                            "not_applicable": True,
                            "reason": (
                                "external validation is explicitly exempt for disease_subtyping in "
                                "the frozen supplemental protocol"
                            ),
                        }
                    rows.append(identity)
                    internal_rows.append(internal_payload)
                    external_rows.append(external_payload)
                    append_checkpoint(
                        checkpoint_path,
                        {
                            "identity": identity,
                            "internal": internal_payload,
                            "external": external_payload,
                        },
                    )
                    print(
                        f"[subtyping {progress}/{total}] done {atlas} {feature_family} {model} k={n_clusters}",
                        flush=True,
                    )

    compact_checkpoint(checkpoint_path)
    factors = attach_candidate_ids(
        pd.DataFrame(rows),
        task="disease_subtyping",
        identity_fields=protocol_for("disease_subtyping").factor_fields,
    )
    internal = pd.DataFrame(internal_rows)
    internal["q_value"] = _fdr(internal["clinical_permutation_p"])
    required_cluster_size = np.ceil(0.10 * internal["n_discovery"]).clip(lower=10)
    internal["validated"] = (
        internal["stability_ari"].ge(0.60)
        & internal["silhouette"].ge(0.10)
        & internal["clinical_multivariate_r2"].ge(0.08)
        & internal["minimum_cluster_size"].ge(required_cluster_size)
        & internal["q_value"].lt(0.05)
    )
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])

    external = pd.DataFrame(external_rows)
    external.insert(0, "candidate_id", factors["candidate_id"])

    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=("disease", "atlas", "feature_family", "model"),
        case_study_id="disease_subtyping",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    return export_and_benchmark(
        args,
        task="disease_subtyping",
        public=public,
        internal=internal,
        external=external,
        output_dir=output_dir,
        kg_audit=kg_audit,
        provenance={
            "discovery": "UCLA ADHD plus bipolar disorder plus schizophrenia",
            "clinical_validation": (
                "imaging-only clustering followed by a permutation MANOVA-like test against "
                "five continuous symptom dimensions withheld from clustering"
            ),
            "clinical_dimensions": list(SUBTYPING_CLINICAL_DIMENSIONS),
            "phenotype_source": UCLA_PHENOTYPE_SOURCE,
            "external_status": (
                "not required by protocol; diagnostic enrichment is deliberately not accepted "
                "as subtype replication"
            ),
            "covariate_adjustment": "age and sex residualization",
            "imaging_representation": (
                "10-component PCA after robust 10th-90th percentile scaling and fixed +/-5 clipping"
            ),
            "bootstrap_metric_space": "single standardized PCA space shared by fitting and centroid assignment",
            "subtype_bootstraps": args.subtype_bootstraps,
            "subtype_permutations": args.subtype_permutations,
            "validation_thresholds": {
                "stability_ari": 0.60,
                "silhouette": 0.10,
                "clinical_multivariate_r2": 0.08,
                "minimum_cluster_fraction": 0.10,
                "minimum_cluster_floor": 10,
                "global_fdr_q": 0.05,
            },
        },
    )


def export_and_benchmark(
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
        provenance={
            **provenance,
            "created_at": utc_now(),
            "protocol": asdict(protocol),
        },
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
        "schema_version": "psychiatric-case-study-closure.v1",
        "created_at": utc_now(),
        "task": task,
        "status": "complete",
        "table_manifest": table_manifest,
        "benchmark_manifest": benchmark_manifest,
    }
    path = output_dir / "closure_manifest.json"
    path.write_text(json.dumps(closure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return closure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("differential_diagnosis", "disease_subtyping", "both"),
        default="both",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
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
    parser.add_argument("--subtype-bootstraps", type=int, default=10)
    parser.add_argument("--subtype-permutations", type=int, default=499)
    parser.add_argument("--cluster-counts", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument(
        "--ucla-phenotype-root",
        type=Path,
        default=DEFAULT_UCLA_PHENOTYPE_ROOT,
    )
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_root != DEFAULT_DATA_ROOT:
        raise ValueError("custom --data-root is not yet supported by the frozen cohort registry")
    if args.quick:
        args.atlases = args.atlases[:1]
        args.cv_repeats = 1
        args.permutations = min(args.permutations, 3)
        args.subtype_bootstraps = min(args.subtype_bootstraps, 3)
        args.subtype_permutations = min(args.subtype_permutations, 19)
        args.cluster_counts = args.cluster_counts[:1]
    tasks = (
        ("differential_diagnosis", "disease_subtyping")
        if args.task == "both"
        else (args.task,)
    )
    manifests: dict[str, Any] = {}
    for task in tasks:
        output_dir = args.output_root / task
        output_dir.mkdir(parents=True, exist_ok=True)
        if task == "differential_diagnosis":
            manifests[task] = build_differential_tables(args, output_dir)
        else:
            manifests[task] = build_subtyping_tables(args, output_dir)
    print(json.dumps(manifests, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
