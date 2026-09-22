"""Run the ADNI progression, prognosis, and imaging-genetics closed loops.

ADNI1/GO/2 is the discovery phase and ADNI3 is a non-overlapping held-out
acquisition/genotyping phase.  Candidate definitions, ranking inputs, internal
outcomes, and external outcomes are exported separately before invoking the
shared 10-trial closed-loop benchmark.  No ADNI3 labels are used for model
selection, candidate ranking, or hyperparameter tuning.
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
from scipy.stats import norm, pearsonr, spearmanr, t
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from core.scripts.case_study_candidate_tables import (
    attach_candidate_ids,
    export_table_bundle,
    score_public_candidates,
)
from core.scripts.case_study_closed_loop import run_benchmark
from core.scripts.case_study_closed_loop_specs import protocol_for
from models.statistical_ml.estimators import make_estimator as make_tabular_estimator
from models.survival_models.estimators import make_survival_estimator
from models.survival_models.metrics import concordance_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = (
    Path(r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc")
    / "case2_adni_genetics_v1"
    / "experiment_tables"
    / "case2_adni_multimodal_v1"
)
DEFAULT_OUTPUT_ROOT = Path(r"\\192.168.3.61\data\Public Dataset\case_study_closed_loop_v1")
DEFAULT_KG = ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"

GENETICS_FILE = "case2_adni_subject_genetics_covariates.parquet"
VISITS_FILE = "case2_adni_clinical_visits.parquet"
IMAGING_FILE = "case2_adni_primary_imaging_markers_long.parquet"

DEMOGRAPHIC_FEATURES = ("AGE", "sex_binary", "PTEDUCAT", "APOE4")
COGNITIVE_FEATURES = ("MMSE", "ADAS13", "CDRSB")
IMAGING_MARKERS = (
    "Hippocampus_icv",
    "Entorhinal_icv",
    "WholeBrain_icv",
    "Ventricles_icv",
    "MidTemp_icv",
    "Fusiform_icv",
)
GLOBAL_GENETIC_FEATURES = (
    "apoe_e4_dosage",
    "ad_prs_p5em08_avg",
    "ad_prs_p1em05_avg",
    "ad_prs_p1em03_avg",
    "ad_prs_p5em02_avg",
)
CURATED_PATHWAY_PREFIX = "pathway_prs__curated__"
CURATED_PATHWAY_SUFFIX = "__p5em02"
PROGNOSTIC_BASIC_COVARIATES = DEMOGRAPHIC_FEATURES
PROGNOSTIC_MARKER_MODEL = "cox_adjusted"

PREDICTION_FEATURE_SETS = {
    "clinical_plus_structural_mri": (*DEMOGRAPHIC_FEATURES, *COGNITIVE_FEATURES, *IMAGING_MARKERS),
    "clinical_plus_genetic_risk": (*DEMOGRAPHIC_FEATURES, *COGNITIVE_FEATURES, *GLOBAL_GENETIC_FEATURES),
    "clinical_plus_structural_and_genetic": (
        *DEMOGRAPHIC_FEATURES,
        *COGNITIVE_FEATURES,
        *IMAGING_MARKERS,
        *GLOBAL_GENETIC_FEATURES,
    ),
    "structural_mri_only": (*DEMOGRAPHIC_FEATURES, *IMAGING_MARKERS),
}
CLINICAL_BASELINE = (*DEMOGRAPHIC_FEATURES, *COGNITIVE_FEATURES)
PROGRESSION_MODELS = (
    "logistic",
    "elastic_net",
    "svm",
    "temporal_mlp",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)
SURVIVAL_MODELS = (
    "cox",
    "elastic_net_cox",
    "deepsurv",
    "random_survival_forest",
    "xgboost_survival",
)
ASSOCIATION_MODELS = ("association_glm", "rank_association", "robust_huber")
HORIZONS = (2, 3, 5)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def normalize_diagnosis(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text in {"emci", "lmci", "mci"} or "mild cognitive" in text:
        return "MCI"
    if text in {"ad", "dementia", "alzheimer's disease"} or "dement" in text:
        return "Dementia"
    if text in {"cn", "normal", "cognitively normal", "smc"}:
        return "CN"
    return str(value or "")


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_adni_cohort(
    data_root: Path,
    *,
    phase: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if phase not in {"all", "discovery", "external"}:
        raise ValueError(f"unknown ADNI phase: {phase!r}")
    genetics = pd.read_parquet(data_root / GENETICS_FILE)
    visits = pd.read_parquet(data_root / VISITS_FILE)
    visits["clinical_date"] = pd.to_datetime(visits["clinical_date"], errors="coerce")
    visits = visits.dropna(subset=["subject_id", "clinical_date"]).sort_values(
        ["subject_id", "clinical_date"]
    )
    baseline = visits.drop_duplicates("subject_id", keep="first").copy()
    baseline_columns = [
        "subject_id",
        "MMSE",
        "ADAS13",
        "CDRSB",
        "Hippocampus",
        "Entorhinal",
        "WholeBrain",
        "Ventricles",
        "MidTemp",
        "Fusiform",
        "ICV",
    ]
    cohort = genetics.merge(
        baseline[baseline_columns], on="subject_id", how="left", validate="one_to_one"
    )
    cohort["phase"] = np.where(cohort["COLPROT"].astype(str).eq("ADNI3"), "external", "discovery")
    cohort["baseline_dx"] = cohort["DX_bl"].map(normalize_diagnosis)
    _numeric(
        cohort,
        [
            *DEMOGRAPHIC_FEATURES,
            *COGNITIVE_FEATURES,
            *GLOBAL_GENETIC_FEATURES,
            "ICV",
            "Hippocampus",
            "Entorhinal",
            "WholeBrain",
            "Ventricles",
            "MidTemp",
            "Fusiform",
            *[column for column in cohort if column.startswith(CURATED_PATHWAY_PREFIX)],
            *[f"PC{index}" for index in range(1, 6)],
        ],
    )
    denominator = cohort["ICV"].replace(0, np.nan)
    for marker in ("Hippocampus", "Entorhinal", "WholeBrain", "Ventricles", "MidTemp", "Fusiform"):
        cohort[f"{marker}_icv"] = cohort[marker] / denominator

    if phase != "all":
        cohort = cohort.loc[cohort["phase"].eq(phase)].copy()
        visits = visits.loc[visits["subject_id"].isin(cohort["subject_id"])].copy()

    survival_rows: list[dict[str, Any]] = []
    genetics_index = cohort.set_index("subject_id", drop=False)
    for subject_id, subject_visits in visits.groupby("subject_id"):
        if subject_id not in genetics_index.index:
            continue
        subject = genetics_index.loc[subject_id]
        if isinstance(subject, pd.DataFrame):
            subject = subject.iloc[0]
        if normalize_diagnosis(subject["DX_bl"]) != "MCI":
            continue
        ordered = subject_visits.sort_values("clinical_date")
        baseline_date = ordered.iloc[0]["clinical_date"]
        last_date = ordered.iloc[-1]["clinical_date"]
        diagnosis = ordered["DX"].map(normalize_diagnosis)
        event_dates = ordered.loc[
            diagnosis.eq("Dementia") & ordered["clinical_date"].gt(baseline_date),
            "clinical_date",
        ]
        event = int(not event_dates.empty)
        endpoint = event_dates.iloc[0] if event else last_date
        duration = float((endpoint - baseline_date).days / 365.25)
        followup = float((last_date - baseline_date).days / 365.25)
        if duration <= 0:
            continue
        survival_rows.append(
            {
                "subject_id": str(subject_id),
                "duration_years": duration,
                "followup_years": followup,
                "event": event,
            }
        )
    survival = pd.DataFrame(survival_rows).merge(
        cohort, on="subject_id", how="inner", validate="one_to_one"
    )
    audit = {
        "created_at": utc_now(),
        "genetic_subjects": int(len(genetics)),
        "clinical_visits": int(len(visits)),
        "cohort_by_phase": cohort["phase"].value_counts().to_dict(),
        "mci_survival_by_phase": survival.groupby("phase").size().to_dict(),
        "mci_events_by_phase": survival.groupby("phase")["event"].sum().to_dict(),
        "phase_rule": "ADNI3 external; ADNI1/GO/2 discovery",
    }
    return cohort, survival, audit


def progression_at_horizon(survival: pd.DataFrame, horizon: int) -> pd.DataFrame:
    eligible = (
        (survival["event"].eq(1) & survival["duration_years"].le(horizon))
        | survival["followup_years"].ge(horizon)
    )
    frame = survival.loc[eligible].copy()
    frame["label"] = (
        frame["event"].eq(1) & frame["duration_years"].le(horizon)
    ).astype(int)
    return frame.reset_index(drop=True)


def _classifier_pipeline(model: str, *, seed: int, params: dict[str, Any] | None = None) -> Pipeline:
    if model in {"random_forest", "extra_trees", "hist_gradient_boosting"}:
        pipeline = make_tabular_estimator(model, "classification", seed=seed)
        if params:
            pipeline.set_params(**params)
        return pipeline
    if model == "logistic":
        estimator = LogisticRegression(max_iter=3000, solver="liblinear", random_state=seed)
    elif model == "elastic_net":
        estimator = LogisticRegression(
            max_iter=5000,
            solver="saga",
            penalty="elasticnet",
            random_state=seed,
        )
    elif model == "svm":
        estimator = SVC(kernel="linear", probability=True, random_state=seed)
    elif model == "temporal_mlp":
        estimator = MLPClassifier(max_iter=600, early_stopping=True, random_state=seed)
    else:
        raise ValueError(model)
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline


def _classifier_grid(model: str) -> dict[str, list[Any]]:
    if model == "logistic":
        return {"model__C": [0.1, 1.0, 10.0]}
    if model == "elastic_net":
        return {"model__C": [0.1, 1.0, 10.0], "model__l1_ratio": [0.1, 0.5, 0.9]}
    if model == "svm":
        return {"model__C": [0.1, 1.0, 10.0]}
    if model == "temporal_mlp":
        return {"model__hidden_layer_sizes": [(32,), (64, 32)], "model__alpha": [0.0001, 0.01]}
    if model in {"random_forest", "extra_trees"}:
        return {
            "model__n_estimators": [200],
            "model__max_depth": [3, None],
            "model__min_samples_leaf": [2, 5],
        }
    if model == "hist_gradient_boosting":
        return {
            "model__max_iter": [200],
            "model__max_leaf_nodes": [7, 15],
            "model__l2_regularization": [0.1, 1.0],
        }
    raise ValueError(model)


def _scores(estimator: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X)[:, 1])
    return np.asarray(estimator.decision_function(X))


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


def _mode_params(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    serialized = [json.dumps(value, sort_keys=True) for value in values]
    return json.loads(max(set(serialized), key=serialized.count)) if serialized else {}


def classification_outer_splits(
    y: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Freeze outer folds so candidate and clinical-baseline deltas are paired."""

    y = np.asarray(y, dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=repeats,
        random_state=seed,
    )
    dummy = np.zeros((len(y), 1), dtype=float)
    return tuple(
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in splitter.split(dummy, y)
    )


def nested_classification(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    seed: int,
    repeats: int,
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    split_seed: int | None = None,
) -> dict[str, Any]:
    if min(np.bincount(y.astype(int), minlength=2)) < 5:
        raise ValueError("fewer than five observations in one class")
    if outer_splits is None:
        outer_splits = classification_outer_splits(y, seed=seed, repeats=repeats)
    if not outer_splits:
        raise ValueError("outer_splits cannot be empty")
    inner_seed = seed if split_seed is None else int(split_seed)
    fold_auc: list[float] = []
    fold_auprc: list[float] = []
    parameters: list[dict[str, Any]] = []
    truth: list[int] = []
    prediction: list[float] = []
    for fold, (train, test) in enumerate(outer_splits, start=1):
        search = GridSearchCV(
            _classifier_pipeline(model, seed=seed + fold),
            _classifier_grid(model),
            scoring="roc_auc",
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=inner_seed + fold,
            ),
            n_jobs=1,
        )
        search.fit(X[train], y[train])
        score = _scores(search.best_estimator_, X[test])
        fold_auc.append(float(roc_auc_score(y[test], score)))
        fold_auprc.append(float(average_precision_score(y[test], score)))
        parameters.append(dict(search.best_params_))
        truth.extend(y[test].astype(int).tolist())
        prediction.extend(score.tolist())
    auc, auc_low, auc_high = _mean_ci(fold_auc)
    auprc, auprc_low, auprc_high = _mean_ci(fold_auprc)
    return {
        "auroc": auc,
        "auroc_ci_low": auc_low,
        "auroc_ci_high": auc_high,
        "auprc": auprc,
        "auprc_ci_low": auprc_low,
        "auprc_ci_high": auprc_high,
        "fold_auroc": np.asarray(fold_auc),
        "selected_params": _mode_params(parameters),
        "oof_true": np.asarray(truth, dtype=int),
        "oof_score": np.asarray(prediction, dtype=float),
    }


def score_permutation_p(
    y: np.ndarray,
    score: np.ndarray,
    *,
    observed: float,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [roc_auc_score(rng.permutation(y), score) for _ in range(permutations)], dtype=float
    )
    exact = float((1 + np.sum(null >= observed)) / (permutations + 1))
    sd = float(null.std(ddof=1))
    tail = float(norm.sf((observed - float(null.mean())) / sd)) if sd > 0 else 1.0
    return {
        "permutation_p_exact": exact,
        "permutation_p_tail": tail,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_sd": sd,
    }


def bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    *,
    baseline_score: np.ndarray | None,
    seed: int,
    samples: int = 500,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if min(np.bincount(y, minlength=2)) < 5:
        return {"n": int(len(y)), "events": int(y.sum()), "auroc": float("nan"), "auroc_ci_low": float("nan"), "auroc_ci_high": float("nan"), "auprc": float("nan"), "auroc_improvement": float("nan"), "auroc_improvement_ci_low": float("nan")}
    observed = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    improvements: list[float] = []
    for _ in range(samples):
        index = rng.integers(0, len(y), size=len(y))
        if np.unique(y[index]).size < 2:
            continue
        auc = float(roc_auc_score(y[index], score[index]))
        aucs.append(auc)
        if baseline_score is not None:
            improvements.append(auc - float(roc_auc_score(y[index], baseline_score[index])))
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "auroc": observed,
        "auroc_ci_low": float(np.quantile(aucs, 0.025)),
        "auroc_ci_high": float(np.quantile(aucs, 0.975)),
        "auprc": float(average_precision_score(y, score)),
        "auroc_improvement": (
            observed - float(roc_auc_score(y, baseline_score)) if baseline_score is not None else float("nan")
        ),
        "auroc_improvement_ci_low": (
            float(np.quantile(improvements, 0.025)) if improvements else float("nan")
        ),
    }


def _fdr(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    values[~np.isfinite(values)] = 1.0
    return multipletests(values, method="fdr_bh")[1]


def family_fdr(frame: pd.DataFrame, p_column: str, family_fields: Sequence[str]) -> np.ndarray:
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
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            result[_identity_key(payload["identity"])] = payload
    return result


def append_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
    tables = output_dir / "tables"
    table_manifest = export_table_bundle(
        task=task,
        public=public,
        internal=internal,
        external=external,
        factor_fields=protocol.factor_fields,
        output_dir=tables,
        provenance={**provenance, "created_at": utc_now(), "protocol": asdict(protocol)},
        kg_audit=kg_audit,
    )
    budgets = sorted({value for value in protocol.budgets if value <= len(public)} | {len(public)})
    benchmark = run_benchmark(
        argparse.Namespace(
            task=task,
            public_candidates=tables / "public_candidates.csv",
            internal_outcomes=tables / "internal_outcomes.csv",
            external_outcomes=tables / "external_outcomes.csv",
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
    )
    closure = {
        "schema_version": "adni-case-study-closure.v1",
        "created_at": utc_now(),
        "task": task,
        "status": "complete",
        "table_manifest": table_manifest,
        "benchmark_manifest": benchmark,
    }
    (output_dir / "closure_manifest.json").write_text(
        json.dumps(closure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return closure


def build_progression(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    survival: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint = (
        output_dir
        / "work"
        / f"progression_paired_v4_cv{args.cv_repeats}_perm{args.permutations}.jsonl"
    )
    completed = load_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    baseline_cache: dict[tuple[int, str], dict[str, Any]] = {}
    feature_sets = {
        name: PREDICTION_FEATURE_SETS[name] for name in args.feature_sets
    }
    total = len(args.horizons) * len(feature_sets) * len(args.progression_models)
    progress = 0
    for horizon in args.horizons:
        frame = progression_at_horizon(survival, horizon)
        discovery = frame.loc[frame["phase"].eq("discovery")].reset_index(drop=True)
        external = frame.loc[frame["phase"].eq("external")].reset_index(drop=True)
        for model in args.progression_models:
            key = (horizon, model)
            X_base = discovery[list(CLINICAL_BASELINE)].to_numpy(dtype=float)
            y = discovery["label"].to_numpy(dtype=int)
            cv_seed = stable_seed(args.seed, "progression-paired-cv", horizon)
            outer_splits = classification_outer_splits(
                y,
                seed=cv_seed,
                repeats=args.cv_repeats,
            )
            baseline_cv = nested_classification(
                X_base,
                y,
                model=model,
                seed=cv_seed,
                repeats=args.cv_repeats,
                outer_splits=outer_splits,
                split_seed=cv_seed,
            )
            baseline_final = _classifier_pipeline(
                model,
                seed=stable_seed(args.seed, "progression-base-final", horizon, model),
                params=baseline_cv["selected_params"],
            ).fit(X_base, y)
            baseline_cache[key] = {
                "cv": baseline_cv,
                "final": baseline_final,
                "cv_seed": cv_seed,
                "outer_splits": outer_splits,
            }
        for feature_family, features in feature_sets.items():
            X = discovery[list(features)].to_numpy(dtype=float)
            y = discovery["label"].to_numpy(dtype=int)
            X_ext = external[list(features)].to_numpy(dtype=float)
            y_ext = external["label"].to_numpy(dtype=int)
            X_ext_base = external[list(CLINICAL_BASELINE)].to_numpy(dtype=float)
            for model in args.progression_models:
                identity = {
                    "outcome": "mci_to_dementia",
                    "horizon": f"{horizon}y",
                    "feature_family": feature_family,
                    "model": model,
                }
                progress += 1
                key = _identity_key(identity)
                if key in completed:
                    payload = completed[key]
                    rows.append(identity); internal_rows.append(payload["internal"]); external_rows.append(payload["external"])
                    print(f"[progression {progress}/{total}] resume {identity}", flush=True)
                    continue
                seed = stable_seed(args.seed, "progression", horizon, feature_family, model)
                paired = baseline_cache[(horizon, model)]
                try:
                    cv = nested_classification(
                        X,
                        y,
                        model=model,
                        seed=int(paired["cv_seed"]),
                        repeats=args.cv_repeats,
                        outer_splits=paired["outer_splits"],
                        split_seed=int(paired["cv_seed"]),
                    )
                    baseline_cv = paired["cv"]
                    improvement = cv["fold_auroc"] - baseline_cv["fold_auroc"]
                    imp_mean, imp_low, imp_high = _mean_ci(improvement)
                    permutation = score_permutation_p(
                        cv["oof_true"], cv["oof_score"], observed=cv["auroc"], permutations=args.permutations, seed=seed + 17
                    )
                    final = _classifier_pipeline(
                        model,
                        seed=seed + 31,
                        params=cv["selected_params"],
                    ).fit(X, y)
                    error = ""
                except Exception as exc:
                    cv = {name: float("nan") for name in ("auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high")}
                    cv["selected_params"] = {}
                    imp_mean = imp_low = imp_high = float("nan")
                    permutation = {"permutation_p_exact": 1.0, "permutation_p_tail": 1.0, "permutation_null_mean": float("nan"), "permutation_null_sd": float("nan")}
                    final = None
                    error = f"{type(exc).__name__}: {exc}"
                metric = {
                    **{
                        name: float("nan")
                        for name in (
                            "auroc",
                            "auroc_ci_low",
                            "auroc_ci_high",
                            "auprc",
                            "auroc_improvement",
                            "auroc_improvement_ci_low",
                        )
                    },
                    "n": 0,
                    "events": 0,
                }
                external_error = "external phase withheld"
                if final is not None and len(y_ext):
                    try:
                        metric = bootstrap_auc(
                            y_ext,
                            _scores(final, X_ext),
                            baseline_score=_scores(
                                baseline_cache[(horizon, model)]["final"],
                                X_ext_base,
                            ),
                            seed=seed + 43,
                        )
                        external_error = ""
                    except Exception as exc:
                        external_error = f"{type(exc).__name__}: {exc}"
                internal_payload = {
                    **{name: cv[name] for name in ("auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high")},
                    "clinical_auroc_improvement": imp_mean,
                    "clinical_auroc_improvement_ci_low": imp_low,
                    "clinical_auroc_improvement_ci_high": imp_high,
                    **permutation,
                    "selected_params": json.dumps(cv["selected_params"], sort_keys=True),
                    "split_seed": int(paired["cv_seed"]),
                    "n_folds": int(len(paired["outer_splits"])),
                    "n_discovery": int(len(y)),
                    "events_discovery": int(y.sum()),
                    "error": error,
                }
                external_payload = {
                    "executable": bool(metric["n"] >= 20 and metric["events"] >= 5),
                    "n_external": int(metric["n"]),
                    "events_external": int(metric["events"]),
                    **{name: metric[name] for name in ("auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auroc_improvement", "auroc_improvement_ci_low")},
                    "error": external_error,
                }
                rows.append(identity); internal_rows.append(internal_payload); external_rows.append(external_payload)
                append_checkpoint(checkpoint, {"identity": identity, "internal": internal_payload, "external": external_payload})
                print(f"[progression {progress}/{total}] done {identity}", flush=True)

    protocol = protocol_for("progression_prediction")
    factors = attach_candidate_ids(pd.DataFrame(rows), task="progression_prediction", identity_fields=protocol.factor_fields)
    internal = pd.DataFrame(internal_rows)
    joined = pd.concat([factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1)
    internal["q_value"] = family_fdr(joined, "permutation_p_tail", ("horizon", "model"))
    internal["validated"] = internal["auroc_ci_low"].gt(0.5) & internal["clinical_auroc_improvement_ci_low"].gt(0) & internal["q_value"].lt(0.05)
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])
    external = pd.DataFrame(external_rows)
    external["validated"] = external["executable"].astype(bool) & external["auroc_ci_low"].gt(0.5) & external["auroc_improvement"].gt(0)
    external.insert(0, "candidate_id", factors["candidate_id"])
    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=("outcome", "horizon", "feature_family", "model"),
        case_study_id="progression_prediction",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    return _export_and_benchmark(
        args, task="progression_prediction", public=public, internal=internal, external=external,
        output_dir=output_dir, kg_audit=kg_audit,
        provenance={"cohort_audit": audit, "discovery": "ADNI1/GO/2 baseline MCI", "external": "withheld during internal tuning" if args.internal_only else "ADNI3 baseline MCI", "horizons_years": list(args.horizons), "clinical_baseline": list(CLINICAL_BASELINE), "paired_cv": True},
    )


class PenalizedCox:
    def __init__(self, *, alpha: float, l1_weight: float):
        self.alpha = alpha
        self.l1_weight = l1_weight
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray) -> "PenalizedCox":
        transformed = self.scaler.fit_transform(self.imputer.fit_transform(X))
        model = PHReg(duration, transformed, status=np.asarray(event, dtype=int))
        self.result_ = model.fit_regularized(alpha=self.alpha, L1_wt=self.l1_weight)
        self.params_ = np.asarray(self.result_.params, dtype=float)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(self.imputer.transform(X))
        return np.asarray(transformed @ self.params_, dtype=float)


def make_survival_model(model: str, *, seed: int):
    if model == "cox":
        return PenalizedCox(alpha=0.01, l1_weight=0.0)
    if model == "elastic_net_cox":
        return PenalizedCox(alpha=0.05, l1_weight=0.5)
    if model == "deepsurv":
        return make_survival_estimator(
            "deepsurv",
            seed=seed,
            hidden_dims=(32, 16),
            epochs=60,
            device="cpu",
        )
    if model == "random_survival_forest":
        return make_survival_estimator(
            "random_survival_forest",
            seed=seed,
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
        )
    if model == "xgboost_survival":
        return make_survival_estimator(
            "xgboost_survival",
            seed=seed,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    raise ValueError(model)


def censored_at_horizon(frame: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    duration = np.minimum(frame["duration_years"].to_numpy(dtype=float), float(horizon))
    event = (frame["event"].eq(1) & frame["duration_years"].le(horizon)).to_numpy(dtype=int)
    return duration, event


def survival_outer_splits(
    event: np.ndarray,
    *,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Freeze survival folds for paired candidate-versus-baseline comparisons."""

    event = np.asarray(event, dtype=int)
    if int(event.sum()) < 5:
        raise ValueError("fewer than five survival events")
    splits = min(5, int(event.sum()), int((event == 0).sum()))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(event), 1), dtype=float)
    return tuple(
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in splitter.split(dummy, event)
    )


def survival_cv(
    X: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    *,
    model: str,
    seed: int,
    outer_splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if int(event.sum()) < 5:
        raise ValueError("fewer than five survival events")
    if outer_splits is None:
        outer_splits = survival_outer_splits(event, seed=seed)
    if not outer_splits:
        raise ValueError("outer_splits cannot be empty")
    fold_c: list[float] = []
    all_duration: list[float] = []
    all_event: list[int] = []
    all_risk: list[float] = []
    for fold, (train, test) in enumerate(outer_splits, start=1):
        estimator = make_survival_model(model, seed=seed + fold)
        estimator.fit(X[train], duration[train], event[train])
        risk = estimator.predict_risk(X[test])
        fold_c.append(float(concordance_index(duration[test], event[test], risk)))
        all_duration.extend(duration[test].tolist())
        all_event.extend(event[test].tolist())
        all_risk.extend(risk.tolist())
    mean, low, high = _mean_ci(fold_c)
    return {
        "c_index": mean,
        "c_index_ci_low": low,
        "c_index_ci_high": high,
        "fold_c_index": np.asarray(fold_c),
        "oof_duration": np.asarray(all_duration),
        "oof_event": np.asarray(all_event, dtype=int),
        "oof_risk": np.asarray(all_risk),
    }


def survival_permutation_p(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    observed: float,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [concordance_index(duration, event, rng.permutation(risk)) for _ in range(permutations)], dtype=float
    )
    exact = float((1 + np.sum(null >= observed)) / (permutations + 1))
    sd = float(null.std(ddof=1))
    return {
        "permutation_p_exact": exact,
        "permutation_p_tail": float(norm.sf((observed - null.mean()) / sd)) if sd > 0 else 1.0,
        "permutation_null_mean": float(null.mean()),
        "permutation_null_sd": sd,
    }


def bootstrap_cindex(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    baseline_risk: np.ndarray | None,
    seed: int,
    samples: int = 500,
) -> dict[str, float | int]:
    observed = float(concordance_index(duration, event, risk))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    improvements: list[float] = []
    for _ in range(samples):
        index = rng.integers(0, len(duration), size=len(duration))
        if event[index].sum() < 2:
            continue
        value = float(concordance_index(duration[index], event[index], risk[index]))
        values.append(value)
        if baseline_risk is not None:
            improvements.append(value - float(concordance_index(duration[index], event[index], baseline_risk[index])))
    return {
        "n": int(len(duration)),
        "events": int(event.sum()),
        "c_index": observed,
        "c_index_ci_low": float(np.quantile(values, 0.025)) if values else float("nan"),
        "c_index_ci_high": float(np.quantile(values, 0.975)) if values else float("nan"),
        "c_index_improvement": observed - float(concordance_index(duration, event, baseline_risk)) if baseline_risk is not None else float("nan"),
        "c_index_improvement_ci_low": float(np.quantile(improvements, 0.025)) if improvements else float("nan"),
    }


def prognostic_marker_columns(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """Return registered marker columns and their outcome-blind scientific family."""

    markers: list[tuple[str, str]] = [
        (marker, "structural_mri") for marker in IMAGING_MARKERS if marker in frame
    ]
    markers.extend(
        (marker, "polygenic_risk_score")
        for marker in GLOBAL_GENETIC_FEATURES
        if marker != "apoe_e4_dosage" and marker in frame
    )
    markers.extend(
        (marker, "pathway_polygenic_risk_score")
        for marker in sorted(frame.columns)
        if marker.startswith(CURATED_PATHWAY_PREFIX)
        and marker.endswith(CURATED_PATHWAY_SUFFIX)
    )
    return tuple(dict.fromkeys(markers))


def prognostic_marker_association(
    frame: pd.DataFrame,
    duration: np.ndarray,
    event: np.ndarray,
    *,
    marker: str,
    covariates: Sequence[str] = PROGNOSTIC_BASIC_COVARIATES,
    seed: int,
    bootstraps: int = 499,
) -> dict[str, float | int]:
    """Fit a covariate-adjusted Cox marker association with subject bootstrap."""

    columns = [*covariates, marker]
    design = frame[columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    valid = (
        np.isfinite(duration)
        & (duration > 0)
        & design[marker].notna().to_numpy()
    )
    design = design.loc[valid].reset_index(drop=True)
    duration = duration[valid]
    event = event[valid]
    if len(design) < 30 or int(event.sum()) < 5:
        raise ValueError("insufficient complete marker observations or survival events")

    for column in covariates:
        median = design[column].median()
        design[column] = design[column].fillna(0.0 if not np.isfinite(median) else median)
    means = design.mean(axis=0)
    scales = design.std(axis=0, ddof=0).replace(0, np.nan)
    standardized = ((design - means) / scales).fillna(0.0).to_numpy(dtype=float)
    marker_index = len(columns) - 1

    fit = PHReg(duration, standardized, status=event, ties="efron").fit(disp=0)
    beta = float(fit.params[marker_index])
    standard_error = float(fit.bse[marker_index])
    log_ci_low = beta - 1.96 * standard_error
    log_ci_high = beta + 1.96 * standard_error

    rng = np.random.default_rng(seed)
    bootstrap_betas: list[float] = []
    for _ in range(max(0, int(bootstraps))):
        index = rng.integers(0, len(duration), size=len(duration))
        if int(event[index].sum()) < 5:
            continue
        try:
            bootstrap_fit = PHReg(
                duration[index],
                standardized[index],
                status=event[index],
                ties="efron",
            ).fit(disp=0)
            value = float(bootstrap_fit.params[marker_index])
            if np.isfinite(value):
                bootstrap_betas.append(value)
        except Exception:
            continue
    bootstrap_array = np.asarray(bootstrap_betas, dtype=float)
    if bootstrap_array.size:
        bootstrap_ci_low = float(np.quantile(bootstrap_array, 0.025))
        bootstrap_ci_high = float(np.quantile(bootstrap_array, 0.975))
        sign_stability = float(np.mean(np.sign(bootstrap_array) == np.sign(beta)))
    else:
        bootstrap_ci_low = bootstrap_ci_high = sign_stability = float("nan")

    return {
        "n": int(len(duration)),
        "events": int(event.sum()),
        "log_hazard_ratio": beta,
        "log_hazard_ratio_ci_low": log_ci_low,
        "log_hazard_ratio_ci_high": log_ci_high,
        "hazard_ratio": float(np.exp(beta)),
        "hazard_ratio_ci_low": float(np.exp(log_ci_low)),
        "hazard_ratio_ci_high": float(np.exp(log_ci_high)),
        "standard_error": standard_error,
        "p_value": float(fit.pvalues[marker_index]),
        "bootstrap_log_hr_ci_low": bootstrap_ci_low,
        "bootstrap_log_hr_ci_high": bootstrap_ci_high,
        "bootstrap_sign_stability": sign_stability,
        "bootstrap_successes": int(bootstrap_array.size),
    }


def build_prognosis_marker_associations(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    survival: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint = (
        output_dir
        / "work"
        / f"prognosis_marker_cox_boot{args.marker_bootstraps}.jsonl"
    )
    completed = load_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    discovery = survival.loc[survival["phase"].eq("discovery")].reset_index(drop=True)
    external = survival.loc[survival["phase"].eq("external")].reset_index(drop=True)
    markers = prognostic_marker_columns(discovery)
    total = len(args.horizons) * len(markers)
    progress = 0

    for horizon in args.horizons:
        duration, event = censored_at_horizon(discovery, horizon)
        duration_ext, event_ext = censored_at_horizon(external, horizon)
        cv_seed = stable_seed(args.seed, "prognosis-marker-paired-cv", horizon)
        outer_splits = survival_outer_splits(event, seed=cv_seed)
        X_base = discovery[list(PROGNOSTIC_BASIC_COVARIATES)].to_numpy(dtype=float)
        baseline_cv = survival_cv(
            X_base,
            duration,
            event,
            model="cox",
            seed=cv_seed,
            outer_splits=outer_splits,
        )
        for marker, feature_family in markers:
            identity = {
                "outcome": "mci_to_dementia",
                "horizon": f"{horizon}y",
                "feature_family": feature_family,
                "marker": marker,
                "model": PROGNOSTIC_MARKER_MODEL,
            }
            progress += 1
            key = _identity_key(identity)
            if key in completed:
                payload = completed[key]
                rows.append(identity)
                internal_rows.append(payload["internal"])
                external_rows.append(payload["external"])
                print(f"[prognosis-marker {progress}/{total}] resume {identity}", flush=True)
                continue

            seed = stable_seed(args.seed, "prognosis-marker", horizon, marker)
            try:
                association = prognostic_marker_association(
                    discovery,
                    duration,
                    event,
                    marker=marker,
                    seed=seed,
                    bootstraps=args.marker_bootstraps,
                )
                X_marker = discovery[
                    [*PROGNOSTIC_BASIC_COVARIATES, marker]
                ].to_numpy(dtype=float)
                marker_cv = survival_cv(
                    X_marker,
                    duration,
                    event,
                    model="cox",
                    seed=cv_seed,
                    outer_splits=outer_splits,
                )
                difference = marker_cv["fold_c_index"] - baseline_cv["fold_c_index"]
                delta, delta_low, delta_high = _mean_ci(difference)
                internal_payload = {
                    **association,
                    "c_index": marker_cv["c_index"],
                    "c_index_ci_low": marker_cv["c_index_ci_low"],
                    "c_index_ci_high": marker_cv["c_index_ci_high"],
                    "covariate_c_index_improvement": delta,
                    "covariate_c_index_improvement_ci_low": delta_low,
                    "covariate_c_index_improvement_ci_high": delta_high,
                    "split_seed": cv_seed,
                    "n_folds": int(len(outer_splits)),
                    "error": "",
                }
            except Exception as exc:
                internal_payload = {
                    "n": 0,
                    "events": 0,
                    "log_hazard_ratio": float("nan"),
                    "log_hazard_ratio_ci_low": float("nan"),
                    "log_hazard_ratio_ci_high": float("nan"),
                    "hazard_ratio": float("nan"),
                    "hazard_ratio_ci_low": float("nan"),
                    "hazard_ratio_ci_high": float("nan"),
                    "standard_error": float("nan"),
                    "p_value": 1.0,
                    "bootstrap_log_hr_ci_low": float("nan"),
                    "bootstrap_log_hr_ci_high": float("nan"),
                    "bootstrap_sign_stability": float("nan"),
                    "bootstrap_successes": 0,
                    "c_index": float("nan"),
                    "c_index_ci_low": float("nan"),
                    "c_index_ci_high": float("nan"),
                    "covariate_c_index_improvement": float("nan"),
                    "covariate_c_index_improvement_ci_low": float("nan"),
                    "covariate_c_index_improvement_ci_high": float("nan"),
                    "split_seed": cv_seed,
                    "n_folds": int(len(outer_splits)),
                    "error": f"{type(exc).__name__}: {exc}",
                }

            external_payload: dict[str, Any] = {
                "executable": False,
                "n": 0,
                "events": 0,
                "log_hazard_ratio": float("nan"),
                "log_hazard_ratio_ci_low": float("nan"),
                "log_hazard_ratio_ci_high": float("nan"),
                "hazard_ratio": float("nan"),
                "hazard_ratio_ci_low": float("nan"),
                "hazard_ratio_ci_high": float("nan"),
                "standard_error": float("nan"),
                "p_value": 1.0,
                "bootstrap_log_hr_ci_low": float("nan"),
                "bootstrap_log_hr_ci_high": float("nan"),
                "bootstrap_sign_stability": float("nan"),
                "bootstrap_successes": 0,
                "error": "external phase withheld",
            }
            if len(external):
                try:
                    external_payload = {
                        "executable": True,
                        **prognostic_marker_association(
                            external,
                            duration_ext,
                            event_ext,
                            marker=marker,
                            seed=seed + 43,
                            bootstraps=args.marker_bootstraps,
                        ),
                        "error": "",
                    }
                except Exception as exc:
                    external_payload["error"] = f"{type(exc).__name__}: {exc}"

            rows.append(identity)
            internal_rows.append(internal_payload)
            external_rows.append(external_payload)
            append_checkpoint(
                checkpoint,
                {
                    "identity": identity,
                    "internal": internal_payload,
                    "external": external_payload,
                },
            )
            print(f"[prognosis-marker {progress}/{total}] done {identity}", flush=True)

    protocol = protocol_for("prognosis")
    factors = attach_candidate_ids(
        pd.DataFrame(rows), task="prognosis", identity_fields=protocol.factor_fields
    )
    internal = pd.DataFrame(internal_rows)
    joined = pd.concat(
        [factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1
    )
    internal["q_value"] = family_fdr(joined, "p_value", ("horizon",))
    analytic_direction = (
        internal["log_hazard_ratio_ci_low"].gt(0)
        | internal["log_hazard_ratio_ci_high"].lt(0)
    )
    bootstrap_direction = (
        internal["bootstrap_log_hr_ci_low"].gt(0)
        | internal["bootstrap_log_hr_ci_high"].lt(0)
    )
    internal["validated"] = (
        internal["n"].ge(80)
        & internal["events"].ge(10)
        & internal["q_value"].lt(0.05)
        & analytic_direction
    )
    internal["strict_validated"] = (
        internal["validated"]
        & bootstrap_direction
        & internal["bootstrap_sign_stability"].ge(0.95)
        & internal["bootstrap_successes"].ge(max(1, int(args.marker_bootstraps * 0.8)))
    )
    internal.insert(0, "candidate_id", factors["candidate_id"])

    external_outcome = pd.DataFrame(external_rows)
    external_joined = pd.concat(
        [factors.reset_index(drop=True), external_outcome.reset_index(drop=True)],
        axis=1,
    )
    external_outcome["q_value"] = family_fdr(
        external_joined, "p_value", ("horizon",)
    )
    external_outcome["direction_concordant"] = (
        np.sign(external_outcome["log_hazard_ratio"])
        == np.sign(internal["log_hazard_ratio"])
    )
    external_outcome["validated"] = (
        external_outcome["executable"].astype(bool)
        & external_outcome["n"].ge(20)
        & external_outcome["events"].ge(5)
        & external_outcome["q_value"].lt(0.05)
        & external_outcome["direction_concordant"]
    )
    external_outcome.insert(0, "candidate_id", factors["candidate_id"])
    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=protocol.factor_fields,
        case_study_id="prognosis",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    return _export_and_benchmark(
        args,
        task="prognosis",
        public=public,
        internal=internal,
        external=external_outcome,
        output_dir=output_dir,
        kg_audit=kg_audit,
        provenance={
            "cohort_audit": audit,
            "discovery": "ADNI1/GO/2 baseline MCI",
            "external": "withheld during internal tuning"
            if args.internal_only
            else "ADNI3 baseline MCI",
            "horizons_years": list(args.horizons),
            "prognosis_mode": "marker_association",
            "covariates": list(PROGNOSTIC_BASIC_COVARIATES),
            "marker_bootstraps": int(args.marker_bootstraps),
            "primary_validation": "BH-FDR adjusted Cox marker association with bootstrap direction stability",
            "secondary_validation": "paired cross-validated C-index improvement over covariates alone",
        },
    )


def build_prognosis_risk_models(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    survival: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint = output_dir / "work" / f"prognosis_paired_v4_perm{args.permutations}.jsonl"
    completed = load_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    feature_sets = {
        name: PREDICTION_FEATURE_SETS[name] for name in args.feature_sets
    }
    total = len(args.horizons) * len(feature_sets) * len(args.survival_models)
    progress = 0
    for horizon in args.horizons:
        discovery = survival.loc[survival["phase"].eq("discovery")].reset_index(drop=True)
        external = survival.loc[survival["phase"].eq("external")].reset_index(drop=True)
        duration, event = censored_at_horizon(discovery, horizon)
        duration_ext, event_ext = censored_at_horizon(external, horizon)
        baseline_cache: dict[str, dict[str, Any]] = {}
        for model in args.survival_models:
            X_base = discovery[list(CLINICAL_BASELINE)].to_numpy(dtype=float)
            cv_seed = stable_seed(args.seed, "prognosis-paired-cv", horizon)
            outer_splits = survival_outer_splits(event, seed=cv_seed)
            base_cv = survival_cv(
                X_base,
                duration,
                event,
                model=model,
                seed=cv_seed,
                outer_splits=outer_splits,
            )
            base_final = make_survival_model(model, seed=stable_seed(args.seed, "prognosis-base-final", horizon, model)).fit(X_base, duration, event)
            baseline_cache[model] = {
                "cv": base_cv,
                "final": base_final,
                "cv_seed": cv_seed,
                "outer_splits": outer_splits,
            }
        for feature_family, features in feature_sets.items():
            X = discovery[list(features)].to_numpy(dtype=float)
            X_ext = external[list(features)].to_numpy(dtype=float)
            X_ext_base = external[list(CLINICAL_BASELINE)].to_numpy(dtype=float)
            for model in args.survival_models:
                identity = {
                    "outcome": "mci_to_dementia",
                    "horizon": f"{horizon}y",
                    "feature_family": feature_family,
                    "marker": feature_family,
                    "model": model,
                }
                progress += 1
                key = _identity_key(identity)
                if key in completed:
                    payload = completed[key]
                    rows.append(identity); internal_rows.append(payload["internal"]); external_rows.append(payload["external"])
                    print(f"[prognosis {progress}/{total}] resume {identity}", flush=True)
                    continue
                seed = stable_seed(args.seed, "prognosis", horizon, feature_family, model)
                paired = baseline_cache[model]
                try:
                    cv = survival_cv(
                        X,
                        duration,
                        event,
                        model=model,
                        seed=int(paired["cv_seed"]),
                        outer_splits=paired["outer_splits"],
                    )
                    baseline_cv = paired["cv"]
                    improvement = cv["fold_c_index"] - baseline_cv["fold_c_index"]
                    imp_mean, imp_low, imp_high = _mean_ci(improvement)
                    permutation = survival_permutation_p(cv["oof_duration"], cv["oof_event"], cv["oof_risk"], observed=cv["c_index"], permutations=args.permutations, seed=seed + 17)
                    final = make_survival_model(model, seed=seed + 31).fit(X, duration, event)
                    error = ""
                except Exception as exc:
                    cv = {"c_index": float("nan"), "c_index_ci_low": float("nan"), "c_index_ci_high": float("nan")}
                    imp_mean = imp_low = imp_high = float("nan")
                    permutation = {"permutation_p_exact": 1.0, "permutation_p_tail": 1.0, "permutation_null_mean": float("nan"), "permutation_null_sd": float("nan")}
                    final = None
                    error = f"{type(exc).__name__}: {exc}"
                metric = {
                    "n": 0,
                    "events": 0,
                    "c_index": float("nan"),
                    "c_index_ci_low": float("nan"),
                    "c_index_ci_high": float("nan"),
                    "c_index_improvement": float("nan"),
                    "c_index_improvement_ci_low": float("nan"),
                }
                external_error = "external phase withheld"
                if final is not None and len(duration_ext):
                    try:
                        metric = bootstrap_cindex(
                            duration_ext,
                            event_ext,
                            final.predict_risk(X_ext),
                            baseline_risk=baseline_cache[model]["final"].predict_risk(
                                X_ext_base
                            ),
                            seed=seed + 43,
                        )
                        external_error = ""
                    except Exception as exc:
                        external_error = f"{type(exc).__name__}: {exc}"
                internal_payload = {
                    "c_index": cv["c_index"], "c_index_ci_low": cv["c_index_ci_low"], "c_index_ci_high": cv["c_index_ci_high"],
                    "clinical_c_index_improvement": imp_mean, "clinical_c_index_improvement_ci_low": imp_low, "clinical_c_index_improvement_ci_high": imp_high,
                    **permutation, "split_seed": int(paired["cv_seed"]), "n_folds": int(len(paired["outer_splits"])), "n_discovery": int(len(duration)), "events_discovery": int(event.sum()), "error": error,
                }
                external_payload = {
                    "executable": bool(metric["n"] >= 20 and metric["events"] >= 5),
                    **metric,
                    "error": external_error,
                }
                rows.append(identity); internal_rows.append(internal_payload); external_rows.append(external_payload)
                append_checkpoint(checkpoint, {"identity": identity, "internal": internal_payload, "external": external_payload})
                print(f"[prognosis {progress}/{total}] done {identity}", flush=True)

    protocol = protocol_for("prognosis")
    factors = attach_candidate_ids(pd.DataFrame(rows), task="prognosis", identity_fields=protocol.factor_fields)
    internal = pd.DataFrame(internal_rows)
    joined = pd.concat([factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1)
    internal["q_value"] = family_fdr(joined, "permutation_p_tail", ("horizon", "model"))
    internal["validated"] = internal["c_index_ci_low"].gt(0.5) & internal["clinical_c_index_improvement_ci_low"].gt(0) & internal["q_value"].lt(0.05)
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])
    external = pd.DataFrame(external_rows)
    external["validated"] = external["executable"].astype(bool) & external["c_index_ci_low"].gt(0.5) & external["c_index_improvement"].gt(0)
    external.insert(0, "candidate_id", factors["candidate_id"])
    public, kg_audit = score_public_candidates(
        factors,
        semantic_fields=("outcome", "horizon", "feature_family", "marker", "model"),
        case_study_id="prognosis",
        kg_path=args.kg.resolve(),
        seed=args.seed,
    )
    return _export_and_benchmark(
        args, task="prognosis", public=public, internal=internal, external=external, output_dir=output_dir, kg_audit=kg_audit,
        provenance={"cohort_audit": audit, "discovery": "ADNI1/GO/2 baseline MCI", "external": "withheld during internal tuning" if args.internal_only else "ADNI3 baseline MCI", "horizons_years": list(args.horizons), "clinical_baseline": list(CLINICAL_BASELINE), "paired_cv": True, "prognosis_mode": "risk_model"},
    )


def build_prognosis(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    survival: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if args.prognosis_mode == "risk_model":
        return build_prognosis_risk_models(args, cohort, survival, audit, output_dir)
    return build_prognosis_marker_associations(args, cohort, survival, audit, output_dir)


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    valid = np.isfinite(values) & np.isfinite(covariates).all(axis=1)
    output = np.full(len(values), np.nan, dtype=float)
    design = np.column_stack([np.ones(valid.sum()), covariates[valid]])
    coefficient = np.linalg.lstsq(design, values[valid], rcond=None)[0]
    output[valid] = values[valid] - design @ coefficient
    return output


def association_test(x: np.ndarray, y: np.ndarray, covariates: np.ndarray, model: str) -> dict[str, float | int]:
    x_res = residualize(x, covariates)
    y_res = residualize(y, covariates)
    valid = np.isfinite(x_res) & np.isfinite(y_res)
    x_res = x_res[valid]; y_res = y_res[valid]
    if len(x_res) < 30 or np.std(x_res) == 0 or np.std(y_res) == 0:
        return {"n": int(len(x_res)), "effect": float("nan"), "p_value": 1.0}
    if model == "association_glm":
        result = pearsonr(x_res, y_res)
        effect, p_value = float(result.statistic), float(result.pvalue)
    elif model == "rank_association":
        result = spearmanr(x_res, y_res)
        effect, p_value = float(result.statistic), float(result.pvalue)
    elif model == "robust_huber":
        x_std = (x_res - x_res.mean()) / x_res.std(ddof=1)
        y_std = (y_res - y_res.mean()) / y_res.std(ddof=1)
        result = sm.RLM(y_std, sm.add_constant(x_std), M=sm.robust.norms.HuberT()).fit()
        effect = float(result.params[1])
        p_value = float(2 * norm.sf(abs(effect / result.bse[1]))) if result.bse[1] > 0 else 1.0
    else:
        raise ValueError(model)
    return {"n": int(len(x_res)), "effect": effect, "p_value": p_value}


def imaging_genetics_covariates(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame[["AGE", "sex_binary", "PTEDUCAT", "PC1", "PC2", "PC3", "PC4", "PC5"]].copy()
    site = pd.get_dummies(frame["SITE"].astype(str), prefix="site", drop_first=True, dtype=float)
    design = pd.concat([numeric.reset_index(drop=True), site.reset_index(drop=True)], axis=1)
    for column in design:
        values = pd.to_numeric(design[column], errors="coerce")
        design[column] = values.fillna(values.median())
    return design.to_numpy(dtype=float)


def build_imaging_genetics(
    args: argparse.Namespace,
    cohort: pd.DataFrame,
    survival: pd.DataFrame,
    audit: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    discovery = cohort.loc[cohort["phase"].eq("discovery")].reset_index(drop=True)
    external = cohort.loc[cohort["phase"].eq("external")].reset_index(drop=True)
    curated = sorted(
        column for column in cohort
        if column.startswith(CURATED_PATHWAY_PREFIX) and column.endswith(CURATED_PATHWAY_SUFFIX)
    )
    gene_scores = ["apoe_e4_dosage", "ad_prs_p1em03_avg", *curated]
    if args.quick:
        gene_scores = gene_scores[:2]
        imaging_markers = IMAGING_MARKERS[:2]
    else:
        imaging_markers = IMAGING_MARKERS
    cov_discovery = imaging_genetics_covariates(discovery)
    cov_external = imaging_genetics_covariates(external)
    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    total = len(gene_scores) * len(imaging_markers) * len(args.association_models)
    progress = 0
    for gene in gene_scores:
        for marker in imaging_markers:
            for model in args.association_models:
                identity = {"gene_pathway": gene, "atlas": "ADNI_structural_MRI", "imaging_phenotype": marker, "model": model}
                progress += 1
                internal_metric = association_test(
                    discovery[gene].to_numpy(dtype=float), discovery[marker].to_numpy(dtype=float), cov_discovery, model
                )
                external_metric = association_test(
                    external[gene].to_numpy(dtype=float), external[marker].to_numpy(dtype=float), cov_external, model
                )
                rows.append(identity)
                internal_rows.append({"n_discovery": internal_metric["n"], "effect": internal_metric["effect"], "p_value": internal_metric["p_value"], "error": ""})
                external_rows.append({"executable": bool(external_metric["n"] >= 30), "n_external": external_metric["n"], "effect": external_metric["effect"], "p_value": external_metric["p_value"]})
                print(f"[imaging-genetics {progress}/{total}] done {gene} {marker} {model}", flush=True)
    protocol = protocol_for("imaging_genetics")
    factors = attach_candidate_ids(pd.DataFrame(rows), task="imaging_genetics", identity_fields=protocol.factor_fields)
    internal = pd.DataFrame(internal_rows)
    joined = pd.concat([factors.reset_index(drop=True), internal.reset_index(drop=True)], axis=1)
    internal["q_value"] = family_fdr(joined, "p_value", ("imaging_phenotype", "model"))
    internal["validated"] = internal["q_value"].lt(0.05) & internal["effect"].abs().gt(0.10)
    internal["strict_validated"] = internal["validated"]
    internal.insert(0, "candidate_id", factors["candidate_id"])
    external_outcome = pd.DataFrame(external_rows)
    external_joined = pd.concat([factors.reset_index(drop=True), external_outcome.reset_index(drop=True)], axis=1)
    external_outcome["q_value"] = family_fdr(external_joined, "p_value", ("imaging_phenotype", "model"))
    external_outcome["direction_concordant"] = np.sign(external_outcome["effect"]) == np.sign(internal["effect"])
    external_outcome["validated"] = external_outcome["executable"].astype(bool) & external_outcome["q_value"].lt(0.05) & external_outcome["effect"].abs().gt(0.10) & external_outcome["direction_concordant"]
    external_outcome.insert(0, "candidate_id", factors["candidate_id"])
    public, kg_audit = score_public_candidates(
        factors, semantic_fields=("gene_pathway", "atlas", "imaging_phenotype", "model"), case_study_id="imaging_genetics", kg_path=args.kg.resolve(), seed=args.seed
    )
    return _export_and_benchmark(
        args, task="imaging_genetics", public=public, internal=internal, external=external_outcome, output_dir=output_dir, kg_audit=kg_audit,
        provenance={"cohort_audit": audit, "discovery": "ADNI1/GO/2 genotype and baseline sMRI", "external": "ADNI3 genotype and baseline sMRI", "gene_scores": gene_scores, "covariates": ["age", "sex", "education", "ancestry_PC1-5", "site"], "imaging_normalization": "regional volume divided by ICV"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("progression_prediction", "prognosis", "imaging_genetics", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument(
        "--benchmark-trials",
        type=int,
        default=10,
        help="Outcome-blind generator trials (5 development; 10 final).",
    )
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument(
        "--feature-sets", nargs="+", default=list(PREDICTION_FEATURE_SETS)
    )
    parser.add_argument(
        "--progression-models", nargs="+", default=list(PROGRESSION_MODELS)
    )
    parser.add_argument(
        "--survival-models", nargs="+", default=list(SURVIVAL_MODELS)
    )
    parser.add_argument(
        "--prognosis-mode",
        choices=("marker_association", "risk_model"),
        default="marker_association",
        help=(
            "Use direct covariate-adjusted marker associations by default; "
            "risk_model preserves the earlier multivariate C-index experiment."
        ),
    )
    parser.add_argument(
        "--marker-bootstraps",
        type=int,
        default=499,
        help="Subject bootstrap replicates for marker-level prognosis stability.",
    )
    parser.add_argument(
        "--association-models", nargs="+", default=list(ASSOCIATION_MODELS)
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Exclude ADNI3 rows and emit placeholder external outcomes during tuning.",
    )
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.cv_repeats = 1
        args.permutations = min(args.permutations, 9)
        args.horizons = args.horizons[:1]
        args.feature_sets = args.feature_sets[:1]
        args.progression_models = args.progression_models[:1]
        args.survival_models = args.survival_models[:1]
        args.association_models = args.association_models[:1]
        args.marker_bootstraps = min(args.marker_bootstraps, 19)
    if args.internal_only and args.task in {"imaging_genetics", "all"}:
        raise ValueError("--internal-only currently supports progression or prognosis")
    cohort, survival, audit = load_adni_cohort(
        args.data_root,
        phase="discovery" if args.internal_only else "all",
    )
    tasks = ("progression_prediction", "prognosis", "imaging_genetics") if args.task == "all" else (args.task,)
    manifests: dict[str, Any] = {}
    for task in tasks:
        output_dir = args.output_root / task
        output_dir.mkdir(parents=True, exist_ok=True)
        if task == "progression_prediction":
            manifests[task] = build_progression(args, cohort, survival, audit, output_dir)
        elif task == "prognosis":
            manifests[task] = build_prognosis(args, cohort, survival, audit, output_dir)
        else:
            manifests[task] = build_imaging_genetics(args, cohort, survival, audit, output_dir)
    print(json.dumps(manifests, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
