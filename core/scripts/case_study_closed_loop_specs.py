"""Frozen scientific contracts for the supplemental Case Study experiments.

The seven supplemental Case Studies use independent discovery cohorts and
task-appropriate validation endpoints. External validation is required for six
tasks. Disease subtyping is the sole explicit exception because the local
HCP-EP and COBRE releases do not contain harmonizable continuous symptom
dimensions. Predictive tasks use paired five-fold splits within every seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class TaskProtocol:
    task: str
    candidate_unit: str
    factor_fields: tuple[str, ...]
    discovery_datasets: tuple[str, ...]
    external_datasets: tuple[str, ...]
    internal_validation: str
    external_validation: str
    primary_metric: str
    models: tuple[str, ...]
    budgets: tuple[int, ...]
    recall_targets: tuple[float, ...]
    minimum_trials: int = 10
    external_required: bool = False


@dataclass(frozen=True)
class EvaluationProfile:
    phase: str
    seeds: tuple[int, ...]
    outer_folds: int
    benchmark_trials: int
    same_splits_across_methods: bool = True


COMMON_RECALL_TARGETS = (0.01, 0.05, 0.10, 0.20, 0.50)

SUPPLEMENTAL_CASE_STUDIES = (
    "differential_diagnosis",
    "disease_subtyping",
    "connectome_behavior",
    "brain_age",
    "progression_prediction",
    "prognosis",
    "imaging_genetics",
)
EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES = tuple(
    task for task in SUPPLEMENTAL_CASE_STUDIES if task != "disease_subtyping"
)
EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES = ("disease_subtyping",)
GENERIC_EXTERNAL_RESULTS_ROLE = "required_except_registered_task_exemptions"

DEVELOPMENT_SEEDS = tuple(range(20260826, 20260831))
FINAL_SEEDS = tuple(range(20260826, 20260836))
EVALUATION_PROFILES: Mapping[str, EvaluationProfile] = {
    "development": EvaluationProfile(
        phase="development",
        seeds=DEVELOPMENT_SEEDS,
        outer_folds=5,
        benchmark_trials=5,
    ),
    "final": EvaluationProfile(
        phase="final",
        seeds=FINAL_SEEDS,
        outer_folds=5,
        benchmark_trials=10,
    ),
}

COMPLETED_EXPERIMENT_LINES = (
    "case1_transdiagnostic",
    "case2_pathway_mediation",
    "biomarker_discovery",
    "differential_diagnosis",
    "disease_subtyping",
    "connectome_behavior",
    "brain_age",
    "progression_prediction",
    "prognosis",
    "imaging_genetics",
)

EXPERIMENT_LINE_IMPLEMENTATIONS: Mapping[str, str] = {
    "case1_transdiagnostic": "case1_transdiagnostic",
    "case2_pathway_mediation": "case2_pathway_mediation",
    "biomarker_discovery": "biomarker_discovery",
    "differential_diagnosis": "differential_diagnosis",
    "disease_subtyping": "disease_subtyping",
    "connectome_behavior": "connectome_behavior",
    "brain_age": "brain_age",
    "progression_prediction": "progression_prediction",
    "prognosis": "prognosis",
    "imaging_genetics": "imaging_genetics",
}


TASK_PROTOCOLS: Mapping[str, TaskProtocol] = {
    "biomarker_discovery": TaskProtocol(
        task="biomarker_discovery",
        candidate_unit="disease x atlas ROI x imaging feature",
        factor_fields=(
            "disease",
            "atlas",
            "roi_index",
            "anatomy",
            "feature_family",
            "feature",
        ),
        discovery_datasets=("TCP",),
        external_datasets=(),
        internal_validation=(
            "Age-, sex-, and site-adjusted case-control association in TCP, with BH-FDR "
            "q<0.05 within each preregistered disease x atlas x feature-family analysis, "
            "|Cohen d|>=0.15, mean held-out directional AUROC>=0.55, and direction "
            "concordance in at least 70% of repeated site-stratified folds. Global-FDR "
            "support is retained as a strict secondary label."
        ),
        external_validation=(
            "Not required for this experiment line; external validation is reserved for "
            "dedicated Case Study 1."
        ),
        primary_metric="validated biomarker discoveries recovered",
        models=(
            "mass_univariate_glm",
            "elastic_net",
            "roi_mlp",
            "brainnetcnn",
            "braingnn",
            "bnt",
            "ibgnn",
            "lggnn",
            "combraintf",
        ),
        budgets=(100, 500, 1000, 5000, 10000, 50000, 100000, 200000),
        recall_targets=COMMON_RECALL_TARGETS,
    ),
    "differential_diagnosis": TaskProtocol(
        task="differential_diagnosis",
        candidate_unit="diagnostic contrast x atlas x whole-brain imaging representation x classifier",
        factor_fields=(
            "diagnostic_contrast",
            "atlas",
            "feature_family",
            "model",
        ),
        discovery_datasets=("UCLA",),
        external_datasets=("HCP-EP", "COBRE"),
        internal_validation=(
            "Nested, site-aware cross-validation with label-permutation FDR; a candidate is "
            "validated when AUROC exceeds the preregistered null and its confidence interval excludes 0.5."
        ),
        external_validation=(
            "Frozen feature definition and model family transferred to the matched psychosis "
            "contrast; no external relabeling, tuning, or ranking updates."
        ),
        primary_metric="validated differential-diagnosis candidates recovered",
        models=("logistic", "elastic_net", "svm", "bnt", "braingnn", "brainnetcnn"),
        budgets=(25, 50, 100, 250, 500, 1000, 2500),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=True,
    ),
    "disease_subtyping": TaskProtocol(
        task="disease_subtyping",
        candidate_unit="disease spectrum x atlas x whole-brain imaging representation x clustering method x cluster count",
        factor_fields=("disease", "atlas", "feature_family", "model", "cluster_count"),
        discovery_datasets=("UCLA",),
        external_datasets=(),
        internal_validation=(
            "Imaging-only clusters must be stable across resamples and separate five continuous "
            "symptom dimensions that were withheld from clustering; global permutation-FDR and "
            "pre-registered effect-size thresholds must pass."
        ),
        external_validation=(
            "Not required by the frozen supplemental protocol. Internal imaging-cluster "
            "stability and held-out continuous symptom separation are the completion gates."
        ),
        primary_metric="stable and clinically distinct subtype solutions recovered",
        models=("kmeans", "gmm", "spectral", "nmf", "consensus", "autoencoder"),
        budgets=(25, 50, 100, 250, 500, 1000),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=False,
    ),
    "connectome_behavior": TaskProtocol(
        task="connectome_behavior",
        candidate_unit="behavior phenotype x atlas/connectome feature x predictive model",
        factor_fields=("phenotype", "atlas", "feature_family", "model"),
        discovery_datasets=("HCP-YA",),
        external_datasets=("ADHD200",),
        internal_validation=(
            "Subject-level nested cross-validation (family-blocked when restricted family IDs are "
            "available); positive out-of-sample correlation and permutation-FDR q<0.05 are required."
        ),
        external_validation=(
            "Frozen preprocessing, atlas, feature transform, and model are evaluated against "
            "full-scale IQ in typically-developing ADHD200 controls without refitting or reranking."
        ),
        primary_metric="replicable connectome-behavior associations recovered",
        models=("ridge", "elastic_net", "svm"),
        budgets=(25, 50, 100, 250, 500, 1000, 2500),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=True,
    ),
    "brain_age": TaskProtocol(
        task="brain_age",
        candidate_unit="modality x atlas/feature representation x age-prediction model",
        factor_fields=("modality", "atlas", "feature_family", "model"),
        discovery_datasets=("HCP-YA",),
        external_datasets=("UCLA-controls", "COBRE-controls", "HCP-EP-controls", "ADHD200-controls"),
        internal_validation=(
            "HCP-YA-only nested age-stratified cross-validation with subject leakage blocked; "
            "the correlation lower bound must reach 0.40, normalized MAE reduction must reach 10%, "
            "and permutation-FDR q must be below 0.05."
        ),
        external_validation=(
            "Frozen model transferred to healthy controls from independent sites within the "
            "HCP-YA age-support interval; report MAE and age correlation without external tuning."
        ),
        primary_metric="generalizable brain-age configurations recovered",
        models=(
            "ridge",
            "elastic_net",
            "svm",
            "bnt",
            "braingnn",
            "brainnetcnn",
            "lggnn",
            "ibgnn",
            "combraintf",
        ),
        budgets=(10, 25, 50, 100, 250, 500, 1000),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=True,
    ),
    "progression_prediction": TaskProtocol(
        task="progression_prediction",
        candidate_unit="baseline marker set x prediction horizon x progression model",
        factor_fields=("outcome", "horizon", "feature_family", "model"),
        discovery_datasets=("ADNI-discovery",),
        external_datasets=("ADNI-heldout-phase",),
        internal_validation=(
            "Landmark-time nested cross-validation with visits after the prediction origin "
            "excluded; AUROC/AUPRC must beat the clinical-only baseline under permutation FDR."
        ),
        external_validation=(
            "Frozen marker set and model applied to a non-overlapping acquisition phase or an "
            "independent dementia cohort with the same horizon and conversion definition."
        ),
        primary_metric="validated progression predictors recovered",
        models=("logistic", "elastic_net", "svm", "temporal_mlp"),
        budgets=(25, 50, 100, 250, 500, 1000),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=True,
    ),
    "prognosis": TaskProtocol(
        task="prognosis",
        candidate_unit="baseline marker x clinical endpoint x survival model",
        factor_fields=("outcome", "horizon", "feature_family", "marker", "model"),
        discovery_datasets=("ADNI-discovery",),
        external_datasets=("ADNI-heldout-phase",),
        internal_validation=(
            "Primary validation is a covariate-adjusted Cox marker association with BH-FDR "
            "control and subject-bootstrap direction stability. Paired cross-validated C-index "
            "improvement over the same covariates is reported as a stricter secondary endpoint."
        ),
        external_validation=(
            "Frozen risk score evaluated in a non-overlapping phase or independent cohort, "
            "including calibration and direction-concordant hazard association."
        ),
        primary_metric="externally reproducible prognostic marker associations recovered",
        models=(
            "cox_adjusted",
            "cox",
            "elastic_net_cox",
            "deepsurv",
            "random_survival_forest",
            "xgboost_survival",
        ),
        budgets=(10, 25, 50, 100),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=True,
    ),
    "imaging_genetics": TaskProtocol(
        task="imaging_genetics",
        candidate_unit="gene/pathway score x imaging phenotype x association model",
        factor_fields=("gene_pathway", "atlas", "imaging_phenotype", "model"),
        discovery_datasets=("ADNI-discovery",),
        external_datasets=("ADNI-heldout-phase",),
        internal_validation=(
            "Genetic association adjusted for age, sex, ancestry PCs, site, and intracranial "
            "volume where applicable; BH-FDR q<0.05 within the registered analysis family."
        ),
        external_validation=(
            "Identical score construction and imaging phenotype tested in a non-overlapping "
            "genotyping phase or independent cohort, requiring FDR support and direction concordance."
        ),
        primary_metric="replicable gene-to-imaging associations recovered",
        models=("association_glm", "rank_association", "robust_huber"),
        budgets=(25, 50, 100, 250, 500, 1000, 2500),
        recall_targets=COMMON_RECALL_TARGETS,
        external_required=True,
    ),
}


def protocol_for(task: str) -> TaskProtocol:
    try:
        return TASK_PROTOCOLS[task]
    except KeyError as exc:
        raise KeyError(f"unknown executable task case study: {task}") from exc


def evaluation_profile_for(phase: str) -> EvaluationProfile:
    try:
        return EVALUATION_PROFILES[phase]
    except KeyError as exc:
        raise KeyError(f"unknown supplemental evaluation phase: {phase}") from exc


def export_protocols(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "case-study-closed-loop-protocols.v4",
                "external_validation_policy": {
                    "required_case_studies": list(
                        EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES
                    ),
                    "exempt_case_studies": list(
                        EXTERNAL_VALIDATION_EXEMPT_CASE_STUDIES
                    ),
                    "generic_results_role": GENERIC_EXTERNAL_RESULTS_ROLE,
                },
                "supplemental_case_studies": list(SUPPLEMENTAL_CASE_STUDIES),
                "evaluation_profiles": {
                    name: asdict(profile)
                    for name, profile in EVALUATION_PROFILES.items()
                },
                "completed_experiment_lines": list(COMPLETED_EXPERIMENT_LINES),
                "experiment_line_implementations": dict(
                    EXPERIMENT_LINE_IMPLEMENTATIONS
                ),
                "tasks": {name: asdict(protocol) for name, protocol in TASK_PROTOCOLS.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
