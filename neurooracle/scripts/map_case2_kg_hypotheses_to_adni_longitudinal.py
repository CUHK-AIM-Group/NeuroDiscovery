"""Rank completed longitudinal ADNI Case 2 tests from KG hypotheses.

The ranking stage is deliberately result blind: only hypothesis content and
the executable pathway, imaging, and outcome labels are available until the
candidate order has been frozen. Experimental coefficients and P values are
merged afterwards for Top-K evaluation and bootstrap stability analysis.

These are association-based mediation screens and do not establish causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

from core.scripts.canonical_kg_release import (
    CURRENT_CANONICAL_SHA256,
    write_release_manifest,
)
from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import (
    _hypothesis_score,
    _normalise_text,
    _path_names,
    _split_genes,
    load_hypotheses,
)
from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    _complete_case_mask,
    build_longitudinal_covariate_matrix,
)
from neurooracle.scripts.run_case2_adni_mediation_smoke import mediation_screen


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
DEFAULT_HYPOTHESIS_PARENT = (
    REPO_ROOT / "neurooracle" / "data" / "experiments" / "case2"
)
DEFAULT_SMOKE_PARENT = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_longitudinal_multimodal_smoke_v1"
)
DEFAULT_PATHWAY_ROOT = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
    / "features"
    / "ad_pathway_prs_v1"
)
DEFAULT_OUTPUT_PARENT = (
    CASE2_ROOT / "experiments" / "case2_adni_kg_guided_longitudinal_v1"
)
DEFAULT_K_VALUES = (5, 10, 20, 50, 100, 210)


PATHWAY_RULES: dict[str, tuple[tuple[tuple[str, ...], float, str], ...]] = {
    "curated__ad_risk_gwas": (
        (("apoe", "alzheimer disease", "alzheimer's disease"), 0.95, "ad_specific"),
        (("amyloid", "a beta", "abeta"), 0.86, "amyloid_mechanism"),
        (("mapt", "tau"), 0.74, "tau_mechanism"),
        (("dementia",), 0.65, "dementia_context"),
    ),
    "curated__mendelian_ad": (
        (("app", "psen1", "psen2", "presenilin"), 0.98, "mendelian_gene"),
        (("amyloid",), 0.78, "amyloid_mechanism"),
        (("alzheimer disease", "alzheimer's disease"), 0.74, "ad_context"),
    ),
    "curated__synaptic": (
        (("synaptic", "synapse"), 0.96, "synaptic_mechanism"),
        (
            (
                "functional connectivity",
                "rs-fmri",
                "alff",
                "regional homogeneity",
                "reho",
            ),
            0.72,
            "functional_neural_measure",
        ),
        (
            (
                "neurotransmitter",
                "slc6a4",
                "oprm1",
                "htr1",
                "htr2",
                "receptor",
            ),
            0.68,
            "neurotransmission_mechanism",
        ),
        (("general cognition", "cognitive"), 0.45, "cognition_context"),
    ),
    "curated__cholinergic": (
        (
            ("cholinergic", "acetylcholine", "chrm", "chrna", "chrnb", "ache", "bche"),
            0.96,
            "cholinergic_mechanism",
        ),
    ),
    "curated__gaba_glutamate": (
        (
            ("gaba", "gabra", "gabbr", "glutamate", "nmda", "grin", "gria"),
            0.96,
            "gaba_glutamate_mechanism",
        ),
    ),
    "curated__microglia_immune": (
        (
            ("microglia", "immune", "inflamm", "trem2", "p2ry12", "cx3cr1"),
            0.94,
            "immune_mechanism",
        ),
    ),
    "curated__myelin": (
        (
            ("myelin", "white matter", "fractional anisotropy", "oligodendro"),
            0.95,
            "myelin_white_matter_mechanism",
        ),
    ),
}


def _latest_file(parent: Path, pattern: str, required_name: str) -> Path:
    candidates = [
        child / required_name
        for child in parent.glob(pattern)
        if (child / required_name).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No {required_name} found below {parent} with pattern {pattern}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _hypothesis_text(hypothesis: dict[str, Any]) -> str:
    return _normalise_text(" | ".join(_path_names(hypothesis)))


def _pathway_alignment(
    hypothesis: dict[str, Any],
    pathway: pd.Series,
) -> tuple[float, str]:
    text = _hypothesis_text(hypothesis)
    source_gene = str(hypothesis.get("source_name", "")).strip().upper()
    path_nodes = {str(name).strip().upper() for name in _path_names(hypothesis)}
    pathway_genes = _split_genes(pathway.get("genes", ""))
    if source_gene and source_gene in pathway_genes:
        return 1.0, "source_gene_in_pathway"
    if path_nodes & pathway_genes:
        return 0.90, "path_gene_in_pathway"

    best = (0.08, "weak_pathway_prior")
    for terms, score, label in PATHWAY_RULES.get(
        str(pathway.get("pathway_id", "")), ()
    ):
        if any(term in text for term in terms) and score > best[0]:
            best = (score, label)
    return best


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _imaging_alignment(
    hypothesis: dict[str, Any],
    modality: str,
    marker: str,
) -> tuple[float, str]:
    text = _hypothesis_text(hypothesis)
    key = (str(modality), str(marker))
    has_tau = _has_any(text, ("tau", "mapt"))
    has_structural = _has_any(
        text,
        (
            "cortical thickness",
            "cortical_thickness",
            "regional volume",
            "regional_volume",
            "gray matter density",
            "gray_matter_density",
            "atrophy",
            "brain volume",
        ),
    )

    if key == ("amyloid_pet", "CENTILOIDS"):
        if _has_any(text, ("amyloid", "centiloid", "abeta", "a beta")):
            return 0.98, "amyloid_pet_exact"
    elif key == ("fdg_pet", "FDG_META_ROI_SUVR"):
        if _has_any(text, ("fdg", "glucose metabolism", "hypometabolism")):
            return 0.98, "fdg_pet_exact"
    elif key == ("tau_pet", "CTX_ENTORHINAL_SUVR"):
        if has_tau and "entorhinal" in text:
            return 1.0, "tau_entorhinal_exact"
        if has_tau:
            return 0.68, "generic_tau_proxy"
    elif key == ("tau_pet", "META_TEMPORAL_SUVR"):
        if has_tau and _has_any(
            text,
            ("temporal", "hippocamp", "parahippocamp", "entorhinal"),
        ):
            return 0.92, "tau_medial_temporal_proxy"
        if has_tau:
            return 0.74, "generic_tau_proxy"
    elif key == ("smri_adnimerge", "Hippocampus"):
        if has_structural and "hippocamp" in text:
            return 0.98, "structural_hippocampus_exact"
    elif key == ("smri_adnimerge", "Entorhinal"):
        if has_structural and "entorhinal" in text:
            return 0.98, "structural_entorhinal_exact"
    elif key == ("smri_adnimerge", "Fusiform"):
        if has_structural and "fusiform" in text:
            return 0.95, "structural_fusiform_exact"
    elif key == ("smri_adnimerge", "MidTemp"):
        if has_structural and _has_any(
            text, ("middle temporal", "midtemp", "temporal lobe")
        ):
            return 0.92, "structural_midtemporal_exact"
    elif key == ("smri_adnimerge", "WholeBrain"):
        if has_structural and _has_any(text, ("whole brain", "brain volume")):
            return 0.90, "structural_wholebrain_exact"
        if has_structural and "cortical thickness" in text:
            return 0.62, "global_structural_proxy"
    elif key == ("smri_adnimerge", "Ventricles"):
        if has_structural and _has_any(text, ("ventricle", "ventricular")):
            return 0.98, "structural_ventricle_exact"
    return 0.03, "weak_imaging_prior"


def _outcome_alignment(
    hypothesis: dict[str, Any], outcome: str
) -> tuple[float, str]:
    target = _normalise_text(hypothesis.get("target_name", ""))
    outcome = str(outcome)
    if outcome == "CDRSB" and _has_any(
        target, ("clinical dementia rating", "cdr-sb", "cdrsb")
    ):
        return 1.0, "cdrsb_exact"
    if outcome == "MMSE" and _has_any(
        target, ("mmse", "mini-mental state")
    ):
        return 1.0, "mmse_exact"
    if outcome == "ADAS13" and _has_any(target, ("adas13", "adas-13")):
        return 1.0, "adas13_exact"

    if outcome == "mPACCdigit":
        if _has_any(
            target,
            (
                "mpacc",
                "preclinical alzheimer cognitive composite",
                "digit symbol",
                "digit substitution",
            ),
        ):
            return 1.0, "mpacc_digit_exact"
        if _has_any(
            target,
            (
                "general cognition",
                "cognitive performance",
                "cognitive ability",
                "cognitive decline",
            ),
        ):
            return 0.90, "broad_cognition_mpacc_digit"
        if _has_any(
            target,
            ("memory decline", "episodic memory", "executive function"),
        ):
            return 0.82, "cognitive_domain_mpacc_digit"
        if _has_any(target, ("alzheimer", "dementia", "mci")):
            return 0.65, "dementia_context_mpacc_digit"
        return 0.04, "weak_outcome_prior"

    if outcome == "FAQ":
        if _has_any(
            target,
            (
                "functional activities questionnaire",
                "instrumental activities of daily living",
                "instrumental activity of daily living",
                "daily functioning",
            ),
        ):
            return 1.0, "faq_function_exact"
        if _has_any(
            target,
            ("functional decline", "clinical progression", "disease progression"),
        ):
            return 0.88, "functional_decline_faq"
        if _has_any(target, ("alzheimer", "dementia", "mci")):
            return 0.75, "dementia_severity_faq"
        if _has_any(
            target,
            ("general cognition", "cognitive decline", "cognitive performance"),
        ):
            return 0.55, "broad_cognition_faq"
        return 0.04, "weak_outcome_prior"

    if outcome == "LDELTOTAL":
        if _has_any(
            target,
            (
                "ldeltotal",
                "logical memory delayed recall",
                "delayed logical memory",
                "delayed recall",
            ),
        ):
            return 1.0, "logical_memory_delayed_exact"
        if _has_any(
            target,
            ("episodic memory", "memory decline", "memory performance"),
        ):
            return 0.95, "episodic_memory_ldeltotal"
        if _has_any(
            target,
            ("general cognition", "cognitive decline", "cognitive performance"),
        ):
            return 0.78, "broad_cognition_ldeltotal"
        if _has_any(target, ("alzheimer", "dementia", "mci")):
            return 0.62, "dementia_context_ldeltotal"
        return 0.04, "weak_outcome_prior"

    if _has_any(
        target,
        (
            "general cognition",
            "cognitive performance",
            "cognitive ability",
            "cognitive decline",
            "memory decline",
        ),
    ):
        mapping = {
            "MMSE": (0.86, "broad_cognition_mmse"),
            "ADAS13": (0.72, "broad_cognition_adas13"),
            "CDRSB": (0.55, "broad_cognition_cdrsb"),
        }
        return mapping[outcome]
    if _has_any(target, ("alzheimer", "dementia", "mci")):
        mapping = {
            "CDRSB": (0.86, "dementia_severity_cdrsb"),
            "ADAS13": (0.72, "dementia_severity_adas13"),
            "MMSE": (0.64, "dementia_severity_mmse"),
        }
        return mapping[outcome]
    return 0.04, "weak_outcome_prior"


def _support_score(
    kg_score: float,
    pathway_alignment: float,
    imaging_alignment: float,
    outcome_alignment: float,
) -> tuple[float, float]:
    alignment = float(
        np.cbrt(pathway_alignment * imaging_alignment * outcome_alignment)
    )
    return float(kg_score * alignment), alignment


def build_ranked_candidates(
    hypotheses: list[dict[str, Any]],
    exposures: pd.DataFrame,
    pathway_catalog: pd.DataFrame,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a complete 210-test ranking without experimental statistics."""

    pathway_columns = [
        "pathway_id",
        "pathway_source",
        "pathway_name",
        "reference_id",
        "gene_count",
        "genes",
    ]
    pathway_metadata = pathway_catalog.loc[:, pathway_columns].drop_duplicates(
        "pathway_id"
    )
    exposure_catalog = exposures.merge(
        pathway_metadata,
        on=["pathway_id", "pathway_source", "pathway_name", "gene_count"],
        how="left",
        validate="one_to_one",
    )
    if exposure_catalog["genes"].isna().any():
        raise ValueError("Selected pathway exposures are missing pathway genes")
    catalog_by_exposure = exposure_catalog.set_index("score_name", drop=False)

    candidate_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    hypothesis_best: dict[str, float] = {
        str(hypothesis.get("id", "")): 0.0 for hypothesis in hypotheses
    }

    for candidate in universe.itertuples(index=False):
        pathway = catalog_by_exposure.loc[str(candidate.exposure)]
        supports: list[dict[str, object]] = []
        for hypothesis in hypotheses:
            hypothesis_id = str(hypothesis.get("id", ""))
            pathway_score, pathway_mapping = _pathway_alignment(
                hypothesis, pathway
            )
            imaging_score, imaging_mapping = _imaging_alignment(
                hypothesis,
                str(candidate.modality),
                str(candidate.marker),
            )
            outcome_score, outcome_mapping = _outcome_alignment(
                hypothesis, str(candidate.outcome)
            )
            kg_score = _hypothesis_score(hypothesis)
            score, alignment = _support_score(
                kg_score,
                pathway_score,
                imaging_score,
                outcome_score,
            )
            meaningful = (
                pathway_score >= 0.30
                and imaging_score >= 0.30
                and outcome_score >= 0.30
            )
            record = {
                "hypothesis_id": hypothesis_id,
                "source_gene": str(hypothesis.get("source_name", "")),
                "target_name": str(hypothesis.get("target_name", "")),
                "path_text": " -> ".join(_path_names(hypothesis)),
                "kg_composite_score": kg_score,
                "pathway_alignment": pathway_score,
                "pathway_mapping": pathway_mapping,
                "imaging_alignment": imaging_score,
                "imaging_mapping": imaging_mapping,
                "outcome_alignment": outcome_score,
                "outcome_mapping": outcome_mapping,
                "chain_alignment": alignment,
                "support_score": score,
                "meaningful_support": meaningful,
            }
            supports.append(record)
            hypothesis_best[hypothesis_id] = max(
                hypothesis_best[hypothesis_id], score
            )

        supports.sort(
            key=lambda row: (
                -float(row["support_score"]),
                str(row["hypothesis_id"]),
            )
        )
        top = supports[0]
        top_scores = [float(row["support_score"]) for row in supports[:3]]
        mapping_score = min(
            1.0,
            top_scores[0]
            + (0.10 * top_scores[1] if len(top_scores) > 1 else 0.0)
            + (0.05 * top_scores[2] if len(top_scores) > 2 else 0.0),
        )
        meaningful = [row for row in supports if row["meaningful_support"]]
        candidate_rows.append(
            {
                "exposure": str(candidate.exposure),
                "pathway_id": str(candidate.pathway_id),
                "pathway_name": str(candidate.pathway_name),
                "threshold_label": str(candidate.threshold_label),
                "gene_count": int(candidate.gene_count),
                "modality": str(candidate.modality),
                "marker": str(candidate.marker),
                "outcome": str(candidate.outcome),
                "mapping_score": mapping_score,
                "primary_hypothesis_id": str(top["hypothesis_id"]),
                "primary_path_text": str(top["path_text"]),
                "primary_kg_score": float(top["kg_composite_score"]),
                "primary_chain_alignment": float(top["chain_alignment"]),
                "primary_pathway_alignment": float(top["pathway_alignment"]),
                "primary_pathway_mapping": str(top["pathway_mapping"]),
                "primary_imaging_alignment": float(top["imaging_alignment"]),
                "primary_imaging_mapping": str(top["imaging_mapping"]),
                "primary_outcome_alignment": float(top["outcome_alignment"]),
                "primary_outcome_mapping": str(top["outcome_mapping"]),
                "meaningful_support_count": len(meaningful),
                "supporting_hypothesis_ids": ";".join(
                    str(row["hypothesis_id"]) for row in meaningful
                ),
            }
        )
        for support_rank, row in enumerate(supports[:5], start=1):
            support_rows.append(
                {
                    "exposure": str(candidate.exposure),
                    "modality": str(candidate.modality),
                    "marker": str(candidate.marker),
                    "outcome": str(candidate.outcome),
                    "support_rank": support_rank,
                    **row,
                }
            )

    ranked = pd.DataFrame(candidate_rows).sort_values(
        [
            "mapping_score",
            "primary_chain_alignment",
            "primary_kg_score",
            "meaningful_support_count",
            "pathway_id",
            "modality",
            "marker",
            "outcome",
        ],
        ascending=[False, False, False, False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "kg_rank", np.arange(1, len(ranked) + 1))

    audit_rows = []
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("id", ""))
        audit_rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "source_gene": str(hypothesis.get("source_name", "")),
                "target_name": str(hypothesis.get("target_name", "")),
                "path_text": " -> ".join(_path_names(hypothesis)),
                "kg_composite_score": _hypothesis_score(hypothesis),
                "best_executable_support_score": hypothesis_best[hypothesis_id],
                "meaningfully_mapped": hypothesis_best[hypothesis_id] >= 0.20,
            }
        )
    return ranked, pd.DataFrame(support_rows), pd.DataFrame(audit_rows)


def _fdr(values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        adjusted[finite] = multipletests(values[finite], method="fdr_bh")[1]
    return adjusted


def _matrix_bh_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Return row-wise BH rejection masks for a 2D P-value matrix."""

    order = np.argsort(p_values, axis=1)
    sorted_p = np.take_along_axis(p_values, order, axis=1)
    thresholds = alpha * np.arange(1, p_values.shape[1] + 1) / p_values.shape[1]
    passes = sorted_p <= thresholds[None, :]
    rejection_counts = np.where(
        passes,
        np.arange(1, p_values.shape[1] + 1)[None, :],
        0,
    ).max(axis=1)
    cutoffs = np.full(len(p_values), -np.inf, dtype=float)
    has_rejections = rejection_counts > 0
    rows = np.flatnonzero(has_rejections)
    cutoffs[rows] = sorted_p[rows, rejection_counts[rows] - 1]
    return p_values <= cutoffs[:, None]


def evaluate_prioritisation(
    ranked_results: pd.DataFrame,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 20_260_731,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate Top-K chain discoveries against random candidate orderings."""

    ranked = ranked_results.sort_values("kg_rank", kind="stable").copy()
    p_values = pd.to_numeric(ranked["sobel_p"], errors="coerce").to_numpy(float)
    path_ok = (
        pd.to_numeric(ranked["a_path_p"], errors="coerce").lt(alpha)
        & pd.to_numeric(ranked["b_path_p"], errors="coerce").lt(alpha)
    ).to_numpy(bool)
    nominal = (p_values < alpha) & path_ok
    rng = np.random.default_rng(seed)

    metric_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    adjusted_rows: list[pd.DataFrame] = []
    population_size = len(ranked)
    for k in sorted(set(int(value) for value in k_values)):
        if k <= 0 or k > population_size:
            continue
        top = ranked.head(k).copy()
        top_q = _fdr(
            pd.to_numeric(top["sobel_p"], errors="coerce").to_numpy(float)
        )
        top["evaluation_k"] = k
        top["sobel_q_within_topk"] = top_q
        top["topk_fdr_chain_hit"] = (
            (top_q < alpha)
            & pd.to_numeric(top["a_path_p"], errors="coerce").lt(alpha)
            & pd.to_numeric(top["b_path_p"], errors="coerce").lt(alpha)
        )
        top["nominal_chain_hit"] = (
            pd.to_numeric(top["sobel_p"], errors="coerce").lt(alpha)
            & pd.to_numeric(top["a_path_p"], errors="coerce").lt(alpha)
            & pd.to_numeric(top["b_path_p"], errors="coerce").lt(alpha)
        )
        adjusted_rows.append(top)
        observed_fdr = int(top["topk_fdr_chain_hit"].sum())
        observed_nominal = int(top["nominal_chain_hit"].sum())

        if k == population_size:
            sampled = np.tile(np.arange(population_size), (n_resamples, 1))
        else:
            random_keys = rng.random((n_resamples, population_size))
            sampled = np.argpartition(random_keys, kth=k - 1, axis=1)[:, :k]
        sampled_p = p_values[sampled]
        random_rejections = _matrix_bh_rejections(sampled_p, alpha)
        sampled_path_ok = path_ok[sampled]
        random_fdr = (random_rejections & sampled_path_ok).sum(axis=1)
        random_nominal = nominal[sampled].sum(axis=1)

        fdr_mean = float(random_fdr.mean())
        nominal_mean = float(random_nominal.mean())
        fdr_ci = np.quantile(random_fdr, [0.025, 0.975])
        nominal_ci = np.quantile(random_nominal, [0.025, 0.975])
        metric_rows.append(
            {
                "k": k,
                "neurodiscovery_topk_fdr_hits": observed_fdr,
                "neurodiscovery_nominal_chain_hits": observed_nominal,
                "random_topk_fdr_mean": fdr_mean,
                "random_topk_fdr_ci95_low": float(fdr_ci[0]),
                "random_topk_fdr_ci95_high": float(fdr_ci[1]),
                "fold_topk_fdr_over_random": (
                    observed_fdr / fdr_mean
                    if fdr_mean > 0
                    else math.inf if observed_fdr > 0 else 1.0
                ),
                "empirical_p_topk_fdr": float(
                    (1 + np.count_nonzero(random_fdr >= observed_fdr))
                    / (n_resamples + 1)
                ),
                "random_nominal_mean": nominal_mean,
                "random_nominal_ci95_low": float(nominal_ci[0]),
                "random_nominal_ci95_high": float(nominal_ci[1]),
                "fold_nominal_over_random": (
                    observed_nominal / nominal_mean
                    if nominal_mean > 0
                    else math.inf if observed_nominal > 0 else 1.0
                ),
                "empirical_p_nominal": float(
                    (1 + np.count_nonzero(random_nominal >= observed_nominal))
                    / (n_resamples + 1)
                ),
                "resamples": n_resamples,
            }
        )
        distribution_rows.extend(
            {
                "k": k,
                "resample": index,
                "random_topk_fdr_hits": int(fdr_hits),
                "random_nominal_chain_hits": int(nominal_hits),
            }
            for index, (fdr_hits, nominal_hits) in enumerate(
                zip(random_fdr, random_nominal), start=1
            )
        )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(distribution_rows),
        pd.concat(adjusted_rows, ignore_index=True),
    )


def _bootstrap_one_candidate(
    candidate: pd.Series,
    analysis_rows: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    family = analysis_rows[
        analysis_rows["modality"].eq(candidate["modality"])
        & analysis_rows["marker"].eq(candidate["marker"])
        & analysis_rows["outcome"].eq(candidate["outcome"])
    ].copy()
    include_icv = str(candidate["modality"]) == "smri_adnimerge"
    mask = _complete_case_mask(
        family,
        str(candidate["exposure"]),
        include_icv=include_icv,
    )
    complete = family.loc[mask].reset_index(drop=True)
    covariates, _ = build_longitudinal_covariate_matrix(
        complete, include_icv=include_icv
    )
    exposure = pd.to_numeric(
        complete[str(candidate["exposure"])], errors="coerce"
    ).to_numpy(float)
    outcome = pd.to_numeric(
        complete["annualized_decline_score"], errors="coerce"
    ).to_numpy(float)
    marker = pd.to_numeric(complete["value"], errors="coerce").to_numpy(float)
    rng = np.random.default_rng(seed)
    indirect: list[float] = []
    a_path: list[float] = []
    b_path: list[float] = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(complete), size=len(complete))
        fitted, marker_mask = mediation_screen(
            exposure[index],
            outcome[index],
            marker[index].reshape(-1, 1),
            covariates[index],
        )
        if not marker_mask[0] or fitted.empty:
            continue
        row = fitted.iloc[0]
        values = (
            float(row["indirect_effect_std"]),
            float(row["a_path_std"]),
            float(row["b_path_std"]),
        )
        if all(np.isfinite(value) for value in values):
            indirect.append(values[0])
            a_path.append(values[1])
            b_path.append(values[2])
    if not indirect:
        raise RuntimeError(
            f"No valid bootstrap replicate for candidate rank {candidate['kg_rank']}"
        )
    indirect_array = np.asarray(indirect, dtype=float)
    point = float(candidate["indirect_effect_std"])
    same_sign = (
        indirect_array > 0 if point >= 0 else indirect_array < 0
    )
    lower_or_equal = int(np.count_nonzero(indirect_array <= 0))
    greater_or_equal = int(np.count_nonzero(indirect_array >= 0))
    empirical_p = min(
        1.0,
        2.0
        * (1 + min(lower_or_equal, greater_or_equal))
        / (len(indirect_array) + 1),
    )
    ci_low, ci_high = np.quantile(indirect_array, [0.025, 0.975])
    return {
        "kg_rank": int(candidate["kg_rank"]),
        "mapping_score": float(candidate["mapping_score"]),
        "primary_hypothesis_id": str(candidate["primary_hypothesis_id"]),
        "exposure": str(candidate["exposure"]),
        "pathway_name": str(candidate["pathway_name"]),
        "modality": str(candidate["modality"]),
        "marker": str(candidate["marker"]),
        "outcome": str(candidate["outcome"]),
        "n_subjects": int(len(complete)),
        "point_indirect_effect_std": point,
        "point_sobel_p": float(candidate["sobel_p"]),
        "bootstrap_replicates_requested": n_resamples,
        "bootstrap_replicates_valid": int(len(indirect_array)),
        "bootstrap_indirect_mean": float(indirect_array.mean()),
        "bootstrap_indirect_median": float(np.median(indirect_array)),
        "bootstrap_ci95_low": float(ci_low),
        "bootstrap_ci95_high": float(ci_high),
        "bootstrap_sign_consistency": float(same_sign.mean()),
        "bootstrap_empirical_p_two_sided": empirical_p,
        "bootstrap_a_path_mean": float(np.mean(a_path)),
        "bootstrap_b_path_mean": float(np.mean(b_path)),
    }


def bootstrap_top_hits(
    ranked_results: pd.DataFrame,
    analysis_rows: pd.DataFrame,
    *,
    top_n: int,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    chain_hits = ranked_results[
        pd.to_numeric(ranked_results["sobel_p"], errors="coerce").lt(alpha)
        & pd.to_numeric(ranked_results["a_path_p"], errors="coerce").lt(alpha)
        & pd.to_numeric(ranked_results["b_path_p"], errors="coerce").lt(alpha)
    ].sort_values("kg_rank", kind="stable")
    selected = chain_hits.head(top_n)
    rows = [
        _bootstrap_one_candidate(
            candidate,
            analysis_rows,
            n_resamples=n_resamples,
            seed=seed + int(candidate["kg_rank"]) * 1009,
        )
        for _, candidate in selected.iterrows()
    ]
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical_release(path: Path) -> dict[str, object]:
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("case_study_id") != "case2_pathway_mediation":
        raise ValueError("canonical release is not scoped to case2_pathway_mediation")
    files = release.get("files") or {}
    for name, expected in CURRENT_CANONICAL_SHA256.items():
        actual = str((files.get(name) or {}).get("sha256", "")).upper()
        if actual != expected:
            raise ValueError(
                f"canonical release {name} SHA-256 mismatch: {actual} != {expected}"
            )
    return release


def run(args: argparse.Namespace) -> dict[str, object]:
    hypotheses_path = args.hypotheses or _latest_file(
        DEFAULT_HYPOTHESIS_PARENT,
        "adni_longitudinal_kg_*",
        "hypotheses_raw.json",
    )
    smoke_results = args.smoke_results or _latest_file(
        DEFAULT_SMOKE_PARENT,
        "*",
        "all_longitudinal_mediation_results.parquet",
    )
    canonical_release_path = (
        args.canonical_release
        if args.canonical_release is not None
        else hypotheses_path.parent / "canonical_kg_release.json"
    )
    smoke_root = smoke_results.parent
    selection_path = smoke_root / "analysis_row_selection.parquet"
    exposure_path = smoke_root / "selected_pathway_exposures.csv"
    pathway_catalog_path = args.pathway_root / "pathway_catalog.csv"
    required = (
        hypotheses_path,
        canonical_release_path,
        smoke_results,
        selection_path,
        exposure_path,
        pathway_catalog_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    canonical_release = _load_canonical_release(canonical_release_path)
    canonical_output_path = args.output_root / "canonical_kg_release.json"
    write_release_manifest(canonical_output_path, canonical_release)

    hypotheses = load_hypotheses(hypotheses_path)
    results = pd.read_parquet(smoke_results)
    analysis_rows = pd.read_parquet(selection_path)
    exposures = pd.read_csv(exposure_path)
    pathway_catalog = pd.read_csv(pathway_catalog_path)
    key_columns = [
        "exposure",
        "pathway_id",
        "pathway_name",
        "threshold_label",
        "gene_count",
        "modality",
        "marker",
        "outcome",
    ]
    universe = results.loc[:, key_columns].drop_duplicates()
    ranked, support, mapping_audit = build_ranked_candidates(
        hypotheses,
        exposures,
        pathway_catalog,
        universe,
    )
    merge_keys = ["exposure", "modality", "marker", "outcome"]
    ranked_results = ranked.merge(
        results,
        on=merge_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    if ranked_results["sobel_p"].isna().any():
        raise ValueError("Ranked candidates are missing completed smoke results")
    metrics, random_distribution, topk_adjusted = evaluate_prioritisation(
        ranked_results,
        k_values=args.k,
        alpha=args.alpha,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    bootstrap = bootstrap_top_hits(
        ranked_results,
        analysis_rows,
        top_n=args.bootstrap_top,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
        alpha=args.alpha,
    )

    output_paths = {
        "ranked_csv": args.output_root / "kg_ranked_longitudinal_results.csv",
        "ranked_parquet": args.output_root
        / "kg_ranked_longitudinal_results.parquet",
        "support": args.output_root / "candidate_hypothesis_support.parquet",
        "mapping_audit": args.output_root / "hypothesis_mapping_audit.csv",
        "metrics": args.output_root / "prioritisation_by_k.csv",
        "random": args.output_root / "random_distribution.parquet",
        "topk": args.output_root / "topk_adjusted_results.csv",
        "bootstrap": args.output_root / "bootstrap_top_hits.csv",
    }
    ranked_results.to_csv(output_paths["ranked_csv"], index=False)
    ranked_results.to_parquet(
        output_paths["ranked_parquet"], index=False, compression="zstd"
    )
    support.to_parquet(output_paths["support"], index=False, compression="zstd")
    mapping_audit.to_csv(output_paths["mapping_audit"], index=False)
    metrics.to_csv(output_paths["metrics"], index=False)
    random_distribution.to_parquet(
        output_paths["random"], index=False, compression="zstd"
    )
    topk_adjusted.to_csv(output_paths["topk"], index=False)
    bootstrap.to_csv(output_paths["bootstrap"], index=False)

    now = datetime.now(timezone(timedelta(hours=8)))
    unique_ok = not ranked_results.duplicated(merge_keys).any()
    rank_ok = ranked_results["kg_rank"].tolist() == list(
        range(1, len(ranked_results) + 1)
    )
    complete_ok = len(ranked_results) == len(universe)
    audit = {
        "status": "passed" if unique_ok and rank_ok and complete_ok else "failed",
        "ranking_result_blind_in_code": True,
        "human_preregistration": False,
        "human_preregistration_note": (
            "The ADNI smoke statistics were inspected before this ranking run; "
            "treat this as a method-development pilot, not a preregistered test."
        ),
        "unique_candidates": bool(unique_ok),
        "complete_rank_sequence": bool(rank_ok),
        "complete_executable_universe": bool(complete_ok),
        "candidate_count": int(len(ranked_results)),
        "raw_hypotheses": int(len(hypotheses)),
        "meaningfully_mapped_hypotheses": int(
            mapping_audit["meaningfully_mapped"].sum()
        ),
        "bootstrap_candidates": int(len(bootstrap)),
    }
    task_manifest = {
        "session_id": now.strftime("case2-kg-longitudinal-%Y%m%d-%H%M%S"),
        "tasks": [
            "freeze KG-derived Case 2 hypotheses",
            "rank the complete executable panel without experimental statistics",
            "merge completed longitudinal mediation results after ranking",
            "compare Top-K discoveries with random candidate orderings",
            "bootstrap the highest-ranked nominal chain discoveries",
            "hash and audit all outputs",
        ],
        "success_criteria": {
            "complete_candidate_universe": len(universe),
            "no_result_column_in_ranking": True,
            "random_resamples": args.resamples,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
    }
    environment = {
        "created_at_hkt": now.isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "random_seed": args.seed,
    }
    task_path = args.output_root / "experiment_task_manifest.json"
    env_path = args.output_root / "environment_manifest.json"
    audit_path = args.output_root / "experiment_audit_report.md"
    experiment_path = args.output_root / "EXPERIMENT.md"
    task_path.write_text(json.dumps(task_manifest, indent=2), encoding="utf-8")
    env_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    audit_path.write_text(
        "# Case Study 2 KG-Guided Longitudinal Audit\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in audit.items())
        + "\n",
        encoding="utf-8",
    )
    experiment_path.write_text(
        "# Case Study 2 KG-Guided Longitudinal Pilot\n\n"
        f"Generated: {now.isoformat(timespec='seconds')}\n\n"
        "Outcome-blind KG ranking of the completed ADNI pathway PRS to imaging "
        "marker to subsequent clinical decline panel. Experimental statistics "
        "are merged only after rank assignment. The analysis is associative, "
        "not causal, and this run is a post hoc method-development pilot.\n\n"
        f"- Raw KG hypotheses: {len(hypotheses)}\n"
        f"- Executable candidate tests: {len(ranked_results)}\n"
        f"- Random orderings: {args.resamples}\n"
        f"- Bootstrapped candidates: {len(bootstrap)}\n",
        encoding="utf-8",
    )

    hash_targets = [
        *output_paths.values(),
        task_path,
        env_path,
        audit_path,
        experiment_path,
        canonical_output_path,
    ]
    hashes = {path.name: _sha256(path) for path in hash_targets}
    hash_path = args.output_root / "output_sha256.json"
    hash_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    payload: dict[str, object] = {
        "created_at_hkt": now.isoformat(timespec="seconds"),
        "analysis_type": "outcome-blind KG prioritisation of longitudinal multimodal mediation tests",
        "causal_interpretation": False,
        "analysis_status": "post_hoc_method_development_pilot",
        "hypotheses_input": str(hypotheses_path),
        "canonical_kg_release": canonical_release,
        "smoke_results_input": str(smoke_results),
        "analysis_rows_input": str(selection_path),
        "output_root": str(args.output_root),
        "raw_hypotheses": len(hypotheses),
        "candidate_tests": len(ranked_results),
        "meaningfully_mapped_hypotheses": int(
            mapping_audit["meaningfully_mapped"].sum()
        ),
        "ranking_policy": (
            "For each executable pathway-marker-outcome chain, take the best "
            "KG composite multiplied by the geometric mean of pathway, imaging, "
            "and outcome alignment; add 0.10 and 0.05 of the next two supports. "
            "No mediation coefficient or P value enters ranking."
        ),
        "discovery_definition": (
            "Sobel BH q<0.05 within the frozen Top-K and both a- and b-path "
            "P<0.05; nominal chain hits require all three P values <0.05."
        ),
        "k_values": metrics["k"].astype(int).tolist(),
        "random_resamples": args.resamples,
        "bootstrap_resamples": args.bootstrap_resamples,
        "audit": audit,
        "outputs": {
            key: str(path) for key, path in {**output_paths, "hashes": hash_path}.items()
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--hypotheses", type=Path)
    parser.add_argument("--canonical-release", type=Path)
    parser.add_argument("--smoke-results", type=Path)
    parser.add_argument("--pathway-root", type=Path, default=DEFAULT_PATHWAY_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_PARENT / timestamp,
    )
    parser.add_argument("--k", type=int, action="append", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--bootstrap-top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20_260_731)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.k = tuple(sorted(set(args.k or DEFAULT_K_VALUES)))
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Last Updated At: 2026-08-16 01:17 HKT
