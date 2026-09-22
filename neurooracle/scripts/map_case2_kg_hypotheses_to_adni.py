"""Map Case Study 2 KG hypotheses to completed ADNI mediation tests.

The mapper is deliberately outcome blind: pathway thresholds, imaging
proxies, and candidate ordering are fixed before mediation results are read.
Reported signals remain exploratory association-based mediation results, not
causal effects.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CASE2_ROOT = Path(
    r"\\192.168.3.61\data\Dataset\genetics\ADNI\derived\qc"
) / "case2_adni_genetics_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPOTHESES = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "experiments"
    / "case2"
    / "adni_pathway_mediation_20260731"
    / "hypotheses_raw.json"
)
DEFAULT_PATHWAY_DIR = (
    CASE2_ROOT
    / "postimputation"
    / "case2_adni_common_dr2_0p8_v1"
    / "features"
    / "ad_pathway_prs_v1"
)
DEFAULT_RESULTS = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_pathway_mediation_schaefer400_v1"
    / "all_marker_results.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    CASE2_ROOT
    / "experiments"
    / "case2_adni_kg_guided_v1"
)

THRESHOLD_PRIORITY = ("p1em03", "p5em02")
CONNECTIVITY_FEATURES = (
    "corr_mean_abs",
    "corr_node_degree_abs_top10",
)
DEFAULT_K_VALUES = (10, 25, 50, 100, 200, 250)
RESULT_COLUMNS = (
    "exposure",
    "outcome",
    "marker_id",
    "source",
    "atlas",
    "roi_index",
    "roi_id",
    "roi_name",
    "parcel_name",
    "hemisphere",
    "network",
    "structure_class",
    "feature",
    "a_path_std",
    "a_path_p",
    "b_path_std",
    "b_path_p",
    "total_effect_std",
    "direct_effect_std",
    "indirect_effect_std",
    "sobel_p",
    "sobel_q_family",
    "n",
    "outcome_higher_is",
    "abs_indirect_effect_std",
    "fdr_significant",
    "exposure_pathway_id",
    "exposure_pathway_source",
    "exposure_pathway_name",
    "exposure_threshold_label",
    "exposure_gene_count",
)

REGION_RULES = (
    {
        "name": "anterior cingulate cortex",
        "keywords": ("anterior cingulate cortex",),
        "parcel_pattern": r"SalVentAttn_Med|Cont_Cing",
        "network": None,
        "specificity": 0.70,
    },
    {
        "name": "posterior cingulate cortex",
        "keywords": ("posterior cingulate cortex",),
        "parcel_pattern": r"Default_pCunPCC",
        "network": None,
        "specificity": 0.80,
    },
    {
        "name": "insular cortex",
        "keywords": ("insular cortex", "insula"),
        "parcel_pattern": r"FrOperIns",
        "network": None,
        "specificity": 0.80,
    },
    {
        "name": "default mode network",
        "keywords": ("default mode network",),
        "parcel_pattern": None,
        "network": "Default",
        "specificity": 0.70,
    },
    {
        "name": "visual cortex",
        "keywords": (
            "visual cortex",
            "fusiform gyrus",
            "lateral occipital",
            "cuneus",
        ),
        "parcel_pattern": None,
        "network": "Vis",
        "specificity": 0.60,
    },
    {
        "name": "parietal cortex",
        "keywords": ("parietal lobe", "superior parietal lobule"),
        "parcel_pattern": r"_Par_",
        "network": None,
        "specificity": 0.55,
    },
)

UNAVAILABLE_REGIONS = (
    "amygdala",
    "hippocampus",
    "thalamus",
    "putamen",
    "basal ganglia",
    "entorhinal cortex",
    "parahippocampal gyrus",
)


def _normalise_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def load_hypotheses(path: Path) -> list[dict[str, Any]]:
    """Load either a list or the standard hypothesis-engine JSON envelope."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        hypotheses = payload
    elif isinstance(payload, dict):
        hypotheses = payload.get("hypotheses", payload.get("results", []))
    else:
        raise ValueError("Hypothesis JSON must contain a list or object")
    if not isinstance(hypotheses, list):
        raise ValueError("Hypothesis JSON does not contain a hypothesis list")
    return hypotheses


def _path_names(hypothesis: dict[str, Any]) -> list[str]:
    names = [str(hypothesis.get("source_name", ""))]
    for edge in hypothesis.get("path", []):
        to_name = str(edge.get("to_name", ""))
        if to_name and (not names or to_name != names[-1]):
            names.append(to_name)
    return names


def _split_genes(value: object) -> set[str]:
    return {
        gene.strip().upper()
        for gene in str(value or "").split(";")
        if gene.strip()
    }


def select_gene_pathways(
    gene: str,
    pathway_catalog: pd.DataFrame,
    score_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Select one prespecified PRS threshold for each pathway containing gene."""

    gene = gene.strip().upper()
    catalog = pathway_catalog.copy()
    catalog["_gene_set"] = catalog["genes"].map(_split_genes)
    catalog = catalog[catalog["_gene_set"].map(lambda genes: gene in genes)]
    if catalog.empty:
        return pd.DataFrame()

    manifest = score_manifest.merge(
        catalog.drop(columns="_gene_set"),
        on=[
            "pathway_id",
            "pathway_source",
            "pathway_name",
            "reference_id",
            "gene_count",
        ],
        how="inner",
        suffixes=("", "_catalog"),
    )
    manifest["_threshold_rank"] = manifest["threshold_label"].map(
        {
            threshold: index
            for index, threshold in enumerate(THRESHOLD_PRIORITY)
        }
    )
    manifest["_threshold_rank"] = manifest["_threshold_rank"].fillna(
        len(THRESHOLD_PRIORITY)
    )
    manifest = (
        manifest.sort_values(
            ["pathway_id", "_threshold_rank", "score_name"],
            kind="stable",
        )
        .drop_duplicates("pathway_id", keep="first")
        .copy()
    )
    gene_count = pd.to_numeric(manifest["gene_count"], errors="coerce").fillna(1)
    manifest["pathway_specificity"] = (
        0.70 + 0.30 / np.sqrt(gene_count.clip(lower=1))
    )
    manifest["source_gene"] = gene
    return manifest.drop(columns="_threshold_rank")


def map_outcome(name: str) -> dict[str, object] | None:
    """Map a claim outcome to one available ADNI endpoint."""

    text = _normalise_text(name)
    if "clinical dementia rating" in text or "cdr-sb" in text or "cdrsb" in text:
        return {
            "outcome": "CDRSB",
            "specificity": 1.0,
            "mapping": "exact_scale",
        }
    if "adas13" in text or "adas-13" in text:
        return {
            "outcome": "ADAS13",
            "specificity": 1.0,
            "mapping": "exact_scale",
        }
    if "alzheimer" in text or re.search(r"\b(mci|dementia)\b", text):
        return {
            "outcome": "diagnosis_code",
            "specificity": 0.85,
            "mapping": "diagnostic_severity_proxy",
        }
    if "mmse" in text or "mini-mental state" in text:
        return {
            "outcome": "MMSE",
            "specificity": 1.0,
            "mapping": "exact_scale",
        }
    if any(
        term in text
        for term in (
            "general cognition",
            "cognitive performance",
            "cognitive ability",
        )
    ):
        return {
            "outcome": "MMSE",
            "specificity": 0.60,
            "mapping": "broad_cognition_proxy",
        }
    return None


def _detect_regions(text: str) -> tuple[list[dict[str, object]], list[str]]:
    matched: list[dict[str, object]] = []
    for rule in REGION_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            matched.append(rule)
    unavailable = [region for region in UNAVAILABLE_REGIONS if region in text]
    return matched, unavailable


def map_imaging_markers(
    hypothesis: dict[str, Any],
    marker_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object] | None, str]:
    """Map an imaging claim to prespecified Schaefer-400 fMRI proxies."""

    text = _normalise_text(" | ".join(_path_names(hypothesis)))
    if "functional connectivity" in text or "rs-fmri" in text:
        modality = "resting_state_functional_connectivity"
        features = CONNECTIVITY_FEATURES
        base_specificity = 0.80
    elif "falff" in text:
        modality = "fractional_alff"
        features = ("roi_falff_proxy",)
        base_specificity = 0.75
    elif "amplitude of low-frequency fluctuation" in text or re.search(
        r"\balff\b",
        text,
    ):
        modality = "alff"
        features = ("roi_alff_proxy",)
        base_specificity = 0.75
    else:
        unavailable_modalities = (
            "regional homogeneity",
            "reho",
            "tau_suvr",
            "amyloid burden",
            "cortical thickness",
            "regional_volume",
            "gray_matter_density",
            "fractional anisotropy",
            "eeg coherence",
        )
        detected = [
            modality_name
            for modality_name in unavailable_modalities
            if modality_name in text
        ]
        reason = (
            "unsupported_imaging_modality:" + ";".join(detected)
            if detected
            else "no_executable_imaging_measure"
        )
        return marker_catalog.iloc[0:0].copy(), None, reason

    regions, unavailable_regions = _detect_regions(text)
    feature_mask = marker_catalog["feature"].isin(features)
    spatial_mask = pd.Series(False, index=marker_catalog.index)
    for rule in regions:
        parcel_pattern = rule["parcel_pattern"]
        network = rule["network"]
        if parcel_pattern:
            spatial_mask |= marker_catalog["parcel_name"].fillna("").str.contains(
                str(parcel_pattern),
                case=False,
                regex=True,
            )
        elif network:
            spatial_mask |= marker_catalog["network"].eq(network)

    if not regions:
        return (
            marker_catalog.iloc[0:0].copy(),
            None,
            "no_cortical_spatial_mapping",
        )

    markers = marker_catalog.loc[feature_mask & spatial_mask].copy()
    if markers.empty:
        return markers, None, "no_matching_schaefer400_marker"

    total_regions = len(regions) + len(unavailable_regions)
    coverage = len(regions) / max(total_regions, 1)
    spatial_specificity = max(float(rule["specificity"]) for rule in regions)
    imaging_specificity = (
        base_specificity
        * spatial_specificity
        * (0.50 + 0.50 * coverage)
    )
    mapping = {
        "modality": modality,
        "features": ";".join(features),
        "mapped_regions": ";".join(str(rule["name"]) for rule in regions),
        "unavailable_regions": ";".join(unavailable_regions),
        "spatial_coverage": coverage,
        "specificity": imaging_specificity,
        "mapping": (
            "exact_cortical_proxy"
            if not unavailable_regions
            else "partial_cortical_proxy"
        ),
    }
    return markers, mapping, ""


def _hypothesis_score(hypothesis: dict[str, Any]) -> float:
    value = hypothesis.get("composite_score")
    if value is None:
        value = hypothesis.get("confidence_score", 0.0)
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _candidate_score(
    kg_score: float,
    pathway_specificity: float,
    imaging_specificity: float,
    outcome_specificity: float,
) -> float:
    return (
        0.55 * kg_score
        + 0.15 * pathway_specificity
        + 0.20 * imaging_specificity
        + 0.10 * outcome_specificity
    )


def _stable_unique(values: Iterable[object]) -> str:
    return ";".join(dict.fromkeys(str(value) for value in values if str(value)))


def _diversified_rank(candidates: pd.DataFrame) -> pd.DataFrame:
    """Round-robin source hypotheses without using experimental outcomes."""

    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values(
        [
            "mapping_score",
            "pathway_specificity",
            "feature_priority",
            "marker_id",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    groups = []
    for hypothesis_id, group in ordered.groupby(
        "primary_hypothesis_id",
        sort=False,
    ):
        groups.append(
            (
                hypothesis_id,
                float(group["mapping_score"].max()),
                deque(group.index.tolist()),
            )
        )
    groups.sort(key=lambda item: (-item[1], item[0]))

    ranked_indices: list[int] = []
    while any(queue for _, _, queue in groups):
        for _, _, queue in groups:
            if queue:
                ranked_indices.append(queue.popleft())
    ranked = ordered.loc[ranked_indices].reset_index(drop=True)
    ranked.insert(0, "kg_rank", np.arange(1, len(ranked) + 1))
    return ranked


def build_executable_candidates(
    hypotheses: list[dict[str, Any]],
    pathway_catalog: pd.DataFrame,
    score_manifest: pd.DataFrame,
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and rank executable experiments without inspecting their results."""

    marker_columns = [
        "marker_id",
        "source",
        "atlas",
        "roi_index",
        "roi_id",
        "roi_name",
        "parcel_name",
        "hemisphere",
        "network",
        "structure_class",
        "feature",
    ]
    marker_catalog = results.loc[:, marker_columns].drop_duplicates("marker_id")
    raw_candidates: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("id", ""))
        gene = str(hypothesis.get("source_name", "")).strip().upper()
        target = str(hypothesis.get("target_name", "")).strip()
        path_text = " -> ".join(_path_names(hypothesis))
        pathways = select_gene_pathways(gene, pathway_catalog, score_manifest)
        outcome_mapping = map_outcome(target)
        markers, imaging_mapping, imaging_reason = map_imaging_markers(
            hypothesis,
            marker_catalog,
        )

        reasons = []
        if pathways.empty:
            reasons.append("gene_not_in_available_pathway_prs")
        if outcome_mapping is None:
            reasons.append("outcome_not_available")
        if imaging_mapping is None:
            reasons.append(imaging_reason)
        executable = not reasons
        audit_rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "source_gene": gene,
                "target_name": target,
                "path_text": path_text,
                "kg_composite_score": _hypothesis_score(hypothesis),
                "pathways_available": int(len(pathways)),
                "markers_available": int(len(markers)),
                "mapped_outcome": (
                    outcome_mapping["outcome"] if outcome_mapping else ""
                ),
                "imaging_mapping": (
                    imaging_mapping["mapping"] if imaging_mapping else ""
                ),
                "executable": executable,
                "reason": ";".join(reasons),
            }
        )
        if not executable:
            continue

        kg_score = _hypothesis_score(hypothesis)
        for pathway in pathways.itertuples(index=False):
            pathway_specificity = float(pathway.pathway_specificity)
            for marker in markers.itertuples(index=False):
                score = _candidate_score(
                    kg_score,
                    pathway_specificity,
                    float(imaging_mapping["specificity"]),
                    float(outcome_mapping["specificity"]),
                )
                raw_candidates.append(
                    {
                        "primary_hypothesis_id": hypothesis_id,
                        "source_gene": gene,
                        "path_text": path_text,
                        "kg_composite_score": kg_score,
                        "exposure": str(pathway.score_name),
                        "pathway_id": str(pathway.pathway_id),
                        "pathway_name": str(pathway.pathway_name),
                        "pathway_source": str(pathway.pathway_source),
                        "pathway_threshold": str(pathway.threshold_label),
                        "pathway_gene_count": int(pathway.gene_count),
                        "pathway_specificity": pathway_specificity,
                        "outcome": str(outcome_mapping["outcome"]),
                        "outcome_mapping": str(outcome_mapping["mapping"]),
                        "outcome_specificity": float(
                            outcome_mapping["specificity"]
                        ),
                        "marker_id": str(marker.marker_id),
                        "imaging_mapping": str(imaging_mapping["mapping"]),
                        "imaging_modality": str(imaging_mapping["modality"]),
                        "mapped_regions": str(imaging_mapping["mapped_regions"]),
                        "unavailable_regions": str(
                            imaging_mapping["unavailable_regions"]
                        ),
                        "spatial_coverage": float(
                            imaging_mapping["spatial_coverage"]
                        ),
                        "imaging_specificity": float(
                            imaging_mapping["specificity"]
                        ),
                        "feature_priority": CONNECTIVITY_FEATURES.index(
                            str(marker.feature)
                        )
                        if str(marker.feature) in CONNECTIVITY_FEATURES
                        else 0,
                        "mapping_score": score,
                    }
                )

    audit = pd.DataFrame(audit_rows)
    candidates = pd.DataFrame(raw_candidates)
    if candidates.empty:
        return candidates, audit

    key = ["exposure", "outcome", "marker_id"]
    support = (
        candidates.groupby(key, sort=False)
        .agg(
            supporting_hypothesis_ids=(
                "primary_hypothesis_id",
                _stable_unique,
            ),
            supporting_genes=("source_gene", _stable_unique),
        )
        .reset_index()
    )
    candidates = (
        candidates.sort_values(
            ["mapping_score", "primary_hypothesis_id"],
            ascending=[False, True],
            kind="stable",
        )
        .drop_duplicates(key, keep="first")
        .merge(support, on=key, how="left", validate="one_to_one")
    )
    candidates = _diversified_rank(candidates)

    result_keys = results.drop_duplicates(key)
    candidates = candidates.merge(
        result_keys,
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    missing_results = candidates["sobel_p"].isna()
    if missing_results.any():
        raise ValueError(
            "Mapped candidates are missing completed mediation results: "
            f"{int(missing_results.sum())}"
        )
    return candidates, audit


def fixed_threshold_exposures(score_manifest: pd.DataFrame) -> tuple[str, ...]:
    """Select one outcome-blind score per pathway for the random universe."""

    manifest = score_manifest.copy()
    manifest["_threshold_rank"] = manifest["threshold_label"].map(
        {
            threshold: index
            for index, threshold in enumerate(THRESHOLD_PRIORITY)
        }
    )
    manifest["_threshold_rank"] = manifest["_threshold_rank"].fillna(
        len(THRESHOLD_PRIORITY)
    )
    selected = (
        manifest.sort_values(
            ["pathway_id", "_threshold_rank", "score_name"],
            kind="stable",
        )
        .drop_duplicates("pathway_id")
        .loc[:, "score_name"]
    )
    return tuple(selected.astype(str))


def _random_baseline_rows(
    *,
    baseline: str,
    population: pd.DataFrame,
    ranked: pd.DataFrame,
    k_values: Iterable[int],
    n_resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    signal = population["exploratory_signal"].fillna(False).to_numpy(bool)
    population_size = int(len(signal))
    population_hits = int(signal.sum())
    rng = np.random.default_rng(seed)
    metric_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for k in k_values:
        if k > len(ranked) or k > population_size:
            continue
        top = ranked.head(k)
        observed = int(top["exploratory_signal"].fillna(False).sum())
        random_hits = rng.hypergeometric(
            population_hits,
            population_size - population_hits,
            k,
            size=n_resamples,
        )
        random_mean = float(random_hits.mean())
        ci_low, ci_high = np.quantile(random_hits, [0.025, 0.975])
        empirical_p = float(
            (1 + np.count_nonzero(random_hits >= observed))
            / (n_resamples + 1)
        )
        metric_rows.append(
            {
                "baseline": baseline,
                "k": int(k),
                "kg_exploratory_hits": observed,
                "kg_exploratory_rate": observed / k,
                "kg_nominal_hits": int(
                    top["nominal_hit"].fillna(False).sum()
                ),
                "kg_family_fdr_hits": int(
                    top["family_fdr_hit"].fillna(False).sum()
                ),
                "kg_median_sobel_p": float(top["sobel_p"].median()),
                "kg_median_abs_indirect_effect_std": float(
                    top["abs_indirect_effect_std"].median()
                ),
                "random_population_size": population_size,
                "random_population_hits": population_hits,
                "random_mean_hits": random_mean,
                "random_ci95_low": float(ci_low),
                "random_ci95_high": float(ci_high),
                "fold_over_random": (
                    observed / random_mean
                    if random_mean > 0
                    else math.inf if observed > 0 else 1.0
                ),
                "empirical_p_random_ge_kg": empirical_p,
                "resamples": n_resamples,
            }
        )
        distribution_rows.extend(
            {
                "baseline": baseline,
                "k": int(k),
                "resample": index,
                "random_hits": int(value),
            }
            for index, value in enumerate(random_hits, start=1)
        )
    return metric_rows, distribution_rows


def evaluate_prioritisation(
    candidates: pd.DataFrame,
    results: pd.DataFrame,
    score_manifest: pd.DataFrame,
    *,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
    nominal_alpha: float = 0.05,
    minimum_effect: float = 0.10,
    n_resamples: int = 10_000,
    seed: int = 20_260_731,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare KG ranking with exact hypergeometric random baselines."""

    ranked = candidates.sort_values("kg_rank", kind="stable").copy()
    for frame in (ranked, results):
        frame["nominal_hit"] = frame["sobel_p"].lt(nominal_alpha)
        frame["exploratory_signal"] = (
            frame["nominal_hit"]
            & frame["abs_indirect_effect_std"].ge(minimum_effect)
        )
        frame["family_fdr_hit"] = frame["sobel_q_family"].lt(0.05)

    fixed_exposures = fixed_threshold_exposures(score_manifest)
    global_universe = results[results["exposure"].isin(fixed_exposures)].copy()
    family_keys = ranked.loc[:, ["exposure", "outcome"]].drop_duplicates()
    matched_universe = results.merge(
        family_keys,
        on=["exposure", "outcome"],
        how="inner",
        validate="many_to_one",
    )

    metric_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for baseline_index, (baseline, population) in enumerate(
        (
            ("global_fixed_threshold_random", global_universe),
            ("matched_exposure_outcome_random", matched_universe),
        )
    ):
        metrics, distribution = _random_baseline_rows(
            baseline=baseline,
            population=population,
            ranked=ranked,
            k_values=k_values,
            n_resamples=n_resamples,
            seed=seed + baseline_index,
        )
        metric_rows.extend(metrics)
        distribution_rows.extend(distribution)
    return ranked, pd.DataFrame(metric_rows), pd.DataFrame(distribution_rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run mapping, attach completed tests, and write auditable outputs."""

    required_paths = (
        args.hypotheses,
        args.pathway_dir / "pathway_catalog.csv",
        args.pathway_dir / "score_manifest.csv",
        args.results,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"Output root is not empty: {args.output_root}. Use --force."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    hypotheses = load_hypotheses(args.hypotheses)
    pathway_catalog = pd.read_csv(args.pathway_dir / "pathway_catalog.csv")
    score_manifest = pd.read_csv(args.pathway_dir / "score_manifest.csv")
    results = pd.read_parquet(args.results, columns=list(RESULT_COLUMNS))

    candidates, audit = build_executable_candidates(
        hypotheses,
        pathway_catalog,
        score_manifest,
        results,
    )
    if candidates.empty:
        raise RuntimeError("No KG hypothesis could be mapped to the ADNI result table")
    ranked, metrics, random_distribution = evaluate_prioritisation(
        candidates,
        results,
        score_manifest,
        k_values=args.k,
        nominal_alpha=args.nominal_alpha,
        minimum_effect=args.minimum_effect,
        n_resamples=args.resamples,
        seed=args.seed,
    )

    ranked.to_parquet(
        args.output_root / "kg_ranked_mediation_results.parquet",
        index=False,
        compression="zstd",
    )
    ranked.to_csv(
        args.output_root / "kg_ranked_mediation_results.csv",
        index=False,
    )
    audit.to_csv(args.output_root / "hypothesis_mapping_audit.csv", index=False)
    audit.loc[~audit["executable"]].to_csv(
        args.output_root / "unmapped_hypotheses.csv",
        index=False,
    )
    metrics.to_csv(args.output_root / "prioritisation_by_k.csv", index=False)
    random_distribution.to_parquet(
        args.output_root / "random_distribution.parquet",
        index=False,
        compression="zstd",
    )

    now_hkt = datetime.now(timezone(timedelta(hours=8))).isoformat()
    payload: dict[str, object] = {
        "created_at_hkt": now_hkt,
        "analysis_type": (
            "outcome-blind KG prioritisation over completed cross-sectional "
            "association-based mediation tests"
        ),
        "causal_interpretation": False,
        "hypotheses_input": str(args.hypotheses),
        "pathway_dir": str(args.pathway_dir),
        "mediation_results": str(args.results),
        "output_root": str(args.output_root),
        "raw_hypotheses": len(hypotheses),
        "executable_source_hypotheses": int(audit["executable"].sum()),
        "unmapped_source_hypotheses": int((~audit["executable"]).sum()),
        "ranked_candidate_experiments": len(ranked),
        "source_genes": sorted(
            {
                gene
                for value in ranked["supporting_genes"]
                for gene in str(value).split(";")
                if gene
            }
        ),
        "pathways": int(ranked["pathway_id"].nunique()),
        "markers": int(ranked["marker_id"].nunique()),
        "outcomes": sorted(ranked["outcome"].unique()),
        "partial_proxy_warning": (
            "Schaefer-400 is cortical. Amygdala and hippocampus in mapped "
            "connectivity claims are not measured; ACC parcels are a partial "
            "cortical proxy and are labeled as such."
        ),
        "threshold_policy": (
            "One outcome-blind PRS per pathway: p1em03 when available, "
            "otherwise p5em02."
        ),
        "ranking_policy": (
            "0.55 KG composite + 0.15 pathway specificity + 0.20 imaging "
            "specificity + 0.10 outcome specificity; deterministic "
            "round-robin diversity across source hypotheses. No mediation "
            "result enters ranking."
        ),
        "exploratory_signal_definition": {
            "sobel_p_lt": args.nominal_alpha,
            "abs_indirect_effect_std_gte": args.minimum_effect,
        },
        "family_fdr_threshold": 0.05,
        "random_resamples": args.resamples,
        "random_seed": args.seed,
        "k_values": sorted(metrics["k"].unique().astype(int).tolist()),
        "outputs": {
            "ranked_results_csv": str(
                args.output_root / "kg_ranked_mediation_results.csv"
            ),
            "ranked_results_parquet": str(
                args.output_root / "kg_ranked_mediation_results.parquet"
            ),
            "mapping_audit": str(
                args.output_root / "hypothesis_mapping_audit.csv"
            ),
            "unmapped_hypotheses": str(
                args.output_root / "unmapped_hypotheses.csv"
            ),
            "prioritisation_by_k": str(
                args.output_root / "prioritisation_by_k.csv"
            ),
            "random_distribution": str(
                args.output_root / "random_distribution.parquet"
            ),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument("--pathway-dir", type=Path, default=DEFAULT_PATHWAY_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        default=None,
        help="Evaluation cutoff; repeat for multiple K values.",
    )
    parser.add_argument("--nominal-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-effect", type=float, default=0.10)
    parser.add_argument("--resamples", type=int, default=10_000)
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

# Last Updated At: 2026-07-31 03:31 HKT
