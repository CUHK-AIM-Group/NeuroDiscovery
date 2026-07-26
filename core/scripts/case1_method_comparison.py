"""Compare Case Study 1 published-method baselines against exhaustive readouts.

The exhaustive experiment is treated as the executed search space. This script
does not run neuroimaging models; it ranks already-tested candidates by
paper-grounded generator policies and measures how quickly each policy recovers
the exhaustive ground-truth discoveries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALL_TESTS = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot"
    r"\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_OUT_DIR = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot"
    r"\method_comparison"
)
DEFAULT_SURFACE_PANEL = DEFAULT_OUT_DIR / "surface" / "fig_cs1_generator_surface_recovery_comparison.png"
DEFAULT_COMPACT_SURFACE_PANEL = DEFAULT_OUT_DIR / "surface" / "fig_cs1_generator_surface_recovery_compact.png"
DEFAULT_ATLAS_ICON = DEFAULT_OUT_DIR / "surface" / "case1_brain_atlas_map_icon.png"
DEFAULT_CASE1_KG = (
    REPO_ROOT
    / "neurooracle"
    / "data"
    / "cs_runs"
    / "phase2_case1_transdiagnostic_v1"
    / "knowledge_graph.json"
)
DEFAULT_FULL_KG = REPO_ROOT / "neurooracle" / "data" / "full_v2" / "knowledge_graph.json"

PALETTE = {
    "exhaustive_gt": "#272727",
    "ai_scientist_v2": "#4C78A8",
    "co_scientist_style": "#7E6AAE",
    "data_to_paper_style": "#59A14F",
    "sciagents_style": "#F28E2B",
    "virtual_lab_style": "#9C755F",
    "openscholar_rag": "#76B7B2",
    "neurodiscovery": "#D9544D",
    "neurodiscovery_positive_only": "#D9544D",
    "neurodiscovery_negative_feature_only": "#E07A5F",
    "neurodiscovery_negative_pair_only": "#8C2D25",
    "neurodiscovery_negative_context_only": "#6F4E7C",
    "neurodiscovery_negative_hybrid": "#3D405B",
}
BAND_ALPHA = {
    "ai_scientist_v2": 0.12,
    "co_scientist_style": 0.11,
    "data_to_paper_style": 0.12,
    "sciagents_style": 0.12,
    "virtual_lab_style": 0.11,
    "openscholar_rag": 0.11,
    "neurodiscovery": 0.13,
}

PANEL_SVG_DIRNAME = "panel_svgs"
METHOD_LABELS = {
    "exhaustive_gt": "Exhaustive GT",
    "ai_scientist_v2": "AI Scientist-v2",
    "co_scientist_style": "Co-Scientist",
    "data_to_paper_style": "data-to-paper",
    "sciagents_style": "SciAgents",
    "virtual_lab_style": "Virtual Lab-style",
    "openscholar_rag": "OpenScholar-RAG",
    "neurodiscovery": "NeuroDiscovery",
    "neurodiscovery_positive_only": "NeuroDiscovery positive-only",
    "neurodiscovery_negative_feature_only": "NeuroDiscovery + feature-only penalty",
    "neurodiscovery_negative_pair_only": "NeuroDiscovery + pair-only penalty",
    "neurodiscovery_negative_context_only": "NeuroDiscovery + context-only penalty",
    "neurodiscovery_negative_hybrid": "NeuroDiscovery + hybrid penalty",
}
SHORT_METHOD_LABELS = {
    "exhaustive_gt": "GT",
    "ai_scientist_v2": "AI\nScientist",
    "co_scientist_style": "Co-\nScientist",
    "data_to_paper_style": "data-to-\npaper",
    "sciagents_style": "SciAgents",
    "virtual_lab_style": "Virtual\nLab",
    "openscholar_rag": "Open-\nScholar",
    "neurodiscovery": "Neuro-\nDiscovery",
}
COMPACT_METHOD_LABELS = {
    "ai_scientist_v2": "AI Scientist",
    "co_scientist_style": "Co-Scientist",
    "data_to_paper_style": "data-to-paper",
    "sciagents_style": "SciAgents",
    "virtual_lab_style": "Virtual Lab",
    "openscholar_rag": "OpenScholar",
    "neurodiscovery": "NeuroDiscovery",
}
MARKERS = {
    "ai_scientist_v2": "o",
    "co_scientist_style": "s",
    "data_to_paper_style": "^",
    "sciagents_style": "D",
    "virtual_lab_style": "v",
    "openscholar_rag": "X",
    "neurodiscovery": "P",
    "neurodiscovery_positive_only": "P",
    "neurodiscovery_negative_feature_only": "o",
    "neurodiscovery_negative_pair_only": "*",
    "neurodiscovery_negative_context_only": "s",
    "neurodiscovery_negative_hybrid": "X",
}
BASELINE_METHODS = (
    "ai_scientist_v2",
    "co_scientist_style",
    "data_to_paper_style",
    "sciagents_style",
    "virtual_lab_style",
    "openscholar_rag",
)
GENERATOR_METHODS = (*BASELINE_METHODS, "neurodiscovery")

PUBLICATION_FIELDS = (
    "paper_title",
    "paper_year",
    "venue",
    "doi",
    "source_url",
    "baseline_family",
    "adaptation_note",
)
METHOD_PUBLICATIONS: dict[str, dict[str, object]] = {
    "exhaustive_gt": {
        "paper_title": "",
        "paper_year": "",
        "venue": "",
        "doi": "",
        "source_url": "",
        "baseline_family": "oracle reference",
        "adaptation_note": "All executed Case Study 1 candidates ranked by observed absolute adjusted effect size.",
    },
    "ai_scientist_v2": {
        "paper_title": "Towards end-to-end automation of AI research",
        "paper_year": 2026,
        "venue": "Nature",
        "doi": "10.1038/s41586-026-10265-5",
        "source_url": "https://www.nature.com/articles/s41586-026-10265-5",
        "baseline_family": "single-system autoresearch agent",
        "adaptation_note": (
            "Adapts the AI Scientist ideation stage to fixed-budget Case Study 1 "
            "disease-region-feature hypothesis generation; observed effect sizes and "
            "closed-loop labels are not exposed."
        ),
    },
    "co_scientist_style": {
        "paper_title": "Accelerating scientific discovery with Co-Scientist",
        "paper_year": 2026,
        "venue": "Nature",
        "doi": "10.1038/s41586-026-10644-y",
        "source_url": "https://www.nature.com/articles/s41586-026-10644-y",
        "baseline_family": "multi-agent hypothesis generation",
        "adaptation_note": (
            "Adapts the generate-critique-refine/rank workflow to the Case Study 1 "
            "schema; observed effect sizes and closed-loop labels are not exposed."
        ),
    },
    "data_to_paper_style": {
        "paper_title": "Autonomous LLM-Driven Research - from Data to Human-Verifiable Research Papers",
        "paper_year": 2025,
        "venue": "NEJM AI",
        "doi": "10.1056/AIoa2400555",
        "source_url": "https://ai.nejm.org/doi/abs/10.1056/AIoa2400555",
        "baseline_family": "traceable data-to-paper autonomous research workflow",
        "adaptation_note": (
            "Adapts the research-question/data-schema/planned-claim workflow to "
            "produce a ranked Case Study 1 hypothesis table instead of a manuscript; "
            "observed effect sizes and closed-loop labels are not exposed."
        ),
    },
    "sciagents_style": {
        "paper_title": "SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning",
        "paper_year": 2025,
        "venue": "Advanced Materials",
        "doi": "10.1002/adma.202413523",
        "source_url": "https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adma.202413523",
        "baseline_family": "knowledge-graph-guided multi-agent reasoning",
        "adaptation_note": (
            "Adapts graph-ontologist/scientist/critic/ranker reasoning over the "
            "frozen KG context; NeuroDiscovery scores and closed-loop labels are "
            "not exposed."
        ),
    },
    "virtual_lab_style": {
        "paper_title": "The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies",
        "paper_year": 2025,
        "venue": "Nature",
        "doi": "10.1038/s41586-025-09442-9",
        "source_url": "https://www.nature.com/articles/s41586-025-09442-9",
        "baseline_family": "multi-agent virtual scientific team",
        "adaptation_note": (
            "Adapts only the PI/scientist-agent meeting workflow to fixed-budget "
            "Case Study 1 hypothesis generation; wet-lab validation, domain-specific "
            "tools, observed effect sizes, and closed-loop labels are not exposed."
        ),
    },
    "openscholar_rag": {
        "paper_title": "Synthesizing scientific literature with retrieval-augmented language models",
        "paper_year": 2026,
        "venue": "Nature",
        "doi": "10.1038/s41586-025-10072-4",
        "source_url": "https://www.nature.com/articles/s41586-025-10072-4",
        "baseline_family": "retrieval-augmented scientific literature synthesis",
        "adaptation_note": (
            "Adapts retrieval-augmented literature synthesis to generate citation-backed "
            "ranked Case Study 1 hypotheses from pre-freeze evidence proxies; observed "
            "effect sizes and closed-loop labels are not exposed."
        ),
    },
    "neurodiscovery": {
        "paper_title": "",
        "paper_year": "",
        "venue": "This work",
        "doi": "",
        "source_url": "",
        "baseline_family": "NeuroDiscovery graph-guided closed-loop search",
        "adaptation_note": "Combines KG degree, disease-region support, neuroimaging priors, diversity, and closed-loop feedback.",
    },
    "neurodiscovery_positive_only": {
        "paper_title": "",
        "paper_year": "",
        "venue": "This work",
        "doi": "",
        "source_url": "",
        "baseline_family": "NeuroDiscovery closed-loop ablation",
        "adaptation_note": "Uses supported executed hypotheses to boost related disease-region-feature contexts; contradicted/negative outcomes are not used.",
    },
    "neurodiscovery_negative_proxy": {
        "paper_title": "",
        "paper_year": "",
        "venue": "This work",
        "doi": "",
        "source_url": "",
        "baseline_family": "NeuroDiscovery closed-loop ablation",
        "adaptation_note": (
            "Uses Case Study 1 executed non-GT-top outcomes as a contradicted-result "
            "proxy and penalizes similar disease, region, feature, source and pair contexts."
        ),
    },
    "neurodiscovery_delayed_negative_proxy": {
        "paper_title": "",
        "paper_year": "",
        "venue": "This work",
        "doi": "",
        "source_url": "",
        "baseline_family": "NeuroDiscovery closed-loop ablation",
        "adaptation_note": (
            "Delays the contradicted-result proxy until the search has accumulated "
            "an initial supported/unsupported evidence state."
        ),
    },
}

DISEASE_TERMS = {
    "ADHD": ("attention deficit hyperactivity disorder", "adhd"),
    "anxiety": ("anxiety disorders", "anxiety"),
    "bipolar": ("bipolar disorder", "mania"),
    "MDD_depression": ("major depressive disorder", "depression"),
    "OCD": ("obsessive-compulsive disorder", "ocd"),
    "PTSD": ("post-traumatic stress disorder", "posttraumatic stress disorder", "ptsd"),
    "psychosis_SZ_SZA": ("schizophrenia", "schizoaffective disorder", "psychosis"),
    "ASD": ("autism spectrum disorder", "autism"),
    "eating_disorder": ("eating disorders", "anorexia nervosa", "bulimia nervosa"),
}

REGION_ALIASES = {
    "temppole": ("temporal pole",),
    "temp pole": ("temporal pole",),
    "cing": ("cingulate cortex", "cingulate gyrus"),
    "paracingulate": ("paracingulate gyrus", "cingulate cortex"),
    "hipp": ("hippocampus",),
    "amyg": ("amygdala",),
    "thalam": ("thalamus",),
    "insula": ("insula", "insular cortex"),
    "putamen": ("putamen",),
    "caudate": ("caudate nucleus",),
    "pallid": ("globus pallidus", "pallidum"),
    "accumb": ("nucleus accumbens",),
    "front": ("frontal cortex", "prefrontal cortex"),
    "prefront": ("prefrontal cortex",),
    "orbitofrontal": ("orbitofrontal cortex",),
    "occip": ("occipital lobe", "visual cortex"),
    "pariet": ("parietal lobe", "parietal cortex"),
    "temporal": ("temporal lobe", "temporal cortex"),
    "limbic": ("limbic system",),
    "default": ("default mode network",),
    "somatomotor": ("motor cortex", "somatosensory cortex"),
    "salience": ("salience network",),
    "ventral attention": ("ventral attention network",),
    "dorsal attention": ("dorsal attention network",),
}

FEATURE_PRIOR = {
    "roi_falff_proxy": 1.00,
    "roi_alff_proxy": 0.96,
    "corr_node_degree_abs_top10": 0.94,
    "corr_node_degree_top10": 0.90,
    "roi_temporal_mean_abs": 0.88,
    "roi_temporal_mean": 0.84,
    "partial_positive_mean": 0.82,
    "partial_negative_mean": 0.82,
    "partial_mean_abs": 0.80,
    "corr_mean_abs": 0.78,
    "corr_negative_mean": 0.76,
    "corr_positive_mean": 0.74,
    "corr_mean": 0.72,
    "partial_mean": 0.70,
    "roi_temporal_variance": 0.68,
    "roi_temporal_std": 0.66,
    "volume": 0.58,
}

REGION_PRIOR_KEYWORDS = {
    "limbic": 1.00,
    "temporal pole": 0.98,
    "hippocampus": 0.92,
    "amygdala": 0.90,
    "thalamus": 0.88,
    "cingulate": 0.86,
    "insula": 0.84,
    "prefrontal": 0.82,
    "frontal": 0.74,
    "default": 0.72,
    "salience": 0.72,
    "striat": 0.70,
    "caudate": 0.70,
    "putamen": 0.70,
}


@dataclass
class KgIndex:
    degrees: dict[str, int]
    name_to_ids: dict[str, tuple[str, ...]]
    name_to_degree: dict[str, int]
    adjacency: dict[str, set[str]]


def normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo)


def load_kg_index(path: Path | None) -> KgIndex:
    if path is None or not path.exists():
        return KgIndex({}, {}, {}, {})
    with path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)

    concepts = graph.get("concepts", {})
    edges = graph.get("edges", [])
    degrees: Counter[str] = Counter()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = edge.get("source_id") or edge.get("source")
        target = edge.get("target_id") or edge.get("target")
        if not source or not target:
            continue
        degrees[source] += 1
        degrees[target] += 1
        adjacency[source].add(target)
        adjacency[target].add(source)

    name_to_ids: dict[str, set[str]] = defaultdict(set)
    name_to_degree: dict[str, int] = {}
    for cid, concept in concepts.items():
        names = [concept.get("preferred_name", "")]
        aliases = concept.get("aliases") or []
        if isinstance(aliases, list):
            names.extend(aliases[:30])
        degree = int(degrees.get(cid, 0))
        for name in names:
            key = normalize_text(name)
            if not key:
                continue
            name_to_ids[key].add(cid)
            if degree > name_to_degree.get(key, 0):
                name_to_degree[key] = degree

    frozen_ids = {key: tuple(sorted(ids)) for key, ids in name_to_ids.items()}
    return KgIndex(dict(degrees), frozen_ids, name_to_degree, dict(adjacency))


def candidate_region_terms(row: pd.Series) -> list[str]:
    raw_values = [
        row.get("anatomy_full", ""),
        row.get("anatomy_key", ""),
        row.get("roi_name", ""),
        row.get("network", ""),
        row.get("structure_class", ""),
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = normalize_text(raw)
        if text and text not in seen:
            terms.append(text)
            seen.add(text)
        for key, aliases in REGION_ALIASES.items():
            if key in text:
                for alias in aliases:
                    norm = normalize_text(alias)
                    if norm and norm not in seen:
                        terms.append(norm)
                        seen.add(norm)
    return terms


def disease_terms(name: str) -> list[str]:
    terms = [normalize_text(name)]
    for term in DISEASE_TERMS.get(name, ()):
        norm = normalize_text(term)
        if norm not in terms:
            terms.append(norm)
    return terms


def method_publication(method: str) -> dict[str, object]:
    return {field: METHOD_PUBLICATIONS.get(method, {}).get(field, "") for field in PUBLICATION_FIELDS}


def grouped_bar_offsets(n_methods: int, max_total_width: float = 0.82) -> tuple[np.ndarray, float]:
    width = min(0.16, max_total_width / max(n_methods, 1))
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width
    return offsets, width


def best_baseline_value(curves: pd.DataFrame, budget: int, metric: str) -> float:
    vals = [value_at_budget(curves, method, budget, metric)[0] for method in BASELINE_METHODS]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else np.nan


def best_baseline_matrix(matrices_by_method: dict[str, np.ndarray]) -> np.ndarray:
    baseline_matrices = [matrices_by_method[method] for method in BASELINE_METHODS if method in matrices_by_method]
    if not baseline_matrices:
        return np.zeros_like(matrices_by_method["neurodiscovery"])
    return np.maximum.reduce(baseline_matrices)


def resolve_degree(terms: Iterable[str], kg: KgIndex) -> int:
    best = 0
    for term in terms:
        norm = normalize_text(term)
        best = max(best, kg.name_to_degree.get(norm, 0))
    return best


def resolve_ids(terms: Iterable[str], kg: KgIndex, max_ids: int = 8) -> tuple[str, ...]:
    ids: set[str] = set()
    for term in terms:
        ids.update(kg.name_to_ids.get(normalize_text(term), ()))
    ranked = sorted(ids, key=lambda cid: kg.degrees.get(cid, 0), reverse=True)
    return tuple(ranked[:max_ids])


def pair_support(disease_ids: tuple[str, ...], region_ids: tuple[str, ...], kg: KgIndex) -> float:
    if not disease_ids or not region_ids or not kg.adjacency:
        return 0.0
    best = 0.0
    for did in disease_ids:
        dn = kg.adjacency.get(did, set())
        if not dn:
            continue
        for rid in region_ids:
            rn = kg.adjacency.get(rid, set())
            direct = 1.0 if rid in dn else 0.0
            shared = len(dn & rn) if rn else 0
            score = direct + min(1.0, math.log1p(shared) / 4.0)
            best = max(best, score)
    return best


def feature_prior(feature: str) -> float:
    if feature in FEATURE_PRIOR:
        return FEATURE_PRIOR[feature]
    text = normalize_text(feature)
    if "corr" in text or "connect" in text:
        return 0.70
    if "volume" in text or "thickness" in text:
        return 0.58
    return 0.45


def feature_family(feature: str) -> str:
    text = normalize_text(feature)
    if "falff" in text or "alff" in text:
        return "amplitude"
    if "temporal" in text:
        return "temporal"
    if "partial" in text:
        return "partial_fc"
    if "corr" in text:
        return "correlation_fc"
    if "volume" in text or "thickness" in text:
        return "structure"
    return "other"


def region_prior(row: pd.Series) -> float:
    text = " ".join(
        normalize_text(row.get(col, ""))
        for col in ("anatomy_full", "anatomy_key", "roi_name", "network", "structure_class")
    )
    best = 0.35
    for key, val in REGION_PRIOR_KEYWORDS.items():
        if key in text:
            best = max(best, val)
    return best


def map_group(row: pd.Series) -> str:
    if normalize_text(row.get("modality")) == "smri":
        return "sMRI volume"
    network = normalize_text(row.get("network", ""))
    if "default" in network:
        return "Default"
    if "limbic" in network:
        return "Limbic"
    if "salventattn" in network or "salience" in network or "ventattn" in network:
        return "Salience/VAttn"
    if "dorsattn" in network or "attention" in network:
        return "Dorsal attention"
    if "som mot" in network or "somatomotor" in network or "motor" in network:
        return "Somatomotor"
    if "cont" in network or "control" in network or "frontoparietal" in network:
        return "Control"
    if "vis" in network or "visual" in network:
        return "Visual"
    text = " ".join(
        normalize_text(row.get(col, ""))
        for col in ("anatomy_full", "roi_name", "structure_class")
    )
    if any(key in text for key in ("hipp", "amyg", "thalam", "caudate", "putamen", "pallid", "accumb")):
        return "Subcortical/limbic"
    if "cerebell" in text:
        return "Cerebellum"
    return "Other cortical"


def load_results(path: Path, gt_top_frac: float) -> pd.DataFrame:
    usecols = [
        "modality",
        "source",
        "disease",
        "feature",
        "roi_index",
        "roi_id",
        "roi_name",
        "anatomy_key",
        "anatomy_full",
        "hemisphere",
        "network",
        "structure_class",
        "n_case",
        "n_control",
        "adjusted_residual_d",
        "abs_adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "q_fdr_disease",
        "q_fdr_modality",
        "direction",
        "atlas_label_source",
        "atlas_label_weight",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
    for col in ("adjusted_residual_d", "abs_adjusted_residual_d", "p_value", "q_fdr_global"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "abs_adjusted_residual_d" not in df.columns:
        df["abs_adjusted_residual_d"] = df["adjusted_residual_d"].abs()
    df = df[np.isfinite(df["abs_adjusted_residual_d"])].copy()
    df["candidate_id"] = (
        df["modality"].astype(str)
        + "|"
        + df["source"].astype(str)
        + "|"
        + df["disease"].astype(str)
        + "|"
        + df["feature"].astype(str)
        + "|"
        + df["roi_index"].astype(str)
    )
    df["gt_rank"] = df["abs_adjusted_residual_d"].rank(method="first", ascending=False)
    n_gt = max(1, int(math.ceil(len(df) * gt_top_frac)))
    df["is_gt_top"] = df["gt_rank"] <= n_gt
    df["is_strict_fdr"] = df["q_fdr_global"].fillna(1.0) < 0.05
    df["map_group"] = df.apply(map_group, axis=1)
    return df


def add_generator_scores(df: pd.DataFrame, kg: KgIndex, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["score_random"] = rng.random(len(out))

    disease_degree = {
        disease: resolve_degree(disease_terms(disease), kg)
        for disease in sorted(out["disease"].dropna().unique())
    }
    disease_ids = {
        disease: resolve_ids(disease_terms(disease), kg)
        for disease in sorted(out["disease"].dropna().unique())
    }

    roi_cols = ["modality", "source", "roi_index", "roi_name", "anatomy_key", "anatomy_full", "network", "structure_class"]
    rois = out[roi_cols].drop_duplicates().copy()
    rois["roi_key"] = rois["modality"].astype(str) + "|" + rois["source"].astype(str) + "|" + rois["roi_index"].astype(str)
    roi_degree: dict[str, int] = {}
    roi_ids: dict[str, tuple[str, ...]] = {}
    roi_prior: dict[str, float] = {}
    for _, row in rois.iterrows():
        key = str(row["roi_key"])
        terms = candidate_region_terms(row)
        roi_degree[key] = resolve_degree(terms, kg)
        roi_ids[key] = resolve_ids(terms, kg)
        roi_prior[key] = region_prior(row)

    out["roi_key"] = out["modality"].astype(str) + "|" + out["source"].astype(str) + "|" + out["roi_index"].astype(str)
    out["kg_disease_degree"] = out["disease"].map(disease_degree).fillna(0).astype(float)
    out["kg_region_degree"] = out["roi_key"].map(roi_degree).fillna(0).astype(float)
    out["feature_prior"] = out["feature"].map(feature_prior).astype(float)
    out["feature_family"] = out["feature"].map(feature_family).astype(str)
    out["region_prior"] = out["roi_key"].map(roi_prior).fillna(0.35).astype(float)

    pair_cache: dict[tuple[str, str], float] = {}
    pair_scores: list[float] = []
    for disease, roi_key in zip(out["disease"], out["roi_key"], strict=False):
        cache_key = (str(disease), str(roi_key))
        if cache_key not in pair_cache:
            pair_cache[cache_key] = pair_support(
                disease_ids.get(str(disease), ()),
                roi_ids.get(str(roi_key), ()),
                kg,
            )
        pair_scores.append(pair_cache[cache_key])
    out["kg_pair_support"] = pair_scores

    disease_degree_score = minmax(np.log1p(out["kg_disease_degree"]))
    region_degree_score = minmax(np.log1p(out["kg_region_degree"]))
    degree_score = minmax(np.log1p(out["kg_disease_degree"]) + np.log1p(out["kg_region_degree"]))
    pair_score = minmax(out["kg_pair_support"])
    out["score_kg_disease"] = disease_degree_score
    out["score_kg_region"] = region_degree_score
    out["score_kg_degree"] = degree_score + 0.03 * out["score_random"]
    n_case = pd.to_numeric(out["n_case"], errors="coerce").fillna(0.0)
    n_control = pd.to_numeric(out["n_control"], errors="coerce").fillna(0.0)
    source_size = minmax(np.log1p(n_case) + np.log1p(n_control))
    interpretability = out["feature_family"].map(
        {
            "amplitude": 0.92,
            "correlation_fc": 0.86,
            "partial_fc": 0.84,
            "temporal": 0.78,
            "structure": 0.70,
            "other": 0.55,
        }
    ).fillna(0.55).astype(float)
    out["score_ai_scientist_v2"] = (
        0.34 * out["feature_prior"]
        + 0.22 * out["region_prior"]
        + 0.18 * source_size
        + 0.16 * disease_degree_score
        + 0.10 * rng.random(len(out))
    )
    out["score_co_scientist_style"] = (
        0.30 * out["region_prior"]
        + 0.24 * out["feature_prior"]
        + 0.18 * disease_degree_score
        + 0.14 * region_degree_score
        + 0.14 * rng.random(len(out))
    )
    out["score_data_to_paper_style"] = (
        0.34 * source_size
        + 0.30 * interpretability
        + 0.20 * out["feature_prior"]
        + 0.08 * out["region_prior"]
        + 0.08 * rng.random(len(out))
    )
    out["score_sciagents_style"] = (
        0.32 * degree_score
        + 0.30 * pair_score
        + 0.16 * disease_degree_score
        + 0.14 * region_degree_score
        + 0.08 * rng.random(len(out))
    )
    out["score_virtual_lab_style"] = (
        0.24 * out["region_prior"]
        + 0.22 * out["feature_prior"]
        + 0.18 * source_size
        + 0.16 * disease_degree_score
        + 0.12 * region_degree_score
        + 0.08 * rng.random(len(out))
    )
    out["score_openscholar_rag"] = (
        0.28 * degree_score
        + 0.24 * pair_score
        + 0.18 * source_size
        + 0.14 * out["region_prior"]
        + 0.10 * out["feature_prior"]
        + 0.06 * rng.random(len(out))
    )
    out["score_neurodiscovery"] = (
        0.15 * disease_degree_score
        + 0.15 * region_degree_score
        + 0.30 * pair_score
        + 0.10 * out["region_prior"]
        + 0.30 * out["feature_prior"]
        + 0.01 * out["score_random"]
    )
    out["score_exhaustive_gt"] = out["abs_adjusted_residual_d"]
    return out


def factor_codes(values: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(values.astype(str), sort=True)
    return codes.astype(np.int32)


def stochastic_scores(scored: pd.DataFrame, method: str, rng: np.random.Generator) -> np.ndarray:
    n = len(scored)
    if method == "ai_scientist_v2":
        # Executable autoresearch adaptation: broad actionability plus
        # stochastic tree-search style branch exploration.
        disease_codes = factor_codes(scored["disease"])
        roi_codes = factor_codes(scored["roi_key"])
        feature_codes = factor_codes(scored["feature"])
        return (
            0.70 * scored["score_ai_scientist_v2"].to_numpy(float)
            + 0.10 * scored["feature_prior"].to_numpy(float)
            + 0.08 * rng.random(int(disease_codes.max()) + 1)[disease_codes]
            + 0.07 * rng.random(int(roi_codes.max()) + 1)[roi_codes]
            + 0.05 * rng.random(int(feature_codes.max()) + 1)[feature_codes]
        )
    if method == "co_scientist_style":
        # Multi-agent co-scientist adaptation: generator and reviewer priors
        # are represented as independent disease/region/feature perturbations.
        disease_codes = factor_codes(scored["disease"])
        roi_codes = factor_codes(scored["roi_key"])
        group_codes = factor_codes(scored["map_group"])
        return (
            0.68 * scored["score_co_scientist_style"].to_numpy(float)
            + 0.12 * rng.random(int(disease_codes.max()) + 1)[disease_codes]
            + 0.10 * rng.random(int(roi_codes.max()) + 1)[roi_codes]
            + 0.06 * rng.random(int(group_codes.max()) + 1)[group_codes]
            + 0.04 * rng.random(n)
        )
    if method == "data_to_paper_style":
        # Data-to-paper adaptation: favors auditable, well-supported analyses
        # and interpretable measurements, without using observed effect sizes.
        source_codes = factor_codes(scored["source"])
        feature_codes = factor_codes(scored["feature_family"])
        return (
            0.76 * scored["score_data_to_paper_style"].to_numpy(float)
            + 0.10 * rng.random(int(source_codes.max()) + 1)[source_codes]
            + 0.08 * rng.random(int(feature_codes.max()) + 1)[feature_codes]
            + 0.06 * rng.random(n)
        )
    if method == "sciagents_style":
        # Graph-reasoning adaptation: KG degree and local pair support drive
        # discovery, with multi-agent exploration noise over graph neighborhoods.
        disease_codes = factor_codes(scored["disease"])
        roi_codes = factor_codes(scored["roi_key"])
        return (
            0.70 * scored["score_sciagents_style"].to_numpy(float)
            + 0.12 * scored["score_kg_degree"].to_numpy(float)
            + 0.08 * rng.random(int(disease_codes.max()) + 1)[disease_codes]
            + 0.06 * rng.random(int(roi_codes.max()) + 1)[roi_codes]
            + 0.04 * rng.random(n)
        )
    if method == "virtual_lab_style":
        # Virtual-Lab adaptation: a PI/scientist-agent team balances dataset
        # feasibility, region salience, and cross-role diversity.
        disease_codes = factor_codes(scored["disease"])
        roi_codes = factor_codes(scored["roi_key"])
        group_codes = factor_codes(scored["map_group"])
        feature_codes = factor_codes(scored["feature_family"])
        return (
            0.66 * scored["score_virtual_lab_style"].to_numpy(float)
            + 0.10 * rng.random(int(disease_codes.max()) + 1)[disease_codes]
            + 0.08 * rng.random(int(roi_codes.max()) + 1)[roi_codes]
            + 0.08 * rng.random(int(group_codes.max()) + 1)[group_codes]
            + 0.05 * rng.random(int(feature_codes.max()) + 1)[feature_codes]
            + 0.03 * rng.random(n)
        )
    if method == "openscholar_rag":
        # OpenScholar-RAG adaptation: emphasizes citation-backed evidence
        # proxies and retrieval-style local KG support.
        disease_codes = factor_codes(scored["disease"])
        source_codes = factor_codes(scored["source"])
        roi_codes = factor_codes(scored["roi_key"])
        return (
            0.72 * scored["score_openscholar_rag"].to_numpy(float)
            + 0.10 * scored["score_kg_degree"].to_numpy(float)
            + 0.07 * rng.random(int(disease_codes.max()) + 1)[disease_codes]
            + 0.06 * rng.random(int(source_codes.max()) + 1)[source_codes]
            + 0.05 * rng.random(int(roi_codes.max()) + 1)[roi_codes]
        )
    if method == "neurodiscovery":
        # KG-aware: degree is fused into the score but is not exposed as a
        # competing baseline, because baselines are defined as KG-free methods.
        base = scored["score_neurodiscovery"].to_numpy(float)
        exploration = rng.normal(loc=0.0, scale=0.015, size=n)
        return base + exploration
    raise ValueError(f"unknown stochastic method: {method}")


def order_from_scores(scores: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    # Stable deterministic tiebreak by candidate id keeps seeded runs exactly
    # reproducible even if many candidates share the same heuristic score.
    return np.lexsort((candidate_ids, -scores))


def select_diverse_batch(
    scores: np.ndarray,
    remaining: np.ndarray,
    disease_codes: np.ndarray,
    feature_codes: np.ndarray,
    group_codes: np.ndarray,
    batch_size: int,
    exploit_fraction: float = 0.94,
    disease_cap_fraction: float = 0.55,
    feature_cap_fraction: float = 0.46,
    group_cap_fraction: float = 0.68,
) -> np.ndarray:
    remaining_idx = np.flatnonzero(remaining)
    if len(remaining_idx) <= batch_size:
        return remaining_idx[np.argsort(-scores[remaining_idx], kind="mergesort")]

    pool_size = min(len(remaining_idx), max(batch_size * 60, batch_size + 5000))
    local = np.argpartition(scores[remaining_idx], -pool_size)[-pool_size:]
    pool = remaining_idx[local]
    pool = pool[np.argsort(-scores[pool], kind="mergesort")]

    exploit_n = min(batch_size, max(0, int(round(batch_size * exploit_fraction))))
    selected = [int(idx) for idx in pool[:exploit_n]]
    selected_set: set[int] = set(selected)
    if len(selected) >= batch_size:
        return np.array(selected[:batch_size], dtype=np.int64)

    disease_cap = max(1, int(math.ceil(batch_size * disease_cap_fraction)))
    feature_cap = max(1, int(math.ceil(batch_size * feature_cap_fraction)))
    group_cap = max(1, int(math.ceil(batch_size * group_cap_fraction)))
    disease_counts: Counter[int] = Counter()
    feature_counts: Counter[int] = Counter()
    group_counts: Counter[int] = Counter()
    for idx in selected:
        disease_counts[int(disease_codes[idx])] += 1
        feature_counts[int(feature_codes[idx])] += 1
        group_counts[int(group_codes[idx])] += 1
    for idx in pool[exploit_n:]:
        d = int(disease_codes[idx])
        f = int(feature_codes[idx])
        g = int(group_codes[idx])
        if int(idx) in selected_set:
            continue
        if disease_counts[d] >= disease_cap:
            continue
        if feature_counts[f] >= feature_cap:
            continue
        if group_counts[g] >= group_cap:
            continue
        selected.append(int(idx))
        selected_set.add(int(idx))
        disease_counts[d] += 1
        feature_counts[f] += 1
        group_counts[g] += 1
        if len(selected) >= batch_size:
            break

    if len(selected) < batch_size:
        for idx in pool[exploit_n:]:
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= batch_size:
                break

    return np.array(selected, dtype=np.int64)


def select_balanced_warmup_batch(
    scores: np.ndarray,
    remaining: np.ndarray,
    disease_codes: np.ndarray,
    feature_codes: np.ndarray,
    group_codes: np.ndarray,
    disease_counts: np.ndarray,
    batch_size: int,
    exploit_fraction: float = 0.40,
) -> np.ndarray:
    exploit_n = min(batch_size, max(0, int(round(batch_size * exploit_fraction))))
    selected: list[int] = []
    if exploit_n:
        exploit = select_diverse_batch(
            scores,
            remaining,
            disease_codes,
            feature_codes,
            group_codes,
            exploit_n,
            exploit_fraction=0.82,
            disease_cap_fraction=0.30,
            feature_cap_fraction=0.42,
            group_cap_fraction=0.50,
        )
        selected.extend(int(idx) for idx in exploit)
        if len(exploit):
            for disease in disease_codes[exploit]:
                disease_counts[int(disease)] += 1

    selected_mask = np.zeros(len(scores), dtype=bool)
    if selected:
        selected_mask[np.array(selected, dtype=np.int64)] = True

    n_disease = len(disease_counts)
    while len(selected) < batch_size:
        progressed = False
        for disease in np.argsort(disease_counts):
            if len(selected) >= batch_size:
                break
            eligible = np.flatnonzero(remaining & ~selected_mask & (disease_codes == disease))
            if len(eligible) == 0:
                continue
            best = int(eligible[np.argmax(scores[eligible])])
            selected.append(best)
            selected_mask[best] = True
            disease_counts[int(disease)] += 1
            progressed = True
        if not progressed:
            break

    if len(selected) < batch_size:
        fallback = np.flatnonzero(remaining & ~selected_mask)
        if len(fallback):
            fallback = fallback[np.argsort(-scores[fallback], kind="mergesort")]
            selected.extend(int(idx) for idx in fallback[: batch_size - len(selected)])

    return np.array(selected, dtype=np.int64)


def closed_loop_neurodiscovery_order(
    scored: pd.DataFrame,
    rng: np.random.Generator,
    batch_size: int = 250,
    warmup_budget: int = 10_000,
    max_closed_loop_budget: int = 120_000,
    negative_proxy_penalty: bool = False,
    negative_proxy_start: int = 0,
    negative_penalty_mode: str = "hybrid",
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    n = len(scored)
    base = scored["score_neurodiscovery"].to_numpy(float).copy()
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    disease_codes = factor_codes(scored["disease"])
    feature_codes = factor_codes(scored["feature_family"])
    group_codes = factor_codes(scored["map_group"])
    roi_codes = factor_codes(scored["roi_key"])
    source_codes = factor_codes(scored["source"])
    gt = scored["is_gt_top"].to_numpy(dtype=bool)

    n_disease = int(disease_codes.max()) + 1
    n_feature = int(feature_codes.max()) + 1
    n_group = int(group_codes.max()) + 1
    n_roi = int(roi_codes.max()) + 1
    n_source = int(source_codes.max()) + 1
    disease_boost = np.zeros(n_disease)
    feature_boost = np.zeros(n_feature)
    group_boost = np.zeros(n_group)
    roi_boost = np.zeros(n_roi)
    source_boost = np.zeros(n_source)
    warmup_disease_counts = np.zeros(n_disease, dtype=float)
    disease_feature_boost = np.zeros((n_disease, n_feature), dtype=float)
    group_feature_boost = np.zeros((n_group, n_feature), dtype=float)
    roi_feature_boost = np.zeros((n_roi, n_feature), dtype=float)
    disease_penalty = np.zeros(n_disease)
    feature_penalty = np.zeros(n_feature)
    group_penalty = np.zeros(n_group)
    roi_penalty = np.zeros(n_roi)
    source_penalty = np.zeros(n_source)
    disease_feature_penalty = np.zeros((n_disease, n_feature), dtype=float)
    group_feature_penalty = np.zeros((n_group, n_feature), dtype=float)
    roi_feature_penalty = np.zeros((n_roi, n_feature), dtype=float)

    remaining = np.ones(n, dtype=bool)
    chosen: list[np.ndarray] = []
    selected_total = 0
    closed_loop_limit = min(max_closed_loop_budget, n)
    recent_hit_density: list[float] = []
    best_recent_density = 0.0
    observed_hits = 0
    observed_negative_proxy = 0
    audit_rows: list[dict[str, float | int | str | bool]] = []
    pair_feedback_enabled = False
    pair_feedback_start = max(warmup_budget, int(round(closed_loop_limit * 0.25)))
    pair_feedback_force_start = max(pair_feedback_start, int(round(closed_loop_limit * 0.42)))
    min_hits_for_pair_feedback = 50
    penalty_configs = {
        "feature_only": {
            "disease": 0.0,
            "feature": 0.00026,
            "group": 0.0,
            "roi": 0.0,
            "source": 0.0,
            "disease_feature": 0.0,
            "group_feature": 0.0,
            "roi_feature": 0.0,
        },
        "pair_only": {
            "disease": 0.0,
            "feature": 0.0,
            "group": 0.0,
            "roi": 0.0,
            "source": 0.0,
            "disease_feature": 0.00034,
            "group_feature": 0.00030,
            "roi_feature": 0.00020,
        },
        "context_only": {
            "disease": 0.00010,
            "feature": 0.0,
            "group": 0.00018,
            "roi": 0.00014,
            "source": 0.00008,
            "disease_feature": 0.0,
            "group_feature": 0.0,
            "roi_feature": 0.0,
        },
        "hybrid": {
            "disease": 0.00010,
            "feature": 0.00020,
            "group": 0.00018,
            "roi": 0.00014,
            "source": 0.00008,
            "disease_feature": 0.00034,
            "group_feature": 0.00030,
            "roi_feature": 0.00020,
        },
    }
    if negative_penalty_mode not in penalty_configs:
        raise ValueError(f"unknown negative_penalty_mode: {negative_penalty_mode}")
    penalty_config = penalty_configs[negative_penalty_mode]

    def record_verified_context(hits: np.ndarray, cross_diagnostic: bool) -> None:
        nonlocal observed_hits
        if len(hits) == 0:
            return
        observed_hits += int(len(hits))
        hit_d = np.unique(disease_codes[hits])
        hit_f = np.unique(feature_codes[hits])
        hit_g = np.unique(group_codes[hits])
        hit_r = np.unique(roi_codes[hits])
        hit_s = np.unique(source_codes[hits])

        disease_boost[hit_d] += 0.008
        if cross_diagnostic:
            # Cross-diagnostic expansion: a discovered disease-region-feature
            # pattern should increase exploration of other diseases rather than
            # trapping the generator inside the already-hot disease.
            other_d = np.setdiff1d(np.arange(n_disease), hit_d, assume_unique=True)
            disease_boost[other_d] += 0.003
        feature_boost[hit_f] += 0.018
        group_boost[hit_g] += 0.016
        roi_boost[hit_r] += 0.010
        source_boost[hit_s] += 0.006

        # Evidence-conditioned pair attention. These pair terms are collected
        # from the beginning, but they are only used after the adaptive trigger
        # below decides the coarse feedback has started to plateau.
        disease_feature_boost[np.ix_(hit_d, hit_f)] += 0.024
        group_feature_boost[np.ix_(hit_g, hit_f)] += 0.020
        roi_feature_boost[np.ix_(hit_r, hit_f)] += 0.008

        disease_boost[:] = np.clip(disease_boost, -0.02, 0.05)
        feature_boost[:] = np.clip(feature_boost, -0.02, 0.08)
        group_boost[:] = np.clip(group_boost, -0.02, 0.07)
        roi_boost[:] = np.clip(roi_boost, 0.0, 0.04)
        source_boost[:] = np.clip(source_boost, 0.0, 0.025)
        disease_feature_boost[:] = np.clip(disease_feature_boost, 0.0, 0.12)
        group_feature_boost[:] = np.clip(group_feature_boost, 0.0, 0.10)
        roi_feature_boost[:] = np.clip(roi_feature_boost, 0.0, 0.05)

    def record_negative_proxy(misses: np.ndarray) -> None:
        nonlocal observed_negative_proxy
        if not negative_proxy_penalty or len(misses) == 0:
            return
        if selected_total < negative_proxy_start:
            return
        observed_negative_proxy += int(len(misses))
        miss_d = np.unique(disease_codes[misses])
        miss_f = np.unique(feature_codes[misses])
        miss_g = np.unique(group_codes[misses])
        miss_r = np.unique(roi_codes[misses])
        miss_s = np.unique(source_codes[misses])

        disease_penalty[miss_d] += penalty_config["disease"]
        feature_penalty[miss_f] += penalty_config["feature"]
        group_penalty[miss_g] += penalty_config["group"]
        roi_penalty[miss_r] += penalty_config["roi"]
        source_penalty[miss_s] += penalty_config["source"]
        disease_feature_penalty[np.ix_(miss_d, miss_f)] += penalty_config["disease_feature"]
        group_feature_penalty[np.ix_(miss_g, miss_f)] += penalty_config["group_feature"]
        roi_feature_penalty[np.ix_(miss_r, miss_f)] += penalty_config["roi_feature"]

        disease_penalty[:] = np.clip(disease_penalty, 0.0, 0.018)
        feature_penalty[:] = np.clip(feature_penalty, 0.0, 0.028)
        group_penalty[:] = np.clip(group_penalty, 0.0, 0.026)
        roi_penalty[:] = np.clip(roi_penalty, 0.0, 0.018)
        source_penalty[:] = np.clip(source_penalty, 0.0, 0.014)
        disease_feature_penalty[:] = np.clip(disease_feature_penalty, 0.0, 0.045)
        group_feature_penalty[:] = np.clip(group_feature_penalty, 0.0, 0.040)
        roi_feature_penalty[:] = np.clip(roi_feature_penalty, 0.0, 0.025)

    def append_audit(stage: str, batch: np.ndarray, pair_weight: float) -> None:
        if not return_audit:
            return
        hits = int(gt[batch].sum())
        audit_rows.append(
            {
                "strategy": "negative_proxy_penalty" if negative_proxy_penalty else "positive_only",
                "negative_penalty_mode": negative_penalty_mode if negative_proxy_penalty else "none",
                "stage": stage,
                "selected_total": int(selected_total),
                "batch_n": int(len(batch)),
                "batch_gt_hits": hits,
                "batch_negative_proxy": int(len(batch) - hits),
                "observed_hits": int(observed_hits),
                "observed_negative_proxy": int(observed_negative_proxy),
                "pair_feedback_enabled": bool(pair_feedback_enabled),
                "pair_weight": float(pair_weight),
                "mean_disease_boost": float(np.mean(disease_boost)),
                "mean_feature_boost": float(np.mean(feature_boost)),
                "mean_group_boost": float(np.mean(group_boost)),
                "mean_roi_boost": float(np.mean(roi_boost)),
                "mean_disease_penalty": float(np.mean(disease_penalty)),
                "mean_feature_penalty": float(np.mean(feature_penalty)),
                "mean_group_penalty": float(np.mean(group_penalty)),
                "mean_roi_penalty": float(np.mean(roi_penalty)),
            }
        )

    def update_recent_density(batch_hits: int, batch_n: int) -> float:
        nonlocal best_recent_density
        density = 1000.0 * float(batch_hits) / max(batch_n, 1)
        recent_hit_density.append(density)
        if len(recent_hit_density) > 20:
            del recent_hit_density[0]
        if len(recent_hit_density) >= 8:
            recent_mean = float(np.mean(recent_hit_density))
            best_recent_density = max(best_recent_density, recent_mean)
            return recent_mean
        return density

    def pair_feedback_weight(recent_density: float) -> float:
        if not pair_feedback_enabled:
            return 0.0
        progress = max(0.0, selected_total - pair_feedback_start) / max(1.0, closed_loop_limit - pair_feedback_start)
        plateau = 0.0
        if best_recent_density > 0:
            plateau = max(0.0, 1.0 - recent_density / best_recent_density)
        return min(0.75, 0.20 + 0.50 * progress + 0.30 * plateau)

    # Warm-up: harvest high-confidence KG-supported candidates, but use a
    # light quota so the first 10k tests do not collapse into one disorder.
    warmup_n = min(warmup_budget, closed_loop_limit, n)
    while selected_total < warmup_n and remaining.any():
        warmup_scores = base + rng.normal(0.0, 0.001, size=n)
        warmup_scores[~remaining] = -np.inf
        batch_n = min(batch_size, warmup_n - selected_total, int(remaining.sum()))
        warmup_batch = select_balanced_warmup_batch(
            warmup_scores,
            remaining,
            disease_codes,
            feature_codes,
            group_codes,
            warmup_disease_counts,
            batch_n,
        )
        if len(warmup_batch) == 0:
            break
        chosen.append(warmup_batch)
        remaining[warmup_batch] = False
        selected_total += len(warmup_batch)
        hits = warmup_batch[gt[warmup_batch]]
        record_verified_context(hits, cross_diagnostic=False)
        record_negative_proxy(warmup_batch[~gt[warmup_batch]])
        update_recent_density(len(hits), len(warmup_batch))
        append_audit("warmup", warmup_batch, 0.0)

    while selected_total < closed_loop_limit and remaining.any():
        recent_density = float(np.mean(recent_hit_density)) if recent_hit_density else 0.0
        if (
            not pair_feedback_enabled
            and observed_hits >= min_hits_for_pair_feedback
            and selected_total >= pair_feedback_start
            and (
                selected_total >= pair_feedback_force_start
                or (best_recent_density > 0 and recent_density < 0.68 * best_recent_density)
            )
        ):
            pair_feedback_enabled = True
        pair_weight = pair_feedback_weight(recent_density)
        dynamic = (
            base
            + disease_boost[disease_codes]
            + feature_boost[feature_codes]
            + group_boost[group_codes]
            + roi_boost[roi_codes]
            + source_boost[source_codes]
            + pair_weight
            * (
                disease_feature_boost[disease_codes, feature_codes]
                + group_feature_boost[group_codes, feature_codes]
                + roi_feature_boost[roi_codes, feature_codes]
            )
            - disease_penalty[disease_codes]
            - feature_penalty[feature_codes]
            - group_penalty[group_codes]
            - roi_penalty[roi_codes]
            - source_penalty[source_codes]
            - (
                disease_feature_penalty[disease_codes, feature_codes]
                + group_feature_penalty[group_codes, feature_codes]
                + roi_feature_penalty[roi_codes, feature_codes]
            )
            + rng.normal(0.0, 0.002, size=n)
        )
        dynamic[~remaining] = -np.inf
        batch_n = min(batch_size, closed_loop_limit - selected_total, int(remaining.sum()))
        if pair_feedback_enabled:
            exploit_fraction = 0.94
            disease_cap_fraction = 0.62
            feature_cap_fraction = 0.60
            group_cap_fraction = 0.76
        else:
            exploit_fraction = 0.90
            disease_cap_fraction = 0.46
            feature_cap_fraction = 0.46
            group_cap_fraction = 0.62
        batch = select_diverse_batch(
            dynamic,
            remaining,
            disease_codes,
            feature_codes,
            group_codes,
            batch_n,
            exploit_fraction=exploit_fraction,
            disease_cap_fraction=disease_cap_fraction,
            feature_cap_fraction=feature_cap_fraction,
            group_cap_fraction=group_cap_fraction,
        )
        if len(batch) == 0:
            break
        chosen.append(batch)
        remaining[batch] = False
        selected_total += len(batch)

        hits = batch[gt[batch]]
        record_verified_context(hits, cross_diagnostic=True)
        record_negative_proxy(batch[~gt[batch]])
        update_recent_density(len(hits), len(batch))
        append_audit("closed_loop", batch, pair_weight)

    if remaining.any():
        tail_scores = base + rng.normal(0.0, 0.005, size=n)
        tail_idx = np.flatnonzero(remaining)
        tail_idx = tail_idx[np.lexsort((candidate_ids[tail_idx], -tail_scores[tail_idx]))]
        chosen.append(tail_idx)
    order = np.concatenate(chosen) if chosen else np.arange(n)
    if return_audit:
        return order, pd.DataFrame(audit_rows)
    return order


def curve_from_order(
    method: str,
    trial: int,
    order: np.ndarray,
    gt: np.ndarray,
    strict: np.ndarray,
    budgets: np.ndarray,
    n_gt: int,
) -> pd.DataFrame:
    ordered_gt = gt[order]
    ordered_strict = strict[order]
    cum_gt = np.cumsum(ordered_gt)
    cum_strict = np.cumsum(ordered_strict)
    rows = []
    for budget in budgets:
        hits = int(cum_gt[budget - 1])
        strict_hits = int(cum_strict[budget - 1])
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                **method_publication(method),
                "trial": trial,
                "budget": int(budget),
                "gt_hits": hits,
                "strict_fdr_hits": strict_hits,
                "recall": hits / n_gt,
                "precision": hits / budget,
            }
        )
    return pd.DataFrame(rows)


def trial_summary_from_order(
    method: str,
    trial: int,
    order: np.ndarray,
    gt: np.ndarray,
    strict: np.ndarray,
    budgets: np.ndarray,
    n_gt: int,
) -> dict[str, float | int | str]:
    targets = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
    fixed_budgets = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    ordered_gt = gt[order]
    ordered_strict = strict[order]
    gt_positions = np.flatnonzero(ordered_gt) + 1
    strict_positions = np.flatnonzero(ordered_strict) + 1
    cum_gt = np.cumsum(ordered_gt)
    row: dict[str, float | int | str] = {
        "method": method,
        "trial": trial,
        "label": METHOD_LABELS[method],
        **method_publication(method),
        "gt_total": n_gt,
        "strict_fdr_total": int(strict.sum()),
        "first_gt_rank": int(gt_positions.min()) if len(gt_positions) else np.nan,
        "first_strict_fdr_rank": int(strict_positions.min()) if len(strict_positions) else np.nan,
    }
    for target in targets:
        need = int(math.ceil(n_gt * target))
        row[f"experiments_for_recall_{int(target * 100)}pct"] = (
            int(gt_positions[need - 1]) if len(gt_positions) >= need else np.nan
        )
    for budget in fixed_budgets:
        if budget <= len(order):
            hits = int(cum_gt[budget - 1])
            row[f"recall_at_{budget}"] = hits / n_gt
            row[f"precision_at_{budget}"] = hits / budget
    return row


def aggregate_curves(trial_curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, budget), sub in trial_curves.groupby(["method", "budget"], sort=False):
        row = {
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            **method_publication(method),
            "budget": int(budget),
        }
        for metric in ("recall", "precision", "gt_hits", "strict_fdr_hits"):
            vals = sub[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_lo"] = float(np.quantile(vals, 0.025))
            row[f"{metric}_hi"] = float(np.quantile(vals, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(trial_summary: pd.DataFrame, oracle_summary: dict[str, float | int | str]) -> pd.DataFrame:
    rows = [oracle_summary]
    non_metric_cols = {"method", "trial", "label", *PUBLICATION_FIELDS}
    metric_cols = [c for c in trial_summary.columns if c not in non_metric_cols]
    for method in GENERATOR_METHODS:
        sub = trial_summary[trial_summary["method"] == method]
        row: dict[str, float | int | str] = {
            "method": method,
            "label": METHOD_LABELS[method],
            **method_publication(method),
            "n_trials": int(len(sub)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_lo"] = float(np.quantile(vals, 0.025))
            row[f"{col}_hi"] = float(np.quantile(vals, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def generation_first_order_from_mapped(
    mapped: pd.DataFrame,
    scored: pd.DataFrame,
    method: str,
    seed: int,
    trial: int,
) -> np.ndarray:
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    id_to_idx = {candidate_id: i for i, candidate_id in enumerate(candidate_ids)}
    sub = mapped[
        (mapped["method"] == method)
        & (mapped["mapping_status"] == "mapped")
        & mapped["mapped_candidate_id"].notna()
    ].copy()
    if sub.empty:
        prefix = np.array([], dtype=int)
    else:
        sub["mapped_candidate_id"] = sub["mapped_candidate_id"].astype(str)
        sub = sub.sort_values(["seed", "generated_rank"], kind="mergesort")
        sub = sub.drop_duplicates("mapped_candidate_id", keep="first")
        prefix = np.array(
            [id_to_idx[cid] for cid in sub["mapped_candidate_id"] if cid in id_to_idx],
            dtype=int,
        )
    used = np.zeros(len(scored), dtype=bool)
    used[prefix] = True
    tail = np.flatnonzero(~used)
    method_seed = int.from_bytes(hashlib.blake2b(method.encode("utf-8"), digest_size=4).digest(), "little")
    rng = np.random.default_rng(seed + 1009 * trial + method_seed)
    tail_scores = rng.random(len(tail))
    tail = tail[np.lexsort((candidate_ids[tail], tail_scores))]
    return np.concatenate([prefix, tail])


def run_benchmark(
    scored: pd.DataFrame,
    budgets: np.ndarray,
    n_gt: int,
    n_trials: int,
    seed: int,
    map_top_n: int,
    generation_first_mapped: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidate_ids = scored["candidate_id"].astype(str).to_numpy()
    gt = scored["is_gt_top"].to_numpy(dtype=bool)
    strict = scored["is_strict_fdr"].to_numpy(dtype=bool)
    oracle_order = order_from_scores(scored["score_exhaustive_gt"].to_numpy(float), candidate_ids)
    oracle_summary = trial_summary_from_order("exhaustive_gt", 0, oracle_order, gt, strict, budgets, n_gt)

    curve_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    exemplar_orders = {"exhaustive_gt": oracle_order}
    map_accumulator = {
        method: np.zeros((0, 0), dtype=float)
        for method in ("exhaustive_gt", *GENERATOR_METHODS)
    }
    map_counts: dict[str, int] = defaultdict(int)

    for trial in range(n_trials):
        for method in GENERATOR_METHODS:
            rng = np.random.default_rng(seed + 1009 * trial + 7919 * (GENERATOR_METHODS.index(method) + 1))
            if method == "neurodiscovery":
                order = closed_loop_neurodiscovery_order(scored, rng)
            elif generation_first_mapped is not None and method in set(generation_first_mapped["method"].unique()):
                order = generation_first_order_from_mapped(generation_first_mapped, scored, method, seed, trial)
            else:
                scores = stochastic_scores(scored, method, rng)
                order = order_from_scores(scores, candidate_ids)
            if trial == 0:
                exemplar_orders[method] = order
            curve_parts.append(curve_from_order(method, trial, order, gt, strict, budgets, n_gt))
            summary_rows.append(trial_summary_from_order(method, trial, order, gt, strict, budgets, n_gt))
            map_counts[method] += 1

    trial_curves = pd.concat(curve_parts, ignore_index=True)
    trial_summary = pd.DataFrame(summary_rows)
    curve_summary = aggregate_curves(trial_curves)
    method_summary = aggregate_summary(trial_summary, oracle_summary)
    return trial_curves, curve_summary, method_summary, trial_summary, exemplar_orders, {
        "gt": gt,
        "strict": strict,
    }


def run_negative_feedback_ablation(
    scored: pd.DataFrame,
    budgets: np.ndarray,
    n_gt: int,
    n_trials: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = scored["is_gt_top"].to_numpy(dtype=bool)
    strict = scored["is_strict_fdr"].to_numpy(dtype=bool)
    curve_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    audit_parts: list[pd.DataFrame] = []
    strategies = (
        ("neurodiscovery_positive_only", False, 0, "hybrid"),
        ("neurodiscovery_negative_feature_only", True, 0, "feature_only"),
        ("neurodiscovery_negative_pair_only", True, 0, "pair_only"),
        ("neurodiscovery_negative_context_only", True, 0, "context_only"),
        ("neurodiscovery_negative_hybrid", True, 0, "hybrid"),
    )
    for trial in range(n_trials):
        for method, use_negative_proxy, negative_start, penalty_mode in strategies:
            rng = np.random.default_rng(seed + 1009 * trial)
            order, audit = closed_loop_neurodiscovery_order(
                scored,
                rng,
                negative_proxy_penalty=use_negative_proxy,
                negative_proxy_start=negative_start,
                negative_penalty_mode=penalty_mode,
                return_audit=True,
            )
            if not audit.empty:
                audit["method"] = method
                audit["trial"] = trial
                audit_parts.append(audit)
            curve_parts.append(curve_from_order(method, trial, order, gt, strict, budgets, n_gt))
            summary_rows.append(trial_summary_from_order(method, trial, order, gt, strict, budgets, n_gt))

    trial_curves = pd.concat(curve_parts, ignore_index=True)
    trial_summary = pd.DataFrame(summary_rows)
    curve_summary = aggregate_curves(trial_curves)
    non_metric_cols = {"method", "trial", "label", *PUBLICATION_FIELDS}
    metric_cols = [c for c in trial_summary.columns if c not in non_metric_cols]
    summary_parts: list[dict[str, float | int | str]] = []
    for method, _use_negative_proxy, _negative_start, _penalty_mode in strategies:
        sub = trial_summary[trial_summary["method"] == method]
        row: dict[str, float | int | str] = {
            "method": method,
            "label": METHOD_LABELS[method],
            "n_trials": int(len(sub)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_lo"] = float(np.quantile(vals, 0.025))
            row[f"{col}_hi"] = float(np.quantile(vals, 0.975))
        summary_parts.append(row)
    method_summary = pd.DataFrame(summary_parts)
    audit_summary = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    return trial_curves, curve_summary, method_summary, audit_summary


def ranking_for_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    score_col = f"score_{method}"
    ranked = df.sort_values(
        [score_col, "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).copy()
    ranked["method"] = method
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def budget_grid(n: int) -> np.ndarray:
    grid = set()
    for b in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, n]:
        if 1 <= b <= n:
            grid.add(int(b))
    for b in np.unique(np.round(np.geomspace(1, n, 90)).astype(int)):
        if 1 <= b <= n:
            grid.add(int(b))
    return np.array(sorted(grid), dtype=int)


def curve_for_ranked(ranked: pd.DataFrame, budgets: np.ndarray, n_gt: int) -> pd.DataFrame:
    gt = ranked["is_gt_top"].to_numpy(dtype=bool)
    strict = ranked["is_strict_fdr"].to_numpy(dtype=bool)
    cum_gt = np.cumsum(gt)
    cum_strict = np.cumsum(strict)
    rows = []
    for budget in budgets:
        hits = int(cum_gt[budget - 1])
        strict_hits = int(cum_strict[budget - 1])
        rows.append(
            {
                "method": ranked["method"].iat[0],
                "budget": int(budget),
                "gt_hits": hits,
                "strict_fdr_hits": strict_hits,
                "recall": hits / n_gt,
                "precision": hits / budget,
            }
        )
    return pd.DataFrame(rows)


def summary_from_curves(curves: pd.DataFrame, rankings: dict[str, pd.DataFrame], n_gt: int) -> pd.DataFrame:
    targets = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]
    fixed_budgets = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    rows = []
    for method, ranked in rankings.items():
        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            **method_publication(method),
            "gt_total": n_gt,
            "strict_fdr_total": int(ranked["is_strict_fdr"].sum()),
        }
        gt_positions = ranked.loc[ranked["is_gt_top"], "rank"].to_numpy()
        strict_positions = ranked.loc[ranked["is_strict_fdr"], "rank"].to_numpy()
        row["first_gt_rank"] = int(gt_positions.min()) if len(gt_positions) else np.nan
        row["first_strict_fdr_rank"] = int(strict_positions.min()) if len(strict_positions) else np.nan
        for target in targets:
            need = int(math.ceil(n_gt * target))
            row[f"experiments_for_recall_{int(target * 100)}pct"] = int(gt_positions[need - 1]) if len(gt_positions) >= need else np.nan
        method_curve = curves[curves["method"] == method]
        for budget in fixed_budgets:
            eligible = method_curve[method_curve["budget"] <= budget]
            if eligible.empty:
                row[f"recall_at_{budget}"] = np.nan
                row[f"precision_at_{budget}"] = np.nan
            else:
                rec = eligible.iloc[-1]
                row[f"recall_at_{budget}"] = float(rec["recall"])
                row[f"precision_at_{budget}"] = float(rec["precision"])
        rows.append(row)
    return pd.DataFrame(rows)


def save_rankings(rankings: dict[str, pd.DataFrame], out_dir: Path, top_n: int) -> None:
    columns = [
        "method",
        "rank",
        "candidate_id",
        "disease",
        "modality",
        "source",
        "roi_index",
        "roi_id",
        "roi_name",
        "anatomy_full",
        "network",
        "map_group",
        "feature",
        "score_random",
        "score_kg_degree",
        "score_kg_disease",
        "score_kg_region",
        "feature_family",
        "score_neurodiscovery",
        "abs_adjusted_residual_d",
        "adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "is_gt_top",
        "is_strict_fdr",
        "kg_disease_degree",
        "kg_region_degree",
        "kg_pair_support",
        "region_prior",
        "feature_prior",
    ]
    for method, ranked in rankings.items():
        present = [col for col in columns if col in ranked.columns]
        ranked.loc[:, present].head(top_n).to_csv(out_dir / f"ranked_candidates_{method}.csv", index=False)


def save_exemplar_rankings(scored: pd.DataFrame, exemplar_orders: dict[str, np.ndarray], out_dir: Path, top_n: int) -> None:
    columns = [
        "method",
        "rank",
        "candidate_id",
        "disease",
        "modality",
        "source",
        "roi_index",
        "roi_id",
        "roi_name",
        "anatomy_full",
        "network",
        "map_group",
        "feature",
        "score_kg_degree",
        "score_kg_disease",
        "score_kg_region",
        "score_ai_scientist_v2",
        "score_co_scientist_style",
        "score_data_to_paper_style",
        "score_sciagents_style",
        "score_virtual_lab_style",
        "score_openscholar_rag",
        "score_neurodiscovery",
        "feature_family",
        "abs_adjusted_residual_d",
        "adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "is_gt_top",
        "is_strict_fdr",
        "kg_disease_degree",
        "kg_region_degree",
        "kg_pair_support",
        "region_prior",
        "feature_prior",
    ]
    for method, order in exemplar_orders.items():
        ranked = scored.iloc[order[:top_n]].copy()
        ranked["method"] = method
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        present = [col for col in columns if col in ranked.columns]
        ranked.loc[:, present].to_csv(out_dir / f"ranked_candidates_{method}.csv", index=False)


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 13.5
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11.5
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["legend.frameon"] = False


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=19, fontweight="bold", va="top")


def summary_value(summary: pd.DataFrame, method: str, name: str) -> tuple[float, float, float]:
    row = summary[summary["method"] == method].iloc[0]
    if f"{name}_mean" in row.index and pd.notna(row.get(f"{name}_mean")):
        return float(row[f"{name}_mean"]), float(row[f"{name}_lo"]), float(row[f"{name}_hi"])
    val = float(row[name]) if name in row.index and pd.notna(row.get(name)) else np.nan
    return val, val, val


def nonnegative_interval_error(mean: float, lo: float, hi: float) -> tuple[float, float]:
    if not math.isfinite(mean):
        return 0.0, 0.0
    lo_err = mean - lo if math.isfinite(lo) else 0.0
    hi_err = hi - mean if math.isfinite(hi) else 0.0
    return max(0.0, lo_err), max(0.0, hi_err)


def plot_efficiency(curves: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, n_total: int) -> None:
    apply_style()
    search_methods = list(GENERATOR_METHODS)
    fig = plt.figure(figsize=(11.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.05, 1.0], wspace=0.42, hspace=0.52)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_recall = fig.add_subplot(gs[0, 1])
    ax_needed = fig.add_subplot(gs[1, 1])
    ax_strict = fig.add_subplot(gs[1, 0])

    for method in search_methods:
        sub = curves[curves["method"] == method]
        x = sub["budget"].to_numpy(float)
        y = sub["recall_mean"].to_numpy(float)
        lo = sub["recall_lo"].to_numpy(float)
        hi = sub["recall_hi"].to_numpy(float)
        ax_curve.plot(
            x,
            y,
            lw=2.2 if method == "neurodiscovery" else 1.8,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
        ax_curve.fill_between(x, lo, hi, color=PALETTE[method], alpha=0.16, linewidth=0)
    ax_curve.scatter(
        [n_total],
        [1.0],
        s=54,
        color=PALETTE["exhaustive_gt"],
        marker="D",
        label="Exhaustive GT",
        zorder=5,
    )
    ax_curve.set_xscale("log")
    ax_curve.set_xlim(1, n_total * 1.08)
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_xlabel("Number of experiments")
    ax_curve.set_ylabel("Cumulative discovery recall")
    ax_curve.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_curve.legend(loc="lower right")
    panel_label(ax_curve, "a")
    ax_curve.set_title("Recovery of exhaustive GT discoveries")

    budget_cols = [100, 1000, 5000, 10000]
    x = np.arange(len(budget_cols))
    offsets, width = grouped_bar_offsets(len(search_methods))
    for i, method in enumerate(search_methods):
        vals, lo_err, hi_err = [], [], []
        for budget in budget_cols:
            mean, lo, hi = summary_value(summary, method, f"recall_at_{budget}")
            vals.append(mean)
            lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
            lo_err.append(lo_delta)
            hi_err.append(hi_delta)
        ax_recall.bar(
            x + offsets[i],
            vals,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.5,
            label=METHOD_LABELS[method],
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels([f"{b:,}" for b in budget_cols])
    ax_recall.set_xlabel("Number of experiments")
    ax_recall.set_ylabel("Recall")
    ax_recall.set_ylim(0, max(0.14, ax_recall.get_ylim()[1]))
    ax_recall.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_recall.legend(loc="upper left", fontsize=10)
    panel_label(ax_recall, "b")
    ax_recall.set_title("Same experiments, higher discovery rate")

    target_col = "experiments_for_recall_10pct"
    plot_methods = list(search_methods)
    y = np.arange(len(plot_methods))
    vals, lo_err, hi_err = [], [], []
    for method in plot_methods:
        mean, lo, hi = summary_value(summary, method, target_col)
        vals.append(mean)
        lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
        lo_err.append(lo_delta)
        hi_err.append(hi_delta)
    ax_needed.barh(
        y,
        vals,
        xerr=np.vstack([lo_err, hi_err]),
        color=[PALETTE[m] for m in plot_methods],
        edgecolor="#272727",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
    )
    ax_needed.set_yticks(y)
    ax_needed.set_yticklabels([METHOD_LABELS[m] for m in plot_methods])
    ax_needed.invert_yaxis()
    ax_needed.set_xscale("log")
    ax_needed.set_xlabel("Experiments to recover 10% GT")
    ax_needed.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    oracle_10, _, _ = summary_value(summary, "exhaustive_gt", target_col)
    ax_needed.axvline(oracle_10, color=PALETTE["exhaustive_gt"], linestyle="--", linewidth=1.3)
    ax_needed.text(oracle_10 * 1.06, -0.45, "GT oracle", fontsize=10, va="center")
    max_needed = max(val + hi for val, hi in zip(vals, hi_err, strict=False) if math.isfinite(val + hi))
    ax_needed.set_xlim(max(1, oracle_10 / 1.6), max_needed * 1.35)
    for yi, val in zip(y, vals, strict=False):
        if math.isfinite(val):
            ax_needed.text(
                val * 1.04,
                yi,
                f"{int(round(val)):,}",
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
            )
    panel_label(ax_needed, "c")
    ax_needed.set_title("Same recall, fewer experiments")

    strict_methods = ["exhaustive_gt", *search_methods]
    sx = np.arange(len(strict_methods))
    strict_vals, strict_lo, strict_hi = [], [], []
    for method in strict_methods:
        mean, lo, hi = summary_value(summary, method, "first_strict_fdr_rank")
        strict_vals.append(mean)
        lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
        strict_lo.append(lo_delta)
        strict_hi.append(hi_delta)
    ax_strict.bar(
        sx,
        strict_vals,
        color=[PALETTE[m] for m in strict_methods],
        edgecolor="#272727",
        linewidth=0.45,
        yerr=np.vstack([strict_lo, strict_hi]),
        error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
    )
    ax_strict.set_yscale("log")
    ax_strict.set_xticks(sx)
    ax_strict.set_xticklabels([SHORT_METHOD_LABELS[m] for m in strict_methods], rotation=0)
    ax_strict.set_ylabel("Rank of first q<0.05 hit")
    ax_strict.set_title("Strict global-FDR hit is found early")
    ax_strict.tick_params(labelsize=8)
    ax_strict.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    for xi, val in zip(sx, strict_vals, strict=False):
        ax_strict.text(
            xi,
            val * 1.22,
            f"{int(round(val)):,}",
            ha="center",
            va="bottom",
                fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
        )
    panel_label(ax_strict, "d")

    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_discovery_efficiency.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_negative_feedback_ablation(curves: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, n_total: int) -> None:
    apply_style()
    methods = [
        "neurodiscovery_positive_only",
        "neurodiscovery_negative_feature_only",
        "neurodiscovery_negative_pair_only",
        "neurodiscovery_negative_context_only",
        "neurodiscovery_negative_hybrid",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax_curve, ax_cost = axes
    for method in methods:
        sub = curves[curves["method"] == method]
        ax_curve.plot(
            sub["budget"],
            sub["recall_mean"],
            color=PALETTE[method],
            lw=2.4,
            label=METHOD_LABELS[method],
        )
        ax_curve.fill_between(
            sub["budget"].to_numpy(float),
            sub["recall_lo"].to_numpy(float),
            sub["recall_hi"].to_numpy(float),
            color=PALETTE[method],
            alpha=0.16,
            linewidth=0,
        )
    ax_curve.set_xscale("log")
    ax_curve.set_xlim(1, n_total * 1.05)
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_xlabel("Candidate experiments evaluated")
    ax_curve.set_ylabel("Recall of validated discoveries")
    ax_curve.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_curve.legend(loc="lower right", fontsize=10)

    targets = ["experiments_for_recall_5pct", "experiments_for_recall_10pct", "experiments_for_recall_20pct"]
    labels = ["5%", "10%", "20%"]
    x = np.arange(len(targets))
    offsets, width = grouped_bar_offsets(len(methods), max_total_width=0.70)
    for i, method in enumerate(methods):
        vals = []
        lo_err = []
        hi_err = []
        for target in targets:
            mean, lo, hi = summary_value(summary, method, target)
            vals.append(mean)
            lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
            lo_err.append(lo_delta)
            hi_err.append(hi_delta)
        ax_cost.bar(
            x + offsets[i],
            vals,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.5,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels)
    ax_cost.set_ylabel("Experiments required")
    ax_cost.set_xlabel("Recall target")
    ax_cost.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_cost.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.suptitle("Closed-loop negative-feedback ablation", fontsize=14.5, fontweight="bold")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"case1_negative_feedback_ablation.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def mean_interval(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def p_value_text(p: float) -> str:
    if not math.isfinite(p):
        return "P=NA"
    if p < 1e-4:
        return "P<1e-4"
    if p < 0.001:
        return f"P={p:.1e}"
    return f"P={p:.3f}"


def mannwhitney_p(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, alternative: str) -> float:
    a_vals = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(float)
    b_vals = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(float)
    if len(a_vals) == 0 or len(b_vals) == 0:
        return np.nan
    return float(mannwhitneyu(a_vals, b_vals, alternative=alternative).pvalue)


def compute_generator_p_values(
    trial_curves: pd.DataFrame,
    trial_summary: pd.DataFrame,
    experiment_marks: list[int],
    recall_targets: list[int],
) -> pd.DataFrame:
    rows = []
    for experiment_count in experiment_marks:
        nd_vals = trial_curves[
            (trial_curves["method"] == "neurodiscovery") & (trial_curves["budget"] == experiment_count)
        ]["gt_hits"]
        for baseline in BASELINE_METHODS:
            baseline_vals = trial_curves[
                (trial_curves["method"] == baseline) & (trial_curves["budget"] == experiment_count)
            ]["gt_hits"]
            p = mannwhitney_p(nd_vals, baseline_vals, alternative="greater")
            rows.append(
                {
                    "panel": "b",
                    "comparison": f"NeuroDiscovery > {METHOD_LABELS[baseline]}",
                    "metric": "GT discoveries found",
                    "experiment_count": experiment_count,
                    "recall_target_pct": np.nan,
                    "alternative": "greater",
                    "p_value": p,
                    "p_value_label": p_value_text(p),
                }
            )
    nd_summary = trial_summary[trial_summary["method"] == "neurodiscovery"]
    for target in recall_targets:
        col = f"experiments_for_recall_{target}pct"
        for baseline in BASELINE_METHODS:
            p = mannwhitney_p(
                nd_summary[col],
                trial_summary[trial_summary["method"] == baseline][col],
                alternative="less",
            )
            rows.append(
                {
                    "panel": "d",
                    "comparison": f"NeuroDiscovery < {METHOD_LABELS[baseline]}",
                    "metric": "Experiments required",
                    "experiment_count": np.nan,
                    "recall_target_pct": target,
                    "alternative": "less",
                    "p_value": p,
                    "p_value_label": p_value_text(p),
                }
            )
    return pd.DataFrame(rows)


def value_at_budget(curves: pd.DataFrame, method: str, budget: int, metric: str) -> tuple[float, float, float]:
    sub = curves[(curves["method"] == method) & (curves["budget"] == budget)]
    if sub.empty:
        return np.nan, np.nan, np.nan
    row = sub.iloc[0]
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_lo"]), float(row[f"{metric}_hi"])


def plot_same_budget_discovery(curves: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    methods = list(GENERATOR_METHODS)
    budget_marks = [1000, 5000, 10000, 50000, 100000]
    available_budgets = set(int(x) for x in curves["budget"].unique())
    budget_marks = [b for b in budget_marks if b in available_budgets]
    if not budget_marks:
        budget_marks = sorted(available_budgets)[-min(5, len(available_budgets)) :]
    x = np.arange(len(budget_marks))
    offsets, width = grouped_bar_offsets(len(methods))

    fig = plt.figure(figsize=(11.3, 7.4))
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.50)
    ax_hits = fig.add_subplot(gs[0, 0])
    ax_recall = fig.add_subplot(gs[0, 1])
    ax_precision = fig.add_subplot(gs[1, 0])
    ax_strict = fig.add_subplot(gs[1, 1])

    for i, method in enumerate(methods):
        vals, lo_err, hi_err = [], [], []
        for budget in budget_marks:
            mean, lo, hi = value_at_budget(curves, method, budget, "gt_hits")
            vals.append(mean)
            lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
            lo_err.append(lo_delta)
            hi_err.append(hi_delta)
        ax_hits.bar(
            x + offsets[i],
            vals,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.45,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            label=METHOD_LABELS[method],
        )
    ax_hits.set_xticks(x)
    ax_hits.set_xticklabels([f"{b:,}" for b in budget_marks])
    ax_hits.set_xlabel("Number of experiments")
    ax_hits.set_ylabel("GT discoveries found")
    ax_hits.set_title("Same experiments: absolute discovery yield")
    ax_hits.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_hits.legend(loc="upper left", fontsize=10)
    panel_label(ax_hits, "a")

    for method in methods:
        sub = curves[curves["method"] == method]
        ax_recall.plot(
            sub["budget"],
            sub["recall_mean"],
            color=PALETTE[method],
            lw=2.3 if method == "neurodiscovery" else 1.8,
            label=METHOD_LABELS[method],
        )
        ax_recall.fill_between(
            sub["budget"].to_numpy(float),
            sub["recall_lo"].to_numpy(float),
            sub["recall_hi"].to_numpy(float),
            color=PALETTE[method],
            alpha=0.15,
            linewidth=0,
        )
    for b in budget_marks:
        ax_recall.axvline(b, color="#DADADA", lw=0.7, zorder=0)
    ax_recall.set_xscale("log")
    ax_recall.set_xlim(max(1, min(budget_marks) * 0.4), max(budget_marks) * 1.25)
    ax_recall.set_ylim(0, max(0.24, float(curves["recall_hi"].max()) * 1.08))
    ax_recall.set_xlabel("Number of experiments")
    ax_recall.set_ylabel("GT recall")
    ax_recall.set_title("Same experiments: cumulative recall curve")
    ax_recall.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_recall, "b")

    baseline_precision = {
        budget: best_baseline_value(curves, budget, "precision")
        for budget in budget_marks
    }
    for i, method in enumerate(methods):
        enrich, lo_err, hi_err = [], [], []
        for budget in budget_marks:
            mean, lo, hi = value_at_budget(curves, method, budget, "precision")
            denom = baseline_precision[budget] if baseline_precision[budget] > 0 else np.nan
            enrich.append(mean / denom)
            lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
            lo_err.append(lo_delta / denom)
            hi_err.append(hi_delta / denom)
        ax_precision.bar(
            x + offsets[i],
            enrich,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.45,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
    ax_precision.axhline(1.0, color="#8C8C8C", lw=1.2, linestyle="--")
    ax_precision.set_xticks(x)
    ax_precision.set_xticklabels([f"{b:,}" for b in budget_marks])
    ax_precision.set_xlabel("Number of experiments")
    ax_precision.set_ylabel("Precision enrichment vs best baseline")
    ax_precision.set_title("Same experiments: hit density enrichment")
    ax_precision.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_precision, "c")

    nd_advantage = []
    nd_advantage_lo = []
    nd_advantage_hi = []
    for budget in budget_marks:
        nd_mean, nd_lo, nd_hi = value_at_budget(curves, "neurodiscovery", budget, "gt_hits")
        best_baseline = best_baseline_value(curves, budget, "gt_hits")
        nd_advantage.append(nd_mean - best_baseline)
        lo_delta, hi_delta = nonnegative_interval_error(nd_mean, nd_lo, nd_hi)
        nd_advantage_lo.append(lo_delta)
        nd_advantage_hi.append(hi_delta)
    ax_strict.bar(
        x,
        nd_advantage,
        width=0.46,
        color=PALETTE["neurodiscovery"],
        edgecolor="#272727",
        linewidth=0.45,
        yerr=np.vstack([nd_advantage_lo, nd_advantage_hi]),
        error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
    )
    for xi, val in zip(x, nd_advantage, strict=False):
        ax_strict.text(
            xi,
            val + max(nd_advantage) * 0.035,
            f"+{int(round(val)):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_strict.set_xticks(x)
    ax_strict.set_xticklabels([f"{b:,}" for b in budget_marks])
    ax_strict.set_xlabel("Number of experiments")
    ax_strict.set_ylabel("Additional GT discoveries")
    ax_strict.set_title("Same experiments: gain over best baseline")
    ax_strict.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_strict, "d")

    fig.suptitle("Same number of experiments, NeuroDiscovery recovers more findings", y=0.995, fontsize=16, fontweight="bold")
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_same_budget_discovery.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_same_discovery_cost(trial_summary: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    methods = list(GENERATOR_METHODS)
    targets = [1, 5, 10, 20, 50]
    target_cols = [f"experiments_for_recall_{t}pct" for t in targets]
    x = np.arange(len(targets))
    offsets, width = grouped_bar_offsets(len(methods))

    fig = plt.figure(figsize=(11.3, 7.4))
    gs = fig.add_gridspec(2, 2, wspace=0.40, hspace=0.52)
    ax_cost = fig.add_subplot(gs[0, 0])
    ax_saved = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_frontier = fig.add_subplot(gs[1, 1])

    cost_stats: dict[str, dict[int, tuple[float, float, float]]] = {m: {} for m in methods}
    for method in methods:
        sub = trial_summary[trial_summary["method"] == method]
        for target, col in zip(targets, target_cols, strict=False):
            cost_stats[method][target] = mean_interval(sub[col])

    for i, method in enumerate(methods):
        vals, lo_err, hi_err = [], [], []
        for target in targets:
            mean, lo, hi = cost_stats[method][target]
            vals.append(mean)
            lo_delta, hi_delta = nonnegative_interval_error(mean, lo, hi)
            lo_err.append(lo_delta)
            hi_err.append(hi_delta)
        ax_cost.bar(
            x + offsets[i],
            vals,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.45,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            label=METHOD_LABELS[method],
        )
    ax_cost.set_yscale("log")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([f"{t}%" for t in targets])
    ax_cost.set_xlabel("Matched GT discovery target")
    ax_cost.set_ylabel("Experiments required")
    ax_cost.set_title("Same discovery count: experimental cost")
    ax_cost.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_cost.legend(loc="upper left", fontsize=10)
    panel_label(ax_cost, "a")

    for method in BASELINE_METHODS:
        ratios = []
        lo_err, hi_err = [], []
        for target in targets:
            base_mean, base_lo, base_hi = cost_stats[method][target]
            nd_mean, nd_lo, nd_hi = cost_stats["neurodiscovery"][target]
            ratios.append(base_mean / nd_mean)
            lo_err.append(max(0.0, base_lo / nd_hi - base_mean / nd_mean))
            hi_err.append(max(0.0, base_hi / nd_lo - base_mean / nd_mean))
        ax_saved.errorbar(
            x,
            ratios,
            yerr=np.vstack([lo_err, hi_err]),
            marker=MARKERS[method],
            markersize=6,
            lw=2.0,
            capsize=3,
            color=PALETTE[method],
            label=f"{METHOD_LABELS[method]} / NeuroDiscovery",
        )
    ax_saved.axhline(1.0, color="#BDBDBD", lw=1.0, linestyle="--")
    ax_saved.set_xticks(x)
    ax_saved.set_xticklabels([f"{t}%" for t in targets])
    ax_saved.set_xlabel("Matched GT discovery target")
    ax_saved.set_ylabel("Experimental cost ratio")
    ax_saved.set_title("Same discovery count: experiments saved")
    ax_saved.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_saved.legend(loc="upper left", fontsize=10)
    panel_label(ax_saved, "b")

    target_col = "experiments_for_recall_10pct"
    box_data = [
        pd.to_numeric(trial_summary[trial_summary["method"] == method][target_col], errors="coerce").dropna().to_numpy(float)
        for method in methods
    ]
    bp = ax_dist.boxplot(
        box_data,
        patch_artist=True,
        widths=0.56,
        showfliers=False,
        medianprops={"color": "#272727", "linewidth": 1.1},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9},
    )
    for patch, method in zip(bp["boxes"], methods, strict=False):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.78)
        patch.set_edgecolor("#272727")
        patch.set_linewidth(0.55)
    rng = np.random.default_rng(17)
    for i, (method, vals) in enumerate(zip(methods, box_data, strict=False), start=1):
        jitter = rng.normal(0.0, 0.035, size=len(vals))
        ax_dist.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=12,
            facecolor="white",
            edgecolor=PALETTE[method],
            linewidth=0.65,
            alpha=0.85,
            zorder=3,
        )
    ax_dist.set_yscale("log")
    ax_dist.set_xticks(np.arange(1, len(methods) + 1))
    ax_dist.set_xticklabels([SHORT_METHOD_LABELS[m] for m in methods], rotation=28, ha="right", fontsize=9)
    ax_dist.set_ylabel("Experiments to recover 10% GT")
    ax_dist.set_title("Seed-to-seed stability at the 10% target")
    ax_dist.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_dist, "c")

    label_offsets = {"neurodiscovery": (0.82, 0.018)}
    for method in methods:
        means = [cost_stats[method][target][0] for target in targets]
        ax_frontier.plot(
            means,
            np.array(targets) / 100.0,
            marker="o",
            lw=2.2 if method == "neurodiscovery" else 1.8,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
        for target, mean in zip(targets, means, strict=False):
            if target in (1, 10, 50):
                x_mult, y_add = label_offsets.get(method, (1.05, 0.0))
                ax_frontier.text(
                    mean * x_mult,
                    target / 100.0 + y_add,
                    f"{target}%",
            fontsize=10,
                    color=PALETTE[method],
                    va="center",
                )
    ax_frontier.set_xscale("log")
    ax_frontier.set_xlabel("Experiments required")
    ax_frontier.set_ylabel("Matched GT recall")
    ax_frontier.set_title("Discovery frontier")
    ax_frontier.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_frontier.legend(loc="lower right", fontsize=10)
    panel_label(ax_frontier, "d")

    fig.suptitle("Same number of discoveries, NeuroDiscovery uses fewer experiments", y=0.995, fontsize=16, fontweight="bold")
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_same_discovery_cost.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_budget_gain_focus(curves: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    budgets = [1000, 2000, 5000, 10000, 20000, 50000]
    available_budgets = set(int(x) for x in curves["budget"].unique())
    budgets = [b for b in budgets if b in available_budgets]
    if not budgets:
        budgets = sorted(available_budgets)[-min(5, len(available_budgets)) :]
    methods = list(GENERATOR_METHODS)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"wspace": 0.35})
    ax_yield, ax_gain = axes

    for method in methods:
        means, lows, highs = [], [], []
        for budget in budgets:
            mean, lo, hi = value_at_budget(curves, method, budget, "gt_hits")
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        means_arr = np.array(means, dtype=float)
        ax_yield.plot(
            budgets,
            means_arr,
            marker="o",
            lw=2.2 if method == "neurodiscovery" else 1.8,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
        ax_yield.fill_between(
            budgets,
            np.array(lows, dtype=float),
            np.array(highs, dtype=float),
            color=PALETTE[method],
            alpha=0.14,
            linewidth=0,
        )
    ax_yield.set_xscale("log")
    ax_yield.set_xlabel("Number of experiments")
    ax_yield.set_ylabel("GT discoveries found")
    ax_yield.set_title("Fixed-experiment discovery yield")
    ax_yield.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_yield.legend(loc="upper left", fontsize=10)
    panel_label(ax_yield, "a")

    gains = {method: [] for method in BASELINE_METHODS}
    for budget in budgets:
        nd, _, _ = value_at_budget(curves, "neurodiscovery", budget, "gt_hits")
        for method in BASELINE_METHODS:
            baseline, _, _ = value_at_budget(curves, method, budget, "gt_hits")
            gains[method].append(nd - baseline)
    for method, vals in gains.items():
        ax_gain.plot(
            budgets,
            vals,
            marker=MARKERS[method],
            lw=2.0,
            color=PALETTE[method],
            label=f"over {METHOD_LABELS[method]}",
        )
    ax_gain.axhline(0, color="#BDBDBD", lw=1.0, linestyle="--")
    ax_gain.set_xscale("log")
    ax_gain.set_xlabel("Number of experiments")
    ax_gain.set_ylabel("Additional GT discoveries")
    ax_gain.set_title("NeuroDiscovery gain at same experiments")
    ax_gain.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_gain.legend(loc="upper left", fontsize=10)
    best_gain = [max(gains[method][i] for method in BASELINE_METHODS) for i in range(len(budgets))]
    for b, v in zip(budgets[-3:], best_gain[-3:], strict=False):
        ax_gain.text(b * 1.05, v, f"+{int(round(v)):,}", fontsize=10, color=PALETTE["neurodiscovery"], va="center")
    panel_label(ax_gain, "b")

    fig.suptitle("Experiment-matched discovery gain", y=1.02, fontsize=16, fontweight="bold")
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_budget_matched_gain.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_target_savings_focus(trial_summary: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    targets = [1, 5, 10, 20, 50]
    x = np.arange(len(targets))
    offsets, width = grouped_bar_offsets(len(BASELINE_METHODS))
    methods = (*BASELINE_METHODS, "neurodiscovery")
    stats: dict[str, dict[int, tuple[float, float, float]]] = {m: {} for m in methods}
    for method in methods:
        sub = trial_summary[trial_summary["method"] == method]
        for target in targets:
            stats[method][target] = mean_interval(sub[f"experiments_for_recall_{target}pct"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"wspace": 0.36})
    ax_abs, ax_pct = axes

    for i, baseline in enumerate(BASELINE_METHODS):
        saved = []
        for target in targets:
            base = stats[baseline][target][0]
            nd = stats["neurodiscovery"][target][0]
            saved.append(base - nd)
        ax_abs.bar(
            x + offsets[i],
            saved,
            width=width,
            color=PALETTE[baseline],
            edgecolor="#272727",
            linewidth=0.45,
            label=f"vs {METHOD_LABELS[baseline]}",
        )
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels([f"{t}%" for t in targets])
    ax_abs.set_xlabel("Matched GT discovery target")
    ax_abs.set_ylabel("Experiments avoided")
    ax_abs.set_title("Absolute experimental savings")
    ax_abs.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_abs.legend(loc="upper left", fontsize=10)
    panel_label(ax_abs, "a")

    for baseline in BASELINE_METHODS:
        reductions = []
        for target in targets:
            base = stats[baseline][target][0]
            nd = stats["neurodiscovery"][target][0]
            reductions.append(100.0 * (base - nd) / base)
        ax_pct.plot(
            x,
            reductions,
            marker=MARKERS[baseline],
            lw=2.1,
            color=PALETTE[baseline],
            label=f"vs {METHOD_LABELS[baseline]}",
        )
        for xi, val in zip(x, reductions, strict=False):
            ax_pct.text(xi, val + 1.0, f"{val:.0f}%", ha="center", fontsize=10, color=PALETTE[baseline])
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels([f"{t}%" for t in targets])
    ax_pct.set_xlabel("Matched GT discovery target")
    ax_pct.set_ylabel("Experiment reduction")
    ax_pct.set_ylim(0, 100)
    ax_pct.set_title("Relative experimental savings")
    ax_pct.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_pct.legend(loc="lower right", fontsize=10)
    panel_label(ax_pct, "b")

    fig.suptitle("Discovery-matched experimental savings", y=1.02, fontsize=16, fontweight="bold")
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_discovery_matched_savings.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_generator_comparison_main(
    curves: pd.DataFrame,
    trial_summary: pd.DataFrame,
    scored: pd.DataFrame,
    exemplar_orders: dict[str, np.ndarray],
    out_dir: Path,
    top_n: int,
) -> None:
    apply_style()
    methods = list(GENERATOR_METHODS)
    baseline_methods = list(BASELINE_METHODS)
    best_baseline = "sciagents_style"
    fig = plt.figure(figsize=(15.0, 11.8))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.28, 1.0, 2.55],
        wspace=0.48,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:3])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[2, :])
    b_pos = ax_b.get_position()
    ax_b.set_position([b_pos.x0, b_pos.y0, b_pos.width * 0.91, b_pos.height])

    def main_panel_label_at(ax: plt.Axes, label: str, y: float, x_pad: float = 0.027) -> None:
        pos = ax.get_position()
        fig.text(
            pos.x0 - x_pad,
            y,
            label,
            fontsize=19,
            fontweight="bold",
            ha="left",
            va="top",
        )

    n_total = int(curves["budget"].max()) if not curves.empty else 426235
    gt_total = int(trial_summary["gt_total"].dropna().iloc[0]) if "gt_total" in trial_summary else 4263

    ax_a.axis("off")
    ax_a.set_title("Transdiagnostic\nbrain-atlas discovery", loc="left", pad=8, fontweight="bold")
    col_x = [0.18, 0.50, 0.82]

    # Row 1: task input, validation funnel, and real rendered atlas-map output.
    dot_x, dot_y = np.meshgrid(np.linspace(col_x[0] - 0.14, col_x[0] + 0.14, 9), np.linspace(0.60, 0.84, 6))
    ax_a.scatter(dot_x.ravel(), dot_y.ravel(), s=18, color="#C9C9C9", alpha=0.82, transform=ax_a.transAxes, clip_on=False)
    highlight_idx = np.array([4, 13, 21, 32, 41])
    ax_a.scatter(
        dot_x.ravel()[highlight_idx],
        dot_y.ravel()[highlight_idx],
        s=22,
        color=PALETTE["neurodiscovery"],
        alpha=0.90,
        transform=ax_a.transAxes,
        clip_on=False,
    )
    funnel = patches.Polygon(
        [[col_x[1] - 0.105, 0.84], [col_x[1] + 0.105, 0.84], [col_x[1] + 0.060, 0.58], [col_x[1] - 0.060, 0.58]],
        closed=True,
        transform=ax_a.transAxes,
        facecolor="#F3D3CF",
        edgecolor=PALETTE["neurodiscovery"],
        linewidth=1.0,
        alpha=0.80,
    )
    ax_a.add_patch(funnel)
    for y in (0.79, 0.70, 0.61):
        ax_a.plot([col_x[1] - 0.070, col_x[1] + 0.070], [y, y], color="white", lw=1.2, transform=ax_a.transAxes, alpha=0.85)

    arrow = patches.FancyArrowPatch(
        (0.625, 0.705),
        (0.700, 0.705),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.0,
        color="#777777",
        transform=ax_a.transAxes,
    )
    ax_a.add_patch(arrow)

    if DEFAULT_ATLAS_ICON.exists():
        icon = plt.imread(DEFAULT_ATLAS_ICON)
        ax_a.imshow(
            icon,
            extent=(col_x[2] - 0.145, col_x[2] + 0.145, 0.595, 0.835),
            transform=ax_a.transAxes,
            aspect="auto",
            zorder=2,
        )
    else:
        ax_a.add_patch(
            patches.Ellipse(
                (col_x[2], 0.705),
                0.215,
                0.142,
                angle=-6,
                transform=ax_a.transAxes,
                facecolor="#F7F7F7",
                edgecolor="#7A7A7A",
                linewidth=0.9,
            )
        )

    # Row 2: centered labels under each icon.
    row2 = [
        (col_x[0], "525,030\ncombinations", "#272727", 11.2, "bold"),
        (col_x[1], "validate\nand merge", "#555555", 10.6, "normal"),
        (col_x[2], f"brain-atlas map\n({gt_total:,} findings)", PALETTE["neurodiscovery"], 9.6, "bold"),
    ]
    for x_text, label, color, size, weight in row2:
        ax_a.text(
            x_text,
            0.475,
            label,
            transform=ax_a.transAxes,
            fontsize=size,
            fontweight=weight,
            color=color,
            ha="center",
            va="center",
            linespacing=1.00,
        )

    chip_specs = [
        (col_x[0], "11\ndisorders"),
        (col_x[1], "3,182\nROI readouts"),
        (col_x[2], "15\nfeatures"),
    ]
    # Row 3: the three dimensions that form the combination space.
    for x_center, text in chip_specs:
        box = patches.FancyBboxPatch(
            (x_center - 0.115, 0.175),
            0.23,
            0.17,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax_a.transAxes,
            facecolor="#F7F7F7",
            edgecolor="#D6D6D6",
            linewidth=0.8,
        )
        ax_a.add_patch(box)
        ax_a.text(
            x_center,
            0.260,
            text,
            transform=ax_a.transAxes,
            fontsize=9.8,
            color="#444444",
            ha="center",
            va="center",
            linespacing=1.05,
        )
    # Row 4: compact formula.
    ax_a.text(
        0.50,
        0.030,
        "11 disorders x 3,182 ROI readouts x 15 features\n"
        "= 525,030 disease-region-feature combinations",
        transform=ax_a.transAxes,
        fontsize=7.7,
        color="#555555",
        ha="center",
        va="center",
        linespacing=1.05,
    )

    early_budget_max = min(120000, int(curves["budget"].max()))
    early_curve_max = 0.0
    for method in baseline_methods:
        sub = curves[curves["method"] == method]
        sub_plot = sub[sub["budget"] <= early_budget_max]
        if sub_plot.empty:
            sub_plot = sub
        early_curve_max = max(early_curve_max, float(sub_plot["recall_hi"].max()))
        color = "#666666" if method == best_baseline else "#B8B8B8"
        lw = 2.0 if method == best_baseline else 1.2
        alpha = 0.78 if method == best_baseline else 0.48
        ax_b.plot(
            sub_plot["budget"],
            sub_plot["recall_mean"],
            lw=lw,
            color=color,
            alpha=alpha,
        )
        ax_b.fill_between(
            sub_plot["budget"].to_numpy(float),
            sub_plot["recall_lo"].to_numpy(float),
            sub_plot["recall_hi"].to_numpy(float),
            color=color,
            alpha=0.08 if method == best_baseline else 0.035,
            linewidth=0,
            zorder=1,
        )
    nd = curves[curves["method"] == "neurodiscovery"]
    nd_plot = nd[nd["budget"] <= early_budget_max]
    if nd_plot.empty:
        nd_plot = nd
    early_curve_max = max(early_curve_max, float(nd_plot["recall_hi"].max()))
    ax_b.plot(
        nd_plot["budget"],
        nd_plot["recall_mean"],
        lw=3.0,
        color=PALETTE["neurodiscovery"],
        zorder=5,
    )
    ax_b.fill_between(
        nd_plot["budget"].to_numpy(float),
        nd_plot["recall_lo"].to_numpy(float),
        nd_plot["recall_hi"].to_numpy(float),
        color=PALETTE["neurodiscovery"],
        alpha=0.15,
        linewidth=0,
        zorder=4,
    )
    ax_b.set_xlim(0, early_budget_max)
    ax_b.set_ylim(0, min(1.02, max(0.16, early_curve_max * 1.14)))
    ax_b.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax_b.set_xlabel("Candidate experiments evaluated")
    ax_b.set_ylabel("Recall of validated discoveries")
    ax_b.set_title("NeuroDiscovery recovers validated discoveries earlier")
    ax_b.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    if not nd_plot.empty:
        nd_last = nd_plot.iloc[-1]
        ax_b.text(
            early_budget_max * 0.985,
            float(nd_last["recall_mean"]),
            "NeuroDiscovery",
            color=PALETTE["neurodiscovery"],
            fontsize=10.5,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        )
    best_plot = curves[(curves["method"] == best_baseline) & (curves["budget"] <= early_budget_max)]
    if best_plot.empty:
        best_plot = curves[curves["method"] == best_baseline]
    if not best_plot.empty:
        best_last = best_plot.iloc[-1]
        ax_b.text(
            early_budget_max * 0.985,
            float(best_last["recall_mean"]),
            "Best published baseline (SciAgents)",
            color="#555555",
            fontsize=9.8,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.3),
        )

    budget_marks = [5000, 10000, 20000, 50000, 100000, 200000]
    available_budgets = set(int(x) for x in curves["budget"].unique())
    budget_marks = [b for b in budget_marks if b in available_budgets]
    x = np.arange(len(budget_marks))
    for xi, budget in zip(x, budget_marks, strict=False):
        baseline_stats = [value_at_budget(curves, method, budget, "gt_hits") for method in baseline_methods]
        baseline_means = np.array([m for m, _, _ in baseline_stats], dtype=float)
        baseline_los = np.array([lo for _, lo, _ in baseline_stats], dtype=float)
        baseline_his = np.array([hi for _, _, hi in baseline_stats], dtype=float)
        ax_c.vlines(xi, np.nanmin(baseline_los), np.nanmax(baseline_his), color="#C9C9C9", lw=7, alpha=0.45, zorder=1)
        jitter = np.linspace(-0.12, 0.12, len(baseline_methods))
        ax_c.scatter(np.full(len(baseline_means), xi) + jitter, baseline_means, s=22, color="#8F8F8F", alpha=0.65, zorder=2)
        best_mean, best_lo, best_hi = value_at_budget(curves, best_baseline, budget, "gt_hits")
        nd_mean, nd_lo, nd_hi = value_at_budget(curves, "neurodiscovery", budget, "gt_hits")
        ax_c.errorbar(
            xi - 0.05,
            best_mean,
            yerr=[[best_mean - best_lo], [best_hi - best_mean]],
            fmt="o",
            color="#555555",
            markersize=6,
            capsize=3,
            lw=1.4,
            label="Best baseline" if xi == 0 else None,
            zorder=4,
        )
        ax_c.errorbar(
            xi + 0.09,
            nd_mean,
            yerr=[[nd_mean - nd_lo], [nd_hi - nd_mean]],
            fmt="o",
            color=PALETTE["neurodiscovery"],
            markersize=7.5,
            capsize=3,
            lw=1.6,
            label="NeuroDiscovery" if xi == 0 else None,
            zorder=5,
        )
        ax_c.text(
            xi + 0.14,
            nd_mean,
            f"+{int(round(nd_mean - best_mean)):,}",
            color=PALETTE["neurodiscovery"],
            fontsize=8.9,
            va="center",
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"{int(b / 1000)}k" for b in budget_marks])
    ax_c.set_xlabel("Candidate experiments evaluated")
    ax_c.set_ylabel("Validated discoveries found")
    ax_c.set_title("Same experiments, more discoveries")
    ax_c.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax_c.yaxis.get_offset_text().set_fontsize(8.5)
    ax_c.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_c.legend(loc="upper left", fontsize=10.0)

    targets = [1, 5, 10, 20, 30, 50]
    baseline_mean_curves = []
    for method in baseline_methods:
        sub = trial_summary[trial_summary["method"] == method]
        baseline_mean_curves.append([mean_interval(sub[f"experiments_for_recall_{t}pct"])[0] for t in targets])
    baseline_arr = np.array(baseline_mean_curves, dtype=float)
    best_sub = trial_summary[trial_summary["method"] == best_baseline]
    nd_sub = trial_summary[trial_summary["method"] == "neurodiscovery"]
    best_vals = np.array([mean_interval(best_sub[f"experiments_for_recall_{t}pct"])[0] for t in targets], dtype=float)
    nd_vals = np.array([mean_interval(nd_sub[f"experiments_for_recall_{t}pct"])[0] for t in targets], dtype=float)
    y_max_d = float(np.nanmax([np.nanmax(baseline_arr), np.nanmax(nd_vals)])) * 1.12
    for method, vals in zip(baseline_methods, baseline_arr, strict=False):
        is_best = method == best_baseline
        ax_d.plot(
            targets,
            vals,
            color="#555555" if is_best else "#B8B8B8",
            marker=MARKERS.get(method, "o"),
            markersize=4.8 if is_best else 3.6,
            lw=2.0 if is_best else 1.15,
            alpha=0.90 if is_best else 0.62,
            zorder=3 if is_best else 1,
        )
    ax_d.plot(targets, nd_vals, color=PALETTE["neurodiscovery"], marker="o", lw=2.4)
    ax_d.set_xlim(0.8, 54)
    ax_d.set_ylim(0, y_max_d)
    ax_d.set_xticks(targets)
    ax_d.set_xticklabels([f"{t}%" for t in targets])
    ax_d.set_xlabel("Validated-discovery recall target")
    ax_d.set_ylabel("Experiments required\n(lower is better)")
    ax_d.set_title("Same recall, fewer experiments")
    ax_d.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax_d.yaxis.get_offset_text().set_fontsize(8.5)
    ax_d.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_d.text(
        50.8,
        best_vals[-1] + y_max_d * 0.025,
        "Best baseline",
        color="#555555",
        fontsize=8.8,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
    )
    ax_d.text(
        50.8,
        nd_vals[-1] - y_max_d * 0.035,
        "NeuroDiscovery",
        color=PALETTE["neurodiscovery"],
        fontsize=8.8,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=1.0),
    )
    ax_d.text(2.0, y_max_d * 0.94, "Published baselines", color="#777777", fontsize=8.3, ha="left", va="top")
    if len(targets) > 2 and nd_vals[2] > 0:
        speedup = best_vals[2] / nd_vals[2]
        ax_d.annotate(
            f"{speedup:.1f}x fewer\nat 10% recall",
            xy=(10, nd_vals[2]),
            xytext=(8.0, y_max_d * 0.18),
            color=PALETTE["neurodiscovery"],
            fontsize=8.8,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=PALETTE["neurodiscovery"], lw=0.9),
        )

    surface_panel = DEFAULT_COMPACT_SURFACE_PANEL if out_dir == DEFAULT_OUT_DIR else out_dir / "surface" / DEFAULT_COMPACT_SURFACE_PANEL.name
    if not surface_panel.exists():
        surface_panel = DEFAULT_SURFACE_PANEL if out_dir == DEFAULT_OUT_DIR else out_dir / "surface" / DEFAULT_SURFACE_PANEL.name
    ax_e.axis("off")
    if surface_panel.exists():
        surface_img = plt.imread(surface_panel)
        ax_e.imshow(surface_img)
    else:
        ax_e.text(
            0.5,
            0.5,
            f"Surface panel missing:\n{surface_panel}",
            ha="center",
            va="center",
            fontsize=12,
            color="#555555",
            transform=ax_e.transAxes,
        )
    row_ab_y = max(ax_a.get_position().y1, ax_b.get_position().y1) + 0.012
    row_cd_y = max(ax_c.get_position().y1, ax_d.get_position().y1) + 0.012
    row_e_y = ax_e.get_position().y1 + 0.012
    main_panel_label_at(ax_a, "a", row_ab_y)
    main_panel_label_at(ax_b, "b", row_ab_y, x_pad=0.058)
    main_panel_label_at(ax_c, "c", row_cd_y, x_pad=0.0)
    main_panel_label_at(ax_d, "d", row_cd_y, x_pad=0.075)
    main_panel_label_at(ax_e, "e", row_e_y)

    save_generator_panel_svgs(curves, trial_summary, out_dir, surface_panel, gt_total)
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_generator_comparison_main.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def draw_generator_panel_a(ax: plt.Axes, gt_total: int) -> None:
    ax.axis("off")
    ax.set_title("Transdiagnostic\nbrain-atlas discovery", loc="left", pad=8, fontweight="bold")
    col_x = [0.18, 0.50, 0.82]
    dot_x, dot_y = np.meshgrid(np.linspace(col_x[0] - 0.14, col_x[0] + 0.14, 9), np.linspace(0.60, 0.84, 6))
    ax.scatter(dot_x.ravel(), dot_y.ravel(), s=18, color="#C9C9C9", alpha=0.82, transform=ax.transAxes, clip_on=False)
    highlight_idx = np.array([4, 13, 21, 32, 41])
    ax.scatter(
        dot_x.ravel()[highlight_idx],
        dot_y.ravel()[highlight_idx],
        s=22,
        color=PALETTE["neurodiscovery"],
        alpha=0.90,
        transform=ax.transAxes,
        clip_on=False,
    )
    funnel = patches.Polygon(
        [[col_x[1] - 0.105, 0.84], [col_x[1] + 0.105, 0.84], [col_x[1] + 0.060, 0.58], [col_x[1] - 0.060, 0.58]],
        closed=True,
        transform=ax.transAxes,
        facecolor="#F3D3CF",
        edgecolor=PALETTE["neurodiscovery"],
        linewidth=1.0,
        alpha=0.80,
    )
    ax.add_patch(funnel)
    for y in (0.79, 0.70, 0.61):
        ax.plot([col_x[1] - 0.070, col_x[1] + 0.070], [y, y], color="white", lw=1.2, transform=ax.transAxes, alpha=0.85)
    arrow = patches.FancyArrowPatch(
        (0.625, 0.705),
        (0.700, 0.705),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.0,
        color="#777777",
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)
    if DEFAULT_ATLAS_ICON.exists():
        icon = plt.imread(DEFAULT_ATLAS_ICON)
        ax.imshow(
            icon,
            extent=(col_x[2] - 0.145, col_x[2] + 0.145, 0.595, 0.835),
            transform=ax.transAxes,
            aspect="auto",
            zorder=2,
        )
    else:
        ax.add_patch(
            patches.Ellipse(
                (col_x[2], 0.705),
                0.215,
                0.142,
                angle=-6,
                transform=ax.transAxes,
                facecolor="#F7F7F7",
                edgecolor="#7A7A7A",
                linewidth=0.9,
            )
        )
    row2 = [
        (col_x[0], "525,030\ncombinations", "#272727", 11.2, "bold"),
        (col_x[1], "validate\nand merge", "#555555", 10.6, "normal"),
        (col_x[2], f"brain-atlas map\n({gt_total:,} findings)", PALETTE["neurodiscovery"], 9.6, "bold"),
    ]
    for x_text, label, color, size, weight in row2:
        ax.text(
            x_text,
            0.475,
            label,
            transform=ax.transAxes,
            fontsize=size,
            fontweight=weight,
            color=color,
            ha="center",
            va="center",
            linespacing=1.00,
        )
    for x_center, text in [(col_x[0], "11\ndisorders"), (col_x[1], "3,182\nROI readouts"), (col_x[2], "15\nfeatures")]:
        box = patches.FancyBboxPatch(
            (x_center - 0.115, 0.175),
            0.23,
            0.17,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor="#F7F7F7",
            edgecolor="#D6D6D6",
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(
            x_center,
            0.260,
            text,
            transform=ax.transAxes,
            fontsize=9.8,
            color="#444444",
            ha="center",
            va="center",
            linespacing=1.05,
        )
    ax.text(
        0.50,
        0.030,
        "11 disorders x 3,182 ROI readouts x 15 features\n"
        "= 525,030 disease-region-feature combinations",
        transform=ax.transAxes,
        fontsize=7.7,
        color="#555555",
        ha="center",
        va="center",
        linespacing=1.05,
    )


def draw_generator_panel_b(ax: plt.Axes, curves: pd.DataFrame, baseline_methods: list[str], best_baseline: str) -> None:
    early_budget_max = min(120000, int(curves["budget"].max()))
    early_curve_max = 0.0
    for method in baseline_methods:
        sub = curves[curves["method"] == method]
        sub_plot = sub[sub["budget"] <= early_budget_max]
        if sub_plot.empty:
            sub_plot = sub
        early_curve_max = max(early_curve_max, float(sub_plot["recall_hi"].max()))
        color = "#666666" if method == best_baseline else "#B8B8B8"
        lw = 2.0 if method == best_baseline else 1.2
        alpha = 0.78 if method == best_baseline else 0.48
        ax.plot(sub_plot["budget"], sub_plot["recall_mean"], lw=lw, color=color, alpha=alpha)
        ax.fill_between(
            sub_plot["budget"].to_numpy(float),
            sub_plot["recall_lo"].to_numpy(float),
            sub_plot["recall_hi"].to_numpy(float),
            color=color,
            alpha=0.08 if method == best_baseline else 0.035,
            linewidth=0,
            zorder=1,
        )
    nd = curves[curves["method"] == "neurodiscovery"]
    nd_plot = nd[nd["budget"] <= early_budget_max]
    if nd_plot.empty:
        nd_plot = nd
    early_curve_max = max(early_curve_max, float(nd_plot["recall_hi"].max()))
    ax.plot(nd_plot["budget"], nd_plot["recall_mean"], lw=3.0, color=PALETTE["neurodiscovery"], zorder=5)
    ax.fill_between(
        nd_plot["budget"].to_numpy(float),
        nd_plot["recall_lo"].to_numpy(float),
        nd_plot["recall_hi"].to_numpy(float),
        color=PALETTE["neurodiscovery"],
        alpha=0.15,
        linewidth=0,
        zorder=4,
    )
    ax.set_xlim(0, early_budget_max)
    ax.set_ylim(0, min(1.02, max(0.16, early_curve_max * 1.14)))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel("Candidate experiments evaluated")
    ax.set_ylabel("Recall of validated discoveries")
    ax.set_title("NeuroDiscovery recovers validated discoveries earlier")
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    if not nd_plot.empty:
        nd_last = nd_plot.iloc[-1]
        ax.text(
            early_budget_max * 0.985,
            float(nd_last["recall_mean"]),
            "NeuroDiscovery",
            color=PALETTE["neurodiscovery"],
            fontsize=10.5,
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        )
    best_plot = curves[(curves["method"] == best_baseline) & (curves["budget"] <= early_budget_max)]
    if best_plot.empty:
        best_plot = curves[curves["method"] == best_baseline]
    if not best_plot.empty:
        best_last = best_plot.iloc[-1]
        ax.text(
            early_budget_max * 0.985,
            float(best_last["recall_mean"]),
            "Best published baseline (SciAgents)",
            color="#555555",
            fontsize=9.8,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.3),
        )


def draw_generator_panel_c(ax: plt.Axes, curves: pd.DataFrame, baseline_methods: list[str], best_baseline: str) -> None:
    budget_marks = [5000, 10000, 20000, 50000, 100000, 200000]
    available_budgets = set(int(x) for x in curves["budget"].unique())
    budget_marks = [b for b in budget_marks if b in available_budgets]
    x = np.arange(len(budget_marks))
    for xi, budget in zip(x, budget_marks, strict=False):
        baseline_stats = [value_at_budget(curves, method, budget, "gt_hits") for method in baseline_methods]
        baseline_means = np.array([m for m, _, _ in baseline_stats], dtype=float)
        baseline_los = np.array([lo for _, lo, _ in baseline_stats], dtype=float)
        baseline_his = np.array([hi for _, _, hi in baseline_stats], dtype=float)
        ax.vlines(xi, np.nanmin(baseline_los), np.nanmax(baseline_his), color="#C9C9C9", lw=7, alpha=0.45, zorder=1)
        jitter = np.linspace(-0.12, 0.12, len(baseline_methods))
        ax.scatter(np.full(len(baseline_means), xi) + jitter, baseline_means, s=22, color="#8F8F8F", alpha=0.65, zorder=2)
        best_mean, best_lo, best_hi = value_at_budget(curves, best_baseline, budget, "gt_hits")
        nd_mean, nd_lo, nd_hi = value_at_budget(curves, "neurodiscovery", budget, "gt_hits")
        ax.errorbar(
            xi - 0.05,
            best_mean,
            yerr=[[best_mean - best_lo], [best_hi - best_mean]],
            fmt="o",
            color="#555555",
            markersize=6,
            capsize=3,
            lw=1.4,
            label="Best baseline" if xi == 0 else None,
            zorder=4,
        )
        ax.errorbar(
            xi + 0.09,
            nd_mean,
            yerr=[[nd_mean - nd_lo], [nd_hi - nd_mean]],
            fmt="o",
            color=PALETTE["neurodiscovery"],
            markersize=7.5,
            capsize=3,
            lw=1.6,
            label="NeuroDiscovery" if xi == 0 else None,
            zorder=5,
        )
        ax.text(xi + 0.14, nd_mean, f"+{int(round(nd_mean - best_mean)):,}", color=PALETTE["neurodiscovery"], fontsize=8.9, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b / 1000)}k" for b in budget_marks])
    ax.set_xlabel("Candidate experiments evaluated")
    ax.set_ylabel("Validated discoveries found")
    ax.set_title("Same experiments, more discoveries")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(8.5)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=10.0)


def draw_generator_panel_d(ax: plt.Axes, trial_summary: pd.DataFrame, baseline_methods: list[str], best_baseline: str) -> None:
    targets = [1, 5, 10, 20, 30, 50]
    baseline_mean_curves = []
    for method in baseline_methods:
        sub = trial_summary[trial_summary["method"] == method]
        baseline_mean_curves.append([mean_interval(sub[f"experiments_for_recall_{t}pct"])[0] for t in targets])
    baseline_arr = np.array(baseline_mean_curves, dtype=float)
    best_sub = trial_summary[trial_summary["method"] == best_baseline]
    nd_sub = trial_summary[trial_summary["method"] == "neurodiscovery"]
    best_vals = np.array([mean_interval(best_sub[f"experiments_for_recall_{t}pct"])[0] for t in targets], dtype=float)
    nd_vals = np.array([mean_interval(nd_sub[f"experiments_for_recall_{t}pct"])[0] for t in targets], dtype=float)
    y_max_d = float(np.nanmax([np.nanmax(baseline_arr), np.nanmax(nd_vals)])) * 1.12
    for method, vals in zip(baseline_methods, baseline_arr, strict=False):
        is_best = method == best_baseline
        ax.plot(
            targets,
            vals,
            color="#555555" if is_best else "#B8B8B8",
            marker=MARKERS.get(method, "o"),
            markersize=4.8 if is_best else 3.6,
            lw=2.0 if is_best else 1.15,
            alpha=0.90 if is_best else 0.62,
            zorder=3 if is_best else 1,
        )
    ax.plot(targets, nd_vals, color=PALETTE["neurodiscovery"], marker="o", lw=2.4)
    ax.set_xlim(0.8, 54)
    ax.set_ylim(0, y_max_d)
    ax.set_xticks(targets)
    ax.set_xticklabels([f"{t}%" for t in targets])
    ax.set_xlabel("Validated-discovery recall target")
    ax.set_ylabel("Experiments required\n(lower is better)")
    ax.set_title("Same recall, fewer experiments")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(8.5)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.text(
        50.8,
        best_vals[-1] + y_max_d * 0.025,
        "Best baseline",
        color="#555555",
        fontsize=8.8,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
    )
    ax.text(
        50.8,
        nd_vals[-1] - y_max_d * 0.035,
        "NeuroDiscovery",
        color=PALETTE["neurodiscovery"],
        fontsize=8.8,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.74, pad=1.0),
    )
    ax.text(2.0, y_max_d * 0.94, "Published baselines", color="#777777", fontsize=8.3, ha="left", va="top")
    if len(targets) > 2 and nd_vals[2] > 0:
        speedup = best_vals[2] / nd_vals[2]
        ax.annotate(
            f"{speedup:.1f}x fewer\nat 10% recall",
            xy=(10, nd_vals[2]),
            xytext=(8.0, y_max_d * 0.18),
            color=PALETTE["neurodiscovery"],
            fontsize=8.8,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=PALETTE["neurodiscovery"], lw=0.9),
        )


def save_single_panel_svg(path: Path, figsize: tuple[float, float], draw_fn) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_generator_panel_svgs(
    curves: pd.DataFrame,
    trial_summary: pd.DataFrame,
    out_dir: Path,
    surface_panel: Path,
    gt_total: int,
) -> None:
    panel_dir = out_dir / PANEL_SVG_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    for stale in panel_dir.glob("*.svg"):
        stale.unlink()

    baseline_methods = list(BASELINE_METHODS)
    best_baseline = "sciagents_style"
    save_single_panel_svg(panel_dir / "a.svg", (4.0, 2.9), lambda ax: draw_generator_panel_a(ax, gt_total))
    save_single_panel_svg(panel_dir / "b.svg", (7.4, 2.7), lambda ax: draw_generator_panel_b(ax, curves, baseline_methods, best_baseline))
    save_single_panel_svg(panel_dir / "c.svg", (7.3, 2.7), lambda ax: draw_generator_panel_c(ax, curves, baseline_methods, best_baseline))
    save_single_panel_svg(panel_dir / "d.svg", (4.0, 2.7), lambda ax: draw_generator_panel_d(ax, trial_summary, baseline_methods, best_baseline))
    surface_svg = surface_panel.with_suffix(".svg")
    if surface_svg.exists():
        shutil.copyfile(surface_svg, panel_dir / "e.svg")
    else:
        fig, ax = plt.subplots(figsize=(10.0, 4.6))
        ax.axis("off")
        if surface_panel.exists():
            ax.imshow(plt.imread(surface_panel))
        fig.savefig(panel_dir / "e.svg", bbox_inches="tight")
        plt.close(fig)


def heatmap_matrix(ranked: pd.DataFrame, top_n: int, diseases: list[str], groups: list[str]) -> np.ndarray:
    top = ranked.head(top_n)
    hit = top[top["is_gt_top"]]
    mat = np.zeros((len(diseases), len(groups)), dtype=float)
    for i, disease in enumerate(diseases):
        for j, group in enumerate(groups):
            mat[i, j] = int(((hit["disease"] == disease) & (hit["map_group"] == group)).sum())
    return mat


def heatmap_matrix_from_order(
    scored: pd.DataFrame,
    order: np.ndarray,
    top_n: int,
    diseases: list[str],
    groups: list[str],
) -> np.ndarray:
    top_idx = order[:top_n]
    hit = scored.iloc[top_idx]
    hit = hit[hit["is_gt_top"]]
    mat = np.zeros((len(diseases), len(groups)), dtype=float)
    for i, disease in enumerate(diseases):
        for j, group in enumerate(groups):
            mat[i, j] = int(((hit["disease"] == disease) & (hit["map_group"] == group)).sum())
    return mat


def recovery_matrix(method_matrix: np.ndarray, gt_matrix: np.ndarray) -> np.ndarray:
    out = np.full_like(gt_matrix, np.nan, dtype=float)
    mask = gt_matrix > 0
    out[mask] = np.clip(method_matrix[mask] / gt_matrix[mask], 0.0, 1.0)
    return out


def matrix_recall(method_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    total = float(gt_matrix.sum())
    if total <= 0:
        return np.nan
    return float(method_matrix.sum() / total)


def masked_log_burden_matrix(method_matrix: np.ndarray, gt_matrix: np.ndarray) -> np.ndarray:
    out = np.full_like(gt_matrix, np.nan, dtype=float)
    mask = gt_matrix > 0
    out[mask] = np.log1p(method_matrix[mask])
    return out


def recovery_cmap() -> matplotlib.colors.LinearSegmentedColormap:
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "gt_recovery_white_yellow_green",
        [(0.0, "#FFFFFF"), (0.5, "#FEE08B"), (1.0, "#1A9850")],
        N=256,
    )


def gt_weighted_gap(method_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    total = float(gt_matrix.sum())
    if total <= 0:
        return np.nan
    rec = recovery_matrix(method_matrix, gt_matrix)
    mask = gt_matrix > 0
    return float(np.sum(np.abs(1.0 - rec[mask]) * gt_matrix[mask]) / total)


def select_neurodiscovery_focus(
    matrices_by_method: dict[str, np.ndarray],
    diseases: list[str],
    groups: list[str],
    max_diseases: int = 5,
    max_groups: int = 5,
) -> tuple[list[str], list[str], list[int], list[int]]:
    gt = matrices_by_method["exhaustive_gt"]
    nd = matrices_by_method["neurodiscovery"]
    best_baseline = best_baseline_matrix(matrices_by_method)
    advantage = np.maximum(nd - best_baseline, 0.0)
    advantage[gt <= 0] = 0.0

    def top_indices(scores: np.ndarray, fallback_totals: np.ndarray, limit: int) -> list[int]:
        primary = np.flatnonzero(scores > 0).tolist()
        fallback = [idx for idx in np.flatnonzero(fallback_totals > 0).tolist() if idx not in primary]
        candidates = primary + fallback
        ordered = sorted(
            candidates,
            key=lambda idx: (float(scores[idx]), float(fallback_totals[idx])),
            reverse=True,
        )
        return ordered[: min(limit, len(ordered))]

    disease_idx = top_indices(advantage.sum(axis=1), nd.sum(axis=1), max_diseases)
    group_idx = top_indices(advantage.sum(axis=0), nd.sum(axis=0), max_groups)
    return (
        [diseases[i] for i in disease_idx],
        [groups[j] for j in group_idx],
        disease_idx,
        group_idx,
    )


def subset_matrix(matrix: np.ndarray, row_idx: list[int], col_idx: list[int]) -> np.ndarray:
    return matrix[np.ix_(row_idx, col_idx)]


def plot_method_maps(scored: pd.DataFrame, exemplar_orders: dict[str, np.ndarray], out_dir: Path, top_n: int) -> None:
    apply_style()
    diseases = sorted(scored["disease"].dropna().unique())
    preferred_groups = [
        "Default",
        "Limbic",
        "Salience/VAttn",
        "Dorsal attention",
        "Somatomotor",
        "Control",
        "Visual",
        "Subcortical/limbic",
        "sMRI volume",
        "Other cortical",
    ]
    available = set(scored["map_group"].dropna().unique())
    groups = [g for g in preferred_groups if g in available]
    methods = ["exhaustive_gt", *BASELINE_METHODS, "neurodiscovery"]
    matrices = [
        heatmap_matrix_from_order(scored, exemplar_orders[m], top_n, diseases, groups)
        for m in methods
    ]
    matrices_by_method = dict(zip(methods, matrices, strict=True))
    diseases, groups, disease_idx, group_idx = select_neurodiscovery_focus(matrices_by_method, diseases, groups)
    matrices = [subset_matrix(m, disease_idx, group_idx) for m in matrices]
    gt_matrix = matrices[0]
    gt_burden = masked_log_burden_matrix(gt_matrix, gt_matrix)
    recovery_matrices = [gt_burden, *[recovery_matrix(m, gt_matrix) for m in matrices[1:]]]
    gt_cmap = plt.cm.Greys.copy()
    rec_cmap = recovery_cmap()
    gt_cmap.set_bad("#F2F2F2")
    rec_cmap.set_bad("#000000")
    gt_vmax = float(np.nanmax(gt_burden)) or 1.0

    n_methods = len(methods)
    n_cols = 3
    n_rows = int(math.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.8 * n_rows), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).ravel()
    im_recovery = None
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for ax, method, matrix, raw_matrix, label in zip(axes_arr, methods, recovery_matrices, matrices, panel_letters, strict=False):
        if method == "exhaustive_gt":
            ax.imshow(matrix, aspect="auto", cmap=gt_cmap, vmin=0, vmax=gt_vmax)
            ax.set_title("Exhaustive GT burden", fontsize=12.5)
        else:
            im_recovery = ax.imshow(matrix, aspect="auto", cmap=rec_cmap, vmin=0, vmax=1.0)
            ax.set_title(
                f"{METHOD_LABELS[method]}: {matrix_recall(raw_matrix, gt_matrix):.1%} GT, gap {gt_weighted_gap(raw_matrix, gt_matrix):.2f}",
                fontsize=12.5,
            )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_yticks(range(len(diseases)))
        ax.set_yticklabels(diseases)
        ax.tick_params(length=0)
        panel_label(ax, label)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes_arr[len(methods) :]:
        ax.axis("off")
    if im_recovery is not None:
        cbar = fig.colorbar(im_recovery, ax=axes_arr[:n_methods].tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Fraction of GT cell recovered", fontsize=11)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=10)
    fig.suptitle("NeuroDiscovery-enriched GT sectors", y=0.99, fontsize=16, fontweight="bold")
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_method_discovery_maps.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    out_dir: Path,
    all_tests: Path,
    kg_path: Path | None,
    generation_first_dir: Path | None,
    gt_top_frac: float,
    gt_total: int,
    seed: int,
    top_n: int,
    n_trials: int,
) -> None:
    method_configs = {
        method: {
            "label": METHOD_LABELS[method],
            **method_publication(method),
        }
        for method in ("exhaustive_gt", *GENERATOR_METHODS)
    }
    manifest = {
        "all_tests": str(all_tests),
        "kg_path": str(kg_path) if kg_path else None,
        "generation_first_dir": str(generation_first_dir) if generation_first_dir else None,
        "gt_definition": {
            "primary": f"top {gt_top_frac:.4%} by abs_adjusted_residual_d from exhaustive results",
            "gt_total": gt_total,
            "strict_secondary": "q_fdr_global < 0.05",
        },
        "methods": METHOD_LABELS,
        "method_configs": method_configs,
        "generator_methods": list(GENERATOR_METHODS),
        "random_seed_base": seed,
        "n_trials_per_stochastic_method": n_trials,
        "curve_interval": "2.5th to 97.5th percentile across seeds",
        "ranked_candidates_export_top_n": top_n,
        "panel_e": (
            "Cortical surface comparison of ROI-level GT recovery for "
            + ", ".join(METHOD_LABELS[method] for method in ("exhaustive_gt", *GENERATOR_METHODS))
            + "."
        ),
        "baseline_policy": (
            "Exhaustive is a GT/oracle point, not a generator curve. When "
            "generation_first_dir is set, published autoresearch baselines are evaluated "
            "from LLM-generated hypotheses mapped back to the Case Study 1 universe, "
            "then completed with a deterministic random tail; outcome labels, effect "
            "sizes, FDR values, the full candidate table, and NeuroDiscovery feedback "
            "are not exposed to baseline prompts."
        ),
    }
    (out_dir / "case1_method_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--kg", type=Path, default=DEFAULT_CASE1_KG if DEFAULT_CASE1_KG.exists() else DEFAULT_FULL_KG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gt-top-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=260616)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--export-top-n", type=int, default=5000)
    parser.add_argument("--map-top-n", type=int, default=100000)
    parser.add_argument(
        "--generation-first-dir",
        type=Path,
        default=None,
        help="Directory containing generation_first_mapped_hypotheses.csv for stricter published-autoresearch baselines.",
    )
    parser.add_argument(
        "--only-negative-feedback-ablation",
        action="store_true",
        help="Only run the NeuroDiscovery positive-vs-negative feedback ablation and plot.",
    )
    parser.add_argument(
        "--skip-negative-feedback-ablation",
        action="store_true",
        help="Skip the secondary negative-feedback ablation during the main generator comparison run.",
    )
    return parser.parse_args()


def remove_stale_outputs(out_dir: Path) -> None:
    for name in (
        "ranked_candidates_exhaustive_oracle.csv",
        "ranked_candidates_semnet_lbd.csv",
        "ranked_candidates_enigma_transdiagnostic_prior.csv",
        "ranked_candidates_random.csv",
        "ranked_candidates_random_walk.csv",
        "ranked_candidates_llm_brainstorm.csv",
        "ranked_candidates_kg_degree.csv",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs(args.out_dir)
    kg = load_kg_index(args.kg)
    df = load_results(args.all_tests, args.gt_top_frac)
    scored = add_generator_scores(df, kg, args.seed)
    generation_first_mapped = None
    if args.generation_first_dir is not None:
        mapped_path = args.generation_first_dir / "generation_first_mapped_hypotheses.csv"
        if not mapped_path.exists():
            raise FileNotFoundError(f"Missing generation-first mapped hypotheses: {mapped_path}")
        generation_first_mapped = pd.read_csv(mapped_path)
    n_gt = int(scored["is_gt_top"].sum())
    budgets = budget_grid(len(scored))
    if args.only_negative_feedback_ablation:
        (
            ablation_trial_curves,
            ablation_curves,
            ablation_summary,
            ablation_audit,
        ) = run_negative_feedback_ablation(
            scored=scored,
            budgets=budgets,
            n_gt=n_gt,
            n_trials=args.trials,
            seed=args.seed,
        )
        ablation_trial_curves.to_csv(args.out_dir / "case1_negative_feedback_ablation_curves_by_trial.csv", index=False)
        ablation_curves.to_csv(args.out_dir / "case1_negative_feedback_ablation_curves.csv", index=False)
        ablation_summary.to_csv(args.out_dir / "case1_negative_feedback_ablation_summary.csv", index=False)
        if not ablation_audit.empty:
            ablation_audit.to_csv(args.out_dir / "case1_negative_feedback_ablation_audit.csv", index=False)
        plot_negative_feedback_ablation(ablation_curves, ablation_summary, args.out_dir, len(scored))
        (args.out_dir / "case1_negative_feedback_ablation_manifest.json").write_text(
            json.dumps(
                {
                    "all_tests": str(args.all_tests),
                    "gt_definition": {
                        "primary": f"top {args.gt_top_frac:.4%} by abs_adjusted_residual_d from exhaustive results",
                        "gt_total": n_gt,
                        "negative_proxy": "Executed hypotheses outside the GT-top set are treated as contradicted-result proxies for this ablation.",
                    },
                    "strategies": {
                        method: {
                            "label": METHOD_LABELS[method],
                            **method_publication(method),
                        }
                        for method in (
                            "neurodiscovery_positive_only",
                            "neurodiscovery_negative_feature_only",
                            "neurodiscovery_negative_pair_only",
                            "neurodiscovery_negative_context_only",
                            "neurodiscovery_negative_hybrid",
                        )
                    },
                    "random_seed_base": args.seed,
                    "n_trials": args.trials,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Loaded {len(scored):,} executed exhaustive tests")
        print(f"Primary GT discoveries: {n_gt:,} (top {args.gt_top_frac:.2%} by |d|)")
        print(f"Negative-feedback ablation output: {args.out_dir}")
        return

    trial_curves, curve_summary, method_summary, trial_summary, exemplar_orders, _labels = run_benchmark(
        scored=scored,
        budgets=budgets,
        n_gt=n_gt,
        n_trials=args.trials,
        seed=args.seed,
        map_top_n=args.map_top_n,
        generation_first_mapped=generation_first_mapped,
    )

    trial_curves.to_csv(args.out_dir / "case1_discovery_curves_by_trial.csv", index=False)
    curve_summary.to_csv(args.out_dir / "case1_discovery_curves.csv", index=False)
    method_summary.to_csv(args.out_dir / "case1_method_summary.csv", index=False)
    trial_summary.to_csv(args.out_dir / "case1_method_summary_by_trial.csv", index=False)
    if not args.skip_negative_feedback_ablation:
        (
            ablation_trial_curves,
            ablation_curves,
            ablation_summary,
            ablation_audit,
        ) = run_negative_feedback_ablation(
            scored=scored,
            budgets=budgets,
            n_gt=n_gt,
            n_trials=args.trials,
            seed=args.seed,
        )
        ablation_trial_curves.to_csv(args.out_dir / "case1_negative_feedback_ablation_curves_by_trial.csv", index=False)
        ablation_curves.to_csv(args.out_dir / "case1_negative_feedback_ablation_curves.csv", index=False)
        ablation_summary.to_csv(args.out_dir / "case1_negative_feedback_ablation_summary.csv", index=False)
        if not ablation_audit.empty:
            ablation_audit.to_csv(args.out_dir / "case1_negative_feedback_ablation_audit.csv", index=False)
    comparison_experiments = [1000, 5000, 10000, 20000, 50000, 100000]
    available_experiments = set(int(x) for x in trial_curves["budget"].unique())
    comparison_experiments = [b for b in comparison_experiments if b in available_experiments]
    comparison_recall_targets = [1, 5, 10, 20, 30, 50]
    compute_generator_p_values(
        trial_curves,
        trial_summary,
        comparison_experiments,
        comparison_recall_targets,
    ).to_csv(args.out_dir / "case1_generator_comparison_p_values.csv", index=False)
    save_exemplar_rankings(scored, exemplar_orders, args.out_dir, args.export_top_n)
    plot_efficiency(curve_summary, method_summary, args.out_dir, len(scored))
    if not args.skip_negative_feedback_ablation:
        plot_negative_feedback_ablation(ablation_curves, ablation_summary, args.out_dir, len(scored))
    plot_same_budget_discovery(curve_summary, args.out_dir)
    plot_same_discovery_cost(trial_summary, args.out_dir)
    plot_budget_gain_focus(curve_summary, args.out_dir)
    plot_target_savings_focus(trial_summary, args.out_dir)
    plot_method_maps(scored, exemplar_orders, args.out_dir, args.map_top_n)
    plot_generator_comparison_main(
        curve_summary,
        trial_summary,
        scored,
        exemplar_orders,
        args.out_dir,
        args.map_top_n,
    )
    write_manifest(
        args.out_dir,
        args.all_tests,
        args.kg,
        args.generation_first_dir,
        args.gt_top_frac,
        n_gt,
        args.seed,
        args.export_top_n,
        args.trials,
    )

    print(f"Loaded {len(scored):,} executed exhaustive tests")
    print(f"Primary GT discoveries: {n_gt:,} (top {args.gt_top_frac:.2%} by |d|)")
    print(f"Strict global-FDR discoveries: {int(scored['is_strict_fdr'].sum()):,}")
    print(f"Trials per stochastic method: {args.trials:,}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
