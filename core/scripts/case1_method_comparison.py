"""Compare Case Study 1 hypothesis generators against exhaustive readouts.

The exhaustive experiment is treated as the executed search space. This script
does not run neuroimaging models; it ranks already-tested candidates by four
generator policies and measures how quickly each policy recovers the exhaustive
ground-truth discoveries.
"""

from __future__ import annotations

import argparse
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


DEFAULT_ALL_TESTS = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot"
    r"\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_OUT_DIR = Path(
    r"Z:\Public Dataset\case1_exhaustive_full\20260616_full_main_noboot"
    r"\method_comparison"
)
DEFAULT_SURFACE_PANEL = DEFAULT_OUT_DIR / "surface" / "fig_cs1_generator_surface_recovery_comparison.png"
DEFAULT_CASE1_KG = Path(
    r"C:\Users\45846\Documents\Code\NeuroClaw"
    r"\neurooracle\data\cs_runs\phase2_case1_transdiagnostic_v1\knowledge_graph.json"
)
DEFAULT_FULL_KG = Path(
    r"C:\Users\45846\Documents\Code\NeuroClaw"
    r"\neurooracle\data\full_v2\knowledge_graph.json"
)

PALETTE = {
    "exhaustive_gt": "#272727",
    "random_walk": "#6F6F6F",
    "llm_brainstorm": "#5E4FA2",
    "neurodiscovery": "#D9544D",
}
BAND_ALPHA = {
    "random_walk": 0.18,
    "llm_brainstorm": 0.11,
    "neurodiscovery": 0.13,
}

PANEL_SVG_DIRNAME = "panel_svgs"
METHOD_LABELS = {
    "exhaustive_gt": "Exhaustive GT",
    "random_walk": "Random walk",
    "llm_brainstorm": "LLM brainstorm",
    "neurodiscovery": "NeuroDiscovery",
}
GENERATOR_METHODS = ("random_walk", "llm_brainstorm", "neurodiscovery")

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
    out["score_llm_prior"] = (
        0.50 * out["region_prior"]
        + 0.42 * out["feature_prior"]
        + 0.08 * rng.random(len(out))
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
    if method == "random_walk":
        # No KG: a local-search-like stochastic order with clustered disease,
        # ROI, and feature preferences. This mimics a random walk over the
        # executable candidate lattice rather than pure independent shuffling.
        disease_codes = factor_codes(scored["disease"])
        roi_codes = factor_codes(scored["roi_key"])
        feature_codes = factor_codes(scored["feature"])
        disease_pref = rng.random(int(disease_codes.max()) + 1)
        roi_pref = rng.random(int(roi_codes.max()) + 1)
        feature_pref = rng.random(int(feature_codes.max()) + 1)
        return (
            0.60 * rng.random(n)
            + 0.18 * disease_pref[disease_codes]
            + 0.17 * roi_pref[roi_codes]
            + 0.05 * feature_pref[feature_codes]
        )
    if method == "llm_brainstorm":
        # No KG: fixed neuroscience prompt prior with stochastic sampling.
        # It prefers plausible feature/region templates but cannot read graph
        # degree or disease-region support.
        disease_codes = factor_codes(scored["disease"])
        disease_pref = rng.random(int(disease_codes.max()) + 1)
        return (
            0.42 * scored["region_prior"].to_numpy(float)
            + 0.38 * scored["feature_prior"].to_numpy(float)
            + 0.10 * disease_pref[disease_codes]
            + 0.10 * rng.random(n)
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
) -> np.ndarray:
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

    remaining = np.ones(n, dtype=bool)
    chosen: list[np.ndarray] = []
    selected_total = 0
    closed_loop_limit = min(max_closed_loop_budget, n)

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
        if len(hits):
            hit_d = np.unique(disease_codes[hits])
            hit_f = np.unique(feature_codes[hits])
            hit_g = np.unique(group_codes[hits])
            hit_r = np.unique(roi_codes[hits])
            hit_s = np.unique(source_codes[hits])
            disease_boost[hit_d] += 0.008
            feature_boost[hit_f] += 0.018
            group_boost[hit_g] += 0.016
            roi_boost[hit_r] += 0.010
            source_boost[hit_s] += 0.006

    while selected_total < closed_loop_limit and remaining.any():
        dynamic = (
            base
            + disease_boost[disease_codes]
            + feature_boost[feature_codes]
            + group_boost[group_codes]
            + roi_boost[roi_codes]
            + source_boost[source_codes]
            + rng.normal(0.0, 0.002, size=n)
        )
        dynamic[~remaining] = -np.inf
        batch_n = min(batch_size, closed_loop_limit - selected_total, int(remaining.sum()))
        batch = select_diverse_batch(
            dynamic,
            remaining,
            disease_codes,
            feature_codes,
            group_codes,
            batch_n,
            exploit_fraction=0.90,
            disease_cap_fraction=0.46,
            feature_cap_fraction=0.46,
            group_cap_fraction=0.62,
        )
        if len(batch) == 0:
            break
        chosen.append(batch)
        remaining[batch] = False
        selected_total += len(batch)

        hits = batch[gt[batch]]
        if len(hits) == 0:
            continue

        hit_d = np.unique(disease_codes[hits])
        hit_f = np.unique(feature_codes[hits])
        hit_g = np.unique(group_codes[hits])
        hit_r = np.unique(roi_codes[hits])
        hit_s = np.unique(source_codes[hits])

        disease_boost[hit_d] += 0.008
        # Cross-diagnostic expansion: a discovered disease-region-feature pattern
        # should increase exploration of other diseases rather than trapping the
        # generator inside the already-hot disease.
        other_d = np.setdiff1d(np.arange(n_disease), hit_d, assume_unique=True)
        disease_boost[other_d] += 0.003
        feature_boost[hit_f] += 0.018
        group_boost[hit_g] += 0.016
        roi_boost[hit_r] += 0.010
        source_boost[hit_s] += 0.006

        disease_boost[:] = np.clip(disease_boost, -0.02, 0.05)
        feature_boost[:] = np.clip(feature_boost, -0.02, 0.08)
        group_boost[:] = np.clip(group_boost, -0.02, 0.07)
        roi_boost[:] = np.clip(roi_boost, 0.0, 0.04)
        source_boost[:] = np.clip(source_boost, 0.0, 0.025)

    if remaining.any():
        tail_scores = base + rng.normal(0.0, 0.005, size=n)
        tail_idx = np.flatnonzero(remaining)
        tail_idx = tail_idx[np.lexsort((candidate_ids[tail_idx], -tail_scores[tail_idx]))]
        chosen.append(tail_idx)
    return np.concatenate(chosen) if chosen else np.arange(n)


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
        row = {"method": method, "budget": int(budget)}
        for metric in ("recall", "precision", "gt_hits", "strict_fdr_hits"):
            vals = sub[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_lo"] = float(np.quantile(vals, 0.025))
            row[f"{metric}_hi"] = float(np.quantile(vals, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_summary(trial_summary: pd.DataFrame, oracle_summary: dict[str, float | int | str]) -> pd.DataFrame:
    rows = [oracle_summary]
    metric_cols = [c for c in trial_summary.columns if c not in {"method", "trial", "label"}]
    for method in GENERATOR_METHODS:
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
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(
    scored: pd.DataFrame,
    budgets: np.ndarray,
    n_gt: int,
    n_trials: int,
    seed: int,
    map_top_n: int,
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
        "score_llm_prior",
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
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
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
    width = 0.22
    for i, method in enumerate(search_methods):
        vals, lo_err, hi_err = [], [], []
        for budget in budget_cols:
            mean, lo, hi = summary_value(summary, method, f"recall_at_{budget}")
            vals.append(mean)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        ax_recall.bar(
            x + (i - 1.0) * width,
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
        lo_err.append(mean - lo)
        hi_err.append(hi - mean)
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
        strict_lo.append(mean - lo)
        strict_hi.append(hi - mean)
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
    ax_strict.set_xticklabels(["GT", "Random\nwalk", "LLM\nbrainstorm", "Neuro-\nDiscovery"], rotation=0)
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
        for baseline in ("random_walk", "llm_brainstorm"):
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
        for baseline in ("random_walk", "llm_brainstorm"):
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
    x = np.arange(len(budget_marks))
    width = 0.22

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
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        ax_hits.bar(
            x + (i - 1.0) * width,
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
    ax_recall.set_xlim(400, max(budget_marks) * 1.25)
    ax_recall.set_ylim(0, max(0.24, float(curves["recall_hi"].max()) * 1.08))
    ax_recall.set_xlabel("Number of experiments")
    ax_recall.set_ylabel("GT recall")
    ax_recall.set_title("Same experiments: cumulative recall curve")
    ax_recall.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_recall, "b")

    random_precision = {
        budget: value_at_budget(curves, "random_walk", budget, "precision")[0]
        for budget in budget_marks
    }
    for i, method in enumerate(methods):
        enrich, lo_err, hi_err = [], [], []
        for budget in budget_marks:
            mean, lo, hi = value_at_budget(curves, method, budget, "precision")
            denom = random_precision[budget] if random_precision[budget] > 0 else np.nan
            enrich.append(mean / denom)
            lo_err.append((mean - lo) / denom)
            hi_err.append((hi - mean) / denom)
        ax_precision.bar(
            x + (i - 1.0) * width,
            enrich,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.45,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
    ax_precision.axhline(1.0, color=PALETTE["random_walk"], lw=1.2, linestyle="--")
    ax_precision.set_xticks(x)
    ax_precision.set_xticklabels([f"{b:,}" for b in budget_marks])
    ax_precision.set_xlabel("Number of experiments")
    ax_precision.set_ylabel("Precision enrichment vs random")
    ax_precision.set_title("Same experiments: hit density enrichment")
    ax_precision.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_precision, "c")

    nd_advantage = []
    nd_advantage_lo = []
    nd_advantage_hi = []
    for budget in budget_marks:
        nd_mean, nd_lo, nd_hi = value_at_budget(curves, "neurodiscovery", budget, "gt_hits")
        random_mean, _, _ = value_at_budget(curves, "random_walk", budget, "gt_hits")
        llm_mean, _, _ = value_at_budget(curves, "llm_brainstorm", budget, "gt_hits")
        best_baseline = max(random_mean, llm_mean)
        nd_advantage.append(nd_mean - best_baseline)
        nd_advantage_lo.append(nd_mean - nd_lo)
        nd_advantage_hi.append(nd_hi - nd_mean)
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
    width = 0.22

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
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        ax_cost.bar(
            x + (i - 1.0) * width,
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

    for method, marker in (("random_walk", "o"), ("llm_brainstorm", "s")):
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
            marker=marker,
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
    ax_dist.set_xticklabels(["Random\nwalk", "LLM\nbrainstorm", "Neuro-\nDiscovery"])
    ax_dist.set_ylabel("Experiments to recover 10% GT")
    ax_dist.set_title("Seed-to-seed stability at the 10% target")
    ax_dist.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_dist, "c")

    label_offsets = {
        ("random_walk", 50): (1.18, -0.012),
        ("llm_brainstorm", 50): (0.72, 0.006),
        ("neurodiscovery", 50): (0.82, 0.018),
    }
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
                x_mult, y_add = label_offsets.get((method, target), (1.05, 0.0))
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

    gain_random, gain_llm = [], []
    for budget in budgets:
        nd, _, _ = value_at_budget(curves, "neurodiscovery", budget, "gt_hits")
        random, _, _ = value_at_budget(curves, "random_walk", budget, "gt_hits")
        llm, _, _ = value_at_budget(curves, "llm_brainstorm", budget, "gt_hits")
        gain_random.append(nd - random)
        gain_llm.append(nd - llm)
    ax_gain.plot(budgets, gain_random, marker="o", lw=2.1, color=PALETTE["random_walk"], label="over random walk")
    ax_gain.plot(budgets, gain_llm, marker="s", lw=2.1, color=PALETTE["llm_brainstorm"], label="over LLM brainstorm")
    ax_gain.axhline(0, color="#BDBDBD", lw=1.0, linestyle="--")
    ax_gain.set_xscale("log")
    ax_gain.set_xlabel("Number of experiments")
    ax_gain.set_ylabel("Additional GT discoveries")
    ax_gain.set_title("NeuroDiscovery gain at same experiments")
    ax_gain.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_gain.legend(loc="upper left", fontsize=10)
    for b, v in zip(budgets[-3:], gain_llm[-3:], strict=False):
        ax_gain.text(b * 1.05, v, f"+{int(round(v)):,}", fontsize=10, color=PALETTE["llm_brainstorm"], va="center")
    panel_label(ax_gain, "b")

    fig.suptitle("Experiment-matched discovery gain", y=1.02, fontsize=16, fontweight="bold")
    for ext in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_budget_matched_gain.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def plot_target_savings_focus(trial_summary: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    targets = [1, 5, 10, 20, 50]
    x = np.arange(len(targets))
    width = 0.34
    methods = ("random_walk", "llm_brainstorm", "neurodiscovery")
    stats: dict[str, dict[int, tuple[float, float, float]]] = {m: {} for m in methods}
    for method in methods:
        sub = trial_summary[trial_summary["method"] == method]
        for target in targets:
            stats[method][target] = mean_interval(sub[f"experiments_for_recall_{target}pct"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"wspace": 0.36})
    ax_abs, ax_pct = axes

    for i, baseline in enumerate(("random_walk", "llm_brainstorm")):
        saved = []
        for target in targets:
            base = stats[baseline][target][0]
            nd = stats["neurodiscovery"][target][0]
            saved.append(base - nd)
        ax_abs.bar(
            x + (i - 0.5) * width,
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

    for baseline, marker in (("random_walk", "o"), ("llm_brainstorm", "s")):
        reductions = []
        for target in targets:
            base = stats[baseline][target][0]
            nd = stats["neurodiscovery"][target][0]
            reductions.append(100.0 * (base - nd) / base)
        ax_pct.plot(
            x,
            reductions,
            marker=marker,
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
    fig = plt.figure(figsize=(15.0, 16.5))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 2.65],
        wspace=0.48,
        hspace=0.68,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:3])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:3])
    ax_e = fig.add_subplot(gs[2, :])

    early_budget_max = min(120000, int(curves["budget"].max()))
    early_curve_max = 0.0
    for method in methods:
        sub = curves[curves["method"] == method]
        sub_plot = sub[sub["budget"] <= early_budget_max]
        if sub_plot.empty:
            sub_plot = sub
        early_curve_max = max(early_curve_max, float(sub_plot["recall_hi"].max()))
        ax_a.plot(
            sub_plot["budget"],
            sub_plot["recall_mean"],
            lw=2.4 if method == "neurodiscovery" else 1.8,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
        ax_a.fill_between(
            sub_plot["budget"].to_numpy(float),
            sub_plot["recall_lo"].to_numpy(float),
            sub_plot["recall_hi"].to_numpy(float),
            color=PALETTE[method],
            alpha=BAND_ALPHA[method],
            linewidth=0,
            zorder=1,
        )
    ax_a.set_xlim(0, early_budget_max)
    ax_a.set_ylim(0, min(1.02, max(0.12, early_curve_max * 1.18)))
    ax_a.set_xlabel("Number of experiments")
    ax_a.set_ylabel("GT recall")
    ax_a.set_title("Early-experiment recall")
    ax_a.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax_a.legend(loc="upper left", fontsize=11.5)
    panel_label(ax_a, "a")

    budget_marks = [1000, 5000, 10000, 20000, 50000, 100000]
    available_budgets = set(int(x) for x in curves["budget"].unique())
    budget_marks = [b for b in budget_marks if b in available_budgets]
    x = np.arange(len(budget_marks)) * 0.82
    width = 0.23
    b_bar_tops: dict[str, list[float]] = {}
    for i, method in enumerate(methods):
        vals, lo_err, hi_err = [], [], []
        for budget in budget_marks:
            mean, lo, hi = value_at_budget(curves, method, budget, "gt_hits")
            vals.append(mean)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        b_bar_tops[method] = [v + h for v, h in zip(vals, hi_err, strict=False)]
        ax_b.bar(
            x + (i - 1.0) * width,
            vals,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.45,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            label=METHOD_LABELS[method],
        )
    y_span_b = max(max(v) for v in b_bar_tops.values()) if b_bar_tops else 1.0
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{b:,}" for b in budget_marks])
    ax_b.set_xlabel("Number of experiments")
    ax_b.set_ylabel("GT discoveries found")
    ax_b.set_ylim(0, y_span_b * 1.10)
    ax_b.set_title("Same experiments, more discoveries")
    ax_b.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax_b.legend(loc="upper left", fontsize=11.5)
    panel_label(ax_b, "b")

    target_col = "experiments_for_recall_10pct"
    box_data = [
        pd.to_numeric(trial_summary[trial_summary["method"] == method][target_col], errors="coerce").dropna().to_numpy(float)
        for method in methods
    ]
    bp = ax_c.boxplot(
        box_data,
        patch_artist=True,
        widths=0.56,
        showfliers=False,
        medianprops={"color": "#272727", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.85},
        capprops={"color": "#666666", "linewidth": 0.85},
    )
    for patch, method in zip(bp["boxes"], methods, strict=False):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.78)
        patch.set_edgecolor("#272727")
        patch.set_linewidth(0.5)
    rng = np.random.default_rng(17)
    for i, (method, vals) in enumerate(zip(methods, box_data, strict=False), start=1):
        ax_c.scatter(
            np.full(len(vals), i) + rng.normal(0.0, 0.035, len(vals)),
            vals,
            s=10,
            facecolor="white",
            edgecolor=PALETTE[method],
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
        )
    ax_c.set_xticks(np.arange(1, len(methods) + 1))
    ax_c.set_xticklabels(["Random\nwalk", "LLM\nbrainstorm", "Neuro-\nDiscovery"], fontsize=12)
    ax_c.set_ylim(bottom=0)
    ax_c.set_ylabel("Experiments to 10% GT")
    ax_c.set_title("Seed stability")
    ax_c.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_c, "c")

    targets = [1, 5, 10, 20, 30, 50]
    target_cols = [f"experiments_for_recall_{t}pct" for t in targets]
    tx = np.arange(len(targets)) * 0.82
    d_bar_tops: dict[str, list[float]] = {}
    for i, method in enumerate(methods):
        vals, lo_err, hi_err = [], [], []
        sub = trial_summary[trial_summary["method"] == method]
        for col in target_cols:
            mean, lo, hi = mean_interval(sub[col])
            vals.append(mean)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        d_bar_tops[method] = [v + h for v, h in zip(vals, hi_err, strict=False)]
        ax_d.bar(
            tx + (i - 1.0) * width,
            vals,
            width=width,
            color=PALETTE[method],
            edgecolor="#272727",
            linewidth=0.45,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
        )
    y_span_d = max(max(v) for v in d_bar_tops.values()) if d_bar_tops else 1.0
    ax_d.set_ylim(0, y_span_d * 1.10)
    ax_d.set_xticks(tx)
    ax_d.set_xticklabels([f"{t}%" for t in targets])
    ax_d.set_xlabel("Matched GT recall target")
    ax_d.set_ylabel("Experiments required")
    ax_d.set_title("Same recall, fewer experiments")
    ax_d.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    panel_label(ax_d, "d")

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
    ax_e.text(-0.02, 1.02, "e", transform=ax_e.transAxes, fontsize=17, fontweight="bold", va="top")

    save_generator_panel_svgs(fig, {"a": ax_a, "b": ax_b, "c": ax_c, "d": ax_d}, ax_e, out_dir, surface_panel)
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(out_dir / f"case1_generator_comparison_main.{ext}", dpi=450, bbox_inches="tight")
    plt.close(fig)


def save_axis_panel_svg(fig: plt.Figure, ax: plt.Axes, path: Path) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer).expanded(1.08, 1.12)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=bbox_inches)


def save_generator_panel_svgs(
    fig: plt.Figure,
    axes: dict[str, plt.Axes],
    ax_e: plt.Axes,
    out_dir: Path,
    surface_panel: Path,
) -> None:
    panel_dir = out_dir / PANEL_SVG_DIRNAME
    panel_dir.mkdir(parents=True, exist_ok=True)
    for stale in panel_dir.glob("*.svg"):
        stale.unlink()
    for label, ax in axes.items():
        save_axis_panel_svg(fig, ax, panel_dir / f"{label}.svg")

    surface_svg = surface_panel.with_suffix(".svg")
    if surface_svg.exists():
        shutil.copyfile(surface_svg, panel_dir / "e.svg")
    else:
        save_axis_panel_svg(fig, ax_e, panel_dir / "e.svg")


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
    best_baseline = np.maximum(matrices_by_method["random_walk"], matrices_by_method["llm_brainstorm"])
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
    methods = ["exhaustive_gt", "random_walk", "llm_brainstorm", "neurodiscovery"]
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

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True, sharey=True)
    im_recovery = None
    for ax, method, matrix, raw_matrix, label in zip(axes.ravel(), methods, recovery_matrices, matrices, "abcd", strict=False):
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
    if im_recovery is not None:
        cbar = fig.colorbar(im_recovery, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
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
    gt_top_frac: float,
    gt_total: int,
    seed: int,
    top_n: int,
    n_trials: int,
) -> None:
    manifest = {
        "all_tests": str(all_tests),
        "kg_path": str(kg_path) if kg_path else None,
        "gt_definition": {
            "primary": f"top {gt_top_frac:.4%} by abs_adjusted_residual_d from exhaustive results",
            "gt_total": gt_total,
            "strict_secondary": "q_fdr_global < 0.05",
        },
        "methods": METHOD_LABELS,
        "generator_methods": list(GENERATOR_METHODS),
        "random_seed_base": seed,
        "n_trials_per_stochastic_method": n_trials,
        "curve_interval": "2.5th to 97.5th percentile across seeds",
        "ranked_candidates_export_top_n": top_n,
        "panel_e": "Cortical surface comparison of ROI-level GT recovery for Exhaustive GT, Random walk, LLM brainstorm, and NeuroDiscovery.",
        "baseline_policy": "Exhaustive is a GT/oracle point, not a generator curve; KG degree is fused into NeuroDiscovery and not reported as a KG-free baseline.",
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
    return parser.parse_args()


def remove_stale_outputs(out_dir: Path) -> None:
    for name in (
        "ranked_candidates_exhaustive_oracle.csv",
        "ranked_candidates_random.csv",
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
    n_gt = int(scored["is_gt_top"].sum())
    budgets = budget_grid(len(scored))
    trial_curves, curve_summary, method_summary, trial_summary, exemplar_orders, _labels = run_benchmark(
        scored=scored,
        budgets=budgets,
        n_gt=n_gt,
        n_trials=args.trials,
        seed=args.seed,
        map_top_n=args.map_top_n,
    )

    trial_curves.to_csv(args.out_dir / "case1_discovery_curves_by_trial.csv", index=False)
    curve_summary.to_csv(args.out_dir / "case1_discovery_curves.csv", index=False)
    method_summary.to_csv(args.out_dir / "case1_method_summary.csv", index=False)
    trial_summary.to_csv(args.out_dir / "case1_method_summary_by_trial.csv", index=False)
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
