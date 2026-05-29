"""Phase 1 ingester: Neurosynth task -> brain-region forward inference.

For each Cognitive Atlas task / concept node already in the KG that maps to a
Neurosynth term, run MKDA forward inference (uniformity test) over the v0.7
corpus (14371 papers, 507891 coords) and emit COGAT_TASK / COGAT_CONCEPT
-> NN_region 'activates' edges where the per-ROI mean z exceeds threshold.

Why forward inference (uniformity, not association):
  Forward inference asks "given studies tagged with term X, where do they
  consistently report activation?". This is the canonical Neurosynth meta-
  analytic statement and matches the semantics of an 'activates' edge:
  task X reliably engages region Y. Reverse inference (association test)
  asks "given activation in Y, how predictive of X?" - that's a separate
  semantic claim and would need a different predicate, so we skip it.

Why MKDA (not ALE / SDM):
  MKDA is what NeuroSynth.org uses, scales to thousands of studies in
  seconds, and ships out-of-the-box in NiMARE. ALE would be marginally
  better for small per-term subsamples (< 50 studies) but term -> region
  edges from < 50-paper terms are noisy anyway and we filter those out.

Term -> Cognitive Atlas matching:
  We require a >= 2-token match between the cogat node name (after
  stop-word stripping: 'task', 'paradigm', 'process', etc.) and a
  Neurosynth term, OR an exact 1-token match for short concept names
  ('attention', 'memory', 'fear'). Looser 1-token-substring matches
  (e.g. 'delayed memory task' -> 'memory') are rejected because the
  same NS term would then map to dozens of cogat tasks and inflate
  edges with weakly-discriminating evidence.

Thresholding:
  Cortical ROIs: |z| >= 3.1 (p < 0.001 uncorrected, Neurosynth's display
    default; consistent with Hansen / AHBA receptor-density thresholds).
  Aseg subcortical: |z| >= 2.3 (loosened by 0.8 because subcortical
    structures are smaller, partial-volume effects shrink mean z, and we
    do not want all subcortical region edges culled).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..graph_manager import KnowledgeGraph
from ..schema import Edge
from .enigma_disease_im import ASEG_ROI_TO_NN, DK_ROI_TO_NN

logger = logging.getLogger(__name__)


_STOP_WORDS = {
    "task", "paradigm", "test", "assessment", "condition", "effect",
    "process", "processing", "function", "functional", "ability",
    "capacity", "related", "during", "of", "the", "a", "an", "to",
    "for", "in", "on", "and", "or", "versus", "vs",
}

_ASEG_ALIAS = {
    "thalamusproper": "thal",
    "caudate":        "caud",
    "putamen":        "put",
    "pallidum":       "pal",
    "accumbensarea":  "accumb",
    "hippocampus":    "hippo",
    "amygdala":       "amyg",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\W_]+", " ", s.lower())).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if t and t not in _STOP_WORDS and len(t) > 2]


def _match_keys(name: str, aliases: Optional[list[str]] = None) -> list[str]:
    """Generate candidate NS-side keys for a cogat node, longest first."""
    cands: list[str] = []
    n = _norm(name)
    if n:
        cands.append(n)
    toks = _tokens(name)
    if toks:
        cands.append(" ".join(toks))
        for k in (3, 2):
            for i in range(len(toks) - k + 1):
                cands.append(" ".join(toks[i : i + k]))
    for a in aliases or []:
        if a:
            cands.append(_norm(a))
            at = _tokens(a)
            if at:
                cands.append(" ".join(at))
    seen, ordered = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    ordered.sort(key=lambda s: -len(s.split()))
    return ordered


def _best_ns_term(name: str, ns_norm: dict[str, str], aliases: Optional[list[str]] = None) -> Optional[str]:
    """Pick the longest matching NS term key for a cogat node, falling back to
    a 1-token full-name match for short cogat nodes (attention, memory)."""
    keys = _match_keys(name, aliases)
    for k in keys:
        if len(k.split()) >= 2 and k in ns_norm:
            return ns_norm[k]
    n = _norm(name)
    if n and len(n.split()) == 1 and n in ns_norm:
        return ns_norm[n]
    return None


def _label_to_nn(label: str, structure: str) -> Optional[str]:
    if structure == "cortex":
        return DK_ROI_TO_NN.get(label)
    key = _ASEG_ALIAS.get(label)
    return ASEG_ROI_TO_NN.get(key) if key else None


def _build_term_index(kg: KnowledgeGraph, ns_terms: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Return ns_term -> list of (cogat_node_id, cogat_name) that map to it.

    A Neurosynth term may anchor multiple cogat nodes (e.g. 'memory' is the
    target for both COGAT_CONCEPT:memory and several memory-task nodes whose
    last salient token is 'memory' but only via 1-token match). The 1-token
    fallback in `_best_ns_term` is name-only and exact, so this still
    discriminates 'memory' (concept) from 'working memory' (different cogat
    node, different NS term).
    """
    ns_norm: dict[str, str] = {}
    for t in ns_terms:
        if len(t) < 3 or t.replace(" ", "").isdigit():
            continue
        ns_norm[_norm(t)] = t
        stripped = " ".join(_tokens(t))
        if stripped and stripped not in ns_norm:
            ns_norm[stripped] = t

    index: dict[str, list[tuple[str, str]]] = {}
    for nid in list(kg._index.keys()):
        if not (nid.startswith("COGAT_TASK:") or nid.startswith("COGAT_CONCEPT:")):
            continue
        concept = kg.get_concept(nid)
        if concept is None:
            continue
        name = concept.preferred_name
        aliases = list(concept.aliases or [])
        term = _best_ns_term(name, ns_norm, aliases)
        if term:
            index.setdefault(term, []).append((nid, name))
    return index


def _per_roi_means(zdata: np.ndarray, ldata: np.ndarray, info) -> dict[str, float]:
    flat_z, flat_l = zdata.ravel(), ldata.ravel()
    bucket: dict[str, list[float]] = {}
    for _, row in info.iterrows():
        lid = int(row["id"])
        m = flat_l == lid
        if not m.any():
            continue
        vals = flat_z[m]
        vals = vals[np.isfinite(vals)]
        if vals.size < 5:
            continue
        bucket.setdefault(row["label"], []).append(float(vals.mean()))
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def ingest_neurosynth_task_im(
    kg: KnowledgeGraph,
    dataset_path: Path,
    cortical_z: float = 3.1,
    subcortical_z: float = 2.3,
    min_studies: int = 80,
    tfidf_threshold: float = 1e-3,
) -> dict:
    """Run MKDA forward inference per Neurosynth term and emit cogat -> NN edges.

    Args:
        kg: KnowledgeGraph with cogat nodes and NN region nodes already loaded.
        dataset_path: Path to the cached NiMARE Dataset pickle.
        cortical_z: |z| threshold for cortical (DK) ROIs (default 3.1, p<0.001).
        subcortical_z: |z| threshold for Aseg subcortical ROIs (default 2.3,
            loosened because partial-volume effects shrink subcortical mean z).
        min_studies: skip terms with fewer than this many tagged papers
            (Neurosynth's own term map cutoff is 80).
        tfidf_threshold: tagging cutoff per study; Neurosynth uses 1e-3.

    Returns:
        Summary dict with per-stage counts.
    """
    from nimare.dataset import Dataset
    from nimare.meta.cbma import MKDAChi2
    import abagen
    import nibabel as nib
    import pandas as pd
    from nilearn.image import resample_to_img

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.warning(f"Neurosynth Dataset not found at {dataset_path}, skipping")
        return {"terms_attempted": 0, "edges_added": 0,
                "reason": "dataset missing"}

    logger.info(f"loading Neurosynth Dataset from {dataset_path} ...")
    ds = Dataset.load(str(dataset_path))
    prefix = "terms_abstract_tfidf__"
    ns_term_cols = [c for c in ds.annotations.columns if c.startswith(prefix)]
    ns_terms = [c[len(prefix):] for c in ns_term_cols]
    logger.info(f"loaded {len(ds.ids)} studies, {len(ns_terms)} terms")

    term_index = _build_term_index(kg, ns_terms)
    logger.info(f"matched {len(term_index)} NS terms to "
                f"{sum(len(v) for v in term_index.values())} cogat nodes")
    if not term_index:
        return {"terms_attempted": 0, "edges_added": 0,
                "reason": "no cogat-NS matches"}

    a = abagen.fetch_desikan_killiany()
    atlas_img = nib.load(a["image"])
    atlas_info = pd.read_csv(a["info"])
    label_data = atlas_img.get_fdata().astype(int)

    # label -> (NN id, structure) once
    label_to_nn: dict[str, tuple[str, str]] = {}
    for _, row in atlas_info.iterrows():
        struct = "cortex" if row["structure"] == "cortex" else "subcortex"
        nn = _label_to_nn(row["label"], struct)
        if nn and kg.has_concept(nn):
            label_to_nn[row["label"]] = (nn, struct)
    logger.info(f"resolved {len(label_to_nn)} DK/Aseg ROI labels to NN nodes in KG")

    edges_added = 0
    terms_attempted = 0
    terms_used = 0
    cogat_nodes_with_edges: set[str] = set()
    skipped_low_studies = 0
    skipped_few_rois = 0

    ann = ds.annotations
    t_start = time.time()

    for ns_term in sorted(term_index.keys()):
        col = prefix + ns_term
        if col not in ann.columns:
            continue
        terms_attempted += 1
        mask = ann[col].values > tfidf_threshold
        n_term = int(mask.sum())
        if n_term < min_studies:
            skipped_low_studies += 1
            continue
        ids_term = ann.loc[mask, "id"].tolist()
        ids_other = ann.loc[~mask, "id"].tolist()
        try:
            mkda = MKDAChi2(kernel__r=10)
            res = mkda.fit(ds.slice(ids_term), ds.slice(ids_other))
            zimg = res.get_map("z_desc-uniformity")
            zimg_r = resample_to_img(zimg, atlas_img, interpolation="linear",
                                     force_resample=True, copy_header=True)
            zdata = zimg_r.get_fdata()
        except Exception as e:
            logger.warning(f"  {ns_term}: MKDA failed ({e}), skip")
            continue

        roi_z = _per_roi_means(zdata, label_data, atlas_info)
        # restrict to labels resolved to KG NN nodes
        roi_z = {k: v for k, v in roi_z.items() if k in label_to_nn}
        if len(roi_z) < 5:
            skipped_few_rois += 1
            continue

        # Pass thresholds: cortex >= cortical_z, subcortex >= subcortical_z
        passes: list[tuple[str, str, float]] = []
        for label, z in roi_z.items():
            nn_id, struct = label_to_nn[label]
            thr = cortical_z if struct == "cortex" else subcortical_z
            if z >= thr:
                passes.append((label, nn_id, float(z)))
        if not passes:
            continue
        terms_used += 1

        cogat_targets = term_index[ns_term]
        n_emit = 0
        for cogat_id, cogat_name in cogat_targets:
            for label, nn_id, z in passes:
                conf = float(min(1.0, max(0.5, z / 8.0)))
                kg.add_edge(Edge(
                    source_id=cogat_id,
                    target_id=nn_id,
                    relation_type="activates",
                    source="Neurosynth_v0.7",
                    confidence=conf,
                    evidence_ref=f"MKDA forward inference on '{ns_term}' "
                                 f"({n_term} studies); ROI '{label}' z={z:.2f}",
                    metadata={
                        "ns_term": ns_term,
                        "n_studies": n_term,
                        "roi_label": label,
                        "z_score": z,
                        "structure": "cortex" if label_to_nn[label][1] == "cortex" else "subcortex",
                    },
                ))
                cogat_nodes_with_edges.add(cogat_id)
                edges_added += 1
                n_emit += 1
        logger.info(f"  {ns_term:30s}: {n_term:>5d} studies, "
                    f"{len(passes)} ROIs >= thr, "
                    f"{len(cogat_targets)} cogat nodes -> {n_emit} edges")

    summary = {
        "terms_attempted": terms_attempted,
        "terms_used":      terms_used,
        "skipped_low_studies": skipped_low_studies,
        "skipped_few_rois":    skipped_few_rois,
        "cogat_nodes_with_edges": len(cogat_nodes_with_edges),
        "edges_added": edges_added,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    logger.info(f"Neurosynth task->IM ingestion complete: {summary}")
    return summary


__all__ = ["ingest_neurosynth_task_im"]
