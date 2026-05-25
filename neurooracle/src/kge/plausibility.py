"""Path-level plausibility + global attestation + surprise_gap.

A NeuroOracle Hypothesis is a chain of HypothesisLink (from_id, relation,
to_id, …). The functions below score the chain as a whole and write the
results into hypothesis.metadata.

  kge_score        ∈ [0, 1]   geometric mean of per-edge plausibility,
                              with a 0.7× weak-link penalty when min < 0.3
  kge_attestation  ∈ [0, 1]   normalised PubMed joint-mention count for the
                              path's full node set
  surprise_gap     ∈ [-1, 1]  kge_score - kge_attestation; positive values
                              flag "every hop is locally plausible but the
                              joint chain has not been written down" — the
                              C-plan target region.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .base import Scorer
from .specificity import path_specificity

logger = logging.getLogger(__name__)


WEAK_LINK_THRESHOLD = 0.3
WEAK_LINK_PENALTY = 0.7
ATTESTATION_SATURATION_HITS = 10


def _geometric_mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    log_sum = sum(math.log(max(x, 1e-9)) for x in xs)
    return math.exp(log_sum / len(xs))


def local_plausibility(hypothesis, scorer: Scorer) -> tuple[float, list[float]]:
    """Geometric mean of per-edge KG scores, with weak-link penalty.

    Returns (path_score, per_edge_scores). Same penalty shape as
    hypothesis_engine._composite_score so the two scales are comparable.
    """
    edges = [
        (link.from_id, link.relation_type, link.to_id)
        for link in getattr(hypothesis, "path", [])
    ]
    if not edges:
        return 0.0, []
    per_edge = scorer.score_batch(edges)
    base = _geometric_mean(per_edge)
    if min(per_edge) < WEAK_LINK_THRESHOLD:
        return base * WEAK_LINK_PENALTY, per_edge
    return base, per_edge


def global_attestation(
    hypothesis,
    pubmed_count_fn,
    saturation_hits: int = ATTESTATION_SATURATION_HITS,
) -> tuple[float, int, str]:
    """Pairwise weakest-link attestation across PubMed.

    For each adjacent (from_name, to_name) pair on the path we issue a
    PubMed AND query (no quotes, so MeSH auto-expansion + [All Fields] match
    fire) and take the minimum hit count. Rationale: a chain is only as
    well-documented as its weakest hop; an end-to-end joint AND query
    returns 0 for any chain longer than two whose authors didn't happen
    to write all nodes in one paper, which collapses the signal.

    Returns (normalised_score, weakest_hit_count, debug_query). The
    debug_query is the pair that produced the minimum, useful for spot-
    checking why a path scored low.
    """
    path = getattr(hypothesis, "path", [])
    pairs: list[tuple[str, str]] = []
    for link in path:
        a, b = (link.from_name or "").strip(), (link.to_name or "").strip()
        if a and b and a != b:
            pairs.append((a, b))
    if not pairs:
        return 0.0, 0, ""

    weakest_hits = None
    weakest_query = ""
    for a, b in pairs:
        query = f"{a} AND {b}"
        try:
            hits = int(pubmed_count_fn(query))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("pubmed_count_fn failed for %s: %s", query[:80], exc)
            hits = 0
        if weakest_hits is None or hits < weakest_hits:
            weakest_hits = hits
            weakest_query = query

    weakest_hits = weakest_hits or 0
    norm = min(weakest_hits / saturation_hits, 1.0)
    return norm, weakest_hits, weakest_query


def surprise_gap(local: float, attest: float) -> float:
    """Convenience wrapper: bounded difference."""
    return max(-1.0, min(1.0, local - attest))


def score_hypothesis(
    hypothesis,
    scorer: Scorer,
    pubmed_count_fn=None,
    skip_existing: bool = True,
    concepts: dict | None = None,
    degrees: dict[str, int] | None = None,
) -> dict:
    """Score one hypothesis, write into ``hypothesis.metadata``, return it.

    If ``skip_existing`` and ``hypothesis.metadata['kge_score']`` is set,
    return the existing dict unchanged. This implements the user's "only
    score new hypotheses" requirement.

    If ``pubmed_count_fn`` is None we still compute kge_score but leave
    attestation/gap as None so the caller can decide whether to back-fill.

    ``concepts`` (KG concepts dict) and ``degrees`` (node-id → degree) are
    optional — when provided, ``path_specificity`` populates
    ``meta['specificity_score']`` and ``meta['specificity_issues']`` so
    downstream ranking can drop hub/CLM-noise paths.
    """
    meta = getattr(hypothesis, "metadata", None)
    if meta is None:
        meta = {}
        hypothesis.metadata = meta

    existing = getattr(hypothesis, "kge_score", None)
    if existing is None:
        existing = meta.get("kge_score")
    if skip_existing and existing is not None:
        return {
            "kge_score": meta.get("kge_score"),
            "kge_attestation": meta.get("kge_attestation"),
            "surprise_gap": meta.get("surprise_gap"),
            "specificity_score": meta.get("specificity_score"),
            "kge_model": meta.get("kge_model"),
            "skipped": True,
        }

    local, per_edge = local_plausibility(hypothesis, scorer)
    meta["kge_score"] = local
    meta["kge_per_edge"] = per_edge
    meta["kge_model"] = scorer.name
    if hasattr(hypothesis, "kge_score"):
        hypothesis.kge_score = local

    spec_score, spec_issues = path_specificity(hypothesis, concepts, degrees)
    meta["specificity_score"] = spec_score
    meta["specificity_issues"] = spec_issues
    if hasattr(hypothesis, "specificity_score"):
        hypothesis.specificity_score = spec_score
        hypothesis.specificity_issues = list(spec_issues)

    if pubmed_count_fn is None:
        meta.pop("kge_attestation", None)
        meta.pop("surprise_gap", None)
        if hasattr(hypothesis, "kge_attestation"):
            hypothesis.kge_attestation = None
            hypothesis.surprise_gap = None
        return {
            "kge_score": local,
            "kge_attestation": None,
            "surprise_gap": None,
            "specificity_score": spec_score,
            "kge_model": scorer.name,
            "skipped": False,
        }

    attest, hit_count, query = global_attestation(hypothesis, pubmed_count_fn)
    gap = surprise_gap(local, attest)
    meta["kge_attestation"] = attest
    meta["kge_attestation_hits"] = hit_count
    meta["kge_attestation_query"] = query
    meta["surprise_gap"] = gap
    if hasattr(hypothesis, "kge_attestation"):
        hypothesis.kge_attestation = attest
        hypothesis.surprise_gap = gap
    return {
        "kge_score": local,
        "kge_attestation": attest,
        "surprise_gap": gap,
        "specificity_score": spec_score,
        "kge_model": scorer.name,
        "skipped": False,
    }
