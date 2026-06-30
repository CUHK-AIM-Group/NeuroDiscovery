"""KG-based plausibility scoring (C plan).

Local plausibility = how likely each (s, p, o) edge in a hypothesis path is,
predicted by a trained KG link predictor.

Global attestation = how often the whole path is jointly mentioned in PubMed.

surprise_gap = local_plausibility - global_attestation. Positive gaps locate
hypotheses where every hop is supported by the KG distribution but the joint
chain has not been written down yet.
"""

from .triple_loader import Triple, load_triples_from_kg, split_triples
from .base import Scorer
from .plausibility import (
    local_plausibility,
    global_attestation,
    surprise_gap,
    score_hypothesis,
)
from .specificity import path_specificity, build_degree_map

try:
    from .complex_scorer import ComplExScorer
except ImportError:  # pragma: no cover
    ComplExScorer = None  # type: ignore[assignment]

__all__ = [
    "Triple",
    "Scorer",
    "ComplExScorer",
    "load_triples_from_kg",
    "split_triples",
    "local_plausibility",
    "global_attestation",
    "surprise_gap",
    "score_hypothesis",
    "path_specificity",
    "build_degree_map",
]
