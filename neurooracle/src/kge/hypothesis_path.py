"""Turn a hypothesis candidate into a measurable structural/GNN score.

`neurooracle.src.novelty_policy` expects every reviewed candidate to carry a
`structural_score` and a `gnn_path_score` in [0, 1]. AutoResearch `idea` mode
produces natural-language hypotheses whose only structured field is `source_ids`
(PMIDs), so those two numbers cannot be read off directly — and inventing them
would be worse than reporting the gap.

This module is the bridge. It maps a candidate's *declared* KG triples onto the
trained link-predictor vocabulary, scores each edge, and reduces the edges to the
same shapes the frozen development scorer used:

  gnn_path_score   geometric mean of per-edge link scores, with a 0.7x penalty
                   when the weakest edge scores below 0.3 (identical in shape to
                   `plausibility.local_plausibility`, so the two are comparable)
  structural_score confidence^0.20 * traceability^0.20 * graph_only_novelty^0.25
                   * testability^0.35, every component floored at 0.01

The contract is deliberately fail-closed. If any required input is missing — no
declared triples, an endpoint or relation outside the trained vocabulary, no
model, no edge confidence, no provenance fraction — the result is
`measured: False` with `None` scores and a plain-language `disclosure`. A caller
must then fall back to the explicit floor and disclose it; it must not treat an
absent score as a low score, and it must not fill the hole with a plausible-looking
number. Nothing here asserts that a mapped hypothesis is true, novel, or
first-reported.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Protocol, Sequence


class LinkScorer(Protocol):
    """The subset of `base.Scorer` this module needs (no torch import required)."""

    ent2idx: Mapping[str, int]
    rel2idx: Mapping[str, int]

    def score_batch(self, triples: list[tuple[str, str, str]]) -> list[float]:
        ...


# Component weights and floor are copied from the frozen development scorer
# (`full_nd.structural_score`) so a score produced here is on the same scale as
# the historical one. Do not tune them per call site.
STRUCTURAL_WEIGHTS = {"confidence": 0.20, "traceability": 0.20,
                      "graph_only_novelty": 0.25, "testability": 0.35}
COMPONENT_FLOOR = 0.01
WEAK_LINK_THRESHOLD = 0.3
WEAK_LINK_PENALTY = 0.7

TRIPLE_KEYS = ("source", "relation", "target")


class CandidateMappingError(ValueError):
    """A declared triple is malformed. This is a caller error, not a low score."""


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))


def declared_triples(candidate: object) -> list[dict]:
    """Return the candidate's declared KG triples, or an empty list.

    Only the explicitly declared field is used. The hypothesis prose is never
    parsed into triples here: guessing edges from text would manufacture the very
    evidence the score is supposed to weigh.
    """
    if not isinstance(candidate, dict):
        return []
    raw = candidate.get("kg_triples")
    if raw is None:
        raw = candidate.get("triples")
    if not isinstance(raw, list):
        return []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip() for key in TRIPLE_KEYS):
            raise CandidateMappingError(f"kg_triples[{index}] must have nonempty source, relation and target")
    return list(raw)


def _resolve_endpoint(value: str, resolve_entity: Callable[[str], str | None] | None,
                      vocabulary: Mapping[str, int]) -> str | None:
    resolved = resolve_entity(value) if resolve_entity is not None else value
    if not isinstance(resolved, str) or resolved not in vocabulary:
        return None
    return resolved


def _clamp_unit(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return min(1.0, max(0.0, number))


def structural_score(components: Mapping[str, float | None]) -> tuple[float | None, list[str]]:
    """Weighted product of the four structural components.

    Returns `(score, unresolved)`. `score` is `None` when any component was not
    measured, so a partial reading can never masquerade as a complete one.
    """
    unresolved = [name for name in STRUCTURAL_WEIGHTS if components.get(name) is None]
    if unresolved:
        return None, unresolved
    product = 1.0
    for name, weight in STRUCTURAL_WEIGHTS.items():
        product *= max(float(components[name]), COMPONENT_FLOOR) ** weight
    return min(1.0, max(0.0, product)), []


def score_candidate(
    candidate: object,
    *,
    scorer: LinkScorer | None,
    resolve_entity: Callable[[str], str | None] | None = None,
    edge_confidence: Callable[[str, str, str], float | None] | None = None,
    pair_support: Callable[[str, str], int] | None = None,
    provenance_fraction: float | None = None,
    testability: float | None = None,
) -> dict:
    """Score one candidate's declared triples, or report why it cannot be scored.

    Every input is optional because every one of them can be genuinely absent at
    call time. Missing inputs produce `measured: False`; they never produce a
    fabricated number.
    """
    triples = declared_triples(candidate)
    unresolved: list[str] = []
    if not triples:
        unresolved.append("no declared kg_triples")
    if scorer is None:
        unresolved.append("no trained link predictor loaded")

    resolved: list[tuple[str, str, str]] = []
    if triples and scorer is not None:
        for item in triples:
            subject = _resolve_endpoint(item["source"], resolve_entity, scorer.ent2idx)
            target = _resolve_endpoint(item["target"], resolve_entity, scorer.ent2idx)
            relation = item["relation"]
            if subject is None:
                unresolved.append(f"endpoint not in graph vocabulary: {item['source']}")
            if target is None:
                unresolved.append(f"endpoint not in graph vocabulary: {item['target']}")
            if relation not in scorer.rel2idx:
                unresolved.append(f"relation not in trained vocabulary: {relation}")
            if subject is not None and target is not None and relation in scorer.rel2idx:
                resolved.append((subject, relation, target))

    gnn: float | None = None
    if resolved and len(resolved) == len(triples):
        per_edge = [_clamp_unit(value) for value in scorer.score_batch(resolved)]
        if len(per_edge) != len(resolved):
            unresolved.append("link predictor returned the wrong number of edge scores")
        elif any(value is None for value in per_edge):
            unresolved.append("link predictor returned a non-finite score")
        else:
            gnn = _geometric_mean(per_edge)
            if min(per_edge) < WEAK_LINK_THRESHOLD:
                gnn *= WEAK_LINK_PENALTY

    confidence: float | None = None
    if resolved and len(resolved) == len(triples) and edge_confidence is not None:
        readings = [edge_confidence(*triple) for triple in resolved]
        if any(value is None for value in readings):
            unresolved.append("missing KG edge confidence")
        else:
            confidence = _geometric_mean([max(COMPONENT_FLOOR, min(1.0, float(v))) for v in readings])
    elif resolved:
        unresolved.append("no KG edge-confidence lookup")

    graph_only_novelty: float | None = None
    if resolved and pair_support is not None:
        endpoints = (resolved[0][0], resolved[-1][2])
        count = pair_support(*endpoints)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            unresolved.append("pair-support count was unavailable")
        else:
            graph_only_novelty = (0.5 * (1 - 1 / len(resolved))
                                  + 0.5 / (1 + math.log1p(count)))
    elif resolved:
        unresolved.append("no pair-support lookup")

    traceability = _clamp_unit(provenance_fraction)
    if traceability is None and resolved:
        unresolved.append("no provenance fraction")
    test = _clamp_unit(testability)
    if test is None and resolved:
        unresolved.append("no testability reading")

    struct, struct_unresolved = structural_score({
        "confidence": confidence,
        "traceability": traceability,
        "graph_only_novelty": graph_only_novelty,
        "testability": test,
    })
    unresolved.extend(name for name in struct_unresolved if name not in unresolved)

    measured = gnn is not None and struct is not None
    return {
        "measured": measured,
        "gnn_path_score": gnn,
        "structural_score": struct,
        "edge_scores": [
            {"source": s, "relation": r, "target": t,
             "score": per_edge[i] if gnn is not None else None}
            for i, (s, r, t) in enumerate(resolved)
        ],
        "components": {
            "gnn_path": gnn, "confidence": confidence, "traceability": traceability,
            "graph_only_novelty": graph_only_novelty, "testability": test,
        },
        "unresolved": unresolved,
        "disclosure": (
            "structural/GNN scores measured from declared KG triples"
            if measured else
            "structural/GNN scores not measured: " + "; ".join(unresolved)
        ),
        "asserts_truth_or_novelty": False,
    }


def iter_endpoints(candidates: Iterable[object]) -> Iterable[str]:
    """Every endpoint name a resolver must handle for these candidates."""
    for candidate in candidates:
        for item in declared_triples(candidate):
            yield item["source"]
            yield item["target"]
