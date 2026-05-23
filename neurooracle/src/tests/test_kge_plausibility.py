"""Tests for the KG plausibility scorer (Phase 4.3).

These tests don't train the real ComplEx model; they use a stub Scorer to
verify path-level logic, schema integration, and skip-existing behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pytest

from neurooracle.src.kge.base import Scorer
from neurooracle.src.kge.plausibility import (
    WEAK_LINK_PENALTY,
    WEAK_LINK_THRESHOLD,
    global_attestation,
    local_plausibility,
    score_hypothesis,
    surprise_gap,
)


# ── stubs ──────────────────────────────────────────────────────────────


@dataclass
class _Link:
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    relation_type: str


@dataclass
class _Hyp:
    path: list[_Link] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class _StubScorer(Scorer):
    """Returns a fixed score per (s, p, o) lookup; default 0.5."""

    def __init__(self, table: dict[tuple[str, str, str], float], name: str = "stub"):
        self.table = table
        self._name = name
        self.calls: list[tuple[str, str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def score_triple(self, s: str, p: str, o: str) -> float:
        self.calls.append((s, p, o))
        return self.table.get((s, p, o), 0.5)


# ── unit tests ─────────────────────────────────────────────────────────


def test_local_plausibility_geometric_mean():
    """Two strong edges ⇒ geometric mean ≈ √(0.8 × 0.6)."""
    h = _Hyp(path=[
        _Link("A", "Aname", "B", "Bname", "treats"),
        _Link("B", "Bname", "C", "Cname", "is_biomarker_of"),
    ])
    scorer = _StubScorer({
        ("A", "treats", "B"): 0.8,
        ("B", "is_biomarker_of", "C"): 0.6,
    })
    score, per = local_plausibility(h, scorer)
    expected = math.sqrt(0.8 * 0.6)
    assert score == pytest.approx(expected, abs=1e-6)
    assert per == [0.8, 0.6]


def test_local_plausibility_weak_link_penalty():
    """A single edge below 0.3 triggers the 0.7× weak-link penalty."""
    h = _Hyp(path=[
        _Link("A", "Aname", "B", "Bname", "treats"),
        _Link("B", "Bname", "C", "Cname", "is_biomarker_of"),
    ])
    scorer = _StubScorer({
        ("A", "treats", "B"): 0.9,
        ("B", "is_biomarker_of", "C"): 0.2,  # below WEAK_LINK_THRESHOLD
    })
    assert WEAK_LINK_THRESHOLD == 0.3
    score, _ = local_plausibility(h, scorer)
    geo = math.sqrt(0.9 * 0.2)
    assert score == pytest.approx(geo * WEAK_LINK_PENALTY, abs=1e-6)


def test_local_plausibility_empty_path():
    h = _Hyp(path=[])
    scorer = _StubScorer({})
    score, per = local_plausibility(h, scorer)
    assert score == 0.0
    assert per == []


def test_global_attestation_dedupes_path_nodes():
    """The query should AND the unique node names, not edge endpoints."""
    h = _Hyp(path=[
        _Link("A", "alpha", "B", "beta", "rel1"),
        _Link("B", "beta", "C", "gamma", "rel2"),
    ])
    seen_queries: list[str] = []

    def fake_count(q: str) -> int:
        seen_queries.append(q)
        return 5  # half of saturation

    score, hits, query = global_attestation(h, fake_count, saturation_hits=10)
    assert hits == 5
    assert score == pytest.approx(0.5)
    assert seen_queries == [query]
    # alpha + beta + gamma must each appear exactly once
    assert query.count('"alpha"') == 1
    assert query.count('"beta"') == 1
    assert query.count('"gamma"') == 1


def test_global_attestation_saturates():
    """≥ saturation_hits → 1.0, never above."""
    h = _Hyp(path=[_Link("A", "alpha", "B", "beta", "rel")])
    score, _, _ = global_attestation(h, lambda q: 1_000, saturation_hits=10)
    assert score == 1.0


def test_surprise_gap_bounds():
    assert surprise_gap(0.9, 0.1) == pytest.approx(0.8)
    assert surprise_gap(0.1, 0.9) == pytest.approx(-0.8)
    # Saturation
    assert surprise_gap(2.0, 0.0) == 1.0
    assert surprise_gap(-2.0, 0.0) == -1.0


def test_score_hypothesis_writes_metadata():
    h = _Hyp(path=[
        _Link("A", "alpha", "B", "beta", "rel"),
    ])
    scorer = _StubScorer({("A", "rel", "B"): 0.7}, name="stub-v1")
    result = score_hypothesis(h, scorer, pubmed_count_fn=lambda q: 2,
                              skip_existing=True)
    assert h.metadata["kge_score"] == pytest.approx(0.7)
    assert h.metadata["kge_per_edge"] == [0.7]
    assert h.metadata["kge_attestation"] == pytest.approx(0.2)
    assert h.metadata["kge_attestation_hits"] == 2
    assert h.metadata["surprise_gap"] == pytest.approx(0.5)
    assert h.metadata["kge_model"] == "stub-v1"
    assert result["skipped"] is False


def test_score_hypothesis_skip_existing():
    """Already-scored hypotheses are not re-scored when skip_existing=True."""
    h = _Hyp(path=[_Link("A", "alpha", "B", "beta", "rel")])
    h.metadata = {
        "kge_score": 0.42,
        "kge_attestation": 0.1,
        "surprise_gap": 0.32,
        "kge_model": "stub-v0",
    }
    scorer = _StubScorer({("A", "rel", "B"): 0.99}, name="stub-v1")
    n_calls_before = len(scorer.calls)
    result = score_hypothesis(h, scorer, pubmed_count_fn=lambda q: 999,
                              skip_existing=True)
    # Scorer should not have been invoked at all.
    assert len(scorer.calls) == n_calls_before
    assert result["skipped"] is True
    assert h.metadata["kge_score"] == 0.42  # untouched
    assert h.metadata["kge_model"] == "stub-v0"


def test_score_hypothesis_no_pubmed_drops_attestation():
    """Without a pubmed_count_fn, only kge_score is written."""
    h = _Hyp(path=[_Link("A", "alpha", "B", "beta", "rel")])
    scorer = _StubScorer({("A", "rel", "B"): 0.7})
    result = score_hypothesis(h, scorer, pubmed_count_fn=None)
    assert h.metadata["kge_score"] == pytest.approx(0.7)
    assert "kge_attestation" not in h.metadata
    assert "surprise_gap" not in h.metadata
    assert result["surprise_gap"] is None
