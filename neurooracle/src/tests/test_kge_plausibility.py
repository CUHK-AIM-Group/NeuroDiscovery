"""Tests for the KG plausibility scorer (Phase 4.3).

These tests don't train the real ComplEx model; they use a stub Scorer to
verify path-level logic, schema integration, and skip-existing behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch

from neurooracle.src.kge.complex_scorer import ComplExScorer, _ComplEx
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


def test_complex_top_pair_scores_matches_exhaustive_scoring() -> None:
    scorer = ComplExScorer(dim=2, device="cpu")
    scorer.ent2idx = {"S:1": 0, "S:2": 1, "T:1": 2, "T:2": 3}
    scorer.rel2idx = {"predicts": 0, "associated_with": 1}
    scorer.model = _ComplEx(4, 2, 2)
    with torch.no_grad():
        scorer.model.ent_re.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        )
        scorer.model.ent_im.weight.zero_()
        scorer.model.rel_re.weight.copy_(
            torch.tensor([[1.0, 0.2], [0.2, 1.0]])
        )
        scorer.model.rel_im.weight.zero_()
    scorer.model.eval()

    sources = ["S:1", "S:2", "OOV:S"]
    targets = ["T:1", "T:2", "OOV:T"]
    relations = ["predicts", "associated_with", "OOV:R"]
    expected = []
    for source in sources[:2]:
        for target in targets[:2]:
            score = max(
                scorer.score_batch(
                    [(source, relation, target) for relation in relations[:2]]
                )
            )
            expected.append((score, source, target))
    expected.sort(key=lambda row: (-row[0], row[1], row[2]))

    actual = scorer.top_pair_scores(
        sources,
        relations,
        targets,
        top_k=3,
        source_chunk_size=1,
    )

    assert [(source, target) for _, source, target in actual] == [
        (source, target) for _, source, target in expected[:3]
    ]
    assert [score for score, _, _ in actual] == pytest.approx(
        [score for score, _, _ in expected[:3]]
    )


def test_complex_top_pair_scores_preserves_per_endpoint_quotas() -> None:
    scorer = ComplExScorer(dim=2, device="cpu")
    scorer.ent2idx = {"S:1": 0, "S:2": 1, "T:1": 2, "T:2": 3}
    scorer.rel2idx = {"predicts": 0}
    scorer.model = _ComplEx(4, 1, 2)
    with torch.no_grad():
        scorer.model.ent_re.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        )
        scorer.model.ent_im.weight.zero_()
        scorer.model.rel_re.weight.copy_(torch.tensor([[1.0, 1.0]]))
        scorer.model.rel_im.weight.zero_()
    scorer.model.eval()

    per_source = scorer.top_pair_scores(
        ["S:1", "S:2"],
        ["predicts"],
        ["T:1", "T:2"],
        top_k=4,
        source_chunk_size=1,
        per_source_k=1,
    )
    per_target = scorer.top_pair_scores(
        ["S:1", "S:2"],
        ["predicts"],
        ["T:1", "T:2"],
        top_k=4,
        source_chunk_size=1,
        per_target_k=1,
    )

    assert {(source, target) for _, source, target in per_source} == {
        ("S:1", "T:1"),
        ("S:2", "T:2"),
    }
    assert {(source, target) for _, source, target in per_target} == {
        ("S:1", "T:1"),
        ("S:2", "T:2"),
    }


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


# Updated: 2026-08-12 23:51 HKT - verify global and per-endpoint chunked KGE retrieval.
    assert per == []


def test_global_attestation_weakest_link():
    """Returns min hits across adjacent (from,to) pairs, no quotes."""
    h = _Hyp(path=[
        _Link("A", "alpha", "B", "beta", "rel1"),
        _Link("B", "beta", "C", "gamma", "rel2"),
    ])
    seen_queries: list[str] = []
    counts = {"alpha AND beta": 8, "beta AND gamma": 3}

    def fake_count(q: str) -> int:
        seen_queries.append(q)
        return counts.get(q, 0)

    score, hits, query = global_attestation(h, fake_count, saturation_hits=10)
    assert hits == 3
    assert score == pytest.approx(0.3)
    assert query == "beta AND gamma"
    assert set(seen_queries) == {"alpha AND beta", "beta AND gamma"}
    # bare phrases — no double-quotes
    assert '"' not in query


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
