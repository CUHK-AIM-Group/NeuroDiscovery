"""Synthetic fixtures only; no model download, KG read, or provider call."""

import math

import pytest

from neurooracle.src.kge import hypothesis_path as hp


class FakeScorer:
    """A stand-in with the two vocabularies and a score table."""

    def __init__(self, scores=None):
        self.ent2idx = {"A": 0, "B": 1, "C": 2}
        self.rel2idx = {"activates": 0, "binds_to": 1}
        self._scores = scores or {}

    def score_batch(self, triples):
        return [self._scores.get(triple, 0.7) for triple in triples]


def candidate(triples=None, **extra):
    row = {"hypothesis_id": "H1", "hypothesis": "Synthetic.", "source_ids": ["PMID:1"]}
    if triples is not None:
        row["kg_triples"] = triples
    row.update(extra)
    return row


def complete_kwargs(scorer, **overrides):
    kwargs = {
        "scorer": scorer,
        "edge_confidence": lambda s, r, t: 0.9,
        "pair_support": lambda a, b: 2,
        "provenance_fraction": 1.0,
        "testability": 1.0,
    }
    kwargs.update(overrides)
    return kwargs


def test_declared_triples_reads_only_the_explicit_field():
    assert hp.declared_triples(candidate()) == []
    assert hp.declared_triples({"triples": [{"source": "A", "relation": "r", "target": "B"}]}) == [
        {"source": "A", "relation": "r", "target": "B"}]
    # Prose is never parsed into edges; a hypothesis without triples stays unscored.
    assert hp.declared_triples({"hypothesis": "A activates B via C"}) == []


@pytest.mark.parametrize("bad", [
    [{"source": "", "relation": "r", "target": "B"}],
    [{"source": "A", "relation": " ", "target": "B"}],
    [{"source": "A", "relation": "r"}],
    ["A activates B"],
])
def test_malformed_triples_raise_instead_of_being_ignored(bad):
    with pytest.raises(hp.CandidateMappingError):
        hp.declared_triples({"kg_triples": bad})


def test_a_fully_measured_candidate_scores_on_the_legacy_scale():
    scorer = FakeScorer()
    result = hp.score_candidate(candidate([{"source": "A", "relation": "activates", "target": "B"}]),
                                **complete_kwargs(scorer))
    assert result["measured"] is True
    assert result["gnn_path_score"] == pytest.approx(0.7)
    assert 0.0 < result["structural_score"] <= 1.0
    assert result["unresolved"] == []
    assert result["asserts_truth_or_novelty"] is False
    # The strong-edge path takes no weak-link penalty.
    assert result["gnn_path_score"] == pytest.approx(result["components"]["gnn_path"])


def test_weakest_edge_triggers_the_penalty():
    scorer = FakeScorer({("A", "activates", "B"): 0.9, ("B", "binds_to", "C"): 0.1})
    result = hp.score_candidate(
        candidate([{"source": "A", "relation": "activates", "target": "B"},
                   {"source": "B", "relation": "binds_to", "target": "C"}]),
        **complete_kwargs(scorer))
    assert result["measured"] is True
    assert result["gnn_path_score"] == pytest.approx(math.sqrt(0.9 * 0.1) * 0.7)


def test_missing_model_or_triples_yields_none_never_a_number():
    for kwargs, marker in (
        (complete_kwargs(None), "no trained link predictor loaded"),
        (complete_kwargs(FakeScorer()), "no declared kg_triples"),
    ):
        row = candidate() if "no declared" in marker else candidate(
            [{"source": "A", "relation": "activates", "target": "B"}])
        result = hp.score_candidate(row, **kwargs)
        assert result["measured"] is False
        assert result["gnn_path_score"] is None
        assert result["structural_score"] is None
        assert marker in result["unresolved"]
        assert marker in result["disclosure"]


def test_out_of_vocabulary_endpoint_is_disclosed_not_defaulted():
    scorer = FakeScorer()
    result = hp.score_candidate(candidate([{"source": "A", "relation": "activates", "target": "ZZZ"}]),
                                **complete_kwargs(scorer))
    assert result["measured"] is False
    assert result["structural_score"] is None
    assert any("endpoint not in graph vocabulary: ZZZ" in item for item in result["unresolved"])


def test_unknown_relation_is_disclosed():
    scorer = FakeScorer()
    result = hp.score_candidate(candidate([{"source": "A", "relation": "invents", "target": "B"}]),
                                **complete_kwargs(scorer))
    assert result["measured"] is False
    assert any("relation not in trained vocabulary" in item for item in result["unresolved"])


def test_partial_readings_do_not_produce_a_structural_score():
    scorer = FakeScorer()
    result = hp.score_candidate(candidate([{"source": "A", "relation": "activates", "target": "B"}]),
                                **complete_kwargs(scorer, edge_confidence=None))
    assert result["measured"] is False
    assert result["gnn_path_score"] is not None  # the GNN term was measured ...
    assert result["structural_score"] is None    # ... but the product was not
    assert "no KG edge-confidence lookup" in result["unresolved"]


def test_resolver_maps_names_before_the_vocabulary_check():
    scorer = FakeScorer()
    result = hp.score_candidate(candidate([{"source": "alpha", "relation": "activates", "target": "beta"}]),
                                **complete_kwargs(scorer, resolve_entity=lambda name: {"alpha": "A", "beta": "B"}.get(name)))
    assert result["measured"] is True
    assert result["edge_scores"] == [{"source": "A", "relation": "activates", "target": "B", "score": 0.7}]


def test_structural_score_refuses_a_partial_component_set():
    score, unresolved = hp.structural_score(
        {"confidence": 0.9, "traceability": 1.0, "graph_only_novelty": None, "testability": 1.0})
    assert score is None
    assert unresolved == ["graph_only_novelty"]


def test_structural_score_is_the_frozen_weighted_product():
    score, unresolved = hp.structural_score(
        {"confidence": 1.0, "traceability": 1.0, "graph_only_novelty": 1.0, "testability": 1.0})
    assert unresolved == []
    assert score == pytest.approx(1.0)
    # A zeroed component is floored at 0.01 rather than collapsing the product to 0.
    floored, _ = hp.structural_score(
        {"confidence": 0.0, "traceability": 1.0, "graph_only_novelty": 1.0, "testability": 1.0})
    assert floored == pytest.approx(hp.COMPONENT_FLOOR ** hp.STRUCTURAL_WEIGHTS["confidence"])


def test_pair_support_dampens_repeated_relations():
    scorer = FakeScorer()
    rare = hp.score_candidate(candidate([{"source": "A", "relation": "activates", "target": "B"}]),
                              **complete_kwargs(scorer, pair_support=lambda a, b: 1))
    common = hp.score_candidate(candidate([{"source": "A", "relation": "activates", "target": "B"}]),
                                **complete_kwargs(scorer, pair_support=lambda a, b: 500))
    assert rare["structural_score"] > common["structural_score"]


def test_iter_endpoints_covers_both_sides():
    assert list(hp.iter_endpoints([candidate([
        {"source": "A", "relation": "activates", "target": "B"}])])) == ["A", "B"]
