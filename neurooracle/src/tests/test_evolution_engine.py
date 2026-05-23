"""Tests for the P0 changes to evolution_engine:

  1. ``_convergence_fusion`` — the 6th mutation operator that merges two
     same-target hypotheses into a multi-biomarker joint prediction with
     ``n_independent_paths == 2``.
  2. ``_check_biological_plausibility`` — directionality + predicate-type
     compatibility constraints, integrated into ``_validate``.

Both methods avoid the full ``HypothesisEngine``/``KnowledgeGraph`` stack by
constructing the engine via ``__new__`` and stubbing only the attributes the
methods read.
"""

from __future__ import annotations

import types
from typing import Optional

import pytest

from neurooracle.src.evolution_engine import EvolutionEngine
from neurooracle.src.hypothesis_engine import Hypothesis, HypothesisLink
from neurooracle.src.schema import ConceptNode


# ── helpers ────────────────────────────────────────────────────────────

def _node(nid: str, name: str, *domains: str) -> ConceptNode:
    return ConceptNode(id=nid, preferred_name=name, domain_tags=list(domains))


def _link(
    src_id: str, src_name: str, tgt_id: str, tgt_name: str,
    relation: str = "is_associated_with", confidence: float = 0.8,
) -> HypothesisLink:
    return HypothesisLink(
        from_id=src_id, from_name=src_name,
        to_id=tgt_id, to_name=tgt_name,
        relation_type=relation, confidence=confidence,
    )


def _hyp(
    hid: str, source_id: str, source_name: str,
    target_id: str, target_name: str,
    path: list[HypothesisLink],
    metadata: Optional[dict] = None,
) -> Hypothesis:
    return Hypothesis(
        id=hid,
        source_id=source_id, source_name=source_name,
        target_id=target_id, target_name=target_name,
        path=path,
        confidence_score=0.8, novelty_score=0.5,
        evidence_score=0.6, testability_score=0.7,
        metadata=metadata or {},
    )


def _bare_engine(index: dict[str, ConceptNode]) -> EvolutionEngine:
    """Build an EvolutionEngine without running ``__init__``.

    Only the attributes consumed by the methods under test are set:
    ``_index`` for plausibility, plus ``_current_population`` and a stub
    ``engine`` for convergence_fusion (which calls ``_build_child`` which
    delegates score computation to the engine).
    """
    eng = EvolutionEngine.__new__(EvolutionEngine)
    eng._index = index
    eng._current_population = []

    # Stub HypothesisEngine: _build_child invokes these scoring methods.
    # We return constants so the child-building flow doesn't blow up.
    stub_engine = types.SimpleNamespace(
        _compute_confidence_score=lambda path: 0.8,
        _compute_novelty_score=lambda path: 0.5,
        _compute_evidence_score=lambda path: 0.6,
        _compute_testability_score=lambda path: (0.7, "stub"),
        _generate_explanation=lambda h: "",
        _composite_score=lambda h: 0.7,
    )
    eng.engine = stub_engine
    return eng


# ── biological plausibility ────────────────────────────────────────────

def test_plausibility_biomarker_of_with_gene_source_ok():
    """gene --[is_biomarker_of]--> disease passes."""
    index = {
        "GENE:APOE": _node("GENE:APOE", "APOE", "gene"),
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
    }
    eng = _bare_engine(index)
    h = _hyp("HYP:1", "GENE:APOE", "APOE", "MSH:D000544", "Alzheimer Disease",
             [_link("GENE:APOE", "APOE", "MSH:D000544", "Alzheimer Disease",
                    relation="is_biomarker_of")])
    assert eng._check_biological_plausibility(h) == []


def test_plausibility_biomarker_of_with_neuroanatomy_source_forbidden():
    """neuroanatomy --[is_biomarker_of]--> disease is the marquee
    rule: brain regions are not biomarkers."""
    index = {
        "NN:hippo": _node("NN:hippo", "Hippocampus", "neuroanatomy"),
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
    }
    eng = _bare_engine(index)
    h = _hyp("HYP:2", "NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease",
             [_link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease",
                    relation="is_biomarker_of")])
    violations = eng._check_biological_plausibility(h)
    assert violations
    assert any("forbidden" in v and "neuroanatomy" in v for v in violations)


def test_plausibility_directionality_protein_to_gene_forbidden():
    """protein -> gene is biologically backwards (transcription is gene -> protein)."""
    index = {
        "PROT:1": _node("PROT:1", "TauProtein", "protein"),
        "GENE:MAPT": _node("GENE:MAPT", "MAPT", "gene"),
    }
    eng = _bare_engine(index)
    h = _hyp("HYP:3", "PROT:1", "TauProtein", "GENE:MAPT", "MAPT",
             [_link("PROT:1", "TauProtein", "GENE:MAPT", "MAPT")])
    violations = eng._check_biological_plausibility(h)
    assert any("directionality violation" in v for v in violations)


def test_plausibility_directionality_gene_to_protein_ok():
    """gene -> protein is the canonical transcription direction."""
    index = {
        "GENE:MAPT": _node("GENE:MAPT", "MAPT", "gene"),
        "PROT:1": _node("PROT:1", "TauProtein", "protein"),
    }
    eng = _bare_engine(index)
    h = _hyp("HYP:4", "GENE:MAPT", "MAPT", "PROT:1", "TauProtein",
             [_link("GENE:MAPT", "MAPT", "PROT:1", "TauProtein")])
    assert eng._check_biological_plausibility(h) == []


def test_plausibility_disease_to_brain_region_forbidden():
    """disease -> brain region is forbidden by the directionality table."""
    index = {
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
        "NN:hippo": _node("NN:hippo", "Hippocampus", "neuroanatomy", "brain_region"),
    }
    eng = _bare_engine(index)
    h = _hyp("HYP:5", "MSH:D000544", "Alzheimer Disease", "NN:hippo", "Hippocampus",
             [_link("MSH:D000544", "Alzheimer Disease", "NN:hippo", "Hippocampus")])
    violations = eng._check_biological_plausibility(h)
    assert any("directionality violation" in v for v in violations)


def test_plausibility_empty_path_no_violations():
    eng = _bare_engine({})
    h = _hyp("HYP:0", "A", "A", "B", "B", [])
    assert eng._check_biological_plausibility(h) == []


def test_plausibility_unknown_node_ids_silently_skipped():
    """If a link's endpoint isn't in the index, the link is skipped (not crashed)."""
    eng = _bare_engine({})
    h = _hyp("HYP:6", "X", "X", "Y", "Y",
             [_link("X", "X", "Y", "Y", relation="is_biomarker_of")])
    # No node info → no domain inference → no violation.
    assert eng._check_biological_plausibility(h) == []


# ── convergence fusion ─────────────────────────────────────────────────

def test_convergence_fusion_merges_two_same_target_hypotheses():
    """Two clean hypotheses sharing target T fuse: n_independent_paths == 2,
    source_name contains '+', and partner metadata is recorded."""
    index = {
        "GENE:APOE": _node("GENE:APOE", "APOE", "gene"),
        "GENE:MAPT": _node("GENE:MAPT", "MAPT", "gene"),
        "NN:hippo": _node("NN:hippo", "Hippocampus", "neuroanatomy"),
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
    }
    eng = _bare_engine(index)

    h1 = _hyp(
        "HYP:1", "GENE:APOE", "APOE", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:APOE", "APOE", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
        metadata={"fitness": 0.7},
    )
    h2 = _hyp(
        "HYP:2", "GENE:MAPT", "MAPT", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:MAPT", "MAPT", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
        metadata={"fitness": 0.6},
    )
    eng._current_population = [h1, h2]

    child = eng._convergence_fusion(h1)
    assert child is not None
    assert child.metadata.get("n_independent_paths") == 2
    assert "+" in child.source_name
    assert "APOE" in child.source_name and "MAPT" in child.source_name
    assert child.metadata.get("co_biomarker_id") == "GENE:MAPT"
    assert child.metadata.get("co_biomarker_name") == "MAPT"
    assert child.metadata.get("fusion_partner_id") == "HYP:2"
    assert child.target_id == "MSH:D000544"


def test_convergence_fusion_skips_already_fused_parent():
    """A hypothesis with n_independent_paths > 1 should not be re-fused
    (prevents long 'A + B + C + D' chains)."""
    index = {
        "GENE:APOE": _node("GENE:APOE", "APOE", "gene"),
        "GENE:MAPT": _node("GENE:MAPT", "MAPT", "gene"),
        "NN:hippo": _node("NN:hippo", "Hippocampus", "neuroanatomy"),
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
    }
    eng = _bare_engine(index)

    fused_parent = _hyp(
        "HYP:fused", "GENE:APOE", "APOE + X", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:APOE", "APOE", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
        metadata={"n_independent_paths": 2},
    )
    other = _hyp(
        "HYP:other", "GENE:MAPT", "MAPT", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:MAPT", "MAPT", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
    )
    eng._current_population = [fused_parent, other]

    assert eng._convergence_fusion(fused_parent) is None


def test_convergence_fusion_no_partner_returns_none():
    """If no other hypothesis shares the target, fusion returns None."""
    index = {
        "GENE:APOE": _node("GENE:APOE", "APOE", "gene"),
        "NN:hippo": _node("NN:hippo", "Hippocampus", "neuroanatomy"),
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
    }
    eng = _bare_engine(index)
    h = _hyp(
        "HYP:lonely", "GENE:APOE", "APOE", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:APOE", "APOE", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
    )
    eng._current_population = [h]
    assert eng._convergence_fusion(h) is None


def test_convergence_fusion_picks_highest_fitness_partner():
    """When multiple partners exist, the one with highest fitness is selected."""
    index = {
        "GENE:APOE": _node("GENE:APOE", "APOE", "gene"),
        "GENE:MAPT": _node("GENE:MAPT", "MAPT", "gene"),
        "GENE:PSEN1": _node("GENE:PSEN1", "PSEN1", "gene"),
        "NN:hippo": _node("NN:hippo", "Hippocampus", "neuroanatomy"),
        "MSH:D000544": _node("MSH:D000544", "Alzheimer Disease", "disease"),
    }
    eng = _bare_engine(index)
    h_main = _hyp(
        "HYP:main", "GENE:APOE", "APOE", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:APOE", "APOE", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
        metadata={"fitness": 0.7},
    )
    h_low = _hyp(
        "HYP:low", "GENE:MAPT", "MAPT", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:MAPT", "MAPT", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
        metadata={"fitness": 0.3},
    )
    h_high = _hyp(
        "HYP:high", "GENE:PSEN1", "PSEN1", "MSH:D000544", "Alzheimer Disease",
        [
            _link("GENE:PSEN1", "PSEN1", "NN:hippo", "Hippocampus"),
            _link("NN:hippo", "Hippocampus", "MSH:D000544", "Alzheimer Disease"),
        ],
        metadata={"fitness": 0.9},
    )
    eng._current_population = [h_main, h_low, h_high]

    child = eng._convergence_fusion(h_main)
    assert child is not None
    assert child.metadata.get("co_biomarker_id") == "GENE:PSEN1"
    assert child.metadata.get("fusion_partner_id") == "HYP:high"
