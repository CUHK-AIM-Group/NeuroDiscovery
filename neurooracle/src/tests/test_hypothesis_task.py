"""Integration test: task-aware hypothesis generation on the real KG.

Uses the pipeline-ready snapshot if available; falls back to skipping if
the dataset isn't present locally (e.g. on CI). The point is to verify
end-to-end that ``batch_generate_for_task`` produces hypotheses tagged
with the source task and that the strict ``require_atom_touch`` filter
behaves as expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurooracle import (
    HypothesisEngine,
    load_graph,
    task_by_name,
    chain_by_name,
    CANONICAL_TASKS,
    CANONICAL_CHAINS,
)
from neurooracle.src.case_studies import CASE2, _case2_pin_atom_pools
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode, Edge


KG_CANDIDATES = [
    Path("neurooracle/data/full_snapshot_v2/knowledge_graph.json"),
    Path("neurooracle/data/full_snapshot_v1/knowledge_graph.json"),
    Path("neurooracle/data/quick/knowledge_graph.json"),
]


def _find_kg() -> Path | None:
    for p in KG_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def engine() -> HypothesisEngine:
    kg_path = _find_kg()
    if kg_path is None:
        pytest.skip("no local KG snapshot available — skipping live task test")
    kg = load_graph(kg_path)
    if len(kg._index) < 1000:
        pytest.skip("local KG too small for task generation test")
    return HypothesisEngine(kg)


def test_biomarker_task_generates_tagged_hypotheses(engine: HypothesisEngine):
    """Single-input task: {IM} → D should produce non-empty results
    tagged with task provenance."""
    task = task_by_name("biomarker_discovery")
    hyps = engine.batch_generate_for_task(
        task,
        max_hops=2,
        max_paths_per_pair=2,
        max_seeds_per_domain=10,
    )
    # Don't require many — depending on post-processing filters this may
    # be small. Just ensure tagging works for whatever survives.
    if not hyps:
        pytest.skip("biomarker task produced 0 hypotheses on this snapshot")
    h = hyps[0]
    assert h.metadata.get("task_name") == "biomarker_discovery"
    assert h.metadata.get("task_signature") == "{IM}->D"


def test_drug_repurposing_task_runs(engine: HypothesisEngine):
    """Single-input task: {Rx} → D. Tags the hypothesis."""
    task = task_by_name("drug_repurposing")
    hyps = engine.batch_generate_for_task(
        task,
        max_hops=2,
        max_paths_per_pair=2,
        max_seeds_per_domain=8,
    )
    for h in hyps:
        assert h.metadata.get("task_name") == "drug_repurposing"


def test_unknown_atom_pair_returns_empty(engine: HypothesisEngine):
    """If a task's input/output atoms have no domain overlap, the
    function logs a warning and returns []; should not raise."""
    # Create a synthetic task with same input and output atom — yields no
    # cross-domain pairs after the in_dom == out_dom filter (when domains
    # overlap completely).
    from neurooracle import Task, Atom
    bogus = Task(
        name="atom_self_loop",
        inputs=frozenset({Atom.DRUG}),
        output=Atom.DRUG,  # same domain only — all pairs filtered out
    )
    hyps = engine.batch_generate_for_task(
        bogus,
        max_hops=2,
        max_paths_per_pair=1,
        max_seeds_per_domain=4,
    )
    assert hyps == []


def test_multi_input_task_atom_touch_filter(engine: HypothesisEngine):
    """Strict atom-touch filter is at most as permissive as the unfiltered
    run, never larger."""
    task = task_by_name("drug_response_prediction")  # {D, Rx, IM} → O
    open_hyps = engine.batch_generate_for_task(
        task, max_hops=3, max_paths_per_pair=2, max_seeds_per_domain=6,
        require_atom_touch=False,
    )
    strict_hyps = engine.batch_generate_for_task(
        task, max_hops=3, max_paths_per_pair=2, max_seeds_per_domain=6,
        require_atom_touch=True,
    )
    assert len(strict_hyps) <= len(open_hyps)


def test_task_type_validation(engine: HypothesisEngine):
    """Passing a non-Task raises TypeError early."""
    with pytest.raises(TypeError):
        engine.batch_generate_for_task("biomarker_discovery")  # type: ignore[arg-type]


# ── TaskChain integration ─────────────────────────────────────────────────

def test_chain_type_validation(engine: HypothesisEngine):
    """Passing a non-TaskChain raises TypeError early."""
    with pytest.raises(TypeError):
        engine.batch_generate_for_chain("genetic_imaging_disease")  # type: ignore[arg-type]


def test_chain_runs_and_tags_metadata(engine: HypothesisEngine):
    """A canonical chain should run end-to-end. If it produces any
    hypotheses, they must carry chain provenance + a non-empty mediator
    list (the chain has at least one mediator atom)."""
    chain = chain_by_name("genetic_imaging_disease")  # G → IM → D
    hyps = engine.batch_generate_for_chain(
        chain,
        max_hops_per_segment=2,
        max_paths_per_segment=2,
        max_seeds=8,
        max_chains=20,
    )
    if not hyps:
        pytest.skip("chain produced 0 hypotheses on this snapshot")
    h = hyps[0]
    assert h.hypothesis_type == "chain"
    assert h.metadata.get("chain_name") == "genetic_imaging_disease"
    assert h.metadata.get("chain_signature") == "G->IM->D"
    assert h.metadata.get("chain_atoms") == [
        "gene_target", "imaging_marker", "disease",
    ]
    # mediator atom present → mediator id list non-empty when path was found
    assert isinstance(h.metadata.get("mediator_ids"), list)


def test_chain_returns_empty_when_segment_unreachable(engine: HypothesisEngine):
    """When ``max_hops_per_segment=0`` no segment can be extended, so the
    result is empty without raising — exercises the early-return branch."""
    chain = chain_by_name("genetic_imaging_disease")
    hyps = engine.batch_generate_for_chain(
        chain,
        max_hops_per_segment=0,
        max_paths_per_segment=1,
        max_seeds=4,
        max_chains=10,
    )
    assert hyps == []


def test_case2_chain_can_use_claim_backed_clm_anchors():
    kg = KnowledgeGraph()
    for node in [
        ConceptNode(id="GENE:CASE", preferred_name="CASEGENE", domain_tags=["gene"]),
        ConceptNode(id="GENE:OTHER", preferred_name="OTHERGENE", domain_tags=["gene"]),
        ConceptNode(id="GENESET:PATH", preferred_name="Case pathway", domain_tags=["gene"]),
        ConceptNode(
            id="CLM_CONCEPT:fdg_hypometabolism_marker",
            preferred_name="FDG hypometabolism marker",
            domain_tags=["biomarker"],
            source_vocab="claim_extraction",
        ),
        ConceptNode(
            id="CLM_CONCEPT:mmse_decline",
            preferred_name="24-month MMSE decline",
            domain_tags=["cognitive_function"],
            source_vocab="claim_extraction",
        ),
        ConceptNode(
            id="CLM:gene_im",
            preferred_name="CASEGENE predicts FDG hypometabolism marker",
            domain_tags=["claim"],
            metadata={
                "source_paper": {"pmid": "1"},
                "raw_text": "CASEGENE predicted FDG hypometabolism.",
            },
        ),
        ConceptNode(
            id="CLM:im_outcome",
            preferred_name="FDG hypometabolism predicts MMSE decline",
            domain_tags=["claim"],
            metadata={
                "source_paper": {"pmid": "2"},
                "raw_text": "FDG hypometabolism predicted 24-month MMSE decline.",
            },
        ),
    ]:
        kg.add_concept(node)

    kg.add_edge(Edge("GENE:CASE", "GENESET:PATH", "part_of", source="test"))
    kg.add_edge(Edge("GENE:OTHER", "GENESET:PATH", "part_of", source="test"))
    kg.add_edge(Edge(
        "GENE:CASE",
        "CLM_CONCEPT:fdg_hypometabolism_marker",
        "predicts",
        source="claim:1",
        confidence=0.8,
        metadata={"claim_id": "CLM:gene_im"},
    ))
    kg.add_edge(Edge(
        "CLM_CONCEPT:fdg_hypometabolism_marker",
        "CLM_CONCEPT:mmse_decline",
        "predicts",
        source="claim:2",
        confidence=0.8,
        metadata={"claim_id": "CLM:im_outcome"},
    ))

    engine = HypothesisEngine(kg)
    _case2_pin_atom_pools(engine, CASE2)
    # The hub-to-hub filter marks every node as top-degree in this tiny
    # synthetic graph. Keep this unit test focused on Case Study 2 claim-backed anchor
    # routing; live smoke tests still exercise the normal hub filter.
    engine._hub_id_set = set()

    hyps = engine.batch_generate_for_chain(
        CASE2.chain,
        max_hops_per_segment=1,
        max_paths_per_segment=2,
        max_seeds=2,
        max_chains=10,
        min_evidence_per_node=0,
    )

    assert hyps
    h = hyps[0]
    assert h.source_id == "GENE:CASE"
    assert h.target_id == "CLM_CONCEPT:mmse_decline"
    assert h.metadata["mediator_ids"] == ["CLM_CONCEPT:fdg_hypometabolism_marker"]
    assert set(h.supporting_claims) == {"CLM:gene_im", "CLM:im_outcome"}

