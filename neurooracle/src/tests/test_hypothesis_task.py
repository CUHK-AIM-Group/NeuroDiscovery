"""Integration test: task-aware hypothesis generation on the real KG.

Uses the current published graph if available; falls back to skipping if
the dataset isn't present locally (e.g. on CI). The point is to verify
end-to-end that ``batch_generate_for_task`` produces hypotheses tagged
with the source task and that the strict ``require_atom_touch`` filter
behaves as expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurooracle import (
    Atom,
    HypothesisEngine,
    load_graph,
    task_by_name,
    chain_by_name,
    CANONICAL_TASKS,
    CANONICAL_CHAINS,
)
from neurooracle.src.case_studies import (
    CASE2,
    _case2_endpoint_atom_roles,
    _case2_pin_atom_pools,
)
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.feedback_state import FeedbackRecord, FeedbackState, SUPPORTED
from neurooracle.src.hypothesis_engine import Hypothesis, HypothesisLink
from neurooracle.src.schema import ConceptNode, Edge
from neurooracle.src.graph_paths import current_graph_path


def _find_kg() -> Path | None:
    try:
        return current_graph_path()
    except FileNotFoundError:
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


def test_formal_adverse_event_task_is_not_penalized_for_missing_imaging() -> None:
    engine = HypothesisEngine.__new__(HypothesisEngine)
    engine._index = {}
    engine.feedback_state = None
    hypothesis = Hypothesis(
        id="HYP:ADVERSE:UNIT",
        hypothesis_type="claim_bridge",
        source_id="CLM_CONCEPT:FLUOXETINE",
        source_name="fluoxetine",
        target_id="CLM_CONCEPT:AKATHISIA",
        target_name="severe akathisia",
        path=[
            HypothesisLink(
                from_id="CLM_CONCEPT:FLUOXETINE",
                from_name="fluoxetine",
                to_id="CLM_CONCEPT:SEROTONIN",
                to_name="serotonin signalling",
                relation_type="modulates",
                confidence=0.8,
                claim_id="CLM:1",
                source_paper={"pmid": "1"},
            ),
            HypothesisLink(
                from_id="CLM_CONCEPT:SEROTONIN",
                from_name="serotonin signalling",
                to_id="CLM_CONCEPT:AKATHISIA",
                to_name="severe akathisia",
                relation_type="predicts",
                confidence=0.8,
                claim_id="CLM:2",
                source_paper={"pmid": "2"},
            ),
        ],
        confidence_score=0.8,
        novelty_score=0.8,
        evidence_score=0.8,
        testability_score=0.15,
        composite_score=0.2,
        metadata={
            "task_name": "adverse_event_prediction",
            "source_atoms": ["drug"],
            "target_atoms": ["outcome"],
            "source_paper_keys": ["1", "2"],
        },
    )

    engine._apply_task_testability_floor(
        hypothesis, task_by_name("adverse_event_prediction")
    )

    assert hypothesis.testability_score == pytest.approx(0.60)
    assert "imaging modality not required" in hypothesis.testability_reason
    assert hypothesis.composite_score > 0.2


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


def test_multi_input_atom_touch_requires_distinct_node_bindings() -> None:
    kg = KnowledgeGraph()
    kg.add_concept(
        ConceptNode(
            id="AMBIGUOUS",
            preferred_name="ambiguous disease imaging marker",
            domain_tags=["disease", "imaging_feature"],
        )
    )
    kg.add_concept(
        ConceptNode(
            id="OUTCOME",
            preferred_name="longitudinal outcome",
            domain_tags=["clinical_outcome"],
        )
    )
    engine = HypothesisEngine(kg)
    hypothesis = Hypothesis(
        source_id="AMBIGUOUS",
        source_name="ambiguous disease imaging marker",
        target_id="OUTCOME",
        target_name="longitudinal outcome",
        path=[
            HypothesisLink(
                from_id="AMBIGUOUS",
                from_name="ambiguous disease imaging marker",
                to_id="OUTCOME",
                to_name="longitudinal outcome",
                relation_type="predicts",
                confidence=0.8,
            )
        ],
    )
    task = task_by_name("prognosis")

    assert engine._path_touches_atoms(hypothesis, task.inputs) is False


def test_multi_input_atom_touch_records_distinct_bindings() -> None:
    kg = KnowledgeGraph()
    kg.add_concept(
        ConceptNode(
            id="DISEASE",
            preferred_name="schizophrenia",
            domain_tags=["disease"],
        )
    )
    kg.add_concept(
        ConceptNode(
            id="MARKER",
            preferred_name="hippocampal volume",
            domain_tags=["imaging_feature"],
        )
    )
    kg.add_concept(
        ConceptNode(
            id="OUTCOME",
            preferred_name="relapse",
            domain_tags=["clinical_outcome"],
        )
    )
    engine = HypothesisEngine(kg)
    hypothesis = Hypothesis(
        source_id="DISEASE",
        source_name="schizophrenia",
        target_id="OUTCOME",
        target_name="relapse",
        path=[
            HypothesisLink(
                from_id="DISEASE",
                from_name="schizophrenia",
                to_id="MARKER",
                to_name="hippocampal volume",
                relation_type="is_associated_with",
                confidence=0.8,
            ),
            HypothesisLink(
                from_id="MARKER",
                from_name="hippocampal volume",
                to_id="OUTCOME",
                to_name="relapse",
                relation_type="predicts",
                confidence=0.8,
            ),
        ],
    )
    task = task_by_name("prognosis")

    assert engine._path_touches_atoms(hypothesis, task.inputs) is True
    assert hypothesis.metadata["input_atom_order"] == ["disease", "imaging_marker"]
    assert hypothesis.metadata["input_entity_ids"] == ["DISEASE", "MARKER"]
    assert len(set(hypothesis.metadata["input_entity_ids"])) == 2


def test_multi_input_atom_touch_prefers_concrete_marker_over_region() -> None:
    kg = KnowledgeGraph()
    kg.add_concept(
        ConceptNode(
            id="DISEASE",
            preferred_name="schizophrenia",
            domain_tags=["disease"],
        )
    )
    kg.add_concept(
        ConceptNode(
            id="REGION",
            preferred_name="anterior cingulate cortex",
            domain_tags=["neuroanatomy"],
        )
    )
    kg.add_concept(
        ConceptNode(
            id="IM:FC",
            preferred_name="default-mode anterior-cingulate connectivity",
            domain_tags=["connectivity"],
        )
    )
    kg.add_concept(
        ConceptNode(
            id="OUTCOME",
            preferred_name="remission",
            domain_tags=["clinical_outcome"],
        )
    )
    engine = HypothesisEngine(kg)
    hypothesis = Hypothesis(
        source_id="DISEASE",
        source_name="schizophrenia",
        target_id="OUTCOME",
        target_name="remission",
        path=[
            HypothesisLink(
                from_id="DISEASE",
                from_name="schizophrenia",
                to_id="REGION",
                to_name="anterior cingulate cortex",
                relation_type="reduces",
                confidence=0.8,
            ),
            HypothesisLink(
                from_id="REGION",
                from_name="anterior cingulate cortex",
                to_id="IM:FC",
                to_name="default-mode anterior-cingulate connectivity",
                relation_type="has_imaging_feature",
                confidence=0.8,
            ),
            HypothesisLink(
                from_id="IM:FC",
                from_name="default-mode anterior-cingulate connectivity",
                to_id="OUTCOME",
                to_name="remission",
                relation_type="predicts",
                confidence=0.8,
            ),
        ],
    )
    task = task_by_name("prognosis")

    assert engine._path_touches_atoms(hypothesis, task.inputs) is True
    assert hypothesis.metadata["input_entity_ids"] == ["DISEASE", "IM:FC"]


def test_task_type_validation(engine: HypothesisEngine):
    """Passing a non-Task raises TypeError early."""
    with pytest.raises(TypeError):
        engine.batch_generate_for_task("biomarker_discovery")  # type: ignore[arg-type]


def test_seed_anchor_diversity_is_reproducible_and_seed_sensitive() -> None:
    kg = KnowledgeGraph()
    for index in range(20):
        kg.add_concept(
            ConceptNode(
                id=f"D:{index:02d}",
                preferred_name=f"Diagnosis {index}",
                domain_tags=["disease"],
            )
        )
    kg.add_concept(
        ConceptNode(
            id="O:target",
            preferred_name="Outcome target",
            domain_tags=["clinical_outcome"],
        )
    )
    for index in range(20):
        kg.add_edge(
            Edge(
                f"D:{index:02d}",
                "O:target",
                "associated_with",
                source="test",
            )
        )
    local_engine = HypothesisEngine(kg)

    first = local_engine._sample_domain_nodes(
        "disease", 10, random_seed=11, diversity_fraction=0.5
    )
    repeated = local_engine._sample_domain_nodes(
        "disease", 10, random_seed=11, diversity_fraction=0.5
    )
    different = local_engine._sample_domain_nodes(
        "disease", 10, random_seed=12, diversity_fraction=0.5
    )

    assert first == repeated
    assert first[:5] == different[:5]
    assert first[5:] != different[5:]


def test_post_process_records_first_rejection_reason() -> None:
    local_engine = HypothesisEngine(KnowledgeGraph())
    rejected = Hypothesis(
        id="HYP:test",
        source_id="SOURCE",
        source_name="source",
        target_id="TARGET",
        target_name="target",
        path=[],
    )

    assert local_engine.post_process(
        [rejected], min_evidence_per_node=0
    ) == []
    assert local_engine.last_post_process_stats == {
        "too_short": 1,
        "accepted_before_dedup": 0,
        "removed_by_pair_dedup": 0,
        "accepted_final": 0,
        "input": 1,
    }


def test_task_endpoint_contract_rejects_tms_as_a_drug() -> None:
    kg = KnowledgeGraph()
    kg.add_concept(ConceptNode(
        id="INTERVENTION:TMS",
        preferred_name="TMS",
        domain_tags=["drug"],
    ))
    kg.add_concept(ConceptNode(
        id="OUTCOME:HEADACHE",
        preferred_name="headache",
        domain_tags=["treatment_outcome"],
    ))
    local_engine = HypothesisEngine(kg)
    hypothesis = Hypothesis(
        id="HYP:tms-adverse",
        source_id="INTERVENTION:TMS",
        source_name="TMS",
        target_id="OUTCOME:HEADACHE",
        target_name="headache",
        metadata={"task_name": "adverse_event_prediction"},
    )

    assert local_engine.post_process(
        [hypothesis], min_evidence_per_node=0
    ) == []
    assert local_engine.last_post_process_stats["invalid_task_endpoint_contract"] == 1


def test_multi_input_task_contract_cannot_be_disabled() -> None:
    local_engine = HypothesisEngine(KnowledgeGraph())
    incomplete = Hypothesis(
        id="HYP:incomplete-drug-response",
        source_id="CLAIM_ENTITY:drug",
        source_name="fluoxetine",
        target_id="CLAIM_ENTITY:outcome",
        target_name="symptom improvement",
        path=[
            HypothesisLink(
                from_id="CLAIM_ENTITY:drug",
                from_name="fluoxetine",
                to_id="CLAIM_ENTITY:disease",
                to_name="major depressive disorder",
                relation_type="treats",
                confidence=0.8,
            ),
            HypothesisLink(
                from_id="CLAIM_ENTITY:disease",
                from_name="major depressive disorder",
                to_id="CLAIM_ENTITY:outcome",
                to_name="symptom improvement",
                relation_type="predicts",
                confidence=0.8,
            ),
        ],
        metadata={
            "source_atoms": ["drug"],
            "target_atoms": ["outcome"],
            "mediator_ids": ["CLAIM_ENTITY:disease"],
            "mediator_atoms": [["disease"]],
        },
    )
    local_engine.batch_generate = lambda **_kwargs: [incomplete]
    local_engine.post_process = lambda hypotheses: hypotheses

    assert local_engine.batch_generate_for_task(
        task_by_name("drug_response_prediction"),
        require_atom_touch=False,
    ) == []


def test_multi_input_scoped_evidence_graph_covers_every_atom() -> None:
    kg = KnowledgeGraph()
    for node in (
        ConceptNode(
            id="DRUG:FLUOXETINE",
            preferred_name="fluoxetine",
            domain_tags=["drug"],
        ),
        ConceptNode(
            id="D:MDD",
            preferred_name="major depressive disorder",
            domain_tags=["disease"],
        ),
        ConceptNode(
            id="IM:AMYGDALA",
            preferred_name="amygdala reactivity",
            domain_tags=["imaging_feature"],
        ),
        ConceptNode(
            id="OUTCOME:HAMD",
            preferred_name="HAM-D symptom improvement",
            domain_tags=["treatment_outcome"],
        ),
    ):
        kg.add_concept(node)
    rows = (
        (
            "CLM:drug-outcome",
            "DRUG:FLUOXETINE",
            "fluoxetine",
            "DRUG",
            "1001",
        ),
        (
            "CLM:disease-outcome",
            "D:MDD",
            "major depressive disorder",
            "DISEASE",
            "1002",
        ),
        (
            "CLM:imaging-outcome",
            "IM:AMYGDALA",
            "amygdala reactivity",
            "IMAGING_MARKER",
            "1003",
        ),
    )
    for claim_id, source_id, source_name, source_type, pmid in rows:
        kg.add_concept(ConceptNode(
            id=claim_id,
            preferred_name=f"{source_name} predicts HAM-D symptom improvement",
            domain_tags=["claim"],
            metadata={
                "subject_id": source_id,
                "subject_name": source_name,
                "subject_type": source_type,
                "predicate": "predicts",
                "object_id": "OUTCOME:HAMD",
                "object_name": "HAM-D symptom improvement",
                "object_type": "OUTCOME",
                "confidence": 0.8,
                "claim_case_study_ids": ["drug_response_prediction"],
                "source_paper": {"pmid": pmid},
                "raw_text": f"{source_name} predicted HAM-D symptom improvement.",
            },
        ))

    local_engine = HypothesisEngine(kg)
    local_engine.set_task_claim_scope("drug_response_prediction")
    task = task_by_name("drug_response_prediction")
    hypotheses = local_engine._batch_generate_multi_input_task_from_scoped_claims(
        task,
        max_hypotheses=10,
        random_seed=3,
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.hypothesis_type == "multi_input_evidence_graph"
    assert set(hypothesis.metadata["input_atom_order"]) == {
        "disease",
        "drug",
        "imaging_marker",
    }
    assert len(hypothesis.path) == 3
    assert len(set(hypothesis.metadata["source_paper_keys"])) == 3
    assert local_engine._path_touches_atoms(hypothesis, task.inputs)
    assert local_engine.post_process([hypothesis]) == [hypothesis]


def test_multi_input_scoped_evidence_graph_accepts_connected_chain() -> None:
    kg = KnowledgeGraph()
    node_specs = (
        ("DRUG:FLUOXETINE", "fluoxetine", "drug"),
        ("D:MDD", "major depressive disorder", "disease"),
        ("IM:AMYGDALA", "amygdala reactivity", "imaging_feature"),
        ("OUTCOME:HAMD", "HAM-D symptom improvement", "treatment_outcome"),
    )
    for node_id, name, domain in node_specs:
        kg.add_concept(ConceptNode(
            id=node_id,
            preferred_name=name,
            domain_tags=[domain],
        ))
    claim_specs = (
        (
            "CLM:drug-disease", "DRUG:FLUOXETINE", "fluoxetine", "DRUG",
            "treats", "D:MDD", "major depressive disorder", "DISEASE", "2001",
        ),
        (
            "CLM:disease-imaging", "D:MDD", "major depressive disorder", "DISEASE",
            "is_associated_with", "IM:AMYGDALA", "amygdala reactivity",
            "IMAGING_MARKER", "2002",
        ),
        (
            "CLM:imaging-outcome", "IM:AMYGDALA", "amygdala reactivity",
            "IMAGING_MARKER", "predicts", "OUTCOME:HAMD",
            "HAM-D symptom improvement", "OUTCOME", "2003",
        ),
    )
    for spec in claim_specs:
        claim_id, sid, sname, stype, predicate, oid, oname, otype, pmid = spec
        kg.add_concept(ConceptNode(
            id=claim_id,
            preferred_name=f"{sname} {predicate} {oname}",
            domain_tags=["claim"],
            metadata={
                "subject_id": sid,
                "subject_name": sname,
                "subject_type": stype,
                "predicate": predicate,
                "object_id": oid,
                "object_name": oname,
                "object_type": otype,
                "confidence": 0.8,
                "claim_case_study_ids": ["drug_response_prediction"],
                "source_paper": {"pmid": pmid},
                "raw_text": f"{sname} {predicate} {oname}.",
            },
        ))

    local_engine = HypothesisEngine(kg)
    local_engine.set_task_claim_scope("drug_response_prediction")
    task = task_by_name("drug_response_prediction")
    hypotheses = local_engine._batch_generate_multi_input_task_from_scoped_claims(
        task,
        max_hypotheses=10,
        random_seed=3,
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert {
        (link.from_id, link.to_id) for link in hypothesis.path
    } == {
        ("DRUG:FLUOXETINE", "D:MDD"),
        ("D:MDD", "IM:AMYGDALA"),
        ("IM:AMYGDALA", "OUTCOME:HAMD"),
    }
    assert local_engine.post_process([hypothesis]) == [hypothesis]


def test_engine_rejects_claim_edges_with_semantic_endpoint_collisions() -> None:
    kg = KnowledgeGraph()
    for node in [
        ConceptNode(
            id="REGION:PONS",
            preferred_name="Pons",
            domain_tags=["neuroanatomy"],
        ),
        ConceptNode(
            id="TASK:FEAR",
            preferred_name="Fear",
            domain_tags=["emotion"],
        ),
        ConceptNode(
            id="DISEASE:PSYCHOSIS",
            preferred_name="Psychosis",
            domain_tags=["disease"],
        ),
        ConceptNode(
            id="CLM:aligned",
            preferred_name="Pons is associated with psychosis",
            domain_tags=["claim"],
            metadata={
                "subject_id": "REGION:PONS",
                "subject_name": "Pons",
                "subject_type": "BRAIN_REGION",
                "predicate": "associated_with",
                "object_id": "DISEASE:PSYCHOSIS",
                "object_name": "psychosis",
                "object_type": "DISEASE",
            },
        ),
        ConceptNode(
            id="CLM:readout",
            preferred_name="Pons activation is associated with psychosis",
            domain_tags=["claim"],
            metadata={
                "subject_id": "REGION:PONS",
                "subject_name": "pons activation",
                "subject_type": "IMAGING_MARKER",
                "predicate": "associated_with",
                "object_id": "DISEASE:PSYCHOSIS",
                "object_name": "psychosis",
                "object_type": "DISEASE",
                "claim_case_study_ids": ["case1_transdiagnostic"],
            },
        ),
        ConceptNode(
            id="CLM:collided",
            preferred_name="Diminished EEG response predicts negative symptoms",
            domain_tags=["claim"],
            metadata={
                "subject_id": "TASK:FEAR",
                "subject_name": "diminished EEG response amplitude",
                "subject_type": "ELECTROPHYSIOLOGY_MARKER",
                "predicate": "predicts",
                "object_id": "DISEASE:PSYCHOSIS",
                "object_name": "negative symptoms across psychosis probands",
                "object_type": "OUTCOME",
            },
        ),
    ]:
        kg.add_concept(node)

    kg.add_edge(Edge(
        "REGION:PONS",
        "DISEASE:PSYCHOSIS",
        "associated_with",
        source="claim:aligned",
        metadata={"claim_id": "CLM:aligned"},
    ))
    kg.add_edge(Edge(
        "TASK:FEAR",
        "DISEASE:PSYCHOSIS",
        "predicts",
        source="claim:collided",
        metadata={"claim_id": "CLM:collided"},
    ))

    local_engine = HypothesisEngine(kg)

    assert ("REGION:PONS", "DISEASE:PSYCHOSIS") in local_engine._claims_by_endpoints
    assert ("TASK:FEAR", "DISEASE:PSYCHOSIS") not in local_engine._claims_by_endpoints
    scoped_records = local_engine._semantic_claim_records_for_scope(
        "case1_transdiagnostic"
    )
    readout_record = next(
        record for record in scoped_records if record["claim_id"] == "CLM:readout"
    )
    assert readout_record["subject"].name == "pons activation"
    assert readout_record["subject"].entity_id.startswith("CLAIM_ENTITY:")
    assert not readout_record["subject"].uses_canonical_id
    assert local_engine._enrich_path(["REGION:PONS", "DISEASE:PSYCHOSIS"])
    assert local_engine._enrich_path(["TASK:FEAR", "DISEASE:PSYCHOSIS"]) == []
    assert local_engine._semantic_edge_rejections == 1


def test_task_semantic_claim_bridge_requires_cross_paper_continuity() -> None:
    from neurooracle import Task

    kg = KnowledgeGraph()
    for node in [
        ConceptNode(
            id="GENE:APOE",
            preferred_name="APOE",
            domain_tags=["gene"],
        ),
        ConceptNode(
            id="IM:HIPPOCAMPAL_VOLUME",
            preferred_name="hippocampal volume",
            domain_tags=["imaging_feature"],
        ),
        ConceptNode(
            id="OUTCOME:MEMORY_DECLINE",
            preferred_name="memory decline",
            domain_tags=["dataset_variable"],
        ),
        ConceptNode(
            id="CLM:gene_imaging",
            preferred_name="APOE predicts hippocampal volume",
            domain_tags=["claim"],
            metadata={
                "subject_id": "GENE:APOE",
                "subject_name": "APOE",
                "subject_type": "GENE",
                "predicate": "predicts",
                "object_id": "IM:HIPPOCAMPAL_VOLUME",
                "object_name": "hippocampal volume",
                "object_type": "IMAGING_MARKER",
                "confidence": 0.8,
                "claim_case_study_ids": ["case2_pathway_mediation"],
                "source_paper": {"pmid": "1001"},
                "raw_text": "APOE predicted lower hippocampal volume.",
            },
        ),
        ConceptNode(
            id="CLM:imaging_outcome",
            preferred_name="Hippocampal volume predicts memory decline",
            domain_tags=["claim"],
            metadata={
                "subject_id": "IM:HIPPOCAMPAL_VOLUME",
                "subject_name": "hippocampal volume",
                "subject_type": "IMAGING_MARKER",
                "predicate": "predicts",
                "object_id": "OUTCOME:MEMORY_DECLINE",
                "object_name": "memory decline",
                "object_type": "OUTCOME",
                "confidence": 0.9,
                "claim_case_study_ids": ["case2_pathway_mediation"],
                "source_paper": {"pmid": "1002"},
                "raw_text": "Hippocampal volume predicted memory decline.",
            },
        ),
    ]:
        kg.add_concept(node)

    local_engine = HypothesisEngine(kg)
    local_engine.set_task_claim_scope("case2_pathway_mediation")
    task = Task(
        name="semantic_prognosis",
        inputs=frozenset({Atom.GENE_TARGET}),
        output=Atom.OUTCOME,
    )

    hypotheses = local_engine._batch_generate_task_from_scoped_claims(
        task,
        max_hypotheses=10,
        random_seed=7,
    )
    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.source_name == "APOE"
    assert hypothesis.target_name == "memory decline"
    assert hypothesis.metadata["mediator_names"] == ["hippocampal volume"]
    assert hypothesis.path[0].to_id == hypothesis.path[1].from_id
    assert set(hypothesis.metadata["source_paper_keys"]) == {"1001", "1002"}
    assert local_engine.post_process(hypotheses) == hypotheses


def test_dynamic_semantic_generation_adds_two_mediator_feedback_mutation() -> None:
    from neurooracle import Task

    kg = KnowledgeGraph()
    concepts = [
        ConceptNode("DISEASE:AD", "Alzheimer disease", domain_tags=["disease"]),
        ConceptNode("DISEASE:MCI", "mild cognitive impairment", domain_tags=["disease"]),
        ConceptNode(
            "IM:HIPPOCAMPAL_VOLUME",
            "hippocampal volume",
            domain_tags=["imaging_feature"],
        ),
        ConceptNode(
            "DISEASE:NEURODEGENERATIVE",
            "neurodegenerative disorder",
            domain_tags=["disease"],
        ),
        ConceptNode(
            "OUTCOME:DECLINE",
            "longitudinal cognitive decline",
            domain_tags=["dataset_variable"],
        ),
    ]
    for concept in concepts:
        kg.add_concept(concept)

    claims = [
        (
            "CLM:AD_IM",
            "DISEASE:AD",
            "Alzheimer disease",
            "DISEASE",
            "IM:HIPPOCAMPAL_VOLUME",
            "hippocampal volume",
            "IMAGING_MARKER",
            "1001",
        ),
        (
            "CLM:MCI_IM",
            "DISEASE:MCI",
            "mild cognitive impairment",
            "DISEASE",
            "IM:HIPPOCAMPAL_VOLUME",
            "hippocampal volume",
            "IMAGING_MARKER",
            "1002",
        ),
        (
            "CLM:IM_DISEASE",
            "IM:HIPPOCAMPAL_VOLUME",
            "hippocampal volume",
            "IMAGING_MARKER",
            "DISEASE:NEURODEGENERATIVE",
            "neurodegenerative disorder",
            "DISEASE",
            "1003",
        ),
        (
            "CLM:DISEASE_OUTCOME",
            "DISEASE:NEURODEGENERATIVE",
            "neurodegenerative disorder",
            "DISEASE",
            "OUTCOME:DECLINE",
            "longitudinal cognitive decline",
            "OUTCOME",
            "1004",
        ),
    ]
    for claim_id, sid, sname, stype, oid, oname, otype, pmid in claims:
        kg.add_concept(
            ConceptNode(
                id=claim_id,
                preferred_name=f"{sname} predicts {oname}",
                domain_tags=["claim"],
                metadata={
                    "subject_id": sid,
                    "subject_name": sname,
                    "subject_type": stype,
                    "predicate": "predicts",
                    "object_id": oid,
                    "object_name": oname,
                    "object_type": otype,
                    "confidence": 0.9,
                    "claim_case_study_ids": ["prognosis"],
                    "source_paper": {"pmid": pmid},
                    "raw_text": f"{sname} predicted {oname}.",
                },
            )
        )

    supported_path = (
        "DISEASE:AD",
        "IM:HIPPOCAMPAL_VOLUME",
        "DISEASE:NEURODEGENERATIVE",
        "OUTCOME:DECLINE",
    )
    local_engine = HypothesisEngine(kg)
    local_engine.set_task_claim_scope("prognosis")
    local_engine.feedback_state = FeedbackState(
        [
            FeedbackRecord(
                SUPPORTED,
                source_id="DISEASE:AD",
                target_id="OUTCOME:DECLINE",
                path_node_ids=supported_path,
            )
        ]
    )
    local_engine.configure_dynamic_generation(
        enabled=True,
        exploration_round=1,
        excluded_paths=[supported_path],
        path_variants_per_endpoint=2,
        feedback_mutation_fraction=0.5,
    )
    task = Task(
        name="semantic_prognosis",
        inputs=frozenset({Atom.DISEASE}),
        output=Atom.OUTCOME,
    )

    hypotheses = local_engine._batch_generate_task_from_scoped_claims(
        task,
        max_hypotheses=10,
        random_seed=11,
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.source_id == "DISEASE:MCI"
    assert hypothesis.metadata["path_template"] == "two_mediator"
    assert hypothesis.metadata["path_node_ids"] == [
        "DISEASE:MCI",
        "IM:HIPPOCAMPAL_VOLUME",
        "DISEASE:NEURODEGENERATIVE",
        "OUTCOME:DECLINE",
    ]
    assert hypothesis.metadata["feedback_mutation"] is True
    assert hypothesis.metadata["feedback_mutation_affinity"] == 1.0
    assert len(hypothesis.path) == 3


def test_functional_localization_bridge_keeps_task_not_imaging_source() -> None:
    from neurooracle import Task

    kg = KnowledgeGraph()
    concepts = [
        ConceptNode(
            id="TASK:IAPS",
            preferred_name="pleasant IAPS stimuli",
            domain_tags=["visual_stimulus"],
        ),
        ConceptNode(
            id="IM:AMYGDALA_ACTIVATION",
            preferred_name="task-fMRI amygdala activation",
            domain_tags=["imaging_feature"],
        ),
        ConceptNode(
            id="D:ANXIETY",
            preferred_name="anxiety disorder",
            domain_tags=["disease"],
        ),
        ConceptNode(
            id="REGION:PRECUNEUS",
            preferred_name="precuneus",
            domain_tags=["neuroanatomy"],
        ),
    ]
    claims = [
        (
            "CLM:task_region",
            "TASK:IAPS",
            "pleasant IAPS stimuli",
            "STIMULUS",
            "REGION:PRECUNEUS",
            "precuneus",
            "BRAIN_REGION",
            "2001",
        ),
        (
            "CLM:region_imaging",
            "REGION:PRECUNEUS",
            "precuneus",
            "BRAIN_REGION",
            "IM:AMYGDALA_ACTIVATION",
            "task-fMRI amygdala activation",
            "IMAGING_MARKER",
            "2002",
        ),
        (
            "CLM:task_disease",
            "TASK:IAPS",
            "pleasant IAPS stimuli",
            "STIMULUS",
            "D:ANXIETY",
            "anxiety disorder",
            "DISEASE",
            "2003",
        ),
        (
            "CLM:disease_imaging",
            "D:ANXIETY",
            "anxiety disorder",
            "DISEASE",
            "IM:AMYGDALA_ACTIVATION",
            "task-fMRI amygdala activation",
            "IMAGING_MARKER",
            "2004",
        ),
    ]
    for node in concepts:
        kg.add_concept(node)
    for claim_id, sid, sname, stype, oid, oname, otype, pmid in claims:
        kg.add_concept(ConceptNode(
            id=claim_id,
            preferred_name=f"{sname} relates to {oname}",
            domain_tags=["claim"],
            metadata={
                "subject_id": sid,
                "subject_name": sname,
                "subject_type": stype,
                "predicate": (
                    "activates" if claim_id == "CLM:task_region"
                    else "is_associated_with"
                ),
                "object_id": oid,
                "object_name": oname,
                "object_type": otype,
                "claim_case_study_ids": ["functional_localization"],
                "source_paper": {"pmid": pmid},
            },
        ))

    local_engine = HypothesisEngine(kg)
    local_engine.set_task_claim_scope("functional_localization")
    task = Task(
        name="functional_localization",
        inputs=frozenset({Atom.COGNITIVE_TASK}),
        output=Atom.IMAGING_MARKER,
    )
    hypotheses = local_engine._batch_generate_task_from_scoped_claims(
        task,
        max_hypotheses=10,
        random_seed=3,
    )

    assert len(hypotheses) == 1
    assert {hypothesis.source_name for hypothesis in hypotheses} == {
        "pleasant IAPS stimuli"
    }
    assert {hypothesis.target_name for hypothesis in hypotheses} == {
        "task-fMRI amygdala activation",
    }
    assert hypotheses[0].metadata["mediator_names"] == ["precuneus"]
    assert hypotheses[0].metadata["mediator_atoms"] == [["imaging_marker"]]


def test_functional_localization_rejects_mistyped_non_neural_mediator() -> None:
    kg = KnowledgeGraph()
    for node in (
        ConceptNode(
            id="TASK:MEMORY",
            preferred_name="memory encoding task",
            domain_tags=["cognitive_function"],
        ),
        ConceptNode(
            id="CLM_CONCEPT:CANNABIS",
            preferred_name="chronic cannabis use",
            domain_tags=["imaging_feature"],
        ),
        ConceptNode(
            id="IM:TOM",
            preferred_name="theory-of-mind network activation",
            domain_tags=["imaging_feature"],
        ),
    ):
        kg.add_concept(node)
    local_engine = HypothesisEngine(kg)
    hypothesis = Hypothesis(
        hypothesis_type="claim_bridge",
        source_id="TASK:MEMORY",
        source_name="memory encoding task",
        target_id="IM:TOM",
        target_name="theory-of-mind network activation",
        path=[
            HypothesisLink(
                from_id="TASK:MEMORY",
                from_name="memory encoding task",
                to_id="CLM_CONCEPT:CANNABIS",
                to_name="chronic cannabis use",
                relation_type="is_associated_with",
                confidence=0.8,
            ),
            HypothesisLink(
                from_id="CLM_CONCEPT:CANNABIS",
                from_name="chronic cannabis use",
                to_id="IM:TOM",
                to_name="theory-of-mind network activation",
                relation_type="is_associated_with",
                confidence=0.8,
            ),
        ],
        metadata={
            "generation_mode": "scoped_semantic_claim_bridge",
            "mediator_ids": ["CLM_CONCEPT:CANNABIS"],
            "mediator_names": ["chronic cannabis use"],
            "mediator_atoms": [["imaging_marker"]],
        },
    )

    assert not local_engine._is_valid_functional_localization_hypothesis(hypothesis)


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
                "subject_id": "GENE:CASE",
                "subject_name": "CASEGENE",
                "object_id": "CLM_CONCEPT:fdg_hypometabolism_marker",
                "object_name": "FDG hypometabolism marker",
                "predicate": "predicts",
                "confidence": 0.8,
                "metadata": {
                    "subject_type": "GENE_TARGET",
                    "object_type": "IMAGING_MARKER",
                },
                "claim_case_study_ids": ["case2_pathway_mediation"],
                "source_paper": {"pmid": "1"},
                "raw_text": "CASEGENE predicted FDG hypometabolism.",
            },
        ),
        ConceptNode(
            id="CLM:im_outcome",
            preferred_name="FDG hypometabolism predicts MMSE decline",
            domain_tags=["claim"],
            metadata={
                "subject_id": "CLM_CONCEPT:fdg_hypometabolism_marker",
                "subject_name": "FDG hypometabolism marker",
                "object_id": "CLM_CONCEPT:mmse_decline",
                "object_name": "24-month MMSE decline",
                "predicate": "predicts",
                "confidence": 0.8,
                "metadata": {
                    "subject_type": "IMAGING_MARKER",
                    "object_type": "OUTCOME",
                },
                "claim_case_study_ids": ["case2_pathway_mediation"],
                "source_paper": {"pmid": "1"},
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
    assert h.source_name == "CASEGENE"
    assert h.target_name == "24-month MMSE decline"
    assert h.metadata["mediator_names"] == ["FDG hypometabolism marker"]
    assert h.metadata["generation_mode"] == "scoped_claim_first"
    assert set(h.supporting_claims) == {"CLM:gene_im", "CLM:im_outcome"}
    assert engine.rank_hypotheses(hyps)


@pytest.mark.parametrize(
    ("name", "declared_type", "expected"),
    [
        (
            "amygdala activity reduction during placebo response",
            "IMAGING_MARKER",
            {Atom.IMAGING_MARKER},
        ),
        (
            "hippocampal atrophy and cognitive decline",
            "OUTCOME",
            {Atom.OUTCOME},
        ),
        (
            "tau PET burden and cognitive decline",
            "IMAGING_AND_OUTCOME_MARKER",
            set(),
        ),
        ("cortical thickness", "", {Atom.IMAGING_MARKER}),
        ("24-month MMSE decline", "", {Atom.OUTCOME}),
        ("hippocampal atrophy and cognitive decline", "", set()),
        ("paralimbic cortex", "BRAIN_REGION", set()),
        ("paralimbic cortical thickness", "BRAIN_REGION", {Atom.IMAGING_MARKER}),
        ("ZBTB20 and EYA4 tau PET genetic variants", "", set()),
        ("APOE pathway", "PATHWAY", {Atom.GENE_TARGET}),
    ],
)
def test_case2_endpoint_roles_prefer_declared_atom_types(
    name: str,
    declared_type: str,
    expected: set[Atom],
) -> None:
    assert _case2_endpoint_atom_roles(name, declared_type) == expected


def test_case2_scoped_claims_rank_ahead_of_general_claims():
    kg = KnowledgeGraph()
    nodes = [
        ConceptNode(id="GENE:SCOPED", preferred_name="APOE", domain_tags=["gene"]),
        ConceptNode(id="GENE:GENERAL", preferred_name="OTHER", domain_tags=["gene"]),
        ConceptNode(
            id="CUI:scoped_imaging",
            preferred_name="hippocampal volume",
            domain_tags=["gene"],
            source_vocab="claim_extraction",
        ),
        ConceptNode(
            id="CLM_CONCEPT:general_imaging",
            preferred_name="cortical thickness",
            domain_tags=["biomarker"],
            source_vocab="claim_extraction",
        ),
        ConceptNode(
            id="CLM_CONCEPT:scoped_outcome",
            preferred_name="24-month MMSE decline",
            domain_tags=["cognitive_function"],
            source_vocab="claim_extraction",
        ),
        ConceptNode(
            id="CLM_CONCEPT:general_outcome",
            preferred_name="cognitive decline",
            domain_tags=["cognitive_function"],
            source_vocab="claim_extraction",
        ),
        ConceptNode(
            id="CLM:scoped_gi",
            preferred_name="APOE predicts hippocampal volume",
            domain_tags=["claim"],
            metadata={
                "subject_id": "GENE:SCOPED",
                "subject_name": "APOE",
                "object_id": "CUI:scoped_imaging",
                "object_name": "hippocampal volume",
                "metadata": {
                    "subject_type": "GENE_TARGET",
                    "object_type": "IMAGING_MARKER",
                },
                "paper_case_study_ids": ["case2_pathway_mediation"],
                "claim_case_study_ids": ["case2_pathway_mediation"],
                "source_paper": {"pmid": "11"},
                "raw_text": "APOE predicted hippocampal volume.",
            },
        ),
        ConceptNode(
            id="CLM:scoped_io",
            preferred_name="Hippocampal volume predicts MMSE decline",
            domain_tags=["claim"],
            metadata={
                "metadata": {
                    "subject_type": "IMAGING_MARKER",
                    "object_type": "OUTCOME",
                    "paper_case_study_ids": ["case2_pathway_mediation"],
                    "claim_case_study_ids": ["case2_pathway_mediation"],
                },
                "subject_id": "CUI:scoped_imaging",
                "subject_name": "hippocampal volume",
                "object_id": "CLM_CONCEPT:scoped_outcome",
                "object_name": "24-month MMSE decline",
                "source_paper": {"pmid": "11"},
                "raw_text": "Hippocampal volume predicted 24-month MMSE decline.",
            },
        ),
        ConceptNode(
            id="CLM:general_gi",
            preferred_name="OTHER predicts cortical thickness",
            domain_tags=["claim"],
            metadata={
                "subject_id": "GENE:GENERAL",
                "subject_name": "OTHER",
                "object_id": "CLM_CONCEPT:general_imaging",
                "object_name": "cortical thickness",
                "paper_case_study_ids": ["case2_pathway_mediation"],
                "claim_case_study_ids": [],
                "source_paper": {"pmid": "21"},
                "raw_text": "OTHER predicted cortical thickness.",
            },
        ),
        ConceptNode(
            id="CLM:general_io",
            preferred_name="Cortical thickness predicts cognitive decline",
            domain_tags=["claim"],
            metadata={
                "subject_id": "CLM_CONCEPT:general_imaging",
                "subject_name": "cortical thickness",
                "object_id": "CLM_CONCEPT:general_outcome",
                "object_name": "cognitive decline",
                "paper_case_study_ids": ["case2_pathway_mediation"],
                "claim_case_study_ids": [],
                "source_paper": {"pmid": "22"},
                "raw_text": "Cortical thickness predicted cognitive decline.",
            },
        ),
    ]
    for node in nodes:
        kg.add_concept(node)

    for source, target, claim_id in [
        ("GENE:SCOPED", "CUI:scoped_imaging", "CLM:scoped_gi"),
        ("CUI:scoped_imaging", "CLM_CONCEPT:scoped_outcome", "CLM:scoped_io"),
        ("GENE:GENERAL", "CLM_CONCEPT:general_imaging", "CLM:general_gi"),
        ("CLM_CONCEPT:general_imaging", "CLM_CONCEPT:general_outcome", "CLM:general_io"),
    ]:
        kg.add_edge(Edge(
            source,
            target,
            "predicts",
            source=f"claim:{claim_id}",
            confidence=0.8,
            metadata={"claim_id": claim_id},
        ))

    engine = HypothesisEngine(kg)
    _case2_pin_atom_pools(engine, CASE2)

    gene_ranker = engine._chain_atom_rankers[Atom.GENE_TARGET]
    im_filter = engine._chain_atom_filters[Atom.IMAGING_MARKER]
    outcome_filter = engine._chain_atom_filters[Atom.OUTCOME]

    assert gene_ranker("GENE:SCOPED", kg._index["GENE:SCOPED"]) > 0
    assert gene_ranker("GENE:GENERAL", kg._index["GENE:GENERAL"]) == 0
    assert im_filter(
        "CUI:scoped_imaging", kg._index["CUI:scoped_imaging"]
    )
    assert not im_filter(
        "CLM_CONCEPT:general_imaging", kg._index["CLM_CONCEPT:general_imaging"]
    )
    assert outcome_filter(
        "CLM_CONCEPT:scoped_outcome", kg._index["CLM_CONCEPT:scoped_outcome"]
    )
    assert not outcome_filter(
        "CLM_CONCEPT:general_outcome", kg._index["CLM_CONCEPT:general_outcome"]
    )
    assert (
        set(engine._path_ignore_ids) | set(engine._intermediate_only_ignore_ids)
    ) <= engine._chain_forbidden_bridge_ids
    assert engine._chain_require_claim_backed_paths
    assert engine._chain_required_claim_scope == "case2_pathway_mediation"
    assert engine._path_claim_edge_count(
        [
            "GENE:SCOPED",
            "CUI:scoped_imaging",
            "CLM_CONCEPT:scoped_outcome",
        ]
    ) == 2
    assert engine._path_claim_edge_count(
        [
            "GENE:GENERAL",
            "CLM_CONCEPT:general_imaging",
            "CLM_CONCEPT:general_outcome",
        ]
    ) == 0

    engine._hub_id_set = set()
    hypotheses = engine.batch_generate_for_chain(
        CASE2.chain,
        max_hops_per_segment=1,
        max_paths_per_segment=2,
        max_seeds=1,
        max_chains=10,
        min_evidence_per_node=0,
    )
    assert hypotheses
    assert hypotheses[0].source_name == "APOE"
    assert hypotheses[0].metadata["mediator_names"] == ["hippocampal volume"]
    assert hypotheses[0].target_name == "24-month MMSE decline"
    repeated = engine.batch_generate_for_chain(
        CASE2.chain,
        max_hops_per_segment=1,
        max_paths_per_segment=2,
        max_seeds=1,
        max_chains=10,
        min_evidence_per_node=0,
    )
    assert [h.id for h in repeated] == [h.id for h in hypotheses]


def test_temporal_decay_accepts_serialized_publication_years():
    string_year = {"source_paper": {"year": "2025"}}
    float_string_year = {"source_paper": {"year": "2024.0"}}
    invalid_year = {"source_paper": {"year": "unknown"}}

    assert HypothesisEngine.compute_temporal_decay(string_year) == pytest.approx(0.97)
    assert HypothesisEngine.compute_temporal_decay(float_string_year) == pytest.approx(0.94)
    assert HypothesisEngine.compute_temporal_decay(invalid_year) == pytest.approx(0.85)


def test_confidence_helpers_accept_serialized_study_type_lists():
    review_list = {
        "evidence": {"study_type": ["review", "systematic_review"]},
        "source_paper": {"year": "2025"},
    }
    mixed_list = {
        "evidence": {"study_type": ["systematic_review", "cohort"]},
        "source_paper": {"year": "2025"},
    }

    assert HypothesisEngine.compute_temporal_decay(review_list) == pytest.approx(1.0)
    assert HypothesisEngine.compute_temporal_decay(mixed_list) == pytest.approx(0.97)

    engine = HypothesisEngine.__new__(HypothesisEngine)
    engine._claims_by_triple = {
        ("A", "predicts", "B"): [
            {
                "evidence": {"study_type": ["review", "systematic_review"]},
                "source_paper": {"pmid": "review-only"},
            },
            {
                "evidence": {"study_type": ["review", "cohort"]},
                "source_paper": {"pmid": "primary"},
            },
        ]
    }
    claim_meta = {"subject_id": "A", "predicate": "predicts", "object_id": "B"}
    assert engine.compute_frequency_boost(claim_meta) == pytest.approx(1.0)

    review_hypothesis = Hypothesis(
        path=[
            HypothesisLink(
                from_id="A",
                from_name="A",
                to_id="B",
                to_name="B",
                relation_type="predicts",
                confidence=0.8,
                evidence={"study_type": ["review", "systematic_review"]},
            )
        ]
    )
    assert engine._compute_evidence_score(review_hypothesis.path) == pytest.approx(0.3)
    assert engine._has_only_review_evidence(review_hypothesis)


# Updated: 2026-08-12 02:28 HKT

