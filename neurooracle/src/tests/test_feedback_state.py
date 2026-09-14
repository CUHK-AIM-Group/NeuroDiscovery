from neurooracle.src.feedback_state import (
    CONTRADICTED,
    EXECUTION_FAILED,
    INCONCLUSIVE,
    SUPPORTED,
    FeedbackRecord,
    FeedbackState,
)
from neurooracle.src.hypothesis_engine import Hypothesis
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.schema import ConceptNode, Edge


def make_hypothesis(hid: str, disease: str, region: str, feature: str) -> Hypothesis:
    return Hypothesis(
        id=hid,
        source_id=disease,
        target_id=f"{region}|{feature}",
        confidence_score=0.8,
        evidence_score=0.8,
        novelty_score=0.8,
        testability_score=0.8,
        metadata={
            "candidate_tuple": {
                "disease_id": disease,
                "region_id": region,
                "feature_id": feature,
                "feature_family": feature.split(":")[0],
                "feature_modality": "fMRI",
                "atlas_name": "AAL",
            }
        },
    )


def test_supported_exact_downweights_repeat_but_boosts_similar():
    supported = make_hypothesis("h1", "D:AD", "ROI:1", "fc:degree")
    similar = make_hypothesis("h2", "D:AD", "ROI:1", "fc:strength")
    record = FeedbackState.record_from_hypothesis(supported, "supported", reason="validated")
    state = FeedbackState([FeedbackRecord.from_dict(record)])

    exact_adj = state.score(supported)
    similar_adj = state.score(similar)

    assert exact_adj.exact_supported
    assert exact_adj.multiplier < 1.0
    assert similar_adj.supported_similarity > 0
    assert similar_adj.multiplier > 1.0


def test_only_scientific_contradiction_penalizes_similar_hypotheses():
    contradicted = make_hypothesis("h1", "D:AD", "ROI:1", "fc:degree")
    failed = make_hypothesis("h2", "D:MDD", "ROI:2", "alff:mean")
    similar_contra = make_hypothesis("h3", "D:AD", "ROI:1", "fc:strength")
    similar_failed = make_hypothesis("h4", "D:MDD", "ROI:2", "alff:sd")
    contradiction_state = FeedbackState([
        FeedbackRecord.from_dict(
            FeedbackState.record_from_hypothesis(contradicted, "contradicted", reason="not replicated")
        ),
    ])
    failure_state = FeedbackState([
        FeedbackRecord.from_dict(
            FeedbackState.record_from_hypothesis(failed, "execution_failed", reason="missing covariates")
        ),
    ])

    contra_adj = contradiction_state.score(similar_contra)
    failed_adj = failure_state.score(similar_failed)

    assert contra_adj.contradicted_similarity > 0
    assert contra_adj.multiplier < 1.0
    assert failed_adj.execution_failed_similarity > 0
    assert failed_adj.multiplier == 1.0


def test_inconclusive_overlay_record_is_accepted_without_scientific_penalty():
    candidate = make_hypothesis("h1", "D:AD", "ROI:1", "fc:degree")
    similar = make_hypothesis("h2", "D:AD", "ROI:1", "fc:strength")
    record = FeedbackState.record_from_hypothesis(
        candidate, "inconclusive", reason="not statistically decisive"
    )
    state = FeedbackState([FeedbackRecord.from_dict(record)])

    adjustment = state.score(similar)

    assert adjustment.inconclusive_similarity > 0
    assert adjustment.multiplier == 1.0
    assert adjustment.additive == 0.0


def test_experimental_claim_schema_loads_candidate_and_assertion_endpoints():
    raw = {
        "status": "supported",
        "candidate_id": "candidate:1",
        "candidate_tuple": {"gene_pathway": "APOE", "imaging_phenotype": "hippocampus"},
        "assertions": [
            {"subject_id": "GENE:APOE", "object_id": "IM:hippocampus"}
        ],
    }

    record = FeedbackRecord.from_dict(raw)

    assert record.hypothesis_id == "candidate:1"
    assert record.source_id == "GENE:APOE"
    assert record.target_id == "IM:hippocampus"


def test_node_priority_uses_only_informative_scientific_feedback():
    state = FeedbackState(
        [
            FeedbackRecord(SUPPORTED, source_id="A", target_id="B"),
            FeedbackRecord(INCONCLUSIVE, source_id="C", target_id="D"),
            FeedbackRecord(EXECUTION_FAILED, source_id="E", target_id="F"),
            FeedbackRecord(CONTRADICTED, source_id="G", target_id="H"),
        ]
    )

    assert state.node_priority("A") > 0
    assert state.node_priority("B") > 0
    assert state.node_priority("C") == 0
    assert state.node_priority("E") == 0
    assert state.node_priority("G") < 0


def test_supported_path_gives_mediator_a_weaker_positive_priority():
    state = FeedbackState(
        [
            FeedbackRecord(
                SUPPORTED,
                source_id="A",
                target_id="B",
                path_node_ids=("A", "M", "B"),
            )
        ]
    )

    assert 0 < state.node_priority("M") < state.node_priority("A")


def test_feedback_record_loads_path_nodes_for_generation_mutation():
    record = FeedbackRecord.from_dict(
        {
            "status": "supported",
            "source_id": "A",
            "target_id": "B",
            "path_node_ids": ["A", "M", "B"],
        }
    )

    assert record.path_node_ids == ("A", "M", "B")


def test_supported_feedback_changes_the_next_generation_anchor_order():
    graph = KnowledgeGraph()
    for node_id in ("A", "B", "T1", "T2"):
        graph.add_concept(
            ConceptNode(
                id=node_id,
                preferred_name=node_id,
                domain_tags=["gene" if node_id in {"A", "B"} else "disease"],
            )
        )
    graph.add_edge(Edge("A", "T1", "associated_with"))
    graph.add_edge(Edge("B", "T1", "associated_with"))
    graph.add_edge(Edge("B", "T2", "associated_with"))
    engine = HypothesisEngine(graph)
    engine.feedback_state = FeedbackState(
        [FeedbackRecord(SUPPORTED, source_id="A", target_id="T1")]
    )

    assert engine._sample_domain_nodes("gene", 1) == ["A"]


def test_dynamic_seed_sampling_rotates_the_legal_core_across_rounds():
    graph = KnowledgeGraph()
    for index in range(8):
        graph.add_concept(
            ConceptNode(
                id=f"GENE:{index}",
                preferred_name=f"gene {index}",
                domain_tags=["gene"],
            )
        )
    engine = HypothesisEngine(graph)
    engine.configure_dynamic_generation(enabled=True, exploration_round=0)
    first = engine._sample_domain_nodes(
        "gene", 4, random_seed=1, diversity_fraction=0.0
    )
    engine.configure_dynamic_generation(enabled=True, exploration_round=1)
    second = engine._sample_domain_nodes(
        "gene", 4, random_seed=1, diversity_fraction=0.0
    )

    assert first == ["GENE:0", "GENE:1", "GENE:2", "GENE:3"]
    assert second[:2] == ["GENE:0", "GENE:1"]
    assert second[2:] == ["GENE:4", "GENE:5"]


# Updated: 2026-08-12 02:28 HKT
