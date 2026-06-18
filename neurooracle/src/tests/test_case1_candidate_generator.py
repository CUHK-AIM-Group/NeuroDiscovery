from __future__ import annotations

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.schema import ConceptNode, Edge


FEATURES = (
    {
        "id": "roi_mean_whole_brain_fc",
        "name": "ROI mean whole-brain FC",
        "family": "seed_fc",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("fc_matrix",),
        "primary": True,
    },
    {
        "id": "roi_alff",
        "name": "ROI ALFF",
        "family": "roi_activity",
        "modality": "fMRI",
        "level": "ROI",
        "requires": ("roi_timeseries",),
        "primary": True,
    },
)


def _toy_engine() -> HypothesisEngine:
    kg = KnowledgeGraph()
    for cid, name in (
        ("D:SCZ", "Schizophrenia"),
        ("D:BD", "Bipolar Disorder"),
        ("D:MDD", "Major Depressive Disorder"),
        ("D:OCD", "Obsessive-Compulsive Disorder"),
    ):
        kg.add_concept(ConceptNode(id=cid, preferred_name=name, domain_tags=["disease"]))
    for cid, name in (
        ("NN:ACC", "Anterior Cingulate Cortex"),
        ("NN:INS", "Insular Cortex"),
    ):
        kg.add_concept(ConceptNode(id=cid, preferred_name=name, domain_tags=["neuroanatomy"]))
    kg.add_concept(
        ConceptNode(
            id="CLM:1",
            preferred_name="claim 1",
            domain_tags=["claim"],
            metadata={
                "raw_text": "Anterior cingulate cortex connectivity is altered in schizophrenia.",
                "evidence": {"study_type": "primary"},
                "source_paper": {"pmid": "1", "year": 2026},
            },
        )
    )
    for disease_id in ("D:SCZ", "D:BD", "D:MDD", "D:OCD"):
        kg.add_edge(
            Edge(
                source_id=disease_id,
                target_id="NN:ACC",
                relation_type="is_associated_with",
                confidence=0.7,
                metadata={"claim_id": "CLM:1"},
            )
        )
    return HypothesisEngine(kg)


def test_case1_candidate_generator_emits_four_methods():
    engine = _toy_engine()
    hypotheses = engine.generate_case1_hypotheses(
        methods=("exhaustive", "random_walk", "llm_brainstorm", "neurodiscovery"),
        disease_names=(
            "Schizophrenia",
            "Bipolar Disorder",
            "Major Depressive Disorder",
            "Obsessive-Compulsive Disorder",
        ),
        atlas_label_names=("Anterior Cingulate Cortex", "Insular Cortex"),
        atlas_label_sources={"Anterior Cingulate Cortex": ("toy_atlas",), "Insular Cortex": ("toy_atlas",)},
        feature_space=FEATURES,
        max_per_method=2,
        random_seed=11,
    )

    methods = {h.metadata["generation_method"] for h in hypotheses}
    assert methods == {"exhaustive", "random_walk", "llm_brainstorm", "neurodiscovery"}
    assert all(h.hypothesis_type == "case1_candidate" for h in hypotheses)
    assert all("direction" not in h.metadata["candidate_tuple"] for h in hypotheses)
    assert all(h.metadata["direction_assumption"].startswith("none") for h in hypotheses)
    assert {len(h.metadata["candidate_tuple"]["diseases"]) for h in hypotheses} == {1}
    assert {h.metadata["total_candidate_space"] for h in hypotheses} == {16}

    neuro = [h for h in hypotheses if h.metadata["generation_method"] == "neurodiscovery"]
    assert neuro
    assert neuro[0].metadata["support_score"] > 0.5
    assert neuro[0].metadata["supporting_claim_count"] == 1
