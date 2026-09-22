from __future__ import annotations

from collections import defaultdict

from neurooracle.scripts.generate_case_study_frozen_baselines import (
    FrozenEdge,
    FrozenGraphIndex,
)
from neurooracle.scripts.hindcasting_v4r2_adapted_baselines import (
    generate_adapted_hypotheses,
    sequence_pair_overlap,
)
from neurooracle.src.case_studies import case_study_by_name


def _synthetic_index(task_tag: str) -> tuple[FrozenGraphIndex, dict[str, list[FrozenEdge]]]:
    concepts = {}
    names = {}
    atoms = {}
    graph = defaultdict(list)
    claims = defaultdict(list)
    direct_pairs = set()

    source_ids = [f"IMG:{task_tag}:{index}" for index in range(8)]
    target_names = (
        "Major Depressive Disorder",
        "Schizophrenia",
        "Bipolar Disorder",
        "Obsessive-Compulsive Disorder",
        "Anorexia Nervosa",
        "Post-Traumatic Stress Disorder",
        "Anxiety Disorders",
        "Substance Use Disorders",
    )
    target_ids = [f"DIS:{task_tag}:{index}" for index in range(8)]
    mediator_ids = [f"MED:{task_tag}:{index}" for index in range(5)]
    imaging_names = (
        "hippocampal volume",
        "cortical thickness",
        "amygdala volume",
        "default mode network functional connectivity",
        "fractional anisotropy",
        "regional cerebral blood flow",
        "cortical surface area",
        "resting-state functional connectivity",
    )
    for index, node_id in enumerate(source_ids):
        concepts[node_id] = {"preferred_name": f"{imaging_names[index]} {task_tag}"}
        names[node_id] = concepts[node_id]["preferred_name"]
        atoms[node_id] = frozenset({"imaging_marker"})
    for node_id, name in zip(target_ids, target_names):
        concepts[node_id] = {"preferred_name": name}
        names[node_id] = name
        atoms[node_id] = frozenset({"disease"})
    for index, node_id in enumerate(mediator_ids):
        concepts[node_id] = {"preferred_name": f"Shared circuit mechanism {task_tag} {index}"}
        names[node_id] = concepts[node_id]["preferred_name"]
        atoms[node_id] = frozenset({"disease"})

    def add(container, edge):
        container[edge.source_id].append(edge)
        container[edge.target_id].append(edge)

    for s_index, source in enumerate(source_ids):
        for m_index, mediator in enumerate(mediator_ids):
            edge = FrozenEdge(
                source_id=source,
                target_id=mediator,
                relation="associated_with",
                confidence=0.60 + 0.01 * ((s_index + m_index) % 10),
                claim_id=f"C:{task_tag}:SM:{s_index}:{m_index}",
                raw_text=(
                    f"brain imaging biomarker structural functional network {task_tag} "
                    "transdiagnostic diagnostic marker"
                ),
                source_paper={"pmid": f"{task_tag}1{s_index}{m_index}", "year": 2018},
                year=2018,
            )
            add(graph, edge)
            add(claims, edge)
            direct_pairs.add(tuple(sorted((source, mediator))))
    for m_index, mediator in enumerate(mediator_ids):
        for t_index, target in enumerate(target_ids):
            edge = FrozenEdge(
                source_id=mediator,
                target_id=target,
                relation="predicts",
                confidence=0.62 + 0.01 * ((m_index + t_index) % 10),
                claim_id=f"C:{task_tag}:MT:{m_index}:{t_index}",
                raw_text=(
                    f"psychiatric neurological disease subtype cluster prognostic {task_tag}"
                ),
                source_paper={"pmid": f"{task_tag}2{m_index}{t_index}", "year": 2019},
                year=2019,
            )
            add(graph, edge)
            add(claims, edge)
            direct_pairs.add(tuple(sorted((mediator, target))))
    return (
        FrozenGraphIndex(
            concepts=concepts,
            names=names,
            atoms=atoms,
            adjacency=dict(graph),
            direct_pairs=direct_pairs,
        ),
        dict(claims),
    )


def test_sciagents_adapter_is_task_conditioned_and_auditable() -> None:
    index, claims = _synthetic_index("A")
    payload = generate_adapted_hypotheses(
        method="sciagents_adapted",
        case=case_study_by_name("case1_transdiagnostic"),
        index=index,
        graph_adjacency=index.adjacency,
        scoped_claim_adjacency=claims,
        freeze_year=2020,
        seed=3,
        target_count=12,
        pool_size=24,
    )
    assert len(payload["hypotheses"]) == 12
    assert len({(row["source_id"], row["target_id"]) for row in payload["hypotheses"]}) == 12
    assert all(row["metadata"]["task_conditioned"] for row in payload["hypotheses"])
    assert all(set(row["metadata"]["role_scores"]) == {
        "ontologist", "scientist", "critic", "task_fit"
    } for row in payload["hypotheses"])
    assert payload["metadata"]["llm_api_enabled"] is False


def test_openscholar_adapter_has_feedback_retrieval_and_citations() -> None:
    index, claims = _synthetic_index("B")
    payload = generate_adapted_hypotheses(
        method="openscholar_rag_adapted",
        case=case_study_by_name("biomarker_discovery"),
        index=index,
        graph_adjacency=index.adjacency,
        scoped_claim_adjacency=claims,
        freeze_year=2020,
        seed=2,
        target_count=12,
        pool_size=24,
    )
    retrieval = payload["metadata"]["adaptation_audit"]["retrieval"]
    assert retrieval["retrieved_edges"] > 0
    assert retrieval["feedback_policy"] == "deterministic_pseudo_relevance_feedback"
    assert retrieval["feedback_expansion_terms"]
    assert len(payload["hypotheses"]) == 12
    assert all(row["metadata"]["retrieval_citations"] for row in payload["hypotheses"])
    assert all(
        row["metadata"]["retrieval_policy"]
        == "query_retrieve_feedback_reretrieve_fixed_compiler"
        for row in payload["hypotheses"]
    )


def test_sequence_overlap_gate_detects_task_blind_order() -> None:
    row = lambda source, target: {"source_id": source, "target_id": target}
    same = {"hypotheses": [row("A", "B"), row("C", "D")]}
    different = {"hypotheses": [row("A", "B"), row("E", "F")]}
    identical = sequence_pair_overlap(same, same, k=2)
    shifted = sequence_pair_overlap(same, different, k=2)
    assert identical["identical_order"] is True
    assert identical["jaccard"] == 1.0
    assert shifted["identical_order"] is False
    assert shifted["jaccard"] < 1.0
