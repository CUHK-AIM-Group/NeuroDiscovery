from __future__ import annotations

import networkx as nx

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.hypothesis_engine import HypothesisEngine
from neurooracle.src.schema import ConceptNode, Edge


def test_batch_generation_caps_inspected_invalid_paths(monkeypatch) -> None:
    kg = KnowledgeGraph()
    for node in (
        ConceptNode(id="D:seed", preferred_name="Seed", domain_tags=["disease"]),
        ConceptNode(id="X:bridge", preferred_name="Bridge", domain_tags=["claim"]),
        ConceptNode(
            id="O:target",
            preferred_name="Target",
            domain_tags=["treatment_outcome"],
        ),
    ):
        kg.add_concept(node)
    kg.add_edge(Edge("D:seed", "X:bridge", "associated_with", source="test"))
    kg.add_edge(Edge("X:bridge", "O:target", "associated_with", source="test"))

    inspected = 0

    def endless_invalid_paths(*args, **kwargs):
        nonlocal inspected
        for _ in range(10_000):
            inspected += 1
            yield ["D:seed", "X:bridge", "O:target"]

    monkeypatch.setattr(nx, "all_simple_paths", endless_invalid_paths)
    engine = HypothesisEngine(kg)
    hypotheses = engine.batch_generate(
        domain_pairs=[("disease", "treatment_outcome")],
        max_hops=2,
        max_paths_per_pair=1,
        max_seeds_per_domain=1,
        min_evidence_per_node=0,
    )

    assert hypotheses == []
    assert inspected == 513


# Created At: 2026-08-01 18:29 HKT
