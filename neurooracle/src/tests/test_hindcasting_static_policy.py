from __future__ import annotations

import networkx as nx

from neurooracle.src.hindcasting_static_policy import (
    FrozenGraphRelationScorer,
    RELATION_COMPONENT_RAW_FIELDS,
    annotate_relation_aware_payload,
    score_relation_component_rows,
)


def _edge(scope: str | None = None) -> dict:
    metadata = {}
    if scope:
        metadata["claim_case_study_ids"] = [scope]
    return {
        "relation_type": "is_associated_with",
        "confidence": 0.9,
        "metadata": metadata,
    }


def _hypothesis(candidate_id: str, source: str, target: str) -> dict:
    return {
        "id": candidate_id,
        "hypothesis_type": "bridge",
        "source_id": source,
        "source_name": source,
        "target_id": target,
        "target_name": target,
        "path": [
            {
                "from_id": source,
                "from_name": source,
                "to_id": target,
                "to_name": target,
                "relation_type": "is_associated_with",
                "confidence": 0.9,
            }
        ],
        "metadata": {},
    }


def test_relation_aware_policy_prefers_scoped_relation_support():
    graph = nx.DiGraph()
    graph.add_edge("A", "X", **_edge("prognosis"))
    graph.add_edge("B", "X", **_edge("prognosis"))
    graph.add_edge("C", "Y", **_edge())
    graph.add_edge("D", "Y", **_edge())
    payload = {
        "hypotheses": [
            _hypothesis("global-only", "C", "D"),
            _hypothesis("scoped", "A", "B"),
        ]
    }

    ranked = annotate_relation_aware_payload(
        payload,
        graph=graph,
        case_study_id="prognosis",
    )

    assert [row["id"] for row in ranked["hypotheses"]] == ["scoped", "global-only"]
    assert ranked["metadata"]["relation_aware_static_policy"]["uses_future_outcomes"] is False
    assert all(
        field in ranked["hypotheses"][0]["metadata"]
        for field in RELATION_COMPONENT_RAW_FIELDS
    )


def test_relation_component_scoring_is_row_order_invariant():
    rows = [
        {
            "candidate_id": "a",
            RELATION_COMPONENT_RAW_FIELDS[0]: 1.0,
            RELATION_COMPONENT_RAW_FIELDS[1]: 2.0,
            RELATION_COMPONENT_RAW_FIELDS[2]: 3.0,
            RELATION_COMPONENT_RAW_FIELDS[3]: 4.0,
        },
        {
            "candidate_id": "b",
            RELATION_COMPONENT_RAW_FIELDS[0]: 4.0,
            RELATION_COMPONENT_RAW_FIELDS[1]: 3.0,
            RELATION_COMPONENT_RAW_FIELDS[2]: 2.0,
            RELATION_COMPONENT_RAW_FIELDS[3]: 1.0,
        },
    ]
    forward, _ = score_relation_component_rows(rows)
    reverse, _ = score_relation_component_rows(list(reversed(rows)))
    assert dict(zip(("a", "b"), forward, strict=True)) == dict(
        zip(("b", "a"), reverse, strict=True)
    )


def test_multi_input_candidates_score_every_registered_input_node():
    graph = nx.DiGraph()
    graph.add_edge("D", "O", **_edge("prognosis"))
    graph.add_edge("IM", "O", **_edge("prognosis"))
    scorer = FrozenGraphRelationScorer(graph, "prognosis")
    hypothesis = _hypothesis("multi", "D", "O")
    hypothesis["metadata"] = {"input_entity_ids": ["D", "IM"]}
    components = scorer.raw_components(hypothesis)
    assert components[RELATION_COMPONENT_RAW_FIELDS[2]] > 0
    assert components[RELATION_COMPONENT_RAW_FIELDS[3]] > 0
