from __future__ import annotations

import json
from pathlib import Path

from neurooracle.src.kge.triple_loader import (
    Triple,
    load_triples_from_kg,
    split_triples,
)


def test_split_keeps_validation_and_test_vocabulary_in_training() -> None:
    triples = [
        Triple("A", "r", "B"),
        Triple("A", "r", "C"),
        Triple("D", "r", "C"),
        Triple("E", "rare", "F"),
        Triple("G", "rare", "F"),
        Triple("G", "rare", "H"),
    ]
    domains = {entity: "x" for entity in "ABCDEFGH"}
    train, val, test = split_triples(triples, domains, seed=7)

    train_entities = {
        entity
        for triple in train
        for entity in (triple.source_id, triple.target_id)
    }
    train_relations = {triple.relation_type for triple in train}
    for triple in (*val, *test):
        assert triple.source_id in train_entities
        assert triple.target_id in train_entities
        assert triple.relation_type in train_relations
    assert len(train) + len(val) + len(test) == len(triples)


def test_triple_loader_accepts_compact_snapshot_json(tmp_path: Path) -> None:
    graph = {
        "metadata": {},
        "concepts": {
            "A": {"domain_tags": ["gene"]},
            "B": {"domain_tags": ["disease"]},
        },
        "edges": [
            {
                "source_id": "A",
                "target_id": "B",
                "relation_type": "is_associated_with",
                "confidence": 0.9,
            }
        ],
    }
    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text(
        json.dumps(graph, separators=(",", ":")),
        encoding="utf-8",
    )

    triples, domains = load_triples_from_kg(graph_path)

    assert triples == [Triple("A", "is_associated_with", "B")]
    assert domains == {"A": "gene", "B": "disease"}
