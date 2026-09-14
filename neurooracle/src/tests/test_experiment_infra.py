"""Regression tests for the compact experiment-infrastructure vocabularies."""

import json

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.ingestion.experiment_infra import (
    CANONICAL_MODALITIES,
    ML_MODELS,
    SPATIAL_REFERENCES,
    SUPPORTED_ATLASES,
    ingest_experiment_infrastructure,
    _build_model_node,
    _build_modality_node,
    _build_spatial_reference_node,
)
from neurooracle.scripts.migrate_infrastructure_taxonomy_v1 import (
    LEGACY_MODEL_NAMES,
    LEGACY_SPATIAL_NAMES,
    desired_model_edges,
    graph_metadata,
    rewrite_graph,
)
from neurooracle.scripts.streaming_graph_json import iter_concepts, iter_edges


def test_expanded_registry_sizes_and_legacy_constant():
    assert SUPPORTED_ATLASES is SPATIAL_REFERENCES
    assert len(SPATIAL_REFERENCES) == 36
    assert len(ML_MODELS) == 36
    assert all(spec.get("kg_node", True) for spec in ML_MODELS.values())


def test_ingestion_uses_canonical_tag_and_legacy_query_alias():
    kg = KnowledgeGraph()
    stats = ingest_experiment_infrastructure(kg)

    assert stats["spatial_references_added"] == 36
    assert stats["atlases_added"] == 36
    assert stats["models_added"] == 36
    assert len(kg.search_by_domain("spatial_reference")) == 36
    assert len(kg.search_by_domain("atlas")) == 36
    assert all(
        node.domain_tags == ["spatial_reference"]
        for node in kg.search_by_domain("spatial_reference")
    )
    assert len(kg.search_by_domain("ml_model")) == 36
    assert kg.get_concept("MODEL:LogisticRegression") is not None
    assert "logistic regression" in kg.get_concept(
        "MODEL:LogisticRegression"
    ).aliases


def test_namespaced_ids_stay_stable_after_domain_rename():
    kg = KnowledgeGraph()
    ingest_experiment_infrastructure(kg)
    node = kg.get_concept("ATLAS:Yeo17")
    assert node is not None
    assert node.domain_tags == ["spatial_reference"]
    assert node.metadata["stable_id_namespace"] == "ATLAS"


def test_streaming_migration_rewrites_and_expands_fixture(tmp_path):
    concepts = {}
    for name in LEGACY_SPATIAL_NAMES:
        node = _build_spatial_reference_node(name, SPATIAL_REFERENCES[name]).to_dict()
        node["domain_tags"] = ["atlas"]
        concepts[node["id"]] = node
    for name in LEGACY_MODEL_NAMES:
        node = _build_model_node(name, ML_MODELS[name]).to_dict()
        concepts[node["id"]] = node
    for name, spec in CANONICAL_MODALITIES.items():
        node = _build_modality_node(name, spec).to_dict()
        concepts[node["id"]] = node

    expected_edges = desired_model_edges()
    legacy_model_ids = {f"MODEL:{name}" for name in LEGACY_MODEL_NAMES}
    edges = [
        edge for pair, edge in expected_edges.items() if pair[0] in legacy_model_ids
    ]
    edges.append(
        {
            "source_id": "MODALITY:fMRI",
            "target_id": "MODALITY:sMRI",
            "relation_type": "supports_modality",
            "source": "fixture_non_model_edge",
            "confidence": 1.0,
            "evidence_ref": "",
            "metadata": {},
        }
    )
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text(
        json.dumps(
            {
                "metadata": {
                    "stats": {
                        "n_concepts": len(concepts),
                        "n_edges": len(edges),
                        "domains": {"atlas": 16, "ml_model": 8, "modality": 12},
                        "sources": {"experiment_infra": 64},
                        "relations": {"supports_modality": 25},
                        "connected_components": 13,
                    }
                },
                "concepts": concepts,
                "edges": edges,
            }
        ),
        encoding="utf-8",
    )

    result = rewrite_graph(source, target, "2026-09-03T00:00:00+00:00")
    migrated_nodes = dict(iter_concepts(target))
    migrated_edges = list(iter_edges(target))
    stats = graph_metadata(target)["stats"]

    assert result["concepts_added"] == 48
    assert result["edges_added"] == 144
    assert len(migrated_nodes) == len(concepts) + 48
    assert len(migrated_edges) == len(edges) + 144
    assert stats["domains"]["spatial_reference"] == 36
    assert "atlas" not in stats["domains"]
    assert all("atlas" not in node["domain_tags"] for node in migrated_nodes.values())
    assert len(
        {
            (edge["source_id"], edge["target_id"])
            for edge in migrated_edges
            if edge["relation_type"] == "supports_modality"
            and edge["source_id"].startswith("MODEL:")
        }
    ) == 168
