from __future__ import annotations

import json

from neurooracle.scripts.canonicalize_manual_claim_ids import (
    build_mapping,
    canonical_claim_id,
    canonical_concept_id,
    extend_mapping_from_extracted,
    graph_claim_rows,
    migrate_graph,
    stage_extracted,
    validate_staged,
)
from neurooracle.scripts.canonicalize_manual_artifact_refs import (
    find_legacy_claim_references,
    migrate_claim_references,
)
from neurooracle.scripts.normalize_manual_residuals import (
    count_manual_residuals,
    normalize_manual_residuals,
)


def _claim_node(claim_id: str, pmid: str) -> dict:
    claim = {
        "id": claim_id,
        "subject_id": "ANCHOR:A",
        "subject_name": "A",
        "predicate": "predicts",
        "object_id": "ANCHOR:B",
        "object_name": "B",
        "source_paper": {"pmid": pmid, "year": 2020},
    }
    return {
        "id": claim_id,
        "preferred_name": "A predicts B",
        "domain_tags": ["claim"],
        "source_vocab": "claim_extraction",
        "metadata": claim,
    }


def test_manual_claim_id_migration_updates_graph_and_archive(tmp_path) -> None:
    legacy_id = "MANUAL:GENMED:B001:1:123"
    graph_only_id = "CLM:graphonly"
    graph = {
        "metadata": {},
        "concepts": {
            "ANCHOR:A": {"id": "ANCHOR:A", "domain_tags": ["biomarker"]},
            "ANCHOR:B": {"id": "ANCHOR:B", "domain_tags": ["disease"]},
            legacy_id: _claim_node(legacy_id, "123"),
            graph_only_id: _claim_node(graph_only_id, "124"),
        },
        "edges": [
            {
                "source_id": legacy_id,
                "target_id": "ANCHOR:A",
                "relation_type": "about",
                "metadata": {"claim_id": legacy_id},
            }
        ],
    }

    mapping = build_mapping(graph["concepts"])
    assert mapping == {legacy_id: canonical_claim_id(legacy_id)}
    migrated, _counts = migrate_graph(graph, mapping)
    new_id = mapping[legacy_id]
    assert legacy_id not in migrated["concepts"]
    assert "legacy_claim_id" not in migrated["concepts"][new_id]["metadata"]
    assert migrated["edges"][0]["source_id"] == new_id
    assert migrated["edges"][0]["metadata"]["claim_id"] == new_id

    source = tmp_path / "claims.jsonl"
    staged = tmp_path / "claims.staged.jsonl"
    source.write_text(
        json.dumps(graph["concepts"][legacy_id]["metadata"]) + "\n",
        encoding="utf-8",
    )
    counts = stage_extracted(source, staged, mapping, graph_claim_rows(migrated))
    assert counts["graph_only_claims_appended_to_extracted"] == 1
    rows = [json.loads(line) for line in staged.read_text(encoding="utf-8").splitlines()]
    renamed = next(row for row in rows if row["id"] == new_id)
    assert "legacy_claim_id" not in renamed
    assert validate_staged(migrated, staged)["graph_claims_missing_from_extracted"] == 0


def test_all_non_clm_mode_includes_other_legacy_claim_ids() -> None:
    legacy_id = "GEN-CSNONE-B1549-001-12531459"
    concepts = {legacy_id: _claim_node(legacy_id, "12531459")}
    assert build_mapping(concepts) == {}
    assert build_mapping(concepts, all_non_clm=True) == {
        legacy_id: canonical_claim_id(legacy_id)
    }


def test_all_manual_concepts_updates_endpoint_references(tmp_path) -> None:
    legacy_anchor = "MANUAL_R2:IMAGING_MARKER:hippocampal_volume"
    claim_id = "CLM:existing"
    graph = {
        "metadata": {},
        "concepts": {
            legacy_anchor: {
                "id": legacy_anchor,
                "preferred_name": "hippocampal volume",
                "domain_tags": ["external"],
                "metadata": {},
            },
            claim_id: {
                **_claim_node(claim_id, "125"),
                "metadata": {
                    **_claim_node(claim_id, "125")["metadata"],
                    "subject_id": legacy_anchor,
                },
            },
        },
        "edges": [
            {
                "source_id": claim_id,
                "target_id": legacy_anchor,
                "relation_type": "about",
                "metadata": {"claim_id": claim_id},
            }
        ],
    }
    mapping = build_mapping(graph["concepts"], all_manual_concepts=True)
    new_anchor = canonical_concept_id(legacy_anchor)
    assert mapping == {legacy_anchor: new_anchor}

    source = tmp_path / "claims.jsonl"
    staged = tmp_path / "claims.staged.jsonl"
    source.write_text(
        json.dumps(graph["concepts"][claim_id]["metadata"]) + "\n",
        encoding="utf-8",
    )
    assert extend_mapping_from_extracted(source, mapping, graph["concepts"]) == {
        "extracted_only_manual_claim_ids_selected": 0,
        "extracted_only_manual_concept_ids_selected": 0,
    }

    migrated, _counts = migrate_graph(graph, mapping)
    assert legacy_anchor not in migrated["concepts"]
    assert "legacy_concept_id" not in migrated["concepts"][new_anchor]["metadata"]
    assert migrated["edges"][0]["target_id"] == new_anchor

    stage_extracted(source, staged, mapping, graph_claim_rows(migrated))
    row = json.loads(staged.read_text(encoding="utf-8"))
    assert row["subject_id"] == new_anchor
    assert validate_staged(
        migrated, staged, all_manual_concepts=True
    )["legacy_manual_concepts_remaining"] == 0


def test_artifact_migration_only_updates_claim_foreign_keys() -> None:
    legacy_id = "MANUAL:CLAIM:B001:1:123:1"
    artifact = {
        "path": [{"claim_id": legacy_id, "raw_text": legacy_id}],
        "supporting_claims": [legacy_id, "CLM:existing"],
        "metadata": {"legacy_claim_id": legacy_id},
    }

    migrated, mapping, changed = migrate_claim_references(artifact)

    assert mapping == {legacy_id: canonical_claim_id(legacy_id)}
    assert changed == 2
    assert migrated["path"][0]["claim_id"] == canonical_claim_id(legacy_id)
    assert migrated["supporting_claims"][0] == canonical_claim_id(legacy_id)
    assert migrated["path"][0]["raw_text"] == legacy_id
    assert migrated["metadata"]["legacy_claim_id"] == legacy_id
    assert find_legacy_claim_references(migrated) == []


def test_manual_residual_normalization_uses_canonical_claim_metadata() -> None:
    legacy_id = "MANUAL:CLAIM:B001:1:123:1"
    canonical_id = canonical_claim_id(legacy_id)
    payload = {
        "subject_type": "MANUAL_TOPIC",
        "object_type": "MANUAL_FINDING",
        "legacy_claim_id": legacy_id,
        "edge": {
            "source": f"claim:{legacy_id}",
            "metadata": {"claim_id": canonical_id},
        },
        "provenance": "manual curation",
    }

    counts = normalize_manual_residuals(payload)

    assert payload["subject_type"] == "TOPIC"
    assert payload["object_type"] == "FINDING"
    assert "legacy_claim_id" not in payload
    assert payload["edge"]["source"] == f"claim:{canonical_id}"
    assert payload["provenance"] == "manual curation"
    assert counts["claim_sources_normalized"] == 1
    assert count_manual_residuals(payload) == {}
