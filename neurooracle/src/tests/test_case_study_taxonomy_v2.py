from __future__ import annotations

import json
from pathlib import Path

from neurooracle.scripts.migrate_case_study_taxonomy_v2 import (
    build_graph_membership_index,
    rewrite_extracted,
    rewrite_graph,
    validate_rewritten_graph,
)
from neurooracle.scripts.migrate_case_study_taxonomy_v2_provenance import (
    apply_provenance_migration,
)
from neurooracle.src.case_studies import list_case_study_names
from neurooracle.src.case_study_scope import (
    claim_case_study_ids_from_dict,
    paper_case_study_ids_from_dict,
)
from neurooracle.src.validation_protocols import HINDCASTING


def _legacy_claim(
    claim_id: str,
    *,
    tasks: list[str],
    scopes: list[str],
) -> dict:
    return {
        "id": claim_id,
        "subject_id": "IM:1",
        "subject_name": "hippocampal volume",
        "predicate": "predicts",
        "object_id": "D:1",
        "object_name": "outcome",
        "source_paper": {
            "pmid": "123",
            "title": "Shared paper",
            "year": 2020,
        },
        "paper_scope": scopes,
        "case3_tasks": tasks,
        "metadata": {
            "paper_scope": scopes,
            "case3_tasks": tasks,
        },
    }


def test_registry_and_hindcasting_are_independent() -> None:
    ids = list_case_study_names()
    assert len(ids) == 17
    assert "case3_hindcasting" not in ids
    assert HINDCASTING.name == "hindcasting"
    assert HINDCASTING.case_study_ids == ids


def test_streaming_migration_preserves_claim_labels_and_unions_paper_labels(
    tmp_path: Path,
) -> None:
    first = _legacy_claim(
        "CLM:1",
        tasks=["brain_age"],
        scopes=["general", "case3"],
    )
    second = _legacy_claim(
        "CLM:2",
        tasks=["prognosis", "transdiagnostic_clustering"],
        scopes=["general", "case1", "case3"],
    )
    graph = {
        "metadata": {"name": "tiny"},
        "concepts": {
            "CLM:1": {
                "id": "CLM:1",
                "preferred_name": "claim one",
                "domain_tags": ["claim"],
                "metadata": first,
            },
            "CLM:2": {
                "id": "CLM:2",
                "preferred_name": "claim two",
                "domain_tags": ["claim"],
                "metadata": second,
            },
        },
        "edges": [],
    }
    graph_path = tmp_path / "knowledge_graph.json"
    rewritten_graph = tmp_path / "knowledge_graph.v2.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    extracted = tmp_path / "extracted_claims.jsonl"
    extracted.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    papers, report = build_graph_membership_index(graph_path)
    assert report["legacy_case3_claims_without_promoted_scope"] == 0
    routing, counters = rewrite_graph(graph_path, rewritten_graph, papers)
    assert counters["claims_rewritten"] == 2
    validate_rewritten_graph(rewritten_graph, 2)

    payload = json.loads(rewritten_graph.read_text(encoding="utf-8"))
    first_v2 = payload["concepts"]["CLM:1"]["metadata"]
    second_v2 = payload["concepts"]["CLM:2"]["metadata"]
    assert claim_case_study_ids_from_dict(first_v2) == ["brain_age"]
    assert claim_case_study_ids_from_dict(second_v2) == [
        "case1_transdiagnostic",
        "prognosis",
    ]
    expected_paper_ids = ["case1_transdiagnostic", "brain_age", "prognosis"]
    assert paper_case_study_ids_from_dict(first_v2) == expected_paper_ids
    assert paper_case_study_ids_from_dict(second_v2) == expected_paper_ids
    assert "paper_scope" not in first_v2
    assert "case3_tasks" not in first_v2

    rewritten_extracted = tmp_path / "extracted_claims.v2.jsonl"
    summary = rewrite_extracted(extracted, rewritten_extracted, routing)
    assert summary["graph_claims_matched"] == 2
    rows = [json.loads(line) for line in rewritten_extracted.read_text().splitlines()]
    assert all(row["paper_case_study_ids"] == expected_paper_ids for row in rows)
    assert all("paper_scope" not in row and "case3_tasks" not in row for row in rows)


def test_provenance_migration_cleans_anchors_and_routes_edges(tmp_path: Path) -> None:
    claim = _legacy_claim(
        "CLM:1",
        tasks=["brain_age"],
        scopes=["general", "case3"],
    )
    claim.pop("paper_scope")
    claim.pop("case3_tasks")
    claim["paper_case_study_ids"] = ["brain_age"]
    claim["claim_case_study_ids"] = ["brain_age"]
    claim["metadata"] = {
        "paper_case_study_ids": ["brain_age"],
        "claim_case_study_ids": ["brain_age"],
        "review_text": "Historical case3_tasks wording remains evidence.",
    }
    graph = {
        "metadata": {"name": "tiny"},
        "concepts": {
            "CLM:1": {
                "id": "CLM:1",
                "domain_tags": ["claim"],
                "metadata": claim,
            },
            "ANCHOR:1": {
                "id": "ANCHOR:1",
                "metadata": {"anchor_role": "subject", "paper_scope": ["case3"]},
            },
        },
        "edges": [
            {
                "source_id": "ANCHOR:1",
                "target_id": "CLM:1",
                "metadata": {"claim_id": "CLM:1", "paper_scope": ["case3"]},
            }
        ],
    }
    graph_path = tmp_path / "knowledge_graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    report = apply_provenance_migration(graph_path)
    assert report["validation"]["is_canonical"] is True
    migrated = json.loads(graph_path.read_text(encoding="utf-8"))
    assert "paper_scope" not in migrated["concepts"]["ANCHOR:1"]["metadata"]
    edge_metadata = migrated["edges"][0]["metadata"]
    assert edge_metadata["paper_case_study_ids"] == ["brain_age"]
    assert edge_metadata["claim_case_study_ids"] == ["brain_age"]
    assert "paper_scope" not in edge_metadata
    assert "case3_tasks" in claim["metadata"]["review_text"]
