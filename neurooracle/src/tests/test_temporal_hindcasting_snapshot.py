from __future__ import annotations

import csv
import json
from pathlib import Path

from neurooracle.scripts.build_temporal_kg_snapshot import (
    TEMPORAL_SNAPSHOT_SCHEMA_VERSION,
    _is_claim_endpoint_node,
    build_snapshot,
    snapshot_matches_input,
)
from neurooracle.scripts.run_case_study_hindcasting import DEFAULT_WINDOWS


def _node(node_id: str, *, source_vocab: str = "claim_extraction", metadata=None) -> dict:
    return {
        "id": node_id,
        "preferred_name": node_id,
        "semantic_types": [],
        "domain_tags": ["claim"] if node_id.startswith("CLM:") else [],
        "source_vocab": source_vocab,
        "definition": "",
        "aliases": [],
        "external_ids": {},
        "spatial_mapping": {},
        "metadata": metadata or {},
    }


def _claim(claim_id: str, year: int, subject_id: str, object_id: str) -> dict:
    return {
        "id": claim_id,
        "subject_id": subject_id,
        "subject_name": subject_id,
        "predicate": "is_associated_with",
        "object_id": object_id,
        "object_name": object_id,
        "source_paper": {"pmid": f"PMID-{claim_id}", "year": year},
    }


def test_default_windows_are_five_consecutive_five_year_forecasts() -> None:
    assert [
        (window.freeze_year, window.future_start_year, window.future_end_year)
        for window in DEFAULT_WINDOWS
    ] == [
        (2016, 2017, 2021),
        (2017, 2018, 2022),
        (2018, 2019, 2023),
        (2019, 2020, 2024),
        (2020, 2021, 2025),
    ]


def test_all_claim_anchor_vocabularies_are_temporally_scoped() -> None:
    for source_vocab in (
        "claim_extraction",
        "claim_extraction_anchor",
        "manual_claim_anchor",
        "manual_claim_anchor_repair",
        "manual_general_claim_anchor",
        "replay_anchor_mint",
    ):
        assert _is_claim_endpoint_node(_node("ENDPOINT", source_vocab=source_vocab))
    assert not _is_claim_endpoint_node(_node("CURATED", source_vocab="ClinicalOutcomes"))


def test_snapshot_removes_post_cutoff_claim_and_curated_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "snapshot"
    input_dir.mkdir()

    historical = _claim("CLM:historical", 2016, "A", "B")
    future = _claim("CLM:future", 2022, "A", "C")
    with (input_dir / "extracted_claims.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(historical) + "\n")
        handle.write(json.dumps(future) + "\n")

    with (input_dir / "papers_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pmid", "year"])
        writer.writeheader()
        writer.writerow({"pmid": "old", "year": 2016})
        writer.writerow({"pmid": "new", "year": 2022})

    concepts = {
        "A": _node("A"),
        "B": _node("B", source_vocab="manual_claim_anchor"),
        "C": _node("C", source_vocab="manual_general_claim_anchor"),
        "STATIC:1": _node("STATIC:1", source_vocab="ontology"),
        "STATIC:2": _node("STATIC:2", source_vocab="ontology"),
        "ATLAS:Schaefer400": _node(
            "ATLAS:Schaefer400",
            source_vocab="atlas",
            metadata={"ref": "Schaefer et al. 2018 Cerebral Cortex"},
        ),
        "CLM:historical": _node("CLM:historical"),
        "CLM:future": _node("CLM:future"),
    }
    edges = [
        {
            "source_id": "A",
            "target_id": "B",
            "relation_type": "is_associated_with",
            "source": "paper",
            "confidence": 0.8,
            "evidence_ref": "",
            "metadata": {"claim_id": "CLM:historical"},
        },
        {
            "source_id": "A",
            "target_id": "C",
            "relation_type": "is_associated_with",
            "source": "paper",
            "confidence": 0.8,
            "evidence_ref": "",
            "metadata": {"claim_id": "CLM:future"},
        },
        {
            "source_id": "STATIC:1",
            "target_id": "STATIC:2",
            "relation_type": "is_a",
            "source": "ontology",
            "confidence": 1.0,
            "evidence_ref": "",
            "metadata": {},
        },
        {
            "source_id": "STATIC:1",
            "target_id": "A",
            "relation_type": "maps_to",
            "source": "HansenReceptor2022",
            "confidence": 0.9,
            "evidence_ref": "Hansen et al. 2022 Nature Neuroscience",
            "metadata": {},
        },
        {
            "source_id": "STATIC:1",
            "target_id": "ATLAS:Schaefer400",
            "relation_type": "maps_to",
            "source": "atlas",
            "confidence": 0.9,
            "evidence_ref": "",
            "metadata": {},
        },
        {
            "source_id": "C",
            "target_id": "ATLAS:Schaefer400",
            "relation_type": "maps_to",
            "source": "ontology",
            "confidence": 0.9,
            "evidence_ref": "",
            "metadata": {},
        },
    ]
    graph = {"metadata": {}, "concepts": concepts, "edges": edges}
    (input_dir / "knowledge_graph.json").write_text(json.dumps(graph), encoding="utf-8")

    # Production releases are multi-gigabyte JSON documents. The builder must
    # stream them instead of materializing the whole graph with json.load().
    with monkeypatch.context() as patch:
        patch.setattr(
            json,
            "load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("full-file json.load is prohibited")
            ),
        )
        manifest = build_snapshot(input_dir, output_dir, 2016)
    snapshot = json.loads((output_dir / "knowledge_graph.json").read_text(encoding="utf-8"))

    assert "CLM:historical" in snapshot["concepts"]
    assert "CLM:future" not in snapshot["concepts"]
    assert "B" in snapshot["concepts"]
    assert "C" not in snapshot["concepts"]
    assert "ATLAS:Schaefer400" not in snapshot["concepts"]
    assert {(edge["source_id"], edge["target_id"]) for edge in snapshot["edges"]} == {
        ("A", "B"),
        ("STATIC:1", "STATIC:2"),
    }
    assert manifest["removed_future_claim_edges"] == 1
    assert manifest["removed_future_dated_non_claim_edges"] == 1
    assert manifest["removed_future_dated_curated_nodes"] == 1
    assert manifest["removed_orphan_claim_endpoint_concepts"] == 1
    assert manifest["removed_orphan_claim_endpoint_concepts_after_edge_filter"] == 0
    assert manifest["kept_undated_non_claim_edges"] == 3
    assert manifest["snapshot_schema_version"] == TEMPORAL_SNAPSHOT_SCHEMA_VERSION
    assert snapshot_matches_input(manifest, input_dir, 2016)

    stale_manifest = dict(manifest)
    stale_manifest["snapshot_schema_version"] = "temporal_kg_snapshot.v1"
    assert not snapshot_matches_input(stale_manifest, input_dir, 2016)

    with (input_dir / "extracted_claims.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert not snapshot_matches_input(manifest, input_dir, 2016)

    retained_papers = list(csv.DictReader((output_dir / "papers_metadata.csv").open(encoding="utf-8")))
    assert [row["year"] for row in retained_papers] == ["2016"]


# Last updated: 2026-08-12 18:00 HKT - cover manual claim anchors and snapshot schema invalidation.
