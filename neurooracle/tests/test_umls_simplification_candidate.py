from __future__ import annotations

import json
import sqlite3

import networkx as nx
import pytest

from neurooracle.scripts.build_umls_simplification_candidate import (
    MAPPING_SOURCE,
    TARGET_SOURCE,
    build_candidate,
    cheap,
    file_sha,
    run_regression,
    validate_candidate,
)
from neurooracle.src.umls_audit_store import UmlsAuditStore, compact_record


def node(node_id, name, *, vocab="legacy", metadata=None, domain="imaging_feature", aliases=None):
    return {
        "id": node_id, "preferred_name": name, "source_vocab": vocab,
        "domain_tags": [domain], "semantic_types": ["T034"], "definition": "",
        "aliases": aliases or [], "external_ids": {}, "spatial_mapping": None,
        "metadata": metadata or {},
    }


def edge(source, target, relation="maps_to", *, umls=False, status="auto_accepted_exact"):
    return {
        "source_id": source, "target_id": target, "relation_type": relation,
        "source": MAPPING_SOURCE if umls else "legacy", "confidence": 0.98,
        "evidence": "unchanged evidence text",
        "metadata": {
            "review_status": status, "method": "atomic_surface_normalized_exact",
            "semantic_compatibility": True, "umls_release": "2026AA",
            "detailed_match_audit": {"original": "preserved"},
        } if umls else {"legacy_field": "retain"},
    }


@pytest.fixture
def source_graph(tmp_path):
    parents = [node("CLM_CONCEPT:p", "FA"), node("CLM_CONCEPT:q", "gray matter")]
    canonical = node("IF:fa", "fractional anisotropy", aliases=["FA"], metadata={"original_field": 7})
    claim = node("CLM:c", "claim", domain="claim", metadata={
        "subject_id": "CLM_CONCEPT:p", "object_id": "CLM_CONCEPT:q", "predicate": "associated_with",
        "subject_name": "FA", "object_name": "gray matter", "confidence": 0.9,
        "raw_text": "A long source claim with traceable evidence and intact measurements.",
        "evidence": {"study_type": "cohort", "sample_size": 300, "p_value": 0.01},
        "source_paper": {"pmid": "12345678", "year": 2025},
        "claim_case_study_ids": ["case2", "case4"],
    })
    cui = node("CUI:C0000002", "gray matter", vocab=TARGET_SOURCE, metadata={
        "semantic_type_names": ["Finding"], "umls_release": "2026AA", "extra_audit": [1, 2],
    })
    atoms = []
    for suffix, status, parent, name in (
        ("u", "unmapped", "p", "FA"),
        ("a", "auto_accepted_exact", "p", "FA"),
        ("r", "needs_review", "q", "gray matter"),
    ):
        atoms.append(node(f"CLM_ATOM:{suffix}", name, vocab="CLM_CONCEPT_atomic_projection", metadata={
            "source_mention_id": f"CLM_CONCEPT:{parent}", "source_text": name,
            "mapping_status": status, "mapping_count": 0 if status == "unmapped" else 1,
            "biomedical_eligible": True, "evidence_span": {"start": 0, "end": len(name)},
            "source_mention_preserved": True, "extra_audit": {"keep": "all"},
        }))
    nodes = [*parents, canonical, claim, cui, *atoms]
    edges = [
        edge("CLM:c", "CLM_CONCEPT:p", "about"),
        edge("CLM_CONCEPT:p", "CLM_CONCEPT:q", "associated_with"),
        edge("CLM_ATOM:a", "IF:fa", umls=True),
        edge("CLM_ATOM:r", "CUI:C0000002", umls=True, status="needs_review"),
    ]
    connectivity = nx.Graph()
    connectivity.add_nodes_from(n["id"] for n in nodes)
    connectivity.add_edges_from((e["source_id"], e["target_id"]) for e in edges)
    graph = {
        "metadata": {"stats": {"n_concepts": len(nodes), "n_edges": len(edges),
                                "connected_components": nx.number_connected_components(connectivity)}},
        "concepts": {n["id"]: n for n in nodes}, "edges": edges,
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    return source, graph


def test_candidate_is_lossless_and_preserves_experiment_inputs(source_graph, tmp_path):
    source, original = source_graph
    before = cheap(source), file_sha(source)
    output = tmp_path / "candidate"
    build = build_candidate(source, output, before[1])
    validation = validate_candidate(output, build)
    regression = run_regression(source, output)
    candidate = json.loads((output / "knowledge_graph.candidate.json").read_text(encoding="utf-8"))
    assert build["counts"]["core_nodes"] == 7
    assert build["counts"]["offloaded_atoms"] == 1
    assert build["core_connected_components"] == 3
    assert "CLM_ATOM:u" not in candidate["concepts"]
    assert candidate["concepts"]["CLM:c"] == original["concepts"]["CLM:c"]
    assert candidate["edges"][:2] == original["edges"][:2]
    assert len(candidate["concepts"]["CLM_ATOM:a"]["metadata"]) == 5
    assert len(candidate["concepts"]["CUI:C0000002"]["metadata"]) == 2
    assert len(candidate["edges"][2]["metadata"]) == 4
    assert all(validation["checks"].values())
    assert all(regression["checks"].values())
    assert regression["kge_triples"] == 3
    assert regression["full_model_training_rerun"] is False
    assert (cheap(source), file_sha(source)) == before
    with UmlsAuditStore(output / "umls_details.sqlite") as store:
        assert store.fetch("atoms/CLM_ATOM:u") == original["concepts"]["CLM_ATOM:u"]
        for record in candidate["concepts"].values():
            assert store.hydrate(record) == original["concepts"][record["id"]]
        for index, record in enumerate(candidate["edges"]):
            assert store.hydrate(record) == original["edges"][index]
        assert store.atoms_for_mention("CLM_CONCEPT:p") == [
            {"atom_id": "CLM_ATOM:u", "mapping_status": "unmapped", "in_core": 0},
            {"atom_id": "CLM_ATOM:a", "mapping_status": "auto_accepted_exact", "in_core": 1},
        ]
        assert store.mappings_for_atom("CLM_ATOM:a") == [original["edges"][2]]
        assert store.mappings_for_atom("CLM_ATOM:r") == []
        assert store.mappings_for_atom("CLM_ATOM:r", include_needs_review=True) == [original["edges"][3]]
        assert store.connection.execute("SELECT COUNT(*) FROM alignment_candidates").fetchone()[0] == 2
        assert store.connection.execute("SELECT DISTINCT decision FROM alignment_candidates").fetchall() == [("needs_semantic_review",)]
        with pytest.raises(sqlite3.OperationalError):
            store.connection.execute("DELETE FROM atoms")
        with pytest.raises(ValueError):
            store.fetch("atoms;DROP TABLE atoms/CLM_ATOM:a")
        with pytest.raises(KeyError):
            store.fetch("atoms/missing")
        with pytest.raises(ValueError):
            store.atoms_for_mention("CLM_CONCEPT:p", limit=0)
        tampered = dict(candidate["concepts"]["CLM_ATOM:a"], preferred_name="changed")
        with pytest.raises(ValueError, match="does not match"):
            store.hydrate(tampered)


def test_offloading_rejects_even_one_incident_edge(source_graph, tmp_path):
    source, graph = source_graph
    graph["edges"].append(edge("CLM_ATOM:u", "IF:fa"))
    source.write_text(json.dumps(graph), encoding="utf-8")
    before = file_sha(source)
    with pytest.raises(ValueError, match="offloaded or absent"):
        build_candidate(source, tmp_path / "invalid")
    assert file_sha(source) == before


def test_source_hash_and_nonoverwrite_guards(source_graph, tmp_path):
    source, _ = source_graph
    with pytest.raises(ValueError, match="SHA-256"):
        build_candidate(source, tmp_path / "bad_hash", "0" * 64)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        build_candidate(source, occupied)
    with pytest.raises(ValueError, match="separate"):
        build_candidate(source, source.parent)


@pytest.mark.parametrize("tamper", ["payload", "query_column", "core"])
def test_validation_detects_tampering(source_graph, tmp_path, tamper):
    source, _ = source_graph
    output = tmp_path / "candidate"
    build = build_candidate(source, output)
    if tamper == "core":
        path = output / "knowledge_graph.candidate.json"
        graph = json.loads(path.read_text(encoding="utf-8"))
        graph["concepts"]["CLM:c"]["metadata"]["confidence"] = 0.1
        path.write_text(json.dumps(graph), encoding="utf-8")
    else:
        with sqlite3.connect(output / "umls_details.sqlite") as connection:
            if tamper == "payload":
                connection.execute("UPDATE atoms SET payload_json=json_set(payload_json,'$.metadata.extra_audit.keep','lost') WHERE record_id='CLM_ATOM:u'")
            else:
                connection.execute("UPDATE atoms SET source_mention_id='CLM_CONCEPT:wrong' WHERE record_id='CLM_ATOM:u'")
    with pytest.raises(ValueError):
        validate_candidate(output, build)


def test_compaction_does_not_mutate_original():
    original = node("CLM_ATOM:a", "A", metadata={"mapping_status": "unmapped", "extra": [1, 2]})
    result = compact_record(original, "atoms", original["id"])
    assert original["metadata"] == {"mapping_status": "unmapped", "extra": [1, 2]}
    assert result["metadata"] == {"mapping_status": "unmapped", "audit_ref": "atoms/CLM_ATOM:a"}
