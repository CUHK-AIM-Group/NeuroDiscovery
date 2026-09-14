from __future__ import annotations

from copy import deepcopy
import json

from neurooracle.src.kg_quality_checks import claim_record_checks, edge_claim_agreement, entity_identifier_checks
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode, Edge
from neurooracle.src.storage import load_graph


def example(evidence=None):
    return {"id": "CLM:x", "preferred_name": "A causes B", "domain_tags": ["claim"], "metadata": {
        "id": "CLM:x", "subject_id": "A", "subject_name": "A", "predicate": "causes",
        "object_id": "B", "object_name": "B", "confidence": .8, "negated": False,
        "evidence": {} if evidence is None else evidence, "source_paper": {}, "metadata": {},
    }}


def test_structured_claim_is_readable_and_checker_is_read_only():
    node = example({"sample_size": 100, "p_value": 0.0})
    original = deepcopy(node)
    assert claim_record_checks(node["id"], node) == []
    assert node == original


def test_text_evidence_remains_legacy_but_no_longer_breaks_roundtrip():
    node = example("The experiment found an association.")
    issues = claim_record_checks(node["id"], node)
    assert {r["code"] for r in issues} == {"claim_evidence_non_object"}


def test_direct_type_is_carried_into_compatible_model():
    node = example()
    node["metadata"]["subject_type"] = "GENE"
    assert not any(r["code"] == "direct_scientific_field_not_carried_into_claim_object" for r in claim_record_checks(node["id"], node))
    node["metadata"]["metadata"]["subject_type"] = "GENE"
    assert not any(r["code"] == "direct_scientific_field_not_carried_into_claim_object" for r in claim_record_checks(node["id"], node))


def test_value_disagreements_are_review_flags_not_identity_merges():
    node = example()
    node["metadata"].update({"subject_type": "GENE", "metadata": {"subject_type": "DISEASE"}})
    assert any(r["code"] == "direct_nested_values_differ" for r in claim_record_checks(node["id"], node))
    assert entity_identifier_checks("CUI:C0000001", {"external_ids": {"UMLS_CUI": "C0000002"}})[0]["code"] == "cui_id_external_identifier_disagreement"
    assert entity_identifier_checks("CUI:placeholder", {})[0]["code"] == "nonstandard_cui_prefixed_id"


def test_claim_edge_checks_do_not_conflate_different_evidence_sources():
    claim = example()["metadata"]
    edge = {"source_id": "A", "target_id": "B", "relation_type": "causes", "metadata": {"claim_id": "CLM:x", "negated": False}}
    assert edge_claim_agreement(edge, claim) == []
    edge["relation_type"] = "inhibits"
    assert edge_claim_agreement(edge, claim)[0]["code"] == "edge_claim_payload_disagreement"
    assert edge_claim_agreement(edge, None)[0]["code"] == "edge_references_missing_claim"


def test_actual_graph_loader_keeps_only_highest_confidence_relation_for_pair(tmp_path):
    path = tmp_path / "witness.json"
    graph = {"metadata": {}, "concepts": {x: ConceptNode(id=x, preferred_name=x).to_dict() for x in ("A", "B")},
             "edges": [Edge("A", "B", "causes", confidence=.7).to_dict(),
                       Edge("A", "B", "inhibits", confidence=.9).to_dict()]}
    path.write_text(json.dumps(graph), encoding="utf-8")
    kg = load_graph(path)
    assert len(graph["edges"]) == 2
    assert kg.G.number_of_edges() == 1
    assert kg.G.edges["A", "B"]["relation_type"] == "inhibits"
    assert json.loads(path.read_text())["edges"] == graph["edges"]
