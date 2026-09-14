from copy import deepcopy
import json

import pytest

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode, Edge
from neurooracle.src.storage import load_graph, save_graph
from neurooracle.src.relation_evidence import group_claim_evidence, relation_key


def payload(cid="CLM:1", pmid="1", **changes):
    return dict(id=cid, subject_id="A", subject_name="specific measurement", predicate="correlates_with",
        object_id="B", object_name="outcome", negated=False, source_paper={"pmid": pmid},
        raw_text="Recorded original evidence.", evidence={"direction": "positive"}, **changes)


def graph():
    kg = KnowledgeGraph()
    for nid in ("A", "B", "CLM:1", "CLM:2"):
        kg.add_concept(ConceptNode(id=nid, preferred_name=nid))
    return kg


def edge(owner="CLM:1", **changes):
    result = dict(source_id="A", target_id="B", relation_type="correlates_with", confidence=0.7,
        metadata={"claim_id": owner, "negated": False})
    result.update(changes)
    return Edge(**result)


def test_shared_relation_preserves_papers_negation_and_context():
    a = payload()
    b = {**payload("CLM:2", "2"), "negated": True, "evidence": {"direction": "negative"},
         "metadata": {"population": {"species": "mouse"}}}
    original = deepcopy([a, b])
    result = list(group_claim_evidence([a,b]))
    assert len(result) == 1
    assert result[0]["paper_count"] == 2 and result[0]["claim_count"] == 2
    assert result[0]["evidence_variant_count"] == 2
    assert {item["negated"] for item in result[0]["members"]} == {True, False}
    assert [a,b] == original


@pytest.mark.parametrize("changes", [{"subject_name": "different brain region measurement"},
    {"subject_id": "unreviewed_alias"}, {"predicate": "causes"}, {"subject_name": "Specific measurement"}])
def test_no_unreviewed_identity_name_predicate_or_gene_case_merging(changes):
    a, b = payload(), {**payload("CLM:2", "2"), **changes}
    assert len(list(group_claim_evidence([a,b]))) == 2


def test_duplicate_ingestion_idempotent_but_conflicting_claim_id_rejected():
    a = payload()
    assert list(group_claim_evidence([a,deepcopy(a)]))[0]["claim_count"] == 1
    with pytest.raises(ValueError, match="conflicting"):
        list(group_claim_evidence([a, {**a, "raw_text": "different"}]))


def test_same_paper_different_claims_are_not_discarded():
    result = list(group_claim_evidence([payload(), payload("CLM:2")]))[0]
    assert result["paper_count"] == 1 and result["claim_count"] == 2


def test_context_preserves_numeric_zero_false_and_unknown_values():
    a = payload()
    b = {**payload("CLM:2"), "evidence": {"direction": "positive", "extra": False}}
    c = {**payload("CLM:3"), "evidence": {"direction": "positive", "extra": 0}}
    assert list(group_claim_evidence([a,b,c]))[0]["evidence_variant_count"] == 3


def test_missing_endpoint_fails_closed():
    with pytest.raises(ValueError):
        relation_key({**payload(), "subject_id": ""})


@pytest.mark.parametrize("minimum", [0, -1, True, 1.1])
def test_invalid_minimum_rejected(minimum):
    with pytest.raises(ValueError):
        list(group_claim_evidence([payload()], minimum_claims=minimum))


def test_nonwinning_edges_negation_and_other_predicates_roundtrip(tmp_path):
    kg = graph()
    values = [edge(), edge("CLM:2", confidence=0.8), edge("CLM:3", relation_type="predicts", confidence=0.6),
              edge("CLM:4", metadata={"claim_id": "CLM:4", "negated": True})]
    for value in values * 2:
        kg.add_edge(value)
    assert kg.G.number_of_edges() == 1
    assert kg.G.edges["A", "B"]["metadata"]["claim_id"] == "CLM:2"
    assert len(list(kg.iter_edge_records())) == 4
    expected = {kg._edge_signature(value.to_dict()) for value in values}
    for _ in range(2):
        path = save_graph(kg, tmp_path / "graph.json")
        kg = load_graph(path)
        assert {kg._edge_signature(row) for row in kg.iter_edge_records()} == expected
    assert json.loads(path.read_text())["metadata"]["stats"]["n_stored_edge_records"] == 4


def test_self_loops_preserved_outside_traversal(tmp_path):
    kg = graph()
    kg.add_edge(edge(target_id="A"))
    assert kg.G.number_of_edges() == 0
    assert len(list(load_graph(save_graph(kg, tmp_path / "graph.json")).iter_edge_records())) == 1


def test_edge_values_detached_and_false_not_equal_to_zero():
    kg = graph()
    a = edge()
    kg.add_edge(a)
    a.metadata["claim_id"] = "mutated"
    kg.add_edge(edge(metadata={"claim_id": "CLM:1", "negated": 0}))
    records = list(kg.iter_edge_records())
    assert len(records) == 2 and records[0]["metadata"]["claim_id"] == "CLM:1"
    records[0]["metadata"]["claim_id"] = "detached"
    assert next(kg.iter_edge_records())["metadata"]["claim_id"] == "CLM:1"


def test_removed_pair_does_not_resurrect_old_evidence():
    kg = graph()
    kg.add_edge(edge()); kg.add_edge(edge("CLM:2"))
    kg.G.remove_edge("A", "B")
    assert list(kg.iter_edge_records()) == []
    kg.add_edge(edge("CLM:3"))
    assert len(list(kg.iter_edge_records())) == 1


def test_live_relation_query_uses_claim_nodes_not_one_traversal_edge():
    kg = graph()
    kg._index["CLM:1"].metadata = payload()
    kg._index["CLM:2"].metadata = payload("CLM:2", "2")
    assert list(kg.iter_relation_evidence(minimum_claims=2))[0]["paper_count"] == 2


def test_duplicate_claim_id_with_different_relation_rejected():
    a = payload()
    with pytest.raises(ValueError, match="conflicting"):
        list(group_claim_evidence([a, {**a, "subject_name": "different entity"}]))


def test_update_nonwinning_claim_does_not_change_another_papers_evidence():
    kg = graph()
    kg._index["CLM:1"].metadata = payload()
    kg._index["CLM:2"].metadata = payload("CLM:2", "2")
    kg.add_edge(edge("CLM:1", confidence=0.7))
    kg.add_edge(edge("CLM:2", confidence=0.8))
    kg.update_claim("CLM:1", new_confidence=0.95)
    records = {row["metadata"]["claim_id"]: row for row in kg.iter_edge_records()}
    assert records["CLM:1"]["confidence"] == 0.95
    assert records["CLM:2"]["confidence"] == 0.8
    assert kg.G.edges["A", "B"]["metadata"]["claim_id"] == "CLM:1"
