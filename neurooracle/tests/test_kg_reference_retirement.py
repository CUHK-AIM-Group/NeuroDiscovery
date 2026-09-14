from copy import deepcopy
import sqlite3

import pytest

from neurooracle.scripts.retire_kg_duplicate_references import (
    CATEGORY, candidate_ordinal, check_proposal, claim_about_state, compute_topology,
    has_about_issue, metadata_after, restore_edges, same_record, select_retirements,
)
from neurooracle.scripts.plan_kg_reference_repairs import propose_reference_patch
from neurooracle.src.schema import ConceptNode, Edge


def fixture(about=False):
    nodes = {key: ConceptNode(id=key, preferred_name=name, domain_tags=[domain]).to_dict()
             for key, name, domain in (("CLM_CONCEPT:a", "Gene A", "gene"),
                                      ("GENE:A", "Gene A", "gene"), ("D:B", "Disease B", "disease"))}
    claim = {"id": "CLM:1", "subject_id": "GENE:A", "subject_name": "Gene A", "subject_type": "GENE",
             "object_id": "D:B", "object_name": "Disease B", "object_type": "DISEASE",
             "predicate": "causes", "negated": False}
    before = Edge("CLM:1" if about else "CLM_CONCEPT:a", "CLM_CONCEPT:a" if about else "D:B",
                  "about" if about else "causes", source="claim:paper", evidence_ref="Exact source",
                  metadata={"claim_id": "CLM:1", **({"anchor_role": "subject"} if about else {"negated": False})}).to_dict()
    proposal, reason = propose_reference_patch(before, claim, nodes)
    assert reason == "eligible"
    detail = {"edge_ordinal": 2, "category": CATEGORY, "proposal": proposal,
              "existing_destination_edges": [{"ordinal": 1, "edge": deepcopy(proposal["after"])}]}
    return nodes, claim, detail


@pytest.mark.parametrize("about", [False, True])
def test_retirement_requires_exact_current_claim_guard_and_keeps_witness(about):
    nodes, claim, detail = fixture(about)
    original = deepcopy(detail)
    result = select_retirements([detail])
    check_proposal(result[0], claim, nodes)
    assert result[0]["retained_original_edge_ordinal"] == 1
    assert result[0]["original_edge_ordinal"] == 2
    assert not result[0]["node_merge"] and not result[0]["claim_mutation"]
    assert detail == original


@pytest.mark.parametrize("field,value", [("confidence", 0.123), ("evidence_ref", "Other source"),
                                         ("relation_type", "inhibits"), ("source", "other:paper")])
def test_different_payload_is_not_retired(field, value):
    _, _, detail = fixture()
    detail["existing_destination_edges"][0]["edge"][field] = value
    with pytest.raises(ValueError, match="independently retained"):
        select_retirements([detail])


@pytest.mark.parametrize("value", [True, 0, None, "false"])
def test_different_negation_value_or_json_type_is_not_equal(value):
    _, _, detail = fixture()
    detail["existing_destination_edges"][0]["edge"]["metadata"]["negated"] = value
    with pytest.raises(ValueError):
        select_retirements([detail])


def test_different_claim_or_extra_evidence_is_not_equal():
    _, _, detail = fixture()
    detail["existing_destination_edges"][0]["edge"]["metadata"]["claim_id"] = "CLM:other"
    with pytest.raises(ValueError):
        select_retirements([detail])
    assert not same_record({"evidence": {"a": 1}}, {"evidence": {"a": 1, "b": 2}})
    assert same_record({"a": 1, "b": False}, {"b": False, "a": 1})


def test_witness_cannot_itself_be_retired_and_scope_excludes_other_categories():
    _, _, detail = fixture()
    other = deepcopy(detail)
    other["edge_ordinal"] = 1
    other["existing_destination_edges"][0]["ordinal"] = 2
    with pytest.raises(ValueError):
        select_retirements([detail, other])
    other["category"] = "distinct_claim_evidence_same_relation"
    assert len(select_retirements([detail, other])) == 1
    with pytest.raises(ValueError, match="duplicate retirement"):
        select_retirements([detail, detail])


@pytest.mark.parametrize("change", ["claim_endpoint", "old_name", "canonical_type", "extra_evidence"])
def test_current_reference_guard_rejects_drift(change):
    nodes, claim, detail = fixture()
    row = select_retirements([detail])[0]
    if change == "claim_endpoint":
        claim["subject_id"] = "OTHER:A"
    elif change == "old_name":
        nodes["CLM_CONCEPT:a"]["preferred_name"] = "Gene A expression"
    elif change == "canonical_type":
        nodes["GENE:A"]["domain_tags"] = ["drug"]
    else:
        row["retained_edge"]["metadata"]["extra_evidence"] = "do not lose"
    with pytest.raises(ValueError):
        check_proposal(row, claim, nodes)


@pytest.mark.parametrize("retired", [[], [1], [2], [5], [1, 2, 5], [1, 2, 3, 4, 5]])
def test_rollback_restores_first_middle_last_adjacent_and_all_edges(retired):
    original = [{"record": number} for number in range(1, 6)]
    rows = [{"original_edge_ordinal": number, "retired_edge": original[number - 1]} for number in retired]
    candidate = [edge for number, edge in enumerate(original, 1) if number not in retired]
    restored = list(restore_edges(candidate, rows, len(original)))
    assert [edge for _, edge, _ in restored] == original
    assert sum(removed for _, _, removed in restored) == len(retired)
    for original_ordinal, edge, removed in restored:
        if not removed:
            assert candidate[candidate_ordinal(original_ordinal, retired) - 1] == edge
        else:
            with pytest.raises(ValueError):
                candidate_ordinal(original_ordinal, retired)


@pytest.mark.parametrize("candidate,retirements,count", [([], [], 1), ([{}, {}], [], 1),
    ([], [{"original_edge_ordinal": 2, "retired_edge": {}}], 1),
    ([], [{"original_edge_ordinal": 1, "retired_edge": {}}] * 2, 1)])
def test_rollback_rejects_missing_surplus_and_invalid_ordinals(candidate, retirements, count):
    with pytest.raises(ValueError):
        list(restore_edges(candidate, retirements, count))


def test_topology_counts_parallel_records_selfloops_new_isolates_and_components(tmp_path):
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE nodes(id TEXT PRIMARY KEY); CREATE TABLE edges(s TEXT,t TEXT,r TEXT,ordinal INTEGER PRIMARY KEY,record_sha256 TEXT); CREATE INDEX edges_pair ON edges(s,t,r);")
    db.executemany("INSERT INTO nodes VALUES (?)", [(name,) for name in ("A", "B", "old", "alone")])
    db.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", [("A", "B", "x", 1, "a"), ("old", "B", "x", 2, "b"),
        ("A", "B", "y", 3, "c"), ("A", "A", "x", 4, "d"), ("A", "A", "x", 5, "e")])
    rows = [{"original_edge_ordinal": 2, "retired_edge": {"source_id": "old", "target_id": "B"}}]
    result = compute_topology(db, rows, tmp_path)
    assert result["before"]["edges"] == 5 and result["after"]["edges"] == 4
    assert result["before"]["connected_components"] == 2 and result["after"]["connected_components"] == 3
    assert result["before"]["isolated_nodes"] == 1 and result["after"]["isolated_nodes"] == 2
    assert result["after"]["self_loop_records"] == 2
    assert result["after"]["unique_nonself_pairs"] == 1
    assert result["after"]["different_relations_same_pair"] == 1
    assert result["after"]["parallel_pair_groups"] == 1
    assert result["newly_isolated_node_ids"] == ["old"]
    metadata = {"stats": {"n_concepts": 4, "n_edges": 5, "connected_components": 2, "relations": {"x": 4, "y": 1},
                          "domains": {"retained": 4}}, "history": {"original": True}}
    before = deepcopy(metadata)
    after = metadata_after(metadata, result)
    assert metadata == before
    assert after["stats"]["n_edges"] == 4 and after["stats"]["relations"] == {"x": 3, "y": 1}
    assert after["stats"]["domains"] == metadata["stats"]["domains"] and after["history"] == metadata["history"]
    db.close()


def test_exact_vs_distinct_parallel_records_are_not_conflated(tmp_path):
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE nodes(id TEXT PRIMARY KEY); CREATE TABLE edges(s TEXT,t TEXT,r TEXT,ordinal INTEGER PRIMARY KEY,record_sha256 TEXT); CREATE INDEX edges_pair ON edges(s,t,r);")
    db.executemany("INSERT INTO nodes VALUES (?)", [(name,) for name in ("A", "B", "C")])
    db.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", [("A", "B", "x", 1, "a"),
        ("A", "B", "x", 2, "a"), ("B", "C", "x", 3, "b"), ("B", "C", "x", 4, "c")])
    result = compute_topology(db, [], tmp_path)
    assert result["after"]["exact_duplicate_edge_records"] == 1
    assert result["after"]["same_relation_different_records"] == 1
    db.close()


def test_about_state_tracks_missing_and_extra_not_scientific_edge_direction():
    _, claim, detail = fixture(True)
    proper_subject = detail["proposal"]["after"]
    proper_object = {**proper_subject, "target_id": claim["object_id"]}
    stale = detail["proposal"]["before"]
    before = claim_about_state(claim, [proper_subject, proper_object, stale])
    after = claim_about_state(claim, [proper_subject, proper_object])
    assert has_about_issue(before) and not has_about_issue(after)
    assert before["subject_links"] == after["subject_links"] == 1
    assert before["object_links"] == after["object_links"] == 1
    assert before["extra_links"] == 1 and after["extra_links"] == 0
    assert has_about_issue(claim_about_state(claim, [proper_subject]))
