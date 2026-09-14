from collections import Counter
from copy import deepcopy
import json

import pytest

from neurooracle.src.kg_metadata_compaction import (
    LAYOUT, LAYOUT_KEY, MEMBERSHIP_KEY, MEMBERSHIP_VERSION, compact_layout_enabled,
    compact_record, empty_slot, membership_schema_version, typed_equal,
)
from neurooracle.src.case_study_membership_contract import claim_evidence_payload
from neurooracle.src.schema import Claim, ConceptNode, Edge
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.storage import load_graph, save_graph, save_display_graph


def claim_record(**extra):
    md = dict(id="CLM:test", subject_id="A", object_id="B", subject_name="Gene A",
              object_name="Disease B", predicate="causes", negated=False, confidence=0.5,
              raw_text="Original evidence.", evidence={"p_value": 0, "legacy": []},
              source_paper={"pmid": "123", "title": "Paper"},
              paper_case_study_ids=["brain_age"], claim_case_study_ids=["brain_age"],
              scope_reaudit={"decision": "finalized", "audit_contract_version": "old", "unique": [1]},
              metadata={})
    md.update(extra)
    return dict(id=md["id"], preferred_name="claim", domain_tags=["claim"], metadata=md)


def projection(record):
    return claim_evidence_payload(record["metadata"], scope_context_sha256="f" * 64,
                                  include_legacy_case2_chain=True)


@pytest.mark.parametrize("left,right,equal", [(False, 0, False), (0, 0.0, False),
    ([False], [0], False), ([1, 2], [2, 1], False), ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
    ("123", 123, False), (None, None, True), ([], [], True)])
def test_typed_equality(left, right, equal):
    assert typed_equal(left, right) is equal


@pytest.mark.parametrize("value,empty", [(None, True), ("", True), ([], True), ({}, True),
    (False, False), (0, False), (0.0, False), (" ", False), ("unknown", False), ([None], False), ({"n": None}, False)])
def test_empty_slots_not_zero_or_unknown(value, empty):
    assert empty_slot(value) is empty


def test_exact_duplicates_removed_without_mutation_or_scientific_change():
    record = claim_record(claim_id="CLM:test", subject="Gene A", object="Disease B",
        raw_sentence="Original evidence.", evidence_text="Original evidence.", pmid="123", title="Paper",
        metadata={"subject_id": "A", "object_id": "B", "paper_case_study_ids": ["brain_age"],
                  "claim_case_study_ids": ["brain_age"], MEMBERSHIP_KEY: MEMBERSHIP_VERSION})
    before = deepcopy(record)
    counts = Counter()
    compacted = compact_record("node", record, counts)
    assert record == before
    assert compacted["metadata"]["metadata"] == {}
    assert all(key not in compacted["metadata"] for key in
               ("claim_id", "subject", "object", "raw_sentence", "evidence_text", "pmid", "title"))
    assert compacted["metadata"]["id"] == "CLM:test"
    assert projection(compacted) == projection(before)
    assert compacted["metadata"]["scope_reaudit"] == before["metadata"]["scope_reaudit"]
    assert compact_record("node", compacted) == compacted
    assert sum(counts.values()) == 12


def test_conflicts_unique_text_identity_and_extensions_survive():
    record = claim_record(claim_id="other", subject="Different", object="Other", pmid=123,
        doi="10.x/unique", title="Different paper", authors=["Unique author"],
        evidence_text="Different evidence", raw_sentence="Different sentence", claim="Paraphrase",
        paper_id="fallback", source_id="fallback2", queue_id="fallback3", original_subject_id="OLD:A",
        custom={"x": 1}, metadata={"subject_id": "OTHER", "paper_case_study_ids": ["different"],
                                  "population": {"n": 0}, "conditions": ["Unique cohort"]})
    assert compact_record("node", record) == record


@pytest.mark.parametrize("value", [None, "", [], {}, 0, False, " ", [None]])
def test_empty_science_cleanup_preserves_contract_bytes(value):
    record = claim_record(population=None, conditions=[], atom_types=[], subject_type="",
                          metadata={"population": value, "conditions": [], "subject_type": "", "object_type": None})
    compacted = compact_record("node", record)
    assert typed_equal(projection(compacted), projection(record))
    assert compacted["metadata"]["evidence"] == record["metadata"]["evidence"]
    assert compacted["metadata"]["source_paper"] == record["metadata"]["source_paper"]
    assert "atom_types" not in compacted["metadata"]


def test_only_boolean_completed_import_states_removed():
    record = claim_record(kg_injected=False, staged_only=True, no_api_extraction=True,
        batch_id="keep", qa_status="manual_reviewed", metadata={"kg_injected": True,
        "staged_only": "true", "reviewer": "keep", "original_object_id": "OLD"})
    counts = Counter()
    result = compact_record("node", record, counts)
    assert "kg_injected" not in result["metadata"]
    assert "staged_only" not in result["metadata"]
    assert result["metadata"]["metadata"] == {"staged_only": "true", "reviewer": "keep", "original_object_id": "OLD"}
    assert result["metadata"]["no_api_extraction"] is True
    assert sum(counts.values()) == 3


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_only_declared_membership_version_shared(kind):
    record = {"id": "A", "metadata": {MEMBERSHIP_KEY: MEMBERSHIP_VERSION,
              "paper_case_study_ids": [], "audit_ref": "edges/1", "review_status": "accepted"}}
    result = compact_record(kind, record)
    assert MEMBERSHIP_KEY not in result["metadata"]
    assert result["metadata"]["audit_ref"] == "edges/1"
    assert membership_schema_version(result, {LAYOUT_KEY: LAYOUT}) == MEMBERSHIP_VERSION
    different = deepcopy(record)
    different["metadata"][MEMBERSHIP_KEY] = "future.v9"
    assert compact_record(kind, different) == different
    no_labels = {"metadata": {MEMBERSHIP_KEY: MEMBERSHIP_VERSION}}
    assert compact_record(kind, no_labels) == no_labels
    assert membership_schema_version({"metadata": {}}, {LAYOUT_KEY: LAYOUT}) is None


def test_default_layout_is_legacy_and_unknown_layout_fails_closed():
    assert not compact_layout_enabled({})
    assert compact_layout_enabled({LAYOUT_KEY: LAYOUT})
    with pytest.raises(ValueError):
        compact_layout_enabled({LAYOUT_KEY: {"version": "future"}})


@pytest.mark.parametrize("use_compact", [False, True])
def test_storage_roundtrip_and_claim_reserialization_do_not_regrow_fields(tmp_path, use_compact):
    kg = KnowledgeGraph()
    record = claim_record(claim_id="CLM:test", raw_sentence="Original evidence.", kg_injected=True,
        metadata={"subject_id": "A", "object_id": "B", "paper_case_study_ids": ["brain_age"],
                  "claim_case_study_ids": ["brain_age"], MEMBERSHIP_KEY: MEMBERSHIP_VERSION})
    if use_compact:
        kg.serialization_metadata = {LAYOUT_KEY: deepcopy(LAYOUT), "detail_store": "current.sqlite"}
    kg.add_concept(ConceptNode.from_dict(record))
    for nid in ("A", "B"):
        kg.add_concept(ConceptNode(id=nid, preferred_name=nid))
    kg.add_edge(Edge(source_id="A", target_id="B", relation_type="causes", confidence=0.5,
        metadata={"claim_id": "CLM:test", MEMBERSHIP_KEY: MEMBERSHIP_VERSION, "claim_case_study_ids": ["brain_age"]}))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    save_graph(kg, first)
    loaded = load_graph(first)
    parsed = json.loads(first.read_text(encoding="utf-8"))
    if not use_compact:
        assert parsed["concepts"]["CLM:test"]["metadata"] == record["metadata"]
        return
    assert parsed["metadata"][LAYOUT_KEY] == LAYOUT
    assert loaded.serialization_metadata["detail_store"] == "current.sqlite"
    claim = loaded.get_claim("CLM:test")
    assert claim.subject_id == "A" and claim.metadata["claim_case_study_ids"] == ["brain_age"]
    loaded._index["CLM:test"].metadata = claim.to_dict()  # legacy writer recreates inner labels
    save_graph(loaded, second)
    reread = json.loads(second.read_text(encoding="utf-8"))
    md = reread["concepts"]["CLM:test"]["metadata"]
    assert "claim_case_study_ids" not in md["metadata"]
    assert "claim_id" not in md and "kg_injected" not in md
    assert reread["edges"][0]["metadata"]["claim_id"] == "CLM:test"
    assert MEMBERSHIP_KEY not in reread["edges"][0]["metadata"]
    display = tmp_path / "display.json"
    save_display_graph(loaded, display)
    assert json.loads(display.read_text(encoding="utf-8"))["metadata"][LAYOUT_KEY] == LAYOUT


def test_concept_atom_roles_not_removed_and_invalid_inputs_rejected():
    concept = {"id": "A", "metadata": {"atom_types": ["GENE"], "kg_injected": True}}
    assert compact_record("node", concept) == concept
    with pytest.raises(TypeError):
        compact_record("node", {"id": "CLM:x", "metadata": "wrong"})
    with pytest.raises(TypeError):
        compact_record("node", claim_record(metadata="wrong"))
    with pytest.raises(ValueError):
        compact_record("metadata", {})
