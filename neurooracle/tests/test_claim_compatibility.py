from copy import deepcopy

import pytest

from neurooracle.src.claim_compatibility import compatible_metadata, normalize_claim_payload
from neurooracle.src.schema import Claim, Evidence, PaperRef


def payload(**extra):
    return {"id": "CLM:compat", "subject_id": "A", "subject_name": "Gene A", "predicate": "causes",
            "object_id": "B", "object_name": "Disease B", "evidence": {}, "source_paper": {}, **extra}


@pytest.mark.parametrize("value", ["A reported finding.", "", 0.5, 0, None, ["legacy item"], False])
def test_legacy_evidence_preserves_value_without_fabricating_statistics(value):
    original = payload(evidence=value)
    untouched = deepcopy(original)
    claim = Claim.from_dict(original)
    assert isinstance(claim.evidence, Evidence)
    encoded = claim.to_dict()
    assert encoded["evidence"]["legacy_value"] == value
    assert encoded["evidence"]["p_value"] is None
    assert encoded["evidence"]["sample_size"] is None
    assert encoded["evidence"]["replicability"] == ""
    assert Claim.from_dict(encoded).to_dict() == encoded
    assert original == untouched


def test_structured_extensions_and_unknown_claim_and_paper_fields_survive():
    original = payload(evidence={"p_value": 0, "effect_direction": "negative", "custom": {"x": [1]}},
                       source_paper={"pmid": "1", "license": "source license"}, custom_audit={"x": [1]})
    encoded = Claim.from_dict(original).to_dict()
    assert encoded["evidence"]["effect_direction"] == "negative"
    assert encoded["evidence"]["p_value"] == 0
    assert encoded["source_paper"]["license"] == "source license"
    assert encoded["custom_audit"] == original["custom_audit"]
    encoded["evidence"]["custom"]["x"].append(2)
    assert original["evidence"]["custom"]["x"] == [1]


@pytest.mark.parametrize("field,value", [("subject_type", "GENE"), ("object_type", "DISEASE"),
                                       ("conditions", ["cohort A"]), ("population", {"n_female": 0})])
def test_direct_fields_fill_missing_nested_fields_without_deleting_original(field, value):
    original = payload(**{field: value}, metadata={field: None})
    claim = Claim.from_dict(original)
    encoded = claim.to_dict()
    assert claim.metadata[field] == value
    assert encoded[field] == value
    assert encoded["metadata"][field] == value
    candidate = normalize_claim_payload(original)
    assert candidate[field] == value
    assert candidate["metadata"][field] == value
    assert original["metadata"][field] is None


def test_conflicting_direct_and_nested_values_both_survive():
    original = payload(subject_type="GENE", metadata={"subject_type": "DISEASE"})
    encoded = Claim.from_dict(original).to_dict()
    assert encoded["subject_type"] == "GENE"
    assert encoded["metadata"]["subject_type"] == "DISEASE"
    assert normalize_claim_payload(original) == original


def test_minimal_candidate_normalization_does_not_rewrite_other_fields():
    original = payload(evidence="source text", subject_type="GENE", metadata={"audit": "keep"},
                       raw_text="unchanged", scope_reaudit={"decision": "unchanged"})
    repaired = normalize_claim_payload(original)
    for key in original.keys() - {"evidence", "metadata"}:
        assert repaired[key] == original[key]
    assert repaired["metadata"]["audit"] == "keep"
    assert normalize_claim_payload(repaired) == repaired


def test_unknown_extension_container_name_is_not_treated_as_an_internal_envelope():
    original = payload(extra_fields={"literal_input": True}, evidence={"extra_fields": ["literal"]})
    encoded = Claim.from_dict(original).to_dict()
    assert encoded["extra_fields"] == original["extra_fields"]
    assert encoded["evidence"]["extra_fields"] == ["literal"]
    assert Claim.from_dict(encoded).to_dict() == encoded


def test_direct_claim_constructor_with_legacy_evidence_is_safe():
    claim = Claim("CLM:x", "A", "A", "causes", "B", "B", evidence="legacy")
    assert claim.to_dict()["evidence"]["legacy_value"] == "legacy"


def test_malformed_nested_metadata_is_not_silently_discarded():
    with pytest.raises(TypeError):
        compatible_metadata({"metadata": "bad layout", "subject_type": "GENE"})


def test_new_evidence_and_paper_serialization_keep_the_original_public_schema():
    assert set(Evidence().to_dict()) == {"study_type", "methodology", "p_value", "effect_size", "effect_metric",
                                       "sample_size", "replicability", "direction"}
    assert "extra_fields" not in PaperRef().to_dict()


def test_real_graph_get_claim_is_compatible_without_mutating_stored_metadata(tmp_path):
    import json
    from neurooracle.src.graph_manager import KnowledgeGraph
    from neurooracle.src.schema import ConceptNode
    from neurooracle.src.storage import load_graph, save_graph

    original = payload(evidence="verbatim legacy evidence", subject_type="GENE", conditions=["cohort A"])
    graph = KnowledgeGraph()
    graph.add_concept(ConceptNode(id=original["id"], preferred_name="claim", domain_tags=["claim"], metadata=deepcopy(original)))
    first, second = tmp_path / "before.json", tmp_path / "after.json"
    save_graph(graph, first)
    loaded = load_graph(first)
    claim = loaded.get_claim(original["id"])
    assert claim.metadata["subject_type"] == "GENE"
    assert claim.metadata["conditions"] == ["cohort A"]
    assert claim.to_dict()["evidence"]["legacy_value"] == original["evidence"]
    save_graph(loaded, second)
    assert json.loads(first.read_text())["concepts"][original["id"]]["metadata"] == original
    assert json.loads(second.read_text())["concepts"][original["id"]]["metadata"] == original
