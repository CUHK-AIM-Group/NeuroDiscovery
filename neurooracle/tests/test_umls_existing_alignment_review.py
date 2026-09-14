from __future__ import annotations

import copy

import pytest

from neurooracle.src.umls_existing_alignment_review import context_summary, recorded_cuis, review_pair


def fixture_pair():
    parent = {"id": "CLM_CONCEPT:p", "preferred_name": "Drug A and Drug B"}
    atom = {
        "id": "CLM_ATOM:a", "preferred_name": "Drug A", "domain_tags": ["drug"], "semantic_types": ["T121"],
        "metadata": {"source_mention_id": parent["id"], "source_text": parent["preferred_name"],
                     "evidence_span": {"field": "preferred_name", "start": 0, "end": 6, "text": "Drug A"},
                     "mapping_status": "auto_accepted_exact", "mapping_count": 1,
                     "atomization_rule": "top_level_composite_split"},
    }
    target = {"id": "CUI:C0000001", "preferred_name": "Drug A", "domain_tags": ["drug"],
              "semantic_types": ["T121"], "external_ids": {"UMLS_CUI": "C0000001"}}
    mappings = [{"mapping_ref": "M1", "record": {"source_id": atom["id"], "target_id": target["id"],
                 "metadata": {"review_status": "auto_accepted_exact", "semantic_compatibility": "compatible", "candidate_overflow": False}}}]
    claims = [{"id": "CLM:c", "metadata": {"negated": True, "source_paper": {"pmid": "1"}, "claim_case_study_ids": ["case2"]}}]
    return atom, target, parent, mappings, context_summary(parent, claims)


def run_pair(items):
    atom, target, parent, mappings, context = items
    return review_pair(atom, target, parent, mappings, claim_context=context, candidate_target_count=1, in_core=bool(mappings))


def test_existing_identity_is_reuse_not_parent_merge():
    items = fixture_pair()
    before = copy.deepcopy(items)
    result = run_pair(items)
    assert result["decision"] == "verified_existing_reuse"
    assert result["existing_direct_mapping_refs"] == ["M1"]
    assert result["merge_authorized"] is False
    assert result["parent_redirect_authorized"] is False
    assert "atom_is_component_not_entire_parent" in result["caution_flags"]
    assert "negated_linked_claim_present" in result["caution_flags"]
    assert items == before


def test_pending_edge_is_never_promoted():
    items = fixture_pair()
    items[0]["metadata"]["mapping_status"] = "needs_review"
    items[3][0]["record"]["metadata"]["review_status"] = "needs_review"
    result = run_pair(items)
    assert result["decision"] == "existing_mapping_needs_review"
    assert result["review_promotion_authorized"] is False


def test_different_recorded_cui_blocks_identity_verification():
    items = fixture_pair()
    items[3][0]["record"]["target_id"] = "CUI:C0000002"
    assert run_pair(items)["decision"] == "review_umls_identity_conflict"


def test_same_cui_with_different_types_is_metadata_review_not_new_entity():
    items = fixture_pair()
    items[1]["semantic_types"] = ["T047"]
    result = run_pair(items)
    assert result["decision"] == "existing_reuse_metadata_mismatch"
    assert result["existing_direct_mapping_refs"] == ["M1"]
    assert result["merge_authorized"] is False


def test_dataset_variable_is_not_generic_entity():
    items = fixture_pair()
    items[1]["id"] = "UKB:123"
    items[1]["external_ids"] = {}
    items[1]["metadata"] = {"parent_dataset": "DATASET:UKB"}
    assert run_pair(items)["decision"] == "review_dataset_or_parcellation_scope"


def test_falff_alias_conflation_is_blocked():
    atom, target, parent, mappings, context = fixture_pair()
    parent["preferred_name"] = "fALFF"
    atom["preferred_name"] = "fALFF"
    atom["metadata"].update(source_text="fALFF", mapping_status="unmapped", mapping_count=0,
                            evidence_span={"field": "preferred_name", "start": 0, "end": 5, "text": "fALFF"})
    target.update(id="IF:alff", preferred_name="amplitude of low-frequency fluctuation", aliases=["ALFF", "fALFF"], external_ids={})
    result = run_pair((atom, target, parent, [], context))
    assert result["decision"] == "blocked_known_alias_conflation"
    assert result["citations"] == ["https://pubmed.ncbi.nlm.nih.gov/18501969/"]


@pytest.mark.parametrize("field,value", [("start", -1), ("end", 99), ("text", "Different"), ("start", False)])
def test_invalid_source_span_is_flagged(field, value):
    items = fixture_pair()
    items[0]["metadata"]["evidence_span"][field] = value
    assert run_pair(items)["decision"] == "review_source_trace"


def test_local_name_is_not_promoted_even_if_unique():
    items = fixture_pair()
    items[0]["metadata"].update(mapping_status="unmapped", mapping_count=0)
    items[1].update(id="LOCAL:a", external_ids={})
    items[3].clear()
    result = run_pair(items)
    assert result["decision"] == "review_local_name"
    assert result["merge_authorized"] is False


def test_cui_comparison_does_not_guess_from_name_or_other_vocab():
    assert recorded_cuis({"id": "CUI:C12345678", "external_ids": {"UMLS_CUI": "C12345678", "NCI": "C111"}}) == {"C12345678"}
    assert recorded_cuis({"id": "IF:a", "preferred_name": "C0000001"}) == set()
