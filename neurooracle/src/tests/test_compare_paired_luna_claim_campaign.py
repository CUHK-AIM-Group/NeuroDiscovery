from __future__ import annotations

from copy import deepcopy

from neurooracle.scripts.compare_paired_luna_claim_campaign import classify_decision


def claim() -> dict:
    return {
        "subject_name": "hippocampal volume",
        "predicate": "predicts",
        "object_name": "memory decline",
        "negated": False,
        "confidence": 0.5,
        "raw_text": "Lower hippocampal volume predicted subsequent memory decline.",
        "evidence": {
            "study_type": "longitudinal cohort",
            "sample_size": 100,
        },
        "paper_case_study_ids": ["progression_prediction"],
        "claim_case_study_ids": ["progression_prediction"],
        "metadata": {
            "subject_type": "brain_region",
            "object_type": "clinical_event",
            "conditions": ["Alzheimer disease"],
            "population": "older adults",
            "raw_stats": {"p_value": 0.01},
            "scope_evidence_spans": [
                "Lower hippocampal volume predicted subsequent memory decline."
            ],
            "scope_confidence": 0.95,
            "scope_decision_basis": "This is a longitudinal imaging-to-outcome result.",
        },
        "scope_reaudit": {
            "gates": {
                "longitudinal_verified": True,
            },
            "confidence": 0.95,
            "decision_basis": "This is a longitudinal imaging-to-outcome result.",
        },
    }


def paper() -> dict:
    return {"paper_case_study_ids": ["progression_prediction"]}


def test_identical_claim_and_paper_scope_are_exact() -> None:
    row = claim()
    decision, _ = classify_decision(
        [row], [deepcopy(row)], left_paper=paper(), right_paper=paper()
    )
    assert decision == "exact_agreement"


def test_scope_confidence_and_basis_are_part_of_exact_agreement() -> None:
    left = claim()
    right = deepcopy(left)
    right["metadata"]["scope_confidence"] = 0.7
    right["scope_reaudit"]["confidence"] = 0.7
    right["metadata"]["scope_decision_basis"] = "A different scope rationale."
    right["scope_reaudit"]["decision_basis"] = "A different scope rationale."
    decision, _ = classify_decision(
        [left], [right], left_paper=paper(), right_paper=paper()
    )
    assert decision == "scope_only_disagreement"


def test_entity_type_and_evidence_payload_are_not_exact() -> None:
    left = claim()
    right = deepcopy(left)
    right["metadata"]["subject_type"] = "biomarker"
    right["evidence"]["sample_size"] = 101
    decision, _ = classify_decision(
        [left], [right], left_paper=paper(), right_paper=paper()
    )
    assert decision == "evidence_aligned_semantic_disagreement"


def test_paper_level_scope_union_is_compared() -> None:
    row = claim()
    decision, _ = classify_decision(
        [row],
        [deepcopy(row)],
        left_paper=paper(),
        right_paper={"paper_case_study_ids": []},
    )
    assert decision == "scope_only_disagreement"
