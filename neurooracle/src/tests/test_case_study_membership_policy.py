from __future__ import annotations

import copy

import pytest

from neurooracle.src.case_study_membership_contract import (
    AUDIT_CONTRACT_VERSION,
    build_final_scope_reaudit,
    source_text_sha256,
    validate_final_scope_reaudit,
)
from neurooracle.src.case_study_membership_policy import (
    GATE_NAMES,
    POLICY,
    RUBRIC_PATH,
    RUBRIC_VERSION,
    case_study_policy_prompt,
    gates_for_labels,
    validate_scope_decision,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


def _payload() -> dict:
    sentence = (
        "In Alzheimer disease, APOE epsilon-4 carriers had lower hippocampal "
        "volume than non-carriers."
    )
    return {
        "id": "CLM:policy-test",
        "subject_name": "APOE epsilon-4",
        "predicate": "correlates_with",
        "object_name": "hippocampal volume",
        "negated": False,
        "raw_text": sentence,
        "evidence": {"study_type": "structural MRI"},
        "source_paper": {
            "pmid": "123456",
            "title": "APOE and hippocampal volume",
            "year": 2026,
        },
        "claim_case_study_ids": [
            "case1_transdiagnostic",
            "case2_pathway_mediation",
            "imaging_genetics",
        ],
        "paper_case_study_ids": [
            "case1_transdiagnostic",
            "case2_pathway_mediation",
            "imaging_genetics",
        ],
        "metadata": {
            "subject_type": "gene",
            "object_type": "brain_region",
            "conditions": ["Alzheimer disease"],
            "scope_evidence_spans": [sentence],
        },
    }


def test_policy_is_parsed_from_frozen_rubric_and_registry_order():
    assert RUBRIC_PATH.exists()
    assert RUBRIC_VERSION == "2026-08-10.peer17.v2"
    assert POLICY.case_study_ids == tuple(CASE_STUDY_IDS)
    assert len(POLICY.conditions) == 17
    assert len(GATE_NAMES) == 9
    assert "Disease-versus-healthy control alone is not differential diagnosis" in (
        POLICY.conditions["differential_diagnosis"]
    )
    assert "Ordinary age-related brain change without a brain-age construct" in (
        POLICY.conditions["brain_age"]
    )
    assert "One directly supported link is sufficient" in (
        POLICY.conditions["case2_pathway_mediation"]
    )


def test_extraction_prompt_is_rendered_from_the_same_frozen_policy():
    prompt = case_study_policy_prompt(extraction=True)
    assert RUBRIC_VERSION in prompt
    for case_study_id in CASE_STUDY_IDS:
        assert f"- {case_study_id}: {POLICY.conditions[case_study_id]}" in prompt
    for gate in GATE_NAMES:
        assert gate in prompt


def test_scope_decision_fails_closed_on_missing_or_malformed_gate():
    labels = ["imaging_genetics"]
    gates = gates_for_labels(labels)
    assert list(validate_scope_decision(labels, gates).labels) == [
        "case2_pathway_mediation",
        "imaging_genetics",
    ]

    missing_gate = dict(gates)
    missing_gate["genetic_to_neural_verified"] = False
    with pytest.raises(ValueError, match="genetic_to_neural_verified"):
        validate_scope_decision(labels, missing_gate)

    string_gate = dict(gates)
    string_gate["genetic_to_neural_verified"] = "true"
    with pytest.raises(ValueError, match="JSON booleans"):
        validate_scope_decision(labels, string_gate)

    with pytest.raises(ValueError, match="unknown case-study IDs"):
        validate_scope_decision(["not_a_case_study"], gates)

    with pytest.raises(ValueError, match="requires a direct"):
        gates_for_labels(["case2_pathway_mediation"])


def test_new_claim_scope_contract_detects_evidence_or_decision_tampering():
    payload = _payload()
    labels = payload["claim_case_study_ids"]
    gates = gates_for_labels(labels)
    context_hash = source_text_sha256(payload["raw_text"])
    payload["scope_reaudit"] = build_final_scope_reaudit(
        payload,
        labels=labels,
        gates=gates,
        confidence=0.95,
        decision_basis=(
            "The claim directly links APOE to a disease-related neural phenotype."
        ),
        scope_context_sha256=context_hash,
        reviewer_id="test-reviewer",
        reasoning_effort="high",
        reviewed_at="2026-08-10T00:00:00+00:00",
        source_kind="abstract",
    )

    validated = validate_final_scope_reaudit(payload)
    assert validated["audit_contract_version"] == AUDIT_CONTRACT_VERSION

    optional_chain_analysis = copy.deepcopy(payload)
    optional_chain_analysis["metadata"]["case2_paper_chain_validation"] = {
        "schema_version": "case2_paper_chain_validation.v1",
        "valid": False,
    }
    assert validate_final_scope_reaudit(optional_chain_analysis)

    changed_evidence = copy.deepcopy(payload)
    changed_evidence["predicate"] = "predicts"
    with pytest.raises(ValueError, match="seal mismatch"):
        validate_final_scope_reaudit(changed_evidence)

    changed_labels = copy.deepcopy(payload)
    changed_labels["claim_case_study_ids"] = ["case1_transdiagnostic"]
    with pytest.raises(ValueError, match="seal mismatch"):
        validate_final_scope_reaudit(changed_labels)
