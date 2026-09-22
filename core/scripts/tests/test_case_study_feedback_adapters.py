from __future__ import annotations

import pytest

from core.scripts.case_study_feedback_adapters import (
    COMPLETED_CASE_STUDY_LINES,
    CONTRADICTED,
    EXECUTION_FAILED,
    INCONCLUSIVE,
    SUPPORTED,
    adapter_for,
)


@pytest.mark.parametrize("task", COMPLETED_CASE_STUDY_LINES)
def test_every_completed_line_projects_semantic_experimental_claims(task: str) -> None:
    adapter = adapter_for(task)
    candidate = {field: f"{field}-value" for field in adapter.factor_fields}

    assertions = adapter.semantic_assertions(candidate)

    assert assertions
    assert all(assertion["subject_id"].startswith("EXP_ATOM:") for assertion in assertions)
    assert all(assertion["object_id"].startswith("EXP_ATOM:") for assertion in assertions)
    assert all(assertion["predicate"] for assertion in assertions)


def test_case1_and_biomarker_discovery_are_independent_scopes() -> None:
    case1 = adapter_for("case1_transdiagnostic")
    biomarker = adapter_for("biomarker_discovery")

    assert case1 is not biomarker
    assert case1.case_study_id == "case1_transdiagnostic"
    assert biomarker.case_study_id == "biomarker_discovery"


def test_case2_feedback_is_a_complete_two_assertion_chain() -> None:
    adapter = adapter_for("case2_pathway_mediation")
    assertions = adapter.semantic_assertions(
        {
            "exposure": "APOE e4",
            "modality": "MRI",
            "marker": "hippocampal volume",
            "outcome": "cognitive decline",
        }
    )

    assert adapter.requires_complete_chain is True
    assert [assertion["predicate"] for assertion in assertions] == [
        "affects",
        "mediates",
    ]
    assert assertions[0]["object_id"] == assertions[1]["subject_id"]


def test_prognosis_feedback_uses_the_concrete_marker_atom() -> None:
    adapter = adapter_for(
        "prognosis",
        factor_fields=("outcome", "horizon", "feature_family", "marker", "model"),
    )
    assertions = adapter.semantic_assertions(
        {
            "outcome": "MCI to dementia",
            "horizon": "5y",
            "feature_family": "structural MRI",
            "marker": "hippocampal volume",
            "model": "adjusted Cox",
        }
    )

    assert len(assertions) == 1
    assert assertions[0]["predicate"] == "predicts_prognosis"
    assert "hippocampal volume" in assertions[0]["subject_name"]


def test_feedback_classifier_preserves_four_distinct_states() -> None:
    adapter = adapter_for("brain_age")

    assert adapter.classify_feedback({"validated": True}) == SUPPORTED
    assert adapter.classify_feedback({"validated": False}) == INCONCLUSIVE
    assert adapter.classify_feedback({"error": "runtime failed"}) == EXECUTION_FAILED
    assert (
        adapter.classify_feedback(
            {
                "validated": False,
                "expected_direction": "decrease",
                "effect_size": 0.4,
                "q_value": 0.01,
            }
        )
        == CONTRADICTED
    )
    assert (
        adapter.classify_feedback(
            {
                "validated": False,
                "expected_direction": "decrease",
                "effect_size": 0.4,
                "p_value": 0.001,
            }
        )
        == INCONCLUSIVE
    )
