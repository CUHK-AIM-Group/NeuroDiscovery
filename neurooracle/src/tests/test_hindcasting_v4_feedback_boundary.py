from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from neurooracle.src.hindcasting_v4_feedback_boundary import (
    COMPUTATIONAL_FEEDBACK_SCHEMA,
    DISCOVERY_INPUT_SCHEMA,
    EVALUATION_HANDOFF_SCHEMA,
    LeakageBoundaryError,
    assert_discovery_payload_blind,
    build_discovery_seal,
    scan_discovery_source,
    validate_computational_feedback,
    validate_discovery_input_manifest,
    validate_evaluation_handoff,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _input_manifest() -> dict:
    return {
        "schema_version": DISCOVERY_INPUT_SCHEMA,
        "freeze_year": 2018,
        "generator_mode": "deterministic_frozen",
        "llm_api_enabled": False,
        "evaluation_data_mounted": False,
        "evaluation_module_imported": False,
        "inputs": [
            {
                "role": "historical_kg",
                "path": "inputs/kg_2018.json",
                "sha256": SHA_A,
                "max_publication_year": 2018,
            },
            {
                "role": "task_schema",
                "path": "inputs/task.json",
                "sha256": SHA_B,
            },
            {
                "role": "deterministic_generator",
                "path": "source/generator.py",
                "sha256": SHA_C,
            },
            {
                "role": "computational_dataset",
                "path": "inputs/tcp.csv",
                "sha256": SHA_D,
                "selected_without_retrospective_evaluation": True,
            },
        ],
    }


def _feedback() -> dict:
    return {
        "schema_version": COMPUTATIONAL_FEEDBACK_SCHEMA,
        "source_kind": "computational_experiment",
        "hypothesis_id": "H-001",
        "experiment_id": "EXP-001",
        "task_id": "biomarker_discovery",
        "dataset_id": "TCP-frozen",
        "executor": "tcp_case_control_glm",
        "dataset_sha256": SHA_A,
        "analysis_plan_sha256": SHA_B,
        "pipeline_sha256": SHA_C,
        "selection_commit_sha256": SHA_D,
        "outcome_observed_after_selection": True,
        "publication_evaluation_fields_read": False,
        "execution_status": "succeeded",
        "feedback_status": "supported",
        "feedback_available": True,
        "statistics": {
            "effect_size": 0.31,
            "q_value": 0.012,
            "cv_auc_mean": 0.59,
            "direction_stability": 0.8,
        },
    }


def test_accepts_blinded_discovery_manifest() -> None:
    audit = validate_discovery_input_manifest(_input_manifest())
    assert audit["status"] == "passed"
    assert audit["freeze_year"] == 2018
    assert audit["llm_api_enabled"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("llm_api_enabled", True), ("evaluation_data_mounted", True)),
)
def test_discovery_manifest_fails_closed(field: str, value: bool) -> None:
    manifest = _input_manifest()
    manifest[field] = value
    with pytest.raises(LeakageBoundaryError):
        validate_discovery_input_manifest(manifest)


def test_rejects_post_cutoff_historical_evidence() -> None:
    manifest = _input_manifest()
    manifest["inputs"][0]["max_publication_year"] = 2019
    with pytest.raises(LeakageBoundaryError, match="after the cutoff"):
        validate_discovery_input_manifest(manifest)


def test_rejects_computational_resource_selected_using_future_evaluation() -> None:
    manifest = _input_manifest()
    manifest["inputs"][-1]["selected_without_retrospective_evaluation"] = False
    with pytest.raises(LeakageBoundaryError, match="selection is not blinded"):
        validate_discovery_input_manifest(manifest)


@pytest.mark.parametrize(
    "payload",
    (
        {"primary_hit": True},
        {"outcome": {"first_future_year": 2020}},
        {"source_kind": "future_publication"},
        {"paper": {"publication_year": 2019}},
        {"supporting_pmids": ["12345678"]},
    ),
)
def test_rejects_publication_evaluation_payloads(payload: dict) -> None:
    with pytest.raises(LeakageBoundaryError):
        assert_discovery_payload_blind(payload, freeze_year=2018)


def test_accepts_own_computational_feedback() -> None:
    audit = validate_computational_feedback(
        _feedback(), freeze_year=2018, committed_candidate_ids=["H-001"]
    )
    assert audit == {
        "status": "passed",
        "hypothesis_id": "H-001",
        "execution_status": "succeeded",
        "feedback_status": "supported",
    }


def test_rejects_future_publication_label_hidden_inside_feedback() -> None:
    feedback = _feedback()
    feedback["primary_hit"] = True
    feedback["primary_year"] = 2019
    with pytest.raises(LeakageBoundaryError, match="retrospective evaluation field"):
        validate_computational_feedback(feedback, freeze_year=2018)


def test_failed_execution_cannot_be_converted_to_scientific_support() -> None:
    feedback = _feedback()
    feedback["execution_status"] = "failed"
    with pytest.raises(LeakageBoundaryError, match="must remain execution_failed"):
        validate_computational_feedback(feedback, freeze_year=2018)


def test_feedback_must_follow_exact_candidate_commitment() -> None:
    with pytest.raises(LeakageBoundaryError, match="committed selection"):
        validate_computational_feedback(
            _feedback(), freeze_year=2018, committed_candidate_ids=["H-002"]
        )


def test_v4_boundary_source_has_no_evaluator_dependency() -> None:
    source = Path(__file__).parents[1] / "hindcasting_v4_feedback_boundary.py"
    assert scan_discovery_source(source)["status"] == "passed"


def test_v3_dynamic_runner_is_detected_as_leaking() -> None:
    source = (
        Path(__file__).parents[2]
        / "scripts"
        / "run_neurodiscovery_dynamic_closed_loop_hindcasting.py"
    )
    with pytest.raises(LeakageBoundaryError, match="retrospective evaluation code"):
        scan_discovery_source(source)


def test_evaluator_may_open_only_after_untampered_discovery_seal() -> None:
    seal = build_discovery_seal(
        freeze_year=2018,
        task_id="biomarker_discovery",
        method="neurodiscovery",
        seed=0,
        ordered_hypothesis_ids=["H-001", "H-002"],
        discovery_input_manifest_sha256=SHA_A,
        discovery_source_bundle_sha256=SHA_B,
        feedback_chain_sha256=SHA_C,
    )
    handoff = {
        "schema_version": EVALUATION_HANDOFF_SCHEMA,
        "discovery_seal_sha256": seal["discovery_seal_sha256"],
        "evaluation_loaded_after_discovery_seal": True,
    }
    assert validate_evaluation_handoff(seal, handoff)["status"] == "passed"

    tampered = deepcopy(seal)
    tampered["ordered_hypothesis_ids"].reverse()
    with pytest.raises(LeakageBoundaryError, match="hash verification failed"):
        validate_evaluation_handoff(tampered, handoff)
