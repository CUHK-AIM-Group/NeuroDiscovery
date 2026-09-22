from __future__ import annotations

import json

from core.scripts.run_case1_unadapted_format_audit import evaluate_artifact


def candidate(index: int) -> str:
    return f"fmri|atlas_multiatlas|MDD|roi_alff_proxy|{index}"


def test_exact_strict_payload_is_fully_valid() -> None:
    artifact = {
        "hypotheses": [
            {
                "rank": index,
                "candidate_id": candidate(index),
                "rationale": "test",
                "confidence": 0.7,
            }
            for index in range(1, 4)
        ]
    }
    audit = evaluate_artifact(
        artifact,
        known_ids={candidate(index) for index in range(1, 4)},
        requested=3,
    )
    assert audit["schema_valid_slots"] == 3
    assert audit["complete_schema_success"] is True
    assert audit["exact_unique_registered_ids_anywhere"] == 3


def test_prose_ids_do_not_count_as_schema_valid() -> None:
    artifact = {"summary": f"Consider {candidate(1)} and {candidate(2)}."}
    audit = evaluate_artifact(
        artifact,
        known_ids={candidate(1), candidate(2)},
        requested=2,
    )
    assert audit["schema_valid_slots"] == 0
    assert audit["exact_unique_registered_ids_anywhere"] == 2


def test_embedded_complete_json_is_extracted_without_repair() -> None:
    payload = {
        "hypotheses": [
            {
                "rank": 1,
                "candidate_id": candidate(1),
                "rationale": "test",
                "confidence": 0.5,
            }
        ]
    }
    artifact = {"final": "Result:\n" + json.dumps(payload)}
    audit = evaluate_artifact(
        artifact,
        known_ids={candidate(1)},
        requested=1,
    )
    assert audit["schema_valid_slots"] == 1
    assert audit["strict_payloads_found"] >= 1


def test_duplicate_and_missing_rank_fail_slots() -> None:
    artifact = {
        "hypotheses": [
            {
                "rank": 1,
                "candidate_id": candidate(1),
                "rationale": "test",
                "confidence": 0.5,
            },
            {
                "rank": 1,
                "candidate_id": candidate(2),
                "rationale": "test",
                "confidence": 0.5,
            },
        ]
    }
    audit = evaluate_artifact(
        artifact,
        known_ids={candidate(1), candidate(2)},
        requested=2,
    )
    assert audit["schema_valid_slots"] == 0
    assert audit["complete_schema_success"] is False
