from __future__ import annotations

from core.scripts.run_case1_generator_executor_factorial import (
    strict_payload_candidates,
)


def test_strict_payload_candidates_requires_complete_ranked_schema():
    ids = {"fmri|atlas|disease|feature|0", "fmri|atlas|disease|feature|1"}
    artifact = {
        "wrapped": {
            "hypotheses": [
                {
                    "rank": 1,
                    "candidate_id": "fmri|atlas|disease|feature|0",
                    "rationale": "first",
                    "confidence": 0.8,
                },
                {
                    "rank": 2,
                    "candidate_id": "fmri|atlas|disease|feature|1",
                    "rationale": "second",
                    "confidence": 0.7,
                },
            ]
        }
    }

    assert strict_payload_candidates(artifact, known_ids=ids, requested=2) == [
        "fmri|atlas|disease|feature|0",
        "fmri|atlas|disease|feature|1",
    ]


def test_strict_payload_candidates_stops_at_invalid_slot():
    ids = {"fmri|atlas|disease|feature|0"}
    artifact = {
        "hypotheses": [
            {
                "rank": 1,
                "candidate_id": "fmri|atlas|disease|feature|0",
                "rationale": "first",
                "confidence": 0.8,
            },
            {
                "rank": 2,
                "candidate_id": "not-registered",
                "rationale": "second",
                "confidence": 0.7,
            },
        ]
    }

    assert strict_payload_candidates(artifact, known_ids=ids, requested=2) == [
        "fmri|atlas|disease|feature|0"
    ]
