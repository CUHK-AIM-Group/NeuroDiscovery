from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.scripts.summarize_case1_generator_legality import (
    _adapted_rows,
    _raw_rows,
    audit_adapted_group,
    audit_neurodiscovery_group,
)


def test_audit_adapted_group_applies_strict_slot_contract() -> None:
    frame = pd.DataFrame(
        [
            {
                "generated_rank": 1,
                "mapping_status": "mapped",
                "mapped_candidate_id": "fmri|atlas|disease|feature|1",
                "generated_rationale": "rationale",
                "generated_confidence": 0.8,
            },
            {
                "generated_rank": 2,
                "mapping_status": "mapped",
                "mapped_candidate_id": "fmri|atlas|disease|feature|1",
                "generated_rationale": "duplicate",
                "generated_confidence": 0.7,
            },
            {
                "generated_rank": 3,
                "mapping_status": "mapped",
                "mapped_candidate_id": "fmri|atlas|disease|feature|3",
                "generated_rationale": "",
                "generated_confidence": 0.6,
            },
        ]
    )

    result = audit_adapted_group(frame, 4)

    assert result["legal_slots"] == 1
    assert result["legal_rate"] == 0.25
    assert result["complete_legal_batch"] is False
    assert result["errors"].split(";") == [
        "rank_2_duplicate_id",
        "rank_3_missing_rationale",
        "rank_4_count_0",
    ]


def test_audit_adapted_group_accepts_complete_batch() -> None:
    frame = pd.DataFrame(
        [
            {
                "generated_rank": rank,
                "mapping_status": "mapped",
                "mapped_candidate_id": f"fmri|atlas|disease|feature|{rank}",
                "generated_rationale": "rationale",
                "generated_confidence": 0.5,
            }
            for rank in range(1, 4)
        ]
    )

    result = audit_adapted_group(frame, 3)

    assert result["legal_slots"] == 3
    assert result["legal_rate"] == 1.0
    assert result["complete_legal_batch"] is True
    assert result["errors"] == ""


def test_adapted_rows_counts_missing_trial_as_zero_legal_slots(tmp_path: Path) -> None:
    mapped = tmp_path / "mapped.csv"
    pd.DataFrame(
        [
            {
                "method": "ai_scientist_v2",
                "seed": 0,
                "generated_rank": 1,
                "mapping_status": "mapped",
                "mapped_candidate_id": "candidate-1",
                "generated_rationale": "rationale",
                "generated_confidence": 0.5,
            }
        ]
    ).to_csv(mapped, index=False)

    rows = _adapted_rows(
        [mapped],
        methods=["ai_scientist_v2"],
        trials=[0, 1],
        matched_slots=1,
        full_slots=1,
    )

    full_rows = [row for row in rows if row["condition"] == "adapted_full"]
    assert [row["legal_slots"] for row in full_rows] == [1, 0]
    assert full_rows[1]["errors"] == "rank_1_count_0"


def test_audit_neurodiscovery_group_uses_only_registered_native_structure() -> None:
    valid = {
        "candidate_id": "candidate-1",
        "hypothesis_id": "candidate-1",
        "case_study_id": "case1_transdiagnostic",
        "candidate_tuple": {
            "disease": "disease",
            "feature_family": "amplitude",
            "map_group": "network",
            "roi_key": "atlas|1",
            "source": "atlas",
        },
        "status": "supported",
        "statistics": {"effect_size": 99.0, "p_value": 0.0},
    }
    duplicate = dict(valid)

    result = audit_neurodiscovery_group(
        [valid, duplicate], {"candidate-1"}, requested=3
    )

    assert result["legal_slots"] == 1
    assert result["errors"].split(";") == [
        "rank_2_duplicate_id",
        "rank_3_count_0",
    ]


def test_raw_rows_count_unsupported_interface_as_zero_slots(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "method": "sciagents",
                "trial": 0,
                "process_status": "unsupported_official_interface",
                "schema_valid_slots": 0,
                "error": "no generic entry point",
            }
        ]
    ).to_csv(tmp_path / "unadapted_trial_audit.csv", index=False)

    rows = _raw_rows(tmp_path, methods=["sciagents"], requested=100)

    assert rows[0]["measurement_status"] == "measured"
    assert rows[0]["requested_slots"] == 100
    assert rows[0]["legal_slots"] == 0
    assert rows[0]["legal_rate"] == 0.0
