from __future__ import annotations

from copy import deepcopy

from neurooracle.scripts.map_case2_component_membership import (
    CASE2_ID,
    MAPPING_ID,
    PROJECTION_FIELD,
    apply_claim_projection,
    projected_labels,
)


def claim(claim_labels: list[str], paper_labels: list[str]) -> dict:
    return {
        "id": "CLM:test",
        "claim_case_study_ids": list(claim_labels),
        "paper_case_study_ids": list(paper_labels),
        "metadata": {
            "claim_case_study_ids": list(claim_labels),
            "paper_case_study_ids": list(paper_labels),
        },
        "scope_reaudit": {"decision_sha256": "immutable-old-seal"},
    }


def test_projected_labels_adds_case2_for_each_component() -> None:
    for component in ("imaging_genetics", "progression_prediction", "prognosis"):
        assert CASE2_ID in projected_labels([component])


def test_claim_and_paper_projection_preserves_old_scope_audit() -> None:
    row = claim(
        ["case1_transdiagnostic", "imaging_genetics"],
        ["case1_transdiagnostic", "imaging_genetics", "prognosis"],
    )
    old_audit = deepcopy(row["scope_reaudit"])
    result = apply_claim_projection(row, applied_at="2026-08-11T00:00:00+00:00")
    assert result["claim_changed"] is True
    assert result["paper_changed"] is True
    assert row["scope_reaudit"] == old_audit
    assert CASE2_ID in row["claim_case_study_ids"]
    assert CASE2_ID in row["paper_case_study_ids"]
    assert row["metadata"]["claim_case_study_ids"] == row["claim_case_study_ids"]
    assert row["metadata"]["paper_case_study_ids"] == row["paper_case_study_ids"]
    records = row["metadata"][PROJECTION_FIELD]
    assert len(records) == 1
    assert records[0]["mapping_id"] == MAPPING_ID
    assert records[0]["semantic_reaudit_deferred"] is True


def test_paper_component_does_not_donate_claim_component() -> None:
    row = claim(
        ["biomarker_discovery"],
        ["biomarker_discovery", "imaging_genetics"],
    )
    result = apply_claim_projection(row, applied_at="2026-08-11T00:00:00+00:00")
    assert result["claim_changed"] is False
    assert result["paper_changed"] is True
    assert CASE2_ID not in row["claim_case_study_ids"]
    assert CASE2_ID in row["paper_case_study_ids"]


def test_legacy_strict_case2_is_preserved_without_component() -> None:
    row = claim([CASE2_ID], [CASE2_ID])
    result = apply_claim_projection(row, applied_at="2026-08-11T00:00:00+00:00")
    assert result["claim_changed"] is False
    assert result["paper_changed"] is False
    assert row["claim_case_study_ids"] == [CASE2_ID]
    assert PROJECTION_FIELD not in row["metadata"]

