from __future__ import annotations

from neurooracle.scripts.audit_dynamic_hindcasting_eligibility import (
    _reject_performance_columns,
    classify_dynamic_window,
)

import pytest


def _classify(**overrides):
    values = {
        "has_terminal_interval": True,
        "complete_path_required": False,
        "early_unique_pairs": 2,
        "early_complete_components": 0,
        "terminal_unique_pairs": 10,
        "terminal_complete_components": 0,
        "min_early_support": 2,
        "min_terminal_pairs": 10,
    }
    values.update(overrides)
    return classify_dynamic_window(**values)


def test_dynamic_primary_requires_early_feedback_and_stable_terminal_evidence() -> None:
    assert _classify() == ("executable", "", "primary")
    status, reason, tier = _classify(early_unique_pairs=1)
    assert (status, tier) == ("non_executable", "excluded")
    assert "early" in reason


def test_dynamic_sparse_terminal_is_exploratory() -> None:
    status, reason, tier = _classify(terminal_unique_pairs=9)
    assert (status, tier) == ("sparse", "exploratory")
    assert "terminal" in reason


def test_dynamic_complete_path_requires_both_temporal_components() -> None:
    status, reason, tier = _classify(
        complete_path_required=True,
        early_complete_components=0,
        terminal_complete_components=1,
    )
    assert (status, tier) == ("non_executable", "excluded")
    assert "early evidence" in reason

    status, reason, tier = _classify(
        complete_path_required=True,
        early_complete_components=1,
        terminal_complete_components=0,
    )
    assert (status, tier) == ("non_executable", "excluded")
    assert "terminal evidence" in reason

    assert _classify(
        complete_path_required=True,
        early_complete_components=1,
        terminal_complete_components=1,
    ) == ("executable", "", "primary")


def test_dynamic_requires_disjoint_terminal_interval() -> None:
    status, reason, tier = _classify(has_terminal_interval=False)
    assert (status, tier) == ("non_executable", "excluded")
    assert "terminal interval" in reason


def test_dynamic_eligibility_rejects_method_performance_columns() -> None:
    _reject_performance_columns(
        ["case_study_id", "future_unique_pairs", "analysis_tier"]
    )
    with pytest.raises(ValueError, match="method-performance columns"):
        _reject_performance_columns(
            ["case_study_id", "neurodiscovery_recall", "analysis_tier"]
        )
