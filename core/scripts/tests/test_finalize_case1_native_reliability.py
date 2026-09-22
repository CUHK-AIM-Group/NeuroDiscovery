from __future__ import annotations

import pandas as pd

from core.scripts.finalize_case1_native_reliability import (
    neuroruntime_completion,
    normalize_method,
    parse_valid_trials,
    read_execution_summaries,
)


def test_valid_trial_alias_is_normalized() -> None:
    assert parse_valid_trials(["ai_scientist_v2_native:104"]) == {
        ("ai_scientist_v2", 104)
    }
    assert normalize_method("brainpilot_native") == "brainpilot_native"


def test_neuroruntime_completion_requires_all_finite_fields(tmp_path) -> None:
    path = tmp_path / "all_tests.csv"
    pd.DataFrame(
        {
            "adjusted_residual_d": [0.2, float("nan"), 0.1],
            "p_value": [0.01, 0.02, 0.03],
            "q_fdr_global": [0.02, 0.03, float("inf")],
        }
    ).to_csv(path, index=False)

    attempted, successful, rate = neuroruntime_completion(path)

    assert attempted == 3
    assert successful == 1
    assert rate == 1 / 3


def test_missing_native_result_is_not_a_returned_process(tmp_path) -> None:
    trial_dir = tmp_path / "method" / "trial_7"
    trial_dir.mkdir(parents=True)
    (trial_dir / "execution_summary.json").write_text(
        """{
          "method": "ai_scientist_v2_native",
          "trial": 7,
          "attempted_candidates": 12,
          "successful_candidates": 0,
          "result_file": "missing-final.txt"
        }""",
        encoding="utf-8",
    )

    rows = read_execution_summaries(tmp_path)

    assert rows[0]["method"] == "ai_scientist_v2"
    assert rows[0]["framework_process_returned"] is False
