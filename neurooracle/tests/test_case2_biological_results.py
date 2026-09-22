from __future__ import annotations

import pandas as pd

from neurooracle.scripts.export_case2_biological_results import (
    build_source_outcome_summary,
    classify_results,
)


def test_classify_results_uses_global_then_family_then_nominal_tiers() -> None:
    results = pd.DataFrame(
        {
            "indirect_bootstrap_p": [0.001, 0.002, 0.03, 0.20],
            "indirect_bootstrap_q_global_168": [0.04, 0.20, 0.40, 0.90],
            "indirect_bootstrap_q_family_8": [0.01, 0.03, 0.20, 0.80],
        }
    )

    assert classify_results(results).tolist() == [
        "primary_global_fdr",
        "supplemental_family_fdr",
        "nominal_only",
        "not_nominal",
    ]


def test_source_outcome_summary_preserves_raw_score_orientation() -> None:
    outcomes = pd.DataFrame(
        {
            "subject_id": ["S1", "S2"] * 3,
            "outcome": ["ADAS13"] * 2 + ["FAQ"] * 2 + ["LDELTOTAL"] * 2,
            "value": [10.0, 20.0, 0.0, 4.0, 5.0, 15.0],
        }
    )

    summary = build_source_outcome_summary(outcomes).set_index("outcome")

    assert summary.loc["ADAS13", "raw_mean"] == 15.0
    assert summary.loc["FAQ", "raw_mean"] == 2.0
    assert summary.loc["LDELTOTAL", "raw_mean"] == 10.0
    assert summary.loc["ADAS13", "raw_score_direction"] == "higher_is_worse"
    assert (
        summary.loc["LDELTOTAL", "raw_score_direction"]
        == "higher_is_better; multiplied_by_-1_for_model"
    )
