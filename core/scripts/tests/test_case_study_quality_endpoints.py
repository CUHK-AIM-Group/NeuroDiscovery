from __future__ import annotations

import pandas as pd

from core.scripts.case_study_quality_endpoints import derive_quality_outcomes


def test_quality_endpoint_preserves_base_label_and_is_method_independent() -> None:
    internal = pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(8)],
            "validated": [True] * 7 + [False],
            "strict_validated": [True] * 7 + [False],
            "r_ci_low": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1],
            "mae_improvement_ci_low": [1, 2, 3, 4, 5, 6, 7, -1],
            "method": ["ignored"] * 8,
        }
    )

    derived, audit = derive_quality_outcomes(internal, quantile=0.75)

    assert derived["base_validated"].sum() == 7
    assert derived["validated"].sum() == 2
    assert derived.loc[derived["validated"], "candidate_id"].tolist() == ["c5", "c6"]
    assert set(derived["feedback_status"]) == {"supported", "inconclusive"}
    assert derived["feedback_utility"].between(0.0, 1.0).all()
    assert audit["selection_uses_method_identity"] is False
    assert audit["selection_uses_generator_scores"] is False


def test_quality_endpoint_requires_finite_robust_metrics() -> None:
    internal = pd.DataFrame(
        {
            "candidate_id": ["c0", "c1"],
            "validated": [True, True],
            "r_ci_low": [0.1, float("nan")],
            "mae_improvement_ci_low": [1.0, 2.0],
        }
    )

    try:
        derive_quality_outcomes(internal)
    except ValueError as error:
        assert "r_ci_low" in str(error)
    else:
        raise AssertionError("non-finite robust metric was accepted")
