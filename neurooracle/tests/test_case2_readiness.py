from __future__ import annotations

import pandas as pd
import pytest

from neurooracle.scripts.audit_case2_readiness import (
    approximate_joint_path_power,
    build_master_candidate_registry,
    build_protocol_lock,
    select_master_pathways,
    unseen_manifest_is_eligible,
)


def _subjects() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["S1", "S2"],
            "score_a_strict": [0.1, 0.2],
            "score_a_wide": [0.3, 0.4],
            "score_b_wide": [0.5, 0.6],
        }
    )


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score_name": "score_a_wide",
                "pathway_id": "path_a",
                "pathway_source": "source",
                "pathway_name": "Path A",
                "threshold_label": "p5em02",
            },
            {
                "score_name": "score_a_strict",
                "pathway_id": "path_a",
                "pathway_source": "source",
                "pathway_name": "Path A",
                "threshold_label": "p1em03",
            },
            {
                "score_name": "score_b_wide",
                "pathway_id": "path_b",
                "pathway_source": "source",
                "pathway_name": "Path B",
                "threshold_label": "p5em02",
            },
        ]
    )


def test_pathway_selection_and_registry_are_deterministic() -> None:
    pathways = select_master_pathways(_subjects(), _manifest())
    assert pathways["score_name"].tolist() == ["score_a_strict", "score_b_wide"]
    registry = build_master_candidate_registry(
        pathways,
        marker_specs=("smri::hippocampus", "pet::amyloid"),
        outcomes=("memory", "function"),
    )
    assert len(registry) == 8
    assert registry["candidate_id"].is_unique
    assert not any("effect" in column for column in registry.columns)


def test_power_approximation_increases_with_n_and_effect() -> None:
    low_n = approximate_joint_path_power(200, 0.15)
    high_n = approximate_joint_path_power(600, 0.15)
    high_effect = approximate_joint_path_power(200, 0.20)
    assert 0.0 <= low_n < high_n <= 1.0
    assert high_effect > low_n
    with pytest.raises(ValueError):
        approximate_joint_path_power(0, 0.15)


def test_unseen_manifest_fails_closed() -> None:
    eligible = {
        "cohort_id": "independent",
        "independent_of_development": True,
        "case2_outcomes_previously_accessed": False,
        "genetics_available": True,
        "imaging_available": True,
        "longitudinal_outcomes_available": True,
        "minimum_complete_case_n": 400,
        "executable_candidate_count": 1000,
        "distinct_pathways": 15,
        "distinct_markers": 8,
        "distinct_outcomes": 6,
        "data_manifest_sha256": "a" * 64,
    }
    assert unseen_manifest_is_eligible(eligible)
    assert not unseen_manifest_is_eligible(
        {**eligible, "case2_outcomes_previously_accessed": True}
    )


def test_protocol_lock_distinguishes_ranking_from_temporal_freeze() -> None:
    lock = build_protocol_lock(
        registry_sha256="b" * 64,
        registry_count=5265,
        development_metadata={"development_subjects": 691},
        final_holdout_ready=False,
        unseen_manifest=None,
    )
    assert lock["status"] == "design_locked_pending_unseen_cohort"
    assert lock["outcome_access"]["permitted"] is False
    assert lock["ranking_freeze"]["is_temporal_kg_freeze"] is False
    assert lock["public_version_label"] is None
    assert len(lock["protocol_hash"]) == 64


# Last Updated At: 2026-08-16 17:46 HKT
