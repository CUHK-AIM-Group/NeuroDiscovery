from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurooracle.scripts.tune_case2_neurodiscovery_confirmatory_policy import (
    PRIOR_PROFILES,
    generator_tuning_outcomes,
    heldout_feedback_outcomes,
    prior_score,
)
from neurooracle.scripts.case2_confirmatory_statistics import (
    global_chain_labels,
    pathway_outcome_family_labels,
)


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mapping_score": [0.1, 0.4, 0.9],
            "meaningful_support_count": [0, 2, 8],
            "primary_chain_alignment": [0.2, 0.5, 0.8],
            "primary_kg_score": [0.9, 0.4, 0.1],
        }
    )


def test_all_prespecified_prior_profiles_are_valid() -> None:
    for profile in PRIOR_PROFILES:
        scores = prior_score(_components(), dict(profile["weights"]))
        assert scores.shape == (3,)
        assert np.isfinite(scores).all()


def test_prior_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        prior_score(
            _components(),
            {"mapping": 1.0, "support": 0.5, "chain": 0.0, "kg": 0.0},
        )


def test_heldout_outcome_labels_are_not_available_to_feedback() -> None:
    public = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "outcome": ["MMSE", "ADAS13", "MMSE"],
        }
    )
    feedback = heldout_feedback_outcomes(
        public,
        np.asarray([True, True, False]),
        heldout_outcome="MMSE",
    )
    assert feedback["validated"].tolist() == [False, True, False]
    assert feedback["feedback_status"].tolist() == [
        "inconclusive",
        "supported",
        "inconclusive",
    ]


def test_generator_tuning_excludes_phase_protocol_target_outcomes() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "neurooracle"
        / "configs"
        / "case2_adni_phase_holdout_v3.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert generator_tuning_outcomes(protocol) == ("MMSE", "ADAS13", "CDRSB")


def test_family_fdr_is_computed_within_pathway_and_outcome() -> None:
    results = pd.DataFrame(
        {
            "exposure": ["A"] * 3 + ["B"] * 3,
            "outcome": ["FAQ"] * 6,
            "sobel_p": [0.001, 0.020, 0.900, 0.040, 0.050, 0.060],
            "a_path_p": [0.001] * 6,
            "b_path_p": [0.001] * 6,
        }
    )
    labels, q_values = pathway_outcome_family_labels(results)

    assert labels.tolist() == [True, True, False, False, False, False]
    assert q_values[0] < 0.05
    assert q_values[3] > 0.05


def test_non_estimable_candidates_remain_in_registered_fdr_families() -> None:
    results = pd.DataFrame(
        {
            "exposure": ["A"] * 10,
            "outcome": ["FAQ"] * 10,
            "sobel_p": [0.006] + [np.nan] * 9,
            "a_path_p": [0.001] * 10,
            "b_path_p": [0.001] * 10,
        }
    )

    family_labels, family_q = pathway_outcome_family_labels(results)
    global_labels, global_q = global_chain_labels(results)

    assert not family_labels.any()
    assert not global_labels.any()
    assert family_q[0] == pytest.approx(0.06)
    assert global_q[0] == pytest.approx(0.06)
    assert np.all(family_q[1:] == 1.0)


# Last Updated At: 2026-08-16 13:06 HKT
