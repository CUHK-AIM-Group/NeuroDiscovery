from __future__ import annotations

import numpy as np
import pandas as pd

from core.scripts.run_adni_closed_loop_experiments import (
    PenalizedCox,
    association_test,
    censored_at_horizon,
    classification_outer_splits,
    normalize_diagnosis,
    prognostic_marker_association,
    prognostic_marker_columns,
    progression_at_horizon,
    survival_outer_splits,
)


def test_diagnosis_normalization() -> None:
    assert normalize_diagnosis("EMCI") == "MCI"
    assert normalize_diagnosis("LMCI") == "MCI"
    assert normalize_diagnosis("Dementia") == "Dementia"
    assert normalize_diagnosis("CN") == "CN"


def test_horizon_outcomes_do_not_call_short_followup_controls_negative() -> None:
    frame = pd.DataFrame(
        {
            "event": [1, 1, 0, 0],
            "duration_years": [1.5, 4.0, 4.0, 1.0],
            "followup_years": [1.5, 4.0, 4.0, 1.0],
        }
    )
    result = progression_at_horizon(frame, 3)
    assert result["label"].tolist() == [1, 0, 0]
    duration, event = censored_at_horizon(frame, 3)
    np.testing.assert_allclose(duration, [1.5, 3.0, 3.0, 1.0])
    np.testing.assert_array_equal(event, [1, 0, 0, 0])


def test_adjusted_association_recovers_positive_direction() -> None:
    rng = np.random.default_rng(42)
    covariates = rng.normal(size=(200, 3))
    x = rng.normal(size=200) + covariates[:, 0]
    y = 0.5 * x + covariates[:, 1] + rng.normal(scale=0.5, size=200)
    for model in ("association_glm", "rank_association", "robust_huber"):
        result = association_test(x, y, covariates, model)
        assert result["n"] == 200
        assert result["effect"] > 0
        assert result["p_value"] < 0.05


def test_penalized_cox_fits_and_predicts() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(80, 4))
    duration = np.exp(-0.3 * X[:, 0] + rng.normal(scale=0.5, size=80))
    event = np.asarray([0, 1] * 40)
    estimator = PenalizedCox(alpha=0.05, l1_weight=0.5).fit(X, duration, event)
    risk = estimator.predict_risk(X[:7])
    assert risk.shape == (7,)
    assert np.isfinite(risk).all()


def test_classification_outer_splits_are_frozen_and_reproducible() -> None:
    y = np.asarray([0, 1] * 25)
    first = classification_outer_splits(y, seed=17, repeats=2)
    second = classification_outer_splits(y, seed=17, repeats=2)
    assert len(first) == 10
    for (train_a, test_a), (train_b, test_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_survival_outer_splits_are_frozen_and_reproducible() -> None:
    event = np.asarray([0, 1] * 20)
    first = survival_outer_splits(event, seed=23)
    second = survival_outer_splits(event, seed=23)
    assert len(first) == 5
    for (train_a, test_a), (train_b, test_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_prognostic_marker_registry_excludes_redundant_apoe_dosage() -> None:
    frame = pd.DataFrame(
        {
            "Hippocampus_icv": [0.1],
            "apoe_e4_dosage": [1.0],
            "ad_prs_p1em03_avg": [0.2],
            "pathway_prs__curated__synaptic__p5em02": [0.3],
        }
    )
    markers = dict(prognostic_marker_columns(frame))
    assert markers["Hippocampus_icv"] == "structural_mri"
    assert markers["ad_prs_p1em03_avg"] == "polygenic_risk_score"
    assert (
        markers["pathway_prs__curated__synaptic__p5em02"]
        == "pathway_polygenic_risk_score"
    )
    assert "apoe_e4_dosage" not in markers


def test_prognostic_marker_association_recovers_stable_risk_direction() -> None:
    rng = np.random.default_rng(71)
    n = 180
    marker = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "AGE": rng.normal(72, 5, size=n),
            "sex_binary": rng.integers(0, 2, size=n),
            "PTEDUCAT": rng.normal(16, 2, size=n),
            "APOE4": rng.integers(0, 3, size=n),
            "marker": marker,
        }
    )
    event_time = np.exp(-0.9 * marker + rng.normal(scale=0.35, size=n))
    censor_time = rng.uniform(0.8, 4.0, size=n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    result = prognostic_marker_association(
        frame,
        duration,
        event,
        marker="marker",
        seed=19,
        bootstraps=99,
    )
    assert result["n"] == n
    assert result["log_hazard_ratio"] > 0
    assert result["p_value"] < 0.05
    assert result["bootstrap_log_hr_ci_low"] > 0
    assert result["bootstrap_sign_stability"] >= 0.95
