from __future__ import annotations

import numpy as np
import pandas as pd

from core.scripts.prepare_biomarker_discovery_tables_v2 import (
    apply_validation_contract,
    feature_family,
    repeated_directional_validation,
)


def test_feature_families_are_registered() -> None:
    assert feature_family("roi_alff_proxy") == "amplitude"
    assert feature_family("roi_temporal_std") == "temporal"
    assert feature_family("corr_mean_abs") == "correlation_fc"
    assert feature_family("partial_mean") == "partial_fc"
    assert feature_family("normalized_volume_fraction") == "structure"


def test_direction_is_selected_on_training_fold() -> None:
    rng = np.random.default_rng(17)
    n_control = 60
    n_case = 30
    labels = np.r_[np.zeros(n_control, dtype=int), np.ones(n_case, dtype=int)]
    matrix = rng.normal(size=(len(labels), 3))
    matrix[:, 0] += 2.0 * labels
    matrix[:, 1] -= 1.8 * labels
    meta = pd.DataFrame(
        {
            "diagnosis_broad_any": [""] * n_control
            + ["synthetic_disease"] * n_case,
            "is_control": labels == 0,
            "Site": np.resize(["site_a", "site_b"], len(labels)),
        }
    )
    covariates = pd.DataFrame(
        {
            "age_z": rng.normal(size=len(labels)),
            "sex_male": rng.integers(0, 2, size=len(labels)),
            "site_site_b": (meta["Site"] == "site_b").astype(float),
        }
    )
    result = repeated_directional_validation(
        matrix=matrix,
        meta=meta,
        covariates=covariates,
        disease="synthetic_disease",
        seeds=(1, 2),
        n_splits=3,
    )
    auc = np.asarray(result["cv_auc_mean"])
    stability = np.asarray(result["cv_direction_stability"])
    assert int(result["cv_splits"]) == 6
    assert auc[0] > 0.80
    assert auc[1] > 0.75
    assert stability[0] > 0.80
    assert stability[1] > 0.80


def test_validation_contract_has_no_fixed_hit_quota() -> None:
    frame = pd.DataFrame(
        {
            "disease": ["d"] * 4,
            "atlas": ["a"] * 4,
            "feature_family": ["f"] * 4,
            "p_value": [1e-8, 1e-7, 0.4, 0.8],
            "adjusted_residual_d": [0.8, -0.7, 0.4, 0.1],
            "cv_auc_mean": [0.80, 0.75, 0.70, 0.52],
            "cv_direction_stability": [1.0, 0.9, 0.9, 0.6],
        }
    )
    result = apply_validation_contract(
        frame,
        family_q=0.05,
        effect_threshold=0.15,
        auc_threshold=0.55,
        stability_threshold=0.70,
    )
    assert result["validated"].tolist() == [True, True, False, False]
    assert result["feedback_status"].tolist() == [
        "supported",
        "supported",
        "inconclusive",
        "inconclusive",
    ]
