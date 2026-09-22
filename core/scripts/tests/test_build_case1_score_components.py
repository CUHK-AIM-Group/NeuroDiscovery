from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.scripts.build_case1_score_components import (
    build_novelty_scores,
    load_public_candidate_registry,
    percentile_score,
    validate_critic_scores,
)
from core.scripts.case1_method_comparison import (
    KgIndex,
    add_generator_scores,
    frozen_auxiliary_exploration_score,
    load_results,
)
from core.scripts.case1_neurodiscovery_config import Case1NeuroDiscoveryConfig


def test_public_registry_does_not_import_hidden_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "modality": ["fmri"],
            "source": ["atlas"],
            "disease": ["ADHD"],
            "feature": ["corr_mean"],
            "roi_index": [1],
            "roi_id": ["r1"],
            "roi_name": ["Default parcel"],
            "anatomy_key": ["default"],
            "anatomy_full": ["Default network"],
            "hemisphere": ["bilateral"],
            "network": ["Default"],
            "structure_class": ["cortex"],
            "n_case": [20],
            "n_control": [20],
            "atlas_label_source": ["registered"],
            "atlas_label_weight": [1.0],
            "adjusted_residual_d": [99.0],
            "p_value": [1e-30],
            "is_gt_top": [True],
        }
    ).to_csv(path, index=False)

    public = load_public_candidate_registry(path)

    assert "adjusted_residual_d" not in public
    assert "p_value" not in public
    assert "is_gt_top" not in public
    assert public.loc[0, "candidate_id"] == "fmri|atlas|ADHD|corr_mean|1"

    empty_kg = KgIndex({}, {}, {}, {}, {})
    scored = add_generator_scores(public, empty_kg, seed=7)
    assert "score_neurodiscovery" in scored
    assert "score_exhaustive_gt" not in scored


def test_percentile_score_is_bounded_and_tie_stable() -> None:
    score = percentile_score(np.asarray([1.0, 1.0, 2.0, 3.0]))
    assert score.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert percentile_score(np.ones(3)).tolist() == [0.5, 0.5, 0.5]


def test_experiment_registry_retains_failed_executions(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    base = {
        "modality": "fmri",
        "source": "atlas",
        "disease": "ADHD",
        "feature": "corr_mean",
        "roi_id": "r1",
        "roi_name": "Default parcel",
        "anatomy_key": "default",
        "anatomy_full": "Default network",
        "hemisphere": "bilateral",
        "network": "Default",
        "structure_class": "cortex",
        "n_case": 20,
        "n_control": 20,
        "q_fdr_disease": 1.0,
        "q_fdr_modality": 1.0,
        "direction": "",
        "atlas_label_source": "registered",
        "atlas_label_weight": 1.0,
    }
    pd.DataFrame(
        [
            {
                **base,
                "roi_index": 1,
                "adjusted_residual_d": 0.5,
                "abs_adjusted_residual_d": 0.5,
                "p_value": 0.001,
                "q_fdr_global": 0.01,
            },
            {
                **base,
                "roi_index": 2,
                "adjusted_residual_d": np.nan,
                "abs_adjusted_residual_d": np.nan,
                "p_value": np.nan,
                "q_fdr_global": np.nan,
            },
        ]
    ).to_csv(path, index=False)

    frame = load_results(path, gt_top_frac=0.01)

    assert len(frame) == 2
    assert frame["execution_succeeded"].tolist() == [True, False]
    assert frame["is_gt_top"].tolist() == [True, False]


def test_novelty_rewards_grounded_under_attested_candidates() -> None:
    frame = pd.DataFrame(
        {
            "kg_disease_degree": [100, 100, 0],
            "kg_region_degree": [100, 100, 0],
            "kg_feature_degree": [100, 100, 0],
            "kg_scoped_disease_degree": [50, 50, 0],
            "kg_scoped_region_degree": [50, 50, 0],
            "kg_scoped_feature_degree": [50, 50, 0],
            "kg_pair_support": [0.0, 10.0, 0.0],
            "kg_disease_feature_support": [0.0, 10.0, 0.0],
            "kg_region_feature_support": [0.0, 10.0, 0.0],
            "kg_scoped_pair_support": [0.0, 10.0, 0.0],
            "kg_scoped_disease_feature_support": [0.0, 10.0, 0.0],
            "kg_scoped_region_feature_support": [0.0, 10.0, 0.0],
        }
    )

    score, audit = build_novelty_scores(frame)

    assert score[0] > score[1]
    assert score[0] > score[2]
    assert audit["fully_resolved_fraction"] == pytest.approx(2 / 3)


def test_critic_validation_recomputes_frozen_weighted_score() -> None:
    rows = validate_critic_scores(
        {
            "scores": [
                {
                    "id": "A0001",
                    "scientific_plausibility": 80,
                    "experimental_testability": 60,
                    "anatomical_specificity": 70,
                    "confound_resistance": 50,
                }
            ]
        },
        {"A0001"},
    )
    assert rows[0]["score_critic"] == pytest.approx(0.66)


def test_frozen_components_define_only_an_auxiliary_exploration_score() -> None:
    frame = pd.DataFrame(
        {
            "score_kge": [0.0, 1.0],
            "score_novelty": [0.0, 1.0],
            "score_critic": [0.0, 1.0],
        }
    )
    auxiliary = frozen_auxiliary_exploration_score(
        frame,
        Case1NeuroDiscoveryConfig(
            kge_weight=1.0,
            novelty_weight=1.0,
            critic_weight=1.0,
        ),
    )

    assert auxiliary is not None
    assert auxiliary.tolist() == pytest.approx([0.0, 1.0])
