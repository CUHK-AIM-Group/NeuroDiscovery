from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from core.scripts.case1_multimodel_pilot import (
    DEFAULT_MODELS,
    FoldData,
    choose_stratification,
    make_folds,
    run_model_fold,
    summarize_attributions,
    summarize_performance,
)


@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
def test_all_case1_models_train_and_attribute(model_name: str) -> None:
    rng = np.random.default_rng(7)
    n_subjects = 24
    n_roi = 12
    labels = np.tile([0, 1], n_subjects // 2).astype(np.int64)
    nodes = rng.normal(size=(n_subjects, n_roi, 15)).astype(np.float32)
    matrices = rng.normal(size=(n_subjects, n_roi, n_roi)).astype(np.float32)
    matrices = (matrices + matrices.transpose(0, 2, 1)) / 2.0
    for matrix in matrices:
        np.fill_diagonal(matrix, 0.0)
    roi_meta = pd.DataFrame(
        {
            "roi_id": np.arange(1, n_roi + 1),
            "parcel_name": [f"7Networks_LH_Vis_{index + 1}" for index in range(n_roi)],
            "hemisphere": "L",
            "network": "Vis",
        }
    )
    fold = FoldData(
        train=np.arange(0, 16),
        val=np.arange(16, 20),
        test=np.arange(20, 24),
    )

    fitted, metrics, epochs_run, evidence = run_model_fold(
        model_name=model_name,
        atlas="schaefer_100_7net",
        disease="synthetic",
        seed=11,
        fold_index=0,
        fold=fold,
        y=labels,
        subjects=[f"sub-{index:03d}" for index in range(n_subjects)],
        adjusted_nodes=nodes,
        adjusted_corr=matrices,
        roi_meta=roi_meta,
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=1,
        graph_density=0.25,
        skip_attribution=False,
    )

    assert fitted is not None
    assert epochs_run >= 1
    assert set(metrics) == {"auroc", "auprc", "balanced_accuracy"}
    assert all(np.isfinite(value) for value in metrics.values())
    expected = n_roi * 15 if model_name in {"elasticnet", "roi_mlp"} else n_roi
    assert len(evidence) == expected


def test_site_stratification_is_used_only_when_every_cell_supports_folds() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1] * 2)
    sites = np.array(["A"] * 6 + ["B"] * 6)

    strata, policy = choose_stratification(labels, sites, n_splits=3)

    assert policy == "diagnosis_x_site"
    folds = make_folds(labels, n_splits=3, seed=7, strata=strata)
    assert len(folds) == 3
    assert all(set(labels[fold.test]) == {0, 1} for fold in folds)


def test_site_stratification_falls_back_when_a_cell_is_too_small() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sites = np.array(["A", "A", "A", "B", "A", "A", "A", "A"])

    strata, policy = choose_stratification(labels, sites, n_splits=3)

    assert policy == "diagnosis"
    assert np.array_equal(strata, labels.astype(str))


def test_performance_variance_uses_seed_means_not_individual_folds() -> None:
    performance = pd.DataFrame(
        [
            {
                "atlas": "atlas",
                "disease": "disease",
                "model": "model",
                "seed": seed,
                "fold": fold,
                "auroc": value,
                "auprc": value,
                "balanced_accuracy": value,
            }
            for seed, values in ((1, (0.0, 1.0, 0.5)), (2, (0.4, 0.6, 0.5)))
            for fold, value in enumerate(values)
        ]
    )

    seed_summary, summary, _, _ = summarize_performance(performance)

    assert seed_summary["auroc"].tolist() == pytest.approx([0.5, 0.5])
    assert summary.loc[0, "auroc_variance"] == pytest.approx(0.0)
    assert summary.loc[0, "n_seeds"] == 2


def test_attribution_variance_uses_seed_means_not_individual_folds() -> None:
    evidence = pd.DataFrame(
        [
            {
                "model": "model",
                "atlas": "atlas",
                "disease": "disease",
                "evidence_resolution": "roi",
                "roi_index": 0,
                "roi_id": 1,
                "roi_name": "ROI 1",
                "hemisphere": "L",
                "network": "Default",
                "feature": "correlation_profile",
                "seed": seed,
                "fold": fold,
                "importance_abs": value,
                "signed_attribution": -value,
            }
            for seed, values in ((1, (0.0, 1.0, 0.5)), (2, (0.4, 0.6, 0.5)))
            for fold, value in enumerate(values)
        ]
    )

    seed_summary, summary = summarize_attributions(evidence)

    assert seed_summary["importance_abs"].tolist() == pytest.approx([0.5, 0.5])
    assert seed_summary["n_folds"].tolist() == [3, 3]
    assert summary.loc[0, "importance_abs_variance"] == pytest.approx(0.0)
    assert summary.loc[0, "signed_attribution_variance"] == pytest.approx(0.0)
    assert summary.loc[0, "n_seeds"] == 2

