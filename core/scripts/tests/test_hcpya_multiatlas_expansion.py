from __future__ import annotations

import numpy as np

from core.scripts.run_hcpya_multiatlas_expansion import (
    compact_fc_features,
    run_cpm_cv,
    run_regression_cv,
)


def synthetic_connectomes(seed: int = 17):
    rng = np.random.default_rng(seed)
    n_subjects, n_rois = 36, 9
    matrices = np.empty((n_subjects, n_rois, n_rois), dtype=np.float32)
    target = rng.normal(size=n_subjects)
    for index in range(n_subjects):
        matrix = rng.normal(scale=0.2, size=(n_rois, n_rois))
        matrix = (matrix + matrix.T) / 2
        matrix[0, 1] = matrix[1, 0] = target[index] + rng.normal(scale=0.1)
        np.fill_diagonal(matrix, 1.0)
        matrices[index] = matrix
    return matrices, target


def test_compact_fc_features_excludes_diagonal():
    matrix = np.eye(5, dtype=np.float32)
    features = compact_fc_features(matrix)
    assert features.shape == (24,)
    assert np.allclose(features, 0.0)


def test_hcpya_regression_and_cpm_cv():
    matrices, target = synthetic_connectomes()
    compact = np.stack([compact_fc_features(matrix) for matrix in matrices])
    upper = np.triu_indices(matrices.shape[1], 1)
    edges = matrices[:, upper[0], upper[1]]
    ids = np.asarray([f"s{index}" for index in range(len(target))])

    regression, regression_predictions, _ = run_regression_cv(
        compact,
        target,
        ids,
        model_name="ridge",
        seed=4,
        folds=3,
        brain_age=False,
    )
    assert regression["n_samples"] == len(target)
    assert len(regression_predictions) == len(target)
    assert regression["aggregate"]["raw_pearson_r"] > 0.5

    cpm, cpm_predictions, _ = run_cpm_cv(
        edges,
        target,
        ids,
        p_threshold=0.05,
        seed=4,
        folds=3,
    )
    assert cpm["n_features"] == edges.shape[1]
    assert len(cpm_predictions) == len(target)
    assert cpm["aggregate"]["raw_pearson_r"] > 0.5


def test_hcpya_brain_age_outputs_corrected_predictions():
    matrices, target = synthetic_connectomes(seed=29)
    age = 25 + 3 * target
    compact = np.stack([compact_fc_features(matrix) for matrix in matrices])
    ids = np.asarray([f"s{index}" for index in range(len(age))])
    metrics, predictions, _ = run_regression_cv(
        compact,
        age,
        ids,
        model_name="ridge",
        seed=5,
        folds=3,
        brain_age=True,
    )
    assert "bias_corrected_mae" in metrics["aggregate"]
    assert "prediction_bias_corrected" in predictions
    assert "brain_pad" in predictions
