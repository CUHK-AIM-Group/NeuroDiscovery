from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.scripts.run_hcp_closed_loop_experiments import (
    BRAIN_AGE_FDR_P_COLUMN,
    _cache_path,
    _edge_projection,
    _healthy_subjects,
    age_stratified_splits,
    align_features,
    bootstrap_external_metrics,
    compact_checkpoint,
    extract_fc_views,
    load_feature_views,
    prediction_permutation_p,
    repeated_regression_cv,
)


def test_vectorized_external_bootstrap_matches_scalar_reference() -> None:
    y = np.asarray([18.0, 21.0, 29.0, 35.0, 42.0, 55.0])
    prediction = np.asarray([20.0, 24.0, 27.0, 37.0, 39.0, 51.0])
    samples = 37
    seed = 19
    baseline = 31.0
    result = bootstrap_external_metrics(
        y,
        prediction,
        frozen_baseline=baseline,
        seed=seed,
        samples=samples,
    )

    rng = np.random.default_rng(seed)
    correlations = []
    improvements = []
    baseline_maes = []
    for _ in range(samples):
        index = rng.integers(0, len(y), size=len(y))
        correlations.append(float(np.corrcoef(y[index], prediction[index])[0, 1]))
        baseline_maes.append(float(np.mean(np.abs(y[index] - baseline))))
        improvements.append(
            float(
                baseline_maes[-1]
                - np.mean(np.abs(y[index] - prediction[index]))
            )
        )
    finite = np.asarray(correlations)[np.isfinite(correlations)]
    np.testing.assert_allclose(result["r_ci_low"], np.quantile(finite, 0.025))
    np.testing.assert_allclose(result["r_ci_high"], np.quantile(finite, 0.975))
    np.testing.assert_allclose(
        result["mae_improvement_ci_low"], np.quantile(improvements, 0.025)
    )
    np.testing.assert_allclose(
        result["mae_improvement_ci_high"], np.quantile(improvements, 0.975)
    )
    normalized = np.asarray(improvements) / np.asarray(baseline_maes)
    np.testing.assert_allclose(
        result["normalized_mae_improvement_ci_low"], np.quantile(normalized, 0.025)
    )


def test_fc_views_are_deterministic_and_finite() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(scale=0.2, size=(12, 12))
    matrix = (matrix + matrix.T) / 2
    first = extract_fc_views(matrix, atlas="toy")
    second = extract_fc_views(matrix, atlas="toy")
    assert set(first) == {
        "fc_edge_projection",
        "node_connectivity_profile",
        "graph_topology_summary",
    }
    assert first["fc_edge_projection"].shape == (256,)
    assert first["node_connectivity_profile"].shape == (36,)
    assert first["graph_topology_summary"].shape == (49,)
    for name in first:
        np.testing.assert_allclose(first[name], second[name])
        assert np.isfinite(first[name]).all()


def test_signed_projection_preserves_shape() -> None:
    values = np.arange(100, dtype=float)
    projection = _edge_projection(values, seed=11, width=17)
    assert projection.shape == (17,)
    assert np.isfinite(projection).all()
    assert not np.allclose(projection, 0)


def test_adhd_cache_path_uses_zero_padding_when_needed(tmp_path) -> None:
    atlas_dir = tmp_path / "aal"
    atlas_dir.mkdir()
    expected = atlas_dir / "sub-adhd200_0010001.pt"
    expected.touch()
    assert _cache_path(tmp_path, "adhd200", "10001", "aal") == expected


def test_align_features_uses_metadata_order() -> None:
    subjects = ["b", "a", "c"]
    views = {
        "fc_edge_projection": np.asarray([[2.0], [1.0], [3.0]]),
    }
    metadata = pd.DataFrame(
        {
            "subject_id": ["a", "b", "missing"],
            "label": [10.0, 20.0, 30.0],
            "age": [1.0, 2.0, 3.0],
            "sex": [0.0, 1.0, 0.0],
        }
    )
    X, y, covariates, kept = align_features(
        subjects, views, metadata, "fc_edge_projection"
    )
    assert kept == ["b", "a"]
    np.testing.assert_array_equal(X[:, 0], [2.0, 1.0])
    np.testing.assert_array_equal(y, [20.0, 10.0])
    np.testing.assert_array_equal(covariates[:, 0], [2.0, 1.0])


def test_explicit_control_vocabularies(tmp_path) -> None:
    fixtures = {
        "ucla_preprocessed/metadata/ucla-rest.csv": pd.DataFrame(
            {"subject_id": ["u1", "u2"], "diagnosis": ["CONTROL", "SCHZ"]}
        ),
        "cobre_preprocessed/metadata/cobre-rest.csv": pd.DataFrame(
            {"subject_id": ["c1", "c2"], "dx": ["No_Known_Disorder", "Schizophrenia_Strict"]}
        ),
        "hcpep_preprocessed/metadata/hcpep-rest.csv": pd.DataFrame(
            {"subject_id": ["h1", "h2"], "phenotype_description": ["In good health", "Affective psychosis"]}
        ),
        "adhd200_preprocessed/metadata/adhd200-rest.csv": pd.DataFrame(
            {"subject_id": ["a1", "a2"], "DX": [0, 1]}
        ),
    }
    for relative, frame in fixtures.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    assert _healthy_subjects(tmp_path, "ucla") == {"u1"}
    assert _healthy_subjects(tmp_path, "cobre") == {"c1"}
    assert _healthy_subjects(tmp_path, "hcpep") == {"h1"}
    assert _healthy_subjects(tmp_path, "adhd200") == {"a1"}


def test_missing_external_atlas_is_a_specific_unavailable_condition(tmp_path) -> None:
    with np.testing.assert_raises_regex(RuntimeError, "No usable adhd200/cc400 FC caches"):
        load_feature_views(
            cache_root=tmp_path,
            work_dir=tmp_path / "work",
            dataset="adhd200",
            atlas="cc400",
            subject_ids=["missing"],
        )


def test_compact_checkpoint_keeps_latest_identity(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    records = [
        {"identity": {"atlas": "a", "model": "m"}, "value": 1},
        {"identity": {"atlas": "b", "model": "m"}, "value": 2},
        {"identity": {"atlas": "a", "model": "m"}, "value": 3},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    audit = compact_checkpoint(path)
    compacted = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert audit == {"rows_before": 3, "rows_after": 2, "duplicates_removed": 1}
    assert {row["identity"]["atlas"]: row["value"] for row in compacted} == {"a": 3, "b": 2}


def test_age_stratified_splits_preserve_age_support() -> None:
    age = np.repeat(np.arange(40, 80), 3).astype(float)
    splits = age_stratified_splits(age, n_splits=5, repeats=2, seed=11)
    assert len(splits) == 10
    for train, test in splits:
        assert len(set(train) & set(test)) == 0
        assert len(test) == 24
        assert age[test].min() <= 45
        assert age[test].max() >= 74


def test_brain_age_cv_returns_unique_oof_predictions_and_normalized_gain() -> None:
    rng = np.random.default_rng(23)
    age = np.linspace(36, 90, 100)
    X = np.column_stack(
        [age + rng.normal(scale=2.0, size=len(age)), rng.normal(size=len(age))]
    )
    result = repeated_regression_cv(
        X,
        age,
        model="ridge",
        seed=29,
        repeats=2,
        stratify_continuous=True,
    )
    assert result["n_folds"] == 10
    assert result["oof_true"].shape == (100,)
    assert result["oof_prediction"].shape == (100,)
    assert np.isfinite(result["oof_prediction"]).all()
    assert result["normalized_mae_improvement"] > 0.5


def test_brain_age_fdr_uses_exact_permutation_probability() -> None:
    result = prediction_permutation_p(
        np.arange(20, dtype=float),
        np.arange(20, dtype=float),
        permutations=9,
        seed=31,
    )
    assert BRAIN_AGE_FDR_P_COLUMN == "permutation_p_exact"
    assert result["permutation_p_exact"] == 0.1
    assert result["permutation_p_tail"] < result["permutation_p_exact"]
