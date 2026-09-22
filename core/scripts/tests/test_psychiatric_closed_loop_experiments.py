from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.scripts.run_psychiatric_closed_loop_experiments import (
    CovariateResidualizedPCA,
    SUBTYPING_CLINICAL_DIMENSIONS,
    align_external_features,
    compact_checkpoint,
    differential_labels,
    family_fdr,
    load_ucla_subtyping_metadata,
    multivariate_clinical_separation,
)


def test_cobre_diagnostic_mapping_accepts_underscore_labels() -> None:
    metadata = pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3", "s4"],
            "phenotype": [
                "Schizophrenia_Strict",
                "Schizoaffective",
                "Bipolar_Disorder",
                "No_Known_Disorder",
            ],
            "age": [20, 21, 22, 23],
            "sex": [0, 1, 0, 1],
            "site": ["COBRE"] * 4,
        }
    )
    mapped = differential_labels(metadata, "COBRE")
    assert mapped["subject_id"].tolist() == ["s1", "s2", "s3"]
    assert mapped["label"].tolist() == [1, 1, 0]


def test_external_alignment_matches_reference_location_and_scale() -> None:
    reference = np.asarray([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]])
    external = np.asarray([[100.0, -5.0], [110.0, 0.0], [120.0, 5.0]])
    aligned = align_external_features(reference, external)
    np.testing.assert_allclose(np.median(aligned, axis=0), np.median(reference, axis=0))
    np.testing.assert_allclose(np.std(aligned, axis=0), np.std(reference, axis=0))


def test_family_fdr_is_applied_within_registered_families() -> None:
    frame = pd.DataFrame(
        {
            "feature_family": ["a", "a", "b", "b"],
            "model": ["m", "m", "m", "m"],
            "p": [0.01, 0.04, 0.01, 0.9],
        }
    )
    q = family_fdr(
        frame,
        p_column="p",
        family_fields=("feature_family", "model"),
    )
    np.testing.assert_allclose(q, [0.02, 0.04, 0.02, 0.9])


def test_checkpoint_compaction_is_last_write_wins(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    rows = [
        {"identity": {"atlas": "a"}, "internal": {"score": 1}},
        {"identity": {"atlas": "b"}, "internal": {"score": 2}},
        {"identity": {"atlas": "a"}, "internal": {"score": 3}},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    audit = compact_checkpoint(path)
    compacted = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert audit == {"rows_before": 3, "rows_after": 2, "duplicates_removed": 1}
    assert [row["identity"]["atlas"] for row in compacted] == ["a", "b"]
    assert compacted[0]["internal"]["score"] == 3


def test_ucla_subtyping_metadata_uses_continuous_symptoms_not_diagnosis(tmp_path) -> None:
    metadata = pd.DataFrame(
        {
            "subject_id": ["sub-a", "sub-b", "sub-c", "sub-d"],
            "phenotype": ["CONTROL", "ADHD", "BIPOLAR", "SCHZ"],
            "age": [20, 21, 22, 23],
            "sex": [0, 1, 0, 1],
            "site": ["UCLA"] * 4,
        }
    )
    tables = {
        "bprs": pd.DataFrame(
            {
                "participant_id": metadata["subject_id"],
                "bprs_positive": [1, 2, 3, 4],
                "bprs_negative": [1, 3, 2, 4],
                "bprs_mania": [1, 2, 4, 3],
                "bprs_depanx": [1, 4, 2, 3],
            }
        ),
        "hamilton": pd.DataFrame(
            {"participant_id": metadata["subject_id"], "hamd_28": [1, 2, 3, 4]}
        ),
        "ymrs": pd.DataFrame(
            {"participant_id": metadata["subject_id"], "ymrs_score": [1, 3, 4, 2]}
        ),
        "asrs": pd.DataFrame(
            {"participant_id": metadata["subject_id"], "asrs_score": [1, 4, 3, 2]}
        ),
    }
    for stem, frame in tables.items():
        frame.to_csv(tmp_path / f"{stem}.tsv", sep="\t", index=False)

    result = load_ucla_subtyping_metadata(metadata, tmp_path)
    assert result["subject_id"].tolist() == ["sub-b", "sub-c", "sub-d"]
    assert result["diagnosis"].tolist() == ["adhd", "bipolar", "schz"]
    assert result[list(SUBTYPING_CLINICAL_DIMENSIONS)].notna().all().all()
    assert "label" not in result


def test_multivariate_clinical_separation_detects_withheld_signal() -> None:
    rng = np.random.default_rng(13)
    labels = np.repeat([0, 1], 30)
    symptoms = rng.normal(scale=0.25, size=(60, 5))
    symptoms[labels == 1, :3] += 1.5
    result = multivariate_clinical_separation(
        labels,
        symptoms,
        permutations=199,
        seed=17,
    )
    assert result["p_value"] <= 0.01
    assert result["multivariate_r2"] > 0.5
    assert result["minimum_cluster"] == 30
    assert result["strongest_dimension"] in SUBTYPING_CLINICAL_DIMENSIONS[:3]


def test_robust_residualized_pca_limits_extreme_feature_influence() -> None:
    rng = np.random.default_rng(31)
    imaging = rng.normal(size=(40, 8))
    imaging[0, 0] = 1e6
    covariates = rng.normal(size=(40, 2))
    model = CovariateResidualizedPCA(max_components=5, robust=True).fit(
        np.column_stack([imaging, covariates])
    )
    transformed = model.transform(np.column_stack([imaging, covariates]))
    assert transformed.shape == (40, 5)
    assert np.isfinite(transformed).all()
    assert np.max(np.abs(model.scaler_.transform(imaging - np.column_stack(
        [np.ones(40), covariates]
    ) @ model.beta_))) > 5
