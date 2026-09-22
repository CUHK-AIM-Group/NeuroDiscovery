from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd

from neurooracle.scripts.run_case2_adni_targeted_fc_mediation import (
    TARGET_LABELS,
    build_exact_kg_ranking,
    build_target_atlas,
    compute_target_fc_features,
    translate_nas_path,
)


def test_translate_nas_path() -> None:
    translated = translate_nas_path(
        r"Z:\Dataset\fMRI\subject.nii.gz"
    )
    assert str(translated).startswith(r"\\192.168.3.61\data\Dataset")


def test_build_target_atlas_preserves_six_requested_labels(tmp_path) -> None:
    source_data = np.zeros((4, 4, 4), dtype=np.int16)
    for index, source_label in enumerate(TARGET_LABELS.values()):
        source_data.flat[index] = source_label
    source = tmp_path / "source.nii.gz"
    output = tmp_path / "target.nii.gz"
    nib.save(
        nib.Nifti1Image(source_data, np.eye(4)),
        source,
    )

    counts = build_target_atlas(source, output)
    target = np.asanyarray(nib.load(output).dataobj)

    assert counts == {name: 1 for name in TARGET_LABELS}
    assert set(np.unique(target)) == set(range(7))


def test_compute_target_fc_features_matches_manual_pair_means() -> None:
    rng = np.random.default_rng(23)
    timeseries = rng.normal(size=(500, 6))
    timeseries[:, 1] += 0.7 * timeseries[:, 0]
    timeseries[:, 3] += 0.6 * timeseries[:, 2]
    timeseries[:, 4] += 0.4 * timeseries[:, 1]
    timeseries[:, 5] += 0.4 * timeseries[:, 3]

    features = compute_target_fc_features(timeseries)
    correlation = np.corrcoef(timeseries, rowvar=False)
    fisher_z = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
    manual_amygdala_hippocampus = np.mean(
        fisher_z[np.ix_([1, 3], [0, 2])]
    )

    assert set(features) == {
        "triad_mean_fc_fisher_z",
        "amygdala_hippocampus_fc_fisher_z",
        "amygdala_acc_fc_fisher_z",
        "hippocampus_acc_fc_fisher_z",
    }
    assert np.isclose(
        features["amygdala_hippocampus_fc_fisher_z"],
        manual_amygdala_hippocampus,
    )
    assert all(np.isfinite(value) for value in features.values())


def test_exact_ranking_places_triad_first_for_each_exposure() -> None:
    kg_mapping = pd.DataFrame(
        {
            "exposure": ["p1", "p2"],
            "outcome": ["CDRSB", "CDRSB"],
            "mapping_score": [0.8, 0.7],
            "kg_rank": [1, 2],
            "primary_hypothesis_id": ["h1", "h2"],
            "pathway_id": ["pw1", "pw2"],
            "pathway_name": ["Pathway 1", "Pathway 2"],
            "supporting_hypothesis_ids": ["h1", "h2"],
            "supporting_genes": ["G1", "G2"],
        }
    )
    rows = []
    for exposure in ("p1", "p2"):
        for feature in (
            "triad_mean_fc_fisher_z",
            "amygdala_hippocampus_fc_fisher_z",
        ):
            rows.append(
                {
                    "exposure": exposure,
                    "outcome": "CDRSB",
                    "feature": feature,
                    "marker_id": feature,
                }
            )

    ranked = build_exact_kg_ranking(
        kg_mapping,
        pd.DataFrame(rows),
    )

    assert ranked["exposure"].tolist() == ["p1", "p2", "p1", "p2"]
    assert ranked["feature"].tolist()[:2] == [
        "triad_mean_fc_fisher_z",
        "triad_mean_fc_fisher_z",
    ]
    assert ranked["kg_rank"].tolist() == [1, 2, 3, 4]

# Last Updated At: 2026-07-31 03:31 HKT
