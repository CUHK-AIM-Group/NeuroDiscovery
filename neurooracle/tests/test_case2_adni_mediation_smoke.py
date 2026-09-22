from __future__ import annotations

import numpy as np
import pandas as pd

from neurooracle.scripts.run_case2_adni_mediation_smoke import (
    build_covariate_matrix,
    complete_case_mask,
    load_pathway_exposures,
    mediation_screen,
)


def test_mediation_screen_prioritizes_true_indirect_path() -> None:
    rng = np.random.default_rng(29)
    n = 800
    covariate = rng.normal(size=n)
    exposure = 0.5 * covariate + rng.normal(size=n)
    true_mediator = (
        0.85 * exposure
        + 0.30 * covariate
        + rng.normal(scale=0.70, size=n)
    )
    null_mediator = rng.normal(size=n)
    outcome = (
        0.90 * true_mediator
        + 0.10 * exposure
        + 0.25 * covariate
        + rng.normal(scale=0.70, size=n)
    )
    mediators = np.column_stack([true_mediator, null_mediator])
    covariates = np.column_stack([np.ones(n), covariate])

    results, marker_mask = mediation_screen(
        exposure,
        outcome,
        mediators,
        covariates,
    )

    assert marker_mask.tolist() == [True, True]
    assert results.loc[0, "indirect_effect_std"] > 0.30
    assert results.loc[0, "sobel_q_family"] < 0.001
    assert (
        abs(results.loc[0, "indirect_effect_std"])
        > abs(results.loc[1, "indirect_effect_std"])
    )


def test_mediation_screen_drops_nonfinite_and_constant_markers() -> None:
    rng = np.random.default_rng(7)
    n = 120
    exposure = rng.normal(size=n)
    outcome = 0.4 * exposure + rng.normal(size=n)
    mediators = np.column_stack(
        [
            rng.normal(size=n),
            np.ones(n),
            rng.normal(size=n),
        ]
    )
    mediators[5, 2] = np.nan

    results, marker_mask = mediation_screen(
        exposure,
        outcome,
        mediators,
        np.ones((n, 1)),
    )

    assert marker_mask.tolist() == [True, False, False]
    assert len(results) == 1


def test_complete_cases_and_covariate_encoding() -> None:
    subjects = pd.DataFrame(
        {
            "apoe_e4_dosage": [0.0, 1.0, 2.0],
            "apoe_complete": [True, False, True],
            "diagnosis_code": [0.0, 1.0, 2.0],
            "AGE": [70.0, 71.0, 72.0],
            "sex_binary": [0.0, 1.0, 0.0],
            "PTEDUCAT": [16.0, 14.0, 18.0],
            "imaging_mean_fd": [0.2, 0.3, 0.8],
            "SITE": ["A", "B", "A"],
            "COLPROT": ["ADNI2", "ADNI2", "ADNIGO"],
            "batch": ["gwas1", "gwas2", "gwas1"],
            **{
                f"PC{index}": [0.1 * index, 0.2 * index, 0.3 * index]
                for index in range(1, 11)
            },
        }
    )

    mask = complete_case_mask(
        subjects,
        "apoe_e4_dosage",
        "diagnosis_code",
        fd_max=0.5,
    )
    assert mask.tolist() == [True, False, False]

    covariates, names = build_covariate_matrix(subjects)
    assert covariates.shape[0] == 3
    assert names[0] == "intercept"
    assert np.linalg.matrix_rank(covariates) == covariates.shape[1]


def test_load_pathway_exposures_with_manifest(tmp_path) -> None:
    pathway_path = tmp_path / "case2_adni_pathway_prs_wide.parquet"
    pd.DataFrame(
        {
            "subject_id": ["001_S_0001", "001_S_0002"],
            "pathway_prs__curated__synaptic__p5em02": [0.1, 0.2],
        }
    ).to_parquet(pathway_path, index=False)
    pd.DataFrame(
        {
            "score_name": [
                "pathway_prs__curated__synaptic__p5em02"
            ],
            "pathway_name": ["Synaptic"],
            "threshold_label": ["p5em02"],
        }
    ).to_csv(tmp_path / "score_manifest.csv", index=False)

    pathway, metadata = load_pathway_exposures(pathway_path)

    assert pathway.shape == (2, 2)
    assert (
        metadata["pathway_prs__curated__synaptic__p5em02"][
            "pathway_name"
        ]
        == "Synaptic"
    )

# Last Updated At: 2026-07-31 02:55 HKT
