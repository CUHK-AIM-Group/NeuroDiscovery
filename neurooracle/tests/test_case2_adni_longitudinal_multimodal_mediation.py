from __future__ import annotations

import pandas as pd

from neurooracle.scripts.run_case2_adni_longitudinal_multimodal_mediation import (
    build_index_analysis_rows,
    select_curated_pathway_exposures,
)


def test_select_curated_pathway_exposures_prefers_stricter_threshold() -> None:
    subjects = pd.DataFrame(
        {
            "subject_id": ["S1"],
            "pathway_prs__curated__immune__p1em03": [0.1],
            "pathway_prs__curated__immune__p5em02": [0.2],
            "pathway_prs__curated__myelin__p5em02": [0.3],
        }
    )
    manifest = pd.DataFrame(
        {
            "score_name": [
                "pathway_prs__curated__immune__p5em02",
                "pathway_prs__curated__immune__p1em03",
                "pathway_prs__curated__myelin__p5em02",
            ],
            "pathway_id": ["immune", "immune", "myelin"],
            "pathway_source": ["NeuroOracle curated"] * 3,
            "pathway_name": ["Immune", "Immune", "Myelin"],
            "threshold_label": ["p5em02", "p1em03", "p5em02"],
        }
    )

    selected = select_curated_pathway_exposures(subjects, manifest)

    assert selected["score_name"].tolist() == [
        "pathway_prs__curated__immune__p1em03",
        "pathway_prs__curated__myelin__p5em02",
    ]


def test_build_index_rows_uses_earliest_imaging_and_nearest_target_followup() -> None:
    markers = pd.DataFrame(
        {
            "visit_id": ["v1", "v1", "v2", "v2"],
            "subject_id": ["S1"] * 4,
            "modality": ["smri_adnimerge"] * 4,
            "imaging_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2021-01-01", "2021-01-01"]
            ),
            "qc_primary_eligible": [True] * 4,
            "qc_whole_brain_eligible": [True] * 4,
            "qc_relaxed_eligible": [True] * 4,
            "marker": ["Hippocampus", "ICV", "Hippocampus", "ICV"],
            "value": [100.0, 1400.0, 90.0, 1405.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "visit_id": ["v1", "v1", "v2"],
            "subject_id": ["S1"] * 3,
            "modality": ["smri_adnimerge"] * 3,
            "imaging_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2021-01-01"]
            ),
            "outcome": ["MMSE"] * 3,
            "baseline_outcome_date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2021-01-02"]
            ),
            "baseline_value": [29.0, 29.0, 27.0],
            "future_outcome_date": pd.to_datetime(
                ["2021-01-01", "2022-01-02", "2023-01-01"]
            ),
            "future_value": [28.0, 26.0, 25.0],
            "followup_days": [366, 732, 730],
            "followup_years": [1.0, 2.0, 2.0],
            "delta_value": [-1.0, -3.0, -2.0],
            "decline_score": [1.0, 3.0, 2.0],
            "annualized_decline_score": [1.0, 1.5, 1.0],
            "worse_direction": [-1.0] * 3,
        }
    )
    subjects = pd.DataFrame(
        {
            "subject_id": ["S1"],
            "AGE": [70.0],
            "sex_binary": [1],
            "PTEDUCAT": [16],
            "SITE": [1],
            "COLPROT": ["ADNI1"],
            "batch": ["batch1"],
            **{f"PC{i}": [0.0] for i in range(1, 11)},
            "pathway_prs__curated__immune__p1em03": [0.2],
        }
    )
    clinical = pd.DataFrame(
        {"subject_id": ["S1"], "clinical_date": pd.to_datetime(["2020-01-01"])}
    )

    result = build_index_analysis_rows(
        markers,
        pairs,
        subjects,
        clinical,
        marker_specs=["smri_adnimerge::Hippocampus"],
        outcomes=["MMSE"],
    )

    assert len(result) == 1
    assert result.loc[0, "visit_id"] == "v1"
    assert result.loc[0, "followup_days"] == 732
    assert result.loc[0, "icv"] == 1400.0


# Last Updated At: 2026-07-31 19:54 HKT
