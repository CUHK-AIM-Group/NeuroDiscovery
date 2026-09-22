from __future__ import annotations

import numpy as np
import pandas as pd

from neurooracle.scripts.map_case2_kg_hypotheses_to_adni import (
    _diversified_rank,
    evaluate_prioritisation,
    map_imaging_markers,
    map_outcome,
    select_gene_pathways,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score_name": [
                "pathway_prs__synaptic__p5em02",
                "pathway_prs__synaptic__p1em03",
                "pathway_prs__immune__p5em02",
            ],
            "pathway_id": ["synaptic", "synaptic", "immune"],
            "pathway_source": ["curated", "curated", "curated"],
            "pathway_name": ["Synaptic", "Synaptic", "Immune"],
            "reference_id": [np.nan, np.nan, np.nan],
            "threshold_label": ["p5em02", "p1em03", "p5em02"],
            "gene_count": [4, 4, 25],
            "variants_in_score_file": [20, 5, 30],
            "score_file": ["a", "b", "c"],
        }
    )


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pathway_id": ["synaptic", "immune"],
            "pathway_source": ["curated", "curated"],
            "pathway_name": ["Synaptic", "Immune"],
            "reference_id": [np.nan, np.nan],
            "gene_count": [4, 25],
            "seed_overlap_count": [2, 1],
            "seed_overlap": ["GABRA1;APOE", "APOE"],
            "enrichment_p": [np.nan, np.nan],
            "enrichment_q": [np.nan, np.nan],
            "genes": ["GABRA1;APOE;SYN1;SYN2", "APOE;TREM2"],
        }
    )


def _marker_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker_id": ["acc_corr", "acc_degree", "visual_corr", "acc_alff"],
            "source": ["test"] * 4,
            "atlas": ["schaefer_400_7net"] * 4,
            "roi_index": [1, 1, 2, 1],
            "roi_id": ["1", "1", "2", "1"],
            "roi_name": ["acc", "acc", "visual", "acc"],
            "parcel_name": [
                "7Networks_LH_SalVentAttn_Med_1",
                "7Networks_LH_SalVentAttn_Med_1",
                "7Networks_LH_Vis_1",
                "7Networks_LH_SalVentAttn_Med_1",
            ],
            "hemisphere": ["LH"] * 4,
            "network": ["SalVentAttn", "SalVentAttn", "Vis", "SalVentAttn"],
            "structure_class": ["cortical"] * 4,
            "feature": [
                "corr_mean_abs",
                "corr_node_degree_abs_top10",
                "corr_mean_abs",
                "roi_alff_proxy",
            ],
        }
    )


def test_pathway_selection_uses_one_outcome_blind_threshold() -> None:
    selected = select_gene_pathways("GABRA1", _catalog(), _manifest())

    assert selected["pathway_id"].tolist() == ["synaptic"]
    assert selected["threshold_label"].tolist() == ["p1em03"]
    assert selected["score_name"].tolist() == [
        "pathway_prs__synaptic__p1em03"
    ]


def test_outcome_mapping_is_specific_and_rejects_unavailable_scales() -> None:
    assert map_outcome("Clinical Dementia Rating - Sum of Boxes") == {
        "outcome": "CDRSB",
        "specificity": 1.0,
        "mapping": "exact_scale",
    }
    assert map_outcome("general cognition")["outcome"] == "MMSE"
    assert map_outcome("Hamilton Depression Rating Scale") is None


def test_connectivity_mapping_uses_partial_acc_cortical_proxy() -> None:
    hypothesis = {
        "source_name": "GABRA1",
        "path": [
            {
                "to_name": "Amygdala",
            },
            {
                "to_name": (
                    "rs-fMRI functional connectivity among Amygdala, "
                    "Hippocampus, and Anterior Cingulate Cortex"
                ),
            },
            {
                "to_name": "Clinical Dementia Rating - Sum of Boxes",
            },
        ],
    }

    markers, mapping, reason = map_imaging_markers(
        hypothesis,
        _marker_catalog(),
    )

    assert reason == ""
    assert set(markers["marker_id"]) == {"acc_corr", "acc_degree"}
    assert mapping["mapping"] == "partial_cortical_proxy"
    assert mapping["mapped_regions"] == "anterior cingulate cortex"
    assert set(mapping["unavailable_regions"].split(";")) == {
        "amygdala",
        "hippocampus",
    }
    assert np.isclose(mapping["spatial_coverage"], 1 / 3)


def test_tau_mapping_is_rejected_in_fmri_only_table() -> None:
    hypothesis = {
        "source_name": "CHRNA4",
        "path": [
            {"to_name": "Entorhinal Cortex"},
            {"to_name": "tau_suvr in Entorhinal Cortex"},
            {"to_name": "Clinical Dementia Rating - Sum of Boxes"},
        ],
    }

    markers, mapping, reason = map_imaging_markers(
        hypothesis,
        _marker_catalog(),
    )

    assert markers.empty
    assert mapping is None
    assert reason == "unsupported_imaging_modality:tau_suvr"


def test_diversified_rank_round_robins_hypotheses() -> None:
    candidates = pd.DataFrame(
        {
            "primary_hypothesis_id": ["h1", "h1", "h1", "h2", "h2"],
            "mapping_score": [0.9, 0.9, 0.9, 0.8, 0.8],
            "pathway_specificity": [1.0] * 5,
            "feature_priority": [0] * 5,
            "marker_id": ["a", "b", "c", "d", "e"],
        }
    )

    ranked = _diversified_rank(candidates)

    assert ranked["primary_hypothesis_id"].tolist() == [
        "h1",
        "h2",
        "h1",
        "h2",
        "h1",
    ]
    assert ranked["kg_rank"].tolist() == [1, 2, 3, 4, 5]


def test_prioritisation_evaluation_is_deterministic() -> None:
    rows = []
    for exposure in (
        "pathway_prs__synaptic__p1em03",
        "pathway_prs__immune__p5em02",
    ):
        for outcome in ("CDRSB", "MMSE"):
            for index in range(20):
                rows.append(
                    {
                        "exposure": exposure,
                        "outcome": outcome,
                        "marker_id": f"{exposure}_{outcome}_{index}",
                        "sobel_p": 0.01 if index < 2 else 0.5,
                        "sobel_q_family": 0.2,
                        "abs_indirect_effect_std": (
                            0.2 if index < 2 else 0.01
                        ),
                    }
                )
    results = pd.DataFrame(rows)
    candidates = results.iloc[[0, 1, 20, 21]].copy()
    candidates.insert(0, "kg_rank", [1, 2, 3, 4])

    _, metrics_a, distribution_a = evaluate_prioritisation(
        candidates,
        results.copy(),
        _manifest(),
        k_values=(2, 4),
        n_resamples=100,
        seed=17,
    )
    _, metrics_b, distribution_b = evaluate_prioritisation(
        candidates,
        results.copy(),
        _manifest(),
        k_values=(2, 4),
        n_resamples=100,
        seed=17,
    )

    pd.testing.assert_frame_equal(metrics_a, metrics_b)
    pd.testing.assert_frame_equal(distribution_a, distribution_b)
    assert metrics_a.loc[metrics_a["k"].eq(2), "kg_exploratory_hits"].eq(2).all()

# Last Updated At: 2026-07-31 03:21 HKT
