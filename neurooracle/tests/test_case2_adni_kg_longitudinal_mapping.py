from __future__ import annotations

import numpy as np
import pandas as pd

from neurooracle.scripts.map_case2_kg_hypotheses_to_adni_longitudinal import (
    _imaging_alignment,
    _matrix_bh_rejections,
    _outcome_alignment,
    build_ranked_candidates,
    evaluate_prioritisation,
)


def _hypothesis(
    hypothesis_id: str,
    source: str,
    nodes: list[str],
    target: str,
    score: float,
) -> dict[str, object]:
    path = []
    previous = source
    for node in [*nodes, target]:
        path.append({"from_name": previous, "to_name": node})
        previous = node
    return {
        "id": hypothesis_id,
        "source_name": source,
        "target_name": target,
        "path": path,
        "composite_score": score,
    }


def _catalogs() -> tuple[pd.DataFrame, pd.DataFrame]:
    exposures = pd.DataFrame(
        [
            {
                "score_name": "prs_ad",
                "pathway_id": "curated__ad_risk_gwas",
                "pathway_source": "NeuroOracle curated",
                "pathway_name": "AD_risk_GWAS",
                "threshold_label": "p1em03",
                "gene_count": 2,
            },
            {
                "score_name": "prs_cholinergic",
                "pathway_id": "curated__cholinergic",
                "pathway_source": "NeuroOracle curated",
                "pathway_name": "Cholinergic",
                "threshold_label": "p1em03",
                "gene_count": 2,
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "pathway_id": "curated__ad_risk_gwas",
                "pathway_source": "NeuroOracle curated",
                "pathway_name": "AD_risk_GWAS",
                "reference_id": "curated",
                "gene_count": 2,
                "genes": "APOE;TREM2",
            },
            {
                "pathway_id": "curated__cholinergic",
                "pathway_source": "NeuroOracle curated",
                "pathway_name": "Cholinergic",
                "reference_id": "curated",
                "gene_count": 2,
                "genes": "CHRM1;CHRNA4",
            },
        ]
    )
    return exposures, catalog


def test_exact_tau_cdr_chain_maps_above_unrelated_candidate() -> None:
    hypotheses = [
        _hypothesis(
            "H1",
            "CHRM1",
            ["Entorhinal Cortex", "tau_suvr in Entorhinal Cortex"],
            "Clinical Dementia Rating - Sum of Boxes",
            0.8,
        ),
        _hypothesis(
            "H2",
            "APOE",
            ["amyloid burden"],
            "general cognition",
            0.7,
        ),
    ]
    exposures, catalog = _catalogs()
    universe = pd.DataFrame(
        [
            {
                **exposures.iloc[1].to_dict(),
                "exposure": "prs_cholinergic",
                "modality": "tau_pet",
                "marker": "CTX_ENTORHINAL_SUVR",
                "outcome": "CDRSB",
            },
            {
                **exposures.iloc[0].to_dict(),
                "exposure": "prs_ad",
                "modality": "smri_adnimerge",
                "marker": "Ventricles",
                "outcome": "ADAS13",
            },
        ]
    )
    ranked, support, audit = build_ranked_candidates(
        hypotheses, exposures, catalog, universe
    )
    assert ranked.iloc[0]["exposure"] == "prs_cholinergic"
    assert ranked.iloc[0]["primary_hypothesis_id"] == "H1"
    assert ranked.iloc[0]["primary_imaging_mapping"] == "tau_entorhinal_exact"
    assert ranked.iloc[0]["primary_outcome_mapping"] == "cdrsb_exact"
    assert not support.empty
    assert audit["meaningfully_mapped"].any()


def test_imaging_and_outcome_mapping_do_not_accept_unavailable_scale() -> None:
    hypothesis = _hypothesis(
        "H1",
        "GABRA1",
        ["Anterior Cingulate Cortex", "functional connectivity"],
        "Positive and Negative Syndrome Scale",
        0.8,
    )
    imaging_score, imaging_label = _imaging_alignment(
        hypothesis, "tau_pet", "CTX_ENTORHINAL_SUVR"
    )
    outcome_score, outcome_label = _outcome_alignment(hypothesis, "CDRSB")
    assert imaging_score == 0.03
    assert imaging_label == "weak_imaging_prior"
    assert outcome_score == 0.04
    assert outcome_label == "weak_outcome_prior"


def test_confirmation_outcomes_have_distinct_semantic_mappings() -> None:
    cognition = _hypothesis(
        "H-cognition",
        "APOE",
        ["hippocampal atrophy"],
        "general cognitive decline",
        0.8,
    )
    function = _hypothesis(
        "H-function",
        "APOE",
        ["cortical atrophy"],
        "instrumental activities of daily living",
        0.8,
    )
    memory = _hypothesis(
        "H-memory",
        "APOE",
        ["entorhinal atrophy"],
        "episodic memory decline",
        0.8,
    )

    assert _outcome_alignment(cognition, "mPACCdigit") == (
        0.90,
        "broad_cognition_mpacc_digit",
    )
    assert _outcome_alignment(function, "FAQ") == (1.0, "faq_function_exact")
    assert _outcome_alignment(memory, "LDELTOTAL") == (
        0.95,
        "episodic_memory_ldeltotal",
    )


def test_matrix_bh_rejections_is_row_wise() -> None:
    p_values = np.array(
        [
            [0.001, 0.01, 0.20, 0.80],
            [0.02, 0.03, 0.04, 0.80],
        ]
    )
    rejected = _matrix_bh_rejections(p_values, 0.05)
    assert rejected[0].tolist() == [True, True, False, False]
    assert rejected[1].tolist() == [False, False, False, False]


def test_prioritisation_is_deterministic_and_uses_topk_bh() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "kg_rank": index + 1,
                "sobel_p": 0.001 if index < 3 else 0.50,
                "a_path_p": 0.01 if index < 3 else 0.50,
                "b_path_p": 0.01 if index < 3 else 0.50,
                "indirect_effect_std": 0.1,
            }
        )
    ranked = pd.DataFrame(rows)
    first = evaluate_prioritisation(
        ranked, k_values=(5, 10), n_resamples=200, seed=7
    )
    second = evaluate_prioritisation(
        ranked, k_values=(5, 10), n_resamples=200, seed=7
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[0].iloc[0]["neurodiscovery_topk_fdr_hits"] == 3
    assert first[0].iloc[0]["neurodiscovery_nominal_chain_hits"] == 3


# Last Updated At: 2026-08-16 01:17 HKT
