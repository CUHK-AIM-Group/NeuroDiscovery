from __future__ import annotations

import pytest

from neurooracle.src.case_studies import (
    CASE1,
    CASE2,
    CASE3,
    GENERATOR_CASE1_CANDIDATE,
    case_study_by_name,
    list_case_study_names,
)


def test_case_study_registry_uses_final_numbering():
    names = list_case_study_names()

    assert names[:3] == (
        "case1_transdiagnostic",
        "case2_pathway_mediation",
        "case3_hindcasting",
    )
    assert case_study_by_name("case1_transdiagnostic") is CASE1
    assert case_study_by_name("case2_pathway_mediation") is CASE2
    assert case_study_by_name("case3_hindcasting") is CASE3
    assert CASE1.generator == GENERATOR_CASE1_CANDIDATE


def test_case_study_registry_rejects_legacy_aliases():
    for old_name in (
        "cs2_transdiagnostic",
        "cs3_pathway_mediation",
        "cs_gamma_hindcasting",
        "cs_y_hindcasting",
    ):
        with pytest.raises(KeyError):
            case_study_by_name(old_name)


def test_case1_feature_space_is_predeclared():
    features = CASE1.extras["feature_space"]
    feature_ids = {f["id"] for f in features}

    assert len(features) == 15
    assert all("direction" not in f for f in features)
    assert all(f["requires"] for f in features)

    assert {
        "roi_alff",
        "roi_falff",
        "roi_temporal_variance",
        "roi_mean_whole_brain_fc",
        "roi_within_network_fc",
        "roi_between_network_fc",
        "roi_node_strength",
        "roi_node_degree",
        "roi_participation_coefficient",
        "roi_local_efficiency",
        "roi_fc_variability",
        "subject_state_occupancy",
    } <= feature_ids

    primary = [f for f in features if f["primary"]]
    structural = [f for f in features if f["family"] == "structural"]
    assert len(primary) == 12
    assert {f["modality"] for f in primary} == {"fMRI"}
    assert {f["modality"] for f in structural} == {"sMRI"}
