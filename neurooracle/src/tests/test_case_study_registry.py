from __future__ import annotations

import pytest

from neurooracle.src.case_studies import (
    CASE1,
    CASE2,
    TASK_CASE_STUDIES,
    GENERATOR_CASE1_CANDIDATE,
    GENERATOR_TASK,
    CASE_STUDY_DISPLAY_NUMBERS,
    case_study_by_name,
    case_study_display_number,
    list_case_study_catalog,
    list_case_study_names,
)


def test_case_study_registry_contains_17_peer_scopes():
    names = list_case_study_names()

    assert len(names) == 17
    assert len(set(names)) == 17
    assert names[:2] == (
        "case1_transdiagnostic",
        "case2_pathway_mediation",
    )
    assert "case3_hindcasting" not in names
    assert len(TASK_CASE_STUDIES) == 15
    assert case_study_by_name("case1_transdiagnostic") is CASE1
    assert case_study_by_name("case2_pathway_mediation") is CASE2
    assert case_study_by_name("brain_age").generator == GENERATOR_TASK
    assert case_study_by_name("drug_repurposing").name == "drug_repurposing"
    assert CASE1.generator == GENERATOR_CASE1_CANDIDATE


def test_case_study_display_numbers_are_derived_without_creating_case3():
    names = list_case_study_names()

    assert CASE_STUDY_DISPLAY_NUMBERS == {
        name: index for index, name in enumerate(names, start=1)
    }
    assert case_study_display_number("case1_transdiagnostic") == 1
    assert case_study_display_number("case2_pathway_mediation") == 2
    assert case_study_display_number("brain_age") == names.index("brain_age") + 1
    with pytest.raises(KeyError):
        case_study_display_number("case3_hindcasting")

    catalog = list_case_study_catalog()
    assert len(catalog) == 17
    assert [item["number"] for item in catalog] == list(range(1, 18))
    assert all(item["id"] != "case3_hindcasting" for item in catalog)


def test_case_study_registry_rejects_legacy_aliases():
    for old_name in (
        "cs2_transdiagnostic",
        "cs3_pathway_mediation",
        "cs_gamma_hindcasting",
        "cs_y_hindcasting",
        "case3_hindcasting",
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
