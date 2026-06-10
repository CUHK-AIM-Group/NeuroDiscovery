from neurooracle.src.outcome_grounding import (
    best_grounding,
    ground_outcome,
    hcp_label_for_target,
    normalize_outcome_name,
    target_to_hcp_label_mapping,
)


def test_normalize_outcome_name_singularizes_common_wide_terms():
    assert normalize_outcome_name("Executive Functions") == "executive function"
    assert normalize_outcome_name("cognitive-performance") == "cognitive performance"


def test_broad_cognition_maps_to_ready_hcp_label():
    grounding = best_grounding("cognitive performance", dataset="HCP_YA", require_local_label=True)

    assert grounding is not None
    assert grounding.dataset == "HCP_YA"
    assert grounding.label_key == "cogfluidcomp"
    assert grounding.label_file == "data/hcp_cogfluidcomp_labels.csv"
    assert grounding.status == "local_label"


def test_executive_function_keeps_multiple_hcp_candidates():
    groundings = ground_outcome("executive functions", dataset="HCP_YA", require_local_label=True)
    labels = [g.label_key for g in groundings]

    assert labels[:2] == ["flanker", "cardsort"]
    assert "wm_2bk_acc" in labels


def test_adni_scale_grounding_is_explicitly_not_local_label_yet():
    grounding = best_grounding("Mini-Mental State Examination", dataset="ADNI")

    assert grounding is not None
    assert grounding.domain_node_id == "ADNI:DOM_NEUROPSYCH"
    assert grounding.metadata_column == "MMSE"
    assert grounding.status == "domain_only"
    assert not grounding.has_local_label


def test_hcp_prediction_compatibility_mapping_includes_case_study_terms():
    mapping = target_to_hcp_label_mapping()

    assert mapping["general cognition"] == "cogfluidcomp"
    assert mapping["cognitive performance"] == "cogfluidcomp"
    assert hcp_label_for_target("executive functions") == "flanker"
