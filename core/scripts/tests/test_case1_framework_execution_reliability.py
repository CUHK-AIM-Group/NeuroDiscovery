from __future__ import annotations

import pandas as pd

from core.scripts.run_case1_framework_execution_reliability import (
    _parse_candidate,
    expand_execution_by_method,
    select_hypotheses,
)


def test_parse_candidate_maps_multiatlas_source() -> None:
    parsed = _parse_candidate(
        "fmri|schaefer_100_7net_multiatlas|MDD_depression|corr_mean_abs|12"
    )
    assert parsed["atlas"] == "schaefer_100_7net"
    assert parsed["roi_index"] == 12


def test_select_hypotheses_uses_unique_executable_cells() -> None:
    rows = pd.DataFrame(
        [
            {
                "method": "neurodiscovery",
                "generator_rank": 1,
                "candidate_id": "fmri|atlas_a_multiatlas|disease_a|feature_a|0",
            },
            {
                "method": "neurodiscovery",
                "generator_rank": 2,
                "candidate_id": "fmri|atlas_a_multiatlas|disease_a|feature_b|1",
            },
            {
                "method": "neurodiscovery",
                "generator_rank": 3,
                "candidate_id": "fmri|atlas_b_multiatlas|disease_b|feature_a|0",
            },
        ]
    )
    selected = select_hypotheses(
        rows,
        roi_counts={"atlas_a": 20, "atlas_b": 30},
        per_method=2,
        max_rois=40,
    )
    assert selected["candidate_id"].tolist() == [
        "fmri|atlas_a_multiatlas|disease_a|feature_a|0",
        "fmri|atlas_b_multiatlas|disease_b|feature_a|0",
    ]


def test_expand_execution_counts_failures_without_repair() -> None:
    selected = pd.DataFrame(
        [
            {
                "method": "neurodiscovery",
                "label": "NeuroDiscovery",
                "selection_slot": 1,
                "atlas": "atlas_a",
                "disease": "disease_a",
            },
            {
                "method": "neurodiscovery",
                "label": "NeuroDiscovery",
                "selection_slot": 2,
                "atlas": "atlas_b",
                "disease": "disease_b",
            },
        ]
    )
    runs = pd.DataFrame(
        [
            {
                "atlas": "atlas_a",
                "disease": "disease_a",
                "model": "bnt",
                "execution_success": True,
            },
            {
                "atlas": "atlas_b",
                "disease": "disease_b",
                "model": "bnt",
                "execution_success": False,
            },
        ]
    )
    _expanded, summary = expand_execution_by_method(selected, runs)
    total = summary[summary["model"].eq("ALL")].iloc[0]
    assert total["attempted_hypothesis_model_executions"] == 2
    assert total["successful_hypothesis_model_executions"] == 1
    assert total["execution_success_rate"] == 0.5
