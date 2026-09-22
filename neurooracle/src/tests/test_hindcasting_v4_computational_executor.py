from __future__ import annotations

import pandas as pd

from neurooracle.src.hindcasting_v4_computational_executor import (
    ComputationalMapping,
    ComputationalOutcomeVault,
    FeedbackRanker,
    PublicCandidateMapper,
    anatomy_tags,
    disease_group,
    feature_coordinate,
)
from neurooracle.scripts.run_hindcasting_v4_discovery import _closed_loop_order


def _public() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "disease": "ADHD",
                "modality": "smri",
                "feature": "normalized_volume_fraction",
                "feature_family": "structure",
                "anatomy_full": "Left Hippocampus",
                "roi_name": "Left Hippocampus",
            },
            {
                "candidate_id": "c2",
                "disease": "bipolar",
                "modality": "fmri",
                "feature": "corr_mean",
                "feature_family": "correlation_fc",
                "anatomy_full": "Right Amygdala",
                "roi_name": "Right Amygdala",
            },
        ]
    )


def test_controlled_semantic_rules_fail_closed() -> None:
    assert disease_group("attention-deficit hyperactivity disorder") == "ADHD"
    assert feature_coordinate("hippocampal volume reduction") == (
        "normalized_volume_fraction",
        "structure",
    )
    assert feature_coordinate("hippocampal cortical thickness") is None
    assert anatomy_tags("bilateral hippocampal volume reduction") == ("hippocampus",)


def test_public_mapper_never_accepts_outcome_columns() -> None:
    public = _public()
    public["p_value"] = 0.01
    try:
        PublicCandidateMapper(public)
    except ValueError as exc:
        assert "outcome columns" in str(exc)
    else:
        raise AssertionError("outcome-bearing public table was accepted")


def test_mapper_uses_public_coordinates_only() -> None:
    mapper = PublicCandidateMapper(_public())
    mapping = mapper.map_one(
        {
            "id": "h1",
            "source_id": "im1",
            "source_name": "hippocampal volume reduction",
            "target_id": "d1",
            "target_name": "attention-deficit hyperactivity disorder",
        }
    )
    assert mapping.executable
    assert mapping.candidate_id == "c1"
    assert mapping.expected_direction == "lower"


def test_case1_vault_distinguishes_contradiction_and_failure() -> None:
    mapper = PublicCandidateMapper(_public())
    mapping = mapper.map_one(
        {
            "id": "h1",
            "source_id": "im1",
            "source_name": "hippocampal volume reduction",
            "target_id": "d1",
            "target_name": "ADHD",
        }
    )
    vault = ComputationalOutcomeVault(
        pd.DataFrame(
            [
                {
                    "candidate_id": "c1",
                    "execution_succeeded": True,
                    "adjusted_residual_d": 0.4,
                    "p_value": 0.001,
                }
            ]
        ),
        task_id="case1_transdiagnostic",
    )
    result = vault.reveal(mapping)
    assert result["feedback_status"] == "contradicted"
    assert 0.0 < result["feedback_utility"] <= 1.0


def test_feedback_ranker_changes_only_after_computational_result() -> None:
    mapper = PublicCandidateMapper(_public())
    mappings = mapper.map_all(
        [
            {
                "id": "h1",
                "source_id": "im1",
                "source_name": "hippocampal volume reduction",
                "target_id": "d1",
                "target_name": "ADHD",
            },
            {
                "id": "h2",
                "source_id": "im2",
                "source_name": "amygdala functional connectivity",
                "target_id": "d2",
                "target_name": "bipolar disorder",
            },
        ]
    )
    ranker = FeedbackRanker(mappings)
    assert not ranker.adjustments(total_observations=0).any()
    ranker.update(
        mappings[0],
        {"feedback_utility": 0.9, "feedback_status": "supported"},
    )
    adjusted = ranker.adjustments(total_observations=1)
    assert adjusted.shape == (2,)
    assert adjusted[0] != adjusted[1]


def test_closed_loop_commits_before_every_reveal(tmp_path) -> None:
    hypotheses = [
        {
            "id": f"h{index}",
            "source_id": f"im{index}",
            "source_name": "hippocampal volume",
            "target_id": "d1",
            "target_name": "ADHD",
            "path": [],
            "composite_score": 1.0 - index / 100.0,
            "metadata": {"case_study_id": "case1_transdiagnostic"},
        }
        for index in range(60)
    ]
    mappings = [
        ComputationalMapping(
            "mapped",
            "strict_disease_measurement_anatomy_match",
            f"h{index}",
            candidate_id=f"c{index}",
            disease="ADHD",
            modality="smri",
            feature="normalized_volume_fraction",
            feature_family="structure",
            anatomy_tag="hippocampus",
        )
        for index in range(60)
    ]
    vault = ComputationalOutcomeVault(
        pd.DataFrame(
            [
                {
                    "candidate_id": f"c{index}",
                    "execution_succeeded": True,
                    "adjusted_residual_d": 0.2 + index / 1000.0,
                    "p_value": 0.001,
                }
                for index in range(60)
            ]
        ),
        task_id="case1_transdiagnostic",
    )
    ordered, feedback, audit = _closed_loop_order(
        hypotheses,
        mappings,
        vault,
        task_id="case1_transdiagnostic",
        seed=0,
        freeze_year=2016,
        batch_size=10,
        warmup_budget=50,
        feedback_weight=0.10,
        pair_feedback_weight=0.08,
        exploration_weight=0.015,
        dataset={
            "dataset_id": "test",
            "executor": "test_executor",
            "outcomes": {"sha256": "a" * 64},
        },
        analysis_plan_sha256="b" * 64,
        pipeline_sha256="c" * 64,
        attempt_dir=tmp_path,
    )
    assert len(ordered) == len(feedback) == 60
    assert audit["batch_count"] == 6
    assert audit["feedback_activated"]
    for batch in range(6):
        assert (tmp_path / "batch_commitments" / f"batch_{batch:03d}.json").is_file()
    assert all(row["outcome_observed_after_selection"] for row in feedback)
