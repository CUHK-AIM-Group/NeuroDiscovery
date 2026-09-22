from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.scripts.case_study_feedback_adapters import (
    EXECUTION_FAILED,
    INCONCLUSIVE,
    SUPPORTED,
)
from neurooracle.scripts.run_neurodiscovery_closed_loop_hindcasting import (
    LoopProfile,
    feedback_outcomes,
    prepare_run_bundle,
    profile_slate,
    rank_bundle,
)


def _write_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    path = [
        {"from_id": "A", "to_id": "M", "relation_type": "modulates"},
        {"from_id": "M", "to_id": "B", "relation_type": "predicts"},
    ]
    hypotheses = [
        {
            "id": "H1",
            "hypothesis_type": "bridge",
            "source_id": "A",
            "target_id": "B",
            "path": path,
            "confidence_score": 0.8,
            "evidence_score": 0.7,
            "novelty_score": 0.6,
            "testability_score": 0.9,
            "metadata": {"domain_a": "gene", "domain_b": "disease"},
        },
        {
            "id": "H2",
            "hypothesis_type": "bridge",
            "source_id": "A",
            "target_id": "B",
            "path": path,
            "confidence_score": 0.9,
            "evidence_score": 0.9,
            "novelty_score": 0.9,
            "testability_score": 0.9,
            "metadata": {"domain_a": "gene", "domain_b": "disease"},
        },
        {
            "id": "H3",
            "hypothesis_type": "bridge",
            "source_id": "C",
            "target_id": "D",
            "path": [
                {"from_id": "C", "to_id": "N", "relation_type": "marks"},
                {"from_id": "N", "to_id": "D", "relation_type": "predicts"},
            ],
            "confidence_score": 0.5,
            "evidence_score": 0.5,
            "novelty_score": 0.5,
            "testability_score": 0.5,
            "metadata": {"domain_a": "imaging", "domain_b": "outcome"},
        },
        {
            "id": "FAIL",
            "hypothesis_type": "generation_failure",
            "source_id": "",
            "target_id": "",
            "path": [],
            "confidence_score": 0.0,
            "evidence_score": 0.0,
            "novelty_score": 0.0,
            "testability_score": 0.0,
            "metadata": {"generation_failure": True},
        },
    ]
    hypotheses_path = tmp_path / "hypotheses.json"
    hypotheses_path.write_text(
        json.dumps({"hypotheses": hypotheses}), encoding="utf-8"
    )
    scored_path = tmp_path / "scored.csv"
    pd.DataFrame(
        [
            {
                "id": "H1",
                "primary_hit": True,
                "primary_year": 2018,
                "any_future_hit": True,
                "first_future_year": 2017,
            },
            {
                "id": "H2",
                "primary_hit": True,
                "primary_year": 2018,
                "any_future_hit": True,
                "first_future_year": 2017,
            },
            {
                "id": "H3",
                "primary_hit": True,
                "primary_year": 2020,
                "any_future_hit": True,
                "first_future_year": 2019,
            },
            {
                "id": "FAIL",
                "primary_hit": False,
                "primary_year": None,
                "any_future_hit": False,
                "first_future_year": None,
            },
        ]
    ).to_csv(scored_path, index=False)
    run = {
        "case_study_id": "biomarker_discovery",
        "seed": 0,
        "freeze_year": 2016,
        "future_start_year": 2017,
        "future_end_year": 2021,
    }
    return run, hypotheses_path, scored_path


def test_bundle_neutralizes_duplicates_and_keeps_outcomes_hidden(tmp_path: Path) -> None:
    run, hypotheses_path, scored_path = _write_fixture(tmp_path)
    bundle = prepare_run_bundle(
        run,
        hypotheses_path=hypotheses_path,
        scored_path=scored_path,
        feedback_years=2,
    )

    assert bundle.audit["semantic_duplicates_neutralized"] == 1
    assert bundle.public.set_index("candidate_id").loc["H2", "is_semantic_duplicate"]
    assert not bundle.hidden.set_index("candidate_id").loc["H2", "full_primary_hit"]
    assert "primary_year" not in bundle.public
    assert "terminal_primary_hit" not in bundle.public


def test_temporal_feedback_does_not_reveal_terminal_support(tmp_path: Path) -> None:
    run, hypotheses_path, scored_path = _write_fixture(tmp_path)
    bundle = prepare_run_bundle(
        run,
        hypotheses_path=hypotheses_path,
        scored_path=scored_path,
        feedback_years=2,
    )
    outcomes = feedback_outcomes(bundle, "temporal_transfer").set_index("candidate_id")

    assert outcomes.loc["H1", "feedback_status"] == SUPPORTED
    assert outcomes.loc["H3", "feedback_status"] == INCONCLUSIVE
    assert outcomes.loc["FAIL", "feedback_status"] == EXECUTION_FAILED
    assert bundle.hidden.set_index("candidate_id").loc["H3", "terminal_primary_hit"]


def test_closed_loop_returns_a_full_permutation(tmp_path: Path) -> None:
    run, hypotheses_path, scored_path = _write_fixture(tmp_path)
    bundle = prepare_run_bundle(
        run,
        hypotheses_path=hypotheses_path,
        scored_path=scored_path,
        feedback_years=2,
    )
    profile = next(profile for profile in profile_slate() if not profile.static_only)
    order, _trace, manifest = rank_bundle(
        bundle,
        profile,
        feedback_mode="temporal_transfer",
    )

    assert sorted(order.tolist()) == list(range(len(bundle.public)))
    assert manifest["outcome_blind"] is True
    assert manifest["mode"] == "temporal_transfer"


def test_delayed_feedback_start_adapts_to_public_candidate_coverage() -> None:
    profile = LoopProfile(
        "delayed",
        "legacy",
        0.20,
        0.20,
        0.25,
        0.35,
        feedback_start_budget=50,
        feedback_start_fraction=0.25,
    )

    sparse = profile.closed_loop_config(1000, valid_candidate_count=14)
    broad = profile.closed_loop_config(1000, valid_candidate_count=600)

    assert sparse.batch_size * sparse.warmup_batches == 10
    assert broad.batch_size * broad.warmup_batches == 50
    assert broad.sampling_temperature == 1e-6
    assert broad.warmup_diversity_penalty == 10.0


# Updated: 2026-08-11 18:31 HKT
