from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.scripts import tune_case1_neurodiscovery as tuning
from core.scripts.case1_method_comparison import (
    KgIndex,
    add_generator_scores,
    closed_loop_neurodiscovery_order,
    kg_query_terms_for_candidates,
    load_results,
)
from core.scripts.case_study_closed_loop_engine import ExperimentalOverlayGraph
from core.scripts.case_study_feedback_adapters import adapter_for
from core.scripts.case1_neurodiscovery_config import (
    Case1NeuroDiscoveryConfig,
    feature_terms,
)
from core.scripts.tune_case1_neurodiscovery import aggregate_runs, stable_fold


def test_masked_feedback_frame_does_not_turn_hidden_fold_into_negative_feedback() -> None:
    scored = pd.DataFrame(
        {
            "adjusted_residual_d": [0.4, -0.3, 0.2],
            "p_value": [0.001, 0.002, 0.01],
        }
    )
    masked = tuning.masked_feedback_frame(
        scored,
        np.array([False, True, False]),
    )

    assert masked["feedback_available"].tolist() == [True, False, True]
    assert masked.loc[1, "adjusted_residual_d"] == 0.0
    assert masked.loc[1, "p_value"] == 1.0


def test_config_round_trip_and_feature_vocabulary(tmp_path: Path) -> None:
    config = Case1NeuroDiscoveryConfig(
        feature_support_weight=0.25,
        warmup_budget=5_000,
        pair_feedback_start_fraction=0.10,
    )
    path = tmp_path / "config.json"
    config.write_json(path)

    assert Case1NeuroDiscoveryConfig.from_json(path) == config
    assert "functional connectivity" in feature_terms("corr_mean")
    assert "gray matter volume" in feature_terms("normalized_volume_fraction")


def test_candidate_universe_retains_execution_failures(tmp_path: Path) -> None:
    path = tmp_path / "all_tests.csv"
    pd.DataFrame(
        {
            "modality": ["func", "func", "func"],
            "source": ["atlas", "atlas", "atlas"],
            "disease": ["ADHD", "ADHD", "ADHD"],
            "feature": ["corr_mean", "corr_mean", "corr_mean"],
            "roi_index": [1, 2, 3],
            "adjusted_residual_d": [0.8, 0.2, np.nan],
            "abs_adjusted_residual_d": [0.8, 0.2, np.nan],
            "p_value": [0.001, 0.2, np.nan],
            "q_fdr_global": [0.01, 0.5, np.nan],
        }
    ).to_csv(path, index=False)

    loaded = load_results(path, gt_top_frac=0.5)

    assert len(loaded) == 3
    assert loaded["execution_succeeded"].tolist() == [True, True, False]
    assert loaded["is_gt_top"].tolist() == [True, False, False]


def test_kg_query_terms_include_executable_features() -> None:
    candidates = pd.DataFrame(
        {
            "disease": ["ADHD"],
            "feature": ["roi_falff_proxy"],
            "anatomy_full": ["insula"],
        }
    )

    terms = kg_query_terms_for_candidates(candidates)

    assert "fALFF" in terms
    assert "fractional amplitude of low frequency fluctuation" in terms
    assert "insula" in terms


def test_feature_kg_support_changes_only_opted_in_neurodiscovery_score() -> None:
    candidates = pd.DataFrame(
        [
            {
                "modality": "func",
                "source": "atlas",
                "disease": "ADHD",
                "feature": "corr_mean",
                "roi_index": 1,
                "roi_name": "insula",
                "anatomy_key": "insula",
                "anatomy_full": "insula",
                "network": "Salience",
                "structure_class": "cortical",
                "n_case": 40,
                "n_control": 40,
                "abs_adjusted_residual_d": 0.1,
            },
            {
                "modality": "func",
                "source": "atlas",
                "disease": "ADHD",
                "feature": "roi_temporal_mean",
                "roi_index": 1,
                "roi_name": "insula",
                "anatomy_key": "insula",
                "anatomy_full": "insula",
                "network": "Salience",
                "structure_class": "cortical",
                "n_case": 40,
                "n_control": 40,
                "abs_adjusted_residual_d": 0.9,
            },
        ]
    )
    adjacency = {
        "D": {"R", "F"},
        "R": {"D", "F"},
        "F": {"D", "R"},
    }
    kg = KgIndex(
        degrees={"D": 4, "R": 4, "F": 4},
        name_to_ids={
            "adhd": ("D",),
            "insula": ("R",),
            "functional connectivity": ("F",),
        },
        name_to_degree={"adhd": 4, "insula": 4, "functional connectivity": 4},
        adjacency=adjacency,
        directed_support={("D", "F"): 2.0, ("R", "F"): 2.0},
        scoped_degrees={"D": 2, "R": 2, "F": 2},
        name_to_scoped_degree={"adhd": 2, "insula": 2, "functional connectivity": 2},
        scoped_adjacency=adjacency,
        scoped_directed_support={("D", "F"): 1.0, ("R", "F"): 1.0},
    )

    legacy = add_generator_scores(
        candidates,
        kg,
        seed=7,
        config=Case1NeuroDiscoveryConfig(feature_support_weight=0.0),
    )
    feature_aware = add_generator_scores(
        candidates,
        kg,
        seed=7,
        config=Case1NeuroDiscoveryConfig(feature_support_weight=1.0),
    )

    assert legacy.loc[0, "score_neurodiscovery_global"] == legacy.loc[
        1, "score_neurodiscovery_global"
    ]
    assert feature_aware.loc[0, "score_neurodiscovery"] > feature_aware.loc[
        1, "score_neurodiscovery"
    ]
    assert feature_aware.loc[0, "kg_disease_feature_support"] > 0
    assert feature_aware.loc[1, "kg_disease_feature_support"] == 0


def test_closed_loop_manifest_records_effective_config() -> None:
    rows = []
    for index in range(24):
        rows.append(
            {
                "candidate_id": f"candidate-{index}",
                "score_neurodiscovery": 1.0 - index / 100.0,
                "disease": f"disease-{index % 3}",
                "feature": f"feature-{index % 4}",
                "feature_family": f"feature-{index % 4}",
                "map_group": f"group-{index % 5}",
                "roi_key": f"roi-{index % 6}",
                "anatomy_full": f"region-{index % 6}",
                "source": f"source-{index % 2}",
                "adjusted_residual_d": 0.3 if index % 3 == 0 else 0.05,
                "p_value": 0.001 if index % 3 == 0 else 0.5,
                "is_gt_top": index % 2 == 0,
            }
        )
    config = Case1NeuroDiscoveryConfig(
        batch_size=4,
        warmup_budget=8,
        max_closed_loop_budget=20,
        warmup_exploit_fraction=0.65,
        feedback_weight=0.14,
    )

    order, manifest = closed_loop_neurodiscovery_order(
        pd.DataFrame(rows),
        np.random.default_rng(42),
        return_overlay_manifest=True,
        config=config,
    )

    assert sorted(order.tolist()) == list(range(24))
    assert manifest["search_config"]["warmup_exploit_fraction"] == 0.65
    assert manifest["search_config"]["feedback_weight"] == 0.14
    assert manifest["records"] == 20

    unavailable = pd.DataFrame(rows)
    unavailable["feedback_available"] = False
    _, unavailable_manifest = closed_loop_neurodiscovery_order(
        unavailable,
        np.random.default_rng(42),
        return_overlay_manifest=True,
        config=config,
    )
    assert unavailable_manifest["records"] == 20
    assert unavailable_manifest["ranking_feedback_records"] == 0
    assert unavailable_manifest["nonzero_feedback_reads"] == 0
    assert unavailable_manifest["selection_changed_batches"] == 0


def test_formal_closed_loop_reveals_only_after_batch_commitment() -> None:
    public_rows = []
    hidden = {}
    for index in range(24):
        candidate_id = f"candidate-{index}"
        public_rows.append(
            {
                "candidate_id": candidate_id,
                "score_neurodiscovery": 1.0 - index / 100.0,
                "disease": f"disease-{index % 3}",
                "feature": f"feature-{index % 4}",
                "feature_family": f"feature-{index % 4}",
                "map_group": f"group-{index % 5}",
                "roi_key": f"roi-{index % 6}",
                "anatomy_full": f"region-{index % 6}",
                "source": f"source-{index % 2}",
            }
        )
        hidden[candidate_id] = {
            "candidate_id": candidate_id,
            "execution_succeeded": True,
            "adjusted_residual_d": 0.3 if index % 3 == 0 else 0.05,
            "p_value": 0.001 if index % 3 == 0 else 0.5,
        }

    events: list[tuple[str, tuple[str, ...]]] = []

    def commit(payload):
        candidate_ids = tuple(payload["candidate_ids"])
        events.append(("commit", candidate_ids))
        assert payload["outcomes_read_before_commit"] is False
        return {
            "commit_sha256": f"commit-{len(events)}",
            "candidate_ids": list(candidate_ids),
        }

    def reveal(candidate_ids, selection_commit):
        candidate_ids = tuple(candidate_ids)
        assert tuple(selection_commit["candidate_ids"]) == candidate_ids
        assert events[-1] == ("commit", candidate_ids)
        events.append(("reveal", candidate_ids))
        return [hidden[candidate_id] for candidate_id in candidate_ids]

    config = Case1NeuroDiscoveryConfig(
        batch_size=4,
        warmup_budget=8,
        max_closed_loop_budget=20,
    )
    order, audit, manifest = closed_loop_neurodiscovery_order(
        pd.DataFrame(public_rows),
        np.random.default_rng(42),
        return_audit=True,
        return_overlay_manifest=True,
        config=config,
        batch_commit_callback=commit,
        outcome_reveal_callback=reveal,
    )

    assert sorted(order.tolist()) == list(range(24))
    assert [name for name, _ in events] == ["commit", "reveal"] * 5
    assert audit["outcomes_revealed_after_commitment"].all()
    assert audit["selection_commit_sha256"].notna().all()
    assert manifest["batch_selection_commits"] == {
        "enabled": True,
        "count": 5,
        "outcome_reveal_count": 5,
        "committed_before_selected_outcome_lookup": True,
    }
    assert manifest["formal_outcome_vault"] == {
        "enabled": True,
        "hidden_outcome_columns_in_scoring_frame": [],
        "scoring_frame_outcome_blind": True,
        "outcomes_revealed_only_after_commitment": True,
    }
    assert manifest["feedback_consumed_during_ranking"] is True


def test_tuning_fold_and_robust_objective_are_deterministic() -> None:
    assert stable_fold("ADHD", "schaefer400", "roi-1") == stable_fold(
        "ADHD", "schaefer400", "roi-1"
    )
    runs = pd.DataFrame(
        [
            {"config_id": "a", "config_json": "{}", "objective": 0.8, "hits_at_5000": 8},
            {"config_id": "a", "config_json": "{}", "objective": 0.2, "hits_at_5000": 2},
            {"config_id": "b", "config_json": "{}", "objective": 0.45, "hits_at_5000": 4},
            {"config_id": "b", "config_json": "{}", "objective": 0.45, "hits_at_5000": 4},
        ]
    )

    summary = aggregate_runs(runs).set_index("config_id")

    assert summary.loc["a", "robust_objective"] == 0.425
    assert summary.loc["b", "robust_objective"] == 0.45
    assert summary.index[0] == "b"


def test_dynamic_tuning_workers_checkpoint_each_run(
    tmp_path: Path, monkeypatch
) -> None:
    n = 12
    scored = pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index}" for index in range(n)],
            "score_neurodiscovery_global_base": np.linspace(1.0, 0.0, n),
            "score_case_study_support_base": np.linspace(0.0, 1.0, n),
            "score_feature_support_global": np.linspace(1.0, 0.0, n),
            "score_feature_support_scoped": np.linspace(0.0, 1.0, n),
            "score_random": np.linspace(0.1, 0.2, n),
            "adjusted_residual_d": np.linspace(0.4, 0.0, n),
            "p_value": np.linspace(0.001, 0.5, n),
            "is_gt_top": [index < 3 for index in range(n)],
        }
    )

    def fake_order(frame, rng, **kwargs):
        del rng, kwargs
        return np.argsort(-frame["score_neurodiscovery"].to_numpy(float))

    monkeypatch.setattr(tuning, "closed_loop_neurodiscovery_order", fake_order)
    checkpoint = tmp_path / "checkpoint.jsonl"
    rows = tuning.run_dynamic_configs(
        scored,
        np.zeros(n, dtype=np.int8),
        [Case1NeuroDiscoveryConfig()],
        hidden_mask=np.zeros(n, dtype=bool),
        evaluation_mask=np.ones(n, dtype=bool),
        seeds=[1, 2, 3],
        stage="test",
        evaluation_fold=0,
        checkpoint_path=checkpoint,
        workers=3,
    )

    assert len(rows) == 3
    assert set(rows["seed"]) == {1, 2, 3}
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 3


def test_inconclusive_feedback_can_reduce_search_utility_without_contradiction() -> None:
    public = pd.DataFrame(
        [
            {
                "candidate_id": f"candidate-{index}",
                "disease": "ADHD",
                "feature": "corr_mean",
                "feature_family": "tested" if index < 2 else "untested",
                "atlas": "atlas",
                "anatomy": "insula",
            }
            for index in range(4)
        ]
    )
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter_for("case1_transdiagnostic"),
        factor_fields=("feature_family",),
        seed=1,
        trial=0,
    )
    for index in (0, 1):
        overlay.append(
            candidate_index=index,
            outcome={
                "validated": False,
                "feedback_status": "inconclusive",
                "effect_size": 0.0,
                "p_value": 1.0,
            },
            round_index=index,
        )

    neutral, _ = overlay.score_candidates(
        feedback_weight=1.0,
        pair_feedback_weight=0.0,
        exploration_weight=0.0,
        inconclusive_failure_weight=0.0,
    )
    utility_aware, _ = overlay.score_candidates(
        feedback_weight=1.0,
        pair_feedback_weight=0.0,
        exploration_weight=0.0,
        inconclusive_failure_weight=1.0,
    )

    np.testing.assert_allclose(neutral, 0.0)
    assert utility_aware[2] > utility_aware[0]
    assert overlay.status_counts["contradicted"] == 0
    assert overlay.status_counts["inconclusive"] == 2
