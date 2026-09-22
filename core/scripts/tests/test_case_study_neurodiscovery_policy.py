from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.scripts.case_study_neurodiscovery_policy import (
    NeuroDiscoveryPolicy,
    apply_neurodiscovery_policy,
    load_neurodiscovery_policy,
    write_neurodiscovery_policy,
)
from core.scripts.tune_case_study_neurodiscovery import (
    dynamic_policy_grid,
    front_loaded_warmup_batches,
    policy_regularization,
    recovery_objective,
    select_top_static_policies,
    static_policy_grid,
    static_control_policy,
    task_budgets,
)


def public_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["c0", "c1", "c2"],
            "kg_global_node_support": [0.0, 0.5, 1.0],
            "kg_global_pair_support": [1.0, 0.5, 0.0],
            "kg_scoped_node_support": [0.2, 0.4, 0.8],
            "kg_scoped_pair_support": [0.8, 0.4, 0.2],
            "score_neurodiscovery": [0.0, 0.0, 0.0],
        }
    )


def test_policy_roundtrip_and_task_guard(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    policy = NeuroDiscoveryPolicy(task="imaging_genetics", batch_size=8)
    write_neurodiscovery_policy(path, policy)

    loaded = load_neurodiscovery_policy(path, task="imaging_genetics")
    assert loaded == policy
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"].endswith(
        ".v8"
    )
    with pytest.raises(ValueError, match="does not match"):
        load_neurodiscovery_policy(path, task="prognosis")


def test_static_score_uses_only_frozen_public_components() -> None:
    public = public_registry()
    policy = NeuroDiscoveryPolicy(
        task="*",
        global_node_weight=0.0,
        global_pair_weight=0.0,
        scoped_node_weight=1.0,
        scoped_pair_weight=0.0,
        tie_break_weight=0.0,
    )

    scored, audit = apply_neurodiscovery_policy(public, policy)

    assert np.allclose(scored["score_neurodiscovery"], public["kg_scoped_node_support"])
    assert audit["uses_experimental_outcomes"] is False


def test_static_score_can_use_frozen_design_prior() -> None:
    public = public_registry()
    public["score_design_prior"] = [0.0, 0.5, 1.0]
    policy = NeuroDiscoveryPolicy(
        task="connectome_behavior",
        global_node_weight=0.0,
        global_pair_weight=0.0,
        scoped_node_weight=0.0,
        scoped_pair_weight=0.0,
        design_prior_weight=0.1,
        tie_break_weight=0.0,
    )

    scored, audit = apply_neurodiscovery_policy(public, policy)

    assert np.allclose(scored["score_neurodiscovery"], [0.0, 0.5, 1.0])
    assert audit["design_prior"]["active"] is True
    assert audit["design_prior"]["uses_experimental_outcomes"] is False


def test_static_score_is_row_order_invariant() -> None:
    policy = NeuroDiscoveryPolicy(task="*", tie_break_weight=0.01)
    first, _ = apply_neurodiscovery_policy(public_registry(), policy)
    shuffled, _ = apply_neurodiscovery_policy(
        public_registry().sample(frac=1.0, random_state=4), policy
    )
    first_scores = first.set_index("candidate_id")["score_neurodiscovery"].sort_index()
    shuffled_scores = shuffled.set_index("candidate_id")[
        "score_neurodiscovery"
    ].sort_index()
    assert np.allclose(first_scores, shuffled_scores)


def test_default_tie_break_does_not_reorder_distinct_evidence_tiers() -> None:
    public = public_registry().iloc[:2].copy()
    public["kg_global_node_support"] = [0.500001, 0.5]
    policy = NeuroDiscoveryPolicy(
        task="*",
        global_node_weight=1.0,
        global_pair_weight=0.0,
        scoped_node_weight=0.0,
        scoped_pair_weight=0.0,
    )

    scored, _ = apply_neurodiscovery_policy(public, policy)

    assert scored.loc[0, "score_neurodiscovery"] > scored.loc[1, "score_neurodiscovery"]


def test_policy_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="at least one"):
        NeuroDiscoveryPolicy(
            global_node_weight=0.0,
            global_pair_weight=0.0,
            scoped_node_weight=0.0,
            scoped_pair_weight=0.0,
            tie_break_weight=0.0,
        ).validate()


def test_v1_policy_migrates_to_all_factor_feedback(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    payload = NeuroDiscoveryPolicy(task="imaging_genetics").to_dict()
    payload["schema_version"] = "case-study-neurodiscovery-policy.v1"
    payload.pop("feedback_projection")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_neurodiscovery_policy(path, task="imaging_genetics")

    assert loaded.feedback_projection == "all_factors"
    assert loaded.preserve_static_until_informative_feedback is False


def test_v2_policy_migrates_without_static_preservation(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v2.json"
    payload = NeuroDiscoveryPolicy(task="imaging_genetics").to_dict()
    payload["schema_version"] = "case-study-neurodiscovery-policy.v2"
    payload.pop("preserve_static_until_informative_feedback")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_neurodiscovery_policy(path, task="imaging_genetics")

    assert loaded.preserve_static_until_informative_feedback is False


def test_v3_policy_is_accepted_as_legacy_schema(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v3.json"
    payload = NeuroDiscoveryPolicy(task="brain_age").to_dict()
    payload["schema_version"] = "case-study-neurodiscovery-policy.v3"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_neurodiscovery_policy(path, task="brain_age")

    assert loaded.task == "brain_age"


def test_v4_policy_migrates_to_beta_count_feedback(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v4.json"
    payload = NeuroDiscoveryPolicy(task="brain_age").to_dict()
    payload["schema_version"] = "case-study-neurodiscovery-policy.v4"
    payload.pop("feedback_model")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_neurodiscovery_policy(path, task="brain_age")

    assert loaded.feedback_model == "beta_counts"


def test_policy_accepts_hierarchical_utility_feedback() -> None:
    policy = replace(NeuroDiscoveryPolicy(), feedback_model="hierarchical_utility")

    policy.validate()

    assert policy.feedback_model == "hierarchical_utility"


def test_policy_accepts_factor_ucb_ridge_surrogate() -> None:
    policy = replace(
        NeuroDiscoveryPolicy(), feedback_model="factor_ucb_ridge_surrogate"
    )

    policy.validate()

    assert policy.feedback_model == "factor_ucb_ridge_surrogate"


def test_v6_policy_defaults_surrogate_ridge_alpha() -> None:
    payload = NeuroDiscoveryPolicy(task="brain_age").to_dict()
    payload["schema_version"] = "case-study-neurodiscovery-policy.v6"
    payload.pop("surrogate_ridge_alpha")

    policy = NeuroDiscoveryPolicy.from_dict(payload)

    assert policy.surrogate_ridge_alpha == 1.0


def test_tuning_uses_registered_task_budgets() -> None:
    assert task_budgets(
        {"task": "biomarker_discovery"},
        n=426_235,
        max_horizon=1_000,
    ) == (100, 500, 1000)


def test_quality_tuning_adds_an_early_budget() -> None:
    assert task_budgets(
        {"task": "connectome_behavior"},
        n=171,
        max_horizon=100,
        include_early_budget=True,
    ) == (10, 25, 50, 100)


def test_static_slate_retains_both_score_families() -> None:
    policies = [
        NeuroDiscoveryPolicy(task="imaging_genetics", score_family="legacy"),
        NeuroDiscoveryPolicy(
            task="imaging_genetics",
            score_family="legacy",
            global_node_weight=0.3,
            global_pair_weight=0.2,
            scoped_node_weight=0.3,
            scoped_pair_weight=0.2,
        ),
        NeuroDiscoveryPolicy(task="imaging_genetics", score_family="relation_aware"),
    ]
    screen = pd.DataFrame(
        {"policy_json": [json.dumps(policy.to_dict()) for policy in policies]}
    )

    selected = select_top_static_policies(screen, 2)

    assert {policy.score_family for policy in selected} == {
        "legacy",
        "relation_aware",
    }


def test_exact_dynamic_tie_prefers_relation_aware_earlier_feedback() -> None:
    preferred = NeuroDiscoveryPolicy(
        task="imaging_genetics",
        score_family="relation_aware",
        batch_size=3,
        feedback_projection="relation_endpoints",
        feedback_weight=2.0,
        pair_feedback_weight=0.15,
        exploration_weight=0.03,
    )
    slower = NeuroDiscoveryPolicy(
        task="imaging_genetics",
        score_family="relation_aware",
        batch_size=14,
        feedback_projection="relation_endpoints",
        feedback_weight=2.0,
        pair_feedback_weight=0.15,
        exploration_weight=0.03,
    )
    legacy = NeuroDiscoveryPolicy(
        task="imaging_genetics",
        score_family="legacy",
        batch_size=3,
        feedback_projection="relation_endpoints",
        feedback_weight=2.0,
        pair_feedback_weight=0.15,
        exploration_weight=0.03,
    )

    assert policy_regularization(preferred) > policy_regularization(slower)
    assert policy_regularization(preferred) > policy_regularization(legacy)


def test_dynamic_grid_does_not_treat_inconclusive_as_negative() -> None:
    policies = dynamic_policy_grid(
        [NeuroDiscoveryPolicy(task="imaging_genetics")],
        smallest_budget=25,
        horizon=100,
    )

    assert policies
    assert all(policy.inconclusive_search_failure_weight == 0.0 for policy in policies)


def test_quality_endpoint_grid_uses_continuous_closed_loop_surrogate() -> None:
    policies = dynamic_policy_grid(
        [NeuroDiscoveryPolicy(task="connectome_behavior")],
        smallest_budget=10,
        horizon=100,
        quality_endpoint=True,
    )

    assert any(policy.batch_size == 1 for policy in policies)
    assert policies
    assert all(policy.feedback_weight > 0.0 for policy in policies)
    assert all(policy.feedback_model == "hierarchical_utility" for policy in policies)
    assert all(policy.feedback_projection == "all_factors" for policy in policies)


def test_brain_age_quality_grid_compares_stable_feedback_models() -> None:
    policies = dynamic_policy_grid(
        [NeuroDiscoveryPolicy(task="brain_age")],
        smallest_budget=10,
        horizon=100,
        quality_endpoint=True,
    )

    assert policies
    assert {policy.batch_size for policy in policies} == {1}
    assert {policy.feedback_model for policy in policies} == {
        "factor_ucb_ridge_surrogate",
        "ridge_surrogate",
    }
    assert {policy.warmup_batches for policy in policies} == {1}


def test_recovery_objective_reports_registered_recall_costs() -> None:
    objective, metrics = recovery_objective(
        np.arange(10),
        np.asarray([False, True, False, True, False, True, False, True, False, True]),
        np.ones(10, dtype=bool),
        (2, 5, 10),
        filter_order=False,
    )

    assert objective > 0.0
    assert metrics["experiments_for_recall_10"] == 2
    assert metrics["experiments_for_recall_50"] == 6
    assert metrics["experiments_for_recall_100"] == 10
    assert metrics["recall_efficiency_50"] == pytest.approx(0.5)


def test_connectome_static_grid_registers_only_weak_design_weights() -> None:
    policies = static_policy_grid("connectome_behavior", design_prior_available=True)

    assert {policy.design_prior_weight for policy in policies} == {
        0.0,
        0.05,
        0.10,
        0.15,
    }


def test_static_control_disables_all_adaptive_ranking() -> None:
    control = static_control_policy(NeuroDiscoveryPolicy(task="biomarker_discovery"))

    assert control.feedback_horizon == 1
    assert control.batch_size == 1
    assert control.sampling_temperature == 0.0
    assert control.feedback_weight == 0.0
    assert control.pair_feedback_weight == 0.0
    assert control.exploration_weight == 0.0


def test_large_pool_grid_is_compact_and_uses_rank_based_warmup() -> None:
    policies = dynamic_policy_grid(
        [NeuroDiscoveryPolicy(task="biomarker_discovery")],
        smallest_budget=100,
        horizon=200_000,
    )

    assert len(policies) == 5
    assert sum(policy.feedback_horizon == 1 for policy in policies) == 1
    adaptive = [policy for policy in policies if policy.feedback_horizon != 1]
    assert all(policy.feedback_batch_schedule == "front_loaded" for policy in adaptive)
    starts = {policy.warmup_batches for policy in adaptive}
    assert len(starts) >= 3
    assert (
        front_loaded_warmup_batches(
            horizon=200_000,
            batch_size=2,
            max_feedback_rounds=128,
            feedback_start_budget=5_000,
        )
        in starts
    )
