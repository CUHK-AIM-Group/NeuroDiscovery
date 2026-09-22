from __future__ import annotations

import numpy as np
import pandas as pd

from core.scripts.tune_shared_neurodiscovery_policy import (
    aggregate_profiles,
    crossfit_feedback,
    delayed_feedback_profiles,
    default_profiles,
    leave_one_task_out,
    materialize_policy,
    profiles_for_slate,
)


def test_default_shared_profiles_are_small_valid_and_unique() -> None:
    profiles = default_profiles()
    assert 2 <= len(profiles) <= 10
    assert len({profile.profile_id() for profile in profiles}) == len(profiles)
    for profile in profiles:
        profile.validate()


def test_delayed_feedback_profiles_are_small_valid_and_unique() -> None:
    profiles = delayed_feedback_profiles()
    assert 2 <= len(profiles) <= 10
    assert len({profile.profile_id() for profile in profiles}) == len(profiles)
    assert profiles_for_slate("v2") == default_profiles()
    assert profiles_for_slate("v3") == profiles
    for profile in profiles:
        profile.validate()


def test_shared_profile_maps_to_candidate_scale_without_changing_weights() -> None:
    profile = next(item for item in default_profiles() if item.name == "relation_scoped_pair")
    small = materialize_policy(
        profile,
        task="small_task",
        candidate_count=500,
        smallest_budget=10,
        horizon=250,
    )
    large = materialize_policy(
        profile,
        task="large_task",
        candidate_count=500_000,
        smallest_budget=5_000,
        horizon=200_000,
    )
    expected_weights = (0.04, 0.06, 0.36, 0.54)
    for policy in (small, large):
        actual = (
            policy.global_node_weight,
            policy.global_pair_weight,
            policy.scoped_node_weight,
            policy.scoped_pair_weight,
        )
        assert np.allclose(actual, expected_weights)
        assert policy.feedback_projection == "relation_endpoints"
    assert small.feedback_batch_schedule == "fixed"
    assert small.batch_size == 3
    assert small.warmup_batches == 4
    assert large.feedback_batch_schedule == "front_loaded"
    assert large.batch_size == 2
    assert large.warmup_batches > 1


def test_small_pool_delayed_feedback_honours_start_fraction() -> None:
    profile = next(
        item
        for item in delayed_feedback_profiles()
        if item.name == "relation_scoped_pair_weak_late"
    )
    policy = materialize_policy(
        profile,
        task="small_task",
        candidate_count=500,
        smallest_budget=10,
        horizon=250,
    )
    assert policy.batch_size == 3
    assert policy.warmup_batches == 17
    assert policy.warmup_batches * policy.batch_size >= 50


def test_small_pool_reserves_a_feedback_active_round() -> None:
    profile = next(
        item
        for item in delayed_feedback_profiles()
        if item.name == "relation_scoped_pair_weak_late"
    )
    policy = materialize_policy(
        profile,
        task="progression_prediction",
        candidate_count=48,
        smallest_budget=25,
        horizon=25,
    )
    planned_rounds = int(np.ceil(policy.feedback_horizon / policy.batch_size))

    assert planned_rounds == 4
    assert policy.warmup_batches == 3
    assert policy.warmup_batches < planned_rounds
    assert policy.preserve_static_until_informative_feedback is True


def test_leave_one_task_out_selection_excludes_held_task() -> None:
    rows = []
    # Profile A wins tasks 1 and 2; profile B wins only task 3 by a large margin.
    for profile, values in {
        "A": {"task1": 1.0, "task2": 1.0, "task3": 0.1},
        "B": {"task1": 0.5, "task2": 0.5, "task3": 1.0},
    }.items():
        for task, relative in values.items():
            rows.append(
                {
                    "profile": profile,
                    "profile_id": profile.lower(),
                    "task": task,
                    "crossfit_relative": relative,
                    "outer_holdout_mean": relative / 2.0,
                }
            )
    result = leave_one_task_out(pd.DataFrame(rows), ("task1", "task2", "task3"))
    selected_for_task3 = result.loc[result["held_out_task"] == "task3"].iloc[0]
    assert selected_for_task3["selected_profile"] == "A"
    assert selected_for_task3["held_out_crossfit_relative"] == 0.1


def test_crossfit_feedback_masks_evaluation_and_outer_folds() -> None:
    internal = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "validated": [True, True, True, True, True],
            "feedback_status": ["supported"] * 5,
        }
    )
    folds = np.arange(5)

    inner, inner_masked = crossfit_feedback(
        internal,
        folds,
        evaluation_fold=1,
        outer_holdout_fold=4,
    )
    outer, outer_masked = crossfit_feedback(
        internal,
        folds,
        evaluation_fold=4,
        outer_holdout_fold=4,
    )

    assert inner_masked == (1, 4)
    assert inner["validated"].tolist() == [True, False, True, True, False]
    assert inner["feedback_status"].tolist() == [
        "supported",
        "inconclusive",
        "supported",
        "supported",
        "inconclusive",
    ]
    assert outer_masked == (4,)
    assert outer["validated"].tolist() == [True, True, True, True, False]


def test_profile_selection_ignores_outer_holdout_objective() -> None:
    rows = []
    for task in ("task1", "task2"):
        for profile, inner_value, outer_value in (
            ("inner_winner", 0.8, 0.1),
            ("outer_winner", 0.4, 1.0),
        ):
            for fold in (0, 1):
                rows.append(
                    {
                        "task": task,
                        "profile": profile,
                        "profile_id": profile,
                        "evaluation_role": "inner_crossfit",
                        "evaluation_fold": fold,
                        "objective": inner_value,
                    }
                )
            rows.append(
                {
                    "task": task,
                    "profile": profile,
                    "profile_id": profile,
                    "evaluation_role": "outer_holdout",
                    "evaluation_fold": 4,
                    "objective": outer_value,
                }
            )

    _, profiles = aggregate_profiles(pd.DataFrame(rows), ("task1", "task2"))

    assert profiles.iloc[0]["profile"] == "inner_winner"
    assert profiles.iloc[0]["outer_holdout_mean"] == 0.1
