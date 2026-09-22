from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from core.scripts.evaluate_cross_release_feedback_guard import (
    aggregate_trials,
    exact_sign_flip_p_value,
    guard_decision,
    implementation_record,
)
from core.scripts.recover_cross_release_feedback_guard import (
    COMMON_COLUMNS,
    recover_trial_rows,
)


def _trial_rows() -> pd.DataFrame:
    rows = []
    values = {
        "static_control": {"task_a": 0.70, "task_b": 0.70},
        "dynamic_original": {"task_a": 0.60, "task_b": 0.50},
        "dynamic_guarded": {"task_a": 0.70, "task_b": 0.60},
    }
    for variant, tasks in values.items():
        for task, objective in tasks.items():
            for fold in range(3):
                for trial in range(2):
                    rows.append(
                        {
                            "variant": variant,
                            "task": task,
                            "evaluation_role": (
                                "outer_holdout" if fold == 2 else "inner_crossfit"
                            ),
                            "evaluation_fold": fold,
                            "trial": trial,
                            "objective": (
                                0.0
                                if fold == 2 and variant == "dynamic_guarded"
                                else objective
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_guard_selection_uses_inner_crossfit_not_outer_holdout() -> None:
    task_summary, variant_summary = aggregate_trials(
        _trial_rows(),
        ("task_a", "task_b"),
    )
    decision = guard_decision(
        task_summary,
        variant_summary,
        ("task_a", "task_b"),
    )

    assert decision["accepted"] is True
    assert decision["outer_holdout_used_for_selection"] is False
    guarded = variant_summary.set_index("variant").loc["dynamic_guarded"]
    assert guarded["outer_holdout_mean"] == 0.0


def test_exact_sign_flip_p_value_is_one_sided_and_exact() -> None:
    assert exact_sign_flip_p_value([1.0, 1.0, 1.0]) == pytest.approx(1 / 8)
    assert exact_sign_flip_p_value([-1.0, -1.0, -1.0]) == pytest.approx(1.0)


def test_guard_evaluation_records_all_ranking_implementation_hashes() -> None:
    record = implementation_record()
    assert record["outcome_labels_used_for_revision"] is False
    assert set(record["files"]) == {
        "guard_evaluator",
        "closed_loop_engine",
        "policy_contract",
        "policy_materializer",
        "benchmark_contract",
    }
    assert all(len(item["sha256"]) == 64 for item in record["files"].values())


def test_legacy_variable_width_trials_are_recovered_by_task_budget(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.csv"
    common = [
        "dynamic_guarded",
        "task_a",
        "profile",
        "profile-id",
        "inner_crossfit",
        "0",
        "0;4",
        "0",
        "17",
        "policy-id",
        "True",
        "False",
        "0",
        "0",
        "a" * 64,
        "0.5",
        "2",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*COMMON_COLUMNS, "hits_at_10", "oracle_fraction_at_10"])
        writer.writerow([*common, "1", "0.5"])
        writer.writerow(
            [
                *common[:1],
                "task_b",
                *common[2:],
                "1",
                "0.5",
                "2",
                "1.0",
            ]
        )

    rows = recover_trial_rows(
        path,
        {
            "task_a": {"budgets": [10]},
            "task_b": {"budgets": [10, 25]},
        },
    )

    assert rows[0]["guard_enabled"] is True
    assert rows[0]["hits_at_10"] == 1
    assert rows[1]["hits_at_25"] == 2
