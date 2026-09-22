from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurooracle.scripts.case2_confirmatory_generator_metrics import (
    aggregate_trials,
    evaluate_orders,
    exact_paired_sign_flip_greater,
    normalized_discounted_cumulative_gain,
    normalized_recall_auc,
    paired_primary_comparisons,
)
from neurooracle.scripts.evaluate_case2_adni_confirmatory_generators import (
    _canonical_sha,
    _feedback_outcomes,
    _verify_existing_closed_loop_trial,
)
from neurooracle.scripts.run_case2_adni_confirmatory_mediation import (
    _holm_adjust_fixed_family,
    _one_sided_p,
    _weakest_link_evidence,
)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(6)],
            "family_fdr_chain_hit": [True, False, False, True, False, False],
            "global_fdr_chain_hit": [True, False, False, False, False, False],
            "nominal_chain_hit": [True, True, False, True, False, False],
        }
    )


def test_normalized_recall_auc_rewards_earlier_discovery() -> None:
    early = normalized_recall_auc(
        np.asarray([True, True, False, False]), [1, 2, 4]
    )
    late = normalized_recall_auc(
        np.asarray([False, False, True, True]), [1, 2, 4]
    )

    assert early > late
    assert early == pytest.approx(0.75)
    assert late == pytest.approx(0.25)


def test_ndcg_rewards_placing_strong_heldout_evidence_first() -> None:
    early = normalized_discounted_cumulative_gain(np.asarray([3.0, 1.0, 0.0]))
    late = normalized_discounted_cumulative_gain(np.asarray([0.0, 1.0, 3.0]))

    assert early == pytest.approx(1.0)
    assert early > late


def test_directional_one_sided_p_and_fixed_holm_are_prespecified() -> None:
    assert _one_sided_p(0.04, 0.2, 1) == pytest.approx(0.02)
    assert _one_sided_p(0.04, -0.2, 1) == pytest.approx(0.98)
    adjusted = _holm_adjust_fixed_family([0.001, 0.01, 0.02, np.nan])

    assert adjusted.tolist() == pytest.approx([0.004, 0.03, 0.04, 1.0])


def test_weakest_link_evidence_uses_largest_component_p() -> None:
    frame = pd.DataFrame(
        {
            "a_path_p": [0.001, 0.01, np.nan],
            "b_path_p": [0.01, 0.02, 0.01],
            "sobel_p": [0.001, 0.05, 0.01],
        }
    )

    score = _weakest_link_evidence(frame)

    assert score.tolist() == pytest.approx([2.0, -np.log10(0.05), 0.0])


def test_order_evaluation_preserves_full_candidate_universe() -> None:
    results = _results().assign(
        directional_replication_hit=[True, False, False, False, False, False],
        heldout_chain_evidence_score=[3.0, 2.0, 1.0, 0.5, 0.0, 0.0],
    )
    orders = {
        ("neurodiscovery", 0): results["candidate_id"].tolist(),
        ("baseline", 0): list(reversed(results["candidate_id"].tolist())),
    }

    summary, budgets, rankings = evaluate_orders(
        orders,
        results,
        budgets=[1, 2, 6],
        recall_targets=[0.5, 1.0],
    )

    assert len(summary) == 2
    assert len(budgets) == 6
    assert len(rankings) == 12
    target = summary.loc[summary["method"].eq("neurodiscovery")].iloc[0]
    assert target["experiments_to_recall_0.5"] == 1
    assert target["experiments_to_recall_1"] == 4
    assert target["experiments_to_replication_recall_1"] == 1
    assert target["heldout_chain_evidence_ndcg"] == pytest.approx(1.0)


def test_phase_closed_loop_feedback_does_not_expose_private_statistics() -> None:
    results = _results().assign(
        executable=True,
        analysis_status="estimated",
        expected_a_sign=1,
        heldout_chain_evidence_score=3.0,
        sobel_p=0.001,
    )
    feedback = _feedback_outcomes(
        results,
        protocol={"cohort": {"holdout_axis": "cohort_phase"}},
    )

    assert feedback["validated"].tolist() == results["nominal_chain_hit"].tolist()
    assert "expected_a_sign" not in feedback.columns
    assert "heldout_chain_evidence_score" not in feedback.columns
    assert "sobel_p" not in feedback.columns


def test_exact_sign_flip_and_paired_sota_table() -> None:
    rows = []
    for trial in range(10):
        rows.append(
            {
                "method": "neurodiscovery",
                "trial": trial,
                "score": 0.8,
            }
        )
        rows.append({"method": "baseline", "trial": trial, "score": 0.2})
    frame = pd.DataFrame(rows)

    assert exact_paired_sign_flip_greater(np.ones(10)) == pytest.approx(1 / 1024)
    comparison = paired_primary_comparisons(
        frame,
        metric="score",
        target_method="neurodiscovery",
        baseline_methods=["baseline"],
        bootstrap_resamples=1000,
        bootstrap_seed=7,
    )

    assert comparison.iloc[0]["target_superior"]
    assert comparison.iloc[0]["paired_difference_ci95_low"] == pytest.approx(0.6)


def test_zero_positive_pool_preserves_unidentifiable_metric_schema() -> None:
    results = _results().assign(
        family_fdr_chain_hit=False,
        global_fdr_chain_hit=False,
    )
    orders = {
        ("neurodiscovery", trial): results["candidate_id"].tolist()
        for trial in range(10)
    }
    orders.update(
        {
            ("baseline", trial): list(reversed(results["candidate_id"].tolist()))
            for trial in range(10)
        }
    )
    summary, _, _ = evaluate_orders(
        orders,
        results,
        budgets=[1, 2, 6],
        recall_targets=[0.5, 1.0],
    )
    aggregate = aggregate_trials(summary)
    metric = "normalized_family_discovery_recall_auc"
    comparison = paired_primary_comparisons(
        summary,
        metric=metric,
        target_method="neurodiscovery",
        baseline_methods=["baseline"],
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )

    assert aggregate[f"{metric}_mean"].isna().all()
    assert aggregate[f"{metric}_sd"].isna().all()
    assert comparison.iloc[0]["n_pairs"] == 10
    assert comparison.iloc[0]["n_finite_pairs"] == 0
    assert np.isnan(comparison.iloc[0]["paired_mean_difference"])
    assert not comparison.iloc[0]["target_superior"]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_existing_closed_loop_commit_chain_is_recovered_without_reranking(
    tmp_path: Path,
) -> None:
    candidate_ids = ["c1", "c2"]
    initial_commit = "a" * 64
    selection = {
        "trial": 0,
        "batch": 0,
        "start_rank": 1,
        "end_rank": 2,
        "candidate_ids": candidate_ids,
    }
    commit = {
        "schema_version": "neurooracle.case2_batch_ranking_commit.v1",
        "protocol_freeze_id": "freeze",
        "committed_at_utc": "2026-08-16T00:00:00+00:00",
        "previous_commit_sha256": initial_commit,
        "selection": selection,
        "outcome_values_in_commit": False,
    }
    commit["commit_sha256"] = _canonical_sha(commit)
    trace = {
        "candidate_ids": candidate_ids,
        "selection_commit": {
            "commit_sha256": commit["commit_sha256"],
            "commit_index": 1,
            "durably_flushed_before_outcome_lookup": True,
        },
    }
    overlays: list[dict[str, object]] = []
    previous_hash = "0" * 64
    for candidate_id in candidate_ids:
        overlay: dict[str, object] = {
            "candidate_id": candidate_id,
            "status": "inconclusive",
            "provenance": {
                "formal_kg_mutated": False,
                "outcome_observed_after_selection": True,
            },
            "previous_hash": previous_hash,
        }
        overlay["record_hash"] = _canonical_sha(overlay)
        previous_hash = str(overlay["record_hash"])
        overlays.append(overlay)

    _write_jsonl(tmp_path / "batch_ranking_commits.jsonl", [commit])
    _write_jsonl(tmp_path / "trace.jsonl", [trace])
    _write_jsonl(tmp_path / "experimental_overlay.jsonl", overlays)
    recovered, commit_manifest, overlay_manifest = (
        _verify_existing_closed_loop_trial(
            tmp_path,
            trial=0,
            candidate_ids=candidate_ids,
            protocol_freeze_id="freeze",
            initial_ranking_commit=initial_commit,
        )
    )

    assert recovered == candidate_ids
    assert commit_manifest["recovered_from_preexisting_pre_reveal_commits"]
    assert overlay_manifest["records"] == 2


# Last Updated At: 2026-08-16 13:58 HKT
