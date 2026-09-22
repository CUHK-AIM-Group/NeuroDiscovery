from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.scripts.freeze_cross_release_neurodiscovery_policies import freeze_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_freeze_bundle_selects_static_and_best_nonstatic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    kg = tmp_path / "kg.json"
    kg.write_text("{}", encoding="utf-8")
    import hashlib

    kg_sha = hashlib.sha256(b"{}").hexdigest()
    _write_json(
        source / "shared_tuning_manifest.json",
        {
            "status": "complete",
            "profile_slate": "v3",
            "folds": 5,
            "trials": 2,
            "selection_uses_external_outcomes": False,
            "selection_uses_outer_holdout": False,
            "selected_shared_profile": {
                "name": "relation_scoped_pair_static",
                "profile_id": "91aabfb3c315ea93",
            },
            "kg": {"path": "old.json", "sha256": "old-sha"},
        },
    )
    pd.DataFrame(
        [
            {
                "profile": "relation_scoped_pair_static",
                "profile_id": "91aabfb3c315ea93",
                "robust_cross_task_score": 0.95,
            },
            {
                "profile": "relation_scoped_pair_weak_late",
                "profile_id": "359b0fc8964424d9",
                "robust_cross_task_score": 0.70,
            },
            {
                "profile": "relation_balanced_pair_weak_late",
                "profile_id": "8523c832d0077ed8",
                "robust_cross_task_score": 0.60,
            },
        ]
    ).to_csv(source / "shared_profile_summary.csv", index=False)
    (source / "profile_trials.csv").write_text("profile\n", encoding="utf-8")

    benchmark = tmp_path / "benchmark"
    task = benchmark / "brain_age"
    public = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "kg_relation_global_node_support": [0.1, 0.2],
            "kg_relation_global_pair_support": [0.1, 0.2],
            "kg_relation_scoped_node_support": [0.1, 0.2],
            "kg_relation_scoped_pair_support": [0.1, 0.2],
        }
    )
    (task / "tables").mkdir(parents=True)
    public.to_csv(task / "tables" / "public_candidates.csv", index=False)
    _write_json(
        task / "tables" / "table_manifest.json",
        {
            "task": "brain_age",
            "kg_scoring": {"kg_sha256": kg_sha},
            "factor_fields": ["modality"],
        },
    )
    _write_json(
        task / "benchmark" / "run_manifest.json",
        {
            "task": "brain_age",
            "candidate_count": 2,
            "factor_fields": ["modality"],
            "methods": ["random_walk", "neurodiscovery"],
            "trials": 10,
            "budgets": [1, 2],
            "recall_targets": [0.1],
        },
    )
    output = tmp_path / "output"
    result = freeze_bundle(
        argparse.Namespace(
            source_tuning_root=source,
            target_benchmark_root=benchmark,
            target_kg=kg,
            expected_target_kg_sha256=kg_sha,
            output_dir=output,
            tasks=None,
        )
    )
    assert result["status"] == "frozen"
    assert result["tracks"]["closed_loop"]["profile"] == (
        "relation_scoped_pair_weak_late"
    )
    assert result["tasks"]["brain_age"]["comparison_horizon"] == 1
    assert (output / "policies" / "closed_loop" / "brain_age.json").is_file()
    assert result["selection_protocol"]["target_external_outcome_tables_opened"] is False
    assert result["implementation"]["materialization_revision"] == (
        "reserve-one-feedback-active-round-and-static-until-informative.v2"
    )
    assert result["implementation"]["outcome_labels_used_for_revision"] is False
    assert set(result["implementation"]["files"]) == {
        "freezer",
        "shared_policy_materializer",
        "policy_contract",
        "closed_loop_engine",
        "guard_evaluator",
        "benchmark_runner",
    }
    assert result["informative_feedback_guard_evaluation"] is None
    assert result["tasks"]["brain_age"]["policies"]["closed_loop"][
        "preserve_static_until_informative_feedback"
    ] is True
