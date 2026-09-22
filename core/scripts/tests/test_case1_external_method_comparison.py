from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from core.scripts import case1_external_method_comparison as external
from core.scripts.case1_neurodiscovery_config import Case1NeuroDiscoveryConfig


def test_cli_allows_default_ranking_components(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "case1_external_method_comparison.py",
            "--all-tests",
            "all_tests.csv",
            "--kg",
            "knowledge_graph.json",
            "--claims",
            "extracted_claims.jsonl",
            "--current-state",
            "CURRENT_STATE.json",
            "--generation-first-dir",
            "official_baselines",
            "--external-root",
            "external",
            "--out-dir",
            "out",
        ],
    )
    args = external.parse_args()
    assert args.neurodiscovery_config is None
    assert args.score_components is None
    assert args.score_components_manifest is None
    assert args.neurodiscovery_overlay_dir is None
    assert tuple(args.budgets) == (5_000, 10_000, 50_000, 100_000, 200_000)
    assert tuple(args.recall_targets) == (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)


def test_reconstruct_orders_uses_frozen_neurodiscovery_config(monkeypatch) -> None:
    config = Case1NeuroDiscoveryConfig(
        warmup_budget=1_250,
        feedback_weight=0.17,
    )
    captured: dict[str, Case1NeuroDiscoveryConfig] = {}

    def fake_closed_loop(scored, rng, *, config):
        captured["config"] = config
        return np.array([1, 0], dtype=int)

    monkeypatch.setattr(
        external.comparison,
        "GENERATOR_METHODS",
        ("neurodiscovery",),
    )
    monkeypatch.setattr(
        external.comparison,
        "closed_loop_neurodiscovery_order",
        fake_closed_loop,
    )

    orders = list(
        external.reconstruct_orders(
            pd.DataFrame({"candidate_id": ["a", "b"]}),
            {},
            n_trials=1,
            seed=7,
            neurodiscovery_config=config,
        )
    )

    assert captured["config"] == config
    assert orders[0][2].tolist() == [1, 0]


def test_verified_overlay_recovers_full_neurodiscovery_order(tmp_path) -> None:
    scored = pd.DataFrame({"candidate_id": ["a", "b"]})
    config = Case1NeuroDiscoveryConfig(
        batch_size=2,
        warmup_budget=2,
        max_closed_loop_budget=2,
    )
    overlay = tmp_path / "seed_7_trial_00.jsonl"
    previous_hash = "0" * 64
    records = []
    for candidate_id in ("b", "a"):
        record = {
            "schema_version": "experimental-claim.v2",
            "case_study_id": external.comparison.CASE1_CASE_STUDY_ID,
            "seed": 7,
            "trial": 0,
            "round": 0,
            "candidate_id": candidate_id,
            "status": "supported",
            "previous_hash": previous_hash,
        }
        record_hash = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        record["record_hash"] = record_hash
        previous_hash = record_hash
        records.append(record)
    overlay.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    order, audit = external.load_neurodiscovery_overlay_order(
        scored,
        tmp_path,
        seed=7,
        trial=0,
        rng=np.random.default_rng(7),
        config=config,
    )

    assert order.tolist() == [1, 0]
    assert audit["records"] == 2
    assert audit["final_chain_hash"] == previous_hash


def test_verified_overlay_rejects_incomplete_candidate_coverage(tmp_path) -> None:
    scored = pd.DataFrame({"candidate_id": ["a", "b"]})
    config = Case1NeuroDiscoveryConfig(
        batch_size=2,
        warmup_budget=2,
        max_closed_loop_budget=2,
    )
    record = {
        "schema_version": "experimental-claim.v2",
        "case_study_id": external.comparison.CASE1_CASE_STUDY_ID,
        "seed": 7,
        "trial": 0,
        "round": 0,
        "candidate_id": "a",
        "status": "supported",
        "previous_hash": "0" * 64,
    }
    record["record_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (tmp_path / "seed_7_trial_00.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected TCP feedback prefix"):
        external.load_neurodiscovery_overlay_order(
            scored,
            tmp_path,
            seed=7,
            trial=0,
            rng=np.random.default_rng(7),
            config=config,
        )


def test_verified_overlay_reconstructs_frozen_unobserved_tail(tmp_path) -> None:
    scored = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "score_neurodiscovery": [0.4, 0.3, 0.2, 0.1],
        }
    )
    config = Case1NeuroDiscoveryConfig(
        batch_size=1,
        warmup_budget=1,
        max_closed_loop_budget=2,
        feature_support_decay_budget=0,
    )
    previous_hash = "0" * 64
    records = []
    for round_index, candidate_id in enumerate(("b", "d")):
        record = {
            "schema_version": "experimental-claim.v2",
            "case_study_id": external.comparison.CASE1_CASE_STUDY_ID,
            "seed": 19,
            "trial": 0,
            "round": round_index,
            "candidate_id": candidate_id,
            "status": "supported",
            "previous_hash": previous_hash,
        }
        record_hash = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        record["record_hash"] = record_hash
        previous_hash = record_hash
        records.append(record)
    (tmp_path / "seed_19_trial_00.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    actual, audit = external.load_neurodiscovery_overlay_order(
        scored,
        tmp_path,
        seed=19,
        trial=0,
        rng=np.random.default_rng(19),
        config=config,
    )

    expected_rng = np.random.default_rng(19)
    for _ in range(2):
        expected_rng.normal(0.0, 1.0, size=4)
    tail_score = scored["score_neurodiscovery"].to_numpy(float) + expected_rng.normal(
        0.0, 0.005, size=4
    )
    remaining = np.array([0, 2])
    expected_tail = remaining[
        np.lexsort(
            (
                scored["candidate_id"].to_numpy()[remaining],
                -tail_score[remaining],
            )
        )
    ]
    assert actual.tolist() == [1, 3, *expected_tail.tolist()]
    assert audit["feedback_prefix_records"] == 2
    assert audit["tail_records"] == 2


def _write_formal_neurodiscovery_seed(
    tmp_path,
    *,
    config: Case1NeuroDiscoveryConfig,
    gt_opened: bool = False,
) -> None:
    candidate_ids = ["a", "b"]
    (tmp_path / "input_manifest.json").write_text(
        json.dumps(
            {
                "candidate_order_sha256": hashlib.sha256(
                    "\n".join(candidate_ids).encode("utf-8")
                ).hexdigest()
            }
        ),
        encoding="utf-8",
    )
    seed_dir = tmp_path / "neurodiscovery" / "seed_00"
    seed_dir.mkdir(parents=True)
    ranking_path = seed_dir / "frozen_order_indices.npy"
    with ranking_path.open("wb") as handle:
        np.save(handle, np.array([1, 0], dtype=np.int64), allow_pickle=False)
    seed_manifest = {
        "schema_version": "case1-neurodiscovery-formal-seed.v1",
        "status": "complete",
        "seed": 0,
        "config": config.to_dict(),
        "closed_loop_integrity": {
            "public_scoring_frame_outcome_blind": True,
            "gt_labels_opened_by_seed_process": gt_opened,
            "feedback_consumed_during_ranking": True,
            "nonzero_feedback_reads": 1,
            "selection_changed_batches": 1,
            "commit_before_reveal": True,
        },
        "ranking_frozen_before_gt_evaluation": True,
        "frozen_ranking": {
            "path": str(ranking_path.resolve()),
            "sha256": external.sha256_file(ranking_path),
        },
    }
    (seed_dir / "seed_manifest.json").write_text(
        json.dumps(seed_manifest),
        encoding="utf-8",
    )


def test_formal_neurodiscovery_order_loads_only_frozen_audited_ranking(
    tmp_path,
    monkeypatch,
) -> None:
    config = Case1NeuroDiscoveryConfig(
        batch_size=2,
        warmup_budget=2,
        max_closed_loop_budget=2,
    )
    _write_formal_neurodiscovery_seed(tmp_path, config=config)
    monkeypatch.setattr(
        external,
        "audit_formal_neurodiscovery_seed",
        lambda root, seed: {"status": "passed", "seed": seed},
    )

    order, audit = external.load_formal_neurodiscovery_order(
        pd.DataFrame({"candidate_id": ["a", "b"]}),
        tmp_path,
        trial=0,
        config=config,
    )

    assert order.tolist() == [1, 0]
    assert audit["closed_loop_integrity"] == {
        "outcome_blind_scoring": True,
        "gt_not_opened": True,
        "feedback_consumed": True,
        "nonzero_feedback": True,
        "selection_changed": True,
        "commit_before_reveal": True,
        "ranking_frozen_before_gt": True,
    }
    assert audit["formal_seed_audit"] == {"status": "passed", "seed": 0}


def test_formal_neurodiscovery_order_rejects_seed_that_opened_gt(tmp_path) -> None:
    config = Case1NeuroDiscoveryConfig(
        batch_size=2,
        warmup_budget=2,
        max_closed_loop_budget=2,
    )
    _write_formal_neurodiscovery_seed(
        tmp_path,
        config=config,
        gt_opened=True,
    )

    with pytest.raises(ValueError, match="failed integrity"):
        external.load_formal_neurodiscovery_order(
            pd.DataFrame({"candidate_id": ["a", "b"]}),
            tmp_path,
            trial=0,
            config=config,
        )


def test_frozen_rankings_reject_mismatched_provenance(tmp_path) -> None:
    candidate_ids = np.array(["a", "b"])
    candidate_hash = hashlib.sha256(b"a\nb").hexdigest()
    manifest = {
        "candidate_id_sha256": candidate_hash,
        "ranking_provenance": {"kg": "old"},
    }
    (tmp_path / "frozen_tcp_rankings_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ranking provenance"):
        external.load_frozen_tcp_rankings(
            tmp_path,
            candidate_ids,
            expected_ranking_provenance={"kg": "new"},
        )


def test_payload_hash_is_key_order_independent() -> None:
    assert external.sha256_payload({"a": 1, "b": 2}) == external.sha256_payload(
        {"b": 2, "a": 1}
    )


def test_internal_sota_evidence_requires_hashed_complete_three_seed_gate(
    tmp_path,
) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    gate_path = evaluation / "sota_gate.json"
    gate_path.write_text(
        json.dumps({"sota_achieved": True}),
        encoding="utf-8",
    )
    manifest_path = evaluation / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "case1-neurodiscovery-formal-evaluation.v1",
                "status": "complete_sota",
                "seeds": [0, 1, 2],
                "ranking_freeze_preceded_gt_load": True,
                "artifacts": {
                    "sota_gate.json": {
                        "path": str(gate_path.resolve()),
                        "sha256": external.sha256_file(gate_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = external.load_internal_sota_evidence(tmp_path)

    assert evidence["internal_sota_gate"]["sota_achieved"] is True
    assert evidence["evaluation_manifest"]["sha256"] == external.sha256_file(
        manifest_path
    )


def test_external_sota_gate_requires_nd_to_win_every_registered_endpoint() -> None:
    methods = sorted(external.PRIMARY_METHODS)
    metric_summary = pd.DataFrame(
        [
            {
                "dataset": "pooled",
                "scope": "tcp_budget",
                "budget": 5_000,
                "method": method,
                "n_trials": 3,
                "n_confirmed_mean": 10.0 if method == "neurodiscovery" else 5.0,
            }
            for method in methods
        ]
    )
    recall_summary = pd.DataFrame(
        [
            {
                "dataset": "pooled",
                "recall_target": 0.1,
                "method": method,
                "n_trials": 3,
                "tcp_experiments_required_mean": (
                    100.0 if method == "neurodiscovery" else 200.0
                ),
            }
            for method in methods
        ]
    )

    gate = external.build_external_sota_gate(
        metric_summary,
        recall_summary,
        budgets=[5_000],
        recall_targets=[0.1],
    )

    assert gate["yield_gates"]["5000"]["passed"] is True
    assert gate["recall_cost_gates"]["0.10"]["passed"] is True
    assert gate["external_sota_achieved"] is True


def test_paired_p_values_deduplicates_overlapping_method_registries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        external.comparison,
        "PRIMARY_BASELINE_METHODS",
        ("brainpilot_native",),
    )
    monkeypatch.setattr(
        external,
        "NATIVE_METHOD_LABELS",
        {"brainpilot_native": "BrainPilot"},
    )
    metrics = pd.DataFrame(
        [
            {
                "dataset": "pooled",
                "scope": "tcp_budget",
                "budget": 100,
                "method": method,
                "trial": trial,
                "n_confirmed": confirmed,
            }
            for trial in (0, 1)
            for method, confirmed in (
                ("neurodiscovery", 3 + trial),
                ("brainpilot_native", 1 + trial),
            )
        ]
    )
    recall_costs = pd.DataFrame(
        [
            {
                "dataset": "pooled",
                "recall_target": 0.1,
                "method": method,
                "trial": trial,
                "tcp_experiments_required": cost,
            }
            for trial in (0, 1)
            for method, cost in (
                ("neurodiscovery", 10 + trial),
                ("brainpilot_native", 20 + trial),
            )
        ]
    )

    rows = external.paired_p_values(metrics, recall_costs)

    assert len(rows) == 2
    assert rows["baseline_method"].tolist() == [
        "brainpilot_native",
        "brainpilot_native",
    ]
