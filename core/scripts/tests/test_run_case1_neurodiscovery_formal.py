from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.scripts import run_case1_neurodiscovery_formal as formal
from core.scripts.run_case1_neurodiscovery_formal import (
    FormalOutcomeVault,
    metrics_from_order,
)


def commitment_payload(batch: int, candidate_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": "case1-neurodiscovery-batch-selection.v1",
        "seed": 0,
        "trial": 0,
        "batch": batch,
        "stage": "closed_loop",
        "start_rank": batch * len(candidate_ids) + 1,
        "end_rank": (batch + 1) * len(candidate_ids),
        "candidate_ids": candidate_ids,
        "overlay_records_available": batch * len(candidate_ids),
        "overlay_informative_records": batch,
        "overlay_score_nonzero": batch > 0,
        "selection_changed_by_overlay": batch > 0,
        "outcomes_read_before_commit": False,
    }


def test_formal_vault_commits_before_exact_reveal_and_chains_batches(
    tmp_path: Path,
) -> None:
    feedback = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "execution_succeeded": [True, True, False],
            "adjusted_residual_d": [0.4, 0.05, np.nan],
            "p_value": [0.001, 0.5, np.nan],
            "expected_direction": ["", "", ""],
        }
    )
    vault = FormalOutcomeVault(
        feedback,
        seed_dir=tmp_path,
        seed=0,
        public_sha256="public",
        feedback_sha256="feedback",
        config_sha256="config",
    )

    first = vault.commit(commitment_payload(0, ["a", "b"]))
    revealed = vault.reveal(["a", "b"], first)
    second = vault.commit(commitment_payload(1, ["c"]))
    vault.reveal(["c"], second)

    assert [row["feedback_status"] for row in revealed] == [
        "supported",
        "inconclusive",
    ]
    assert all(row["feedback_available"] is True for row in revealed)
    assert second["previous_commit_sha256"] == first["commit_sha256"]
    assert vault.manifest() == {
        "commit_count": 2,
        "reveal_count": 2,
        "all_commits_revealed": True,
        "final_commit_chain_hash": second["commit_sha256"],
        "gt_fields_read": False,
    }
    assert (tmp_path / "batches/batch_0000/selection_commitment.json").is_file()
    assert (tmp_path / "batches/batch_0000/experimental_feedback.json").is_file()
    assert (tmp_path / "batches/batch_0000/reveal_audit.json").is_file()


def test_formal_vault_rejects_noncommitted_candidate(tmp_path: Path) -> None:
    feedback = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "execution_succeeded": [True, True],
            "adjusted_residual_d": [0.4, 0.3],
            "p_value": [0.001, 0.001],
            "expected_direction": ["", ""],
        }
    )
    vault = FormalOutcomeVault(
        feedback,
        seed_dir=tmp_path,
        seed=0,
        public_sha256="public",
        feedback_sha256="feedback",
        config_sha256="config",
    )
    commitment = vault.commit(commitment_payload(0, ["a"]))

    with pytest.raises(RuntimeError, match="does not match committed"):
        vault.reveal(["b"], commitment)


def test_metrics_from_frozen_order_uses_registered_budgets_and_targets() -> None:
    n = 200_000
    order = np.arange(n, dtype=np.int64)
    gt = np.zeros(n, dtype=bool)
    gt[:4_263] = True
    strict = np.zeros(n, dtype=bool)
    strict[0] = True

    curves, costs = metrics_from_order(
        order,
        gt,
        strict,
        method="neurodiscovery",
        seed=0,
    )

    assert [row["budget"] for row in curves] == [
        5_000,
        10_000,
        50_000,
        100_000,
        200_000,
    ]
    assert curves[0]["gt_hits"] == 4_263
    assert curves[0]["strict_fdr_hits"] == 1
    assert [row["experiments_required"] for row in costs] == [
        43,
        214,
        427,
        853,
        1_279,
        2_132,
    ]


def test_exact_sign_test_and_holm_adjust_are_small_seed_honest() -> None:
    assert formal.exact_positive_sign_test(np.array([3.0, 2.0, 1.0])) == 0.125
    assert formal.exact_positive_sign_test(np.array([3.0, 2.0, -1.0])) == 0.5
    assert formal.exact_positive_sign_test(np.array([0.0, 0.0, 0.0])) == 1.0
    adjusted = formal.holm_adjust([0.01, 0.03, 0.02])
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_evaluate_audits_frozen_seeds_before_opening_gt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public_path = tmp_path / "public.csv"
    gt_path = tmp_path / "gt.csv"
    (tmp_path / "input_manifest.json").write_text(
        """
        {
          "artifacts": {
            "public_candidates": {"path": "public.csv", "sha256": "public"},
            "gt_labels": {"path": "gt.csv", "sha256": "gt"}
          }
        }
        """,
        encoding="utf-8",
    )
    opened: list[str] = []

    def fake_verify(record):
        return tmp_path / str(record["path"])

    def fake_read_csv(path, *args, **kwargs):
        del args, kwargs
        name = Path(path).name
        opened.append(name)
        if Path(path) == public_path:
            return pd.DataFrame({"candidate_id": ["a"]})
        raise AssertionError("GT was opened before the frozen-seed audit")

    def fail_audit(root, seed):
        del root, seed
        raise RuntimeError("seed audit failed first")

    monkeypatch.setattr(formal, "verify_artifact", fake_verify)
    monkeypatch.setattr(formal.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(formal, "audit_seed", fail_audit)

    with pytest.raises(RuntimeError, match="seed audit failed first"):
        formal.evaluate(
            SimpleNamespace(
                output_root=tmp_path,
                seeds=[0, 1, 2],
                baseline_policies=tmp_path / "policies.jsonl",
            )
        )

    assert opened == [public_path.name]
    assert gt_path.name not in opened


def _write_tiny_complete_formal_seed(
    tmp_path: Path,
    *,
    ranking: np.ndarray,
) -> None:
    public = pd.DataFrame({"candidate_id": ["a", "b"]})
    feedback = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "execution_succeeded": [False, False],
            "adjusted_residual_d": [np.nan, np.nan],
            "p_value": [np.nan, np.nan],
            "expected_direction": ["", ""],
        }
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    public_path = inputs / "public.csv"
    feedback_path = inputs / "feedback.csv"
    all_tests_path = inputs / "all_tests.csv"
    public.to_csv(public_path, index=False)
    feedback.to_csv(feedback_path, index=False)
    all_tests_path.write_text("candidate_id\na\nb\n", encoding="utf-8")
    input_manifest = {
        "schema_version": formal.INPUT_SCHEMA,
        "candidate_count": 2,
        "candidate_id_sha256": formal.candidate_id_sha256(public["candidate_id"]),
        "candidate_order_sha256": formal.candidate_order_sha256(["a", "b"]),
        "artifacts": {
            "public_candidates": formal.artifact(public_path),
            "feedback_outcomes": formal.artifact(feedback_path),
        },
        "source_files": {
            "all_tests": formal.artifact(all_tests_path),
        },
    }
    formal.write_json(tmp_path / "input_manifest.json", input_manifest)

    config = formal.Case1NeuroDiscoveryConfig(
        batch_size=1,
        warmup_budget=1,
        max_closed_loop_budget=2,
    )
    config_path = tmp_path / "frozen_config.json"
    config.write_json(config_path)
    source_config_path = tmp_path / "best_config.json"
    config.write_json(source_config_path)
    tuning_manifest_path = tmp_path / "tuning_manifest.json"
    formal.write_json(
        tuning_manifest_path,
        {
            "schema_version": "case1-neurodiscovery-tuning.v1",
            "selected_formal_config": config.to_dict(),
            "external_validation_used_for_tuning": False,
            "outer_holdout_effects_hidden_until_config_freeze": True,
            "feedback_masking_contract": {
                "hidden_feedback_available": False,
                "hidden_rows_update_factor_or_pair_feedback": False,
            },
            "all_tests_sha256": formal.sha256_file(all_tests_path),
            "score_components": {
                "candidate_id_sha256": input_manifest["candidate_id_sha256"],
            },
            "code_provenance": {},
            "artifacts": {
                "best_config.json": formal.artifact(source_config_path),
            },
        },
    )
    seed_dir = tmp_path / "neurodiscovery" / "seed_00"
    seed_dir.mkdir(parents=True)
    vault = FormalOutcomeVault(
        feedback,
        seed_dir=seed_dir,
        seed=0,
        public_sha256=formal.sha256_file(public_path),
        feedback_sha256=formal.sha256_file(feedback_path),
        config_sha256=formal.sha256_file(config_path),
    )
    commits = []
    revealed_rows = []
    for batch, candidate_id in enumerate(("a", "b")):
        commit = vault.commit(commitment_payload(batch, [candidate_id]))
        commits.append(commit)
        revealed_rows.extend(vault.reveal([candidate_id], commit))

    ranking_path = seed_dir / "frozen_order_indices.npy"
    with ranking_path.open("wb") as handle:
        np.save(handle, ranking.astype(np.int32), allow_pickle=False)
    batch_audit_path = seed_dir / "batch_audit.csv"
    pd.DataFrame(
        {
            "selection_commit_sha256": [
                commit["commit_sha256"] for commit in commits
            ],
            "outcomes_revealed_after_commitment": [True, True],
            "batch_gt_hits": [np.nan, np.nan],
        }
    ).to_csv(batch_audit_path, index=False)

    overlay_path = seed_dir / "experimental_kg_overlay.jsonl"
    previous = "0" * 64
    overlay_records = []
    for row in revealed_rows:
        record = {
            "schema_version": "experimental-claim.v2",
            "case_study_id": formal.CASE_STUDY_ID,
            "seed": 0,
            "trial": 0,
            "candidate_id": row["candidate_id"],
            "status": row["feedback_status"],
            "previous_hash": previous,
        }
        record_hash = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        record["record_hash"] = record_hash
        previous = record_hash
        overlay_records.append(record)
    overlay_path.write_text(
        "".join(json.dumps(record) + "\n" for record in overlay_records),
        encoding="utf-8",
    )
    overlay_manifest_path = seed_dir / "experimental_kg_overlay.manifest.json"
    formal.write_json(
        overlay_manifest_path,
        {
            "seed": 0,
            "trial": 0,
            "path": str(overlay_path.resolve()),
            "sha256": formal.sha256_file(overlay_path),
            "records": 2,
            "records_by_status": {"execution_failed": 2},
            "final_chain_hash": previous,
            "feedback_consumed_during_ranking": True,
            "nonzero_feedback_reads": 1,
            "selection_changed_batches": 1,
            "mutates_formal_kg": False,
            "batch_selection_commits": {
                "count": 2,
                "outcome_reveal_count": 2,
                "committed_before_selected_outcome_lookup": True,
            },
            "formal_outcome_vault": {
                "enabled": True,
                "scoring_frame_outcome_blind": True,
                "outcomes_revealed_only_after_commitment": True,
            },
        },
    )
    protocol_dir = tmp_path / "protocol_snapshot"
    protocol_dir.mkdir()
    protocol_sources = {}
    for relative_path in formal.PROTOCOL_SOURCE_RELATIVE_PATHS:
        snapshot_path = protocol_dir / Path(relative_path).name
        snapshot_path.write_text(relative_path + "\n", encoding="utf-8")
        protocol_sources[relative_path] = {
            "source_path": relative_path,
            "source_sha256": formal.sha256_file(snapshot_path),
            "snapshot": formal.artifact(snapshot_path),
        }
    protocol_manifest_path = tmp_path / "protocol_snapshot_manifest.json"
    formal.write_json(
        protocol_manifest_path,
        {
            "schema_version": "case1-neurodiscovery-protocol.v1",
            "sources": protocol_sources,
        },
    )
    baseline_path = tmp_path / "baseline_policies.jsonl"
    baseline_path.write_text("{}\n", encoding="utf-8")
    formal_design_path = tmp_path / "formal_design.json"
    formal.write_json(
        formal_design_path,
        {
            "schema_version": "case1-neurodiscovery-formal-design.v1",
            "input_manifest_sha256": formal.sha256_file(
                tmp_path / "input_manifest.json"
            ),
            "candidate_id_sha256": input_manifest["candidate_id_sha256"],
            "formal_seeds": list(formal.FORMAL_SEEDS),
            "rng_seeds": {
                str(seed): formal.FORMAL_RNG_SEED_BASE + 1009 * seed
                for seed in formal.FORMAL_SEEDS
            },
            "budgets": [1, 2],
            "recall_targets": list(formal.RECALL_TARGETS),
            "config_artifact": formal.artifact(config_path),
            "source_config_artifact": formal.artifact(source_config_path),
            "tuning_manifest": formal.artifact(tuning_manifest_path),
            "protocol_snapshot_manifest": formal.artifact(
                protocol_manifest_path
            ),
            "baseline_policies": formal.artifact(baseline_path),
            "sota_definition": formal.SOTA_DEFINITION,
            "external_sota_definition": formal.EXTERNAL_SOTA_DEFINITION,
            "external_outcomes_available_to_ranking": False,
            "gt_available_to_seed_process": False,
        },
    )
    formal.write_json(
        seed_dir / "seed_manifest.json",
        {
            "schema_version": formal.SEED_SCHEMA,
            "status": "complete",
            "seed": 0,
            "trial": 0,
            "candidate_count": 2,
            "candidate_id_sha256": input_manifest["candidate_id_sha256"],
            "input_manifest_sha256": formal.sha256_file(
                tmp_path / "input_manifest.json"
            ),
            "config": config.to_dict(),
            "config_artifact": formal.artifact(config_path),
            "protocol_snapshot_manifest": formal.artifact(
                protocol_manifest_path
            ),
            "formal_design": formal.artifact(formal_design_path),
            "feedback_horizon": 2,
            "vault_audit": vault.manifest(),
            "closed_loop_integrity": {
                "public_scoring_frame_outcome_blind": True,
                "gt_labels_opened_by_seed_process": False,
                "feedback_consumed_during_ranking": True,
                "nonzero_feedback_reads": 1,
                "selection_changed_batches": 1,
                "commit_before_reveal": True,
            },
            "frozen_ranking": formal.artifact(ranking_path),
            "batch_audit": formal.artifact(batch_audit_path),
            "overlay_manifest": formal.artifact(overlay_manifest_path),
            "experimental_overlay": formal.artifact(overlay_path),
            "ranking_frozen_before_gt_evaluation": True,
        },
    )


def test_audit_seed_binds_frozen_ranking_to_commitment_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(formal, "BUDGETS", (1, 2))
    _write_tiny_complete_formal_seed(
        tmp_path,
        ranking=np.array([0, 1], dtype=np.int64),
    )

    audit = formal.audit_seed(tmp_path, 0)

    assert audit["status"] == "passed"
    assert audit["committed_prefix_records"] == 2
    assert audit["checks"]["ranking_prefix_matches_commitments"] is True


def test_audit_seed_rejects_ranking_not_matching_commitments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(formal, "BUDGETS", (1, 2))
    _write_tiny_complete_formal_seed(
        tmp_path,
        ranking=np.array([1, 0], dtype=np.int64),
    )

    with pytest.raises(RuntimeError, match="prefix differs from committed"):
        formal.audit_seed(tmp_path, 0)


def test_audit_seeds_is_gt_blind_and_labels_diagnostic_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    formal.write_json(
        tmp_path / "input_manifest.json",
        {"canonical_kg_release": {"kg_sha256": "latest-kg"}},
    )
    formal.write_json(
        tmp_path / "formal_design.json",
        {"baseline_policies": {"sha256": "old-policy"}},
    )

    def fake_audit_seed(_root: Path, seed: int) -> dict[str, object]:
        return {
            "seed": seed,
            "status": "passed",
            "checks": {"gt_not_opened": True},
            "frozen_ranking_sha256": f"ranking-{seed}",
            "config_sha256": "config",
            "protocol_snapshot_manifest_sha256": "protocol",
            "formal_design_sha256": "design",
        }

    monkeypatch.setattr(formal, "audit_seed", fake_audit_seed)
    output_path = tmp_path / "audit.json"
    result = formal.audit_seeds(
        SimpleNamespace(
            seeds=[0, 1, 2],
            output_root=tmp_path,
            audit_classification="diagnostic",
            audit_output=output_path,
        )
    )

    assert result["status"] == "passed"
    assert result["diagnostic_only"] is True
    assert result["performance_evaluated"] is False
    assert result["gt_artifact_resolved_or_opened_by_audit"] is False
    assert result["cross_seed_checks"]["frozen_rankings_distinct"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
