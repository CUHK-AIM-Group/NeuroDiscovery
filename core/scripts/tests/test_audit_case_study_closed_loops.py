from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.scripts.audit_case_study_closed_loops import _validated_ids, audit_task


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path, digest: str) -> dict[str, object]:
    return {"path": str(path), "sha256": digest, "rows": 1}


def _write_robustness_manifest(root: Path, task: str) -> None:
    artifact = root / "robustness_source.json"
    digest = _write(artifact, "{}\n")
    manifest = {
        "status": "complete",
        "task": task,
        "models": ["ridge", "bnt"],
        "sources": [
            {
                "name": "test sweep",
                "status": "complete",
                "failed_jobs": 0,
                "artifacts": {
                    "manifest": {"path": str(artifact), "sha256": digest}
                },
            }
        ],
    }
    _write(
        root / task / "model_robustness" / "manifest.json",
        json.dumps(manifest),
    )


def test_validated_ids_supports_boolean_csv_values(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.csv"
    _write(path, "candidate_id,validated\na,True\nb,false\nc,1\n")
    assert _validated_ids(path) == {"a", "c"}


def test_required_external_can_complete_with_a_negative_validation_result(tmp_path: Path) -> None:
    task = "brain_age"
    task_dir = tmp_path / task
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / "tables" / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    closure = {
        "status": "complete",
        "table_manifest": {
            "candidate_count": 1,
            "internal_validated": 1,
            "external_executable": 1,
            "external_validated": 0,
            "files": files,
            "outcome_isolation": {
                "public_contains_validation_columns": False,
                "external_read_required_during_generation": False,
                "public_internal_external_are_distinct_files": True,
            },
            "kg_scoring": {
                "kg_sha256": "kg",
                "semantic_fields": ["modality", "feature"],
                "matched_semantic_fields": 2,
            },
        },
        "benchmark_manifest": {
            "methods": ["random_walk", "neurodiscovery"],
            "method_trial_counts": {"random_walk": 10, "neurodiscovery": 10},
            "frozen_discovery": {"external_data_read_before_freeze": False},
            "external": {"loaded_after_freeze": True},
            "experimental_kg_delta": {
                "schema_version": "experimental-kg-overlay-bundle.v2",
                "records": 10,
                "final_chain_hash": "chain",
                "overlay_count": 10,
                "feedback_consumed_during_ranking": True,
                "semantic_projection_verified": True,
                "per_seed_isolation_verified": True,
                "mutates_formal_kg": False,
            },
            "formal_kg_mutated": False,
        },
    }
    _write(task_dir / "closure_manifest.json", json.dumps(closure))
    result = audit_task(
        tmp_path,
        task,
        current_kg_sha256="kg",
        verify_hashes=True,
    )
    assert result.complete
    assert result.internal_complete
    assert result.external_applicable
    assert result.external_complete
    assert result.external_validated == 0
    assert result.external_analysis_role == "required"
    assert any("external validation produced" in item for item in result.warnings)


def test_internal_only_task_does_not_require_external_execution(tmp_path: Path) -> None:
    task = "disease_subtyping"
    task_dir = tmp_path / task
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / "tables" / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    closure = {
        "status": "complete",
        "table_manifest": {
            "candidate_count": 1,
            "internal_validated": 1,
            "external_executable": 0,
            "external_validated": 0,
            "files": files,
            "outcome_isolation": {
                "public_contains_validation_columns": False,
                "external_read_required_during_generation": False,
                "public_internal_external_are_distinct_files": True,
            },
            "kg_scoring": {
                "kg_sha256": "kg",
                "semantic_fields": ["disease", "feature"],
                "matched_semantic_fields": 2,
            },
        },
        "benchmark_manifest": {
            "methods": ["random_walk", "neurodiscovery"],
            "method_trial_counts": {"random_walk": 10, "neurodiscovery": 10},
            "frozen_discovery": {"external_data_read_before_freeze": False},
            "external": {"loaded_after_freeze": False},
            "experimental_kg_delta": {
                "schema_version": "experimental-kg-overlay-bundle.v2",
                "records": 10,
                "final_chain_hash": "chain",
                "overlay_count": 10,
                "feedback_consumed_during_ranking": True,
                "semantic_projection_verified": True,
                "per_seed_isolation_verified": True,
                "mutates_formal_kg": False,
            },
            "formal_kg_mutated": False,
        },
    }
    _write(task_dir / "closure_manifest.json", json.dumps(closure))

    result = audit_task(tmp_path, task, current_kg_sha256="kg", verify_hashes=True)

    assert result.internal_complete
    assert not result.external_applicable
    assert result.external_complete
    assert result.complete
    assert result.external_execution_ok
    assert not any("external validation produced" in item for item in result.warnings)


def test_model_robustness_can_be_required(tmp_path: Path) -> None:
    task = "brain_age"
    task_dir = tmp_path / task
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / "tables" / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    closure = {
        "status": "complete",
        "table_manifest": {
            "candidate_count": 1,
            "external_executable": 1,
            "files": files,
            "outcome_isolation": {
                "public_contains_validation_columns": False,
                "external_read_required_during_generation": False,
                "public_internal_external_are_distinct_files": True,
            },
            "kg_scoring": {
                "kg_sha256": "kg",
                "semantic_fields": ["modality", "feature"],
                "matched_semantic_fields": 2,
            },
        },
        "benchmark_manifest": {
            "methods": ["random_walk", "neurodiscovery"],
            "method_trial_counts": {"random_walk": 10, "neurodiscovery": 10},
            "frozen_discovery": {"external_data_read_before_freeze": False},
            "external": {"loaded_after_freeze": True},
            "experimental_kg_delta": {
                "schema_version": "experimental-kg-overlay-bundle.v2",
                "records": 10,
                "final_chain_hash": "chain",
                "overlay_count": 10,
                "feedback_consumed_during_ranking": True,
                "semantic_projection_verified": True,
                "per_seed_isolation_verified": True,
                "mutates_formal_kg": False,
            },
            "formal_kg_mutated": False,
        },
    }
    _write(task_dir / "closure_manifest.json", json.dumps(closure))
    missing = audit_task(
        tmp_path,
        task,
        current_kg_sha256="kg",
        require_model_robustness=True,
    )
    assert not missing.complete
    _write_robustness_manifest(tmp_path, task)
    complete = audit_task(
        tmp_path,
        task,
        current_kg_sha256="kg",
        verify_hashes=True,
        require_model_robustness=True,
    )
    assert complete.complete
    assert complete.model_robustness_models == ["bnt", "ridge"]


def test_external_leakage_fails_generic_task(tmp_path: Path) -> None:
    task = "connectome_behavior"
    task_dir = tmp_path / task
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / "tables" / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    closure = {
        "status": "complete",
        "table_manifest": {
            "candidate_count": 1,
            "external_executable": 1,
            "files": files,
            "outcome_isolation": {
                "public_contains_validation_columns": False,
                "external_read_required_during_generation": False,
                "public_internal_external_are_distinct_files": True,
            },
            "kg_scoring": {
                "kg_sha256": "kg",
                "semantic_fields": ["modality", "feature"],
                "matched_semantic_fields": 2,
            },
        },
        "benchmark_manifest": {
            "methods": ["neurodiscovery"],
            "method_trial_counts": {"neurodiscovery": 10},
            "frozen_discovery": {"external_data_read_before_freeze": True},
            "external": {"loaded_after_freeze": True},
            "experimental_kg_delta": {
                "schema_version": "experimental-kg-overlay-bundle.v2",
                "records": 10,
                "final_chain_hash": "chain",
                "overlay_count": 10,
                "feedback_consumed_during_ranking": True,
                "semantic_projection_verified": True,
                "per_seed_isolation_verified": True,
                "mutates_formal_kg": False,
            },
            "formal_kg_mutated": False,
        },
    }
    _write(task_dir / "closure_manifest.json", json.dumps(closure))
    result = audit_task(tmp_path, task, current_kg_sha256="kg")
    assert not result.complete
    assert not result.rankings_frozen_before_external
    assert not result.outcome_blind_ranking_freeze_ok


def test_required_external_must_be_loaded_after_ranking_freeze(
    tmp_path: Path,
) -> None:
    task = "connectome_behavior"
    task_dir = tmp_path / task
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / "tables" / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    closure = {
        "status": "complete",
        "table_manifest": {
            "candidate_count": 1,
            "internal_validated": 1,
            "external_executable": 1,
            "files": files,
            "outcome_isolation": {
                "public_contains_validation_columns": False,
                "external_read_required_during_generation": False,
                "public_internal_external_are_distinct_files": True,
            },
            "kg_scoring": {
                "kg_sha256": "kg",
                "semantic_fields": ["phenotype", "feature"],
                "matched_semantic_fields": 2,
            },
        },
        "benchmark_manifest": {
            "methods": ["random_walk", "neurodiscovery"],
            "method_trial_counts": {"random_walk": 10, "neurodiscovery": 10},
            "frozen_discovery": {"external_data_read_before_freeze": False},
            "external": {"loaded_after_freeze": False},
            "experimental_kg_delta": {
                "schema_version": "experimental-kg-overlay-bundle.v2",
                "records": 10,
                "final_chain_hash": "chain",
                "overlay_count": 10,
                "feedback_consumed_during_ranking": True,
                "semantic_projection_verified": True,
                "per_seed_isolation_verified": True,
                "mutates_formal_kg": False,
            },
            "formal_kg_mutated": False,
        },
    }
    _write(task_dir / "closure_manifest.json", json.dumps(closure))

    result = audit_task(tmp_path, task, current_kg_sha256="kg", verify_hashes=True)

    assert result.internal_complete
    assert result.outcome_blind_ranking_freeze_ok
    assert not result.external_loaded_after_freeze
    assert not result.external_complete
    assert not result.complete


def test_legacy_biomarker_requires_explicit_feedback_chain(tmp_path: Path) -> None:
    task_dir = tmp_path / "biomarker_discovery" / "tables"
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    manifest = {
        "status": "complete",
        "candidate_count": 1,
        "external_executable": 1,
        "files": files,
        "outcome_isolation": {
            "public_contains_validation_columns": False,
            "external_read_required_during_generation": False,
            "public_internal_external_are_distinct_files": True,
        },
        "source_ranking_manifest": {
            "external_data_read_before_freeze": False,
            "orders": {
                "n_trials_by_method": {"random_walk": 10, "neurodiscovery": 10}
            },
            "inputs": {"kg": {"sha256": "kg"}},
        },
    }
    _write(task_dir / "biomarker_closure_manifest.json", json.dumps(manifest))
    result = audit_task(tmp_path, "biomarker_discovery", current_kg_sha256="kg")
    assert not result.complete
    assert result.trials_ok
    assert not result.feedback_chain_ok


def test_legacy_biomarker_passes_with_verified_feedback_chain(tmp_path: Path) -> None:
    task_dir = tmp_path / "biomarker_discovery" / "tables"
    files = {}
    for name in ("public_candidates", "internal_outcomes", "external_outcomes"):
        path = task_dir / f"{name}.csv"
        files[name] = _descriptor(path, _write(path, "candidate_id\nx\n"))
    manifest = {
        "status": "complete",
        "candidate_count": 1,
        "external_executable": 1,
        "files": files,
        "outcome_isolation": {
            "public_contains_validation_columns": False,
            "external_read_required_during_generation": False,
            "public_internal_external_are_distinct_files": True,
        },
        "source_ranking_manifest": {
            "external_data_read_before_freeze": False,
            "orders": {
                "n_trials_by_method": {"random_walk": 10, "neurodiscovery": 10}
            },
            "inputs": {"kg": {"sha256": "kg"}},
        },
        "experimental_kg_delta": {
            "records": 10,
            "final_chain_hash": "chain",
            "mutates_formal_kg": False,
            "source_rankings_verified": True,
            "batch_feedback_verified": True,
        },
        "formal_kg_mutated": False,
    }
    _write(task_dir / "biomarker_closure_manifest.json", json.dumps(manifest))
    result = audit_task(tmp_path, "biomarker_discovery", current_kg_sha256="kg")
    assert result.complete
