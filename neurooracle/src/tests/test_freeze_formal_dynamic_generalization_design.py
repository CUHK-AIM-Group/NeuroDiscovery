from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.freeze_formal_dynamic_generalization_design import (
    freeze_formal_dynamic_design,
)
from neurooracle.src.experiment_source_bundle import sha256_file


RUNTIME = {
    "profile": "legacy_closed_supported_gate",
    "proposal_batch_size": 600,
    "max_unique_proposals": 2400,
    "max_generation_rounds": 4,
    "max_stagnant_generation_rounds": 2,
    "max_executions": 300,
    "execution_batch_size": 10,
    "refresh_every_executions": 50,
    "refill_below": 200,
    "feedback_years": 1,
    "feedback_start_budget": 50,
    "min_supported_before_feedback": 2,
    "seed_diversity_fraction": 0.35,
    "candidate_pool_mode": "hybrid",
    "task_scope_fraction": 0.5,
    "evidence_frontier_fraction": 1.0,
    "protect_general_top_k": 0,
    "path_variants_per_endpoint": 2,
    "feedback_mutation_fraction": 0.35,
    "max_paths_per_endpoint": 4,
    "endpoint_canonical_quality_weight": 0.0,
    "kge_weight": 0.0,
}


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    canonical = {}
    hashes = {}
    for name in ("knowledge_graph", "extracted_claims", "current_state"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        canonical[name] = _record(path)
        hashes[name] = sha256_file(path)
    static = tmp_path / "static.json"
    static.write_text(
        json.dumps(
            {
                "status": "frozen_before_formal_generation",
                "canonical_release": canonical,
                "temporal_inputs": {
                    "snapshot_root": "snapshots",
                    "semantic_endpoint_identity_version": "v7",
                    "relation_contract_version": "v3",
                    "discovery_metric_contract_version": "v1",
                },
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "status": "frozen_for_confirmatory_application",
                "canonical_release": {
                    "knowledge_graph_sha256": hashes["knowledge_graph"],
                    "claims_sha256": hashes["extracted_claims"],
                    "state_sha256": hashes["current_state"],
                },
                "development_protocol": {
                    "case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ]
                },
                "selected_policy": {
                    "feedback_enabled": True,
                    **RUNTIME,
                    "selection_reason": "frozen before cross-task application",
                },
            }
        ),
        encoding="utf-8",
    )
    matrix = tmp_path / "dynamic.csv"
    matrix.write_text(
        "case_study_id,freeze_year,future_start_year,future_end_year,analysis_tier\n"
        "connectome_behavior,2018,2019,2023,primary\n"
        "differential_diagnosis,2019,2020,2024,primary\n",
        encoding="utf-8",
    )
    eligibility = tmp_path / "dynamic.json"
    eligibility.write_text(
        json.dumps(
            {
                "status": "frozen_before_dynamic_generalization",
                "feedback_years": 1,
                "selection_contract": {
                    "performance_columns_consumed": False,
                    "task_or_window_selection_after_method_scoring_permitted": False,
                    "excluded_case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ],
                },
                "primary_case_study_ids": [
                    "connectome_behavior",
                    "differential_diagnosis",
                ],
                "primary_windows": 2,
                "locked_matrix": {
                    "path": str(matrix),
                    "sha256": sha256_file(matrix),
                    "rows": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return static, policy, eligibility


def test_freeze_design_derives_matrix_and_preserves_evidence_tier(tmp_path: Path) -> None:
    static, policy, eligibility = _inputs(tmp_path)
    result = freeze_formal_dynamic_design(
        static_design_path=static,
        frozen_policy_path=policy,
        eligibility_manifest_path=eligibility,
        source_bundle_manifest=tmp_path / "future_bundle.json",
        output_path=tmp_path / "design.json",
        result_root=tmp_path / "results",
        workspace_root=tmp_path,
        seeds=[0, 1],
        budgets=[10, 20, 50, 100, 200, 300],
    )
    assert result["primary_matrix"]["case_studies"] == 2
    assert result["primary_matrix"]["case_study_windows"] == 2
    assert result["primary_matrix"]["paired_runs_per_arm"] == 4
    assert result["primary_matrix"]["claim_as_untouched_confirmatory_permitted"] is False
    assert result["shared_configuration"] == RUNTIME


def test_freeze_design_rejects_development_task_in_generalization_lock(
    tmp_path: Path,
) -> None:
    static, policy, eligibility = _inputs(tmp_path)
    matrix = tmp_path / "dynamic.csv"
    matrix.write_text(
        "case_study_id,freeze_year,future_start_year,future_end_year,analysis_tier\n"
        "case1_transdiagnostic,2018,2019,2023,primary\n",
        encoding="utf-8",
    )
    payload = json.loads(eligibility.read_text(encoding="utf-8"))
    payload["primary_case_study_ids"] = ["case1_transdiagnostic"]
    payload["primary_windows"] = 1
    payload["locked_matrix"] = {
        "path": str(matrix),
        "sha256": sha256_file(matrix),
        "rows": 1,
    }
    eligibility.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="development Case Studies entered"):
        freeze_formal_dynamic_design(
            static_design_path=static,
            frozen_policy_path=policy,
            eligibility_manifest_path=eligibility,
            source_bundle_manifest=tmp_path / "bundle.json",
            output_path=tmp_path / "design.json",
            result_root=tmp_path / "results",
            workspace_root=tmp_path,
        )


def test_freeze_design_accepts_verified_cross_release_policy_application(
    tmp_path: Path,
) -> None:
    static, source_policy, eligibility = _inputs(tmp_path)
    source = json.loads(source_policy.read_text(encoding="utf-8"))
    source["canonical_release"] = {
        "knowledge_graph_sha256": "OLD_KG",
        "claims_sha256": "OLD_CLAIMS",
        "state_sha256": "OLD_STATE",
    }
    source_policy.write_text(json.dumps(source), encoding="utf-8")

    canonical = json.loads(static.read_text(encoding="utf-8"))["canonical_release"]
    current_hashes = {
        "knowledge_graph_sha256": canonical["knowledge_graph"]["sha256"],
        "claims_sha256": canonical["extracted_claims"]["sha256"],
        "state_sha256": canonical["current_state"]["sha256"],
    }
    selected_payload = json.dumps(
        source["selected_policy"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    application_policy = tmp_path / "application_policy.json"
    application_policy.write_text(
        json.dumps(
            {
                "status": "frozen_for_confirmatory_application",
                "canonical_release": current_hashes,
                "development_protocol": source["development_protocol"],
                "selected_policy": source["selected_policy"],
                "policy_application": {
                    "mode": "unchanged_cross_release_application",
                    "source_policy": _record(source_policy),
                    "source_selected_policy_sha256": hashlib.sha256(
                        selected_payload
                    ).hexdigest().upper(),
                    "selection_canonical_release": source["canonical_release"],
                    "application_canonical_release": current_hashes,
                    "original_development_case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ],
                    "additional_exposed_case_study_ids": [],
                    "all_excluded_case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ],
                    "frozen_runtime_unchanged": True,
                    "new_release_outcomes_used_for_policy_selection": False,
                    "post_update_retuning": False,
                },
            }
        ),
        encoding="utf-8",
    )

    result = freeze_formal_dynamic_design(
        static_design_path=static,
        frozen_policy_path=application_policy,
        eligibility_manifest_path=eligibility,
        source_bundle_manifest=tmp_path / "future_bundle.json",
        output_path=tmp_path / "design.json",
        result_root=tmp_path / "results",
        workspace_root=tmp_path,
        seeds=[0, 1],
        budgets=[10, 20, 50, 100, 200, 300],
    )
    assert result["policy_application"]["mode"] == (
        "unchanged_cross_release_application"
    )
