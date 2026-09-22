from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurooracle.scripts.run_formal_dynamic_generalization import (
    _normalized_arm_command,
    _policy_runtime_configuration,
    _verify_design,
    build_commands,
)
from neurooracle.src.experiment_source_bundle import sha256_file


RUNTIME_CONFIG = {
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


def _write_design(tmp_path: Path) -> Path:
    canonical = {}
    for name in ("knowledge_graph", "extracted_claims", "current_state"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        canonical[name] = _record(path)

    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "case_study_id,freeze_year,future_start_year,future_end_year,analysis_tier\n"
        "connectome_behavior,2019,2020,2024,primary\n",
        encoding="utf-8",
    )
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text(
        json.dumps(
            {
                "status": "frozen_before_dynamic_generalization",
                "feedback_years": 1,
                "primary_windows": 1,
                "primary_case_study_ids": ["connectome_behavior"],
                "selection_contract": {
                    "performance_columns_consumed": False,
                    "task_or_window_selection_after_method_scoring_permitted": False,
                },
                "locked_matrix": {
                    "path": str(matrix),
                    "sha256": sha256_file(matrix),
                    "rows": 1,
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
                "development_protocol": {
                    "case_study_ids": [
                        "case1_transdiagnostic",
                        "biomarker_discovery",
                    ]
                },
                "selected_policy": {
                    "feedback_enabled": True,
                    **RUNTIME_CONFIG,
                    "selection_reason": "selected before cross-task application",
                },
            }
        ),
        encoding="utf-8",
    )
    design = tmp_path / "design.json"
    design.write_text(
        json.dumps(
            {
                "status": "frozen_before_dynamic_generalization",
                "canonical_release": canonical,
                "frozen_policy": _record(policy),
                "shared_configuration": RUNTIME_CONFIG,
                "fixed_budget_contract": {
                    "rank_points": [10, 20, 50, 100, 200, 300]
                },
                "dynamic_eligibility": {
                    "manifest": _record(eligibility),
                    "matrix": _record(matrix),
                },
                "development_case_study_ids": [
                    "case1_transdiagnostic",
                    "biomarker_discovery",
                ],
                "primary_matrix": {
                    "case_study_ids": ["connectome_behavior"],
                    "case_study_windows": 1,
                    "seeds": list(range(10)),
                    "paired_runs_per_arm": 10,
                    "previously_exposed_historical_years": True,
                    "claim_as_untouched_confirmatory_permitted": False,
                },
                "temporal_inputs": {"snapshot_root": str(tmp_path / "snapshots")},
                "endpoints": {
                    "primary": {"metric": "terminal_unique_primary_discoveries", "k": 300},
                    "metrics": ["terminal_unique_primary_discoveries"],
                },
            }
        ),
        encoding="utf-8",
    )
    return design


def test_policy_runtime_configuration_removes_only_arm_and_provenance_fields() -> None:
    policy = {
        "selected_policy": {
            "feedback_enabled": True,
            **RUNTIME_CONFIG,
            "selection_reason": "frozen",
        }
    }
    assert _policy_runtime_configuration(policy) == RUNTIME_CONFIG


def test_design_accepts_exact_frozen_policy_and_method_blind_lock(tmp_path: Path) -> None:
    design_path = _write_design(tmp_path)
    design, eligibility = _verify_design(design_path, tmp_path)
    assert design["shared_configuration"] == RUNTIME_CONFIG
    assert eligibility == tmp_path / "eligibility.json"


def test_design_rejects_runtime_policy_drift(tmp_path: Path) -> None:
    design_path = _write_design(tmp_path)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    design["shared_configuration"]["task_scope_fraction"] = 0.75
    design_path.write_text(json.dumps(design), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the frozen selected policy"):
        _verify_design(design_path, tmp_path)


def test_paired_commands_differ_only_by_feedback_and_output(tmp_path: Path) -> None:
    design_path = _write_design(tmp_path)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    commands = build_commands(
        python=Path("python.exe"),
        code_root=tmp_path / "archive",
        design=design,
        workspace_root=tmp_path,
        eligibility_manifest=tmp_path / "eligibility.json",
        bundle_manifest=tmp_path / "bundle.json",
        output_root=tmp_path / "out",
        force=False,
    )
    assert _normalized_arm_command(commands["closed"]) == _normalized_arm_command(
        commands["open"]
    )
    assert "--feedback-enabled" in commands["closed"]
    assert "--no-feedback-enabled" in commands["open"]
    budget_start = commands["closed"].index("--budgets") + 1
    profile_start = commands["closed"].index("--profile")
    assert commands["closed"][budget_start:profile_start] == [
        "10",
        "20",
        "50",
        "100",
        "200",
        "300",
    ]
