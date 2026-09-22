from __future__ import annotations

from pathlib import Path

from neurooracle.scripts.run_hindcasting_v3 import (
    _adaptive_feedback_start_budget,
    build_commands,
)


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _plan(tmp_path: Path) -> dict:
    windows = [
        {
            "freeze_year": year,
            "future_start_year": year + 1,
            "future_end_year": year + 5,
        }
        for year in range(2016, 2021)
    ]
    return {
        "cohort": {
            "task_ids": ["case1_transdiagnostic", "biomarker_discovery"]
        },
        "matrix": {
            "windows": windows,
            "seeds": list(range(10)),
            "experiment_counts": [10, 20, 50, 100, 200, 500, 1000],
        },
        "inputs": {
            "snapshot_root": str(tmp_path / "snapshots"),
            "kge_root": str(tmp_path / "kge"),
            "future_claims": {"path": str(tmp_path / "future.jsonl")},
        },
        "configuration": {
            "neurodiscovery": {
                "profile": "generator_composite_closed_supported_gate",
                "proposal_batch_size": 500,
                "max_unique_proposals": 4000,
                "max_generation_rounds": 8,
                "max_stagnant_generation_rounds": 2,
                "max_executions": 1000,
                "execution_batch_size": 10,
                "refresh_every_executions": 50,
                "refill_below": 200,
                "feedback_years": 2,
                "feedback_start_budget": 50,
                "min_supported_before_feedback": 1,
                "seed_diversity_fraction": 0.35,
                "candidate_pool_mode": "scoped",
                "task_scope_fraction": 1.0,
                "evidence_frontier_fraction": 1.0,
                "protect_general_top_k": 0,
                "path_variants_per_endpoint": 2,
                "feedback_mutation_fraction": 0.15,
                "max_paths_per_endpoint": 4,
                "endpoint_canonical_quality_weight": 0.2,
                "kge_weight": 0.15,
                "kge_checkpoint_pattern": "kg_{freeze_year}_complex_dim64_ep10.pt",
            },
            "baselines": {
                "target_per_case_study": 1000,
                "random_trials": 1000,
            },
        },
        "smoke": {
            "task_id": "case1_transdiagnostic",
            "window": windows[0],
            "seed": 0,
            "experiment_counts": [10, 20, 50, 100],
            "maximum_executions": 100,
        },
    }


def test_formal_commands_match_locked_matrix_and_budget(tmp_path: Path) -> None:
    commands = build_commands(
        plan=_plan(tmp_path),
        bundle_manifest=tmp_path / "bundle_manifest.json",
        python=tmp_path / "python",
        code_root=tmp_path / "code",
        mode="formal",
        output_root=tmp_path / "results",
        eligibility_manifest=tmp_path / "eligibility.json",
        force=False,
    )

    neuro = commands["neurodiscovery"]
    assert _value(neuro, "--max-executions") == "1000"
    assert _value(neuro, "--proposal-batch-size") == "500"
    assert _value(neuro, "--kge-weight") == "0.15"
    assert "--feedback-enabled" in neuro
    assert "--skip-source-bundle-reference-rehash" in neuro
    seed_start = neuro.index("--seeds") + 1
    assert neuro[seed_start : seed_start + 10] == [str(value) for value in range(10)]
    baseline = commands["baseline_generation"]
    assert _value(baseline, "--target-per-case-study") == "1000"
    assert "sciagents" in baseline and "openscholar_rag" in baseline


def test_smoke_commands_are_one_cell_and_not_formal_scale(tmp_path: Path) -> None:
    commands = build_commands(
        plan=_plan(tmp_path),
        bundle_manifest=tmp_path / "bundle_manifest.json",
        python=tmp_path / "python",
        code_root=tmp_path / "code",
        mode="smoke",
        output_root=tmp_path / "smoke",
        eligibility_manifest=tmp_path / "smoke_eligibility.json",
        force=False,
    )

    neuro = commands["neurodiscovery"]
    assert _value(neuro, "--max-executions") == "100"
    assert _value(neuro, "--proposal-batch-size") == "120"
    assert neuro[neuro.index("--seeds") + 1 : neuro.index("--source-bundle-manifest")] == [
        "0"
    ]
    assert _value(commands["baseline_generation"], "--target-per-case-study") == "100"
    assert _value(commands["baseline_evaluation"], "--random-trials") == "50"


def test_adaptive_feedback_boundary_matches_dynamic_runner() -> None:
    assert _adaptive_feedback_start_budget(
        configured_budget=50,
        execution_batch_size=10,
        initial_valid=120,
    ) == 30
    assert _adaptive_feedback_start_budget(
        configured_budget=50,
        execution_batch_size=10,
        initial_valid=500,
    ) == 50
    assert _adaptive_feedback_start_budget(
        configured_budget=50,
        execution_batch_size=10,
        initial_valid=12,
    ) == 10
