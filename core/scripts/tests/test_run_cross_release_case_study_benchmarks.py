from __future__ import annotations

from pathlib import Path

from core.scripts.run_cross_release_case_study_benchmarks import (
    _compact_benchmark_manifest,
    build_command,
)


def test_build_command_reuses_frozen_search_policies(tmp_path: Path) -> None:
    public = tmp_path / "public.csv"
    internal = tmp_path / "internal.csv"
    external = tmp_path / "external.csv"
    search = tmp_path / "search.jsonl"
    policy = tmp_path / "policy.json"
    for path in (public, internal, external, search, policy):
        path.write_text("x", encoding="utf-8")
    manifest = {
        "factor_fields": ["outcome", "model"],
        "methods": ["random_walk", "ai_scientist_v2", "neurodiscovery"],
        "trials": 10,
        "budgets": [10, 20],
        "recall_targets": [0.1, 0.2],
        "external": {"path": str(external)},
        "precomputed_baseline_rankings": None,
        "frozen_discovery": {
            "inputs": {
                "public_candidates": {"path": str(public)},
                "internal_outcomes": {"path": str(internal)},
                "search_policies": {"path": str(search)},
            }
        },
    }
    command = build_command(
        python=Path("python.exe"),
        task="prognosis",
        run_manifest=manifest,
        policy_path=policy,
        output_dir=tmp_path / "out",
        seed=7,
    )
    assert "--search-policies" in command
    assert "--external-outcomes" in command
    assert "--neurodiscovery-config" in command
    assert command[command.index("--seed") + 1] == "7"


def test_compact_benchmark_preserves_feedback_audit_without_overlays(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    compact = _compact_benchmark_manifest(
        {
            "experimental_kg_delta": {
                "records": 10,
                "feedback_consumed_during_ranking": True,
                "overlays": [{"records": 5}, {"records": 5}],
            },
            "artifacts": {"large": {"path": "unused"}},
        },
        manifest_path,
    )
    assert "overlays" not in compact["experimental_kg_delta"]
    assert compact["experimental_kg_delta"][
        "feedback_consumed_during_ranking"
    ] is True
    assert compact["experimental_kg_delta"][
        "overlay_records_elided_from_closure_manifest"
    ] == 2
