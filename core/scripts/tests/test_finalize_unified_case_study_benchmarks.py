from __future__ import annotations

import csv
import hashlib
import json
from argparse import Namespace
from pathlib import Path

from core.scripts.finalize_unified_case_study_benchmarks import (
    REQUIRED_REPORT_OUTPUTS,
    _require_release_and_isolation_contract,
    build_dynamic_verify_command,
    build_report_command,
    build_static_verify_command,
    dedicated_suite_passed,
    execution_complete,
    verify_unified_report,
    verification_passed,
)


CANONICAL_KG_SHA256 = "A" * 64
SELECTION_KG_SHA256 = "B" * 64


def _valid_benchmark_status(
    *,
    static_verification_path: Path | None = None,
    dynamic_verification_path: Path | None = None,
) -> dict[str, object]:
    return {
        "ready_for_final_inference": True,
        "closure": {
            "audit_all_complete": True,
            "internal_complete": 8,
            "internal_expected": 8,
            "external_complete": 0,
            "external_expected": 0,
            "formal_external_validation_case_studies": [
                "case1_transdiagnostic"
            ],
            "generic_external_results_role": (
                "exploratory_only_not_completion_gate"
            ),
            "kg_sha256": CANONICAL_KG_SHA256,
        },
        "policy_ablation": {
            "ready_for_inference": True,
            "tasks": 8,
            "jobs": 16,
            "formal_scopes": ["internal"],
            "generic_external_results_role": (
                "exploratory_only_not_completion_gate"
            ),
            "selection_protocol": {
                "source_folds": 5,
                "source_trials": 2,
                "source_task_weighting": "equal after within-task normalization",
                "external_outcomes_used_for_selection": False,
                "target_internal_outcome_tables_opened": False,
                "target_external_outcome_tables_opened": False,
                "informative_feedback_guard_selected_on_source_release": True,
            },
        },
        "dedicated_case_studies": {
            "ready_for_inference": True,
            "audit_checks": 44,
            "case_studies": 2,
            "internal_complete": 2,
            "external_complete": 1,
            "external_not_applicable": 1,
            "canonical_release": {
                "files": {
                    "knowledge_graph": {"sha256": CANONICAL_KG_SHA256},
                }
            },
        },
        "case_study_executability": {
            "ready_for_inference": True,
            "formal_case_studies": 17,
            "windows": 85,
            "status_counts": {
                "executable": 30,
                "sparse": 15,
                "non_executable": 40,
            },
            "kg_sha256": CANONICAL_KG_SHA256,
        },
        "static_hindcasting": {
            "ready_for_inference": True,
            "neurodiscovery_runs_complete": 300,
            "neurodiscovery_runs_expected": 300,
            "baseline_runs_complete": 600,
            "baseline_runs_expected": 600,
            "verification": {"path": str(static_verification_path or "static.json")},
            "configuration_provenance": {
                "canonical_release": {
                    "knowledge_graph": {"sha256": CANONICAL_KG_SHA256},
                },
                "policy_application": {
                    "mode": "unchanged_cross_release_application",
                    "selection_release": {
                        "knowledge_graph_sha256": SELECTION_KG_SHA256,
                    },
                    "application_release": {
                        "knowledge_graph_sha256": CANONICAL_KG_SHA256,
                    },
                    "new_release_future_outcomes_used_for_policy_selection": False,
                    "post_update_retuning_permitted": False,
                },
                "outcome_isolation": {
                    "formal_kg_mutation_permitted": False,
                    "post_outcome_parameter_tuning_permitted": False,
                },
                "reproducibility": {
                    "bundle_verified_before_and_after_each_stage": True,
                    "formal_runs_rehash_all_referenced_inputs": True,
                    "smoke_outputs_excluded_from_formal_analysis": True,
                },
                "methods": {
                    "primary": [
                        {
                            "id": "neurodiscovery",
                            "uses_future_outcomes": False,
                            "generation_pool_size": 1200,
                        },
                        {"id": "sciagents", "uses_future_outcomes": False},
                        {"id": "openscholar_rag", "uses_future_outcomes": False},
                    ]
                },
                "fixed_budget_contract": {
                    "executed_hypotheses_per_run": 1000,
                },
            },
        },
        "dynamic_hindcasting": {
            "ready_for_inference": True,
            "closed_runs_complete": 50,
            "closed_runs_expected": 50,
            "open_runs_complete": 50,
            "open_runs_expected": 50,
            "verification": {"path": str(dynamic_verification_path or "dynamic.json")},
            "configuration_provenance": {
                "canonical_release": {
                    "knowledge_graph": {"sha256": CANONICAL_KG_SHA256},
                },
                "outcome_isolation": {
                    "generation_uses_only_frozen_KG": True,
                    "closed_feedback_uses_only_early_interval": True,
                    "open_feedback_is_withheld": True,
                    "terminal_labels_available_to_generation": False,
                    "formal_KG_mutation_permitted": False,
                },
                "reproducibility": {
                    "bundle_verified_before_and_after_each_stage": True,
                    "formal_runs_rehash_all_referenced_inputs": True,
                    "smoke_outputs_excluded_from_formal_analysis": True,
                },
            },
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _descriptor(path: Path, *, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def test_completion_and_verification_contracts(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_json(root / "execution_manifest.json", {"status": "complete", "mode": "smoke"})
    assert not execution_complete(root)
    _write_json(root / "execution_manifest.json", {"status": "complete", "mode": "formal"})
    assert execution_complete(root)

    _write_json(
        root / "verification" / "verification_manifest.json",
        {"status": "passed"},
    )
    assert not verification_passed(root, static=True)
    assert not verification_passed(root, static=False)
    _write_json(
        root / "verification" / "verification_manifest.json",
        {
            "status": "passed",
            "mode": "formal",
            "temporal_isolation_audit_by_method": {
                method: {"hypotheses": 1, "flags": 0, "evidence_years": 0}
                for method in ("neurodiscovery", "sciagents", "openscholar_rag")
            },
            "closed": {"runs": 1},
            "open": {"runs": 1},
            "formal_kg_mutated": False,
            "future_outcomes_used_for_task_or_window_selection": False,
            "claim_as_untouched_confirmatory_permitted": False,
        },
    )
    assert verification_passed(root, static=True)
    assert verification_passed(root, static=False)

    _write_json(root / "dedicated_suite_manifest.json", {"status": "complete"})
    _write_json(
        root / "audit" / "dedicated_audit.json",
        {"status": "passed", "failed": 0},
    )
    assert dedicated_suite_passed(root)


def test_commands_rehash_and_require_complete(tmp_path: Path) -> None:
    args = Namespace(
        python=tmp_path / "python.exe",
        workspace_root=tmp_path / "workspace",
        static_root=tmp_path / "static",
        static_design=tmp_path / "static.json",
        static_verifier=tmp_path / "verify_static.py",
        dynamic_root=tmp_path / "dynamic",
        dynamic_design=tmp_path / "dynamic.json",
        dynamic_verifier=tmp_path / "verify_dynamic.py",
        summarizer=tmp_path / "summarize.py",
        closure_root=tmp_path / "closure",
        closure_audit=tmp_path / "audit.json",
        policy_suite_root=tmp_path / "policy_suite",
        dedicated_suite_root=tmp_path / "dedicated_suite",
        report_root=tmp_path / "report",
    )
    static = build_static_verify_command(args)
    dynamic = build_dynamic_verify_command(args)
    report = build_report_command(args)
    assert "--rehash-references" in static
    assert "--rehash-references" in dynamic
    assert "--workspace-root" in dynamic
    assert ["--policy-suite-root", str(args.policy_suite_root)] == report[-5:-3]
    assert ["--dedicated-suite-root", str(args.dedicated_suite_root)] == report[-3:-1]
    assert report[-1] == "--require-complete"


def test_unified_completion_audit_rehashes_and_checks_required_tables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "report"
    input_path = tmp_path / "input.json"
    _write_json(input_path, {"status": "frozen"})
    static_verification_path = tmp_path / "static_verification.json"
    _write_json(
        static_verification_path,
        {
            "status": "passed",
            "mode": "formal",
            "temporal_isolation_audit_by_method": {
                "neurodiscovery": {
                    "hypotheses": 360000,
                    "flags": 1,
                    "evidence_years": 1,
                },
                "sciagents": {
                    "hypotheses": 300000,
                    "flags": 0,
                    "evidence_years": 1,
                },
                "openscholar_rag": {
                    "hypotheses": 300000,
                    "flags": 0,
                    "evidence_years": 1,
                },
            },
        },
    )
    dynamic_verification_path = tmp_path / "dynamic_verification.json"
    _write_json(
        dynamic_verification_path,
        {
            "status": "passed",
            "paired_runs": 50,
            "closed": {"runs": 50},
            "open": {"runs": 50},
            "formal_kg_mutated": False,
            "future_outcomes_used_for_task_or_window_selection": False,
            "claim_as_untouched_confirmatory_permitted": False,
        },
    )
    status_path = root / "benchmark_status.json"
    _write_json(
        status_path,
        _valid_benchmark_status(
            static_verification_path=static_verification_path,
            dynamic_verification_path=dynamic_verification_path,
        ),
    )

    rows_by_name: dict[str, list[dict[str, object]]] = {}
    for name in REQUIRED_REPORT_OUTPUTS - {"benchmark_status.json"}:
        rows_by_name[name] = [{"value": 1}]
    rows_by_name["case_study_unified_coverage.csv"] = [
        {"case_study_id": f"case_{index}"} for index in range(17)
    ]
    rows_by_name["closure_configuration.csv"] = [
        {
            "task": f"task_{index}",
            "formal_kg_mutated": False,
            "uses_experimental_outcomes": False,
            "trials": 10,
            "methods": ";".join(
                (
                    "random_walk",
                    "ai_scientist_v2",
                    "open_coscientist",
                    "sciagents",
                    "virtual_lab",
                    "brainpilot_native",
                    "biomni_native",
                    "neurodiscovery",
                )
            ),
            "neurodiscovery_score_family": "relation_aware",
            "run_manifest_sha256": f"manifest-{index}",
        }
        for index in range(8)
    ]
    rows_by_name["policy_ablation_config.csv"] = [
        {
            "task": f"task_{index}",
            "track": track,
            "trials": 10,
            "policy_id": f"policy-{track}",
        }
        for index in range(8)
        for track in ("closed_loop", "static_ablation")
    ]
    rows_by_name["policy_ablation_paired_comparisons_holm.csv"] = [
        {
            "scope": "internal",
            "n_pairs": 10,
            "p_value_exact_one_sided": 0.01,
            "p_holm_within_endpoint_family": 0.02,
            "confirmatory_inference": False,
            "evidence_tier": "posthoc_multiplicity_adjusted_policy_ablation",
        }
    ]

    outputs = {"benchmark_status.json": _descriptor(status_path)}
    for name, rows in rows_by_name.items():
        path = root / name
        _write_csv(path, rows)
        outputs[name] = _descriptor(path, rows=len(rows))
    _write_json(
        root / "unified_benchmark_manifest.json",
        {
            "status": "complete",
            "ready_for_final_inference": True,
            "interpretation_contract": {
                "engineering_completion_is_not_scientific_superiority": True,
                "negative_endpoints_are_retained": True,
                "protocol_inapplicability_is_not_failure": True,
                "formal_hindcasting_requires_independent_verification": True,
                "dedicated_case_studies_require_independent_audit": True,
                "policy_ablation_paired_tests_are_secondary_not_confirmatory": True,
                "only_case1_requires_external_validation": True,
                "generic_external_results_are_exploratory_only": True,
            },
            "inputs": [
                _descriptor(input_path),
                _descriptor(static_verification_path),
                _descriptor(dynamic_verification_path),
            ],
            "outputs": outputs,
        },
    )

    audit = verify_unified_report(root)
    assert audit["status"] == "passed"
    assert audit["case_studies_verified"] == 17
    assert audit["static_temporal_hypotheses"] == 960000
    assert audit["dynamic_closed_runs"] == 50
    assert audit["dynamic_open_runs"] == 50
    (root / "static_primary_endpoint.csv").write_text("corrupted", encoding="utf-8")
    try:
        verify_unified_report(root)
    except ValueError as error:
        assert "byte count mismatch" in str(error) or "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("corrupted report output was not rejected")


def test_release_and_isolation_contract_rejects_leakage_and_release_mismatch() -> None:
    status = _valid_benchmark_status()
    assert _require_release_and_isolation_contract(status) == CANONICAL_KG_SHA256

    leaked = json.loads(json.dumps(status))
    leaked["policy_ablation"]["selection_protocol"][
        "target_external_outcome_tables_opened"
    ] = True
    try:
        _require_release_and_isolation_contract(leaked)
    except ValueError as error:
        assert "target_external_outcome_tables_opened" in str(error)
    else:
        raise AssertionError("target external outcome leakage was not rejected")

    external_scope = json.loads(json.dumps(status))
    external_scope["policy_ablation"]["formal_scopes"] = ["internal", "external"]
    try:
        _require_release_and_isolation_contract(external_scope)
    except ValueError as error:
        assert "internal-only" in str(error)
    else:
        raise AssertionError("generic external policy ablation was not rejected")

    mismatched = json.loads(json.dumps(status))
    mismatched["dedicated_case_studies"]["canonical_release"]["files"][
        "knowledge_graph"
    ]["sha256"] = "C" * 64
    try:
        _require_release_and_isolation_contract(mismatched)
    except ValueError as error:
        assert "different canonical KG" in str(error)
    else:
        raise AssertionError("cross-suite canonical KG mismatch was not rejected")
