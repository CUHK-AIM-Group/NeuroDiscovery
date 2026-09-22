from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from core.scripts.summarize_unified_case_study_benchmarks import (
    build_case_study_coverage_rows,
    build_endpoint_evidence_rows,
    build_task_diagnostic_rows,
    build_report,
    endpoint_state,
    exact_paired_sign_flip_p,
    holm_adjust,
    summarize_dedicated_case_studies,
    summarize_dynamic_hindcasting,
    summarize_executability_coverage,
    summarize_policy_ablation,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_holm_adjust_returns_values_in_original_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_exact_paired_sign_flip_p_uses_all_sign_assignments() -> None:
    assert exact_paired_sign_flip_p([1.0] * 10) == 1 / 1024


def test_endpoint_state_keeps_negative_and_full_pool_states_distinct() -> None:
    assert endpoint_state(0, 0, gt_total=0) == "no_validated_discoveries"
    assert endpoint_state(0, 0, gt_total=8) == "no_method_hits_at_endpoint"
    assert endpoint_state(8, 8, gt_total=8, full_pool=True) == "full_pool_sanity"
    assert endpoint_state(4, 3, gt_total=8) == "neurodiscovery_strict_win"
    assert (
        endpoint_state(8, 12, gt_total=8, lower_is_better=True)
        == "neurodiscovery_strict_win"
    )


def test_endpoint_evidence_separates_supported_sparse_and_saturated_results() -> None:
    budget_rows = [
        {
            "task": "informative",
            "scope": "internal",
            "experiments": 100,
            "candidate_count": 1000,
            "gt_total": 100,
            "best_comparator_methods": "baseline",
            "state": "neurodiscovery_strict_win",
        },
        {
            "task": "sparse",
            "scope": "external",
            "experiments": 25,
            "candidate_count": 100,
            "gt_total": 8,
            "best_comparator_methods": "baseline",
            "state": "neurodiscovery_strict_win",
        },
        {
            "task": "saturated",
            "scope": "internal",
            "experiments": 50,
            "candidate_count": 100,
            "gt_total": 90,
            "best_comparator_methods": "baseline",
            "state": "neurodiscovery_loss",
        },
    ]
    paired_rows = [
        {
            "task": row["task"],
            "scope": row["scope"],
            "comparison": "same_experiments_hits",
            "value": row["experiments"],
            "baseline": "baseline",
            "p_holm_within_endpoint": 0.01,
        }
        for row in budget_rows
    ]
    evidence = build_endpoint_evidence_rows(budget_rows, paired_rows)
    assert evidence[0]["inference_state"] == "holm_supported_win"
    assert evidence[0]["scientific_win_claim_permitted"] is True
    assert evidence[1]["endpoint_reliability"].startswith("sparse_")
    assert evidence[1]["inference_state"] == "descriptive_win_only"
    assert evidence[2]["endpoint_reliability"].startswith("saturated_")
    assert evidence[2]["inference_state"].startswith("descriptive_loss")

    diagnostics = build_task_diagnostic_rows(evidence)
    indexed = {(row["task"], row["scope"]): row for row in diagnostics}
    assert indexed[("informative", "internal")]["diagnosis"] == (
        "supported_neurodiscovery_advantage"
    )
    assert indexed[("sparse", "external")]["diagnosis"] == (
        "low_positive_count_bottleneck"
    )
    assert indexed[("saturated", "internal")]["diagnosis"] == (
        "saturated_candidate_space_bottleneck"
    )
    assert all(not row["target_specific_tuning_permitted"] for row in diagnostics)


def test_partial_report_preserves_protocol_inapplicability(tmp_path: Path) -> None:
    task = "disease_subtyping"
    closure_root = tmp_path / "closure"
    benchmark = closure_root / task / "benchmark"
    audit_path = tmp_path / "closure_audit.json"
    static_root = tmp_path / "static"
    dynamic_root = tmp_path / "dynamic"
    output_dir = tmp_path / "report"

    _write_json(
        audit_path,
        {
            "all_complete": True,
            "model_robustness_required": True,
            "internal_complete_tasks": 1,
            "external_applicable_tasks": 0,
            "external_complete_tasks": 0,
            "task_count": 1,
            "kg_sha256": "KG",
            "tasks": [
                {
                    "task": task,
                    "status": "complete",
                    "external_applicable": False,
                    "candidate_count": 20,
                    "internal_validated": 8,
                    "external_validated": 0,
                    "cross_cohort_replicated": 0,
                    "model_robustness_models": ["kmeans", "spectral"],
                    "warnings": [],
                }
            ],
        },
    )
    _write_json(
        benchmark / "run_manifest.json",
        {
            "status": "complete",
            "candidate_count": 20,
            "methods": ["random_walk", "neurodiscovery"],
            "trials": 10,
            "formal_kg_mutated": False,
        },
    )
    metric_rows = [
        {
            "scope": "internal",
            "method": method,
            "experiments": experiments,
            "n_trials": 10,
            "hits_mean": hits,
            "hits_variance": variance,
            "recall_mean": hits / 8,
            "recall_variance": 0,
            "precision_mean": hits / experiments,
            "precision_variance": 0,
            "executable_mean": experiments,
            "executable_variance": 0,
            "gt_total_mean": 8,
            "gt_total_variance": 0,
        }
        for experiments, values in ((10, (("random_walk", 3, 1), ("neurodiscovery", 5, 2))), (20, (("random_walk", 8, 0), ("neurodiscovery", 8, 0))))
        for method, hits, variance in values
    ]
    _write_csv(benchmark / "internal_metrics_summary.csv", metric_rows)
    _write_csv(
        benchmark / "external_metrics_summary.csv",
        [
            {
                "scope": "external",
                "method": "neurodiscovery",
                "experiments": 10,
                "hits_mean": 99,
                "hits_variance": 0,
                "gt_total_mean": 99,
            }
        ],
    )
    _write_csv(
        benchmark / "internal_recall_cost_summary.csv",
        [
            {
                "scope": "internal",
                "method": method,
                "recall_target": 0.5,
                "n_trials": 10,
                "hits_needed_mean": 4,
                "hits_needed_variance": 0,
                "experiments_required_mean": experiments,
                "experiments_required_variance": variance,
                "executable_experiments_required_mean": experiments,
                "executable_experiments_required_variance": variance,
                "gt_total_mean": 8,
                "gt_total_variance": 0,
            }
            for method, experiments, variance in (
                ("random_walk", 12, 2),
                ("neurodiscovery", 8, 1),
            )
        ],
    )
    _write_csv(
        benchmark / "external_recall_cost_summary.csv",
        [
            {
                "scope": "external",
                "method": "neurodiscovery",
                "recall_target": 0.5,
                "experiments_required_mean": 1,
                "experiments_required_variance": 0,
                "gt_total_mean": 99,
            }
        ],
    )
    _write_csv(
        benchmark / "paired_p_values.csv",
        [
            {
                "scope": "internal",
                "comparison": "same_experiments_hits",
                "value": 10,
                "target": "neurodiscovery",
                "baseline": "random_walk",
                "n_pairs": 10,
                "p_value": 0.01,
            },
            {
                "scope": "external",
                "comparison": "same_experiments_hits",
                "value": 10,
                "target": "neurodiscovery",
                "baseline": "random_walk",
                "n_pairs": 10,
                "p_value": 0.001,
            },
        ],
    )

    static_design = tmp_path / "static_design.json"
    dynamic_design = tmp_path / "dynamic_design.json"
    _write_json(
        static_design,
        {
            "evidence_tier": "static-test",
            "primary_matrix": {"paired_runs_per_method": 2},
            "endpoints": {"primary": {"metric": "unique_primary_discoveries", "k": 10}},
        },
    )
    _write_json(
        dynamic_design,
        {
            "evidence_tier": "dynamic-test",
            "primary_matrix": {"paired_runs_per_arm": 2},
            "endpoints": {
                "primary": {"metric": "terminal_unique_primary_discoveries", "k": 10}
            },
        },
    )

    manifest = build_report(
        Namespace(
            closure_root=closure_root,
            closure_audit=audit_path,
            static_root=static_root,
            static_design=static_design,
            dynamic_root=dynamic_root,
            dynamic_design=dynamic_design,
            output_dir=output_dir,
        )
    )

    assert manifest["status"] == "partial_running"
    assert not manifest["ready_for_final_inference"]
    status = json.loads((output_dir / "benchmark_status.json").read_text())
    assert status["closure"]["internal_complete"] == 1
    assert status["closure"]["external_expected"] == 0
    assert status["static_hindcasting"]["status"] == "running"
    assert status["dynamic_hindcasting"]["status"] == "running"
    task_rows = list(
        csv.DictReader(
            (output_dir / "closure_task_status.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    assert task_rows[0]["external_reference_state"] == "not_applicable_by_protocol"
    budget_rows = list(
        csv.DictReader(
            (output_dir / "closure_budget_comparisons.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    assert budget_rows[0]["state"] == "neurodiscovery_strict_win"
    assert budget_rows[-1]["state"] == "full_pool_sanity"
    assert {row["scope"] for row in budget_rows} == {"internal"}
    paired_rows = list(
        csv.DictReader(
            (output_dir / "closure_paired_comparisons_holm.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    assert {row["scope"] for row in paired_rows} == {"internal"}


def test_complete_hindcasting_requires_formal_verification(tmp_path: Path) -> None:
    task = "disease_subtyping"
    closure_root = tmp_path / "closure"
    benchmark = closure_root / task / "benchmark"
    audit_path = tmp_path / "closure_audit.json"
    _write_json(
        audit_path,
        {
            "all_complete": True,
            "model_robustness_required": True,
            "internal_complete_tasks": 1,
            "external_applicable_tasks": 0,
            "external_complete_tasks": 0,
            "task_count": 1,
            "kg_sha256": "KG",
            "tasks": [
                {
                    "task": task,
                    "status": "complete",
                    "external_applicable": False,
                    "candidate_count": 10,
                    "internal_validated": 1,
                    "external_validated": 0,
                    "cross_cohort_replicated": 0,
                    "model_robustness_models": ["kmeans"],
                    "warnings": [],
                }
            ],
        },
    )
    _write_json(
        benchmark / "run_manifest.json",
        {
            "status": "complete",
            "candidate_count": 10,
            "formal_kg_mutated": False,
        },
    )
    _write_csv(
        benchmark / "internal_metrics_summary.csv",
        [
            {
                "scope": "internal",
                "method": method,
                "experiments": 5,
                "n_trials": 10,
                "hits_mean": hits,
                "hits_variance": 0,
                "recall_mean": hits,
                "recall_variance": 0,
                "precision_mean": hits / 5,
                "precision_variance": 0,
                "executable_mean": 5,
                "executable_variance": 0,
                "gt_total_mean": 1,
                "gt_total_variance": 0,
            }
            for method, hits in (("random_walk", 0), ("neurodiscovery", 1))
        ],
    )
    _write_csv(
        benchmark / "internal_recall_cost_summary.csv",
        [
            {
                "scope": "internal",
                "method": method,
                "recall_target": 1,
                "n_trials": 10,
                "hits_needed_mean": 1,
                "hits_needed_variance": 0,
                "experiments_required_mean": experiments,
                "experiments_required_variance": 0,
                "executable_experiments_required_mean": experiments,
                "executable_experiments_required_variance": 0,
                "gt_total_mean": 1,
                "gt_total_variance": 0,
            }
            for method, experiments in (("random_walk", 5), ("neurodiscovery", 2))
        ],
    )
    _write_csv(
        benchmark / "paired_p_values.csv",
        [
            {
                "scope": "internal",
                "comparison": "same_experiments_hits",
                "value": 5,
                "target": "neurodiscovery",
                "baseline": "random_walk",
                "n_pairs": 10,
                "p_value": 0.01,
            }
        ],
    )

    static_root = tmp_path / "static"
    dynamic_root = tmp_path / "dynamic"
    _write_json(static_root / "execution_manifest.json", {"status": "complete", "mode": "formal"})
    _write_json(dynamic_root / "execution_manifest.json", {"status": "complete", "mode": "formal"})
    _write_csv(
        static_root / "summary" / "mean_variance_summary.csv",
        [
            {
                "scope": "macro_case_study_equal",
                "method": "neurodiscovery",
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "k": 10,
                "metric": "unique_primary_discoveries",
                "n": 10,
                "mean": 2,
                "sample_variance": 1,
            }
        ],
    )
    _write_csv(
        static_root / "summary" / "paired_comparisons.csv",
        [
            {
                "scope": "macro_case_study_equal",
                "reference_method": "neurodiscovery",
                "comparison_method": "sciagents",
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "k": 10,
                "metric": "unique_primary_discoveries",
                "n_pairs": 10,
                "p_reference_greater_exact_sign_flip": 0.01,
            }
        ],
    )
    _write_json(static_root / "summary" / "summary_manifest.json", {"status": "complete"})
    dynamic_rows = [
            {
                "scope": "macro_case_study_equal",
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "requested_k": 10,
                "metric": "terminal_unique_primary_discoveries",
                "n_pairs": 10,
                "closed_mean": 2,
                "open_mean": 1,
                "p_closed_greater_exact_sign_flip": 0.01,
            }
        ] + [
            {
                "scope": "case_study_equal_window",
                "case_study_id": task_name,
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "requested_k": 10,
                "metric": "terminal_unique_primary_discoveries",
                "n_pairs": 10,
                "closed_mean": 2,
                "open_mean": 1,
                "p_closed_greater_exact_sign_flip": p_value,
            }
            for task_name, p_value in (
                ("disease_subtyping", 0.01),
                ("imaging_genetics", 0.04),
                ("connectome_behavior", 0.03),
            )
        ]
    _write_csv(
        dynamic_root / "comparison" / "mean_variance_paired_summary.csv",
        dynamic_rows,
    )
    _write_json(dynamic_root / "comparison" / "comparison_manifest.json", {"status": "complete"})

    static_design = tmp_path / "static_design.json"
    dynamic_design = tmp_path / "dynamic_design.json"
    _write_json(
        static_design,
        {
            "evidence_tier": "static-test",
            "primary_matrix": {"paired_runs_per_method": 1},
            "endpoints": {"primary": {"metric": "unique_primary_discoveries", "k": 10}},
        },
    )
    _write_json(
        dynamic_design,
        {
            "evidence_tier": "dynamic-test",
            "primary_matrix": {"paired_runs_per_arm": 1},
            "endpoints": {
                "primary": {"metric": "terminal_unique_primary_discoveries", "k": 10}
            },
        },
    )
    args = Namespace(
        closure_root=closure_root,
        closure_audit=audit_path,
        static_root=static_root,
        static_design=static_design,
        dynamic_root=dynamic_root,
        dynamic_design=dynamic_design,
        output_dir=tmp_path / "report",
    )
    unverified = build_report(args)
    assert not unverified["ready_for_final_inference"]

    _write_json(
        static_root / "verification" / "verification_manifest.json",
        {"status": "passed", "mode": "formal"},
    )
    _write_json(
        dynamic_root / "verification" / "verification_manifest.json",
        {"status": "passed"},
    )
    verified = build_report(args)
    assert verified["ready_for_final_inference"]
    assert verified["status"] == "complete"
    dynamic = summarize_dynamic_hindcasting(
        root=dynamic_root,
        design_path=dynamic_design,
        inputs={},
    )
    task_rows = dynamic["tables"]["dynamic_case_study_primary.csv"]
    assert [row["p_holm_within_endpoint"] for row in task_rows] == [
        0.03,
        0.06,
        0.06,
    ]
    assert all(row["holm_family_size"] == 3 for row in task_rows)


def test_policy_ablation_requires_complete_hashed_suite(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    task = "brain_age"
    jobs = []
    for track, hits, cost in (
        ("static_ablation", 2, 8),
        ("closed_loop", 4, 5),
    ):
        benchmark = root / track / task / "benchmark"
        run_manifest = benchmark / "run_manifest.json"
        closure_manifest = benchmark.parent / "closure_manifest.json"
        _write_json(
            run_manifest,
            {
                "status": "complete",
                "candidate_count": 10,
                "trials": 10,
                "neurodiscovery_policy": {"policy_id": track},
                "closed_loop_config": {
                    "preserve_static_until_informative_feedback": True
                },
            },
        )
        _write_json(closure_manifest, {"status": "complete", "track": track})
        for scope in ("internal", "external"):
            _write_csv(
                benchmark / f"{scope}_metrics_summary.csv",
                [
                    {
                        "scope": scope,
                        "method": "neurodiscovery",
                        "experiments": 5,
                        "n_trials": 10,
                        "hits_mean": hits,
                        "hits_variance": 1,
                        "recall_mean": hits / 5,
                        "recall_variance": 0,
                        "precision_mean": hits / 5,
                        "precision_variance": 0,
                        "executable_mean": 5,
                        "executable_variance": 0,
                        "gt_total_mean": 5,
                        "gt_total_variance": 0,
                    }
                ],
            )
            _write_csv(
                benchmark / f"{scope}_metrics_by_trial.csv",
                [
                    {
                        "scope": scope,
                        "method": "neurodiscovery",
                        "trial": trial,
                        "experiments": 5,
                        "hits": hits,
                        "recall": hits / 5,
                        "precision": hits / 5,
                        "executable": 5,
                        "gt_total": 5,
                    }
                    for trial in range(10)
                ],
            )
            _write_csv(
                benchmark / f"{scope}_recall_cost_summary.csv",
                [
                    {
                        "scope": scope,
                        "method": "neurodiscovery",
                        "recall_target": 0.5,
                        "n_trials": 10,
                        "hits_needed_mean": 3,
                        "hits_needed_variance": 0,
                        "experiments_required_mean": cost,
                        "experiments_required_variance": 1,
                        "executable_experiments_required_mean": cost,
                        "executable_experiments_required_variance": 1,
                        "gt_total_mean": 5,
                        "gt_total_variance": 0,
                    }
                ],
            )
            _write_csv(
                benchmark / f"{scope}_recall_cost_by_trial.csv",
                [
                    {
                        "scope": scope,
                        "method": "neurodiscovery",
                        "trial": trial,
                        "recall_target": 0.5,
                        "hits_needed": 3,
                        "experiments_required": cost,
                        "executable_experiments_required": cost,
                        "gt_total": 5,
                    }
                    for trial in range(10)
                ],
            )
        jobs.append(
            {
                "track": track,
                "task": task,
                "status": "complete",
                "output_dir": str(benchmark),
                "result_manifest_sha256": "",
                "closure_manifest": str(closure_manifest),
                "closure_manifest_sha256": "",
            }
        )

    import hashlib

    for job in jobs:
        result = Path(job["output_dir"]) / "run_manifest.json"
        closure = Path(job["closure_manifest"])
        job["result_manifest_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
        job["closure_manifest_sha256"] = hashlib.sha256(closure.read_bytes()).hexdigest()
    policy = root / "policy.json"
    _write_json(
        policy,
        {
            "status": "frozen",
            "selection_protocol": {"external_labels_used": False},
        },
    )
    _write_json(
        root / "rerun_suite_manifest.json",
        {
            "status": "complete",
            "tracks": ["static_ablation", "closed_loop"],
            "tasks": [task],
            "policy_manifest": {
                "path": str(policy),
                "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            },
            "jobs": jobs,
        },
    )

    inputs: dict[str, dict[str, object]] = {}
    summary = summarize_policy_ablation(suite_root=root, inputs=inputs)
    assert summary["status"] == "complete"
    assert summary["jobs"] == 2
    assert summary["formal_scopes"] == ["internal"]
    assert (
        summary["generic_external_results_role"]
        == "exploratory_only_not_completion_gate"
    )
    budget = summary["tables"]["policy_ablation_budget_comparisons.csv"]
    assert {row["scope"] for row in budget} == {"internal"}
    assert budget[0]["closed_minus_static_hits"] == 2
    recall = summary["tables"]["policy_ablation_recall_cost_comparisons.csv"]
    assert {row["scope"] for row in recall} == {"internal"}
    assert recall[0]["experiments_saved_by_closed_loop"] == 3
    paired = summary["tables"]["policy_ablation_paired_comparisons_holm.csv"]
    assert len(paired) == 2
    assert {row["scope"] for row in paired} == {"internal"}
    assert all(row["n_pairs"] == 10 for row in paired)
    assert all(row["p_value_exact_one_sided"] == 1 / 1024 for row in paired)
    assert all(row["p_holm_within_endpoint_family"] == 1 / 1024 for row in paired)
    assert not any("external_" in path.lower() for path in inputs)
    assert summary["statistical_contract"]["confirmatory_inference"] is False


def test_running_dedicated_suite_is_reported_without_audit(tmp_path: Path) -> None:
    root = tmp_path / "dedicated"
    _write_json(
        root / "dedicated_suite_manifest.json",
        {
            "status": "running",
            "trials": 10,
            "protocol": {
                "case1_external": "UCLA, COBRE, HCP-EP, and ADHD200",
                "case2_external": "not applicable",
            },
            "stages": {
                "case2_generate": {"status": "complete"},
                "case2_map": {"status": "running"},
            },
        },
    )

    summary = summarize_dedicated_case_studies(suite_root=root, inputs={})

    assert summary["status"] == "running"
    assert summary["ready_for_inference"] is False
    assert summary["stages_complete"] == 1
    assert summary["stages_total"] == 2
    rows = summary["tables"]["dedicated_case_study_status.csv"]
    assert rows[0]["internal_status"] == "pending"
    assert rows[1]["internal_status"] == "running"
    assert rows[1]["external_status"] == "not_applicable_by_protocol"


def test_completed_dedicated_suite_exports_case2_paired_comparisons(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dedicated"
    _write_json(
        root / "dedicated_suite_manifest.json",
        {
            "status": "complete",
            "trials": 10,
            "protocol": {
                "case1_external": "UCLA, COBRE, HCP-EP, and ADHD200",
                "case2_external": "not applicable",
            },
            "canonical_release": {"kg_sha256": "test"},
        },
    )
    _write_json(
        root / "audit" / "dedicated_audit.json",
        {"status": "passed", "failed": 0, "checks": 12},
    )

    sources = (
        root
        / "case1_transdiagnostic"
        / "internal_method_comparison"
        / "case1_discovery_curves_by_trial.csv",
        root
        / "case1_transdiagnostic"
        / "internal_method_comparison"
        / "case1_method_summary_by_trial.csv",
        root
        / "case1_transdiagnostic"
        / "external_method_comparison"
        / "case1_external_metrics_by_trial.csv",
        root
        / "case1_transdiagnostic"
        / "external_method_comparison"
        / "case1_external_recall_cost_by_trial.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "generation_quality_primary_summary.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "internal_primary_budget_summary.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "internal_primary_recall_cost_summary.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "internal_primary_p_values.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "internal_same_experiments_headline.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "internal_same_recall_headline.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "external_primary_metrics_summary.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "external_primary_recall_cost_summary.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "external_primary_p_values.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "external_pooled_same_experiments_headline.csv",
        root
        / "case1_transdiagnostic"
        / "final_summary"
        / "external_pooled_same_recall_headline.csv",
        root
        / "case2_pathway_mediation"
        / "comparison"
        / "case2_method_metrics_by_trial.csv",
        root
        / "case2_pathway_mediation"
        / "comparison"
        / "case2_method_metrics_aggregate.csv",
    )
    for source in sources:
        _write_csv(source, [{"method": "neurodiscovery", "value": 1}])
    paired = (
        root
        / "case2_pathway_mediation"
        / "comparison"
        / "case2_paired_comparisons.csv"
    )
    _write_csv(
        paired,
        [
            {
                "baseline_method": "sciagents",
                "k": 20,
                "n_pairs": 10,
                "exact_p_value": 0.001,
                "holm_p_value": 0.006,
            }
        ],
    )

    inputs: dict[str, dict[str, object]] = {}
    summary = summarize_dedicated_case_studies(suite_root=root, inputs=inputs)

    assert summary["ready_for_inference"] is True
    rows = summary["tables"]["dedicated_case2_paired_comparisons.csv"]
    assert rows == [
        {
            "task": "case2_pathway_mediation",
            "scope": "internal",
            "baseline_method": "sciagents",
            "k": "20",
            "n_pairs": "10",
            "exact_p_value": "0.001",
            "holm_p_value": "0.006",
        }
    ]
    assert str(paired.resolve()) in inputs


def test_executability_coverage_audits_every_formal_case_study(
    tmp_path: Path,
) -> None:
    import hashlib

    kg_sha = "A" * 64
    audit_path = tmp_path / "audit" / "executability_audit.json"
    matrix_path = tmp_path / "audit" / "executability_matrix.csv"
    locked_path = tmp_path / "lock" / "eligibility_matrix_locked.csv"
    windows = [
        {"freeze_year": 2019, "future_start_year": 2020, "future_end_year": 2024},
        {"freeze_year": 2020, "future_start_year": 2021, "future_end_year": 2025},
    ]
    rows = [
        {
            "case_study_id": "case1_transdiagnostic",
            **window,
            "future_unique_pairs": 12,
            "structural_status": "executable",
            "status_reason": "",
        }
        for window in windows
    ] + [
        {
            "case_study_id": "brain_age",
            **window,
            "future_unique_pairs": 3,
            "structural_status": "sparse",
            "status_reason": "below stable threshold",
        }
        for window in windows
    ]
    _write_json(
        audit_path,
        {
            "case_studies": ["case1_transdiagnostic", "brain_age"],
            "status_counts": {"executable": 2, "sparse": 2, "non_executable": 0},
            "canonical_release": {
                "files": {"knowledge_graph": {"sha256": kg_sha}}
            },
            "rows": rows,
        },
    )
    _write_csv(matrix_path, rows)
    _write_csv(locked_path, [{**row, "analysis_tier": "primary"} for row in rows])

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest().upper()
    eligibility_path = tmp_path / "lock" / "eligibility_manifest.json"
    _write_json(
        eligibility_path,
        {
            "status": "locked_before_new_all_task_re_evaluation",
            "source": {
                "executability_audit_json": str(audit_path),
                "executability_audit_json_sha256": digest(audit_path),
                "executability_matrix_csv": str(matrix_path),
                "executability_matrix_csv_sha256": digest(matrix_path),
                "canonical_release": {
                    "files": {"knowledge_graph": {"sha256": kg_sha}}
                },
            },
            "formal_case_study_ids": ["case1_transdiagnostic", "brain_age"],
            "formal_case_studies": 2,
            "common_temporal_windows": windows,
            "status_counts": {"executable": 2, "sparse": 2, "non_executable": 0},
            "primary_case_study_ids": ["case1_transdiagnostic"],
            "exploratory_case_study_ids": ["brain_age"],
            "locked_matrix": {
                "path": str(locked_path),
                "sha256": digest(locked_path),
            },
        },
    )
    design_path = tmp_path / "formal_static_design.json"
    _write_json(
        design_path,
        {
            "canonical_release": {"knowledge_graph": {"sha256": kg_sha}},
            "primary_matrix": {"formal_case_studies": 2},
            "temporal_inputs": {
                "eligibility_manifest": {
                    "path": str(eligibility_path),
                    "sha256": digest(eligibility_path),
                }
            },
        },
    )

    inputs: dict[str, dict[str, object]] = {}
    summary = summarize_executability_coverage(
        static_design_path=design_path,
        inputs=inputs,
    )
    assert summary["formal_case_studies"] == 2
    assert summary["windows"] == 4
    assert summary["status_counts"] == {
        "executable": 2,
        "sparse": 2,
        "non_executable": 0,
    }
    case_rows = summary["tables"]["case_study_executability_summary.csv"]
    assert case_rows[0]["hindcasting_role"] == "primary"
    assert case_rows[1]["hindcasting_role"] == "exploratory_only"
    assert all(
        not row["method_performance_consumed_for_selection"]
        for row in summary["tables"]["case_study_executability_windows.csv"]
    )

    coverage = build_case_study_coverage_rows(
        executability=summary,
        closure={"tables": {"closure_task_status.csv": []}},
        dedicated={
            "tables": {
                "dedicated_case_study_status.csv": [
                    {
                        "task": "case1_transdiagnostic",
                        "internal_status": "complete",
                        "external_status": "complete",
                    }
                ]
            }
        },
    )
    assert len(coverage) == 2
    assert coverage[0]["unified_coverage_state"] == (
        "dataset_validation_and_primary_hindcasting"
    )
    assert coverage[1]["unified_coverage_state"] == "exploratory_hindcasting_only"
