from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from core.scripts.audit_dedicated_case_study_benchmarks import (
    EXPECTED_METHODS,
    audit,
)
from core.scripts.canonical_kg_release import CURRENT_CANONICAL_SHA256
from core.scripts.run_dedicated_case_study_benchmarks import (
    completed_stage_is_valid,
    copy_verified,
    descriptor,
    suite_is_complete,
)
from core.scripts.summarize_unified_case_study_benchmarks import (
    summarize_dedicated_case_studies,
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


def _release() -> dict[str, object]:
    return {
        "files": {
            name: {"sha256": value}
            for name, value in CURRENT_CANONICAL_SHA256.items()
        }
    }


def _stage_record(path: Path) -> dict[str, object]:
    return {
        "status": "complete",
        "outputs": {str(path.resolve()): descriptor(path)},
    }


def _build_passing_suite(root: Path) -> None:
    trials = range(10)
    methods = sorted(EXPECTED_METHODS)
    case1 = root / "case1_transdiagnostic"
    internal = case1 / "internal_method_comparison"
    external = case1 / "external_method_comparison"
    case2 = root / "case2_pathway_mediation" / "comparison"

    internal_manifest = internal / "case1_method_comparison_manifest.json"
    _write_json(
        internal_manifest,
        {
            "canonical_kg_release": _release(),
            "experimental_kg_delta": {
                "overlay_count": 10,
                "feedback_consumed_during_ranking": True,
                "per_seed_isolation_verified": True,
                "mutates_formal_kg": False,
            },
        },
    )
    _write_csv(
        internal / "case1_method_summary_by_trial.csv",
        [{"method": method, "trial": trial} for method in methods for trial in trials],
    )
    _write_csv(
        internal / "case1_discovery_curves_by_trial.csv",
        [
            {"method": method, "trial": trial, "budget": 100, "gt_hits": 1}
            for method in methods
            for trial in trials
        ],
    )

    external_manifest = external / "case1_external_method_comparison_manifest.json"
    _write_json(
        external_manifest,
        {
            "canonical_kg_release": _release(),
            "frozen_tcp_rankings": {
                "external_data_read_before_freeze": False,
                "orders": {
                    "n_trials_by_method": {method: 10 for method in methods}
                },
            },
            "protocol": {
                "external_validation_only": True,
                "external_feedback_to_tcp_ranking": False,
                "external_datasets": ["ucla", "cobre", "hcpep", "adhd200"],
            },
        },
    )
    _write_csv(
        external / "case1_external_metrics_by_trial.csv",
        [
            {"method": method, "trial": trial, "dataset": dataset}
            for method in methods
            for trial in trials
            for dataset in ("ucla", "cobre", "hcpep", "adhd200", "pooled")
        ],
    )
    _write_csv(
        external / "case1_external_recall_cost_by_trial.csv",
        [
            {
                "method": method,
                "trial": trial,
                "dataset": "pooled",
                "recall_target": 0.1,
            }
            for method in methods
            for trial in trials
        ],
    )
    final_root = case1 / "final_summary"
    internal_p_values = final_root / "internal_primary_p_values.csv"
    _write_csv(
        internal_p_values,
        [
            {
                "comparison_type": comparison,
                "budget_or_recall_target": value,
                "baseline_method": method,
                "n_paired_seeds": 10,
                "neurodiscovery_mean": 2,
                "baseline_mean": 1,
                "mean_paired_difference": 1,
                "wilcoxon_statistic": 55,
                "p_value_one_sided": 0.001,
                "alternative": alternative,
                "baseline_label": method,
            }
            for method in methods
            if method != "neurodiscovery"
            for comparison, value, alternative in (
                ("same_experiments_gt_hits", 100, "greater"),
                ("same_recall_experiments", 0.1, "less"),
            )
        ],
    )
    external_p_values = final_root / "external_primary_p_values.csv"
    _write_csv(
        external_p_values,
        [
            {
                "dataset": dataset,
                "comparison_type": comparison,
                "budget_or_recall_target": value,
                "baseline_method": method,
                "n_paired_seeds": 10,
                "neurodiscovery_mean": 2,
                "baseline_mean": 1,
                "mean_paired_difference": 1,
                "wilcoxon_statistic": 55,
                "p_value_one_sided": 0.001,
                "alternative": alternative,
                "baseline_label": method,
            }
            for dataset in ("ucla", "cobre", "hcpep", "adhd200", "pooled")
            for method in methods
            if method != "neurodiscovery"
            for comparison, value, alternative in (
                ("same_experiments_confirmed", 100, "greater"),
                ("same_recall_tcp_experiments", 0.1, "less"),
            )
        ],
    )
    final_files = {
        "method_scope.json": {"kind": "json"},
        "generation_quality_primary_summary.csv": {"kind": "csv"},
        "internal_primary_budget_summary.csv": {"kind": "csv"},
        "internal_primary_recall_cost_summary.csv": {"kind": "csv"},
        "internal_primary_p_values.csv": {"kind": "existing"},
        "internal_same_experiments_headline.csv": {"kind": "csv"},
        "internal_same_recall_headline.csv": {"kind": "csv"},
        "external_primary_metrics_summary.csv": {"kind": "csv"},
        "external_primary_recall_cost_summary.csv": {"kind": "csv"},
        "external_primary_p_values.csv": {"kind": "existing"},
        "external_pooled_same_experiments_headline.csv": {"kind": "csv"},
        "external_pooled_same_recall_headline.csv": {"kind": "csv"},
    }
    for name, metadata in final_files.items():
        path = final_root / name
        if metadata["kind"] == "json":
            _write_json(path, {"complete": True})
        elif metadata["kind"] == "csv":
            _write_csv(path, [{"value": 1}])
    final_manifest = final_root / "case1_final_manifest.json"
    _write_json(
        final_manifest,
        {
            "authoritative": True,
            "statistics": {
                "stochastic_repetitions": 10,
                "dispersion": "sample variance across seeds (ddof=1)",
                "paired_test": "one-sided paired Wilcoxon signed-rank test",
            },
            "outputs": {
                name: descriptor(final_root / name) for name in final_files
            },
        },
    )

    case2_manifest = case2 / "manifest.json"
    _write_json(
        case2_manifest,
        {
            "canonical_kg_release": _release(),
            "candidate_count": 2,
            "baseline_rankings_frozen_before_result_access": True,
            "neurodiscovery_outcomes_revealed_batchwise_only": True,
            "family_fdr_definition": (
                "Prespecified exposure-outcome imaging-family BH q<0.05 and both "
                "mediation paths P<0.05"
            ),
            "primary_ranking_endpoint": {
                "metric": "family_fdr_chain_hits",
                "recall_metric": "family_fdr_chain_recall",
                "cumulative_over_frozen_labels": True,
            },
            "neurodiscovery_closed_loop": {
                "complete_chain_required": True,
                "experimental_kg_delta": {
                    "overlay_count": 10,
                    "feedback_consumed_during_ranking": True,
                    "per_seed_isolation_verified": True,
                    "mutates_formal_kg": False,
                },
            },
        },
    )
    _write_csv(
        case2 / "case2_method_metrics_by_trial.csv",
        [
            {
                "method": method,
                "trial": trial,
                "k": k,
                "topk_fdr_chain_hits": hits,
                "family_fdr_chain_hits": hits,
                "family_fdr_chain_recall": hits / 2,
            }
            for method in methods
            for trial in trials
            for k, hits in ((1, 1), (2, 2))
        ],
    )
    _write_csv(
        case2 / "case2_method_metrics_aggregate.csv",
        [{"method": method, "n_trials": 10} for method in methods],
    )
    _write_csv(
        case2 / "case2_paired_comparisons.csv",
        [
            {
                "metric": "family_fdr_chain_hits",
                "k": k,
                "target": "neurodiscovery",
                "baseline": method,
                "n_pairs": 10,
                "target_mean": hits,
                "baseline_mean": hits - 1,
                "mean_difference": 1,
                "difference_variance": 0,
                "p_neurodiscovery_greater_exact_sign_flip": 0.01,
                "p_baseline_greater_exact_sign_flip": 1.0,
                "full_pool_sanity": k == 2,
                "p_neurodiscovery_greater_holm_within_k": 0.06,
                "p_baseline_greater_holm_within_k": 1.0,
            }
            for method in methods
            if method != "neurodiscovery"
            for k, hits in ((1, 1), (2, 2))
        ],
    )
    policy_audit = case2 / "case2_policy_independence_audit.json"
    _write_json(policy_audit, {"passed": True})

    stage_files = {
        "case2_generate": root / "case2_pathway_mediation/hypothesis_generation/hypotheses_raw.json",
        "case2_map": root / "case2_pathway_mediation/mapped_experiment/manifest.json",
        "case2_compare": case2_manifest,
        "case1_internal": internal_manifest,
        "case1_external": external_manifest,
        "case1_finalize": final_manifest,
    }
    for path in stage_files.values():
        if not path.exists():
            _write_json(path, {"complete": True})

    _write_json(
        root / "dedicated_suite_manifest.json",
        {
            "status": "complete",
            "trials": 10,
            "canonical_release": _release(),
            "protocol": {
                "case2_external": (
                    "not applicable: no independent matched cohort is registered"
                )
            },
            "frozen_reuse": {
                "selection_release_precedes_application_release": True,
                "target_release_outcomes_used_for_policy_selection": False,
            },
            "stages": {
                name: _stage_record(path) for name, path in stage_files.items()
            },
        },
    )


def test_runner_resume_and_copy_guards(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    _write_json(suite, {"jobs": [{"status": "complete"}]})
    assert suite_is_complete(suite)[0]
    _write_json(suite, {"jobs": [{"status": "running"}]})
    assert not suite_is_complete(suite)[0]
    _write_json(suite, {"jobs": [{"status": "failed", "task": "x"}]})
    with pytest.raises(RuntimeError):
        suite_is_complete(suite)

    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("frozen", encoding="utf-8")
    copy_verified(source, target)
    record = {"status": "complete", "outputs": {str(target.resolve()): descriptor(target)}}
    assert completed_stage_is_valid(record, (target,))
    target.write_text("changed", encoding="utf-8")
    assert not completed_stage_is_valid(record, (target,))
    with pytest.raises(ValueError):
        copy_verified(source, target)


def test_full_dedicated_audit_passes_and_detects_external_leakage(
    tmp_path: Path,
) -> None:
    _build_passing_suite(tmp_path)
    result = audit(tmp_path)
    assert result["status"] == "passed"
    assert result["failed"] == 0
    inputs: dict[str, dict[str, object]] = {}
    summary = summarize_dedicated_case_studies(
        suite_root=tmp_path,
        inputs=inputs,
    )
    assert summary["ready_for_inference"] is True
    assert summary["internal_complete"] == 2
    assert summary["external_complete"] == 1
    assert summary["external_not_applicable"] == 1
    assert len(summary["tables"]["dedicated_case1_internal_summary.csv"]) == 70
    assert (
        len(summary["tables"]["dedicated_case1_internal_primary_p_values.csv"])
        == 12
    )
    assert (
        len(summary["tables"]["dedicated_case1_external_primary_p_values.csv"])
        == 60
    )
    assert len(summary["tables"]["dedicated_case2_internal_aggregate.csv"]) == 7

    external_manifest = (
        tmp_path
        / "case1_transdiagnostic/external_method_comparison"
        / "case1_external_method_comparison_manifest.json"
    )
    payload = json.loads(external_manifest.read_text(encoding="utf-8"))
    payload["frozen_tcp_rankings"]["external_data_read_before_freeze"] = True
    _write_json(external_manifest, payload)
    result = audit(tmp_path)
    assert result["status"] == "failed"
    assert any(
        row["check"] == "tcp_rankings_frozen_before_external_read"
        for row in result["failed_checks"]
    )

    payload["frozen_tcp_rankings"]["external_data_read_before_freeze"] = False
    _write_json(external_manifest, payload)
    p_values_path = (
        tmp_path
        / "case1_transdiagnostic/final_summary/external_primary_p_values.csv"
    )
    with p_values_path.open(encoding="utf-8") as handle:
        p_values = list(csv.DictReader(handle))
    p_values[0]["p_value_one_sided"] = "1.5"
    _write_csv(p_values_path, p_values)
    result = audit(tmp_path)
    assert result["status"] == "failed"
    assert any(
        row["scope"] == "case1:external"
        and row["check"] == "paired_primary_endpoint_statistics"
        for row in result["failed_checks"]
    )
