"""Independently audit the latest-KG Case Study 1 formal result bundle.

This audit intentionally consumes only already-frozen policies, rankings, and
result tables.  It does not regenerate hypotheses or rankings and does not
open the original internal or external outcome tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHODS = {
    "neurodiscovery",
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
}
BASELINES = METHODS - {"neurodiscovery"}
SEEDS = {0, 1, 2}
BUDGETS = {5_000, 10_000, 50_000, 100_000, 200_000}
RECALL_TARGETS = {0.01, 0.05, 0.10, 0.20, 0.30, 0.50}
SEALED_KG = "2c02732582da9907c68300d791c3560b8e66cc268d987d453e7c971bd6cdeff5"
SEALED_CLAIMS = "705b079989fdec3d7756cf737f12b41756eb8058f8ed66a009d6f832489f5ba4"
SEALED_STATE = "744b75718b2beebfdaf9055595cb2161ed841643acd58f365f55717bc7b5360e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-8)


def unique_keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[Any, ...]]:
    return {
        tuple(value.item() if hasattr(value, "item") else value for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    }


def validate_summary(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    group_columns: list[str],
    raw_metric: str,
    summary_mean: str,
    summary_sd: str,
    n_column: str,
) -> bool:
    grouped = raw.groupby(group_columns, sort=False)[raw_metric].agg(
        ["count", "mean", "std"]
    )
    indexed = summary.set_index(group_columns)
    if set(grouped.index) != set(indexed.index):
        return False
    for key, row in grouped.iterrows():
        summary_row = indexed.loc[key]
        if isinstance(summary_row, pd.DataFrame):
            return False
        if int(summary_row[n_column]) != int(row["count"]):
            return False
        if not close(summary_row[summary_mean], row["mean"]):
            return False
        if not close(summary_row[summary_sd], row["std"]):
            return False
    return True


def verify_artifact_record(record: dict[str, Any]) -> bool:
    path = Path(str(record.get("path", "")))
    return (
        path.is_file()
        and int(record.get("bytes", path.stat().st_size)) == path.stat().st_size
        and sha256_file(path).lower() == str(record.get("sha256", "")).lower()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    baseline_dir = root / "baseline_generation" / "frozen_18_policies"
    formal_root = root / "neurodiscovery_formal"
    evaluation_dir = formal_root / "evaluation"
    external_dir = root / "external_evaluation"
    output = args.output or (
        root / "final_audit" / "case1_formal_result_audit.json"
    )

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": bool(passed), **details}

    baseline_manifest_path = baseline_dir / "frozen_baseline_policy_manifest.json"
    baseline_manifest = load_json(baseline_manifest_path)
    policy_path = baseline_dir / "case1_search_policies.jsonl"
    policy_keys: list[tuple[str, int]] = []
    with policy_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            policy_keys.append((str(payload["method"]), int(payload["trial"])))
    expected_baseline_keys = {
        (method, seed) for method in BASELINES for seed in SEEDS
    }
    record(
        "baseline_matrix_exact",
        baseline_manifest.get("status") == "frozen"
        and baseline_manifest.get("outcome_tables_opened") is False
        and int(baseline_manifest["matrix"]["observed_policy_count"]) == 18
        and len(policy_keys) == len(set(policy_keys)) == 18
        and set(policy_keys) == expected_baseline_keys,
        observed_policy_count=len(policy_keys),
    )
    combined = baseline_manifest["combined_policy_artifact"]
    record(
        "baseline_policy_hash",
        verify_artifact_record(combined)
        and Path(combined["path"]).resolve() == policy_path.resolve(),
        sha256=sha256_file(policy_path),
    )
    identities = baseline_manifest["sealed_identities"]
    record(
        "baseline_sealed_kg_and_registry",
        str(identities["knowledge_graph_sha256"]).lower() == SEALED_KG
        and bool(identities.get("public_registry_sha256"))
        and bool(identities.get("official_adapter_sha256")),
        knowledge_graph_sha256=str(identities["knowledge_graph_sha256"]).lower(),
        public_registry_sha256=str(identities["public_registry_sha256"]).lower(),
    )

    structural_path = formal_root / "structural_audit" / "three_seed_closed_loop_audit.json"
    structural = load_json(structural_path)
    seed_audits = structural.get("seed_audits") or []
    per_seed_integrity = all(
        audit.get("status") == "passed"
        and int(audit.get("commit_count", -1)) == 800
        and int(audit.get("nonzero_feedback_reads", -1)) == 799
        and int(audit.get("selection_changed_batches", -1)) == 799
        and int(audit.get("committed_prefix_records", -1)) == 200_000
        and all(bool(value) for value in (audit.get("checks") or {}).values())
        for audit in seed_audits
    )
    record(
        "formal_true_closed_loop",
        structural.get("status") == "passed"
        and structural.get("classification") == "formal"
        and structural.get("diagnostic_only") is False
        and structural.get("performance_evaluated") is False
        and structural.get("gt_artifact_resolved_or_opened_by_audit") is False
        and set(int(seed) for seed in structural.get("seeds", [])) == SEEDS
        and all(bool(value) for value in structural["cross_seed_checks"].values())
        and per_seed_integrity,
        ranking_sha256=[audit.get("frozen_ranking_sha256") for audit in seed_audits],
    )

    evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
    evaluation_manifest = load_json(evaluation_manifest_path)
    closed_loop = load_json(evaluation_dir / "closed_loop_audit.json")
    record(
        "internal_evaluation_manifest",
        evaluation_manifest.get("status") == "complete_sota"
        and set(evaluation_manifest.get("methods", [])) == METHODS
        and set(int(seed) for seed in evaluation_manifest.get("seeds", [])) == SEEDS
        and evaluation_manifest.get("ranking_freeze_preceded_gt_load") is True
        and evaluation_manifest.get("candidate_count") == 426_555
        and evaluation_manifest.get("gt_total") == 4_263
        and closed_loop.get("status") == "passed"
        and closed_loop.get("per_seed_isolation") is True
        and closed_loop.get("gt_opened_only_after_rankings_frozen") is True
        and closed_loop.get("external_outcomes_used_during_ranking") is False,
    )
    internal_artifacts_ok = all(
        verify_artifact_record(record_payload)
        for record_payload in evaluation_manifest.get("artifacts", {}).values()
    )
    record(
        "internal_artifact_hashes",
        internal_artifacts_ok,
        artifact_count=len(evaluation_manifest.get("artifacts", {})),
    )

    internal_curves = pd.read_csv(evaluation_dir / "internal_curves_by_seed.csv")
    internal_curve_summary = pd.read_csv(
        evaluation_dir / "internal_curves_mean_sample_sd.csv"
    )
    internal_costs = pd.read_csv(
        evaluation_dir / "internal_recall_cost_by_seed.csv"
    )
    internal_cost_summary = pd.read_csv(
        evaluation_dir / "internal_recall_cost_mean_sample_sd.csv"
    )
    expected_curve_keys = {
        (method, seed, budget)
        for method in METHODS
        for seed in SEEDS
        for budget in BUDGETS
    }
    expected_cost_keys = {
        (method, seed, target)
        for method in METHODS
        for seed in SEEDS
        for target in RECALL_TARGETS
    }
    record(
        "internal_seed_matrix_exact",
        unique_keys(internal_curves, ["method", "seed", "budget"])
        == expected_curve_keys
        and len(internal_curves) == 105
        and unique_keys(internal_costs, ["method", "seed", "recall_target"])
        == expected_cost_keys
        and len(internal_costs) == 126,
        curve_rows=len(internal_curves),
        recall_cost_rows=len(internal_costs),
    )
    record(
        "internal_mean_and_sample_sd_recomputed",
        validate_summary(
            internal_curves,
            internal_curve_summary,
            group_columns=["method", "budget"],
            raw_metric="gt_hits",
            summary_mean="gt_hits_mean",
            summary_sd="gt_hits_sd",
            n_column="n_seeds",
        )
        and validate_summary(
            internal_costs,
            internal_cost_summary,
            group_columns=["method", "recall_target"],
            raw_metric="experiments_required",
            summary_mean="experiments_required_mean",
            summary_sd="experiments_required_sd",
            n_column="n_seeds",
        ),
        ddof=1,
    )

    internal_gate = load_json(evaluation_dir / "sota_gate.json")
    internal_yield_passes = 0
    for budget in sorted(BUDGETS):
        endpoint = internal_curve_summary[internal_curve_summary["budget"] == budget]
        nd = float(endpoint.loc[endpoint["method"] == "neurodiscovery", "gt_hits_mean"].iloc[0])
        best = float(endpoint.loc[endpoint["method"] != "neurodiscovery", "gt_hits_mean"].max())
        gate = internal_gate["yield_gates"][str(budget)]
        passed = nd > best and close(gate["neurodiscovery_mean"], nd) and close(gate["best_baseline_mean"], best) and gate["passed"] is True
        internal_yield_passes += int(passed)
    internal_cost_passes = 0
    for target in sorted(RECALL_TARGETS):
        endpoint = internal_cost_summary[np.isclose(internal_cost_summary["recall_target"], target)]
        nd = float(endpoint.loc[endpoint["method"] == "neurodiscovery", "experiments_required_mean"].iloc[0])
        best = float(endpoint.loc[endpoint["method"] != "neurodiscovery", "experiments_required_mean"].min())
        gate = internal_gate["recall_cost_gates"][f"{target:.2f}"]
        passed = nd < best and close(gate["neurodiscovery_mean"], nd) and close(gate["best_baseline_mean"], best) and gate["passed"] is True
        internal_cost_passes += int(passed)
    record(
        "internal_sota_gate_recomputed",
        internal_yield_passes == 5
        and internal_cost_passes == 6
        and internal_gate.get("sota_achieved") is True,
        yield_gates_passed=internal_yield_passes,
        recall_cost_gates_passed=internal_cost_passes,
    )

    external_manifest_path = external_dir / "case1_external_method_comparison_manifest.json"
    external_manifest = load_json(external_manifest_path)
    canonical = external_manifest["canonical_kg_release"]
    canonical_files = canonical["files"]
    frozen = external_manifest["frozen_tcp_rankings"]
    record(
        "external_manifest_and_blinding",
        external_manifest.get("trials") == 3
        and external_manifest.get("seed_base") == 260_810
        and set(external_manifest.get("methods", [])) == METHODS
        and external_manifest["protocol"]["external_validation_only"] is True
        and external_manifest["protocol"]["external_feedback_to_tcp_ranking"] is False
        and frozen.get("external_data_read_before_freeze") is False
        and frozen.get("neurodiscovery_order_materialization")
        == "formal_commitment_audited_frozen_rankings"
        and int(frozen["orders"]["n_orders"]) == 21
        and set(int(value) for value in frozen["orders"]["n_trials_by_method"].values()) == {3},
        frozen_order_count=int(frozen["orders"]["n_orders"]),
        pooled_external_gt=int(external_manifest["external_gt_totals"]["pooled"]),
    )
    record(
        "external_sealed_kg",
        str(canonical_files["knowledge_graph"]["sha256"]).lower() == SEALED_KG
        and str(canonical_files["extracted_claims"]["sha256"]).lower() == SEALED_CLAIMS
        and str(canonical_files["current_state"]["sha256"]).lower() == SEALED_STATE
        and str(external_manifest["search_policies_sha256"]).lower()
        == sha256_file(policy_path).lower(),
        knowledge_graph_sha256=str(canonical_files["knowledge_graph"]["sha256"]).lower(),
        extracted_claims_sha256=str(canonical_files["extracted_claims"]["sha256"]).lower(),
        current_state_sha256=str(canonical_files["current_state"]["sha256"]).lower(),
    )
    record(
        "frozen_external_ranking_artifacts",
        verify_artifact_record(frozen["candidate_table"])
        and verify_artifact_record(frozen["orders"])
        and sha256_file(Path(frozen["manifest_path"]))
        == frozen["manifest_sha256"],
    )

    external_metrics = pd.read_csv(external_dir / "case1_external_metrics_by_trial.csv")
    external_metric_summary = pd.read_csv(
        external_dir / "case1_external_metrics_summary.csv"
    )
    external_costs = pd.read_csv(
        external_dir / "case1_external_recall_cost_by_trial.csv"
    )
    external_cost_summary = pd.read_csv(
        external_dir / "case1_external_recall_cost_summary.csv"
    )
    pooled_metrics = external_metrics[
        (external_metrics["dataset"] == "pooled")
        & (external_metrics["scope"] == "tcp_budget")
    ].copy()
    pooled_metric_summary = external_metric_summary[
        (external_metric_summary["dataset"] == "pooled")
        & (external_metric_summary["scope"] == "tcp_budget")
    ].copy()
    pooled_costs = external_costs[external_costs["dataset"] == "pooled"].copy()
    pooled_cost_summary = external_cost_summary[
        external_cost_summary["dataset"] == "pooled"
    ].copy()
    record(
        "external_seed_matrix_exact",
        unique_keys(pooled_metrics, ["method", "trial", "budget"])
        == expected_curve_keys
        and len(pooled_metrics) == 105
        and unique_keys(pooled_costs, ["method", "trial", "recall_target"])
        == expected_cost_keys
        and len(pooled_costs) == 126,
        curve_rows=len(pooled_metrics),
        recall_cost_rows=len(pooled_costs),
    )
    record(
        "external_mean_and_sample_sd_recomputed",
        validate_summary(
            pooled_metrics,
            pooled_metric_summary,
            group_columns=["dataset", "method", "scope", "budget"],
            raw_metric="n_confirmed",
            summary_mean="n_confirmed_mean",
            summary_sd="n_confirmed_sd",
            n_column="n_trials",
        )
        and validate_summary(
            pooled_costs,
            pooled_cost_summary,
            group_columns=["dataset", "method", "recall_target"],
            raw_metric="tcp_experiments_required",
            summary_mean="tcp_experiments_required_mean",
            summary_sd="tcp_experiments_required_sd",
            n_column="n_trials",
        ),
        ddof=1,
    )

    external_gate_path = external_dir / "external_sota_gate.json"
    external_gate = load_json(external_gate_path)
    external_yield_passes = 0
    for budget in sorted(BUDGETS):
        endpoint = pooled_metric_summary[pooled_metric_summary["budget"] == budget]
        nd = float(endpoint.loc[endpoint["method"] == "neurodiscovery", "n_confirmed_mean"].iloc[0])
        best = float(endpoint.loc[endpoint["method"] != "neurodiscovery", "n_confirmed_mean"].max())
        gate = external_gate["yield_gates"][str(budget)]
        passed = nd > best and close(gate["neurodiscovery_mean"], nd) and close(gate["best_baseline_mean"], best) and gate["passed"] is True
        external_yield_passes += int(passed)
    external_cost_passes = 0
    for target in sorted(RECALL_TARGETS):
        endpoint = pooled_cost_summary[np.isclose(pooled_cost_summary["recall_target"], target)]
        nd = float(endpoint.loc[endpoint["method"] == "neurodiscovery", "tcp_experiments_required_mean"].iloc[0])
        best = float(endpoint.loc[endpoint["method"] != "neurodiscovery", "tcp_experiments_required_mean"].min())
        gate = external_gate["recall_cost_gates"][f"{target:.2f}"]
        passed = nd < best and close(gate["neurodiscovery_mean"], nd) and close(gate["best_baseline_mean"], best) and gate["passed"] is True
        external_cost_passes += int(passed)
    record(
        "external_sota_gate_recomputed",
        external_yield_passes == 5
        and external_cost_passes == 6
        and external_gate.get("external_sota_achieved") is True,
        yield_gates_passed=external_yield_passes,
        recall_cost_gates_passed=external_cost_passes,
    )

    overall_path = external_dir / "overall_sota_gate.json"
    overall = load_json(overall_path)
    evidence = overall["internal_evidence"]
    record(
        "overall_sota_evidence",
        overall.get("internal_sota_achieved") is True
        and overall.get("external_sota_achieved") is True
        and overall.get("overall_sota_achieved") is True
        and sha256_file(evaluation_manifest_path)
        == evidence["evaluation_manifest"]["sha256"]
        and sha256_file(evaluation_dir / "sota_gate.json")
        == evidence["internal_sota_gate"]["sha256"],
    )

    key_artifacts = {
        "baseline_policy_manifest": baseline_manifest_path,
        "baseline_policies": policy_path,
        "formal_design": formal_root / "formal_design.json",
        "structural_audit": structural_path,
        "internal_evaluation_manifest": evaluation_manifest_path,
        "internal_curves_by_seed": evaluation_dir / "internal_curves_by_seed.csv",
        "internal_curve_summary": evaluation_dir / "internal_curves_mean_sample_sd.csv",
        "internal_recall_cost_by_seed": evaluation_dir / "internal_recall_cost_by_seed.csv",
        "internal_recall_cost_summary": evaluation_dir / "internal_recall_cost_mean_sample_sd.csv",
        "internal_sota_gate": evaluation_dir / "sota_gate.json",
        "external_ranking_manifest": Path(frozen["manifest_path"]),
        "external_metrics_by_trial": external_dir / "case1_external_metrics_by_trial.csv",
        "external_metrics_summary": external_dir / "case1_external_metrics_summary.csv",
        "external_recall_cost_by_trial": external_dir / "case1_external_recall_cost_by_trial.csv",
        "external_recall_cost_summary": external_dir / "case1_external_recall_cost_summary.csv",
        "external_sota_gate": external_gate_path,
        "overall_sota_gate": overall_path,
        "external_manifest": external_manifest_path,
    }
    artifact_records = {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in key_artifacts.items()
    }

    failed_checks = [name for name, value in checks.items() if not value["passed"]]
    payload = {
        "schema_version": "case1-formal-result-independent-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed" if not failed_checks else "failed",
        "run_root": str(root),
        "experiment_execution": {
            "methods": sorted(METHODS),
            "seeds": sorted(SEEDS),
            "method_seed_runs_completed": 21,
            "method_seed_runs_expected": 21,
        },
        "sota_summary": {
            "internal_yield_gates_passed": internal_yield_passes,
            "internal_recall_cost_gates_passed": internal_cost_passes,
            "external_yield_gates_passed": external_yield_passes,
            "external_recall_cost_gates_passed": external_cost_passes,
            "internal_sota_achieved": internal_gate.get("sota_achieved") is True,
            "external_sota_achieved": external_gate.get("external_sota_achieved") is True,
            "overall_sota_achieved": overall.get("overall_sota_achieved") is True,
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "key_artifacts": artifact_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failed_checks:
        raise RuntimeError(f"Case Study 1 formal audit failed: {failed_checks}")


if __name__ == "__main__":
    main()
