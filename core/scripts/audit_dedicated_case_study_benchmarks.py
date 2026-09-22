"""Audit the dedicated CS1 and CS2 benchmark closure without outcome reuse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scripts.canonical_kg_release import CURRENT_CANONICAL_SHA256


SCHEMA = "dedicated-case-study-audit.v1"
EXPECTED_METHODS = {
    "ai_scientist_v2",
    "biomni_native",
    "brainpilot_native",
    "neurodiscovery",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
}
REQUIRED_STAGES = {
    "case2_generate",
    "case2_map",
    "case2_compare",
    "case1_internal",
    "case1_external",
    "case1_finalize",
}
EXTERNAL_DATASETS = {"ucla", "cobre", "hcpep", "adhd200"}
CASE1_FINAL_REQUIRED_OUTPUTS = {
    "method_scope.json",
    "generation_quality_primary_summary.csv",
    "internal_primary_budget_summary.csv",
    "internal_primary_recall_cost_summary.csv",
    "internal_primary_p_values.csv",
    "internal_same_experiments_headline.csv",
    "internal_same_recall_headline.csv",
    "external_primary_metrics_summary.csv",
    "external_primary_recall_cost_summary.csv",
    "external_primary_p_values.csv",
    "external_pooled_same_experiments_headline.csv",
    "external_pooled_same_recall_headline.csv",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_check(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    check: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    rows.append(
        {
            "scope": scope,
            "check": check,
            "passed": bool(passed),
            "observed": json.dumps(observed, ensure_ascii=False, sort_keys=True),
            "expected": json.dumps(expected, ensure_ascii=False, sort_keys=True),
        }
    )


def canonical_hashes(release: dict[str, Any]) -> dict[str, str]:
    files = release.get("files") or {}
    return {
        name: str((files.get(name) or {}).get("sha256", "")).upper()
        for name in CURRENT_CANONICAL_SHA256
    }


def trial_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    trials: dict[str, set[int]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if not method:
            continue
        trials.setdefault(method, set()).add(int(row["trial"]))
    return {method: len(values) for method, values in sorted(trials.items())}


def audit_case1_paired_statistics(
    path: Path,
    *,
    scope: str,
    trials: int,
    comparison_alternatives: dict[str, str],
    datasets: set[str] | None = None,
    allow_unreachable_pairs: bool = False,
) -> dict[str, Any]:
    rows = read_csv(path)
    required_columns = {
        "comparison_type",
        "budget_or_recall_target",
        "baseline_method",
        "n_paired_seeds",
        "mean_paired_difference",
        "wilcoxon_statistic",
        "p_value_one_sided",
        "alternative",
    }
    if datasets is not None:
        required_columns.add("dataset")
    columns = set(rows[0]) if rows else set()
    baselines = {str(row.get("baseline_method") or "") for row in rows}
    comparisons = {str(row.get("comparison_type") or "") for row in rows}
    observed_datasets = (
        {str(row.get("dataset") or "") for row in rows}
        if datasets is not None
        else set()
    )
    row_values_valid = bool(rows)
    for row in rows:
        try:
            n_pairs = int(row.get("n_paired_seeds") or 0)
        except (TypeError, ValueError):
            row_values_valid = False
            break
        permitted_pairs = {trials}
        if allow_unreachable_pairs:
            permitted_pairs.add(0)
        if n_pairs not in permitted_pairs:
            row_values_valid = False
            break
        comparison = str(row.get("comparison_type") or "")
        if row.get("alternative") != comparison_alternatives.get(comparison):
            row_values_valid = False
            break
        p_value = str(row.get("p_value_one_sided") or "").strip()
        if n_pairs == 0:
            if p_value:
                row_values_valid = False
                break
            continue
        try:
            if not 0.0 <= float(p_value) <= 1.0:
                row_values_valid = False
                break
        except (TypeError, ValueError):
            row_values_valid = False
            break

    valid = (
        required_columns <= columns
        and baselines == EXPECTED_METHODS - {"neurodiscovery"}
        and comparisons == set(comparison_alternatives)
        and (datasets is None or observed_datasets == datasets)
        and row_values_valid
    )
    return {
        "scope": scope,
        "valid": valid,
        "rows": len(rows),
        "columns": sorted(columns),
        "baselines": sorted(baselines),
        "comparison_types": sorted(comparisons),
        "datasets": sorted(observed_datasets),
        "paired_test": "one-sided paired Wilcoxon signed-rank test",
    }


def audit_stage_outputs(
    suite: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    stages = suite.get("stages") or {}
    complete = {
        name for name, record in stages.items() if record.get("status") == "complete"
    }
    add_check(
        rows,
        scope="suite",
        check="required_stages_complete",
        passed=REQUIRED_STAGES <= complete,
        observed=sorted(complete),
        expected=sorted(REQUIRED_STAGES),
    )
    for stage_name in sorted(REQUIRED_STAGES & complete):
        for key, descriptor in (stages[stage_name].get("outputs") or {}).items():
            path = Path(str(descriptor.get("path") or key))
            observed = sha256_file(path) if path.is_file() else None
            expected = str(descriptor.get("sha256", "")).upper()
            add_check(
                rows,
                scope=f"stage:{stage_name}",
                check=f"output_hash:{path.name}",
                passed=observed == expected,
                observed=observed,
                expected=expected,
            )


def audit_case1(
    root: Path,
    *,
    trials: int,
    rows: list[dict[str, Any]],
) -> None:
    case_root = root / "case1_transdiagnostic"
    internal_root = case_root / "internal_method_comparison"
    external_root = case_root / "external_method_comparison"
    internal = load_json(internal_root / "case1_method_comparison_manifest.json")
    external = load_json(external_root / "case1_external_method_comparison_manifest.json")
    final = load_json(case_root / "final_summary" / "case1_final_manifest.json")

    for label, manifest in (("internal", internal), ("external", external)):
        observed = canonical_hashes(manifest.get("canonical_kg_release") or {})
        add_check(
            rows,
            scope=f"case1:{label}",
            check="canonical_release",
            passed=observed == CURRENT_CANONICAL_SHA256,
            observed=observed,
            expected=CURRENT_CANONICAL_SHA256,
        )

    internal_counts = trial_counts(
        read_csv(internal_root / "case1_method_summary_by_trial.csv")
    )
    add_check(
        rows,
        scope="case1:internal",
        check="methods_and_trials",
        passed=(set(internal_counts) == EXPECTED_METHODS)
        and all(count == trials for count in internal_counts.values()),
        observed=internal_counts,
        expected={method: trials for method in sorted(EXPECTED_METHODS)},
    )

    external_rows = read_csv(external_root / "case1_external_metrics_by_trial.csv")
    external_counts = trial_counts(external_rows)
    datasets = {str(row.get("dataset", "")) for row in external_rows}
    add_check(
        rows,
        scope="case1:external",
        check="methods_and_trials",
        passed=(set(external_counts) == EXPECTED_METHODS)
        and all(count == trials for count in external_counts.values()),
        observed=external_counts,
        expected={method: trials for method in sorted(EXPECTED_METHODS)},
    )
    add_check(
        rows,
        scope="case1:external",
        check="registered_external_datasets",
        passed=EXTERNAL_DATASETS <= datasets and "pooled" in datasets,
        observed=sorted(datasets),
        expected=sorted(EXTERNAL_DATASETS | {"pooled"}),
    )

    frozen = external.get("frozen_tcp_rankings") or {}
    frozen_trials = (frozen.get("orders") or {}).get("n_trials_by_method") or {}
    add_check(
        rows,
        scope="case1:external",
        check="tcp_rankings_frozen_before_external_read",
        passed=(frozen.get("external_data_read_before_freeze") is False)
        and set(frozen_trials) == EXPECTED_METHODS
        and all(int(value) == trials for value in frozen_trials.values()),
        observed={
            "external_data_read_before_freeze": frozen.get(
                "external_data_read_before_freeze"
            ),
            "n_trials_by_method": frozen_trials,
        },
        expected={
            "external_data_read_before_freeze": False,
            "n_trials_by_method": {
                method: trials for method in sorted(EXPECTED_METHODS)
            },
        },
    )

    final_root = case_root / "final_summary"
    internal_statistics = audit_case1_paired_statistics(
        final_root / "internal_primary_p_values.csv",
        scope="internal",
        trials=trials,
        comparison_alternatives={
            "same_experiments_gt_hits": "greater",
            "same_recall_experiments": "less",
        },
    )
    external_statistics = audit_case1_paired_statistics(
        final_root / "external_primary_p_values.csv",
        scope="external",
        trials=trials,
        comparison_alternatives={
            "same_experiments_confirmed": "greater",
            "same_recall_tcp_experiments": "less",
        },
        datasets=EXTERNAL_DATASETS | {"pooled"},
        allow_unreachable_pairs=True,
    )
    add_check(
        rows,
        scope="case1:internal",
        check="paired_primary_endpoint_statistics",
        passed=bool(internal_statistics["valid"]),
        observed=internal_statistics,
        expected={
            "baselines": sorted(EXPECTED_METHODS - {"neurodiscovery"}),
            "n_pairs": trials,
            "test": "one-sided paired Wilcoxon signed-rank test",
        },
    )
    add_check(
        rows,
        scope="case1:external",
        check="paired_primary_endpoint_statistics",
        passed=bool(external_statistics["valid"]),
        observed=external_statistics,
        expected={
            "baselines": sorted(EXPECTED_METHODS - {"neurodiscovery"}),
            "datasets": sorted(EXTERNAL_DATASETS | {"pooled"}),
            "n_pairs": f"0 or {trials}; zero only when the target is unreachable",
            "test": "one-sided paired Wilcoxon signed-rank test",
        },
    )
    protocol = external.get("protocol") or {}
    add_check(
        rows,
        scope="case1:external",
        check="external_isolation_protocol",
        passed=(protocol.get("external_validation_only") is True)
        and (protocol.get("external_feedback_to_tcp_ranking") is False)
        and set(protocol.get("external_datasets") or ()) == EXTERNAL_DATASETS,
        observed=protocol,
        expected={
            "external_validation_only": True,
            "external_feedback_to_tcp_ranking": False,
            "external_datasets": sorted(EXTERNAL_DATASETS),
        },
    )

    delta = internal.get("experimental_kg_delta") or {}
    add_check(
        rows,
        scope="case1:internal",
        check="closed_loop_overlay_isolated",
        passed=(int(delta.get("overlay_count", 0)) == trials)
        and (delta.get("feedback_consumed_during_ranking") is True)
        and (delta.get("per_seed_isolation_verified") is True)
        and (delta.get("mutates_formal_kg") is False),
        observed={
            key: delta.get(key)
            for key in (
                "overlay_count",
                "feedback_consumed_during_ranking",
                "per_seed_isolation_verified",
                "mutates_formal_kg",
            )
        },
        expected={
            "overlay_count": trials,
            "feedback_consumed_during_ranking": True,
            "per_seed_isolation_verified": True,
            "mutates_formal_kg": False,
        },
    )
    add_check(
        rows,
        scope="case1:final",
        check="authoritative_final_manifest",
        passed=final.get("authoritative") is True,
        observed=final.get("authoritative"),
        expected=True,
    )
    final_outputs = final.get("outputs") or {}
    output_names = set(final_outputs)
    hashes_valid = CASE1_FINAL_REQUIRED_OUTPUTS <= output_names
    invalid_outputs: list[str] = []
    for name in sorted(CASE1_FINAL_REQUIRED_OUTPUTS & output_names):
        item = final_outputs.get(name) or {}
        output_path = Path(str(item.get("path") or final_root / name))
        expected_hash = str(item.get("sha256") or "").upper()
        if not output_path.is_file() or sha256_file(output_path) != expected_hash:
            hashes_valid = False
            invalid_outputs.append(name)
    statistics = final.get("statistics") or {}
    statistics_valid = (
        int(statistics.get("stochastic_repetitions") or 0) == trials
        and statistics.get("dispersion") == "sample variance across seeds (ddof=1)"
        and statistics.get("paired_test")
        == "one-sided paired Wilcoxon signed-rank test"
    )
    add_check(
        rows,
        scope="case1:final",
        check="authoritative_outputs_hashed",
        passed=hashes_valid,
        observed={
            "output_names": sorted(output_names),
            "invalid_outputs": invalid_outputs,
        },
        expected={"required_outputs": sorted(CASE1_FINAL_REQUIRED_OUTPUTS)},
    )
    add_check(
        rows,
        scope="case1:final",
        check="registered_statistical_protocol",
        passed=statistics_valid,
        observed=statistics,
        expected={
            "stochastic_repetitions": trials,
            "dispersion": "sample variance across seeds (ddof=1)",
            "paired_test": "one-sided paired Wilcoxon signed-rank test",
        },
    )


def audit_case2(
    root: Path,
    *,
    trials: int,
    rows: list[dict[str, Any]],
) -> None:
    case_root = root / "case2_pathway_mediation"
    comparison_root = case_root / "comparison"
    manifest = load_json(comparison_root / "manifest.json")
    observed = canonical_hashes(manifest.get("canonical_kg_release") or {})
    add_check(
        rows,
        scope="case2:internal",
        check="canonical_release",
        passed=observed == CURRENT_CANONICAL_SHA256,
        observed=observed,
        expected=CURRENT_CANONICAL_SHA256,
    )

    metrics = read_csv(comparison_root / "case2_method_metrics_by_trial.csv")
    counts = trial_counts(metrics)
    add_check(
        rows,
        scope="case2:internal",
        check="methods_and_trials",
        passed=(set(counts) == EXPECTED_METHODS)
        and all(count == trials for count in counts.values()),
        observed=counts,
        expected={method: trials for method in sorted(EXPECTED_METHODS)},
    )
    required_metric_columns = {
        "method",
        "trial",
        "k",
        "family_fdr_chain_hits",
        "family_fdr_chain_recall",
        "topk_fdr_chain_hits",
    }
    observed_metric_columns = set(metrics[0]) if metrics else set()
    add_check(
        rows,
        scope="case2:internal",
        check="primary_endpoint_columns",
        passed=required_metric_columns <= observed_metric_columns,
        observed=sorted(observed_metric_columns),
        expected=sorted(required_metric_columns),
    )

    endpoint = manifest.get("primary_ranking_endpoint") or {}
    primary_metric = str(endpoint.get("metric") or "")
    family_definition = str(manifest.get("family_fdr_definition") or "")
    add_check(
        rows,
        scope="case2:internal",
        check="primary_endpoint_is_prespecified_family_fdr",
        passed=(primary_metric == "family_fdr_chain_hits")
        and ("family" in family_definition.lower())
        and ("q<0.05" in family_definition.replace(" ", "")),
        observed={
            "metric": primary_metric,
            "definition": family_definition,
            "explicit_in_manifest": bool(endpoint),
        },
        expected={
            "metric": "family_fdr_chain_hits",
            "definition": "prespecified family BH q<0.05 with complete paths",
        },
    )

    monotonic = True
    monotonic_failures: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    if required_metric_columns <= observed_metric_columns:
        for metric_row in metrics:
            key = (str(metric_row["method"]), int(metric_row["trial"]))
            grouped.setdefault(key, []).append(
                (int(metric_row["k"]), int(metric_row["family_fdr_chain_hits"]))
            )
        for (method, trial), values in sorted(grouped.items()):
            ordered = sorted(values)
            hits = [value for _, value in ordered]
            if any(right < left for left, right in zip(hits, hits[1:])):
                monotonic = False
                monotonic_failures.append(
                    {"method": method, "trial": trial, "values": ordered}
                )
    else:
        monotonic = False
    add_check(
        rows,
        scope="case2:internal",
        check="family_fdr_discovery_curve_is_cumulative",
        passed=monotonic,
        observed=monotonic_failures,
        expected="non-decreasing family_fdr_chain_hits within every method/trial",
    )

    paired_path = comparison_root / "case2_paired_comparisons.csv"
    paired = read_csv(paired_path) if paired_path.is_file() else []
    paired_columns = set(paired[0]) if paired else set()
    required_paired_columns = {
        "metric",
        "k",
        "target",
        "baseline",
        "n_pairs",
        "mean_difference",
        "difference_variance",
        "p_neurodiscovery_greater_exact_sign_flip",
        "p_neurodiscovery_greater_holm_within_k",
        "full_pool_sanity",
    }
    expected_baselines = EXPECTED_METHODS - {"neurodiscovery"}
    observed_baselines = {str(row.get("baseline", "")) for row in paired}
    paired_valid = bool(paired) and required_paired_columns <= paired_columns
    if paired_valid:
        paired_valid = (
            observed_baselines == expected_baselines
            and all(row.get("metric") == "family_fdr_chain_hits" for row in paired)
            and all(row.get("target") == "neurodiscovery" for row in paired)
            and all(int(row.get("n_pairs", 0)) == trials for row in paired)
            and all(
                0.0 <= float(row["p_neurodiscovery_greater_exact_sign_flip"]) <= 1.0
                and 0.0
                <= float(row["p_neurodiscovery_greater_holm_within_k"])
                <= 1.0
                for row in paired
            )
        )
    add_check(
        rows,
        scope="case2:internal",
        check="paired_primary_endpoint_statistics",
        passed=paired_valid,
        observed={
            "path_exists": paired_path.is_file(),
            "rows": len(paired),
            "columns": sorted(paired_columns),
            "baselines": sorted(observed_baselines),
        },
        expected={
            "metric": "family_fdr_chain_hits",
            "baselines": sorted(expected_baselines),
            "n_pairs": trials,
            "tests": "exact sign-flip with Holm correction within K",
        },
    )
    policy_audit = load_json(
        comparison_root / "case2_policy_independence_audit.json"
    )
    add_check(
        rows,
        scope="case2:internal",
        check="baseline_policy_independence",
        passed=policy_audit.get("passed") is True,
        observed=policy_audit,
        expected={"passed": True},
    )

    closed = manifest.get("neurodiscovery_closed_loop") or {}
    delta = closed.get("experimental_kg_delta") or {}
    observed_protocol = {
        "baseline_rankings_frozen_before_result_access": manifest.get(
            "baseline_rankings_frozen_before_result_access"
        ),
        "neurodiscovery_outcomes_revealed_batchwise_only": manifest.get(
            "neurodiscovery_outcomes_revealed_batchwise_only"
        ),
        "complete_chain_required": closed.get("complete_chain_required"),
        "feedback_consumed_during_ranking": delta.get(
            "feedback_consumed_during_ranking"
        ),
        "per_seed_isolation_verified": delta.get("per_seed_isolation_verified"),
        "mutates_formal_kg": delta.get("mutates_formal_kg"),
        "overlay_count": delta.get("overlay_count"),
    }
    expected_protocol = {
        "baseline_rankings_frozen_before_result_access": True,
        "neurodiscovery_outcomes_revealed_batchwise_only": True,
        "complete_chain_required": True,
        "feedback_consumed_during_ranking": True,
        "per_seed_isolation_verified": True,
        "mutates_formal_kg": False,
        "overlay_count": trials,
    }
    add_check(
        rows,
        scope="case2:internal",
        check="closed_loop_and_no_leakage_protocol",
        passed=observed_protocol == expected_protocol,
        observed=observed_protocol,
        expected=expected_protocol,
    )


def audit(root: Path) -> dict[str, Any]:
    suite_path = root / "dedicated_suite_manifest.json"
    suite = load_json(suite_path)
    trials = int(suite.get("trials", 0))
    rows: list[dict[str, Any]] = []

    observed = canonical_hashes(suite.get("canonical_release") or {})
    add_check(
        rows,
        scope="suite",
        check="canonical_release",
        passed=observed == CURRENT_CANONICAL_SHA256,
        observed=observed,
        expected=CURRENT_CANONICAL_SHA256,
    )
    add_check(
        rows,
        scope="suite",
        check="registered_trials",
        passed=trials == 10,
        observed=trials,
        expected=10,
    )
    reuse = suite.get("frozen_reuse") or {}
    add_check(
        rows,
        scope="suite",
        check="cross_release_selection_isolation",
        passed=(reuse.get("selection_release_precedes_application_release") is True)
        and (reuse.get("target_release_outcomes_used_for_policy_selection") is False),
        observed={
            "selection_release_precedes_application_release": reuse.get(
                "selection_release_precedes_application_release"
            ),
            "target_release_outcomes_used_for_policy_selection": reuse.get(
                "target_release_outcomes_used_for_policy_selection"
            ),
        },
        expected={
            "selection_release_precedes_application_release": True,
            "target_release_outcomes_used_for_policy_selection": False,
        },
    )
    add_check(
        rows,
        scope="case2:external",
        check="external_status_explicit",
        passed=str((suite.get("protocol") or {}).get("case2_external", "")).startswith(
            "not applicable"
        ),
        observed=(suite.get("protocol") or {}).get("case2_external"),
        expected="not applicable",
    )
    audit_stage_outputs(suite, rows)
    audit_case1(root, trials=trials, rows=rows)
    audit_case2(root, trials=trials, rows=rows)

    failed = [row for row in rows if not row["passed"]]
    payload = {
        "schema_version": SCHEMA,
        "created_at": utc_now(),
        "status": "passed" if not failed else "failed",
        "suite_root": str(root.resolve()),
        "suite_manifest": str(suite_path.resolve()),
        "suite_manifest_sha256": sha256_file(suite_path),
        "checks": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "failed_checks": failed,
        "protocol_scope": {
            "case1": "internal TCP plus external UCLA/COBRE/HCP-EP/ADHD200",
            "case2": "internal ADNI mediation; external not applicable",
        },
        "case2_primary_endpoint": {
            "metric": "family_fdr_chain_hits",
            "recall_metric": "family_fdr_chain_recall",
            "topk_fdr_chain_hits_is_primary": False,
            "reason": (
                "family-FDR labels are prespecified and frozen across budgets; "
                "Top-K-local BH counts may decrease when K changes"
            ),
        },
    }
    audit_root = root / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    with (audit_root / "dedicated_audit_checks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (audit_root / "dedicated_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(args.suite_root)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
