"""Build a read-only report over validation and hindcasting benchmarks.

The report never regenerates candidates, reads labels during ranking, or mutates
an experiment directory. It preserves negative endpoints and incomplete formal
runs as explicit states instead of silently dropping them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_METHOD = "neurodiscovery"
FORMAL_EXTERNAL_VALIDATION_CASE_STUDIES = ("case1_transdiagnostic",)
GENERIC_EXTERNAL_RESULTS_ROLE = "exploratory_only_not_completion_gate"
GENERIC_FORMAL_VALIDATION_SCOPES = ("internal",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted P values in the original order."""

    count = len(p_values)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(sorted(range(count), key=lambda item: p_values[item])):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def holm_adjust_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    p_field: str,
    output_field: str = "p_holm_within_endpoint",
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        families[tuple(row.get(field) for field in group_fields)].append(index)
    for indices in families.values():
        adjusted = holm_adjust([as_float(output[index][p_field], default=1.0) for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            output[index][output_field] = value
            output[index]["holm_family_size"] = len(indices)
    return output


def exact_paired_sign_flip_p(differences: Sequence[float]) -> float:
    """Exact one-sided paired randomization P value for a positive mean."""

    values = [float(value) for value in differences if math.isfinite(float(value))]
    if not values:
        return float("nan")
    if len(values) > 20:
        raise ValueError("exact sign-flip test is limited to 20 paired observations")
    observed = sum(values) / len(values)
    extreme = 0
    permutations = 1 << len(values)
    tolerance = 1e-12
    for mask in range(permutations):
        permuted = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(values)
        ) / len(values)
        extreme += permuted >= observed - tolerance
    return extreme / permutations


def sample_variance(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0
    mean = sum(finite) / len(finite)
    return sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)


def build_endpoint_evidence_rows(
    budget_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach reliability and multiplicity-aware interpretation to each endpoint."""

    paired_index: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for row in paired_rows:
        if row.get("comparison") != "same_experiments_hits":
            continue
        key = (
            str(row.get("task") or ""),
            str(row.get("scope") or ""),
            as_int(row.get("value")),
            str(row.get("baseline") or ""),
        )
        paired_index[key] = row

    output: list[dict[str, Any]] = []
    for source in budget_rows:
        row = dict(source)
        gt_total = as_float(row.get("gt_total"))
        candidate_count = as_int(row.get("candidate_count"))
        gt_density = gt_total / candidate_count if candidate_count > 0 else 0.0
        if gt_total <= 0:
            reliability = "not_evaluable_no_validated_discoveries"
            caveat = "The registered endpoint contains no validated discoveries."
        elif gt_total < 10:
            reliability = "sparse_lt10_validated_discoveries"
            caveat = "Fewer than 10 validated discoveries make method ordering unstable."
        elif gt_density >= 0.8:
            reliability = "saturated_gt_density_ge_0.8"
            caveat = "At least 80% of the candidate pool is positive, limiting ranking resolution."
        else:
            reliability = "informative"
            caveat = ""

        best_methods = [
            method
            for method in str(row.get("best_comparator_methods") or "").split(";")
            if method
        ]
        p_values: list[float] = []
        missing: list[str] = []
        for method in best_methods:
            paired = paired_index.get(
                (
                    str(row.get("task") or ""),
                    str(row.get("scope") or ""),
                    as_int(row.get("experiments")),
                    method,
                )
            )
            if paired is None:
                missing.append(method)
            else:
                p_values.append(
                    as_float(paired.get("p_holm_within_endpoint"), default=1.0)
                )

        state = str(row.get("state") or "")
        supported_win = (
            state == "neurodiscovery_strict_win"
            and reliability == "informative"
            and bool(best_methods)
            and not missing
            and bool(p_values)
            and max(p_values) <= 0.05
        )
        if supported_win:
            inference = "holm_supported_win"
        elif state == "neurodiscovery_strict_win":
            inference = "descriptive_win_only"
        elif state == "neurodiscovery_loss":
            inference = "descriptive_loss_reverse_test_not_registered"
        elif state == "neurodiscovery_tie":
            inference = "tie"
        else:
            inference = state

        output.append(
            {
                **row,
                "gt_density": gt_density,
                "endpoint_reliability": reliability,
                "best_comparator_p_holm_max": max(p_values) if p_values else "",
                "best_comparator_p_holm_values": ";".join(map(str, p_values)),
                "missing_paired_comparators": ";".join(missing),
                "inference_state": inference,
                "scientific_win_claim_permitted": supported_win,
                "caveat": caveat,
            }
        )
    return output


def build_task_diagnostic_rows(
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Classify the largest non-full endpoint without prescribing target tuning."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in endpoint_rows:
        if row.get("state") == "full_pool_sanity":
            continue
        key = (str(row.get("task") or ""), str(row.get("scope") or ""))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (task, scope), rows in sorted(grouped.items()):
        row = max(rows, key=lambda item: as_int(item.get("experiments")))
        reliability = str(row.get("endpoint_reliability") or "")
        inference = str(row.get("inference_state") or "")
        if reliability == "not_evaluable_no_validated_discoveries":
            diagnosis = "validation_endpoint_bottleneck"
            next_action = "expand or redesign the registered validation endpoint"
        elif reliability == "sparse_lt10_validated_discoveries":
            diagnosis = "low_positive_count_bottleneck"
            next_action = "increase cohort/readout coverage before ranking-policy conclusions"
        elif reliability == "saturated_gt_density_ge_0.8":
            diagnosis = "saturated_candidate_space_bottleneck"
            next_action = "increase candidate resolution or use a more discriminating endpoint"
        elif inference == "holm_supported_win":
            diagnosis = "supported_neurodiscovery_advantage"
            next_action = "retain the frozen policy and test transportability"
        elif inference.startswith("descriptive_loss"):
            diagnosis = "potential_ranking_policy_bottleneck"
            next_action = "investigate only on an independent development release"
        elif inference == "tie":
            diagnosis = "no_method_separation"
            next_action = "seek a higher-resolution endpoint before policy changes"
        else:
            diagnosis = "descriptive_or_inconclusive"
            next_action = "retain as descriptive evidence pending independent validation"
        output.append(
            {
                "task": task,
                "scope": scope,
                "reference_budget": row.get("experiments"),
                "candidate_count": row.get("candidate_count"),
                "gt_total": row.get("gt_total"),
                "gt_density": row.get("gt_density"),
                "neurodiscovery_hits_mean": row.get("neurodiscovery_hits_mean"),
                "best_comparator_hits_mean": row.get("best_comparator_hits_mean"),
                "best_comparator_methods": row.get("best_comparator_methods"),
                "endpoint_reliability": reliability,
                "inference_state": inference,
                "diagnosis": diagnosis,
                "next_action_without_target_leakage": next_action,
                "target_specific_tuning_permitted": False,
            }
        )
    return output


def endpoint_state(
    reference: float,
    best_comparator: float,
    *,
    gt_total: float,
    full_pool: bool = False,
    lower_is_better: bool = False,
) -> str:
    if gt_total <= 0:
        return "no_validated_discoveries"
    if full_pool:
        return "full_pool_sanity"
    if reference == 0 and best_comparator == 0:
        return "no_method_hits_at_endpoint"
    difference = best_comparator - reference if lower_is_better else reference - best_comparator
    if difference > 1e-12:
        return "neurodiscovery_strict_win"
    if difference < -1e-12:
        return "neurodiscovery_loss"
    return "neurodiscovery_tie"


def _best_rows(
    rows: Sequence[Mapping[str, Any]], *, value_field: str, lower_is_better: bool
) -> tuple[float, list[str]]:
    comparators = [row for row in rows if row.get("method") != REFERENCE_METHOD]
    if not comparators:
        raise ValueError("benchmark contains no comparator method")
    values = [as_float(row.get(value_field)) for row in comparators]
    best = min(values) if lower_is_better else max(values)
    methods = sorted(
        str(row["method"])
        for row in comparators
        if abs(as_float(row.get(value_field)) - best) <= 1e-12
    )
    return best, methods


def _record_input(path: Path, inputs: dict[str, dict[str, Any]]) -> None:
    key = str(path.resolve())
    if key not in inputs:
        inputs[key] = file_descriptor(path)


def _resolve_reference_path(value: Any, *, anchor: Path) -> Path:
    path = Path(str(value or ""))
    if not str(path):
        raise ValueError("referenced input path is empty")
    if path.is_absolute():
        return path
    candidates = (path, *(parent / path for parent in (anchor.parent, *anchor.parents)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (anchor.parent / path).resolve()


def _verified_reference(
    record: Mapping[str, Any] | str,
    *,
    anchor: Path,
    inputs: dict[str, dict[str, Any]],
    label: str,
) -> Path:
    if isinstance(record, Mapping):
        value = record.get("path")
        expected = str(record.get("sha256") or "").upper()
    else:
        value = record
        expected = ""
    path = _resolve_reference_path(value, anchor=anchor)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    observed = sha256_file(path)
    if expected and observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} observed={observed}"
        )
    _record_input(path, inputs)
    return path


def summarize_executability_coverage(
    *,
    static_design_path: Path,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify and summarize the method-blind 17-Case-Study coverage lock."""

    design = load_json(static_design_path)
    formal_count = as_int((design.get("primary_matrix") or {}).get("formal_case_studies"))
    eligibility_record = (design.get("temporal_inputs") or {}).get(
        "eligibility_manifest"
    )
    if not eligibility_record:
        if formal_count:
            raise ValueError(
                "formal static design declares Case Studies without an eligibility lock"
            )
        return {
            "status": "not_requested",
            "ready_for_inference": True,
            "tables": {},
        }

    eligibility_path = _verified_reference(
        eligibility_record,
        anchor=static_design_path,
        inputs=inputs,
        label="hindcasting eligibility manifest",
    )
    eligibility = load_json(eligibility_path)
    if eligibility.get("status") != "locked_before_new_all_task_re_evaluation":
        raise ValueError("hindcasting eligibility manifest is not frozen")

    source = eligibility.get("source") or {}
    audit_path = _verified_reference(
        {
            "path": source.get("executability_audit_json"),
            "sha256": source.get("executability_audit_json_sha256"),
        },
        anchor=eligibility_path,
        inputs=inputs,
        label="Case Study executability audit",
    )
    matrix_path = _verified_reference(
        {
            "path": source.get("executability_matrix_csv"),
            "sha256": source.get("executability_matrix_csv_sha256"),
        },
        anchor=eligibility_path,
        inputs=inputs,
        label="Case Study executability matrix",
    )
    locked_matrix_path = _verified_reference(
        eligibility.get("locked_matrix") or {},
        anchor=eligibility_path,
        inputs=inputs,
        label="locked hindcasting eligibility matrix",
    )

    audit = load_json(audit_path)
    audit_rows = [dict(row) for row in (audit.get("rows") or [])]
    formal_ids = [str(value) for value in (eligibility.get("formal_case_study_ids") or [])]
    audit_ids = [str(value) for value in (audit.get("case_studies") or [])]
    if formal_count and len(formal_ids) != formal_count:
        raise ValueError(
            "formal Case Study count differs between static design and eligibility lock"
        )
    if len(formal_ids) != len(set(formal_ids)) or set(formal_ids) != set(audit_ids):
        raise ValueError("executability audit does not cover the frozen Case Study set")

    common_windows = eligibility.get("common_temporal_windows") or []
    expected_rows = len(formal_ids) * len(common_windows)
    if len(audit_rows) != expected_rows:
        raise ValueError(
            f"executability audit row count mismatch: expected={expected_rows} "
            f"observed={len(audit_rows)}"
        )

    tier_by_status = {
        "executable": "primary",
        "sparse": "exploratory",
        "non_executable": "excluded",
    }
    observed_counts = {status: 0 for status in tier_by_status}
    window_rows: list[dict[str, Any]] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in audit_rows:
        status = str(raw.get("structural_status") or "")
        if status not in tier_by_status:
            raise ValueError(f"unknown structural executability status: {status}")
        case_study_id = str(raw.get("case_study_id") or "")
        if case_study_id not in formal_ids:
            raise ValueError(f"unregistered Case Study in executability audit: {case_study_id}")
        observed_counts[status] += 1
        row = {
            **raw,
            "analysis_tier": tier_by_status[status],
            "enters_primary_hindcasting": status == "executable",
            "method_performance_consumed_for_selection": False,
        }
        window_rows.append(row)
        rows_by_case[case_study_id].append(row)

    declared_counts = {
        str(key): as_int(value)
        for key, value in (eligibility.get("status_counts") or {}).items()
    }
    audit_counts = {
        str(key): as_int(value)
        for key, value in (audit.get("status_counts") or {}).items()
    }
    if observed_counts != declared_counts or observed_counts != audit_counts:
        raise ValueError("executability status counts differ across frozen artifacts")

    design_kg = str(
        ((design.get("canonical_release") or {}).get("knowledge_graph") or {}).get(
            "sha256"
        )
        or ""
    ).upper()
    eligibility_kg = str(
        (((source.get("canonical_release") or {}).get("files") or {}).get(
            "knowledge_graph", {}
        )).get("sha256")
        or ""
    ).upper()
    audit_kg = str(
        ((((audit.get("canonical_release") or {}).get("files") or {}).get(
            "knowledge_graph", {}
        )).get("sha256"))
        or ""
    ).upper()
    if not design_kg or len({design_kg, eligibility_kg, audit_kg}) != 1:
        raise ValueError("executability coverage is not bound to the formal KG release")

    primary_ids = set(eligibility.get("primary_case_study_ids") or [])
    exploratory_ids = set(eligibility.get("exploratory_case_study_ids") or [])
    summary_rows: list[dict[str, Any]] = []
    for case_study_id in formal_ids:
        rows = rows_by_case[case_study_id]
        primary = sum(row["structural_status"] == "executable" for row in rows)
        exploratory = sum(row["structural_status"] == "sparse" for row in rows)
        excluded = sum(row["structural_status"] == "non_executable" for row in rows)
        if (case_study_id in primary_ids) != (primary > 0):
            raise ValueError(f"primary Case Study declaration mismatch: {case_study_id}")
        if (case_study_id in exploratory_ids) != (exploratory > 0):
            raise ValueError(f"exploratory Case Study declaration mismatch: {case_study_id}")
        summary_rows.append(
            {
                "case_study_id": case_study_id,
                "windows_total": len(rows),
                "primary_windows": primary,
                "exploratory_windows": exploratory,
                "excluded_windows": excluded,
                "future_unique_pairs_total": sum(
                    as_int(row.get("future_unique_pairs")) for row in rows
                ),
                "hindcasting_role": (
                    "primary"
                    if primary
                    else "exploratory_only"
                    if exploratory
                    else "not_executable"
                ),
                "method_performance_consumed_for_selection": False,
            }
        )

    return {
        "status": "complete",
        "ready_for_inference": True,
        "formal_case_studies": len(formal_ids),
        "windows": len(audit_rows),
        "status_counts": observed_counts,
        "kg_sha256": design_kg,
        "eligibility_manifest": str(eligibility_path),
        "executability_audit": str(audit_path),
        "executability_matrix": str(matrix_path),
        "locked_matrix": str(locked_matrix_path),
        "tables": {
            "case_study_executability_windows.csv": window_rows,
            "case_study_executability_summary.csv": summary_rows,
        },
    }


def build_case_study_coverage_rows(
    *,
    executability: Mapping[str, Any],
    closure: Mapping[str, Any],
    dedicated: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join dataset validation and hindcasting coverage without inventing results."""

    summary_rows = (executability.get("tables") or {}).get(
        "case_study_executability_summary.csv", []
    )
    closure_rows = {
        str(row.get("task") or ""): row
        for row in (closure.get("tables") or {}).get("closure_task_status.csv", [])
    }
    dedicated_rows = {
        str(row.get("task") or ""): row
        for row in (dedicated.get("tables") or {}).get(
            "dedicated_case_study_status.csv", []
        )
    }
    output: list[dict[str, Any]] = []
    for row in summary_rows:
        task = str(row["case_study_id"])
        if task in dedicated_rows:
            validation = dedicated_rows[task]
            protocol = "dedicated_case_study"
            internal_status = str(validation.get("internal_status") or "pending")
            external_status = str(validation.get("external_status") or "pending")
        elif task in closure_rows:
            validation = closure_rows[task]
            protocol = "generic_dataset_validation"
            internal_status = str(validation.get("status") or "pending")
            external_applicable = str(
                validation.get("external_applicable") or ""
            ).lower() in {"1", "true", "yes"}
            external_status = (
                "complete" if external_applicable else "not_applicable_by_protocol"
            )
        else:
            protocol = "not_executed_in_current_dataset_benchmark"
            internal_status = "not_executed"
            external_status = "not_executed"

        primary = as_int(row.get("primary_windows"))
        exploratory = as_int(row.get("exploratory_windows"))
        if internal_status == "complete" and primary:
            coverage_state = "dataset_validation_and_primary_hindcasting"
        elif internal_status == "complete":
            coverage_state = "dataset_validation_without_primary_hindcasting"
        elif protocol != "not_executed_in_current_dataset_benchmark" and primary:
            coverage_state = "dataset_validation_pending_and_primary_hindcasting"
        elif protocol != "not_executed_in_current_dataset_benchmark":
            coverage_state = "dataset_validation_pending_without_primary_hindcasting"
        elif primary:
            coverage_state = "primary_hindcasting_only"
        elif exploratory:
            coverage_state = "exploratory_hindcasting_only"
        else:
            coverage_state = "not_executable_under_current_registered_benchmarks"

        output.append(
            {
                "case_study_id": task,
                "dataset_validation_protocol": protocol,
                "internal_validation_status": internal_status,
                "external_validation_status": external_status,
                "hindcasting_primary_windows": primary,
                "hindcasting_exploratory_windows": exploratory,
                "hindcasting_excluded_windows": as_int(row.get("excluded_windows")),
                "hindcasting_role": row.get("hindcasting_role"),
                "unified_coverage_state": coverage_state,
                "protocol_inapplicability_is_not_failure": True,
            }
        )
    return output


def _group(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row.get(field) or "")].append(row)
    return output


def summarize_closure(
    *,
    closure_root: Path,
    closure_audit_path: Path,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    audit = load_json(closure_audit_path)
    _record_input(closure_audit_path, inputs)
    if not audit.get("all_complete"):
        raise ValueError("closure audit is not complete")
    if not audit.get("model_robustness_required"):
        raise ValueError("closure audit did not require model robustness")
    if int(audit.get("external_applicable_tasks", -1)) != 0:
        raise ValueError(
            "generic closure must not require external validation; "
            "only dedicated Case Study 1 has a formal external endpoint"
        )

    budget_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    configuration_rows: list[dict[str, Any]] = []

    for task_audit in audit.get("tasks") or []:
        task = str(task_audit["task"])
        benchmark = closure_root / task / "benchmark"
        manifest_path = benchmark / "run_manifest.json"
        manifest = load_json(manifest_path)
        _record_input(manifest_path, inputs)
        if manifest.get("formal_kg_mutated") is not False:
            raise ValueError(f"closure run does not attest an immutable KG: {task}")
        candidate_count = int(task_audit.get("candidate_count") or manifest.get("candidate_count") or 0)
        policy = manifest.get("neurodiscovery_policy") or {}
        configuration_rows.append(
            {
                "task": task,
                "candidate_count": candidate_count,
                "factor_fields": ";".join(map(str, manifest.get("factor_fields") or ())),
                "methods": ";".join(map(str, manifest.get("methods") or ())),
                "trials": manifest.get("trials", ""),
                "budgets": ";".join(map(str, manifest.get("budgets") or ())),
                "recall_targets": ";".join(
                    map(str, manifest.get("recall_targets") or ())
                ),
                "neurodiscovery_policy_id": policy.get("policy_id", ""),
                "neurodiscovery_score_family": policy.get("score_family", ""),
                "neurodiscovery_score_weights_json": json.dumps(
                    policy.get("score_weights") or {},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "closed_loop_config_json": json.dumps(
                    manifest.get("closed_loop_config") or {},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "uses_experimental_outcomes": policy.get(
                    "uses_experimental_outcomes", ""
                ),
                "formal_kg_mutated": manifest.get("formal_kg_mutated"),
                "run_manifest_sha256": sha256_file(manifest_path),
            }
        )
        scope_status: dict[str, dict[str, Any]] = {}

        for scope in ("internal", "external"):
            applicable = scope == "internal" or bool(task_audit.get("external_applicable"))
            if not applicable:
                scope_status[scope] = {
                    "scope": scope,
                    "applicable": False,
                    "state": "not_applicable_by_protocol",
                }
                continue

            metrics_path = benchmark / f"{scope}_metrics_summary.csv"
            costs_path = benchmark / f"{scope}_recall_cost_summary.csv"
            metrics = read_csv(metrics_path)
            costs = read_csv(costs_path)
            _record_input(metrics_path, inputs)
            _record_input(costs_path, inputs)

            per_budget = _group(metrics, "experiments")
            scope_budget_rows: list[dict[str, Any]] = []
            for experiments_text, rows_at_budget in per_budget.items():
                reference_rows = [
                    row for row in rows_at_budget if row.get("method") == REFERENCE_METHOD
                ]
                if len(reference_rows) != 1:
                    raise ValueError(
                        f"{task}/{scope}/{experiments_text}: expected one NeuroDiscovery row"
                    )
                reference_row = reference_rows[0]
                reference = as_float(reference_row.get("hits_mean"))
                best, methods = _best_rows(
                    rows_at_budget, value_field="hits_mean", lower_is_better=False
                )
                experiments = as_int(experiments_text)
                gt_total = as_float(reference_row.get("gt_total_mean"))
                result = {
                    "task": task,
                    "scope": scope,
                    "experiments": experiments,
                    "candidate_count": candidate_count,
                    "gt_total": gt_total,
                    "neurodiscovery_hits_mean": reference,
                    "neurodiscovery_hits_variance": as_float(
                        reference_row.get("hits_variance")
                    ),
                    "best_comparator_hits_mean": best,
                    "best_comparator_methods": ";".join(methods),
                    "absolute_difference": reference - best,
                    "relative_gain_over_best": (
                        (reference - best) / best if best > 0 else ""
                    ),
                    "state": endpoint_state(
                        reference,
                        best,
                        gt_total=gt_total,
                        full_pool=experiments >= candidate_count,
                    ),
                    "evidence_tier": "registered_curve_exploratory_no_single_primary_budget",
                    "confirmatory_inference": False,
                }
                budget_rows.append(result)
                scope_budget_rows.append(result)

            per_target = _group(costs, "recall_target")
            for target_text, rows_at_target in per_target.items():
                reference_rows = [
                    row for row in rows_at_target if row.get("method") == REFERENCE_METHOD
                ]
                if len(reference_rows) != 1:
                    raise ValueError(
                        f"{task}/{scope}/{target_text}: expected one NeuroDiscovery row"
                    )
                reference_row = reference_rows[0]
                reference = as_float(reference_row.get("experiments_required_mean"))
                best, methods = _best_rows(
                    rows_at_target,
                    value_field="experiments_required_mean",
                    lower_is_better=True,
                )
                gt_total = as_float(reference_row.get("gt_total_mean"))
                recall_rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "recall_target": as_float(target_text),
                        "gt_total": gt_total,
                        "neurodiscovery_experiments_mean": reference,
                        "neurodiscovery_experiments_variance": as_float(
                            reference_row.get("experiments_required_variance")
                        ),
                        "best_comparator_experiments_mean": best,
                        "best_comparator_methods": ";".join(methods),
                        "experiments_saved": best - reference,
                        "fraction_fewer_than_best": (
                            (best - reference) / best if best > 0 else ""
                        ),
                        "state": endpoint_state(
                            reference,
                            best,
                            gt_total=gt_total,
                            lower_is_better=True,
                        ),
                        "evidence_tier": "registered_curve_exploratory_no_single_primary_target",
                        "confirmatory_inference": False,
                    }
                )

            non_full = [
                row
                for row in scope_budget_rows
                if row["state"] != "full_pool_sanity"
            ]
            reference_row = max(non_full, key=lambda row: int(row["experiments"])) if non_full else None
            scope_status[scope] = {
                "scope": scope,
                "applicable": True,
                "state": reference_row["state"] if reference_row else "no_non_full_pool_endpoint",
                "reference_budget": reference_row["experiments"] if reference_row else "",
                "neurodiscovery_hits_mean": (
                    reference_row["neurodiscovery_hits_mean"] if reference_row else ""
                ),
                "best_comparator_hits_mean": (
                    reference_row["best_comparator_hits_mean"] if reference_row else ""
                ),
                "best_comparator_methods": (
                    reference_row["best_comparator_methods"] if reference_row else ""
                ),
            }

        p_path = benchmark / "paired_p_values.csv"
        source_p = read_csv(p_path)
        _record_input(p_path, inputs)
        for row in source_p:
            if row.get("scope") == "external" and not task_audit.get("external_applicable"):
                continue
            paired_rows.append({"task": task, **row})

        task_rows.append(
            {
                "task": task,
                "status": task_audit.get("status"),
                "candidate_count": candidate_count,
                "internal_validated": task_audit.get("internal_validated"),
                "external_applicable": task_audit.get("external_applicable"),
                "external_validated": "",
                "cross_cohort_replicated": "",
                "external_analysis_role": task_audit.get(
                    "external_analysis_role", GENERIC_EXTERNAL_RESULTS_ROLE
                ),
                "model_robustness_models": ";".join(
                    task_audit.get("model_robustness_models") or []
                ),
                "internal_reference_state": scope_status["internal"]["state"],
                "internal_reference_budget": scope_status["internal"].get(
                    "reference_budget", ""
                ),
                "external_reference_state": scope_status["external"]["state"],
                "external_reference_budget": scope_status["external"].get(
                    "reference_budget", ""
                ),
                "warnings": " | ".join(task_audit.get("warnings") or []),
            }
        )

    paired_rows = holm_adjust_rows(
        paired_rows,
        group_fields=("task", "scope", "comparison", "value"),
        p_field="p_value",
    )
    endpoint_evidence_rows = build_endpoint_evidence_rows(
        budget_rows,
        paired_rows,
    )
    task_diagnostic_rows = build_task_diagnostic_rows(endpoint_evidence_rows)
    return {
        "status": "complete",
        "internal_complete": int(audit["internal_complete_tasks"]),
        "internal_expected": int(audit["task_count"]),
        "external_complete": int(audit["external_complete_tasks"]),
        "external_expected": int(audit["external_applicable_tasks"]),
        "formal_external_validation_case_studies": list(
            FORMAL_EXTERNAL_VALIDATION_CASE_STUDIES
        ),
        "generic_external_results_role": GENERIC_EXTERNAL_RESULTS_ROLE,
        "audit_all_complete": bool(audit["all_complete"]),
        "kg_sha256": audit.get("kg_sha256"),
        "tables": {
            "closure_budget_comparisons.csv": budget_rows,
            "closure_recall_cost_comparisons.csv": recall_rows,
            "closure_paired_comparisons_holm.csv": paired_rows,
            "closure_endpoint_evidence.csv": endpoint_evidence_rows,
            "closure_task_diagnostics.csv": task_diagnostic_rows,
            "closure_task_status.csv": task_rows,
            "closure_configuration.csv": configuration_rows,
        },
    }


def _neurodiscovery_rows(path: Path) -> list[dict[str, str]]:
    rows = [row for row in read_csv(path) if row.get("method") == REFERENCE_METHOD]
    if not rows:
        raise ValueError(f"no NeuroDiscovery rows in {path}")
    return rows


def _indexed_rows(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, source: Path
) -> dict[tuple[str, ...], Mapping[str, Any]]:
    indexed: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in fields)
        if key in indexed:
            raise ValueError(f"duplicate ablation endpoint {key} in {source}")
        indexed[key] = row
    return indexed


def summarize_policy_ablation(
    *, suite_root: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compare the frozen static and feedback-enabled policy tracks.

    This is a read-only policy ablation. It verifies the suite seal and every
    result/closure hash before comparing only the NeuroDiscovery rows, so the
    ablation cannot silently mix tasks or partial reruns. Generic Case Study
    external artifacts are intentionally ignored because only dedicated CS1
    has a formal external-validation endpoint.
    """

    suite_path = suite_root / "rerun_suite_manifest.json"
    suite = load_json(suite_path)
    _record_input(suite_path, inputs)
    if suite.get("status") != "complete":
        raise ValueError("policy-ablation suite is not complete")
    required_tracks = {"static_ablation", "closed_loop"}
    if set(map(str, suite.get("tracks") or ())) != required_tracks:
        raise ValueError("policy-ablation suite must contain static and closed-loop tracks")

    policy_descriptor = suite.get("policy_manifest") or {}
    policy_path = Path(str(policy_descriptor.get("path") or ""))
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    if sha256_file(policy_path).lower() != str(
        policy_descriptor.get("sha256") or ""
    ).lower():
        raise ValueError("policy manifest SHA-256 mismatch")
    policy = load_json(policy_path)
    _record_input(policy_path, inputs)
    if policy.get("status") != "frozen":
        raise ValueError("policy manifest is not frozen")

    jobs = list(suite.get("jobs") or ())
    expected_jobs = len(suite.get("tasks") or ()) * len(required_tracks)
    if len(jobs) != expected_jobs:
        raise ValueError(
            f"policy-ablation suite has {len(jobs)} jobs; expected {expected_jobs}"
        )
    for job in jobs:
        if job.get("status") not in {"complete", "reused_complete"}:
            raise ValueError(
                f"incomplete policy-ablation job: {job.get('track')}/{job.get('task')}"
            )
        result_path = Path(str(job.get("output_dir") or "")) / "run_manifest.json"
        closure_path = Path(str(job.get("closure_manifest") or ""))
        for path, expected_hash, label in (
            (result_path, job.get("result_manifest_sha256"), "result"),
            (closure_path, job.get("closure_manifest_sha256"), "closure"),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path).lower() != str(expected_hash or "").lower():
                raise ValueError(
                    f"{label} SHA-256 mismatch for {job.get('track')}/{job.get('task')}"
                )
            _record_input(path, inputs)

    budget_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for task in map(str, suite.get("tasks") or ()):
        manifests: dict[str, dict[str, Any]] = {}
        for track in sorted(required_tracks):
            benchmark = suite_root / track / task / "benchmark"
            manifest_path = benchmark / "run_manifest.json"
            manifest = load_json(manifest_path)
            manifests[track] = manifest
            config = manifest.get("closed_loop_config") or {}
            config_rows.append(
                {
                    "task": task,
                    "track": track,
                    "policy_id": (manifest.get("neurodiscovery_policy") or {}).get(
                        "policy_id", ""
                    ),
                    "candidate_count": manifest.get("candidate_count", 0),
                    "trials": manifest.get("trials", 0),
                    "batch_size": config.get("batch_size", ""),
                    "warmup_batches": config.get("warmup_batches", ""),
                    "feedback_weight": config.get("feedback_weight", ""),
                    "pair_feedback_weight": config.get("pair_feedback_weight", ""),
                    "exploration_weight": config.get("exploration_weight", ""),
                    "diversity_penalty": config.get("diversity_penalty", ""),
                    "preserve_static_until_informative_feedback": config.get(
                        "preserve_static_until_informative_feedback", ""
                    ),
                }
            )

        candidate_count = int(manifests["closed_loop"].get("candidate_count") or 0)
        for scope in GENERIC_FORMAL_VALIDATION_SCOPES:
            metric_paths = {
                track: suite_root
                / track
                / task
                / "benchmark"
                / f"{scope}_metrics_summary.csv"
                for track in required_tracks
            }
            if not all(path.is_file() for path in metric_paths.values()):
                continue
            metric_indexes = {
                track: _indexed_rows(
                    _neurodiscovery_rows(path), ("experiments",), source=path
                )
                for track, path in metric_paths.items()
            }
            if set(metric_indexes["static_ablation"]) != set(
                metric_indexes["closed_loop"]
            ):
                raise ValueError(f"mismatched policy-ablation budgets for {task}/{scope}")
            metric_trial_paths = {
                track: suite_root
                / track
                / task
                / "benchmark"
                / f"{scope}_metrics_by_trial.csv"
                for track in required_tracks
            }
            if not all(path.is_file() for path in metric_trial_paths.values()):
                missing = [
                    str(path)
                    for path in metric_trial_paths.values()
                    if not path.is_file()
                ]
                raise FileNotFoundError(
                    "missing policy-ablation trial metrics: " + "; ".join(missing)
                )
            metric_trial_indexes = {
                track: _indexed_rows(
                    _neurodiscovery_rows(path),
                    ("trial", "experiments"),
                    source=path,
                )
                for track, path in metric_trial_paths.items()
            }
            for path in metric_trial_paths.values():
                _record_input(path, inputs)
            if set(metric_trial_indexes["static_ablation"]) != set(
                metric_trial_indexes["closed_loop"]
            ):
                raise ValueError(
                    f"mismatched policy-ablation metric trials for {task}/{scope}"
                )
            task_metric_rows: list[dict[str, Any]] = []
            for key in sorted(
                metric_indexes["closed_loop"], key=lambda item: as_int(item[0])
            ):
                static = metric_indexes["static_ablation"][key]
                closed = metric_indexes["closed_loop"][key]
                experiments = as_int(key[0])
                static_hits = as_float(static.get("hits_mean"))
                closed_hits = as_float(closed.get("hits_mean"))
                row = {
                    "task": task,
                    "scope": scope,
                    "experiments": experiments,
                    "candidate_count": candidate_count,
                    "gt_total": as_float(closed.get("gt_total_mean")),
                    "static_hits_mean": static_hits,
                    "static_hits_variance": as_float(static.get("hits_variance")),
                    "closed_loop_hits_mean": closed_hits,
                    "closed_loop_hits_variance": as_float(closed.get("hits_variance")),
                    "closed_minus_static_hits": closed_hits - static_hits,
                    "state": (
                        "closed_loop_better"
                        if closed_hits > static_hits
                        else "static_better"
                        if closed_hits < static_hits
                        else "tie"
                    ),
                    "full_pool_sanity": experiments >= candidate_count,
                }
                budget_rows.append(row)
                task_metric_rows.append(row)
                trial_keys = sorted(
                    (
                        trial_key
                        for trial_key in metric_trial_indexes["closed_loop"]
                        if as_int(trial_key[1]) == experiments
                    ),
                    key=lambda item: as_int(item[0]),
                )
                differences = [
                    as_float(
                        metric_trial_indexes["closed_loop"][trial_key].get("hits")
                    )
                    - as_float(
                        metric_trial_indexes["static_ablation"][trial_key].get(
                            "hits"
                        )
                    )
                    for trial_key in trial_keys
                ]
                if not differences:
                    raise ValueError(
                        f"no paired policy-ablation metric trials for "
                        f"{task}/{scope}/{experiments}"
                    )
                paired_rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "comparison": "same_experiments_closed_minus_static_hits",
                        "value": experiments,
                        "n_pairs": len(differences),
                        "mean_difference": sum(differences) / len(differences),
                        "sample_variance_difference": sample_variance(differences),
                        "p_value_exact_one_sided": exact_paired_sign_flip_p(
                            differences
                        ),
                        "alternative": "closed_loop_greater_than_static_ablation",
                        "evidence_tier": "posthoc_multiplicity_adjusted_policy_ablation",
                        "confirmatory_inference": False,
                    }
                )

            non_full = [row for row in task_metric_rows if not row["full_pool_sanity"]]
            reference = max(non_full, key=lambda row: row["experiments"]) if non_full else None
            task_rows.append(
                {
                    "task": task,
                    "scope": scope,
                    "reference_budget": reference["experiments"] if reference else "",
                    "static_hits_mean": reference["static_hits_mean"] if reference else "",
                    "closed_loop_hits_mean": (
                        reference["closed_loop_hits_mean"] if reference else ""
                    ),
                    "closed_minus_static_hits": (
                        reference["closed_minus_static_hits"] if reference else ""
                    ),
                    "state": reference["state"] if reference else "no_non_full_endpoint",
                }
            )

            cost_paths = {
                track: suite_root
                / track
                / task
                / "benchmark"
                / f"{scope}_recall_cost_summary.csv"
                for track in required_tracks
            }
            if not all(path.is_file() for path in cost_paths.values()):
                continue
            cost_indexes = {
                track: _indexed_rows(
                    _neurodiscovery_rows(path), ("recall_target",), source=path
                )
                for track, path in cost_paths.items()
            }
            if set(cost_indexes["static_ablation"]) != set(cost_indexes["closed_loop"]):
                raise ValueError(
                    f"mismatched policy-ablation recall targets for {task}/{scope}"
                )
            cost_trial_paths = {
                track: suite_root
                / track
                / task
                / "benchmark"
                / f"{scope}_recall_cost_by_trial.csv"
                for track in required_tracks
            }
            if not all(path.is_file() for path in cost_trial_paths.values()):
                missing = [
                    str(path)
                    for path in cost_trial_paths.values()
                    if not path.is_file()
                ]
                raise FileNotFoundError(
                    "missing policy-ablation trial recall costs: "
                    + "; ".join(missing)
                )
            cost_trial_indexes = {
                track: _indexed_rows(
                    _neurodiscovery_rows(path),
                    ("trial", "recall_target"),
                    source=path,
                )
                for track, path in cost_trial_paths.items()
            }
            for path in cost_trial_paths.values():
                _record_input(path, inputs)
            if set(cost_trial_indexes["static_ablation"]) != set(
                cost_trial_indexes["closed_loop"]
            ):
                raise ValueError(
                    f"mismatched policy-ablation recall-cost trials for {task}/{scope}"
                )
            for key in sorted(
                cost_indexes["closed_loop"], key=lambda item: as_float(item[0])
            ):
                static = cost_indexes["static_ablation"][key]
                closed = cost_indexes["closed_loop"][key]
                static_cost = as_float(static.get("experiments_required_mean"))
                closed_cost = as_float(closed.get("experiments_required_mean"))
                recall_rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "recall_target": as_float(key[0]),
                        "gt_total": as_float(closed.get("gt_total_mean")),
                        "static_experiments_mean": static_cost,
                        "static_experiments_variance": as_float(
                            static.get("experiments_required_variance")
                        ),
                        "closed_loop_experiments_mean": closed_cost,
                        "closed_loop_experiments_variance": as_float(
                            closed.get("experiments_required_variance")
                        ),
                        "experiments_saved_by_closed_loop": static_cost - closed_cost,
                        "fraction_fewer_than_static": (
                            (static_cost - closed_cost) / static_cost
                            if static_cost > 0
                            else ""
                        ),
                    }
                )

                recall_target = as_float(key[0])
                trial_keys = sorted(
                    (
                        trial_key
                        for trial_key in cost_trial_indexes["closed_loop"]
                        if abs(as_float(trial_key[1]) - recall_target) <= 1e-12
                    ),
                    key=lambda item: as_int(item[0]),
                )
                differences = [
                    as_float(
                        cost_trial_indexes["static_ablation"][trial_key].get(
                            "experiments_required"
                        )
                    )
                    - as_float(
                        cost_trial_indexes["closed_loop"][trial_key].get(
                            "experiments_required"
                        )
                    )
                    for trial_key in trial_keys
                ]
                if not differences:
                    raise ValueError(
                        f"no paired policy-ablation recall-cost trials for "
                        f"{task}/{scope}/{recall_target}"
                    )
                paired_rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "comparison": "same_recall_static_minus_closed_experiments",
                        "value": recall_target,
                        "n_pairs": len(differences),
                        "mean_difference": sum(differences) / len(differences),
                        "sample_variance_difference": sample_variance(differences),
                        "p_value_exact_one_sided": exact_paired_sign_flip_p(
                            differences
                        ),
                        "alternative": "closed_loop_requires_fewer_experiments_than_static_ablation",
                        "evidence_tier": "posthoc_multiplicity_adjusted_policy_ablation",
                        "confirmatory_inference": False,
                    }
                )

    paired_rows = holm_adjust_rows(
        paired_rows,
        group_fields=("task", "scope", "comparison"),
        p_field="p_value_exact_one_sided",
        output_field="p_holm_within_endpoint_family",
    )

    return {
        "status": "complete",
        "ready_for_inference": True,
        "suite_manifest": str(suite_path.resolve()),
        "policy_manifest": str(policy_path.resolve()),
        "policy_manifest_sha256": sha256_file(policy_path),
        "selection_protocol": policy.get("selection_protocol") or {},
        "formal_scopes": list(GENERIC_FORMAL_VALIDATION_SCOPES),
        "generic_external_results_role": GENERIC_EXTERNAL_RESULTS_ROLE,
        "statistical_contract": {
            "unit": "paired_seed",
            "test": "exact_one_sided_paired_sign_flip",
            "alternative": "closed_loop_better_than_static_ablation",
            "multiple_testing": "Holm within task, scope, and endpoint family",
            "evidence_tier": "posthoc_multiplicity_adjusted_policy_ablation",
            "confirmatory_inference": False,
        },
        "tasks": len(suite.get("tasks") or ()),
        "jobs": len(jobs),
        "tables": {
            "policy_ablation_budget_comparisons.csv": budget_rows,
            "policy_ablation_recall_cost_comparisons.csv": recall_rows,
            "policy_ablation_paired_comparisons_holm.csv": paired_rows,
            "policy_ablation_task_status.csv": task_rows,
            "policy_ablation_config.csv": config_rows,
        },
    }


def summarize_dedicated_case_studies(
    *, suite_root: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Collect the dedicated CS1/CS2 protocols after their independent audit."""

    suite_path = suite_root / "dedicated_suite_manifest.json"
    audit_path = suite_root / "audit" / "dedicated_audit.json"
    suite = load_json(suite_path)
    _record_input(suite_path, inputs)
    if suite.get("status") != "complete" or not audit_path.is_file():
        stages = suite.get("stages") or {}

        def case_status(prefix: str) -> str:
            statuses = [
                str(stage.get("status") or "pending")
                for name, stage in stages.items()
                if str(name).startswith(prefix)
            ]
            if any(status == "failed" for status in statuses):
                return "failed"
            if any(status == "running" for status in statuses):
                return "running"
            if statuses and all(status in {"complete", "reused_complete"} for status in statuses):
                return "complete"
            return "pending"

        protocol = suite.get("protocol") or {}
        audit_status = "pending" if not audit_path.is_file() else "not_checked"
        return {
            "status": (
                "awaiting_audit"
                if suite.get("status") == "complete"
                else str(suite.get("status") or "running")
            ),
            "ready_for_inference": False,
            "suite_manifest": str(suite_path.resolve()),
            "audit_manifest": str(audit_path),
            "case_studies": 2,
            "stages_complete": sum(
                str(stage.get("status") or "") in {"complete", "reused_complete"}
                for stage in stages.values()
            ),
            "stages_total": len(stages),
            "tables": {
                "dedicated_case_study_status.csv": [
                    {
                        "task": "case1_transdiagnostic",
                        "internal_status": case_status("case1_"),
                        "external_status": case_status("case1_external"),
                        "external_protocol": protocol.get("case1_external"),
                        "trials": suite.get("trials"),
                        "audit_status": audit_status,
                    },
                    {
                        "task": "case2_pathway_mediation",
                        "internal_status": case_status("case2_"),
                        "external_status": "not_applicable_by_protocol",
                        "external_protocol": protocol.get("case2_external"),
                        "trials": suite.get("trials"),
                        "audit_status": audit_status,
                    },
                ]
            },
        }

    audit = load_json(audit_path)
    _record_input(audit_path, inputs)
    if audit.get("status") != "passed" or int(audit.get("failed", 1)) != 0:
        raise ValueError("dedicated CS1/CS2 audit did not pass")

    case1_internal = suite_root / "case1_transdiagnostic/internal_method_comparison"
    case1_external = suite_root / "case1_transdiagnostic/external_method_comparison"
    case1_final = suite_root / "case1_transdiagnostic/final_summary"
    case2_internal = suite_root / "case2_pathway_mediation/comparison"
    sources = {
        "dedicated_case1_internal_curves.csv": (
            case1_internal / "case1_discovery_curves_by_trial.csv"
        ),
        "dedicated_case1_internal_summary.csv": (
            case1_internal / "case1_method_summary_by_trial.csv"
        ),
        "dedicated_case1_external_metrics.csv": (
            case1_external / "case1_external_metrics_by_trial.csv"
        ),
        "dedicated_case1_external_recall_cost.csv": (
            case1_external / "case1_external_recall_cost_by_trial.csv"
        ),
        "dedicated_case1_generation_quality_summary.csv": (
            case1_final / "generation_quality_primary_summary.csv"
        ),
        "dedicated_case1_internal_primary_summary.csv": (
            case1_final / "internal_primary_budget_summary.csv"
        ),
        "dedicated_case1_internal_primary_recall_cost.csv": (
            case1_final / "internal_primary_recall_cost_summary.csv"
        ),
        "dedicated_case1_internal_primary_p_values.csv": (
            case1_final / "internal_primary_p_values.csv"
        ),
        "dedicated_case1_internal_same_experiments_headline.csv": (
            case1_final / "internal_same_experiments_headline.csv"
        ),
        "dedicated_case1_internal_same_recall_headline.csv": (
            case1_final / "internal_same_recall_headline.csv"
        ),
        "dedicated_case1_external_primary_summary.csv": (
            case1_final / "external_primary_metrics_summary.csv"
        ),
        "dedicated_case1_external_primary_recall_cost.csv": (
            case1_final / "external_primary_recall_cost_summary.csv"
        ),
        "dedicated_case1_external_primary_p_values.csv": (
            case1_final / "external_primary_p_values.csv"
        ),
        "dedicated_case1_external_same_experiments_headline.csv": (
            case1_final / "external_pooled_same_experiments_headline.csv"
        ),
        "dedicated_case1_external_same_recall_headline.csv": (
            case1_final / "external_pooled_same_recall_headline.csv"
        ),
        "dedicated_case2_internal_metrics.csv": (
            case2_internal / "case2_method_metrics_by_trial.csv"
        ),
        "dedicated_case2_internal_aggregate.csv": (
            case2_internal / "case2_method_metrics_aggregate.csv"
        ),
        "dedicated_case2_paired_comparisons.csv": (
            case2_internal / "case2_paired_comparisons.csv"
        ),
    }
    tables: dict[str, list[dict[str, Any]]] = {}
    for output_name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        _record_input(source, inputs)
        rows = read_csv(source)
        task = (
            "case1_transdiagnostic"
            if "case1" in output_name
            else "case2_pathway_mediation"
        )
        scope = "external" if "external" in output_name else "internal"
        tables[output_name] = [
            {"task": task, "scope": scope, **row} for row in rows
        ]

    protocol = suite.get("protocol") or {}
    tables["dedicated_case_study_status.csv"] = [
        {
            "task": "case1_transdiagnostic",
            "internal_status": "complete",
            "external_status": "complete",
            "external_protocol": protocol.get("case1_external"),
            "trials": suite.get("trials"),
            "audit_status": audit.get("status"),
        },
        {
            "task": "case2_pathway_mediation",
            "internal_status": "complete",
            "external_status": "not_applicable_by_protocol",
            "external_protocol": protocol.get("case2_external"),
            "trials": suite.get("trials"),
            "audit_status": audit.get("status"),
        },
    ]
    return {
        "status": "complete",
        "ready_for_inference": True,
        "suite_manifest": str(suite_path.resolve()),
        "audit_manifest": str(audit_path.resolve()),
        "audit_checks": int(audit.get("checks", 0)),
        "case_studies": 2,
        "internal_complete": 2,
        "external_complete": 1,
        "external_not_applicable": 1,
        "canonical_release": suite.get("canonical_release"),
        "tables": tables,
    }


def _completed_run_manifests(root: Path) -> int:
    count = 0
    if not root.is_dir():
        return count
    for path in root.rglob("run_manifest.json"):
        try:
            payload = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("status") in {None, "complete"}:
            count += 1
    return count


def _verification_status(
    root: Path, *, expected_mode: str | None = "formal"
) -> dict[str, Any]:
    path = root / "verification" / "verification_manifest.json"
    if not path.is_file():
        return {"status": "pending", "path": str(path)}
    payload = load_json(path)
    mode_ok = expected_mode is None or payload.get("mode") == expected_mode
    return {
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "path": str(path.resolve()),
        "ready": payload.get("status") == "passed" and mode_ok,
    }


def summarize_static_hindcasting(
    *, root: Path, design_path: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    design = load_json(design_path)
    _record_input(design_path, inputs)
    expected = int(design["primary_matrix"]["paired_runs_per_method"])
    generation_count = len(list((root / "neurodiscovery_generation").rglob("hypotheses_raw.json")))
    baseline_count = len(list((root / "frozen_baseline_generation").rglob("hypotheses_raw.json")))
    execution_path = root / "execution_manifest.json"
    execution = load_json(execution_path) if execution_path.is_file() else {}
    if execution_path.is_file():
        _record_input(execution_path, inputs)
    computation_complete = (
        execution.get("status") == "complete" and execution.get("mode") == "formal"
    )
    verification = _verification_status(root)
    if Path(verification["path"]).is_file():
        _record_input(Path(verification["path"]), inputs)

    tables: dict[str, list[dict[str, Any]]] = {}
    if computation_complete:
        summary_path = root / "summary" / "mean_variance_summary.csv"
        paired_path = root / "summary" / "paired_comparisons.csv"
        manifest_path = root / "summary" / "summary_manifest.json"
        for path in (summary_path, paired_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            _record_input(path, inputs)
        rows = read_csv(summary_path)
        paired = read_csv(paired_path)
        primary = design["endpoints"]["primary"]
        k = int(primary["k"])
        metric = str(primary["metric"])
        tables["static_primary_endpoint.csv"] = [
            row
            for row in rows
            if row.get("scope") == "macro_case_study_equal"
            and row.get("case_study_id") == "ALL"
            and as_int(row.get("k")) == k
            and row.get("metric") == metric
        ]
        tables["static_case_study_primary.csv"] = [
            row
            for row in rows
            if row.get("scope") == "case_study_equal_window"
            and as_int(row.get("k")) == k
            and row.get("metric") == metric
        ]
        tables["static_primary_paired_comparisons.csv"] = [
            row
            for row in paired
            if row.get("scope") == "macro_case_study_equal"
            and row.get("case_study_id") == "ALL"
            and as_int(row.get("k")) == k
            and row.get("metric") == metric
        ]

    return {
        "status": "complete" if computation_complete else "running",
        "ready_for_inference": computation_complete and bool(verification.get("ready")),
        "evidence_tier": design.get("evidence_tier"),
        "configuration_provenance": {
            "design_path": str(design_path.resolve()),
            "design_sha256": sha256_file(design_path),
            "canonical_release": design.get("canonical_release") or {},
            "policy_application": design.get("policy_application") or {},
            "methods": design.get("methods") or {},
            "fixed_budget_contract": design.get("fixed_budget_contract") or {},
            "outcome_isolation": design.get("outcome_isolation") or {},
            "endpoints": design.get("endpoints") or {},
            "reproducibility": design.get("reproducibility") or {},
        },
        "neurodiscovery_runs_complete": generation_count,
        "neurodiscovery_runs_expected": expected,
        "baseline_runs_complete": baseline_count,
        "baseline_runs_expected": expected * 2,
        "verification": verification,
        "tables": tables,
    }


def summarize_dynamic_hindcasting(
    *, root: Path, design_path: Path, inputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    design = load_json(design_path)
    _record_input(design_path, inputs)
    expected = int(design["primary_matrix"]["paired_runs_per_arm"])
    closed_count = _completed_run_manifests(root / "closed")
    open_count = _completed_run_manifests(root / "open")
    execution_path = root / "execution_manifest.json"
    execution = load_json(execution_path) if execution_path.is_file() else {}
    if execution_path.is_file():
        _record_input(execution_path, inputs)
    computation_complete = (
        execution.get("status") == "complete" and execution.get("mode") == "formal"
    )
    # The dynamic verifier has no smoke mode and therefore does not emit a
    # ``mode`` field; a passed manifest is necessarily the formal verifier.
    verification = _verification_status(root, expected_mode=None)
    if Path(verification["path"]).is_file():
        _record_input(Path(verification["path"]), inputs)

    tables: dict[str, list[dict[str, Any]]] = {}
    if computation_complete:
        summary_path = root / "comparison" / "mean_variance_paired_summary.csv"
        manifest_path = root / "comparison" / "comparison_manifest.json"
        for path in (summary_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            _record_input(path, inputs)
        rows = read_csv(summary_path)
        primary = design["endpoints"]["primary"]
        k = int(primary["k"])
        metric = str(primary["metric"])
        tables["dynamic_primary_endpoint.csv"] = [
            row
            for row in rows
            if row.get("scope") == "macro_case_study_equal"
            and row.get("case_study_id") == "ALL"
            and as_int(row.get("requested_k")) == k
            and row.get("metric") == metric
        ]
        dynamic_case_study_primary = [
            row
            for row in rows
            if row.get("scope") == "case_study_equal_window"
            and as_int(row.get("requested_k")) == k
            and row.get("metric") == metric
        ]
        tables["dynamic_case_study_primary.csv"] = holm_adjust_rows(
            dynamic_case_study_primary,
            group_fields=("scope", "requested_k", "metric"),
            p_field="p_closed_greater_exact_sign_flip",
        )

    return {
        "status": "complete" if computation_complete else "running",
        "ready_for_inference": computation_complete and bool(verification.get("ready")),
        "evidence_tier": design.get("evidence_tier"),
        "configuration_provenance": {
            "design_path": str(design_path.resolve()),
            "design_sha256": sha256_file(design_path),
            "canonical_release": design.get("canonical_release") or {},
            "frozen_policy": design.get("frozen_policy") or {},
            "shared_configuration": design.get("shared_configuration") or {},
            "fixed_budget_contract": design.get("fixed_budget_contract") or {},
            "outcome_isolation": design.get("outcome_isolation") or {},
            "endpoints": design.get("endpoints") or {},
            "reproducibility": design.get("reproducibility") or {},
        },
        "closed_runs_complete": closed_count,
        "closed_runs_expected": expected,
        "open_runs_complete": open_count,
        "open_runs_expected": expected,
        "verification": verification,
        "tables": tables,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {}
    closure = summarize_closure(
        closure_root=args.closure_root,
        closure_audit_path=args.closure_audit,
        inputs=inputs,
    )
    static = summarize_static_hindcasting(
        root=args.static_root, design_path=args.static_design, inputs=inputs
    )
    dynamic = summarize_dynamic_hindcasting(
        root=args.dynamic_root, design_path=args.dynamic_design, inputs=inputs
    )
    policy_suite_root = getattr(args, "policy_suite_root", None)
    policy_ablation = (
        summarize_policy_ablation(suite_root=policy_suite_root, inputs=inputs)
        if policy_suite_root is not None
        else {
            "status": "not_requested",
            "ready_for_inference": True,
            "tables": {},
        }
    )
    dedicated_suite_root = getattr(args, "dedicated_suite_root", None)
    dedicated = (
        summarize_dedicated_case_studies(
            suite_root=dedicated_suite_root, inputs=inputs
        )
        if dedicated_suite_root is not None
        else {
            "status": "not_requested",
            "ready_for_inference": True,
            "tables": {},
        }
    )
    executability = summarize_executability_coverage(
        static_design_path=args.static_design,
        inputs=inputs,
    )
    if executability.get("status") == "complete":
        executability["tables"]["case_study_unified_coverage.csv"] = (
            build_case_study_coverage_rows(
                executability=executability,
                closure=closure,
                dedicated=dedicated,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_tables: dict[str, dict[str, Any]] = {}
    for section in (
        closure,
        policy_ablation,
        dedicated,
        executability,
        static,
        dynamic,
    ):
        for filename, rows in section.pop("tables").items():
            output_path = args.output_dir / filename
            write_csv(output_path, rows)
            output_tables[filename] = {
                **file_descriptor(output_path),
                "rows": len(rows),
            }

    ready = (
        closure["audit_all_complete"]
        and policy_ablation["ready_for_inference"]
        and dedicated["ready_for_inference"]
        and executability["ready_for_inference"]
        and static["ready_for_inference"]
        and dynamic["ready_for_inference"]
    )
    status_payload = {
        "schema_version": "unified-case-study-benchmark-status.v1",
        "created_at": utc_now(),
        "ready_for_final_inference": ready,
        "closure": closure,
        "policy_ablation": policy_ablation,
        "dedicated_case_studies": dedicated,
        "case_study_executability": executability,
        "static_hindcasting": static,
        "dynamic_hindcasting": dynamic,
    }
    status_path = args.output_dir / "benchmark_status.json"
    status_path.write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_tables[status_path.name] = file_descriptor(status_path)

    manifest = {
        "schema_version": "unified-case-study-benchmark-report.v1",
        "created_at": utc_now(),
        "status": "complete" if ready else "partial_running",
        "ready_for_final_inference": ready,
        "interpretation_contract": {
            "engineering_completion_is_not_scientific_superiority": True,
            "negative_endpoints_are_retained": True,
            "protocol_inapplicability_is_not_failure": True,
            "full_pool_rows_are_sanity_checks_not_ranking_evidence": True,
            "closure_curves_have_no_single_preregistered_primary_budget": True,
            "best_comparator_selection_is_descriptive_not_confirmatory": True,
            "policy_ablation_paired_tests_are_secondary_not_confirmatory": True,
            "policy_ablation_holm_correction_is_within_registered_endpoint_family": True,
            "formal_hindcasting_requires_independent_verification": True,
            "dedicated_case_studies_require_independent_audit": True,
            "only_case1_requires_external_validation": True,
            "generic_external_results_are_exploratory_only": True,
            "case2_external_not_applicable_is_explicit": True,
            "all_formal_case_studies_have_method_blind_executability_status": True,
            "sparse_or_non_executable_case_studies_are_not_counted_as_failures": True,
        },
        "inputs": sorted(inputs.values(), key=lambda item: item["path"]),
        "outputs": output_tables,
    }
    manifest_path = args.output_dir / "unified_benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-root", required=True, type=Path)
    parser.add_argument("--closure-audit", required=True, type=Path)
    parser.add_argument("--static-root", required=True, type=Path)
    parser.add_argument("--static-design", required=True, type=Path)
    parser.add_argument("--dynamic-root", required=True, type=Path)
    parser.add_argument("--dynamic-design", required=True, type=Path)
    parser.add_argument("--policy-suite-root", type=Path)
    parser.add_argument("--dedicated-suite-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return a non-zero exit code until both formal hindcasting verifiers pass.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_report(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if args.require_complete and not manifest["ready_for_final_inference"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
