"""Wait for formal hindcasting, verify it, and finalize the unified report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REQUIRED_REPORT_OUTPUTS = {
    "benchmark_status.json",
    "closure_configuration.csv",
    "closure_paired_comparisons_holm.csv",
    "policy_ablation_config.csv",
    "policy_ablation_paired_comparisons_holm.csv",
    "dedicated_case_study_status.csv",
    "case_study_unified_coverage.csv",
    "static_primary_endpoint.csv",
    "static_case_study_primary.csv",
    "static_primary_paired_comparisons.csv",
    "dynamic_primary_endpoint.csv",
    "dynamic_case_study_primary.csv",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_descriptor(descriptor: dict[str, Any], *, label: str) -> Path:
    path = Path(str(descriptor.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    if path.stat().st_size != int(descriptor.get("bytes", -1)):
        raise ValueError(f"{label} byte count mismatch: {path}")
    if sha256_file(path) != str(descriptor.get("sha256") or "").upper():
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _require_boolean(value: Any, *, expected: bool, label: str) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            value = normalized == "true"
    if value is not expected:
        raise ValueError(f"{label} must be {expected}")


def _require_release_and_isolation_contract(status: dict[str, Any]) -> str:
    """Verify the no-leakage, cross-task, and canonical-release contract."""

    closure = status["closure"]
    if (
        int(closure.get("internal_complete", -1)) != 8
        or int(closure.get("internal_expected", -1)) != 8
        or int(closure.get("external_complete", -1)) != 0
        or int(closure.get("external_expected", -1)) != 0
        or closure.get("formal_external_validation_case_studies")
        != ["case1_transdiagnostic"]
        or closure.get("generic_external_results_role")
        != "exploratory_only_not_completion_gate"
    ):
        raise ValueError(
            "dataset-validation closure matrix is not 8 generic internal / "
            "0 generic external with dedicated CS1 external validation"
        )
    canonical_hash = str(closure.get("kg_sha256") or "").upper()
    if len(canonical_hash) != 64:
        raise ValueError("closure canonical KG SHA-256 is invalid")

    policy = status["policy_ablation"]
    if (
        policy.get("formal_scopes") != ["internal"]
        or policy.get("generic_external_results_role")
        != "exploratory_only_not_completion_gate"
    ):
        raise ValueError(
            "generic policy ablation must be internal-only; external validation "
            "is formal only for dedicated Case Study 1"
        )
    selection = policy.get("selection_protocol") or {}
    if (
        int(selection.get("source_folds", 0)) < 2
        or int(selection.get("source_trials", 0)) < 2
        or str(selection.get("source_task_weighting") or "")
        != "equal after within-task normalization"
    ):
        raise ValueError("shared policy was not selected by multi-fold, cross-task development")
    for field in (
        "external_outcomes_used_for_selection",
        "target_internal_outcome_tables_opened",
        "target_external_outcome_tables_opened",
    ):
        _require_boolean(selection.get(field), expected=False, label=f"policy {field}")
    _require_boolean(
        selection.get("informative_feedback_guard_selected_on_source_release"),
        expected=True,
        label="source-release feedback guard",
    )
    if int(policy.get("tasks", -1)) != 8 or int(policy.get("jobs", -1)) != 16:
        raise ValueError("shared policy ablation does not cover 8 tasks and 16 jobs")

    dedicated = status["dedicated_case_studies"]
    if (
        int(dedicated.get("audit_checks", -1)) != 44
        or int(dedicated.get("case_studies", -1)) != 2
        or int(dedicated.get("internal_complete", -1)) != 2
        or int(dedicated.get("external_complete", -1)) != 1
        or int(dedicated.get("external_not_applicable", -1)) != 1
    ):
        raise ValueError("dedicated CS1/CS2 audit contract is incomplete")
    dedicated_hash = str(
        (((dedicated.get("canonical_release") or {}).get("files") or {}).get(
            "knowledge_graph"
        ) or {}).get("sha256")
        or ""
    ).upper()
    if dedicated_hash != canonical_hash:
        raise ValueError("dedicated suite used a different canonical KG")

    executability = status["case_study_executability"]
    counts = executability.get("status_counts") or {}
    if (
        int(executability.get("formal_case_studies", -1)) != 17
        or int(executability.get("windows", -1)) != 85
        or int(counts.get("executable", -1)) != 30
        or int(counts.get("sparse", -1)) != 15
        or int(counts.get("non_executable", -1)) != 40
        or str(executability.get("kg_sha256") or "").upper() != canonical_hash
    ):
        raise ValueError("17-Case-Study executability accounting is inconsistent")

    static = status["static_hindcasting"]
    static_config = static.get("configuration_provenance") or {}
    static_release = static_config.get("canonical_release") or {}
    static_policy = static_config.get("policy_application") or {}
    static_outcomes = static_config.get("outcome_isolation") or {}
    static_repro = static_config.get("reproducibility") or {}
    if str(
        ((static_release.get("knowledge_graph") or {}).get("sha256") or "")
    ).upper() != canonical_hash:
        raise ValueError("static hindcasting used a different canonical KG")
    if static_policy.get("mode") != "unchanged_cross_release_application":
        raise ValueError("static hindcasting policy was not an unchanged cross-release application")
    for field in (
        "new_release_future_outcomes_used_for_policy_selection",
        "post_update_retuning_permitted",
    ):
        _require_boolean(static_policy.get(field), expected=False, label=f"static {field}")
    selection_hash = str(
        ((static_policy.get("selection_release") or {}).get("knowledge_graph_sha256") or "")
    ).upper()
    application_hash = str(
        ((static_policy.get("application_release") or {}).get("knowledge_graph_sha256") or "")
    ).upper()
    if not selection_hash or selection_hash == application_hash or application_hash != canonical_hash:
        raise ValueError("static cross-release selection/application hashes are invalid")
    _require_boolean(
        static_outcomes.get("formal_kg_mutation_permitted"),
        expected=False,
        label="static formal KG mutation",
    )
    _require_boolean(
        static_outcomes.get("post_outcome_parameter_tuning_permitted"),
        expected=False,
        label="static post-outcome tuning",
    )
    for field in (
        "bundle_verified_before_and_after_each_stage",
        "formal_runs_rehash_all_referenced_inputs",
        "smoke_outputs_excluded_from_formal_analysis",
    ):
        _require_boolean(static_repro.get(field), expected=True, label=f"static {field}")
    methods = ((static_config.get("methods") or {}).get("primary") or [])
    if {str(row.get("id")) for row in methods} != {
        "neurodiscovery",
        "sciagents",
        "openscholar_rag",
    } or any(row.get("uses_future_outcomes") is not False for row in methods):
        raise ValueError("static method/outcome-blindness contract is invalid")

    dynamic = status["dynamic_hindcasting"]
    dynamic_config = dynamic.get("configuration_provenance") or {}
    dynamic_release = dynamic_config.get("canonical_release") or {}
    dynamic_outcomes = dynamic_config.get("outcome_isolation") or {}
    dynamic_repro = dynamic_config.get("reproducibility") or {}
    if str(
        ((dynamic_release.get("knowledge_graph") or {}).get("sha256") or "")
    ).upper() != canonical_hash:
        raise ValueError("dynamic hindcasting used a different canonical KG")
    for field, expected in (
        ("generation_uses_only_frozen_KG", True),
        ("closed_feedback_uses_only_early_interval", True),
        ("open_feedback_is_withheld", True),
        ("terminal_labels_available_to_generation", False),
        ("formal_KG_mutation_permitted", False),
    ):
        _require_boolean(dynamic_outcomes.get(field), expected=expected, label=f"dynamic {field}")
    for field in (
        "bundle_verified_before_and_after_each_stage",
        "formal_runs_rehash_all_referenced_inputs",
        "smoke_outputs_excluded_from_formal_analysis",
    ):
        _require_boolean(dynamic_repro.get(field), expected=True, label=f"dynamic {field}")
    return canonical_hash


def _require_hindcasting_verifier_contract(
    status: dict[str, Any],
    *,
    verified_report_inputs: set[Path],
) -> dict[str, int]:
    """Verify the substantive static/dynamic audit payloads, not only status flags."""

    static = status["static_hindcasting"]
    static_path = Path(str((static.get("verification") or {}).get("path") or "")).resolve()
    if static_path not in verified_report_inputs:
        raise ValueError("static verifier manifest is not a hashed report input")
    static_audit = load_json(static_path)
    if static_audit.get("status") != "passed" or static_audit.get("mode") != "formal":
        raise ValueError("static hindcasting verifier did not pass in formal mode")
    temporal = static_audit.get("temporal_isolation_audit_by_method") or {}
    expected_methods = {"neurodiscovery", "sciagents", "openscholar_rag"}
    if set(map(str, temporal)) != expected_methods:
        raise ValueError("static verifier lacks method-complete temporal isolation audits")
    static_config = static.get("configuration_provenance") or {}
    target = int(
        (static_config.get("fixed_budget_contract") or {}).get(
            "executed_hypotheses_per_run", -1
        )
    )
    runs_per_method = int(static.get("neurodiscovery_runs_expected", -1))
    method_rows = {
        str(row.get("id")): row
        for row in ((static_config.get("methods") or {}).get("primary") or [])
    }
    if target <= 0 or runs_per_method <= 0:
        raise ValueError("static verifier expected matrix is invalid")
    for method in expected_methods:
        expected_pool = (
            int(method_rows[method].get("generation_pool_size", -1))
            if method == "neurodiscovery"
            else target
        )
        audit = temporal.get(method) or {}
        if (
            expected_pool <= 0
            or int(audit.get("hypotheses", -1)) != runs_per_method * expected_pool
            or int(audit.get("flags", -1)) < 0
            or int(audit.get("evidence_years", -1)) < 0
        ):
            raise ValueError(f"static temporal-isolation audit is incomplete for {method}")

    dynamic = status["dynamic_hindcasting"]
    dynamic_path = Path(
        str((dynamic.get("verification") or {}).get("path") or "")
    ).resolve()
    if dynamic_path not in verified_report_inputs:
        raise ValueError("dynamic verifier manifest is not a hashed report input")
    dynamic_audit = load_json(dynamic_path)
    if dynamic_audit.get("status") != "passed":
        raise ValueError("dynamic hindcasting verifier did not pass")
    closed_expected = int(dynamic.get("closed_runs_expected", -1))
    open_expected = int(dynamic.get("open_runs_expected", -1))
    if (
        int((dynamic_audit.get("closed") or {}).get("runs", -1)) != closed_expected
        or int((dynamic_audit.get("open") or {}).get("runs", -1)) != open_expected
        or int(dynamic_audit.get("paired_runs", -1)) != closed_expected
        or closed_expected != open_expected
    ):
        raise ValueError("dynamic verifier run matrix is incomplete")
    for field, expected in (
        ("formal_kg_mutated", False),
        ("future_outcomes_used_for_task_or_window_selection", False),
        ("claim_as_untouched_confirmatory_permitted", False),
    ):
        _require_boolean(
            dynamic_audit.get(field),
            expected=expected,
            label=f"dynamic verifier {field}",
        )
    return {
        "static_temporal_hypotheses": sum(
            int((temporal.get(method) or {}).get("hypotheses", 0))
            for method in expected_methods
        ),
        "dynamic_closed_runs": closed_expected,
        "dynamic_open_runs": open_expected,
    }


def verify_unified_report(report_root: Path) -> dict[str, Any]:
    """Independently verify the finalized report and its completion contract."""

    manifest_path = report_root / "unified_benchmark_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete" or not manifest.get(
        "ready_for_final_inference"
    ):
        raise ValueError("unified benchmark manifest is not complete")

    contract = manifest.get("interpretation_contract") or {}
    required_contract_flags = (
        "engineering_completion_is_not_scientific_superiority",
        "negative_endpoints_are_retained",
        "protocol_inapplicability_is_not_failure",
        "formal_hindcasting_requires_independent_verification",
        "dedicated_case_studies_require_independent_audit",
        "policy_ablation_paired_tests_are_secondary_not_confirmatory",
        "only_case1_requires_external_validation",
        "generic_external_results_are_exploratory_only",
    )
    if not all(contract.get(field) is True for field in required_contract_flags):
        raise ValueError("unified report interpretation contract is incomplete")

    verified_report_inputs: set[Path] = set()
    for index, descriptor in enumerate(manifest.get("inputs") or ()):
        verified_report_inputs.add(
            _verify_descriptor(dict(descriptor), label=f"report input {index}")
        )

    outputs = manifest.get("outputs") or {}
    missing = sorted(REQUIRED_REPORT_OUTPUTS - set(map(str, outputs)))
    if missing:
        raise ValueError(f"unified report is missing required outputs: {missing}")
    output_paths: dict[str, Path] = {}
    for name, descriptor in outputs.items():
        path = _verify_descriptor(dict(descriptor), label=f"report output {name}")
        output_paths[str(name)] = path
        expected_rows = descriptor.get("rows")
        if expected_rows is not None and len(read_csv(path)) != int(expected_rows):
            raise ValueError(f"report output row count mismatch: {name}")

    status = load_json(output_paths["benchmark_status.json"])
    if not status.get("ready_for_final_inference"):
        raise ValueError("benchmark status did not pass the final inference gate")
    if not (status.get("closure") or {}).get("audit_all_complete"):
        raise ValueError("dataset-validation closure audit is incomplete")
    for section in (
        "policy_ablation",
        "dedicated_case_studies",
        "case_study_executability",
        "static_hindcasting",
        "dynamic_hindcasting",
    ):
        if not (status.get(section) or {}).get("ready_for_inference"):
            raise ValueError(f"unified report section is not inference-ready: {section}")
    canonical_hash = _require_release_and_isolation_contract(status)
    verifier_counts = _require_hindcasting_verifier_contract(
        status,
        verified_report_inputs=verified_report_inputs,
    )

    static = status["static_hindcasting"]
    if int(static.get("neurodiscovery_runs_complete", -1)) != int(
        static.get("neurodiscovery_runs_expected", -2)
    ) or int(static.get("baseline_runs_complete", -1)) != int(
        static.get("baseline_runs_expected", -2)
    ):
        raise ValueError("static hindcasting matrix is incomplete")
    dynamic = status["dynamic_hindcasting"]
    if int(dynamic.get("closed_runs_complete", -1)) != int(
        dynamic.get("closed_runs_expected", -2)
    ) or int(dynamic.get("open_runs_complete", -1)) != int(
        dynamic.get("open_runs_expected", -2)
    ):
        raise ValueError("dynamic hindcasting matrix is incomplete")

    coverage_rows = read_csv(output_paths["case_study_unified_coverage.csv"])
    if len(coverage_rows) != 17 or len(
        {row.get("case_study_id") for row in coverage_rows}
    ) != 17:
        raise ValueError("unified coverage table does not contain 17 Case Studies")

    configuration_rows = read_csv(output_paths["closure_configuration.csv"])
    if len(configuration_rows) != int(status["closure"].get("internal_expected", -1)):
        raise ValueError("closure configuration table does not cover every task")
    if any(
        str(row.get("formal_kg_mutated") or "").lower() != "false"
        or str(row.get("uses_experimental_outcomes") or "").lower() != "false"
        for row in configuration_rows
    ):
        raise ValueError("closure configuration violates outcome/KG isolation")
    expected_methods = {
        "random_walk",
        "ai_scientist_v2",
        "open_coscientist",
        "sciagents",
        "virtual_lab",
        "brainpilot_native",
        "biomni_native",
        "neurodiscovery",
    }
    if any(
        int(row.get("trials") or 0) != 10
        or set(filter(None, str(row.get("methods") or "").split(";"))) != expected_methods
        or row.get("neurodiscovery_score_family") != "relation_aware"
        or not row.get("run_manifest_sha256")
        for row in configuration_rows
    ):
        raise ValueError("closure method/seed/configuration contract is incomplete")

    ablation_config = read_csv(output_paths["policy_ablation_config.csv"])
    ablation_tracks: dict[str, set[str]] = {}
    for row in ablation_config:
        if int(row.get("trials") or 0) != 10 or not row.get("policy_id"):
            raise ValueError("policy ablation configuration lacks 10 paired seeds or policy IDs")
        ablation_tracks.setdefault(str(row.get("task") or ""), set()).add(
            str(row.get("track") or "")
        )
    if len(ablation_tracks) != 8 or any(
        tracks != {"closed_loop", "static_ablation"}
        for tracks in ablation_tracks.values()
    ):
        raise ValueError("policy ablation does not contain both tracks for all 8 tasks")

    ablation_rows = read_csv(
        output_paths["policy_ablation_paired_comparisons_holm.csv"]
    )
    if not ablation_rows:
        raise ValueError("policy ablation has no paired statistical comparisons")
    if any(
        int(row.get("n_pairs") or 0) != 10
        or row.get("scope") != "internal"
        or not row.get("p_value_exact_one_sided")
        or not row.get("p_holm_within_endpoint_family")
        or str(row.get("confirmatory_inference") or "").lower() != "false"
        or row.get("evidence_tier")
        != "posthoc_multiplicity_adjusted_policy_ablation"
        for row in ablation_rows
    ):
        raise ValueError("policy-ablation statistical contract is incomplete")

    for name in (
        "closure_paired_comparisons_holm.csv",
        "static_primary_endpoint.csv",
        "static_case_study_primary.csv",
        "static_primary_paired_comparisons.csv",
        "dynamic_primary_endpoint.csv",
        "dynamic_case_study_primary.csv",
    ):
        if not read_csv(output_paths[name]):
            raise ValueError(f"required inferential output is empty: {name}")

    return {
        "schema_version": "unified-benchmark-completion-audit.v1",
        "status": "passed",
        "verified_at": utc_now(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "inputs_verified": len(manifest.get("inputs") or ()),
        "outputs_verified": len(outputs),
        "required_outputs_verified": len(REQUIRED_REPORT_OUTPUTS),
        "case_studies_verified": len(coverage_rows),
        "policy_ablation_paired_rows_verified": len(ablation_rows),
        "canonical_kg_sha256": canonical_hash,
        "cross_release_and_isolation_contract_verified": True,
        **verifier_counts,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execution_complete(root: Path) -> bool:
    path = root / "execution_manifest.json"
    if not path.is_file():
        return False
    payload = load_json(path)
    return payload.get("status") == "complete" and payload.get("mode") == "formal"


def verification_passed(root: Path, *, static: bool) -> bool:
    path = root / "verification" / "verification_manifest.json"
    if not path.is_file():
        return False
    payload = load_json(path)
    if payload.get("status") != "passed":
        return False
    if static:
        temporal = payload.get("temporal_isolation_audit_by_method") or {}
        return payload.get("mode") == "formal" and set(map(str, temporal)) == {
            "neurodiscovery",
            "sciagents",
            "openscholar_rag",
        }
    return (
        payload.get("formal_kg_mutated") is False
        and payload.get("future_outcomes_used_for_task_or_window_selection") is False
        and payload.get("claim_as_untouched_confirmatory_permitted") is False
        and int((payload.get("closed") or {}).get("runs", 0)) > 0
        and int((payload.get("open") or {}).get("runs", 0)) > 0
    )


def dedicated_suite_passed(root: Path) -> bool:
    suite_path = root / "dedicated_suite_manifest.json"
    audit_path = root / "audit" / "dedicated_audit.json"
    if not suite_path.is_file() or not audit_path.is_file():
        return False
    suite = load_json(suite_path)
    audit = load_json(audit_path)
    return (
        suite.get("status") == "complete"
        and audit.get("status") == "passed"
        and int(audit.get("failed", 1)) == 0
    )


def run_logged(
    command: Sequence[str], *, stdout_path: Path, stderr_path: Path, cwd: Path
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def build_static_verify_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.static_verifier),
        "--output-root",
        str(args.static_root),
        "--design",
        str(args.static_design),
        "--rehash-references",
    ]


def build_dynamic_verify_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(args.dynamic_verifier),
        "--output-root",
        str(args.dynamic_root),
        "--design",
        str(args.dynamic_design),
        "--workspace-root",
        str(args.workspace_root),
        "--rehash-references",
    ]


def build_report_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python),
        str(args.summarizer),
        "--closure-root",
        str(args.closure_root),
        "--closure-audit",
        str(args.closure_audit),
        "--static-root",
        str(args.static_root),
        "--static-design",
        str(args.static_design),
        "--dynamic-root",
        str(args.dynamic_root),
        "--dynamic-design",
        str(args.dynamic_design),
        "--output-dir",
        str(args.report_root),
    ]
    if getattr(args, "policy_suite_root", None) is not None:
        command.extend(["--policy-suite-root", str(args.policy_suite_root)])
    if getattr(args, "dedicated_suite_root", None) is not None:
        command.extend(["--dedicated-suite-root", str(args.dedicated_suite_root)])
    command.append("--require-complete")
    return command


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    status_path = args.report_root / "finalizer_status.json"
    started_at = utc_now()
    while True:
        static_complete = execution_complete(args.static_root)
        dynamic_complete = execution_complete(args.dynamic_root)
        dedicated_complete = (
            dedicated_suite_passed(args.dedicated_suite_root)
            if args.dedicated_suite_root is not None
            else True
        )
        atomic_json(
            status_path,
            {
                "schema_version": "unified-benchmark-finalizer.v1",
                "status": (
                    "waiting"
                    if not (static_complete and dynamic_complete and dedicated_complete)
                    else "verifying"
                ),
                "started_at": started_at,
                "updated_at": utc_now(),
                "static_execution_complete": static_complete,
                "dynamic_execution_complete": dynamic_complete,
                "dedicated_suite_complete": dedicated_complete,
            },
        )
        if static_complete and dynamic_complete and dedicated_complete:
            break
        time.sleep(args.poll_seconds)

    logs = args.report_root / "finalizer_logs"
    try:
        if not verification_passed(args.static_root, static=True):
            run_logged(
                build_static_verify_command(args),
                stdout_path=logs / "static_verify.stdout.log",
                stderr_path=logs / "static_verify.stderr.log",
                cwd=args.workspace_root,
            )
        if not verification_passed(args.dynamic_root, static=False):
            run_logged(
                build_dynamic_verify_command(args),
                stdout_path=logs / "dynamic_verify.stdout.log",
                stderr_path=logs / "dynamic_verify.stderr.log",
                cwd=args.workspace_root,
            )
        run_logged(
            build_report_command(args),
            stdout_path=logs / "unified_report.stdout.log",
            stderr_path=logs / "unified_report.stderr.log",
            cwd=args.workspace_root,
        )
        benchmark_status = load_json(args.report_root / "benchmark_status.json")
        if not benchmark_status.get("ready_for_final_inference"):
            raise ValueError("unified report did not pass its final inference gate")
        completion_audit = verify_unified_report(args.report_root)
        completion_audit_path = args.report_root / "completion_audit.json"
        atomic_json(completion_audit_path, completion_audit)
        result = {
            "schema_version": "unified-benchmark-finalizer.v1",
            "status": "complete",
            "started_at": started_at,
            "finished_at": utc_now(),
            "static_verification_passed": True,
            "dynamic_verification_passed": True,
            "dedicated_suite_passed": True,
            "ready_for_final_inference": True,
            "unified_report_integrity_passed": True,
            "completion_audit": str(
                completion_audit_path.resolve()
            ),
            "completion_audit_sha256": sha256_file(completion_audit_path),
            "benchmark_status": str(
                (args.report_root / "benchmark_status.json").resolve()
            ),
            "unified_manifest": str(
                (args.report_root / "unified_benchmark_manifest.json").resolve()
            ),
        }
    except Exception as exc:
        result = {
            "schema_version": "unified-benchmark-finalizer.v1",
            "status": "failed",
            "started_at": started_at,
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json(status_path, result)
        raise
    atomic_json(status_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--closure-root", required=True, type=Path)
    parser.add_argument("--closure-audit", required=True, type=Path)
    parser.add_argument("--static-root", required=True, type=Path)
    parser.add_argument("--static-design", required=True, type=Path)
    parser.add_argument("--static-verifier", required=True, type=Path)
    parser.add_argument("--dynamic-root", required=True, type=Path)
    parser.add_argument("--dynamic-design", required=True, type=Path)
    parser.add_argument("--dynamic-verifier", required=True, type=Path)
    parser.add_argument("--summarizer", required=True, type=Path)
    parser.add_argument("--policy-suite-root", type=Path)
    parser.add_argument("--dedicated-suite-root", type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    print(json.dumps(finalize(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
