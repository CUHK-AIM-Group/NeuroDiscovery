"""Audit the v3 leakage and the v4 computational-feedback boundary.

This verifier is intentionally pre-execution.  A passing result means that the
new protocol and boundary code reject known leakage routes; it does not promote
old v3 results or claim that a v4 experiment has already run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from neurooracle.src.hindcasting_v4_feedback_boundary import (
    LeakageBoundaryError,
    scan_discovery_source,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
V3_PROTOCOL = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v3_expandable_20260826/protocol"
    / "formal_hindcasting_design_v3.json"
)
V3_RUNNER = (
    ROOT
    / "neurooracle/scripts/run_neurodiscovery_dynamic_closed_loop_hindcasting.py"
)
V4_PROTOCOL = (
    ROOT
    / "neurooracle/data/experiments/hindcasting"
    / "formal_hindcasting_v4_computational_feedback_20260916/protocol"
    / "formal_hindcasting_design_v4.json"
)
V4_BOUNDARY = ROOT / "neurooracle/src/hindcasting_v4_feedback_boundary.py"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_v3() -> dict[str, Any]:
    design = _read_json(V3_PROTOCOL)
    temporal = dict(design.get("temporal_information_flow") or {})
    admission = dict(design.get("task_admission_contract") or {})
    config = dict(design.get("locked_neurodiscovery_config") or {})
    windows = list(design.get("fixed_temporal_windows") or [])
    findings: list[dict[str, Any]] = []

    comparison = str(
        (design.get("fixed_methods") or {}).get("comparison_rule") or ""
    )
    if "early-literature feedback" in comparison:
        findings.append(
            {
                "kind": "future_publication_feedback",
                "evidence": "methods.comparison_rule",
            }
        )
    if int(config.get("feedback_years") or 0) > 0:
        findings.append(
            {
                "kind": "future_publication_feedback",
                "evidence": "locked_neurodiscovery_config.feedback_years",
                "value": int(config["feedback_years"]),
            }
        )
    if "support from the first two future years" in str(
        temporal.get("feedback_reveal_rule") or ""
    ):
        findings.append(
            {
                "kind": "future_publication_feedback",
                "evidence": "temporal_information_flow.feedback_reveal_rule",
            }
        )
    if any(
        "feedback_start_year" in window or "feedback_end_year" in window
        for window in windows
    ):
        findings.append(
            {
                "kind": "future_window_partitioned_for_feedback",
                "evidence": "fixed_temporal_windows",
            }
        )
    if "future years" in str(admission.get("early_feedback_rule") or ""):
        findings.append(
            {
                "kind": "future_informed_task_admission",
                "evidence": "task_admission_contract.early_feedback_rule",
            }
        )
    if "terminal three years" in str(admission.get("terminal_rule") or ""):
        findings.append(
            {
                "kind": "future_informed_task_admission",
                "evidence": "task_admission_contract.terminal_rule",
            }
        )
    try:
        scan_discovery_source(V3_RUNNER)
    except LeakageBoundaryError as exc:
        findings.append(
            {
                "kind": "discovery_imports_publication_evaluator",
                "evidence": str(exc),
            }
        )

    return {
        "status": "failed_as_expected" if findings else "unexpected_pass",
        "eligible_for_computational_feedback_only_claim": False,
        "protocol": str(V3_PROTOCOL),
        "protocol_sha256": sha256_file(V3_PROTOCOL),
        "runner": str(V3_RUNNER),
        "runner_sha256": sha256_file(V3_RUNNER),
        "finding_count": len(findings),
        "findings": findings,
    }


def _audit_v4() -> dict[str, Any]:
    design = _read_json(V4_PROTOCOL)
    checks: dict[str, bool] = {}
    checks["separate_version"] = design.get("schema_version") == (
        "neurodiscovery-formal-hindcasting-design.v4"
    )
    checks["execution_not_claimed"] = str(design.get("status") or "").endswith(
        "execution_not_authorized"
    )
    methods = list((design.get("scope") or {}).get("method_series") or [])
    checks["exact_method_series"] = methods == [
        "neurodiscovery",
        "sciagents",
        "openscholar_rag",
    ]
    checks["llm_disabled"] = (
        (design.get("scope") or {}).get("llm_api_calls_permitted") is False
    )
    checks["v3_not_reused"] = (
        (design.get("supersession") or {}).get("v3_results_eligible_for_v4_claim")
        is False
        and (design.get("supersession") or {}).get("v3_results_may_be_pooled_with_v4")
        is False
    )
    windows = list(design.get("fixed_temporal_windows") or [])
    checks["five_windows"] = [row.get("freeze_year") for row in windows] == [
        2016,
        2017,
        2018,
        2019,
        2020,
    ]
    checks["all_future_years_hidden"] = bool(windows) and all(
        row.get("post_cutoff_publications_available_during_discovery") is False
        and int(row.get("retrospective_evaluation_start_year"))
        == int(row.get("freeze_year")) + 1
        and int(row.get("retrospective_evaluation_end_year"))
        == int(row.get("freeze_year")) + 5
        for row in windows
    )
    phases = dict(design.get("phase_separation") or {})
    evaluation = dict(phases.get("phase_4_retrospective_evaluation") or {})
    checks["evaluation_after_seal"] = (
        evaluation.get("first_permitted_access_to_post_cutoff_publications")
        == "after discovery seal verification"
        and evaluation.get("feedback_to_discovery") is False
        and evaluation.get("reranking_or_regeneration_after_access") is False
    )
    feedback = dict(design.get("computational_feedback_contract") or {})
    checks["computational_feedback_only"] = (
        feedback.get("source_kind") == "computational_experiment"
        and feedback.get("selection_commitment_required_before_outcome_read") is True
    )
    admission = dict(design.get("task_and_cohort_contract") or {})
    forbidden_admission = set(admission.get("admission_must_not_depend_on") or [])
    checks["task_admission_not_future_enriched"] = {
        "future paper count",
        "future exact hits",
        "early-feedback availability",
        "terminal-window support",
        "method performance",
    }.issubset(forbidden_admission)
    checks["metric_not_silently_changed"] = (
        (design.get("evaluation_contract") or {}).get("metric_status")
        == "pending_user_design_decision"
    )
    source_audit = scan_discovery_source(V4_BOUNDARY)
    checks["boundary_has_no_evaluator_import"] = source_audit["status"] == "passed"

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed_pre_execution" if not failed else "failed",
        "execution_completed": False,
        "results_available": False,
        "protocol": str(V4_PROTOCOL),
        "protocol_sha256": sha256_file(V4_PROTOCOL),
        "boundary_source": source_audit,
        "checks": checks,
        "failed_checks": failed,
    }


def audit() -> dict[str, Any]:
    v3 = _audit_v3()
    v4 = _audit_v4()
    passed = (
        v3["status"] == "failed_as_expected"
        and int(v3["finding_count"]) > 0
        and v4["status"] == "passed_pre_execution"
    )
    return {
        "schema_version": "hindcasting-v4-no-leakage-preexecution-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "scope": "protocol_and_guard_pre_execution_only",
        "v3": v3,
        "v4": v4,
        "claim_status": (
            "requires_new_v4_execution_and_post-run audits"
            if passed
            else "not_eligible"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
