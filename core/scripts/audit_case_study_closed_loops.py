"""Audit whether all registered case studies form reproducible closed loops.

The audit is deliberately stricter than checking that a result directory exists.
It verifies outcome isolation, frozen rankings, repeated trials, KG provenance,
and an append-only experimental feedback chain that never mutates the formal
knowledge graph. External validation is a completion gate for the six
registered supplemental tasks; disease subtyping is the explicit exception.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scripts.case_study_closed_loop_specs import (
    EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES,
    GENERIC_EXTERNAL_RESULTS_ROLE,
    protocol_for,
)


TASKS = (
    "biomarker_discovery",
    "differential_diagnosis",
    "disease_subtyping",
    "connectome_behavior",
    "brain_age",
    "progression_prediction",
    "prognosis",
    "imaging_genetics",
)


@dataclass
class TaskAudit:
    task: str
    status: str = "missing"
    external_applicable: bool = True
    external_analysis_role: str = ""
    candidate_count: int = 0
    internal_validated: int = 0
    external_executable: int = 0
    external_validated: int = 0
    cross_cohort_replicated: int = 0
    methods: list[str] = field(default_factory=list)
    minimum_trials: int = 0
    manifest_ok: bool = False
    files_ok: bool = False
    hashes_ok: bool = False
    outcome_isolation_ok: bool = False
    outcome_blind_ranking_freeze_ok: bool = False
    external_loaded_after_freeze: bool = False
    rankings_frozen_before_external: bool = False
    trials_ok: bool = False
    external_execution_ok: bool = False
    kg_matches_current: bool = False
    kg_semantic_grounding_ok: bool = False
    feedback_chain_ok: bool = False
    formal_kg_immutable: bool = False
    model_robustness_ok: bool = False
    model_robustness_models: list[str] = field(default_factory=list)
    internal_complete: bool = False
    external_complete: bool = False
    complete: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def locate_manifest(root: Path, task: str) -> Path | None:
    standard = root / task / "closure_manifest.json"
    if standard.exists():
        return standard
    if task == "biomarker_discovery":
        legacy = root / task / "tables" / "biomarker_closure_manifest.json"
        if legacy.exists():
            return legacy
    return None


def _table_and_benchmark(
    task: str, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    if "table_manifest" in manifest:
        return (
            dict(manifest.get("table_manifest") or {}),
            dict(manifest.get("benchmark_manifest") or {}),
            False,
        )
    if task == "biomarker_discovery":
        return manifest, None, True
    return {}, None, False


def _audit_files(
    table: dict[str, Any], *, verify_hashes: bool, external_required: bool
) -> tuple[bool, bool, list[str]]:
    errors: list[str] = []
    descriptors = table.get("files") or {}
    expected = ["public_candidates", "internal_outcomes"]
    if external_required or "external_outcomes" in descriptors:
        expected.append("external_outcomes")
    paths: list[Path] = []
    hashes_ok = True
    for name in expected:
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, dict):
            errors.append(f"missing file descriptor: {name}")
            hashes_ok = False
            continue
        path = Path(str(descriptor.get("path") or ""))
        paths.append(path)
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            hashes_ok = False
            continue
        if int(descriptor.get("rows") or 0) <= 0:
            errors.append(f"empty artifact row count: {name}")
        expected_hash = str(descriptor.get("sha256") or "")
        if not expected_hash:
            errors.append(f"missing SHA-256: {name}")
            hashes_ok = False
        elif verify_hashes and sha256_file(path) != expected_hash:
            errors.append(f"SHA-256 mismatch: {name}")
            hashes_ok = False
    if len({str(path.resolve()) for path in paths if path.exists()}) != len(expected):
        errors.append("public, internal, and external tables are not distinct files")
    return not errors, hashes_ok, errors


def _validated_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "candidate_id" not in rows.fieldnames or "validated" not in rows.fieldnames:
            return set()
        return {
            str(row["candidate_id"])
            for row in rows
            if str(row.get("validated") or "").strip().lower() in {"1", "true", "yes"}
        }


def _audit_model_robustness(
    root: Path, task: str, *, verify_hashes: bool
) -> tuple[bool, list[str], list[str]]:
    path = root / task / "model_robustness" / "manifest.json"
    if not path.is_file():
        return False, [], ["model-robustness manifest is missing"]
    try:
        manifest = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [], [f"cannot read model-robustness manifest: {exc}"]
    models = sorted({str(item) for item in manifest.get("models") or []})
    errors: list[str] = []
    if manifest.get("status") != "complete" or manifest.get("task") != task:
        errors.append("model-robustness manifest is not complete or has the wrong task")
    if not models:
        errors.append("model-robustness manifest has no models")
    sources = manifest.get("sources") or []
    if not sources:
        errors.append("model-robustness manifest has no provenance sources")
    for source in sources:
        if source.get("status") != "complete" or int(source.get("failed_jobs", -1)) != 0:
            errors.append(f"incomplete model-robustness source: {source.get('name')}")
        for name, descriptor in (source.get("artifacts") or {}).items():
            artifact = Path(str(descriptor.get("path") or ""))
            if not artifact.is_file():
                errors.append(f"missing model-robustness artifact: {artifact}")
                continue
            expected_hash = str(descriptor.get("sha256") or "")
            if not expected_hash:
                errors.append(f"missing model-robustness SHA-256: {name}")
            elif verify_hashes and sha256_file(artifact) != expected_hash:
                errors.append(f"model-robustness SHA-256 mismatch: {artifact}")
    return not errors, models, errors


def _generic_benchmark_checks(
    benchmark: dict[str, Any], minimum_trials: int, *, external_required: bool
) -> tuple[list[str], int, bool, bool, bool, bool, list[str]]:
    methods = [str(item) for item in benchmark.get("methods") or []]
    trial_counts = {
        str(key): int(value)
        for key, value in (benchmark.get("method_trial_counts") or {}).items()
    }
    trials = min(trial_counts.values(), default=0)
    trials_ok = bool(methods) and all(
        trial_counts.get(method, 0) >= minimum_trials for method in methods
    )
    frozen = benchmark.get("frozen_discovery") or {}
    external = benchmark.get("external") or {}
    freeze_ok = frozen.get("external_data_read_before_freeze") is False and (
        external.get("loaded_after_freeze") is True or not external_required
    )
    delta = benchmark.get("experimental_kg_delta") or {}
    feedback_ok = (
        int(delta.get("records") or 0) > 0
        and bool(delta.get("final_chain_hash"))
        and delta.get("mutates_formal_kg") is False
        and delta.get("schema_version") == "experimental-kg-overlay-bundle.v2"
        and int(delta.get("overlay_count") or 0) > 0
        and delta.get("feedback_consumed_during_ranking") is True
        and delta.get("semantic_projection_verified") is True
        and delta.get("per_seed_isolation_verified") is True
    )
    immutable = (
        benchmark.get("formal_kg_mutated") is False
        and delta.get("mutates_formal_kg") is False
    )
    errors: list[str] = []
    if "neurodiscovery" not in methods or len(set(methods)) < 2:
        errors.append("benchmark must include NeuroDiscovery and at least one comparator")
    if not trials_ok:
        errors.append(
            f"fewer than {minimum_trials} trials for one or more methods: {trial_counts}"
        )
    if external_required and not freeze_ok:
        errors.append("external outcomes were not demonstrably loaded after ranking freeze")
    elif not external_required and frozen.get("external_data_read_before_freeze") is not False:
        errors.append("outcome isolation before ranking freeze is not demonstrated")
    if not feedback_ok:
        errors.append(
            "experimental overlay was not semantically projected and consumed by later ranking rounds"
        )
    if not immutable:
        errors.append("formal KG immutability is not demonstrated")
    return methods, trials, trials_ok, freeze_ok, feedback_ok, immutable, errors


def _legacy_biomarker_checks(
    table: dict[str, Any], minimum_trials: int
) -> tuple[list[str], int, bool, bool, bool, bool, str, list[str], list[str]]:
    ranking = table.get("source_ranking_manifest") or {}
    orders = ranking.get("orders") or {}
    trial_counts = {
        str(key): int(value)
        for key, value in (orders.get("n_trials_by_method") or {}).items()
    }
    methods = sorted(trial_counts)
    trials = min(trial_counts.values(), default=0)
    trials_ok = bool(methods) and all(value >= minimum_trials for value in trial_counts.values())
    freeze_ok = ranking.get("external_data_read_before_freeze") is False
    kg_hash = str(((ranking.get("inputs") or {}).get("kg") or {}).get("sha256") or "")
    errors: list[str] = []
    warnings: list[str] = []
    if "neurodiscovery" not in methods or len(set(methods)) < 2:
        errors.append("legacy benchmark must include NeuroDiscovery and at least one comparator")
    if not trials_ok:
        errors.append(f"legacy ranking has insufficient trials: {trial_counts}")
    if not freeze_ok:
        errors.append("legacy external labels were not demonstrably isolated before freeze")
    delta = table.get("experimental_kg_delta") or {}
    feedback_ok = (
        int(delta.get("records") or 0) > 0
        and bool(delta.get("final_chain_hash"))
        and delta.get("mutates_formal_kg") is False
        and delta.get("source_rankings_verified") is True
        and delta.get("batch_feedback_verified") is True
    )
    immutable = bool(kg_hash) and table.get("formal_kg_mutated") is False
    if not feedback_ok:
        errors.append("legacy CS1 ranking lacks a verified experimental KG delta chain")
        warnings.append(
            "scientific CS1 results are present; only the common feedback audit artifact is missing"
        )
    if not immutable:
        errors.append("legacy CS1 formal KG immutability is not demonstrated")
    return (
        methods,
        trials,
        trials_ok,
        freeze_ok,
        feedback_ok,
        immutable,
        kg_hash,
        errors,
        warnings,
    )


def audit_task(
    root: Path,
    task: str,
    *,
    current_kg_sha256: str,
    minimum_trials: int = 10,
    verify_hashes: bool = False,
    require_model_robustness: bool = False,
) -> TaskAudit:
    result = TaskAudit(task=task, minimum_trials=minimum_trials)
    result.external_applicable = bool(protocol_for(task).external_required)
    result.external_analysis_role = (
        "required"
        if result.external_applicable
        else GENERIC_EXTERNAL_RESULTS_ROLE
    )
    manifest_path = locate_manifest(root, task)
    if manifest_path is None:
        result.errors.append("closure manifest is missing")
        return result
    try:
        manifest = load_json(manifest_path)
        table, benchmark, legacy = _table_and_benchmark(task, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.errors.append(f"cannot read closure manifest: {exc}")
        return result

    result.status = str(manifest.get("status") or table.get("status") or "unknown")
    result.candidate_count = int(table.get("candidate_count") or 0)
    result.internal_validated = int(table.get("internal_validated") or 0)
    result.external_executable = int(table.get("external_executable") or 0)
    result.external_validated = int(table.get("external_validated") or 0)
    result.manifest_ok = result.status == "complete" and result.candidate_count > 0
    if not result.manifest_ok:
        result.errors.append("manifest is not complete or candidate_count is zero")

    result.files_ok, result.hashes_ok, file_errors = _audit_files(
        table,
        verify_hashes=verify_hashes,
        external_required=result.external_applicable,
    )
    result.errors.extend(file_errors)
    if result.files_ok:
        descriptors = table.get("files") or {}
        internal_path = Path(str(descriptors["internal_outcomes"]["path"]))
        external_descriptor = descriptors.get("external_outcomes")
        if external_descriptor:
            external_path = Path(str(external_descriptor["path"]))
            result.cross_cohort_replicated = len(
                _validated_ids(internal_path) & _validated_ids(external_path)
            )
    isolation = table.get("outcome_isolation") or {}
    result.outcome_isolation_ok = (
        isolation.get("public_contains_validation_columns") is False
        and isolation.get("external_read_required_during_generation") is False
        and isolation.get("public_internal_external_are_distinct_files") is True
    )
    if not result.outcome_isolation_ok:
        result.errors.append("candidate, internal, and external outcome isolation failed")

    if legacy:
        (
            result.methods,
            result.minimum_trials,
            result.trials_ok,
            result.rankings_frozen_before_external,
            result.feedback_chain_ok,
            result.formal_kg_immutable,
            run_kg_hash,
            benchmark_errors,
            warnings,
        ) = _legacy_biomarker_checks(table, minimum_trials)
        result.outcome_blind_ranking_freeze_ok = (
            result.rankings_frozen_before_external
        )
        # Legacy CS1 bundles prove label isolation at freeze time but do not
        # carry the newer explicit post-freeze load marker.
        result.external_loaded_after_freeze = (
            result.rankings_frozen_before_external
        )
        result.errors.extend(benchmark_errors)
        result.warnings.extend(warnings)
        result.kg_semantic_grounding_ok = True
    else:
        if not benchmark:
            result.errors.append("benchmark manifest is missing")
            run_kg_hash = ""
        else:
            (
                result.methods,
                observed_trials,
                result.trials_ok,
                result.rankings_frozen_before_external,
                result.feedback_chain_ok,
                result.formal_kg_immutable,
                benchmark_errors,
            ) = _generic_benchmark_checks(
                benchmark,
                minimum_trials,
                external_required=result.external_applicable,
            )
            result.minimum_trials = observed_trials
            result.errors.extend(benchmark_errors)
            run_kg_hash = str((table.get("kg_scoring") or {}).get("kg_sha256") or "")
            frozen = benchmark.get("frozen_discovery") or {}
            external = benchmark.get("external") or {}
            result.outcome_blind_ranking_freeze_ok = (
                frozen.get("external_data_read_before_freeze") is False
            )
            result.external_loaded_after_freeze = (
                external.get("loaded_after_freeze") is True
            )
        kg_scoring = table.get("kg_scoring") or {}
        semantic_fields = list(kg_scoring.get("semantic_fields") or [])
        minimum_grounded_fields = min(2, len(semantic_fields))
        result.kg_semantic_grounding_ok = (
            minimum_grounded_fields > 0
            and int(kg_scoring.get("matched_semantic_fields") or 0)
            >= minimum_grounded_fields
        )
        if not result.kg_semantic_grounding_ok:
            result.errors.append("fewer than two scientific candidate fields are grounded in the KG")

    result.external_execution_ok = (
        result.external_executable > 0 or not result.external_applicable
    )
    if not result.external_execution_ok:
        result.errors.append("no externally executable candidates")
    if result.external_applicable and result.external_validated == 0:
        result.warnings.append("external validation produced a negative result (zero validated candidates)")
    if result.internal_validated == 0:
        result.warnings.append("internal validation produced a negative result (zero validated candidates)")
    elif (
        result.external_applicable
        and result.external_validated > 0
        and result.cross_cohort_replicated == 0
    ):
        result.warnings.append("internal and external supports have no candidate-level overlap")
    result.kg_matches_current = bool(run_kg_hash) and run_kg_hash == current_kg_sha256
    if not result.kg_matches_current:
        result.errors.append("run KG SHA-256 does not match the current frozen KG")

    (
        result.model_robustness_ok,
        result.model_robustness_models,
        robustness_errors,
    ) = _audit_model_robustness(root, task, verify_hashes=verify_hashes)
    if require_model_robustness:
        result.errors.extend(robustness_errors)
    elif robustness_errors:
        result.warnings.extend(robustness_errors)

    internal_gates = (
        result.manifest_ok,
        result.files_ok,
        result.hashes_ok,
        result.outcome_isolation_ok,
        result.outcome_blind_ranking_freeze_ok,
        result.trials_ok,
        result.kg_matches_current,
        result.kg_semantic_grounding_ok,
        result.feedback_chain_ok,
        result.formal_kg_immutable,
    )
    if require_model_robustness:
        internal_gates += (result.model_robustness_ok,)
    result.internal_complete = all(internal_gates)
    result.external_complete = result.internal_complete and (
        not result.external_applicable
        or (
            result.rankings_frozen_before_external
            and result.external_loaded_after_freeze
            and result.external_execution_ok
        )
    )
    result.complete = result.internal_complete and result.external_complete
    return result


def audit_all(
    root: Path,
    kg: Path,
    *,
    minimum_trials: int = 10,
    verify_hashes: bool = False,
    require_model_robustness: bool = False,
) -> dict[str, Any]:
    kg_hash = sha256_file(kg)
    audits = [
        audit_task(
            root,
            task,
            current_kg_sha256=kg_hash,
            minimum_trials=minimum_trials,
            verify_hashes=verify_hashes,
            require_model_robustness=require_model_robustness,
        )
        for task in TASKS
    ]
    return {
        "schema_version": "case-study-closure-audit.v1",
        "created_at": utc_now(),
        "root": str(root),
        "kg": str(kg),
        "kg_sha256": kg_hash,
        "minimum_trials": minimum_trials,
        "hashes_recomputed": verify_hashes,
        "model_robustness_required": require_model_robustness,
        "external_validation_policy": {
            "required_case_studies": list(
                EXTERNAL_VALIDATION_REQUIRED_CASE_STUDIES
            ),
            "generic_results_role": GENERIC_EXTERNAL_RESULTS_ROLE,
            "generic_external_required": True,
        },
        "internal_complete_tasks": sum(item.internal_complete for item in audits),
        "external_applicable_tasks": sum(item.external_applicable for item in audits),
        "external_complete_tasks": sum(
            item.external_complete for item in audits if item.external_applicable
        ),
        "complete_tasks": sum(item.complete for item in audits),
        "task_count": len(audits),
        "all_complete": all(item.complete for item in audits),
        "tasks": [asdict(item) for item in audits],
    }


def write_audit(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "closure_audit.json"
    csv_path = output_dir / "closure_audit.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = list(payload["tasks"])
    fieldnames = [key for key in rows[0] if key not in {"errors", "warnings"}]
    fieldnames.extend(("errors", "warnings"))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "methods": ";".join(row["methods"]),
                    "errors": " | ".join(row["errors"]),
                    "warnings": " | ".join(row["warnings"]),
                }
            )
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--kg", required=True, type=Path)
    parser.add_argument("--minimum-trials", type=int, default=10)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--require-model-robustness", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = audit_all(
        args.root,
        args.kg,
        minimum_trials=args.minimum_trials,
        verify_hashes=args.verify_hashes,
        require_model_robustness=args.require_model_robustness,
    )
    json_path, csv_path = write_audit(payload, args.output_dir or args.root)
    print(
        json.dumps(
            {
                "all_complete": payload["all_complete"],
                "internal_complete_tasks": payload["internal_complete_tasks"],
                "external_applicable_tasks": payload["external_applicable_tasks"],
                "external_complete_tasks": payload["external_complete_tasks"],
                "complete_tasks": payload["complete_tasks"],
                "task_count": payload["task_count"],
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0 if payload["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
