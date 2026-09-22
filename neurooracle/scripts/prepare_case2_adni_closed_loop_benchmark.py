"""Freeze and verify the single-seed Case Study 2 closed-loop pilot.

Preparation is result-aware only inside the evaluator bundle.  The public
candidate registry remains outcome blind.  No model endpoint is contacted.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.scripts.prepare_case2_adni_formal_benchmark import (
    PUBLIC_COLUMNS,
    _same_pin as v1_same_pin,
    artifact_pins,
    audit_generator_bundle as audit_v1_generator_bundle,
    benchmark_environment_preflight,
    canonical_sha,
    file_pin,
    hkt_now,
    load_json,
    pin_repository,
    runtime_preflight,
    sha256_file,
    source_code_paths as v1_source_code_paths,
    write_json,
)
from neurooracle.scripts.case2_closed_loop_luna_cli_worker import (
    validate_inner_response_schema,
)
from neurooracle.scripts.case2_closed_loop_luna_thread_gateway import (
    _response_artifacts,
    canonical_json as gateway_canonical_json,
    chat_response,
    render_dispatch,
    request_identity,
    sha256_bytes as gateway_sha256_bytes,
    validate_envelope,
)
EXPECTED_METHODS = (
    "ai_scientist_v2",
    "open_coscientist",
    "sciagents",
    "virtual_lab",
    "brainpilot_native",
    "biomni_native",
    "neurodiscovery",
)
EXPECTED_COORDINATE_VALUES = {
    "exposure": (
        "pathway_prs__curated__ad_risk_gwas__p1em03",
        "pathway_prs__curated__cholinergic__p1em03",
        "pathway_prs__curated__gaba_glutamate__p1em03",
        "pathway_prs__curated__mendelian_ad__p1em03",
        "pathway_prs__curated__microglia_immune__p1em03",
        "pathway_prs__curated__myelin__p5em02",
        "pathway_prs__curated__synaptic__p5em02",
    ),
    "marker": (
        "CENTILOIDS",
        "CTX_ENTORHINAL_SUVR",
        "Entorhinal",
        "FDG_META_ROI_SUVR",
        "Hippocampus",
        "META_TEMPORAL_SUVR",
        "Ventricles",
        "WholeBrain",
    ),
    "outcome": ("ADAS13", "FAQ", "LDELTOTAL"),
}
V1_RUNTIME_SOURCE_REPLACEMENTS = frozenset(
    {"core/scripts/case_study_official_adapter_client.py"}
)
ORACLE_COLUMNS = (
    "candidate_id",
    "complete_case_n",
    "a_path_std",
    "a_path_hc3_p",
    "b_path_std",
    "b_path_hc3_p",
    "indirect_effect_std",
    "indirect_bootstrap_ci_lower",
    "indirect_bootstrap_ci_upper",
    "indirect_bootstrap_p",
    "indirect_bootstrap_q_family_8",
    "indirect_bootstrap_q_global_168",
    "bootstrap_weakest_link_evidence",
    "absolute_indirect_effect",
    "feedback_status",
    "supplemental_family_fdr_hit",
    "global_fdr_hit",
    "nominal_bootstrap_hit",
)
FORBIDDEN_GENERATOR_TOKENS = (
    "a_path_hc3_p",
    "b_path_hc3_p",
    "indirect_bootstrap_p",
    "indirect_bootstrap_q_family_8",
    "indirect_bootstrap_q_global_168",
    "bootstrap_weakest_link_evidence",
    "supplemental_family_fdr_hit",
    "global_fdr_hit",
    "nominal_bootstrap_hit",
    "FORMAL_RESULTS",
    "EXPERIMENT_ORACLE",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema_version") == "neurooracle.case2_closed_loop_benchmark.v3",
        "Wrong closed-loop config schema",
    )
    _require(
        config.get("benchmark_id") == "case2_adni_closed_loop_benchmark_v3_luna_seed1",
        "Wrong closed-loop benchmark ID",
    )
    _require(config.get("case_study_id") == "case2_pathway_mediation", "Wrong case study")
    _require(config["data"]["candidate_count"] == 168, "Candidate count must remain 168")
    _require(config["data"]["source_subject_count"] == 691, "Source cohort must remain 691")
    _require(tuple(config["methods"]["all"]) == EXPECTED_METHODS, "Method suite changed")
    hybrid = config["hybrid_resume"]
    _require(hybrid.get("enabled") is True, "Hybrid continuation must remain enabled")
    _require(hybrid.get("reused_method") == "neurodiscovery", "Wrong reused method")
    _require(hybrid.get("rerun_neurodiscovery") is False, "NeuroDiscovery reuse changed")
    _require(
        tuple(hybrid.get("methods_to_run") or ()) == EXPECTED_METHODS[:-1],
        "Hybrid continuation method set changed",
    )
    design = config["sequential_design"]
    batches = [int(value) for value in design["main_round_batch_sizes"]]
    _require(len(batches) >= 2 and all(value > 0 for value in batches), "Invalid batch schedule")
    _require(sum(batches) == 168, "Main action schedule must sum to 168")
    _require(int(design["primary_action_horizon"]) == 168, "Primary horizon must be 168")
    _require(int(design["trial_count"]) == 1, "This freeze is the single-seed pilot")
    _require(int(design["seed"]) == 247629302, "Pilot seed changed")
    backend = config["model_backend"]
    _require(backend["logical_model"] == "gpt-5.6-luna", "Model changed")
    _require(backend["reasoning_effort"] == "max", "Reasoning effort changed")
    _require(float(backend["requested_temperature"]) == 0.0, "Requested temperature changed")
    _require(
        backend["provider_temperature_control"] == "not_exposed_by_codex_thread",
        "The provider temperature limitation is not disclosed",
    )
    _require(int(backend["requested_max_output_tokens"]) == 8192, "Requested output cap changed")
    if "provider_capture_max_attempts_per_request" in backend:
        _require(
            int(backend["provider_capture_max_attempts_per_request"]) == 8,
            "Provider capture-attempt budget changed",
        )
    _require(
        int(backend["framework_process_retries"]) == 1,
        "Framework must not restart a checkpointed logical request",
    )
    _require(
        backend.get("request_hash_lock") is True and backend.get("response_hash_lock") is True,
        "Thread replay request/response hash locks must remain enabled",
    )
    _require(
        backend.get("execution_channel") == "codex_thread_replay",
        "Case 2 continuation must use the frozen Codex thread replay channel",
    )
    _require(
        backend.get("dispatch_delivery_mode") == "message-specified single read-only dispatch file",
        "Dispatch delivery mode changed",
    )
    _require(
        backend.get("response_capture_mode") == "codex-cli output-last-message direct capture",
        "Raw provider response capture mode changed",
    )
    response_schema_path = REPO_ROOT / "neurooracle/configs/case2_luna_response_envelope.schema.json"
    _require(
        backend.get("response_schema_control")
        == "codex-cli --output-schema required on every formal resume",
        "Provider response-schema control changed",
    )
    _require(
        response_schema_path.is_file()
        and sha256_file(response_schema_path) == backend.get("response_schema_sha256"),
        "Provider response schema changed",
    )
    _require(
        backend.get("dispatch_materialization_mode")
        == "byte-identical SHA256-verified host copy to provider_workspace/CURRENT_DISPATCH.txt, removed after sealing",
        "Provider-workspace dispatch materialization changed",
    )
    _require(backend.get("provider_command_audit_required") is True, "Provider command audit disabled")
    _require(
        "command log audit" in str(backend.get("provider_boundary_enforcement") or ""),
        "Provider boundary enforcement is not disclosed",
    )
    _require(
        isinstance(backend.get("codex_thread_id"), str)
        and len(str(backend["codex_thread_id"])) == 36,
        "Codex provider thread identity is invalid",
    )
    _require(
        backend.get("codex_host_id") == "local",
        "Codex provider host changed",
    )
    _require(
        backend.get("provider_api_calls") is False
        and backend.get("provider_keys_required") is False,
        "Thread replay must not call a provider API or require provider keys",
    )
    _require(
        tuple(backend["automatic_route"]) == ("codex_thread_replay:gpt-5.6-luna:max",),
        "Thread replay route changed",
    )
    _require(backend["fallback_enabled"] is False, "Model fallback enabled")
    _require(backend["manual_response_content_edit_allowed"] is False, "Manual answer edits enabled")
    coordinate_compiler = config.get("native_coordinate_compiler")
    if coordinate_compiler is not None:
        _require(
            isinstance(coordinate_compiler, Mapping),
            "Native coordinate compiler must be one object",
        )
        _require(coordinate_compiler.get("enabled") is True, "Coordinate compiler is disabled")
        _require(
            coordinate_compiler.get("schema_version") == "public_coordinate_aliases_v1",
            "Coordinate compiler schema changed",
        )
        _require(
            tuple(coordinate_compiler.get("applicable_methods") or ())
            == ("open_coscientist",),
            "Coordinate compiler method scope changed",
        )
        _require(
            tuple(coordinate_compiler.get("match_fields") or ())
            == ("exposure", "marker", "outcome"),
            "Coordinate compiler match fields changed",
        )
        _require(
            {
                field: tuple(values)
                for field, values in (
                    coordinate_compiler.get("global_coordinate_values") or {}
                ).items()
            }
            == EXPECTED_COORDINATE_VALUES,
            "Coordinate compiler global value domain changed",
        )
        _require(
            coordinate_compiler.get("registry_alias_fields")
            == {"exposure": ["pathway_name"]},
            "Coordinate compiler registry aliases changed",
        )
        _require(
            isinstance(coordinate_compiler.get("aliases"), Mapping),
            "Coordinate compiler aliases are missing",
        )
        serialized_compiler = json.dumps(
            coordinate_compiler, sort_keys=True, ensure_ascii=False
        ).casefold()
        for token in FORBIDDEN_GENERATOR_TOKENS:
            _require(
                token.casefold() not in serialized_compiler,
                f"Coordinate compiler contains evaluator-only token: {token}",
            )
        _require(
            config.get("design_status", {}).get("coordinate_compiler_outcome_blind")
            is True,
            "Coordinate compiler outcome blindness is not disclosed",
        )
        _require(
            float(backend.get("framework_round_timeout_seconds", 0)) == 43200.0,
            "Framework round timeout must be 43200 seconds for the amended protocol",
        )
        _require(
            float(backend["response_timeout_seconds"])
            < float(backend["framework_round_timeout_seconds"]),
            "Framework round timeout must exceed the per-response timeout",
        )
    migration = config.get("provider_migration")
    if migration is not None:
        _require(isinstance(migration, Mapping), "Provider migration must be one object")
        _require(migration.get("enabled") is True, "Provider migration is not enabled")
        migration_kind = str(
            migration.get("migration_kind") or "provider_thread_change_v1"
        )
        superseded_attempt = config.get("superseded_execution_attempt")
        if superseded_attempt is None and migration_kind == "provider_thread_change_v1":
            _require(
                migration.get("source_freeze_id")
                == config.get("supersedes_preexecution_thread_replay_freeze_id"),
                "Provider-migration source freeze is not the superseded freeze",
            )
        _require(
            migration.get("new_thread_id") == backend["codex_thread_id"],
            "Provider-migration destination thread changed",
        )
        _require(
            tuple(migration.get("reused_completed_methods") or ()) == ("ai_scientist_v2",),
            "Only the completed AI Scientist-v2 trajectory may be migrated",
        )
        _require(
            migration.get("import_exact_sealed_response_cache") is True,
            "Exact sealed-response cache import is required",
        )
        if migration_kind == "provider_thread_change_v1":
            _require(
                migration.get("old_thread_id") != migration.get("new_thread_id"),
                "Provider migration requires distinct old and new threads",
            )
            _require(
                int(migration.get("expected_source_sealed_response_count", -1)) == 117,
                "Expected source sealed-response count changed",
            )
            _require(
                int(migration.get("expected_importable_sealed_request_count", -1)) == 115,
                "Expected importable sealed-response count changed",
            )
            _require(
                tuple(migration.get("excluded_unsealed_request_ids") or ())
                == ("luna_204560043ac330e7c0d1ea8646083571",),
                "Unsealed request exclusion changed",
            )
            _require(
                tuple(migration.get("excluded_downstream_invalid_response_ids") or ())
                == (
                    "luna_a6a26d2211151aec25b10e313ebdc5e9",
                    "luna_f778c89698a70406954ecde0fd5a3f98",
                ),
                "Downstream-invalid response exclusion changed",
            )
        elif migration_kind == "provider_thread_change_v2":
            _require(coordinate_compiler is not None, "Provider migration lost the compiler amendment")
            _require(
                migration.get("source_freeze_id")
                == config.get("supersedes_execution_freeze_id"),
                "Provider-migration source freeze is not the superseded execution",
            )
            _require(
                migration.get("old_thread_id") != migration.get("new_thread_id"),
                "Provider migration requires distinct old and new threads",
            )
            source_thread_ids = tuple(migration.get("source_thread_ids") or ())
            _require(
                migration.get("old_thread_id") in source_thread_ids
                and len(source_thread_ids) >= 1
                and len(set(source_thread_ids)) == len(source_thread_ids),
                "Provider-migration source thread set is invalid",
            )
            _require(
                int(migration.get("expected_source_sealed_response_count", -1))
                == int(migration.get("expected_importable_sealed_request_count", -2))
                > 0,
                "Provider migration must import every sealed, downstream-valid response",
            )
            excluded_unsealed = tuple(
                migration.get("excluded_unsealed_request_ids") or ()
            )
            _require(
                len(excluded_unsealed) == 1
                and excluded_unsealed[0].startswith("luna_"),
                "Provider migration must identify exactly one unsealed request",
            )
            _require(
                not tuple(
                    migration.get("excluded_downstream_invalid_response_ids") or ()
                ),
                "Provider migration cannot exclude downstream-invalid responses",
            )
            _require(
                migration.get("incomplete_method_to_resume") == "open_coscientist",
                "Provider migration must resume Open CoScientist",
            )
        elif migration_kind == "outcome_blind_compiler_amendment_v1":
            _require(coordinate_compiler is not None, "Compiler amendment has no compiler")
            _require(
                migration.get("source_freeze_id")
                == config.get("supersedes_execution_freeze_id"),
                "Compiler-amendment source freeze is not the superseded execution",
            )
            _require(
                migration.get("old_thread_id") == migration.get("new_thread_id"),
                "Compiler amendment must retain the active provider thread",
            )
            source_thread_ids = tuple(migration.get("source_thread_ids") or ())
            _require(
                migration.get("old_thread_id") in source_thread_ids
                and len(set(source_thread_ids)) == len(source_thread_ids),
                "Compiler-amendment source thread set is invalid",
            )
            _require(
                int(migration.get("expected_source_sealed_response_count", -1))
                == int(migration.get("expected_importable_sealed_request_count", -2))
                > 0,
                "Compiler amendment must import every sealed source response",
            )
            _require(
                not tuple(migration.get("excluded_unsealed_request_ids") or ())
                and not tuple(
                    migration.get("excluded_downstream_invalid_response_ids") or ()
                ),
                "Compiler amendment cannot silently exclude source responses",
            )
            _require(
                isinstance(
                    migration.get("compiler_amendment_evidence_relative_paths"), list
                )
                and len(migration["compiler_amendment_evidence_relative_paths"]) >= 7,
                "Compiler-amendment evidence list is incomplete",
            )
        else:
            raise ValueError(f"Unsupported provider migration kind: {migration_kind}")
        _require(
            migration.get("manual_response_content_edit_allowed") is False,
            "Provider migration cannot permit response edits",
        )
    superseded_attempt = config.get("superseded_execution_attempt")
    if superseded_attempt is not None:
        _require(isinstance(superseded_attempt, Mapping), "Superseded attempt must be one object")
        _require(
            superseded_attempt.get("source_freeze_id")
            == config.get("supersedes_preexecution_thread_replay_freeze_id"),
            "Superseded execution-attempt freeze changed",
        )
        _require(
            superseded_attempt.get("provider_thread_id") == backend["codex_thread_id"],
            "Superseded execution-attempt provider changed",
        )
        _require(
            superseded_attempt.get("failed_request_id")
            == "luna_f778c89698a70406954ecde0fd5a3f98",
            "Superseded failed request changed",
        )
        _require(int(superseded_attempt.get("failed_capture_attempt_count", -1)) == 2, "Failed capture count changed")
        _require(superseded_attempt.get("selection_lock_created") is False, "Superseded attempt committed a selection")
        _require(superseded_attempt.get("response_sealed") is False, "Superseded invalid response was sealed")
        _require(
            isinstance(superseded_attempt.get("evidence_relative_paths"), list)
            and len(superseded_attempt["evidence_relative_paths"]) >= 10,
            "Superseded execution evidence list is incomplete",
        )


def provider_attestation_preflight(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    attestation = load_json(path)
    backend = config["model_backend"]
    _require(
        attestation.get("schema_version") == backend["provider_attestation_schema"],
        "Provider attestation schema changed",
    )
    _require(attestation.get("thread_id") == backend["codex_thread_id"], "Attested thread changed")
    _require(attestation.get("host_id") == backend["codex_host_id"], "Attested host changed")
    _require(attestation.get("model") == backend["logical_model"], "Attested model changed")
    _require(
        attestation.get("reasoning_effort") == backend["reasoning_effort"],
        "Attested reasoning effort changed",
    )
    boundaries = attestation.get("provider_boundaries") or {}
    for key in (
        "repository_access",
        "network_access",
        "hidden_evaluator_access",
        "repository_edits",
        "cross_request_experimental_context_allowed",
        "other_file_access",
    ):
        _require(boundaries.get(key) is False, f"Provider boundary is not closed: {key}")
    _require(
        boundaries.get("file_access")
        == "only the single message-specified hash-locked dispatch file for the current request_id",
        "Dispatch-only file access boundary changed",
    )
    _require(boundaries.get("dispatch_file_read_only") is True, "Dispatch access is not read-only")
    _require(
        boundaries.get("dispatch_request_id_and_sha256_must_match") is True,
        "Dispatch identity verification is not attested",
    )
    _require(
        boundaries.get("dispatch_file_sha256_must_match") is True,
        "Dispatch byte-hash verification is not attested",
    )
    _require(
        boundaries.get("raw_answers_used_without_manual_content_edit") is True,
        "Raw provider answer preservation is not attested",
    )
    _require(
        attestation.get("response_capture_mode") == backend["response_capture_mode"],
        "Provider response-capture mode changed",
    )
    _require(
        "CURRENT_DISPATCH.txt" in str(attestation.get("dispatch_materialization_mode") or "")
        and "SHA256" in str(attestation.get("dispatch_materialization_mode") or ""),
        "Provider dispatch materialization is not attested",
    )
    _require(attestation.get("provider_command_audit_required") is True, "Command audit not attested")
    _require(
        attestation.get("technical_filesystem_capability_is_broader_than_authorized_boundary")
        is True,
        "Provider technical capability disclosure is missing",
    )
    capture_status: dict[str, Any] = {}
    for name in ("ready_capture", "dispatch_boundary_capture"):
        capture = attestation.get(name)
        _require(isinstance(capture, Mapping), f"Provider capture is missing: {name}")
        capture_path = Path(str(capture["path"]))
        _require(capture_path.is_file(), f"Provider capture file is missing: {name}")
        _require(capture_path.stat().st_size == int(capture["bytes"]), f"Provider capture size changed: {name}")
        _require(sha256_file(capture_path) == capture["sha256"], f"Provider capture hash changed: {name}")
        capture_status[name] = {
            "path": str(capture_path),
            "bytes": int(capture["bytes"]),
            "sha256": capture["sha256"],
        }
    visibility = attestation.get("visibility_test")
    _require(isinstance(visibility, Mapping), "Provider visibility test is missing")
    visibility_input = Path(str(visibility["input_path"]))
    visibility_response = Path(str(visibility["response_path"]))
    _require(visibility_input.is_file() and visibility_response.is_file(), "Visibility evidence missing")
    _require(visibility_input.stat().st_size == int(visibility["input_bytes"]), "Visibility input size changed")
    _require(visibility_response.stat().st_size == int(visibility["response_bytes"]), "Visibility response size changed")
    _require(sha256_file(visibility_input) == visibility["input_sha256"], "Visibility input hash changed")
    _require(sha256_file(visibility_response) == visibility["response_sha256"], "Visibility response hash changed")
    _require(
        visibility_response.read_text(encoding="utf-8-sig").strip()
        == visibility["input_sha256"]
        == visibility["reported_input_sha256"],
        "Provider visibility hash self-test failed",
    )
    schema_control = attestation.get("response_schema_control")
    _require(isinstance(schema_control, Mapping), "Response-schema attestation is missing")
    schema_path = Path(str(schema_control["schema_path"]))
    schema_test_path = Path(str(schema_control["format_test_response_path"]))
    _require(schema_path.is_file() and schema_test_path.is_file(), "Response-schema evidence missing")
    _require(sha256_file(schema_path) == schema_control["schema_sha256"], "Response schema hash changed")
    _require(
        schema_control["schema_sha256"] == backend["response_schema_sha256"],
        "Attested response schema differs from protocol",
    )
    _require(
        schema_test_path.stat().st_size == int(schema_control["format_test_response_bytes"]),
        "Schema format-test size changed",
    )
    _require(
        sha256_file(schema_test_path) == schema_control["format_test_response_sha256"],
        "Schema format-test hash changed",
    )
    _require(
        json.loads(schema_test_path.read_text(encoding="utf-8-sig"))
        == schema_control["expected_test_envelope"],
        "Schema format-test envelope changed",
    )
    return {
        "status": "passed",
        "attestation_path": str(path),
        "attestation_sha256": sha256_file(path),
        "thread_id": attestation["thread_id"],
        "host_id": attestation["host_id"],
        "model": attestation["model"],
        "reasoning_effort": attestation["reasoning_effort"],
        "ready_message_id": attestation["ready_message_id"],
        "dispatch_boundary_message_id": attestation["dispatch_boundary_message_id"],
        "response_capture_mode": attestation["response_capture_mode"],
        "capture_evidence": capture_status,
        "visibility_test_status": visibility["status"],
        "response_schema_status": schema_control["status"],
        "technical_boundary_enforcement": attestation["technical_boundary_enforcement"],
        "provider_api_calls_made": False,
        "provider_keys_loaded": False,
    }


def verify_v1_integrity_for_v2(
    source_root: Path, *, fast_large_files: bool = True,
) -> dict[str, Any]:
    """Verify the immutable v1 freeze without reinterpreting the runtime key file."""

    ready = load_json(source_root / "READY_TO_RUN.json")
    lock_path = source_root / "BENCHMARK_FREEZE.lock.json"
    lock = load_json(lock_path)
    _require(ready.get("status") == "ready_for_generator_execution", "v1 benchmark is not ready")
    _require(lock.get("status") == "locked_ready_for_generator_execution", "v1 lock is not ready")
    _require(sha256_file(lock_path) == ready["benchmark_lock_sha256"], "v1 lock changed")
    _require(lock["freeze_id"] == ready["freeze_id"], "v1 freeze IDs differ")
    material = lock["lock_material"]
    for relative, pin in material["artifact_pins"].items():
        path = source_root / Path(relative)
        _require(path.is_file(), f"v1 artifact missing: {relative}")
        _require(path.stat().st_size == int(pin["bytes"]), f"v1 artifact size changed: {relative}")
        _require(sha256_file(path) == pin["sha256"], f"v1 artifact changed: {relative}")
    for name, pin in material["external_input_pins"].items():
        _require(
            v1_same_pin(Path(pin["path"]), pin, fast_large_files=fast_large_files),
            f"v1 external input changed: {name}",
        )
    source_transition_records: list[dict[str, Any]] = []
    for relative, pin in material["source_code_pins"].items():
        source_path = Path(pin["path"])
        current_sha = sha256_file(source_path) if source_path.is_file() else None
        drifted = (
            current_sha != pin["sha256"]
            or not source_path.is_file()
            or source_path.stat().st_size != int(pin["bytes"])
        )
        source_transition_records.append(
            {
                "relative_path": relative,
                "v1_locked_sha256": pin["sha256"],
                "current_workspace_sha256": current_sha,
                "workspace_drifted_from_v1": drifted,
                "v2_execution_policy": (
                    "copy_and_pin_in_generator_bundle"
                    if relative in V1_RUNTIME_SOURCE_REPLACEMENTS
                    else "pin_current_workspace_source_in_v2"
                ),
            }
        )
    source_config = load_json(source_root / "PROTOCOL.json")
    leakage = audit_v1_generator_bundle(source_root / "generator_inputs", source_config)
    _require(leakage["status"] == "passed", "v1 generator leakage audit failed")
    return {
        "benchmark_id": ready["benchmark_id"],
        "freeze_id": ready["freeze_id"],
        "status": "verified_v1_integrity_for_v2",
        "artifact_count": len(material["artifact_pins"]),
        "external_input_count": len(material["external_input_pins"]),
        "source_code_pin_count": len(material["source_code_pins"]),
        "v1_to_v2_source_transition": source_transition_records,
        "v1_frozen_source_files_modified": False,
        "runtime_key_shape_reinterpreted": False,
    }


def verify_v1_repository_execution_inputs(
    source_root: Path, *, baseline_root_override: Path | None = None
) -> dict[str, Any]:
    """Verify an exact v1 baseline snapshot while ignoring generated caches.

    A later formal freeze may use an isolated Git worktree snapshot instead of
    the user's mutable baseline checkout.  In that case the repository path is
    intentionally different, while HEAD, tracked diff bytes, dirty source-file
    pins, and the exclusion rule must remain exactly equal to the v1 lock.
    """

    source_config = load_json(source_root / "PROTOCOL.json")
    runtime_lock = load_json(source_root / "METHOD_RUNTIME_LOCK.json")
    locked_baseline_root = Path(source_config["paths"]["baseline_repository_root"])
    baseline_root = (
        baseline_root_override.resolve()
        if baseline_root_override is not None
        else locked_baseline_root
    )
    records: list[dict[str, Any]] = []
    execution_keys = (
        "head",
        "tracked_diff_sha256",
        "tracked_diff_bytes",
        "dirty_or_untracked_source_files",
        "exclusion_rule",
    )
    for method, repository_name in source_config["methods"]["repository_map"].items():
        locked = runtime_lock["repository_pins"][method]
        current = pin_repository(baseline_root / repository_name)
        if baseline_root_override is None:
            _require(
                current["repository"] == locked["repository"],
                f"v1 baseline repository path changed: {method}",
            )
        else:
            _require(
                Path(current["repository"]).name == Path(locked["repository"]).name,
                f"v1 baseline snapshot repository name changed: {method}",
            )
        for key in execution_keys:
            _require(current[key] == locked[key], f"v1 baseline execution input changed: {method}:{key}")
        records.append(
            {
                "method": method,
                "locked_repository": locked["repository"],
                "verified_repository": current["repository"],
                "isolated_snapshot": baseline_root_override is not None,
                "head": current["head"],
                "execution_input_file_count": len(current["dirty_or_untracked_source_files"]),
                "locked_excluded_generated_path_count": int(
                    locked["excluded_untracked_or_generated_path_count"]
                ),
                "current_excluded_generated_path_count": int(
                    current["excluded_untracked_or_generated_path_count"]
                ),
                "execution_inputs_verified": True,
            }
        )
    return {
        "schema_version": "neurooracle.case2_v1_repository_execution_input_audit.v1",
        "status": "verified",
        "locked_baseline_root": str(locked_baseline_root),
        "verified_baseline_root": str(baseline_root),
        "isolated_snapshot": baseline_root_override is not None,
        "excluded_generated_cache_drift_allowed": True,
        "records": records,
    }


def build_oracle(formal_results: Path, public_registry: pd.DataFrame) -> pd.DataFrame:
    formal = pd.read_csv(formal_results, dtype={"candidate_id": str})
    required = {
        "candidate_id",
        "complete_case_n",
        "a_path_std",
        "a_path_hc3_p",
        "b_path_std",
        "b_path_hc3_p",
        "indirect_effect_std",
        "indirect_bootstrap_ci_lower",
        "indirect_bootstrap_ci_upper",
        "indirect_bootstrap_p",
        "indirect_bootstrap_q_family_8",
        "indirect_bootstrap_q_global_168",
        "analysis_status",
    }
    _require(required <= set(formal.columns), "Formal result schema changed")
    _require(len(formal) == 168 and formal["candidate_id"].is_unique, "Formal results changed")
    _require(formal["analysis_status"].astype(str).eq("estimated").all(), "Non-estimated result")
    public_ids = public_registry["candidate_id"].astype(str).tolist()
    _require(set(formal["candidate_id"].astype(str)) == set(public_ids), "Candidate universe changed")
    out = formal.set_index("candidate_id").loc[public_ids].reset_index()
    numeric = [column for column in required if column not in {"candidate_id", "analysis_status"}]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="raise")
        _require(out[column].map(math.isfinite).all(), f"Non-finite {column}")
    weakest = out[["a_path_hc3_p", "b_path_hc3_p", "indirect_bootstrap_p"]].max(axis=1)
    out["bootstrap_weakest_link_evidence"] = (-weakest.clip(lower=1e-300).map(math.log10)).clip(upper=50.0)
    out["absolute_indirect_effect"] = out["indirect_effect_std"].abs()
    complete_nominal = (
        out["a_path_hc3_p"].lt(0.05)
        & out["b_path_hc3_p"].lt(0.05)
        & out["indirect_bootstrap_p"].lt(0.05)
    )
    out["feedback_status"] = complete_nominal.map({True: "supported", False: "inconclusive"})
    out["supplemental_family_fdr_hit"] = out["indirect_bootstrap_q_family_8"].lt(0.05)
    out["global_fdr_hit"] = out["indirect_bootstrap_q_global_168"].lt(0.05)
    out["nominal_bootstrap_hit"] = out["indirect_bootstrap_p"].lt(0.05)
    oracle = out.loc[:, ORACLE_COLUMNS].copy()
    _require(int(oracle["feedback_status"].eq("supported").sum()) == 13, "Feedback status count changed")
    _require(int(oracle["nominal_bootstrap_hit"].sum()) == 15, "Nominal hit count changed")
    _require(int(oracle["supplemental_family_fdr_hit"].sum()) == 6, "Family hit count changed")
    _require(int(oracle["global_fdr_hit"].sum()) == 0, "Global endpoint changed")
    return oracle


def audit_generator_bundle(root: Path) -> dict[str, Any]:
    findings: list[str] = []
    scanned = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        scanned += 1
        if path.suffix.casefold() not in {".json", ".csv", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for token in FORBIDDEN_GENERATOR_TOKENS:
            if token.casefold() in text.casefold():
                findings.append(f"{path.relative_to(root).as_posix()}:{token}")
    return {
        "schema_version": "neurooracle.case2_closed_loop_generator_leakage_audit.v3",
        "status": "passed" if not findings else "failed",
        "files_scanned": scanned,
        "findings": findings,
    }


def _resolve_local_module(module: str) -> Path | None:
    if not module.startswith(("core.", "neurooracle.")):
        return None
    base = REPO_ROOT.joinpath(*module.split("."))
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    return next((path for path in candidates if path.is_file()), None)


def _local_import_closure(initial: set[Path]) -> set[Path]:
    """Pin every local Python module imported by a frozen runtime source."""

    closure = set(initial)
    queue = [path for path in initial if path.suffix.casefold() == ".py"]
    parsed: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in parsed:
            continue
        parsed.add(path)
        relative = path.relative_to(REPO_ROOT)
        module_parts = list(relative.with_suffix("").parts)
        package_parts = module_parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        candidate_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidate_modules.update(alias.name for alias in node.names)
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                keep = len(package_parts) - node.level + 1
                prefix = package_parts[: max(0, keep)]
                base_parts = [*prefix, *(node.module or "").split(".")]
                base_module = ".".join(part for part in base_parts if part)
            else:
                base_module = node.module or ""
            if base_module:
                candidate_modules.add(base_module)
                candidate_modules.update(
                    f"{base_module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        for module in candidate_modules:
            dependency = _resolve_local_module(module)
            if dependency is not None and dependency not in closure:
                closure.add(dependency)
                queue.append(dependency)
    return closure


def source_code_paths(config_path: Path) -> list[Path]:
    v1_config = REPO_ROOT / "neurooracle/configs/case2_adni_formal_benchmark_v1.json"
    paths = {
        path
        for path in v1_source_code_paths(v1_config)
        if path.relative_to(REPO_ROOT).as_posix() not in V1_RUNTIME_SOURCE_REPLACEMENTS
    }
    relative = (
        "neurooracle/scripts/prepare_case2_adni_closed_loop_benchmark.py",
        "neurooracle/scripts/run_case2_adni_closed_loop_benchmark.py",
        "neurooracle/scripts/evaluate_case2_adni_closed_loop_benchmark.py",
        "neurooracle/scripts/case2_closed_loop_luna_thread_gateway.py",
        "neurooracle/scripts/case2_closed_loop_luna_cli_worker.py",
        "neurooracle/configs/case2_luna_response_envelope.schema.json",
        "neurooracle/scripts/start_case2_closed_loop_luna_thread_gateway.ps1",
        "neurooracle/scripts/start_case2_closed_loop_luna_cli_worker.ps1",
        "neurooracle/scripts/start_case2_formal_brainpilot_service.ps1",
        "neurooracle/scripts/materialize_case2_closed_loop_subgraph.py",
        "neurooracle/tests/test_case2_adni_closed_loop_benchmark.py",
        "core/scripts/case_study_closed_loop.py",
        "core/scripts/case_study_closed_loop_engine.py",
        "core/scripts/case_study_feedback_adapters.py",
        "neurooracle/data/case_study_reaudit/full_graph_v3/RUBRIC.md",
        "neurooracle/src/CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md",
    )
    paths.update(REPO_ROOT / item for item in relative)
    paths.add(config_path.resolve())
    paths = _local_import_closure(paths)
    missing = sorted(str(path) for path in paths if not path.is_file())
    _require(not missing, f"Pinned source files missing: {missing}")
    return sorted(paths)


def import_neurodiscovery_hybrid_resume(
    staging: Path,
    *,
    source_root: Path,
    public_path: Path,
    oracle: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and import the API-independent NeuroDiscovery trajectory."""

    hybrid = config["hybrid_resume"]
    expected_freeze = str(hybrid["reused_source_freeze_id"])
    source_ready_path = source_root / "READY_TO_RUN.json"
    source_lock_path = source_root / "BENCHMARK_FREEZE.lock.json"
    source_ready = load_json(source_ready_path)
    source_lock = load_json(source_lock_path)
    _require(source_ready["freeze_id"] == expected_freeze, "Reused ready freeze changed")
    _require(source_lock["freeze_id"] == expected_freeze, "Reused freeze lock changed")
    _require(
        source_ready["benchmark_lock_sha256"] == sha256_file(source_lock_path),
        "Reused freeze lock hash changed",
    )

    source_audit_path = source_root / "runtime/FROZEN_WORKTREE_AUDIT.json"
    source_audit = load_json(source_audit_path)
    _require(source_audit.get("freeze_id") == expected_freeze, "Reused worktree audit changed")
    for key in (
        "source_pins_verified",
        "local_import_closure_pinned",
        "static_runtime_assets_pinned",
        "runner_import_smoke_passed",
        "hypothesis_cli_import_smoke_passed",
        "primary_workspace_isolated",
    ):
        _require(source_audit.get(key) is True, f"Reused worktree audit failed: {key}")
    source_worktree = Path(source_audit["runtime_worktree"])
    for relative, pin in source_lock["lock_material"]["source_code_pins"].items():
        path = source_worktree / relative
        _require(path.is_file(), f"Reused frozen source is missing: {relative}")
        _require(path.stat().st_size == int(pin["bytes"]), f"Reused source size changed: {relative}")
        _require(sha256_file(path) == pin["sha256"], f"Reused source hash changed: {relative}")
    for name, pin in source_lock["lock_material"]["external_input_pins"].items():
        _require(_same_pin(Path(pin["path"]), pin), f"Reused external input changed: {name}")

    source_protocol = load_json(source_root / "PROTOCOL.json")
    _require(source_protocol["data"] == config["data"], "Reused data design changed")
    _require(
        source_protocol["sequential_design"]["seed"] == config["sequential_design"]["seed"],
        "Reused seed changed",
    )
    _require(
        source_protocol["sequential_design"]["main_round_batch_sizes"]
        == config["sequential_design"]["main_round_batch_sizes"],
        "Reused batch schedule changed",
    )
    _require(
        source_protocol["methods"]["neurodiscovery"] == config["methods"]["neurodiscovery"],
        "Reused NeuroDiscovery policy changed",
    )
    for path_key in ("formal_results", "pathway_catalog", "kg_current_state"):
        _require(
            source_protocol["paths"][path_key] == config["paths"][path_key],
            f"Reused NeuroDiscovery input path changed: {path_key}",
        )
    source_public = source_root / "generator_inputs/PUBLIC_CANDIDATE_REGISTRY.csv"
    _require(sha256_file(source_public) == sha256_file(public_path), "Reused public registry changed")

    source_run = source_root / "runs/neurodiscovery/trial_00"
    source_trajectory_dir = source_root / "trajectories/neurodiscovery/trial_00"
    source_trajectory_path = source_trajectory_dir / "trajectory.csv"
    source_trajectory_lock_path = source_trajectory_dir / "TRAJECTORY.lock.json"
    source_trajectory_lock = load_json(source_trajectory_lock_path)
    _require(
        source_trajectory_lock.get("status") == "locked_complete_method_selected_trajectory",
        "Reused NeuroDiscovery trajectory is incomplete",
    )
    _require(
        sha256_file(source_trajectory_path) == source_trajectory_lock["trajectory_sha256"],
        "Reused NeuroDiscovery trajectory hash changed",
    )
    _require(
        sha256_file(source_run / "static_initial_ranking.csv")
        == source_trajectory_lock["static_initial_ranking_sha256"],
        "Reused NeuroDiscovery static ranking changed",
    )
    trajectory = pd.read_csv(source_trajectory_path, dtype={"candidate_id": str})
    _require(len(trajectory) == 168, "Reused NeuroDiscovery trajectory length changed")
    _require(trajectory["action_slot"].tolist() == list(range(1, 169)), "Reused action slots changed")
    _require(trajectory["candidate_id"].is_unique, "Reused NeuroDiscovery candidates repeat")
    _require(set(trajectory["candidate_id"]) == set(oracle["candidate_id"]), "Reused universe changed")

    selected_in_rounds: list[str] = []
    feedback_columns = (
        "candidate_id",
        "complete_case_n",
        "a_path_std",
        "a_path_hc3_p",
        "b_path_std",
        "b_path_hc3_p",
        "indirect_effect_std",
        "indirect_bootstrap_ci_lower",
        "indirect_bootstrap_ci_upper",
        "indirect_bootstrap_p",
        "feedback_status",
    )
    oracle_indexed = oracle.set_index("candidate_id", drop=False)
    for round_index, requested in enumerate(config["sequential_design"]["main_round_batch_sizes"]):
        round_dir = source_run / f"round_{round_index:02d}_main"
        selection_path = round_dir / "selection.csv"
        selection_lock_path = round_dir / "SELECTION.lock.json"
        feedback_path = round_dir / "feedback.csv"
        round_lock_path = round_dir / "ROUND.lock.json"
        selection_lock = load_json(selection_lock_path)
        round_lock = load_json(round_lock_path)
        _require(selection_lock["selection_sha256"] == sha256_file(selection_path), "Selection drift")
        _require(
            round_lock["selection_lock_sha256"] == sha256_file(selection_lock_path),
            "Selection-lock drift",
        )
        _require(round_lock["feedback_sha256"] == sha256_file(feedback_path), "Feedback drift")
        _require(selection_lock["requested_actions"] == int(requested), "Round size changed")
        _require(selection_lock["outcomes_accessed_before_commit"] is False, "Selection leaked outcomes")
        _require(selection_lock["automatic_completion"] is False, "Selection was auto-completed")
        _require(round_lock["q_values_revealed"] is False, "Reused round revealed q values")
        _require(
            round_lock["unselected_results_revealed"] is False,
            "Reused round revealed unselected outcomes",
        )
        selection = pd.read_csv(selection_path, dtype={"candidate_id": str})
        feedback = pd.read_csv(feedback_path, dtype={"candidate_id": str})
        ids = selection.loc[selection["valid_selected"].astype(bool), "candidate_id"].tolist()
        _require(len(ids) == int(requested), "Reused NeuroDiscovery round is incomplete")
        _require(feedback["candidate_id"].tolist() == ids, "Reused feedback order changed")
        for row in feedback.loc[:, feedback_columns].to_dict("records"):
            expected = oracle_indexed.loc[row["candidate_id"]]
            _require(int(row["complete_case_n"]) == int(expected["complete_case_n"]), "Feedback n changed")
            for column in feedback_columns[2:-1]:
                _require(
                    math.isclose(float(row[column]), float(expected[column]), rel_tol=1e-12, abs_tol=1e-12),
                    f"Reused feedback value changed: {row['candidate_id']}:{column}",
                )
            _require(str(row["feedback_status"]) == str(expected["feedback_status"]), "Feedback status changed")
        selected_in_rounds.extend(ids)
    _require(
        selected_in_rounds == trajectory["candidate_id"].tolist(),
        "Reused trajectory differs from its committed round selections",
    )

    copied_run = staging / "runs/neurodiscovery/trial_00"
    copied_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run, copied_run)
    copied_trajectory_dir = staging / "trajectories/neurodiscovery/trial_00"
    copied_trajectory_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_trajectory_path, copied_trajectory_dir / "trajectory.csv")
    provenance_dir = staging / "provenance/neurodiscovery_hybrid_resume"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_trajectory_lock_path, provenance_dir / "SOURCE_TRAJECTORY.lock.json")
    shutil.copy2(source_audit_path, provenance_dir / "SOURCE_FROZEN_WORKTREE_AUDIT.json")

    imported_lock = dict(source_trajectory_lock)
    imported_lock.update(
        {
            "schema_version": "neurooracle.case2_closed_loop_trajectory_lock.v3",
            "benchmark_id": config["benchmark_id"],
            "execution_provenance": "hash-verified hybrid import from the superseded v2 freeze",
            "source_benchmark_id": source_protocol["benchmark_id"],
            "source_freeze_id": expected_freeze,
            "source_trajectory_lock_sha256": sha256_file(source_trajectory_lock_path),
            "model_provider_used": False,
            "provider_backend_change_affects_this_method": False,
        }
    )
    write_json(copied_trajectory_dir / "TRAJECTORY.lock.json", imported_lock)
    source_run_pins = artifact_pins(source_run)
    copied_run_pins = artifact_pins(copied_run)
    _require(source_run_pins == copied_run_pins, "Copied NeuroDiscovery run tree changed")
    audit = {
        "schema_version": "neurooracle.case2_neurodiscovery_hybrid_resume.v1",
        "status": "verified_and_imported",
        "source_benchmark_id": source_protocol["benchmark_id"],
        "source_freeze_id": expected_freeze,
        "destination_benchmark_id": config["benchmark_id"],
        "method": "neurodiscovery",
        "source_ready_sha256": sha256_file(source_ready_path),
        "source_freeze_lock_sha256": sha256_file(source_lock_path),
        "source_worktree_audit_sha256": sha256_file(source_audit_path),
        "source_trajectory_sha256": sha256_file(source_trajectory_path),
        "source_trajectory_lock_sha256": sha256_file(source_trajectory_lock_path),
        "source_run_file_count": len(source_run_pins),
        "source_run_tree_sha256": canonical_sha(source_run_pins),
        "copied_run_tree_sha256": canonical_sha(copied_run_pins),
        "candidate_count": 168,
        "round_count": 8,
        "seed_and_schedule_identical": True,
        "public_registry_identical": True,
        "feedback_matches_current_oracle": True,
        "result_values_manually_edited": False,
        "model_provider_used": False,
        "provider_backend_change_affects_reused_method": False,
    }
    write_json(provenance_dir / "IMPORT_AUDIT.json", audit)
    return audit


def audit_superseded_execution_attempt(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Verify that a superseded pre-selection run produced no usable answer."""

    attempt = config.get("superseded_execution_attempt")
    if attempt is None:
        return None
    _require(isinstance(attempt, Mapping), "Superseded attempt must be one object")
    source_root = Path(str(attempt["source_root"]))
    source_lock_path = source_root / "BENCHMARK_FREEZE.lock.json"
    source_ready_path = source_root / "READY_TO_RUN.json"
    source_audit_path = source_root / "runtime/FROZEN_WORKTREE_AUDIT.json"
    source_lock = load_json(source_lock_path)
    source_ready = load_json(source_ready_path)
    source_audit = load_json(source_audit_path)
    expected_freeze = str(attempt["source_freeze_id"])
    expected_lock_sha = str(attempt["source_lock_sha256"])
    request_id = str(attempt["failed_request_id"])
    _require(source_lock.get("freeze_id") == expected_freeze, "Superseded attempt freeze changed")
    _require(source_ready.get("freeze_id") == expected_freeze, "Superseded attempt ready changed")
    _require(source_audit.get("freeze_id") == expected_freeze, "Superseded attempt worktree audit changed")
    _require(sha256_file(source_lock_path) == expected_lock_sha, "Superseded attempt lock hash changed")
    _require(
        source_ready.get("benchmark_lock_sha256") == expected_lock_sha,
        "Superseded attempt ready lock hash changed",
    )
    selection_lock = source_root / "runs/open_coscientist/trial_00/round_00_main/SELECTION.lock.json"
    _require(not selection_lock.exists(), "Superseded attempt created a round-0 selection lock")
    response_meta = source_root / f"runtime/lg/responses/{request_id}.meta.json"
    _require(not response_meta.exists(), "Superseded invalid response was sealed")
    request_path = source_root / f"runtime/lg/requests/{request_id}.json"
    request = load_json(request_path)
    _require(request.get("request_id") == request_id, "Superseded request identity changed")
    _require(
        request.get("thread_id") == attempt["provider_thread_id"],
        "Superseded request provider changed",
    )
    invalid_rows: list[dict[str, Any]] = []
    suffix = request_id[-16:]
    for index in range(1, int(attempt["failed_capture_attempt_count"]) + 1):
        audit_path = source_root / f"runtime/lg/c/{suffix}/a{index}.json"
        raw_path = source_root / f"runtime/lg/f/{suffix}.a{index}.txt"
        capture_audit = load_json(audit_path)
        _require(capture_audit.get("request_id") == request_id, "Failed capture request changed")
        _require(capture_audit.get("attempt") == index, "Failed capture attempt changed")
        _require(capture_audit.get("status") == "invalid_provider_final", "Failed capture status changed")
        _require(raw_path.is_file(), "Failed raw capture is missing")
        envelope = validate_envelope(json.loads(raw_path.read_text(encoding="utf-8-sig")))
        try:
            validate_inner_response_schema(request, envelope)
        except Exception as exc:
            invalid_rows.append(
                {
                    "attempt": index,
                    "capture_audit_sha256": sha256_file(audit_path),
                    "raw_response_sha256": sha256_file(raw_path),
                    "validation_error_type": type(exc).__name__,
                    "validation_error": str(exc)[:500],
                }
            )
        else:
            raise ValueError(f"Superseded failed capture unexpectedly validates: attempt {index}")
    evidence_pins: dict[str, Any] = {}
    for relative in attempt["evidence_relative_paths"]:
        relative_path = Path(str(relative))
        evidence_pins[relative_path.as_posix()] = file_pin(source_root / relative_path)
    return {
        "schema_version": "neurooracle.case2_superseded_preselection_attempt.v1",
        "status": "verified_safe_to_supersede",
        "source_freeze_id": expected_freeze,
        "source_freeze_lock_sha256": expected_lock_sha,
        "provider_thread_id": attempt["provider_thread_id"],
        "failed_request_id": request_id,
        "failed_capture_attempt_count": len(invalid_rows),
        "failed_captures": invalid_rows,
        "selection_lock_created": False,
        "response_sealed": False,
        "formal_candidate_feedback_revealed": False,
        "evidence_pins": evidence_pins,
    }


def import_provider_migration_hybrid_resume(
    staging: Path,
    *,
    public_path: Path,
    oracle: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Import one completed trajectory and only fully sealed replay responses.

    The import never turns an incomplete request into a response and never
    changes response bytes. Every source thread identity remains attached to
    its original request, seal, receipt, and capture. The same verifier also
    supports an outcome-blind compiler amendment and a later quota-driven
    provider-thread migration after partial execution; neither path imports an
    unsealed request or edits a sealed response.
    """

    migration = config.get("provider_migration")
    if migration is None:
        return None
    _require(isinstance(migration, Mapping), "Provider migration must be one object")
    migration_kind = str(
        migration.get("migration_kind") or "provider_thread_change_v1"
    )
    source_root = Path(str(migration["source_root"]))
    source_ready_path = source_root / "READY_TO_RUN.json"
    source_lock_path = source_root / "BENCHMARK_FREEZE.lock.json"
    source_protocol_path = source_root / "PROTOCOL.json"
    source_audit_path = source_root / "runtime/FROZEN_WORKTREE_AUDIT.json"
    source_ready = load_json(source_ready_path)
    source_lock = load_json(source_lock_path)
    source_protocol = load_json(source_protocol_path)
    source_audit = load_json(source_audit_path)
    expected_freeze = str(migration["source_freeze_id"])
    expected_lock_sha = str(migration["source_lock_sha256"])
    old_thread_id = str(migration["old_thread_id"])
    new_thread_id = str(migration["new_thread_id"])
    source_thread_ids = tuple(
        str(value)
        for value in (migration.get("source_thread_ids") or (old_thread_id,))
    )
    _require(source_ready.get("freeze_id") == expected_freeze, "Migration source ready freeze changed")
    _require(source_lock.get("freeze_id") == expected_freeze, "Migration source freeze changed")
    _require(sha256_file(source_lock_path) == expected_lock_sha, "Migration source lock hash changed")
    _require(
        source_ready.get("benchmark_lock_sha256") == expected_lock_sha,
        "Migration source ready lock hash changed",
    )
    _require(source_audit.get("freeze_id") == expected_freeze, "Migration worktree audit changed")
    for key in (
        "source_pins_verified",
        "local_import_closure_pinned",
        "static_runtime_assets_pinned",
        "runner_import_smoke_passed",
        "luna_thread_gateway_import_smoke_passed",
        "luna_cli_worker_import_smoke_passed",
        "primary_workspace_isolated",
    ):
        _require(source_audit.get(key) is True, f"Migration source audit failed: {key}")
    _require(
        source_protocol["model_backend"]["codex_thread_id"] == old_thread_id,
        "Migration source provider thread changed",
    )
    _require(
        source_protocol["sequential_design"] == config["sequential_design"],
        "Migration seed or sequential schedule changed",
    )
    _require(
        tuple(source_protocol["methods"]["all"]) == EXPECTED_METHODS,
        "Migration method suite changed",
    )
    source_public = source_root / "generator_inputs/PUBLIC_CANDIDATE_REGISTRY.csv"
    _require(
        sha256_file(source_public) == sha256_file(public_path),
        "Migration candidate registry changed",
    )

    compiler_amendment_audit: dict[str, Any] | None = None
    compiler_evidence_sources: list[tuple[Path, Path]] = []
    if migration_kind == "outcome_blind_compiler_amendment_v1":
        evidence_relatives = [
            Path(str(value))
            for value in migration["compiler_amendment_evidence_relative_paths"]
        ]
        _require(
            len({value.as_posix() for value in evidence_relatives})
            == len(evidence_relatives),
            "Compiler-amendment evidence paths repeat",
        )
        evidence_pins: dict[str, dict[str, Any]] = {}
        for relative in evidence_relatives:
            source = source_root / relative
            _require(
                source.is_file(),
                f"Compiler-amendment evidence is missing: {relative.as_posix()}",
            )
            evidence_pins[relative.as_posix()] = file_pin(source)
            compiler_evidence_sources.append((source, relative))

        failed_round = source_root / "runs/open_coscientist/trial_00/round_00_main"
        selection_lock = load_json(failed_round / "SELECTION.lock.json")
        compile_audit = load_json(failed_round / "native_compile_audit.json")
        _require(
            int(selection_lock.get("requested_actions", -1)) == 21
            and int(selection_lock.get("valid_selected", -1)) == 0
            and int(selection_lock.get("failed_actions", -1)) == 21,
            "Superseded compiler failure signature changed",
        )
        _require(
            selection_lock.get("outcomes_accessed_before_commit") is False,
            "Superseded compiler attempt accessed outcomes before selection commit",
        )
        _require(
            compile_audit.get("mapping_mode") == "deterministic_exact_native_only"
            and int(compile_audit.get("exact_mentions_seen", -1)) == 0
            and int(compile_audit.get("valid_unique_anchors", -1)) == 0,
            "Superseded native compiler diagnosis changed",
        )
        _require(
            not (
                source_root
                / "runs/open_coscientist/trial_00/round_01_main/SELECTION.lock.json"
            ).exists(),
            "Superseded compiler attempt unexpectedly committed round 1",
        )
        compiler_amendment_audit = {
            "schema_version": "neurooracle.case2_compiler_amendment_source_audit.v1",
            "status": "verified_outcome_blind_interface_failure",
            "source_freeze_id": expected_freeze,
            "method": "open_coscientist",
            "failed_round": 0,
            "requested_actions": 21,
            "valid_selected": 0,
            "failed_actions": 21,
            "exact_mentions_seen": 0,
            "source_mapping_mode": "deterministic_exact_native_only",
            "outcomes_accessed_before_commit": False,
            "round_1_selection_committed": False,
            "evidence_pins": evidence_pins,
        }

    method = "ai_scientist_v2"
    source_run = source_root / f"runs/{method}/trial_00"
    source_trajectory_dir = source_root / f"trajectories/{method}/trial_00"
    source_trajectory_path = source_trajectory_dir / "trajectory.csv"
    source_trajectory_lock_path = source_trajectory_dir / "TRAJECTORY.lock.json"
    source_trajectory_lock = load_json(source_trajectory_lock_path)
    _require(
        source_trajectory_lock.get("status") == "locked_complete_method_selected_trajectory",
        "Migrated AI Scientist-v2 trajectory is incomplete",
    )
    _require(
        source_trajectory_lock.get("trajectory_sha256") == sha256_file(source_trajectory_path),
        "Migrated AI Scientist-v2 trajectory hash changed",
    )
    _require(
        int(source_trajectory_lock.get("main_action_slots", -1)) == 168
        and int(source_trajectory_lock.get("real_candidate_count", -1)) == 168
        and source_trajectory_lock.get("all_real_candidates_method_selected") is True
        and source_trajectory_lock.get("automatic_completion") is False,
        "Migrated AI Scientist-v2 trajectory contract changed",
    )
    trajectory = pd.read_csv(source_trajectory_path, dtype={"candidate_id": str})
    _require(len(trajectory) == 168, "Migrated AI Scientist-v2 trajectory length changed")
    _require(trajectory["action_slot"].tolist() == list(range(1, 169)), "Migrated action slots changed")
    _require(trajectory["candidate_id"].is_unique, "Migrated AI Scientist-v2 candidates repeat")
    _require(
        set(trajectory["candidate_id"]) == set(oracle["candidate_id"]),
        "Migrated AI Scientist-v2 candidate universe changed",
    )

    feedback_columns = (
        "candidate_id",
        "complete_case_n",
        "a_path_std",
        "a_path_hc3_p",
        "b_path_std",
        "b_path_hc3_p",
        "indirect_effect_std",
        "indirect_bootstrap_ci_lower",
        "indirect_bootstrap_ci_upper",
        "indirect_bootstrap_p",
        "feedback_status",
    )
    oracle_indexed = oracle.set_index("candidate_id", drop=False)
    selected_in_rounds: list[str] = []
    for round_index, requested in enumerate(config["sequential_design"]["main_round_batch_sizes"]):
        round_dir = source_run / f"round_{round_index:02d}_main"
        selection_path = round_dir / "selection.csv"
        selection_lock_path = round_dir / "SELECTION.lock.json"
        feedback_path = round_dir / "feedback.csv"
        round_lock_path = round_dir / "ROUND.lock.json"
        selection_lock = load_json(selection_lock_path)
        round_lock = load_json(round_lock_path)
        _require(selection_lock.get("selection_sha256") == sha256_file(selection_path), "Migrated selection drift")
        _require(
            round_lock.get("selection_lock_sha256") == sha256_file(selection_lock_path),
            "Migrated selection-lock drift",
        )
        _require(round_lock.get("feedback_sha256") == sha256_file(feedback_path), "Migrated feedback drift")
        _require(int(selection_lock.get("requested_actions", -1)) == int(requested), "Migrated round size changed")
        _require(selection_lock.get("outcomes_accessed_before_commit") is False, "Migrated selection leaked outcomes")
        _require(selection_lock.get("automatic_completion") is False, "Migrated selection was auto-completed")
        _require(round_lock.get("q_values_revealed") is False, "Migrated round revealed q values")
        _require(round_lock.get("unselected_results_revealed") is False, "Migrated round revealed unselected results")
        selection = pd.read_csv(selection_path, dtype={"candidate_id": str})
        feedback = pd.read_csv(feedback_path, dtype={"candidate_id": str})
        ids = selection.loc[selection["valid_selected"].astype(bool), "candidate_id"].tolist()
        _require(len(ids) == int(requested), "Migrated AI Scientist-v2 round is incomplete")
        _require(feedback["candidate_id"].tolist() == ids, "Migrated feedback order changed")
        for row in feedback.loc[:, feedback_columns].to_dict("records"):
            expected = oracle_indexed.loc[row["candidate_id"]]
            _require(int(row["complete_case_n"]) == int(expected["complete_case_n"]), "Migrated feedback n changed")
            for column in feedback_columns[2:-1]:
                _require(
                    math.isclose(float(row[column]), float(expected[column]), rel_tol=1e-12, abs_tol=1e-12),
                    f"Migrated feedback value changed: {row['candidate_id']}:{column}",
                )
            _require(str(row["feedback_status"]) == str(expected["feedback_status"]), "Migrated feedback status changed")
        selected_in_rounds.extend(ids)
    _require(
        selected_in_rounds == trajectory["candidate_id"].tolist(),
        "Migrated trajectory differs from its committed round selections",
    )

    copied_run = staging / f"runs/{method}/trial_00"
    copied_run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_run, copied_run)
    copied_trajectory_dir = staging / f"trajectories/{method}/trial_00"
    copied_trajectory_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_trajectory_path, copied_trajectory_dir / "trajectory.csv")
    source_run_pins = artifact_pins(source_run)
    copied_run_pins = artifact_pins(copied_run)
    _require(source_run_pins == copied_run_pins, "Copied AI Scientist-v2 run tree changed")

    provenance_dir = staging / "provenance/provider_migration_hybrid_resume"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (source_ready_path, "SOURCE_READY_TO_RUN.json"),
        (source_lock_path, "SOURCE_BENCHMARK_FREEZE.lock.json"),
        (source_protocol_path, "SOURCE_PROTOCOL.json"),
        (source_audit_path, "SOURCE_FROZEN_WORKTREE_AUDIT.json"),
        (source_trajectory_lock_path, "SOURCE_AI_SCIENTIST_TRAJECTORY.lock.json"),
    ):
        shutil.copy2(source, provenance_dir / name)
    if compiler_amendment_audit is not None:
        evidence_root = provenance_dir / "compiler_amendment_evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        copied_evidence_paths: dict[str, str] = {}
        for index, (source, relative) in enumerate(compiler_evidence_sources):
            # Keep the UNC staging path below the legacy Windows MAX_PATH limit.
            # The original relative path remains the stable audit key, while the
            # copy uses a short deterministic name plus a verified content hash.
            copied_name = f"evidence_{index:02d}{relative.suffix}"
            target = evidence_root / copied_name
            shutil.copy2(source, target)
            _require(
                sha256_file(source) == sha256_file(target),
                f"Compiler-amendment evidence copy changed: {relative.as_posix()}",
            )
            copied_evidence_paths[relative.as_posix()] = copied_name
        compiler_amendment_audit["copied_evidence_paths"] = copied_evidence_paths
        write_json(
            provenance_dir / "COMPILER_AMENDMENT_SOURCE_AUDIT.json",
            compiler_amendment_audit,
        )
    imported_trajectory_lock = dict(source_trajectory_lock)
    imported_trajectory_lock.update(
        {
            "execution_provenance": (
                "hash-verified outcome-blind compiler-amendment import"
                if migration_kind == "outcome_blind_compiler_amendment_v1"
                else "hash-verified provider-migration import from the superseded v3e freeze"
            ),
            "source_freeze_id": expected_freeze,
            "source_trajectory_lock_sha256": sha256_file(source_trajectory_lock_path),
            "model_provider_used": True,
            "source_provider_thread_id": old_thread_id,
            "destination_provider_thread_id": new_thread_id,
            "provider_backend_change_affects_this_completed_method": False,
            "response_content_manually_edited": False,
        }
    )
    write_json(copied_trajectory_dir / "TRAJECTORY.lock.json", imported_trajectory_lock)

    source_lg = source_root / "runtime/lg"
    target_lg = staging / "runtime/lg"
    for name in ("requests", "dispatches", "responses", "receipts", "c", "c_recovery", "downstream_invalid"):
        (target_lg / name).mkdir(parents=True, exist_ok=True)
    copied_cache_relatives: list[str] = []

    def copy_cache_file(relative: Path) -> None:
        source = source_lg / relative
        target = target_lg / relative
        _require(source.is_file(), f"Migration cache file is missing: {relative.as_posix()}")
        _require(not target.exists(), f"Migration cache target already exists: {relative.as_posix()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        _require(sha256_file(source) == sha256_file(target), f"Migration cache copy changed: {relative.as_posix()}")
        copied_cache_relatives.append(relative.as_posix())

    def copy_optional_cache_tree(relative: Path) -> None:
        source = source_lg / relative
        if not source.is_dir():
            return
        target = target_lg / relative
        _require(not target.exists(), f"Migration cache directory already exists: {relative.as_posix()}")
        shutil.copytree(source, target)
        for item in sorted(path for path in source.rglob("*") if path.is_file()):
            item_relative = relative / item.relative_to(source)
            target_item = target_lg / item_relative
            _require(sha256_file(item) == sha256_file(target_item), f"Migration capture copy changed: {item_relative.as_posix()}")
            copied_cache_relatives.append(item_relative.as_posix())

    sealed_ids: list[str] = []
    unsealed_ids: list[str] = []
    downstream_invalid_responses: list[dict[str, str]] = []
    structured_response_count = 0
    inner_schema_hashes: dict[str, str] = {}
    imported_response_thread_counts: dict[str, int] = {}
    expected_downstream_invalid = sorted(
        str(value)
        for value in migration.get("excluded_downstream_invalid_response_ids") or ()
    )
    request_paths = sorted((source_lg / "requests").glob("luna_*.json"))
    for request_path in request_paths:
        request_id = request_path.stem
        record = load_json(request_path)
        computed_id, computed_sha = request_identity(str(record["endpoint"]), record["request_body"])
        _require(computed_id == request_id, f"Migration request ID changed: {request_id}")
        _require(computed_sha == record.get("request_sha256"), f"Migration request hash changed: {request_id}")
        record_thread_id = str(record.get("thread_id") or "")
        _require(
            record_thread_id in source_thread_ids,
            f"Migration request provider changed: {request_id}",
        )
        _require(record.get("host_id") == "local", f"Migration request host changed: {request_id}")
        dispatch_relative = Path("dispatches") / f"{request_id}.txt"
        dispatch_path = source_lg / dispatch_relative
        _require(dispatch_path.is_file(), f"Migration dispatch is missing: {request_id}")
        _require(
            dispatch_path.read_text(encoding="utf-8-sig") == render_dispatch(record),
            f"Migration dispatch bytes changed: {request_id}",
        )
        raw_path = source_lg / "responses" / f"{request_id}.txt"
        meta_path = source_lg / "responses" / f"{request_id}.meta.json"
        _require(raw_path.is_file() == meta_path.is_file(), f"Partial sealed response: {request_id}")
        if not raw_path.is_file():
            unsealed_ids.append(request_id)
            continue
        envelope, _meta, verified_raw_path, verified_meta_path = _response_artifacts(source_lg, record)
        try:
            inner_validation = validate_inner_response_schema(record, envelope)
        except Exception as exc:
            _require(
                request_id in expected_downstream_invalid,
                f"Unexpected downstream-invalid sealed response {request_id}: {type(exc).__name__}: {exc}",
            )
            downstream_invalid_responses.append(
                {
                    "request_id": request_id,
                    "raw_response_sha256": sha256_file(verified_raw_path),
                    "response_meta_sha256": sha256_file(verified_meta_path),
                    "validation_error_type": type(exc).__name__,
                    "validation_error": str(exc)[:500],
                }
            )
            continue
        _require(
            request_id not in expected_downstream_invalid,
            f"Expected downstream-invalid response unexpectedly validated: {request_id}",
        )
        if inner_validation.get("required"):
            structured_response_count += 1
            if inner_validation.get("inner_schema_sha256"):
                inner_schema_hashes[request_id] = str(inner_validation["inner_schema_sha256"])
        receipt_relative = Path("receipts") / f"{request_id}.json"
        receipt_path = source_lg / receipt_relative
        _require(receipt_path.is_file(), f"Migration receipt is missing: {request_id}")
        receipt = load_json(receipt_path)
        _require(receipt.get("request_id") == request_id, f"Migration receipt ID changed: {request_id}")
        _require(receipt.get("request_sha256") == computed_sha, f"Migration receipt request hash changed: {request_id}")
        _require(
            receipt.get("thread_id") == record_thread_id,
            f"Migration receipt provider changed: {request_id}",
        )
        _require(receipt.get("raw_response_sha256") == sha256_file(verified_raw_path), f"Migration receipt raw hash changed: {request_id}")
        _require(receipt.get("response_meta_sha256") == sha256_file(verified_meta_path), f"Migration receipt meta hash changed: {request_id}")
        wire = chat_response(record, envelope)
        wire_sha = gateway_sha256_bytes(gateway_canonical_json(wire).encode("utf-8"))
        _require(receipt.get("wire_response_sha256") == wire_sha, f"Migration receipt wire hash changed: {request_id}")
        for relative in (
            Path("requests") / f"{request_id}.json",
            dispatch_relative,
            Path("responses") / f"{request_id}.txt",
            Path("responses") / f"{request_id}.meta.json",
            receipt_relative,
        ):
            copy_cache_file(relative)
        capture_suffix = request_id[-16:]
        copy_optional_cache_tree(Path("c") / capture_suffix)
        copy_optional_cache_tree(Path("c_recovery") / capture_suffix)
        sealed_ids.append(request_id)
        imported_response_thread_counts[record_thread_id] = (
            imported_response_thread_counts.get(record_thread_id, 0) + 1
        )

    expected_unsealed = sorted(
        str(value) for value in migration.get("excluded_unsealed_request_ids") or ()
    )
    _require(sorted(unsealed_ids) == expected_unsealed, "Migration unsealed request set changed")
    _require(
        sorted(row["request_id"] for row in downstream_invalid_responses)
        == expected_downstream_invalid,
        "Migration downstream-invalid response set changed",
    )
    expected_importable = int(migration["expected_importable_sealed_request_count"])
    expected_source_sealed = int(migration["expected_source_sealed_response_count"])
    _require(len(sealed_ids) == expected_importable, "Migration importable sealed response count changed")
    _require(
        len(sealed_ids) + len(downstream_invalid_responses) == expected_source_sealed,
        "Migration source sealed response count changed",
    )
    _require(
        len(request_paths)
        == expected_importable + len(expected_downstream_invalid) + len(expected_unsealed),
        "Migration request count changed",
    )
    target_cache_pins = artifact_pins(target_lg)
    _require(
        sorted(target_cache_pins) == sorted(copied_cache_relatives),
        "Migration cache contains an untracked file",
    )
    migration_audit = {
        "schema_version": "neurooracle.case2_provider_migration_hybrid_resume.v1",
        "status": "verified_and_imported",
        "migration_kind": migration_kind,
        "source_benchmark_id": source_protocol["benchmark_id"],
        "source_freeze_id": expected_freeze,
        "source_freeze_lock_sha256": expected_lock_sha,
        "destination_benchmark_id": config["benchmark_id"],
        "old_provider_thread_id": old_thread_id,
        "new_provider_thread_id": new_thread_id,
        "allowed_source_thread_ids": list(source_thread_ids),
        "imported_response_thread_counts": imported_response_thread_counts,
        "reused_completed_methods": [method],
        "source_ai_scientist_trajectory_sha256": sha256_file(source_trajectory_path),
        "source_ai_scientist_trajectory_lock_sha256": sha256_file(source_trajectory_lock_path),
        "source_ai_scientist_run_file_count": len(source_run_pins),
        "source_ai_scientist_run_tree_sha256": canonical_sha(source_run_pins),
        "copied_ai_scientist_run_tree_sha256": canonical_sha(copied_run_pins),
        "source_request_count": len(request_paths),
        "source_outer_sealed_response_count": expected_source_sealed,
        "imported_sealed_request_count": len(sealed_ids),
        "excluded_unsealed_request_ids": expected_unsealed,
        "excluded_downstream_invalid_responses": downstream_invalid_responses,
        "imported_request_ids": sealed_ids,
        "structured_response_count": structured_response_count,
        "inner_schema_hashes": inner_schema_hashes,
        "copied_cache_file_count": len(target_cache_pins),
        "copied_cache_tree_sha256": canonical_sha(target_cache_pins),
        "downstream_invalid_attempts_imported": False,
        "downstream_invalid_attempts_preserved_at_source": bool(
            expected_downstream_invalid
        ),
        "downstream_invalid_import_reason": (
            "Not required for replay; left immutable at the superseded source to avoid Windows UNC long-path rewriting"
            if expected_downstream_invalid
            else None
        ),
        "candidate_registry_identical": True,
        "seed_and_schedule_identical": True,
        "feedback_matches_current_oracle": True,
        "response_bytes_manually_edited": False,
        "unsealed_request_imported": False,
        "new_provider_calls_made_during_import": False,
        "compiler_amendment_source_audit": compiler_amendment_audit,
    }
    write_json(provenance_dir / "IMPORT_AUDIT.json", migration_audit)
    return migration_audit


def _write_runbook(
    path: Path, config_path: Path, output_root: Path, config: Mapping[str, Any]
) -> None:
    backend = config["model_backend"]
    thread_id = backend["codex_thread_id"]
    host_id = backend["codex_host_id"]
    provider_attestation = load_json(Path(config["paths"]["provider_attestation"]))
    provider_workspace = provider_attestation["provider_workspace"]
    baseline_root = config["paths"]["baseline_repository_root"]
    migration = config.get("provider_migration") or {}
    if migration.get("migration_kind") == "outcome_blind_compiler_amendment_v1":
        resume_description = (
            "This freeze imports the completed NeuroDiscovery and AI Scientist-v2 "
            "trajectories plus every fully sealed response from the superseded v3g "
            "freeze. It preserves each response's original thread identity. The v3g "
            "Open CoScientist 0/21 exact-ID interface failure is pinned as diagnostic "
            "evidence; no v3g Open CoScientist selection is imported."
        )
    else:
        expected_source = int(
            migration.get("expected_source_sealed_response_count", 0)
        )
        expected_importable = int(
            migration.get("expected_importable_sealed_request_count", 0)
        )
        excluded_unsealed = len(
            migration.get("excluded_unsealed_request_ids") or []
        )
        excluded_invalid = len(
            migration.get("excluded_downstream_invalid_response_ids") or []
        )
        resume_description = (
            "This freeze imports the completed NeuroDiscovery and AI Scientist-v2 "
            f"trajectories plus {expected_importable} exact sealed, downstream-valid "
            f"responses from {expected_source} sealed source responses. It explicitly "
            f"excludes {excluded_unsealed} unsealed request(s) and {excluded_invalid} "
            "downstream-invalid response(s). Imported records retain their source thread "
            "IDs, while newly generated responses use the destination thread ID."
        )
    text = f"""# Case Study 2 single-seed closed-loop pilot

This is a post-result-design-freeze exploratory pilot. It cannot support a clear-SOTA or significance claim.

{resume_description}

```powershell
.\\.venv-case2-benchmark\\Scripts\\python.exe neurooracle\\scripts\\prepare_case2_adni_closed_loop_benchmark.py --config \"{config_path}\" --verify-only

powershell -ExecutionPolicy Bypass -File neurooracle\\scripts\\start_case2_closed_loop_luna_thread_gateway.ps1 -DataDir \"{output_root}\\runtime\\lg\" -ThreadId \"{thread_id}\" -HostId \"{host_id}\" -Python \"{REPO_ROOT}\\.venv-case2-benchmark\\Scripts\\python.exe\"

powershell -ExecutionPolicy Bypass -File neurooracle\\scripts\\start_case2_closed_loop_luna_cli_worker.ps1 -DataDir \"{output_root}\\runtime\\lg\" -ThreadId \"{thread_id}\" -HostId \"{host_id}\" -ProviderWorkspace \"{provider_workspace}\" -Python \"{REPO_ROOT}\\.venv-case2-benchmark\\Scripts\\python.exe\"

powershell -ExecutionPolicy Bypass -File neurooracle\\scripts\\start_case2_formal_brainpilot_service.ps1 -DataDir \"{output_root}\\runtime\\brainpilot\" -Model \"gpt-5.6-luna\" -ReasoningEffort \"max\" -BaseUrl \"http://127.0.0.1:18083/v1\" -BaselineRoot \"{baseline_root}\"

.\\.venv-case2-benchmark\\Scripts\\python.exe neurooracle\\scripts\\run_case2_adni_closed_loop_benchmark.py --benchmark-root \"{output_root}\" --brainpilot-runtime-manifest \"{output_root}\\runtime\\brainpilot\\runtime_manifest.json\"

.\\.venv-case2-benchmark\\Scripts\\python.exe neurooracle\\scripts\\evaluate_case2_adni_closed_loop_benchmark.py --benchmark-root \"{output_root}\"
```

Every round selection is hash-committed before selected-candidate feedback is revealed. No q value or unselected result enters a generator task. Invalid proposals consume action slots. Recovery rounds never use an evaluator/compiler completion rule.

For each pending model request, the frozen CLI worker copies the exact `runtime/lg/dispatches/<request_id>.txt` bytes to the designated `provider_workspace/CURRENT_DISPATCH.txt` path and verifies both SHA256 values. It resumes the designated Luna thread, lets it read only that current file, enforces the frozen envelope with Codex CLI `--output-schema`, and captures the provider final answer directly through `--output-last-message`. It then validates and hash-seals that exact response without manual transcription or content edits and removes the staged dispatch. This is a disclosed persistent Codex thread replay, not a DeepSeek API call.
"""
    path.write_text(text, encoding="utf-8")


def prepare(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)
    superseded_attempt_audit = audit_superseded_execution_attempt(config)
    paths = config["paths"]
    source_root = Path(paths["source_static_benchmark_root"])
    source_verify = verify_v1_integrity_for_v2(source_root, fast_large_files=True)
    _require(source_verify["freeze_id"] == config["source_static_freeze_id"], "v1 freeze ID changed")
    repository_input_audit = verify_v1_repository_execution_inputs(
        source_root,
        baseline_root_override=Path(paths["baseline_repository_root"]),
    )

    output_root = Path(paths["output_root"])
    if output_root.exists():
        raise FileExistsError(f"Closed-loop target already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename compact: the UNC benchmark root is already
    # close to the legacy Windows MAX_PATH boundary.
    staging = output_root.parent / f".c2l_{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Staging target exists: {staging}")
    staging.mkdir()

    generator = staging / "generator_inputs"
    evaluator = staging / "evaluator_only"
    generator.mkdir()
    evaluator.mkdir()
    adapter_relative = "core/scripts/case_study_official_adapter_client.py"
    source_adapter = REPO_ROOT / adapter_relative
    frozen_adapter = generator / "official_adapter.py"
    source_before = file_pin(source_adapter)
    shutil.copy2(source_adapter, frozen_adapter)
    source_after = file_pin(source_adapter)
    _require(
        (source_before["bytes"], source_before["sha256"])
        == (source_after["bytes"], source_after["sha256"]),
        "Shared official adapter changed while its v2 snapshot was being created",
    )
    _require(
        sha256_file(frozen_adapter) == source_before["sha256"],
        "Frozen official adapter copy changed",
    )
    source_v1_lock = load_json(source_root / "BENCHMARK_FREEZE.lock.json")
    v1_adapter_pin = source_v1_lock["lock_material"]["source_code_pins"][adapter_relative]
    write_json(
        generator / "FROZEN_RUNTIME_SOURCE_AUDIT.json",
        {
            "schema_version": "neurooracle.case2_closed_loop_frozen_runtime_source.v1",
            "relative_path": adapter_relative,
            "v1_locked_sha256": v1_adapter_pin["sha256"],
            "workspace_snapshot_sha256": source_before["sha256"],
            "frozen_copy_sha256": sha256_file(frozen_adapter),
            "workspace_drifted_from_v1": source_before["sha256"] != v1_adapter_pin["sha256"],
            "execution_path": frozen_adapter.relative_to(staging).as_posix(),
            "copy_stability_verified": True,
            "secret_values_present": False,
        },
    )
    source_public = source_root / "generator_inputs/PUBLIC_CANDIDATE_REGISTRY.csv"
    public_path = generator / "PUBLIC_CANDIDATE_REGISTRY.csv"
    shutil.copy2(source_public, public_path)
    public = pd.read_csv(public_path, dtype=str, keep_default_na=False)
    _require(tuple(public.columns) == PUBLIC_COLUMNS, "Public registry schema changed")
    _require(len(public) == 168 and public["candidate_id"].is_unique, "Public registry changed")

    contract = {
        "schema_version": "neurooracle.case2_closed_loop_generator_contract.v3",
        "benchmark_id": config["benchmark_id"],
        "candidate_count": 168,
        "methods": list(EXPECTED_METHODS),
        "trial": 0,
        "seed": int(config["sequential_design"]["seed"]),
        "main_round_batch_sizes": config["sequential_design"]["main_round_batch_sizes"],
        "primary_action_horizon": 168,
        "native_retrieval_enabled": False,
        "result_isolation": "Round zero contains public coordinates only. Later tasks receive observations only for candidates committed in earlier rounds.",
        "invalid_action_rule": "Invalid, repeated, missing or unmappable proposals consume action slots with zero utility.",
        "automatic_completion": False,
    }
    write_json(generator / "SHARED_TASK_CONTRACT.json", contract)
    leakage = audit_generator_bundle(generator)
    _require(leakage["status"] == "passed", f"Generator leakage audit failed: {leakage['findings']}")
    write_json(generator / "GENERATOR_LEAKAGE_AUDIT.json", leakage)

    oracle = build_oracle(Path(paths["formal_results"]), public)
    oracle_path = evaluator / "EXPERIMENT_ORACLE.csv"
    oracle.to_csv(oracle_path, index=False)
    reference_columns = (
        "candidate_id",
        "bootstrap_weakest_link_evidence",
        "absolute_indirect_effect",
        "supplemental_family_fdr_hit",
        "global_fdr_hit",
        "nominal_bootstrap_hit",
    )
    reference_path = evaluator / "REFERENCE_LABELS.csv"
    oracle.loc[:, reference_columns].to_csv(reference_path, index=False)
    write_json(
        evaluator / "EVALUATOR_CONTRACT.json",
        {
            "schema_version": "neurooracle.case2_closed_loop_evaluator_contract.v3",
            "benchmark_id": config["benchmark_id"],
            "visibility": "evaluator/controller only; never passed wholesale to a method",
            "oracle_sha256": sha256_file(oracle_path),
            "reference_sha256": sha256_file(reference_path),
            "feedback_status_count": int(oracle["feedback_status"].eq("supported").sum()),
            "nominal_hit_count": int(oracle["nominal_bootstrap_hit"].sum()),
            "family_hit_count": int(oracle["supplemental_family_fdr_hit"].sum()),
            "global_hit_count": int(oracle["global_fdr_hit"].sum()),
            "single_seed_descriptive_only": True,
        },
    )
    hybrid_audit = import_neurodiscovery_hybrid_resume(
        staging,
        source_root=Path(paths["reuse_neurodiscovery_from_root"]),
        public_path=public_path,
        oracle=oracle,
        config=config,
    )
    provider_migration_audit = import_provider_migration_hybrid_resume(
        staging,
        public_path=public_path,
        oracle=oracle,
        config=config,
    )

    write_json(staging / "PROTOCOL.json", config)
    provider_status = provider_attestation_preflight(Path(paths["provider_attestation"]), config)
    method_preflight = runtime_preflight(config)
    _require(all(row["status"] == "ok" for row in method_preflight), "Method preflight failed")
    write_json(
        staging / "METHOD_RUNTIME_LOCK.json",
        {
            "schema_version": "neurooracle.case2_closed_loop_runtime_lock.v3",
            "benchmark_id": config["benchmark_id"],
            "model_backend": config["model_backend"],
            "provider_attestation_preflight": provider_status,
            "benchmark_environment_preflight": benchmark_environment_preflight(),
            "method_preflight": method_preflight,
            "source_v1_runtime_lock_sha256": sha256_file(source_root / "METHOD_RUNTIME_LOCK.json"),
            "source_v1_integrity_audit": source_verify,
            "source_v1_repository_execution_input_audit": repository_input_audit,
            "hybrid_resume_audit": hybrid_audit,
            "provider_migration_hybrid_resume_audit": provider_migration_audit,
            "superseded_execution_attempt_audit": superseded_attempt_audit,
            "api_calls_made": False,
            "experiment_model_completions_made": False,
        },
    )
    _write_runbook(staging / "RUNBOOK.md", config_path, output_root, config)

    source_pins = {
        path.relative_to(REPO_ROOT).as_posix(): file_pin(path)
        for path in source_code_paths(config_path)
    }
    provider_attestation = load_json(Path(paths["provider_attestation"]))
    external_paths = {
        "source_v1_freeze_lock": source_root / "BENCHMARK_FREEZE.lock.json",
        "source_v1_ready": source_root / "READY_TO_RUN.json",
        "source_public_registry": source_public,
        "formal_results": Path(paths["formal_results"]),
        "formal_results_lock": Path(paths["formal_results_lock"]),
        "kg_current_state": Path(paths["kg_current_state"]),
        "knowledge_graph": Path(paths["knowledge_graph"]),
        "extracted_claims": Path(paths["extracted_claims"]),
        "pathway_catalog": Path(paths["pathway_catalog"]),
        "provider_attestation": Path(paths["provider_attestation"]),
        "reused_v2_freeze_lock": Path(paths["reuse_neurodiscovery_from_root"])
        / "BENCHMARK_FREEZE.lock.json",
        "reused_v2_ready": Path(paths["reuse_neurodiscovery_from_root"]) / "READY_TO_RUN.json",
        "reused_v2_worktree_audit": Path(paths["reuse_neurodiscovery_from_root"])
        / "runtime/FROZEN_WORKTREE_AUDIT.json",
        "reused_neurodiscovery_trajectory": Path(paths["reuse_neurodiscovery_from_root"])
        / "trajectories/neurodiscovery/trial_00/trajectory.csv",
        "reused_neurodiscovery_trajectory_lock": Path(paths["reuse_neurodiscovery_from_root"])
        / "trajectories/neurodiscovery/trial_00/TRAJECTORY.lock.json",
        "provider_ready_capture": Path(provider_attestation["ready_capture"]["path"]),
        "provider_dispatch_boundary_capture": Path(
            provider_attestation["dispatch_boundary_capture"]["path"]
        ),
        "provider_visibility_test_input": Path(
            provider_attestation["visibility_test"]["input_path"]
        ),
        "provider_visibility_test_response": Path(
            provider_attestation["visibility_test"]["response_path"]
        ),
        "provider_response_schema_format_test": Path(
            provider_attestation["response_schema_control"]["format_test_response_path"]
        ),
    }
    migration = config.get("provider_migration")
    if isinstance(migration, Mapping):
        migration_root = Path(str(migration["source_root"]))
        external_paths.update(
            {
                "provider_migration_source_freeze_lock": migration_root
                / "BENCHMARK_FREEZE.lock.json",
                "provider_migration_source_ready": migration_root / "READY_TO_RUN.json",
                "provider_migration_source_protocol": migration_root / "PROTOCOL.json",
                "provider_migration_source_worktree_audit": migration_root
                / "runtime/FROZEN_WORKTREE_AUDIT.json",
                "provider_migration_source_ai_scientist_trajectory": migration_root
                / "trajectories/ai_scientist_v2/trial_00/trajectory.csv",
                "provider_migration_source_ai_scientist_trajectory_lock": migration_root
                / "trajectories/ai_scientist_v2/trial_00/TRAJECTORY.lock.json",
            }
        )
        for index, relative in enumerate(
            migration.get("compiler_amendment_evidence_relative_paths") or ()
        ):
            external_paths[f"compiler_amendment_evidence_{index:02d}"] = (
                migration_root / str(relative)
            )
    superseded_attempt = config.get("superseded_execution_attempt")
    if isinstance(superseded_attempt, Mapping):
        superseded_root = Path(str(superseded_attempt["source_root"]))
        for index, relative in enumerate(superseded_attempt["evidence_relative_paths"]):
            external_paths[f"superseded_execution_evidence_{index:02d}"] = (
                superseded_root / str(relative)
            )
    external_pins = {name: file_pin(path) for name, path in external_paths.items()}
    artifacts = artifact_pins(staging)
    material = {
        "benchmark_id": config["benchmark_id"],
        "protocol_sha256": sha256_file(config_path),
        "source_v1_freeze_id": source_verify["freeze_id"],
        "source_v1_integrity_audit": source_verify,
        "source_v1_repository_execution_inputs_verified": True,
        "source_code_pins": source_pins,
        "external_input_pins": external_pins,
        "artifact_pins": artifacts,
        "seed": int(config["sequential_design"]["seed"]),
        "main_round_batch_sizes": config["sequential_design"]["main_round_batch_sizes"],
        "generator_leakage_audit": leakage,
        "hybrid_resume_audit": hybrid_audit,
        "provider_migration_hybrid_resume_audit": provider_migration_audit,
        "superseded_execution_attempt_audit": superseded_attempt_audit,
        "native_coordinate_compiler": config.get("native_coordinate_compiler"),
        "native_coordinate_compiler_sha256": (
            canonical_sha(config["native_coordinate_compiler"])
            if config.get("native_coordinate_compiler") is not None
            else None
        ),
        "api_calls_made": False,
        "experiment_model_completions_made": False,
    }
    freeze_id = canonical_sha(material)
    lock = {
        "schema_version": "neurooracle.case2_closed_loop_freeze_lock.v3",
        "benchmark_id": config["benchmark_id"],
        "freeze_id": freeze_id,
        "status": "locked_ready_for_single_seed_execution",
        "created_at_hkt": hkt_now(),
        "post_result_design_freeze": True,
        "single_seed_descriptive_only": True,
        "generator_result_isolation": True,
        "provider": "gpt-5.6-luna/max via codex_cli_thread_replay_direct_capture",
        "hybrid_resume": True,
        "reused_methods": ["neurodiscovery", "ai_scientist_v2"],
        "provider_migration_exact_response_cache_imported": provider_migration_audit is not None,
        "outcome_blind_coordinate_compiler_enabled": (
            config.get("native_coordinate_compiler") is not None
        ),
        "no_external_validation_queue": True,
        "lock_material": material,
    }
    lock_path = staging / "BENCHMARK_FREEZE.lock.json"
    write_json(lock_path, lock)
    ready = {
        "schema_version": "neurooracle.case2_closed_loop_ready.v3",
        "benchmark_id": config["benchmark_id"],
        "freeze_id": freeze_id,
        "benchmark_lock_sha256": sha256_file(lock_path),
        "status": "ready_for_single_seed_closed_loop_execution",
        "prepared_at_hkt": hkt_now(),
        "candidate_count": 168,
        "method_count": 7,
        "trial_count": 1,
        "expected_main_action_slots_per_method": 168,
        "api_calls_made": False,
        "experiment_model_completions_made": False,
        "provider_attestation_verified": True,
        "hybrid_neurodiscovery_import_verified": True,
        "hybrid_ai_scientist_import_verified": provider_migration_audit is not None,
        "sealed_response_cache_import_verified": provider_migration_audit is not None,
        "outcome_blind_coordinate_compiler_verified": (
            config.get("native_coordinate_compiler") is not None
        ),
        "formal_biological_results_modified": False,
        "external_validation_queue_created": False,
    }
    write_json(staging / "READY_TO_RUN.json", ready)
    os.replace(staging, output_root)
    return ready


def _same_pin(path: Path, pin: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(pin["bytes"])
        and sha256_file(path) == pin["sha256"]
    )


def verify_freeze(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    ready = load_json(output_root / "READY_TO_RUN.json")
    lock_path = output_root / "BENCHMARK_FREEZE.lock.json"
    lock = load_json(lock_path)
    _require(ready["freeze_id"] == lock["freeze_id"], "Freeze identity mismatch")
    _require(ready["benchmark_lock_sha256"] == sha256_file(lock_path), "Freeze lock changed")
    config = load_json(output_root / "PROTOCOL.json")
    validate_config(config)
    material = lock["lock_material"]
    for relative, pin in material["source_code_pins"].items():
        _require(_same_pin(REPO_ROOT / relative, pin), f"Source drift: {relative}")
    for name, pin in material["external_input_pins"].items():
        _require(_same_pin(Path(pin["path"]), pin), f"External input drift: {name}")
    for relative, pin in material["artifact_pins"].items():
        _require(_same_pin(output_root / relative, pin), f"Frozen artifact drift: {relative}")
    leakage = audit_generator_bundle(output_root / "generator_inputs")
    _require(leakage["status"] == "passed", f"Generator leakage: {leakage['findings']}")
    source_root = Path(config["paths"]["source_static_benchmark_root"])
    v1 = verify_v1_integrity_for_v2(source_root, fast_large_files=True)
    _require(v1["freeze_id"] == config["source_static_freeze_id"], "Source v1 freeze drift")
    repository_input_audit = verify_v1_repository_execution_inputs(
        source_root,
        baseline_root_override=Path(config["paths"]["baseline_repository_root"]),
    )
    return {
        "benchmark_id": config["benchmark_id"],
        "freeze_id": lock["freeze_id"],
        "status": "verified_ready_for_single_seed_execution",
        "source_code_pin_count": len(material["source_code_pins"]),
        "artifact_count": len(material["artifact_pins"]),
        "external_input_count": len(material["external_input_pins"]),
        "source_v1_freeze_verified": True,
        "source_v1_repository_execution_inputs_verified": (
            repository_input_audit["status"] == "verified"
        ),
        "api_calls_made_during_preparation": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        config = load_json(args.config.resolve())
        result = verify_freeze(Path(config["paths"]["output_root"]))
    else:
        result = prepare(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
