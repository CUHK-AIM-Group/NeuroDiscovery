"""Fail-closed information boundary for leakage-free hindcasting discovery.

The discovery process may consume evidence dated no later than the historical
cutoff and feedback produced by its own computational experiments.  Later
publication records and the labels derived from them belong to a separate
retrospective-evaluation process that runs only after discovery is sealed.

This module deliberately contains no import from the literature evaluator.
It validates manifests and feedback records, scans a discovery source bundle
for forbidden evaluator dependencies, and builds deterministic discovery
seals for the later evaluation handoff.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


DISCOVERY_INPUT_SCHEMA = "hindcasting-v4-discovery-input-manifest.v1"
COMPUTATIONAL_FEEDBACK_SCHEMA = "hindcasting-v4-computational-feedback.v1"
DISCOVERY_SEAL_SCHEMA = "hindcasting-v4-discovery-seal.v1"
EVALUATION_HANDOFF_SCHEMA = "hindcasting-v4-evaluation-handoff.v1"


class LeakageBoundaryError(ValueError):
    """Raised when discovery could observe retrospective evaluation data."""


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# These names are outputs of the post-cutoff publication evaluator, not
# computational measurements.  They must never occur in a discovery payload.
FORBIDDEN_EVALUATION_KEYS = frozenset(
    {
        "future_claims",
        "future_claim_records",
        "future_indexes",
        "future_publications",
        "future_labels",
        "future_support_labels",
        "primary_hit",
        "primary_year",
        "endpoint_hit",
        "endpoint_year",
        "intermediate_hit",
        "intermediate_year",
        "any_future_hit",
        "first_future_year",
        "support_year",
        "first_support_year",
        "terminal_primary_hit",
        "terminal_any_hit",
        "evaluation_label",
        "retrospective_label",
        "supporting_paper_ids",
        "supporting_pmids",
        "supporting_dois",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "post_cutoff",
    "later_publication",
    "future_support",
    "terminal_support",
    "retrospective_ground_truth",
)
FORBIDDEN_SOURCE_KINDS = frozenset(
    {
        "future_publication",
        "post_cutoff_publication",
        "later_publication",
        "literature_support",
        "publication_support_label",
        "retrospective_ground_truth",
    }
)
PUBLICATION_YEAR_KEYS = frozenset(
    {"publication_year", "paper_year", "claim_year", "evidence_year"}
)

ALLOWED_DISCOVERY_INPUT_ROLES = frozenset(
    {
        "historical_kg",
        "historical_claims",
        "historical_publications",
        "freeze_year_kge",
        "task_schema",
        "deterministic_generator",
        "computational_dataset",
        "computational_pipeline",
        "computational_analysis_plan",
        "method_config",
        "seed_schedule",
    }
)
HISTORICAL_INPUT_ROLES = frozenset(
    {"historical_kg", "historical_claims", "historical_publications"}
)
COMPUTATIONAL_INPUT_ROLES = frozenset(
    {
        "computational_dataset",
        "computational_pipeline",
        "computational_analysis_plan",
    }
)

FEEDBACK_STATUSES = frozenset(
    {"supported", "contradicted", "inconclusive", "execution_failed"}
)
EXECUTION_STATUSES = frozenset({"succeeded", "failed"})

FORBIDDEN_DISCOVERY_MODULES = frozenset(
    {
        "neurooracle.scripts.case_study_hindcasting_eval",
        "neurooracle.scripts.evaluate_hindcasting",
        "neurooracle.scripts.evaluate_formal_hindcasting",
    }
)
FORBIDDEN_DISCOVERY_SYMBOLS = frozenset(
    {
        "_future_indexes",
        "_score_hypothesis",
        "load_future_claim_records",
    }
)


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _walk(value: object, path: str = "$") -> Iterable[tuple[str, object, object]]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            child_path = f"{path}.{raw_key}"
            yield child_path, raw_key, child
            yield from _walk(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield from _walk(child, child_path)


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip()
    if not _SHA256_RE.fullmatch(digest):
        raise LeakageBoundaryError(f"{field} must be a 64-character SHA-256")
    return digest.casefold()


def _publication_year(value: object, *, path: str) -> int:
    if isinstance(value, bool):
        raise LeakageBoundaryError(f"{path} is not a valid publication year")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LeakageBoundaryError(f"{path} is not a valid publication year") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise LeakageBoundaryError(f"{path} is not a valid publication year")
    return int(number)


def assert_discovery_payload_blind(
    payload: object,
    *,
    freeze_year: int,
) -> None:
    """Reject post-cutoff literature and evaluator-derived fields recursively."""

    cutoff = int(freeze_year)
    for path, raw_key, value in _walk(payload):
        key = _normalized_key(raw_key)
        if key in FORBIDDEN_EVALUATION_KEYS or any(
            fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS
        ):
            raise LeakageBoundaryError(
                f"retrospective evaluation field is forbidden in discovery: {path}"
            )
        if key in PUBLICATION_YEAR_KEYS and value not in (None, ""):
            year = _publication_year(value, path=path)
            if year > cutoff:
                raise LeakageBoundaryError(
                    f"post-cutoff evidence reached discovery at {path}: "
                    f"{year} > {cutoff}"
                )
        if key in {"source_kind", "evidence_kind", "data_role"}:
            kind = _normalized_key(value)
            if kind in FORBIDDEN_SOURCE_KINDS:
                raise LeakageBoundaryError(
                    f"forbidden evaluation source reached discovery at {path}: {value!r}"
                )


def validate_discovery_input_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a fail-closed manifest for one freeze-year discovery process."""

    if manifest.get("schema_version") != DISCOVERY_INPUT_SCHEMA:
        raise LeakageBoundaryError("discovery input manifest has the wrong schema")
    try:
        freeze_year = int(manifest["freeze_year"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LeakageBoundaryError("discovery manifest needs an integer freeze_year") from exc

    if manifest.get("generator_mode") != "deterministic_frozen":
        raise LeakageBoundaryError(
            "v4 discovery requires the deterministic frozen generator"
        )
    if manifest.get("llm_api_enabled") is not False:
        raise LeakageBoundaryError(
            "LLM-backed generation is forbidden because model weights can contain "
            "post-cutoff literature"
        )
    if manifest.get("evaluation_data_mounted") is not False:
        raise LeakageBoundaryError(
            "retrospective evaluation data must not be mounted in discovery"
        )
    if manifest.get("evaluation_module_imported") is not False:
        raise LeakageBoundaryError(
            "the retrospective evaluator must not be imported in discovery"
        )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise LeakageBoundaryError("discovery manifest needs a non-empty inputs list")
    role_counts: dict[str, int] = {}
    for index, record in enumerate(inputs):
        if not isinstance(record, Mapping):
            raise LeakageBoundaryError(f"inputs[{index}] must be an object")
        role = _normalized_key(record.get("role"))
        if role not in ALLOWED_DISCOVERY_INPUT_ROLES:
            raise LeakageBoundaryError(
                f"inputs[{index}] has a forbidden or unknown role: {role!r}"
            )
        _require_sha256(record.get("sha256"), field=f"inputs[{index}].sha256")
        role_counts[role] = role_counts.get(role, 0) + 1
        if role in HISTORICAL_INPUT_ROLES:
            try:
                max_year = int(record["max_publication_year"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LeakageBoundaryError(
                    f"inputs[{index}] must bind max_publication_year"
                ) from exc
            if max_year > freeze_year:
                raise LeakageBoundaryError(
                    f"inputs[{index}] includes evidence after the cutoff"
                )
        if role in COMPUTATIONAL_INPUT_ROLES and record.get(
            "selected_without_retrospective_evaluation"
        ) is not True:
            raise LeakageBoundaryError(
                f"inputs[{index}] computational resource selection is not blinded"
            )

    required_roles = {"historical_kg", "task_schema", "deterministic_generator"}
    missing_roles = sorted(required_roles - set(role_counts))
    if missing_roles:
        raise LeakageBoundaryError(
            f"discovery manifest is missing required roles: {missing_roles}"
        )
    assert_discovery_payload_blind(manifest, freeze_year=freeze_year)
    return {
        "status": "passed",
        "freeze_year": freeze_year,
        "input_count": len(inputs),
        "role_counts": role_counts,
        "llm_api_enabled": False,
        "evaluation_data_mounted": False,
    }


def validate_computational_feedback(
    record: Mapping[str, Any],
    *,
    freeze_year: int,
    committed_candidate_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate one feedback record produced by an executed computation."""

    if record.get("schema_version") != COMPUTATIONAL_FEEDBACK_SCHEMA:
        raise LeakageBoundaryError("computational feedback has the wrong schema")
    if record.get("source_kind") != "computational_experiment":
        raise LeakageBoundaryError(
            "feedback source_kind must be computational_experiment"
        )
    for field in ("hypothesis_id", "experiment_id", "task_id", "dataset_id", "executor"):
        if not str(record.get(field) or "").strip():
            raise LeakageBoundaryError(f"computational feedback is missing {field}")
    for field in (
        "dataset_sha256",
        "analysis_plan_sha256",
        "pipeline_sha256",
        "selection_commit_sha256",
    ):
        _require_sha256(record.get(field), field=field)
    if record.get("outcome_observed_after_selection") is not True:
        raise LeakageBoundaryError(
            "experimental outcomes may be revealed only after selection commitment"
        )
    if record.get("publication_evaluation_fields_read") is not False:
        raise LeakageBoundaryError(
            "feedback does not attest that publication evaluation fields were unread"
        )
    execution_status = str(record.get("execution_status") or "")
    feedback_status = str(record.get("feedback_status") or "")
    if execution_status not in EXECUTION_STATUSES:
        raise LeakageBoundaryError(f"unknown execution_status: {execution_status!r}")
    if feedback_status not in FEEDBACK_STATUSES:
        raise LeakageBoundaryError(f"unknown feedback_status: {feedback_status!r}")
    if not isinstance(record.get("feedback_available"), bool):
        raise LeakageBoundaryError("feedback_available must be boolean")
    if execution_status == "failed" and feedback_status != "execution_failed":
        raise LeakageBoundaryError(
            "failed execution must remain execution_failed, not scientific evidence"
        )
    if execution_status == "succeeded" and feedback_status == "execution_failed":
        raise LeakageBoundaryError(
            "successful execution cannot be labeled execution_failed"
        )
    statistics = record.get("statistics")
    if execution_status == "succeeded" and not isinstance(statistics, Mapping):
        raise LeakageBoundaryError(
            "successful computational feedback needs a statistics object"
        )
    if committed_candidate_ids is not None:
        committed = {str(value) for value in committed_candidate_ids}
        if str(record["hypothesis_id"]) not in committed:
            raise LeakageBoundaryError(
                "feedback hypothesis was not present in the committed selection"
            )
    assert_discovery_payload_blind(record, freeze_year=int(freeze_year))
    return {
        "status": "passed",
        "hypothesis_id": str(record["hypothesis_id"]),
        "execution_status": execution_status,
        "feedback_status": feedback_status,
    }


def scan_discovery_source(path: Path) -> dict[str, Any]:
    """Reject imports or direct symbol use from the publication evaluator."""

    path = path.resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_DISCOVERY_MODULES:
                    findings.append(f"import {alias.name} at line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module in FORBIDDEN_DISCOVERY_MODULES:
                findings.append(f"from {module} at line {node.lineno}")
            for alias in node.names:
                if alias.name in FORBIDDEN_DISCOVERY_SYMBOLS:
                    findings.append(
                        f"symbol {alias.name} imported at line {node.lineno}"
                    )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_DISCOVERY_SYMBOLS:
            findings.append(f"symbol {node.id} used at line {node.lineno}")
    if findings:
        raise LeakageBoundaryError(
            f"discovery source imports retrospective evaluation code: {findings}"
        )
    return {"status": "passed", "path": str(path), "sha256": sha256_file(path)}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_discovery_seal(
    *,
    freeze_year: int,
    task_id: str,
    method: str,
    seed: int,
    ordered_hypothesis_ids: Sequence[str],
    discovery_input_manifest_sha256: str,
    discovery_source_bundle_sha256: str,
    feedback_chain_sha256: str,
) -> dict[str, Any]:
    """Build an immutable handoff record before any later literature is loaded."""

    if not ordered_hypothesis_ids:
        raise LeakageBoundaryError("cannot seal an empty discovery sequence")
    if len(set(ordered_hypothesis_ids)) != len(ordered_hypothesis_ids):
        raise LeakageBoundaryError("discovery sequence contains duplicate IDs")
    payload = {
        "schema_version": DISCOVERY_SEAL_SCHEMA,
        "status": "discovery_complete",
        "freeze_year": int(freeze_year),
        "task_id": str(task_id),
        "method": str(method),
        "seed": int(seed),
        "ordered_hypothesis_ids": [str(value) for value in ordered_hypothesis_ids],
        "discovery_input_manifest_sha256": _require_sha256(
            discovery_input_manifest_sha256,
            field="discovery_input_manifest_sha256",
        ),
        "discovery_source_bundle_sha256": _require_sha256(
            discovery_source_bundle_sha256,
            field="discovery_source_bundle_sha256",
        ),
        "feedback_chain_sha256": _require_sha256(
            feedback_chain_sha256,
            field="feedback_chain_sha256",
        ),
        "evaluation_data_loaded": False,
    }
    payload["ordered_hypothesis_ids_sha256"] = canonical_sha256(
        payload["ordered_hypothesis_ids"]
    )
    payload["discovery_seal_sha256"] = canonical_sha256(payload)
    return payload


def validate_evaluation_handoff(
    discovery_seal: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that retrospective evaluation opened only after discovery froze."""

    if discovery_seal.get("schema_version") != DISCOVERY_SEAL_SCHEMA:
        raise LeakageBoundaryError("discovery seal has the wrong schema")
    if discovery_seal.get("status") != "discovery_complete":
        raise LeakageBoundaryError("discovery was not complete before evaluation")
    stored = str(discovery_seal.get("discovery_seal_sha256") or "")
    unsigned = dict(discovery_seal)
    unsigned.pop("discovery_seal_sha256", None)
    if stored != canonical_sha256(unsigned):
        raise LeakageBoundaryError("discovery seal hash verification failed")
    if handoff.get("schema_version") != EVALUATION_HANDOFF_SCHEMA:
        raise LeakageBoundaryError("evaluation handoff has the wrong schema")
    if handoff.get("discovery_seal_sha256") != stored:
        raise LeakageBoundaryError("evaluation is not bound to the discovery seal")
    if handoff.get("evaluation_loaded_after_discovery_seal") is not True:
        raise LeakageBoundaryError(
            "evaluation data was not demonstrably loaded after discovery sealing"
        )
    return {"status": "passed", "discovery_seal_sha256": stored}


__all__ = [
    "COMPUTATIONAL_FEEDBACK_SCHEMA",
    "DISCOVERY_INPUT_SCHEMA",
    "DISCOVERY_SEAL_SCHEMA",
    "EVALUATION_HANDOFF_SCHEMA",
    "LeakageBoundaryError",
    "assert_discovery_payload_blind",
    "build_discovery_seal",
    "canonical_sha256",
    "scan_discovery_source",
    "sha256_file",
    "validate_computational_feedback",
    "validate_discovery_input_manifest",
    "validate_evaluation_handoff",
]
