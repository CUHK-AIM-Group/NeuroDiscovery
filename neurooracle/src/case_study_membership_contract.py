"""Content-addressed Case Study membership contract for newly extracted claims.

Version 4 seals the relaxed, component-based Case 2 membership epoch without
rewriting version-3 records.  The legacy validator remains available here so
already-sealed extraction campaigns can still be checked exactly.  The
contract is intentionally model-independent: once sealed, the same evidence
and policy reuse the stored decision instead of re-running a model.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from .case2_chain_validation import (
    CASE2_CHAIN_VALIDATION_FIELD,
    CASE2_ID,
    validate_case2_paper_chain_record,
)
from .case_study_membership_policy import (
    CaseStudyMembershipPolicy,
    LEGACY_POLICY,
    POLICY,
    validate_scope_decision,
    validate_scope_decision_for_policy,
)
from .case_study_scope import normalize_case_study_ids


AUDIT_NAME = "case_study_membership"
AUDIT_VERSION = "5"
AUDIT_CONTRACT_VERSION = "case_study_membership_contract.v4"
LEGACY_AUDIT_CONTRACT_VERSION = "case_study_membership_contract.v3"
CONTRACT_FIELD_NAMES = (
    "audit_contract_version",
    "rubric_sha256",
    "case_study_registry_sha256",
    "scope_context_sha256",
    "claim_evidence_sha256",
    "audit_key",
    "decision_sha256",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_text_sha256(source_text: str) -> str:
    return _sha256_text(str(source_text or ""))


def _registry_sha256(policy: CaseStudyMembershipPolicy) -> str:
    return _sha256_text(_canonical_json(list(policy.case_study_ids)))


CASE_STUDY_REGISTRY_SHA256 = _registry_sha256(POLICY)


def _clean_doi(value: object) -> str:
    text = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def _clean_title(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def strongest_paper_key(payload: Mapping[str, Any]) -> str:
    """Return a stable identity key using the same priority as KG deduplication."""

    paper = payload.get("source_paper") or {}
    if not isinstance(paper, Mapping):
        paper = {}
    for prefix, value in (
        ("pmid", paper.get("pmid")),
        ("doi", _clean_doi(paper.get("doi"))),
        ("pmcid", paper.get("pmcid")),
        ("arxiv", paper.get("arxiv_id")),
        ("openalex", paper.get("openalex_id")),
    ):
        normalized = str(value or "").strip().lower()
        if normalized:
            return f"{prefix}:{normalized}"
    title = _clean_title(paper.get("title"))
    year = str(paper.get("year") or paper.get("publication_year") or "")
    if title:
        return f"title_year:{title}|{year}"
    claim_id = str(payload.get("id") or "").strip()
    return f"claim_fallback:{claim_id}"


def claim_evidence_payload(
    payload: Mapping[str, Any],
    *,
    scope_context_sha256: str,
    include_legacy_case2_chain: bool = False,
) -> dict[str, Any]:
    """Select immutable evidence fields and exclude mutable routing fields."""

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    result = {
        "paper_key": strongest_paper_key(payload),
        "claim_id": str(payload.get("id") or ""),
        "subject_name": payload.get("subject_name") or payload.get("subject") or "",
        "predicate": payload.get("predicate") or "",
        "object_name": payload.get("object_name") or payload.get("object") or "",
        "negated": bool(payload.get("negated", False)),
        "raw_text": payload.get("raw_text") or "",
        "subject_type": payload.get("subject_type") or metadata.get("subject_type") or "",
        "object_type": payload.get("object_type") or metadata.get("object_type") or "",
        "conditions": payload.get("conditions") or metadata.get("conditions") or [],
        "population": payload.get("population") or metadata.get("population"),
        "evidence": payload.get("evidence") or {},
        "source_paper": payload.get("source_paper") or {},
        "scope_evidence_spans": metadata.get("scope_evidence_spans") or [],
        "extraction_profile": metadata.get("extraction_profile") or {},
        "scope_context_sha256": scope_context_sha256,
    }
    if include_legacy_case2_chain:
        result[CASE2_CHAIN_VALIDATION_FIELD] = metadata.get(
            CASE2_CHAIN_VALIDATION_FIELD
        )
    return result


def claim_evidence_sha256(
    payload: Mapping[str, Any],
    *,
    scope_context_sha256: str,
    include_legacy_case2_chain: bool = False,
) -> str:
    return _sha256_text(
        _canonical_json(
            claim_evidence_payload(
                payload,
                scope_context_sha256=scope_context_sha256,
                include_legacy_case2_chain=include_legacy_case2_chain,
            )
        )
    )


def decision_sha256(
    *,
    labels: Iterable[str],
    gates: Mapping[str, bool],
    confidence: float,
    decision_basis: str,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "claim_case_study_ids": list(labels),
                "gates": dict(gates),
                "confidence": float(confidence),
                "decision_basis": str(decision_basis),
            }
        )
    )


def contract_fields(
    payload: Mapping[str, Any],
    *,
    labels: Iterable[str],
    gates: Mapping[str, bool],
    confidence: float,
    decision_basis: str,
    scope_context_sha256: str,
    audit_contract_version: str = AUDIT_CONTRACT_VERSION,
    policy: CaseStudyMembershipPolicy = POLICY,
) -> dict[str, str]:
    evidence_hash = claim_evidence_sha256(
        payload,
        scope_context_sha256=scope_context_sha256,
        include_legacy_case2_chain=(
            audit_contract_version == LEGACY_AUDIT_CONTRACT_VERSION
        ),
    )
    context = {
        "audit_contract_version": audit_contract_version,
        "rubric_version": policy.version,
        "rubric_sha256": policy.rubric_sha256,
        "case_study_registry_sha256": _registry_sha256(policy),
        "scope_context_sha256": scope_context_sha256,
        "claim_evidence_sha256": evidence_hash,
    }
    audit_key = _sha256_text(_canonical_json(context))
    return {
        **context,
        "audit_key": audit_key,
        "decision_sha256": decision_sha256(
            labels=labels,
            gates=gates,
            confidence=confidence,
            decision_basis=decision_basis,
        ),
    }


def build_final_scope_reaudit(
    payload: Mapping[str, Any],
    *,
    labels: object,
    gates: object,
    confidence: float,
    decision_basis: str,
    scope_context_sha256: str,
    reviewer_id: str,
    reasoning_effort: str,
    reviewed_at: str,
    source_kind: str,
) -> dict[str, Any]:
    """Validate and seal a new claim's final Case Study assignment."""

    decision = validate_scope_decision(labels, gates)
    provided_labels = tuple(normalize_case_study_ids(labels, strict=True))
    if provided_labels != decision.labels:
        raise ValueError(
            "claim_case_study_ids must include deterministic Case 2 membership"
        )
    normalized_confidence = float(confidence)
    if not 0.0 <= normalized_confidence <= 1.0:
        raise ValueError("scope_confidence must be in 0..1")
    normalized_basis = str(decision_basis or "").strip()
    if len(normalized_basis) < 12:
        raise ValueError("scope_decision_basis is missing or too short")
    if not re.fullmatch(r"[0-9a-f]{64}", scope_context_sha256):
        raise ValueError("scope_context_sha256 must be a lowercase SHA-256")

    contract = contract_fields(
        payload,
        labels=decision.labels,
        gates=decision.gates,
        confidence=normalized_confidence,
        decision_basis=normalized_basis,
        scope_context_sha256=scope_context_sha256,
    )
    return {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        **contract,
        "review_stage": "combined_extraction_and_scope_audit",
        "reviewer_id": str(reviewer_id or "unknown"),
        "reasoning_effort": str(reasoning_effort or "unspecified"),
        "reviewed_at": str(reviewed_at),
        "source_kind": str(source_kind),
        "decision": "finalized",
        "review_status": "final_complete",
        "confidence": normalized_confidence,
        "decision_basis": normalized_basis,
        "gates": dict(decision.gates),
        "previous_claim_case_study_ids": [],
    }


def validate_final_scope_reaudit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate either a current v4 seal or an immutable legacy v3 seal."""

    audit = payload.get("scope_reaudit") or {}
    if not isinstance(audit, Mapping):
        raise ValueError("scope_reaudit must be an object")
    if audit.get("review_status") != "final_complete" or audit.get("decision") != "finalized":
        raise ValueError("claim scope audit is not final_complete")
    contract_version = str(audit.get("audit_contract_version") or "")
    if contract_version == AUDIT_CONTRACT_VERSION:
        policy = POLICY
        decision_validator = validate_scope_decision
        legacy_case2_chain = False
    elif contract_version == LEGACY_AUDIT_CONTRACT_VERSION:
        policy = LEGACY_POLICY
        decision_validator = lambda value, gate_value: validate_scope_decision_for_policy(
            value,
            gate_value,
            policy=LEGACY_POLICY,
            derive_case2=False,
        )
        legacy_case2_chain = True
    else:
        raise ValueError("claim scope audit contract is not a supported extraction contract")

    labels = payload.get("claim_case_study_ids") or []
    decision = decision_validator(labels, audit.get("gates"))
    if contract_version == AUDIT_CONTRACT_VERSION:
        provided_labels = tuple(normalize_case_study_ids(labels, strict=True))
        if provided_labels != decision.labels:
            raise ValueError(
                "claim_case_study_ids omit deterministic Case 2 membership"
            )
    confidence = float(audit.get("confidence"))
    decision_basis = str(audit.get("decision_basis") or "").strip()
    context_hash = str(audit.get("scope_context_sha256") or "")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    claim_index_raw = metadata.get("extraction_claim_index")
    try:
        claim_index = int(claim_index_raw) if claim_index_raw is not None else None
    except (TypeError, ValueError):
        claim_index = None
    if legacy_case2_chain:
        validate_case2_paper_chain_record(
            metadata.get(CASE2_CHAIN_VALIDATION_FIELD),
            source_context_sha256=context_hash,
            claim_index=claim_index,
            case2_labeled=CASE2_ID in decision.labels,
        )
    expected = contract_fields(
        payload,
        labels=decision.labels,
        gates=decision.gates,
        confidence=confidence,
        decision_basis=decision_basis,
        scope_context_sha256=context_hash,
        audit_contract_version=contract_version,
        policy=policy,
    )
    mismatched = [
        field
        for field, value in expected.items()
        if str(audit.get(field) or "") != value
    ]
    if mismatched:
        raise ValueError(
            "claim scope audit seal mismatch: " + ", ".join(mismatched)
        )
    return dict(audit)


__all__ = [
    "AUDIT_CONTRACT_VERSION",
    "AUDIT_NAME",
    "AUDIT_VERSION",
    "CASE_STUDY_REGISTRY_SHA256",
    "CONTRACT_FIELD_NAMES",
    "LEGACY_AUDIT_CONTRACT_VERSION",
    "build_final_scope_reaudit",
    "claim_evidence_payload",
    "contract_fields",
    "source_text_sha256",
    "strongest_paper_key",
    "validate_final_scope_reaudit",
]
