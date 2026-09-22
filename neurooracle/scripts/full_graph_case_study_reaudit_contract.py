"""Content-addressed immutability contract for Case Study re-audit decisions."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from neurooracle.src.case_study_membership_policy import RUBRIC_PATH

DEFAULT_RUBRIC = RUBRIC_PATH
AUDIT_NAME = "full_graph_case_study_membership_reaudit"
AUDIT_VERSION = "3"
AUDIT_CONTRACT_VERSION = "case_study_reaudit_contract.v1"
CONTRACT_FIELD_NAMES = (
    "audit_contract_version",
    "rubric_sha256",
    "case_study_registry_sha256",
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


@lru_cache(maxsize=8)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _registry_sha256(case_study_ids: tuple[str, ...]) -> str:
    return _sha256_text(_canonical_json(list(case_study_ids)))


def registry_sha256(case_study_ids: Iterable[str]) -> str:
    return _registry_sha256(tuple(case_study_ids))


def claim_evidence_payload(paper_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return only evidence-bearing fields; exclude mutable routing/audit fields."""
    return {
        "paper_key": str(paper_key),
        "claim_id": str(payload.get("id") or ""),
        "subject_name": payload.get("subject_name") or payload.get("subject") or "",
        "predicate": payload.get("predicate") or "",
        "object_name": payload.get("object_name") or payload.get("object") or "",
        "negated": bool(payload.get("negated", False)),
        "raw_text": payload.get("raw_text") or "",
        "subject_type": payload.get("subject_type") or "",
        "object_type": payload.get("object_type") or "",
        "conditions": payload.get("conditions") or [],
        "evidence": payload.get("evidence") or {},
        "source_paper": payload.get("source_paper") or {},
    }


def claim_evidence_sha256(paper_key: str, payload: dict[str, Any]) -> str:
    return _sha256_text(_canonical_json(claim_evidence_payload(paper_key, payload)))


def decision_sha256(labels: Iterable[str], gates: dict[str, Any]) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "claim_case_study_ids": list(labels),
                "gates": {str(key): bool(value) for key, value in sorted(gates.items())},
            }
        )
    )


def contract_context(
    *, rubric_version: str, case_study_ids: Iterable[str], rubric_path: Path = DEFAULT_RUBRIC
) -> dict[str, str]:
    return {
        "audit_contract_version": AUDIT_CONTRACT_VERSION,
        "rubric_version": str(rubric_version),
        "rubric_sha256": file_sha256(rubric_path),
        "case_study_registry_sha256": registry_sha256(case_study_ids),
    }


def claim_input_contract_fields(
    *,
    paper_key: str,
    payload: dict[str, Any],
    rubric_version: str,
    case_study_ids: Iterable[str],
    rubric_path: Path = DEFAULT_RUBRIC,
) -> dict[str, str]:
    context = contract_context(
        rubric_version=rubric_version,
        case_study_ids=case_study_ids,
        rubric_path=rubric_path,
    )
    evidence_hash = claim_evidence_sha256(paper_key, payload)
    audit_key = _sha256_text(
        _canonical_json(
            {
                **context,
                "claim_evidence_sha256": evidence_hash,
            }
        )
    )
    return {
        **context,
        "claim_evidence_sha256": evidence_hash,
        "audit_key": audit_key,
    }


def claim_contract_fields(
    *,
    paper_key: str,
    payload: dict[str, Any],
    labels: Iterable[str],
    gates: dict[str, Any],
    rubric_version: str,
    case_study_ids: Iterable[str],
    rubric_path: Path = DEFAULT_RUBRIC,
) -> dict[str, str]:
    return {
        **claim_input_contract_fields(
            paper_key=paper_key,
            payload=payload,
            rubric_version=rubric_version,
            case_study_ids=case_study_ids,
            rubric_path=rubric_path,
        ),
        "decision_sha256": decision_sha256(labels, gates),
    }


def embedded_contract_matches(
    audit: dict[str, Any], expected: dict[str, str]
) -> bool:
    return all(str(audit.get(key) or "") == value for key, value in expected.items())
