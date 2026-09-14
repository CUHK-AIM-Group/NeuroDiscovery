"""Canonical multi-label case-study membership helpers.

Case-study membership is independent of validation protocol.  Hindcasting is
therefore never interpreted as a paper or claim label.  The helpers retain a
read-only migration path for the legacy ``paper_scope``/``case3_tasks`` schema,
while canonical writers emit only ``paper_case_study_ids`` and
``claim_case_study_ids``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .case_studies import list_case_study_names


GENERAL_CORPUS_SCOPE = "general"
CASE_STUDY_IDS = list_case_study_names()
CASE_STUDY_ID_SET = frozenset(CASE_STUDY_IDS)
TASK_CASE_STUDY_IDS = tuple(
    case_study_id
    for case_study_id in CASE_STUDY_IDS
    if case_study_id not in {"case1_transdiagnostic", "case2_pathway_mediation"}
)

_CASE1_ALIASES = frozenset(
    {
        "case1",
        "cs1",
        "case_1",
        "case-1",
        "case study 1",
        "case1_transdiagnostic",
        "transdiagnostic_clustering",
    }
)
_CASE2_ALIASES = frozenset(
    {
        "case2",
        "cs2",
        "case_2",
        "case-2",
        "case study 2",
        "case2_pathway_mediation",
        "pathway_polygenic_mediation",
    }
)
_NON_CASE_STUDY_LABELS = frozenset(
    {
        "",
        GENERAL_CORPUS_SCOPE,
        "case3",
        "cs3",
        "case_3",
        "case-3",
        "case study 3",
        "case3_hindcasting",
        "hindcasting",
        "temporal_claim_hindcasting",
    }
)


def _values(value: object) -> list[object]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return list(value)
    return [value]


def canonical_case_study_id(value: object) -> str:
    """Normalize one label, returning ``""`` for non-case-study labels."""

    text = str(value or "").strip().lower()
    if text in _CASE1_ALIASES:
        return "case1_transdiagnostic"
    if text in _CASE2_ALIASES:
        return "case2_pathway_mediation"
    if text in CASE_STUDY_ID_SET:
        return text
    if text in _NON_CASE_STUDY_LABELS:
        return ""
    return ""


def normalize_case_study_ids(
    value: object,
    *,
    strict: bool = False,
) -> list[str]:
    """Return unique formal IDs in registry order.

    ``strict=True`` raises on unknown non-empty values, but still accepts legacy
    Case 1/2 aliases and intentionally ignores the retired Case 3 umbrella and
    hindcasting protocol labels.
    """

    selected: set[str] = set()
    unknown: list[str] = []
    for raw in _values(value):
        text = str(raw or "").strip().lower()
        canonical = canonical_case_study_id(text)
        if canonical:
            selected.add(canonical)
        elif text not in _NON_CASE_STUDY_LABELS:
            unknown.append(text)
    if strict and unknown:
        raise ValueError(f"unknown case-study IDs: {', '.join(sorted(set(unknown)))}")
    return [case_study_id for case_study_id in CASE_STUDY_IDS if case_study_id in selected]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _legacy_task_values(claim_data: Mapping[str, Any]) -> list[object]:
    metadata = _mapping(claim_data.get("metadata"))
    for holder in (claim_data, metadata):
        for key in ("case3_tasks", "case3_subtasks"):
            if key in holder:
                return _values(holder.get(key))
    return []


def legacy_case_study_ids_from_claim_dict(
    claim_data: Mapping[str, Any],
) -> list[str]:
    """Map the legacy Case 1/2/3 routing fields to formal IDs."""

    metadata = _mapping(claim_data.get("metadata"))
    selected: list[object] = []

    for holder in (claim_data, metadata):
        selected.extend(_values(holder.get("case_study_ids")))
        selected.extend(_values(holder.get("case_study")))
        selected.extend(_values(holder.get("case_id")))
        scopes = _values(holder.get("paper_scope"))
        if any(str(scope or "").strip().lower() in _CASE1_ALIASES for scope in scopes):
            selected.append("case1_transdiagnostic")
        if any(str(scope or "").strip().lower() in _CASE2_ALIASES for scope in scopes):
            selected.append("case2_pathway_mediation")
        if bool(holder.get("case1_eligible", False)):
            selected.append("case1_transdiagnostic")
        if bool(holder.get("case2_eligible", False)):
            selected.append("case2_pathway_mediation")

    selected.extend(_legacy_task_values(claim_data))
    return normalize_case_study_ids(selected)


def claim_case_study_ids_from_dict(
    claim_data: Mapping[str, Any],
) -> list[str]:
    """Read canonical claim membership, falling back to the legacy schema."""

    metadata = _mapping(claim_data.get("metadata"))
    for holder in (claim_data, metadata):
        if "claim_case_study_ids" in holder:
            return normalize_case_study_ids(holder.get("claim_case_study_ids"))
    return legacy_case_study_ids_from_claim_dict(claim_data)


def paper_case_study_ids_from_dict(
    claim_data: Mapping[str, Any],
) -> list[str]:
    """Read paper membership, falling back to claim/legacy membership."""

    metadata = _mapping(claim_data.get("metadata"))
    for holder in (claim_data, metadata):
        if "paper_case_study_ids" in holder:
            return normalize_case_study_ids(holder.get("paper_case_study_ids"))
    return claim_case_study_ids_from_dict(claim_data)


def remove_legacy_scope_fields(target: dict[str, Any]) -> None:
    """Remove legacy routing keys from one serialized claim dictionary."""

    for key in (
        "paper_scope",
        "case3_tasks",
        "case3_subtasks",
        "case1_eligible",
        "case2_eligible",
        "case_study",
        "case_id",
        "case_study_ids",
    ):
        target.pop(key, None)


def apply_case_study_membership(
    target: dict[str, Any],
    *,
    paper_case_study_ids: object,
    claim_case_study_ids: object,
    remove_legacy: bool = True,
) -> None:
    """Write canonical membership to a serialized claim or metadata object."""

    target["paper_case_study_ids"] = normalize_case_study_ids(paper_case_study_ids)
    target["claim_case_study_ids"] = normalize_case_study_ids(claim_case_study_ids)
    if remove_legacy:
        remove_legacy_scope_fields(target)


__all__ = [
    "CASE_STUDY_IDS",
    "CASE_STUDY_ID_SET",
    "GENERAL_CORPUS_SCOPE",
    "TASK_CASE_STUDY_IDS",
    "apply_case_study_membership",
    "canonical_case_study_id",
    "claim_case_study_ids_from_dict",
    "legacy_case_study_ids_from_claim_dict",
    "normalize_case_study_ids",
    "paper_case_study_ids_from_dict",
    "remove_legacy_scope_fields",
]
