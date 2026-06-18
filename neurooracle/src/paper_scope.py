"""Utilities for assigning paper-level graph scope labels to claim evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

VALID_PAPER_SCOPES = ("general", "case1", "case2", "case3")


def normalize_paper_scope(scope: object) -> list[str]:
    """Return canonical paper scopes, preserving known labels only."""
    if scope is None or scope == "":
        return []
    if isinstance(scope, str):
        raw_values: Iterable[object] = [scope]
    elif isinstance(scope, Iterable) and not isinstance(scope, Mapping):
        raw_values = scope
    else:
        raw_values = [scope]

    out: list[str] = []
    for value in raw_values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if text in {"cs1", "case_1", "case-1", "case study 1", "case1_transdiagnostic"}:
            text = "case1"
        elif text in {"cs2", "case_2", "case-2", "case study 2"}:
            text = "case2"
        elif text in {"cs3", "case_3", "case-3", "case study 3", "hindcasting"}:
            text = "case3"
        elif text in {"general_neuromed", "manual_general", "base", "full_v2_base"}:
            text = "general"
        if text in VALID_PAPER_SCOPES and text not in out:
            out.append(text)
    return out


def infer_paper_scope_from_claim_dict(
    claim_data: Mapping[str, object],
    *,
    default: Iterable[str] = ("general",),
) -> list[str]:
    """Infer canonical paper scopes from a serialized claim dictionary.

    The explicit ``paper_scope`` field wins. Otherwise, derive scope from the
    curation metadata used by manual and case-targeted ingestion.
    """
    metadata_obj = claim_data.get("metadata")
    metadata = metadata_obj if isinstance(metadata_obj, Mapping) else {}

    explicit = normalize_paper_scope(claim_data.get("paper_scope"))
    explicit += [
        scope
        for scope in normalize_paper_scope(metadata.get("paper_scope"))
        if scope not in explicit
    ]
    if explicit:
        return explicit

    pieces = [
        claim_data.get("id"),
        metadata.get("curation_scope"),
        metadata.get("kg_injection_source"),
        metadata.get("case_study"),
        metadata.get("case_id"),
        claim_data.get("curation_scope"),
        claim_data.get("kg_injection_source"),
        claim_data.get("case_study"),
        claim_data.get("case_id"),
    ]
    text = " ".join(str(p or "") for p in pieces).lower()

    scopes: list[str] = []
    if (
        "case1" in text
        or "cs1" in text
        or "case_1" in text
        or "case-1" in text
        or "transdiagnostic" in text
    ):
        scopes.append("case1")
    if "case2" in text or "cs2" in text or "case_2" in text or "case-2" in text:
        scopes.append("case2")
    if (
        "case3" in text
        or "cs3" in text
        or "case_3" in text
        or "case-3" in text
        or "hindcast" in text
    ):
        scopes.append("case3")
    if (
        "general_neuromed" in text
        or "manual_general" in text
        or "genmed" in text
        or "strict_neuroscience" in text
    ):
        scopes.append("general")

    if scopes:
        return normalize_paper_scope(scopes)
    return normalize_paper_scope(default)
