"""Opt-in current-KG storage compaction; no scientific-value imputation.

Only exact duplicate aliases, explicitly empty optional slots, completed-import
booleans and one declared membership-version constant are removed. Conflicts,
unique values, current audit seals and paper identity fallbacks remain intact.
The default legacy serialization contract is unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import json


LAYOUT_KEY = "metadata_layout"
LAYOUT_VERSION = "kg.metadata_compact.v1"
MEMBERSHIP_KEY = "case_study_membership_schema_version"
MEMBERSHIP_VERSION = "case_study_membership.v2"
LAYOUT = {
    "version": LAYOUT_VERSION,
    "membership_default": {MEMBERSHIP_KEY: MEMBERSHIP_VERSION},
    "membership_default_applies_to": "records_with_canonical_membership_fields",
    "claim_identity": "metadata.id",
    "claim_membership": "outer_claim_payload",
    "import_state": "completed_graph_omits_kg_injected_and_staged_only_booleans",
}
OPTIONAL_EMPTY = ("atom_types", "population", "conditions", "subject_type", "object_type", "case_study_scope")
SCIENTIFIC = ("subject_type", "object_type", "conditions", "population")
MEMBERSHIP_FIELDS = ("paper_case_study_ids", "claim_case_study_ids")
PAPER_FIELDS = ("pmid", "doi", "title", "year", "authors", "journal")


def typed_equal(left, right):
    """Do not equate False, 0 and 0.0, or reorder evidence/label lists."""
    if type(left) is not type(right):
        return False
    if isinstance(left, (dict, list)):
        options = dict(ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.dumps(left, **options) == json.dumps(right, **options)
    return left == right


def empty_slot(value):
    # Do not interpret whitespace, placeholder strings, False, zero or partially
    # populated objects as missing scientific data.
    return value is None or (type(value) in (str, list, dict) and len(value) == 0)


def compact_layout_enabled(metadata):
    layout = metadata.get(LAYOUT_KEY) if isinstance(metadata, Mapping) else None
    if layout is None:
        return False
    if not isinstance(layout, Mapping) or layout != LAYOUT:
        raise ValueError("unsupported KG metadata layout; refusing silent conversion")
    return True


def membership_schema_version(record, graph_metadata):
    """Resolve the declared default only for records actually carrying labels."""
    md = record.get("metadata") or {}
    holders = [md]
    if str(record.get("id") or "").startswith("CLM:"):
        holders.append(md.get("metadata") or {})
    for holder in holders:
        if MEMBERSHIP_KEY in holder:
            return holder[MEMBERSHIP_KEY]
    if compact_layout_enabled(graph_metadata) and any(
        any(field in holder for field in MEMBERSHIP_FIELDS) for holder in holders
    ):
        return MEMBERSHIP_VERSION
    return None


def compact_record(kind, record, changes=None):
    """Copy-on-write compaction. ``changes`` stores aggregate counts, not preimages."""
    if kind not in {"node", "edge"}:
        raise ValueError("expected node or edge")
    md = record.get("metadata")
    if md is None:
        return record
    if not isinstance(md, dict):
        raise TypeError("record metadata must be an object")
    out = dict(record)
    out["metadata"] = md = dict(md)
    claim = kind == "node" and str(record.get("id") or "").startswith("CLM:")
    scope = "node/claim" if claim else ("edge/all" if kind == "edge" else "node/nonclaim")

    def drop(holder, key, path, reason):
        del holder[key]
        if changes is not None:
            name = f"{reason}:{scope}.{path}.{key}"
            changes[name] = changes.get(name, 0) + 1

    def version(holder, path):
        if (holder.get(MEMBERSHIP_KEY) == MEMBERSHIP_VERSION
                and any(key in holder for key in MEMBERSHIP_FIELDS)):
            drop(holder, MEMBERSHIP_KEY, path, "shared_version")

    def equal_alias(holder, key, canonical, canonical_key, path):
        if key in holder and canonical_key in canonical and typed_equal(holder[key], canonical[canonical_key]):
            drop(holder, key, path, "exact_duplicate")

    version(md, "metadata")
    if not claim:
        return out
    inner = md.get("metadata")
    if inner is not None and not isinstance(inner, dict):
        raise TypeError("nested claim metadata must be an object")
    if inner is not None:
        md["metadata"] = inner = dict(inner)
        version(inner, "metadata.metadata")
        for key in ("subject_id", "object_id", *MEMBERSHIP_FIELDS):
            equal_alias(inner, key, md, key, "metadata.metadata")
    equal_alias(md, "claim_id", md, "id", "metadata")
    for alias, canonical in (("subject", "subject_name"), ("object", "object_name"),
                             ("raw_sentence", "raw_text"), ("evidence_text", "raw_text")):
        equal_alias(md, alias, md, canonical, "metadata")
    paper = md.get("source_paper")
    if isinstance(paper, dict):
        for key in PAPER_FIELDS:
            equal_alias(md, key, paper, key, "metadata")
    for holder, path in ((md, "metadata"), (inner, "metadata.metadata")):
        if holder is None:
            continue
        for key in ("kg_injected", "staged_only"):
            if key in holder and type(holder[key]) is bool:
                drop(holder, key, path, "retired_import_state")
        fields = OPTIONAL_EMPTY if holder is md else SCIENTIFIC
        for key in fields:
            # The audit contract distinguishes nested population=[]/""/{}
            # from a missing population (None). Do not invalidate that seal.
            if holder is inner and key == "population" and holder.get(key) is not None:
                continue
            if key in holder and empty_slot(holder[key]):
                drop(holder, key, path, "empty_optional_slot")
    from .kg_bulk_cleanup import simplify_record
    return simplify_record(kind, out, changes)
