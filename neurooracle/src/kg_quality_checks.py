"""Read-only structural/serialization checks, not biomedical truth certification."""

from __future__ import annotations

import math
import re
from copy import deepcopy

from .metadata_field_audit import nonempty
from .schema import Claim


SCIENTIFIC_METADATA_FIELDS = ("subject_type", "object_type", "conditions", "population")
STANDARD_CUI = re.compile(r"C\d{7}\Z")


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def claim_record_checks(node_id: str, record: dict) -> list[dict]:
    """Check actual Claim decoding/encoding without changing the stored node."""
    md = record.get("metadata") or {}
    issues = []

    def add(code: str, **details):
        issues.append({"code": code, **details})

    if md.get("id") != node_id:
        add("claim_payload_id_differs_from_node_id", payload_id=md.get("id"))
    for key in ("subject_id", "object_id", "subject_name", "object_name", "predicate"):
        if not nonempty(md.get(key)):
            add("claim_required_field_empty", field=key)
    if md.get("subject_id") and md.get("subject_id") == md.get("object_id"):
        add("claim_self_endpoint")
    confidence = md.get("confidence")
    if not is_number(confidence) or not 0 <= confidence <= 1:
        add("claim_confidence_not_finite_unit_number", value=confidence)
    if not isinstance(md.get("negated", False), bool):
        add("claim_negation_not_boolean", value=md.get("negated"))
    evidence = md.get("evidence")
    if evidence is not None and not isinstance(evidence, dict):
        add("claim_evidence_non_object", value_type=type(evidence).__name__)
    if isinstance(evidence, dict):
        p = evidence.get("p_value")
        if is_number(p) and not 0 <= p <= 1:
            add("numeric_p_value_out_of_range", value=p)
        n = evidence.get("sample_size")
        if is_number(n) and (n <= 0 or int(n) != n):
            add("numeric_sample_size_non_positive_or_fractional", value=n)
    nested = md.get("metadata") or {}
    if not isinstance(nested, dict):
        add("nested_claim_metadata_not_object", value_type=type(nested).__name__)
        nested = {}
    for key in SCIENTIFIC_METADATA_FIELDS:
        if nonempty(md.get(key)) and nonempty(nested.get(key)) and md[key] != nested[key]:
            add("direct_nested_values_differ", field=key, direct=md[key], nested=nested[key])
    try:
        obj = Claim.from_dict(deepcopy(md))
    except Exception as error:
        add("claim_decoder_raises", exception=type(error).__name__, message=str(error)[:300])
        return issues
    for key in SCIENTIFIC_METADATA_FIELDS:
        if nonempty(md.get(key)) and not nonempty(obj.metadata.get(key)):
            add("direct_scientific_field_not_carried_into_claim_object", field=key)
    try:
        encoded = obj.to_dict()
    except Exception as error:
        add("claim_reencoder_raises", exception=type(error).__name__, message=str(error)[:300])
        return issues
    if isinstance(evidence, dict):
        lost = [key for key, value in evidence.items() if nonempty(value) and key not in encoded["evidence"]]
        if lost:
            add("nonempty_evidence_extensions_omitted_by_schema", fields=sorted(lost))
    return issues


def entity_identifier_checks(node_id: str, record: dict) -> list[dict]:
    identifiers = record.get("external_ids") or {}
    issues = []
    external = str(identifiers.get("UMLS_CUI") or "").removeprefix("CUI:")
    if node_id.startswith("CUI:"):
        suffix = node_id[4:]
        if not STANDARD_CUI.fullmatch(suffix):
            issues.append({"code": "nonstandard_cui_prefixed_id", "recorded_id": node_id})
        elif external and suffix != external:
            issues.append({"code": "cui_id_external_identifier_disagreement", "node_cui": suffix, "external_cui": external})
    return issues


def edge_claim_agreement(edge: dict, claim: dict | None) -> list[dict]:
    md = edge.get("metadata") or {}
    claim_id = md.get("claim_id")
    if not claim_id:
        return []
    if claim is None:
        return [{"code": "edge_references_missing_claim", "claim_id": claim_id}]
    issues = []
    for edge_key, claim_key in (("source_id", "subject_id"), ("target_id", "object_id"), ("relation_type", "predicate")):
        if edge.get(edge_key) != claim.get(claim_key):
            issues.append({"code": "edge_claim_payload_disagreement", "field": edge_key,
                           "edge_value": edge.get(edge_key), "claim_value": claim.get(claim_key)})
    if "negated" in md and md["negated"] != claim.get("negated", False):
        issues.append({"code": "edge_claim_negation_disagreement", "edge_value": md["negated"], "claim_value": claim.get("negated", False)})
    return issues
