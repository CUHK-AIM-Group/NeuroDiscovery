"""Lossless, conservative normalization of legacy claim payload layouts."""

from __future__ import annotations

from copy import deepcopy

SCIENTIFIC_FIELDS = ("subject_type", "object_type", "conditions", "population")


def has_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(has_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_value(item) for item in value)
    return True  # zero and False are real values, not missing data


def compatible_metadata(payload):
    """Fill missing nested scientific values; never resolve conflicting values."""
    metadata = deepcopy(payload.get("metadata") or {})
    if not isinstance(metadata, dict):
        raise TypeError("claim metadata must be an object")
    for key in SCIENTIFIC_FIELDS:
        if has_value(payload.get(key)) and not has_value(metadata.get(key)):
            metadata[key] = deepcopy(payload[key])
    return metadata


def normalize_claim_payload(payload):
    """Minimal candidate edit: preserve every existing field and both layouts."""
    from .schema import Evidence

    result = deepcopy(payload)
    if "evidence" in result and not isinstance(result["evidence"], dict):
        result["evidence"] = Evidence.from_dict(result["evidence"]).to_dict()
    metadata = compatible_metadata(result)
    if metadata != (result.get("metadata") or {}):
        result["metadata"] = metadata
    return result
