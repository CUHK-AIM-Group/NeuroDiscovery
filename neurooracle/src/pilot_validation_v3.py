"""Offline result-contract checks. Passing is not scientific acceptance.

Legacy diagnostics detect bounded risks, not general semantic entailment.
The v3 contract requires source-local bindings and keeps unknowns explicit.
No old candidate is rewritten or silently promoted to the new schema.
"""
from __future__ import annotations

import json
import re

from neurooracle.src import pilot_validation_v2 as previous


SCHEMA = "source_result_v3"
FIELDS = ("population", "exposure", "outcome", "comparator", "timepoint", "disease_stage")
KINDS = {"between_groups", "within_group_change", "association", "interaction", "other"}
SIGNIFICANCE = {"reported_significant", "reported_not_significant", "not_reported", "derived", "unresolved"}

# Diagnostic risks describe an encoding a reviewer must look at. They are
# deliberately NOT contract violations: flagging a risk is not the same as
# proving an error, and treating them as blocking made 11 conservative hints
# behave like 11 failures. Kept in one place so the split cannot drift.
ADVISORY_REASONS = frozenset({
    "timepoint_used_as_disease_stage",
    "groupwise_significance_requires_contrast_review",
    "null_report_is_not_explicit_test_significance",
})


def literal_binding(binding, quotes, text):
    if not isinstance(text, str) or not text:
        return False
    if not isinstance(binding, dict):
        return False
    index, start, end = (binding.get(key) for key in ("quote_index", "start", "end"))
    if any(type(value) is not int for value in (index, start, end)):
        return False
    if not 0 <= index < len(quotes) or not isinstance(quotes[index], str):
        return False
    quote = quotes[index]
    return (0 <= start < end <= len(quote) and quote[start:end] == text
            and binding.get("text") == text)


def legacy_risks(observation):
    """Flag specific unsafe encodings without inventing corrected science."""
    proposition = observation.get("proposition") if isinstance(observation.get("proposition"), dict) else {}
    conditions = observation.get("conditions") if isinstance(observation.get("conditions"), dict) else {}
    qualifiers = proposition.get("qualifiers") if isinstance(proposition.get("qualifiers"), dict) else {}
    result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
    quotes = observation.get("quotes") if isinstance(observation.get("quotes"), list) else []
    source = " ".join(previous.normalize_source(quote) for quote in quotes if isinstance(quote, str)).lower()
    reasons = []
    for stage in (conditions.get("stage"), qualifiers.get("disease_stage")):
        if isinstance(stage, str) and re.fullmatch(r"\s*(?:baseline|follow[- ]?up|cross[- ]sectional|post[- ]treatment)\s*", stage, re.I):
            reasons.append("timepoint_used_as_disease_stage")
    split_significance = bool(re.search(r"\bbut\s+not\b.{0,100}\bsignificant\b|\bsignificant\b.{0,100}\bbut\s+not\b", source))
    if (split_significance and proposition.get("relation") == "group_difference"
            and result.get("significance") == "reported_significant"):
        reasons.append("groupwise_significance_requires_contrast_review")
    if result.get("significance") == "not_significant":
        explicit_null = any(marker in source for marker in previous.NEGATED_SIGNIFICANCE)
        numeric_null = any(operator in {">", ">=", "≥", "="} and value >= .05 for operator, value in previous.p_values(source))
        if not explicit_null and not numeric_null:
            reasons.append("null_report_is_not_explicit_test_significance")
    return sorted(set(reasons))


def observation_errors(observation, abstract):
    """Validate one v3 result; source strings are literal, not canonical labels."""
    errors = []
    if not isinstance(observation, dict):
        return ["observation_not_object"]
    quotes = observation.get("quotes")
    if not isinstance(quotes, list) or not quotes or any(not isinstance(quote, str) or not quote or abstract.count(quote) != 1 for quote in quotes):
        return ["quotes_not_unique_source_literals"]
    unit = observation.get("structured_result")
    if not isinstance(unit, dict) or unit.get("schema") != SCHEMA:
        return ["structured_result_v3_missing"]
    if unit.get("kind") not in KINDS:
        errors.append("invalid_comparison_kind")
    for field in FIELDS:
        entry = unit.get(field)
        if not isinstance(entry, dict):
            errors.append(field + "_slot_missing")
            continue
        value, bindings = entry.get("value"), entry.get("bindings")
        if value is None:
            if bindings != [] or not isinstance(entry.get("unknown_reason"), str) or not entry["unknown_reason"].strip():
                errors.append(field + "_unknown_needs_reason")
        elif not isinstance(value, str) or not value.strip() or not isinstance(bindings, list) or not bindings:
            errors.append(field + "_must_be_one_bound_source_value")
        elif not all(literal_binding(binding, quotes, value) for binding in bindings):
            errors.append(field + "_binding_mismatch")
    stage = unit.get("disease_stage") or {}
    if isinstance(stage, dict) and isinstance(stage.get("value"), str) and re.fullmatch(r"baseline|follow[- ]?up|cross[- ]sectional|post[- ]treatment", stage["value"], re.I):
        errors.append("timepoint_used_as_disease_stage")
    contrast = unit.get("contrast")
    if not isinstance(contrast, dict) or not literal_binding(contrast, quotes, contrast.get("text")):
        errors.append("contrast_binding_missing_or_invalid")
        contrast = {}
    comparator = unit.get("comparator")
    if unit.get("kind") == "association" and isinstance(comparator, dict) and comparator.get("value") is not None:
        if comparator["value"] not in contrast.get("text", ""):
            errors.append("association_comparator_not_in_bound_contrast")
    atomicity = unit.get("atomicity")
    if not isinstance(atomicity, dict) or atomicity.get("status") not in {"single", "unresolved"}:
        errors.append("atomicity_missing")
    elif atomicity["status"] != "single":
        errors.append("atomicity_unresolved_requires_review")
    for field in ("population", "exposure", "outcome"):
        entry = unit.get(field) or {}
        value = entry.get("value") if isinstance(entry, dict) else None
        if isinstance(value, str) and re.search(r"\b(?:and|or)\b|;", value, re.I):
            errors.append(field + "_possibly_composite_requires_review")
    test = unit.get("test")
    if not isinstance(test, dict) or test.get("status") not in SIGNIFICANCE:
        errors.append("test_status_missing")
    else:
        status = test["status"]
        if status in {"reported_significant", "reported_not_significant"}:
            if not isinstance(test.get("contrast"), dict) or test["contrast"] != contrast:
                errors.append("significance_contrast_binding_mismatch")
            evidence = test.get("evidence")
            if not isinstance(evidence, dict) or not literal_binding(evidence, quotes, evidence.get("text")):
                errors.append("test_evidence_binding_invalid")
            else:
                text = evidence["text"].lower()
                if (evidence.get("quote_index") != contrast.get("quote_index")
                        or not (contrast.get("start", -1) <= evidence["start"] < evidence["end"] <= contrast.get("end", -1))):
                    errors.append("test_evidence_outside_comparison_clause")
                if status == "reported_significant" and not previous.significance_supported_by_text(text):
                    errors.append("test_significance_not_explicit")
                if status == "reported_not_significant" and not any(marker in text for marker in previous.NEGATED_SIGNIFICANCE):
                    if not any(operator in {">", ">=", "≥", "="} and value >= .05 for operator, value in previous.p_values(text)):
                        errors.append("test_nonsignificance_not_explicit")
        if status in {"derived", "unresolved"}:
            errors.append(status + "_test_requires_review")
        expected = {"reported_significant": "reported_significant", "reported_not_significant": "not_significant", "not_reported": "not_reported"}
        result = observation.get("result") if isinstance(observation.get("result"), dict) else {}
        if status in expected and result.get("significance") != expected[status]:
            errors.append("legacy_and_structured_test_disagree")
    if unit.get("kind") == "association":
        exposure = unit.get("exposure") or {}
        if not isinstance(exposure, dict) or exposure.get("value") is None:
            errors.append("association_exposure_unresolved")
    if unit.get("kind") in {"between_groups", "within_group_change"}:
        comparator = unit.get("comparator")
        if not isinstance(comparator, dict) or comparator.get("value") is None:
            errors.append("comparison_comparator_unresolved")
    for field in ("population", "outcome"):
        entry = unit.get(field)
        if not isinstance(entry, dict) or entry.get("value") is None:
            errors.append(field + "_unresolved_requires_review")
    return sorted(set(errors))


def candidate_risks(candidate):
    """Non-blocking diagnostic risks for a candidate, keyed by observation."""
    risks = []
    for index, observation in enumerate((candidate or {}).get("observations") or []):
        if not isinstance(observation, dict):
            continue
        for reason in legacy_risks(observation):
            risks.append({"observation_index": index, "reason": reason})
    return risks


def validate_candidate(candidate, abstract):
    """Return HOLD reasons. Never change candidate, publish, or call a provider."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("observations"), list):
        return [{"reason": "observations_not_array"}]
    errors, identities = [], {}
    if not candidate["observations"]:
        return [{"reason": "no_observations_requires_review"}]
    for index, observation in enumerate(candidate["observations"]):
        if not isinstance(observation, dict):
            errors.append({"observation_index": index, "reason": "observation_not_object"})
            continue
        if observation.get("role") != "primary_result":
            continue
        for reason in observation_errors(observation, abstract):
            errors.append({"observation_index": index, "reason": reason})
        unit = observation.get("structured_result")
        if isinstance(unit, dict):
            identity = json.dumps({key: unit.get(key) for key in ("kind", *FIELDS, "contrast")}, sort_keys=True)
            if identity in identities:
                errors.append({"observation_index": index, "reason": "repeated_result_identity_requires_review", "other_index": identities[identity]})
            else:
                identities[identity] = index
    return errors
