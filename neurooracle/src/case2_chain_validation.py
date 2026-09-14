"""Deterministic paper-level validation for Case Study 2 extraction.

The frozen Case Study rubric requires one paper to support the complete
genetic/pathway -> brain imaging/physiology -> longitudinal clinical/cognitive
outcome mediation chain.  A claim-level boolean is therefore insufficient.
This module validates a grounded paper-level evidence object before any
Case-2-labelled claim can receive a final scope seal.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

CASE2_ID = "case2_pathway_mediation"
CASE2_GATE = "case2_full_chain_verified"
CASE2_CHAIN_EVIDENCE_FIELD = "case2_paper_chain_evidence"
CASE2_CHAIN_VALIDATION_FIELD = "case2_paper_chain_validation"
CASE2_CHAIN_VALIDATION_SCHEMA = "case2_paper_chain_validation.v1"
CASE2_CHAIN_VALIDATOR_VERSION = "2026-08-10.case2-paper-chain.v1"

COMPONENT_KEYS = (
    "genetic_or_pathway",
    "brain_imaging_or_physiology",
    "longitudinal_clinical_or_cognitive_outcome",
    "mediation_or_causal_chain",
)

_GENETIC_RE = re.compile(
    r"\b(?:gene|genes|genetic|genomic|genotype|polygenic|prs|variant|allele|"
    r"mutation|snp|gwas|methylat\w*|transcript\w*|molecular pathway|"
    r"risk pathway)\b",
    re.IGNORECASE,
)
_NEURAL_RE = re.compile(
    r"\b(?:brain|neural|neuronal|cortical|cortex|hippocamp\w*|amygdal\w*|"
    r"white matter|gray matter|grey matter|connectiv\w*|network|volume|"
    r"thickness|surface area|diffusion|fractional anisotropy|fmri|mri|pet|"
    r"eeg|meg|bold|suvr|perfusion|metaboli\w*|physiolog\w*)\b",
    re.IGNORECASE,
)
_LONGITUDINAL_RE = re.compile(
    r"\b(?:longitudinal|prospective|follow[ -]?up|later|subsequent|"
    r"incident|incidence|conversion|converted|progression|progressed|decline|"
    r"declined|trajectory|trajectories|over\s+(?:\w+\s+){0,2}(?:year|month|"
    r"week)s?|after\s+(?:\w+\s+){0,2}(?:year|month|week)s?)\b",
    re.IGNORECASE,
)
_OUTCOME_RE = re.compile(
    r"\b(?:clinical|cognitive|cognition|memory|symptom|severity|diagnos\w*|"
    r"dementia|impairment|outcome|function|disability|mortality|survival|"
    r"response|remission|relapse)\b",
    re.IGNORECASE,
)
_MEDIATION_RE = re.compile(
    r"(?:\bmediation(?: analysis| model| effect)?\b|"
    r"(?<!-)\bmediat(?:es|ed|ing)\b[^.;:]{0,100}\b(?:association|effect|"
    r"relationship|link|impact|influence|outcome|decline|risk)\b|"
    r"\bindirect effect\b|\bcausal (?:chain|path|pathway)\b|"
    r"through (?:the )?(?:brain|neural|imaging)|intermediary|intermediate "
    r"mechanism|accounted for (?:the )?(?:association|effect)|pathway from)",
    re.IGNORECASE,
)


def _sha256(value: object) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _spans(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _candidate_indices(items: Sequence[Mapping[str, Any]]) -> list[int]:
    candidates: list[int] = []
    for index, item in enumerate(items):
        labels = item.get("case_study_ids") or []
        if isinstance(labels, str):
            labels = [labels]
        gates = item.get("case_study_gates") or {}
        if CASE2_ID in labels or (
            isinstance(gates, Mapping) and gates.get(CASE2_GATE) is True
        ):
            candidates.append(index)
    return candidates


def _component_has_semantic_cue(
    key: str,
    spans: Sequence[str],
    candidate_items: Sequence[Mapping[str, Any]],
) -> bool:
    text = " ".join(spans)
    endpoint_types = {
        str(item.get(field) or "").strip().upper().replace("-", "_")
        for item in candidate_items
        for field in ("subject_type", "object_type")
    }
    predicates = {
        str(item.get("predicate") or "").strip().lower()
        for item in candidate_items
    }
    if key == "genetic_or_pathway":
        return bool(_GENETIC_RE.search(text)) or "GENE_TARGET" in endpoint_types
    if key == "brain_imaging_or_physiology":
        return bool(_NEURAL_RE.search(text)) or "IMAGING_MARKER" in endpoint_types
    if key == "longitudinal_clinical_or_cognitive_outcome":
        outcome_type = bool(
            endpoint_types.intersection({"OUTCOME", "COGNITIVE_TASK"})
        )
        return bool(_LONGITUDINAL_RE.search(text)) and (
            bool(_OUTCOME_RE.search(text)) or outcome_type
        )
    if key == "mediation_or_causal_chain":
        return bool(_MEDIATION_RE.search(text)) or "mediates" in predicates
    raise KeyError(key)


def _paper_chain_leg_indices(
    candidate_indices: Sequence[int],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    genetic_to_neural: list[int] = []
    neural_to_longitudinal_outcome: list[int] = []
    mediation_relation: list[int] = []
    genetic_neural_entities: dict[int, list[str]] = {}
    neural_outcome_entities: dict[int, list[str]] = {}
    for index in candidate_indices:
        item = items[index]
        types = {
            str(item.get(field) or "").strip().upper().replace("-", "_")
            for field in ("subject_type", "object_type")
        }
        raw = " ".join(
            str(item.get(field) or "")
            for field in ("subject", "predicate", "object", "raw_sentence")
        )
        has_genetic = "GENE_TARGET" in types or bool(_GENETIC_RE.search(raw))
        has_neural = "IMAGING_MARKER" in types or bool(_NEURAL_RE.search(raw))
        has_outcome = bool(
            types.intersection({"OUTCOME", "COGNITIVE_TASK"})
        ) or bool(_OUTCOME_RE.search(raw))
        has_longitudinal = bool(_LONGITUDINAL_RE.search(raw))
        if has_genetic and has_neural:
            genetic_to_neural.append(index)
            genetic_neural_entities[index] = [
                str(item.get(field) or "").strip()
                for field, type_field in (
                    ("subject", "subject_type"),
                    ("object", "object_type"),
                )
                if str(item.get(type_field) or "")
                .strip()
                .upper()
                .replace("-", "_")
                == "IMAGING_MARKER"
                and str(item.get(field) or "").strip()
            ]
        if has_neural and has_outcome and has_longitudinal:
            neural_to_longitudinal_outcome.append(index)
            neural_outcome_entities[index] = [
                str(item.get(field) or "").strip()
                for field, type_field in (
                    ("subject", "subject_type"),
                    ("object", "object_type"),
                )
                if str(item.get(type_field) or "")
                .strip()
                .upper()
                .replace("-", "_")
                == "IMAGING_MARKER"
                and str(item.get(field) or "").strip()
            ]
        # The explicit evidence span proves that mediation language exists in
        # the paper; the structured participant must additionally encode that
        # relation with the closed predicate ``mediates``.  This prevents an
        # unrelated "immune-mediated" or background sentence elsewhere in the
        # paper from completing the chain.
        if str(item.get("predicate") or "").strip().lower() == "mediates":
            mediation_relation.append(index)

    def _entity_match(left: str, right: str) -> bool:
        left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
        if not left_tokens or not right_tokens:
            return False
        if left_tokens == right_tokens:
            return True
        if left_tokens <= right_tokens or right_tokens <= left_tokens:
            return True
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.5

    connected_mediators: list[dict[str, Any]] = []
    for genetic_index, genetic_entities in genetic_neural_entities.items():
        for outcome_index, outcome_entities in neural_outcome_entities.items():
            for genetic_entity in genetic_entities:
                for outcome_entity in outcome_entities:
                    if _entity_match(genetic_entity, outcome_entity):
                        connected_mediators.append(
                            {
                                "genetic_to_neural_claim_index": genetic_index,
                                "neural_to_outcome_claim_index": outcome_index,
                                "genetic_leg_neural_entity": genetic_entity,
                                "outcome_leg_neural_entity": outcome_entity,
                            }
                        )
    mediation_connected_mediators = [
        connection
        for connection in connected_mediators
        if connection["genetic_to_neural_claim_index"] in mediation_relation
        or connection["neural_to_outcome_claim_index"] in mediation_relation
    ]
    return {
        "genetic_to_neural_claim_indices": genetic_to_neural,
        "neural_to_longitudinal_outcome_claim_indices": (
            neural_to_longitudinal_outcome
        ),
        "mediation_relation_claim_indices": mediation_relation,
        "connected_neural_mediators": connected_mediators,
        "mediation_connected_neural_mediators": mediation_connected_mediators,
    }


def build_case2_paper_chain_validation(
    items: Sequence[Mapping[str, Any]],
    *,
    source_text: str,
    source_context_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic, content-addressed paper-level Case 2 verdict.

    Every Case-2 candidate claim must repeat the same four-component evidence
    object.  Each span must occur verbatim (after whitespace normalization) in
    the supplied full abstract/body.  The validator additionally applies
    conservative semantic cue checks so arbitrary grounded text cannot satisfy
    a component merely because the model placed it in the right JSON field.
    """

    normalized_items = [item for item in items if isinstance(item, Mapping)]
    candidate_indices = _candidate_indices(normalized_items)
    source_normalized = _normalized(source_text)
    reasons: list[str] = []
    components = {key: [] for key in COMPONENT_KEYS}
    evidence_objects: list[dict[str, list[str]]] = []

    for index in candidate_indices:
        raw = normalized_items[index].get(CASE2_CHAIN_EVIDENCE_FIELD)
        if not isinstance(raw, Mapping):
            reasons.append(f"claim_index_{index}:missing_case2_paper_chain_evidence")
            continue
        normalized_evidence = {
            key: _spans(raw.get(key)) for key in COMPONENT_KEYS
        }
        evidence_objects.append(normalized_evidence)

    if candidate_indices and len(evidence_objects) == len(candidate_indices):
        reference = evidence_objects[0]
        if any(value != reference for value in evidence_objects[1:]):
            reasons.append("case2_paper_chain_evidence_not_identical_across_claims")
        components = reference

    checks: dict[str, bool] = {}
    candidate_items = [normalized_items[index] for index in candidate_indices]
    for key in COMPONENT_KEYS:
        present = bool(components[key])
        grounded = present and all(
            _normalized(span) in source_normalized for span in components[key]
        )
        semantic = present and _component_has_semantic_cue(
            key, components[key], candidate_items
        )
        checks[f"{key}_present"] = present
        checks[f"{key}_grounded"] = grounded
        checks[f"{key}_semantic_cue"] = semantic
        if candidate_indices and not present:
            reasons.append(f"{key}:missing")
        elif candidate_indices and not grounded:
            reasons.append(f"{key}:not_verbatim_in_source")
        elif candidate_indices and not semantic:
            reasons.append(f"{key}:semantic_cue_missing")

    chain_legs = _paper_chain_leg_indices(candidate_indices, normalized_items)
    checks["genetic_to_neural_claim_leg"] = bool(
        chain_legs["genetic_to_neural_claim_indices"]
    )
    checks["neural_to_longitudinal_outcome_claim_leg"] = bool(
        chain_legs["neural_to_longitudinal_outcome_claim_indices"]
    )
    checks["mediation_relation_claim"] = bool(
        chain_legs["mediation_relation_claim_indices"]
    )
    checks["connected_neural_mediator"] = bool(
        chain_legs["connected_neural_mediators"]
    )
    checks["mediation_connected_to_neural_mediator"] = bool(
        chain_legs["mediation_connected_neural_mediators"]
    )
    for check_name in (
        "genetic_to_neural_claim_leg",
        "neural_to_longitudinal_outcome_claim_leg",
        "mediation_relation_claim",
        "connected_neural_mediator",
        "mediation_connected_to_neural_mediator",
    ):
        if candidate_indices and not checks[check_name]:
            reasons.append(f"{check_name}:missing")

    if not candidate_indices:
        reasons.append("not_case2_candidate")

    record: dict[str, Any] = {
        "schema_version": CASE2_CHAIN_VALIDATION_SCHEMA,
        "validator_version": CASE2_CHAIN_VALIDATOR_VERSION,
        "source_context_sha256": str(source_context_sha256),
        "candidate_claim_indices": candidate_indices,
        "participating_claim_indices": candidate_indices,
        **chain_legs,
        "components": components,
        "checks": checks,
        "valid": bool(candidate_indices) and not reasons,
        "reasons": reasons,
    }
    record["validation_sha256"] = _sha256(record)
    return record


def validate_case2_paper_chain_record(
    record: object,
    *,
    source_context_sha256: str,
    claim_index: int | None,
    case2_labeled: bool,
) -> dict[str, Any]:
    """Validate a serialized record, failing closed for Case-2 membership."""

    if not isinstance(record, Mapping):
        if case2_labeled:
            raise ValueError("Case 2 label requires paper-level chain validation")
        return {}
    if record.get("schema_version") != CASE2_CHAIN_VALIDATION_SCHEMA:
        raise ValueError("unknown Case 2 paper-chain validation schema")
    if record.get("validator_version") != CASE2_CHAIN_VALIDATOR_VERSION:
        raise ValueError("stale Case 2 paper-chain validator version")
    if str(record.get("source_context_sha256") or "") != str(
        source_context_sha256
    ):
        raise ValueError("Case 2 paper-chain source hash mismatch")
    supplied_hash = str(record.get("validation_sha256") or "")
    unhashed = {key: value for key, value in record.items() if key != "validation_sha256"}
    if supplied_hash != _sha256(unhashed):
        raise ValueError("Case 2 paper-chain validation hash mismatch")

    if case2_labeled:
        if record.get("valid") is not True:
            raise ValueError("Case 2 paper-level complete chain is not verified")
        participants = record.get("participating_claim_indices") or []
        if claim_index is None or claim_index not in participants:
            raise ValueError("Case 2 claim is not a verified chain participant")
        components = record.get("components") or {}
        checks = record.get("checks") or {}
        for key in COMPONENT_KEYS:
            if not _spans(components.get(key)):
                raise ValueError(f"Case 2 chain component missing: {key}")
            for suffix in ("present", "grounded", "semantic_cue"):
                if checks.get(f"{key}_{suffix}") is not True:
                    raise ValueError(
                        f"Case 2 chain component check failed: {key}_{suffix}"
                    )
        for check_name in (
            "genetic_to_neural_claim_leg",
            "neural_to_longitudinal_outcome_claim_leg",
            "mediation_relation_claim",
            "connected_neural_mediator",
            "mediation_connected_to_neural_mediator",
        ):
            if checks.get(check_name) is not True:
                raise ValueError(f"Case 2 paper-chain check failed: {check_name}")
    return dict(record)


__all__ = [
    "CASE2_CHAIN_EVIDENCE_FIELD",
    "CASE2_CHAIN_VALIDATION_FIELD",
    "CASE2_CHAIN_VALIDATION_SCHEMA",
    "CASE2_CHAIN_VALIDATOR_VERSION",
    "CASE2_GATE",
    "CASE2_ID",
    "COMPONENT_KEYS",
    "build_case2_paper_chain_validation",
    "validate_case2_paper_chain_record",
]
