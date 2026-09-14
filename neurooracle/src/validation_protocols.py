"""Validation protocols that can be applied independently of case studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .case_studies import list_case_study_names


@dataclass(frozen=True)
class ValidationProtocol:
    """A reusable evaluation protocol and the case studies it supports."""

    name: str
    label: str
    description: str
    case_study_ids: tuple[str, ...]

    def supports(self, case_study_id: str) -> bool:
        return case_study_id in self.case_study_ids

    def to_dict(self) -> dict[str, str | list[str]]:
        payload = asdict(self)
        payload["case_study_ids"] = list(self.case_study_ids)
        return payload


HINDCASTING = ValidationProtocol(
    name="hindcasting",
    label="Temporal literature hindcasting",
    description=(
        "Freeze the knowledge graph at a historical cutoff, generate hypotheses "
        "within one formal case-study scope, and evaluate them against later papers."
    ),
    case_study_ids=list_case_study_names(),
)

VALIDATION_PROTOCOLS: tuple[ValidationProtocol, ...] = (HINDCASTING,)
VALIDATION_PROTOCOL_BY_NAME = {
    protocol.name: protocol for protocol in VALIDATION_PROTOCOLS
}


def validation_protocol_by_name(name: str) -> ValidationProtocol:
    protocol = VALIDATION_PROTOCOL_BY_NAME.get(name)
    if protocol is None:
        valid = ", ".join(VALIDATION_PROTOCOL_BY_NAME)
        raise KeyError(f"unknown validation protocol: {name!r} (valid: {valid})")
    return protocol


def list_validation_protocol_names() -> tuple[str, ...]:
    return tuple(protocol.name for protocol in VALIDATION_PROTOCOLS)


__all__ = [
    "HINDCASTING",
    "VALIDATION_PROTOCOLS",
    "VALIDATION_PROTOCOL_BY_NAME",
    "ValidationProtocol",
    "list_validation_protocol_names",
    "validation_protocol_by_name",
]
