"""Single source of truth for formal Case Study membership decisions.

The current peer-17 rubric remains the canonical human-readable policy.  This
module parses that file once and exposes the exact same ordered registry,
decision conditions, mandatory gates, prompt text, version, and SHA-256 to
both full-graph re-audit code and new-claim extraction.  The prior rubric is
loaded separately so already-sealed extraction records remain verifiable.

Keeping the rubric as the source avoids the previous failure mode where the
extractor and re-auditor carried independently edited summaries of the rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterable, Mapping

from .case_study_scope import CASE_STUDY_IDS, normalize_case_study_ids


_RUBRIC_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "case_study_reaudit"
    / "full_graph_v3"
)
LEGACY_RUBRIC_PATH = _RUBRIC_DIR / "RUBRIC.md"
RUBRIC_PATH = Path(__file__).with_name("CASE_STUDY_MEMBERSHIP_RUBRIC_V2.md")
CASE2_ID = "case2_pathway_mediation"
CASE2_COMPONENT_IDS = frozenset(
    {"imaging_genetics", "progression_prediction", "prognosis"}
)

_VERSION_RE = re.compile(r"^Rubric version:\s*`([^`]+)`\s*$", re.MULTILINE)
_RULE_START_RE = re.compile(r"^\d+\.\s+`([^`]+)`:\s*(.*)$")


@dataclass(frozen=True)
class CaseStudyMembershipPolicy:
    """Parsed, immutable view of one Case Study membership rubric epoch."""

    version: str
    rubric_sha256: str
    case_study_ids: tuple[str, ...]
    conditions: Mapping[str, str]
    gate_requirements: Mapping[str, str]
    gate_names: tuple[str, ...]
    evidence_policy: str


@dataclass(frozen=True)
class ScopeDecision:
    """Strictly validated claim-level Case Study decision."""

    labels: tuple[str, ...]
    gates: Mapping[str, bool]


def _normalize_markdown_text(lines: list[str]) -> str:
    return " ".join(part.strip() for part in lines if part.strip())


def _parse_conditions(rubric_text: str) -> tuple[tuple[str, ...], dict[str, str]]:
    section = rubric_text.split("## Formal IDs and decision conditions", 1)[1]
    section = section.split("## Mandatory gates", 1)[0]
    order: list[str] = []
    conditions: dict[str, str] = {}
    current_id = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_lines
        if not current_id:
            return
        conditions[current_id] = _normalize_markdown_text(current_lines)
        order.append(current_id)
        current_id = ""
        current_lines = []

    for line in section.splitlines():
        match = _RULE_START_RE.match(line)
        if match:
            flush()
            current_id = match.group(1)
            current_lines = [match.group(2)]
        elif current_id and (line.startswith("   ") or line.startswith("    ")):
            current_lines.append(line)
    flush()
    return tuple(order), conditions


def _parse_gates(rubric_text: str) -> tuple[dict[str, str], tuple[str, ...]]:
    section = rubric_text.split("## Mandatory gates", 1)[1]
    section = section.split("## Mutation gate", 1)[0]
    requirements: dict[str, str] = {}
    gate_names: list[str] = []
    for line in section.splitlines():
        if not line.lstrip().startswith("-") or "→" not in line:
            continue
        left, right = line.split("→", 1)
        labels = re.findall(r"`([^`]+)`", left)
        gate_matches = re.findall(r"`([^`]+)`", right)
        if not labels or len(gate_matches) != 1:
            raise RuntimeError(f"cannot parse mandatory gate line: {line!r}")
        gate = gate_matches[0]
        if gate not in gate_names:
            gate_names.append(gate)
        for label in labels:
            requirements[label] = gate
    return requirements, tuple(gate_names)


def _load_policy(rubric_path: Path) -> CaseStudyMembershipPolicy:
    rubric_bytes = rubric_path.read_bytes()
    rubric_text = rubric_bytes.decode("utf-8")
    version_match = _VERSION_RE.search(rubric_text)
    if version_match is None:
        raise RuntimeError(f"rubric version is missing from {rubric_path}")

    parsed_ids, conditions = _parse_conditions(rubric_text)
    registry = tuple(CASE_STUDY_IDS)
    if parsed_ids != registry:
        raise RuntimeError(
            "rubric Case Study order differs from the formal registry: "
            f"rubric={parsed_ids!r}, registry={registry!r}"
        )

    gate_requirements, gate_names = _parse_gates(rubric_text)
    unknown_gate_labels = sorted(set(gate_requirements) - set(registry))
    if unknown_gate_labels:
        raise RuntimeError(
            f"rubric mandatory gates reference unknown IDs: {unknown_gate_labels}"
        )

    evidence_section = rubric_text.splitlines()[4:15]
    evidence_policy = _normalize_markdown_text(evidence_section)
    return CaseStudyMembershipPolicy(
        version=version_match.group(1),
        rubric_sha256=hashlib.sha256(rubric_bytes).hexdigest(),
        case_study_ids=registry,
        conditions=MappingProxyType(dict(conditions)),
        gate_requirements=MappingProxyType(dict(gate_requirements)),
        gate_names=gate_names,
        evidence_policy=evidence_policy,
    )


LEGACY_POLICY = _load_policy(LEGACY_RUBRIC_PATH)
POLICY = _load_policy(RUBRIC_PATH)
RUBRIC_VERSION = POLICY.version
RUBRIC_SHA256 = POLICY.rubric_sha256
GATE_REQUIREMENTS = POLICY.gate_requirements
GATE_NAMES = POLICY.gate_names


def case_study_policy_prompt(*, extraction: bool) -> str:
    """Render the frozen policy for either extraction or standalone review."""

    lines = [
        f"FROZEN CASE-STUDY MEMBERSHIP POLICY — {RUBRIC_VERSION}",
        POLICY.evidence_policy,
        "Formal non-exclusive IDs and exact decision conditions:",
    ]
    for case_study_id in POLICY.case_study_ids:
        lines.append(f"- {case_study_id}: {POLICY.conditions[case_study_id]}")
    lines.extend(
        [
            "Mandatory gates (a label is invalid unless its gate is true):",
            *(
                f"- {label} -> {gate}"
                for label, gate in POLICY.gate_requirements.items()
            ),
            "Return every applicable label, not only the best one. "
            "Use [] for a general-only claim. Never emit general or hindcasting as IDs.",
            "Case 2 is deterministic: every imaging_genetics, progression_prediction, "
            "or prognosis claim also receives case2_pathway_mediation. Never emit "
            "case2_pathway_mediation without at least one of those direct component labels.",
        ]
    )
    if extraction:
        lines.append(
            "The search query, candidate queue, title, venue, and future-work text "
            "must not donate a missing claim relation. Assign scope to the extracted "
            "claim itself; paper context may only verify terminology and must not "
            "donate an otherwise missing component relation."
        )
    return "\n".join(lines)


def _canonicalize_case2_membership(labels: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(labels)
    has_component = bool(selected.intersection(CASE2_COMPONENT_IDS))
    if CASE2_ID in selected and not has_component:
        raise ValueError(
            "case2_pathway_mediation requires a direct imaging_genetics, "
            "progression_prediction, or prognosis component"
        )
    if has_component:
        selected.add(CASE2_ID)
    return tuple(value for value in POLICY.case_study_ids if value in selected)


def validate_scope_decision_for_policy(
    labels: object,
    gates: object,
    *,
    policy: CaseStudyMembershipPolicy,
    derive_case2: bool,
) -> ScopeDecision:
    """Validate one decision against an explicit rubric epoch."""

    if not isinstance(labels, list) or any(not isinstance(value, str) for value in labels):
        raise ValueError("case_study_ids must be a JSON string list")
    normalized = normalize_case_study_ids(labels, strict=True)
    if len(normalized) != len(labels):
        raise ValueError("case_study_ids must be unique formal IDs")
    normalized_labels = tuple(normalized)
    if derive_case2:
        normalized_labels = _canonicalize_case2_membership(normalized_labels)

    if not isinstance(gates, Mapping) or set(gates) != set(policy.gate_names):
        raise ValueError(
            "case_study_gates must contain exactly: " + ", ".join(policy.gate_names)
        )
    if any(not isinstance(gates[name], bool) for name in policy.gate_names):
        raise ValueError("case_study_gates values must be JSON booleans")
    normalized_gates = {name: gates[name] for name in policy.gate_names}
    for label in normalized_labels:
        required_gate = policy.gate_requirements.get(label)
        if required_gate and not normalized_gates[required_gate]:
            raise ValueError(
                f"{label} assigned without mandatory gate {required_gate}"
            )
    return ScopeDecision(normalized_labels, normalized_gates)


def validate_scope_decision(
    labels: object,
    gates: object,
) -> ScopeDecision:
    """Validate and canonicalize a current-epoch Case Study decision."""

    return validate_scope_decision_for_policy(
        labels,
        gates,
        policy=POLICY,
        derive_case2=True,
    )


def gates_for_labels(labels: Iterable[str]) -> dict[str, bool]:
    """Build the minimal valid gate object for trusted/manual decisions."""

    normalized = _canonicalize_case2_membership(
        tuple(normalize_case_study_ids(list(labels), strict=True))
    )
    gates = {name: False for name in GATE_NAMES}
    for label in normalized:
        gate = GATE_REQUIREMENTS.get(label)
        if gate:
            gates[gate] = True
    return gates


__all__ = [
    "CASE2_COMPONENT_IDS",
    "CASE2_ID",
    "CaseStudyMembershipPolicy",
    "GATE_NAMES",
    "GATE_REQUIREMENTS",
    "LEGACY_POLICY",
    "LEGACY_RUBRIC_PATH",
    "POLICY",
    "RUBRIC_PATH",
    "RUBRIC_SHA256",
    "RUBRIC_VERSION",
    "ScopeDecision",
    "case_study_policy_prompt",
    "gates_for_labels",
    "validate_scope_decision",
    "validate_scope_decision_for_policy",
]
