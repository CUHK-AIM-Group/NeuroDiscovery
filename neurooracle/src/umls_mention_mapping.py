"""Auditable UMLS alignment for claim-derived mention nodes.

``CLM_CONCEPT`` nodes are evidence-bearing mentions, not canonical concepts.
They must therefore keep their identifiers and graph identity.  This module
provides deterministic helpers for deriving atomic mention projections and
linking those projections to one or more UMLS CUI nodes with ``maps_to``
edges.  It deliberately performs no graph mutation.

Only conservative, normalized-exact matching is supported here.  Fuzzy or
embedding candidates require a separate human-review workflow and must not be
promoted by this module.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


UMLS_MENTION_PREFIX = "CLM_CONCEPT:"
ATOMIC_MENTION_PREFIX = "CLM_ATOM:"

INFRASTRUCTURE_DOMAINS = frozenset(
    {
        "spatial_reference",
        "atlas",
        "modality",
        "dataset",
        "ml_model",
        "recipe",
        "claim",
    }
)

BIOMEDICAL_DOMAINS = frozenset(
    {
        "disease",
        "drug",
        "gene",
        "neurotransmitter",
        "neuroanatomy",
        "cognitive_function",
        "paradigm",
        "visual_stimulus",
        "emotion",
        "vigilance",
        "biomarker",
        "imaging_feature",
        "connectivity",
        "treatment_outcome",
        "dataset_variable",
        "individual_data_anchor",
        "claim_concept",
    }
)

ATOM_TYPE_TO_DOMAIN = {
    "disease": "disease",
    "drug": "drug",
    "gene": "gene",
    "gene_target": "gene",
    "imaging_marker": "biomarker",
    "cognitive_task": "cognitive_function",
    "outcome": "treatment_outcome",
    "individual_data": "individual_data_anchor",
}

_GENERIC_NON_ENTITY_TERMS = frozenset(
    {
        "analysis",
        "association",
        "change",
        "changes",
        "comparison",
        "control",
        "controls",
        "data",
        "dataset",
        "difference",
        "differences",
        "bilateral",
        "effect",
        "effects",
        "finding",
        "findings",
        "group",
        "groups",
        "measure",
        "measurement",
        "method",
        "model",
        "left",
        "outcome",
        "participants",
        "patients",
        "performance",
        "result",
        "results",
        "response",
        "right",
        "risk",
        "sample",
        "score",
        "study",
        "variable",
    }
)

_TRIM_CHARS = " \t\r\n,;:|/\\+&.-"
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
_SPACE_RE = re.compile(r"\s+")
_TOP_LEVEL_SEPARATOR_RE = re.compile(
    r"\s+(?:and|or|versus|vs\.?)\s+|\s*[;|]\s*|\s+[+&/]\s+|,\s+",
    flags=re.IGNORECASE,
)
_TRAILING_PAREN_RE = re.compile(r"^(?P<base>.+?)\s*\((?P<inner>[^()]{2,80})\)\s*$")
_LEADING_MODIFIER_RE = re.compile(
    r"^(?:abnormal|altered|decreased|elevated|enhanced|greater|higher|"
    r"impaired|increased|larger|lower|reduced|smaller)\s+",
    flags=re.IGNORECASE,
)
_URL_RE = re.compile(r"(?:https?://|www\.)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class LookupVariant:
    """One deterministic normalized-exact lookup form for an atomic unit."""

    text: str
    normalized: str
    rule: str
    source_field: str
    priority: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "normalized": self.normalized,
            "rule": self.rule,
            "source_field": self.source_field,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class AtomicMentionCandidate:
    """A possible atomic projection of one preserved ``CLM_CONCEPT`` node."""

    id: str
    source_mention_id: str
    text: str
    normalized: str
    start: int
    end: int
    atomization_rule: str
    variants: tuple[LookupVariant, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_mention_id": self.source_mention_id,
            "text": self.text,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
            "atomization_rule": self.atomization_rule,
            "variants": [variant.to_dict() for variant in self.variants],
        }


def normalize_term(text: str) -> str:
    """Normalize a term for conservative exact matching.

    The normalization is intentionally transparent: Unicode NFKC, underscore
    and dash harmonization, surrounding punctuation trimming, case folding,
    and whitespace collapse.  There is no stemming, token deletion, fuzzy
    distance, or embedding similarity.
    """

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.translate(_DASH_TRANSLATION).replace("_", " ")
    value = _SPACE_RE.sub(" ", value).strip(_TRIM_CHARS)
    return value.casefold()


def is_lexically_eligible(text: str) -> tuple[bool, str]:
    """Return whether a surface is a complete, entity-like mention."""

    normalized = normalize_term(text)
    if not normalized:
        return False, "empty"
    if len(normalized) < 2:
        return False, "too_short"
    if len(normalized) > 240:
        return False, "too_long"
    if _URL_RE.search(normalized):
        return False, "url"
    if not any(char.isalpha() for char in normalized):
        return False, "no_letters"
    if len(normalized.split()) > 40:
        return False, "too_many_tokens"
    if normalized in _GENERIC_NON_ENTITY_TERMS:
        return False, "generic_non_entity"
    return True, "eligible"


def biomedical_context(
    domain_tags: Iterable[str] | None,
    atom_types: Iterable[str] | None = None,
) -> tuple[bool, tuple[str, ...], str]:
    """Classify whether a claim mention belongs in the biomedical denominator."""

    domains = {str(value).strip().casefold() for value in domain_tags or [] if value}
    for atom_type in atom_types or []:
        mapped = ATOM_TYPE_TO_DOMAIN.get(str(atom_type).strip().casefold())
        if mapped:
            domains.add(mapped)

    biomedical = tuple(sorted(domains & BIOMEDICAL_DOMAINS))
    if biomedical:
        return True, biomedical, "biomedical_domain_or_atom_type"
    if domains and domains <= INFRASTRUCTURE_DOMAINS:
        return False, tuple(sorted(domains)), "infrastructure_only"
    if not domains:
        return False, (), "missing_biomedical_domain"
    return False, tuple(sorted(domains)), "unsupported_domain"


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    return start, end


def _variant(
    text: str,
    *,
    rule: str,
    source_field: str,
    priority: int,
) -> LookupVariant | None:
    normalized = normalize_term(text)
    if not normalized:
        return None
    return LookupVariant(
        text=text.strip(),
        normalized=normalized,
        rule=rule,
        source_field=source_field,
        priority=priority,
    )


def _lookup_variants(
    surface: str,
    *,
    source_field: str,
    aliases: Sequence[str] = (),
) -> tuple[LookupVariant, ...]:
    variants: list[LookupVariant] = []
    base = _variant(
        surface,
        rule="normalized_exact",
        source_field=source_field,
        priority=10,
    )
    if base:
        variants.append(base)

    parenthetical = _TRAILING_PAREN_RE.match(surface.strip())
    if parenthetical:
        parenthetical_base = _variant(
            parenthetical.group("base"),
            rule="parenthetical_base_exact",
            source_field=source_field,
            priority=30,
        )
        parenthetical_inner = _variant(
            parenthetical.group("inner"),
            rule="parenthetical_inner_exact",
            source_field=source_field,
            priority=35,
        )
        if parenthetical_base:
            variants.append(parenthetical_base)
        if parenthetical_inner:
            variants.append(parenthetical_inner)

    modifier_match = _LEADING_MODIFIER_RE.match(surface.strip())
    if modifier_match:
        stripped = surface.strip()[modifier_match.end() :]
        modifier_variant = _variant(
            stripped,
            rule="leading_modifier_stripped_exact",
            source_field=source_field,
            priority=40,
        )
        if modifier_variant:
            variants.append(modifier_variant)

    for index, alias in enumerate(aliases[:8]):
        alias_variant = _variant(
            alias,
            rule="alias_normalized_exact",
            source_field=f"aliases[{index}]",
            priority=20,
        )
        if alias_variant:
            variants.append(alias_variant)

    deduplicated: dict[str, LookupVariant] = {}
    for item in variants:
        current = deduplicated.get(item.normalized)
        if current is None or item.priority < current.priority:
            deduplicated[item.normalized] = item
    return tuple(sorted(deduplicated.values(), key=lambda item: (item.priority, item.normalized)))


def _atomic_id(source_mention_id: str, start: int, end: int, normalized: str) -> str:
    payload = f"{source_mention_id}\x1f{start}\x1f{end}\x1f{normalized}".encode("utf-8")
    return f"{ATOMIC_MENTION_PREFIX}{hashlib.sha256(payload).hexdigest()[:24]}"


def _candidate(
    source_mention_id: str,
    source_text: str,
    start: int,
    end: int,
    *,
    rule: str,
    aliases: Sequence[str] = (),
) -> AtomicMentionCandidate:
    start, end = _trim_span(source_text, start, end)
    surface = source_text[start:end]
    normalized = normalize_term(surface)
    return AtomicMentionCandidate(
        id=_atomic_id(source_mention_id, start, end, normalized),
        source_mention_id=source_mention_id,
        text=surface,
        normalized=normalized,
        start=start,
        end=end,
        atomization_rule=rule,
        variants=_lookup_variants(
            surface,
            source_field="preferred_name",
            aliases=aliases,
        ),
    )


def _top_level_split_spans(text: str) -> list[tuple[int, int]]:
    depth = 0
    depth_at: list[int] = [0] * (len(text) + 1)
    for index, char in enumerate(text):
        depth_at[index] = depth
        if char in "([{" :
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
    depth_at[len(text)] = depth

    separators = [
        match
        for match in _TOP_LEVEL_SEPARATOR_RE.finditer(text)
        if depth_at[match.start()] == 0
    ]
    if not separators:
        return []

    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in separators:
        start, end = _trim_span(text, cursor, separator.start())
        if start < end:
            spans.append((start, end))
        cursor = separator.end()
    start, end = _trim_span(text, cursor, len(text))
    if start < end:
        spans.append((start, end))
    return spans


def atomize_mention_candidates(
    source_mention_id: str,
    preferred_name: str,
    aliases: Sequence[str] = (),
) -> dict:
    """Return full and conservative top-level split candidates.

    The caller should prefer the full candidate whenever it has a compatible
    exact UMLS match.  Split candidates are selected only when the full form
    does not map and every top-level component is lexically complete.
    """

    source_text = str(preferred_name or "")
    full_start, full_end = _trim_span(source_text, 0, len(source_text))
    full = _candidate(
        source_mention_id,
        source_text,
        full_start,
        full_end,
        rule="full_mention",
        aliases=aliases,
    )

    spans = _top_level_split_spans(source_text)
    split: list[AtomicMentionCandidate] = []
    if 2 <= len(spans) <= 8:
        possible = [
            _candidate(
                source_mention_id,
                source_text,
                start,
                end,
                rule="top_level_composite_split",
            )
            for start, end in spans
        ]
        if all(is_lexically_eligible(item.text)[0] for item in possible):
            split = possible

    return {
        "source_text": source_text,
        "full": full.to_dict(),
        "split": [item.to_dict() for item in split],
        "has_syntactic_composite": bool(split),
    }


# UMLS semantic type names are stored verbatim from MRSTY.  Compatibility is
# intentionally explicit and reviewable rather than inferred from embeddings.
DOMAIN_TO_ALLOWED_STY: Mapping[str, frozenset[str]] = {
    "disease": frozenset(
        {
            "Acquired Abnormality",
            "Anatomical Abnormality",
            "Congenital Abnormality",
            "Disease or Syndrome",
            "Finding",
            "Injury or Poisoning",
            "Mental or Behavioral Dysfunction",
            "Neoplastic Process",
            "Pathologic Function",
            "Sign or Symptom",
        }
    ),
    "drug": frozenset(
        {
            "Antibiotic",
            "Biologically Active Substance",
            "Clinical Drug",
            "Hazardous or Poisonous Substance",
            "Hormone",
            "Immunologic Factor",
            "Neuroreactive Substance or Biogenic Amine",
            "Organic Chemical",
            "Pharmacologic Substance",
            "Steroid",
            "Vitamin",
        }
    ),
    "gene": frozenset(
        {
            "Amino Acid, Peptide, or Protein",
            "Biologically Active Substance",
            "Enzyme",
            "Gene or Genome",
            "Nucleic Acid, Nucleoside, or Nucleotide",
            "Receptor",
        }
    ),
    "neurotransmitter": frozenset(
        {
            "Amino Acid, Peptide, or Protein",
            "Biologically Active Substance",
            "Neuroreactive Substance or Biogenic Amine",
            "Organic Chemical",
        }
    ),
    "neuroanatomy": frozenset(
        {
            "Body Location or Region",
            "Body Part, Organ, or Organ Component",
            "Cell",
            "Cell Component",
            "Embryonic Structure",
            "Tissue",
        }
    ),
    "cognitive_function": frozenset(
        {
            "Finding",
            "Functional Concept",
            "Individual Behavior",
            "Mental Process",
            "Social Behavior",
        }
    ),
    "paradigm": frozenset(
        {
            "Diagnostic Procedure",
            "Educational Activity",
            "Individual Behavior",
            "Mental Process",
            "Research Activity",
        }
    ),
    "visual_stimulus": frozenset(
        {"Finding", "Functional Concept", "Mental Process", "Phenomenon or Process"}
    ),
    "emotion": frozenset(
        {"Finding", "Individual Behavior", "Mental Process", "Sign or Symptom"}
    ),
    "vigilance": frozenset(
        {"Finding", "Mental Process", "Organism Function", "Sign or Symptom"}
    ),
    "biomarker": frozenset(
        {
            "Amino Acid, Peptide, or Protein",
            "Biologically Active Substance",
            "Body Substance",
            "Clinical Attribute",
            "Finding",
            "Gene or Genome",
            "Laboratory or Test Result",
            "Sign or Symptom",
        }
    ),
    "imaging_feature": frozenset(
        {
            "Body Location or Region",
            "Body Part, Organ, or Organ Component",
            "Clinical Attribute",
            "Diagnostic Procedure",
            "Finding",
            "Functional Concept",
            "Laboratory or Test Result",
            "Quantitative Concept",
        }
    ),
    "connectivity": frozenset(
        {
            "Body Location or Region",
            "Body Part, Organ, or Organ Component",
            "Finding",
            "Functional Concept",
            "Physiologic Function",
        }
    ),
    "treatment_outcome": frozenset(
        {
            "Clinical Attribute",
            "Finding",
            "Functional Concept",
            "Laboratory or Test Result",
            "Mental Process",
            "Organism Function",
            "Qualitative Concept",
            "Quantitative Concept",
            "Sign or Symptom",
        }
    ),
    "dataset_variable": frozenset(
        {
            "Age Group",
            "Clinical Attribute",
            "Finding",
            "Functional Concept",
            "Laboratory or Test Result",
            "Mental Process",
            "Organism Attribute",
            "Organism Function",
            "Population Group",
            "Qualitative Concept",
            "Quantitative Concept",
            "Sign or Symptom",
        }
    ),
    "individual_data_anchor": frozenset(
        {
            "Age Group",
            "Clinical Attribute",
            "Finding",
            "Gene or Genome",
            "Individual Behavior",
            "Mental Process",
            "Organism Attribute",
            "Population Group",
            "Sign or Symptom",
        }
    ),
}

_GENERIC_CLAIM_ALLOWED_STY = frozenset().union(*DOMAIN_TO_ALLOWED_STY.values())


def semantic_compatibility(
    domain_tags: Iterable[str] | None,
    atom_types: Iterable[str] | None,
    semantic_types: Sequence[Mapping[str, str]],
) -> tuple[str, tuple[str, ...], str]:
    """Return ``(status, matched_STY_names, basis)`` for a CUI candidate."""

    domains = {str(value).strip().casefold() for value in domain_tags or [] if value}
    for atom_type in atom_types or []:
        mapped = ATOM_TYPE_TO_DOMAIN.get(str(atom_type).strip().casefold())
        if mapped:
            domains.add(mapped)

    sty_names = {str(item.get("sty") or "") for item in semantic_types if item.get("sty")}
    if not sty_names:
        return "unknown", (), "missing_mrsty"

    allowed: set[str] = set()
    explicit_domains: list[str] = []
    for domain in sorted(domains):
        domain_allowed = DOMAIN_TO_ALLOWED_STY.get(domain)
        if domain_allowed:
            explicit_domains.append(domain)
            allowed.update(domain_allowed)

    if not allowed and "claim_concept" in domains:
        allowed.update(_GENERIC_CLAIM_ALLOWED_STY)
        basis = "generic_claim_concept"
    elif allowed:
        basis = "domain:" + ",".join(explicit_domains)
    else:
        return "unknown", (), "no_semantic_contract"

    overlap = tuple(sorted(sty_names & allowed))
    if overlap:
        return "compatible", overlap, basis
    return "incompatible", (), basis


def preferred_domain(domain_tags: Iterable[str], atom_types: Iterable[str]) -> str:
    """Return one deterministic reporting bucket for a mention."""

    domains = {str(value).strip().casefold() for value in domain_tags if value}
    for atom_type in atom_types:
        mapped = ATOM_TYPE_TO_DOMAIN.get(str(atom_type).strip().casefold())
        if mapped:
            domains.add(mapped)
    order = (
        "disease",
        "drug",
        "gene",
        "neurotransmitter",
        "neuroanatomy",
        "cognitive_function",
        "paradigm",
        "biomarker",
        "imaging_feature",
        "connectivity",
        "treatment_outcome",
        "dataset_variable",
        "individual_data_anchor",
        "claim_concept",
    )
    return next((domain for domain in order if domain in domains), "other")


def select_atomic_units(record: Mapping, full_has_compatible_match: bool) -> tuple[list[dict], str]:
    """Select full or split units without deleting the source mention."""

    if full_has_compatible_match:
        return [dict(record["full"])], "full_exact_preferred_over_split"
    split = [dict(item) for item in record.get("split") or []]
    if len(split) >= 2:
        return split, "composite_split_after_full_exact_miss"
    return [dict(record["full"])], "full_unsplit_no_valid_composite"


__all__ = [
    "ATOMIC_MENTION_PREFIX",
    "BIOMEDICAL_DOMAINS",
    "DOMAIN_TO_ALLOWED_STY",
    "INFRASTRUCTURE_DOMAINS",
    "UMLS_MENTION_PREFIX",
    "AtomicMentionCandidate",
    "LookupVariant",
    "atomize_mention_candidates",
    "biomedical_context",
    "is_lexically_eligible",
    "normalize_term",
    "preferred_domain",
    "select_atomic_units",
    "semantic_compatibility",
]
