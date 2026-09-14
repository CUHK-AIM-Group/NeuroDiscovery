"""Conservative semantic QA for claim endpoints mapped to canonical KG nodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Any, Mapping

from .atoms import Atom, DOMAIN_TO_ATOMS


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPACE_RE = re.compile(r"\s+")
_GENERIC_SINGLE_TOKENS = frozenset({
    "activation", "activity", "behavior", "biomarker", "brain", "cognition",
    "connectivity", "cortex", "cortical", "disease", "disorder", "feature", "function",
    "marker", "network", "outcome", "response", "risk", "symptom", "treatment",
})
SEMANTIC_ENDPOINT_IDENTITY_VERSION = "semantic-endpoint-identity.v8"
_ABBREVIATIONS = {
    "adhd": ("attention", "deficit", "hyperactivity", "disorder"),
    "asd": ("autism", "spectrum", "disorder"),
    "mdd": ("major", "depressive", "disorder"),
    "ocd": ("obsessive", "compulsive", "disorder"),
    "ptsd": ("posttraumatic", "stress", "disorder"),
    "scz": ("schizophrenia",),
}
_IMAGING_NAME_RE = re.compile(
    r"\b(mri|fmri|pet|spect|dti|bold|alff|falff|reho|suvr|volume|thickness|"
    r"surface area|anisotropy|diffusivity|connectivity|activation|reactivity|atrophy|"
    r"(?:regional|network|interregional) homogeneity|circularity|gyrification|curvature|"
    r"sulcal depth|folding|shape index|surface complexity|fractal dimension|"
    r"deactivation|hypoactivation|hyperactivation|activity|responsiveness|recruitment|entropy|"
    r"amplitude|power band|evoked potential|eeg|erp|meg|"
    r"laterality|synchronization|coherence|oscillation|signal amplitude|"
    r"causal interactions?|effective connectivity|functional coupling|network flexibility|"
    r"network module|network organization|binding potential|tracer binding|uptake|"
    r"hypometabolism|perfusion|cerebral blood flow|rcbf|flair|hypersignal|tractography|"
    r"connectome|controllability|centrality|efficiency|node strength|small[- ]worldness|"
    r"modularity|participation coefficient|rich[- ]club|structural covariance|"
    r"functional hierarchy|network stability|network similarity|"
    r"morphometr(?:y|ic)|microstructure|integrity|gray[- ]white matter contrast|"
    r"grey[- ]white matter contrast|regional vulnerability index|"
    r"white matter|gray matter|grey matter|brain[- ]pad|"
    r"brain[- ](?:predicted )?(?:age|disease duration)(?: gap|"
    r"difference|index|score)?)\b",
    re.I,
)
_GENETIC_NAME_RE = re.compile(
    r"\b(gene|genotype|variant|allele|snp|polygenic|pathway|expression|methylation)\b",
    re.I,
)
_TASK_NAME_RE = re.compile(
    r"\b(task|stimulus|stimuli|paradigm|n back|stroop|go no go|stop signal|"
    r"working memory|face processing|emotion recognition|reward anticipation|"
    r"visuo-spatial imagery|visuospatial imagery|mental imagery|episodic memory retrieval|"
    r"self processing|altered consciousness state|response inhibition|decision making|"
    r"perseverative cognition|inhibitory control|social cognition|interoception|"
    r"emotion regulation|reward processing|language processing|action observation|"
    r"theory of mind|mentalizing|memory encoding|memory retrieval)\b",
    re.I,
)
_FUNCTIONAL_IMAGING_NAME_RE = re.compile(
    r"\b(fmri|bold|alff|falff|reho|activation|deactivation|hypoactivation|"
    r"hyperactivation|activity|reactivity|responsiveness|connectivity|coupling|recruitment|"
    r"synchronization|coherence|oscillation|entropy|network|perfusion|amplitude|"
    r"power band|evoked potential|eeg|erp|meg|"
    r"cerebral blood flow)\b",
    re.I,
)
_FUNCTIONAL_READOUT_MEASUREMENT_RE = re.compile(
    r"\b(?:alff|falff|reho|activation|deactivation|hypoactivation|hyperactivation|"
    r"activity|reactivity|responsiveness|connectivity|coupling|recruitment|"
    r"synchronization|coherence|oscillation|entropy|perfusion|amplitude|"
    r"power band|evoked potentials?|eeg|erp|meg|cerebral blood flow|rcbf|"
    r"(?:regional|network|interregional) homogeneity|"
    r"network (?:flexibility|module|organization))\b",
    re.I,
)
_FUNCTIONAL_GENERIC_TOKENS = frozenset({
    "activation", "activity", "alff", "amplitude", "average", "band", "bold",
    "brain", "coherence", "connectivity", "coupling", "deactivation", "eeg", "erp",
    "entropy", "evoked", "falff", "fmri", "functional", "hyperactivation",
    "hypoactivation", "mean", "meg", "network", "oscillation", "perfusion",
    "and", "fluctuation", "for", "frequency", "from", "in", "low", "of",
    "potential", "power", "reactivity", "readout", "regional", "reho", "response", "resting",
    "signal", "state", "synchronization", "task", "the", "to", "whole",
})
_NON_TASK_INSTRUMENT_RE = re.compile(
    r"\b(scale|questionnaire|inventory|quotient|schedule|survey|rating|checklist)\b",
    re.I,
)
_NON_TASK_PHENOTYPE_RE = re.compile(
    r"\b(?:abilit(?:y|ies)|capacity|deficits?|impairments?|performance|scores?|"
    r"severity|symptoms?|traits?|functioning)\b",
    re.I,
)
_CLINICAL_STATE_RE = re.compile(
    r"\b(psychosis|addiction|hyperactivity|schizophrenia|bipolar|autism|"
    r"disease|disorder|syndrome)\b",
    re.I,
)
_METHOD_OR_PROCEDURE_ENTITY_RE = re.compile(
    r"\b(?:algorithms?|architecture|classifier|classification algorithms?|"
    r"(?:brain[- ]age|deep[- ]learning|machine[- ]learning|prediction|"
    r"regression|statistical|time[- ]to[- ]event) model|"
    r"(?:convolutional|fully[- ]convolutional|graph) (?:neural )?network|"
    r"deep[- ]learning (?:algorithm|architecture|pipeline)|pipeline|"
    r"(?:extreme[- ]learning machine|multivoxel pattern analysis|mvpa)|"
    r"principal component analysis|random forest|registration|segmentation|"
    r"software|support vector machine)\b",
    re.I,
)
_CONCRETE_IMAGING_MEASUREMENT_RE = re.compile(
    r"\b(?:suvr|volum(?:e|es|etric|etry)|thickness|thinning|surface area|"
    r"fractional anisotropy|mean diffusivity|radial diffusivity|axial diffusivity|"
    r"anisotropy|diffusivity|"
    r"atrophy|binding potential|tracer binding|uptake|hypometabolism|perfusion|"
    r"(?:regional|network|interregional) homogeneity|circularity|gyrification|curvature|"
    r"sulcal depth|folding|shape index|surface complexity|fractal dimension|"
    r"cerebral blood flow|rcbf|hypersignal|tractography|lesion(?: volume| burden)?|"
    r"white matter hyperintensit(?:y|ies)|gray matter density|grey matter density|"
    r"gray[- ]white matter contrast|grey[- ]white matter contrast|"
    r"connectome (?:stability|similarity)|(?:network|regional) controllability|"
    r"(?:network|regional|nodal|global|functional|structural) (?:centrality|efficiency)|"
    r"node strength|small[- ]worldness|modularity|participation coefficient|"
    r"rich[- ]club|structural covariance|functional hierarchy|"
    r"regional vulnerability index|white matter (?:integrity|microstructure)|"
    r"morphometric (?:index|indices|pattern|profile|signature|trajectory|value|values)|"
    r"brain[- ]pad|brain[- ](?:predicted )?(?:age|disease duration)(?: gap|"
    r"difference|index|score)?)\b",
    re.I,
)
_DIFFUSION_ABBREVIATION_RE = re.compile(
    r"\b(?:white matter|tract|fascicul|cingulum|corpus callosum|diffusion|dti)\b"
    r".{0,120}\b(?:fa|md|rd|ad)\b|"
    r"\b(?:fa|md|rd|ad)\b.{0,120}"
    r"\b(?:white matter|tract|fascicul|cingulum|corpus callosum|diffusion|dti)\b",
    re.I,
)
_MOLECULAR_PET_READOUT_RE = re.compile(
    r"\b(?:(?:positive|negative)\s+(?:cerebral\s+)?(?:amyloid|tau|fdg)\s+pet|"
    r"(?:amyloid|tau|fdg)\s+pet\s+(?:burden|positivity|signal|suvr|uptake))\b",
    re.I,
)
_ANATOMICAL_RESPONSE_READOUT_RE = re.compile(
    r"\b(?:amygdal\w*|brain|caudate|cerebell\w*|cingulat\w*|cortex|cortical|"
    r"hippocamp\w*|insula\w*|neural|prefrontal|putamen|region\w*|striat\w*|"
    r"thalam\w*|ventral striatum)\b.{0,160}\bresponses?\s+to\b|"
    r"\b(?:bold|fmri)\s+(?:signal\s+)?responses?\b",
    re.I,
)
_NON_IMAGING_ASSAY_ENTITY_RE = re.compile(
    r"\b(?:(?:enzyme|kinase|phosphatase)\s+(?:activity|expression|level)|"
    r"protein\s+(?:activity|concentration|expression|level)|immunoreactivity|"
    r"electrocardiograph(?:y|ic)?|ecg|ekg|magnetocardiograph(?:y|ic)?|mcg|"
    r"cardiotocograph(?:y|ic)?|ctg|heart[- ]rate variability|hrv|"
    r"fetal autonomic brain age|"
    r"(?:cerebrospinal fluid|csf|plasma|saliva|serum|tissue|urine)\s+"
    r"(?:[a-z0-9-]+\s+){0,3}(?:activity|assay|biomarker|concentration|expression|level|marker)|"
    r"(?:blood|cerebrospinal fluid|csf|plasma|saliva|serum|tissue|urine)\s+"
    r"(?:assay|biomarker|concentration|expression|level|marker))\b",
    re.I,
)


def normalize_entity_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("\u03b2", " beta ").replace("\u03b5", " epsilon ")
    text = text.replace("'s", " ")
    return _SPACE_RE.sub(" ", " ".join(_TOKEN_RE.findall(text))).strip()


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def entity_name_tokens(value: Any) -> frozenset[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(normalize_entity_name(value)):
        expanded = _ABBREVIATIONS.get(token)
        if expanded:
            tokens.update(_stem_token(item) for item in expanded)
        else:
            tokens.add(_stem_token(token))
    return frozenset(tokens)


def looks_like_imaging_measurement(value: Any) -> bool:
    return bool(_IMAGING_NAME_RE.search(normalize_entity_name(value)))


def looks_like_concrete_imaging_measurement(value: Any) -> bool:
    """Return whether a phrase names an imaging-derived readout, not a modality."""

    normalized = normalize_entity_name(value)
    # Functional readouts can be specific even when a new metric term has not yet
    # reached the broader imaging lexicon (for example, network homogeneity).
    if is_specific_functional_imaging_readout(normalized):
        return True
    if (
        _DIFFUSION_ABBREVIATION_RE.search(normalized)
        or _MOLECULAR_PET_READOUT_RE.search(normalized)
        or _ANATOMICAL_RESPONSE_READOUT_RE.search(normalized)
    ):
        return True
    return bool(
        looks_like_imaging_measurement(normalized)
        and _CONCRETE_IMAGING_MEASUREMENT_RE.search(normalized)
    )


def looks_like_method_or_procedure_entity(value: Any) -> bool:
    """Return whether an endpoint names an analytic method instead of its readout."""

    return bool(_METHOD_OR_PROCEDURE_ENTITY_RE.search(normalize_entity_name(value)))


def looks_like_non_imaging_assay_entity(value: Any) -> bool:
    """Return whether an endpoint is a molecular/fluid assay rather than imaging."""

    return bool(_NON_IMAGING_ASSAY_ENTITY_RE.search(normalize_entity_name(value)))


def looks_like_cognitive_task_or_stimulus(value: Any) -> bool:
    """Return whether a phrase names an experimental task, stimulus, or operation."""

    return bool(_TASK_NAME_RE.search(normalize_entity_name(value)))


def looks_like_functional_imaging_readout(value: Any) -> bool:
    """Return whether a phrase names a functional neural or imaging readout."""

    return bool(_FUNCTIONAL_IMAGING_NAME_RE.search(normalize_entity_name(value)))


def is_specific_functional_imaging_readout(value: Any) -> bool:
    """Require a functional readout to retain a concrete anatomical qualifier."""

    normalized = normalize_entity_name(value)
    if (
        not looks_like_functional_imaging_readout(normalized)
        or not _FUNCTIONAL_READOUT_MEASUREMENT_RE.search(normalized)
    ):
        return False
    content = {
        token for token in normalized.split()
        if token not in _FUNCTIONAL_GENERIC_TOKENS
    }
    return bool(content)


def looks_like_non_task_construct(value: Any) -> bool:
    """Reject instruments and clinical states that were globally typed as tasks."""

    normalized = normalize_entity_name(value)
    if (
        _NON_TASK_INSTRUMENT_RE.search(normalized)
        or _NON_TASK_PHENOTYPE_RE.search(normalized)
    ):
        return True
    return bool(
        _CLINICAL_STATE_RE.search(normalized)
        and not looks_like_cognitive_task_or_stimulus(normalized)
    )


def name_compatibility_score(left: Any, right: Any) -> float:
    """Return a conservative lexical identity score in [0, 1]."""

    left_normalized = normalize_entity_name(left)
    right_normalized = normalize_entity_name(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0

    left_tokens = entity_name_tokens(left_normalized)
    right_tokens = entity_name_tokens(right_normalized)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0

    # A concrete imaging-derived quantity is not identical to its anatomical
    # carrier.  For example, "temporal lobe atrophy" must not collapse into a
    # bare "Temporal lobe" node merely because the region words overlap.  It
    # may still reuse a canonical alias that names another concrete readout.
    left_concrete_imaging = looks_like_concrete_imaging_measurement(left_normalized)
    right_concrete_imaging = looks_like_concrete_imaging_measurement(right_normalized)
    if left_concrete_imaging != right_concrete_imaging:
        return 0.0

    shorter = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
    longer = right_tokens if shorter is left_tokens else left_tokens
    containment = len(overlap) / len(shorter)
    jaccard = len(overlap) / len(left_tokens | right_tokens)

    if shorter <= longer:
        if len(shorter) >= 2:
            return max(0.90, jaccard)
        token = next(iter(shorter))
        if len(token) >= 4 and token not in _GENERIC_SINGLE_TOKENS:
            return 0.80
    if len(overlap) >= 2 and containment >= 0.75:
        return max(0.75, jaccard)
    return max(jaccard, 0.5 * containment)


def _node_value(node: Any, field: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(field, default)
    return getattr(node, field, default)


def concept_atom_roles(node: Any) -> frozenset[Atom]:
    if node is None:
        return frozenset()
    metadata = _node_value(node, "metadata", {}) or {}
    declared = metadata.get("atom_types") or _node_value(node, "atom_types", []) or []
    roles: set[Atom] = set()
    for value in declared:
        try:
            roles.add(Atom(str(value).strip().lower()))
        except ValueError:
            continue
    for domain in _node_value(node, "domain_tags", []) or []:
        roles.update(DOMAIN_TO_ATOMS.get(str(domain), ()))
    return frozenset(roles)


def declared_type_atoms(declared_type: Any, name: Any = "") -> frozenset[Atom]:
    declared = re.sub(r"[^A-Z0-9]+", "_", str(declared_type or "").upper()).strip("_")
    roles: set[Atom] = set()

    if any(token in declared for token in (
        "IMAGING", "NEUROIMAGING", "CONNECTIVITY", "BRAIN_REGION", "NEURAL_CIRCUIT",
        "PET_MARKER", "ELECTROPHYSIOLOGY", "STRUCTURAL_MRI", "FUNCTIONAL_MRI",
        "TASK_FMRI_BIOMARKER", "FMRI", "BOLD", "DTI", "PET_IMAGING", "SPECT_IMAGING",
    )):
        roles.add(Atom.IMAGING_MARKER)
    if any(token in declared for token in (
        "GENE", "GENETIC", "GENOMIC", "POLYGENIC", "PATHWAY", "MOLECULAR_TARGET",
        "CELLULAR_PROCESS",
    )):
        roles.add(Atom.GENE_TARGET)
    if any(token in declared for token in (
        "DRUG", "MEDICATION", "PHARMACOLOGICAL", "THERAPEUTIC_AGENT",
    )):
        roles.add(Atom.DRUG)
    if any(token in declared for token in (
        "COGNITIVE_TASK", "EXPERIMENTAL_TASK", "PARADIGM", "STIMULUS", "MENTAL_STATE",
    )):
        roles.add(Atom.COGNITIVE_TASK)
    if any(token in declared for token in (
        "OUTCOME", "CLINICAL_EVENT", "SYMPTOM", "RATING_SCALE", "COGNITIVE_FUNCTION",
        "CLINICAL_PHENOTYPE", "TREATMENT_RESPONSE", "ADVERSE_EVENT",
    )):
        roles.add(Atom.OUTCOME)
    if any(token in declared for token in (
        "DISEASE", "DISORDER", "DIAGNOSIS", "CLINICAL_CONDITION",
    )):
        roles.add(Atom.DISEASE)
    if any(token in declared for token in (
        "INDIVIDUAL_DATA", "DEMOGRAPHIC", "BEHAVIORAL_TRAIT", "BEHAVIOURAL_TRAIT",
        "PERSONALITY", "LIFESTYLE", "AGE_VARIABLE",
    )):
        roles.add(Atom.INDIVIDUAL_DATA)

    # Extraction occasionally labels a measured cognitive phenotype as the task
    # that elicited it.  A score, deficit, symptom, or performance is a result,
    # not an experimental operation; retain both downstream phenotype roles.
    if Atom.COGNITIVE_TASK in roles and _NON_TASK_PHENOTYPE_RE.search(
        normalize_entity_name(name)
    ):
        roles.discard(Atom.COGNITIVE_TASK)
        roles.update((Atom.OUTCOME, Atom.INDIVIDUAL_DATA))

    if not roles:
        text = str(name or "")
        if looks_like_imaging_measurement(text):
            roles.add(Atom.IMAGING_MARKER)
        elif _GENETIC_NAME_RE.search(text):
            roles.add(Atom.GENE_TARGET)
        elif _TASK_NAME_RE.search(text):
            roles.add(Atom.COGNITIVE_TASK)
    return frozenset(roles)


def _claim_value(claim: Mapping[str, Any], field: str) -> Any:
    value = claim.get(field)
    if value not in (None, "", []):
        return value
    metadata = claim.get("metadata") or {}
    return metadata.get(field)


def _roles_compatible(declared: frozenset[Atom], canonical: frozenset[Atom]) -> bool:
    # Missing type evidence is unknown, not evidence of compatibility.
    if not declared or not canonical:
        return False
    if declared & canonical:
        return True
    clinical_pairs = {
        frozenset((Atom.DISEASE, Atom.OUTCOME)),
        frozenset((Atom.OUTCOME, Atom.INDIVIDUAL_DATA)),
    }
    return any(
        frozenset((left, right)) in clinical_pairs
        for left in declared
        for right in canonical
    )


@dataclass(frozen=True)
class ClaimEndpointAudit:
    valid: bool
    reason: str
    subject_name_score: float
    object_name_score: float
    subject_declared_atoms: tuple[str, ...]
    object_declared_atoms: tuple[str, ...]
    subject_canonical_atoms: tuple[str, ...]
    object_canonical_atoms: tuple[str, ...]


@dataclass(frozen=True)
class SemanticEndpoint:
    entity_id: str
    name: str
    canonical_id: str
    atoms: tuple[str, ...]
    uses_canonical_id: bool
    name_score: float
    role_compatible: bool


def audit_claim_endpoints(
    claim: Mapping[str, Any],
    concepts: Mapping[str, Any],
    *,
    minimum_name_score: float = 0.70,
) -> ClaimEndpointAudit:
    subject_id = str(_claim_value(claim, "subject_id") or "")
    object_id = str(_claim_value(claim, "object_id") or "")
    subject_name = str(_claim_value(claim, "subject_name") or "")
    object_name = str(_claim_value(claim, "object_name") or "")
    subject_node = concepts.get(subject_id)
    object_node = concepts.get(object_id)

    if not subject_id or not object_id or subject_node is None or object_node is None:
        return ClaimEndpointAudit(False, "missing_canonical_endpoint", 0.0, 0.0, (), (), (), ())

    def best_name_score(name: str, node: Any) -> float:
        candidates = [_node_value(node, "preferred_name", "")]
        candidates.extend(_node_value(node, "aliases", []) or [])
        return max((name_compatibility_score(name, value) for value in candidates), default=0.0)

    subject_name_score = best_name_score(subject_name, subject_node)
    object_name_score = best_name_score(object_name, object_node)
    subject_declared = declared_type_atoms(_claim_value(claim, "subject_type"), subject_name)
    object_declared = declared_type_atoms(_claim_value(claim, "object_type"), object_name)
    subject_canonical = concept_atom_roles(subject_node)
    object_canonical = concept_atom_roles(object_node)

    reason = "ok"
    if subject_name_score < minimum_name_score:
        reason = "subject_name_mismatch"
    elif object_name_score < minimum_name_score:
        reason = "object_name_mismatch"
    elif not subject_declared:
        reason = "subject_declared_type_unresolved"
    elif not object_declared:
        reason = "object_declared_type_unresolved"
    elif not subject_canonical:
        reason = "subject_canonical_type_unresolved"
    elif not object_canonical:
        reason = "object_canonical_type_unresolved"
    elif not _roles_compatible(subject_declared, subject_canonical):
        reason = "subject_atom_mismatch"
    elif not _roles_compatible(object_declared, object_canonical):
        reason = "object_atom_mismatch"

    values = lambda roles: tuple(sorted(role.value for role in roles))
    return ClaimEndpointAudit(
        reason == "ok",
        reason,
        subject_name_score,
        object_name_score,
        values(subject_declared),
        values(object_declared),
        values(subject_canonical),
        values(object_canonical),
    )


def semantic_claim_endpoint(
    claim: Mapping[str, Any],
    side: str,
    concepts: Mapping[str, Any],
    *,
    minimum_name_score: float = 0.70,
) -> SemanticEndpoint | None:
    """Project one claim endpoint to a safe canonical or claim-local identity."""

    if side not in {"subject", "object"}:
        raise ValueError("side must be 'subject' or 'object'")
    canonical_id = str(_claim_value(claim, f"{side}_id") or "")
    name = str(_claim_value(claim, f"{side}_name") or "").strip()
    normalized = normalize_entity_name(name)
    tokens = entity_name_tokens(name)
    if not canonical_id or not normalized or not tokens:
        return None
    if len(tokens) == 1 and next(iter(tokens)) in _GENERIC_SINGLE_TOKENS:
        return None

    node = concepts.get(canonical_id)
    declared = declared_type_atoms(_claim_value(claim, f"{side}_type"), name)
    canonical = concept_atom_roles(node)
    name_score = 0.0
    if node is not None:
        candidates = [_node_value(node, "preferred_name", "")]
        candidates.extend(_node_value(node, "aliases", []) or [])
        name_score = max(
            (name_compatibility_score(name, candidate) for candidate in candidates),
            default=0.0,
        )
    role_compatible = _roles_compatible(declared, canonical)
    use_canonical = (
        node is not None
        and name_score >= minimum_name_score
        and role_compatible
    )
    if use_canonical:
        entity_id = canonical_id
        roles = declared or canonical
    else:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]
        entity_id = f"CLAIM_ENTITY:{digest}"
        roles = declared
        if not roles and node is not None and name_score >= minimum_name_score:
            roles = canonical
    return SemanticEndpoint(
        entity_id=entity_id,
        name=name,
        canonical_id=canonical_id,
        atoms=tuple(sorted(role.value for role in roles)),
        uses_canonical_id=use_canonical,
        name_score=name_score,
        role_compatible=role_compatible,
    )


def semantic_claim_pair(
    claim: Mapping[str, Any],
    concepts: Mapping[str, Any],
) -> tuple[SemanticEndpoint, SemanticEndpoint] | None:
    subject = semantic_claim_endpoint(claim, "subject", concepts)
    obj = semantic_claim_endpoint(claim, "object", concepts)
    if subject is None or obj is None or subject.entity_id == obj.entity_id:
        return None
    return subject, obj


__all__ = [
    "ClaimEndpointAudit",
    "SemanticEndpoint",
    "SEMANTIC_ENDPOINT_IDENTITY_VERSION",
    "audit_claim_endpoints",
    "concept_atom_roles",
    "declared_type_atoms",
    "entity_name_tokens",
    "looks_like_concrete_imaging_measurement",
    "is_specific_functional_imaging_readout",
    "looks_like_cognitive_task_or_stimulus",
    "looks_like_functional_imaging_readout",
    "looks_like_imaging_measurement",
    "looks_like_method_or_procedure_entity",
    "looks_like_non_imaging_assay_entity",
    "looks_like_non_task_construct",
    "name_compatibility_score",
    "normalize_entity_name",
    "semantic_claim_endpoint",
    "semantic_claim_pair",
]


# Updated: 2026-08-12 13:49:00 HKT - separate concrete imaging readouts from bare modalities and method names.
# Updated: 2026-08-13 05:40:10 HKT - prevent broad single-token cortical aliases from absorbing specific imaging readouts.
# Updated: 2026-08-13 05:42:01 HKT - version semantic endpoint identity for cache invalidation.
# Updated: 2026-08-13 06:09:34 HKT - separate concrete imaging readouts from their bare anatomical carriers.
