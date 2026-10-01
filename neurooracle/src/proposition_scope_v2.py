"""Corrected scope, orientation and null semantics for shared propositions.

The first pilot adjudication layer treated claims as the same proposition when
they differed in a necessary condition. It merged "hippocampal volume is lower
in schizophrenia" with "hippocampal volume is lower in preclinical dementia",
and its prompt stated that different populations or species are never a
difference. Those merges are wrong: they inflate the multi-paper rate by erasing
the science.

These rules run *before* any model verdict. A model may further restrict a pair,
but it cannot promote a pair to ``equivalent`` that this gate blocks. Cohort or
sample identity, modality and adjustment are recorded as variants and do not by
themselves block equivalence, so genuine cross-cohort replication stays
reachable. Absent scope is never read as "same as the other paper".
"""
from __future__ import annotations

try:
    from .shared_proposition_registry import normalize_term, relation_family
except ImportError:  # scripts place ``neurooracle/src`` on sys.path
    from shared_proposition_registry import normalize_term, relation_family


SYMMETRIC_FAMILIES = frozenset({"association", "group_difference", "difference", "correlation"})

# Concept families let orientation tolerate the endpoint-wording drift that
# blocked almost all genuine replications ("hippocampal volume -> hippocampus"
# versus "post-traumatic stress disorder -> hippocampal volume"). Exact endpoint
# equality is tried first; this is a fallback, never a merge decision.
CONCEPT_ALIASES = {
    "hippocampus": ("hippocamp", "hippocampal", "subiculum", "ca1", "cornu ammonis"),
    "amygdala": ("amygdala", "amygdalar"),
    "entorhinal": ("entorhinal",),
    "frontal": ("frontal",),
    "temporal": ("temporal",),
    "cingulate": ("cingulate", "cingulum"),
    "white_matter": ("white matter", "fasciculus", "tract", "fornix", "corpus callosum"),
    "ventricle": ("ventricle", "ventricular"),
    "retina": ("retina", "retinal", "ganglion cell", "nerve fiber"),
    "thalamus": ("thalam",),
    "caudate": ("caudate",),
    "putamen": ("putamen", "lenticular", "striatum"),
    "brain": ("brain", "cortex", "cortical", "gray matter", "grey matter"),
    "volume": ("volume", "volumetry", "atrophy", "size", "volumetric"),
    "shape": ("shape", "deformation", "surface"),
    "metabolite": ("naa", "creatine", "mi/cr", "choline", "spectroscop"),
    "diffusion": ("fractional anisotropy", "diffusivity", "diffusion", "tractography"),
    "perfusion": ("blood flow", "perfusion", "arterial spin", "rcbf"),
    "memory": ("memory", "declarative", "episodic", "verbal", "visuospatial"),
    "cognition": ("cognit", "neuropsycholog", "adas", "mmse", "discriminat"),
    "amyloid": ("amyloid", "abeta", "a\u03b2", "pet", "suvr"),
    "tau": ("tau", "ptau", "ttau", "neurofibrillary"),
}

BLOCKING_SLOTS = (
    "species", "population", "disease_stage", "comparator", "measurement",
    "anatomy", "intervention", "dose", "task", "timepoint",
)

VARIANT_SLOTS = ("modality", "adjustment", "cohort", "sample", "dataset", "region")

DEFAULT_REQUIRED_SLOTS = ("population", "measurement")
REQUIRED_SLOTS_BY_FAMILY = {
    "group_difference": ("population", "comparator", "measurement"),
    "association": ("population", "measurement"),
}

# Slots whose values are reliable, closed vocabularies: an inequality is a real
# necessary-condition difference, so it hard-blocks equivalence.
HARD_IDENTITY_SLOTS = ("species", "disease_stage")

# Population/comparator differences are hard only when both sides name a
# recognisable condition and those conditions are disjoint (different disease).
# Wording differences otherwise need model justification, not a silent decision.
DISEASE_TERM_SLOTS = ("population", "comparator")

# Controlled disease/condition vocabulary. Only used to detect a *disjoint*
# condition, never to decide that two wordings are the same.
DISEASE_TERMS = {
    "schizophrenia": ("schizophrenia", "schizophrenic", "sz"),
    "schizoaffective": ("schizoaffective", "sad"),
    "bipolar": ("bipolar", "bd", "bdp"),
    "alzheimer": ("alzheimer", "dat"),
    "dementia": ("dementia", "pred", "preclinical"),
    "mci": ("mci", "mild cognitive impairment", "amci", "amnestic"),
    "ptsd": ("ptsd", "posttraumatic", "post-traumatic"),
    "epilepsy": ("epilepsy", "tle", "seizure", "epileptogenic"),
    "hiv": ("hiv", "pwh"),
    "depression": ("depress", "mdd"),
    "stroke": ("stroke", "mcao", "infarct", "poststroke", "vascular"),
    "parkinson": ("parkinson",),
    "anxiety": ("anxiety", "gad", "panic", "ocd"),
    "tumor": ("tumor", "tumour", "glioma", "radiotherapy", "radiation"),
    "heart_failure": ("heart failure", "cardiac"),
    "healthy": ("healthy", "control", "normal", "unimpaired", "nondemented", "non-demented"),
    "aging": ("aging", "ageing", "elderly", "older adult", "age-related", "super-ager", "superager"),
    "development": ("children", "adolescent", "pediatric", "paediatric", "infant", "young"),
    "genetic_risk": ("psen1", "presenilin", "mutation carrier", "mutation carriers", "e280a", "app mutation"),
}

# Age-band terms. "child" and "adult" are mutually exclusive developmental
# stages, so naming them on opposite sides is a necessary-condition difference.
AGE_TERMS = {
    "pediatric": ("child", "children", "pediatric", "paediatric", "adolescent", "infant", "neonat"),
    "adult": ("adult", "middle-aged", "middle aged", "elderly", "older", "aged", "senior"),
}

NULL_MARKERS = (
    "no significant", "not significant", "non-significant", "nonsignificant",
    "did not differ", "did not significantly", "no difference", "not differ",
    "comparable", "no association", "no relationship", "no correlation",
    "no evidence of", "did not predict", "failed to", "no change", "did not change",
)

VERDICT_HINTS = ("equivalent_candidate", "related", "distinct", "unresolved")


def safe_family(proposition):
    try:
        return relation_family((proposition or {}).get("relation"))
    except ValueError:
        return None


def scope_slots(proposition):
    """Merge ``qualifiers`` and the extraction ``scope`` block into one mapping.

    ``region`` is treated as an alias of ``anatomy`` because the extraction
    schema uses either slot for the same necessary condition.
    """
    merged = {}
    for source in ((proposition or {}).get("qualifiers") or {}, (proposition or {}).get("scope") or {}):
        if not isinstance(source, dict):
            continue
        for slot, value in source.items():
            if value is None or value == "" or value == [] or value == {}:
                continue
            if isinstance(value, (list, tuple, set)):
                normalized = tuple(sorted({normalize_term(item) for item in value} - {""}))
                if not normalized:
                    continue
            else:
                normalized = normalize_term(value)
                if not normalized:
                    continue
            merged.setdefault(slot, normalized)
    if "region" in merged and "anatomy" not in merged:
        merged["anatomy"] = merged["region"]
    return merged


def orientation_matches(proposition_a, proposition_b):
    """Return True/False, or None when the relation families already differ."""
    family_a, family_b = safe_family(proposition_a), safe_family(proposition_b)
    if family_a is None or family_a != family_b:
        return None
    subject_a, object_a = normalize_term(proposition_a.get("subject")), normalize_term(proposition_a.get("object"))
    subject_b, object_b = normalize_term(proposition_b.get("subject")), normalize_term(proposition_b.get("object"))
    direct = subject_a == subject_b and object_a == object_b
    swapped = subject_a == object_b and object_a == subject_b
    if family_a in SYMMETRIC_FAMILIES:
        return direct or swapped
    return direct


def concepts(value):
    if not value:
        return frozenset()
    text = normalize_term(value).replace("-", " ")
    return frozenset(label for label, aliases in CONCEPT_ALIASES.items()
                     if any(alias in text for alias in aliases))


def concept_orientation_matches(proposition_a, proposition_b):
    """Orientation tolerance for endpoint-wording drift; exact match still wins."""
    family_a, family_b = safe_family(proposition_a), safe_family(proposition_b)
    if family_a is None or family_a != family_b:
        return None
    subject_a, object_a = concepts(proposition_a.get("subject")), concepts(proposition_a.get("object"))
    subject_b, object_b = concepts(proposition_b.get("subject")), concepts(proposition_b.get("object"))
    if not subject_a or not object_a or not subject_b or not object_b:
        return False
    direct = bool(subject_a & subject_b) and bool(object_a & object_b)
    swapped = bool(subject_a & object_b) and bool(object_a & subject_b)
    return direct or (family_a in SYMMETRIC_FAMILIES and swapped)


def _values_of(value):
    return set(value) if isinstance(value, (tuple, list, set)) else {value}


def disease_terms(value):
    """Return every controlled condition mentioned in a free-text scope value."""
    if not isinstance(value, str):
        return frozenset()
    terms = set(_term_hits(value, DISEASE_TERMS))
    for label in _term_hits(value, AGE_TERMS):
        terms.add("age_" + label)
    return frozenset(terms)


def _term_hits(value, vocabulary):
    if not isinstance(value, str):
        return frozenset()
    text = " " + normalize_term(value).replace("-", " ") + " "
    found = set()
    for label, aliases in vocabulary.items():
        for alias in aliases:
            alias = alias.replace("-", " ")
            if " " in alias:
                if alias in text:
                    found.add(label)
            elif any(token == alias or (len(alias) >= 4 and token.startswith(alias))
                     for token in text.split()):
                found.add(label)
    return frozenset(found)


def disjoint_conditions(value_a, value_b):
    terms_a, terms_b = disease_terms(value_a), disease_terms(value_b)
    return bool(terms_a and terms_b and not (terms_a & terms_b))


# Conditions that name a disease or genetic-risk cohort rather than a generic
# healthy/aging reference. Such a condition present on exactly ONE side is a
# necessary-condition difference: the other paper studied a different population.
SPECIFIC_CONDITION_LABELS = frozenset(
    set(DISEASE_TERMS) - {"healthy", "aging", "development"}
) | {"age_pediatric"}


def one_sided_specific_condition(value_a, value_b):
    specific_a = disease_terms(value_a) & SPECIFIC_CONDITION_LABELS
    specific_b = disease_terms(value_b) & SPECIFIC_CONDITION_LABELS
    return bool(specific_a) != bool(specific_b)


def compare_scope(proposition_a, proposition_b):
    """Classify every shared slot as a hard conflict, a soft difference or a variant.

    Only lexical, closed-vocabulary differences are hard. Disease/population
    identity is hard when the wording is far apart; near-identical wording and
    all measurement/anatomy wording differences are soft, so the model must
    explicitly justify them instead of the gate silently deciding.
    """
    family = safe_family(proposition_a)
    slots_a, slots_b = scope_slots(proposition_a), scope_slots(proposition_b)
    hard_conflicts, soft_conflicts, unknown = {}, {}, {}
    for slot in BLOCKING_SLOTS:
        in_a, in_b = slot in slots_a, slot in slots_b
        if not in_a and not in_b:
            continue
        if not (in_a and in_b):
            unknown[slot] = [slots_a.get(slot), slots_b.get(slot)]
            continue
        value_a, value_b = slots_a[slot], slots_b[slot]
        if value_a == value_b:
            continue
        if isinstance(value_a, tuple) or isinstance(value_b, tuple):
            if set(value_a if isinstance(value_a, tuple) else (value_a,)) != set(
                    value_b if isinstance(value_b, tuple) else (value_b,)):
                soft_conflicts[slot] = [value_a, value_b]
            continue
        if slot in HARD_IDENTITY_SLOTS:
            hard_conflicts[slot] = [value_a, value_b]
        elif slot in DISEASE_TERM_SLOTS:
            if disjoint_conditions(value_a, value_b) or one_sided_specific_condition(value_a, value_b):
                hard_conflicts[slot] = [value_a, value_b]
            else:
                soft_conflicts[slot] = [value_a, value_b]
        else:
            soft_conflicts[slot] = [value_a, value_b]
    variants = {slot: [slots_a.get(slot), slots_b.get(slot)]
                for slot in VARIANT_SLOTS if slots_a.get(slot) != slots_b.get(slot)}
    required = REQUIRED_SLOTS_BY_FAMILY.get(family, DEFAULT_REQUIRED_SLOTS)
    missing_required = [slot for slot in required if slot not in slots_a or slot not in slots_b]
    return {
        "family": family,
        "same_family": family is not None and family == safe_family(proposition_b),
        "orientation_ok": orientation_matches(proposition_a, proposition_b),
        "hard_conflicts": hard_conflicts,
        "soft_conflicts": soft_conflicts,
        "conflicts": dict(hard_conflicts, **soft_conflicts),
        "unknown_scope": unknown,
        "variants": variants,
        "missing_required": missing_required,
    }


def evidence_stance(observation):
    """Classify what one observation actually reports.

    ``reported_null`` is separated from ``explicit_negation`` on purpose: a
    non-significant result is not the assertion that an effect is absent, and it
    is not automatically counterevidence for a different population's claim.
    """
    proposition = (observation or {}).get("proposition") or {}
    result = (observation or {}).get("result") or {}
    significance = result.get("significance")
    text = " ".join(
        [str((observation or {}).get("statement") or "")]
        + [str(quote) for quote in ((observation or {}).get("quotes") or [])]
    ).lower()
    if significance == "not_significant" or any(marker in text for marker in NULL_MARKERS):
        return "reported_null"
    if (proposition.get("polarity") or "asserted") == "negated":
        return "explicit_negation"
    if significance == "reported_significant":
        return "asserted_effect"
    return "unspecified"


def adjudication_gate(proposition_a, proposition_b, observation_a=None, observation_b=None):
    """Decide what the pair can become, independently of any model verdict."""
    comparison = compare_scope(proposition_a, proposition_b)
    if comparison["orientation_ok"] is False:
        comparison["orientation_ok"] = concept_orientation_matches(proposition_a, proposition_b)
    reasons = []
    stance_a = evidence_stance(observation_a)
    stance_b = evidence_stance(observation_b)
    if not comparison["same_family"]:
        hint, blocked = "distinct", True
        reasons.append("relation_family_differs")
    elif comparison["orientation_ok"] is False:
        hint, blocked = "distinct", True
        reasons.append("asymmetric_relation_orientation_differs")
    elif comparison["hard_conflicts"]:
        hint, blocked = "related", True
        reasons.extend("hard_scope_conflict:%s" % slot for slot in sorted(comparison["hard_conflicts"]))
    elif "reported_null" in {stance_a, stance_b} and "asserted_effect" in {stance_a, stance_b}:
        hint, blocked = "unresolved", True
        reasons.append("reported_null_versus_assertion_preserve_as_dispute")
    else:
        hint, blocked = "model_must_justify", False
        reasons.append("no_hard_conflict_found")
        reasons.extend("missing_scope:%s" % slot for slot in comparison["missing_required"])
        reasons.extend("scope_not_comparable:%s" % slot for slot in sorted(comparison["unknown_scope"]))
        reasons.extend("soft_difference:%s" % slot for slot in sorted(comparison["soft_conflicts"]))
    if comparison["variants"]:
        reasons.extend("variant:%s" % slot for slot in sorted(comparison["variants"]))
    return {
        "hint": hint,
        "hard_blocked": blocked,
        "model_justification_required": hint == "model_must_justify",
        "reasons": reasons,
        "stance_a": stance_a,
        "stance_b": stance_b,
        "conflicts": comparison["conflicts"],
        "hard_conflicts": comparison["hard_conflicts"],
        "soft_conflicts": comparison["soft_conflicts"],
        "unknown_scope": comparison["unknown_scope"],
        "missing_required": comparison["missing_required"],
        "variants": comparison["variants"],
    }


def counterevidence_permitted(proposition_a, proposition_b, observation_a, observation_b):
    """Only a matched scope plus an explicit negation is counterevidence."""
    comparison = compare_scope(proposition_a, proposition_b)
    if not comparison["same_family"] or comparison["orientation_ok"] is not True:
        return False
    if comparison["conflicts"] or comparison["missing_required"] or comparison["unknown_scope"]:
        return False
    return {evidence_stance(observation_a), evidence_stance(observation_b)} == {"asserted_effect", "explicit_negation"}
