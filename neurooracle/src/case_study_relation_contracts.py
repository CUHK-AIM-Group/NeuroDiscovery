"""Declarative relation contracts shared by generation and hindcasting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .atoms import Atom
from .claim_semantics import (
    is_specific_functional_imaging_readout,
    looks_like_cognitive_task_or_stimulus,
    looks_like_concrete_imaging_measurement,
    looks_like_imaging_measurement,
    looks_like_method_or_procedure_entity,
    looks_like_non_imaging_assay_entity,
    looks_like_non_task_construct,
)


ASSOCIATIVE = frozenset({
    "associated_with",
    "is_associated_with",
    "correlates_with",
    "links",
})
PREDICTIVE = frozenset({
    "distinguishes",
    "is_biomarker_of",
    "is_risk_factor_for",
    "predicts",
})
EFFECT = frozenset({
    "causes",
    "increases",
    "mediates",
    "modulates",
    "reduces",
})
GENETIC = frozenset({
    "gene_associated_with_anatomy",
    "gene_enriched_in_region",
    "receptor_density_in",
}) | ASSOCIATIVE | EFFECT | frozenset({"predicts"})
THERAPEUTIC = frozenset({"is_indicated_for", "is_treated_by", "treats"})
ADVERSE = frozenset({"has_adverse_effect", "causes", "increases", "predicts"}) | ASSOCIATIVE
LOCALIZATION = frozenset({
    "activates",
    "deactivates",
    "engages",
    "localizes_to",
    "maps_to",
    "recruits",
})
DECODING = frozenset({"decoded_from", "distinguishes", "predicts"}) | ASSOCIATIVE


@dataclass(frozen=True)
class PairRule:
    left: frozenset[Atom]
    right: frozenset[Atom]
    predicates: frozenset[str]
    directed: bool = False


def _rule(
    left: Atom | tuple[Atom, ...],
    right: Atom | tuple[Atom, ...],
    predicates: frozenset[str],
    *,
    directed: bool = False,
) -> PairRule:
    return PairRule(
        frozenset(left if isinstance(left, tuple) else (left,)),
        frozenset(right if isinstance(right, tuple) else (right,)),
        predicates,
        directed,
    )


_IM = Atom.IMAGING_MARKER
_D = Atom.DISEASE
_G = Atom.GENE_TARGET
_RX = Atom.DRUG
_O = Atom.OUTCOME
_IDV = Atom.INDIVIDUAL_DATA
_TASK = Atom.COGNITIVE_TASK


CASE_STUDY_PAIR_RULES: dict[str, tuple[PairRule, ...]] = {
    "case1_transdiagnostic": (_rule(_IM, _D, ASSOCIATIVE | PREDICTIVE | EFFECT),),
    "case2_pathway_mediation": (
        _rule(_G, _IM, GENETIC),
        _rule(_IM, _O, ASSOCIATIVE | PREDICTIVE | EFFECT),
    ),
    "biomarker_discovery": (_rule(_IM, _D, ASSOCIATIVE | PREDICTIVE),),
    "disease_subtyping": (
        _rule(_IM, _D, ASSOCIATIVE | PREDICTIVE),
        _rule(_IDV, _D, ASSOCIATIVE | PREDICTIVE),
        _rule(_IM, _IDV, ASSOCIATIVE | PREDICTIVE),
    ),
    "progression_prediction": (
        _rule(_IM, (_D, _O), PREDICTIVE | frozenset({"is_associated_with"})),
    ),
    "imaging_genetics": (_rule(_G, _IM, GENETIC),),
    "differential_diagnosis": (_rule(_IM, _D, frozenset({"distinguishes"})),),
    "drug_response_prediction": (
        _rule(_RX, _D, THERAPEUTIC | ASSOCIATIVE),
        _rule(_RX, _IM, ASSOCIATIVE | EFFECT),
        _rule(_RX, _O, PREDICTIVE | ASSOCIATIVE),
        _rule(_D, _IM, ASSOCIATIVE | PREDICTIVE),
        _rule(_D, _O, PREDICTIVE | ASSOCIATIVE),
        _rule(_IM, _O, PREDICTIVE | ASSOCIATIVE),
    ),
    "personalised_treatment": (
        _rule(_D, _RX, THERAPEUTIC | PREDICTIVE),
        _rule(_IM, _RX, PREDICTIVE | ASSOCIATIVE),
        _rule(_IDV, _RX, PREDICTIVE | ASSOCIATIVE),
        _rule(_D, _IM, ASSOCIATIVE | PREDICTIVE),
        _rule(_D, _IDV, ASSOCIATIVE),
        _rule(_IM, _IDV, ASSOCIATIVE | PREDICTIVE),
    ),
    "drug_repurposing": (_rule(_RX, _D, THERAPEUTIC),),
    "adverse_event_prediction": (_rule(_RX, _O, ADVERSE),),
    "neuromodulation_target": (
        _rule(_TASK, _IM, LOCALIZATION | ASSOCIATIVE | EFFECT),
        _rule(_D, _IM, ASSOCIATIVE | EFFECT | THERAPEUTIC),
        _rule(_D, _TASK, ASSOCIATIVE | EFFECT),
    ),
    "functional_localization": (_rule(_TASK, _IM, LOCALIZATION, directed=True),),
    "cognitive_decoding": (_rule(_IM, _TASK, DECODING),),
    "connectome_behavior": (
        _rule(_IM, (_IDV, _O), ASSOCIATIVE | frozenset({"predicts"})),
    ),
    "brain_age": (
        _rule(_IM, (_IDV, _O), ASSOCIATIVE | PREDICTIVE | frozenset({"increases", "reduces"})),
    ),
    "prognosis": (
        _rule(_D, _IM, ASSOCIATIVE | PREDICTIVE),
        _rule(_D, _O, ASSOCIATIVE | PREDICTIVE),
        _rule(_IM, _O, ASSOCIATIVE | PREDICTIVE),
    ),
}


# These same-role relations may support a two-paper bridge during generation,
# but they are not themselves a successful endpoint relation for hindcasting.
CASE_STUDY_SUPPORT_RULES: dict[str, tuple[PairRule, ...]] = {
    "imaging_genetics": (
        _rule(_G, _G, GENETIC),
        _rule(_IM, _IM, ASSOCIATIVE | EFFECT | PREDICTIVE),
    ),
    "adverse_event_prediction": (
        _rule(_RX, _RX, ASSOCIATIVE | EFFECT),
        _rule(_O, _O, ASSOCIATIVE | EFFECT | PREDICTIVE),
    ),
    "functional_localization": (
        _rule(_IM, _IM, LOCALIZATION | ASSOCIATIVE | EFFECT),
    ),
    "cognitive_decoding": (
        _rule(_IM, _IM, ASSOCIATIVE | EFFECT | PREDICTIVE),
        _rule(_TASK, _TASK, ASSOCIATIVE | PREDICTIVE),
    ),
}


COMPLETE_PATH_CASE_STUDIES = frozenset({
    "case2_pathway_mediation",
    "disease_subtyping",
    "drug_response_prediction",
    "neuromodulation_target",
    "personalised_treatment",
    "prognosis",
})


_NON_DRUG_INTERVENTION_RE = re.compile(
    r"\b(?:acupuncture|behavior(?:al)?\s+therapy|dbs|deep\s+brain\s+stimulation|"
    r"ect|electroconvulsive|exercise|implant|psychotherapy|resection|surgery|"
    r"tms|transcranial|training)\b",
    re.IGNORECASE,
)
_AGE_RE = re.compile(
    r"\b(?:ageing|aging|biological\s+age|brain\s+age|chronological\s+age|"
    r"predicted\s+age|age\s+gap|age\s+estimate)\b",
    re.IGNORECASE,
)
_NON_DRUG_ENTITY_RE = re.compile(
    r"\b(?:activit|brain|connectiv|cortex|density|disease|disorder|effect|"
    r"metaboli|outcome|patient|response|risk|symptom|volume|women?)\w*\b",
    re.IGNORECASE,
)
_DRUG_LIKE_RE = re.compile(
    r"\b(?:antidepressants?|antipsychotics?|benzodiazepines?|neuroleptics?|"
    r"ssris?|snris?|inhibitors?|agonists?|antagonists?|lithium)\b|"
    r"\b[a-z]+(?:antamine|antine|apine|caine|cycline|done|dopa|entin|idol|"
    r"idone|line|mab|mazepine|nib|olol|oxetine|pezil|prazole|pril|proate|"
    r"sartan|statin|stigmine|trigine|vir|zepam|zolam)\b",
    re.IGNORECASE,
)
_CONTROLLED_DRUG_ID_RE = re.compile(
    r"^(?:ATC|DRUGBANK|RXCUI|RXNORM):",
    re.IGNORECASE,
)
_VAGUE_DRUG_NAME_RE = re.compile(
    r"^(?:(?:commonly|widely)\s+)?(?:prescribed\s+)?(?:drug|drugs|medication|medications)$",
    re.IGNORECASE,
)
_GENETIC_DESCRIPTOR_RE = re.compile(
    r"\b(?:allele|cellular\s+process|gene|genetic|genotype|haplotype|methylation|"
    r"molecular\s+target|mutation|pathway|polymorphism|polygenic|protein|receptor|"
    r"signaling|snp|transcript|variant)\w*\b",
    re.IGNORECASE,
)
_GENE_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,11}\b")
_NON_GENETIC_ENTITY_RE = re.compile(
    r"\b(?:alzheimer|autis|bipolar|clinical|dementia|depress|disease|disorder|"
    r"mci|neurodegeneration|parkinson|patient|phenomena|psychosis|schizophren)\w*\b",
    re.IGNORECASE,
)
_CONTROLLED_GENETIC_ID_RE = re.compile(
    r"^(?:ENSEMBL|GENE|HGNC|KEGG|NCBIGENE|PATHWAY|REACTOME|UNIPROT):",
    re.IGNORECASE,
)
_CONTROLLED_DISEASE_ID_RE = re.compile(
    r"^(?:CUI|DOID|ICD(?:9|10)?|MESH|MONDO|MSH|OMIM|ORPHA):",
    re.IGNORECASE,
)
_NON_DISEASE_INPUT_RE = re.compile(
    r"^\s*(?:(?:clinical|cognitive|disease|rapid|symptom)\s+)*"
    r"(?:conversion|first\s+onset|future|incident|later|progression|"
    r"transition|with\s+age\s+progression)\b|"
    r"^\s*(?:an?\s+|the\s+)?(?:chance|likelihood|probability)\s+of\b|"
    r"\bprogression\s+(?:from|into|of|to|towards?)\b|"
    r"\b(?:clinical|disease|humanistic|symptom)\s+burden\b|"
    r"\b(?:dementia|disease|disorder|syndrome)\s+development\b|"
    r"\b(?:clinical|drug|efficacy|therapeutic|treatment)\s+"
    r"(?:outcome|response)\b|"
    r"\bresponse\s+to\s+(?:an?\s+)?(?:drug|medication|therapy|treatment)\b|"
    r"\balong\s+(?:an?|the)\s+continuum\b|"
    r"\bthan\b|"
    r"^\s*(?:an?\s+)?(?:chronic\s+|lifelong\s+)?"
    r"(?:mental|neurologic(?:al)?|neurodevelopmental|psychiatric)\s+"
    r"(?:condition|disease|disorder|syndrome)\s*$|"
    r"\b(?:biomarker\s+pathways?|converters?|exposure|prognos(?:is|tic)|"
    r"risk|siblings?|status|subgroups?\s+with|years?\s+later)\b",
    re.IGNORECASE,
)
_DISEASE_LIKE_RE = re.compile(
    r"\b(?:aciduria|adhd|alzheimer|astrocytoma|ataxia|autis\w*|bipolar|"
    r"cancer|dementia|degeneration|depress\w*|disease|disorder|dyslexia|"
    r"eclampsia|epilep\w*|glioma|impairment|injury|malaria|mci|"
    r"medulloblastoma|myopathy|neoplasm|parkinson|psychosis|schizophren\w*|"
    r"sclerosis|stroke|syndrome|thrombosis|tumou?r)\b",
    re.IGNORECASE,
)
_ADVERSE_OUTCOME_RE = re.compile(
    r"\b(?:adverse|akathisia|arrhythm|bleeding|complication|death|dyskinesia|"
    r"extrapyramidal|fracture|headache|hypomania|injury|mania|metabolic|"
    r"mortality|nausea|osteopor|sedation|seizure|side[ -]?effect|suicid|"
    r"toxic|toxicity|weight gain)\w*\b",
    re.IGNORECASE,
)
_DRUG_RESPONSE_OUTCOME_RE = re.compile(
    r"\b(?:benefit|change|delta|efficacy|improv|reduc|remission|respond|"
    r"response|score|symptom|treatment resistance)\w*\b",
    re.IGNORECASE,
)
_PROGNOSIS_OUTCOME_RE = re.compile(
    r"\b(?:change|conversion|convert|course|death|declin|dementia|disability|"
    r"follow[ -]?up|function|hospital|improv|mortality|outcome|progress|prognos|"
    r"recurr|recovery|reduc|relapse|remission|resilien|respond|response|severity|"
    r"survival|symptom|trajectory|treatment resistance|worsen)\w*\b",
    re.IGNORECASE,
)
_NON_PROGNOSIS_OUTCOME_RE = re.compile(
    r"\b(?:algorithm|biomarker|classifier|model|prediction|signature)\s+"
    r"(?:accuracy|performance|validation)\b",
    re.IGNORECASE,
)

_CONCRETE_IMAGING_ENDPOINT_CASES = frozenset({
    "biomarker_discovery",
    "brain_age",
    "case1_transdiagnostic",
    "transdiagnostic_clustering",
    "case2_pathway_mediation",
    "cognitive_decoding",
    "connectome_behavior",
    "differential_diagnosis",
    "disease_subtyping",
    "drug_response_prediction",
    "imaging_genetics",
    "personalised_treatment",
    "prognosis",
    "progression_prediction",
})


def _node_domains(endpoint: Any, concepts: Mapping[str, Any]) -> set[str]:
    node = concepts.get(str(getattr(endpoint, "canonical_id", "")))
    if node is None:
        return set()
    values = node.get("domain_tags", []) if isinstance(node, dict) else node.domain_tags
    return {str(value).strip().casefold().replace("-", "_") for value in values or []}


def endpoint_matches_atom(
    endpoint: Any,
    atom: Atom,
    concepts: Mapping[str, Any],
) -> bool:
    if atom.value not in set(getattr(endpoint, "atoms", ()) or ()):
        return False
    name = str(getattr(endpoint, "name", "") or "")
    domains = _node_domains(endpoint, concepts)
    if atom == Atom.DISEASE:
        # Claim-local atom labels occasionally type a prognosis endpoint as a
        # disease because the phrase contains the eventual diagnosis.  Such
        # phrases are valid outcomes, but not a baseline disease input.
        if _NON_DISEASE_INPUT_RE.search(name):
            return False
        canonical_id = str(getattr(endpoint, "canonical_id", "") or "")
        if (
            domains & {"clinical_condition", "disease"}
            and _CONTROLLED_DISEASE_ID_RE.match(canonical_id)
        ):
            return True
        return bool(_DISEASE_LIKE_RE.search(name))
    if atom == Atom.IMAGING_MARKER:
        if (
            looks_like_method_or_procedure_entity(name)
            or looks_like_non_imaging_assay_entity(name)
        ):
            return False
        return bool(
            domains & {"connectivity", "imaging_feature", "neuroanatomy"}
            or looks_like_imaging_measurement(name)
            or is_specific_functional_imaging_readout(name)
        )
    if atom == Atom.COGNITIVE_TASK:
        return bool(
            looks_like_cognitive_task_or_stimulus(name)
            and not looks_like_imaging_measurement(name)
            and not looks_like_non_task_construct(name)
        )
    if atom == Atom.DRUG:
        if _NON_DRUG_INTERVENTION_RE.search(name) or _NON_DRUG_ENTITY_RE.search(name):
            return False
        if _VAGUE_DRUG_NAME_RE.fullmatch(name.strip()):
            return False
        canonical_id = str(getattr(endpoint, "canonical_id", "") or "")
        if (
            "drug" in domains
            and _CONTROLLED_DRUG_ID_RE.match(canonical_id)
            and len(name.split()) <= 12
        ):
            return True
        return bool(_DRUG_LIKE_RE.search(name))
    if atom == Atom.GENE_TARGET:
        canonical_id = str(getattr(endpoint, "canonical_id", "") or "")
        if (
            domains & {"gene", "gene_pathway", "genetics", "molecular_target", "pathway"}
            and _CONTROLLED_GENETIC_ID_RE.match(canonical_id)
        ):
            return True
        if _GENETIC_DESCRIPTOR_RE.search(name):
            return True
        if _NON_GENETIC_ENTITY_RE.search(name):
            return False
        return bool(_GENE_SYMBOL_RE.search(name))
    return True


def _matches_any(
    endpoint: Any,
    atoms: frozenset[Atom],
    concepts: Mapping[str, Any],
) -> bool:
    return any(endpoint_matches_atom(endpoint, atom, concepts) for atom in atoms)


def case_study_endpoint_names_allowed(
    case_study_id: str | None,
    subject: Any,
    obj: Any,
) -> bool:
    """Apply case-level name constraints that are independent of predicates."""

    subject_atoms = set(getattr(subject, "atoms", ()) or ())
    object_atoms = set(getattr(obj, "atoms", ()) or ())

    def endpoint_for(atom: Atom) -> Any | None:
        if atom.value in object_atoms:
            return obj
        if atom.value in subject_atoms:
            return subject
        return None

    imaging = endpoint_for(Atom.IMAGING_MARKER)
    if imaging is not None:
        imaging_name = str(getattr(imaging, "name", "") or "")
        if (
            looks_like_method_or_procedure_entity(imaging_name)
            or looks_like_non_imaging_assay_entity(imaging_name)
        ):
            return False
        if (
            case_study_id in _CONCRETE_IMAGING_ENDPOINT_CASES
            and not looks_like_concrete_imaging_measurement(imaging_name)
        ):
            return False

    if case_study_id == "brain_age":
        imaging_endpoint = endpoint_for(Atom.IMAGING_MARKER)
        if imaging_endpoint is None:
            return False
        age_endpoint = obj if imaging_endpoint is subject else subject
        return bool(
            _AGE_RE.search(str(getattr(age_endpoint, "name", "") or ""))
        )
    outcome = endpoint_for(Atom.OUTCOME)
    if outcome is None:
        return True
    outcome_name = str(getattr(outcome, "name", "") or "")
    if looks_like_imaging_measurement(outcome_name):
        return False
    if case_study_id == "adverse_event_prediction":
        drug = endpoint_for(Atom.DRUG)
        if drug is not None and _ADVERSE_OUTCOME_RE.search(
            str(getattr(drug, "name", "") or "")
        ):
            return False
        return bool(_ADVERSE_OUTCOME_RE.search(outcome_name))
    if case_study_id == "drug_response_prediction":
        return bool(_DRUG_RESPONSE_OUTCOME_RE.search(outcome_name))
    if case_study_id == "prognosis":
        return bool(
            _PROGNOSIS_OUTCOME_RE.search(outcome_name)
            and not _NON_PROGNOSIS_OUTCOME_RE.search(outcome_name)
        )
    return True


def case_study_pair_allowed(
    claim: Mapping[str, Any],
    concepts: Mapping[str, Any],
    projected: tuple[Any, Any],
    case_study_id: str | None,
    *,
    include_support: bool = False,
) -> bool:
    if case_study_id is None:
        return True
    rules = CASE_STUDY_PAIR_RULES.get(case_study_id)
    if not rules:
        return False
    predicate = str(claim.get("predicate") or "").strip().casefold()
    subject, obj = projected
    def matches(rule: PairRule) -> bool:
        if predicate not in rule.predicates:
            return False
        forward = _matches_any(subject, rule.left, concepts) and _matches_any(
            obj, rule.right, concepts
        )
        reverse = (
            not rule.directed
            and _matches_any(subject, rule.right, concepts)
            and _matches_any(obj, rule.left, concepts)
        )
        return bool(forward or reverse)

    for rule in rules:
        if not matches(rule):
            continue
        if not case_study_endpoint_names_allowed(case_study_id, subject, obj):
            continue
        return True
    if include_support:
        return any(
            matches(rule)
            for rule in CASE_STUDY_SUPPORT_RULES.get(case_study_id, ())
        )
    return False


def case_study_endpoint_atom_routes(
    case_study_id: str,
    source_atoms: Iterable[Atom],
) -> tuple[tuple[Atom, Atom], ...]:
    """Return legal directed generation routes from the shared pair contract.

    A task registry entry names its primary output atom, while its literature
    contract may admit a closely related endpoint role. Progression prediction,
    for example, permits both DISEASE conversion and longitudinal OUTCOME
    endpoints. Generation must enumerate those same routes so valid future
    evidence is not structurally unreachable before ranking begins.
    """

    requested = frozenset(source_atoms)
    routes: set[tuple[Atom, Atom]] = set()
    for rule in CASE_STUDY_PAIR_RULES.get(case_study_id, ()):
        for source in requested & rule.left:
            routes.update((source, target) for target in rule.right)
        if not rule.directed:
            for source in requested & rule.right:
                routes.update((source, target) for target in rule.left)
    return tuple(sorted(routes, key=lambda route: (route[0].value, route[1].value)))


def requires_complete_path(case_study_id: str | None) -> bool:
    return bool(case_study_id in COMPLETE_PATH_CASE_STUDIES)


__all__ = [
    "CASE_STUDY_PAIR_RULES",
    "CASE_STUDY_SUPPORT_RULES",
    "COMPLETE_PATH_CASE_STUDIES",
    "case_study_endpoint_atom_routes",
    "case_study_endpoint_names_allowed",
    "case_study_pair_allowed",
    "endpoint_matches_atom",
    "requires_complete_path",
]


# Updated: 2026-08-12 16:02:00 HKT - share endpoint atom routes between generation and hindcasting contracts.
# Updated: 2026-08-12 21:37 HKT - require the non-imaging endpoint of a brain-age relation to encode age.
