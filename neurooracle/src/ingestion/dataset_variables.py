"""Phase 1 ingester: dataset variables (`individual_data` atom).

Conservative seeding (per user choice 2026-05-21): we import only the **Category /
Domain** layer of UKB / ADNI / HCP-YA, not the ~10k individual showcase fields.
This keeps the graph compact while still giving the `connectome_behavior`,
`brain_age`, `prognosis`, and `disease_subtyping` tasks concrete targets.

Three groups:

1. **UKB Showcase categories** — top-level Showcase categories filtered to
   neuro / mental / cognitive / lifestyle relevant ones (~30 nodes). Each
   carries the UKB Category ID in `external_ids` so we can deepen it later.

2. **ADNI data domains** (~15) — the high-level partition of ADNI tabular
   data (clinical, cognitive, MRI, PET, CSF biomarker, genetics, etc.).

3. **HCP-YA behavioral domains** (~12) — NIH Toolbox + HCP self-report domain
   groups (cognition, motor, sensory, emotion, personality, etc.).

All nodes carry domain tag `dataset_variable`. Categories that represent a
clinically actionable outcome (depression severity, cognitive decline, etc.)
also receive `treatment_outcome` so they double as endpoints for prognosis /
response tasks. Every node is linked to its parent dataset (DATASET:UKB,
DATASET:ADNI, DATASET:HCP_YA) via a `provides_modality` edge.

This is a curated knowledge module — it does not download anything.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)


# ── UKB Showcase categories (neuro / mental / cognitive / lifestyle subset) ───
#
# `ukb_id` references https://biobank.ndph.ox.ac.uk/showcase/cats.cgi
# `is_outcome` flags categories whose variables are commonly used as endpoints.

UKB_CATEGORIES: dict[str, dict] = {
    "UKB:CAT100026": {
        "name": "Brain MRI",
        "aliases": ["UKB Brain MRI", "imaging derived phenotypes"],
        "ukb_id": "100026",
        "description": "UKB structural / functional / diffusion MRI imaging-derived phenotypes.",
        "is_outcome": False,
    },
    "UKB:CAT110011": {
        "name": "Cognitive function",
        "aliases": ["UKB cognition", "cognitive tests"],
        "ukb_id": "110011",
        "description": "UKB cognitive task battery (RT, pairs matching, fluid intelligence, …).",
        "is_outcome": True,
    },
    "UKB:CAT100037": {
        "name": "Mental health",
        "aliases": ["UKB mental health questionnaire", "MHQ"],
        "ukb_id": "100037",
        "description": "UKB online mental-health questionnaire (depression, anxiety, mania, psychosis).",
        "is_outcome": True,
    },
    "UKB:CAT100038": {
        "name": "Mental wellbeing",
        "ukb_id": "100038",
        "description": "Self-reported wellbeing, life satisfaction, neuroticism subscales.",
        "is_outcome": True,
    },
    "UKB:CAT100040": {
        "name": "Sleep",
        "ukb_id": "100040",
        "description": "Sleep duration, chronotype, insomnia, daytime dozing.",
        "is_outcome": True,
    },
    "UKB:CAT100041": {
        "name": "Smoking",
        "ukb_id": "100041",
        "description": "Self-reported tobacco / smoking exposure variables.",
        "is_outcome": False,
    },
    "UKB:CAT100051": {
        "name": "Alcohol",
        "ukb_id": "100051",
        "description": "Alcohol intake frequency, type, and weekly volume.",
        "is_outcome": False,
    },
    "UKB:CAT100054": {
        "name": "Physical activity",
        "ukb_id": "100054",
        "description": "Self-reported and accelerometer-derived activity measures.",
        "is_outcome": False,
    },
    "UKB:CAT100070": {
        "name": "Diet",
        "ukb_id": "100070",
        "description": "Food-frequency intake variables.",
        "is_outcome": False,
    },
    "UKB:CAT17518": {
        "name": "Hand grip strength",
        "ukb_id": "17518",
        "description": "Dynamometer hand-grip strength — a robust ageing / sarcopenia marker.",
        "is_outcome": True,
    },
    "UKB:CAT100078": {
        "name": "Anthropometry",
        "ukb_id": "100078",
        "description": "Height, weight, BMI, waist-hip ratio.",
        "is_outcome": False,
    },
    "UKB:CAT100080": {
        "name": "Blood pressure",
        "ukb_id": "100080",
        "description": "Resting systolic / diastolic / pulse rate.",
        "is_outcome": False,
    },
    "UKB:CAT17518_HearingTest": {
        "name": "Hearing function",
        "ukb_id": "100049",
        "description": "Speech-in-noise digit triplet hearing test.",
        "is_outcome": True,
    },
    "UKB:CAT100013": {
        "name": "First occurrences (HES + GP)",
        "ukb_id": "1712",
        "description": "First occurrence of ICD-10 diagnoses across primary / secondary care — drives prognosis labels.",
        "is_outcome": True,
    },
    "UKB:CAT100086": {
        "name": "Pain",
        "ukb_id": "100048",
        "description": "Self-reported chronic / recent pain by body site.",
        "is_outcome": True,
    },
    "UKB:CAT100079": {
        "name": "Visual acuity",
        "ukb_id": "100013",
        "description": "Logarithmic-MAR acuity, autorefraction, intraocular pressure.",
        "is_outcome": True,
    },
    "UKB:CAT100090": {
        "name": "Family history of disease",
        "ukb_id": "100034",
        "description": "Reported parental / sibling history of major chronic diseases.",
        "is_outcome": False,
    },
    "UKB:CAT17819": {
        "name": "Genotyping arrays",
        "ukb_id": "263",
        "description": "Affymetrix UK BiLEVE / Axiom arrays — basis for PRS and GWAS variables.",
        "is_outcome": False,
    },
    "UKB:CAT17518_Education": {
        "name": "Education and occupation",
        "ukb_id": "100066",
        "description": "Years of education, job coding, qualifications.",
        "is_outcome": False,
    },
    "UKB:CAT_Demographics": {
        "name": "Sociodemographics",
        "ukb_id": "100011",
        "description": "Age, sex, ethnicity, marital status, household.",
        "is_outcome": False,
    },
    "UKB:CAT_Mortality": {
        "name": "Mortality and survival",
        "ukb_id": "100093",
        "description": "Date and cause of death — survival prognosis target.",
        "is_outcome": True,
    },
}


# ── ADNI data domains ─────────────────────────────────────────────────────────

ADNI_DOMAINS: dict[str, dict] = {
    "ADNI:DOM_DX":           {"name": "ADNI clinical diagnosis", "is_outcome": True,
                              "description": "Diagnostic group at each visit (CN, SMC, EMCI, LMCI, AD)."},
    "ADNI:DOM_NEUROPSYCH":   {"name": "ADNI neuropsychological battery", "is_outcome": True,
                              "description": "ADAS-Cog, MMSE, CDR-SB, FAQ, RAVLT, etc. tabular scores."},
    "ADNI:DOM_MRI":          {"name": "ADNI MRI imaging", "is_outcome": False,
                              "description": "Volumetric and cortical-thickness imaging-derived phenotypes."},
    "ADNI:DOM_PET_AMYLOID":  {"name": "ADNI amyloid PET", "is_outcome": False,
                              "description": "Florbetapir / florbetaben / Pittsburgh Compound B SUVR."},
    "ADNI:DOM_PET_TAU":      {"name": "ADNI tau PET", "is_outcome": False,
                              "description": "Flortaucipir SUVR by region."},
    "ADNI:DOM_PET_FDG":      {"name": "ADNI FDG-PET", "is_outcome": False,
                              "description": "Glucose-metabolism imaging by region."},
    "ADNI:DOM_CSF":          {"name": "ADNI CSF biomarkers", "is_outcome": False,
                              "description": "Aβ42, p-tau, t-tau, ratios; Roche Elecsys panel."},
    "ADNI:DOM_GENETICS":     {"name": "ADNI genetics", "is_outcome": False,
                              "description": "APOE genotype + GWAS arrays + WGS subset."},
    "ADNI:DOM_BIOMARKER_BLOOD": {"name": "ADNI blood biomarkers", "is_outcome": False,
                                 "description": "Plasma p-tau, NfL, GFAP."},
    "ADNI:DOM_DEMOG":        {"name": "ADNI demographics", "is_outcome": False,
                              "description": "Age, sex, education, race, marital status."},
    "ADNI:DOM_LIFESTYLE":    {"name": "ADNI lifestyle / medical history", "is_outcome": False,
                              "description": "Comorbidities, medications, family history."},
    "ADNI:DOM_PROGRESSION":  {"name": "ADNI conversion / progression", "is_outcome": True,
                              "description": "Time-to-conversion (CN→MCI, MCI→AD) and slope variables."},
    "ADNI:DOM_FUNCTIONAL":   {"name": "ADNI functional ability", "is_outcome": True,
                              "description": "FAQ, ADL, IADL self-report functional scores."},
    "ADNI:DOM_NEUROBEHAV":   {"name": "ADNI neurobehavioral", "is_outcome": True,
                              "description": "GDS, NPI, behavioral / mood ratings."},
}


# ── HCP-YA behavioral domains ─────────────────────────────────────────────────

HCP_DOMAINS: dict[str, dict] = {
    "HCP:DOM_COG_FLUID":    {"name": "HCP fluid cognition",
                             "description": "NIH Toolbox fluid composite (DCCS, flanker, list sort, picture sequence)."},
    "HCP:DOM_COG_CRYST":    {"name": "HCP crystallized cognition",
                             "description": "NIH Toolbox crystallized composite (picture vocab, oral reading)."},
    "HCP:DOM_COG_TOTAL":    {"name": "HCP total cognition",
                             "description": "NIH Toolbox total composite cognition score."},
    "HCP:DOM_MOTOR":        {"name": "HCP motor function",
                             "description": "Endurance (2-min walk), grip strength, dexterity (9-hole peg), gait."},
    "HCP:DOM_SENSORY":      {"name": "HCP sensory",
                             "description": "Vision (LogMAR), audition (words-in-noise), pain, taste, smell."},
    "HCP:DOM_EMOTION":      {"name": "HCP emotion",
                             "description": "Negative affect (anger, fear, sadness), positive affect, emotion recognition."},
    "HCP:DOM_PERSONALITY":  {"name": "HCP personality (NEO-FFI)",
                             "description": "Big-Five personality domains (NEO-FFI 60-item)."},
    "HCP:DOM_PSYCH":        {"name": "HCP psychiatric / life function",
                             "description": "ASR self-report, DSM-oriented scales, life satisfaction."},
    "HCP:DOM_SUBSTANCE":    {"name": "HCP substance use",
                             "description": "Self-reported alcohol, tobacco, drug-use history."},
    "HCP:DOM_SOCIAL":       {"name": "HCP social relationships",
                             "description": "PROMIS social satisfaction, friendship, hostility."},
    "HCP:DOM_DELAY":        {"name": "HCP delay discounting",
                             "description": "Two-magnitude delay-discounting AUC."},
    "HCP:DOM_DEMOG":        {"name": "HCP demographics",
                             "description": "Age, sex, education, family structure, SES proxies."},
}


def _add_var_node(
    kg: KnowledgeGraph,
    nid: str,
    info: dict,
    source_vocab: str,
    parent_dataset: str | None = None,
    extra_id_key: str | None = None,
    is_outcome: bool | None = None,
) -> bool:
    """Add a single dataset-variable node (idempotent). Returns True if newly added."""
    existed = kg.has_concept(nid)
    domains = [DomainTag.DATASET_VARIABLE.value]
    if is_outcome is None:
        is_outcome = bool(info.get("is_outcome"))
    if is_outcome:
        domains.append(DomainTag.TREATMENT_OUTCOME.value)
    external_ids: dict[str, str] = {}
    if extra_id_key and info.get(extra_id_key):
        external_ids[source_vocab] = info[extra_id_key]
    raw_name = info["name"]
    # Prefix the display name with the dataset tag so that path strings
    # disambiguate dataset_variable hubs from same-named disease/outcome
    # nodes (e.g. UKB CAT100086 'Pain' vs MeSH D010146 'Pain'). Skip when
    # the name already starts with the dataset acronym.
    prefix = nid.split(":", 1)[0] if ":" in nid else ""
    if prefix and not raw_name.lower().startswith(prefix.lower()):
        display_name = f"{prefix}: {raw_name}"
        aliases = list(info.get("aliases", [])) + [raw_name]
    else:
        display_name = raw_name
        aliases = list(info.get("aliases", []))
    kg.add_concept(ConceptNode(
        id=nid,
        preferred_name=display_name,
        aliases=aliases,
        domain_tags=domains,
        source_vocab=source_vocab,
        definition=info.get("description", ""),
        external_ids=external_ids,
        metadata={
            "parent_dataset": parent_dataset,
            "is_outcome":     is_outcome,
        },
    ))
    if parent_dataset and kg.has_concept(parent_dataset):
        # Dataset → variable: dataset "provides" this variable as a kind of modality.
        kg.add_edge(Edge(
            source_id=parent_dataset,
            target_id=nid,
            relation_type="provides_modality",
            source=source_vocab,
            confidence=1.0,
        ))
    return not existed


def ingest_dataset_variables(kg: KnowledgeGraph) -> dict:
    """Seed UKB / ADNI / HCP-YA category-level variables. Idempotent."""
    counts = {"ukb_added": 0, "adni_added": 0, "hcp_added": 0}

    for nid, info in UKB_CATEGORIES.items():
        if _add_var_node(kg, nid, info, "UKB-Showcase",
                         parent_dataset="DATASET:UKB", extra_id_key="ukb_id"):
            counts["ukb_added"] += 1

    for nid, info in ADNI_DOMAINS.items():
        if _add_var_node(kg, nid, info, "ADNI",
                         parent_dataset="DATASET:ADNI"):
            counts["adni_added"] += 1

    for nid, info in HCP_DOMAINS.items():
        if _add_var_node(kg, nid, info, "HCP",
                         parent_dataset="DATASET:HCP_YA"):
            counts["hcp_added"] += 1

    counts["total"] = sum(counts.values())
    logger.info(f"dataset variables ingestion complete: {counts}")
    return counts


__all__ = [
    "ingest_dataset_variables",
    "UKB_CATEGORIES",
    "ADNI_DOMAINS",
    "HCP_DOMAINS",
]
