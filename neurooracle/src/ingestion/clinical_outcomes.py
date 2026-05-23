"""Phase 1 ingester: clinical outcome variables (`treatment_outcome` atom).

Seeds the OUTCOME atom with two complementary node groups:

1. **Clinical rating scales** (~40) — quantitative outcome instruments used as
   trial endpoints in neurology / psychiatry / cognitive disorders. Examples:
   HAM-D, MADRS, MDS-UPDRS, ADAS-Cog, MMSE, MoCA, PANSS, BDI, NIHSS, EDSS, etc.
   Each is tagged with the disease family it primarily measures and the typical
   score direction (`lower_better` / `higher_better`).

2. **MedDRA System Organ Classes** (27) — the top-level WHO/ISO AE category
   tree (e.g. "Nervous system disorders", "Psychiatric disorders"). These give
   the `adverse_event_prediction` task a concrete OUTCOME endpoint pool.

All nodes carry domain tag `treatment_outcome` (and `dataset_variable` for
lineage) so they can serve as endpoints in tasks `drug_response_prediction`,
`adverse_event_prediction`, and `prognosis`.

This is a curated knowledge module — it does not download anything.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)


# ── Clinical rating scales ────────────────────────────────────────────────────

CLINICAL_RATING_SCALES: dict[str, dict] = {
    # Depression
    "OUTCOME:HAM-D": {
        "name": "Hamilton Depression Rating Scale",
        "aliases": ["HAM-D", "HDRS", "Hamilton Rating Scale for Depression"],
        "disease_family": "depression",
        "direction": "lower_better",
        "description": "Clinician-rated depression severity instrument; HAM-D-17 is the standard.",
    },
    "OUTCOME:MADRS": {
        "name": "Montgomery-Asberg Depression Rating Scale",
        "aliases": ["MADRS"],
        "disease_family": "depression",
        "direction": "lower_better",
        "description": "10-item clinician-rated depression scale; sensitive to change.",
    },
    "OUTCOME:BDI": {
        "name": "Beck Depression Inventory",
        "aliases": ["BDI", "BDI-II"],
        "disease_family": "depression",
        "direction": "lower_better",
        "description": "21-item self-report depression severity questionnaire.",
    },
    "OUTCOME:PHQ-9": {
        "name": "Patient Health Questionnaire-9",
        "aliases": ["PHQ-9", "PHQ9"],
        "disease_family": "depression",
        "direction": "lower_better",
        "description": "9-item self-report depression screening instrument.",
    },
    # Anxiety
    "OUTCOME:HAM-A": {
        "name": "Hamilton Anxiety Rating Scale",
        "aliases": ["HAM-A", "HARS"],
        "disease_family": "anxiety",
        "direction": "lower_better",
        "description": "14-item clinician-rated anxiety severity scale.",
    },
    "OUTCOME:GAD-7": {
        "name": "Generalized Anxiety Disorder 7-item",
        "aliases": ["GAD-7", "GAD7"],
        "disease_family": "anxiety",
        "direction": "lower_better",
        "description": "7-item self-report generalized-anxiety screening tool.",
    },
    # Schizophrenia / psychosis
    "OUTCOME:PANSS": {
        "name": "Positive and Negative Syndrome Scale",
        "aliases": ["PANSS"],
        "disease_family": "schizophrenia",
        "direction": "lower_better",
        "description": "30-item schizophrenia symptom severity scale (positive, negative, general).",
    },
    "OUTCOME:BPRS": {
        "name": "Brief Psychiatric Rating Scale",
        "aliases": ["BPRS"],
        "disease_family": "schizophrenia",
        "direction": "lower_better",
        "description": "18-item psychotic-symptom rating scale.",
    },
    "OUTCOME:SANS": {
        "name": "Scale for the Assessment of Negative Symptoms",
        "aliases": ["SANS"],
        "disease_family": "schizophrenia",
        "direction": "lower_better",
        "description": "Negative-symptom severity scale, paired with SAPS.",
    },
    "OUTCOME:CGI": {
        "name": "Clinical Global Impression",
        "aliases": ["CGI", "CGI-S", "CGI-I"],
        "disease_family": "psychiatric_general",
        "direction": "lower_better",
        "description": "Severity / improvement single-item clinician rating used across psychiatric trials.",
    },
    # Bipolar
    "OUTCOME:YMRS": {
        "name": "Young Mania Rating Scale",
        "aliases": ["YMRS"],
        "disease_family": "bipolar",
        "direction": "lower_better",
        "description": "11-item mania severity scale.",
    },
    # Cognition / dementia
    "OUTCOME:MMSE": {
        "name": "Mini-Mental State Examination",
        "aliases": ["MMSE"],
        "disease_family": "dementia",
        "direction": "higher_better",
        "description": "30-point cognitive screening instrument; widely used in AD trials.",
    },
    "OUTCOME:MoCA": {
        "name": "Montreal Cognitive Assessment",
        "aliases": ["MoCA"],
        "disease_family": "dementia",
        "direction": "higher_better",
        "description": "30-point MCI/dementia screening; more sensitive than MMSE in MCI.",
    },
    "OUTCOME:ADAS-Cog": {
        "name": "Alzheimer Disease Assessment Scale - Cognitive",
        "aliases": ["ADAS-Cog", "ADAS-Cog11", "ADAS-Cog13"],
        "disease_family": "dementia",
        "direction": "lower_better",
        "description": "Standard cognitive endpoint in AD pharmacological trials.",
    },
    "OUTCOME:CDR-SB": {
        "name": "Clinical Dementia Rating - Sum of Boxes",
        "aliases": ["CDR-SB", "CDR-SOB"],
        "disease_family": "dementia",
        "direction": "lower_better",
        "description": "Global dementia severity score; primary endpoint in many DMT trials.",
    },
    "OUTCOME:NPI": {
        "name": "Neuropsychiatric Inventory",
        "aliases": ["NPI"],
        "disease_family": "dementia",
        "direction": "lower_better",
        "description": "12-domain neuropsychiatric symptom severity scale for dementia.",
    },
    # Parkinson's disease
    "OUTCOME:MDS-UPDRS": {
        "name": "MDS Unified Parkinson Disease Rating Scale",
        "aliases": ["MDS-UPDRS", "UPDRS"],
        "disease_family": "parkinson",
        "direction": "lower_better",
        "description": "Standard motor + non-motor PD severity scale; primary endpoint in PD trials.",
    },
    "OUTCOME:Hoehn-Yahr": {
        "name": "Hoehn and Yahr Stage",
        "aliases": ["H&Y", "Hoehn-Yahr"],
        "disease_family": "parkinson",
        "direction": "lower_better",
        "description": "5-stage clinical PD severity classification.",
    },
    "OUTCOME:PDQ-39": {
        "name": "Parkinson Disease Questionnaire-39",
        "aliases": ["PDQ-39"],
        "disease_family": "parkinson",
        "direction": "lower_better",
        "description": "39-item PD-specific quality-of-life measure.",
    },
    # ALS / motor
    "OUTCOME:ALSFRS-R": {
        "name": "ALS Functional Rating Scale - Revised",
        "aliases": ["ALSFRS-R", "ALSFRS"],
        "disease_family": "als",
        "direction": "higher_better",
        "description": "12-item ALS function score; primary efficacy endpoint in ALS trials.",
    },
    # Huntington
    "OUTCOME:UHDRS": {
        "name": "Unified Huntington Disease Rating Scale",
        "aliases": ["UHDRS"],
        "disease_family": "huntington",
        "direction": "lower_better",
        "description": "Motor + cognitive + behavioral + functional Huntington scale.",
    },
    # Stroke
    "OUTCOME:NIHSS": {
        "name": "NIH Stroke Scale",
        "aliases": ["NIHSS"],
        "disease_family": "stroke",
        "direction": "lower_better",
        "description": "15-item neurological-deficit severity scale used in acute stroke.",
    },
    "OUTCOME:mRS": {
        "name": "modified Rankin Scale",
        "aliases": ["mRS", "Modified Rankin Scale"],
        "disease_family": "stroke",
        "direction": "lower_better",
        "description": "0-6 disability scale; standard 90-day stroke outcome.",
    },
    "OUTCOME:Barthel": {
        "name": "Barthel Index",
        "aliases": ["Barthel", "Barthel Index", "BI"],
        "disease_family": "stroke",
        "direction": "higher_better",
        "description": "10-item activities-of-daily-living independence index.",
    },
    # Multiple sclerosis
    "OUTCOME:EDSS": {
        "name": "Expanded Disability Status Scale",
        "aliases": ["EDSS"],
        "disease_family": "multiple_sclerosis",
        "direction": "lower_better",
        "description": "0-10 disability scale used as primary endpoint in MS trials.",
    },
    "OUTCOME:MSFC": {
        "name": "Multiple Sclerosis Functional Composite",
        "aliases": ["MSFC"],
        "disease_family": "multiple_sclerosis",
        "direction": "higher_better",
        "description": "Multivariate MS function metric (timed walk + 9HPT + PASAT).",
    },
    # Epilepsy
    "OUTCOME:SeizureFreq": {
        "name": "Seizure Frequency",
        "aliases": ["seizure frequency", "monthly seizure count", "seizure rate"],
        "disease_family": "epilepsy",
        "direction": "lower_better",
        "description": "Monthly / per-period seizure count; primary endpoint in epilepsy trials.",
    },
    "OUTCOME:QOLIE-31": {
        "name": "Quality of Life in Epilepsy-31",
        "aliases": ["QOLIE-31"],
        "disease_family": "epilepsy",
        "direction": "higher_better",
        "description": "31-item epilepsy-specific quality-of-life measure.",
    },
    # Autism / developmental
    "OUTCOME:ADOS": {
        "name": "Autism Diagnostic Observation Schedule",
        "aliases": ["ADOS", "ADOS-2"],
        "disease_family": "autism",
        "direction": "lower_better",
        "description": "Structured ASD assessment; calibrated severity score used as outcome.",
    },
    "OUTCOME:SRS": {
        "name": "Social Responsiveness Scale",
        "aliases": ["SRS", "SRS-2"],
        "disease_family": "autism",
        "direction": "lower_better",
        "description": "65-item social-communication impairment scale.",
    },
    # ADHD
    "OUTCOME:ADHD-RS": {
        "name": "ADHD Rating Scale",
        "aliases": ["ADHD-RS", "ADHD-RS-IV"],
        "disease_family": "adhd",
        "direction": "lower_better",
        "description": "18-item DSM-aligned ADHD symptom severity measure.",
    },
    "OUTCOME:Conners": {
        "name": "Conners Rating Scale",
        "aliases": ["Conners", "Conners-3"],
        "disease_family": "adhd",
        "direction": "lower_better",
        "description": "ADHD symptom + comorbidity rating scale (parent / teacher / self).",
    },
    # Sleep
    "OUTCOME:PSQI": {
        "name": "Pittsburgh Sleep Quality Index",
        "aliases": ["PSQI"],
        "disease_family": "sleep",
        "direction": "lower_better",
        "description": "19-item self-report sleep-quality index.",
    },
    "OUTCOME:ESS": {
        "name": "Epworth Sleepiness Scale",
        "aliases": ["ESS"],
        "disease_family": "sleep",
        "direction": "lower_better",
        "description": "8-item daytime-sleepiness self-report.",
    },
    # Pain
    "OUTCOME:VAS-Pain": {
        "name": "Visual Analog Scale - Pain",
        "aliases": ["VAS Pain", "Pain VAS"],
        "disease_family": "pain",
        "direction": "lower_better",
        "description": "0-10 cm visual-analog pain intensity rating.",
    },
    "OUTCOME:BPI": {
        "name": "Brief Pain Inventory",
        "aliases": ["BPI"],
        "disease_family": "pain",
        "direction": "lower_better",
        "description": "Pain severity + interference self-report scale.",
    },
    # General functional
    "OUTCOME:GAF": {
        "name": "Global Assessment of Functioning",
        "aliases": ["GAF"],
        "disease_family": "psychiatric_general",
        "direction": "higher_better",
        "description": "0-100 overall psychosocial functioning rating (DSM-IV legacy).",
    },
    "OUTCOME:WHODAS": {
        "name": "WHO Disability Assessment Schedule",
        "aliases": ["WHODAS", "WHODAS 2.0"],
        "disease_family": "general",
        "direction": "lower_better",
        "description": "12/36-item cross-disease functional disability instrument.",
    },
    "OUTCOME:SF-36": {
        "name": "Short Form 36",
        "aliases": ["SF-36"],
        "disease_family": "general",
        "direction": "higher_better",
        "description": "36-item generic health-related quality-of-life measure.",
    },
    # Generic responder / change endpoints
    "OUTCOME:Responder": {
        "name": "Treatment Responder Status",
        "aliases": ["responder", "response status", "responder vs non-responder"],
        "disease_family": "general",
        "direction": "categorical",
        "description": "Binary / categorical treatment-response label (≥50% improvement, remission, etc.).",
    },
    "OUTCOME:Remission": {
        "name": "Remission Status",
        "aliases": ["remission", "clinical remission"],
        "disease_family": "general",
        "direction": "categorical",
        "description": "Categorical remission endpoint (often per-scale threshold).",
    },
    "OUTCOME:RelapseFree": {
        "name": "Relapse-Free Survival",
        "aliases": ["relapse-free survival", "time to relapse"],
        "disease_family": "general",
        "direction": "higher_better",
        "description": "Time-to-event endpoint widely used in MS, mood, and addiction trials.",
    },
}


# ── MedDRA System Organ Classes (top-level AE categories) ─────────────────────

MEDDRA_SOC: dict[str, dict] = {
    "OUTCOME:AE_BloodLymph":    {"name": "Blood and lymphatic system disorders"},
    "OUTCOME:AE_Cardiac":       {"name": "Cardiac disorders"},
    "OUTCOME:AE_Congenital":    {"name": "Congenital, familial and genetic disorders"},
    "OUTCOME:AE_Ear":           {"name": "Ear and labyrinth disorders"},
    "OUTCOME:AE_Endocrine":     {"name": "Endocrine disorders"},
    "OUTCOME:AE_Eye":           {"name": "Eye disorders"},
    "OUTCOME:AE_GI":            {"name": "Gastrointestinal disorders"},
    "OUTCOME:AE_General":       {"name": "General disorders and administration site conditions"},
    "OUTCOME:AE_Hepatobiliary": {"name": "Hepatobiliary disorders"},
    "OUTCOME:AE_Immune":        {"name": "Immune system disorders"},
    "OUTCOME:AE_Infections":    {"name": "Infections and infestations"},
    "OUTCOME:AE_Injury":        {"name": "Injury, poisoning and procedural complications"},
    "OUTCOME:AE_Investigations": {"name": "Investigations"},
    "OUTCOME:AE_Metabolism":    {"name": "Metabolism and nutrition disorders"},
    "OUTCOME:AE_Musculoskeletal": {"name": "Musculoskeletal and connective tissue disorders"},
    "OUTCOME:AE_Neoplasms":     {"name": "Neoplasms benign, malignant and unspecified"},
    "OUTCOME:AE_Nervous":       {"name": "Nervous system disorders"},
    "OUTCOME:AE_Pregnancy":     {"name": "Pregnancy, puerperium and perinatal conditions"},
    "OUTCOME:AE_Product":       {"name": "Product issues"},
    "OUTCOME:AE_Psychiatric":   {"name": "Psychiatric disorders"},
    "OUTCOME:AE_Renal":         {"name": "Renal and urinary disorders"},
    "OUTCOME:AE_Reproductive":  {"name": "Reproductive system and breast disorders"},
    "OUTCOME:AE_Respiratory":   {"name": "Respiratory, thoracic and mediastinal disorders"},
    "OUTCOME:AE_Skin":          {"name": "Skin and subcutaneous tissue disorders"},
    "OUTCOME:AE_Social":        {"name": "Social circumstances"},
    "OUTCOME:AE_Surgical":      {"name": "Surgical and medical procedures"},
    "OUTCOME:AE_Vascular":      {"name": "Vascular disorders"},
}


def ingest_clinical_outcomes(kg: KnowledgeGraph) -> dict:
    """Seed clinical rating scales + MedDRA SOC into the KG.

    Idempotent — repeated runs only top up missing aliases / descriptions.
    """
    scales_added = 0
    soc_added = 0

    for nid, info in CLINICAL_RATING_SCALES.items():
        existed = kg.has_concept(nid)
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=info["name"],
            aliases=info.get("aliases", []),
            domain_tags=[
                DomainTag.TREATMENT_OUTCOME.value,
                DomainTag.DATASET_VARIABLE.value,
            ],
            source_vocab="ClinicalOutcomes",
            definition=info.get("description", ""),
            metadata={
                "disease_family": info.get("disease_family", "general"),
                "direction":      info.get("direction", "categorical"),
                "scale_kind":     "rating_scale",
            },
        ))
        if not existed:
            scales_added += 1

    for nid, info in MEDDRA_SOC.items():
        existed = kg.has_concept(nid)
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=info["name"],
            aliases=info.get("aliases", []),
            domain_tags=[
                DomainTag.TREATMENT_OUTCOME.value,
                DomainTag.DATASET_VARIABLE.value,
            ],
            source_vocab="MedDRA-SOC",
            definition=info.get("description", "WHO MedDRA System Organ Class (AE category)."),
            metadata={"scale_kind": "ae_soc"},
        ))
        if not existed:
            soc_added += 1

    summary = {
        "clinical_scales_added": scales_added,
        "meddra_soc_added": soc_added,
        "total": scales_added + soc_added,
    }
    logger.info(f"clinical outcomes ingestion complete: {summary}")
    return summary


__all__ = ["ingest_clinical_outcomes", "CLINICAL_RATING_SCALES", "MEDDRA_SOC"]
