"""Phase 1 bridges: connect outcome / individual_data / drug nodes into the rest of the KG.

The clinical-outcomes and dataset-variables ingesters seed nodes but leave them
isolated, which kills any task path that targets `treatment_outcome` /
`dataset_variable` (drug_response_prediction, adverse_event_prediction,
connectome_behavior, brain_age, prognosis, …). The drug atom has plenty of
out-edges (drug --treats--> disease) but almost no in-edges (no
disease --is_treated_by--> drug), which kills personalised_treatment.

This module wires:

1. **Rating scales → disease anchor** (`measures` edge). E.g. HAM-D measures
   major depressive disorder; MDS-UPDRS measures Parkinson; ADAS-Cog measures
   Alzheimer / Dementia. Anchor IDs are resolved via
   RATING_SCALE_DISEASE_ANCHORS, keyed by `disease_family` from
   CLINICAL_RATING_SCALES.

2. **Rating scales → host dataset variable** (`assessed_in` edge). E.g. MMSE
   / MoCA / ADAS-Cog → ADNI:DOM_NEUROPSYCH and HCP:DOM_COG_TOTAL; HAM-D / MADRS
   → UKB:CAT100037 (Mental health). This lets prognosis / connectome_behavior
   reach outcome scales through dataset-variable hubs.

3. **AE SOC → disease umbrella** (`affects_system` edge).

4. **UKB / ADNI / HCP categories → disease anchors** (`provides_signal_for`).

5. **Drug indication mirroring** (`is_treated_by` edge). For every existing
   `drug --treats--> disease`, mirror as `disease --is_treated_by--> drug`.
   Closes the personalised_treatment gap so {D,IM,Idv} → drug paths exist.

6. **Imaging dataset_variable ↔ IM anchor** (`measured_in_modality` /
   `modality_provides` edges). Imaging-flavored dataset_variable nodes
   (ADNI:DOM_PET_AMYLOID, UKB:CAT100026 Brain MRI, HCP:DOM_RFMRI, …) are
   linked bi-directionally to representative IM concepts so that
   brain_age / connectome_behavior / task_brain_behavior /
   drug_imaging_outcome can route IM ↔ Idv directly.

Idempotent — repeated runs only add edges that aren't already present.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import Edge

logger = logging.getLogger(__name__)


# disease_family → canonical disease node ID. None means "no good anchor; skip".
RATING_SCALE_DISEASE_ANCHORS: dict[str, str | None] = {
    "depression":           "COGAT_DISORDER:dso_1470",     # Major Depressive Disorder
    "anxiety":              "MSH:D001008",                  # Anxiety Disorders
    "schizophrenia":        "COGAT_DISORDER:dso_5419",     # Schizophrenia
    "bipolar":              "MSH:D001714",                  # Bipolar Disorder
    "dementia":             "MSH:D003704",                  # Dementia
    "parkinson":            "MSH:D010300",                  # Parkinson Disease
    "huntington":           "MSH:D006816",                  # Huntington Disease
    "als":                  "MSH:D000690",                  # ALS
    "multiple_sclerosis":   "MSH:D009103",                  # Multiple Sclerosis
    "epilepsy":             "MSH:D012640",                  # Seizures (root epilepsy in KG)
    "autism":               "COGAT_DISORDER:dso_0060041",  # ASD
    "adhd":                 "MSH:D001289",                  # ADHD
    "stroke":               "MSH:D020521",                  # Stroke
    "sleep":                "MSH:D012893",                  # Sleep Wake Disorders
    "pain":                 "MSH:D010146",                  # Pain (tagged cognitive_function but bridges OK)
    "psychiatric_general":  "MSH:D001523",                  # Mental Disorders
    "general":              "MSH:D001523",                  # Mental Disorders (catch-all)
}


# scale ID → list of dataset-variable hosts where this scale is administered.
RATING_SCALE_DATASET_HOSTS: dict[str, list[str]] = {
    # Cognitive / dementia scales — used in ADNI neuropsych battery + HCP cognition
    "OUTCOME:MMSE":       ["ADNI:DOM_NEUROPSYCH", "HCP:DOM_COG_TOTAL"],
    "OUTCOME:MoCA":       ["ADNI:DOM_NEUROPSYCH"],
    "OUTCOME:ADAS-Cog":   ["ADNI:DOM_NEUROPSYCH"],
    "OUTCOME:CDR-SB":     ["ADNI:DOM_NEUROPSYCH", "ADNI:DOM_DX"],
    "OUTCOME:NPI":        ["ADNI:DOM_NEUROBEHAV"],
    # Mood / psychiatric scales — UKB MHQ + ADNI neurobehav
    "OUTCOME:HAM-D":      ["UKB:CAT100037"],
    "OUTCOME:MADRS":      ["UKB:CAT100037"],
    "OUTCOME:BDI":        ["UKB:CAT100037", "ADNI:DOM_NEUROBEHAV"],
    "OUTCOME:PHQ-9":      ["UKB:CAT100037", "HCP:DOM_PSYCH"],
    "OUTCOME:HAM-A":      ["UKB:CAT100037"],
    "OUTCOME:GAD-7":      ["UKB:CAT100037", "HCP:DOM_PSYCH"],
    "OUTCOME:PANSS":      ["UKB:CAT100037"],
    "OUTCOME:BPRS":       ["UKB:CAT100037"],
    "OUTCOME:SANS":       ["UKB:CAT100037"],
    "OUTCOME:CGI":        ["UKB:CAT100037"],
    "OUTCOME:YMRS":       ["UKB:CAT100037"],
    # Functional ability — ADNI functional, HCP motor
    "OUTCOME:ALSFRS-R":   ["ADNI:DOM_FUNCTIONAL"],
    # Movement / PD
    "OUTCOME:MDS-UPDRS":  ["ADNI:DOM_FUNCTIONAL"],
    "OUTCOME:Hoehn-Yahr": ["ADNI:DOM_DX"],
    "OUTCOME:PDQ-39":     ["ADNI:DOM_FUNCTIONAL"],
    # Sleep
    "OUTCOME:PSQI":       ["UKB:CAT100040", "HCP:DOM_PSYCH"],
    "OUTCOME:ESS":        ["UKB:CAT100040"],
    # Pain
    "OUTCOME:VAS-Pain":   ["UKB:CAT100086", "HCP:DOM_SENSORY"],
    "OUTCOME:BPI":        ["UKB:CAT100086"],
    # Generic functional
    "OUTCOME:GAF":        ["UKB:CAT100037"],
    "OUTCOME:WHODAS":     ["UKB:CAT100037", "ADNI:DOM_FUNCTIONAL"],
    "OUTCOME:SF-36":      ["UKB:CAT100037"],
    # Neuropsych — also live in HCP fluid / total composites
    "OUTCOME:Conners":    ["HCP:DOM_PSYCH"],
    "OUTCOME:ADHD-RS":    ["HCP:DOM_PSYCH"],
    # Stroke functional
    "OUTCOME:NIHSS":      ["ADNI:DOM_FUNCTIONAL"],
    "OUTCOME:mRS":        ["ADNI:DOM_FUNCTIONAL"],
    "OUTCOME:Barthel":    ["ADNI:DOM_FUNCTIONAL"],
    # MS
    "OUTCOME:EDSS":       ["ADNI:DOM_FUNCTIONAL"],
    "OUTCOME:MSFC":       ["ADNI:DOM_FUNCTIONAL"],
    # Generic responder / remission flow into UKB MHQ + ADNI DX
    "OUTCOME:Responder":  ["UKB:CAT100037", "ADNI:DOM_DX"],
    "OUTCOME:Remission":  ["UKB:CAT100037", "ADNI:DOM_DX"],
    "OUTCOME:RelapseFree":["ADNI:DOM_DX"],
}


# AE SOC → disease umbrella (a generic anchor so adverse_event_prediction can
# traverse drug → ... → SOC). "Mental Disorders" is a hub with degree ~2900;
# "Nervous System Diseases" likewise. Mostly we attach to one disease umbrella.
SOC_DISEASE_UMBRELLA: dict[str, str] = {
    "OUTCOME:AE_Nervous":     "MSH:D001523",   # Mental Disorders (umbrella)
    "OUTCOME:AE_Psychiatric": "MSH:D001523",
}


# AE SOC → host dataset where this category of adverse event is recordable.
# Strong matches (ADNI/PPMI) have MedDRA-coded AE forms in the trial CRF.
# UKB matches are surrogate — derived from HES ICD-10 / GP records via the
# standard ICD→MedDRA mapping. AE_Reproductive intentionally absent: not
# reliably captured in any public neuroscience cohort.
AE_SOC_DATASET_HOSTS: dict[str, list[str]] = {
    # Strong: trial-style MedDRA AE forms
    "OUTCOME:AE_Cardiac":          ["ADNI:DOM_DX", "UKB:CAT100013"],
    "OUTCOME:AE_Hepatobiliary":    ["ADNI:DOM_BIOMARKER_BLOOD", "UKB:CAT100013"],
    "OUTCOME:AE_Metabolism":       ["ADNI:DOM_BIOMARKER_BLOOD", "UKB:CAT100013"],
    "OUTCOME:AE_Nervous":          ["ADNI:DOM_DX", "UKB:CAT100013"],
    "OUTCOME:AE_Psychiatric":      ["UKB:CAT100037", "ADNI:DOM_NEUROBEHAV"],
    "OUTCOME:AE_Gastrointestinal": ["ADNI:DOM_DX", "UKB:CAT100013"],
    "OUTCOME:AE_Skin":             ["UKB:CAT100013"],
    # AE_Reproductive — no reliable public-cohort coverage; deliberately omitted
}


# Dataset categories → modality / disease anchors. Edges go *into* the dataset
# variable from the anchor (anchor --provides_signal_for--> dataset_variable),
# so paths like `disease --...--> dataset_variable` exist.
DATASET_VAR_DISEASE_ANCHORS: dict[str, list[str]] = {
    "ADNI:DOM_DX":               ["MSH:D000544", "MSH:D003704"],          # Alzheimer, Dementia
    "ADNI:DOM_NEUROPSYCH":       ["MSH:D000544", "MSH:D003704"],
    "ADNI:DOM_PET_AMYLOID":      ["MSH:D000544"],
    "ADNI:DOM_PET_TAU":          ["MSH:D000544"],
    "ADNI:DOM_PET_FDG":          ["MSH:D000544", "MSH:D003704"],
    "ADNI:DOM_CSF":              ["MSH:D000544"],
    "ADNI:DOM_BIOMARKER_BLOOD":  ["MSH:D000544"],
    "ADNI:DOM_PROGRESSION":      ["MSH:D000544", "MSH:D003704"],
    "ADNI:DOM_FUNCTIONAL":       ["MSH:D003704"],
    "ADNI:DOM_NEUROBEHAV":       ["MSH:D003704", "COGAT_DISORDER:dso_1470"],
    "UKB:CAT110011":             ["MSH:D003704"],                          # Cognitive function ↔ Dementia
    "UKB:CAT100037":             ["MSH:D001523", "COGAT_DISORDER:dso_1470"],  # Mental health ↔ MDD / Mental Disorders
    "UKB:CAT100038":             ["MSH:D001523"],
    "UKB:CAT100040":             ["MSH:D012893"],                          # Sleep
    "UKB:CAT100086":             ["MSH:D010146"],                          # Pain
    "UKB:CAT100013":             ["MSH:D001523"],                          # First occurrences -> mental disorders umbrella
    "UKB:CAT_Mortality":         ["MSH:D001523"],
    "HCP:DOM_PSYCH":             ["MSH:D001523"],
    "HCP:DOM_SUBSTANCE":         ["MSH:D019966"],                          # Substance-Related Disorders
}


def _add_edge_if_new(
    kg: KnowledgeGraph,
    source_id: str,
    target_id: str,
    relation_type: str,
    source: str,
    confidence: float = 0.9,
) -> bool:
    if source_id == target_id:
        return False
    if not (kg.has_concept(source_id) and kg.has_concept(target_id)):
        return False
    if kg.G.has_edge(source_id, target_id):
        # Update relation only if missing
        existing = kg.G[source_id][target_id]
        if existing.get("relation_type") == relation_type:
            return False
    kg.add_edge(Edge(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        source=source,
        confidence=confidence,
    ))
    return True


def _bridge_rating_scales_to_diseases(kg: KnowledgeGraph) -> int:
    """Link each rating scale to its disease anchor via `measures`."""
    from .clinical_outcomes import CLINICAL_RATING_SCALES

    added = 0
    skipped_no_anchor = 0
    for scale_id, info in CLINICAL_RATING_SCALES.items():
        if not kg.has_concept(scale_id):
            continue
        family = info.get("disease_family", "general")
        anchor = RATING_SCALE_DISEASE_ANCHORS.get(family)
        if anchor is None:
            skipped_no_anchor += 1
            continue
        if _add_edge_if_new(kg, scale_id, anchor, "measures", "ClinicalOutcomes-Bridge"):
            added += 1
    if skipped_no_anchor:
        logger.info(f"  scales without disease anchor: {skipped_no_anchor}")
    return added


def _bridge_rating_scales_to_dataset_vars(kg: KnowledgeGraph) -> int:
    """Link rating scales to host dataset variables via `assessed_in`."""
    added = 0
    for scale_id, host_ids in RATING_SCALE_DATASET_HOSTS.items():
        if not kg.has_concept(scale_id):
            continue
        for host_id in host_ids:
            if _add_edge_if_new(kg, scale_id, host_id, "assessed_in", "ClinicalOutcomes-Bridge"):
                added += 1
    return added


def _bridge_soc_to_disease_umbrella(kg: KnowledgeGraph) -> int:
    """Link MedDRA SOC nodes to a disease umbrella via `affects_system`."""
    from .clinical_outcomes import MEDDRA_SOC

    added = 0
    default_umbrella = "MSH:D001523"  # Mental Disorders
    for soc_id in MEDDRA_SOC:
        if not kg.has_concept(soc_id):
            continue
        umbrella = SOC_DISEASE_UMBRELLA.get(soc_id, default_umbrella)
        if _add_edge_if_new(kg, soc_id, umbrella, "affects_system", "MedDRA-Bridge"):
            added += 1
    return added


def _bridge_ae_soc_to_dataset_vars(kg: KnowledgeGraph) -> int:
    """Link AE SOC nodes to host dataset variables via `assessed_in`.

    Mirrors the rating-scale → host-variable bridge so adverse_event_prediction
    paths can land on a concrete dataset (ADNI AE form, UKB HES ICD-derived AE,
    PPMI AE form via PPMI dataset hub if seeded).
    """
    added = 0
    for soc_id, host_ids in AE_SOC_DATASET_HOSTS.items():
        if not kg.has_concept(soc_id):
            continue
        for host_id in host_ids:
            if _add_edge_if_new(kg, soc_id, host_id, "assessed_in", "MedDRA-Bridge"):
                added += 1
    return added


def _bridge_dataset_vars_to_disease(kg: KnowledgeGraph) -> int:
    """Link dataset variable nodes to their disease anchors.

    Direction: disease --provides_signal_for--> dataset_variable.
    Reason: prognosis / connectome_behavior tasks search disease -> dataset_variable.
    """
    added = 0
    for var_id, disease_ids in DATASET_VAR_DISEASE_ANCHORS.items():
        if not kg.has_concept(var_id):
            continue
        for d_id in disease_ids:
            if _add_edge_if_new(kg, d_id, var_id, "provides_signal_for", "DatasetVar-Bridge"):
                added += 1
    return added


def _bridge_drug_indications(kg: KnowledgeGraph) -> int:
    """Mirror clean `drug --treats--> disease` edges as `disease --is_treated_by--> drug`.

    Most KG `treats` edges are oriented drug→disease (medical-writing convention),
    so disease nodes have low drug indegree, which kills personalised_treatment
    paths {D, IM, Idv} → drug. Mirroring closes the gap without introducing new
    semantic claims — `is_treated_by` is purely the inverse of `treats`.

    Atom filter: only mirror when source carries the DRUG domain tag and target
    carries the DISEASE domain tag. Raw `treats` edges are noisy (NeuroAnatomy
    nodes appear as treats-source for many CLM concepts), so blind mirroring
    pollutes disease out-edges with brain regions and CLM helpers. The
    {drug → disease} subset is the only direction that justifies a clean
    `disease --is_treated_by--> drug` reverse.
    """
    added = 0
    treats_pairs = [
        (src, tgt)
        for src, tgt, data in kg.G.edges(data=True)
        if data.get("relation_type") == "treats"
    ]
    for src, tgt in treats_pairs:
        src_node = kg._index.get(src)
        tgt_node = kg._index.get(tgt)
        if src_node is None or tgt_node is None:
            continue
        if "drug" not in src_node.domain_tags:
            continue
        if "disease" not in tgt_node.domain_tags:
            continue
        if _add_edge_if_new(kg, tgt, src, "is_treated_by", "DrugIndication-Mirror"):
            added += 1
    return added


# Imaging-flavored dataset_variable nodes → IM anchor concepts. Edges go
# bi-directionally (anchor --measured_in_modality--> variable AND
# variable --modality_provides--> anchor) so brain_age / connectome_behavior /
# task_brain_behavior / drug_imaging_outcome can route IM ↔ Idv either way.
#
# Anchor IDs are the modality-level CLM_CONCEPT nodes the KG actually carries
# (verified against full_snapshot_v1). Endpoints that aren't present are
# silently skipped by `_add_edge_if_new`, so the dict can over-list candidates.
IMAGING_DATASET_VAR_TO_IM_ANCHORS: dict[str, list[str]] = {
    # ADNI imaging domains
    "ADNI:DOM_MRI": [
        "CLM_CONCEPT:neuroimaging",
        "CLM_CONCEPT:magnetic_resonance_imaging",
    ],
    "ADNI:DOM_PET_AMYLOID": [
        "CLM_CONCEPT:neuroimaging",
        "CLM_CONCEPT:Positron_Emission_Tomography",
    ],
    "ADNI:DOM_PET_TAU": [
        "CLM_CONCEPT:neuroimaging",
        "CLM_CONCEPT:Positron_Emission_Tomography",
    ],
    "ADNI:DOM_PET_FDG": [
        "CLM_CONCEPT:neuroimaging",
        "CLM_CONCEPT:Positron_Emission_Tomography",
    ],
    # UKB imaging categories
    "UKB:CAT100026": [
        "CLM_CONCEPT:neuroimaging",
        "CLM_CONCEPT:magnetic_resonance_imaging",
    ],
}


def _bridge_imaging_dataset_vars_to_im(kg: KnowledgeGraph) -> int:
    """Wire imaging dataset_variable hubs to IM modality anchors, both directions.

    A: imaging anchor --measured_in_modality--> dataset_variable
    B: dataset_variable --modality_provides--> imaging anchor

    Closes the IM↔Idv adjacency gap. Endpoints that aren't in the KG are
    silently skipped (idempotent + safe to over-list candidates).
    """
    added = 0
    for var_id, anchor_ids in IMAGING_DATASET_VAR_TO_IM_ANCHORS.items():
        if not kg.has_concept(var_id):
            continue
        for a_id in anchor_ids:
            if _add_edge_if_new(kg, a_id, var_id, "measured_in_modality", "ImagingModality-Bridge"):
                added += 1
            if _add_edge_if_new(kg, var_id, a_id, "modality_provides", "ImagingModality-Bridge"):
                added += 1
    return added


def ingest_outcome_bridges(kg: KnowledgeGraph) -> dict:
    """Wire outcome / dataset-variable nodes into the rest of the KG. Idempotent."""
    counts = {
        "scale_disease_edges":       _bridge_rating_scales_to_diseases(kg),
        "scale_host_edges":          _bridge_rating_scales_to_dataset_vars(kg),
        "soc_umbrella_edges":        _bridge_soc_to_disease_umbrella(kg),
        "ae_soc_host_edges":         _bridge_ae_soc_to_dataset_vars(kg),
        "dataset_var_disease_edges": _bridge_dataset_vars_to_disease(kg),
        "drug_indication_edges":     _bridge_drug_indications(kg),
        "imaging_modality_edges":    _bridge_imaging_dataset_vars_to_im(kg),
    }
    counts["total"] = sum(counts.values())
    logger.info(f"outcome bridges complete: {counts}")
    return counts


__all__ = ["ingest_outcome_bridges",
           "RATING_SCALE_DISEASE_ANCHORS", "RATING_SCALE_DATASET_HOSTS",
           "SOC_DISEASE_UMBRELLA", "AE_SOC_DATASET_HOSTS",
           "DATASET_VAR_DISEASE_ANCHORS",
           "IMAGING_DATASET_VAR_TO_IM_ANCHORS"]
