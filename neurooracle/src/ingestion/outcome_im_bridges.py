"""Phase 1.7 — bridges that give OUTCOME nodes incoming edges.

The :mod:`outcome_bridges` module wires rating scales OUT of themselves
(scale → disease, scale → dataset_variable host) but leaves the
``OUTCOME:*`` family with **zero incoming edges**. As a result the chain
``D → IM → O`` is unreachable: there is nothing to land on at the
outcome side.

This module closes the gap with three bridge tables:

1. **IM concept → rating_scale** (``predicts``): canonical imaging
   biomarkers that predict scale change. Hippocampal volume → ADAS-Cog,
   amyloid burden → CDR-SB, dorsolateral PFC FC → HAM-D, DaTscan →
   MDS-UPDRS, WMH severity → NIHSS, etc. Closes the ``IM → O`` adjacency
   so ``D → IM → O[longitudinal]`` (disease_biomarker_prognosis) and
   ``Rx → IM → O`` (drug_imaging_outcome) chains can land on rating scales.

2. **disease → rating_scale** (``is_assessed_by``): mirror of the
   existing ``scale → measures → disease`` edges, so disease nodes have
   a direct out-edge to the scales used to quantify them. Lets
   ``{D, IM} → O[longitudinal]`` (prognosis) traverse a disease→scale
   tail when the IM side runs out of evidence-rich neighbors.

3. **drug → AE SOC** (``has_adverse_effect``): a small curated set of
   ~30 (drug, SOC) pairs from the SmPC / pharmacology literature
   (e.g. levodopa → Psychiatric disorders, SSRIs → Reproductive and
   breast disorders, antipsychotics → Cardiac disorders). Closes
   ``Rx → O`` adjacency for adverse_event_prediction.

Idempotent — repeated runs only add edges that aren't already present.
Endpoints that don't exist in the KG are silently skipped, so the seed
table can over-list candidates without breaking ingestion.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import Edge

logger = logging.getLogger(__name__)


# ── 1. IM concept → rating_scale (predicts) ──────────────────────────────
# Each entry = (im_concept_id, scale_id). Edge: im --predicts--> scale.
# Curation source: textbook / clinical-trial-endpoint conventions —
# imaging features that the literature consistently uses to predict the
# specific clinical scale. Endpoints not in the KG are skipped silently.
IM_TO_SCALE_EDGES: list[tuple[str, str]] = [
    # ── AD / dementia: structural/molecular IM → cognitive scales ───────
    ("NN:901",                              "OUTCOME:ADAS-Cog"),   # Hippocampus → ADAS-Cog
    ("NN:307",                              "OUTCOME:ADAS-Cog"),   # Entorhinal Cortex → ADAS-Cog
    ("CLM_CONCEPT:bilateral_hippocampal_volume", "OUTCOME:ADAS-Cog"),
    ("CLM_CONCEPT:left_hippocampal_volume",      "OUTCOME:MMSE"),
    ("NN:901",                              "OUTCOME:MMSE"),
    ("NN:307",                              "OUTCOME:MMSE"),
    ("NN:901",                              "OUTCOME:CDR-SB"),
    ("NN:307",                              "OUTCOME:CDR-SB"),
    ("CLM_CONCEPT:amyloid_burden",          "OUTCOME:CDR-SB"),
    ("CLM_CONCEPT:amyloid_burden",          "OUTCOME:ADAS-Cog"),
    ("CLM_CONCEPT:amyloid_positivity",      "OUTCOME:CDR-SB"),
    ("CLM_CONCEPT:amyloid_accumulation",    "OUTCOME:ADAS-Cog"),
    ("CLM_CONCEPT:tau_PET",                 "OUTCOME:CDR-SB"),
    ("CLM_CONCEPT:tau_PET",                 "OUTCOME:ADAS-Cog"),
    ("CLM_CONCEPT:Positron_Emission_Tomography", "OUTCOME:ADAS-Cog"),
    ("CLM_CONCEPT:automated_cortical_thickness_measurements_from_MRI",
                                            "OUTCOME:MMSE"),
    ("CLM_CONCEPT:automated_cortical_thickness_measurements_from_MRI",
                                            "OUTCOME:MoCA"),
    ("MSH:D005625",                         "OUTCOME:MoCA"),  # Frontal Lobe → MoCA
    # ── Mood / depression: limbic + DMN → mood scales ───────────────────
    ("NN:902",                              "OUTCOME:HAM-D"),    # Amygdala → HAM-D
    ("NN:902",                              "OUTCOME:MADRS"),
    ("NN:902",                              "OUTCOME:BDI"),
    ("NN:4000",                             "OUTCOME:HAM-D"),    # DMN → HAM-D
    ("NN:4000",                             "OUTCOME:MADRS"),
    ("MSH:D005625",                         "OUTCOME:HAM-D"),    # Frontal Lobe → HAM-D
    ("MSH:D005625",                         "OUTCOME:PHQ-9"),
    # ── Anxiety: amygdala-centric ──────────────────────────────────────
    ("NN:902",                              "OUTCOME:HAM-A"),
    ("NN:902",                              "OUTCOME:GAD-7"),
    # ── Psychosis: prefrontal + striatum (Striatum NN:920 not in KG; use
    #     Insula as proxy limbic-salience anchor for PANSS) ─────────────
    ("MSH:D005625",                         "OUTCOME:PANSS"),
    ("NN:600",                              "OUTCOME:PANSS"),
    ("MSH:D005625",                         "OUTCOME:BPRS"),
    # ── Parkinson: dopaminergic imaging → MDS-UPDRS ────────────────────
    ("CLM_CONCEPT:abnormal_DaTScan",        "OUTCOME:MDS-UPDRS"),
    ("CLM_CONCEPT:SPECT_with_DaTscan",      "OUTCOME:MDS-UPDRS"),
    ("CLM_CONCEPT:nigrostriatal_dopamine_system", "OUTCOME:MDS-UPDRS"),
    ("CLM_CONCEPT:abnormal_DaTScan",        "OUTCOME:Hoehn-Yahr"),
    ("CLM_CONCEPT:severe_nigrostriatal_dopaminergic_degeneration",
                                            "OUTCOME:MDS-UPDRS"),
    # ── Stroke / vascular: WMH → NIHSS / mRS ───────────────────────────
    ("CLM_CONCEPT:WMH_severity",            "OUTCOME:NIHSS"),
    ("CLM_CONCEPT:normalized_effective_WMH_volume", "OUTCOME:NIHSS"),
    ("CLM_CONCEPT:WMH_severity",            "OUTCOME:mRS"),
    ("CLM_CONCEPT:WMH_severity",            "OUTCOME:Barthel"),
    # ── ALS / MS: cortical thickness + WMH → functional scales ─────────
    ("CLM_CONCEPT:automated_cortical_thickness_measurements_from_MRI",
                                            "OUTCOME:ALSFRS-R"),
    ("CLM_CONCEPT:WMH_severity",            "OUTCOME:EDSS"),
    ("CLM_CONCEPT:WMH_severity",            "OUTCOME:MSFC"),
    # ── ADHD: PFC FC → ADHD-RS / Conners ───────────────────────────────
    ("MSH:D005625",                         "OUTCOME:ADHD-RS"),
    ("MSH:D005625",                         "OUTCOME:Conners"),
    # ── Sleep: DMN / hippocampus → PSQI / ESS ──────────────────────────
    ("NN:4000",                             "OUTCOME:PSQI"),
    ("NN:901",                              "OUTCOME:PSQI"),
    ("NN:4000",                             "OUTCOME:ESS"),
    # ── Generic responder / remission: imaging-defined responder bridge ──
    ("CLM_CONCEPT:functional_connectivity_changes", "OUTCOME:Responder"),
    ("CLM_CONCEPT:amyloid_burden",          "OUTCOME:Responder"),
    ("NN:4000",                             "OUTCOME:Remission"),
    ("NN:902",                              "OUTCOME:Remission"),
]


# ── 2. disease → rating_scale (is_assessed_by, reverse of measures) ──────
# Mirror of `scale --measures--> disease` so disease nodes have a direct
# tail to outcome scales. Mirroring is intentional and idempotent — the
# semantic predicate `is_assessed_by` is the canonical reverse of
# `measures`.
DISEASE_ASSESSED_BY_REVERSE: bool = True


# ── 3. drug → AE SOC (has_adverse_effect) ────────────────────────────────
# Curated set of (drug_id, SOC_id) pairs. Drug IDs follow the ATC namespace
# seeded by ingestion.atc_drugs (ATC:N03* / N04* / N05* / N06* / N07*).
# Endpoints not in the KG are silently skipped.
DRUG_TO_AE_EDGES: list[tuple[str, str]] = [
    # ── ATC class-level edges (4-char class IDs) — let molecule → class → SOC
    #     form 2-hop paths that pass min_hops=2 in post_process. ──────────
    # Antiepileptics
    ("ATC:N03A",   "OUTCOME:AE_Nervous"),
    ("ATC:N03A",   "OUTCOME:AE_Hepatobiliary"),
    ("ATC:N03A",   "OUTCOME:AE_Skin"),
    # Dopa derivatives & DA agonists (PD)
    ("ATC:N04BA",  "OUTCOME:AE_Psychiatric"),
    ("ATC:N04BA",  "OUTCOME:AE_Cardiac"),
    ("ATC:N04BC",  "OUTCOME:AE_Psychiatric"),
    ("ATC:N04BD",  "OUTCOME:AE_Hepatobiliary"),
    # Antipsychotics
    ("ATC:N05AH",  "OUTCOME:AE_Metabolism"),
    ("ATC:N05AH",  "OUTCOME:AE_Cardiac"),
    ("ATC:N05AH",  "OUTCOME:AE_Nervous"),
    ("ATC:N05AA",  "OUTCOME:AE_Nervous"),
    ("ATC:N05AA",  "OUTCOME:AE_Cardiac"),
    ("ATC:N05AX",  "OUTCOME:AE_Metabolism"),
    ("ATC:N05AX",  "OUTCOME:AE_Cardiac"),
    # Anxiolytics / hypnotics
    ("ATC:N05BA",  "OUTCOME:AE_Nervous"),
    ("ATC:N05BA",  "OUTCOME:AE_Psychiatric"),
    ("ATC:N05CF",  "OUTCOME:AE_Nervous"),
    # SSRIs / TCAs / SNRIs
    ("ATC:N06AB",  "OUTCOME:AE_Reproductive"),
    ("ATC:N06AB",  "OUTCOME:AE_Gastrointestinal"),
    ("ATC:N06AB",  "OUTCOME:AE_Cardiac"),
    ("ATC:N06AA",  "OUTCOME:AE_Cardiac"),
    ("ATC:N06AA",  "OUTCOME:AE_Nervous"),
    ("ATC:N06AX",  "OUTCOME:AE_Cardiac"),
    # Stimulants
    ("ATC:N06BA",  "OUTCOME:AE_Cardiac"),
    ("ATC:N06BA",  "OUTCOME:AE_Psychiatric"),
    # Anti-dementia
    ("ATC:N06DA",  "OUTCOME:AE_Gastrointestinal"),
    ("ATC:N06DA",  "OUTCOME:AE_Nervous"),
    ("ATC:N06DX",  "OUTCOME:AE_Nervous"),
    # ── Molecule-level edges (kept for richer connectivity, picked up via
    #     {Rx} → IM → SOC if hypothesis engine prefers longer paths) ────
    # Antiepileptics (N03)
    ("ATC:N03AX14", "OUTCOME:AE_Nervous"),       # Levetiracetam → CNS
    ("ATC:N03AX14", "OUTCOME:AE_Psychiatric"),
    ("ATC:N03AG01", "OUTCOME:AE_Nervous"),       # Valproate
    ("ATC:N03AG01", "OUTCOME:AE_Hepatobiliary"),
    ("ATC:N03AF01", "OUTCOME:AE_Nervous"),       # Carbamazepine
    ("ATC:N03AF01", "OUTCOME:AE_Skin"),
    # Antiparkinsonians (N04)
    ("ATC:N04BA02", "OUTCOME:AE_Psychiatric"),   # Levodopa+Benserazide
    ("ATC:N04BA02", "OUTCOME:AE_Nervous"),
    ("ATC:N04BA02", "OUTCOME:AE_Cardiac"),
    ("ATC:N04BC05", "OUTCOME:AE_Psychiatric"),   # Pramipexole → ICD
    ("ATC:N04BC05", "OUTCOME:AE_Nervous"),
    ("ATC:N04BD01", "OUTCOME:AE_Hepatobiliary"), # Selegiline
    # Antipsychotics (N05A)
    ("ATC:N05AH04", "OUTCOME:AE_Metabolism"),    # Quetiapine
    ("ATC:N05AH04", "OUTCOME:AE_Cardiac"),
    ("ATC:N05AH03", "OUTCOME:AE_Metabolism"),    # Olanzapine
    ("ATC:N05AH03", "OUTCOME:AE_Nervous"),
    ("ATC:N05AX08", "OUTCOME:AE_Metabolism"),    # Risperidone
    ("ATC:N05AX08", "OUTCOME:AE_Nervous"),
    ("ATC:N05AX12", "OUTCOME:AE_Cardiac"),       # Aripiprazole
    ("ATC:N05AA01", "OUTCOME:AE_Nervous"),       # Chlorpromazine
    ("ATC:N05AA01", "OUTCOME:AE_Cardiac"),
    # Anxiolytics / hypnotics (N05B/C)
    ("ATC:N05BA01", "OUTCOME:AE_Nervous"),       # Diazepam
    ("ATC:N05BA01", "OUTCOME:AE_Psychiatric"),
    ("ATC:N05BA12", "OUTCOME:AE_Nervous"),       # Alprazolam
    ("ATC:N05CF02", "OUTCOME:AE_Nervous"),       # Zolpidem
    # Antidepressants (N06A)
    ("ATC:N06AB03", "OUTCOME:AE_Reproductive"),  # Fluoxetine
    ("ATC:N06AB03", "OUTCOME:AE_Gastrointestinal"),
    ("ATC:N06AB04", "OUTCOME:AE_Reproductive"),  # Citalopram
    ("ATC:N06AB04", "OUTCOME:AE_Cardiac"),
    ("ATC:N06AB05", "OUTCOME:AE_Reproductive"),  # Paroxetine
    ("ATC:N06AB06", "OUTCOME:AE_Nervous"),       # Sertraline
    ("ATC:N06AB06", "OUTCOME:AE_Gastrointestinal"),
    ("ATC:N06AX16", "OUTCOME:AE_Cardiac"),       # Venlafaxine
    ("ATC:N06AA09", "OUTCOME:AE_Cardiac"),       # Amitriptyline
    ("ATC:N06AA09", "OUTCOME:AE_Nervous"),
    # Anti-dementia (N06D)
    ("ATC:N06DA02", "OUTCOME:AE_Gastrointestinal"),  # Donepezil
    ("ATC:N06DA02", "OUTCOME:AE_Nervous"),
    ("ATC:N06DA03", "OUTCOME:AE_Gastrointestinal"),  # Rivastigmine
    ("ATC:N06DX01", "OUTCOME:AE_Nervous"),           # Memantine
    # Stimulants (N06B)
    ("ATC:N06BA04", "OUTCOME:AE_Cardiac"),       # Methylphenidate
    ("ATC:N06BA04", "OUTCOME:AE_Psychiatric"),
    ("ATC:N06BA09", "OUTCOME:AE_Cardiac"),       # Atomoxetine
    # Other (N07)
    ("ATC:N07XX02", "OUTCOME:AE_Hepatobiliary"), # Riluzole
]


def _add_edge_if_new(
    kg: KnowledgeGraph,
    source_id: str,
    target_id: str,
    relation_type: str,
    source: str,
    confidence: float = 0.85,
) -> bool:
    if source_id == target_id:
        return False
    if not (kg.has_concept(source_id) and kg.has_concept(target_id)):
        return False
    if kg.G.has_edge(source_id, target_id):
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


def _bridge_im_to_scales(kg: KnowledgeGraph) -> int:
    added = 0
    for im_id, scale_id in IM_TO_SCALE_EDGES:
        if _add_edge_if_new(kg, im_id, scale_id, "predicts",
                            "OutcomeIM-Bridge"):
            added += 1
    return added


def _bridge_disease_to_scales(kg: KnowledgeGraph) -> int:
    """Mirror of `scale → measures → disease` as `disease → is_assessed_by → scale`.

    Dynamic — walks existing `measures` edges instead of a hardcoded table,
    so any future scale added to clinical_outcomes auto-mirrors.
    """
    added = 0
    measures_pairs = [
        (src, tgt)
        for src, tgt, data in kg.G.edges(data=True)
        if data.get("relation_type") == "measures"
    ]
    for scale_id, disease_id in measures_pairs:
        # scale -> disease forward; we want disease -> scale reverse
        if _add_edge_if_new(kg, disease_id, scale_id, "is_assessed_by",
                            "OutcomeIM-Bridge"):
            added += 1
    return added


def _bridge_drug_to_ae(kg: KnowledgeGraph) -> int:
    added = 0
    for drug_id, soc_id in DRUG_TO_AE_EDGES:
        if _add_edge_if_new(kg, drug_id, soc_id, "has_adverse_effect",
                            "OutcomeIM-Bridge"):
            added += 1
    return added


def ingest_outcome_im_bridges(kg: KnowledgeGraph) -> dict:
    """Wire OUTCOME nodes with incoming IM/disease/drug edges. Idempotent."""
    counts = {
        "im_to_scale_edges":      _bridge_im_to_scales(kg),
        "disease_to_scale_edges": _bridge_disease_to_scales(kg),
        "drug_to_ae_edges":       _bridge_drug_to_ae(kg),
    }
    counts["total"] = sum(counts.values())
    logger.info(f"outcome IM bridges complete: {counts}")
    return counts


__all__ = [
    "ingest_outcome_im_bridges",
    "IM_TO_SCALE_EDGES",
    "DRUG_TO_AE_EDGES",
]
