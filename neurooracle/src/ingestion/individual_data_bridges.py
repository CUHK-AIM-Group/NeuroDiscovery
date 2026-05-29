"""Phase 1.6 — bridges connecting INDIVIDUAL_DATA anchors to the rest of the KG.

The :mod:`individual_data_anchors` module seeds concept-level anchors (Aging,
APOE, Big-5 traits, …). On their own those anchors are still isolated from
the KG. This module wires them in three directions:

1. **Anchor → dataset_variable hub** (``assessed_in`` edge): tells the
   hypothesis engine where each anchor is actually measured. E.g.
   Aging --assessed_in--> UKB:CAT_Demographics. This closes the path
   ``IM ... → anchor → dataset_variable`` for all INDIVIDUAL_DATA-output
   tasks (brain_age, connectome_behavior, task_brain_behavior,
   disease_biomarker_prognosis).

2. **IM concept → anchor** (``correlates_with`` / ``modulates``): the
   neuroscience consensus map. cortical thickness ↔ Aging, hippocampal
   volume ↔ APOE-ε4, amygdala FC ↔ Neuroticism, insular volume ↔ Smoking,
   etc. These edges are seeded as curated facts (not paper claims) — the
   set is intentionally small (~30 entries) and high-confidence.

3. **Cognitive task / disease ↔ anchor** (a few targeted edges):
   working memory ↔ Aging, AD ↔ APOE-ε4, MDD ↔ Neuroticism. Lets
   prognosis / connectome_behavior reach personality and demographics
   through cognitive / clinical anchors.

Idempotent — repeated runs only add edges that aren't already present.
Endpoints that don't exist in the KG are silently skipped, so the seed
table can over-list candidates without breaking ingestion.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import Edge

logger = logging.getLogger(__name__)


# ── 1. anchor → dataset_variable hosts ───────────────────────────────────
# Each anchor lives in one or more UKB/ADNI/HCP host categories. Edge:
#   anchor --assessed_in--> dataset_variable
# Reuses the existing 'assessed_in' relation already used by rating scales.
ANCHOR_TO_DATASET_VARS: dict[str, list[str]] = {
    # demographics
    "INDIVIDUAL_DATA:aging":
        ["UKB:CAT_Demographics", "ADNI:DOM_DEMOG", "HCP:DOM_DEMOG"],
    "INDIVIDUAL_DATA:sex":
        ["UKB:CAT_Demographics", "ADNI:DOM_DEMOG", "HCP:DOM_DEMOG"],
    "INDIVIDUAL_DATA:education":
        ["UKB:CAT_Demographics", "HCP:DOM_DEMOG"],
    "INDIVIDUAL_DATA:socioeconomic_status":
        ["UKB:CAT_Demographics"],
    # lifestyle
    "MSH:D012907":
        ["UKB:CAT100041"],                                    # Smoking
    "MSH:D000428":
        ["UKB:CAT100051"],                                    # Alcohol
    "MSH:D015444":
        ["UKB:CAT100054"],                                    # Exercise
    "INDIVIDUAL_DATA:diet":
        ["UKB:CAT100070"],
    "MSH:D012893":
        ["UKB:CAT100040", "HCP:DOM_PSYCH"],                   # Sleep
    # anthropometric
    "INDIVIDUAL_DATA:body_mass_index":
        ["UKB:CAT_Demographics"],
    "INDIVIDUAL_DATA:blood_pressure":
        ["UKB:CAT_Demographics"],
    # genetics
    "CLM_CONCEPT:epsilon4_allele_of_the_apolipoprotein_E_gene_(APOE)":
        ["ADNI:DOM_GENETICS"],
    "INDIVIDUAL_DATA:polygenic_risk_score":
        ["ADNI:DOM_GENETICS", "UKB:CAT_Demographics"],
    # personality
    "MSH:D010551":
        ["HCP:DOM_PSYCH"],                                    # Personality umbrella
    "INDIVIDUAL_DATA:big5_neuroticism":     ["HCP:DOM_PSYCH"],
    "INDIVIDUAL_DATA:big5_extraversion":    ["HCP:DOM_PSYCH"],
    "INDIVIDUAL_DATA:big5_conscientiousness": ["HCP:DOM_PSYCH"],
    "INDIVIDUAL_DATA:big5_openness":        ["HCP:DOM_PSYCH"],
    "INDIVIDUAL_DATA:big5_agreeableness":   ["HCP:DOM_PSYCH"],
}


# ── 2. IM concept ↔ anchor (textbook-level associations) ─────────────────
# Each tuple = (im_concept_id, anchor_id, relation_type). Both directions are
# emitted by the bridge: forward as the listed relation_type, reverse as
# 'correlates_with' (symmetric, low-bias). All endpoints are validated at
# runtime — missing IDs are silently dropped.
#
# Curation source: textbook neuroscience consensus (NeuroOracle KG charter:
# only obvious, well-replicated facts here — paper-level claims belong in
# the claim ingestion pipeline). Added 2026-05-21 to unblock brain_age /
# connectome_behavior / task_brain_behavior tasks.
IM_TO_ANCHOR_EDGES: list[tuple[str, str, str]] = [
    # ── Aging ↔ structural & functional imaging ────────────────────────
    # (cortical thickness / GM volume decline with age — universal finding)
    ("CLM_CONCEPT:automated_cortical_thickness_measurements_from_MRI",
     "INDIVIDUAL_DATA:aging", "correlates_with"),
    ("MSH:D005625", "INDIVIDUAL_DATA:aging", "correlates_with"),     # Frontal Lobe
    ("NN:300",      "INDIVIDUAL_DATA:aging", "correlates_with"),     # Temporal Lobe
    ("NN:901",      "INDIVIDUAL_DATA:aging", "correlates_with"),     # Hippocampus
    ("NN:307",      "INDIVIDUAL_DATA:aging", "correlates_with"),     # Entorhinal Cortex
    ("NN:4000",     "INDIVIDUAL_DATA:aging", "modulates"),           # DMN modulated by age
    # ── APOE ε4 ↔ medial temporal lobe ─────────────────────────────────
    ("NN:901",      "CLM_CONCEPT:epsilon4_allele_of_the_apolipoprotein_E_gene_(APOE)",
                    "modulates"),
    ("NN:307",      "CLM_CONCEPT:epsilon4_allele_of_the_apolipoprotein_E_gene_(APOE)",
                    "modulates"),
    # ── Personality ↔ limbic / prefrontal ──────────────────────────────
    ("NN:902",      "INDIVIDUAL_DATA:big5_neuroticism", "correlates_with"),  # Amygdala
    ("MSH:D005625", "INDIVIDUAL_DATA:big5_conscientiousness",
                    "correlates_with"),                              # Frontal Lobe
    ("NN:600",      "INDIVIDUAL_DATA:big5_extraversion", "correlates_with"), # Insula
    ("NN:4000",     "INDIVIDUAL_DATA:big5_openness", "correlates_with"),     # DMN
    # ── Smoking ↔ insula / OFC volume reduction ────────────────────────
    ("NN:600",      "MSH:D012907", "modulates"),                     # Insula
    ("MSH:D005625", "MSH:D012907", "modulates"),                     # Frontal Lobe (proxy for OFC)
    # ── Alcohol ↔ cerebellum / hippocampal atrophy ─────────────────────
    ("MSH:D002531", "MSH:D000428", "modulates"),                     # Cerebellum
    ("NN:901",      "MSH:D000428", "modulates"),                     # Hippocampus
    # ── Exercise ↔ hippocampal volume / DMN ────────────────────────────
    ("NN:901",      "MSH:D015444", "correlates_with"),
    ("NN:4000",     "MSH:D015444", "correlates_with"),
    # ── Sleep ↔ DMN / hippocampus ──────────────────────────────────────
    ("NN:4000",     "MSH:D012893", "modulates"),
    ("NN:901",      "MSH:D012893", "modulates"),
    # ── BMI ↔ gray matter / FC ─────────────────────────────────────────
    ("CLM_CONCEPT:BMI", "INDIVIDUAL_DATA:body_mass_index", "is_a"),  # Identity bridge
    ("MSH:D005625", "INDIVIDUAL_DATA:body_mass_index", "modulates"),
    ("NN:4000",     "INDIVIDUAL_DATA:body_mass_index", "modulates"),
    # ── Blood pressure ↔ white matter ──────────────────────────────────
    ("CLM_CONCEPT:automated_cortical_thickness_measurements_from_MRI",
     "INDIVIDUAL_DATA:blood_pressure", "modulates"),
    # ── Education / SES ↔ cognitive reserve regions ────────────────────
    ("MSH:D005625", "INDIVIDUAL_DATA:education", "correlates_with"),
    ("NN:300",      "INDIVIDUAL_DATA:education", "correlates_with"),
]


# ── 3. cognitive function / disease ↔ anchor edges ───────────────────────
COGNITIVE_TO_ANCHOR_EDGES: list[tuple[str, str, str]] = [
    # working memory declines with age
    ("COGAT_CONCEPT:trm_4a3fd79d0a891", "INDIVIDUAL_DATA:aging", "correlates_with"),
    # AD ↔ APOE
    ("MSH:D000544",
     "CLM_CONCEPT:epsilon4_allele_of_the_apolipoprotein_E_gene_(APOE)",
     "is_associated_with"),
    # MDD ↔ Neuroticism
    ("COGAT_DISORDER:dso_1470",
     "INDIVIDUAL_DATA:big5_neuroticism", "correlates_with"),
    # Anxiety ↔ Neuroticism
    ("MSH:D001008", "INDIVIDUAL_DATA:big5_neuroticism", "correlates_with"),
    # Sleep disorders ↔ Sleep anchor (already same MSH ID — skipped via self-loop)
]


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


def _bridge_anchors_to_dataset_vars(kg: KnowledgeGraph) -> int:
    added = 0
    for anchor_id, hosts in ANCHOR_TO_DATASET_VARS.items():
        if not kg.has_concept(anchor_id):
            continue
        for host_id in hosts:
            if _add_edge_if_new(kg, anchor_id, host_id, "assessed_in",
                                "IndividualData-Bridge"):
                added += 1
    return added


def _bridge_im_to_anchors(kg: KnowledgeGraph) -> int:
    added = 0
    for im_id, anchor_id, rel in IM_TO_ANCHOR_EDGES:
        # forward direction (IM → anchor)
        if _add_edge_if_new(kg, im_id, anchor_id, rel, "IndividualData-Bridge"):
            added += 1
        # reverse direction (anchor → IM, symmetric correlate)
        if _add_edge_if_new(kg, anchor_id, im_id, "correlates_with",
                            "IndividualData-Bridge"):
            added += 1
    return added


def _bridge_cognitive_to_anchors(kg: KnowledgeGraph) -> int:
    added = 0
    for src_id, anchor_id, rel in COGNITIVE_TO_ANCHOR_EDGES:
        if _add_edge_if_new(kg, src_id, anchor_id, rel,
                            "IndividualData-Bridge"):
            added += 1
        if _add_edge_if_new(kg, anchor_id, src_id, "correlates_with",
                            "IndividualData-Bridge"):
            added += 1
    return added


def ingest_individual_data_bridges(kg: KnowledgeGraph) -> dict:
    """Wire INDIVIDUAL_DATA anchors into the rest of the KG. Idempotent."""
    counts = {
        "anchor_to_dataset_edges": _bridge_anchors_to_dataset_vars(kg),
        "im_to_anchor_edges":      _bridge_im_to_anchors(kg),
        "cognitive_to_anchor_edges": _bridge_cognitive_to_anchors(kg),
    }
    counts["total"] = sum(counts.values())
    logger.info(f"individual_data bridges complete: {counts}")
    return counts


__all__ = [
    "ingest_individual_data_bridges",
    "ANCHOR_TO_DATASET_VARS",
    "IM_TO_ANCHOR_EDGES",
    "COGNITIVE_TO_ANCHOR_EDGES",
]
