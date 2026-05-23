"""Phase 1.6 — concept-level anchors for INDIVIDUAL_DATA atom.

The KG carries dataset_variable hubs (UKB:CAT*, ADNI:DOM_*, HCP:DOM_*) but
they are pure structural pointers — no concept-level entry point exists for
"Aging", "Smoking", "APOE carrier status" etc., so any task whose output
atom is INDIVIDUAL_DATA (brain_age, connectome_behavior,
task_brain_behavior, disease_biomarker_prognosis) cannot find a path
because there is nothing to traverse on the conceptual side of the bridge.

This module seeds ~20 concept-level anchor nodes covering five axes:

* **demographics** — Aging, Sex, Education, Socioeconomic Status
* **lifestyle** — Smoking, Alcohol Drinking, Exercise, Diet, Sleep
* **anthropometrics** — Body Mass Index, Blood Pressure
* **genetics** — APOE epsilon4 (existing), Polygenic Risk Score
* **personality** — Big-Five traits + general Personality

Anchors carry ``domain_tags=["individual_data_anchor"]`` (added in
:data:`schema.DomainTag`) so the atom-aware filters route them as
INDIVIDUAL_DATA targets without conflating them with the dataset_variable
hubs themselves. The corresponding bridges (anchor → dataset_variable,
IM ↔ anchor) are seeded by :mod:`individual_data_bridges`.

Idempotent — repeated runs only add nodes that aren't already present.
Existing well-anchored concepts (MSH:D012907 Smoking, MSH:D000428 Alcohol
Drinking, MSH:D015444 Exercise, MSH:D012893 Sleep Wake Disorders,
MSH:D010551 Personality, CLM_CONCEPT:epsilon4_allele_of_the_apolipoprotein_E_gene_(APOE))
are reused — this module attaches the ``individual_data_anchor`` tag onto
them rather than creating duplicates.
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag

logger = logging.getLogger(__name__)


# Canonical anchor table.
#
# Schema per entry:
#   id            — KG node id (existing MSH/NN/CLM_CONCEPT preferred; otherwise
#                   custom INDIVIDUAL_DATA:* namespace).
#   name          — preferred display name.
#   axis          — semantic axis tag ("demographics" | "lifestyle" |
#                   "anthropometric" | "genetics" | "personality").
#   reuse_id      — if set, points at an existing KG node whose domain_tags we
#                   should *augment* with individual_data_anchor instead of
#                   creating a new node.
INDIVIDUAL_DATA_ANCHORS: dict[str, dict] = {
    # ── demographics ────────────────────────────────────────────────────
    "INDIVIDUAL_DATA:aging": {
        "name": "Aging",
        "axis": "demographics",
        "definition": "Chronological age and biological aging processes; "
                      "the canonical INDIVIDUAL_DATA target for brain-age "
                      "regression and longitudinal studies.",
    },
    "INDIVIDUAL_DATA:sex": {
        "name": "Biological Sex",
        "axis": "demographics",
        "definition": "Male/female demographic split; covariate or effect "
                      "modifier in most imaging studies.",
    },
    "INDIVIDUAL_DATA:education": {
        "name": "Educational Attainment",
        "axis": "demographics",
        "definition": "Years of education; cognitive-reserve proxy.",
    },
    "INDIVIDUAL_DATA:socioeconomic_status": {
        "name": "Socioeconomic Status",
        "axis": "demographics",
        "definition": "Composite of income, occupation, education; "
                      "Townsend deprivation index in UKB.",
    },
    # ── lifestyle (reuse where possible) ────────────────────────────────
    "MSH:D012907": {
        "name": "Smoking",
        "axis": "lifestyle",
        "reuse": True,
        "definition": "Tobacco-smoking exposure (current/former/never, "
                      "pack-years).",
    },
    "MSH:D000428": {
        "name": "Alcohol Drinking",
        "axis": "lifestyle",
        "reuse": True,
        "definition": "Alcohol-consumption exposure (g/week, AUDIT score).",
    },
    "MSH:D015444": {
        "name": "Exercise",
        "axis": "lifestyle",
        "reuse": True,
        "definition": "Physical-activity exposure (MET-min/week, IPAQ).",
    },
    "INDIVIDUAL_DATA:diet": {
        "name": "Dietary Pattern",
        "axis": "lifestyle",
        "definition": "Mediterranean/MIND/processed-food diet scores; "
                      "UKB CAT100054 Diet variables.",
    },
    "MSH:D012893": {
        "name": "Sleep Quality",
        "axis": "lifestyle",
        "reuse": True,
        "reuse_alias": "Sleep Wake Disorders",
        "definition": "Sleep duration / efficiency / chronotype; "
                      "Pittsburgh Sleep Quality Index.",
    },
    # ── anthropometrics ─────────────────────────────────────────────────
    "INDIVIDUAL_DATA:body_mass_index": {
        "name": "Body Mass Index",
        "axis": "anthropometric",
        "definition": "BMI = weight / height^2; modulator of cortical "
                      "thickness, FC, white-matter integrity.",
    },
    "INDIVIDUAL_DATA:blood_pressure": {
        "name": "Blood Pressure",
        "axis": "anthropometric",
        "definition": "Systolic / diastolic blood pressure; vascular-aging "
                      "risk factor for white-matter hyperintensities.",
    },
    # ── genetics ────────────────────────────────────────────────────────
    "CLM_CONCEPT:epsilon4_allele_of_the_apolipoprotein_E_gene_(APOE)": {
        "name": "APOE epsilon4 allele",
        "axis": "genetics",
        "reuse": True,
        "definition": "APOE-ε4 carrier status; strongest common-variant "
                      "risk factor for late-onset AD; modulator of "
                      "hippocampal/entorhinal volume and amyloid burden.",
    },
    "INDIVIDUAL_DATA:polygenic_risk_score": {
        "name": "Polygenic Risk Score",
        "axis": "genetics",
        "definition": "Aggregated SNP-level risk score (PRS) for a target "
                      "trait or disease; computed from GWAS summary stats.",
    },
    # ── personality ─────────────────────────────────────────────────────
    "MSH:D010551": {
        "name": "Personality",
        "axis": "personality",
        "reuse": True,
        "definition": "Big-Five / NEO-PI personality dimensions (umbrella).",
    },
    "INDIVIDUAL_DATA:big5_neuroticism": {
        "name": "Neuroticism",
        "axis": "personality",
        "definition": "Big-Five trait — tendency toward negative emotion. "
                      "Correlates with amygdala reactivity / FC.",
    },
    "INDIVIDUAL_DATA:big5_extraversion": {
        "name": "Extraversion",
        "axis": "personality",
        "definition": "Big-Five trait — sociability / positive affect. "
                      "Correlates with reward-system activity.",
    },
    "INDIVIDUAL_DATA:big5_conscientiousness": {
        "name": "Conscientiousness",
        "axis": "personality",
        "definition": "Big-Five trait — self-discipline / goal direction. "
                      "Correlates with PFC structure and function.",
    },
    "INDIVIDUAL_DATA:big5_openness": {
        "name": "Openness to Experience",
        "axis": "personality",
        "definition": "Big-Five trait — intellectual curiosity / aesthetic "
                      "sensitivity.",
    },
    "INDIVIDUAL_DATA:big5_agreeableness": {
        "name": "Agreeableness",
        "axis": "personality",
        "definition": "Big-Five trait — prosocial orientation.",
    },
    "INDIVIDUAL_DATA:grit": {
        "name": "Grit",
        "axis": "personality",
        "definition": "Self-reported persistence / passion for long-term "
                      "goals; HCP Grit-S scale.",
    },
}


def ingest_individual_data_anchors(kg: KnowledgeGraph) -> dict:
    """Seed concept-level INDIVIDUAL_DATA anchor nodes. Idempotent."""
    counts = {"new": 0, "tagged_existing": 0}

    for nid, info in INDIVIDUAL_DATA_ANCHORS.items():
        existing = kg._index.get(nid)
        if existing is not None:
            if DomainTag.INDIVIDUAL_DATA_ANCHOR.value not in existing.domain_tags:
                existing.domain_tags.append(DomainTag.INDIVIDUAL_DATA_ANCHOR.value)
                kg.G.nodes[nid]["domain_tags"] = list(existing.domain_tags)
                if info.get("axis"):
                    md = existing.metadata if existing.metadata is not None else {}
                    md["individual_data_axis"] = info["axis"]
                    existing.metadata = md
                    kg.G.nodes[nid]["metadata"] = dict(md)
                counts["tagged_existing"] += 1
            continue

        # Brand new anchor node
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=info["name"],
            domain_tags=[DomainTag.INDIVIDUAL_DATA_ANCHOR.value],
            source_vocab="IndividualDataAnchor",
            definition=info.get("definition", ""),
            metadata={"individual_data_axis": info.get("axis", "")},
        ))
        counts["new"] += 1

    counts["total"] = counts["new"] + counts["tagged_existing"]
    logger.info(f"individual_data anchors ingestion complete: {counts}")
    return counts


__all__ = ["ingest_individual_data_anchors", "INDIVIDUAL_DATA_ANCHORS"]
