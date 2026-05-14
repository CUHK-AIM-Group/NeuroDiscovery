"""Phase 1: Visual stimulus node seeding.

Seeds a 3-tier taxonomy of visual stimuli for brain-decoding datasets
(NSD / BOLD5000 / SEED-DV).

    TIER 1   stimulus modality   (image, video)
    TIER 2   abstract category   (face, body, scene, object, motion, word,
                                  animal, food, tool) — each linked to the
                                  functional ROI it reliably activates
    TIER 3   dataset-specific    (COCO supercategories, Places365 groups,
                                  SEED-DV video concepts)

Edges created:
    VS:* --is_a--> VS:*                   (TIER 3 → TIER 2 → TIER 1)
    VS:<tier2> --evokes--> VROI:*         (face → FFA, scene → PPA, ...)
    DATASET:<dataset> --provides_modality--> VS:<tier1>
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)


# ── TIER 1: stimulus modality ──────────────────────────────────────────

TIER1: dict[str, dict] = {
    "VS:image": {
        "preferred_name": "image stimulus",
        "aliases": ["still image", "picture stimulus", "photograph"],
        "description": "Static 2D image used as visual stimulus (NSD, BOLD5000).",
    },
    "VS:video": {
        "preferred_name": "video stimulus",
        "aliases": ["dynamic video", "film clip stimulus", "motion video"],
        "description": "Dynamic video stimulus with temporal content (SEED-DV, SEED emotion family).",
    },
}


# ── TIER 2: abstract category → functional ROI ────────────────────────

# Each tier-2 node declares:
#   - parent (TIER 1 VS:* node, via is_a)
#   - evokes_roi (VROI:* or NN:* that this category reliably activates)
#   - dataset-localizer reference
TIER2: dict[str, dict] = {
    "VS:face": {
        "preferred_name": "face stimulus",
        "aliases": ["human face", "face image", "face photograph"],
        "parent": "VS:image",
        "evokes_roi": "VROI:FFA",
        "description": "Face-selective stimulus reliably activating fusiform face area.",
    },
    "VS:body": {
        "preferred_name": "body stimulus",
        "aliases": ["human body", "body part stimulus"],
        "parent": "VS:image",
        "evokes_roi": "VROI:EBA",
        "description": "Body-part stimulus activating extrastriate body area.",
    },
    "VS:scene": {
        "preferred_name": "scene stimulus",
        "aliases": ["place stimulus", "environmental scene", "landscape"],
        "parent": "VS:image",
        "evokes_roi": "VROI:PPA",
        "description": "Scene/place stimulus activating parahippocampal place area.",
    },
    "VS:object": {
        "preferred_name": "object stimulus",
        "aliases": ["object image", "everyday object"],
        "parent": "VS:image",
        "evokes_roi": "VROI:LOC",
        "description": "Object stimulus activating lateral occipital complex.",
    },
    "VS:motion": {
        "preferred_name": "motion stimulus",
        "aliases": ["moving stimulus", "motion image", "dynamic pattern"],
        "parent": "VS:video",
        "evokes_roi": "VROI:MTplus",
        "description": "Motion stimulus activating MT+ (V5) complex.",
    },
    "VS:word": {
        "preferred_name": "word stimulus",
        "aliases": ["written word", "text stimulus", "letter string"],
        "parent": "VS:image",
        "evokes_roi": "VROI:VWFA",
        "description": "Word/text stimulus activating visual word form area.",
    },
    "VS:animal": {
        "preferred_name": "animal stimulus",
        "aliases": ["animal image", "animate category"],
        "parent": "VS:image",
        "evokes_roi": "VROI:FFA",  # animates activate face-adjacent fusiform
        "description": "Animate (non-human) stimulus activating fusiform animal-selective patches.",
    },
    "VS:food": {
        "preferred_name": "food stimulus",
        "aliases": ["food image", "edible object"],
        "parent": "VS:image",
        "evokes_roi": "VROI:LOC",  # object-like, also ventral-temporal
        "description": "Food stimulus; activates ventral object pathway + reward system.",
    },
    "VS:tool": {
        "preferred_name": "tool stimulus",
        "aliases": ["tool image", "manipulable object"],
        "parent": "VS:image",
        "evokes_roi": "VROI:LOC",
        "description": "Tool stimulus; engages LOC + parietal tool-use network.",
    },
}


# ── TIER 3: dataset-specific concrete categories ──────────────────────

# COCO 12 supercategories (NSD uses COCO)
COCO_SUPER = {
    "VS:coco:person":       ("person",        "VS:body"),
    "VS:coco:vehicle":      ("vehicle",       "VS:object"),
    "VS:coco:outdoor":      ("outdoor",       "VS:scene"),
    "VS:coco:animal":       ("animal",        "VS:animal"),
    "VS:coco:accessory":    ("accessory",     "VS:object"),
    "VS:coco:sports":       ("sports",        "VS:object"),
    "VS:coco:kitchen":      ("kitchen",       "VS:tool"),
    "VS:coco:food":         ("food",          "VS:food"),
    "VS:coco:furniture":    ("furniture",     "VS:object"),
    "VS:coco:electronic":   ("electronic",    "VS:object"),
    "VS:coco:appliance":    ("appliance",     "VS:object"),
    "VS:coco:indoor":       ("indoor",        "VS:scene"),
}

# Places365 8 high-level scene groups
PLACES_SUPER = {
    "VS:places:indoor_home":       ("indoor_home",       "VS:scene"),
    "VS:places:indoor_public":     ("indoor_public",     "VS:scene"),
    "VS:places:indoor_work":       ("indoor_work",       "VS:scene"),
    "VS:places:outdoor_natural":   ("outdoor_natural",   "VS:scene"),
    "VS:places:outdoor_urban":     ("outdoor_urban",     "VS:scene"),
    "VS:places:transport":         ("transport",         "VS:scene"),
    "VS:places:sports_leisure":    ("sports_leisure",    "VS:scene"),
    "VS:places:commercial":        ("commercial",        "VS:scene"),
}

# SEED-DV 9 video concepts (from EEG2Video NeurIPS 2024 paper)
SEED_DV_CONCEPTS = {
    "VS:seeddv:animal":   ("animal_video",   "VS:animal"),
    "VS:seeddv:people":   ("people_video",   "VS:face"),
    "VS:seeddv:dancing":  ("dancing_video",  "VS:motion"),
    "VS:seeddv:sea":      ("sea_video",      "VS:scene"),
    "VS:seeddv:mountain": ("mountain_video", "VS:scene"),
    "VS:seeddv:city":     ("city_video",     "VS:scene"),
    "VS:seeddv:cars":     ("cars_video",     "VS:object"),
    "VS:seeddv:eating":   ("eating_video",   "VS:food"),
    "VS:seeddv:indoor":   ("indoor_video",   "VS:scene"),
}


def _seed_tier(
    kg: KnowledgeGraph,
    tier_map: dict,
    *,
    with_parent: bool,
    stats_key: str,
    stats: dict,
    source_vocab: str,
    domain_tag: str,
) -> None:
    """Shared helper to add a set of visual_stimulus nodes and their is_a edges."""
    for nid, info in tier_map.items():
        if kg.has_concept(nid):
            continue
        if with_parent:
            # tier-2 style: dict info has preferred_name/aliases/description/parent/evokes_roi
            preferred = info["preferred_name"]
            aliases = info.get("aliases", [])
            description = info.get("description", "")
            metadata = {
                "evokes_roi": info.get("evokes_roi"),
                "tier": 2,
            }
        else:
            # tier-3 style: info is (preferred_name, parent_tier2_id)
            preferred, _parent = info
            aliases = []
            description = ""
            metadata = {"tier": 3}
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=preferred,
            domain_tags=[domain_tag],
            source_vocab=source_vocab,
            aliases=aliases,
            definition=description,
            metadata=metadata,
        ))
        stats[stats_key] = stats.get(stats_key, 0) + 1


def ingest_visual_stimuli(kg: KnowledgeGraph) -> dict:
    """Seed the visual_stimulus taxonomy and its connections. Idempotent."""
    stats = {
        "tier1_added": 0,
        "tier2_added": 0,
        "tier3_added": 0,
        "is_a_edges": 0,
        "evokes_edges": 0,
        "dataset_edges": 0,
    }
    tag = DomainTag.VISUAL_STIMULUS.value

    # TIER 1
    for nid, info in TIER1.items():
        if kg.has_concept(nid):
            continue
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=info["preferred_name"],
            domain_tags=[tag],
            source_vocab="visual_stimulus_taxonomy",
            aliases=info["aliases"],
            definition=info["description"],
            metadata={"tier": 1},
        ))
        stats["tier1_added"] += 1

    # TIER 2
    for nid, info in TIER2.items():
        if kg.has_concept(nid):
            continue
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=info["preferred_name"],
            domain_tags=[tag],
            source_vocab="visual_stimulus_taxonomy",
            aliases=info["aliases"],
            definition=info["description"],
            metadata={"tier": 2, "evokes_roi": info.get("evokes_roi")},
        ))
        stats["tier2_added"] += 1

        # is_a edge to TIER 1
        parent = info.get("parent")
        if parent and kg.has_concept(parent):
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(source_id=nid, target_id=parent,
                             relation_type="is_a",
                             source="visual_stimulus_taxonomy", confidence=1.0))
            if kg.G.number_of_edges() > before:
                stats["is_a_edges"] += 1

        # evokes edge → functional ROI
        roi = info.get("evokes_roi")
        if roi and kg.has_concept(roi):
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(source_id=nid, target_id=roi,
                             relation_type="evokes",
                             source="visual_stimulus_taxonomy", confidence=0.85))
            if kg.G.number_of_edges() > before:
                stats["evokes_edges"] += 1

    # TIER 3 — COCO / Places / SEED-DV
    for source_label, mapping in [
        ("COCO",        COCO_SUPER),
        ("Places365",   PLACES_SUPER),
        ("SEED-DV",     SEED_DV_CONCEPTS),
    ]:
        for nid, (preferred, parent) in mapping.items():
            if kg.has_concept(nid):
                continue
            kg.add_concept(ConceptNode(
                id=nid,
                preferred_name=preferred,
                domain_tags=[tag],
                source_vocab=f"visual_stimulus_taxonomy:{source_label}",
                aliases=[],
                definition=f"{source_label} category: {preferred}",
                metadata={"tier": 3, "parent_category": parent,
                          "source_dataset_family": source_label},
            ))
            stats["tier3_added"] += 1

            if kg.has_concept(parent):
                before = kg.G.number_of_edges()
                kg.add_edge(Edge(source_id=nid, target_id=parent,
                                 relation_type="is_a",
                                 source="visual_stimulus_taxonomy", confidence=1.0))
                if kg.G.number_of_edges() > before:
                    stats["is_a_edges"] += 1

    # DATASET → TIER 1 modality edge
    # provides_modality is reused: NSD / BOLD5000 provide image, SEED-DV provides video.
    dataset_stimulus_map = {
        "DATASET:NSD":      "VS:image",
        "DATASET:BOLD5000": "VS:image",
        "DATASET:SEED_DV":  "VS:video",
    }
    for ds_id, stim_id in dataset_stimulus_map.items():
        if kg.has_concept(ds_id) and kg.has_concept(stim_id):
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(source_id=ds_id, target_id=stim_id,
                             relation_type="provides_modality",
                             source="visual_stimulus_taxonomy", confidence=1.0))
            if kg.G.number_of_edges() > before:
                stats["dataset_edges"] += 1

    logger.info(
        "visual_stimulus ingest: tier1=%d tier2=%d tier3=%d | is_a=%d evokes=%d ds=%d",
        stats["tier1_added"], stats["tier2_added"], stats["tier3_added"],
        stats["is_a_edges"], stats["evokes_edges"], stats["dataset_edges"],
    )
    return stats
