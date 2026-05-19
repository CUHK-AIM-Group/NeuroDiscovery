"""Phase 1: Functional visual ROI seed nodes.

Seeds canonical functional-defined visual regions (FFA, PPA, EBA, VWFA, LOC,
MT+, V3, V4) that NeuroNames does not cover. These are essential for visual-
decoding hypotheses (NSD / BOLD5000 / SEED-DV) but are defined by fMRI
functional localizers rather than gross anatomy.

Each seed carries:
- a stable source_vocab ("visual_functional_roi")
- a list of aliases matching common paper terminology
- a `localizer` metadata key indicating the functional task used to define it
- a `parent_anatomy` cross-ref pointing at the nearest anatomical parent
  (NeuroNames node), so structural queries still work

Run as part of Phase 1 ingestion; align_graph_to_umls() skips these via the
same experiment_infra UMLS-skip mechanism (source_vocab is non-UMLS).
"""

from __future__ import annotations

import logging

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)


# id → {preferred_name, aliases, localizer, parent_anatomy_nn_id, description}
#
# Node IDs use the prefix `VROI:` to mark visual functional ROIs, keeping
# them separable from NN:* anatomical backbone while still domain-tagged as
# neuroanatomy (so they participate in existing neuroanatomy↔disease pairs).
FUNCTIONAL_ROIS: dict[str, dict] = {
    "VROI:FFA": {
        "preferred_name": "Fusiform Face Area",
        "aliases": ["FFA", "fusiform face area", "right fusiform face area",
                    "left fusiform face area"],
        "localizer": "face > object contrast",
        "parent_anatomy": "NN:305",   # Fusiform Gyrus
        "description": "Face-selective region on the lateral fusiform gyrus (Kanwisher 1997).",
    },
    "VROI:PPA": {
        "preferred_name": "Parahippocampal Place Area",
        "aliases": ["PPA", "parahippocampal place area", "place area"],
        "localizer": "scene > object contrast",
        "parent_anatomy": "NN:308",   # Parahippocampal Gyrus
        "description": "Scene/place-selective region on the collateral sulcus (Epstein 1998).",
    },
    "VROI:EBA": {
        "preferred_name": "Extrastriate Body Area",
        "aliases": ["EBA", "extrastriate body area", "body area"],
        "localizer": "body > object contrast",
        "parent_anatomy": None,       # lateral occipitotemporal, no single NN match
        "description": "Body-part-selective region in lateral occipitotemporal cortex (Downing 2001).",
    },
    "VROI:VWFA": {
        "preferred_name": "Visual Word Form Area",
        "aliases": ["VWFA", "visual word form area", "word form area"],
        "localizer": "words > consonant strings",
        "parent_anatomy": "NN:305",   # Fusiform Gyrus (left mid-fusiform)
        "description": "Word-selective region in left mid-fusiform gyrus (Cohen 2000).",
    },
    "VROI:LOC": {
        "preferred_name": "Lateral Occipital Complex",
        "aliases": ["LOC", "lateral occipital complex", "lateral occipital cortex"],
        "localizer": "object > scrambled contrast",
        # NN_HO:20022 Lateral Occipital Cortex superior division exists; use
        # the anatomical one as parent.
        "parent_anatomy": "NN:NN_HO:20022",
        "description": "Object-selective region spanning lateral occipital cortex (Malach 1995).",
    },
    "VROI:MTplus": {
        "preferred_name": "Middle Temporal Visual Area (MT+)",
        "aliases": ["MT+", "V5/MT", "hMT+", "motion-selective region",
                    "middle temporal visual area"],
        "localizer": "moving > stationary contrast",
        "parent_anatomy": "NN:303",   # Middle Temporal Gyrus
        "description": "Motion-selective region at the temporo-occipital junction (Zeki 1991).",
    },
    "VROI:V3": {
        "preferred_name": "Visual Area V3",
        "aliases": ["V3", "tertiary visual cortex", "V3v", "V3d"],
        "localizer": "retinotopic mapping",
        "parent_anatomy": "NN:401",   # Primary Visual Cortex (nearest)
        "description": "Third-tier retinotopic visual area (dorsal V3d / ventral V3v).",
    },
    "VROI:V4": {
        "preferred_name": "Visual Area V4",
        "aliases": ["V4", "hV4", "V4alpha", "color-selective region"],
        "localizer": "color / retinotopic mapping",
        "parent_anatomy": "NN:401",
        "description": "Color- and form-selective retinotopic area in ventral occipital cortex.",
    },
}


def ingest_visual_functional_roi(kg: KnowledgeGraph) -> dict:
    """Seed functional visual ROI nodes. Idempotent."""
    stats = {"rois_added": 0, "parent_edges_added": 0}

    for nid, info in FUNCTIONAL_ROIS.items():
        if kg.has_concept(nid):
            continue
        kg.add_concept(ConceptNode(
            id=nid,
            preferred_name=info["preferred_name"],
            domain_tags=[DomainTag.NEUROANATOMY.value],
            source_vocab="visual_functional_roi",
            aliases=info["aliases"],
            definition=info["description"],
            metadata={
                "localizer": info["localizer"],
                "parent_anatomy": info.get("parent_anatomy"),
                "seed_source": "NSD/BOLD5000 functional ROI convention",
            },
        ))
        stats["rois_added"] += 1

        parent = info.get("parent_anatomy")
        if parent and kg.has_concept(parent):
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=nid,
                target_id=parent,
                relation_type="part_of",
                source="visual_functional_roi",
                confidence=0.9,
            ))
            if kg.G.number_of_edges() > before:
                stats["parent_edges_added"] += 1

    logger.info(
        "visual_functional_roi ingest: %d ROIs, %d part_of edges",
        stats["rois_added"], stats["parent_edges_added"],
    )
    return stats
