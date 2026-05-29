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
        "preferred_name": "Fusiform Face Area (FFA, visual ROI)",
        "aliases": ["FFA", "fusiform face area", "right fusiform face area",
                    "left fusiform face area", "Fusiform Face Area"],
        "localizer": "face > object contrast",
        "parent_anatomy": "NN:305",   # Fusiform Gyrus
        "description": "Face-selective region on the lateral fusiform gyrus (Kanwisher 1997).",
    },
    "VROI:PPA": {
        "preferred_name": "Parahippocampal Place Area (PPA, visual ROI)",
        "aliases": ["PPA", "parahippocampal place area", "place area",
                    "Parahippocampal Place Area"],
        "localizer": "scene > object contrast",
        "parent_anatomy": "NN:308",   # Parahippocampal Gyrus
        "description": "Scene/place-selective region on the collateral sulcus (Epstein 1998).",
    },
    "VROI:EBA": {
        "preferred_name": "Extrastriate Body Area (EBA, visual ROI)",
        "aliases": ["EBA", "extrastriate body area", "body area",
                    "Extrastriate Body Area"],
        "localizer": "body > object contrast",
        "parent_anatomy": None,       # lateral occipitotemporal, no single NN match
        "description": "Body-part-selective region in lateral occipitotemporal cortex (Downing 2001).",
    },
    "VROI:VWFA": {
        "preferred_name": "Visual Word Form Area (VWFA, visual ROI)",
        "aliases": ["VWFA", "visual word form area", "word form area",
                    "Visual Word Form Area"],
        "localizer": "words > consonant strings",
        "parent_anatomy": "NN:305",   # Fusiform Gyrus (left mid-fusiform)
        "description": "Word-selective region in left mid-fusiform gyrus (Cohen 2000).",
    },
    "VROI:LOC": {
        "preferred_name": "Lateral Occipital Complex (LOC, visual ROI)",
        "aliases": ["LOC", "lateral occipital complex", "lateral occipital cortex",
                    "Lateral Occipital Complex"],
        "localizer": "object > scrambled contrast",
        # NN_HO:20022 Lateral Occipital Cortex superior division exists; use
        # the anatomical one as parent.
        "parent_anatomy": "NN:NN_HO:20022",
        "description": "Object-selective region spanning lateral occipital cortex (Malach 1995).",
    },
    "VROI:MTplus": {
        "preferred_name": "MT+ (visual motion area)",
        "aliases": ["MT+", "V5/MT", "hMT+", "motion-selective region",
                    "middle temporal visual area",
                    "Middle Temporal Visual Area",
                    "Middle Temporal Visual Area (MT+)"],
        "localizer": "moving > stationary contrast",
        "parent_anatomy": "NN:303",   # Middle Temporal Gyrus
        "description": "Motion-selective region at the temporo-occipital junction (Zeki 1991).",
    },
    "VROI:V3": {
        "preferred_name": "V3 (visual area)",
        "aliases": ["V3", "tertiary visual cortex", "V3v", "V3d",
                    "Visual Area V3", "third visual area"],
        "localizer": "retinotopic mapping",
        "parent_anatomy": "NN:401",   # Primary Visual Cortex (nearest)
        "description": "Third-tier retinotopic visual area (dorsal V3d / ventral V3v).",
    },
    "VROI:V4": {
        "preferred_name": "V4 (visual area)",
        "aliases": ["V4", "hV4", "V4alpha", "color-selective region",
                    "Visual Area V4", "fourth visual area"],
        "localizer": "color / retinotopic mapping",
        "parent_anatomy": "NN:401",
        "description": "Color- and form-selective retinotopic area in ventral occipital cortex.",
    },
}


def ingest_visual_functional_roi(kg: KnowledgeGraph) -> dict:
    """Seed functional visual ROI nodes. Idempotent.

    Cross-source name-collision handling: if a same-name node already exists
    (e.g. claim-extracted `fusiform face area`), the seed's aliases / tags
    / metadata are merged into that existing node rather than skipped, so
    downstream traversal still benefits from the seeded localizer + parent
    anatomy. Disambiguating preferred_name suffixes ('(visual ROI)' /
    '(visual area)') prevent traversal from confusing these with same-token
    nodes elsewhere in the KG.
    """
    stats = {"rois_added": 0, "rois_merged": 0, "parent_edges_added": 0}

    for nid, info in FUNCTIONAL_ROIS.items():
        if kg.has_concept(nid):
            continue

        seed_metadata = {
            "localizer": info["localizer"],
            "parent_anatomy": info.get("parent_anatomy"),
            "seed_source": "NSD/BOLD5000 functional ROI convention",
        }

        # Detect cross-source collisions on each alias (claim_extraction
        # often pre-creates `fusiform face area` etc. as long-name claim
        # subjects, which we want to enrich, not duplicate).
        merged_into = None
        for alias in [info["preferred_name"], *info["aliases"]]:
            collisions = kg.find_by_name_exact(
                alias,
                exclude_source_vocab="visual_functional_roi",
                exclude_id_prefixes=("CLAIM:",),
            )
            if collisions:
                merged_into = collisions[0]
                break

        if merged_into is not None:
            kg.merge_seed_into_existing(
                merged_into.id,
                seed_aliases=[info["preferred_name"], *info["aliases"]],
                seed_metadata={**seed_metadata, "aliased_from_seed": nid},
                seed_domain_tags=[DomainTag.NEUROANATOMY.value],
            )
            stats["rois_merged"] += 1
            parent_for_edge = merged_into.id
        else:
            kg.add_concept(ConceptNode(
                id=nid,
                preferred_name=info["preferred_name"],
                domain_tags=[DomainTag.NEUROANATOMY.value],
                source_vocab="visual_functional_roi",
                aliases=info["aliases"],
                definition=info["description"],
                metadata=seed_metadata,
            ))
            stats["rois_added"] += 1
            parent_for_edge = nid

        parent = info.get("parent_anatomy")
        if parent and kg.has_concept(parent):
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=parent_for_edge,
                target_id=parent,
                relation_type="part_of",
                source="visual_functional_roi",
                confidence=0.9,
            ))
            if kg.G.number_of_edges() > before:
                stats["parent_edges_added"] += 1

    logger.info(
        "visual_functional_roi ingest: %d new + %d merged ROIs, %d part_of edges",
        stats["rois_added"], stats["rois_merged"], stats["parent_edges_added"],
    )
    return stats
