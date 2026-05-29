"""Bridge edges that anchor the IM (imaging marker) layer to the rest of the KG.

Two related but separable jobs:

1. ATLAS -> NN region (predicate `defines_region`).
   Every ATLAS:* node ingested by `experiment_infra` was tree-only until now -
   the Phase 1 audit found 16/16 atlas nodes with zero non-tree edges. We close
   that by mapping each parcellation onto the NeuroNames ROI nodes that
   represent its parcels, leveraging the fact that NeuroNames already segregates
   atlas-specific labels into sub-namespaces (`NN:NN_AAL:*`, `NN:NN_HO:*`,
   `NN:NN_TAL:*`, `NN:NN_PAULI:*`). Desikan-Killiany is the exception: its 34
   cortical ROIs map onto canonical NN primary nodes via the same lookup table
   ENIGMA-disease-IM uses, so we reuse `enigma_disease_im.DK_ROI_TO_NN` /
   `ASEG_ROI_TO_NN` to avoid drift.

2. Imaging feature concept -> MODALITY (predicate `measured_by_modality`).
   Adds a small set of imaging-feature concept nodes (cortical thickness,
   surface area, regional volume, FA/MD/RD/AD, FC, ALFF, ReHo, BOLD amplitude,
   amyloid SUVR, tau SUVR, FDG uptake) and links each to the modality that
   physically produces it. This gives downstream hypothesis paths an explicit
   answer to "what scan would measure this marker"; up to now only the textual
   `metadata.modality` on ENIGMA edges carried that information.

Both jobs are pure static metadata - no downloads, idempotent, run in
milliseconds. They live in one module because they share the IM-anchoring
goal and would otherwise produce two trivial files.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge
from .enigma_disease_im import ASEG_ROI_TO_NN, DK_ROI_TO_NN

logger = logging.getLogger(__name__)


# --- ATLAS -> ROI maps -------------------------------------------------------

# Atlases whose ROI list maps onto a single NN sub-namespace prefix.
# AAL90 / AAL116 share the same NN_AAL pool because NeuroNames does not
# version the AAL atlas internally; we record the parcellation count in
# edge metadata so downstream code can disambiguate.
ATLAS_NN_PREFIX_MAP: dict[str, dict] = {
    "AAL90":               {"prefix": "NN:NN_AAL:",   "max_regions": 90,
                            "note": "AAL 90-region cortical+subcortical"},
    "AAL116":              {"prefix": "NN:NN_AAL:",   "max_regions": 116,
                            "note": "AAL 116-region (adds 26 cerebellar)"},
    "HarvardOxford_sub":   {"prefix": "NN:NN_HO:",    "max_regions": None,
                            "note": "Harvard-Oxford full coverage"},
}

# Desikan-Killiany ROIs map onto canonical NN primary nodes via the
# enigma_disease_im lookup tables. We dedupe targets so an atlas does not
# get duplicate edges if two ROI codes share a canonical NN id (e.g.
# medial/lateral OFC both -> NN:102).
DESIKAN_NN_TARGETS: list[str] = sorted({*DK_ROI_TO_NN.values()})
ASEG_NN_TARGETS:    list[str] = sorted({*ASEG_ROI_TO_NN.values()})


# --- Imaging-feature -> Modality table --------------------------------------

#: Each entry: (concept_id, preferred_name, description, modality, aliases).
#: These nodes get domain `imaging_feature`, source `experiment_infra`, and
#: are the canonical anchors when a hypothesis says "the marker X reduces
#: in disease D" - they make explicit what scan produces X.
IMAGING_FEATURES: list[dict] = [
    # Structural MRI
    {"id": "IF:cortical_thickness", "name": "cortical thickness",
     "modality": "sMRI",
     "desc": "vertex-wise distance from white-grey to pial surface (mm)",
     "aliases": ["CortThick", "cortical thickness map", "CT"]},
    {"id": "IF:cortical_surface_area", "name": "cortical surface area",
     "modality": "sMRI",
     "desc": "vertex-wise white-matter or pial surface area (mm^2)",
     "aliases": ["CortSurf", "cortical surface area map", "SA"]},
    {"id": "IF:regional_volume", "name": "regional volume",
     "modality": "sMRI",
     "desc": "ICV-corrected volume of subcortical / cortical ROI (mm^3)",
     "aliases": ["SubVol", "regional brain volume", "ROI volume"]},
    {"id": "IF:gray_matter_density", "name": "gray matter density",
     "modality": "sMRI",
     "desc": "voxel-based morphometry gray-matter probability density",
     "aliases": ["VBM", "GM density", "gray matter volume (VBM)"]},
    # Diffusion MRI
    {"id": "IF:fractional_anisotropy", "name": "fractional anisotropy",
     "modality": "dMRI",
     "desc": "DTI scalar; directional coherence of water diffusion",
     "aliases": ["FA", "DTI FA"]},
    {"id": "IF:mean_diffusivity", "name": "mean diffusivity",
     "modality": "dMRI",
     "desc": "DTI scalar; mean apparent diffusion coefficient",
     "aliases": ["MD", "DTI MD"]},
    {"id": "IF:radial_diffusivity", "name": "radial diffusivity",
     "modality": "dMRI",
     "desc": "DTI scalar; perpendicular diffusion (myelin marker)",
     "aliases": ["RD", "DTI RD"]},
    {"id": "IF:axial_diffusivity", "name": "axial diffusivity",
     "modality": "dMRI",
     "desc": "DTI scalar; principal-axis diffusion (axonal marker)",
     "aliases": ["AD", "DTI AD"]},
    # Functional MRI
    {"id": "IF:functional_connectivity", "name": "functional connectivity",
     "modality": "fMRI",
     "desc": "BOLD time-series correlation between ROI pairs (rs-fMRI)",
     "aliases": ["FC", "rs-FC", "resting-state functional connectivity"]},
    {"id": "IF:alff", "name": "amplitude of low-frequency fluctuation",
     "modality": "fMRI",
     "desc": "0.01-0.08 Hz BOLD power amplitude per voxel/ROI",
     "aliases": ["ALFF", "fALFF"]},
    {"id": "IF:reho", "name": "regional homogeneity",
     "modality": "fMRI",
     "desc": "Kendall coefficient of concordance among neighbour voxels",
     "aliases": ["ReHo"]},
    {"id": "IF:bold_amplitude", "name": "task BOLD amplitude",
     "modality": "fMRI",
     "desc": "task-evoked BOLD response magnitude (GLM beta)",
     "aliases": ["BOLD amplitude", "GLM beta", "task activation"]},
    # PET
    {"id": "IF:amyloid_suvr", "name": "amyloid SUVR",
     "modality": "PET",
     "desc": "amyloid-PET standardised uptake value ratio (Pittsburgh-B / Florbetapir)",
     "aliases": ["amyloid PET SUVR", "PiB SUVR", "florbetapir SUVR"]},
    {"id": "IF:tau_suvr", "name": "tau SUVR",
     "modality": "PET",
     "desc": "tau-PET standardised uptake value ratio (AV-1451 / flortaucipir)",
     "aliases": ["tau PET SUVR", "AV-1451 SUVR", "flortaucipir SUVR"]},
    {"id": "IF:fdg_uptake", "name": "FDG uptake",
     "modality": "PET",
     "desc": "regional cerebral metabolic rate of glucose (FDG-PET)",
     "aliases": ["FDG PET", "FDG-PET", "CMRglc"]},
]


def _build_imaging_feature_node(spec: dict) -> ConceptNode:
    return ConceptNode(
        id=spec["id"],
        preferred_name=spec["name"],
        domain_tags=[DomainTag.IMAGING_FEATURE.value],
        source_vocab="experiment_infra",
        aliases=list(spec.get("aliases", [])),
        definition=spec["desc"],
        metadata={"modality": spec["modality"]},
    )


def _atlas_targets(kg: KnowledgeGraph, atlas_name: str) -> list[str]:
    """Resolve the NN node ids that the given atlas covers."""
    if atlas_name == "Desikan":
        return [n for n in DESIKAN_NN_TARGETS if kg.has_concept(n)]
    if atlas_name == "Aseg":  # not currently a registered atlas, kept for symmetry
        return [n for n in ASEG_NN_TARGETS if kg.has_concept(n)]
    spec = ATLAS_NN_PREFIX_MAP.get(atlas_name)
    if spec is None:
        return []
    prefix = spec["prefix"]
    cap = spec.get("max_regions")
    targets = sorted([nid for nid in kg.G.nodes() if nid.startswith(prefix)])
    if cap is not None:
        targets = targets[:cap]
    return targets


def ingest_atlas_roi_modality(kg: KnowledgeGraph) -> dict:
    """Add atlas->ROI and imaging_feature->modality bridge edges.

    Idempotent: re-running on a populated graph adds zero new edges/nodes.
    """
    stats = {
        "atlases_linked":         0,
        "atlas_roi_edges":        0,
        "imaging_features_added": 0,
        "if_modality_edges":      0,
        "atlases_skipped":        [],
    }

    # 1. ATLAS -> ROI ----------------------------------------------------
    atlas_specs: list[tuple[str, str]] = [
        ("Desikan", "Desikan"),
        ("AAL90", "AAL90"),
        ("AAL116", "AAL116"),
        ("HarvardOxford_sub", "HarvardOxford_sub"),
    ]
    for atlas_name, lookup_name in atlas_specs:
        atlas_id = f"ATLAS:{atlas_name}"
        if not kg.has_concept(atlas_id):
            stats["atlases_skipped"].append(atlas_name)
            continue
        targets = _atlas_targets(kg, lookup_name)
        if not targets:
            stats["atlases_skipped"].append(atlas_name)
            continue
        stats["atlases_linked"] += 1
        for nn_id in targets:
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=atlas_id,
                target_id=nn_id,
                relation_type="defines_region",
                source="experiment_infra",
                confidence=1.0,
                evidence_ref=f"{atlas_name} parcellation membership",
                metadata={"atlas": atlas_name},
            ))
            if kg.G.number_of_edges() > before:
                stats["atlas_roi_edges"] += 1

    # 2. Imaging features -> Modality ------------------------------------
    for spec in IMAGING_FEATURES:
        node = _build_imaging_feature_node(spec)
        if not kg.has_concept(node.id):
            kg.add_concept(node)
            stats["imaging_features_added"] += 1
        mod_id = f"MODALITY:{spec['modality']}"
        if not kg.has_concept(mod_id):
            continue
        before = kg.G.number_of_edges()
        kg.add_edge(Edge(
            source_id=node.id,
            target_id=mod_id,
            relation_type="measured_by_modality",
            source="experiment_infra",
            confidence=1.0,
            evidence_ref=f"{spec['name']} is produced by {spec['modality']}",
        ))
        if kg.G.number_of_edges() > before:
            stats["if_modality_edges"] += 1

    logger.info(
        "atlas_roi_modality ingest: %d atlases linked (%d ROI edges), "
        "%d imaging features (%d modality edges); skipped %s",
        stats["atlases_linked"], stats["atlas_roi_edges"],
        stats["imaging_features_added"], stats["if_modality_edges"],
        stats["atlases_skipped"] or "none",
    )
    return stats
