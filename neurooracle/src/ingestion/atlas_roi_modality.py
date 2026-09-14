"""Spatial-reference mappings and imaging-feature modality bridges.

Two related but separable jobs:

1. Load the sealed ``spatial_mapping.v1`` resource for all 36 registered
   ``ATLAS:*`` spatial-reference nodes. Every finite brain parcellation gets an
   atlas-specific ROI node and an exact ``defines_region`` membership edge;
   EEG layouts and the continuous voxel reference remain compact structured
   mappings on their reference nodes. Four legacy crosswalks (AAL90, AAL116,
   Desikan and HarvardOxford_sub) additionally receive explicit ROI-level
   ``maps_to`` edges. This replaces the former prefix-and-sort heuristic, which
   could connect a reference to unrelated NeuroNames entries.

2. Imaging feature concept -> MODALITY (predicate `measured_by_modality`).
   Adds a small set of imaging-feature concept nodes (cortical thickness,
   surface area, regional volume, FA/MD/RD/AD, FC, ALFF, ReHo, BOLD amplitude,
   amyloid SUVR, tau SUVR, FDG uptake) and links each to the modality that
   physically produces it. This gives downstream hypothesis paths an explicit
   answer to "what scan would measure this marker"; up to now only the textual
   `metadata.modality` on ENIGMA edges carried that information.

The resource is versioned, locally sealed and source-provenanced. Ingestion is
offline and idempotent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge
from .enigma_disease_im import DK_ROI_TO_NN

logger = logging.getLogger(__name__)


# --- Sealed spatial-mapping resource ----------------------------------------

SPATIAL_MAPPING_VERSION = "spatial_mapping.v1.20260904"
SPATIAL_MAPPING_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "spatial_mappings"
    / "spatial_mapping_v1_20260904"
)
EEG_REFERENCES = {
    "ATLAS:EEG_10_20",
    "ATLAS:EEG_10_10",
    "ATLAS:EEG_SEED_62",
    "ATLAS:EEG_SEED_VIG_17",
    "ATLAS:EEG_BCI_32",
}
CROSSWALK_REFERENCES = {
    "ATLAS:AAL90",
    "ATLAS:AAL116",
    "ATLAS:Desikan",
    "ATLAS:HarvardOxford_sub",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_spatial_mapping_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and cryptographically verify the small local mapping resource."""
    seal_path = SPATIAL_MAPPING_ROOT / "RESOURCE_SEAL.json"
    reference_path = SPATIAL_MAPPING_ROOT / "REFERENCE_MAPPINGS.json"
    audit_path = SPATIAL_MAPPING_ROOT / "AUDIT_REPORT.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("resource_version") != SPATIAL_MAPPING_VERSION:
        raise RuntimeError("spatial-mapping resource version mismatch")
    if _sha256_file(reference_path) != seal.get("reference_manifest_sha256"):
        raise RuntimeError("spatial-mapping reference manifest hash mismatch")
    if _sha256_file(audit_path) != seal.get("audit_report_sha256"):
        raise RuntimeError("spatial-mapping audit hash mismatch")
    manifest = json.loads(reference_path.read_text(encoding="utf-8"))
    references = manifest.get("references") or {}
    if len(references) != 36:
        raise RuntimeError(f"expected 36 spatial mappings, found {len(references)}")
    for reference_id, summary in references.items():
        resource_path = SPATIAL_MAPPING_ROOT / str(summary["resource"])
        expected = seal.get("element_files", {}).get(resource_path.name)
        if not expected or _sha256_file(resource_path) != expected:
            raise RuntimeError(f"spatial-mapping element hash mismatch: {reference_id}")
        if summary.get("resource_sha256") != expected:
            raise RuntimeError(f"spatial-mapping summary hash mismatch: {reference_id}")
    return manifest, seal


def _iter_mapping_rows(reference_id: str, summary: dict[str, Any]) -> Iterator[dict[str, Any]]:
    path = SPATIAL_MAPPING_ROOT / str(summary["resource"])
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("reference_id") != reference_id:
                raise RuntimeError(
                    f"mapping reference mismatch at {path.name}:{line_number}"
                )
            yield row


def _reference_spatial_mapping(
    summary: dict[str, Any], seal: dict[str, Any]
) -> dict[str, Any]:
    value = dict(summary)
    value.update(
        {
            "resource_root": "neurooracle/data/spatial_mappings/spatial_mapping_v1_20260904",
            "resource_seal_sha256": seal["seal_sha256"],
            "source_manifest_sha256": seal["source_manifest_sha256"],
        }
    )
    return value


def _element_spatial_mapping(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "reference_id",
        "reference_name",
        "element_index",
        "element_type",
        "source_index",
        "hemisphere",
        "network",
        "coordinate_space",
        "centroid",
        "voxel_count",
        "spatial_support",
        "source_key",
    )
    value = {field: row.get(field) for field in fields}
    value["schema_version"] = SPATIAL_MAPPING_VERSION
    value["resource"] = f"elements/{row['reference_name']}.jsonl"
    if row.get("metadata"):
        value["metadata"] = dict(row["metadata"])
    return value


def _build_spatial_region_node(row: dict[str, Any]) -> ConceptNode:
    return ConceptNode(
        id=str(row["element_id"]),
        preferred_name=str(row["label"]),
        semantic_types=["T029"],
        domain_tags=[DomainTag.NEUROANATOMY.value],
        source_vocab="spatial_mapping",
        definition=(
            f"Element {row['element_index']} of the {row['reference_name']} "
            "spatial reference."
        ),
        aliases=list(row.get("aliases") or []),
        external_ids={
            "spatial_reference": str(row["reference_name"]),
            "source_index": str(row["source_index"]),
        },
        spatial_mapping=_element_spatial_mapping(row),
        metadata={
            "spatial_mapping_element": True,
            "spatial_reference": str(row["reference_name"]),
            "resource_version": SPATIAL_MAPPING_VERSION,
        },
    )


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _harvard_oxford_name_index(kg: KnowledgeGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for node_id, node in kg._index.items():
        if not node_id.startswith("NN:NN_HO:"):
            continue
        key = _normalized_label(node.preferred_name)
        if key in result:
            raise RuntimeError(f"duplicate Harvard-Oxford NeuroNames label: {key}")
        result[key] = node_id
    return result


def _crosswalk_target(
    kg: KnowledgeGraph,
    reference_id: str,
    row: dict[str, Any],
    ho_names: dict[str, str],
) -> tuple[str | None, str | None]:
    if reference_id in {"ATLAS:AAL90", "ATLAS:AAL116"}:
        target = f"NN:NN_AAL:{30000 + int(row['source_index'])}"
        return (target, "AAL source-index identity") if kg.has_concept(target) else (None, None)
    if reference_id == "ATLAS:HarvardOxford_sub":
        target = ho_names.get(_normalized_label(str(row["label"])))
        return (target, "exact normalized Harvard-Oxford label") if target else (None, None)
    if reference_id == "ATLAS:Desikan":
        label = re.sub(r"^[LR]\s+", "", str(row["label"]), flags=re.IGNORECASE)
        target = DK_ROI_TO_NN.get(_normalized_label(label))
        return (target, "curated Desikan-Killiany canonical crosswalk") if target and kg.has_concept(target) else (None, None)
    return None, None


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
     "aliases": ["CortThick", "cortical thickness map"]},
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
     "aliases": ["ALFF"]},
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


def ingest_atlas_roi_modality(kg: KnowledgeGraph) -> dict:
    """Add sealed spatial mappings and imaging-feature modality bridges.

    Idempotent: re-running on a populated graph adds zero new edges/nodes.
    """
    stats = {
        "spatial_references_mapped": 0,
        "finite_elements_mapped": 0,
        "roi_nodes_added": 0,
        "sensor_elements_mapped": 0,
        "continuous_references_mapped": 0,
        "legacy_atlas_roi_edges_removed": 0,
        "atlases_linked": 0,
        "atlas_roi_edges": 0,
        "crosswalk_edges": 0,
        "crosswalk_edges_by_reference": {},
        "crosswalk_unresolved": [],
        "imaging_features_added": 0,
        "if_modality_edges": 0,
        "atlases_skipped": [],
    }

    # 1. Spatial references, atlas-specific ROIs and exact crosswalks ----
    manifest, seal = _load_spatial_mapping_manifest()
    references: dict[str, dict[str, Any]] = manifest["references"]
    ho_names = _harvard_oxford_name_index(kg)
    graph_changed = False
    for reference_id, summary in references.items():
        if not kg.has_concept(reference_id):
            stats["atlases_skipped"].append(reference_id.removeprefix("ATLAS:"))
            continue

        reference_node = kg.get_concept(reference_id)
        mapping = _reference_spatial_mapping(summary, seal)
        reference_node.spatial_mapping = mapping
        kg.G.nodes[reference_id]["spatial_mapping"] = mapping
        stats["spatial_references_mapped"] += 1
        stats["atlases_linked"] += 1

        # Remove only legacy heuristic memberships. Exact atlas-specific
        # membership edges have SPATIAL_REGION targets and are retained.
        expected_prefix = f"SPATIAL_REGION:{reference_id.removeprefix('ATLAS:')}:"
        for _, target_id, edge_data in list(kg.G.out_edges(reference_id, data=True)):
            if (
                edge_data.get("relation_type") == "defines_region"
                and not target_id.startswith(expected_prefix)
            ):
                kg.G.remove_edge(reference_id, target_id)
                stats["legacy_atlas_roi_edges_removed"] += 1
                graph_changed = True

        rows = list(_iter_mapping_rows(reference_id, summary))
        stats["finite_elements_mapped"] += len(rows)
        if summary.get("continuous"):
            stats["continuous_references_mapped"] += 1
            continue
        if reference_id in EEG_REFERENCES:
            stats["sensor_elements_mapped"] += len(rows)
            continue

        crosswalk_count = 0
        for row in rows:
            node = _build_spatial_region_node(row)
            before_nodes = kg.G.number_of_nodes()
            kg.add_concept(node)
            # add_concept merges the object index but deliberately does not
            # rewrite NetworkX attributes, so synchronize both representations.
            kg.G.nodes[node.id].update(kg.get_concept(node.id).to_dict())
            if kg.G.number_of_nodes() > before_nodes:
                stats["roi_nodes_added"] += 1
                graph_changed = True

            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=reference_id,
                target_id=node.id,
                relation_type="defines_region",
                source=SPATIAL_MAPPING_VERSION,
                confidence=1.0,
                evidence_ref=(
                    f"{summary['resource']}@{summary['resource_sha256']}"
                ),
                metadata={
                    "spatial_reference": row["reference_name"],
                    "spatial_mapping_version": SPATIAL_MAPPING_VERSION,
                    "element_index": row["element_index"],
                    "source_index": row["source_index"],
                    "coordinate_space": row.get("coordinate_space"),
                    "spatial_support": row.get("spatial_support"),
                },
            ))
            if kg.G.number_of_edges() > before:
                stats["atlas_roi_edges"] += 1
                graph_changed = True

            if reference_id not in CROSSWALK_REFERENCES:
                continue
            target_id, method = _crosswalk_target(kg, reference_id, row, ho_names)
            if not target_id:
                stats["crosswalk_unresolved"].append(node.id)
                continue
            before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=node.id,
                target_id=target_id,
                relation_type="maps_to",
                source=SPATIAL_MAPPING_VERSION,
                confidence=1.0,
                evidence_ref=str(method),
                metadata={
                    "spatial_reference": row["reference_name"],
                    "mapping_method": method,
                    "source_index": row["source_index"],
                    "resource_sha256": summary["resource_sha256"],
                },
            ))
            if kg.G.number_of_edges() > before:
                stats["crosswalk_edges"] += 1
                crosswalk_count += 1
                graph_changed = True
        if reference_id in CROSSWALK_REFERENCES:
            stats["crosswalk_edges_by_reference"][reference_id] = crosswalk_count

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

    if graph_changed:
        kg.invalidate_semantic_view()

    logger.info(
        "atlas_roi_modality ingest: %d references mapped (%d ROI edges, %d crosswalks), "
        "%d imaging features (%d modality edges); skipped %s",
        stats["spatial_references_mapped"], stats["atlas_roi_edges"],
        stats["crosswalk_edges"],
        stats["imaging_features_added"], stats["if_modality_edges"],
        stats["atlases_skipped"] or "none",
    )
    return stats
