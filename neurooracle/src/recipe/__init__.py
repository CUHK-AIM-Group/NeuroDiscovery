"""IM (imaging marker) brainstorming module.

Phase 1 catalogue of imaging-derived markers, composed structurally from KG
primitives (modalities + IF:* operations + regions + tasks/concepts). The
LLM never invents an operation or modality; it picks them from the palette
and outputs follow a fixed schema validated against a static
modality<->operation compatibility table.

The previous free-form `recipe.generator` predated the atom alphabet and
mixed IMs with gene/biomarker quantities; that scope is no longer useful
once we have explicit atoms. This module keeps the same flow shape (build
palette -> brainstorm -> validate -> link -> tag) but with IM as the only
target.
"""

from .imaging_marker import (
    ImagingMarker,
    IMPalette,
    ValidationReport,
    IM_FAMILIES,
    OP_TO_MODALITIES,
    ALL_OPERATIONS as IM_ALL_OPERATIONS,
    IMAGING_MODALITIES,
    build_im_palette,
    brainstorm_ims,
    brainstorm_ims_batched,
    validate_ims,
    link_ims_to_kg,
    tag_atoms,
)
from .genetic_marker import (
    GeneticMarker,
    GMPalette,
    ValidationReport as GMValidationReport,
    GM_FAMILIES,
    OP_TO_DATA_TYPES,
    ALL_OPERATIONS as GM_ALL_OPERATIONS,
    DATA_TYPES,
    FAMILY_TO_OPS,
    CURATED_GENE_SETS,
    METHYLATION_CLOCKS,
    GTEX_BRAIN_TISSUES,
    NEURO_GWAS_SOURCES,
    build_gm_palette,
    brainstorm_gms,
    brainstorm_gms_batched,
    validate_gms,
    link_gms_to_kg,
    tag_atoms as tag_gm_atoms,
)

__all__ = [
    # imaging
    "ImagingMarker",
    "IMPalette",
    "ValidationReport",
    "IM_FAMILIES",
    "OP_TO_MODALITIES",
    "IM_ALL_OPERATIONS",
    "IMAGING_MODALITIES",
    "build_im_palette",
    "brainstorm_ims",
    "brainstorm_ims_batched",
    "validate_ims",
    "link_ims_to_kg",
    "tag_atoms",
    # genetic
    "GeneticMarker",
    "GMPalette",
    "GMValidationReport",
    "GM_FAMILIES",
    "OP_TO_DATA_TYPES",
    "GM_ALL_OPERATIONS",
    "DATA_TYPES",
    "FAMILY_TO_OPS",
    "CURATED_GENE_SETS",
    "METHYLATION_CLOCKS",
    "GTEX_BRAIN_TISSUES",
    "NEURO_GWAS_SOURCES",
    "build_gm_palette",
    "brainstorm_gms",
    "brainstorm_gms_batched",
    "validate_gms",
    "link_gms_to_kg",
    "tag_gm_atoms",
]
