"""Knowledge graph schema: node and edge data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .paper_scope import infer_paper_scope_from_claim_dict, normalize_paper_scope


class DomainTag(str, Enum):
    """High-level domain classification for concepts."""
    NEUROANATOMY = "neuroanatomy"
    DISEASE = "disease"
    GENE = "gene"
    NEUROTRANSMITTER = "neurotransmitter"
    DRUG = "drug"
    COGNITIVE_FUNCTION = "cognitive_function"
    CELL_TYPE = "cell_type"
    BIOMARKER = "biomarker"
    PARADIGM = "paradigm"  # experimental paradigm (BrainMap)
    CONNECTIVITY = "connectivity"  # functional/structural connections
    IMAGING_FEATURE = "imaging_feature"  # cortical thickness, volume, FA, FC, SUVR, etc.
    DATASET_VARIABLE = "dataset_variable"  # genetics, environment, medication, etc.
    # Phase 1.5 Experiment infrastructure (atlas/modality/dataset/ml_model)
    # + reserved RECIPE tag (former Phase 4.3, removed 2026-05-13 but kept
    # in UMLS-skip set for forward compat).
    RECIPE = "recipe"          # reserved
    ATLAS = "atlas"            # brain parcellation (ATLAS:*)
    MODALITY = "modality"      # imaging/data modality (MODALITY:*)
    DATASET = "dataset"        # research dataset (DATASET:*)
    ML_MODEL = "ml_model"      # ML architecture (MODEL:*)
    # Brain decoding stimuli & psychological-state targets
    VISUAL_STIMULUS = "visual_stimulus"  # image/video stimulus (NSD/BOLD5000/SEED-DV)
    EMOTION = "emotion"                  # affective state label (SEED family)
    VIGILANCE = "vigilance"              # alertness/drowsiness label (SEED-VIG)
    # Treatment-effect / clinical-outcome variables (Phase 1, atom-algebra rollout
    # 2026-05-20): a finer-grained sibling of DATASET_VARIABLE used specifically
    # to tag nodes that represent quantitative drug-response or therapy outcomes
    # (e.g. ΔADAS-Cog at 12 mo, ΔMDS-UPDRS, NIHSS at 6 mo, responder/non-responder
    # labels). Distinguishes treatment-outcome variables from generic individual-data
    # variables (age, APOE-ε4 status, lifestyle) when both share dataset_variable.
    TREATMENT_OUTCOME = "treatment_outcome"
    # Concept-level anchors for INDIVIDUAL_DATA (Aging, APOE, Big-5 traits, …),
    # seeded by ingestion.individual_data_anchors. Distinct from dataset_variable
    # (which is the UKB/ADNI/HCP host category hub) so the bridge layer can
    # connect concept-side IM/disease/gene nodes to dataset variables in two
    # hops without polluting either side's domain.
    INDIVIDUAL_DATA_ANCHOR = "individual_data_anchor"


class SemanticType(str, Enum):
    """UMLS semantic types relevant to neuroscience."""
    DISEASE_OR_SYNDROME = "T047"
    MENTAL_DYSFUNCTION = "T048"
    NEOPLASTIC_PROCESS = "T191"
    BODY_PART_ORGAN = "T023"
    BODY_LOCATION = "T029"
    CELL = "T025"
    NEUROTRANSMITTER = "T116"
    AMINO_ACID_PEPTIDE = "T116"  # overlaps with neurotransmitter in UMLS
    PHARMACOLOGIC_SUBSTANCE = "T121"
    GENE_OR_GENOME = "T028"
    INTELLECTUAL_PRODUCT = "T170"


@dataclass
class ConceptNode:
    """A concept node in the knowledge graph."""
    id: str                          # unique identifier (CUI, or custom like "NN:1234")
    preferred_name: str              # standard display name
    semantic_types: list[str] = field(default_factory=list)  # TUI codes
    domain_tags: list[str] = field(default_factory=list)     # DomainTag values
    source_vocab: str = ""           # originating vocabulary (MeSH, NeuroNames, etc.)
    definition: str = ""             # text definition
    aliases: list[str] = field(default_factory=list)         # synonyms / alternate names
    external_ids: dict[str, str] = field(default_factory=dict)  # cross-references
    atlas_mapping: Optional[dict] = None  # MNI coords, atlas region ID, etc.
    metadata: dict = field(default_factory=dict)             # catch-all for extra info

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "preferred_name": self.preferred_name,
            "semantic_types": self.semantic_types,
            "domain_tags": self.domain_tags,
            "source_vocab": self.source_vocab,
            "definition": self.definition,
            "aliases": self.aliases,
            "external_ids": self.external_ids,
            "atlas_mapping": self.atlas_mapping,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConceptNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


RELATION_TYPES = {
    # taxonomic / structural
    "is_a",               # A is a subtype of B
    "part_of",            # A is anatomical part of B
    "has_part",           # inverse of part_of
    # causal / functional
    "causes",             # A causes B
    "associated_with",    # A is associated with B (loose)
    "predisposes",        # A increases risk of B
    # therapeutic
    "treats",             # A treats B
    "contraindicated_for",  # A is contraindicated for B
    # molecular / genetic
    "gene_associated_with_disease",
    "gene_associated_with_anatomy",  # gene -> neuroanatomy (HPO-derived; mutation-causes-phenotype-affecting-region)
    "gene_enriched_in_region",       # gene -> neuroanatomy (AHBA-derived; cross-donor z-score expression enrichment)
    "receptor_density_in",           # gene -> neuroanatomy (Hansen 2022 PET tracer-derived receptor density per ROI)
    "protein_encoded_by",
    "modulates",          # A modulates activity of B
    "binds_to",           # A binds to receptor B
    # neuroanatomy
    "projects_to",        # A projects neural connections to B
    "connects_to",        # structural connectivity A-B
    "activates",          # A functionally activates B
    "coactivates",        # A and B co-activate (BrainMap)
    # evidence
    "supported_by",
    "contradicts",
    "about",
    # claim predicates (from paper extraction)
    "reduces",
    "increases",
    "correlates_with",
    "is_biomarker_of",
    "is_risk_factor_for",
    "is_associated_with",
    "predicts",
    "mediates",
    "inhibits",
    "distinguishes",
    # Phase 1.5 Experiment infrastructure edges
    "supports_modality",  # Model → Modality (compat declaration)
    "provides_modality",  # Dataset → Modality (what the dataset contains)
    # Brain decoding edges (NSD/BOLD5000/SEED-DV/SEED family)
    "evokes",             # visual_stimulus → neuroanatomy (encoding direction)
    "decoded_from",       # visual_stimulus ← neuroanatomy (decoding direction)
    "elicits",            # stimulus → emotion/vigilance (behavioral label)
    # Outcome / dataset-variable bridges
    "measures",                # rating_scale → disease (the scale measures the disease)
    "assessed_in",             # rating_scale → dataset_variable host
    "affects_system",          # MedDRA SOC → disease umbrella
    "provides_signal_for",     # disease → dataset_variable (anchor → variable)
    # Therapy reverse edges (closes drug indegree gap)
    "is_indicated_for",        # drug → disease (canonical indication)
    "is_treated_by",           # disease → drug (reverse of treats; for personalised_treatment)
    # Imaging modality edges (closes IM↔Idv gap)
    "measured_in_modality",    # imaging_marker → dataset_variable (modality host)
    "modality_provides",       # dataset_variable → imaging_marker (reverse)
    # Outcome closure edges (closes IM/D/Rx → OUTCOME gap)
    "is_assessed_by",          # disease → rating_scale (reverse of measures)
    "has_adverse_effect",      # drug → AE SOC (drug → outcome)
    # Atlas / imaging-feature anchoring edges
    "defines_region",          # atlas → neuroanatomy ROI (atlas covers ROI)
    "measured_by_modality",    # imaging_feature → modality (e.g. CortThick → sMRI)
    "is_imaging_feature_of",   # imaging_feature → neuroanatomy (feature lives on ROI)
    "has_imaging_feature",     # neuroanatomy → imaging_feature (inverse, for D/G → region → IM closure)
}


# Edge tier classification — used by `KnowledgeGraph.export_display_subgraph`
# to derive a human-facing view of the KG that drops provenance, inverse
# mirrors, and admin/bridge edges. The full graph (self.G) is unchanged and
# the hypothesis engine still uses every edge.
#
# Tiers:
#   "discovery" — direct semantic claim, surface to readers (claim predicates,
#                 canonical pharmacology / connectivity / scale relations)
#   "skeleton"  — taxonomy/hierarchy (is_a, part_of), usually rendered as a
#                 tree, not as graph edges; opt-in via `include_skeleton=True`
#   "inverse"   — inverse mirror of a canonical Tier-1 edge, kept for reverse
#                 path traversal in the hypothesis engine; never display
#   "bridge"    — admin/bridge edge that connects atom families through hub
#                 nodes (dataset-variable, disease-umbrella); never display
#   "provenance"— claim→subject/object 'about' edges, support links; never display
#   "deprecated"— removed Phase 4.3 relations kept in RELATION_TYPES for
#                 forward-compat; never display
#
# Relations not listed here default to "discovery" so the long-tail of
# claim-extracted predicates ("improves", "disrupts", …) is still surfaced.
EDGE_TIER: dict[str, str] = {
    # Tier 1 — discovery
    "reduces":                       "discovery",
    "increases":                     "discovery",
    "correlates_with":               "discovery",
    "causes":                        "discovery",
    "is_biomarker_of":               "discovery",
    "is_risk_factor_for":            "discovery",
    "is_associated_with":            "discovery",
    "associated_with":               "discovery",
    "predicts":                      "discovery",
    "mediates":                      "discovery",
    "inhibits":                      "discovery",
    "distinguishes":                 "discovery",
    "contradicts":                   "discovery",
    "predisposes":                   "discovery",
    "contraindicated_for":           "discovery",
    "gene_associated_with_disease":  "discovery",
    "gene_associated_with_anatomy":  "discovery",
    "gene_enriched_in_region":       "discovery",
    "receptor_density_in":           "discovery",
    "treats":                        "discovery",
    "protein_encoded_by":            "discovery",
    "binds_to":                      "discovery",
    "modulates":                     "discovery",
    "projects_to":                   "discovery",
    "connects_to":                   "discovery",
    "activates":                     "discovery",
    "coactivates":                   "discovery",
    "evokes":                        "discovery",
    "elicits":                       "discovery",
    "measures":                      "discovery",
    # Tier 2 — skeleton
    "is_a":                          "skeleton",
    "part_of":                       "skeleton",
    "supports_modality":             "skeleton",
    "provides_modality":             "skeleton",
    "defines_region":                "skeleton",
    "measured_by_modality":          "skeleton",
    "is_imaging_feature_of":         "skeleton",
    # Tier 3 — inverse mirrors of canonical Tier-1 edges
    "has_imaging_feature":           "inverse",
    "is_treated_by":                 "inverse",
    "is_assessed_by":                "inverse",
    "modality_provides":             "inverse",
    "has_part":                      "inverse",
    "decoded_from":                  "inverse",
    # Tier 3 — admin / bridge
    "assessed_in":                   "bridge",
    "affects_system":                "bridge",
    "provides_signal_for":           "bridge",
    "measured_in_modality":          "bridge",
    "has_adverse_effect":            "bridge",
    # Tier 3 — provenance
    "about":                         "provenance",
    "supported_by":                  "provenance",
}

DISPLAY_TIERS_DEFAULT: frozenset[str] = frozenset({"discovery"})

# Claim-specific predicates (extracted from papers)
CLAIM_PREDICATES = {
    "reduces",              # A reduces B
    "increases",            # A increases B
    "correlates_with",      # A correlates with B
    "causes",               # A causes B
    "is_biomarker_of",      # A is a biomarker for B
    "is_risk_factor_for",   # A is a risk factor for B
    "treats",               # A treats B
    "modulates",            # A modulates B
    "activates",            # A activates B
    "inhibits",             # A inhibits B
    "predicts",             # A predicts B
    "mediates",             # A mediates the relationship between B and C
    "is_associated_with",   # A is associated with B
    "distinguishes",        # A distinguishes B from C
}


@dataclass
class Edge:
    """A directed edge in the knowledge graph."""
    source_id: str                   # source ConceptNode.id
    target_id: str                   # target ConceptNode.id
    relation_type: str               # one of RELATION_TYPES
    source: str = ""                 # provenance: 'NeuroNames', 'MeSH', 'DisGeNET', etc.
    confidence: float = 1.0          # 0.0-1.0
    evidence_ref: str = ""           # citation or reference
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "source": self.source,
            "confidence": self.confidence,
            "evidence_ref": self.evidence_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Evidence:
    """Experimental evidence supporting a scientific claim."""
    study_type: str = ""             # "fMRI", "lesion", "meta-analysis", "GWAS", "animal_model"
    methodology: str = ""            # "resting-state FC", "voxel-based morphometry", "DTI", ...
    p_value: Optional[float] = None
    effect_size: Optional[float] = None      # Cohen's d, r, OR, beta
    effect_metric: str = ""          # "Cohen's d", "r", "OR", "beta", "AUC"
    sample_size: Optional[int] = None
    replicability: str = "single_study"  # "replicated", "single_study", "controversial"
    direction: str = ""              # "positive", "negative"

    def to_dict(self) -> dict:
        return {
            "study_type": self.study_type,
            "methodology": self.methodology,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_metric": self.effect_metric,
            "sample_size": self.sample_size,
            "replicability": self.replicability,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PaperRef:
    """Reference to a source paper."""
    pmid: str = ""                   # PubMed ID
    doi: str = ""
    title: str = ""
    authors: str = ""
    year: Optional[int] = None
    journal: str = ""

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaperRef:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Claim:
    """A structured scientific claim extracted from a paper.

    A claim is both stored as a node (for detailed querying) and
    generates simplified edges (for multi-hop traversal).
    """
    id: str                              # CLM:uuid
    subject_id: str                      # ConceptNode.id in the graph
    subject_name: str                    # human-readable subject name
    predicate: str                       # one of CLAIM_PREDICATES
    object_id: str                       # ConceptNode.id in the graph
    object_name: str                     # human-readable object name
    negated: bool = False                # "X does NOT affect Y"
    confidence: float = 0.5              # overall confidence 0-1
    evidence: Evidence = field(default_factory=Evidence)
    source_paper: PaperRef = field(default_factory=PaperRef)
    raw_text: str = ""                   # original sentence from paper
    paper_scope: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        claim_data = {
            "id": self.id,
            "metadata": self.metadata,
            "paper_scope": self.paper_scope,
        }
        paper_scope = self.paper_scope or infer_paper_scope_from_claim_dict(claim_data)
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "object_name": self.object_name,
            "negated": self.negated,
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
            "source_paper": self.source_paper.to_dict(),
            "raw_text": self.raw_text,
            "paper_scope": paper_scope,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        d = d.copy()
        if "evidence" in d and isinstance(d["evidence"], dict):
            d["evidence"] = Evidence.from_dict(d["evidence"])
        if "source_paper" in d and isinstance(d["source_paper"], dict):
            d["source_paper"] = PaperRef.from_dict(d["source_paper"])
        if "paper_scope" in d:
            d["paper_scope"] = normalize_paper_scope(d["paper_scope"])
        else:
            d["paper_scope"] = infer_paper_scope_from_claim_dict(d)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_edge(self) -> Edge:
        """Convert claim to a simplified graph edge for traversal."""
        return Edge(
            source_id=self.subject_id,
            target_id=self.object_id,
            relation_type=self.predicate,
            source=f"claim:{self.source_paper.pmid or self.id}",
            confidence=self.confidence,
            evidence_ref=self.source_paper.title,
            metadata={"claim_id": self.id, "negated": self.negated},
        )
