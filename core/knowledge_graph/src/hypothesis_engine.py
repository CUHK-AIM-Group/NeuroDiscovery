"""Hypothesis engine: batch-generate, persist, and rank testable hypotheses.

Phase 3 of the NeuroClaw discovery loop:
  1. batch_generate() — traverse the graph to produce hypotheses at scale
  2. save / load — persist hypotheses to JSON
  3. rank_hypotheses() — sort by novelty, evidence, testability, confidence
  4. (Phase 5-6) hypotheses become executable NeuroClaw analysis tasks

Usage:
    from core.knowledge_graph import load_graph, HypothesisEngine

    kg = load_graph()
    engine = HypothesisEngine(kg)

    # batch generate across all domain pairs
    hypotheses = engine.batch_generate()
    engine.save_hypotheses(hypotheses, "data/hypotheses.json")

    # or load and re-rank
    hypotheses = engine.load_hypotheses("data/hypotheses.json")
    ranked = engine.rank_hypotheses(hypotheses)
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode

logger = logging.getLogger(__name__)

# ── data structures ────────────────────────────────────────────────────

@dataclass
class HypothesisLink:
    """A single step in a hypothesis chain."""
    from_id: str
    from_name: str
    to_id: str
    to_name: str
    relation_type: str
    confidence: float
    claim_id: str = ""
    raw_text: str = ""
    evidence: dict = field(default_factory=dict)
    source_paper: dict = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A generated hypothesis with full evidence chain."""
    id: str = ""
    hypothesis_type: str = ""  # "path", "bridge", "gap", "contradiction"
    source_id: str = ""
    source_name: str = ""
    target_id: str = ""
    target_name: str = ""
    path: list[HypothesisLink] = field(default_factory=list)
    confidence_score: float = 0.0
    novelty_score: float = 0.0
    evidence_score: float = 0.0
    testability_score: float = 0.0
    composite_score: float = 0.0
    supporting_claims: list[str] = field(default_factory=list)
    explanation: str = ""
    testability_reason: str = ""
    metadata: dict = field(default_factory=dict)
    critic_score: float = 0.0
    critic_feedback: list[dict] = field(default_factory=list)
    critic_rounds: int = 0
    evolve_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Hypothesis:
        d = d.copy()
        if "path" in d and isinstance(d["path"], list):
            d["path"] = [HypothesisLink(**p) if isinstance(p, dict) else p for p in d["path"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Contradiction:
    """A pair of conflicting claims."""
    concept_a_id: str = ""
    concept_a_name: str = ""
    concept_b_id: str = ""
    concept_b_name: str = ""
    claim_for_id: str = ""
    claim_for_predicate: str = ""
    claim_for_text: str = ""
    claim_against_id: str = ""
    claim_against_predicate: str = ""
    claim_against_text: str = ""
    severity: float = 0.0


@dataclass
class Gap:
    """An unexplored relationship between two concepts."""
    concept_a_id: str = ""
    concept_a_name: str = ""
    concept_b_id: str = ""
    concept_b_name: str = ""
    distance: int = 0
    connecting_concepts: list[str] = field(default_factory=list)
    domain_a: str = ""
    domain_b: str = ""
    potential_relation: str = ""


# ── constants ──────────────────────────────────────────────────────────

OPPOSING_PREDICATES = {
    ("increases", "reduces"),
    ("reduces", "increases"),
    ("causes", "inhibits"),
    ("inhibits", "causes"),
    ("treats", "contraindicated_for"),
    ("contraindicated_for", "treats"),
    ("activates", "inhibits"),
    ("inhibits", "activates"),
}

STUDY_TYPE_WEIGHTS = {
    "meta_analysis": 1.0,
    "clinical_trial": 0.9,
    "longitudinal": 0.85,
    "cohort": 0.85,
    "case_control": 0.8,
    "systematic_review": 0.8,
    "PET": 0.8, "fMRI": 0.8, "EEG": 0.8, "sMRI": 0.8,
    "cross_sectional": 0.7,
    "animal_model": 0.6,
    "review": 0.3,
    "narrative_review": 0.2,
}

# Review-only study types (no independent empirical evidence)
_REVIEW_TYPES = {"review", "narrative_review", "systematic_review"}

COMMON_RELATIONS = {"is_a", "part_of", "associated_with", "about", "is_associated_with"}

# Noisy entity name patterns — hypotheses involving these are low quality
_NOISE_WORDS = frozenset({
    "unseen", "risk", "effect", "level", "status", "change", "type",
    "group", "factor", "model", "method", "unknown", "other", "none",
    "miscellaneous", "various", "difference", "increase", "decrease",
})

NOISE_PATTERNS = [
    re.compile(r"^[A-Z][a-z]?$"),                                  # 1-2 letter: "Id", "Ca", "Mg"
    re.compile(r"^[A-Z][a-z]{2,4}$"),                              # Short mixed-case: "Tics", "Risk"
    re.compile(r"^\d+$"),                                           # Pure numbers
]

# Vague relation types that add little signal
VAGUE_RELATIONS = {"is_associated_with", "associated_with", "about"}

# domain pairs worth exploring — aligned with NeuroClaw imaging experiments
# target datasets: UKB (T1w/dMRI/rfMRI/SWI), ADNI (T1w/PET/fMRI/DTI), HCP-YA (T1w/T2w/fMRI/dMRI/MEG)
# experiment models: BrainGNN, NeuroStorm, SVM, XGBoost on raw images + handcrafted features
DEFAULT_DOMAIN_PAIRS = [
    # core: brain region features → disease (volume, thickness, activation, connectivity)
    ("neuroanatomy", "disease"),
    ("neuroanatomy", "cognitive_function"),
    ("cognitive_function", "disease"),
    ("disease", "disease"),
    # imaging biomarkers (PET amyloid/tau/FDG, sMRI volumetrics)
    ("disease", "biomarker"),
    # genetics → brain structure (UKB 500k WGS + imaging)
    ("gene", "disease"),
    ("gene", "neuroanatomy"),
    # drug effects on brain (limited but testable with PET in ADNI)
    ("disease", "drug"),
    ("drug", "disease"),
]

# Domains that are NOT directly measurable from brain imaging
# These hypotheses will be filtered out in post_process
NON_MEASURABLE_BIOMARKER_TYPES = {
    "neurotransmitter",   # needs specialized PET tracers (e.g., 11C-raclopride for DA)
    "protein",            # needs tissue biopsy or CSF
    "enzyme",             # needs molecular assays
    "receptor",           # needs specialized PET (e.g., 11C-PIB for Aβ, but that's biomarker domain)
    # fluid biomarkers — not available in UKB/HCP-YA, only ADNI CSF subset
    "csf_biomarker",
    "blood_biomarker",
    "saliva_biomarker",
    "tear_biomarker",
}

# Specific entity name patterns that are NOT directly measurable from imaging
_NON_MEASURABLE_PATTERNS = [
    re.compile(r"(neurotransmitter|dopamine|serotonin|norepinephrine|gaba|glutamate|acetylcholine)\s+(level|concentration|release|synthesis)", re.I),
    re.compile(r"(alpha|beta|gamma|delta|kappa)\s*synuclein\s*(pathology|aggregation|expression)", re.I),
    re.compile(r"(amyloid|tau|phosphorylated)\s*(beta|protein|peptide)\s*(aggregation|production|clearance)", re.I),
    re.compile(r"(enzyme|kinase|phosphatase)\s*(activity|expression)", re.I),
    re.compile(r"(receptor|transporter)\s*(density|binding|expression)", re.I),
    re.compile(r"(TNF|interleukin|IL-\d|cytokine|chemokine)\s*(alpha|beta|level|concentration|production)", re.I),
    re.compile(r"CSF\s+(Aβ|amyloid|tau|p-tau|NFL|neurofilament)", re.I),
    re.compile(r"(blood|plasma|serum)\s+(biomarker|marker|level|concentration)", re.I),
    re.compile(r"(CSF|cerebrospinal fluid)\s+", re.I),
    re.compile(r"(saliva|tear|urine)\s+(biomarker|marker|level)", re.I),
    re.compile(r"(biopsy|tissue sample)", re.I),
]

# Non-neurological target domains — brain regions should not directly predict these
_NON_NEUROLOGICAL_TARGETS = re.compile(
    r"(urinary|incontinence|frequency|enuresis|bladder|renal|kidney|liver|"
    r"gastrointestinal|cardiac|pulmonary|dermatol|orthopedic|musculoskeletal|"
    r"fracture|sprain|tumor|cancer|carcinoma|leukemia|lymphoma)", re.I
)

# NeuroClaw testable modalities and their keywords
# Aligned with UKB/ADNI/HCP-YA available data + deep learning models
TESTABLE_MODALITIES = {
    "sMRI": ["cortical thickness", "volume", "atrophy", "gray matter", "white matter",
             "brain structure", "morphometry", "VBM", "FreeSurfer", "recon-all",
             "brain region", "hippocampus", "amygdala", "thalamus", "caudate",
             "putamen", "cerebellum", "insula", "cortex", "ventricle"],
    "fMRI": ["functional connectivity", "BOLD", "activation", "resting-state",
             "task-based", "network", "default mode", "fMRI", "brain response",
             "neural activity", "brain activation"],
    "dMRI": ["DTI", "diffusion", "fractional anisotropy", "tractography",
             "white matter integrity", "structural connectivity", "FA", "MD",
             "connectivity matrix", "fiber bundle", "white matter tract"],
    "PET": ["PET", "tracer", "amyloid", "tau", "FDG", "SUVr", "binding potential",
            "glucose metabolism", "florbetapir", "flortaucipir"],
    "EEG": ["EEG", "ERP", "oscillation", "power spectrum", "alpha", "beta", "theta",
            "delta", "gamma", "microstate", "coherence", "event-related"],
    "organ_volume": ["organ volume", "liver volume", "kidney volume", "spleen volume",
                     "MedSAM", "segmentation", "organ size"],
}

# Deep learning model keywords for testability scoring
DL_MODEL_KEYWORDS = [
    "BrainGNN", "NeuroStorm", "GNN", "graph neural", "region of interest", "ROI",
    "connectivity matrix", "adjacency", "node feature", "graph convolution",
    "deep learning", "CNN", "ResNet", "attention", "transformer",
    "voxel", "patch", "whole-brain",
]

# ── Dataset-Available Variables ──────────────────────────────────────
# Defines what can be measured in each dataset. Hypotheses must start
# from these features and end at dataset-available outcomes.

DATASET_FEATURES = {
    "UKB": {
        # sMRI (T1w): FreeSurfer-derived ROI measures
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_subcortical_volume": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_cortical_area":     {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_cortical_volume":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_voxel":             {"modality": "sMRI", "tool": "voxel",       "level": "voxel"},
        # dMRI: diffusion metrics per tract
        "dmri_fa":  {"modality": "dMRI", "tool": "TBSS", "level": "tract"},
        "dmri_md":  {"modality": "dMRI", "tool": "TBSS", "level": "tract"},
        "dmri_sc":  {"modality": "dMRI", "tool": "tractography", "level": "connectivity"},
        # rfMRI: functional connectivity
        "rfmri_fc": {"modality": "fMRI", "tool": "rfMRI", "level": "connectivity"},
        # lesion segmentation
        "lesion_volume": {"modality": "sMRI", "tool": "MedSAM", "level": "ROI"},
        # non-imaging
        "genetics":       {"modality": "genetics",    "tool": "WGS/GSA",     "level": "SNP"},
        "environment":    {"modality": "environment",  "tool": "questionnaire","level": "variable"},
        "physical":       {"modality": "physical",     "tool": "measurement",  "level": "variable"},
        "hospitalization":{"modality": "clinical",     "tool": "ICD10",        "level": "outcome"},
    },
    "ADNI": {
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_subcortical_volume": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_voxel":             {"modality": "sMRI", "tool": "voxel",       "level": "voxel"},
        "pet_amyloid": {"modality": "PET", "tool": "florbetapir",  "level": "ROI"},
        "pet_tau":     {"modality": "PET", "tool": "flortaucipir", "level": "ROI"},
        "pet_fdg":     {"modality": "PET", "tool": "FDG",          "level": "ROI"},
        "fmri_fc":     {"modality": "fMRI", "tool": "task/resting", "level": "connectivity"},
        "dti_fa":      {"modality": "dMRI", "tool": "DTI",          "level": "tract"},
        "lesion_volume": {"modality": "sMRI", "tool": "MedSAM", "level": "ROI"},
        "genetics":    {"modality": "genetics", "tool": "APOE/GWAS", "level": "SNP"},
        "medication":  {"modality": "clinical", "tool": "medication_log", "level": "variable"},
    },
    "HCP_YA": {
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_myelin":            {"modality": "sMRI", "tool": "T1w/T2w",    "level": "ROI"},
        "smri_voxel":             {"modality": "sMRI", "tool": "voxel",       "level": "voxel"},
        "rfmri_fc":  {"modality": "fMRI", "tool": "rfMRI",    "level": "connectivity"},
        "tfmri_task":{"modality": "fMRI", "tool": "task fMRI","level": "activation"},
        "dmri_sc":   {"modality": "dMRI", "tool": "HARDI",    "level": "connectivity"},
        "meg":       {"modality": "MEG",  "tool": "MEG",      "level": "connectivity"},
    },
}

DATASET_OUTCOMES = {
    "UKB": [
        "disease_diagnosis",   # ICD10 codes
        "mortality",           # death registry
        "cognitive_score",     # touchscreen cognitive tests
        "imaging_phenotype",   # derived imaging phenotypes
    ],
    "ADNI": [
        "diagnosis",           # CN / MCI / AD
        "conversion",          # MCI → AD conversion
        "cognitive_decline",   # ADAS-Cog, MMSE decline
        "biomarker_status",    # amyloid+/tau+ status
    ],
    "HCP_YA": [
        "behavioral_score",    # NIH Toolbox
        "cognitive_task",      # task fMRI performance
        "personality",         # NEO-FFI
    ],
}

# Imaging feature templates — dynamically combined with AAL atlas regions
# {region} is replaced with actual neuroanatomy node names at generation time
IMAGING_FEATURE_TEMPLATES = {
    # sMRI FreeSurfer ROI features
    "cortical thickness of {region}":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "ADNI", "HCP_YA"]},
    "gray matter volume of {region}":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "ADNI", "HCP_YA"]},
    "subcortical volume of {region}":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "ADNI", "HCP_YA"]},
    "cortical area of {region}":        {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "HCP_YA"]},
    # dMRI tract features
    "fractional anisotropy of {region}": {"modality": "dMRI", "tool": "TBSS", "level": "tract",
                                           "datasets": ["UKB", "HCP_YA"]},
    "mean diffusivity of {region}":      {"modality": "dMRI", "tool": "TBSS", "level": "tract",
                                           "datasets": ["UKB", "HCP_YA"]},
    # PET ROI features (ADNI)
    "amyloid SUVR of {region}":          {"modality": "PET", "tool": "florbetapir", "level": "ROI",
                                           "datasets": ["ADNI"]},
    "tau SUVR of {region}":              {"modality": "PET", "tool": "flortaucipir", "level": "ROI",
                                           "datasets": ["ADNI"]},
    "FDG uptake of {region}":            {"modality": "PET", "tool": "FDG", "level": "ROI",
                                           "datasets": ["ADNI"]},
    # lesion segmentation
    "lesion volume of {region}":          {"modality": "sMRI", "tool": "MedSAM", "level": "ROI",
                                           "datasets": ["UKB", "ADNI"]},
}

# Connectivity feature templates — {a} and {b} are AAL regions
CONNECTIVITY_FEATURE_TEMPLATES = {
    "functional connectivity between {a} and {b}":    {"modality": "fMRI", "tool": "rfMRI",
                                                        "level": "connectivity",
                                                        "datasets": ["UKB", "ADNI", "HCP_YA"]},
    "effective connectivity from {a} to {b}":         {"modality": "fMRI", "tool": "DCM/GC",
                                                        "level": "connectivity",
                                                        "datasets": ["ADNI", "HCP_YA"]},
    "structural connectivity between {a} and {b}":    {"modality": "dMRI", "tool": "tractography",
                                                        "level": "connectivity",
                                                        "datasets": ["UKB", "HCP_YA"]},
}

# Domain pairs for imaging-driven hypothesis generation
# source domain → target domain, aligned with dataset modalities
IMAGING_DOMAIN_PAIRS = [
    # sMRI features → disease
    ("neuroanatomy", "disease"),
    # connectivity → disease
    ("connectivity", "disease"),
    # sMRI features → cognitive function
    ("neuroanatomy", "cognitive_function"),
    # gene → brain structure (UKB genetics + imaging)
    ("gene", "neuroanatomy"),
    # disease → drug (ADNI)
    ("disease", "drug"),
]

# AAL atlas regions used for imaging feature generation
# Subset of neuroanatomy nodes from NN_AAL source
_AAL_REGION_KEYWORDS = [
    "Precentral", "Frontal_Sup", "Frontal_Mid", "Frontal_Inf", "Rolandic_Oper",
    "Supp_Motor", "Olfactory", "Frontal_Sup_Med", "Frontal_Med_Orb",
    "Rectus", "Insula", "Cingulate", "Hippocampus", "Parahippocampal",
    "Amygdala", "Calcarine", "Cuneus", "Lingual", "Occipital",
    "Fusiform", "Postcentral", "Parietal", "SupraMarginal", "Angular",
    "Precuneus", "Paracentral", "Caudate", "Putamen", "Pallidum",
    "Thalamus", "Heschl", "Temporal", "Temporal_Pole",
]

# ── engine ─────────────────────────────────────────────────────────────

class HypothesisEngine:
    """Batch-generate, persist, and rank testable hypotheses from a knowledge graph."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.G = kg.G
        self._index = kg._index
        # Build claims index for frequency_boost: (subj, pred, obj) → [claim_meta, ...]
        self._claims_by_triple: dict[tuple[str, str, str], list[dict]] = {}
        for nid, node in self._index.items():
            if "claim" not in node.domain_tags:
                continue
            meta = node.metadata
            key = (meta.get("subject_id", ""), meta.get("predicate", ""), meta.get("object_id", ""))
            if key[0] and key[2]:
                self._claims_by_triple.setdefault(key, []).append(meta)

    # ── batch generation ───────────────────────────────────────────────

    def batch_generate(
        self,
        domain_pairs: Optional[list[tuple[str, str]]] = None,
        max_hops: int = 3,
        max_paths_per_pair: int = 5,
        max_seeds_per_domain: int = 50,
    ) -> list[Hypothesis]:
        """Batch-generate hypotheses across the entire graph.

        Strategy: for each domain pair, sample seed concepts from domain_a,
        find paths to concepts in domain_b within max_hops hops.
        """
        if domain_pairs is None:
            domain_pairs = DEFAULT_DOMAIN_PAIRS

        all_hypotheses: list[Hypothesis] = []
        seen_pairs: set[tuple[str, str]] = set()
        _hyp_counter = 0

        for dom_a, dom_b in domain_pairs:
            logger.info(f"generating hypotheses: {dom_a} -> {dom_b}")

            seeds_a = self._sample_domain_nodes(dom_a, max_seeds_per_domain)
            targets_b = {
                nid for nid, data in self.G.nodes(data=True)
                if dom_b in data.get("domain_tags", [])
                and "claim" not in data.get("domain_tags", [])
            }

            for seed_id in seeds_a:
                if seed_id not in self.G:
                    continue

                # BFS from seed
                try:
                    reachable = nx.single_source_shortest_path(
                        self.G, seed_id, cutoff=max_hops
                    )
                except nx.NetworkXError:
                    continue

                # find targets in domain_b
                candidates = [
                    nid for nid in reachable
                    if nid in targets_b and nid != seed_id
                ]

                pair_count = 0
                for target_id in candidates:
                    pair_key = tuple(sorted([seed_id, target_id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    raw_path = reachable[target_id]
                    links = self._enrich_path(raw_path)
                    if not links:
                        continue

                    conf = self._compute_confidence_score(links)
                    nov = self._compute_novelty_score(links)
                    evi = self._compute_evidence_score(links)
                    test, test_reason = self._compute_testability_score(links)
                    claim_ids = [l.claim_id for l in links if l.claim_id]

                    _hyp_counter += 1
                    h = Hypothesis(
                        id=f"HYP:{_hyp_counter:06d}",
                        hypothesis_type="bridge",
                        source_id=seed_id,
                        source_name=self._index[seed_id].preferred_name,
                        target_id=target_id,
                        target_name=self._index[target_id].preferred_name,
                        path=links,
                        confidence_score=conf,
                        novelty_score=nov,
                        evidence_score=evi,
                        testability_score=test,
                        composite_score=0.0,  # set below
                        supporting_claims=claim_ids,
                        testability_reason=test_reason,
                        metadata={"domain_a": dom_a, "domain_b": dom_b},
                    )
                    h.explanation = self._generate_explanation(h)
                    h.composite_score = self._composite_score(h)
                    all_hypotheses.append(h)

                    pair_count += 1
                    if pair_count >= max_paths_per_pair:
                        break

        logger.info(f"batch generation complete: {len(all_hypotheses)} hypotheses from {len(domain_pairs)} domain pairs")

        all_hypotheses = self.post_process(all_hypotheses)
        return all_hypotheses

    def post_process(
        self,
        hypotheses: list[Hypothesis],
        min_hops: int = 2,
        filter_vague_relations: bool = True,
        filter_non_measurable: bool = True,
    ) -> list[Hypothesis]:
        """Filter low-quality hypotheses after generation.

        Filters:
        1. Noisy entities — source/target name matches NOISE_PATTERNS
        2. 1-hop hypotheses — too simple, just restates existing edges
        3. Vague relations — all links are is_associated_with / associated_with / about
        4. Non-measurable biomarkers — entities not directly measurable from brain imaging
        """
        before = len(hypotheses)
        filtered = []

        for h in hypotheses:
            # filter noisy entities (source, target, and all intermediate nodes)
            all_names = {h.source_name, h.target_name}
            for link in h.path:
                all_names.add(link.from_name)
                all_names.add(link.to_name)
            if any(self._is_noisy_entity(name) for name in all_names):
                continue

            # filter 1-hop (single direct edge = no discovery value)
            if len(h.path) < min_hops:
                continue

            # filter all-vague-relations
            if filter_vague_relations:
                relation_types = {l.relation_type for l in h.path}
                if relation_types and relation_types <= VAGUE_RELATIONS:
                    continue

            # filter non-measurable biomarkers (not testable from imaging)
            if filter_non_measurable:
                if self._has_non_measurable_entity(h):
                    continue

            # filter biologically implausible paths (brain region → non-neurological target)
            if self._has_implausible_path(h):
                continue

            filtered.append(h)

        logger.info(f"post_process: {before} -> {len(filtered)} hypotheses "
                     f"(removed {before - len(filtered)})")
        return filtered

    def _has_non_measurable_entity(self, h: Hypothesis) -> bool:
        """Check if hypothesis involves entities not measurable from brain imaging.

        Filters out hypotheses where source or target is:
        - A non-measurable domain (neurotransmitter levels, protein expression, etc.)
        - Matches non-measurable entity name patterns (CSF markers, blood markers, etc.)
        """
        for node_name, node_id in [(h.source_name, h.source_id), (h.target_name, h.target_id)]:
            # check domain tags
            node = self._index.get(node_id)
            if node:
                domains = set(node.domain_tags) - {"claim"}
                # allow neurotransmitter/protein as intermediate hops only if source or target is neuroanatomy
                if domains & NON_MEASURABLE_BIOMARKER_TYPES:
                    # check if the OTHER end is a brain region (then it's a valid "X affects brain" hypothesis)
                    other_name = h.target_name if node_name == h.source_name else h.source_name
                    other_id = h.target_id if node_name == h.source_name else h.source_id
                    other_node = self._index.get(other_id)
                    if other_node and "neuroanatomy" not in other_node.domain_tags:
                        return True

            # check name patterns
            for pattern in _NON_MEASURABLE_PATTERNS:
                if pattern.search(node_name):
                    return True

        return False

    @staticmethod
    def _is_noisy_entity(name: str) -> bool:
        """Check if an entity name matches known noise patterns."""
        if not name or len(name.strip()) == 0:
            return True
        name_clean = name.strip()
        for pattern in NOISE_PATTERNS:
            if pattern.match(name_clean):
                return True
        # check if name contains any noise word
        words = set(re.split(r"[\s\-_,/]+", name_clean.lower()))
        if words & _NOISE_WORDS:
            return True
        return False

    def _has_implausible_path(self, h: Hypothesis) -> bool:
        """Check if hypothesis path has biologically implausible connections.

        Filters paths where a brain region directly predicts a non-neurological
        condition (e.g., amygdala → urinary incontinence) without a plausible
        intermediate neurological mechanism.
        """
        # Check if source is a brain region and target is non-neurological
        source_node = self._index.get(h.source_id)
        target_node = self._index.get(h.target_id)

        if not source_node or not target_node:
            return False

        source_is_brain = "neuroanatomy" in source_node.domain_tags
        target_is_neuro = any(d in target_node.domain_tags for d in
                              ["neuroanatomy", "disease", "cognitive_function",
                               "biomarker", "gene", "drug", "neurotransmitter"])

        # If source is brain region and target is non-neurological, check target name
        if source_is_brain and not target_is_neuro:
            if _NON_NEUROLOGICAL_TARGETS.search(h.target_name):
                return True

        # Also check intermediate nodes in the path
        for link in h.path:
            if _NON_NEUROLOGICAL_TARGETS.search(link.to_name):
                # Check if the previous node is a brain region
                prev_node = self._index.get(link.from_id)
                if prev_node and "neuroanatomy" in prev_node.domain_tags:
                    # Only filter if there's no disease intermediate
                    has_disease_intermediate = any(
                        "disease" in self._index.get(l.from_id, ConceptNode(id="", preferred_name="")).domain_tags
                        for l in h.path[:h.path.index(link)]
                    )
                    if not has_disease_intermediate:
                        return True

        return False

    # ── imaging-driven batch generation ──────────────────────────────

    def batch_generate_imaging(
        self,
        dataset: str = "UKB",
        max_paths_per_pair: int = 5,
        max_seeds: int = 50,
        max_hops: int = 3,
        include_connectivity: bool = True,
    ) -> list[Hypothesis]:
        """Generate hypotheses driven by imaging features available in a dataset.

        Strategy:
        1. Find AAL atlas neuroanatomy nodes in the graph as ROI seeds
        2. For each ROI × imaging feature template, construct a feature name
           (e.g., "cortical thickness of Hippocampus_L")
        3. Find graph paths from each ROI to disease/cognitive_function nodes
        4. Filter using expanded exclusion rules
        5. Annotate each hypothesis with dataset metadata
        """
        dataset_key = dataset.upper().replace("-", "_")
        if dataset_key not in DATASET_FEATURES:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_FEATURES.keys())}")

        ds_features = DATASET_FEATURES[dataset_key]
        ds_outcomes = DATASET_OUTCOMES.get(dataset_key, [])

        # 1. Find AAL atlas ROI nodes
        aal_nodes = self._find_aal_regions(max_seeds)
        if not aal_nodes:
            logger.warning("No AAL atlas regions found in graph")
            return []

        logger.info(f"Found {len(aal_nodes)} AAL regions for imaging hypothesis generation")

        # 2. Collect outcome nodes (disease, cognitive_function)
        outcome_nodes = self._collect_outcome_nodes()
        if not outcome_nodes:
            logger.warning("No outcome nodes (disease/cognitive_function) found")
            return []

        # 3. Determine which imaging templates apply to this dataset
        applicable_templates = {
            name: meta for name, meta in IMAGING_FEATURE_TEMPLATES.items()
            if dataset_key in meta["datasets"]
        }

        all_hypotheses: list[Hypothesis] = []
        _hyp_counter = 0
        seen_pairs: set[tuple[str, str]] = set()

        # 4. Generate ROI-level imaging hypotheses
        for region_id, region_name in aal_nodes.items():
            for feat_template, feat_meta in applicable_templates.items():
                feature_name = feat_template.replace("{region}", region_name)

                # Find paths from this ROI to outcomes
                try:
                    reachable = nx.single_source_shortest_path(
                        self.G, region_id, cutoff=max_hops
                    )
                except nx.NetworkXError:
                    continue

                candidates = [
                    nid for nid in reachable
                    if nid in outcome_nodes and nid != region_id
                ]

                pair_count = 0
                for target_id in candidates:
                    pair_key = (region_id, target_id, feat_template)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    raw_path = reachable[target_id]
                    links = self._enrich_path(raw_path)
                    if not links:
                        continue

                    # Skip if path contains non-measurable entities
                    if self._path_has_non_measurable(links):
                        continue

                    conf = self._compute_confidence_score(links)
                    nov = self._compute_novelty_score(links)
                    evi = self._compute_evidence_score(links)
                    test, test_reason = self._compute_testability_score(links)
                    # Boost testability for imaging-driven hypotheses
                    test = min(test + 0.15, 1.0)
                    claim_ids = [l.claim_id for l in links if l.claim_id]

                    _hyp_counter += 1
                    target_node = self._index.get(target_id)
                    h = Hypothesis(
                        id=f"HYP:IMG:{_hyp_counter:06d}",
                        hypothesis_type="imaging",
                        source_id=region_id,
                        source_name=feature_name,
                        target_id=target_id,
                        target_name=target_node.preferred_name if target_node else target_id,
                        path=links,
                        confidence_score=conf,
                        novelty_score=nov,
                        evidence_score=evi,
                        testability_score=test,
                        composite_score=0.0,
                        supporting_claims=claim_ids,
                        testability_reason=test_reason,
                        metadata={
                            "dataset": dataset_key,
                            "input_modality": feat_meta["modality"],
                            "input_feature": feature_name,
                            "input_level": feat_meta["level"],
                            "input_tool": feat_meta["tool"],
                            "input_region": region_name,
                            "outcome_type": self._classify_outcome(target_node),
                        },
                    )
                    h.explanation = self._generate_explanation(h)
                    h.composite_score = self._composite_score(h)
                    all_hypotheses.append(h)

                    pair_count += 1
                    if pair_count >= max_paths_per_pair:
                        break

        # 5. Generate connectivity-level hypotheses
        if include_connectivity:
            conn_templates = {
                name: meta for name, meta in CONNECTIVITY_FEATURE_TEMPLATES.items()
                if dataset_key in meta["datasets"]
            }
            if conn_templates:
                hyps = self._generate_connectivity_hypotheses(
                    aal_nodes, outcome_nodes, conn_templates,
                    dataset_key, max_paths_per_pair, max_hops, _hyp_counter, seen_pairs,
                )
                _hyp_counter += len(hyps)
                all_hypotheses.extend(hyps)

        logger.info(
            f"imaging batch generation ({dataset_key}): "
            f"{len(all_hypotheses)} hypotheses from {len(aal_nodes)} regions"
        )

        all_hypotheses = self.post_process(all_hypotheses)
        return all_hypotheses

    def _find_aal_regions(self, max_n: int) -> dict[str, str]:
        """Find AAL atlas neuroanatomy nodes. Returns {node_id: region_name}."""
        candidates = {}
        for nid, data in self.G.nodes(data=True):
            if "neuroanatomy" not in data.get("domain_tags", []):
                continue
            name = data.get("preferred_name", "")
            # Match against AAL region keywords
            name_lower = name.lower()
            for kw in _AAL_REGION_KEYWORDS:
                if kw.lower() in name_lower:
                    candidates[nid] = name
                    break
        # Sort by degree (more connected = richer paths)
        sorted_items = sorted(
            candidates.items(),
            key=lambda item: self.G.degree(item[0]),
            reverse=True,
        )
        return dict(sorted_items[:max_n])

    def _collect_outcome_nodes(self) -> set[str]:
        """Collect all disease + cognitive_function nodes as potential outcomes."""
        outcome_ids = set()
        for nid, data in self.G.nodes(data=True):
            domains = set(data.get("domain_tags", []))
            if "claim" in domains:
                continue
            if domains & {"disease", "cognitive_function"}:
                outcome_ids.add(nid)
        return outcome_ids

    def _classify_outcome(self, node: Optional[ConceptNode]) -> str:
        """Classify outcome node type for metadata."""
        if not node:
            return "unknown"
        domains = set(node.domain_tags)
        if "disease" in domains:
            return "disease"
        if "cognitive_function" in domains:
            return "cognitive_function"
        if "biomarker" in domains:
            return "biomarker"
        return "other"

    def _path_has_non_measurable(self, links: list[HypothesisLink]) -> bool:
        """Check if any intermediate node in the path is non-measurable."""
        for link in links:
            for name, nid in [(link.from_name, link.from_id), (link.to_name, link.to_id)]:
                node = self._index.get(nid)
                if node:
                    domains = set(node.domain_tags) - {"claim"}
                    if domains & NON_MEASURABLE_BIOMARKER_TYPES:
                        return True
                for pattern in _NON_MEASURABLE_PATTERNS:
                    if pattern.search(name):
                        return True
        return False

    def _generate_connectivity_hypotheses(
        self,
        aal_nodes: dict[str, str],
        outcome_nodes: set[str],
        conn_templates: dict,
        dataset_key: str,
        max_paths_per_pair: int,
        max_hops: int,
        hyp_counter_start: int,
        seen_pairs: set,
    ) -> list[Hypothesis]:
        """Generate hypotheses for connectivity features (FC/EC/SC between region pairs)."""
        hypotheses = []
        counter = hyp_counter_start
        region_ids = list(aal_nodes.keys())

        # Sample region pairs (limit to avoid O(n^2) explosion)
        max_pairs = min(len(region_ids) * 3, 200)
        import random
        if len(region_ids) > 20:
            sampled_pairs = []
            for _ in range(max_pairs):
                a, b = random.sample(region_ids, 2)
                sampled_pairs.append((a, b))
        else:
            sampled_pairs = [(a, b) for i, a in enumerate(region_ids) for b in region_ids[i+1:]]
            sampled_pairs = sampled_pairs[:max_pairs]

        for region_a_id, region_b_id in sampled_pairs:
            name_a = aal_nodes[region_a_id]
            name_b = aal_nodes[region_b_id]

            for feat_template, feat_meta in conn_templates.items():
                feature_name = feat_template.replace("{a}", name_a).replace("{b}", name_b)

                # Find paths from region_a to outcomes (potentially through region_b)
                try:
                    reachable = nx.single_source_shortest_path(
                        self.G, region_a_id, cutoff=max_hops
                    )
                except nx.NetworkXError:
                    continue

                candidates = [
                    nid for nid in reachable
                    if nid in outcome_nodes and nid != region_a_id
                ]

                pair_count = 0
                for target_id in candidates:
                    pair_key = (region_a_id, target_id, feat_template)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    raw_path = reachable[target_id]
                    links = self._enrich_path(raw_path)
                    if not links:
                        continue

                    if self._path_has_non_measurable(links):
                        continue

                    conf = self._compute_confidence_score(links)
                    nov = self._compute_novelty_score(links)
                    evi = self._compute_evidence_score(links)
                    test, test_reason = self._compute_testability_score(links)
                    test = min(test + 0.15, 1.0)
                    claim_ids = [l.claim_id for l in links if l.claim_id]

                    counter += 1
                    target_node = self._index.get(target_id)
                    h = Hypothesis(
                        id=f"HYP:IMG:{counter:06d}",
                        hypothesis_type="imaging_connectivity",
                        source_id=region_a_id,
                        source_name=feature_name,
                        target_id=target_id,
                        target_name=target_node.preferred_name if target_node else target_id,
                        path=links,
                        confidence_score=conf,
                        novelty_score=nov,
                        evidence_score=evi,
                        testability_score=test,
                        composite_score=0.0,
                        supporting_claims=claim_ids,
                        testability_reason=test_reason,
                        metadata={
                            "dataset": dataset_key,
                            "input_modality": feat_meta["modality"],
                            "input_feature": feature_name,
                            "input_level": feat_meta["level"],
                            "input_tool": feat_meta["tool"],
                            "input_region_a": name_a,
                            "input_region_b": name_b,
                            "outcome_type": self._classify_outcome(target_node),
                        },
                    )
                    h.explanation = self._generate_explanation(h)
                    h.composite_score = self._composite_score(h)
                    hypotheses.append(h)

                    pair_count += 1
                    if pair_count >= max_paths_per_pair:
                        break

        return hypotheses

    # ── persistence ────────────────────────────────────────────────────

    def save_hypotheses(self, hypotheses: list[Hypothesis], path: str | Path) -> None:
        """Save hypotheses to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_hypotheses": len(hypotheses),
            "hypotheses": [h.to_dict() for h in hypotheses],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"saved {len(hypotheses)} hypotheses to {path}")

    def load_hypotheses(self, path: str | Path) -> list[Hypothesis]:
        """Load hypotheses from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        hypotheses = [Hypothesis.from_dict(h) for h in data["hypotheses"]]
        logger.info(f"loaded {len(hypotheses)} hypotheses from {path}")
        return hypotheses

    # ── ranking ────────────────────────────────────────────────────────

    def rank_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        weights: Optional[dict[str, float]] = None,
        top_n: int = 100,
        skip_post_process: bool = False,
    ) -> list[Hypothesis]:
        """Rank hypotheses by composite score (novelty, evidence, testability, confidence).

        Args:
            hypotheses: list of hypotheses to rank
            weights: custom weights dict, keys: confidence, evidence, novelty, testability
            top_n: return top N results
            skip_post_process: if True, skip the post-processing filter
        """
        if not skip_post_process:
            hypotheses = self.post_process(hypotheses)

        if weights is None:
            # testability weighted highest — must be verifiable with imaging experiments
            weights = {
                "confidence": 0.20,
                "evidence": 0.20,
                "novelty": 0.25,
                "testability": 0.35,
            }

        for h in hypotheses:
            h.composite_score = (
                (h.confidence_score ** weights["confidence"])
                * (h.evidence_score ** weights["evidence"])
                * (h.novelty_score ** weights["novelty"])
                * (max(h.testability_score, 0.01) ** weights["testability"])
            )

        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)
        return hypotheses[:top_n]

    # ── query-based (kept for interactive use) ─────────────────────────

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        max_paths: int = 20,
    ) -> list[Hypothesis]:
        """Find hypothesis paths between two concepts with evidence enrichment."""
        if source_id not in self.G or target_id not in self.G:
            return []

        claim_nodes = {nid for nid, n in self._index.items() if "claim" in n.domain_tags}
        intermediate_exclude = claim_nodes - {source_id, target_id}

        subgraph = self.G.copy()
        subgraph.remove_nodes_from(intermediate_exclude)

        if source_id not in subgraph or target_id not in subgraph:
            return []

        try:
            raw_paths = list(nx.all_simple_paths(
                subgraph, source_id, target_id, cutoff=max_hops
            ))
        except nx.NetworkXError:
            return []

        raw_paths = raw_paths[:max_paths]
        return self._build_hypotheses_from_paths(raw_paths, "path")

    def bridge_discovery(
        self,
        concept_id: str,
        target_domain: str,
        max_hops: int = 3,
        max_results: int = 20,
    ) -> list[Hypothesis]:
        """Find cross-domain connections through intermediate claims."""
        if concept_id not in self.G:
            return []

        target_nodes = {
            nid for nid, data in self.G.nodes(data=True)
            if target_domain in data.get("domain_tags", [])
        }
        if not target_nodes:
            return []

        try:
            reachable = nx.single_source_shortest_path(
                self.G, concept_id, cutoff=max_hops
            )
        except nx.NetworkXError:
            return []

        candidates = {
            nid for nid in reachable
            if nid in target_nodes and nid != concept_id
            and "claim" not in self._index.get(nid, ConceptNode(id="", preferred_name="")).domain_tags
        }

        hypotheses = []
        for target_id in candidates:
            raw_path = reachable[target_id]
            links = self._enrich_path(raw_path)
            if not links:
                continue

            conf = self._compute_confidence_score(links)
            nov = self._compute_novelty_score(links)
            evi = self._compute_evidence_score(links)
            test, test_reason = self._compute_testability_score(links)
            claim_ids = [l.claim_id for l in links if l.claim_id]

            h = Hypothesis(
                hypothesis_type="bridge",
                source_id=concept_id,
                source_name=self._index[concept_id].preferred_name,
                target_id=target_id,
                target_name=self._index[target_id].preferred_name,
                path=links,
                confidence_score=conf,
                novelty_score=nov,
                evidence_score=evi,
                testability_score=test,
                supporting_claims=claim_ids,
                testability_reason=test_reason,
            )
            h.explanation = self._generate_explanation(h)
            h.composite_score = self._composite_score(h)
            hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)
        return hypotheses[:max_results]

    def discover_hypotheses(
        self,
        concept_id: str,
        max_hops: int = 3,
        max_results: int = 30,
        exclude_domains: Optional[set[str]] = None,
    ) -> list[Hypothesis]:
        """Find hypotheses radiating from a single concept to all reachable domains."""
        if concept_id not in self.G:
            return []

        exclude = exclude_domains or {"claim"}
        source_node = self._index.get(concept_id)
        source_domains = set(source_node.domain_tags) - exclude if source_node else set()

        try:
            reachable = nx.single_source_shortest_path(self.G, concept_id, cutoff=max_hops)
        except nx.NetworkXError:
            return []

        candidates = []
        for target_id, raw_path in reachable.items():
            if target_id == concept_id:
                continue
            target_node = self._index.get(target_id)
            if not target_node:
                continue
            target_domains = set(target_node.domain_tags) - exclude
            if not target_domains or target_domains <= source_domains:
                continue
            candidates.append((target_id, raw_path))

        hypotheses = []
        for target_id, raw_path in candidates:
            links = self._enrich_path(raw_path)
            if not links:
                continue
            conf = self._compute_confidence_score(links)
            nov = self._compute_novelty_score(links)
            evi = self._compute_evidence_score(links)
            test, test_reason = self._compute_testability_score(links)
            claim_ids = [l.claim_id for l in links if l.claim_id]

            h = Hypothesis(
                hypothesis_type="discover",
                source_id=concept_id,
                source_name=self._index[concept_id].preferred_name,
                target_id=target_id,
                target_name=self._index[target_id].preferred_name,
                path=links,
                confidence_score=conf,
                novelty_score=nov,
                evidence_score=evi,
                testability_score=test,
                supporting_claims=claim_ids,
                testability_reason=test_reason,
            )
            h.explanation = self._generate_explanation(h)
            h.composite_score = self._composite_score(h)
            hypotheses.append(h)

        hypotheses = self.post_process(hypotheses)
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)
        return hypotheses[:max_results]

    def find_trending(
        self,
        since_year: int = 2020,
        min_claims: int = 3,
        direction: str = "strengthening",
        max_results: int = 30,
    ) -> list[dict]:
        """Find concept pairs with strengthening/weakening evidence over time.

        Returns list of dicts with: concept_a, concept_b, years, slope, direction, claims.
        """
        from collections import Counter

        # Group claims by (subject, object)
        claim_groups: dict[tuple[str, str], list[dict]] = {}
        for nid, node in self._index.items():
            if "claim" not in node.domain_tags:
                continue
            meta = node.metadata
            sid = meta.get("subject_id", "")
            oid = meta.get("object_id", "")
            if not sid or not oid:
                continue
            key = (sid, oid)
            claim_groups.setdefault(key, []).append(meta)

        results = []
        for (sid, oid), claims in claim_groups.items():
            years = []
            for c in claims:
                sp = c.get("source_paper", {})
                y = sp.get("year")
                if y and y >= since_year:
                    years.append(y)

            if len(years) < min_claims:
                continue

            year_counts = Counter(years)
            ys = sorted(year_counts.keys())
            cs = [year_counts[y] for y in ys]
            slope = _simple_slope(ys, cs)

            if direction == "strengthening" and slope <= 0.3:
                continue
            if direction == "weakening" and slope >= -0.3:
                continue
            if direction == "emerging" and max(ys) < 2025:
                continue

            src_node = self._index.get(sid)
            tgt_node = self._index.get(oid)

            results.append({
                "concept_a": src_node.preferred_name if src_node else sid,
                "concept_b": tgt_node.preferred_name if tgt_node else oid,
                "concept_a_id": sid,
                "concept_b_id": oid,
                "year_counts": {str(y): year_counts[y] for y in ys},
                "slope": round(slope, 3),
                "direction": direction,
                "n_claims": len(claims),
            })

        results.sort(key=lambda r: abs(r["slope"]), reverse=True)
        return results[:max_results]

    def contradiction_detection(
        self,
        domain_filter: Optional[str] = None,
        max_results: int = 50,
    ) -> list[Contradiction]:
        """Find pairs of claims that assert opposite things about the same concept pair."""
        claim_lookup: dict[tuple[str, str], list[ConceptNode]] = {}
        for nid, node in self._index.items():
            if "claim" not in node.domain_tags:
                continue
            meta = node.metadata
            sid = meta.get("subject_id", "")
            oid = meta.get("object_id", "")
            if not sid or not oid:
                continue

            if domain_filter:
                src_node = self._index.get(sid)
                tgt_node = self._index.get(oid)
                domains = set()
                if src_node:
                    domains.update(src_node.domain_tags)
                if tgt_node:
                    domains.update(tgt_node.domain_tags)
                if domain_filter not in domains:
                    continue

            key = (sid, oid)
            claim_lookup.setdefault(key, []).append(node)

        contradictions = []
        for (sid, oid), claims in claim_lookup.items():
            if len(claims) < 2:
                continue
            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    c1, c2 = claims[i], claims[j]
                    m1, m2 = c1.metadata, c2.metadata
                    severity = self._check_contradiction(m1, m2)
                    if severity > 0:
                        contradictions.append(Contradiction(
                            concept_a_id=sid,
                            concept_a_name=m1.get("subject_name", sid),
                            concept_b_id=oid,
                            concept_b_name=m1.get("object_name", oid),
                            claim_for_id=c1.id,
                            claim_for_predicate=m1.get("predicate", ""),
                            claim_for_text=m1.get("raw_text", ""),
                            claim_against_id=c2.id,
                            claim_against_predicate=m2.get("predicate", ""),
                            claim_against_text=m2.get("raw_text", ""),
                            severity=severity,
                        ))

        contradictions.sort(key=lambda c: c.severity, reverse=True)
        return contradictions[:max_results]

    def gap_detection(
        self,
        domain_a: str,
        domain_b: Optional[str] = None,
        max_results: int = 50,
    ) -> list[Gap]:
        """Find concept pairs 2 hops apart with no direct edge."""
        if domain_b is None:
            domain_b = domain_a

        nodes_a = {
            nid for nid, data in self.G.nodes(data=True)
            if domain_a in data.get("domain_tags", [])
            and "claim" not in data.get("domain_tags", [])
        }
        nodes_b = {
            nid for nid, data in self.G.nodes(data=True)
            if domain_b in data.get("domain_tags", [])
            and "claim" not in data.get("domain_tags", [])
        }

        gaps = []
        seen = set()

        for a_id in nodes_a:
            if a_id not in self.G:
                continue
            hop1 = set(self.G.successors(a_id)) | set(self.G.predecessors(a_id))
            hop2 = set()
            for n1 in hop1:
                if "claim" in self._index.get(n1, ConceptNode(id="", preferred_name="")).domain_tags:
                    continue
                hop2.update(self.G.successors(n1))
                hop2.update(self.G.predecessors(n1))

            hop2 -= {a_id}
            hop2 -= hop1

            for b_id in hop2 & nodes_b:
                pair = tuple(sorted([a_id, b_id]))
                if pair in seen:
                    continue
                seen.add(pair)

                if self.G.has_edge(a_id, b_id) or self.G.has_edge(b_id, a_id):
                    continue

                try:
                    path = nx.shortest_path(self.G, a_id, b_id)
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue

                if len(path) > 3:
                    continue

                connecting = [n for n in path[1:-1]
                              if "claim" not in self._index.get(n, ConceptNode(id="", preferred_name="")).domain_tags]

                a_node = self._index.get(a_id)
                b_node = self._index.get(b_id)

                gaps.append(Gap(
                    concept_a_id=a_id,
                    concept_a_name=a_node.preferred_name if a_node else a_id,
                    concept_b_id=b_id,
                    concept_b_name=b_node.preferred_name if b_node else b_id,
                    distance=len(path) - 1,
                    connecting_concepts=connecting,
                    domain_a=domain_a,
                    domain_b=domain_b,
                    potential_relation=self._infer_relation(path),
                ))

        gaps.sort(key=lambda g: (0 if g.domain_a != g.domain_b else 1, g.distance))
        return gaps[:max_results]

    # ── name resolution ────────────────────────────────────────────────

    def resolve_name(self, query: str) -> Optional[str]:
        """Resolve a name to a concept ID. Returns None if not found."""
        if not query:
            return None

        for node in self._index.values():
            if node.preferred_name == query:
                return node.id

        query_lower = query.lower()
        for node in self._index.values():
            if node.preferred_name.lower() == query_lower:
                return node.id

        for node in self._index.values():
            for alias in node.aliases:
                if alias.lower() == query_lower:
                    return node.id

        candidates = []
        for node in self._index.values():
            name_lower = node.preferred_name.lower()
            if query_lower in name_lower or name_lower in query_lower:
                candidates.append(node)
                continue
            for alias in node.aliases:
                if query_lower in alias.lower() or alias.lower() in query_lower:
                    candidates.append(node)
                    break

        if len(candidates) == 1:
            return candidates[0].id
        elif len(candidates) > 1:
            candidates.sort(key=lambda n: len(n.preferred_name))
            return candidates[0].id

        return None

    # ── internal helpers ───────────────────────────────────────────────

    def _sample_domain_nodes(self, domain: str, max_n: int) -> list[str]:
        """Sample up to max_n non-claim nodes from a domain, preferring nodes with edges."""
        nodes = [
            nid for nid, data in self.G.nodes(data=True)
            if domain in data.get("domain_tags", [])
            and "claim" not in data.get("domain_tags", [])
        ]
        # sort by degree (more connected = more useful as seed)
        nodes.sort(key=lambda n: self.G.degree(n), reverse=True)
        return nodes[:max_n]

    def _build_hypotheses_from_paths(
        self, raw_paths: list[list[str]], hyp_type: str
    ) -> list[Hypothesis]:
        """Build Hypothesis objects from raw node-ID paths."""
        hypotheses = []
        for raw_path in raw_paths:
            links = self._enrich_path(raw_path)
            if not links:
                continue

            conf = self._compute_confidence_score(links)
            nov = self._compute_novelty_score(links)
            evi = self._compute_evidence_score(links)
            test, test_reason = self._compute_testability_score(links)
            claim_ids = [l.claim_id for l in links if l.claim_id]

            h = Hypothesis(
                hypothesis_type=hyp_type,
                source_id=raw_path[0],
                source_name=self._index[raw_path[0]].preferred_name,
                target_id=raw_path[-1],
                target_name=self._index[raw_path[-1]].preferred_name,
                path=links,
                confidence_score=conf,
                novelty_score=nov,
                evidence_score=evi,
                testability_score=test,
                supporting_claims=claim_ids,
                testability_reason=test_reason,
            )
            h.explanation = self._generate_explanation(h)
            h.composite_score = self._composite_score(h)
            hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)
        return hypotheses

    def _enrich_path(self, raw_path: list[str]) -> list[HypothesisLink]:
        """Convert a raw node-ID path into rich HypothesisLink objects."""
        links = []
        for i in range(len(raw_path) - 1):
            src_id, tgt_id = raw_path[i], raw_path[i + 1]
            if not self.G.has_edge(src_id, tgt_id):
                continue

            edge_data = self.G.edges[src_id, tgt_id]
            src_node = self._index.get(src_id)
            tgt_node = self._index.get(tgt_id)

            claim_id = edge_data.get("metadata", {}).get("claim_id", "")
            claim_node = self._index.get(claim_id) if claim_id else None

            evidence = {}
            paper = {}
            raw_text = ""

            if claim_node and claim_node.metadata:
                meta = claim_node.metadata
                evidence = meta.get("evidence", {})
                paper = meta.get("source_paper", {})
                raw_text = meta.get("raw_text", "")

            links.append(HypothesisLink(
                from_id=src_id,
                from_name=src_node.preferred_name if src_node else src_id,
                to_id=tgt_id,
                to_name=tgt_node.preferred_name if tgt_node else tgt_id,
                relation_type=edge_data.get("relation_type", "unknown"),
                confidence=edge_data.get("confidence", 0.5),
                claim_id=claim_id,
                raw_text=raw_text,
                evidence=evidence,
                source_paper=paper,
            ))

        return links

    # ── scoring ────────────────────────────────────────────────────────

    def compute_frequency_boost(self, claim_meta: dict) -> float:
        """Frequency boost based on independent primary study replication.

        Same (subject, predicate, object) triple appearing in multiple
        independent primary studies → higher boost.
        """
        key = (
            claim_meta.get("subject_id", ""),
            claim_meta.get("predicate", ""),
            claim_meta.get("object_id", ""),
        )
        all_claims = self._claims_by_triple.get(key, [])
        primary_pmids = set()
        for c in all_claims:
            st = c.get("evidence", {}).get("study_type", "")
            if st not in _REVIEW_TYPES:
                pmid = c.get("source_paper", {}).get("pmid", "")
                if pmid:
                    primary_pmids.add(pmid)

        if len(primary_pmids) >= 3:
            return 1.2   # multiple independent validations
        elif len(primary_pmids) >= 1:
            return 1.0   # has primary support
        else:
            return 0.5   # review-only, no primary source

    @staticmethod
    def compute_temporal_decay(claim_meta: dict, reference_year: int = 2026) -> float:
        """Temporal decay: newer primary studies get higher weight.

        Reviews get no time bonus (1.0). Primary studies decay 3% per year, floor 0.7.
        """
        st = claim_meta.get("evidence", {}).get("study_type", "")
        if st in _REVIEW_TYPES:
            return 1.0
        year = claim_meta.get("source_paper", {}).get("year", 0)
        if not year:
            return 0.85  # unknown year, neutral
        age = reference_year - year
        return max(0.7, 1.0 - 0.03 * age)

    def _compute_confidence_score(self, path: list[HypothesisLink]) -> float:
        """Confidence = geometric_mean(per-link scores).

        Per-link score = base × study_type_weight × frequency_boost
                        × temporal_decay × p_value_bonus × sample_size_bonus
        """
        if not path:
            return 0.0

        scores = []
        for link in path:
            s = link.confidence

            # Study type weight
            study_weight = STUDY_TYPE_WEIGHTS.get(
                link.evidence.get("study_type", ""), 0.3
            )

            # Frequency boost & temporal decay (from claim metadata)
            claim_meta = link.evidence  # evidence dict has study_type, p_value, etc.
            # Also try to get the full claim metadata from source_paper
            # For HypothesisLink, evidence and source_paper are separate dicts
            full_meta = {
                "evidence": link.evidence,
                "source_paper": link.source_paper,
                "subject_id": link.from_id,
                "predicate": link.relation_type,
                "object_id": link.to_id,
            }
            freq_boost = self.compute_frequency_boost(full_meta)
            temp_decay = self.compute_temporal_decay(full_meta)

            # Statistical info bonuses
            p_val_bonus = 1.5 if link.evidence.get("p_value") else 1.0
            sample_bonus = 1.3 if link.evidence.get("sample_size") else 1.0

            # Combine: base × study_weight × freq × decay × stats
            s = s * study_weight * freq_boost * temp_decay * p_val_bonus * sample_bonus

            # Replicability adjustment
            if link.evidence.get("replicability") == "replicated":
                s = min(s + 0.1, 1.0)
            elif link.evidence.get("replicability") == "controversial":
                s = max(s - 0.1, 0.0)

            scores.append(min(s, 1.0))

        return self._geometric_mean(scores)

    def _compute_novelty_score(self, path: list[HypothesisLink]) -> float:
        """Score how novel/surprising a hypothesis is.

        Lower = more expected (direct known relationship), Higher = more surprising.
        """
        score = 0.3  # base

        # hop bonus: longer paths = more novel connections
        score += 0.1 * min(len(path) - 1, 3)

        # cross-domain bonus: connecting different domains is more novel
        domains_seen = set()
        for link in path:
            src = self._index.get(link.from_id)
            tgt = self._index.get(link.to_id)
            if src:
                domains_seen.update(src.domain_tags)
            if tgt:
                domains_seen.update(tgt.domain_tags)
        domains_seen.discard("claim")
        n_domains = len(domains_seen)
        if n_domains >= 3:
            score += 0.15
        elif n_domains >= 2:
            score += 0.10

        # rare relation bonus: non-generic relations are more novel
        rare_count = sum(1 for l in path if l.relation_type not in COMMON_RELATIONS)
        score += 0.05 * min(rare_count, 3)

        # evidence diversity: more papers = better supported, less novel
        # fewer papers = more speculative, more novel
        pmids = {l.source_paper.get("pmid", "") for l in path if l.source_paper.get("pmid")}
        if len(pmids) == 0:
            score += 0.10  # no paper support = speculative but novel
        elif len(pmids) == 1:
            score += 0.05  # single source = weak replication

        return min(score, 1.0)

    def _compute_evidence_score(self, path: list[HypothesisLink]) -> float:
        """Score evidence quality. Uses both structured metadata and text availability."""
        scores = []
        for link in path:
            ev = link.evidence
            s = 0.3  # lower baseline — most claims lack structured stats

            # structured statistical evidence
            p = ev.get("p_value")
            if p is not None:
                if p < 0.001:
                    s += 0.25
                elif p < 0.01:
                    s += 0.20
                elif p < 0.05:
                    s += 0.15

            n = ev.get("sample_size")
            if n is not None:
                if n > 1000:
                    s += 0.15
                elif n > 100:
                    s += 0.10
                elif n > 30:
                    s += 0.05

            if ev.get("effect_size") is not None:
                s += 0.10

            # text-based evidence: having a raw sentence = better than nothing
            if link.raw_text and len(link.raw_text) > 20:
                s += 0.10

            # claim-backed: link was extracted from a structured claim
            if link.claim_id:
                s += 0.05

            # PMID presence: traceable source
            if link.source_paper.get("pmid"):
                s += 0.05

            scores.append(min(s, 1.0))

        return self._geometric_mean(scores)

    def _compute_testability_score(self, path: list[HypothesisLink]) -> tuple[float, str]:
        """Score how testable a hypothesis is with NeuroClaw imaging experiments.

        Boosts for:
        - Brain region features directly measurable from sMRI (volume, thickness)
        - Connectivity features (functional/structural) for GNN models
        - Modalities available in UKB/ADNI/HCP-YA
        - Deep learning model compatibility (BrainGNN, NeuroStorm)
        - Target diseases present in datasets (AD, PD, depression, etc.)

        Returns (score, reason_string).
        """
        all_text = " ".join(
            l.raw_text + " " + l.from_name + " " + l.to_name + " " + l.relation_type
            for l in path
        ).lower()

        # check which modalities are mentioned
        matched_modalities = []
        for modality, keywords in TESTABLE_MODALITIES.items():
            for kw in keywords:
                if kw.lower() in all_text:
                    matched_modalities.append(modality)
                    break

        if not matched_modalities:
            return 0.15, "no imaging modality detected"

        score = 0.25  # base for having a modality

        # modality bonus (more = more testable angles)
        score += 0.10 * min(len(matched_modalities), 3)

        # heavy bonus for sMRI features (volume/thickness — directly measurable in all 3 datasets)
        if "sMRI" in matched_modalities:
            score += 0.15

        # heavy bonus for connectivity features (input to BrainGNN/GNN models)
        if "dMRI" in matched_modalities or "fMRI" in matched_modalities:
            score += 0.15

        # bonus for PET (available in ADNI, key for AD research)
        if "PET" in matched_modalities:
            score += 0.10

        # bonus for brain region specificity (testable with atlas parcellation)
        brain_region_keywords = ["cortex", "hippocampus", "amygdala", "thalamus",
                                 "cerebellum", "striatum", "insula", "gyrus",
                                 "caudate", "putamen", "pallidum", "accumbens",
                                 "precuneus", "cuneus", "lingual", "fusiform",
                                 "parahippocampal", "entorhinal", "parietal",
                                 "frontal", "temporal", "occipital"]
        regions_found = [kw for kw in brain_region_keywords if kw in all_text]
        if regions_found:
            score += 0.10  # atlas-based ROI analysis
            if len(regions_found) >= 2:
                score += 0.05  # pair of regions = connectivity hypothesis

        # bonus for diseases present in target datasets
        dataset_diseases = [
            "alzheimer", "parkinson", "depression", "schizophrenia", "adhd",
            "autism", "epilepsy", "multiple sclerosis", "anxiety", "bipolar",
            "dementia", "mci", "mild cognitive",
        ]
        if any(d in all_text for d in dataset_diseases):
            score += 0.05

        # bonus for DL-model-compatible features (graph structure, ROI, connectivity matrix)
        if any(kw.lower() in all_text for kw in DL_MODEL_KEYWORDS):
            score += 0.05

        # build reason string
        modalities_str = ", ".join(matched_modalities)
        reason = f"modalities: {modalities_str}"
        if regions_found:
            reason += f" | brain regions: {', '.join(regions_found[:4])}"
        if any(d in all_text for d in dataset_diseases):
            matched_diseases = [d for d in dataset_diseases if d in all_text]
            reason += f" | diseases: {', '.join(matched_diseases[:3])}"

        return min(score, 1.0), reason

    def _composite_score(self, h: Hypothesis) -> float:
        c = max(h.confidence_score, 0.01)
        e = max(h.evidence_score, 0.01)
        n = max(h.novelty_score, 0.01)
        t = max(h.testability_score, 0.01)
        # testability weighted highest — hypotheses must be experimentally verifiable
        # with NeuroClaw imaging pipeline (sMRI/fMRI/dMRI/PET + BrainGNN/NeuroStorm)
        return (c ** 0.20) * (e ** 0.20) * (n ** 0.25) * (t ** 0.35)

    def _check_contradiction(self, m1: dict, m2: dict) -> float:
        """Check if two claims contradict each other. Returns severity 0-1."""
        p1 = m1.get("predicate", "")
        p2 = m2.get("predicate", "")
        n1 = m1.get("negated", False)
        n2 = m2.get("negated", False)

        if p1 == p2 and n1 != n2:
            return 1.0

        if (p1, p2) in OPPOSING_PREDICATES:
            return 0.8

        if p1 == p2 and not n1 and not n2:
            d1 = m1.get("evidence", {}).get("direction", "")
            d2 = m2.get("evidence", {}).get("direction", "")
            if d1 and d2 and d1 != d2:
                return 0.6

        return 0.0

    def _infer_relation(self, path: list[str]) -> str:
        """Infer a potential relation from a path's edge types."""
        relations = []
        for i in range(len(path) - 1):
            if self.G.has_edge(path[i], path[i + 1]):
                rt = self.G.edges[path[i], path[i + 1]].get("relation_type", "")
                if rt and rt not in ("about", "is_a", "part_of"):
                    relations.append(rt)

        if relations:
            for r in relations:
                if r not in COMMON_RELATIONS:
                    return r
            return relations[0]
        return "associated_with"

    def _generate_explanation(self, h: Hypothesis) -> str:
        """Generate a human-readable explanation for a hypothesis."""
        path_str = " --> ".join(
            f"{l.from_name} --[{l.relation_type}]--> {l.to_name}" for l in h.path
        )
        if not path_str:
            return ""

        pmids = {l.source_paper.get("pmid", "") for l in h.path if l.source_paper.get("pmid")}
        key_finding = ""
        for l in h.path:
            if l.raw_text:
                key_finding = l.raw_text[:150]
                if len(l.raw_text) > 150:
                    key_finding += "..."
                break

        lines = [
            f"Hypothesis: {h.source_name} may relate to {h.target_name} via {len(h.path)}-hop path.",
            f"Path: {path_str}",
            f"Evidence: {len(h.supporting_claims)} claims from {len(pmids)} papers",
        ]
        if key_finding:
            lines.append(f"Key finding: '{key_finding}'")
        if h.testability_reason:
            lines.append(f"Testability: {h.testability_reason}")
        lines.append(
            f"Confidence: {h.confidence_score:.2f} | "
            f"Novelty: {h.novelty_score:.2f} | "
            f"Evidence: {h.evidence_score:.2f} | "
            f"Testability: {h.testability_score:.2f}"
        )
        return "\n".join(lines)

    @staticmethod
    def _geometric_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        product = math.prod(values)
        return product ** (1.0 / len(values))


def _simple_slope(xs: list[int], ys: list[int]) -> float:
    """Simple linear regression slope without numpy."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den
