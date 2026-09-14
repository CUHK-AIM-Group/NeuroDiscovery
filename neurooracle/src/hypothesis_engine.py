"""Hypothesis engine: batch-generate, persist, and rank testable hypotheses.

Phase 3 of the NeuroClaw discovery loop:
  1. batch_generate() — traverse the graph to produce hypotheses at scale
  2. save / load — persist hypotheses to JSON
  3. rank_hypotheses() — sort by novelty, evidence, testability, confidence
  4. (Phase 5-6) hypotheses become executable NeuroClaw analysis tasks

Usage:
    from neurooracle import load_graph, HypothesisEngine

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
import hashlib
import heapq
import itertools
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import networkx as nx

from .graph_manager import KnowledgeGraph
from .atoms import Atom
from .schema import ConceptNode
from .node_alias_safety import allow_substring_fallback, blocked_canonical_ids
from .feedback_state import FeedbackState, SUPPORTED
from .claim_semantics import (
    ClaimEndpointAudit,
    SemanticEndpoint,
    audit_claim_endpoints,
    concept_atom_roles,
    is_specific_functional_imaging_readout,
    looks_like_cognitive_task_or_stimulus,
    looks_like_imaging_measurement,
    looks_like_non_task_construct,
    semantic_claim_pair,
)
from .case_study_relation_contracts import (
    case_study_endpoint_names_allowed,
    case_study_pair_allowed,
    endpoint_matches_atom,
)
from .case1_hypothesis import (
    case1_directional_statement,
    case1_directional_title,
    propose_case1_direction,
)

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
    kge_score: float | None = None
    kge_attestation: float | None = None
    surprise_gap: float | None = None
    specificity_score: float | None = None
    specificity_issues: list[str] = field(default_factory=list)

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

# Review-only study types (no independent empirical evidence).
# Used by compute_frequency_boost and compute_temporal_decay. Edge-level
# weighting by study_type lives in phase4_optimize.apply_evidence_weighting.
_REVIEW_TYPES = {"review", "narrative_review", "systematic_review"}


def _is_review_study_type(value: object) -> bool:
    """Return whether serialized study-type metadata is review-only."""
    if isinstance(value, (list, tuple, set, frozenset)):
        populated = [item for item in value if str(item or "").strip()]
        return bool(populated) and all(_is_review_study_type(item) for item in populated)
    return str(value or "").strip().lower() in _REVIEW_TYPES

COMMON_RELATIONS = {"is_a", "part_of", "associated_with", "about", "is_associated_with"}

# Pure taxonomy / provenance edges. A node connected ONLY by these has no
# empirical evidence anchoring it to anything mechanistic — a "MeSH disease
# leaf" with only `is_a` parents, or a concept that exists only as the
# subject of `about` provenance edges. Walking through such nodes produces
# hypotheses that look like graph paths but carry zero biological signal.
TREE_RELATIONS = frozenset({"is_a", "part_of", "about"})

# Noisy entity name patterns — hypotheses involving these are low quality.
# Two categories:
#   (a) process-word ≠ entity: nominalized verbs/states ("loss", "progression")
#       that pop up as bridge nodes but carry no biological content.
#   (b) generic containers: vague collective terms ("tissue volumes", "Family")
#       that don't refer to a specific measurable thing.
_NOISE_WORDS = frozenset({
    # original set
    "unseen", "risk", "effect", "level", "status", "change", "type",
    "group", "factor", "model", "method", "unknown", "other", "none",
    "miscellaneous", "various", "difference", "increase", "decrease",
    # nominalized processes/states (category a)
    "loss", "progression", "reduction", "elevation", "alteration",
    "disruption", "dysfunction", "impairment", "deterioration",
    "improvement", "recovery", "response", "onset", "activation",
    "inhibition", "regulation", "modulation", "stimulation",
    "expression", "function", "functions",
    # generic containers (category b)
    "family", "members", "phenomenon", "phenomena", "processes",
    "mechanisms", "pathways", "symptoms", "manifestations",
    "volumes", "volume",
    # life events / demographics that are not biological entities
    "stress", "life", "events", "exposure", "outcome", "outcomes",
    "quality",
})

_NOISE_STOPWORDS = frozenset({
    "a", "an", "and", "by", "for", "in", "of", "or", "the", "to", "with",
})

NOISE_PATTERNS = [
    re.compile(r"^[A-Z][a-z]?$"),                                  # 1-2 letter: "Id", "Ca", "Mg"
    re.compile(r"^[A-Z][a-z]{2,4}$"),                              # Short mixed-case: "Tics", "Risk"
    re.compile(r"^\d+$"),                                           # Pure numbers
]

# (C-1) Generic-phrase patterns for INTERMEDIATE nodes. The token-based
# `_NOISE_WORDS` filter misses phrases like "functional connectivity" or
# "neural activity" because no individual word is in the noise list, but
# the WHOLE phrase carries no measurable content. We only block these when
# they appear as INTERMEDIATE nodes (paths can legitimately end in
# "functional connectivity" as an outcome metric).
_GENERIC_INTERMEDIATE_PATTERNS = [
    re.compile(r"^(abnormal|altered|impaired|reduced|increased|disrupted|aberrant)?\s*"
               r"(brain|neural|neuronal|cortical|cerebral)\s+"
               r"(activity|activation|function|functioning|connectivity|"
               r"network|networks|signaling|metabolism|response|responses)$",
               re.I),
    re.compile(r"^(functional|structural|anatomical|effective)\s+"
               r"(connectivity|network|networks|integrity|abnormalit(y|ies))$", re.I),
    re.compile(r"^(disease|symptom|clinical|treatment|therapeutic)\s+"
               r"(progression|outcome|outcomes|response|severity|burden|stage|staging)$", re.I),
    re.compile(r"^(common|typical|specific|various|different)\s+"
               r"(features|patterns|mechanisms|processes)$", re.I),
    re.compile(r"^(neuro)?(degeneration|inflammation|protection|plasticity|genesis|imaging)$",
               re.I),
    re.compile(r"^(grey|gray|white)\s+matter$", re.I),
    re.compile(r"^(cognitive|behavioral|emotional|motor|sensory)\s+"
               r"(deficit|deficits|dysfunction|impairment|abnormalit(y|ies))$", re.I),
]

# (C-3) Target-name patterns that LOOK like outcomes (so they pass
# _is_dataset_outcome's keyword fallback) but are actually too broad to
# drive a DL experiment. We block these even if their domain says
# disease/cognitive_function.
_TARGET_TOO_BROAD_PATTERNS = [
    # bare umbrella nouns (single token)
    re.compile(r"^(skill|skills|ability|abilities|outcome|outcomes|"
               r"symptom|symptoms|manifestation|manifestations|"
               r"phenomenon|phenomena|finding|findings|"
               r"deficit|deficits|impairment|impairments|"
               r"function|functions|functioning|behavior|behaviors|"
               r"capability|capabilities|condition|conditions|"
               r"disease|diseases|disorder|disorders|syndrome|syndromes|"
               r"focus|integration|balance|knowledge|autonomy|"
               r"performance|adaptation|resilience|vulnerability|"
               r"recovery|progression|mechanism|process)$", re.I),
    # broad-category disease umbrellas (when these are the literal target,
    # they're too generic — but specific subtypes like "Alzheimer Disease"
    # don't match these patterns)
    re.compile(r"^(neurological|psychiatric|mental|cognitive|behavioral|"
               r"neurodegenerative|cardiovascular)\s+"
               r"(disease|diseases|disorder|disorders|condition|conditions)$", re.I),
    re.compile(r"^(human\s+)?(disease|diseases|disorder|disorders)$", re.I),
    re.compile(r"^(brain|mental|psychiatric|psychological)\s+health$", re.I),
    re.compile(r"^clinical\s+(features|outcome|outcomes|presentation|status)$", re.I),
    # "X deficits/impairments" patterns (too vague as targets)
    re.compile(r"^(motor|cognitive|neurocognitive|functional|social|"
               r"verbal|visual|sensory|emotional|behavioral)\s+"
               r"(deficit|deficits|impairment|impairments|dysfunction|"
               r"disability|decline|deterioration)$", re.I),
]

# Vague relation types that add little signal
VAGUE_RELATIONS = {"is_associated_with", "associated_with", "about"}

# (P2) Umbrella source-name patterns. These are entities that pass the
# `_is_generic_intermediate` / `_is_too_broad_target` filters because they
# look like specific biological objects, but in practice they are umbrella
# nouns that don't constrain a downstream DL experiment as a SOURCE seed.
#
# Empirically (cycle_001 audit): 519 / 1054 hypotheses (49%) seeded from
# these umbrellas. Examples in the top-15 source pool:
#   "scalp EEG", "neuroimaging", "magnetic resonance spectroscopy"
#       -> measurement modality, not an entity to predict from
#   "cortical reorganization", "synaptic plasticity"
#       -> abstract process; no concrete biomarker to feed a model
#   "intestinal microbiota", "Neuroglia", "Nervous System"
#       -> biological super-categories
#   "high inflammation", "EEG abnormalities"
#       -> qualitative state without measurable axis
#   "direct pathway", "neurovascular unit", "Corpus Callosum"
#       -> anatomical umbrellas (specific subregions are still allowed)
#
# Only blocks SOURCE seed selection. The same name may be valid as an
# intermediate (carrying mechanism) or as a target outcome.
_UMBRELLA_SOURCE_PATTERNS = [
    # Measurement / imaging modalities used as entity names
    re.compile(r"^(scalp\s+)?(eeg|meg|fmri|mri|pet|ct|ecg)"
               r"(\s+(abnormalit(y|ies)|finding|findings|signal|signals|"
               r"data|recording|recordings|measurement|measurements))?$", re.I),
    re.compile(r"^(neuro)?imaging$", re.I),
    re.compile(r"^(functional|structural|diffusion|resting[\s-]+state)\s+"
               r"(mri|imaging|connectivity)$", re.I),
    re.compile(r"^(magnetic\s+resonance\s+(imaging|spectroscopy)|"
               r"positron\s+emission\s+tomography|"
               r"electroencephalogra(phy|m)|"
               r"magnetoencephalogra(phy|m))$", re.I),

    # Abstract processes / states (no measurable axis to seed from)
    re.compile(r"^(cortical|neural|synaptic|brain)\s+"
               r"(reorganization|remodeling|adaptation|plasticity)$", re.I),
    re.compile(r"^(neuro)?(plasticity|inflammation|degeneration|protection|"
               r"genesis|modulation|transmission)$", re.I),
    re.compile(r"^(high|low|elevated|reduced|increased|decreased|chronic|acute)\s+"
               r"(inflammation|stress|activity|excitability|connectivity)$", re.I),

    # System / super-category nouns
    re.compile(r"^(central|peripheral|autonomic|somatic)?\s*nervous\s+system$", re.I),
    re.compile(r"^(neurogli(a|al\s+cells)|glia|glial\s+cells|neurons?)$", re.I),
    re.compile(r"^(immune|endocrine|cardiovascular|gastrointestinal)\s+system$", re.I),
    re.compile(r"^(intestinal|gut|oral|skin)\s+(microbiota|microbiome|flora)$", re.I),

    # Pathway / circuit umbrellas (specific subcomponents like "D1 MSN" still pass)
    re.compile(r"^(direct|indirect|hyperdirect)\s+pathway$", re.I),
    re.compile(r"^(neurovascular|neuromuscular)\s+unit$", re.I),

    # Generic anatomy umbrellas at the gross level (specific subnuclei still pass:
    # "CA1", "ventral striatum", "BA17" are not blocked)
    re.compile(r"^(corpus\s+callosum|basal\s+ganglia|limbic\s+system|"
               r"reticular\s+formation|brainstem|forebrain|midbrain|hindbrain)$", re.I),
    re.compile(r"^(grey|gray|white)\s+matter$", re.I),

    # Biomarker class umbrellas (specific markers like "anti-MOG" still pass)
    re.compile(r"^(oligoclonal\s+bands|inflammatory\s+markers?|"
               r"oxidative\s+stress\s+markers?)$", re.I),

    # ── P2 v1.5 patterns added 2026-05-24 after first run exposed ──
    # second-order umbrellas (hubs/circuitry/networks/systems suffixes,
    # vague modifiers, sample-size descriptors).

    # Bare topology nouns (single token)
    re.compile(r"^(hubs?|circuits?|circuitries|networks?|systems?|"
               r"pathways?|connections?|wirings?)$", re.I),

    # Common-modifier + topology suffix (1-token modifier)
    # Specific named networks like "Default Mode Network" / "salience network"
    # would match here too, so exclude them in `_named_network_exception`.
    # Empirically this catches "connectivity hubs", "network hubs",
    # "metabolic networks", "reward circuitry", "sensory systems",
    # "fronto-striatal circuitry", "prefrontal network".
    re.compile(r"^(connectivity|network|reward|sensory|motor|cognitive|emotional|"
               r"limbic|cortical|subcortical|neural|brain|metabolic|"
               r"prefrontal|frontal|parietal|temporal|occipital|"
               r"striato-?\w*|fronto-?\w*|cortico-?\w*|cerebro-?\w*)\s+"
               r"(hubs?|circuitr(y|ies)|circuits?|networks?|systems?)$",
               re.I),

    # Vague qualifier + (optional middle token) + topology / connectivity
    # Catches "selected neural circuits", "shared neural networks",
    # "between-network functional connectivity changes",
    # "undirected functional connectivity alterations".
    # NOTE: `static`/`dynamic` are NOT vague (they name fMRI analysis
    # types like "static functional connectivity"), so they are excluded.
    re.compile(r"^(selected|shared|altered|aberrant|abnormal|impaired|"
               r"reduced|increased|decreased|distributed|widespread|"
               r"specific|various|different|key|core|main|primary|"
               r"between-network|within-network|undirected|directed|"
               r"true)\s+"
               r"(\w+(-\w+)?\s+){0,3}"
               r"(hubs?|circuits?|circuitr(y|ies)|networks?|systems?|"
               r"connectivity|connections?|"
               r"changes?|alterations?|disruptions?|disturbances?|"
               r"abnormalit(y|ies)|deficits?|impairments?|dysfunctions?|"
               r"neurodegeneration|degeneration|inflammation|damage)$",
               re.I),

    # Connectivity-as-noun wrapped in change-words ("X functional connectivity changes")
    re.compile(r"^.*(connectivity|network)\s+"
               r"(changes?|alterations?|abnormalit(y|ies)|"
               r"disruptions?|disturbances?)$", re.I),

    # Treatment / disease "effects" / "outcomes" as a source.
    # Too vague to seed from ("treatment effects -> X" doesn't say which
    # treatment, which effect axis). Specific outcomes like "PASI score"
    # or "MMSE decline" are unaffected.
    re.compile(r"^(treatment|disease|therapy|therapeutic|clinical)\s+"
               r"(effects?|outcomes?|response|responses)$", re.I),

    # Sample-size descriptors (claim_extractor noise: "250 healthy controls")
    re.compile(r"^\d+\s+(healthy|normal|patient|patients?|controls?|"
               r"subjects?|participants?|individuals?|men|women|"
               r"adults|children|adolescents|elderly|cases?)\b", re.I),

    # Generic process+functioning compounds
    re.compile(r"^(gi|gut|metabolic|immune|cognitive|emotional|"
               r"behavioral|social|sensorimotor|autonomic)\s+"
               r"(functioning?|regulation|processing|control|"
               r"dysregulation|dysfunction)$", re.I),

    # Bare process nouns at single-token level (coupled with "X" prefix
    # like "structural damage", "iron deposition" stay specific via
    # required prefix; we only block the bare forms here).
    re.compile(r"^(neurodegeneration|neuroinflammation|neuromodulation|"
               r"neurogenesis|neuroprotection|neuroplasticity|"
               r"oxidation|reduction|signaling|transmission)$", re.I),
]

# CognitiveAtlas / MeSH concept ids that are top-degree generic hubs
# in the KG. The audit found these at degrees 700-9000+, with names that
# are real English words (not caught by _NOISE_WORDS) but referring to
# extremely abstract umbrella concepts:
#
#   COGAT trm_4a3fd79d0a891  "memory"      degree 2248
#   COGAT trm_4a3fd79d0a80f  "logic"       degree 2052
#   COGAT trm_5159c80c1dd24  "loss"        degree 1034
#   COGAT trm_4a3fd79d09741  "activation"  degree  840
#   COGAT trm_4a3fd79d0afcf  "risk"        degree  722
#   COGAT trm_4a3fd79d0b2a8  "stress"      degree  139
#   MSH:D001921              "Brain"       degree 9157
#   MSH:D009474              "Neurons"     degree 1354
#
# Hypotheses with these as intermediate nodes or endpoints are too vague
# to drive a downstream DL experiment ("FPN -> memory" is not testable
# because we don't know which memory subsystem). Filtered in post_process.
# Seed tables: (pre-UMLS id, expected preferred_name). After UMLS
# canonicalization the original id may have been remapped to CUI:Cxxx, so
# the engine resolves these to live KG ids via id-or-name lookup at init.
_PATH_IGNORE_SEED: tuple[tuple[str, str], ...] = (
    ("COGAT_CONCEPT:trm_4a3fd79d0a891",   "memory"),
    ("COGAT_CONCEPT:trm_4a3fd79d0a80f",   "logic"),
    ("COGAT_CONCEPT:trm_5159c80c1dd24",   "loss"),
    ("COGAT_CONCEPT:trm_4a3fd79d09741",   "activation"),
    ("COGAT_CONCEPT:trm_4a3fd79d0afcf",   "risk"),
    ("COGAT_CONCEPT:trm_4a3fd79d0b2a8",   "stress"),
    ("MSH:D001921",                       "Brain"),
    ("MSH:D009474",                       "Neurons"),
)

# Disease/category mega-hubs that are valid as hypothesis endpoints
# ("predict Alzheimer" is fine) but NOT as intermediate transit nodes
# ("A -> Alzheimer -> B" is just "A relates to AD, AD relates to B" — no
# discovery value). Audit found 37.8% of hypotheses transit through these.
# 2026-06-02: added 11 mega-hubs exposed by UMLS canonicalization (anatomy
# umbrellas + broad disease/symptom terms that accumulated cross-vocab edges).
_INTERMEDIATE_ONLY_SEED: tuple[tuple[str, str], ...] = (
    ("COGAT_DISORDER:dso_5419",           "schizophrenia"),
    ("MSH:D009103",                       "Multiple Sclerosis"),
    ("COGAT_DISORDER:dso_3312",           "bipolar disorder"),
    ("MSH:D000544",                       "Alzheimer Disease"),
    ("MSH:D004827",                       "Epilepsy"),
    ("MSH:D010300",                       "Parkinson Disease"),
    ("COGAT_DISORDER:dso_0060041",        "autism spectrum disorder"),
    ("MSH:D001289",                       "Attention Deficit Disorder with Hyperactivity"),
    ("MSH:D003863",                       "Depression"),
    ("MSH:D001523",                       "Mental Disorders"),
    ("MSH:D012640",                       "Seizures"),
    ("MSH:D003704",                       "Dementia"),
    ("MSH:D001321",                       "Autistic Disorder"),
    ("MSH:D060825",                       "Cognitive Dysfunction"),
    ("COGAT_DISORDER:dso_1094",           "attention deficit hyperactivity disorder"),
    ("MSH:D001714",                       "Bipolar Disorder"),
    ("MSH:D010842",                       "Pica"),
    ("COGAT_CONCEPT:trm_4a3fd79d09902",   "attention"),
    ("MSH:D001519",                       "Behavior"),
    ("COGAT_CONCEPT:trm_4a3fd79d09735",   "action"),
    ("MSH:D004644",                       "Emotions"),
    # Anatomy umbrella mega-hubs (post-canonicalize degree 900+)
    ("CUI:C0152279",                      "Lateral Ventricles"),
    ("CUI:C0010090",                      "Corpus Callosum"),
    ("CUI:C0007776",                      "Cerebral Cortex"),
    ("CUI:C0007765",                      "Cerebellum"),
    ("CUI:C0039452",                      "Telencephalon"),
    ("NN:3000",                           "Ventricular System"),
    # Broad disease/symptom terms (post-canonicalize degree 600+)
    ("CUI:C0025363",                      "Intellectual Disability"),
    ("CUI:C0557874",                      "Global developmental delay"),
    ("CUI:C1864897",                      "Cognitive delay"),
    ("CUI:C0026825",                      "Muscle Hypotonia"),
    ("CUI:C0011573",                      "Depressive Disorder"),
)

# Frozensets retained for any external import; engine instances use the
# resolved sets built in __init__ instead.
PATH_IGNORE_NODE_IDS = frozenset(nid for nid, _ in _PATH_IGNORE_SEED)
INTERMEDIATE_ONLY_IGNORE_IDS = frozenset(nid for nid, _ in _INTERMEDIATE_ONLY_SEED)


def _resolve_blacklist(
    seeds: tuple[tuple[str, str], ...],
    index: dict,
) -> frozenset[str]:
    """Resolve (original_id, expected_name) pairs to live KG ids.

    For each seed: keep original_id if still present; otherwise look up by
    preferred_name (case-insensitive) across the KG. Drops seeds that
    resolve to nothing — the corresponding concept simply isn't in this KG.
    """
    name_to_id: dict[str, str] = {}
    for nid, node in index.items():
        nm = (node.preferred_name or "").lower()
        if nm:
            name_to_id.setdefault(nm, nid)
    resolved: set[str] = set()
    for original_id, expected_name in seeds:
        if original_id in index:
            resolved.add(original_id)
            continue
        match = name_to_id.get(expected_name.lower())
        if match is not None:
            resolved.add(match)
    return frozenset(resolved)

DIRECTIONAL_RELATIONS = {
    "causes", "treats", "increases", "reduces", "modulates",
    "activates", "inhibits", "is_biomarker_of", "is_risk_factor_for",
    "predicts", "distinguishes", "mediates",
    # Brain decoding directional predicates
    "evokes", "decoded_from", "elicits",
    # Gene-specific discovery predicates (GENE -> DISEASE / neuroanatomy).
    # All four are canonical single-direction edges (DisGeNET / HPO / AHBA /
    # Hansen 2022) and the schema files them under tier "discovery". Without
    # these, a `Gene -gene_associated_with_disease-> Disease` chain looks
    # non-directional to post_process and gets dropped at the final gate.
    "gene_associated_with_disease",
    "gene_associated_with_anatomy",
    "gene_enriched_in_region",
    "receptor_density_in",
    # IM/region/scale closure edges — each encodes a single semantic
    # direction (feature-of-region, scale-measures-disease, disease-
    # assessed-by-scale) and is what stitches GENE→IM→DISEASE→OUTCOME
    # chains together. Without these the chain reads as a 40%-directional
    # narrative and is dropped by _has_thin_directional_density.
    "has_imaging_feature",
    "is_imaging_feature_of",
    "measures",
    "is_assessed_by",
}


# ── Atom-aware intermediate-node constraint ──────────────────────────────
# Every node visited by a hypothesis path must be a "scientifically
# meaningful" node — i.e. it must play one of the canonical atoms
# (DISEASE/DRUG/IM/GENE/COGNITIVE_TASK/OUTCOME/INDIVIDUAL_DATA). Nodes that
# only carry infrastructure tags (atlas/modality/dataset/ml_model) or
# meta tags (claim/recipe) describe the apparatus, not the science, and
# must not appear on a hypothesis path.
#
# Lazy-imported on first use to avoid circular import with atoms.py.
_ALLOWED_ATOM_DOMAINS_CACHE: Optional[frozenset[str]] = None


def _allowed_atom_domains() -> frozenset[str]:
    global _ALLOWED_ATOM_DOMAINS_CACHE
    if _ALLOWED_ATOM_DOMAINS_CACHE is None:
        from .atoms import ATOM_TO_DOMAINS
        merged: set[str] = set()
        for doms in ATOM_TO_DOMAINS.values():
            merged |= doms
        _ALLOWED_ATOM_DOMAINS_CACHE = frozenset(merged)
    return _ALLOWED_ATOM_DOMAINS_CACHE

# domain pairs worth exploring — aligned with NeuroClaw imaging experiments
# target datasets: UKB (T1w/dMRI/rfMRI/SWI), ADNI (T1w/PET/fMRI/DTI), HCP-YA (T1w/T2w/fMRI/dMRI/MEG)
# experiment models: BrainGNN, NeuroStorm, SVM, XGBoost on raw images + handcrafted features
#
# Design principle: target should be a dataset OUTCOME (what we want to predict),
# source should be a MEASURABLE feature (what the dataset provides as input).
# - UKB outcomes: fluid intelligence, neuroticism, dementia diagnosis, motor tests
# - ADNI outcomes: MCI→AD conversion, CDR-SB, cognitive composite
# - HCP outcomes: fluid/crystallized IQ, emotion recognition, personality traits
#
# Allowed sources (what we can measure): neuroanatomy (MRI regions), connectivity
# networks, gene, biomarker (CSF/PET), drug (for intervention studies).
# Allowed targets (what we predict): disease (diagnostic labels), cognitive_function
# (the OUTCOMES — includes behavior, personality, affect).
DEFAULT_DOMAIN_PAIRS = [
    # core: measurable features → clinical/behavioral OUTCOMES
    ("neuroanatomy", "disease"),             # MRI → diagnosis
    ("neuroanatomy", "cognitive_function"),  # MRI → cognition/behavior
    ("connectivity", "disease"),             # dMRI/fMRI connectivity → diagnosis
    ("connectivity", "cognitive_function"),  # connectivity → cognition
    # genetics → outcomes (UKB 500k WGS)
    ("gene", "disease"),
    ("gene", "cognitive_function"),          # GWAS → behavior/IQ
    # fluid biomarkers → outcomes (ADNI CSF, blood)
    ("biomarker", "disease"),
    ("biomarker", "cognitive_function"),
    # drug → outcomes (ADNI pharmaceutical arms)
    ("drug", "disease"),
    ("drug", "cognitive_function"),
    # cross-outcome (comorbidity, transdiagnostic)
    ("disease", "disease"),
    ("cognitive_function", "disease"),       # e.g. anxiety → MS diagnosis risk
    ("disease", "cognitive_function"),       # e.g. AD → processing speed decline
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

# DATASET-OUTCOME whitelist — covers actual predicted variables in UKB/ADNI/HCP-YA
# papers (see README "Dataset Outcomes" for references to typical prediction tasks).
# Target must match one of these patterns to pass the post_process filter.
# We also auto-accept any concept in the `disease` domain (clinical diagnosis
# IS the most common outcome) and any MSH/CogAtlas concept in the
# `cognitive_function` domain (behavior/cognition).
#
# Categories cover:
# - Clinical diagnostic labels (Alzheimer, schizophrenia, MCI, etc.) — all 3 datasets
# - AD staging / conversion (CN→MCI→AD, ATN) — ADNI
# - Clinical scales (CDR, MMSE, ADAS-Cog, PHQ-9, MoCA, NPI) — ADNI + UKB
# - Cognitive abilities (IQ, memory, attention, processing speed) — all 3
# - Specific cognitive tests (PMAT, flanker, N-back, delay discounting) — HCP
# - Personality (Big Five) — HCP + UKB
# - Behavior/affect (anxiety, depression, aggression, risk-taking) — all 3
# - Motor/sensory (grip strength, gait, reaction time, dexterity) — UKB + HCP
# - Brain age / neurodegeneration markers — UKB + ADNI
# - NeuroSTORM-evaluated phenotypes: MND, early psychosis (HCP-EP), ADHD200,
#   COBRE, UCLA L5c, TCP psychiatric scales, fMRI task state classification
# - Subject fingerprinting / re-identification
_OUTCOME_KEYWORDS = re.compile(
    r"("
    # cognitive abilities — general
    r"intelligence|cognition|cognitive\s+(function|ability|performance|deterioration|impairment|dysfunction|decline|test|assessment|composite|score)|"
    r"memory|attention|executive|processing\s+speed|reasoning|language|"
    r"fluency|perception|reaction\s+time|fluid\s+intelligence|"
    r"crystallized\s+intelligence|working\s+memory|episodic\s+memory|"
    r"semantic\s+memory|verbal\s+(memory|fluency|learning)|visuospatial|"
    # specific HCP NIH Toolbox / cognitive tasks
    r"pmat|flanker|card\s+sort|n-?back|list\s+sort|picture\s+sequence|"
    r"pattern\s+comparison|picture\s+vocabulary|oral\s+reading|"
    r"delay\s+discounting|risk[- ]taking|go[- ]no[- ]go|"
    # HCP Penn CNB cognitive battery
    r"penn\s+(word|matrix|line\s+orientation|continuous\s+performance|progressive\s+matrices|fear|emotion|cnb)|"
    r"matrix\s+pattern|numeric\s+memory|prospective\s+memory|pairs\s+matching|"
    r"trail\s+making|symbol\s+digit|boston\s+naming|animal\s+fluency|"
    r"category\s+fluency|logical\s+memory|clock\s+drawing|ravlt|"
    # HCP 7 task states (NeuroSTORM state classification)
    r"emotion\s+task|gambling\s+task|language\s+task|motor\s+task|"
    r"relational\s+task|social\s+task|working\s+memory\s+task|"
    # clinical scales (ADNI/UKB/TCP/HCP)
    r"\b(cdr|cdr-sb|mmse|moca|adas|adas-cog|npi|faq|gds|phq-?9|gad-?7|bai|hdrs|hrsd|hamd|ham-d|"
    r"bdi|ymrs|panss|sans|saps|audit|asrs|pro|adi|srs|tci|neo-?ffi|asr|abcl|"
    r"cidi|cidi-sf|eysenck|swemwbs|psqi|ftnd|ssaga|masq|promis|upsit)\b|"
    r"adult\s+self\s+report|adult\s+behavior\s+checklist|"
    # personality / affect
    r"neuroticism|extraversion|agreeableness|conscientiousness|openness|"
    r"personality|temperament|affect|mood|emotion|anxiety|depression|"
    r"well-?being|satisfaction|life\s+satisfaction|psychological|stress\s+response|"
    r"anxiety\s+sensitivity|cautiousness|"
    r"affect\s+(positive|negative)|emotion\s+recognition|emotional\s+regulation|"
    r"perceived\s+(stress|rejection|hostility)|anger|fear|sadness|"
    # social functioning (HCP + UKB)
    r"loneliness|social\s+(isolation|support|relationship|cognition)|"
    r"meaning\s+and\s+purpose|instrumental\s+support|emotional\s+support|"
    r"friendship|"
    # behavior
    r"behavior|aggression|impulsivity|addiction|substance|alcohol|smoking|"
    r"tobacco|cannabis|cocaine|opiate|opioid|hallucinogen|"
    r"drug\s+use|substance\s+use|sleep\s+quality|insomnia|"
    # diagnoses / clinical outcomes — added NeuroSTORM-evaluated cohorts and ADNI stages
    r"alzheimer|parkinson|schizophrenia|autism|adhd|bipolar|epilepsy|"
    r"mci|mild\s+cognitive|dementia|psychosis|early\s+psychosis|stroke|post[- ]stroke|"
    r"multiple\s+sclerosis|huntington|frontotemporal|lewy\s+body|"
    r"motor\s+neuron\s+disease|mnd|als|"
    r"transdiagnostic|psychiatric\s+disorder|mental\s+health\s+disorder|"
    r"ocd|ptsd|phobia|panic|agoraphobia|somatoform|eating\s+disorder|"
    # ADNI-specific diagnostic stages
    r"\b(cn|smc|emci|lmci|ad\b|preclinical|at\b|atn|alzheimer\s+continuum)\b|"
    r"significant\s+memory\s+concern|subjective\s+(memory|cognitive)\s+(concern|complaint|decline)|"
    r"cognitively\s+(normal|unimpaired)|"
    r"disorder|syndrome|diagnosis|onset|conversion|progression|severity|"
    r"symptom|manifestation|prognosis|outcome|treatment\s+response|"
    r"disease\s+(stage|staging|duration|burden)|"
    # cardiovascular / metabolic diseases (UKB ICD-10)
    r"myocardial\s+infarction|heart\s+failure|hypertension|atrial\s+fibrillation|"
    r"coronary|cardiovascular\s+disease|diabetes|type\s*[12]\s+diabetes|"
    r"chronic\s+kidney|fatty\s+liver|nafld|metabolic\s+syndrome|obesity|"
    # AD-specific biomarker status
    r"amyloid\s+(status|positivity|positive|negative|load|burden|suvr)|"
    r"tau\s+(status|positivity|positive|tangle|pathology|burden|suvr)|"
    r"atn\s+(profile|stage|classification)|"
    r"neurodegeneration\s+(stage|status)|"
    # brain age / aging
    r"brain\s+age|brain-?age(-?gap)?|aging|age[- ]related|age\s+acceleration|"
    # motor / sensory
    r"grip\s+strength|gait|motor\s+coordination|motor\s+function|"
    r"balance|tremor|dexterity|walking\s+speed|two[- ]minute\s+walk|endurance|"
    r"visual\s+(acuity|field)|audition|hearing|olfaction|taste|pain|"
    r"chronic\s+pain|musculoskeletal\s+pain|"
    # mortality / longevity
    r"mortality|all-?cause\s+death|survival|life\s+expectancy"
    r")", re.I
)

# Target domains considered as valid dataset outcomes
_OUTCOME_DOMAINS = {"disease", "cognitive_function"}

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
    # NAS-available patient cohorts with preprocessed ROI time series.
    # Phenotype CSVs live under Z:\Dataset\fMRI\phenotype and the dataset-
    # specific rest csvs. All supply rfMRI volumes or ROI series; structural
    # T1 is available for HCP-EP and HCP-Aging (the other four are rfMRI-only
    # public releases).
    "ABIDE": {
        "rfmri_fc":     {"modality": "fMRI", "tool": "rfMRI",       "level": "connectivity"},
        "rfmri_roi_ts": {"modality": "fMRI", "tool": "rfMRI",       "level": "ROI"},
    },
    "ADHD200": {
        "rfmri_fc":     {"modality": "fMRI", "tool": "rfMRI",       "level": "connectivity"},
        "rfmri_roi_ts": {"modality": "fMRI", "tool": "rfMRI",       "level": "ROI"},
    },
    "COBRE": {
        "rfmri_fc":     {"modality": "fMRI", "tool": "rfMRI",       "level": "connectivity"},
        "rfmri_roi_ts": {"modality": "fMRI", "tool": "rfMRI",       "level": "ROI"},
    },
    "UCLA": {
        # UCLA CNP — rest + 6 task contrasts, cross-diagnosis cohort.
        "rfmri_fc":     {"modality": "fMRI", "tool": "rfMRI",       "level": "connectivity"},
        "rfmri_roi_ts": {"modality": "fMRI", "tool": "rfMRI",       "level": "ROI"},
        "tfmri_task":   {"modality": "fMRI", "tool": "task fMRI",   "level": "activation"},
    },
    "HCP_EP": {
        # HCP Early Psychosis — patient cohort, T1w + rfMRI cleaned.
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_subcortical_volume": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "rfmri_fc":     {"modality": "fMRI", "tool": "rfMRI",       "level": "connectivity"},
        "rfmri_roi_ts": {"modality": "fMRI", "tool": "rfMRI",       "level": "ROI"},
    },
    "HCP_AGING": {
        # HCP-Aging — T1w + rfMRI REST1/REST2 + 3 task contrasts.
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_subcortical_volume": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_myelin":             {"modality": "sMRI", "tool": "T1w/T2w",    "level": "ROI"},
        "rfmri_fc":     {"modality": "fMRI", "tool": "rfMRI",       "level": "connectivity"},
        "rfmri_roi_ts": {"modality": "fMRI", "tool": "rfMRI",       "level": "ROI"},
        "tfmri_task":   {"modality": "fMRI", "tool": "task fMRI",   "level": "activation"},
    },
    "TCP": {
        # Transdiagnostic Connectome Project — BIDS release includes T1w/T2w
        # anatomical scans, resting-state fMRI, and task fMRI.
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_subcortical_volume": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "smri_myelin":             {"modality": "sMRI", "tool": "T1w/T2w",    "level": "ROI"},
        "rfmri_fc":                {"modality": "fMRI", "tool": "rfMRI",      "level": "connectivity"},
        "rfmri_roi_ts":            {"modality": "fMRI", "tool": "rfMRI",      "level": "ROI"},
        "tfmri_task":              {"modality": "fMRI", "tool": "task fMRI",  "level": "activation"},
    },
    # ── Visual decoding (fMRI) ──────────────────────────────────────────
    # NSD & BOLD5000: image-stimulus visual task fMRI, no rest.
    "NSD": {
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "tfmri_visual_voxel":      {"modality": "fMRI", "tool": "task fMRI",
                                     "level": "voxel", "stimulus": "natural_image"},
        "tfmri_visual_roi":        {"modality": "fMRI", "tool": "task fMRI",
                                     "level": "ROI",   "stimulus": "natural_image"},
    },
    "BOLD5000": {
        "smri_cortical_thickness": {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI"},
        "tfmri_visual_voxel":      {"modality": "fMRI", "tool": "task fMRI",
                                     "level": "voxel", "stimulus": "ImageNet_COCO_Scene"},
        "tfmri_visual_roi":        {"modality": "fMRI", "tool": "task fMRI",
                                     "level": "ROI",   "stimulus": "ImageNet_COCO_Scene"},
    },
    # ── Visual decoding (EEG) ───────────────────────────────────────────
    "SEED_DV": {
        "eeg_psd": {"modality": "EEG", "tool": "PSD", "level": "channel"},
        "eeg_de":  {"modality": "EEG", "tool": "DE",  "level": "channel"},
    },
    # ── Emotion decoding (EEG + eye tracking) ───────────────────────────
    "SEED": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eeg_psd":      {"modality": "EEG", "tool": "PSD", "level": "channel"},
    },
    "SEED_IV": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eye_movement": {"modality": "eye_tracking", "tool": "saccade/fixation",
                         "level": "variable"},
    },
    "SEED_V": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eye_movement": {"modality": "eye_tracking", "tool": "saccade/fixation",
                         "level": "variable"},
    },
    "SEED_VII": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eye_movement": {"modality": "eye_tracking", "tool": "saccade/fixation",
                         "level": "variable"},
    },
    "SEED_GER": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eye_movement": {"modality": "eye_tracking", "tool": "saccade/fixation",
                         "level": "variable"},
    },
    "SEED_FRA": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eye_movement": {"modality": "eye_tracking", "tool": "saccade/fixation",
                         "level": "variable"},
    },
    # ── Vigilance decoding (EEG) ────────────────────────────────────────
    "SEED_VIG": {
        "eeg_de":       {"modality": "EEG", "tool": "DE",  "level": "channel"},
        "eog":          {"modality": "EOG", "tool": "EOG", "level": "channel"},
        "eye_movement": {"modality": "eye_tracking", "tool": "gaze/blink",
                         "level": "variable"},
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
    # ABIDE — ASD vs controls, rest only.
    "ABIDE": [
        "diagnosis",           # ASD vs TD
        "symptom_severity",    # ADOS, ADI-R, SRS
        "cognitive_score",     # FIQ/VIQ/PIQ
    ],
    # ADHD200 — ADHD subtype vs TDC.
    "ADHD200": [
        "diagnosis",           # ADHD (combined/inattentive/hyperactive) vs TDC
        "symptom_severity",    # ADHD-RS, Conners
        "cognitive_score",     # WASI/WISC
    ],
    # COBRE — schizophrenia vs controls.
    "COBRE": [
        "diagnosis",           # schizophrenia vs HC
        "symptom_severity",    # PANSS positive/negative/general
        "cognitive_score",     # WAIS
    ],
    # UCLA CNP — schizophrenia/bipolar/ADHD vs controls.
    "UCLA": [
        "diagnosis",           # SCZ / BP / ADHD / HC
        "symptom_severity",    # HAM-D, YMRS, ADHD-RS
        "cognitive_task",      # 6 task contrasts
    ],
    # HCP-EP — early psychosis (FES + AR) vs HC.
    "HCP_EP": [
        "diagnosis",           # affective/non-affective psychosis vs HC
        "symptom_severity",    # PANSS, SANS, YMRS
        "cognitive_score",     # MATRICS Consensus Cognitive Battery
    ],
    # HCP-Aging — lifespan 36-100 yrs, healthy aging.
    "HCP_AGING": [
        "cognitive_decline",   # NIH Toolbox across age
        "behavioral_score",    # same battery as HCP-YA
        "cognitive_task",      # CARIT/FACENAME/VISMOTOR
    ],
    # ── Visual decoding outcomes ────────────────────────────────────────
    "NSD": [
        "image_category",         # COCO 80-class
        "image_semantic",         # CLIP / language-model embedding
        "stimulus_reconstruction",# pixel / latent reconstruction
    ],
    "BOLD5000": [
        "image_category",         # ImageNet 1000-class / COCO / Scene
        "scene_type",             # Scene 365-class
        "image_semantic",
    ],
    "SEED_DV": [
        "video_class",            # discrete video categories
        "video_semantic",
        "video_reconstruction",
    ],
    # ── Emotion decoding outcomes ───────────────────────────────────────
    "SEED":     ["emotion_3class"],            # positive/neutral/negative
    "SEED_IV":  ["emotion_4class"],            # happy/sad/fear/neutral
    "SEED_V":   ["emotion_5class"],            # +disgust
    "SEED_VII": ["emotion_7class", "emotion_continuous"],
    "SEED_GER": ["emotion_3class"],
    "SEED_FRA": ["emotion_3class"],
    # ── Vigilance decoding outcomes ─────────────────────────────────────
    "SEED_VIG": ["vigilance_continuous", "perclos"],
}

# Imaging feature templates — dynamically combined with AAL atlas regions
# {region} is replaced with actual neuroanatomy node names at generation time
IMAGING_FEATURE_TEMPLATES = {
    # sMRI FreeSurfer ROI features
    "cortical thickness of {region}":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "ADNI", "HCP_YA", "HCP_EP", "HCP_AGING"]},
    "gray matter volume of {region}":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "ADNI", "HCP_YA", "HCP_EP", "HCP_AGING"]},
    "subcortical volume of {region}":   {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "ADNI", "HCP_YA", "HCP_EP", "HCP_AGING"]},
    "cortical area of {region}":        {"modality": "sMRI", "tool": "FreeSurfer", "level": "ROI",
                                          "datasets": ["UKB", "HCP_YA", "HCP_AGING"]},
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
                                                        "datasets": ["UKB", "ADNI", "HCP_YA",
                                                                     "ABIDE", "ADHD200", "COBRE",
                                                                     "UCLA", "HCP_EP", "HCP_AGING"]},
    "effective connectivity from {a} to {b}":         {"modality": "fMRI", "tool": "DCM/GC",
                                                        "level": "connectivity",
                                                        "datasets": ["ADNI", "HCP_YA",
                                                                     "UCLA", "HCP_EP", "HCP_AGING"]},
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

# Brain decoding domain pairs (NSD / BOLD5000 / SEED family).
# These are SEPARATE from IMAGING_DOMAIN_PAIRS because decoding hypotheses
# reverse the usual direction: instead of "brain feature → clinical outcome",
# they go "stimulus ↔ brain" or "brain → psychological-state label".
DECODING_DOMAIN_PAIRS = [
    # Encoding: stimulus drives brain response
    ("visual_stimulus", "neuroanatomy"),
    ("visual_stimulus", "imaging_feature"),
    ("visual_stimulus", "connectivity"),
    # Decoding: brain predicts stimulus identity
    ("neuroanatomy",    "visual_stimulus"),
    ("imaging_feature", "visual_stimulus"),
    # EEG → emotion (SEED/SEED-IV/SEED-V/SEED-VII/SEED-GER/SEED-FRA)
    ("imaging_feature", "emotion"),
    ("neuroanatomy",    "emotion"),
    # EEG → vigilance (SEED-VIG)
    ("imaging_feature", "vigilance"),
    ("neuroanatomy",    "vigilance"),
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
        # P1: traversal walks the semantic layer only (no `about` provenance edges).
        # The full graph remains accessible via self.kg.G for claim lookup.
        self.G = kg.semantic_view if hasattr(kg, "semantic_view") else kg.G
        self._index = kg._index
        # Resolve blacklists to live KG ids: hardcoded ids may have been
        # remapped to CUI:Cxxx by UMLS canonicalization, so look up by name
        # when the original id is missing.
        self._path_ignore_ids = _resolve_blacklist(_PATH_IGNORE_SEED, self._index)
        self._intermediate_only_ignore_ids = _resolve_blacklist(
            _INTERMEDIATE_ONLY_SEED, self._index,
        )
        # Per-atom anchor-pool filters for batch_generate_for_chain. CS pre_hooks
        # install entries here to restrict which nodes can serve as a given
        # chain atom anchor (e.g. Case Study 2 forces IMAGING_MARKER to IM:*-prefix
        # atoms only, so the chain truly traverses the marker layer instead
        # of falling back to raw neuroanatomy CUIs that share the domain tag).
        # Filter signature: (node_id, ConceptNode) -> bool. None / missing
        # entry = no extra filter beyond ATOM_TO_DOMAINS membership.
        self._chain_atom_filters: dict = {}
        # Case-study hooks may pin exact claim endpoints to an atom even when
        # canonicalisation merged the endpoint into a node whose global domain
        # tags describe another sense of the same identifier.
        self._chain_atom_explicit_pools: dict = {}
        self._chain_claim_endpoint_roles: dict[tuple[str, str], set] = {}
        self._chain_claim_first_scope: str | None = None
        # Optional case-study hooks can widen a chain atom's domain pool
        # without changing the global atom algebra (e.g. Case Study 2 lets concrete
        # CLM_CONCEPT cognitive phenotypes serve as OUTCOME anchors).
        self._chain_atom_extra_domains: dict = {}
        # Optional per-atom seed/anchor rankers. Higher score is tried first.
        self._chain_atom_rankers: dict = {}
        # Case-study hooks may exclude graph concepts that cannot form an
        # executable mechanism when they occur inside an atom-to-atom segment.
        # Atom anchors themselves are unaffected.
        self._chain_forbidden_bridge_ids: set[str] = set()
        # Optional chain-level preference for paths that contain at least one
        # edge backed directly by a Phase-2 claim. This is deliberately a
        # preference, not a hard requirement, so sparse but biologically useful
        # bridge paths can still survive as a small fallback set.
        self._chain_prefer_claim_backed_paths: bool = False
        self._chain_claimless_path_fraction: float = 0.15
        self._chain_require_claim_backed_paths: bool = False
        self._chain_required_claim_scope: str | None = None
        # Task generators can be constrained to endpoints and at least one
        # direct claim edge from one audited Case Study scope.
        self._task_required_claim_scope: str | None = None
        self._claim_scope_endpoint_cache: dict[str, frozenset[str]] = {}
        self._claim_scope_endpoint_support_cache: dict[str, dict[str, int]] = {}
        self._semantic_scope_claim_cache: dict[str, list[dict]] = {}
        self._semantic_historical_pairs: set[tuple[str, str]] | None = None
        self.last_post_process_stats: dict[str, int] = {}
        # Build claims index for frequency_boost: (subj, pred, obj) → [claim_meta, ...]
        self._claims_by_triple: dict[tuple[str, str, str], list[dict]] = {}
        self._claims_by_endpoints: dict[
            tuple[str, str], list[tuple[str, dict]]
        ] = {}
        self._claim_endpoint_audits: dict[str, ClaimEndpointAudit] = {}
        claim_semantic_rejections: Counter[str] = Counter()
        for nid, node in self._index.items():
            if "claim" not in node.domain_tags:
                continue
            meta = node.metadata
            audit = audit_claim_endpoints(meta, self._index)
            self._claim_endpoint_audits[nid] = audit
            if not audit.valid:
                claim_semantic_rejections[audit.reason] += 1
                continue
            key = (meta.get("subject_id", ""), meta.get("predicate", ""), meta.get("object_id", ""))
            if key[0] and key[2]:
                self._claims_by_triple.setdefault(key, []).append(meta)
                self._claims_by_endpoints.setdefault((key[0], key[2]), []).append(
                    (nid, meta)
                )
        self._semantic_edge_rejections = 0
        if claim_semantic_rejections:
            logger.info(
                "claim endpoint semantic projection: %d canonical claim(s) accepted; "
                "%d collision(s) reserved for claim-local generation (%s)",
                len(self._claim_endpoint_audits) - sum(claim_semantic_rejections.values()),
                sum(claim_semantic_rejections.values()),
                dict(claim_semantic_rejections),
            )
        # Lazy evidence-degree cache for the min_evidence_per_node walk filter.
        self._non_tree_degree: Optional[dict[str, int]] = None
        self.feedback_state: FeedbackState | None = None
        self._dynamic_generation_enabled = False
        self._generation_exploration_round = 0
        self._generation_excluded_paths: frozenset[tuple[str, ...]] = frozenset()
        self._generation_path_templates: tuple[str, ...] = ("one_mediator",)
        self._generation_path_variants_per_endpoint = 1
        self._generation_feedback_mutation_fraction = 0.0
        self._generation_max_paths_per_endpoint = 2

    def load_feedback_state(self, path: str | Path) -> None:
        """Load supported/contradicted/execution-failed feedback for ranking."""
        self.feedback_state = FeedbackState.load(path)
        logger.info("loaded %d feedback record(s) from %s", len(self.feedback_state.records), path)

    def configure_dynamic_generation(
        self,
        *,
        enabled: bool,
        exploration_round: int = 0,
        excluded_paths: Iterable[tuple[str, ...]] = (),
        path_templates: tuple[str, ...] = ("one_mediator", "two_mediator"),
        path_variants_per_endpoint: int = 2,
        feedback_mutation_fraction: float = 0.35,
        max_paths_per_endpoint: int = 4,
    ) -> None:
        """Configure bounded candidate expansion for a dynamic discovery loop.

        Static generation keeps the historical one-mediator behaviour. Dynamic
        runs rotate the evidence-ranked seed core, skip paths proposed in prior
        rounds, admit audited two-mediator claim chains, and reserve part of the
        next proposal batch for one-factor variants of supported paths.
        """

        allowed_templates = {"one_mediator", "two_mediator"}
        unknown = set(path_templates) - allowed_templates
        if unknown:
            raise ValueError(f"unknown dynamic path template(s): {sorted(unknown)}")
        if exploration_round < 0:
            raise ValueError("exploration_round must be non-negative")
        if path_variants_per_endpoint < 1 or max_paths_per_endpoint < 1:
            raise ValueError("dynamic path limits must be positive")
        if not 0.0 <= feedback_mutation_fraction <= 1.0:
            raise ValueError("feedback_mutation_fraction must be in [0, 1]")

        self._dynamic_generation_enabled = bool(enabled)
        self._generation_exploration_round = int(exploration_round)
        self._generation_excluded_paths = frozenset(
            tuple(str(node_id) for node_id in path if str(node_id or ""))
            for path in excluded_paths
            if path
        )
        self._generation_path_templates = (
            tuple(path_templates) if enabled else ("one_mediator",)
        )
        self._generation_path_variants_per_endpoint = (
            int(path_variants_per_endpoint) if enabled else 1
        )
        self._generation_feedback_mutation_fraction = (
            float(feedback_mutation_fraction) if enabled else 0.0
        )
        self._generation_max_paths_per_endpoint = (
            int(max_paths_per_endpoint) if enabled else 2
        )

    @staticmethod
    def _hypothesis_path_nodes(hypothesis: Hypothesis) -> tuple[str, ...]:
        nodes = [str(hypothesis.source_id or "")]
        for link in hypothesis.path or []:
            if link.from_id and nodes[-1] != str(link.from_id):
                nodes.append(str(link.from_id))
            if link.to_id:
                nodes.append(str(link.to_id))
        if len(nodes) == 1 and hypothesis.target_id:
            nodes.append(str(hypothesis.target_id))
        return tuple(node_id for node_id in nodes if node_id)

    def _feedback_mutation_affinity(self, path_nodes: tuple[str, ...]) -> float:
        """Return 1 for a legal one-factor variant of a supported path."""

        if self.feedback_state is None or len(path_nodes) < 3:
            return 0.0
        affinity = 0.0
        for record in self.feedback_state.records:
            if record.status != SUPPORTED or len(record.path_node_ids) != len(path_nodes):
                continue
            differences = sum(
                left != right
                for left, right in zip(path_nodes, record.path_node_ids, strict=True)
            )
            if differences == 1:
                affinity = max(affinity, min(1.0, float(record.weight)))
        return affinity

    def _build_non_tree_degree(self) -> dict[str, int]:
        """Count incident non-tree edges per node.

        A "tree edge" is is_a / part_of / about — pure taxonomy or
        provenance. Nodes whose entire neighbourhood is tree-only are
        ontology leaves with no empirical anchor; routing a hypothesis
        through them produces graph paths that read like mechanism but
        are just "MeSH says X is_a Y is_a Z".

        Counts each undirected incidence once: for every edge u→v whose
        relation_type is NOT in TREE_RELATIONS, increment both u and v.
        Cached on first access; rebuilt only if the engine is re-init'd.
        """
        deg: dict[str, int] = {}
        for u, v, data in self.G.edges(data=True):
            if data.get("relation_type") in TREE_RELATIONS:
                continue
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        return deg

    def _node_non_tree_degree(self, nid: str) -> int:
        """Cached lookup. Lazily builds the per-node count on first call."""
        if self._non_tree_degree is None:
            self._non_tree_degree = self._build_non_tree_degree()
        return self._non_tree_degree.get(nid, 0)

    def _path_meets_evidence_floor(
        self, raw_path: list[str], min_evidence_per_node: int,
    ) -> bool:
        """All nodes in raw_path have non-tree degree >= min_evidence_per_node.

        Endpoints are checked too: a hypothesis whose source or target is
        an evidence-orphaned ontology node is just as uninformative as one
        with such a node in the middle.
        """
        if min_evidence_per_node <= 0:
            return True
        for nid in raw_path:
            if self._node_non_tree_degree(nid) < min_evidence_per_node:
                return False
        return True

    @staticmethod
    def _claim_meta_scopes(claim_meta: dict) -> set[str]:
        nested_meta = claim_meta.get("metadata") or {}
        scopes: set[str] = set()
        for holder in (claim_meta, nested_meta):
            values = holder.get("claim_case_study_ids") or []
            if isinstance(values, str):
                values = [values]
            scopes.update(str(value).strip() for value in values if value)
        return scopes

    def _claim_entries_for_edge(
        self,
        src_id: str,
        tgt_id: str,
        required_scope: str | None = None,
    ) -> list[tuple[str, dict]]:
        entries = self._claims_by_endpoints.get((src_id, tgt_id), [])
        if required_scope is None:
            return entries
        return [
            (claim_id, meta)
            for claim_id, meta in entries
            if required_scope in self._claim_meta_scopes(meta)
        ]

    def set_task_claim_scope(self, case_study_id: str | None) -> None:
        self._task_required_claim_scope = str(case_study_id or "").strip() or None

    @staticmethod
    def _semantic_claim_paper_key(claim_id: str, metadata: dict) -> str:
        paper = metadata.get("source_paper") or {}
        if not isinstance(paper, dict):
            paper = {}
        return str(
            paper.get("pmid")
            or paper.get("doi")
            or paper.get("title")
            or claim_id
        )

    def _semantic_claim_records_for_scope(self, scope: str) -> list[dict]:
        cached = self._semantic_scope_claim_cache.get(scope)
        if cached is not None:
            return cached

        build_all_pairs = self._semantic_historical_pairs is None
        all_pairs: set[tuple[str, str]] = set()
        records: list[dict] = []
        for claim_id, node in self._index.items():
            if "claim" not in (node.domain_tags or []):
                continue
            metadata = node.metadata or {}
            scopes = self._claim_meta_scopes(metadata)
            if not build_all_pairs and scope not in scopes:
                continue
            if metadata.get("negated"):
                continue
            projected = semantic_claim_pair(metadata, self._index)
            if projected is None:
                continue
            subject, obj = projected
            pair = tuple(sorted((subject.entity_id, obj.entity_id)))
            if build_all_pairs:
                all_pairs.add(pair)
            if scope not in scopes:
                continue
            if not case_study_pair_allowed(
                metadata,
                self._index,
                projected,
                scope,
                include_support=True,
            ):
                continue
            try:
                confidence = float(metadata.get("confidence") or 0.5)
            except (TypeError, ValueError):
                confidence = 0.5
            source_paper = metadata.get("source_paper") or {}
            if not isinstance(source_paper, dict):
                source_paper = {"reference": str(source_paper)}
            records.append({
                "claim_id": claim_id,
                "subject": subject,
                "object": obj,
                "predicate": str(metadata.get("predicate") or "is_associated_with"),
                "confidence": min(1.0, max(0.0, confidence)),
                "raw_text": str(metadata.get("raw_text") or ""),
                "evidence": metadata.get("evidence") or {},
                "source_paper": dict(source_paper),
                "paper_key": self._semantic_claim_paper_key(claim_id, metadata),
            })
        if build_all_pairs:
            self._semantic_historical_pairs = all_pairs
        self._semantic_scope_claim_cache[scope] = records
        return records

    @staticmethod
    def _semantic_bridge_jitter(seed: int | None, *values: str) -> float:
        payload = "\x1f".join([str(seed or 0), *values]).encode("utf-8")
        return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big") / float(
            2**64 - 1
        )

    @staticmethod
    def _semantic_bridge_link(
        record: dict,
        source,
        target,
    ) -> HypothesisLink:
        original_subject = record["subject"]
        original_object = record["object"]
        forward = (
            original_subject.entity_id == source.entity_id
            and original_object.entity_id == target.entity_id
        )
        evidence = record.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {"description": str(evidence)}
        return HypothesisLink(
            from_id=source.entity_id,
            from_name=source.name,
            to_id=target.entity_id,
            to_name=target.name,
            relation_type=(
                record["predicate"] if forward else "is_associated_with"
            ),
            confidence=record["confidence"],
            claim_id=record["claim_id"],
            raw_text=record["raw_text"],
            evidence=dict(evidence),
            source_paper=dict(record["source_paper"]),
        )

    def _semantic_endpoint_allowed_for_task(
        self,
        task,
        endpoint,
        *,
        source_side: bool,
    ) -> bool:
        input_atoms = {atom.value for atom in task.inputs}
        is_localization = (
            input_atoms == {"cognitive_task"}
            and task.output.value == "imaging_marker"
        )
        if not is_localization:
            return True
        if source_side:
            return (
                "cognitive_task" in endpoint.atoms
                and looks_like_cognitive_task_or_stimulus(endpoint.name)
                and not looks_like_imaging_measurement(endpoint.name)
            )
        node = self._index.get(endpoint.canonical_id)
        domains = set(node.domain_tags or []) if node is not None else set()
        return (
            "imaging_marker" in endpoint.atoms
            and (
                is_specific_functional_imaging_readout(endpoint.name)
                or "neuroanatomy" in domains
            )
        )

    def _semantic_mediator_allowed_for_task(self, task, endpoint) -> bool:
        input_atoms = {atom.value for atom in task.inputs}
        is_localization = (
            input_atoms == {"cognitive_task"}
            and task.output.value == "imaging_marker"
        )
        if not is_localization:
            return True
        atoms = set(endpoint.atoms)
        if atoms & {"disease", "outcome"}:
            return False
        if "imaging_marker" in atoms:
            return True
        node = self._index.get(endpoint.canonical_id)
        return node is not None and "neuroanatomy" in set(node.domain_tags or [])

    def _batch_generate_task_from_scoped_claims(
        self,
        task,
        *,
        max_hypotheses: int,
        random_seed: int | None,
    ) -> list[Hypothesis]:
        """Generate conservative cross-paper semantic bridges.

        Static runs retain the historical source-mediator-target template.
        Dynamic runs may additionally use one extra claim-backed mediator and
        multiple path instantiations per endpoint pair. Every template remains
        continuous, task-contract compatible, cross-paper, and absent as a
        direct endpoint pair from the historical graph.
        """

        scope = self._task_required_claim_scope
        if not scope or max_hypotheses <= 0:
            return []
        records = self._semantic_claim_records_for_scope(scope)
        if not records:
            return []

        input_atoms = {atom.value for atom in task.inputs}
        output_atom = task.output.value
        left_by_mediator: dict[str, list[tuple[dict, object, object]]] = defaultdict(list)
        right_by_mediator: dict[str, list[tuple[dict, object, object]]] = defaultdict(list)

        for record in records:
            subject = record["subject"]
            obj = record["object"]
            for endpoint, other in ((subject, obj), (obj, subject)):
                endpoint_atoms = set(endpoint.atoms)
                if (
                    endpoint_atoms & input_atoms
                    and other.atoms
                    and self._semantic_endpoint_allowed_for_task(
                        task,
                        endpoint,
                        source_side=True,
                    )
                ):
                    left_by_mediator[other.entity_id].append(
                        (record, endpoint, other)
                    )
                if (
                    output_atom in endpoint_atoms
                    and other.atoms
                    and self._semantic_endpoint_allowed_for_task(
                        task,
                        endpoint,
                        source_side=False,
                    )
                ):
                    right_by_mediator[other.entity_id].append(
                        (record, other, endpoint)
                    )

        historical_pairs = self._semantic_historical_pairs or set()
        templates = set(self._generation_path_templates)
        variants_per_endpoint = self._generation_path_variants_per_endpoint
        candidate_groups: dict[
            tuple[tuple[str, str], str], list[dict[str, object]]
        ] = defaultdict(list)

        def retain_candidate(row: dict[str, object]) -> None:
            path_nodes = tuple(str(value) for value in row["path_nodes"])
            if path_nodes in self._generation_excluded_paths:
                return
            row["feedback_mutation_affinity"] = self._feedback_mutation_affinity(
                path_nodes
            )
            row["score"] = float(row["score"]) + 0.16 * float(
                row["feedback_mutation_affinity"]
            )
            endpoint_pair = tuple(sorted((path_nodes[0], path_nodes[-1])))
            group_key = (endpoint_pair, str(row["path_template"]))
            bucket = candidate_groups[group_key]
            if any(tuple(existing["path_nodes"]) == path_nodes for existing in bucket):
                return
            bucket.append(row)
            bucket.sort(
                key=lambda candidate: (
                    -float(candidate["score"]),
                    tuple(candidate["path_nodes"]),
                )
            )
            del bucket[variants_per_endpoint:]

        shared_mediators = sorted(set(left_by_mediator) & set(right_by_mediator))
        one_mediator_side_cap = 16 if self._dynamic_generation_enabled else 64
        if "one_mediator" in templates:
            mediator_iterable = shared_mediators
        else:
            mediator_iterable = []
        for mediator_id in mediator_iterable:
            left_rows = sorted(
                left_by_mediator[mediator_id],
                key=lambda row: (
                    -row[0]["confidence"],
                    row[0]["claim_id"],
                    row[1].entity_id,
                ),
            )[:one_mediator_side_cap]
            right_rows = sorted(
                right_by_mediator[mediator_id],
                key=lambda row: (
                    -row[0]["confidence"],
                    row[0]["claim_id"],
                    row[2].entity_id,
                ),
            )[:one_mediator_side_cap]
            mediator_degree = len(left_rows) + len(right_rows)
            support = min(1.0, math.log1p(mediator_degree) / math.log(17.0))
            specificity = 1.0 / (1.0 + math.log1p(max(0, mediator_degree - 1)))
            for left_record, source, mediator in left_rows:
                if not self._semantic_mediator_allowed_for_task(task, mediator):
                    continue
                for right_record, _right_mediator, target in right_rows:
                    if left_record["claim_id"] == right_record["claim_id"]:
                        continue
                    if left_record["paper_key"] == right_record["paper_key"]:
                        continue
                    if source.entity_id == target.entity_id:
                        continue
                    endpoint_pair = tuple(sorted((source.entity_id, target.entity_id)))
                    if endpoint_pair in historical_pairs:
                        continue
                    evidence = (
                        left_record["confidence"] + right_record["confidence"]
                    ) / 2.0
                    directional = (
                        int(left_record["predicate"] in DIRECTIONAL_RELATIONS)
                        + int(right_record["predicate"] in DIRECTIONAL_RELATIONS)
                    ) / 2.0
                    jitter = self._semantic_bridge_jitter(
                        random_seed,
                        scope,
                        source.entity_id,
                        mediator_id,
                        target.entity_id,
                    )
                    score = (
                        0.50 * evidence
                        + 0.18 * support
                        + 0.12 * specificity
                        + 0.12 * directional
                        + 0.08 * jitter
                    )
                    retain_candidate({
                        "score": score,
                        "path_template": "one_mediator",
                        "path_nodes": (
                            source.entity_id,
                            mediator_id,
                            target.entity_id,
                        ),
                        "endpoints": (source, mediator, target),
                        "records": (left_record, right_record),
                    })

        if "two_mediator" in templates:
            bridge_rows: list[tuple[float, dict, object, object]] = []
            for bridge_record in records:
                subject = bridge_record["subject"]
                obj = bridge_record["object"]
                for first_mediator, second_mediator in ((subject, obj), (obj, subject)):
                    if (
                        first_mediator.entity_id == second_mediator.entity_id
                        or first_mediator.entity_id not in left_by_mediator
                        or second_mediator.entity_id not in right_by_mediator
                        or not self._semantic_mediator_allowed_for_task(
                            task, first_mediator
                        )
                        or not self._semantic_mediator_allowed_for_task(
                            task, second_mediator
                        )
                    ):
                        continue
                    bridge_quality = (
                        float(bridge_record["confidence"])
                        + 0.05
                        * self._semantic_bridge_jitter(
                            random_seed,
                            scope,
                            first_mediator.entity_id,
                            second_mediator.entity_id,
                        )
                    )
                    bridge_rows.append(
                        (
                            bridge_quality,
                            bridge_record,
                            first_mediator,
                            second_mediator,
                        )
                    )

            bridge_cap = max(200, min(3000, max_hypotheses * 2))
            bridge_rows.sort(
                key=lambda row: (
                    -row[0],
                    row[1]["claim_id"],
                    row[2].entity_id,
                    row[3].entity_id,
                )
            )
            for _, bridge_record, first_mediator, second_mediator in bridge_rows[:bridge_cap]:
                left_rows = sorted(
                    left_by_mediator[first_mediator.entity_id],
                    key=lambda row: (
                        -row[0]["confidence"],
                        row[0]["claim_id"],
                        row[1].entity_id,
                    ),
                )[:3]
                right_rows = sorted(
                    right_by_mediator[second_mediator.entity_id],
                    key=lambda row: (
                        -row[0]["confidence"],
                        row[0]["claim_id"],
                        row[2].entity_id,
                    ),
                )[:3]
                mediator_degree = len(left_rows) + len(right_rows)
                support = min(1.0, math.log1p(mediator_degree) / math.log(13.0))
                for left_record, source, _ in left_rows:
                    for right_record, _, target in right_rows:
                        claim_ids = {
                            left_record["claim_id"],
                            bridge_record["claim_id"],
                            right_record["claim_id"],
                        }
                        path_nodes = (
                            source.entity_id,
                            first_mediator.entity_id,
                            second_mediator.entity_id,
                            target.entity_id,
                        )
                        if len(claim_ids) < 3 or len(set(path_nodes)) < 4:
                            continue
                        paper_keys = {
                            left_record["paper_key"],
                            bridge_record["paper_key"],
                            right_record["paper_key"],
                        }
                        if len(paper_keys) < 2 or source.entity_id == target.entity_id:
                            continue
                        endpoint_pair = tuple(
                            sorted((source.entity_id, target.entity_id))
                        )
                        if endpoint_pair in historical_pairs:
                            continue
                        evidence = (
                            left_record["confidence"]
                            + bridge_record["confidence"]
                            + right_record["confidence"]
                        ) / 3.0
                        directional = sum(
                            record["predicate"] in DIRECTIONAL_RELATIONS
                            for record in (left_record, bridge_record, right_record)
                        ) / 3.0
                        jitter = self._semantic_bridge_jitter(
                            random_seed,
                            scope,
                            *path_nodes,
                        )
                        score = (
                            0.52 * evidence
                            + 0.16 * support
                            + 0.12 * directional
                            + 0.08 * jitter
                            + 0.12 * (len(paper_keys) / 3.0)
                        )
                        retain_candidate({
                            "score": score,
                            "path_template": "two_mediator",
                            "path_nodes": path_nodes,
                            "endpoints": (
                                source,
                                first_mediator,
                                second_mediator,
                                target,
                            ),
                            "records": (
                                left_record,
                                bridge_record,
                                right_record,
                            ),
                        })

        ranked = sorted(
            (
                candidate
                for bucket in candidate_groups.values()
                for candidate in bucket
            ),
            key=lambda row: (
                -float(row["score"]),
                tuple(row["path_nodes"]),
            ),
        )
        selected: list[dict[str, object]] = []
        seen: set[tuple[str, ...]] = set()
        source_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        mediator_counts: Counter[str] = Counter()

        quota_stages = (
            (4, 12, 24),
            (12, 32, 64),
            (10**9, 10**9, 10**9),
        )

        def select_from(pool: list[dict[str, object]], desired_total: int) -> None:
            for source_limit, target_limit, mediator_limit in quota_stages:
                for row in pool:
                    path_nodes = tuple(str(value) for value in row["path_nodes"])
                    if path_nodes in seen:
                        continue
                    mediator_ids = path_nodes[1:-1]
                    if (
                        source_counts[path_nodes[0]] >= source_limit
                        or target_counts[path_nodes[-1]] >= target_limit
                        or any(
                            mediator_counts[mediator_id] >= mediator_limit
                            for mediator_id in mediator_ids
                        )
                    ):
                        continue
                    seen.add(path_nodes)
                    source_counts[path_nodes[0]] += 1
                    target_counts[path_nodes[-1]] += 1
                    for mediator_id in mediator_ids:
                        mediator_counts[mediator_id] += 1
                    selected.append(row)
                    if len(selected) >= desired_total:
                        return

        mutation_target = min(
            max_hypotheses,
            int(math.ceil(max_hypotheses * self._generation_feedback_mutation_fraction)),
        )
        if mutation_target:
            mutation_pool = [
                row
                for row in ranked
                if float(row["feedback_mutation_affinity"]) > 0.0
            ]
            select_from(mutation_pool, mutation_target)
        select_from(ranked, max_hypotheses)

        generated: list[Hypothesis] = []
        for row in selected:
            endpoints = tuple(row["endpoints"])
            records_used = tuple(row["records"])
            path_nodes = tuple(str(value) for value in row["path_nodes"])
            source = endpoints[0]
            target = endpoints[-1]
            mediators = endpoints[1:-1]
            links = [
                self._semantic_bridge_link(record, left, right)
                for record, left, right in zip(
                    records_used,
                    endpoints[:-1],
                    endpoints[1:],
                    strict=True,
                )
            ]
            digest = hashlib.sha1(
                "\x1f".join([scope, *path_nodes]).encode("utf-8")
            ).hexdigest()[:16]
            paper_keys = list(
                dict.fromkeys(record["paper_key"] for record in records_used)
            )
            hypothesis = Hypothesis(
                id=f"HYP:SEMANTIC_BRIDGE:{digest}",
                hypothesis_type="claim_bridge",
                source_id=source.entity_id,
                source_name=source.name,
                target_id=target.entity_id,
                target_name=target.name,
                path=links,
                confidence_score=self._compute_confidence_score(links),
                novelty_score=1.0,
                evidence_score=self._compute_evidence_score(links),
                testability_score=0.6,
                composite_score=float(row["score"]),
                supporting_claims=list(
                    dict.fromkeys(record["claim_id"] for record in records_used)
                ),
                testability_reason=(
                    f"{len(links)} source-linked claims from independent papers "
                    "form a continuous, task-compatible semantic bridge."
                ),
                metadata={
                    "generation_mode": "scoped_semantic_claim_bridge",
                    "semantic_projection": "claim_endpoint_semantics.v1",
                    "claim_case_study_id": scope,
                    "path_template": str(row["path_template"]),
                    "path_node_ids": list(path_nodes),
                    "feedback_mutation": bool(
                        float(row["feedback_mutation_affinity"]) > 0.0
                    ),
                    "feedback_mutation_affinity": float(
                        row["feedback_mutation_affinity"]
                    ),
                    "feedback_mutation_score_applied": True,
                    "source_atoms": list(source.atoms),
                    "target_atoms": list(target.atoms),
                    "mediator_ids": [mediator.entity_id for mediator in mediators],
                    "mediator_names": [mediator.name for mediator in mediators],
                    "mediator_atoms": [list(mediator.atoms) for mediator in mediators],
                    "cross_paper": True,
                    "source_paper_keys": paper_keys,
                },
            )
            hypothesis.explanation = self._generate_explanation(hypothesis)
            generated.append(hypothesis)
        return generated

    def _batch_generate_multi_input_task_from_scoped_claims(
        self,
        task,
        *,
        max_hypotheses: int,
        random_seed: int | None,
    ) -> list[Hypothesis]:
        """Assemble compact connected evidence graphs covering every task atom."""

        scope = self._task_required_claim_scope
        if not scope or len(task.inputs) < 2 or max_hypotheses <= 0:
            return []
        records = self._semantic_claim_records_for_scope(scope)
        if not records:
            return []

        input_order = tuple(sorted(task.inputs, key=lambda atom: atom.value))
        edges: list[tuple[dict, object, object]] = []
        adjacency: dict[str, list[int]] = defaultdict(list)
        endpoints: dict[str, object] = {}
        for record in records:
            subject = record["subject"]
            obj = record["object"]
            if subject.entity_id == obj.entity_id:
                continue
            edge_index = len(edges)
            edges.append((record, subject, obj))
            adjacency[subject.entity_id].append(edge_index)
            adjacency[obj.entity_id].append(edge_index)
            endpoints[subject.entity_id] = subject
            endpoints[obj.entity_id] = obj

        output_ids = [
            endpoint_id
            for endpoint_id, endpoint in endpoints.items()
            if endpoint_matches_atom(endpoint, task.output, self._index)
            and self._semantic_endpoint_allowed_for_task(
                task,
                endpoint,
                source_side=False,
            )
        ]
        output_ids.sort(key=lambda endpoint_id: (-len(adjacency[endpoint_id]), endpoint_id))
        # Most scoped evidence components are tiny. Degree-only truncation can
        # miss the few complete, specific mechanisms in the long tail, so scan
        # every output for ordinary scopes and retain a high safety cap only for
        # unusually broad corpora.
        max_outputs = max(250, min(5000, max_hypotheses * 25))
        max_edges = len(input_order) + 1
        beam_width = max(48, min(160, max_hypotheses * 2))
        candidates: dict[tuple[tuple[str, ...], str], tuple] = {}
        for output_id in output_ids[:max_outputs]:
            output = endpoints[output_id]
            frontier = [(frozenset({output_id}), tuple(), tuple())]
            for depth in range(1, max_edges + 1):
                next_states: dict[tuple, tuple] = {}
                for node_ids, edge_ids, bindings in frontier:
                    used_edges = set(edge_ids)
                    incident = {
                        edge_index
                        for node_id in node_ids
                        for edge_index in adjacency.get(node_id, ())
                        if edge_index not in used_edges
                    }
                    incident_rows = sorted(
                        incident,
                        key=lambda edge_index: (
                            -edges[edge_index][0]["confidence"],
                            edges[edge_index][0]["claim_id"],
                            edge_index,
                        ),
                    )[:48]
                    for edge_index in incident_rows:
                        record, subject, obj = edges[edge_index]
                        if subject.entity_id in node_ids and obj.entity_id not in node_ids:
                            new_endpoint = obj
                        elif obj.entity_id in node_ids and subject.entity_id not in node_ids:
                            new_endpoint = subject
                        else:
                            continue
                        binding_map = {Atom(atom_name): entity_id for atom_name, entity_id in bindings}
                        unmatched = [
                            atom
                            for atom in input_order
                            if atom not in binding_map
                            and endpoint_matches_atom(new_endpoint, atom, self._index)
                        ]
                        binding_options: list[Atom | None] = [None, *unmatched]
                        for matched_atom in binding_options:
                            updated = dict(binding_map)
                            if matched_atom is not None:
                                if new_endpoint.entity_id in updated.values():
                                    continue
                                updated[matched_atom] = new_endpoint.entity_id
                            updated_bindings = tuple(
                                (atom.value, updated[atom])
                                for atom in input_order
                                if atom in updated
                            )
                            updated_edges = tuple(sorted((*edge_ids, edge_index)))
                            updated_nodes = frozenset((*node_ids, new_endpoint.entity_id))
                            records_used = [edges[index][0] for index in updated_edges]
                            paper_keys = [record_used["paper_key"] for record_used in records_used]
                            if len(updated) == len(input_order):
                                input_ids = [updated[atom] for atom in input_order]
                                if len(set(input_ids)) != len(input_ids) or len(set(paper_keys)) < 2:
                                    continue
                                inputs = tuple(endpoints[entity_id] for entity_id in input_ids)
                                if not case_study_endpoint_names_allowed(scope, inputs[0], output):
                                    continue
                                evidence = sum(
                                    record_used["confidence"] for record_used in records_used
                                ) / len(records_used)
                                paper_diversity = min(
                                    1.0,
                                    len(set(paper_keys)) / max(2, len(records_used)),
                                )
                                compactness = min(1.0, len(input_order) / len(records_used))
                                jitter = self._semantic_bridge_jitter(
                                    random_seed,
                                    scope,
                                    *sorted(input_ids),
                                    output.entity_id,
                                )
                                score = (
                                    0.58 * evidence
                                    + 0.22 * paper_diversity
                                    + 0.12 * compactness
                                    + 0.08 * jitter
                                )
                                key = (tuple(sorted(input_ids)), output.entity_id)
                                edge_rows = tuple(edges[index] for index in updated_edges)
                                row = (
                                    score,
                                    output,
                                    inputs,
                                    edge_rows,
                                    tuple(paper_keys),
                                )
                                current = candidates.get(key)
                                if current is None or score > current[0]:
                                    candidates[key] = row
                                continue
                            if depth >= max_edges:
                                continue
                            evidence = sum(
                                record_used["confidence"] for record_used in records_used
                            ) / len(records_used)
                            state_quality = (
                                len(updated),
                                len(set(paper_keys)),
                                evidence,
                                -len(updated_edges),
                            )
                            signature = (
                                tuple(sorted(updated_nodes)),
                                updated_edges,
                                updated_bindings,
                            )
                            current = next_states.get(signature)
                            if current is None or state_quality > current[0]:
                                next_states[signature] = (
                                    state_quality,
                                    (updated_nodes, updated_edges, updated_bindings),
                                )
                frontier = [
                    value[1]
                    for _, value in sorted(
                        next_states.items(),
                        key=lambda item: (item[1][0], item[0]),
                        reverse=True,
                    )[:beam_width]
                ]
                if not frontier:
                    break

        logger.info(
            "task '%s': scoped connected-graph search used %d claims, %d outputs, "
            "and found %d complete candidate(s)",
            task.name,
            len(edges),
            len(output_ids),
            len(candidates),
        )

        ranked = sorted(
            candidates.values(),
            key=lambda row: (
                -row[0],
                row[1].entity_id,
                tuple(endpoint.entity_id for endpoint in row[2]),
            ),
        )
        selected: list[tuple] = []
        output_counts: Counter[str] = Counter()
        for output_limit in (4, 12, 10**9):
            selected_keys = {
                (
                    tuple(endpoint.entity_id for endpoint in row[2]),
                    row[1].entity_id,
                )
                for row in selected
            }
            for row in ranked:
                key = (
                    tuple(endpoint.entity_id for endpoint in row[2]),
                    row[1].entity_id,
                )
                if key in selected_keys or output_counts[row[1].entity_id] >= output_limit:
                    continue
                selected.append(row)
                selected_keys.add(key)
                output_counts[row[1].entity_id] += 1
                if len(selected) >= max_hypotheses:
                    break
            if len(selected) >= max_hypotheses:
                break

        generated: list[Hypothesis] = []
        for score, output, inputs, edge_rows, paper_keys in selected:
            links = [
                self._semantic_bridge_link(record, subject, obj)
                for record, subject, obj in edge_rows
            ]
            digest = hashlib.sha1(
                "\x1f".join(
                    [
                        scope,
                        *sorted(endpoint.entity_id for endpoint in inputs),
                        output.entity_id,
                    ]
                ).encode("utf-8")
            ).hexdigest()[:16]
            first_input = inputs[0]
            hypothesis = Hypothesis(
                id=f"HYP:MULTI_INPUT:{digest}",
                hypothesis_type="multi_input_evidence_graph",
                source_id=first_input.entity_id,
                source_name=first_input.name,
                target_id=output.entity_id,
                target_name=output.name,
                path=links,
                confidence_score=self._compute_confidence_score(links),
                novelty_score=1.0,
                evidence_score=self._compute_evidence_score(links),
                testability_score=0.7,
                composite_score=score,
                supporting_claims=list(dict.fromkeys(link.claim_id for link in links)),
                testability_reason=(
                    "Independent source-linked claims cover every registered task input "
                    "and converge on one measurable output."
                ),
                metadata={
                    "generation_mode": "scoped_multi_input_evidence_graph",
                    "semantic_projection": "claim_endpoint_semantics.v1",
                    "claim_case_study_id": scope,
                    "task_name": task.name,
                    "task_signature": task.signature,
                    "task_modifier": task.modifier.value,
                    "task_kind": "task",
                    "source_atoms": list(first_input.atoms),
                    "target_atoms": list(output.atoms),
                    "input_atom_order": [atom.value for atom in input_order],
                    "input_entity_ids": [endpoint.entity_id for endpoint in inputs],
                    "input_entity_names": [endpoint.name for endpoint in inputs],
                    "input_entity_atoms": [[atom.value] for atom in input_order],
                    "cross_paper": True,
                    "source_paper_keys": list(dict.fromkeys(paper_keys)),
                },
            )
            input_text = "; ".join(
                f"{atom.value}: {endpoint.name}"
                for atom, endpoint in zip(input_order, inputs)
            )
            hypothesis.explanation = (
                f"Joint hypothesis: {input_text} collectively predict {output.name}.\n"
                f"Evidence graph: {len(links)} source-linked claims from "
                f"{len(set(paper_keys))} independent papers."
            )
            generated.append(hypothesis)
        return generated

    def _claim_endpoint_ids_for_scope(self, required_scope: str) -> frozenset[str]:
        cached = self._claim_scope_endpoint_cache.get(required_scope)
        if cached is not None:
            return cached
        endpoint_ids: set[str] = set()
        for (source_id, target_id), entries in self._claims_by_endpoints.items():
            if any(
                required_scope in self._claim_meta_scopes(meta)
                for _claim_id, meta in entries
            ):
                endpoint_ids.update((source_id, target_id))
        result = frozenset(endpoint_ids)
        self._claim_scope_endpoint_cache[required_scope] = result
        return result

    def _claim_endpoint_support_for_scope(self, required_scope: str) -> dict[str, int]:
        cached = self._claim_scope_endpoint_support_cache.get(required_scope)
        if cached is not None:
            return cached
        support: dict[str, int] = {}
        for (source_id, target_id), entries in self._claims_by_endpoints.items():
            count = sum(
                required_scope in self._claim_meta_scopes(meta)
                for _claim_id, meta in entries
            )
            if count:
                support[source_id] = support.get(source_id, 0) + count
                support[target_id] = support.get(target_id, 0) + count
        self._claim_scope_endpoint_support_cache[required_scope] = support
        return support

    def _path_claim_edge_count(
        self,
        raw_path: list[str],
        required_scope: str | None = None,
    ) -> int:
        """Count path edges backed by direct claims in the requested scope.

        The graph is a ``DiGraph``, so multiple claims sharing an endpoint pair
        collapse to one displayed edge. The claim-node endpoint index preserves
        every source-linked claim and is therefore authoritative for scoping.
        """
        n = 0
        for i in range(len(raw_path) - 1):
            src_id, tgt_id = raw_path[i], raw_path[i + 1]
            if not self.G.has_edge(src_id, tgt_id):
                continue
            if required_scope is None:
                required_scope = self._chain_required_claim_scope
            if self._claim_entries_for_edge(src_id, tgt_id, required_scope):
                n += 1
                continue
            edge_data = self.G.edges[src_id, tgt_id]
            claim_id = edge_data.get("metadata", {}).get("claim_id")
            if not claim_id:
                continue
            if required_scope is not None:
                claim_node = self._index.get(claim_id)
                claim_meta = claim_node.metadata if claim_node is not None else {}
                if required_scope not in self._claim_meta_scopes(claim_meta):
                    continue
            n += 1
        return n

    def _sort_chain_candidates(
        self,
        candidates: list[tuple[list[str], list[int]]],
        next_atom=None,
    ) -> list[tuple[list[str], list[int]]]:
        """Order chain candidates, optionally preferring claim-backed paths."""
        anchor_ranker = self._chain_atom_rankers.get(next_atom)
        if (
            not self._chain_prefer_claim_backed_paths
            and not self._chain_require_claim_backed_paths
            and anchor_ranker is None
        ):
            return candidates

        def _sort_key(item):
            path, _anchors = item
            claim_edges = self._path_claim_edge_count(path)
            anchor_score = (
                anchor_ranker(path[-1], self._index.get(path[-1]))
                if anchor_ranker is not None
                else 0
            )
            return (
                claim_edges > 0,
                anchor_score,
                claim_edges,
                len(path),
            )

        return sorted(
            candidates,
            key=_sort_key,
            reverse=True,
        )

    def _select_chain_survivors(
        self,
        candidates: list[tuple[list[str], list[int]]],
        max_chains: int,
    ) -> list[tuple[list[str], list[int]]]:
        """Keep mostly claim-backed chain paths with a small claimless fallback."""
        if not self._chain_prefer_claim_backed_paths and not self._chain_require_claim_backed_paths:
            return candidates[:max_chains]

        ordered = self._sort_chain_candidates(candidates)
        claim_backed = [
            item for item in ordered
            if self._path_claim_edge_count(item[0]) > 0
        ]
        claimless = [
            item for item in ordered
            if self._path_claim_edge_count(item[0]) == 0
        ]
        if self._chain_require_claim_backed_paths:
            return claim_backed[:max_chains]
        if not claim_backed:
            return ordered[:max_chains]

        fallback_quota = 0
        if claimless:
            fallback_quota = max(1, int(max_chains * self._chain_claimless_path_fraction))
            fallback_quota = min(fallback_quota, len(claimless))
        primary_quota = max_chains - fallback_quota
        survivors = claim_backed[:primary_quota]
        if len(survivors) < primary_quota:
            fallback_quota += primary_quota - len(survivors)
        survivors.extend(claimless[:fallback_quota])
        if len(survivors) < max_chains:
            used = {tuple(path) for path, _anchors in survivors}
            for item in ordered:
                if tuple(item[0]) in used:
                    continue
                survivors.append(item)
                if len(survivors) >= max_chains:
                    break
        return survivors[:max_chains]

    # ── batch generation ───────────────────────────────────────────────

    def _path_intermediates_are_atoms(
        self,
        raw_path: list[str],
        allowed_domains: Optional[frozenset[str]] = None,
    ) -> bool:
        """All non-endpoint nodes in `raw_path` carry ≥1 allowed-atom domain.

        Endpoints (raw_path[0], raw_path[-1]) are skipped — task-driven
        callers already constrain those by the input/output atom. The check
        targets the bridge nodes (raw_path[1:-1]) and rejects paths that
        transit infrastructure (atlas/modality/dataset/ml_model) or meta
        (claim/recipe) nodes. Empty intermediate set passes trivially.

        ``allowed_domains`` defaults to the union of all atom domains; a
        stricter caller (e.g. requiring intermediates be IM-only) can pass
        a narrower set.
        """
        if len(raw_path) <= 2:
            return True
        if allowed_domains is None:
            allowed_domains = _allowed_atom_domains()
        for nid in raw_path[1:-1]:
            node = self._index.get(nid)
            if node is None:
                return False
            node_doms = set(node.domain_tags or [])
            if not (node_doms & allowed_domains):
                return False
        return True

    def _path_distinct_atom_domains(self, raw_path: list[str]) -> int:
        """Count distinct atom domains touched by nodes along raw_path.

        Used by the metapath bag-constraint to require that a candidate
        hypothesis path crosses several semantic domains rather than
        loitering inside a single one (e.g. two neuroanatomy hops).

        Only domains in ``_allowed_atom_domains()`` are counted -- pure
        infrastructure / claim tags are ignored. Returns 0 if no node on
        the path carries any atom domain (should not happen after the
        intermediates-are-atoms filter, but kept defensive).
        """
        allowed = _allowed_atom_domains()
        seen: set[str] = set()
        for nid in raw_path:
            node = self._index.get(nid)
            if node is None:
                continue
            for d in (node.domain_tags or []):
                if d in allowed:
                    seen.add(d)
        return len(seen)

    def batch_generate(
        self,
        domain_pairs: Optional[list[tuple[str, str]]] = None,
        max_hops: int = 4,
        max_paths_per_pair: int = 5,
        max_seeds_per_domain: int = 50,
        output_atom=None,
        skip_post_process: bool = False,
        min_hops: int = 2,
        metapath_min_domains: int = 2,
        prefer_longer_paths: bool = True,
        min_evidence_per_node: int = 1,
        random_seed: int | None = None,
        seed_diversity_fraction: float = 0.0,
    ) -> list[Hypothesis]:
        """Batch-generate hypotheses across the entire graph.

        Strategy: for each domain pair, sample seed concepts from domain_a,
        find paths to concepts in domain_b within max_hops hops.

        If ``output_atom`` (a :class:`neurooracle.src.atoms.Atom`) is supplied,
        the post-processing target-domain check accepts that atom's domain
        pool as valid outcomes -- needed for tasks like personalised_treatment
        whose target atom (DRUG) is otherwise excluded from the default
        outcome set.

        Args (3-hop refactor stage 1):
            min_hops: minimum edge count per kept path. Default 2 keeps the
                "no direct edges" guarantee (raw_path length 3 = 2 edges).
                Raise to 3 to force 3-hop-or-longer hypotheses; useful when
                the seed pool contains many atom anchors that need to chain
                through a mediator before reaching a clinical/IM endpoint.
            metapath_min_domains: minimum number of *distinct* atom domains
                that a path must touch (counting source / intermediates /
                target). Default 2 enforces that source and target sit in
                different domains (which the domain_pair gate already does
                in most cases). Set to 3 to require an explicit cross-domain
                mediator between source and target.
            prefer_longer_paths: when more candidate paths exist than the
                per-pair quota, sort longer paths first so the limited slots
                go to richer chains rather than 2-hop shortcuts.
            min_evidence_per_node: minimum non-tree (i.e. not is_a /
                part_of / about) edge count required at every node on the
                path. Default 1 drops nodes that are pure ontology leaves
                with no empirical anchor. Raise to 2+ to demand multiple
                 independent evidence edges per visited node.
            random_seed: optional stable seed for a diverse tail of source
                anchors. ``None`` preserves the historical deterministic order.
            seed_diversity_fraction: fraction of each source-anchor quota drawn
                from an evidence-weighted stochastic tail. The highest-support
                anchors always occupy the remaining fraction.
        """
        if domain_pairs is None:
            domain_pairs = DEFAULT_DOMAIN_PAIRS

        required_scope = self._task_required_claim_scope
        scoped_endpoint_ids = (
            self._claim_endpoint_ids_for_scope(required_scope)
            if required_scope is not None
            else None
        )
        if required_scope is not None and not scoped_endpoint_ids:
            logger.warning("task claim scope '%s' has no claim endpoints", required_scope)
            return []

        all_hypotheses: list[Hypothesis] = []
        seen_pairs: set[tuple[str, str]] = set()
        _hyp_counter = 0

        for dom_a, dom_b in domain_pairs:
            logger.info(f"generating hypotheses: {dom_a} -> {dom_b}")

            # Scope support is a ranking prior inside _sample_domain_nodes.
            # Keep the endpoint pool graph-wide so task evidence can bridge
            # into previously unlabelled but atom-compatible concepts.
            seeds_a = self._sample_domain_nodes(
                dom_a,
                max_seeds_per_domain,
                random_seed=random_seed,
                diversity_fraction=seed_diversity_fraction,
            )
            if min_evidence_per_node > 0:
                seeds_a = [
                    s for s in seeds_a
                    if self._node_non_tree_degree(s) >= min_evidence_per_node
                ]
            targets_b = {
                nid for nid, data in self.G.nodes(data=True)
                if dom_b in data.get("domain_tags", [])
                and "claim" not in data.get("domain_tags", [])
                and nid not in self._path_ignore_ids
                and (min_evidence_per_node <= 0
                     or self._node_non_tree_degree(nid) >= min_evidence_per_node)
            }

            for seed_id in seeds_a:
                if seed_id not in self.G:
                    continue

                # BFS reach-set: cheap pre-filter to know which targets are
                # reachable within max_hops before the more expensive
                # all_simple_paths enumeration runs per (seed, target).
                try:
                    reachable = nx.single_source_shortest_path(
                        self.G, seed_id, cutoff=max_hops
                    )
                except nx.NetworkXError:
                    continue

                candidates = [
                    nid for nid in reachable
                    if nid in targets_b and nid != seed_id
                ]
                # ``reachable`` preserves BFS order, so this keeps the closest
                # task-valid targets while bounding scans over very broad
                # outcome domains (notably multi-input drug-response tasks).
                candidate_cap = 64
                candidates = candidates[:candidate_cap]

                pair_count = 0
                for target_id in candidates:
                    pair_key = tuple(sorted([seed_id, target_id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # Enumerate up to N simple paths between this (seed, target).
                    # Bound both accepted and inspected paths: dense pairs can
                    # yield many paths that fail the atom/evidence filters before
                    # one accepted path is found.
                    raw_paths: list[list[str]] = []
                    enum_cap = max(max_paths_per_pair * 4, 8)
                    inspect_cap = 512
                    inspected = 0
                    try:
                        for p in nx.all_simple_paths(
                            self.G, seed_id, target_id, cutoff=max_hops
                        ):
                            inspected += 1
                            if inspected > inspect_cap:
                                break
                            if len(p) - 1 < min_hops:
                                continue
                            if not self._path_intermediates_are_atoms(p):
                                continue
                            if (metapath_min_domains > 1 and
                                self._path_distinct_atom_domains(p) < metapath_min_domains):
                                continue
                            if not self._path_meets_evidence_floor(p, min_evidence_per_node):
                                continue
                            if (
                                required_scope is not None
                                and self._path_claim_edge_count(p, required_scope) < 1
                            ):
                                continue
                            if tuple(p) in self._generation_excluded_paths:
                                continue
                            raw_paths.append(p)
                            if len(raw_paths) >= enum_cap:
                                break
                    except (nx.NetworkXError, nx.NodeNotFound):
                        continue

                    if not raw_paths:
                        # Fallback: keep the BFS shortest path if it satisfies
                        # min_hops. Without this, raising metapath_min_domains
                        # / min_hops on a sparse pair returns nothing rather
                        # than the historical 2-hop shortest path.
                        sp = reachable[target_id]
                        if (len(sp) - 1 >= min_hops
                                and self._path_intermediates_are_atoms(sp)
                                and (metapath_min_domains <= 1
                                     or self._path_distinct_atom_domains(sp)
                                        >= metapath_min_domains)
                                and self._path_meets_evidence_floor(
                                    sp, min_evidence_per_node)
                                and (
                                    required_scope is None
                                    or self._path_claim_edge_count(sp, required_scope) >= 1
                                )
                                and tuple(sp) not in self._generation_excluded_paths):
                            raw_paths = [sp]
                        else:
                            continue

                    if prefer_longer_paths:
                        raw_paths.sort(key=len, reverse=True)
                    raw_paths = raw_paths[:max_paths_per_pair]

                    for raw_path in raw_paths:
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
                            metadata={"domain_a": dom_a, "domain_b": dom_b,
                                      "n_hops": len(raw_path) - 1},
                        )
                        h.explanation = self._generate_explanation(h)
                        h.composite_score = self._composite_score(h)
                        all_hypotheses.append(h)

                        pair_count += 1
                        if pair_count >= max_paths_per_pair:
                            break

        logger.info(f"batch generation complete: {len(all_hypotheses)} hypotheses from {len(domain_pairs)} domain pairs")

        if skip_post_process:
            return all_hypotheses
        all_hypotheses = self.post_process(all_hypotheses)
        return all_hypotheses

    # ── task-aware generation (Phase 2 of atom-algebra rollout) ───────────

    def batch_generate_for_task(
        self,
        task,                                    # neurooracle.src.atoms.Task
        max_hops: int = 4,
        max_paths_per_pair: int = 5,
        max_seeds_per_domain: int = 50,
        require_atom_touch: bool = False,
        min_hops: int = 2,
        metapath_min_domains: int = 2,
        prefer_longer_paths: bool = True,
        min_evidence_per_node: int = 1,
        random_seed: int | None = None,
        seed_diversity_fraction: float = 0.0,
    ) -> list[Hypothesis]:
        """Generate hypotheses scoped to a canonical Task.

        Internally this maps each input atom to its KG domain pool and the
        output atom to its target domain pool, then delegates to
        ``batch_generate`` over the resulting (input_domain × output_domain)
        pairs. Generated hypotheses are tagged with ``task_name`` /
        ``task_signature`` / ``task_modifier`` in their metadata so downstream
        consumers (NeuroBench, the explorer, leaderboards) can filter by task.

        For multi-input tasks (e.g. drug_response_prediction = {D, Rx, IM}→O):
            * paths *start* from any input-atom node and end at an output-atom
              node — the simplest behaviour, equivalent to the union of the
              corresponding domain pairs.
            * every formal multi-input task must visit at least one node from
              EVERY input atom (source/target included). ``require_atom_touch``
              is retained for API compatibility but can no longer weaken this
              registered task contract.

        Args:
            task: A :class:`neurooracle.src.atoms.Task` — typically one
                  pulled from ``CANONICAL_TASKS``.
            max_hops, max_paths_per_pair, max_seeds_per_domain: forwarded
                  to :meth:`batch_generate`.
            require_atom_touch: explicitly request the multi-atom path check;
                  registered multi-input tasks enable it automatically.

        Returns:
            List of :class:`Hypothesis`, tagged with the source task in
            metadata. Sorted/filtered by the existing ``post_process``
            pipeline (which ``batch_generate`` calls internally).
        """
        from .atoms import Task as _Task, ATOM_TO_DOMAINS  # local import: avoid circulars

        if not isinstance(task, _Task):
            raise TypeError(f"expected atoms.Task, got {type(task).__name__}")

        output_domains = ATOM_TO_DOMAINS[task.output]
        input_domains: set[str] = set()
        for in_atom in task.inputs:
            input_domains |= ATOM_TO_DOMAINS[in_atom]

        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        # Atom-domain maps are sets. Sorting prevents Python hash randomisation
        # from changing the task expansion and therefore the ranked hypotheses.
        for in_dom in sorted(input_domains):
            for out_dom in sorted(output_domains):
                if in_dom == out_dom:
                    continue
                key = (in_dom, out_dom)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)

        if not pairs:
            logger.warning(
                f"task '{task.name}' produced no domain pairs "
                f"(input atoms: {[a.value for a in task.inputs]}, "
                f"output atom: {task.output.value})"
            )
            return []

        logger.info(
            f"task '{task.name}' [{task.signature}] expanded to "
            f"{len(pairs)} domain pair(s)"
        )

        hyps = self.batch_generate(
            domain_pairs=pairs,
            max_hops=max_hops,
            max_paths_per_pair=max_paths_per_pair,
            max_seeds_per_domain=max_seeds_per_domain,
            skip_post_process=True,
            min_hops=min_hops,
            metapath_min_domains=metapath_min_domains,
            prefer_longer_paths=prefer_longer_paths,
            min_evidence_per_node=min_evidence_per_node,
            random_seed=random_seed,
            seed_diversity_fraction=seed_diversity_fraction,
        )

        semantic_max = min(
            5000,
            max(
                100,
                max_seeds_per_domain
                * max_paths_per_pair
                * max(1, len(task.inputs)),
            ),
        )
        semantic_generator_name = (
            "_batch_generate_multi_input_task_from_scoped_claims"
            if len(task.inputs) > 1
            else "_batch_generate_task_from_scoped_claims"
        )
        semantic_generator = getattr(self, semantic_generator_name, None)
        semantic_hyps = (
            semantic_generator(
                task,
                max_hypotheses=semantic_max,
                random_seed=random_seed,
            )
            if getattr(self, "_task_required_claim_scope", None)
            and callable(semantic_generator)
            else []
        )
        if semantic_hyps:
            logger.info(
                "task '%s': added %d scoped evidence hypothesis(es)",
                task.name,
                len(semantic_hyps),
            )
            hyps.extend(semantic_hyps)

        # tag with task provenance BEFORE post_process so the task-aware
        # _is_dataset_outcome filter can read task_name.
        prepared: list[Hypothesis] = []
        for h in hyps:
            h.metadata = dict(h.metadata or {})
            h.metadata["task_name"] = task.name
            h.metadata["task_signature"] = task.signature
            h.metadata["task_modifier"] = task.modifier.value
            h.metadata["task_kind"] = "task"
            path_nodes = self._hypothesis_path_nodes(h)
            h.metadata.setdefault("path_node_ids", list(path_nodes))
            if path_nodes in self._generation_excluded_paths:
                continue
            if "feedback_mutation_affinity" not in h.metadata:
                mutation_affinity = self._feedback_mutation_affinity(path_nodes)
                h.metadata["feedback_mutation"] = mutation_affinity > 0.0
                h.metadata["feedback_mutation_affinity"] = mutation_affinity
            self._apply_task_testability_floor(h, task)
            if mutation_affinity := float(
                h.metadata.get("feedback_mutation_affinity") or 0.0
            ):
                if not h.metadata.get("feedback_mutation_score_applied"):
                    h.composite_score += 0.16 * mutation_affinity
                    h.metadata["feedback_mutation_score_applied"] = True
            prepared.append(h)
        hyps = prepared

        # Now apply post_process with task tags in place.
        before = len(hyps)
        hyps = self.post_process(hyps)
        logger.info(
            f"task '{task.name}': {before} raw -> {len(hyps)} after task-aware post_process"
        )

        # Registered multi-input tasks are meaningful only when every input is
        # represented in the proposed path.  Do not allow callers to silently
        # downgrade this task contract.
        strict_atom_touch = require_atom_touch or len(task.inputs) > 1
        if strict_atom_touch and len(task.inputs) > 1:
            kept = [h for h in hyps if self._path_touches_atoms(h, task.inputs)]
            logger.info(
                f"multi_input_atom_contract: kept {len(kept)}/{len(hyps)} "
                f"hypotheses for task '{task.name}'"
            )
            hyps = kept

        return hyps

    def _apply_task_testability_floor(self, h: Hypothesis, task) -> None:
        """Correct the imaging-only legacy floor for valid formal tasks.

        The generic scorer was written for NeuroClaw imaging experiments and
        assigns 0.15 when no modality is named.  That is appropriate for an
        unconstrained imaging hypothesis, but not for registered tasks such as
        adverse-event prediction whose atom contract does not contain IM.  This
        conservative floor uses only the task schema and source provenance; it
        never reads future support or validation labels.
        """

        if h.testability_score > 0.15 or not self._task_endpoint_contract_allowed(h):
            return

        formal_atoms = set(task.inputs) | {task.output}
        requires_imaging = Atom.IMAGING_MARKER in formal_atoms
        floor = 0.45 if requires_imaging else 0.50

        links = list(h.path or [])
        if links and all(link.claim_id for link in links):
            floor += 0.05

        metadata = h.metadata or {}
        paper_keys = {
            str(value)
            for value in metadata.get("source_paper_keys") or []
            if value
        }
        for link in links:
            paper = link.source_paper if isinstance(link.source_paper, dict) else {}
            paper_key = paper.get("pmid") or paper.get("doi") or paper.get("title")
            if paper_key:
                paper_keys.add(str(paper_key))
        if len(paper_keys) >= 2:
            floor += 0.05

        adjusted = min(0.60, floor)
        if adjusted <= h.testability_score:
            return
        previous = float(h.testability_score)
        h.testability_score = adjusted
        h.testability_reason = (
            f"formal task contract: {task.signature}; "
            + (
                "specific imaging endpoint registered"
                if requires_imaging
                else "imaging modality not required"
            )
            + ("; independent source papers" if len(paper_keys) >= 2 else "")
        )
        metadata["task_testability_floor"] = {
            "previous_score": previous,
            "adjusted_score": adjusted,
            "requires_imaging": requires_imaging,
            "independent_papers": len(paper_keys),
        }
        h.metadata = metadata
        h.composite_score = self._composite_score(h)

    def _path_touches_atoms(self, h: Hypothesis, atoms) -> bool:
        """Bind every multi-input atom to a distinct node on the evidence path."""
        from .atoms import ATOM_TO_DOMAINS

        input_order = tuple(sorted(dict.fromkeys(atoms), key=lambda atom: atom.value))
        needed = {atom: ATOM_TO_DOMAINS[atom] for atom in input_order}
        metadata = h.metadata or {}
        supplemental_roles: dict[str, set[Atom]] = defaultdict(set)

        def add_roles(node_id: str, values) -> None:
            for value in values or []:
                try:
                    supplemental_roles[node_id].add(Atom(str(value)))
                except ValueError:
                    continue

        add_roles(h.source_id, metadata.get("source_atoms"))
        add_roles(h.target_id, metadata.get("target_atoms"))
        mediator_ids = metadata.get("mediator_ids") or []
        mediator_atoms = metadata.get("mediator_atoms") or []
        for node_id, values in zip(mediator_ids, mediator_atoms):
            add_roles(str(node_id), values)
        input_entity_ids = metadata.get("input_entity_ids") or []
        input_entity_atoms = metadata.get("input_entity_atoms") or []
        for node_id, values in zip(input_entity_ids, input_entity_atoms):
            add_roles(str(node_id), values)

        node_ids: list[str] = []
        node_names: dict[str, str] = {}

        def add_node(node_id: str, node_name: str = "") -> None:
            node_id = str(node_id or "")
            if not node_id:
                return
            if node_id not in node_ids:
                node_ids.append(node_id)
            if node_name and node_id not in node_names:
                node_names[node_id] = str(node_name)

        add_node(h.source_id, h.source_name)
        add_node(h.target_id, h.target_name)
        for link in (h.path or []):
            if getattr(link, "from_id", None):
                add_node(link.from_id, getattr(link, "from_name", ""))
            if getattr(link, "to_id", None):
                add_node(link.to_id, getattr(link, "to_name", ""))

        def atom_specificity(
            atom: Atom,
            node_id: str,
            node_domains: set[str],
        ) -> int:
            if atom is not Atom.IMAGING_MARKER:
                return 0
            if node_id.startswith(("IM:", "NCL_IMAGING:", "NCL_BIOMARKER:")):
                return 0
            if node_domains & {"imaging_feature", "connectivity", "biomarker"}:
                return 0
            if "neuroanatomy" in node_domains:
                return 1
            return 2

        candidates: dict[Atom, list[tuple[int, int, int, str]]] = {
            atom: [] for atom in input_order
        }
        for node_index, nid in enumerate(node_ids):
            node = self._index.get(nid)
            node_domains = set(node.domain_tags or []) if node is not None else set()
            canonical_roles = set(concept_atom_roles(node))
            explicit_roles = supplemental_roles.get(nid, set())
            for atom, atom_domains in needed.items():
                if atom in explicit_roles:
                    priority = 0
                elif atom in canonical_roles:
                    priority = 1
                elif atom_domains & node_domains:
                    priority = 2
                else:
                    continue
                candidates[atom].append((
                    priority,
                    atom_specificity(atom, nid, node_domains),
                    node_index,
                    nid,
                ))

        if any(not candidates[atom] for atom in input_order):
            return False

        assignment_order = sorted(
            input_order,
            key=lambda atom: (len(candidates[atom]), atom.value),
        )
        bindings: dict[Atom, str] = {}
        used_nodes: set[str] = set()

        def assign(index: int) -> bool:
            if index == len(assignment_order):
                return True
            atom = assignment_order[index]
            for _priority, _specificity, _node_index, node_id in sorted(candidates[atom]):
                if node_id in used_nodes:
                    continue
                bindings[atom] = node_id
                used_nodes.add(node_id)
                if assign(index + 1):
                    return True
                used_nodes.remove(node_id)
                bindings.pop(atom, None)
            return False

        if not assign(0):
            return False

        h.metadata = dict(metadata)
        h.metadata["input_atom_order"] = [atom.value for atom in input_order]
        h.metadata["input_entity_ids"] = [bindings[atom] for atom in input_order]
        h.metadata["input_entity_names"] = [
            node_names.get(bindings[atom])
            or getattr(self._index.get(bindings[atom]), "preferred_name", "")
            or bindings[atom]
            for atom in input_order
        ]
        h.metadata["input_entity_atoms"] = [[atom.value] for atom in input_order]
        h.metadata["input_binding_mode"] = "distinct_path_atom_assignment"
        h.metadata["input_atom_contract_version"] = "distinct_nodes.v1"
        return True

    def _hypothesis_semantic_endpoint(
        self,
        h: Hypothesis,
        *,
        source_side: bool,
    ) -> SemanticEndpoint:
        metadata = h.metadata or {}
        node_id = h.source_id if source_side else h.target_id
        name = h.source_name if source_side else h.target_name
        key = "source_atoms" if source_side else "target_atoms"
        roles = set(concept_atom_roles(self._index.get(node_id)))
        for value in metadata.get(key) or []:
            try:
                roles.add(Atom(str(value)))
            except ValueError:
                continue
        return SemanticEndpoint(
            entity_id=node_id,
            name=name,
            canonical_id=node_id,
            atoms=tuple(sorted(atom.value for atom in roles)),
            uses_canonical_id=node_id in self._index,
            name_score=1.0,
            role_compatible=True,
        )

    def _task_endpoint_contract_allowed(self, h: Hypothesis) -> bool:
        """Require generated endpoints to satisfy the registered task atoms."""

        task_name = str((h.metadata or {}).get("task_name") or "")
        if not task_name:
            return True
        from .atoms import task_by_name

        try:
            task = task_by_name(task_name)
        except KeyError:
            return True
        source = self._hypothesis_semantic_endpoint(h, source_side=True)
        target = self._hypothesis_semantic_endpoint(h, source_side=False)
        source_allowed = any(
            endpoint_matches_atom(source, atom, self._index)
            for atom in task.inputs
        )
        target_allowed = endpoint_matches_atom(target, task.output, self._index)
        scope = str(
            (h.metadata or {}).get("claim_case_study_id")
            or (h.metadata or {}).get("case_study_id")
            or task.name
        )
        return bool(
            source_allowed
            and target_allowed
            and case_study_endpoint_names_allowed(scope, source, target)
        )

    # ── chain-aware generation (TaskChain mediation paths) ────────────────

    @staticmethod
    def _claim_entity_id(atom, canonical_id: str, name: str) -> str:
        digest = hashlib.sha1(
            f"{atom.value}\x1f{canonical_id}\x1f{name}".encode("utf-8")
        ).hexdigest()[:16]
        return f"CLAIM_ENTITY:{digest}"

    def _batch_generate_from_scoped_claims(
        self,
        chain,
        max_chains: int,
    ) -> list[Hypothesis]:
        """Assemble an audited chain from claims belonging to one paper.

        Canonical concept IDs are useful graph anchors, but historical CUI
        collisions can merge two claim-local meanings into one global node.
        The claim payload remains source-linked and preserves the exact entity
        names and atom declarations, so scoped claim-chain generation treats it
        as the authority and records the canonical IDs only as provenance.
        """
        scope = self._chain_claim_first_scope
        if not scope:
            return []

        claims_by_paper: dict[str, list[tuple[str, dict]]] = {}
        for claim_id, node in self._index.items():
            if "claim" not in (node.domain_tags or []):
                continue
            meta = node.metadata or {}
            if scope not in self._claim_meta_scopes(meta):
                continue
            paper = meta.get("source_paper") or {}
            paper_key = str(
                paper.get("pmid")
                or paper.get("doi")
                or paper.get("title")
                or claim_id
            )
            claims_by_paper.setdefault(paper_key, []).append((claim_id, meta))

        def _normalise(value: object) -> str:
            return re.sub(r"\s+", " ", str(value or "").strip().lower())

        def _entity_rows(
            entries: list[tuple[str, dict]],
        ) -> dict[object, list[tuple[str, str, float]]]:
            rows = {atom: {} for atom in chain.chain}
            for claim_id, meta in entries:
                confidence = float(meta.get("confidence") or 0.5)
                for side in ("subject", "object"):
                    endpoint_id = str(meta.get(f"{side}_id") or "")
                    endpoint_name = str(meta.get(f"{side}_name") or "").strip()
                    if not endpoint_id or not endpoint_name:
                        continue
                    roles = self._chain_claim_endpoint_roles.get(
                        (claim_id, side), set()
                    )
                    for atom in roles:
                        if atom not in rows:
                            continue
                        key = (endpoint_id, endpoint_name)
                        current = rows[atom].get(key, (0.0, 0))
                        rows[atom][key] = (
                            current[0] + confidence,
                            current[1] + 1,
                        )

            ranked: dict[object, list[tuple[str, str, float]]] = {}
            for atom, values in rows.items():
                ranked[atom] = [
                    (endpoint_id, endpoint_name, score + 0.05 * count)
                    for (endpoint_id, endpoint_name), (score, count)
                    in sorted(
                        values.items(),
                        key=lambda item: (
                            item[1][0] + 0.05 * item[1][1],
                            item[0][1],
                        ),
                        reverse=True,
                    )[:3]
                ]
            return ranked

        def _entry_match_score(
            entry: tuple[str, dict],
            left: tuple[str, str, float],
            right: tuple[str, str, float],
        ) -> tuple[int, float, str]:
            claim_id, meta = entry
            subject = (
                str(meta.get("subject_id") or ""),
                _normalise(meta.get("subject_name")),
            )
            obj = (
                str(meta.get("object_id") or ""),
                _normalise(meta.get("object_name")),
            )
            left_key = (left[0], _normalise(left[1]))
            right_key = (right[0], _normalise(right[1]))
            exact = (
                (subject == left_key and obj == right_key)
                or (subject == right_key and obj == left_key)
            )
            text = _normalise(
                " ".join(
                    [
                        meta.get("subject_name", ""),
                        meta.get("object_name", ""),
                        meta.get("raw_text", ""),
                    ]
                )
            )
            left_seen = left_key[1] in text
            right_seen = right_key[1] in text
            level = 3 if exact else 2 if left_seen and right_seen else 1 if left_seen or right_seen else 0
            return level, float(meta.get("confidence") or 0.5), claim_id

        def _best_entry(
            entries: list[tuple[str, dict]],
            left: tuple[str, str, float],
            right: tuple[str, str, float],
        ) -> tuple[str, dict]:
            return max(
                entries,
                key=lambda entry: _entry_match_score(entry, left, right),
            )

        def _link(
            entry: tuple[str, dict],
            from_entity: tuple[str, str, float],
            to_entity: tuple[str, str, float],
            from_atom,
            to_atom,
            fallback_relation: str,
        ) -> HypothesisLink:
            claim_id, meta = entry
            subject_name = _normalise(meta.get("subject_name"))
            object_name = _normalise(meta.get("object_name"))
            direct = (
                subject_name == _normalise(from_entity[1])
                and object_name == _normalise(to_entity[1])
            )
            relation = (
                str(meta.get("predicate") or fallback_relation)
                if direct
                else fallback_relation
            )
            evidence = meta.get("evidence") or {}
            if not isinstance(evidence, dict):
                evidence = {"description": str(evidence)}
            source_paper = meta.get("source_paper") or {}
            if not isinstance(source_paper, dict):
                source_paper = {"reference": str(source_paper)}
            return HypothesisLink(
                from_id=self._claim_entity_id(
                    from_atom, from_entity[0], from_entity[1]
                ),
                from_name=from_entity[1],
                to_id=self._claim_entity_id(to_atom, to_entity[0], to_entity[1]),
                to_name=to_entity[1],
                relation_type=relation,
                confidence=float(meta.get("confidence") or 0.5),
                claim_id=claim_id,
                raw_text=str(meta.get("raw_text") or ""),
                evidence=dict(evidence),
                source_paper=dict(source_paper),
            )

        generated: list[Hypothesis] = []
        source_atom, mediator_atom, target_atom = chain.chain
        for paper_key, entries in claims_by_paper.items():
            entities = _entity_rows(entries)
            sources = entities.get(source_atom, [])
            mediators = entities.get(mediator_atom, [])
            targets = entities.get(target_atom, [])
            if not sources or not mediators or not targets:
                continue

            for source, mediator, target in itertools.product(
                sources, mediators, targets
            ):
                labels = {
                    _normalise(source[1]),
                    _normalise(mediator[1]),
                    _normalise(target[1]),
                }
                if len(labels) < 3:
                    continue
                gene_im_entry = _best_entry(entries, source, mediator)
                im_outcome_entry = _best_entry(entries, mediator, target)
                links = [
                    _link(
                        gene_im_entry,
                        source,
                        mediator,
                        source_atom,
                        mediator_atom,
                        "is_associated_with",
                    ),
                    _link(
                        im_outcome_entry,
                        mediator,
                        target,
                        mediator_atom,
                        target_atom,
                        "mediates",
                    ),
                ]
                supporting_claims = list(
                    dict.fromkeys(link.claim_id for link in links if link.claim_id)
                )
                source_id = links[0].from_id
                mediator_id = links[0].to_id
                target_id = links[-1].to_id
                digest = hashlib.sha1(
                    "\x1f".join(
                        [paper_key, source_id, mediator_id, target_id]
                    ).encode("utf-8")
                ).hexdigest()[:12]
                confidence = self._compute_confidence_score(links)
                novelty = self._compute_novelty_score(links)
                evidence = self._compute_evidence_score(links)
                testability, test_reason = self._compute_testability_score(links)
                hypothesis = Hypothesis(
                    id=f"HYP:CLAIM_CHAIN:{chain.name}:{digest}",
                    hypothesis_type="claim_chain",
                    source_id=source_id,
                    source_name=source[1],
                    target_id=target_id,
                    target_name=target[1],
                    path=links,
                    confidence_score=confidence,
                    novelty_score=novelty,
                    evidence_score=evidence,
                    testability_score=testability,
                    supporting_claims=supporting_claims,
                    testability_reason=test_reason,
                    metadata={
                        "chain_name": chain.name,
                        "chain_signature": chain.signature,
                        "chain_atoms": [atom.value for atom in chain.chain],
                        "chain_modifier": chain.modifier.value,
                        "mediator_ids": [mediator_id],
                        "mediator_names": [mediator[1]],
                        "task_kind": "claim_chain",
                        "generation_mode": "scoped_claim_first",
                        "claim_case_study_id": scope,
                        "source_paper_key": paper_key,
                        "canonical_entity_ids": {
                            "source": source[0],
                            "mediator": mediator[0],
                            "target": target[0],
                        },
                    },
                )
                hypothesis.explanation = self._generate_explanation(hypothesis)
                hypothesis.composite_score = self._composite_score(hypothesis)
                generated.append(hypothesis)

        best_by_names: dict[tuple[str, str, str], Hypothesis] = {}
        for hypothesis in generated:
            mediator_name = hypothesis.metadata["mediator_names"][0]
            key = (
                _normalise(hypothesis.source_name),
                _normalise(mediator_name),
                _normalise(hypothesis.target_name),
            )
            current = best_by_names.get(key)
            if current is None or hypothesis.composite_score > current.composite_score:
                best_by_names[key] = hypothesis
        ranked = sorted(
            best_by_names.values(),
            key=lambda hypothesis: (
                hypothesis.composite_score,
                hypothesis.evidence_score,
                hypothesis.id,
            ),
            reverse=True,
        )
        logger.info(
            "chain '%s': scoped claim-first assembly produced %d unique chain(s) "
            "from %d paper(s)",
            chain.name,
            len(ranked),
            len(claims_by_paper),
        )
        return ranked[:max_chains]

    def batch_generate_for_chain(
        self,
        chain,                                   # neurooracle.src.atoms.TaskChain
        max_hops_per_segment: int = 2,
        max_paths_per_segment: int = 3,
        max_seeds: int = 30,
        max_chains: int = 200,
        min_evidence_per_node: int = 1,
    ) -> list[Hypothesis]:
        """Generate hypotheses scoped to a canonical TaskChain.

        Unlike :meth:`batch_generate_for_task` (where input atoms are parallel
        and a path only needs to start at any input and end at the output),
        a chain forces the path to transit atom domains in the listed order:
        ``chain[0] → chain[1] → ... → chain[-1]``. Intermediate atoms are
        treated as mechanistic mediators, not parallel inputs.

        Strategy: stitch segments. For each adjacent atom pair (Aᵢ, Aᵢ₊₁) we
        run a BFS-bounded search of length ≤ ``max_hops_per_segment``, then
        join consecutive segments by matching the segment-end node of the
        previous segment to the segment-start of the next.

        The generated paths are flattened back into linear ``HypothesisLink``
        chains (the same shape ``batch_generate`` produces), but tagged with
        ``chain_name`` / ``chain_signature`` / ``chain_atoms`` /
        ``mediator_ids`` in metadata so downstream consumers can identify
        them as mediated.

        Args:
            chain: A :class:`neurooracle.src.atoms.TaskChain` — typically one
                pulled from ``CANONICAL_CHAINS``.
            max_hops_per_segment: Max edge count between two consecutive
                atom-domain anchors. Total path length is bounded by
                (len(chain) - 1) × max_hops_per_segment.
            max_paths_per_segment: Number of intermediate paths kept for
                each (segment-source, segment-target) pair.
            max_seeds: Cap on seed nodes drawn from the chain's source atom.
            max_chains: Hard cap on total returned hypotheses. Stitching
                explodes combinatorially; this prevents runaway output.

        Returns:
            List of :class:`Hypothesis` representing complete mediation
            paths through the chain. Empty list if any segment has no
            connections in the current KG.
        """
        from .atoms import TaskChain as _TaskChain, ATOM_TO_DOMAINS

        if not isinstance(chain, _TaskChain):
            raise TypeError(f"expected atoms.TaskChain, got {type(chain).__name__}")

        if self._chain_claim_first_scope:
            return self._batch_generate_from_scoped_claims(chain, max_chains)

        atoms = chain.chain
        # Per-atom domain pools, with claim/PATH_IGNORE nodes excluded.
        atom_node_pools: list[set[str]] = []
        for atom in atoms:
            doms = ATOM_TO_DOMAINS[atom] | frozenset(
                self._chain_atom_extra_domains.get(atom, frozenset())
            )
            extra_filter = self._chain_atom_filters.get(atom)
            explicit_pool = self._chain_atom_explicit_pools.get(atom, set())
            pool = {
                nid for nid, data in self.G.nodes(data=True)
                if ((set(data.get("domain_tags", [])) & doms)
                    or nid in explicit_pool)
                and "claim" not in data.get("domain_tags", [])
                and nid not in self._path_ignore_ids
                and (min_evidence_per_node <= 0
                     or self._node_non_tree_degree(nid) >= min_evidence_per_node)
                and (extra_filter is None
                     or extra_filter(nid, self._index.get(nid)))
            }
            atom_node_pools.append(pool)
            if extra_filter is not None:
                logger.info(
                    f"chain '{chain.name}': atom {atom.value} pool restricted "
                    f"by filter -> {len(pool)} node(s)"
                )

        if any(not p for p in atom_node_pools):
            empty = [atoms[i].value for i, p in enumerate(atom_node_pools) if not p]
            logger.warning(
                f"chain '{chain.name}' [{chain.signature}]: empty node pool "
                f"for atom(s) {empty} — returning []"
            )
            return []

        # Seed from chain[0]; prefer high-degree nodes for connectivity.
        seeds = [
            n for n in atom_node_pools[0]
            if n in self.G
        ]
        seed_ranker: Optional[Callable] = self._chain_atom_rankers.get(atoms[0])
        if seed_ranker is not None:
            seeds.sort(
                key=lambda n: (
                    seed_ranker(n, self._index.get(n)),
                    self.G.degree(n),
                ),
                reverse=True,
            )
        else:
            seeds.sort(key=lambda n: self.G.degree(n), reverse=True)
        seeds = seeds[:max_seeds]

        # ``frontier`` holds (path, anchor_indices) pairs. We track anchor
        # positions during stitching so we can identify mediators precisely
        # — re-deriving them post-hoc is unreliable when an intermediate
        # node happens to also carry the next anchor atom's domain tag.
        frontier: list[tuple[list[str], list[int]]] = [([s], [0]) for s in seeds]

        for seg_idx in range(len(atoms) - 1):
            next_pool = atom_node_pools[seg_idx + 1]
            extended: list[tuple[list[str], list[int]]] = []

            for partial, anchors in frontier:
                head = partial[-1]
                if head not in self.G:
                    continue
                try:
                    reachable = nx.single_source_shortest_path(
                        self.G, head, cutoff=max_hops_per_segment
                    )
                except nx.NetworkXError:
                    continue

                # Candidate next-anchor nodes within next atom's pool.
                partial_candidates: list[tuple[list[str], list[int]]] = []
                for tgt, sub_path in reachable.items():
                    if tgt == head or tgt not in next_pool:
                        continue
                    # Disallow re-entering atoms already visited as anchors.
                    anchor_set = set(partial)
                    if any(n in anchor_set for n in sub_path[1:]):
                        continue
                    # Within-segment bridge nodes must be atom-domain nodes
                    # (not claims, not infrastructure). Endpoints sub_path[0]
                    # and sub_path[-1] are anchors — already pinned to atom
                    # pools — so we only check sub_path[1:-1].
                    _allowed = _allowed_atom_domains()
                    bad_bridge = False
                    for n in sub_path[1:-1]:
                        node = self._index.get(n)
                        if node is None:
                            bad_bridge = True
                            break
                        if n in self._chain_forbidden_bridge_ids:
                            bad_bridge = True
                            break
                        node_doms = set(node.domain_tags or [])
                        if "claim" in node_doms or not (node_doms & _allowed):
                            bad_bridge = True
                            break
                        if (min_evidence_per_node > 0
                                and self._node_non_tree_degree(n)
                                    < min_evidence_per_node):
                            bad_bridge = True
                            break
                    if bad_bridge:
                        continue
                    new_path = partial + sub_path[1:]
                    new_anchors = anchors + [len(new_path) - 1]
                    partial_candidates.append((new_path, new_anchors))

                partial_candidates = self._sort_chain_candidates(
                    partial_candidates,
                    next_atom=atoms[seg_idx + 1],
                )
                extended.extend(partial_candidates[:max_paths_per_segment])

                if len(extended) >= max_chains * 4:
                    # safety brake — segment expansion combinatorially explodes
                    break

            if not extended:
                logger.info(
                    f"chain '{chain.name}': segment {seg_idx} "
                    f"({atoms[seg_idx].value}→{atoms[seg_idx + 1].value}) "
                    f"produced no extensions — returning []"
                )
                return []
            frontier = extended

        # Build hypotheses from the surviving full chains.
        survivors = self._select_chain_survivors(frontier, max_chains)
        logger.info(
            f"chain '{chain.name}' [{chain.signature}]: "
            f"{len(survivors)} mediation path(s) from {len(seeds)} seed(s)"
        )

        hyps: list[Hypothesis] = []
        for path, anchors in survivors:
            links = self._enrich_path(path)
            if not links:
                continue

            conf = self._compute_confidence_score(links)
            nov = self._compute_novelty_score(links)
            evi = self._compute_evidence_score(links)
            test, test_reason = self._compute_testability_score(links)
            claim_ids = [l.claim_id for l in links if l.claim_id]

            mediator_ids = [path[i] for i in anchors[1:-1]] if len(anchors) >= 3 else []
            mediator_names = [
                self._index[m].preferred_name
                for m in mediator_ids if m in self._index
            ]

            path_digest = hashlib.sha1(
                "\x1f".join(path).encode("utf-8")
            ).hexdigest()[:12]
            h = Hypothesis(
                id=f"HYP:CHAIN:{chain.name}:{path_digest}",
                hypothesis_type="chain",
                source_id=path[0],
                source_name=self._index[path[0]].preferred_name,
                target_id=path[-1],
                target_name=self._index[path[-1]].preferred_name,
                path=links,
                confidence_score=conf,
                novelty_score=nov,
                evidence_score=evi,
                testability_score=test,
                supporting_claims=claim_ids,
                testability_reason=test_reason,
                metadata={
                    "chain_name": chain.name,
                    "chain_signature": chain.signature,
                    "chain_atoms": [a.value for a in atoms],
                    "chain_modifier": chain.modifier.value,
                    "mediator_ids": mediator_ids,
                    "mediator_names": mediator_names,
                    "task_kind": "chain",
                },
            )
            h.explanation = self._generate_explanation(h)
            h.composite_score = self._composite_score(h)
            hyps.append(h)

        # Prefix-dedup: same (source + intermediate anchor atoms) prefix
        # often produces several near-duplicate chains that differ only in
        # the terminal outcome (e.g. SLC6A4 → Amygdala → IM → MADRS vs the
        # same prefix → BDI). Keep the highest-composite per prefix so the
        # critic doesn't waste rounds on permutations of the same mechanism.
        if hyps:
            best_by_prefix: dict[tuple, Hypothesis] = {}
            for h in hyps:
                prefix = (h.source_id, *((h.metadata or {}).get("mediator_ids") or []))
                cur = best_by_prefix.get(prefix)
                if cur is None or h.composite_score > cur.composite_score:
                    best_by_prefix[prefix] = h
            deduped = list(best_by_prefix.values())
            if len(deduped) < len(hyps):
                logger.info(
                    f"chain '{chain.name}': prefix-dedup {len(hyps)} -> "
                    f"{len(deduped)} (kept best outcome per anchor prefix)"
                )
            hyps = deduped

        hyps = self.post_process(hyps)
        return hyps

    def post_process(
        self,
        hypotheses: list[Hypothesis],
        min_hops: int = 2,
        filter_vague_relations: bool = True,
        filter_non_measurable: bool = True,
        max_hops_filter: int = 5,
        min_evidence_per_node: int = 1,
    ) -> list[Hypothesis]:
        """Filter low-quality hypotheses after generation.

        Filters:
        1. Noisy entities — source/target name matches NOISE_PATTERNS
        2. 1-hop hypotheses — too simple, just restates existing edges
        3. Vague relations — all links are is_associated_with / associated_with / about
        4. Non-measurable biomarkers — entities not directly measurable from brain imaging
        5. Pure association chains — no directional predicates (causes/treats/increases/etc.)
        6. Overly long paths — exceeds max_hops_filter (default 5) to reduce noise accumulation
        7. Tree-only nodes — any node on the path whose entire neighbourhood
           is is_a / part_of / about edges (no empirical anchor). Controlled
           by min_evidence_per_node; default 1 enforces "at least one
           non-tree edge per visited node".
        """
        before = len(hypotheses)
        filtered = []
        rejection_counts: Counter[str] = Counter()

        for h in hypotheses:
            # filter noisy entities. Only check CLM_CONCEPT nodes — Phase 1
            # curated vocabularies (MSH/COGAT/NN/...) are authoritative and
            # shouldn't be rejected just because their canonical name happens
            # to share a token with the noise word list (e.g. "Cognitive
            # Dysfunction" is a valid MeSH term despite "dysfunction" ∈
            # _NOISE_WORDS).
            noisy_names: list[str] = []
            for nid, name in [(h.source_id, h.source_name), (h.target_id, h.target_name)]:
                if nid.startswith("CLM_CONCEPT") and self._is_noisy_entity(name):
                    noisy_names.append(name)
            for link in h.path:
                if link.from_id.startswith("CLM_CONCEPT") and self._is_noisy_entity(link.from_name):
                    noisy_names.append(link.from_name)
                if link.to_id.startswith("CLM_CONCEPT") and self._is_noisy_entity(link.to_name):
                    noisy_names.append(link.to_name)
            if noisy_names:
                rejection_counts["noisy_entity"] += 1
                continue

            if not self._task_endpoint_contract_allowed(h):
                rejection_counts["invalid_task_endpoint_contract"] += 1
                continue

            if (
                (h.metadata or {}).get("task_name") == "functional_localization"
                and not self._is_valid_functional_localization_hypothesis(h)
            ):
                rejection_counts["invalid_functional_localization_path"] += 1
                continue

            # Scoped claim-first chains have already passed the formal
            # case-study reaudit and use claim-local entity IDs on purpose.
            # Graph-degree, cross-PMID bridge, and ontology-hub filters below
            # are designed for inferred graph walks and would reject these
            # source-linked within-paper mediation chains by construction.
            if (h.metadata or {}).get("generation_mode") == "scoped_claim_first":
                if (
                    len(h.path) >= min_hops
                    and h.source_name
                    and h.target_name
                    and (h.metadata or {}).get("mediator_names")
                    and all(link.claim_id for link in h.path)
                ):
                    filtered.append(h)
                    rejection_counts["accepted_scoped_claim_first"] += 1
                else:
                    rejection_counts["invalid_scoped_claim_first"] += 1
                continue

            if (
                (h.metadata or {}).get("generation_mode")
                == "scoped_multi_input_evidence_graph"
            ):
                metadata = h.metadata or {}
                expected_atoms = set(metadata.get("input_atom_order") or [])
                observed_atoms = {
                    str(value)
                    for values in metadata.get("input_entity_atoms") or []
                    for value in values or []
                }
                input_ids = [
                    str(value)
                    for value in metadata.get("input_entity_ids") or []
                    if value
                ]
                paper_keys = {
                    str(value)
                    for value in metadata.get("source_paper_keys") or []
                    if value
                }
                names = [
                    *(metadata.get("input_entity_names") or []),
                    h.target_name,
                ]
                if (
                    len(expected_atoms) >= 2
                    and expected_atoms <= observed_atoms
                    and len(input_ids) == len(expected_atoms)
                    and len(set(input_ids)) == len(input_ids)
                    and len(h.path) >= len(expected_atoms)
                    and len(paper_keys) >= 2
                    and all(link.claim_id for link in h.path)
                    and all(name and not self._is_noisy_entity(name) for name in names)
                ):
                    filtered.append(h)
                    rejection_counts[
                        "accepted_scoped_multi_input_evidence_graph"
                    ] += 1
                else:
                    rejection_counts[
                        "invalid_scoped_multi_input_evidence_graph"
                    ] += 1
                continue

            if (
                (h.metadata or {}).get("generation_mode")
                == "scoped_semantic_claim_bridge"
            ):
                paper_keys = {
                    str(value)
                    for value in (h.metadata or {}).get("source_paper_keys", [])
                    if value
                }
                continuous = all(
                    h.path[index].to_id == h.path[index + 1].from_id
                    for index in range(len(h.path) - 1)
                )
                names = [
                    h.source_name,
                    h.target_name,
                    *((h.metadata or {}).get("mediator_names") or []),
                ]
                if (
                    len(h.path) >= min_hops
                    and continuous
                    and len(paper_keys) >= 2
                    and all(link.claim_id for link in h.path)
                    and all(name and not self._is_noisy_entity(name) for name in names)
                ):
                    filtered.append(h)
                    rejection_counts["accepted_scoped_semantic_claim_bridge"] += 1
                else:
                    rejection_counts["invalid_scoped_semantic_claim_bridge"] += 1
                continue

            # filter tree-only nodes: any path node whose non-tree degree
            # falls below the floor is an ontology leaf with no empirical
            # anchor. Walking through it produces graph paths without
            # biological signal.
            if min_evidence_per_node > 0:
                path_nodes: list[str] = [h.source_id, h.target_id]
                for link in h.path:
                    path_nodes.append(link.from_id)
                    path_nodes.append(link.to_id)
                if any(self._node_non_tree_degree(nid) < min_evidence_per_node
                       for nid in path_nodes if nid):
                    rejection_counts["tree_only_node"] += 1
                    continue

            # filter 1-hop (single direct edge = no discovery value)
            if len(h.path) < min_hops:
                rejection_counts["too_short"] += 1
                continue

            # filter all-vague-relations
            if filter_vague_relations:
                relation_types = {l.relation_type for l in h.path}
                if relation_types and relation_types <= VAGUE_RELATIONS:
                    rejection_counts["all_vague_relations"] += 1
                    continue

            # filter single-PMID bridges (all hops cite the same paper = not a real bridge)
            if len(h.path) >= 2:
                pmids = set()
                for link in h.path:
                    pmid = link.source_paper.get("pmid", "") if isinstance(link.source_paper, dict) else ""
                    if pmid:
                        pmids.add(pmid)
                has_curated_task_edge = (
                    (h.metadata or {}).get("task_name") == "functional_localization"
                    and any(
                        not link.claim_id
                        and Atom.COGNITIVE_TASK
                        in concept_atom_roles(self._index.get(link.from_id))
                        for link in h.path
                    )
                )
                if len(pmids) == 1 and not has_curated_task_edge:
                    rejection_counts["single_paper_bridge"] += 1
                    continue

            # filter non-measurable biomarkers (not testable from imaging)
            if filter_non_measurable:
                if self._has_non_measurable_entity(h):
                    rejection_counts["non_measurable_entity"] += 1
                    continue

            # filter biologically implausible paths (brain region → non-neurological target)
            if self._has_implausible_path(h):
                rejection_counts["implausible_path"] += 1
                continue

            # filter paths with weak evidence (target not mentioned in raw_text)
            if self._has_weak_evidence(h):
                rejection_counts["weak_evidence"] += 1
                continue

            # filter paths where both ends of any edge are broad hubs
            # ("Brain Diseases --causes--> Cognitive Dysfunction" is uninformative)
            if self._has_hub_to_hub_edge(h):
                rejection_counts["hub_to_hub_edge"] += 1
                continue

            # filter paths touching any vague COGAT/MeSH umbrella hub
            # (memory/logic/loss/activation/risk/stress/Brain/Neurons).
            # These nodes are too abstract to drive a DL experiment whether
            # they appear as source, target, or intermediate.
            if self._touches_path_ignore_node(h):
                rejection_counts["ignored_umbrella_node"] += 1
                continue

            # filter paths that transit through disease mega-hubs as
            # intermediate nodes (A → Disease → B is uninformative).
            # These nodes are still valid as source/target endpoints.
            if self._transits_intermediate_only_hub(h):
                rejection_counts["intermediate_hub"] += 1
                continue

            # (C-1) filter paths whose INTERMEDIATE node is a generic
            # phrase ("neural activity", "disease progression", "grey
            # matter", ...). Endpoints are not checked here.
            if self._has_intermediate_generic_phrase(h):
                rejection_counts["generic_intermediate"] += 1
                continue

            # (C-2) filter paths whose directional density is too thin
            # (3+ hops with < 50% directional relations = too vague to
            # be a mechanism hypothesis).
            if self._has_thin_directional_density(h):
                rejection_counts["thin_directional_density"] += 1
                continue

            # filter: target must be a dataset outcome (diagnosis/cognition/behavior/
            # personality/motor). Predicting "White Matter" or "Neurons" is not a
            # hypothesis UKB/ADNI/HCP can directly test — those are imaging features
            # used as INPUTS, not outcomes.
            if not self._is_dataset_outcome(h):
                rejection_counts["non_dataset_outcome"] += 1
                continue

            # (C-3) filter: target name is an umbrella concept ("skill",
            # "disease", "neurological disorder", "clinical features")
            # even though it passes the outcome keyword check. These
            # can't anchor a concrete DL label.
            if self._is_too_broad_target(h.target_name):
                rejection_counts["broad_target"] += 1
                continue

            # (P2) filter: source is an umbrella concept (imaging modality,
            # super-category, abstract process). batch_generate seeds via
            # _sample_domain_nodes which already drops these, but other
            # entry points (task pipelines, manual paths) skip that filter
            # so we mirror the gate at post_process for defence-in-depth.
            if self._is_umbrella_source(h.source_name):
                rejection_counts["umbrella_source"] += 1
                continue

            # filter paths with no directional predicates (pure association chains)
            if len(h.path) >= 2:
                relation_types = {l.relation_type for l in h.path}
                if not (relation_types & DIRECTIONAL_RELATIONS):
                    rejection_counts["no_directional_relation"] += 1
                    continue

            # filter paths that exceed max hop length (noise accumulation)
            if len(h.path) > max_hops_filter:
                rejection_counts["too_long"] += 1
                continue

            filtered.append(h)

        # Deduplicate ordinary paths by endpoint pair. Multi-input hypotheses use
        # the complete input tuple; otherwise distinct evidence graphs collapse
        # merely because their display source is the first registered input.
        pair_groups = defaultdict(list)
        for h in filtered:
            if (h.metadata or {}).get("generation_mode") == "scoped_multi_input_evidence_graph":
                key = (
                    tuple((h.metadata or {}).get("input_entity_ids") or []),
                    h.target_id,
                )
            else:
                key = (h.source_id, h.target_id)
            pair_groups[key].append(h)

        deduplicated = []
        for key, group in pair_groups.items():
            # Sort by composite score descending
            group.sort(key=lambda x: x.composite_score, reverse=True)
            # Static generation keeps the historical top-2 rule. A dynamic
            # loop may retain a few more legal path instantiations so later
            # rounds can explore distinct mediators without duplicating paths.
            deduplicated.extend(group[: self._generation_max_paths_per_endpoint])

        rejection_counts["accepted_before_dedup"] = len(filtered)
        rejection_counts["removed_by_pair_dedup"] = len(filtered) - len(deduplicated)
        rejection_counts["accepted_final"] = len(deduplicated)
        rejection_counts["input"] = before
        self.last_post_process_stats = dict(rejection_counts)

        logger.info(f"post_process: {before} -> {len(filtered)} filtered -> {len(deduplicated)} deduplicated "
                     f"(removed {before - len(deduplicated)} total); reasons={dict(rejection_counts)}")
        return deduplicated

    def _is_valid_functional_localization_hypothesis(self, h: Hypothesis) -> bool:
        """Keep task-to-neural paths and reject disease-mediated association chains."""

        if (
            not h.path
            or looks_like_imaging_measurement(h.source_name)
            or looks_like_non_task_construct(h.source_name)
        ):
            return False
        mode = (h.metadata or {}).get("generation_mode")
        source_node = self._index.get(h.source_id)
        source_atoms = concept_atom_roles(source_node)
        if mode == "scoped_semantic_claim_bridge":
            if not looks_like_cognitive_task_or_stimulus(h.source_name):
                return False
        elif (
            Atom.COGNITIVE_TASK not in source_atoms
            or source_atoms & {Atom.DISEASE, Atom.OUTCOME, Atom.INDIVIDUAL_DATA}
        ):
            return False

        target_node = self._index.get(h.target_id)
        target_is_imaging = (
            is_specific_functional_imaging_readout(h.target_name)
            or (
                target_node is not None
                and "neuroanatomy" in set(target_node.domain_tags or [])
            )
        )
        if not target_is_imaging:
            return False

        if mode == "scoped_semantic_claim_bridge":
            mediator_ids = (h.metadata or {}).get("mediator_ids") or []
            mediator_names = (h.metadata or {}).get("mediator_names") or []
            if not mediator_ids or len(mediator_ids) != len(mediator_names):
                return False
            for mediator_id, mediator_name in zip(mediator_ids, mediator_names):
                node = self._index.get(str(mediator_id))
                is_neuroanatomy = bool(
                    node is not None
                    and "neuroanatomy" in set(node.domain_tags or [])
                )
                if (
                    not is_neuroanatomy
                    and not is_specific_functional_imaging_readout(mediator_name)
                ):
                    return False
            return True

        intermediate_ids = [link.to_id for link in h.path[:-1]]
        if not intermediate_ids:
            return False
        for node_id in intermediate_ids:
            atoms = concept_atom_roles(self._index.get(node_id))
            if Atom.DISEASE in atoms or Atom.OUTCOME in atoms:
                return False
            if Atom.IMAGING_MARKER not in atoms:
                return False
        return True

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
        # Token-level noise should only reject names that are essentially
        # made of vague/process words. Phrases such as "polygenic risk score",
        # "hippocampus volume", or "cortical thickness change" contain a
        # generic token but are still measurable entities.
        words = {
            w for w in re.split(r"[\s\-_,/]+", name_clean.lower())
            if w and w not in _NOISE_STOPWORDS
        }
        if words and words <= _NOISE_WORDS:
            return True
        return False

    @staticmethod
    def _is_generic_intermediate(name: str) -> bool:
        """(C-1) Phrase-level filter for intermediate node names that pass
        token-level `_NOISE_WORDS` but are still too vague.

        Examples that get blocked:
          - "neural activity"  (no individual noise token)
          - "functional connectivity" (legit metric but not a mechanism)
          - "disease progression"
          - "grey matter"  (umbrella)
          - "cognitive deficit"

        Only call on intermediate nodes — these phrases can be valid as
        endpoints (e.g. "functional connectivity" as a target metric).
        """
        if not name:
            return True
        s = name.strip()
        for pattern in _GENERIC_INTERMEDIATE_PATTERNS:
            if pattern.match(s):
                return True
        return False

    @staticmethod
    def _is_too_broad_target(name: str) -> bool:
        """(C-3) Block target names that pass the outcome keyword regex but
        are umbrella concepts ("disease", "skill", "neurological disorder",
        "clinical features"). A DL experiment can't be designed against
        these — you don't know which subtype to label.
        """
        if not name:
            return True
        s = name.strip()
        for pattern in _TARGET_TOO_BROAD_PATTERNS:
            if pattern.match(s):
                return True
        return False

    # (P2) Whitelist for the network-suffix umbrella check: these are
    # named, well-defined functional networks. They look like
    # "<modifier> network" but identify a specific, atlasable circuit
    # so we want them to pass even if the umbrella regex matches.
    _NAMED_NETWORK_EXCEPTIONS = frozenset({
        "default mode network", "salience network", "executive control network",
        "frontoparietal network", "central executive network",
        "dorsal attention network", "ventral attention network",
        "somatomotor network", "visual network", "limbic network",
        "language network", "auditory network", "cingulo-opercular network",
        # canonical anatomical pathways with strong specificity
        "mesolimbic pathway", "mesocortical pathway", "nigrostriatal pathway",
        "tuberoinfundibular pathway",
    })

    @staticmethod
    def _is_umbrella_source(name: str) -> bool:
        """(P2) Block source seeds that are umbrella concepts: imaging
        modalities, abstract processes, super-category nouns, or pathway
        umbrellas. These pass the noise/intermediate filters because they
        look like real entities, but they don't constrain a downstream DL
        experiment when used as the seed of a hypothesis.

        Only call on the SOURCE node. Endpoints can legitimately be
        umbrellas (predicting "neuroimaging finding" from a biomarker is
        fine), and intermediates are filtered separately.
        """
        if not name:
            return True
        s = name.strip()
        if s.lower() in HypothesisEngine._NAMED_NETWORK_EXCEPTIONS:
            return False
        for pattern in _UMBRELLA_SOURCE_PATTERNS:
            if pattern.match(s):
                return True
        return False

    def _has_intermediate_generic_phrase(self, h: Hypothesis) -> bool:
        """(C-1) Reject paths whose intermediate node is a generic phrase
        like "neural activity" or "disease progression". Endpoints are
        excluded from this check because some metrics (e.g. "functional
        connectivity") legitimately appear as outcomes.
        """
        if len(h.path) < 2:
            return False
        intermediate_names: list[str] = []
        for i, link in enumerate(h.path):
            # link.from_name is intermediate when i >= 1
            # link.to_name   is intermediate when i <  len(path) - 1
            if i >= 1:
                intermediate_names.append(link.from_name or "")
            if i < len(h.path) - 1:
                intermediate_names.append(link.to_name or "")
        for name in intermediate_names:
            if self._is_generic_intermediate(name):
                return True
        return False

    def _has_thin_directional_density(self, h: Hypothesis) -> bool:
        """(C-2) Reject paths where directional relations are too sparse.

        Current rule (older): >= 1 directional anywhere = pass.
        Problem: a 4-hop path with 1 directional + 3 vague edges still
        looks like a real chain to scoring but is essentially a vague
        association narrative.

        New rule:
          - 1-2 hop path: at least 1 directional (unchanged)
          - 3+ hop path: at least half of the edges must be directional
        """
        n = len(h.path)
        if n < 3:
            return False
        directional = sum(1 for l in h.path if l.relation_type in DIRECTIONAL_RELATIONS)
        return directional * 2 < n   # < 50% directional

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

    def _has_hub_to_hub_edge(self, h: Hypothesis) -> bool:
        """Reject paths containing any edge whose endpoints are both broad hubs.

        Example: "Brain Diseases --causes--> Cognitive Dysfunction" — both ends
        are top-level categories; the edge is too generic to be a mechanistic
        step in a hypothesis.

        Hub set is the top-N nodes by non-'about' degree, computed once and
        cached. Uses a low bar (N=50) because hubs are self-evidently generic.
        """
        if not hasattr(self, "_hub_id_set"):
            # Build once per engine instance
            from collections import Counter
            degree = Counter()
            for u, v, data in self.G.edges(data=True):
                if data.get("relation_type") != "about":
                    degree[u] += 1
                    degree[v] += 1
            top = degree.most_common(50)
            self._hub_id_set = {cid for cid, _ in top}

        for link in h.path:
            if link.from_id in self._hub_id_set and link.to_id in self._hub_id_set:
                return True
        return False

    def _touches_path_ignore_node(self, h: Hypothesis) -> bool:
        """Reject paths whose source, target, or any intermediate node is in
        the path-ignore set (vague COGAT/MeSH umbrella hubs).

        Catches concepts the token-based _is_noisy_entity misses because
        the names ("memory", "logic", "Brain", "Neurons") are legitimate
        English words but the KG concept id refers to an over-general
        umbrella that's not testable.
        """
        ignore = self._path_ignore_ids
        if h.source_id in ignore:
            return True
        if h.target_id in ignore:
            return True
        for link in h.path:
            if link.from_id in ignore:
                return True
            if link.to_id in ignore:
                return True
        return False

    def _transits_intermediate_only_hub(self, h: Hypothesis) -> bool:
        """Reject paths that use disease mega-hubs as intermediate transit.

        Intermediate-only-ignore nodes are valid as source/target
        (predicting Alzheimer is a real hypothesis) but not as middle
        hops (A -> Alzheimer -> B is just "both relate to AD").

        Exception: chain hypotheses pin specific atom positions (e.g.
        GENE→IM→DISEASE→OUTCOME) — a disease at the DISEASE-anchor
        position is required by the chain semantics, not a coincidental
        hub. Mediator anchors recorded at generation time are exempted.
        """
        if len(h.path) < 2:
            return False
        ignore = self._intermediate_only_ignore_ids
        anchor_exempt: set[str] = set()
        if h.hypothesis_type == "chain":
            anchor_exempt = set(h.metadata.get("mediator_ids") or [])
        for i, link in enumerate(h.path):
            if i >= 1 and link.from_id in ignore and link.from_id not in anchor_exempt:
                return True
            if i < len(h.path) - 1 and link.to_id in ignore and link.to_id not in anchor_exempt:
                return True
        return False

    def _is_dataset_outcome(self, h: Hypothesis) -> bool:
        """Check if target is a UKB/ADNI/HCP-testable outcome.

        The target's valid domain set is determined by the task this hypothesis
        was generated for: read ``metadata['task_name']``, look up the task's
        output atom in :data:`neurooracle.src.atoms.CANONICAL_TASKS`, and use
        ``ATOM_TO_DOMAINS[output]`` as the allowed outcome domains.

        This makes the filter task-aware: ``personalised_treatment`` (output
        DRUG) accepts drug-domain targets; ``brain_age`` (output
        INDIVIDUAL_DATA) accepts only dataset_variable targets, etc. The
        previous version used a fixed pool {disease, cognitive_function}
        which (a) blocked drug targets that personalised_treatment legitimately
        wants and (b) accepted clinical-disease targets for brain_age, which
        produced low-quality samples like ``IM → Dementia → ADNI:DOM_DX``.

        Falls back to the legacy union (disease + cognitive_function +
        decoding domains + outcome keyword regex) when ``task_name`` is
        missing (e.g. chain hypotheses, free-form imaging mode).
        """
        target = self._index.get(h.target_id)
        if target is None:
            return False

        domains = set(target.domain_tags)
        task_name = (h.metadata or {}).get("task_name") or ""

        if task_name:
            allowed = self._task_outcome_domains(task_name)
            if allowed is not None:
                if domains & allowed:
                    return True
                # Encoding edge case: cognitive_decoding / functional_localization
                # accept neuroanatomy as a target only when source is a
                # decoding-style stimulus.
                if "neuroanatomy" in allowed and "neuroanatomy" in domains:
                    return True
                return False

        # Legacy fallback for chains and untagged hypotheses.
        outcome_domains = _OUTCOME_DOMAINS | {"visual_stimulus", "emotion", "vigilance"}
        if domains & outcome_domains:
            return True
        if "neuroanatomy" in domains:
            source = self._index.get(h.source_id)
            if source:
                source_domains = set(source.domain_tags)
                if source_domains & {"visual_stimulus", "emotion", "vigilance"}:
                    return True
        if _OUTCOME_KEYWORDS.search(h.target_name):
            return True
        return False

    @staticmethod
    def _task_outcome_domains(task_name: str) -> Optional[frozenset[str]]:
        """Return the allowed target-domain set for a named task.

        Looks up the task's output atom in CANONICAL_TASKS and returns its
        ATOM_TO_DOMAINS pool. Returns None if the task name is unrecognised
        so callers can fall back to legacy logic. Cached per-process.
        """
        cache = HypothesisEngine._task_outcome_cache
        if task_name in cache:
            return cache[task_name]

        from .atoms import CANONICAL_TASKS, ATOM_TO_DOMAINS
        for task in CANONICAL_TASKS:
            if task.name == task_name:
                allowed = ATOM_TO_DOMAINS.get(task.output, frozenset())
                cache[task_name] = allowed
                return allowed
        cache[task_name] = None
        return None

    _task_outcome_cache: dict[str, Optional[frozenset[str]]] = {}

    def _has_weak_evidence(self, h: Hypothesis) -> bool:
        """Check if hypothesis path has weak evidence (target not mentioned in raw_text).

        For hypotheses where the target is a specific brain region, check if any hop's
        raw_text actually mentions that region. If not, the path is likely spurious
        (e.g., IL-1β → Internal Capsula where the evidence text talks about "grey matter"
        but never mentions internal capsule).

        Exception: paths anchored by curated functional facts (e.g. `evokes` from
        visual_stimulus to a functional ROI) carry programmatic confidence, not
        paper evidence — skip the raw_text requirement for them.
        """
        target_node = self._index.get(h.target_id)
        if not target_node or "neuroanatomy" not in target_node.domain_tags:
            return False

        # Skip paths whose source is a visual_stimulus / emotion / vigilance node, or
        # which contain at least one curated functional edge (evokes / decoded_from /
        # elicits). These are seeded from neuroscience textbooks, not paper claims.
        source_node = self._index.get(h.source_id)
        if source_node:
            decoding_domains = {"visual_stimulus", "emotion", "vigilance"}
            if any(t in decoding_domains for t in source_node.domain_tags):
                return False
        if any(l.relation_type in {"evokes", "decoded_from", "elicits"} for l in h.path):
            return False

        # Extract key terms from target name (e.g., "Internal Capsula" → ["internal", "capsula"])
        target_terms = set(re.findall(r'\b\w{4,}\b', h.target_name.lower()))
        if not target_terms:
            return False

        # Check if any hop mentions the target region
        for link in h.path:
            evidence = link.evidence if isinstance(link.evidence, dict) else {}
            raw = link.raw_text or evidence.get("raw_text", "")
            if raw:
                raw_lower = raw.lower()
                # If any target term appears in raw_text, evidence is OK
                if any(term in raw_lower for term in target_terms):
                    return False

        # No hop mentions the target region → weak evidence
        logger.debug(f"weak evidence: {h.id} target '{h.target_name}' not mentioned in any raw_text")
        return True

    # ── imaging-driven batch generation ──────────────────────────────

    def generate_case1_hypotheses(
        self,
        methods: tuple[str, ...] = (
            "exhaustive",
            "random_walk",
            "llm_brainstorm",
            "neurodiscovery",
        ),
        disease_names: tuple[str, ...] = (),
        atlas_rois: tuple[dict, ...] = (),
        atlas_label_names: tuple[str, ...] = (),
        atlas_label_sources: dict[str, tuple[str, ...]] | None = None,
        feature_space: tuple[dict, ...] = (),
        max_per_method: int | dict[str, int] = 40,
        random_seed: int | None = None,
    ) -> list[Hypothesis]:
        """Generate Case Study 1 hypotheses as executable experiment candidates.

        The candidate tuple is:

        ``disease x atlas/ROI x feature``.

        Each candidate asserts a measurable disease-associated alteration.
        Validation estimates the effect sign from data rather than fabricating
        an increase or decrease without feature-specific evidence. Cross-disease
        clusters are a downstream summary over validated single-disease
        candidates, not a generator-time assumption. Four generation strategies
        are supported:

        - ``exhaustive``: stable enumeration baseline. It can cover the full
          disease x region x feature space when uncapped.
        - ``random_walk``: stochastic, KG-aware sampling from disease/region
          neighborhoods when evidence exists, with random fallback.
        - ``llm_brainstorm``: a replaceable LLM-style prior sampler. It does not
          call an external API here; it encodes broad neuroscience templates so
          the pipeline stays runnable when model endpoints are unstable.
        - ``neurodiscovery``: NeuroOracle-guided ranking of the same candidate
          space using claim/edge support and executability.
        """
        methods = tuple(methods or ("exhaustive", "random_walk", "llm_brainstorm", "neurodiscovery"))
        allowed = {"exhaustive", "random_walk", "llm_brainstorm", "neurodiscovery"}
        unknown = sorted(set(methods) - allowed)
        if unknown:
            raise ValueError(f"unknown Case Study 1 generation method(s): {unknown}")

        diseases = self._case1_collect_diseases(disease_names)
        regions = self._case1_collect_regions(
            atlas_rois=atlas_rois,
            atlas_label_names=atlas_label_names,
            atlas_label_sources=atlas_label_sources or {},
        )
        features = [dict(f) for f in feature_space if f.get("id")]
        if not diseases:
            raise ValueError("Case Study 1 needs at least one disease option")
        if not regions:
            raise ValueError("Case Study 1 needs at least one atlas/ROI option")
        if not features:
            raise ValueError("Case Study 1 needs at least one feature option")
        support = self._case1_collect_support(diseases, regions)
        total_candidates = self._case1_total_candidate_count(
            n_diseases=len(diseases),
            n_regions=len(regions),
            n_features=len(features),
        )
        rng = random.Random(random_seed)
        limits = self._case1_method_limits(methods, max_per_method)

        by_method: dict[str, list[Hypothesis]] = {}
        for method in methods:
            limit = limits.get(method, 0)
            if limit == 0:
                by_method[method] = []
                continue
            if method == "exhaustive":
                hyps = self._case1_generate_exhaustive(
                    diseases, regions, features, support, limit, total_candidates
                )
            elif method == "random_walk":
                hyps = self._case1_generate_random_walk(
                    diseases, regions, features, support, limit, rng, total_candidates
                )
            elif method == "llm_brainstorm":
                hyps = self._case1_generate_llm_brainstorm(
                    diseases, regions, features, support, limit, rng, total_candidates
                )
            else:
                hyps = self._case1_generate_neurodiscovery(
                    diseases, regions, features, support, limit, total_candidates
                )
            by_method[method] = hyps

        out: list[Hypothesis] = []
        for method in methods:
            out.extend(by_method.get(method, ()))

        logger.info(
            "case1 candidate generation: %d diseases x %d regions x %d features "
            "-> %d possible candidates, emitted %d hypotheses across %s",
            len(diseases),
            len(regions),
            len(features),
            total_candidates,
            len(out),
            ", ".join(methods),
        )
        return out

    @staticmethod
    def _case1_method_limits(methods: tuple[str, ...], max_per_method: int | dict[str, int]) -> dict[str, int]:
        if isinstance(max_per_method, dict):
            return {m: int(max_per_method.get(m, 0)) for m in methods}
        return {m: int(max_per_method) for m in methods}

    @staticmethod
    def _case1_total_candidate_count(
        n_diseases: int,
        n_regions: int,
        n_features: int,
    ) -> int:
        return n_diseases * n_regions * n_features

    def _case1_collect_diseases(self, disease_names: tuple[str, ...]) -> list[dict]:
        by_norm: dict[str, ConceptNode] = {}
        for node in self._index.values():
            domains = set(node.domain_tags or [])
            if "disease" not in domains:
                continue
            keys = [node.preferred_name, *list(node.aliases or [])]
            for key in keys:
                norm = self._normalize_case1_name(key)
                if norm and norm not in by_norm:
                    by_norm[norm] = node

        diseases: list[dict] = []
        seen: set[str] = set()
        for name in disease_names:
            norm = self._normalize_case1_name(name)
            node = by_norm.get(norm)
            if node is None:
                for key, candidate in by_norm.items():
                    if norm and (norm in key or key in norm):
                        node = candidate
                        break
            if node is not None:
                did = node.id
                dname = node.preferred_name
            else:
                did = f"CASE1:DISEASE:{self._slug(name)}"
                dname = name
            if did in seen:
                continue
            seen.add(did)
            diseases.append({"id": did, "name": dname, "norm": norm})
        return diseases

    def _case1_collect_regions(
        self,
        atlas_rois: tuple[dict, ...] = (),
        atlas_label_names: tuple[str, ...] = (),
        atlas_label_sources: dict[str, tuple[str, ...]] | None = None,
    ) -> list[dict]:
        atlas_label_sources = atlas_label_sources or {}
        node_by_norm: dict[str, ConceptNode] = {}
        for node in self._index.values():
            if "neuroanatomy" not in (node.domain_tags or []):
                continue
            norm = self._normalize_region_label(node.preferred_name)
            if norm and norm not in node_by_norm:
                node_by_norm[norm] = node

        regions: list[dict] = []
        if atlas_rois:
            for spec in atlas_rois:
                atlas_name = str(spec.get("atlas_name", "unknown_atlas"))
                roi_index = str(spec.get("roi_index", ""))
                label = str(spec.get("label", "") or spec.get("name", "")).strip()
                if not roi_index or not label:
                    continue
                norm = self._normalize_region_label(label)
                node = node_by_norm.get(norm)
                rid = f"CASE1:ROI:{self._slug(atlas_name)}:{self._slug(roi_index)}"
                regions.append({
                    "id": rid,
                    "name": str(spec.get("name") or f"{atlas_name} ROI {roi_index}: {label}"),
                    "norm": norm,
                    "atlas_names": (atlas_name,),
                    "atlas_name": atlas_name,
                    "roi_index": roi_index,
                    "atlas_label": label,
                    "kg_node_id": node.id if node else "",
                })
        elif atlas_label_names:
            for label in atlas_label_names:
                norm = self._normalize_region_label(label)
                if not norm:
                    continue
                atlases = tuple(atlas_label_sources.get(label, ())) or ("unknown_atlas",)
                node = node_by_norm.get(norm)
                for atlas_name in atlases:
                    rid = f"CASE1:ROI:{self._slug(atlas_name)}:{self._slug(label)}"
                    regions.append({
                        "id": rid,
                        "name": f"{atlas_name}: {node.preferred_name if node else label}",
                        "norm": norm,
                        "atlas_names": (atlas_name,),
                        "atlas_name": atlas_name,
                        "roi_index": "",
                        "atlas_label": label,
                        "kg_node_id": node.id if node else "",
                    })
        else:
            for norm, node in sorted(node_by_norm.items(), key=lambda x: x[1].preferred_name)[:500]:
                regions.append({
                    "id": node.id,
                    "name": node.preferred_name,
                    "norm": norm,
                    "atlas_names": tuple(),
                    "atlas_name": "",
                    "roi_index": "",
                    "atlas_label": node.preferred_name,
                    "kg_node_id": node.id,
                })
        return sorted(regions, key=lambda r: (r.get("atlas_name", ""), self._roi_sort_key(r.get("roi_index", "")), r["name"]))

    @staticmethod
    def _roi_sort_key(value: str) -> tuple[int, str]:
        value = str(value or "")
        try:
            return (int(float(value)), value)
        except ValueError:
            return (10**9, value)

    def _case1_collect_support(self, diseases: list[dict], regions: list[dict]) -> dict:
        disease_by_id = {d["id"]: d for d in diseases}
        region_norms = {r["norm"] for r in regions}
        support: dict[str, dict] = {
            "by_region": {},
            "by_disease": {},
        }

        for u, v, edge_data in self.G.edges(data=True):
            disease_id = ""
            region_id = ""
            original_direction = "forward"
            if u in disease_by_id:
                disease_id = u
                region_id = v
            elif v in disease_by_id:
                disease_id = v
                region_id = u
                original_direction = "reverse"
            else:
                continue

            region_node = self._index.get(region_id)
            if region_node is None:
                continue
            if "neuroanatomy" not in (region_node.domain_tags or []):
                continue
            region_norm = self._normalize_region_label(region_node.preferred_name)
            if region_norms and region_norm not in region_norms:
                continue

            disease = disease_by_id[disease_id]
            claim_id = (edge_data.get("metadata") or {}).get("claim_id", "")
            claim_node = self._index.get(claim_id) if claim_id else None
            evidence = {}
            source_paper = {}
            raw_text = ""
            if claim_node is not None and claim_node.metadata:
                evidence = claim_node.metadata.get("evidence", {}) or {}
                source_paper = claim_node.metadata.get("source_paper", {}) or {}
                raw_text = claim_node.metadata.get("raw_text", "") or ""
            link = HypothesisLink(
                from_id=disease_id,
                from_name=disease["name"],
                to_id=region_id,
                to_name=region_node.preferred_name,
                relation_type=edge_data.get("relation_type", "is_associated_with"),
                confidence=float(edge_data.get("confidence", 0.5) or 0.5),
                claim_id=claim_id,
                raw_text=raw_text,
                evidence={
                    **evidence,
                    "original_edge_direction": original_direction,
                    **(edge_data.get("metadata") or {}),
                },
                source_paper=source_paper,
            )
            by_region = support["by_region"].setdefault(
                region_norm,
                {"disease_ids": set(), "links_by_disease": {}, "claim_ids": set(), "n_edges": 0},
            )
            by_region["disease_ids"].add(disease_id)
            by_region["links_by_disease"].setdefault(disease_id, []).append(link)
            by_region["n_edges"] += 1
            if claim_id:
                by_region["claim_ids"].add(claim_id)
            support["by_disease"].setdefault(disease_id, []).append((region_norm, link))
        return support

    def _case1_generate_exhaustive(
        self,
        diseases: list[dict],
        regions: list[dict],
        features: list[dict],
        support: dict,
        limit: int,
        total_candidates: int,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        for disease in diseases:
            for region in regions:
                for feature in features:
                    out.append(self._case1_make_hypothesis(
                        "exhaustive", disease, region, feature, support, total_candidates
                    ))
                    if limit > 0 and len(out) >= limit:
                        return out
        return out

    def _case1_generate_random_walk(
        self,
        diseases: list[dict],
        regions: list[dict],
        features: list[dict],
        support: dict,
        limit: int,
        rng: random.Random,
        total_candidates: int,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        seen: set[tuple] = set()
        by_disease = support.get("by_disease", {})
        attempts = max(200, limit * 30)
        for _ in range(attempts):
            if limit > 0 and len(out) >= limit:
                break
            disease = rng.choice(diseases)
            supported_regions = by_disease.get(disease["id"], [])
            if supported_regions and rng.random() < 0.7:
                region_norm, _ = rng.choice(supported_regions)
                region = next((r for r in regions if r["norm"] == region_norm), rng.choice(regions))
            else:
                region = rng.choice(regions)
            feature = rng.choice(features)
            key = self._case1_tuple_key(disease, region, feature)
            if key in seen:
                continue
            seen.add(key)
            out.append(self._case1_make_hypothesis(
                "random_walk", disease, region, feature, support, total_candidates
            ))
        return out

    def _case1_generate_llm_brainstorm(
        self,
        diseases: list[dict],
        regions: list[dict],
        features: list[dict],
        support: dict,
        limit: int,
        rng: random.Random,
        total_candidates: int,
    ) -> list[Hypothesis]:
        templates = (
            {
                "disease_terms": ("schizophrenia", "bipolar", "depressive"),
                "region_terms": ("anterior cingulate", "insula", "prefrontal", "striat"),
                "feature_families": ("seed_fc", "graph", "dynamic_fc"),
            },
            {
                "disease_terms": ("adhd", "anxiety", "post traumatic", "substance"),
                "region_terms": ("salience", "cingulate", "thalam", "frontoparietal"),
                "feature_families": ("seed_fc", "graph", "roi_activity"),
            },
            {
                "disease_terms": ("obsessive", "anorexia", "anxiety"),
                "region_terms": ("orbitofrontal", "caudate", "striat", "insula"),
                "feature_families": ("seed_fc", "graph"),
            },
            {
                "disease_terms": ("schizophrenia", "depressive", "bipolar", "adhd"),
                "region_terms": ("default mode", "hippocamp", "temporal", "parietal"),
                "feature_families": ("roi_activity", "seed_fc", "dynamic_fc"),
            },
        )
        out: list[Hypothesis] = []
        seen: set[tuple] = set()
        attempts = max(200, limit * 25)
        for _ in range(attempts):
            if limit > 0 and len(out) >= limit:
                break
            tmpl = rng.choice(templates)
            group_pool = [
                d for d in diseases
                if any(term in d["name"].casefold() for term in tmpl["disease_terms"])
            ] or diseases
            disease = rng.choice(group_pool)
            region_pool = [
                r for r in regions
                if any(term in r["norm"] for term in tmpl["region_terms"])
            ] or regions
            feature_pool = [
                f for f in features
                if f.get("family") in tmpl["feature_families"]
            ] or features
            region = rng.choice(region_pool)
            feature = rng.choice(feature_pool)
            key = self._case1_tuple_key(disease, region, feature)
            if key in seen:
                continue
            seen.add(key)
            h = self._case1_make_hypothesis(
                "llm_brainstorm", disease, region, feature, support, total_candidates
            )
            h.metadata["llm_brainstorm_template"] = tmpl
            out.append(h)
        return out

    def _case1_generate_neurodiscovery(
        self,
        diseases: list[dict],
        regions: list[dict],
        features: list[dict],
        support: dict,
        limit: int,
        total_candidates: int,
    ) -> list[Hypothesis]:
        heap: list[tuple[float, int, dict, dict, dict]] = []
        heap_limit = limit * 2000 if limit > 0 else 0
        counter = 0
        for disease in diseases:
            for region in regions:
                base = self._case1_support_score(disease, region, support)
                region_bonus = self._case1_region_prior(region)
                for feature in features:
                    feature_bonus = self._case1_feature_prior(feature)
                    score = 0.62 * base + 0.23 * feature_bonus + 0.15 * region_bonus
                    item = (score, counter, disease, region, feature)
                    counter += 1
                    if heap_limit <= 0:
                        heapq.heappush(heap, item)
                    elif len(heap) < heap_limit:
                        heapq.heappush(heap, item)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, item)
        picked = sorted(heap, key=lambda x: (x[0], x[1]), reverse=True)
        if limit > 0:
            diversified: list[tuple[float, int, dict, dict, dict]] = []
            per_region: dict[str, int] = {}
            per_disease_region: dict[tuple, int] = {}
            for item in picked:
                _, _, disease, region, _ = item
                region_key = region.get("norm") or region["id"]
                dr_key = (disease["id"], region_key)
                if per_disease_region.get(dr_key, 0) >= 2:
                    continue
                if per_region.get(region_key, 0) >= 8:
                    continue
                diversified.append(item)
                per_disease_region[dr_key] = per_disease_region.get(dr_key, 0) + 1
                per_region[region_key] = per_region.get(region_key, 0) + 1
                if len(diversified) >= limit:
                    break
            if len(diversified) < limit:
                used = {item[1] for item in diversified}
                for item in picked:
                    if item[1] in used:
                        continue
                    diversified.append(item)
                    if len(diversified) >= limit:
                        break
            picked = diversified
        return [
            self._case1_make_hypothesis(
                "neurodiscovery", disease, region, feature, support, total_candidates
            )
            for _, _, disease, region, feature in picked
        ]

    def _case1_make_hypothesis(
        self,
        method: str,
        disease: dict,
        region: dict,
        feature: dict,
        support: dict,
        total_candidates: int,
    ) -> Hypothesis:
        signature = "|".join([method, disease["id"], region["id"], feature["id"]])
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        source_id = disease["id"]
        source_name = disease["name"]
        target_id = f"CASE1:CANDIDATE:{self._slug(region['id'])}:{self._slug(feature['id'])}"
        target_name = f"{region['name']} | {feature['name']}"

        region_support = support.get("by_region", {}).get(region["norm"], {})
        links: list[HypothesisLink] = []
        claim_ids: set[str] = set()
        supported_links = list(
            (region_support.get("links_by_disease") or {}).get(disease["id"], [])
        )
        if supported_links:
            link = supported_links[0]
            claim_ids.update(l.claim_id for l in supported_links if l.claim_id)
            links.append(HypothesisLink(
                from_id=disease["id"],
                from_name=disease["name"],
                to_id=target_id,
                to_name=target_name,
                relation_type=link.relation_type,
                confidence=link.confidence,
                claim_id=link.claim_id,
                raw_text=link.raw_text,
                evidence={
                    **(link.evidence or {}),
                    "kg_region_id": link.to_id,
                    "kg_region_name": link.to_name,
                    "candidate_feature_id": feature["id"],
                },
                source_paper=link.source_paper,
            ))
        else:
            links.append(HypothesisLink(
                from_id=disease["id"],
                from_name=disease["name"],
                to_id=target_id,
                to_name=target_name,
                relation_type="candidate_disease_region_feature_test",
                confidence=0.35,
                evidence={
                    "candidate_feature_id": feature["id"],
                    "candidate_region": region["name"],
                    "claim_backed": False,
                },
            ))

        support_score = self._case1_support_score(disease, region, support)
        covered = 1 if disease["id"] in set(region_support.get("disease_ids", set())) else 0
        feature_prior = self._case1_feature_prior(feature)
        method_conf_bonus = {
            "exhaustive": 0.35,
            "random_walk": 0.42,
            "llm_brainstorm": 0.48,
            "neurodiscovery": 0.58,
        }[method]
        confidence = min(0.98, method_conf_bonus + 0.35 * support_score + 0.10 * feature_prior)
        evidence = min(0.98, 0.18 + 0.72 * support_score)
        novelty = {
            "exhaustive": 0.55,
            "random_walk": 0.68,
            "llm_brainstorm": 0.63,
            "neurodiscovery": 0.62,
        }[method]
        testability = 0.92 if feature.get("primary", False) else 0.78
        if feature.get("level") == "subject":
            testability -= 0.08

        direction_proposal = propose_case1_direction(
            disease,
            region,
            feature,
            supported_links,
        )
        direction = direction_proposal["direction"]
        explanation = case1_directional_statement(
            disease["name"], region["name"], feature["name"], direction
        )
        h = Hypothesis(
            id=f"case1_{method}_{digest}",
            hypothesis_type="case1_candidate",
            source_id=source_id,
            source_name=source_name,
            target_id=target_id,
            target_name=target_name,
            path=links,
            confidence_score=confidence,
            evidence_score=evidence,
            novelty_score=novelty,
            testability_score=max(0.01, testability),
            supporting_claims=sorted(claim_ids),
            explanation=explanation,
            testability_reason=(
                f"Executable as a disease-group comparison on atlas {', '.join(region.get('atlas_names') or ()) or 'KG'} "
                f"using feature {feature['id']} ({feature.get('requires', ())})."
            ),
            metadata={
                "case_study": "case1_transdiagnostic",
                "generation_method": method,
                "candidate_tuple": {
                    "disease_id": disease["id"],
                    "disease_name": disease["name"],
                    "disease_ids": (disease["id"],),
                    "diseases": (disease["name"],),
                    "region_id": region["id"],
                    "region_name": region["name"],
                    "atlas_names": tuple(region.get("atlas_names") or ()),
                    "atlas_name": region.get("atlas_name", ""),
                    "roi_index": region.get("roi_index", ""),
                    "atlas_label": region.get("atlas_label", region["name"]),
                    "feature_id": feature["id"],
                    "feature_name": feature["name"],
                    "feature_family": feature.get("family", ""),
                    "feature_modality": feature.get("modality", ""),
                    "feature_level": feature.get("level", ""),
                    "feature_requires": tuple(feature.get("requires", ())),
                    "direction": direction,
                },
                "direction_assumption": direction,
                "direction_source": direction_proposal["source"],
                "directional_evidence_votes": direction_proposal["directional_evidence_votes"],
                "display_title": case1_directional_title(
                    disease["name"], region["name"], feature["name"], direction
                ),
                "total_candidate_space": total_candidates,
                "support_score": support_score,
                "supporting_disease_count": covered,
                "supporting_claim_count": len(claim_ids),
                "feature_prior": feature_prior,
            },
        )
        h.composite_score = self._composite_score(h)
        return h

    @staticmethod
    def _case1_tuple_key(disease: dict, region: dict, feature: dict) -> tuple:
        return (
            disease["id"],
            region["id"],
            feature["id"],
        )

    def _case1_support_score(self, disease: dict, region: dict, support: dict) -> float:
        region_support = support.get("by_region", {}).get(region["norm"], {})
        if not region_support:
            return 0.02
        disease_links = (region_support.get("links_by_disease") or {}).get(disease["id"], [])
        if not disease_links:
            return 0.02
        coverage = 1.0 if disease_links else 0.0
        n_edges = len(disease_links)
        n_claims = len({link.claim_id for link in disease_links if link.claim_id})
        density = min(1.0, math.log1p(n_edges + n_claims) / math.log(20))
        return min(1.0, 0.68 * coverage + 0.32 * density)

    @staticmethod
    def _case1_feature_prior(feature: dict) -> float:
        family = feature.get("family", "")
        if family in {"seed_fc", "graph"}:
            return 0.92
        if family == "roi_activity":
            return 0.84
        if family == "dynamic_fc":
            return 0.76
        if family == "structural":
            return 0.68
        return 0.55

    @staticmethod
    def _case1_region_prior(region: dict) -> float:
        name = f"{region.get('name', '')} {region.get('norm', '')}".casefold()
        high_value = (
            "cingulate", "insula", "prefrontal", "striat", "caudate",
            "hippocamp", "amygdala", "thalam", "default mode", "salience",
            "frontoparietal", "orbitofrontal",
        )
        return 0.9 if any(term in name for term in high_value) else 0.5

    @staticmethod
    def _normalize_case1_name(label: str) -> str:
        s = (label or "").casefold()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _slug(label: str) -> str:
        s = (label or "").casefold()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return re.sub(r"_+", "_", s).strip("_") or "x"

    @staticmethod
    def _normalize_region_label(label: str) -> str:
        s = (label or "").casefold()
        s = s.replace("_", " ").replace("-", " ")
        s = re.sub(r"\b(left|right|lh|rh|l|r)\b", " ", s)
        s = re.sub(r"\bcortex\b", " ", s)
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _matching_atlases_for_region(
        region_norm: str,
        atlas_norm: set[str],
        atlas_source_norm: dict[str, tuple[str, ...]],
    ) -> list[str]:
        if not region_norm or not atlas_norm:
            return []
        matches: set[str] = set()
        region_tokens = set(region_norm.split())
        for label_norm, atlases in atlas_source_norm.items():
            if not label_norm:
                continue
            label_tokens = set(label_norm.split())
            direct = (
                region_norm == label_norm
                or region_norm in label_norm
                or label_norm in region_norm
            )
            token_overlap = (
                len(region_tokens) >= 2
                and len(label_tokens) >= 2
                and len(region_tokens & label_tokens) >= min(len(region_tokens), 2)
            )
            if direct or token_overlap:
                matches.update(atlases)
        return sorted(matches)

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
                    # Skip 1-hop paths (direct edges = no discovery value).
                    # Doing this here, before counting against
                    # max_paths_per_pair, prevents 1-hop candidates from
                    # consuming the per-pair budget that should go to
                    # multi-hop bridges.
                    if len(raw_path) < 3:
                        continue
                    if not self._path_intermediates_are_atoms(raw_path):
                        continue
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
            if nid in self._path_ignore_ids:
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
                    # Skip 1-hop paths (direct edges = no discovery value).
                    # Doing this here, before counting against
                    # max_paths_per_pair, prevents 1-hop candidates from
                    # consuming the per-pair budget that should go to
                    # multi-hop bridges.
                    if len(raw_path) < 3:
                        continue
                    if not self._path_intermediates_are_atoms(raw_path):
                        continue
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
                            "input_region": f"{name_a} - {name_b}",
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
            base_score = (
                (h.confidence_score ** weights["confidence"])
                * (h.evidence_score ** weights["evidence"])
                * (h.novelty_score ** weights["novelty"])
                * (max(h.testability_score, 0.01) ** weights["testability"])
            )
            h.composite_score = self._apply_feedback_adjustment(h, base_score)

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
        # Also strip vague umbrella hubs from the search subgraph so paths
        # never include them as intermediates. Endpoints are excluded from
        # the strip so a caller can still query them directly.
        intermediate_exclude |= (self._path_ignore_ids - {source_id, target_id})

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

        # Retired aliases must not reappear through the substring fallback.
        blocked_ids = blocked_canonical_ids(query)
        nodes = self._index.values()
        if blocked_ids:
            nodes = [node for node in nodes if node.id not in blocked_ids]

        for node in nodes:
            if node.preferred_name == query:
                return node.id

        query_lower = query.lower()
        for node in nodes:
            if node.preferred_name.lower() == query_lower:
                return node.id

        for node in nodes:
            for alias in node.aliases:
                if alias.lower() == query_lower:
                    return node.id

        if not allow_substring_fallback(query):
            return None

        candidates = []
        for node in nodes:
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

    def _sample_domain_nodes(
        self,
        domain: str,
        max_n: int,
        *,
        allowed_ids: frozenset[str] | None = None,
        random_seed: int | None = None,
        diversity_fraction: float = 0.0,
    ) -> list[str]:
        """Sample up to max_n non-claim nodes from a domain, preferring nodes with edges.

        (P2) Umbrella-source filter: drops imaging modalities, abstract
        processes, super-category nouns and pathway umbrellas before sorting.
        These look like real entities to the type system but don't
        constrain a DL experiment when used as the seed of a hypothesis.
        """
        all_nodes = [
            (nid, data) for nid, data in self.G.nodes(data=True)
            if domain in data.get("domain_tags", [])
            and "claim" not in data.get("domain_tags", [])
            and nid not in self._path_ignore_ids
            and (allowed_ids is None or nid in allowed_ids)
        ]
        nodes = []
        n_umbrella_dropped = 0
        for nid, data in all_nodes:
            name = data.get("preferred_name") or ""
            if self._is_umbrella_source(name):
                n_umbrella_dropped += 1
                continue
            nodes.append(nid)
        if n_umbrella_dropped:
            logger.info(
                "domain=%s seed pool: dropped %d umbrella sources, kept %d",
                domain, n_umbrella_dropped, len(nodes),
            )
        if not 0.0 <= float(diversity_fraction) <= 1.0:
            raise ValueError("diversity_fraction must be in [0, 1]")

        scope_support = (
            self._claim_endpoint_support_for_scope(self._task_required_claim_scope)
            if self._task_required_claim_scope is not None
            else {}
        )
        feedback_priority = (
            {node_id: self.feedback_state.node_priority(node_id) for node_id in nodes}
            if self.feedback_state is not None
            else {}
        )
        nodes.sort(
            key=lambda n: (
                -feedback_priority.get(n, 0.0),
                -scope_support.get(n, 0),
                -self._node_non_tree_degree(n),
                -self.G.degree(n),
                n,
            )
        )
        if max_n >= len(nodes):
            return nodes[:max_n]

        if random_seed is None or diversity_fraction <= 0.0:
            if not self._dynamic_generation_enabled:
                return nodes[:max_n]
            core_n = max_n
            random_n = 0
        else:
            core_n = max(
                1,
                min(max_n, int(math.ceil(max_n * (1.0 - diversity_fraction)))),
            )
            random_n = max_n - core_n

        if self._dynamic_generation_enabled and self._generation_exploration_round > 0:
            fixed_n = max(1, core_n // 2)
            selected = nodes[:fixed_n]
            rotating_pool = nodes[fixed_n:]
            rotating_n = min(core_n - fixed_n, len(rotating_pool))
            if rotating_n:
                start = (
                    self._generation_exploration_round * rotating_n
                ) % len(rotating_pool)
                selected.extend(
                    rotating_pool[(start + offset) % len(rotating_pool)]
                    for offset in range(rotating_n)
                )
        else:
            selected = nodes[:core_n]

        selected_set = set(selected)
        tail = [node_id for node_id in nodes if node_id not in selected_set]
        random_n = min(random_n, len(tail))

        def weighted_random_key(node_id: str) -> tuple[float, str]:
            digest = hashlib.sha256(
                f"{random_seed}\x1f{domain}\x1f{node_id}".encode("utf-8")
            ).digest()
            unit = (int.from_bytes(digest[:8], "big") + 0.5) / float(2**64)
            evidence_weight = 1.0 + math.log1p(
                scope_support.get(node_id, 0)
                + self._node_non_tree_degree(node_id)
            )
            return (-math.log(unit) / evidence_weight, node_id)

        if random_n:
            selected.extend(sorted(tail, key=weighted_random_key)[:random_n])
        return selected

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
            if claim_id:
                audit = self._claim_endpoint_audits.get(claim_id)
                if audit is not None and not audit.valid:
                    self._semantic_edge_rejections += 1
                    return []

            scoped_entries = self._claim_entries_for_edge(
                src_id, tgt_id, self._chain_required_claim_scope
            )
            scoped_meta = None
            if scoped_entries:
                claim_id, scoped_meta = max(
                    scoped_entries,
                    key=lambda item: (
                        float(item[1].get("confidence") or 0.0),
                        item[0],
                    ),
                )
                claim_node = self._index.get(claim_id)

            evidence = {}
            paper = {}
            raw_text = ""

            if scoped_meta is not None:
                meta = scoped_meta
                evidence = meta.get("evidence", {})
                paper = meta.get("source_paper", {})
                raw_text = meta.get("raw_text", "")
            elif claim_node and claim_node.metadata:
                meta = claim_node.metadata
                evidence = meta.get("evidence", {})
                paper = meta.get("source_paper", {})
                raw_text = meta.get("raw_text", "")

            from_name = src_node.preferred_name if src_node else src_id
            to_name = tgt_node.preferred_name if tgt_node else tgt_id
            relation_type = edge_data.get("relation_type", "unknown")
            confidence = edge_data.get("confidence", 0.5)
            if scoped_meta is not None:
                from_name = scoped_meta.get("subject_name") or from_name
                to_name = scoped_meta.get("object_name") or to_name
                relation_type = scoped_meta.get("predicate") or relation_type
                confidence = float(scoped_meta.get("confidence") or confidence)

            links.append(HypothesisLink(
                from_id=src_id,
                from_name=from_name,
                to_id=tgt_id,
                to_name=to_name,
                relation_type=relation_type,
                confidence=confidence,
                claim_id=claim_id,
                raw_text=raw_text,
                evidence=evidence,
                source_paper=paper,
            ))

        return links

    # ── scoring ────────────────────────────────────────────────────────

    def compute_frequency_boost(self, claim_meta: dict) -> float:
        """Frequency boost based on independent PRIMARY study replication.

        Prefers the merged `primary_supporting_papers` list set by
        `phase4_optimize.merge_duplicate_claims` (already filtered for
        non-review study types). Falls back to rebuilding from the
        pre-merge index, matching the same filter logic.
        """
        # Fast path: canonical claim carries primary-PMID list
        primary = claim_meta.get("primary_supporting_papers")
        if primary is not None and isinstance(primary, list):
            n = len(primary)
            if n >= 3:
                return 1.2
            elif n >= 1:
                return 1.0
            else:
                return 0.5

        # Fallback: scan all claims with the same SPO, filter reviews
        key = (
            claim_meta.get("subject_id", ""),
            claim_meta.get("predicate", ""),
            claim_meta.get("object_id", ""),
        )
        all_claims = self._claims_by_triple.get(key, [])
        primary_pmids = set()
        for c in all_claims:
            evidence = c.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}
            st = evidence.get("study_type", "")
            if not _is_review_study_type(st):
                source_paper = c.get("source_paper", {})
                if not isinstance(source_paper, dict):
                    source_paper = {}
                pmid = source_paper.get("pmid", "")
                if pmid:
                    primary_pmids.add(pmid)

        if len(primary_pmids) >= 3:
            return 1.2
        elif len(primary_pmids) >= 1:
            return 1.0
        else:
            return 0.5

    @staticmethod
    def compute_temporal_decay(claim_meta: dict, reference_year: int = 2026) -> float:
        """Temporal decay: newer primary studies get higher weight.

        Reviews get no time bonus (1.0). Primary studies decay 3% per year, floor 0.7.
        """
        evidence = claim_meta.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        st = evidence.get("study_type", "")
        if _is_review_study_type(st):
            return 1.0
        source_paper = claim_meta.get("source_paper", {})
        if not isinstance(source_paper, dict):
            source_paper = {}
        raw_year = source_paper.get("year", 0)
        if not raw_year:
            return 0.85  # unknown year, neutral
        try:
            if isinstance(raw_year, bool):
                raise ValueError("boolean is not a publication year")
            year = int(float(str(raw_year).strip()))
        except (TypeError, ValueError, OverflowError):
            return 0.85
        if year <= 0:
            return 0.85
        age = reference_year - year
        return max(0.7, 1.0 - 0.03 * age)

    def _compute_confidence_score(self, path: list[HypothesisLink]) -> float:
        """Confidence = geometric mean of per-link scores, with weak-link penalty.

        Per-link score = edge.confidence × freq_boost × temporal_decay
          (edge.confidence already includes study_type weighting from
          phase4_optimize.apply_evidence_weighting and the claim-level
          statistical quality signals from claim_extractor._estimate_confidence)

        Aggregate: geometric mean (one weak link crushes the path)
          + weakest-link penalty (×0.7 when min_edge < 0.1)

        Single source of truth for each multiplier:
        - study_type → phase4_optimize.WEIGHT_MAP (canonical, idempotent)
        - p_value/sample_size/replicability → claim_extractor._estimate_confidence
        - freq across primary PMIDs → compute_frequency_boost
        - publication recency → compute_temporal_decay
        """
        if not path:
            return 0.0

        import math

        scores = []
        min_conf = float("inf")
        for link in path:
            raw = max(link.confidence, 1e-3)  # tiny floor for log()
            min_conf = min(min_conf, raw)

            full_meta = {
                "evidence": link.evidence,
                "source_paper": link.source_paper,
                "subject_id": link.from_id,
                "predicate": link.relation_type,
                "object_id": link.to_id,
            }
            freq_boost = self.compute_frequency_boost(full_meta)
            temp_decay = self.compute_temporal_decay(full_meta)

            s = raw * freq_boost * temp_decay
            scores.append(min(s, 1.0))

        log_sum = sum(math.log(max(s, 1e-6)) for s in scores)
        gm = math.exp(log_sum / len(scores))

        if min_conf < 0.1:
            gm *= 0.7

        return max(min(gm, 1.0), 0.0)

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
        pmids = {
            l.source_paper.get("pmid", "")
            for l in path
            if isinstance(l.source_paper, dict) and l.source_paper.get("pmid")
        }
        if len(pmids) == 0:
            score += 0.10  # no paper support = speculative but novel
        elif len(pmids) == 1:
            score += 0.05  # single source = weak replication

        return min(score, 1.0)

    def _compute_evidence_score(self, path: list[HypothesisLink]) -> float:
        """Score evidence quality: traceability and text availability.

        DOES NOT use p_value/sample_size/effect_size — those signals already
        flow into edge.confidence via claim_extractor._estimate_confidence
        and are aggregated by _compute_confidence_score. Counting them again
        here was double-dipping.

        This score asks a different question: "How well-anchored is the
        evidence in source documents?" — which complements confidence's
        "How statistically strong is the evidence?". Path-level: most
        well-extracted edges score 0.6-0.8; we reserve >0.9 for paths whose
        every step has rich provenance.
        """
        scores = []
        for link in path:
            evidence = link.evidence if isinstance(link.evidence, dict) else {}
            source_paper = link.source_paper if isinstance(link.source_paper, dict) else {}
            study_type = evidence.get("study_type", "")
            s = 0.2 if _is_review_study_type(study_type) else 0.3

            if link.raw_text and len(link.raw_text) > 20:
                s += 0.20
            if link.claim_id:
                s += 0.15
            if source_paper.get("pmid"):
                s += 0.15
            if evidence.get("study_type"):
                s += 0.10

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
        """Weighted geometric mean of the 4 score components.

        Geometric: a hypothesis is only as good as its weakest dimension.
        A path with great evidence but 0 testability is worthless to us.

        Matches the linear fitness in evolution_engine._score_fitness
        (same weights, different aggregation — fitness adds convergence /
        diversity / length modifiers not relevant here).
        """
        c = max(h.confidence_score, 0.01)
        e = max(h.evidence_score, 0.01)
        n = max(h.novelty_score, 0.01)
        t = max(h.testability_score, 0.01)
        score = (c ** 0.20) * (e ** 0.20) * (n ** 0.25) * (t ** 0.35)

        if self._has_only_review_evidence(h):
            score *= 0.7

        return self._apply_feedback_adjustment(h, score)

    def _apply_feedback_adjustment(self, h: Hypothesis, base_score: float) -> float:
        """Apply loaded closed-loop feedback and store an auditable adjustment."""
        metadata = h.metadata or {}
        if self.feedback_state is None:
            metadata.pop("feedback_adjustment", None)
            h.metadata = metadata
            return float(min(1.0, max(0.0, base_score)))
        adjustment = self.feedback_state.score(h)
        adjusted = self.feedback_state.apply(base_score, adjustment)
        metadata["feedback_adjustment"] = {
            **adjustment.as_dict(),
            "base_score": float(base_score),
            "adjusted_score": float(adjusted),
        }
        h.metadata = metadata
        return adjusted

    @staticmethod
    def _has_only_review_evidence(h: Hypothesis) -> bool:
        """True if every link in the path comes from a review/narrative_review."""
        if not h.path:
            return False
        for link in h.path:
            evidence = link.evidence if isinstance(link.evidence, dict) else {}
            study_type = evidence.get("study_type", "")
            if study_type and not _is_review_study_type(study_type):
                return False
        return True

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
            e1 = m1.get("evidence", {})
            e2 = m2.get("evidence", {})
            if not isinstance(e1, dict):
                e1 = {}
            if not isinstance(e2, dict):
                e2 = {}
            d1 = e1.get("direction", "")
            d2 = e2.get("direction", "")
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

        pmids = {
            l.source_paper.get("pmid", "")
            for l in h.path
            if isinstance(l.source_paper, dict) and l.source_paper.get("pmid")
        }
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


# Updated: 2026-08-12 02:28 HKT
