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
        # chain atom anchor (e.g. CS3 forces IMAGING_MARKER to IM:*-prefix
        # atoms only, so the chain truly traverses the marker layer instead
        # of falling back to raw neuroanatomy CUIs that share the domain tag).
        # Filter signature: (node_id, ConceptNode) -> bool. None / missing
        # entry = no extra filter beyond ATOM_TO_DOMAINS membership.
        self._chain_atom_filters: dict = {}
        # Build claims index for frequency_boost: (subj, pred, obj) → [claim_meta, ...]
        self._claims_by_triple: dict[tuple[str, str, str], list[dict]] = {}
        for nid, node in self._index.items():
            if "claim" not in node.domain_tags:
                continue
            meta = node.metadata
            key = (meta.get("subject_id", ""), meta.get("predicate", ""), meta.get("object_id", ""))
            if key[0] and key[2]:
                self._claims_by_triple.setdefault(key, []).append(meta)
        # Lazy evidence-degree cache for the min_evidence_per_node walk filter.
        self._non_tree_degree: Optional[dict[str, int]] = None

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
        """
        if domain_pairs is None:
            domain_pairs = DEFAULT_DOMAIN_PAIRS

        all_hypotheses: list[Hypothesis] = []
        seen_pairs: set[tuple[str, str]] = set()
        _hyp_counter = 0

        for dom_a, dom_b in domain_pairs:
            logger.info(f"generating hypotheses: {dom_a} -> {dom_b}")

            seeds_a = self._sample_domain_nodes(dom_a, max_seeds_per_domain)
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

                pair_count = 0
                for target_id in candidates:
                    pair_key = tuple(sorted([seed_id, target_id]))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # Enumerate up to N simple paths between this (seed, target).
                    # Cap the enumerator so a high-degree pair cannot blow up
                    # runtime: take 4x the per-pair quota then prune.
                    raw_paths: list[list[str]] = []
                    enum_cap = max(max_paths_per_pair * 4, 8)
                    try:
                        for p in nx.all_simple_paths(
                            self.G, seed_id, target_id, cutoff=max_hops
                        ):
                            if len(p) - 1 < min_hops:
                                continue
                            if not self._path_intermediates_are_atoms(p):
                                continue
                            if (metapath_min_domains > 1 and
                                self._path_distinct_atom_domains(p) < metapath_min_domains):
                                continue
                            if not self._path_meets_evidence_floor(p, min_evidence_per_node):
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
                                    sp, min_evidence_per_node)):
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
            * ``require_atom_touch=True`` enforces the stricter constraint
              that each path must visit at least one node from EVERY input
              atom (source/target included). On sparse KGs this filter is
              aggressive and may drop most candidates; off by default.

        Args:
            task: A :class:`neurooracle.src.atoms.Task` — typically one
                  pulled from ``CANONICAL_TASKS``.
            max_hops, max_paths_per_pair, max_seeds_per_domain: forwarded
                  to :meth:`batch_generate`.
            require_atom_touch: enable strict multi-atom path constraint.

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
        for in_dom in input_domains:
            for out_dom in output_domains:
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
        )

        # tag with task provenance BEFORE post_process so the task-aware
        # _is_dataset_outcome filter can read task_name.
        for h in hyps:
            h.metadata = dict(h.metadata or {})
            h.metadata["task_name"] = task.name
            h.metadata["task_signature"] = task.signature
            h.metadata["task_modifier"] = task.modifier.value
            h.metadata["task_kind"] = "task"

        # Now apply post_process with task tags in place.
        before = len(hyps)
        hyps = self.post_process(hyps)
        logger.info(
            f"task '{task.name}': {before} raw -> {len(hyps)} after task-aware post_process"
        )

        # optional strict multi-atom touch filter
        if require_atom_touch and len(task.inputs) > 1:
            kept = [h for h in hyps if self._path_touches_atoms(h, task.inputs)]
            logger.info(
                f"require_atom_touch: kept {len(kept)}/{len(hyps)} "
                f"hypotheses for task '{task.name}'"
            )
            hyps = kept

        return hyps

    def _path_touches_atoms(self, h: Hypothesis, atoms) -> bool:
        """Check whether a hypothesis path visits ≥1 node from every atom's
        domain pool. Used by ``require_atom_touch`` filtering."""
        from .atoms import ATOM_TO_DOMAINS

        needed = {a: ATOM_TO_DOMAINS[a] for a in atoms}
        visited: set = set()

        node_ids: list[str] = []
        if h.source_id: node_ids.append(h.source_id)
        if h.target_id: node_ids.append(h.target_id)
        for link in (h.path or []):
            if getattr(link, "from_id", None):
                node_ids.append(link.from_id)
            if getattr(link, "to_id", None):
                node_ids.append(link.to_id)

        for nid in node_ids:
            node = self._index.get(nid)
            if node is None:
                continue
            node_domains = set(node.domain_tags or [])
            for atom, atom_doms in needed.items():
                if atom in visited:
                    continue
                if atom_doms & node_domains:
                    visited.add(atom)
            if len(visited) == len(needed):
                return True
        return len(visited) == len(needed)

    # ── chain-aware generation (TaskChain mediation paths) ────────────────

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

        atoms = chain.chain
        # Per-atom domain pools, with claim/PATH_IGNORE nodes excluded.
        atom_node_pools: list[set[str]] = []
        for atom in atoms:
            doms = ATOM_TO_DOMAINS[atom]
            extra_filter = self._chain_atom_filters.get(atom)
            pool = {
                nid for nid, data in self.G.nodes(data=True)
                if (set(data.get("domain_tags", [])) & doms)
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
                kept_for_this_partial = 0
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
                    extended.append((new_path, new_anchors))
                    kept_for_this_partial += 1
                    if kept_for_this_partial >= max_paths_per_segment:
                        break

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
        survivors = frontier[:max_chains]
        logger.info(
            f"chain '{chain.name}' [{chain.signature}]: "
            f"{len(survivors)} mediation path(s) from {len(seeds)} seed(s)"
        )

        hyps: list[Hypothesis] = []
        _chain_counter = 0
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

            _chain_counter += 1
            h = Hypothesis(
                id=f"HYP:CHAIN:{chain.name}:{_chain_counter:06d}",
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
                    continue

            # filter 1-hop (single direct edge = no discovery value)
            if len(h.path) < min_hops:
                continue

            # filter all-vague-relations
            if filter_vague_relations:
                relation_types = {l.relation_type for l in h.path}
                if relation_types and relation_types <= VAGUE_RELATIONS:
                    continue

            # filter single-PMID bridges (all hops cite the same paper = not a real bridge)
            if len(h.path) >= 2:
                pmids = set()
                for link in h.path:
                    pmid = link.source_paper.get("pmid", "") if isinstance(link.source_paper, dict) else ""
                    if pmid:
                        pmids.add(pmid)
                if len(pmids) == 1:
                    continue

            # filter non-measurable biomarkers (not testable from imaging)
            if filter_non_measurable:
                if self._has_non_measurable_entity(h):
                    continue

            # filter biologically implausible paths (brain region → non-neurological target)
            if self._has_implausible_path(h):
                continue

            # filter paths with weak evidence (target not mentioned in raw_text)
            if self._has_weak_evidence(h):
                continue

            # filter paths where both ends of any edge are broad hubs
            # ("Brain Diseases --causes--> Cognitive Dysfunction" is uninformative)
            if self._has_hub_to_hub_edge(h):
                continue

            # filter paths touching any vague COGAT/MeSH umbrella hub
            # (memory/logic/loss/activation/risk/stress/Brain/Neurons).
            # These nodes are too abstract to drive a DL experiment whether
            # they appear as source, target, or intermediate.
            if self._touches_path_ignore_node(h):
                continue

            # filter paths that transit through disease mega-hubs as
            # intermediate nodes (A → Disease → B is uninformative).
            # These nodes are still valid as source/target endpoints.
            if self._transits_intermediate_only_hub(h):
                continue

            # (C-1) filter paths whose INTERMEDIATE node is a generic
            # phrase ("neural activity", "disease progression", "grey
            # matter", ...). Endpoints are not checked here.
            if self._has_intermediate_generic_phrase(h):
                continue

            # (C-2) filter paths whose directional density is too thin
            # (3+ hops with < 50% directional relations = too vague to
            # be a mechanism hypothesis).
            if self._has_thin_directional_density(h):
                continue

            # filter: target must be a dataset outcome (diagnosis/cognition/behavior/
            # personality/motor). Predicting "White Matter" or "Neurons" is not a
            # hypothesis UKB/ADNI/HCP can directly test — those are imaging features
            # used as INPUTS, not outcomes.
            if not self._is_dataset_outcome(h):
                continue

            # (C-3) filter: target name is an umbrella concept ("skill",
            # "disease", "neurological disorder", "clinical features")
            # even though it passes the outcome keyword check. These
            # can't anchor a concrete DL label.
            if self._is_too_broad_target(h.target_name):
                continue

            # (P2) filter: source is an umbrella concept (imaging modality,
            # super-category, abstract process). batch_generate seeds via
            # _sample_domain_nodes which already drops these, but other
            # entry points (task pipelines, manual paths) skip that filter
            # so we mirror the gate at post_process for defence-in-depth.
            if self._is_umbrella_source(h.source_name):
                continue

            # filter paths with no directional predicates (pure association chains)
            if len(h.path) >= 2:
                relation_types = {l.relation_type for l in h.path}
                if not (relation_types & DIRECTIONAL_RELATIONS):
                    continue

            # filter paths that exceed max hop length (noise accumulation)
            if len(h.path) > max_hops_filter:
                continue

            filtered.append(h)

        # Deduplicate: for each (source, target) pair, keep top 2 by composite score
        from collections import defaultdict
        pair_groups = defaultdict(list)
        for h in filtered:
            key = (h.source_id, h.target_id)
            pair_groups[key].append(h)

        deduplicated = []
        for key, group in pair_groups.items():
            # Sort by composite score descending
            group.sort(key=lambda x: x.composite_score, reverse=True)
            # Keep top 2 (or 1 if only one exists)
            deduplicated.extend(group[:2])

        logger.info(f"post_process: {before} -> {len(filtered)} filtered -> {len(deduplicated)} deduplicated "
                     f"(removed {before - len(deduplicated)} total)")
        return deduplicated

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
            raw = link.raw_text or link.evidence.get("raw_text", "") if isinstance(link.evidence, dict) else ""
            if raw:
                raw_lower = raw.lower()
                # If any target term appears in raw_text, evidence is OK
                if any(term in raw_lower for term in target_terms):
                    return False

        # No hop mentions the target region → weak evidence
        logger.debug(f"weak evidence: {h.id} target '{h.target_name}' not mentioned in any raw_text")
        return True

    # ── imaging-driven batch generation ──────────────────────────────

    def find_transdiagnostic_clusters(
        self,
        min_diseases_per_cluster: int = 3,
        max_clusters: int = 20,
        modality_partitioned: bool = True,
        min_abs_d: float = 0.05,
        max_fdr_p: float = 0.05,
    ) -> list[Hypothesis]:
        """CS2 generator — mine ENIGMA case-control edges into transdiagnostic
        (region, modality, sign) clusters that span ≥ ``min_diseases_per_cluster``
        diseases.

        Each ENIGMA edge in the v2 KG carries metadata
        ``{cohens_d, fdr_p, modality, comparison, n_controls, n_patients,
        hemisphere_kept}``. We bucket edges by ``(region_id, modality, sign(d))``
        and emit a cluster Hypothesis whenever the bucket pools effects from at
        least ``min_diseases_per_cluster`` distinct comparisons that pass the
        ``min_abs_d`` / ``max_fdr_p`` significance gates.

        The hypothesis is encoded as a star (not a sequential chain): ``path``
        carries one ``HypothesisLink`` per disease→region edge, which is enough
        for the critic / novelty stages to read the supporting evidence even
        though the links don't form a connected chain. ``hypothesis_type`` is
        ``"transdiagnostic_cluster"`` so post_process can route around the
        chain-style filters.
        """
        from collections import defaultdict

        bucket: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        for u, v, d in self.G.edges(data=True):
            m = d.get("metadata") or {}
            cohens_d = m.get("cohens_d")
            modality = m.get("modality")
            fdr_p = m.get("fdr_p")
            if cohens_d is None or modality is None or fdr_p is None:
                continue
            try:
                cd = float(cohens_d)
                fp = float(fdr_p)
            except (TypeError, ValueError):
                continue
            if abs(cd) < min_abs_d or fp > max_fdr_p:
                continue
            sign = 1 if cd > 0 else -1
            mod_key = modality if modality_partitioned else "_any"
            bucket[(v, mod_key, sign)].append({
                "disease_id": u,
                "region_id": v,
                "cohens_d": cd,
                "fdr_p": fp,
                "comparison": m.get("comparison", ""),
                "rel": d.get("relation_type", "unknown"),
                "modality": modality,
                "hemi": m.get("hemisphere_kept", ""),
                "n_controls": m.get("n_controls"),
                "n_patients": m.get("n_patients"),
            })

        clusters: list[Hypothesis] = []
        for (region_id, mod_key, sign), entries in bucket.items():
            distinct_diseases = {(e["disease_id"], e["comparison"]) for e in entries}
            if len(distinct_diseases) < min_diseases_per_cluster:
                continue

            region_node = self._index.get(region_id)
            region_name = region_node.preferred_name if region_node else region_id
            modality = entries[0]["modality"] if mod_key == "_any" else mod_key

            entries.sort(key=lambda e: abs(e["cohens_d"]), reverse=True)
            kept_first: dict[str, dict] = {}
            for e in entries:
                key = e["comparison"] or e["disease_id"]
                if key not in kept_first:
                    kept_first[key] = e
            kept = list(kept_first.values())

            links: list[HypothesisLink] = []
            for e in kept:
                disease = self._index.get(e["disease_id"])
                disease_name = disease.preferred_name if disease else e["disease_id"]
                links.append(HypothesisLink(
                    from_id=e["disease_id"],
                    from_name=disease_name,
                    to_id=region_id,
                    to_name=region_name,
                    relation_type=e["rel"],
                    confidence=min(1.0, abs(e["cohens_d"])),
                    evidence={
                        "cohens_d": e["cohens_d"],
                        "fdr_p": e["fdr_p"],
                        "modality": e["modality"],
                        "comparison": e["comparison"],
                        "hemisphere": e["hemi"],
                        "n_controls": e["n_controls"],
                        "n_patients": e["n_patients"],
                    },
                ))

            anchor_disease = links[0].from_id
            anchor_name = links[0].from_name
            mean_abs_d = sum(abs(l.confidence) for l in links) / max(1, len(links))
            sign_label = "increase" if sign > 0 else "decrease"

            disease_listing = ", ".join(sorted({l.from_name for l in links}))
            explanation = (
                f"Transdiagnostic {sign_label} of {region_name} {modality} "
                f"shared across {len(kept)} disorders: {disease_listing}. "
                f"Mean |Cohen's d| = {mean_abs_d:.2f}."
            )

            h = Hypothesis(
                id=f"cluster_{region_id}_{mod_key}_{sign_label}",
                hypothesis_type="transdiagnostic_cluster",
                source_id=anchor_disease,
                source_name=f"Transdiagnostic cluster ({len(kept)} disorders)",
                target_id=region_id,
                target_name=f"{region_name} ({modality}, {sign_label})",
                path=links,
                confidence_score=min(1.0, mean_abs_d),
                evidence_score=min(1.0, math.log1p(len(kept)) / math.log(12)),
                novelty_score=0.5,
                testability_score=0.8,
                supporting_claims=[],
                explanation=explanation,
                testability_reason=(
                    f"ENIGMA case-control effects on {region_name} ({modality}) "
                    f"are directly testable on UKB / ENIGMA-cohort imaging."
                ),
                metadata={
                    "case_study": "cs2_transdiagnostic",
                    "cluster_region_id": region_id,
                    "cluster_region_name": region_name,
                    "cluster_modality": modality,
                    "cluster_sign": sign_label,
                    "diseases": sorted({l.from_name for l in links}),
                    "disease_ids": sorted({l.from_id for l in links}),
                    "n_diseases": len(kept),
                    "mean_abs_d": mean_abs_d,
                    "max_abs_d": max(abs(l.confidence) for l in links),
                },
            )
            h.composite_score = (
                0.5 * h.evidence_score
                + 0.3 * h.confidence_score
                + 0.2 * h.testability_score
            )
            clusters.append(h)

        clusters.sort(
            key=lambda c: (c.metadata["n_diseases"], c.metadata["mean_abs_d"]),
            reverse=True,
        )
        if max_clusters and max_clusters > 0:
            clusters = clusters[:max_clusters]

        logger.info(
            f"transdiagnostic clustering: {len(bucket)} (region,modality,sign) "
            f"buckets -> {len(clusters)} clusters spanning >= "
            f"{min_diseases_per_cluster} diseases"
        )
        return clusters

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
            st = c.get("evidence", {}).get("study_type", "")
            if st not in _REVIEW_TYPES:
                pmid = c.get("source_paper", {}).get("pmid", "")
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
        st = claim_meta.get("evidence", {}).get("study_type", "")
        if st in _REVIEW_TYPES:
            return 1.0
        year = claim_meta.get("source_paper", {}).get("year", 0)
        if not year:
            return 0.85  # unknown year, neutral
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
        pmids = {l.source_paper.get("pmid", "") for l in path if l.source_paper.get("pmid")}
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
        _REVIEW_TYPES = {"narrative_review", "review"}
        scores = []
        for link in path:
            study_type = (link.evidence.get("study_type") or "").lower()
            s = 0.2 if study_type in _REVIEW_TYPES else 0.3

            if link.raw_text and len(link.raw_text) > 20:
                s += 0.20
            if link.claim_id:
                s += 0.15
            if link.source_paper.get("pmid"):
                s += 0.15
            if link.evidence.get("study_type"):
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

        return score

    @staticmethod
    def _has_only_review_evidence(h: Hypothesis) -> bool:
        """True if every link in the path comes from a review/narrative_review."""
        _REVIEW_TYPES = {"narrative_review", "review"}
        if not h.path:
            return False
        for link in h.path:
            study_type = (link.evidence.get("study_type") or "").lower()
            if study_type and study_type not in _REVIEW_TYPES:
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
