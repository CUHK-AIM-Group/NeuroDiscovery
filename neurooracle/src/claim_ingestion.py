"""Claim ingestion: resolve entities, refine predicates, add claims to knowledge graph."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .atoms import Atom, DOMAIN_TO_ATOMS
from .claim_semantics import (
    _roles_compatible, concept_atom_roles, declared_type_atoms,
    name_compatibility_score,
)
from .case_study_membership_contract import validate_final_scope_reaudit
from .claim_evidence_identity import evidence_dedup_key, evidence_signature
from .claim_extractor import ClaimExtractor, ExtractionResult
from .graph_manager import KnowledgeGraph
from .schema import CLAIM_PREDICATES, Claim, ConceptNode, DomainTag, Edge, PaperRef

logger = logging.getLogger(__name__)

CLAIM_REJECTION_AUDIT_FILENAME = "claim_ingestion_rejections.jsonl"

# Loading a million-line canonical store before every 100-paper batch would
# dominate ingestion time.  Keep per-process indexes keyed by the absolute
# output path; all supported extraction drivers serialize persistence calls.
_JSONL_FIELD_CACHE: dict[tuple[Path, str], set[str]] = {}

# ── Build-time noise filter ────────────────────────────────────────────
# Gates the step-6 "mint new CLM_CONCEPT" fallback in resolve_entity.
# Curated-vocab matches (MSH/NN/COGAT/...) happen in steps 1-5 and are
# unaffected. Only names that would otherwise be auto-minted face the check.

_NOISE_PREFIXES = (
    "impaired ", "increased ", "decreased ", "reduced ",
    "altered ", "elevated ", "abnormal ", "deficient ",
    "excessive ", "diminished ", "enhanced ", "disrupted ",
    "lower ", "higher ", "greater ", "lesser ",
)
_NOISE_SUFFIXES = (
    " findings", " levels", " changes", " symptoms",
    " manifestations", " status", " outcomes", " profile",
    " profiles", " patterns", " features",
)
# Trailing "X of Y" / "X in Y" tails — captures both halves for salvage
_SALVAGE_SPLIT_RE = re.compile(
    r"^(.+?)\s+(?:of|in|for|during|with)\s+(.+)$", re.I
)

# Toggled by ingest_claims based on its keep_noise parameter
_NOISE_FILTER_ENABLED: bool = True

# When True, resolve_entity will NEVER mint new CLM_CONCEPT nodes.
# Unresolved subjects/objects are dropped (the claim is skipped entirely if
# either endpoint fails to map). Phase 1 already covers most medical terms
# via NeuroNames/MeSH/DisGeNET/CognitiveAtlas + UMLS alignment, so new nodes
# from Phase 2 are almost always LLM variants of existing concepts or low-
# quality noise. Toggled by ingest_claims(strict_phase1=True).
_STRICT_PHASE1: bool = False

_DROP_LOG_DEFAULT_PATH = Path("neurooracle/data/build_artifacts/dropped_entities.jsonl")


def _is_noisy_name(name: str) -> bool:
    """Match the Web UI 'clean' rules exactly.

    Uses HypothesisEngine._is_noisy_entity (lazy import to avoid circular
    dependency with hypothesis_engine.py). Also applies prefix/suffix
    heuristics that catch LLM-extracted chaff like "MRI findings".
    """
    if not name:
        return False
    # Lazy import — hypothesis_engine imports from this module indirectly
    from .hypothesis_engine import HypothesisEngine
    if HypothesisEngine._is_noisy_entity(name):
        return True
    lname = name.lower()
    return (any(lname.startswith(p) for p in _NOISE_PREFIXES)
            or any(lname.endswith(s) for s in _NOISE_SUFFIXES))


def _noise_reasons(name: str) -> list[str]:
    """Human-readable reasons for audit log."""
    reasons: list[str] = []
    if not name:
        return ["empty name"]
    from .hypothesis_engine import HypothesisEngine
    if HypothesisEngine._is_noisy_entity(name):
        reasons.append("generic/nominalized token")
    lname = name.lower()
    for p in _NOISE_PREFIXES:
        if lname.startswith(p):
            reasons.append(f"prefix '{p.strip()}'")
            break
    for s in _NOISE_SUFFIXES:
        if lname.endswith(s):
            reasons.append(f"suffix '{s.strip()}'")
            break
    return reasons


def _salvage_noisy_name(name: str) -> str:
    """Strip noise affixes and return a cleaner candidate.

    Strategy:
    1. "X of|in|for|... Y" → prefer the non-noisy half. If both halves are
       non-noise, favor the right (usually the semantic object).
    2. Iteratively strip noise prefixes/suffixes until stable.

    Returns "" if no salvage produces a non-empty change.
    """
    cleaned = name.strip()
    if not cleaned:
        return ""

    # Step 1: split "X of/in/for Y" — pick the non-noisy half
    m = _SALVAGE_SPLIT_RE.match(cleaned)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        # Local noise check (don't recurse into hypothesis_engine for every split)
        def _quick_noise(s: str) -> bool:
            if not s:
                return True
            ls = s.lower()
            return (any(ls.startswith(p) for p in _NOISE_PREFIXES)
                    or any(ls.endswith(sfx) for sfx in _NOISE_SUFFIXES))
        l_noise = _quick_noise(left)
        r_noise = _quick_noise(right)
        if l_noise and not r_noise:
            cleaned = right
        elif r_noise and not l_noise:
            cleaned = left
        elif not l_noise and not r_noise:
            # Both clean — right half is usually the semantic object
            cleaned = right
        # if both noisy-looking, fall through — maybe affix strip helps

    # Step 2: iteratively strip prefixes/suffixes until stable
    for _ in range(4):  # bounded, avoid pathological loops
        before = cleaned
        lname = cleaned.lower()
        for p in _NOISE_PREFIXES:
            if lname.startswith(p):
                cleaned = cleaned[len(p):].strip()
                break
        lname = cleaned.lower()
        for s in _NOISE_SUFFIXES:
            if lname.endswith(s):
                cleaned = cleaned[:-len(s)].strip()
                break
        if cleaned == before:
            break

    # Step 3: pop trailing single-word noise tokens (e.g., "cognitive functions"
    # → "cognitive"; "adverse events" → "adverse"). Uses the hypothesis_engine
    # NOISE_WORDS list for consistency.
    try:
        from .hypothesis_engine import _NOISE_WORDS as _HE_NOISE_WORDS
        tokens = cleaned.split()
        while len(tokens) > 1 and tokens[-1].lower().strip(".,") in _HE_NOISE_WORDS:
            tokens.pop()
        while len(tokens) > 1 and tokens[0].lower().strip(".,") in _HE_NOISE_WORDS:
            tokens.pop(0)
        cleaned = " ".join(tokens).strip()
    except Exception:
        pass

    if cleaned and cleaned.lower() != name.strip().lower():
        return cleaned
    return ""


# ── Vague endpoint pre-filter ─────────────────────────────────────────
# Internalizes judge_clm_endpoints.py logic: reject names that are too
# vague to serve as hypothesis endpoints BEFORE minting a CLM_CONCEPT node.

_VAGUE_ENDPOINT_EXACT = frozenset({
    "focus", "integration", "balance", "knowledge", "autonomy",
    "performance", "adaptation", "resilience", "vulnerability",
    "recovery", "progression", "mechanism", "process", "outcome",
    "outcomes", "survival", "improvement", "response", "effect",
    "effects", "impact", "factor", "factors", "role", "function",
    "functions", "activity", "condition", "conditions", "treatment",
    "intervention", "approach", "strategy", "method",
})

_VAGUE_ENDPOINT_GENERIC_NOUNS = frozenset({
    "ability", "abilities", "abnormality", "abnormalities",
    "alteration", "alterations", "characteristic", "characteristics",
    "complication", "complications", "consequence", "consequences",
    "decline", "deficit", "deficits", "deterioration", "disability",
    "disturbance", "disturbances", "dysfunction", "feature", "features",
    "impairment", "impairments", "manifestation", "manifestations",
    "mechanism", "mechanisms", "outcome", "outcomes", "process",
    "processes", "relationship", "relationships", "subgroup", "subgroups",
})

_VAGUE_ENDPOINT_GENERIC_MODIFIERS = frozenset({
    "acute", "aggressive", "behavioral", "brain", "cerebral", "chronic",
    "clinical", "cognitive", "common", "cortical", "emotional", "functional",
    "general", "global", "intact", "long", "term", "motor", "neural",
    "neurological", "neurocognitive", "overall", "personal", "physiological",
    "psychiatric", "psychological", "sensory", "short", "significant",
    "social", "specific", "structural", "subjective", "verbal", "visual",
})

_VAGUE_ENDPOINT_STOPWORDS = frozenset({
    "a", "an", "and", "by", "for", "in", "of", "or", "the", "to", "with",
})

_VAGUE_ENDPOINT_PATTERNS_RE = re.compile(
    r"^(motor|cognitive|neurocognitive|functional|social|verbal|visual|"
    r"sensory|emotional|behavioral|clinical|neurological|psychiatric|"
    r"psychological|physiological|structural|significant|long-term|"
    r"short-term|acute|chronic|general|overall|common|specific|intact|"
    r"aggressive|personal)\s+"
    r"(?:\w+\s+)?"
    r"(deficit|deficits|impairment|impairments|dysfunction|disability|"
    r"decline|deterioration|disturbance|disturbances|abnormality|"
    r"abnormalities|alteration|alterations|features|abilities|"
    r"relationships|outcomes|subgroup|subgroups|mechanism|processes)$",
    re.I,
)


def _has_only_generic_endpoint_context(sl: str) -> bool:
    """Return True when a generic endpoint noun has no specific anchor.

    "deficit" and "cognitive deficit" stay blocked, but concrete phenotypes
    such as "verbal episodic memory deficit" or "24-month MMSE decline" are
    useful endpoints and should survive Phase-2 concept minting.
    """
    words = re.findall(r"[a-z0-9]+", sl)
    if not words or words[-1] not in _VAGUE_ENDPOINT_GENERIC_NOUNS:
        return False

    context = [
        w for w in words[:-1]
        if w not in _VAGUE_ENDPOINT_STOPWORDS and not w.isdigit()
    ]
    if not context:
        return True
    return all(w in _VAGUE_ENDPOINT_GENERIC_MODIFIERS for w in context)


def _is_vague_endpoint_name(name: str) -> bool:
    """Return True if name is too vague to be a useful hypothesis endpoint.

    Catches patterns like:
    - Single generic words: "balance", "focus", "outcome"
    - "adjective + generic noun": "cognitive deficits", "clinical features"
    - Names ending in vague suffixes: "aggressive subgroup"
    - Very short names (< 3 chars, likely parsing artifacts)
    """
    if not name:
        return True
    s = name.strip()
    if len(s) < 2:
        return True
    sl = s.lower()

    # Single-word exact match
    if sl in _VAGUE_ENDPOINT_EXACT:
        return True

    # Generic endpoint noun with only broad modifiers.
    if _has_only_generic_endpoint_context(sl):
        return True

    # Pattern match (adjective + generic noun)
    if _VAGUE_ENDPOINT_PATTERNS_RE.match(sl) and _has_only_generic_endpoint_context(sl):
        return True

    return False


class _DropLog:
    """Append-only audit log of salvaged / dropped entity names."""

    def __init__(self) -> None:
        self.fp = None
        self.n_dropped: int = 0
        self.n_salvaged: int = 0
        self.path: Optional[Path] = None

    def open(self, path: Path) -> None:
        if self.fp is not None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = path.open("a", encoding="utf-8")
        self.path = path
        # marker so successive runs are distinguishable when grepping
        self.fp.write(json.dumps({
            "ts": datetime.utcnow().isoformat(), "kind": "run_start",
        }, ensure_ascii=False) + "\n")

    def record(
        self,
        raw_name: str,
        kind: str,
        cleaned: str = "",
        matched_id: Optional[str] = None,
        reasons: Optional[list[str]] = None,
    ) -> None:
        if kind == "salvaged":
            self.n_salvaged += 1
        else:
            self.n_dropped += 1
        if self.fp is None:
            return
        self.fp.write(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "raw_name": raw_name,
            "kind": kind,
            "cleaned": cleaned,
            "matched_id": matched_id,
            "reasons": reasons or [],
        }, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.fp is not None:
            try:
                self.fp.flush()
                self.fp.close()
            finally:
                self.fp = None


_DROP_LOG = _DropLog()


# ── RELATE: predicate refinement ──────────────────────────────────────

_VAGUE_PREDICATES = {"is_associated_with", "correlates_with"}

# Rule-based keyword patterns for refining is_associated_with
_PREDICATE_KEYWORDS: dict[str, list[re.Pattern]] = {
    "is_risk_factor_for": [
        re.compile(r"\brisk\s+factor\b", re.I),
        re.compile(r"\bincreases?\s+(?:the\s+)?risk\b", re.I),
        re.compile(r"\bassociated\s+with\s+(?:increased|higher)\s+risk\b", re.I),
        re.compile(r"\bpredispos\w*\b", re.I),
    ],
    "is_biomarker_of": [
        re.compile(r"\bbiomarker\b", re.I),
        re.compile(r"\bdiagnostic\b", re.I),
        re.compile(r"\bpredicts?\s+(?:diagnosis|progression|conversion)\b", re.I),
        re.compile(r"\bsensitivity\s+and\s+specificity\b", re.I),
    ],
    "causes": [
        re.compile(r"\bcauses?\b", re.I),
        re.compile(r"\binduces?\b", re.I),
        re.compile(r"\bleads?\s+to\b", re.I),
        re.compile(r"\bpathogen\w*\b", re.I),
    ],
    "predicts": [
        re.compile(r"\bpredicts?\b", re.I),
        re.compile(r"\bprognostic\b", re.I),
        re.compile(r"\bforecasts?\b", re.I),
    ],
    "treats": [
        re.compile(r"\btreats?\b", re.I),
        re.compile(r"\btherapeutic\b", re.I),
        re.compile(r"\bintervention\b", re.I),
        re.compile(r"\badministered\b", re.I),
    ],
    "inhibits": [
        re.compile(r"\binhibits?\b", re.I),
        re.compile(r"\bsuppress\w*\b", re.I),
        re.compile(r"\bblocks?\b", re.I),
        re.compile(r"\bantagonist\b", re.I),
    ],
    "activates": [
        re.compile(r"\bactivat\w*\b", re.I),
        re.compile(r"\benhances?\b", re.I),
        re.compile(r"\bstimulat\w*\b", re.I),
        re.compile(r"\bagonist\b", re.I),
    ],
    "increases": [
        re.compile(r"\bincreases?\b", re.I),
        re.compile(r"\belevated\b", re.I),
        re.compile(r"\bhigher\s+(?:levels?|concentrations?|expression)\b", re.I),
        re.compile(r"\bup-?regulat\w*\b", re.I),
    ],
    "reduces": [
        re.compile(r"\breduces?\b", re.I),
        re.compile(r"\bdecreases?\b", re.I),
        re.compile(r"\blower\b", re.I),
        re.compile(r"\bdown-?regulat\w*\b", re.I),
    ],
    "modulates": [
        re.compile(r"\bmodulat\w*\b", re.I),
        re.compile(r"\bregulat\w*\b", re.I),
        re.compile(r"\binfluences?\b", re.I),
    ],
}

# Penalty factor applied to edge confidence when raw_text doesn't support
# the assigned predicate. Keeps precise predicates but marks unsupported
# ones as low-confidence, reducing their influence in hypothesis scoring.
_UNSUPPORTED_PREDICATE_PENALTY = 0.5
_BACKGROUND_CLAIM_PENALTY = 0.5
_MIN_INGEST_CLAIM_CONFIDENCE = float(
    os.environ.get("NEUROORACLE_MIN_INGEST_CLAIM_CONFIDENCE", "0.30")
)
_BACKGROUND_SKIP_PREDICATES = {"gene_associated_with_disease"}
_BACKGROUND_SKIP_STUDY_TYPES = {"review", "narrative_review"}
_BACKGROUND_SUSPECT_PREDICATES = {
    "gene_associated_with_disease",
    "is_risk_factor_for",
    "is_biomarker_of",
}
_BACKGROUND_CUE_PATTERNS = (
    re.compile(r"\bin a separate study\b", re.I),
    re.compile(r"\bprevious(?:ly)?\b", re.I),
    re.compile(r"\brecently implicated\b", re.I),
    re.compile(r"\bestablished\b.{0,40}\brisk factor\b", re.I),
    re.compile(r"\bknown\b.{0,40}\brisk factor\b", re.I),
    re.compile(r"\bhas been associated with\b", re.I),
    re.compile(r"\bhave been associated with\b", re.I),
)

_MODALITY_GUARD_PREDICATES = {
    "is_biomarker_of",
    "predicts",
    "distinguishes",
    "causes",
    "increases",
    "reduces",
    "modulates",
    "treats",
    "has_adverse_effect",
}

_METHOD_GUARD_PREDICATES = _MODALITY_GUARD_PREDICATES | {
    "correlates_with",
    "is_associated_with",
}
_ENDPOINT_GUARD_PREDICATES = {
    "causes",
    "increases",
    "reduces",
    "modulates",
    "activates",
    "inhibits",
}

_PURE_MODALITY_NAMES = frozenset({
    "ct",
    "computed tomography",
    "pet",
    "positron emission tomography",
    "fdg pet",
    "fdg-pet",
    "amyloid pet",
    "spect",
    "single photon emission tomography",
    "single-photon emission tomography",
    "single photon emission tomography scanning",
    "single-photon emission tomography scanning",
    "single photon emission computed tomography",
    "single-photon emission computed tomography",
    "mri",
    "magnetic resonance imaging",
    "structural mri",
    "structural magnetic resonance imaging",
    "fmri",
    "functional mri",
    "functional magnetic resonance imaging",
    "dti",
    "diffusion tensor imaging",
    "diffusion mri",
    "diffusion magnetic resonance imaging",
    "eeg",
    "electroencephalography",
    "meg",
    "magnetoencephalography",
})

_MODALITY_TERM_RE = re.compile(
    r"\b("
    r"ct|computed tomography|pet|positron emission tomography|fdg[-\s]?pet|"
    r"amyloid pet|spect|single[-\s]photon emission tomography|"
    r"single[-\s]photon emission computed tomography|"
    r"mri|magnetic resonance imaging|structural mri|"
    r"structural magnetic resonance imaging|fmri|functional mri|"
    r"functional magnetic resonance imaging|dti|diffusion tensor imaging|"
    r"diffusion mri|diffusion magnetic resonance imaging|eeg|"
    r"electroencephalography|meg|magnetoencephalography"
    r")\b",
    re.I,
)

_MODALITY_MEASUREMENT_RE = re.compile(
    r"\b("
    r"hypometabolism|hypermetabolism|suvr|standardized uptake value|"
    r"hypointensity|hyperintensity|signal|signal intensity|"
    r"metabolism|metabolic|cerebral metabolism|glucose metabolism|"
    r"volume|volumetric|atrophy|thickness|thinning|surface area|"
    r"fractional anisotropy|\bfa\b|mean diffusivity|radial diffusivity|"
    r"axial diffusivity|connectivity|blood flow|perfusion|binding|uptake|"
    r"activation|deactivation|activation pattern|deactivation pattern|"
    r"localization|lateralization|texture|texture analysis|"
    r"receptor density|cortical thickness|white matter integrity|"
    r"gray matter volume|grey matter volume|plaque burden|plaque count|"
    r"lesion|lesions|lesion load|infarct|infarcts|infarct count|"
    r"white matter change|white matter changes|counts?|pittsburgh compound-b|"
    r"pittsburgh compound b|"
    r"\bpib\b|amyloid-beta|amyloid beta|tau"
    r")\b",
    re.I,
)

_METHOD_PROCEDURE_RE = re.compile(
    r"\b("
    r"manual segmentation|serial segmentation|segmentation|registration|"
    r"quantification method|quantification methods|methods?|"
    r"classifier|classification algorithm|algorithm|pipeline|software|"
    r"scanning|imaging technique|imaging method|neuroimaging technique|"
    r"neuroimaging method|mapping technique|analysis technique|"
    r"support vector machine|machine learning model|statistical model|"
    r"intraperitoneal injection|subcutaneous injection|intravenous injection|"
    r"injection of|administration of"
    r")\b",
    re.I,
)

_GENERIC_METHOD_ENTITY_RE = re.compile(
    r"\b("
    r"methods?|techniques?|algorithms?|classifiers?|pipelines?|software|"
    r"registration|segmentation|mapping techniques?|analysis techniques?|"
    r"diagnostic test sensitivity|diagnostic test specificity|test batter(?:y|ies)|"
    r"magnetic resonance scans?|mri scans?|serial mri scans?|multiple serial mri scans?"
    r")\b",
    re.I,
)

_GENERIC_IMAGING_ENTITY_RE = re.compile(
    r"\b("
    r"brain imaging|structural imaging|functional imaging|neuroimaging|"
    r"imaging biomarkers?|imaging and .*biomarkers?"
    r")\b",
    re.I,
)

_CONTINUOUS_ENDPOINT_RE = re.compile(
    r"\b(severity|survival|score|scores|performance|function|outcome|"
    r"outcomes|decline|impairment)\b",
    re.I,
)
_ASSOCIATION_CUE_RE = re.compile(
    r"\b(related to|correlat(?:e|es|ed|ion|ions)? with|associated with|"
    r"relationship with)\b",
    re.I,
)
_DISEASE_ENDPOINT_RE = re.compile(
    r"\b(alzheimer|dementia|schizophrenia|epilepsy|disease|syndrome|"
    r"disorder|impairment|psychosis|stroke|sclerosis|parkinson|"
    r"mild cognitive impairment|frontotemporal|vascular dementia)\b",
    re.I,
)
_BIOMARKER_ABUNDANCE_CUE_RE = re.compile(
    r"\b(accumulat(?:e|es|ed|ion|ions|ing)|deposition|deposits?|burden|"
    r"levels?|abundance|expression|amount|concentration|load|plaques?|"
    r"tangles?)\b",
    re.I,
)
_MEASUREMENT_GROUP_COMPARISON_RE = re.compile(
    r"\b(compared with|compared to|relative to|versus|vs\.?|patients? "
    r"(?:show(?:ed|s)?|had|have|exhibited)|controls?|case-control)\b",
    re.I,
)
_DISEASE_RISK_DIRECTION_CUE_RE = re.compile(
    r"\b(risk|protect(?:s|ed|ive|ion)?|prevent(?:s|ed|ion)?|incidence|"
    r"prevalence|development|developing|onset)\b",
    re.I,
)
_MEASUREMENT_SUBJECT_TYPES = (
    "biomarker",
    "brain_region",
    "network",
    "cognitive",
    "rating_scale",
    "clinical_marker",
)
_DISEASE_GENERIC_TOKENS = {
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "syndrome",
    "syndromes",
    "development",
    "onset",
    "risk",
}
_DISEASE_ABBREVIATION_PATTERNS = (
    ("alzheimer", re.compile(r"\b(ad|alzheimers?)\b", re.I)),
    ("mild cognitive impairment", re.compile(r"\bmci\b", re.I)),
    ("frontotemporal dementia", re.compile(r"\bftd\b", re.I)),
    ("dementia with lewy bodies", re.compile(r"\bdlb\b", re.I)),
    ("multiple sclerosis", re.compile(r"\bms\b", re.I)),
    ("parkinson", re.compile(r"\bpd\b", re.I)),
    ("amyotrophic lateral sclerosis", re.compile(r"\bals\b", re.I)),
    ("major depressive disorder", re.compile(r"\bmdd\b", re.I)),
    ("bipolar disorder", re.compile(r"\bbd\b", re.I)),
    ("schizophrenia", re.compile(r"\bschizophren\w*\b", re.I)),
)


def _predicate_supported_by_text(predicate: str, raw_text: str) -> bool:
    """Check if raw_text contains keywords supporting the given predicate."""
    if not raw_text or predicate in _VAGUE_PREDICATES:
        return True
    patterns = _PREDICATE_KEYWORDS.get(predicate)
    if not patterns:
        return True
    return any(p.search(raw_text) for p in patterns)


def _normalize_directional_association_claim(claim: Claim) -> bool:
    """Restore association predicates after LLM over-refines "lower X related to Y"."""
    if claim.predicate not in {"reduces", "increases"}:
        return False
    subject_type = str(claim.metadata.get("subject_type", "")).lower()
    object_type = str(claim.metadata.get("object_type", "")).lower()
    direction_text = getattr(claim.evidence, "direction", "") or ""
    raw_text = " ".join([claim.raw_text or "", direction_text])
    if (
        any(t in subject_type for t in _MEASUREMENT_SUBJECT_TYPES)
        and (object_type == "disease" or _DISEASE_ENDPOINT_RE.search(claim.object_name or ""))
        and _MEASUREMENT_GROUP_COMPARISON_RE.search(raw_text)
        and not _DISEASE_RISK_DIRECTION_CUE_RE.search(raw_text)
    ):
        original = claim.predicate
        claim.predicate = "distinguishes"
        claim.metadata["predicate_original"] = original
        claim.metadata["predicate_normalized_reason"] = "measurement differs in disease group"
        return True
    if (
        "biomarker" in subject_type
        and (object_type == "disease" or _DISEASE_ENDPOINT_RE.search(claim.object_name or ""))
        and _BIOMARKER_ABUNDANCE_CUE_RE.search(raw_text)
    ):
        original = claim.predicate
        claim.predicate = "is_associated_with"
        claim.confidence = min(claim.confidence, 0.5)
        claim.metadata["predicate_original"] = original
        claim.metadata["predicate_normalized_reason"] = "biomarker abundance in disease"
        return True
    if not _CONTINUOUS_ENDPOINT_RE.search(claim.object_name or ""):
        return False
    if not _ASSOCIATION_CUE_RE.search(raw_text):
        return False
    original = claim.predicate
    claim.predicate = "correlates_with"
    claim.metadata["predicate_original"] = original
    claim.metadata["predicate_normalized_reason"] = "directional association endpoint"
    return True


def _normalise_guard_name(name: str) -> str:
    s = re.sub(r"[-_/]+", " ", (name or "").strip().lower())
    return re.sub(r"\s+", " ", s).strip()


def _modality_method_guard_reasons(claim: Claim) -> list[str]:
    """Return reasons to skip method/modality-as-biomedical-entity claims.

    This is a conservative ingestion backstop for LLM outputs. It does not
    rewrite claims; it only skips cases where the subject is a pure modality or
    procedure being used as if it were a biomarker, predictor, or treatment.
    Concrete modality-derived measurements such as "FDG hypometabolism",
    "amyloid PET SUVR", and "dopamine transporter binding" are retained.
    """
    reasons: list[str] = []

    for role, name in (
        ("subject", claim.subject_name),
        ("object", claim.object_name),
    ):
        endpoint = _normalise_guard_name(name)
        if claim.predicate in _MODALITY_GUARD_PREDICATES:
            if endpoint in _PURE_MODALITY_NAMES:
                reasons.append(f"pure imaging modality {role}")
            elif _MODALITY_TERM_RE.search(endpoint) and not _MODALITY_MEASUREMENT_RE.search(endpoint):
                reasons.append(f"modality {role} without concrete measurement")
            elif _GENERIC_IMAGING_ENTITY_RE.search(endpoint) and not _MODALITY_MEASUREMENT_RE.search(endpoint):
                reasons.append(f"generic imaging {role} without concrete measurement")

        if (
            claim.predicate in _METHOD_GUARD_PREDICATES
            and _METHOD_PROCEDURE_RE.search(endpoint)
        ):
            reasons.append(f"method/procedure {role}")
        elif (
            claim.predicate in _METHOD_GUARD_PREDICATES
            and _GENERIC_METHOD_ENTITY_RE.search(endpoint)
            and not _MODALITY_MEASUREMENT_RE.search(endpoint)
        ):
            reasons.append(f"generic method/test {role} without concrete measurement")

    return reasons


def _endpoint_supported_by_raw_text(endpoint: str, raw_text: str) -> bool:
    """Return True when the evidence sentence visibly contains the endpoint.

    The extraction prompt requires raw_sentence to support both endpoints. This
    backstop catches LLM-injected disease objects while allowing common disease
    abbreviations such as AD, MCI, PD, and MS.
    """
    raw = _normalise_guard_name(raw_text)
    endpoint_norm = _normalise_guard_name(endpoint)
    if not endpoint_norm:
        return True
    if endpoint_norm in raw:
        return True

    for anchor, pattern in _DISEASE_ABBREVIATION_PATTERNS:
        if anchor in endpoint_norm and pattern.search(raw_text or ""):
            return True

    tokens = [
        t for t in re.findall(r"[a-z0-9]+", endpoint_norm)
        if len(t) >= 4 and t not in _DISEASE_GENERIC_TOKENS
    ]
    if not tokens:
        return True
    return any(re.search(rf"\b{re.escape(t)}\w*\b", raw, re.I) for t in tokens)


def _unsupported_endpoint_guard_reasons(claim: Claim) -> list[str]:
    reasons: list[str] = []
    if claim.predicate not in _ENDPOINT_GUARD_PREDICATES:
        return reasons

    object_type = str(claim.metadata.get("object_type", "")).lower()
    if object_type == "disease" or _DISEASE_ENDPOINT_RE.search(claim.object_name or ""):
        if not _endpoint_supported_by_raw_text(claim.object_name, claim.raw_text or ""):
            reasons.append("disease object absent from raw evidence")
    return reasons


def _background_claim_reasons(claim: Claim) -> list[str]:
    """Conservative detector for introduction/background-style claims.

    This never invents new semantics. It only identifies claims that look like
    prior-work restatements so ingestion can either downweight them or, for the
    narrowest case, skip clearly background-only review summaries.
    """
    reasons: list[str] = []
    if claim.predicate not in _BACKGROUND_SUSPECT_PREDICATES:
        return reasons

    raw = (claim.raw_text or "").strip()
    if raw:
        for pat in _BACKGROUND_CUE_PATTERNS:
            if pat.search(raw):
                reasons.append(f"background cue: {pat.pattern}")
                break

    study_type = (claim.evidence.study_type or "").strip().lower()
    if study_type in _BACKGROUND_SKIP_STUDY_TYPES:
        reasons.append(f"study_type={study_type}")

    return reasons


def refine_predicate(claim: Claim, llm_client: Optional[OpenAI] = None, model: str = "") -> str:
    """Refine vague predicates using rule-based keywords + LLM fallback.

    RELATE-inspired 2-stage pipeline:
    1. Rule-based: match raw_text keywords against predicate patterns
    2. LLM fallback: ask LLM to choose the most precise predicate

    Returns the refined predicate (or original if no refinement found).
    """
    if claim.predicate not in _VAGUE_PREDICATES:
        return claim.predicate

    raw = claim.raw_text or ""

    # Stage 1: rule-based keyword matching
    for predicate, patterns in _PREDICATE_KEYWORDS.items():
        for pattern in patterns:
            if pattern.search(raw):
                logger.debug(
                    f"refined {claim.predicate} → {predicate} "
                    f"(keyword match in '{raw[:80]}')"
                )
                return predicate

    # Stage 2: LLM fallback for ambiguous cases
    if llm_client and raw:
        refined = _llm_refine_predicate(claim, llm_client, model)
        if refined and refined in CLAIM_PREDICATES:
            return refined

    return claim.predicate


def _llm_refine_predicate(claim: Claim, client: OpenAI, model: str) -> Optional[str]:
    """Ask LLM to choose the most precise predicate for an ambiguous claim."""
    prompt = f"""Choose the most precise predicate for this claim. The current predicate `{claim.predicate}` is too vague — you MUST pick a more specific one from the list below.

Subject: {claim.subject_name} (type: {claim.metadata.get('subject_type', 'unknown')})
Object: {claim.object_name} (type: {claim.metadata.get('object_type', 'unknown')})
Context: {claim.raw_text[:300]}
Study type: {claim.evidence.study_type or 'unknown'}

Decision rubric (pick ONE):
- `is_risk_factor_for` — longitudinal/prospective studies showing X increases future risk of Y
- `is_biomarker_of` — X measurable indicator used for diagnosis/staging of Y
- `causes` — RCT, Mendelian randomization, or mechanistic evidence of causation
- `predicts` — X has prognostic value for Y outcome
- `treats` — therapeutic intervention X for condition Y
- `inhibits` — X suppresses/blocks/antagonizes Y
- `activates` — X enhances/stimulates/agonizes Y
- `increases` — X elevates levels/expression of Y
- `reduces` — X decreases levels/expression of Y
- `modulates` — X has regulatory influence on Y (unclear direction)
- `correlates_with` — cross-sectional direction-unknown association
- `mediates` — X acts as intermediate step between two things
- `distinguishes` — X differentiates between groups/conditions
- `part_of` — X is anatomical/compositional part of Y
- `co_occurs_with` — X and Y observed together without causal inference

Rules:
1. Do NOT keep `is_associated_with` or `correlates_with` unless NO other predicate fits.
2. If the context describes levels/expression, use `increases`/`reduces`.
3. If the subject is a drug and object is a disease, prefer `treats`.
4. If the study is longitudinal and talks about future outcomes, prefer `is_risk_factor_for` or `predicts`.
5. When direction is unclear but both entities co-vary, prefer `correlates_with` over `is_associated_with`.

Output ONLY the predicate name, nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a biomedical ontology expert. Output only the predicate name."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        pred = response.choices[0].message.content.strip().lower()
        # Strip any punctuation or quotes
        pred = re.sub(r"[^a-z_]", "", pred)
        if pred in CLAIM_PREDICATES:
            return pred
    except Exception as e:
        logger.debug(f"LLM predicate refinement failed: {e}")

    return None


# mapping from claim entity types to domain tags
ENTITY_TYPE_TO_DOMAIN = {
    # Legacy raw types (back-compat with already-extracted claims)
    "brain_region": DomainTag.NEUROANATOMY,
    "disease": DomainTag.DISEASE,
    "gene": DomainTag.GENE,
    "neurotransmitter": DomainTag.NEUROTRANSMITTER,
    "protein": DomainTag.GENE,
    "drug": DomainTag.DRUG,
    "network": DomainTag.CONNECTIVITY,
    "biomarker": DomainTag.BIOMARKER,
    "cognitive_function": DomainTag.COGNITIVE_FUNCTION,
    # 7-atom aligned types (new, emitted by atom-aware extractor)
    "imaging_marker":  DomainTag.BIOMARKER,
    "imaging_feature": DomainTag.IMAGING_FEATURE,
    "clinical_marker": DomainTag.BIOMARKER,
    "gene_target":     DomainTag.GENE,
    "outcome":         DomainTag.TREATMENT_OUTCOME,
    "clinical_outcome": DomainTag.TREATMENT_OUTCOME,
    "clinical_event":  DomainTag.TREATMENT_OUTCOME,
    "rating_scale":    DomainTag.TREATMENT_OUTCOME,
    "adverse_event":   DomainTag.TREATMENT_OUTCOME,
    "individual_data": DomainTag.DATASET_VARIABLE,
}


_ENTITY_TYPE_ALIASES = {
    "imagingmarker": "imaging_marker",
    "imaging_marker": "imaging_marker",
    "imaging_feature": "imaging_feature",
    "imagingfeature": "imaging_feature",
    "clinicalmarker": "clinical_marker",
    "clinical_marker": "clinical_marker",
    "genetarget": "gene_target",
    "gene_target": "gene_target",
    "outcome": "outcome",
    "clinicaloutcome": "clinical_outcome",
    "clinical_outcome": "clinical_outcome",
    "clinicalevent": "clinical_event",
    "clinical_event": "clinical_event",
    "individualdata": "individual_data",
    "individual_data": "individual_data",
}


def _normalize_entity_type(entity_type: str) -> str:
    """Normalize LLM-emitted entity types before domain-tag mapping.

    Claim extraction now emits both legacy fine-grained labels
    (``biomarker``, ``rating_scale``) and atom-level labels
    (``IMAGING_MARKER``, ``OUTCOME``, ``GENE_TARGET``). Keep both forms
    domain-compatible so newly minted CLM_CONCEPT nodes do not default to
    disease.
    """
    if not entity_type:
        return ""
    normalized = str(entity_type).strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    return _ENTITY_TYPE_ALIASES.get(normalized, normalized)


ENTITY_RESOLUTION_VERSION = "entity-resolution.v3"


def _word_boundary_match(short: str, long: str) -> bool:
    """Require phrase boundaries for every alias length, including FACE/TERA."""
    return bool(short) and short in long and re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", long) is not None


def _resolution_roles(entity_name: str, entity_type: str) -> frozenset[Atom]:
    roles = declared_type_atoms(entity_type, entity_name)
    if roles:
        return roles
    domain = ENTITY_TYPE_TO_DOMAIN.get(_normalize_entity_type(entity_type))
    return frozenset(DOMAIN_TO_ATOMS.get(domain.value, ())) if domain else frozenset()


def _resolution_compatible(node: ConceptNode, entity_name: str, entity_type: str) -> bool:
    """All lookup paths require nonempty, compatible type evidence and names."""
    # Observation nodes can have disease/gene tags and human-readable aliases;
    # those do not make them entity identities. CLM_CONCEPT mentions are valid.
    if node is None or node.id.startswith("CLM:"):
        return False
    return _roles_compatible(_resolution_roles(entity_name, entity_type), concept_atom_roles(node)) and any(
        name_compatibility_score(entity_name, label) >= 0.70
        for label in (node.preferred_name, *node.aliases)
    )


def _mint_domain(entity_name: str, entity_type: str) -> str:
    roles = _resolution_roles(entity_name, entity_type)
    expected = ENTITY_TYPE_TO_DOMAIN.get(_normalize_entity_type(entity_type))
    if expected and roles & set(DOMAIN_TO_ATOMS.get(expected.value, ())):
        return expected.value
    domains = {Atom.IMAGING_MARKER: "imaging_feature", Atom.GENE_TARGET: "gene", Atom.DRUG: "drug",
               Atom.DISEASE: "disease", Atom.OUTCOME: "treatment_outcome", Atom.COGNITIVE_TASK: "paradigm",
               Atom.INDIVIDUAL_DATA: "dataset_variable"}
    return domains[next(iter(roles))] if len(roles) == 1 else "external"


# ── Name resolution index (O(1) lookup instead of O(n) scan) ──────────
class _ResolutionIndex:
    """Pre-built lookup tables for fast entity resolution."""

    def __init__(self):
        # Keep singleton entries compact in million-node graphs; allocate a
        # set only when a name actually has multiple candidate identities.
        self._exact: dict[str, str | set[str]] = {}
        self._lower: dict[str, str | set[str]] = {}
        self._alias_lower: dict[str, str | set[str]] = {}
        self._built = False
        self._kg: Optional[KnowledgeGraph] = None
        self._size = 0

    def build(self, kg: KnowledgeGraph):
        self._exact.clear()
        self._lower.clear()
        self._alias_lower.clear()
        for node in kg._index.values():
            self.add(node.id, node.preferred_name, node.aliases)
        self._built = True
        self._kg = kg
        self._size = len(kg._index)
        logger.info(f"resolution index built: {len(self._exact)} exact, {len(self._alias_lower)} aliases")

    def is_for(self, kg: KnowledgeGraph) -> bool:
        return self._built and self._kg is kg and self._size == len(kg._index)

    def add(self, node_id: str, preferred_name: str, aliases: list[str] = None):
        """Incrementally add a new node to the index."""
        if self._kg is not None:
            self._size = len(self._kg._index)
        if node_id.startswith("CLM:"):
            return
        self._append(self._exact, preferred_name, node_id)
        self._append(self._lower, preferred_name.lower(), node_id)
        for alias in (aliases or []):
            if alias:
                self._append(self._alias_lower, alias.lower(), node_id)
        if self._kg is not None:
            self._size = len(self._kg._index)

    @staticmethod
    def _append(index, key, node_id):
        previous = index.get(key)
        if previous is None:
            index[key] = node_id
        elif isinstance(previous, str):
            if previous != node_id:
                index[key] = {previous, node_id}
        else:
            previous.add(node_id)

    @staticmethod
    def _ids(value):
        return {value} if isinstance(value, str) else value

    @staticmethod
    def _unique(ids: str | set[str]) -> Optional[str]:
        return ids if isinstance(ids, str) else next(iter(ids)) if len(ids) == 1 else None

    def lookup_candidates(self, name: str) -> set[str]:
        return self._ids(self._lower.get(name.lower(), set())) | self._ids(self._alias_lower.get(name.lower(), set()))

    def lookup_exact(self, name: str) -> Optional[str]:
        return self._unique(self._exact.get(name, set()))

    def lookup_lower(self, name: str) -> Optional[str]:
        return self._unique(self._lower.get(name.lower(), set()))

    def lookup_alias(self, name: str) -> Optional[str]:
        return self._unique(self._alias_lower.get(name.lower(), set()))


_resolution_idx = _ResolutionIndex()


def _safe_salvage_id(kg: KnowledgeGraph, original: str, salvaged: str, entity_type: str) -> Optional[str]:
    candidates = [node_id for node_id in _resolution_idx.lookup_candidates(salvaged)
                  if node_id in kg._index and _resolution_compatible(kg._index[node_id], original, entity_type)]
    return candidates[0] if len(candidates) == 1 else None


# ── Persistent dedup state (across ingest_claims calls) ───────────────
# Building these from scratch every call requires scanning all CLM nodes
# (~178K+ in a mature graph), which costs 30-60s per call. Cache them
# module-level so they survive between disease-year batches in the same
# process, while binding the cache to one KnowledgeGraph instance at a time.
_persistent_seen_evidence: set[str] = set()
_dedup_kg: Optional[KnowledgeGraph] = None
_dedup_revision = -1
_dedup_papers = None


def _seed_dedup_from_kg(kg: KnowledgeGraph):
    """Build cross-run dedup state from existing CLM nodes. Idempotent."""
    global _dedup_kg, _dedup_revision, _dedup_papers
    if _dedup_kg is kg and _dedup_revision == kg.claim_revision and _dedup_papers is kg.paper_identities:
        return
    _persistent_seen_evidence.clear()
    for node in kg._index.values():
        if not node.id.startswith("CLM:"):
            continue
        meta = node.metadata
        if not isinstance(meta, dict):
            continue
        key = evidence_dedup_key(meta, papers=kg.paper_identities)
        if key:
            _persistent_seen_evidence.add(key)
    _dedup_kg = kg
    _dedup_revision = kg.claim_revision
    _dedup_papers = kg.paper_identities
    logger.info("dedup state seeded: %s exact evidence records", len(_persistent_seen_evidence))


def resolve_entity(
    kg: KnowledgeGraph,
    entity_name: str,
    entity_type: str = "",
) -> Optional[str]:
    """Resolve an entity name to a concept ID in the knowledge graph.

    Exact names and aliases share a type-checked candidate set. Fuzzy matches
    require phrase boundaries and semantic name compatibility. Ambiguity or
    unknown types never select an arbitrary existing identity. The existing
    strict/noise controls still govern unresolved mention creation.
    """
    if not entity_name:
        return None
    normalized_entity_type = _normalize_entity_type(entity_type)

    # The current KG's verified dictionary takes precedence over multiple old
    # mention spellings. It matches the complete endpoint and checks its role;
    # no fuzzy match or compound stripping is introduced by this route.
    registry = kg.relation_identities
    if registry:
        term = registry.term_for({"subject_name": entity_name, "subject_type": entity_type}, "subject")
        if term and not term["target_id"].startswith("CLM:") and kg.has_concept(term["target_id"]):
            return term["target_id"]

    # Build index on first call
    if not _resolution_idx.is_for(kg):
        _resolution_idx.build(kg)

    entity_lower = entity_name.lower()
    candidates = [kg._index[node_id] for node_id in sorted(_resolution_idx.lookup_candidates(entity_name))
                  if node_id in kg._index and _resolution_compatible(kg._index[node_id], entity_name, entity_type)]
    if not candidates and _resolution_roles(entity_name, entity_type):
        for node in kg._index.values():
            if any((_word_boundary_match(label.lower(), entity_lower)
                    or _word_boundary_match(entity_lower, label.lower()))
                   for label in (node.preferred_name, *node.aliases)) and _resolution_compatible(node, entity_name, entity_type):
                candidates.append(node)
    if len(candidates) == 1:
        return candidates[0].id
    if candidates:
        logger.warning("ambiguous entity resolution retained as an unresolved mention: %r (%d candidates)",
                       entity_name, len(candidates))

    # 6. not found — noise check before minting a brand-new CLM_CONCEPT.
    # Curated matches (steps 1-5) already returned above, so we only see
    # names that would otherwise pollute the graph with auto-generated
    # low-quality nodes. First try to salvage by stripping noise affixes
    # and rechecking the curated index; if still noise, drop the entity.
    if _NOISE_FILTER_ENABLED and _is_noisy_name(entity_name):
        salvaged = _salvage_noisy_name(entity_name)
        if salvaged:
            hit = _safe_salvage_id(kg, entity_name, salvaged, entity_type)
            if hit:
                _DROP_LOG.record(entity_name, "salvaged", salvaged, hit)
                return hit
        _DROP_LOG.record(entity_name, "dropped", salvaged, None, _noise_reasons(entity_name))
        logger.debug(f"dropped noise entity: {entity_name!r} (salvage={salvaged!r})")
        return None

    # 6b. strict_phase1 mode — drop anything not already curated, even if it
    # passes the noise filter. Phase 1 covers most neuroscience terms; Phase
    # 2 LLM extraction should reuse those nodes, not proliferate new ones.
    if _STRICT_PHASE1:
        # Try one last salvage: maybe the entity is an obvious variant
        # (plural, minor morphology) of something in the index.
        salvaged = _salvage_noisy_name(entity_name) if _NOISE_FILTER_ENABLED else None
        if salvaged:
            hit = _safe_salvage_id(kg, entity_name, salvaged, entity_type)
            if hit:
                _DROP_LOG.record(entity_name, "salvaged_strict", salvaged, hit)
                return hit
        _DROP_LOG.record(entity_name, "strict_dropped", salvaged, None,
                         ["not in phase1 curated index"])
        logger.debug(f"strict_phase1 dropped entity: {entity_name!r}")
        return None

    # 6c. CLM_CONCEPT endpoint quality pre-check: reject names that are
    # too vague to serve as hypothesis endpoints BEFORE minting a node.
    # This internalizes what judge_clm_endpoints.py does post-hoc.
    if _is_vague_endpoint_name(entity_name):
        _DROP_LOG.record(entity_name, "vague_endpoint", None, None,
                         ["too vague to be a hypothesis endpoint"])
        logger.debug(f"vague endpoint dropped: {entity_name!r}")
        return None

    # Preserve a typed/unclassified mention; never overwrite a conflicting ID.
    domain = _mint_domain(entity_name, entity_type)
    new_id = f"CLM_CONCEPT:{entity_name.replace(' ', '_')}"
    existing = kg.get_concept(new_id)
    if existing is not None:
        if existing.preferred_name == entity_name and existing.source_vocab == "claim_extraction" and existing.domain_tags == [domain]:
            return new_id
        digest = hashlib.sha256((entity_lower + "|" + normalized_entity_type).encode("utf-8")).hexdigest()[:16]
        new_id += "__" + digest
        existing = kg.get_concept(new_id)
        if existing is not None:
            if existing.preferred_name == entity_name and existing.source_vocab == "claim_extraction" and existing.domain_tags == [domain]:
                return new_id
            raise ValueError("unresolved mention ID collision; refusing to overwrite an existing entity")
    kg.add_concept(ConceptNode(
        id=new_id,
        preferred_name=entity_name,
        domain_tags=[domain],
        source_vocab="claim_extraction",
    ))
    _resolution_idx.add(new_id, entity_name)
    logger.info(f"created new concept: {new_id} ({entity_name})")
    return new_id


def _resolve_canonical_hint(kg: KnowledgeGraph, hint: str) -> Optional[str]:
    """Try to resolve a canonical-ID hint emitted by the atom-aware extractor.

    Accepts hints like "HGNC:APOE", "MSH:D000544", "ATC:N06DA02",
    "OUTCOME:HAM-D", "COGAT_DISORDER:dso_1470", "COGAT_TASK:trm_xxx".

    Strategy:
    1. Direct ID lookup if the hint is already a node ID.
    2. Prefix-aware fallback for HGNC: hints — try `HGNC:<symbol>` directly,
       then look up the symbol as a preferred_name / alias (gene symbols are
       widely indexed by symbol).
    3. Otherwise: use the hint payload as a preferred_name lookup so that an
       imprecise hint still benefits from the index without minting a node.
    """
    if not hint:
        return None
    hint = hint.strip()
    if not hint or hint.startswith("CLM:"):
        return None

    if kg.has_concept(hint):
        return hint

    if not _resolution_idx.is_for(kg):
        _resolution_idx.build(kg)

    if ":" in hint:
        prefix, payload = hint.split(":", 1)
        payload = payload.strip()
        if not payload:
            return None
        if prefix == "HGNC":
            for cand in (payload, payload.upper()):
                node_id = _resolution_idx._unique(_resolution_idx.lookup_candidates(cand))
                if node_id:
                    return node_id
        else:
            node_id = _resolution_idx._unique(_resolution_idx.lookup_candidates(payload))
            if node_id:
                return node_id
        return None

    return _resolution_idx._unique(_resolution_idx.lookup_candidates(hint))


def resolve_claim_entities(
    kg: KnowledgeGraph,
    claim: Claim,
) -> Claim:
    """Resolve subject and object names to concept IDs.

    Honors `subject_canonical_hint` / `object_canonical_hint` from the
    atom-aware extractor first; falls back to name+type resolution otherwise.
    """
    # The current KG may carry finite, independently source-reviewed repairs.
    # Apply them before hints/fuzzy resolution; other graphs retain the normal
    # resolver. An applicable rule rechecks its exact target seals and fails
    # closed if they changed, while preserving the original evidence payload.
    from .kg_multipaper_reinforcement import apply_multipaper_reimport as apply_reinforcement_multipaper_reimport
    reinforcement_multipaper = apply_reinforcement_multipaper_reimport(kg, claim)
    if reinforcement_multipaper is not None:
        return reinforcement_multipaper
    from .kg_multipaper_growth import apply_multipaper_reimport as apply_growth_multipaper_reimport
    growth_multipaper = apply_growth_multipaper_reimport(kg, claim)
    if growth_multipaper is not None:
        return growth_multipaper
    from .kg_multipaper_continuity import apply_multipaper_reimport as apply_continuity_multipaper_reimport
    continuity_multipaper = apply_continuity_multipaper_reimport(kg, claim)
    if continuity_multipaper is not None:
        return continuity_multipaper
    from .kg_multipaper_accretion import apply_multipaper_reimport as apply_accretion_multipaper_reimport
    accretion_multipaper = apply_accretion_multipaper_reimport(kg, claim)
    if accretion_multipaper is not None:
        return accretion_multipaper
    from .kg_multipaper_synthesis import apply_multipaper_reimport as apply_synthesis_multipaper_reimport
    synthesis_multipaper = apply_synthesis_multipaper_reimport(kg, claim)
    if synthesis_multipaper is not None:
        return synthesis_multipaper
    from .kg_multipaper_fusion import apply_multipaper_reimport as apply_fusion_multipaper_reimport
    fusion_multipaper = apply_fusion_multipaper_reimport(kg, claim)
    if fusion_multipaper is not None:
        return fusion_multipaper
    from .kg_multipaper_accumulation import apply_multipaper_reimport as apply_accumulation_multipaper_reimport
    accumulation_multipaper = apply_accumulation_multipaper_reimport(kg, claim)
    if accumulation_multipaper is not None:
        return accumulation_multipaper
    from .kg_multipaper_aggregation import apply_multipaper_reimport as apply_aggregation_multipaper_reimport
    aggregation_multipaper = apply_aggregation_multipaper_reimport(kg, claim)
    if aggregation_multipaper is not None:
        return aggregation_multipaper
    from .kg_multipaper_integration import apply_multipaper_reimport as apply_integration_multipaper_reimport
    integration_multipaper = apply_integration_multipaper_reimport(kg, claim)
    if integration_multipaper is not None:
        return integration_multipaper
    from .kg_multipaper_harmonization import apply_multipaper_reimport as apply_harmonization_multipaper_reimport
    harmonization_multipaper = apply_harmonization_multipaper_reimport(kg, claim)
    if harmonization_multipaper is not None:
        return harmonization_multipaper
    from .kg_multipaper_reconciliation import apply_multipaper_reimport as apply_reconciliation_multipaper_reimport
    reconciliation_multipaper = apply_reconciliation_multipaper_reimport(kg, claim)
    if reconciliation_multipaper is not None:
        return reconciliation_multipaper
    from .kg_multipaper_alignment import apply_multipaper_reimport as apply_alignment_multipaper_reimport
    alignment_multipaper = apply_alignment_multipaper_reimport(kg, claim)
    if alignment_multipaper is not None:
        return alignment_multipaper
    from .kg_multipaper_convergence import apply_multipaper_reimport as apply_convergence_multipaper_reimport
    convergence_multipaper = apply_convergence_multipaper_reimport(kg, claim)
    if convergence_multipaper is not None:
        return convergence_multipaper
    from .kg_multipaper_predicates import apply_multipaper_reimport as apply_predicate_multipaper_reimport
    predicate_multipaper = apply_predicate_multipaper_reimport(kg, claim)
    if predicate_multipaper is not None:
        return predicate_multipaper
    from .kg_multipaper_aliases import apply_multipaper_reimport as apply_alias_multipaper_reimport
    alias_multipaper = apply_alias_multipaper_reimport(kg, claim)
    if alias_multipaper is not None:
        return alias_multipaper
    from .kg_multipaper_broadening import apply_multipaper_reimport as apply_broad_multipaper_reimport
    broad_multipaper = apply_broad_multipaper_reimport(kg, claim)
    if broad_multipaper is not None:
        return broad_multipaper
    from .kg_multipaper_semantics import apply_multipaper_reimport as apply_semantic_multipaper_reimport
    semantic_multipaper = apply_semantic_multipaper_reimport(kg, claim)
    if semantic_multipaper is not None:
        return semantic_multipaper
    from .kg_multipaper_expansion import apply_multipaper_reimport as apply_expanded_multipaper_reimport
    expanded_multipaper = apply_expanded_multipaper_reimport(kg, claim)
    if expanded_multipaper is not None:
        return expanded_multipaper
    from .kg_multipaper_claim_repair import apply_multipaper_reimport
    multipaper = apply_multipaper_reimport(kg, claim)
    if multipaper is not None:
        return multipaper
    from .kg_source_scoped_sets_repair import apply_source_sets_reimport
    source_sets = apply_source_sets_reimport(kg, claim)
    if source_sets is not None:
        return source_sets
    from .kg_semantic_measurement_repair import apply_semantic_reimport
    semantic = apply_semantic_reimport(kg, claim)
    if semantic is not None:
        return semantic
    from .kg_systematic_consolidation import apply_systematic_reimport
    systematic = apply_systematic_reimport(kg, claim)
    if systematic is not None:
        return systematic
    from .kg_reviewed_relation_repair import apply_reviewed_reimport
    reviewed = apply_reviewed_reimport(kg, claim)
    if reviewed is not None:
        return reviewed

    meta = claim.metadata or {}

    subject_id = _resolve_canonical_hint(kg, meta.get("subject_canonical_hint", ""))
    if subject_id and not _resolution_compatible(kg.get_concept(subject_id), claim.subject_name, meta.get("subject_type", "")):
        subject_id = None
    if not subject_id:
        subject_id = resolve_entity(kg, claim.subject_name, meta.get("subject_type", ""))

    object_id = _resolve_canonical_hint(kg, meta.get("object_canonical_hint", ""))
    if object_id and not _resolution_compatible(kg.get_concept(object_id), claim.object_name, meta.get("object_type", "")):
        object_id = None
    if not object_id:
        object_id = resolve_entity(kg, claim.object_name, meta.get("object_type", ""))

    # A rejected resolution must not leave a stale, previously supplied ID.
    claim.subject_id = subject_id or ""
    claim.object_id = object_id or ""

    return claim


def ingest_claims(
    kg: KnowledgeGraph,
    results: list[ExtractionResult],
    refine_vague_predicates: bool = True,
    llm_base_url: str = "",
    llm_api_key: str = "",
    llm_model: str = "",
    keep_noise: bool = False,
    strict_phase1: bool = False,
    require_final_scope_audit: bool = True,
    drop_log_path: Optional[Path] = None,
) -> dict:
    """Ingest extracted claims into the knowledge graph.

    For each claim:
    1. Resolve subject/object to existing concepts (or create new ones)
    2. Refine vague predicates (RELATE: is_associated_with → precise predicate)
    3. Add a Claim node with full metadata
    4. Add a simplified edge for traversal

    Args:
        keep_noise: if True, skip build-time noise filter (debug mode).
        strict_phase1: if True, do NOT mint new CLM_CONCEPT nodes. Claims whose
            subject or object cannot resolve to a Phase-1-curated node are
            dropped. Use this when Phase 1 (NeuroNames/MeSH/DisGeNET/Cognitive
            Atlas + UMLS) is considered sufficient to cover medical terminology.
        require_final_scope_audit: fail the entire batch before graph mutation
            unless every claim has a valid current-policy ``scope_reaudit`` seal.
            This is fail-closed by default. Historical/manual repair paths must
            opt out explicitly with ``require_final_scope_audit=False``.
        drop_log_path: override path for the dropped-entities audit log.

    Returns summary dict.
    """
    claims_added = 0
    edges_added = 0
    errors = 0
    claims_skipped_noise = 0
    claims_skipped_unresolved = 0
    predicates_refined = 0
    claims_marked_background = 0
    claims_skipped_background = 0
    claims_skipped_modality_method = 0
    claims_skipped_low_confidence = 0
    claims_skipped_unsupported_endpoint = 0
    claim_outcomes: list[dict] = []

    def _record_claim_outcome(
        claim: Claim,
        *,
        status: str,
        reason: str,
        details: Optional[dict] = None,
    ) -> None:
        outcome = {
            "claim_id": str(claim.id),
            "status": status,
            "reason": reason,
        }
        if details:
            outcome["details"] = details
        claim_outcomes.append(outcome)

    # Configure build-time noise filter + strict_phase1 mode
    global _NOISE_FILTER_ENABLED, _STRICT_PHASE1
    _NOISE_FILTER_ENABLED = not keep_noise
    _STRICT_PHASE1 = strict_phase1
    if not keep_noise:
        _DROP_LOG.open(drop_log_path or _DROP_LOG_DEFAULT_PATH)

    # Initialize LLM client POOL for predicate refinement (all 4 keys, not just 1)
    # and decide concurrency: refinement is IO-bound (LLM calls), safe to
    # parallelize with threads. KG mutation stays serial (networkx is not
    # thread-safe).
    llm_clients: list[OpenAI] = []
    refine_workers = 0
    if refine_vague_predicates:
        base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
        keys_raw = os.environ.get("OPENAI_API_KEYS", "")
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if not keys and (llm_api_key or os.environ.get("OPENAI_API_KEY")):
            keys = [llm_api_key or os.environ.get("OPENAI_API_KEY", "")]
        model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
        if keys:
            import httpx
            llm_clients = [
                OpenAI(base_url=base_url, api_key=k, http_client=httpx.Client(verify=False))
                for k in keys
            ]
            # One worker per key × 3 so we overlap latency without exceeding
            # per-key rate limits. 4 keys → 12 workers, same pattern as
            # extraction/critic phases.
            refine_workers = max(len(llm_clients) * 3, 1)

    # Gather all claims across results, then optionally pre-refine predicates
    # in parallel (ingest is serial but refinement is IO-bound).
    all_claims: list = []
    extraction_failures: list[str] = []
    for result in results:
        if result.error:
            errors += 1
            extraction_failures.append(
                f"{result.paper.pmid or result.paper.doi or result.paper.title}: "
                f"{result.error}"
            )
            continue
        all_claims.extend(result.claims)

    if require_final_scope_audit:
        if extraction_failures:
            preview = "; ".join(extraction_failures[:5])
            _DROP_LOG.close()
            raise ValueError(
                "refusing KG mutation because automated extraction failed for "
                f"{len(extraction_failures)} paper(s): {preview}"
            )
        invalid_scope_audits: list[str] = []
        for claim in all_claims:
            try:
                validate_final_scope_reaudit(claim.to_dict())
            except Exception as exc:
                invalid_scope_audits.append(f"{claim.id}: {exc}")
        if invalid_scope_audits:
            preview = "; ".join(invalid_scope_audits[:5])
            _DROP_LOG.close()
            raise ValueError(
                "refusing KG mutation because claim scope audit validation failed "
                f"for {len(invalid_scope_audits)} claim(s): {preview}"
            )

    # Parallel rule-based + LLM refinement of vague predicates.
    # Only claims whose predicate is VAGUE and whose rule-based pass misses
    # actually hit the LLM, so most claims take <1ms here.
    if refine_vague_predicates and llm_clients and all_claims:
        from concurrent.futures import ThreadPoolExecutor

        def _refine_one(idx_claim):
            idx, claim = idx_claim
            # Round-robin client selection (no lock needed — read-only dispatch)
            client = llm_clients[idx % len(llm_clients)]
            original = claim.predicate
            new_pred = refine_predicate(claim, client, model)
            return idx, claim.id, original, new_pred

        with ThreadPoolExecutor(max_workers=refine_workers) as executor:
            futures = [
                executor.submit(_refine_one, (i, c))
                for i, c in enumerate(all_claims)
            ]
            # Collect results; assign back via index so we preserve per-claim
            # state (claim object reference stays intact).
            for f in futures:
                try:
                    idx, cid, original, new_pred = f.result()
                    if new_pred != original:
                        all_claims[idx].predicate = new_pred
                        predicates_refined += 1
                except Exception as e:
                    logger.debug(f"refine_predicate worker failed: {e}")

    # Serial KG mutation: resolve entities + add concept/edges.
    # Deduplicate complete observations, never just a same-paper triple.
    # Use persistent module-level state to avoid rescanning all CLM nodes
    # on every batch (would cost 30-60s per disease-year on a mature KG).
    _seed_dedup_from_kg(kg)
    _seen_evidence = _persistent_seen_evidence
    claims_skipped_dedup = 0

    logger.debug("evidence dedup state: %s observations", len(_seen_evidence))

    for claim in all_claims:
        try:
            if _normalize_directional_association_claim(claim):
                predicates_refined += 1

            modality_method_reasons = _modality_method_guard_reasons(claim)
            if modality_method_reasons:
                claims_skipped_modality_method += 1
                _record_claim_outcome(
                    claim,
                    status="rejected",
                    reason="modality_method_guard",
                    details={"guard_reasons": modality_method_reasons},
                )
                logger.debug(
                    f"skipped modality/method claim {claim.id}: "
                    f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}; "
                    f"reasons={modality_method_reasons}"
                )
                continue

            unsupported_endpoint_reasons = _unsupported_endpoint_guard_reasons(claim)
            if unsupported_endpoint_reasons:
                claims_skipped_unsupported_endpoint += 1
                _record_claim_outcome(
                    claim,
                    status="rejected",
                    reason="unsupported_endpoint_guard",
                    details={"guard_reasons": unsupported_endpoint_reasons},
                )
                logger.debug(
                    f"skipped unsupported-endpoint claim {claim.id}: "
                    f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}; "
                    f"reasons={unsupported_endpoint_reasons}"
                )
                continue

            background_reasons = _background_claim_reasons(claim)
            if background_reasons:
                claim.metadata["background_suspect"] = True
                claim.metadata["background_reasons"] = background_reasons
                study_type = (claim.evidence.study_type or "").strip().lower()
                if (
                    claim.predicate in _BACKGROUND_SKIP_PREDICATES
                    and study_type in _BACKGROUND_SKIP_STUDY_TYPES
                ):
                    claims_skipped_background += 1
                    _record_claim_outcome(
                        claim,
                        status="rejected",
                        reason="background_claim_guard",
                        details={"guard_reasons": background_reasons},
                    )
                    logger.debug(
                        f"skipped background claim {claim.id}: "
                        f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}"
                    )
                    continue
                claim.confidence *= _BACKGROUND_CLAIM_PENALTY
                claims_marked_background += 1

            # Predicate-evidence confidence penalty: if raw_text doesn't
            # contain keywords supporting the predicate, reduce confidence
            # before the low-confidence gate and before writing metadata.
            if not _predicate_supported_by_text(claim.predicate, claim.raw_text):
                claim.confidence *= _UNSUPPORTED_PREDICATE_PENALTY

            if claim.confidence < _MIN_INGEST_CLAIM_CONFIDENCE:
                claims_skipped_low_confidence += 1
                _record_claim_outcome(
                    claim,
                    status="rejected",
                    reason="low_confidence",
                    details={
                        "confidence": claim.confidence,
                        "minimum_confidence": _MIN_INGEST_CLAIM_CONFIDENCE,
                    },
                )
                logger.debug(
                    f"skipped low-confidence claim {claim.id}: "
                    f"confidence={claim.confidence:.3f}; "
                    f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}"
                )
                continue

            # resolve entities
            claim = resolve_claim_entities(kg, claim)

            if not claim.subject_id or not claim.object_id:
                # Distinguish noise drop, strict_phase1 drop, and real error
                if _STRICT_PHASE1:
                    claims_skipped_unresolved += 1
                    _record_claim_outcome(
                        claim,
                        status="rejected",
                        reason="unresolved_strict_phase1",
                    )
                    logger.debug(
                        f"strict_phase1 skipped claim {claim.id}: "
                        f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}"
                    )
                elif _NOISE_FILTER_ENABLED and (
                    _is_noisy_name(claim.subject_name)
                    or _is_noisy_name(claim.object_name)
                ):
                    claims_skipped_noise += 1
                    _record_claim_outcome(
                        claim,
                        status="rejected",
                        reason="unresolved_noise",
                        details={
                            "subject_reasons": _noise_reasons(claim.subject_name),
                            "object_reasons": _noise_reasons(claim.object_name),
                        },
                    )
                    logger.debug(
                        f"skipped noise claim {claim.id}: "
                        f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}"
                    )
                else:
                    logger.warning(f"could not resolve entities for claim {claim.id}")
                    errors += 1
                    _record_claim_outcome(
                        claim,
                        status="rejected",
                        reason="entity_resolution_error",
                    )
                continue

            missing_endpoint_ids = [
                endpoint_id
                for endpoint_id in (claim.subject_id, claim.object_id)
                if not kg.has_concept(endpoint_id)
            ]
            if missing_endpoint_ids:
                errors += 1
                _record_claim_outcome(
                    claim,
                    status="rejected",
                    reason="resolved_endpoint_absent_from_graph",
                    details={"missing_endpoint_ids": missing_endpoint_ids},
                )
                logger.warning(
                    "resolved endpoint ids absent from graph for claim %s: %s",
                    claim.id,
                    missing_endpoint_ids,
                )
                continue

            payload = claim.to_dict()
            if kg.has_concept(claim.id):
                existing = kg.get_concept(claim.id).metadata
                conflict = evidence_signature(existing) != evidence_signature(payload)
                if conflict:
                    errors += 1
                else:
                    claims_skipped_dedup += 1
                _record_claim_outcome(claim, status="rejected",
                    reason="conflicting_claim_id" if conflict else "duplicate_claim_id")
                continue

            observation_key = evidence_dedup_key(payload, papers=kg.paper_identities)
            if observation_key is not None and observation_key in _seen_evidence:
                claims_skipped_dedup += 1
                _record_claim_outcome(
                    claim,
                    status="rejected",
                    reason="duplicate_paper_evidence",
                    details={
                        "pmid": claim.source_paper.pmid,
                        "subject_id": claim.subject_id,
                        "predicate": claim.predicate,
                        "object_id": claim.object_id,
                    },
                )
                logger.debug(
                    f"dedup skipped claim {claim.id}: "
                    f"exact recorded observation {observation_key}"
                )
                continue

            # Different predicates, polarity, contexts and source sentences
            # remain independent evidence even when the paper/endpoints match.

            # add claim node
            kg.add_concept(ConceptNode(
                id=claim.id,
                preferred_name=f"{claim.subject_name} {claim.predicate} {claim.object_name}",
                domain_tags=["claim"],
                source_vocab="claim_extraction",
                definition=claim.raw_text,
                metadata=claim.to_dict(),
            ))

            # add simplified edge
            edge = claim.to_edge()
            kg.add_edge(edge)
            edges_added += 1

            # add about edges (claim → subject, claim → object)
            kg.add_edge(Edge(
                source_id=claim.id,
                target_id=claim.subject_id,
                relation_type="about",
                source="claim_extraction",
                confidence=claim.confidence,
            ))
            kg.add_edge(Edge(
                source_id=claim.id,
                target_id=claim.object_id,
                relation_type="about",
                source="claim_extraction",
                confidence=claim.confidence,
            ))

            # Dedup state is committed only after the claim node and all edges
            # have been added.  A failed mutation therefore remains retryable.
            if observation_key is not None:
                _seen_evidence.add(observation_key)
            claims_added += 1
            _record_claim_outcome(
                claim,
                status="accepted",
                reason="ingested",
            )

        except Exception as e:
            logger.warning(f"failed to ingest claim {claim.id}: {e}")
            errors += 1
            _record_claim_outcome(
                claim,
                status="rejected",
                reason="ingestion_error",
                details={"error": str(e)},
            )

    # Cache is valid only for graph mutations accounted for by this ingestion.
    # On failures, rebuild next time (partial mutations may have occurred).
    global _dedup_revision
    _dedup_revision = kg.claim_revision if not errors else -1

    if len(claim_outcomes) != len(all_claims):
        _DROP_LOG.close()
        raise RuntimeError(
            "claim ingestion accounting error: "
            f"{len(all_claims)} candidate claims but {len(claim_outcomes)} outcomes"
        )

    accepted_claim_ids = [
        outcome["claim_id"]
        for outcome in claim_outcomes
        if outcome["status"] == "accepted"
    ]
    rejected_claims = [
        outcome for outcome in claim_outcomes if outcome["status"] == "rejected"
    ]
    if len(accepted_claim_ids) != claims_added:
        _DROP_LOG.close()
        raise RuntimeError(
            "claim ingestion accounting error: "
            f"claims_added={claims_added} but accepted outcomes={len(accepted_claim_ids)}"
        )

    summary = {
        "claims_added": claims_added,
        "edges_added": edges_added,
        "errors": errors,
        "claims_skipped_noise": claims_skipped_noise,
        "claims_skipped_unresolved": claims_skipped_unresolved,
        "claims_skipped_dedup": claims_skipped_dedup,
        "claims_marked_background": claims_marked_background,
        "claims_skipped_background": claims_skipped_background,
        "claims_skipped_modality_method": claims_skipped_modality_method,
        "claims_skipped_low_confidence": claims_skipped_low_confidence,
        "claims_skipped_unsupported_endpoint": claims_skipped_unsupported_endpoint,
        "entities_salvaged": _DROP_LOG.n_salvaged,
        "entities_dropped": _DROP_LOG.n_dropped,
        "papers_processed": len(results),
        "predicates_refined": predicates_refined,
        "strict_phase1": strict_phase1,
        "require_final_scope_audit": require_final_scope_audit,
        "candidate_claims": len(all_claims),
        "accepted_claim_ids": accepted_claim_ids,
        "rejected_claims": rejected_claims,
        "claim_outcomes": claim_outcomes,
    }
    _DROP_LOG.close()
    logger.info(
        "claim ingestion complete: %s",
        {
            key: value
            for key, value in summary.items()
            if key not in {"accepted_claim_ids", "rejected_claims", "claim_outcomes"}
        },
    )
    return summary


def _cached_jsonl_field(path: Path, field: str) -> set[str]:
    """Return a per-process exact-value index for one top-level JSONL field."""
    path = Path(path).resolve()
    cache_key = (path, field)
    cached = _JSONL_FIELD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    values: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line).get(field)
                except Exception as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if value:
                    values.add(str(value))
    _JSONL_FIELD_CACHE[cache_key] = values
    return values


def _append_jsonl_records(path: Path, records: list[dict]) -> None:
    """Durably append a validated group of complete JSONL records."""
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def persist_ingestion_results(
    kg: KnowledgeGraph,
    results: list[ExtractionResult],
    ingest_summary: dict,
    claims_path: Path,
    *,
    label: str,
    year: Optional[int] = None,
    rejection_path: Optional[Path] = None,
) -> dict:
    """Persist only graph-accepted claims and audit every rejected candidate.

    ``ingest_claims`` returns one ordered outcome for every candidate claim.
    This function validates that contract against the original results and the
    in-memory graph before touching either JSONL file.  Accepted claims go to
    the canonical ``extracted_claims.jsonl``; all other candidates go to a
    separate, idempotent rejection audit.
    """
    claims_path = Path(claims_path)
    rejection_path = Path(rejection_path) if rejection_path else (
        claims_path.parent / CLAIM_REJECTION_AUDIT_FILENAME
    )
    if claims_path.resolve() == rejection_path.resolve():
        raise ValueError("canonical claims path and rejection audit path must differ")

    entries: list[tuple[ExtractionResult, Claim]] = []
    for result in results:
        if result is None or result.error:
            continue
        entries.extend((result, claim) for claim in result.claims)

    outcomes = ingest_summary.get("claim_outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("ingest summary is missing ordered claim_outcomes")
    if len(entries) != len(outcomes):
        raise ValueError(
            "ingestion persistence accounting mismatch: "
            f"{len(entries)} candidate claims but {len(outcomes)} outcomes"
        )

    timestamp = datetime.now().isoformat()
    accepted_records: list[dict] = []
    rejection_records: list[dict] = []
    accepted_ids: list[str] = []

    for occurrence, ((result, claim), outcome) in enumerate(
        zip(entries, outcomes),
        start=1,
    ):
        claim_id = str(claim.id)
        if str(outcome.get("claim_id", "")) != claim_id:
            raise ValueError(
                "ingestion outcome order mismatch at occurrence "
                f"{occurrence}: claim={claim_id!r}, outcome={outcome.get('claim_id')!r}"
            )
        status = outcome.get("status")
        if status not in {"accepted", "rejected"}:
            raise ValueError(
                f"invalid ingestion outcome status for {claim_id}: {status!r}"
            )

        record_year = year if year is not None else (result.paper.year or 0)
        claim_payload = claim.to_dict()
        canonical_record = dict(claim_payload)
        canonical_record["disease"] = label
        canonical_record["year"] = record_year
        canonical_record["extraction_timestamp"] = timestamp

        if status == "accepted":
            if not kg.has_concept(claim_id):
                raise ValueError(
                    f"refusing canonical write: accepted claim {claim_id} is absent from graph"
                )
            accepted_ids.append(claim_id)
            accepted_records.append(canonical_record)
            continue

        rejection_identity = {
            "claim_id": claim_id,
            "reason": outcome.get("reason", "unspecified"),
            "details": outcome.get("details", {}),
            "source_label": label,
            "year": record_year,
            "claim": claim_payload,
        }
        digest = hashlib.sha256(
            json.dumps(
                rejection_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        rejection_records.append({
            "schema_version": "claim-ingestion-rejection.v1",
            "audit_id": f"REJ:{digest}",
            "claim_id": claim_id,
            "status": "rejected",
            "reason": outcome.get("reason", "unspecified"),
            "details": outcome.get("details", {}),
            "source_label": label,
            "year": record_year,
            "rejected_at": timestamp,
            "claim": claim_payload,
        })

    summary_accepted_ids = [
        str(value) for value in ingest_summary.get("accepted_claim_ids", [])
    ]
    if accepted_ids != summary_accepted_ids:
        raise ValueError(
            "ingestion persistence accounting mismatch: accepted ids differ "
            "from ingest summary"
        )
    if len(accepted_records) != int(ingest_summary.get("claims_added", -1)):
        raise ValueError(
            "ingestion persistence accounting mismatch: accepted record count "
            "differs from claims_added"
        )
    if len(set(accepted_ids)) != len(accepted_ids):
        raise ValueError("ingestion persistence accounting mismatch: duplicate accepted ids")

    known_claim_ids = _cached_jsonl_field(claims_path, "id")
    pending_claim_ids = set(known_claim_ids)
    new_accepted: list[dict] = []
    for record in accepted_records:
        claim_id = str(record.get("id", ""))
        if claim_id in pending_claim_ids:
            continue
        pending_claim_ids.add(claim_id)
        new_accepted.append(record)

    known_audit_ids = _cached_jsonl_field(rejection_path, "audit_id")
    pending_audit_ids = set(known_audit_ids)
    new_rejections: list[dict] = []
    for record in rejection_records:
        audit_id = str(record["audit_id"])
        if audit_id in pending_audit_ids:
            continue
        pending_audit_ids.add(audit_id)
        new_rejections.append(record)

    _append_jsonl_records(claims_path, new_accepted)
    known_claim_ids.update(str(record["id"]) for record in new_accepted)
    _append_jsonl_records(rejection_path, new_rejections)
    known_audit_ids.update(str(record["audit_id"]) for record in new_rejections)

    return {
        "accepted_candidates": len(accepted_records),
        "claims_written": len(new_accepted),
        "claims_already_present": len(accepted_records) - len(new_accepted),
        "rejected_candidates": len(rejection_records),
        "rejections_written": len(new_rejections),
        "rejections_already_present": len(rejection_records) - len(new_rejections),
        "claims_path": str(claims_path.resolve()),
        "rejection_path": str(rejection_path.resolve()),
    }
