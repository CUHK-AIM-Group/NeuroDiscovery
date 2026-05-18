"""Claim ingestion: resolve entities, refine predicates, add claims to knowledge graph."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .claim_extractor import ClaimExtractor, ExtractionResult
from .graph_manager import KnowledgeGraph
from .schema import CLAIM_PREDICATES, Claim, ConceptNode, DomainTag, Edge, PaperRef

logger = logging.getLogger(__name__)

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
    " deficits", " manifestations", " abnormalities",
    " dysfunctions", " status", " outcomes", " profile",
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

_DROP_LOG_DEFAULT_PATH = Path("core/knowledge_graph/data/build_artifacts/dropped_entities.jsonl")


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

_VAGUE_ENDPOINT_SUFFIXES = (
    " subgroup", " subgroups", " alteration", " alterations",
    " impairment", " impairments", " deficit", " deficits",
    " dysfunction", " disability", " decline", " deterioration",
    " disturbance", " disturbances", " abnormality", " abnormalities",
    " complication", " complications", " consequence", " consequences",
    " manifestation", " manifestations", " characteristic", " characteristics",
    " relationship", " relationships",
)

_VAGUE_ENDPOINT_EXACT = frozenset({
    "focus", "integration", "balance", "knowledge", "autonomy",
    "performance", "adaptation", "resilience", "vulnerability",
    "recovery", "progression", "mechanism", "process", "outcome",
    "outcomes", "survival", "improvement", "response", "effect",
    "effects", "impact", "factor", "factors", "role", "function",
    "functions", "activity", "condition", "conditions", "treatment",
    "intervention", "approach", "strategy", "method",
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

    # Suffix match
    if any(sl.endswith(suf) for suf in _VAGUE_ENDPOINT_SUFFIXES):
        return True

    # Pattern match (adjective + generic noun)
    if _VAGUE_ENDPOINT_PATTERNS_RE.match(sl):
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


def _predicate_supported_by_text(predicate: str, raw_text: str) -> bool:
    """Check if raw_text contains keywords supporting the given predicate."""
    if not raw_text or predicate in _VAGUE_PREDICATES:
        return True
    patterns = _PREDICATE_KEYWORDS.get(predicate)
    if not patterns:
        return True
    return any(p.search(raw_text) for p in patterns)


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
    "brain_region": DomainTag.NEUROANATOMY,
    "disease": DomainTag.DISEASE,
    "gene": DomainTag.GENE,
    "neurotransmitter": DomainTag.NEUROTRANSMITTER,
    "protein": DomainTag.GENE,
    "drug": DomainTag.DRUG,
    "network": DomainTag.CONNECTIVITY,
    "biomarker": DomainTag.BIOMARKER,
    "cognitive_function": DomainTag.COGNITIVE_FUNCTION,
}


# Min length for short-string matches. Below this, only exact (case-insensitive)
# matches are allowed — otherwise 2-letter aliases like "IC" (Internal Capsula)
# match any substring containing those letters (e.g. "specific" contains "ic").
_MIN_SUBSTRING_LEN = 4


def _word_boundary_match(short: str, long: str) -> bool:
    """True if `short` appears in `long` as a whole token.

    Uses \\b word boundaries so "IC" matches "IC lesion" but NOT "specific".
    Assumes both args are lowercase.
    """
    return re.search(r"\b" + re.escape(short) + r"\b", long) is not None


# ── Name resolution index (O(1) lookup instead of O(n) scan) ──────────
class _ResolutionIndex:
    """Pre-built lookup tables for fast entity resolution."""

    def __init__(self):
        self._exact: dict[str, str] = {}          # preferred_name -> id
        self._lower: dict[str, str] = {}          # preferred_name.lower() -> id
        self._alias_lower: dict[str, str] = {}    # alias.lower() -> id
        self._built = False

    def build(self, kg: KnowledgeGraph):
        self._exact.clear()
        self._lower.clear()
        self._alias_lower.clear()
        for node in kg._index.values():
            self._exact[node.preferred_name] = node.id
            lower = node.preferred_name.lower()
            if lower not in self._lower:
                self._lower[lower] = node.id
            for alias in node.aliases:
                al = alias.lower()
                if al not in self._alias_lower:
                    self._alias_lower[al] = node.id
        self._built = True
        logger.info(f"resolution index built: {len(self._exact)} exact, {len(self._alias_lower)} aliases")

    def add(self, node_id: str, preferred_name: str, aliases: list[str] = None):
        """Incrementally add a new node to the index."""
        self._exact[preferred_name] = node_id
        lower = preferred_name.lower()
        if lower not in self._lower:
            self._lower[lower] = node_id
        for alias in (aliases or []):
            al = alias.lower()
            if al not in self._alias_lower:
                self._alias_lower[al] = node_id

    def lookup_exact(self, name: str) -> Optional[str]:
        return self._exact.get(name)

    def lookup_lower(self, name: str) -> Optional[str]:
        return self._lower.get(name.lower())

    def lookup_alias(self, name: str) -> Optional[str]:
        return self._alias_lower.get(name.lower())


_resolution_idx = _ResolutionIndex()


# ── Persistent dedup state (across ingest_claims calls) ───────────────
# Building these from scratch every call requires scanning all CLM nodes
# (~178K+ in a mature graph), which costs 30-60s per call. Cache them
# module-level so they survive between disease-year batches in the same
# process, and use the seeded-flag to know when a fresh seed is needed.
_persistent_seen_triples: set[tuple[str, str, str, str]] = set()
_persistent_seen_pairs: dict[tuple[str, str, str], str] = {}
_dedup_seeded: bool = False


def _seed_dedup_from_kg(kg: KnowledgeGraph):
    """Build cross-run dedup state from existing CLM nodes. Idempotent."""
    global _dedup_seeded
    if _dedup_seeded:
        return
    for node in kg._index.values():
        if not node.id.startswith("CLM:"):
            continue
        meta = node.metadata
        if not isinstance(meta, dict):
            continue
        pmid = ""
        sp = meta.get("source_paper")
        if isinstance(sp, dict):
            pmid = sp.get("pmid", "")
        sid = meta.get("subject_id", "")
        pred = meta.get("predicate", "")
        oid = meta.get("object_id", "")
        if sid and pred and oid:
            _persistent_seen_triples.add((pmid, sid, pred, oid))
            pair_key = (pmid, sid, oid)
            existing_pred = _persistent_seen_pairs.get(pair_key)
            if existing_pred is None or (existing_pred in _VAGUE_PREDICATES and pred not in _VAGUE_PREDICATES):
                _persistent_seen_pairs[pair_key] = pred
    _dedup_seeded = True
    logger.info(f"dedup state seeded: {len(_persistent_seen_triples):,} triples, {len(_persistent_seen_pairs):,} pairs")


def resolve_entity(
    kg: KnowledgeGraph,
    entity_name: str,
    entity_type: str = "",
) -> Optional[str]:
    """Resolve an entity name to a concept ID in the knowledge graph.

    Strategy:
    1. Exact match on preferred_name (O(1) via index)
    2. Case-insensitive match (O(1) via index)
    3. Alias match (O(1) via index)
    4. Safe fuzzy match: substring only when both strings are long enough,
       short aliases (<4 chars) require word-boundary match
    5. If entity_type is given, prefer candidates whose domain matches
    6. If not found, create a new concept node
    """
    if not entity_name:
        return None

    # Build index on first call
    if not _resolution_idx._built:
        _resolution_idx.build(kg)

    # 1. exact match (O(1))
    hit = _resolution_idx.lookup_exact(entity_name)
    if hit:
        return hit

    # 2. case-insensitive match (O(1))
    entity_lower = entity_name.lower()
    hit = _resolution_idx.lookup_lower(entity_name)
    if hit:
        return hit

    # 3. alias match (O(1))
    hit = _resolution_idx.lookup_alias(entity_name)
    if hit:
        return hit

    # 4. safe fuzzy match (still O(n) but only reached for cache misses)
    candidates = []
    entity_len = len(entity_lower)
    for node in kg._index.values():
        name_lower = node.preferred_name.lower()
        if entity_len >= _MIN_SUBSTRING_LEN and len(name_lower) >= _MIN_SUBSTRING_LEN:
            if entity_lower in name_lower or name_lower in entity_lower:
                candidates.append(node)
                continue
        for alias in node.aliases:
            alias_lower = alias.lower()
            if len(alias_lower) < _MIN_SUBSTRING_LEN:
                if _word_boundary_match(alias_lower, entity_lower):
                    candidates.append(node)
                    break
            else:
                if entity_lower in alias_lower or alias_lower in entity_lower:
                    candidates.append(node)
                    break

    # 5. prefer domain-matching candidate when entity_type is known
    expected_domain = ENTITY_TYPE_TO_DOMAIN.get(entity_type) if entity_type else None
    if candidates and expected_domain is not None:
        typed = [c for c in candidates if expected_domain.value in c.domain_tags]
        if typed:
            candidates = typed

    if len(candidates) == 1:
        return candidates[0].id
    elif len(candidates) > 1:
        candidates.sort(key=lambda n: len(n.preferred_name))
        return candidates[0].id

    # 6. not found — noise check before minting a brand-new CLM_CONCEPT.
    # Curated matches (steps 1-5) already returned above, so we only see
    # names that would otherwise pollute the graph with auto-generated
    # low-quality nodes. First try to salvage by stripping noise affixes
    # and rechecking the curated index; if still noise, drop the entity.
    if _NOISE_FILTER_ENABLED and _is_noisy_name(entity_name):
        salvaged = _salvage_noisy_name(entity_name)
        if salvaged:
            salvaged_lower = salvaged.lower()
            for node in kg._index.values():
                if node.preferred_name.lower() == salvaged_lower:
                    _DROP_LOG.record(entity_name, "salvaged", salvaged, node.id)
                    return node.id
                for alias in node.aliases:
                    if alias.lower() == salvaged_lower:
                        _DROP_LOG.record(entity_name, "salvaged", salvaged, node.id)
                        return node.id
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
            salvaged_lower = salvaged.lower()
            for node in kg._index.values():
                if node.preferred_name.lower() == salvaged_lower:
                    _DROP_LOG.record(entity_name, "salvaged_strict", salvaged, node.id)
                    return node.id
                for alias in node.aliases:
                    if alias.lower() == salvaged_lower:
                        _DROP_LOG.record(entity_name, "salvaged_strict", salvaged, node.id)
                        return node.id
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

    # 6d. Dedup by preferred_name: if an existing CLM_CONCEPT already has
    # the same name (case-insensitive), reuse it instead of minting a
    # duplicate with a different ID.
    for node in kg._index.values():
        if node.id.startswith("CLM_CONCEPT:") and node.preferred_name.lower() == entity_lower:
            return node.id

    # 7. mint a new concept
    domain = ENTITY_TYPE_TO_DOMAIN.get(entity_type, DomainTag.DISEASE)
    new_id = f"CLM_CONCEPT:{entity_name.replace(' ', '_')}"
    kg.add_concept(ConceptNode(
        id=new_id,
        preferred_name=entity_name,
        domain_tags=[domain.value],
        source_vocab="claim_extraction",
    ))
    _resolution_idx.add(new_id, entity_name)
    logger.info(f"created new concept: {new_id} ({entity_name})")
    return new_id


def resolve_claim_entities(
    kg: KnowledgeGraph,
    claim: Claim,
) -> Claim:
    """Resolve subject and object names to concept IDs."""
    subject_id = resolve_entity(kg, claim.subject_name, claim.metadata.get("subject_type", ""))
    object_id = resolve_entity(kg, claim.object_name, claim.metadata.get("object_type", ""))

    if subject_id:
        claim.subject_id = subject_id
    if object_id:
        claim.object_id = object_id

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
        drop_log_path: override path for the dropped-entities audit log.

    Returns summary dict.
    """
    claims_added = 0
    edges_added = 0
    errors = 0
    claims_skipped_noise = 0
    claims_skipped_unresolved = 0
    predicates_refined = 0

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
    for result in results:
        if result.error:
            errors += 1
            continue
        all_claims.extend(result.claims)

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
    # Triple dedup: skip claims whose (PMID, subject_id, predicate, object_id)
    # has already been ingested in this batch or exists in the graph.
    # Use persistent module-level state to avoid rescanning all CLM nodes
    # on every batch (would cost 30-60s per disease-year on a mature KG).
    _seed_dedup_from_kg(kg)
    _seen_triples = _persistent_seen_triples
    _seen_pairs = _persistent_seen_pairs
    claims_skipped_dedup = 0

    logger.debug(f"triple dedup state: {len(_seen_triples)} triples")

    for claim in all_claims:
        try:
            # resolve entities
            claim = resolve_claim_entities(kg, claim)

            if not claim.subject_id or not claim.object_id:
                # Distinguish noise drop, strict_phase1 drop, and real error
                if _STRICT_PHASE1:
                    claims_skipped_unresolved += 1
                    logger.debug(
                        f"strict_phase1 skipped claim {claim.id}: "
                        f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}"
                    )
                elif _NOISE_FILTER_ENABLED and (
                    _is_noisy_name(claim.subject_name)
                    or _is_noisy_name(claim.object_name)
                ):
                    claims_skipped_noise += 1
                    logger.debug(
                        f"skipped noise claim {claim.id}: "
                        f"{claim.subject_name!r} {claim.predicate} {claim.object_name!r}"
                    )
                else:
                    logger.warning(f"could not resolve entities for claim {claim.id}")
                    errors += 1
                continue

            # Triple dedup: same paper + same (subject, predicate, object) = duplicate
            pmid = claim.source_paper.pmid or ""
            triple_key = (pmid, claim.subject_id, claim.predicate, claim.object_id)
            if triple_key in _seen_triples:
                claims_skipped_dedup += 1
                logger.debug(
                    f"dedup skipped claim {claim.id}: "
                    f"({pmid}, {claim.subject_id}, {claim.predicate}, {claim.object_id})"
                )
                continue
            _seen_triples.add(triple_key)

            # Cross-predicate PMID dedup: same paper + same (subject, object)
            # but different predicate. Keep only the most precise one.
            pair_key = (pmid, claim.subject_id, claim.object_id)
            existing_pred = _seen_pairs.get(pair_key)
            if existing_pred is not None:
                if claim.predicate in _VAGUE_PREDICATES and existing_pred not in _VAGUE_PREDICATES:
                    # Already have a precise predicate, skip this vague one
                    claims_skipped_dedup += 1
                    logger.debug(
                        f"cross-pred dedup skipped vague {claim.id}: "
                        f"already have {existing_pred} for pair"
                    )
                    continue
            _seen_pairs[pair_key] = claim.predicate

            # add claim node
            kg.add_concept(ConceptNode(
                id=claim.id,
                preferred_name=f"{claim.subject_name} {claim.predicate} {claim.object_name}",
                domain_tags=["claim"],
                source_vocab="claim_extraction",
                definition=claim.raw_text,
                metadata=claim.to_dict(),
            ))
            claims_added += 1

            # Predicate-evidence confidence penalty: if raw_text doesn't
            # contain keywords supporting the predicate, reduce confidence.
            if not _predicate_supported_by_text(claim.predicate, claim.raw_text):
                claim.confidence *= _UNSUPPORTED_PREDICATE_PENALTY

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

        except Exception as e:
            logger.warning(f"failed to ingest claim {claim.id}: {e}")
            errors += 1

    summary = {
        "claims_added": claims_added,
        "edges_added": edges_added,
        "errors": errors,
        "claims_skipped_noise": claims_skipped_noise,
        "claims_skipped_unresolved": claims_skipped_unresolved,
        "claims_skipped_dedup": claims_skipped_dedup,
        "entities_salvaged": _DROP_LOG.n_salvaged,
        "entities_dropped": _DROP_LOG.n_dropped,
        "papers_processed": len(results),
        "predicates_refined": predicates_refined,
        "strict_phase1": strict_phase1,
    }
    _DROP_LOG.close()
    logger.info(f"claim ingestion complete: {summary}")
    return summary
