"""Claim ingestion: resolve entities, refine predicates, add claims to knowledge graph."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import OpenAI

from .claim_extractor import ClaimExtractor, ExtractionResult
from .graph_manager import KnowledgeGraph
from .schema import CLAIM_PREDICATES, Claim, ConceptNode, DomainTag, Edge, PaperRef

logger = logging.getLogger(__name__)

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


def resolve_entity(
    kg: KnowledgeGraph,
    entity_name: str,
    entity_type: str = "",
) -> Optional[str]:
    """Resolve an entity name to a concept ID in the knowledge graph.

    Strategy:
    1. Exact match on preferred_name
    2. Case-insensitive match
    3. Alias match
    4. Fuzzy substring match
    5. If not found, create a new concept node
    """
    if not entity_name:
        return None

    # 1. exact match
    for node in kg._index.values():
        if node.preferred_name == entity_name:
            return node.id

    # 2. case-insensitive match
    entity_lower = entity_name.lower()
    for node in kg._index.values():
        if node.preferred_name.lower() == entity_lower:
            return node.id

    # 3. alias match
    for node in kg._index.values():
        for alias in node.aliases:
            if alias.lower() == entity_lower:
                return node.id

    # 4. substring match (entity contained in name or vice versa)
    candidates = []
    for node in kg._index.values():
        name_lower = node.preferred_name.lower()
        if entity_lower in name_lower or name_lower in entity_lower:
            candidates.append(node)
        for alias in node.aliases:
            if entity_lower in alias.lower() or alias.lower() in entity_lower:
                candidates.append(node)
                break

    if len(candidates) == 1:
        return candidates[0].id
    elif len(candidates) > 1:
        # prefer shortest name (most specific match)
        candidates.sort(key=lambda n: len(n.preferred_name))
        return candidates[0].id

    # 5. not found — create new concept
    domain = ENTITY_TYPE_TO_DOMAIN.get(entity_type, DomainTag.DISEASE)
    new_id = f"CLM_CONCEPT:{entity_name.replace(' ', '_')}"
    kg.add_concept(ConceptNode(
        id=new_id,
        preferred_name=entity_name,
        domain_tags=[domain.value],
        source_vocab="claim_extraction",
    ))
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
) -> dict:
    """Ingest extracted claims into the knowledge graph.

    For each claim:
    1. Resolve subject/object to existing concepts (or create new ones)
    2. Refine vague predicates (RELATE: is_associated_with → precise predicate)
    3. Add a Claim node with full metadata
    4. Add a simplified edge for traversal

    Returns summary dict.
    """
    claims_added = 0
    edges_added = 0
    errors = 0
    predicates_refined = 0

    # initialize LLM client for predicate refinement if needed
    # Uses first key from OPENAI_API_KEYS pool, falls back to OPENAI_API_KEY
    llm_client = None
    if refine_vague_predicates:
        base_url = llm_base_url or os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
        keys_raw = os.environ.get("OPENAI_API_KEYS", "")
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        api_key = llm_api_key or (keys[0] if keys else os.environ.get("OPENAI_API_KEY", ""))
        model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
        if api_key:
            import httpx
            llm_client = OpenAI(base_url=base_url, api_key=api_key, http_client=httpx.Client(verify=False))

    for result in results:
        if result.error:
            errors += 1
            continue

        for claim in result.claims:
            try:
                # resolve entities
                claim = resolve_claim_entities(kg, claim)

                # refine vague predicates (RELATE)
                if refine_vague_predicates:
                    original_pred = claim.predicate
                    claim.predicate = refine_predicate(claim, llm_client, model)
                    if claim.predicate != original_pred:
                        predicates_refined += 1

                if not claim.subject_id or not claim.object_id:
                    logger.warning(f"could not resolve entities for claim {claim.id}")
                    errors += 1
                    continue

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
        "papers_processed": len(results),
        "predicates_refined": predicates_refined,
    }
    logger.info(f"claim ingestion complete: {summary}")
    return summary
