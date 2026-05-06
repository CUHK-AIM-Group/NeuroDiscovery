"""Claim ingestion: resolve entities, add claims to knowledge graph."""

from __future__ import annotations

import logging
from typing import Optional

from .claim_extractor import ClaimExtractor, ExtractionResult
from .graph_manager import KnowledgeGraph
from .schema import Claim, ConceptNode, DomainTag, Edge, PaperRef

logger = logging.getLogger(__name__)

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
) -> dict:
    """Ingest extracted claims into the knowledge graph.

    For each claim:
    1. Resolve subject/object to existing concepts (or create new ones)
    2. Add a Claim node with full metadata
    3. Add a simplified edge for traversal

    Returns summary dict.
    """
    claims_added = 0
    edges_added = 0
    errors = 0

    for result in results:
        if result.error:
            errors += 1
            continue

        for claim in result.claims:
            try:
                # resolve entities
                claim = resolve_claim_entities(kg, claim)

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
    }
    logger.info(f"claim ingestion complete: {summary}")
    return summary
