"""Repair KG in place: remove edges poisoned by short-alias substring matches.

Bug: resolve_entity() substring match let 2-3 char aliases (like "IC" for
Internal Capsula, "AQ" for autism spectrum quotient) match any entity name
containing those letters (e.g. "specific molecules" matched "IC"). This
produced ~46% poisoned claims.

This script:
1. Identifies all concepts with short aliases (<4 chars)
2. For each claim concept whose subject/object resolved to a short-alias node,
   re-resolves the entity name using the FIXED resolve_entity logic
3. Updates claim metadata and rewrites the corresponding edges:
   - removes stale edges pointing to the wrongly-resolved node
   - adds new edges to the correctly-resolved (or newly created) node
4. Leaves the claim nodes and 'about' edges in place; only semantic edges
   and the claim's subject_id/object_id fields are corrected.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from .claim_ingestion import ENTITY_TYPE_TO_DOMAIN, resolve_entity
from .graph_manager import KnowledgeGraph
from .schema import Edge
from .storage import load_graph, save_graph

logger = logging.getLogger(__name__)


def find_short_alias_nodes(kg: KnowledgeGraph, min_len: int = 4) -> set[str]:
    """Return IDs of concepts whose preferred_name OR any alias is shorter than min_len.

    Both are problematic: the pre-fix resolve_entity's substring step (4) matched
    entity names against any short string, whether it was the preferred_name
    (e.g. "Ca", "p53") or an alias (e.g. "IC" for Internal Capsula).
    """
    ids = set()
    for c in kg._index.values():
        if len(c.preferred_name) < min_len:
            ids.add(c.id)
            continue
        for a in c.aliases:
            if len(a) < min_len:
                ids.add(c.id)
                break
    return ids


def _build_name_index(kg: KnowledgeGraph) -> dict[str, list[str]]:
    """Build a lowercase name/alias → concept_id index for fast lookup."""
    idx: dict[str, list[str]] = {}
    for cid, c in kg._index.items():
        keys = {c.preferred_name.lower()}
        for a in c.aliases:
            keys.add(a.lower())
        for k in keys:
            idx.setdefault(k, []).append(cid)
    return idx


def repair(kg: KnowledgeGraph, dry_run: bool = False) -> dict:
    """Repair phantom matches. Returns summary stats."""
    short_alias_nodes = find_short_alias_nodes(kg)
    logger.info(f"found {len(short_alias_nodes)} concepts with short (<4) aliases")

    name_index = _build_name_index(kg)
    logger.info(f"built name index: {len(name_index)} unique keys")

    claim_concepts = [
        c for c in kg._index.values()
        if "claim" in c.domain_tags and c.source_vocab == "claim_extraction"
    ]
    logger.info(f"scanning {len(claim_concepts)} claim concepts")

    stats = Counter()
    edges_to_remove: list[tuple[str, str]] = []
    edges_to_add: list[Edge] = []
    # Track metadata updates so we apply them after edge mutations
    meta_updates: list[tuple[str, dict]] = []
    progress_every = max(1, len(claim_concepts) // 20)

    for i, claim in enumerate(claim_concepts):
        if i % progress_every == 0:
            logger.info(
                f"  progress {i}/{len(claim_concepts)} "
                f"(detected={stats['phantom_matches_detected']}, "
                f"repaired={stats['claims_repaired']})"
            )
        md = claim.metadata or {}
        s_id = md.get("subject_id", "")
        o_id = md.get("object_id", "")
        s_name = md.get("subject_name", "")
        o_name = md.get("object_name", "")
        s_type = md.get("subject_type", "")
        o_type = md.get("object_type", "")
        predicate = md.get("predicate", "")
        confidence = md.get("confidence", 0.5)

        stats["claims_scanned"] += 1

        new_s_id = _maybe_reresolve(kg, s_id, s_name, s_type, short_alias_nodes, name_index, stats)
        new_o_id = _maybe_reresolve(kg, o_id, o_name, o_type, short_alias_nodes, name_index, stats)

        subject_changed = new_s_id and new_s_id != s_id
        object_changed = new_o_id and new_o_id != o_id

        if not (subject_changed or object_changed):
            continue

        stats["claims_repaired"] += 1

        # 1) remove old semantic edge (subject -> object with predicate)
        if s_id and o_id and kg.G.has_edge(s_id, o_id):
            existing = kg.G.edges[s_id, o_id]
            if existing.get("relation_type") == predicate:
                edges_to_remove.append((s_id, o_id))

        # 2) remove old 'about' edges (claim -> old_subject, claim -> old_object)
        if subject_changed and kg.G.has_edge(claim.id, s_id):
            edges_to_remove.append((claim.id, s_id))
        if object_changed and kg.G.has_edge(claim.id, o_id):
            edges_to_remove.append((claim.id, o_id))

        # 3) add new semantic edge (new_subject -> new_object with predicate)
        final_s = new_s_id or s_id
        final_o = new_o_id or o_id
        if final_s and final_o and predicate:
            edges_to_add.append(Edge(
                source_id=final_s,
                target_id=final_o,
                relation_type=predicate,
                source="claim_extraction",
                confidence=confidence,
            ))

        # 4) add new 'about' edges
        if subject_changed:
            edges_to_add.append(Edge(
                source_id=claim.id,
                target_id=final_s,
                relation_type="about",
                source="claim_extraction",
                confidence=confidence,
            ))
        if object_changed:
            edges_to_add.append(Edge(
                source_id=claim.id,
                target_id=final_o,
                relation_type="about",
                source="claim_extraction",
                confidence=confidence,
            ))

        # 5) queue metadata update
        updates = {}
        if subject_changed:
            updates["subject_id"] = final_s
        if object_changed:
            updates["object_id"] = final_o
        meta_updates.append((claim.id, updates))

    logger.info(
        f"repair plan: remove {len(edges_to_remove)} edges, "
        f"add {len(edges_to_add)} edges, update {len(meta_updates)} claim metadata"
    )

    if dry_run:
        stats["mode"] = "dry_run"
        return dict(stats)

    # Apply edge removals
    removed = 0
    for s, t in edges_to_remove:
        if kg.G.has_edge(s, t):
            kg.G.remove_edge(s, t)
            removed += 1
    stats["edges_removed"] = removed
    logger.info(f"removed {removed} edges")

    # Apply edge additions with progress logging
    before = kg.G.number_of_edges()
    n_add = len(edges_to_add)
    log_every = max(1, n_add // 20)
    for i, e in enumerate(edges_to_add):
        if i % log_every == 0:
            logger.info(f"  adding edges {i}/{n_add}")
        kg.add_edge(e)
    stats["edges_added"] = kg.G.number_of_edges() - before
    logger.info(f"added {stats['edges_added']} new edges (skipped {n_add - stats['edges_added']} dupes)")

    # Apply metadata updates
    for cid, updates in meta_updates:
        node = kg._index.get(cid)
        if node is None:
            continue
        node.metadata.update(updates)
        # keep G node attributes in sync
        if cid in kg.G.nodes:
            kg.G.nodes[cid].update(node.to_dict())
    stats["metadata_updated"] = len(meta_updates)

    return dict(stats)


def _maybe_reresolve(
    kg: KnowledgeGraph,
    current_id: str,
    entity_name: str,
    entity_type: str,
    short_alias_nodes: set[str],
    name_index: dict[str, list[str]],
    stats: Counter,
) -> str:
    """If current_id was likely a phantom short-alias match, re-resolve.

    Returns the new resolved id (may equal current_id if still correct, or
    may be a different existing/new node). Uses pre-built name_index for speed.
    """
    if not entity_name or not current_id:
        return ""
    if current_id not in short_alias_nodes:
        return ""  # not a suspect — leave unchanged

    node = kg._index.get(current_id)
    if node is None:
        return ""

    # Check: does entity_name match the current node by exact/case/alias-exact?
    en = entity_name.lower()
    pname = node.preferred_name.lower()
    if en == pname:
        return current_id  # fine
    if any(en == a.lower() for a in node.aliases):
        return current_id  # fine (legitimate alias match)
    # Long substring match both ways is still acceptable under the new policy
    if len(en) >= 4 and len(pname) >= 4:
        if en in pname or pname in en:
            return current_id  # fine

    # This was a phantom short-alias substring match. Re-resolve using index.
    stats["phantom_matches_detected"] += 1

    # Fast path: exact (case-insensitive) match via index
    if en in name_index:
        new_id = name_index[en][0]
        if new_id != current_id:
            stats["phantom_matches_rerouted"] += 1
        return new_id

    # Otherwise: create a new claim concept (same as resolve_entity step 5)
    new_id = f"CLM_CONCEPT:{entity_name.replace(' ', '_')}"
    if new_id not in kg._index:
        from .schema import ConceptNode, DomainTag
        domain = ENTITY_TYPE_TO_DOMAIN.get(entity_type, DomainTag.DISEASE)
        kg.add_concept(ConceptNode(
            id=new_id,
            preferred_name=entity_name,
            domain_tags=[domain.value],
            source_vocab="claim_extraction",
        ))
        # keep name_index fresh
        name_index.setdefault(en, []).append(new_id)
        stats["new_concepts_created"] += 1
    stats["phantom_matches_rerouted"] += 1
    return new_id


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Repair short-alias phantom matches in KG")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to knowledge_graph.json (defaults to configured location)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (defaults to overwriting input)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without modifying the graph")
    args = parser.parse_args()

    kg = load_graph(args.input)
    logger.info(f"loaded KG: {len(kg._index)} concepts, {kg.G.number_of_edges()} edges")

    summary = repair(kg, dry_run=args.dry_run)
    logger.info(f"repair summary: {summary}")

    if not args.dry_run:
        out = args.output or args.input
        save_graph(kg, out)
        logger.info(f"saved repaired KG to {out}")
