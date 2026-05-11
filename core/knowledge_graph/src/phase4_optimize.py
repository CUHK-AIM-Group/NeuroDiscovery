"""Phase 4: Knowledge Graph Quality Optimization.

Merges duplicate concepts, adds bridge edges, and applies
evidence quality weighting to improve hypothesis generation.

Usage:
    python -m core.knowledge_graph.phase4_optimize
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from .graph_manager import KnowledgeGraph
from .storage import load_graph, save_graph

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
GRAPH_FILE = DATA_DIR / "knowledge_graph.json"


# ── 1. Merge Duplicate Concepts ─────────────────────────────────────

def merge_duplicate_concepts(kg: KnowledgeGraph) -> int:
    """Merge concepts with the same preferred_name (case-insensitive).

    Strategy: keep the node with the most edges as canonical, redirect
    all edges from duplicates to canonical, then remove duplicates.

    Returns number of merges performed.
    """
    # Group by lowercase name
    name_groups: dict[str, list[str]] = defaultdict(list)
    for nid, node in kg._index.items():
        if "claim" not in node.domain_tags:
            name_groups[node.preferred_name.lower().strip()].append(nid)

    merges = 0
    for name, nids in name_groups.items():
        if len(nids) < 2:
            continue

        # Pick canonical: node with highest degree
        canonical = max(nids, key=lambda n: kg.G.degree(n) if n in kg.G else 0)
        duplicates = [n for n in nids if n != canonical]

        for dup_id in duplicates:
            if dup_id not in kg.G:
                continue

            # Redirect all edges from duplicate to canonical
            # Outgoing edges
            for _, target, data in list(kg.G.out_edges(dup_id, data=True)):
                if target == canonical:
                    continue
                if not kg.G.has_edge(canonical, target):
                    kg.G.add_edge(canonical, target, **data)
                else:
                    # Merge: keep higher confidence edge
                    existing = kg.G.edges[canonical, target]
                    if data.get("confidence", 0) > existing.get("confidence", 0):
                        for k, v in data.items():
                            kg.G.edges[canonical, target][k] = v

            # Incoming edges
            for source, _, data in list(kg.G.in_edges(dup_id, data=True)):
                if source == canonical:
                    continue
                if not kg.G.has_edge(source, canonical):
                    kg.G.add_edge(source, canonical, **data)
                else:
                    existing = kg.G.edges[source, canonical]
                    if data.get("confidence", 0) > existing.get("confidence", 0):
                        for k, v in data.items():
                            kg.G.edges[source, canonical][k] = v

            # Merge aliases
            dup_node = kg._index.get(dup_id)
            can_node = kg._index.get(canonical)
            if dup_node and can_node:
                for alias in dup_node.aliases:
                    if alias not in can_node.aliases:
                        can_node.aliases.append(alias)
                # Merge metadata
                for k, v in dup_node.metadata.items():
                    if k not in can_node.metadata:
                        can_node.metadata[k] = v

            # Remove duplicate node
            kg.G.remove_node(dup_id)
            if dup_id in kg._index:
                del kg._index[dup_id]
            merges += 1

    logger.info(f"merged {merges} duplicate concepts")
    return merges


def merge_duplicate_claims(kg: KnowledgeGraph) -> int:
    """Merge claim nodes that share (subject_id, predicate, object_id).

    The same factual claim ("PSEN1 causes AD") often gets re-extracted
    from 30+ different papers, producing 30+ distinct CLM:* nodes. This
    breaks `compute_frequency_boost` which counts PMIDs on the canonical
    edge — with claims scattered, each claim has 1 PMID.

    After merge:
    - Canonical = claim with highest metadata.confidence (tie: most
      complete evidence fields, tie: earliest id alphabetically)
    - canonical.metadata['supporting_papers'] = [all PMIDs]
    - canonical.metadata['n_supporting'] = len(supporting_papers)
    - All 'about' edges redirect to canonical
    - All semantic edges (subject->object) keep canonical's confidence
      (highest), but we bump it by freq_boost based on n_supporting

    Returns number of claims merged away.
    """
    # Group claim nodes by their SPO signature
    from collections import defaultdict
    groups: dict[tuple, list[str]] = defaultdict(list)
    for cid, node in kg._index.items():
        if "claim" not in node.domain_tags:
            continue
        if node.source_vocab != "claim_extraction":
            continue
        md = node.metadata
        s = md.get("subject_id", "")
        p = md.get("predicate", "")
        o = md.get("object_id", "")
        if not (s and p and o):
            continue
        groups[(s, p, o)].append(cid)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    logger.info(f"found {len(dup_groups)} claim SPO groups with duplicates "
                f"({sum(len(v) for v in dup_groups.values())} claim nodes total)")

    merged = 0
    for (s, p, o), claim_ids in dup_groups.items():
        # Pick canonical: highest claim.metadata.confidence, tiebreak by id
        def rank_key(cid):
            n = kg._index[cid]
            conf = n.metadata.get("confidence", 0.5)
            # prefer claims with more evidence fields populated
            ev = n.metadata.get("evidence", {})
            richness = sum(1 for k in ("p_value", "sample_size", "effect_size",
                                        "study_type", "methodology") if ev.get(k))
            return (-conf, -richness, cid)

        claim_ids.sort(key=rank_key)
        canonical_id = claim_ids[0]
        canonical = kg._index[canonical_id]
        duplicates = claim_ids[1:]

        # Collect supporting PMIDs from all claims in the group
        # Separate lists: all PMIDs vs primary-only (excluding review types).
        # compute_frequency_boost expects primary counts to award 1.2× bonus —
        # counting review PMIDs there would inflate the boost artificially.
        REVIEW_TYPES = {"review", "narrative_review", "systematic_review"}
        supporting = []
        primary_supporting = []
        seen_pmids = set()
        seen_primary = set()
        for cid in claim_ids:
            node = kg._index[cid]
            paper = node.metadata.get("source_paper", {})
            pmid = paper.get("pmid") if isinstance(paper, dict) else None
            if not pmid:
                continue
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                supporting.append(pmid)
            study_type = node.metadata.get("evidence", {}).get("study_type", "")
            if study_type not in REVIEW_TYPES and pmid not in seen_primary:
                seen_primary.add(pmid)
                primary_supporting.append(pmid)

        canonical.metadata["supporting_papers"] = supporting
        canonical.metadata["n_supporting"] = len(supporting)
        canonical.metadata["primary_supporting_papers"] = primary_supporting
        canonical.metadata["n_primary_supporting"] = len(primary_supporting)

        # Redirect incoming + outgoing edges from duplicates to canonical,
        # then remove the duplicate node.
        for dup_id in duplicates:
            if dup_id not in kg.G:
                continue
            # incoming
            for source, _, data in list(kg.G.in_edges(dup_id, data=True)):
                if source == canonical_id:
                    continue
                if not kg.G.has_edge(source, canonical_id):
                    kg.G.add_edge(source, canonical_id, **data)
            # outgoing
            for _, target, data in list(kg.G.out_edges(dup_id, data=True)):
                if target == canonical_id:
                    continue
                if not kg.G.has_edge(canonical_id, target):
                    kg.G.add_edge(canonical_id, target, **data)
            kg.G.remove_node(dup_id)
            if dup_id in kg._index:
                del kg._index[dup_id]
            merged += 1

    logger.info(f"merged {merged} duplicate claim nodes into canonical claims")
    return merged


# ── 2. Add Bridge Edges ─────────────────────────────────────────────

def add_bridge_edges(kg: KnowledgeGraph) -> int:
    """Add bridge edges between disconnected components using shared UMLS CUIs
    or shared domain tags.

    Returns number of bridge edges added.
    """
    components = list(nx.weakly_connected_components(kg.G))
    if len(components) < 2:
        logger.info("graph is already fully connected")
        return 0

    # Sort by size (largest first)
    components.sort(key=len, reverse=True)
    largest = components[0]
    bridges = 0

    # Strategy 1: Connect nodes with same UMLS CUI across components
    cui_to_nodes: dict[str, list[str]] = defaultdict(list)
    for nid, node in kg._index.items():
        cui = node.metadata.get("umls_cui")
        if cui and "claim" not in node.domain_tags:
            cui_to_nodes[cui].append(nid)

    for cui, nids in cui_to_nodes.items():
        # Find which components these nodes belong to
        comp_map: dict[int, list[str]] = defaultdict(list)
        for nid in nids:
            for ci, comp in enumerate(components[:20]):  # check top 20 components
                if nid in comp:
                    comp_map[ci].append(nid)
                    break

        if len(comp_map) >= 2:
            # Connect nodes from different components
            comp_ids = list(comp_map.keys())
            for i in range(len(comp_ids) - 1):
                src = comp_map[comp_ids[i]][0]
                tgt = comp_map[comp_ids[i + 1]][0]
                if not kg.G.has_edge(src, tgt) and not kg.G.has_edge(tgt, src):
                    kg.G.add_edge(src, tgt, relation_type="same_umls_cui",
                                  confidence=0.8, source="bridge_cui",
                                  source_id=src, target_id=tgt)
                    bridges += 1

    # Strategy 2: Connect brain regions to diseases via shared claims
    brain_nodes = [nid for nid, n in kg._index.items()
                   if "neuroanatomy" in n.domain_tags and "claim" not in n.domain_tags]
    disease_nodes = [nid for nid, n in kg._index.items()
                     if "disease" in n.domain_tags and "claim" not in n.domain_tags]

    # For each disconnected disease, find if it has claims linking to brain regions
    for disease_id in disease_nodes[:100]:  # limit to avoid too many
        if disease_id not in kg.G:
            continue
        # Check if in a small component
        for ci, comp in enumerate(components[1:], 1):  # skip largest
            if disease_id in comp and len(comp) < 50:
                # Find nearby brain regions via claims
                for neighbor in list(kg.G.predecessors(disease_id)) + list(kg.G.successors(disease_id)):
                    neighbor_node = kg._index.get(neighbor)
                    if neighbor_node and "neuroanatomy" in neighbor_node.domain_tags:
                        # Connect to a brain region in the largest component
                        for brain_id in brain_nodes[:20]:
                            if brain_id in largest and not kg.G.has_edge(brain_id, disease_id):
                                brain_name = kg._index[brain_id].preferred_name.lower()
                                if any(kw in brain_name for kw in ["cortex", "hippocampus", "amygdala",
                                                                    "thalamus", "striatum", "insula"]):
                                    kg.G.add_edge(brain_id, disease_id,
                                                  relation_type="bridge_connect",
                                                  confidence=0.3, source="bridge_domain",
                                                  source_id=brain_id, target_id=disease_id)
                                    bridges += 1
                                    break
                        break

    logger.info(f"added {bridges} bridge edges")
    return bridges


# ── 3. Filter Isolated Nodes ────────────────────────────────────────

def remove_isolated_claims(kg: KnowledgeGraph) -> int:
    """Remove claim nodes with no edges (orphaned claims)."""
    to_remove = []
    for nid in list(kg.G.nodes()):
        node = kg._index.get(nid)
        if node and "claim" in node.domain_tags and kg.G.degree(nid) == 0:
            to_remove.append(nid)

    for nid in to_remove:
        kg.G.remove_node(nid)
        if nid in kg._index:
            del kg._index[nid]

    logger.info(f"removed {len(to_remove)} isolated claim nodes")
    return len(to_remove)


# ── 4. Evidence Quality Weighting ───────────────────────────────────

def apply_evidence_weighting(kg: KnowledgeGraph) -> int:
    """Downweight edges from review papers, boost edges with statistical evidence.

    Adjusts edge confidence based on study_type.
    Returns number of edges modified.

    This function is idempotent: it resets edge confidence to the original
    claim confidence (stored on the claim ConceptNode) before applying the
    current weight. This ensures that re-running with different weights
    produces consistent results regardless of prior runs.

    Weight rationale:
    - Reviews/narrative reviews comprise ~85% of our claims (text-only extraction
      picks up heavy review literature). The prior weights of 0.3 / 0.2 crushed
      85% of the knowledge base down to conf<0.05, poisoning path-finding.
    - Systematic reviews aggregate multiple primary studies and represent GRADE
      top-tier evidence — they should score HIGHER than individual case-control.
    - Narrative reviews still encode expert consensus on well-established facts
      (e.g. "PSEN1 mutations cause AD") — they shouldn't be marginalized.
    - Primary studies (RCT/longitudinal/cohort) remain highest-weighted so they
      still dominate when available.
    """
    WEIGHT_MAP = {
        # primary evidence (unchanged)
        "meta_analysis": 1.0,
        "systematic_review": 0.95,  # was 0.8 — GRADE A
        "clinical_trial": 0.9,
        "longitudinal": 0.85,
        "cohort": 0.85,
        "case_control": 0.8,
        # imaging modalities (unchanged)
        "PET": 0.8, "fMRI": 0.8, "EEG": 0.8, "sMRI": 0.8,
        "MEG": 0.8, "DTI": 0.8,
        "cross_sectional": 0.7,
        # translational / secondary
        "animal_model": 0.6,
        "lesion": 0.7,
        "case_report": 0.5,
        "GWAS": 0.85,  # large-n genetic evidence
        # reviews (boosted — they encode real consensus, often multi-study)
        "review": 0.65,             # was 0.3
        "narrative_review": 0.55,   # was 0.2
    }
    DEFAULT_WEIGHT = 0.7

    modified = 0
    for src, tgt, data in kg.G.edges(data=True):
        claim_id = data.get("metadata", {}).get("claim_id", "")
        if not claim_id:
            continue

        claim_node = kg._index.get(claim_id)
        if not claim_node:
            continue

        study_type = claim_node.metadata.get("evidence", {}).get("study_type", "")
        weight = WEIGHT_MAP.get(study_type, DEFAULT_WEIGHT)

        # Reset to claim-level confidence, then apply current weight.
        # This guarantees idempotence: running twice with the same WEIGHT_MAP
        # gives the same result, and changing the map takes effect on next run
        # without cumulative multiplication.
        base_conf = claim_node.metadata.get("confidence", 0.5)
        current_conf = data.get("confidence", 0.5)
        new_conf = min(base_conf * weight, 1.0)

        if abs(new_conf - current_conf) > 0.01:
            data["confidence"] = new_conf
            meta = data.setdefault("metadata", {})
            meta["evidence_weight"] = weight
            data["evidence_weight"] = weight  # also at top level for backward compat
            modified += 1

    logger.info(f"applied evidence weighting to {modified} edges")
    return modified


# ── Main ─────────────────────────────────────────────────────────────

def run_phase4():
    """Run full Phase 4 optimization pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 4: QUALITY OPTIMIZATION")
    logger.info("=" * 60)

    # Load graph
    kg = load_graph(GRAPH_FILE)
    stats_before = kg.stats()
    components_before = len(list(nx.weakly_connected_components(kg.G)))
    isolated_before = len([n for n in kg.G.nodes() if kg.G.degree(n) == 0])

    logger.info(f"Before: {stats_before['n_concepts']} concepts, {stats_before['n_edges']} edges")
    logger.info(f"Before: {components_before} components, {isolated_before} isolated nodes")

    # Step 1: Merge duplicates
    merges = merge_duplicate_concepts(kg)
    logger.info(f"After concept merge: {kg.stats()['n_concepts']} concepts")

    # Step 1b: Merge duplicate claim nodes by SPO (enables frequency_boost)
    claim_merges = merge_duplicate_claims(kg)
    logger.info(f"After claim merge: {kg.stats()['n_concepts']} concepts")

    # Step 2: Remove isolated claims
    removed = remove_isolated_claims(kg)
    logger.info(f"After cleanup: {kg.stats()['n_concepts']} concepts, {kg.stats()['n_edges']} edges")

    # Step 3: Add bridge edges
    bridges = add_bridge_edges(kg)
    logger.info(f"After bridges: {kg.stats()['n_edges']} edges")

    # Step 4: Evidence weighting
    weighted = apply_evidence_weighting(kg)

    # Save
    save_graph(kg, GRAPH_FILE)
    stats_after = kg.stats()
    components_after = len(list(nx.weakly_connected_components(kg.G)))
    isolated_after = len([n for n in kg.G.nodes() if kg.G.degree(n) == 0])

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Concepts: {stats_before['n_concepts']} -> {stats_after['n_concepts']}")
    logger.info(f"Edges: {stats_before['n_edges']} -> {stats_after['n_edges']}")
    logger.info(f"Components: {components_before} -> {components_after}")
    logger.info(f"Isolated: {isolated_before} -> {isolated_after}")
    logger.info(f"Merges: {merges}")
    logger.info(f"Claim merges: {claim_merges}")
    logger.info(f"Bridges: {bridges}")
    logger.info(f"Weighted: {weighted}")

    return {
        "concepts_before": stats_before["n_concepts"],
        "concepts_after": stats_after["n_concepts"],
        "edges_before": stats_before["n_edges"],
        "edges_after": stats_after["n_edges"],
        "components_before": components_before,
        "components_after": components_after,
        "isolated_before": isolated_before,
        "isolated_after": isolated_after,
        "merges": merges,
        "bridges": bridges,
        "weighted": weighted,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_phase4()
