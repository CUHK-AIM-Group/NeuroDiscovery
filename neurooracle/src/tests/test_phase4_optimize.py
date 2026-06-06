"""Tests for conservative duplicate merging in phase4_optimize."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.phase4_optimize import (
    collect_same_prefix_same_domain_duplicate_groups,
    merge_duplicate_concepts,
)
from neurooracle.src.schema import ConceptNode, DomainTag, Edge


def _build_duplicate_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()

    kg.add_concept(ConceptNode(
        id="NN:seed",
        preferred_name="Seed Region",
        domain_tags=[DomainTag.NEUROANATOMY.value],
        source_vocab="test",
    ))
    kg.add_concept(ConceptNode(
        id="MSH:target",
        preferred_name="Target Disease",
        domain_tags=[DomainTag.DISEASE.value],
        source_vocab="test",
    ))

    # Safe internal Cognitive Atlas duplicate: should merge.
    kg.add_concept(ConceptNode(
        id="COGAT_TASK:tsk_primary",
        preferred_name="Boston Naming Test",
        domain_tags=[DomainTag.PARADIGM.value],
        source_vocab="CognitiveAtlas",
        aliases=["BNT"],
    ))
    kg.add_concept(ConceptNode(
        id="COGAT_TASK:tsk_dup",
        preferred_name="Boston Naming Test",
        domain_tags=[DomainTag.PARADIGM.value],
        source_vocab="CognitiveAtlas",
        aliases=["Boston naming"],
    ))

    # Same-prefix/same-domain but unsafe: distinct CUIs with same label.
    kg.add_concept(ConceptNode(
        id="CUI:C1",
        preferred_name="Major Depressive Disorder",
        domain_tags=[DomainTag.DISEASE.value],
        source_vocab="MeSH",
    ))
    kg.add_concept(ConceptNode(
        id="CUI:C2",
        preferred_name="Major Depressive Disorder",
        domain_tags=[DomainTag.DISEASE.value],
        source_vocab="CognitiveAtlas",
    ))

    # Same-prefix/same-domain but unsafe: atlas-specific NeuroNames ROIs.
    kg.add_concept(ConceptNode(
        id="NN:NN_HO:20036",
        preferred_name="Lingual Gyrus",
        domain_tags=[DomainTag.NEUROANATOMY.value],
        source_vocab="NeuroNames",
    ))
    kg.add_concept(ConceptNode(
        id="NN:NN_TAL:10044",
        preferred_name="Lingual Gyrus",
        domain_tags=[DomainTag.NEUROANATOMY.value],
        source_vocab="NeuroNames",
    ))

    kg.add_edge(Edge(
        source_id="NN:seed",
        target_id="COGAT_TASK:tsk_primary",
        relation_type="activates",
        source="test",
        confidence=0.6,
    ))
    kg.add_edge(Edge(
        source_id="COGAT_TASK:tsk_primary",
        target_id="MSH:target",
        relation_type="associated_with",
        source="test",
        confidence=0.7,
    ))
    kg.add_edge(Edge(
        source_id="COGAT_TASK:tsk_dup",
        target_id="MSH:target",
        relation_type="associated_with",
        source="test",
        confidence=0.8,
    ))
    return kg


def test_collect_same_prefix_same_domain_duplicate_groups_marks_safe_vs_blocked():
    kg = _build_duplicate_graph()

    groups = collect_same_prefix_same_domain_duplicate_groups(kg)
    by_name = {group["preferred_name"].lower(): group for group in groups}

    assert len(groups) == 3

    bnt = by_name["boston naming test"]
    assert bnt["prefix"] == "COGAT_TASK"
    assert bnt["safe_to_merge"] is True
    assert bnt["rationale"] == "safe_cogat_internal_duplicate"

    mdd = by_name["major depressive disorder"]
    assert mdd["prefix"] == "CUI"
    assert mdd["safe_to_merge"] is False
    assert mdd["rationale"] == "blocked_distinct_umls_cuis_same_label"

    lingual = by_name["lingual gyrus"]
    assert lingual["prefix"] == "NN"
    assert lingual["safe_to_merge"] is False
    assert lingual["rationale"] == "blocked_atlas_specific_neuroanatomy"


def test_merge_duplicate_concepts_only_merges_safe_groups():
    kg = _build_duplicate_graph()

    merged = merge_duplicate_concepts(kg)

    assert merged == 1
    assert "COGAT_TASK:tsk_primary" in kg._index
    assert "COGAT_TASK:tsk_dup" not in kg._index

    canonical = kg.get_concept("COGAT_TASK:tsk_primary")
    assert canonical is not None
    assert "Boston naming" in canonical.aliases
    assert canonical.metadata["dedup_rule"] == "conservative_same_prefix_same_domain_v1"
    assert canonical.metadata["dedup_merged_ids"] == ["COGAT_TASK:tsk_dup"]

    assert kg.G.has_edge("NN:seed", "COGAT_TASK:tsk_primary")
    assert kg.G.has_edge("COGAT_TASK:tsk_primary", "MSH:target")
    assert kg.G.edges["COGAT_TASK:tsk_primary", "MSH:target"]["confidence"] == 0.8

    # Unsafe duplicate groups remain untouched.
    assert "CUI:C1" in kg._index and "CUI:C2" in kg._index
    assert "NN:NN_HO:20036" in kg._index and "NN:NN_TAL:10044" in kg._index
