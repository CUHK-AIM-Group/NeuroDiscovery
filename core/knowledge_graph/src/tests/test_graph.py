"""Tests for the knowledge graph module.

Run with: python -m pytest core/knowledge_graph/src/tests/test_graph.py -v
Or: python core/knowledge_graph/src/tests/test_graph.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# allow running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from core.knowledge_graph.src.graph_manager import KnowledgeGraph
from core.knowledge_graph.src.schema import ConceptNode, DomainTag, Edge
from core.knowledge_graph.src.storage import load_graph, save_graph


def _build_sample_graph() -> KnowledgeGraph:
    """Build a small sample neuroscience graph for testing."""
    kg = KnowledgeGraph()

    # neuroanatomy
    kg.add_concept(ConceptNode(
        id="NN:hippocampus", preferred_name="Hippocampus",
        domain_tags=[DomainTag.NEUROANATOMY.value], source_vocab="test",
        aliases=["Hippocampal formation"],
    ))
    kg.add_concept(ConceptNode(
        id="NN:amygdala", preferred_name="Amygdala",
        domain_tags=[DomainTag.NEUROANATOMY.value], source_vocab="test",
    ))
    kg.add_concept(ConceptNode(
        id="NN:pfc", preferred_name="Prefrontal Cortex",
        domain_tags=[DomainTag.NEUROANATOMY.value], source_vocab="test",
        aliases=["PFC", "Frontal cortex"],
    ))
    kg.add_concept(ConceptNode(
        id="NN:temporal_lobe", preferred_name="Temporal Lobe",
        domain_tags=[DomainTag.NEUROANATOMY.value], source_vocab="test",
    ))

    # diseases
    kg.add_concept(ConceptNode(
        id="MSH:alzheimer", preferred_name="Alzheimer Disease",
        domain_tags=[DomainTag.DISEASE.value], source_vocab="test",
    ))
    kg.add_concept(ConceptNode(
        id="MSH:depression", preferred_name="Major Depressive Disorder",
        domain_tags=[DomainTag.DISEASE.value], source_vocab="test",
    ))

    # genes
    kg.add_concept(ConceptNode(
        id="GENE:APOE", preferred_name="APOE",
        domain_tags=[DomainTag.GENE.value], source_vocab="test",
    ))
    kg.add_concept(ConceptNode(
        id="GENE:BDNF", preferred_name="BDNF",
        domain_tags=[DomainTag.GENE.value], source_vocab="test",
    ))

    # neurotransmitters
    kg.add_concept(ConceptNode(
        id="MSH:serotonin", preferred_name="Serotonin",
        domain_tags=[DomainTag.NEUROTRANSMITTER.value], source_vocab="test",
    ))

    # anatomical hierarchy
    kg.add_edge(Edge(source_id="NN:hippocampus", target_id="NN:temporal_lobe", relation_type="part_of", source="test"))
    kg.add_edge(Edge(source_id="NN:amygdala", target_id="NN:temporal_lobe", relation_type="part_of", source="test"))

    # connectivity
    kg.add_edge(Edge(source_id="NN:hippocampus", target_id="NN:pfc", relation_type="projects_to", source="test"))
    kg.add_edge(Edge(source_id="NN:amygdala", target_id="NN:pfc", relation_type="projects_to", source="test"))

    # gene-disease
    kg.add_edge(Edge(source_id="GENE:APOE", target_id="MSH:alzheimer", relation_type="gene_associated_with_disease", source="test", confidence=0.95))
    kg.add_edge(Edge(source_id="GENE:BDNF", target_id="MSH:depression", relation_type="gene_associated_with_disease", source="test", confidence=0.7))

    # disease-anatomy
    kg.add_edge(Edge(source_id="MSH:alzheimer", target_id="NN:hippocampus", relation_type="associated_with", source="test"))
    kg.add_edge(Edge(source_id="MSH:depression", target_id="NN:pfc", relation_type="associated_with", source="test"))

    # neurotransmitter modulates
    kg.add_edge(Edge(source_id="MSH:serotonin", target_id="NN:pfc", relation_type="modulates", source="test"))

    return kg


def test_basic_operations():
    """Test add/get/search concepts and edges."""
    kg = _build_sample_graph()

    assert len(kg) == 9
    assert kg.has_concept("NN:hippocampus")
    assert not kg.has_concept("NN:nonexistent")

    node = kg.get_concept("NN:hippocampus")
    assert node is not None
    assert node.preferred_name == "Hippocampus"
    assert "Hippocampal formation" in node.aliases

    # search by name
    results = kg.search_by_name("hippocampus")
    assert len(results) == 1
    assert results[0].id == "NN:hippocampus"

    # search by alias
    results = kg.search_by_name("PFC")
    assert len(results) == 1
    assert results[0].id == "NN:pfc"

    print("PASS: basic operations")


def test_neighbors():
    """Test neighbor queries."""
    kg = _build_sample_graph()

    # hippocampus has out-edges to temporal_lobe (part_of) and pfc (projects_to)
    out_neighbors = kg.get_neighbors("NN:hippocampus", direction="out")
    out_ids = [n for n, _ in out_neighbors]
    assert "NN:temporal_lobe" in out_ids
    assert "NN:pfc" in out_ids

    # hippocampus has in-edge from alzheimer (associated_with)
    in_neighbors = kg.get_neighbors("NN:hippocampus", direction="in")
    in_ids = [n for n, _ in in_neighbors]
    assert "MSH:alzheimer" in in_ids

    # filter by relation type
    part_of = kg.get_neighbors("NN:hippocampus", relation_type="part_of", direction="out")
    assert len(part_of) == 1
    assert part_of[0][0] == "NN:temporal_lobe"

    print("PASS: neighbors")


def test_paths():
    """Test path finding between concepts."""
    kg = _build_sample_graph()

    # APOE -> alzheimer -> hippocampus -> pfc  (3 hops)
    paths = kg.find_paths("GENE:APOE", "NN:pfc", max_hops=3)
    assert len(paths) > 0

    # find_paths returns [(node, relation), ...] so 3-hop = 4 entries (3 edges + terminal)
    found_3hop = any(len(p) == 4 for p in paths)
    assert found_3hop, f"expected a 3-hop path, got: {paths}"

    # BDNF -> depression -> pfc  (2 hops = 3 entries)
    paths = kg.find_paths("GENE:BDNF", "NN:pfc", max_hops=3)
    assert len(paths) > 0

    # serotonin -> pfc  (1 hop = 2 entries)
    paths = kg.find_paths("MSH:serotonin", "NN:pfc", max_hops=3)
    assert len(paths) > 0
    assert any(len(p) == 2 for p in paths)

    print("PASS: paths")


def test_domain_subgraph():
    """Test domain-based subgraph extraction."""
    kg = _build_sample_graph()

    neuro = kg.get_subgraph_by_domain(DomainTag.NEUROANATOMY.value)
    assert neuro.number_of_nodes() == 4  # hippocampus, amygdala, pfc, temporal_lobe

    disease = kg.get_subgraph_by_domain(DomainTag.DISEASE.value)
    assert disease.number_of_nodes() == 2

    print("PASS: domain subgraph")


def test_stats():
    """Test statistics."""
    kg = _build_sample_graph()
    stats = kg.stats()

    assert stats["n_concepts"] == 9
    assert stats["n_edges"] == 9
    assert DomainTag.NEUROANATOMY.value in stats["domains"]
    assert DomainTag.DISEASE.value in stats["domains"]

    print("PASS: stats")


def test_serialization():
    """Test JSON save/load roundtrip."""
    kg = _build_sample_graph()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_graph(kg, tmp_path)
        loaded = load_graph(tmp_path)

        assert len(loaded) == len(kg)
        assert loaded.stats()["n_edges"] == kg.stats()["n_edges"]

        # verify a specific node survived roundtrip
        node = loaded.get_concept("NN:hippocampus")
        assert node is not None
        assert node.preferred_name == "Hippocampus"
        assert "Hippocampal formation" in node.aliases

        # verify paths still work
        paths = loaded.find_paths("GENE:APOE", "NN:pfc", max_hops=3)
        assert len(paths) > 0

        print("PASS: serialization roundtrip")
    finally:
        tmp_path.unlink()


def test_node_merge():
    """Test that adding a concept with existing ID merges rather than overwrites."""
    kg = KnowledgeGraph()

    kg.add_concept(ConceptNode(
        id="NN:test", preferred_name="Test Region",
        domain_tags=[DomainTag.NEUROANATOMY.value],
        aliases=["alias1"],
    ))

    # add again with extra info
    kg.add_concept(ConceptNode(
        id="NN:test", preferred_name="Test Region",
        domain_tags=[DomainTag.DISEASE.value],
        aliases=["alias2"],
        definition="A test region",
    ))

    node = kg.get_concept("NN:test")
    assert node is not None
    assert "alias1" in node.aliases
    assert "alias2" in node.aliases
    assert DomainTag.NEUROANATOMY.value in node.domain_tags
    assert DomainTag.DISEASE.value in node.domain_tags
    assert node.definition == "A test region"

    print("PASS: node merge")


if __name__ == "__main__":
    test_basic_operations()
    test_neighbors()
    test_paths()
    test_domain_subgraph()
    test_stats()
    test_serialization()
    test_node_merge()
    print("\n=== ALL TESTS PASSED ===")
