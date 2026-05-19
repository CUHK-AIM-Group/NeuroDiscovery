"""Tests for the knowledge graph schema module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from neurooracle.src.schema import (
    Claim,
    ConceptNode,
    DomainTag,
    Edge,
    Evidence,
    PaperRef,
    SemanticType,
)


def test_concept_node_creation():
    """Test creating a ConceptNode."""
    node = ConceptNode(
        id="NN:1234",
        preferred_name="Hippocampus",
        domain_tags=[DomainTag.NEUROANATOMY.value],
        source_vocab="NeuroNames",
    )
    assert node.id == "NN:1234"
    assert node.preferred_name == "Hippocampus"
    assert DomainTag.NEUROANATOMY.value in node.domain_tags


def test_concept_node_to_dict():
    """Test ConceptNode serialization."""
    node = ConceptNode(
        id="NN:1234",
        preferred_name="Hippocampus",
        aliases=["Hippocampal formation"],
    )
    d = node.to_dict()
    assert d["id"] == "NN:1234"
    assert d["preferred_name"] == "Hippocampus"
    assert "Hippocampal formation" in d["aliases"]


def test_concept_node_from_dict():
    """Test ConceptNode deserialization."""
    d = {
        "id": "NN:1234",
        "preferred_name": "Hippocampus",
        "domain_tags": ["neuroanatomy"],
        "aliases": ["Hippocampal formation"],
    }
    node = ConceptNode.from_dict(d)
    assert node.id == "NN:1234"
    assert node.preferred_name == "Hippocampus"
    assert "Hippocampal formation" in node.aliases


def test_concept_node_roundtrip():
    """Test ConceptNode serialization roundtrip."""
    original = ConceptNode(
        id="GENE:APOE",
        preferred_name="APOE",
        domain_tags=[DomainTag.GENE.value],
        definition="Apolipoprotein E",
        aliases=["ApoE"],
        external_ids={"NCBI": "348"},
    )
    d = original.to_dict()
    restored = ConceptNode.from_dict(d)
    assert restored.id == original.id
    assert restored.preferred_name == original.preferred_name
    assert restored.definition == original.definition
    assert restored.aliases == original.aliases
    assert restored.external_ids == original.external_ids


def test_edge_creation():
    """Test creating an Edge."""
    edge = Edge(
        source_id="NN:hippocampus",
        target_id="NN:temporal_lobe",
        relation_type="part_of",
        source="NeuroNames",
    )
    assert edge.source_id == "NN:hippocampus"
    assert edge.target_id == "NN:temporal_lobe"
    assert edge.relation_type == "part_of"
    assert edge.confidence == 1.0


def test_edge_to_dict():
    """Test Edge serialization."""
    edge = Edge(
        source_id="GENE:APOE",
        target_id="MSH:alzheimer",
        relation_type="gene_associated_with_disease",
        confidence=0.95,
    )
    d = edge.to_dict()
    assert d["source_id"] == "GENE:APOE"
    assert d["target_id"] == "MSH:alzheimer"
    assert d["confidence"] == 0.95


def test_edge_from_dict():
    """Test Edge deserialization."""
    d = {
        "source_id": "NN:A",
        "target_id": "NN:B",
        "relation_type": "projects_to",
        "source": "test",
        "confidence": 0.8,
    }
    edge = Edge.from_dict(d)
    assert edge.source_id == "NN:A"
    assert edge.confidence == 0.8


def test_evidence_creation():
    """Test creating Evidence."""
    ev = Evidence(
        study_type="fMRI",
        methodology="resting-state FC",
        p_value=0.001,
        effect_size=0.45,
        sample_size=100,
    )
    assert ev.study_type == "fMRI"
    assert ev.p_value == 0.001


def test_evidence_roundtrip():
    """Test Evidence serialization roundtrip."""
    original = Evidence(
        study_type="meta-analysis",
        effect_size=0.6,
        effect_metric="Cohen's d",
        replicability="replicated",
    )
    d = original.to_dict()
    restored = Evidence.from_dict(d)
    assert restored.study_type == original.study_type
    assert restored.effect_size == original.effect_size


def test_paper_ref_creation():
    """Test creating a PaperRef."""
    ref = PaperRef(
        pmid="12345678",
        doi="10.1234/test",
        title="Test Paper",
        year=2024,
    )
    assert ref.pmid == "12345678"
    assert ref.year == 2024


def test_paper_ref_roundtrip():
    """Test PaperRef serialization roundtrip."""
    original = PaperRef(
        pmid="12345678",
        title="Test Paper",
        authors="Smith et al.",
        journal="Nature",
    )
    d = original.to_dict()
    restored = PaperRef.from_dict(d)
    assert restored.pmid == original.pmid
    assert restored.title == original.title


def test_claim_creation():
    """Test creating a Claim."""
    claim = Claim(
        id="CLM:001",
        subject_id="GENE:APOE",
        subject_name="APOE",
        predicate="is_risk_factor_for",
        object_id="MSH:alzheimer",
        object_name="Alzheimer Disease",
        confidence=0.9,
    )
    assert claim.id == "CLM:001"
    assert claim.predicate == "is_risk_factor_for"


def test_claim_to_edge():
    """Test converting Claim to Edge."""
    claim = Claim(
        id="CLM:001",
        subject_id="GENE:APOE",
        subject_name="APOE",
        predicate="is_risk_factor_for",
        object_id="MSH:alzheimer",
        object_name="Alzheimer Disease",
        confidence=0.9,
        source_paper=PaperRef(pmid="12345"),
    )
    edge = claim.to_edge()
    assert edge.source_id == "GENE:APOE"
    assert edge.target_id == "MSH:alzheimer"
    assert edge.relation_type == "is_risk_factor_for"
    assert edge.confidence == 0.9
    assert "12345" in edge.source


def test_claim_roundtrip():
    """Test Claim serialization roundtrip."""
    original = Claim(
        id="CLM:002",
        subject_id="NN:hippocampus",
        subject_name="Hippocampus",
        predicate="associated_with",
        object_id="MSH:alzheimer",
        object_name="Alzheimer Disease",
        negated=False,
        confidence=0.75,
        evidence=Evidence(study_type="fMRI", p_value=0.01),
        source_paper=PaperRef(pmid="98765"),
        raw_text="Hippocampus is associated with Alzheimer Disease.",
    )
    d = original.to_dict()
    restored = Claim.from_dict(d)
    assert restored.id == original.id
    assert restored.subject_id == original.subject_id
    assert restored.predicate == original.predicate
    assert restored.evidence.study_type == "fMRI"
    assert restored.source_paper.pmid == "98765"


def test_domain_tag_values():
    """Test DomainTag enum values."""
    assert DomainTag.NEUROANATOMY.value == "neuroanatomy"
    assert DomainTag.DISEASE.value == "disease"
    assert DomainTag.GENE.value == "gene"
    assert DomainTag.NEUROTRANSMITTER.value == "neurotransmitter"


def test_semantic_type_values():
    """Test SemanticType enum values."""
    assert SemanticType.DISEASE_OR_SYNDROME.value == "T047"
    assert SemanticType.BODY_PART_ORGAN.value == "T023"


if __name__ == "__main__":
    test_concept_node_creation()
    test_concept_node_to_dict()
    test_concept_node_from_dict()
    test_concept_node_roundtrip()
    test_edge_creation()
    test_edge_to_dict()
    test_edge_from_dict()
    test_evidence_creation()
    test_evidence_roundtrip()
    test_paper_ref_creation()
    test_paper_ref_roundtrip()
    test_claim_creation()
    test_claim_to_edge()
    test_claim_roundtrip()
    test_domain_tag_values()
    test_semantic_type_values()
    print("\n=== ALL SCHEMA TESTS PASSED ===")
