"""Local deterministic KG checks; these tests never call a model or a service."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from neurooracle.src import claim_ingestion as ingestion
from neurooracle.src.claim_semantics import audit_claim_endpoints, semantic_claim_endpoint
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import ConceptNode


@pytest.fixture(autouse=True)
def isolated_resolution(monkeypatch):
    monkeypatch.setattr(ingestion, "_resolution_idx", ingestion._ResolutionIndex())
    monkeypatch.setattr(ingestion, "_NOISE_FILTER_ENABLED", False)
    monkeypatch.setattr(ingestion, "_STRICT_PHASE1", False)
    monkeypatch.setattr(ingestion, "_DROP_LOG", SimpleNamespace(record=lambda *args, **kwargs: None))


def graph(*nodes):
    kg = KnowledgeGraph()
    for node in nodes:
        kg.add_concept(node)
    return kg


def node(node_id, name, domain, aliases=()):
    return ConceptNode(id=node_id, preferred_name=name, domain_tags=[domain], aliases=list(aliases))


@pytest.mark.parametrize("name", ["total cortical surface area", "left superior-frontal surface-area reduction",
                                  "bilateral putamen volume reduction"])
def test_real_word_interior_gene_alias_cases_do_not_recur(name):
    kg = graph(node("CUI:C1414531", "FANCE", "gene", ["FACE"]), node("CUI:C1421437", "VCP", "gene", ["TERA"]))
    before = {key: deepcopy(value.to_dict()) for key, value in kg._index.items()}
    resolved = ingestion.resolve_entity(kg, name, "IMAGING_MARKER")
    assert resolved not in before
    assert kg.get_concept(resolved).preferred_name == name
    assert kg.get_concept(resolved).domain_tags == ["biomarker"]
    assert {key: kg.get_concept(key).to_dict() for key in before} == before


@pytest.mark.parametrize("alias,text,match", [("face", "surface", False), ("tera", "bilateral", False),
                                            ("ic", "specific", False), ("face", "face gene", True),
                                            ("cortical surface area", "total cortical surface area", True)])
def test_phrase_boundaries_apply_to_all_alias_lengths(alias, text, match):
    assert ingestion._word_boundary_match(alias, text) is match


@pytest.mark.parametrize("lookup", ["preferred", "case", "alias", "fuzzy"])
def test_wrong_type_cannot_escape_through_any_lookup_path(lookup):
    name = "hippocampal volume"
    target = node("WRONG", name if lookup in {"preferred", "case"} else "Gene A", "gene",
                  [name] if lookup == "alias" else ["hippocampal"] if lookup == "fuzzy" else [])
    kg = graph(target)
    resolved = ingestion.resolve_entity(kg, name.upper() if lookup == "case" else name, "IMAGING_MARKER")
    assert resolved != "WRONG"


def test_compatible_domain_alias_is_reused_even_when_an_incompatible_homonym_was_first():
    kg = graph(node("WRONG", "Other gene", "gene", ["regional marker"]),
               node("RIGHT", "Regional imaging marker", "imaging_feature", ["regional marker"]))
    assert ingestion.resolve_entity(kg, "regional marker", "IMAGING_MARKER") == "RIGHT"
    assert len(kg._index) == 2


def test_compatible_exact_and_alias_lookup_preserve_legitimate_gene_names():
    kg = graph(node("G:FANCE", "FANCE", "gene", ["FACE"]))
    for name in ("FANCE", "fance", "FACE"):
        assert ingestion.resolve_entity(kg, name, "GENE") == "G:FANCE"
    assert len(kg._index) == 1


def test_homonyms_are_not_resolved_by_order_or_shortest_name():
    kg = graph(node("G:1", "Gene Alpha", "gene", ["shared marker"]),
               node("G:2", "G2", "gene", ["shared marker"]))
    resolved = ingestion.resolve_entity(kg, "shared marker", "GENE")
    assert resolved not in {"G:1", "G:2"}
    assert ingestion.resolve_entity(kg, "shared marker", "GENE") == resolved
    assert len(kg._index) == 3


def test_unknown_type_does_not_become_a_disease_or_reuse_a_typed_homonym():
    kg = graph(node("D:1", "unclassified laboratory readout", "disease"))
    resolved = ingestion.resolve_entity(kg, "unclassified laboratory readout", "FLUID_BIOMARKER")
    assert resolved != "D:1"
    assert kg.get_concept(resolved).domain_tags == ["external"]
    assert ingestion.resolve_entity(kg, "unclassified laboratory readout", "FLUID_BIOMARKER") == resolved
    assert len(kg._index) == 2


def test_conflicting_minted_id_is_not_overwritten():
    existing = node("CLM_CONCEPT:hippocampal_volume", "hippocampal volume", "gene")
    kg = graph(existing)
    before = deepcopy(existing.to_dict())
    resolved = ingestion.resolve_entity(kg, "hippocampal volume", "IMAGING_MARKER")
    assert resolved != existing.id and kg.get_concept(existing.id).to_dict() == before
    assert ingestion.resolve_entity(kg, "hippocampal volume", "IMAGING_MARKER") == resolved


def test_external_addition_invalidates_name_index():
    kg = graph(node("G:1", "Gene Alpha", "gene"))
    assert ingestion.resolve_entity(kg, "Gene Alpha", "GENE") == "G:1"
    kg.add_concept(node("G:2", "Gene Beta", "gene"))
    assert ingestion.resolve_entity(kg, "Gene Beta", "GENE") == "G:2"
    assert len(kg._index) == 2


def test_singleton_index_entries_stay_compact_but_collisions_are_retained():
    kg = graph(node("G:1", "Gene Alpha", "gene", ["shared"]), node("G:2", "Gene Beta", "gene", ["shared"]))
    index = ingestion._ResolutionIndex()
    index.build(kg)
    assert index._exact["Gene Alpha"] == "G:1"
    assert index._alias_lower["shared"] == {"G:1", "G:2"}
    assert index.lookup_alias("shared") is None


def test_salvage_cannot_bypass_type_or_ambiguity_checks():
    kg = graph(node("G:1", "marker alpha", "gene"), node("G:2", "Other gene", "gene", ["marker alpha"]))
    ingestion._resolution_idx.build(kg)
    assert ingestion._safe_salvage_id(kg, "marker alpha", "marker alpha", "GENE") is None
    assert ingestion._safe_salvage_id(kg, "marker alpha", "marker alpha", "IMAGING_MARKER") is None


def test_measurement_is_not_collapsed_to_its_anatomical_carrier():
    kg = graph(node("A:P", "Putamen", "neuroanatomy"))
    assert ingestion.resolve_entity(kg, "bilateral putamen volume reduction", "IMAGING_MARKER") != "A:P"


def test_bad_canonical_hint_is_checked_and_failed_strict_resolution_clears_stale_id(monkeypatch):
    kg = graph(node("G:FANCE", "FANCE", "gene", ["FACE"]), node("D:1", "Disease B", "disease"))
    claim = SimpleNamespace(subject_name="total cortical surface area", subject_id="G:FANCE",
                            object_name="Disease B", object_id="D:1",
                            metadata={"subject_type": "IMAGING_MARKER", "object_type": "DISEASE",
                                      "subject_canonical_hint": "G:FANCE", "object_canonical_hint": "D:1"})
    monkeypatch.setattr(ingestion, "_STRICT_PHASE1", True)
    ingestion.resolve_claim_entities(kg, claim)
    assert claim.subject_id == "" and claim.object_id == "D:1"
    assert len(kg._index) == 2


@pytest.mark.parametrize("side,missing,reason", [
    ("subject", "declared", "subject_declared_type_unresolved"),
    ("object", "declared", "object_declared_type_unresolved"),
    ("subject", "canonical", "subject_canonical_type_unresolved"),
    ("object", "canonical", "object_canonical_type_unresolved"),
])
def test_unknown_endpoint_roles_never_pass_semantic_validation(side, missing, reason):
    nodes = {"A": node("A", "readout alpha", "gene"), "B": node("B", "condition beta", "disease")}
    claim = {"subject_id": "A", "subject_name": "readout alpha", "subject_type": "GENE",
             "object_id": "B", "object_name": "condition beta", "object_type": "DISEASE"}
    if missing == "declared":
        claim[side + "_type"] = "UNCLASSIFIED"
    else:
        nodes[claim[side + "_id"]].domain_tags = ["external"]
    audit = audit_claim_endpoints(claim, nodes)
    assert not audit.valid and audit.reason == reason
    projected = semantic_claim_endpoint(claim, side, nodes)
    assert not projected.uses_canonical_id and not projected.role_compatible


def test_fluid_marker_is_not_silently_certified_as_imaging():
    claim = {"subject_id": "F", "subject_name": "CSF total tau elevation", "subject_type": "FLUID_BIOMARKER",
             "object_id": "D", "object_name": "Disease B", "object_type": "DISEASE"}
    nodes = {"F": node("F", claim["subject_name"], "biomarker"), "D": node("D", "Disease B", "disease")}
    audit = audit_claim_endpoints(claim, nodes)
    assert not audit.valid and audit.reason == "subject_declared_type_unresolved"
