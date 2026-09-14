from copy import deepcopy
import pytest

from neurooracle.src.kg_literal_endpoint_repair import (endpoint_gate, literal_node,
    reuse_gate, reviewed_claim, reviewed_edges, apply_edge, reverse_claim)
from neurooracle.src.kg_identity_pilot import digest


def claim(name="bilateral amygdala volume", typ="IMAGING_MARKER"):
    return dict(id="CLM:test", metadata=dict(id="CLM:test", subject_id="CUI:C1421437",
        subject_name=name, subject_type=typ, object_id="CUI:disease", object_name="anxiety",
        object_type="DISEASE", predicate="is_associated_with", negated=True,
        raw_text="Independent source quotation.", source_paper={"pmid":"31756730"},
        conditions={"age":"adults", "laterality":"bilateral"}, metadata={"subject_id":"CUI:C1421437"}))


def change(row):
    node = literal_node(row["metadata"]["subject_name"])
    event, out = reviewed_claim(row, [dict(side="subject", old_id="CUI:C1421437",
        target_id=node["id"], name=node["preferred_name"])])
    return event, out


@pytest.mark.parametrize("name", ["bilateral amygdala volume", "left dorsolateral prefrontal cortex activation",
    "voxel-mirrored homotopic connectivity alterations", "right hippocampal fractional anisotropy"])
def test_complete_measurements(name):
    assert endpoint_gate(claim(name)["metadata"], "subject") is None


@pytest.mark.parametrize("name,typ", [("VCP","IMAGING_MARKER"), ("FANCE","IMAGING_MARKER"),
    ("gut microbiota alterations","IMAGING_MARKER"), ("collybistin-gephyrin interaction","IMAGING_MARKER"),
    ("COMT genotype cortical volume interaction","IMAGING_MARKER"),
    ("plasma protein activity","IMAGING_MARKER"), ("MRI classifier","IMAGING_MARKER"),
    ("brain alterations","IMAGING_MARKER"), ("bilateral amygdala volume","BIOMARKER"),
    ("bilateral amygdala volume","GENE_TARGET"), ("bilateral amygdala volume",None),
    ("bilateral amygdala volume","OUTCOME")])
def test_ambiguous_not_retyped(name, typ):
    assert endpoint_gate(claim(name, typ)["metadata"], "subject") is not None


def test_nested_conflicts():
    md = claim()["metadata"]; md["metadata"]["subject_id"] = "different"
    assert endpoint_gate(md,"subject") == "nested_id_conflict"
    md["metadata"]["subject_id"] = md["subject_id"]; md["metadata"]["subject_type"]="GENE_TARGET"
    assert endpoint_gate(md,"subject") == "nested_type_conflict"


def test_exact_reuse_not_fuzzy_case_or_laterality():
    a = literal_node("bilateral amygdala volume")
    assert a == literal_node(" bilateral   amygdala volume ")
    assert a["id"] != literal_node("left amygdala volume")["id"]
    assert a["id"] != literal_node("Bilateral amygdala volume")["id"]
    assert not a["metadata"] and not a["semantic_types"] and not a["external_ids"]
    assert reuse_gate(a, a["preferred_name"], []) is None
    assert reuse_gate(a, "amygdala volume", [])
    assert reuse_gate(a, a["preferred_name"], [dict(name=a["preferred_name"],declared_roles=["gene_target"])])
    assert reuse_gate(a, a["preferred_name"], [dict(name="left amygdala volume",declared_roles=["imaging_marker"])])


def test_inverse_proves_all_scientific_fields_unchanged():
    original = claim(); before = deepcopy(original)
    event, out = change(original)
    assert original == before and reverse_claim(out,event) == original
    assert out["metadata"]["negated"] is True
    bad = deepcopy(out); bad["metadata"]["conditions"]["age"]="children"
    with pytest.raises(ValueError): reverse_claim(bad,event)


def edges(row):
    md = row["metadata"]
    return [(1,dict(source_id=row["id"],target_id=md["subject_id"],relation_type="about",metadata={})),
        (2,dict(source_id=row["id"],target_id=md["object_id"],relation_type="about",metadata={})),
        (3,dict(source_id=md["subject_id"],target_id=md["object_id"],relation_type=md["predicate"],
            metadata={"claim_id":row["id"],"negated":True},source="31756730",evidence_ref="Original title"))]


def test_owned_edge_closure_and_inverse():
    row=claim(); event,out=change(row); refs=edges(row)
    changes=reviewed_edges(row["id"],row,out,refs)
    assert len(changes)==2
    for ev in changes:
        original=dict(refs)[ev["ordinal"]]; updated=apply_edge(original,ev)
        assert digest(updated)==ev["current_edge_sha256"]
        assert apply_edge(updated,ev,reverse=True)==original


def test_claim_only_representation_keeps_two_about_no_new_science():
    row=claim(); event,out=change(row); refs=edges(row)[:2]
    changes=reviewed_edges(row["id"],row,out,refs)
    assert len(changes)==1 and changes[0]["ordinal"]==1


@pytest.mark.parametrize("problem", ["missing","duplicate","wrong_endpoint","wrong_owner","wrong_predicate"])
def test_mixed_reference_closure_blocks_whole_claim(problem):
    row=claim(); event,out=change(row); refs=edges(row)
    if problem=="missing": refs.pop(0)
    if problem=="duplicate": refs.append((4,deepcopy(refs[0][1])))
    if problem=="wrong_endpoint": refs[2][1]["target_id"]="wrong"
    if problem=="wrong_owner": refs[0][1]["metadata"]["claim_id"]="CLM:wrong"
    if problem=="wrong_predicate": refs[2][1]["relation_type"]="causes"
    with pytest.raises(ValueError): reviewed_edges(row["id"],row,out,refs)
