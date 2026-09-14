"""Offline end-to-end ingest/save/reload checks, separate from frozen R30 build."""
import pytest

from neurooracle.tests.test_verified_entity_terms import saved_fixture
from neurooracle.tests.test_claim_evidence_identity import ingest
from neurooracle.src.schema import Claim,ConceptNode,Evidence,PaperRef,Edge
from neurooracle.src.storage import load_graph,save_graph


@pytest.mark.parametrize("object_type",["disease","OUTCOME",""])
def test_actual_ingestion_preserves_registry_and_independent_papers(tmp_path,object_type):
    path,_,_=saved_fixture(tmp_path); kg=load_graph(path)
    kg.add_concept(ConceptNode(id="IM:hippocampal",preferred_name="hippocampal volume",domain_tags=["imaging_feature"]))
    def candidate(cid,pmid,negated=False):
        return Claim(id=cid,subject_id="",subject_name="hippocampal volume",predicate="predicts",object_id="",
            object_name="bipolar disorder",source_paper=PaperRef(pmid=pmid),confidence=.95,negated=negated,
            raw_text="Lower hippocampal volume predicted bipolar disorder.",evidence=Evidence(study_type="cohort"),
            metadata={"subject_type":"imaging_marker","object_type":object_type})
    first=candidate("CLM:first","111")
    assert ingest(kg,[first])["claims_added"]==1
    assert first.object_id=="CUI:test" and kg.relation_identities is not None
    loaded=load_graph(save_graph(kg,tmp_path/"ingested.json"))
    assert ingest(loaded,[candidate("CLM:second","222",True)])["claims_added"]==1
    assert ingest(loaded,[candidate("CLM:exact-repeat","111")])["claims_added"]==0
    final=load_graph(save_graph(loaded,tmp_path/"ingested.json"))
    groups=list(final.iter_relation_evidence(minimum_claims=2))
    assert len(groups)==1 and groups[0]["paper_count"]==2
    assert {m["negated"] for m in groups[0]["members"]}=={False,True}
    assert final.relation_identities is not None
    assert len(list(tmp_path.glob("ingested.json.entity_terms.*.json")))==1


@pytest.mark.parametrize("mutation",["new_atom","mapping_of_nonproof_node"])
def test_new_ontology_evidence_requires_global_uniqueness_recheck(tmp_path,mutation):
    path,_,_=saved_fixture(tmp_path); kg=load_graph(path)
    if mutation=="new_atom":
        kg.add_concept(ConceptNode(id="CLM_ATOM:other",preferred_name="bipolar disorder"))
    else:
        kg.add_edge(Edge("CLM_CONCEPT:parent","CUI:test","maps_to"))
    assert kg.relation_identities is None
    assert load_graph(save_graph(kg,tmp_path/"changed-ontology.json")).relation_identities is None
