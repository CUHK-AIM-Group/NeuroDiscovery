from copy import deepcopy
import json
import pytest

from neurooracle.src.claim_evidence_identity import evidence_dedup_key
from neurooracle.src.claim_extractor import ExtractionResult
from neurooracle.src.claim_ingestion import ingest_claims
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.schema import Claim, ConceptNode, Evidence, PaperRef
from neurooracle.src.storage import save_graph, load_graph, save_display_graph


def claim(cid="CLM:one",pmid="123"):
    return Claim(id=cid,subject_id="S",subject_name="hippocampal volume",predicate="predicts",object_id="O",
        object_name="memory deficit",source_paper=PaperRef(pmid=pmid),confidence=.95,
        evidence=Evidence(study_type="cohort"),raw_text="Hippocampal volume predicted memory deficit.",
        metadata={"subject_type":"imaging_marker","object_type":"clinical_outcome"})


@pytest.mark.parametrize("field,value",[("negated",True),("predicate","correlates_with"),
    ("raw_text","A different recorded sentence."),("subject_name","left hippocampal volume"),
    ("population",{"species":"mouse"}),("conditions",{"time":"followup"}),
    ("unknown_scientific_field",{"value":0})])
def test_distinct_observations_not_duplicates(field,value):
    a=claim().to_dict(); b={**deepcopy(a),field:value}
    assert evidence_dedup_key(a)!=evidence_dedup_key(b)


def test_full_evidence_and_typed_values_preserved():
    a=claim().to_dict(); b=deepcopy(a); c=deepcopy(a)
    b["evidence"]["new_value"]=False; c["evidence"]["new_value"]=0
    assert len({evidence_dedup_key(x) for x in [a,b,c]})==3


def test_unknown_source_context_not_discarded_by_paper_normalization():
    a=claim().to_dict(); b=deepcopy(a); b["source_paper"]["cohort"]="independent subset"
    assert evidence_dedup_key(a)!=evidence_dedup_key(b)


def test_different_ids_same_complete_evidence_idempotent():
    a=claim().to_dict(); b=claim("CLM:two").to_dict(); b["confidence"]=.4
    assert evidence_dedup_key(a)==evidence_dedup_key(b)


def test_different_papers_and_missing_papers():
    assert evidence_dedup_key(claim(pmid="123").to_dict())!=evidence_dedup_key(claim(pmid="456").to_dict())
    a=claim(pmid="").to_dict(); assert evidence_dedup_key(a) is None
    a["source_paper"]["title"]="Identical title"; assert evidence_dedup_key(a) is None
    a["source_paper"]["doi"]="https://doi.org/10.1000/ABC"; b=deepcopy(a); b["source_paper"]["doi"]="doi:10.1000/abc"
    assert evidence_dedup_key(a)==evidence_dedup_key(b)


def graph():
    kg=KnowledgeGraph()
    kg.add_concept(ConceptNode(id="S",preferred_name="hippocampal volume",domain_tags=["imaging_feature"]))
    kg.add_concept(ConceptNode(id="O",preferred_name="memory deficit",domain_tags=["treatment_outcome"]))
    return kg


def ingest(kg,claims):
    return ingest_claims(kg,[ExtractionResult(paper=claims[0].source_paper,claims=claims)],
        refine_vague_predicates=False,keep_noise=True,require_final_scope_audit=False)


def test_real_ingestion_preserves_contradiction_context_and_predicates():
    kg=graph(); a=claim(); b=claim("CLM:two"); b.negated=True
    c=claim("CLM:three"); c.metadata["conditions"]={"time":"baseline"}
    d=claim("CLM:four"); d.predicate="correlates_with"; d.raw_text="Hippocampal volume correlated with memory deficit."
    assert ingest(kg,[a,b,c,d])["claims_added"]==4
    assert len([e for e in kg.iter_edge_records() if e.get("metadata",{}).get("claim_id")])==4


def test_real_ingestion_exact_duplicate_and_reload(tmp_path):
    kg=graph(); assert ingest(kg,[claim()])["claims_added"]==1
    result=ingest(kg,[claim("CLM:repeat")]); assert result["claims_added"]==0
    assert result["rejected_claims"][0]["reason"]=="duplicate_paper_evidence"
    loaded=load_graph(save_graph(kg,tmp_path/"graph.json"))
    assert ingest(loaded,[claim("CLM:repeat2")])["claims_added"]==0


def test_existing_claim_conflict_reported_not_overwritten():
    kg=graph(); ingest(kg,[claim()]); other=claim(); other.negated=True
    result=ingest(kg,[other]); assert result["rejected_claims"][0]["reason"]=="conflicting_claim_id"
    assert kg.get_concept("CLM:one").metadata["negated"] is False


def test_cache_invalidated_when_existing_evidence_changes():
    kg=graph(); ingest(kg,[claim()]); kg.update_claim("CLM:one",new_evidence=Evidence(study_type="different"))
    assert ingest(kg,[claim("CLM:two")])["claims_added"]==1


def test_exports_do_not_retain_snapshot_only_index_pointer(tmp_path):
    kg=graph(); kg.serialization_metadata["relation_evidence"]={"shared_relation_index":"stale.jsonl"}
    for filename,fn in [("full.json",save_graph),("display.json",save_display_graph)]:
        path=fn(kg,tmp_path/filename)
        assert "relation_evidence" not in json.loads(path.read_text(encoding="utf-8"))["metadata"]
    assert "relation_evidence" in kg.serialization_metadata
