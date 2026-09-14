from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import apply_kg_bulk_cleanup as batch
from apply_kg_relation_identity import stream_patch,verify_catalog_and_graph
from neurooracle.src.correlation_grouping import POLICY,IndexTerms
from neurooracle.src.kg_bulk_cleanup import simplify_record,verify_simplification
from neurooracle.src.schema import ConceptNode
from neurooracle.tests.test_kg_bulk_cleanup import row
from neurooracle.tests.test_shared_relation_catalog import setup_files,fp
from neurooracle.src.shared_relation_catalog import find_shared_relations


def fixture_records():
    a=row();b=deepcopy(a);b['id']=b['metadata']['id']='CLM:second'
    md=b['metadata'];md['source_paper']['pmid']='456'
    md['subject_id'],md['object_id']=md['object_id'],md['subject_id']
    md['subject_name'],md['object_name']=md['object_name'],md['subject_name']
    yield 'metadata','metadata',{}
    for nid in ('A','B'):yield 'node',nid,ConceptNode(id=nid,preferred_name=nid).to_dict()
    for r in (a,b):yield 'node',r['id'],r
    for i,r in enumerate((a,b),1):
        md=r['metadata'];yield 'edge',str(i),dict(source_id=md['subject_id'],target_id=md['object_id'],relation_type='correlates_with',
            confidence=0.5,metadata={'claim_id':r['id'],'negated':md['negated']})


def record_stream(data):
    yield 'metadata','metadata',data['metadata']
    for nid,row in data['concepts'].items():yield 'node',nid,row
    for n,row in enumerate(data['edges'],1):yield 'edge',str(n),row


def test_metadata_transform_and_full_relation_verifier(tmp_path):
    original=list(fixture_records());prepared=[]
    for kind,key,r in original:
        out=simplify_record(kind,r) if kind!='metadata' else r
        if kind=='node' and key.startswith('CLM:'):verify_simplification(r,out)
        prepared.append((kind,key,out))
    output=io.BytesIO();catalog=tmp_path/'catalog.jsonl';terms=IndexTerms()
    result=stream_patch(iter(prepared),output,[],{},catalog,relation_key_func=terms.relation_key,
        extra_graph_metadata={'relation_grouping':POLICY},relation_version='kg.relation_evidence.v4')
    data=json.loads(output.getvalue());checks=verify_catalog_and_graph(record_stream(data),result,catalog,relation_key_func=terms.relation_key)
    assert checks['dangling_endpoints']==0 and checks['shared_groups']==1
    assert result['counts']=={'nodes':4,'claims':2,'edges':2}
    assert data['edges']==[r for k,_,r in original if k=='edge']
    data['concepts']['CLM:test']['metadata']['evidence']['p_value']=0
    with pytest.raises(Exception):verify_catalog_and_graph(record_stream(data),result,catalog,relation_key_func=terms.relation_key)


@pytest.mark.parametrize('policy',[False,True])
@pytest.mark.parametrize('predicate',['correlates_with','predicts','causes'])
def test_validated_sidecar_reverse_query_only_for_opt_in_correlation(tmp_path,policy,predicate):
    graph,catalog,receipt,campaign,control=setup_files(tmp_path)
    catalog.write_text(json.dumps(dict(subject_id='A',subject_name='measurement',predicate=predicate,object_id='B',object_name='outcome',paper_count=2))+'\n')
    accepted=json.loads(receipt.read_text());accepted['shared_relations']=fp(catalog)
    if policy:accepted['relation_grouping']=POLICY
    receipt.write_text(json.dumps(accepted));control.update(current_acceptance=fp(receipt),current_shared_relations=fp(catalog));campaign.write_text(json.dumps(control))
    assert len(find_shared_relations(campaign,subject='measurement',object_name='outcome'))==1
    assert len(find_shared_relations(campaign,subject='outcome',object_name='measurement'))==int(policy and predicate=='correlates_with')


def test_batch_does_not_rewrite_closed_automation_or_scientific_scope():
    code=Path(batch.__file__).read_text(encoding='utf8')
    assert 'automation_update(' not in code and 'NIGHT_WINDOW' not in code
    assert 'os.replace(TEMP,SOURCE)' in code
    assert 'verify_simplification(source,out)' in code
    assert "checks.pop('all_nonidentity_claim_fields_preserved',None)" in code
