from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_kg_expanded_literal_scope import gate
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges
from neurooracle.src.kg_identity_pilot import digest


def fixture():
    r=claim('brain structural alterations')
    q=dict(side='subject',name=r['metadata']['subject_name'],current_node_id=r['metadata']['subject_id'],claim_sha256=digest(r))
    return r,q,dict(labels=['VCP','TERA'],semantic_types=['T028']),edges(r)


def test_interior_hit_is_not_alone_a_scientific_validation():
    result=gate(*fixture())
    assert not result['structural_holds'] and result['word_interior_alias_hits'] and result['declared_roles']==['imaging_marker']


@pytest.mark.parametrize('problem',['whole_gene','nested','closure','not_gene'])
def test_structural_or_identity_hold(problem):
    r,q,w,refs=fixture()
    if problem=='whole_gene':w['labels'].append('brain')
    elif problem=='nested':r['metadata']['metadata']['subject_id']='other';q['claim_sha256']=digest(r)
    elif problem=='closure':refs.pop(0)
    else:w['semantic_types']=['T116']
    assert gate(r,q,w,refs)['structural_holds']


def test_stale_current_projection_rejected():
    r,q,w,refs=fixture();q['claim_sha256']='0'*64
    with pytest.raises(ValueError):gate(r,q,w,refs)
