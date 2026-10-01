import numpy as np
import pytest

from models.kg_link_prediction.full_graph_audit import check_pair_folds, exclusion


def test_audit_rejects_reversed_endpoint_leak():
    triples=np.array([[0,0,1],[1,1,0],[1,0,2]])
    with pytest.raises(ValueError,match='cross folds'):
        check_pair_folds(triples,np.array([0,1,2]),3)
    assert check_pair_folds(triples,np.array([0,0,2]),3)==0


def test_audit_eligibility_explains_exclusions():
    nodes={'a':('gene',False),'b':('gene',False),'paper':('publication',True)}
    edge=dict(source_id='a',target_id='b',relation_type='associated_with',confidence=.8)
    assert exclusion(edge,nodes) is None
    assert exclusion(dict(edge,target_id='paper'),nodes)=='infrastructure_node'
    assert exclusion(dict(edge,negated=True),nodes)=='negated'
    assert exclusion(dict(edge,confidence=float('nan')),nodes)=='low_nonfinite_confidence'
    assert exclusion(dict(edge,target_id='missing'),nodes)=='dangling'
