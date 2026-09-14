from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_source_scope_resolution import coverage_delta,collect_hold_hashes


def fixture():
    fields=[('metadata.raw_text',100,99),('metadata.evidence.p_value',5,5),('metadata.evidence.effect_size',10,10),
        ('metadata.evidence.sample_size',20,19),('metadata.metadata.curation',40,39)]
    a=[dict(scope='node/claim',field=f,denominator=100,present=100,nonempty=old,nonempty_pct=old,types={'str':100}) for f,old,new in fields]
    b=[dict(scope='node/claim',field=f,denominator=99,present=99,nonempty=new,nonempty_pct=100*new/99,types={'str':99}) for f,old,new in fields]
    return a,b


def test_one_claim_coverage_delta_is_not_a_bulk_field_cleanup():
    a,b=fixture();r=coverage_delta(a,b)
    assert r['previous_claims']==100 and r['current_claims']==99 and r['new_claim_fields']==0
    assert r['p_value_nonempty_pct']==500/99


@pytest.mark.parametrize('problem',['too_many_removed','new_field','new_type','nonempty_without_presence'])
def test_unexplained_coverage_change_is_rejected(problem):
    a,b=fixture()
    if problem=='too_many_removed':b[0]['present']=98
    if problem=='new_field':b.append(dict(b[0],field='new'))
    if problem=='new_type':b[0]['types']={'dict':99}
    if problem=='nonempty_without_presence':b[1]['nonempty']=3
    with pytest.raises(ValueError):coverage_delta(a,b)


def test_current_and_legacy_hash_fields_and_scope_only_holds():
    data=[dict(claim_id='a',claim_sha256='x'),dict(claim_id='b',current_node_sha256='y'),dict(claim_id='c')]
    assert collect_hold_hashes(data,{'c':'z'})=={'a':'x','b':'y','c':'z'}


def test_conflicting_hold_hashes_cannot_be_published_as_current():
    with pytest.raises(ValueError,match='inconsistent'):
        collect_hold_hashes([dict(claim_id='a',claim_sha256='old')],{'a':'new'})
