from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_research_statement_retirement import coverage_delta


def fixture():
    specs=[('metadata.raw_text',100,96),('metadata.evidence.p_value',5,5),('metadata.evidence.effect_size',10,10),
        ('metadata.evidence.sample_size',20,20),('metadata.metadata.original_predicate',50,46)]
    a=[dict(scope='node/claim',field=f,denominator=100,present=100,nonempty=old,nonempty_pct=old,types={'str':100}) for f,old,new in specs]
    b=[dict(scope='node/claim',field=f,denominator=96,present=96,nonempty=new,nonempty_pct=100*new/96,types={'str':96}) for f,old,new in specs]
    return a,b


def test_four_null_statistic_claims_only_change_statistics_denominator():
    a,b=fixture();r=coverage_delta(a,b)
    assert r['current_claims']==96 and r['previous_claims']==100 and r['new_claim_fields']==0
    assert r['p_value_nonempty_pct']==500/96


@pytest.mark.parametrize('problem',['more_removed','new_field','new_type','mixed_denominator','nonempty_statistic_removed','type_loss_mismatch'])
def test_unexplained_coverage_changes_block_publication(problem):
    a,b=fixture()
    if problem=='more_removed':b[0]['present']=95
    if problem=='new_field':b.append(dict(b[0],field='new'))
    if problem=='new_type':b[0]['types']={'dict':96}
    if problem=='mixed_denominator':b[0]['denominator']=95
    if problem=='nonempty_statistic_removed':b[1]['nonempty']=4
    if problem=='type_loss_mismatch':b[0]['types']={'str':95}
    with pytest.raises(ValueError):coverage_delta(a,b)
