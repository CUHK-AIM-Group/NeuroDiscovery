from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_gene_boundary_repair import review_counts,coverage_summary


def fixture():
    old=[dict(claim_id='a',side='subject'),dict(claim_id='b',side='object')]
    plan=dict(old_watchlist_endpoints=2,changed_endpoints=2,expanded_review_endpoints=5,
        newly_registered_lexical_candidates=3,held_endpoints=3,
        events=[dict(claim_id=cid,changes=[dict(side=side)]) for cid,side in [('a','subject'),('c','object')]])
    return plan,old


def test_old_and_new_queue_counts_are_not_conflated():
    p,old=fixture();r=review_counts(p,old)
    assert r['original_watchlist_repaired']==1 and r['original_watchlist_remaining']==1
    assert r['newly_discovered_repaired']==1 and r['expanded_remaining']==3
    assert r['not_all_confirmed_errors'] is True


@pytest.mark.parametrize('field',['old_watchlist_endpoints','changed_endpoints','expanded_review_endpoints','held_endpoints'])
def test_inconsistent_count_rejected(field):
    p,old=fixture();p[field]+=1
    with pytest.raises(Exception,match='differ'):review_counts(p,old)


def coverage_fixture():
    return [dict(scope='node/claim',field=f,denominator=100,present=100,nonempty_pct=pct) for f,pct in [
        ('metadata.raw_text',100),('metadata.evidence.p_value',1),('metadata.evidence.effect_size',2),
        ('metadata.evidence.sample_size',3),('metadata.metadata.audit',100)]]


def test_coverage_equality_not_confused_with_new_nonclaim_denominator():
    a=coverage_fixture();b=deepcopy(a)+[dict(scope='node/source_mention',field='top.id',denominator=999,present=999)]
    r=coverage_summary(a,b)
    assert r['claim_fields_unchanged']==5 and r['average_nested_fields']==1
    assert r['p_value_nonempty_pct']==1


def test_any_claim_coverage_change_blocks_reporting():
    a=coverage_fixture();b=deepcopy(a);b[1]['nonempty_pct']=100
    with pytest.raises(Exception,match='coverage changed'):coverage_summary(a,b)
