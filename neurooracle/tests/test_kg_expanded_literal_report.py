from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_expanded_literal_repair import queue_summary,coverage_summary,HISTORIC_THREE


def fixture():
    extra=[dict(claim_id='CLM:new'+str(i),side='subject') for i in range(4)]
    before=[*extra,*[dict(claim_id=cid,side=side) for cid,side in sorted(HISTORIC_THREE)]]
    after=before[3:]
    plan=dict(events=[dict(claim_id=r['claim_id'],changes=[dict(side=r['side'])]) for r in extra[:3]],
        old_watchlist_endpoints=7,held_endpoints=4,changed_endpoints=3,original_watchlist_449_remaining_after=3,original_watchlist_449_repaired=0)
    return plan,before,after


def test_expanded_repair_is_not_credited_to_historic_449():
    p,b,a=fixture();r=queue_summary(p,b,a)
    assert r['original_watchlist']==449 and r['original_watchlist_repaired_this_batch']==0
    assert r['original_watchlist_remaining']==3 and r['expanded_repaired_this_batch']==3 and r['expanded_remaining']==4
    assert r['not_all_confirmed_errors']


@pytest.mark.parametrize('mutation',['missing','extra','duplicate','outside','historic','wrong_remaining'])
def test_bad_queue_or_false_historic_completion_rejected(mutation):
    p,b,a=fixture()
    if mutation=='missing':a.pop()
    elif mutation=='extra':a.append(b[0])
    elif mutation=='duplicate':a[-1]=a[0]
    elif mutation=='outside':p['events'][0]['claim_id']='CLM:outside'
    elif mutation=='historic':p['original_watchlist_449_repaired']=3
    else:p['original_watchlist_449_remaining_after']=0
    with pytest.raises(ValueError):queue_summary(p,b,a)


def test_claim_field_coverage_must_be_identical():
    before=[dict(scope='node/claim',field='metadata.raw_text',denominator=10,present=10,nonempty=10)]
    after=deepcopy(before);after[0]['present']=9
    with pytest.raises(ValueError,match='coverage changed'):coverage_summary(before,after)
