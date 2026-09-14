from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_reviewed_literal_reuse import queue_summary,coverage_summary
from neurooracle.src.kg_reviewed_literal_reuse import BROAD_REASSIGN


def fixture():
    historic={str(i):dict(claim_id='CLM:'+str(i),side='subject') for i in range(447)}
    events=[dict(claim_id='CLM:'+str(i),changes=[dict(side='subject')]) for i in range(425)]
    prior=dict(endpoint_reviews=historic,events=events)
    before=[*list(historic.values())[425:],dict(claim_id='CLM:outside',side='object')]
    fixed=[dict(claim_id='CLM:'+str(i),changes=[dict(side='subject')]) for i in range(425,444)]
    fixed.extend(dict(claim_id=cid,changes=[dict(side='subject')]) for cid in sorted(BROAD_REASSIGN))
    plan=dict(events=fixed,expanded_review_endpoints=23,held_endpoints=4,original_watchlist_repaired=19,original_watchlist_remaining_after=3)
    return plan,prior,before,before[19:]


def test_historic_corrections_do_not_include_two_nonhistoric_reassignments():
    plan,prior,before,after=fixture();result=queue_summary(plan,prior,before,after)
    assert result['original_watchlist']==449 and result['original_watchlist_remaining_before']==22
    assert result['original_watchlist_repaired_this_batch']==19 and result['original_watchlist_remaining']==3
    assert result['expanded_remaining']==4 and result['not_all_confirmed_errors']


@pytest.mark.parametrize('mutation',['missing','extra','duplicate','outside','broad','wrong_remaining','old_proof'])
def test_false_queue_completion_is_rejected(mutation):
    plan,prior,before,after=fixture()
    if mutation=='missing':after.pop()
    elif mutation=='extra':after.append(before[0])
    elif mutation=='duplicate':after[-1]=after[0]
    elif mutation=='outside':plan['events'][0]['claim_id']='CLM:outside'
    elif mutation=='broad':plan['events'].pop()
    elif mutation=='wrong_remaining':plan['original_watchlist_remaining_after']=0
    else:prior['events'].pop()
    with pytest.raises(ValueError):queue_summary(plan,prior,before,after)


def test_identity_batch_does_not_change_claim_field_coverage():
    before=[dict(scope='node/claim',field='metadata.raw_text',denominator=10,present=10,nonempty=10)]
    after=deepcopy(before);after[0]['present']=9
    with pytest.raises(ValueError,match='coverage changed'):coverage_summary(before,after)
