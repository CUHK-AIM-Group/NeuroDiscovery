from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_historic_literal_repair import queue_summary, coverage_summary


def fixture():
    historic = {str(i): dict(claim_id='CLM:'+str(i), side='subject') for i in range(447)}
    before = [*historic.values(), dict(claim_id='CLM:outside', side='object')]
    fixed = [dict(claim_id='CLM:'+str(i), changes=[dict(side='subject')]) for i in range(400)]
    after = before[400:]
    plan = dict(endpoint_reviews=historic, events=fixed, old_watchlist_endpoints=448, changed_endpoints=400,
        held_endpoints=48, original_watchlist_449_remaining_after=47)
    return plan, before, after


def test_original_and_expanded_queues_remain_distinct():
    plan, before, after = fixture(); result = queue_summary(plan, before, after)
    assert result['original_watchlist'] == 449 and result['original_watchlist_remaining_before'] == 447
    assert result['original_watchlist_remaining'] == 47 and result['expanded_remaining'] == 48
    assert result['not_all_confirmed_errors']


@pytest.mark.parametrize('mutation', ['missing','extra','duplicate','outside','wrong_remaining'])
def test_false_queue_completion_or_outside_scope_is_rejected(mutation):
    plan, before, after = fixture()
    if mutation == 'missing': after.pop()
    elif mutation == 'extra': after.append(before[0])
    elif mutation == 'duplicate': after[-1] = after[0]
    elif mutation == 'outside': plan['events'][0]['claim_id'] = 'CLM:outside'
    else: plan['original_watchlist_449_remaining_after'] = 0
    with pytest.raises(ValueError): queue_summary(plan, before, after)


def test_identity_batch_cannot_change_claim_field_coverage():
    before = [dict(scope='node/claim', field='metadata.raw_text', denominator=10, present=10, nonempty=10)]
    after = deepcopy(before); after[0]['present'] = 9
    with pytest.raises(ValueError, match='coverage changed'): coverage_summary(before, after)
