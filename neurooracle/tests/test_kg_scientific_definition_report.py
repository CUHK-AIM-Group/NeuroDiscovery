from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_kg_scientific_definition_repair import queue_summary,coverage_summary,scope_validation,HISTORIC_THREE
from neurooracle.src import kg_scientific_definition_repair as repair
from neurooracle.src.metadata_field_audit import Coverage
from neurooracle.tests.test_kg_scientific_definition_repair import fixture


def queue_fixture():
    before=[dict(claim_id=cid,side=side) for cid,side in sorted(HISTORIC_THREE)]
    before.extend(dict(claim_id='CLM:other'+str(i),side='subject') for i in range(4))
    after=[r for r in before if r['claim_id'] not in {repair.ENIGMA,repair.FES}]
    p=dict(events=[dict(claim_id=cid,changes=repair.changes_for(cid)) for cid in repair.SPECS],
        old_watchlist_endpoints=7,held_endpoints=5,changed_endpoints=9,repaired_queued_endpoints=2,
        original_watchlist_449_remaining_after=1,original_watchlist_449_repaired=2)
    return p,before,after


def test_only_two_of_nine_changes_are_credited_to_endpoint_queue():
    r=queue_summary(*queue_fixture())
    assert r['expanded_repaired_this_batch']==2 and r['expanded_remaining']==5
    assert r['original_watchlist_remaining']==1 and r['original_watchlist_repaired_this_batch']==2


@pytest.mark.parametrize('kind',['missing','extra','duplicate','false_credit','historic_closed','wrong_change'])
def test_queue_scope_and_historic_remaining_are_exact(kind):
    p,b,a=queue_fixture()
    if kind=='missing':a.pop()
    elif kind=='extra':a.append(b[0])
    elif kind=='duplicate':a[-1]=a[0]
    elif kind=='false_credit':p['repaired_queued_endpoints']=9
    elif kind=='historic_closed':p['original_watchlist_449_remaining_after']=0
    else:p['events'][-1]['changes'][0]['side']='object'
    with pytest.raises(ValueError):queue_summary(p,b,a)


def coverage_fixture():
    before=Coverage();after=Coverage()
    for i in range(10):
        row=dict(metadata=dict(raw_text='source',metadata=dict(subject_type='IMAGING_MARKER'),
            evidence=dict(direction='' if i<3 else 'positive',p_value=None,effect_size=None,sample_size=10)))
        before.add('node/claim',row)
        row['metadata']['evidence']['direction']='positive';after.add('node/claim',row)
    return before.rows(),after.rows()


def test_only_three_nonempty_direction_values_change_coverage():
    r=coverage_summary(*coverage_fixture())
    assert r['changed_direction_values']==3 and r['direction_nonempty_before']==7 and r['direction_nonempty_after']==10
    assert r['claim_fields_with_coverage_change']==1 and r['statistical_field_coverage_unchanged']


@pytest.mark.parametrize('kind',['two','four','statistic','new_field','missing_direction','type'])
def test_coverage_cannot_mask_other_changes(kind):
    b,a=coverage_fixture()
    if kind in ('two','four'):
        row=next(r for r in a if r['field']=='metadata.evidence.direction');row['nonempty']+=1 if kind=='four' else -1
    elif kind=='statistic':next(r for r in a if r['field']=='metadata.evidence.sample_size')['nonempty']-=1
    elif kind=='new_field':a.append(dict(a[0],field='metadata.extra'))
    elif kind=='missing_direction':b=[r for r in b if r['field']!='metadata.evidence.direction']
    else:next(r for r in a if r['field']=='metadata.evidence.direction')['types']={'NoneType':10}
    with pytest.raises(ValueError):coverage_summary(b,a)


def scope_fixture():
    events=[]
    for cid in repair.SPECS:
        row,reviews,proof=fixture(cid);event,_=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews);events.append(event)
    bycid={e['claim_id']:e for e in events};prenatal='CLM:CASE1MAN:36716140:8023'
    before=dict(current_claim_hashes={repair.BACE:bycid[repair.BACE]['claim_sha256'],repair.ACC:bycid[repair.ACC]['claim_sha256'],prenatal:'old'},
        findings=[dict(claim_id=cid) for cid in (repair.BACE,repair.ACC,prenatal)],resolved={})
    p=dict(events=events,unmodified_sample_size_reviews={cid:dict(claim_sha256=cid,science=dict(evidence=dict(sample_size=170)),
        source_paper=dict(pmid='30384145')) for cid in ('male','female')})
    graph={'sha256':'graph'};plan_fp={'sha256':'plan'}
    after=repair.update_scope_findings(before,p);after.update(at='now',graph=graph,latest_scientific_definition_plan=plan_fp)
    return p,before,after,graph,plan_fp


def test_remaining_science_and_historical_audits_stay_separately_registered():scope_validation(*scope_fixture())


@pytest.mark.parametrize('kind',['closed','audit','hash','missing','sample'])
def test_scientific_register_cannot_silently_clear_unknowns(kind):
    p,b,a,g,fp=scope_fixture()
    if kind=='closed':a['issue_register_is_exhaustive']=True
    elif kind=='audit':a['old_audits_not_revalidated']=False
    elif kind=='hash':a['current_claim_hashes'][repair.ENIGMA]='wrong'
    elif kind=='missing':a['findings'].pop()
    else:p['unmodified_sample_size_reviews']['male']['science']['evidence']['sample_size']=90
    with pytest.raises(ValueError):scope_validation(p,b,a,g,fp)
