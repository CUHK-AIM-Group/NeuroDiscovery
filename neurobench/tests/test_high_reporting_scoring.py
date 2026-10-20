import hashlib
import json
import sqlite3

import pytest

from neurobench.runner.high_reporting import summarize,csv_text
from neurobench.runner.high_scoring import blind_reports,parse_scores,DIMS


def test_all_denominator_and_unknown_partial_usage_remain_visible(tmp_path):
    db=sqlite3.connect(tmp_path/'state.sqlite3')
    db.execute('create table slots(case_id,condition_id,model_block,arm,status,receipt_path)')
    for arm,status in [('A','COMPLETE'),('B','UNKNOWN')]:
        folder=tmp_path/'attempts'/('model__'+arm)/'T03';folder.mkdir(parents=True)
        receipt=folder/'RECEIPT.json'
        receipt.write_text(json.dumps({'usage':{'usage_complete':True,'input_tokens':20,'output_tokens':10,'total_tokens':30},'elapsed_seconds':2}),encoding='utf-8')
        db.execute('insert into slots values(?,?,?,?,?,?)',('T03','model__'+arm,'model',arm,status,str(receipt)))
    for arm in ('A','B'):
        db.execute('insert into slots values(?,?,?,?,?,?)',('T15','model__'+arm,'model',arm,'PENDING',None))
    db.commit();db.close()
    metrics,coverage=summarize(tmp_path)
    assert coverage['planned_primary_slots']==4 and coverage['complete_bundles']['model']['complete_bundles']==0
    unknown=next(r for r in metrics if r['status']=='UNKNOWN')
    assert unknown['observed_total_tokens']==30 and unknown['complete_total_tokens'] is None
    assert not unknown['usage_complete_for_attempt']
    pending=next(r for r in metrics if r['status']=='PENDING')
    assert pending['observed_total_tokens'] is None
    assert 'UNKNOWN' in csv_text(metrics) and 'PENDING' in csv_text(metrics)


def test_blinding_stable_and_no_identity_or_metrics_in_reports():
    rows=[dict(arm=arm,condition_id='exact-model__'+arm,text='verbatim '+arm,
               answer_sha256=hashlib.sha256(arm.encode()).hexdigest(),elapsed_seconds=14,total_tokens=12345) for arm in 'ABCD']
    reports,mapping=blind_reports('T03','exact-model',rows)
    assert (reports,mapping)==blind_reports('T03','exact-model',list(reversed(rows)))
    assert 'exact-model' not in json.dumps(reports) and '12345' not in json.dumps(reports)
    assert {r['arm'] for r in mapping}==set('ABCD')
    assert {v[0]['report_text'] for v in reports.values()}=={'verbatim '+arm for arm in 'ABCD'}


def test_score_weighting_and_alias_identity():
    rows=[dict(model='Candidate01',run_index=1,brief_justification='fixture',**dict(zip(DIMS,(8,9,7))))]
    scores=parse_scores(json.dumps(rows),{'Candidate01'})
    assert scores[0]['quality_100']==81
    with pytest.raises(ValueError): parse_scores(json.dumps(rows*2),{'Candidate01'})
    rows[0]['planning_completeness']=True
    with pytest.raises(ValueError): parse_scores(json.dumps(rows),{'Candidate01'})
    rows[0]['planning_completeness']=11
    with pytest.raises(ValueError): parse_scores(json.dumps(rows),{'Candidate01'})
