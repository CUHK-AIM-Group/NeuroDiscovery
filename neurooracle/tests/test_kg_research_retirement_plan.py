from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_research_statement_retirement import expected_census, selected_reference_closure


def db_fixture():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE claims(cid TEXT,paper_sig TEXT,legacy_key TEXT)')
    db.execute('CREATE TABLE papers(sig TEXT,pmid TEXT,doi TEXT)')
    db.executemany('INSERT INTO papers VALUES (?,?,?)',[('p1','1','doi1'),('p2','2','doi2'),('p3','','doi3'),('p4','2','doi2')])
    db.executemany('INSERT INTO claims VALUES (?,?,?)',[('a','p1','s1'),('b','p2','s2'),('c','p2','s2'),('d','p3','s3'),('e','p4','s2')])
    return db


def test_retiring_last_claim_updates_active_bibliography_and_source_counts():
    db=db_fixture()
    assert expected_census(db,{'a'},['b','c'])==dict(claims=4,unique_bibliographies=3,distinct_legacy_source_keys=2,
        shared_claims=2,missing_pmid_with_doi_claims=1,both_ids=1)
    assert db.execute('SELECT COUNT(*) FROM claims').fetchone()[0]==5


def test_bibliography_or_identifier_pair_survives_other_claims():
    db=db_fixture();out=expected_census(db,{'b','e'},[])
    assert out['unique_bibliographies']==3 and out['both_ids']==2 and out['distinct_legacy_source_keys']==3


def test_missing_pmid_doi_claim_counter_is_not_a_bibliography_counter():
    db=db_fixture();out=expected_census(db,{'d'},[])
    assert out['missing_pmid_with_doi_claims']==0 and out['both_ids']==2


def test_foreign_exact_reference_prevents_deletion():
    selected={'a'};removed=[dict(claim_id='a',owned_edges=[dict(ordinal=2)])]
    records=[dict(kind='node',key='a',references=[dict(claim_id='a')]),dict(kind='edge',key=2,references=[dict(claim_id='a')])]
    assert selected_reference_closure(records,selected,removed)==records
    records.append(dict(kind='node',key='b',references=[dict(claim_id='a')]))
    with pytest.raises(ValueError,match='unexpected exact'):selected_reference_closure(records,selected,removed)
