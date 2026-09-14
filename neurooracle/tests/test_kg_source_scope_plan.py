from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plan_kg_source_scope_resolution import expected_shared


def database():
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE claims(cid TEXT,relation_id TEXT)')
    db.executemany('INSERT INTO claims VALUES (?,?)',[('a','old'),('b','old'),('c','new'),('d','solo'),('e','stable'),('f','stable')])
    prior=[{'members':[{'claim_id':cid} for cid in group]} for group in [('a','b'),('e','f')]]
    return db,prior


def test_deleting_one_of_two_shared_members_does_not_leave_singleton_shared():
    db,prior=database()
    assert expected_shared(db,[],{'a'},prior)==['e','f']
    assert db.execute('SELECT COUNT(*) FROM claims').fetchone()[0]==6
    db.close()


def test_identity_and_deletion_recomputed_together_without_losing_unaffected():
    db,prior=database()
    events=[dict(claim_id='b',old_relation_id='old',new_relation_id='new')]
    assert expected_shared(db,events,{'a'},prior)==['b','c','e','f']
    db.close()


def test_deleting_an_unshared_claim_keeps_existing_shared_groups():
    db,prior=database()
    assert expected_shared(db,[],{'d'},prior)==['a','b','e','f']
    db.close()
