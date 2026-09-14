from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_kg_remaining_scoped_literals import detail_references, scope_projection, owned_projection


def test_detail_count_is_exact_not_suffix_match():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE atoms(source_mention_id TEXT)')
    db.execute('CREATE TABLE mappings(source_id TEXT,target_id TEXT)')
    db.executemany('INSERT INTO atoms VALUES (?)',[('CLM_CONCEPT:a',),('CLM_CONCEPT:aa',)])
    db.executemany('INSERT INTO mappings VALUES (?,?)',[('CLM_CONCEPT:a','CUI:b'),('CUI:c','CLM_CONCEPT:a')])
    assert detail_references(db,{'CLM_CONCEPT:a'})=={'CLM_CONCEPT:a':dict(atoms=1,mappings_source=1,mappings_target=1)}
    assert detail_references(db,set())=={}
    db.close()


def test_node_projection_is_review_scope_not_a_complete_preimage():
    node=dict(id='CLM_CONCEPT:a',preferred_name='x',metadata=dict(anchor_role='imaging',unrelated='not copied'),definition='not copied')
    p=scope_projection(node)
    assert p['reviewed_metadata']==dict(anchor_role='imaging') and 'definition' not in p and 'metadata' not in p
    assert p['complete_record_not_saved']


def test_owned_edge_projection_retains_nonendpoint_payload_hash():
    edge=dict(source_id='CLM:a',target_id='CLM_CONCEPT:x',relation_type='about',metadata=dict(extra=7))
    a=owned_projection(7,edge);edge['target_id']='CLM_CONCEPT:y';b=owned_projection(7,edge)
    assert a['edge_sha256']!=b['edge_sha256'] and a['nonendpoint_sha256']==b['nonendpoint_sha256']
    assert 'metadata' not in a and a['metadata_keys']==['extra']
