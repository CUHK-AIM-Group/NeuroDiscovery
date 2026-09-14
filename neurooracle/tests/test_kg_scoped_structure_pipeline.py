"""Tiny forward/inverse streaming witnesses; never read/write the real KG."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import apply_kg_scoped_structure as e
from neurooracle.src import kg_scoped_structure as r
from neurooracle.tests.test_kg_scoped_structure import fixture,about


def transform_fixture():
    row,ch,proof=fixture();event,current=r.reviewed_claim(row,ch,proof);edge=about(row)
    new=[r.literal_node(c['name']) for c in ch]
    added=r.reviewed_added_about(row['id'],row,current,[(1,edge)])[0]
    plan=dict(events=[event],edge_events=r.reviewed_edges(row['id'],row,current,[(1,edge)]),
        new_literals=[dict(id=n['id'],name=n['preferred_name'],node_sha256=r.digest(n)) for n in new],existing_targets={},
        added_about_edges=[dict(ordinal=2,row=added['row'],proof=added)])
    return row,edge,proof,plan


def test_source_digests_exclude_appended_structural_edge(monkeypatch):
    row,edge,proof,plan=transform_fixture();monkeypatch.setattr(e,'PUBLIC_PROOF',proof,raising=False)
    records=[('metadata','',{}),('node',row['id'],row),('edge','1',edge)]
    t=e.LiteralTransform(plan);out=list(t.records(iter(records)))
    assert sum(k=='node' for k,_,_ in out)==3 and sum(k=='edge' for k,_,_ in out)==2
    assert out[-1][2]['relation_type']=='about'
    assert e.reverse_claim(next(v for k,i,v in out if i==row['id']),plan['events'][0])==row
    import hashlib
    assert t.digests['edges'].hexdigest()==hashlib.sha256((e.compact(edge)+'\n').encode()).hexdigest()


def test_tampered_appended_edge_rejected(monkeypatch):
    row,edge,proof,plan=transform_fixture();monkeypatch.setattr(e,'PUBLIC_PROOF',proof,raising=False)
    plan['added_about_edges'][0]['row']=deepcopy(plan['added_about_edges'][0]['row'])
    plan['added_about_edges'][0]['row']['confidence']=.9
    with pytest.raises(ValueError):list(e.LiteralTransform(plan).records(iter([('node',row['id'],row),('edge','1',edge)])))


def test_engine_does_not_reopen_closed_night_window():
    code=Path(e.__file__).read_text(encoding='utf8')
    assert 'NIGHT_WINDOW_20260909.json' not in code
    assert 'verified_identity_proofs_complete=True' in code
    assert 'seen_added==set(added)' in code
    assert 'only appended about edges permitted' in code
