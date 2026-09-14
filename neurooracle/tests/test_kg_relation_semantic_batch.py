from copy import deepcopy
import pytest
from neurooracle.src import kg_relation_semantic_batch as r
from neurooracle.src.kg_identity_pilot import digest

@pytest.mark.parametrize('name',list(r.TARGETS))
def test_only_finite_case_variants(name):
    assert r.canonical_name(name)==name
    assert r.canonical_name(name[0].upper()+name[1:])==name
    assert r.canonical_name(name.upper()) is None

@pytest.mark.parametrize('name',['bilateral hippocampal volume','right hippocampal volume','mean bilateral hippocampal volume',
 'sum of left and right hippocampal volumes','hippocampal subfield volumetry and multimodal neuroimaging',
 'cortical thickness and white matter microstructure features','hippocampal volumes','VCP','FA','left hippocampal subfield volume'])
def test_scientific_qualifiers_and_compounds_never_collapsed(name):assert r.canonical_name(name) is None

def test_hyphen_is_one_finite_exception():
    assert r.canonical_name('white-matter fractional anisotropy')=='white matter fractional anisotropy'
    assert r.canonical_name('white-matter microstructure') is None

def row(cid='CLM:test'):
    return dict(id=cid,preferred_name='alpha correlates_with beta',metadata=dict(subject_id='A',subject_name='alpha',
        object_id='B',object_name='beta',predicate='correlates_with',negated=False,raw_text='Exact raw source.',
        source_paper=dict(pmid='123'),scope_reaudit=dict(decision='old'),metadata={},evidence=dict(sample_size=None)))

def event_and_current(record,fields):
    out=r.rewrite(record,fields)
    e=dict(claim_id=record['id'],claim_sha256=digest(record),current_node_sha256=digest(out),
        field_changes=[dict(path=list(p),old=o,new=n) for p,o,n in fields],protected_fields_sha256=r.protected_digest(record,fields))
    return e,out

def edges(cid='CLM:test'):
    return [(1,dict(source_id=cid,target_id='A',relation_type='about',metadata={})),
            (2,dict(source_id=cid,target_id='B',relation_type='about',metadata={})),
            (3,dict(source_id='A',target_id='B',relation_type='correlates_with',metadata=dict(claim_id=cid,negated=False,original_predicate='old'),evidence_ref='untouched'))]

def test_exact_field_inverse_no_full_preimage():
    original=row();e,current=event_and_current(original,[(('metadata','subject_id'),'A','C')])
    assert r.reverse_claim(current,e)==original
    assert current['metadata']['raw_text']==original['metadata']['raw_text']

@pytest.mark.parametrize('field',['raw_text','source_paper','scope_reaudit'])
def test_inverse_rejects_nonapproved_science_or_history(field):
    e,current=event_and_current(row(),[(('metadata','subject_id'),'A','C')]);current['metadata'][field]='tampered'
    with pytest.raises(ValueError):r.reverse_claim(current,e)

@pytest.mark.parametrize('count',[2,3])
def test_owned_closure_zero_or_one_materialized_science(count):
    original=row();current=deepcopy(original);current['metadata']['subject_id']='C'
    events=r.reviewed_edges('CLM:test',original,current,edges()[:count])
    assert len(events)==count-1
    for e in events:
        old=dict(edges())[e['ordinal']];out=r.apply_edge(old,e)
        assert r.apply_edge(out,e,reverse=True)==old

@pytest.mark.parametrize('bad',['missing','duplicate_about','duplicate_science','wrong_owner','wrong_predicate','wrong_negation'])
def test_bad_edge_closure_rejected(bad):
    original=row();values=edges()
    if bad=='missing':values=values[1:]
    elif bad=='duplicate_about':values.append((4,deepcopy(values[0][1])))
    elif bad=='duplicate_science':values.append((4,deepcopy(values[-1][1])))
    elif bad=='wrong_owner':values[-1][1]['metadata']['claim_id']='CLM:other'
    elif bad=='wrong_predicate':values[-1][1]['relation_type']='causes'
    else:values[-1][1]['metadata']['negated']=True
    with pytest.raises((ValueError,RuntimeError)):r.reviewed_edges('CLM:test',original,original,values)

def test_negation_is_changed_on_existing_edge_only():
    original=row();current=deepcopy(original);current['metadata']['negated']=True
    events=r.reviewed_edges('CLM:test',original,current,edges());assert len(events)==1
    after=r.apply_edge(edges()[-1][1],events[0]);assert after['metadata']['negated'] is True
    assert after['metadata']['original_predicate']=='old' and after['evidence_ref']=='untouched'

def test_edge_negation_not_silently_added():
    original=row();current=deepcopy(original);current['metadata']['negated']=True;values=edges();del values[-1][1]['metadata']['negated']
    with pytest.raises(ValueError):r.reviewed_edges('CLM:test',original,current,values)

def test_retired_exact_ids_not_substrings():
    p=dict(removed_nodes=[dict(node_id='OLD')])
    r.no_retired_references('node','X',dict(text='OLD appears in prose'),p)
    with pytest.raises(ValueError):r.no_retired_references('node','X',dict(metadata=dict(id='OLD')),p)

@pytest.mark.parametrize('key,value',[('aliases',['same']),('semantic_types',['T028']),('external_ids',{'x':'y'}),
 ('definition_sha256',digest('measurement')),('spatial_mapping_sha256',digest({'x':1})),('metadata_keys',['new']),('source_vocab','ontology')])
def test_defined_or_external_witness_not_approved(key,value):
    w=dict(aliases=[],semantic_types=[],external_ids={},definition_sha256=digest(''),spatial_mapping_sha256=digest(None),metadata_keys=[],source_vocab='replay_anchor_mint')
    w[key]=value
    with pytest.raises(ValueError):r.check_witness(w)

def test_five_finite_science_repairs_and_sample_boundaries():
    assert set(r.SCIENCE_FIELDS)==r.SCIENCE and len(r.SCIENCE)==5
    assert r.SCIENCE_FIELDS[r.SMA]['metadata','evidence','sample_size']==883
    assert r.SCIENCE_FIELDS[r.GBA]['metadata','evidence','sample_size']==53
    assert r.SCIENCE_FIELDS[r.EXERCISE]['metadata','negated'] is True
    assert r.SCIENCE_FIELDS[r.MDD]['metadata','evidence','sample_size'] is None
    assert all('raw_text' not in path and 'source_paper' not in path and 'scope_reaudit' not in path for d in r.SCIENCE_FIELDS.values() for path in d)

def test_one_reusable_new_literal_not_actual_dementia_or_gene():
    assert r.NEW_NODE['preferred_name']==r.GBA_NAME
    assert not r.NEW_NODE['semantic_types'] and not r.NEW_NODE['external_ids']
    assert r.NEW_NODE==r.literal_node(r.GBA_NAME)
