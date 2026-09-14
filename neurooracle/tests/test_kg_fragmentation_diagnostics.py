from copy import deepcopy
import sqlite3
import pytest

from neurooracle.src import kg_fragmentation_diagnostics as d
from neurooracle.src.correlation_grouping import IndexTerms
from neurooracle.src.relation_evidence import relation_id
from neurooracle.scripts import inspect_kg_fragmentation_roots as r
from neurooracle.scripts import inspect_kg_current_relation_reuse as old


class Terms(IndexTerms):
    def term_for(self,claim,side):return None


class Papers:
    def resolve(self,md):return dict(paper_key=md['source_paper']['key'],status=md['source_paper'].get('status','verified'))
    def publication_review(self,pk):return dict(status='retracted' if 'retracted' in pk else 'not_reviewed')


def claim(cid='CLM:a',**kwargs):
    md=dict(id=cid,subject_id='N:a',subject_name='Left hippocampal volume',predicate='is_associated_with',
        object_id='N:b',object_name='Memory',negated=False,source_paper=dict(key='pmid:1'),evidence={},metadata={})
    md.update(kwargs);return dict(id=cid,metadata=md)


def card(**kwargs):
    row=dict(sn='hippocampal volume',tn='memory',p='is_associated_with',neg='false',ctx='one')
    row.update(kwargs);return row


def db_for(records):
    db=sqlite3.connect(':memory:');r.initialize(db);batch=[]
    for row in records:
        md=row['metadata'];base=old.claim_projection(row,Terms(),Papers());extra,_=d.extra_keys(md,Terms())
        extra['alias_route']=extra['orthographic']
        batch.append((*base,*[extra[k] for k in r.EXTRAS],'null','null','{}','{}'))
    r.append_claims(db,batch);return db


def test_probe_does_not_change_original():
    md=claim()['metadata'];before=deepcopy(md);d.extra_keys(md,Terms());assert md==before


@pytest.mark.parametrize('a,b',[('grey matter volume','gray-matter volume'),('white‐matter FA','white matter FA'),("Parkinson’s disease","Parkinson's disease")])
def test_orthography_probe_is_explicit(a,b):assert d.orthographic(a)==d.orthographic(b)


def test_symbols_casefold_only_proposes_not_approves():
    a,b=card(sn='Vcp'),card(sn='VCP')
    assert d.orthographic(a['sn'])==d.orthographic(b['sn'])
    assert d.semantic_decision(a,b,identity_proof=True)['decision']=='hold_identity_or_scope_evidence'


@pytest.mark.parametrize('a,b',[('left hippocampal volume','right hippocampal volume'),('total volume','mean volume'),
    ('cortical thickness','cortical volume'),('PET signal','fMRI signal'),('reduced volume','increased volume')])
def test_explicit_scope_differences_are_not_merges(a,b):
    x=d.semantic_decision(card(sn=a),card(sn=b),identity_proof=True,scope_reviewed=True)
    assert x['decision']=='keep_distinct_scientific_scope' and not x['evidence_merge_allowed']


@pytest.mark.parametrize('name',['left hippocampal volume','mean hippocampal volume','reduced hippocampal volume'])
def test_unspecified_scope_is_not_same_scope(name):
    assert d.semantic_decision(card(sn=name),card(),identity_proof=True)['decision']=='hold_identity_or_scope_evidence'


def test_shared_relation_keeps_negative_context_and_cohorts_separate():
    x=d.semantic_decision(card(),card(neg='true',ctx='other'),identity_proof=True)
    assert x['decision']=='share_relation_preserve_each_observation'
    assert not x['evidence_merge_allowed'] and not x['scientific_consensus_inferred'] and not x['independent_studies_inferred']


def test_causal_correlation_not_silently_equated():
    assert d.semantic_decision(card(p='causes'),card(),identity_proof=True)['decision']=='keep_distinct_predicates'
    assert d.semantic_decision(card(p='causes'),card(),identity_proof=True,predicate_proof=True)['decision']=='keep_distinct_predicates'


def test_unproved_exact_name_still_needs_identity():
    assert d.semantic_decision(card(),card())['decision']=='hold_identity_or_scope_evidence'


def test_registry_only_eligible_complete_term_can_route():
    class T(Terms):
        def term_for(self,md,side):
            return dict(target_id='CUI:1',canonical_name='Whole volume') if side=='subject' else None
    md=claim()['metadata'];keys,sides=d.extra_keys(md,T())
    assert sides==['subject'] and keys['registry']!=relation_id(Terms().relation_key(md))
    assert md['subject_id']=='N:a'


def test_scope_probe_is_deliberately_unsafe_and_labeled():
    assert d.scoped_probe('left mean hippocampal volume')==d.scoped_probe('right total hippocampal volume')
    assert 'candidate' in d.PROBE_ONLY


def test_accepted_correlation_symmetry_is_not_extended_to_causality():
    a=claim(predicate='causes')['metadata'];b=deepcopy(a)
    b['subject_name'],b['object_name']=a['object_name'],a['subject_name']
    ka,_=d.extra_keys(a,Terms());kb,_=d.extra_keys(b,Terms())
    assert ka['orientation']!=kb['orientation'] and ka['topic']==kb['topic']


def test_stats_do_not_call_same_paper_two_sources():
    db=db_for([claim(),claim('CLM:b')])
    try:
        s=r.group_statistics(db,'rid');assert s['shared_groups']==1 and s['multi_source_key_groups']==0
    finally:db.close()


def test_name_id_split_is_a_candidate_not_fine_merge():
    db=db_for([claim(),claim('CLM:b',subject_id='N:other',source_paper=dict(key='pmid:2'))])
    try:
        a=r.group_statistics(db,'rid');b=r.group_statistics(db,'surface')
        assert a['groups']==2 and b['groups']==1 and b['cross_source_fragment_candidates']==1
        assert db.execute('SELECT COUNT(DISTINCT rid) FROM claims').fetchone()[0]==2
    finally:db.close()


def test_unverified_and_retracted_not_default_two_papers():
    db=db_for([claim(),claim('CLM:b',source_paper=dict(key='pmid:retracted')),claim('CLM:c',source_paper=dict(key='unknown:3',status='unverified'))])
    try:
        s=r.group_statistics(db,'rid');assert s['multi_source_key_groups']==1 and s['multi_verified_not_retracted_groups']==0
    finally:db.close()


@pytest.mark.parametrize('field',['bad','rid;DROP TABLE claims',''])
def test_finite_sql_fields(field):
    db=db_for([claim()])
    try:
        with pytest.raises(ValueError):r.group_statistics(db,field)
    finally:db.close()


def test_disjoint_masks_not_sum_of_probe_counts():
    assert d.partition_summary([{'mask':3},{'mask':1},{'mask':0}])==[
        dict(mask=0,fine_relations=1),dict(mask=1,fine_relations=1),dict(mask=3,fine_relations=1)]


def test_all_probe_columns_present_and_fine_key_unchanged():
    db=db_for([claim()])
    try:
        for field in d.PROBES:
            s=r.group_statistics(db,field);assert s['groups']==s['claims']==1
    finally:db.close()
