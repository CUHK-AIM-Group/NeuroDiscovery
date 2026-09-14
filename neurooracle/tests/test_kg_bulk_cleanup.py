from collections import Counter
from copy import deepcopy
import pytest

from neurooracle.src.kg_bulk_cleanup import (
    EMPTY_HINTS,SCOPE_ALIASES,simplify_record,verify_simplification,legacy_scope_metadata,symmetric_correlation_key)
from neurooracle.src.case_study_membership_contract import claim_evidence_payload
from neurooracle.src.claim_evidence_identity import evidence_signature
from neurooracle.src.relation_evidence import evidence_context


def row():
    return dict(id='CLM:test',preferred_name='original full name',metadata=dict(id='CLM:test',subject_id='A',
        subject_name='whole measured quantity',object_id='B',object_name='complete outcome',predicate='correlates_with',
        negated=False,confidence=0.8,raw_text='Original evidence',evidence=dict(p_value=None,effect_size=0,unknown=False),
        source_paper=dict(pmid='123',title='Original paper'),subject_type='IMAGING_MARKER',conditions=['condition'],
        scope_reaudit=dict(confidence=0.8,decision_basis='Original review',rubric_version='v4',review_status='final_complete',
            review_stage='initial',claim_evidence_sha256='original-seal'),
        metadata=dict(subject_type='IMAGING_MARKER',conditions=['condition'],subject_canonical_hint='',object_canonical_hint='',
            subject_atlas='',object_atlas='',raw_stats={},scope_confidence=0.8,scope_decision_basis='Original review',
            scope_rubric_version='v4',scope_review_status='final_complete',scope_assignment_stage='initial',batch_id='keep')))


def test_exact_bulk_cleanup_science_audit_and_context_invariant():
    source=row();saved=deepcopy(source);counts=Counter();out=simplify_record('node',source,counts)
    assert source==saved
    assert out['metadata']['metadata']=={'batch_id':'keep'}
    assert sum(counts.values())==12
    verify_simplification(source,out)
    assert simplify_record('node',out)==out
    args=dict(scope_context_sha256='original',include_legacy_case2_chain=True)
    assert claim_evidence_payload(source['metadata'],**args)==claim_evidence_payload(out['metadata'],**args)
    assert evidence_context(source['metadata'])==evidence_context(out['metadata'])


@pytest.mark.parametrize('value',[None,False,0,0.0,' ','unknown',[],[None],{}, {'n':None}])
@pytest.mark.parametrize('field',EMPTY_HINTS)
def test_no_broad_empty_or_placeholder_interpretation(field,value):
    source=row();source['metadata']['metadata'][field]=value
    out=simplify_record('node',source)
    assert type(out['metadata']['metadata'][field]) is type(value)
    assert out['metadata']['metadata'][field]==value


@pytest.mark.parametrize('value',[None,'',[],False,0,{'p_value_raw':'not reported'},{'p_value_raw':None}])
def test_raw_statistics_unique_values_never_removed(value):
    source=row();source['metadata']['metadata']['raw_stats']=value
    assert simplify_record('node',source)['metadata']['metadata']['raw_stats']==value


@pytest.mark.parametrize('value',[None,False,0,0.0,'',[],{}])
def test_scientific_falsy_fallback_is_unchanged(value):
    source=row();source['metadata']['population']=value;source['metadata']['metadata']['population']=deepcopy(value)
    out=simplify_record('node',source)
    assert 'population' in out['metadata']['metadata']


@pytest.mark.parametrize('alias,target',list(SCOPE_ALIASES.items()))
def test_scope_conflicts_and_missing_canonical_are_preserved(alias,target):
    source=row();source['metadata']['metadata'][alias]='conflicting value'
    out=simplify_record('node',source)
    assert out['metadata']['metadata'][alias]=='conflicting value'
    del source['metadata']['scope_reaudit'][target]
    assert alias in simplify_record('node',source)['metadata']['metadata']


def test_scope_compatibility_view_keeps_conflicts_and_copies_values():
    source=row();out=simplify_record('node',source)
    assert legacy_scope_metadata(out['metadata'])['scope_confidence']==0.8
    out['metadata']['metadata']['scope_confidence']=0
    assert legacy_scope_metadata(out['metadata'])['scope_confidence']==0
    assert out['metadata']['scope_reaudit']['confidence']==0.8


@pytest.mark.parametrize('mutation',['evidence','audit','raw_text','unique','top','type'])
def test_independent_verifier_rejects_out_of_scope_changes(mutation):
    source=row();out=deepcopy(simplify_record('node',source))
    if mutation=='evidence':out['metadata']['evidence']['p_value']=0
    elif mutation=='audit':out['metadata']['scope_reaudit']['confidence']=0.1
    elif mutation=='raw_text':out['metadata']['raw_text']='different'
    elif mutation=='unique':del out['metadata']['metadata']['batch_id']
    elif mutation=='top':out['preferred_name']='shortened'
    else:out['metadata']['metadata']['batch_id']=False
    with pytest.raises(ValueError):verify_simplification(source,out)


@pytest.mark.parametrize('predicate',['causes','predicts','treats','is_associated_with'])
def test_only_exact_correlation_predicate_is_symmetric(predicate):
    key=('Z','z',predicate,'A','a')
    assert symmetric_correlation_key(key)==key


def test_correlation_full_names_and_ids_preserved():
    key=('Z','complete left quantity','correlates_with','A','full outcome')
    result=symmetric_correlation_key(key)
    assert result==('A','full outcome','correlates_with','Z','complete left quantity')
    assert symmetric_correlation_key(result)==result


def test_nonclaim_nodes_and_edges_unchanged():
    record=row();record['id']='CUI:test'
    assert simplify_record('node',record) is record
    record=row();assert simplify_record('edge',record) is record
