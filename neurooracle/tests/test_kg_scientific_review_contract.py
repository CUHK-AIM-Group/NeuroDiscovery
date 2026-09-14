from copy import deepcopy
import pytest
from neurooracle.src.kg_scientific_review_contract import SCOPE_FIELDS,evaluate


def case():
    endpoint=dict(concept='reviewed concept',identity_proof=True,node_kind='entity',scope={f:'not_applicable' for f in SCOPE_FIELDS})
    obs=dict(subject=deepcopy(endpoint),object=deepcopy(endpoint),predicate='is_associated_with',negated=False,paper_key='pmid:1')
    return dict(observations=[deepcopy(obs),deepcopy(obs)])


def test_relation_is_not_evidence_deletion_or_replication():
    c=case();c['observations'][1].update(negated=True,paper_key='pmid:2')
    r=evaluate(c)
    assert r['relation_decision']=='share_relation' and r['polarity_difference']
    assert not r['automatic_evidence_deletion'] and not r['independent_studies_inferred'] and not r['consensus_inferred']


@pytest.mark.parametrize('field',SCOPE_FIELDS)
def test_explicit_scope_difference_is_barrier(field):
    c=case();c['observations'][0]['subject']['scope'][field]='A';c['observations'][1]['subject']['scope'][field]='B'
    assert evaluate(c)['relation_decision']=='keep_distinct_or_repair_invalid_endpoint'


@pytest.mark.parametrize('field',SCOPE_FIELDS)
def test_missing_scope_is_not_equality_even_if_both_unknown(field):
    c=case()
    for o in c['observations']:o['object']['scope'][field]='unknown'
    assert evaluate(c)['relation_decision']=='hold_identity_or_scope'


def test_claim_node_cannot_become_canonical_entity():
    c=case();c['observations'][0]['subject']['node_kind']='claim'
    assert evaluate(c)['relation_decision']=='keep_distinct_or_repair_invalid_endpoint'


def test_unknown_identity_never_approved_by_equal_labels():
    c=case();c['observations'][1]['subject']['identity_proof']=False
    assert evaluate(c)['relation_decision']=='hold_identity_or_scope'


@pytest.mark.parametrize('predicate',['causes','distinguishes','predicts','reduces','increases'])
def test_different_predicates_cannot_be_smuggled_through_review_flag(predicate):
    c=case();c['predicate_equivalence_reviewed']=True;c['observations'][1]['predicate']=predicate
    assert evaluate(c)['relation_decision']=='keep_distinct_predicates'


def test_association_predicates_require_pair_specific_source_review():
    c=case();c['observations'][1]['predicate']='correlates_with'
    assert evaluate(c)['relation_decision']=='keep_distinct_predicates'
    c['predicate_equivalence_reviewed']=True
    assert evaluate(c)['relation_decision']=='share_relation'


@pytest.mark.parametrize('issue',['causal_overstatement','classifier_overstatement','aim_not_result','source_internal_conflict','hypothesis_not_result','analysis_scope_overstatement','insufficient_source'])
def test_scientific_hold_overrides_apparent_identity_match(issue):
    c=case();c['assertion_issue']=issue
    assert evaluate(c)['relation_decision']=='hold_scientific_assertion'


def test_denominators_do_not_create_false_independent_studies():
    c=case();c['observations'][0]['sample_size']=38;c['observations'][1]['sample_size']=53
    r=evaluate(c);assert r['denominator_must_remain_per_analysis'] and not r['independent_studies_inferred']


@pytest.mark.parametrize('value',[None,0,1,'false','true'])
def test_nonboolean_negation_is_not_coerced(value):
    c=case();c['observations'][0]['negated']=value
    with pytest.raises(ValueError):evaluate(c)


def test_unrecognized_assertion_issue_fails_closed():
    c=case();c['assertion_issue']='unrecognised scientific issue'
    with pytest.raises(ValueError):evaluate(c)


def test_string_identity_review_and_unknown_node_kind_fail_closed():
    c=case();c['observations'][0]['subject']['identity_proof']='true'
    with pytest.raises(ValueError):evaluate(c)
    c=case();c['observations'][0]['subject']['node_kind']='unknown'
    with pytest.raises(ValueError):evaluate(c)
