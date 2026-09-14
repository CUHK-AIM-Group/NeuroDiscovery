from copy import deepcopy
import pytest

from neurooracle.src import kg_systematic_consolidation as r73
from neurooracle.src.claim_ingestion import resolve_claim_entities
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_reviewed_relation_repair import reimport_rule,revise_claim
from neurooracle.src.kg_root_application import reverse_event
from neurooracle.src.schema import Claim,ConceptNode


def fixture(monkeypatch):
    md=dict(id='CLM:r73',subject_id='CUI:wrong_gene',subject_name='left hippocampal volume',subject_type='biomarker',
        object_id='CUI:disease',object_name='disease',object_type='DISEASE',predicate='correlates_with',negated=False,
        source_paper={'pmid':'12345','title':'Own primary source'},raw_text='A left hippocampal volume measurement was reported.',
        metadata={'subject_type':'biomarker','unknown':[False,0,None],'subject_id':'CUI:wrong_gene'},
        evidence={'sample_size':0,'direction':'negative','unknown':None})
    before=dict(id=md['id'],preferred_name='left hippocampal volume correlates_with disease',metadata=md,unknown='retain')
    node=r73.literal(md['subject_name']);d=ConceptNode(id='CUI:disease',preferred_name='disease').to_dict()
    event,after=revise_claim(before,{'subject_id':node['id']},review_id='closed_family')
    rule=reimport_rule(before,after,{node['id']:node,d['id']:d},'closed_family')
    kg=KnowledgeGraph()
    for n in (node,d):kg.add_concept(ConceptNode.from_dict(n))
    policy=dict(version=r73.VERSION,rules=[rule]);kg.serialization_metadata['systematic_consolidation']=policy
    monkeypatch.setattr(r73,'APPROVED_RULESET_DIGEST',digest(policy['rules']))
    return kg,before,after,event


def test_reimport_and_complete_inverse_preserve_evidence(monkeypatch):
    kg,before,after,event=fixture(monkeypatch)
    claim=Claim.from_dict(before['metadata']);old=deepcopy(claim.to_dict())
    assert resolve_claim_entities(kg,claim) is claim
    assert claim.subject_id==after['metadata']['subject_id']
    for k in ('raw_text','source_paper','evidence','negated'):assert claim.to_dict()[k]==old[k]
    assert claim.metadata['unknown']==old['metadata']['unknown']
    assert reverse_event(after,event)==before


@pytest.mark.parametrize('field,value',[('raw_text','another observation'),('subject_name','right hippocampal volume'),
    ('subject_name','hippocampal volume'),('subject_name','left hippocampal atrophy'),('negated',True),('predicate','causes')])
def test_unreviewed_scope_does_not_match(monkeypatch,field,value):
    kg,before,_,_=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);setattr(claim,field,value)
    assert r73.apply_systematic_reimport(kg,claim) is None


@pytest.mark.parametrize('problem',['target_missing','target_modified','supplied_id','nested_id','policy'])
def test_conflicts_reject_before_mutation(monkeypatch,problem):
    kg,before,after,_=fixture(monkeypatch);claim=Claim.from_dict(before['metadata'])
    if problem=='target_missing':kg._index.pop(after['metadata']['subject_id'])
    elif problem=='target_modified':kg.get_concept(after['metadata']['subject_id']).domain_tags=['gene']
    elif problem=='supplied_id':claim.subject_id='CUI:other'
    elif problem=='nested_id':claim.metadata['subject_id']='CUI:other'
    else:kg.serialization_metadata['systematic_consolidation']['rules'][0]['pmid']='999'
    old=deepcopy(claim.to_dict())
    with pytest.raises(ValueError):r73.apply_systematic_reimport(kg,claim)
    assert claim.to_dict()==old


def test_duplicate_source_rows_may_converge_only_to_identical_output(monkeypatch):
    kg,before,after,_=fixture(monkeypatch);rules=kg.serialization_metadata['systematic_consolidation']['rules']
    second=deepcopy(rules[0]);second['review_id']='second_same_observation';rules.append(second)
    monkeypatch.setattr(r73,'APPROVED_RULESET_DIGEST',digest(rules))
    assert r73.apply_systematic_reimport(kg,Claim.from_dict(before['metadata'])).subject_id==after['metadata']['subject_id']
    second['new_relation']['subject_id']='CUI:inconsistent';monkeypatch.setattr(r73,'APPROVED_RULESET_DIGEST',digest(rules))
    with pytest.raises(ValueError,match='ambiguous'):r73.apply_systematic_reimport(kg,Claim.from_dict(before['metadata']))


def test_literal_identity_preserves_laterality_compound_and_case():
    names=['left hippocampal volume','right hippocampal volume','hippocampal volume','hippocampal volume and thickness','Hippocampal volume']
    nodes=[r73.literal(n) for n in names]
    assert len({n['id'] for n in nodes})==len(names)
    assert [n['preferred_name'] for n in nodes]==names


def test_source_mention_type_update_uses_inverse_without_other_changes():
    before=ConceptNode(id='CLM_CONCEPT:unclassified_mention_bound',preferred_name='ALPS index',domain_tags=['external']).to_dict()
    before['metadata']={'unknown':['keep']}
    event,after=r73.type_event(before,['imaging_feature'],review_id='source_method')
    patch=r73.SystematicTransform([],[],[],[],[],[event])
    assert list(patch.records([('node',before['id'],before)]))==[('node',before['id'],after)]
    assert reverse_event(after,event)==before


@pytest.mark.parametrize('problem',['external','typed','nonliteral'])
def test_canonical_identity_cannot_be_retyped(problem):
    before=ConceptNode(id='CLM_CONCEPT:unclassified_mention_bound',preferred_name='mention',domain_tags=['external']).to_dict()
    if problem=='external':before['semantic_types']=['T028']
    elif problem=='typed':before['domain_tags']=['gene']
    else:before['id']='CUI:bound'
    with pytest.raises(ValueError):r73.type_event(before,['imaging_feature'],review_id='source_method')


def owned_self_edges():
    cid='CLM:self';old='CUI:wrong';before={'id':cid,'metadata':{'subject_id':old,'object_id':old,'predicate':'correlates_with'}}
    after=deepcopy(before);after['metadata']['object_id']='CLM_CONCEPT:measurement'
    owned=[(i,dict(source_id=cid,target_id=old,relation_type='about',metadata={'claim_id':cid,'anchor_role':side}))
        for i,side in enumerate(('subject','object'))]
    owned.append((2,dict(source_id=old,target_id=old,relation_type='correlates_with',metadata={'claim_id':cid})))
    return before,after,owned


def test_explicit_owned_roles_allow_removal_of_false_original_self_link():
    before,after,owned=owned_self_edges();events=r73.revise_owned_edges(before,after,owned)
    assert [e['ordinal'] for e in events]==[1,2]
    from neurooracle.src.kg_root_repairs import apply_event
    for event in events:
        old=dict(owned)[event['ordinal']];new=apply_event(old,event)
        assert new['target_id']==after['metadata']['object_id']
        assert new['metadata']==old['metadata'] and reverse_event(new,event)==old


@pytest.mark.parametrize('problem',['roleless','duplicate','conflicting_role','foreign_owner','missing_object'])
def test_ambiguous_or_incomplete_owned_edges_are_held(problem):
    before,after,owned=owned_self_edges()
    if problem=='roleless':owned[0][1]['metadata'].pop('anchor_role')
    elif problem=='duplicate':owned[1][1]['metadata']['anchor_role']='subject'
    elif problem=='conflicting_role':owned[1][1]['target_id']='CUI:other'
    elif problem=='foreign_owner':owned[2][1]['metadata']['claim_id']='CLM:other'
    else:owned.pop(1)
    with pytest.raises(ValueError):r73.revise_owned_edges(before,after,owned)
