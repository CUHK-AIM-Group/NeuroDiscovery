from copy import deepcopy
import pytest
from neurooracle.src import kg_multipaper_harmonization as repair
from neurooracle.src.claim_ingestion import resolve_claim_entities
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_reviewed_relation_repair import revise_claim,reimport_rule
from neurooracle.src.kg_root_application import reverse_event
from neurooracle.src.schema import Claim,ConceptNode

def fixture(monkeypatch):
    target=repair.shared_literal('brain age gap','imaging_biomarker')
    disease=ConceptNode(id='CUI:disease',preferred_name='hypertension').to_dict()
    kg=KnowledgeGraph()
    for n in (target,disease):kg.add_concept(ConceptNode.from_dict(n))
    originals=[];finals=[];events=[];rules=[]
    for i,pmid in enumerate(('111','222')):
        md=dict(id='CLM:'+pmid,subject_id='CUI:brain' if i==0 else 'CLM_CONCEPT:old_bag',
            subject_name='brain-age gap' if i==0 else 'brain age gap',predicate='correlates_with',
            object_id=disease['id'],object_name='hypertension',negated=False,
            source_paper=dict(pmid=pmid,title='Source '+pmid),raw_text='BAG relates to hypertension in cohort '+pmid,
            metadata=dict(subject_type='biomarker',unknown=[0,False,None]),
            evidence=dict(sample_size=0 if i==0 else 70,effect_size=None,methodology='MRI '+pmid),
            population='cohort '+pmid,conditions=['adjusted for age'])
        before=dict(id=md['id'],preferred_name=md['subject_name']+' correlates_with hypertension',metadata=md,unknown='keep')
        event,after=revise_claim(before,dict(subject_id=target['id'],subject_name='brain age gap'),review_id='reviewed')
        rules.append(reimport_rule(before,after,{target['id']:target,disease['id']:disease},'reviewed'))
        originals.append(before);finals.append(after);events.append(event)
    monkeypatch.setattr(repair,'APPROVED_RULESET_DIGEST',digest(rules))
    kg.serialization_metadata['multipaper_claim_harmonization']=dict(version=repair.VERSION,rules=rules)
    return kg,originals,finals,events

def test_two_own_sources_converge_without_changing_evidence(monkeypatch):
    kg,before,after,events=fixture(monkeypatch);ids=[]
    for b,a,e in zip(before,after,events):
        claim=Claim.from_dict(b['metadata']);original=deepcopy(claim.to_dict())
        assert resolve_claim_entities(kg,claim) is claim
        assert claim.subject_id==a['metadata']['subject_id']
        ids.append(claim.subject_id)
        for key in ('source_paper','raw_text','evidence','negated','population','conditions'):
            assert claim.to_dict().get(key)==original.get(key)
        assert claim.metadata['unknown']==[0,False,None]
        assert reverse_event(a,e)==b
    assert len(set(ids))==1 and before[0]['metadata']['source_paper']!=before[1]['metadata']['source_paper']

@pytest.mark.parametrize('change',[('pmid','333'),('raw_text','Unreviewed sentence'),('subject_name','regional predicted age gap'),
    ('predicate','causes'),('negated',True),('role','gene')])
def test_unreviewed_sources_scales_roles_and_polarity_do_not_match(monkeypatch,change):
    kg,before,_,_=fixture(monkeypatch);claim=Claim.from_dict(before[0]['metadata']);key,value=change
    if key=='pmid':claim.source_paper.pmid=value
    elif key=='role':claim.metadata['subject_type']=value
    else:setattr(claim,key,value)
    original=deepcopy(claim.to_dict())
    assert repair.apply_multipaper_reimport(kg,claim) is None
    assert claim.to_dict()==original

@pytest.mark.parametrize('damage',['missing_target','target_type','supplied_id','nested_id','policy','ambiguous'])
def test_invalid_policy_or_target_fails_before_mutation(monkeypatch,damage):
    kg,before,after,_=fixture(monkeypatch);claim=Claim.from_dict(before[0]['metadata'])
    nid=after[0]['metadata']['subject_id'];rules=kg.serialization_metadata['multipaper_claim_harmonization']['rules']
    if damage=='missing_target':kg._index.pop(nid)
    elif damage=='target_type':kg.get_concept(nid).domain_tags=['gene']
    elif damage=='supplied_id':claim.subject_id='CUI:unapproved'
    elif damage=='nested_id':claim.metadata['subject_id']='CUI:unapproved'
    elif damage=='policy':rules[0]['pmid']='333'
    else:
        other=deepcopy(rules[0]);other['new_relation']['subject_id']='CUI:unapproved';rules.append(other)
        monkeypatch.setattr(repair,'APPROVED_RULESET_DIGEST',digest(rules))
    original=deepcopy(claim.to_dict())
    with pytest.raises(ValueError):repair.apply_multipaper_reimport(kg,claim)
    assert claim.to_dict()==original

def test_legacy_and_current_source_forms_reimport_to_same_shared_target(monkeypatch):
    kg,before,after,_=fixture(monkeypatch);rules=kg.serialization_metadata['multipaper_claim_harmonization']['rules']
    legacy=deepcopy(rules[0]);legacy['old_relation']['subject_id']='CUI:older_brain';rules.append(legacy)
    monkeypatch.setattr(repair,'APPROVED_RULESET_DIGEST',digest(rules))
    for nid in ('CUI:brain','CUI:older_brain',after[0]['metadata']['subject_id']):
        claim=Claim.from_dict(before[0]['metadata']);claim.subject_id=nid
        assert repair.apply_multipaper_reimport(kg,claim).subject_id==after[0]['metadata']['subject_id']

def test_shared_concepts_keep_variant_measurement_and_process_scopes_distinct():
    nodes=[repair.shared_literal(n,d) for n,d in [('APOE ε4 allele','genetic_variant'),
        ('APOE ε4 carrier status','genetic_variant'),('APOE expression','molecular_expression'),
        ('age','demographic_factor'),('aging','aging_process'),('brain age gap','imaging_biomarker')]]
    assert len({n['id'] for n in nodes})==len(nodes)
    assert not any(n['external_ids'] or n['semantic_types'] for n in nodes)
    for n in nodes:assert repair.MultipaperTransform([],[],[],[],[n],[]).new_nodes[n['id']]==n

@pytest.mark.parametrize('field,value',[('preferred_name','APOE'),('domain_tags',['gene']),
    ('definition','all assays are equivalent'),('external_ids',{'UMLS_CUI':'fake'})])
def test_new_concept_tampering_rejected(field,value):
    n=repair.shared_literal('APOE ε4 allele','genetic_variant');n[field]=value
    with pytest.raises(ValueError):repair.MultipaperTransform([],[],[],[],[n],[])


@pytest.mark.parametrize('prior_module,prior_key',[
    ('kg_multipaper_claim_repair','multipaper_claim_repair'),
    ('kg_multipaper_expansion','multipaper_claim_expansion'),
    ('kg_multipaper_semantics','multipaper_claim_semantics'),
    ('kg_multipaper_broadening','multipaper_claim_broadening'),
    ('kg_multipaper_aliases','multipaper_claim_aliases'),
    ('kg_multipaper_predicates','multipaper_claim_predicates')])
def test_prior_policy_remains_available_for_unrelated_owning_source(monkeypatch,prior_module,prior_key):
    from importlib import import_module
    previous_fixture=import_module('neurooracle.tests.test_'+prior_module).fixture
    previous=import_module('neurooracle.src.'+prior_module)
    kg,_,_,_=fixture(monkeypatch)
    prior_kg,before,after,_=previous_fixture(monkeypatch)
    policy=deepcopy(prior_kg.serialization_metadata[prior_key])
    for rule in policy['rules']:rule['pmid']+='9'
    monkeypatch.setattr(previous,'APPROVED_RULESET_DIGEST',digest(policy['rules']))
    kg.serialization_metadata[prior_key]=policy
    target=prior_kg.get_concept(after[0]['metadata']['subject_id'])
    kg.add_concept(ConceptNode.from_dict(target.to_dict()))
    claim=Claim.from_dict(before[0]['metadata']);claim.source_paper.pmid+='9'
    original=deepcopy(claim.to_dict())
    assert resolve_claim_entities(kg,claim) is claim
    assert claim.subject_id==target.id
    assert claim.raw_text==original['raw_text'] and claim.to_dict()['evidence']==original['evidence']


@pytest.mark.parametrize('prior_module,prior_key', [
    ('kg_multipaper_broadening','multipaper_claim_broadening'),
    ('kg_multipaper_aliases','multipaper_claim_aliases'),
    ('kg_multipaper_predicates','multipaper_claim_predicates')])
def test_latest_source_rule_takes_priority_over_matching_prior_policy(monkeypatch,prior_module,prior_key):
    from importlib import import_module
    prior_fixture=import_module('neurooracle.tests.test_'+prior_module).fixture
    kg,before,after,_=fixture(monkeypatch)
    prior_kg,_,prior_after,_=prior_fixture(monkeypatch)
    kg.serialization_metadata[prior_key]=deepcopy(prior_kg.serialization_metadata[prior_key])
    prior_target=prior_kg.get_concept(prior_after[0]['metadata']['subject_id'])
    kg.add_concept(ConceptNode.from_dict(prior_target.to_dict()))
    claim=Claim.from_dict(before[0]['metadata'])
    evidence=deepcopy(claim.to_dict())
    assert resolve_claim_entities(kg,claim) is claim
    assert claim.subject_id==after[0]['metadata']['subject_id']
    assert claim.subject_id!=prior_target.id
    for field in ('raw_text','source_paper','evidence','negated','population','conditions'):
        assert claim.to_dict().get(field)==evidence.get(field)


@pytest.mark.parametrize('old_predicate',['has_adverse_effect','predicts','modulates'])
def test_source_bound_association_calibration_preserves_observation_and_edges(monkeypatch,old_predicate):
    from neurooracle.src.kg_systematic_consolidation import revise_owned_edges
    from neurooracle.src.kg_root_repairs import apply_event
    kg,originals,finals,_=fixture(monkeypatch)
    before=deepcopy(originals[0]);md=before['metadata']
    md['predicate']=old_predicate;md['metadata']['predicate']=old_predicate
    before['preferred_name']=' '.join(md[k] for k in ('subject_name','predicate','object_name'))
    event,after=revise_claim(before,dict(subject_id=finals[0]['metadata']['subject_id'],
        subject_name='brain age gap',predicate='is_associated_with'),review_id='calibration')
    targets={after['metadata'][s+'_id']:kg.get_concept(after['metadata'][s+'_id']).to_dict() for s in ('subject','object')}
    rules=[reimport_rule(before,after,targets,'calibration')]
    monkeypatch.setattr(repair,'APPROVED_RULESET_DIGEST',digest(rules))
    kg.serialization_metadata['multipaper_claim_harmonization']['rules']=rules
    for source in (before,after):
        incoming=Claim.from_dict(source['metadata']);saved=deepcopy(incoming.to_dict())
        assert repair.apply_multipaper_reimport(kg,incoming) is incoming
        assert incoming.predicate==incoming.metadata['predicate']=='is_associated_with'
        for field in ('raw_text','source_paper','evidence','negated','population','conditions'):
            assert incoming.to_dict().get(field)==saved.get(field)
    assert reverse_event(after,event)==before
    cid=before['id']
    owned=[(i,dict(source_id=cid,target_id=md[s+'_id'],relation_type='about',
        metadata=dict(claim_id=cid,anchor_role=s,unknown=[0,False,None]))) for i,s in enumerate(('subject','object'),1)]
    owned.append((3,dict(source_id=md['subject_id'],target_id=md['object_id'],relation_type=old_predicate,
        weight=0.0,metadata=dict(claim_id=cid,original_predicate=old_predicate,unknown='keep'))))
    changes={e['ordinal']:e for e in revise_owned_edges(before,after,owned)}
    revised=[(i,apply_event(edge,changes[i]) if i in changes else deepcopy(edge)) for i,edge in owned]
    assert revised[-1][1]['relation_type']=='is_associated_with'
    assert revised[-1][1]['metadata']==owned[-1][1]['metadata'] and revised[-1][1]['weight']==0.0
    assert all(reverse_event(edge,changes[i])==dict(owned)[i] for i,edge in revised if i in changes)
    assert revise_owned_edges(after,after,revised)==[]
    unreviewed=Claim.from_dict(before['metadata']);unreviewed.raw_text+=' An unreviewed extra conclusion.'
    saved=deepcopy(unreviewed.to_dict())
    assert repair.apply_multipaper_reimport(kg,unreviewed) is None and unreviewed.to_dict()==saved
