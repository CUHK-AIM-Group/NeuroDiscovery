from copy import deepcopy
import pytest
from neurooracle.src import kg_semantic_measurement_repair as r74
from neurooracle.src.claim_ingestion import resolve_claim_entities
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_reviewed_relation_repair import revise_claim,reimport_rule
from neurooracle.src.kg_root_application import reverse_event
from neurooracle.src.schema import Claim,ConceptNode


def fixture(monkeypatch):
    md=dict(id='CLM:measurement',subject_id='CUI:wrong_gene',subject_name='DNA methylation GDF15',
        object_id='CUI:outcome',object_name='cognitive trajectory',predicate='correlates_with',negated=False,
        source_paper={'pmid':'41274863','title':'own source'},raw_text='DNAm and plasma measures have different associations.',
        metadata={'subject_type':'biomarker','object_type':'OUTCOME','subject_id':'CUI:wrong_gene','unknown':[0,False,None]},
        evidence={'sample_size':0,'direction':'negative','population':'cohort A','unknown':None})
    before=dict(id=md['id'],preferred_name='DNA methylation GDF15 correlates_with cognitive trajectory',metadata=md,unknown='preserve')
    n=r74.literal(md['subject_name'],'biochemical_measure','41274863');d=ConceptNode(id='CUI:outcome',preferred_name='cognitive trajectory').to_dict()
    event,after=revise_claim(before,{'subject_id':n['id']},review_id='source')
    rule=reimport_rule(before,after,{n['id']:n,d['id']:d},'source')
    kg=KnowledgeGraph()
    for v in (n,d):kg.add_concept(ConceptNode.from_dict(v))
    kg.serialization_metadata['semantic_measurement_repair']={'version':r74.VERSION,'rules':[rule]}
    monkeypatch.setattr(r74,'APPROVED_RULESET_DIGEST',digest([rule]))
    return kg,before,after,event


def test_current_dispatch_preserves_raw_scientific_payload_and_inverts(monkeypatch):
    kg,before,after,event=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);original=deepcopy(claim.to_dict())
    assert resolve_claim_entities(kg,claim) is claim
    assert claim.subject_id==after['metadata']['subject_id']
    for key in ('raw_text','source_paper','evidence','negated'):assert claim.to_dict()[key]==original[key]
    assert claim.metadata['unknown']==[0,False,None]
    assert reverse_event(after,event)==before


@pytest.mark.parametrize('change',[('pmid','111'),('subject_name','plasma GDF15'),('raw_text','another source sentence'),('predicate','causes'),('negated',True)])
def test_unreviewed_source_assay_and_assertion_are_not_rewritten(monkeypatch,change):
    kg,before,_,_=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);key,value=change
    if key=='pmid':claim.source_paper.pmid=value
    else:setattr(claim,key,value)
    assert r74.apply_semantic_reimport(kg,claim) is None


@pytest.mark.parametrize('case',['missing','modified','supplied','nested','policy','ambiguous'])
def test_conflicts_fail_before_any_mutation(monkeypatch,case):
    kg,before,after,_=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);nid=after['metadata']['subject_id']
    if case=='missing':kg._index.pop(nid)
    elif case=='modified':kg.get_concept(nid).domain_tags=['gene']
    elif case=='supplied':claim.subject_id='CUI:other_gene'
    elif case=='nested':claim.metadata['subject_id']='CUI:other_gene'
    elif case=='policy':kg.serialization_metadata['semantic_measurement_repair']['rules'][0]['pmid']='1'
    else:
        rules=kg.serialization_metadata['semantic_measurement_repair']['rules'];other=deepcopy(rules[0]);other['new_relation']['subject_id']='CUI:other';rules.append(other)
        monkeypatch.setattr(r74,'APPROVED_RULESET_DIGEST',digest(rules))
    original=deepcopy(claim.to_dict())
    with pytest.raises(ValueError):r74.apply_semantic_reimport(kg,claim)
    assert claim.to_dict()==original


def test_source_scopes_and_measurements_cannot_collapse():
    nodes=[r74.literal(n,d,p) for n,d,p in [
        ('DNA methylation GDF15','biochemical_measure','41274863'),('plasma GDF15','biochemical_measure','41274863'),
        ('obsessive-compulsive symptom severity','clinical_measure','15691530'),
        ('obsessive-compulsive symptom severity','clinical_measure','15978549'),
        ('creatine supplementation','intervention','41280460'),('creatine','molecular_mention','41280460')]]
    assert len({n['id'] for n in nodes})==len(nodes)
    assert all(not n['semantic_types'] and not n['external_ids'] for n in nodes)


@pytest.mark.parametrize('case',['name','source','domain','external','definition'])
def test_tampered_node_scope_is_rejected(case):
    n=r74.literal('magnesium','molecular_mention','40647320')
    if case=='name':n['preferred_name']='MRS2'
    elif case=='source':n['id']=n['id'].replace('40647320','40647321')
    elif case=='domain':n['domain_tags']=['gene']
    elif case=='external':n['external_ids']={'gene':'123'}
    else:n['definition']='generic equivalent'
    with pytest.raises(ValueError):r74.SemanticTransform([],[],[],[],[n],[])


def test_inherited_and_current_inputs_replay_to_one_final_identity(monkeypatch):
    kg,before,after,_=fixture(monkeypatch);rules=kg.serialization_metadata['semantic_measurement_repair']['rules']
    legacy=deepcopy(rules[0]);legacy['old_relation']['subject_id']='CUI:legacy_gene';rules.append(legacy)
    monkeypatch.setattr(r74,'APPROVED_RULESET_DIGEST',digest(rules))
    for old in ('CUI:wrong_gene','CUI:legacy_gene',after['metadata']['subject_id']):
        claim=Claim.from_dict(before['metadata']);claim.subject_id=old;claim.metadata['subject_id']=old
        assert r74.apply_semantic_reimport(kg,claim).subject_id==after['metadata']['subject_id']


def test_exact_source_title_variants_do_not_generalize():
    from neurooracle.scripts.review_kg_semantic_measurement_batch import source_gate,TITLE_VARIANTS
    md={'source_paper':{'pmid':'29592889','title':TITLE_VARIANTS['29592889']}}
    doc={'pmid':'29592889','own_article_ids':{'pubmed':'29592889'},'title':'Dissociable influences of APOE ε4 and polygenic risk of AD dementia on amyloid and cognition.', 'abstract':[{'text':'hippocampal atrophy'}]}
    assert source_gate(md,{'29592889':doc})[1] is None
    md['source_paper']['title']+=' unrelated scope'
    assert source_gate(md,{'29592889':doc})[1]=='owning_title_needs_identity_review'
