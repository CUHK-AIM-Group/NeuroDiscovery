from copy import deepcopy
from xml.etree import ElementTree as ET
import pytest
from neurooracle.src import kg_multipaper_claim_repair as repair
from neurooracle.src.claim_ingestion import resolve_claim_entities
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_reviewed_relation_repair import revise_claim,reimport_rule
from neurooracle.src.kg_root_application import reverse_event
from neurooracle.src.schema import Claim,ConceptNode
from neurooracle.scripts.extend_kg_multipaper_papers import own_record,reconstruct

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
    kg.serialization_metadata['multipaper_claim_repair']=dict(version=repair.VERSION,rules=rules)
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
    nid=after[0]['metadata']['subject_id'];rules=kg.serialization_metadata['multipaper_claim_repair']['rules']
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
    kg,before,after,_=fixture(monkeypatch);rules=kg.serialization_metadata['multipaper_claim_repair']['rules']
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

def article():
    return ET.fromstring('<PubmedArticle><MedlineCitation><PMID>111</PMID><Article><ArticleTitle>A complete <i>tau</i> title.</ArticleTitle><Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal><ArticleDate><Year>2019</Year></ArticleDate><PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList></Article></MedlineCitation><PubmedData><History><PubMedPubDate PubStatus="pubmed"><Year>2025</Year></PubMedPubDate></History><ArticleIdList><ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1234/OWN</ArticleId></ArticleIdList><ReferenceList><Reference><ArticleIdList><ArticleId IdType="pubmed">999</ArticleId><ArticleId IdType="doi">10.1234/reference</ArticleId></ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle>')

def test_authority_uses_own_complete_title_ids_and_publication_dates_only():
    result=own_record(article(),dict(path='retained.xml',sha256='a'*64))
    assert result['doi']==['10.1234/own'] and result['pmid']=='111'
    assert result['years']==['2019','2020']
    assert result['title']=='a complete tau title'
    assert '<i>tau</i>' in result['complete_title_evidence']['title_xml']

@pytest.mark.parametrize('damage',['own_pmid','retraction','missing_title'])
def test_authority_rejects_mismatched_or_unreviewed_source(damage):
    doc=article()
    if damage=='own_pmid':doc.find('./PubmedData/ArticleIdList/ArticleId').text='999'
    elif damage=='retraction':doc.find('./MedlineCitation/Article/PublicationTypeList/PublicationType').text='Retracted Publication'
    else:doc.find('./MedlineCitation/Article').remove(doc.find('./MedlineCitation/Article/ArticleTitle'))
    with pytest.raises((ValueError,RuntimeError,AssertionError)):own_record(doc,dict(path='retained.xml',sha256='a'*64))

def test_extension_cannot_overwrite_preexisting_authority():
    with pytest.raises((ValueError,RuntimeError,AssertionError)):
        reconstruct({'records':{'111':{'existing':'preserved'}}},['111'],[])

