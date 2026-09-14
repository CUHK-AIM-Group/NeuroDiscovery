from copy import deepcopy
import pytest
from neurooracle.src import kg_source_scoped_sets_repair as r75
from neurooracle.src.claim_ingestion import resolve_claim_entities
from neurooracle.src.graph_manager import KnowledgeGraph
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.kg_reviewed_relation_repair import revise_claim,reimport_rule
from neurooracle.src.kg_root_application import reverse_event
from neurooracle.src.schema import Claim,ConceptNode


def fixture(monkeypatch):
    md=dict(id='CLM:measurement',subject_id='CUI:wrong_gene',subject_name='psychiatric risk genes',
        object_id='CUI:outcome',object_name='cognitive trajectory',predicate='correlates_with',negated=False,
        source_paper={'pmid':'37286292','title':'own source'},raw_text='Risk gene sets and their expression profiles have distinct scopes.',
        metadata={'subject_type':'biomarker','object_type':'OUTCOME','subject_id':'CUI:wrong_gene','unknown':[0,False,None]},
        evidence={'sample_size':0,'direction':'negative','population':'cohort A','unknown':None})
    before=dict(id=md['id'],preferred_name='psychiatric risk genes correlates_with cognitive trajectory',metadata=md,unknown='preserve')
    n=r75.literal(md['subject_name'],'gene_set','37286292');d=ConceptNode(id='CUI:outcome',preferred_name='cognitive trajectory').to_dict()
    event,after=revise_claim(before,{'subject_id':n['id']},review_id='source')
    rule=reimport_rule(before,after,{n['id']:n,d['id']:d},'source')
    kg=KnowledgeGraph()
    for v in (n,d):kg.add_concept(ConceptNode.from_dict(v))
    kg.serialization_metadata['source_scoped_sets_repair']={'version':r75.VERSION,'rules':[rule]}
    monkeypatch.setattr(r75,'APPROVED_RULESET_DIGEST',digest([rule]))
    return kg,before,after,event


def test_current_dispatch_preserves_raw_scientific_payload_and_inverts(monkeypatch):
    kg,before,after,event=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);original=deepcopy(claim.to_dict())
    assert resolve_claim_entities(kg,claim) is claim
    assert claim.subject_id==after['metadata']['subject_id']
    for key in ('raw_text','source_paper','evidence','negated'):assert claim.to_dict()[key]==original[key]
    assert claim.metadata['unknown']==[0,False,None]
    assert reverse_event(after,event)==before


@pytest.mark.parametrize('change',[('pmid','111'),('subject_name','normative expression profiles of disorder risk genes'),('raw_text','another source sentence'),('predicate','causes'),('negated',True)])
def test_unreviewed_source_assay_and_assertion_are_not_rewritten(monkeypatch,change):
    kg,before,_,_=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);key,value=change
    if key=='pmid':claim.source_paper.pmid=value
    else:setattr(claim,key,value)
    assert r75.apply_source_sets_reimport(kg,claim) is None


@pytest.mark.parametrize('case',['missing','modified','supplied','nested','policy','ambiguous'])
def test_conflicts_fail_before_any_mutation(monkeypatch,case):
    kg,before,after,_=fixture(monkeypatch);claim=Claim.from_dict(before['metadata']);nid=after['metadata']['subject_id']
    if case=='missing':kg._index.pop(nid)
    elif case=='modified':kg.get_concept(nid).domain_tags=['gene']
    elif case=='supplied':claim.subject_id='CUI:other_gene'
    elif case=='nested':claim.metadata['subject_id']='CUI:other_gene'
    elif case=='policy':kg.serialization_metadata['source_scoped_sets_repair']['rules'][0]['pmid']='1'
    else:
        rules=kg.serialization_metadata['source_scoped_sets_repair']['rules'];other=deepcopy(rules[0]);other['new_relation']['subject_id']='CUI:other';rules.append(other)
        monkeypatch.setattr(r75,'APPROVED_RULESET_DIGEST',digest(rules))
    original=deepcopy(claim.to_dict())
    with pytest.raises(ValueError):r75.apply_source_sets_reimport(kg,claim)
    assert claim.to_dict()==original


def test_source_scopes_and_measurements_cannot_collapse():
    nodes=[r75.literal(n,d,p) for n,d,p in [
        ('psychiatric risk genes','gene_set','37286292'),('normative expression profiles of disorder risk genes','gene_set','37286292'),
        ('gene expression','expression_profile','32792264'),
        ('gene expression','expression_profile','35899321'),
        ('rs1006737','genetic_variant_mention','19781653'),('heritability','genetic_construct','33057169')]]
    assert len({n['id'] for n in nodes})==len(nodes)
    assert all(not n['semantic_types'] and not n['external_ids'] for n in nodes)


@pytest.mark.parametrize('case',['name','source','domain','external','definition'])
def test_tampered_node_scope_is_rejected(case):
    n=r75.literal('RNA transcripts','transcript_set','34002691')
    if case=='name':n['preferred_name']='MRS2'
    elif case=='source':n['id']=n['id'].replace('34002691','34002692')
    elif case=='domain':n['domain_tags']=['gene']
    elif case=='external':n['external_ids']={'gene':'123'}
    else:n['definition']='generic equivalent'
    with pytest.raises(ValueError):r75.SourceSetsTransform([],[],[],[],[n],[])


def test_inherited_and_current_inputs_replay_to_one_final_identity(monkeypatch):
    kg,before,after,_=fixture(monkeypatch);rules=kg.serialization_metadata['source_scoped_sets_repair']['rules']
    legacy=deepcopy(rules[0]);legacy['old_relation']['subject_id']='CUI:legacy_gene';rules.append(legacy)
    monkeypatch.setattr(r75,'APPROVED_RULESET_DIGEST',digest(rules))
    for old in ('CUI:wrong_gene','CUI:legacy_gene',after['metadata']['subject_id']):
        claim=Claim.from_dict(before['metadata']);claim.subject_id=old;claim.metadata['subject_id']=old
        assert r75.apply_source_sets_reimport(kg,claim).subject_id==after['metadata']['subject_id']


def test_exact_source_title_variants_do_not_generalize():
    from neurooracle.scripts.review_kg_source_scoped_sets_batch import source_gate,TITLE_VARIANTS
    md={'source_paper':{'pmid':'32853481','title':TITLE_VARIANTS['32853481']}}
    doc={'pmid':'32853481','own_article_ids':{'pubmed':'32853481'},'title':'GBA Variants in Parkinson\'s Disease: Clinical, Metabolomic, and Multimodal Neuroimaging Phenotypes.', 'abstract':[{'text':'hippocampal atrophy'}]}
    assert source_gate(md,{'32853481':doc})[1] is None
    md['source_paper']['title']+=' unrelated scope'
    assert source_gate(md,{'32853481':doc})[1]=='owning_title_needs_identity_review'


def test_work_versions_share_only_reviewed_method_identity():
    node=r75.work_literal()
    assert r75.reviewed_literal(r75.WORK_NAME,'intervention','40463528')==node
    assert r75.reviewed_literal(r75.WORK_NAME,'intervention','41167554')==node
    assert r75.reviewed_literal(r75.WORK_NAME,'intervention','40835066')!=node
    assert r75.reviewed_literal('disease-specific tTIS target','intervention','41167554')!=node
    assert 'not independent primary studies' in node['definition']
    assert r75.SourceSetsTransform([],[],[],[],[node],[]).new_nodes[node['id']]==node


@pytest.mark.parametrize('change',['name','definition','external','domain'])
def test_work_scope_tamper_is_rejected(change):
    node=r75.work_literal()
    if change=='name':node['preferred_name']='tTIS equivalent treatment'
    elif change=='definition':node['definition']+=' Includes withdrawn intermediate.'
    elif change=='external':node['external_ids']={'pmid':'40835066'}
    else:node['domain_tags']=['gene']
    with pytest.raises(ValueError):r75.SourceSetsTransform([],[],[],[],[node],[])


@pytest.mark.parametrize('fault',[None,'reverse_link','withdrawal','sentence','membership','registration'])
def test_work_review_requires_reciprocal_primary_evidence_and_complete_members(tmp_path,monkeypatch,fault):
    from neurooracle.scripts import review_kg_source_scoped_sets_batch as review
    import json
    monkeypatch.setattr(review,'OUTPUT',tmp_path)
    ids=['CLM:6f9f487c3c4d','CLM:78f9d81c35dd','CLM:ae6ac9e1a05f']
    sentence='Transcranial temporal interference stimulation (tTIS) is a novel, non-invasive method developed to selectively modulate deep brain regions and associated neural circuits.'
    docs={p:dict(pmid=p,title='Systematic review',source={},own_article_ids={'pubmed':p},publication_types=['Systematic Review'],
        abstract=[{'text':sentence+' Registration CRD42024559678'}],comments_corrections=[]) for p in ['40463528','41167554','40835066']}
    docs['40463528']['comments_corrections']=[{'ref_type':'UpdateIn','pmid':'41167554'}]
    docs['41167554']['comments_corrections']=[{'ref_type':'UpdateOf','pmid':p} for p in ['40463528','40835066']]
    docs['40835066'].update(title='WITHDRAWN: Systematic review',publication_types=['Retraction Notice'],
        comments_corrections=[{'ref_type':'UpdateIn','pmid':'41167554'}])
    science={cid:dict(raw_text=sentence,negated=False,predicate='modulates',subject_name=r75.WORK_NAME,
        object_name='deep brain regions',source_paper={'pmid':'41167554' if i==0 else '40463528'}) for i,cid in enumerate(ids)}
    if fault=='reverse_link':docs['41167554']['comments_corrections']=docs['41167554']['comments_corrections'][1:]
    elif fault=='withdrawal':docs['40835066']['title']='Normal publication'
    elif fault=='sentence':science[ids[0]]['raw_text']='Validated therapeutic effect.'
    elif fault=='membership':ids.pop()
    elif fault=='registration':docs['40463528']['abstract']=[{'text':sentence}]
    path=tmp_path/'groups.jsonl';path.write_text(json.dumps({'id':'REL:45ed1797a7f27d353f75cfece1f73b23ee75d38104f1e25630df56c01a4222b0','members':[{'claim_id':cid} for cid in ids]})+'\n',encoding='utf8')
    base={'current_graph':{},'current_shared_relations':{'path':str(path)}}
    if fault:
        with pytest.raises(Exception):review.verified_work_review(base,science,docs)
        assert not (tmp_path/'VERIFIED_WORK_VERSION_REVIEW.json').exists()
    else:
        result=review.verified_work_review(base,science,docs)
        assert set(result['complete_current_member_ids'])==set(ids)
        evidence=json.loads((tmp_path/'VERIFIED_WORK_VERSION_REVIEW.json').read_text(encoding='utf8'))
        assert evidence['independent_primary_studies']==0
        assert evidence['withdrawn_intermediate_excluded']=='40835066'
