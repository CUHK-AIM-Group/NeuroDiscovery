from copy import deepcopy
from xml.sax.saxutils import escape
import pytest
from neurooracle.src import kg_scientific_definition_repair as repair
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_literal_endpoint_repair import edges


def public_fixture():
    articles=[];fulltexts={}
    for pmid,doi in repair.DOIS.items():
        text='Plasma BACE1 concentrations were measured. BACE1 concentration was negatively associated with hippocampal volume in the MCI group.' if pmid=='36847009' else 'Original abstract.'
        articles.append(f'<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>Own title</ArticleTitle><Abstract><AbstractText>{text}</AbstractText></Abstract></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">{doi}</ArticleId></ArticleIdList></PubmedData></PubmedArticle>')
        if pmid in repair.FULL_FRAGMENTS:
            text='; '.join(repair.FULL_FRAGMENTS[pmid])
            fulltexts[pmid]=f'<article><front><article-meta><article-id pub-id-type="pmid">{pmid}</article-id><article-id pub-id-type="doi">{doi}</article-id></article-meta></front><body><sec><p>{escape(text)}</p></sec></body></article>'
    return '<PubmedArticleSet>'+''.join(articles)+'</PubmedArticleSet>',fulltexts


def fixture(cid=repair.BACE):
    pmid,predicate,sides=repair.SPECS[cid]
    md=dict(id=cid,subject_id='CUI:subject',subject_name='subject',object_id='CUI:object',object_name='object',predicate=predicate,negated=False,
        raw_text='Original scientific quotation.',evidence=dict(direction='',p_value=0.123,sample_size=31,effect_size=None),
        source_paper=dict(pmid=pmid,doi='' if cid==repair.BACE else repair.DOIS[pmid]),
        metadata={},scope_reaudit=dict(decision='finalized',decision_basis='Historical meaning, not revalidated.'),conditions={'age':'adults'})
    for side,name,nid,new in sides:md[side+'_id']=nid;md[side+'_name']=name
    if cid==repair.BACE:md['metadata']['subject_type']='PATHWAY';md['evidence']['direction']='negative'
    row=dict(id=cid,preferred_name=f"{md['subject_name']} {predicate} {md['object_name']}",metadata=md)
    reviews={cid:dict(claim_id=cid,claim_sha256=digest(row))};proof=repair.source_proof(*public_fixture())
    return row,reviews,proof


@pytest.mark.parametrize('cid',sorted(repair.SPECS))
def test_finite_scientific_changes_are_fully_invertible_and_keep_original_evidence(cid):
    row,reviews,proof=fixture(cid);before=deepcopy(row)
    event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews)
    assert row==before and repair.reverse_claim(current,event)==before and repair.change_claim(row,event)==current
    for key in ('raw_text','source_paper','scope_reaudit','conditions'):assert current['metadata'][key]==before['metadata'][key]
    assert {k:v for k,v in current['metadata']['evidence'].items() if k!='direction'}=={k:v for k,v in before['metadata']['evidence'].items() if k!='direction'}


def test_average_sum_and_separate_have_distinct_ids():
    assert len({repair.TARGETS[n] for n in (repair.MEAN,repair.SUM,repair.SEPARATE)})==3
    assert repair.changes_for(repair.ENIGMA)[0]['new_name']==repair.MEAN
    assert len(repair.changes_for(repair.ENIGMA))==1
    assert repair.changes_for(repair.CHILD)[0]['new_name']==repair.SUM


def test_duplicate_expressions_share_identity_but_not_audit_or_claim_id():
    results=[]
    for cid in (repair.DPD_A,repair.DPD_B):
        row,reviews,proof=fixture(cid);event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews);results.append(current)
    assert results[0]['id']!=results[1]['id']
    for key in ('subject_id','object_id','subject_name','object_name','predicate','evidence'):assert results[0]['metadata'][key]==results[1]['metadata'][key]
    assert results[0]['metadata']['evidence']['direction']=='negative'


@pytest.mark.parametrize('kind',['source','raw','audit','statistic','predicate','target','extra_field','new_node'])
def test_unreviewed_changes_rejected(kind):
    row,reviews,proof=fixture();event,current=repair.reviewed_claim(row,repair.changes_for(row['id']),proof,reviews)
    if kind=='source':row['metadata']['source_paper']['pmid']='another';action=lambda:repair.change_claim(row,event)
    elif kind=='target':event['changes'][0]['target_id']='unreviewed';action=lambda:repair.change_claim(row,event)
    elif kind=='extra_field':event['field_changes'].append(dict(path=['metadata','raw_text'],old=row['metadata']['raw_text'],new='replacement'));action=lambda:repair.change_claim(row,event)
    elif kind=='new_node':action=lambda:repair.literal_node('bilateral hippocampal volume')
    else:
        if kind=='raw':current['metadata']['raw_text']='Replacement evidence'
        elif kind=='audit':current['metadata']['scope_reaudit']['decision_basis']='new scientific approval'
        elif kind=='statistic':current['metadata']['evidence']['p_value']=0.01
        else:current['metadata']['predicate']='causes'
        action=lambda:repair.reverse_claim(current,event)
    with pytest.raises(ValueError):action()


@pytest.mark.parametrize('kind',['pmid','doi','definition','incomplete','concentration'])
def test_own_source_identity_and_definitions_required(kind):
    abstract,full=public_fixture()
    if kind=='pmid':full['37721751']=full['37721751'].replace('37721751','37721752')
    elif kind=='doi':full['37721751']=full['37721751'].replace(repair.DOIS['37721751'],'10.1/wrong')
    elif kind=='definition':full['20735996']=full['20735996'].replace('sum','mean')
    elif kind=='incomplete':del full['35145436']
    else:abstract=abstract.replace('concentrations were measured','pathways were described')
    with pytest.raises(ValueError):repair.source_proof(abstract,full)


@pytest.mark.parametrize('cid',[repair.ACC,repair.DPD_A,repair.BACE,repair.ENIGMA])
def test_scientific_and_about_edges_both_remain_consistent(cid):
    row,reviews,proof=fixture(cid);event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews);owned=edges(row)
    owned[-1][1]['metadata']['negated']=False
    evs=repair.reviewed_edges(cid,row,current,owned)
    for ev in evs:
        original=dict(owned)[ev['ordinal']];patched=repair.apply_edge(original,ev)
        assert repair.apply_edge(patched,ev,reverse=True)==original
    if cid==repair.ACC:assert evs[0]['changes']['relation_type']==dict(old='increases',new='is_associated_with')
    with pytest.raises(ValueError):repair.reviewed_edges(cid,row,current,owned+[(4,deepcopy(owned[0][1]))])


def test_scientific_edge_negation_is_not_silently_preserved_as_conflict():
    row,reviews,proof=fixture();event,current=repair.reviewed_claim(row,repair.changes_for(row['id']),proof,reviews)
    with pytest.raises(ValueError,match='negation disagrees'):repair.reviewed_edges(row['id'],row,current,edges(row))


@pytest.mark.parametrize('kind',['unchanged_cldn','wrong_predicate','extra_field','other_claim','causal_predicate'])
def test_only_reviewed_cldn_unchanged_original_predicate_metadata_is_allowed(kind):
    cid=repair.BACE if kind=='other_claim' else repair.CLDN
    row,reviews,proof=fixture(cid);event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews)
    owned=edges(row);owned[-1][1]['metadata'].update(negated=False,original_predicate='correlates_with')
    if kind=='wrong_predicate':owned[-1][1]['metadata']['original_predicate']='is_associated_with'
    elif kind=='extra_field':owned[-1][1]['metadata']['unreviewed']='value'
    elif kind=='causal_predicate':owned[-1][1]['metadata']['original_predicate']='causes'
    if kind=='unchanged_cldn':
        events=repair.reviewed_edges(cid,row,current,owned)
        for e in events:
            assert repair.apply_edge(dict(owned)[e['ordinal']],e)['metadata']==dict(owned)[e['ordinal']]['metadata']
    else:
        with pytest.raises(ValueError,match='unreviewed scientific edge payload'):repair.reviewed_edges(cid,row,current,owned)


def test_scope_register_keeps_residuals_and_does_not_guess_sample_size():
    events=[]
    for cid in repair.SPECS:
        row,reviews,proof=fixture(cid);event,current=repair.reviewed_claim(row,repair.changes_for(cid),proof,reviews);events.append(event)
    ev={e['claim_id']:e for e in events};prenatal='CLM:CASE1MAN:36716140:8023'
    scope=dict(current_claim_hashes={repair.BACE:ev[repair.BACE]['claim_sha256'],repair.ACC:ev[repair.ACC]['claim_sha256'],prenatal:'unmodified'},
        findings=[dict(claim_id=cid) for cid in (repair.BACE,repair.ACC,prenatal)],resolved={})
    samples={cid:dict(claim_sha256=cid,science=dict(evidence=dict(sample_size=170)),source_paper=dict(pmid='30384145')) for cid in ('male','female')}
    after=repair.update_scope_findings(scope,dict(events=events,unmodified_sample_size_reviews=samples))
    assert len(after['current_claim_hashes'])==6 and len(after['findings'])==6 and after['old_audits_not_revalidated']
    assert scope['current_claim_hashes'][repair.BACE]==ev[repair.BACE]['claim_sha256']
    assert after['current_claim_hashes'][repair.ENIGMA]==ev[repair.ENIGMA]['current_node_sha256']
    assert after['current_claim_hashes'][prenatal]=='unmodified' and samples['male']['science']['evidence']['sample_size']==170
