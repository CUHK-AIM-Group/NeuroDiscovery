"""Synthetic unit records, not saved KG preimages; no network/model calls."""
from copy import deepcopy
import pytest
from neurooracle.src import kg_scoped_structure as r
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim


def fixture(cid=r.COLLISION):
    if cid==r.COLLISION:
        names=(r.ACTIVATION,r.INTERACTION);types=('IMAGING_MARKER','INDIVIDUAL_DATA');pmid='22306803'
    elif cid==r.DIRECTION:
        names=('BACE1 amyloid-processing pathway','bilateral hippocampal volume');types=('PATHWAY','IMAGING_MARKER');pmid='36847009'
    else:
        pmid,name,_=r.MRI_SPECS[cid];names=(name,'synthetic outcome');types=('IMAGING_MARKER','OUTCOME')
    md=dict(id=cid,subject_id='CUI:C1421437',object_id=('CUI:C1421437' if cid==r.COLLISION else 'CLM_CONCEPT:synthetic_outcome'),
        subject_name=names[0],object_name=names[1],predicate='is_associated_with',negated=False,confidence=.8,
        source_paper=dict(pmid=pmid,title='synthetic test only'),raw_text='Synthetic unit-test statement, not a KG record copy.',
        evidence=dict(sample_size=104,direction='original synthetic direction',p_value=None),
        metadata=dict(subject_type=types[0],object_type=types[1],conditions=['left/right and subgroup retained']))
    if cid==r.DIRECTION:
        md['raw_text']='Plasma BACE1 concentration was negatively associated with hippocampal volume in MCI due to AD.'
        md['evidence']['direction']='positive'
    row=dict(id=cid,preferred_name='synthetic claim',metadata=md)
    sides=('subject','object') if cid==r.COLLISION else (() if cid==r.DIRECTION else ('subject',))
    changes=[dict(side=s,name=md[s+'_name'],old_id=md[s+'_id'],target_id=r.literal_node(md[s+'_name'])['id']) for s in sides]
    proof={pmid:dict(synthetic=True)}
    return row,changes,proof


def about(row,target=None):
    return dict(source_id=row['id'],target_id=target or row['metadata']['subject_id'],relation_type='about',
        source='claim_extraction',confidence=row['metadata']['confidence'],evidence_ref='',metadata={})


@pytest.mark.parametrize('cid',[r.COLLISION,r.DIRECTION,*r.MRI_SPECS])
def test_repair_exact_roundtrip_and_no_other_changes(cid):
    row,ch,p=fixture(cid);original=deepcopy(row)
    event,out=r.reviewed_claim(row,ch,p)
    assert row==original and r.reverse_claim(out,event)==row
    if cid!=r.DIRECTION:assert nonidentity_claim(row)==nonidentity_claim(out)
    else:
        expected=deepcopy(row);expected['metadata']['evidence']['direction']='negative'
        assert out==expected and out['metadata']['negated'] is False


def test_double_collision_closes_both_about_without_scientific_edge():
    row,ch,p=fixture();event,out=r.reviewed_claim(row,ch,p);old=about(row)
    events=r.reviewed_edges(row['id'],row,out,[(17,old)])
    added=r.reviewed_added_about(row['id'],row,out,[(17,old)])
    assert len(events)==len(added)==1
    current=r.apply_edge(old,events[0]);assert r.apply_edge(current,events[0],reverse=True)==old
    assert {current['target_id'],added[0]['row']['target_id']}=={out['metadata']['subject_id'],out['metadata']['object_id']}
    assert added[0]['row']['relation_type']=='about' and added[0]['not_a_new_scientific_assertion']


@pytest.mark.parametrize('mutator',[
    lambda row,ch:ch.pop(),
    lambda row,ch:ch.append(deepcopy(ch[0])),
    lambda row,ch:ch[0].update(target_id='CUI:C1421437'),
    lambda row,ch:ch[0].update(name='prefrontal activation'),
    lambda row,ch:row['metadata'].update(negated=True),
    lambda row,ch:row['metadata']['source_paper'].update(pmid='99999'),
    lambda row,ch:row['metadata']['metadata'].update(subject_id='CUI:wrong'),
    lambda row,ch:row['metadata'].update(subject_type='GENE_TARGET'),
    lambda row,ch:row['metadata']['metadata'].update(object_type='GENE_TARGET'),
    lambda row,ch:row['metadata']['evidence'].update(sample_size=100),
])
def test_collision_scope_drift_rejected(mutator):
    row,ch,p=fixture();mutator(row,ch)
    with pytest.raises(ValueError):r.reviewed_claim(row,ch,p)


@pytest.mark.parametrize('mutation',[
    dict(metadata={'role':'object'}),dict(metadata={'claim_id':'CLM:other'}),
    dict(source_id='CLM:other'),dict(target_id='CUI:other'),dict(evidence_ref='unreviewed evidence'),
    dict(confidence=.9),dict(relation_type='causes'),dict(source='another import'),dict(id='unique-edge-id'),
])
def test_no_clone_of_non_neutral_or_foreign_payload(mutation):
    row,ch,p=fixture();_,out=r.reviewed_claim(row,ch,p);edge=about(row);edge.update(mutation)
    with pytest.raises(ValueError):r.reviewed_added_about(row['id'],row,out,[(17,edge)])


@pytest.mark.parametrize('count',[0,2,3])
def test_unreviewed_about_multiplicity_rejected(count):
    row,ch,p=fixture();_,out=r.reviewed_claim(row,ch,p)
    with pytest.raises(ValueError):r.reviewed_edges(row['id'],row,out,[(i,about(row)) for i in range(count)])


@pytest.mark.parametrize('field',['raw_text','predicate','negated'])
def test_direction_other_fields_drift_rejected(field):
    row,ch,p=fixture(r.DIRECTION);row['metadata'][field]='different'
    with pytest.raises(ValueError):r.reviewed_claim(row,ch,p)


@pytest.mark.parametrize('cid',list(r.MRI_SPECS))
def test_explicit_mri_not_a_gene_or_unspecified_entity(cid):
    row,ch,p=fixture(cid);row['metadata']['metadata']['subject_type']='ENTITY'
    with pytest.raises(ValueError):r.reviewed_claim(row,ch,p)


def test_qmri_symptom_scope_not_approved():
    with pytest.raises(ValueError):r.literal_node('lateral prefrontal cortex qMRI myelin content')


def test_whole_literal_shared_without_claim_or_paper_salt():
    one=r.literal_node(r.ACTIVATION);two=r.literal_node(r.ACTIVATION)
    assert one==two and not one['metadata']
    interaction=r.literal_node(r.INTERACTION)
    assert interaction['domain_tags']==['dataset_variable'] and interaction['id']!=one['id']
    assert not interaction['semantic_types'] and not interaction['external_ids']


def test_new_direction_cannot_be_flipped_twice():
    row,ch,p=fixture(r.DIRECTION);event,out=r.reviewed_claim(row,ch,p)
    with pytest.raises(ValueError):r.reviewed_claim(out,ch,p)
    out['metadata']['confidence']=1
    with pytest.raises(ValueError):r.reverse_claim(out,event)


def test_sources_require_owned_article_not_reference_pmid():
    with pytest.raises(ValueError,match='missing own source'):
        r.source_proof([b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>999</PMID><Article><ArticleTitle>Test</ArticleTitle></Article></MedlineCitation><PubmedData><ReferenceList><Reference><ArticleIdList><ArticleId IdType="pubmed">22306803</ArticleId></ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle></PubmedArticleSet>'])


def test_duplicate_source_owners_rejected():
    xml=b'<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>22306803</PMID><Article><ArticleTitle>Test</ArticleTitle></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'
    with pytest.raises(ValueError,match='duplicate'):r.source_documents([xml,xml])
