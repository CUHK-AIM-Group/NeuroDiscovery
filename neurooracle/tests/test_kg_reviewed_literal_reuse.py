from copy import deepcopy
import pytest

from neurooracle.src import kg_reviewed_literal_reuse as repair
from neurooracle.src.kg_identity_pilot import digest,nonidentity_claim
from neurooracle.tests.test_kg_literal_endpoint_repair import claim,edges


def fixture(cid=repair.CHORIO):
    side,name,target=repair.SPECS[cid];row=claim(name);row['id']=cid;md=row['metadata'];md['id']=cid;md['confidence']=0.7
    old=repair.REGIONAL if cid in repair.BROAD_REASSIGN else ('CUI:C1414531' if 'surface' in name or 'interface' in name else 'CUI:C1421437')
    md[side+'_id']=old;md[side+'_name']=name;md['metadata'][side+'_id']=old
    if cid in repair.BRANCH_CLAIMS:md['object_id']=repair.COGNITIVE;md['metadata']['object_id']=repair.COGNITIVE
    if cid==repair.MOUSE:
        md['source_paper']['pmid']='15885483'
        md['raw_text']='MC3T3-E1 cells showed high VCP expression mainly in cytoplasm and mild physiological stress did not change VCP levels or distribution.'
    witnesses={g:dict(node_id=g,semantic_types=['T028'],labels=labels,node_sha256='a'*64) for g,labels in
        [('CUI:C1414531',['FANCE','FACE']),('CUI:C1421437',['VCP','TERA'])]}
    reviews={cid:dict(claim_id=cid,claim_sha256=digest(row))}
    changes=[dict(side=side,name=name,old_id=old,target_id=target)]
    refs=edges(row)
    for _,r in refs:
        if r['relation_type']=='about':r.update(source_id=cid,source='claim_extraction',confidence=0.7,evidence_ref='',metadata={})
        else:r['metadata']['claim_id']=cid
    if cid in repair.BRANCH_CLAIMS:
        extra=deepcopy(next(r for _,r in refs if r['relation_type']=='about' and r['target_id']==repair.COGNITIVE))
        extra['target_id']=repair.OLD_COGNITIVE;refs.append((4,extra))
    return row,changes,witnesses,reviews,refs


@pytest.mark.parametrize('cid',list(repair.SPECS))
def test_every_finite_identity_preserves_science_and_original_names(cid):
    row,changes,witnesses,reviews,refs=fixture(cid);before=deepcopy(row)
    event,current=repair.reviewed_claim(row,changes,witnesses,reviews,dict(source_is_protein_not_human_gene=True))
    assert row==before and nonidentity_claim(current)==nonidentity_claim(row)
    assert repair.reverse_claim(current,event)==row


@pytest.mark.parametrize('cid',list(repair.BRANCH_CLAIMS))
def test_only_the_duplicate_neutral_about_is_retired(cid):
    row,changes,witnesses,reviews,refs=fixture(cid)
    event,current=repair.reviewed_claim(row,changes,witnesses,reviews,{})
    retained,removed=repair.split_owned(cid,row,refs)
    assert len(retained)==3 and len(removed)==1
    changed={r['ordinal']:r for r in repair.reviewed_edges(cid,row,current,refs)}
    current_refs=[(o,repair.apply_edge(r,changed[o]) if o in changed else r) for o,r in retained]
    assert repair.candidate_retirement_proof(cid,current,current_refs,removed[0])
    assert all(r['target_id']!=repair.OLD_COGNITIVE for _,r in current_refs)


@pytest.mark.parametrize('mutation',['metadata','confidence','wrong_current','extra'])
def test_nonidentical_old_branch_is_not_deleted(mutation):
    row,changes,witnesses,reviews,refs=fixture()
    if mutation=='metadata':refs[-1][1]['metadata']['extra']='meaningful'
    elif mutation=='confidence':refs[-1][1]['confidence']=0.1
    elif mutation=='wrong_current':row['metadata']['object_id']='different'
    else:refs.append((5,deepcopy(refs[-1][1])))
    with pytest.raises(ValueError):repair.split_owned(row['id'],row,refs)


def test_one_case_variant_exception_does_not_change_stored_claim_case():
    cid='CLM:CASE1MAN:22706988:6561';row,changes,witnesses,reviews,refs=fixture(cid)
    event,current=repair.reviewed_claim(row,changes,witnesses,reviews,{})
    assert current['metadata']['object_name'].startswith('chronic ')
    changes[0]['name']='Chronic mild traumatic brain injury in Iraq and Afghanistan veterans'
    with pytest.raises(ValueError,match='target differs'):repair.reviewed_claim(row,changes,witnesses,reviews,{})


def test_mouse_protein_requires_species_proof_and_is_not_a_T028_gene():
    row,changes,witnesses,reviews,refs=fixture(repair.MOUSE)
    with pytest.raises(ValueError,match='unproved'):repair.reviewed_claim(row,changes,witnesses,reviews,{})
    node=repair.literal_node(repair.MOUSE_NAME)
    assert node['semantic_types']==['T116'] and node['external_ids']['NCBI_Taxonomy']=='10090'
    assert node['metadata']=={} and node['aliases']==[]


def test_unknown_claim_or_target_is_not_generalized():
    row,changes,witnesses,reviews,refs=fixture();changes[0]['target_id']='CLM_CONCEPT:other'
    with pytest.raises(ValueError,match='target differs'):repair.reviewed_claim(row,changes,witnesses,reviews,{})
    row['id']='CLM:unreviewed'
    with pytest.raises(ValueError,match='exact reviewed'):repair.reviewed_claim(row,changes,witnesses,reviews,{})


def test_public_protein_identity_not_alias_text_alone():
    xml='<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>15885483</PMID><Article><ArticleTitle>VCP</ArticleTitle><Abstract><AbstractText>MC3T3-E1 mouse osteoblast-like cells: VCP protein expression in untransformed osteoblastic cells.</AbstractText></Abstract></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1016/j.orthres.2004.12.012</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'
    protein=dict(primaryAccession='Q01853',uniProtkbId='TERA_MOUSE',organism=dict(taxonId=10090,scientificName='Mus musculus'),
        genes=[dict(geneName=dict(value='Vcp'))],proteinDescription=dict(alternativeNames=[dict(fullName=dict(value='Valosin-containing protein'),shortNames=[dict(value='VCP')])]))
    assert repair.public_scope_proof(xml,protein)['source_is_protein_not_human_gene']
    protein['organism']['taxonId']=9606
    with pytest.raises(ValueError,match='mouse protein'):repair.public_scope_proof(xml,protein)
