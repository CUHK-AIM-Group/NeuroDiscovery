from copy import deepcopy
from xml.sax.saxutils import escape
import pytest

from neurooracle.src import kg_research_statement_retirement as scope
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_kg_literal_endpoint_repair import claim, edges


def source_xml():
    return '<PubmedArticleSet>'+''.join(
        '<PubmedArticle><MedlineCitation><PMID>'+r['pmid']+'</PMID><Article><ArticleTitle>Reviewed title</ArticleTitle>'
        '<Abstract><AbstractText>'+escape(r['fragment'])+'</AbstractText></Abstract></Article></MedlineCitation>'
        '<PubmedData><ArticleIdList><ArticleId IdType="doi">'+r['doi']+'</ArticleId></ArticleIdList></PubmedData></PubmedArticle>'
        for r in scope.REVIEWS.values())+'</PubmedArticleSet>'


def research_claim(monkeypatch, cid):
    r=scope.REVIEWS[cid]; row=claim(r['subject'], None); row['id']=cid
    md=row['metadata'];md.update(id=cid,subject_id='SUBJECT:'+cid,object_id='OBJECT:'+cid,
        subject_name=r['subject'],object_name=r['object'],negated=False,raw_text=r['raw'],
        source_paper=dict(pmid=r['pmid'],doi=r['doi']),metadata=dict(original_predicate=r['original']),
        evidence=dict(legacy_value=r['raw'],p_value=None,effect_size=None,sample_size=None))
    monkeypatch.setitem(scope.REVIEWS,cid,dict(r,sha=digest(row)))
    es=edges(row)
    for _,edge in es:
        if 'negated' in edge['metadata']:edge['metadata']['negated']=False
    return row,es,scope.public_source_proof(source_xml())


@pytest.mark.parametrize('cid', sorted(scope.REVIEWS))
def test_only_exact_reviewed_research_intentions_are_retired(monkeypatch,cid):
    row,owned,proof=research_claim(monkeypatch,cid);before=deepcopy(row)
    result=scope.reviewed_claim(row,owned,proof)
    assert row==before and result['claim_id']==cid
    assert result['no_null_or_replacement_claim_synthesized']
    assert result['source_document_not_deleted'] and not result['record_preimages_saved']
    assert len(result['owned_edges'])==3 and 'before_node' not in result


@pytest.mark.parametrize('field,value',[
    ('predicate','correlates_with'),('negated',True),('raw_text','A new trial showed an effect.'),
    ('subject_name','different treatment'),('object_name','different outcome'),
])
def test_changed_scientific_statement_is_not_selected(monkeypatch,field,value):
    cid=next(iter(scope.REVIEWS));row,owned,proof=research_claim(monkeypatch,cid)
    row['metadata'][field]=value
    with pytest.raises(ValueError):scope.reviewed_claim(row,owned,proof)


def test_research_intention_is_required_even_when_hash_is_rebound_for_unit_test(monkeypatch):
    cid=next(iter(scope.REVIEWS));row,owned,proof=research_claim(monkeypatch,cid)
    row['metadata']['metadata']['original_predicate']='is_associated_with'
    monkeypatch.setitem(scope.REVIEWS,cid,dict(scope.REVIEWS[cid],sha=digest(row)))
    with pytest.raises(ValueError,match='original research-intention'):scope.reviewed_claim(row,owned,proof)


@pytest.mark.parametrize('mutation',['missing','duplicate_about','wrong_owner','wrong_science'])
def test_incomplete_or_mixed_edge_closure_is_not_removed(monkeypatch,mutation):
    cid=next(iter(scope.REVIEWS));row,owned,proof=research_claim(monkeypatch,cid)
    if mutation=='missing':owned=owned[:-1]
    elif mutation=='duplicate_about':owned[1]=(2,deepcopy(owned[0][1]))
    elif mutation=='wrong_owner':owned[2][1]['metadata']['claim_id']='CLM:other'
    else:owned[2][1]['relation_type']='causes'
    with pytest.raises(ValueError):scope.reviewed_claim(row,owned,proof)


def test_other_hypothesis_or_review_is_not_a_deletion_target(monkeypatch):
    cid=next(iter(scope.REVIEWS));row,owned,proof=research_claim(monkeypatch,cid);row['id']='CLM:other'
    with pytest.raises(ValueError,match='unreviewed'):scope.reviewed_claim(row,owned,proof)


def test_wrong_public_doi_is_not_accepted():
    xml=source_xml().replace('10.4088/jcp.v64n0402','10.4088/other')
    with pytest.raises(ValueError,match='PMID/DOI'):scope.public_source_proof(xml)


def test_title_or_genre_without_reviewed_abstract_context_is_insufficient():
    r=next(iter(scope.REVIEWS.values()));xml=source_xml().replace(escape(r['fragment']),'Review article.')
    with pytest.raises(ValueError,match='complete-source context'):scope.public_source_proof(xml)


def test_numeric_observation_cannot_be_silently_deleted(monkeypatch):
    cid=next(iter(scope.REVIEWS));row,owned,proof=research_claim(monkeypatch,cid)
    row['metadata']['evidence']['p_value']=0.01
    monkeypatch.setitem(scope.REVIEWS,cid,dict(scope.REVIEWS[cid],sha=digest(row)))
    with pytest.raises(ValueError,match='saved evidence'):scope.reviewed_claim(row,owned,proof)
