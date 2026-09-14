from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_kg_scientific_definition_scope import candidate_names,field_hashes,differences
from fetch_kg_scientific_definition_sources import own_pmcs


def test_measurement_names_not_molecular_substrings():
    assert candidate_names(dict(preferred_name='plasma BACE1 concentration',aliases=[]))
    assert candidate_names(dict(preferred_name='BACE1 protein',aliases=[]))
    assert not candidate_names(dict(preferred_name='BACE10',aliases=[]))
    assert not candidate_names(dict(preferred_name='left hippocampal volume',aliases=[]))


def test_scientific_field_proofs_not_full_preimages():
    value={'metadata':{'raw_text':'private scientific source fragment','evidence':{'p':None}},'id':'CLM:one'}
    proof=list(field_hashes(value))
    assert any(r['path']==['metadata','evidence','p'] for r in proof)
    assert 'private scientific source fragment' not in str(proof)


def test_duplicate_differences_do_not_assume_equal_audit():
    a={'metadata':{'audit':{'decision':'include'},'name':'one'}};b={'metadata':{'audit':{'decision':'exclude'},'name':'two'}}
    assert {tuple(r['path']) for r in differences(a,b)}=={('metadata','audit','decision'),('metadata','name')}
    assert list(differences({'one':1},{}))==[dict(path=['one'],left_present=True,right_present=False)]


def test_own_pmc_not_reference_identifier():
    xml='<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>38223081</PMID></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="pmc">PMC10784022</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'
    assert own_pmcs(xml,{'38223081'})=={'38223081':'PMC10784022'}
