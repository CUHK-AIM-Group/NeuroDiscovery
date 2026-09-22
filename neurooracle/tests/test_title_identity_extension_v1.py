from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from neurooracle.src.kg_title_identity_extension_v1 import (
    OwningTitleWitnesses, require_monotonic_registry, require_record_extension,
)
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.pubmed_title_evidence import complete_title_evidence
from neurooracle.tests.test_complete_pubmed_title_evidence import xml
from neurooracle.tests.test_kg_paper_authorities import registry, source_record, claim


def fixture(tmp_path):
    old = registry(source_record(title='A (18)F study.')).export_payload()
    article = xml('A <sup>18</sup>F study.')
    root = ET.Element('PubmedArticleSet');root.append(article)
    path = tmp_path/'own.xml';path.write_bytes(ET.tostring(root))
    stat = path.stat()
    fp = dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    witness = dict(response_path=str(path),response_sha256=fp['sha256'],
                   url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=1')
    new = deepcopy(old)
    new['records']['1']['complete_title_evidence'] = complete_title_evidence(article,old['records']['1'],witness)
    calls = []
    def check_file(expected,full_hash=False):
        p = Path(expected['path']);s = p.stat()
        assert (s.st_size,s.st_mtime_ns)==(expected['bytes'],expected['mtime_ns'])
        if full_hash:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
        calls.append(full_hash)
        return p
    return old,new,fp,OwningTitleWitnesses([fp],check_file),calls


def test_appended_evidence_resolves_presentation_only_without_mutating_original(tmp_path):
    old,new,_,proof,calls = fixture(tmp_path)
    frozen = deepcopy(old);original = claim(pmid='1',title='A 18F study')
    before = deepcopy(original)
    require_monotonic_registry(old,new);proof.require_authority(new['records']['1'],old['records'])
    assert VerifiedPaperIdentities(old).resolve(original)['status']=='conflict'
    assert VerifiedPaperIdentities(new).resolve(original)['status']=='verified'
    assert old==frozen and original==before
    proof.verify(new['records']['1'])
    assert calls.count(True)==1


@pytest.mark.parametrize('field,value',[
    ('title','A 19F study'),('years',['2025']),('doi',['10.1234/different']),
    ('pmcid',['PMC2']),('preprint',True),('witness',{'response_sha256':'c'*64}),
])
def test_existing_bibliography_and_authority_cannot_change(tmp_path,field,value):
    old,new,_,_,_=fixture(tmp_path);new['records']['1'][field]=value
    with pytest.raises(ValueError):require_monotonic_registry(old,new)


@pytest.mark.parametrize('field,value',[
    ('publication_reviews',{'1':{'status':'not_reviewed'}}),
    ('observed_collisions',{'doi':[],'pmcid':['PMC1']}),('source','different authority'),
])
def test_registry_publication_and_collision_controls_cannot_change(tmp_path,field,value):
    old,new,_,_,_=fixture(tmp_path);new[field]=value
    with pytest.raises(ValueError,match='metadata'):require_monotonic_registry(old,new)


def test_previously_accepted_title_proof_cannot_be_replaced_or_removed(tmp_path):
    _,accepted,_,_,_=fixture(tmp_path)
    for change in ('replace','remove'):
        new=deepcopy(accepted)
        if change=='replace':new['records']['1']['complete_title_evidence']['witness']['url']+='&other=1'
        else:new['records']['1'].pop('complete_title_evidence')
        with pytest.raises(ValueError):require_monotonic_registry(accepted,new)


def test_owning_raw_file_is_required_even_for_schema_valid_title(tmp_path):
    old,new,fp,proof,_=fixture(tmp_path)
    new['records']['1']['complete_title_evidence']['witness']['response_path']=str(tmp_path/'another.xml')
    require_monotonic_registry(old,new)
    with pytest.raises(ValueError,match='declared owning'):proof.verify(new['records']['1'])


def test_cached_proof_detects_later_raw_mutation(tmp_path):
    old,new,fp,proof,_=fixture(tmp_path);proof.require_authority(new['records']['1'],old['records'])
    Path(fp['path']).write_bytes(b'changed')
    with pytest.raises(AssertionError):proof.verify(new['records']['1'])


def test_schema_valid_alternate_markup_still_must_match_actual_raw_xml(tmp_path):
    old,new,_,proof,_=fixture(tmp_path)
    new['records']['1']['complete_title_evidence']['title_xml']='<ArticleTitle>A (18)F study.</ArticleTitle>'
    require_monotonic_registry(old,new)
    with pytest.raises(ValueError,match='own raw XML'):proof.verify(new['records']['1'])


def test_no_authority_from_an_unaccepted_summary_only_record(tmp_path):
    _,new,_,proof,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='New authority'):proof.require_authority(new['records']['1'],{})


def test_raw_owning_authority_can_be_added_and_proved(tmp_path):
    from neurooracle.scripts.extend_cached_kg_paper_authority_20260913 import authority_record
    _,old,fp,proof,_=fixture(tmp_path)
    witness=old['records']['1']['complete_title_evidence']['witness']
    article=proof.article('1',witness)
    actual=authority_record(article,fp)
    actual['complete_title_evidence']=complete_title_evidence(article,actual,witness)
    proof.require_authority(actual,{})
    assert len(proof.inputs)==1


def test_arbitrary_extra_fields_do_not_pass_as_title_annotations(tmp_path):
    old,new,_,_,_=fixture(tmp_path);new['records']['1']['approved_support']=True
    with pytest.raises(ValueError):require_record_extension(old['records']['1'],new['records']['1'])


def test_previous_record_cannot_disappear(tmp_path):
    old,new,_,_,_=fixture(tmp_path);new['records'].clear()
    with pytest.raises(ValueError,match='removed'):require_monotonic_registry(old,new)
