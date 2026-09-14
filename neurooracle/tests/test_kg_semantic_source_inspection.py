from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from inspect_kg_semantic_hold_sources import label_key, projection, paper_identity_matches
from neurooracle.src.kg_identity_pilot import digest


def test_projection_keeps_science_without_serializing_full_audit_or_record():
    row = dict(id='CLM:example', domain_tags=['claim'], aliases=['not-a-preimage'], metadata=dict(
        subject_id='a', subject_name='complete scope', predicate='is_associated_with',
        object_id='b', object_name='outcome', negated=False, raw_text='Proposed, not tested.',
        source_paper=dict(pmid='123', doi='10.1/example'),
        metadata=dict(original_predicate='may_explain', batch=1234),
        scope_reaudit=dict(decision='finalized', contract='old', evidence='unchanged audit')))
    result = projection(row)
    assert result['claim_sha256'] == digest(row)
    assert result['science']['negated'] is False
    assert result['inner_science'] == {'original_predicate': 'may_explain'}
    assert result['prior_audit_sha256'] == digest(row['metadata']['scope_reaudit'])
    assert result['prior_audit_decision'] == 'finalized'
    assert result['audit_not_revalidated'] is True
    assert result['complete_record_not_saved'] is True
    assert 'aliases' not in result and 'metadata' not in result
    assert 'scope_reaudit' not in result
    assert 'batch' not in result['inner_science']


def test_projection_does_not_fill_missing_types_or_negation():
    result = projection(dict(id='CLM:missing', metadata={}))
    assert result['science'] == {}
    assert result['inner_science'] == {}
    assert result['prior_audit_decision'] is None


def test_complete_name_search_does_not_strip_qualifiers_or_punctuation():
    assert label_key('  Left  DLPFC volume ') == 'left dlpfc volume'
    assert label_key('left DLPFC volume') != label_key('DLPFC volume')
    assert label_key('gray-matter alteration') != label_key('gray matter alteration')
    assert label_key(None) == ''


def test_doi_census_case_is_not_a_different_paper():
    assert paper_identity_matches(dict(pmid='123', doi='10.1002/AB.123'), dict(pmid='123', doi='10.1002/ab.123'))
    assert not paper_identity_matches(dict(pmid='124', doi='10.1002/AB.123'), dict(pmid='123', doi='10.1002/ab.123'))
    assert not paper_identity_matches(dict(pmid='123', doi='10.1002/AB.124'), dict(pmid='123', doi='10.1002/ab.123'))
    assert not paper_identity_matches(dict(pmid='', doi='10.1002/AB.123'), dict(pmid='123', doi='10.1002/ab.123'))
