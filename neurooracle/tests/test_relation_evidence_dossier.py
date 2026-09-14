from copy import deepcopy
import pytest

from neurooracle.src.kg_identity_pilot import digest
from neurooracle.src.relation_evidence_dossier import summarize, observation


class Papers:
    def resolve(self, md):
        return dict(paper_key=md['source_paper']['key'], status='verified')
    def publication_review(self, key):
        return dict(status='not_reported')


def case(negative=False, same_paper=False, reviewed=False):
    rows = []
    for index in range(2):
        cid = 'CLM:' + str(index)
        md = dict(id=cid, negated=bool(index and negative), raw_text='A complete observation',
            source_paper=dict(key='pmid:1' if same_paper else 'pmid:' + str(index)),
            conditions=['only under treatment'], population='sample',
            evidence=dict(sample_size=0, null_field=None, methodology='test'),
            metadata=dict(unknown_science_field={'value': False}))
        record = dict(id=cid, metadata=md)
        reviews = {cid: dict(claim_sha256=digest(record), source_role='review_or_evidence_synthesis')} if reviewed else {}
        rows.append(observation(record, Papers(), reviews))
    members = [dict(claim_id=o['claim_id'], paper_key=o['source_identity']['paper_key'],
        paper_status='verified', negated=o['claim']['negated']) for o in rows]
    return dict(id='REL:one', members=members, verified_paper_count=len({m['paper_key'] for m in members})), rows


def test_distinct_articles_never_imply_independent_studies():
    group, rows = case()
    out = summarize(group, rows)
    assert out['verified_article_count'] == 2
    assert out['independent_primary_study_count'] is None
    assert out['independence_status'] == 'not_established'
    assert out['cross_source_identical_text_groups'] == 1
    assert len(out['observations']) == 2


def test_known_reviews_are_not_primary_trials():
    group, rows = case(reviewed=True)
    out = summarize(group, rows)
    assert out['source_review_article_count'] == 2
    assert out['independent_primary_study_count'] == 0
    assert out['source_role_unresolved_count'] == 0


def test_one_unknown_source_does_not_become_zero_primary_studies():
    group, rows = case(reviewed=True)
    rows[1]['source_review'] = None
    out = summarize(group, rows)
    assert out['independent_primary_study_count'] is None
    assert out['source_role_unresolved_count'] == 1


def test_negative_and_positive_keep_context_without_claiming_contradiction():
    group, rows = case(negative=True)
    out = summarize(group, rows)
    assert out['mixed_negation_requires_context_review']
    assert out['negation_counts'] == dict(not_negated=1, negated=1)
    assert out['contradiction_inferred'] is False and out['consensus_inferred'] is False
    assert out['observations'][1]['claim']['conditions'] == ['only under treatment']
    assert out['observations'][1]['claim']['evidence']['sample_size'] == 0
    assert out['observations'][1]['claim']['metadata']['unknown_science_field']['value'] is False


def test_same_paper_is_one_source_even_for_different_claims():
    group, rows = case(same_paper=True)
    out = summarize(group, rows)
    assert out['claim_count'] == 2 and out['source_key_count'] == 1
    assert out['cross_source_identical_text_groups'] == 0


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'source', 'negation', 'identity', 'nonboolean'])
def test_dossier_rejects_broken_source_or_membership(damage):
    group, rows = case()
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows.append(deepcopy(rows[0]))
    elif damage == 'source': rows[0]['source_identity']['paper_key'] = 'other'
    elif damage == 'negation': rows[0]['claim']['negated'] = True
    elif damage == 'identity': rows[0]['claim']['id'] = 'CLM:other'
    elif damage == 'nonboolean': rows[0]['claim']['negated'] = 'false'
    with pytest.raises(ValueError): summarize(group, rows)


def test_stale_source_role_not_reused_after_scientific_change():
    record = dict(id='CLM:a', metadata=dict(id='CLM:a', source_paper=dict(key='pmid:1')))
    with pytest.raises(ValueError, match='not bound'):
        observation(record, Papers(), {'CLM:a': dict(claim_sha256='outdated')})
