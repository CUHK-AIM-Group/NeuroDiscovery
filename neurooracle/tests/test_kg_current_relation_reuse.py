"""Read-only relation reuse census: never certify a coarser merge."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import inspect_kg_current_relation_reuse as review
from neurooracle.src.correlation_grouping import IndexTerms
from neurooracle.src.relation_evidence import relation_id


class Papers:
    def resolve(self, claim):
        return dict(paper_key=claim['source_paper']['key'], status=claim['source_paper'].get('status','verified'))
    def publication_review(self, key):
        return dict(status='retracted' if key.endswith(':retracted') else 'not_reviewed')


def claim(cid='CLM:a', **patch):
    md = dict(id=cid, subject_id='N:1', subject_name='Left volume', predicate='is_associated_with',
        object_id='N:2', object_name='Outcome', source_paper=dict(key='pmid:1'),
        negated=False, evidence=dict(direction='positive'), metadata={})
    md.update(patch)
    return dict(id=cid, metadata=md)


def database(records):
    db = sqlite3.connect(':memory:')
    review.initialize(db)
    review.add_rows(db, [review.claim_projection(r, IndexTerms(), Papers()) for r in records])
    return db


def test_exact_full_name_keeps_case_and_qualifiers_but_normalizes_whitespace():
    a = claim()['metadata']; b = deepcopy(a)
    b['subject_name'] = '  Left\nvolume  '
    assert review.diagnostic_keys(a)['surface'] == review.diagnostic_keys(b)['surface']
    b['subject_name'] = 'Left anterior volume'
    assert review.diagnostic_keys(a)['surface'] != review.diagnostic_keys(b)['surface']


def test_casefold_is_a_separate_diagnostic_not_the_exact_key():
    a = claim(subject_name='Vcp')['metadata']; b = claim(subject_name='VCP')['metadata']
    ak, bk = review.diagnostic_keys(a), review.diagnostic_keys(b)
    assert ak['surface'] != bk['surface'] and ak['folded'] == bk['folded']


@pytest.mark.parametrize('predicate', ['causes','predicts','treats','is_associated_with','reduces'])
def test_only_accepted_correlation_policy_is_symmetric(predicate):
    assert review.diagnostic_triple('b',predicate,'a') != review.diagnostic_triple('a',predicate,'b')


def test_correlation_symmetry_does_not_mutate_original_claim_orientation():
    row = claim(subject_name='Z', object_name='A', predicate='correlates_with')
    old = deepcopy(row); review.claim_projection(row, IndexTerms(), Papers())
    assert row == old
    assert review.diagnostic_triple('b','correlates_with','a') == review.diagnostic_triple('a','correlates_with','b')


@pytest.mark.parametrize('field', ['subject_id','subject_name','object_id','object_name','predicate'])
def test_incomplete_diagnostic_key_rejected(field):
    md = claim()['metadata']; md[field] = ''
    with pytest.raises(ValueError): review.diagnostic_keys(md)


def test_name_only_fragmentation_does_not_rewrite_fine_identities():
    records = [claim(), claim('CLM:b', subject_id='N:other', source_paper=dict(key='pmid:2'))]
    db = database(records)
    try:
        fine = review.group_statistics(db,'rid'); names = review.group_statistics(db,'surface')
        assert fine['groups'] == fine['singleton_groups'] == 2
        assert names['groups'] == names['multiple_fine_relation_groups'] == names['shared_groups'] == 1
        assert names['claims_in_multiple_fine_relation_groups'] == 2
        assert names['multi_verified_not_retracted_groups'] == 1
        samples = review.bounded_examples(db,'surface')
        assert len(samples) == 1 and samples[0]['candidate_only']
        assert len({m['rid'] for m in samples[0]['members']}) == 2
        assert db.execute('SELECT COUNT(*) FROM claims').fetchone()[0] == 2
    finally: db.close()


def test_id_only_can_falsely_combine_different_complete_scope():
    db = database([claim(), claim('CLM:b', subject_name='Bilateral mean volume')])
    try:
        assert review.group_statistics(db,'rid')['groups'] == 2
        physical = review.group_statistics(db,'physical')
        assert physical['groups'] == physical['multiple_exact_surface_groups'] == 1
        assert physical['multiple_fine_relation_groups'] == 1
    finally: db.close()


def test_same_paper_repeated_claims_do_not_become_two_articles():
    db = database([claim(),claim('CLM:b')])
    try:
        stats = review.group_statistics(db,'rid')
        assert stats['claims'] == stats['shared_claims'] == 2
        assert stats['groups'] == stats['shared_groups'] == 1
        assert stats['multi_source_key_groups'] == stats['multi_verified_paper_groups'] == 0
    finally: db.close()


def test_unverified_and_retracted_sources_do_not_inflate_default_count():
    db = database([claim(), claim('CLM:b',source_paper=dict(key='pmid:retracted')),
        claim('CLM:c',source_paper=dict(key='unknown:3',status='unverified'))])
    try:
        stats = review.group_statistics(db,'rid')
        assert stats['multi_source_key_groups'] == stats['multi_verified_paper_groups'] == 1
        assert stats['multi_verified_not_retracted_groups'] == 0
    finally: db.close()


def test_conflict_is_not_a_verified_paper_and_does_not_disappear():
    db = database([claim(),claim('CLM:b',source_paper=dict(key='unresolved:CLM:b',status='conflict'))])
    try:
        stats = review.group_statistics(db,'rid')
        assert stats['claims'] == 2 and stats['multi_source_key_groups'] == 1
        assert stats['multi_verified_paper_groups'] == 0
    finally: db.close()


def test_context_and_typed_negation_remain_distinct_per_claim():
    db = database([claim(),claim('CLM:b',negated=0),claim('CLM:c',negated=True),
        claim('CLM:d',evidence=dict(direction='negative'))])
    try:
        stats = review.group_statistics(db,'rid')
        assert stats['claims'] == 4 and stats['groups'] == 1
        assert stats['mixed_negation_groups'] == stats['multiple_evidence_context_groups'] == 1
        assert db.execute('SELECT COUNT(DISTINCT neg) FROM claims').fetchone()[0] == 3
        assert db.execute('SELECT COUNT(DISTINCT ctx) FROM claims').fetchone()[0] == 4
    finally: db.close()


def test_duplicate_claim_id_and_inner_identity_conflict_rejected():
    with pytest.raises(sqlite3.IntegrityError):
        db = database([claim(),claim()])
    row = claim(); row['metadata']['id'] = 'CLM:wrong'
    with pytest.raises(ValueError): review.claim_projection(row,IndexTerms(),Papers())


@pytest.mark.parametrize('field', ['name','claims; DROP TABLE claims',''])
def test_sql_grouping_is_finite(field):
    db = database([claim()])
    try:
        with pytest.raises(ValueError): review.group_statistics(db,field)
    finally: db.close()


def test_distribution_uses_record_weight_and_nearest_rank():
    result = review.distribution(Counter({0:1,2:8,20:1}))
    assert result['records'] == 10 and result['mean'] == 3.6
    assert (result['minimum'],result['median'],result['p90'],result['maximum']) == (0,2,2,20)
    with pytest.raises(ValueError): review.distribution(Counter())


def test_all_current_query_crosschecks_fail_closed():
    stats = dict(groups=2,claims=3,singleton_groups=1,shared_groups=1,shared_claims=2,
        multi_source_key_groups=1,multi_verified_paper_groups=1,multi_verified_not_retracted_groups=1)
    campaign = dict(counts=dict(claims=3),relation_evidence_counts=dict(all_fine_grained_relation_groups=2,
        all_claims=3,shared_groups=1,indexed_claims=2,multi_paper_shared_groups=1,verified_multi_paper_shared_groups=1))
    review.crosscheck_fine_groups(stats,campaign,[dict(claim_count=2)],[1],[1])
    for key in stats:
        wrong = dict(stats); wrong[key] += 1
        with pytest.raises(ValueError): review.crosscheck_fine_groups(wrong,campaign,[dict(claim_count=2)],[1],[1])
