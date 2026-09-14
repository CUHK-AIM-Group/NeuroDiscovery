"""Current shared-relation observations, not an inferred consensus or trial count.

The bounded derived file includes current shared observations only. It adds no
node/edge metadata. Unknown article roles and cohort independence stay unknown.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from .kg_identity_pilot import digest
from .relation_evidence import name_key
from .shared_relation_catalog import check_file, find_shared_relations

VERSION = 'kg.relation_evidence_dossier.v1'
REVIEW_TYPES = frozenset({'Review', 'Systematic Review', 'Meta-Analysis'})


def observation(record, papers, source_reviews):
    md = record['metadata']
    identity = papers.resolve(md)
    review = source_reviews.get(record['id'])
    if review is not None and review['claim_sha256'] != digest(record):
        raise ValueError('source review is not bound to this current claim')
    return dict(claim_id=record['id'], claim_sha256=digest(record), claim=deepcopy(md),
                source_identity=identity, publication_review=papers.publication_review(identity['paper_key']),
                source_review=deepcopy(review))


def summarize(group, observations):
    expected = {m['claim_id']: m for m in group['members']}
    observed = {o['claim_id']: o for o in observations}
    if len(observed) != len(observations) or set(observed) != set(expected):
        raise ValueError('dossier membership is not the complete current relation')
    polarity = Counter()
    review_keys, role_unknown, text_keys = set(), set(), defaultdict(set)
    for cid, item in observed.items():
        md, identity = item['claim'], item['source_identity']
        if md.get('id') != cid or type(md.get('negated')) is not bool:
            raise ValueError('invalid observation identity or negation')
        if identity['paper_key'] != expected[cid]['paper_key'] or identity['status'] != expected[cid]['paper_status']:
            raise ValueError('dossier source differs from accepted relation catalog')
        if md['negated'] is not expected[cid]['negated']:
            raise ValueError('dossier polarity differs from relation catalog')
        polarity['negated' if md['negated'] else 'not_negated'] += 1
        pk = identity['paper_key']
        review = item.get('source_review')
        if review and review['claim_sha256'] == item['claim_sha256'] and review.get('source_role') == 'review_or_evidence_synthesis':
            review_keys.add(pk)
        else:
            role_unknown.add(pk)
        text = name_key(md.get('raw_text'))
        if text:
            text_keys[text].add(pk)
    sources = {m['paper_key'] for m in group['members']}
    # A reviewed article role is reusable within an article; this does NOT
    # establish that any particular sentence is its own empirical result.
    role_unknown -= review_keys
    all_reviews = bool(sources) and sources <= review_keys
    return dict(relation_id=group['id'], claim_count=len(observations), source_key_count=len(sources),
                verified_article_count=group['verified_paper_count'],
                source_review_article_count=len(review_keys), source_role_unresolved_count=len(role_unknown),
                independent_primary_study_count=0 if all_reviews else None,
                independence_status='all_sources_are_reviews_not_primary_trials' if all_reviews else 'not_established',
                negation_counts=dict(polarity),
                mixed_negation_requires_context_review=all(polarity[k] for k in ('negated', 'not_negated')),
                cross_source_identical_text_groups=sum(len(keys) > 1 for keys in text_keys.values()),
                consensus_inferred=False, contradiction_inferred=False,
                review_is_not_primary_trial=True, observations=sorted(observations, key=lambda x: x['claim_id']))


def find_relation_dossiers(campaign_path, **filters):
    campaign_path = Path(campaign_path)
    campaign = json.loads(campaign_path.read_text(encoding='utf8'))
    groups = find_shared_relations(campaign_path, **filters)
    receipt = json.loads(check_file(campaign['current_acceptance'], full_hash=True).read_text(encoding='utf8'))
    fp = campaign.get('current_evidence_dossiers')
    if not fp or receipt.get('evidence_dossiers') != fp or not receipt.get('checks', {}).get('shared_observation_dossiers_complete'):
        raise ValueError('current evidence dossiers are not validated')
    path = check_file(fp, full_hash=True)
    wanted = {g['id']: g for g in groups}
    result = []
    seen = set()
    with path.open(encoding='utf8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['relation_id'] in wanted:
                if row['relation_id'] in seen:
                    raise ValueError('duplicate dossier identity')
                reproduced = summarize(wanted[row['relation_id']], row['observations'])
                if reproduced != row:
                    raise ValueError('dossier summary is inconsistent')
                result.append(dict(relation=wanted[row['relation_id']], evidence=row))
                seen.add(row['relation_id'])
    if seen != set(wanted):
        raise ValueError('missing current relation dossier')
    check_file(campaign['current_graph'])
    check_file(fp)
    if json.loads(campaign_path.read_text(encoding='utf8')) != campaign:
        raise ValueError('current KG changed during evidence lookup')
    return result
