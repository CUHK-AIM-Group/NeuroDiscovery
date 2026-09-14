from copy import deepcopy
import json

import pytest

from core.web.claim_layer_v1 import AcceptedClaimLayer, EvidenceUnavailable, FIELDS, VERSION, project
from neurooracle.src.relation_evidence import relation_id
from neurooracle.tests.test_claim_evidence_query import fingerprint, setup


def release(tmp_path, *, versions=False):
    path, records, dossier = setup(tmp_path)
    campaign = json.loads(path.read_text())
    catalog = [json.loads(x) for x in (tmp_path / 'catalog.jsonl').read_text().splitlines()]
    old = dossier['relation_id']
    claim = {k: catalog[0][k] for k in FIELDS}
    claim.update(subject_name='scoped exposure', subject_id='SCOPED:X')
    target = relation_id(tuple(claim[k] for k in FIELDS))
    reviews = {}
    for obs in dossier['observations']:
        if versions:
            obs['source_review'].update(verified_work_key='one_work', verified_version_pmids=['111', '222'],
                                       preferred_version_pmid='222', version_identity_witness={'sha256': 'f' * 64})
        reviews[obs['claim_id']] = dict(claim_sha256=obs['claim_sha256'], pmid=obs['claim']['source_paper']['pmid'],
                                       observation_role='own_result', proposition_support='supports',
                                       source_anchor='Source result', scope_note='Source-qualified finding')
    group = dict(shared_claim_id=target, claim=claim, source_relation_ids=[old],
                 original_claim_ids=[r['id'] for r in records[:3]], reviewed_claim_ids=list(reviews),
                 reviewed_supporting_article_count=1 if versions else 2, scope_note='Limited to the tested context')
    payload = dict(groups=[group], source_reviews=reviews)
    return path, campaign, records, catalog, dossier, payload


def attach(tmp_path, path, campaign, payload, *, bad=False):
    projection = tmp_path / ('bad_projection.json' if bad else 'projection.json')
    projection.write_text(json.dumps(payload), encoding='utf8')
    manifest = dict(schema=VERSION, status='ACCEPTED', base_graph=campaign['current_graph'],
                    base_acceptance=campaign['current_acceptance'], base_dossiers=campaign['current_evidence_dossiers'],
                    projection=fingerprint(projection), counts={'reviewed_multipaper_claims_after': 1},
                    checks=dict(source_anchors_and_whole_scopes_validated=True, original_observations_unchanged=True))
    layer = tmp_path / ('bad_layer.json' if bad else 'layer.json')
    layer.write_text(json.dumps(manifest), encoding='utf8')
    updated = dict(campaign, current_claim_layer=fingerprint(layer))
    path.write_text(json.dumps(updated), encoding='utf8')
    return updated


def test_repaired_proposition_resolves_from_all_original_and_previous_ids(tmp_path):
    path, campaign, records, _, dossier, payload = release(tmp_path)
    attach(tmp_path, path, campaign, payload)
    service = AcceptedClaimLayer(path)
    target = payload['groups'][0]['shared_claim_id']
    expected = service.query(relation_id=target)
    assert expected['claim']['subject_name'] == 'scoped exposure'
    assert expected['canonical_scope_note'] == 'Limited to the tested context'
    for cid in ('CLM:0', 'CLM:1', 'CLM:2'):
        assert service.query(claim_id=cid)['papers'] == expected['papers']
    assert service.query(relation_id=dossier['relation_id'])['papers'] == expected['papers']
    original = {o['claim_id']: o['original_claim'] for p in expected['papers'] for o in p['observations']}
    assert original == {r['id']: r['metadata'] for r in records[:3]}
    assert service.search(dossier['relation_id'])['total'] == 1
    assert service.status()['reviewed_multipaper_claim_count'] == 1
    assert service.query(claim_id='CLM:3')['article_count'] == 1
    assert expected['source'] == 'accepted_claim_layer' and expected['claim_layer_revision']


def test_absent_layer_and_live_removal_return_the_accepted_baseline(tmp_path):
    path, campaign, _, _, _, payload = release(tmp_path)
    service = AcceptedClaimLayer(path)
    assert service.query(claim_id='CLM:0')['claim']['subject_name'] == 'exposure'
    attach(tmp_path, path, campaign, payload)
    assert service.query(claim_id='CLM:0')['claim']['subject_name'] == 'scoped exposure'
    path.write_text(json.dumps(campaign), encoding='utf8')
    assert service.query(claim_id='CLM:0')['claim']['subject_name'] == 'exposure'
    assert 'claim_layer_revision' not in service.status()


@pytest.mark.parametrize('fault', ['missing_member', 'outside_review', 'wrong_hash', 'wrong_paper', 'unverified', 'double_assignment', 'wrong_count'])
def test_projection_rejects_invalid_complete_scope_or_source_binding(tmp_path, fault):
    _, _, _, catalog, dossier, payload = release(tmp_path)
    group = payload['groups'][0]
    if fault == 'missing_member': group['original_claim_ids'].pop()
    elif fault == 'outside_review': group['reviewed_claim_ids'].append('CLM:3')
    elif fault == 'wrong_hash': payload['source_reviews']['CLM:0']['claim_sha256'] = 'x' * 64
    elif fault == 'wrong_paper': payload['source_reviews']['CLM:0']['pmid'] = '222'
    elif fault == 'unverified': dossier['observations'][0]['source_identity']['status'] = 'unverified'
    elif fault == 'double_assignment': payload['groups'].append(deepcopy(group))
    else: group['reviewed_supporting_article_count'] = 3
    with pytest.raises((EvidenceUnavailable, ValueError)):
        project(catalog, [dossier], payload)


def test_existing_version_identity_and_review_provenance_survive_re_review(tmp_path):
    _, _, _, catalog, dossier, payload = release(tmp_path, versions=True)
    _, _, _, details = project(catalog, [dossier], payload)
    data = next(iter(details.values()))
    assert data['reviewed_supporting_article_count'] == 1
    assert data['publication_record_count'] == 2
    assert data['verified_versions_deduplicated'] == 1
    for paper in data['papers']:
        for obs in paper['observations']:
            assert obs['source_review']['previous_source_review']['verified_work_key'] == 'one_work'
    payload['source_reviews']['CLM:0']['verified_work_key'] = 'another_work'
    with pytest.raises(EvidenceUnavailable, match='Conflicting'):
        project(catalog, [dossier], payload)


def test_rejected_layer_cannot_poison_return_to_previous_valid_revision(tmp_path):
    path, campaign, _, _, _, payload = release(tmp_path)
    good = attach(tmp_path, path, campaign, payload)
    service = AcceptedClaimLayer(path)
    assert service.query(claim_id='CLM:0')['claim']['subject_name'] == 'scoped exposure'
    bad = deepcopy(payload)
    bad['groups'][0]['reviewed_supporting_article_count'] = 99
    attach(tmp_path, path, campaign, bad, bad=True)
    with pytest.raises(EvidenceUnavailable): service.status()
    path.write_text(json.dumps(good), encoding='utf8')
    assert service.query(claim_id='CLM:0')['claim']['subject_name'] == 'scoped exposure'


def test_modified_projection_fails_closed_even_after_cache_load(tmp_path):
    path, campaign, _, _, _, payload = release(tmp_path)
    attach(tmp_path, path, campaign, payload)
    service = AcceptedClaimLayer(path)
    service.status()
    with (tmp_path / 'projection.json').open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError): service.query(claim_id='CLM:0')
