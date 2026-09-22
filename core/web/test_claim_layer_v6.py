"""Batch/single equivalence and real revision-boundary failure checks."""
from copy import deepcopy
import json

import pytest
from fastapi.testclient import TestClient

from core.web.claim_layer_v6 import AcceptedClaimLayer, validate_queries
from core.web.claim_evidence import EvidenceUnavailable
from core.web.test_claim_layer_v1 import release, attach


@pytest.fixture
def prepared(tmp_path):
    path, campaign, _, _, dossier, payload = release(tmp_path)
    attach(tmp_path, path, campaign, payload)
    service = AcceptedClaimLayer(path)
    queries = [{'claim_id': 'CLM:0'}, {'claim_id': 'CLM:1'},
               {'relation_id': dossier['relation_id']},
               {'relation_id': payload['groups'][0]['shared_claim_id']},
               {'claim_id': 'CLM:3'}]
    return service, queries, tmp_path


def test_mixed_original_alias_canonical_and_unprojected_batch_is_exact(prepared):
    service, queries, _ = prepared
    expected = [service.query(**query) for query in queries]
    actual = service.query_batch(queries=queries)
    assert actual['results'] == expected
    assert actual['query_count'] == len(queries)
    assert all(r['claim_layer_revision'] == actual['claim_layer_revision'] for r in actual['results'])
    actual['results'][0]['papers'].clear()
    assert service.query_batch(queries=queries)['results'] == expected


def test_shared_base_snapshot_matches_single_queries_and_reloads_on_revision_change(tmp_path):
    path, campaign, _, _, dossier, payload = release(tmp_path)
    service = AcceptedClaimLayer(path)
    queries = [{'claim_id': 'CLM:0'}, {'claim_id': 'CLM:2'}, {'relation_id': dossier['relation_id']}]
    expected = [service.query(**q) for q in queries]
    assert service.query_batch(queries=queries)['results'] == expected
    assert service._batch_base_details
    attach(tmp_path, path, campaign, payload)
    expected_new = [service.query(**q) for q in queries]
    assert service.query_batch(queries=queries)['results'] == expected_new
    assert expected_new != expected


def test_shared_base_batch_does_not_repeat_single_lookup(prepared, monkeypatch):
    from core.web.test_claim_layer_v1 import release
    service, _, root = prepared
    # Remove the optional projection; this exercises unchanged shared evidence.
    campaign = json.loads(service.path.read_text(encoding='utf8'))
    campaign.pop('current_claim_layer')
    service.path.write_text(json.dumps(campaign), encoding='utf8')
    expected = service.query(claim_id='CLM:0')
    def forbidden(*args, **kwargs):
        raise AssertionError('Shared batch fell back to repeated single-source lookup')
    monkeypatch.setattr('core.web.claim_evidence.query_claim_evidence', forbidden)
    actual = service.query_batch(queries=[{'claim_id': 'CLM:0'}]*128)
    assert actual['results'] == [expected]*128


@pytest.mark.parametrize('filename', ['projection.json', 'graph.json', 'reviews.json', 'census.sqlite'])
@pytest.mark.parametrize('when', ['before', 'during'])
def test_dependency_change_before_or_during_batch_returns_no_evidence(prepared, monkeypatch, filename, when):
    service, queries, root = prepared
    service.status()
    def mutate():
        with (root / filename).open('ab') as stream:
            stream.write(b' ')
    if when == 'before':
        mutate()
    else:
        original = service._ensure_current
        def changed_after_begin():
            campaign = original()
            mutate()
            return campaign
        monkeypatch.setattr(service, '_ensure_current', changed_after_begin)
    with pytest.raises((ValueError, EvidenceUnavailable)):
        service.query_batch(queries=queries[:3])


def test_campaign_change_during_batch_fails_closed(prepared, monkeypatch):
    service, queries, _ = prepared
    service.status()
    original = service._ensure_current
    def changed_after_begin():
        campaign = original()
        changed = dict(campaign, status='MANUAL_ACTIVE', active_process={'pid': 1})
        service.path.write_text(json.dumps(changed), encoding='utf8')
        return campaign
    monkeypatch.setattr(service, '_ensure_current', changed_after_begin)
    with pytest.raises(EvidenceUnavailable):
        service.query_batch(queries=queries[:3])


def test_projected_batch_checks_every_dependency_at_both_boundaries_only(prepared, monkeypatch):
    service, queries, _ = prepared
    service.status()
    old_check = service._check_current
    calls = []
    def checked(campaign):
        calls.append(campaign)
        old_check(campaign)
    monkeypatch.setattr(service, '_check_current', checked)
    service.query_batch(queries=[queries[0]] * 128)
    assert len(calls) == 2


@pytest.mark.parametrize('queries', [[], [{}], [{'claim_id': 'CLM:0', 'relation_id': 'REL:0'}],
                                   [{'claim_id': 1}], [{'claim_id': 'REL:0'}],
                                   [{'claim_id': 'CLM:' + 'a'*300}], [{'claim_id': 'CLM:0'}]*129])
def test_invalid_batch_rejected_before_reading(queries):
    with pytest.raises(ValueError):
        validate_queries(queries)


def test_http_batch_matches_single_queries_and_rejects_partial_results(prepared):
    from core.web.server import create_app
    service, queries, root = prepared
    app = create_app()
    app.state.accepted_claim_evidence = service
    with TestClient(app) as client:
        expected = [client.get('/api/kg/claim-evidence', params=q).json() for q in queries]
        response = client.post('/api/kg/claim-evidence-batch', json={'queries': queries})
        assert response.status_code == 200
        assert response.json()['results'] == expected
        missing = client.post('/api/kg/claim-evidence-batch', json={'queries': [queries[0], {'claim_id': 'CLM:missing'}]})
        assert missing.status_code == 404 and 'results' not in missing.json()
        invalid = client.post('/api/kg/claim-evidence-batch', json={'queries': []})
        assert invalid.status_code == 422
        with (root/'projection.json').open('a') as stream:
            stream.write(' ')
        stale = client.post('/api/kg/claim-evidence-batch', json={'queries': queries})
        assert stale.status_code == 503 and stale.json()['available'] is False
