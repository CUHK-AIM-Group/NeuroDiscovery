"""Run real batch/revision/HTTP regression cases with the additive-title reader."""
import pytest
import json

from core.web import test_claim_layer_v6 as legacy
from core.web.claim_layer_v8 import AcceptedClaimLayer, validate_queries
from core.web.test_claim_layer_v6 import (
    prepared,
    test_mixed_original_alias_canonical_and_unprojected_batch_is_exact,
    test_shared_base_snapshot_matches_single_queries_and_reloads_on_revision_change,
    test_shared_base_batch_does_not_repeat_single_lookup,
    test_dependency_change_before_or_during_batch_returns_no_evidence,
    test_campaign_change_during_batch_fails_closed,
    test_projected_batch_checks_every_dependency_at_both_boundaries_only,
    test_invalid_batch_rejected_before_reading,
    test_http_batch_matches_single_queries_and_rejects_partial_results,
)


@pytest.fixture(autouse=True)
def additive_reader(monkeypatch):
    monkeypatch.setattr(legacy, 'AcceptedClaimLayer', AcceptedClaimLayer)
    monkeypatch.setattr(legacy, 'validate_queries', validate_queries)


@pytest.mark.parametrize('role', ['historical_web', 'other_code', 'source_input'])
def test_only_the_historical_web_audit_witness_may_drift(tmp_path, monkeypatch, role):
    from core.web.test_claim_layer_v1 import release, attach
    from neurooracle.tests.test_claim_evidence_query import fingerprint
    from core.web import claim_layer_v7
    from core.web.claim_evidence import EvidenceUnavailable
    path, campaign, _, _, _, payload = release(tmp_path)
    web_dir = tmp_path / 'web'
    web_dir.mkdir()
    server = web_dir / 'server.py'
    server.write_text('old application', encoding='utf-8')
    old = fingerprint(server)
    if role == 'source_input':
        payload['source_support_inputs'] = [old]
    updated = attach(tmp_path, path, campaign, payload)
    manifest_path = tmp_path / 'layer.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['review_inputs'] = [old]
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    updated['current_claim_layer'] = fingerprint(manifest_path)
    path.write_text(json.dumps(updated), encoding='utf-8')
    witness = manifest_path.read_bytes()
    if role != 'other_code':
        monkeypatch.setattr(claim_layer_v7, '__file__', str(web_dir / 'claim_layer_v7.py'))
    server.write_text('new application', encoding='utf-8')
    service = AcceptedClaimLayer(path)
    if role == 'historical_web':
        result = service.query_batch(queries=[{'claim_id': 'CLM:0'}])
        assert result['historical_code_drift'][0]['accepted_sha256'] == old['sha256']
        assert result['results'][0]['article_count'] == 2
        with (tmp_path / 'projection.json').open('a') as stream:
            stream.write(' ')
        with pytest.raises((ValueError, EvidenceUnavailable)):
            service.status()
    else:
        with pytest.raises((ValueError, EvidenceUnavailable)):
            service.status()
    assert manifest_path.read_bytes() == witness
