"""The additive-title path retains original-relation and completion guards."""
from copy import deepcopy
import json
import sqlite3

import pytest

from core.web.claim_evidence import EvidenceUnavailable
from core.web.claim_layer_extension_v6 import extend_base
from core.web import test_claim_layer_v2 as legacy
from neurooracle.src.kg_paper_identity import VerifiedPaperIdentities
from neurooracle.src.kg_identity_pilot import digest
from neurooracle.tests.test_claim_evidence_query import fingerprint
from neurooracle.tests.test_kg_paper_authorities import claim
from neurooracle.tests.test_title_identity_extension_v1 import fixture as title_fixture


def test_unchanged_relation_extension_retains_complete_original(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, 'extend_base', extend_base)
    legacy.test_extension_reads_the_unchanged_complete_original_relation(tmp_path)


@pytest.mark.parametrize('fault', ['hash', 'offset', 'duplicate', 'old_member',
                                  'incomplete', 'graph', 'altered_identity'])
def test_original_guards_still_reject_invalid_extensions(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(legacy, 'extend_base', extend_base)
    legacy.test_extension_rejects_false_bindings_and_incomplete_scope(tmp_path, fault)


def prepared_completion(tmp_path):
    campaign, _, relations, dossiers, payload, _ = legacy.fixture(tmp_path)
    old, new, raw_fp, _, _ = title_fixture(tmp_path)
    previous = tmp_path / 'old-title-authority.json'
    previous.write_text(json.dumps(old), encoding='utf-8')
    added = tmp_path / 'added-title-authority.json'
    added.write_text(json.dumps(new), encoding='utf-8')
    campaign['current_paper_identities'] = fingerprint(previous)
    original = claim(pmid='1', title='A 18F study')
    obs = dict(claim_id='CLM:own-title', claim_sha256='original-node-sha',
               claim=original, source_review=None,
               source_identity=VerifiedPaperIdentities(old).resolve(original),
               publication_review=VerifiedPaperIdentities(old).publication_review('pmid:1'))
    entry = dict(claim_id=obs['claim_id'], claim_sha256=obs['claim_sha256'], pmid='1',
                 previous_source_identity=deepcopy(obs['source_identity']),
                 source_identity=VerifiedPaperIdentities(new).resolve(original))
    assert obs['source_identity']['status'] == 'conflict'
    assert entry['source_identity']['status'] == 'verified'
    dossiers[0]['observations'] = [obs]
    payload['source_reviews'] = {obs['claim_id']: {'proposition_support': 'supports'}}
    payload['original_relation_extension'].update(
        members=[], supplemental_identity_registry=fingerprint(added),
        raw_authority_files=[raw_fp], existing_identity_completions=[entry])
    return campaign, relations, dossiers, payload, added, raw_fp


def test_own_title_completion_preserves_bibliography_and_prior_identity(tmp_path):
    campaign, relations, dossiers, payload, _, raw_fp = prepared_completion(tmp_path)
    before = deepcopy(dossiers)
    _, result, inputs = extend_base(relations, dossiers, payload, campaign)
    assert dossiers == before
    old, new = before[0]['observations'][0], result[0]['observations'][0]
    assert old['claim'] == new['claim']
    assert old['source_review'] == new['source_review']
    assert new['previous_source_identity'] == old['source_identity']
    assert new['source_identity']['status'] == 'verified'
    assert raw_fp in inputs


@pytest.mark.parametrize('fault', ['no_raw', 'wrong_markup', 'wrong_pmid', 'repeat',
                                  'prior_review', 'prior_verified', 'no_review', 'missing_member'])
def test_title_completion_cannot_bypass_source_and_history_guards(tmp_path, fault):
    campaign, relations, dossiers, payload, added, _ = prepared_completion(tmp_path)
    ext = payload['original_relation_extension']
    entry = ext['existing_identity_completions'][0]
    obs = dossiers[0]['observations'][0]
    if fault == 'no_raw':
        ext['raw_authority_files'] = []
    elif fault == 'wrong_markup':
        data = json.loads(added.read_text(encoding='utf-8'))
        data['records']['1']['complete_title_evidence']['title_xml'] = '<ArticleTitle>A (18)F study.</ArticleTitle>'
        added.write_text(json.dumps(data), encoding='utf-8')
        ext['supplemental_identity_registry'] = fingerprint(added)
    elif fault == 'wrong_pmid':
        entry['source_identity']['paper_key'] = 'pmid:2'
    elif fault == 'repeat':
        ext['existing_identity_completions'].append(deepcopy(entry))
    elif fault == 'prior_review':
        obs['source_review'] = {'proposition_support': 'supports'}
    elif fault == 'prior_verified':
        obs['source_identity'] = deepcopy(entry['source_identity'])
        entry['previous_source_identity'] = deepcopy(obs['source_identity'])
    elif fault == 'no_review':
        payload['source_reviews'].clear()
    else:
        dossiers[0]['observations'] = []
    with pytest.raises((ValueError, EvidenceUnavailable)):
        extend_base(relations, dossiers, payload, campaign)


def test_previously_extended_original_replays_its_real_prior_identity_before_completion(tmp_path):
    campaign, records, relations, dossiers, payload, entries = legacy.fixture(tmp_path)
    old, new, raw_fp, _, _ = title_fixture(tmp_path)
    previous = tmp_path/'previous-extension-identities.json'
    previous.write_text(json.dumps(old), encoding='utf-8')
    added = tmp_path/'added-title-identities.json'
    added.write_text(json.dumps(new), encoding='utf-8')
    campaign['current_paper_identities'] = fingerprint(previous)
    # Change only the synthetic fixture's bibliography, then rebuild its real
    # byte-offset/index seals. The original relation and claim ID are retained.
    original = deepcopy(records[3])
    original['metadata']['source_paper'] = claim(pmid='1', title='A 18F study')['source_paper']
    graph = tmp_path/'graph.json'
    before = json.dumps(records[3],ensure_ascii=False,separators=(',',':')).encode()
    after = json.dumps(original,ensure_ascii=False,separators=(',',':')).encode()
    graph.write_bytes(graph.read_bytes().replace(before, after))
    campaign['current_graph'] = fingerprint(graph)
    entry = deepcopy(entries[3])
    entry['node_sha'] = digest(original)
    entry['byte_offset'] = graph.read_bytes().find(after)
    with sqlite3.connect(tmp_path/'global_index.sqlite') as db:
        db.execute('UPDATE observations SET node_sha=?,byte_offset=? WHERE cid=?',
                   (entry['node_sha'],entry['byte_offset'],entry['claim_id']))
    manifest = tmp_path/'index_acceptance.json'
    manifest.write_text(json.dumps(dict(graph=campaign['current_graph'],database=fingerprint(tmp_path/'global_index.sqlite'))))
    # Remove the synthetic earlier source review for this previously unreviewed
    # identity; production source-review protection is tested above.
    review_path = tmp_path/'completion-source-reviews.json'
    review_path.write_text('{}')
    campaign['current_source_role_reviews'] = fingerprint(review_path)
    prior = VerifiedPaperIdentities(old).resolve(original['metadata'])
    resolved = VerifiedPaperIdentities(new).resolve(original['metadata'])
    completion = dict(claim_id=entry['claim_id'],claim_sha256=entry['node_sha'],pmid='1',
                      previous_source_identity=prior,source_identity=resolved)
    ext = payload['original_relation_extension']
    ext.update(global_index_acceptance=fingerprint(manifest),members=[entry],
               supplemental_identity_registry=fingerprint(added),raw_authority_files=[raw_fp],
               existing_identity_completions=[completion],
               original_identity_registries={entry['claim_id']:fingerprint(previous)})
    payload['source_reviews'] = {entry['claim_id']:{'proposition_support':'supports'}}
    _, result, inputs = extend_base(relations,dossiers,payload,campaign)
    actual = result[-1]['observations'][0]
    assert actual['claim'] == original['metadata']
    assert actual['previous_source_identity'] == prior
    assert actual['source_identity'] == resolved
    assert fingerprint(previous) in inputs
    ext['original_identity_registries'][entry['claim_id']] = fingerprint(added)
    with pytest.raises(EvidenceUnavailable,match='Previous source identity changed'):
        extend_base(relations,dossiers,payload,campaign)
