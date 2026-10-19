"""Formal release selection and original-evidence integrity, without model calls."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import urllib.error

from fastapi.testclient import TestClient
import pytest

from core.formal_graph_evidence import FormalGraphEvidence
from core.graph_installation import (GraphPackage, InstalledClaimLayer, configured_campaign_path,
                                     export_formal_package, formal_source, install_archive,
                                     install_formal_snapshot, pack_archive, selected_graph_path)
from core.web.claim_evidence import EvidenceUnavailable


def write_formal(root, *, padding='', ensure_ascii=False):
    directory = root / 'neurooracle/data/full_v2'
    directory.mkdir(parents=True)
    records = {}
    for key, name, tag in [('N:brain', '海马体', 'neuroanatomy'), ('N:memory', 'Memory', 'cognitive_function')]:
        records[key] = dict(id=key, preferred_name=name, domain_tags=[tag], metadata={})
    for number, pmid in enumerate(['12345', '12345', '67890']):
        cid = f'CLM:{number}'
        md = dict(id=cid, subject_id='N:brain', subject_name='Hippocampus 海马体',
                  predicate='predicts', object_id='N:memory', object_name='Memory', negated=number == 1,
                  source_paper=dict(pmid=pmid, title=f'海马体 memory {number}', year=2025),
                  evidence=dict(sample_size=50, effect_size=0.42, study_type='review'),
                  raw_text=f'原始提取文本 {number}\u2029 memory.',
                  metadata=dict(subject_type='brain_region', object_type='cognitive_function', population='Adults'),
                  scope_reaudit=dict(decision='finalized', gates={'neural_disease_change_verified': True}))
        records[cid] = dict(id=cid, preferred_name='Brain predicts memory', domain_tags=['claim'], metadata=md)
    graph = directory / 'knowledge_graph.json'
    graph.write_text(json.dumps(dict(metadata=dict(padding=padding), concepts=records, edges=[]),
                               ensure_ascii=ensure_ascii), encoding='utf-8')
    raw = graph.read_bytes()
    state = dict(status='canonical_current', canonical_files=dict(knowledge_graph=dict(
                 path='C:/old-workstation/full_v2/knowledge_graph.json', bytes=len(raw),
                 sha256=hashlib.sha256(raw).hexdigest())), formal_kg_statistics=dict(general=dict(claims=3, papers=2)))
    path = directory / 'CURRENT_STATE.json'
    path.write_text(json.dumps(state), encoding='utf-8')
    return path, graph, records


@pytest.fixture
def formal(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROCLAW_GRAPH_INSTALLATION', raising=False)
    monkeypatch.delenv('NEUROCLAW_KG_CAMPAIGN', raising=False)
    return write_formal(tmp_path)


def test_formal_release_overrides_obsolete_selector_and_demo(formal, tmp_path):
    state, graph, records = formal
    config = tmp_path / 'neurooracle/configs/graph_explorer.json'
    config.parent.mkdir()
    config.write_text(json.dumps(dict(version=1, campaign='obsolete/CAMPAIGN.json')))
    (graph.parent.parent / 'knowledge_graph.json').write_text('{}')
    assert selected_graph_path(tmp_path) == graph
    from neurooracle.src.graph_paths import current_graph_path
    assert current_graph_path(tmp_path) == graph
    assert configured_campaign_path(tmp_path) is None
    assert formal_source(tmp_path) == (state, graph)
    service = InstalledClaimLayer(tmp_path)
    result = service.query(claim_id='CLM:0')
    observation = result['papers'][0]['observations'][0]
    assert observation['original_claim'] == records['CLM:0']['metadata']
    assert observation['raw_text'] == records['CLM:0']['metadata']['raw_text']
    assert result['source'] == 'formal_current_graph'
    assert result['reviewed_supporting_article_count'] == 0
    assert observation['source_review'] is None
    assert result['papers'][0]['verified'] is False
    assert service.search()['total'] == 3
    assert service.search(minimum_papers=2)['total'] == 0
    topic = service.topic_search(topic='hippocampus memory', limit=3, evidence_claims=3)
    assert topic['graph_revision'] == result['graph_revision']
    assert topic['claim_returned'] == 2  # one representative per declared source
    assert all(not s['supports_reviewed_proposition'] for s in topic['studies'])


@pytest.mark.parametrize('ensure_ascii', [True, False])
def test_offsets_survive_unicode_and_stream_buffer_compaction(tmp_path, ensure_ascii):
    state, graph, records = write_formal(tmp_path, padding='测试\u2029' * 400000, ensure_ascii=ensure_ascii)
    service = FormalGraphEvidence(state, cache_directory=tmp_path / 'cache')
    result = service.query_batch(queries=[{'claim_id': f'CLM:{i}'} for i in range(3)])
    for i, document in enumerate(result['results']):
        assert document['papers'][0]['observations'][0]['original_claim'] == records[f'CLM:{i}']['metadata']
    assert service.graph_nodes(['N:brain', 'absent']) == {'N:brain': records['N:brain']}
    assert len(service.chain_index(predicates=['predicts'])['records']) == 3
    assert service.pair_support_counts([('N:brain', 'N:memory')]) == {('N:brain', 'N:memory'): 2}


def test_bad_formal_source_never_falls_back_to_demo(formal, tmp_path):
    state, graph, _ = formal
    service = FormalGraphEvidence(state, cache_directory=tmp_path / 'cache')
    service.search()
    graph.write_bytes(graph.read_bytes().replace(b'Memory', b'Memorz'))
    with pytest.raises(EvidenceUnavailable, match='hash'):
        service.query(claim_id='CLM:0')
    graph.write_text('{}')
    (graph.parent.parent / 'knowledge_graph.json').write_text('{}')
    with pytest.raises(ValueError, match='differs'):
        selected_graph_path(tmp_path)


def test_keywords_do_not_match_random_internal_id_fragments(formal, tmp_path):
    state, graph, records = formal
    key = 'N:hash_12cd44abcdef'
    records[key] = dict(records.pop('N:memory'), id=key)
    for cid in ('CLM:0', 'CLM:1', 'CLM:2'):
        records[cid]['metadata']['object_id'] = key
    records['CLM:2']['metadata']['subject_name'] = 'CD44'
    graph.write_text(json.dumps(dict(metadata={}, concepts=records, edges=[]), ensure_ascii=False), encoding='utf-8')
    data = json.loads(state.read_text())
    data['canonical_files']['knowledge_graph'].update(bytes=graph.stat().st_size,
        sha256=hashlib.sha256(graph.read_bytes()).hexdigest())
    state.write_text(json.dumps(data))
    service = FormalGraphEvidence(state, cache_directory=tmp_path / 'cache')
    assert service.search(query='CD44')['total'] == 1
    assert service.search(query=key)['total'] == 3
    assert service.search(query='CLM:0')['total'] == 1


def test_cache_corruption_and_bad_count_are_rejected(formal, tmp_path):
    state, graph, _ = formal
    service = FormalGraphEvidence(state, cache_directory=tmp_path / 'cache')
    service.search()
    cache = service._index_path(service._read_current())
    with cache.open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(EvidenceUnavailable, match='cache hash'):
        service.search()
    data = json.loads(state.read_text())
    data['formal_kg_statistics']['general']['claims'] = 4
    state.write_text(json.dumps(data))
    with pytest.raises(EvidenceUnavailable, match='claim count'):
        FormalGraphEvidence(state, cache_directory=tmp_path / 'other-cache').search()


def test_formal_package_survives_moving_and_removing_original(formal, tmp_path, monkeypatch):
    state, graph, records = formal
    original_state, original_graph = state.read_bytes(), graph.read_bytes()
    export = tmp_path / 'package'
    export_formal_package(state, export, release_id='formal-test')
    archive = tmp_path / 'formal.zip'
    release = pack_archive(export / 'manifest.json', archive,
                           remote_path='neurooracle/data/releases/test/formal.zip')
    destination = tmp_path / '中文安装目录/neurooracle/data'
    installed = install_archive(archive, destination, expected_sha256=release['archive']['sha256'])
    shutil.rmtree(graph.parent)
    root = destination.parents[1]
    package = GraphPackage(destination / 'graph-packages' / installed['package_sha256'] / 'manifest.json')
    assert package.state.read_bytes() == original_state
    assert package.graph.read_bytes() == original_graph
    assert package.campaign is None
    monkeypatch.setenv('NEUROCLAW_GRAPH_INSTALLATION', str(destination / 'graph-installation.json'))
    monkeypatch.setenv('NEUROCLAW_KG_CAMPAIGN', 'obsolete-development.json')
    assert selected_graph_path(root) == package.graph
    from neurooracle.src.graph_paths import current_graph_path, resolve_graph_path
    assert current_graph_path(root) == package.graph
    with pytest.raises(FileNotFoundError):
        resolve_graph_path(graph)  # missing frozen inputs never become the latest graph
    result = InstalledClaimLayer(root).query(claim_id='CLM:2')
    assert result['papers'][0]['observations'][0]['original_claim'] == records['CLM:2']['metadata']
    from core.topic_evidence import default_layer
    assert default_layer().topic_search(topic='hippocampus')['claim_returned'] > 0


def test_http_concept_evidence_and_research_select_same_formal_release(formal, tmp_path, monkeypatch):
    state, graph, _ = formal
    from core.web import server
    actual_root, importer = server.REPO_ROOT, server._import_from_path
    monkeypatch.setattr(server, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(server, '_import_from_path', lambda name, path: importer(name, actual_root / path.relative_to(tmp_path)))
    monkeypatch.setenv('NEURODISCOVERY_WORKBENCH_DB', str(tmp_path / 'workbench.sqlite'))
    app = server.create_app()
    with TestClient(app) as client:
        graph_status = client.get('/api/neurooracle/graph/status').json()
        evidence_status = client.get('/api/kg/evidence-status').json()
        claim = client.get('/api/kg/claim-evidence', params=dict(claim_id='CLM:0'))
        search = client.get('/api/kg/shared-claims')
        topic = client.get('/api/kg/topic-evidence', params=dict(topic='hippocampus memory'))
        assert claim.status_code == search.status_code == topic.status_code == 200
        assert graph_status['evidence_configured'] and not graph_status['evidence_install_required']
        assert evidence_status['source'] == 'formal_current_graph'
        assert graph_status['path'] == str(graph)
        assert graph_status['graph_revision'] == evidence_status['graph_revision'] == claim.json()['graph_revision']
        assert search.json()['total'] == 3
        assert topic.json()['claim_returned'] > 0
        assert client.get('/api/kg/shared-claims', params=dict(minimum_papers=2)).json()['total'] == 0
        loaded = client.post('/api/kg/load')
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()['n_claims'] == 3


@pytest.mark.parametrize('mode', ['complete', 'compressed_hash', 'uncompressed_hash', 'claim_count'])
def test_public_formal_download_is_atomic_and_does_not_download_demo(formal, tmp_path, monkeypatch, mode):
    import zstandard
    from core.web import server
    state, graph, _ = formal
    state_data = json.loads(state.read_text())
    compressed = zstandard.ZstdCompressor().compress(graph.read_bytes())
    compressed_path = tmp_path / 'source.zst'
    compressed_path.write_bytes(compressed)
    publication = dict(schema_version='neurodiscovery.graph-publication.v1', source_status='canonical_current',
                       release_id='formal-test', source_state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
                       graph=dict(file='knowledge_graph.json.zst', compression='zstandard', bytes=len(compressed),
                                  sha256=hashlib.sha256(compressed).hexdigest(), uncompressed_bytes=graph.stat().st_size,
                                  uncompressed_sha256=state_data['canonical_files']['knowledge_graph']['sha256']),
                       counts=dict(claims=3, papers=2))
    original_publication = json.dumps(publication, ensure_ascii=False, indent=2).encode()
    root = tmp_path / 'desktop'
    data = root / 'neurooracle/data'
    if mode != 'complete':
        install_formal_snapshot(original_publication, compressed_path, data)
        pointer = (data / 'graph-installation.json').read_bytes()
        if mode == 'claim_count':
            publication['counts']['claims'] = 4
        else:
            publication['graph']['sha256' if mode == 'compressed_hash' else 'uncompressed_sha256'] = '0' * 64
    published_bytes = json.dumps(publication, ensure_ascii=False, indent=2).encode()
    actual_root, importer = server.REPO_ROOT, server._import_from_path
    monkeypatch.setattr(server, 'REPO_ROOT', root)
    monkeypatch.setattr(server, '_import_from_path', lambda name, path: importer(name, actual_root / path.relative_to(root)))
    monkeypatch.setenv('NEURODISCOVERY_WORKBENCH_DB', str(tmp_path / 'download-workbench.sqlite'))
    calls = []

    class Response(io.BytesIO):
        status = 200

        def __init__(self, payload):
            super().__init__(payload)
            self.headers = {'Content-Length': str(len(payload))}

    class Opener:
        def open(self, request, timeout):
            url = request.full_url
            calls.append(url)
            if url.endswith('/desktop/latest.json'):
                raise urllib.error.HTTPError(url, 404, 'not found', {}, None)
            if url.endswith('/full_v2_20260906_bbccd816de32/manifest.json'):
                return Response(published_bytes)
            if url.endswith('/knowledge_graph.json.zst'):
                return Response(compressed)
            raise AssertionError('Unexpected request: ' + url)

    monkeypatch.setattr(server.urllib.request, 'build_opener', lambda *args: Opener())
    with TestClient(server.create_app()) as client:
        assert client.post('/api/neurooracle/graph/download', json={'force': True}).status_code == 200
        for _ in range(100):
            status = client.get('/api/neurooracle/graph/status').json()
            if not status['download']['running']:
                break
            time.sleep(.03)
        assert not status['download']['running'], status
        if mode == 'complete':
            assert status['download']['completed'] and status['evidence_configured'], status
            assert client.get('/api/kg/shared-claims').json()['total'] == 3
            assert client.get('/api/kg/claim-evidence', params={'claim_id': 'CLM:0'}).status_code == 200
            assert client.get('/api/neurooracle/graph/update-status').json()['update_available'] is False
            from core.graph_installation import installed_package
            assert installed_package(root).state.read_bytes() == original_publication
        else:
            assert not status['download']['completed']
            assert (data / 'graph-installation.json').read_bytes() == pointer
    assert not any(url.endswith('.gz') for url in calls)
