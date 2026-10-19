"""Portable install integration against the existing accepted-evidence fixture."""
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import io
import json
from pathlib import Path
import shutil
import time
import urllib.error
import zipfile

from fastapi.testclient import TestClient
import pytest

from core.graph_installation import (
    GraphPackage, InstalledClaimLayer, configured_campaign_path, export_package,
    installed_package, install_archive, pack_archive, selected_graph_path,
)
from core.web.claim_layer_v8 import AcceptedClaimLayer
from neurooracle.src.kg_storage import storage_path
from neurooracle.tests.test_claim_evidence_query import fingerprint, setup


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROCLAW_GRAPH_INSTALLATION', raising=False)
    monkeypatch.delenv('NEUROCLAW_KG_CAMPAIGN', raising=False)
    source = tmp_path / 'source'
    source.mkdir()
    campaign, _, _ = setup(source, node_fields=dict(preferred_name='Synthetic claim', domain_tags=['claim']))
    expected = AcceptedClaimLayer(campaign).query(claim_id='CLM:0')
    package = tmp_path / 'export'
    export_package(campaign, package, release_id='test-accepted-1')
    archive = tmp_path / 'graph.zip'
    release = pack_archive(package / 'manifest.json', archive,
                           remote_path='neurooracle/data/releases/test/graph.zip')
    return source, campaign, expected, package, archive, release


def test_installed_graph_evidence_and_research_survive_copy_and_source_removal(prepared, tmp_path, monkeypatch):
    source, campaign, expected, package, archive, release = prepared
    originals = {p.name: p.read_bytes() for p in source.iterdir()}
    repo = tmp_path / 'first location'
    data = repo / 'neurooracle/data'
    install_archive(archive, data, expected_sha256=release['archive']['sha256'])
    # Both source and the first installation disappear from their original
    # paths. Receipt paths still contain those original locations.
    gc.collect()  # The existing SQLite fixture uses transaction context managers.
    source.rename(tmp_path / 'unavailable-source')
    moved = tmp_path / '另一个安装目录'
    shutil.copytree(repo, moved, copy_function=shutil.copyfile)
    repo.rename(tmp_path / 'unavailable-installation')
    monkeypatch.setenv('NEUROCLAW_GRAPH_INSTALLATION', str(moved / 'neurooracle/data/graph-installation.json'))
    monkeypatch.setenv('NEUROCLAW_KG_CAMPAIGN', str(tmp_path / 'stale-development-pointer.json'))
    selected = installed_package(moved)
    assert selected_graph_path(moved) == selected.graph
    assert configured_campaign_path(moved) == selected.campaign
    service = InstalledClaimLayer(moved)
    assert service.query(claim_id='CLM:0') == expected
    assert service.query_batch(queries=[{'claim_id': 'CLM:0'}])['results'][0] == expected
    assert service.topic_search(topic='exposure')['claim_returned'] > 0
    from core.topic_evidence import default_layer
    assert default_layer().query(claim_id='CLM:0') == expected
    for row in selected.manifest['files']:
        if Path(row['source']['path']).name in originals:
            assert selected.resolve(row['source']['path']).read_bytes() == originals[Path(row['source']['path']).name]
    assert storage_path(campaign) == campaign.resolve()  # context did not leak


def test_installations_do_not_leak_storage_bindings_across_threads(prepared, tmp_path):
    _, campaign, expected, _, archive, _ = prepared
    repos = [tmp_path / 'a', tmp_path / 'b']
    for root in repos:
        install_archive(archive, root / 'neurooracle/data')
    services = [InstalledClaimLayer(root) for root in repos]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda service: service.query(claim_id='CLM:0'), services))
    assert outputs == [expected, expected]
    assert storage_path(campaign) == campaign.resolve()


def test_tampered_dependency_is_not_replaced_with_available_development_data(prepared, tmp_path):
    _, _, _, _, archive, _ = prepared
    root = tmp_path / 'installed'
    install_archive(archive, root / 'neurooracle/data')
    package = installed_package(root)
    service = InstalledClaimLayer(root)
    assert service.status()['available']
    source = next(row['source']['path'] for row in package.manifest['files'] if row['source']['path'].endswith('dossiers.jsonl'))
    target = package.resolve(source)
    raw = target.read_bytes()
    target.write_bytes(raw.replace(b'Own result', b'Bad result'))
    with pytest.raises(ValueError, match='hash|size'):
        service.query(claim_id='CLM:0')


@pytest.mark.parametrize('member', ['../outside.json', '/outside.json', 'C:/outside.json', 'folder/../../outside.json'])
def test_archive_paths_cannot_escape_or_replace_an_existing_installation(prepared, tmp_path, member):
    _, _, expected, _, archive, _ = prepared
    data = tmp_path / 'repo/neurooracle/data'
    install_archive(archive, data)
    pointer = (data / 'graph-installation.json').read_bytes()
    bad = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(bad, 'w') as zipped:
        zipped.writestr(member, '{}')
    with pytest.raises(ValueError, match='inside'):
        install_archive(bad, data)
    assert (data / 'graph-installation.json').read_bytes() == pointer
    assert InstalledClaimLayer(tmp_path / 'repo').query(claim_id='CLM:0') == expected


def test_invalid_hash_or_incomplete_bundle_keeps_the_previous_pointer(prepared, tmp_path):
    _, _, _, _, archive, _ = prepared
    data = tmp_path / 'repo/neurooracle/data'
    install_archive(archive, data)
    pointer = (data / 'graph-installation.json').read_bytes()
    with pytest.raises(ValueError, match='hash'):
        install_archive(archive, data, expected_sha256='0' * 64)
    bad = tmp_path / 'incomplete.zip'
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad, 'w') as output:
        for name in source.namelist():
            if not name.endswith('dossiers.jsonl'):
                output.writestr(name, source.read(name))
    with pytest.raises((ValueError, OSError)):
        install_archive(bad, data)
    assert (data / 'graph-installation.json').read_bytes() == pointer


def test_existing_service_rebinds_after_a_verified_install(prepared, tmp_path):
    _, _, expected, package, archive, _ = prepared
    root = tmp_path / 'repo'
    data = root / 'neurooracle/data'
    service = InstalledClaimLayer(root)
    assert service.status() == {'configured': False, 'available': False}
    install_archive(archive, data)
    first_path = service.path
    assert service.query(claim_id='CLM:0') == expected
    value = json.loads((package / 'manifest.json').read_text())
    value['release_id'] = 'test-accepted-2'
    (package / 'manifest.json').write_text(json.dumps(value), encoding='utf-8')
    second = tmp_path / 'second.zip'
    pack_archive(package / 'manifest.json', second, remote_path='neurooracle/data/releases/test2/graph.zip')
    install_archive(second, data)
    assert service.path != first_path
    assert service.query(claim_id='CLM:0') == expected
    assert installed_package(root).manifest['release_id'] == 'test-accepted-2'


def test_manifest_cannot_point_to_a_different_graph_revision(prepared):
    _, _, _, package, _, _ = prepared
    value = json.loads((package / 'manifest.json').read_text())
    value['graph'] = next(row['source']['path'] for row in value['files'] if row['source']['path'].endswith('terms.json'))
    (package / 'manifest.json').write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='different revisions'):
        GraphPackage(package / 'manifest.json').validate()


def test_http_concept_status_and_evidence_use_the_same_installed_revision(prepared, tmp_path, monkeypatch):
    from core.web import server
    _, _, expected, _, archive, _ = prepared
    root = tmp_path / 'repo'
    install_archive(archive, root / 'neurooracle/data')
    monkeypatch.setenv('NEUROCLAW_GRAPH_INSTALLATION', str(root / 'neurooracle/data/graph-installation.json'))
    monkeypatch.setenv('NEURODISCOVERY_WORKBENCH_DB', str(tmp_path / 'workbench.sqlite3'))
    with TestClient(server.create_app()) as client:
        status = client.get('/api/neurooracle/graph/status').json()
        assert status['available'] and status['evidence_configured']
        assert status['path'] == str(installed_package(root).graph)
        assert status['graph_revision'] == expected['graph_revision']
        loaded = client.post('/api/kg/load')
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()['n_claims'] == 4
        claims = client.get('/api/kg/shared-claims')
        assert claims.status_code == 200
        evidence = client.get('/api/kg/claim-evidence?claim_id=CLM:0')
        assert evidence.status_code == 200
        assert evidence.json() == expected
        assert claims.json()['graph_revision'] == expected['graph_revision']


def test_missing_evidence_has_an_installation_error_code(tmp_path, monkeypatch):
    from core.web import server
    monkeypatch.delenv('NEUROCLAW_GRAPH_INSTALLATION', raising=False)
    monkeypatch.setenv('NEUROCLAW_KG_CAMPAIGN', str(tmp_path / 'uninstalled-campaign.json'))
    monkeypatch.setenv('NEURODISCOVERY_WORKBENCH_DB', str(tmp_path / 'workbench.sqlite3'))
    with TestClient(server.create_app()) as client:
        response = client.get('/api/kg/shared-claims')
        assert response.status_code == 503
        assert response.json()['code'] == 'evidence_not_installed'


@pytest.mark.parametrize('mode', ['complete', 'bad_hash', 'legacy_only'])
def test_download_installs_one_package_and_bad_download_preserves_old_data(prepared, tmp_path, monkeypatch, mode):
    from core.web import server
    _, _, expected, _, archive, release = prepared
    root = tmp_path / 'desktop'
    data = root / 'neurooracle/data'
    if mode == 'bad_hash':
        install_archive(archive, data)
        pointer = (data / 'graph-installation.json').read_bytes()
        release['archive']['sha256'] = '0' * 64
    actual_root, importer = server.REPO_ROOT, server._import_from_path
    monkeypatch.setattr(server, 'REPO_ROOT', root)
    monkeypatch.setattr(server, '_import_from_path', lambda name, path: importer(name, actual_root / path.relative_to(root)))
    monkeypatch.setenv('NEURODISCOVERY_WORKBENCH_DB', str(tmp_path / 'download-workbench.sqlite3'))
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
                if mode == 'legacy_only':
                    raise urllib.error.HTTPError(url, 404, 'Not found', {}, None)
                return Response(json.dumps(release).encode())
            if mode == 'legacy_only' and url.endswith('/full_v2_20260906_bbccd816de32/manifest.json'):
                raise urllib.error.HTTPError(url, 404, 'Not found', {}, None)
            if mode == 'legacy_only' and url.endswith('/neurooracle/data/knowledge_graph.json.gz'):
                return Response(gzip.compress(b'{"metadata":{},"concepts":{},"edges":[]}'))
            if url.endswith('/' + release['archive']['path']):
                return Response(archive.read_bytes())
            raise AssertionError('Unexpected network request: ' + url)
    monkeypatch.setattr(server.urllib.request, 'build_opener', lambda *args: Opener())
    with TestClient(server.create_app()) as client:
        answer = client.post('/api/neurooracle/graph/download', json={'force': True})
        assert answer.status_code == 200
        for _ in range(100):
            state = client.get('/api/neurooracle/graph/status').json()
            if not state['download']['running']:
                break
            time.sleep(.03)
        assert not state['download']['running']
        if mode == 'bad_hash':
            assert not state['download']['completed']
            assert 'hash' in state['download']['error']
            assert (data / 'graph-installation.json').read_bytes() == pointer
        elif mode == 'complete':
            assert state['download']['completed'] and state['evidence_configured'], state
            assert state['graph_revision'] == expected['graph_revision']
            assert client.get('/api/neurooracle/graph/update-status').json()['update_available'] is False
        else:
            assert state['download']['completed'] and state['available']
            assert state['evidence_install_required'] and not state['evidence_configured']
            response = client.get('/api/kg/shared-claims')
            assert response.status_code == 503 and response.json()['code'] == 'evidence_not_installed'
        if mode != 'legacy_only':
            assert client.get('/api/kg/claim-evidence?claim_id=CLM:0').json() == expected
            assert not (data / 'knowledge_graph.json.gz').exists()
    assert len(calls) >= 2
