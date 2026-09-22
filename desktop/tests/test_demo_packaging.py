import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from core.web import server


ROOT = Path(__file__).resolve().parents[2]


def test_demo_routes_are_absent_and_normal_features_work():
    original = Path.is_file

    def is_file(path):
        return path.name == 'DEMO_DISTRIBUTION.json' or original(path)

    with patch.object(Path, 'is_file', is_file):
        app = server.create_app()
    client = TestClient(app)
    assert client.get('/api/distribution/demo').json() == {
        'distribution': 'demo', 'human_evaluation': False,
    }
    for route in ('/study', '/discovery-study', '/api/studies',
                  '/api/studies/config', '/api/studies/discovery/config',
                  '/static/study.html', '/static/discovery-study.js',
                  '/materials/private.json'):
        assert client.get(route).status_code == 404, route
    assert client.post('/api/studies/auth', json={}).status_code == 404
    assert client.post('/api/studies/sessions', json={}).status_code == 404
    assert not any('/api/studies' in route.path for route in app.routes)
    assert client.get('/').status_code == 200
    assert client.get('/api/health').status_code == 200
    response = client.post('/api/chat', json={'message': '/help', 'language': 'English'})
    assert response.status_code == 200
    assert response.json()['model_used'] == 'local help'


def test_normal_build_keeps_evaluation():
    app = server.create_app()
    assert any(route.path == '/api/studies/auth' for route in app.routes)
    assert any(route.path == '/discovery-study' for route in app.routes)
    assert TestClient(app).get('/api/distribution/demo').status_code == 404


def test_demo_package_is_separate_and_has_no_evaluation_entrypoint():
    config = json.loads((ROOT / 'desktop/electron-builder.demo.json').read_text())
    assert config['extraMetadata']['distribution'] == 'demo'
    assert config['directories']['output'] == 'dist-demo'
    assert config['extraResources'][0]['from'] == 'runtime-demo'
    assert 'evaluation-main.js' not in config['files']
    script = (ROOT / 'desktop/scripts/prepare-bundled-runtime.ps1').read_text()
    for name in ('study.html', 'study-workspace.*', 'discovery-study*',
                 'discovery_study.py', 'user_study.py'):
        assert name in script
    assert 'const HUMAN_EVALUATION_ENABLED = false;' in script
