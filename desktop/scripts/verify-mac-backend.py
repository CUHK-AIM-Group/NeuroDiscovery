"""Offline smoke checks against an isolated copy of a staged distribution."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from unittest.mock import patch


@contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def verify(backend, variant):
    backend = Path(backend).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="neurodiscovery-backend-check-") as temporary:
        isolated = Path(temporary) / "backend"
        shutil.copytree(backend, isolated)
        sys.path[:] = [str(isolated)] + [entry for entry in sys.path
            if entry and ("site-packages" in entry or Path(entry).resolve().is_relative_to(Path(sys.base_prefix).resolve()))]
        from fastapi.testclient import TestClient
        original_connect = socket.socket.connect

        def local_connect(connection, address):
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return original_connect(connection, address)
            raise RuntimeError("External network forbidden in backend smoke check")

        with working_directory(isolated), patch.object(Path, "home", return_value=Path(temporary)), \
                patch.object(socket.socket, "connect", local_connect), \
                patch.dict(os.environ, {"NEUROORACLE_STUDY_PASSWORD": "", "NEURODISCOVERY_STUDY_PASSWORD": "",
                                       "NEURODISCOVERY_STUDY_ROOT": str(Path(temporary) / "ranking")}):
            if variant == "evaluation":
                from core.web import evaluation_app as module
                app = module.create_app(Path(temporary) / "answers")
            else:
                from core.web import server as module
                app = module.create_app()
            if not Path(module.__file__).resolve().is_relative_to(isolated):
                raise RuntimeError("Imported checkout code instead of staged backend")
            with TestClient(app, base_url="http://127.0.0.1") as client:
                def expect(route, status=200):
                    response = client.get(route)
                    if response.status_code != status:
                        raise RuntimeError(f"{variant} {route}: {response.status_code}, expected {status}: {response.text[:300]}")
                    return response

                expect("/")
                health = expect("/api/health").json()
                if variant == "demo":
                    assert expect("/api/distribution/demo").json()["human_evaluation"] is False
                    for route in ("/study", "/discovery-study", "/api/studies/config",
                                  "/api/studies/discovery/config", "/static/study.html",
                                  "/static/discovery-study.js", "/static/evaluation-home.html"):
                        expect(route, 404)
                    assert not any("/api/studies" in route.path for route in app.routes)
                    assert not (isolated / "neurooracle/src/user_study.py").exists()
                else:
                    for route in ("/study", "/discovery-study", "/api/studies/config",
                                  "/api/studies/discovery/config", "/static/evaluation-export.js"):
                        expect(route)
                    config = expect("/api/studies/config").json()
                    assert "case1_tcp_expert_study_v3.json" in json.dumps(config)
                if variant == "evaluation":
                    assert health["mode"] == "human-evaluation-only"
                    assert config["evaluation_only"] is True
                    for route in ("/api/chat", "/api/models", "/api/graph/status", "/api/studies/results",
                                  "/static/index.html", "/openapi.json"):
                        expect(route, 404)
                else:
                    response = client.post("/api/chat", json={"message": "/help", "language": "English"})
                    assert response.status_code == 200, response.text
                    assert response.json()["model_used"] == "local help"
            print(json.dumps({"variant": variant, "backend_smoke": "passed", "external_network": "forbidden"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--variant", choices=("full", "evaluation", "demo"), required=True)
    arguments = parser.parse_args()
    verify(arguments.backend, arguments.variant)
