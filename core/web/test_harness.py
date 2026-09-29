"""Offline harness contracts; no research, real study data or providers."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from core.cli import main
from core.harness import manifest, register_routes


def test_capabilities_are_shared_and_offline(capsys):
    assert main(["capabilities"]) == 0
    assert json.loads(capsys.readouterr().out) == manifest()
    assert manifest()["autoresearch_modes"] == ["off", "data", "model", "idea", "end-to-end"]
    assert not manifest()["execution_policy"]["navigation_starts_research"]


def test_informational_commands_do_not_import_execution_engine():
    program = "from core.cli import main; import sys; main(['capabilities']); assert 'core.agent.main' not in sys.modules; assert 'core.web.server' not in sys.modules"
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True)


@pytest.mark.parametrize("mode", ["data", "model", "idea", "end-to-end"])
def test_research_command_is_only_a_checklist(mode, capsys):
    assert main(["autoresearch", mode]) == 0
    assert "Please provide" in capsys.readouterr().out


@pytest.mark.parametrize("command,capability", [("graph", "graph"), ("evaluate", "evaluation")])
def test_links_do_not_launch_browser_by_default(command, capability, capsys, monkeypatch):
    opened = []
    monkeypatch.setattr("core.cli.webbrowser.open", opened.append)
    main([command, "--port", "7099"])
    assert f"http://127.0.0.1:7099/harness?capability={capability}" in capsys.readouterr().out
    assert opened == []
    main([command, "--open"])
    assert opened == [f"http://127.0.0.1:7083/harness?capability={capability}"]


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port", "-1"])
def test_invalid_ports_are_rejected(port):
    with pytest.raises(SystemExit) as error:
        main(["gui", "--port", port])
    assert error.value.code == 2


def test_chat_delegates_without_replacing_runtime(tmp_path, monkeypatch):
    calls = []

    class Session:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def start(self):
            calls.append("start")

    monkeypatch.setitem(sys.modules, "core.agent.main", SimpleNamespace(AgentSession=Session))
    main(["chat", "--workspace", str(tmp_path), "--autoresearch", "off"])
    assert calls == [{"workspace": tmp_path.resolve(), "autoresearch_mode": "off"}, "start"]
    with pytest.raises(SystemExit):
        main(["chat", "--workspace", str(tmp_path / "missing")])
    assert len(calls) == 2


def test_gui_defaults_to_separate_loopback_port(monkeypatch, capsys):
    calls = []
    monkeypatch.setitem(sys.modules, "core.web.server", SimpleNamespace(run_server=lambda **kwargs: calls.append(kwargs)))
    main(["gui"])
    assert calls == [{"host": "127.0.0.1", "port": 7083}]
    assert "/harness" in capsys.readouterr().out


@pytest.mark.parametrize("evaluation_enabled", [True, False])
def test_routes_share_manifest_without_initializing_research(evaluation_enabled):
    app = FastAPI()
    register_routes(app, Path(__file__).with_name("static"), evaluation_enabled=evaluation_enabled)
    with TestClient(app) as client:
        response = client.get("/api/harness")
        assert response.json() == manifest(evaluation_enabled=evaluation_enabled)
        ids = [item["id"] for item in response.json()["capabilities"]]
        assert ("evaluation" in ids) is evaluation_enabled
        page = client.get("/harness")
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-store"
        assert '/static/harness.js' in page.text
