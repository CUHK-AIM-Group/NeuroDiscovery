"""API integration tests with an in-memory model stub, never a live provider."""
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.web import server
from neurooracle.tests.test_client_novelty_policy import candidate


@pytest.fixture
def client(monkeypatch):
    from core.agent import main

    class FakeSession:
        calls = []

        def __init__(self, **kwargs):
            self.env = {"llm_backend": {"provider": "fixture", "model": "offline-test"}}
            self.history = []
            self._llm = None
            self.autoresearch_mode = kwargs.get("autoresearch_mode", "off")

        def configure_autoresearch(self, mode):
            self.autoresearch_mode = mode

        def request_cancel(self):
            pass

        def set_llm_client(self, llm):
            self._llm = llm

        def _chat(self):
            self.calls.append(deepcopy(self.history))
            return "Synthetic reply; no model was called."

    monkeypatch.setattr(main, "AgentSession", FakeSession)
    monkeypatch.setattr(main, "build_llm_client", lambda env: None)
    monkeypatch.setattr(server, "_workspace_change_snapshot", lambda path: {})
    monkeypatch.setattr(server, "_workspace_change_summary", lambda *args: [])
    with TestClient(server.create_app()) as test_client:
        yield test_client, FakeSession.calls


@pytest.mark.parametrize("mode", ["strict", "novelty_first", "weighted"])
def test_http_transmits_mode_to_runtime_and_response(client, mode):
    api, calls = client
    response = api.post("/api/chat", json={"message": "Select reviewed hypotheses", "novelty_mode": mode})
    assert response.status_code == 200
    assert response.json()["novelty_mode"] == mode
    assert f"novelty_mode={mode}" in calls[-1][0]["content"]


def test_http_validation_happens_before_model_use(client):
    api, calls = client
    assert api.post("/api/chat", json={"message": "test", "novelty_mode": "bogus"}).status_code == 400
    assert not calls
    response = api.post("/api/chat", json={"message": "/help idea"})
    assert response.json()["novelty_mode"] == "strict"
    assert not calls


def test_websocket_validates_and_replaces_preferences(client):
    api, calls = client
    with api.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["type"] == "init"
        ws.send_json({"message": "test", "novelty_mode": "bogus"})
        assert ws.receive_json()["type"] == "error"
        assert not calls
        for mode in ("weighted", "strict", "novelty_first"):
            ws.send_json({"message": "Synthetic selection request", "novelty_mode": mode})
            response = ws.receive_json()
            assert response["type"] == "done"
            assert response["novelty_mode"] == mode
            assert f"novelty_mode={mode}" in calls[-1][0]["content"]
            assert calls[-1][0]["content"].count("[Hypothesis selection preference:") == 1


def test_selection_endpoint_recomputes_evidence_gates(client):
    api, calls = client
    known = candidate("known", "exact_prior")
    known.update(novel_candidate_gate_passed=True, novelty_priority_points=1)
    for mode, expected in (("strict", []), ("novelty_first", ["known"]), ("weighted", ["known"])):
        response = api.post("/api/hypotheses/select", json={"candidates": [known], "novelty_mode": mode})
        assert response.status_code == 200
        assert response.json()["selected_ids"] == expected
        assert response.json()["candidates"][0]["literature_classification"] == "exact_prior"
    assert api.post("/api/hypotheses/select", json={"candidates": [{}]}).status_code == 400
    assert not calls


def test_policy_catalog_matches_client_and_both_submit_paths(client):
    api, _ = client
    catalog = api.get("/api/hypotheses/selection-policy").json()
    assert catalog["modes"] == ["strict", "novelty_first", "weighted"]
    assert catalog["default_mode"] == "strict"
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    for mode in catalog["modes"]:
        assert f'name="novelty-mode" value="{mode}"' in html
    assert html.count("novelty_mode: noveltyMode") == 2  # send + resend
    assert "session.noveltyMode = mode" in html
    assert "novelty_mode: pendingUserMessage?.noveltyMode || null" in html
    assert 'aria-haspopup="dialog"' in html
