"""Client/runtime wiring and cancellation tests, without reading keys or calling models."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

from core.agent import main
from core.web import server


@pytest.fixture
def api(monkeypatch):
    instances = []

    class StubSession:
        def __init__(self, **kwargs):
            self.workspace = kwargs.get("workspace")
            self.autoresearch_mode = kwargs.get("autoresearch_mode", "off")
            self.env = {"llm_backend": {"provider": "openai", "model": "synthetic"}}
            self.history = []
            self._llm = None
            self.started = threading.Event()
            self.cancelled = threading.Event()
            self.autoresearch_state = None
            instances.append(self)

        def set_llm_client(self, client):
            self._llm = client

        def configure_autoresearch(self, mode):
            self.autoresearch_mode = mode
            self.cancelled.clear()

        def request_cancel(self):
            self.cancelled.set()

        def _chat(self):
            self.started.set()
            if self.history[-1]["content"].startswith("wait for cancellation"):
                assert self.cancelled.wait(5), "Cancellation did not reach the backend worker"
            self.autoresearch_state = {"mode": self.autoresearch_mode,
                                       "status": "cancelled" if self.cancelled.is_set() else "completed"}
            return "Synthetic runtime response"

    monkeypatch.setattr(main, "AgentSession", StubSession)
    monkeypatch.setattr(main, "build_llm_client", lambda env: object())
    monkeypatch.setattr(server, "_workspace_change_snapshot", lambda path: {})
    monkeypatch.setattr(server, "_workspace_change_summary", lambda *args: [])
    with TestClient(server.create_app()) as client:
        yield client, instances


@pytest.mark.parametrize("mode", ["data", "model", "idea", "end-to-end", "off"])
def test_http_passes_selected_scope_to_execution_loop(api, mode):
    client, sessions = api
    response = client.post("/api/chat", json={"message": "Synthetic task", "autoresearch_mode": mode})
    assert response.status_code == 200
    assert sessions[-1].autoresearch_mode == mode
    assert response.json()["autoresearch"] == {"mode": mode, "status": "completed"}


def test_help_never_launches_even_with_autoresearch_on(api):
    client, sessions = api
    response = client.post("/api/chat", json={"message": "/help idea", "autoresearch_mode": "idea"})
    assert response.status_code == 200 and response.json()["model_used"] == "local help"
    assert not sessions


def test_stop_targets_only_its_request_and_keeps_handle_until_worker_finishes(api):
    client, sessions = api
    request_id = "test_request_123456789"
    with ThreadPoolExecutor(max_workers=1) as pool:
        response = pool.submit(client.post, "/api/chat", json={"message": "wait for cancellation",
                                                             "autoresearch_mode": "data", "request_id": request_id})
        # Condition-based synchronization, bounded independently from tool/experiment timings.
        import time
        deadline = time.monotonic() + 3
        while not sessions and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sessions and sessions[0].started.wait(3)
        assert client.post("/api/chat/cancel", json={"request_id": "another_request_123"}).json() == {"cancelled": False}
        assert not sessions[0].cancelled.is_set()
        duplicate = client.post("/api/chat", json={"message": "duplicate", "request_id": request_id})
        assert duplicate.status_code == 409
        assert client.post("/api/chat/cancel", json={"request_id": request_id}).json() == {"cancelled": True}
        result = response.result(timeout=3)
        assert result.status_code == 200
        assert result.json()["autoresearch"]["status"] == "cancelled"
    assert client.post("/api/chat/cancel", json={"request_id": request_id}).json() == {"cancelled": False}


def test_request_id_validation_is_not_chat_wide_cancellation(api):
    client, sessions = api
    assert client.post("/api/chat", json={"message": "test", "request_id": "chat1"}).status_code == 400
    assert not sessions


def test_websocket_autoresearch_uses_tools_not_text_only_streaming(api, monkeypatch, tmp_path):
    client, sessions = api
    async def forbidden_stream(*args, **kwargs):
        pytest.fail("AutoResearch must use the tool-enabled execution loop")
    monkeypatch.setattr(server, "_stream_openai", forbidden_stream)
    with client.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["type"] == "init"
        for mode in ("data", "idea"):
            ws.send_json({"message": "Synthetic task", "autoresearch_mode": mode,
                          "workspace_path": str(tmp_path), "client_surface": "desktop"})
            response = ws.receive_json()
            assert response["type"] == "done"
            assert response["autoresearch"]["mode"] == mode
            assert sessions[-1].autoresearch_mode == mode
            assert sessions[-1].workspace == tmp_path
            assert "Never inspect" in sessions[-1].history[0]["content"]


def test_websocket_cancel_reaches_running_worker(api):
    client, sessions = api
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"message": "wait for cancellation", "autoresearch_mode": "model"})
        assert sessions[-1].started.wait(3)
        ws.send_json({"type": "cancel"})
        response = ws.receive_json()
        assert response["autoresearch"]["status"] == "cancelled"


def test_websocket_disconnect_cancels_running_worker(api):
    client, sessions = api
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_json()
        ws.send_json({"message": "wait for cancellation", "autoresearch_mode": "model"})
        assert sessions[-1].started.wait(3)
    assert sessions[-1].cancelled.wait(3)


def test_client_send_resend_and_stop_share_request_identity_and_export_status():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    assert html.count("request_id: sessionRequestFor(session.id).requestId") == 2
    assert html.count("autoresearch_mode: sessionRequestFor(session.id).researchMode") == 2
    assert html.count("autoresearch: data.autoresearch || null") == 2
    assert "autoresearch: message.autoresearch || null" in html
    stop = html.split("function stopCurrentRequest()", 1)[1].split("function resendPreviousUserMessage", 1)[0]
    assert "cancelSessionRequest(state.activeSessionId)" in stop
    assert "fetch('/api/chat/cancel'" in html
