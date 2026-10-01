"""Artifact preview tests use temporary files/state; never traverse research data."""
from pathlib import Path
import pytest
from core.web.artifacts import resolve_artifact, TEXT_LIMIT


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "report.md").write_text("# Research report\n", encoding="utf-8")
    state = {"sessions": [{"id": "s", "projectId": "p"}], "projects": [{"id": "p", "workspacePath": str(root)}]}
    return root, state


def test_resolve_saved_conversation_workspace(fixture):
    root, state = fixture
    target, media = resolve_artifact(state, "s", "report.md", root.parent)
    assert target == root / "report.md"
    assert media.startswith("text/plain")
    assert resolve_artifact(state, "s", str(target), root.parent)[0] == target
    with pytest.raises(FileNotFoundError):
        resolve_artifact(state, "other", str(target), root)


@pytest.mark.parametrize("reference", ["../outside.txt", ".env", "//server/file.txt", "\\\\server\\file.txt", "report.md:secret", "file:///tmp/a.txt", "report.md\x00"])
def test_reject_paths_outside_artifacts(fixture, reference):
    root, state = fixture
    (root.parent / "outside.txt").write_text("outside")
    (root / ".env").write_text("test fixture only")
    with pytest.raises((PermissionError, FileNotFoundError)):
        resolve_artifact(state, "s", reference, root)


def test_symlink_and_size_limits(fixture):
    root, state = fixture
    large = root / "large.csv"
    large.write_bytes(b"a" * (TEXT_LIMIT + 1))
    with pytest.raises(OverflowError):
        resolve_artifact(state, "s", large.name, root)
    assert resolve_artifact(state, "s", large.name, root, download=True)[0] == large
    outside = root.parent / "outside.txt"
    outside.write_text("outside")
    link = root / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not grant symlink creation")
    with pytest.raises(PermissionError):
        resolve_artifact(state, "s", link.name, root)


def test_http_preview_is_read_only_and_download_is_safe(fixture):
    from fastapi.testclient import TestClient
    from core.web.server import create_app
    from core.web.workbench import WorkbenchStore
    root, state = fixture
    store = WorkbenchStore()
    store.save_state(0, state)
    original = (root / "report.md").read_bytes()
    with TestClient(create_app()) as client:
        result = client.get("/api/workbench/artifact", params={"chat_id": "s", "path": "report.md"})
        assert result.status_code == 200
        assert result.content == original
        assert result.headers["content-disposition"].startswith("attachment")
        assert "sandbox" in result.headers["content-security-policy"]
        metadata = client.head("/api/workbench/artifact", params={"chat_id": "s", "path": "report.md"})
        assert metadata.status_code == 200 and metadata.content == b""
        assert int(metadata.headers["content-length"]) == len(original)
        assert client.get("/api/workbench/artifact", params={"chat_id": "unknown", "path": "report.md"}).status_code == 404
        outside = root.parent / "outside.txt"
        outside.write_text("outside")
        assert client.get("/api/workbench/artifact", params={"chat_id": "s", "path": str(outside)}).status_code == 403
        assert client.head("/api/workbench/artifact", params={"chat_id": "s", "path": str(outside)}).status_code == 403
        large = root / "large.txt"
        large.write_bytes(b"x" * (TEXT_LIMIT + 1))
        assert client.get("/api/workbench/artifact", params={"chat_id": "s", "path": large.name}).status_code == 413
        metadata = client.head("/api/workbench/artifact", params={"chat_id": "s", "path": large.name})
        assert metadata.status_code == 200 and metadata.content == b""
        assert int(metadata.headers["content-length"]) == TEXT_LIMIT + 1
        assert client.get("/api/workbench/artifact", params={"chat_id": "s", "path": large.name, "download": "true"}).status_code == 200
    assert store.state()["revision"] == 1
    assert (root / "report.md").read_bytes() == original
