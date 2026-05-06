"""Tests for ShadowCheckpointManager."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from .manager import ShadowCheckpointManager


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def manager(tmp_path: Path) -> ShadowCheckpointManager:
    """Create a ShadowCheckpointManager with a temporary base directory."""
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()
    return ShadowCheckpointManager(repo_root, max_checkpoints=5)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── checkpoint() ─────────────────────────────────────────────────────────────


class TestCheckpoint:
    def test_checkpoint_creates_snapshot(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        _write_file(workspace / "a.txt", "hello")
        result = manager.checkpoint(workspace, label="first")
        assert "commit" in result
        assert len(result["commit"]) == 40  # SHA-1 hex
        assert result["files_changed"] >= 1
        assert result["label"] == "first"

    def test_checkpoint_skips_when_no_changes(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        _write_file(workspace / "a.txt", "hello")
        manager.checkpoint(workspace)
        manager.begin_turn()
        result = manager.checkpoint(workspace)
        assert result["skipped"] is True

    def test_checkpoint_dedup_within_turn(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        manager.checkpoint(workspace)
        _write_file(workspace / "a.txt", "v2")
        result = manager.checkpoint(workspace)
        assert result["skipped"] is True
        assert result["reason"] == "dedup"

    def test_checkpoint_dedup_resets_after_begin_turn(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        manager.checkpoint(workspace)
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        result = manager.checkpoint(workspace)
        assert "commit" in result


# ── list_checkpoints() ───────────────────────────────────────────────────────


class TestListCheckpoints:
    def test_list_empty(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        assert manager.list_checkpoints(workspace) == []

    def test_list_returns_chronological_order(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        manager.checkpoint(workspace, label="first")
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        manager.checkpoint(workspace, label="second")

        cps = manager.list_checkpoints(workspace)
        assert len(cps) == 2
        assert cps[0]["label"] == "first"
        assert cps[1]["label"] == "second"


# ── diff_checkpoint() ────────────────────────────────────────────────────────


class TestDiffCheckpoint:
    def test_diff_shows_changes(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        r1 = manager.checkpoint(workspace)
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        manager.checkpoint(workspace)

        diff = manager.diff_checkpoint(workspace, r1["commit"])
        assert "a.txt" in diff["files"]
        assert "v2" in diff["diff_text"]

    def test_diff_single_file(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        _write_file(workspace / "b.txt", "keep")
        r1 = manager.checkpoint(workspace)
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        manager.checkpoint(workspace)

        diff = manager.diff_checkpoint_file(workspace, r1["commit"], "a.txt")
        assert "v2" in diff["diff_text"]


# ── restore_checkpoint() ─────────────────────────────────────────────────────


class TestRestoreCheckpoint:
    def test_restore_entire_workspace(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        r1 = manager.checkpoint(workspace)
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        manager.checkpoint(workspace)

        manager.restore_checkpoint(workspace, r1["commit"])
        assert _read_file(workspace / "a.txt") == "v1"

    def test_restore_single_file(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        _write_file(workspace / "b.txt", "keep")
        r1 = manager.checkpoint(workspace)
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        _write_file(workspace / "b.txt", "changed")
        manager.checkpoint(workspace)

        manager.restore_checkpoint(workspace, r1["commit"], filepath="a.txt")
        assert _read_file(workspace / "a.txt") == "v1"
        assert _read_file(workspace / "b.txt") == "changed"

    def test_restore_creates_pre_rollback_snapshot(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        r1 = manager.checkpoint(workspace)
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v2")
        manager.checkpoint(workspace)

        manager.restore_checkpoint(workspace, r1["commit"])
        cps = manager.list_checkpoints(workspace)
        assert any("pre-rollback" in cp["message"] for cp in cps)

    def test_restore_invalid_hash_raises(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        manager.checkpoint(workspace)
        with pytest.raises(ValueError, match="Invalid commit hash"):
            manager.restore_checkpoint(workspace, "not_a_valid_hash")


# ── get_files_at_checkpoint() ────────────────────────────────────────────────


class TestGetFilesAtCheckpoint:
    def test_lists_files(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "v1")
        _write_file(workspace / "sub" / "b.txt", "v2")
        r = manager.checkpoint(workspace)

        files = manager.get_files_at_checkpoint(workspace, r["commit"])
        assert "a.txt" in files
        # Git always uses forward slashes, even on Windows
        assert "sub/b.txt" in files


# ── _prune() ─────────────────────────────────────────────────────────────────


class TestPrune:
    def test_prune_removes_old_checkpoints(
        self, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "repo_root"
        repo_root.mkdir()
        mgr = ShadowCheckpointManager(repo_root, max_checkpoints=3)
        ws = tmp_path / "workspace"
        ws.mkdir()

        for i in range(5):
            mgr.begin_turn()
            _write_file(ws / "f.txt", f"version {i}")
            mgr.checkpoint(ws, label=f"cp-{i}")

        cps = mgr.list_checkpoints(ws)
        assert len(cps) <= 3
        # The most recent checkpoints should survive
        labels = [cp["label"] for cp in cps]
        assert "cp-4" in labels

    def test_prune_noop_when_under_limit(
        self, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "repo_root"
        repo_root.mkdir()
        mgr = ShadowCheckpointManager(repo_root, max_checkpoints=10)
        ws = tmp_path / "workspace"
        ws.mkdir()

        for i in range(3):
            mgr.begin_turn()
            _write_file(ws / "f.txt", f"v{i}")
            mgr.checkpoint(ws)

        cps = mgr.list_checkpoints(ws)
        assert len(cps) == 3


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_list_on_nonexistent_workspace(
        self, manager: ShadowCheckpointManager, tmp_path: Path
    ) -> None:
        fake_ws = tmp_path / "does_not_exist"
        assert manager.list_checkpoints(fake_ws) == []

    def test_checkpoint_creates_shadow_dir(
        self, manager: ShadowCheckpointManager, workspace: Path
    ) -> None:
        manager.begin_turn()
        _write_file(workspace / "a.txt", "data")
        manager.checkpoint(workspace)
        assert (manager._shadow_git_dir(workspace) / ".git" / "HEAD").exists()
