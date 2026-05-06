"""Tests for the session manager module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.session.manager import SessionManager


def test_session_manager_init(tmp_path):
    """Test SessionManager initialization."""
    env = {"setup_type": "conda", "python_path": "/usr/bin/python3"}
    manager = SessionManager(env, keep_recent=10)
    assert manager.env == env
    assert manager.keep_recent == 10


def test_maybe_compress_no_compression():
    """Test that short history is not compressed."""
    env = {}
    manager = SessionManager(env, keep_recent=10)
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    original_len = len(history)
    manager.maybe_compress(history)
    assert len(history) == original_len


def test_maybe_compress_with_compression():
    """Test compression when history exceeds threshold."""
    env = {}
    manager = SessionManager(env, keep_recent=2)
    history = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Response 3"},
    ]
    manager.maybe_compress(history)
    # Should have: system + summary + 2 recent pairs (4 messages)
    assert len(history) == 6  # system + summary + 2 user + 2 assistant
    # First message should be system
    assert history[0]["role"] == "system"
    # Second message should be the summary
    assert "Context summary" in history[1]["content"]
    # Last 4 messages should be the recent ones
    assert history[2]["content"] == "Message 2"
    assert history[3]["content"] == "Response 2"
    assert history[4]["content"] == "Message 3"
    assert history[5]["content"] == "Response 3"


def test_maybe_compress_preserves_system():
    """Test that system messages are preserved during compression."""
    env = {}
    manager = SessionManager(env, keep_recent=1)
    history = [
        {"role": "system", "content": "System prompt 1"},
        {"role": "system", "content": "System prompt 2"},
        {"role": "user", "content": "Old message"},
        {"role": "assistant", "content": "Old response"},
        {"role": "user", "content": "Recent message"},
        {"role": "assistant", "content": "Recent response"},
    ]
    manager.maybe_compress(history)
    # Both system messages should be preserved
    system_msgs = [m for m in history if m["role"] == "system"]
    assert len(system_msgs) == 2


def test_save_checkpoint(tmp_path):
    """Test saving a checkpoint."""
    env = {
        "setup_type": "conda",
        "python_path": "/usr/bin/python3",
        "conda_env": "neuroclaw",
        "cuda": "11.8",
        "llm_backend": {"provider": "openai", "model": "gpt-4", "api_key_env": "OPENAI_API_KEY"},
    }
    manager = SessionManager(env)
    history = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
    ]
    checkpoint_path = manager.save_checkpoint(history, metadata={"test": True})
    assert checkpoint_path.exists()
    # Verify content
    with checkpoint_path.open() as f:
        data = json.load(f)
    assert "timestamp" in data
    assert data["history"] == history
    assert data["metadata"]["test"] is True
    # Verify api_key_env is not in the snapshot
    assert "api_key_env" not in str(data.get("env_snapshot", {}))


def test_load_latest_checkpoint(tmp_path):
    """Test loading the latest checkpoint."""
    env = {"setup_type": "conda"}
    manager = SessionManager(env)
    # Save two checkpoints
    history1 = [{"role": "user", "content": "First"}]
    history2 = [{"role": "user", "content": "Second"}]
    manager.save_checkpoint(history1)
    manager.save_checkpoint(history2)
    # Load latest
    latest = manager.load_latest_checkpoint()
    assert latest is not None
    assert latest["history"] == history2


def test_load_latest_checkpoint_empty(tmp_path):
    """Test loading when no checkpoints exist."""
    env = {}
    manager = SessionManager(env)
    # Ensure no checkpoints exist
    for f in manager._prune_old_checkpoints.__self__._checkpoint_dir.glob("checkpoint_*.json"):
        f.unlink()
    result = manager.load_latest_checkpoint()
    assert result is None


def test_prune_old_checkpoints(tmp_path):
    """Test that old checkpoints are pruned."""
    env = {}
    manager = SessionManager(env)
    # Save more than MAX_CHECKPOINTS
    for i in range(10):
        history = [{"role": "user", "content": f"Message {i}"}]
        manager.save_checkpoint(history)
    # Check that only MAX_CHECKPOINTS remain
    checkpoints = sorted(manager._checkpoint_dir.glob("checkpoint_*.json"))
    assert len(checkpoints) <= 5  # MAX_CHECKPOINTS = 5


if __name__ == "__main__":
    test_session_manager_init()
    test_maybe_compress_no_compression()
    test_maybe_compress_with_compression()
    test_maybe_compress_preserves_system()
    print("\n=== ALL SESSION MANAGER TESTS PASSED ===")
