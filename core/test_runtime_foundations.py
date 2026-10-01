from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from core.instructions import load_instructions, scoped_instructions
from core.session.context import compact_context
from core.autoresearch_runtime import AutoResearchRun


def test_instructions_are_workspace_scoped_and_source_bound(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (tmp_path / "AGENTS.md").write_text("Not selected", encoding="utf-8")
    for filename in ("SOUL.md", "AGENTS.md", "USER.md", "MEMORY.md"):
        (workspace / filename).write_text(filename + "\nScoped content", encoding="utf-8")
    prompt, sources = load_instructions(workspace)
    assert prompt.startswith("SOUL.md\nScoped content")
    assert "Not selected" not in prompt
    assert len(sources) == 4
    assert "not authorization" in prompt
    assert sources[1]["sha256"] == hashlib.sha256((workspace / "AGENTS.md").read_bytes()).hexdigest()


def test_instructions_reject_oversize_instead_of_silently_truncating(tmp_path):
    (tmp_path / "AGENTS.md").write_text("a" * 256001)
    with pytest.raises(ValueError, match="256 KB"):
        load_instructions(tmp_path)


def test_nested_instructions_apply_only_to_target_subtree(tmp_path):
    nested = tmp_path / "src" / "module"
    nested.mkdir(parents=True)
    (tmp_path / "src" / "AGENTS.md").write_text("Parent scope")
    (nested / "AGENTS.md").write_text("Child scope")
    text = scoped_instructions(tmp_path, nested / "new.py")
    assert text.index("Parent scope") < text.index("Child scope")
    assert scoped_instructions(tmp_path, tmp_path / "other" / "file.py") == ""


def test_context_archive_is_exact_and_retains_system_and_recent_turn(tmp_path):
    messages = [{"role": "system", "content": "Keep constraints"},
                {"role": "user", "content": "Old request"},
                {"role": "assistant", "content": "actual evidence " * 10000},
                {"role": "user", "content": "Latest request"}]
    original = deepcopy(messages)
    report = compact_context(messages, tmp_path, 2048)
    assert report["status"] == "compacted"
    archive = tmp_path / ".neurodiscovery" / "context" / (report["sha256"] + ".json")
    assert json.loads(archive.read_text(encoding="utf-8")) == original[1:3]
    assert messages[0] == original[0]
    assert messages[-1] == original[-1]
    assert report["after"] < report["before"]
    assert "incomplete excerpts" in messages[1]["content"]


def test_tool_transaction_is_never_split(tmp_path):
    messages = [{"role": "system", "content": "Policy"}, {"role": "user", "content": "Task"}]
    for number in range(5):
        messages.extend([{"role": "assistant", "tool_calls": [{"id": str(number), "function": {"name": "read"}}]},
                         {"role": "tool", "tool_call_id": str(number), "content": "x" * 14000}])
    latest = deepcopy(messages[-4:])
    report = compact_context(messages, tmp_path, 2048)
    assert report["status"] == "compacted"
    assert messages[1] == {"role": "user", "content": "Task"}
    assert messages[-4:] == latest
    pending = set()
    for message in messages:
        for call in message.get("tool_calls", []):
            pending.add(call["id"])
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
    assert not pending


def test_pending_tool_with_injected_user_does_not_become_compaction_boundary(tmp_path):
    messages = [{"role": "user", "content": "x" * 20000},
                {"role": "assistant", "tool_calls": [{"id": "pending"}]},
                {"role": "user", "content": "Do not drop pending tool"}]
    original = deepcopy(messages)
    assert compact_context(messages, tmp_path, 2048)["status"] == "pressure"
    assert messages == original


def test_checkpoint_resume_preserves_parent_and_remaining_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "8")
    previous = AutoResearchRun(tmp_path, "data")
    previous.state.update(status="interrupted", iterations=3, objective="Original objective", conversation_scope="chat")
    previous.save()
    frozen = previous.path.read_bytes()
    current = AutoResearchRun(tmp_path, "data")
    current.state["conversation_scope"] = "chat"
    current.inherit_checkpoint(previous.path.parent.name, "Fix failed output")
    assert current.limit == 5
    assert current.state["objective"] == "Original objective"
    assert not current.state["evidence"]
    assert previous.path.read_bytes() == frozen
    current.state["conversation_scope"] = "other"
    with pytest.raises(ValueError, match="another conversation"):
        current.inherit_checkpoint(previous.path.parent.name, "Wrong chat")


def test_checkpoint_budget_cannot_be_reset_by_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROCLAW_MAX_TOOL_ITERATIONS", "1")
    previous = AutoResearchRun(tmp_path, "data")
    previous.begin_iteration()
    previous.halt("budget_exhausted", "Budget")
    current = AutoResearchRun(tmp_path, "data")
    with pytest.raises(ValueError, match="budget is exhausted"):
        current.inherit_checkpoint(previous.path.parent.name, "Continue")


def test_save_survives_a_transient_replace_share_violation(tmp_path, monkeypatch):
    """A momentary Windows lock on run.json must not abort a research run."""
    run = AutoResearchRun(tmp_path, "idea")
    real_replace = Path.replace
    calls = {"count": 0}

    def flaky_replace(self, target):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError(13, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    run.state["iterations"] = 7
    run.save()
    assert calls["count"] == 2
    assert json.loads(run.path.read_text(encoding="utf-8"))["iterations"] == 7
    # The temporary file is renamed, not left behind as a stale partial write.
    assert not run.path.with_suffix(".tmp").exists()


def test_save_reraises_when_the_target_stays_locked(tmp_path, monkeypatch):
    """A real, persistent failure must still surface after the bounded retries."""
    run = AutoResearchRun(tmp_path, "idea")

    def always_locked(self, target):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_locked)
    with pytest.raises(PermissionError):
        run.save()
