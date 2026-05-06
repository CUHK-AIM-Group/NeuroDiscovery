"""Unit tests for SubagentManager."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSubagentHandle:
    """Tests for SubagentHandle dataclass."""

    def test_default_values(self):
        from core.subagent.manager import SubagentHandle

        handle = SubagentHandle(
            session_id="test123",
            session=None,
            thread=None,
            mode="run_and_return",
            status="pending",
        )
        assert handle.session_id == "test123"
        assert handle.status == "pending"
        assert handle.mode == "run_and_return"
        assert handle.persona == ""
        assert handle.task == ""
        assert isinstance(handle.result_queue, queue.Queue)
        assert isinstance(handle.input_queue, queue.Queue)


class TestSubagentResult:
    """Tests for SubagentResult dataclass."""

    def test_basic_result(self):
        from core.subagent.manager import SubagentResult

        result = SubagentResult(
            session_id="abc",
            status="completed",
            response="hello",
            history=[{"role": "user", "content": "hi"}],
        )
        assert result.session_id == "abc"
        assert result.status == "completed"
        assert result.response == "hello"
        assert result.error == ""


class TestSubagentManager:
    """Tests for SubagentManager."""

    @patch("core.subagent.manager.SubagentManager._execute_subagent")
    def test_spawn_returns_session_id(self, mock_execute):
        from core.subagent.manager import SubagentManager

        mock_execute.side_effect = lambda handle, *args: (
            handle.result_queue.put(
                MagicMock(status="completed", response="done")
            )
        )

        manager = SubagentManager(env={}, workspace=Path("/tmp"))
        sid = manager.spawn("test task")
        assert isinstance(sid, str)
        assert len(sid) == 12

    @patch("core.subagent.manager.SubagentManager._execute_subagent")
    def test_spawn_fire_and_forget(self, mock_execute):
        from core.subagent.manager import SubagentManager

        mock_execute.side_effect = lambda handle, *args: (
            handle.result_queue.put(
                MagicMock(status="completed", response="done")
            )
        )

        manager = SubagentManager(env={}, workspace=Path("/tmp"))
        sid = manager.spawn("test task", mode="fire_and_forget")
        assert isinstance(sid, str)
        # fire_and_forget should return immediately
        active = manager.list_active()
        assert len(active) == 1
        assert active[0]["mode"] == "fire_and_forget"

    @patch("core.subagent.manager.SubagentManager._execute_subagent")
    def test_list_active(self, mock_execute):
        from core.subagent.manager import SubagentManager

        mock_execute.side_effect = lambda handle, *args: time.sleep(10)

        manager = SubagentManager(env={}, workspace=Path("/tmp"))
        manager.spawn("task 1")
        manager.spawn("task 2", persona="biostatistician")

        active = manager.list_active()
        assert len(active) == 2
        tasks = {a["task"] for a in active}
        assert "task 1" in tasks
        assert "task 2" in tasks

    @patch("core.subagent.manager.SubagentManager._execute_subagent")
    def test_cancel(self, mock_execute):
        from core.subagent.manager import SubagentManager

        mock_execute.side_effect = lambda handle, *args: time.sleep(10)

        manager = SubagentManager(env={}, workspace=Path("/tmp"))
        sid = manager.spawn("long task")
        manager.cancel(sid)

        result = manager.get_result(sid, timeout=1.0)
        assert result.status == "cancelled"

    @patch("core.subagent.manager.SubagentManager._execute_subagent")
    def test_shutdown_all(self, mock_execute):
        from core.subagent.manager import SubagentManager

        mock_execute.side_effect = lambda handle, *args: time.sleep(10)

        manager = SubagentManager(env={}, workspace=Path("/tmp"))
        manager.spawn("task 1")
        manager.spawn("task 2")
        manager.shutdown_all()

        active = manager.list_active()
        # All should be cancelled
        for a in active:
            assert a["status"] == "cancelled"

    def test_get_result_not_found(self):
        from core.subagent.manager import SubagentManager

        manager = SubagentManager(env={}, workspace=Path("/tmp"))
        result = manager.get_result("nonexistent", timeout=0.1)
        assert result.status == "not_found"

    def test_build_persona_prefix(self):
        from core.subagent.manager import SubagentManager

        prefix = SubagentManager._build_persona_prefix("biostatistician")
        assert "biostatistician" in prefix.lower()

        prefix = SubagentManager._build_persona_prefix("custom_expert")
        assert "custom_expert" in prefix

    @patch("core.subagent.manager.SubagentManager._execute_subagent")
    def test_concurrent_subagents(self, mock_execute):
        from core.subagent.manager import SubagentManager

        def slow_execute(handle, *args):
            time.sleep(0.1)
            handle.result_queue.put(
                MagicMock(status="completed", response=f"done-{handle.session_id}")
            )

        mock_execute.side_effect = slow_execute

        manager = SubagentManager(env={}, workspace=Path("/tmp"), max_concurrent=3)
        sids = [manager.spawn(f"task {i}") for i in range(3)]

        results = [manager.get_result(sid, timeout=5.0) for sid in sids]
        assert all(r.status == "completed" for r in results)
