import pytest

from core.goal_runtime import GoalStore, goal_marker


def test_markers_are_explicit_and_fenced_examples_are_ignored():
    assert goal_marker("Here is an example: <goal:complete>") is None
    assert goal_marker("```\n<goal:complete>\n```") is None
    assert goal_marker("    <goal:complete>") is None
    assert goal_marker("<goal:complete>\nI still need to check the result.") is None
    assert goal_marker("Work finished.\n<goal:complete>\n") == "complete"
    assert goal_marker("\n<goal:blocked>\n") == "blocked"


def test_goal_turns_are_durable_bounded_and_require_verification(tmp_path):
    store = GoalStore(tmp_path / "runtime.db")
    goal = store.create("chat", tmp_path, "Write a report", "A saved report exists", max_turns=2, goal_id="goal_test_1")
    with pytest.raises(ValueError, match="unfinished goal"):
        store.create("chat", tmp_path, "Another report")

    first = store.bind_request(goal["goal_id"], "req_first")
    assert first["turns_started"] == 1
    after_first = store.record_turn(goal["goal_id"], "req_first", {"content": "I made progress"}, "completed")
    assert after_first["status"] == "active"

    second = store.bind_request(goal["goal_id"], "req_second")
    assert second["turns_started"] == 2
    verifying = store.record_turn(goal["goal_id"], "req_second", {"content": "Done\n<goal:complete>"}, "completed")
    assert verifying["status"] == "verifying"
    rejected = store.mark_verification(goal["goal_id"], False, "The report was not shown", request_id="req_second")
    assert rejected["status"] == "paused"  # the bounded budget is exhausted
    assert rejected["verification"]["accepted"] is False
    with pytest.raises(ValueError, match="awaiting verification"):
        store.mark_verification(goal["goal_id"], True, "late", request_id="req_second")


def test_pause_resume_cancel_and_stale_requests_fail_closed(tmp_path):
    store = GoalStore(tmp_path / "runtime.db")
    goal = store.create("chat", tmp_path, "Inspect a file", max_turns=3, goal_id="goal_test_2")
    bound = store.bind_request(goal["goal_id"], "req_pause")
    paused = store.control(goal["goal_id"], "chat", "pause")
    assert paused["status"] == "paused" and paused["active_request_id"] is None
    # A late result from a user pause cannot restart the goal.
    late = store.record_turn(goal["goal_id"], bound["active_request_id"], {"content": "late"}, "cancelled")
    assert late["status"] == "paused"
    resumed = store.control(goal["goal_id"], "chat", "resume")
    assert resumed["status"] == "active"
    with pytest.raises(ValueError, match="request ownership"):
        store.record_turn(goal["goal_id"], "wrong_request", {"content": "bad"}, "completed")
    cancelled = store.control(goal["goal_id"], "chat", "cancel")
    assert cancelled["status"] == "cancelled"
    with pytest.raises(ValueError, match="terminal"):
        store.control(goal["goal_id"], "chat", "resume")


def test_restart_pauses_goal_and_cancels_undispatched_goal_turn(tmp_path):
    store = GoalStore(tmp_path / "runtime.db")
    goal = store.create("chat", tmp_path, "Inspect", max_turns=2, goal_id="goal_restart")
    bound = store.bind_request(goal["goal_id"], "req_restart")
    store.store.enqueue({
        "request_id": bound["active_request_id"], "chat_id": "chat", "workspace_path": str(tmp_path),
        "message": "goal", "goal_id": goal["goal_id"],
    })
    store.pause_on_restart()
    saved = store.get(goal["goal_id"])
    queued = store.store.get(bound["active_request_id"])
    assert saved["status"] == "paused" and saved["active_request_id"] is None
    assert queued["status"] == "cancelled"


def test_unqueued_turn_does_not_consume_budget_or_allow_implicit_resume(tmp_path):
    store = GoalStore(tmp_path / "runtime.db")
    goal = store.create("chat", tmp_path, "Inspect", max_turns=1, goal_id="goal_abort")
    store.bind_request(goal["goal_id"], "req_unqueued")
    paused = store.abort_unqueued_request(goal["goal_id"], "req_unqueued", "Queue rejected the request")
    assert paused["status"] == "paused"
    assert paused["turns_started"] == 0
    assert paused["active_request_id"] is None
    with pytest.raises(ValueError, match="not ready"):
        store.bind_request(goal["goal_id"], "req_implicit")
    store.control(goal["goal_id"], "chat", "resume")
    store.bind_request(goal["goal_id"], "req_queued")
    store.store.enqueue({
        "request_id": "req_queued", "chat_id": "chat", "workspace_path": str(tmp_path),
        "message": "goal", "goal_id": goal["goal_id"],
    })
    with pytest.raises(ValueError, match="Queued goal requests"):
        store.abort_unqueued_request(goal["goal_id"], "req_queued", "Too late")
