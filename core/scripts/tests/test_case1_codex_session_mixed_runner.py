from __future__ import annotations

import json
from pathlib import Path

from core.scripts.case1_codex_session_mixed_runner import (
    _cli_failure_category,
    _controlled_prompt,
    _find_bound_turn,
    _parse_cli_events,
)


THREAD_ID = "01a04766-2364-7e51-b552-5a8560ca933b"
TURN_ID = "01a04766-2415-7303-a96c-139cb1bb442b"
REQUEST_SHA = "a" * 64


def test_controlled_prompt_preserves_request_without_json_escaping() -> None:
    prompt = 'line one\n{"candidate":"value"}'
    value = _controlled_prompt(
        {
            "request_sha256": REQUEST_SHA,
            "request": {
                "prompt": prompt,
                "json_schema": {"type": "object"},
                "force_json": None,
                "max_tokens": 12000,
            },
        }
    )
    assert prompt in value
    assert REQUEST_SHA in value
    assert "不得调用工具" in value


def test_finds_one_completed_turn_bound_to_request(tmp_path: Path) -> None:
    log = tmp_path / f"rollout-{THREAD_ID}.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": THREAD_ID}},
        {"type": "turn_context", "payload": {"turn_id": TURN_ID}},
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": TURN_ID,
                "item": {"type": "UserMessage", "content": REQUEST_SHA},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": TURN_ID,
                "item": {
                    "type": "AgentMessage",
                    "phase": "final_answer",
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
        },
        {
            "type": "event_msg",
            "ordinal": 9,
            "payload": {"type": "task_complete", "turn_id": TURN_ID},
        },
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    observed_log, observed_turn = _find_bound_turn(
        log_root=tmp_path,
        thread_id=THREAD_ID,
        request_sha256=REQUEST_SHA,
    )
    assert observed_log == log
    assert observed_turn == TURN_ID


def test_finds_latest_completed_retry_for_request(tmp_path: Path) -> None:
    first_turn = TURN_ID
    second_turn = "01a04777-290a-7f61-82b4-0e88f84c3bcd"
    log = tmp_path / f"rollout-{THREAD_ID}.jsonl"
    rows = [{"type": "session_meta", "payload": {"id": THREAD_ID}}]
    for ordinal, turn_id in ((10, first_turn), (20, second_turn)):
        rows.extend(
            [
                {"type": "turn_context", "payload": {"turn_id": turn_id}},
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "item_completed",
                        "turn_id": turn_id,
                        "item": {"type": "UserMessage", "content": REQUEST_SHA},
                    },
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "item_completed",
                        "turn_id": turn_id,
                        "item": {
                            "type": "AgentMessage",
                            "phase": "final_answer",
                            "content": [{"type": "text", "text": "ok"}],
                        },
                    },
                },
                {
                    "type": "event_msg",
                    "ordinal": ordinal,
                    "payload": {"type": "task_complete", "turn_id": turn_id},
                },
            ]
        )
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _, observed_turn = _find_bound_turn(
        log_root=tmp_path,
        thread_id=THREAD_ID,
        request_sha256=REQUEST_SHA,
    )
    assert observed_turn == second_turn


def test_cli_event_summary_requires_completed_turn() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": THREAD_ID}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "turn.completed"}),
        ]
    )
    result = _parse_cli_events(stdout, THREAD_ID)
    assert result["event_type_counts"]["turn.completed"] == 1
    assert result["thread_id"] == THREAD_ID


def test_classifies_active_writer_as_retryable() -> None:
    assert _cli_failure_category("thread-store conflict: already has an active writer") == (
        "thread_writer_conflict"
    )
