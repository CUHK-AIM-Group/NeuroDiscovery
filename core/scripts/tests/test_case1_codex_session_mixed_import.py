from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.scripts.case1_codex_session_mixed_import import (
    _request_cache_key,
    import_answer,
)


THREAD_ID = "01a04745-3f75-7ec0-a67d-323a0274c2a5"
TURN_ID = "01a04756-f8bc-7a33-a927-dbee5e5f2365"


def _fixture(
    tmp_path: Path,
    *,
    model: str = "gpt-5.6-luna",
    session_thread_id: str = THREAD_ID,
) -> tuple[Path, Path, Path, Path]:
    trial = tmp_path / "trial"
    pending_dir = trial / "codex_session_shadow" / "pending"
    cache_dir = trial / "open_coscientist_trial_cache"
    pending_dir.mkdir(parents=True)
    cache_dir.mkdir()
    request = {
        "prompt": "Return an answer object.",
        "prompt_sha256": "",
        "model": "openai/deepseek-v4-pro",
        "temperature": 0.7,
        "max_tokens": 12000,
        "tools": None,
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
        "force_json": None,
    }
    import hashlib

    request["prompt_sha256"] = hashlib.sha256(request["prompt"].encode()).hexdigest()
    request_sha = _request_cache_key(request)
    pending = {
        "schema_version": "case1-codex-session-shadow-request.v1",
        "status": "pending",
        "scientific_classification": "mixed_backend_user_approved_pending_import",
        "formal_cache_written": False,
        "request_sha256": request_sha,
        "cache_identity_sha256": "identity-test",
        "requested_backend": {
            "model": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "thread_id": THREAD_ID,
        },
        "request": request,
    }
    pending_path = pending_dir / f"{request_sha}.json"
    pending_path.write_text(json.dumps(pending), encoding="utf-8")
    session_log = tmp_path / "session.jsonl"
    records = [
        {"type": "session_meta", "payload": {"id": session_thread_id}},
        {
            "type": "turn_context",
            "payload": {
                "turn_id": TURN_ID,
                "model": model,
                "effort": "max",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": TURN_ID,
                "item": {
                    "type": "UserMessage",
                    "id": "user-1",
                    "content": f"process {request_sha}",
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "turn_id": TURN_ID,
                "item": {
                    "type": "AgentMessage",
                    "id": "message-1",
                    "phase": "final_answer",
                    "content": [
                        {"type": "text", "text": '{"answer":"ok"}'}
                    ],
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": TURN_ID},
        },
    ]
    session_log.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    return pending_path, session_log, cache_dir, trial


def test_imports_validated_luna_answer_and_writes_receipt(tmp_path: Path) -> None:
    pending, log, cache, trial = _fixture(tmp_path)
    result = import_answer(
        pending_path=pending,
        session_log=log,
        turn_id=TURN_ID,
        cache_dir=cache,
        trial_dir=trial,
    )
    cache_payload = json.loads((cache / f"{pending.stem}.json").read_text())
    assert cache_payload["response"] == {"answer": "ok"}
    assert result["cache_count_before"] == 0
    assert result["cache_count_after"] == 1
    assert result["prompts_or_responses_printed"] is False
    receipt = json.loads(
        (trial / "codex_session_shadow" / "imported" / pending.name).read_text()
    )
    assert receipt["backend"]["model"] == "gpt-5.6-luna"
    assert receipt["validation"]["passed"] is True


def test_rejects_turn_from_another_model(tmp_path: Path) -> None:
    pending, log, cache, trial = _fixture(tmp_path, model="gpt-5.6-sol")
    with pytest.raises(RuntimeError, match="model does not match"):
        import_answer(
            pending_path=pending,
            session_log=log,
            turn_id=TURN_ID,
            cache_dir=cache,
            trial_dir=trial,
        )
    assert not list(cache.glob("*.json"))


def test_allows_audited_luna_thread_override(tmp_path: Path) -> None:
    alternate = "01a04766-2364-7e51-b552-5a8560ca933b"
    pending, log, cache, trial = _fixture(
        tmp_path, session_thread_id=alternate
    )
    result = import_answer(
        pending_path=pending,
        session_log=log,
        turn_id=TURN_ID,
        cache_dir=cache,
        trial_dir=trial,
        backend_thread_id_override=alternate,
    )
    receipt = json.loads(
        (trial / "codex_session_shadow" / "imported" / pending.name).read_text()
    )
    assert receipt["backend"]["thread_id"] == alternate
    assert receipt["capture_requested_thread_id"] == THREAD_ID
    assert result["cache_count_after"] == 1
