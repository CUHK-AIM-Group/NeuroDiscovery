"""Run captured Case Study 1 requests through a Luna Max Codex session.

The runner resumes one persistent Codex backend thread, supplies each exact
captured prompt over stdin (so the model does not need a file-reading tool),
finds the completed turn in the local session log, and delegates validation
and cache writing to ``case1_codex_session_mixed_import``.  Prompts and model
responses are never printed by this process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from core.scripts.case1_codex_session_mixed_import import import_answer


RUNNER_SCHEMA = "case1-codex-session-mixed-runner-checkpoint.v1"
MODEL = "gpt-5.6-luna"
EFFORT = "max"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _controlled_prompt(pending: dict[str, Any]) -> str:
    request = pending["request"]
    prompt = request["prompt"]
    metadata = {
        "request_sha256": pending["request_sha256"],
        "json_schema": request.get("json_schema"),
        "force_json": request.get("force_json"),
        "max_tokens": request.get("max_tokens"),
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return (
        "你是 Case Study 1 的受控 LLM 后端。模型必须保持 "
        "gpt-5.6-luna、max reasoning。下面是一个独立 API 请求。只执行 "
        "BEGIN_REQUEST_PROMPT 与 END_REQUEST_PROMPT 之间的请求正文，并严格遵守 "
        "REQUEST_METADATA_JSON 中的 json_schema、force_json 与 max_tokens。不得生成 "
        "schema 未声明的字段；当 additionalProperties=false 时尤其不得添加任何"
        "额外字段，提交前自行核对键名与必需字段。不得访问或"
        "推断实验结果、ground truth、effect size、p/FDR、NeuroDiscovery score 或外部"
        "验证结果；不得调用工具、修改文件或混入其他请求。最终回答就是该 API 请求的 "
        "message.content：只返回最终正文，不要分析、说明、Markdown 围栏或请求哈希。\n\n"
        f"REQUEST_METADATA_JSON\n{metadata_text}\n\n"
        f"BEGIN_REQUEST_PROMPT\n{prompt}\nEND_REQUEST_PROMPT"
    )


def _session_logs(log_root: Path, thread_id: str) -> list[Path]:
    return sorted(
        log_root.rglob(f"*{thread_id}*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _find_bound_turn(
    *,
    log_root: Path,
    thread_id: str,
    request_sha256: str,
) -> tuple[Path, str]:
    matches: list[tuple[float, Path, str]] = []
    for log_path in _session_logs(log_root, thread_id):
        session_id = ""
        contexts: set[str] = set()
        user_bound: set[str] = set()
        final_turns: set[str] = set()
        completed_turns: dict[str, float] = {}
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                payload = record.get("payload") or {}
                record_type = record.get("type")
                if record_type == "session_meta":
                    session_id = str(
                        payload.get("id") or payload.get("session_id") or ""
                    )
                if record_type == "turn_context":
                    turn_id = str(payload.get("turn_id") or "")
                    if turn_id:
                        contexts.add(turn_id)
                if record_type != "event_msg":
                    continue
                turn_id = str(payload.get("turn_id") or "")
                if not turn_id:
                    continue
                if payload.get("type") == "task_complete":
                    completed_turns[turn_id] = float(
                        payload.get("completed_at") or record.get("ordinal") or 0.0
                    )
                if payload.get("type") != "item_completed":
                    continue
                item = payload.get("item") or {}
                if item.get("type") == "UserMessage" and request_sha256 in json.dumps(
                    item, ensure_ascii=False
                ):
                    user_bound.add(turn_id)
                if (
                    item.get("type") == "AgentMessage"
                    and item.get("phase") == "final_answer"
                ):
                    final_turns.add(turn_id)
        if session_id != thread_id:
            continue
        for turn_id in contexts & user_bound & final_turns & completed_turns.keys():
            matches.append((completed_turns[turn_id], log_path, turn_id))
    if not matches:
        raise RuntimeError(
            "no completed Codex turn was found for the captured request"
        )
    # A rejected malformed/schema-invalid answer may be retried.  Bind the
    # importer to the newest completed attempt while preserving older turns in
    # the immutable session log for audit.
    _, log_path, turn_id = max(matches, key=lambda row: row[0])
    return log_path, turn_id


def _parse_cli_events(
    stdout: str,
    expected_thread_id: str | None,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    counts = Counter(str(event.get("type") or "") for event in events)
    started_ids = {
        str(event.get("thread_id") or "")
        for event in events
        if event.get("type") == "thread.started"
    } - {""}
    if len(started_ids) != 1:
        raise RuntimeError("Codex CLI did not report exactly one backend thread")
    actual_thread_id = next(iter(started_ids))
    if expected_thread_id is not None and actual_thread_id != expected_thread_id:
        raise RuntimeError("Codex CLI resumed an unexpected thread")
    if counts["turn.completed"] != 1:
        raise RuntimeError("Codex CLI did not report exactly one completed turn")
    return {
        "event_count": len(events),
        "event_type_counts": dict(sorted(counts.items())),
        "thread_id": actual_thread_id,
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
    }


def _cli_failure_category(stderr: str) -> str:
    normalized = stderr.casefold()
    if "active writer" in normalized or "thread-store conflict" in normalized:
        return "thread_writer_conflict"
    if "context window" in normalized or "maximum context" in normalized:
        return "context_limit"
    if "usage limit" in normalized or "quota" in normalized:
        return "quota_exhausted"
    if "timed out" in normalized or "timeout" in normalized:
        return "transient_timeout"
    return "unclassified_cli_failure"


def _pending_queue(trial_dir: Path) -> list[Path]:
    pending_dir = trial_dir / "codex_session_shadow" / "pending"
    cache_dir = trial_dir / "open_coscientist_trial_cache"
    receipt_dir = trial_dir / "codex_session_shadow" / "imported"
    rows: list[tuple[float, Path]] = []
    for path in pending_dir.glob("*.json"):
        cache_path = cache_dir / path.name
        receipt_path = receipt_dir / path.name
        if cache_path.exists() and receipt_path.exists():
            continue
        if cache_path.exists() != receipt_path.exists():
            raise RuntimeError(f"partial mixed import state for request {path.stem}")
        pending = json.loads(path.read_text(encoding="utf-8"))
        rows.append((float(pending.get("created_at") or 0.0), path))
    return [path for _, path in sorted(rows, key=lambda row: (row[0], row[1].name))]


def run_queue(
    *,
    trial_dir: Path,
    thread_id: str,
    initial_thread_turns: int,
    max_turns_per_thread: int,
    max_requests: int,
    codex_executable: Path,
    session_log_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    trial_dir = Path(os.path.abspath(str(trial_dir)))
    cache_dir = trial_dir / "open_coscientist_trial_cache"
    shadow_dir = trial_dir / "codex_session_shadow"
    staging_dir = shadow_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = shadow_dir / "runner_checkpoint.json"
    queue = _pending_queue(trial_dir)
    selected = queue[:max_requests]
    started_at = time.time()
    completed: list[dict[str, Any]] = []
    current_thread_id = thread_id
    current_thread_turns = initial_thread_turns
    backend_sessions: list[dict[str, Any]] = [
        {
            "thread_id": current_thread_id,
            "successful_requests_before_run": initial_thread_turns,
            "successful_requests_in_run": 0,
        }
    ]
    checkpoint: dict[str, Any] = {
        "schema_version": RUNNER_SCHEMA,
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "thread_id": thread_id,
        "current_thread_id": current_thread_id,
        "current_thread_turns": current_thread_turns,
        "max_turns_per_thread": max_turns_per_thread,
        "backend_sessions": backend_sessions,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "pending_at_start": len(queue),
        "selected_requests": len(selected),
        "completed": completed,
        "prompts_or_responses_printed": False,
    }
    _write_json_atomic(checkpoint_path, checkpoint)

    try:
        for index, pending_path in enumerate(selected, start=1):
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            request_sha256 = str(pending["request_sha256"])
            response_path = staging_dir / f"{request_sha256}.txt"
            controlled_prompt = _controlled_prompt(pending)
            create_new_thread = current_thread_turns >= max_turns_per_thread
            if create_new_thread:
                command = [
                    str(codex_executable),
                    "exec",
                    "--json",
                    "-m",
                    MODEL,
                    "-c",
                    'model_reasoning_effort="max"',
                    "-s",
                    "read-only",
                    "--skip-git-repo-check",
                    "-C",
                    str(trial_dir),
                    "-o",
                    str(response_path),
                    "-",
                ]
            else:
                command = [
                    str(codex_executable),
                    "exec",
                    "resume",
                    "--json",
                    "-m",
                    MODEL,
                    "-c",
                    'model_reasoning_effort="max"',
                    "-c",
                    'sandbox_mode="read-only"',
                    "--skip-git-repo-check",
                    "-o",
                    str(response_path),
                    current_thread_id,
                    "-",
                ]
            request_started = time.time()
            result: subprocess.CompletedProcess[str] | None = None
            failure_category = ""
            for cli_attempt, delay_before_attempt in enumerate((0.0, 2.0, 8.0, 30.0), start=1):
                if delay_before_attempt:
                    time.sleep(delay_before_attempt)
                result = subprocess.run(
                    command,
                    input=controlled_prompt,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    cwd=trial_dir,
                    timeout=timeout_seconds,
                    check=False,
                )
                if result.returncode == 0:
                    break
                failure_category = _cli_failure_category(result.stderr)
                if (
                    create_new_thread
                    or failure_category != "thread_writer_conflict"
                    or cli_attempt == 4
                ):
                    break
            assert result is not None
            if result.returncode != 0:
                raise RuntimeError(
                    "Codex CLI request failed with exit code "
                    f"{result.returncode}; category={failure_category}; stderr_sha256="
                    f"{_sha256_bytes(result.stderr.encode('utf-8'))}"
                )
            cli_audit = _parse_cli_events(
                result.stdout,
                None if create_new_thread else current_thread_id,
            )
            actual_thread_id = str(cli_audit["thread_id"])
            if create_new_thread:
                current_thread_id = actual_thread_id
                current_thread_turns = 0
                backend_sessions.append(
                    {
                        "thread_id": current_thread_id,
                        "successful_requests_before_run": 0,
                        "successful_requests_in_run": 0,
                    }
                )
            if not response_path.is_file():
                raise RuntimeError("Codex CLI did not write its final response file")
            log_path, turn_id = _find_bound_turn(
                log_root=session_log_root,
                thread_id=current_thread_id,
                request_sha256=request_sha256,
            )
            try:
                imported = import_answer(
                    pending_path=pending_path,
                    session_log=log_path,
                    turn_id=turn_id,
                    cache_dir=cache_dir,
                    trial_dir=trial_dir,
                    backend_thread_id_override=current_thread_id,
                )
            except Exception as exc:
                rejected_path = (
                    shadow_dir
                    / "rejected"
                    / f"{request_sha256}.{turn_id}.json"
                )
                _write_json_atomic(
                    rejected_path,
                    {
                        "schema_version": (
                            "case1-codex-session-mixed-rejected-answer.v1"
                        ),
                        "status": "rejected_before_formal_cache_write",
                        "rejected_at": time.time(),
                        "request_sha256": request_sha256,
                        "model": MODEL,
                        "reasoning_effort": EFFORT,
                        "thread_id": current_thread_id,
                        "turn_id": turn_id,
                        "response_content_sha256": _sha256_bytes(
                            response_path.read_bytes()
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "formal_cache_written": False,
                        "prompts_or_responses_printed": False,
                    },
                )
                raise
            response_sha256 = _sha256_bytes(response_path.read_bytes())
            if response_sha256 != imported["response_content_sha256"]:
                raise RuntimeError("CLI response file differs from the session final answer")
            row = {
                "queue_index": index,
                "request_sha256": request_sha256,
                "turn_id": turn_id,
                "thread_id": current_thread_id,
                "duration_seconds": time.time() - request_started,
                "response_content_sha256": response_sha256,
                "cache_count_after": imported["cache_count_after"],
                "validation": imported["validation"],
                "session_log_path": str(log_path),
                "cli_audit": cli_audit,
            }
            completed.append(row)
            current_thread_turns += 1
            backend_sessions[-1]["successful_requests_in_run"] = int(
                backend_sessions[-1]["successful_requests_in_run"]
            ) + 1
            checkpoint.update(
                {
                    "updated_at": time.time(),
                    "completed": completed,
                    "current_thread_id": current_thread_id,
                    "current_thread_turns": current_thread_turns,
                    "backend_sessions": backend_sessions,
                    "remaining_from_initial_queue": len(queue) - len(completed),
                }
            )
            _write_json_atomic(checkpoint_path, checkpoint)
            print(
                json.dumps(
                    {
                        "status": "request_imported",
                        "queue_index": index,
                        "selected_requests": len(selected),
                        "request_sha256": request_sha256,
                        "turn_id": turn_id,
                        "thread_id": current_thread_id,
                        "duration_seconds": row["duration_seconds"],
                        "cache_count_after": imported["cache_count_after"],
                        "validation_passed": imported["validation"]["passed"],
                        "prompts_or_responses_printed": False,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except BaseException as exc:
        checkpoint.update(
            {
                "status": "failed",
                "updated_at": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed": completed,
            }
        )
        _write_json_atomic(checkpoint_path, checkpoint)
        raise

    checkpoint.update(
        {
            "status": "completed",
            "updated_at": time.time(),
            "duration_seconds": time.time() - started_at,
            "completed": completed,
            "remaining_pending": len(_pending_queue(trial_dir)),
        }
    )
    _write_json_atomic(checkpoint_path, checkpoint)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--initial-thread-turns", type=int, default=0)
    parser.add_argument("--max-turns-per-thread", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--codex", type=Path, default=Path(shutil.which("codex") or ""))
    parser.add_argument(
        "--session-log-root",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_requests < 1:
        raise RuntimeError("--max-requests must be positive")
    if args.initial_thread_turns < 0:
        raise RuntimeError("--initial-thread-turns cannot be negative")
    if args.max_turns_per_thread < 1:
        raise RuntimeError("--max-turns-per-thread must be positive")
    if not args.codex.is_file():
        raise FileNotFoundError(args.codex)
    result = run_queue(
        trial_dir=args.trial_dir,
        thread_id=args.thread_id,
        initial_thread_turns=args.initial_thread_turns,
        max_turns_per_thread=args.max_turns_per_thread,
        max_requests=args.max_requests,
        codex_executable=args.codex,
        session_log_root=args.session_log_root,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_requests": len(result["completed"]),
                "remaining_pending": result["remaining_pending"],
                "duration_seconds": result["duration_seconds"],
                "prompts_or_responses_printed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
