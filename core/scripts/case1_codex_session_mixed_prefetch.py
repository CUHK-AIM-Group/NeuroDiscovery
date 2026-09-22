"""Prefetch independent Luna Max answers without touching the formal cache.

This utility is intentionally split from the formal importer.  It may run
several fresh Codex sessions concurrently, but it only writes immutable
per-request prefetch audit records and response staging files.  A separate,
strictly sequential import step is required before any answer becomes part of
the Open Co-Scientist cache.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import jsonschema

from core.scripts.case1_codex_session_mixed_import import _load_bound_final_answer
from core.scripts.case1_codex_session_mixed_runner import (
    EFFORT,
    MODEL,
    _controlled_prompt,
    _find_bound_turn,
    _parse_cli_events,
    _write_json_atomic,
)


PREFETCH_SCHEMA = "case1-codex-session-mixed-prefetch.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _schema_pending_paths(
    *, trial_dir: Path, schema_name: str, offset: int, limit: int
) -> list[Path]:
    pending_dir = trial_dir / "codex_session_shadow" / "pending"
    rows: list[tuple[float, str, Path]] = []
    for path in pending_dir.glob("*.json"):
        pending = _load_json(path)
        schema = (pending.get("request") or {}).get("json_schema") or {}
        if str(schema.get("name") or "") != schema_name:
            continue
        rows.append(
            (
                float(pending.get("created_at") or 0.0),
                str(pending.get("request_sha256") or ""),
                path,
            )
        )
    ordered = [path for _, _, path in sorted(rows)]
    selected = ordered[offset : offset + limit]
    if len(selected) != limit:
        raise RuntimeError(
            f"requested {limit} {schema_name!r} paths at offset {offset}, "
            f"but only selected {len(selected)}"
        )
    return selected


def _validate_answer(
    *,
    pending: dict[str, Any],
    session_log: Path,
    thread_id: str,
    turn_id: str,
) -> tuple[str, str, dict[str, Any]]:
    request = pending["request"]
    backend = pending["requested_backend"]
    content, message_id, turn_audit = _load_bound_final_answer(
        session_log=session_log,
        thread_id=thread_id,
        turn_id=turn_id,
        request_sha256=str(pending["request_sha256"]),
        expected_model=str(backend["model"]),
        expected_effort=str(backend["reasoning_effort"]),
    )
    schema_wrapper = request.get("json_schema")
    if schema_wrapper is not None:
        response = json.loads(content)
        if not isinstance(response, dict):
            raise RuntimeError("structured Luna response is not a JSON object")
        actual_schema = schema_wrapper.get("schema", schema_wrapper)
        jsonschema.validate(instance=response, schema=actual_schema)
        validation = {
            "mode": "json_schema",
            "passed": True,
            "schema_sha256": _sha256_bytes(
                json.dumps(
                    actual_schema,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ),
        }
    else:
        if not content.strip():
            raise RuntimeError("Luna response is empty")
        validation = {"mode": "nonempty_text", "passed": True}
    return content, message_id, {"validation": validation, **turn_audit}


def _prefetch_one(
    *,
    pending_path: Path,
    trial_dir: Path,
    codex_executable: Path,
    session_log_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    pending = _load_json(pending_path)
    request_sha256 = str(pending["request_sha256"])
    cache_path = trial_dir / "open_coscientist_trial_cache" / pending_path.name
    receipt_path = (
        trial_dir / "codex_session_shadow" / "imported" / pending_path.name
    )
    if cache_path.exists() or receipt_path.exists():
        raise RuntimeError(f"request is already formally imported: {request_sha256}")
    staging_path = (
        trial_dir / "codex_session_shadow" / "staging" / f"{request_sha256}.txt"
    )
    staging_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(staging_path),
        "-",
    ]
    started_at = time.time()
    result = subprocess.run(
        command,
        input=_controlled_prompt(pending),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=trial_dir,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Codex CLI prefetch failed with exit code "
            f"{result.returncode}; stderr_sha256="
            f"{_sha256_bytes(result.stderr.encode('utf-8'))}"
        )
    cli_audit = _parse_cli_events(result.stdout, None)
    thread_id = str(cli_audit["thread_id"])
    session_log, turn_id = _find_bound_turn(
        log_root=session_log_root,
        thread_id=thread_id,
        request_sha256=request_sha256,
    )
    try:
        content, message_id, answer_audit = _validate_answer(
            pending=pending,
            session_log=session_log,
            thread_id=thread_id,
            turn_id=turn_id,
        )
    except BaseException as exc:
        rejected_path = (
            trial_dir
            / "codex_session_shadow"
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
                "thread_id": thread_id,
                "turn_id": turn_id,
                "response_content_sha256": (
                    _sha256_bytes(staging_path.read_bytes())
                    if staging_path.is_file()
                    else None
                ),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "formal_cache_written": False,
                "prompts_or_responses_printed": False,
                "origin": "parallel_prefetch_validation",
            },
        )
        raise
    response_sha256 = _sha256_bytes(content.encode("utf-8"))
    if not staging_path.is_file():
        raise RuntimeError("Codex CLI did not write its final response file")
    if _sha256_bytes(staging_path.read_bytes()) != response_sha256:
        raise RuntimeError("CLI response file differs from the session final answer")
    record = {
        "schema_version": PREFETCH_SCHEMA,
        "status": "validated_not_formally_imported",
        "request_sha256": request_sha256,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "message_id": message_id,
        "session_log_path": str(session_log),
        "response_content_sha256": response_sha256,
        "validation": answer_audit["validation"],
        "turn_audit": {
            key: value
            for key, value in answer_audit.items()
            if key != "validation"
        },
        "duration_seconds": time.time() - started_at,
        "formal_cache_written": False,
        "prompts_or_responses_printed": False,
        "cli_audit": cli_audit,
    }
    record_path = (
        trial_dir
        / "codex_session_shadow"
        / "prefetched"
        / f"{request_sha256}.json"
    )
    _write_json_atomic(record_path, record)
    return record


def run_prefetch(
    *,
    trial_dir: Path,
    schema_name: str,
    offset: int,
    limit: int,
    max_workers: int,
    codex_executable: Path,
    session_log_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    trial_dir = Path(os.path.abspath(str(trial_dir)))
    selected = _schema_pending_paths(
        trial_dir=trial_dir,
        schema_name=schema_name,
        offset=offset,
        limit=limit,
    )
    checkpoint_path = (
        trial_dir
        / "codex_session_shadow"
        / f"prefetch_{schema_name}_{offset}_{limit}_checkpoint.json"
    )
    checkpoint: dict[str, Any] = {
        "schema_version": PREFETCH_SCHEMA,
        "status": "running",
        "schema_name": schema_name,
        "offset": offset,
        "limit": limit,
        "max_workers": max_workers,
        "started_at": time.time(),
        "completed": [],
        "failures": [],
        "formal_cache_written": False,
        "prompts_or_responses_printed": False,
    }
    _write_json_atomic(checkpoint_path, checkpoint)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _prefetch_one,
                pending_path=path,
                trial_dir=trial_dir,
                codex_executable=codex_executable,
                session_log_root=session_log_root,
                timeout_seconds=timeout_seconds,
            ): path
            for path in selected
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                record = future.result()
                checkpoint["completed"].append(
                    {
                        "request_sha256": record["request_sha256"],
                        "thread_id": record["thread_id"],
                        "turn_id": record["turn_id"],
                        "duration_seconds": record["duration_seconds"],
                    }
                )
                status = "prefetched"
            except BaseException as exc:
                checkpoint["failures"].append(
                    {
                        "request_sha256": path.stem,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
                status = "prefetch_failed"
            checkpoint["updated_at"] = time.time()
            _write_json_atomic(checkpoint_path, checkpoint)
            print(
                json.dumps(
                    {
                        "status": status,
                        "request_sha256": path.stem,
                        "completed": len(checkpoint["completed"]),
                        "failures": len(checkpoint["failures"]),
                        "selected": len(selected),
                        "prompts_or_responses_printed": False,
                    }
                ),
                flush=True,
            )
    checkpoint["status"] = (
        "completed" if not checkpoint["failures"] else "completed_with_failures"
    )
    checkpoint["finished_at"] = time.time()
    _write_json_atomic(checkpoint_path, checkpoint)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--schema-name", required=True)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--codex", required=True, type=Path)
    parser.add_argument(
        "--session-log-root",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit < 1 or args.max_workers < 1:
        raise RuntimeError("offset/work limits are invalid")
    if not args.codex.is_file():
        raise FileNotFoundError(args.codex)
    result = run_prefetch(
        trial_dir=args.trial_dir,
        schema_name=args.schema_name,
        offset=args.offset,
        limit=args.limit,
        max_workers=args.max_workers,
        codex_executable=args.codex,
        session_log_root=args.session_log_root,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed": len(result["completed"]),
                "failures": len(result["failures"]),
                "formal_cache_written": False,
                "prompts_or_responses_printed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
