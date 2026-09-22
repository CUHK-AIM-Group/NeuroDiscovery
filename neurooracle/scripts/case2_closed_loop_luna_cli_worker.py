"""Drive the frozen Luna Codex session and seal exact CLI-captured answers.

The replay gateway remains a passive OpenAI-compatible endpoint.  This worker
polls its hash-locked request directory, resumes one frozen Codex CLI session,
and asks that session to read only the corresponding dispatch packet.  Codex's
``--output-last-message`` option writes the provider's final answer directly to
the response path, avoiding any manual transcription or content editing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.scripts.case2_closed_loop_luna_thread_gateway import (
    MODEL,
    REASONING_EFFORT,
    RESPONSE_CAPTURE_MODE,
    atomic_write_json,
    read_json,
    seal_response,
    sha256_file,
    validate_envelope,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_provider_prompt(
    record: Mapping[str, Any], dispatch_path: Path, *, attempt: int = 1
) -> str:
    dispatch_sha = sha256_file(dispatch_path)
    response_format = (record.get("request_body") or {}).get("response_format")
    inner_constraint = ""
    if isinstance(response_format, Mapping) and response_format.get("type") in {
        "json_schema",
        "json_object",
    }:
        required: list[str] = []
        if response_format.get("type") == "json_schema":
            wrapper = response_format.get("json_schema")
            schema = wrapper.get("schema") if isinstance(wrapper, Mapping) else None
            if isinstance(schema, Mapping) and isinstance(schema.get("required"), list):
                required = [str(value) for value in schema["required"]]
        inner_constraint = (
            "message.content 必须是恰好一个完整 JSON object 的字符串表示；"
            "所有 required 属性都必须位于同一最外层花括号内部，最后一个右花括号后不得有任何字符。"
            "提交前按等价于 json.loads(message.content) 的方式自检，并核对原请求内层 schema。"
            f"内层顶级 required 属性为: {json.dumps(required, ensure_ascii=False)}。"
        )
    return (
        "处理一个正式 CS2 replay request。只读下面唯一指定的 dispatch 文件。"
        "分别核对：(1) 文件内 request_id 等于本消息 request_id；"
        "(2) 文件内 request_sha256 等于本消息 request_sha256；"
        "(3) dispatch 文件完整字节 SHA256 等于本消息 dispatch_file_sha256。"
        "三者是不同字段，不要相互混淆。随后严格执行 ORIGINAL_API_REQUEST_JSON。"
        "把本轮视为无状态模型调用：不得使用此前 request 的候选、反馈或答案；"
        "禁止读取任何其他路径、仓库、oracle、结果或网络，禁止写文件。"
        "最终只返回 dispatch 规定的 JSON envelope，不要 Markdown 或解释。"
        f"这是该 request 的 capture attempt {attempt}；必须完整重新生成本次答案。"
        f"{inner_constraint}\n\n"
        f"request_id: {record['request_id']}\n"
        f"request_sha256: {record['request_sha256']}\n"
        f"dispatch_file_sha256: {dispatch_sha}\n"
        f"dispatch_path: {dispatch_path}\n"
    )


def validate_inner_response_schema(
    record: Mapping[str, Any], envelope: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate structured message content before a replay response is sealed.

    The outer Codex CLI schema only constrains the transport envelope.  Native
    frameworks may additionally send an OpenAI ``response_format`` JSON Schema
    whose object is serialized inside ``message.content``.  A transport-valid
    but truncated inner object must be retried here instead of being hash-locked
    and replayed to every downstream framework retry.
    """

    body = record.get("request_body")
    _require(isinstance(body, Mapping), "Frozen request body is missing")
    response_format = body.get("response_format")
    if not isinstance(response_format, Mapping):
        return {"required": False, "validated": False}
    format_type = str(response_format.get("type") or "")
    if format_type == "json_object":
        content = envelope["message"].get("content")
        _require(isinstance(content, str) and content.strip(), "JSON response content is empty")
        inner = json.loads(content)
        _require(isinstance(inner, dict), "JSON response content must encode one object")
        return {
            "required": True,
            "validated": True,
            "format_type": format_type,
            "inner_top_level_properties": sorted(inner),
        }
    if format_type != "json_schema":
        return {"required": False, "validated": False, "format_type": format_type}
    wrapper = response_format.get("json_schema")
    _require(isinstance(wrapper, Mapping), "response_format.json_schema is missing")
    schema = wrapper.get("schema")
    _require(isinstance(schema, Mapping), "Inner response JSON Schema is missing")
    content = envelope["message"].get("content")
    _require(isinstance(content, str) and content.strip(), "Structured response content is empty")
    inner = json.loads(content)
    _require(isinstance(inner, dict), "Structured response content must encode one object")
    jsonschema.validate(instance=inner, schema=dict(schema))
    schema_sha256 = __import__("hashlib").sha256(
        json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "required": True,
        "validated": True,
        "format_type": format_type,
        "inner_schema_sha256": schema_sha256,
        "inner_top_level_properties": sorted(inner),
    }


def pending_records(data_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    rows: list[tuple[dict[str, Any], Path]] = []
    for request_path in sorted((data_dir / "requests").glob("luna_*.json")):
        request_id = request_path.stem
        meta_path = data_dir / "responses" / f"{request_id}.meta.json"
        if meta_path.is_file():
            continue
        record = read_json(request_path)
        dispatch_path = data_dir / "dispatches" / f"{request_id}.txt"
        _require(dispatch_path.is_file(), f"Dispatch packet is missing: {dispatch_path}")
        rows.append((record, dispatch_path))
    return rows


def next_capture_attempt(capture_dir: Path) -> int:
    """Return the first unused monotonically increasing capture attempt."""

    attempts: list[int] = []
    for path in capture_dir.glob("a*.json"):
        match = re.fullmatch(r"a([1-9][0-9]*)\.json", path.name)
        if match:
            attempts.append(int(match.group(1)))
    return max(attempts, default=0) + 1


def _archive_failed_capture(raw_path: Path, attempt: int) -> Path | None:
    if not raw_path.exists():
        return None
    failed_dir = raw_path.parents[1] / "f"
    failed_dir.mkdir(parents=True, exist_ok=True)
    destination = failed_dir / f"{raw_path.stem[-16:]}.a{attempt}.txt"
    if destination.exists():
        raise FileExistsError(f"Failed-capture audit already exists: {destination}")
    raw_path.replace(destination)
    return destination


def process_request(
    *,
    data_dir: Path,
    record: Mapping[str, Any],
    dispatch_path: Path,
    thread_id: str,
    host_id: str,
    provider_workspace: Path,
    codex_executable: Path,
    response_schema: Path,
    max_attempts: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    request_id = str(record["request_id"])
    _require(record.get("thread_id") == thread_id, "Frozen request thread identity changed")
    _require(record.get("host_id") == host_id, "Frozen request host identity changed")
    _require(record.get("model") == MODEL, "Frozen request model changed")
    _require(record.get("reasoning_effort") == REASONING_EFFORT, "Reasoning effort changed")
    raw_path = data_dir / "responses" / f"{request_id}.txt"
    meta_path = data_dir / "responses" / f"{request_id}.meta.json"
    capture_dir = data_dir / "c" / request_id[-16:]
    capture_dir.mkdir(parents=True, exist_ok=True)
    provider_dispatch_path = provider_workspace / "CURRENT_DISPATCH.txt"
    dispatch_bytes = dispatch_path.read_bytes()
    provider_dispatch_tmp = provider_workspace / ".CURRENT_DISPATCH.txt.tmp"
    provider_dispatch_tmp.write_bytes(dispatch_bytes)
    provider_dispatch_tmp.replace(provider_dispatch_path)
    _require(
        sha256_file(provider_dispatch_path) == sha256_file(dispatch_path),
        "Provider-workspace dispatch materialization changed bytes",
    )
    first_attempt = next_capture_attempt(capture_dir)
    _require(
        first_attempt <= max_attempts,
        f"Capture attempt budget exhausted ({max_attempts}) for {request_id}",
    )

    for attempt in range(first_attempt, max_attempts + 1):
        prompt = build_provider_prompt(
            record, provider_dispatch_path, attempt=attempt
        )
        if meta_path.is_file():
            return read_json(meta_path)
        if raw_path.exists():
            _archive_failed_capture(raw_path, attempt)
        stdout_path = capture_dir / f"a{attempt}.jsonl"
        stderr_path = capture_dir / f"a{attempt}.err"
        command = [
            str(codex_executable),
            "exec",
            "resume",
            thread_id,
            "--all",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--ignore-user-config",
            "-m",
            MODEL,
            "-c",
            f'model_reasoning_effort="{REASONING_EFFORT}"',
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-schema",
            str(response_schema),
            "--json",
            "-o",
            str(raw_path),
            prompt,
        ]
        started = utc_now()
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            try:
                completed = subprocess.run(
                    command,
                    cwd=provider_workspace,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=timeout_seconds,
                    check=False,
                )
                return_code: int | None = int(completed.returncode)
                timed_out = False
            except subprocess.TimeoutExpired:
                return_code = None
                timed_out = True
        attempt_audit = {
            "schema_version": "neurooracle.codex_cli_capture_attempt.v1",
            "request_id": request_id,
            "request_sha256": record["request_sha256"],
            "source_dispatch_path": str(dispatch_path),
            "provider_dispatch_path": str(provider_dispatch_path),
            "dispatch_sha256": sha256_file(dispatch_path),
            "provider_dispatch_sha256": sha256_file(provider_dispatch_path),
            "thread_id": thread_id,
            "host_id": host_id,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "response_capture_mode": RESPONSE_CAPTURE_MODE,
            "response_schema_path": str(response_schema),
            "response_schema_sha256": sha256_file(response_schema),
            "attempt": attempt,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "return_code": return_code,
            "timed_out": timed_out,
            "stdout_relative_path": stdout_path.relative_to(data_dir).as_posix(),
            "stderr_relative_path": stderr_path.relative_to(data_dir).as_posix(),
            "raw_response_present": raw_path.is_file(),
            "manual_response_content_edit": False,
            "provider_boundary_enforcement": "per-turn prompt plus complete command-log audit",
            "local_approval_and_sandbox_bypassed": True,
        }
        if return_code == 0 and raw_path.is_file():
            try:
                raw_text = raw_path.read_text(encoding="utf-8-sig")
                envelope = validate_envelope(json.loads(raw_text))
                inner_validation = validate_inner_response_schema(record, envelope)
                raw_sha = sha256_file(raw_path)
                attempt_audit.update(
                    {
                        "status": "valid_provider_final_captured",
                        "raw_response_sha256": raw_sha,
                        "inner_response_validation": inner_validation,
                    }
                )
                atomic_write_json(capture_dir / f"a{attempt}.json", attempt_audit)
                meta = seal_response(
                    data_dir,
                    request_id,
                    thread_id=thread_id,
                    host_id=host_id,
                    provider_turn_id=f"codex-cli-resume:{thread_id}:{request_id}:attempt-{attempt}",
                    provider_message_id=f"output-last-message:sha256:{raw_sha}",
                )
                atomic_write_json(
                    capture_dir / "lock.json",
                    {
                        "schema_version": "neurooracle.codex_cli_capture_lock.v1",
                        "status": "sealed_exact_output_last_message_capture",
                        "request_id": request_id,
                        "request_sha256": record["request_sha256"],
                        "thread_id": thread_id,
                        "host_id": host_id,
                        "model": MODEL,
                        "reasoning_effort": REASONING_EFFORT,
                        "response_capture_mode": RESPONSE_CAPTURE_MODE,
                        "response_schema_sha256": sha256_file(response_schema),
                        "attempt": attempt,
                        "raw_response_sha256": raw_sha,
                        "response_meta_sha256": sha256_file(meta_path),
                        "inner_response_validation": inner_validation,
                        "manual_response_content_edit": False,
                        "provider_command_audit_required": True,
                        "local_approval_and_sandbox_bypassed": True,
                        "sealed_at_utc": utc_now(),
                    },
                )
                provider_dispatch_path.unlink(missing_ok=True)
                return meta
            except Exception as exc:
                attempt_audit.update(
                    {
                        "status": "invalid_provider_final",
                        "validation_error_type": type(exc).__name__,
                        "validation_error": str(exc)[:1000],
                    }
                )
        else:
            attempt_audit.update(
                {
                    "status": "codex_cli_transport_failed",
                    "raw_response_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
                }
            )
        atomic_write_json(capture_dir / f"a{attempt}.json", attempt_audit)
        _archive_failed_capture(raw_path, attempt)
        if attempt < max_attempts:
            time.sleep(2.0)
    raise RuntimeError(f"Codex CLI provider failed after {max_attempts} attempts: {request_id}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--host-id", default="local")
    parser.add_argument("--provider-workspace", type=Path, required=True)
    parser.add_argument("--codex-executable", type=Path)
    parser.add_argument("--response-schema", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--request-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    provider_workspace = args.provider_workspace.resolve()
    _require(provider_workspace.is_dir(), "Provider workspace is missing")
    executable = args.codex_executable
    if executable is None:
        located = shutil.which("codex")
        _require(bool(located), "Codex CLI executable was not found")
        executable = Path(str(located))
    executable = executable.resolve()
    _require(executable.is_file(), "Codex CLI executable is missing")
    response_schema = args.response_schema.resolve()
    _require(response_schema.is_file(), "Response JSON schema is missing")
    _require(args.max_attempts >= 1, "max-attempts must be positive")
    _require(args.poll_seconds > 0, "poll-seconds must be positive")
    for name in ("requests", "dispatches", "responses", "c", "f"):
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "status": "ready",
                "thread_id": args.thread_id,
                "host_id": args.host_id,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "response_capture_mode": RESPONSE_CAPTURE_MODE,
                "data_dir": str(data_dir),
                "provider_workspace": str(provider_workspace),
                "codex_executable": str(executable),
                "response_schema": str(response_schema),
                "response_schema_sha256": sha256_file(response_schema),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    while True:
        rows = pending_records(data_dir)
        if rows:
            record, dispatch_path = rows[0]
            meta = process_request(
                data_dir=data_dir,
                record=record,
                dispatch_path=dispatch_path,
                thread_id=args.thread_id,
                host_id=args.host_id,
                provider_workspace=provider_workspace,
                codex_executable=executable,
                response_schema=response_schema,
                max_attempts=args.max_attempts,
                timeout_seconds=args.request_timeout_seconds,
            )
            print(
                json.dumps(
                    {
                        "status": "sealed",
                        "request_id": meta["request_id"],
                        "raw_response_sha256": meta["raw_response_sha256"],
                    }
                ),
                flush=True,
            )
        elif args.once:
            return 0
        else:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
