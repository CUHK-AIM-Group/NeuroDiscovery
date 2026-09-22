"""Recover one transport-valid Luna replay response rejected by a downstream schema.

The normal replay worker validates the outer OpenAI-compatible response envelope.
This narrowly scoped utility additionally validates ``message.content`` against the
JSON Schema carried by the original frozen request.  It preserves direct Codex CLI
capture and never edits provider response text.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.scripts.case2_closed_loop_luna_cli_worker import (
    build_provider_prompt,
    utc_now,
)
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


RECOVERY_SCHEMA = "neurooracle.codex_thread_replay_downstream_schema_recovery.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _inner_schema(record: Mapping[str, Any]) -> dict[str, Any]:
    body = record.get("request_body")
    _require(isinstance(body, Mapping), "Frozen request body is missing")
    response_format = body.get("response_format")
    _require(isinstance(response_format, Mapping), "response_format is missing")
    _require(response_format.get("type") == "json_schema", "json_schema response is required")
    wrapper = response_format.get("json_schema")
    _require(isinstance(wrapper, Mapping), "response_format.json_schema is missing")
    schema = wrapper.get("schema")
    _require(isinstance(schema, Mapping), "Inner response schema is missing")
    return dict(schema)


def _recovery_prompt(
    record: Mapping[str, Any], dispatch_path: Path, schema: Mapping[str, Any]
) -> str:
    required = [str(item) for item in schema.get("required", [])]
    judgment = (
        schema.get("properties", {})
        .get("judgment_explanation", {})
        .get("required", [])
    )
    return (
        build_provider_prompt(record, dispatch_path)
        + "\nDOWNSTREAM_SCHEMA_RECOVERY_V1\n"
        + "Previous direct captures for this exact frozen request passed the outer "
        "envelope check but message.content failed the original response_format JSON "
        "Schema. Do not copy or repair any previous answer. Generate a fresh answer "
        "from the ORIGINAL_API_REQUEST_JSON. Before returning, parse your proposed "
        "message.content yourself and verify it is one complete JSON object that "
        "validates against the response_format schema. Serialize that complete object "
        "as the message.content string, then return the required outer envelope.\n"
        + "Observed format defects to avoid: the inner object omitted its final closing "
        "brace, and decision_summary/confidence_level were placed inside "
        "judgment_explanation. They must remain top-level properties.\n"
        + f"Required inner top-level properties: {json.dumps(required)}\n"
        + f"Required judgment_explanation properties only: {json.dumps(list(judgment))}\n"
        + "Do not change the candidate IDs or research task. Do not use other files, "
        "prior candidate information, outcomes, or network access.\n"
    )


def _validate_inner(raw_path: Path, schema: Mapping[str, Any]) -> dict[str, Any]:
    outer = validate_envelope(json.loads(raw_path.read_text(encoding="utf-8-sig")))
    content = outer["message"].get("content")
    _require(isinstance(content, str) and content.strip(), "message.content is empty")
    inner = json.loads(content)
    _require(isinstance(inner, dict), "message.content must encode one JSON object")
    jsonschema.validate(instance=inner, schema=dict(schema))
    return inner


def recover(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    provider_workspace = args.provider_workspace.resolve()
    request_id = args.request_id
    request_path = data_dir / "requests" / f"{request_id}.json"
    dispatch_path = data_dir / "dispatches" / f"{request_id}.txt"
    raw_path = data_dir / "responses" / f"{request_id}.txt"
    meta_path = data_dir / "responses" / f"{request_id}.meta.json"
    _require(request_path.is_file(), "Frozen request is missing")
    _require(dispatch_path.is_file(), "Frozen dispatch is missing")
    _require(not raw_path.exists() and not meta_path.exists(), "Active response slot is not empty")
    _require(provider_workspace.is_dir(), "Provider workspace is missing")
    _require(args.response_schema.is_file(), "Outer response schema is missing")
    _require(args.codex_executable.is_file(), "Codex executable is missing")
    record = read_json(request_path)
    _require(record.get("request_id") == request_id, "Request ID mismatch")
    _require(record.get("thread_id") == args.thread_id, "Thread ID mismatch")
    _require(record.get("host_id") == args.host_id, "Host ID mismatch")
    _require(record.get("model") == MODEL, "Model mismatch")
    _require(record.get("reasoning_effort") == REASONING_EFFORT, "Reasoning mismatch")
    schema = _inner_schema(record)

    capture_dir = data_dir / "c_recovery" / request_id[-16:]
    capture_dir.mkdir(parents=True, exist_ok=True)
    provider_dispatch_path = provider_workspace / "CURRENT_DISPATCH.txt"
    temporary = provider_workspace / ".CURRENT_DISPATCH.txt.tmp"
    temporary.write_bytes(dispatch_path.read_bytes())
    temporary.replace(provider_dispatch_path)
    _require(
        sha256_file(provider_dispatch_path) == sha256_file(dispatch_path),
        "Provider dispatch bytes changed",
    )
    prompt = _recovery_prompt(record, provider_dispatch_path, schema)

    for attempt in range(1, args.max_attempts + 1):
        stdout_path = capture_dir / f"a{attempt}.jsonl"
        stderr_path = capture_dir / f"a{attempt}.err"
        command = [
            str(args.codex_executable),
            "exec",
            "resume",
            args.thread_id,
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
            str(args.response_schema),
            "--json",
            "-o",
            str(raw_path),
            prompt,
        ]
        started = utc_now()
        timed_out = False
        return_code: int | None
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            try:
                result = subprocess.run(
                    command,
                    cwd=provider_workspace,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                return_code = int(result.returncode)
            except subprocess.TimeoutExpired:
                return_code = None
                timed_out = True

        audit: dict[str, Any] = {
            "schema_version": RECOVERY_SCHEMA,
            "status": "invalid",
            "request_id": request_id,
            "request_sha256": record["request_sha256"],
            "thread_id": args.thread_id,
            "host_id": args.host_id,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "response_capture_mode": RESPONSE_CAPTURE_MODE,
            "source_dispatch_path": str(dispatch_path),
            "provider_dispatch_path": str(provider_dispatch_path),
            "dispatch_sha256": sha256_file(dispatch_path),
            "provider_dispatch_sha256": sha256_file(provider_dispatch_path),
            "outer_response_schema_sha256": sha256_file(args.response_schema),
            "inner_response_schema_sha256": _canonical_sha256(schema),
            "recovery_prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "recovery_script_sha256": sha256_file(Path(__file__)),
            "attempt": attempt,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "return_code": return_code,
            "timed_out": timed_out,
            "manual_response_content_edit": False,
            "experimental_outcomes_revealed": False,
            "local_approval_and_sandbox_bypassed": True,
        }
        try:
            _require(return_code == 0 and raw_path.is_file(), "Provider capture failed")
            inner = _validate_inner(raw_path, schema)
            raw_sha = sha256_file(raw_path)
            audit.update(
                {
                    "status": "valid_outer_and_inner_schema_capture",
                    "raw_response_sha256": raw_sha,
                    "inner_top_level_properties": sorted(inner),
                }
            )
            atomic_write_json(capture_dir / f"a{attempt}.audit.json", audit)
            meta = seal_response(
                data_dir,
                request_id,
                thread_id=args.thread_id,
                host_id=args.host_id,
                provider_turn_id=(
                    f"codex-cli-resume-downstream-schema-recovery:{args.thread_id}:"
                    f"{request_id}:attempt-{attempt}"
                ),
                provider_message_id=f"output-last-message:sha256:{raw_sha}",
            )
            lock = {
                **audit,
                "status": "sealed_direct_capture_after_downstream_schema_recovery",
                "response_meta_sha256": sha256_file(meta_path),
            }
            atomic_write_json(capture_dir / "lock.json", lock)
            provider_dispatch_path.unlink(missing_ok=True)
            return lock
        except Exception as exc:
            audit["validation_error_type"] = type(exc).__name__
            audit["validation_error"] = str(exc)[:2000]
            if raw_path.is_file():
                audit["raw_response_sha256"] = sha256_file(raw_path)
                invalid_path = capture_dir / f"a{attempt}.raw.invalid.txt"
                _require(not invalid_path.exists(), "Invalid capture archive already exists")
                raw_path.replace(invalid_path)
            atomic_write_json(capture_dir / f"a{attempt}.audit.json", audit)
            if attempt == args.max_attempts:
                provider_dispatch_path.unlink(missing_ok=True)
                raise

    raise RuntimeError("Unreachable recovery state")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--host-id", default="local")
    parser.add_argument("--provider-workspace", type=Path, required=True)
    parser.add_argument("--codex-executable", type=Path)
    parser.add_argument("--response-schema", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args(argv)
    if args.codex_executable is None:
        located = shutil.which("codex")
        _require(bool(located), "Codex executable was not found")
        args.codex_executable = Path(str(located))
    args.codex_executable = args.codex_executable.resolve()
    args.response_schema = args.response_schema.resolve()
    _require(args.max_attempts >= 1, "max-attempts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = recover(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
