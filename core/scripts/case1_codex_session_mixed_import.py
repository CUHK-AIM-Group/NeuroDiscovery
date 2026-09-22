"""Validate and import one Codex-session answer into an OpenCo trial cache.

The importer is intentionally one-request-at-a-time.  It binds a captured
exact request to one completed Codex turn, verifies the requested Luna model
and reasoning effort from the local session log, validates structured output,
and writes the native Open Co-Scientist cache shape plus a content-free audit
receipt.  It never prints prompts or model responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import jsonschema


PENDING_SCHEMA = "case1-codex-session-shadow-request.v1"
RECEIPT_SCHEMA = "case1-codex-session-mixed-import-receipt.v1"
MANIFEST_SCHEMA = "case1-codex-session-mixed-backend-manifest.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_json_atomic(
    path: Path,
    value: Any,
    *,
    ensure_ascii: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _request_cache_key(request: dict[str, Any]) -> str:
    key_data: dict[str, Any] = {
        "prompt": request["prompt"],
        "model": request["model"],
        "temperature": request["temperature"],
        "max_tokens": request["max_tokens"],
    }
    if request.get("tools") is not None:
        key_data["tools"] = json.dumps(request["tools"], sort_keys=True)
    if request.get("json_schema") is not None:
        key_data["json_schema"] = json.dumps(
            request["json_schema"], sort_keys=True
        )
    if request.get("force_json") is not None:
        key_data["force_json"] = request["force_json"]
    return _sha256_text(json.dumps(key_data, sort_keys=True))


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                raise RuntimeError("Codex final answer contains an invalid content block")
            text = block.get("text")
            if not isinstance(text, str):
                raise RuntimeError("Codex final answer contains a non-text block")
            parts.append(text)
        return "".join(parts)
    raise RuntimeError("Codex final answer has an unsupported content representation")


def _load_bound_final_answer(
    *,
    session_log: Path,
    thread_id: str,
    turn_id: str,
    request_sha256: str,
    expected_model: str,
    expected_effort: str,
) -> tuple[str, str, dict[str, Any]]:
    session_id = ""
    turn_contexts: list[dict[str, Any]] = []
    user_items: list[dict[str, Any]] = []
    final_items: list[dict[str, Any]] = []
    task_complete = False

    with session_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload") or {}
            if record.get("type") == "session_meta":
                session_id = str(payload.get("id") or payload.get("session_id") or "")
            if (
                record.get("type") == "turn_context"
                and payload.get("turn_id") == turn_id
            ):
                turn_contexts.append(payload)
            if (
                record.get("type") == "event_msg"
                and payload.get("turn_id") == turn_id
            ):
                if payload.get("type") == "task_complete":
                    task_complete = True
                if payload.get("type") == "item_completed":
                    item = payload.get("item") or {}
                    if item.get("type") == "UserMessage":
                        user_items.append(item)
                    if (
                        item.get("type") == "AgentMessage"
                        and item.get("phase") == "final_answer"
                    ):
                        final_items.append(item)

    if session_id != thread_id:
        raise RuntimeError("session log thread id does not match the pending request")
    if len(turn_contexts) != 1:
        raise RuntimeError("expected exactly one matching turn context")
    context = turn_contexts[0]
    if str(context.get("model") or "") != expected_model:
        raise RuntimeError("Codex turn model does not match the requested backend")
    if str(context.get("effort") or "") != expected_effort:
        raise RuntimeError("Codex turn reasoning effort does not match the request")
    if not task_complete:
        raise RuntimeError("Codex turn is not complete")
    if not any(
        request_sha256 in json.dumps(item, ensure_ascii=False)
        for item in user_items
    ):
        raise RuntimeError("Codex turn is not bound to the captured request hash")
    if len(final_items) != 1:
        raise RuntimeError("expected exactly one final answer in the Codex turn")

    final = final_items[0]
    content = _message_text(final.get("content"))
    if not content.strip():
        raise RuntimeError("Codex final answer is empty")
    message_id = str(final.get("id") or "")
    if not message_id:
        raise RuntimeError("Codex final answer has no message id")
    turn_audit = {
        "session_log_path": str(session_log),
        "turn_id": turn_id,
        "message_id": message_id,
        "turn_context_sha256": _canonical_json_sha256(context),
        "user_item_sha256": _canonical_json_sha256(user_items[0]),
        "final_item_sha256": _canonical_json_sha256(final),
    }
    return content, message_id, turn_audit


def import_answer(
    *,
    pending_path: Path,
    session_log: Path,
    turn_id: str,
    cache_dir: Path,
    trial_dir: Path,
    backend_thread_id_override: str | None = None,
) -> dict[str, Any]:
    pending = _load_json_object(pending_path, "pending request")
    if pending.get("schema_version") != PENDING_SCHEMA:
        raise RuntimeError("unsupported pending request schema")
    if pending.get("scientific_classification") != (
        "mixed_backend_user_approved_pending_import"
    ):
        raise RuntimeError("pending request is not approved for mixed import")
    if bool(pending.get("formal_cache_written")):
        raise RuntimeError("pending request already claims a formal cache write")

    request = pending.get("request")
    backend = pending.get("requested_backend")
    if not isinstance(request, dict) or not isinstance(backend, dict):
        raise RuntimeError("pending request is missing request/backend metadata")
    request_sha256 = str(pending.get("request_sha256") or "")
    if _request_cache_key(request) != request_sha256:
        raise RuntimeError("captured request hash does not recompute exactly")
    if pending_path.stem != request_sha256:
        raise RuntimeError("pending filename is not the exact request hash")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("captured prompt is empty")
    if _sha256_text(prompt) != pending.get("request", {}).get("prompt_sha256"):
        raise RuntimeError("captured prompt hash mismatch")
    if request.get("tools") is not None:
        raise RuntimeError("tool-loop requests require a dedicated bridge")

    expected_thread = str(backend.get("thread_id") or "")
    expected_model = str(backend.get("model") or "")
    expected_effort = str(backend.get("reasoning_effort") or "")
    actual_thread = backend_thread_id_override or expected_thread
    if not actual_thread:
        raise RuntimeError("actual Codex backend thread id is empty")
    content, message_id, turn_audit = _load_bound_final_answer(
        session_log=session_log,
        thread_id=actual_thread,
        turn_id=turn_id,
        request_sha256=request_sha256,
        expected_model=expected_model,
        expected_effort=expected_effort,
    )

    json_schema = request.get("json_schema")
    if json_schema is not None:
        response = json.loads(content)
        if not isinstance(response, dict):
            raise RuntimeError("structured Luna response is not a JSON object")
        actual_schema = json_schema.get("schema", json_schema)
        jsonschema.validate(instance=response, schema=actual_schema)
        validation = {
            "mode": "json_schema",
            "schema_sha256": _canonical_json_sha256(actual_schema),
            "passed": True,
        }
        response_kind = "parsed_json_object"
    else:
        response = {"text": content}
        validation = {"mode": "nonempty_text", "passed": True}
        response_kind = "open_coscientist_text_wrapper"

    cache_dir = Path(os.path.abspath(str(cache_dir)))
    trial_dir = Path(os.path.abspath(str(trial_dir)))
    if trial_dir != cache_dir and trial_dir not in cache_dir.parents:
        raise RuntimeError("formal cache directory must be inside the trial")
    cache_path = cache_dir / f"{request_sha256}.json"
    if cache_path.exists():
        raise RuntimeError("formal cache entry already exists")
    old_count = len(list(cache_dir.glob("*.json")))
    cache_payload = {
        "request": {
            "model": request["model"],
            "temperature": request["temperature"],
            "max_tokens": request["max_tokens"],
            "prompt_preview": (
                prompt[:200] + "..." if len(prompt) > 200 else prompt
            ),
        },
        "response": response,
    }
    # Match upstream Open Co-Scientist's default ASCII-safe cache encoding.
    _write_json_atomic(cache_path, cache_payload, ensure_ascii=True)
    new_count = len(list(cache_dir.glob("*.json")))
    if new_count != old_count + 1:
        raise RuntimeError("formal cache count did not increase by exactly one")

    imported_at = time.time()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "imported_formal_mixed_backend",
        "scientific_classification": "mixed_deepseek_v4_pro_then_gpt_5_6_luna",
        "imported_at": imported_at,
        "request_sha256": request_sha256,
        "prompt_sha256": request["prompt_sha256"],
        "pending_file_sha256": _sha256_bytes(pending_path.read_bytes()),
        "cache_identity_sha256": pending["cache_identity_sha256"],
        "response_content_sha256": _sha256_text(content),
        "response_kind": response_kind,
        "validation": validation,
        "backend": {
            "model": expected_model,
            "reasoning_effort": expected_effort,
            "thread_id": actual_thread,
            "turn_id": turn_id,
            "message_id": message_id,
        },
        "capture_requested_thread_id": expected_thread,
        "turn_audit": turn_audit,
        "formal_cache": {
            "path": str(cache_path),
            "file_sha256": _sha256_bytes(cache_path.read_bytes()),
            "count_before": old_count,
            "count_after": new_count,
        },
        "prompts_or_responses_printed": False,
    }
    receipt_dir = trial_dir / "codex_session_shadow" / "imported"
    receipt_path = receipt_dir / f"{request_sha256}.json"
    if receipt_path.exists():
        raise RuntimeError("mixed-backend receipt already exists")
    _write_json_atomic(receipt_path, receipt)

    manifest_path = trial_dir / "codex_session_mixed_backend_manifest.json"
    if manifest_path.exists():
        manifest = _load_json_object(manifest_path, "mixed-backend manifest")
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise RuntimeError("mixed-backend manifest schema mismatch")
        if manifest.get("cache_identity_sha256") != pending["cache_identity_sha256"]:
            raise RuntimeError("mixed-backend manifest cache identity mismatch")
        stable = manifest.get("continuation_backend") or {}
        if any(
            stable.get(key) != receipt["backend"][key]
            for key in ("model", "reasoning_effort")
        ):
            raise RuntimeError("mixed-backend model or reasoning effort changed")
        thread_ids = list(stable.get("thread_ids") or [])
        legacy_thread_id = str(stable.get("thread_id") or "")
        if legacy_thread_id and legacy_thread_id not in thread_ids:
            thread_ids.append(legacy_thread_id)
        if actual_thread not in thread_ids:
            thread_ids.append(actual_thread)
        manifest["continuation_backend"] = {
            "model": expected_model,
            "reasoning_effort": expected_effort,
            "thread_ids": sorted(thread_ids),
        }
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "scientific_classification": (
                "mixed_deepseek_v4_pro_then_gpt_5_6_luna"
            ),
            "user_decision": "mixed continuation approved",
            "approved_at": imported_at,
            "source_thread_id": "01a023b5-044a-7331-ae63-dde31b561fcb",
            "cache_identity_sha256": pending["cache_identity_sha256"],
            "base_backend": {
                "model": "deepseek-v4-pro",
                "cache_entries_before_continuation": old_count,
            },
            "continuation_backend": {
                "model": expected_model,
                "reasoning_effort": expected_effort,
                "thread_ids": [actual_thread],
            },
            "formal_reporting_requirement": (
                "Label this trial and all derived results as mixed backend; "
                "do not report it as a pure DeepSeek or pure Luna seed."
            ),
        }
    receipts = sorted(receipt_dir.glob("*.json"))
    manifest.update(
        {
            "last_imported_at": imported_at,
            "imported_request_count": len(receipts),
            "imported_request_sha256": [path.stem for path in receipts],
            "formal_cache_entries_after_latest_import": new_count,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    return {
        "status": receipt["status"],
        "request_sha256": request_sha256,
        "turn_id": turn_id,
        "message_id": message_id,
        "response_content_sha256": receipt["response_content_sha256"],
        "validation": validation,
        "cache_count_before": old_count,
        "cache_count_after": new_count,
        "receipt_path": str(receipt_path),
        "manifest_path": str(manifest_path),
        "prompts_or_responses_printed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pending", required=True, type=Path)
    parser.add_argument("--session-log", required=True, type=Path)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--backend-thread-id-override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = import_answer(
        pending_path=args.pending,
        session_log=args.session_log,
        turn_id=args.turn_id,
        cache_dir=args.cache_dir,
        trial_dir=args.trial_dir,
        backend_thread_id_override=args.backend_thread_id_override,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
