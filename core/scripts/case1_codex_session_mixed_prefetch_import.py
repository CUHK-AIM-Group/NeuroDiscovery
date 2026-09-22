"""Sequentially import validated Luna prefetch records into the formal cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from core.scripts.case1_codex_session_mixed_import import import_answer
from core.scripts.case1_codex_session_mixed_prefetch import (
    EFFORT,
    MODEL,
    PREFETCH_SCHEMA,
    _schema_pending_paths,
)
from core.scripts.case1_codex_session_mixed_runner import _write_json_atomic


IMPORT_SCHEMA = "case1-codex-session-mixed-prefetch-import.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def run_import(
    *,
    trial_dir: Path,
    schema_name: str,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    trial_dir = Path(os.path.abspath(str(trial_dir)))
    selected = _schema_pending_paths(
        trial_dir=trial_dir,
        schema_name=schema_name,
        offset=offset,
        limit=limit,
    )
    cache_dir = trial_dir / "open_coscientist_trial_cache"
    shadow_dir = trial_dir / "codex_session_shadow"
    checkpoint_path = (
        shadow_dir / f"prefetch_import_{schema_name}_{offset}_{limit}.json"
    )
    checkpoint: dict[str, Any] = {
        "schema_version": IMPORT_SCHEMA,
        "status": "running",
        "schema_name": schema_name,
        "offset": offset,
        "limit": limit,
        "started_at": time.time(),
        "imported": [],
        "already_imported": [],
        "not_prefetched": [],
        "prompts_or_responses_printed": False,
    }
    _write_json_atomic(checkpoint_path, checkpoint)
    for pending_path in selected:
        request_sha256 = pending_path.stem
        cache_path = cache_dir / pending_path.name
        receipt_path = shadow_dir / "imported" / pending_path.name
        if cache_path.exists() or receipt_path.exists():
            if not (cache_path.exists() and receipt_path.exists()):
                raise RuntimeError(
                    f"partial formal import state for {request_sha256}"
                )
            checkpoint["already_imported"].append(request_sha256)
            continue
        record_path = shadow_dir / "prefetched" / pending_path.name
        if not record_path.is_file():
            checkpoint["not_prefetched"].append(request_sha256)
            continue
        record = _load_json(record_path)
        if record.get("schema_version") != PREFETCH_SCHEMA:
            raise RuntimeError("prefetch schema mismatch")
        if record.get("status") != "validated_not_formally_imported":
            raise RuntimeError("prefetch record is not validated")
        if record.get("request_sha256") != request_sha256:
            raise RuntimeError("prefetch request hash mismatch")
        if record.get("model") != MODEL or record.get("reasoning_effort") != EFFORT:
            raise RuntimeError("prefetch backend identity mismatch")
        if bool(record.get("formal_cache_written")):
            raise RuntimeError("prefetch record already claims a cache write")
        staging_path = shadow_dir / "staging" / f"{request_sha256}.txt"
        if not staging_path.is_file():
            raise RuntimeError("prefetched staging response is missing")
        if _sha256_bytes(staging_path.read_bytes()) != record.get(
            "response_content_sha256"
        ):
            raise RuntimeError("prefetched staging response hash mismatch")
        imported = import_answer(
            pending_path=pending_path,
            session_log=Path(str(record["session_log_path"])),
            turn_id=str(record["turn_id"]),
            cache_dir=cache_dir,
            trial_dir=trial_dir,
            backend_thread_id_override=str(record["thread_id"]),
        )
        if imported["response_content_sha256"] != record[
            "response_content_sha256"
        ]:
            raise RuntimeError("formal import response hash differs from prefetch")
        checkpoint["imported"].append(
            {
                "request_sha256": request_sha256,
                "thread_id": record["thread_id"],
                "turn_id": record["turn_id"],
                "cache_count_after": imported["cache_count_after"],
            }
        )
        checkpoint["updated_at"] = time.time()
        _write_json_atomic(checkpoint_path, checkpoint)
        print(
            json.dumps(
                {
                    "status": "prefetch_imported",
                    "request_sha256": request_sha256,
                    "imported": len(checkpoint["imported"]),
                    "not_prefetched": len(checkpoint["not_prefetched"]),
                    "cache_count_after": imported["cache_count_after"],
                    "prompts_or_responses_printed": False,
                }
            ),
            flush=True,
        )
    checkpoint["status"] = "completed"
    checkpoint["finished_at"] = time.time()
    _write_json_atomic(checkpoint_path, checkpoint)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--schema-name", required=True)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--limit", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.limit < 1:
        raise RuntimeError("offset/limit are invalid")
    result = run_import(
        trial_dir=args.trial_dir,
        schema_name=args.schema_name,
        offset=args.offset,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "imported": len(result["imported"]),
                "already_imported": len(result["already_imported"]),
                "not_prefetched": len(result["not_prefetched"]),
                "prompts_or_responses_printed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
