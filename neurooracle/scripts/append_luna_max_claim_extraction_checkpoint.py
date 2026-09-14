"""Canonical, fail-closed checkpoint writer for Luna-max extraction shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neurooracle.scripts.finalize_luna_max_claim_extraction_fork import (  # noqa: E402
    finalize_shard,
    paper_key,
    read_json,
    read_jsonl,
    sha256_file,
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def registry_owner(campaign: Path, shard: str, worker_thread_id: str) -> None:
    registry = read_json(campaign / "FORK_THREAD_REGISTRY.json")
    if registry.get("model") != "gpt-5.6-luna" or registry.get("reasoning_effort") != "max":
        fail("registry model/reasoning lock mismatch")
    if registry.get("formal_kg_mutation_permitted") is not False:
        fail("formal KG mutation is not fail-closed")
    owners = [row for row in registry.get("canonical_workers", []) if row.get("shard") == shard]
    if len(owners) != 1:
        fail(f"canonical registry does not contain exactly one owner for {shard}")
    if str(owners[0].get("thread_id")) != worker_thread_id:
        fail(f"worker thread is not canonical owner of {shard}")
    attestation = read_json(campaign / "RECOVERY_ATTESTATION.json")
    if attestation.get("status") != "verified_semantic_equivalence":
        fail("campaign recovery attestation is not verified")
    expected = str(
        ((attestation.get("current_files") or {}).get("checkpoint_writer") or {}).get("sha256")
        or ""
    )
    if not expected or sha256_file(Path(__file__).resolve()) != expected:
        fail("checkpoint writer differs from the attested source")


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        fail("checkpoint is empty")
    for index, row in enumerate(rows, 1):
        if set(row) != {"queue_index", "items"}:
            fail(f"checkpoint row {index} must contain exactly queue_index/items")
        if not isinstance(row["queue_index"], int) or isinstance(row["queue_index"], bool):
            fail(f"checkpoint row {index} queue_index must be an integer")
        if not isinstance(row["items"], list):
            fail(f"checkpoint row {index} items must be a list")
        if len(row["items"]) > 12:
            fail(f"checkpoint row {index} contains more than 12 claims")
        if index > 1 and row["queue_index"] != rows[index - 2]["queue_index"] + 1:
            fail("checkpoint queue is not contiguous")
    if len(rows) > 25:
        fail("checkpoint may contain at most 25 papers")
    if len(rows) >= 10 and all(not row["items"] for row in rows):
        fail("all-zero checkpoint of ten or more papers is rejected")
    return rows


def existing_prefix(fork_dir: Path, tasks: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[int, bytes, str]:
    path = fork_dir / "raw_items.jsonl"
    if not path.exists():
        return 0, b"", ""
    original = path.read_bytes()
    rows = read_jsonl(path)
    start = int(manifest["queue_start"])
    if len(rows) > len(tasks):
        fail("existing raw_items exceeds task count")
    for offset, row in enumerate(rows):
        task = tasks[offset]
        expected_q = start + offset
        if int(row.get("queue_index")) != expected_q:
            fail(f"existing raw_items is not a contiguous prefix at {expected_q}")
        if str(row.get("paper_key") or "").lower() != paper_key(task):
            fail(f"existing raw_items paper key mismatch at {expected_q}")
        if str(row.get("source_context_sha256") or "") != str(task.get("source_context_sha256") or ""):
            fail(f"existing raw_items source hash mismatch at {expected_q}")
        if not isinstance(row.get("items"), list):
            fail(f"existing raw_items items is not a list at {expected_q}")
    return len(rows), original, sha256_file(path)


def bind_rows(rows: list[dict[str, Any]], tasks: list[dict[str, Any]], start_offset: int) -> list[dict[str, Any]]:
    if start_offset + len(rows) > len(tasks):
        fail("checkpoint exceeds remaining task queue")
    bound: list[dict[str, Any]] = []
    manifest_start = int(tasks[0]["queue_index"])
    for offset, checkpoint in enumerate(rows):
        task = tasks[start_offset + offset]
        expected_q = manifest_start + start_offset + offset
        if checkpoint["queue_index"] != expected_q:
            fail(f"checkpoint must start at next queue index {expected_q}")
        bound.append({
            "queue_index": expected_q,
            "paper_key": paper_key(task),
            "pmid": str((task.get("paper") or {}).get("pmid") or ""),
            "source_context_sha256": str(task["source_context_sha256"]),
            "items": checkpoint["items"],
        })
    return bound


def append_checkpoint(fork_dir: Path, input_path: Path, worker_thread_id: str) -> dict[str, Any]:
    fork_dir = fork_dir.resolve()
    input_path = input_path.resolve()
    manifest = read_json(fork_dir / "TASK_MANIFEST.json")
    tasks = read_jsonl(fork_dir / "task.jsonl")
    shard = str(manifest.get("shard_id") or fork_dir.name)
    campaign = fork_dir.parent.parent
    if input_path.parent != fork_dir:
        fail("checkpoint input must be inside the assigned shard directory")
    registry_owner(campaign, shard, worker_thread_id)
    if manifest.get("formal_kg_mutation_permitted") is not False:
        fail("task manifest permits formal KG mutation")
    if manifest.get("model") != "gpt-5.6-luna" or manifest.get("reasoning_effort") != "max":
        fail("task manifest model/reasoning lock mismatch")
    rows = load_checkpoint(input_path)
    existing_count, original_bytes, original_hash = existing_prefix(fork_dir, tasks, manifest)
    if existing_count == 0 and len(rows) != 10:
        fail("the first post-migration checkpoint must contain exactly 10 papers")
    lock = fork_dir / ".checkpoint.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("checkpoint writer is already active for this shard") from exc
    try:
        os.close(fd)
        bound = bind_rows(rows, tasks, existing_count)
        path = fork_dir / "raw_items.jsonl"
        if path.exists() and sha256_file(path) != original_hash:
            fail("raw_items changed after prefix validation; refusing concurrent write")
        new_bytes = original_bytes + "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in bound
        ).encode("utf-8")
        tmp = path.with_name(path.name + ".checkpoint.tmp")
        tmp.write_bytes(new_bytes)
        os.replace(tmp, path)
        try:
            summary = finalize_shard(fork_dir, allow_partial=True)
        except Exception:
            rollback = path.with_name(path.name + ".rollback.tmp")
            rollback.write_bytes(original_bytes)
            os.replace(rollback, path)
            raise
        log_path = fork_dir / "checkpoint_log.jsonl"
        log_rows = read_jsonl(log_path) if log_path.is_file() else []
        log_rows.append(
            {
                "schema_version": "neurooracle.luna_max_checkpoint_log.v1",
                "worker_thread_id": worker_thread_id,
                "shard_id": shard,
                "queue_start": int(bound[0]["queue_index"]),
                "queue_end": int(bound[-1]["queue_index"]),
                "papers_appended": len(bound),
                "processed_papers": int(summary["processed_papers"]),
                "raw_items_sha256": sha256_file(path),
            }
        )
        log_tmp = log_path.with_name(log_path.name + ".checkpoint.tmp")
        log_tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in log_rows),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(log_tmp, log_path)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--worker-thread-id", required=True)
    args = parser.parse_args()
    result = append_checkpoint(args.fork_dir, args.input, args.worker_thread_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
