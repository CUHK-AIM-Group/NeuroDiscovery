"""Recover a completed BrainPilot Case Study 1 batch from its native event log.

This is an evidence-preserving recovery path for a client-side control failure.
It never sends a message to BrainPilot and never repairs or regenerates a
scientific proposal.  The last structurally valid native ``result_deliver``
artifact is selected, matching the live client's last-delivery semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


METHOD = "brainpilot_native"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--start-rank", type=int, required=True)
    parser.add_argument("--end-rank", type=int, required=True)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any, *, trailing_newline: bool = True) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    return text + ("\n" if trailing_newline else "")


def write_once_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to overwrite non-identical artifact: {path}")
        return
    path.write_bytes(encoded)


def load_stable_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"Event log changed while being read: {path}")
    events: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at line {line_number}: {path}") from exc
        if not isinstance(event, dict):
            raise RuntimeError(f"Non-object event at line {line_number}: {path}")
        events.append(event)
    if not events:
        raise RuntimeError(f"Empty event log: {path}")
    return raw, events


def load_registry_ids(path: Path, expected_sha256: str) -> set[str]:
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Public registry hash mismatch: {actual_sha256} != {expected_sha256.lower()}"
        )
    candidate_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid registry JSONL at line {line_number}: {path}"
                ) from exc
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not candidate_id:
                raise RuntimeError(
                    f"Registry row {line_number} has no candidate_id: {path}"
                )
            candidate_ids.add(candidate_id)
    if not candidate_ids:
        raise RuntimeError(f"Public registry contains no candidate IDs: {path}")
    return candidate_ids


def balanced_json_objects(text: str) -> Iterable[str]:
    start = 0
    while start < len(text):
        if text[start] != "{":
            start += 1
            continue
        depth = 0
        in_string = False
        escaped = False
        index = start
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    start = index
                    break
            index += 1
        start += 1


def extract_candidate_object(value: Any, depth: int = 0) -> dict[str, Any] | None:
    if depth > 8 or value is None:
        return None
    if isinstance(value, dict):
        if value.get("method") == METHOD and isinstance(value.get("hypotheses"), list):
            return value
        for nested in value.values():
            found = extract_candidate_object(nested, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for nested in value:
            found = extract_candidate_object(nested, depth + 1)
            if found is not None:
                return found
        return None
    if not isinstance(value, str):
        return None
    try:
        found = extract_candidate_object(json.loads(value), depth + 1)
        if found is not None:
            return found
    except json.JSONDecodeError:
        pass
    for candidate in balanced_json_objects(value):
        try:
            found = extract_candidate_object(json.loads(candidate), depth + 1)
        except json.JSONDecodeError:
            continue
        if found is not None:
            return found
    return None


def parse_result_delivery(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    implicit_result = payload.get("to") == "principal" and payload.get("msg_type") is None
    if payload.get("msg_type") != "result_deliver" and not implicit_result:
        return None
    content = payload.get("content")
    if not isinstance(content, str):
        return None
    return extract_candidate_object(content)


def validate_candidate(
    payload: dict[str, Any],
    *,
    allowed_ids: set[str],
    start_rank: int,
    end_rank: int,
) -> str | None:
    hypotheses = payload.get("hypotheses")
    expected_ranks = list(range(start_rank, end_rank + 1))
    if not isinstance(hypotheses, list) or len(hypotheses) != len(expected_ranks):
        return "wrong_hypothesis_count"
    ranks: list[int] = []
    candidate_ids: list[str] = []
    for item in hypotheses:
        if not isinstance(item, dict):
            return "non_object_hypothesis"
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            return "non_integer_rank"
        candidate_id = str(item.get("candidate_id") or "").strip()
        rationale = item.get("rationale")
        confidence = item.get("confidence")
        if not candidate_id or candidate_id not in allowed_ids:
            return "candidate_outside_sealed_registry"
        if not isinstance(rationale, str) or not rationale.strip():
            return "missing_rationale"
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return "non_numeric_confidence"
        if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            return "confidence_out_of_range"
        ranks.append(rank)
        candidate_ids.append(candidate_id)
    if sorted(ranks) != expected_ranks or len(set(ranks)) != len(ranks):
        return "rank_coverage_mismatch"
    if len(set(candidate_ids)) != len(candidate_ids):
        return "duplicate_candidate_id_within_batch"
    return None


def event_timestamp_bounds(events: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    timestamps: list[tuple[datetime, str]] = []
    for event in events:
        raw = event.get("_ts") or event.get("timestamp")
        if not isinstance(raw, str):
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        timestamps.append((parsed, raw))
    if not timestamps:
        return None, None
    timestamps.sort(key=lambda item: item[0])
    return timestamps[0][1], timestamps[-1][1]


def main() -> None:
    args = parse_args()
    if args.start_rank < 1 or args.end_rank < args.start_rank:
        raise ValueError("Invalid rank interval")
    batch_dir = args.batch_dir
    prompt_path = batch_dir / "prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)

    raw_events, all_events = load_stable_jsonl(args.events_jsonl)
    session_events = [
        event for event in all_events if event.get("session_id") == args.session_id
    ]
    if len(session_events) != len(all_events):
        raise RuntimeError(
            "Event log contains records outside the expected BrainPilot session"
        )
    terminal_states = [
        event
        for event in session_events
        if event.get("type") == "CUSTOM" and event.get("name") == "session_state"
    ]
    if not terminal_states:
        raise RuntimeError("No BrainPilot session_state event was recorded")
    terminal_active = (
        terminal_states[-1].get("value", {}).get("runState", {}).get("active")
    )
    if terminal_active is not False:
        raise RuntimeError("BrainPilot session is not terminally inactive; refusing recovery")

    allowed_ids = load_registry_ids(args.registry, args.expected_registry_sha256)
    chunks: dict[str, str] = {}
    valid_deliveries: list[dict[str, Any]] = []
    invalid_delivery_reasons: Counter[str] = Counter()
    for event_index, event in enumerate(session_events):
        if event.get("type") != "TOOL_CALL_ARGS" or not isinstance(
            event.get("delta"), str
        ):
            continue
        run_id = str(event.get("run_id") or "unknown")
        tool_call_id = str(event.get("tool_call_id") or "anonymous")
        event_key = f"{run_id}\0{tool_call_id}"
        chunks[event_key] = chunks.get(event_key, "") + event["delta"]
        candidate = parse_result_delivery(chunks[event_key])
        if candidate is None:
            continue
        invalid_reason = validate_candidate(
            candidate,
            allowed_ids=allowed_ids,
            start_rank=args.start_rank,
            end_rank=args.end_rank,
        )
        if invalid_reason is not None:
            invalid_delivery_reasons[invalid_reason] += 1
            continue
        serialized = canonical_json(candidate, trailing_newline=False)
        valid_deliveries.append(
            {
                "event_index": event_index,
                "event_key": event_key,
                "run_id": run_id,
                "tool_call_id": tool_call_id,
                "content": serialized,
                "sha256": sha256_bytes(serialized.encode("utf-8")),
            }
        )
    if not valid_deliveries:
        raise RuntimeError("No structurally valid native result_deliver artifact found")

    selected = valid_deliveries[-1]
    content_counts = Counter(item["sha256"] for item in valid_deliveries)
    first_timestamp, last_timestamp = event_timestamp_bounds(session_events)
    metadata = {
        "schema_version": "brainpilot-case1-client-recovery.v1",
        "started_at": first_timestamp,
        "finished_at": last_timestamp,
        "session_id": args.session_id,
        "session_reused": False,
        "end_reason": "recovered_last_valid_result_deliver_after_client_control_failure",
        "event_count": len(session_events),
        "terminal_session_active": terminal_active,
        "source_events_jsonl_sha256": sha256_bytes(raw_events),
        "prompt_sha256": sha256_file(prompt_path),
        "registry_sha256": sha256_file(args.registry),
        "expected_rank_interval": [args.start_rank, args.end_rank],
        "valid_result_deliveries": len(valid_deliveries),
        "unique_valid_result_artifacts": len(content_counts),
        "valid_result_artifact_counts_by_sha256": dict(sorted(content_counts.items())),
        "invalid_delivery_reasons": dict(sorted(invalid_delivery_reasons.items())),
        "selected_delivery_event_index": selected["event_index"],
        "selected_delivery_run_id": selected["run_id"],
        "selected_delivery_tool_call_id": selected["tool_call_id"],
        "selected_artifact_sha256": selected["sha256"],
        "selection_rule": "last structurally valid native result_deliver in event order",
        "model_generation_performed_by_recovery": False,
        "scientific_proposal_repaired_or_replaced": False,
    }

    write_once_text(
        batch_dir / "events.json", canonical_json(session_events, trailing_newline=True)
    )
    write_once_text(batch_dir / "final.txt", selected["content"])
    write_once_text(batch_dir / "client_meta.json", canonical_json(metadata))
    print(
        json.dumps(
            {
                "session_id": args.session_id,
                "events": len(session_events),
                "valid_deliveries": len(valid_deliveries),
                "unique_valid_artifacts": len(content_counts),
                "selected_artifact_sha256": selected["sha256"],
                "selected_delivery_event_index": selected["event_index"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
