"""Run the compact Case Study reaudit with a single validated ledger writer.

HTTP workers never touch SQLite.  The main thread validates and conditionally
writes one completed request at a time.  The formal knowledge graph is never
mutated by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from neurooracle.scripts.benchmark_compact_case_study_reaudit import (
    COMPACT_PROMPT,
    compact_payload,
    execute_batch,
    paper_aware_batches,
)
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    claim_contract_fields,
)
from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    RUBRIC_VERSION,
    compact_json,
    connect,
)
from neurooracle.scripts.run_full_graph_case_study_reaudit import validate_response
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def acquire_lock(output_dir: Path) -> tuple[int, Path]:
    lock_path = output_dir / "compact_primary.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags)
    except FileExistsError as exc:
        detail = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"another compact primary runner owns {lock_path}: {detail}") from exc
    os.write(
        descriptor,
        (compact_json({"pid": os.getpid(), "created_at": utc_now()}) + "\n").encode("utf-8"),
    )
    os.fsync(descriptor)
    return descriptor, lock_path


def release_lock(descriptor: int, lock_path: Path) -> None:
    os.close(descriptor)
    lock_path.unlink(missing_ok=True)


def fetch_rows_by_ids(
    connection: sqlite3.Connection, claim_ids: list[str]
) -> list[tuple[str, str, str]]:
    if not claim_ids:
        return []
    placeholders = ",".join("?" for _ in claim_ids)
    fetched = connection.execute(
        f"SELECT claim_id, paper_key, payload_json FROM claims WHERE claim_id IN ({placeholders})",
        claim_ids,
    ).fetchall()
    by_id = {
        str(row[0]): (str(row[0]), str(row[1]), str(row[2])) for row in fetched
    }
    missing = [claim_id for claim_id in claim_ids if claim_id not in by_id]
    if missing:
        raise ValueError(f"staged results reference missing claims: {missing[:5]}")
    return [by_id[claim_id] for claim_id in claim_ids]


def stored_review(
    review: dict[str, Any],
    row: tuple[str, str, str],
    *,
    model: str,
    reasoning_effort: str,
    reviewer_id: str,
    api_metadata: dict[str, Any],
    task_payload_sha256: str,
    reviewed_at: str,
) -> dict[str, Any]:
    _, paper_key, payload_json = row
    payload = json.loads(payload_json)
    stored = {
        **review,
        "rubric_version": RUBRIC_VERSION,
        "review_stage": "primary",
        "reviewer_id": reviewer_id,
        "reasoning_effort": reasoning_effort,
        "reviewed_at": reviewed_at,
        "compact_transport": {
            "schema": "compact_case_study_reaudit.v1",
            "model": model,
            "task_payload_sha256": task_payload_sha256,
            "prompt_sha256": hashlib.sha256(COMPACT_PROMPT.encode("utf-8")).hexdigest(),
            "api_metadata": api_metadata,
        },
    }
    stored.update(
        claim_contract_fields(
            paper_key=paper_key,
            payload=payload,
            labels=review["claim_case_study_ids"],
            gates=review["gates"],
            rubric_version=RUBRIC_VERSION,
            case_study_ids=CASE_STUDY_IDS,
        )
    )
    return stored


def persist_validated_batch(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str]],
    reviews: list[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str,
    reviewer_id: str,
    api_metadata: dict[str, Any],
    task_payload_sha256: str,
) -> dict[str, int]:
    validated = validate_response(rows, {"reviews": reviews})
    row_by_id = {row[0]: row for row in rows}
    reviewed_at = utc_now()
    status_counts = {"final_complete": 0, "secondary_pending": 0}
    connection.execute("BEGIN IMMEDIATE")
    try:
        for review in validated:
            status = (
                "secondary_pending"
                if review["needs_secondary_review"]
                else "final_complete"
            )
            stored = stored_review(
                review,
                row_by_id[review["claim_id"]],
                model=model,
                reasoning_effort=reasoning_effort,
                reviewer_id=reviewer_id,
                api_metadata=api_metadata,
                task_payload_sha256=task_payload_sha256,
                reviewed_at=reviewed_at,
            )
            cursor = connection.execute(
                """
                UPDATE claims SET review_status=?, review_json=?, reviewed_at=?
                WHERE claim_id=? AND review_status='pending'
                """,
                (
                    status,
                    compact_json(stored),
                    reviewed_at,
                    review["claim_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"conditional ledger write failed for {review['claim_id']}"
                )
            status_counts[status] += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return status_counts


def mark_failed_batch(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str]],
    *,
    error: str,
    model: str,
    reasoning_effort: str,
) -> int:
    failed_at = utc_now()
    failure_json = compact_json(
        {
            "rubric_version": RUBRIC_VERSION,
            "review_stage": "primary",
            "reviewer_id": f"model:{model}:compact",
            "reasoning_effort": reasoning_effort,
            "failed_at": failed_at,
            "error": error,
        }
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        changed = 0
        for claim_id, _, _ in rows:
            cursor = connection.execute(
                """
                UPDATE claims SET review_status='primary_failed', review_json=?, reviewed_at=?
                WHERE claim_id=? AND review_status='pending'
                """,
                (failure_json, failed_at, claim_id),
            )
            changed += cursor.rowcount
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return changed


def import_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    descriptor, lock_path = acquire_lock(output_dir)
    connection = connect(output_dir / "reaudit.sqlite")
    imported = {"final_complete": 0, "secondary_pending": 0, "already_reviewed": 0}
    try:
        for source_dir in args.benchmark_dir:
            source_dir = source_dir.resolve()
            if (source_dir / "DO_NOT_IMPORT.json").exists():
                reason = (source_dir / "DO_NOT_IMPORT.json").read_text(
                    encoding="utf-8", errors="replace"
                )
                raise ValueError(
                    f"benchmark directory is explicitly quarantined: {source_dir}: {reason}"
                )
            if not (source_dir / "summary.json").exists():
                raise ValueError(f"benchmark directory has no completed summary: {source_dir}")
            for results_path in sorted(source_dir.glob("workers_*/results.jsonl")):
                with results_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        result = json.loads(line)
                        if not result.get("ok"):
                            continue
                        reviews = result["reviews"]
                        claim_ids = [str(review["claim_id"]) for review in reviews]
                        rows = fetch_rows_by_ids(connection, claim_ids)
                        statuses = {
                            str(row[0]): str(row[1])
                            for row in connection.execute(
                                f"SELECT claim_id, review_status FROM claims WHERE claim_id IN ({','.join('?' for _ in claim_ids)})",
                                claim_ids,
                            ).fetchall()
                        }
                        if any(statuses[claim_id] != "pending" for claim_id in claim_ids):
                            imported["already_reviewed"] += len(claim_ids)
                            continue
                        counts = persist_validated_batch(
                            connection,
                            rows,
                            reviews,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                            reviewer_id=f"model:{args.model}:compact_benchmark",
                            api_metadata=result.get("metadata") or {},
                            task_payload_sha256=sha256_json(
                                {"source": str(results_path), "claim_ids": claim_ids}
                            ),
                        )
                        for status, count in counts.items():
                            imported[status] += count
        imported["source_directories"] = [str(path.resolve()) for path in args.benchmark_dir]
        return imported
    finally:
        connection.close()
        release_lock(descriptor, lock_path)


def production_run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    output_dir = args.output_dir.resolve()
    descriptor, lock_path = acquire_lock(output_dir)
    connection = connect(output_dir / "reaudit.sqlite")
    connection.row_factory = sqlite3.Row
    api_args = SimpleNamespace(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        transport_retries=args.transport_retries,
        schema_retries=args.schema_retries,
        transport=args.transport,
        max_output_tokens=args.max_output_tokens,
    )
    started = time.monotonic()
    totals = {
        "attempted": 0,
        "final_complete": 0,
        "secondary_pending": 0,
        "primary_failed": 0,
        "requests": 0,
    }
    reports_dir = output_dir / "run_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    events_path = reports_dir / f"compact_primary_{stamp}.jsonl"
    try:
        if args.retry_failed:
            connection.execute(
                "UPDATE claims SET review_status='pending', review_json=NULL, reviewed_at=NULL WHERE review_status='primary_failed'"
            )
            connection.commit()
        initial = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        remaining_limit = args.max_claims if args.max_claims > 0 else None
        wave = 0
        with events_path.open("a", encoding="utf-8") as event_log:
            while remaining_limit is None or totals["attempted"] < remaining_limit:
                wave_limit = args.workers * args.claims_per_request
                if remaining_limit is not None:
                    wave_limit = min(wave_limit, remaining_limit - totals["attempted"])
                raw_rows = connection.execute(
                    """
                    SELECT claim_id, paper_key, payload_json FROM claims
                    WHERE review_status='pending' ORDER BY graph_ordinal LIMIT ?
                    """,
                    (wave_limit,),
                ).fetchall()
                rows = [(str(row[0]), str(row[1]), str(row[2])) for row in raw_rows]
                if not rows:
                    break
                wave += 1
                batches = paper_aware_batches(rows, args.claims_per_request)
                prepared = [
                    (
                        batch,
                        compact_payload(
                            connection,
                            batch,
                            abstract_chars=args.abstract_chars,
                            paper_context_limit=args.paper_context_limit,
                        ),
                    )
                    for batch in batches
                ]
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(execute_batch, batch, payload, api_args): (
                            batch,
                            payload,
                        )
                        for batch, payload in prepared
                    }
                    for future in as_completed(futures):
                        batch, payload = futures[future]
                        result = future.result()
                        totals["requests"] += 1
                        totals["attempted"] += len(batch)
                        if result["ok"]:
                            counts = persist_validated_batch(
                                connection,
                                batch,
                                result["reviews"],
                                model=args.model,
                                reasoning_effort=args.reasoning_effort,
                                reviewer_id=f"model:{args.model}:compact",
                                api_metadata=result.get("metadata") or {},
                                task_payload_sha256=sha256_json(payload),
                            )
                            for status, count in counts.items():
                                totals[status] += count
                        else:
                            error = str(result.get("error") or "unknown compact request failure")
                            totals["primary_failed"] += mark_failed_batch(
                                connection,
                                batch,
                                error=error,
                                model=args.model,
                                reasoning_effort=args.reasoning_effort,
                            )
                        event_log.write(compact_json({"wave": wave, **result}) + "\n")
                        event_log.flush()
                elapsed = time.monotonic() - started
                rate = totals["attempted"] * 3600 / elapsed if elapsed else 0.0
                print(
                    "PROGRESS "
                    f"wave={wave} attempted={totals['attempted']} "
                    f"final={totals['final_complete']} secondary={totals['secondary_pending']} "
                    f"failed={totals['primary_failed']} rate={rate:.2f}_claims_h",
                    flush=True,
                )
        final = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        report = {
            "schema": "compact_case_study_reaudit_production_run.v1",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "workers": args.workers,
            "claims_per_request": args.claims_per_request,
            "initial_status_counts": initial,
            "final_status_counts": final,
            "run_totals": totals,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "formal_kg_mutated": False,
            "events_path": str(events_path.resolve()),
        }
        (reports_dir / f"compact_primary_{stamp}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        connection.close()
        api_args.api_key = ""
        release_lock(descriptor, lock_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("--model", default="gpt-5.5")
    common.add_argument("--reasoning-effort", default="high")

    import_parser = subparsers.add_parser("import-benchmark", parents=[common])
    import_parser.add_argument("--benchmark-dir", type=Path, nargs="+", required=True)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_parser.add_argument("--transport", choices=("urllib", "openai_sdk"), default="urllib")
    run_parser.add_argument("--workers", type=int, default=64)
    run_parser.add_argument("--claims-per-request", type=int, default=12)
    run_parser.add_argument("--max-claims", type=int, default=0)
    run_parser.add_argument("--abstract-chars", type=int, default=2500)
    run_parser.add_argument("--paper-context-limit", type=int, default=20)
    run_parser.add_argument("--timeout", type=float, default=600.0)
    run_parser.add_argument("--transport-retries", type=int, default=2)
    run_parser.add_argument("--schema-retries", type=int, default=2)
    run_parser.add_argument("--max-output-tokens", type=int, default=12000)
    run_parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "import-benchmark":
        result = import_benchmark(args)
    else:
        result = production_run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
