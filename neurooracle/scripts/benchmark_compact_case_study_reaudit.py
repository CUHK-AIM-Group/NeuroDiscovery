"""Benchmark compact, paper-aware Case Study reaudit requests without KG writes.

The benchmark reads only ``pending`` claims from the immutable reaudit ledger.
Each concurrency setting receives a disjoint slice, and every validated response
is written to an isolated staging directory.  Neither the ledger nor the formal
knowledge graph is mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    RUBRIC_VERSION,
    compact_json,
    connect,
)
from neurooracle.scripts.run_full_graph_case_study_reaudit import (
    GATE_NAMES,
    GATE_REQUIREMENTS,
    SYSTEM_PROMPT,
    request_json,
    validate_response,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS


_LABEL_INDEX = {label: index for index, label in enumerate(CASE_STUDY_IDS)}
_GATE_INDEX = {gate: index for index, gate in enumerate(GATE_NAMES)}
_REQUIRED_INDEXES = ", ".join(
    f"{_LABEL_INDEX[label]}->{_GATE_INDEX[gate]}"
    for label, gate in GATE_REQUIREMENTS.items()
)
COMPACT_PROMPT = SYSTEM_PROMPT.split("Return JSON only.", 1)[0] + f"""
Compact transport changes representation only; it does not add, relax, or
reinterpret any frozen policy condition above. Label indexes follow the supplied
registry order 0..{len(CASE_STUDY_IDS) - 1}. Gate bits follow this order:
{", ".join(GATE_NAMES)}. Required label->gate indexes: {_REQUIRED_INDEXES}.
Return JSON only as {{"b":"exact supplied batch token","r":[[i,[label_indexes],confidence_0_to_100,gate_bitmask,secondary_boolean,reason_code],...]}}.
Preserve i order. label_indexes must be sorted unique integers. gate_bitmask is
0..{(1 << len(GATE_NAMES)) - 1}. reason_code is a short evidence-specific phrase,
max 12 words. Set secondary true for material ambiguity or confidence below 80.
Do not explain outside this compact array."""


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return 0


def select_pending_rows(
    connection: sqlite3.Connection,
    *,
    offset: int,
    limit: int,
) -> list[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT claim_id, paper_key, payload_json FROM claims
        WHERE review_status='pending'
        ORDER BY paper_key, graph_ordinal
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]


def paper_aware_batches(
    rows: list[tuple[str, str, str]], size: int
) -> list[list[tuple[str, str, str]]]:
    """Keep adjacent same-paper rows together while respecting request size."""
    batches: list[list[tuple[str, str, str]]] = []
    current: list[tuple[str, str, str]] = []
    for row in rows:
        if len(current) >= size:
            batches.append(current)
            current = []
        current.append(row)
    if current:
        batches.append(current)
    return batches


def compact_payload(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str]],
    *,
    abstract_chars: int,
    paper_context_limit: int,
) -> dict[str, Any]:
    paper_keys = list(dict.fromkeys(row[1] for row in rows))
    paper_indexes = {paper_key: index for index, paper_key in enumerate(paper_keys)}
    papers: list[list[Any]] = []
    selected_ids = {row[0] for row in rows}
    for paper_key in paper_keys:
        paper_row = connection.execute(
            "SELECT source_paper_json, abstract FROM papers WHERE paper_key=?",
            (paper_key,),
        ).fetchone()
        if paper_row is None:
            raise RuntimeError(f"missing paper ledger row: {paper_key}")
        source = json.loads(str(paper_row[0]))
        context_rows = connection.execute(
            """
            SELECT claim_id, payload_json FROM claims WHERE paper_key=?
            ORDER BY graph_ordinal LIMIT ?
            """,
            (paper_key, paper_context_limit),
        ).fetchall()
        context: list[list[Any]] = []
        for claim_id, payload_json in context_rows:
            if str(claim_id) in selected_ids:
                continue
            item = json.loads(str(payload_json))
            context.append(
                [
                    item.get("subject_name"),
                    item.get("predicate"),
                    item.get("object_name"),
                    item.get("raw_text"),
                ]
            )
        papers.append(
            [
                paper_indexes[paper_key],
                source.get("title"),
                str(paper_row[1] or "")[:abstract_chars],
                context,
            ]
        )

    claims: list[list[Any]] = []
    for index, (_, paper_key, payload_json) in enumerate(rows):
        item = json.loads(payload_json)
        evidence = item.get("evidence") or {}
        claims.append(
            [
                index,
                paper_indexes[paper_key],
                item.get("subject_name"),
                item.get("predicate"),
                item.get("object_name"),
                bool(item.get("negated", False)),
                item.get("raw_text"),
                item.get("conditions") or [],
                evidence.get("study_type"),
            ]
        )
    batch_token = hashlib.sha256(
        "\n".join(row[0] for row in rows).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "b": batch_token,
        "v": RUBRIC_VERSION,
        "allowed_case_study_ids": list(CASE_STUDY_IDS),
        "paper_fields": ["i", "title", "abstract", "other_claim_context"],
        "claim_fields": [
            "i",
            "paper_i",
            "subject",
            "predicate",
            "object",
            "negated",
            "raw_text",
            "conditions",
            "study_type",
        ],
        "p": papers,
        "c": claims,
    }


def expand_compact_response(
    expected_rows: list[tuple[str, str, str]],
    result: dict[str, Any],
    *,
    batch_token: str,
) -> list[dict[str, Any]]:
    if result.get("b") != batch_token:
        raise ValueError("compact response batch token mismatch")
    compact_reviews = result.get("r")
    if not isinstance(compact_reviews, list) or len(compact_reviews) != len(expected_rows):
        raise ValueError("compact response count does not match input")
    expanded: list[dict[str, Any]] = []
    for expected_index, row in enumerate(compact_reviews):
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("compact review must contain exactly six fields")
        index, label_indexes, confidence_int, gate_mask, secondary, reason_code = row
        if index != expected_index:
            raise ValueError("compact response order/index mismatch")
        if (
            not isinstance(label_indexes, list)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in label_indexes)
            or label_indexes != sorted(set(label_indexes))
            or any(value < 0 or value >= len(CASE_STUDY_IDS) for value in label_indexes)
        ):
            raise ValueError("label indexes must be sorted unique valid integers")
        if (
            not isinstance(confidence_int, int)
            or isinstance(confidence_int, bool)
            or not 0 <= confidence_int <= 100
        ):
            raise ValueError("confidence must be an integer from 0 to 100")
        if (
            not isinstance(gate_mask, int)
            or isinstance(gate_mask, bool)
            or not 0 <= gate_mask < (1 << len(GATE_NAMES))
        ):
            raise ValueError("gate bitmask is invalid")
        if not isinstance(secondary, bool):
            raise ValueError("secondary flag must be boolean")
        reason = str(reason_code or "").strip()
        if not reason or len(reason) > 120:
            raise ValueError("reason code is missing or too long")
        labels = [CASE_STUDY_IDS[value] for value in label_indexes]
        gates = {
            gate_name: bool(gate_mask & (1 << gate_index))
            for gate_index, gate_name in enumerate(GATE_NAMES)
        }
        for label in labels:
            required_gate = GATE_REQUIREMENTS.get(label)
            if required_gate and not gates[required_gate]:
                raise ValueError(f"label {label} lacks mandatory gate")
        expanded.append(
            {
                "claim_id": expected_rows[expected_index][0],
                "claim_case_study_ids": labels,
                "confidence": confidence_int / 100,
                "reason": f"Compact primary evidence: {reason}.",
                "needs_secondary_review": secondary or confidence_int < 80,
                "gates": gates,
            }
        )
    return validate_response(expected_rows, {"reviews": expanded})


def execute_batch(
    rows: list[tuple[str, str, str]],
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    last_error: Exception | None = None
    for schema_attempt in range(1, args.schema_retries + 1):
        try:
            result, metadata = request_json(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                wire_api="responses",
                payload=payload,
                timeout=args.timeout,
                retries=args.transport_retries,
                responses_plain_json=True,
                transport=args.transport,
                instructions=COMPACT_PROMPT,
                max_output_tokens=args.max_output_tokens,
            )
            if metadata.get("status") not in {None, "", "completed"}:
                raise ValueError(f"Responses status is not completed: {metadata.get('status')}")
            if metadata.get("incomplete_details"):
                raise ValueError("Responses API reported incomplete output")
            reviews = expand_compact_response(rows, result, batch_token=payload["b"])
            return {
                "ok": True,
                "reviews": reviews,
                "metadata": metadata,
                "latency_seconds": time.monotonic() - started,
                "schema_attempt": schema_attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    return {
        "ok": False,
        "claim_ids": [row[0] for row in rows],
        "error": f"{type(last_error).__name__}: {last_error}",
        "latency_seconds": time.monotonic() - started,
        "schema_attempt": args.schema_retries,
    }


def run_setting(
    connection: sqlite3.Connection,
    *,
    workers: int,
    offset: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    requested = workers * args.claims_per_request
    rows = select_pending_rows(connection, offset=offset, limit=requested)
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
    setting_dir = output_dir / f"workers_{workers:03d}"
    setting_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute_batch, batch, payload, args): batch
            for batch, payload in prepared
        }
        completed_requests = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_requests += 1
            if completed_requests % max(1, workers // 4) == 0:
                print(
                    f"SETTING workers={workers} requests={completed_requests}/{len(futures)}",
                    flush=True,
                )
    elapsed = time.monotonic() - started

    staged_path = setting_dir / "results.jsonl"
    with staged_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(compact_json(result) + "\n")

    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    reviews = [review for result in successes for review in result["reviews"]]
    latencies = [float(result["latency_seconds"]) for result in results]
    valid_claims = len(reviews)
    failed_claims = sum(len(result["claim_ids"]) for result in failures)
    secondary_claims = sum(bool(review["needs_secondary_review"]) for review in reviews)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for result in successes:
        usage = result.get("metadata", {}).get("usage") or {}
        input_tokens += usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens += usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens += usage_value(usage, "total_tokens")
    report = {
        "workers": workers,
        "offset": offset,
        "claims_per_request": args.claims_per_request,
        "requested_claims": len(rows),
        "requests": len(batches),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "valid_claims": valid_claims,
        "failed_claims": failed_claims,
        "technical_failure_percent": round(failed_claims * 100 / len(rows), 4) if rows else 0.0,
        "secondary_claims": secondary_claims,
        "secondary_percent": round(secondary_claims * 100 / valid_claims, 4) if valid_claims else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "effective_claims_per_hour": round(valid_claims * 3600 / elapsed, 2) if elapsed else 0.0,
        "latency_p50_seconds": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_p95_seconds": round(percentile(latencies, 0.95), 3),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_per_valid_claim": round(input_tokens / valid_claims, 2) if valid_claims else 0.0,
            "output_tokens_per_valid_claim": round(output_tokens / valid_claims, 2) if valid_claims else 0.0,
        },
        "staged_results": str(staged_path.resolve()),
        "failure_examples": [
            {"claim_ids": row["claim_ids"], "error": row["error"]}
            for row in failures[:10]
        ],
    }
    (setting_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    args.api_key = api_key
    ledger_path = args.output_dir.resolve() / "reaudit.sqlite"
    connection = connect(ledger_path)
    connection.row_factory = sqlite3.Row
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    benchmark_dir = args.output_dir.resolve() / "compact_benchmarks" / timestamp
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    try:
        ledger_before = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        reports: list[dict[str, Any]] = []
        offset = args.start_offset
        for workers in args.workers:
            report = run_setting(
                connection,
                workers=workers,
                offset=offset,
                args=args,
                output_dir=benchmark_dir,
            )
            reports.append(report)
            offset += workers * args.claims_per_request
            if report["technical_failure_percent"] > args.stop_failure_percent:
                print(
                    f"STOP failure threshold exceeded at workers={workers}", flush=True
                )
                break
        ledger_after = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        if ledger_after != ledger_before:
            raise RuntimeError("benchmark invariant violated: ledger status counts changed")
        eligible = [
            report
            for report in reports
            if report["technical_failure_percent"] <= args.max_failure_percent
            and report["secondary_percent"] <= args.max_secondary_percent
        ]
        best = max(eligible, key=lambda row: row["effective_claims_per_hour"], default=None)
        summary = {
            "schema": "compact_case_study_reaudit_benchmark.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "rubric_version": RUBRIC_VERSION,
            "ledger_status_counts_before": ledger_before,
            "ledger_status_counts_after": ledger_after,
            "formal_kg_mutated": False,
            "reports": reports,
            "acceptance": {
                "target_claims_per_hour": args.target_claims_per_hour,
                "max_technical_failure_percent": args.max_failure_percent,
                "max_secondary_percent": args.max_secondary_percent,
            },
            "best_eligible_setting": best,
            "one_day_target_met": bool(
                best and best["effective_claims_per_hour"] >= args.target_claims_per_hour
            ),
        }
        (benchmark_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        connection.close()
        args.api_key = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--transport", choices=("urllib", "openai_sdk"), default="urllib")
    parser.add_argument("--workers", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--claims-per-request", type=int, default=12)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--abstract-chars", type=int, default=2500)
    parser.add_argument("--paper-context-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--target-claims-per-hour", type=float, default=15000.0)
    parser.add_argument("--max-failure-percent", type=float, default=1.0)
    parser.add_argument("--max-secondary-percent", type=float, default=5.0)
    parser.add_argument("--stop-failure-percent", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
