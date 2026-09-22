"""Run resumable semantic Case Study membership review against the audit ledger.

This program never mutates the formal KG.  It writes validated per-claim review
decisions into the SQLite ledger prepared by
``prepare_full_graph_case_study_reaudit``.  Existing labels are deliberately
excluded from the model prompt to prevent anchoring.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    compact_json,
    connect,
)
from neurooracle.scripts.full_graph_case_study_reaudit_contract import (
    claim_contract_fields,
)
from neurooracle.src.case_study_scope import CASE_STUDY_IDS
from neurooracle.src.case_study_membership_policy import (
    GATE_NAMES,
    GATE_REQUIREMENTS,
    RUBRIC_VERSION,
    case_study_policy_prompt,
    validate_scope_decision,
)


SYSTEM_PROMPT = f"""You are the primary semantic auditor for a neuroscience
knowledge graph. Review every supplied claim independently.

{case_study_policy_prompt(extraction=False)}

Return JSON only. Preserve supplied claim order and IDs. For each claim return:
claim_id, claim_case_study_ids, confidence (0..1), reason (concise and
evidence-specific), needs_secondary_review, and gates with exactly these
{len(GATE_NAMES)} JSON booleans: {", ".join(GATE_NAMES)}.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_order(values: list[str]) -> list[str]:
    selected = set(values)
    return [value for value in CASE_STUDY_IDS if value in selected]


def review_payload(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str]],
    *,
    abstract_chars: int,
    paper_context_limit: int,
) -> dict[str, Any]:
    paper_keys = list(dict.fromkeys(row[1] for row in rows))
    papers: list[dict[str, Any]] = []
    paper_labels: dict[str, str] = {}
    for index, paper_key in enumerate(paper_keys, start=1):
        label = f"P{index}"
        paper_labels[paper_key] = label
        paper_row = connection.execute(
            "SELECT source_paper_json, abstract FROM papers WHERE paper_key=?",
            (paper_key,),
        ).fetchone()
        if paper_row is None:
            raise RuntimeError(f"missing paper ledger row: {paper_key}")
        context_rows = connection.execute(
            """
            SELECT payload_json FROM claims WHERE paper_key=?
            ORDER BY graph_ordinal LIMIT ?
            """,
            (paper_key, paper_context_limit),
        ).fetchall()
        context = []
        for (payload_json,) in context_rows:
            item = json.loads(payload_json)
            context.append(
                {
                    "claim_id": item["id"],
                    "subject": item.get("subject_name"),
                    "predicate": item.get("predicate"),
                    "object": item.get("object_name"),
                    "raw_text": item.get("raw_text"),
                    "study_type": (item.get("evidence") or {}).get("study_type"),
                }
            )
        abstract = str(paper_row[1] or "")
        if abstract_chars > 0:
            abstract = abstract[:abstract_chars]
        papers.append(
            {
                "paper_ref": label,
                "source_paper": json.loads(paper_row[0]),
                "abstract": abstract,
                "all_available_claim_context": context,
            }
        )

    claims: list[dict[str, Any]] = []
    for claim_id, paper_key, payload_json in rows:
        item = json.loads(payload_json)
        claims.append(
            {
                "claim_id": claim_id,
                "paper_ref": paper_labels[paper_key],
                "subject": item.get("subject_name"),
                "predicate": item.get("predicate"),
                "object": item.get("object_name"),
                "negated": item.get("negated"),
                "raw_text": item.get("raw_text"),
                "subject_type": item.get("subject_type"),
                "object_type": item.get("object_type"),
                "conditions": item.get("conditions"),
                "evidence": item.get("evidence"),
            }
        )
    return {
        "rubric_version": RUBRIC_VERSION,
        "allowed_case_study_ids": list(CASE_STUDY_IDS),
        "papers": papers,
        "claims_to_review": claims,
        "required_output": {"reviews": "one object per supplied claim, same order"},
    }


def request_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    wire_api: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
    responses_plain_json: bool = False,
    transport: str = "urllib",
    instructions: str = SYSTEM_PROMPT,
    max_output_tokens: int = 12000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = "Audit this batch:\n" + json.dumps(payload, ensure_ascii=False)
    if transport == "openai_sdk":
        if wire_api != "responses":
            raise ValueError("openai_sdk transport currently requires wire_api=responses")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("openai package is required for openai_sdk transport") from exc
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max(0, retries - 1),
        )
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=prompt,
                reasoning={"effort": reasoning_effort},
                store=False,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        content = response.output_text
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Responses SDK returned no output_text")
        usage = response.usage
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif usage is None:
            usage = {}
        return json.loads(content), {
            "model": response.model or model,
            "usage": usage,
            "status": str(response.status or ""),
            "incomplete_details": (
                response.incomplete_details.model_dump()
                if hasattr(response.incomplete_details, "model_dump")
                else response.incomplete_details
            ),
            "attempt": None,
            "wire_api": wire_api,
            "transport": transport,
            "structured_output_requested": False,
        }
    if transport != "urllib":
        raise ValueError(f"unsupported transport: {transport}")
    if wire_api == "responses":
        request_payload = {
            "model": model,
            "instructions": instructions,
            "input": prompt,
            "reasoning": {"effort": reasoning_effort},
            "text": {"format": {"type": "json_object"}},
            "store": False,
            "max_output_tokens": max_output_tokens,
        }
        endpoint = "responses"
    elif wire_api == "chat_completions":
        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": reasoning_effort,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "max_tokens": max_output_tokens,
        }
        endpoint = "chat/completions"
    else:
        raise ValueError(f"unsupported wire API: {wire_api}")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    last_error = ""
    attempt = 1
    structured_output_requested = wire_api == "responses" and not responses_plain_json
    while attempt <= retries:
        if wire_api == "responses":
            if structured_output_requested:
                request_payload["text"] = {"format": {"type": "json_object"}}
            else:
                request_payload.pop("text", None)
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/{endpoint}",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            if wire_api == "responses":
                content = raw.get("output_text")
                if not isinstance(content, str) or not content.strip():
                    parts = []
                    for output in raw.get("output") or []:
                        if not isinstance(output, dict):
                            continue
                        for item in output.get("content") or []:
                            if isinstance(item, dict) and item.get("type") == "output_text":
                                parts.append(str(item.get("text") or ""))
                    content = "".join(parts)
                if not content:
                    raise ValueError("Responses API returned no output_text")
                api_metadata = {
                    "model": raw.get("model") or model,
                    "usage": raw.get("usage") or {},
                    "status": raw.get("status"),
                    "incomplete_details": raw.get("incomplete_details"),
                    "attempt": attempt,
                    "wire_api": wire_api,
                    "structured_output_requested": structured_output_requested,
                }
            else:
                content = raw["choices"][0]["message"]["content"]
                api_metadata = {
                    "model": raw.get("model") or model,
                    "usage": raw.get("usage") or {},
                    "finish_reason": raw["choices"][0].get("finish_reason"),
                    "attempt": attempt,
                    "wire_api": wire_api,
                }
            return json.loads(content), {
                **api_metadata,
            }
        except urllib.error.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if (
                wire_api == "responses"
                and structured_output_requested
                and exc.code in {400, 404, 422, 500, 502}
            ):
                # Some OpenAI-compatible Responses proxies reject
                # ``text.format`` even though plain JSON-only prompting works.
                # Retry the same logical attempt without structured-output
                # parameters; strict local validation still gates ingestion.
                structured_output_requested = False
                continue
            if attempt < retries:
                time.sleep(min(30, 2 * attempt))
            attempt += 1
        except (
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(30, 2 * attempt))
            attempt += 1
    raise RuntimeError(last_error)


def validate_response(
    expected_rows: list[tuple[str, str, str]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews = result.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(expected_rows):
        raise ValueError("response review count does not match input")
    expected_ids = [row[0] for row in expected_rows]
    actual_ids = [str(row.get("claim_id") or "") for row in reviews if isinstance(row, dict)]
    if actual_ids != expected_ids:
        raise ValueError("response claim order or IDs do not match input")

    validated: list[dict[str, Any]] = []
    for row in reviews:
        labels = row.get("claim_case_study_ids")
        if not isinstance(labels, list) or any(not isinstance(value, str) for value in labels):
            raise ValueError("claim_case_study_ids must be a string list")
        unknown = sorted(set(labels) - set(CASE_STUDY_IDS))
        if unknown:
            raise ValueError(f"unknown Case Study IDs: {unknown}")
        normalized = registry_order(labels)
        if len(normalized) != len(labels):
            raise ValueError("duplicate Case Study IDs in response")
        confidence = float(row.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence outside 0..1")
        reason = str(row.get("reason") or "").strip()
        if len(reason) < 12:
            raise ValueError("review reason is missing or too short")
        gates = row.get("gates")
        if not isinstance(gates, dict) or set(gates) != set(GATE_NAMES):
            raise ValueError("review gates are missing or unexpected")
        if any(not isinstance(gates[name], bool) for name in GATE_NAMES):
            raise ValueError("review gates must contain JSON booleans")
        secondary = row.get("needs_secondary_review", False)
        if not isinstance(secondary, bool):
            raise ValueError("needs_secondary_review must be a JSON boolean")
        decision = validate_scope_decision(labels, gates)
        normalized = list(decision.labels)
        normalized_gates = dict(decision.gates)
        validated.append(
            {
                "claim_id": str(row["claim_id"]),
                "claim_case_study_ids": normalized,
                "confidence": confidence,
                "reason": reason,
                "needs_secondary_review": secondary,
                "gates": normalized_gates,
            }
        )
    return validated


def chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    ledger_path = args.output_dir.resolve() / "reaudit.sqlite"
    connection = connect(ledger_path)
    connection.row_factory = sqlite3.Row
    started = time.monotonic()
    completed_this_run = 0
    attempted_this_run = 0
    failures: list[dict[str, Any]] = []
    usage_totals: dict[str, int] = {}

    def persist_reviews(
        reviews: list[dict[str, Any]],
        reviewed_rows: list[tuple[str, str, str]],
        api_metadata: dict[str, Any],
    ) -> None:
        nonlocal completed_this_run
        for key, value in (api_metadata.get("usage") or {}).items():
            if isinstance(value, int):
                usage_totals[key] = usage_totals.get(key, 0) + value
        reviewed_at = utc_now()
        row_by_id = {
            str(claim_id): (str(paper_key), json.loads(payload_json))
            for claim_id, paper_key, payload_json in reviewed_rows
        }
        for review in reviews:
            stored = {
                **review,
                "rubric_version": RUBRIC_VERSION,
                "review_stage": "primary",
                "reviewer_id": f"model:{args.model}",
                "reasoning_effort": args.reasoning_effort,
                "reviewed_at": reviewed_at,
                "api_metadata": api_metadata,
            }
            paper_key, payload = row_by_id[review["claim_id"]]
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
            connection.execute(
                """
                UPDATE claims SET review_status=?, review_json=?, reviewed_at=?
                WHERE claim_id=? AND review_status='pending'
                """,
                (
                    "secondary_pending"
                    if review["needs_secondary_review"]
                    else "final_complete",
                    compact_json(stored),
                    reviewed_at,
                    review["claim_id"],
                ),
            )
        connection.commit()
        completed_this_run += len(reviews)

    def mark_failed(failed_rows: list[tuple[str, str, str]], exc: Exception) -> None:
        failed_at = utc_now()
        error_text = f"{type(exc).__name__}: {exc}"
        failure_review = compact_json(
            {
                "rubric_version": RUBRIC_VERSION,
                "review_stage": "primary",
                "reviewer_id": f"model:{args.model}",
                "reasoning_effort": args.reasoning_effort,
                "failed_at": failed_at,
                "error": error_text,
            }
        )
        connection.executemany(
            """
            UPDATE claims SET review_status='primary_failed', review_json=?,
                reviewed_at=? WHERE claim_id=? AND review_status='pending'
            """,
            ((failure_review, failed_at, row[0]) for row in failed_rows),
        )
        connection.commit()
        failures.append(
            {
                "claim_ids": [row[0] for row in failed_rows],
                "error": error_text,
            }
        )
    try:
        total = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        initial_pending = connection.execute(
            "SELECT COUNT(*) FROM claims WHERE review_status='pending'"
        ).fetchone()[0]
        target = initial_pending if args.max_claims <= 0 else min(initial_pending, args.max_claims)
        while attempted_this_run < target:
            wave_size = min(args.batch_size * args.workers, target - attempted_this_run)
            raw_rows = connection.execute(
                """
                SELECT claim_id, paper_key, payload_json FROM claims
                WHERE review_status='pending' ORDER BY graph_ordinal LIMIT ?
                """,
                (wave_size,),
            ).fetchall()
            rows = [(str(row[0]), str(row[1]), str(row[2])) for row in raw_rows]
            if not rows:
                break
            batches = chunks(rows, args.batch_size)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for batch in batches:
                    payload = review_payload(
                        connection,
                        batch,
                        abstract_chars=args.abstract_chars,
                        paper_context_limit=args.paper_context_limit,
                    )
                    future = executor.submit(
                        request_json,
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        wire_api=args.wire_api,
                        payload=payload,
                        timeout=args.timeout,
                        retries=args.retries,
                        responses_plain_json=getattr(
                            args, "responses_plain_json", False
                        ),
                        transport=getattr(args, "transport", "urllib"),
                    )
                    futures[future] = batch
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        result, api_metadata = future.result()
                        reviews = validate_response(batch, result)
                        persist_reviews(reviews, batch, api_metadata)
                        attempted_this_run += len(reviews)
                        elapsed = time.monotonic() - started
                        rate = completed_this_run / elapsed if elapsed else 0.0
                        print(
                            f"PROGRESS completed={completed_this_run}/{target} "
                            f"attempted={attempted_this_run}/{target} "
                            f"rate={rate:.2f}_claims_s",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if (
                            len(batch) > 1
                            and getattr(args, "split_failed_batches", False)
                        ):
                            print(
                                f"SPLIT_RETRY batch_first={batch[0][0]} "
                                f"size={len(batch)} error={exc}",
                                flush=True,
                            )
                            for row in batch:
                                single = [row]
                                try:
                                    payload = review_payload(
                                        connection,
                                        single,
                                        abstract_chars=args.abstract_chars,
                                        paper_context_limit=args.paper_context_limit,
                                    )
                                    result, api_metadata = request_json(
                                        base_url=args.base_url,
                                        api_key=api_key,
                                        model=args.model,
                                        reasoning_effort=args.reasoning_effort,
                                        wire_api=args.wire_api,
                                        payload=payload,
                                        timeout=args.timeout,
                                        retries=args.retries,
                                        responses_plain_json=getattr(
                                            args, "responses_plain_json", False
                                        ),
                                        transport=getattr(args, "transport", "urllib"),
                                    )
                                    persist_reviews(
                                        validate_response(single, result),
                                        single,
                                        api_metadata,
                                    )
                                except Exception as split_exc:  # noqa: BLE001
                                    mark_failed(single, split_exc)
                                    print(
                                        f"FAILED split_claim={row[0]}: {split_exc}",
                                        flush=True,
                                    )
                            attempted_this_run += len(batch)
                            elapsed = time.monotonic() - started
                            rate = completed_this_run / elapsed if elapsed else 0.0
                            print(
                                f"PROGRESS completed={completed_this_run}/{target} "
                                f"attempted={attempted_this_run}/{target} "
                                f"rate={rate:.2f}_claims_s",
                                flush=True,
                            )
                            continue
                        attempted_this_run += len(batch)
                        mark_failed(batch, exc)
                        print(
                            f"FAILED batch_first={batch[0][0]} size={len(batch)}: {exc}",
                            flush=True,
                        )
            if failures and args.fail_fast:
                break
            if attempted_this_run < target and args.wave_delay > 0:
                print(
                    f"WAVE_DELAY seconds={args.wave_delay:g} "
                    f"attempted={attempted_this_run}/{target}",
                    flush=True,
                )
                time.sleep(args.wave_delay)

        pending_now = connection.execute(
            "SELECT COUNT(*) FROM claims WHERE review_status!='final_complete'"
        ).fetchone()[0]
        status_counts = dict(
            connection.execute(
                "SELECT review_status, COUNT(*) FROM claims GROUP BY review_status"
            ).fetchall()
        )
        report = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "wire_api": args.wire_api,
            "transport": getattr(args, "transport", "urllib"),
            "responses_plain_json": getattr(args, "responses_plain_json", False),
            "wave_delay_seconds": args.wave_delay,
            "workers": args.workers,
            "claims_per_request": args.batch_size,
            "completed_this_run": completed_this_run,
            "attempted_this_run": attempted_this_run,
            "failures": failures,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "total_claims": total,
            "pending_claims": pending_now,
            "completion_percent": round((total - pending_now) * 100 / total, 6),
            "review_status_counts": status_counts,
            "usage": usage_totals,
        }
        report_dir = args.output_dir.resolve() / "run_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        (report_dir / f"primary_{stamp}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--api-key-env", default="FULL_REAUDIT_API_KEY")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--wire-api",
        choices=("chat_completions", "responses"),
        default="chat_completions",
    )
    parser.add_argument(
        "--transport",
        choices=("urllib", "openai_sdk"),
        default="urllib",
        help="HTTP implementation used for model requests.",
    )
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="xhigh"
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-claims", type=int, default=100)
    parser.add_argument(
        "--abstract-chars",
        type=int,
        default=0,
        help="abstract character cap; 0 sends the complete cached abstract",
    )
    parser.add_argument("--paper-context-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--wave-delay",
        type=float,
        default=0.0,
        help="Pause between worker waves to respect sustained proxy rate limits.",
    )
    parser.add_argument(
        "--responses-plain-json",
        action="store_true",
        help=(
            "For Responses-compatible proxies that reject text.format, skip "
            "the structured-output probe and rely on JSON-only prompting plus "
            "strict local validation."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--split-failed-batches",
        action="store_true",
        help=(
            "When a multi-claim request fails, retry each claim separately and "
            "only mark individually failing claims as primary_failed."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
