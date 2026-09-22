"""Replay compact API decisions against immutable human-finalized claims.

This is a read-only quality gate.  It does not update the ledger or formal KG.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from neurooracle.scripts.benchmark_compact_case_study_reaudit import (
    compact_payload,
    execute_batch,
    paper_aware_batches,
)
from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    compact_json,
    connect,
)
from neurooracle.scripts.run_full_graph_case_study_reaudit import GATE_NAMES


def set_metrics(
    predicted: list[set[str]], gold: list[set[str]]
) -> dict[str, float | int]:
    true_positive = sum(len(p & g) for p, g in zip(predicted, gold, strict=True))
    false_positive = sum(len(p - g) for p, g in zip(predicted, gold, strict=True))
    false_negative = sum(len(g - p) for p, g in zip(predicted, gold, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    output_dir = args.output_dir.resolve()
    connection = connect(output_dir / "reaudit.sqlite")
    connection.row_factory = None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    replay_dir = output_dir / "quality_replays" / timestamp
    replay_dir.mkdir(parents=True, exist_ok=False)
    try:
        selected = connection.execute(
            """
            SELECT claim_id, paper_key, payload_json, review_json, graph_ordinal
            FROM claims
            WHERE review_status='final_complete'
              AND json_extract(review_json, '$.reviewer_id') IN (
                  'codex_host_manual', 'codex_host_manual_secondary'
              )
            ORDER BY reviewed_at DESC, graph_ordinal DESC
            LIMIT ?
            """,
            (args.sample_claims,),
        ).fetchall()
        selected.sort(key=lambda row: (str(row[1]), int(row[4])))
        rows = [(str(row[0]), str(row[1]), str(row[2])) for row in selected]
        gold_by_id = {str(row[0]): json.loads(str(row[3])) for row in selected}
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
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(execute_batch, batch, payload, api_args): batch
                for batch, payload in prepared
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed % max(1, len(futures) // 4) == 0:
                    print(f"QUALITY requests={completed}/{len(futures)}", flush=True)
        elapsed = time.monotonic() - started
        with (replay_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(compact_json(result) + "\n")

        reviews = [
            review for result in results if result["ok"] for review in result["reviews"]
        ]
        failures = [result for result in results if not result["ok"]]
        label_predicted: list[set[str]] = []
        label_gold: list[set[str]] = []
        gate_predicted: list[set[str]] = []
        gate_gold: list[set[str]] = []
        exact_labels = 0
        exact_gates = 0
        exact_decisions = 0
        direct_total = 0
        direct_exact = 0
        disagreements: list[dict[str, Any]] = []
        for review in reviews:
            gold = gold_by_id[review["claim_id"]]
            predicted_labels = set(review["claim_case_study_ids"])
            gold_labels = set(gold["claim_case_study_ids"])
            predicted_gates = {
                name for name, value in review["gates"].items() if value
            }
            gold_gates = {name for name in GATE_NAMES if gold["gates"].get(name, False)}
            labels_match = predicted_labels == gold_labels
            gates_match = predicted_gates == gold_gates
            exact_labels += labels_match
            exact_gates += gates_match
            exact_decisions += labels_match and gates_match
            if not review["needs_secondary_review"]:
                direct_total += 1
                direct_exact += labels_match and gates_match
            if not (labels_match and gates_match) and len(disagreements) < 100:
                disagreements.append(
                    {
                        "claim_id": review["claim_id"],
                        "predicted_labels": sorted(predicted_labels),
                        "gold_labels": sorted(gold_labels),
                        "predicted_gates": sorted(predicted_gates),
                        "gold_gates": sorted(gold_gates),
                        "confidence": review["confidence"],
                        "secondary": review["needs_secondary_review"],
                        "reason": review["reason"],
                    }
                )
            label_predicted.append(predicted_labels)
            label_gold.append(gold_labels)
            gate_predicted.append(predicted_gates)
            gate_gold.append(gold_gates)
        valid = len(reviews)
        requested = len(rows)
        report = {
            "schema": "compact_case_study_reaudit_quality_replay.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "workers": args.workers,
            "claims_per_request": args.claims_per_request,
            "requested_claims": requested,
            "valid_claims": valid,
            "failed_claims": requested - valid,
            "technical_failure_percent": round((requested - valid) * 100 / requested, 4),
            "secondary_claims": sum(review["needs_secondary_review"] for review in reviews),
            "secondary_percent": round(
                sum(review["needs_secondary_review"] for review in reviews) * 100 / valid, 4
            ) if valid else 0.0,
            "exact_label_percent": round(exact_labels * 100 / valid, 4) if valid else 0.0,
            "exact_gate_percent": round(exact_gates * 100 / valid, 4) if valid else 0.0,
            "exact_decision_percent": round(exact_decisions * 100 / valid, 4) if valid else 0.0,
            "direct_nonsecondary_claims": direct_total,
            "direct_exact_decision_percent": round(direct_exact * 100 / direct_total, 4) if direct_total else 0.0,
            "label_micro": set_metrics(label_predicted, label_gold),
            "gate_micro": set_metrics(gate_predicted, gate_gold),
            "elapsed_seconds": round(elapsed, 3),
            "claims_per_hour": round(valid * 3600 / elapsed, 2) if elapsed else 0.0,
            "thresholds": {
                "min_direct_exact_decision_percent": args.min_direct_exact_decision_percent,
                "min_label_micro_f1": args.min_label_micro_f1,
                "max_technical_failure_percent": args.max_technical_failure_percent,
            },
            "quality_gate_passed": bool(
                valid
                and (requested - valid) * 100 / requested <= args.max_technical_failure_percent
                and direct_total
                and direct_exact * 100 / direct_total >= args.min_direct_exact_decision_percent
                and set_metrics(label_predicted, label_gold)["f1"] >= args.min_label_micro_f1
            ),
            "failure_examples": [
                {"claim_ids": result["claim_ids"], "error": result["error"]}
                for result in failures[:10]
            ],
            "disagreement_examples": disagreements,
            "formal_kg_mutated": False,
            "ledger_mutated": False,
        }
        (replay_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--transport", choices=("urllib", "openai_sdk"), default="urllib")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--claims-per-request", type=int, default=12)
    parser.add_argument("--sample-claims", type=int, default=384)
    parser.add_argument("--abstract-chars", type=int, default=2500)
    parser.add_argument("--paper-context-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--min-direct-exact-decision-percent", type=float, default=95.0)
    parser.add_argument("--min-label-micro-f1", type=float, default=0.98)
    parser.add_argument("--max-technical-failure-percent", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
