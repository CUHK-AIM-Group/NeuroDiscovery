"""Replay five specialized compact auditors against immutable human decisions."""

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
    paper_aware_batches,
)
from neurooracle.scripts.grouped_compact_case_study_reaudit import (
    GROUP_LABELS,
    execute_group_batch,
    merge_group_reviews,
)
from neurooracle.scripts.prepare_full_graph_case_study_reaudit import (
    DEFAULT_OUTPUT_DIR,
    compact_json,
    connect,
)
from neurooracle.scripts.replay_compact_case_study_reaudit_quality import set_metrics
from neurooracle.scripts.run_full_graph_case_study_reaudit import GATE_NAMES


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    output_dir = args.output_dir.resolve()
    connection = connect(output_dir / "reaudit.sqlite")
    connection.row_factory = None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    replay_dir = output_dir / "quality_replays_grouped" / stamp
    replay_dir.mkdir(parents=True, exist_ok=False)
    try:
        selected = connection.execute(
            """
            SELECT claim_id, paper_key, payload_json, review_json, graph_ordinal
            FROM claims
            WHERE review_status='final_complete'
              AND json_extract(review_json, '$.reviewer_id') IN
                  ('codex_host_manual', 'codex_host_manual_secondary')
            ORDER BY reviewed_at DESC, graph_ordinal DESC LIMIT ?
            """,
            (args.sample_claims,),
        ).fetchall()
        selected.sort(key=lambda row: (str(row[1]), int(row[4])))
        rows = [(str(row[0]), str(row[1]), str(row[2])) for row in selected]
        gold = {str(row[0]): json.loads(str(row[3])) for row in selected}
        batches = paper_aware_batches(rows, args.claims_per_request)
        payloads = [
            compact_payload(
                connection,
                batch,
                abstract_chars=args.abstract_chars,
                paper_context_limit=args.paper_context_limit,
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
            transport=args.transport,
            max_output_tokens=args.max_output_tokens,
        )
        grouped_results: dict[int, dict[str, dict[str, Any]]] = {
            index: {} for index in range(len(batches))
        }
        started = time.monotonic()
        total_requests = len(batches) * len(GROUP_LABELS)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    execute_group_batch, group, batches[index], payloads[index], api_args
                ): (index, group)
                for index in range(len(batches))
                for group in GROUP_LABELS
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index, group = futures[future]
                grouped_results[index][group] = future.result()
                if completed % max(1, total_requests // 4) == 0:
                    print(f"GROUPED_QUALITY requests={completed}/{total_requests}", flush=True)
        elapsed = time.monotonic() - started

        reviews: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        with (replay_dir / "specialist_results.jsonl").open("w", encoding="utf-8") as handle:
            for index, batch in enumerate(batches):
                record = {"batch_index": index, "groups": grouped_results[index]}
                handle.write(compact_json(record) + "\n")
                try:
                    reviews.extend(merge_group_reviews(batch, grouped_results[index]))
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "batch_index": index,
                            "claim_ids": [row[0] for row in batch],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        label_predicted: list[set[str]] = []
        label_gold: list[set[str]] = []
        gate_predicted: list[set[str]] = []
        gate_gold: list[set[str]] = []
        exact_labels = exact_gates = exact_decisions = 0
        secondary = 0
        for review in reviews:
            target = gold[review["claim_id"]]
            pl = set(review["claim_case_study_ids"])
            gl = set(target["claim_case_study_ids"])
            pg = {name for name, value in review["gates"].items() if value}
            gg = {name for name in GATE_NAMES if target["gates"].get(name, False)}
            exact_labels += pl == gl
            exact_gates += pg == gg
            exact_decisions += pl == gl and pg == gg
            secondary += review["needs_secondary_review"]
            label_predicted.append(pl)
            label_gold.append(gl)
            gate_predicted.append(pg)
            gate_gold.append(gg)
        valid = len(reviews)
        report = {
            "schema": "grouped_compact_case_study_reaudit_quality_replay.v1",
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "groups": list(GROUP_LABELS),
            "requested_claims": len(rows),
            "valid_claims": valid,
            "failed_claims": len(rows) - valid,
            "secondary_percent": round(secondary * 100 / valid, 4) if valid else 0.0,
            "exact_label_percent": round(exact_labels * 100 / valid, 4) if valid else 0.0,
            "exact_gate_percent": round(exact_gates * 100 / valid, 4) if valid else 0.0,
            "exact_decision_percent": round(exact_decisions * 100 / valid, 4) if valid else 0.0,
            "label_micro": set_metrics(label_predicted, label_gold),
            "gate_micro": set_metrics(gate_predicted, gate_gold),
            "elapsed_seconds": round(elapsed, 3),
            "claims_per_hour": round(valid * 3600 / elapsed, 2) if elapsed else 0.0,
            "failures": failures,
            "ledger_mutated": False,
            "formal_kg_mutated": False,
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
    parser.add_argument("--sample-claims", type=int, default=192)
    parser.add_argument("--abstract-chars", type=int, default=2500)
    parser.add_argument("--paper-context-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--transport-retries", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
