from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotate_adni_relevance_batch as annotate


ROOT = Path(__file__).resolve().parents[2]
PROGRESS_PATH = ROOT / "outputs/user_study/api_batches/remaining_campaign_progress.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _counts() -> tuple[int, int]:
    bank = annotate._load_json(annotate.DEFAULT_BANK)
    processed = annotate._processed_ids(annotate.DEFAULT_ANNOTATIONS)
    ids = [str(row["id"]) for row in bank.get("hypotheses", []) if isinstance(row, dict)]
    completed = sum(candidate_id in processed for candidate_id in ids)
    return len(ids), completed


def _existing_campaign_state() -> tuple[int, set[str]]:
    campaign_batches: set[int] = set()
    candidate_ids: set[str] = set()
    pattern = re.compile(r"_api_remaining_(\d+)\.json$")
    for path in (
        ROOT / "neurooracle/data/user_study"
    ).glob("adni_manual_relevance_annotations_v1_batch_*_api_remaining_*.json"):
        match = pattern.search(path.name)
        if not match:
            continue
        payload = annotate._load_json(path)
        assignments = payload.get("assignments")
        if not isinstance(assignments, dict):
            continue
        campaign_batches.add(int(match.group(1)))
        candidate_ids.update(str(candidate_id) for candidate_id in assignments)
    highest_batch = max(campaign_batches, default=0)
    if campaign_batches and campaign_batches != set(range(1, highest_batch + 1)):
        raise RuntimeError("Campaign batch files are not contiguous; refusing an unsafe resume")
    return highest_batch, candidate_ids


def _batch_namespace(
    args: argparse.Namespace,
    campaign_batch: int,
    batch_size: int,
) -> Namespace:
    absolute_batch = args.start_batch_number + campaign_batch - 1
    batch_id = f"batch{absolute_batch:02d}_api_remaining_{campaign_batch:03d}"
    return Namespace(
        bank=annotate.DEFAULT_BANK,
        truth=annotate.DEFAULT_TRUTH,
        annotations=annotate.DEFAULT_ANNOTATIONS,
        batch_size=batch_size,
        workers=args.workers,
        status="any",
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        retries=args.retries,
        batch_id=batch_id,
        output=ROOT
        / "neurooracle/data/user_study"
        / f"adni_manual_relevance_annotations_v1_batch_{absolute_batch:02d}_api_remaining_{campaign_batch:03d}.json",
        raw_output=ROOT
        / "outputs/user_study/api_batches"
        / f"{batch_id}_raw.json",
    )


def run(args: argparse.Namespace) -> None:
    if not os.environ.get(args.api_key_env, "").strip():
        raise RuntimeError(f"{args.api_key_env} is not set")
    total, completed_now = _counts()
    campaign_batch, existing_campaign_ids = _existing_campaign_state()
    completed_before = completed_now - len(existing_campaign_ids)
    campaign_total = total - completed_before
    total_batches = math.ceil(campaign_total / args.batch_size)
    started_at = time.time()

    while True:
        total_now, completed_now = _counts()
        remaining = total_now - completed_now
        if remaining <= 0:
            _write_json_atomic(
                args.progress,
                {
                    "status": "complete",
                    "campaign_completed_batches": campaign_batch,
                    "campaign_total_batches": total_batches,
                    "campaign_completed_hypotheses": campaign_total,
                    "campaign_total_hypotheses": campaign_total,
                    "overall_completed_hypotheses": completed_now,
                    "overall_total_hypotheses": total_now,
                    "percent": 100.0,
                    "started_at_unix": started_at,
                    "updated_at_unix": time.time(),
                },
            )
            print("CAMPAIGN COMPLETE", flush=True)
            return

        campaign_batch += 1
        current_size = min(args.batch_size, remaining)
        batch_args = _batch_namespace(args, campaign_batch, current_size)
        _write_json_atomic(
            args.progress,
            {
                "status": "running",
                "current_batch": campaign_batch,
                "campaign_completed_batches": campaign_batch - 1,
                "campaign_total_batches": total_batches,
                "campaign_completed_hypotheses": completed_now - completed_before,
                "campaign_total_hypotheses": campaign_total,
                "overall_completed_hypotheses": completed_now,
                "overall_total_hypotheses": total_now,
                "percent": round(100.0 * (completed_now - completed_before) / campaign_total, 2),
                "started_at_unix": started_at,
                "updated_at_unix": time.time(),
            },
        )
        try:
            result = annotate.run(batch_args)
        except Exception as exc:
            _write_json_atomic(
                args.progress,
                {
                    "status": "failed",
                    "current_batch": campaign_batch,
                    "campaign_completed_batches": campaign_batch - 1,
                    "campaign_total_batches": total_batches,
                    "campaign_completed_hypotheses": completed_now - completed_before,
                    "campaign_total_hypotheses": campaign_total,
                    "overall_completed_hypotheses": completed_now,
                    "overall_total_hypotheses": total_now,
                    "percent": round(100.0 * (completed_now - completed_before) / campaign_total, 2),
                    "error": str(exc),
                    "started_at_unix": started_at,
                    "updated_at_unix": time.time(),
                },
            )
            raise
        total_after, completed_after = _counts()
        _write_json_atomic(
            args.progress,
            {
                "status": "running",
                "last_completed_batch": campaign_batch,
                "campaign_completed_batches": campaign_batch,
                "campaign_total_batches": total_batches,
                "campaign_completed_hypotheses": completed_after - completed_before,
                "campaign_total_hypotheses": campaign_total,
                "overall_completed_hypotheses": completed_after,
                "overall_total_hypotheses": total_after,
                "percent": round(100.0 * (completed_after - completed_before) / campaign_total, 2),
                "last_batch_result": result,
                "started_at_unix": started_at,
                "updated_at_unix": time.time(),
            },
        )
        print(
            f"PROGRESS {campaign_batch}/{total_batches} "
            f"{completed_after - completed_before}/{campaign_total}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--start-batch-number", type=int, default=12)
    parser.add_argument("--progress", type=Path, default=PROGRESS_PATH)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
