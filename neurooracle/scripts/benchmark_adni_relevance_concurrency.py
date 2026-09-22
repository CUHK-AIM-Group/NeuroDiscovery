from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import annotate_adni_relevance_batch as annotate


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def call_one(
    candidate: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result, metadata = annotate._request_json(
            base_url=base_url,
            api_key=api_key,
            model=model,
            candidate=candidate,
            timeout=timeout,
            retries=retries,
        )
        rows = annotate._validate(candidate, result)
        return {
            "hypothesis_id": str(candidate["id"]),
            "ok": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "paper_assessments": len(rows),
            "metadata": metadata,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "hypothesis_id": str(candidate["id"]),
            "ok": False,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_level(
    candidates: list[dict[str, Any]],
    workers: int,
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    print(f"LEVEL_START workers={workers} requests={len(candidates)}", flush=True)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                call_one,
                candidate,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
            )
            for candidate in candidates
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                f"LEVEL_PROGRESS workers={workers} "
                f"done={len(results)}/{len(candidates)} ok={row['ok']} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    successful = [row for row in results if row["ok"]]
    latencies = [float(row["elapsed_seconds"]) for row in successful]
    prompt_tokens = sum(
        int(row.get("metadata", {}).get("usage", {}).get("prompt_tokens") or 0)
        for row in successful
    )
    completion_tokens = sum(
        int(row.get("metadata", {}).get("usage", {}).get("completion_tokens") or 0)
        for row in successful
    )
    summary = {
        "workers": workers,
        "requests": len(candidates),
        "successes": len(successful),
        "failures": len(results) - len(successful),
        "success_rate": round(len(successful) / len(candidates), 4),
        "wall_seconds": round(elapsed, 3),
        "throughput_requests_per_minute": round(60 * len(successful) / elapsed, 3),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 3) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "errors": [row["error"] for row in results if not row["ok"]],
        "results": sorted(results, key=lambda row: row["hypothesis_id"]),
    }
    print(
        f"LEVEL_DONE workers={workers} successes={len(successful)} "
        f"failures={summary['failures']} wall_seconds={summary['wall_seconds']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--levels", default="1,2,4,8,16")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/user_study/api_batches/concurrency_benchmark.json",
    )
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    levels = [int(value.strip()) for value in args.levels.split(",") if value.strip()]
    if any(level < 1 for level in levels):
        raise ValueError("All concurrency levels must be positive")

    bank = annotate._load_json(annotate.DEFAULT_BANK)
    truth = annotate._load_json(annotate.DEFAULT_TRUTH)
    candidates = annotate._select_candidates(
        bank,
        truth,
        annotate._processed_ids(annotate.DEFAULT_ANNOTATIONS),
        args.requests,
        "any",
    )
    if len(candidates) != args.requests:
        raise RuntimeError(f"Only {len(candidates)} unprocessed hypotheses are available")

    payload = {
        "schema_version": "1.0",
        "base_url": args.base_url,
        "model": args.model,
        "requests_per_level": len(candidates),
        "retries_per_request": args.retries,
        "candidate_ids": [str(candidate["id"]) for candidate in candidates],
        "started_at_unix": time.time(),
        "levels": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for workers in levels:
        payload["levels"].append(run_level(candidates, workers, args, api_key))
        payload["updated_at_unix"] = time.time()
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["completed_at_unix"] = time.time()
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "levels": levels}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
