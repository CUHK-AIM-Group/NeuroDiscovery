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


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _call(
    candidate: dict[str, Any],
    *,
    args: argparse.Namespace,
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result, metadata = annotate._request_json(
            base_url=args.base_url,
            api_key=api_key,
            model=model,
            candidate=candidate,
            timeout=args.timeout,
            retries=args.retries,
            reasoning_effort=reasoning_effort,
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


def _run_level(
    candidates: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    api_key: str,
    model: str,
    reasoning_effort: str,
    workers: int,
) -> dict[str, Any]:
    selected = candidates[:workers]
    print(
        f"LEVEL_START model={model} effort={reasoning_effort} workers={workers} requests={len(selected)}",
        flush=True,
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _call,
                candidate,
                args=args,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            for candidate in selected
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"LEVEL_PROGRESS model={model} effort={reasoning_effort} workers={workers} "
                f"done={len(rows)}/{len(selected)} ok={row['ok']} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )
    wall = time.perf_counter() - started
    successful = [row for row in rows if row["ok"]]
    latencies = [float(row["elapsed_seconds"]) for row in successful]
    result = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "workers": workers,
        "requests": len(selected),
        "candidate_ids": [str(candidate["id"]) for candidate in selected],
        "successes": len(successful),
        "failures": len(rows) - len(successful),
        "success_rate": round(len(successful) / len(selected), 4),
        "wall_seconds": round(wall, 3),
        "throughput_hypotheses_per_minute": round(60 * len(successful) / wall, 3),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": round(_percentile(latencies, 0.50), 3) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "errors": [row["error"] for row in rows if not row["ok"]],
        "results": sorted(rows, key=lambda row: row["hypothesis_id"]),
    }
    print(
        f"LEVEL_DONE model={model} effort={reasoning_effort} workers={workers} "
        f"success={len(successful)}/{len(selected)} wall={result['wall_seconds']}s",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:9449/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--model-specs",
        default="gpt-5.5:high,gpt-5.6-terra:xhigh,gpt-5.6-sol:medium",
    )
    parser.add_argument("--levels", default="1,2,4,8,16")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/user_study/api_batches/adni_real_api_matrix.json",
    )
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    model_specs: list[tuple[str, str]] = []
    for value in args.model_specs.split(","):
        model, separator, effort = value.strip().partition(":")
        if not separator or not model or not effort:
            raise ValueError(f"Invalid model spec: {value!r}")
        model_specs.append((model, effort))
    levels = [int(value) for value in args.levels.split(",") if value.strip()]
    max_requests = max(levels)

    bank = annotate._load_json(annotate.DEFAULT_BANK)
    truth = annotate._load_json(annotate.DEFAULT_TRUTH)
    candidates = annotate._select_candidates(
        bank,
        truth,
        annotate._processed_ids(annotate.DEFAULT_ANNOTATIONS),
        max_requests,
        "any",
    )
    if len(candidates) != max_requests:
        raise RuntimeError(f"Only {len(candidates)} unprocessed hypotheses are available")

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "workload": "real_adni_hypothesis_with_five_literature_records",
        "base_url": args.base_url,
        "model_specs": [
            {"model": model, "reasoning_effort": effort}
            for model, effort in model_specs
        ],
        "levels": levels,
        "timeout_seconds": args.timeout,
        "retries_per_request": args.retries,
        "candidate_pool_ids": [str(candidate["id"]) for candidate in candidates],
        "started_at_unix": time.time(),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model, reasoning_effort in model_specs:
        for workers in levels:
            payload["results"].append(
                _run_level(
                    candidates,
                    args=args,
                    api_key=api_key,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    workers=workers,
                )
            )
            payload["updated_at_unix"] = time.time()
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["completed_at_unix"] = time.time()
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
