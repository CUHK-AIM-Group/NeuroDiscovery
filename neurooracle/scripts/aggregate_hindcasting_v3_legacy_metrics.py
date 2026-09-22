"""Re-score the locked Hindcasting v3 outputs with the manuscript's legacy metrics.

This is a diagnostic bridge, not a replacement for the registered v3 endpoint.
The legacy combined-support metric gives a hypothesis-slot credit when either its
exact endpoint or at least one path edge is supported in the future window.

The script does not alter any formal run output.  It reads the completed method
outputs and writes a separate ``legacy_metric_bridge`` directory containing
run-, cell-, and balanced summaries plus a provenance manifest.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


METHODS = ("neurodiscovery", "sciagents", "openscholar_rag")
DISPLAY_NAMES = {
    "neurodiscovery": "NeuroDiscovery",
    "sciagents": "SciAgents",
    "openscholar_rag": "OpenScholar-RAG",
}
EXPECTED_TASKS = ("case1_transdiagnostic", "biomarker_discovery")
EXPECTED_FREEZE_YEARS = (2016, 2017, 2018, 2019, 2020)
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_BUDGETS = (10, 20, 50, 100, 200, 500, 1000)
RANDOM_TRIALS = 1000


@dataclass(frozen=True)
class RunKey:
    method: str
    task: str
    seed: int
    freeze_year: int
    future_start_year: int
    future_end_year: int


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _stable_seed(*parts: Any) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return fmean(materialized) if materialized else math.nan


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return fmean(materialized) if materialized else None


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def _simulate_random_counts(
    *,
    pool_size: int,
    combined_hits: int,
    terminal_combined_hits: int,
    k: int,
    seed_parts: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Draw the old task-matched slot-count null from one evaluated pool."""
    if not 0 <= terminal_combined_hits <= combined_hits <= pool_size:
        raise ValueError(
            "Invalid hit partition: "
            f"terminal={terminal_combined_hits}, combined={combined_hits}, pool={pool_size}"
        )
    draw_size = min(k, pool_size)
    if draw_size == pool_size:
        combined = np.full(RANDOM_TRIALS, combined_hits, dtype=np.float64)
        terminal = np.full(RANDOM_TRIALS, terminal_combined_hits, dtype=np.float64)
        return combined, terminal, "degenerate_full_pool"
    colors = np.asarray(
        [
            terminal_combined_hits,
            combined_hits - terminal_combined_hits,
            pool_size - combined_hits,
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    draws = rng.multivariate_hypergeometric(colors, draw_size, size=RANDOM_TRIALS)
    terminal = draws[:, 0].astype(np.float64)
    combined = (draws[:, 0] + draws[:, 1]).astype(np.float64)
    return combined, terminal, "same_evaluated_pool_random_resampling"


def _run_row(
    *,
    key: RunKey,
    k: int,
    selected: Sequence[Mapping[str, Any]],
    pool_size: int,
    endpoint_field: str,
    endpoint_year_field: str,
    endpoint_key_field: str,
    path_field: str,
    combined_field: str,
    first_year_field: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    terminal_start_year = key.freeze_year + 3
    endpoint_rows = [row for row in selected if _bool(row.get(endpoint_field))]
    path_rows = [row for row in selected if _bool(row.get(path_field))]
    combined_rows = [row for row in selected if _bool(row.get(combined_field))]
    endpoint_keys = {
        str(row.get(endpoint_key_field) or "")
        for row in endpoint_rows
        if str(row.get(endpoint_key_field) or "")
    }
    terminal_endpoint = [
        row
        for row in endpoint_rows
        if _int(row.get(endpoint_year_field), -1) >= terminal_start_year
    ]
    terminal_combined = [
        row
        for row in combined_rows
        if _int(row.get(first_year_field), -1) >= terminal_start_year
    ]
    lead_times = [
        _int(row.get(first_year_field)) - key.freeze_year
        for row in combined_rows
        if str(row.get(first_year_field) or "").strip()
    ]
    row = {
        "method": key.method,
        "task": key.task,
        "seed": key.seed,
        "freeze_year": key.freeze_year,
        "future_start_year": key.future_start_year,
        "future_end_year": key.future_end_year,
        "terminal_start_year": terminal_start_year,
        "k": k,
        "evaluated_slots": len(selected),
        "ranking_pool_size": pool_size,
        "endpoint_hits": len(endpoint_rows),
        "unique_endpoint_discoveries": len(endpoint_keys),
        "any_path_edge_hits": len(path_rows),
        "intermediate_only_hits": len(combined_rows) - len(endpoint_rows),
        "combined_hits": len(combined_rows),
        "combined_hit_rate": len(combined_rows) / len(selected) if selected else 0.0,
        "terminal_endpoint_hits": len(terminal_endpoint),
        "terminal_combined_hits": len(terminal_combined),
        "terminal_combined_hit_rate": (
            len(terminal_combined) / len(selected) if selected else 0.0
        ),
        "combined_lead_time_sum": sum(lead_times),
        "combined_lead_time_n": len(lead_times),
        "mean_combined_lead_time": fmean(lead_times) if lead_times else None,
    }
    return row, np.empty(0), np.empty(0), "deferred"


def _score_prefixes(
    *,
    key: RunKey,
    pool: Sequence[Mapping[str, Any]],
    endpoint_field: str,
    endpoint_year_field: str,
    endpoint_key_field: str,
    path_field: str,
    combined_field: str,
    first_year_field: str,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, str]]:
    rows: list[dict[str, Any]] = []
    combined_trials: dict[int, np.ndarray] = {}
    terminal_trials: dict[int, np.ndarray] = {}
    statuses: dict[int, str] = {}
    terminal_start_year = key.freeze_year + 3
    pool_combined = sum(_bool(row.get(combined_field)) for row in pool)
    pool_terminal = sum(
        _bool(row.get(combined_field))
        and _int(row.get(first_year_field), -1) >= terminal_start_year
        for row in pool
    )
    for k in EXPECTED_BUDGETS:
        selected = pool[: min(k, len(pool))]
        row, _, _, _ = _run_row(
            key=key,
            k=k,
            selected=selected,
            pool_size=len(pool),
            endpoint_field=endpoint_field,
            endpoint_year_field=endpoint_year_field,
            endpoint_key_field=endpoint_key_field,
            path_field=path_field,
            combined_field=combined_field,
            first_year_field=first_year_field,
        )
        random_combined, random_terminal, status = _simulate_random_counts(
            pool_size=len(pool),
            combined_hits=pool_combined,
            terminal_combined_hits=pool_terminal,
            k=k,
            seed_parts=(
                "hindcasting-v3-legacy-bridge",
                key.method,
                key.task,
                key.seed,
                key.freeze_year,
                k,
            ),
        )
        rows.append(row)
        combined_trials[k] = random_combined
        terminal_trials[k] = random_terminal
        statuses[k] = status
    return rows, combined_trials, terminal_trials, statuses


def _load_neurodiscovery(
    formal_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int, int, int], np.ndarray],
    dict[tuple[str, str, int, int, int], np.ndarray],
    dict[str, Any],
]:
    run_rows: list[dict[str, Any]] = []
    random_combined: dict[tuple[str, str, int, int, int], np.ndarray] = {}
    random_terminal: dict[tuple[str, str, int, int, int], np.ndarray] = {}
    path_lengths: Counter[int] = Counter()
    runs = 0
    metric_checks = 0
    for hidden_path in sorted((formal_root / "neurodiscovery").glob("seed_*/*/kg*/hidden_outcomes.csv")):
        run_dir = hidden_path.parent
        metrics = _read_csv(run_dir / "metrics_by_k.csv")
        if not metrics:
            raise ValueError(f"Empty metrics file: {run_dir / 'metrics_by_k.csv'}")
        first = metrics[0]
        key = RunKey(
            method="neurodiscovery",
            task=str(first["case_study_id"]),
            seed=_int(first["seed"]),
            freeze_year=_int(first["freeze_year"]),
            future_start_year=_int(first["future_start_year"]),
            future_end_year=_int(first["future_end_year"]),
        )
        hidden = _read_csv(hidden_path)
        if len(hidden) != 1000:
            raise ValueError(f"Expected 1000 executed hypotheses, found {len(hidden)}: {hidden_path}")
        for row in hidden:
            row["endpoint_hit"] = row.get("primary_hit", "False")
            row["endpoint_year"] = row.get("primary_year", "")
            row["endpoint_discovery_key"] = row.get("primary_discovery_key", "")
            # Every v3 NeuroDiscovery output is a one-edge proposed endpoint.
            # Under the evaluator's path-edge convention, the edge is also the path.
            row["any_path_edge_hit"] = row.get("any_future_hit", "False")
        hit_keys = {
            str(row.get("endpoint_discovery_key") or "")
            for row in hidden
            if _bool(row.get("endpoint_hit"))
        }
        if any(not value.startswith("endpoint:") for value in hit_keys):
            raise ValueError(f"Non-endpoint NeuroDiscovery primary key in {hidden_path}")

        executed = json.loads((run_dir / "executed_hypotheses.json").read_text(encoding="utf-8"))
        hypotheses = list(executed.get("hypotheses") or [])
        if len(hypotheses) != len(hidden):
            raise ValueError(f"Executed/hidden row mismatch in {run_dir}")
        path_lengths.update(len(list(hypothesis.get("path") or [])) for hypothesis in hypotheses)

        scored, combined_trials, terminal_trials, _ = _score_prefixes(
            key=key,
            pool=hidden,
            endpoint_field="endpoint_hit",
            endpoint_year_field="endpoint_year",
            endpoint_key_field="endpoint_discovery_key",
            path_field="any_path_edge_hit",
            combined_field="any_future_hit",
            first_year_field="first_future_year",
        )
        metrics_by_k = {_int(row["requested_k"]): row for row in metrics}
        for row in scored:
            old = metrics_by_k[row["k"]]
            checks = {
                "endpoint_hits": _int(old["full_primary_hits"]),
                "unique_endpoint_discoveries": _int(old["unique_primary_discoveries"]),
                "combined_hits": _int(old["full_any_hits"]),
                "terminal_endpoint_hits": _int(old["terminal_primary_hits"]),
                "terminal_combined_hits": _int(old["terminal_any_hits"]),
            }
            for field, expected in checks.items():
                metric_checks += 1
                if row[field] != expected:
                    raise ValueError(
                        f"NeuroDiscovery bridge mismatch {field}: {row[field]} != {expected} "
                        f"for {key} K={row['k']}"
                    )
            run_rows.append(row)
            trial_key = (key.method, key.task, key.seed, key.freeze_year, row["k"])
            random_combined[trial_key] = combined_trials[row["k"]]
            random_terminal[trial_key] = terminal_trials[row["k"]]
        runs += 1
    audit = {
        "runs": runs,
        "metric_checks": metric_checks,
        "path_length_distribution": {str(k): v for k, v in sorted(path_lengths.items())},
        "all_hypotheses_are_one_edge_endpoints": set(path_lengths) == {1},
    }
    if not audit["all_hypotheses_are_one_edge_endpoints"]:
        raise ValueError(f"Unexpected NeuroDiscovery path lengths: {path_lengths}")
    return run_rows, random_combined, random_terminal, audit


def _load_baselines(
    formal_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int, int, int], np.ndarray],
    dict[tuple[str, str, int, int, int], np.ndarray],
    dict[str, Any],
]:
    run_rows: list[dict[str, Any]] = []
    random_combined: dict[tuple[str, str, int, int, int], np.ndarray] = {}
    random_terminal: dict[tuple[str, str, int, int, int], np.ndarray] = {}
    runs = 0
    metric_checks = 0
    path_lengths: dict[str, Counter[int]] = {method: Counter() for method in METHODS[1:]}
    for method in METHODS[1:]:
        pattern = f"{method}/seed_*/*/kg*/hindcasting/scored_hypotheses.csv"
        for scored_path in sorted((formal_root / "baseline_evaluation").glob(pattern)):
            eval_dir = scored_path.parent
            manifest = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
            seed_name = scored_path.parents[3].name
            key = RunKey(
                method=method,
                task=str(manifest["case_study_id"]),
                seed=int(seed_name.removeprefix("seed_")),
                freeze_year=int(manifest["freeze_year"]),
                future_start_year=int(manifest["future_start_year"]),
                future_end_year=int(manifest["future_end_year"]),
            )
            scored_pool = _read_csv(scored_path)
            scored_pool.sort(key=lambda row: _int(row.get("rank")))
            if len(scored_pool) != 1000:
                raise ValueError(f"Expected baseline pool size 1000: {scored_path}")
            path_lengths[method].update(_int(row.get("path_edges")) for row in scored_pool)
            scored, combined_trials, terminal_trials, _ = _score_prefixes(
                key=key,
                pool=scored_pool,
                endpoint_field="endpoint_hit",
                endpoint_year_field="endpoint_year",
                endpoint_key_field="endpoint_discovery_key",
                path_field="any_path_edge_hit",
                combined_field="any_future_hit",
                first_year_field="first_future_year",
            )
            for row in scored:
                old = manifest["topk"][str(row["k"])]["observed"]
                checks = {
                    "endpoint_hits": int(old["endpoint_hits"]),
                    "unique_endpoint_discoveries": int(old["unique_endpoint_discoveries"]),
                    "any_path_edge_hits": int(old["any_path_edge_hits"]),
                    "combined_hits": int(old["any_future_hits"]),
                }
                for field, expected in checks.items():
                    metric_checks += 1
                    if row[field] != expected:
                        raise ValueError(
                            f"Baseline bridge mismatch {field}: {row[field]} != {expected} "
                            f"for {key} K={row['k']}"
                        )
                run_rows.append(row)
                trial_key = (key.method, key.task, key.seed, key.freeze_year, row["k"])
                random_combined[trial_key] = combined_trials[row["k"]]
                random_terminal[trial_key] = terminal_trials[row["k"]]
            runs += 1
    audit = {
        "runs": runs,
        "metric_checks": metric_checks,
        "path_length_distribution": {
            method: {str(k): v for k, v in sorted(counts.items())}
            for method, counts in path_lengths.items()
        },
    }
    return run_rows, random_combined, random_terminal, audit


COUNT_FIELDS = (
    "endpoint_hits",
    "unique_endpoint_discoveries",
    "any_path_edge_hits",
    "intermediate_only_hits",
    "combined_hits",
    "terminal_endpoint_hits",
    "terminal_combined_hits",
)


def _validate_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = {
        (
            row["method"],
            row["task"],
            int(row["seed"]),
            int(row["freeze_year"]),
            int(row["k"]),
        )
        for row in rows
    }
    expected = {
        (method, task, seed, freeze_year, k)
        for method in METHODS
        for task in EXPECTED_TASKS
        for seed in EXPECTED_SEEDS
        for freeze_year in EXPECTED_FREEZE_YEARS
        for k in EXPECTED_BUDGETS
    }
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra or len(keys) != len(rows):
        raise ValueError(
            f"Bridge matrix mismatch: rows={len(rows)}, keys={len(keys)}, "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        "rows": len(rows),
        "unique_rows": len(keys),
        "expected_rows": len(expected),
        "missing_rows": len(missing),
        "extra_rows": len(extra),
    }


def _aggregate_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["task"], int(row["freeze_year"]), int(row["k"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, task, freeze_year, k), group in sorted(groups.items()):
        if len(group) != len(EXPECTED_SEEDS):
            raise ValueError(f"Expected 10 seeds for {(method, task, freeze_year, k)}")
        lead_sum = sum(float(row["combined_lead_time_sum"]) for row in group)
        lead_n = sum(int(row["combined_lead_time_n"]) for row in group)
        output.append(
            {
                "method": method,
                "task": task,
                "freeze_year": freeze_year,
                "k": k,
                "seeds": len(group),
                **{field: _mean(float(row[field]) for row in group) for field in COUNT_FIELDS},
                "combined_hit_rate": _mean(float(row["combined_hit_rate"]) for row in group),
                "terminal_combined_hit_rate": _mean(
                    float(row["terminal_combined_hit_rate"]) for row in group
                ),
                "pooled_mean_combined_lead_time": lead_sum / lead_n if lead_n else None,
                "combined_lead_time_n": lead_n,
            }
        )
    return output


def _aggregate_balanced(
    run_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    run_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        groups[(row["method"], int(row["k"]))].append(row)
    for row in run_rows:
        run_groups[(row["method"], int(row["k"]))].append(row)
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for k in EXPECTED_BUDGETS:
            group = groups[(method, k)]
            raw = run_groups[(method, k)]
            if len(group) != len(EXPECTED_TASKS) * len(EXPECTED_FREEZE_YEARS):
                raise ValueError(f"Unbalanced cell group for {(method, k)}")
            lead_sum = sum(float(row["combined_lead_time_sum"]) for row in raw)
            lead_n = sum(int(row["combined_lead_time_n"]) for row in raw)
            combined_mean = _mean(float(row["combined_hits"]) for row in group)
            output.append(
                {
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "k": k,
                    "tasks": len(EXPECTED_TASKS),
                    "windows_per_task": len(EXPECTED_FREEZE_YEARS),
                    "seeds_per_cell": len(EXPECTED_SEEDS),
                    **{field: _mean(float(row[field]) for row in group) for field in COUNT_FIELDS},
                    "combined_hit_rate": _mean(float(row["combined_hit_rate"]) for row in group),
                    "terminal_combined_hit_rate": _mean(
                        float(row["terminal_combined_hit_rate"]) for row in group
                    ),
                    "pooled_mean_combined_lead_time": lead_sum / lead_n if lead_n else None,
                    "seed_averaged_task_window_total_combined_hits": (
                        combined_mean * len(EXPECTED_TASKS) * len(EXPECTED_FREEZE_YEARS)
                    ),
                    "all_run_slot_combined_hits": sum(int(row["combined_hits"]) for row in raw),
                }
            )
    return output


def _random_summary(
    balanced: Sequence[Mapping[str, Any]],
    random_combined: Mapping[tuple[str, str, int, int, int], np.ndarray],
    random_terminal: Mapping[tuple[str, str, int, int, int], np.ndarray],
) -> list[dict[str, Any]]:
    observed = {(row["method"], int(row["k"])): row for row in balanced}
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for k in EXPECTED_BUDGETS:
            keys = [
                (method, task, seed, freeze_year, k)
                for task in EXPECTED_TASKS
                for freeze_year in EXPECTED_FREEZE_YEARS
                for seed in EXPECTED_SEEDS
            ]
            combined_matrix = np.vstack([random_combined[key] for key in keys])
            terminal_matrix = np.vstack([random_terminal[key] for key in keys])
            null = combined_matrix.mean(axis=0)
            terminal_null = terminal_matrix.mean(axis=0)
            obs = float(observed[(method, k)]["combined_hits"])
            terminal_obs = float(observed[(method, k)]["terminal_combined_hits"])
            null_mean = float(null.mean())
            terminal_null_mean = float(terminal_null.mean())
            degenerate = k == 1000
            output.append(
                {
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "k": k,
                    "observed_balanced_combined_hits": obs,
                    "random_mean_combined_hits": null_mean,
                    "random_sd_combined_hits": float(null.std(ddof=1)),
                    "random_ci95_combined_low": float(np.quantile(null, 0.025)),
                    "random_ci95_combined_high": float(np.quantile(null, 0.975)),
                    "combined_lift_over_random": obs / null_mean if null_mean > 0 else None,
                    "p_random_ge_observed": (
                        (1 + int(np.count_nonzero(null >= obs - 1e-12)))
                        / (RANDOM_TRIALS + 1)
                    ),
                    "observed_balanced_terminal_combined_hits": terminal_obs,
                    "random_mean_terminal_combined_hits": terminal_null_mean,
                    "terminal_combined_lift_over_random": (
                        terminal_obs / terminal_null_mean if terminal_null_mean > 0 else None
                    ),
                    "p_random_terminal_ge_observed": (
                        (1 + int(np.count_nonzero(terminal_null >= terminal_obs - 1e-12)))
                        / (RANDOM_TRIALS + 1)
                    ),
                    "random_trials": RANDOM_TRIALS,
                    "random_pool": "same method-run evaluated pool of 1000 hypotheses",
                    "diagnostic_status": (
                        "degenerate_full_pool_not_informative"
                        if degenerate
                        else "diagnostic_only"
                    ),
                }
            )
    return output


def _report(
    *,
    output_dir: Path,
    balanced: Sequence[Mapping[str, Any]],
    random_summary: Sequence[Mapping[str, Any]],
    audits: Mapping[str, Any],
) -> Path:
    by_key = {(row["method"], int(row["k"])): row for row in balanced}
    random_by_key = {(row["method"], int(row["k"])): row for row in random_summary}
    lines = [
        "# Hindcasting v3 legacy-metric bridge",
        "",
        "> Diagnostic re-scoring only. This does not replace the locked v3 primary endpoint.",
        "",
        "The legacy combined-support metric counts a ranked hypothesis slot when its exact "
        "endpoint or any path edge receives future support. `intermediate_only_hits` is the "
        "additional credit contributed by path support when the exact endpoint is not supported.",
        "",
        "## Balanced five-window results",
        "",
        "Values are averaged across 10 seeds within task-window, across five windows within "
        "task, and then equally across the two tasks.",
        "",
        "| Method | Endpoint @100 | Intermediate-only @100 | Combined @100 | Terminal combined @100 | Endpoint @1000 | Intermediate-only @1000 | Combined @1000 | Terminal combined @1000 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        k100 = by_key[(method, 100)]
        k1000 = by_key[(method, 1000)]
        lines.append(
            "| {name} | {e100} | {i100} | {c100} | {t100} | {e1000} | {i1000} | {c1000} | {t1000} |".format(
                name=DISPLAY_NAMES[method],
                e100=_fmt(float(k100["endpoint_hits"])),
                i100=_fmt(float(k100["intermediate_only_hits"])),
                c100=_fmt(float(k100["combined_hits"])),
                t100=_fmt(float(k100["terminal_combined_hits"])),
                e1000=_fmt(float(k1000["endpoint_hits"])),
                i1000=_fmt(float(k1000["intermediate_only_hits"])),
                c1000=_fmt(float(k1000["combined_hits"])),
                t1000=_fmt(float(k1000["terminal_combined_hits"])),
            )
        )
    lines.extend(
        [
            "",
            "## Old-style random-resampling diagnostic at K=100",
            "",
            "The null samples 100 slots without replacement from each method-run's own evaluated "
            "pool of 1,000 hypotheses, then applies the same seed/window/task balancing. It tests "
            "within-pool ranking enrichment, not whether one method generated a better pool.",
            "",
            "| Method | Observed combined | Random mean | Lift | Empirical P | Terminal observed | Terminal random | Terminal lift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = random_by_key[(method, 100)]
        lines.append(
            "| {name} | {obs} | {rand} | {lift} | {p} | {tobs} | {trand} | {tlift} |".format(
                name=DISPLAY_NAMES[method],
                obs=_fmt(float(row["observed_balanced_combined_hits"])),
                rand=_fmt(float(row["random_mean_combined_hits"])),
                lift=_fmt(_float_or_none(row["combined_lift_over_random"]), 2),
                p=_fmt(float(row["p_random_ge_observed"]), 6),
                tobs=_fmt(float(row["observed_balanced_terminal_combined_hits"])),
                trand=_fmt(float(row["random_mean_terminal_combined_hits"])),
                tlift=_fmt(_float_or_none(row["terminal_combined_lift_over_random"]), 2),
            )
        )
    lines.extend(
        [
            "",
            "At K=1,000 the evaluated pool also contains exactly 1,000 hypotheses for every "
            "method-run. Randomly drawing the full pool is identical to the observed pool, so lift "
            "is mechanically 1 and the random test is not informative.",
            "",
            "## Count-scale reconciliation",
            "",
            "| Method | K | Balanced hits per run | Seed-averaged total over 10 task-windows | Total across all 100 run cells |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for k in (100, 1000):
            row = by_key[(method, k)]
            lines.append(
                f"| {DISPLAY_NAMES[method]} | {k} | "
                f"{_fmt(float(row['combined_hits']))} | "
                f"{_fmt(float(row['seed_averaged_task_window_total_combined_hits']))} | "
                f"{int(row['all_run_slot_combined_hits'])} |"
            )
    lines.extend(
        [
            "",
            "The manuscript's former totals (487 at K=100 and 4,701 at K=1,000) pooled four "
            "tasks and four cutoffs and used a much broader path-rich candidate representation. "
            "The closest scale here is the seed-averaged task-window total, not the 100-run total.",
            "",
            "## Validation and limitations",
            "",
            f"- Reconstructed run-budget rows: {audits['matrix']['rows']} / {audits['matrix']['expected_rows']}.",
            f"- NeuroDiscovery current-metric cross-checks: {audits['neurodiscovery']['metric_checks']} passed.",
            f"- Baseline evaluator cross-checks: {audits['baselines']['metric_checks']} passed.",
            "- All 100,000 executed NeuroDiscovery hypotheses contain a one-edge endpoint path; "
            "therefore legacy combined support adds no intermediate-only credit for NeuroDiscovery.",
            "- Candidate representations are structurally different: NeuroDiscovery has one edge, "
            "SciAgents has two edges, and OpenScholar-RAG has no stored path edge in every evaluated "
            "hypothesis. Any-edge credit therefore gives methods different numbers of chances to hit.",
            "- NeuroDiscovery's 1,000-candidate random pool is the dynamically executed sequence, "
            "including candidates proposed after early feedback. Its lift is a post-loop ranking "
            "diagnostic, not a leakage-free historical null.",
            "- The baseline names in this formal run denote deterministic adapters; no LLM API "
            "was called.",
            "- Same-pool random lift measures ranking only. It must not be used as the primary "
            "three-method comparison, especially at K=1,000 where it is degenerate.",
            "",
        ]
    )
    report_path = output_dir / "LEGACY_METRIC_BRIDGE.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(formal_root: Path, output_dir: Path) -> dict[str, Any]:
    nd_rows, nd_random, nd_terminal_random, nd_audit = _load_neurodiscovery(formal_root)
    baseline_rows, baseline_random, baseline_terminal_random, baseline_audit = _load_baselines(
        formal_root
    )
    run_rows = [*nd_rows, *baseline_rows]
    random_combined = {**nd_random, **baseline_random}
    random_terminal = {**nd_terminal_random, **baseline_terminal_random}
    matrix = _validate_matrix(run_rows)
    cell_rows = _aggregate_cells(run_rows)
    balanced = _aggregate_balanced(run_rows, cell_rows)
    random_rows = _random_summary(balanced, random_combined, random_terminal)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "legacy_metric_run_budget.csv"
    cell_path = output_dir / "legacy_metric_task_window.csv"
    balanced_path = output_dir / "legacy_metric_balanced_summary.csv"
    random_path = output_dir / "legacy_metric_random_diagnostics.csv"
    _write_csv(run_path, run_rows)
    _write_csv(cell_path, cell_rows)
    _write_csv(balanced_path, balanced)
    _write_csv(random_path, random_rows)

    audits = {
        "matrix": matrix,
        "neurodiscovery": nd_audit,
        "baselines": baseline_audit,
    }
    report_path = _report(
        output_dir=output_dir,
        balanced=balanced,
        random_summary=random_rows,
        audits=audits,
    )
    manifest = {
        "schema_version": "hindcasting-v3-legacy-metric-bridge.v1",
        "status": "complete_diagnostic_not_registered_primary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "formal_root": str(formal_root.resolve()),
        "metric_definition": {
            "endpoint_hits": "hypothesis slots with exact future endpoint support",
            "any_path_edge_hits": "hypothesis slots with at least one future-supported path edge",
            "intermediate_only_hits": "combined-support slots without exact endpoint support",
            "combined_hits": "union of endpoint and any-path-edge support (legacy manuscript metric)",
            "terminal_combined_hits": "combined support first appearing in the last three future years",
        },
        "aggregation": (
            "Average 10 seeds within task-window; average five windows within task; "
            "weight the two tasks equally."
        ),
        "random_resampling": {
            "trials": RANDOM_TRIALS,
            "pool": "same method-run evaluated pool of 1000 hypotheses",
            "interpretation": "diagnostic ranking enrichment only",
            "k_1000": "degenerate because K equals pool size",
        },
        "audits": audits,
        "outputs": {},
    }
    manifest_path = output_dir / "manifest.json"
    for path in (run_path, cell_path, balanced_path, random_path, report_path):
        manifest["outputs"][path.name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else formal_root / "legacy_metric_bridge"
    )
    manifest = run(formal_root, output_dir)
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(output_dir),
        "matrix": manifest["audits"]["matrix"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
