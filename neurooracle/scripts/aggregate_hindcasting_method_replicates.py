"""Aggregate multi-method, multi-seed hindcasting outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def collect(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in roots:
        for path in sorted(root.glob("*/seed_*/*/kg*_to_*_*/hindcasting/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed_part = next(part for part in path.parts if part.startswith("seed_"))
            seed = int(seed_part.split("_", 1)[1])
            for raw_k, result in (payload.get("topk") or {}).items():
                observed = result.get("observed") or {}
                random_row = result.get("random_same_hypothesis_pool") or {}
                rows.append({
                    "method": payload.get("method"),
                    "seed": seed,
                    "case_study_id": payload.get("case_study_id"),
                    "freeze_year": payload.get("freeze_year"),
                    "future_start_year": payload.get("future_start_year"),
                    "future_end_year": payload.get("future_end_year"),
                    "k": int(raw_k),
                    "n_hypotheses": payload.get("n_hypotheses"),
                    "future_unique_pairs": (payload.get("future_stats") or {}).get("future_unique_pairs", 0),
                    "primary_hits": observed.get("primary_hits", 0),
                    "primary_hit_rate": observed.get("primary_hit_rate", 0.0),
                    "endpoint_hits": observed.get("endpoint_hits", 0),
                    "any_future_hits": observed.get("any_future_hits", 0),
                    "mean_primary_lead_time": observed.get("mean_primary_lead_time"),
                    "random_applicable": bool(random_row.get("applicable")),
                    "random_mean_primary_hits": random_row.get("mean_primary_hits"),
                    "p_primary_hits_ge_observed": random_row.get("p_primary_hits_ge_observed"),
                    "metrics_path": str(path),
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no metrics.json files found")
    return frame.sort_values(
        ["method", "seed", "case_study_id", "freeze_year", "k"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_replicates(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    per_seed = (
        frame.groupby([*group, "seed"], as_index=False)
        .agg(
            primary_hits=("primary_hits", "sum"),
            endpoint_hits=("endpoint_hits", "sum"),
            any_future_hits=("any_future_hits", "sum"),
            evaluated=("k", "sum"),
            windows=("freeze_year", "size"),
        )
    )
    rows: list[dict[str, object]] = []
    for keys, part in per_seed.groupby(group, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group, key_values))
        row["n_seeds"] = int(part["seed"].nunique())
        row["windows_per_seed"] = float(part["windows"].mean())
        for column in ("primary_hits", "endpoint_hits", "any_future_hits"):
            values = part[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_ci95_low"] = float(np.quantile(values, 0.025))
            row[f"{column}_ci95_high"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    key = ["seed", "case_study_id", "freeze_year", "k"]
    pivot = frame.pivot_table(index=key, columns="method", values="primary_hits", aggfunc="first")
    if "neurodiscovery" not in pivot.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for baseline in sorted(column for column in pivot.columns if column != "neurodiscovery"):
        for k, part in pivot[["neurodiscovery", baseline]].dropna().groupby(level="k"):
            nd = part["neurodiscovery"].to_numpy(float)
            other = part[baseline].to_numpy(float)
            delta = nd - other
            try:
                statistic, p_value = wilcoxon(nd, other, alternative="greater")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            rows.append({
                "baseline": baseline,
                "k": int(k),
                "paired_windows": len(delta),
                "neurodiscovery_hits_mean": float(np.mean(nd)),
                "baseline_hits_mean": float(np.mean(other)),
                "mean_paired_delta": float(np.mean(delta)),
                "median_paired_delta": float(np.median(delta)),
                "neurodiscovery_win_fraction": float(np.mean(delta > 0)),
                "wilcoxon_greater_statistic": float(statistic),
                "wilcoxon_greater_p": float(p_value),
            })
    return pd.DataFrame(rows)


def run(roots: list[Path], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect(roots)
    task_summary = summarize_replicates(frame, ["method", "case_study_id", "k"])
    overall_summary = summarize_replicates(frame, ["method", "k"])
    tests = paired_tests(frame)
    tables = {
        "metrics_by_window_seed.csv": frame,
        "task_replicate_summary.csv": task_summary,
        "overall_replicate_summary.csv": overall_summary,
        "paired_method_tests.csv": tests,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    manifest = {
        "schema_version": "hindcasting-method-replicate-summary.v1",
        "roots": [str(root) for root in roots],
        "methods": sorted(frame["method"].dropna().unique().tolist()),
        "seeds": sorted(int(value) for value in frame["seed"].unique()),
        "case_studies": sorted(frame["case_study_id"].dropna().unique().tolist()),
        "metrics_rows": len(frame),
        "expected_metrics_rows": 3 * 10 * 17 * 5 * 3,
        "complete": len(frame) == 3 * 10 * 17 * 5 * 3,
        "tables": {name: {"path": str(output_dir / name), "rows": len(table)} for name, table in tables.items()},
    }
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.roots, args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Updated: 2026-08-11 13:03 HKT
