"""Compare paired dynamic closed-loop and outcome-withheld open-loop runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, variance
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.scripts.summarize_hindcasting_replicates import (
    exact_sign_flip_p_value,
)


DEFAULT_METRICS = (
    "terminal_unique_primary_discoveries",
    "unique_primary_discoveries",
    "recovered_future_pairs",
    "future_pair_recall",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _run_key(row: Mapping[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        int(row["seed"]),
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


def _without_feedback_flag(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.pop("feedback_enabled", None)
    return normalized


def _execution_signature(hypothesis: Mapping[str, Any]) -> tuple[Any, ...]:
    path = tuple(
        (
            str(link.get("from_id") or ""),
            str(link.get("relation_type") or ""),
            str(link.get("to_id") or ""),
        )
        for link in (hypothesis.get("path") or [])
        if isinstance(link, Mapping)
    )
    return (
        str(hypothesis.get("hypothesis_type") or ""),
        str(hypothesis.get("source_id") or ""),
        str(hypothesis.get("target_id") or ""),
        path,
    )


def _run_artifact_dirs(root: Path) -> dict[tuple[int, str, int, int, int], Path]:
    indexed: dict[tuple[int, str, int, int, int], Path] = {}
    for path in root.rglob("run_manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "case_study_id" not in payload:
            continue
        key = _run_key(payload)
        if key in indexed:
            raise ValueError(f"duplicate run artifacts for {key} under {root}")
        indexed[key] = path.parent
    return indexed


def _executed_hypotheses(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "executed_hypotheses.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing paired execution artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    hypotheses = payload.get("hypotheses") or []
    if int(payload.get("n_hypotheses") or 0) != len(hypotheses):
        raise ValueError(f"inconsistent execution count in {path}")
    return list(hypotheses)


def _audit_design(closed_root: Path, open_root: Path) -> dict[str, Any]:
    closed_path = closed_root / "dynamic_closed_loop_manifest.json"
    open_path = open_root / "dynamic_closed_loop_manifest.json"
    if not closed_path.is_file() or not open_path.is_file():
        raise FileNotFoundError(
            "paired dynamic comparison requires both top-level run manifests"
        )
    closed = json.loads(closed_path.read_text(encoding="utf-8"))
    opened = json.loads(open_path.read_text(encoding="utf-8"))

    closed_config = dict(closed.get("config") or {})
    open_config = dict(opened.get("config") or {})
    if closed_config.get("feedback_enabled") is not True:
        raise ValueError("closed arm does not declare feedback_enabled=true")
    if open_config.get("feedback_enabled") is not False:
        raise ValueError("open arm does not declare feedback_enabled=false")
    if _without_feedback_flag(closed_config) != _without_feedback_flag(open_config):
        raise ValueError("closed/open non-feedback configurations differ")

    identical_fields = (
        "snapshot_root",
        "future_claims",
        "case_studies",
        "windows",
        "seeds",
        "budgets",
        "profile",
        "source_bundle",
        "eligibility",
        "runs",
    )
    mismatched = [field for field in identical_fields if closed.get(field) != opened.get(field)]
    if mismatched:
        raise ValueError(f"closed/open top-level design differs: {mismatched}")

    closed_runs = {_run_key(row): row for row in closed.get("run_summaries") or []}
    open_runs = {_run_key(row): row for row in opened.get("run_summaries") or []}
    if set(closed_runs) != set(open_runs):
        raise ValueError("closed/open run-manifest matrices differ")
    if len(closed_runs) != int(closed.get("runs") or 0):
        raise ValueError("closed manifest run count is inconsistent")
    if len(open_runs) != int(opened.get("runs") or 0):
        raise ValueError("open manifest run count is inconsistent")
    closed_dirs = _run_artifact_dirs(closed_root)
    open_dirs = _run_artifact_dirs(open_root)
    if set(closed_dirs) != set(closed_runs) or set(open_dirs) != set(open_runs):
        raise ValueError("top-level and on-disk run artifact matrices differ")

    closed_activated = 0
    closed_supported = 0
    open_withheld = 0
    warmup_prefix_hypotheses = 0
    for key in sorted(closed_runs):
        closed_run = closed_runs[key]
        open_run = open_runs[key]
        if _without_feedback_flag(closed_run.get("config") or {}) != _without_feedback_flag(
            open_run.get("config") or {}
        ):
            raise ValueError(f"paired run non-feedback configuration differs: {key}")
        if (closed_run.get("config") or {}).get("feedback_enabled") is not True:
            raise ValueError(f"closed paired run has feedback disabled: {key}")
        if (open_run.get("config") or {}).get("feedback_enabled") is not False:
            raise ValueError(f"open paired run has feedback enabled: {key}")

        for arm, run in (("closed", closed_run), ("open", open_run)):
            isolation = dict(run.get("temporal_isolation") or {})
            if isolation.get("terminal_labels_available_to_generator") is not False:
                raise ValueError(f"{arm} terminal labels are not isolated: {key}")
            if isolation.get("formal_kg_mutated") is not False:
                raise ValueError(f"{arm} formal KG was mutated: {key}")
        if int(open_run.get("supported_feedback_records") or 0) != 0:
            raise ValueError(f"open arm exposed supported feedback: {key}")
        if bool(open_run.get("closed_loop_activated")):
            raise ValueError(f"open arm activated feedback: {key}")
        if open_run.get("first_feedback_generation_round") is not None:
            raise ValueError(f"open arm generated from feedback: {key}")
        if int(closed_run.get("withheld_supported_outcomes") or 0) != 0:
            raise ValueError(f"closed arm withheld observable feedback: {key}")

        closed_start = int(closed_run.get("feedback_start_budget") or 0)
        open_start = int(open_run.get("feedback_start_budget") or 0)
        if closed_start < 1 or closed_start != open_start:
            raise ValueError(f"paired feedback start differs or is invalid: {key}")
        closed_executed = _executed_hypotheses(closed_dirs[key])
        open_executed = _executed_hypotheses(open_dirs[key])
        if len(closed_executed) < closed_start or len(open_executed) < open_start:
            raise ValueError(f"paired run is shorter than its feedback warm-up: {key}")
        closed_prefix = [
            _execution_signature(row) for row in closed_executed[:closed_start]
        ]
        open_prefix = [
            _execution_signature(row) for row in open_executed[:open_start]
        ]
        if closed_prefix != open_prefix:
            first_mismatch = next(
                (
                    index + 1
                    for index, (left, right) in enumerate(
                        zip(closed_prefix, open_prefix, strict=True)
                    )
                    if left != right
                ),
                None,
            )
            raise ValueError(
                f"closed/open execution diverged before feedback for {key} "
                f"at rank {first_mismatch}"
            )

        closed_activated += int(bool(closed_run.get("closed_loop_activated")))
        closed_supported += int(closed_run.get("supported_feedback_records") or 0)
        open_withheld += int(open_run.get("withheld_supported_outcomes") or 0)
        warmup_prefix_hypotheses += closed_start

    return {
        "status": "passed",
        "closed_manifest_sha256": _sha256_file(closed_path),
        "open_manifest_sha256": _sha256_file(open_path),
        "paired_runs": len(closed_runs),
        "non_feedback_configuration_identical": True,
        "terminal_labels_isolated": True,
        "formal_kg_unmodified": True,
        "open_feedback_records_exposed": 0,
        "open_feedback_activated_runs": 0,
        "open_withheld_supported_outcomes": open_withheld,
        "closed_supported_feedback_records": closed_supported,
        "closed_feedback_activated_runs": closed_activated,
        "warmup_execution_prefix_identical": True,
        "warmup_prefix_hypotheses_compared": warmup_prefix_hypotheses,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _sample_variance(values: Sequence[float]) -> float:
    return float(variance(values)) if len(values) > 1 else 0.0


def _key(row: Mapping[str, str]) -> tuple[int, str, int, int, int, int]:
    return (
        int(row["seed"]),
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
        int(row["requested_k"]),
    )


def compare(args: argparse.Namespace) -> dict[str, Any]:
    design_audit = _audit_design(args.closed_root, args.open_root)
    closed_path = args.closed_root / "metrics_by_run.csv"
    open_path = args.open_root / "metrics_by_run.csv"
    closed_rows = {_key(row): row for row in _read_csv(closed_path)}
    open_rows = {_key(row): row for row in _read_csv(open_path)}
    if set(closed_rows) != set(open_rows):
        missing = sorted(set(closed_rows) - set(open_rows))[:3]
        extra = sorted(set(open_rows) - set(closed_rows))[:3]
        raise ValueError(f"closed/open matrices differ: missing={missing}, extra={extra}")

    observations: list[dict[str, Any]] = []
    for key in sorted(closed_rows):
        seed, case_id, freeze, start, end, k = key
        for metric in args.metrics:
            closed_raw = closed_rows[key].get(metric)
            open_raw = open_rows[key].get(metric)
            if closed_raw in (None, "") or open_raw in (None, ""):
                continue
            closed_value = float(closed_raw)
            open_value = float(open_raw)
            observations.append(
                {
                    "seed": seed,
                    "case_study_id": case_id,
                    "freeze_year": freeze,
                    "future_start_year": start,
                    "future_end_year": end,
                    "requested_k": k,
                    "metric": metric,
                    "closed_value": closed_value,
                    "open_value": open_value,
                    "paired_difference": closed_value - open_value,
                }
            )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        groups[
            (
                row["case_study_id"],
                row["freeze_year"],
                row["future_start_year"],
                row["future_end_year"],
                row["requested_k"],
                row["metric"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, rows in sorted(groups.items()):
        case_id, freeze, start, end, k, metric = group_key
        closed = [float(row["closed_value"]) for row in rows]
        opened = [float(row["open_value"]) for row in rows]
        differences = [float(row["paired_difference"]) for row in rows]
        summaries.append(
            {
                "scope": "case_window",
                "case_study_id": case_id,
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
                "requested_k": k,
                "metric": metric,
                "n_pairs": len(rows),
                "closed_mean": fmean(closed),
                "closed_sample_variance": _sample_variance(closed),
                "open_mean": fmean(opened),
                "open_sample_variance": _sample_variance(opened),
                "mean_paired_difference": fmean(differences),
                "paired_difference_sample_variance": _sample_variance(differences),
                "p_closed_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                "closed_wins": sum(value > 0 for value in differences),
                "ties": sum(abs(value) <= 1e-15 for value in differences),
                "closed_losses": sum(value < 0 for value in differences),
            }
        )

    per_case_seed: dict[tuple[int, str, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in observations:
        per_case_seed[
            (
                row["seed"],
                row["case_study_id"],
                row["requested_k"],
                row["metric"],
            )
        ].append(row)
    case_seed_means = {
        key: {
            "closed": fmean(float(row["closed_value"]) for row in grouped),
            "open": fmean(float(row["open_value"]) for row in grouped),
        }
        for key, grouped in per_case_seed.items()
    }
    for case_id, k, metric in sorted(
        {(key[1], key[2], key[3]) for key in case_seed_means}
    ):
        seeds = sorted(
            key[0]
            for key in case_seed_means
            if key[1:] == (case_id, k, metric)
        )
        closed = [
            case_seed_means[(seed, case_id, k, metric)]["closed"]
            for seed in seeds
        ]
        opened = [
            case_seed_means[(seed, case_id, k, metric)]["open"]
            for seed in seeds
        ]
        differences = [left - right for left, right in zip(closed, opened)]
        summaries.append(
            {
                "scope": "case_study_equal_window",
                "case_study_id": case_id,
                "freeze_year": "WITHIN_CASE",
                "future_start_year": "WITHIN_CASE",
                "future_end_year": "WITHIN_CASE",
                "requested_k": k,
                "metric": metric,
                "n_pairs": len(seeds),
                "closed_mean": fmean(closed),
                "closed_sample_variance": _sample_variance(closed),
                "open_mean": fmean(opened),
                "open_sample_variance": _sample_variance(opened),
                "mean_paired_difference": fmean(differences),
                "paired_difference_sample_variance": _sample_variance(differences),
                "p_closed_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                "closed_wins": sum(value > 0 for value in differences),
                "ties": sum(abs(value) <= 1e-15 for value in differences),
                "closed_losses": sum(value < 0 for value in differences),
            }
        )

    per_seed_case_equal: dict[tuple[int, int, str], list[dict[str, float]]] = (
        defaultdict(list)
    )
    for (seed, _case_id, k, metric), values in case_seed_means.items():
        per_seed_case_equal[(seed, k, metric)].append(values)
    for k, metric in sorted({(key[1], key[2]) for key in per_seed_case_equal}):
        seeds = sorted(
            key[0] for key in per_seed_case_equal if key[1:] == (k, metric)
        )
        closed = [
            fmean(row["closed"] for row in per_seed_case_equal[(seed, k, metric)])
            for seed in seeds
        ]
        opened = [
            fmean(row["open"] for row in per_seed_case_equal[(seed, k, metric)])
            for seed in seeds
        ]
        differences = [left - right for left, right in zip(closed, opened)]
        summaries.append(
            {
                "scope": "macro_case_study_equal",
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "requested_k": k,
                "metric": metric,
                "n_pairs": len(seeds),
                "closed_mean": fmean(closed),
                "closed_sample_variance": _sample_variance(closed),
                "open_mean": fmean(opened),
                "open_sample_variance": _sample_variance(opened),
                "mean_paired_difference": fmean(differences),
                "paired_difference_sample_variance": _sample_variance(differences),
                "p_closed_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                "closed_wins": sum(value > 0 for value in differences),
                "ties": sum(abs(value) <= 1e-15 for value in differences),
                "closed_losses": sum(value < 0 for value in differences),
            }
        )

    per_seed_flat: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        per_seed_flat[(row["seed"], row["requested_k"], row["metric"])].append(row)
    for k, metric in sorted({(key[1], key[2]) for key in per_seed_flat}):
        seeds = sorted(key[0] for key in per_seed_flat if key[1:] == (k, metric))
        closed = [
            fmean(float(row["closed_value"]) for row in per_seed_flat[(seed, k, metric)])
            for seed in seeds
        ]
        opened = [
            fmean(float(row["open_value"]) for row in per_seed_flat[(seed, k, metric)])
            for seed in seeds
        ]
        differences = [left - right for left, right in zip(closed, opened)]
        summaries.append(
            {
                "scope": "macro_case_window",
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "requested_k": k,
                "metric": metric,
                "n_pairs": len(seeds),
                "closed_mean": fmean(closed),
                "closed_sample_variance": _sample_variance(closed),
                "open_mean": fmean(opened),
                "open_sample_variance": _sample_variance(opened),
                "mean_paired_difference": fmean(differences),
                "paired_difference_sample_variance": _sample_variance(differences),
                "p_closed_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                "closed_wins": sum(value > 0 for value in differences),
                "ties": sum(abs(value) <= 1e-15 for value in differences),
                "closed_losses": sum(value < 0 for value in differences),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "paired_observations.csv", observations)
    _write_csv(args.output_root / "mean_variance_paired_summary.csv", summaries)
    (args.output_root / "design_audit.json").write_text(
        json.dumps(design_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "dynamic-closed-open-comparison.v2",
        "closed_root": str(args.closed_root.resolve()),
        "open_root": str(args.open_root.resolve()),
        "metrics": list(args.metrics),
        "variance_definition": "sample variance across paired seeds (ddof=1)",
        "paired_test": "exact one-sided paired sign-flip; H1 closed > open",
        "primary_aggregation": (
            "within each seed and Case Study, average eligible windows; then take "
            "an equal-weight mean across Case Studies; finally summarize paired seeds"
        ),
        "diagnostic_aggregation": "flat equal-weight macro across Case Study windows",
        "n_observations": len(observations),
        "n_summary_rows": len(summaries),
        "design_audit": design_audit,
    }
    (args.output_root / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-root", type=Path, required=True)
    parser.add_argument("--open-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(compare(parse_args()), indent=2, ensure_ascii=False))
