"""Summarize repeated hindcasting evaluations without re-reading future claims.

The evaluator writes one ``metrics.json`` per method, seed, case study, and
temporal window.  This utility treats those runs as paired observations and
reports means, sample variances, and exact one-sided paired sign-flip tests.
It is intentionally a read-only analysis step: no hypothesis generation,
ranking, or outcome construction occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, variance
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_METRICS = (
    "unique_primary_discoveries",
    "future_pair_recall",
    "primary_hits",
)

WINDOW_FIELDS = (
    "case_study_id",
    "freeze_year",
    "future_start_year",
    "future_end_year",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    for ancestor in (root, *root.parents):
        candidate = ancestor / path
        if candidate.is_file():
            return candidate.resolve()
    return (root / path).resolve()


def _sample_variance(values: Sequence[float]) -> float:
    return float(variance(values)) if len(values) > 1 else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _window_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


def _load_primary_eligibility(
    manifest_path: Path,
) -> tuple[dict[str, Any], set[tuple[str, int, int, int]], Path]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    locked = manifest.get("locked_matrix") or {}
    matrix_path = _resolve_path(str(locked.get("path") or ""), manifest_path.parent)
    if not matrix_path.is_file():
        raise ValueError(f"locked eligibility matrix is absent: {matrix_path}")
    expected_hash = str(locked.get("sha256") or "").upper()
    observed_hash = _sha256(matrix_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError(
            "locked eligibility matrix hash mismatch: "
            f"expected={expected_hash or 'MISSING'} observed={observed_hash}"
        )

    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    primary_rows = [row for row in matrix_rows if row.get("analysis_tier") == "primary"]
    primary = {_window_key(row) for row in primary_rows}
    if len(primary) != len(primary_rows):
        raise ValueError("locked eligibility matrix contains duplicate primary windows")
    expected_count = int(manifest.get("primary_windows") or 0)
    if len(primary) != expected_count:
        raise ValueError(
            "locked eligibility primary-window count mismatch: "
            f"manifest={expected_count} matrix={len(primary)}"
        )
    expected_cases = {str(value) for value in manifest.get("primary_case_study_ids") or ()}
    observed_cases = {key[0] for key in primary}
    if observed_cases != expected_cases:
        raise ValueError(
            "locked eligibility primary Case Study mismatch: "
            f"missing={sorted(expected_cases - observed_cases)} "
            f"extra={sorted(observed_cases - expected_cases)}"
        )
    return manifest, primary, matrix_path


def _assert_locked_primary_matrix(
    rows: Sequence[Mapping[str, Any]],
    primary_windows: set[tuple[str, int, int, int]],
) -> None:
    strata: dict[tuple[str, int, int, str], set[tuple[str, int, int, int]]] = (
        defaultdict(set)
    )
    for row in rows:
        key = _window_key(row)
        if key not in primary_windows:
            raise ValueError(f"observation is outside locked primary eligibility: {key}")
        if str(row.get("benchmark_status") or "") != "executable":
            raise ValueError(
                "locked primary observation is not executable: "
                f"{key} status={row.get('benchmark_status')!r}"
            )
        strata[
            (
                str(row["method"]),
                int(row["seed"]),
                int(row["k"]),
                str(row["metric"]),
            )
        ].add(key)
    for stratum, observed in sorted(strata.items()):
        if observed != primary_windows:
            raise ValueError(
                f"incomplete locked primary matrix for {stratum}: "
                f"missing={sorted(primary_windows - observed)[:3]} "
                f"extra={sorted(observed - primary_windows)[:3]}"
            )


def exact_sign_flip_p_value(differences: Sequence[float]) -> float:
    """Return the exact one-sided P(ref > comparator) for paired differences."""

    values = [float(value) for value in differences]
    if not values or all(abs(value) <= 1e-15 for value in values):
        return 1.0
    if len(values) > 20:
        raise ValueError("exact sign-flip test supports at most 20 paired observations")
    observed = fmean(values)
    extreme = 0
    total = 1 << len(values)
    for bits in range(total):
        statistic = fmean(
            value if bits & (1 << index) else -value
            for index, value in enumerate(values)
        )
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / total


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


def _load_observations(
    evaluation_root: Path,
    manifest: Mapping[str, Any],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs") or ():
        metrics_path = _resolve_path(str(run["metrics_path"]), evaluation_root)
        payload = _read_json(metrics_path)
        window = (
            int(run["freeze_year"]),
            int(run["future_start_year"]),
            int(run["future_end_year"]),
        )
        for k, result in sorted(
            (payload.get("topk") or {}).items(), key=lambda item: int(item[0])
        ):
            observed = result.get("observed") or {}
            for metric in metrics:
                value = observed.get(metric)
                if value is None:
                    continue
                rows.append(
                    {
                        "method": str(run["method"]),
                        "seed": int(run["seed"]),
                        "case_study_id": str(run["case_study_id"]),
                        "freeze_year": window[0],
                        "future_start_year": window[1],
                        "future_end_year": window[2],
                        "k": int(k),
                        "metric": metric,
                        "value": float(value),
                        "benchmark_status": str(payload.get("benchmark_status") or ""),
                        "metrics_path": str(metrics_path),
                    }
                )
    return rows


def _assert_paired_matrix(rows: Sequence[Mapping[str, Any]]) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    by_method: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
            int(row["k"]),
            str(row["metric"]),
        )
        if key in by_method[str(row["method"])]:
            raise ValueError(f"duplicate replicate observation: {row['method']} {key}")
        by_method[str(row["method"])].add(key)
    reference = by_method[methods[0]] if methods else set()
    for method in methods[1:]:
        if by_method[method] != reference:
            missing = sorted(reference - by_method[method])[:3]
            extra = sorted(by_method[method] - reference)[:3]
            raise ValueError(
                f"unpaired evaluation matrix for {method}: missing={missing}, extra={extra}"
            )


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
            int(row["k"]),
            str(row["metric"]),
        )
        groups[key].append(float(row["value"]))

    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        method, case_id, freeze, start, end, k, metric = key
        output.append(
            {
                "scope": "case_window",
                "method": method,
                "case_study_id": case_id,
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": fmean(values),
                "sample_variance": _sample_variance(values),
            }
        )

    per_case_seed: dict[tuple[str, int, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        per_case_seed[
            (
                str(row["method"]),
                int(row["seed"]),
                str(row["case_study_id"]),
                int(row["k"]),
                str(row["metric"]),
            )
        ].append(float(row["value"]))
    case_seed_means = {
        key: fmean(values) for key, values in per_case_seed.items()
    }
    case_groups: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for (method, _seed, case_id, k, metric), value in case_seed_means.items():
        case_groups[(method, case_id, k, metric)].append(value)
    for (method, case_id, k, metric), values in sorted(case_groups.items()):
        output.append(
            {
                "scope": "case_study_equal_window",
                "method": method,
                "case_study_id": case_id,
                "freeze_year": "WITHIN_CASE",
                "future_start_year": "WITHIN_CASE",
                "future_end_year": "WITHIN_CASE",
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": fmean(values),
                "sample_variance": _sample_variance(values),
            }
        )

    per_seed: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        per_seed[
            (
                str(row["method"]),
                int(row["seed"]),
                int(row["k"]),
                str(row["metric"]),
            )
        ].append(float(row["value"]))
    macro: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (method, _seed, k, metric), values in per_seed.items():
        macro[(method, k, metric)].append(fmean(values))
    for (method, k, metric), values in sorted(macro.items()):
        output.append(
            {
                "scope": "macro_case_window",
                "method": method,
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": fmean(values),
                "sample_variance": _sample_variance(values),
            }
        )

    equal_case_per_seed: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for (method, seed, _case_id, k, metric), value in case_seed_means.items():
        equal_case_per_seed[(method, seed, k, metric)].append(value)
    equal_case_macro: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (method, _seed, k, metric), values in equal_case_per_seed.items():
        equal_case_macro[(method, k, metric)].append(fmean(values))
    for (method, k, metric), values in sorted(equal_case_macro.items()):
        output.append(
            {
                "scope": "macro_case_study_equal",
                "method": method,
                "case_study_id": "ALL",
                "freeze_year": "ALL",
                "future_start_year": "ALL",
                "future_end_year": "ALL",
                "k": k,
                "metric": metric,
                "n": len(values),
                "mean": fmean(values),
                "sample_variance": _sample_variance(values),
            }
        )
    return output


def _paired_rows(
    rows: Sequence[Mapping[str, Any]],
    reference_method: str,
) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows})
    if reference_method not in methods:
        raise ValueError(f"reference method is absent: {reference_method}")

    values = {
        (
            str(row["method"]),
            int(row["seed"]),
            str(row["case_study_id"]),
            int(row["freeze_year"]),
            int(row["future_start_year"]),
            int(row["future_end_year"]),
            int(row["k"]),
            str(row["metric"]),
        ): float(row["value"])
        for row in rows
    }
    strata = sorted(
        {
            (key[2], key[3], key[4], key[5], key[6], key[7])
            for key in values
        }
    )
    output: list[dict[str, Any]] = []
    for comparator in methods:
        if comparator == reference_method:
            continue
        for case_id, freeze, start, end, k, metric in strata:
            seeds = sorted(
                key[1]
                for key in values
                if key[0] == reference_method
                and key[2:] == (case_id, freeze, start, end, k, metric)
            )
            ref = [
                values[(reference_method, seed, case_id, freeze, start, end, k, metric)]
                for seed in seeds
            ]
            comp = [
                values[(comparator, seed, case_id, freeze, start, end, k, metric)]
                for seed in seeds
            ]
            differences = [left - right for left, right in zip(ref, comp)]
            output.append(
                {
                    "scope": "case_window",
                    "reference_method": reference_method,
                    "comparison_method": comparator,
                    "case_study_id": case_id,
                    "freeze_year": freeze,
                    "future_start_year": start,
                    "future_end_year": end,
                    "k": k,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "reference_mean": fmean(ref),
                    "reference_sample_variance": _sample_variance(ref),
                    "comparison_mean": fmean(comp),
                    "comparison_sample_variance": _sample_variance(comp),
                    "mean_paired_difference": fmean(differences),
                    "paired_difference_sample_variance": _sample_variance(differences),
                    "p_reference_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                    "reference_wins": sum(value > 0 for value in differences),
                    "ties": sum(abs(value) <= 1e-15 for value in differences),
                    "reference_losses": sum(value < 0 for value in differences),
                }
            )

        per_case_seed_ref: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
        per_case_seed_comp: dict[tuple[int, str, int, str], list[float]] = defaultdict(list)
        for key, value in values.items():
            method, seed, case_id, _freeze, _start, _end, k, metric = key
            if method == reference_method:
                per_case_seed_ref[(seed, case_id, k, metric)].append(value)
            elif method == comparator:
                per_case_seed_comp[(seed, case_id, k, metric)].append(value)

        case_seed_ref = {
            key: fmean(group) for key, group in per_case_seed_ref.items()
        }
        case_seed_comp = {
            key: fmean(group) for key, group in per_case_seed_comp.items()
        }
        case_strata = sorted({(key[1], key[2], key[3]) for key in case_seed_ref})
        for case_id, k, metric in case_strata:
            seeds = sorted(
                key[0]
                for key in case_seed_ref
                if key[1:] == (case_id, k, metric)
            )
            ref = [case_seed_ref[(seed, case_id, k, metric)] for seed in seeds]
            comp = [case_seed_comp[(seed, case_id, k, metric)] for seed in seeds]
            differences = [left - right for left, right in zip(ref, comp)]
            output.append(
                {
                    "scope": "case_study_equal_window",
                    "reference_method": reference_method,
                    "comparison_method": comparator,
                    "case_study_id": case_id,
                    "freeze_year": "WITHIN_CASE",
                    "future_start_year": "WITHIN_CASE",
                    "future_end_year": "WITHIN_CASE",
                    "k": k,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "reference_mean": fmean(ref),
                    "reference_sample_variance": _sample_variance(ref),
                    "comparison_mean": fmean(comp),
                    "comparison_sample_variance": _sample_variance(comp),
                    "mean_paired_difference": fmean(differences),
                    "paired_difference_sample_variance": _sample_variance(differences),
                    "p_reference_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                    "reference_wins": sum(value > 0 for value in differences),
                    "ties": sum(abs(value) <= 1e-15 for value in differences),
                    "reference_losses": sum(value < 0 for value in differences),
                }
            )

        per_seed_ref: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        per_seed_comp: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        for (seed, _case_id, k, metric), value in case_seed_ref.items():
            per_seed_ref[(seed, k, metric)].append(value)
        for (seed, _case_id, k, metric), value in case_seed_comp.items():
            per_seed_comp[(seed, k, metric)].append(value)
        for k, metric in sorted({(key[1], key[2]) for key in per_seed_ref}):
            seeds = sorted(key[0] for key in per_seed_ref if key[1:] == (k, metric))
            ref = [fmean(per_seed_ref[(seed, k, metric)]) for seed in seeds]
            comp = [fmean(per_seed_comp[(seed, k, metric)]) for seed in seeds]
            differences = [left - right for left, right in zip(ref, comp)]
            output.append(
                {
                    "scope": "macro_case_study_equal",
                    "reference_method": reference_method,
                    "comparison_method": comparator,
                    "case_study_id": "ALL",
                    "freeze_year": "ALL",
                    "future_start_year": "ALL",
                    "future_end_year": "ALL",
                    "k": k,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "reference_mean": fmean(ref),
                    "reference_sample_variance": _sample_variance(ref),
                    "comparison_mean": fmean(comp),
                    "comparison_sample_variance": _sample_variance(comp),
                    "mean_paired_difference": fmean(differences),
                    "paired_difference_sample_variance": _sample_variance(differences),
                    "p_reference_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                    "reference_wins": sum(value > 0 for value in differences),
                    "ties": sum(abs(value) <= 1e-15 for value in differences),
                    "reference_losses": sum(value < 0 for value in differences),
                }
            )

        flat_seed_ref: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        flat_seed_comp: dict[tuple[int, int, str], list[float]] = defaultdict(list)
        for key, value in values.items():
            method, seed, _case, _freeze, _start, _end, k, metric = key
            if method == reference_method:
                flat_seed_ref[(seed, k, metric)].append(value)
            elif method == comparator:
                flat_seed_comp[(seed, k, metric)].append(value)
        for k, metric in sorted({(key[1], key[2]) for key in flat_seed_ref}):
            seeds = sorted(key[0] for key in flat_seed_ref if key[1:] == (k, metric))
            ref = [fmean(flat_seed_ref[(seed, k, metric)]) for seed in seeds]
            comp = [fmean(flat_seed_comp[(seed, k, metric)]) for seed in seeds]
            differences = [left - right for left, right in zip(ref, comp)]
            output.append(
                {
                    "scope": "macro_case_window",
                    "reference_method": reference_method,
                    "comparison_method": comparator,
                    "case_study_id": "ALL",
                    "freeze_year": "ALL",
                    "future_start_year": "ALL",
                    "future_end_year": "ALL",
                    "k": k,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "reference_mean": fmean(ref),
                    "reference_sample_variance": _sample_variance(ref),
                    "comparison_mean": fmean(comp),
                    "comparison_sample_variance": _sample_variance(comp),
                    "mean_paired_difference": fmean(differences),
                    "paired_difference_sample_variance": _sample_variance(differences),
                    "p_reference_greater_exact_sign_flip": exact_sign_flip_p_value(differences),
                    "reference_wins": sum(value > 0 for value in differences),
                    "ties": sum(abs(value) <= 1e-15 for value in differences),
                    "reference_losses": sum(value < 0 for value in differences),
                }
            )
    return output


def _holm_adjust_paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adjust NeuroDiscovery-vs-baseline tests within each reported endpoint."""

    output = [dict(row) for row in rows]
    families: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        families[
            (
                row["scope"],
                row["case_study_id"],
                row["freeze_year"],
                row["future_start_year"],
                row["future_end_year"],
                int(row["k"]),
                row["metric"],
            )
        ].append(index)

    for family_indices in families.values():
        ordered = sorted(
            family_indices,
            key=lambda index: float(
                output[index]["p_reference_greater_exact_sign_flip"]
            ),
        )
        adjusted_running = 0.0
        family_size = len(ordered)
        for rank, index in enumerate(ordered):
            raw = float(output[index]["p_reference_greater_exact_sign_flip"])
            adjusted_running = max(adjusted_running, (family_size - rank) * raw)
            output[index]["p_holm_within_endpoint"] = min(1.0, adjusted_running)
            output[index]["holm_family_size"] = family_size
    return output


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.evaluation_root / "evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    rows = _load_observations(args.evaluation_root, manifest, args.metrics)
    if not rows:
        raise ValueError("evaluation manifest contains no requested metric observations")
    eligibility_path = getattr(args, "eligibility_manifest", None)
    eligibility: dict[str, Any] | None = None
    eligibility_matrix: Path | None = None
    if eligibility_path is not None:
        eligibility, primary_windows, eligibility_matrix = _load_primary_eligibility(
            Path(eligibility_path)
        )
        _assert_locked_primary_matrix(rows, primary_windows)
    _assert_paired_matrix(rows)
    summary_rows = _summary_rows(rows)
    paired_rows = _holm_adjust_paired_rows(
        _paired_rows(rows, args.reference_method)
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "replicate_observations.csv", rows)
    _write_csv(args.output_root / "mean_variance_summary.csv", summary_rows)
    _write_csv(args.output_root / "paired_comparisons.csv", paired_rows)
    result = {
        "schema_version": "hindcasting-replicate-summary.v2",
        "evaluation_root": str(args.evaluation_root.resolve()),
        "evaluation_manifest": str(manifest_path.resolve()),
        "reference_method": args.reference_method,
        "metrics": list(args.metrics),
        "variance_definition": "sample variance across seeds (ddof=1)",
        "paired_test": "exact one-sided paired sign-flip; H1 reference > comparator",
        "multiple_comparison_correction": (
            "Holm adjustment across comparison methods within each reported "
            "scope/case/window/K/metric endpoint"
        ),
        "primary_aggregation": (
            "within each seed and Case Study, average eligible windows; then take an "
            "equal-weight mean across Case Studies; finally report mean and sample "
            "variance across paired seeds"
        ),
        "diagnostic_aggregation": "flat equal-weight macro across eligible Case Study windows",
        "eligibility_manifest": (
            str(Path(eligibility_path).resolve()) if eligibility_path is not None else None
        ),
        "eligibility_manifest_sha256": (
            _sha256(Path(eligibility_path).resolve()) if eligibility_path is not None else None
        ),
        "eligibility_matrix": str(eligibility_matrix) if eligibility_matrix else None,
        "eligibility_matrix_sha256": (
            str((eligibility or {}).get("locked_matrix", {}).get("sha256") or "")
            or None
        ),
        "primary_case_study_ids": (
            list((eligibility or {}).get("primary_case_study_ids") or ())
            if eligibility is not None
            else None
        ),
        "n_observations": len(rows),
        "n_summary_rows": len(summary_rows),
        "n_paired_rows": len(paired_rows),
    }
    (args.output_root / "summary_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-method", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument(
        "--eligibility-manifest",
        type=Path,
        help="Optional locked eligibility manifest; when provided, only its complete primary matrix is accepted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(summarize(parse_args()), indent=2, ensure_ascii=False))
