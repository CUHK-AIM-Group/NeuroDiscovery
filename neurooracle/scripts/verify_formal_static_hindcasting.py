"""Independently verify a completed formal static hindcasting benchmark.

The formal runner already guards each stage.  This verifier is deliberately
separate: it reopens the final artifacts and proves that the registered
case/window/seed matrix, execution budgets, evaluation prefixes, and summary
statistics are internally consistent.  It never writes into generation or
evaluation directories.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import fmean, stdev, variance
from typing import Any, Iterable, Mapping, Sequence

from neurooracle.src.experiment_source_bundle import verify_source_bundle
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESIGN = (
    ROOT
    / "neurooracle/data/experiments/hindcasting/optimization_protocol_20260812"
    / "formal_static_comparison_design_endpoint_v7_20260813.json"
)
STAGES = ("neurodiscovery", "baselines", "merge", "evaluate", "summarize")
RUN_FIELDS = (
    "seed",
    "case_study_id",
    "freeze_year",
    "future_start_year",
    "future_end_year",
)
RATE_FIELDS = (
    "primary_hit_rate",
    "unique_primary_discovery_rate",
    "future_pair_recall",
    "endpoint_hit_rate",
    "any_path_edge_hit_rate",
    "any_future_hit_rate",
    "mean_path_edge_hit_rate",
)
CUMULATIVE_FIELDS = (
    "primary_hits",
    "unique_primary_discoveries",
    "recovered_future_pairs",
    "endpoint_hits",
    "unique_endpoint_discoveries",
    "any_path_edge_hits",
    "all_path_edges_hits",
    "any_future_hits",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest().upper()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(value: Any, root: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path.resolve()
    for ancestor in (root, *root.parents):
        candidate = (ancestor / path).resolve()
        if candidate.exists():
            return candidate
    return (root / path).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_key(row: Mapping[str, Any]) -> tuple[int, str, int, int, int]:
    return (
        int(row["seed"]),
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


def _window_key(row: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
    )


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _failure_slot(hypothesis: Mapping[str, Any]) -> bool:
    return hypothesis.get("hypothesis_type") == "generation_failure" or bool(
        (hypothesis.get("metadata") or {}).get("generation_failure")
    )


def verify_hypothesis_payload(
    path: Path,
    *,
    target: int,
    expected_pool: int,
) -> tuple[list[dict[str, Any]], str, int]:
    """Verify one ranked pool and return hypotheses, hash, prefix failures."""

    payload, digest = _read_json_with_sha(path)
    hypotheses = payload.get("hypotheses") or []
    if not isinstance(hypotheses, list):
        raise ValueError(f"hypotheses is not a list: {path}")
    if len(hypotheses) != expected_pool:
        raise ValueError(
            f"candidate-pool mismatch for {path}: "
            f"expected={expected_pool} observed={len(hypotheses)}"
        )
    if len(hypotheses) < target:
        raise ValueError(f"execution prefix is shorter than {target}: {path}")
    identifiers = [str(row.get("id") or "") for row in hypotheses]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"hypothesis without id: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate hypothesis id: {path}")
    failures = sum(_failure_slot(row) for row in hypotheses[:target])
    return hypotheses, digest, failures


def _strict_bool(value: Any, *, field: str, path: Path) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse temporal-isolation flag {field} in {path}")


def audit_hypothesis_temporal_isolation(
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    freeze_year: int,
    path: Path,
) -> dict[str, int]:
    """Reject future evidence or outcome flags embedded in generated records."""

    future_flag_fields = {
        "uses_future_outcomes",
        "frontier_uses_future_outcomes",
        "future_outcomes_used",
        "future_outcomes_used_for_generation",
        "terminal_labels_available_to_generator",
        "uses_experimental_outcomes",
    }
    evidence_year_fields = {
        "year",
        "publication_year",
        "pub_year",
        "claim_year",
        "evidence_year",
    }
    counts = {"hypotheses": len(hypotheses), "flags": 0, "evidence_years": 0}
    for hypothesis in hypotheses:
        stack: list[Any] = [hypothesis]
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    normalized = str(key).strip().lower()
                    if normalized in future_flag_fields:
                        counts["flags"] += 1
                        if _strict_bool(nested, field=normalized, path=path):
                            raise ValueError(
                                f"future outcome flag {normalized} is true in {path}"
                            )
                    elif normalized == "freeze_year":
                        try:
                            observed_freeze = int(nested)
                        except (TypeError, ValueError):
                            raise ValueError(f"invalid freeze_year in {path}") from None
                        if observed_freeze != freeze_year:
                            raise ValueError(
                                f"embedded freeze_year {observed_freeze} differs from "
                                f"run freeze year {freeze_year} in {path}"
                            )
                    elif normalized in evidence_year_fields:
                        try:
                            observed_year = int(nested)
                        except (TypeError, ValueError):
                            pass
                        else:
                            counts["evidence_years"] += 1
                            if observed_year > freeze_year:
                                raise ValueError(
                                    f"future evidence year {observed_year} exceeds "
                                    f"freeze year {freeze_year} in {path}"
                                )
                    stack.append(nested)
            elif isinstance(value, list):
                stack.extend(value)
    return counts


def verify_topk_payload(
    payload: Mapping[str, Any],
    *,
    rank_points: Sequence[int],
    candidate_pool_size: int,
    random_trials: int,
) -> None:
    """Verify fixed-prefix semantics and cumulative top-k metrics."""

    topk = payload.get("topk") or {}
    observed_points = sorted(int(value) for value in topk)
    expected_points = sorted(int(value) for value in rank_points)
    if observed_points != expected_points:
        raise ValueError(
            f"top-k mismatch: expected={expected_points} observed={observed_points}"
        )
    previous: dict[str, float] = {}
    for k in expected_points:
        row = topk[str(k)]
        expected_slots = min(k, candidate_pool_size)
        if int(row.get("requested_experiment_slots", -1)) != k:
            raise ValueError(f"requested experiment slots differ at K={k}")
        if int(row.get("executed_ranked_prefix_slots", -1)) != expected_slots:
            raise ValueError(f"executed prefix differs at K={k}")
        observed = row.get("observed") or {}
        if int(observed.get("n", -1)) != expected_slots:
            raise ValueError(f"observed experiment count differs at K={k}")
        for field in RATE_FIELDS:
            value = observed.get(field)
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{field} is outside [0, 1] at K={k}")
        for field in CUMULATIVE_FIELDS:
            if field not in observed:
                continue
            value = float(observed[field])
            if value < previous.get(field, 0.0):
                raise ValueError(f"{field} decreases at K={k}")
            previous[field] = value
        if float(observed.get("unique_primary_discoveries", 0)) > float(
            observed.get("primary_hits", 0)
        ):
            raise ValueError(f"unique primary discoveries exceed hits at K={k}")
        random = row.get("random_same_hypothesis_pool") or {}
        if random.get("applicable") and int(random.get("trials", -1)) != random_trials:
            raise ValueError(f"random diagnostic trial count differs at K={k}")


def _csv_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value for {field}: {value!r}")


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _aggregate_scored_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    future_pair_total: int,
) -> dict[str, Any]:
    """Independently rebuild the evaluator's cumulative observed metrics."""

    n = len(rows)
    if n == 0:
        raise ValueError("registered rank points must execute at least one slot")
    primary = [_csv_bool(row.get("primary_hit"), field="primary_hit") for row in rows]
    endpoint = [_csv_bool(row.get("endpoint_hit"), field="endpoint_hit") for row in rows]
    any_path = [
        _csv_bool(row.get("any_path_edge_hit"), field="any_path_edge_hit")
        for row in rows
    ]
    all_path = [
        _csv_bool(row.get("all_path_edges_hit"), field="all_path_edges_hit")
        for row in rows
    ]
    any_future = [
        _csv_bool(row.get("any_future_hit"), field="any_future_hit") for row in rows
    ]
    primary_keys = {
        str(row.get("primary_discovery_key") or "")
        for row, hit in zip(rows, primary, strict=True)
        if hit and str(row.get("primary_discovery_key") or "")
    }
    endpoint_keys = {
        str(row.get("endpoint_discovery_key") or "")
        for row, hit in zip(rows, endpoint, strict=True)
        if hit and str(row.get("endpoint_discovery_key") or "")
    }
    recovered_pairs = {
        pair
        for row, hit in zip(rows, primary, strict=True)
        if hit
        for pair in str(row.get("primary_recovered_pairs") or "").split(";")
        if pair
    }
    path_rates = [float(row.get("path_edge_hit_rate") or 0.0) for row in rows]
    lead_times = [
        value
        for value in (_optional_float(row.get("primary_lead_time")) for row in rows)
        if value is not None
    ]
    primary_hits = sum(primary)
    endpoint_hits = sum(endpoint)
    any_path_hits = sum(any_path)
    any_future_hits = sum(any_future)
    return {
        "n": n,
        "primary_hits": primary_hits,
        "primary_hit_rate": primary_hits / n,
        "unique_primary_discoveries": len(primary_keys),
        "unique_primary_discovery_rate": len(primary_keys) / n,
        "recovered_future_pairs": len(recovered_pairs),
        "future_pair_recall": (
            len(recovered_pairs) / future_pair_total if future_pair_total > 0 else None
        ),
        "endpoint_hits": endpoint_hits,
        "endpoint_hit_rate": endpoint_hits / n,
        "unique_endpoint_discoveries": len(endpoint_keys),
        "any_path_edge_hits": any_path_hits,
        "any_path_edge_hit_rate": any_path_hits / n,
        "all_path_edges_hits": sum(all_path),
        "any_future_hits": any_future_hits,
        "any_future_hit_rate": any_future_hits / n,
        "mean_path_edge_hit_rate": fmean(path_rates),
        "mean_primary_lead_time": fmean(lead_times) if lead_times else None,
    }


def _percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _empirical_p_ge(values: Sequence[int], observed: int) -> float:
    return (1 + sum(int(value) >= observed for value in values)) / (len(values) + 1)


def _assert_metrics_equal(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field, expected_value in expected.items():
        if field not in actual or not _close(actual.get(field), expected_value):
            raise ValueError(f"{label} differs: {field}")


def verify_scored_topk_payload(
    payload: Mapping[str, Any],
    *,
    scored: Sequence[Mapping[str, Any]],
    random_trial_rows: Sequence[Mapping[str, Any]],
    rank_points: Sequence[int],
    random_trials: int,
) -> dict[str, int]:
    """Recompute observed and random-diagnostic summaries from their CSV rows."""

    if [int(row.get("rank") or -1) for row in scored] != list(
        range(1, len(scored) + 1)
    ):
        raise ValueError("scored hypothesis ranks are not contiguous")
    future_pair_total = int(
        ((payload.get("future_stats") or {}).get("future_unique_pairs") or 0)
    )
    trial_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in random_trial_rows:
        trial_groups[int(row["k"])].append(row)

    applicable_points: set[int] = set()
    for k in (int(value) for value in rank_points):
        subset = list(scored[: min(k, len(scored))])
        observed = (payload.get("topk") or {})[str(k)]["observed"]
        expected_observed = _aggregate_scored_rows(
            subset,
            future_pair_total=future_pair_total,
        )
        _assert_metrics_equal(
            observed,
            expected_observed,
            label=f"observed top-k metrics at K={k}",
        )

        random = (payload.get("topk") or {})[str(k)].get(
            "random_same_hypothesis_pool"
        ) or {}
        applicable = len(subset) < len(scored)
        if bool(random.get("applicable")) != applicable:
            raise ValueError(f"random diagnostic applicability differs at K={k}")
        rows = trial_groups.get(k, [])
        if not applicable:
            if rows:
                raise ValueError(f"inapplicable random diagnostic has trials at K={k}")
            if int(random.get("candidate_pool_size", -1)) != len(scored):
                raise ValueError(f"random diagnostic pool size differs at K={k}")
            continue

        applicable_points.add(k)
        if len(rows) != random_trials:
            raise ValueError(f"random trial row count differs at K={k}")
        if sorted(int(row["trial"]) for row in rows) != list(
            range(1, random_trials + 1)
        ):
            raise ValueError(f"random trial identifiers differ at K={k}")
        fields = (
            "primary_hits",
            "unique_primary_discoveries",
            "any_future_hits",
            "endpoint_hits",
            "unique_endpoint_discoveries",
        )
        values = {
            field: [int(row[field]) for row in rows]
            for field in fields
        }
        if any(value < 0 or value > len(subset) for group in values.values() for value in group):
            raise ValueError(f"random trial count is outside the executed prefix at K={k}")
        expected_random = {
            "applicable": True,
            "trials": random_trials,
            "mean_primary_hits": fmean(values["primary_hits"]),
            "mean_unique_primary_discoveries": fmean(
                values["unique_primary_discoveries"]
            ),
            "mean_any_future_hits": fmean(values["any_future_hits"]),
            "mean_endpoint_hits": fmean(values["endpoint_hits"]),
            "mean_unique_endpoint_discoveries": fmean(
                values["unique_endpoint_discoveries"]
            ),
            "sd_primary_hits": stdev(values["primary_hits"]),
            "variance_unique_primary_discoveries": variance(
                values["unique_primary_discoveries"]
            ),
            "sd_any_future_hits": stdev(values["any_future_hits"]),
            "sd_endpoint_hits": stdev(values["endpoint_hits"]),
            "p_primary_hits_ge_observed": _empirical_p_ge(
                values["primary_hits"], int(expected_observed["primary_hits"])
            ),
            "p_unique_primary_discoveries_ge_observed": _empirical_p_ge(
                values["unique_primary_discoveries"],
                int(expected_observed["unique_primary_discoveries"]),
            ),
            "p_any_future_hits_ge_observed": _empirical_p_ge(
                values["any_future_hits"], int(expected_observed["any_future_hits"])
            ),
            "p_endpoint_hits_ge_observed": _empirical_p_ge(
                values["endpoint_hits"], int(expected_observed["endpoint_hits"])
            ),
        }
        _assert_metrics_equal(
            random,
            expected_random,
            label=f"random diagnostic summary at K={k}",
        )
        for field, source in (
            ("ci95_primary_hits", "primary_hits"),
            ("ci95_any_future_hits", "any_future_hits"),
            ("ci95_endpoint_hits", "endpoint_hits"),
        ):
            expected_interval = [
                _percentile(values[source], 0.025),
                _percentile(values[source], 0.975),
            ]
            actual_interval = list(random.get(field) or ())
            if len(actual_interval) != 2 or any(
                not _close(actual, expected)
                for actual, expected in zip(
                    actual_interval, expected_interval, strict=True
                )
            ):
                raise ValueError(f"random diagnostic {field} differs at K={k}")

    extra_points = set(trial_groups) - applicable_points
    if extra_points:
        raise ValueError(f"random trials contain unregistered K values: {sorted(extra_points)}")
    return {
        "rank_points_recomputed": len(tuple(rank_points)),
        "random_trial_rows_recomputed": len(random_trial_rows),
    }


def _sample_variance(values: Sequence[float]) -> float:
    return float(variance(values)) if len(values) > 1 else 0.0


def _exact_sign_flip(differences: Sequence[float]) -> float:
    values = [float(value) for value in differences]
    if not values or all(abs(value) <= 1e-15 for value in values):
        return 1.0
    if len(values) > 20:
        raise ValueError("exact sign-flip verification supports at most 20 pairs")
    observed = fmean(values)
    extreme = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        statistic = fmean(sign * value for sign, value in zip(signs, values))
        extreme += statistic >= observed - 1e-15
    return extreme / (1 << len(values))


def _observation_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["method"]),
        int(row["seed"]),
        str(row["case_study_id"]),
        int(row["freeze_year"]),
        int(row["future_start_year"]),
        int(row["future_end_year"]),
        int(row["k"]),
        str(row["metric"]),
    )


def _summary_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["scope"]),
        str(row["method"]),
        str(row["case_study_id"]),
        str(row["freeze_year"]),
        str(row["future_start_year"]),
        str(row["future_end_year"]),
        int(row["k"]),
        str(row["metric"]),
    )


def _paired_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["scope"]),
        str(row["reference_method"]),
        str(row["comparison_method"]),
        str(row["case_study_id"]),
        str(row["freeze_year"]),
        str(row["future_start_year"]),
        str(row["future_end_year"]),
        int(row["k"]),
        str(row["metric"]),
    )


def _aggregate_vectors(
    observations: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], list[float]]:
    vectors: dict[tuple[Any, ...], list[float]] = {}
    case_seed: dict[tuple[str, int, str, int, str], list[float]] = defaultdict(list)
    flat_seed: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in observations:
        method = str(row["method"])
        seed = int(row["seed"])
        case = str(row["case_study_id"])
        k = int(row["k"])
        metric = str(row["metric"])
        value = float(row["value"])
        case_seed[(method, seed, case, k, metric)].append(value)
        flat_seed[(method, seed, k, metric)].append(value)

    case_seed_mean = {key: fmean(values) for key, values in case_seed.items()}
    case_groups: dict[tuple[str, str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for (method, seed, case, k, metric), value in case_seed_mean.items():
        case_groups[(method, case, k, metric)].append((seed, value))
    for (method, case, k, metric), rows in case_groups.items():
        rows.sort()
        vectors[("case_study_equal_window", method, case, k, metric)] = [
            value for _, value in rows
        ]

    equal_case_seed: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for (method, seed, _case, k, metric), value in case_seed_mean.items():
        equal_case_seed[(method, seed, k, metric)].append(value)
    equal_case_groups: dict[tuple[str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for (method, seed, k, metric), values in equal_case_seed.items():
        equal_case_groups[(method, k, metric)].append((seed, fmean(values)))
    for (method, k, metric), rows in equal_case_groups.items():
        rows.sort()
        vectors[("macro_case_study_equal", method, "ALL", k, metric)] = [
            value for _, value in rows
        ]

    flat_groups: dict[tuple[str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for (method, seed, k, metric), values in flat_seed.items():
        flat_groups[(method, k, metric)].append((seed, fmean(values)))
    for (method, k, metric), rows in flat_groups.items():
        rows.sort()
        vectors[("macro_case_window", method, "ALL", k, metric)] = [
            value for _, value in rows
        ]
    return vectors


def _expected_summary(
    observations: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, float | int]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in observations:
        groups[
            (
                "case_window",
                str(row["method"]),
                str(row["case_study_id"]),
                str(row["freeze_year"]),
                str(row["future_start_year"]),
                str(row["future_end_year"]),
                int(row["k"]),
                str(row["metric"]),
            )
        ].append(float(row["value"]))
    for key, values in _aggregate_vectors(observations).items():
        scope, method, case, k, metric = key
        marker = "WITHIN_CASE" if scope == "case_study_equal_window" else "ALL"
        groups[(scope, method, case, marker, marker, marker, k, metric)] = values
    return {
        key: {
            "n": len(values),
            "mean": fmean(values),
            "sample_variance": _sample_variance(values),
        }
        for key, values in groups.items()
    }


def _paired_record(
    reference: Sequence[float],
    comparison: Sequence[float],
) -> dict[str, float | int]:
    differences = [left - right for left, right in zip(reference, comparison)]
    return {
        "n_pairs": len(differences),
        "reference_mean": fmean(reference),
        "reference_sample_variance": _sample_variance(reference),
        "comparison_mean": fmean(comparison),
        "comparison_sample_variance": _sample_variance(comparison),
        "mean_paired_difference": fmean(differences),
        "paired_difference_sample_variance": _sample_variance(differences),
        "p_reference_greater_exact_sign_flip": _exact_sign_flip(differences),
        "reference_wins": sum(value > 0 for value in differences),
        "ties": sum(abs(value) <= 1e-15 for value in differences),
        "reference_losses": sum(value < 0 for value in differences),
    }


def _expected_paired(
    observations: Sequence[Mapping[str, Any]],
    *,
    reference_method: str,
) -> dict[tuple[Any, ...], dict[str, float | int]]:
    methods = sorted({str(row["method"]) for row in observations})
    values = {_observation_key(row): float(row["value"]) for row in observations}
    expected: dict[tuple[Any, ...], dict[str, float | int]] = {}
    strata = sorted({key[2:] for key in values if key[0] == reference_method})
    for comparator in methods:
        if comparator == reference_method:
            continue
        for case, freeze, start, end, k, metric in strata:
            seeds = sorted(
                key[1]
                for key in values
                if key[0] == reference_method
                and key[2:] == (case, freeze, start, end, k, metric)
            )
            ref = [
                values[(reference_method, seed, case, freeze, start, end, k, metric)]
                for seed in seeds
            ]
            comp = [
                values[(comparator, seed, case, freeze, start, end, k, metric)]
                for seed in seeds
            ]
            key = (
                "case_window",
                reference_method,
                comparator,
                case,
                str(freeze),
                str(start),
                str(end),
                k,
                metric,
            )
            expected[key] = _paired_record(ref, comp)

        vectors = _aggregate_vectors(observations)
        for scope in (
            "case_study_equal_window",
            "macro_case_study_equal",
            "macro_case_window",
        ):
            ref_keys = [
                key for key in vectors if key[0] == scope and key[1] == reference_method
            ]
            for ref_key in ref_keys:
                _, _, case, k, metric = ref_key
                comp_key = (scope, comparator, case, k, metric)
                marker = "WITHIN_CASE" if scope == "case_study_equal_window" else "ALL"
                key = (
                    scope,
                    reference_method,
                    comparator,
                    case,
                    marker,
                    marker,
                    marker,
                    k,
                    metric,
                )
                expected[key] = _paired_record(vectors[ref_key], vectors[comp_key])

    families: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
    for key in expected:
        scope, _ref, _comp, case, freeze, start, end, k, metric = key
        families[(scope, case, freeze, start, end, k, metric)].append(key)
    for family in families.values():
        ordered = sorted(
            family,
            key=lambda key: float(expected[key]["p_reference_greater_exact_sign_flip"]),
        )
        running = 0.0
        size = len(ordered)
        for rank, key in enumerate(ordered):
            raw = float(expected[key]["p_reference_greater_exact_sign_flip"])
            running = max(running, (size - rank) * raw)
            expected[key]["p_holm_within_endpoint"] = min(1.0, running)
            expected[key]["holm_family_size"] = size
    return expected


def verify_summary_tables(
    summary_root: Path,
    *,
    observations: Sequence[Mapping[str, Any]],
    reference_method: str,
) -> dict[str, int]:
    """Recompute every summary, paired test, and Holm-adjusted P value."""

    observed_rows = _read_csv(summary_root / "replicate_observations.csv")
    actual_observations = {
        _observation_key(row): float(row["value"]) for row in observed_rows
    }
    expected_observations = {
        _observation_key(row): float(row["value"]) for row in observations
    }
    if actual_observations.keys() != expected_observations.keys():
        raise ValueError("replicate observation matrix differs from evaluation artifacts")
    for key, expected in expected_observations.items():
        if not _close(actual_observations[key], expected):
            raise ValueError(f"replicate observation value differs: {key}")

    expected_summary = _expected_summary(observations)
    summary_rows = _read_csv(summary_root / "mean_variance_summary.csv")
    actual_summary = {_summary_key(row): row for row in summary_rows}
    if actual_summary.keys() != expected_summary.keys():
        raise ValueError("mean/variance summary matrix differs from recomputation")
    for key, expected in expected_summary.items():
        actual = actual_summary[key]
        for field, value in expected.items():
            if not _close(actual[field], value):
                raise ValueError(f"summary {field} differs for {key}")

    expected_paired = _expected_paired(
        observations, reference_method=reference_method
    )
    paired_rows = _read_csv(summary_root / "paired_comparisons.csv")
    actual_paired = {_paired_key(row): row for row in paired_rows}
    if actual_paired.keys() != expected_paired.keys():
        raise ValueError("paired comparison matrix differs from recomputation")
    for key, expected in expected_paired.items():
        actual = actual_paired[key]
        for field, value in expected.items():
            if not _close(actual[field], value):
                raise ValueError(f"paired comparison {field} differs for {key}")
    return {
        "observations": len(observed_rows),
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
    }


def _verify_execution(
    output_root: Path,
    design_path: Path,
    design: Mapping[str, Any],
    *,
    allow_smoke: bool,
) -> tuple[dict[str, Any], Path]:
    execution = _read_json(output_root / "execution_manifest.json")
    expected_mode = "smoke" if allow_smoke else "formal"
    if execution.get("status") != "complete" or execution.get("mode") != expected_mode:
        raise ValueError(f"execution is not a complete {expected_mode} run")
    if str(execution.get("design_sha256") or "").upper() != _sha256(design_path):
        raise ValueError("execution design hash mismatch")
    bundle = Path(str(execution.get("source_bundle_manifest") or "")).resolve()
    if not bundle.is_file() or str(
        execution.get("source_bundle_manifest_sha256") or ""
    ).upper() != _sha256(bundle):
        raise ValueError("source bundle manifest mismatch")
    declared_bundle = _resolve(
        design["reproducibility"]["source_bundle_manifest"], ROOT
    )
    if bundle != declared_bundle:
        raise ValueError("execution used a different source bundle")
    records = list(execution.get("records") or ())
    if [row.get("stage") for row in records] != list(STAGES):
        raise ValueError("formal stages are incomplete or out of order")
    archive_root = (bundle.parent / "files").resolve()
    for record in records:
        if int(record.get("returncode", -1)) != 0:
            raise ValueError(f"stage did not return zero: {record.get('stage')}")
        command = list(record.get("command") or ())
        if len(command) < 2 or not Path(command[1]).resolve().is_relative_to(archive_root):
            raise ValueError(f"stage did not execute frozen source: {record.get('stage')}")
        for side in ("bundle_before", "bundle_after"):
            guard = record.get(side) or {}
            if guard.get("status") != "passed" or not guard.get(
                "archived_source_verified"
            ):
                raise ValueError(f"source guard failed for {record.get('stage')} {side}")
            if not allow_smoke and not guard.get("referenced_inputs_verified"):
                raise ValueError(
                    f"referenced inputs were not rehashed for {record.get('stage')} {side}"
                )
    return execution, bundle


def verify_formal_static_hindcasting(
    *,
    output_root: Path,
    design_path: Path = DEFAULT_DESIGN,
    allow_smoke: bool = False,
    rehash_references: bool = False,
) -> dict[str, Any]:
    """Verify the complete benchmark and return a machine-readable audit."""

    output_root = output_root.resolve()
    design_path = design_path.resolve()
    design = _read_json(design_path)
    if design.get("status") != "frozen_before_formal_generation":
        raise ValueError("design is not frozen")
    execution, bundle = _verify_execution(
        output_root, design_path, design, allow_smoke=allow_smoke
    )
    eligibility = load_locked_hindcasting_eligibility(
        Path(str(execution["eligibility_manifest"]))
    )
    formal_matrix = design["primary_matrix"]
    if allow_smoke:
        seeds = tuple(
            int(value)
            for value in _read_json(
                output_root / "paired_generation/generation_manifest.json"
            )["seeds"]
        )
        windows = set(eligibility.primary_windows)
    else:
        seeds = tuple(int(value) for value in formal_matrix["seeds"])
        windows = set(eligibility.primary_windows)
        if len(windows) != int(formal_matrix["case_study_windows"]):
            raise ValueError("formal eligibility window count differs from design")
        if {key[0] for key in windows} != set(formal_matrix["case_study_ids"]):
            raise ValueError("formal eligibility Case Studies differ from design")
    expected_matrix = {
        (seed, case, freeze, start, end)
        for seed in seeds
        for case, freeze, start, end in windows
    }
    methods = tuple(row["id"] for row in design["methods"]["primary"])
    target = int(design["fixed_budget_contract"]["executed_hypotheses_per_run"])
    rank_points = tuple(int(value) for value in design["fixed_budget_contract"]["rank_points"])
    random_trials = int(
        design["fixed_budget_contract"]["random_trials_for_pool_diagnostic"]
    )
    if allow_smoke:
        paired_smoke = _read_json(
            output_root / "paired_generation/generation_manifest.json"
        )
        target = int(paired_smoke["target_per_case_study"])
        evaluation_smoke = _read_json(output_root / "evaluation/evaluation_manifest.json")
        rank_points = tuple(int(value) for value in evaluation_smoke["top_k"])
        random_trials = int(evaluation_smoke["random_trials"])

    paired = _read_json(output_root / "paired_generation/generation_manifest.json")
    if set(paired.get("methods") or ()) != set(methods):
        raise ValueError("paired generation methods differ from design")
    if int(paired.get("target_per_case_study", -1)) != target:
        raise ValueError("paired generation budget differs from design")
    paired_runs = list(paired.get("runs") or ())
    if len(paired_runs) != len(methods) * len(expected_matrix):
        raise ValueError("paired generation run count is incomplete")

    ranking_by_run: dict[
        tuple[str, *tuple[Any, ...]],
        tuple[tuple[str, ...], frozenset[int]],
    ] = {}
    artifact_hashes: list[str] = []
    observed_by_method: dict[str, set[tuple[int, str, int, int, int]]] = defaultdict(set)
    audit_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "runs": 0,
            "executed_slots": 0,
            "failure_slots": 0,
            "stored_candidates": 0,
            "candidate_tail_beyond_budget": 0,
        }
    )
    temporal_audit: dict[str, dict[str, int]] = defaultdict(
        lambda: {"hypotheses": 0, "flags": 0, "evidence_years": 0}
    )
    neuro_pool = next(
        int(row["generation_pool_size"])
        for row in design["methods"]["primary"]
        if row["id"] == "neurodiscovery"
    )
    if allow_smoke:
        neuro_pool = int(
            _read_json(output_root / "neurodiscovery_generation/generation_manifest.json")[
                "generation_pool_size"
            ]
        )
    seen: set[tuple[Any, ...]] = set()
    for row in paired_runs:
        method = str(row["method"])
        run_key = _run_key(row)
        full_key = (method, *run_key)
        if full_key in seen:
            raise ValueError(f"duplicate paired generation run: {full_key}")
        seen.add(full_key)
        observed_by_method[method].add(run_key)
        expected_pool = neuro_pool if method == "neurodiscovery" else target
        path = Path(str(row["hypotheses_path"])).resolve()
        hypotheses, digest, failures = verify_hypothesis_payload(
            path, target=target, expected_pool=expected_pool
        )
        temporal_counts = audit_hypothesis_temporal_isolation(
            hypotheses,
            freeze_year=run_key[2],
            path=path,
        )
        for field, value in temporal_counts.items():
            temporal_audit[method][field] += value
        if int(row.get("n_hypotheses", -1)) != target:
            raise ValueError(f"executed slots differ in paired row: {full_key}")
        if int(row.get("candidate_pool_size", -1)) != expected_pool:
            raise ValueError(f"candidate pool differs in paired row: {full_key}")
        if int(row.get("generation_failure_slots", -1)) != failures:
            raise ValueError(f"failure slots differ in paired row: {full_key}")
        if int(row.get("candidate_tail_beyond_budget", -1)) != expected_pool - target:
            raise ValueError(f"candidate tail differs in paired row: {full_key}")
        # The full records are several kilobytes each and are no longer needed
        # after their structure has been checked. Retain only the exact ranking
        # identity and prefix failure positions needed for downstream audits.
        ranking_by_run[full_key] = (
            tuple(str(hypothesis["id"]) for hypothesis in hypotheses),
            frozenset(
                index
                for index, hypothesis in enumerate(hypotheses[:target])
                if _failure_slot(hypothesis)
            ),
        )
        artifact_hashes.append(f"{path}|{digest}")
        counts = audit_counts[method]
        counts["runs"] += 1
        counts["executed_slots"] += target
        counts["failure_slots"] += failures
        counts["stored_candidates"] += expected_pool
        counts["candidate_tail_beyond_budget"] += expected_pool - target
    for method in methods:
        if observed_by_method[method] != expected_matrix:
            raise ValueError(f"paired generation matrix is incomplete for {method}")
        if dict(paired["audit_by_method"][method]) != dict(audit_counts[method]):
            raise ValueError(f"paired generation audit totals differ for {method}")

    evaluation = _read_json(output_root / "evaluation/evaluation_manifest.json")
    if set(evaluation.get("methods") or ()) != set(methods):
        raise ValueError("evaluation methods differ from design")
    if sorted(int(value) for value in evaluation.get("top_k") or ()) != sorted(
        rank_points
    ):
        raise ValueError("evaluation rank points differ from design")
    if int(evaluation.get("random_trials", -1)) != random_trials:
        raise ValueError("evaluation random trial count differs from design")
    evaluation_runs = list(evaluation.get("runs") or ())
    if len(evaluation_runs) != len(methods) * len(expected_matrix):
        raise ValueError("evaluation run count is incomplete")

    observations: list[dict[str, Any]] = []
    observed_evaluation: dict[str, set[tuple[int, str, int, int, int]]] = defaultdict(set)
    topk_audit = {
        "evaluation_runs": 0,
        "rank_points_recomputed": 0,
        "random_trial_rows_recomputed": 0,
    }
    metrics = tuple(
        _read_json(output_root / "summary/summary_manifest.json")["metrics"]
    )
    semantic = design["temporal_inputs"]
    for row in evaluation_runs:
        method = str(row["method"])
        run_key = _run_key(row)
        full_key = (method, *run_key)
        if run_key in observed_evaluation[method]:
            raise ValueError(f"duplicate evaluation run: {full_key}")
        observed_evaluation[method].add(run_key)
        metrics_path = Path(str(row["metrics_path"])).resolve()
        payload, digest = _read_json_with_sha(metrics_path)
        artifact_hashes.append(f"{metrics_path}|{digest}")
        expected_ids, failure_positions = ranking_by_run[full_key]
        if int(payload.get("ranking_pool_size", -1)) != len(expected_ids):
            raise ValueError(f"evaluation ranking pool differs: {full_key}")
        if str(payload.get("method")) != method or _window_key(payload) != _window_key(row):
            raise ValueError(f"evaluation identity differs: {full_key}")
        if payload.get("benchmark_status") != "executable":
            raise ValueError(f"formal evaluation is not executable: {full_key}")
        if payload.get("semantic_projection") != semantic["semantic_endpoint_identity_version"]:
            raise ValueError(f"semantic projection differs: {full_key}")
        if payload.get("case_study_relation_contract") != semantic["relation_contract_version"]:
            raise ValueError(f"relation contract differs: {full_key}")
        if payload.get("discovery_metric_contract") != semantic[
            "discovery_metric_contract_version"
        ]:
            raise ValueError(f"discovery metric contract differs: {full_key}")
        verify_topk_payload(
            payload,
            rank_points=rank_points,
            candidate_pool_size=len(expected_ids),
            random_trials=random_trials,
        )
        scored_path = metrics_path.parent / "scored_hypotheses.csv"
        scored = _read_csv(scored_path)
        if tuple(row["id"] for row in scored) != expected_ids:
            raise ValueError(f"scored ranking differs from generated ranking: {full_key}")
        artifact_hashes.append(f"{scored_path}|{_sha256(scored_path)}")
        random_trials_path = metrics_path.parent / "random_trials.csv"
        random_trial_rows = _read_csv(random_trials_path)
        artifact_hashes.append(f"{random_trials_path}|{_sha256(random_trials_path)}")
        recomputed = verify_scored_topk_payload(
            payload,
            scored=scored,
            random_trial_rows=random_trial_rows,
            rank_points=rank_points,
            random_trials=random_trials,
        )
        topk_audit["evaluation_runs"] += 1
        topk_audit["rank_points_recomputed"] += recomputed[
            "rank_points_recomputed"
        ]
        topk_audit["random_trial_rows_recomputed"] += recomputed[
            "random_trial_rows_recomputed"
        ]
        for index in failure_positions:
            scored_row = scored[index]
            if any(
                str(scored_row.get(field) or "").strip().lower() in {"true", "1"}
                for field in ("primary_hit", "endpoint_hit", "any_future_hit")
            ):
                raise ValueError(f"generation failure received credit: {full_key}")
        for k in rank_points:
            observed = payload["topk"][str(k)]["observed"]
            for metric in metrics:
                if metric not in observed:
                    raise ValueError(f"summary metric {metric} absent: {full_key} K={k}")
                observations.append(
                    {
                        "method": method,
                        "seed": run_key[0],
                        "case_study_id": run_key[1],
                        "freeze_year": run_key[2],
                        "future_start_year": run_key[3],
                        "future_end_year": run_key[4],
                        "k": k,
                        "metric": metric,
                        "value": float(observed[metric]),
                    }
                )
    for method in methods:
        if observed_evaluation[method] != expected_matrix:
            raise ValueError(f"evaluation matrix is incomplete for {method}")

    summary_manifest = _read_json(output_root / "summary/summary_manifest.json")
    if summary_manifest.get("reference_method") != "neurodiscovery":
        raise ValueError("summary reference method is not NeuroDiscovery")
    if summary_manifest.get("variance_definition") != "sample variance across seeds (ddof=1)":
        raise ValueError("summary variance definition differs from design")
    table_counts = verify_summary_tables(
        output_root / "summary",
        observations=observations,
        reference_method="neurodiscovery",
    )
    if int(summary_manifest.get("n_observations", -1)) != table_counts["observations"]:
        raise ValueError("summary manifest observation count differs")
    if int(summary_manifest.get("n_summary_rows", -1)) != table_counts["summary_rows"]:
        raise ValueError("summary manifest row count differs")
    if int(summary_manifest.get("n_paired_rows", -1)) != table_counts["paired_rows"]:
        raise ValueError("summary manifest paired row count differs")

    bundle_verification = None
    if rehash_references:
        bundle_verification = verify_source_bundle(
            bundle,
            require_live_source=False,
            verify_references=True,
        )
    artifact_digest = hashlib.sha256(
        "\n".join(sorted(artifact_hashes)).encode("utf-8")
    ).hexdigest().upper()
    primary_k = (
        max(rank_points)
        if allow_smoke
        else int(design["endpoints"]["primary"]["k"])
    )
    primary_rows = [
        row
        for row in _read_csv(output_root / "summary/mean_variance_summary.csv")
        if row["scope"] == "macro_case_study_equal"
        and row["case_study_id"] == "ALL"
        and int(row["k"]) == primary_k
        and row["metric"] == design["endpoints"]["primary"]["metric"]
    ]
    if len(primary_rows) != len(methods):
        raise ValueError("registered primary endpoint is incomplete")
    return {
        "schema_version": "formal-static-hindcasting-verification.v1",
        "verified_at": _utc_now(),
        "status": "passed",
        "mode": "smoke" if allow_smoke else "formal",
        "output_root": str(output_root),
        "design_path": str(design_path),
        "design_sha256": _sha256(design_path),
        "source_bundle_manifest": str(bundle),
        "source_bundle_manifest_sha256": _sha256(bundle),
        "references_rehashed": bool(rehash_references),
        "bundle_verification": bundle_verification,
        "methods": list(methods),
        "case_studies": sorted({key[0] for key in windows}),
        "case_study_windows": len(windows),
        "seeds": list(seeds),
        "runs_per_method": len(expected_matrix),
        "generation_runs": len(paired_runs),
        "evaluation_runs": len(evaluation_runs),
        "fixed_budget": target,
        "rank_points": list(rank_points),
        "metrics": list(metrics),
        "generation_audit_by_method": dict(audit_counts),
        "temporal_isolation_audit_by_method": dict(temporal_audit),
        "summary_tables": table_counts,
        "topk_recomputation": topk_audit,
        "artifact_hash_records": len(artifact_hashes),
        "artifact_set_sha256": artifact_digest,
        "registered_primary_endpoint": primary_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--rehash-references", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report or (
        args.output_root / "verification/verification_manifest.json"
    )
    try:
        result = verify_formal_static_hindcasting(
            output_root=args.output_root,
            design_path=args.design,
            allow_smoke=args.allow_smoke,
            rehash_references=args.rehash_references,
        )
    except Exception as exc:
        _atomic_json(
            report_path,
            {
                "schema_version": "formal-static-hindcasting-verification.v1",
                "verified_at": _utc_now(),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    _atomic_json(report_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
