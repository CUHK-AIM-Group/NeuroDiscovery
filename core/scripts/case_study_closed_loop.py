"""Case-study-neutral closed-loop discovery benchmark.

The module keeps the executable candidate registry, internal validation labels,
and external validation labels in separate files.  Search methods only receive
the public registry.  NeuroDiscovery may consume an internal label after that
candidate has been executed; external labels are loaded only after all discovery
orders have been materialized and hashed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_search_policy import (
    SearchPolicy,
    policy_from_payload,
    validate_exact_policy,
)
from core.scripts.case_study_closed_loop_engine import run_closed_loop_order
from core.scripts.case_study_feedback_adapters import (
    FEEDBACK_STATUSES,
    adapter_for,
)
from core.scripts.case_study_neurodiscovery_policy import (
    FEEDBACK_BATCH_SCHEDULES,
    FEEDBACK_MODELS,
    FEEDBACK_PROJECTIONS,
    apply_neurodiscovery_policy,
    load_neurodiscovery_policy,
)
from core.scripts.case_study_score_components import (
    embedded_score_component_audit,
    load_score_component_bundle,
)


SCHEMA_VERSION = "case-study-closed-loop.v2"
PUBLIC_REGISTRY_SCHEMA = "case-study-candidate-registry.v1"
OUTCOME_SCHEMA = "case-study-hidden-outcomes.v2"
FROZEN_SCHEMA = "case-study-frozen-rankings.v1"

DEFAULT_METHODS = (
    "random_walk",
    "neurodiscovery",
)

FORBIDDEN_PUBLIC_COLUMNS = frozenset(
    {
        "validated",
        "strict_validated",
        "internal_label",
        "external_label",
        "gt_label",
        "is_gt",
        "effect",
        "effect_size",
        "cohen_d",
        "adjusted_residual_d",
        "p_value",
        "pvalue",
        "q_value",
        "q_fdr",
        "fdr",
        "test_statistic",
        "feedback_status",
        "execution_succeeded",
        "error",
    }
)


@dataclass(frozen=True)
class RankingRecord:
    method: str
    trial: int
    order: np.ndarray


@dataclass(frozen=True)
class ClosedLoopConfig:
    batch_size: int = 64
    warmup_batches: int = 1
    sampling_temperature: float = 0.01
    feedback_weight: float = 0.45
    pair_feedback_weight: float = 0.25
    exploration_weight: float = 0.10
    diversity_penalty: float = 0.08
    warmup_diversity_penalty: float | None = None
    min_supported_before_feedback: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 3.0
    max_feedback_rounds: int = 128
    feedback_horizon: int | None = None
    inconclusive_search_failure_weight: float = 0.0
    feedback_projection: str = "all_factors"
    feedback_batch_schedule: str = "fixed"
    feedback_model: str = "beta_counts"
    surrogate_ridge_alpha: float = 1.0
    preserve_static_until_informative_feedback: bool = False

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.warmup_batches < 1:
            raise ValueError("warmup_batches must be positive")
        if self.max_feedback_rounds < 1:
            raise ValueError("max_feedback_rounds must be positive")
        if self.min_supported_before_feedback < 0:
            raise ValueError("min_supported_before_feedback must be non-negative")
        if self.feedback_horizon is not None and self.feedback_horizon < 1:
            raise ValueError("feedback_horizon must be positive when provided")
        for name in (
            "sampling_temperature",
            "feedback_weight",
            "pair_feedback_weight",
            "exploration_weight",
            "diversity_penalty",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.warmup_diversity_penalty is not None and (
            not math.isfinite(float(self.warmup_diversity_penalty))
            or self.warmup_diversity_penalty < 0
        ):
            raise ValueError("warmup_diversity_penalty must be finite and non-negative")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if (
            not math.isfinite(float(self.surrogate_ridge_alpha))
            or self.surrogate_ridge_alpha <= 0
        ):
            raise ValueError("surrogate_ridge_alpha must be finite and positive")
        if not 0.0 <= self.inconclusive_search_failure_weight <= 1.0:
            raise ValueError("inconclusive_search_failure_weight must be in [0, 1]")
        if self.feedback_projection not in FEEDBACK_PROJECTIONS:
            raise ValueError(
                f"unknown feedback projection: {self.feedback_projection!r}"
            )
        if self.feedback_batch_schedule not in FEEDBACK_BATCH_SCHEDULES:
            raise ValueError(
                f"unknown feedback batch schedule: {self.feedback_batch_schedule!r}"
            )
        if self.feedback_model not in FEEDBACK_MODELS:
            raise ValueError(f"unknown feedback model: {self.feedback_model!r}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _normalize_score(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=True)
    finite = np.isfinite(numeric)
    if not finite.any():
        return np.zeros(len(numeric), dtype=float)
    low = float(np.nanmin(numeric[finite]))
    high = float(np.nanmax(numeric[finite]))
    numeric[~finite] = low
    if math.isclose(low, high):
        return np.full(len(numeric), 0.5, dtype=float)
    return (numeric - low) / (high - low)


def _coerce_bool_series(values: pd.Series, *, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise ValueError(f"{column} must contain only boolean or 0/1 values")
        return numeric.astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{column} contains invalid boolean values: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def validate_public_registry(
    frame: pd.DataFrame,
    *,
    factor_fields: Sequence[str],
) -> pd.DataFrame:
    """Validate and normalize the public, outcome-blind candidate registry."""

    required = {"candidate_id", "score_neurodiscovery", *factor_fields}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"public registry is missing columns: {missing}")
    forbidden = sorted(FORBIDDEN_PUBLIC_COLUMNS & set(frame.columns))
    hidden_prefixed = sorted(
        column for column in frame.columns if str(column).startswith("hidden_")
    )
    if forbidden or hidden_prefixed:
        raise ValueError(
            "public registry exposes hidden outcome fields: "
            f"{sorted(set(forbidden + hidden_prefixed))}"
        )
    out = frame.copy()
    out["candidate_id"] = out["candidate_id"].astype(str).str.strip()
    if out["candidate_id"].eq("").any():
        raise ValueError("candidate_id cannot be empty")
    if out["candidate_id"].duplicated().any():
        duplicated = out.loc[out["candidate_id"].duplicated(), "candidate_id"].iloc[0]
        raise ValueError(f"candidate_id values must be unique: {duplicated}")
    for field in factor_fields:
        out[field] = out[field].fillna("").astype(str)
    score = pd.to_numeric(out["score_neurodiscovery"], errors="coerce")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("score_neurodiscovery must be finite")
    out["score_neurodiscovery"] = score.astype(float)
    return out.reset_index(drop=True)


def validate_hidden_outcomes(
    public: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    external: bool = False,
) -> pd.DataFrame:
    """Validate internal or external labels without attaching them to public data."""

    required = {"candidate_id", "validated"}
    if external:
        required.add("executable")
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"hidden outcomes are missing columns: {missing}")
    out = outcomes.copy()
    out["candidate_id"] = out["candidate_id"].astype(str).str.strip()
    if out["candidate_id"].duplicated().any():
        raise ValueError("hidden outcome candidate_id values must be unique")
    public_ids = set(public["candidate_id"].astype(str))
    outcome_ids = set(out["candidate_id"])
    unknown = sorted(outcome_ids - public_ids)
    if unknown:
        raise ValueError(f"hidden outcomes reference unknown candidates: {unknown[:5]}")
    if not external and outcome_ids != public_ids:
        missing_ids = sorted(public_ids - outcome_ids)
        raise ValueError(
            "internal outcomes must label every public candidate; missing "
            f"{missing_ids[:5]}"
        )
    out["validated"] = _coerce_bool_series(out["validated"], column="validated")
    if "feedback_status" in out.columns:
        statuses = (
            out["feedback_status"].fillna("").astype(str).str.strip().str.casefold()
        )
        unknown = sorted(set(statuses) - set(FEEDBACK_STATUSES))
        if unknown:
            raise ValueError(f"feedback_status contains invalid values: {unknown[:5]}")
        supported_mismatch = out["validated"] != statuses.eq("supported")
        if supported_mismatch.any():
            raise ValueError(
                "validated must be true exactly for feedback_status=supported"
            )
        out["feedback_status"] = statuses
    if "execution_succeeded" in out.columns:
        out["execution_succeeded"] = _coerce_bool_series(
            out["execution_succeeded"], column="execution_succeeded"
        )
        impossible = out["validated"] & ~out["execution_succeeded"]
        if impossible.any():
            raise ValueError("a failed execution cannot be validated")
    if "strict_validated" in out.columns:
        out["strict_validated"] = _coerce_bool_series(
            out["strict_validated"], column="strict_validated"
        )
    if "feedback_available" in out.columns:
        out["feedback_available"] = _coerce_bool_series(
            out["feedback_available"], column="feedback_available"
        )
    if "feedback_utility" in out.columns:
        utility = pd.to_numeric(out["feedback_utility"], errors="coerce")
        available = (
            out["feedback_available"]
            if "feedback_available" in out.columns
            else pd.Series(True, index=out.index)
        )
        if utility[available].isna().any():
            raise ValueError(
                "feedback_utility must be finite when feedback is available"
            )
        finite = utility.notna()
        if ((utility[finite] < 0.0) | (utility[finite] > 1.0)).any():
            raise ValueError("feedback_utility must be in [0, 1]")
        out["feedback_utility"] = utility
    if external:
        out["executable"] = _coerce_bool_series(out["executable"], column="executable")
        invalid = out["validated"] & ~out["executable"]
        if invalid.any():
            raise ValueError(
                "an external candidate cannot validate when not executable"
            )
    return out.reset_index(drop=True)


def align_outcome_array(
    public: pd.DataFrame,
    outcomes: pd.DataFrame,
    column: str,
    *,
    default: bool = False,
) -> np.ndarray:
    lookup = outcomes.set_index("candidate_id")[column]
    aligned = public["candidate_id"].map(lookup)
    return aligned.fillna(default).astype(bool).to_numpy()


def validate_full_permutation(order: np.ndarray, n_candidates: int) -> np.ndarray:
    compact = np.asarray(order, dtype=np.int64)
    if compact.ndim != 1 or len(compact) != n_candidates:
        raise ValueError("ranking must contain every candidate exactly once")
    if compact.size and (compact.min() < 0 or compact.max() >= n_candidates):
        raise ValueError("ranking contains an out-of-range candidate index")
    if len(np.unique(compact)) != n_candidates:
        raise ValueError("ranking is not a permutation")
    return compact


def random_walk_order(
    public: pd.DataFrame,
    *,
    factor_fields: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Walk the factor-sharing candidate graph, with random restarts."""

    n = len(public)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    values = {field: public[field].astype(str).to_numpy() for field in factor_fields}
    groups: dict[tuple[str, str], np.ndarray] = {}
    pointers: dict[tuple[str, str], int] = {}
    for field in factor_fields:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(values[field]):
            grouped[str(value)].append(index)
        for value, indices in grouped.items():
            shuffled = np.asarray(indices, dtype=np.int64)
            rng.shuffle(shuffled)
            key = (field, value)
            groups[key] = shuffled
            pointers[key] = 0

    fallback = np.arange(n, dtype=np.int64)
    rng.shuffle(fallback)
    fallback_pointer = 0
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    current = int(fallback[0])

    def next_from_group(key: tuple[str, str]) -> int | None:
        indices = groups[key]
        pointer = pointers[key]
        while pointer < len(indices) and visited[int(indices[pointer])]:
            pointer += 1
        pointers[key] = pointer
        if pointer >= len(indices):
            return None
        chosen = int(indices[pointer])
        pointers[key] = pointer + 1
        return chosen

    for position in range(n):
        if visited[current]:
            while fallback_pointer < n and visited[int(fallback[fallback_pointer])]:
                fallback_pointer += 1
            current = int(fallback[fallback_pointer])
        order[position] = current
        visited[current] = True

        fields = list(factor_fields)
        rng.shuffle(fields)
        candidates: list[int] = []
        for field in fields:
            key = (field, str(values[field][current]))
            candidate = next_from_group(key)
            if candidate is not None:
                candidates.append(candidate)
        if candidates:
            current = int(candidates[int(rng.integers(0, len(candidates)))])
            continue
        while fallback_pointer < n and visited[int(fallback[fallback_pointer])]:
            fallback_pointer += 1
        if fallback_pointer < n:
            current = int(fallback[fallback_pointer])

    return validate_full_permutation(order, n)


def _policy_rule_scores(
    public: pd.DataFrame,
    policy: SearchPolicy,
    *,
    factor_fields: Sequence[str],
) -> np.ndarray:
    validate_exact_policy(
        policy,
        expected_schema=policy.schema_version,
        candidate_ids=set(public["candidate_id"].astype(str)),
        rule_fields=frozenset(factor_fields),
    )
    score = np.zeros(len(public), dtype=float)
    hierarchical = (
        str(policy.metadata.get("rule_matching") or "")
        == "hierarchical_partial_plus_joint"
    )
    for rule in policy.rules:
        mask = np.ones(len(public), dtype=bool)
        component_masks: list[np.ndarray] = []
        for field, expected in rule.when.items():
            component = public[field].astype(str).to_numpy() == str(expected)
            component_masks.append(component)
            mask &= component
        if hierarchical and component_masks:
            component_weight = 0.35 * float(rule.weight) / len(component_masks)
            for component in component_masks:
                score[component] += component_weight
            score[mask] += 0.65 * float(rule.weight)
        else:
            score[mask] += float(rule.weight)
    id_to_index = {
        candidate_id: index
        for index, candidate_id in enumerate(public["candidate_id"].astype(str))
    }
    anchor_count = len(policy.anchors)
    for rank, anchor in enumerate(policy.anchors):
        index = id_to_index[anchor.candidate_id]
        score[index] += (
            3.0 + float(anchor.score) + (anchor_count - rank) / max(1, anchor_count)
        )
    return score


def compile_policy_order(
    public: pd.DataFrame,
    policy: SearchPolicy,
    *,
    factor_fields: Sequence[str],
) -> np.ndarray:
    """Compile one exact, outcome-blind policy into a deterministic permutation."""

    score = _policy_rule_scores(public, policy, factor_fields=factor_fields)
    candidate_ids = public["candidate_id"].astype(str).to_numpy()
    order = np.lexsort((candidate_ids, -score)).astype(np.int64)
    return validate_full_permutation(order, len(public))


def load_search_policies(
    path: Path,
    *,
    public: pd.DataFrame,
    factor_fields: Sequence[str],
    expected_schema: str,
) -> dict[tuple[str, int], SearchPolicy]:
    policies: dict[tuple[str, int], SearchPolicy] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            policy = policy_from_payload(
                payload,
                expected_schema=expected_schema,
                rule_fields=frozenset(factor_fields),
            )
            validate_exact_policy(
                policy,
                expected_schema=expected_schema,
                candidate_ids=set(public["candidate_id"].astype(str)),
                rule_fields=frozenset(factor_fields),
            )
            key = (policy.method, policy.trial)
            if key in policies:
                raise ValueError(
                    f"duplicate search policy at line {line_number}: {key}"
                )
            policies[key] = policy
    return policies


def _feedback_posterior(
    successes: int,
    attempts: int,
    *,
    alpha: float,
    beta: float,
) -> float:
    return (successes + alpha) / (attempts + alpha + beta)


def _select_diverse_batch(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    public: pd.DataFrame,
    *,
    factor_fields: Sequence[str],
    batch_size: int,
    diversity_penalty: float,
) -> np.ndarray:
    """Greedy public-factor diversity without hard-coded task labels."""

    available = np.asarray(candidate_indices, dtype=np.int64)
    if len(available) <= batch_size:
        return available[np.argsort(-scores[available], kind="stable")]

    pool_size = min(len(available), max(batch_size * 8, batch_size + 256))
    local_scores = scores[available]
    if pool_size < len(available):
        top_local = np.argpartition(local_scores, -pool_size)[-pool_size:]
        pool = available[top_local]
    else:
        pool = available
    pool = pool[np.lexsort((pool, -scores[pool]))]

    values = {field: public[field].astype(str).to_numpy() for field in factor_fields}
    target_unique = max(2, int(math.ceil(math.sqrt(batch_size))))
    cap_by_field: dict[str, int] = {}
    for field in factor_fields:
        unique = max(1, len(np.unique(values[field][pool])))
        denominator = min(unique, target_unique)
        cap_by_field[field] = max(
            1,
            int(
                math.ceil(
                    batch_size / denominator * (1.0 + max(0.0, diversity_penalty) * 2.0)
                )
            ),
        )

    selected: list[int] = []
    deferred: list[int] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for raw_index in pool:
        index = int(raw_index)
        exceeds = any(
            counts[(field, str(values[field][index]))] >= cap_by_field[field]
            for field in factor_fields
        )
        if exceeds:
            deferred.append(index)
            continue
        selected.append(index)
        for field in factor_fields:
            counts[(field, str(values[field][index]))] += 1
        if len(selected) >= batch_size:
            break
    if len(selected) < batch_size:
        chosen = set(selected)
        for index in itertools.chain(deferred, map(int, pool)):
            if index in chosen:
                continue
            selected.append(index)
            chosen.add(index)
            if len(selected) >= batch_size:
                break
    return np.asarray(selected, dtype=np.int64)


def _legacy_closed_loop_neurodiscovery_order(
    public: pd.DataFrame,
    validated: np.ndarray,
    *,
    factor_fields: Sequence[str],
    rng: np.random.Generator,
    config: ClosedLoopConfig = ClosedLoopConfig(),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rank candidates in batches and update only from already executed labels."""

    config.validate()
    n = len(public)
    labels = np.asarray(validated, dtype=bool)
    if labels.shape != (n,):
        raise ValueError("validated label vector does not match candidate registry")
    base = _normalize_score(public["score_neurodiscovery"])
    # A small, seeded Plackett-Luce perturbation makes repeated trials genuine
    # searches while preserving the outcome-blind KG ranking as the main signal.
    trial_perturbation = rng.gumbel(0.0, float(config.sampling_temperature), size=n)
    values = {field: public[field].astype(str).to_numpy() for field in factor_fields}
    factor_codes: dict[str, np.ndarray] = {}
    factor_attempted: dict[str, np.ndarray] = {}
    factor_successful: dict[str, np.ndarray] = {}
    for field in factor_fields:
        codes, uniques = pd.factorize(values[field], sort=True)
        factor_codes[field] = codes.astype(np.int64)
        factor_attempted[field] = np.zeros(len(uniques), dtype=np.int64)
        factor_successful[field] = np.zeros(len(uniques), dtype=np.int64)
    pair_codes: dict[tuple[str, str], np.ndarray] = {}
    pair_attempted: dict[tuple[str, str], np.ndarray] = {}
    pair_successful: dict[tuple[str, str], np.ndarray] = {}
    for left, right in itertools.combinations(factor_fields, 2):
        tuples = pd.MultiIndex.from_arrays([values[left], values[right]])
        codes, uniques = pd.factorize(tuples, sort=True)
        key = (left, right)
        pair_codes[key] = codes.astype(np.int64)
        pair_attempted[key] = np.zeros(len(uniques), dtype=np.int64)
        pair_successful[key] = np.zeros(len(uniques), dtype=np.int64)
    executed = np.zeros(n, dtype=bool)
    order: list[int] = []
    trace: list[dict[str, Any]] = []
    batch_index = 0

    feedback_horizon = min(n, config.feedback_horizon or n)
    effective_batch_size = max(
        config.batch_size,
        int(math.ceil(feedback_horizon / config.max_feedback_rounds)),
    )

    while len(order) < feedback_horizon:
        remaining = np.flatnonzero(~executed)
        score = base.copy() + trial_perturbation
        if batch_index >= config.warmup_batches:
            global_rate = _feedback_posterior(
                int(labels[executed].sum()),
                int(executed.sum()),
                alpha=config.prior_alpha,
                beta=config.prior_beta,
            )
            factor_sum = np.zeros(n, dtype=float)
            observation_count = np.zeros(n, dtype=float)
            for field in factor_fields:
                codes = factor_codes[field]
                attempts = factor_attempted[field]
                successes = factor_successful[field]
                posterior = (successes + config.prior_alpha) / (
                    attempts + config.prior_alpha + config.prior_beta
                )
                factor_sum += posterior[codes]
                observation_count += attempts[codes]
            if factor_fields:
                score += config.feedback_weight * (
                    factor_sum / len(factor_fields) - global_rate
                )

            pair_sum = np.zeros(n, dtype=float)
            for key, codes in pair_codes.items():
                attempts = pair_attempted[key]
                successes = pair_successful[key]
                posterior = (successes + config.prior_alpha) / (
                    attempts + config.prior_alpha + config.prior_beta
                )
                pair_sum += posterior[codes]
                observation_count += attempts[codes]
            if pair_codes:
                score += config.pair_feedback_weight * (
                    pair_sum / len(pair_codes) - global_rate
                )
            score += config.exploration_weight * np.sqrt(
                math.log(int(executed.sum()) + 2.0) / (observation_count + 1.0)
            )

        selected = _select_diverse_batch(
            remaining,
            score,
            public,
            factor_fields=factor_fields,
            batch_size=min(
                effective_batch_size,
                feedback_horizon - len(order),
                len(remaining),
            ),
            diversity_penalty=config.diversity_penalty,
        )
        selected_labels = labels[selected].astype(np.int64)
        executed[selected] = True
        order.extend(map(int, selected))
        for field in factor_fields:
            codes = factor_codes[field][selected]
            factor_attempted[field] += np.bincount(
                codes, minlength=len(factor_attempted[field])
            )
            factor_successful[field] += np.bincount(
                codes,
                weights=selected_labels,
                minlength=len(factor_successful[field]),
            ).astype(np.int64)
        for key, all_codes in pair_codes.items():
            codes = all_codes[selected]
            pair_attempted[key] += np.bincount(
                codes, minlength=len(pair_attempted[key])
            )
            pair_successful[key] += np.bincount(
                codes,
                weights=selected_labels,
                minlength=len(pair_successful[key]),
            ).astype(np.int64)
        trace.append(
            {
                "batch": batch_index,
                "start_rank": len(order) - len(selected) + 1,
                "end_rank": len(order),
                "candidate_ids": public.iloc[selected]["candidate_id"]
                .astype(str)
                .tolist(),
                "validated_in_batch": int(labels[selected].sum()),
                "cumulative_validated": int(labels[np.asarray(order, dtype=int)].sum()),
            }
        )
        batch_index += 1

    if len(order) < n:
        remaining = np.flatnonzero(~executed)
        tail_score = base + trial_perturbation
        tail = remaining[np.lexsort((remaining, -tail_score[remaining]))]
        order.extend(map(int, tail))

    compact = validate_full_permutation(np.asarray(order, dtype=np.int64), n)
    return compact, trace


def closed_loop_neurodiscovery_order(
    public: pd.DataFrame,
    validated: np.ndarray,
    *,
    factor_fields: Sequence[str],
    rng: np.random.Generator,
    config: ClosedLoopConfig = ClosedLoopConfig(),
    task: str = "generic",
    seed: int = 0,
    trial: int = 0,
    outcomes: pd.DataFrame | None = None,
    overlay_path: Path | None = None,
    return_overlay_manifest: bool = False,
    audit_records: bool = True,
    collect_trace: bool = True,
) -> (
    tuple[np.ndarray, list[dict[str, Any]]]
    | tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]
):
    """Run the shared semantic overlay engine while preserving the v1 API."""

    labels = np.asarray(validated, dtype=bool)
    if labels.shape != (len(public),):
        raise ValueError("validated label vector does not match candidate registry")
    if outcomes is None:
        outcomes = pd.DataFrame(
            {
                "candidate_id": public["candidate_id"].astype(str),
                "validated": labels,
            }
        )
    adapter = adapter_for(task, factor_fields=factor_fields)
    order, trace, overlay = run_closed_loop_order(
        public,
        outcomes,
        adapter=adapter,
        factor_fields=factor_fields,
        rng=rng,
        config=config,
        seed=seed,
        trial=trial,
        overlay_path=overlay_path,
        audit_records=audit_records,
        collect_trace=collect_trace,
    )
    compact = validate_full_permutation(order, len(public))
    if return_overlay_manifest:
        return compact, trace, overlay
    return compact, trace


def build_rankings(
    public: pd.DataFrame,
    internal_outcomes: pd.DataFrame,
    *,
    factor_fields: Sequence[str],
    methods: Sequence[str],
    n_trials: int,
    seed: int,
    task: str = "generic",
    policies: Mapping[tuple[str, int], SearchPolicy] | None = None,
    config: ClosedLoopConfig = ClosedLoopConfig(),
    overlay_dir: Path | None = None,
) -> tuple[list[RankingRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    if n_trials < 2:
        raise ValueError("formal comparisons require at least two trials")
    labels = align_outcome_array(public, internal_outcomes, "validated")
    policies = policies or {}
    records: list[RankingRecord] = []
    traces: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    for trial in range(n_trials):
        for method_index, method in enumerate(methods):
            rng = np.random.default_rng(seed + 1009 * trial + 7919 * (method_index + 1))
            if method == "random_walk":
                order = random_walk_order(public, factor_fields=factor_fields, rng=rng)
            elif method == "neurodiscovery":
                overlay_path = (
                    overlay_dir / f"seed_{seed}_trial_{trial:02d}.jsonl"
                    if overlay_dir is not None
                    else None
                )
                order, trace, overlay = closed_loop_neurodiscovery_order(
                    public,
                    labels,
                    factor_fields=factor_fields,
                    rng=rng,
                    config=config,
                    task=task,
                    seed=seed,
                    trial=trial,
                    outcomes=internal_outcomes,
                    overlay_path=overlay_path,
                    return_overlay_manifest=True,
                )
                traces.extend(
                    {"method": method, "trial": trial, **batch} for batch in trace
                )
                overlays.append(overlay)
            else:
                policy = policies.get((method, trial))
                if policy is None:
                    raise ValueError(f"missing SearchPolicy for {method} trial {trial}")
                order = compile_policy_order(
                    public,
                    policy,
                    factor_fields=factor_fields,
                )
            records.append(RankingRecord(method, trial, order))
    return records, traces, overlays


def freeze_rankings(
    public: pd.DataFrame,
    records: Sequence[RankingRecord],
    *,
    output_dir: Path,
    task: str,
    factor_fields: Sequence[str],
    input_files: Mapping[str, Path],
) -> dict[str, Any]:
    """Persist discovery orders before any external outcome file is opened."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "frozen_public_candidates.csv.gz"
    public.to_csv(candidates_path, index=False, compression="gzip")
    payload: dict[str, np.ndarray] = {}
    order_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        order = validate_full_permutation(record.order, len(public)).astype(np.int32)
        key = f"order_{index:04d}"
        payload[key] = order
        order_records.append(
            {
                "array_key": key,
                "method": record.method,
                "trial": int(record.trial),
                "n_slots": int(len(order)),
                "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
            }
        )
    orders_path = output_dir / "frozen_orders.npz"
    np.savez_compressed(orders_path, **payload)
    candidate_ids = public["candidate_id"].astype(str).tolist()
    manifest = {
        "schema_version": FROZEN_SCHEMA,
        "created_at": utc_now(),
        "task": task,
        "external_data_read_before_freeze": False,
        "candidate_registry_schema": PUBLIC_REGISTRY_SCHEMA,
        "candidate_count": len(public),
        "candidate_id_sha256": sha256_lines(candidate_ids),
        "factor_fields": list(factor_fields),
        "candidate_table": {
            "path": str(candidates_path),
            "sha256": sha256_file(candidates_path),
        },
        "orders": {
            "path": str(orders_path),
            "sha256": sha256_file(orders_path),
            "records": order_records,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_files.items()
        },
    }
    manifest_path = output_dir / "frozen_rankings_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def load_frozen_rankings(
    frozen_dir: Path,
    public: pd.DataFrame,
) -> tuple[list[RankingRecord], dict[str, Any]]:
    manifest_path = frozen_dir / "frozen_rankings_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = sha256_lines(public["candidate_id"].astype(str).tolist())
    if manifest["candidate_id_sha256"] != expected:
        raise ValueError("frozen candidate IDs do not match the public registry")
    orders_path = Path(manifest["orders"]["path"])
    if sha256_file(orders_path) != manifest["orders"]["sha256"]:
        raise ValueError("frozen order archive failed SHA-256 verification")
    archive = np.load(orders_path, allow_pickle=False)
    records: list[RankingRecord] = []
    for item in manifest["orders"]["records"]:
        order = np.asarray(archive[item["array_key"]], dtype=np.int64)
        compact = order.astype(np.int32)
        if hashlib.sha256(compact.tobytes()).hexdigest() != item["order_sha256"]:
            raise ValueError(f"frozen order failed verification: {item['array_key']}")
        records.append(
            RankingRecord(
                method=str(item["method"]),
                trial=int(item["trial"]),
                order=validate_full_permutation(order, len(public)),
            )
        )
    manifest["reused"] = True
    return records, manifest


def load_precomputed_baseline_rankings(
    manifest_path: Path,
    public: pd.DataFrame,
    *,
    methods: Sequence[str],
    n_trials: int,
) -> tuple[list[RankingRecord], dict[str, Any]]:
    """Load outcome-blind baseline orders while excluding NeuroDiscovery itself."""

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("external_data_read_before_freeze")):
        raise ValueError("precomputed baseline rankings accessed external outcomes")
    expected_candidates = sha256_lines(public["candidate_id"].astype(str).tolist())
    if manifest.get("candidate_id_sha256") != expected_candidates:
        raise ValueError("precomputed baseline candidate IDs do not match the registry")

    orders = dict(manifest.get("orders") or {})
    orders_path = Path(str(orders.get("path") or "")).resolve()
    if not orders_path.is_file():
        raise FileNotFoundError(orders_path)
    if sha256_file(orders_path) != str(orders.get("sha256") or ""):
        raise ValueError(
            "precomputed baseline order archive failed SHA-256 verification"
        )

    requested = tuple(
        method for method in methods if method not in {"random_walk", "neurodiscovery"}
    )
    expected_pairs = {
        (method, trial) for method in requested for trial in range(n_trials)
    }
    archive = np.load(orders_path, allow_pickle=False)
    records: list[RankingRecord] = []
    seen: set[tuple[str, int]] = set()
    for item in orders.get("records") or []:
        method = str(item.get("method") or "")
        trial = int(item.get("trial", -1))
        pair = (method, trial)
        if pair not in expected_pairs:
            continue
        if pair in seen:
            raise ValueError(f"duplicate precomputed ranking: {method} trial {trial}")
        key = str(item.get("array_key") or "")
        if key not in archive:
            raise ValueError(f"precomputed order is absent from archive: {key}")
        order = np.asarray(archive[key], dtype=np.int64)
        compact = validate_full_permutation(order, len(public)).astype(np.int32)
        if hashlib.sha256(compact.tobytes()).hexdigest() != item.get("order_sha256"):
            raise ValueError(f"precomputed order failed verification: {key}")
        records.append(RankingRecord(method, trial, compact))
        seen.add(pair)

    missing = sorted(expected_pairs - seen)
    if missing:
        raise ValueError(f"precomputed baseline rankings are incomplete: {missing[:8]}")
    audit = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "orders_path": str(orders_path),
        "orders_sha256": sha256_file(orders_path),
        "methods": list(requested),
        "trials": int(n_trials),
        "external_data_read_before_freeze": False,
        "neurodiscovery_ranking_reused": False,
    }
    return records, audit


def evaluate_rankings(
    records: Sequence[RankingRecord],
    validated: np.ndarray,
    *,
    budgets: Sequence[int],
    recall_targets: Sequence[float],
    executable: np.ndarray | None = None,
    scope: str = "internal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.asarray(validated, dtype=bool)
    executable_array = (
        np.ones(len(labels), dtype=bool)
        if executable is None
        else np.asarray(executable, dtype=bool)
    )
    if labels.shape != executable_array.shape:
        raise ValueError("validated and executable arrays must have equal shape")
    if np.any(labels & ~executable_array):
        raise ValueError("validated candidates must be executable")
    total = int(labels.sum())
    metric_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for record in records:
        order = validate_full_permutation(record.order, len(labels))
        ordered_labels = labels[order]
        ordered_executable = executable_array[order]
        cumulative_hits = np.cumsum(ordered_labels.astype(np.int64))
        cumulative_executable = np.cumsum(ordered_executable.astype(np.int64))
        for budget in sorted(set(map(int, budgets))):
            if budget < 1 or budget > len(order):
                continue
            hits = int(cumulative_hits[budget - 1])
            executed = int(cumulative_executable[budget - 1])
            metric_rows.append(
                {
                    "scope": scope,
                    "method": record.method,
                    "trial": record.trial,
                    "experiments": budget,
                    "hits": hits,
                    "recall": hits / total if total else np.nan,
                    "precision": hits / executed if executed else np.nan,
                    "executable": executed,
                    "gt_total": total,
                }
            )
        hit_positions = np.flatnonzero(ordered_labels) + 1
        executable_prefix = np.cumsum(ordered_executable.astype(np.int64))
        for target in recall_targets:
            if not 0 < float(target) <= 1:
                raise ValueError("recall targets must be in (0, 1]")
            needed = int(math.ceil(total * float(target)))
            if needed and len(hit_positions) >= needed:
                tcp_cost = int(hit_positions[needed - 1])
                executable_cost = int(executable_prefix[tcp_cost - 1])
            else:
                tcp_cost = np.nan
                executable_cost = np.nan
            cost_rows.append(
                {
                    "scope": scope,
                    "method": record.method,
                    "trial": record.trial,
                    "recall_target": float(target),
                    "hits_needed": needed,
                    "experiments_required": tcp_cost,
                    "executable_experiments_required": executable_cost,
                    "gt_total": total,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(cost_rows)


def aggregate_with_variance(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    excluded = set(group_columns) | {"trial"}
    numeric = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        row["n_trials"] = int(group["trial"].nunique()) if "trial" in group else 1
        for column in numeric:
            values = (
                pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            )
            if not len(values):
                continue
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_variance"] = variance
        rows.append(row)
    return pd.DataFrame(rows)


def paired_randomization_p_value(
    neurodiscovery: Sequence[float],
    baseline: Sequence[float],
    *,
    alternative: str,
) -> float:
    """Exact one-sided paired sign-flip test for the usual ten-seed design."""

    left = np.asarray(neurodiscovery, dtype=float)
    right = np.asarray(baseline, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    differences = left[finite] - right[finite]
    if not len(differences) or np.allclose(differences, 0.0):
        return 1.0
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    observed = float(np.mean(differences))
    if len(differences) <= 20:
        extreme = 0
        total = 1 << len(differences)
        for bits in range(total):
            signs = np.fromiter(
                (
                    1.0 if bits & (1 << index) else -1.0
                    for index in range(len(differences))
                ),
                dtype=float,
            )
            statistic = float(np.mean(differences * signs))
            if alternative == "greater" and statistic >= observed - 1e-15:
                extreme += 1
            if alternative == "less" and statistic <= observed + 1e-15:
                extreme += 1
        return extreme / total
    rng = np.random.default_rng(0)
    extreme = 0
    total = 100_000
    for _ in range(total):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        statistic = float(np.mean(differences * signs))
        if alternative == "greater" and statistic >= observed:
            extreme += 1
        if alternative == "less" and statistic <= observed:
            extreme += 1
    return (extreme + 1) / (total + 1)


def recovery_curve_auc_by_trial(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize recall across the informative, non-saturated search horizon."""

    required = {"scope", "method", "trial", "experiments", "recall"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"metrics missing recovery-AUC columns: {missing}")

    rows: list[dict[str, Any]] = []
    for scope, scoped in metrics.groupby("scope", sort=False):
        saturation = scoped.groupby("experiments", sort=True)["recall"].min()
        informative = saturation[saturation < 1.0 - 1e-12].index.to_numpy(dtype=float)
        if len(informative):
            horizon = float(np.max(informative))
        else:
            horizon = float(np.min(saturation.index.to_numpy(dtype=float)))
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("recovery-AUC horizon must be positive and finite")

        eligible = scoped[pd.to_numeric(scoped["experiments"]) <= horizon]
        for (method, trial), group in eligible.groupby(["method", "trial"], sort=False):
            ordered = group.sort_values("experiments")
            experiments = ordered["experiments"].to_numpy(dtype=float)
            recall = ordered["recall"].to_numpy(dtype=float)
            if not len(experiments):
                continue
            x = np.concatenate(([0.0], experiments))
            y = np.concatenate(([0.0], recall))
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "trial": int(trial),
                    "recovery_auc": float(np.trapezoid(y, x) / horizon),
                    "auc_horizon": horizon,
                    "auc_points": int(len(experiments)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(
    metrics: pd.DataFrame,
    recall_costs: pd.DataFrame,
    *,
    target_method: str = "neurodiscovery",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baselines = sorted(set(metrics["method"].astype(str)) - {target_method})
    for (scope, experiments), group in metrics.groupby(
        ["scope", "experiments"], sort=False
    ):
        target = group[group["method"] == target_method].set_index("trial")
        for baseline in baselines:
            other = group[group["method"] == baseline].set_index("trial")
            joined = target[["hits"]].join(
                other[["hits"]], lsuffix="_target", rsuffix="_baseline", how="inner"
            )
            rows.append(
                {
                    "scope": scope,
                    "comparison": "same_experiments_hits",
                    "value": experiments,
                    "target": target_method,
                    "baseline": baseline,
                    "n_pairs": len(joined),
                    "p_value": paired_randomization_p_value(
                        joined["hits_target"],
                        joined["hits_baseline"],
                        alternative="greater",
                    ),
                }
            )
    for (scope, target_recall), group in recall_costs.groupby(
        ["scope", "recall_target"], sort=False
    ):
        target = group[group["method"] == target_method].set_index("trial")
        for baseline in baselines:
            other = group[group["method"] == baseline].set_index("trial")
            joined = (
                target[["experiments_required"]]
                .join(
                    other[["experiments_required"]],
                    lsuffix="_target",
                    rsuffix="_baseline",
                    how="inner",
                )
                .dropna()
            )
            rows.append(
                {
                    "scope": scope,
                    "comparison": "same_recall_experiments",
                    "value": target_recall,
                    "target": target_method,
                    "baseline": baseline,
                    "n_pairs": len(joined),
                    "p_value": paired_randomization_p_value(
                        joined["experiments_required_target"],
                        joined["experiments_required_baseline"],
                        alternative="less",
                    ),
                }
            )
    recovery_auc = recovery_curve_auc_by_trial(metrics)
    for scope, group in recovery_auc.groupby("scope", sort=False):
        target = group[group["method"] == target_method].set_index("trial")
        for baseline in baselines:
            other = group[group["method"] == baseline].set_index("trial")
            joined = (
                target[["recovery_auc"]]
                .join(
                    other[["recovery_auc"]],
                    lsuffix="_target",
                    rsuffix="_baseline",
                    how="inner",
                )
                .dropna()
            )
            rows.append(
                {
                    "scope": scope,
                    "comparison": "recovery_curve_auc",
                    "value": float(group["auc_horizon"].iloc[0]),
                    "target": target_method,
                    "baseline": baseline,
                    "n_pairs": len(joined),
                    "p_value": paired_randomization_p_value(
                        joined["recovery_auc_target"],
                        joined["recovery_auc_baseline"],
                        alternative="greater",
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_kg_delta(
    public: pd.DataFrame,
    internal_outcomes: pd.DataFrame,
    traces: Sequence[Mapping[str, Any]],
    *,
    path: Path,
    task: str,
    overlay_manifests: Sequence[Mapping[str, Any]] | None = None,
    factor_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bundle per-seed overlays into one tamper-evident experimental ledger."""

    overlay_manifests = list(overlay_manifests or [])
    source_records: list[dict[str, Any]] = []
    compact_overlay_index = bool(overlay_manifests) and any(
        str(item.get("storage_format") or "") == "jsonl.gz"
        for item in overlay_manifests
    )
    if overlay_manifests:
        for manifest in overlay_manifests:
            source_path = Path(str(manifest.get("path") or ""))
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"experimental overlay is missing: {source_path}"
                )
            if sha256_file(source_path) != str(manifest.get("sha256") or ""):
                raise ValueError(f"experimental overlay hash mismatch: {source_path}")
            if compact_overlay_index:
                source_records.append(
                    {
                        "schema_version": "experimental-overlay-reference.v1",
                        "task": task,
                        "case_study_id": manifest.get("case_study_id"),
                        "seed": int(manifest.get("seed") or 0),
                        "trial": int(manifest.get("trial") or 0),
                        "status": "overlay_reference",
                        "source_overlay": {
                            "path": str(source_path),
                            "sha256": manifest.get("sha256"),
                            "storage_format": manifest.get("storage_format"),
                            "records": int(manifest.get("records") or 0),
                            "records_by_status": manifest.get("records_by_status")
                            or {},
                            "final_chain_hash": manifest.get("final_chain_hash"),
                        },
                        "assertions": [
                            {
                                "predicate": "audits_experimental_overlay",
                                "object": str(manifest.get("final_chain_hash") or ""),
                            }
                        ],
                    }
                )
            else:
                for line in source_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        source_records.append(json.loads(line))
    else:
        outcome_lookup = internal_outcomes.set_index("candidate_id", drop=False)
        adapter = adapter_for(task, factor_fields=factor_fields)
        for trace in traces:
            for candidate_id in trace["candidate_ids"]:
                candidate = public.loc[
                    public["candidate_id"].astype(str).eq(str(candidate_id))
                ].iloc[0]
                outcome = outcome_lookup.loc[str(candidate_id)].to_dict()
                source_records.append(
                    {
                        "schema_version": "experimental-claim.v2",
                        "task": task,
                        "case_study_id": adapter.case_study_id,
                        "method": str(trace["method"]),
                        "trial": int(trace["trial"]),
                        "round": int(trace["batch"]),
                        "candidate_id": str(candidate_id),
                        "hypothesis_id": str(candidate_id),
                        "status": adapter.classify_feedback(outcome),
                        "candidate_tuple": {
                            field: str(candidate.get(field, ""))
                            for field in adapter.factor_fields
                        },
                        "assertions": adapter.semantic_assertions(candidate.to_dict()),
                    }
                )

    previous_hash = "0" * 64
    count = 0
    status_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for source in source_records:
            payload = dict(source)
            source_hash = payload.pop("record_hash", None)
            payload.pop("previous_hash", None)
            payload["schema_version"] = "experimental-kg-delta.v2"
            payload["source_overlay_record_hash"] = source_hash
            payload["previous_hash"] = previous_hash
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            payload["record_hash"] = record_hash
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            previous_hash = record_hash
            status_counts[str(payload.get("status") or "")] += 1
            count += 1
    represented_records = count
    if compact_overlay_index:
        represented_records = sum(
            int(item.get("records") or 0) for item in overlay_manifests
        )
        status_counts = Counter()
        for item in overlay_manifests:
            status_counts.update(
                {
                    str(status): int(value)
                    for status, value in (item.get("records_by_status") or {}).items()
                }
            )
    consumed = bool(overlay_manifests) and all(
        bool(item.get("feedback_consumed_during_ranking")) for item in overlay_manifests
    )
    semantic_verified = (
        all(
            bool(item.get("semantic_projection_verified")) for item in overlay_manifests
        )
        if compact_overlay_index
        else bool(source_records)
        and all(bool(item.get("assertions")) for item in source_records)
    )
    return {
        "schema_version": "experimental-kg-overlay-bundle.v2",
        "path": str(path),
        "sha256": sha256_file(path),
        "records": represented_records,
        "ledger_records": count,
        "storage_mode": (
            "compressed_overlay_index" if compact_overlay_index else "expanded_jsonl"
        ),
        "records_by_status": dict(sorted(status_counts.items())),
        "final_chain_hash": previous_hash,
        "source_overlays": overlay_manifests,
        "overlay_count": len(overlay_manifests),
        "feedback_consumed_during_ranking": consumed,
        "nonzero_feedback_reads": int(
            sum(
                int(item.get("nonzero_feedback_reads") or 0)
                for item in overlay_manifests
            )
        ),
        "selection_changed_batches": int(
            sum(
                int(item.get("selection_changed_batches") or 0)
                for item in overlay_manifests
            )
        ),
        "semantic_projection_verified": semantic_verified,
        "per_seed_isolation_verified": len(
            {
                (int(item.get("seed") or 0), int(item.get("trial") or 0))
                for item in overlay_manifests
            }
        )
        == len(overlay_manifests),
        "mutates_formal_kg": False,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_fields = tuple(args.factor_fields)

    public_raw = _read_csv(args.public_candidates)
    policy_path = getattr(args, "neurodiscovery_config", None)
    policy_audit: dict[str, Any] | None = None
    policy = None
    if policy_path:
        policy = load_neurodiscovery_policy(policy_path, task=args.task)
        public_raw, policy_audit = apply_neurodiscovery_policy(public_raw, policy)
    score_components_path = getattr(args, "score_components", None)
    score_manifest_path = getattr(args, "score_components_manifest", None)
    if bool(score_components_path) != bool(score_manifest_path):
        raise ValueError(
            "--score-components and --score-components-manifest must be provided together"
        )
    if score_components_path:
        public_raw, score_component_audit = load_score_component_bundle(
            public_raw,
            table_path=score_components_path,
            manifest_path=score_manifest_path,
        )
    else:
        score_component_audit = embedded_score_component_audit(public_raw)
    public = validate_public_registry(public_raw, factor_fields=factor_fields)
    internal_raw = _read_csv(args.internal_outcomes)
    internal = validate_hidden_outcomes(public, internal_raw)

    policies: dict[tuple[str, int], SearchPolicy] = {}
    precomputed_records: list[RankingRecord] = []
    precomputed_audit: dict[str, Any] | None = None
    input_files: dict[str, Path] = {
        "public_candidates": args.public_candidates.resolve(),
        "internal_outcomes": args.internal_outcomes.resolve(),
    }
    if policy_path:
        input_files["neurodiscovery_config"] = policy_path.resolve()
    if score_components_path:
        input_files["score_components"] = score_components_path.resolve()
        input_files["score_components_manifest"] = score_manifest_path.resolve()
    if args.search_policies:
        policies = load_search_policies(
            args.search_policies,
            public=public,
            factor_fields=factor_fields,
            expected_schema=args.policy_schema,
        )
        input_files["search_policies"] = args.search_policies.resolve()
    precomputed_path = getattr(args, "precomputed_rankings_manifest", None)
    if precomputed_path:
        if args.search_policies:
            raise ValueError(
                "search policies and precomputed baseline rankings are mutually exclusive"
            )
        precomputed_records, precomputed_audit = load_precomputed_baseline_rankings(
            precomputed_path,
            public,
            methods=args.methods,
            n_trials=args.trials,
        )
        input_files["precomputed_rankings_manifest"] = precomputed_path.resolve()

    default_horizon = (
        args.feedback_horizon
        if args.feedback_horizon is not None
        else max(args.budgets)
    )
    if policy is not None:
        config = ClosedLoopConfig(
            **policy.closed_loop_kwargs(default_horizon=default_horizon)
        )
    else:
        config = ClosedLoopConfig(
            batch_size=args.batch_size,
            warmup_batches=args.warmup_batches,
            sampling_temperature=float(getattr(args, "sampling_temperature", 0.01)),
            max_feedback_rounds=args.max_feedback_rounds,
            feedback_horizon=default_horizon,
        )
    config.validate()
    generated_methods = [
        method
        for method in args.methods
        if precomputed_audit is None or method in {"random_walk", "neurodiscovery"}
    ]
    records, traces, overlay_manifests = build_rankings(
        public,
        internal,
        factor_fields=factor_fields,
        methods=generated_methods,
        n_trials=args.trials,
        seed=args.seed,
        task=args.task,
        policies=policies,
        config=config,
        overlay_dir=output_dir / "experimental_overlays",
    )
    records.extend(precomputed_records)
    method_order = {method: index for index, method in enumerate(args.methods)}
    records.sort(key=lambda item: (item.trial, method_order[item.method]))

    frozen_dir = output_dir / "frozen_discovery_rankings"
    frozen_manifest = freeze_rankings(
        public,
        records,
        output_dir=frozen_dir,
        task=args.task,
        factor_fields=factor_fields,
        input_files=input_files,
    )

    internal_labels = align_outcome_array(public, internal, "validated")
    metrics, costs = evaluate_rankings(
        records,
        internal_labels,
        budgets=args.budgets,
        recall_targets=args.recall_targets,
        scope="internal",
    )
    metrics_path = output_dir / "internal_metrics_by_trial.csv"
    costs_path = output_dir / "internal_recall_cost_by_trial.csv"
    metrics.to_csv(metrics_path, index=False)
    costs.to_csv(costs_path, index=False)
    metric_summary = aggregate_with_variance(
        metrics, ["scope", "method", "experiments"]
    )
    cost_summary = aggregate_with_variance(costs, ["scope", "method", "recall_target"])
    metric_summary_path = output_dir / "internal_metrics_summary.csv"
    cost_summary_path = output_dir / "internal_recall_cost_summary.csv"
    metric_summary.to_csv(metric_summary_path, index=False)
    cost_summary.to_csv(cost_summary_path, index=False)
    internal_auc = recovery_curve_auc_by_trial(metrics)
    internal_auc.to_csv(output_dir / "internal_recovery_auc_by_trial.csv", index=False)
    aggregate_with_variance(internal_auc, ["scope", "method"]).to_csv(
        output_dir / "internal_recovery_auc_summary.csv", index=False
    )

    all_metrics = [metrics]
    all_costs = [costs]
    external_audit: dict[str, Any] | None = None
    if args.external_outcomes:
        # This is intentionally the first external-outcome read in the process.
        if not Path(frozen_manifest["manifest_path"]).is_file():
            raise RuntimeError(
                "discovery rankings were not frozen before external loading"
            )
        external_raw = _read_csv(args.external_outcomes)
        external = validate_hidden_outcomes(public, external_raw, external=True)
        external_labels = align_outcome_array(
            public, external, "validated", default=False
        )
        external_executable = align_outcome_array(
            public, external, "executable", default=False
        )
        ext_metrics, ext_costs = evaluate_rankings(
            records,
            external_labels,
            budgets=args.budgets,
            recall_targets=args.recall_targets,
            executable=external_executable,
            scope="external",
        )
        ext_metrics_path = output_dir / "external_metrics_by_trial.csv"
        ext_costs_path = output_dir / "external_recall_cost_by_trial.csv"
        ext_metrics.to_csv(ext_metrics_path, index=False)
        ext_costs.to_csv(ext_costs_path, index=False)
        aggregate_with_variance(ext_metrics, ["scope", "method", "experiments"]).to_csv(
            output_dir / "external_metrics_summary.csv", index=False
        )
        aggregate_with_variance(ext_costs, ["scope", "method", "recall_target"]).to_csv(
            output_dir / "external_recall_cost_summary.csv", index=False
        )
        external_auc = recovery_curve_auc_by_trial(ext_metrics)
        external_auc.to_csv(
            output_dir / "external_recovery_auc_by_trial.csv", index=False
        )
        aggregate_with_variance(external_auc, ["scope", "method"]).to_csv(
            output_dir / "external_recovery_auc_summary.csv", index=False
        )
        all_metrics.append(ext_metrics)
        all_costs.append(ext_costs)
        external_audit = {
            "path": str(args.external_outcomes.resolve()),
            "sha256": sha256_file(args.external_outcomes.resolve()),
            "loaded_after_freeze": True,
            "executable_candidates": int(external_executable.sum()),
            "validated_candidates": int(external_labels.sum()),
        }

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_costs = pd.concat(all_costs, ignore_index=True)
    p_values = paired_comparisons(combined_metrics, combined_costs)
    p_values_path = output_dir / "paired_p_values.csv"
    p_values.to_csv(p_values_path, index=False)

    trace_path = output_dir / "neurodiscovery_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    delta = write_kg_delta(
        public,
        internal,
        traces,
        path=output_dir / "experimental_kg_delta.jsonl",
        task=args.task,
        overlay_manifests=overlay_manifests,
        factor_fields=factor_fields,
    )

    method_trial_counts = (
        pd.DataFrame(
            [{"method": record.method, "trial": record.trial} for record in records]
        )
        .groupby("method")["trial"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "task": args.task,
        "status": "complete",
        "candidate_count": len(public),
        "factor_fields": list(factor_fields),
        "methods": list(args.methods),
        "trials": args.trials,
        "method_trial_counts": method_trial_counts,
        "budgets": list(args.budgets),
        "recall_targets": list(args.recall_targets),
        "internal_gt_total": int(internal_labels.sum()),
        "external": external_audit,
        "frozen_discovery": frozen_manifest,
        "closed_loop_config": config.__dict__,
        "neurodiscovery_policy": policy_audit,
        "score_component_bundle": score_component_audit,
        "precomputed_baseline_rankings": precomputed_audit,
        "experimental_kg_delta": delta,
        "formal_kg_mutated": False,
        "artifacts": {},
    }
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            manifest["artifacts"][path.name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--public-candidates", type=Path, required=True)
    parser.add_argument("--internal-outcomes", type=Path, required=True)
    parser.add_argument("--external-outcomes", type=Path)
    parser.add_argument(
        "--neurodiscovery-config",
        type=Path,
        help="Frozen outcome-blind NeuroDiscovery policy JSON.",
    )
    parser.add_argument(
        "--score-components",
        type=Path,
        help="Frozen candidate_id plus score_kge/score_novelty/score_critic CSV.",
    )
    parser.add_argument(
        "--score-components-manifest",
        type=Path,
        help="Audit manifest paired with --score-components.",
    )
    parser.add_argument("--search-policies", type=Path)
    parser.add_argument(
        "--precomputed-rankings-manifest",
        type=Path,
        help=(
            "Frozen outcome-blind baseline orders. Random walk and NeuroDiscovery "
            "are always regenerated."
        ),
    )
    parser.add_argument("--policy-schema", default="case-study-search-policy.v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factor-fields", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--sampling-temperature", type=float, default=0.01)
    parser.add_argument("--max-feedback-rounds", type=int, default=128)
    parser.add_argument("--feedback-horizon", type=int)
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000]
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10, 0.20, 0.50],
    )
    return parser


def main() -> None:
    manifest = run_benchmark(build_parser().parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# Updated: 2026-08-11 20:30 HKT
