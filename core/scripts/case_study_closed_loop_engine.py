"""Shared closed-loop engine backed by a per-seed experimental KG overlay."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from core.scripts.case_study_feedback_adapters import (
    CONTRADICTED,
    EXECUTION_FAILED,
    INCONCLUSIVE,
    SUPPORTED,
    CaseStudyFeedbackAdapter,
)


OVERLAY_SCHEMA = "experimental-kg-overlay.v2"
OVERLAY_RECORD_SCHEMA = "experimental-claim.v2"
FEEDBACK_PROJECTIONS = frozenset(
    {"all_factors", "generalizable_factors", "relation_endpoints"}
)
FEEDBACK_MODELS = frozenset(
    {
        "beta_counts",
        "ridge_surrogate",
        "factor_ucb_ridge_surrogate",
        "hierarchical_utility",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(values: pd.Series | np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    finite = np.isfinite(numeric)
    if not finite.any():
        return np.zeros(len(numeric), dtype=float)
    low = float(np.nanmin(numeric[finite]))
    high = float(np.nanmax(numeric[finite]))
    numeric[~finite] = low
    if math.isclose(low, high):
        return np.zeros(len(numeric), dtype=float)
    return (numeric - low) / (high - low)


def _json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _statistics(outcome: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "candidate_id",
        "validated",
        "strict_validated",
        "feedback_status",
        "executable",
        "execution_succeeded",
    }
    return {
        str(key): _json_value(value)
        for key, value in outcome.items()
        if str(key) not in excluded and _json_value(value) not in (None, "")
    }


@dataclass(frozen=True)
class ScoreComponentConfig:
    """Frozen, outcome-blind optional score columns.

    A component is active only when its column is present in the public table.
    This makes manifests distinguish implemented wiring from an actually used
    KGE, novelty model, or critic score.
    """

    kge_column: str = "score_kge"
    kge_weight: float = 0.20
    novelty_column: str = "score_novelty"
    novelty_weight: float = 0.08
    critic_column: str = "score_critic"
    critic_weight: float = 0.08


def compose_static_score(
    public: pd.DataFrame,
    config: ScoreComponentConfig = ScoreComponentConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    base = _normalize(public["score_neurodiscovery"])
    numerator = base.copy()
    denominator = 1.0
    components: dict[str, Any] = {
        "base": {
            "column": "score_neurodiscovery",
            "active": True,
            "weight": 1.0,
        }
    }
    for name, column, weight in (
        ("kge", config.kge_column, config.kge_weight),
        ("novelty", config.novelty_column, config.novelty_weight),
        ("critic", config.critic_column, config.critic_weight),
    ):
        present = column in public.columns
        finite_values = (
            int(np.isfinite(pd.to_numeric(public[column], errors="coerce")).sum())
            if present
            else 0
        )
        if present and finite_values != len(public):
            raise ValueError(f"{column} must be finite for every candidate")
        active = bool(present and weight > 0)
        descriptor = {
            "column": column,
            "weight": float(weight),
            "active": active,
            "finite_values": finite_values,
        }
        if active:
            values = _normalize(public[column])
            numerator += float(weight) * values
            denominator += float(weight)
            descriptor.update(
                {
                    "frozen_before_experiment": True,
                }
            )
        else:
            descriptor["inactive_reason"] = (
                "column_not_present" if not present else "zero_weight"
            )
        components[name] = descriptor
    return numerator / denominator, components


class ExperimentalOverlayGraph:
    """Semantic overlay written as an append-only hash chain."""

    def __init__(
        self,
        public: pd.DataFrame,
        *,
        adapter: CaseStudyFeedbackAdapter,
        factor_fields: Sequence[str],
        seed: int,
        trial: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 3.0,
        feedback_projection: str = "all_factors",
        feedback_model: str = "beta_counts",
        surrogate_ridge_alpha: float = 1.0,
        stream_path: Path | None = None,
        audit_records: bool = True,
    ) -> None:
        self.public = public.reset_index(drop=True)
        self.adapter = adapter
        self.factor_fields = tuple(factor_fields)
        self.seed = int(seed)
        self.trial = int(trial)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.feedback_projection = str(feedback_projection)
        if self.feedback_projection not in FEEDBACK_PROJECTIONS:
            raise ValueError(
                f"unknown feedback projection: {self.feedback_projection!r}"
            )
        self.feedback_model = str(feedback_model)
        if self.feedback_model not in FEEDBACK_MODELS:
            raise ValueError(f"unknown feedback model: {self.feedback_model!r}")
        self.surrogate_ridge_alpha = float(surrogate_ridge_alpha)
        if (
            not math.isfinite(self.surrogate_ridge_alpha)
            or self.surrogate_ridge_alpha <= 0
        ):
            raise ValueError("surrogate_ridge_alpha must be finite and positive")
        self.records: list[dict[str, Any]] = []
        self.record_count = 0
        self.semantic_assertion_count = 0
        self.audit_records = bool(audit_records)
        self.semantic_projection_verified = bool(audit_records)
        self.candidate_ids = self.public["candidate_id"].astype(str).to_numpy()
        self.stream_path = stream_path.resolve() if stream_path is not None else None
        if self.stream_path is not None and not self.audit_records:
            raise ValueError("counts-only overlays cannot be streamed to disk")
        self.stream_handle = None
        if self.stream_path is not None:
            self.stream_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream_handle = gzip.open(
                self.stream_path,
                "wt",
                encoding="utf-8",
                compresslevel=1,
            )
        self.previous_hash = "0" * 64
        self.status_counts: Counter[str] = Counter()
        self.ranking_status_counts: Counter[str] = Counter()
        self.feedback_record_count = 0
        self.utility_record_count = 0
        self.utility_sum = 0.0
        self.utility_indices: list[int] = []
        self.utility_values: list[float] = []
        self.ranking_reads_after_feedback = 0
        self.nonzero_feedback_reads = 0
        self.selection_changed_batches = 0
        self.informative_records_before_ranking = 0

        self.values = {
            field: self.public[field].fillna("").astype(str).to_numpy()
            for field in self.factor_fields
        }
        projection_audit: dict[str, Any] = {}
        if self.feedback_projection == "relation_endpoints":
            endpoint_groups: list[tuple[str, ...]] = []
            relation_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
            for relation in self.adapter.relations:
                subject = tuple(
                    field
                    for field in relation.subject_fields
                    if field in self.factor_fields
                )
                obj = tuple(
                    field
                    for field in relation.object_fields
                    if field in self.factor_fields
                )
                for group in (subject, obj):
                    if group and group not in endpoint_groups:
                        endpoint_groups.append(group)
                pair = (subject, obj)
                if subject and obj and subject != obj and pair not in relation_pairs:
                    relation_pairs.append(pair)
            if not endpoint_groups:
                endpoint_groups = [(field,) for field in self.factor_fields]
            feedback_groups = endpoint_groups
            feedback_pairs = relation_pairs
        elif self.feedback_projection == "generalizable_factors":
            cardinalities = {
                field: int(self.public[field].fillna("").astype(str).nunique())
                for field in self.factor_fields
            }
            cardinality_limit = max(2, int(math.sqrt(max(len(self.public), 1))))
            selected_fields = [
                field
                for field in self.factor_fields
                if 1 < cardinalities[field] <= cardinality_limit
            ]
            if not selected_fields:
                selected_fields = [
                    field for field in self.factor_fields if cardinalities[field] > 1
                ]
            if not selected_fields:
                selected_fields = list(self.factor_fields)
            feedback_groups = [(field,) for field in selected_fields]
            feedback_pairs = list(itertools.combinations(feedback_groups, 2))
            projection_audit = {
                "factor_cardinalities": cardinalities,
                "cardinality_limit": cardinality_limit,
                "selected_fields": selected_fields,
                "selection_rule": (
                    "non-constant factors with cardinality <= floor(sqrt(n)); "
                    "fallback to all non-constant factors"
                ),
            }
        else:
            feedback_groups = [(field,) for field in self.factor_fields]
            feedback_pairs = list(itertools.combinations(feedback_groups, 2))

        self.feedback_groups = tuple(feedback_groups)
        self.feedback_pairs = tuple(feedback_pairs)
        self.feedback_projection_audit = projection_audit
        self.factor_codes: dict[tuple[str, ...], np.ndarray] = {}
        self.factor_counts: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
        for group in self.feedback_groups:
            arrays = [self.values[field] for field in group]
            if len(arrays) == 1:
                codes, uniques = pd.factorize(arrays[0], sort=True)
            else:
                codes, uniques = pd.factorize(
                    pd.MultiIndex.from_arrays(arrays), sort=True
                )
            self.factor_codes[group] = codes.astype(np.int64)
            self.factor_counts[group] = {
                status: np.zeros(len(uniques), dtype=np.int64)
                for status in (SUPPORTED, CONTRADICTED, INCONCLUSIVE, EXECUTION_FAILED)
            }
        self.pair_codes: dict[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray] = {}
        self.pair_counts: dict[
            tuple[tuple[str, ...], tuple[str, ...]], dict[str, np.ndarray]
        ] = {}
        for left, right in self.feedback_pairs:
            tuples = pd.MultiIndex.from_arrays(
                [self.factor_codes[left], self.factor_codes[right]]
            )
            codes, uniques = pd.factorize(tuples, sort=True)
            key = (left, right)
            self.pair_codes[key] = codes.astype(np.int64)
            self.pair_counts[key] = {
                status: np.zeros(len(uniques), dtype=np.int64)
                for status in (SUPPORTED, CONTRADICTED, INCONCLUSIVE, EXECUTION_FAILED)
            }
        self.factor_utility_sum = {
            group: np.zeros_like(next(iter(counts.values())), dtype=float)
            for group, counts in self.factor_counts.items()
        }
        self.factor_utility_count = {
            group: np.zeros_like(next(iter(counts.values())), dtype=np.int64)
            for group, counts in self.factor_counts.items()
        }
        self.pair_utility_sum = {
            key: np.zeros_like(next(iter(counts.values())), dtype=float)
            for key, counts in self.pair_counts.items()
        }
        self.pair_utility_count = {
            key: np.zeros_like(next(iter(counts.values())), dtype=np.int64)
            for key, counts in self.pair_counts.items()
        }
        self.surrogate_design, self.surrogate_design_audit = (
            self._build_surrogate_design()
        )

    def _build_surrogate_design(self) -> tuple[np.ndarray, dict[str, Any]]:
        blocks: list[np.ndarray] = []
        descriptors: list[dict[str, Any]] = []
        max_categories = min(256, len(self.public))
        factor_cardinalities = {
            group: int(codes.max()) + 1 if len(codes) else 0
            for group, codes in self.factor_codes.items()
        }
        for group, codes in self.factor_codes.items():
            categories = factor_cardinalities[group]
            if categories <= 1 or categories > max_categories:
                continue
            blocks.append(np.eye(categories, dtype=float)[codes])
            descriptors.append(
                {"kind": "factor", "fields": list(group), "columns": categories}
            )
        skipped_constant_factor_pairs: list[dict[str, list[str]]] = []
        for (left, right), codes in self.pair_codes.items():
            if factor_cardinalities[left] <= 1 or factor_cardinalities[right] <= 1:
                skipped_constant_factor_pairs.append(
                    {"left": list(left), "right": list(right)}
                )
                continue
            categories = int(codes.max()) + 1 if len(codes) else 0
            if categories <= 1 or categories > max_categories:
                continue
            blocks.append(np.eye(categories, dtype=float)[codes])
            descriptors.append(
                {
                    "kind": "factor_pair",
                    "fields": [list(left), list(right)],
                    "columns": categories,
                }
            )
        kg_columns = []
        for column in sorted(
            name for name in self.public.columns if str(name).startswith("kg_")
        ):
            numeric = pd.to_numeric(self.public[column], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(numeric).all() or np.ptp(numeric) <= 1e-12:
                continue
            blocks.append(_normalize(numeric).reshape(-1, 1))
            kg_columns.append(str(column))
        if not blocks:
            blocks.append(np.zeros((len(self.public), 1), dtype=float))
        design = np.column_stack(blocks)
        return design, {
            "columns": int(design.shape[1]),
            "categorical_blocks": descriptors,
            "skipped_constant_factor_pairs": skipped_constant_factor_pairs,
            "kg_columns": kg_columns,
            "ridge_alpha": self.surrogate_ridge_alpha,
        }

    @property
    def informative_feedback_count(self) -> int:
        if self.feedback_model in {
            "ridge_surrogate",
            "factor_ucb_ridge_surrogate",
            "hierarchical_utility",
        }:
            return int(self.utility_record_count)
        return int(
            self.ranking_status_counts[SUPPORTED]
            + self.ranking_status_counts[CONTRADICTED]
        )

    def append(
        self,
        *,
        candidate_index: int,
        outcome: Mapping[str, Any],
        round_index: int,
    ) -> dict[str, Any]:
        candidate_id = str(self.candidate_ids[int(candidate_index)])
        status = self.adapter.classify_feedback(outcome)
        available_raw = outcome.get("feedback_available", True)
        if isinstance(available_raw, str):
            feedback_available = available_raw.strip().casefold() not in {
                "",
                "0",
                "false",
                "no",
            }
        else:
            feedback_available = bool(available_raw)
        feedback_utility: float | None = None
        if feedback_available and "feedback_utility" in outcome:
            try:
                candidate_utility = float(outcome.get("feedback_utility"))
            except (TypeError, ValueError):
                candidate_utility = float("nan")
            if math.isfinite(candidate_utility):
                if not 0.0 <= candidate_utility <= 1.0:
                    raise ValueError("feedback_utility must be in [0, 1]")
                feedback_utility = candidate_utility
        if not self.audit_records:
            payload = {
                "id": (
                    f"TRANSIENT:{self.seed}:{self.trial}:{round_index}:"
                    f"{int(candidate_index)}"
                ),
                "status": status,
            }
            assertions: list[dict[str, Any]] = []
        else:
            candidate = self.public.iloc[int(candidate_index)].to_dict()
            candidate_tuple = {
                field: str(candidate.get(field, "")) for field in self.factor_fields
            }
            assertions = self.adapter.semantic_assertions(candidate)
            record_id = hashlib.sha256(
                json.dumps(
                    {
                        "task": self.adapter.task,
                        "seed": self.seed,
                        "trial": self.trial,
                        "round": int(round_index),
                        "candidate_id": candidate_id,
                        "status": status,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
            payload = {
                "schema_version": OVERLAY_RECORD_SCHEMA,
                "id": f"EXP_CLAIM:{record_id}",
                "case_study_id": self.adapter.case_study_id,
                "task": self.adapter.task,
                "seed": self.seed,
                "trial": self.trial,
                "round": int(round_index),
                "hypothesis_id": candidate_id,
                "candidate_id": candidate_id,
                "status": status,
                "candidate_tuple": candidate_tuple,
                "assertions": assertions,
                "requires_complete_chain": self.adapter.requires_complete_chain,
                "statistics": _statistics(outcome),
                "provenance": {
                    "evidence_kind": "experimental",
                    "formal_kg_mutated": False,
                    "outcome_observed_after_selection": True,
                    "feedback_available_for_ranking": feedback_available,
                },
                "previous_hash": self.previous_hash,
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            payload["record_hash"] = record_hash
            self.previous_hash = record_hash
        if self.audit_records and self.stream_handle is None:
            self.records.append(payload)
        elif self.stream_handle is not None:
            self.stream_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.record_count += 1
        self.semantic_assertion_count += len(assertions)
        self.semantic_projection_verified = self.semantic_projection_verified and bool(
            assertions
        )
        self.status_counts[status] += 1
        if feedback_available:
            self.feedback_record_count += 1
            self.ranking_status_counts[status] += 1
            for group, codes in self.factor_codes.items():
                code = int(codes[candidate_index])
                self.factor_counts[group][status][code] += 1
                if feedback_utility is not None:
                    self.factor_utility_sum[group][code] += feedback_utility
                    self.factor_utility_count[group][code] += 1
            for key, codes in self.pair_codes.items():
                code = int(codes[candidate_index])
                self.pair_counts[key][status][code] += 1
                if feedback_utility is not None:
                    self.pair_utility_sum[key][code] += feedback_utility
                    self.pair_utility_count[key][code] += 1
            if feedback_utility is not None:
                self.utility_record_count += 1
                self.utility_sum += feedback_utility
                self.utility_indices.append(int(candidate_index))
                self.utility_values.append(feedback_utility)
        return payload

    def _posterior_adjustment(
        self,
        codes: np.ndarray,
        counts: Mapping[str, np.ndarray],
        *,
        global_supported: float,
        global_contradicted: float,
        inconclusive_failure_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        supported = counts[SUPPORTED]
        contradicted = counts[CONTRADICTED]
        informative = supported + contradicted
        attempted = informative + counts[INCONCLUSIVE] + counts[EXECUTION_FAILED]
        search_failures = contradicted + (
            float(inconclusive_failure_weight) * counts[INCONCLUSIVE]
        )
        denominator = supported + search_failures + self.prior_alpha + self.prior_beta
        support_rate = (supported + self.prior_alpha) / denominator
        semantic_denominator = informative + self.prior_alpha + self.prior_beta
        contradiction_rate = (contradicted + self.prior_alpha) / semantic_denominator
        adjustment = (
            support_rate[codes]
            - global_supported
            - 0.75 * (contradiction_rate[codes] - global_contradicted)
        )
        return adjustment, attempted[codes].astype(float)

    def score_candidates(
        self,
        *,
        feedback_weight: float,
        pair_feedback_weight: float,
        exploration_weight: float,
        inconclusive_failure_weight: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n = len(self.public)
        if not self.feedback_record_count:
            return np.zeros(n, dtype=float), {
                "overlay_read": False,
                "records_available": 0,
                "informative_records": 0,
                "nonzero": False,
            }

        informative = self.informative_feedback_count
        if not 0.0 <= float(inconclusive_failure_weight) <= 1.0:
            raise ValueError("inconclusive_failure_weight must be in [0, 1]")
        global_search_failures = self.ranking_status_counts[CONTRADICTED] + (
            float(inconclusive_failure_weight)
            * self.ranking_status_counts[INCONCLUSIVE]
        )
        global_denominator = (
            self.ranking_status_counts[SUPPORTED]
            + global_search_failures
            + self.prior_alpha
            + self.prior_beta
        )
        global_supported = (
            self.ranking_status_counts[SUPPORTED] + self.prior_alpha
        ) / global_denominator
        semantic_global_denominator = informative + self.prior_alpha + self.prior_beta
        global_contradicted = (
            self.ranking_status_counts[CONTRADICTED] + self.prior_alpha
        ) / semantic_global_denominator

        if self.feedback_model == "hierarchical_utility" and self.utility_record_count:
            global_utility = self.utility_sum / self.utility_record_count
            prior_strength = self.prior_alpha + self.prior_beta
            factor_adjustment = np.zeros(n, dtype=float)
            factor_observations = np.zeros(n, dtype=float)
            for group, codes in self.factor_codes.items():
                counts = self.factor_utility_count[group]
                sums = self.factor_utility_sum[group]
                posterior = (sums + prior_strength * global_utility) / (
                    counts + prior_strength
                )
                factor_adjustment += posterior[codes] - global_utility
                factor_observations += counts[codes]
            if self.factor_codes:
                factor_adjustment /= len(self.factor_codes)

            pair_adjustment = np.zeros(n, dtype=float)
            active_pair_blocks = 0
            for key, codes in self.pair_codes.items():
                counts = self.pair_utility_count[key]
                if int(counts.max(initial=0)) < 2:
                    continue
                sums = self.pair_utility_sum[key]
                posterior = (sums + prior_strength * global_utility) / (
                    counts + prior_strength
                )
                observed_twice = counts[codes] >= 2
                pair_adjustment += np.where(
                    observed_twice,
                    posterior[codes] - global_utility,
                    0.0,
                )
                active_pair_blocks += 1
            if active_pair_blocks:
                pair_adjustment /= active_pair_blocks

            semantic_feedback = (
                float(feedback_weight) * factor_adjustment
                + float(pair_feedback_weight) * pair_adjustment
            )
            exploration = float(exploration_weight) * np.sqrt(
                math.log(self.utility_record_count + 2.0) / (factor_observations + 1.0)
            )
            score = semantic_feedback + exploration
            nonzero = bool(np.any(np.abs(semantic_feedback) > 1e-12))
            self.ranking_reads_after_feedback += 1
            self.informative_records_before_ranking += int(informative)
            if nonzero:
                self.nonzero_feedback_reads += 1
            return score, {
                "overlay_read": True,
                "records_available": self.feedback_record_count,
                "informative_records": int(informative),
                "nonzero": nonzero,
                "feedback_model": self.feedback_model,
                "utility_records": int(self.utility_record_count),
                "active_pair_blocks": int(active_pair_blocks),
                "prior_strength": float(prior_strength),
                "mean_semantic_adjustment": float(np.mean(semantic_feedback)),
                "max_abs_semantic_adjustment": float(np.max(np.abs(semantic_feedback))),
                "mean_exploration_bonus": float(np.mean(exploration)),
            }

        if self.feedback_model in {
            "ridge_surrogate",
            "factor_ucb_ridge_surrogate",
        } and self.utility_record_count:
            observed = np.asarray(self.utility_indices, dtype=np.int64)
            targets = np.asarray(self.utility_values, dtype=float)
            model = Ridge(alpha=self.surrogate_ridge_alpha, fit_intercept=True)
            model.fit(self.surrogate_design[observed], targets)
            predicted = np.clip(model.predict(self.surrogate_design), 0.0, 1.0)
            semantic_feedback = float(feedback_weight) * (
                predicted - float(targets.mean())
            )
            observations = np.zeros(n, dtype=float)
            for group, codes in self.factor_codes.items():
                observations += self.factor_utility_count[group][codes]
            if self.feedback_model == "factor_ucb_ridge_surrogate":
                factor_uncertainties = [
                    np.sqrt(
                        math.log(self.utility_record_count + 2.0)
                        / (self.factor_utility_count[group][codes] + 1.0)
                    )
                    for group, codes in self.factor_codes.items()
                    if len(self.factor_utility_count[group]) > 1
                ]
                uncertainty = (
                    np.max(np.vstack(factor_uncertainties), axis=0)
                    if factor_uncertainties
                    else np.zeros(n, dtype=float)
                )
                exploration_mode = "max_factor_ucb"
            else:
                uncertainty = np.sqrt(
                    math.log(self.utility_record_count + 2.0) / (observations + 1.0)
                )
                exploration_mode = "pooled_factor_count"
            exploration = float(exploration_weight) * uncertainty
            score = semantic_feedback + exploration
            nonzero = bool(np.any(np.abs(semantic_feedback) > 1e-12))
            self.ranking_reads_after_feedback += 1
            self.informative_records_before_ranking += int(informative)
            if nonzero:
                self.nonzero_feedback_reads += 1
            return score, {
                "overlay_read": True,
                "records_available": self.feedback_record_count,
                "informative_records": int(informative),
                "nonzero": nonzero,
                "feedback_model": self.feedback_model,
                "utility_records": int(self.utility_record_count),
                "exploration_mode": exploration_mode,
                "mean_semantic_adjustment": float(np.mean(semantic_feedback)),
                "max_abs_semantic_adjustment": float(np.max(np.abs(semantic_feedback))),
                "mean_exploration_bonus": float(np.mean(exploration)),
            }

        factor_adjustment = np.zeros(n, dtype=float)
        observations = np.zeros(n, dtype=float)
        for group, codes in self.factor_codes.items():
            adjustment, attempted = self._posterior_adjustment(
                codes,
                self.factor_counts[group],
                global_supported=global_supported,
                global_contradicted=global_contradicted,
                inconclusive_failure_weight=inconclusive_failure_weight,
            )
            factor_adjustment += adjustment
            observations += attempted
        if self.factor_codes:
            factor_adjustment /= len(self.factor_codes)

        pair_adjustment = np.zeros(n, dtype=float)
        for key, codes in self.pair_codes.items():
            adjustment, attempted = self._posterior_adjustment(
                codes,
                self.pair_counts[key],
                global_supported=global_supported,
                global_contradicted=global_contradicted,
                inconclusive_failure_weight=inconclusive_failure_weight,
            )
            pair_adjustment += adjustment
            observations += attempted
        if self.pair_codes:
            pair_adjustment /= len(self.pair_codes)

        semantic_feedback = (
            float(feedback_weight) * factor_adjustment
            + float(pair_feedback_weight) * pair_adjustment
        )
        exploration = float(exploration_weight) * np.sqrt(
            math.log(self.feedback_record_count + 2.0) / (observations + 1.0)
        )
        score = semantic_feedback + exploration
        nonzero = bool(np.any(np.abs(semantic_feedback) > 1e-12))
        self.ranking_reads_after_feedback += 1
        self.informative_records_before_ranking += int(informative)
        if nonzero:
            self.nonzero_feedback_reads += 1
        return score, {
            "overlay_read": True,
            "records_available": self.record_count,
            "informative_records": int(informative),
            "nonzero": nonzero,
            "mean_semantic_adjustment": float(np.mean(semantic_feedback)),
            "max_abs_semantic_adjustment": float(np.max(np.abs(semantic_feedback))),
            "mean_exploration_bonus": float(np.mean(exploration)),
            "inconclusive_failure_weight": float(inconclusive_failure_weight),
            "feedback_model": self.feedback_model,
        }

    def note_selection_change(self, changed: bool) -> None:
        if changed:
            self.selection_changed_batches += 1

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": OVERLAY_SCHEMA,
            "task": self.adapter.task,
            "case_study_id": self.adapter.case_study_id,
            "seed": self.seed,
            "trial": self.trial,
            "path": None,
            "sha256": None,
            "records": self.record_count,
            "records_by_status": dict(sorted(self.status_counts.items())),
            "ranking_feedback_records": int(self.feedback_record_count),
            "ranking_feedback_by_status": dict(
                sorted(self.ranking_status_counts.items())
            ),
            "utility_feedback_records": int(self.utility_record_count),
            "semantic_assertions": int(self.semantic_assertion_count),
            "final_chain_hash": self.previous_hash,
            "ranking_reads_after_feedback": self.ranking_reads_after_feedback,
            "nonzero_feedback_reads": self.nonzero_feedback_reads,
            "selection_changed_batches": self.selection_changed_batches,
            "informative_records_before_ranking": self.informative_records_before_ranking,
            "feedback_consumed_during_ranking": self.ranking_reads_after_feedback > 0,
            "semantic_projection_verified": (
                self.record_count > 0 and self.semantic_projection_verified
            ),
            "per_seed_isolation": True,
            "mutates_formal_kg": False,
            "audit_records_retained": self.audit_records,
            "feedback_projection": self.feedback_projection,
            "feedback_model": self.feedback_model,
            "surrogate_design": self.surrogate_design_audit,
            "feedback_projection_audit": self.feedback_projection_audit,
            "feedback_factor_groups": [list(group) for group in self.feedback_groups],
            "feedback_relation_pairs": [
                {"subject_fields": list(left), "object_fields": list(right)}
                for left, right in self.feedback_pairs
            ],
            "qualifier_fields_excluded": (
                list(self.adapter.qualifier_fields)
                if self.feedback_projection == "relation_endpoints"
                else []
            ),
        }

    def write(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.stream_handle is not None:
            if path != self.stream_path:
                raise ValueError("streamed overlay path changed before finalization")
            self.stream_handle.close()
            self.stream_handle = None
            storage_format = "jsonl.gz"
        else:
            with path.open("w", encoding="utf-8") as handle:
                for record in self.records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            storage_format = "jsonl"
        manifest = self.manifest()
        manifest["path"] = str(path)
        manifest["sha256"] = _sha256_file(path)
        manifest["storage_format"] = storage_format
        return manifest


def _select_diverse_batch(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    public: pd.DataFrame,
    *,
    factor_fields: Sequence[str],
    batch_size: int,
    diversity_penalty: float,
    factor_codes: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    available = np.asarray(candidate_indices, dtype=np.int64)
    if len(available) <= batch_size:
        return available[np.argsort(-scores[available], kind="stable")]
    pool_size = min(len(available), max(batch_size * 8, batch_size + 256))
    local_scores = scores[available]
    if pool_size < len(available):
        local = np.argpartition(local_scores, -pool_size)[-pool_size:]
        pool = available[local]
    else:
        pool = available
    pool = pool[np.lexsort((pool, -scores[pool]))]

    codes = factor_codes or {
        field: pd.factorize(public[field].fillna("").astype(str), sort=True)[0]
        for field in factor_fields
    }
    target_unique = max(2, int(math.ceil(math.sqrt(batch_size))))
    caps: dict[str, int] = {}
    for field in factor_fields:
        unique = max(1, len(np.unique(codes[field][pool])))
        caps[field] = max(
            1,
            int(
                math.ceil(
                    batch_size
                    / min(unique, target_unique)
                    * (1.0 + max(0.0, diversity_penalty) * 2.0)
                )
            ),
        )
    selected: list[int] = []
    deferred: list[int] = []
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for raw_index in pool:
        index = int(raw_index)
        if any(
            counts[(field, int(codes[field][index]))] >= caps[field]
            for field in factor_fields
        ):
            deferred.append(index)
            continue
        selected.append(index)
        for field in factor_fields:
            counts[(field, int(codes[field][index]))] += 1
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


def run_closed_loop_order(
    public: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    adapter: CaseStudyFeedbackAdapter,
    factor_fields: Sequence[str],
    rng: np.random.Generator,
    config: Any,
    seed: int,
    trial: int,
    overlay_path: Path | None = None,
    score_components: ScoreComponentConfig = ScoreComponentConfig(),
    audit_records: bool = True,
    collect_trace: bool = True,
    batch_commit_callback: (
        Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None
    ) = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Execute one outcome-blind-first, batch-feedback discovery order."""

    n = len(public)
    if n == 0:
        raise ValueError("closed-loop candidate registry cannot be empty")
    if set(outcomes["candidate_id"].astype(str)) != set(
        public["candidate_id"].astype(str)
    ):
        raise ValueError("closed-loop outcomes must cover the public registry exactly")
    outcome_lookup = outcomes.set_index("candidate_id", drop=False)
    static_score, component_manifest = compose_static_score(public, score_components)
    # Draw once per trial so each seed can explore a nearby outcome-blind ranking.
    # Policies may defer this perturbation until informative feedback exists,
    # preserving the stronger static KG order when experiments are inconclusive.
    trial_perturbation = rng.gumbel(0.0, float(config.sampling_temperature), size=n)
    stream_path = None
    if overlay_path is not None and n >= 100_000:
        stream_path = overlay_path.with_suffix(overlay_path.suffix + ".gz")
    overlay = ExperimentalOverlayGraph(
        public,
        adapter=adapter,
        factor_fields=factor_fields,
        seed=seed,
        trial=trial,
        prior_alpha=float(config.prior_alpha),
        prior_beta=float(config.prior_beta),
        feedback_projection=str(getattr(config, "feedback_projection", "all_factors")),
        feedback_model=str(getattr(config, "feedback_model", "beta_counts")),
        surrogate_ridge_alpha=float(getattr(config, "surrogate_ridge_alpha", 1.0)),
        stream_path=stream_path,
        audit_records=audit_records,
    )
    diversity_codes = {
        field: pd.factorize(public[field].fillna("").astype(str), sort=True)[0].astype(
            np.int64
        )
        for field in factor_fields
    }
    executed = np.zeros(n, dtype=bool)
    order: list[int] = []
    trace: list[dict[str, Any]] = []
    horizon = min(n, int(config.feedback_horizon or n))
    base_batch_size = int(config.batch_size)
    batch_schedule = str(getattr(config, "feedback_batch_schedule", "fixed"))
    if batch_schedule == "fixed":
        batch_size = max(
            base_batch_size,
            int(math.ceil(horizon / int(config.max_feedback_rounds))),
        )
        planned_rounds = int(math.ceil(horizon / batch_size))
    elif batch_schedule == "front_loaded":
        planned_rounds = min(
            int(config.max_feedback_rounds),
            int(math.ceil(horizon / base_batch_size)),
        )
        batch_size = base_batch_size
    else:
        raise ValueError(f"unknown feedback batch schedule: {batch_schedule!r}")
    round_index = 0
    adaptive_ranking_activated = False
    selection_commit_count = 0
    while len(order) < horizon:
        remaining = np.flatnonzero(~executed)
        warmup_active = round_index < int(config.warmup_batches)
        min_supported = int(getattr(config, "min_supported_before_feedback", 0))
        support_gate_active = (
            not warmup_active
            and overlay.ranking_status_counts[SUPPORTED] < min_supported
        )
        feedback_gate_open = not warmup_active and not support_gate_active
        if not feedback_gate_open:
            overlay_score = np.zeros(n, dtype=float)
            read_audit = {
                "overlay_read": False,
                "records_available": overlay.record_count,
                "informative_records": int(overlay.informative_feedback_count),
                "nonzero": False,
                "deferred_by_warmup": warmup_active,
                "deferred_by_feedback_gate": support_gate_active,
            }
        else:
            overlay_score, read_audit = overlay.score_candidates(
                feedback_weight=float(config.feedback_weight),
                pair_feedback_weight=float(config.pair_feedback_weight),
                exploration_weight=float(config.exploration_weight),
                inconclusive_failure_weight=float(
                    getattr(config, "inconclusive_search_failure_weight", 0.0)
                ),
            )
            read_audit["deferred_by_warmup"] = False
            read_audit["deferred_by_feedback_gate"] = False
        informative_records = overlay.informative_feedback_count
        informative_feedback_active = bool(
            feedback_gate_open and informative_records > 0
        )
        adaptive_ranking_activated = (
            adaptive_ranking_activated or informative_feedback_active
        )
        preserve_static = (
            bool(getattr(config, "preserve_static_until_informative_feedback", False))
            and not adaptive_ranking_activated
        )
        active_perturbation = 0.0 if preserve_static else trial_perturbation
        dynamic_score = static_score + active_perturbation + overlay_score
        warmup_diversity = getattr(config, "warmup_diversity_penalty", None)
        active_diversity_penalty = (
            float(warmup_diversity)
            if not feedback_gate_open and warmup_diversity is not None
            else float(config.diversity_penalty)
        )
        if batch_schedule == "front_loaded":
            progress = min(1.0, float(round_index + 1) / planned_rounds)
            target_rank = min(
                horizon,
                max(
                    (round_index + 1) * base_batch_size,
                    int(math.ceil(horizon * progress**4)),
                ),
            )
            scheduled_batch_size = max(1, target_rank - len(order))
        else:
            scheduled_batch_size = batch_size
        select_n = min(
            scheduled_batch_size,
            horizon - len(order),
            len(remaining),
        )
        if preserve_static:
            selected = remaining[np.lexsort((remaining, -static_score[remaining]))][
                :select_n
            ]
        else:
            selected = _select_diverse_batch(
                remaining,
                dynamic_score,
                public,
                factor_fields=factor_fields,
                batch_size=select_n,
                diversity_penalty=active_diversity_penalty,
                factor_codes=diversity_codes,
            )
        counterfactual = _select_diverse_batch(
            remaining,
            static_score + trial_perturbation,
            public,
            factor_fields=factor_fields,
            batch_size=select_n,
            diversity_penalty=active_diversity_penalty,
            factor_codes=diversity_codes,
        )
        changed = bool(
            read_audit["overlay_read"] and not np.array_equal(selected, counterfactual)
        )
        overlay.note_selection_change(changed)

        # A formal evaluator may persist a hash commitment here. This callback
        # deliberately runs before outcome_lookup is accessed for the selected
        # candidates, making the sequential reveal order externally auditable.
        selection_commit: dict[str, Any] = {}
        if batch_commit_callback is not None:
            commit_payload = {
                "schema_version": "closed-loop-batch-selection.v1",
                "seed": int(seed),
                "trial": int(trial),
                "batch": int(round_index),
                "start_rank": int(len(order) + 1),
                "end_rank": int(len(order) + len(selected)),
                "candidate_ids": public.iloc[selected]["candidate_id"]
                .astype(str)
                .tolist(),
                "overlay_records_available": int(read_audit["records_available"]),
                "overlay_informative_records": int(
                    read_audit["informative_records"]
                ),
                "selection_changed_by_overlay": changed,
                "static_order_preserved": preserve_static,
            }
            returned = batch_commit_callback(commit_payload)
            selection_commit = dict(returned or {})
            selection_commit_count += 1

        batch_status = Counter()
        batch_records: list[str] = []
        for index in selected:
            candidate_id = str(public.iloc[int(index)]["candidate_id"])
            outcome = outcome_lookup.loc[candidate_id].to_dict()
            record = overlay.append(
                candidate_index=int(index),
                outcome=outcome,
                round_index=round_index,
            )
            batch_status[str(record["status"])] += 1
            if collect_trace:
                batch_records.append(str(record["id"]))
        executed[selected] = True
        order.extend(map(int, selected))
        if collect_trace:
            trace.append(
                {
                    "batch": round_index,
                    "start_rank": len(order) - len(selected) + 1,
                    "end_rank": len(order),
                    "candidate_ids": public.iloc[selected]["candidate_id"]
                    .astype(str)
                    .tolist(),
                    "experimental_claim_ids": batch_records,
                    "feedback_status_counts": dict(sorted(batch_status.items())),
                    "validated_in_batch": int(batch_status[SUPPORTED]),
                    "cumulative_validated": int(overlay.status_counts[SUPPORTED]),
                    "overlay_read": bool(read_audit["overlay_read"]),
                    "overlay_records_available": int(read_audit["records_available"]),
                    "overlay_informative_records": int(
                        read_audit["informative_records"]
                    ),
                    "overlay_score_nonzero": bool(read_audit["nonzero"]),
                    "overlay_deferred_by_warmup": bool(
                        read_audit.get("deferred_by_warmup", False)
                    ),
                    "overlay_deferred_by_feedback_gate": bool(
                        read_audit.get("deferred_by_feedback_gate", False)
                    ),
                    "selection_changed_by_overlay": changed,
                    "static_order_preserved": preserve_static,
                    "informative_feedback_active": informative_feedback_active,
                    "selection_commit": selection_commit,
                }
            )
        round_index += 1

    if len(order) < n:
        remaining = np.flatnonzero(~executed)
        preserve_static_tail = (
            bool(getattr(config, "preserve_static_until_informative_feedback", False))
            and not adaptive_ranking_activated
        )
        tail_score = static_score + (
            0.0 if preserve_static_tail else trial_perturbation
        )
        tail = remaining[np.lexsort((remaining, -tail_score[remaining]))]
        order.extend(map(int, tail))
    compact = np.asarray(order, dtype=np.int64)
    if len(compact) != n or len(np.unique(compact)) != n:
        raise RuntimeError("closed-loop engine did not produce a full permutation")

    final_overlay_path = stream_path or overlay_path
    overlay_manifest = (
        overlay.write(final_overlay_path)
        if final_overlay_path is not None
        else overlay.manifest()
    )
    overlay_manifest["score_components"] = component_manifest
    overlay_manifest["ranking_randomization"] = {
        "sampler": "seeded_plackett_luce_gumbel",
        "sampling_temperature": float(config.sampling_temperature),
        "outcome_blind": True,
    }
    overlay_manifest["static_order_guard"] = {
        "enabled": bool(
            getattr(config, "preserve_static_until_informative_feedback", False)
        ),
        "adaptive_ranking_activated": bool(adaptive_ranking_activated),
        "activation_condition": (
            "feedback gate open and at least one supported or contradicted "
            "experimental result available"
        ),
    }
    overlay_manifest["rounds"] = round_index
    overlay_manifest["feedback_horizon"] = horizon
    overlay_manifest["feedback_batch_schedule"] = {
        "name": batch_schedule,
        "base_batch_size": base_batch_size,
        "planned_rounds": planned_rounds,
        "front_loaded_power": 4 if batch_schedule == "front_loaded" else None,
    }
    overlay_manifest["diversity_schedule"] = {
        "warmup": getattr(config, "warmup_diversity_penalty", None),
        "feedback": float(config.diversity_penalty),
        "parameter_semantics": "larger values relax per-factor quotas",
    }
    overlay_manifest["feedback_activation_gate"] = {
        "min_supported_before_feedback": int(
            getattr(config, "min_supported_before_feedback", 0)
        ),
        "semantics": (
            "retain the outcome-blind static ranking until this many supported "
            "executions have accumulated"
        ),
    }
    overlay_manifest["preserve_static_until_informative_feedback"] = bool(
        getattr(config, "preserve_static_until_informative_feedback", False)
    )
    overlay_manifest["batch_selection_commits"] = {
        "enabled": batch_commit_callback is not None,
        "count": int(selection_commit_count),
        "committed_before_selected_outcome_lookup": bool(
            batch_commit_callback is not None
            and selection_commit_count == round_index
        ),
    }
    return compact, trace, overlay_manifest


__all__ = [
    "OVERLAY_RECORD_SCHEMA",
    "OVERLAY_SCHEMA",
    "ExperimentalOverlayGraph",
    "ScoreComponentConfig",
    "compose_static_score",
    "run_closed_loop_order",
]


# Updated: 2026-08-16 02:01 HKT
