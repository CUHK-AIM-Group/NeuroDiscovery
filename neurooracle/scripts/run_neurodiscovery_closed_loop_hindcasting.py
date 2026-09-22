"""Tune and evaluate NeuroDiscovery with a hidden retrospective experiment oracle.

The public candidate table contains only information available at the freeze
year. Outcomes are held separately and revealed only after a candidate is
selected. Two closed-loop views are reported:

* temporal_transfer: years 1-2 provide feedback; years 3-5 are untouched test.
* retrospective_oracle: the full five-year outcome is returned after execution.

No wet-lab or neuroimaging experiment is claimed. The second view estimates the
search-policy contribution under a reproducible retrospective oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop import ClosedLoopConfig
from core.scripts.case_study_closed_loop_engine import (
    ScoreComponentConfig,
    run_closed_loop_order,
)
from core.scripts.case_study_feedback_adapters import (
    EXECUTION_FAILED,
    INCONCLUSIVE,
    SUPPORTED,
    CaseStudyFeedbackAdapter,
    RelationTemplate,
)
from neurooracle.scripts.run_case_study_hindcasting import (
    _hypothesis_payload_semantic_key,
)
from neurooracle.src.hindcasting_static_policy import (
    RELATION_COMPONENT_RAW_FIELDS,
    score_relation_component_rows,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "neurodiscovery-closed-loop-hindcasting.v1"
DEFAULT_BUDGETS = (10, 20, 50, 100, 200, 500, 1000)
DEFAULT_DEV_FREEZE_YEARS = (2016, 2017, 2018)
DEFAULT_TEST_FREEZE_YEARS = (2019, 2020)
FACTOR_FIELDS = (
    "source_id",
    "target_id",
    "relation_signature",
    "mediator_id",
    "path_length_bucket",
)


@dataclass(frozen=True)
class LoopProfile:
    name: str
    score_family: str
    confidence_weight: float
    evidence_weight: float
    novelty_weight: float
    testability_weight: float
    static_only: bool = False
    batch_size: int = 10
    feedback_start_budget: int = 50
    feedback_start_fraction: float = 0.25
    sampling_temperature: float = 1e-6
    feedback_weight: float = 0.45
    pair_feedback_weight: float = 0.15
    exploration_weight: float = 0.04
    diversity_penalty: float = 0.04
    warmup_diversity_penalty: float = 10.0
    min_supported_before_feedback: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 3.0
    inconclusive_search_failure_weight: float = 0.03
    relation_regularization_weight: float = 0.10

    def validate(self) -> None:
        if self.score_family not in {
            "legacy",
            "evidence",
            "generator_composite",
            "relation_aware",
            "relation_regularized",
        }:
            raise ValueError(f"unknown score family for {self.name}: {self.score_family}")
        weights = (
            self.confidence_weight,
            self.evidence_weight,
            self.novelty_weight,
            self.testability_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError(f"invalid static weights for {self.name}")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError(f"static weights must sum to one for {self.name}")
        if not 0.0 <= self.relation_regularization_weight <= 1.0:
            raise ValueError("relation_regularization_weight must be in [0, 1]")
        if self.feedback_start_budget < 1:
            raise ValueError("feedback_start_budget must be positive")
        if not 0.0 < self.feedback_start_fraction <= 1.0:
            raise ValueError("feedback_start_fraction must be in (0, 1]")
        config = self.closed_loop_config(1, 1)
        config.validate()

    def closed_loop_config(
        self,
        horizon: int,
        valid_candidate_count: int | None = None,
    ) -> ClosedLoopConfig:
        valid_count = max(1, int(valid_candidate_count or horizon))
        adaptive_start = max(
            self.batch_size,
            int(math.ceil(valid_count * self.feedback_start_fraction)),
        )
        feedback_start = min(self.feedback_start_budget, adaptive_start)
        warmup_batches = max(1, int(math.ceil(feedback_start / self.batch_size)))
        return ClosedLoopConfig(
            batch_size=self.batch_size,
            warmup_batches=warmup_batches,
            sampling_temperature=0.0 if self.static_only else self.sampling_temperature,
            feedback_weight=0.0 if self.static_only else self.feedback_weight,
            pair_feedback_weight=0.0 if self.static_only else self.pair_feedback_weight,
            exploration_weight=0.0 if self.static_only else self.exploration_weight,
            diversity_penalty=0.0 if self.static_only else self.diversity_penalty,
            warmup_diversity_penalty=(
                None if self.static_only else self.warmup_diversity_penalty
            ),
            min_supported_before_feedback=(
                0 if self.static_only else self.min_supported_before_feedback
            ),
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            max_feedback_rounds=max(128, int(math.ceil(horizon / self.batch_size))),
            feedback_horizon=horizon,
            inconclusive_search_failure_weight=(
                0.0 if self.static_only else self.inconclusive_search_failure_weight
            ),
            feedback_projection="all_factors",
            feedback_batch_schedule="fixed",
        )


def profile_slate() -> tuple[LoopProfile, ...]:
    """A compact delayed-feedback slate rather than a large overfit grid."""

    return (
        LoopProfile(
            "legacy_static",
            "legacy",
            0.20,
            0.20,
            0.25,
            0.35,
            static_only=True,
        ),
        LoopProfile(
            "evidence_static",
            "evidence",
            0.30,
            0.35,
            0.10,
            0.25,
            static_only=True,
        ),
        LoopProfile(
            "generator_composite_closed_supported_gate",
            "generator_composite",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.25,
            pair_feedback_weight=0.12,
            exploration_weight=0.0,
            diversity_penalty=0.06,
            min_supported_before_feedback=1,
            inconclusive_search_failure_weight=0.0,
        ),
        LoopProfile(
            "relation_scoped_pair_static",
            "relation_aware",
            0.20,
            0.20,
            0.25,
            0.35,
            static_only=True,
        ),
        LoopProfile(
            "legacy_relation_regularized_static",
            "relation_regularized",
            0.20,
            0.20,
            0.25,
            0.35,
            static_only=True,
            relation_regularization_weight=0.10,
        ),
        LoopProfile(
            "legacy_closed_delayed_weak",
            "legacy",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.20,
            pair_feedback_weight=0.08,
            exploration_weight=0.01,
            diversity_penalty=0.06,
            inconclusive_search_failure_weight=0.01,
        ),
        LoopProfile(
            "legacy_closed_delayed_balanced",
            "legacy",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.25,
            pair_feedback_weight=0.12,
            exploration_weight=0.02,
            diversity_penalty=0.06,
            inconclusive_search_failure_weight=0.01,
        ),
        LoopProfile(
            "legacy_closed_delayed_positive_only",
            "legacy",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.20,
            pair_feedback_weight=0.08,
            exploration_weight=0.0,
            diversity_penalty=10.0,
            inconclusive_search_failure_weight=0.0,
        ),
        LoopProfile(
            "legacy_closed_supported_gate",
            "legacy",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.25,
            pair_feedback_weight=0.12,
            exploration_weight=0.0,
            diversity_penalty=0.06,
            min_supported_before_feedback=2,
            inconclusive_search_failure_weight=0.0,
        ),
        LoopProfile(
            "relation_scoped_pair_closed_supported_gate",
            "relation_aware",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.25,
            pair_feedback_weight=0.12,
            exploration_weight=0.0,
            diversity_penalty=0.06,
            min_supported_before_feedback=2,
            inconclusive_search_failure_weight=0.0,
        ),
        LoopProfile(
            "legacy_relation_regularized_closed_supported_gate",
            "relation_regularized",
            0.20,
            0.20,
            0.25,
            0.35,
            feedback_weight=0.25,
            pair_feedback_weight=0.12,
            feedback_start_fraction=0.25,
            exploration_weight=0.0,
            diversity_penalty=0.06,
            warmup_diversity_penalty=10.0,
            min_supported_before_feedback=2,
            inconclusive_search_failure_weight=0.0,
            relation_regularization_weight=0.10,
        ),
        LoopProfile(
            "evidence_closed_delayed_weak",
            "evidence",
            0.30,
            0.35,
            0.10,
            0.25,
            feedback_weight=0.20,
            pair_feedback_weight=0.08,
            exploration_weight=0.01,
            diversity_penalty=0.06,
            inconclusive_search_failure_weight=0.01,
        ),
    )


@dataclass
class RunBundle:
    task: str
    seed: int
    freeze_year: int
    future_start_year: int
    future_end_year: int
    feedback_end_year: int
    public: pd.DataFrame
    hidden: pd.DataFrame
    audit: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_year(value: object) -> int | None:
    number = _finite_float(value, default=float("nan"))
    return int(number) if math.isfinite(number) else None


def _bool_value(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _relation_signature(hypothesis: Mapping[str, Any]) -> str:
    relations = [
        str(link.get("relation_type") or "unknown")
        for link in hypothesis.get("path") or []
    ]
    return ">".join(relations) or "generation_failure"


def _mediator_id(hypothesis: Mapping[str, Any]) -> str:
    source_id = str(hypothesis.get("source_id") or "")
    target_id = str(hypothesis.get("target_id") or "")
    nodes: list[str] = []
    for link in hypothesis.get("path") or []:
        for key in ("from_id", "to_id"):
            node_id = str(link.get(key) or "")
            if node_id and node_id not in {source_id, target_id} and node_id not in nodes:
                nodes.append(node_id)
    return "|".join(nodes[:2]) or "none"


def _static_score(public: pd.DataFrame, profile: LoopProfile) -> np.ndarray:
    if profile.score_family == "generator_composite":
        if "generator_composite_score" in public:
            raw = public["generator_composite_score"]
        elif "composite_score" in public:
            raw = public["composite_score"]
        else:
            raw = public["confidence_score"]
        scores = pd.to_numeric(raw, errors="coerce").fillna(0.0).to_numpy(float)
        scores = np.clip(scores, 0.0, 1.0)
        invalid = public["is_generation_failure"].to_numpy(bool) | public[
            "is_semantic_duplicate"
        ].to_numpy(bool)
        scores[invalid] = 0.0
        return scores
    if profile.score_family in {"relation_aware", "relation_regularized"}:
        invalid = public["is_generation_failure"].to_numpy(bool) | public[
            "is_semantic_duplicate"
        ].to_numpy(bool)
        relation_scores = np.zeros(len(public), dtype=float)
        if (~invalid).any():
            valid_scores, _normalized = score_relation_component_rows(
                public.loc[~invalid].to_dict("records")
            )
            relation_scores[~invalid] = valid_scores
        if profile.score_family == "relation_aware":
            return relation_scores

        legacy_scores = _legacy_static_score(public, profile)
        valid = ~invalid
        scores = np.zeros(len(public), dtype=float)
        if valid.any():
            legacy_normalized = _minmax_static(legacy_scores[valid])
            relation_normalized = _minmax_static(relation_scores[valid])
            relation_weight = float(profile.relation_regularization_weight)
            scores[valid] = (
                (1.0 - relation_weight) * legacy_normalized
                + relation_weight * relation_normalized
            )
        return scores
    return _legacy_static_score(public, profile)


def _minmax_static(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high):
        return np.zeros(len(values), dtype=float)
    return (values - low) / (high - low)


def _legacy_static_score(public: pd.DataFrame, profile: LoopProfile) -> np.ndarray:
    weights = {
        "confidence_score": profile.confidence_weight,
        "evidence_score": profile.evidence_weight,
        "novelty_score": profile.novelty_weight,
        "testability_score": profile.testability_weight,
    }
    log_score = np.zeros(len(public), dtype=float)
    for column, weight in weights.items():
        values = pd.to_numeric(public[column], errors="coerce").fillna(0.0).to_numpy(float)
        log_score += float(weight) * np.log(np.clip(values, 0.01, 1.0))
    score = np.exp(log_score)
    invalid = public["is_generation_failure"].to_numpy(bool) | public[
        "is_semantic_duplicate"
    ].to_numpy(bool)
    score[invalid] = 0.0
    return score


def prepare_run_bundle(
    run: Mapping[str, Any],
    *,
    hypotheses_path: Path,
    scored_path: Path,
    feedback_years: int,
) -> RunBundle:
    payload = json.loads(hypotheses_path.read_text(encoding="utf-8"))
    hypotheses = list(payload.get("hypotheses") or [])
    scored = pd.read_csv(scored_path, low_memory=False)
    scored_by_id = scored.drop_duplicates("id", keep="first").set_index("id")
    seen: set[tuple[str, tuple[str, ...]]] = set()
    public_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    duplicates = 0
    generation_failures = 0
    feedback_end_year = min(
        int(run["future_end_year"]), int(run["freeze_year"]) + int(feedback_years)
    )

    for hypothesis in hypotheses:
        candidate_id = str(hypothesis.get("id") or "")
        if candidate_id not in scored_by_id.index:
            raise ValueError(f"{candidate_id!r} is missing from {scored_path}")
        result = scored_by_id.loc[candidate_id]
        metadata = dict(hypothesis.get("metadata") or {})
        generation_failure = bool(
            hypothesis.get("hypothesis_type") == "generation_failure"
            or metadata.get("generation_failure") is True
        )
        semantic_duplicate = False
        if not generation_failure:
            semantic_key = _hypothesis_payload_semantic_key(hypothesis)
            semantic_duplicate = semantic_key in seen
            seen.add(semantic_key)
        generation_failures += int(generation_failure)
        duplicates += int(semantic_duplicate)
        path_length = len(hypothesis.get("path") or [])
        public_rows.append(
            {
                "candidate_id": candidate_id,
                "source_id": str(hypothesis.get("source_id") or "none"),
                "target_id": str(hypothesis.get("target_id") or "none"),
                "relation_signature": _relation_signature(hypothesis),
                "mediator_id": _mediator_id(hypothesis),
                "path_length_bucket": str(min(path_length, 4)),
                "source_domain": str(metadata.get("domain_a") or "unknown"),
                "target_domain": str(metadata.get("domain_b") or "unknown"),
                "confidence_score": _finite_float(hypothesis.get("confidence_score")),
                "evidence_score": _finite_float(hypothesis.get("evidence_score")),
                "novelty_score": _finite_float(hypothesis.get("novelty_score")),
                "testability_score": _finite_float(hypothesis.get("testability_score")),
                **{
                    field: _finite_float(metadata.get(field), default=float("nan"))
                    for field in RELATION_COMPONENT_RAW_FIELDS
                },
                "is_generation_failure": generation_failure,
                "is_semantic_duplicate": semantic_duplicate,
            }
        )
        primary_year = _optional_year(result.get("primary_year"))
        first_future_year = _optional_year(result.get("first_future_year"))
        eligible = not generation_failure and not semantic_duplicate
        full_primary = eligible and _bool_value(result.get("primary_hit", False))
        full_any = eligible and _bool_value(result.get("any_future_hit", False))
        hidden_rows.append(
            {
                "candidate_id": candidate_id,
                "execution_succeeded": not generation_failure and not semantic_duplicate,
                "primary_year": primary_year,
                "first_future_year": first_future_year,
                "full_primary_hit": full_primary,
                "full_any_hit": full_any,
                "early_primary_hit": bool(
                    full_primary and primary_year is not None and primary_year <= feedback_end_year
                ),
                "early_any_hit": bool(
                    full_any
                    and first_future_year is not None
                    and first_future_year <= feedback_end_year
                ),
                "terminal_primary_hit": bool(
                    full_primary and primary_year is not None and primary_year > feedback_end_year
                ),
                "terminal_any_hit": bool(
                    full_any
                    and first_future_year is not None
                    and first_future_year > feedback_end_year
                ),
            }
        )

    public = pd.DataFrame(public_rows)
    hidden = pd.DataFrame(hidden_rows)
    if public["candidate_id"].duplicated().any():
        raise ValueError(f"candidate IDs are not unique in {hypotheses_path}")
    forbidden = {
        "primary_year",
        "first_future_year",
        "full_primary_hit",
        "full_any_hit",
        "early_primary_hit",
        "early_any_hit",
        "terminal_primary_hit",
        "terminal_any_hit",
        "feedback_status",
        "validated",
    }
    if forbidden & set(public.columns):
        raise RuntimeError("hidden outcomes leaked into the public registry")
    return RunBundle(
        task=str(run["case_study_id"]),
        seed=int(run["seed"]),
        freeze_year=int(run["freeze_year"]),
        future_start_year=int(run["future_start_year"]),
        future_end_year=int(run["future_end_year"]),
        feedback_end_year=feedback_end_year,
        public=public,
        hidden=hidden,
        audit={
            "hypotheses_path": str(hypotheses_path),
            "scored_path": str(scored_path),
            "candidate_count": len(public),
            "valid_unique_candidates": int(
                (~public["is_generation_failure"] & ~public["is_semantic_duplicate"]).sum()
            ),
            "generation_failures": generation_failures,
            "semantic_duplicates_neutralized": duplicates,
        },
    )


def feedback_outcomes(bundle: RunBundle, mode: str) -> pd.DataFrame:
    if mode not in {"temporal_transfer", "retrospective_oracle"}:
        raise ValueError(f"unknown feedback mode: {mode}")
    support_column = (
        "early_primary_hit" if mode == "temporal_transfer" else "full_primary_hit"
    )
    rows: list[dict[str, Any]] = []
    for row in bundle.hidden.to_dict("records"):
        if not bool(row["execution_succeeded"]):
            status = EXECUTION_FAILED
        elif bool(row[support_column]):
            status = SUPPORTED
        else:
            status = INCONCLUSIVE
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "validated": status == SUPPORTED,
                "feedback_status": status,
                "execution_succeeded": bool(row["execution_succeeded"]),
                "primary_year": row["primary_year"],
                "first_future_year": row["first_future_year"],
                "feedback_window_end": (
                    bundle.feedback_end_year
                    if mode == "temporal_transfer"
                    else bundle.future_end_year
                ),
                "oracle_kind": mode,
            }
        )
    return pd.DataFrame(rows)


def generic_feedback_adapter(task: str) -> CaseStudyFeedbackAdapter:
    return CaseStudyFeedbackAdapter(
        task=task,
        case_study_id=task,
        factor_fields=FACTOR_FIELDS,
        relations=(
            RelationTemplate(
                ("source_id",),
                "HYPOTHESIS_ENDPOINT",
                "experimentally_associated_with",
                ("target_id",),
                "HYPOTHESIS_ENDPOINT",
            ),
        ),
        qualifier_fields=("relation_signature", "mediator_id", "path_length_bucket"),
    )


def rank_bundle(
    bundle: RunBundle,
    profile: LoopProfile,
    *,
    feedback_mode: str,
    collect_trace: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    profile.validate()
    public = bundle.public.copy()
    public["score_neurodiscovery"] = _static_score(public, profile)
    static_order = np.lexsort(
        (np.arange(len(public)), -public["score_neurodiscovery"].to_numpy(float))
    )
    if profile.static_only:
        return static_order, [], {
            "mode": "static",
            "profile": profile.name,
            "outcome_blind": True,
            "feedback_records": 0,
        }
    valid_candidate_count = int(
        (~public["is_generation_failure"] & ~public["is_semantic_duplicate"]).sum()
    )
    config = profile.closed_loop_config(len(public), valid_candidate_count)
    config.validate()
    rng_seed = bundle.seed + 1009 * bundle.freeze_year
    order, trace, manifest = run_closed_loop_order(
        public,
        feedback_outcomes(bundle, feedback_mode),
        adapter=generic_feedback_adapter(bundle.task),
        factor_fields=FACTOR_FIELDS,
        rng=np.random.default_rng(rng_seed),
        config=config,
        seed=rng_seed,
        trial=bundle.seed,
        score_components=ScoreComponentConfig(
            kge_weight=0.0,
            novelty_weight=0.0,
            critic_weight=0.0,
        ),
        audit_records=False,
        collect_trace=collect_trace,
    )
    manifest.update(
        {
            "mode": feedback_mode,
            "profile": profile.name,
            "outcome_blind": True,
            "feedback_end_year": (
                bundle.feedback_end_year
                if feedback_mode == "temporal_transfer"
                else bundle.future_end_year
            ),
            "valid_candidate_count": valid_candidate_count,
            "feedback_start_rank": config.batch_size * config.warmup_batches,
        }
    )
    return order, trace, manifest


def metric_rows(
    bundle: RunBundle,
    order: np.ndarray,
    *,
    method: str,
    profile: str,
    feedback_mode: str,
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    hidden = bundle.hidden.reset_index(drop=True)
    labels = {
        name: hidden[name].to_numpy(bool)
        for name in (
            "full_primary_hit",
            "full_any_hit",
            "terminal_primary_hit",
            "terminal_any_hit",
        )
    }
    rows: list[dict[str, Any]] = []
    for requested in budgets:
        k = min(int(requested), len(order))
        selected = order[:k]
        row = {
            "method": method,
            "profile": profile,
            "feedback_mode": feedback_mode,
            "case_study_id": bundle.task,
            "seed": bundle.seed,
            "freeze_year": bundle.freeze_year,
            "future_start_year": bundle.future_start_year,
            "future_end_year": bundle.future_end_year,
            "feedback_end_year": bundle.feedback_end_year,
            "requested_k": int(requested),
            "executed_hypotheses": k,
        }
        for label, values in labels.items():
            row[label.replace("_hit", "_hits")] = int(values[selected].sum())
            row[label.replace("_hit", "_pool")] = int(values.sum())
        rows.append(row)
    return rows


def recovery_objective(rows: Sequence[Mapping[str, Any]]) -> float:
    ordered = sorted(rows, key=lambda row: int(row["requested_k"]))
    budget_weights = np.asarray(
        [1.0 / math.sqrt(max(1, int(row["requested_k"]))) for row in ordered],
        dtype=float,
    )
    scores: list[tuple[float, float]] = []
    for hit_column, pool_column, label_weight in (
        ("terminal_primary_hits", "terminal_primary_pool", 0.70),
        ("terminal_any_hits", "terminal_any_pool", 0.30),
    ):
        pool = int(ordered[0][pool_column]) if ordered else 0
        if pool <= 0:
            continue
        recall = np.asarray([int(row[hit_column]) / pool for row in ordered], dtype=float)
        scores.append((label_weight, float(np.average(recall, weights=budget_weights))))
    if not scores:
        return float("nan")
    denominator = sum(weight for weight, _ in scores)
    return float(sum(weight * score for weight, score in scores) / denominator)


def _load_runs(
    generation_root: Path,
    evaluation_root: Path,
    *,
    tasks: set[str] | None,
    seeds: set[int] | None,
    freeze_years: set[int] | None,
) -> list[dict[str, Any]]:
    generation = json.loads(
        (generation_root / "generation_manifest.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (evaluation_root / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    scored_paths: dict[tuple[str, int, int], Path] = {}
    for row in evaluation["runs"]:
        if str(row.get("method")) != "neurodiscovery":
            continue
        key = (
            str(row["case_study_id"]),
            int(row["seed"]),
            int(row["freeze_year"]),
        )
        scored_paths[key] = _resolve(row["metrics_path"]).parent / "scored_hypotheses.csv"

    rows: list[dict[str, Any]] = []
    for raw in generation["runs"]:
        task = str(raw["case_study_id"])
        seed = int(raw["seed"])
        freeze_year = int(raw["freeze_year"])
        if tasks is not None and task not in tasks:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if freeze_years is not None and freeze_year not in freeze_years:
            continue
        key = (task, seed, freeze_year)
        if key not in scored_paths:
            raise KeyError(f"no evaluated candidate table for {key}")
        row = dict(raw)
        row["hypotheses_path_resolved"] = _resolve(raw["hypotheses_path"])
        row["scored_path_resolved"] = scored_paths[key]
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            int(row["freeze_year"]),
            int(row["seed"]),
            str(row["case_study_id"]),
        ),
    )


def _aggregate_development(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_summary = (
        trials.dropna(subset=["objective"])
        .groupby(["profile", "static_only", "case_study_id"], as_index=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_sd=("objective", "std"),
            runs=("objective", "size"),
        )
        .fillna(0.0)
    )
    rows: list[dict[str, Any]] = []
    for (profile, static_only), group in task_summary.groupby(
        ["profile", "static_only"], sort=False
    ):
        values = group["objective_mean"].to_numpy(float)
        rows.append(
            {
                "profile": profile,
                "static_only": bool(static_only),
                "eligible_tasks": len(values),
                "task_mean": float(values.mean()) if len(values) else float("nan"),
                "task_min": float(values.min()) if len(values) else float("nan"),
                "robust_score": (
                    float(0.80 * values.mean() + 0.20 * values.min())
                    if len(values)
                    else float("nan")
                ),
            }
        )
    profile_summary = pd.DataFrame(rows).sort_values(
        ["robust_score", "task_mean"], ascending=False
    )
    return task_summary, profile_summary


def _profile_by_name(name: str) -> LoopProfile:
    profiles = {profile.name: profile for profile in profile_slate()}
    return profiles[name]


def run(args: argparse.Namespace) -> dict[str, Any]:
    profiles = profile_slate()
    if getattr(args, "profiles", None):
        profile_by_name = {profile.name: profile for profile in profiles}
        profiles = tuple(profile_by_name[name] for name in args.profiles)
    for profile in profiles:
        profile.validate()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = set(args.case_study_ids) if args.case_study_ids else None
    dev_runs = _load_runs(
        args.generation_root,
        args.evaluation_root,
        tasks=tasks,
        seeds=set(args.tuning_seeds),
        freeze_years=set(args.dev_freeze_years),
    )
    if args.smoke:
        preferred = next(
            (
                row
                for row in dev_runs
                if str(row["case_study_id"]) == "biomarker_discovery"
            ),
            dev_runs[0],
        )
        dev_runs = [preferred]
        profiles = profiles[:3]

    development_rows: list[dict[str, Any]] = []
    bundle_audits: list[dict[str, Any]] = []
    for run_index, source_run in enumerate(dev_runs, 1):
        bundle = prepare_run_bundle(
            source_run,
            hypotheses_path=source_run["hypotheses_path_resolved"],
            scored_path=source_run["scored_path_resolved"],
            feedback_years=args.feedback_years,
        )
        bundle_audits.append(
            {
                "split": "development",
                "case_study_id": bundle.task,
                "seed": bundle.seed,
                "freeze_year": bundle.freeze_year,
                **bundle.audit,
            }
        )
        for profile in profiles:
            order, _trace, _manifest = rank_bundle(
                bundle,
                profile,
                feedback_mode="temporal_transfer",
            )
            metrics = metric_rows(
                bundle,
                order,
                method="nd_static" if profile.static_only else "nd_closed_loop",
                profile=profile.name,
                feedback_mode="temporal_transfer",
                budgets=args.budgets,
            )
            development_rows.append(
                {
                    "profile": profile.name,
                    "static_only": profile.static_only,
                    "case_study_id": bundle.task,
                    "seed": bundle.seed,
                    "freeze_year": bundle.freeze_year,
                    "objective": recovery_objective(metrics),
                }
            )
        print(
            f"[tune {run_index}/{len(dev_runs)}] {bundle.task} "
            f"seed={bundle.seed} KG_{bundle.freeze_year}",
            flush=True,
        )

    development = pd.DataFrame(development_rows)
    task_summary, profile_summary = _aggregate_development(development)
    closed = profile_summary.loc[~profile_summary["static_only"]]
    static = profile_summary.loc[profile_summary["static_only"]]
    if closed.empty or static.empty:
        raise RuntimeError("profile slate must contain static and closed-loop candidates")
    selected_profile = _profile_by_name(str(closed.iloc[0]["profile"]))
    selected_static = _profile_by_name(str(static.iloc[0]["profile"]))

    development.to_csv(args.output_root / "development_trials.csv", index=False)
    task_summary.to_csv(args.output_root / "development_task_summary.csv", index=False)
    profile_summary.to_csv(args.output_root / "development_profile_summary.csv", index=False)
    pd.DataFrame(bundle_audits).to_csv(
        args.output_root / "candidate_pool_audit.csv", index=False
    )
    (args.output_root / "selected_profiles.json").write_text(
        json.dumps(
            {
                "selection_protocol": {
                    "development_freeze_years": list(args.dev_freeze_years),
                    "tuning_seeds": list(args.tuning_seeds),
                    "objective": (
                        "early-budget weighted terminal recall; 70% strict primary support "
                        "and 30% secondary any-edge support when both are available"
                    ),
                },
                "best_static": asdict(selected_static),
                "best_closed_loop": asdict(selected_profile),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.smoke:
        manifest = {
            "schema_version": SCHEMA,
            "status": "smoke_complete",
            "created_at": utc_now(),
            "development_runs": len(dev_runs),
            "selected_closed_loop": selected_profile.name,
        }
        (args.output_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    if args.development_only:
        manifest = {
            "schema_version": SCHEMA,
            "status": "development_complete",
            "created_at": utc_now(),
            "generation_root": str(args.generation_root),
            "evaluation_root": str(args.evaluation_root),
            "output_root": str(args.output_root),
            "dev_freeze_years": list(args.dev_freeze_years),
            "tuning_seeds": list(args.tuning_seeds),
            "budgets": list(args.budgets),
            "selected_static": selected_static.name,
            "selected_closed_loop": selected_profile.name,
            "development_runs": len(dev_runs),
            "test_data_accessed": False,
        }
        (args.output_root / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    test_runs = _load_runs(
        args.generation_root,
        args.evaluation_root,
        tasks=tasks,
        seeds=set(args.test_seeds),
        freeze_years=set(args.test_freeze_years),
    )
    test_metrics: list[dict[str, Any]] = []
    loop_audits: list[dict[str, Any]] = []
    for run_index, source_run in enumerate(test_runs, 1):
        bundle = prepare_run_bundle(
            source_run,
            hypotheses_path=source_run["hypotheses_path_resolved"],
            scored_path=source_run["scored_path_resolved"],
            feedback_years=args.feedback_years,
        )
        bundle_audits.append(
            {
                "split": "test",
                "case_study_id": bundle.task,
                "seed": bundle.seed,
                "freeze_year": bundle.freeze_year,
                **bundle.audit,
            }
        )
        static_order, _, static_manifest = rank_bundle(
            bundle,
            selected_static,
            feedback_mode="temporal_transfer",
        )
        test_metrics.extend(
            metric_rows(
                bundle,
                static_order,
                method="nd_open_loop",
                profile=selected_static.name,
                feedback_mode="none",
                budgets=args.budgets,
            )
        )
        for mode, method in (
            ("temporal_transfer", "nd_closed_loop_transfer"),
            ("retrospective_oracle", "nd_closed_loop_oracle"),
        ):
            order, _trace, loop_manifest = rank_bundle(
                bundle,
                selected_profile,
                feedback_mode=mode,
            )
            test_metrics.extend(
                metric_rows(
                    bundle,
                    order,
                    method=method,
                    profile=selected_profile.name,
                    feedback_mode=mode,
                    budgets=args.budgets,
                )
            )
            loop_audits.append(
                {
                    "case_study_id": bundle.task,
                    "seed": bundle.seed,
                    "freeze_year": bundle.freeze_year,
                    "method": method,
                    "overlay_manifest": loop_manifest,
                }
            )
        loop_audits.append(
            {
                "case_study_id": bundle.task,
                "seed": bundle.seed,
                "freeze_year": bundle.freeze_year,
                "method": "nd_open_loop",
                "overlay_manifest": static_manifest,
            }
        )
        print(
            f"[test {run_index}/{len(test_runs)}] {bundle.task} "
            f"seed={bundle.seed} KG_{bundle.freeze_year}",
            flush=True,
        )

    metrics = pd.DataFrame(test_metrics)
    metrics.to_csv(args.output_root / "test_metrics_by_run.csv", index=False)
    bundle_audit = pd.DataFrame(bundle_audits)
    bundle_audit.to_csv(args.output_root / "candidate_pool_audit.csv", index=False)
    summary_columns = [
        "full_primary_hits",
        "full_any_hits",
        "terminal_primary_hits",
        "terminal_any_hits",
    ]
    aggregate = (
        metrics.groupby(
            ["method", "profile", "feedback_mode", "requested_k"], as_index=False
        )[summary_columns]
        .agg(["mean", "std", "sum"])
    )
    aggregate.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(args.output_root / "test_summary.csv", index=False)
    task_summary_test = (
        metrics.groupby(
            ["method", "case_study_id", "requested_k"], as_index=False
        )[summary_columns]
        .mean()
    )
    task_summary_test.to_csv(args.output_root / "test_task_summary.csv", index=False)
    with (args.output_root / "loop_audit.jsonl").open("w", encoding="utf-8") as target:
        for row in loop_audits:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": SCHEMA,
        "status": "complete",
        "created_at": utc_now(),
        "generation_root": str(args.generation_root),
        "evaluation_root": str(args.evaluation_root),
        "output_root": str(args.output_root),
        "dev_freeze_years": list(args.dev_freeze_years),
        "test_freeze_years": list(args.test_freeze_years),
        "feedback_years": args.feedback_years,
        "tuning_seeds": list(args.tuning_seeds),
        "test_seeds": list(args.test_seeds),
        "budgets": list(args.budgets),
        "selected_static": selected_static.name,
        "selected_closed_loop": selected_profile.name,
        "development_runs": len(dev_runs),
        "test_runs": len(test_runs),
        "experimental_interpretation": {
            "actual_experiments_run": False,
            "oracle": "retrospective future-literature support revealed after selection",
            "formal_kg_mutated": False,
            "experimental_overlay_used": True,
            "nonpublication_treated_as_contradiction": False,
        },
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=ROOT
        / "neurooracle/data/experiments/hindcasting/neurodiscovery_20260811_all_seed0_9",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=ROOT
        / "neurooracle/data/experiments/hindcasting/neurodiscovery_20260811_all_seed0_9_eval",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "neurooracle/data/experiments/hindcasting/neurodiscovery_closed_loop_20260811",
    )
    parser.add_argument("--case-study-ids", nargs="*", default=None)
    parser.add_argument("--dev-freeze-years", nargs="+", type=int, default=list(DEFAULT_DEV_FREEZE_YEARS))
    parser.add_argument("--test-freeze-years", nargs="+", type=int, default=list(DEFAULT_TEST_FREEZE_YEARS))
    parser.add_argument("--tuning-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--feedback-years", type=int, default=2)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=[profile.name for profile in profile_slate()],
        default=None,
        help="Optional preregistered subset of static and closed-loop profiles.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Select profiles on development years without loading held-out test runs.",
    )
    args = parser.parse_args(argv)
    if args.feedback_years < 1:
        parser.error("--feedback-years must be positive")
    if any(budget < 1 for budget in args.budgets):
        parser.error("--budgets must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


# Updated: 2026-08-11 20:30 HKT
# Updated: 2026-08-13 03:42 HKT - preserve the generator's outcome-blind composite ranking in dynamic closed-loop execution.
