"""Frozen, leak-resistant NeuroDiscovery policies for generic case studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


POLICY_SCHEMA = "case-study-neurodiscovery-policy.v8"
LEGACY_POLICY_SCHEMAS = frozenset(
    {
        "case-study-neurodiscovery-policy.v1",
        "case-study-neurodiscovery-policy.v2",
        "case-study-neurodiscovery-policy.v3",
        "case-study-neurodiscovery-policy.v4",
        "case-study-neurodiscovery-policy.v5",
        "case-study-neurodiscovery-policy.v6",
        "case-study-neurodiscovery-policy.v7",
    }
)
FEEDBACK_PROJECTIONS = frozenset(
    {"all_factors", "generalizable_factors", "relation_endpoints"}
)
FEEDBACK_BATCH_SCHEDULES = frozenset({"fixed", "front_loaded"})
FEEDBACK_MODELS = frozenset(
    {
        "beta_counts",
        "ridge_surrogate",
        "factor_ucb_ridge_surrogate",
        "hierarchical_utility",
    }
)
KG_SCORE_FAMILIES = {
    "legacy": {
        "global_node": "kg_global_node_support",
        "global_pair": "kg_global_pair_support",
        "scoped_node": "kg_scoped_node_support",
        "scoped_pair": "kg_scoped_pair_support",
    },
    "relation_aware": {
        "global_node": "kg_relation_global_node_support",
        "global_pair": "kg_relation_global_pair_support",
        "scoped_node": "kg_relation_scoped_node_support",
        "scoped_pair": "kg_relation_scoped_pair_support",
    },
}
KG_SCORE_COLUMNS = tuple(
    dict.fromkeys(
        column for family in KG_SCORE_FAMILIES.values() for column in family.values()
    )
)
DESIGN_PRIOR_COLUMN = "score_design_prior"


@dataclass(frozen=True)
class NeuroDiscoveryPolicy:
    """Outcome-blind static score and online feedback schedule."""

    task: str = "*"
    score_family: str = "legacy"
    global_node_weight: float = 0.20
    global_pair_weight: float = 0.20
    scoped_node_weight: float = 0.20
    scoped_pair_weight: float = 0.39
    design_prior_weight: float = 0.0
    # Resolve exact score ties without perturbing distinct evidence tiers.
    tie_break_weight: float = 1e-9
    tie_break_seed: int = 20260810
    batch_size: int = 64
    warmup_batches: int = 1
    min_supported_before_feedback: int = 0
    sampling_temperature: float = 0.01
    feedback_weight: float = 0.45
    pair_feedback_weight: float = 0.25
    exploration_weight: float = 0.10
    diversity_penalty: float = 0.08
    warmup_diversity_penalty: float | None = None
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

    def validate(self, *, task: str | None = None) -> None:
        if not self.task:
            raise ValueError("policy task cannot be empty")
        if self.score_family not in KG_SCORE_FAMILIES:
            raise ValueError(f"unknown KG score family: {self.score_family!r}")
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
        if task is not None and self.task not in {"*", str(task)}:
            raise ValueError(
                f"policy task {self.task!r} does not match benchmark task {task!r}"
            )
        score_weights = (
            self.global_node_weight,
            self.global_pair_weight,
            self.scoped_node_weight,
            self.scoped_pair_weight,
            self.design_prior_weight,
            self.tie_break_weight,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in score_weights):
            raise ValueError("static score weights must be finite and non-negative")
        if sum(score_weights) <= 0:
            raise ValueError("at least one static score weight must be positive")
        if self.batch_size < 1 or self.warmup_batches < 1:
            raise ValueError("batch_size and warmup_batches must be positive")
        if self.min_supported_before_feedback < 0:
            raise ValueError("min_supported_before_feedback must be non-negative")
        if self.max_feedback_rounds < 1:
            raise ValueError("max_feedback_rounds must be positive")
        if self.feedback_horizon is not None and self.feedback_horizon < 1:
            raise ValueError("feedback_horizon must be positive when provided")
        non_negative = (
            self.sampling_temperature,
            self.feedback_weight,
            self.pair_feedback_weight,
            self.exploration_weight,
            self.diversity_penalty,
        )
        if any(not math.isfinite(float(value)) or value < 0 for value in non_negative):
            raise ValueError("closed-loop weights must be finite and non-negative")
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

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": POLICY_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuroDiscoveryPolicy":
        schema = str(payload.get("schema_version", POLICY_SCHEMA))
        if schema not in {POLICY_SCHEMA, *LEGACY_POLICY_SCHEMAS}:
            raise ValueError(f"unsupported NeuroDiscovery policy schema: {schema!r}")
        values = {
            key: value for key, value in payload.items() if key != "schema_version"
        }
        if schema == "case-study-neurodiscovery-policy.v1":
            values.setdefault("feedback_projection", "all_factors")
        if schema in LEGACY_POLICY_SCHEMAS:
            values.setdefault("preserve_static_until_informative_feedback", False)
            values.setdefault("feedback_model", "beta_counts")
            values.setdefault("surrogate_ridge_alpha", 1.0)
            values.setdefault("min_supported_before_feedback", 0)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown NeuroDiscovery policy fields: {unknown}")
        policy = cls(**values)
        policy.validate()
        return policy

    def policy_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def closed_loop_kwargs(self, *, default_horizon: int) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "warmup_batches": self.warmup_batches,
            "min_supported_before_feedback": self.min_supported_before_feedback,
            "sampling_temperature": self.sampling_temperature,
            "feedback_weight": self.feedback_weight,
            "pair_feedback_weight": self.pair_feedback_weight,
            "exploration_weight": self.exploration_weight,
            "diversity_penalty": self.diversity_penalty,
            "warmup_diversity_penalty": self.warmup_diversity_penalty,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "max_feedback_rounds": self.max_feedback_rounds,
            "feedback_horizon": self.feedback_horizon or int(default_horizon),
            "inconclusive_search_failure_weight": (
                self.inconclusive_search_failure_weight
            ),
            "feedback_projection": self.feedback_projection,
            "feedback_batch_schedule": self.feedback_batch_schedule,
            "feedback_model": self.feedback_model,
            "surrogate_ridge_alpha": self.surrogate_ridge_alpha,
            "preserve_static_until_informative_feedback": (
                self.preserve_static_until_informative_feedback
            ),
        }


def load_neurodiscovery_policy(path: Path, *, task: str) -> NeuroDiscoveryPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("NeuroDiscovery policy must be a JSON object")
    policy = NeuroDiscoveryPolicy.from_dict(payload)
    policy.validate(task=task)
    return policy


def write_neurodiscovery_policy(path: Path, policy: NeuroDiscoveryPolicy) -> None:
    policy.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(policy.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _stable_tie_break(candidate_ids: pd.Series, seed: int) -> np.ndarray:
    denominator = float(2**64 - 1)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}|{candidate_id}".encode("utf-8")).digest()[:8],
                "big",
            )
            / denominator
            for candidate_id in candidate_ids.astype(str)
        ],
        dtype=float,
    )


def apply_neurodiscovery_policy(
    public: pd.DataFrame,
    policy: NeuroDiscoveryPolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompose the public KG score without reading experimental outcomes."""

    policy.validate()
    columns = KG_SCORE_FAMILIES[policy.score_family]
    missing = sorted(set(columns.values()) - set(public.columns))
    if missing:
        raise ValueError(f"public candidate table lacks KG score components: {missing}")
    if "candidate_id" not in public:
        raise ValueError("public candidate table lacks candidate_id")
    out = public.copy()
    weights = {
        "global_node": float(policy.global_node_weight),
        "global_pair": float(policy.global_pair_weight),
        "scoped_node": float(policy.scoped_node_weight),
        "scoped_pair": float(policy.scoped_pair_weight),
    }
    numerator = np.zeros(len(out), dtype=float)
    for name, column in columns.items():
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for every candidate")
        numerator += weights[name] * values
    design_prior_active = float(policy.design_prior_weight) > 0.0
    if design_prior_active:
        if DESIGN_PRIOR_COLUMN not in out:
            raise ValueError(
                f"public candidate table lacks {DESIGN_PRIOR_COLUMN} required by policy"
            )
        design_prior = pd.to_numeric(
            out[DESIGN_PRIOR_COLUMN], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(design_prior).all():
            raise ValueError(
                f"{DESIGN_PRIOR_COLUMN} must be finite for every candidate"
            )
        if np.any((design_prior < 0.0) | (design_prior > 1.0)):
            raise ValueError(f"{DESIGN_PRIOR_COLUMN} must be in [0, 1]")
        numerator += float(policy.design_prior_weight) * design_prior
    tie = _stable_tie_break(out["candidate_id"], policy.tie_break_seed)
    numerator += float(policy.tie_break_weight) * tie
    denominator = (
        sum(weights.values())
        + float(policy.design_prior_weight)
        + float(policy.tie_break_weight)
    )
    out["score_neurodiscovery"] = numerator / denominator
    audit = {
        "schema_version": POLICY_SCHEMA,
        "policy_id": policy.policy_id(),
        "task": policy.task,
        "score_family": policy.score_family,
        "uses_experimental_outcomes": False,
        "score_columns": columns,
        "score_weights": weights,
        "design_prior": {
            "column": DESIGN_PRIOR_COLUMN,
            "weight": float(policy.design_prior_weight),
            "active": design_prior_active,
            "frozen_before_experiment": design_prior_active,
            "uses_experimental_outcomes": False,
        },
        "tie_break_weight": float(policy.tie_break_weight),
        "tie_break_seed": int(policy.tie_break_seed),
        "score_min": float(out["score_neurodiscovery"].min()),
        "score_max": float(out["score_neurodiscovery"].max()),
    }
    return out, audit


__all__ = [
    "FEEDBACK_BATCH_SCHEDULES",
    "FEEDBACK_MODELS",
    "FEEDBACK_PROJECTIONS",
    "DESIGN_PRIOR_COLUMN",
    "KG_SCORE_COLUMNS",
    "KG_SCORE_FAMILIES",
    "LEGACY_POLICY_SCHEMAS",
    "POLICY_SCHEMA",
    "NeuroDiscoveryPolicy",
    "apply_neurodiscovery_policy",
    "load_neurodiscovery_policy",
    "write_neurodiscovery_policy",
]
