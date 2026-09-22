"""Frozen configuration and feature vocabulary for Case Study 1 search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


FEATURE_TERMS: dict[str, tuple[str, ...]] = {
    "roi_falff_proxy": (
        "fALFF",
        "fractional amplitude of low frequency fluctuation",
        "fractional amplitude of low-frequency fluctuations",
    ),
    "roi_alff_proxy": (
        "ALFF",
        "amplitude of low frequency fluctuation",
        "amplitude of low-frequency fluctuations",
    ),
    "roi_temporal_mean": ("mean BOLD signal", "regional mean signal"),
    "roi_temporal_mean_abs": ("mean absolute BOLD signal", "regional mean signal"),
    "roi_temporal_std": (
        "BOLD signal variability",
        "temporal variability",
        "signal variability",
    ),
    "roi_temporal_variance": (
        "BOLD signal variability",
        "temporal variability",
        "signal variability",
    ),
    "corr_mean": (
        "functional connectivity",
        "resting state functional connectivity",
        "resting-state functional connectivity",
    ),
    "corr_mean_abs": (
        "functional connectivity",
        "resting state functional connectivity",
        "resting-state functional connectivity",
    ),
    "corr_positive_mean": (
        "positive functional connectivity",
        "functional connectivity",
    ),
    "corr_negative_mean": (
        "negative functional connectivity",
        "anticorrelation",
        "functional connectivity",
    ),
    "corr_node_degree_abs_top10": (
        "degree centrality",
        "node degree",
        "functional connectivity strength",
        "functional connectivity",
    ),
    "partial_mean": ("partial correlation", "functional connectivity"),
    "partial_mean_abs": ("partial correlation", "functional connectivity"),
    "partial_positive_mean": (
        "partial correlation",
        "positive functional connectivity",
        "functional connectivity",
    ),
    "partial_negative_mean": (
        "partial correlation",
        "negative functional connectivity",
        "anticorrelation",
        "functional connectivity",
    ),
    "normalized_volume_fraction": (
        "gray matter volume",
        "regional brain volume",
        "brain volume",
        "cortical volume",
        "normalized volume",
    ),
}


def feature_terms(feature: str) -> tuple[str, ...]:
    """Return registered KG query terms for one executable imaging feature."""

    raw = str(feature or "").strip()
    return tuple(dict.fromkeys((raw, *FEATURE_TERMS.get(raw, ()))))


@dataclass(frozen=True)
class Case1NeuroDiscoveryConfig:
    """Small, auditable parameter surface for NeuroDiscovery.

    Defaults reproduce the pre-tuning Case Study 1 behavior. A tuned JSON can
    be frozen beside experiment outputs and passed to the formal runner.
    """

    global_support_weight: float = 0.80
    scoped_support_weight: float = 0.20
    feature_support_weight: float = 0.0
    kge_weight: float = 0.0
    novelty_weight: float = 0.0
    critic_weight: float = 0.0
    feature_support_decay_budget: int = 0
    batch_size: int = 250
    warmup_budget: int = 10_000
    max_closed_loop_budget: int = 120_000
    warmup_exploit_fraction: float = 0.40
    feedback_weight: float = 0.10
    pair_feedback_weight: float = 0.08
    exploration_weight: float = 0.015
    inconclusive_search_failure_weight: float = 0.0
    pre_pair_exploit_fraction: float = 0.90
    post_pair_exploit_fraction: float = 0.94
    pair_feedback_start_fraction: float = 0.25
    pair_feedback_force_fraction: float = 0.42
    min_hits_for_pair_feedback: int = 50

    def validate(self) -> None:
        if self.global_support_weight < 0 or self.scoped_support_weight < 0:
            raise ValueError("static support weights must be non-negative")
        if self.global_support_weight + self.scoped_support_weight <= 0:
            raise ValueError("at least one static support weight must be positive")
        for name in (
            "feature_support_weight",
            "warmup_exploit_fraction",
            "pre_pair_exploit_fraction",
            "post_pair_exploit_fraction",
            "pair_feedback_start_fraction",
            "pair_feedback_force_fraction",
            "inconclusive_search_failure_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "feedback_weight",
            "pair_feedback_weight",
            "exploration_weight",
            "kge_weight",
            "novelty_weight",
            "critic_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.warmup_budget < 0:
            raise ValueError("warmup_budget must be non-negative")
        if self.max_closed_loop_budget < 1:
            raise ValueError("max_closed_loop_budget must be positive")
        if self.warmup_budget > self.max_closed_loop_budget:
            raise ValueError("warmup_budget cannot exceed max_closed_loop_budget")
        if self.feature_support_decay_budget < 0:
            raise ValueError("feature_support_decay_budget must be non-negative")
        if self.pair_feedback_force_fraction < self.pair_feedback_start_fraction:
            raise ValueError(
                "pair_feedback_force_fraction cannot precede pair_feedback_start_fraction"
            )
        if self.min_hits_for_pair_feedback < 0:
            raise ValueError("min_hits_for_pair_feedback must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Case1NeuroDiscoveryConfig":
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("unknown NeuroDiscovery parameters: " + ", ".join(unknown))
        config = cls(**raw)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: Path) -> "Case1NeuroDiscoveryConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("NeuroDiscovery config JSON must contain an object")
        return cls.from_dict(raw)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


__all__ = ["Case1NeuroDiscoveryConfig", "FEATURE_TERMS", "feature_terms"]
