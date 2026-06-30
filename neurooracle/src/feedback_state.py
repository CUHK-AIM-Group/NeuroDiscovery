"""Closed-loop feedback state for hypothesis ranking.

The state is intentionally lightweight: it can be written by experiment/audit
code as JSON or JSONL records, then loaded by HypothesisEngine before ranking.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED = "supported"
CONTRADICTED = "contradicted"
EXECUTION_FAILED = "execution_failed"
VALID_STATUSES = {SUPPORTED, CONTRADICTED, EXECUTION_FAILED}


@dataclass
class FeedbackRecord:
    status: str
    hypothesis_id: str = ""
    source_id: str = ""
    target_id: str = ""
    candidate_tuple: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    reason: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeedbackRecord":
        status = str(raw.get("status") or raw.get("outcome") or "").strip().lower()
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown feedback status: {status!r}")
        candidate_tuple = raw.get("candidate_tuple") or raw.get("metadata", {}).get("candidate_tuple") or {}
        return cls(
            status=status,
            hypothesis_id=str(raw.get("hypothesis_id") or raw.get("id") or ""),
            source_id=str(raw.get("source_id") or ""),
            target_id=str(raw.get("target_id") or ""),
            candidate_tuple=dict(candidate_tuple) if isinstance(candidate_tuple, dict) else {},
            weight=max(0.0, float(raw.get("weight", 1.0) or 0.0)),
            reason=str(raw.get("reason") or raw.get("note") or ""),
        )


@dataclass
class FeedbackAdjustment:
    multiplier: float = 1.0
    additive: float = 0.0
    matched_records: int = 0
    supported_similarity: float = 0.0
    contradicted_similarity: float = 0.0
    execution_failed_similarity: float = 0.0
    exact_supported: bool = False
    exact_contradicted: bool = False
    exact_execution_failed: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "additive": self.additive,
            "matched_records": self.matched_records,
            "supported_similarity": self.supported_similarity,
            "contradicted_similarity": self.contradicted_similarity,
            "execution_failed_similarity": self.execution_failed_similarity,
            "exact_supported": self.exact_supported,
            "exact_contradicted": self.exact_contradicted,
            "exact_execution_failed": self.exact_execution_failed,
            "reasons": self.reasons[:5],
        }


class FeedbackState:
    """Experiment-result feedback used to adjust later hypothesis ranking."""

    def __init__(self, records: list[FeedbackRecord] | None = None):
        self.records = records or []

    @classmethod
    def load(cls, path: str | Path) -> "FeedbackState":
        path = Path(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return cls([])
        if path.suffix.lower() == ".jsonl":
            raw_records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            raw = json.loads(text)
            if isinstance(raw, dict):
                raw_records = raw.get("records") or raw.get("feedback") or raw.get("results") or []
            elif isinstance(raw, list):
                raw_records = raw
            else:
                raise ValueError(f"unsupported feedback state shape in {path}")
        return cls([FeedbackRecord.from_dict(r) for r in raw_records])

    @staticmethod
    def record_from_hypothesis(h: Any, status: str, weight: float = 1.0, reason: str = "") -> dict[str, Any]:
        metadata = getattr(h, "metadata", {}) or {}
        return {
            "status": status,
            "hypothesis_id": getattr(h, "id", ""),
            "source_id": getattr(h, "source_id", ""),
            "target_id": getattr(h, "target_id", ""),
            "candidate_tuple": metadata.get("candidate_tuple", {}),
            "weight": weight,
            "reason": reason,
        }

    def score(self, h: Any) -> FeedbackAdjustment:
        if not self.records:
            return FeedbackAdjustment()
        h_tuple = (getattr(h, "metadata", {}) or {}).get("candidate_tuple", {}) or {}
        h_id = str(getattr(h, "id", "") or "")
        source_id = str(getattr(h, "source_id", "") or "")
        target_id = str(getattr(h, "target_id", "") or "")
        adjustment = FeedbackAdjustment()
        for rec in self.records:
            sim = self._similarity(h_tuple, source_id, target_id, h_id, rec)
            if sim <= 0:
                continue
            weighted = min(1.0, sim * rec.weight)
            adjustment.matched_records += 1
            if rec.reason and len(adjustment.reasons) < 5:
                adjustment.reasons.append(rec.reason)
            if rec.status == SUPPORTED:
                adjustment.supported_similarity = max(adjustment.supported_similarity, weighted)
                if self._is_exact(h_tuple, source_id, target_id, h_id, rec):
                    adjustment.exact_supported = True
            elif rec.status == CONTRADICTED:
                adjustment.contradicted_similarity = max(adjustment.contradicted_similarity, weighted)
                if self._is_exact(h_tuple, source_id, target_id, h_id, rec):
                    adjustment.exact_contradicted = True
            elif rec.status == EXECUTION_FAILED:
                adjustment.execution_failed_similarity = max(adjustment.execution_failed_similarity, weighted)
                if self._is_exact(h_tuple, source_id, target_id, h_id, rec):
                    adjustment.exact_execution_failed = True

        if adjustment.exact_supported:
            adjustment.multiplier *= 0.35
            adjustment.additive += 0.02
        elif adjustment.supported_similarity > 0:
            adjustment.multiplier *= 1.0 + 0.18 * adjustment.supported_similarity

        if adjustment.exact_contradicted:
            adjustment.multiplier *= 0.20
            adjustment.additive -= 0.08
        elif adjustment.contradicted_similarity > 0:
            adjustment.multiplier *= max(0.55, 1.0 - 0.30 * adjustment.contradicted_similarity)
            adjustment.additive -= 0.03 * adjustment.contradicted_similarity

        if adjustment.exact_execution_failed:
            adjustment.multiplier *= 0.35
            adjustment.additive -= 0.05
        elif adjustment.execution_failed_similarity > 0:
            adjustment.multiplier *= max(0.65, 1.0 - 0.22 * adjustment.execution_failed_similarity)
            adjustment.additive -= 0.02 * adjustment.execution_failed_similarity

        adjustment.multiplier = float(min(1.35, max(0.05, adjustment.multiplier)))
        adjustment.additive = float(min(0.08, max(-0.15, adjustment.additive)))
        return adjustment

    @staticmethod
    def apply(base_score: float, adjustment: FeedbackAdjustment) -> float:
        score = base_score * adjustment.multiplier + adjustment.additive
        if not math.isfinite(score):
            return 0.0
        return float(min(1.0, max(0.0, score)))

    @classmethod
    def _is_exact(
        cls,
        h_tuple: dict[str, Any],
        source_id: str,
        target_id: str,
        h_id: str,
        rec: FeedbackRecord,
    ) -> bool:
        if h_id and rec.hypothesis_id and h_id == rec.hypothesis_id:
            return True
        if rec.source_id and rec.target_id and source_id == rec.source_id and target_id == rec.target_id:
            return True
        return bool(h_tuple and rec.candidate_tuple and cls._tuple_similarity(h_tuple, rec.candidate_tuple) >= 0.999)

    @classmethod
    def _similarity(
        cls,
        h_tuple: dict[str, Any],
        source_id: str,
        target_id: str,
        h_id: str,
        rec: FeedbackRecord,
    ) -> float:
        if cls._is_exact(h_tuple, source_id, target_id, h_id, rec):
            return 1.0
        scores = []
        if rec.source_id and source_id == rec.source_id:
            scores.append(0.40)
        if rec.target_id and target_id == rec.target_id:
            scores.append(0.45)
        if h_tuple and rec.candidate_tuple:
            scores.append(cls._tuple_similarity(h_tuple, rec.candidate_tuple))
        return max(scores) if scores else 0.0

    @staticmethod
    def _tuple_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
        key_weights = {
            "disease_id": 0.24,
            "region_id": 0.24,
            "feature_id": 0.24,
            "feature_family": 0.12,
            "feature_modality": 0.08,
            "atlas_name": 0.08,
        }
        total = 0.0
        matched = 0.0
        for key, weight in key_weights.items():
            av = a.get(key)
            bv = b.get(key)
            if av in (None, "") or bv in (None, ""):
                continue
            total += weight
            if str(av) == str(bv):
                matched += weight
        if total == 0:
            return 0.0
        return matched / total
