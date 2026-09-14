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
INCONCLUSIVE = "inconclusive"
EXECUTION_FAILED = "execution_failed"
VALID_STATUSES = {SUPPORTED, CONTRADICTED, INCONCLUSIVE, EXECUTION_FAILED}


@dataclass
class FeedbackRecord:
    status: str
    hypothesis_id: str = ""
    source_id: str = ""
    target_id: str = ""
    candidate_tuple: dict[str, Any] = field(default_factory=dict)
    path_node_ids: tuple[str, ...] = ()
    weight: float = 1.0
    reason: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeedbackRecord":
        status = str(raw.get("status") or raw.get("outcome") or "").strip().lower()
        if status not in VALID_STATUSES:
            raise ValueError(f"unknown feedback status: {status!r}")
        candidate_tuple = raw.get("candidate_tuple") or raw.get("metadata", {}).get("candidate_tuple") or {}
        path_node_ids = raw.get("path_node_ids") or raw.get("metadata", {}).get("path_node_ids") or ()
        assertions = raw.get("assertions") or []
        first_assertion = assertions[0] if assertions and isinstance(assertions[0], dict) else {}
        last_assertion = assertions[-1] if assertions and isinstance(assertions[-1], dict) else {}
        return cls(
            status=status,
            hypothesis_id=str(
                raw.get("hypothesis_id") or raw.get("candidate_id") or raw.get("id") or ""
            ),
            source_id=str(raw.get("source_id") or first_assertion.get("subject_id") or ""),
            target_id=str(raw.get("target_id") or last_assertion.get("object_id") or ""),
            candidate_tuple=dict(candidate_tuple) if isinstance(candidate_tuple, dict) else {},
            path_node_ids=tuple(
                str(node_id) for node_id in path_node_ids if str(node_id or "")
            ),
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
    inconclusive_similarity: float = 0.0
    execution_failed_similarity: float = 0.0
    exact_supported: bool = False
    exact_contradicted: bool = False
    exact_inconclusive: bool = False
    exact_execution_failed: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "additive": self.additive,
            "matched_records": self.matched_records,
            "supported_similarity": self.supported_similarity,
            "contradicted_similarity": self.contradicted_similarity,
            "inconclusive_similarity": self.inconclusive_similarity,
            "execution_failed_similarity": self.execution_failed_similarity,
            "exact_supported": self.exact_supported,
            "exact_contradicted": self.exact_contradicted,
            "exact_inconclusive": self.exact_inconclusive,
            "exact_execution_failed": self.exact_execution_failed,
            "reasons": self.reasons[:5],
        }


class FeedbackState:
    """Experiment-result feedback used to adjust later hypothesis ranking."""

    def __init__(self, records: list[FeedbackRecord] | None = None):
        self.records = records or []
        self._node_priority_cache: dict[str, float] = {}
        self._node_priority_record_count = -1

    def node_priority(self, node_id: str) -> float:
        """Return a bounded anchor prior from informative endpoint feedback.

        Supported endpoints should be explored again in nearby mechanisms,
        while scientifically contradicted endpoints should be deprioritized.
        Inconclusive and execution-failed records remain neutral because they
        are not negative scientific evidence.
        """

        node_id = str(node_id or "")
        if not node_id:
            return 0.0
        if self._node_priority_record_count != len(self.records):
            weights: dict[str, list[float]] = {}
            for record in self.records:
                if record.status not in {SUPPORTED, CONTRADICTED}:
                    continue
                weight = min(1.0, max(0.0, float(record.weight)))
                sign = 1.0 if record.status == SUPPORTED else -1.0
                endpoint_ids = {record.source_id, record.target_id} - {""}
                mediator_ids = set(record.path_node_ids[1:-1]) - endpoint_ids - {""}
                for endpoint in endpoint_ids:
                    bucket = weights.setdefault(endpoint, [0.0, 0.0])
                    bucket[0 if sign > 0 else 1] += weight
                for mediator in mediator_ids:
                    bucket = weights.setdefault(mediator, [0.0, 0.0])
                    bucket[0 if sign > 0 else 1] += 0.5 * weight
            self._node_priority_cache = {
                endpoint: float(math.tanh(math.log1p(values[0]) - math.log1p(values[1])))
                for endpoint, values in weights.items()
            }
            self._node_priority_record_count = len(self.records)
        return self._node_priority_cache.get(node_id, 0.0)

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
            "path_node_ids": metadata.get("path_node_ids", ()),
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
            elif rec.status == INCONCLUSIVE:
                adjustment.inconclusive_similarity = max(adjustment.inconclusive_similarity, weighted)
                if self._is_exact(h_tuple, source_id, target_id, h_id, rec):
                    adjustment.exact_inconclusive = True
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

        # Inconclusive and failed executions are audit/feasibility outcomes, not
        # negative scientific evidence. Avoid rerunning the exact failed action,
        # but do not penalize scientifically similar hypotheses.
        if adjustment.exact_execution_failed:
            adjustment.multiplier *= 0.35
            adjustment.additive -= 0.05

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
        preferred_weights = {
            "disease_id": 0.24,
            "region_id": 0.24,
            "feature_id": 0.24,
            "feature_family": 0.12,
            "feature_modality": 0.08,
            "atlas_name": 0.08,
        }
        shared = sorted(
            key
            for key in set(a) & set(b)
            if a.get(key) not in (None, "") and b.get(key) not in (None, "")
        )
        if not shared:
            return 0.0
        key_weights = {
            key: preferred_weights.get(key, 0.12)
            for key in shared
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


# Updated: 2026-08-12 02:28 HKT
