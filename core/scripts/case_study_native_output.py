"""Strict parsing and slot-level validation for native agent hypotheses."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from core.scripts.case_study_search_policy import PolicyAnchor


def extract_json(text: str) -> dict[str, Any]:
    tagged = re.findall(r"<solution>\s*(.*?)\s*</solution>", text, flags=re.S | re.I)
    decoder = json.JSONDecoder()
    for candidate in [*reversed(tagged), text.strip()]:
        candidate = candidate.strip()
        try:
            payload: Any = json.loads(candidate)
        except json.JSONDecodeError:
            decoded: list[tuple[int, int, Any]] = []
            for start, char in enumerate(candidate):
                if char not in "[{":
                    continue
                try:
                    value, end = decoder.raw_decode(candidate[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, (dict, list)):
                    decoded.append((start + end, end, value))
            if not decoded:
                continue
            payload = max(decoded, key=lambda item: (item[0], item[1]))[2]
        if isinstance(payload, list):
            return {"hypotheses": payload}
        if isinstance(payload, dict):
            return payload
    raise ValueError("native output contains no JSON object or array")


def validate_ranked_hypotheses(
    *,
    method: str,
    trial: int,
    payloads: list[tuple[int, int, dict[str, Any] | None, str | None]],
    candidate_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for start_rank, end_rank, payload, batch_error in payloads:
        items = payload.get("hypotheses") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []
        by_rank: dict[int, list[Any]] = {}
        for item in items:
            try:
                rank = int(item.get("rank")) if isinstance(item, dict) else -1
            except (TypeError, ValueError):
                rank = -1
            by_rank.setdefault(rank, []).append(item)

        for rank in range(start_rank, end_rank + 1):
            candidates = by_rank.get(rank, [])
            item = (
                candidates[0]
                if len(candidates) == 1 and isinstance(candidates[0], dict)
                else None
            )
            errors: list[str] = []
            if batch_error:
                errors.append(batch_error)
            if not candidates:
                errors.append("missing_rank")
            elif len(candidates) > 1:
                errors.append("duplicate_rank")
            elif item is None:
                errors.append("non_object_hypothesis")
            item = item or {}
            candidate_id = str(item.get("candidate_id") or "").strip()
            rationale = str(item.get("rationale") or "").strip()
            if candidate_id not in candidate_ids:
                errors.append("invalid_candidate_id")
            if not rationale:
                errors.append("missing_rationale")
            try:
                confidence = float(item.get("confidence"))
                if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                confidence = np.nan
                errors.append("invalid_confidence")
            if not errors and candidate_id in seen:
                errors.append("duplicate_hypothesis")
            if not errors:
                seen.add(candidate_id)
            rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "generated_rank": rank,
                    "candidate_id": candidate_id,
                    "rationale": rationale,
                    "confidence": confidence,
                    "valid": not errors,
                    "validation_status": (
                        "valid" if not errors else ";".join(dict.fromkeys(errors))
                    ),
                }
            )
    return pd.DataFrame(rows)


def anchors_from_validated(frame: pd.DataFrame) -> tuple[PolicyAnchor, ...]:
    valid = frame[frame["valid"].fillna(False).astype(bool)].sort_values(
        "generated_rank", kind="stable"
    )
    return tuple(
        PolicyAnchor(
            candidate_id=str(row.candidate_id),
            score=float(row.confidence),
            rationale=str(row.rationale),
        )
        for row in valid.itertuples(index=False)
    )


# Last Updated At: 2026-08-01 10:20 HKT
