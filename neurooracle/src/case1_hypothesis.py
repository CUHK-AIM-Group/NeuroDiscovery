"""Directional wording helpers for atomic Case Study 1 hypotheses.

Case 1 candidates are predictions, not underspecified analysis questions.  Every
candidate therefore commits to an increase or decrease before validation.  When
the graph contains feature-specific directional evidence we preserve it;
otherwise a stable generator prior makes the proposal reproducible and
falsifiable without presenting that prior as observed evidence.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


_INCREASE_WORDS = (
    "increase",
    "increased",
    "increases",
    "increasing",
    "higher",
    "elevated",
    "positive",
    "enhanced",
)
_DECREASE_WORDS = (
    "decrease",
    "decreased",
    "decreases",
    "decreasing",
    "lower",
    "reduced",
    "negative",
    "diminished",
)
_DIRECTION_FIELDS = (
    "effect_direction",
    "association_direction",
    "direction",
    "change_direction",
    "trend",
    "sign",
)
_FEATURE_FIELDS = ("feature_id", "feature_name", "measure", "metric", "phenotype")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()


def _direction_from_text(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    positive = any(re.search(rf"\b{re.escape(word)}\b", text) for word in _INCREASE_WORDS)
    negative = any(re.search(rf"\b{re.escape(word)}\b", text) for word in _DECREASE_WORDS)
    if positive == negative:
        return ""
    return "increase" if positive else "decrease"


def _feature_specific(evidence: dict[str, Any], raw_text: str, feature: dict[str, Any]) -> bool:
    feature_values = {
        _norm(feature.get("id")),
        _norm(feature.get("name")),
    }
    feature_values.discard("")
    described = {_norm(evidence.get(key)) for key in _FEATURE_FIELDS}
    described.discard("")
    if feature_values & described:
        return True
    # A free-text claim can be used only when it names the proposed feature.  The
    # synthetic candidate_feature_id added to KG links is deliberately excluded.
    text = _norm(raw_text)
    return bool(text and any(value in text for value in feature_values if len(value) >= 4))


def propose_case1_direction(
    disease: dict[str, Any],
    region: dict[str, Any],
    feature: dict[str, Any],
    links: Iterable[Any] = (),
) -> dict[str, Any]:
    """Return an explicit, reproducible direction and its provenance."""
    votes: list[str] = []
    for link in links:
        if isinstance(link, dict):
            evidence = link.get("evidence") if isinstance(link.get("evidence"), dict) else {}
            raw_text = str(link.get("raw_text") or "")
            relation = str(link.get("relation_type") or "")
        else:
            evidence = getattr(link, "evidence", {}) or {}
            raw_text = str(getattr(link, "raw_text", "") or "")
            relation = str(getattr(link, "relation_type", "") or "")
        if not _feature_specific(evidence, raw_text, feature):
            continue
        for key in _DIRECTION_FIELDS:
            direction = _direction_from_text(evidence.get(key))
            if direction:
                votes.append(direction)
        direction = _direction_from_text(f"{relation} {raw_text}")
        if direction:
            votes.append(direction)
        for key in ("cohens_d", "effect_size", "beta", "correlation", "rho", "r"):
            try:
                value = float(evidence.get(key))
            except (TypeError, ValueError):
                continue
            if value:
                votes.append("increase" if value > 0 else "decrease")

    if votes and votes.count("increase") != votes.count("decrease"):
        direction = "increase" if votes.count("increase") > votes.count("decrease") else "decrease"
        return {
            "direction": direction,
            "source": "feature_specific_graph_evidence",
            "directional_evidence_votes": len(votes),
        }

    basis = "|".join(
        str(value or "").strip().casefold()
        for value in (
            disease.get("id") or disease.get("name"),
            region.get("id") or region.get("name"),
            feature.get("id") or feature.get("name"),
        )
    )
    direction = "increase" if hashlib.sha256(basis.encode("utf-8")).digest()[0] % 2 == 0 else "decrease"
    return {
        "direction": direction,
        "source": "generator_directional_prior",
        "directional_evidence_votes": 0,
    }


def case1_directional_title(disease_name: str, region_name: str, feature_name: str, direction: str) -> str:
    adjective = "increased" if direction == "increase" else "decreased"
    return f"{disease_name} is associated with {adjective} {feature_name} in {region_name}"


def case1_directional_statement(
    disease_name: str,
    region_name: str,
    feature_name: str,
    direction: str,
) -> str:
    adjective = "increased" if direction == "increase" else "decreased"
    return (
        f"Patients with {disease_name} are hypothesized to show {adjective} {feature_name} "
        f"in {region_name} relative to healthy controls."
    )
