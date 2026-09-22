"""Shared, outcome-blind search-policy contract for Case Study 1 generators.

Generator adapters receive a public candidate registry that contains only
executable coordinates. They return exact candidate anchors and optional
factor rules. This module compiles those sparse proposals into a deterministic
ranking over the full candidate space without consulting experimental results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SCHEMA_VERSION = "case1-search-policy-v1"
COMPILER_VERSION = "case1-policy-compiler-v3-anatomy-lateral-max"

PUBLIC_COLUMNS = (
    "candidate_id",
    "disease",
    "feature",
    "feature_family",
    "modality",
    "source",
    "roi_index",
    "roi_name",
    "anatomy_full",
    "anatomy_id",
    "hemisphere",
    "map_group",
    "network",
    "structure_class",
)

RULE_FIELDS = frozenset(
    {
        "candidate_id",
        "disease",
        "feature",
        "feature_family",
        "modality",
        "source",
        "roi_index",
        "roi_name",
        "anatomy_full",
        "anatomy_id",
        "hemisphere",
        "map_group",
        "network",
        "structure_class",
    }
)

QUOTA_FIELDS = frozenset(
    {
        "disease",
        "feature",
        "feature_family",
        "modality",
        "source",
        "hemisphere",
        "map_group",
        "network",
        "structure_class",
    }
)


@dataclass(frozen=True)
class PolicyAnchor:
    """One exact executable candidate proposed by a generator."""

    candidate_id: str
    score: float = 1.0
    rationale: str = ""
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyRule:
    """A factor or interaction preference over public candidate fields."""

    weight: float
    when: Mapping[str, str]
    rationale: str = ""


@dataclass(frozen=True)
class SearchPolicy:
    """Portable output produced by every adapted CS1 generator."""

    method: str
    trial: int
    anchors: tuple[PolicyAnchor, ...] = ()
    rules: tuple[PolicyRule, ...] = ()
    quotas: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _feature_family(feature: Any) -> str:
    text = _text(feature).casefold()
    if "falff" in text or "alff" in text:
        return "amplitude"
    if "temporal" in text:
        return "temporal"
    if "partial" in text:
        return "partial_fc"
    if "corr" in text or "connect" in text:
        return "correlation_fc"
    if "volume" in text or "thickness" in text or "area" in text:
        return "structure"
    return "other"


def _text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def build_public_registry(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the outcome-blind registry exposed to baseline generators."""

    required = {"candidate_id", "disease", "feature", "modality", "source", "roi_index"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate registry is missing required columns: {missing}")

    registry = pd.DataFrame(index=candidates.index)
    for column in PUBLIC_COLUMNS:
        if column == "feature_family":
            if column in candidates:
                registry[column] = _text_series(candidates[column])
            else:
                features = _text_series(candidates["feature"])
                family_lookup = {
                    feature: _feature_family(feature) for feature in features.unique()
                }
                registry[column] = features.map(family_lookup)
        elif column == "anatomy_id":
            registry[column] = (
                _text_series(candidates["modality"])
                + "|"
                + _text_series(candidates["source"])
                + "|"
                + _text_series(candidates["roi_index"])
            )
        elif column in candidates:
            registry[column] = _text_series(candidates[column])
        else:
            registry[column] = ""

    if registry["candidate_id"].eq("").any():
        raise ValueError("candidate_id cannot be empty")
    if registry["candidate_id"].duplicated().any():
        duplicates = (
            registry.loc[registry["candidate_id"].duplicated(), "candidate_id"]
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"candidate_id must be unique; duplicate examples: {duplicates}"
        )
    return registry.reset_index(drop=True)


def public_registry_records(candidates: pd.DataFrame) -> list[dict[str, str]]:
    return build_public_registry(candidates).to_dict(orient="records")


def policy_to_payload(policy: SearchPolicy) -> dict[str, Any]:
    return {
        "schema_version": policy.schema_version,
        "method": policy.method,
        "trial": policy.trial,
        "anchors": [
            {
                "candidate_id": anchor.candidate_id,
                "score": anchor.score,
                "rationale": anchor.rationale,
                "evidence_ids": list(anchor.evidence_ids),
            }
            for anchor in policy.anchors
        ],
        "rules": [
            {
                "weight": rule.weight,
                "when": dict(rule.when),
                "rationale": rule.rationale,
            }
            for rule in policy.rules
        ],
        "quotas": dict(policy.quotas),
        "metadata": dict(policy.metadata),
    }


def policy_from_payload(payload: Mapping[str, Any]) -> SearchPolicy:
    schema_version = _text(payload.get("schema_version") or SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported search-policy schema: {schema_version}")

    anchors: list[PolicyAnchor] = []
    for raw in payload.get("anchors") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("every policy anchor must be an object")
        candidate_id = _text(raw.get("candidate_id"))
        if not candidate_id:
            raise ValueError("policy anchor candidate_id cannot be empty")
        anchors.append(
            PolicyAnchor(
                candidate_id=candidate_id,
                score=float(raw.get("score", 1.0)),
                rationale=_text(raw.get("rationale")),
                evidence_ids=tuple(
                    _text(value)
                    for value in (raw.get("evidence_ids") or [])
                    if _text(value)
                ),
            )
        )

    rules: list[PolicyRule] = []
    for raw in payload.get("rules") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("every policy rule must be an object")
        when = raw.get("when") or {}
        if not isinstance(when, Mapping) or not when:
            raise ValueError("policy rule 'when' must be a non-empty object")
        unknown_fields = sorted(set(when) - RULE_FIELDS)
        if unknown_fields:
            raise ValueError(f"policy rule uses non-public fields: {unknown_fields}")
        rules.append(
            PolicyRule(
                weight=float(raw.get("weight", 0.0)),
                when={str(key): _text(value) for key, value in when.items()},
                rationale=_text(raw.get("rationale")),
            )
        )

    method = _text(payload.get("method"))
    if not method:
        raise ValueError("search policy requires a method")
    raw_quotas = payload.get("quotas") or {}
    if not isinstance(raw_quotas, Mapping):
        raise ValueError("search policy quotas must be an object")
    return SearchPolicy(
        method=method,
        trial=int(payload.get("trial", 0)),
        anchors=tuple(anchors),
        rules=tuple(rules),
        quotas=dict(raw_quotas),
        metadata=dict(payload.get("metadata") or {}),
        schema_version=schema_version,
    )


def _validate_public_policy(
    policy: SearchPolicy,
    public: pd.DataFrame,
) -> None:
    if policy.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported search-policy schema: {policy.schema_version}")
    if not _text(policy.method):
        raise ValueError("search policy requires a method")
    if policy.trial < 0:
        raise ValueError("search policy trial must be non-negative")

    known_ids = set(public["candidate_id"])
    anchor_ids = [anchor.candidate_id for anchor in policy.anchors]
    duplicates = sorted(
        candidate_id
        for candidate_id in set(anchor_ids)
        if anchor_ids.count(candidate_id) > 1
    )
    if duplicates:
        raise ValueError(
            f"policy anchor candidate_id values must be unique: {duplicates[:5]}"
        )
    unknown_ids = sorted(set(anchor_ids) - known_ids)
    if unknown_ids:
        raise ValueError(
            f"policy references unknown candidate_id values: {unknown_ids[:5]}"
        )
    for anchor in policy.anchors:
        if not math.isfinite(anchor.score) or not 0.0 <= anchor.score <= 1.0:
            raise ValueError(
                f"policy anchor score must be finite and in [0, 1]: {anchor.candidate_id}"
            )
    for rule in policy.rules:
        if not math.isfinite(rule.weight) or not -1.0 <= rule.weight <= 1.0:
            raise ValueError("policy rule weight must be finite and in [-1, 1]")
        unknown_fields = sorted(set(rule.when) - RULE_FIELDS)
        if unknown_fields:
            raise ValueError(f"policy rule uses non-public fields: {unknown_fields}")
        for field_name, value in rule.when.items():
            if value and value not in set(public[field_name]):
                raise ValueError(f"policy rule has unknown {field_name}={value!r}")
    _validated_quota_config(policy, public)


def validate_policy(policy: SearchPolicy, registry: pd.DataFrame) -> None:
    _validate_public_policy(policy, build_public_registry(registry))


def _validated_quota_config(
    policy: SearchPolicy,
    registry: pd.DataFrame,
) -> tuple[int, dict[str, int]] | None:
    if not policy.quotas:
        return None

    unknown_keys = sorted(set(policy.quotas) - {"prefix_size", "max_per_value"})
    if unknown_keys:
        raise ValueError(f"unknown search-policy quota keys: {unknown_keys}")
    try:
        prefix_size = int(policy.quotas["prefix_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("quotas.prefix_size must be a positive integer") from exc
    if prefix_size <= 0 or prefix_size > len(registry):
        raise ValueError(f"quotas.prefix_size must be between 1 and {len(registry)}")

    raw_caps = policy.quotas.get("max_per_value")
    if not isinstance(raw_caps, Mapping) or not raw_caps:
        raise ValueError("quotas.max_per_value must be a non-empty object")
    caps: dict[str, int] = {}
    for raw_field, raw_cap in raw_caps.items():
        field_name = _text(raw_field)
        if field_name not in QUOTA_FIELDS:
            raise ValueError(f"quota uses unsupported public field: {field_name!r}")
        try:
            cap = int(raw_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"quota cap for {field_name!r} must be a positive integer"
            ) from exc
        if cap <= 0 or cap > prefix_size:
            raise ValueError(
                f"quota cap for {field_name!r} must be between 1 and prefix_size"
            )
        if cap * int(registry[field_name].nunique()) < prefix_size:
            raise ValueError(
                f"quota cap for {field_name!r} cannot fill the requested prefix"
            )
        caps[field_name] = cap
    return prefix_size, caps


def _apply_prefix_quotas(
    base_order: np.ndarray,
    registry: pd.DataFrame,
    quota_config: tuple[int, dict[str, int]] | None,
) -> np.ndarray:
    if quota_config is None:
        return base_order

    prefix_size, caps = quota_config
    value_arrays = {
        field_name: registry[field_name].to_numpy(dtype=str) for field_name in caps
    }
    counts: dict[str, dict[str, int]] = {field_name: {} for field_name in caps}
    selected: list[int] = []
    deferred: list[int] = []
    for raw_index in base_order:
        index = int(raw_index)
        eligible = len(selected) < prefix_size and all(
            counts[field_name].get(value_arrays[field_name][index], 0) < cap
            for field_name, cap in caps.items()
        )
        if not eligible:
            deferred.append(index)
            continue
        selected.append(index)
        for field_name in caps:
            value = value_arrays[field_name][index]
            counts[field_name][value] = counts[field_name].get(value, 0) + 1

    if len(selected) < prefix_size:
        missing = prefix_size - len(selected)
        selected.extend(deferred[:missing])
        deferred = deferred[missing:]
    return np.asarray([*selected, *deferred], dtype=np.int64)


def _match_mask(registry: pd.DataFrame, when: Mapping[str, str]) -> np.ndarray:
    mask = np.ones(len(registry), dtype=bool)
    for field_name, expected in when.items():
        mask &= registry[field_name].to_numpy(dtype=str) == expected
    return mask


def compile_policy_scores(
    candidates: pd.DataFrame,
    policy: SearchPolicy,
) -> np.ndarray:
    """Compile sparse generator output into full, deterministic candidate scores."""

    registry = build_public_registry(candidates)
    return _compile_public_policy_scores(registry, policy)


def _compile_public_policy_scores(
    registry: pd.DataFrame,
    policy: SearchPolicy,
) -> np.ndarray:
    _validate_public_policy(policy, registry)
    scores = np.zeros(len(registry), dtype=float)
    id_to_index = {
        candidate_id: index
        for index, candidate_id in enumerate(registry["candidate_id"])
    }

    factor_fields = (
        "disease",
        "feature",
        "anatomy_id",
        "hemisphere",
        "map_group",
        "source",
    )
    factor_codes: dict[str, np.ndarray] = {}
    factor_sizes: dict[str, int] = {}
    for field_name in factor_fields:
        codes, uniques = pd.factorize(registry[field_name], sort=False)
        factor_codes[field_name] = codes
        factor_sizes[field_name] = len(uniques)

    disease_weight = np.zeros(factor_sizes["disease"], dtype=float)
    feature_weight = np.zeros(factor_sizes["feature"], dtype=float)
    anatomy_weight = np.zeros(factor_sizes["anatomy_id"], dtype=float)
    hemisphere_weight = np.zeros(factor_sizes["hemisphere"], dtype=float)
    map_group_weight = np.zeros(factor_sizes["map_group"], dtype=float)
    source_weight = np.zeros(factor_sizes["source"], dtype=float)
    disease_feature_weight = np.zeros(
        (factor_sizes["disease"], factor_sizes["feature"]),
        dtype=float,
    )
    disease_anatomy_weight = np.zeros(
        (factor_sizes["disease"], factor_sizes["anatomy_id"]),
        dtype=float,
    )
    feature_anatomy_weight = np.zeros(
        (factor_sizes["feature"], factor_sizes["anatomy_id"]),
        dtype=float,
    )
    disease_feature_hemisphere_weight = np.zeros(
        (
            factor_sizes["disease"],
            factor_sizes["feature"],
            factor_sizes["hemisphere"],
        ),
        dtype=float,
    )
    anchor_bonus = np.zeros(len(registry), dtype=float)

    for rank, anchor in enumerate(policy.anchors):
        anchor_index = id_to_index[anchor.candidate_id]
        anchor_weight = max(0.0, float(anchor.score))
        rank_discount = 1.0 / math.sqrt(rank + 1.0)
        weight = anchor_weight * rank_discount
        disease = factor_codes["disease"][anchor_index]
        feature = factor_codes["feature"][anchor_index]
        anatomy = factor_codes["anatomy_id"][anchor_index]
        hemisphere = factor_codes["hemisphere"][anchor_index]
        map_group = factor_codes["map_group"][anchor_index]
        source = factor_codes["source"][anchor_index]

        disease_weight[disease] = max(disease_weight[disease], 0.02 * weight)
        feature_weight[feature] = max(feature_weight[feature], 0.03 * weight)
        anatomy_weight[anatomy] = max(anatomy_weight[anatomy], 0.35 * weight)
        hemisphere_weight[hemisphere] = max(
            hemisphere_weight[hemisphere],
            0.05 * weight,
        )
        map_group_weight[map_group] = max(
            map_group_weight[map_group],
            0.08 * weight,
        )
        source_weight[source] = max(source_weight[source], 0.01 * weight)
        disease_feature_weight[disease, feature] = max(
            disease_feature_weight[disease, feature],
            0.20 * weight,
        )
        disease_anatomy_weight[disease, anatomy] = max(
            disease_anatomy_weight[disease, anatomy],
            0.70 * weight,
        )
        feature_anatomy_weight[feature, anatomy] = max(
            feature_anatomy_weight[feature, anatomy],
            0.65 * weight,
        )
        disease_feature_hemisphere_weight[disease, feature, hemisphere] = max(
            disease_feature_hemisphere_weight[disease, feature, hemisphere],
            0.50 * weight,
        )
        anchor_bonus[anchor_index] = max(
            anchor_bonus[anchor_index],
            3.0 + 0.5 * anchor_weight + 0.5 * rank_discount,
        )

    disease_codes = factor_codes["disease"]
    feature_codes = factor_codes["feature"]
    anatomy_codes = factor_codes["anatomy_id"]
    hemisphere_codes = factor_codes["hemisphere"]
    scores = np.maximum.reduce(
        (
            disease_weight[disease_codes],
            feature_weight[feature_codes],
            anatomy_weight[anatomy_codes],
            hemisphere_weight[hemisphere_codes],
            map_group_weight[factor_codes["map_group"]],
            source_weight[factor_codes["source"]],
            disease_feature_weight[disease_codes, feature_codes],
            disease_anatomy_weight[disease_codes, anatomy_codes],
            feature_anatomy_weight[feature_codes, anatomy_codes],
            disease_feature_hemisphere_weight[
                disease_codes,
                feature_codes,
                hemisphere_codes,
            ],
            anchor_bonus,
        )
    )

    for rule in policy.rules:
        scores[_match_mask(registry, rule.when)] += float(rule.weight)

    return scores


def compile_policy_order(candidates: pd.DataFrame, policy: SearchPolicy) -> np.ndarray:
    registry = build_public_registry(candidates)
    scores = _compile_public_policy_scores(registry, policy)
    candidate_ids = registry["candidate_id"].to_numpy(dtype=str)
    base_order = np.lexsort((candidate_ids, -scores))
    quota_config = _validated_quota_config(policy, registry)
    return _apply_prefix_quotas(base_order, registry, quota_config)


def policy_from_mapped_hypotheses(
    mapped: pd.DataFrame,
    *,
    method: str,
    trial: int,
    seed: int = 0,
) -> SearchPolicy:
    """Convert legacy mapped output into exact anchors without failure sentinels."""

    sub = mapped[mapped["method"].eq(method)].copy()
    if not sub.empty and "seed" in sub:
        available = sorted(
            int(value)
            for value in pd.to_numeric(sub["seed"], errors="coerce").dropna().unique()
        )
        if available:
            selected = available[trial % len(available)]
            sub = sub[pd.to_numeric(sub["seed"], errors="coerce").eq(selected)].copy()

    sort_columns = [column for column in ("seed", "generated_rank") if column in sub]
    if sort_columns:
        sub = sub.sort_values(sort_columns, kind="mergesort")

    anchors: list[PolicyAnchor] = []
    seen: set[str] = set()
    for rank, row in enumerate(sub.to_dict(orient="records"), start=1):
        if row.get("mapping_status") != "mapped":
            continue
        candidate_id = _text(row.get("mapped_candidate_id"))
        if not candidate_id or candidate_id in seen:
            continue
        confidence = pd.to_numeric(row.get("generated_confidence"), errors="coerce")
        score = float(confidence) if pd.notna(confidence) else 1.0
        anchors.append(
            PolicyAnchor(
                candidate_id=candidate_id,
                score=max(0.05, score),
                rationale=_text(row.get("generated_rationale")),
                evidence_ids=(),
            )
        )
        seen.add(candidate_id)

    return SearchPolicy(
        method=method,
        trial=trial,
        anchors=tuple(anchors),
        metadata={
            "source": "legacy-mapped-output",
            "seed": seed,
            "mapped_anchor_count": len(anchors),
        },
    )


def classify_observed_feedback(
    adjusted_d: float,
    p_value: float,
    *,
    expected_direction: str | None = None,
    alpha: float = 0.01,
    min_abs_d: float = 0.15,
) -> str:
    """Classify only the observable result of an executed hypothesis."""

    if not math.isfinite(adjusted_d) or not math.isfinite(p_value):
        return "inconclusive"
    if p_value >= alpha or abs(adjusted_d) <= min_abs_d:
        return "inconclusive"

    expected = _text(expected_direction).casefold()
    if expected in {"decrease", "decreased", "negative", "lower", "down"}:
        return "supported" if adjusted_d < 0 else "contradicted"
    if expected in {"increase", "increased", "positive", "higher", "up"}:
        return "supported" if adjusted_d > 0 else "contradicted"
    return "supported"


def dump_registry_jsonl(candidates: pd.DataFrame) -> str:
    """Serialize the public registry without leaking result columns."""

    records = public_registry_records(candidates)
    return "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
    )


def candidate_ids_from_policy(policy: SearchPolicy) -> tuple[str, ...]:
    return tuple(anchor.candidate_id for anchor in policy.anchors)


def policy_rules_from_factor_preferences(
    preferences: Iterable[tuple[float, Mapping[str, Any], str]],
) -> tuple[PolicyRule, ...]:
    return tuple(
        PolicyRule(
            weight=float(weight),
            when={str(key): _text(value) for key, value in when.items()},
            rationale=_text(rationale),
        )
        for weight, when, rationale in preferences
    )
