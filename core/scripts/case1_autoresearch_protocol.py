"""Blinded round protocol for full autoresearch comparisons in Case Study 1.

Every executed hypothesis must be selected explicitly from the round menu.  The
protocol never expands sparse framework output into an implicit full ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROTOCOL_SCHEMA = "case1-autoresearch-closed-loop.v1"
SELECTION_SCHEMA = "case1-autoresearch-selection.v1"
FEEDBACK_SCHEMA = "case1-autoresearch-feedback.v1"

PUBLIC_MENU_FIELDS = (
    "candidate_id",
    "disease",
    "modality",
    "source",
    "roi_index",
    "anatomy_full",
    "hemisphere",
    "network",
    "structure_class",
    "feature",
    "feature_family",
)

HIDDEN_OUTCOME_FIELDS = frozenset(
    {
        "adjusted_residual_d",
        "abs_adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "q_fdr_disease",
        "q_fdr_modality",
        "direction",
        "execution_succeeded",
        "gt_rank",
        "is_gt_top",
        "is_strict_fdr",
    }
)

HIDDEN_SCORE_PREFIXES = ("score_", "kg_")


@dataclass(frozen=True)
class Case1AutoresearchProtocol:
    """Frozen controls shared by all research controllers."""

    rounds: int = 5
    batch_size: int = 20
    menu_multiplier: int = 20
    support_alpha: float = 0.01
    support_min_abs_d: float = 0.15
    gt_top_fraction: float = 0.01
    menu_seed_base: int = 20260821
    open_coscientist_iterations: int = 1
    open_coscientist_overgeneration_factor: float = 1.0
    open_coscientist_tool_generation_enabled: bool = False
    open_coscientist_debate_cohorts_enabled: bool = True
    native_retrieval_enabled: bool = True

    @property
    def menu_size(self) -> int:
        return self.batch_size * self.menu_multiplier

    @property
    def experiment_slots(self) -> int:
        return self.rounds * self.batch_size

    def validate(self, *, candidate_count: int | None = None) -> None:
        if self.rounds < 1 or self.batch_size < 1:
            raise ValueError("rounds and batch_size must be positive")
        if self.menu_multiplier < 1:
            raise ValueError("menu_multiplier must be positive")
        if not 0.0 < self.support_alpha < 1.0:
            raise ValueError("support_alpha must be in (0, 1)")
        if self.support_min_abs_d < 0 or not math.isfinite(
            self.support_min_abs_d
        ):
            raise ValueError("support_min_abs_d must be finite and non-negative")
        if not 0.0 < self.gt_top_fraction <= 1.0:
            raise ValueError("gt_top_fraction must be in (0, 1]")
        if self.open_coscientist_iterations < 1:
            raise ValueError("Open Co-Scientist must retain at least one evolution loop")
        if (
            not math.isfinite(self.open_coscientist_overgeneration_factor)
            or self.open_coscientist_overgeneration_factor < 1.0
        ):
            raise ValueError(
                "Open Co-Scientist overgeneration factor must be finite and >= 1"
            )
        if candidate_count is not None and self.rounds * self.menu_size > candidate_count:
            raise ValueError(
                "candidate registry is too small for non-repeating round menus"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": PROTOCOL_SCHEMA, **asdict(self)}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).replace("\t", " ").splitlines()).strip()


def blinded_public_registry(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return the only candidate columns a competing controller may inspect."""

    missing = [field for field in PUBLIC_MENU_FIELDS if field not in candidates]
    if missing:
        raise ValueError(f"candidate registry is missing public fields: {missing}")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise ValueError("candidate_id values must be unique")
    public = candidates.loc[:, PUBLIC_MENU_FIELDS].copy()
    forbidden = [
        column
        for column in public
        if column in HIDDEN_OUTCOME_FIELDS
        or str(column).startswith(HIDDEN_SCORE_PREFIXES)
    ]
    if forbidden:
        raise AssertionError(f"blinded registry leaked hidden columns: {forbidden}")
    for column in public:
        public[column] = public[column].map(_clean_cell)
    if public["candidate_id"].eq("").any():
        raise ValueError("candidate_id cannot be empty")
    return public.reset_index(drop=True)


def menu_indices(
    candidate_count: int,
    *,
    seed: int,
    protocol: Case1AutoresearchProtocol,
) -> list[np.ndarray]:
    """Create outcome-blind, non-repeating menus shared by every method."""

    protocol.validate(candidate_count=candidate_count)
    rng = np.random.default_rng(protocol.menu_seed_base + 1_000_003 * int(seed))
    required = protocol.rounds * protocol.menu_size
    indices = rng.choice(candidate_count, size=required, replace=False)
    return [
        indices[start : start + protocol.menu_size].astype(np.int64, copy=False)
        for start in range(0, required, protocol.menu_size)
    ]


def render_menu_tsv(menu: pd.DataFrame) -> str:
    fields = [field for field in PUBLIC_MENU_FIELDS if field in menu]
    lines = ["\t".join(fields)]
    for row in menu.loc[:, fields].itertuples(index=False, name=None):
        lines.append("\t".join(_clean_cell(value) for value in row))
    return "\n".join(lines)


def classify_outcome(
    outcome: Mapping[str, Any],
    protocol: Case1AutoresearchProtocol,
) -> str:
    succeeded = bool(outcome.get("execution_succeeded", True))
    if not succeeded:
        return "execution_failed"
    try:
        effect = float(outcome.get("adjusted_residual_d"))
        p_value = float(outcome.get("p_value"))
    except (TypeError, ValueError):
        return "execution_failed"
    if not math.isfinite(effect) or not math.isfinite(p_value):
        return "execution_failed"
    if p_value < protocol.support_alpha and abs(effect) > protocol.support_min_abs_d:
        return "supported"
    return "inconclusive"


def reveal_feedback(
    candidate_ids: Sequence[str],
    outcomes_by_id: Mapping[str, Mapping[str, Any]],
    *,
    round_index: int,
    protocol: Case1AutoresearchProtocol,
) -> list[dict[str, Any]]:
    """Reveal only results for committed candidates, never global GT labels."""

    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        outcome = outcomes_by_id[str(candidate_id)]
        status = classify_outcome(outcome, protocol)
        effect = pd.to_numeric(outcome.get("adjusted_residual_d"), errors="coerce")
        p_value = pd.to_numeric(outcome.get("p_value"), errors="coerce")
        rows.append(
            {
                "schema_version": FEEDBACK_SCHEMA,
                "round": int(round_index),
                "candidate_id": str(candidate_id),
                "status": status,
                "effect_size_d": (
                    float(effect) if status != "execution_failed" else None
                ),
                "nominal_p_value": (
                    float(p_value) if status != "execution_failed" else None
                ),
                "observed_direction": (
                    "case_higher"
                    if status != "execution_failed" and float(effect) > 0
                    else "case_lower"
                    if status != "execution_failed" and float(effect) < 0
                    else "none"
                ),
            }
        )
    return rows


def write_selection_commitment(
    path: Path,
    *,
    method: str,
    seed: int,
    round_index: int,
    requested_slots: int,
    selected_candidate_ids: Sequence[str],
    validation_rows: Sequence[Mapping[str, Any]],
    menu_sha256: str,
) -> dict[str, Any]:
    """Commit a selection before any selected outcome is read."""

    payload = {
        "schema_version": SELECTION_SCHEMA,
        "method": str(method),
        "seed": int(seed),
        "round": int(round_index),
        "requested_slots": int(requested_slots),
        "selected_candidate_ids": [str(value) for value in selected_candidate_ids],
        "validation_rows": [dict(row) for row in validation_rows],
        "menu_sha256": str(menu_sha256),
        "outcomes_read_before_commit": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["commit_sha256"] = sha256_bytes(canonical.encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def feedback_table(feedback: Iterable[Mapping[str, Any]]) -> str:
    records = list(feedback)
    if not records:
        return "No experiments have been run yet."
    lines = [
        "round\tcandidate_id\tstatus\teffect_size_d\tnominal_p_value\tobserved_direction"
    ]
    for row in records:
        lines.append(
            "\t".join(
                _clean_cell(row.get(field))
                for field in (
                    "round",
                    "candidate_id",
                    "status",
                    "effect_size_d",
                    "nominal_p_value",
                    "observed_direction",
                )
            )
        )
    return "\n".join(lines)


def build_round_research_goal(
    *,
    method: str,
    seed: int,
    round_index: int,
    menu: pd.DataFrame,
    prior_feedback: Sequence[Mapping[str, Any]],
    protocol: Case1AutoresearchProtocol,
) -> str:
    """Build the shared scientific task while preserving native workflows."""

    start_rank = round_index * protocol.batch_size + 1
    end_rank = start_rank + protocol.batch_size - 1
    return f"""Transdiagnostic brain-atlas discovery, closed-loop round {round_index + 1} of {protocol.rounds}.

Scientific objective: identify disease-by-brain-region-by-imaging-feature tests
with reproducible case-control alterations. Direction is not preregistered;
positive and negative effects are equally valid discoveries.

Use your framework's native planning, literature search, specialist discussion,
reflection, criticism, and ranking capabilities. Select exactly
{protocol.batch_size} experiments from the CURRENT ROUND MENU. Every selected
experiment must reproduce its candidate_id verbatim. Do not invent, repair, or
generalize an ID. Do not repeat any candidate listed in PRIOR EXPERIMENTAL
FEEDBACK. Rank the selections {start_rank}-{end_rank}. Results from this round
will be returned before the next round.

The menu is outcome-blind. It contains no effect size, P value, FDR, GT label,
NeuroDiscovery score, KG score, or result-derived field. The seed ({seed}) is an
experimental replicate identifier, not scientific evidence.

PRIOR EXPERIMENTAL FEEDBACK
{feedback_table(prior_feedback)}

CURRENT ROUND MENU
{render_menu_tsv(menu)}

Return the requested framework-native artifact, but make the final executable
selection machine-readable. The final selection must contain exactly one item
for every rank {start_rank}-{end_rank}, with candidate_id, one-sentence
rationale, and confidence in [0,1]. Missing, duplicate, off-menu, or malformed
items consume their experiment slots and are not repaired by the evaluator.

End with exactly one JSON object in this shape:
{{"method":"{method}","hypotheses":[{{"rank":{start_rank},
"candidate_id":"<verbatim menu ID>","rationale":"<one sentence>",
"confidence":0.75}}]}}
The hypotheses array must contain all {protocol.batch_size} consecutive ranks.
"""


def validate_selected_ids(
    selected: Sequence[Mapping[str, Any]],
    *,
    menu_candidate_ids: set[str],
    previously_executed: set[str],
    start_rank: int,
    requested_slots: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Validate every requested slot without filling failed proposals."""

    by_rank: dict[int, list[Mapping[str, Any]]] = {}
    for raw in selected:
        try:
            rank = int(raw.get("rank"))
        except (TypeError, ValueError):
            rank = -1
        by_rank.setdefault(rank, []).append(raw)

    accepted: list[str] = []
    accepted_set: set[str] = set()
    audit: list[dict[str, Any]] = []
    for rank in range(start_rank, start_rank + requested_slots):
        items = by_rank.get(rank, [])
        errors: list[str] = []
        item: Mapping[str, Any] = items[0] if len(items) == 1 else {}
        if not items:
            errors.append("missing_rank")
        elif len(items) > 1:
            errors.append("duplicate_rank")
        candidate_id = str(item.get("candidate_id") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        try:
            confidence = float(item.get("confidence"))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            confidence = None
            errors.append("invalid_confidence")
        if candidate_id not in menu_candidate_ids:
            errors.append("off_menu_candidate_id")
        if candidate_id in previously_executed:
            errors.append("previously_executed")
        if candidate_id in accepted_set:
            errors.append("duplicate_candidate")
        if not rationale:
            errors.append("missing_rationale")
        valid = not errors
        if valid:
            accepted.append(candidate_id)
            accepted_set.add(candidate_id)
        audit.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "rationale": rationale,
                "confidence": confidence,
                "valid": valid,
                "status": "valid" if valid else ";".join(dict.fromkeys(errors)),
            }
        )
    return accepted, audit


def evaluate_prefix(
    slot_rows: pd.DataFrame,
    *,
    requested_budgets: Sequence[int],
) -> pd.DataFrame:
    """Evaluate direct selections; invalid generation slots remain in the cost."""

    rows: list[dict[str, Any]] = []
    ordered = slot_rows.sort_values("slot", kind="stable")
    for budget in requested_budgets:
        prefix = ordered[ordered["slot"] <= int(budget)]
        requested = int(min(budget, len(ordered)))
        valid = int(prefix["valid"].fillna(False).astype(bool).sum())
        executed = int(prefix["execution_succeeded"].fillna(False).astype(bool).sum())
        gt_hits = int(prefix["is_gt_top"].fillna(False).astype(bool).sum())
        supported = int(prefix["supported"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "budget": requested,
                "valid_hypotheses": valid,
                "format_success_rate": valid / requested if requested else 0.0,
                "execution_succeeded": executed,
                "execution_success_rate_per_slot": (
                    executed / requested if requested else 0.0
                ),
                "gt_hits": gt_hits,
                "gt_precision_per_slot": gt_hits / requested if requested else 0.0,
                "supported_hits": supported,
                "supported_precision_per_slot": (
                    supported / requested if requested else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "Case1AutoresearchProtocol",
    "FEEDBACK_SCHEMA",
    "HIDDEN_OUTCOME_FIELDS",
    "PROTOCOL_SCHEMA",
    "PUBLIC_MENU_FIELDS",
    "SELECTION_SCHEMA",
    "blinded_public_registry",
    "build_round_research_goal",
    "classify_outcome",
    "evaluate_prefix",
    "feedback_table",
    "menu_indices",
    "render_menu_tsv",
    "reveal_feedback",
    "sha256_file",
    "validate_selected_ids",
    "write_selection_commitment",
]
