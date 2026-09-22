"""Run feedback-conditioned NeuroDiscovery hindcasting with dynamic refills.

Unlike the legacy closed-loop evaluator, this runner does not freeze one
candidate table before the first execution. It maintains a reservoir of public
hypotheses, reveals only early-window outcomes after each selected batch, and
then regenerates hypotheses from the same frozen KG with an ephemeral feedback
state. The formal KG is never modified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.scripts.case_study_closed_loop_engine import _select_diverse_batch
from neurooracle.scripts.case_study_hindcasting_eval import (
    DISCOVERY_METRIC_CONTRACT_VERSION,
    _edge_pair,
    _future_indexes,
    _is_claim_backed_edge,
    _score_hypothesis,
    load_future_claim_records,
)
from neurooracle.scripts.generate_neurodiscovery_hindcasting_replicates import (
    NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT,
    NEURODISCOVERY_KGE_WEIGHT,
    generate_one,
)
from neurooracle.scripts.run_case_study_hindcasting import (
    DEFAULT_WINDOWS,
    _hypothesis_payload_semantic_key,
    parse_window,
)
from neurooracle.scripts.run_neurodiscovery_closed_loop_hindcasting import (
    FACTOR_FIELDS,
    LoopProfile,
    _finite_float,
    _mediator_id,
    _relation_signature,
    _static_score,
    profile_slate,
)
from neurooracle.src.case_studies import (
    CaseStudy,
    case_study_by_name,
    list_case_study_names,
)
from neurooracle.src.claim_semantics import (
    SEMANTIC_ENDPOINT_IDENTITY_VERSION,
    concept_atom_roles,
    semantic_claim_pair,
)
from neurooracle.src.feedback_state import (
    INCONCLUSIVE,
    SUPPORTED,
    FeedbackRecord,
    FeedbackState,
)
from neurooracle.src.hypothesis_cli import load_graph
from neurooracle.src.hindcasting_static_policy import RELATION_COMPONENT_RAW_FIELDS
from neurooracle.src.hindcasting_eligibility import (
    load_locked_hindcasting_eligibility,
    sha256_file,
)
from neurooracle.src.kge.complex_scorer import ComplExScorer


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "neurodiscovery-dynamic-closed-loop-hindcasting.v4"
DEFAULT_BUDGETS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
SOURCE_BUNDLE_SCHEMA = "neurodiscovery-experiment-source-bundle.v1"
REQUIRED_RUN_ARTIFACTS = (
    "executed_hypotheses.json",
    "feedback_overlay.csv",
    "generation_rounds.csv",
    "hidden_outcomes.csv",
    "metrics_by_k.csv",
    "proposed_hypotheses.jsonl.gz",
)
ALLOWED_GENERATOR_CLOSURE_NAMES = frozenset(
    {
        "branch_engines",
        "case",
        "config",
        "graph",
        "kge_checkpoint",
        "kge_scorer",
        "profile",
        "rounds_dir",
        "seed",
        "window",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_generator_closure(generator: Callable[..., Any]) -> tuple[str, ...]:
    captured = tuple(sorted(generator.__code__.co_freevars))
    unexpected = sorted(set(captured) - ALLOWED_GENERATOR_CLOSURE_NAMES)
    if unexpected:
        raise ValueError(
            "dynamic generator closure captured hidden future outcomes: "
            f"{unexpected}"
        )
    return captured


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verified_source_bundle(
    manifest_path: Path | None,
    *,
    require_live_source: bool | None = None,
    verify_references: bool = True,
) -> dict[str, Any] | None:
    """Verify that the live run still matches one immutable source bundle."""

    if manifest_path is None:
        return None
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SOURCE_BUNDLE_SCHEMA:
        raise ValueError("source bundle uses an incompatible schema")

    bundle_root = manifest_path.parent
    archive_root = (bundle_root / "files").resolve()
    executing_from_archive = Path(__file__).resolve().is_relative_to(archive_root)
    if require_live_source is None:
        require_live_source = not executing_from_archive
    for record in payload.get("files") or ():
        expected = str(record.get("sha256") or "").upper()
        live_path = Path(str(record.get("source_path") or ""))
        bundled_path = bundle_root / str(record.get("bundle_path") or "")
        if not expected or not bundled_path.is_file():
            raise ValueError(f"source bundle file record is incomplete: {record}")
        if require_live_source:
            if not live_path.is_file():
                raise ValueError(f"live source is absent: {live_path}")
            if sha256_file(live_path) != expected:
                raise ValueError(f"live source differs from frozen bundle: {live_path}")
        if sha256_file(bundled_path) != expected:
            raise ValueError(f"archived source differs from bundle manifest: {bundled_path}")

    for record in payload.get("references") or ():
        expected = str(record.get("sha256") or "").upper()
        expected_bytes = int(record.get("bytes") or -1)
        referenced_path = Path(str(record.get("path") or ""))
        if (
            not expected
            or expected_bytes < 0
            or not referenced_path.is_file()
        ):
            raise ValueError(f"source bundle reference is incomplete: {record}")
        if referenced_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"referenced input differs from frozen bundle: {referenced_path}"
            )
        if verify_references and sha256_file(referenced_path) != expected:
            raise ValueError(f"referenced input differs from frozen bundle: {referenced_path}")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "bundled_files": len(payload.get("files") or ()),
        "referenced_files": len(payload.get("references") or ()),
        "execution_source": "archive" if executing_from_archive else "live_workspace",
        "live_source_match_required": bool(require_live_source),
        "live_source_verified": bool(require_live_source),
        "archived_source_verified": True,
        "referenced_inputs_verified": bool(verify_references),
        "environment_variables_captured": False,
    }


@dataclass(frozen=True)
class DynamicLoopConfig:
    feedback_enabled: bool = True
    proposal_batch_size: int = 3000
    max_unique_proposals: int = 10000
    max_generation_rounds: int = 8
    max_stagnant_generation_rounds: int = 2
    max_executions: int = 1000
    execution_batch_size: int = 10
    refresh_every_executions: int = 100
    refill_below: int = 500
    feedback_years: int = 2
    feedback_start_budget: int = 50
    min_supported_before_feedback: int = 2
    seed_diversity_fraction: float = 0.35
    candidate_pool_mode: str = "hybrid"
    task_scope_fraction: float = 0.15
    evidence_frontier_fraction: float = 0.0
    protect_general_top_k: int = 100
    path_variants_per_endpoint: int = 2
    feedback_mutation_fraction: float = 0.35
    max_paths_per_endpoint: int = 4
    endpoint_canonical_quality_weight: float = (
        NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT
    )
    kge_weight: float = 0.0

    def validate(self) -> None:
        positive = (
            self.proposal_batch_size,
            self.max_unique_proposals,
            self.max_generation_rounds,
            self.max_stagnant_generation_rounds,
            self.max_executions,
            self.execution_batch_size,
            self.refresh_every_executions,
            self.feedback_years,
            self.feedback_start_budget,
            self.path_variants_per_endpoint,
            self.max_paths_per_endpoint,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("dynamic-loop sizes must be positive")
        if self.max_unique_proposals < self.max_executions:
            raise ValueError("max_unique_proposals must cover max_executions")
        if self.refill_below < 0:
            raise ValueError("refill_below must be non-negative")
        if not 0.0 <= self.seed_diversity_fraction <= 1.0:
            raise ValueError("seed_diversity_fraction must be in [0, 1]")
        if not 0.0 <= self.task_scope_fraction <= 1.0:
            raise ValueError("task_scope_fraction must be in [0, 1]")
        if not 0.0 <= self.evidence_frontier_fraction <= 1.0:
            raise ValueError("evidence_frontier_fraction must be in [0, 1]")
        if not 0.0 <= self.feedback_mutation_fraction <= 1.0:
            raise ValueError("feedback_mutation_fraction must be in [0, 1]")
        if not 0.0 <= self.endpoint_canonical_quality_weight <= 1.0:
            raise ValueError(
                "endpoint_canonical_quality_weight must be in [0, 1]"
            )
        if not 0.0 <= self.kge_weight <= 1.0:
            raise ValueError("kge_weight must be in [0, 1]")
        if self.candidate_pool_mode not in {"hybrid", "general", "scoped"}:
            raise ValueError("invalid candidate_pool_mode")


@dataclass
class DynamicLoopResult:
    proposed: list[dict[str, Any]]
    executed: list[dict[str, Any]]
    hidden: list[dict[str, Any]]
    feedback: list[dict[str, Any]]
    generation_rounds: list[dict[str, Any]]
    stop_reason: str
    feedback_start_budget: int
    generation_failure_slots: int = 0


def _candidate_tuple(hypothesis: Mapping[str, Any]) -> dict[str, str]:
    path_length = len(hypothesis.get("path") or [])
    return {
        "source_id": str(hypothesis.get("source_id") or "none"),
        "target_id": str(hypothesis.get("target_id") or "none"),
        "relation_signature": _relation_signature(hypothesis),
        "mediator_id": _mediator_id(hypothesis),
        "path_length_bucket": str(min(path_length, 4)),
    }


def _public_row(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(hypothesis.get("metadata") or {})
    candidate_tuple = _candidate_tuple(hypothesis)
    return {
        "candidate_id": str(hypothesis.get("id") or ""),
        **candidate_tuple,
        "source_domain": str(metadata.get("domain_a") or "unknown"),
        "target_domain": str(metadata.get("domain_b") or "unknown"),
        "confidence_score": _finite_float(hypothesis.get("confidence_score")),
        "evidence_score": _finite_float(hypothesis.get("evidence_score")),
        "novelty_score": _finite_float(hypothesis.get("novelty_score")),
        "testability_score": _finite_float(hypothesis.get("testability_score")),
        "generator_composite_score": _finite_float(
            hypothesis.get("composite_score"),
            default=_finite_float(hypothesis.get("confidence_score")),
        ),
        **{
            field: _finite_float(metadata.get(field), default=float("nan"))
            for field in RELATION_COMPONENT_RAW_FIELDS
        },
        "is_generation_failure": False,
        "is_semantic_duplicate": False,
    }


def _feedback_record(
    hypothesis: Mapping[str, Any],
    *,
    status: str,
    execution_rank: int,
) -> tuple[FeedbackRecord, dict[str, Any]]:
    raw = {
        "status": status,
        "hypothesis_id": str(hypothesis.get("id") or ""),
        "source_id": str(hypothesis.get("source_id") or ""),
        "target_id": str(hypothesis.get("target_id") or ""),
        "candidate_tuple": _candidate_tuple(hypothesis),
        "path_node_ids": list(_hypothesis_payload_semantic_key(dict(hypothesis))[1]),
        "weight": 1.0,
        "reason": (
            "strict primary support appeared inside the early feedback window"
            if status == SUPPORTED
            else "no strict primary support was visible inside the early feedback window"
        ),
        "execution_rank": execution_rank,
        "formal_kg_mutated": False,
    }
    return FeedbackRecord.from_dict(raw), raw


def _stable_generation_seed(seed: int, freeze_year: int, round_index: int) -> int:
    digest = hashlib.sha256(
        f"{seed}\x1f{freeze_year}\x1f{round_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _rank_reservoir(
    candidates: list[dict[str, Any]],
    *,
    feedback_state: FeedbackState,
    feedback_active: bool,
    profile: LoopProfile,
) -> tuple[pd.DataFrame, np.ndarray]:
    public = pd.DataFrame([_public_row(candidate) for candidate in candidates])
    base = _static_score(public, profile)
    if not feedback_active or profile.static_only:
        return public, base
    adjusted = np.asarray(base, dtype=float).copy()
    source_affinities = np.zeros(len(candidates), dtype=float)
    exact_pairs = np.zeros(len(candidates), dtype=bool)
    supported = [
        record for record in feedback_state.records if record.status == SUPPORTED
    ]
    for index, (candidate, base_score) in enumerate(
        zip(candidates, base, strict=True)
    ):
        source_id = str(candidate.get("source_id") or "")
        target_id = str(candidate.get("target_id") or "")
        affinity = 0.0
        exact = False
        for record in supported:
            if not record.source_id or source_id != record.source_id:
                continue
            if record.target_id and target_id == record.target_id:
                exact = True
                break
            affinity = max(
                affinity,
                min(1.0, max(0.0, float(record.weight))),
            )
        source_affinities[index] = affinity
        exact_pairs[index] = exact
        if exact:
            adjusted[index] = max(0.0, float(base_score) * 0.35)
        elif affinity > 0.0:
            multiplier = 1.0 + min(0.35, profile.feedback_weight * affinity)
            additive = min(0.08, 0.10 * profile.pair_feedback_weight * affinity)
            adjusted[index] = min(
                1.0,
                max(0.0, float(base_score) * multiplier + additive),
            )
    public["feedback_source_affinity"] = source_affinities
    public["feedback_exact_pair"] = exact_pairs
    return public, adjusted


def _select_dynamic_batch(
    candidates: list[dict[str, Any]],
    scores: np.ndarray,
    public: pd.DataFrame,
    *,
    select_n: int,
    feedback_active: bool,
    feedback_mutation_fraction: float,
    mutations_executed: int,
    executions_after_batch: int,
    diversity_penalty: float,
) -> np.ndarray:
    """Select one batch while enforcing a cumulative feedback-mutation quota."""

    all_indices = np.arange(len(candidates), dtype=np.int64)
    if not feedback_active or feedback_mutation_fraction <= 0.0:
        return _select_diverse_batch(
            all_indices,
            scores,
            public,
            factor_fields=FACTOR_FIELDS,
            batch_size=select_n,
            diversity_penalty=diversity_penalty,
        )

    mutation_mask = np.asarray(
        [
            bool((candidate.get("metadata") or {}).get("feedback_mutation"))
            for candidate in candidates
        ],
        dtype=bool,
    )
    desired_cumulative_mutations = int(
        round(executions_after_batch * feedback_mutation_fraction)
    )
    mutation_n = min(
        select_n,
        max(0, desired_cumulative_mutations - mutations_executed),
        int(mutation_mask.sum()),
    )
    exploration_n = min(
        select_n - mutation_n,
        int((~mutation_mask).sum()),
    )

    def select(indices: np.ndarray, count: int) -> list[int]:
        if count <= 0 or not len(indices):
            return []
        chosen = _select_diverse_batch(
            indices,
            scores,
            public,
            factor_fields=FACTOR_FIELDS,
            batch_size=count,
            diversity_penalty=diversity_penalty,
        )
        return [int(index) for index in chosen]

    selected = select(all_indices[~mutation_mask], exploration_n)
    selected.extend(select(all_indices[mutation_mask], mutation_n))
    if len(selected) < select_n:
        selected_set = set(selected)
        remaining = np.asarray(
            [index for index in all_indices if int(index) not in selected_set],
            dtype=np.int64,
        )
        selected.extend(select(remaining, select_n - len(selected)))
    return np.asarray(selected[:select_n], dtype=np.int64)


def execute_dynamic_loop(
    *,
    case_study_id: str,
    seed: int,
    freeze_year: int,
    future_start_year: int,
    future_end_year: int,
    config: DynamicLoopConfig,
    profile: LoopProfile,
    generate_round: Callable[
        [
            int,
            int,
            FeedbackState,
            frozenset[tuple[str, tuple[str, ...]]],
        ],
        list[dict[str, Any]],
    ],
    score_candidate: Callable[[dict[str, Any]], dict[str, Any]],
) -> DynamicLoopResult:
    """Execute a dynamic reservoir loop with hidden outcomes behind a callback."""

    config.validate()
    feedback_end_year = min(future_end_year, freeze_year + config.feedback_years)
    proposed: list[dict[str, Any]] = []
    by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    reservoir: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    executed_keys: set[tuple[str, tuple[str, ...]]] = set()
    executed: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    feedback_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    state = FeedbackState([])
    supported_count = 0
    feedback_mutations_executed = 0
    generation_round = 0
    stagnant_rounds = 0
    last_generation_execution = 0
    initial_valid = 0
    feedback_start = config.feedback_start_budget
    feedback_generation_seen = False

    def refill() -> int:
        nonlocal generation_round, stagnant_rounds, initial_valid
        nonlocal feedback_generation_seen
        if generation_round >= config.max_generation_rounds:
            return 0
        if len(by_key) >= config.max_unique_proposals:
            return 0
        generation_seed = _stable_generation_seed(seed, freeze_year, generation_round)
        feedback_active = (
            config.feedback_enabled
            and len(executed) >= feedback_start
            and supported_count >= config.min_supported_before_feedback
        )
        feedback_generation_seen = feedback_generation_seen or feedback_active
        generation_state = state if feedback_active else FeedbackState([])
        raw_candidates = generate_round(
            generation_round,
            generation_seed,
            generation_state,
            frozenset(by_key),
        )
        added = 0
        duplicates = 0
        invalid = 0
        for raw in raw_candidates:
            metadata = dict(raw.get("metadata") or {})
            if (
                raw.get("hypothesis_type") == "generation_failure"
                or metadata.get("generation_failure") is True
                or not (raw.get("path") or [])
            ):
                invalid += 1
                continue
            key = _hypothesis_payload_semantic_key(raw)
            if key in by_key:
                duplicates += 1
                continue
            if len(by_key) >= config.max_unique_proposals:
                break
            candidate = dict(raw)
            metadata.update(
                {
                    "case_study_id": case_study_id,
                    "dynamic_generation_round": generation_round,
                    "dynamic_generation_seed": generation_seed,
                }
            )
            candidate["metadata"] = metadata
            candidate["id"] = (
                f"NDCL:{case_study_id}:{freeze_year}:{seed}:"
                f"{len(by_key) + 1:06d}"
            )
            by_key[key] = candidate
            reservoir[key] = candidate
            proposed.append(candidate)
            added += 1
        generation_rows.append(
            {
                "round": generation_round,
                "generation_seed": generation_seed,
                "feedback_active": feedback_active,
                "feedback_records": len(state.records),
                "supported_records": supported_count,
                "raw_candidates": len(raw_candidates),
                "new_unique_candidates": added,
                "duplicates": duplicates,
                "invalid": invalid,
                "cumulative_unique_proposals": len(by_key),
                "reservoir_after_refill": len(reservoir),
                "executed_before_refill": len(executed),
                "one_mediator_candidates": sum(
                    (candidate.get("metadata") or {}).get("path_template")
                    == "one_mediator"
                    for candidate in raw_candidates
                ),
                "two_mediator_candidates": sum(
                    (candidate.get("metadata") or {}).get("path_template")
                    == "two_mediator"
                    for candidate in raw_candidates
                ),
                "feedback_mutation_candidates": sum(
                    bool((candidate.get("metadata") or {}).get("feedback_mutation"))
                    for candidate in raw_candidates
                ),
            }
        )
        if generation_round == 0:
            initial_valid = added
        stagnant_rounds = stagnant_rounds + 1 if added == 0 else 0
        generation_round += 1
        return added

    # Sparse tasks start feedback earlier, matching the legacy adaptive gate.
    feedback_start = config.feedback_start_budget
    refill()
    feedback_start = min(
        config.feedback_start_budget,
        max(config.execution_batch_size, math.ceil(max(1, initial_valid) * 0.25)),
    )

    stop_reason = "execution_budget_reached"
    while len(executed) < config.max_executions:
        if not reservoir:
            if (
                generation_round >= config.max_generation_rounds
                or stagnant_rounds >= config.max_stagnant_generation_rounds
                or len(by_key) >= config.max_unique_proposals
            ):
                stop_reason = "candidate_space_exhausted"
                break
            refill()
            last_generation_execution = len(executed)
            if not reservoir:
                continue

        feedback_active = (
            config.feedback_enabled
            and len(executed) >= feedback_start
            and supported_count >= config.min_supported_before_feedback
        )
        keys = list(reservoir)
        candidates = [reservoir[key] for key in keys]
        public, scores = _rank_reservoir(
            candidates,
            feedback_state=state,
            feedback_active=feedback_active,
            profile=profile,
        )
        select_n = min(
            config.execution_batch_size,
            config.max_executions - len(executed),
            len(candidates),
        )
        selected_indices = _select_dynamic_batch(
            candidates,
            scores,
            public,
            select_n=select_n,
            feedback_active=feedback_active,
            feedback_mutation_fraction=config.feedback_mutation_fraction,
            mutations_executed=feedback_mutations_executed,
            executions_after_batch=(
                max(0, len(executed) - feedback_start) + select_n
            ),
            diversity_penalty=(
                profile.diversity_penalty
                if feedback_active
                else profile.warmup_diversity_penalty
            ),
        )
        for raw_index in selected_indices:
            index = int(raw_index)
            key = keys[index]
            candidate = reservoir.pop(key)
            feedback_mutations_executed += int(
                bool((candidate.get("metadata") or {}).get("feedback_mutation"))
            )
            executed_keys.add(key)
            outcome = score_candidate(candidate)
            rank = len(executed) + 1
            primary_year = outcome.get("primary_year")
            first_future_year = outcome.get("first_future_year")
            early_primary = bool(
                outcome.get("primary_hit")
                and primary_year is not None
                and int(primary_year) <= feedback_end_year
            )
            status = SUPPORTED if early_primary else INCONCLUSIVE
            record, feedback_row = _feedback_record(
                candidate,
                status=status,
                execution_rank=rank,
            )
            if config.feedback_enabled and status == SUPPORTED:
                state.records.append(record)
            supported_count += int(config.feedback_enabled and status == SUPPORTED)
            execution_row = dict(candidate)
            execution_metadata = dict(execution_row.get("metadata") or {})
            execution_metadata.update(
                {
                    "execution_rank": rank,
                    "selection_score": float(scores[index]),
                    "feedback_source_affinity_at_selection": float(
                        public.iloc[index].get("feedback_source_affinity", 0.0)
                    ),
                    "feedback_exact_pair_at_selection": bool(
                        public.iloc[index].get("feedback_exact_pair", False)
                    ),
                    "feedback_active_at_selection": feedback_active,
                    "cumulative_unique_proposals_at_selection": len(by_key),
                }
            )
            execution_row["metadata"] = execution_metadata
            executed.append(execution_row)
            hidden.append(
                {
                    "candidate_id": candidate["id"],
                    "execution_rank": rank,
                    "primary_hit": bool(outcome.get("primary_hit")),
                    "primary_year": primary_year,
                    "primary_discovery_key": str(
                        outcome.get("primary_discovery_key") or ""
                    ),
                    "primary_recovered_pairs": str(
                        outcome.get("primary_recovered_pairs") or ""
                    ),
                    "any_future_hit": bool(outcome.get("any_future_hit")),
                    "first_future_year": first_future_year,
                    "early_primary_hit": early_primary,
                    "terminal_primary_hit": bool(
                        outcome.get("primary_hit")
                        and primary_year is not None
                        and int(primary_year) > feedback_end_year
                    ),
                    "terminal_any_hit": bool(
                        outcome.get("any_future_hit")
                        and first_future_year is not None
                        and int(first_future_year) > feedback_end_year
                    ),
                    "feedback_status": status,
                    "generation_failure": False,
                }
            )
            feedback_row.update(
                {
                    "feedback_end_year": feedback_end_year,
                    "outcome_observed_after_selection": True,
                    "available_to_generator": config.feedback_enabled,
                }
            )
            feedback_rows.append(feedback_row)

        executions_since_generation = len(executed) - last_generation_execution
        feedback_active_after_batch = (
            config.feedback_enabled
            and len(executed) >= feedback_start
            and supported_count >= config.min_supported_before_feedback
        )
        feedback_phase_transition = (
            feedback_active_after_batch and not feedback_generation_seen
        )
        if feedback_phase_transition:
            # Preserve at least one generation round for the first usable result.
            # Otherwise a small initial reservoir can exhaust its no-feedback
            # refill allowance before delayed support becomes observable.
            stagnant_rounds = 0
        can_generate = (
            generation_round < config.max_generation_rounds
            and len(by_key) < config.max_unique_proposals
        )
        should_refresh = (
            len(executed) < config.max_executions
            and can_generate
            and (
                feedback_phase_transition
                or (
                    stagnant_rounds < config.max_stagnant_generation_rounds
                    and (
                        executions_since_generation
                        >= config.refresh_every_executions
                        or (not reservoir and executions_since_generation > 0)
                        or (
                            feedback_active_after_batch
                            and len(reservoir) < config.refill_below
                            and executions_since_generation
                            >= config.execution_batch_size
                        )
                    )
                )
            )
        )
        if should_refresh:
            refill()
            last_generation_execution = len(executed)

    if len(executed_keys) != len(executed):
        raise RuntimeError("dynamic loop executed one semantic hypothesis twice")
    generation_failure_slots = 0
    if len(executed) < config.max_executions:
        generation_failure_slots = config.max_executions - len(executed)
        for rank in range(len(executed) + 1, config.max_executions + 1):
            candidate_id = (
                f"NDCL:{case_study_id}:{freeze_year}:{seed}:"
                f"FAILURE:{rank:06d}"
            )
            executed.append(
                {
                    "id": candidate_id,
                    "hypothesis_type": "generation_failure",
                    "source_id": "",
                    "source_name": "",
                    "target_id": "",
                    "target_name": "",
                    "path": [],
                    "confidence_score": 0.0,
                    "novelty_score": 0.0,
                    "evidence_score": 0.0,
                    "testability_score": 0.0,
                    "composite_score": 0.0,
                    "supporting_claims": [],
                    "explanation": (
                        "No valid candidate was available for this fixed-budget slot."
                    ),
                    "metadata": {
                        "case_study_id": case_study_id,
                        "generator_method": "neurodiscovery_dynamic",
                        "generation_failure": True,
                        "fixed_budget_slot": True,
                        "failure_reason": stop_reason,
                        "execution_rank": rank,
                    },
                }
            )
            hidden.append(
                {
                    "candidate_id": candidate_id,
                    "execution_rank": rank,
                    "primary_hit": False,
                    "primary_year": None,
                    "primary_discovery_key": "",
                    "primary_recovered_pairs": "",
                    "any_future_hit": False,
                    "first_future_year": None,
                    "early_primary_hit": False,
                    "terminal_primary_hit": False,
                    "terminal_any_hit": False,
                    "feedback_status": "generation_failure",
                    "generation_failure": True,
                }
            )
    return DynamicLoopResult(
        proposed=proposed,
        executed=executed,
        hidden=hidden,
        feedback=feedback_rows,
        generation_rounds=generation_rows,
        stop_reason=stop_reason,
        feedback_start_budget=feedback_start,
        generation_failure_slots=generation_failure_slots,
    )


def _historical_pairs_from_graph(
    graph: Any,
    claims_path: Path,
) -> tuple[set[tuple[str, str]], dict[str, frozenset[str]], dict[str, int]]:
    stats: Counter[str] = Counter()
    pairs: set[tuple[str, str]] = set()
    endpoint_atoms: dict[str, set[str]] = {
        str(node_id): {atom.value for atom in concept_atom_roles(node)}
        for node_id, node in graph._index.items()
        if concept_atom_roles(node)
    }
    for source_id, target_id, edge in graph.G.edges(data=True):
        relation = str(edge.get("relation_type") or "")
        if (
            not source_id
            or not target_id
            or source_id == target_id
            or relation in {"is_a", "part_of", "about"}
        ):
            continue
        if _is_claim_backed_edge(edge):
            stats["claim_backed_canonical_edges_excluded"] += 1
            continue
        pairs.add(_edge_pair(str(source_id), str(target_id)))
        stats["curated_pairs"] += 1
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            claim = json.loads(line)
            if claim.get("negated"):
                stats["negated_claims_excluded"] += 1
                continue
            projected = semantic_claim_pair(claim, graph._index)
            if projected is None:
                stats["claims_without_semantic_pair"] += 1
                continue
            subject, obj = projected
            endpoint_atoms.setdefault(subject.entity_id, set()).update(subject.atoms)
            endpoint_atoms.setdefault(obj.entity_id, set()).update(obj.atoms)
            pairs.add(_edge_pair(subject.entity_id, obj.entity_id))
            stats["semantic_claim_pairs"] += 1
    stats["historical_unique_pairs"] = len(pairs)
    return (
        pairs,
        {node_id: frozenset(atoms) for node_id, atoms in endpoint_atoms.items()},
        dict(stats),
    )


def _historical_pair_cache_fingerprint(
    graph_path: Path,
    claims_path: Path,
) -> dict[str, Any]:
    return {
        "graph_size": graph_path.stat().st_size,
        "graph_mtime_ns": graph_path.stat().st_mtime_ns,
        "claims_size": claims_path.stat().st_size,
        "claims_mtime_ns": claims_path.stat().st_mtime_ns,
        "semantic_endpoint_identity_version": SEMANTIC_ENDPOINT_IDENTITY_VERSION,
    }


def _historical_pairs_cached(
    *,
    graph: Any,
    graph_path: Path,
    claims_path: Path,
    cache_path: Path,
) -> tuple[
    set[tuple[str, str]],
    dict[str, frozenset[str]],
    dict[str, int],
    bool,
]:
    fingerprint = _historical_pair_cache_fingerprint(graph_path, claims_path)
    if cache_path.is_file():
        with gzip.open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("fingerprint") == fingerprint:
            return cached["pairs"], cached["endpoint_atoms"], cached["stats"], True
    pairs, endpoint_atoms, stats = _historical_pairs_from_graph(graph, claims_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(
            {
                "fingerprint": fingerprint,
                "pairs": pairs,
                "endpoint_atoms": endpoint_atoms,
                "stats": stats,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(cache_path)
    return pairs, endpoint_atoms, stats, False


def _future_records_cached(
    *,
    claims_path: Path,
    min_year: int,
    max_year: int,
    case_study_ids: frozenset[str],
    cache_path: Path,
) -> tuple[list[dict[str, Any]], bool]:
    fingerprint = {
        "claims_size": claims_path.stat().st_size,
        "claims_mtime_ns": claims_path.stat().st_mtime_ns,
        "min_year": min_year,
        "max_year": max_year,
        "case_study_ids": sorted(case_study_ids),
    }
    if cache_path.is_file():
        with gzip.open(cache_path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("fingerprint") == fingerprint:
            return cached["records"], True
    records = load_future_claim_records(
        claims_path,
        min_year=min_year,
        max_year=max_year,
        case_study_ids=case_study_ids,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(
            {"fingerprint": fingerprint, "records": records},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(cache_path)
    return records, False


def _metrics_rows(
    *,
    result: DynamicLoopResult,
    case_study_id: str,
    seed: int,
    freeze_year: int,
    future_start_year: int,
    future_end_year: int,
    budgets: Sequence[int],
    future_pair_total: int | None = None,
    method: str = "neurodiscovery_dynamic_closed_loop",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hidden = result.hidden
    closed_loop_activated = any(
        bool(row["feedback_active"]) for row in result.generation_rounds
    )
    supported_feedback_records = sum(
        row["status"] == SUPPORTED and row.get("available_to_generator", True)
        for row in result.feedback
    )
    withheld_supported_outcomes = sum(
        row["status"] == SUPPORTED and not row.get("available_to_generator", True)
        for row in result.feedback
    )
    for requested in budgets:
        selected = hidden[: min(int(requested), len(hidden))]
        unique_primary = {
            str(row.get("primary_discovery_key") or "")
            for row in selected
            if row.get("primary_hit") and row.get("primary_discovery_key")
        }
        early_unique_primary = {
            str(row.get("primary_discovery_key") or "")
            for row in selected
            if row.get("early_primary_hit") and row.get("primary_discovery_key")
        }
        terminal_unique_primary = {
            str(row.get("primary_discovery_key") or "")
            for row in selected
            if row.get("terminal_primary_hit") and row.get("primary_discovery_key")
        }
        recovered_pairs = {
            pair
            for row in selected
            if row.get("primary_hit")
            for pair in str(row.get("primary_recovered_pairs") or "").split(";")
            if pair
        }
        rows.append(
            {
                "method": method,
                "case_study_id": case_study_id,
                "seed": seed,
                "freeze_year": freeze_year,
                "future_start_year": future_start_year,
                "future_end_year": future_end_year,
                "requested_k": int(requested),
                "executed_hypotheses": len(selected),
                "generation_failure_slots": sum(
                    bool(row.get("generation_failure")) for row in selected
                ),
                "full_primary_hits": sum(bool(row["primary_hit"]) for row in selected),
                "unique_primary_discoveries": len(unique_primary),
                "recovered_future_pairs": len(recovered_pairs),
                "future_pair_recall": (
                    len(recovered_pairs) / future_pair_total
                    if future_pair_total is not None and future_pair_total > 0
                    else None
                ),
                "full_any_hits": sum(bool(row["any_future_hit"]) for row in selected),
                "early_primary_hits": sum(bool(row["early_primary_hit"]) for row in selected),
                "early_unique_primary_discoveries": len(early_unique_primary),
                "terminal_primary_hits": sum(bool(row["terminal_primary_hit"]) for row in selected),
                "terminal_unique_primary_discoveries": len(terminal_unique_primary),
                "terminal_any_hits": sum(bool(row["terminal_any_hit"]) for row in selected),
                "unique_proposals": len(result.proposed),
                "generation_rounds": len(result.generation_rounds),
                "supported_feedback_records": supported_feedback_records,
                "withheld_supported_outcomes": withheld_supported_outcomes,
                "closed_loop_activated": closed_loop_activated,
                "candidate_space_exhausted": result.stop_reason == "candidate_space_exhausted",
            }
        )
    return rows


def _write_run_outputs(
    run_dir: Path,
    *,
    result: DynamicLoopResult,
    metrics: list[dict[str, Any]],
    run_manifest: Mapping[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        run_dir / "proposed_hypotheses.jsonl.gz",
        "wt",
        encoding="utf-8",
        compresslevel=1,
    ) as handle:
        for hypothesis in result.proposed:
            handle.write(json.dumps(hypothesis, ensure_ascii=False) + "\n")
    _atomic_write_json(
        run_dir / "executed_hypotheses.json",
        {"n_hypotheses": len(result.executed), "hypotheses": result.executed},
    )
    _write_csv(run_dir / "hidden_outcomes.csv", result.hidden)
    _write_csv(run_dir / "feedback_overlay.csv", result.feedback)
    _write_csv(run_dir / "generation_rounds.csv", result.generation_rounds)
    _write_csv(run_dir / "metrics_by_k.csv", metrics)
    _atomic_write_json(run_dir / "run_manifest.json", run_manifest)


def _load_reusable_run(
    run_dir: Path,
    *,
    seed: int,
    case_study_id: str,
    freeze_year: int,
    future_start_year: int,
    future_end_year: int,
    config: DynamicLoopConfig,
    profile: LoopProfile,
    source_bundle: Mapping[str, Any] | None,
    budgets: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load only an exact, complete v4 checkpoint; reject all stale artifacts."""

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        if run_dir.is_dir() and any(run_dir.iterdir()):
            raise ValueError(
                f"partial dynamic run exists at {run_dir}; rerun with --force"
            )
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "schema_version": SCHEMA,
        "status": "complete",
        "seed": int(seed),
        "case_study_id": case_study_id,
        "freeze_year": int(freeze_year),
        "future_start_year": int(future_start_year),
        "future_end_year": int(future_end_year),
    }
    mismatches = [
        field
        for field, expected in expected_identity.items()
        if manifest.get(field) != expected
    ]
    if (
        mismatches
        or manifest.get("config") != asdict(config)
        or manifest.get("profile") != asdict(profile)
        or manifest.get("source_bundle") != source_bundle
        or int(manifest.get("executed_hypotheses") or 0) != config.max_executions
        or "generation_failure_slots" not in manifest
    ):
        raise ValueError(
            f"stale or incompatible dynamic run at {run_dir}; rerun with --force"
        )
    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"incomplete dynamic run at {run_dir}: missing={missing}; rerun with --force"
        )
    executed = json.loads(
        (run_dir / "executed_hypotheses.json").read_text(encoding="utf-8")
    )
    hypotheses = list(executed.get("hypotheses") or ())
    if (
        int(executed.get("n_hypotheses") or 0) != config.max_executions
        or len(hypotheses) != config.max_executions
    ):
        raise ValueError(
            f"dynamic checkpoint has the wrong execution budget at {run_dir}; "
            "rerun with --force"
        )
    metrics_path = run_dir / "metrics_by_k.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_metrics = [dict(row) for row in csv.DictReader(handle)]
    if (
        len(raw_metrics) != len(budgets)
        or [int(row["requested_k"]) for row in raw_metrics]
        != [int(value) for value in budgets]
        or any("generation_failure_slots" not in row for row in raw_metrics)
    ):
        raise ValueError(
            f"dynamic checkpoint has an incompatible metric matrix at {run_dir}; "
            "rerun with --force"
        )
    metrics = pd.read_csv(metrics_path).to_dict("records")
    return manifest, metrics


def _selected_profile(name: str) -> LoopProfile:
    profiles = {profile.name: profile for profile in profile_slate()}
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"unknown closed-loop profile: {name}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_bundle = _verified_source_bundle(
        getattr(args, "source_bundle_manifest", None),
        verify_references=not bool(
            getattr(args, "skip_source_bundle_reference_rehash", False)
        ),
    )
    config = DynamicLoopConfig(
        feedback_enabled=args.feedback_enabled,
        proposal_batch_size=args.proposal_batch_size,
        max_unique_proposals=args.max_unique_proposals,
        max_generation_rounds=args.max_generation_rounds,
        max_stagnant_generation_rounds=args.max_stagnant_generation_rounds,
        max_executions=args.max_executions,
        execution_batch_size=args.execution_batch_size,
        refresh_every_executions=args.refresh_every_executions,
        refill_below=args.refill_below,
        feedback_years=args.feedback_years,
        feedback_start_budget=args.feedback_start_budget,
        min_supported_before_feedback=args.min_supported_before_feedback,
        seed_diversity_fraction=args.seed_diversity_fraction,
        candidate_pool_mode=args.candidate_pool_mode,
        task_scope_fraction=args.task_scope_fraction,
        evidence_frontier_fraction=args.evidence_frontier_fraction,
        protect_general_top_k=args.protect_general_top_k,
        path_variants_per_endpoint=args.path_variants_per_endpoint,
        feedback_mutation_fraction=args.feedback_mutation_fraction,
        max_paths_per_endpoint=args.max_paths_per_endpoint,
        endpoint_canonical_quality_weight=(
            args.endpoint_canonical_quality_weight
        ),
        kge_weight=args.kge_weight,
    )
    config.validate()
    profile = _selected_profile(args.profile)
    eligibility = (
        load_locked_hindcasting_eligibility(args.eligibility_manifest)
        if args.eligibility_manifest is not None
        else None
    )
    case_ids = (
        args.case_study_ids
        or (
            list(eligibility.primary_case_study_ids)
            if eligibility is not None
            else list_case_study_names()
        )
    )
    cases = tuple(case_study_by_name(name) for name in case_ids)
    selected_primary_windows = (
        eligibility.selected_windows(
            case_study_ids=(case.name for case in cases),
            windows=args.windows,
        )
        if eligibility is not None
        else None
    )
    active_windows = tuple(
        window
        for window in args.windows
        if selected_primary_windows is None
        or any(
            key[1:] == (
                window.freeze_year,
                window.future_start_year,
                window.future_end_year,
            )
            for key in selected_primary_windows
        )
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_root = args.index_cache_root or (args.output_root / "_cache")
    min_future_year = min(window.future_start_year for window in active_windows)
    max_future_year = max(window.future_end_year for window in active_windows)
    task_set = frozenset(
        key[0] for key in selected_primary_windows
    ) if selected_primary_windows is not None else frozenset(case.name for case in cases)
    task_digest = hashlib.sha256(
        "\x1f".join(sorted(task_set)).encode("utf-8")
    ).hexdigest()[:12]
    print(
        f"[index] future claims {min_future_year}-{max_future_year} "
        f"for {len(task_set)} task(s)",
        flush=True,
    )
    future_records, future_records_cache_hit = _future_records_cached(
        claims_path=args.future_claims,
        min_year=min_future_year,
        max_year=max_future_year,
        case_study_ids=task_set,
        cache_path=(
            cache_root
            / f"future_records_{min_future_year}_{max_future_year}_{task_digest}.pkl.gz"
        ),
    )
    print(
        f"[index] future records={len(future_records):,} "
        f"cache_hit={future_records_cache_hit}",
        flush=True,
    )
    all_metrics: list[dict[str, Any]] = []
    completed_runs: list[dict[str, Any]] = []

    for window in active_windows:
        eligible_cases = tuple(
            case
            for case in cases
            if selected_primary_windows is None
            or (
                case.name,
                window.freeze_year,
                window.future_start_year,
                window.future_end_year,
            )
            in selected_primary_windows
        )
        if not eligible_cases:
            continue
        kge_scorer = None
        kge_checkpoint: Path | None = None
        if args.kge_root is not None and config.kge_weight > 0.0:
            kge_checkpoint = args.kge_root / args.kge_checkpoint_pattern.format(
                freeze_year=window.freeze_year
            )
            if not kge_checkpoint.is_file():
                raise FileNotFoundError(kge_checkpoint)
            print(f"[load] frozen KGE {kge_checkpoint.name}", flush=True)
            kge_scorer = ComplExScorer.load(kge_checkpoint, device=args.kge_device)
        pending: list[tuple[int, CaseStudy, Path]] = []
        for seed in args.seeds:
            for case in eligible_cases:
                run_dir = (
                    args.output_root
                    / f"seed_{seed:02d}"
                    / case.name
                    / window.label
                )
                if not args.force:
                    reusable = _load_reusable_run(
                        run_dir,
                        seed=seed,
                        case_study_id=case.name,
                        freeze_year=window.freeze_year,
                        future_start_year=window.future_start_year,
                        future_end_year=window.future_end_year,
                        config=config,
                        profile=profile,
                        source_bundle=source_bundle,
                        budgets=args.budgets,
                    )
                    if reusable is not None:
                        manifest, metrics = reusable
                        all_metrics.extend(metrics)
                        completed_runs.append(manifest)
                        continue
                pending.append((seed, case, run_dir))
        if not pending:
            continue

        graph_path = args.snapshot_root / f"kg_{window.freeze_year}" / "knowledge_graph.json"
        claims_path = graph_path.parent / "extracted_claims.jsonl"
        if not graph_path.is_file() or not claims_path.is_file():
            raise FileNotFoundError(f"incomplete frozen snapshot: {graph_path.parent}")
        print(f"[load] dynamic closed-loop KG_{window.freeze_year}", flush=True)
        graph = load_graph(graph_path)
        (
            historical_pairs,
            historical_endpoint_atoms,
            historical_stats,
            cache_hit,
        ) = _historical_pairs_cached(
            graph=graph,
            graph_path=graph_path,
            claims_path=claims_path,
            cache_path=(
                cache_root / f"kg_{window.freeze_year}_historical_pairs.pkl.gz"
            ),
        )
        future_by_task: dict[str, tuple[dict[str, Any], dict[str, int]]] = {}
        for run_index, (seed, case, run_dir) in enumerate(pending, start=1):
            if case.name not in future_by_task:
                future_by_task[case.name] = _future_indexes(
                    args.future_claims,
                    graph._index,
                    historical_pairs,
                    window.future_start_year,
                    window.future_end_year,
                    case_study_id=case.name,
                    future_records=future_records,
                    historical_endpoint_atoms=historical_endpoint_atoms,
                )
            future_index, future_stats = future_by_task[case.name]
            rounds_dir = run_dir / "round_work"
            # Reuse expensive branch indexes across dynamic rounds, but never
            # across seeds. This keeps paired replicates independent even if a
            # HypothesisEngine later gains another mutable lazy cache.
            branch_engines: dict[str, Any] = {}

            def generate_round(
                round_index: int,
                generation_seed: int,
                feedback_state: FeedbackState,
                excluded_semantic_keys: frozenset[
                    tuple[str, tuple[str, ...]]
                ],
            ) -> list[dict[str, Any]]:
                output = rounds_dir / f"round_{round_index:02d}" / "hypotheses.json"
                payload = generate_one(
                    graph=graph,
                    case=case,
                    output=output,
                    seed=generation_seed,
                    freeze_year=window.freeze_year,
                    target=config.proposal_batch_size,
                    fixed_budget=0,
                    force=True,
                    seed_diversity_fraction=config.seed_diversity_fraction,
                    candidate_pool_mode=config.candidate_pool_mode,
                    task_scope_fraction=config.task_scope_fraction,
                    evidence_frontier_fraction=config.evidence_frontier_fraction,
                    protect_general_top_k=config.protect_general_top_k,
                    static_score_family=(
                        "relation_aware"
                        if profile.score_family == "relation_aware"
                        else "legacy"
                    ),
                    feedback_state=feedback_state,
                    branch_engines=branch_engines,
                    dynamic_generation=True,
                    replicate_index=seed,
                    generation_round=round_index,
                    excluded_semantic_keys=excluded_semantic_keys,
                    path_variants_per_endpoint=config.path_variants_per_endpoint,
                    feedback_mutation_fraction=config.feedback_mutation_fraction,
                    max_paths_per_endpoint=config.max_paths_per_endpoint,
                    endpoint_canonical_quality_weight=(
                        config.endpoint_canonical_quality_weight
                    ),
                    kge_scorer=kge_scorer,
                    kge_weight=config.kge_weight,
                    kge_checkpoint=str(kge_checkpoint) if kge_checkpoint else None,
                )
                return list(payload.get("hypotheses") or [])

            def score_candidate(hypothesis: dict[str, Any]) -> dict[str, Any]:
                return _score_hypothesis(
                    hypothesis,
                    future_index,
                    window.freeze_year,
                    case_study_id=case.name,
                )

            generator_closure_freevars = _audit_generator_closure(generate_round)

            print(
                f"[dynamic {run_index}/{len(pending)}] {case.name} "
                f"seed={seed} KG_{window.freeze_year}",
                flush=True,
            )
            result = execute_dynamic_loop(
                case_study_id=case.name,
                seed=seed,
                freeze_year=window.freeze_year,
                future_start_year=window.future_start_year,
                future_end_year=window.future_end_year,
                config=config,
                profile=profile,
                generate_round=generate_round,
                score_candidate=score_candidate,
            )
            metrics = _metrics_rows(
                result=result,
                case_study_id=case.name,
                seed=seed,
                freeze_year=window.freeze_year,
                future_start_year=window.future_start_year,
                future_end_year=window.future_end_year,
                budgets=args.budgets,
                future_pair_total=int(future_stats.get("future_unique_pairs") or 0),
                method=(
                    "neurodiscovery_dynamic_closed_loop"
                    if config.feedback_enabled
                    else "neurodiscovery_dynamic_open_loop"
                ),
            )
            manifest = {
                "schema_version": SCHEMA,
                "discovery_metric_contract": DISCOVERY_METRIC_CONTRACT_VERSION,
                "status": "complete",
                "created_at": utc_now(),
                "case_study_id": case.name,
                "seed": seed,
                "freeze_year": window.freeze_year,
                "future_start_year": window.future_start_year,
                "future_end_year": window.future_end_year,
                "feedback_end_year": min(
                    window.future_end_year,
                    window.freeze_year + config.feedback_years,
                ),
                "profile": asdict(profile),
                "config": asdict(config),
                "feedback_start_budget": result.feedback_start_budget,
                "unique_proposals": len(result.proposed),
                "executed_hypotheses": len(result.executed),
                "generation_failure_slots": result.generation_failure_slots,
                "supported_feedback_records": sum(
                    row["status"] == SUPPORTED
                    and row.get("available_to_generator", True)
                    for row in result.feedback
                ),
                "withheld_supported_outcomes": sum(
                    row["status"] == SUPPORTED
                    and not row.get("available_to_generator", True)
                    for row in result.feedback
                ),
                "first_supported_execution_rank": next(
                    (
                        int(row["execution_rank"])
                        for row in result.feedback
                        if row["status"] == SUPPORTED
                        and row.get("available_to_generator", True)
                    ),
                    None,
                ),
                "closed_loop_activated": any(
                    bool(row["feedback_active"])
                    for row in result.generation_rounds
                ),
                "first_feedback_generation_round": next(
                    (
                        int(row["round"])
                        for row in result.generation_rounds
                        if bool(row["feedback_active"])
                    ),
                    None,
                ),
                "generation_rounds": len(result.generation_rounds),
                "stop_reason": result.stop_reason,
                "future_stats": future_stats,
                "historical_stats": historical_stats,
                "historical_cache_hit": cache_hit,
                "future_records_cache_hit": future_records_cache_hit,
                "generator_engines_reused_across_rounds": True,
                "generator_engines_reused_across_seeds": False,
                "generator_engine_scope": "seed_case_window",
                "source_bundle": source_bundle,
                "kge_prior": {
                    "enabled": kge_scorer is not None,
                    "checkpoint": (
                        str(kge_checkpoint) if kge_checkpoint else None
                    ),
                    "weight": config.kge_weight,
                    "device": args.kge_device,
                    "uses_future_outcomes": False,
                },
                "temporal_isolation": {
                    "generation_graph": f"KG_{window.freeze_year}",
                    "feedback_source": (
                        (
                            f"strict primary support in {window.future_start_year}-"
                            f"{min(window.future_end_year, window.freeze_year + config.feedback_years)}"
                        )
                        if config.feedback_enabled
                        else "withheld; open-loop control"
                    ),
                    "terminal_evaluation_source": (
                        f"{window.freeze_year + config.feedback_years + 1}-"
                        f"{window.future_end_year}"
                    ),
                    "terminal_labels_available_to_generator": False,
                    "generator_closure_audit_passed": True,
                    "generator_closure_freevars": list(generator_closure_freevars),
                    "generator_closure_forbidden_freevars": [],
                    "formal_kg_mutated": False,
                },
            }
            _write_run_outputs(
                run_dir,
                result=result,
                metrics=metrics,
                run_manifest=manifest,
            )
            if not args.retain_round_work and rounds_dir.is_dir():
                shutil.rmtree(rounds_dir)
            all_metrics.extend(metrics)
            completed_runs.append(manifest)

        del graph
        del historical_pairs, historical_endpoint_atoms
        del kge_scorer

    metrics_frame = pd.DataFrame(all_metrics)
    if not metrics_frame.empty:
        metrics_frame.to_csv(args.output_root / "metrics_by_run.csv", index=False)
        summary = (
            metrics_frame.groupby(
                ["method", "case_study_id", "requested_k"], as_index=False
            )[
                [
                    "executed_hypotheses",
                    "generation_failure_slots",
                    "full_primary_hits",
                    "unique_primary_discoveries",
                    "recovered_future_pairs",
                    "future_pair_recall",
                    "full_any_hits",
                    "early_primary_hits",
                    "early_unique_primary_discoveries",
                    "terminal_primary_hits",
                    "terminal_unique_primary_discoveries",
                    "terminal_any_hits",
                    "unique_proposals",
                    "generation_rounds",
                ]
            ]
            .agg(["mean", "std", "sum"])
        )
        summary.columns = [
            "_".join(column).rstrip("_") if isinstance(column, tuple) else str(column)
            for column in summary.columns
        ]
        summary.to_csv(args.output_root / "metrics_summary.csv", index=False)

    source_bundle_at_completion = _verified_source_bundle(
        getattr(args, "source_bundle_manifest", None),
        verify_references=not bool(
            getattr(args, "skip_source_bundle_reference_rehash", False)
        ),
    )
    if source_bundle_at_completion != source_bundle:
        raise ValueError("source bundle verification changed during the run")

    top_manifest = {
        "schema_version": SCHEMA,
        "discovery_metric_contract": DISCOVERY_METRIC_CONTRACT_VERSION,
        "status": "complete",
        "created_at": utc_now(),
        "snapshot_root": str(args.snapshot_root),
        "future_claims": str(args.future_claims),
        "output_root": str(args.output_root),
        "index_cache_root": str(cache_root),
        "case_studies": sorted({str(row["case_study_id"]) for row in completed_runs}),
        "windows": [
            {
                "freeze_year": freeze,
                "future_start_year": start,
                "future_end_year": end,
            }
            for freeze, start, end in sorted(
                {
                    (
                        int(row["freeze_year"]),
                        int(row["future_start_year"]),
                        int(row["future_end_year"]),
                    )
                    for row in completed_runs
                }
            )
        ],
        "seeds": list(args.seeds),
        "budgets": list(args.budgets),
        "profile": asdict(profile),
        "config": asdict(config),
        "kge_root": str(args.kge_root) if args.kge_root else None,
        "kge_checkpoint_pattern": args.kge_checkpoint_pattern,
        "kge_device": args.kge_device,
        "source_bundle": source_bundle_at_completion,
        "eligibility": (
            {
                "manifest_path": str(eligibility.manifest_path),
                "manifest_sha256": sha256_file(eligibility.manifest_path),
                "matrix_path": str(eligibility.matrix_path),
                "matrix_sha256": eligibility.matrix_sha256,
                "analysis_tier": "primary",
                "selected_case_windows": len(selected_primary_windows or ()),
            }
            if eligibility is not None
            else None
        ),
        "runs": len(completed_runs),
        "runs_with_supported_feedback": sum(
            int(manifest.get("supported_feedback_records", 0)) > 0
            for manifest in completed_runs
        ),
        "closed_loop_activated_runs": sum(
            bool(manifest.get("closed_loop_activated"))
            for manifest in completed_runs
        ),
        "run_summaries": completed_runs,
    }
    _atomic_write_json(args.output_root / "dynamic_closed_loop_manifest.json", top_manifest)
    return top_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=(
            ROOT
            / "neurooracle/data/experiments/hindcasting/"
            "snapshots_full_v2_endpoint_v3"
        ),
    )
    parser.add_argument(
        "--index-cache-root",
        type=Path,
        default=None,
        help="Optional shared cache for compact future records and historical pairs.",
    )
    parser.add_argument(
        "--future-claims",
        type=Path,
        default=ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "neurooracle/data/experiments/hindcasting/"
            "neurodiscovery_dynamic_closed_loop_current"
        ),
    )
    parser.add_argument(
        "--case-study-ids",
        nargs="*",
        choices=list_case_study_names(),
        default=None,
    )
    parser.add_argument(
        "--eligibility-manifest",
        type=Path,
        default=None,
        help=(
            "Optional immutable method-blind eligibility lock; only its primary "
            "Case Study/window rows are run."
        ),
    )
    parser.add_argument(
        "--source-bundle-manifest",
        type=Path,
        default=None,
        help=(
            "Optional immutable source-bundle manifest. Every live source, "
            "archived copy, and referenced input is hash-verified before a run."
        ),
    )
    parser.add_argument(
        "--skip-source-bundle-reference-rehash",
        action="store_true",
        help=(
            "Check reference existence and sizes without re-hashing them; use "
            "only when an outer formal runner already performed the deep check."
        ),
    )
    parser.add_argument("--windows", nargs="*", type=parse_window, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--profile", default="legacy_closed_supported_gate")
    parser.add_argument(
        "--feedback-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose early-window outcomes to generation (disable for open-loop control).",
    )
    parser.add_argument("--proposal-batch-size", type=int, default=3000)
    parser.add_argument("--max-unique-proposals", type=int, default=50000)
    parser.add_argument("--max-generation-rounds", type=int, default=20)
    parser.add_argument("--max-stagnant-generation-rounds", type=int, default=2)
    parser.add_argument("--max-executions", type=int, default=10000)
    parser.add_argument("--execution-batch-size", type=int, default=10)
    parser.add_argument("--refresh-every-executions", type=int, default=100)
    parser.add_argument("--refill-below", type=int, default=500)
    parser.add_argument("--feedback-years", type=int, default=2)
    parser.add_argument("--feedback-start-budget", type=int, default=50)
    parser.add_argument("--min-supported-before-feedback", type=int, default=2)
    parser.add_argument("--seed-diversity-fraction", type=float, default=0.35)
    parser.add_argument("--path-variants-per-endpoint", type=int, default=2)
    parser.add_argument("--feedback-mutation-fraction", type=float, default=0.35)
    parser.add_argument("--max-paths-per-endpoint", type=int, default=4)
    parser.add_argument(
        "--endpoint-canonical-quality-weight",
        type=float,
        default=NEURODISCOVERY_ENDPOINT_QUALITY_WEIGHT,
        help=(
            "Outcome-blind weight assigned to frozen endpoint canonicalization "
            "quality during proposal generation."
        ),
    )
    parser.add_argument(
        "--kge-root",
        type=Path,
        default=None,
        help="Directory containing one frozen ComplEx checkpoint per freeze year.",
    )
    parser.add_argument(
        "--kge-checkpoint-pattern",
        default="kg_{freeze_year}_complex_dim64_ep10.pt",
    )
    parser.add_argument(
        "--kge-weight",
        type=float,
        default=0.0,
        help=(
            "Weak outcome-blind structural prior blended into NeuroDiscovery; "
            f"the screened development default is {NEURODISCOVERY_KGE_WEIGHT:.2f}."
        ),
    )
    parser.add_argument("--kge-device", default="cuda")
    parser.add_argument(
        "--candidate-pool-mode",
        choices=("hybrid", "general", "scoped"),
        default="hybrid",
    )
    parser.add_argument("--task-scope-fraction", type=float, default=0.15)
    parser.add_argument(
        "--evidence-frontier-fraction",
        type=float,
        default=0.0,
        help=(
            "Global fraction of each task-scoped proposal batch reserved for "
            "independently grounded, historically absent endpoint relations."
        ),
    )
    parser.add_argument("--protect-general-top-k", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retain-round-work", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 <= args.endpoint_canonical_quality_weight <= 1.0:
        parser.error("--endpoint-canonical-quality-weight must be in [0, 1]")
    return args


def main() -> None:
    manifest = run(parse_args())
    print(
        json.dumps(
            {
                "output_root": manifest["output_root"],
                "runs": manifest["runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


# Updated: 2026-08-12 14:50:00 HKT - add a manifest-tracked evidence-frontier mixture to the dynamic loop.
# Updated: 2026-08-12 20:34 HKT - project closed-loop support onto the directed task input rather than a shared output.
# Updated: 2026-08-12 21:02 HKT - enforce feedback-mutation fraction as a cumulative execution quota.
# Updated: 2026-08-12 22:28 HKT - load freeze-specific KGE checkpoints for outcome-blind candidate retrieval.
# Updated: 2026-08-13 01:38 HKT - preserve replicate identity through dynamic frozen-KGE exploration.
# Updated: 2026-08-13 03:42 HKT - execute candidates with the generator's final outcome-blind composite score when requested.
# Updated: 2026-08-13 05:42:01 HKT - invalidate historical-pair caches when semantic endpoint identity changes.
# Updated: 2026-08-13 06:22:29 HKT - cache frozen endpoint atoms and enforce them in dynamic future truth.
