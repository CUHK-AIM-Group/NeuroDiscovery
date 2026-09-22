"""Run a direct, blinded, closed-loop comparison of CS1 research controllers.

Unlike the legacy benchmark, this runner never expands a sparse framework
proposal into a full candidate ranking. Every requested slot is an explicit
experiment selected from a shared outcome-blind menu, and failed slots remain
part of the experimental cost.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.scripts.canonical_kg_release import validate_canonical_kg_release
from core.scripts.case1_autoresearch_controllers import (
    FRAMEWORK_METHODS,
    ControllerSelection,
    make_framework_controller,
)
from core.scripts.case1_autoresearch_protocol import (
    HIDDEN_OUTCOME_FIELDS,
    Case1AutoresearchProtocol,
    blinded_public_registry,
    build_round_research_goal,
    classify_outcome,
    evaluate_prefix,
    menu_indices,
    render_menu_tsv,
    reveal_feedback,
    sha256_bytes,
    sha256_file,
    validate_selected_ids,
    write_selection_commitment,
)
from core.scripts.case1_method_comparison import (
    add_generator_scores,
    kg_query_terms_for_candidates,
    load_kg_index,
    load_results,
)
from core.scripts.case_study_closed_loop_engine import (
    ExperimentalOverlayGraph,
    ScoreComponentConfig,
    _select_diverse_batch,
    compose_static_score,
)
from core.scripts.case_study_feedback_adapters import adapter_for
from core.scripts.case_study_score_components import load_score_component_bundle


ROOT = Path(__file__).resolve().parents[2]
CASE_STUDY_ID = "case1_transdiagnostic"
NEURODISCOVERY = "neurodiscovery"
ALL_METHODS = (NEURODISCOVERY, *FRAMEWORK_METHODS)
DEFAULT_ALL_TESTS = Path(
    r"\\192.168.3.61\data\Public Dataset\case1_exhaustive_full"
    r"\20260616_full_main_noboot\case1_exhaustive_full_all_tests_labeled.csv"
)
DEFAULT_KG = ROOT / "neurooracle/data/full_v2/knowledge_graph.json"
DEFAULT_CLAIMS = ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl"
DEFAULT_STATE = ROOT / "neurooracle/data/full_v2/CURRENT_STATE.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_completed_seed_artifacts(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    protocol: Case1AutoresearchProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """Reuse a fully committed seed run without overwriting its artifacts.

    A missing seed manifest means that the job is incomplete and may be rerun.
    Once a manifest exists, however, every aggregate and per-round commitment
    must agree with the requested protocol. Inconsistent completed artifacts
    stop the resume instead of being silently replaced.
    """

    seed_dir = output_dir / method / f"seed_{seed:02d}"
    manifest_path = seed_dir / "seed_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completed seed manifest: {manifest_path}") from exc

    identity_ok = (
        manifest.get("schema_version") == "case1-autoresearch-seed-run.v1"
        and manifest.get("method") == method
        and manifest.get("seed") == seed
        and manifest.get("protocol") == protocol.to_dict()
        and manifest.get("requested_slots") == protocol.experiment_slots
    )
    if not identity_ok:
        raise RuntimeError(
            f"completed seed manifest does not match the requested run: {manifest_path}"
        )

    slots_path = seed_dir / "slots.csv"
    curve_path = seed_dir / "direct_curve.csv"
    try:
        slots = pd.read_csv(slots_path)
        curve = pd.read_csv(curve_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"completed seed aggregates are missing or unreadable: {seed_dir}"
        ) from exc

    required_slot_columns = {
        "method",
        "seed",
        "round",
        "slot",
        "valid",
        "execution_succeeded",
        "is_gt_top",
    }
    expected_budgets = list(
        range(protocol.batch_size, protocol.experiment_slots + 1, protocol.batch_size)
    )
    aggregates_ok = (
        len(slots) == protocol.experiment_slots
        and required_slot_columns.issubset(slots.columns)
        and set(slots["method"].astype(str)) == {method}
        and set(pd.to_numeric(slots["seed"], errors="coerce")) == {seed}
        and len(curve) == protocol.rounds
        and {"method", "seed", "budget"}.issubset(curve.columns)
        and set(curve["method"].astype(str)) == {method}
        and set(pd.to_numeric(curve["seed"], errors="coerce")) == {seed}
        and pd.to_numeric(curve["budget"], errors="coerce").tolist()
        == expected_budgets
    )
    if not aggregates_ok:
        raise RuntimeError(f"completed seed aggregates failed validation: {seed_dir}")

    for round_index in range(protocol.rounds):
        round_dir = seed_dir / f"round_{round_index + 1:02d}"
        commitment_path = round_dir / "selection_commitment.json"
        audit_path = round_dir / "round_audit.json"
        feedback_path = round_dir / "experimental_feedback.json"
        try:
            commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            json.loads(feedback_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"completed seed round artifacts are missing or invalid: {round_dir}"
            ) from exc
        round_ok = (
            commitment.get("method") == method
            and commitment.get("seed") == seed
            and commitment.get("round") == round_index
            and commitment.get("requested_slots") == protocol.batch_size
            and commitment.get("outcomes_read_before_commit") is False
            and audit.get("selection_commitment_sha256")
            == commitment.get("commit_sha256")
            and audit.get("menu_sha256") == commitment.get("menu_sha256")
            and audit.get("outcomes_revealed_after_commitment") is True
        )
        if not round_ok:
            raise RuntimeError(
                f"completed seed round commitment failed validation: {round_dir}"
            )

    return slots, curve, manifest


def outcome_blind_scoring_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    forbidden = set(HIDDEN_OUTCOME_FIELDS) | {
        "is_gt_top",
        "is_strict_fdr",
        "gt_rank",
        "adjusted_residual_d",
        "abs_adjusted_residual_d",
        "p_value",
        "q_fdr_global",
        "q_fdr_disease",
        "q_fdr_modality",
        "direction",
        "execution_succeeded",
    }
    columns = [
        column
        for column in candidates.columns
        if column not in forbidden
        and not str(column).startswith(("score_", "kg_"))
    ]
    public = candidates.loc[:, columns].copy()
    if any(column in public for column in forbidden):
        raise AssertionError("outcome-blind scoring frame leaked an outcome column")
    return public


class OutcomeVault:
    """Keep hidden outcomes behind a commitment-file assertion."""

    def __init__(self, candidates: pd.DataFrame) -> None:
        if candidates["candidate_id"].astype(str).duplicated().any():
            raise ValueError("candidate outcomes contain duplicate candidate IDs")
        self._rows = candidates.assign(
            candidate_id=candidates["candidate_id"].astype(str)
        ).set_index("candidate_id", drop=False)

    @property
    def candidate_count(self) -> int:
        return int(len(self._rows))

    @property
    def gt_total(self) -> int:
        return int(self._rows["is_gt_top"].fillna(False).astype(bool).sum())

    def reveal(
        self,
        candidate_ids: Sequence[str],
        *,
        commitment_path: Path,
    ) -> dict[str, dict[str, Any]]:
        if not commitment_path.is_file():
            raise RuntimeError("outcomes cannot be read before selection commitment")
        commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
        if commitment.get("outcomes_read_before_commit") is not False:
            raise RuntimeError("invalid selection commitment guard")
        committed = [str(value) for value in commitment["selected_candidate_ids"]]
        requested = [str(value) for value in candidate_ids]
        if requested != committed:
            raise RuntimeError("outcome request does not match committed candidates")
        return {
            candidate_id: self._rows.loc[candidate_id].to_dict()
            for candidate_id in requested
        }


class NeuroDiscoveryController:
    """Outcome-blind KG/KGE/critic ranker with a per-seed experimental overlay."""

    def __init__(
        self,
        *,
        scored_public: pd.DataFrame,
        static_score: np.ndarray,
        static_audit: Mapping[str, Any],
        seed: int,
        seed_dir: Path,
    ) -> None:
        self.method = NEURODISCOVERY
        self.public = scored_public.reset_index(drop=True)
        self.static_score = np.asarray(static_score, dtype=float)
        self.static_audit = dict(static_audit)
        self.seed = int(seed)
        self.seed_dir = seed_dir
        self.rng = np.random.default_rng(31_337 + self.seed)
        self.id_to_index = {
            value: index
            for index, value in enumerate(self.public["candidate_id"].astype(str))
        }
        overlay_public = self.public.copy()
        overlay_public["atlas"] = overlay_public["source"].astype(str)
        overlay_public["anatomy"] = overlay_public["anatomy_full"].astype(str)
        self.overlay_path = seed_dir / "experimental_kg_overlay.jsonl.gz"
        self.overlay = ExperimentalOverlayGraph(
            overlay_public,
            adapter=adapter_for(CASE_STUDY_ID),
            factor_fields=(
                "disease",
                "feature_family",
                "map_group",
                "roi_key",
                "source",
            ),
            seed=self.seed,
            trial=self.seed,
            prior_alpha=1.0,
            prior_beta=3.0,
            feedback_projection="all_factors",
            feedback_model="beta_counts",
            stream_path=self.overlay_path,
        )

    def select(
        self,
        *,
        menu_candidate_indices: np.ndarray,
        start_rank: int,
        requested_slots: int,
        round_index: int,
    ) -> ControllerSelection:
        feedback_score, feedback_audit = self.overlay.score_candidates(
            feedback_weight=0.12,
            pair_feedback_weight=0.08,
            exploration_weight=0.015,
            inconclusive_failure_weight=0.20,
        )
        combined = self.static_score + feedback_score
        tie_break = self.rng.random(len(combined)) * 1e-10
        combined = combined + tie_break
        static_counterfactual = _select_diverse_batch(
            np.asarray(menu_candidate_indices, dtype=np.int64),
            self.static_score + tie_break,
            self.public,
            factor_fields=("disease", "feature_family", "source"),
            batch_size=requested_slots,
            diversity_penalty=0.15,
        )
        selected = _select_diverse_batch(
            np.asarray(menu_candidate_indices, dtype=np.int64),
            combined,
            self.public,
            factor_fields=("disease", "feature_family", "source"),
            batch_size=requested_slots,
            diversity_penalty=0.15,
        )
        selection_changed = not np.array_equal(selected, static_counterfactual)
        self.overlay.note_selection_change(selection_changed)
        counterfactual_overlap = len(
            set(np.asarray(selected, dtype=np.int64).tolist())
            & set(np.asarray(static_counterfactual, dtype=np.int64).tolist())
        )
        items: list[dict[str, Any]] = []
        for offset, index in enumerate(selected):
            row = self.public.iloc[int(index)]
            items.append(
                {
                    "rank": start_rank + offset,
                    "candidate_id": str(row["candidate_id"]),
                    "rationale": (
                        "Selected by frozen KG/KGE/novelty/critic evidence and the "
                        "per-seed experimental overlay under diversity constraints."
                    ),
                    "confidence": float(np.clip(combined[int(index)], 0.0, 1.0)),
                }
            )
        return ControllerSelection(
            items=tuple(items),
            metadata={
                "controller": "neurodiscovery_direct_closed_loop",
                "round": int(round_index),
                "static_components": self.static_audit,
                "feedback": feedback_audit,
                "selection_changed_from_static": selection_changed,
                "static_counterfactual_overlap": int(counterfactual_overlap),
                "diversity_fields": ["disease", "feature_family", "source"],
                "implicit_ranking_expansion": False,
            },
        )

    def observe(
        self,
        outcomes: Mapping[str, Mapping[str, Any]],
        *,
        round_index: int,
        protocol: Case1AutoresearchProtocol,
    ) -> None:
        for candidate_id, outcome in outcomes.items():
            status = classify_outcome(outcome, protocol)
            self.overlay.append(
                candidate_index=self.id_to_index[str(candidate_id)],
                outcome={
                    "feedback_status": status,
                    "validated": status == "supported",
                    "execution_succeeded": bool(
                        outcome.get("execution_succeeded", False)
                    ),
                    "adjusted_residual_d": outcome.get("adjusted_residual_d"),
                    "p_value": outcome.get("p_value"),
                },
                round_index=round_index,
            )

    def close(self) -> dict[str, Any]:
        manifest = self.overlay.write(self.overlay_path)
        write_json(self.seed_dir / "experimental_kg_overlay.manifest.json", manifest)
        return manifest


def build_task(
    *,
    method: str,
    seed: int,
    round_index: int,
    prompt: str,
    menu_path: Path,
    protocol: Case1AutoresearchProtocol,
    prior_native_summaries: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "case1-autoresearch-native-task.v1",
        "case_study_id": CASE_STUDY_ID,
        "trial": int(seed),
        "seed": int(seed),
        "round_index": int(round_index),
        "research_goal": prompt,
        "public_registry_path": str(menu_path.resolve()),
        "candidate_id_template": "modality|source|disease|feature|roi_index",
        "candidate_id_fields": [
            "modality",
            "source",
            "disease",
            "feature",
            "roi_index",
        ],
        "n_anchors": int(protocol.batch_size),
        "policy_mode": "anchors",
        "policy_schema_version": "case1-search-policy-v1",
        "open_coscientist_max_iterations": int(
            protocol.open_coscientist_iterations
        ),
        "open_coscientist_overgeneration_factor": float(
            protocol.open_coscientist_overgeneration_factor
        ),
        "open_coscientist_tool_generation_enabled": bool(
            protocol.open_coscientist_tool_generation_enabled
        ),
        "open_coscientist_debate_cohorts_enabled": bool(
            protocol.open_coscientist_debate_cohorts_enabled
        ),
        "previous_round_summaries": [
            str(value)[-6000:] for value in prior_native_summaries[-2:]
        ],
        "round_contexts": [],
        "native_retrieval_enabled": bool(protocol.native_retrieval_enabled),
        "benchmark_method": method,
    }


def slot_rows_for_round(
    *,
    method: str,
    seed: int,
    round_index: int,
    start_rank: int,
    validation_rows: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, Any]],
    protocol: Case1AutoresearchProtocol,
    controller_error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, audit in enumerate(validation_rows):
        valid = bool(audit.get("valid"))
        candidate_id = str(audit.get("candidate_id") or "")
        outcome = outcomes.get(candidate_id, {}) if valid else {}
        succeeded = bool(outcome.get("execution_succeeded", False)) if valid else False
        status = classify_outcome(outcome, protocol) if valid else "not_executed"
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "round": int(round_index),
                "slot": int(start_rank + offset),
                "rank": int(audit["rank"]),
                "candidate_id": candidate_id,
                "valid": valid,
                "validation_status": str(audit.get("status") or ""),
                "execution_succeeded": succeeded,
                "feedback_status": status,
                "supported": status == "supported",
                "is_gt_top": bool(outcome.get("is_gt_top", False)) if valid else False,
                "adjusted_residual_d": (
                    outcome.get("adjusted_residual_d") if succeeded else None
                ),
                "p_value": outcome.get("p_value") if succeeded else None,
                "controller_error": controller_error,
            }
        )
    return rows


def run_seed_method(
    *,
    method: str,
    seed: int,
    protocol: Case1AutoresearchProtocol,
    menus: Sequence[np.ndarray],
    menu_public: pd.DataFrame,
    scored_public: pd.DataFrame,
    static_score: np.ndarray,
    static_audit: Mapping[str, Any],
    vault: OutcomeVault,
    output_dir: Path,
    secret: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seed_dir = output_dir / method / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prior_feedback: list[dict[str, Any]] = []
    prior_native_summaries: list[str] = []
    previously_executed: set[str] = set()
    slot_rows: list[dict[str, Any]] = []

    if method == NEURODISCOVERY:
        controller: Any = NeuroDiscoveryController(
            scored_public=scored_public,
            static_score=static_score,
            static_audit=static_audit,
            seed=seed,
            seed_dir=seed_dir,
        )
    else:
        controller = make_framework_controller(
            method,
            secret=secret,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            workflow_timeout_seconds=args.workflow_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            max_retries=args.max_retries,
            native_retrieval_enabled=protocol.native_retrieval_enabled,
            brainpilot_url=args.brainpilot_url,
            seed_dir=seed_dir,
        )

    controller_metadata: list[dict[str, Any]] = []
    overlay_manifest: dict[str, Any] | None = None
    try:
        for round_index, indices in enumerate(menus):
            round_dir = seed_dir / f"round_{round_index + 1:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            menu = menu_public.iloc[indices].reset_index(drop=True)
            menu_text = render_menu_tsv(menu)
            menu_hash = sha256_bytes(menu_text.encode("utf-8"))
            menu_path = round_dir / "public_menu.csv"
            menu.to_csv(menu_path, index=False)
            (round_dir / "public_menu.tsv").write_text(
                menu_text + "\n", encoding="utf-8"
            )
            start_rank = round_index * protocol.batch_size + 1
            prompt = build_round_research_goal(
                method=method,
                seed=seed,
                round_index=round_index,
                menu=menu,
                prior_feedback=prior_feedback,
                protocol=protocol,
            )
            if prior_native_summaries:
                prompt += (
                    "\nPRIOR FRAMEWORK WORKING SUMMARY\n"
                    + prior_native_summaries[-1][-6000:]
                    + "\nUse it only as your own process memory; current menu IDs remain binding.\n"
                )
            (round_dir / "research_goal.txt").write_text(prompt, encoding="utf-8")
            task = build_task(
                method=method,
                seed=seed,
                round_index=round_index,
                prompt=prompt,
                menu_path=menu_path,
                protocol=protocol,
                prior_native_summaries=prior_native_summaries,
            )

            controller_error = ""
            try:
                if method == NEURODISCOVERY:
                    selection = controller.select(
                        menu_candidate_indices=indices,
                        start_rank=start_rank,
                        requested_slots=protocol.batch_size,
                        round_index=round_index,
                    )
                    write_json(round_dir / "task.json", task)
                    write_json(
                        round_dir / "native_result.json",
                        {
                            "method": method,
                            "hypotheses": list(selection.items),
                            "metadata": dict(selection.metadata),
                        },
                    )
                else:
                    selection = controller.select(
                        task=task,
                        round_dir=round_dir,
                        prompt=prompt,
                        start_rank=start_rank,
                    )
            except Exception as exc:
                controller_error = f"{type(exc).__name__}: {exc}"
                (round_dir / "controller_error.txt").write_text(
                    controller_error + "\n" + traceback.format_exc(),
                    encoding="utf-8",
                )
                selection = ControllerSelection(
                    items=(),
                    metadata={"controller_error": controller_error},
                )

            controller_metadata.append(dict(selection.metadata))
            selected_ids, validation = validate_selected_ids(
                selection.items,
                menu_candidate_ids=set(menu["candidate_id"].astype(str)),
                previously_executed=previously_executed,
                start_rank=start_rank,
                requested_slots=protocol.batch_size,
            )
            commitment_path = round_dir / "selection_commitment.json"
            commitment = write_selection_commitment(
                commitment_path,
                method=method,
                seed=seed,
                round_index=round_index,
                requested_slots=protocol.batch_size,
                selected_candidate_ids=selected_ids,
                validation_rows=validation,
                menu_sha256=menu_hash,
            )
            outcomes = vault.reveal(selected_ids, commitment_path=commitment_path)
            feedback = reveal_feedback(
                selected_ids,
                outcomes,
                round_index=round_index,
                protocol=protocol,
            )
            write_json(round_dir / "experimental_feedback.json", feedback)
            write_json(
                round_dir / "round_audit.json",
                {
                    "selection_commitment_sha256": commitment["commit_sha256"],
                    "menu_sha256": menu_hash,
                    "controller_metadata": dict(selection.metadata),
                    "controller_error": controller_error,
                    "valid_selections": len(selected_ids),
                    "requested_slots": protocol.batch_size,
                    "implicit_ranking_expansion": False,
                    "outcomes_revealed_after_commitment": True,
                },
            )
            if method == NEURODISCOVERY:
                controller.observe(
                    outcomes,
                    round_index=round_index,
                    protocol=protocol,
                )
            slot_rows.extend(
                slot_rows_for_round(
                    method=method,
                    seed=seed,
                    round_index=round_index,
                    start_rank=start_rank,
                    validation_rows=validation,
                    outcomes=outcomes,
                    protocol=protocol,
                    controller_error=controller_error,
                )
            )
            previously_executed.update(selected_ids)
            prior_feedback.extend(feedback)
            if selection.native_summary:
                prior_native_summaries.append(selection.native_summary)
    finally:
        if method == NEURODISCOVERY:
            overlay_manifest = controller.close()
        else:
            controller.close()

    slots = pd.DataFrame(slot_rows)
    slots.to_csv(seed_dir / "slots.csv", index=False)
    budgets = list(range(protocol.batch_size, protocol.experiment_slots + 1, protocol.batch_size))
    curve = evaluate_prefix(slots, requested_budgets=budgets)
    curve["gt_total"] = vault.gt_total
    curve["gt_recall"] = (
        curve["gt_hits"] / vault.gt_total if vault.gt_total else 0.0
    )
    curve.insert(0, "seed", seed)
    curve.insert(0, "method", method)
    curve.to_csv(seed_dir / "direct_curve.csv", index=False)
    seed_manifest = {
        "schema_version": "case1-autoresearch-seed-run.v1",
        "created_at": utc_now(),
        "method": method,
        "seed": seed,
        "protocol": protocol.to_dict(),
        "requested_slots": protocol.experiment_slots,
        "valid_hypotheses": int(slots["valid"].sum()),
        "execution_succeeded": int(slots["execution_succeeded"].sum()),
        "gt_hits": int(slots["is_gt_top"].sum()),
        "gt_total": vault.gt_total,
        "gt_recall": (
            float(slots["is_gt_top"].sum()) / vault.gt_total
            if vault.gt_total
            else 0.0
        ),
        "supported_hits": int(slots["supported"].sum()),
        "controller_metadata_by_round": controller_metadata,
        "experimental_overlay": overlay_manifest,
        "shared_heuristic_expansion": False,
    }
    write_json(seed_dir / "seed_manifest.json", seed_manifest)
    return slots, curve, seed_manifest


def aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in curves.columns
        if column not in {"method", "seed", "budget"}
    ]
    grouped = curves.groupby(["method", "budget"], sort=False)
    rows: list[dict[str, Any]] = []
    for (method, budget), group in grouped:
        row: dict[str, Any] = {
            "method": method,
            "budget": int(budget),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_variance"] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_selection_overlap(slots: pd.DataFrame) -> pd.DataFrame:
    """Measure whether equal hit counts came from the same selected experiments."""

    rows: list[dict[str, Any]] = []
    for seed, seed_rows in slots.groupby("seed", sort=True):
        method_order = list(dict.fromkeys(seed_rows["method"].astype(str)))
        selected: dict[str, set[str]] = {}
        for method in method_order:
            method_rows = seed_rows.loc[seed_rows["method"].astype(str) == method]
            valid_values = method_rows["valid"]
            if pd.api.types.is_bool_dtype(valid_values):
                valid_mask = valid_values.fillna(False)
            else:
                valid_mask = (
                    valid_values.fillna("").astype(str).str.casefold().isin({"true", "1"})
                )
            valid_rows = method_rows.loc[valid_mask]
            selected[method] = {
                candidate_id
                for candidate_id in valid_rows["candidate_id"].astype(str)
                if candidate_id
            }
        for method_a, method_b in combinations(method_order, 2):
            ids_a = selected[method_a]
            ids_b = selected[method_b]
            intersection = len(ids_a & ids_b)
            union = len(ids_a | ids_b)
            smaller = min(len(ids_a), len(ids_b))
            rows.append(
                {
                    "seed": int(seed),
                    "method_a": method_a,
                    "method_b": method_b,
                    "selected_a": len(ids_a),
                    "selected_b": len(ids_b),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 0.0,
                    "overlap_fraction_smaller": (
                        intersection / smaller if smaller else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_pairwise_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    if overlap.empty:
        return overlap.copy()
    rows: list[dict[str, Any]] = []
    metrics = ("intersection", "jaccard", "overlap_fraction_smaller")
    for (method_a, method_b), group in overlap.groupby(
        ["method_a", "method_b"], sort=False
    ):
        row: dict[str, Any] = {
            "method_a": method_a,
            "method_b": method_b,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_variance"] = (
                float(values.var(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-tests", type=Path, default=DEFAULT_ALL_TESTS)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--current-state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--score-table", type=Path)
    parser.add_argument("--score-manifest", type=Path)
    parser.add_argument("--require-score-components", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--menu-multiplier", type=int, default=20)
    parser.add_argument("--support-alpha", type=float, default=0.01)
    parser.add_argument("--support-min-abs-d", type=float, default=0.15)
    parser.add_argument("--gt-top-fraction", type=float, default=0.01)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--brainpilot-url", default="http://127.0.0.1:9460/api")
    parser.add_argument("--workflow-timeout-seconds", type=int, default=3600)
    parser.add_argument("--request-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--open-coscientist-iterations", type=int, default=1)
    parser.add_argument(
        "--open-coscientist-overgeneration-factor", type=float, default=1.0
    )
    parser.add_argument("--open-coscientist-tool-generation", action="store_true")
    parser.add_argument(
        "--disable-open-coscientist-debate-cohorts",
        action="store_true",
        help=(
            "Disable the outcome-blind disjoint menu cohorts used to keep parallel "
            "Open Co-Scientist debates from proposing the same candidate."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Run independent method/seed jobs concurrently; defaults to serial execution.",
    )
    parser.add_argument("--disable-native-retrieval", action="store_true")
    parser.add_argument("--allow-relocated-kg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(dict.fromkeys(args.methods))
    seeds = tuple(dict.fromkeys(args.seeds))
    if args.parallel_workers < 1:
        raise ValueError("--parallel-workers must be positive")
    protocol = Case1AutoresearchProtocol(
        rounds=args.rounds,
        batch_size=args.batch_size,
        menu_multiplier=args.menu_multiplier,
        support_alpha=args.support_alpha,
        support_min_abs_d=args.support_min_abs_d,
        gt_top_fraction=args.gt_top_fraction,
        open_coscientist_iterations=args.open_coscientist_iterations,
        open_coscientist_overgeneration_factor=(
            args.open_coscientist_overgeneration_factor
        ),
        open_coscientist_tool_generation_enabled=(
            args.open_coscientist_tool_generation
        ),
        open_coscientist_debate_cohorts_enabled=(
            not args.disable_open_coscientist_debate_cohorts
        ),
        native_retrieval_enabled=not args.disable_native_retrieval,
    )
    release = validate_canonical_kg_release(
        kg_path=args.kg,
        claims_path=args.claims,
        state_path=args.current_state,
        case_study_id=CASE_STUDY_ID,
        allow_relocated_artifacts=args.allow_relocated_kg,
    )
    candidates = load_results(args.all_tests, protocol.gt_top_fraction)
    protocol.validate(candidate_count=len(candidates))
    vault = OutcomeVault(candidates)

    public_input = outcome_blind_scoring_frame(candidates)
    kg = load_kg_index(args.kg, kg_query_terms_for_candidates(public_input))
    scored_public = add_generator_scores(public_input, kg, seed=protocol.menu_seed_base)
    component_audit: dict[str, Any] = {
        "active": False,
        "reason": "score bundle not supplied",
    }
    if bool(args.score_table) != bool(args.score_manifest):
        raise ValueError("--score-table and --score-manifest must be supplied together")
    if args.score_table and args.score_manifest:
        scored_public, component_audit = load_score_component_bundle(
            scored_public,
            table_path=args.score_table,
            manifest_path=args.score_manifest,
        )
    elif args.require_score_components and NEURODISCOVERY in methods:
        raise RuntimeError("formal NeuroDiscovery runs require frozen score components")
    static_score, static_audit = compose_static_score(
        scored_public,
        ScoreComponentConfig(),
    )
    menu_public = blinded_public_registry(scored_public)
    menu_public.to_csv(args.out_dir / "public_candidate_registry.csv", index=False)
    write_json(args.out_dir / "score_component_audit.json", component_audit)

    secret = os.environ.get(args.api_key_env, "").strip()
    if any(method in FRAMEWORK_METHODS for method in methods) and not secret:
        raise RuntimeError(f"missing API key environment variable: {args.api_key_env}")

    all_slots: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    seed_manifests: list[dict[str, Any]] = []
    completed: list[
        tuple[int, str, pd.DataFrame, pd.DataFrame, dict[str, Any]]
    ] = []
    resumed_completed_jobs = 0
    jobs: list[tuple[int, str, Sequence[np.ndarray]]] = []
    for seed in seeds:
        shared_menus = menu_indices(len(menu_public), seed=seed, protocol=protocol)
        menu_hashes = [
            sha256_bytes(
                render_menu_tsv(menu_public.iloc[indices]).encode("utf-8")
            )
            for indices in shared_menus
        ]
        write_json(
            args.out_dir / "shared_menus" / f"seed_{seed:02d}.json",
            {
                "seed": seed,
                "menu_hashes": menu_hashes,
                "same_for_every_method": True,
                "outcome_blind": True,
            },
        )
        for method in methods:
            reused = load_completed_seed_artifacts(
                output_dir=args.out_dir,
                method=method,
                seed=seed,
                protocol=protocol,
            )
            if reused is None:
                jobs.append((seed, method, shared_menus))
                continue
            slots, curve, seed_manifest = reused
            completed.append((seed, method, slots, curve, seed_manifest))
            resumed_completed_jobs += 1
            print(
                json.dumps(
                    {
                        "event": "method_seed_reused",
                        "method": method,
                        "seed": seed,
                    }
                ),
                flush=True,
            )

    def execute_job(
        job: tuple[int, str, Sequence[np.ndarray]],
    ) -> tuple[int, str, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        seed, method, shared_menus = job
        print(
            json.dumps(
                {"event": "method_seed_start", "method": method, "seed": seed}
            ),
            flush=True,
        )
        slots, curve, seed_manifest = run_seed_method(
                method=method,
                seed=seed,
                protocol=protocol,
                menus=shared_menus,
                menu_public=menu_public,
                scored_public=scored_public,
                static_score=static_score,
                static_audit=static_audit,
                vault=vault,
                output_dir=args.out_dir,
                secret=secret,
                args=args,
            )
        print(
            json.dumps(
                {
                    "event": "method_seed_complete",
                    "method": method,
                    "seed": seed,
                    "gt_hits": seed_manifest["gt_hits"],
                    "valid_hypotheses": seed_manifest["valid_hypotheses"],
                }
            ),
            flush=True,
        )
        return seed, method, slots, curve, seed_manifest

    if args.parallel_workers == 1:
        completed.extend(execute_job(job) for job in jobs)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
            futures = [pool.submit(execute_job, job) for job in jobs]
            for future in as_completed(futures):
                completed.append(future.result())

    for _, _, slots, curve, seed_manifest in sorted(
        completed, key=lambda value: (value[0], methods.index(value[1]))
    ):
        all_slots.append(slots)
        all_curves.append(curve)
        seed_manifests.append(seed_manifest)

    slots_table = pd.concat(all_slots, ignore_index=True)
    curves_table = pd.concat(all_curves, ignore_index=True)
    aggregate = aggregate_curves(curves_table)
    overlap = pairwise_selection_overlap(slots_table)
    overlap_aggregate = aggregate_pairwise_overlap(overlap)
    slots_table.to_csv(args.out_dir / "all_direct_slots.csv", index=False)
    curves_table.to_csv(args.out_dir / "all_seed_direct_curves.csv", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_direct_curves_mean_variance.csv", index=False)
    overlap.to_csv(args.out_dir / "pairwise_selection_overlap_by_seed.csv", index=False)
    overlap_aggregate.to_csv(
        args.out_dir / "pairwise_selection_overlap_mean_variance.csv", index=False
    )

    manifest = {
        "schema_version": "case1-full-autoresearch-comparison.v1",
        "created_at": utc_now(),
        "case_study_id": CASE_STUDY_ID,
        "methods": list(methods),
        "seeds": list(seeds),
        "protocol": protocol.to_dict(),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "api_key_source": f"environment:{args.api_key_env}",
        "api_key_persisted": False,
        "candidate_table": {
            "path": str(args.all_tests.resolve()),
            "sha256": sha256_file(args.all_tests),
            "candidates": vault.candidate_count,
            "gt_top_candidates": vault.gt_total,
            "gt_definition": (
                f"top {protocol.gt_top_fraction:.6g} by frozen exhaustive outcome score"
            ),
        },
        "canonical_kg_release": release,
        "score_components": component_audit,
        "static_score_components": static_audit,
        "shared_menus": True,
        "shared_heuristic_expansion": False,
        "parallel_workers": int(args.parallel_workers),
        "resumed_completed_jobs": resumed_completed_jobs,
        "invalid_slots_consume_cost": True,
        "outcomes_revealed_only_after_commitment": True,
        "standardized_experiment_environment": (
            "Frozen exhaustive TCP outcome table; native executor reliability is a "
            "separate capability endpoint and is not imputed for unsupported frameworks."
        ),
        "seed_manifests": seed_manifests,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(args.out_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(args.out_dir.resolve()),
                "methods": list(methods),
                "seeds": list(seeds),
                "slots": len(slots_table),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
