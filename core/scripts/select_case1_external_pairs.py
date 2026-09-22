"""Select blinded confirmed-versus-falsified Expert Study comparisons.

Pairs are formed within the same disease-versus-control task.  Atlas, ROI, and
feature may differ, which avoids the earlier failure mode where both sides
showed nearly identical literature.  Every hypothesis is used at most once,
and near-ties in the frozen NeuroDiscovery score are excluded before
difficulty tertiles are computed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DIFFICULTIES = ("easy", "medium", "hard")
REQUIRED_SESSIONS = 6


def quantile(values: list[float], fraction: float) -> float:
    return float(pd.Series(values, dtype=float).quantile(fraction))


def eligible_pairs(
    labels: pd.DataFrame,
    *,
    minimum_score_gap: float,
) -> list[dict[str, Any]]:
    eligible = labels[labels["external_label"].isin({"confirmed", "falsified"})]
    pairs: list[dict[str, Any]] = []
    for disease, group in eligible.groupby("disease"):
        confirmed = group[group["external_label"] == "confirmed"]
        falsified = group[group["external_label"] == "falsified"]
        for _, positive in confirmed.iterrows():
            for _, negative in falsified.iterrows():
                same_source = str(positive["source"]) == str(negative["source"])
                same_roi = (
                    same_source
                    and int(positive["roi_index"]) == int(negative["roi_index"])
                )
                same_feature = str(positive["feature"]) == str(negative["feature"])
                if same_roi and same_feature:
                    continue
                gap = abs(
                    float(positive["score_neurodiscovery"])
                    - float(negative["score_neurodiscovery"])
                )
                if gap < minimum_score_gap:
                    continue
                pairs.append(
                    {
                        "confirmed_id": str(positive["candidate_id"]),
                        "falsified_id": str(negative["candidate_id"]),
                        "disease": str(disease),
                        "confirmed_source": str(positive["source"]),
                        "falsified_source": str(negative["source"]),
                        "confirmed_feature": str(positive["feature"]),
                        "falsified_feature": str(negative["feature"]),
                        "confirmed_roi_index": int(positive["roi_index"]),
                        "confirmed_roi_name": str(positive["roi_name"]),
                        "falsified_roi_index": int(negative["roi_index"]),
                        "falsified_roi_name": str(negative["roi_name"]),
                        "confirmed_rank": int(positive["rank"]),
                        "falsified_rank": int(negative["rank"]),
                        "confirmed_generator_score": float(
                            positive["score_neurodiscovery"]
                        ),
                        "falsified_generator_score": float(
                            negative["score_neurodiscovery"]
                        ),
                        "generator_score_gap": gap,
                        "structural_distance": (
                            int(not same_source)
                            + int(not same_roi)
                            + int(not same_feature)
                        ),
                    }
                )
    if not pairs:
        return []

    gaps = [float(pair["generator_score_gap"]) for pair in pairs]
    hard_upper = quantile(gaps, 1 / 3)
    medium_upper = quantile(gaps, 2 / 3)
    for pair in pairs:
        gap = float(pair["generator_score_gap"])
        pair["difficulty"] = (
            "hard"
            if gap <= hard_upper
            else "medium"
            if gap <= medium_upper
            else "easy"
        )
        pair["difficulty_cutoffs"] = {
            "near_tie_minimum": minimum_score_gap,
            "hard_upper": hard_upper,
            "medium_upper": medium_upper,
        }
    return pairs


def select_pairs(
    proposals: list[dict[str, Any]],
    *,
    target_per_difficulty: int,
    falsified_max_reuse: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    for proposal in proposals:
        proposal["_tie"] = rng.random()

    confirmed_ids = {item["confirmed_id"] for item in proposals}
    falsified_ids = {item["falsified_id"] for item in proposals}
    maximum_pairs = min(
        target_per_difficulty * len(DIFFICULTIES),
        len(confirmed_ids),
        len(falsified_ids) * falsified_max_reuse,
    )
    if not proposals or maximum_pairs == 0:
        return []

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix

        candidate_ids = sorted(confirmed_ids | falsified_ids)
        row_by_candidate = {
            candidate_id: index
            for index, candidate_id in enumerate(candidate_ids)
        }
        total_row = len(candidate_ids)
        matrix = lil_matrix(
            (len(candidate_ids) + 1, len(proposals)),
            dtype=float,
        )
        costs = np.empty(len(proposals), dtype=float)
        for column, item in enumerate(proposals):
            matrix[row_by_candidate[item["confirmed_id"]], column] = 1.0
            matrix[row_by_candidate[item["falsified_id"]], column] = 1.0
            matrix[total_row, column] = 1.0
            rank_cost = (
                int(item["confirmed_rank"]) + int(item["falsified_rank"])
            ) / 10_000_000
            distance_bonus = int(item["structural_distance"]) / 100_000
            costs[column] = -1.0 + rank_cost - distance_bonus + item["_tie"] / 1e9

        lower = np.zeros(matrix.shape[0], dtype=float)
        upper = np.ones(matrix.shape[0], dtype=float)
        for candidate_id in falsified_ids:
            upper[row_by_candidate[candidate_id]] = falsified_max_reuse
        lower[total_row] = maximum_pairs
        upper[total_row] = maximum_pairs
        constraints = LinearConstraint(matrix.tocsr(), lower, upper)
        result = milp(
            c=costs,
            integrality=np.ones(len(proposals), dtype=int),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": 60},
        )
        if result.x is not None and result.success:
            return [
                proposal
                for proposal, chosen in zip(proposals, result.x, strict=True)
                if chosen > 0.5
            ]
        # If the exact target is infeasible, maximize the total under the
        # hypothesis reuse limits.
        relaxed_lower = np.zeros(matrix.shape[0], dtype=float)
        upper[total_row] = maximum_pairs
        relaxed = LinearConstraint(matrix.tocsr(), relaxed_lower, upper)
        result = milp(
            c=costs,
            integrality=np.ones(len(proposals), dtype=int),
            bounds=Bounds(0, 1),
            constraints=relaxed,
            options={"time_limit": 60},
        )
        if result.x is not None and result.success:
            return [
                proposal
                for proposal, chosen in zip(proposals, result.x, strict=True)
                if chosen > 0.5
            ]
    except (ImportError, ValueError):
        pass

    # Deterministic fallback if the exact balanced matching is infeasible or
    # SciPy's MILP solver is unavailable.
    selected: list[dict[str, Any]] = []
    used_confirmed: set[str] = set()
    falsified_uses: Counter[str] = Counter()
    options = sorted(
        proposals,
        key=lambda item: (
            -int(item["structural_distance"]),
            max(int(item["confirmed_rank"]), int(item["falsified_rank"])),
            item["_tie"],
        ),
    )
    for item in options:
        if len(selected) >= maximum_pairs:
            break
        if item["confirmed_id"] in used_confirmed:
            continue
        if falsified_uses[item["falsified_id"]] >= falsified_max_reuse:
            continue
        selected.append(item)
        used_confirmed.add(item["confirmed_id"])
        falsified_uses[item["falsified_id"]] += 1
    return selected


def assign_selected_difficulties(
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split the final bank by frozen score gap, with exact balanced counts."""
    ordered = sorted(
        selected,
        key=lambda item: (
            float(item["generator_score_gap"]),
            item["confirmed_id"],
            item["falsified_id"],
        ),
    )
    base, remainder = divmod(len(ordered), len(DIFFICULTIES))
    counts = {
        "hard": base + int(remainder > 0),
        "medium": base + int(remainder > 1),
        "easy": base,
    }
    hard_upper = float(ordered[counts["hard"] - 1]["generator_score_gap"])
    medium_upper = float(
        ordered[counts["hard"] + counts["medium"] - 1]["generator_score_gap"]
    )
    cursor = 0
    for difficulty in ("hard", "medium", "easy"):
        for item in ordered[cursor : cursor + counts[difficulty]]:
            item["difficulty"] = difficulty
            item["difficulty_cutoffs"] = {
                "near_tie_minimum": item["difficulty_cutoffs"][
                    "near_tie_minimum"
                ],
                "hard_upper": hard_upper,
                "medium_upper": medium_upper,
                "assignment_scope": "selected_bank",
            }
        cursor += counts[difficulty]
    return selected


def build_outputs(
    selected: list[dict[str, Any]],
    *,
    target_per_difficulty: int,
    minimum_score_gap: float,
    falsified_max_reuse: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    rng = random.Random(seed + 991)
    session_buckets: list[list[dict[str, Any]]] = [
        [] for _ in range(REQUIRED_SESSIONS)
    ]
    session_cursor = 0
    for difficulty in DIFFICULTIES:
        records = [
            proposal
            for proposal in selected
            if proposal["difficulty"] == difficulty
        ]
        rng.shuffle(records)
        for proposal in records:
            session_buckets[session_cursor % REQUIRED_SESSIONS].append(proposal)
            session_cursor += 1
    for bucket in session_buckets:
        rng.shuffle(bucket)

    schedule: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    pair_index = 0
    for session_number, bucket in enumerate(session_buckets, start=1):
        for session_question_number, proposal in enumerate(bucket, start=1):
            pair_index += 1
            confirmed_left = rng.random() < 0.5
            left_id = (
                proposal["confirmed_id"]
                if confirmed_left
                else proposal["falsified_id"]
            )
            right_id = (
                proposal["falsified_id"]
                if confirmed_left
                else proposal["confirmed_id"]
            )
            pair_id = f"tcp-ext-pair-{pair_index:03d}"
            public_pair = {
                "pair_id": pair_id,
                "left_id": left_id,
                "right_id": right_id,
                "comparison_task": (
                    f"{proposal['disease']}|case-vs-control"
                ),
                "disease": proposal["disease"],
                "structural_distance": proposal["structural_distance"],
                "difficulty": proposal["difficulty"],
                "difficulty_basis": "absolute_frozen_neurodiscovery_score_gap",
                "generator_score_gap": round(
                    float(proposal["generator_score_gap"]), 8
                ),
                "session_number": session_number,
                "session_question_number": session_question_number,
                "manual_review": {
                    "status": "pending_literature_review",
                    "quality": "pending",
                },
            }
            hidden_pair = {
                "pair_id": pair_id,
                "left_id": left_id,
                "right_id": right_id,
                "preferred_id": proposal["confirmed_id"],
                "left_status": "confirmed" if confirmed_left else "falsified",
                "right_status": "falsified" if confirmed_left else "confirmed",
                "confirmed_generator_score": proposal[
                    "confirmed_generator_score"
                ],
                "falsified_generator_score": proposal[
                    "falsified_generator_score"
                ],
                "generator_score_gap": round(
                    float(proposal["generator_score_gap"]), 8
                ),
                "difficulty": proposal["difficulty"],
                "session_number": session_number,
                "session_question_number": session_question_number,
            }
            schedule.append(public_pair)
            truth.append(hidden_pair)
            rows.append({**proposal, **public_pair})

    counts = Counter(pair["difficulty"] for pair in schedule)
    summary = {
        "schema_version": "case1-tcp-external-pair-selection-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_policy": {
            "outcomes": "exactly one strictly confirmed and one strictly falsified",
            "same_task": (
                "same disease-versus-control contrast; atlas, ROI, and feature "
                "may differ to avoid nearly identical evidence panels"
            ),
            "hypothesis_reuse": {
                "confirmed": False,
                "falsified_max_appearances": falsified_max_reuse,
                "rationale": (
                    "Strict falsified hypotheses are scarce; confirmed hypotheses "
                    "remain unique while falsified hypotheses may appear at most "
                    f"{falsified_max_reuse} times."
                ),
            },
            "minimum_generator_score_gap": minimum_score_gap,
            "difficulty": (
                "The selected bank's frozen NeuroDiscovery score gaps are split "
                "into balanced thirds; largest is easy and smallest meaningful "
                "is hard."
            ),
            "seed": seed,
        },
        "requested_pairs": target_per_difficulty * len(DIFFICULTIES),
        "selected_pairs": len(schedule),
        "requested_per_difficulty": target_per_difficulty,
        "selected_by_difficulty": dict(counts),
        "selected_by_disease": dict(
            Counter(pair["disease"] for pair in schedule)
        ),
        "session_protocol": {
            "required_sessions": REQUIRED_SESSIONS,
            "active_seconds_per_session": 600,
            "fixed_shared_seed": seed,
            "same_questions_for_all_participants": True,
            "pairs_by_session": dict(
                Counter(str(pair["session_number"]) for pair in schedule)
            ),
        },
        "literature_status": (
            "Selection complete; five papers and hypothesis-specific relevance "
            "reasons still required for each selected hypothesis."
        ),
    }
    public = {**summary, "pair_schedule": schedule}
    hidden = {
        "schema_version": "case1-tcp-external-pair-truth-v1",
        "created_at": summary["created_at"],
        "blinding": "Never bundle or serve this file to study participants.",
        "pair_truth": truth,
    }
    return public, hidden, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select strict confirmed-versus-falsified TCP Expert Study pairs."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--truth-output-dir",
        type=Path,
        default=None,
        help=(
            "Private output directory for labels and side-specific truth. "
            "Defaults to <output-dir>/private_truth."
        ),
    )
    parser.add_argument("--target-per-difficulty", type=int, default=80)
    parser.add_argument("--minimum-score-gap", type=float, default=0.03)
    parser.add_argument(
        "--falsified-max-reuse",
        type=int,
        default=1,
        help=(
            "Maximum appearances of a strict falsified hypothesis. Confirmed "
            "hypotheses are always used at most once."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.falsified_max_reuse < 1:
        parser.error("--falsified-max-reuse must be at least 1")

    labels = pd.read_csv(args.labels, low_memory=False)
    proposals = eligible_pairs(
        labels,
        minimum_score_gap=args.minimum_score_gap,
    )
    selected = select_pairs(
        proposals,
        target_per_difficulty=args.target_per_difficulty,
        falsified_max_reuse=args.falsified_max_reuse,
        seed=args.seed,
    )
    selected = assign_selected_difficulties(selected)
    public, hidden, rows = build_outputs(
        selected,
        target_per_difficulty=args.target_per_difficulty,
        minimum_score_gap=args.minimum_score_gap,
        falsified_max_reuse=args.falsified_max_reuse,
        seed=args.seed,
    )
    public["eligible_pair_proposals"] = len(proposals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth_output_dir = args.truth_output_dir or (args.output_dir / "private_truth")
    truth_output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "case1_tcp_external_pair_selection.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (truth_output_dir / "case1_tcp_external_pair_truth.json").write_text(
        json.dumps(hidden, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows.to_csv(
        truth_output_dir / "case1_tcp_external_pair_selection.csv",
        index=False,
    )
    print(json.dumps({key: value for key, value in public.items() if key != "pair_schedule"}, indent=2))


if __name__ == "__main__":
    main()
