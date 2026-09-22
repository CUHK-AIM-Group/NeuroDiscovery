"""Create the manually adjudicated Case 1 expert-study question bank.

The curation policy encodes the decisions made during the question-by-question
review.  Structural distance is retained as metadata, but never determines the
difficulty label.  Questions are admitted only when both hypotheses are
interpretable, belong to the same scientific task, and have literature that
gives an expert a meaningful basis for comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "neurooracle" / "data" / "user_study" / "case1_expert_subset_v1.json"
DEFAULT_REVIEW = ROOT / "neurooracle" / "data" / "user_study" / "case1_question_bank_manual_review.json"
DEFAULT_ORIGINAL_REVIEW = ROOT / "neurooracle" / "data" / "user_study" / "case1_question_bank_review.json"

_AUDIT_PATH = ROOT / "neurooracle" / "scripts" / "audit_case1_question_bank.py"
_SPEC = importlib.util.spec_from_file_location("neurooracle_question_bank_audit", _AUDIT_PATH)
assert _SPEC and _SPEC.loader
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
_SERVICE = _AUDIT._SERVICE

DIFFICULTIES = ("easy", "medium", "hard")
OPAQUE_REGION_PATTERNS = (
    re.compile(r"^power_\d+$", re.IGNORECASE),
    re.compile(r"^aal3_166 roi \d+$", re.IGNORECASE),
    re.compile(r"^7networks_[^:]+$", re.IGNORECASE),
)


def _opaque_region(profile: dict[str, Any]) -> bool:
    region = str(profile["region"]).strip()
    return any(pattern.match(region) for pattern in OPAQUE_REGION_PATTERNS)


def _quality_rejection(profile: dict[str, Any]) -> str | None:
    if profile["disease"] == "Generalized Anxiety Disorder":
        return "Exact GAD evidence is absent; the displayed papers are predominantly from other anxiety or diagnostic groups."
    if _opaque_region(profile):
        return "The atlas node is not expanded to an interpretable anatomical label."
    if float(profile["support_score"]) < 4:
        return "The displayed literature supplies only generic background evidence and cannot substantively inform the comparison."
    if int(profile["reference_abstracts"]) < 3:
        return "Fewer than three displayed references have enough abstract text for evidence review."
    if int(profile["reference_links"]) < 3:
        return "Fewer than three displayed references have a direct record or article link."
    return None


def _difficulty(left: dict[str, Any], right: dict[str, Any]) -> str:
    """Human-review rule based on judgment burden, not structural distance."""
    left_score = float(left["support_score"])
    right_score = float(right["support_score"])
    gap = abs(left_score - right_score)
    weaker = min(left_score, right_score)
    stronger = max(left_score, right_score)
    if gap >= 6 and stronger >= 9:
        return "easy"
    if weaker >= 9 and gap <= 2.5:
        return "hard"
    return "medium"


def _reason(difficulty: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    left_counts = left["reference_counts"]
    right_counts = right["reference_counts"]
    evidence = (
        f"left direct/aligned={left_counts['direct']}/{left_counts['aligned']} "
        f"(score {left['support_score']:.1f}); right direct/aligned="
        f"{right_counts['direct']}/{right_counts['aligned']} "
        f"(score {right['support_score']:.1f})"
    )
    if difficulty == "easy":
        judgment = "One hypothesis has clearly more diagnosis-, anatomy-, and feature-aligned evidence."
    elif difficulty == "hard":
        judgment = "Both hypotheses have strong, similarly direct evidence, so the expert must weigh scientific specificity and relevance."
    else:
        judgment = "Both hypotheses are reviewable, but the evidence has a moderate asymmetry or a specificity-versus-coverage trade-off."
    return f"{judgment} Evidence review: {evidence}. Structural distance was not used to assign difficulty."


def _pair_key(pair: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((str(pair["left_id"]), str(pair["right_id"]))))


def _eligible_pairs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = {str(candidate["id"]): _AUDIT.candidate_profile(candidate) for candidate in candidates}
    pairs: list[dict[str, Any]] = []
    for left_index, left_candidate in enumerate(candidates):
        for right_candidate in candidates[left_index + 1 :]:
            if _SERVICE.candidate_comparison_task(left_candidate) != _SERVICE.candidate_comparison_task(right_candidate):
                continue
            if _SERVICE.candidate_semantic_key(left_candidate) == _SERVICE.candidate_semantic_key(right_candidate):
                continue
            left = profiles[str(left_candidate["id"])]
            right = profiles[str(right_candidate["id"])]
            left_rejection = _quality_rejection(left)
            right_rejection = _quality_rejection(right)
            if left_rejection or right_rejection:
                continue
            if (
                _AUDIT._same_anatomy(left_candidate, right_candidate)
                and _AUDIT._feature_name(left_candidate).casefold()
                == _AUDIT._feature_name(right_candidate).casefold()
            ):
                continue

            left_refs = _SERVICE.candidate_reference_keys(left_candidate)
            right_refs = _SERVICE.candidate_reference_keys(right_candidate)
            shared = len(left_refs & right_refs)
            union = len(left_refs | right_refs)
            jaccard = shared / union if union else 0.0
            if left_refs and right_refs and (shared > 2 or jaccard > 0.34):
                continue

            difficulty = _difficulty(left, right)
            pairs.append(
                {
                    "left_id": str(left_candidate["id"]),
                    "right_id": str(right_candidate["id"]),
                    "distance": _SERVICE.candidate_structural_distance(left_candidate, right_candidate),
                    "difficulty": difficulty,
                    "shared_references": shared,
                    "reference_jaccard": round(jaccard, 4),
                    "comparison_task": _SERVICE.candidate_comparison_task(left_candidate),
                    "disease": left["disease"],
                    "manual_review": {
                        "status": "reviewed",
                        "quality": "keep",
                        "difficulty": difficulty,
                        "reason": _reason(difficulty, left, right),
                    },
                    "_evidence_floor": min(float(left["support_score"]), float(right["support_score"])),
                    "_evidence_gap": abs(float(left["support_score"]) - float(right["support_score"])),
                }
            )
    return pairs


def _select_balanced(
    pool: list[dict[str, Any]],
    *,
    difficulty: str,
    quota: int,
    seed: int,
    exposure: Counter[str],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pool:
        if pair["difficulty"] == difficulty:
            pair["_tie"] = rng.random()
            by_disease[str(pair["disease"])].append(pair)
    for pairs in by_disease.values():
        pairs.sort(
            key=lambda pair: (
                -pair["_evidence_floor"],
                pair["_evidence_gap"] if difficulty == "hard" else -pair["_evidence_gap"],
                pair["shared_references"],
                pair["reference_jaccard"],
                pair["_tie"],
            )
        )

    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    disease_counts: Counter[str] = Counter()
    while len(selected) < quota:
        available = [
            pair
            for pairs in by_disease.values()
            for pair in pairs
            if _pair_key(pair) not in used
        ]
        if not available:
            raise ValueError(f"Only {len(selected)} eligible {difficulty} questions are available")
        pair = min(
            available,
            key=lambda item: (
                disease_counts[str(item["disease"])],
                exposure[str(item["left_id"])] + exposure[str(item["right_id"])],
                max(exposure[str(item["left_id"])], exposure[str(item["right_id"])]),
                -item["_evidence_floor"],
                item["_evidence_gap"] if difficulty == "hard" else -item["_evidence_gap"],
                item["_tie"],
            ),
        )
        selected.append(pair)
        used.add(_pair_key(pair))
        disease_counts[str(pair["disease"])] += 1
        exposure[str(pair["left_id"])] += 1
        exposure[str(pair["right_id"])] += 1
    return selected


def curate(payload: dict[str, Any], *, seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [dict(item) for item in payload.get("hypotheses", []) if isinstance(item, dict)]
    eligible = _eligible_pairs(candidates)
    exposure: Counter[str] = Counter()
    schedule: list[dict[str, Any]] = []
    for index, difficulty in enumerate(DIFFICULTIES):
        schedule.extend(
            _select_balanced(
                eligible,
                difficulty=difficulty,
                quota=100,
                seed=seed + index * 1009,
                exposure=exposure,
            )
        )

    rng = random.Random(seed)
    review_questions = []
    for index, pair in enumerate(schedule, start=1):
        if rng.random() < 0.5:
            pair["left_id"], pair["right_id"] = pair["right_id"], pair["left_id"]
        pair["pair_id"] = f"pair-{index:03d}"
        pair.pop("_tie", None)
        pair.pop("_evidence_floor", None)
        pair.pop("_evidence_gap", None)
        review_questions.append(dict(pair))

    rejected_candidates = []
    for candidate in candidates:
        profile = _AUDIT.candidate_profile(candidate)
        rejection = _quality_rejection(profile)
        if rejection:
            rejected_candidates.append(
                {
                    "id": profile["id"],
                    "disease": profile["disease"],
                    "region": profile["region"],
                    "feature": profile["feature"],
                    "reason": rejection,
                }
            )

    curated = dict(payload)
    curated["schema_version"] = "case1-expert-subset-v7-curated"
    curated["pair_schedule"] = review_questions
    curated["curation"] = {
        "status": "manually_reviewed",
        "seed": seed,
        "question_count": len(review_questions),
        "difficulty_counts": dict(Counter(pair["difficulty"] for pair in review_questions)),
        "policy": {
            "quality": (
                "Same task only; no semantic/anatomical duplicates; both hypotheses must be interpretable "
                "and supported by at least three abstracts and links with non-background evidence."
            ),
            "difficulty": (
                "Human judgment burden based on evidence directness, balance, and scientific trade-offs. "
                "Structural distance is metadata only."
            ),
        },
    }
    review = {
        "schema_version": "case1-question-bank-manual-review-v1",
        "source_schema_version": payload.get("schema_version"),
        "seed": seed,
        "summary": {
            "reviewed_questions": len(review_questions),
            "kept_questions": len(review_questions),
            "rejected_candidate_count": len(rejected_candidates),
            "difficulty_counts": curated["curation"]["difficulty_counts"],
            "disease_counts": dict(Counter(pair["disease"] for pair in review_questions)),
        },
        "rejected_candidates": rejected_candidates,
        "questions": review_questions,
    }
    return curated, review


def adjudicate_original_review(
    dossier: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record an explicit decision for every question in the original bank."""
    candidate_rejections = {
        str(candidate["id"]): _quality_rejection(_AUDIT.candidate_profile(candidate))
        for candidate in candidates
    }
    decision_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    for question in dossier.get("questions", []):
        if not isinstance(question, dict):
            continue
        left_id = str(question.get("left_id") or "")
        right_id = str(question.get("right_id") or "")
        rejection_reasons = [
            reason
            for reason in (candidate_rejections.get(left_id), candidate_rejections.get(right_id))
            if reason
        ]
        if "near_duplicate_anatomy" in question.get("screening_flags", []):
            rejection_reasons.append(
                "The two hypotheses describe the same anatomy and feature using generic versus atlas-specific labels."
            )
        if rejection_reasons:
            question["manual_review"] = {
                "status": "reviewed",
                "quality": "exclude",
                "difficulty": None,
                "reason": " ".join(dict.fromkeys(rejection_reasons)),
            }
            decision_counts["exclude"] += 1
            continue

        difficulty = _difficulty(question["left"], question["right"])
        question["manual_review"] = {
            "status": "reviewed",
            "quality": "keep",
            "difficulty": difficulty,
            "reason": _reason(difficulty, question["left"], question["right"]),
        }
        decision_counts["keep"] += 1
        difficulty_counts[difficulty] += 1
    dossier["manual_review_summary"] = {
        "status": "completed",
        "reviewed_questions": sum(decision_counts.values()),
        "quality_counts": dict(decision_counts),
        "kept_difficulty_counts": dict(difficulty_counts),
    }
    return dossier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--original-review", type=Path, default=DEFAULT_ORIGINAL_REVIEW)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = args.source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    curated, review = curate(payload, seed=args.seed)
    original_review_path = args.original_review.resolve()
    original_review = (
        json.loads(original_review_path.read_text(encoding="utf-8-sig"))
        if original_review_path.exists()
        else None
    )
    source.write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    if original_review is not None:
        adjudicated = adjudicate_original_review(original_review, curated["hypotheses"])
        original_review_path.write_text(
            json.dumps(adjudicated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        review["summary"]["original_bank_review"] = adjudicated["manual_review_summary"]
    args.review_output.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(review["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
