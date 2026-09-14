"""Client-facing selection of reviewed hypotheses, without changing evidence labels.

Adapted from the separately frozen novelty-gated experiment's selection policy.
This module does not import, edit, resume, or run that experiment. Decision
utilities are not probabilities of truth, novelty, or first publication.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODES = ("strict", "novelty_first", "weighted")
DEFAULT_MODE = "strict"
POLICY_VERSION = "reviewed-novelty-v1"
KNOWN = frozenset(("exact_prior", "same_scientific_conclusion"))
PROMISING = frozenset(("substantive_extension", "potential_new_relation"))
CLASSES = KNOWN | PROMISING | {"uncertain"}
PRIORITY_POINTS = {
    "substantive_extension": 1.0, "potential_new_relation": 0.7,
    "uncertain": 0.2, "same_scientific_conclusion": 0.0, "exact_prior": 0.0,
}
WEIGHTS = {"novelty": 0.4, "structural": 0.2, "gnn": 0.2, "critic": 0.2}


def validate_mode(value: object = DEFAULT_MODE) -> str:
    if not isinstance(value, str) or value not in MODES:
        raise ValueError("novelty_mode must be strict, novelty_first, or weighted")
    return value


def _score(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"{name} must be a finite score between 0 and 1")
    return float(value)


def _flag(record: dict, name: str) -> bool:
    value = record.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _review(record: object) -> dict:
    if (not isinstance(record, dict) or not isinstance(record.get("classification"), str)
            or record["classification"] not in CLASSES):
        raise ValueError("A valid literature classification is required")
    refs = record.get("reference_ids")
    if not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs):
        raise ValueError("reference_ids must be a list of nonempty reference identifiers")
    if not isinstance(record.get("scientific_delta"), str):
        raise ValueError("scientific_delta must be a string; use empty text when unknown")
    _flag(record, "executable_test_covers_delta")
    return record


def assess(science: dict, expert_novelties: list, adjudication: dict) -> dict:
    """Apply the same science/novelty gates in every mode to documented reviews.

    Callers must supply actual literature review evidence, not model self-scores
    pretending to be an independent review. This function validates contracts;
    it cannot certify the contents of a cited paper.
    """
    if (not isinstance(science, dict) or not isinstance(science.get("verdict"), str)
            or science["verdict"] not in {"pass", "fail", "revise"}):
        raise ValueError("science.verdict must be pass, fail, or revise")
    critic = _score(science.get("critic_score"), "critic_score")
    if not isinstance(expert_novelties, list) or len(expert_novelties) != 3:
        raise ValueError("Exactly three independent expert perspectives are required")
    reviews = [_review(r) for r in expert_novelties]
    adj = _review(adjudication)
    sufficient = _flag(adj, "search_evidence_sufficient")
    operational = _flag(adj, "registered_test_matches_hypothesis")
    classification = adj["classification"]
    known_veto = classification in KNOWN or any(r["classification"] in KNOWN for r in reviews)
    votes = sum(
        r["classification"] in PROMISING and bool(r["reference_ids"])
        and bool(r["scientific_delta"].strip()) and r["executable_test_covers_delta"]
        for r in reviews
    )
    novel = bool(
        not known_veto and classification in PROMISING and votes >= 2
        and adj["reference_ids"] and adj["scientific_delta"].strip()
        and adj["executable_test_covers_delta"] and sufficient
    )
    priority = classification if novel else "same_scientific_conclusion" if known_veto else "uncertain"
    return {
        "scientific_quality_pass": science["verdict"] == "pass" and critic >= 0.6 and operational,
        "novel_candidate_gate_passed": novel,
        "known_prior_veto": known_veto,
        "novelty_vote_count": votes,
        "literature_classification": classification,
        "selection_priority_class": priority,
        "novelty_priority_points": PRIORITY_POINTS[priority],
        "not_a_first_report_probability": True,
        "eligible_to_claim_new_finding": False,
    }


def choose(rows: list[dict], mode: str, limit: int = 5) -> tuple[list[dict], list[str]]:
    """Deterministically rank assessed rows; preserve all unselected records."""
    validate_mode(mode)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 2000:
        raise ValueError("limit must be an integer between 0 and 2000")
    result = []
    seen = set()
    for source in rows:
        row = dict(source)
        hid = row.get("hypothesis_id")
        if not isinstance(hid, str) or not hid.strip() or hid in seen:
            raise ValueError("Every candidate must have a unique nonempty hypothesis_id")
        seen.add(hid)
        for field in ("structural_score", "gnn_path_score", "critic_score", "novelty_priority_points"):
            _score(row.get(field), field)
        novel = _flag(row, "novel_candidate_gate_passed")
        known = _flag(row, "known_prior_veto")
        science_pass = _flag(row, "scientific_quality_pass")
        quality = 0.3 * row["structural_score"] + 0.3 * row["gnn_path_score"] + 0.4 * row["critic_score"]
        weighted = (0.4 * row["novelty_priority_points"] + 0.2 * row["structural_score"]
                    + 0.2 * row["gnn_path_score"] + 0.2 * row["critic_score"])
        eligible = science_pass and (mode != "strict" or novel)
        row.update(
            novelty_mode=mode, quality_score=quality, weighted_score=weighted,
            novelty_tier=0 if novel else 2 if known else 1,
            eligible_for_experiment=eligible, selected=False,
            selection_reason=("outside_experiment_budget" if eligible else
                              "scientific_quality_or_operationalization_failed" if not science_pass else
                              "strict_novelty_gate_failed"),
            final_score=weighted if mode == "weighted" else quality,
            eligible_to_claim_new_finding=False,
        )
        result.append(row)

    def order(row):
        return ((row["novelty_tier"],) if mode == "novelty_first" else ()) + (-row["final_score"], row["hypothesis_id"])

    selected = sorted((r for r in result if r["eligible_for_experiment"]), key=order)[:limit]
    selected_ids = [r["hypothesis_id"] for r in selected]
    for row in selected:
        row["selected"] = True
        row["selection_reason"] = (
            "selected_novelty_candidate" if row["novel_candidate_gate_passed"] else
            "selected_known_replication_not_new_finding" if row["known_prior_veto"] else
            "selected_uncertain_exploratory_not_verified_novel"
        )
    return result, selected_ids


def select_reviewed(candidates: object, mode: str = DEFAULT_MODE, limit: int = 5) -> dict:
    """Public boundary: recompute gates from reviews, never trust client gate flags."""
    validate_mode(mode)
    if not isinstance(candidates, list) or len(candidates) > 2000:
        raise ValueError("candidates must be a list of at most 2000 reviewed hypotheses")
    assessed = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be an object")
        assessment = assess(candidate.get("science"), candidate.get("expert_novelties"), candidate.get("adjudication"))
        assessed.append({
            **candidate, **assessment,
            "critic_score": candidate["science"]["critic_score"],
        })
    rows, ids = choose(assessed, mode, limit)
    return {"policy_version": POLICY_VERSION, "novelty_mode": mode, "weights": dict(WEIGHTS),
            "selected_ids": ids, "candidates": rows, "eligible_to_claim_new_finding": False}


def build_selection_prompt(mode: str = DEFAULT_MODE) -> str:
    validate_mode(mode)
    rules = {
        "strict": "Exclude known conclusions and unresolved novelty. Return fewer candidates or none; never fill the quota.",
        "novelty_first": "Rank evidence-backed novel candidates first, uncertain exploratory candidates next, known replication candidates last.",
        "weighted": "Use 40% novelty decision utility, 20% structural score, 20% GNN score, and 20% scientific review score.",
    }
    return (
        f"[Hypothesis selection preference: novelty_mode={mode}]\n"
        "Apply this only when generating/selecting research hypotheses, not to unrelated tasks. "
        + rules[mode] + " Preserve original literature classifications and all rejected candidates. "
        "Known relations remain replication, uncertain relations remain unverified; selection is not a new-finding claim. "
        "All modes require scientific validity and an executable test. Do not invent missing reviews, GNN scores, or literature evidence. "
        "Use neurooracle.src.novelty_policy.select_reviewed to apply the deterministic policy to documented "
        "science, three expert_novelties, and independent adjudication records; its CLI is "
        f"python -m neurooracle.src.novelty_policy --input reviewed.json --output selection.json --mode {mode}. "
        "Consult that module's input contract before constructing records. If evidence is unavailable, disclose the gap. "
        "Record this choice before a new workflow is frozen. Never alter an already frozen/running experiment or "
        "reinterpret historical results because the client preference changed. This preference does not authorize experiments."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of reviewed candidates")
    parser.add_argument("--output", type=Path, required=True, help="New selection JSON (never overwritten)")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    candidates = json.loads(args.input.read_text(encoding="utf-8"))
    result = select_reviewed(candidates, args.mode, args.limit)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
