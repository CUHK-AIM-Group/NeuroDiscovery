"""Prepare an evidence-aware review dossier for the Case 1 expert-study bank.

This script does not replace expert review.  It freezes the exact pair order
currently produced by a study seed and summarizes the evidence shown on both
sides so a reviewer can adjudicate every question without treating structural
distance as the definition of difficulty.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "neurooracle" / "data" / "user_study" / "case1_expert_subset_v1.json"
DEFAULT_OUTPUT = ROOT / "neurooracle" / "data" / "user_study" / "case1_question_bank_review.json"
_SERVICE_PATH = ROOT / "neurooracle" / "src" / "user_study.py"
_SPEC = importlib.util.spec_from_file_location("neurooracle_question_bank_audit", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
_SERVICE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SERVICE)


DISEASE_ALIASES = {
    "attention deficit disorder with hyperactivity": ("attention deficit", "adhd"),
    "anorexia nervosa": ("anorexia", "eating disorder"),
    "bipolar disorder": ("bipolar", "mania", "manic"),
    "generalized anxiety disorder": ("generalized anxiety", "gad"),
    "major depressive disorder": ("major depressive", "depression", "mdd"),
    "obsessive-compulsive disorder": ("obsessive compulsive", "ocd"),
    "psychotic disorders": ("psychotic", "psychosis"),
    "schizophrenia": ("schizophrenia", "schizophrenic"),
    "stress disorders, post-traumatic": ("post traumatic", "posttraumatic", "ptsd"),
    "substance-related disorders": (
        "substance",
        "addiction",
        "alcohol",
        "cannabis",
        "cocaine",
        "opioid",
        "methamphetamine",
    ),
    "anxiety disorders": ("anxiety",),
}

FEATURE_ALIASES = {
    "roi alff": (" alff ", "amplitude of low frequency", "amplitude of low-frequency"),
    "roi falff": (" falff ", "fractional amplitude of low frequency", "fractional amplitude of low-frequency"),
    "roi between-network fc": ("between network", "between-network", "internetwork", "inter-network"),
    "roi cortical thickness": ("cortical thickness", "cortthick"),
    "roi fc variability": ("connectivity variability", "dynamic functional connectivity", "dynamic connectivity"),
    "roi gray-matter volume": ("gray matter volume", "grey matter volume", "voxel based morphometry", " vbm "),
    "roi local efficiency": ("local efficiency",),
    "roi mean whole-brain fc": (
        "global brain connectivity",
        "whole brain functional connectivity",
        "functional connectivity strength",
    ),
    "roi node degree": ("node degree", "degree centrality"),
    "roi node strength": ("node strength", "weighted degree"),
    "roi participation coefficient": ("participation coefficient",),
    "roi surface area": ("surface area",),
    "roi temporal variance": ("temporal variance", "temporal variability", "signal variability"),
    "roi within-network fc": ("within network", "within-network", "intranetwork", "intra-network"),
    "subject state occupancy": ("state occupancy", "connectivity state", "dynamic state"),
}

GENERIC_FMRI_TERMS = (
    "functional connectivity",
    "connectome",
    "network",
    "resting state",
    "resting-state",
    " fmri ",
)


def _normalized(text: Any) -> str:
    value = re.sub(r"[_/()-]+", " ", str(text or "").casefold())
    return f" {' '.join(value.split())} "


def _candidate_tuple(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    value = metadata.get("candidate_tuple")
    return value if isinstance(value, dict) else {}


def _disease_name(candidate: dict[str, Any]) -> str:
    value = _candidate_tuple(candidate)
    return str(value.get("disease_name") or (value.get("diseases") or [""])[0]).strip()


def _region_name(candidate: dict[str, Any]) -> str:
    value = _candidate_tuple(candidate)
    return str(value.get("region_name") or candidate.get("target_name") or "").strip()


def _feature_name(candidate: dict[str, Any]) -> str:
    value = _candidate_tuple(candidate)
    return str(value.get("feature_name") or value.get("feature_id") or "").strip()


def _direction(candidate: dict[str, Any]) -> str:
    return str(_candidate_tuple(candidate).get("direction") or "").strip().casefold()


def _region_aliases(candidate: dict[str, Any]) -> tuple[str, ...]:
    value = _candidate_tuple(candidate)
    raw = str(value.get("atlas_label") or value.get("region_name") or candidate.get("target_name") or "")
    if ":" in raw:
        raw = raw.rsplit(":", 1)[-1]
    normalized = _normalized(raw).strip()
    normalized = re.sub(r"\b(left|right|bilateral|superior|inferior|medial|lateral)\b", " ", normalized)
    normalized = " ".join(normalized.split())
    aliases = {normalized}
    replacements = {
        "posterior cingulate cortex": "posterior cingulate",
        "anterior cingulate cortex": "anterior cingulate",
        "temporal pole superior": "temporal pole",
        "temporal pole": "temporal pole",
    }
    if normalized in replacements:
        aliases.add(replacements[normalized])
    aliases.discard("")
    return tuple(sorted(aliases, key=len, reverse=True))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_normalized(term).strip() in text for term in terms if str(term).strip())


def _reference_text(reference: dict[str, Any]) -> str:
    excerpts = reference.get("excerpts") if isinstance(reference.get("excerpts"), list) else []
    return _normalized(
        " ".join(
            [
                str(reference.get("title") or ""),
                str(reference.get("abstract") or ""),
                *(str(item) for item in excerpts),
            ]
        )
    )


def _reference_alignment(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    text = _reference_text(reference)
    disease = _disease_name(candidate).casefold()
    disease_terms = DISEASE_ALIASES.get(disease, (disease,))
    disease_match = _contains_any(text, disease_terms)
    region_match = _contains_any(text, _region_aliases(candidate))
    feature = _feature_name(candidate).casefold()
    feature_terms = FEATURE_ALIASES.get(feature, (feature,))
    feature_match = _contains_any(text, feature_terms)
    generic_fmri = _contains_any(text, GENERIC_FMRI_TERMS)
    if disease_match and region_match and feature_match:
        level = "direct"
    elif disease_match and region_match and generic_fmri:
        level = "aligned"
    elif disease_match and (region_match or feature_match):
        level = "partial"
    elif disease_match:
        level = "background"
    else:
        level = "off_target"
    return {
        "title": str(reference.get("title") or "").strip(),
        "year": reference.get("year"),
        "level": level,
        "disease_match": disease_match,
        "region_match": region_match,
        "feature_match": feature_match,
        "has_abstract": bool(str(reference.get("abstract") or "").strip()),
        "has_direct_link": bool(
            reference.get("direct_url")
            or reference.get("doi_url")
            or reference.get("pubmed_url")
            or reference.get("pmc_url")
            or reference.get("url")
        ),
    }


def _source_alignment(candidate: dict[str, Any]) -> dict[str, Any]:
    source_text = _normalized(
        " ".join(
            str(edge.get("raw_text") or "")
            for edge in candidate.get("path", [])
            if isinstance(edge, dict)
        )
    )
    if not source_text.strip():
        return {"region_match": False, "feature_match": False}
    feature = _feature_name(candidate).casefold()
    return {
        "region_match": _contains_any(source_text, _region_aliases(candidate)),
        "feature_match": _contains_any(source_text, FEATURE_ALIASES.get(feature, (feature,))),
    }


def candidate_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    references = candidate.get("literature") if isinstance(candidate.get("literature"), list) else []
    alignments = [
        _reference_alignment(candidate, reference)
        for reference in references[:5]
        if isinstance(reference, dict)
    ]
    counts = {
        level: sum(item["level"] == level for item in alignments)
        for level in ("direct", "aligned", "partial", "background", "off_target")
    }
    support_score = (
        counts["direct"] * 4
        + counts["aligned"] * 3
        + counts["partial"] * 2
        + counts["background"] * 0.5
        - counts["off_target"]
    )
    return {
        "id": str(candidate.get("id") or ""),
        "title": str(candidate.get("title") or ""),
        "disease": _disease_name(candidate),
        "region": _region_name(candidate),
        "feature": _feature_name(candidate),
        "direction": _direction(candidate),
        "source_alignment": _source_alignment(candidate),
        "reference_counts": counts,
        "reference_abstracts": sum(item["has_abstract"] for item in alignments),
        "reference_links": sum(item["has_direct_link"] for item in alignments),
        "support_score": support_score,
        "references": alignments,
    }


def current_runtime_schedule(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    shuffled = [candidates[index] for index in order]
    return _SERVICE.build_progressive_pair_schedule(shuffled, max_pairs=300, seed=seed)


def _same_anatomy(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_regions = set(_region_aliases(left))
    right_regions = set(_region_aliases(right))
    return bool(left_regions and right_regions and left_regions & right_regions)


def review_dossier(source: Path, seed: int) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    candidates = [dict(item) for item in payload.get("hypotheses", []) if isinstance(item, dict)]
    by_id = {str(item["id"]): item for item in candidates}
    profiles = {candidate_id: candidate_profile(item) for candidate_id, item in by_id.items()}
    schedule = current_runtime_schedule(candidates, seed)
    questions = []
    for pair in schedule:
        left = by_id[pair["left_id"]]
        right = by_id[pair["right_id"]]
        left_profile = profiles[pair["left_id"]]
        right_profile = profiles[pair["right_id"]]
        flags = []
        if _same_anatomy(left, right) and _feature_name(left).casefold() == _feature_name(right).casefold():
            flags.append("near_duplicate_anatomy")
        if left_profile["reference_counts"]["off_target"] >= 3:
            flags.append("left_literature_mostly_off_target")
        if right_profile["reference_counts"]["off_target"] >= 3:
            flags.append("right_literature_mostly_off_target")
        if left_profile["support_score"] <= 1 and right_profile["support_score"] <= 1:
            flags.append("both_sides_lack_discriminative_evidence")
        if min(left_profile["reference_abstracts"], right_profile["reference_abstracts"]) < 3:
            flags.append("insufficient_abstract_coverage")
        if min(left_profile["reference_links"], right_profile["reference_links"]) < 3:
            flags.append("insufficient_direct_links")

        gap = abs(float(left_profile["support_score"]) - float(right_profile["support_score"]))
        strong_both = min(left_profile["support_score"], right_profile["support_score"]) >= 7
        if "near_duplicate_anatomy" in flags or "both_sides_lack_discriminative_evidence" in flags:
            suggestion = "exclude"
        elif gap >= 6 or (
            max(left_profile["reference_counts"]["direct"], right_profile["reference_counts"]["direct"]) >= 1
            and min(left_profile["reference_counts"]["direct"], right_profile["reference_counts"]["direct"]) == 0
            and gap >= 3
        ):
            suggestion = "easy"
        elif strong_both and gap <= 2:
            suggestion = "hard"
        else:
            suggestion = "medium"

        questions.append(
            {
                **pair,
                "disease": left_profile["disease"],
                "left": left_profile,
                "right": right_profile,
                "screening_flags": flags,
                "screening_suggestion": suggestion,
                "manual_review": {
                    "status": "pending",
                    "quality": None,
                    "difficulty": None,
                    "reason": "",
                },
            }
        )
    suggestion_counts: dict[str, int] = {}
    for question in questions:
        suggestion = question["screening_suggestion"]
        suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
    return {
        "schema_version": "case1-question-bank-review-v1",
        "source": source.relative_to(ROOT).as_posix(),
        "seed": seed,
        "review_policy": {
            "quality": (
                "Exclude semantic duplicates, malformed hypotheses, and comparisons whose displayed "
                "literature does not give an expert a discriminative scientific basis."
            ),
            "difficulty": (
                "Manual difficulty considers evidence directness, evidence-strength asymmetry, "
                "clinical/anatomical nuance, and judgment trade-offs; structural distance is context only."
            ),
        },
        "question_count": len(questions),
        "screening_suggestion_counts": suggestion_counts,
        "questions": questions,
    }


def print_compact(dossier: dict[str, Any], start: int, limit: int) -> None:
    rows = dossier["questions"][max(0, start - 1) : max(0, start - 1) + max(0, limit)]
    for question in rows:
        left, right = question["left"], question["right"]
        print(
            f"{question['pair_id']} old={question['difficulty']} d={question['distance']} "
            f"screen={question['screening_suggestion']} flags={','.join(question['screening_flags']) or '-'}"
        )
        print(
            f"  L {left['region']} | {left['feature']} | support={left['support_score']:.1f} "
            f"refs={left['reference_counts']}"
        )
        print(
            f"  R {right['region']} | {right['feature']} | support={right['support_score']:.1f} "
            f"refs={right['reference_counts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    dossier = review_dossier(args.source.resolve(), args.seed)
    if args.limit > 0:
        print_compact(dossier, args.start, args.limit)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output.relative_to(ROOT).as_posix(),
                "question_count": dossier["question_count"],
                "screening_suggestion_counts": dossier["screening_suggestion_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
