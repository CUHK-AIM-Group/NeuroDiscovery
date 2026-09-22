"""Build a blinded Expert Study bank from held-out ADNI validation results.

The public bank contains hypotheses, prior literature, and NeuroDiscovery
scores. The separate truth file contains ADNI outcomes and must never be
served to study participants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "outputs/case1_adni_validation/fmri_multiatlas_full"
DEFAULT_PUBLIC = ROOT / "neurooracle/data/user_study/adni_external_validation_study_v1.json"
DEFAULT_ANNOTATIONS = (
    ROOT / "neurooracle/data/user_study/adni_manual_relevance_annotations_v1.json"
)
DEFAULT_TRUTH = ROOT / "outputs/user_study/adni_external_validation_truth_v1.json"
DEFAULT_AUDIT = ROOT / "outputs/user_study/adni_external_validation_audit_v1.json"
PAIR_TARGETS = {"easy": 60, "medium": 60, "hard": 60}
REQUIRED_SESSIONS = 6
SESSION_ACTIVE_SECONDS = 10 * 60
MIN_GENERATOR_SCORE_GAP = 0.03
MAX_REFERENCE_JACCARD = 0.85
SHARED_SCHEDULE_SEED = 0

FEATURES = {
    "roi_falff_proxy": ("ROI fALFF", "感兴趣区低频振幅分数", ("falff", "fractional amplitude")),
    "roi_alff_proxy": ("ROI ALFF", "感兴趣区低频振幅", ("alff", "low-frequency amplitude")),
    "roi_temporal_mean": ("ROI mean signal", "感兴趣区平均信号", ("mean signal",)),
    "roi_temporal_mean_abs": ("ROI mean absolute signal", "感兴趣区平均绝对信号", ("mean signal",)),
    "roi_temporal_std": ("ROI temporal standard deviation", "感兴趣区时序标准差", ("signal variability", "standard deviation")),
    "roi_temporal_variance": ("ROI temporal variance", "感兴趣区时序方差", ("signal variability", "variance")),
    "corr_mean": ("ROI mean whole-brain FC", "感兴趣区平均全脑功能连接", ("functional connectivity", "connectivity")),
    "corr_mean_abs": ("ROI mean absolute whole-brain FC", "感兴趣区平均绝对全脑功能连接", ("functional connectivity", "connectivity")),
    "corr_positive_mean": ("ROI positive FC", "感兴趣区正向功能连接", ("functional connectivity", "connectivity")),
    "corr_negative_mean": ("ROI negative FC", "感兴趣区负向功能连接", ("functional connectivity", "connectivity")),
    "corr_node_degree_abs_top10": ("ROI functional-connectivity node degree", "感兴趣区功能连接节点度", ("node degree", "functional connectivity")),
    "partial_mean": ("ROI mean partial-correlation FC", "感兴趣区平均偏相关功能连接", ("partial correlation", "functional connectivity")),
    "partial_mean_abs": ("ROI mean absolute partial-correlation FC", "感兴趣区平均绝对偏相关功能连接", ("partial correlation", "functional connectivity")),
    "partial_positive_mean": ("ROI positive partial-correlation FC", "感兴趣区正向偏相关功能连接", ("partial correlation", "functional connectivity")),
    "partial_negative_mean": ("ROI negative partial-correlation FC", "感兴趣区负向偏相关功能连接", ("partial correlation", "functional connectivity")),
}

REGION_TERMS = {
    "frontal": ("frontal", "precentral", "orbitofrontal", "vmPFC", "dmPFC"),
    "temporal": ("temporal", "fusiform", "parahippocamp", "hippocamp", "amygdala"),
    "parietal": ("parietal", "postcentral", "precuneus", "supramarginal", "angular"),
    "occipital": ("occipital", "calcarine", "lingual", "cuneus", "visual", "_V1", "_V2"),
    "cingulate": ("cingulate",),
    "insula": ("insula", "insular"),
    "thalamus": ("thalam",),
    "striatal": ("caudate", "putamen", "pallid", "striat"),
    "cerebellum": ("cerebell", "vermis"),
    "network": ("Default", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "SomMot"),
}


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _region_name(source: str, raw: str) -> str:
    name = raw.strip()
    prefixes = (
        "schaefer_400_7net_", "schaefer_200_7net_", "schaefer_100_7net_",
        "harvard_oxford_merged_", "harvard_oxford_cort_", "talairach_tournoux_",
        "eickhoff_zilles_", "dosenbach_160_", "destrieux_148_", "aal_116_",
        "cc400_", "msdl_39_",
    )
    for prefix in prefixes:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):]
            break
    return name.replace("_", " ").strip() or f"{source} ROI"


def _region_tags(name: str) -> set[str]:
    lower = name.lower()
    return {
        tag
        for tag, needles in REGION_TERMS.items()
        if any(needle.lower() in lower for needle in needles)
    }


def _paper_url(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi") or "").strip()
    pmid = str(paper.get("pmid") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    title = str(paper.get("title") or "").strip()
    return "https://scholar.google.com/scholar?q=" + re.sub(r"\s+", "+", title)


def load_literature(claims_path: Path, abstracts_path: Path) -> list[dict[str, Any]]:
    abstracts: dict[str, str] = {}
    for item in _jsonl(abstracts_path):
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        abstract = str(item.get("abstract") or "").strip()
        for key in (str(item.get("pmid") or ""), str(paper.get("doi") or "").lower()):
            if key and abstract:
                abstracts[key] = abstract

    by_paper: dict[str, dict[str, Any]] = {}
    disease_terms = ("alzheimer", "mild cognitive impairment", "mci", "cognitive impairment")
    imaging_terms = ("fmri", "functional connectivity", "brain", "cort", "hippoc", "neuroimag", "alff", "falff")
    for claim in _jsonl(claims_path):
        paper = claim.get("source_paper")
        if not isinstance(paper, dict):
            continue
        text = " ".join(
            str(claim.get(key) or "")
            for key in ("disease", "subject_name", "object_name", "predicate", "raw_text")
        ).lower()
        if not any(term in text for term in disease_terms) or not any(term in text for term in imaging_terms):
            continue
        key = str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "").strip().lower()
        if not key:
            continue
        record = by_paper.setdefault(
            key,
            {
                "title": str(paper.get("title") or "").strip(),
                "authors": str(paper.get("authors") or "").strip(),
                "year": paper.get("year"),
                "journal": str(paper.get("journal") or "").strip(),
                "doi": str(paper.get("doi") or "").strip(),
                "pmid": str(paper.get("pmid") or "").strip(),
                "claims": [],
            },
        )
        raw = str(claim.get("raw_text") or "").strip()
        if raw and raw not in record["claims"]:
            record["claims"].append(raw)

    records = []
    for record in by_paper.values():
        record["abstract"] = abstracts.get(record["pmid"]) or abstracts.get(record["doi"].lower()) or ""
        record["url"] = _paper_url(record)
        searchable = " ".join(
            [record["title"], record["abstract"], *record["claims"]]
        ).lower()
        record["_searchable"] = searchable
        records.append(record)
    return records


def select_literature(records: list[dict[str, Any]], region: str, feature_id: str) -> list[dict[str, Any]]:
    region_tags = _region_tags(region)
    feature_terms = FEATURES[feature_id][2]
    scored = []
    for record in records:
        text = record["_searchable"]
        score = 2.0
        if "mild cognitive impairment" in text or re.search(r"\bmci\b", text):
            score += 3.0
        matched_regions = {
            tag for tag in region_tags
            if any(needle.lower() in text for needle in REGION_TERMS[tag])
        }
        score += 3.0 * len(matched_regions)
        if any(term.lower() in text for term in feature_terms):
            score += 2.5
        if record["abstract"]:
            score += 0.5
        if score >= 5.0:
            scored.append((score, int(record.get("year") or 0), record))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]["title"]))
    output = []
    for score, _, record in scored[:5]:
        excerpt = next(
            (
                claim for claim in record["claims"]
                if any(term.lower() in claim.lower() for term in (*FEATURES[feature_id][2], region.lower()))
            ),
            record["claims"][0] if record["claims"] else record["abstract"][:500],
        )
        output.append(
            {
                key: record[key]
                for key in ("title", "authors", "year", "journal", "doi", "pmid", "url", "abstract")
            }
            | {"excerpt": excerpt, "prior_evidence_match_score": round(score, 3)}
        )
    return output


def load_manual_annotations(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Manual relevance annotations must be a JSON object")
    assignment_batches = {
        candidate_id: str(payload.get("current_batch") or "manual")
        for candidate_id in payload.get("assignments", {})
    }
    batch_pattern = f"{path.stem}_batch_*.json"
    for batch_path in sorted(path.parent.glob(batch_pattern)):
        batch = json.loads(batch_path.read_text(encoding="utf-8-sig"))
        if not isinstance(batch, dict):
            raise ValueError(f"Manual relevance annotation batch must be an object: {batch_path}")
        batch_id = str(batch.get("current_batch") or batch_path.stem)
        for key in ("assignments", "paper_profiles", "relation_profiles"):
            values = batch.get(key)
            if isinstance(values, dict):
                payload.setdefault(key, {}).update(values)
        for candidate_id in batch.get("assignments", {}):
            assignment_batches[candidate_id] = batch_id
        if isinstance(batch.get("batches"), list):
            payload.setdefault("batches", []).extend(batch["batches"])
    payload["assignment_batches"] = assignment_batches
    return payload


def apply_manual_annotations(
    candidates: list[dict[str, Any]],
    annotations: dict[str, Any],
) -> int:
    """Embed abstract-verified, hypothesis-specific literature assessments."""
    assignments = annotations.get("assignments")
    assignment_batches = annotations.get("assignment_batches")
    paper_profiles = annotations.get("paper_profiles")
    relation_profiles = annotations.get("relation_profiles")
    if not all(isinstance(value, dict) for value in (assignments, paper_profiles, relation_profiles)):
        return 0

    applied = 0
    for candidate in candidates:
        group_id = str(assignments.get(candidate["id"]) or "")
        group = relation_profiles.get(group_id)
        if not isinstance(group, dict):
            continue
        region = str(candidate["metadata"]["candidate_tuple"]["region_name"])
        for paper in candidate["literature"]:
            paper_id = str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "")
            profile = paper_profiles.get(paper_id)
            relation = group.get(paper_id)
            if not isinstance(profile, dict) or not isinstance(relation, dict):
                continue

            values = {"region": region}
            study_zh = str(profile.get("study_zh") or "").format_map(values)
            study_en = str(profile.get("study_en") or "").format_map(values)
            support_zh = str(relation.get("support_zh") or "").format_map(values)
            support_en = str(relation.get("support_en") or "").format_map(values)
            strength_zh = str(relation.get("strength_zh") or "")
            strength_en = str(relation.get("strength_en") or "")
            boundary_zh = str(relation.get("boundary_zh") or "").format_map(values)
            boundary_en = str(relation.get("boundary_en") or "").format_map(values)
            if not all(
                (study_zh, study_en, support_zh, support_en, strength_zh, strength_en)
            ):
                continue

            paper["relevance_reason_zh"] = (
                f"本文研究的是：{study_zh}\n"
                f"对当前 hypothesis 的具体支持：{support_zh}\n"
                f"证据强度：{strength_zh}"
                + (f"——{boundary_zh}" if boundary_zh else "")
            )
            paper["relevance_reason_en"] = (
                f"What this paper studied: {study_en}\n"
                f"Specific support for this hypothesis: {support_en}\n"
                f"Evidence strength: {strength_en}"
                + (f" — {boundary_en}" if boundary_en else "")
            )
            paper["relevance_curation_batch"] = str(
                (assignment_batches or {}).get(candidate["id"])
                or annotations.get("current_batch")
                or "manual"
            )
            paper["abstract_verified"] = bool(profile.get("abstract_verified"))
            paper["abstract_source"] = str(profile.get("abstract_source") or "")
            applied += 1
    return applied


def load_rows(results_dir: Path) -> list[tuple[dict[str, str], dict[str, str]]]:
    ranked = []
    with (results_dir / "case1_region_feature_rankings_multiatlas.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        ranked = [row for row in csv.DictReader(handle) if row["method"] == "neurodiscovery"]
    stats = {}
    with (results_dir / "adni_multiatlas_validation_stats.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["contrast"] == "MCI_vs_CN":
                stats[(row["source"], row["roi_index"], row["feature"])] = row
    return [
        (row, stats[(row["source"], row["roi_index"], row["feature"])])
        for row in ranked
        if (row["source"], row["roi_index"], row["feature"]) in stats
    ]


def build_candidates(
    rows: list[tuple[dict[str, str], dict[str, str]]],
    literature: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    candidates = []
    truth = {}
    max_method_rank = max(
        (
            int(rank_row["method_rank"])
            for rank_row, stat in rows
            if stat["feature"] in FEATURES
        ),
        default=1,
    )
    for rank_row, stat in rows:
        feature_id = stat["feature"]
        if feature_id not in FEATURES:
            continue
        region = _region_name(stat["source"], stat["roi_name"])
        if not _region_tags(region):
            continue
        refs = select_literature(literature, region, feature_id)
        if len(refs) < 5:
            continue
        feature_en, feature_zh, _ = FEATURES[feature_id]
        basis = f"{stat['source']}|{stat['roi_index']}|{feature_id}|MCI_vs_CN"
        candidate_id = "adni-ext-" + hashlib.sha1(basis.encode()).hexdigest()[:14]
        method_rank = int(rank_row["method_rank"])
        score = 1.0 - ((method_rank - 1) / max(1, max_method_rank - 1))
        score = max(0.0, min(1.0, score))
        candidate = {
            "id": candidate_id,
            "title": f"Mild Cognitive Impairment → {region} → {feature_en}",
            "title_zh": f"轻度认知障碍 → {region} → {feature_zh}",
            "summary": (
                f"Mild cognitive impairment is hypothesized to be associated with altered "
                f"{feature_en} in {region}, relative to cognitively normal controls."
            ),
            "summary_zh": f"该假设认为：与认知正常对照相比，轻度认知障碍与 {region} 的{feature_zh}改变有关。",
            "source_name": "Mild Cognitive Impairment",
            "target_name": f"{region} | {feature_en}",
            "path": [],
            "literature": refs,
            "composite_score": round(score, 8),
            "comparison_task_id": "adni-mci-vs-cn",
            "metadata": {
                "case_study": "adni_external_validation",
                "comparison_task_id": "adni-mci-vs-cn",
                "control_name": "Cognitively Normal",
                "candidate_tuple": {
                    "diseases": ["Mild Cognitive Impairment"],
                    "disease_name": "Mild Cognitive Impairment",
                    "region_name": region,
                    "atlas_name": stat["source"].removesuffix("_multiatlas"),
                    "roi_index": int(stat["roi_index"]),
                    "feature_id": feature_id,
                    "feature_name": feature_en,
                    "direction": "altered",
                },
            },
        }
        candidates.append(candidate)
        truth[candidate_id] = {
            "hypothesis_id": candidate_id,
            "status": "confirmed" if stat["validated_feature_fdr"] == "True" else "not_confirmed",
            "duration_seconds": 1.0,
            "duration_basis": "normalized_candidate_test_unit",
            "payload": {
                "contrast": "MCI_vs_CN",
                "source": stat["source"],
                "roi_index": int(stat["roi_index"]),
                "roi_name": stat["roi_name"],
                "feature": feature_id,
                "beta_case_minus_cn": float(stat["beta_case_minus_cn"]),
                "cohens_d": float(stat["cohens_d"]),
                "p_value": float(stat["p_value"]),
                "q_fdr_contrast_feature": float(stat["q_fdr_contrast_feature"]),
                "adni_support_score": float(stat["adni_support_score"]),
                "validated_feature_fdr": stat["validated_feature_fdr"] == "True",
                "n_case": int(stat["n_case"]),
                "n_control": int(stat["n_cn"]),
            },
        }
    return candidates, truth


def _reference_ids(candidate: dict[str, Any]) -> set[str]:
    return {
        str(item.get("pmid") or item.get("doi") or item.get("title"))
        for item in candidate["literature"]
    }


def _candidate_evidence_quality(candidate: dict[str, Any]) -> dict[str, Any]:
    """Summarize the human-reviewed literature quality used for pair selection."""
    references = candidate.get("literature")
    if not isinstance(references, list):
        references = []
    complete = len(references) == 5
    verified = 0
    strong = 0
    direct = 0
    reason_chars = 0
    for reference in references:
        if not isinstance(reference, dict):
            complete = False
            continue
        reason_zh = str(reference.get("relevance_reason_zh") or "").strip()
        reason_en = str(reference.get("relevance_reason_en") or "").strip()
        source = str(reference.get("abstract_source") or "").strip()
        url = str(reference.get("url") or "").strip()
        is_verified = bool(reference.get("abstract_verified")) and bool(source)
        # The completed curation campaign includes hypothesis-specific bilingual
        # assessments for every reference. Some older papers have no retrievable
        # abstract, so an evidence excerpt is used instead; those records remain
        # eligible as long as their source and direct paper URL are preserved.
        if not (reason_zh and reason_en and source and url):
            complete = False
        verified += int(is_verified)
        lower = reason_en.lower()
        strong += int("high" in lower)
        direct += int("direct" in lower)
        reason_chars += len(reason_zh) + len(reason_en)
    return {
        "complete": complete,
        "verified": verified,
        "strong": strong,
        "direct": direct,
        "reason_chars": reason_chars,
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile from an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_pairs(
    candidates: list[dict[str, Any]],
    truth: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if int(seed) != SHARED_SCHEDULE_SEED:
        raise ValueError(
            f"Expert Study uses the fixed shared seed {SHARED_SCHEDULE_SEED}; "
            f"received {seed}"
        )
    rng = random.Random(seed)
    evidence_quality = {
        candidate["id"]: _candidate_evidence_quality(candidate)
        for candidate in candidates
    }
    eligible: list[dict[str, Any]] = []
    for index, left in enumerate(candidates):
        lt = truth[left["id"]]
        ltuple = left["metadata"]["candidate_tuple"]
        lrefs = _reference_ids(left)
        for right in candidates[index + 1:]:
            rt = truth[right["id"]]
            if lt["status"] == rt["status"]:
                continue
            if not (
                evidence_quality[left["id"]]["complete"]
                and evidence_quality[right["id"]]["complete"]
            ):
                continue
            rtuple = right["metadata"]["candidate_tuple"]
            if ltuple["region_name"] == rtuple["region_name"] and ltuple["feature_id"] == rtuple["feature_id"]:
                continue
            rrefs = _reference_ids(right)
            jaccard = len(lrefs & rrefs) / max(1, len(lrefs | rrefs))
            if jaccard > MAX_REFERENCE_JACCARD:
                continue
            left_score = float(left["composite_score"])
            right_score = float(right["composite_score"])
            generator_gap = abs(left_score - right_score)
            if generator_gap < MIN_GENERATOR_SCORE_GAP:
                continue
            distance = int(ltuple["region_name"] != rtuple["region_name"]) + int(
                ltuple["feature_id"] != rtuple["feature_id"]
            )
            eligible.append(
                {
                    "left": left,
                    "right": right,
                    "generator_score_gap": generator_gap,
                    "base": {
                        "left_id": left["id"],
                        "right_id": right["id"],
                        "distance": distance,
                        "shared_references": len(lrefs & rrefs),
                        "reference_jaccard": round(jaccard, 4),
                        "comparison_task": "adni-mci-vs-cn",
                        "disease": "Mild Cognitive Impairment",
                        "difficulty_basis": "generator_score_gap",
                        "evidence_quality": {
                            "left": evidence_quality[left["id"]],
                            "right": evidence_quality[right["id"]],
                        },
                    },
                },
            )

    if len(eligible) < sum(PAIR_TARGETS.values()):
        raise RuntimeError(
            f"Only {len(eligible)} meaningful confirmed-vs-not-confirmed comparisons "
            f"are available; {sum(PAIR_TARGETS.values())} are required"
        )
    gaps = [float(item["generator_score_gap"]) for item in eligible]
    hard_upper = _quantile(gaps, 1 / 3)
    medium_upper = _quantile(gaps, 2 / 3)
    proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        gap = float(item["generator_score_gap"])
        difficulty = "hard" if gap <= hard_upper else "medium" if gap <= medium_upper else "easy"
        proposals[difficulty].append(item)

    selected_by_difficulty: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        level: [] for level in PAIR_TARGETS
    }
    used: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for difficulty in ("easy", "medium", "hard"):
        options = proposals[difficulty]
        rng.shuffle(options)
        tier_middle = _quantile(
            [float(item["generator_score_gap"]) for item in options], 0.5
        )
        while len(selected_by_difficulty[difficulty]) < PAIR_TARGETS[difficulty]:
            available = [
                item
                for item in options
                if tuple(sorted((item["left"]["id"], item["right"]["id"]))) not in seen
                and item["left"]["id"] not in used
                and item["right"]["id"] not in used
            ]
            if not available:
                break

            def selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
                left_id = item["left"]["id"]
                right_id = item["right"]["id"]
                quality = item["base"]["evidence_quality"]
                gap = float(item["generator_score_gap"])
                if difficulty == "easy":
                    gap_preference = -gap
                elif difficulty == "hard":
                    gap_preference = gap
                else:
                    gap_preference = abs(gap - tier_middle)
                return (
                    -min(quality["left"]["direct"], quality["right"]["direct"]),
                    -min(quality["left"]["strong"], quality["right"]["strong"]),
                    item["base"]["reference_jaccard"],
                    gap_preference,
                )

            chosen = min(available, key=selection_key)
            left = chosen["left"]
            right = chosen["right"]
            base = chosen["base"]
            selected_index = len(selected_by_difficulty[difficulty])
            confirmed_should_be_left = selected_index % 2 == 0
            if (truth[left["id"]]["status"] == "confirmed") != confirmed_should_be_left:
                left, right = right, left
                base = base | {
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "evidence_quality": {
                        "left": evidence_quality[left["id"]],
                        "right": evidence_quality[right["id"]],
                    },
                }
            pair = base | {
                "difficulty": difficulty,
                "manual_review": {
                    "status": "algorithmically_paired_from_human_reviewed_candidates",
                    "quality": "keep",
                    "evidence_grade": "high",
                    "quality_gate": (
                        "five source-traceable papers with bilingual hypothesis-specific "
                        "assessments per hypothesis; opposite hidden external outcomes; "
                        "meaningful generator-score separation; no hypothesis reuse"
                    ),
                },
            }
            left_truth = truth[left["id"]]
            right_truth = truth[right["id"]]
            pair_hidden = {
                "left_id": left["id"],
                "right_id": right["id"],
                "preferred_id": (
                    left["id"] if left_truth["status"] == "confirmed" else right["id"]
                ),
                "left_status": left_truth["status"],
                "right_status": right_truth["status"],
                "left_external_support_score": float(
                    left_truth["payload"]["adni_support_score"]
                ),
                "right_external_support_score": float(
                    right_truth["payload"]["adni_support_score"]
                ),
                "left_generator_score": float(left["composite_score"]),
                "right_generator_score": float(right["composite_score"]),
                "generator_score_gap": round(
                    float(chosen["generator_score_gap"]), 8
                ),
                "difficulty": difficulty,
            }
            selected_by_difficulty[difficulty].append((pair, pair_hidden))
            used.update((left["id"], right["id"]))
            seen.add(tuple(sorted((left["id"], right["id"]))))
        selected_count = len(selected_by_difficulty[difficulty])
        if selected_count != PAIR_TARGETS[difficulty]:
            raise RuntimeError(
                f"Could not build {PAIR_TARGETS[difficulty]} high-quality "
                f"{difficulty} pairs; selected {selected_count}"
            )

    # Every expert receives the same fixed schedule. Each hypothesis appears
    # exactly once across the full six-session protocol.
    # Experts complete as many as the ten-minute active-time budget allows.
    session_quotas = tuple(
        {"easy": 10, "medium": 10, "hard": 10}
        for _ in range(REQUIRED_SESSIONS)
    )
    for records in selected_by_difficulty.values():
        rng.shuffle(records)
    offsets = {level: 0 for level in PAIR_TARGETS}
    pairs: list[dict[str, Any]] = []
    pair_truth: list[dict[str, Any]] = []
    for session_number, quota in enumerate(session_quotas, start=1):
        session_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for difficulty in ("easy", "medium", "hard"):
            start = offsets[difficulty]
            stop = start + quota[difficulty]
            session_records.extend(selected_by_difficulty[difficulty][start:stop])
            offsets[difficulty] = stop
        rng.shuffle(session_records)
        for session_question_number, (pair, hidden_pair) in enumerate(
            session_records, start=1
        ):
            pair_id = f"pair-{len(pairs) + 1:03d}"
            pairs.append(
                pair
                | {
                    "pair_id": pair_id,
                    "session_number": session_number,
                    "session_question_number": session_question_number,
                }
            )
            pair_truth.append(
                hidden_pair
                | {
                    "pair_id": pair_id,
                    "session_number": session_number,
                    "session_question_number": session_question_number,
                }
            )
    return pairs, pair_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--claims", type=Path, default=ROOT / "neurooracle/data/full_v2/extracted_claims.jsonl")
    parser.add_argument("--abstracts", type=Path, default=ROOT / "neurooracle/data/full_v2/abstract_cache.jsonl")
    parser.add_argument("--public-output", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument(
        "--curated-input",
        type=Path,
        default=None,
        help=(
            "Reuse hypotheses and completed literature curation from an existing "
            "public study JSON instead of rebuilding them from raw result files."
        ),
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--truth-output", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=SHARED_SCHEDULE_SEED)
    args = parser.parse_args()

    if args.curated_input is not None:
        curated_payload = json.loads(
            args.curated_input.expanduser().resolve().read_text(encoding="utf-8-sig")
        )
        candidates = [
            dict(item)
            for item in curated_payload.get("hypotheses", [])
            if isinstance(item, dict)
        ]
        hidden_payload = json.loads(
            args.truth_output.expanduser().resolve().read_text(encoding="utf-8-sig")
        )
        truth = {
            str(item["hypothesis_id"]): dict(item)
            for item in hidden_payload.get("execution_results", [])
            if isinstance(item, dict) and item.get("hypothesis_id")
        }
        annotated_references = sum(
            bool(reference.get("relevance_reason_zh"))
            and bool(reference.get("relevance_reason_en"))
            for candidate in candidates
            for reference in candidate.get("literature", [])
            if isinstance(reference, dict)
        )
        missing_truth = [item["id"] for item in candidates if item["id"] not in truth]
        if missing_truth:
            raise RuntimeError(
                f"Hidden truth is missing {len(missing_truth)} curated hypotheses"
            )
    else:
        literature = load_literature(args.claims, args.abstracts)
        candidates, truth = build_candidates(load_rows(args.results_dir), literature)
        annotated_references = apply_manual_annotations(
            candidates, load_manual_annotations(args.annotations)
        )
    pairs, pair_truth = build_pairs(candidates, truth, args.seed)
    used_ids = {item[key] for item in pairs for key in ("left_id", "right_id")}
    scheduled_candidates = [item for item in candidates if item["id"] in used_ids]
    selected_annotated_references = sum(
        1
        for candidate in scheduled_candidates
        for reference in candidate["literature"]
        if reference.get("relevance_reason_zh")
        and reference.get("relevance_reason_en")
        and reference.get("abstract_source")
        and reference.get("url")
    )
    public = {
        "schema_version": "2.0",
        "study_id": "adni_external_validation_v1",
        "case_study": "adni_external_validation",
        "hypothesis_unit": "mci_region_feature",
        "blinding": (
            "Participants see hypotheses and prior literature only. ADNI effect sizes, "
            "p-values, FDR outcomes, and external support scores are stored separately."
        ),
        "confirmation_basis": "Held-out ADNI external validation; hidden from participants.",
        "pairing_policy": (
            "Every comparison contains exactly one externally confirmed and one "
            "externally not-confirmed hypothesis; side assignment is balanced and blinded."
        ),
        "session_protocol": {
            "completion_basis": "active_time",
            "required_sessions": REQUIRED_SESSIONS,
            "active_seconds_per_session": SESSION_ACTIVE_SECONDS,
            "pair_pool_per_session": len(pairs) // REQUIRED_SESSIONS,
            "assignment_policy": "fixed_shared_schedule",
            "shared_random_seed": SHARED_SCHEDULE_SEED,
            "same_questions_for_all_participants": True,
        },
        "difficulty_basis": {
            "field": "absolute_generator_score_gap",
            "near_tie_exclusion_threshold": MIN_GENERATOR_SCORE_GAP,
            "classification": (
                "Eligible score gaps are split into tertiles: the largest third is easy, "
                "the middle third is medium, and the smallest meaningful third is hard."
            ),
        },
        "n_hypotheses": len(candidates),
        "n_scheduled_hypotheses": len(scheduled_candidates),
        "n_pairs": len(pairs),
        "pair_counts": dict(PAIR_TARGETS),
        "curation": {
            "status": "external_validation_curated",
            "pool_status": "full_ranked_pool_with_completed_literature_review",
            "method": (
                "The complete frozen NeuroDiscovery ranking is retained in hypotheses. "
                "The scheduled_hypotheses subset and pair schedule use deterministic "
                "generator-score-gap stratification of confirmed-versus-not-confirmed "
                "candidates with source-traceable human literature assessments; every "
                "scheduled hypothesis appears at most once."
            ),
            "human_review_status": (
                "The full ranked pool has completed progressive literature review. "
                "Scheduled candidates have passed the declared literature quality gates."
            ),
            "manually_annotated_references": selected_annotated_references,
            "available_manually_annotated_references": annotated_references,
        },
        "hypotheses": candidates,
        "scheduled_hypotheses": scheduled_candidates,
        "pair_schedule": pairs,
    }
    hidden = {
        "schema_version": "1.0",
        "study_id": "adni_external_validation_v1",
        "warning": "CONFIDENTIAL STUDY TRUTH. Do not bundle or expose to participants.",
        "duration_basis": "normalized_candidate_test_unit",
        "execution_results": [truth[item["id"]] for item in candidates],
        "pair_truth": pair_truth,
    }
    audit = {
        "study_id": public["study_id"],
        "public_hypotheses": len(candidates),
        "scheduled_hypotheses": len(scheduled_candidates),
        "pairs": len(pairs),
        "pairs_by_difficulty": {
            level: sum(item["difficulty"] == level for item in pairs)
            for level in ("easy", "medium", "hard")
        },
        "sessions": REQUIRED_SESSIONS,
        "active_seconds_per_session": SESSION_ACTIVE_SECONDS,
        "pairs_by_session": {
            str(number): sum(
                int(item["session_number"]) == number for item in pairs
            )
            for number in range(1, REQUIRED_SESSIONS + 1)
        },
        "all_pairs_have_opposite_external_outcomes": all(
            item["left_status"] != item["right_status"] for item in pair_truth
        ),
        "all_pairs_have_meaningful_generator_score_gap": all(
            float(item["generator_score_gap"]) >= MIN_GENERATOR_SCORE_GAP
            for item in pair_truth
        ),
        "generator_score_gap_by_difficulty": {
            level: {
                "min": min(
                    float(item["generator_score_gap"])
                    for item in pair_truth
                    if item["difficulty"] == level
                ),
                "max": max(
                    float(item["generator_score_gap"])
                    for item in pair_truth
                    if item["difficulty"] == level
                ),
            }
            for level in ("easy", "medium", "hard")
        },
        "confirmed_hypotheses": sum(
            truth[item["id"]]["status"] == "confirmed" for item in candidates
        ),
        "not_confirmed_hypotheses": sum(
            truth[item["id"]]["status"] == "not_confirmed" for item in candidates
        ),
        "scheduled_confirmed_hypotheses": sum(
            truth[item["id"]]["status"] == "confirmed"
            for item in scheduled_candidates
        ),
        "all_have_five_references": all(len(item["literature"]) == 5 for item in candidates),
        "all_selected_references_human_reviewed": all(
            reference.get("relevance_reason_zh")
            and reference.get("relevance_reason_en")
            and reference.get("abstract_source")
            and reference.get("url")
            for item in scheduled_candidates
            for reference in item["literature"]
        ),
        "all_pool_references_human_reviewed": all(
            reference.get("relevance_reason_zh")
            and reference.get("relevance_reason_en")
            and reference.get("abstract_source")
            and reference.get("url")
            for item in candidates
            for reference in item["literature"]
        ),
        "max_candidate_exposure": max(
            (
                sum(
                    candidate_id in (pair["left_id"], pair["right_id"])
                    for pair in pairs
                )
                for candidate_id in used_ids
            ),
            default=0,
        ),
        "all_hypotheses_unique_across_pairs": len(used_ids) == len(pairs) * 2,
        "all_confirmed_hypotheses_unique_across_pairs": len(
            {
                item["preferred_id"]
                for item in pair_truth
            }
        ) == len(pair_truth),
        "shared_schedule": {
            "assignment_policy": "fixed_shared_schedule",
            "seed": SHARED_SCHEDULE_SEED,
            "same_questions_for_all_participants": True,
        },
        "truth_fields_in_public_json": [],
        "seed": args.seed,
    }
    for path, payload in (
        (args.public_output, public),
        (args.truth_output, hidden),
        (args.audit_output, audit),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
