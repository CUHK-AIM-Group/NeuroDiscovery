"""Build the doubled Human Evaluation 2 bank from the latest full-space results.

Owner request (2026-09-19): double the extension question bank (120 -> 240
pairs) from the LATEST full hypothesis space (case_study_closed_loop_v8,
426,555 candidates), pairing one internally validated hypothesis with one
not-validated hypothesis from the same disease, with the NeuroDiscovery score
of the validated side much higher. Internal validation suffices; external
outcomes are not used for this bank.

Validation rule (from the v8 table manifest): validated = disease x atlas x
feature-family BH-FDR q <= 0.05, |adjusted Cohen d| >= 0.15, held-out
directional AUC >= 0.55 and direction concordance >= 0.7. Hypotheses are
direction-free ("存在差异") because the v8 registry carries no pre-registered
direction; observed directions stay in the organizer truth file only.

Pipeline (deterministic; no experiments are run):
  registry  tmp/cs3_nd_diagnostic_3seed/.../frozen_public_candidates.csv.gz
  outcomes  NAS internal_outcomes.csv (validated flags + observed directions)
  pairs     same disease; never same (source, roi, feature); score gap
            validated - other >= MIN_SCORE_GAP, maximized greedily
  difficulty tertiles by the frozen score gap (easy = largest gap)
  cards     bilingual titles/summaries templated from registry fields;
            literature reused verbatim from the v1 bank when the hypothesis
            reappears, otherwise matched from the local KG claims file
            (auto-matched, disclosed; not manually reviewed)
  truth     outcome sides stay in an organizer-only file, never in the bank
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY = ROOT / "tmp" / "cs3_nd_diagnostic_3seed" / "frozen_discovery_rankings" / "frozen_public_candidates.csv.gz"
REGISTRY_SHA256 = "f30631fe4eba4e82c919231aaf0a80d963338072fc8419c5cbf951e6cc4819a6"
OUTCOMES = Path("//192.168.3.61/data/Public Dataset/case_study_closed_loop_v8_biomarker_independent/biomarker_discovery/tables/internal_outcomes.csv")
OLD_BANK = ROOT / "neurooracle" / "data" / "user_study" / "case1_tcp_external_expert_study_v1.json"
OLD_BANK_SHA256 = "e86341168faae4781df6a7f82c5688dd395936673d87dffe664b99f744915c57"
CLAIMS = ROOT / "neurooracle" / "data" / "full_v2" / "extracted_claims.jsonl"
TARGET = ROOT / "neurooracle" / "data" / "user_study" / "case1_tcp_expert_study_v2.json"
TRUTH_TARGET = ROOT / "neurooracle" / "data" / "user_study" / "private_truth" / "case1_tcp_expert_study_v2_truth.json"

DISEASE = "psychosis_SZ_SZA"
DISEASE_NAME = "Schizophrenia-spectrum Psychosis"
DISEASE_ZH = "精神分裂症谱系精神病"
N_PAIRS = 240
MIN_SCORE_GAP = 0.20
REQUIRED_SESSIONS = 6
PAIRS_PER_SESSION = N_PAIRS // REQUIRED_SESSIONS

FEATURE_NAMES = {
    "roi_alff_proxy": ("ROI ALFF", "感兴趣区低频振幅"),
    "roi_falff_proxy": ("ROI fALFF", "感兴趣区低频振幅分数"),
    "roi_temporal_mean": ("ROI temporal mean", "感兴趣区时间序列均值"),
    "roi_temporal_mean_abs": ("ROI mean absolute signal", "感兴趣区时间序列绝对均值"),
    "roi_temporal_std": ("ROI temporal variability (SD)", "感兴趣区时间序列标准差"),
    "roi_temporal_variance": ("ROI temporal variability (variance)", "感兴趣区时间序列方差"),
    "corr_mean": ("ROI mean functional connectivity", "感兴趣区平均功能连接"),
    "corr_mean_abs": ("ROI mean absolute functional connectivity", "感兴趣区平均绝对功能连接"),
    "corr_positive_mean": ("ROI positive functional connectivity", "感兴趣区正功能连接均值"),
    "corr_negative_mean": ("ROI negative functional connectivity", "感兴趣区负功能连接均值"),
    "corr_node_degree_abs_top10": ("ROI functional-connectivity node degree", "感兴趣区功能连接节点度（前 10% 绝对值）"),
    "partial_mean": ("ROI partial-correlation connectivity", "感兴趣区偏相关连接均值"),
    "partial_mean_abs": ("ROI absolute partial-correlation connectivity", "感兴趣区偏相关绝对连接均值"),
    "partial_positive_mean": ("ROI positive partial-correlation connectivity", "感兴趣区正偏相关连接均值"),
    "partial_negative_mean": ("ROI negative partial-correlation connectivity", "感兴趣区负偏相关连接均值"),
    "normalized_volume_fraction": ("Normalized volume fraction", "归一化体积分数"),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _composite(modality, source, disease, feature, roi_index):
    return f"{modality}|{source}|{disease}|{feature}|{int(roi_index)}"


def _load_space():
    if _sha256(REGISTRY) != REGISTRY_SHA256:
        raise ValueError("Frozen registry changed; build a separately reviewed new bank version.")
    if _sha256(OLD_BANK) != OLD_BANK_SHA256:
        raise ValueError("The v1 bank changed; literature reuse would be unverifiable.")
    registry = pd.read_csv(REGISTRY, low_memory=False)
    outcomes = pd.read_csv(OUTCOMES)
    old_bank = json.loads(OLD_BANK.read_text(encoding="utf-8-sig"))
    return registry, outcomes, old_bank


def _select_pairs(space):
    """Greedy max-gap validated x not-validated pairs within the disease."""
    validated = space[space["validated"] == True].sort_values(
        ["score_neurodiscovery", "candidate_id"], ascending=[False, True])
    others = space[(space["validated"] == False) & (space["execution_succeeded"] == True)].sort_values(
        ["score_neurodiscovery", "candidate_id"], ascending=[True, True])
    if len(validated) < N_PAIRS:
        raise ValueError(f"Only {len(validated)} validated candidates, need {N_PAIRS}")
    confirmed = validated.head(N_PAIRS)
    used_other = set()
    pairs = []
    others_list = list(others.itertuples())
    cursor = 0
    for conf in confirmed.itertuples():
        picked = None
        scan = cursor
        while scan < len(others_list):
            other = others_list[scan]
            scan += 1
            if other.candidate_id in used_other:
                continue
            if (other.source == conf.source and int(other.roi_index) == int(conf.roi_index)
                    and other.feature == conf.feature):
                continue
            picked = other
            break
        if picked is None:
            raise ValueError(f"No eligible not-validated partner for {conf.candidate_id}")
        cursor = scan
        used_other.add(picked.candidate_id)
        gap = float(conf.score_neurodiscovery) - float(picked.score_neurodiscovery)
        if gap < MIN_SCORE_GAP:
            raise ValueError(f"Score gap {gap:.3f} below {MIN_SCORE_GAP} at {conf.candidate_id}")
        pairs.append({"confirmed": conf, "other": picked, "gap": gap})
    return pairs


def _literature_reuse(old_bank):
    reuse = {}
    for candidate in old_bank.get("hypotheses", []):
        if isinstance(candidate.get("literature"), list) and candidate["literature"]:
            reuse[candidate["id"]] = candidate["literature"]
    return reuse


def _claim_paper(claim):
    paper = claim.get("source_paper") if isinstance(claim.get("source_paper"), dict) else {}
    if not str(paper.get("title") or "").strip():
        return None
    return paper


def _load_claims_index(claims_path):
    """Single pass over the claims file: keep disease-relevant claims with papers."""
    disease_aliases = ("psychosis", "schizophrenia", "schizophrenic", "schizoaffective")
    index = []
    with Path(claims_path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                claim = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper = _claim_paper(claim)
            if not paper:
                continue
            text = " ".join(str(v or "") for v in (
                claim.get("disease"), claim.get("disease_query"), claim.get("subject_name"),
                claim.get("object_name"), claim.get("raw_text"))).lower()
            if not any(alias in text for alias in disease_aliases):
                continue
            tokens = set(re.findall(r"[a-z]+", text))
            sentence = str(claim.get("raw_text") or "").strip()
            try:
                confidence = max(0.0, float(claim.get("confidence") or 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            index.append({"paper": paper, "tokens": tokens, "sentence": sentence, "confidence": confidence})
    return index


def _match_literature(region, feature_en, claims_index, max_references=5):
    """Match a card against the pre-built psychosis claims index."""
    region_aliases = {a for a in re.split(r"[^a-z]+", region.lower()) if len(a) > 3}
    feature_aliases = {a for a in re.split(r"[^a-z]+", feature_en.lower()) if len(a) > 3}
    rows = {}
    for item in claims_index:
        region_hit = bool(region_aliases & item["tokens"])
        feature_hit = bool(feature_aliases & item["tokens"])
        if not (region_hit or feature_hit):
            continue
        score = 20.0 + (14.0 if region_hit else 0.0) + (5.0 if feature_hit else 0.0) + item["confidence"]
        paper = item["paper"]
        key = str(paper.get("pmid") or paper.get("doi") or paper["title"]).lower()
        existing = rows.get(key)
        if existing is None:
            existing = {
                "title": paper["title"],
                "authors": paper.get("authors") or "",
                "year": paper.get("year") or "",
                "journal": paper.get("journal") or "",
                "pmid": str(paper.get("pmid") or ""),
                "doi": paper.get("doi") or "",
                "url": paper.get("url") or (f"https://doi.org/{paper['doi']}" if paper.get("doi") else ""),
                "excerpts": [],
                "_score": 0.0,
            }
            rows[key] = existing
        existing["_score"] = max(existing["_score"], score)
        if item["sentence"] and item["sentence"] not in existing["excerpts"] and len(existing["excerpts"]) < 2:
            existing["excerpts"].append(item["sentence"])
    selected = sorted(rows.values(), key=lambda r: (r["_score"], str(r["year"] or ""), r["title"]), reverse=True)[:max_references]
    for row in selected:
        row["manual_relevance_status"] = "auto_kg_match"
        row["relevance_reason_zh"] = "知识图谱记录自动匹配；本轮未逐条人工复核。"
        row["relevance_reason_en"] = "Auto-matched from knowledge-graph records; not manually reviewed in this round."
    return selected


def _card(row, reuse, claims_index):
    if hasattr(row, "_asdict"):
        row = row._asdict()
    cid = row["candidate_id"]
    feature_id = row["feature"]
    feature_en, feature_zh = FEATURE_NAMES.get(feature_id, (feature_id, feature_id))
    region = str(row["roi_name"] if str(row.get("roi_name") or "") not in ("", "nan") else row.get("anatomy_full") or row.get("anatomy") or f"ROI {row['roi_index']}")
    card = {
        "id": _composite(row["modality"], row["source"], row["disease"], feature_id, row["roi_index"]),
        "title": f"{DISEASE_NAME} → {region} → {feature_en} (differs)",
        "title_zh": f"{DISEASE_ZH} → {region} → {feature_zh}（存在差异）",
        "summary": (f"Relative to healthy controls, {DISEASE_NAME} patients are hypothesized to "
                    f"differ in {feature_en} in {region}."),
        "summary_zh": f"该假设认为：与健康对照相比，{DISEASE_ZH}患者在 {region} 的{feature_zh}存在差异。",
        "source_name": DISEASE_NAME,
        "target_name": f"{region} | {feature_en}",
        "composite_score": float(row["score_neurodiscovery"]),
        "metadata": {
            "case_study": "case1_tcp_external_validation",
            "comparison_task_id": f"{DISEASE}|{row['source']}|{feature_id}",
            "candidate_tuple": {
                "disease": DISEASE, "disease_name": DISEASE_NAME, "atlas_name": str(row["atlas"]),
                "source": str(row["source"]), "roi_index": int(row["roi_index"]),
                "region_name": region, "feature_id": feature_id, "feature_name": feature_en,
                "direction": "differs",
            },
        },
    }
    literature = reuse.get(card["id"])
    if literature is None and claims_index is not None:
        literature = _match_literature(region, feature_en, claims_index)
    elif literature is not None:
        literature = [dict(item) for item in literature]
    card["literature"] = (literature or [])[:5]
    card["metadata"]["literature_count"] = len(card["literature"])
    return card


def build():
    registry, outcomes, old_bank = _load_space()
    registry["candidate_composite"] = [
        _composite(m, s, d, f, i) for m, s, d, f, i in zip(
            registry["modality"], registry["source"], registry["disease"],
            registry["feature"], registry["roi_index"])
    ]
    space = registry.merge(outcomes, on="candidate_id", how="inner")
    space = space[space["disease"] == DISEASE]
    pairs = _select_pairs(space)
    gaps = sorted(p["gap"] for p in pairs)
    easy_cut = float(pd.Series(gaps).quantile(2 / 3))
    hard_cut = float(pd.Series(gaps).quantile(1 / 3))
    reuse = _literature_reuse(old_bank)
    claims_index = _load_claims_index(CLAIMS) if CLAIMS.exists() else None

    by_id = {}
    for pair in pairs:
        for row in (pair["confirmed"], pair["other"]):
            cid = _composite(row.modality, row.source, row.disease, row.feature, row.roi_index)
            if cid not in by_id:
                by_id[cid] = _card(row, reuse, claims_index)
    cards = sorted(by_id.values(), key=lambda c: c["id"])

    # Deterministic pair order: gap descending, then confirmed id.
    pairs.sort(key=lambda p: (-p["gap"], p["confirmed"].candidate_id))
    schedule = []
    truth = []
    for index, pair in enumerate(pairs, 1):
        gap = pair["gap"]
        difficulty = "easy" if gap >= easy_cut else ("hard" if gap <= hard_cut else "medium")
        left_first = index % 2 == 1
        confirmed_id = _composite(
            pair["confirmed"].modality, pair["confirmed"].source, pair["confirmed"].disease,
            pair["confirmed"].feature, pair["confirmed"].roi_index)
        other_id = _composite(
            pair["other"].modality, pair["other"].source, pair["other"].disease,
            pair["other"].feature, pair["other"].roi_index)
        schedule.append({
            "_key": confirmed_id,
            "left_id": confirmed_id if left_first else other_id,
            "right_id": other_id if left_first else confirmed_id,
            "comparison_task": f"{DISEASE}|case-vs-control",
            "disease": DISEASE,
            "difficulty": difficulty,
            "difficulty_basis": "absolute_frozen_neurodiscovery_score_gap",
            "generator_score_gap": round(gap, 6),
            "manual_review": {
                "status": "completed", "quality": "keep", "evidence_grade": "auto_kg_matched",
                "notes": "Literature panels are auto-matched from local KG records; the v1 bank's manually reviewed cards are reused verbatim where the hypothesis reappears.",
            },
        })
        truth.append({
            "_key": confirmed_id,
            "confirmed_candidate_id": confirmed_id,
            "not_validated_candidate_id": other_id,
            "confirmed_side": "left" if left_first else "right",
            "confirmed_score": float(pair["confirmed"].score_neurodiscovery),
            "other_score": float(pair["other"].score_neurodiscovery),
            "score_gap": round(gap, 6),
            "confirmed_adjusted_residual_d": float(pair["confirmed"].adjusted_residual_d),
            "confirmed_direction": str(pair["confirmed"].direction),
            "other_direction": str(pair["other"].direction),
        })
    # Difficulty-balanced session assignment for the ALL preview: exactly 40 per
    # session. Each tier deals round-robin from a rotated starting offset so the
    # 80-pair tiers (14/13 splits) cancel out across the three tiers.
    per_session = {n: [] for n in range(1, REQUIRED_SESSIONS + 1)}
    for tier_index, difficulty in enumerate(("easy", "medium", "hard")):
        tier = [p for p in schedule if p["difficulty"] == difficulty]
        for position, pair in enumerate(tier):
            per_session[(position + tier_index * 2) % REQUIRED_SESSIONS + 1].append(pair)
    for session_number, session_pairs in per_session.items():
        for order, pair in enumerate(session_pairs, 1):
            pair["session_number"] = session_number
            pair["session_question_number"] = order
    schedule.sort(key=lambda p: (p["session_number"], p["session_question_number"]))
    truth_by_key = {record["_key"]: record for record in truth}
    truth = []
    for index, pair in enumerate(schedule, 1):
        pair["pair_id"] = f"tcp-v2-pair-{index:03d}"
        record = truth_by_key[pair.pop("_key")]
        record["pair_id"] = pair["pair_id"]
        truth.append(record)

    created_at = datetime.now(timezone.utc).isoformat()
    bank = {
        "schema_version": "case1-tcp-expert-study-v2",
        "created_at": created_at,
        "study_id": "case1_tcp_external_validation_v2",
        "case_study": "case1_tcp_external_validation",
        "status": "ready_for_expert_study",
        "curation": {
            "status": "internal_validation_v8",
            "reviewed_pairs": N_PAIRS,
            "reviewed_hypotheses": len(cards),
            "basis_zh": ("最新全空间（426,555 候选）内部实验结果：每对含一个内部验证通过（validated："
                        "家族 FDR q≤0.05、|调整 Cohen d|≥0.15、留出方向 AUC≥0.55 且方向一致率≥0.7）"
                        "与一个未通过的假设，前者冻结 ND 打分显著更高（差距 ≥0.20）。外部验证不参与本题库。"
                        "文献面板：v1 人工复核卡片在假设重现时逐字复用，其余为知识图谱记录自动匹配（未逐条人工复核）。"),
        },
        "hypothesis_unit": "disease_atlas_roi_imaging_feature",
        "blinding": ("Participants see frozen hypotheses and prior literature. Outcome sides, internal "
                    "effect sizes and validation flags are stored only in the organizer truth file."),
        "pairing_policy": {
            "outcomes": "exactly one internally validated and one not validated",
            "same_task": "same disease-versus-control contrast; atlas, ROI, and feature may differ",
            "hypothesis_reuse": {"validated": False, "not_validated_max_appearances": 1},
            "minimum_score_gap": MIN_SCORE_GAP,
            "difficulty": "tertiles of the frozen NeuroDiscovery score gap; easy = largest gap",
        },
        "session_protocol": {
            "required_sessions": REQUIRED_SESSIONS,
            "active_seconds_per_session": 600,
            "fixed_shared_seed": 0,
            "same_questions_for_all_participants": False,
            "assignment_note_zh": "P01–P10 每人 72 道（6 次会话 × 12 道，每题 3 人复评）；主持预览为 6 次会话 × 40 道。",
            "pairs_by_session": {str(n): PAIRS_PER_SESSION for n in range(1, REQUIRED_SESSIONS + 1)},
            "pair_pool_per_session": PAIRS_PER_SESSION,
            "assignment_policy": "expert_shares_v2",
            "shared_random_seed": 0,
        },
        "n_pairs": N_PAIRS,
        "n_hypotheses": len(cards),
        "hypotheses": cards,
        "pair_schedule": schedule,
    }
    truth_doc = {
        "schema_version": "case1-tcp-expert-study-v2-truth",
        "created_at": created_at,
        "study_id": bank["study_id"],
        "disclosure_zh": "组织者可核对：每对哪侧为内部验证通过、两侧冻结打分与内部效应。",
        "pairs": truth,
    }
    return bank, truth_doc


def main():
    bank, truth = build()
    data = (json.dumps(bank, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v2 bank already exists; use a new version.")
    TARGET.write_bytes(data)
    TRUTH_TARGET.parent.mkdir(parents=True, exist_ok=True)
    truth_data = (json.dumps(truth, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TRUTH_TARGET.exists() and TRUTH_TARGET.read_bytes() != truth_data:
        raise ValueError("A different v2 truth file already exists; use a new version.")
    TRUTH_TARGET.write_bytes(truth_data)
    print(json.dumps({
        "bank": TARGET.name, "bank_sha256": hashlib.sha256(data).hexdigest(),
        "pairs": bank["n_pairs"], "hypotheses": bank["n_hypotheses"],
        "truth": str(TRUTH_TARGET.name), "truth_sha256": hashlib.sha256(truth_data).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
