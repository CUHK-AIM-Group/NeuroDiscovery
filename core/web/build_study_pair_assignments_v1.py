"""Deal the TCP extension study's 120 curated pairs into 10 expert shares.

Owner request (2026-09-18): the Expert Study (Extension) splits into 10 shares
like the discovery panel. The 120 curated comparison pairs are dealt to
P01..P10, 12 pairs each, balanced by difficulty: within each difficulty tier
(40 easy, 40 medium, 40 hard) the pairs in frozen schedule order are dealt
round-robin, so every expert receives exactly 4 easy, 4 medium and 4 hard
pairs. Dealing is deterministic, uses no randomness, and does not modify the
candidate bank. Each pair is covered by exactly one expert share; the
organizer keeps the full 6-session x 20-pair schedule via the ALL preview.

This script only reads the frozen candidate bank; it never runs experiments.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent.parent / "neurooracle" / "data" / "user_study"
BANK = HERE / "case1_tcp_external_expert_study_v1.json"
BANK_SHA256 = "e86341168faae4781df6a7f82c5688dd395936673d87dffe664b99f744915c57"
TARGET = HERE / "case1_tcp_expert_pair_assignments_v1.json"

EXPERTS = [f"P{index:02d}" for index in range(1, 11)]
PAIRS_PER_EXPERT = 12
DIFFICULTIES = ("easy", "medium", "hard")

RULE_ZH = (
    "120 道策展比较题按难度分层（易/中/难各 40 道）轮流分成 10 份："
    "每层内按既定顺序依次分配给 P01–P10，每位专家恰好 12 道（4 易、4 中、4 难）。"
    "每道题只归入一个专家份额；分配完全确定、不使用随机。"
    "“全部 120 道 · 主持预览”保留原 6 次会话 × 20 道完整流程，供组织者查看。"
)


def build(raw=None):
    raw = raw if raw is not None else BANK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA256:
        raise ValueError("The candidate bank changed; build a separately reviewed new assignment version.")
    bank = json.loads(raw)
    schedule = bank.get("pair_schedule")
    if not isinstance(schedule, list) or len(schedule) != 120:
        raise ValueError("Expected the frozen 120-pair curated schedule.")
    assignments = {expert: [] for expert in EXPERTS}
    for difficulty in DIFFICULTIES:
        tier = [pair for pair in schedule if str(pair.get("difficulty") or "").lower() == difficulty]
        if len(tier) != 40:
            raise ValueError(f"Expected 40 {difficulty} pairs, found {len(tier)}.")
        for position, pair in enumerate(tier):
            pair_id = str(pair.get("pair_id") or "").strip()
            if not pair_id:
                raise ValueError("A curated pair lacks pair_id.")
            assignments[EXPERTS[position % len(EXPERTS)]].append(pair_id)
    coverage = {}
    for expert, dealt in assignments.items():
        if len(dealt) != PAIRS_PER_EXPERT or len(set(dealt)) != PAIRS_PER_EXPERT:
            raise ValueError(f"{expert} does not hold {PAIRS_PER_EXPERT} distinct pairs.")
        for pair_id in dealt:
            coverage[pair_id] = coverage.get(pair_id, 0) + 1
    if sorted(coverage) != sorted(str(pair["pair_id"]) for pair in schedule) or set(coverage.values()) != {1}:
        raise ValueError("Every pair must belong to exactly one expert share.")
    return {
        "version": 1,
        "candidate_bank": BANK.name,
        "candidate_manifest_sha256": BANK_SHA256,
        "rule_zh": RULE_ZH,
        "pairs_per_expert": PAIRS_PER_EXPERT,
        "difficulty_per_expert": {name: 4 for name in DIFFICULTIES},
        "experts": assignments,
        "coverage": coverage,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different pair-assignment table already exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "experts": len(result["experts"]), "pairs_per_expert": PAIRS_PER_EXPERT}, indent=2))


if __name__ == "__main__":
    main()
