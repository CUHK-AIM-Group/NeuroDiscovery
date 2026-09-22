"""Extension pair shares v2 for the 240-pair v2 bank: three reviewers per pair.

Owner request (2026-09-19): the doubled bank (240 pairs) gives every pair three
expert reviewers; 240 x 3 = 720 slots = 10 experts x 72 pairs, keeping the
ten-minute rhythm at six sessions of 12 pairs per expert (one expert-hour).

Dealing (deterministic, no randomness): within each difficulty tier (80 pairs),
every pair appears three times consecutively and instances are dealt
round-robin to P01..P10, so the three instances of one pair always land on
different experts. Each expert receives 24 pairs per tier (4 easy + 4 medium +
4 hard per session, six sessions). Per-expert sessions split the dealt list by
position mod 6, preserving difficulty balance within every session. ALL preview
keeps the bank's own 6-session x 40-pair schedule for the organizer.

This script only reads the frozen v2 bank; it never runs experiments.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent.parent / "neurooracle" / "data" / "user_study"
BANK = HERE / "case1_tcp_expert_study_v2.json"
BANK_SHA256 = "ff9b1d51e469f64bc336d6035271992f1ad1a8eed94750255e0e7e8d5986393d"
TARGET = HERE / "case1_tcp_expert_pair_assignments_v2.json"

EXPERTS = [f"P{index:02d}" for index in range(1, 11)]
REVIEWERS_PER_PAIR = 3
PAIRS_PER_EXPERT = 72
SESSIONS_PER_EXPERT = 6
PAIRS_PER_SESSION = PAIRS_PER_EXPERT // SESSIONS_PER_EXPERT
DIFFICULTIES = ("easy", "medium", "hard")

RULE_ZH = (
    "240 道策展比较题每题由 3 位专家复评：按难度分层（易/中/难各 80 道），"
    "每题在同一层内连续出现 3 次后轮流分配给 P01–P10，每位专家每层 24 道、共 72 道"
    "（每次会话 12 道 × 6 次会话，约一小时）；同一题的 3 个实例必落在不同专家。"
    "分配完全确定、不使用随机。“全部 240 道 · 主持预览”保留题库自带的 6 次会话 × 40 道流程。"
)


def build(raw=None):
    raw = raw if raw is not None else BANK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA256:
        raise ValueError("The v2 bank changed; build a separately reviewed new assignment version.")
    bank = json.loads(raw)
    schedule = bank.get("pair_schedule")
    if not isinstance(schedule, list) or len(schedule) != 240:
        raise ValueError("Expected the frozen 240-pair v2 bank.")
    assignments = {expert: [] for expert in EXPERTS}
    for difficulty in DIFFICULTIES:
        tier = [str(p["pair_id"]) for p in schedule if str(p.get("difficulty") or "").lower() == difficulty]
        if len(tier) != 80:
            raise ValueError(f"Expected 80 {difficulty} pairs, found {len(tier)}.")
        slots = [pair_id for pair_id in tier for _ in range(REVIEWERS_PER_PAIR)]
        for position, pair_id in enumerate(slots):
            assignments[EXPERTS[position % len(EXPERTS)]].append(pair_id)
    sessions = {}
    coverage = {}
    for expert, dealt in assignments.items():
        if len(dealt) != PAIRS_PER_EXPERT:
            raise ValueError(f"{expert} holds {len(dealt)} pairs, expected {PAIRS_PER_EXPERT}.")
        if len(set(dealt)) != len(dealt):
            raise ValueError(f"{expert} holds a duplicated pair.")
        per_session = {str(n): [] for n in range(1, SESSIONS_PER_EXPERT + 1)}
        for position, pair_id in enumerate(dealt):
            per_session[str(position % SESSIONS_PER_EXPERT + 1)].append(pair_id)
        sessions[expert] = per_session
        for pair_id in dealt:
            coverage[pair_id] = coverage.get(pair_id, 0) + 1
    if set(coverage.values()) != {REVIEWERS_PER_PAIR}:
        raise ValueError("Every pair must have exactly three reviewers.")
    return {
        "version": 2,
        "candidate_bank": BANK.name,
        "candidate_manifest_sha256": BANK_SHA256,
        "rule_zh": RULE_ZH,
        "pairs_per_expert": PAIRS_PER_EXPERT,
        "reviewers_per_pair": REVIEWERS_PER_PAIR,
        "sessions_per_expert": SESSIONS_PER_EXPERT,
        "pairs_per_session": PAIRS_PER_SESSION,
        "experts": assignments,
        "sessions": sessions,
        "coverage": coverage,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v2 assignment table already exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "experts": len(result["experts"]), "pairs_per_expert": PAIRS_PER_EXPERT}, indent=2))


if __name__ == "__main__":
    main()
