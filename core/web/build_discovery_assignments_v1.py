"""Deal the frozen v6 expert panel to ten experts: deterministic round-robin.

Study design (2026-09-18): 34 materials, 10 experts x 10 materials = 100 review
slots. Ranks 1-32 (the panel order, strongest effects first) get 3 reviewers
each; ranks 33-34 get 2 (96 + 4 = 100). The slot multiset in ascending rank
order is dealt one by one to P01..P10, so the repeated instances of one card
sit exactly 10 positions apart and always land on different experts. The
dealing is deterministic and uses no randomness; experts receive their cards in
ascending rank order and the study server shuffles per session as before.

Coverage priority follows the effect ranking: the two weakest cards are the
ones with only two reviewers. This file discloses that rule to reviewers.

This script only reads the frozen v6 pack; it never runs experiments.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v6.json"
PACK_SHA256 = "1079b398a43a1f842781348464bc7aa892f7063480a5e8db41295f97172e6681"
TARGET = HERE / "cs1_discovery_assignments_v1.json"

EXPERTS = [f"P{index:02d}" for index in range(1, 11)]
CARDS_PER_EXPERT = 10
THREE_REVIEWER_RANKS = 32

RULE_ZH = (
    "34 份材料按效应排名（packet-01 最强至 packet-34 最弱）；rank 1–32 各由 3 位专家评审，"
    "rank 33–34 各由 2 位评审，共 100 人次 = 10 位专家 × 每人 10 份。"
    "构造方法：把卡位多重集按排名升序依次轮流分配给 P01–P10；同一材料的多个实例相隔 10 个位置，"
    "必然落在不同专家；每位专家恰好 10 份且互不重复。分配完全确定，不使用随机。"
    "覆盖优先级随效应排名：最弱的两份材料为仅 2 人评审的材料。"
)


def build(raw=None):
    raw = raw if raw is not None else PACK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PACK_SHA256:
        raise ValueError("The v6 pack changed; build a separately reviewed new assignment version.")
    pack = json.loads(raw)
    cards = [card["id"] for card in pack["cards"]]
    if len(cards) != 34 or len(set(cards)) != 34:
        raise ValueError("Expected the frozen 34-card v6 panel.")
    slots = [rank for rank in range(1, 35) for _ in range(3 if rank <= THREE_REVIEWER_RANKS else 2)]
    if len(slots) != len(EXPERTS) * CARDS_PER_EXPERT:
        raise ValueError("Slot count does not match the 10 experts x 10 cards design.")
    assignments = {expert: [] for expert in EXPERTS}
    for position, rank in enumerate(slots):
        assignments[EXPERTS[position % len(EXPERTS)]].append(cards[rank - 1])
    coverage = {card: 0 for card in cards}
    for expert, dealt in assignments.items():
        if len(dealt) != CARDS_PER_EXPERT or len(set(dealt)) != CARDS_PER_EXPERT:
            raise ValueError(f"{expert} does not hold {CARDS_PER_EXPERT} distinct cards.")
        for card in dealt:
            coverage[card] += 1
    for rank, card in enumerate(cards, start=1):
        expected = 3 if rank <= THREE_REVIEWER_RANKS else 2
        if coverage[card] != expected:
            raise ValueError(f"{card} (rank {rank}) has {coverage[card]} reviewers, expected {expected}.")
    return {
        "version": 1,
        "pack_id": pack["pack_id"],
        "pack_sha256": PACK_SHA256,
        "rule_zh": RULE_ZH,
        "cards_per_expert": CARDS_PER_EXPERT,
        "reviewers_per_card": {"ranks_1_32": 3, "ranks_33_34": 2},
        "experts": assignments,
        "coverage": coverage,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different assignment table already exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({
        "target": TARGET.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "experts": len(result["experts"]),
        "cards_per_expert": CARDS_PER_EXPERT,
        "reviewer_slots": sum(result["coverage"].values()),
    }, indent=2))


if __name__ == "__main__":
    main()
