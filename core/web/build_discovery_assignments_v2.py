"""Deterministically deal the frozen v7 mixed panel to ten experts.

All four calibration cards receive three reviewers.  To retain exactly
10 experts x 10 cards = 100 review slots, supported cards packet-29 and
packet-30 receive two reviewers and every other card receives three.
"""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v7.json"
PACK_SHA256 = "787094275b59cc037dc5633d1bd5cb6695ce9666ab9851085c49c59a0c528568"
TARGET = HERE / "cs1_discovery_assignments_v2.json"
EXPERTS = [f"P{index:02d}" for index in range(1, 11)]
CARDS_PER_EXPERT = 10
TWO_REVIEWER_CARDS = {"packet-29", "packet-30"}

RULE_ZH = (
    "34 份材料共 100 人次 = 10 位专家 × 每人 10 份。4 份校准材料与其余 28 份材料各由 3 位专家评审；"
    "发现层 packet-29 与 packet-30 各由 2 位专家评审。卡位多重集按材料编号升序轮流分配给 P01–P10；"
    "同一材料的实例连续且不超过 3 个，因此必然落到不同专家。分配完全确定，不使用随机。"
)


def build(raw=None):
    raw = raw if raw is not None else PACK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PACK_SHA256:
        raise ValueError("The v7 pack changed; build a separately reviewed assignment version.")
    pack = json.loads(raw)
    cards = [card["id"] for card in pack["cards"]]
    if cards != [f"packet-{rank:02d}" for rank in range(1, 35)]:
        raise ValueError("Expected the ordered 34-card v7 panel.")
    calibration = {card["id"] for card in pack["cards"]
                   if card["pre"]["study_context"]["layer"] == "calibration"}
    if calibration != {"packet-31", "packet-32", "packet-33", "packet-34"}:
        raise ValueError("Unexpected calibration-card positions.")
    slots = [card for card in cards for _ in range(2 if card in TWO_REVIEWER_CARDS else 3)]
    if len(slots) != len(EXPERTS) * CARDS_PER_EXPERT:
        raise ValueError("Slot count does not match the 10 x 10 design.")
    assignments = {expert: [] for expert in EXPERTS}
    for position, card in enumerate(slots):
        assignments[EXPERTS[position % len(EXPERTS)]].append(card)
    coverage = {card: 0 for card in cards}
    for expert, dealt in assignments.items():
        if len(dealt) != CARDS_PER_EXPERT or len(set(dealt)) != CARDS_PER_EXPERT:
            raise ValueError(f"{expert} does not hold ten distinct cards.")
        for card in dealt:
            coverage[card] += 1
    for card in cards:
        expected = 2 if card in TWO_REVIEWER_CARDS else 3
        if coverage[card] != expected:
            raise ValueError(f"{card} has {coverage[card]} reviewers, expected {expected}.")
    if any(coverage[card] != 3 for card in calibration):
        raise ValueError("Every calibration case must receive three reviewers.")
    return {
        "version": 2,
        "pack_id": pack["pack_id"],
        "pack_sha256": PACK_SHA256,
        "rule_zh": RULE_ZH,
        "cards_per_expert": CARDS_PER_EXPERT,
        "reviewers_per_card": {"default": 3, "two_reviewer_cards": sorted(TWO_REVIEWER_CARDS)},
        "experts": assignments,
        "coverage": coverage,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v2 assignment table already exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "reviewer_slots": sum(result["coverage"].values())}, indent=2))


if __name__ == "__main__":
    main()
