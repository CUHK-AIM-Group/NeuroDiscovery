"""Rebalance assignments only; frozen questions, scientific data and timing stay intact."""
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ROOT / "core/web/study_materials"
RANKING = ROOT / "neurooracle/data/user_study"
HE1_REVISION = "he1-topic30-v6"
HE2_POLICY = "prebalanced-position-thirds-v1"
HE1_TARGET = MATERIALS / "cs1_discovery_assignments_v6.json"
HE2_TARGET = RANKING / "case1_tcp_expert_pair_assignments_v3.json"


def frozen(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen source changed: {path.name}")
    return json.loads(raw)


def build_he1():
    pack = frozen(MATERIALS / "cs1_discovery_pilot_v10.json", "2400d3d7bf1c701c7a9f0cef47420b43d6da08b0a0b37b6e3cd007d083828e91")
    previous = frozen(MATERIALS / "cs1_discovery_assignments_v5.json", "3f0fec4cc881c1a0481d2073f1600757962885715b7dd10895fd3d1c8260257f")
    table = copy.deepcopy(previous)
    topics = ("跨诊断脑连接", "影像遗传学", "预后研究")
    pools = {topic: [c["id"] for c in pack["cards"] if c["pre"]["topic"] == topic] for topic in topics}
    assert [len(pools[t]) for t in topics] == [20, 7, 7]
    cursor = Counter()
    for index, expert in enumerate(sorted(table["experts"])):
        quotas = (3, 3 if index % 2 == 0 else 4, 4 if index % 2 == 0 else 3)
        dealt = []
        for topic, count in zip(topics, quotas):
            pool = pools[topic]
            dealt.extend(pool[(cursor[topic] + n) % len(pool)] for n in range(count))
            cursor[topic] += count
        assert len(dealt) == len(set(dealt)) == 10
        table["experts"][expert] = dealt
    table["coverage"] = dict(Counter(card for dealt in table["experts"].values() for card in dealt))
    assert set(table["coverage"]) == {c["id"] for c in pack["cards"]}
    table["allocation_revision"] = HE1_REVISION
    table["topic_slots"] = dict(cursor)
    table["rule_zh"] = "每位专家评审 10 份材料，其中 3 份为跨诊断脑连接（30%）；其余为影像遗传学 3 份与预后 4 份，或影像遗传学 4 份与预后 3 份，按编号交替分配。保留原有展示顺序逻辑，不新增题序均衡。"
    table["rule_en"] = "Each expert reviews 10 cases: 3 transdiagnostic connectivity cases (30%), plus 3 imaging-genetics and 4 prognosis cases or 4 imaging-genetics and 3 prognosis cases, alternating by assignment. The existing presentation-order logic is unchanged."
    return table


def _match_band(remaining, seed):
    """One slot per pair and four slots per expert-session; deterministic matching."""
    rng = random.Random(seed)
    pair_order = sorted(remaining)
    rng.shuffle(pair_order)
    choices = {}
    for pair in pair_order:
        slots = [(session, slot) for session in sorted(remaining[pair]) for slot in range(4)]
        rng.shuffle(slots)
        choices[pair] = slots
    owner = {}

    def augment(pair, visited):
        for slot in choices[pair]:
            if slot in visited:
                continue
            visited.add(slot)
            if slot not in owner or augment(owner[slot], visited):
                owner[slot] = pair
                return True
        return False

    for pair in pair_order:
        if not augment(pair, set()):
            raise ValueError("No balanced position matching; do not publish partial assignments")
    return {pair: slot[0] for slot, pair in owner.items()}


def build_he2():
    bank = frozen(RANKING / "case1_tcp_expert_study_v2.json", "ff9b1d51e469f64bc336d6035271992f1ad1a8eed94750255e0e7e8d5986393d")
    previous = frozen(RANKING / "case1_tcp_expert_pair_assignments_v2.json", "39416993fc435eaf00f385165f1f765abf00dfa64ca22c21244cea8b2f5fe72c")
    table = copy.deepcopy(previous)
    difficulty = {p["pair_id"]: p["difficulty"] for p in bank["pair_schedule"]}
    remaining = defaultdict(set)
    for expert, rounds in previous["sessions"].items():
        for number, pairs in rounds.items():
            for pair in pairs:
                remaining[pair].add((expert, number))
    assert len(remaining) == 240 and all(len(v) == 3 for v in remaining.values())
    ordered = {expert: {number: [] for number in rounds} for expert, rounds in previous["sessions"].items()}
    for band in range(3):
        selected = _match_band(remaining, 20260920 + band)
        by_session = defaultdict(list)
        for pair, session in selected.items():
            remaining[pair].remove(session)
            by_session[session].append(pair)
        for (expert, number), pairs in sorted(by_session.items()):
            assert len(pairs) == 4
            # Interleave available difficulty levels within each frozen four-item band.
            pools = {d: sorted(p for p in pairs if difficulty[p] == d) for d in ("easy", "medium", "hard")}
            start = (int(expert[1:]) + int(number) + band) % 3
            tiers = ("easy", "medium", "hard")
            while any(pools.values()):
                for offset in range(3):
                    pool = pools[tiers[(start + offset) % 3]]
                    if pool:
                        ordered[expert][number].append(pool.pop(0))
    assert all(not value for value in remaining.values())
    positions = defaultdict(list)
    for expert, rounds in ordered.items():
        for number, pairs in rounds.items():
            assert len(pairs) == 12 and set(pairs) == set(previous["sessions"][expert][number])
            for index, pair in enumerate(pairs):
                positions[pair].append(index // 4)
    assert all(sorted(bands) == [0, 1, 2] for bands in positions.values())
    table.update(version=3, order_policy=HE2_POLICY, sessions=ordered,
                 position_bands={"early": [1, 4], "middle": [5, 8], "late": [9, 12]},
                 parent_assignment_sha256="39416993fc435eaf00f385165f1f765abf00dfa64ca22c21244cea8b2f5fe72c")
    table["rule_zh"] = "240 道比较题仍各分配给 3 位不同专家，每人 72 道，分 6 次、每次 12 道。保留各专家每次的题目、易中难各 4 道以及 10 分钟有效作答规则；只预先均衡轮内题序，让每道题在三位专家那里分别位于前段（1–4）、中段（5–8）和后段（9–12）。实际完成数量按回收答案统计。"
    table["rule_en"] = "The same 240 comparisons are assigned to three different experts each, with 72 per expert across six 12-item sessions. Session membership, four items per difficulty level and the 10-minute active-time limit are unchanged. Only order is prebalanced: each comparison appears once in positions 1–4, 5–8 and 9–12 across its three experts. Actual completion is counted from returned answers."
    return table


if __name__ == "__main__":
    # Emit an apply_patch document; never overwrite a frozen assignment version.
    outputs = {HE1_TARGET: build_he1(), HE2_TARGET: build_he2()}
    lines = ["*** Begin Patch"]
    for target, payload in outputs.items():
        raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if target.exists():
            if target.read_text(encoding="utf-8") != raw:
                raise ValueError(f"Different assignment release exists: {target}")
            continue
        lines.append("*** Add File: " + target.as_posix())
        lines.extend("+" + line for line in raw.splitlines())
    lines.append("*** End Patch")
    print("\n".join(lines))
