"""Build the v7 expert panel with a small, real calibration layer.

Owner decision (2026-09-19): keep the six 1--5 capability questions and the
34-card / 10-expert burden, but mix 30 externally supported findings with four
completed hypotheses that did not meet the frozen external-consistency rule.
Every card states up front that NeuroDiscovery proposed it and whether the
completed experiment was consistent with the hypothesis.

No experiment is rerun here.  The 30 supported cards are the first 30 cards of
the byte-frozen v6 strongest-first panel.  The four calibration cards are real
completed cards from the byte-frozen v5 panel, chosen to broaden endpoint and
diagnosis coverage.  Their non-support label means only that they did not meet
the predeclared external-consistency rule; it is not a low-quality label and is
not an expert-score target.
"""
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent / "study_materials"
V5 = HERE / "cs1_discovery_pilot_v5.json"
V6 = HERE / "cs1_discovery_pilot_v6.json"
TARGET = HERE / "cs1_discovery_pilot_v7.json"

V5_SHA256 = "d5648f5f4781936dddffc678be56a9be21827b4469ccd51f3527f2cf0fd2025b"
V6_SHA256 = "1079b398a43a1f842781348464bc7aa892f7063480a5e8db41295f97172e6681"
PACK_ID = "cs1-discovery-capabilities-20260919-v7"
PROTOCOL_VERSION = "cs1-complete-output-calibration-v7"

# These are completed, externally analysed hypotheses from v5.  They were not
# in the v6 passing pool because their UCLA joint-direction frequency did not
# reach the frozen 0.80 external-consistency rule.
CALIBRATION = [
    ("packet-03", "35b55341fad57c6ce37f"),  # Limbic--SomMot, ADHD + psychosis
    ("packet-04", "27981a1d1ecac5af5013"),  # DorsAttn--Vis, three diagnoses
    ("packet-06", "91db7740c66a41792684"),  # ROI104--SomMot, ADHD + bipolar
    ("packet-10", "a1f1a727dd523265d698"),  # Limbic--Vis, bipolar + psychosis
]


def load_checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen parent changed: {path.name}; review a new version instead.")
    return json.loads(raw)


def context(aligned):
    if aligned:
        return {
            "proposer": "NeuroDiscovery",
            "origin_zh": "本假设由 NeuroDiscovery 提出",
            "origin_en": "This hypothesis was proposed by NeuroDiscovery",
            "layer": "supported_discovery",
            "layer_zh": "外部支持发现",
            "layer_en": "Externally supported finding",
            "alignment": "consistent",
            "alignment_zh": "实验结果：与假设一致",
            "alignment_en": "Experimental result: consistent with the hypothesis",
            "detail_zh": "该材料达到预先固定的探索性外部一致条件；仍不等同于正式确证或全领域首次发现。",
            "detail_en": ("This case met the frozen exploratory external-consistency rule; "
                          "that is not the same as formal confirmation or a field-first discovery."),
        }
    return {
        "proposer": "NeuroDiscovery",
        "origin_zh": "本假设由 NeuroDiscovery 提出",
        "origin_en": "This hypothesis was proposed by NeuroDiscovery",
        "layer": "calibration",
        "layer_zh": "校准材料",
        "layer_en": "Calibration case",
        "alignment": "not_consistent",
        "alignment_zh": "实验结果：与假设不一致",
        "alignment_en": "Experimental result: not consistent with the hypothesis",
        "detail_zh": ("该材料未达到预先固定的外部一致条件。它是真实的不支持结果，"
                      "不代表实验执行失败，也不应仅因结果不支持而降低严谨性评分。"),
        "detail_en": ("This case did not meet the frozen external-consistency rule. It is a real "
                      "non-supporting result, not an execution failure, and should not receive a "
                      "lower rigor score merely because the result is non-supporting."),
    }


def build():
    v5 = load_checked(V5, V5_SHA256)
    v6 = load_checked(V6, V6_SHA256)
    if len(v6["cards"]) != 34 or len(v5["cards"]) != 10:
        raise ValueError("Unexpected frozen parent panel size.")

    pack = copy.deepcopy(v6)
    pack["pack_id"] = PACK_ID
    pack["protocol_version"] = PROTOCOL_VERSION
    pack["cards"] = []
    pack["organizer"]["mapping"] = []

    # Retain the strongest 30 externally supported cases from v6.
    for rank, (card, private) in enumerate(zip(v6["cards"][:30], v6["organizer"]["mapping"][:30]), 1):
        card = copy.deepcopy(card)
        private = copy.deepcopy(private)
        expected_id = f"packet-{rank:02d}"
        if card["id"] != expected_id or private["id"] != expected_id:
            raise ValueError("v6 strongest-first order changed.")
        card["pre"]["study_context"] = context(True)
        private["v7_layer"] = "supported_discovery"
        private["frozen_experiment_alignment"] = "consistent"
        pack["cards"].append(card)
        pack["organizer"]["mapping"].append(private)

    v5_cards = {card["id"]: card for card in v5["cards"]}
    v5_mapping = {entry["id"]: entry for entry in v5["organizer"]["mapping"]}
    calibration_manifest = []
    for offset, (old_id, hypothesis_id) in enumerate(CALIBRATION, 31):
        if v5_mapping[old_id]["hypothesis_id"] != hypothesis_id:
            raise ValueError(f"v5 calibration binding changed for {old_id}.")
        new_id = f"packet-{offset:02d}"
        card = copy.deepcopy(v5_cards[old_id])
        private = copy.deepcopy(v5_mapping[old_id])
        card["id"] = new_id
        private["id"] = new_id
        private["reused_from_v5_card_id"] = old_id
        card["pre"]["study_context"] = context(False)
        private["v7_layer"] = "calibration"
        private["frozen_experiment_alignment"] = "not_consistent"
        private["calibration_reason"] = "Did not meet the frozen external-consistency rule."
        pack["cards"].append(card)
        pack["organizer"]["mapping"].append(private)
        calibration_manifest.append({"v5_id": old_id, "v7_id": new_id, "hypothesis_id": hypothesis_id})

    pack["public_meta"].update({
        "version_label": "v7 · 2026-09-19 · 单轮 + 校准",
        "card_count": 34,
        "target_card_count": 34,
        "scope": ("34 份真实、已完成实验的 NeuroDiscovery 假设材料：30 份达到探索性外部一致条件，"
                  "4 份未达到该条件的校准材料；按 10 名专家 × 每人 10 份、共 100 人次设计。"),
        "selection_note": ("发现层沿用 v6 外部支持排名前 30；校准层从已完成但未达到冻结外部一致条件的材料中"
                           "固定选取 4 份，以覆盖不同连接定义与诊断组合。校准材料不计作发现。"),
        "calibration_note": ("每份材料开头公开提出者与实验一致/不一致状态。六项评分仍衡量研究能力与质量；"
                             "结果不支持本身不是低分理由。"),
    })
    pack["common_pre"].insert(0, {
        "title": "材料构成与评分原则",
        "text": ("本面板含 30 份外部支持发现和 4 份真实校准材料。每份均由 NeuroDiscovery 提出并已完成实验，"
                 "开头直接标明结果与假设一致或不一致。请继续按同一套六题评价新颖性、证据、反馈利用、设计、"
                 "验证严谨性和科学信息增量；不要仅因结果为支持或不支持而升降分。")
    })
    pack["common_post"][0]["text"] = (
        "本面板保留 30 份达到冻结探索性外部一致条件的发现材料，并加入 4 份已完成但未达到该条件的真实校准材料。"
        "校准材料用于检查评分是否区分研究质量与结果正负，不计作发现，也不把不支持结果解释为实验执行失败。"
        "材料选取不是代表性抽样，不能据此估计自然发现成功率。")
    pack["organizer"].update({
        "parent_v6_sha256": V6_SHA256,
        "v7_amendment": {
            "supported_count": 30,
            "calibration_count": 4,
            "supported_source": "v6 ranks 1--30, unchanged apart from the added display context",
            "calibration_source": calibration_manifest,
            "label_rule": ("consistent = passed the frozen exploratory external-consistency rule; "
                           "not_consistent = did not pass that rule"),
            "score_rule": "The same six 1--5 questions apply to both layers; higher remains better.",
            "unchanged": "Question wording, response options, scoring formula and all scientific result rows.",
        },
    })

    if pack["questions"] != v6["questions"] or pack["scoring"] != v6["scoring"]:
        raise ValueError("v7 must not change the six questions or their scoring.")
    if len(pack["cards"]) != 34 or len({card["id"] for card in pack["cards"]}) != 34:
        raise ValueError("v7 must contain 34 distinct cards.")
    return pack


def main():
    pack = build()
    data = (json.dumps(pack, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v7 pack already exists; create a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({
        "target": TARGET.name,
        "pack_id": pack["pack_id"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "cards": len(pack["cards"]),
        "supported": sum(c["pre"]["study_context"]["alignment"] == "consistent" for c in pack["cards"]),
        "calibration": sum(c["pre"]["study_context"]["alignment"] == "not_consistent" for c in pack["cards"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
