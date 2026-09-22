"""Build the v7 English display catalog from the reviewed v3 catalog.

All card scientific text is reused from v5/v6 and is already translated in
v3.  This version adds only the v7 mixed-panel and up-front status language,
then verifies that every Chinese string in the public v7 projection is covered.
"""
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v7.json"
PACK_SHA256 = "787094275b59cc037dc5633d1bd5cb6695ce9666ab9851085c49c59a0c528568"
PREVIOUS = HERE / "cs1_discovery_en_v3.json"
PREVIOUS_SHA256 = "6b4577a73b33e2e6432a97d15381f8bcdcd686d8f86512218da22d34f96ad77e"
TARGET = HERE / "cs1_discovery_en_v4.json"
CJK = re.compile(r"[\u3400-\u9fff]")

ADDITIONS = {
    "v7 · 2026-09-19 · 单轮 + 校准": "v7 · 2026-09-19 · single round + calibration",
    "34 份真实、已完成实验的 NeuroDiscovery 假设材料：30 份达到探索性外部一致条件，4 份未达到该条件的校准材料；按 10 名专家 × 每人 10 份、共 100 人次设计。":
        "Thirty-four real NeuroDiscovery hypotheses with completed experiments: 30 met the exploratory external-consistency rule and 4 calibration cases did not; the design uses 10 experts × 10 cases each, for 100 reviews.",
    "发现层沿用 v6 外部支持排名前 30；校准层从已完成但未达到冻结外部一致条件的材料中固定选取 4 份，以覆盖不同连接定义与诊断组合。校准材料不计作发现。":
        "The finding layer retains the top 30 externally supported v6 cases. The calibration layer fixes four completed cases that did not meet the frozen external-consistency rule, covering different connectivity definitions and diagnosis combinations. Calibration cases are not counted as discoveries.",
    "每份材料开头公开提出者与实验一致/不一致状态。六项评分仍衡量研究能力与质量；结果不支持本身不是低分理由。":
        "Each case begins by disclosing the proposer and whether the experiment was consistent with the hypothesis. The six ratings still measure research capability and quality; a non-supporting result is not itself a reason for a low score.",
    "材料构成与评分原则": "Panel composition and scoring principle",
    "本面板含 30 份外部支持发现和 4 份真实校准材料。每份均由 NeuroDiscovery 提出并已完成实验，开头直接标明结果与假设一致或不一致。请继续按同一套六题评价新颖性、证据、反馈利用、设计、验证严谨性和科学信息增量；不要仅因结果为支持或不支持而升降分。":
        "This panel contains 30 externally supported findings and 4 real calibration cases. Every hypothesis was proposed by NeuroDiscovery and experimentally tested, and each case states up front whether the result was consistent or inconsistent. Use the same six questions to rate novelty, evidence, feedback use, design, validation rigor and scientific information gain; do not raise or lower a score merely because the result was supporting or non-supporting.",
    "本面板保留 30 份达到冻结探索性外部一致条件的发现材料，并加入 4 份已完成但未达到该条件的真实校准材料。校准材料用于检查评分是否区分研究质量与结果正负，不计作发现，也不把不支持结果解释为实验执行失败。材料选取不是代表性抽样，不能据此估计自然发现成功率。":
        "The panel retains 30 finding cases that met the frozen exploratory external-consistency rule and adds 4 real completed calibration cases that did not. Calibration cases test whether ratings distinguish research quality from result direction; they are not counted as discoveries, and non-support is not treated as execution failure. This is not a representative sample and cannot estimate the natural discovery success rate.",
    "本假设由 NeuroDiscovery 提出": "This hypothesis was proposed by NeuroDiscovery",
    "外部支持发现": "Externally supported finding",
    "校准材料": "Calibration case",
    "实验结果：与假设一致": "Experimental result: consistent with the hypothesis",
    "实验结果：与假设不一致": "Experimental result: not consistent with the hypothesis",
    "该材料达到预先固定的探索性外部一致条件；仍不等同于正式确证或全领域首次发现。":
        "This case met the frozen exploratory external-consistency rule; that is not the same as formal confirmation or a field-first discovery.",
    "该材料未达到预先固定的外部一致条件。它是真实的不支持结果，不代表实验执行失败，也不应仅因结果不支持而降低严谨性评分。":
        "This case did not meet the frozen external-consistency rule. It is a real non-supporting result, not an execution failure, and should not receive a lower rigor score merely because the result is non-supporting.",
}


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)


def checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen input changed: {path.name}")
    return json.loads(raw)


def build():
    pack = checked(PACK, PACK_SHA256)
    previous = checked(PREVIOUS, PREVIOUS_SHA256)
    strings = dict(previous["strings"])
    for source, target in ADDITIONS.items():
        if source in strings and strings[source] != target:
            raise ValueError(f"Conflicting translation for: {source}")
        strings[source] = target
    fields = {key: pack[key] for key in
              ("public_meta", "questions", "issues", "common_pre", "common_post", "cards")}
    missing = sorted({text for text in strings_in(fields) if CJK.search(text) and text not in strings})
    if missing:
        raise ValueError(f"v7 catalog lacks {len(missing)} Chinese strings: {missing[:3]}")
    return {
        "version": "cs1-discovery-display-en-v4",
        "source_pack": PACK.name,
        "source_sha256": PACK_SHA256,
        "covers_packs": ["cs1_discovery_pilot_v5.json", "cs1_discovery_pilot_v6.json", PACK.name],
        "strings": strings,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v4 catalog exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "strings": len(result["strings"])}, indent=2))


if __name__ == "__main__":
    main()
