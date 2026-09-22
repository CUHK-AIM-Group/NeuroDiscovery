"""Derive a single-round display protocol without changing any case evidence."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials"
PARENT = HERE / "cs1_discovery_pilot_v2.json"
TARGET = HERE / "cs1_discovery_pilot_v3.json"
PARENT_HASH = "778cb31d5feb133c304cfdd04a33b72960e4b3549c5d0bd50958ff10ff68869f"


def build(parent=PARENT):
    raw = Path(parent).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_HASH:
        raise ValueError("Unexpected v2 evidence pack; do not silently rebase.")
    pack = copy.deepcopy(json.loads(raw))
    pack["pack_id"] = "cs1-discovery-capabilities-20260915-v3"
    pack["protocol_version"] = "cs1-complete-output-single-round-v3"
    pack["public_meta"].update({
        "version_label": "v3 · 2026-09-15 · 单轮",
        "review_flow": "single_round",
        "review_context": "在完整材料可见条件下评价；不作结果盲评或前后评分变化分析。",
    })
    for question in pack["questions"]:
        question["stage"] = "review"
    pack["common_pre"][0]["text"] = (
        "检验 ADHD、双相及 SZ/SZA 中具名诊断组合，是否在固定连接测量上分别相对对照呈同向改变。"
        "假设在该候选 TCP 测量前锁定，但形成过程中使用了更早同 seed 的 TCP 反馈。"
        "本轮同页提供假设、依据、方法、当前候选结果及真实父反馈，六项评分均在完整材料可见条件下完成。"
        "这是已完成研究成果的专家评价，不是对未知结果的预测或结果盲评。"
    )
    pack["scoring"]["version"] = "six-capabilities-full-output-single-round-v2"
    pack["scoring"]["interpretation"] = (
        "Descriptive expert-rated research-capability index (1–5) with complete research "
        "outputs available during all ratings. Not an outcome-blind hypothesis evaluation, "
        "validated latent scale, discovery probability, or scientific ground truth. "
        "Do not pool with the two-stage protocol."
    )
    pack["organizer"]["single_round_amendment"] = {
        "parent_v2_sha256": PARENT_HASH,
        "reason": "User requested completing each case in one round to reduce delayed recall.",
        "unchanged": "All ten cards, source mappings, scientific results, six question texts and option-score mappings.",
        "changed": "Presentation, response stage, information-condition wording, protocol/scoring identity.",
        "old_sessions": "Preserve snapshots and answers; do not convert or pool across protocols.",
    }
    return pack


if __name__ == "__main__":
    pack = build()
    if TARGET.exists():
        if json.loads(TARGET.read_text(encoding="utf-8")) != pack:
            raise ValueError("v3 already exists with different content; create a new version.")
    else:
        with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(pack, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(TARGET.resolve())
