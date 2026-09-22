"""Merge the two unscored options in a new, source-preserving pilot version."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials"
PARENT = HERE / "cs1_discovery_pilot_v3.json"
TARGET = HERE / "cs1_discovery_pilot_v4.json"
PARENT_HASH = "6fe71e71213ea55003276a8409546daef30481d8b2cbd46bbc8df8b03b159c3c"


def build(parent=PARENT):
    raw = Path(parent).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_HASH:
        raise ValueError("Unexpected v3 pack; do not silently rebase.")
    pack = copy.deepcopy(json.loads(raw))
    pack["pack_id"] = "cs1-discovery-capabilities-20260915-v4"
    pack["protocol_version"] = "cs1-complete-output-single-round-v4"
    pack["public_meta"]["version_label"] = "v4 · 2026-09-15 · 单轮"
    for question in pack["questions"]:
        question["options"] = [o for o in question["options"] if o["score"] is not None]
        question["options"].append({"value": "unable", "score": None, "label": "无法判断（不计分）"})
    pack["scoring"]["version"] = "six-capabilities-single-round-one-missing-v3"
    pack["scoring"]["non_numeric"] = ["unable"]
    for item in pack["common_pre"]:
        item["text"] = item["text"].replace("可选择“材料不足”", "可选择“无法判断（不计分）”")
    pack["organizer"]["unscored_option_amendment"] = {
        "parent_v3_sha256": PARENT_HASH,
        "change": "One unscored option: unable -> null; no missing-reason question.",
        "unchanged": "All case evidence, question wording, five numeric anchors and averaging rules.",
        "history": "Old answers and their original reason categories remain unchanged in their snapshots.",
    }
    return pack


if __name__ == "__main__":
    pack = build()
    if TARGET.exists():
        if json.loads(TARGET.read_text(encoding="utf-8")) != pack:
            raise ValueError("v4 already exists with different content; create a new version.")
    else:
        with TARGET.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(pack, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(TARGET.resolve())
