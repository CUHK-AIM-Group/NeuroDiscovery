"""Build per-card meaning sentences for the frozen v7 mixed panel."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v7.json"
PACK_SHA256 = "787094275b59cc037dc5633d1bd5cb6695ce9666ab9851085c49c59a0c528568"
PREVIOUS = HERE / "cs1_discovery_significance_v1.json"
PREVIOUS_SHA256 = "49b5cae7d3006122d398286b036f02a776559ff765e654655914070d52809373"
CATALOG = HERE / "cs1_discovery_en_v3.json"
CATALOG_SHA256 = "6b4577a73b33e2e6432a97d15381f8bcdcd686d8f86512218da22d34f96ad77e"
TARGET = HERE / "cs1_discovery_significance_v2.json"

GROUPS_ZH = {
    frozenset({"bipolar", "psychosis_SZ_SZA"}): "双相障碍与精神分裂症/分裂情感性障碍患者",
    frozenset({"ADHD", "psychosis_SZ_SZA"}): "ADHD 与精神分裂症/分裂情感性障碍患者",
    frozenset({"ADHD", "bipolar"}): "ADHD 与双相障碍患者",
    frozenset({"ADHD", "bipolar", "psychosis_SZ_SZA"}):
        "ADHD、双相障碍与精神分裂症/分裂情感性障碍患者",
}
GROUPS_EN = {
    frozenset({"bipolar", "psychosis_SZ_SZA"}):
        "people with bipolar disorder and schizophrenia/schizoaffective disorder",
    frozenset({"ADHD", "psychosis_SZ_SZA"}):
        "people with ADHD and schizophrenia/schizoaffective disorder",
    frozenset({"ADHD", "bipolar"}): "people with ADHD and bipolar disorder",
    frozenset({"ADHD", "bipolar", "psychosis_SZ_SZA"}):
        "people with ADHD, bipolar disorder, and schizophrenia/schizoaffective disorder",
}


def checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen input changed: {path.name}")
    return json.loads(raw)


def build(raw=None):
    raw = raw if raw is not None else PACK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PACK_SHA256:
        raise ValueError("The v7 pack changed; build a separately reviewed significance version.")
    pack = json.loads(raw)
    prior = checked(PREVIOUS, PREVIOUS_SHA256)["significance"]
    strings = checked(CATALOG, CATALOG_SHA256)["strings"]
    result = {}
    for card, private in zip(pack["cards"], pack["organizer"]["mapping"]):
        context = card["pre"]["study_context"]
        if context["alignment"] == "consistent" and card["id"] in prior:
            result[card["id"]] = prior[card["id"]]
            continue
        candidate = private["source_evidence"]["candidate"]
        groups_key = frozenset(candidate["canonical_record"]["named"])
        title_zh = card["pre"]["title"]
        title_en = strings.get(title_zh)
        if not title_en or groups_key not in GROUPS_ZH:
            raise ValueError(f"Missing reviewed template inputs for {card['id']}.")
        result[card["id"]] = {
            "zh": (f"若后续研究确认，这条命题意味着{GROUPS_ZH[groups_key]}在「{title_zh}」上存在共同降低；"
                   "本次不一致结果目前主要帮助界定该跨诊断命题的适用边界，不能作为已验证标志物。"),
            "en": (f"If later confirmed, this claim would imply a shared decrease in \"{title_en}\" among "
                   f"{GROUPS_EN[groups_key]}. The present inconsistent result instead helps define the "
                   "boundary of the transdiagnostic claim and is not a validated biomarker."),
        }
    return {"version": 2, "pack_id": pack["pack_id"], "pack_sha256": PACK_SHA256,
            "disclaimer_zh": "意义句是面向评审的通俗说明，不是新的科学结论。", "significance": result}


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v2 significance file exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "cards": len(result["significance"])}, indent=2))


if __name__ == "__main__":
    main()
