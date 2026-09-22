"""Build bilingual reference/gloss annotations for the frozen v7 panel.

All annotations are derived from the cited KG records embedded in the frozen
card provenance.  This presentation layer does not alter the scientific pack.
"""
import hashlib
import json
from pathlib import Path

from core.web.build_reference_notes_v1 import definition_plain_zh_en, note_for


HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v7.json"
PACK_SHA256 = "787094275b59cc037dc5633d1bd5cb6695ce9666ab9851085c49c59a0c528568"
TARGET = HERE / "cs1_discovery_reference_notes_v2.json"

NETWORK_ZH = {"Vis": "视觉", "SomMot": "躯体运动", "DorsAttn": "背侧注意",
              "SalVentAttn": "显著/腹侧注意", "Limbic": "边缘",
              "Cont": "额顶控制", "Default": "默认模式"}


def legacy_definition_plain(candidate):
    """Plain gloss for the four byte-frozen v5 definition phrasings."""
    spec = candidate["canonical_record"]["spec"]
    if spec["kind"] == "network_pair":
        a, b = spec["network_a"], spec["network_b"]
        if a == b:
            return (
                f"把{NETWORK_ZH.get(a, a)}（{a}）网络内部各分区之间的功能连接取平均，作为本假设的测量指标；它不针对单个分区。",
                f"The measure averages functional connectivity among all parcels within the {a} network; it does not target a single parcel.",
            )
        return (
            f"把{NETWORK_ZH.get(a, a)}（{a}）与{NETWORK_ZH.get(b, b)}（{b}）网络之间所有分区的功能连接取平均，作为本假设的测量指标；它不针对单个分区。",
            f"The measure averages functional connectivity across all parcel pairs between the {a} and {b} networks; it does not target a single parcel.",
        )
    roi = candidate.get("source_ROI_label") or {}
    network = candidate["endpoint"].split(":")[-1]
    number = int(roi["roi"])
    side_zh, side_en = ("左侧", "left") if "_LH_" in roi["name"] else ("右侧", "right")
    return (
        f"把{side_zh} ROI{number:03d} 分区与{NETWORK_ZH.get(network, network)}（{network}）网络所有分区之间的功能连接取平均，作为本假设的测量指标；它不代表整个脑区。",
        f"The measure averages functional connectivity between the {side_en} ROI{number:03d} parcel and every parcel of the {network} network; it does not represent an entire anatomical region.",
    )


def build(raw=None):
    raw = raw if raw is not None else PACK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PACK_SHA256:
        raise ValueError("The v7 pack changed; build a separately reviewed notes version.")
    pack = json.loads(raw)
    notes, definition_plain, total = {}, {}, 0
    for card, private in zip(pack["cards"], pack["organizer"]["mapping"]):
        candidate = private.get("source_evidence", {}).get("candidate")
        if not candidate:
            raise ValueError(f"{card['id']} lacks its frozen candidate provenance.")
        canonical = candidate["canonical_record"]
        try:
            definition_zh, definition_en = definition_plain_zh_en(card["pre"]["definition"])
        except ValueError:
            definition_zh, definition_en = legacy_definition_plain(candidate)
        definition_plain[card["id"]] = {"zh": definition_zh, "en": definition_en}
        records = [item["record"] for item in candidate["first_occurrence_cited_KG_records"]
                   if item.get("present", True)]
        references = card["pre"]["references"]
        if len(records) != len(references):
            raise ValueError(f"{card['id']}: reference count differs from frozen KG records.")
        card_notes = {}
        for reference, record in zip(references, records):
            if reference["recorded_sentence"] != record.get("sentence", ""):
                raise ValueError(f"{card['id']} {reference['id']}: displayed sentence changed.")
            card_notes[reference["id"]] = note_for(record, canonical["named"], canonical["direction"])
            total += 1
        notes[card["id"]] = card_notes
    return {
        "version": 2,
        "pack_id": pack["pack_id"],
        "pack_sha256": PACK_SHA256,
        "note_count": total,
        "disclaimer_zh": "注释由冻结的 KG 记录字段模板化生成；文献原句见上方引文，原文未在本版独立核验。",
        "definition_plain": definition_plain,
        "notes": notes,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different v2 reference-notes file exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "cards": len(result["notes"]), "note_count": result["note_count"]}, indent=2))


if __name__ == "__main__":
    main()
