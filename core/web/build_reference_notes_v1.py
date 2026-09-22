"""Per-reference display notes for the frozen v6 panel: what each cited study
did and how it relates to the card's hypothesis, plus a plain-language gloss
of each card's measurement definition.

Owner request (2026-09-18): replace the two per-reference caveat paragraphs
with a short note stating what the reference did and its relation to the
hypothesis, expand "why this hypothesis" into a readable evidence-backed
narrative, and keep the material approachable. Every note is TEMPLATED from
frozen records (KG record fields, selection named groups/direction, and the
pack's own definition strings); no new scientific content is authored here.
The verbatim recorded sentences and definitions stay on the cards unchanged;
the global literature-verification caveat in common_pre is untouched.

This is a presentation-layer annotation file bound to the exact v6 pack hash,
like the English display catalog. It never modifies the pack. It only reads
frozen records; it never runs experiments.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v6.json"
PACK_SHA256 = "1079b398a43a1f842781348464bc7aa892f7063480a5e8db41295f97172e6681"
SELECTION = HERE / "sources_v6" / "selection_v6.json"
SELECTION_SHA256 = "84e79d82f5f248c2c8d9818eeb1f2bc94e7628c3516fd5a3c8d92984e2122c85"
SOURCE_HASHES = {
    "old_r1/candidate_index.json": "a16ff8cc00d87dcf2c1d6eeafb2d0ad4ae3ea8e55f7d8f061d2f7ad309ce1b74",
    "r7/candidate_index.json": "40964c8ba056a2cc44615857030f48a177e24baff50a25a2a6b168bc2f74f19d",
}
TARGET = HERE / "cs1_discovery_reference_notes_v1.json"

DOMAIN_ZH = {"ADHD": "ADHD", "bipolar": "双相", "psychosis_SZ_SZA": "SZ/SZA", "psychosis": "精神病",
             "SCHZ": "SCHZ", "SZ": "SZ", "nonaffective_psychosis": "非情感性精神病",
             "affective_psychosis": "情感性精神病"}
DOMAIN_EN = {"ADHD": "ADHD", "bipolar": "bipolar disorder", "psychosis_SZ_SZA": "SZ/SZA",
             "psychosis": "psychosis", "SCHZ": "SCHZ", "SZ": "SZ",
             "nonaffective_psychosis": "non-affective psychosis", "affective_psychosis": "affective psychosis"}

# Clauses use only the record's own subject/object; domains carry the
# population in a controlled vocabulary. reduces/increases take the object
# (their subject is the population, already named by the domains).
DID_ZH = {
    "reduces": lambda s, o: f"观察到 {o} 降低",
    "increases": lambda s, o: f"观察到 {o} 升高",
    "correlates_with": lambda s, o: f"{s} 与 {o} 相关",
    "distinguishes": lambda s, o: f"{s} 能区分 {o}",
    "is_associated_with": lambda s, o: f"{s} 与 {o} 相关联",
    "associated_with": lambda s, o: f"{s} 与 {o} 相关联",
    "is_biomarker_of": lambda s, o: f"{s} 被标记为 {o} 的候选标志物",
    "predicts": lambda s, o: f"{s} 可预测 {o}",
    "is_risk_factor_for": lambda s, o: f"{s} 是 {o} 的风险因素",
}
DID_EN = {
    "reduces": lambda s, o: f"reported reduced {o}",
    "increases": lambda s, o: f"reported increased {o}",
    "correlates_with": lambda s, o: f"{s} correlated with {o}",
    "distinguishes": lambda s, o: f"{s} distinguished {o}",
    "is_associated_with": lambda s, o: f"{s} was associated with {o}",
    "associated_with": lambda s, o: f"{s} was associated with {o}",
    "is_biomarker_of": lambda s, o: f"{s} was flagged as a candidate marker of {o}",
    "predicts": lambda s, o: f"{s} predicted {o}",
    "is_risk_factor_for": lambda s, o: f"{s} was a risk factor for {o}",
}
DIRECTIONAL = {"reduces": -1, "increases": 1}

NETWORK_ZH = {"SomMot": "躯体运动（SomMot）", "SalVentAttn": "显著/腹侧注意（SalVentAttn）",
              "DorsAttn": "背侧注意（DorsAttn）", "Default": "默认模式（Default）",
              "Vis": "视觉（Vis）", "Limbic": "边缘（Limbic）", "Cont": "额顶控制（Cont）"}
SIDE = {"LH": ("左侧", "left"), "RH": ("右侧", "right")}

ROI_DEFINITION = re.compile(
    r"^Schaefer-400 / 7-network；ROI(\d+) = (7Networks_(LH|RH)_\S+?)。该分区到 ?"
    r"(?:全部 (\d+) 个 )?(\S+?)(?: 分区的| 的) (\d+) 条非对角 Pearson r 边的算术均值")
NETWORK_DEFINITION = re.compile(
    r"^Schaefer-400 / 7-network；(\S+?)（(\d+) 个分区）与 (\S+?)（(\d+) 个分区），"
    r"共 ([\d,]+) 条唯一非对角 Pearson r 边的算术均值")


def definition_plain_zh_en(definition):
    """Plain-language gloss derived from the pack's own definition string."""
    roi = ROI_DEFINITION.match(definition)
    if roi:
        number, atlas, hemi, _all, network, edges = roi.groups()
        side_zh, side_en = SIDE[hemi]
        net_zh = NETWORK_ZH.get(network, network)
        zh = (f"把{side_zh} ROI{number} 分区与{net_zh}网络所有分区之间的功能连接取平均，"
              f"作为本假设的测量指标。它只是这一个分区到该网络的平均连接，不代表整个脑区或整个网络。")
        en = (f"The measure averages functional connectivity between the {side_en} ROI{number} parcel "
              f"and every parcel of the {network} network. It is only that parcel-to-network average, "
              f"not an entire anatomical region.")
        return zh, en
    pair = NETWORK_DEFINITION.match(definition)
    if pair:
        net_a, _n_a, net_b, _n_b, edges = pair.groups()
        net_a_zh = NETWORK_ZH.get(net_a, net_a)
        net_b_zh = NETWORK_ZH.get(net_b, net_b)
        if net_a == net_b:
            zh = (f"把{net_a_zh}网络内部各分区之间的功能连接取平均（共 {edges} 条），"
                  f"作为本假设的测量指标；它反映该网络内部的总体连接强度，不针对单个分区。")
            en = (f"The measure averages every parcel-to-parcel functional connection within the "
                  f"{net_a} network ({edges} connections in total); it reflects overall "
                  f"within-network connectivity rather than any single parcel.")
        else:
            zh = (f"把{net_a_zh}网络与{net_b_zh}网络之间所有分区间的功能连接取平均（共 {edges} 条），"
                  f"作为本假设的测量指标；它反映两个网络之间的总体连接强度，不针对单个分区。")
            en = (f"The measure averages every parcel-to-parcel functional connection between the "
                  f"{net_a} and {net_b} networks ({edges} connections in total); it reflects overall "
                  f"between-network connectivity rather than any single parcel.")
        return zh, en
    raise ValueError(f"Definition does not match either reviewed family: {definition[:80]}")


def _load_checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source changed: {path.name}; build a separately reviewed new version.")
    return json.loads(raw)


def _iter_index_entries(node):
    if isinstance(node, dict):
        if "first_occurrence_cited_KG_records" in node:
            yield node
        for value in node.values():
            yield from _iter_index_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_index_entries(value)


def _domains(values, mapping, sep="、"):
    return sep.join(mapping.get(v, v) for v in values)


def note_for(record, named, direction):
    predicate = record["predicate"]
    subject, object_ = record.get("subject") or "", record.get("object") or ""
    if record.get("negated"):
        raise ValueError(f"Negated KG record needs a reviewed template: {record['id']}")
    clause_zh = DID_ZH.get(predicate, lambda s, o: f"{predicate}：{s} — {o}")(subject, object_)
    clause_en = DID_EN.get(predicate, lambda s, o: f"{predicate}: {s} — {o}")(subject, object_)
    domains = record.get("domains") or []
    did_zh = f"该文献记录（{_domains(domains, DOMAIN_ZH)}）：{clause_zh}。"
    did_en = f"KG record ({_domains(domains, DOMAIN_EN)}): {clause_en}."

    overlap = [d for d in domains if d in named]
    if overlap:
        relation_zh = f"覆盖本假设预定的 {_domains(overlap, DOMAIN_ZH)}"
        relation_en = f"Covers the hypothesis's pre-registered {_domains(overlap, DOMAIN_EN, ', ')}"
    else:
        relation_zh = "不涉及本假设预定的诊断组合，属背景证据"
        relation_en = "Does not cover the hypothesis's pre-registered diagnoses; cited as background evidence"
    sign = DIRECTIONAL.get(predicate)
    if sign is not None:
        if sign == direction:
            relation_zh += "；报告方向（降低）与本假设“共同降低”一致" if direction == -1 else "；报告方向（升高）与本假设“共同升高”一致"
            relation_en += "; its reported direction matches the hypothesis's joint direction"
        else:
            relation_zh += "；报告方向与本假设方向相反"
            relation_en += "; its reported direction runs opposite to the hypothesis's direction"
    else:
        relation_zh += "；不直接给出方向，作为关联/区分类证据引用"
        relation_en += "; it is non-directional and cited as associational evidence"
    relation_zh += "。它是系统提出本假设时引用的既有证据之一。"
    relation_en += ". It is one of the prior-evidence records cited when the system proposed this hypothesis."
    return {"did_zh": did_zh, "did_en": did_en, "relation_zh": relation_zh, "relation_en": relation_en}


def build(pack_raw=None, selection_raw=None, index_raws=None):
    pack_raw = pack_raw if pack_raw is not None else PACK.read_bytes()
    if hashlib.sha256(pack_raw).hexdigest() != PACK_SHA256:
        raise ValueError("The v6 pack changed; build a separately reviewed new notes version.")
    pack = json.loads(pack_raw)
    selection = _load_checked(SELECTION, SELECTION_SHA256) if selection_raw is None else json.loads(selection_raw)
    by_hypothesis = {entry["hypothesis_id"]: entry for entry in selection["selected"]}
    indexes = {}
    for rel, expected in SOURCE_HASHES.items():
        raw = index_raws[rel] if index_raws else None
        data = json.loads(raw) if raw else _load_checked(HERE / "sources_v6" / rel, expected)
        for entry in _iter_index_entries(data):
            indexes[entry["canonical_record"]["hypothesis_id"]] = entry

    mapping = pack["organizer"]["mapping"]
    notes = {}
    definition_plain = {}
    total = 0
    for card in pack["cards"]:
        rank = int(card["id"].split("-")[1])
        definition_zh, definition_en = definition_plain_zh_en(card["pre"]["definition"])
        definition_plain[card["id"]] = {"zh": definition_zh, "en": definition_en}
        hypothesis_id = mapping[rank - 1]["hypothesis_id"]
        entry = indexes.get(hypothesis_id)
        if entry is None:
            raise ValueError(f"Candidate index lacks {hypothesis_id} ({card['id']}).")
        selected = by_hypothesis.get(hypothesis_id)
        if selected is None:
            raise ValueError(f"Selection lacks {hypothesis_id} ({card['id']}).")
        named, direction = selected["named"], selected["direction"]
        records = [item["record"] for item in entry["first_occurrence_cited_KG_records"] if item.get("present", True)]
        references = card["pre"]["references"]
        if len(records) != len(references):
            raise ValueError(f"{card['id']}: reference count differs from the cited KG records.")
        card_notes = {}
        for reference, record in zip(references, records):
            # Bind the note to the exact displayed sentence, not just position.
            if reference["recorded_sentence"] != record.get("sentence", ""):
                raise ValueError(f"{card['id']} {reference['id']}: displayed sentence differs from the KG record.")
            card_notes[reference["id"]] = note_for(record, named, direction)
            total += 1
        notes[card["id"]] = card_notes
    return {
        "version": 1,
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
        raise ValueError("A different reference-notes file already exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "note_count": result["note_count"], "cards": len(result["notes"])}, indent=2))


if __name__ == "__main__":
    main()
