"""Per-card "what it would mean" sentences for the frozen v6 expert panel.

Owner request (2026-09-19): end each card's "系统提出的假设" block with one
plain sentence on the value if the hypothesis holds (brain research and
clinical relevance). Sentences are TEMPLATED from the frozen v6 selection
(named diagnosis groups, all joint-decrease) and the pack's own card titles;
English display sentences reuse the bound v3 catalog title translations so the
wording matches the rest of the page. No new scientific content is authored
here, and the v6 pack stays byte-frozen: this is a presentation-layer
annotation file bound to the exact pack hash, like the reference notes.
"""
import hashlib
import json
from pathlib import Path

from core.web.discovery_localization import catalog

HERE = Path(__file__).resolve().parent / "study_materials"
PACK = HERE / "cs1_discovery_pilot_v6.json"
PACK_SHA256 = "1079b398a43a1f842781348464bc7aa892f7063480a5e8db41295f97172e6681"
SELECTION = HERE / "sources_v6" / "selection_v6.json"
SELECTION_SHA256 = "84e79d82f5f248c2c8d9818eeb1f2bc94e7628c3516fd5a3c8d92984e2122c85"
TARGET = HERE / "cs1_discovery_significance_v1.json"

GROUPS_ZH = {
    frozenset({"bipolar", "psychosis_SZ_SZA"}): "双相障碍与精神分裂症/分裂情感性障碍患者",
    frozenset({"ADHD", "psychosis_SZ_SZA"}): "ADHD 与精神分裂症/分裂情感性障碍患者",
}
GROUPS_EN = {
    frozenset({"bipolar", "psychosis_SZ_SZA"}): "people with bipolar disorder and schizophrenia/schizoaffective disorder",
    frozenset({"ADHD", "psychosis_SZ_SZA"}): "people with ADHD and schizophrenia/schizoaffective disorder",
}

DISCLAIMER_ZH = "意义句按冻结选取的诊断组合与卡片标题模板化生成，是面向评审的通俗说明，不是新的科学结论。"


def sentence_zh(groups, title):
    return (f"如果成真，这条假设意味着{groups}在「{title}」这一指标上存在稳定的同方向降低——"
            f"对脑研究，它把不同诊断连接到同一个可检验的网络机制上；"
            f"对临床，这类可复现的连接指标有望成为辅助分型或追踪疗效的候选线索。")


def sentence_en(groups, title_en):
    return (f"If confirmed, this hypothesis would mean a stable, same-direction decrease in "
            f"\"{title_en}\" for {groups}. For brain research, it ties different diagnoses to one "
            f"testable network mechanism; clinically, such a reproducible connectivity measure is a "
            f"candidate lead for subtyping and for tracking treatment response.")


def _load_checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source changed: {path.name}; build a separately reviewed new version.")
    return json.loads(raw)


def build(pack_raw=None, selection_raw=None, catalog_strings=None):
    pack_raw = pack_raw if pack_raw is not None else PACK.read_bytes()
    if hashlib.sha256(pack_raw).hexdigest() != PACK_SHA256:
        raise ValueError("The v6 pack changed; build a separately reviewed new significance version.")
    pack = json.loads(pack_raw)
    selection = json.loads(selection_raw) if selection_raw is not None else _load_checked(SELECTION, SELECTION_SHA256)
    strings = catalog_strings if catalog_strings is not None else catalog()[0]["strings"]
    selected = {entry["hypothesis_id"]: entry for entry in selection["selected"]}
    significance = {}
    for card, private in zip(pack["cards"], pack["organizer"]["mapping"]):
        entry = selected.get(private["hypothesis_id"])
        if entry is None:
            raise ValueError(f"Selection lacks {private['hypothesis_id']} ({card['id']}).")
        if entry["direction"] != -1:
            raise ValueError(f"Significance template covers joint-decrease cards only: {card['id']}")
        groups_key = frozenset(entry["named"])
        if groups_key not in GROUPS_ZH:
            raise ValueError(f"Uncovered diagnosis group combination on {card['id']}: {sorted(groups_key)}")
        title = card["pre"]["title"]
        title_en = strings.get(title)
        if not title_en:
            raise ValueError(f"English catalog lacks the card title for {card['id']}.")
        significance[card["id"]] = {
            "zh": sentence_zh(GROUPS_ZH[groups_key], title),
            "en": sentence_en(GROUPS_EN[groups_key], title_en),
        }
    return {
        "version": 1,
        "pack_id": pack["pack_id"],
        "pack_sha256": PACK_SHA256,
        "disclaimer_zh": DISCLAIMER_ZH,
        "significance": significance,
    }


def main():
    result = build()
    data = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different significance file already exists; use a new version.")
    TARGET.write_bytes(data)
    print(json.dumps({"target": TARGET.name, "sha256": hashlib.sha256(data).hexdigest(),
                      "cards": len(result["significance"])}, indent=2))


if __name__ == "__main__":
    main()
