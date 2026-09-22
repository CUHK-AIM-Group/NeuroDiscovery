"""Versioned, bilingual significance prose; no new or modified experiments."""
import copy
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
PARENT = HERE / "cs1_discovery_pilot_v8.json"
PARENT_SHA = "e5a63a40d9da88ac4af8cd2eb90a5cf59f23d87a8a74eff49821b8309fe9bb65"
CONTENT_REVISION = "case-specific-significance-v1"
PACK_ID = "neurodiscovery-multitopic-20260920-v9"
PLAIN_EN = {
    "packet-35": "Higher Alzheimer's disease polygenic risk scores are expected to be associated with smaller hippocampal volumes at baseline, relative to intracranial volume.",
    "packet-36": "More copies of the APOE ε4 allele are expected to be associated with smaller entorhinal cortex volumes at baseline, relative to intracranial volume.",
    "packet-37": "More copies of the APOE ε4 allele are expected to be associated with smaller hippocampal volumes at baseline, relative to intracranial volume.",
    "packet-38": "More copies of the APOE ε4 allele are expected to be associated with smaller middle temporal gyrus volumes at baseline, relative to intracranial volume.",
    "packet-39": "The hypothesis asks whether APOE ε4 copy number is associated with ventricular size at baseline, relative to intracranial volume.",
    "packet-40": "The hypothesis tests whether a polygenic score for an AD-risk GWAS gene set is associated with hippocampal volume at baseline, relative to intracranial volume.",
    "packet-41": "Higher polygenic scores for the Mendelian-AD gene set are expected to be associated with smaller hippocampal volumes at baseline, relative to intracranial volume.",
    "packet-42": "Among people with mild cognitive impairment (MCI), larger entorhinal cortex volumes at baseline are expected to be associated with a lower subsequent risk of dementia.",
    "packet-43": "Among people with mild cognitive impairment (MCI), the hypothesis tests whether hippocampal volume at baseline is associated with time to dementia.",
    "packet-44": "Among people with mild cognitive impairment (MCI), larger middle temporal gyrus volumes at baseline are expected to be associated with a lower subsequent risk of dementia.",
    "packet-45": "Among people with mild cognitive impairment (MCI), the hypothesis tests whether ventricular size at baseline is associated with time to dementia.",
    "packet-46": "Among people with mild cognitive impairment (MCI), larger whole-brain volumes at baseline are expected to be associated with a lower subsequent risk of dementia.",
    "packet-47": "Among people with mild cognitive impairment (MCI), higher Alzheimer's disease polygenic risk scores are expected to be associated with a shorter time to dementia.",
    "packet-48": "Among people with mild cognitive impairment (MCI), the hypothesis tests whether a polygenic score for the Mendelian-AD gene set is associated with time to dementia.",
}


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError("The previous release changed; do not reinterpret its records.")
    pack = json.loads(raw)
    significance = json.loads((HERE / "discovery_significance_v9_authoring.json").read_bytes())
    if set(significance) != {card["id"] for card in pack["cards"]}:
        raise ValueError("Provide individually authored significance for every case.")
    pack["pack_id"] = PACK_ID
    pack["protocol_version"] = "complete-output-multitopic-v9"
    pack["public_meta"]["version_label"] = "v9 · 2026-09-20 · 多主题"
    pack["public_meta"]["content_revision"] = CONTENT_REVISION
    # Embed prose in the immutable session snapshot, not only the config map.
    for card in pack["cards"]:
        note = significance[card["id"]]
        if set(note) != {"zh", "en"} or not all(note.values()):
            raise ValueError(f"Incomplete bilingual significance: {card['id']}")
        card["pre"]["significance"] = copy.deepcopy(note)
    pack_raw = serialize(pack)
    pack_sha = hashlib.sha256(pack_raw).hexdigest()
    binding = {"version": 2, "pack_id": PACK_ID, "pack_sha256": pack_sha}
    outputs = {"cs1_discovery_pilot_v9.json": pack_raw}
    for kind in ("assignments", "reference_notes"):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v3.json").read_bytes())
        sidecar.update(binding)
        outputs[f"cs1_discovery_{kind}_v4.json"] = serialize(sidecar)
    outputs["cs1_discovery_significance_v4.json"] = serialize({**binding, "significance": significance})
    catalog = json.loads((HERE / "cs1_discovery_en_v5.json").read_bytes())
    catalog.update(version="discovery-en-v7", source_pack="cs1_discovery_pilot_v9.json", source_sha256=pack_sha)
    catalog["strings"][pack["public_meta"]["version_label"]] = "v9 · 2026-09-20 · Multiple topics"
    for note in significance.values():
        catalog["strings"][note["zh"]] = note["en"]
    for card in pack["cards"]:
        if card["id"] in PLAIN_EN:
            catalog["strings"][card["pre"]["hypothesis_plain"]] = PLAIN_EN[card["id"]]
    outputs["cs1_discovery_en_v7.json"] = serialize(catalog)
    return outputs


def preview(pack, catalog):
    """Author-facing review of the revised first section, without collecting data."""
    esc = html.escape
    cards = []
    for card in pack["cards"]:
        pre = card["pre"]
        cards.append(f'<article id="{esc(card["id"])}"><span class="topic">{esc(card["id"])} · {esc(pre["topic"])}</span>'
                     f'<h2>{esc(pre["title"])}</h2><div class="columns"><section lang="zh"><h3>这个发现是什么</h3>'
                     f'<p>{esc(pre["hypothesis_plain"])}</p><p class="sig">{esc(pre["significance"]["zh"])}</p></section>'
                     f'<section lang="en"><h3>What this finding is</h3><p>{esc(catalog["strings"].get(pre["hypothesis_plain"], pre["hypothesis_plain"]))}</p>'
                     f'<p class="sig">{esc(pre["significance"]["en"])}</p></section></div></article>')
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
        '<title>NeuroDiscovery · 逐例研究与临床意义</title><style>' \
        'body{margin:0;background:#f5f8f9;color:#17252b;font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}' \
        'main{max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px;margin:8px 0 20px}' \
        'h3{font-size:16px;color:#266b79;margin:0}article{background:white;border:1px solid #dce6e9;border-radius:16px;padding:28px;margin:24px 0}' \
        '.topic{font-size:13px;color:#62757d}.columns{display:grid;grid-template-columns:1fr 1fr;gap:32px}.sig{border-left:3px solid #78a7ad;padding-left:16px;color:#34505b}' \
        '@media(max-width:760px){.columns{grid-template-columns:1fr}article{padding:20px}}' \
        '</style><main><h1>逐例研究与临床意义 · 34 份材料</h1><p>第一节文案，中英文并列。问题、分配与实验数值保持不变。</p>' \
        + ''.join(cards) + '</main></html>'


if __name__ == "__main__":
    outputs = build()
    for name, raw in outputs.items():
        target = HERE / name
        if target.exists() and target.read_bytes() != raw:
            raise SystemExit(f"Different release exists: {target}; create a new version.")
        target.write_bytes(raw)
    target = HERE.parents[2] / "HUMAN_EVALUATION_CASE_SIGNIFICANCE_20260920.html"
    target.write_text(preview(json.loads(outputs["cs1_discovery_pilot_v9.json"]),
                              json.loads(outputs["cs1_discovery_en_v7.json"])), encoding="utf-8")
    print(json.dumps({"files": {name: hashlib.sha256(raw).hexdigest() for name, raw in outputs.items()},
                      "preview": str(target)}, indent=2))
