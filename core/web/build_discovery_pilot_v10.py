"""Freeze a reader-focused, bilingual presentation without changing study results."""
import copy
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
PARENT = HERE / "cs1_discovery_pilot_v9.json"
PARENT_SHA = "1f7cff10da224774f747b91b71a8435ddda0e9f02641aec1553730c1241d76c1"
CONTENT_REVISION = "readable-case-narrative-v1"
PACK_ID = "neurodiscovery-multitopic-20260920-v10"


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError("The preceding release changed; preserve its records.")
    pack = json.loads(raw)
    narratives = json.loads((HERE / "discovery_readability_v10_authoring.json").read_bytes())
    references = json.loads((HERE / "discovery_references_v10_authoring.json").read_bytes())
    if set(narratives) != {card["id"] for card in pack["cards"]}:
        raise ValueError("Every case needs an individually authored narrative.")
    if set(references) != {str(ref["pmid"]) for card in pack["cards"] for ref in card["pre"]["references"]}:
        raise ValueError("Reference summaries must match the existing sources exactly.")
    for card in pack["cards"]:
        reading = copy.deepcopy(narratives[card["id"]])
        if set(reading) != {"rationale", "feedback", "methods", "results"}:
            raise ValueError(f"Unexpected narrative fields: {card['id']}")
        for key, note in reading.items():
            if key == "feedback" and note is None:
                continue
            if set(note) != {"zh", "en"} or not all(note.values()):
                raise ValueError(f"Missing bilingual text: {card['id']} {key}")
        reading["references"] = {ref["id"]: copy.deepcopy(references[str(ref["pmid"])])
                                 for ref in card["pre"]["references"]}
        card["pre"]["reading"] = reading
    pack["pack_id"] = PACK_ID
    pack["protocol_version"] = "complete-output-multitopic-v10"
    pack["public_meta"].update(version_label="v10 · 2026-09-20 · 多主题", content_revision=CONTENT_REVISION)
    pack_raw = serialize(pack)
    pack_sha = hashlib.sha256(pack_raw).hexdigest()
    binding = {"version": 2, "pack_id": PACK_ID, "pack_sha256": pack_sha}
    outputs = {"cs1_discovery_pilot_v10.json": pack_raw}
    for kind in ("assignments", "reference_notes", "significance"):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v4.json").read_bytes())
        sidecar.update(binding)
        outputs[f"cs1_discovery_{kind}_v5.json"] = serialize(sidecar)
    catalog = json.loads((HERE / "cs1_discovery_en_v7.json").read_bytes())
    catalog.update(version="discovery-en-v8", source_pack="cs1_discovery_pilot_v10.json", source_sha256=pack_sha)
    catalog["strings"][pack["public_meta"]["version_label"]] = "v10 · 2026-09-20 · Multiple topics"
    for note in list(references.values()) + [note for case in narratives.values() for note in case.values() if note]:
        catalog["strings"][note["zh"]] = note["en"]
    outputs["cs1_discovery_en_v8.json"] = serialize(catalog)
    return outputs


def preview(pack, catalog):
    esc = html.escape
    blocks = []
    for card in pack["cards"]:
        pre, reading = card["pre"], card["pre"]["reading"]
        columns = []
        for lang in ("zh", "en"):
            en = lang == "en"
            hypothesis = catalog["strings"].get(pre["hypothesis_plain"], pre["hypothesis_plain"]) if en else pre["hypothesis_plain"]
            titles = ["What this finding is", "Where it came from", "How it was tested", "What the experiments found"] if en else ["这个发现是什么", "它从何而来", "实验怎么做", "实验结果"]
            refs = ''.join(f'<li><a href="{esc(ref["url"])}" target="_blank" rel="noopener noreferrer">{esc(ref["id"])}</a> · {esc(reading["references"][ref["id"]][lang])}</li>' for ref in pre["references"])
            feedback = f'<h4>{"Learning from earlier experiments" if en else "前序实验的启发"}</h4><p>{esc(reading["feedback"][lang])}</p>' if reading["feedback"] else ""
            columns.append(f'<section lang="{lang}"><h3>01 · {titles[0]}</h3><p>{esc(hypothesis)}</p><p class="sig">{esc(pre["significance"][lang])}</p>'
                           f'<h3>02 · {titles[1]}</h3><p>{esc(reading["rationale"][lang])}</p><ul>{refs}</ul>{feedback}'
                           f'<h3>03 · {titles[2]}</h3><p>{esc(reading["methods"][lang])}</p>'
                           f'<h3>04 · {titles[3]}</h3><p>{esc(reading["results"][lang])}</p></section>')
        blocks.append(f'<article id="{esc(card["id"])}"><span class="topic">{esc(card["id"])} · {esc(pre["topic"])}</span><h2>{esc(pre["title"])}</h2><div class="columns">'+''.join(columns)+'</div></article>')
    nav = ' '.join(f'<a href="#{esc(card["id"])}">{esc(card["id"].split("-")[-1])}</a>' for card in pack["cards"])
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
        '<title>NeuroDiscovery · 专家研究材料</title><style>' \
        'body{margin:0;background:#f5f8f9;color:#17252b;font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}' \
        'main{max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px;margin:8px 0 20px}' \
        'h3{font-size:17px;color:#266b79;margin:24px 0 8px}h4{font-size:15px;margin-bottom:4px}p{margin:8px 0 16px}' \
        'article{background:white;border:1px solid #dce6e9;border-radius:16px;padding:28px;margin:24px 0;scroll-margin:16px}' \
        '.topic,li{font-size:13px;color:#62757d}li{margin:10px 0}.columns{display:grid;grid-template-columns:1fr 1fr;gap:32px}' \
        '.sig{border-left:3px solid #78a7ad;padding-left:16px;color:#34505b}a{color:#266b79}nav{display:flex;flex-wrap:wrap;gap:6px}' \
        'nav a{padding:3px 10px;border:1px solid #cbdde1;border-radius:6px;text-decoration:none}' \
        '@media(max-width:760px){.columns{grid-template-columns:1fr}article{padding:20px}}' \
        '</style><main><h1>专家研究材料 · 中英文逐例阅读</h1><nav>'+nav+'</nav>'+''.join(blocks)+'</main></html>'


if __name__ == "__main__":
    outputs = build()
    for name, raw in outputs.items():
        target = HERE / name
        if target.exists() and target.read_bytes() != raw:
            raise SystemExit(f"Different release already exists: {target}; create a new version.")
        target.write_bytes(raw)
    target = HERE.parents[2] / "HUMAN_EVALUATION_READABLE_CASES_20260920.html"
    target.write_text(preview(json.loads(outputs["cs1_discovery_pilot_v10.json"]), json.loads(outputs["cs1_discovery_en_v8.json"])), encoding="utf-8")
    print(json.dumps({"files": {name: hashlib.sha256(raw).hexdigest() for name, raw in outputs.items()}, "preview": str(target)}, indent=2))
