"""Restore study scale and source-specific explanations without changing results."""
import copy
import hashlib
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
PARENT = HERE / "cs1_discovery_pilot_v11.json"
PARENT_SHA = "a1bfc030cb1df2d2e76052095f6fe654c3cc354c6d651efb76f55c5989f84301"
LITERATURE = HERE / "sources_v12/literature.json"
LITERATURE_SHA = "068905992b70e0b079893b52f61016aba77a1dc044b408aac39d40bd96cf9433"
PACK_ID = "neurodiscovery-multitopic-20260920-v12"
CONTEXT_REVISION = "study-scale-and-reference-context-v1"


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def checked(path, checksum):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError(f"Frozen source changed: {path}")
    return json.loads(raw)


def study_scale(card):
    """Use effective row-level N, never a topic-wide constant or sum of reused controls."""
    post = card["post"]
    if "experimental_results" in post:
        configs = post["experimental_results"]
        counts = {phase: sorted({c[phase]["primary"]["n"] for c in configs})
                  for phase in ("internal", "external")}
        if any(len(values) != 1 for values in counts.values()):
            raise ValueError("A varying-N case needs a separately authored scale description.")
        ni, ne = counts["internal"][0], counts["external"][0]
        facts = {"internal_n": ni, "external_n": ne, "configurations_per_cohort": len(configs)}
        if post["result_kind"] == "imaging_genetics":
            assert len(configs) == 3
            return {"zh": f"2 个数据批次：ADNI1/GO/2 内部分析 {ni} 人，ADNI3 外部分析 {ne} 人。每批数据使用 3 种关联方法，共 6 组分析。",
                    "en": f"Two data phases: {ni} participants in the internal ADNI1/GO/2 analysis and {ne} in external ADNI3. Three association methods per cohort give six analyses."}, facts
        assert [c["horizon_years"] for c in configs] == [2, 3, 5]
        events = {phase: [c[phase]["primary"]["events"] for c in configs]
                  for phase in ("internal", "external")}
        facts["events_by_2_3_5_year_window"] = events
        return {"zh": f"2 个 MCI 数据批次：内部 ADNI1/GO/2 为 {ni} 人，外部 ADNI3 为 {ne} 人。分别检验 2、3、5 年随访窗口，共 6 组生存分析。",
                "en": f"Two MCI data phases: {ni} internal ADNI1/GO/2 participants and {ne} external ADNI3 participants. Two-, three- and five-year windows give six survival analyses."}, facts
    internal = {r["domain"]: r for r in post["internal_rows"]}
    primary = [r for r in post["external_rows"] if r["variant"].startswith("primary_")]
    external = {(r["cohort"], r["domain"]): r for r in primary}
    b, s = internal["bipolar"], internal["psychosis_SZ_SZA"]
    ub, us = external[("UCLA", "bipolar")], external[("UCLA", "psychosis_SZ_SZA")]
    c, e = external[("COBRE", "psychosis_SZ_SZA")], external[("HCP-EP", "non_affective_psychosis")]
    assert b["controls"] == s["controls"] and ub["controls"] == us["controls"]
    facts = {"cohorts": 4, "internal_comparisons": len(internal), "external_primary_comparisons": len(primary),
             "groups": [{k: r[k] for k in ("domain", "cases", "controls")}
                        for r in post["internal_rows"]] +
                       [{k: r[k] for k in ("cohort", "domain", "cases", "controls")} for r in primary]}
    return {
        "zh": f"4 个队列，共 {len(internal)} 项内部与 {len(primary)} 项外部主要比较。TCP：双相障碍 {b['cases']} 人、精神分裂症/分裂情感性障碍 {s['cases']} 人、对照 {b['controls']} 人；UCLA：双相障碍 {ub['cases']} 人、精神分裂症 {us['cases']} 人、对照 {ub['controls']} 人。COBRE：精神分裂症谱系 {c['cases']} 人、对照 {c['controls']} 人；HCP-EP：非情感性早期精神病 {e['cases']} 人、对照 {e['controls']} 人。",
        "en": f"Four cohorts, with {len(internal)} internal and {len(primary)} primary external comparisons. TCP: {b['cases']} bipolar, {s['cases']} schizophrenia/schizoaffective and {b['controls']} control participants. UCLA: {ub['cases']} bipolar, {us['cases']} schizophrenia and {ub['controls']} control participants. COBRE: {c['cases']} schizophrenia-spectrum and {c['controls']} control participants. HCP-EP: {e['cases']} early non-affective psychosis and {e['controls']} control participants."
    }, facts


def build():
    parent = checked(PARENT, PARENT_SHA)
    literature = checked(LITERATURE, LITERATURE_SHA)
    papers = {p["id"]: p for p in literature["papers"]}
    writing = json.loads((HERE / "discovery_context_v12_authoring.json").read_bytes())
    assert set(writing["cases"]) == {c["id"] for c in parent["cards"]}
    assert set(writing["papers"]) == set(papers) == {
        str(ref["pmid"]) for card in parent["cards"] for ref in card["pre"]["references"]}
    pack = copy.deepcopy(parent)
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v12")
    pack["public_meta"].update(version_label="v12 · 2026-09-20 · 多主题", context_revision=CONTEXT_REVISION)
    audit, note_rows = {}, {}
    mapping = {m.get("id", m.get("card_id")): m for m in parent["organizer"]["mapping"]}
    for card in pack["cards"]:
        cid, pre = card["id"], card["pre"]
        prose = writing["cases"][cid]
        assert set(prose) == {"methods", "results", "relations"}
        assert set(prose["relations"]) == {r["id"] for r in pre["references"]}
        reading = pre["reading"]
        reading.update(methods=copy.deepcopy(prose["methods"]), results=copy.deepcopy(prose["results"]))
        reading["scale"], facts = study_scale(card)
        reading["reference_details"], note_rows[cid] = {}, {}
        for ref in pre["references"]:
            paper, pmid, rid = papers[str(ref["pmid"])], str(ref["pmid"]), ref["id"]
            detail = {"title": html.unescape(re.sub(r"<[^>]+>", "", paper["title"])),
                      "journal": paper["journalInfo"]["journal"]["title"], "year": int(paper["pubYear"]),
                      "did": copy.deepcopy(writing["papers"][pmid]), "relation": copy.deepcopy(prose["relations"][rid])}
            reading["reference_details"][rid] = detail
            note_rows[cid][rid] = {f"{key}_{lang}": detail[key][lang]
                                   for key in ("did", "relation") for lang in ("zh", "en")}
        original = mapping[cid]
        if "source_evidence" in original:
            candidate = original["source_evidence"]["candidate"]
            cited = {str(x["record"]["source_paper"]["pmid"])
                     for x in candidate["first_occurrence_cited_KG_records"]
                     if x.get("present", True)}
            assert cited == {str(r["pmid"]) for r in pre["references"]}
            original_ids = candidate["canonical_record"]["evidence_ids"]
        else:
            original_ids = [p["claim_id"] for p in original["original_proposal"]["evidence_path"]]
            assert len(original_ids) == len(pre["references"])
        audit[cid] = {"reference_count": len(pre["references"]), "pmids": [r["pmid"] for r in pre["references"]],
                      "original_claim_ids": original_ids, "scale_facts": facts}
    pack["organizer"]["presentation_context"] = {
        "parent_pack": PARENT.name, "parent_sha256": PARENT_SHA, "literature_source_sha256": LITERATURE_SHA,
        "source_url": literature["url"], "retrieved_at": literature["retrieved_at"],
        "reference_count": sum(r["reference_count"] for r in audit.values()), "distinct_papers": len(papers),
        "new_references_added": 0, "case_audit": audit,
        "scope": "All citations in the executed hypotheses are displayed. This is not an exhaustive literature search. "
                 "Packet-05's earlier draft cited additional records removed by the original pre-experiment review; "
                 "they are not silently restored as support for the revised hypothesis. "
                 "Source bibliography remains unchanged; verified titles and journal years are display annotations. "
                 "All numerical results, selection, source mappings, scoring and recorded feedback remain unchanged."}
    raw = serialize(pack)
    binding = {"version": 2, "pack_id": PACK_ID, "pack_sha256": hashlib.sha256(raw).hexdigest()}
    outputs = {"cs1_discovery_pilot_v12.json": raw}
    for kind, old_version, new_version in (("assignments", 7, 8), ("reference_notes", 6, 7), ("significance", 6, 7)):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v{old_version}.json").read_bytes())
        sidecar.update(binding)
        if kind == "reference_notes":
            sidecar["notes"] = note_rows
        outputs[f"cs1_discovery_{kind}_v{new_version}.json"] = serialize(sidecar)
    catalog = json.loads((HERE / "cs1_discovery_en_v9.json").read_bytes())
    catalog.update(version="discovery-en-v11", source_pack="cs1_discovery_pilot_v12.json", source_sha256=binding["pack_sha256"])
    catalog["strings"][pack["public_meta"]["version_label"]] = "v12 · 2026-09-20 · Multiple topics"
    def add_bilingual(value):
        if isinstance(value, dict):
            if set(value) == {"zh", "en"}:
                previous = catalog["strings"].get(value["zh"])
                if previous is not None and previous != value["en"]:
                    raise ValueError("A new translation would alter a historical string.")
                catalog["strings"][value["zh"]] = value["en"]
            else:
                for item in value.values():
                    add_bilingual(item)
    for card in pack["cards"]:
        add_bilingual(card["pre"]["reading"])
    # Keep all historical strings as well as complete translations of the new prose.
    outputs["cs1_discovery_en_v11.json"] = serialize(catalog)
    return outputs


if __name__ == "__main__":
    for name, raw in build().items():
        target = HERE / name
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError(f"Different release exists: {target}")
        else:
            with target.open("xb") as handle:
                handle.write(raw)
        print(name, hashlib.sha256(raw).hexdigest())
