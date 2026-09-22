"""Select a success-led, 40% connectivity panel from unchanged completed cases."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
PARENT = HERE / "cs1_discovery_pilot_v10.json"
PARENT_SHA = "2400d3d7bf1c701c7a9f0cef47420b43d6da08b0a0b37b6e3cd007d083828e91"
PACK_ID = "neurodiscovery-multitopic-20260920-v11"
ALLOCATION_REVISION = "he1-topic40-success-led-v7"
CASE1 = ("packet-01", "packet-02", "packet-03", "packet-05")
OTHER_SUPPORTED = ("packet-36", "packet-37", "packet-46")
PARTIAL = ("packet-38", "packet-42", "packet-44")
SELECTED = tuple(sorted(CASE1 + OTHER_SUPPORTED + PARTIAL))


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def matching_supported_configs(card):
    return [cfg for cfg in card["post"].get("experimental_results", [])
            if all(cfg[phase]["status"] == "complete"
                   and cfg[phase]["supported"] is True
                   and cfg[phase]["mechanical_matches_generated_direction"] is True
                   for phase in ("internal", "external"))
            and cfg["internal"]["primary"]["effect"] * cfg["external"]["primary"]["effect"] > 0]


def build():
    raw = PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError("The frozen parent material changed.")
    parent = json.loads(raw)
    by_id = {c["id"]: c for c in parent["cards"]}
    mapping = {m.get("id", m.get("card_id")): m for m in parent["organizer"]["mapping"]}
    evidence = {}
    for cid in CASE1:
        card = by_id[cid]
        assessment = mapping[cid]["source_evidence"]["candidate"]["original_external_assessment"]
        assert assessment["pragmatic_triage_pass"] is True
        assert assessment["UCLA_all_named_same_direction"] is True
        assert all(row["standardized_beta"] < 0 for row in card["post"]["internal_rows"])
        primary = [r for r in card["post"]["external_rows"] if r["role"] == "whole_candidate_components"]
        assert primary and all(row["standardized_beta"] < 0 for row in primary)
        # Original exploratory criteria are retained; direction frequency is not a P value.
        evidence[cid] = {"role": "main_supported", "original_external_triage_pass": True,
                         "tcp_joint": card["post"]["tcp_joint"], "ucla_joint": card["post"]["ucla_joint"],
                         "formal_confirmation": assessment["formal_confirmation"]}
    for cid in OTHER_SUPPORTED:
        supported = matching_supported_configs(by_id[cid])
        assert supported, cid
        evidence[cid] = {"role": "main_supported", "matching_supported_configs": [
            {"model": c["model"], "horizon_years": c["horizon_years"],
             "internal_q": c["internal"]["q_value"], "external_q": c["external"]["q_value"]}
            for c in supported]}
    for cid in PARTIAL:
        card = by_id[cid]
        assert not matching_supported_configs(card), cid
        assert all(c[p]["mechanical_matches_generated_direction"] is True
                   for c in card["post"]["experimental_results"] for p in ("internal", "external"))
        assert any(c[p]["supported"] is True
                   for c in card["post"]["experimental_results"] for p in ("internal", "external"))
        evidence[cid] = {"role": "supplementary_partial", "both_cohorts_supported": False}

    pack = copy.deepcopy(parent)
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v11")
    pack["cards"] = [copy.deepcopy(by_id[cid]) for cid in SELECTED]
    pack["public_meta"].update(version_label="v11 · 2026-09-20 · 多主题", card_count=10, target_card_count=10)
    pack["scoring"]["applicable_items_by_card"] = {
        cid: copy.deepcopy(parent["scoring"]["applicable_items_by_card"][cid]) for cid in SELECTED}
    pack["organizer"].update(
        source_pack=PARENT.name, source_pack_sha256=PARENT_SHA,
        selection="Outcome-informed showcase: 4 connectivity, 3 imaging-genetics and 3 prognosis cases. "
                  "Seven main cases have consistent support under their original analysis criteria; "
                  "three supplementary cases have direction agreement and partial support. "
                  "All ten experts review the same ten unique cases. Historical materials remain intact. "
                  "This selected-panel rating is not a discovery-success-rate estimate.",
        mapping=[copy.deepcopy(mapping[cid]) for cid in SELECTED], selection_evidence=evidence)
    pack_raw = serialize(pack)
    binding = {"version": 2, "pack_id": PACK_ID, "pack_sha256": hashlib.sha256(pack_raw).hexdigest()}
    assignments = {**binding, "cards_per_expert": 10,
                   "experts": {f"P{i:02d}": list(SELECTED) for i in range(1, 11)},
                   "coverage": {cid: 10 for cid in SELECTED},
                   "allocation_revision": ALLOCATION_REVISION,
                   "topic_slots": {"跨诊断脑连接": 40, "影像遗传学": 30, "预后研究": 30},
                   "rule_zh": "每位专家评审同一组 10 份材料：4 份跨诊断脑连接（40%）、3 份影像遗传学和 3 份预后研究。保留现有展示顺序逻辑。",
                   "rule_en": "Each expert reviews the same 10 cases: 4 transdiagnostic connectivity cases (40%), 3 imaging-genetics cases and 3 prognosis cases. The existing presentation-order logic is unchanged."}
    assert Counter(c["pre"]["topic"] for c in pack["cards"]) == {"跨诊断脑连接": 4, "影像遗传学": 3, "预后研究": 3}
    outputs = {"cs1_discovery_pilot_v11.json": pack_raw,
               "cs1_discovery_assignments_v7.json": serialize(assignments)}
    for kind, field in (("reference_notes", "notes"), ("significance", "significance")):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v5.json").read_bytes())
        sidecar.update(binding)
        sidecar[field] = {cid: sidecar[field][cid] for cid in SELECTED}
        if kind == "reference_notes":
            sidecar["definition_plain"] = {cid: v for cid, v in sidecar.get("definition_plain", {}).items() if cid in SELECTED}
            sidecar["note_count"] = sum(len(value) for value in sidecar["notes"].values())
        outputs[f"cs1_discovery_{kind}_v6.json"] = serialize(sidecar)
    catalog = json.loads((HERE / "cs1_discovery_en_v8.json").read_bytes())
    catalog.update(version="discovery-en-v9", source_pack="cs1_discovery_pilot_v11.json", source_sha256=binding["pack_sha256"])
    # Keep old translations so saved historical sessions remain bilingual.
    catalog["strings"][pack["public_meta"]["version_label"]] = "v11 · 2026-09-20 · Multiple topics"
    outputs["cs1_discovery_en_v9.json"] = serialize(catalog)
    return outputs


if __name__ == "__main__":
    # Mechanical versioned material export; never overwrite a differing released file.
    outputs = build()
    for name, raw in outputs.items():
        target = HERE / name
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError(f"Different release already exists: {target}")
        else:
            with target.open("xb") as handle:
                handle.write(raw)
    print(json.dumps({"files": {name: hashlib.sha256(raw).hexdigest() for name, raw in outputs.items()},
                      "cases": 10, "main_supported": 7, "partial": 3, "case1_fraction": 0.4}))
