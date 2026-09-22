"""Present exact preceding/follow-up hypotheses in compact bilingual reading blocks."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
PARENT = HERE / "cs1_discovery_pilot_v13.json"
PARENT_SHA = "70c015ee0c101eab73f1c3aa4da531532113b10f20e1edc31f44f2c0e141eb4d"
EVIDENCE = HERE / "sources_v14/lineage_details.json"
EVIDENCE_SHA = "6ca9da47e323431a5352adaf7f7adc3e49cbb595670884b887ac00de7bfaa939"
PACK_ID = "neurodiscovery-multitopic-20260920-v14"
LINEAGE_REVISION = "specific-hypothesis-lineage-v1"


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen source changed: {path}")
    return json.loads(raw)


def validate_chain(chain, mode, forward):
    fields = {"mode", "source", "feedback", "next", "change"} | ({"outcome"} if forward else set())
    if set(chain) != fields or chain["mode"] != mode or not chain["next"]:
        raise ValueError("Each displayed chain must match its documented relationship and stage.")
    for note in [chain[k] for k in ("source", "feedback", "change")] + chain["next"] + ([chain["outcome"]] if forward else []):
        if set(note) != {"zh", "en"} or not all(isinstance(v, str) and v.strip() for v in note.values()):
            raise ValueError("Each chain statement requires both languages.")


def build():
    pack = copy.deepcopy(checked(PARENT, PARENT_SHA))
    evidence = checked(EVIDENCE, EVIDENCE_SHA)
    writing = json.loads((HERE / "discovery_lineage_v14_authoring.json").read_bytes())
    if set(writing) != {"prior", "forward"}:
        raise ValueError("Unexpected narrative fields.")
    if any(set(writing[k]) != set(evidence[k]) for k in writing):
        raise ValueError("Narratives must match the documented case selection.")
    for card in pack["cards"]:
        cid, reading = card["id"], card["pre"]["reading"]
        if cid in writing["prior"]:
            chain = writing["prior"][cid]
            validate_chain(chain, "explicit_parent", False)
            if not reading.get("feedback") or not card["post"]["feedback"].get("binding_verified"):
                raise ValueError("A preceding chain needs an existing verified parent.")
            reading["feedback_chain"] = copy.deepcopy(chain)
        if cid in writing["forward"]:
            chain = writing["forward"][cid]
            mode = "explicit_parent" if evidence["forward"][cid]["kind"] == "explicit_parent_and_completed_child" else "round_followup"
            validate_chain(chain, mode, True)
            if not reading.get("next_research"):
                raise ValueError("No new forward relationship may be invented by the display.")
            reading["next_research_chain"] = copy.deepcopy(chain)
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v14")
    pack["public_meta"].update(version_label="v14 · 2026-09-20 · 多主题", lineage_detail_revision=LINEAGE_REVISION)
    pack["organizer"]["lineage_details"] = {
        "parent_pack": PARENT.name, "parent_sha256": PARENT_SHA,
        "evidence_source": "sources_v14/lineage_details.json", "evidence_sha256": EVIDENCE_SHA,
        "prior_cases": list(writing["prior"]), "forward_cases": list(writing["forward"]),
        "new_relationships_added": 0, "scientific_results_and_questions_unchanged": True,
    }
    raw = serialize(pack)
    binding = {"pack_id": PACK_ID, "pack_sha256": hashlib.sha256(raw).hexdigest()}
    outputs = {"cs1_discovery_pilot_v14.json": raw}
    for kind, old, new in (("assignments", 9, 10), ("reference_notes", 8, 9), ("significance", 8, 9)):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v{old}.json").read_bytes())
        sidecar.update(binding)
        outputs[f"cs1_discovery_{kind}_v{new}.json"] = serialize(sidecar)
    catalog = json.loads((HERE / "cs1_discovery_en_v12.json").read_bytes())
    catalog.update(version="discovery-en-v13", source_pack="cs1_discovery_pilot_v14.json", source_sha256=binding["pack_sha256"])

    def add(value):
        if isinstance(value, dict):
            if set(value) == {"zh", "en"}:
                previous = catalog["strings"].get(value["zh"])
                if previous is not None and previous != value["en"]:
                    raise ValueError("A new translation would alter a historical string.")
                catalog["strings"][value["zh"]] = value["en"]
            else:
                for child in value.values():
                    add(child)
        elif isinstance(value, list):
            for child in value:
                add(child)
    add(writing)
    catalog["strings"][pack["public_meta"]["version_label"]] = "v14 · 2026-09-20 · Multiple topics"
    outputs["cs1_discovery_en_v13.json"] = serialize(catalog)
    return outputs


if __name__ == "__main__":
    for name, raw in build().items():
        target = HERE / name
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError(f"Different release exists: {target}")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
        print(name, hashlib.sha256(raw).hexdigest())
