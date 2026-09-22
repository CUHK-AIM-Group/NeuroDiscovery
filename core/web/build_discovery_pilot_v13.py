"""Add source-bound forward research narratives, retaining all existing evidence."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
PARENT = HERE / "cs1_discovery_pilot_v12.json"
PARENT_SHA = "26ddc96e0a9bb13f993930259d28cd7a81114191d9e2e5d3d7e028b8d1f59dda"
EVIDENCE = HERE / "sources_v13/next_research.json"
EVIDENCE_SHA = "bd216df0fb73cd1e15195578d59389909b8e5607da9c954db460d8ae050d3dc7"
PACK_ID = "neurodiscovery-multitopic-20260920-v13"
NEXT_RESEARCH_REVISION = "source-linked-next-research-v1"


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def checked(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen source changed: {path}")
    return json.loads(raw)


def build():
    pack = copy.deepcopy(checked(PARENT, PARENT_SHA))
    evidence = checked(EVIDENCE, EVIDENCE_SHA)
    writing = json.loads((HERE / "discovery_next_research_v13_authoring.json").read_bytes())
    ids = {card["id"] for card in pack["cards"]}
    if set(writing) != set(evidence["cases"]) or set(writing) | set(evidence["omitted"]) != ids:
        raise ValueError("Each case must have documented continuation or an explicit omission record.")
    if set(writing) & set(evidence["omitted"]):
        raise ValueError("A case cannot have both continuation and omission.")
    for card in pack["cards"]:
        if card["id"] in writing:
            note = writing[card["id"]]
            if set(note) != {"zh", "en"} or not all(isinstance(v, str) and v.strip() for v in note.values()):
                raise ValueError("Both languages are required.")
            card["pre"]["reading"]["next_research"] = copy.deepcopy(note)
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v13")
    pack["public_meta"].update(version_label="v13 · 2026-09-20 · 多主题", next_research_revision=NEXT_RESEARCH_REVISION)
    pack["organizer"]["next_research_context"] = {
        "parent_pack": PARENT.name, "parent_sha256": PARENT_SHA,
        "evidence_source": "sources_v13/next_research.json", "evidence_sha256": EVIDENCE_SHA,
        "cases": copy.deepcopy(evidence["cases"]), "omitted": copy.deepcopy(evidence["omitted"]),
        "individual_parent_links_added": 0, "questions_and_applicability_unchanged": True,
        "scope": evidence["scope"],
    }
    raw = serialize(pack)
    binding = {"pack_id": PACK_ID, "pack_sha256": hashlib.sha256(raw).hexdigest()}
    outputs = {"cs1_discovery_pilot_v13.json": raw}
    for kind, old, new in (("assignments", 8, 9), ("reference_notes", 7, 8), ("significance", 7, 8)):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v{old}.json").read_bytes())
        sidecar.update(binding)
        outputs[f"cs1_discovery_{kind}_v{new}.json"] = serialize(sidecar)
    catalog = json.loads((HERE / "cs1_discovery_en_v11.json").read_bytes())
    catalog.update(version="discovery-en-v12", source_pack="cs1_discovery_pilot_v13.json", source_sha256=binding["pack_sha256"])
    for note in writing.values():
        existing = catalog["strings"].get(note["zh"])
        if existing is not None and existing != note["en"]:
            raise ValueError("A new translation would change a historical string.")
        catalog["strings"][note["zh"]] = note["en"]
    catalog["strings"][pack["public_meta"]["version_label"]] = "v13 · 2026-09-20 · Multiple topics"
    outputs["cs1_discovery_en_v12.json"] = serialize(catalog)
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
