"""Build a source-reviewed five-reference panel without rewriting prior sessions."""
import argparse
import copy
import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

HERE = Path(__file__).with_name("study_materials")
ROOT = HERE.parents[2]
PARENT = HERE / "cs1_discovery_pilot_v14.json"
PARENT_SHA = "acbc50ac1c8d724081043f6e871387b5bf1f75e7b51a4e2290102d51a9235b6d"
WRITING = HERE / "discovery_literature_v15_authoring.json"
SOURCES = HERE / "sources_v15/related_literature.json"
PACK_ID = "neurodiscovery-multitopic-20260922-v15"


def serialize(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def save_new(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Different frozen output exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def capture_sources():
    writing = json.loads(WRITING.read_bytes())
    database = ROOT / "tmp/kgwg_article_retrieval_v3/ARTICLES.sqlite"
    captured = {}
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for pmid in sorted(writing["papers"]):
            row = connection.execute("SELECT document,authority,document_sha,source_state FROM articles WHERE pmid=?", (pmid,)).fetchone()
            if row is None or row[3] != "OWN_RECORD_CACHED":
                raise ValueError(f"Missing or held source: {pmid}")
            document, authority = [json.loads(zlib.decompress(payload)) for payload in row[:2]]
            if document["pmid"] != pmid or document.get("comments_corrections") or authority.get("preprint"):
                raise ValueError(f"Identity or notice review required: {pmid}")
            xml_path = Path(document["source"]["path"])
            raw = xml_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != document["source"]["sha256"]:
                raise ValueError(f"Owning XML changed: {pmid}")
            matches = [record for record in ET.fromstring(raw).findall("PubmedArticle")
                       if record.findtext("MedlineCitation/PMID") == pmid]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous owning XML: {pmid}")
            article = matches[0].find("MedlineCitation/Article")
            title = "".join(article.find("ArticleTitle").itertext())
            journal = article.findtext("Journal/Title")
            date = article.find("Journal/JournalIssue/PubDate")
            year = date.findtext("Year") or (date.findtext("MedlineDate") or "")[:4]
            if not journal or not year.isdigit():
                raise ValueError(f"Missing journal/year: {pmid}")
            captured[pmid] = {"document": document, "authority": authority,
                              "source_state": row[3], "document_sha": row[2],
                              "title": title, "journal": journal, "year": int(year)}
    save_new(SOURCES, serialize({"review_date": writing["review_date"],
                                "source_database": str(database.relative_to(ROOT)), "papers": captured}))


def build():
    raw_parent = PARENT.read_bytes()
    if hashlib.sha256(raw_parent).hexdigest() != PARENT_SHA:
        raise ValueError("Frozen parent changed")
    parent = json.loads(raw_parent)
    pack = copy.deepcopy(parent)
    writing = json.loads(WRITING.read_bytes())
    sources = json.loads(SOURCES.read_bytes())["papers"]
    if set(sources) != set(writing["papers"]) or set(writing["additions"]) != {card["id"] for card in pack["cards"]}:
        raise ValueError("Review coverage mismatch")
    existing = {}
    for card in parent["cards"]:
        for reference in card["pre"]["references"]:
            existing[reference["pmid"]] = (reference, card["pre"]["reading"]["reference_details"][reference["id"]])
    notes = json.loads((HERE / "cs1_discovery_reference_notes_v9.json").read_bytes())
    audit = {}
    for card in pack["cards"]:
        card_id, pre = card["id"], card["pre"]
        reading = pre["reading"]
        audit[card_id] = {"original_pmids": [ref["pmid"] for ref in pre["references"]], "added": []}
        for pmid in writing["additions"][card_id]:
            reference_id = f"P{len(pre['references']) + 1}"
            if pmid in existing:
                reference, detail = copy.deepcopy(existing[pmid])
                detail["relation"] = copy.deepcopy(writing["reused_relations"][card_id][pmid])
                review = {"reused_review": "sources_v12/literature.json", "scope": "Previously reviewed abstract; case-specific relevance reassessed"}
            else:
                source, authored = sources[pmid], writing["papers"][pmid]
                document = source["document"]
                abstract = "\n".join(part["text"] for part in document["abstract"])
                if authored["anchor"].casefold() not in abstract.casefold():
                    raise ValueError(f"Review anchor not in source: {pmid}")
                if document["pmid"] != pmid or document.get("comments_corrections") or source["source_state"] != "OWN_RECORD_CACHED":
                    raise ValueError(f"Invalid reviewed source: {pmid}")
                detail = {key: source[key] for key in ("title", "journal", "year")}
                detail.update({key: copy.deepcopy(authored[key]) for key in ("did", "relation")})
                reference = {"pmid": pmid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                             "title": detail["title"], "journal": detail["journal"], "year": detail["year"],
                             "verification": "Owning cached PubMed abstract reviewed 2026-09-22; relevance is not independent replication.",
                             "note": detail["relation"]["zh"], "recorded_sentence": detail["did"]["en"]}
                review = {"document_sha": source["document_sha"], "anchor": authored["anchor"], "scope": "own_abstract"}
            reference["id"] = reference_id
            pre["references"].append(reference)
            reading["reference_details"][reference_id] = detail
            reading["references"][reference_id] = copy.deepcopy(detail["relation"])
            notes["notes"][card_id][reference_id] = {f"{key}_{language}": detail[key][language]
                                                   for key in ("did", "relation") for language in ("zh", "en")}
            audit[card_id]["added"].append({"pmid": pmid, **review})
        pmids = [ref["pmid"] for ref in pre["references"]]
        if len(pmids) < 5 or len(pmids) != len(set(pmids)):
            raise ValueError(f"At least five distinct papers required: {card_id}")
    pack.update(pack_id=PACK_ID, protocol_version="complete-output-multitopic-v15")
    pack["public_meta"].update(version_label="v15 · 2026-09-22 · 多主题", related_literature_revision="five-reviewed-references-v1")
    pack["organizer"]["related_literature"] = {
        "parent_pack": PARENT.name, "parent_sha256": PARENT_SHA,
        "source_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
        "authoring_sha256": hashlib.sha256(WRITING.read_bytes()).hexdigest(),
        "case_audit": audit, "review_scope": writing["review_scope"],
        "display": "One related-studies list; original citation provenance retained only in audit metadata.",
        "scientific_results_questions_and_scoring_unchanged": True,
        "independent_cohort_replication_not_established": True,
    }
    raw = serialize(pack)
    binding = {"pack_id": PACK_ID, "pack_sha256": hashlib.sha256(raw).hexdigest()}
    outputs = {"cs1_discovery_pilot_v15.json": raw}
    notes.update(binding, note_count=sum(len(card["pre"]["references"]) for card in pack["cards"]))
    outputs["cs1_discovery_reference_notes_v10.json"] = serialize(notes)
    for kind, old, new in (("assignments", 10, 11), ("significance", 9, 10)):
        sidecar = json.loads((HERE / f"cs1_discovery_{kind}_v{old}.json").read_bytes())
        sidecar.update(binding)
        outputs[f"cs1_discovery_{kind}_v{new}.json"] = serialize(sidecar)
    catalog = json.loads((HERE / "cs1_discovery_en_v13.json").read_bytes())
    catalog.update(version="discovery-en-v14", source_pack="cs1_discovery_pilot_v15.json", source_sha256=binding["pack_sha256"])
    for card in pack["cards"]:
        for detail in card["pre"]["reading"]["reference_details"].values():
            for key in ("did", "relation"):
                if detail[key]["zh"] in catalog["strings"] and catalog["strings"][detail[key]["zh"]] != detail[key]["en"]:
                    raise ValueError("Historical translation conflict")
                catalog["strings"][detail[key]["zh"]] = detail[key]["en"]
    catalog["strings"][pack["public_meta"]["version_label"]] = "v15 · 2026-09-22 · Multiple topics"
    outputs["cs1_discovery_en_v14.json"] = serialize(catalog)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-sources", action="store_true")
    args = parser.parse_args()
    if args.capture_sources:
        capture_sources()
    else:
        for name, raw in build().items():
            save_new(HERE / name, raw)
            print(name, hashlib.sha256(raw).hexdigest())
