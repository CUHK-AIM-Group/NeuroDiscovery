"""Build reference notes v2 for the HE2 v2 bank.

Owner requests (2026-09-21): complete the Study-details panel as far as
possible — local abstracts (v1), a stronger cohort-size extractor, and the
current journal impact factor looked up ONLINE by the owner request and
curated into a pinned table before this build runs.

Differences from v1:
  cohort  case-insensitive "n = N" counts join the people-keyword counts and
          the pooled maximum wins (totals exceed subgroup counts far more
          often than not); "controls" joins the people keywords
  jif     every reference whose journal appears in the curated table gets
          credibility.journal_impact_factor; preprint servers are recorded
          as not_applicable and unknown journals as not_verified
  cover   every unique bank reference gets an entry, even without an
          abstract, so JIF/cohort reach all 143 records

The JIF table is curated data (web-searched values); this builder never
searches the web itself. The bank stays untouched; sessions merge the
sidecar at creation time.
"""
import hashlib
import json
import re
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BANK = ROOT / "neurooracle" / "data" / "user_study" / "case1_tcp_expert_study_v2.json"
BANK_SHA256 = "ff9b1d51e469f64bc336d6035271992f1ad1a8eed94750255e0e7e8d5986393d"
PUBMED_CACHE = ROOT / "neurooracle" / "data" / "full_snapshot_v2" / "abstract_cache.jsonl"
CORPUS = ROOT / "tmp" / "kg_rebuild_full_20260920_v1" / "CORPUS.sqlite"
JIF_TABLE = ROOT / "neurooracle" / "data" / "user_study" / "journal_impact_factors_2024_v1.json"
JIF_TABLE_SHA256 = "48cb353debae40023cae040d7146451bc369f73de6b78dda6bff0ece26babd40"
TARGET = BANK.with_name(BANK.stem + "_reference_notes_v2.json")

N_EQUALS = re.compile(r"\bn\s*=\s*(\d{2,5})\b", re.IGNORECASE)
COUNT_KEYWORD = re.compile(
    r"\b(\d{2,4})\s+(?:participants|patients|subjects|individuals|volunteers|adults|controls?)\b",
    re.IGNORECASE)
P_VALUE = re.compile(r"\bp\s*([<=>])\s*((?:0?\.\d+|1\.0+))\b", re.IGNORECASE)
MIN_ABSTRACT_CHARS = 80
PREPRINT_JOURNALS = {"medrxiv", "biorxiv", "research square"}


def canonical_journal(journal):
    """Match the census normalisation: split preprint suffixes, strip punctuation."""
    text = str(journal or "").strip().split(";")[0].strip()
    text = re.sub(r"\s*[:：]\s*", " ", text)
    text = re.sub(r"[.,()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    aliases = {"brain": "brain a journal of neurology"}
    return aliases.get(text, text)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference_key(paper):
    return str(paper.get("pmid") or paper.get("doi") or paper.get("title") or "").strip().lower()


def _load_bank_refs():
    if _sha256(BANK) != BANK_SHA256:
        raise ValueError("The v2 bank changed; build a separately reviewed notes version.")
    bank = json.loads(BANK.read_bytes().decode("utf-8-sig"))
    refs = {}
    for hypothesis in bank["hypotheses"]:
        for paper in hypothesis.get("literature", []):
            key = _reference_key(paper)
            if key:
                refs.setdefault(key, paper)
    return refs


def _load_jif_table():
    if JIF_TABLE_SHA256 == "PENDING":
        raise ValueError("Pin JIF_TABLE_SHA256 after curating the JIF table.")
    if _sha256(JIF_TABLE) != JIF_TABLE_SHA256:
        raise ValueError("The curated JIF table changed; pin the new hash in a reviewed build.")
    table = json.loads(JIF_TABLE.read_bytes().decode("utf-8-sig"))
    journals = table.get("journals") or {}
    if not journals:
        raise ValueError("The curated JIF table is empty.")
    return journals


def _corpus_abstracts(pmids):
    if not CORPUS.exists():
        raise FileNotFoundError(f"Frozen corpus not found: {CORPUS}")
    resolved = {}
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        pub_of = {}
        for pub_id, ids_json in cur.execute("SELECT pub_id, ids_json FROM publications"):
            try:
                ids = json.loads(ids_json)
            except (TypeError, json.JSONDecodeError):
                continue
            pmid = str(ids.get("pmid") or "").strip()
            if pmid in pmids and pmid not in pub_of:
                pub_of[pmid] = pub_id
        for pmid, pub_id in pub_of.items():
            rows = cur.execute(
                "SELECT quality, payload FROM sources WHERE pub_id = ?"
                " AND abstract_sha IS NOT NULL AND abstract_sha != ''",
                (pub_id,),
            ).fetchall()
            best = None
            for quality, blob in rows:
                try:
                    obj = json.loads(zlib.decompress(blob).decode("utf-8", errors="replace"))
                except Exception:
                    continue
                text = str(obj.get("abstract") or obj.get("abstract_text") or "").strip()
                if len(text) < MIN_ABSTRACT_CHARS:
                    continue
                candidate = (text, f"corpus_{str(quality or 'source').lower()}")
                if best is None or quality == "HISTORICAL_ABSTRACT_CACHE":
                    best = candidate
                if quality == "HISTORICAL_ABSTRACT_CACHE":
                    break
            if best:
                resolved[pmid] = best
    finally:
        con.close()
    return resolved


def _pubmed_cache_abstracts(pmids):
    resolved = {}
    if not PUBMED_CACHE.exists():
        return resolved
    with PUBMED_CACHE.open("rb") as handle:
        for line in handle:
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            pmid = str(obj.get("pmid") or "").strip()
            abstract = str(obj.get("abstract") or "").strip()
            if pmid in pmids and len(abstract) >= MIN_ABSTRACT_CHARS and pmid not in resolved:
                resolved[pmid] = (abstract, "pubmed_abstract_cache")
    return resolved


def _extract_cohort(abstract):
    counts = [int(m.group(1)) for m in N_EQUALS.finditer(abstract)]
    counts += [int(m.group(1)) for m in COUNT_KEYWORD.finditer(abstract)]
    if not counts:
        return None, ""
    return max(counts), "max_of_n_equals_and_count_keyword"


def _extract_p_value(abstract):
    match = P_VALUE.search(abstract)
    if not match:
        return ""
    number = match.group(2)
    if number.startswith("."):
        number = "0" + number
    return f"p {match.group(1)} {number}"


def _jif_entry(journal, jif_journals):
    canonical = canonical_journal(journal)
    if not canonical:
        return {"status": "not_available"}
    if canonical in PREPRINT_JOURNALS:
        return {"status": "not_applicable"}
    record = jif_journals.get(canonical)
    if not record:
        return {"status": "not_verified"}
    if record.get("status") == "not_applicable":
        return {"status": "not_applicable"}
    value = record.get("value")
    if value is None:
        return {"status": "not_available"}
    entry = {"value": float(value), "status": "reported_online_search"}
    if record.get("metric_year"):
        entry["metric_year"] = str(record["metric_year"])
    return entry


def build():
    refs = _load_bank_refs()
    jif_journals = _load_jif_table()
    wanted = {
        key: paper for key, paper in refs.items()
        if not str(paper.get("abstract") or "").strip()
    }
    pmids = {
        str(paper.get("pmid") or "").strip(): key
        for key, paper in wanted.items()
        if str(paper.get("pmid") or "").strip()
    }
    resolved = _corpus_abstracts(set(pmids))
    for pmid, found in _pubmed_cache_abstracts(set(pmids)).items():
        resolved.setdefault(pmid, found)
    key_by_pmid = {pmid: key for pmid, key in pmids.items()}

    notes = {}
    for key, paper in sorted(refs.items()):
        entry = {}
        pmid = str(paper.get("pmid") or "").strip()
        found = resolved.get(pmid) if key_by_pmid.get(pmid) == key else None
        if found and not str(paper.get("abstract") or "").strip():
            abstract, source_label = found
            entry.update({
                "abstract": abstract,
                "abstract_source": source_label,
                "abstract_verified": True,
            })
        credibility = {}
        abstract_text = entry.get("abstract") or str(paper.get("abstract") or "")
        if abstract_text:
            cohort_n, method = _extract_cohort(abstract_text)
            if cohort_n:
                credibility["cohort"] = {
                    "n_total": cohort_n,
                    "display_zh": f"≈{cohort_n}（摘要自动提取，未经人工核对）",
                    "display_en": f"≈{cohort_n} (auto-extracted from the abstract, not manually verified)",
                    "extraction": method,
                }
            p_value = _extract_p_value(abstract_text)
            if p_value:
                credibility["p_values"] = {
                    "status": "reported_elsewhere_in_abstract",
                    "display": p_value,
                }
        credibility["journal_impact_factor"] = _jif_entry(paper.get("journal"), jif_journals)
        if credibility:
            entry["credibility"] = credibility
        if entry:
            notes[key] = entry

    doc = {
        "schema_version": "case1-tcp-expert-study-v2-reference-notes-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank_file": BANK.name,
        "bank_sha256": BANK_SHA256,
        "sources": {
            "corpus": str(CORPUS.relative_to(ROOT)),
            "pubmed_abstract_cache": str(PUBMED_CACHE.relative_to(ROOT)),
            "journal_impact_factors": str(JIF_TABLE.relative_to(ROOT)),
        },
        "disclosure_zh": ("本地摘要来自冻结语料库与 PubMed 摘要缓存；队列规模与 p 值为摘要文本"
                          "规则提取，未经人工核对；期刊 JIF 来自联网检索的整理表（标注年份），"
                          "属期刊层面指标；仍属自动匹配参考文献，本轮未逐条人工复核。"),
        "unique_bank_references": len(refs),
        "references_without_bank_abstract": len(wanted),
        "references": notes,
    }
    return doc


def main():
    doc = build()
    data = (json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n").encode("utf-8")
    if TARGET.exists() and TARGET.read_bytes() != data:
        raise ValueError("A different reference-notes file already exists; use a new version.")
    TARGET.write_bytes(data)
    refs = doc["references"]
    abstracts = sum(1 for n in refs.values() if n.get("abstract"))
    cohorts = sum(1 for n in refs.values() if (n.get("credibility") or {}).get("cohort"))
    pvals = sum(1 for n in refs.values() if (n.get("credibility") or {}).get("p_values"))
    jifs = sum(1 for n in refs.values() if (n.get("credibility") or {}).get("journal_impact_factor", {}).get("value") is not None)
    print(json.dumps({
        "notes": TARGET.name,
        "notes_sha256": hashlib.sha256(data).hexdigest(),
        "unique_bank_references": doc["unique_bank_references"],
        "entries": len(refs),
        "abstracts_resolved": abstracts,
        "cohort_extracted": cohorts,
        "p_value_extracted": pvals,
        "jif_reported": jifs,
    }, indent=2))


if __name__ == "__main__":
    main()
