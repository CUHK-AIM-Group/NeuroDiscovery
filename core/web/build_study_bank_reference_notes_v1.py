"""Build reference notes v1 for the HE2 v2 bank.

Owner request (2026-09-21): the auto-matched literature panel in Human
Evaluation 2 should be completed as far as possible from LOCAL sources.
Abstracts exist locally; cohort sizes and p-value mentions are extracted
from the local abstract text on a best-effort basis and labelled as such.

Pipeline (deterministic; read-only against frozen stores):
  bank      neurooracle/data/user_study/case1_tcp_expert_study_v2.json (SHA-pinned)
  abstracts tmp/kg_rebuild_full_20260920_v1/CORPUS.sqlite sources payloads
            (frozen corpus, read-only), falling back to the PubMed
            abstract_cache.jsonl snapshot
  cohort    first "N = n" in the abstract, else the largest explicit
            "<n> participants/patients/subjects/..." count; always labelled
            auto-extracted, never presented as verified
  p-values  first "p </=/> x" mention in the abstract; status stays
            "reported_elsewhere_in_abstract" because the mention is not tied
            to the evidence sentences selected for the hypothesis
The sidecar never rewrites the bank; sessions merge it at creation time.
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
TARGET = BANK.with_name(BANK.stem + "_reference_notes_v1.json")

N_EQUALS = re.compile(r"\bN\s*=\s*(\d{2,5})\b")
COUNT_KEYWORD = re.compile(
    r"\b(\d{2,4})\s+(?:participants|patients|subjects|individuals|volunteers|adults)\b", re.IGNORECASE)
P_VALUE = re.compile(r"\bp\s*([<=>])\s*((?:0?\.\d+|1\.0+))\b", re.IGNORECASE)
MIN_ABSTRACT_CHARS = 80


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


def _corpus_abstracts(pmids):
    """pmid -> (abstract, source_label) from the frozen corpus (read-only)."""
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
    match = N_EQUALS.search(abstract)
    if match:
        return int(match.group(1)), "n_equals"
    counts = [int(m.group(1)) for m in COUNT_KEYWORD.finditer(abstract)]
    if counts:
        return max(counts), "count_keyword"
    return None, ""


def _extract_p_value(abstract):
    match = P_VALUE.search(abstract)
    if not match:
        return ""
    return f"p {match.group(1)} {match.group(2)}"


def build():
    refs = _load_bank_refs()
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

    notes = {}
    for pmid, key in sorted(pmids.items(), key=lambda item: item[1]):
        found = resolved.get(pmid)
        if not found:
            continue
        abstract, source_label = found
        entry = {
            "abstract": abstract,
            "abstract_source": source_label,
            "abstract_verified": True,
        }
        credibility = {}
        cohort_n, method = _extract_cohort(abstract)
        if cohort_n:
            credibility["cohort"] = {
                "n_total": cohort_n,
                "display_zh": f"≈{cohort_n}（摘要自动提取，未经人工核对）",
                "display_en": f"≈{cohort_n} (auto-extracted from the abstract, not manually verified)",
                "extraction": method,
            }
        p_value = _extract_p_value(abstract)
        if p_value:
            credibility["p_values"] = {
                "status": "reported_elsewhere_in_abstract",
                "display": p_value,
            }
        if credibility:
            entry["credibility"] = credibility
        notes[key] = entry

    doc = {
        "schema_version": "case1-tcp-expert-study-v2-reference-notes-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank_file": BANK.name,
        "bank_sha256": BANK_SHA256,
        "sources": {
            "corpus": str(CORPUS.relative_to(ROOT)),
            "pubmed_abstract_cache": str(PUBMED_CACHE.relative_to(ROOT)),
        },
        "disclosure_zh": ("本地摘要来自冻结语料库与 PubMed 摘要缓存；队列规模与 p 值为摘要文本"
                          "规则提取，未经人工核对；仍属自动匹配参考文献，本轮未逐条人工复核。"),
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
    cohorts = sum(1 for n in doc["references"].values() if n.get("credibility", {}).get("cohort"))
    pvals = sum(1 for n in doc["references"].values() if n.get("credibility", {}).get("p_values"))
    print(json.dumps({
        "notes": TARGET.name,
        "notes_sha256": hashlib.sha256(data).hexdigest(),
        "unique_bank_references": doc["unique_bank_references"],
        "without_bank_abstract": doc["references_without_bank_abstract"],
        "abstracts_resolved": len(doc["references"]),
        "cohort_extracted": cohorts,
        "p_value_extracted": pvals,
    }, indent=2))


if __name__ == "__main__":
    main()
