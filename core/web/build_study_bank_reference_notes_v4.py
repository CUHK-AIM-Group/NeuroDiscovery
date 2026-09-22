"""Build reference notes v4 for the HE2 v3 bank (graph-rematched literature).

Owner decisions (2026-09-21):
  * p-values are dropped from the HE2 credibility panel entirely; citation
    counts (OpenAlex cited_by_count) replace them.
  * The v3 bank re-matched each hypothesis to the closest in-graph works by
    structured disease/region/measure/direction facets; these notes complete
    the per-reference credibility sidecar for that bank.

Differences from notes v2:
  bank    the v3 graph-rematched bank (pinned sha)
  pvalue  NOT extracted and NOT emitted anywhere
  cit     credibility.citation_count from OpenAlex, merged from the v2-bank
          v1 counts and the 2026-09-21 OpenAlex fetch (case-insensitive
          pmid/doi index); unresolved preprints simply omit the field
  jif     the extended curated 2024 table (v2, 88 journals)
  cover   every unique bank reference gets an entry, even without an abstract

The bank and the older sidecars stay untouched; sessions merge this sidecar
at creation time through the user_study notes fallback chain.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.web.build_study_bank_reference_notes_v2 import (
    CORPUS,
    PUBMED_CACHE,
    _corpus_abstracts,
    _extract_cohort,
    _jif_entry,
    _pubmed_cache_abstracts,
    _reference_key,
    _sha256,
)

ROOT = Path(__file__).resolve().parent.parent.parent
BANK = ROOT / "neurooracle" / "data" / "user_study" / "case1_tcp_expert_study_v3.json"
BANK_SHA256 = "b9d7d026b221ddc526011407737d3075c003790b436b7300d128fbf9863b98c8"
JIF_TABLE = ROOT / "neurooracle" / "data" / "user_study" / "journal_impact_factors_2024_v2.json"
JIF_TABLE_SHA256 = "679f567de9d647cfecf70b79a042aeefa82152b40943481b9e39a2a17025f383"
CIT_V1 = ROOT / "neurooracle" / "data" / "user_study" / "case1_tcp_expert_study_v2_citation_counts_v1.json"
CIT_EXT = ROOT / "tmp" / "he2_literature_rematch_20260921" / "citation_counts_fetched_v2.json"
CIT_EXT_SHA256 = "90cfd68180637cca03b762e299baf122b247cf7da5d84f73fb028d69d9b73acd"
TARGET = BANK.with_name(BANK.stem + "_reference_notes_v4.json")


def _load_bank_refs():
    if _sha256(BANK) != BANK_SHA256:
        raise ValueError("The v3 bank changed; build a separately reviewed notes version.")
    bank = json.loads(BANK.read_bytes().decode("utf-8-sig"))
    refs = {}
    for hypothesis in bank["hypotheses"]:
        for paper in hypothesis.get("literature", []):
            key = _reference_key(paper)
            if key:
                refs.setdefault(key, paper)
    return refs


def _load_citations():
    if _sha256(CIT_EXT) != CIT_EXT_SHA256:
        raise ValueError("The fetched citation-count file changed; re-pin its hash.")
    merged = {}
    for path in (CIT_V1, CIT_EXT):
        doc = json.loads(path.read_bytes().decode("utf-8-sig"))
        for key, value in (doc.get("citations") or {}).items():
            merged[str(key).strip().lower()] = value
    return merged


def _load_jif_journals():
    if _sha256(JIF_TABLE) != JIF_TABLE_SHA256:
        raise ValueError("The curated JIF v2 table changed; pin the new hash in a reviewed build.")
    table = json.loads(JIF_TABLE.read_bytes().decode("utf-8-sig"))
    journals = table.get("journals") or {}
    if not journals:
        raise ValueError("The curated JIF table is empty.")
    return journals


def _citation_entry(paper, index):
    for ident in (paper.get("pmid"), paper.get("doi")):
        text = str(ident or "").strip()
        if text and text.lower() in index:
            found = index[text.lower()]
            entry = {
                "count": int(found["count"]),
                "source": found.get("source", "OpenAlex"),
                "retrieved_at": found.get("retrieved_at", ""),
            }
            if found.get("source_url"):
                entry["source_url"] = found["source_url"]
            return entry
    return None


def build():
    refs = _load_bank_refs()
    jif_journals = _load_jif_journals()
    citations = _load_citations()

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
        citation = _citation_entry(paper, citations)
        if citation is not None:
            credibility["citation_count"] = citation
        credibility["journal_impact_factor"] = _jif_entry(paper.get("journal"), jif_journals)
        if credibility:
            entry["credibility"] = credibility
        if entry:
            notes[key] = entry

    doc = {
        "schema_version": "case1-tcp-expert-study-v3-reference-notes-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bank_file": BANK.name,
        "bank_sha256": BANK_SHA256,
        "sources": {
            "corpus": str(CORPUS.relative_to(ROOT)),
            "pubmed_abstract_cache": str(PUBMED_CACHE.relative_to(ROOT)),
            "journal_impact_factors": str(JIF_TABLE.relative_to(ROOT)),
            "citation_counts_v1": str(CIT_V1.relative_to(ROOT)),
            "citation_counts_extension": str(CIT_EXT.relative_to(ROOT)),
        },
        "disclosure_zh": ("本地摘要来自冻结语料库与 PubMed 摘要缓存；队列规模为摘要文本规则提取，未经人工核对；"
                          "引用量来自 OpenAlex（2026-09-21 检索，含 v2 bank 沿用值）；期刊 JIF 来自联网检索整理的"
                          "2024 数据年表格（个别条目为 Scopus 口径或第三方追踪值，已逐条注明）；按用户要求，"
                          "本版不再提供 p 值。参考文献按疾病/脑区/指标/方向四维与图谱结构化重配，"
                          "自动完成、未逐条人工复核。"),
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
    cites = sum(1 for n in refs.values() if (n.get("credibility") or {}).get("citation_count"))
    jifs = sum(1 for n in refs.values() if (n.get("credibility") or {}).get("journal_impact_factor", {}).get("value") is not None)
    assert not any("p_values" in (n.get("credibility") or {}) for n in refs.values())
    print(json.dumps({
        "notes": TARGET.name,
        "notes_sha256": hashlib.sha256(data).hexdigest(),
        "unique_bank_references": doc["unique_bank_references"],
        "entries": len(refs),
        "abstracts_resolved": abstracts,
        "cohort_extracted": cohorts,
        "citation_count_reported": cites,
        "jif_reported": jifs,
    }, indent=2))


if __name__ == "__main__":
    main()
