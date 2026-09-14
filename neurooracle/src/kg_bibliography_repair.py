"""Narrow offline title repair: exact identifiers plus a complete abstract quote.

Not an identity matching fallback. No title prefixes, similarity, version links,
synthetic-ID suffixes or majority votes qualify a source for repair.
"""
from copy import deepcopy
import hashlib
import html
import re
import unicodedata

from .kg_identity_pilot import digest
from .kg_paper_identity import identifiers
from .pubmed_title_evidence import complete_title_evidence


def quote_key(value):
    return " ".join(unicodedata.normalize("NFKC", html.unescape(str(value or ""))).casefold().split())


def primary_abstract(article):
    return " ".join("".join(x.itertext()) for x in article.findall("./MedlineCitation/Article/Abstract/AbstractText"))


def without_selected_title(row):
    """Mask only the single allowed field; preserve every other bit of meaning."""
    out = deepcopy(row)
    del out["metadata"]["source_paper"]["title"]
    return out


def review_title(row, registry, articles, witnesses):
    md = row["metadata"]
    source = md.get("source_paper") or {}
    before = registry.resolve(md)
    if before["status"] != "conflict" or before["reasons"] != ["complete_title_disagreement"]:
        return None, "not_an_isolated_title_conflict"
    ids = identifiers(source)
    payload = registry.export_payload()
    # Candidate identity comes exclusively from explicit valid identifiers.
    candidates = set()
    if ids["pmid"] in payload["records"]:
        candidates.add(ids["pmid"])
    for kind in ("doi", "pmcid"):
        if ids[kind]:
            candidates.update(p for p, v in payload["records"].items() if ids[kind] in v[kind])
    if len(candidates) != 1:
        return None, "identifier_owner_not_unique"
    pmid = next(iter(candidates))
    authority = payload["records"][pmid]
    if (authority["preprint"] or source.get("arxiv_id") or
        re.search(r"\b(?:biorxiv|medrxiv|arxiv|research square|preprints?)\b", str(source.get("journal") or ""), re.I) or
        re.search(r"(?:BIORXIV|MEDRXIV|ARXIV)", md["id"], re.I)):
        return None, "publication_version_out_of_scope"
    if any(holder.get("title") for holder in (md, md.get("metadata") or {})):
        return None, "alternate_title_requires_separate_review"
    article = articles.get(pmid)
    if article is None or article.tag != "PubmedArticle":
        return None, "owning_journal_xml_missing"
    # Checks owning PMID, DOI/PMCID sets, complete title and XML structure.
    complete_title_evidence(article, authority, witnesses[pmid])
    quote = quote_key(md.get("raw_text"))
    abstract = quote_key(primary_abstract(article))
    if len(quote) < 40 or len(re.findall(r"\w+", quote)) < 8 or quote not in abstract:
        return None, "complete_quote_not_in_own_primary_abstract"
    if ids["pmid"] and ids["pmid"] != pmid:
        return None, "source_pmid_disagrees"
    title = "".join(article.find("./MedlineCitation/Article/ArticleTitle").itertext()).strip()
    if not title or not isinstance(source.get("title"), str) or title == source["title"]:
        return None, "no_title_repair"
    out = deepcopy(row)
    out["metadata"]["source_paper"]["title"] = title
    after = registry.resolve(out["metadata"])
    if after != dict(paper_key="pmid:" + pmid, status="verified", reasons=[]):
        return None, "full_identity_guard_still_fails"
    assert without_selected_title(row) == without_selected_title(out)
    event = dict(claim_id=md["id"], source_node_sha256=digest(row), current_node_sha256=digest(out),
        field="metadata.source_paper.title", value=title, canonical_paper_key="pmid:" + pmid,
        raw_text_sha256=hashlib.sha256(str(md.get("raw_text") or "").encode()).hexdigest(),
        normalized_complete_quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),
        normalized_primary_abstract_sha256=hashlib.sha256(abstract.encode()).hexdigest(),
        complete_quote_characters=len(quote), public_xml_witness=deepcopy(witnesses[pmid]))
    return event, "exact_identifiers_and_owning_complete_abstract_quote"


def apply_title(row, event):
    if row["id"] != event["claim_id"] or digest(row) != event["source_node_sha256"]:
        raise ValueError("reviewed current claim changed")
    if event["field"] != "metadata.source_paper.title" or not isinstance(event["value"], str) or not event["value"]:
        raise ValueError("only a nonempty source title can be corrected")
    out = deepcopy(row)
    out["metadata"]["source_paper"]["title"] = event["value"]
    if digest(out) != event["current_node_sha256"]:
        raise ValueError("approved title result differs")
    if without_selected_title(out) != without_selected_title(row):
        raise ValueError("non-title content changed")
    return out


def verify_title_result(row, event, registry, articles, witnesses):
    """Reproduce the new title/content witness without keeping an old record."""
    pmid = event["canonical_paper_key"].removeprefix("pmid:")
    authority = registry.export_payload()["records"][pmid]
    article = articles[pmid]
    complete_title_evidence(article, authority, witnesses[pmid])
    md = row["metadata"]
    title = "".join(article.find("./MedlineCitation/Article/ArticleTitle").itertext()).strip()
    quote = quote_key(md.get("raw_text"))
    abstract = quote_key(primary_abstract(article))
    if (row["id"] != event["claim_id"] or digest(row) != event["current_node_sha256"] or
        md["source_paper"]["title"] != title or title != event["value"] or
        len(quote) < 40 or len(re.findall(r"\w+", quote)) < 8 or quote not in abstract or
        hashlib.sha256(str(md.get("raw_text") or "").encode()).hexdigest() != event["raw_text_sha256"] or
        hashlib.sha256(quote.encode()).hexdigest() != event["normalized_complete_quote_sha256"] or
        hashlib.sha256(abstract.encode()).hexdigest() != event["normalized_primary_abstract_sha256"] or
        witnesses[pmid] != event["public_xml_witness"] or
        registry.resolve(md) != dict(paper_key="pmid:" + pmid, status="verified", reasons=[])):
        raise ValueError("candidate title/identity/content proof differs")
