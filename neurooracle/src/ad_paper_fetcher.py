"""Fetch Alzheimer's Disease related papers from PubMed for claim extraction.

Uses Biopython's Entrez to search PubMed and retrieve abstracts.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .schema import PaperRef

logger = logging.getLogger(__name__)

# try to import biopython, fall back to requests if not available
try:
    from Bio import Entrez
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    import requests


def fetch_ad_papers(
    max_results: int = 50,
    email: str = "neuroclaw@example.com",
    min_year: int = 2020,
    query_extra: str = "",
) -> list[tuple[str, PaperRef]]:
    """Fetch Alzheimer's Disease paper abstracts from PubMed.

    Args:
        max_results: Maximum number of papers to fetch.
        email: Email for NCBI Entrez (required by NCBI).
        min_year: Minimum publication year.
        query_extra: Additional query terms to append.

    Returns:
        List of (abstract_text, PaperRef) tuples.
    """
    base_query = (
        f'(Alzheimer[Title/Abstract] OR "Alzheimer\'s disease"[Title/Abstract]) '
        f'AND "brain imaging"[Title/Abstract] '
        f'AND {min_year}:{min_year}[pdat]'
    )
    if query_extra:
        base_query = f'({base_query}) AND ({query_extra})'

    logger.info(f"searching PubMed: {base_query}")

    if HAS_BIOPYTHON:
        return _fetch_with_biopython(base_query, max_results, email)
    else:
        return _fetch_with_requests(base_query, max_results)


def _fetch_with_biopython(
    query: str,
    max_results: int,
    email: str,
) -> list[tuple[str, PaperRef]]:
    """Fetch papers using Biopython Entrez."""
    Entrez.email = email

    # search
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()

    pmids = record.get("IdList", [])
    logger.info(f"found {len(pmids)} PMIDs")

    if not pmids:
        return []

    # fetch details
    papers = []
    batch_size = 20
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records.get("PubmedArticle", []):
            try:
                abstract, paper_ref = _parse_pubmed_article(article)
                if abstract:
                    papers.append((abstract, paper_ref))
            except Exception as e:
                logger.warning(f"failed to parse article: {e}")
                continue

        # rate limit: NCBI allows 3 requests/sec without API key
        if i + batch_size < len(pmids):
            time.sleep(0.5)

    logger.info(f"fetched {len(papers)} papers with abstracts")
    return papers


def _parse_pubmed_article(article: dict) -> tuple[str, PaperRef]:
    """Parse a PubMed XML article into (abstract, PaperRef)."""
    medline = article.get("MedlineCitation", {})
    article_data = medline.get("Article", {})

    # PMID
    pmid = str(medline.get("PMID", ""))

    # title
    title = str(article_data.get("ArticleTitle", ""))

    # abstract
    abstract_sections = article_data.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstract_sections, list):
        parts = []
        for section in abstract_sections:
            text = str(section).strip()
            if not text:
                continue
            label = ""
            attrs = getattr(section, "attributes", None)
            if isinstance(attrs, dict):
                label = str(attrs.get("Label") or attrs.get("NlmCategory") or "").strip()
            if label and not text.lower().startswith(label.lower()):
                text = f"{label}: {text}"
            parts.append(text)
        abstract = " ".join(parts)
    else:
        abstract = str(abstract_sections)

    if not abstract.strip():
        return "", PaperRef()

    # authors
    author_list = article_data.get("AuthorList", [])
    authors = []
    for a in author_list[:5]:  # first 5 authors
        last = a.get("LastName", "")
        first = a.get("ForeName", "")
        if last:
            authors.append(f"{last} {first}".strip())
    authors_str = ", ".join(authors)
    if len(author_list) > 5:
        authors_str += " et al."

    # year
    pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    year = None
    year_str = pub_date.get("Year", "")
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            pass

    # journal
    journal = str(article_data.get("Journal", {}).get("Title", ""))

    # DOI
    doi = ""
    for eid in article_data.get("ELocationID", []):
        if str(eid.attributes.get("EIdType", "")) == "doi":
            doi = str(eid)
            break

    paper_ref = PaperRef(
        pmid=pmid,
        doi=doi,
        title=title,
        authors=authors_str,
        year=year,
        journal=journal,
    )

    return abstract, paper_ref


def _fetch_with_requests(
    query: str,
    max_results: int,
) -> list[tuple[str, PaperRef]]:
    """Fallback: fetch papers using requests (no Biopython dependency)."""
    import requests

    # search
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }
    resp = requests.get(search_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pmids = data.get("esearchresult", {}).get("idlist", [])
    logger.info(f"found {len(pmids)} PMIDs")

    if not pmids:
        return []

    # fetch details
    papers = []
    batch_size = 20
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }
        resp = requests.get(fetch_url, params=params, timeout=60)
        resp.raise_for_status()

        # parse XML manually (minimal dependency)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)

        for article_elem in root.findall(".//PubmedArticle"):
            try:
                abstract, paper_ref = _parse_pubmed_xml_element(article_elem)
                if abstract:
                    papers.append((abstract, paper_ref))
            except Exception as e:
                logger.warning(f"failed to parse article: {e}")
                continue

        if i + batch_size < len(pmids):
            time.sleep(0.5)

    logger.info(f"fetched {len(papers)} papers with abstracts")
    return papers


def _parse_pubmed_xml_element(elem) -> tuple[str, PaperRef]:
    """Parse a PubmedArticle XML element."""
    pmid_el = elem.find(".//PMID")
    pmid = pmid_el.text if pmid_el is not None else ""

    title_el = elem.find(".//ArticleTitle")
    title = title_el.text if title_el is not None else ""

    abstract_parts = []
    for abstract_el in elem.findall(".//Abstract/AbstractText"):
        text = " ".join("".join(abstract_el.itertext()).split())
        if not text:
            continue
        label = (
            abstract_el.attrib.get("Label")
            or abstract_el.attrib.get("NlmCategory")
            or ""
        ).strip()
        if label and not text.lower().startswith(label.lower()):
            text = f"{label}: {text}"
        abstract_parts.append(text)
    abstract = " ".join(abstract_parts)
    if not abstract:
        return "", PaperRef()

    # authors
    authors = []
    for author in elem.findall(".//Author")[:5]:
        last = author.find("LastName")
        first = author.find("ForeName")
        if last is not None:
            name = last.text or ""
            if first is not None and first.text:
                name += " " + first.text
            authors.append(name)
    authors_str = ", ".join(authors)

    # year
    year_el = elem.find(".//PubDate/Year")
    year = int(year_el.text) if year_el is not None and year_el.text else None

    # journal
    journal_el = elem.find(".//Journal/Title")
    journal = journal_el.text if journal_el is not None else ""

    paper_ref = PaperRef(
        pmid=pmid,
        title=title,
        authors=authors_str,
        year=year,
        journal=journal,
    )

    return abstract, paper_ref
