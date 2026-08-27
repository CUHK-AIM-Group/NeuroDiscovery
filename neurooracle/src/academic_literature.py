"""Shared, fail-closed academic literature retrieval primitives.

The module deliberately separates provider search metadata from abstract
retrieval.  A caller must deduplicate the lite candidate before requesting the
core record.  This preserves the NeuroOracle v3 order:

    search -> formal/staging deduplication -> complete abstract retrieval

Search provenance is never Case Study membership evidence.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


LOGGER = logging.getLogger(__name__)
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_ARTICLE = "https://www.ebi.ac.uk/europepmc/webservices/rest/article"
USER_AGENT = "NeuroClaw-Academic-MCP/1.0"
YEAR_START = 1980
YEAR_END = 2026
MIN_ABSTRACT_CHARACTERS = 200
MAX_PAGE_SIZE = 1000

ASSIGNMENT_POLICY = {
    "schema_version": "case_study_membership.v2",
    "claim_labels_nonexclusive": True,
    "assign_all_17_case_studies": True,
    "paper_membership_derived_from_claim_union": True,
    "general_membership": "implicit_shared_corpus",
    "hindcasting": "validation_protocol_not_case_study_id",
    "search_provenance_is_membership": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def clean_title(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", clean_text(value).lower()))


def parse_year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    if not match:
        return None
    year = int(match.group(0))
    return year if YEAR_START <= year <= YEAR_END else None


def normalize_doi(value: object) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(" .;,)\t\r\n")


def yes(value: object) -> bool:
    return str(value or "").strip().lower() in {"y", "yes", "true", "1"}


def publication_types(result: Mapping[str, Any]) -> list[str]:
    value: object = result.get("pubTypeList")
    if isinstance(value, Mapping):
        value = value.get("pubType")
    if isinstance(value, list):
        return sorted({clean_text(item) for item in value if clean_text(item)})
    text = clean_text(value)
    return [text] if text else []


def full_query(expression: str) -> str:
    expression = str(expression or "").strip()
    if not expression:
        raise ValueError("Europe PMC query expression cannot be empty")
    return (
        f"({expression}) AND HAS_ABSTRACT:Y AND "
        f"FIRST_PDATE:[{YEAR_START}-01-01 TO {YEAR_END}-12-31]"
    )


def _request_json(url: str, *, attempts: int = 5) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("provider response is not a JSON object")
            return payload
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = min(30.0, 2.0**attempt)
            LOGGER.warning(
                "Europe PMC request failed; retrying in %.1fs: %s", delay, exc
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Europe PMC request failed after {attempts} attempts"
    ) from last_error


def fetch_search_page(
    expression: str,
    *,
    cursor: str = "*",
    page_size: int = 100,
    result_type: str = "lite",
    sort: str = "CITED desc",
) -> dict[str, Any]:
    """Fetch one search page; use ``lite`` before the deduplication gate."""

    result_type = str(result_type).strip().lower()
    if result_type not in {"idlist", "lite", "core"}:
        raise ValueError("result_type must be idlist, lite, or core")
    params = {
        "query": full_query(expression),
        "resultType": result_type,
        "format": "json",
        "sort": sort,
        "pageSize": max(1, min(int(page_size), MAX_PAGE_SIZE)),
        "cursorMark": str(cursor or "*"),
    }
    return _request_json(EUROPE_PMC_SEARCH + "?" + urllib.parse.urlencode(params))


def fetch_article_core(source: str, source_id: str) -> dict[str, Any]:
    """Fetch the complete core record for one already-deduplicated candidate."""

    source = re.sub(r"[^A-Za-z0-9]", "", str(source or "")).upper()
    source_id = str(source_id or "").strip()
    if not source or not source_id or len(source_id) > 256:
        raise ValueError("a valid Europe PMC source and identifier are required")
    path = "/".join(
        (
            EUROPE_PMC_ARTICLE.rstrip("/"),
            urllib.parse.quote(source, safe=""),
            urllib.parse.quote(source_id, safe=""),
        )
    )
    payload = _request_json(path + "?resultType=core&format=json")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Europe PMC article response lacks a core result")
    return result


def _paper_identity_fields(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        "pmid": clean_text(result.get("pmid")),
        "pmcid": clean_text(result.get("pmcid")),
        "doi": normalize_doi(result.get("doi")),
        "europepmc_source": clean_text(result.get("source")).upper(),
        "europepmc_id": clean_text(result.get("id")),
    }


def _paper_id(identity: Mapping[str, str]) -> str:
    if identity.get("pmid"):
        return f"pmid:{identity['pmid']}"
    if identity.get("doi"):
        return f"doi:{identity['doi']}"
    if identity.get("pmcid"):
        return f"pmcid:{identity['pmcid'].lower()}"
    if identity.get("europepmc_id"):
        return (
            f"europepmc:{identity.get('europepmc_source', '').lower()}:"
            f"{identity['europepmc_id'].lower()}"
        )
    return ""


def normalize_search_candidate(
    result: Mapping[str, Any],
    *,
    search_query_ids: Iterable[str] = (),
    search_target_case_study_ids: Iterable[str] = (),
    primary_search_case_study_id: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """Normalize lite metadata for deduplication without retrieving an abstract."""

    if yes(result.get("isRetracted")):
        return None, "retracted"
    title = clean_text(result.get("title"))
    if not title:
        return None, "missing_title"
    year = parse_year(result.get("pubYear") or result.get("firstPublicationDate"))
    if year is None:
        return None, "year_outside_window_or_missing"
    identity = _paper_identity_fields(result)
    paper_id = _paper_id(identity)
    if not paper_id:
        return None, "missing_stable_identifier"
    candidate = {
        "schema_version": "academic_literature_identity_candidate.v1",
        "paper_id": paper_id,
        **identity,
        "title": title,
        "year": year,
        "journal": clean_text(result.get("journalTitle")),
        "authors": clean_text(result.get("authorString")),
        "search_sources": ["europepmc"],
        "search_query_ids": sorted({str(x) for x in search_query_ids if str(x)}),
        "search_target_case_study_ids": sorted(
            {str(x) for x in search_target_case_study_ids if str(x)}
        ),
        "primary_search_case_study_id": str(primary_search_case_study_id or ""),
        "search_provenance_only": True,
        "assignment_policy": ASSIGNMENT_POLICY,
        "status": "identity_ready_for_dedup",
        "abstract_retrieved": False,
    }
    return candidate, "identity_ready"


def normalize_core_result(
    result: Mapping[str, Any],
    *,
    search_query_ids: Iterable[str] = (),
    search_target_case_study_ids: Iterable[str] = (),
    primary_search_case_study_id: str = "",
    expected_candidate: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Normalize and verify one complete provider abstract after deduplication."""

    candidate, status = normalize_search_candidate(
        result,
        search_query_ids=search_query_ids,
        search_target_case_study_ids=search_target_case_study_ids,
        primary_search_case_study_id=primary_search_case_study_id,
    )
    if candidate is None:
        return None, status
    if expected_candidate is not None:
        expected_source = clean_text(expected_candidate.get("europepmc_source")).upper()
        expected_id = clean_text(expected_candidate.get("europepmc_id"))
        if (
            expected_source != candidate["europepmc_source"]
            or expected_id != candidate["europepmc_id"]
        ):
            return None, "provider_identity_mismatch"
        if clean_title(expected_candidate.get("title")) != clean_title(candidate["title"]):
            return None, "provider_title_mismatch"

    abstract = clean_text(result.get("abstractText"))
    if not abstract:
        return None, "missing_abstract"
    if len(abstract) < MIN_ABSTRACT_CHARACTERS:
        return None, "abstract_too_short"
    if abstract.casefold() == candidate["title"].casefold():
        return None, "abstract_duplicates_title"
    try:
        cited_by_count = int(result.get("citedByCount") or 0)
    except (TypeError, ValueError):
        cited_by_count = 0

    record = {
        **candidate,
        "schema_version": "case_study_literature_candidate.v3",
        "abstract": abstract,
        "publication_types": publication_types(result),
        "language": clean_text(result.get("language")),
        "cited_by_count": cited_by_count,
        "url": (
            f"https://europepmc.org/article/{candidate['europepmc_source']}/"
            f"{candidate['europepmc_id']}"
        ),
        "abstract_source": "europepmc",
        "identifier_match_verified": True,
        "source_title_agreement": {
            "accepted": True,
            "basis": (
                "lite identity was deduplicated before the matching Europe PMC "
                "core record and complete abstract were retrieved"
            ),
        },
        "status": "abstract_verified_ready_for_claim_extraction",
        "kg_injection": False,
        "abstract_retrieved": True,
        "abstract_characters": len(abstract),
        "abstract_words": len(abstract.split()),
        "collected_at": utc_now(),
    }
    return record, "ready"


__all__ = [
    "ASSIGNMENT_POLICY",
    "EUROPE_PMC_ARTICLE",
    "EUROPE_PMC_SEARCH",
    "MAX_PAGE_SIZE",
    "MIN_ABSTRACT_CHARACTERS",
    "USER_AGENT",
    "YEAR_END",
    "YEAR_START",
    "clean_text",
    "clean_title",
    "fetch_article_core",
    "fetch_search_page",
    "full_query",
    "normalize_core_result",
    "normalize_doi",
    "normalize_search_candidate",
    "parse_year",
    "publication_types",
    "utc_now",
]
