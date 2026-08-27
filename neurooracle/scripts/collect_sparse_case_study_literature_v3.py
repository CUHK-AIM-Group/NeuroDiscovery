#!/usr/bin/env python3
"""Collect an abstract-ready KG-v3 literature top-up for sparse Case Studies.

This is a collection-only pipeline.  It searches Europe PMC, rejects papers
already present in the formal KG or any active staging corpus, downloads the
complete provider abstract, and writes 100-paper extraction batches.  Search
provenance is deliberately *not* treated as Case Study membership; downstream
claim extraction must assign every applicable Case Study independently.

The formal KG is fail-closed: its three canonical files are hashed before the
run and verified unchanged before a completed collection is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurooracle.src.paper_identity import (  # noqa: E402
    CandidatePaperDeduplicator,
    GlobalPaperIdentityIndex,
    paper_identity_aliases,
)
from neurooracle.src import academic_literature as academic  # noqa: E402


LOGGER = logging.getLogger("collect_sparse_case_study_literature_v3")
DATA_ROOT = REPO_ROOT / "neurooracle" / "data"
FORMAL_ROOT = DATA_ROOT / "full_v2"
FORMAL_CLAIMS = FORMAL_ROOT / "extracted_claims.jsonl"
FORMAL_FILES = (
    FORMAL_ROOT / "knowledge_graph.json",
    FORMAL_CLAIMS,
    FORMAL_ROOT / "CURRENT_STATE.json",
)
STAGING_ROOTS = (
    DATA_ROOT / "case_study_staging",
    DATA_ROOT / "phase2_staging",
)
IDENTITY_INDEX = DATA_ROOT / "build_artifacts" / "paper_identity_dedup_v3.sqlite3"
DEFAULT_OUTPUT = (
    DATA_ROOT / "case_study_staging" / "kg_v3_sparse_case_studies_100k_20260811"
)
EUROPE_PMC_SEARCH = academic.EUROPE_PMC_SEARCH
USER_AGENT = academic.USER_AGENT
YEAR_START = academic.YEAR_START
YEAR_END = academic.YEAR_END
MIN_ABSTRACT_CHARACTERS = academic.MIN_ABSTRACT_CHARACTERS
DEFAULT_PAGE_SIZE = 1000

ASSIGNMENT_POLICY = academic.ASSIGNMENT_POLICY


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    expression: str


@dataclass(frozen=True)
class Target:
    case_study_id: str
    display_number: int
    baseline_papers: int
    quota: int
    queries: tuple[SearchQuery, ...]


TARGETS: tuple[Target, ...] = (
    Target(
        "cognitive_decoding",
        14,
        1_675,
        12_000,
        (
            SearchQuery(
                "cognitive_decoding_explicit",
                '(TITLE_ABS:"neural decoding" OR TITLE_ABS:"brain decoding" OR '
                'TITLE_ABS:"cognitive decoding" OR TITLE_ABS:"multivariate pattern analysis" '
                'OR TITLE_ABS:MVPA OR TITLE_ABS:"representational similarity analysis" OR '
                'TITLE_ABS:"brain-computer interface" OR TITLE_ABS:"brain computer interface") '
                'AND (TITLE_ABS:brain OR TITLE_ABS:neural OR TITLE_ABS:neuroimaging OR '
                'TITLE_ABS:fMRI OR TITLE_ABS:EEG OR TITLE_ABS:MEG)',
            ),
            SearchQuery(
                "cognitive_decoding_high_recall",
                '(TITLE_ABS:decod* OR TITLE_ABS:classif* OR TITLE_ABS:"pattern analysis" '
                'OR TITLE_ABS:"machine learning") AND (TITLE_ABS:fMRI OR TITLE_ABS:EEG '
                'OR TITLE_ABS:MEG OR TITLE_ABS:neuroimaging) AND (TITLE_ABS:cognit* OR '
                'TITLE_ABS:memory OR TITLE_ABS:language OR TITLE_ABS:perception OR '
                'TITLE_ABS:emotion OR TITLE_ABS:attention OR TITLE_ABS:"mental state")',
            ),
        ),
    ),
    Target(
        "brain_age",
        16,
        1_774,
        8_000,
        (
            SearchQuery(
                "brain_age_explicit",
                '(TITLE_ABS:"brain age" OR TITLE_ABS:"brain-age" OR TITLE_ABS:BrainAGE '
                'OR TITLE_ABS:"brain age gap" OR TITLE_ABS:"predicted brain age" OR '
                '((TITLE_ABS:"age prediction" OR TITLE_ABS:"age estimation" OR '
                'TITLE_ABS:"normative model") AND (TITLE_ABS:brain OR '
                'TITLE_ABS:neuroimaging OR TITLE_ABS:"structural MRI")))',
            ),
            SearchQuery(
                "brain_age_high_recall",
                '(TITLE_ABS:aging OR TITLE_ABS:ageing OR TITLE_ABS:lifespan OR '
                'TITLE_ABS:"age-related") AND (TITLE_ABS:brain OR TITLE_ABS:neuroimaging '
                'OR TITLE_ABS:"brain MRI" OR TITLE_ABS:"brain structure") AND '
                '(TITLE_ABS:predict* OR TITLE_ABS:estimat* OR TITLE_ABS:model* OR '
                'TITLE_ABS:normative OR TITLE_ABS:"machine learning" OR '
                'TITLE_ABS:"deep learning")',
            ),
        ),
    ),
    Target(
        "drug_response_prediction",
        8,
        2_738,
        10_000,
        (
            SearchQuery(
                "drug_response_neurobiomarker",
                '(TITLE_ABS:predict* OR TITLE_ABS:biomarker* OR TITLE_ABS:stratification '
                'OR TITLE_ABS:baseline) AND (TITLE_ABS:"treatment response" OR '
                'TITLE_ABS:"drug response" OR TITLE_ABS:remission OR '
                'TITLE_ABS:resistance OR TITLE_ABS:responder*) AND '
                '(TITLE_ABS:neuroimaging OR TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:EEG OR TITLE_ABS:MEG OR TITLE_ABS:"brain connectivity" '
                'OR TITLE_ABS:"brain structure" OR TITLE_ABS:"brain activation") AND '
                '(TITLE_ABS:drug OR TITLE_ABS:pharmacotherap* OR TITLE_ABS:medication '
                'OR TITLE_ABS:antidepressant* OR TITLE_ABS:antipsychotic* OR '
                'TITLE_ABS:levodopa OR TITLE_ABS:pharmacological)',
            ),
            SearchQuery(
                "drug_response_high_recall",
                '(TITLE_ABS:"treatment response" OR TITLE_ABS:"drug response" OR '
                'TITLE_ABS:remission OR TITLE_ABS:resistance OR TITLE_ABS:responder*) '
                'AND (TITLE_ABS:antidepressant* OR TITLE_ABS:antipsychotic* OR '
                'TITLE_ABS:levodopa OR TITLE_ABS:antiepileptic* OR TITLE_ABS:medication '
                'OR TITLE_ABS:pharmacotherap* OR TITLE_ABS:pharmacological) AND '
                '(TITLE_ABS:brain OR TITLE_ABS:neural OR TITLE_ABS:neuroimaging OR '
                'TITLE_ABS:fMRI OR TITLE_ABS:EEG OR TITLE_ABS:psychiatr* OR '
                'TITLE_ABS:neurolog*)',
            ),
        ),
    ),
    Target(
        "personalised_treatment",
        9,
        5_414,
        9_000,
        (
            SearchQuery(
                "personalised_treatment_explicit",
                '(TITLE_ABS:"precision medicine" OR TITLE_ABS:"personalized treatment" '
                'OR TITLE_ABS:"personalised treatment" OR TITLE_ABS:"treatment selection" '
                'OR TITLE_ABS:"treatment stratification" OR '
                'TITLE_ABS:"individualized treatment" OR '
                'TITLE_ABS:"individualised treatment" OR TITLE_ABS:"precision psychiatry" '
                'OR TITLE_ABS:"precision neurology") AND (TITLE_ABS:brain OR '
                'TITLE_ABS:neural OR TITLE_ABS:neuroimaging OR TITLE_ABS:neurolog* '
                'OR TITLE_ABS:psychiatr*)',
            ),
            SearchQuery(
                "personalised_treatment_high_recall",
                '(TITLE_ABS:"treatment selection" OR TITLE_ABS:"treatment stratification" '
                'OR TITLE_ABS:"individual treatment" OR TITLE_ABS:"predictive biomarker" '
                'OR TITLE_ABS:"precision psychiatry" OR TITLE_ABS:"precision neurology" '
                'OR TITLE_ABS:"personalized medicine" OR TITLE_ABS:"personalised medicine") '
                'AND (TITLE_ABS:brain OR TITLE_ABS:neural OR TITLE_ABS:neuroimaging '
                'OR TITLE_ABS:neurolog* OR TITLE_ABS:psychiatr*)',
            ),
        ),
    ),
    Target(
        "neuromodulation_target",
        12,
        5_982,
        10_000,
        (
            SearchQuery(
                "neuromodulation_target_circuit",
                '(TITLE_ABS:neuromodulation OR TITLE_ABS:neurostimulation OR '
                'TITLE_ABS:"deep brain stimulation" OR '
                'TITLE_ABS:"transcranial magnetic stimulation" OR TITLE_ABS:rTMS OR '
                'TITLE_ABS:"transcranial direct current stimulation" OR TITLE_ABS:tDCS '
                'OR TITLE_ABS:"vagus nerve stimulation") AND (TITLE_ABS:target* OR '
                'TITLE_ABS:circuit* OR TITLE_ABS:connectiv* OR TITLE_ABS:"brain region" '
                'OR TITLE_ABS:network*)',
            ),
            SearchQuery(
                "neuromodulation_target_high_recall",
                '(TITLE_ABS:neuromodulation OR TITLE_ABS:neurostimulation OR '
                'TITLE_ABS:"deep brain stimulation" OR '
                'TITLE_ABS:"transcranial magnetic stimulation" OR TITLE_ABS:rTMS OR '
                'TITLE_ABS:"transcranial direct current stimulation" OR TITLE_ABS:tDCS '
                'OR TITLE_ABS:"vagus nerve stimulation") AND (TITLE_ABS:target* OR '
                'TITLE_ABS:circuit* OR TITLE_ABS:connectiv* OR TITLE_ABS:"brain region" '
                'OR TITLE_ABS:network* OR TITLE_ABS:site*)',
            ),
        ),
    ),
    Target(
        "progression_prediction",
        5,
        7_899,
        10_000,
        (
            SearchQuery(
                "progression_prediction_strict",
                '(TITLE_ABS:progression OR TITLE_ABS:conversion OR TITLE_ABS:trajectory '
                'OR TITLE_ABS:longitudinal OR TITLE_ABS:"disease course" OR '
                'TITLE_ABS:"clinical decline") AND (TITLE_ABS:predict* OR '
                'TITLE_ABS:prognos* OR TITLE_ABS:biomarker* OR TITLE_ABS:baseline) AND '
                '(TITLE_ABS:neuroimaging OR TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:EEG OR TITLE_ABS:MEG OR TITLE_ABS:connectome OR '
                'TITLE_ABS:"brain structure" OR TITLE_ABS:"brain connectivity")',
            ),
            SearchQuery(
                "progression_prediction_high_recall",
                '(TITLE_ABS:progression OR TITLE_ABS:conversion OR TITLE_ABS:trajectory '
                'OR TITLE_ABS:longitudinal OR TITLE_ABS:"disease course" OR '
                'TITLE_ABS:"clinical decline") AND (TITLE_ABS:neuroimaging OR '
                'TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI OR TITLE_ABS:PET OR '
                'TITLE_ABS:EEG OR TITLE_ABS:MEG OR TITLE_ABS:connectome OR '
                'TITLE_ABS:"brain structure" OR TITLE_ABS:"brain connectivity") AND '
                '(TITLE_ABS:Alzheimer* OR TITLE_ABS:Parkinson* OR TITLE_ABS:dementia '
                'OR TITLE_ABS:"multiple sclerosis" OR TITLE_ABS:psychiatr* OR '
                'TITLE_ABS:neurolog* OR TITLE_ABS:stroke OR TITLE_ABS:"cognitive decline")',
            ),
        ),
    ),
    Target(
        "adverse_event_prediction",
        11,
        9_317,
        9_000,
        (
            SearchQuery(
                "adverse_event_prediction_strict",
                '(TITLE_ABS:"adverse event" OR TITLE_ABS:"adverse effect" OR '
                'TITLE_ABS:"side effect" OR TITLE_ABS:neurotoxicity OR '
                'TITLE_ABS:"treatment-emergent" OR TITLE_ABS:dyskinesia OR '
                'TITLE_ABS:"cognitive adverse") AND (TITLE_ABS:predict* OR TITLE_ABS:risk '
                'OR TITLE_ABS:biomarker* OR TITLE_ABS:baseline OR TITLE_ABS:associat*) '
                'AND (TITLE_ABS:brain OR TITLE_ABS:neural OR TITLE_ABS:neurolog* OR '
                'TITLE_ABS:psychiatr* OR TITLE_ABS:neuroimaging OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:EEG)',
            ),
            SearchQuery(
                "adverse_event_high_recall",
                '(TITLE_ABS:"adverse event" OR TITLE_ABS:"adverse effect" OR '
                'TITLE_ABS:"side effect" OR TITLE_ABS:neurotoxicity OR '
                'TITLE_ABS:"treatment-emergent" OR TITLE_ABS:dyskinesia OR '
                'TITLE_ABS:"cognitive adverse") AND (TITLE_ABS:drug OR '
                'TITLE_ABS:medication OR TITLE_ABS:treatment OR TITLE_ABS:stimulation '
                'OR TITLE_ABS:therapy) AND (TITLE_ABS:brain OR TITLE_ABS:neural OR '
                'TITLE_ABS:neurolog* OR TITLE_ABS:psychiatr* OR '
                'TITLE_ABS:neuroimaging OR TITLE_ABS:EEG)',
            ),
        ),
    ),
    Target(
        "differential_diagnosis",
        7,
        11_330,
        10_000,
        (
            SearchQuery(
                "differential_diagnosis_strict",
                '(TITLE_ABS:"differential diagnosis" OR '
                'TITLE_ABS:"diagnostic discrimination" OR '
                'TITLE_ABS:"diagnostic classification" OR '
                'TITLE_ABS:"disease classification" OR '
                'TITLE_ABS:"distinguish patients" OR TITLE_ABS:"discriminate between") '
                'AND (TITLE_ABS:neuroimaging OR TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:PET OR TITLE_ABS:EEG OR TITLE_ABS:MEG OR '
                'TITLE_ABS:connectome OR TITLE_ABS:"brain connectivity")',
            ),
            SearchQuery(
                "differential_diagnosis_high_recall",
                '(TITLE_ABS:"differential diagnosis" OR TITLE_ABS:discriminat* OR '
                'TITLE_ABS:classif* OR TITLE_ABS:"diagnostic accuracy") AND '
                '(TITLE_ABS:neuroimaging OR TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:PET OR TITLE_ABS:EEG OR TITLE_ABS:MEG OR '
                'TITLE_ABS:connectome) AND (TITLE_ABS:patient* OR TITLE_ABS:disease '
                'OR TITLE_ABS:disorder OR TITLE_ABS:diagnos*)',
            ),
        ),
    ),
    Target(
        "disease_subtyping",
        4,
        21_016,
        8_000,
        (
            SearchQuery(
                "disease_subtyping_neural",
                '(TITLE_ABS:subtyp* OR TITLE_ABS:biotype* OR TITLE_ABS:cluster* OR '
                'TITLE_ABS:heterogeneity OR TITLE_ABS:stratification) AND '
                '(TITLE_ABS:neuroimaging OR TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:PET OR TITLE_ABS:EEG OR TITLE_ABS:MEG OR '
                'TITLE_ABS:connectome OR TITLE_ABS:"brain connectivity") AND '
                '(TITLE_ABS:patient* OR TITLE_ABS:disease OR TITLE_ABS:disorder OR '
                'TITLE_ABS:psychiatr* OR TITLE_ABS:neurolog*)',
            ),
            SearchQuery(
                "disease_subtyping_high_recall",
                '(TITLE_ABS:subtyp* OR TITLE_ABS:biotype* OR '
                'TITLE_ABS:"data-driven cluster" OR TITLE_ABS:"patient stratification" '
                'OR TITLE_ABS:"disease heterogeneity") AND (TITLE_ABS:brain OR '
                'TITLE_ABS:neural OR TITLE_ABS:neuroimaging OR TITLE_ABS:EEG OR '
                'TITLE_ABS:connectiv*)',
            ),
        ),
    ),
    Target(
        "prognosis",
        17,
        15_183,
        7_000,
        (
            SearchQuery(
                "prognosis_neural_biomarker",
                '(TITLE_ABS:prognos* OR TITLE_ABS:"clinical outcome" OR '
                'TITLE_ABS:"functional outcome" OR TITLE_ABS:survival OR '
                'TITLE_ABS:recovery) AND (TITLE_ABS:predict* OR TITLE_ABS:biomarker* '
                'OR TITLE_ABS:baseline OR TITLE_ABS:associat*) AND '
                '(TITLE_ABS:neuroimaging OR TITLE_ABS:"brain MRI" OR TITLE_ABS:fMRI '
                'OR TITLE_ABS:PET OR TITLE_ABS:EEG OR TITLE_ABS:MEG OR '
                'TITLE_ABS:connectome OR TITLE_ABS:"brain connectivity")',
            ),
            SearchQuery(
                "prognosis_high_recall",
                '(TITLE_ABS:outcome* OR TITLE_ABS:recovery OR TITLE_ABS:survival OR '
                'TITLE_ABS:prognos*) AND (TITLE_ABS:brain OR TITLE_ABS:neurolog* OR '
                'TITLE_ABS:stroke OR TITLE_ABS:traumatic OR TITLE_ABS:psychiatr*) AND '
                '(TITLE_ABS:MRI OR TITLE_ABS:fMRI OR TITLE_ABS:PET OR TITLE_ABS:EEG '
                'OR TITLE_ABS:neuroimaging OR TITLE_ABS:connectiv*)',
            ),
        ),
    ),
    Target(
        "connectome_behavior",
        15,
        16_436,
        7_000,
        (
            SearchQuery(
                "connectome_behavior_association",
                '(TITLE_ABS:connectome OR TITLE_ABS:"functional connectivity" OR '
                'TITLE_ABS:"structural connectivity" OR TITLE_ABS:"brain network") '
                'AND (TITLE_ABS:behavior* OR TITLE_ABS:behaviour* OR TITLE_ABS:cognit* '
                'OR TITLE_ABS:symptom* OR TITLE_ABS:function*) AND '
                '(TITLE_ABS:associat* OR TITLE_ABS:correlat* OR TITLE_ABS:predict* '
                'OR TITLE_ABS:relationship*)',
            ),
            SearchQuery(
                "connectome_behavior_high_recall",
                '(TITLE_ABS:"functional connectivity" OR '
                'TITLE_ABS:"structural connectivity" OR TITLE_ABS:connectome OR '
                'TITLE_ABS:"network connectivity") AND (TITLE_ABS:individual* OR '
                'TITLE_ABS:behavior* OR TITLE_ABS:behaviour* OR TITLE_ABS:cognit* OR '
                'TITLE_ABS:clinical OR TITLE_ABS:symptom*)',
            ),
        ),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def parse_year(value: object) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    if not match:
        return None
    year = int(match.group(0))
    return year if YEAR_START <= year <= YEAR_END else None


def normalize_doi(value: object) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(" .;,)")


def yes(value: object) -> bool:
    return str(value or "").strip().lower() in {"y", "yes", "true", "1"}


def publication_types(result: dict[str, Any]) -> list[str]:
    value = result.get("pubTypeList")
    if isinstance(value, dict):
        value = value.get("pubType")
    if isinstance(value, list):
        return sorted({clean_text(item) for item in value if clean_text(item)})
    text = clean_text(value)
    return [text] if text else []


def normalize_result(
    result: dict[str, Any],
    *,
    target: Target,
    query: SearchQuery,
    expected_candidate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    return academic.normalize_core_result(
        result,
        search_query_ids=(query.query_id,),
        search_target_case_study_ids=(target.case_study_id,),
        primary_search_case_study_id=target.case_study_id,
        expected_candidate=expected_candidate,
    )


def normalize_candidate_result(
    result: dict[str, Any],
    *,
    target: Target,
    query: SearchQuery,
) -> tuple[dict[str, Any] | None, str]:
    return academic.normalize_search_candidate(
        result,
        search_query_ids=(query.query_id,),
        search_target_case_study_ids=(target.case_study_id,),
        primary_search_case_study_id=target.case_study_id,
    )


def full_query(expression: str) -> str:
    return (
        f"({expression}) AND HAS_ABSTRACT:Y AND "
        f"FIRST_PDATE:[{YEAR_START}-01-01 TO {YEAR_END}-12-31]"
    )


def request_json(params: dict[str, object], *, attempts: int = 5) -> dict[str, Any]:
    url = EUROPE_PMC_SEARCH + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = min(30.0, 2.0**attempt)
            LOGGER.warning("Europe PMC request failed; retrying in %.1fs: %s", delay, exc)
            time.sleep(delay)
    raise RuntimeError(f"Europe PMC request failed after {attempts} attempts") from last_error


def fetch_page(query: SearchQuery, cursor: str, page_size: int) -> dict[str, Any]:
    return academic.fetch_search_page(
        query.expression,
        cursor=cursor,
        page_size=page_size,
        result_type="lite",
    )


def hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def formal_snapshot() -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    for path in FORMAL_FILES:
        stat = path.stat()
        snapshot[str(path.resolve())] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hash_file(path),
        }
    return snapshot


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_number}")
            yield row


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "sparse_case_study_collection_state.v1",
            "created_at": utc_now(),
            "queries": {},
            "pages_completed": 0,
            "raw_results_scanned": 0,
            "invalid_results": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError(f"invalid state payload: {path}")
    return state


def query_state(state: dict[str, Any], target: Target, query: SearchQuery) -> dict[str, Any]:
    key = f"{target.case_study_id}/{query.query_id}"
    queries = state.setdefault("queries", {})
    return queries.setdefault(
        key,
        {
            "case_study_id": target.case_study_id,
            "query_id": query.query_id,
            "cursor": "*",
            "pages": 0,
            "raw_results": 0,
            "reported_hit_count": None,
            "exhausted": False,
            "errors": 0,
        },
    )


def load_existing_collection(
    papers_path: Path,
    links_path: Path,
    identity_index: GlobalPaperIdentityIndex,
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
    dict[str, set[str]],
    set[tuple[str, str, str]],
    Counter[str],
    int,
]:
    alias_to_paper: dict[str, str] = {}
    paper_cases: dict[str, set[str]] = defaultdict(set)
    paper_queries: dict[str, set[str]] = defaultdict(set)
    links: set[tuple[str, str, str]] = set()
    primary_counts: Counter[str] = Counter()
    total = 0

    for row in iter_jsonl(papers_path):
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise RuntimeError(f"collected row lacks paper_id: {papers_path}")
        total += 1
        primary = str(row.get("primary_search_case_study_id") or "")
        if primary:
            primary_counts[primary] += 1
            paper_cases[paper_id].add(primary)
        for case_id in row.get("search_target_case_study_ids") or []:
            paper_cases[paper_id].add(str(case_id))
        for query_id in row.get("search_query_ids") or []:
            paper_queries[paper_id].add(str(query_id))
        for alias in paper_identity_aliases(row):
            prior = alias_to_paper.setdefault(alias, paper_id)
            if prior != paper_id:
                raise RuntimeError(f"within-campaign identity collision: {alias}")
        identity_index.add_overlay(
            row,
            origin="current_collection",
            source_path=str(papers_path.resolve()),
        )

    for row in iter_jsonl(links_path):
        paper_id = str(row.get("paper_id") or "")
        case_id = str(row.get("case_study_id") or "")
        query_id = str(row.get("query_id") or "")
        if not (paper_id and case_id and query_id):
            continue
        links.add((paper_id, case_id, query_id))
        paper_cases[paper_id].add(case_id)
        paper_queries[paper_id].add(query_id)
    return alias_to_paper, paper_cases, paper_queries, links, primary_counts, total


def record_link(
    *,
    links_path: Path,
    links: set[tuple[str, str, str]],
    paper_cases: dict[str, set[str]],
    paper_queries: dict[str, set[str]],
    paper_id: str,
    target: Target,
    query: SearchQuery,
) -> None:
    key = (paper_id, target.case_study_id, query.query_id)
    paper_cases[paper_id].add(target.case_study_id)
    paper_queries[paper_id].add(query.query_id)
    if key in links:
        return
    links.add(key)
    append_jsonl(
        links_path,
        {
            "schema_version": "search_provenance_link.v1",
            "paper_id": paper_id,
            "case_study_id": target.case_study_id,
            "query_id": query.query_id,
            "linked_at": utc_now(),
            "membership_asserted": False,
        },
    )


def local_match(record: dict[str, Any], alias_to_paper: dict[str, str]) -> str:
    return next(
        (alias_to_paper[alias] for alias in paper_identity_aliases(record) if alias in alias_to_paper),
        "",
    )


def collect_query(
    *,
    target: Target,
    query: SearchQuery,
    state: dict[str, Any],
    state_path: Path,
    papers_path: Path,
    links_path: Path,
    alias_to_paper: dict[str, str],
    paper_cases: dict[str, set[str]],
    paper_queries: dict[str, set[str]],
    links: set[tuple[str, str, str]],
    primary_counts: Counter[str],
    total_ref: list[int],
    target_total: int,
    case_limit: int | None,
    page_size: int,
    deduplicator: CandidatePaperDeduplicator,
) -> None:
    qstate = query_state(state, target, query)
    if qstate.get("exhausted"):
        return

    while total_ref[0] < target_total:
        if case_limit is not None and primary_counts[target.case_study_id] >= case_limit:
            return
        cursor = str(qstate.get("cursor") or "*")
        try:
            payload = fetch_page(query, cursor, page_size)
        except Exception as exc:
            qstate["errors"] = int(qstate.get("errors") or 0) + 1
            append_jsonl(
                state_path.parent / "search_errors.jsonl",
                {
                    "at": utc_now(),
                    "case_study_id": target.case_study_id,
                    "query_id": query.query_id,
                    "cursor": cursor,
                    "error": repr(exc),
                },
            )
            write_json(state_path, state)
            LOGGER.error("query failed: %s/%s: %s", target.case_study_id, query.query_id, exc)
            return

        qstate["reported_hit_count"] = int(payload.get("hitCount") or 0)
        results = ((payload.get("resultList") or {}).get("result")) or []
        if not isinstance(results, list):
            results = []
        next_cursor = str(payload.get("nextCursorMark") or "")
        qstate["pages"] = int(qstate.get("pages") or 0) + 1
        qstate["raw_results"] = int(qstate.get("raw_results") or 0) + len(results)
        state["pages_completed"] = int(state.get("pages_completed") or 0) + 1
        state["raw_results_scanned"] = int(state.get("raw_results_scanned") or 0) + len(results)
        invalid = Counter(state.get("invalid_results") or {})

        for result in results:
            if total_ref[0] >= target_total:
                break
            if case_limit is not None and primary_counts[target.case_study_id] >= case_limit:
                break
            if not isinstance(result, dict):
                invalid["non_object_result"] += 1
                continue
            candidate, status = normalize_candidate_result(
                result, target=target, query=query
            )
            if candidate is None:
                invalid[status] += 1
                continue

            within = local_match(candidate, alias_to_paper)
            if within:
                record_link(
                    links_path=links_path,
                    links=links,
                    paper_cases=paper_cases,
                    paper_queries=paper_queries,
                    paper_id=within,
                    target=target,
                    query=query,
                )
                deduplicator.is_seen(
                    candidate,
                    source="europepmc",
                    preset=target.case_study_id,
                    query_index=None,
                    stage="within_current_collection_merge",
                )
                continue

            if deduplicator.is_seen(
                candidate,
                source="europepmc",
                preset=target.case_study_id,
                query_index=None,
                stage="post_search_pre_abstract",
            ):
                continue

            try:
                core_result = academic.fetch_article_core(
                    str(candidate["europepmc_source"]),
                    str(candidate["europepmc_id"]),
                )
            except Exception as exc:
                invalid["abstract_retrieval_failed"] += 1
                append_jsonl(
                    state_path.parent / "abstract_errors.jsonl",
                    {
                        "at": utc_now(),
                        "case_study_id": target.case_study_id,
                        "query_id": query.query_id,
                        "paper_id": candidate.get("paper_id"),
                        "europepmc_source": candidate.get("europepmc_source"),
                        "europepmc_id": candidate.get("europepmc_id"),
                        "error": repr(exc),
                    },
                )
                continue

            record, status = normalize_result(
                core_result,
                target=target,
                query=query,
                expected_candidate=candidate,
            )
            if record is None:
                invalid[status] += 1
                continue

            if not deduplicator.accept(
                record,
                source="europepmc",
                preset=target.case_study_id,
                query_index=None,
                collection_path=str(papers_path.resolve()),
            ):
                continue

            append_jsonl(papers_path, record)
            paper_id = str(record["paper_id"])
            for alias in paper_identity_aliases(record):
                alias_to_paper[alias] = paper_id
            primary_counts[target.case_study_id] += 1
            total_ref[0] += 1
            record_link(
                links_path=links_path,
                links=links,
                paper_cases=paper_cases,
                paper_queries=paper_queries,
                paper_id=paper_id,
                target=target,
                query=query,
            )

        state["invalid_results"] = dict(sorted(invalid.items()))
        qstate["cursor"] = next_cursor or cursor
        if not results or not next_cursor or next_cursor == cursor:
            qstate["exhausted"] = True
        deduplicator.flush()
        write_json(state_path, state)
        print(
            f"[{target.case_study_id}/{query.query_id}] "
            f"case={primary_counts[target.case_study_id]:,}/{case_limit or '-'} "
            f"total={total_ref[0]:,}/{target_total:,} "
            f"page={qstate['pages']:,} raw={state['raw_results_scanned']:,} "
            f"dedup_excluded={deduplicator.excluded:,}",
            flush=True,
        )
        if qstate.get("exhausted"):
            return
        time.sleep(0.15)


def collect(
    *,
    output_dir: Path,
    target_total: int,
    page_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_path = output_dir / "abstracts_ready_for_extraction.jsonl"
    links_path = output_dir / "dedup_audit_search_provenance_links.jsonl"
    state_path = output_dir / "collection_state.json"
    baseline_path = output_dir / "formal_kg_baseline.json"
    dedup_audit_path = output_dir / "dedup_audit_excluded_papers.jsonl"
    started = time.time()

    current_formal = formal_snapshot()
    if baseline_path.exists():
        formal_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if formal_baseline.get("files") != current_formal:
            raise RuntimeError("formal KG changed since this collection started; refusing resume")
    else:
        formal_baseline = {"captured_at": utc_now(), "files": current_formal}
        write_json(baseline_path, formal_baseline)

    state = load_state(state_path)
    identity_index = GlobalPaperIdentityIndex(IDENTITY_INDEX)
    try:
        sync_stats = identity_index.sync(
            formal_claim_store=FORMAL_CLAIMS,
            staging_roots=STAGING_ROOTS,
        )
        (
            alias_to_paper,
            paper_cases,
            paper_queries,
            links,
            primary_counts,
            existing_total,
        ) = load_existing_collection(papers_path, links_path, identity_index)
        total_ref = [existing_total]
        deduplicator = CandidatePaperDeduplicator(identity_index, dedup_audit_path)

        # Pass 1 honours the deficit-oriented per-Case-Study allocation.
        for target in TARGETS:
            if total_ref[0] >= target_total:
                break
            case_limit = min(target.quota, target_total)
            for query in target.queries:
                collect_query(
                    target=target,
                    query=query,
                    state=state,
                    state_path=state_path,
                    papers_path=papers_path,
                    links_path=links_path,
                    alias_to_paper=alias_to_paper,
                    paper_cases=paper_cases,
                    paper_queries=paper_queries,
                    links=links,
                    primary_counts=primary_counts,
                    total_ref=total_ref,
                    target_total=target_total,
                    case_limit=case_limit,
                    page_size=page_size,
                    deduplicator=deduplicator,
                )
                if primary_counts[target.case_study_id] >= case_limit:
                    break

        # Pass 2 fills any shortfall from unexhausted relevant searches.  It does
        # not change the provenance-only status of the search labels.
        if total_ref[0] < target_total:
            for target in TARGETS:
                if total_ref[0] >= target_total:
                    break
                for query in target.queries:
                    collect_query(
                        target=target,
                        query=query,
                        state=state,
                        state_path=state_path,
                        papers_path=papers_path,
                        links_path=links_path,
                        alias_to_paper=alias_to_paper,
                        paper_cases=paper_cases,
                        paper_queries=paper_queries,
                        links=links,
                        primary_counts=primary_counts,
                        total_ref=total_ref,
                        target_total=target_total,
                        case_limit=None,
                        page_size=page_size,
                        deduplicator=deduplicator,
                    )

        deduplicator.flush()
        state["updated_at"] = utc_now()
        state["accepted_unique_papers"] = total_ref[0]
        state["primary_counts"] = dict(sorted(primary_counts.items()))
        state["deduplication"] = deduplicator.summary()
        state["identity_sync"] = sync_stats
        write_json(state_path, state)
        return {
            "papers": total_ref[0],
            "primary_counts": primary_counts,
            "paper_cases": paper_cases,
            "paper_queries": paper_queries,
            "deduplication": deduplicator.summary(),
            "identity_sync": sync_stats,
            "elapsed_seconds": round(time.time() - started, 3),
            "formal_baseline": formal_baseline,
            "state": state,
        }
    finally:
        identity_index.close()


def finalize(
    *,
    output_dir: Path,
    collection: dict[str, Any],
    target_total: int,
    batch_size: int,
) -> dict[str, Any]:
    papers_path = output_dir / "abstracts_ready_for_extraction.jsonl"
    links_path = output_dir / "dedup_audit_search_provenance_links.jsonl"
    paper_cases: dict[str, set[str]] = defaultdict(set)
    paper_queries: dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(links_path):
        paper_id = str(row.get("paper_id") or "")
        case_id = str(row.get("case_study_id") or "")
        query_id = str(row.get("query_id") or "")
        if paper_id and case_id:
            paper_cases[paper_id].add(case_id)
        if paper_id and query_id:
            paper_queries[paper_id].add(query_id)

    batch_dir = output_dir / "extraction_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for stale in batch_dir.glob("batch_*.jsonl"):
        stale.unlink()

    temp_queue = papers_path.with_suffix(".jsonl.finalizing")
    aliases_seen: dict[str, str] = {}
    case_nonexclusive = Counter()
    primary_counts = Counter()
    abstract_characters = 0
    abstract_min: int | None = None
    abstract_max = 0
    paper_count = 0
    current_batch: list[dict[str, Any]] = []
    batch_count = 0

    with temp_queue.open("w", encoding="utf-8", newline="\n") as queue_handle:
        for row in iter_jsonl(papers_path):
            paper_id = str(row.get("paper_id") or "")
            cases = set(row.get("search_target_case_study_ids") or [])
            cases.update(paper_cases.get(paper_id, set()))
            queries = set(row.get("search_query_ids") or [])
            queries.update(paper_queries.get(paper_id, set()))
            row["search_target_case_study_ids"] = sorted(cases)
            row["search_query_ids"] = sorted(queries)
            row["search_provenance_only"] = True
            row["assignment_policy"] = ASSIGNMENT_POLICY
            row["kg_injection"] = False
            abstract = clean_text(row.get("abstract"))
            row["abstract"] = abstract
            row["abstract_characters"] = len(abstract)
            row["abstract_words"] = len(abstract.split())
            if len(abstract) < MIN_ABSTRACT_CHARACTERS:
                raise RuntimeError(f"final queue contains short abstract: {paper_id}")

            for alias in paper_identity_aliases(row):
                prior = aliases_seen.setdefault(alias, paper_id)
                if prior != paper_id:
                    raise RuntimeError(f"final queue identity collision: {alias}")

            paper_count += 1
            case_nonexclusive.update(cases)
            primary_counts.update([str(row.get("primary_search_case_study_id") or "")])
            abstract_characters += len(abstract)
            abstract_min = len(abstract) if abstract_min is None else min(abstract_min, len(abstract))
            abstract_max = max(abstract_max, len(abstract))
            queue_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

            batch_row = dict(row)
            batch_row["batch_id"] = paper_count // batch_size + (1 if paper_count % batch_size else 0)
            batch_row["batch_position"] = ((paper_count - 1) % batch_size) + 1
            current_batch.append(batch_row)
            if len(current_batch) == batch_size:
                batch_count += 1
                batch_path = batch_dir / f"batch_{batch_count:04d}.jsonl"
                with batch_path.open("w", encoding="utf-8", newline="\n") as handle:
                    for item in current_batch:
                        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                current_batch.clear()

    if current_batch:
        batch_count += 1
        batch_path = batch_dir / f"batch_{batch_count:04d}.jsonl"
        with batch_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in current_batch:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_queue.replace(papers_path)

    after = formal_snapshot()
    baseline_files = collection["formal_baseline"]["files"]
    if after != baseline_files:
        raise RuntimeError("formal KG changed during collection; refusing completion seal")

    status = "complete" if paper_count >= target_total else "source_exhausted_shortfall"
    manifest = {
        "schema_version": "sparse_case_study_literature_collection.v3",
        "created_at": collection["formal_baseline"]["captured_at"],
        "completed_at": utc_now(),
        "status": status,
        "mode": "search_deduplicate_download_full_abstracts_only",
        "years": {"start": YEAR_START, "end": YEAR_END},
        "target_unique_papers": target_total,
        "collected_unique_papers": paper_count,
        "batch_size": batch_size,
        "total_batches": batch_count,
        "source": "Europe PMC official REST API",
        "priority_rationale": "lowest formal paper counts first; drug_repurposing intentionally excluded",
        "targets": [
            {
                "display_number": target.display_number,
                "case_study_id": target.case_study_id,
                "formal_baseline_papers": target.baseline_papers,
                "planned_primary_quota": target.quota,
                "collected_primary_papers": int(primary_counts[target.case_study_id]),
                "search_provenance_papers_nonexclusive": int(case_nonexclusive[target.case_study_id]),
                "queries": [
                    {"query_id": query.query_id, "expression": query.expression}
                    for query in target.queries
                ],
            }
            for target in TARGETS
        ],
        "abstracts": {
            "full_provider_abstracts_not_truncated": True,
            "minimum_characters_gate": MIN_ABSTRACT_CHARACTERS,
            "minimum_observed_characters": abstract_min,
            "maximum_observed_characters": abstract_max,
            "mean_observed_characters": round(abstract_characters / paper_count, 3)
            if paper_count
            else None,
        },
        "deduplication": {
            **collection["deduplication"],
            "identity_order": [
                "PMID",
                "DOI",
                "PMCID",
                "arXiv",
                "OpenAlex",
                "normalized title + year",
            ],
            "formal_kg_and_active_staging_checked_before_acceptance": True,
            "within_campaign_unique_aliases": len(aliases_seen),
        },
        "search_state": collection["state"],
        "assignment_policy": ASSIGNMENT_POLICY,
        "claim_extraction_performed": False,
        "kg_injection": False,
        "formal_kg_mutated": False,
        "formal_kg_files": after,
        "outputs": {
            "ready_queue": str(papers_path.resolve()),
            "extraction_batches": str(batch_dir.resolve()),
            "dedup_audit": str((output_dir / "dedup_audit_excluded_papers.jsonl").resolve()),
            "search_provenance_links": str(links_path.resolve()),
        },
        "output_sha256": {
            "ready_queue": hash_file(papers_path),
            "collection_state": hash_file(output_dir / "collection_state.json"),
        },
        "elapsed_seconds": collection["elapsed_seconds"],
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "COLLECTION_COMPLETE.json",
        {
            "schema_version": "collection_complete.v1",
            "completed_at": manifest["completed_at"],
            "status": status,
            "papers": paper_count,
            "batches": batch_count,
            "ready_queue_sha256": manifest["output_sha256"]["ready_queue"],
            "kg_injection": False,
        },
    )
    (output_dir / "README.md").write_text(
        "# KG v3 sparse Case Study literature top-up\n\n"
        "Collection-only staging corpus. Papers were searched in Europe PMC for "
        "1980–2026, deduplicated against the formal KG and active staging before "
        "acceptance, and retain complete provider abstracts. Search target labels "
        "are provenance only. Claim extraction must independently assign all "
        "applicable Case Study IDs; labels are non-exclusive. Nothing in this "
        "directory has been injected into the formal KG.\n",
        encoding="utf-8",
    )
    return manifest


def print_plan(target_total: int) -> None:
    payload = {
        "years": [YEAR_START, YEAR_END],
        "target_unique_papers": target_total,
        "planned_quota_sum": sum(target.quota for target in TARGETS),
        "targets": [
            {
                "display_number": target.display_number,
                "case_study_id": target.case_study_id,
                "baseline_papers": target.baseline_papers,
                "quota": target.quota,
                "query_ids": [query.query_id for query in target.queries],
            }
            for target in TARGETS
        ],
        "excluded_by_policy": ["drug_repurposing"],
        "claim_extraction": False,
        "kg_injection": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-total", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.target_total <= 0:
        raise SystemExit("--target-total must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 1 <= args.page_size <= 1000:
        raise SystemExit("--page-size must be in [1, 1000]")
    if args.print_plan:
        print_plan(args.target_total)
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    collection = collect(
        output_dir=output_dir,
        target_total=args.target_total,
        page_size=args.page_size,
    )
    manifest = finalize(
        output_dir=output_dir,
        collection=collection,
        target_total=args.target_total,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
