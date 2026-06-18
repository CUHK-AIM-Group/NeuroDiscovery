"""Collect Case Study 3 task-conditioned literature with query templates.

This is a high-recall supplement to the KG-atom chain collector.  It searches
by canonical task semantics, writes collect-only metadata, and caches abstracts
without extracting claims or modifying a knowledge graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from neurooracle.src.abstract_cache import AbstractCache, default_cache_path
from neurooracle.src.batch_extract import _fetch_pubmed_details, _search_pubmed
from neurooracle.src.case_targeted_extract import (
    _normalise_doi,
    _search_arxiv,
    _search_europepmc,
    _search_openalex,
    _search_preprint_server,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    ROOT
    / "neurooracle"
    / "data"
    / "cs_runs"
    / "case3_hindcasting"
    / "task_template_literature_20260616"
)


TASK_QUERIES: dict[str, list[str]] = {
    "biomarker_discovery": [
        "neuroimaging biomarker disease diagnosis MRI PET fMRI",
        "brain imaging marker clinical diagnosis prediction neurological psychiatric",
        "structural MRI biomarker disease classification cortical thickness volume",
        "functional connectivity biomarker diagnosis resting state fMRI",
        "PET biomarker amyloid tau FDG disease progression diagnosis",
        "radiomics machine learning neuroimaging biomarker diagnosis",
    ],
    "disease_subtyping": [
        "neuroimaging disease subtypes clustering MRI fMRI",
        "brain imaging patient stratification psychiatric neurological disorder",
        "connectome clustering disease subtype symptom dimension",
        "normative modeling neuroimaging subtype heterogeneity",
        "unsupervised learning MRI disease subtyping",
    ],
    "progression_prediction": [
        "neuroimaging predicts disease progression longitudinal MRI",
        "brain imaging cognitive decline progression prediction",
        "MRI PET longitudinal biomarker conversion dementia Parkinson",
        "functional connectivity predicts clinical progression follow-up",
        "cortical thinning predicts symptom progression longitudinal",
    ],
    "imaging_genetics": [
        "imaging genetics MRI gene variant brain structure",
        "GWAS neuroimaging cortical thickness brain volume",
        "polygenic score brain imaging MRI cognition",
        "APOE amyloid PET tau PET MRI cognition",
        "gene expression transcriptomic neuroimaging association",
        "genetic risk functional connectivity fMRI",
    ],
    "differential_diagnosis": [
        "neuroimaging differential diagnosis MRI psychiatric disorders",
        "MRI distinguishes Alzheimer frontotemporal dementia Lewy body",
        "brain imaging distinguish schizophrenia bipolar depression",
        "machine learning neuroimaging differential diagnosis disease classification",
        "PET MRI differential diagnosis neurological disorder",
    ],
    "drug_response_prediction": [
        "neuroimaging predicts treatment response antidepressant fMRI MRI",
        "brain imaging treatment response antipsychotic schizophrenia",
        "functional connectivity predicts drug response depression",
        "PET MRI biomarker treatment response neurological psychiatric",
        "baseline neuroimaging predicts medication response clinical outcome",
    ],
    "personalised_treatment": [
        "neuroimaging personalized treatment prediction psychiatric disorder",
        "MRI fMRI guide treatment selection depression schizophrenia",
        "brain imaging precision medicine treatment stratification",
        "connectivity biomarker personalized therapy response",
        "machine learning neuroimaging treatment recommendation",
    ],
    "drug_repurposing": [
        "drug repurposing neuroimaging biomarker neurological disease",
        "pharmacoimaging drug repurposing brain disorder",
        "connectivity signature drug repurposing psychiatric neurological",
        "transcriptomic neuroimaging drug repurposing brain disease",
    ],
    "adverse_event_prediction": [
        "neuroimaging predicts adverse events treatment brain",
        "MRI biomarker adverse effect medication neurological psychiatric",
        "brain imaging toxicity prediction drug adverse event",
        "functional connectivity adverse effect treatment response",
    ],
    "neuromodulation_target": [
        "neuroimaging neuromodulation target deep brain stimulation",
        "fMRI connectivity TMS target depression OCD",
        "brain stimulation target imaging biomarker clinical outcome",
        "DBS target connectivity Parkinson depression neuroimaging",
        "lesion network mapping stimulation target neurological psychiatric",
    ],
    "functional_localization": [
        "task fMRI functional localization cognitive task brain region",
        "language memory motor task fMRI activation localization",
        "functional mapping MRI cognitive neuroscience task",
        "brain region activation task performance fMRI",
    ],
    "cognitive_decoding": [
        "fMRI cognitive decoding brain activity task prediction",
        "neuroimaging decoding cognition memory attention language",
        "machine learning fMRI decode cognitive state",
        "brain activation predicts task cognitive performance",
    ],
    "connectome_behavior": [
        "functional connectivity behavior cognition individual differences",
        "connectome predicts cognitive performance personality symptoms",
        "resting state fMRI behavior prediction connectome",
        "structural connectivity behavior cognition diffusion MRI",
        "brain connectivity clinical symptom severity prediction",
    ],
    "brain_age": [
        "brain age MRI prediction cognition disease",
        "brain age gap neuroimaging psychiatric neurological disorder",
        "deep learning brain age MRI clinical outcome",
        "brain age cognitive decline longitudinal neuroimaging",
        "brain age biomarker depression schizophrenia Alzheimer",
    ],
    "prognosis": [
        "neuroimaging prognosis clinical outcome longitudinal disease",
        "MRI predicts prognosis neurological psychiatric disorder",
        "brain imaging survival disability outcome prediction",
        "functional connectivity prognosis symptom outcome follow-up",
        "PET MRI prognostic biomarker disease outcome",
    ],
}


SOURCE_CHOICES = ("pubmed", "openalex", "europepmc", "arxiv", "biorxiv", "medrxiv")


PUBMED_IMAGING_TERMS = [
    "neuroimaging",
    "brain imaging",
    "MRI",
    "fMRI",
    "PET",
    "SPECT",
    "DTI",
    "diffusion MRI",
    "functional connectivity",
    "connectome",
    "cortical thickness",
    "brain volume",
]


TASK_PUBMED_TERMS: dict[str, list[str]] = {
    "biomarker_discovery": [
        "biomarker", "marker", "diagnosis", "classification", "prediction",
    ],
    "disease_subtyping": [
        "subtype", "subtyping", "cluster", "clustering", "stratification", "heterogeneity",
    ],
    "progression_prediction": [
        "progression", "conversion", "longitudinal", "cognitive decline", "follow-up",
    ],
    "imaging_genetics": [
        "imaging genetics", "GWAS", "polygenic", "genetic", "gene expression", "transcriptomic",
    ],
    "differential_diagnosis": [
        "differential diagnosis", "distinguish", "discriminate", "classification",
    ],
    "drug_response_prediction": [
        "treatment response", "drug response", "medication response",
        "antidepressant", "antipsychotic",
    ],
    "personalised_treatment": [
        "personalized treatment", "personalised treatment", "precision medicine",
        "treatment selection", "treatment stratification",
    ],
    "drug_repurposing": [
        "drug repurposing", "pharmacoimaging", "connectivity signature", "drug",
    ],
    "adverse_event_prediction": [
        "adverse event", "adverse effect", "side effect", "toxicity", "safety",
    ],
    "neuromodulation_target": [
        "neuromodulation", "TMS", "DBS", "brain stimulation", "stimulation target",
    ],
    "functional_localization": [
        "functional localization", "activation", "mapping", "task fMRI", "brain region",
    ],
    "cognitive_decoding": [
        "decoding", "decode", "cognitive state", "machine learning", "prediction",
    ],
    "connectome_behavior": [
        "behavior", "behaviour", "cognition", "symptom", "individual differences",
    ],
    "brain_age": [
        "brain age", "age gap", "predicted age", "brain aging", "brain ageing",
    ],
    "prognosis": [
        "prognosis", "outcome", "survival", "disability", "clinical outcome",
    ],
}


def _tiab_or(terms: Iterable[str]) -> str:
    cleaned = [t.strip() for t in terms if t and t.strip()]
    return "(" + " OR ".join(f'"{term}"[Title/Abstract]' for term in cleaned) + ")"


def _pubmed_query(phrase: str, year_start: int, year_end: int) -> str:
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", phrase) if len(t) > 2]
    core = " AND ".join(f"{term}[Title/Abstract]" for term in terms[:8])
    if not core:
        core = f'"{phrase}"[Title/Abstract]'
    return f"({core}) AND {year_start}:{year_end}[pdat]"


def _pubmed_broad_query(task: str, phrase: str, year_start: int, year_end: int) -> str:
    """High-recall fallback for task searches whose strict phrase query is empty."""
    phrase_terms = [
        t for t in re.findall(r"[A-Za-z][A-Za-z0-9-]+", phrase)
        if len(t) > 3 and t.lower() not in {"brain", "disease", "clinical"}
    ][:5]
    task_terms = TASK_PUBMED_TERMS.get(task, [])
    return (
        f"{_tiab_or(PUBMED_IMAGING_TERMS)} AND "
        f"{_tiab_or(task_terms)} AND "
        f"{_tiab_or(phrase_terms)} AND "
        f"{year_start}:{year_end}[pdat]"
    )


def _year_windows(year_start: int, year_end: int, window: int) -> list[tuple[int, int]]:
    if window <= 0:
        return [(year_start, year_end)]
    spans = []
    start = year_start
    while start <= year_end:
        end = min(year_end, start + window - 1)
        spans.append((start, end))
        start = end + 1
    return spans


def _init_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "doi", "title", "authors", "year", "journal", "source",
            "label", "query", "abstract_length", "collected_at",
        ])


def _load_seen(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    dois: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ("id", "pmid"):
                    value = (row.get(key) or "").strip()
                    if value:
                        ids.add(value)
                doi = _normalise_doi(row.get("doi") or "")
                if doi:
                    dois.add(doi)
    return ids, dois


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    _init_csv(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "doi", "title", "authors", "year", "journal", "source",
            "label", "query", "abstract_length", "collected_at",
        ])
        writer.writerows(rows)


def _search_source(
    source: str,
    phrase: str,
    *,
    year_start: int,
    year_end: int,
    max_results: int,
    preprint_target_papers: int | None = None,
    preprint_max_scan: int | None = None,
) -> list[tuple[str, str, object]]:
    if source == "pubmed":
        pmids = _search_pubmed(_pubmed_query(phrase, year_start, year_end), max_results)
        papers = _fetch_pubmed_details(pmids)
        return [(paper.pmid, abstract, paper) for abstract, paper in papers]
    if source == "openalex":
        return _search_openalex(
            phrase,
            year_start=year_start,
            year_end=year_end,
            max_results=max_results,
            sort="cited_by_count:desc",
        )
    if source == "europepmc":
        return _search_europepmc(
            phrase,
            year_start=year_start,
            year_end=year_end,
            max_results=max_results,
        )
    if source == "arxiv":
        return _search_arxiv(
            phrase,
            year_start=year_start,
            year_end=year_end,
            max_results=max_results,
        )
    if source in ("biorxiv", "medrxiv"):
        target_papers = preprint_target_papers or max_results
        max_scan = preprint_max_scan or max_results * 30
        records, _scanned = _search_preprint_server(
            source,
            preset="case3_task_template",
            phrases=[phrase],
            year_start=year_start,
            year_end=year_end,
            target_papers=target_papers,
            max_scan=max_scan,
        )
        return records
    raise ValueError(f"unknown source: {source}")


def collect(
    *,
    data_dir: Path,
    tasks: list[str],
    sources: list[str],
    year_start: int,
    year_end: int,
    max_results: int,
    existing_metadata: list[Path],
    include_seen: bool,
    preprint_target_papers: int | None = None,
    preprint_max_scan: int | None = None,
    year_window: int = 0,
) -> dict[str, object]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "collection_metadata.csv"
    summary_path = data_dir / "case3_task_template_collection_summary.json"
    cache = AbstractCache(default_cache_path(data_dir))
    seen_ids, seen_dois = (set(), set()) if include_seen else _load_seen([csv_path, *existing_metadata])
    logging.info("seen refs: ids=%d dois=%d", len(seen_ids), len(seen_dois))

    summary: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "year_start": year_start,
        "year_end": year_end,
        "rows": [],
    }
    total_added = 0

    windows = _year_windows(year_start, year_end, year_window)

    for task in tasks:
        phrases = TASK_QUERIES[task]
        for source in sources:
            added = 0
            hits = []
            rows = []
            query_counter = 0
            for win_start, win_end in windows:
                for idx, phrase in enumerate(phrases, 1):
                    query_counter += 1
                    try:
                        if source == "pubmed":
                            pmids = _search_pubmed(
                                _pubmed_query(phrase, win_start, win_end),
                                max_results,
                            )
                            if not pmids:
                                pmids = _search_pubmed(
                                    _pubmed_broad_query(task, phrase, win_start, win_end),
                                    max_results,
                                )
                            pmids = [pmid for pmid in pmids if pmid not in seen_ids]
                            papers = _fetch_pubmed_details(pmids, cache=cache)
                            records = [(paper.pmid, abstract, paper) for abstract, paper in papers]
                        else:
                            records = _search_source(
                                source,
                                phrase,
                                year_start=win_start,
                                year_end=win_end,
                                max_results=max_results,
                                preprint_target_papers=preprint_target_papers,
                                preprint_max_scan=preprint_max_scan,
                            )
                    except Exception as exc:
                        logging.warning("%s %s query failed: %s", task, source, exc)
                        records = []
                    q_added = 0
                    to_cache = []
                    now = datetime.now().isoformat()
                    for cache_id, abstract, paper in records:
                        doi = _normalise_doi(getattr(paper, "doi", ""))
                        if cache_id in seen_ids or (doi and doi in seen_dois):
                            continue
                        seen_ids.add(cache_id)
                        if doi:
                            seen_dois.add(doi)
                        to_cache.append((cache_id, abstract, paper))
                        rows.append({
                            "id": cache_id,
                            "doi": getattr(paper, "doi", ""),
                            "title": getattr(paper, "title", ""),
                            "authors": getattr(paper, "authors", ""),
                            "year": getattr(paper, "year", ""),
                            "journal": getattr(paper, "journal", ""),
                            "source": source,
                            "label": f"task:{task}",
                            "query": phrase,
                            "abstract_length": len(abstract or ""),
                            "collected_at": now,
                        })
                        q_added += 1
                    if to_cache:
                        cache.put_many(to_cache)
                    hits.append({
                        "query_index": idx,
                        "year_start": win_start,
                        "year_end": win_end,
                        "hits": len(records),
                        "added": q_added,
                    })
                    added += q_added
                    logging.info(
                        "%s source=%s query %d/%d window=%s-%s hits=%d added=%d total_added=%d",
                        task, source, query_counter, len(phrases) * len(windows),
                        win_start, win_end, len(records), q_added, added,
                    )
                    time.sleep(3.1 if source == "arxiv" else 0.6)
            _append_rows(csv_path, rows)
            total_added += added
            row = {"task": task, "source": source, "added": added, "query_hits": hits}
            summary["rows"].append(row)  # type: ignore[index]
            summary["updated_at"] = datetime.now().isoformat()
            summary["total_added"] = total_added
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASK_QUERIES), default=sorted(TASK_QUERIES))
    parser.add_argument("--sources", nargs="*", choices=SOURCE_CHOICES, default=["pubmed", "openalex", "europepmc"])
    parser.add_argument("--year-start", type=int, default=2000)
    parser.add_argument("--year-end", type=int, default=2026)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--existing-metadata", type=Path, action="append", default=[])
    parser.add_argument("--include-seen", action="store_true")
    parser.add_argument(
        "--year-window",
        type=int,
        default=0,
        help="Split searches into N-year windows for higher recall; 0 keeps one full-range search.",
    )
    parser.add_argument(
        "--preprint-target-papers",
        type=int,
        default=None,
        help="Per-query target for bioRxiv/medRxiv searches (default: --max-results).",
    )
    parser.add_argument(
        "--preprint-max-scan",
        type=int,
        default=None,
        help="Per-query scan cap for bioRxiv/medRxiv searches (default: --max-results * 30).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = collect(
        data_dir=args.data_dir,
        tasks=args.tasks,
        sources=args.sources,
        year_start=args.year_start,
        year_end=args.year_end,
        max_results=args.max_results,
        existing_metadata=args.existing_metadata,
        include_seen=args.include_seen,
        preprint_target_papers=args.preprint_target_papers,
        preprint_max_scan=args.preprint_max_scan,
        year_window=args.year_window,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
