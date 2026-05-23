"""Batch claim extraction pipeline for multiple brain diseases.

Searches PubMed for papers across multiple diseases and years,
extracts claims via LLM, and saves everything to the knowledge graph.

Features:
  - Checkpoint/resume: progress saved after each batch
  - CSV metadata export
  - Rate limiting for NCBI and LLM APIs
  - Configurable diseases, year range, papers per year

Usage:
    /c/Users/45846/anaconda3/envs/neuroclaw/python.exe -m neurooracle.batch_extract
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Optional

from .claim_extractor import ClaimExtractor
from .claim_ingestion import ingest_claims
from .graph_manager import KnowledgeGraph
from .schema import PaperRef
from .storage import load_graph, save_graph
from .abstract_cache import AbstractCache, default_cache_path

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────

DISEASES = [
    "Alzheimer's disease",
    "Parkinson's disease",
    "depression",
    "schizophrenia",
    "ADHD",
    "autism spectrum disorder",
    "epilepsy",
    "multiple sclerosis",
    "anxiety disorder",
    "bipolar disorder",
]

YEAR_START = 2000
YEAR_END = 2026
PAPERS_PER_YEAR = 20
NCBI_API_KEY = "1e72705978ad50249ffc129798ba3958f308"

DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_FILE = DATA_DIR / "batch_checkpoint.json"
PAPERS_CSV = DATA_DIR / "papers_metadata.csv"
GRAPH_FILE = DATA_DIR / "knowledge_graph.json"
CLAIMS_FILE = DATA_DIR / "extracted_claims.jsonl"


# ── PubMed Search ──────────────────────────────────────────────────

def _search_pubmed(query: str, max_results: int) -> list[str]:
    """Search PubMed and return list of PMIDs. Retries with backoff on 429/503."""
    import requests

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
        "api_key": NCBI_API_KEY,
    }
    backoff = 2.0
    for attempt in range(4):
        try:
            resp = requests.get(search_url, params=params, timeout=30)
            if resp.status_code in (429, 502, 503):
                logger.warning(f"PubMed esearch {resp.status_code}, backing off {backoff:.0f}s (attempt {attempt+1})")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.warning(f"PubMed search failed (attempt {attempt+1}): {e}")
            time.sleep(backoff)
            backoff *= 2
    return []


def _fetch_pubmed_details(
    pmids: list[str],
    cache: Optional[AbstractCache] = None,
) -> list[tuple[str, PaperRef]]:
    """Fetch paper details for a list of PMIDs.

    Uses POST and batches to avoid HTTP 414 (URI Too Long) that hits when
    fetching hundreds of PMIDs in a single GET request.

    If ``cache`` is provided, pmids already in the cache are served from
    disk and excluded from the network call; newly fetched abstracts are
    written back so subsequent runs can skip PubMed entirely.
    """
    import requests
    import xml.etree.ElementTree as ET

    if not pmids:
        return []

    papers: list[tuple[str, PaperRef]] = []

    # Cache hit path — short-circuit before hitting NCBI.
    pmids_to_fetch: list[str] = []
    if cache is not None:
        hits, misses = cache.get_many(pmids)
        if hits:
            papers.extend(hits)
            logger.info(f"  cache: {len(hits)}/{len(pmids)} pmids served from disk")
        pmids_to_fetch = misses
    else:
        pmids_to_fetch = list(pmids)

    if not pmids_to_fetch:
        return papers

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    # Batch size 200 keeps the POST body well within NCBI limits; use POST
    # so the id list goes in the request body rather than the URL.
    BATCH_SIZE = 200
    new_records: list[tuple[str, str, PaperRef]] = []

    for i in range(0, len(pmids_to_fetch), BATCH_SIZE):
        batch = pmids_to_fetch[i:i + BATCH_SIZE]
        data = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
            "api_key": NCBI_API_KEY,
        }
        try:
            resp = requests.post(fetch_url, data=data, timeout=120)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            logger.warning(f"PubMed fetch failed (batch {i}-{i+len(batch)}): {e}")
            continue

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None else ""

                abstract_el = article.find(".//AbstractText")
                abstract = abstract_el.text if abstract_el is not None else ""
                if not abstract or not abstract.strip():
                    continue

                authors = []
                for author in article.findall(".//Author")[:5]:
                    last = author.find("LastName")
                    first = author.find("ForeName")
                    if last is not None:
                        name = last.text or ""
                        if first is not None and first.text:
                            name += " " + first.text
                        authors.append(name)
                authors_str = ", ".join(authors)

                year_el = article.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None and year_el.text else None

                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else ""

                doi_el = article.find(".//ArticleIdList/ArticleId[@IdType='doi']")
                doi = doi_el.text if doi_el is not None else ""

                paper_ref = PaperRef(
                    pmid=pmid, doi=doi, title=title,
                    authors=authors_str, year=year, journal=journal,
                )
                papers.append((abstract, paper_ref))
                if cache is not None and pmid:
                    new_records.append((pmid, abstract, paper_ref))

            except Exception as e:
                logger.warning(f"failed to parse article: {e}")
                continue

    if cache is not None and new_records:
        n = cache.put_many(new_records)
        logger.info(f"  cache: wrote {n} new abstracts ({len(cache):,} total)")

    return papers


def search_disease_year(
    disease: str,
    year: int,
    max_results: int = 20,
    broad: bool = False,
    cache: Optional[AbstractCache] = None,
) -> tuple[list[tuple[str, PaperRef]], int]:
    """Search PubMed for papers about a specific disease and year.

    Returns:
        (papers, total_hits): papers list and total PubMed hit count for this query.
    """
    if broad:
        # broad: disease + any neuroscience-related term
        query = (
            f'({disease}[Title/Abstract]) '
            f'AND ("brain"[Title/Abstract] OR "neural"[Title/Abstract] '
            f'OR "cognitive"[Title/Abstract] OR "neurological"[Title/Abstract] '
            f'OR "psychiatric"[Title/Abstract] OR "cerebrospinal"[Title/Abstract] '
            f'OR "brain imaging"[Title/Abstract] OR "neuroimaging"[Title/Abstract] '
            f'OR "MRI"[Title/Abstract] OR "fMRI"[Title/Abstract] OR "PET"[Title/Abstract] '
            f'OR "EEG"[Title/Abstract] OR "CT"[Title/Abstract]) '
            f'AND {year}:{year}[pdat]'
        )
    else:
        # narrow: disease + imaging modalities only
        query = (
            f'({disease}[Title/Abstract]) '
            f'AND ("brain imaging"[Title/Abstract] OR "neuroimaging"[Title/Abstract] '
            f'OR "MRI"[Title/Abstract] OR "fMRI"[Title/Abstract] OR "PET"[Title/Abstract]) '
            f'AND {year}:{year}[pdat]'
        )

    # get total hit count first
    import requests
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "retmode": "json",
        "api_key": NCBI_API_KEY,
    }
    total_hits = 0
    backoff = 2.0
    for attempt in range(4):
        try:
            resp = requests.get(search_url, params=params, timeout=30)
            if resp.status_code in (429, 502, 503):
                logger.warning(f"PubMed count {resp.status_code}, backing off {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            total_hits = int(resp.json().get("esearchresult", {}).get("count", 0))
            # Heuristic: for common disease queries in years 2000+, PubMed always has
            # non-trivial hits. A count of 0 is almost certainly a silent rate-limit
            # response from NCBI rather than a truly empty year. Retry a few times
            # with backoff; if it persists, we'll leave total_hits=0 and the caller
            # will decide whether to mark done.
            if total_hits == 0 and attempt < 3:
                logger.warning(f"PubMed count=0 (likely throttled), retry {attempt+1} after {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            break
        except Exception as e:
            logger.warning(f"PubMed count failed (attempt {attempt+1}): {e}")
            time.sleep(backoff)
            backoff *= 2

    time.sleep(0.4)

    pmids = _search_pubmed(query, max_results)
    if not pmids:
        return [], total_hits

    time.sleep(0.4)  # NCBI rate limit
    return _fetch_pubmed_details(pmids, cache=cache), total_hits


# ── Checkpoint Management ──────────────────────────────────────────

def _load_checkpoint(checkpoint_file: Path = None) -> dict:
    """Load checkpoint to resume from where we left off."""
    checkpoint_file = checkpoint_file or CHECKPOINT_FILE
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {
        "completed_diseases": [],
        "completed_years": {},  # {disease: [years]}
        "total_papers": 0,
        "total_claims": 0,
        "paper_counts": {},  # {disease: {year: {total_hits, fetched}}}
        "last_updated": "",
    }


def _save_checkpoint(checkpoint: dict, checkpoint_file: Path = None):
    """Save checkpoint."""
    checkpoint_file = checkpoint_file or CHECKPOINT_FILE
    checkpoint["last_updated"] = datetime.now().isoformat()
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)


# ── CSV Export ─────────────────────────────────────────────────────

def _init_csv(csv_path: Path):
    """Initialize CSV file with headers."""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pmid", "doi", "title", "authors", "year", "journal",
                "disease", "abstract_length", "n_claims_extracted",
                "extraction_timestamp",
            ])


def _append_to_csv(csv_path: Path, papers_meta: list[dict]):
    """Append paper metadata to CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for meta in papers_meta:
            writer.writerow([
                meta.get("pmid", ""),
                meta.get("doi", ""),
                meta.get("title", ""),
                meta.get("authors", ""),
                meta.get("year", ""),
                meta.get("journal", ""),
                meta.get("disease", ""),
                meta.get("abstract_length", ""),
                meta.get("n_claims", ""),
                meta.get("timestamp", ""),
            ])


# ── Claims JSONL Export ────────────────────────────────────────────

def _append_claims_to_jsonl(
    jsonl_path: Path,
    results: list,
    disease: str,
    year: int,
):
    """Append extracted claims to JSONL file (one JSON object per line)."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for result in results:
            for claim in result.claims:
                record = claim.to_dict()
                record["disease"] = disease
                record["year"] = year
                record["extraction_timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main Pipeline ──────────────────────────────────────────────────

def run_batch_extraction(
    diseases: Optional[list[str]] = None,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    papers_per_year: int = PAPERS_PER_YEAR,
    resume: bool = True,
    broad: bool = False,
    max_workers: int = 8,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
) -> dict:
    """Run batch claim extraction across multiple diseases and years.

    Args:
        diseases: List of disease names to search. Defaults to DISEASES.
        year_start: Start year (inclusive).
        year_end: End year (inclusive).
        papers_per_year: Number of papers to fetch per disease per year.
        resume: Whether to resume from checkpoint.
        broad: Use broader PubMed query.
        max_workers: Number of parallel LLM workers. Default 8.
        data_dir: Output directory for KG/checkpoint/CSV/JSONL. Defaults to
            module-level DATA_DIR. Pass a different path to run isolated streams
            (e.g. quick 20-papers/year vs full 500-papers/year in parallel).

    Returns:
        Summary dict.
    """
    diseases = diseases or DISEASES

    # resolve output paths (per-run instead of module globals)
    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = data_dir / "batch_checkpoint.json"
        papers_csv = data_dir / "papers_metadata.csv"
        graph_file = data_dir / "knowledge_graph.json"
        claims_file = data_dir / "extracted_claims.jsonl"
    else:
        checkpoint_file = CHECKPOINT_FILE
        papers_csv = PAPERS_CSV
        graph_file = GRAPH_FILE
        claims_file = CLAIMS_FILE

    # load graph
    kg = load_graph(graph_file)
    logger.info(f"loaded graph: {kg.stats()['n_concepts']} concepts, {kg.stats()['n_edges']} edges")

    # load checkpoint
    checkpoint = _load_checkpoint(checkpoint_file) if resume else {
        "completed_diseases": [],
        "completed_years": {},
        "total_papers": 0,
        "total_claims": 0,
        "paper_counts": {},
        "last_updated": "",
    }
    completed_years = checkpoint.get("completed_years", {})
    paper_counts = checkpoint.get("paper_counts", {})  # {disease: {year: {total_hits, fetched}}}

    # init CSV
    _init_csv(papers_csv)

    # Abstract cache: persists fetched abstracts so re-runs can skip PubMed.
    # One cache file per data_dir; lookups are pmid-keyed.
    cache_path = default_cache_path(data_dir if data_dir is not None else None)
    abstract_cache = AbstractCache(cache_path)

    # init extractor
    extractor = ClaimExtractor()
    logger.info(f"using {max_workers} parallel LLM workers")

    # stats
    total_papers = checkpoint.get("total_papers", 0)
    total_claims = checkpoint.get("total_claims", 0)
    start_time = time.time()

    def _ingest_and_save(
        kg_, results_, meta_, papers_csv_, claims_file_,
        disease_years_, completed_years_, checkpoint_,
        checkpoint_file_, graph_file_, total_papers_, total_claims_,
        year_start_, keep_noise_, strict_phase1_,
    ):
        """Ingest extraction results into KG, update CSV/JSONL/checkpoint."""
        nonlocal total_papers, total_claims

        d = meta_["disease"]
        yr = meta_["year"]
        papers_list = meta_["papers"]

        batch_claims = 0
        for result in results_:
            if result.claims:
                batch_claims += len(result.claims)

        ingest_claims(kg_, results_, keep_noise=keep_noise_,
                      strict_phase1=strict_phase1_)

        papers_meta = []
        for (abstract, ref), result in zip(papers_list, results_):
            papers_meta.append({
                "pmid": ref.pmid,
                "doi": ref.doi,
                "title": ref.title,
                "authors": ref.authors,
                "year": ref.year,
                "journal": ref.journal,
                "disease": d,
                "abstract_length": len(abstract),
                "n_claims": len(result.claims),
                "timestamp": datetime.now().isoformat(),
            })
        _append_to_csv(papers_csv_, papers_meta)
        _append_claims_to_jsonl(claims_file_, results_, d, yr)

        total_papers += len(papers_list)
        total_claims += batch_claims
        logger.info(f"  {yr}: ingested {batch_claims} claims (total: {total_claims})")

        disease_years_.append(yr)
        completed_years_[d] = disease_years_
        checkpoint_["total_papers"] = total_papers
        checkpoint_["total_claims"] = total_claims
        _save_checkpoint(checkpoint_, checkpoint_file_)

        if (yr - year_start_ + 1) % 15 == 0:
            save_graph(kg_, graph_file_)
            logger.info(f"  graph checkpoint saved")

    expected_years = set(range(year_start, year_end + 1))

    # Pipeline: use a background thread for extraction so ingestion and
    # extraction of the next batch can overlap.
    extract_pool = ThreadPoolExecutor(max_workers=1)
    pending_extraction: Optional[Future] = None
    pending_meta: Optional[dict] = None  # {disease, year, papers, total_hits}

    def _submit_extraction(disease_name, yr, papers_list, hits):
        """Submit extraction job to background thread."""
        nonlocal pending_extraction, pending_meta
        pending_meta = {"disease": disease_name, "year": yr, "papers": papers_list, "total_hits": hits}
        pending_extraction = extract_pool.submit(
            extractor.extract_batch, papers_list, max_workers
        )

    def _collect_extraction():
        """Wait for pending extraction and return (meta, results)."""
        nonlocal pending_extraction, pending_meta
        if pending_extraction is None:
            return None, None
        results = pending_extraction.result()
        meta = pending_meta
        pending_extraction = None
        pending_meta = None
        return meta, results

    for disease in diseases:
        done_years = set(completed_years.get(disease, []))
        if disease in checkpoint.get("completed_diseases", []) and expected_years.issubset(done_years):
            logger.info(f"skipping {disease} (already completed)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"DISEASE: {disease}")
        logger.info(f"{'='*60}")

        disease_years = completed_years.get(disease, [])

        for year in range(year_start, year_end + 1):
            if year in disease_years:
                logger.info(f"  {year}: already done, skipping")
                continue

            logger.info(f"  {year}: searching...")
            papers, total_hits = search_disease_year(
                disease, year, papers_per_year, broad=broad, cache=abstract_cache,
            )
            logger.info(f"  {year}: found {len(papers)} papers (total hits in PubMed: {total_hits})")

            # track per-year-per-disease counts
            paper_counts.setdefault(disease, {})[str(year)] = {
                "total_hits": total_hits,
                "fetched": len(papers),
            }
            checkpoint["paper_counts"] = paper_counts

            if not papers:
                logger.warning(
                    f"  {year}: 0/{total_hits} fetched — NOT marking as done. "
                    f"Will retry on next run."
                )
                time.sleep(3)
                continue

            # If there's a pending extraction, ingest its results while we
            # kick off extraction for the current batch.
            if pending_extraction is not None:
                # Start extraction for current year in background
                # But first, collect previous results
                prev_meta, prev_results = _collect_extraction()
                if prev_meta and prev_results:
                    # Ingest previous batch (main thread, safe for KG)
                    _ingest_and_save(
                        kg, prev_results, prev_meta, papers_csv, claims_file,
                        disease_years, completed_years, checkpoint,
                        checkpoint_file, graph_file, total_papers, total_claims,
                        year_start, keep_noise, strict_phase1,
                    )
                    total_papers = checkpoint["total_papers"]
                    total_claims = checkpoint["total_claims"]

            # Submit current year extraction to background
            _submit_extraction(disease, year, papers, total_hits)

            time.sleep(0.5)

        # Collect any remaining extraction for this disease
        if pending_extraction is not None:
            prev_meta, prev_results = _collect_extraction()
            if prev_meta and prev_results:
                _ingest_and_save(
                    kg, prev_results, prev_meta, papers_csv, claims_file,
                    disease_years, completed_years, checkpoint,
                    checkpoint_file, graph_file, total_papers, total_claims,
                    year_start, keep_noise, strict_phase1,
                )
                total_papers = checkpoint["total_papers"]
                total_claims = checkpoint["total_claims"]

        # mark disease as complete only if all target years landed in checkpoint
        if expected_years.issubset(set(completed_years.get(disease, []))):
            if disease not in checkpoint.get("completed_diseases", []):
                checkpoint.setdefault("completed_diseases", []).append(disease)
            _save_checkpoint(checkpoint, checkpoint_file)

        # save graph after each disease
        save_graph(kg, graph_file)
        logger.info(f"  {disease} complete, graph saved")

    # Collect final pending extraction
    if pending_extraction is not None:
        prev_meta, prev_results = _collect_extraction()
        if prev_meta and prev_results:
            _ingest_and_save(
                kg, prev_results, prev_meta, papers_csv, claims_file,
                disease_years, completed_years, checkpoint,
                checkpoint_file, graph_file, total_papers, total_claims,
                year_start, keep_noise, strict_phase1,
            )
            total_papers = checkpoint["total_papers"]
            total_claims = checkpoint["total_claims"]

    extract_pool.shutdown(wait=False)

    # final save
    save_graph(kg, graph_file)
    elapsed = time.time() - start_time

    stats = kg.stats()
    summary = {
        "total_papers": total_papers,
        "total_claims": total_claims,
        "total_concepts": stats["n_concepts"],
        "total_edges": stats["n_edges"],
        "elapsed_minutes": elapsed / 60,
        "diseases_processed": len(diseases),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCH EXTRACTION COMPLETE")
    logger.info(f"  Papers: {total_papers}")
    logger.info(f"  Claims: {total_claims}")
    logger.info(f"  Concepts: {stats['n_concepts']}")
    logger.info(f"  Edges: {stats['n_edges']}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  CSV: {papers_csv}")
    logger.info(f"{'='*60}")

    # print paper counts summary
    logger.info(f"\nPaper counts per disease per year:")
    for disease in diseases:
        counts = paper_counts.get(disease, {})
        total_found = sum(v.get("fetched", 0) for v in counts.values())
        total_hits = sum(v.get("total_hits", 0) for v in counts.values())
        logger.info(f"  {disease}: {total_found} papers fetched / {total_hits} total PubMed hits")
        for year in range(year_start, year_end + 1):
            yr = counts.get(str(year), {})
            if yr:
                logger.info(f"    {year}: {yr.get('fetched',0)} fetched / {yr.get('total_hits',0)} hits")

    summary["paper_counts"] = paper_counts
    return summary


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch claim extraction from PubMed")
    parser.add_argument("--diseases", nargs="+", default=None, help="Disease names to search")
    parser.add_argument("--year-start", type=int, default=YEAR_START)
    parser.add_argument("--year-end", type=int, default=YEAR_END)
    parser.add_argument("--papers-per-year", type=int, default=PAPERS_PER_YEAR)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoint)")
    parser.add_argument("--broad", action="store_true", help="Use broader PubMed query (more results)")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of parallel LLM workers (default: 8)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Output directory for KG/checkpoint/CSV/JSONL. Use a unique "
                             "path to run isolated streams in parallel (quick vs full).")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--keep-noise", action="store_true",
                        help="Disable build-time noise filter; keep all auto-minted "
                             "CLM_CONCEPT entities (debug only)")
    parser.add_argument("--strict-phase1", action="store_true",
                        help="Do NOT mint any new CLM_CONCEPT nodes from Phase 2. "
                             "Claims whose subject/object cannot resolve to a "
                             "Phase-1-curated node are dropped. Use when Phase 1 "
                             "(NeuroNames/MeSH/DisGeNET/CognitiveAtlas + UMLS) is "
                             "already considered comprehensive.")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_batch_extraction(
        diseases=args.diseases,
        year_start=args.year_start,
        year_end=args.year_end,
        papers_per_year=args.papers_per_year,
        resume=not args.no_resume,
        broad=args.broad,
        max_workers=args.max_workers,
        data_dir=args.data_dir,
        keep_noise=args.keep_noise,
        strict_phase1=args.strict_phase1,
    )


if __name__ == "__main__":
    main()
