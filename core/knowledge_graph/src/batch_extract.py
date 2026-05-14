"""Batch claim extraction pipeline for multiple brain diseases.

Searches PubMed for papers across multiple diseases and years,
extracts claims via LLM, and saves everything to the knowledge graph.

Features:
  - Checkpoint/resume: progress saved after each batch
  - CSV metadata export
  - Rate limiting for NCBI and LLM APIs
  - Configurable diseases, year range, papers per year

Usage:
    /c/Users/45846/anaconda3/envs/neuroclaw/python.exe -m core.knowledge_graph.batch_extract
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .claim_extractor import ClaimExtractor
from .claim_ingestion import ingest_claims
from .graph_manager import KnowledgeGraph
from .schema import PaperRef
from .storage import load_graph, save_graph

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


def _fetch_pubmed_details(pmids: list[str]) -> list[tuple[str, PaperRef]]:
    """Fetch paper details for a list of PMIDs.

    Uses POST and batches to avoid HTTP 414 (URI Too Long) that hits when
    fetching hundreds of PMIDs in a single GET request.
    """
    import requests
    import xml.etree.ElementTree as ET

    if not pmids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    # Batch size 200 keeps the POST body well within NCBI limits; use POST
    # so the id list goes in the request body rather than the URL.
    BATCH_SIZE = 200
    papers: list[tuple[str, PaperRef]] = []

    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i + BATCH_SIZE]
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

            except Exception as e:
                logger.warning(f"failed to parse article: {e}")
                continue

    return papers


def search_disease_year(
    disease: str,
    year: int,
    max_results: int = 20,
    broad: bool = False,
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
    return _fetch_pubmed_details(pmids), total_hits


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

    # init extractor
    extractor = ClaimExtractor()
    logger.info(f"using {max_workers} parallel LLM workers")

    # stats
    total_papers = checkpoint.get("total_papers", 0)
    total_claims = checkpoint.get("total_claims", 0)
    start_time = time.time()

    expected_years = set(range(year_start, year_end + 1))
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
            papers, total_hits = search_disease_year(disease, year, papers_per_year, broad=broad)
            logger.info(f"  {year}: found {len(papers)} papers (total hits in PubMed: {total_hits})")

            # track per-year-per-disease counts
            paper_counts.setdefault(disease, {})[str(year)] = {
                "total_hits": total_hits,
                "fetched": len(papers),
            }
            checkpoint["paper_counts"] = paper_counts

            if not papers:
                # Never mark a year as "done" unless we actually fetched papers.
                # PubMed rate-limit responses often look like total_hits=0, which
                # is indistinguishable from "genuinely empty year". Better to
                # leave it open for retry on next run than skip silently.
                logger.warning(
                    f"  {year}: 0/{total_hits} fetched — NOT marking as done. "
                    f"Will retry on next run."
                )
                time.sleep(3)  # brief pause so we don't hammer NCBI if throttled
                continue

            # extract claims (parallel)
            results = extractor.extract_batch(papers, max_workers=max_workers)

            # ingest claims
            batch_claims = 0
            for result in results:
                if result.claims:
                    batch_claims += len(result.claims)

            ingest_claims(kg, results, keep_noise=keep_noise,
                          strict_phase1=strict_phase1)

            # save paper metadata to CSV
            papers_meta = []
            for (abstract, ref), result in zip(papers, results):
                papers_meta.append({
                    "pmid": ref.pmid,
                    "doi": ref.doi,
                    "title": ref.title,
                    "authors": ref.authors,
                    "year": ref.year,
                    "journal": ref.journal,
                    "disease": disease,
                    "abstract_length": len(abstract),
                    "n_claims": len(result.claims),
                    "timestamp": datetime.now().isoformat(),
                })
            _append_to_csv(papers_csv, papers_meta)

            # save claims to JSONL
            _append_claims_to_jsonl(claims_file, results, disease, year)

            total_papers += len(papers)
            total_claims += batch_claims
            logger.info(f"  {year}: extracted {batch_claims} claims (total: {total_claims})")

            # update checkpoint
            disease_years.append(year)
            completed_years[disease] = disease_years
            checkpoint["total_papers"] = total_papers
            checkpoint["total_claims"] = total_claims
            _save_checkpoint(checkpoint, checkpoint_file)

            # save graph periodically (every 5 years)
            if (year - year_start + 1) % 5 == 0:
                save_graph(kg, graph_file)
                logger.info(f"  graph checkpoint saved")

            # rate limiting
            time.sleep(0.5)

        # mark disease as complete only if all target years landed in checkpoint
        if expected_years.issubset(set(completed_years.get(disease, []))):
            if disease not in checkpoint.get("completed_diseases", []):
                checkpoint.setdefault("completed_diseases", []).append(disease)
            _save_checkpoint(checkpoint, checkpoint_file)

        # save graph after each disease
        save_graph(kg, graph_file)
        logger.info(f"  {disease} complete, graph saved")

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
