"""Chain-aware Phase 2 driver: search PubMed for papers covering a target
``Task`` / ``TaskChain`` (rather than disease × imaging modality), extract
claims, and ingest. Also exposes a ``--rerun-cached`` mode that re-extracts
from the local abstract cache without hitting PubMed at all.

This module reuses the building blocks from ``batch_extract`` (LLM
extractor, ingest, CSV/JSONL append, checkpoint) but swaps the search
strategy and adds two new entry points:

  * ``run_chain_extraction(chain_or_task)`` — generate compound queries
    from KG top-K terms, fetch + extract + ingest. Cache hits skip PubMed.
  * ``run_rerun_cached()`` — iterate the local abstract cache, re-extract
    every (or sampled subset of) cached abstract under the current prompt,
    and ingest. No network calls.
  * ``run_fill_sparse()`` — rank chains by claim sparsity (chain_coverage),
    then call ``run_chain_extraction`` for each below threshold.

Usage::

    python -m neurooracle.phase2 chain --chain genetic_imaging_disease \
        --year-start 2018 --year-end 2025 --max-results 200

    python -m neurooracle.phase2 rerun-cached --max-papers 1000

    python -m neurooracle.phase2 fill-sparse --min-claims 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .abstract_cache import AbstractCache, default_cache_path
from .atoms import (
    Atom, Task, TaskChain, CANONICAL_TASKS, CANONICAL_CHAINS,
    chain_by_name, task_by_name,
)
from .batch_extract import (
    DATA_DIR, NCBI_API_KEY,
    _search_pubmed, _fetch_pubmed_details, _init_csv,
    _append_to_csv, _append_claims_to_jsonl,
)
from .chain_coverage import analyse as analyse_coverage, find_sparse
from .chain_queries import build_chain_queries, build_task_queries
from .claim_extractor import ClaimExtractor
from .claim_ingestion import ingest_claims
from .schema import PaperRef
from .storage import load_graph, save_graph

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_data_paths(data_dir: Optional[Path]):
    if data_dir is None:
        ddir = DATA_DIR
    else:
        ddir = Path(data_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": ddir,
        "checkpoint": ddir / "batch_checkpoint.json",
        "papers_csv": ddir / "papers_metadata.csv",
        "graph": ddir / "knowledge_graph.json",
        "claims": ddir / "extracted_claims.jsonl",
    }


def _ingest_results(
    kg, results, papers_list, paths, *, disease_label: str, year_label: int,
    keep_noise: bool, strict_phase1: bool,
) -> tuple[int, int]:
    """Ingest a batch and append to CSV/JSONL. Returns (n_papers, n_claims)."""
    batch_claims = sum(len(r.claims) for r in results if r and r.claims)

    ingest_claims(kg, results, keep_noise=keep_noise, strict_phase1=strict_phase1)

    papers_meta = []
    for (abstract, ref), result in zip(papers_list, results):
        papers_meta.append({
            "pmid": ref.pmid,
            "doi": ref.doi,
            "title": ref.title,
            "authors": ref.authors,
            "year": ref.year,
            "journal": ref.journal,
            "disease": disease_label,
            "abstract_length": len(abstract),
            "n_claims": len(result.claims) if result else 0,
            "timestamp": datetime.now().isoformat(),
            "extraction_error": result.error if result else "missing extraction result",
        })
    _append_to_csv(paths["papers_csv"], papers_meta)
    _append_claims_to_jsonl(paths["claims"], results, disease_label, year_label)

    return len(papers_list), batch_claims


# ── Mode 1: chain-aware extraction ─────────────────────────────────────────────


def run_chain_extraction(
    chain_or_task_name: str,
    *,
    is_chain: bool,
    year_start: int = 2010,
    year_end: int = 2025,
    max_results_per_query: int = 200,
    terms_per_atom: int = 12,
    n_subqueries: int = 3,
    max_workers: int = 12,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
    sample_rate_seen: float = 0.05,
    lock_model: bool = False,
) -> dict:
    """Run chain/task-targeted PubMed extraction.

    ``sample_rate_seen``: pmids already in our local extraction history
    (i.e. that produced claims in earlier runs) are skipped, except that
    a random ``sample_rate_seen`` fraction are re-extracted as QC.
    """
    paths = _resolve_data_paths(data_dir)
    kg = load_graph(paths["graph"])
    logger.info(f"loaded graph: {kg.stats()['n_concepts']} concepts, "
                f"{kg.stats()['n_edges']} edges")

    # Build queries from KG top-K
    if is_chain:
        chain = chain_by_name(chain_or_task_name)
        target_atoms = chain.chain
        label = f"chain:{chain.name}"
    else:
        task = task_by_name(chain_or_task_name)
        target_atoms = tuple(task.inputs) + (task.output,)
        label = f"task:{task.name}"

    # Existing-pmid set: from papers_metadata.csv (proxy for "already extracted").
    seen_pmids = _load_seen_pmids(paths["papers_csv"])
    logger.info(f"seen-pmids index: {len(seen_pmids):,}")

    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    extractor = ClaimExtractor(lock_model=lock_model)
    _init_csv(paths["papers_csv"])

    rng = random.Random(0xCFEB)

    total_papers = 0
    total_claims = 0
    skipped_seen = 0
    resampled = 0
    qc_pmids: list[str] = []

    for year in range(year_start, year_end + 1):
        if is_chain:
            queries = build_chain_queries(
                chain, kg, year=year,
                terms_per_atom=terms_per_atom,
                n_subqueries=n_subqueries,
            )
        else:
            queries = build_task_queries(
                task, kg, year=year,
                terms_per_atom=terms_per_atom,
                n_subqueries=n_subqueries,
            )
        if not queries:
            logger.warning(f"  {year}: no queries generated (some atom had 0 KG terms); skipping")
            continue

        logger.info(f"  {year}: {len(queries)} compound queries → searching PubMed")
        batch_pmids: set[str] = set()
        for q in queries:
            try:
                pmids = _search_pubmed(q, max_results_per_query)
                logger.info(f"    [{year}] {len(pmids)} hits  q={q[:120]}...")
                batch_pmids.update(pmids)
            except Exception as e:
                logger.warning(f"    search failed: {e}")
            time.sleep(0.4)

        if not batch_pmids:
            logger.info(f"  {year}: 0 pmids across all queries")
            continue

        # Partition: new pmids vs. already-seen pmids.
        new_pmids: list[str] = []
        seen_in_batch: list[str] = []
        for p in batch_pmids:
            if p in seen_pmids:
                seen_in_batch.append(p)
            else:
                new_pmids.append(p)
        # QC sample of seen pmids
        sample_n = int(round(len(seen_in_batch) * sample_rate_seen))
        if sample_n > 0:
            qc_sample = rng.sample(seen_in_batch, min(sample_n, len(seen_in_batch)))
        else:
            qc_sample = []
        skipped_seen += len(seen_in_batch) - len(qc_sample)
        resampled += len(qc_sample)
        qc_pmids.extend(qc_sample)
        to_fetch = new_pmids + qc_sample
        logger.info(f"  {year}: {len(new_pmids)} new + {len(qc_sample)} QC-resampled "
                    f"(skipped {len(seen_in_batch) - len(qc_sample)} already-seen)")

        if not to_fetch:
            continue

        # Fetch (cache-aware)
        papers = _fetch_pubmed_details(to_fetch, cache=cache)
        if not papers:
            continue

        # Extract + ingest
        results = extractor.extract_batch(papers, max_workers=max_workers)
        n_papers, n_claims = _ingest_results(
            kg, results, papers, paths,
            disease_label=label, year_label=year,
            keep_noise=keep_noise, strict_phase1=strict_phase1,
        )
        total_papers += n_papers
        total_claims += n_claims
        logger.info(f"  {year}: ingested {n_claims} claims (total: {total_claims})")

        # Save graph after each year for safety
        save_graph(kg, paths["graph"])

    save_graph(kg, paths["graph"])
    logger.info(f"\n  CHAIN-EXTRACTION SUMMARY for {label}")
    logger.info(f"    new papers extracted: {total_papers}")
    logger.info(f"    new claims extracted: {total_claims}")
    logger.info(f"    seen-pmids skipped:    {skipped_seen}")
    logger.info(f"    QC re-extracted:       {resampled}")

    return {
        "label": label,
        "total_papers": total_papers,
        "total_claims": total_claims,
        "skipped_seen": skipped_seen,
        "resampled": resampled,
        "qc_pmids": qc_pmids,
    }


def _load_seen_pmids(papers_csv: Path) -> set[str]:
    """Load pmids that have already produced claims (from papers_metadata.csv)."""
    if not papers_csv.exists():
        return set()
    seen: set[str] = set()
    with open(papers_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = (row.get("pmid") or "").strip()
            if p:
                seen.add(p)
    return seen


def _row_claim_count(row: dict) -> int:
    value = row.get("n_claims_extracted")
    if value in (None, ""):
        value = row.get("n_claims")
    try:
        return int(value or 0)
    except Exception:
        return 0


def _read_paper_rows(source_csv: Path) -> list[dict]:
    if not source_csv.exists():
        raise FileNotFoundError(f"no source CSV at {source_csv}")
    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


_SECOND_PASS_RESULT_CUE_RE = re.compile(
    r"\b(results?|findings?|conclusions?|show(?:ed|s)?|found|observed|"
    r"associated|correlated|predicted|distinguished|reduced|increased|"
    r"higher|lower|significant|p\s*[<=>])\b",
    re.I,
)


def _select_failed_pmids(source_csv: Path, *, max_papers: Optional[int] = None) -> list[str]:
    rows = _read_paper_rows(source_csv)
    pmids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid or pmid in seen:
            continue
        if (row.get("extraction_error") or "").strip():
            seen.add(pmid)
            pmids.append(pmid)
            if max_papers is not None and len(pmids) >= max_papers:
                break
    return pmids


def _select_second_pass_pmids(
    source_csv: Path,
    cache: AbstractCache,
    *,
    min_abstract_chars: int = 1000,
    require_result_cue: bool = True,
    max_papers: Optional[int] = None,
) -> list[str]:
    rows = _read_paper_rows(source_csv)
    pmids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid or pmid in seen:
            continue
        if (row.get("extraction_error") or "").strip():
            continue
        if _row_claim_count(row) != 0:
            continue

        rec = cache.get(pmid)
        if rec is None:
            continue
        abstract, _paper = rec
        if len(abstract) < min_abstract_chars:
            continue
        if require_result_cue and not _SECOND_PASS_RESULT_CUE_RE.search(abstract):
            continue

        seen.add(pmid)
        pmids.append(pmid)
        if max_papers is not None and len(pmids) >= max_papers:
            break
    return pmids


# ── Mode 2: rerun-cached ───────────────────────────────────────────────────────


def run_rerun_cached(
    *,
    max_papers: Optional[int] = None,
    pmids: Optional[Iterable[str]] = None,
    max_workers: int = 12,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
    label: str = "rerun_cached",
    lock_model: bool = False,
) -> dict:
    """Re-extract from the local abstract cache without hitting PubMed.

    Useful after prompt iteration: the LLM cost is paid again, but
    network latency / rate limits are gone.

    Args:
        pmids: optional explicit subset of pmids (overrides max_papers).
    """
    paths = _resolve_data_paths(data_dir)
    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    if len(cache) == 0:
        logger.warning("abstract cache is empty — nothing to rerun")
        return {"total_papers": 0, "total_claims": 0}

    kg = load_graph(paths["graph"])
    extractor = ClaimExtractor(lock_model=lock_model)
    _init_csv(paths["papers_csv"])

    # Build the (abstract, paper) iterator
    if pmids is not None:
        target = list(pmids)
        records = [(p, *cache.get(p)) for p in target if cache.get(p) is not None]
        # records is list of (pmid, abstract, paper)
    else:
        records = list(cache.iter_records())
        if max_papers is not None and max_papers > 0:
            records = records[:max_papers]

    logger.info(f"re-extracting {len(records):,} cached papers ({max_workers} workers)")

    BATCH = 200
    total_papers = 0
    total_claims = 0
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        papers_list = [(abstract, paper) for _, abstract, paper in chunk]
        results = extractor.extract_batch(papers_list, max_workers=max_workers)
        n_p, n_c = _ingest_results(
            kg, results, papers_list, paths,
            disease_label=label, year_label=0,
            keep_noise=keep_noise, strict_phase1=strict_phase1,
        )
        total_papers += n_p
        total_claims += n_c
        logger.info(f"  chunk {i//BATCH + 1}: {n_c} claims (running total: {total_claims})")
        # Safety save every 10 chunks
        if (i // BATCH) % 10 == 9:
            save_graph(kg, paths["graph"])

    save_graph(kg, paths["graph"])
    logger.info(f"\n  RERUN-CACHED SUMMARY")
    logger.info(f"    papers re-extracted: {total_papers}")
    logger.info(f"    claims extracted:    {total_claims}")
    return {"total_papers": total_papers, "total_claims": total_claims}


def run_retry_failed(
    *,
    source_csv: Path,
    max_papers: Optional[int] = None,
    max_workers: int = 1,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
    lock_model: bool = False,
) -> dict:
    """Re-extract papers whose prior metadata row has an extraction error."""
    pmids = _select_failed_pmids(source_csv, max_papers=max_papers)
    if not pmids:
        logger.info(f"no failed papers found in {source_csv}")
        return {"total_papers": 0, "total_claims": 0, "pmids": []}

    logger.info(f"retrying {len(pmids)} failed papers from {source_csv}")
    result = run_rerun_cached(
        pmids=pmids,
        max_workers=max_workers,
        data_dir=data_dir,
        keep_noise=keep_noise,
        strict_phase1=strict_phase1,
        label="retry_failed",
        lock_model=lock_model,
    )
    return {**result, "pmids": pmids}


def run_second_pass_zero(
    *,
    source_csv: Path,
    max_papers: Optional[int] = None,
    min_abstract_chars: int = 1000,
    require_result_cue: bool = True,
    max_workers: int = 1,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
    lock_model: bool = False,
) -> dict:
    """Second-pass audit of zero-claim papers likely to contain results."""
    paths = _resolve_data_paths(data_dir)
    cache = AbstractCache(default_cache_path(paths["data_dir"]))
    if len(cache) == 0:
        logger.warning("abstract cache is empty — nothing to audit")
        return {"total_papers": 0, "total_claims": 0, "pmids": []}

    pmids = _select_second_pass_pmids(
        source_csv,
        cache,
        min_abstract_chars=min_abstract_chars,
        require_result_cue=require_result_cue,
        max_papers=max_papers,
    )
    if not pmids:
        logger.info(f"no zero-claim second-pass candidates found in {source_csv}")
        return {"total_papers": 0, "total_claims": 0, "pmids": []}

    records = []
    for pmid in pmids:
        rec = cache.get(pmid)
        if rec is not None:
            abstract, paper = rec
            records.append((pmid, abstract, paper))

    kg = load_graph(paths["graph"])
    extractor = ClaimExtractor(lock_model=lock_model)
    _init_csv(paths["papers_csv"])
    logger.info(
        f"second-pass auditing {len(records)} zero-claim papers "
        f"(min_abstract_chars={min_abstract_chars}, "
        f"require_result_cue={require_result_cue}, workers={max_workers})"
    )

    total_papers = 0
    total_claims = 0
    BATCH = 200
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        papers_list = [(abstract, paper) for _, abstract, paper in chunk]
        results = extractor.extract_batch(
            papers_list,
            max_workers=max_workers,
            second_pass=True,
        )
        n_p, n_c = _ingest_results(
            kg, results, papers_list, paths,
            disease_label="second_pass_zero", year_label=0,
            keep_noise=keep_noise, strict_phase1=strict_phase1,
        )
        total_papers += n_p
        total_claims += n_c
        logger.info(
            f"  second-pass chunk {i//BATCH + 1}: {n_c} claims "
            f"(running total: {total_claims})"
        )

    save_graph(kg, paths["graph"])
    logger.info("\n  SECOND-PASS ZERO SUMMARY")
    logger.info(f"    papers audited:   {total_papers}")
    logger.info(f"    claims extracted: {total_claims}")
    return {"total_papers": total_papers, "total_claims": total_claims, "pmids": pmids}


# ── Mode 3: fill-sparse ────────────────────────────────────────────────────────


def run_fill_sparse(
    *,
    min_claims: int = 50,
    min_edges: int = 100,
    year_start: int = 2010,
    year_end: int = 2025,
    max_results_per_query: int = 200,
    terms_per_atom: int = 12,
    n_subqueries: int = 3,
    max_workers: int = 12,
    data_dir: Optional[Path] = None,
    keep_noise: bool = False,
    strict_phase1: bool = False,
    lock_model: bool = False,
) -> dict:
    """Detect sparse chains and run chain-aware extraction for each."""
    paths = _resolve_data_paths(data_dir)
    kg = load_graph(paths["graph"])
    coverage = analyse_coverage(kg)
    sparse = find_sparse(coverage, min_claims=min_claims, min_edges=min_edges)
    logger.info(f"found {len(sparse)} sparse chains/tasks (of {len(coverage)})")
    for c in sparse:
        logger.info(f"  sparse: {c.kind:<6} {c.name:<32} sig={c.signature}  "
                    f"edges={c.n_directed_edges}  claims={c.n_claims_supporting}")

    summary = {"sparse_count": len(sparse), "results": []}
    for c in sparse:
        res = run_chain_extraction(
            c.name,
            is_chain=(c.kind == "chain"),
            year_start=year_start, year_end=year_end,
            max_results_per_query=max_results_per_query,
            terms_per_atom=terms_per_atom,
            n_subqueries=n_subqueries,
            max_workers=max_workers,
            data_dir=data_dir,
            keep_noise=keep_noise,
            strict_phase1=strict_phase1,
            lock_model=lock_model,
        )
        summary["results"].append({"name": c.name, **res})
    return summary


# ── Mode 4: backfill-cache ─────────────────────────────────────────────────────


def run_backfill_cache(
    *,
    pmids_source_csv: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    batch_size: int = 200,
) -> dict:
    """Fetch abstracts for all pmids in papers_metadata.csv that aren't cached.

    Use this once after introducing the cache so subsequent --rerun-cached
    or chain-aware runs hit cache for the entire historical corpus.
    """
    paths = _resolve_data_paths(data_dir)
    csv_path = pmids_source_csv or paths["papers_csv"]
    if not csv_path.exists():
        logger.error(f"no source CSV at {csv_path}")
        return {"fetched": 0}
    cache = AbstractCache(default_cache_path(paths["data_dir"]))

    pmids: list[str] = []
    seen: set[str] = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = (row.get("pmid") or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            if p not in cache:
                pmids.append(p)
    logger.info(f"backfill: {len(pmids):,} pmids missing from cache")

    fetched = 0
    for i in range(0, len(pmids), batch_size):
        chunk = pmids[i:i + batch_size]
        # _fetch_pubmed_details writes through the cache
        before = len(cache)
        _fetch_pubmed_details(chunk, cache=cache)
        delta = len(cache) - before
        fetched += delta
        if (i // batch_size) % 10 == 9:
            logger.info(f"  backfill progress: {fetched:,} / {len(pmids):,}")
    logger.info(f"backfill complete: {fetched:,} new abstracts cached "
                f"(cache size: {len(cache):,})")
    return {"fetched": fetched}
