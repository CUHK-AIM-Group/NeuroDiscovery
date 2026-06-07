"""Phase 2: LLM claim extraction + biomarker scan.

Extract structured scientific claims from PubMed literature, resolve entities,
and ingest them into the graph. Supports progressive upgrade: start from
abstract-only fast extraction, later enrich evidence with full text.

Subcommands:

    # 1) Main pipeline (disease x imaging x year)
    python -m neurooracle.phase2 --broad --max-workers 12

    # 2) chain-aware: search by atom-sequence terms to fill sparse chains
    python -m neurooracle.phase2 chain --chain genetic_imaging_disease \
        --year-start 2018 --year-end 2025 --max-results 200

    python -m neurooracle.phase2 chain --task biomarker_discovery --year-end 2025

    # 3) re-extract everything cached locally (no PubMed calls)
    python -m neurooracle.phase2 rerun-cached --max-papers 1000

    # 4) auto-detect sparse chains from KG and backfill them
    python -m neurooracle.phase2 fill-sparse --min-claims 50

    # 5) one-shot: fetch abstracts for pmids in papers_metadata.csv
    #    that are not yet in the abstract cache
    python -m neurooracle.phase2 backfill-cache

    # 6) coverage report (KG → JSON of per-chain density)
    python -m neurooracle.phase2 coverage --graph .../knowledge_graph.json \
        --out coverage.json

    # 7) biomarker mention scanner
    python -m neurooracle.phase2 biomarker-scan \
        --graph neurooracle/data/full_snapshot_v2/knowledge_graph.json \
        --claims neurooracle/data/full_snapshot_v2/extracted_claims.jsonl \
        --output neurooracle/data/full_snapshot_v2/biomarker_mentions.json \
        --mode local

Default Phase-2 workspace:
    neurooracle/data/full_snapshot_v2/

Override with ``--data-dir`` when you want an isolated scratch run.
"""

import argparse
import logging
import os
import sys
from pathlib import Path


def _load_env_keys() -> None:
    """Load repo-local .env.keys into os.environ if present.

    The file currently uses shell-style lines such as:
        export OPENAI_API_KEYS="k1,k2"
        export OPENAI_BASE_URL=https://...

    We intentionally keep parsing minimal here and only fill variables that
    are not already present in the process environment.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env.keys"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


_load_env_keys()

from .src.batch_extract import main as batch_extract_main
from .src.biomarker_scan import main as biomarker_scan_main
from .src.chain_coverage import main as coverage_main
from .src.chain_extract import (
    run_chain_extraction, run_rerun_cached,
    run_fill_sparse, run_backfill_cache,
)


def _cmd_chain():
    p = argparse.ArgumentParser(description="Chain/task-aware Phase-2 extraction")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--chain", type=str, help="TaskChain name (e.g. genetic_imaging_disease)")
    g.add_argument("--task", type=str, help="Task name (e.g. biomarker_discovery)")
    p.add_argument("--year-start", type=int, default=2010)
    p.add_argument("--year-end", type=int, default=2025)
    p.add_argument("--max-results", type=int, default=200,
                   help="Max PubMed results per compound query (default 200)")
    p.add_argument("--terms-per-atom", type=int, default=12)
    p.add_argument("--n-subqueries", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--keep-noise", action="store_true")
    p.add_argument("--strict-phase1", action="store_true")
    p.add_argument("--lock-model", action="store_true",
                   help="Disable adaptive model upgrade/downgrade and keep "
                        "the extractor pinned to OPENAI_MODEL for this run.")
    p.add_argument("--qc-rate", type=float, default=0.05,
                   help="Fraction of already-seen pmids to re-extract for QC (default 0.05)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(sys.argv[2:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    name = args.chain or args.task
    is_chain = bool(args.chain)
    run_chain_extraction(
        name, is_chain=is_chain,
        year_start=args.year_start, year_end=args.year_end,
        max_results_per_query=args.max_results,
        terms_per_atom=args.terms_per_atom,
        n_subqueries=args.n_subqueries,
        max_workers=args.max_workers,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        keep_noise=args.keep_noise,
        strict_phase1=args.strict_phase1,
        sample_rate_seen=args.qc_rate,
        lock_model=args.lock_model,
    )


def _cmd_rerun_cached():
    p = argparse.ArgumentParser(description="Re-extract from local abstract cache")
    p.add_argument("--max-papers", type=int, default=None,
                   help="Cap the number of cached papers to re-extract")
    p.add_argument("--pmids", nargs="+", default=None,
                   help="Explicit pmid list (overrides --max-papers)")
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--keep-noise", action="store_true")
    p.add_argument("--strict-phase1", action="store_true")
    p.add_argument("--lock-model", action="store_true",
                   help="Disable adaptive model upgrade/downgrade and keep "
                        "the extractor pinned to OPENAI_MODEL for this run.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(sys.argv[2:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_rerun_cached(
        max_papers=args.max_papers,
        pmids=args.pmids,
        max_workers=args.max_workers,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        keep_noise=args.keep_noise,
        strict_phase1=args.strict_phase1,
        lock_model=args.lock_model,
    )


def _cmd_fill_sparse():
    p = argparse.ArgumentParser(description="Detect + backfill sparse chains")
    p.add_argument("--min-claims", type=int, default=50)
    p.add_argument("--min-edges", type=int, default=100)
    p.add_argument("--year-start", type=int, default=2010)
    p.add_argument("--year-end", type=int, default=2025)
    p.add_argument("--max-results", type=int, default=200)
    p.add_argument("--terms-per-atom", type=int, default=12)
    p.add_argument("--n-subqueries", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--keep-noise", action="store_true")
    p.add_argument("--strict-phase1", action="store_true")
    p.add_argument("--lock-model", action="store_true",
                   help="Disable adaptive model upgrade/downgrade and keep "
                        "the extractor pinned to OPENAI_MODEL for this run.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(sys.argv[2:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_fill_sparse(
        min_claims=args.min_claims, min_edges=args.min_edges,
        year_start=args.year_start, year_end=args.year_end,
        max_results_per_query=args.max_results,
        terms_per_atom=args.terms_per_atom,
        n_subqueries=args.n_subqueries,
        max_workers=args.max_workers,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        keep_noise=args.keep_noise,
        strict_phase1=args.strict_phase1,
        lock_model=args.lock_model,
    )


def _cmd_backfill_cache():
    p = argparse.ArgumentParser(description="Backfill abstract cache from papers_metadata.csv")
    p.add_argument("--source", type=str, default=None,
                   help="Source CSV (default: <data_dir>/papers_metadata.csv)")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(sys.argv[2:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_backfill_cache(
        pmids_source_csv=Path(args.source) if args.source else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        batch_size=args.batch_size,
    )


_SUBCOMMANDS = {
    "biomarker-scan":  biomarker_scan_main,
    "coverage":        coverage_main,
    "chain":           _cmd_chain,
    "rerun-cached":    _cmd_rerun_cached,
    "fill-sparse":     _cmd_fill_sparse,
    "backfill-cache":  _cmd_backfill_cache,
}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        cmd = sys.argv[1]
        # biomarker-scan and coverage parse their own argv, leave argv intact
        # but drop the subcommand token so their argparse sees clean argv.
        if cmd in ("biomarker-scan", "coverage"):
            sys.argv.pop(1)
        _SUBCOMMANDS[cmd]()
    else:
        # default: original disease × year batch extraction
        batch_extract_main()
