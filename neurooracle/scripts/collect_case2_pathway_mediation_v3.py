"""Collect KG-v3 Case 2 literature without extracting claims or mutating the KG."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from neurooracle.src.case_targeted_extract import (
    CASE_STUDY_YEAR_END,
    CASE_STUDY_YEAR_START,
    DEFAULT_FORMAL_CLAIM_STORE,
    DEFAULT_PAPER_IDENTITY_INDEX,
    DEFAULT_STAGING_ROOTS,
    run_case_targeted_extraction,
)
from neurooracle.src.paper_identity import (
    CandidatePaperDeduplicator,
    GlobalPaperIdentityIndex,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO
    / "neurooracle"
    / "data"
    / "case_study_staging"
    / "kg_v3_case2_pathway_mediation_20260810"
)
DEFAULT_SOURCES = ("pubmed", "europepmc", "openalex")
DEFAULT_PRESETS = ("case2_pathway_mediation", "case2_supplemental_classic")
AVAILABLE_PRESETS = (*DEFAULT_PRESETS, "case2_high_recall_expansion")
DEFAULT_TARGET_TOTAL = 20_000


def _collection_paper_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        action="append",
        choices=("pubmed", "europepmc", "openalex", "arxiv", "biorxiv", "medrxiv"),
        help="Repeat to select sources; default: PubMed, Europe PMC, OpenAlex.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=AVAILABLE_PRESETS,
        help="Repeat to select query families; default: strict-chain plus classic seeds.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=DEFAULT_TARGET_TOTAL,
        help="Stop after this many unique abstract-ready candidates exist in the output.",
    )
    parser.add_argument("--target-per-source", type=int, default=10_000)
    parser.add_argument("--max-results-per-query", type=int, default=200)
    parser.add_argument("--identity-index", type=Path, default=DEFAULT_PAPER_IDENTITY_INDEX)
    parser.add_argument("--formal-claims", type=Path, default=DEFAULT_FORMAL_CLAIM_STORE)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = tuple(args.source or DEFAULT_SOURCES)
    presets = tuple(args.preset or DEFAULT_PRESETS)
    formal_claims = args.formal_claims.resolve()
    formal_before = {
        "size_bytes": formal_claims.stat().st_size,
        "mtime_ns": formal_claims.stat().st_mtime_ns,
    }

    started = datetime.now(timezone.utc)
    collection_csv = output / "collection_metadata.csv"
    existing_before = _collection_paper_count(collection_csv)
    index = GlobalPaperIdentityIndex(args.identity_index.resolve())
    try:
        staging_roots = [*DEFAULT_STAGING_ROOTS, output]
        index_stats = index.sync(
            formal_claim_store=formal_claims,
            staging_roots=staging_roots,
        )
        gate = CandidatePaperDeduplicator(index, output / "paper_dedup_audit.jsonl")
        runs = []
        for preset in presets:
            for source in sources:
                current_total = _collection_paper_count(collection_csv)
                remaining = max(0, args.target_total - current_total)
                if remaining == 0:
                    break
                run_target = min(args.target_per_source, remaining)
                logging.info("starting collect-only preset=%s source=%s", preset, source)
                runs.append(
                    run_case_targeted_extraction(
                        preset=preset,
                        year_start=CASE_STUDY_YEAR_START,
                        year_end=CASE_STUDY_YEAR_END,
                        target_papers=run_target,
                        max_results_per_query=args.max_results_per_query,
                        source=source,
                        data_dir=output,
                        collect_only=True,
                        formal_claim_store=formal_claims,
                        staging_roots=staging_roots,
                        identity_index_path=args.identity_index,
                        paper_identity_index=index,
                        deduplicator=gate,
                    )
                )
            if _collection_paper_count(collection_csv) >= args.target_total:
                break
        gate.flush()
    finally:
        index.close()

    formal_after = {
        "size_bytes": formal_claims.stat().st_size,
        "mtime_ns": formal_claims.stat().st_mtime_ns,
    }
    if formal_after != formal_before:
        raise RuntimeError("formal KG claim store changed during collect-only campaign")

    total_after = _collection_paper_count(collection_csv)
    summary = {
        "schema_version": "kg_v3_case2_collection_campaign.v1",
        "case_study_id": "case2_pathway_mediation",
        "year_start": CASE_STUDY_YEAR_START,
        "year_end": CASE_STUDY_YEAR_END,
        "sources": list(sources),
        "presets": list(presets),
        "target_total": args.target_total,
        "existing_candidates_before": existing_before,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "output": str(output),
        "kg_injection": False,
        "formal_claim_store_unchanged": True,
        "formal_claim_store_state": formal_after,
        "identity_index": index_stats,
        "runs": runs,
        "total_new_papers": sum(run["total_papers"] for run in runs),
        "total_unique_candidates_after": total_after,
        "target_reached": total_after >= args.target_total,
        "total_skipped_seen": sum(run["skipped_seen"] for run in runs),
    }
    (output / "CAMPAIGN_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
