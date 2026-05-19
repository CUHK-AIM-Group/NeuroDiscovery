"""Phase 2.6: Biomarker mention scanner.

Scans papers for mentions of biomarker nodes in the knowledge graph.
Two modes:

    local    — scan already-extracted papers (extracted_claims.jsonl +
               papers_metadata.csv). Cheap, no network calls.

    pubmed   — query PubMed esearch once per biomarker to count paper mentions
               beyond our extracted pool. Slow, rate-limited, optional.

Output: `biomarker_mentions.json` mapping
    {
        biomarker_id: {
            "name": ...,
            "aliases": [...],
            "local": {
                "n_papers": N,
                "n_claims": M,
                "pmids": [sorted list of PMIDs],
                "sample_contexts": [up to 3 raw_text excerpts],
            },
            "pubmed": {              # only when --mode=pubmed
                "query": ...,
                "hit_count": ...,
            },
        },
        ...
    }

This annotation layer does NOT modify the KG or claim extraction. It's a
queryable "which papers talk about biomarker X" index for downstream phases.

Usage:
    python -m neurooracle.phase2 biomarker-scan \
        --graph neurooracle/data/knowledge_graph.json \
        --claims neurooracle/data/extracted_claims.jsonl \
        --output neurooracle/data/biomarker_mentions.json \
        --mode local

    # Extend with PubMed counts (rate-limited, ~0.3-0.5s per biomarker)
    python -m neurooracle.phase2 biomarker-scan \
        --graph neurooracle/data/knowledge_graph.json \
        --claims neurooracle/data/extracted_claims.jsonl \
        --output neurooracle/data/biomarker_mentions.json \
        --mode pubmed --limit-biomarkers 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .graph_manager import KnowledgeGraph
from .schema import ConceptNode
from .storage import load_graph

logger = logging.getLogger(__name__)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_RATE_LIMIT = 0.34  # 3 req/sec without key
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

# Minimum length for a name/alias to be used as a search term. Short tokens
# like "IC", "AD", "CA" produce too many false positives.
_MIN_NAME_LEN = 4

# Biomarker node domain tags. Any concept tagged with these is a scan target.
BIOMARKER_DOMAINS = {"biomarker", "imaging_feature"}

# Default neuroanatomy inclusion: brain regions function as biomarker proxies
# too (e.g. "hippocampal atrophy" — hippocampus is the measurable location).
INCLUDE_NEUROANATOMY_BY_DEFAULT = True


@dataclass
class BiomarkerMentionRecord:
    biomarker_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    # local mode results
    n_papers: int = 0
    n_claims: int = 0
    pmids: list[str] = field(default_factory=list)
    sample_contexts: list[str] = field(default_factory=list)
    # pubmed mode results (optional)
    pubmed_query: str = ""
    pubmed_hit_count: Optional[int] = None

    def to_dict(self) -> dict:
        out = {
            "biomarker_id": self.biomarker_id,
            "name": self.name,
            "aliases": self.aliases,
            "domain_tags": self.domain_tags,
            "local": {
                "n_papers": self.n_papers,
                "n_claims": self.n_claims,
                "pmids": self.pmids,
                "sample_contexts": self.sample_contexts,
            },
        }
        if self.pubmed_hit_count is not None:
            out["pubmed"] = {
                "query": self.pubmed_query,
                "hit_count": self.pubmed_hit_count,
            }
        return out


# ── Term selection ──────────────────────────────────────────────────────

def _collect_search_terms(node: ConceptNode) -> list[str]:
    """Return the list of surface terms to search for this biomarker.

    Includes preferred_name and sufficiently long aliases. Short tokens
    (< _MIN_NAME_LEN chars) are excluded to avoid phantom matches.
    """
    terms: list[str] = []
    if node.preferred_name and len(node.preferred_name) >= _MIN_NAME_LEN:
        terms.append(node.preferred_name)
    for alias in node.aliases or []:
        if not alias or len(alias) < _MIN_NAME_LEN:
            continue
        if alias.lower() == (node.preferred_name or "").lower():
            continue
        terms.append(alias)
    # Dedupe, preserve order
    seen = set()
    out = []
    for t in terms:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _select_biomarker_targets(
    kg: KnowledgeGraph,
    include_neuroanatomy: bool = INCLUDE_NEUROANATOMY_BY_DEFAULT,
) -> list[ConceptNode]:
    """Pick all KG nodes that should be scanned for paper mentions."""
    targets: list[ConceptNode] = []
    for node in kg._index.values():
        if "claim" in node.domain_tags:
            continue
        # Must have at least one usable search term
        if not _collect_search_terms(node):
            continue

        tags = set(node.domain_tags)
        if tags & BIOMARKER_DOMAINS:
            targets.append(node)
        elif include_neuroanatomy and "neuroanatomy" in tags:
            targets.append(node)
    return targets


def _compile_aho_corasick(term_to_ids: dict[str, set[str]]):
    """Build an Aho-Corasick automaton for all biomarker terms.

    Terms are matched case-insensitively (we lowercase everything at scan time).
    The automaton returns the original lowercase term + attached biomarker IDs.

    Returns the automaton (already finalized) and a set of "word-boundary safe"
    chars for post-filtering partial matches.
    """
    import ahocorasick  # lazy import
    A = ahocorasick.Automaton()
    for term, ids in term_to_ids.items():
        key = term.lower().strip()
        if not key:
            continue
        # add_word overwrites prior payload for the same key. Merge ids first.
        existing = A.get(key) if key in A else None
        merged = set(ids)
        if existing is not None:
            _, prev_ids = existing
            merged |= set(prev_ids)
        A.add_word(key, (key, tuple(sorted(merged))))
    A.make_automaton()
    return A


_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9_]")


def _is_word_boundary_match(haystack: str, start: int, end: int) -> bool:
    """True if [start, end) is surrounded by non-word chars (or string edge)."""
    if start > 0 and _WORD_CHAR_RE.match(haystack[start - 1]):
        return False
    if end < len(haystack) and _WORD_CHAR_RE.match(haystack[end]):
        return False
    return True


# ── Local mode ──────────────────────────────────────────────────────────

def _iter_claims(claims_path: Path) -> Iterable[dict]:
    """Yield claim dicts one at a time from a JSONL file."""
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed jsonl line")


def scan_local(
    kg: KnowledgeGraph,
    claims_path: Path,
    include_neuroanatomy: bool = INCLUDE_NEUROANATOMY_BY_DEFAULT,
    max_contexts_per_biomarker: int = 3,
    context_chars: int = 240,
) -> list[BiomarkerMentionRecord]:
    """Scan extracted_claims.jsonl for biomarker mentions.

    For each biomarker node, counts how many unique papers mention any of
    its terms in either the paper title or the claim raw_text.

    Uses Aho-Corasick automaton for O(text_len + n_matches) scanning, which
    handles thousands of biomarker terms × tens of thousands of claims in
    seconds rather than hours.
    """
    targets = _select_biomarker_targets(kg, include_neuroanatomy)
    logger.info(f"biomarker-scan local: {len(targets)} biomarker targets")

    term_to_ids: dict[str, set[str]] = {}
    records: dict[str, BiomarkerMentionRecord] = {}
    pmid_sets: dict[str, set[str]] = {}
    for node in targets:
        terms = _collect_search_terms(node)
        if not terms:
            continue
        for t in terms:
            term_to_ids.setdefault(t, set()).add(node.id)
        records[node.id] = BiomarkerMentionRecord(
            biomarker_id=node.id,
            name=node.preferred_name,
            aliases=[t for t in node.aliases if t],
            domain_tags=list(node.domain_tags),
        )
        pmid_sets[node.id] = set()

    logger.info(f"biomarker-scan local: {len(term_to_ids)} unique search terms")
    automaton = _compile_aho_corasick(term_to_ids)
    logger.info("biomarker-scan local: Aho-Corasick automaton built")

    n_claims_seen = 0
    for claim in _iter_claims(claims_path):
        n_claims_seen += 1
        if n_claims_seen % 5000 == 0:
            logger.info(f"  scanned {n_claims_seen:,} claims...")

        raw_text = (claim.get("raw_text") or "").strip()
        src = claim.get("source_paper") or {}
        title = (src.get("title") or "").strip()
        pmid = str(src.get("pmid") or "").strip()

        haystack = " ".join(filter(None, [title, raw_text]))
        if not haystack:
            continue

        haystack_lower = haystack.lower()

        # Collect which biomarker ids matched in this claim (dedupe per claim)
        matched_ids: set[str] = set()
        for end_index, (term, ids) in automaton.iter(haystack_lower):
            start_index = end_index - len(term) + 1
            # Enforce word-boundary semantics post hoc
            if not _is_word_boundary_match(haystack_lower, start_index, end_index + 1):
                continue
            matched_ids.update(ids)

        if not matched_ids:
            continue

        for bid in matched_ids:
            rec = records[bid]
            rec.n_claims += 1
            if pmid and pmid not in pmid_sets[bid]:
                pmid_sets[bid].add(pmid)
                rec.n_papers += 1
            if len(rec.sample_contexts) < max_contexts_per_biomarker and raw_text:
                ctx = raw_text[:context_chars]
                if len(raw_text) > context_chars:
                    ctx += "..."
                rec.sample_contexts.append(ctx)

    # Finalize pmids list (sorted)
    for bid, rec in records.items():
        rec.pmids = sorted(pmid_sets[bid])

    logger.info(f"biomarker-scan local: scanned {n_claims_seen:,} claims")
    return list(records.values())


# ── PubMed mode ────────────────────────────────────────────────────────

def _pubmed_count(query: str, timeout: int = 20) -> Optional[int]:
    """Call PubMed esearch with retmax=0 to get hit count. Respects rate limit."""
    import requests  # lazy
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        resp = requests.get(PUBMED_ESEARCH, params=params, timeout=timeout)
        if resp.status_code in (429, 502, 503):
            time.sleep(2.0)
            return None
        resp.raise_for_status()
        return int(resp.json().get("esearchresult", {}).get("count", 0))
    except Exception as exc:
        logger.warning(f"pubmed count failed for {query!r}: {exc}")
        return None


def scan_pubmed(
    records: list[BiomarkerMentionRecord],
    limit_biomarkers: int = 0,
    sleep_s: float = PUBMED_RATE_LIMIT,
) -> list[BiomarkerMentionRecord]:
    """Annotate each record with its PubMed hit count.

    Mutates records in place. Skips biomarkers with no usable name.
    """
    todo = records if not limit_biomarkers else records[:limit_biomarkers]
    logger.info(f"biomarker-scan pubmed: querying {len(todo)} biomarkers")
    for i, rec in enumerate(todo, 1):
        # Build query: preferred name only (safer than OR-ing all aliases)
        if not rec.name or len(rec.name) < _MIN_NAME_LEN:
            continue
        query = f'"{rec.name}"[Title/Abstract]'
        count = _pubmed_count(query)
        rec.pubmed_query = query
        rec.pubmed_hit_count = count
        if i % 20 == 0:
            logger.info(f"  [{i}/{len(todo)}] pubmed hit_count median so far...")
        time.sleep(sleep_s)
    return todo


# ── Main entry ─────────────────────────────────────────────────────────

def run_biomarker_scan(
    graph_path: Path,
    claims_path: Path,
    output_path: Path,
    mode: str = "local",
    include_neuroanatomy: bool = INCLUDE_NEUROANATOMY_BY_DEFAULT,
    limit_biomarkers: int = 0,
) -> dict:
    """Top-level runner. Returns summary dict."""
    kg = load_graph(graph_path)
    records = scan_local(
        kg, claims_path,
        include_neuroanatomy=include_neuroanatomy,
    )

    if mode == "pubmed":
        # Sort by local n_papers desc so we prioritize informative biomarkers
        records_sorted = sorted(records, key=lambda r: r.n_papers, reverse=True)
        scan_pubmed(records_sorted, limit_biomarkers=limit_biomarkers)
        records = records_sorted

    # Emit in dict form keyed by id
    payload = {
        "version": 1,
        "mode": mode,
        "n_biomarkers": len(records),
        "n_with_hits": sum(1 for r in records if r.n_papers > 0),
        "biomarkers": {r.biomarker_id: r.to_dict() for r in records},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"wrote {output_path}: {payload['n_biomarkers']} biomarkers, "
                f"{payload['n_with_hits']} with ≥1 paper hit")

    summary = {
        "n_biomarkers": payload["n_biomarkers"],
        "n_with_hits": payload["n_with_hits"],
        "top_by_papers": [
            {"id": r.biomarker_id, "name": r.name, "n_papers": r.n_papers}
            for r in sorted(records, key=lambda x: x.n_papers, reverse=True)[:15]
        ],
        "output": str(output_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 2.6 biomarker mention scanner")
    parser.add_argument("--graph", required=True, help="Path to knowledge_graph.json")
    parser.add_argument("--claims", required=True, help="Path to extracted_claims.jsonl")
    parser.add_argument("--output", required=True, help="Output biomarker_mentions.json")
    parser.add_argument("--mode", choices=["local", "pubmed"], default="local",
                        help="local = scan existing papers only; pubmed = also query PubMed")
    parser.add_argument("--no-neuroanatomy", action="store_true",
                        help="Skip neuroanatomy nodes (scan only biomarker/imaging_feature)")
    parser.add_argument("--limit-biomarkers", type=int, default=0,
                        help="In pubmed mode, only query top-N biomarkers by local n_papers")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    summary = run_biomarker_scan(
        graph_path=Path(args.graph),
        claims_path=Path(args.claims),
        output_path=Path(args.output),
        mode=args.mode,
        include_neuroanatomy=not args.no_neuroanatomy,
        limit_biomarkers=args.limit_biomarkers,
    )

    print(f"\nBiomarker scan results:")
    print(f"  Biomarkers scanned: {summary['n_biomarkers']}")
    print(f"  Biomarkers with ≥1 paper hit: {summary['n_with_hits']}")
    print(f"  Output: {summary['output']}")
    print(f"\nTop 15 biomarkers by paper coverage:")
    for i, r in enumerate(summary["top_by_papers"], 1):
        print(f"  #{i:2d}  n={r['n_papers']:5d}  [{r['id']}] {r['name']}")


if __name__ == "__main__":
    main()
