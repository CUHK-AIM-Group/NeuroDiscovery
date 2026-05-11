"""Phase 4.1: Literature Novelty Check.

Checks whether hypotheses have already been explored in published literature
using PubMed E-utilities and Semantic Scholar API.

Usage:
    python -m core.knowledge_graph.phase4 novelty --input data/hypotheses_refined.json
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .hypothesis_engine import Hypothesis

logger = logging.getLogger(__name__)

# PubMed E-utilities (no API key needed for basic usage)
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_RATE_LIMIT = 0.34  # 3 requests per second without API key

# Semantic Scholar — unauthenticated public API is strictly rate-limited
# (often 429s at even 1 req/s). We use a higher base delay and exponential
# backoff on 429. Set SEMANTIC_SCHOLAR_API_KEY env var for higher limits
# (https://www.semanticscholar.org/product/api#api-key-form).
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_RATE_LIMIT = 2.0  # seconds between requests (was 1.0)
SEMANTIC_SCHOLAR_MAX_RETRIES = 3


@dataclass
class NoveltyResult:
    """Novelty check result for a single hypothesis."""
    hypothesis_id: str
    pubmed_hits: int = 0
    semantic_hits: int = 0
    pubmed_novelty: float = 1.0
    semantic_novelty: float = 1.0
    lit_novelty: float = 1.0  # combined literature novelty
    final_novelty: float = 1.0  # combined with graph novelty
    top_papers: list[dict] = None

    def __post_init__(self):
        if self.top_papers is None:
            self.top_papers = []

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "pubmed_hits": self.pubmed_hits,
            "semantic_hits": self.semantic_hits,
            "pubmed_novelty": round(self.pubmed_novelty, 4),
            "semantic_novelty": round(self.semantic_novelty, 4),
            "lit_novelty": round(self.lit_novelty, 4),
            "final_novelty": round(self.final_novelty, 4),
            "top_papers": self.top_papers[:3],
        }


class NoveltyChecker:
    """Check hypothesis novelty against published literature."""

    def __init__(
        self,
        alpha: float = 0.5,
        use_pubmed: bool = True,
        use_semantic: bool = True,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            alpha: weight for graph novelty (1-alpha for literature novelty)
            use_pubmed: check PubMed
            use_semantic: check Semantic Scholar
            cache_path: path to cache file for hit counts
        """
        self.alpha = alpha
        self.use_pubmed = use_pubmed
        self.use_semantic = use_semantic
        self.cache: dict[str, dict] = {}
        self.cache_path = cache_path
        if cache_path:
            self._load_cache()

    def check(self, hypothesis: Hypothesis) -> NoveltyResult:
        """Check novelty for a single hypothesis."""
        query = self._build_query(hypothesis)
        result = NoveltyResult(hypothesis_id=hypothesis.id)

        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            result.pubmed_hits = cached.get("pubmed_hits", 0)
            result.semantic_hits = cached.get("semantic_hits", 0)
        else:
            if self.use_pubmed:
                result.pubmed_hits = self._search_pubmed(query)
                time.sleep(PUBMED_RATE_LIMIT)
            if self.use_semantic:
                result.semantic_hits = self._search_semantic(query)
                time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)
            # Cache the result
            self.cache[cache_key] = {
                "pubmed_hits": result.pubmed_hits,
                "semantic_hits": result.semantic_hits,
            }

        # Compute novelty scores
        result.pubmed_novelty = self._hit_count_to_novelty(result.pubmed_hits)
        result.semantic_novelty = self._hit_count_to_novelty(result.semantic_hits)

        # Combined literature novelty (average of available sources)
        scores = []
        if self.use_pubmed:
            scores.append(result.pubmed_novelty)
        if self.use_semantic:
            scores.append(result.semantic_novelty)
        result.lit_novelty = sum(scores) / max(len(scores), 1)

        # Final novelty: combine graph novelty with literature novelty
        graph_novelty = hypothesis.novelty_score
        result.final_novelty = self.alpha * graph_novelty + (1 - self.alpha) * result.lit_novelty

        return result

    def check_batch(self, hypotheses: list[Hypothesis]) -> list[NoveltyResult]:
        """Check novelty for multiple hypotheses."""
        results = []
        for i, h in enumerate(hypotheses):
            logger.info(f"checking novelty {i+1}/{len(hypotheses)}: {h.id} ({h.source_name} → {h.target_name})")
            try:
                result = self.check(h)
                results.append(result)
                logger.info(
                    f"  pubmed={result.pubmed_hits} semantic={result.semantic_hits} "
                    f"lit_novelty={result.lit_novelty:.2f} final={result.final_novelty:.2f}"
                )
            except Exception as e:
                logger.error(f"  failed: {e}")
                results.append(NoveltyResult(hypothesis_id=h.id))

        # Save cache
        if self.cache_path:
            self._save_cache()

        return results

    def _build_query(self, hypothesis: Hypothesis) -> str:
        """Build search query from hypothesis source and target."""
        source = hypothesis.source_name
        target = hypothesis.target_name
        return f'"{source}" AND "{target}"'

    def _search_pubmed(self, query: str) -> int:
        """Search PubMed and return hit count."""
        try:
            params = {
                "db": "pubmed",
                "term": query,
                "rettype": "json",
                "retmode": "json",
                "retmax": 0,  # only need count
            }
            resp = requests.get(PUBMED_ESEARCH, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("esearchresult", {}).get("count", 0))
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            return 0

    def _search_semantic(self, query: str) -> int:
        """Search Semantic Scholar and return hit count.

        Handles 429 (rate-limit) with exponential backoff. If a
        SEMANTIC_SCHOLAR_API_KEY env var is set, use it for higher limits.
        """
        import os
        headers = {}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if api_key:
            headers["x-api-key"] = api_key

        params = {
            "query": query,
            "limit": 1,
            "fields": "title,year,citationCount",
        }

        backoff = 2.0
        for attempt in range(SEMANTIC_SCHOLAR_MAX_RETRIES):
            try:
                resp = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params,
                                     headers=headers, timeout=15)
                if resp.status_code == 429:
                    logger.debug(f"Semantic Scholar 429, backing off {backoff:.1f}s (attempt {attempt+1})")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data.get("total", 0)
            except requests.exceptions.RequestException as e:
                logger.debug(f"Semantic Scholar request failed: {e}")
                time.sleep(backoff)
                backoff *= 2
            except Exception as e:
                logger.warning(f"Semantic Scholar search failed: {e}")
                return 0

        logger.warning(f"Semantic Scholar: exhausted retries for query '{query[:60]}'")
        return 0

    @staticmethod
    def _hit_count_to_novelty(hit_count: int) -> float:
        """Convert hit count to novelty score (0-1).

        Formula: novelty = 1 / (1 + log(hit_count + 1))
        - 0 hits → 1.0 (very novel)
        - 1-5 hits → 0.7-0.85 (emerging)
        - 10-50 hits → 0.5-0.65 (established)
        - 100+ hits → <0.4 (well-known)
        """
        return 1.0 / (1.0 + math.log(hit_count + 1))

    def _load_cache(self):
        """Load cache from file."""
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            logger.info(f"loaded {len(self.cache)} cached novelty results")
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"saved {len(self.cache)} novelty results to cache")
        except Exception as e:
            logger.warning(f"failed to save cache: {e}")
