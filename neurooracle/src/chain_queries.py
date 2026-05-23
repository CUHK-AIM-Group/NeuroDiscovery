"""Chain-aware PubMed query builder.

Phase 2's original query was disease + imaging-modality, which only covers
two of the seven atom positions and reliably hits chains anchored on
``DISEASE`` × ``IMAGING_MARKER``. Sparser chains — ``GENE_TARGET → IM →
DISEASE``, ``DRUG → IM → OUTCOME``, ``COGNITIVE_TASK → IM → INDIVIDUAL_DATA``
— surface only by accident.

This module pulls top-K names from the existing KG for each atom in a
``Task`` or ``TaskChain`` and assembles a compound PubMed query that
forces the abstract to mention something from each atom's vocabulary. The
"top-K by claim mentions" ranking biases toward concepts the literature
already discusses heavily — reproducing existing density, then expanding
into longer-tail terms with multiple sub-queries.

Ranking signal: the number of curated edges adjacent to the concept (or
"about" edges from claims) approximates how often the literature mentions
it. If those signals are unavailable, fall back to alphabetical sort by
preferred_name.

Usage::

    from neurooracle.src.atoms import chain_by_name
    from neurooracle.src.chain_queries import build_chain_queries

    chain = chain_by_name("genetic_imaging_disease")
    for q in build_chain_queries(chain, kg, year=2024,
                                 terms_per_atom=8, n_subqueries=4):
        pmids = _search_pubmed(q, max_results=200)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

from .atoms import Atom, Task, TaskChain, ATOM_TO_DOMAINS
from .graph_manager import KnowledgeGraph
from .schema import ConceptNode

logger = logging.getLogger(__name__)


# ── Term selection from KG ─────────────────────────────────────────────────────

# Curated-vocab prefixes we trust to give clean, MeSH-quality names.
_PREFERRED_PREFIXES = (
    "MSH:",         # MeSH descriptors — best PubMed mapping
    "HGNC:",        # gene symbols
    "DISGENET:",
    "ATC:",         # drug ATC codes (we want the name not the code)
    "COGAT_TASK:", "COGAT_CONCEPT:", "COGAT_DISORDER:",
    "NN:",          # NeuroNames (anatomy)
    "BM_REGION:", "BM_PARADIGM:", "BM_EXP:",
    "NCBI_Gene:",
)


def _name_quality(name: str) -> int:
    """Higher = cleaner search term. Long acronyms with weird punctuation lose."""
    if not name:
        return -100
    score = 0
    n = name.strip()
    if 3 <= len(n) <= 50:
        score += 5
    if re.match(r"^[A-Za-z][A-Za-z0-9 \-,/']*$", n):
        score += 5
    if re.search(r"\d{3,}", n):
        # numeric ID smells (e.g. "T1234")
        score -= 3
    if n.isupper() and len(n) > 6:
        score -= 1
    return score


def _is_curated(node_id: str) -> bool:
    return any(node_id.startswith(p) for p in _PREFERRED_PREFIXES)


def kg_top_terms_for_atom(
    kg: KnowledgeGraph,
    atom: Atom,
    k: int = 12,
    min_degree: int = 2,
) -> list[str]:
    """Top-K canonical names from the KG that can serve as ``atom``.

    Ranking: degree in the KG (a rough proxy for how often the concept
    co-occurs with others in the literature, since edges are claim-derived)
    × name-quality bonus. Curated-vocab nodes preferred over auto-minted
    ``CLM_CONCEPT:`` shells.
    """
    domains = ATOM_TO_DOMAINS[atom]
    candidates: list[tuple[float, str]] = []
    for cid, node in kg._index.items():
        if not isinstance(node, ConceptNode):
            continue
        if not any(d in domains for d in node.domain_tags):
            continue
        # Skip claim shells (CLM:...) — they're claim *records*, not concepts
        if cid.startswith("CLM:") or cid.startswith("CLM_CONCEPT:"):
            # Allow CLM_CONCEPT only as last-resort fallback (low priority).
            if not cid.startswith("CLM_CONCEPT:"):
                continue
        # Degree = how many claims/edges touch it
        try:
            deg = kg.G.degree(cid)
        except Exception:
            deg = 0
        if deg < min_degree:
            continue
        name = (node.preferred_name or "").strip()
        if not name:
            continue
        nq = _name_quality(name)
        if nq < 0:
            continue
        curated_bonus = 10 if _is_curated(cid) else 0
        score = math.log1p(deg) * 5 + nq + curated_bonus
        candidates.append((score, name))

    # Dedupe by lowercase name, keep highest-scoring variant
    by_name: dict[str, float] = {}
    for s, n in candidates:
        key = n.lower()
        if key not in by_name or s > by_name[key]:
            by_name[key] = s
    ranked = sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)

    return [name for name, _ in ranked[:k]]


# ── Query assembly ─────────────────────────────────────────────────────────────


def _quote(term: str) -> str:
    """Wrap multi-word terms in quotes; escape internal quotes."""
    t = term.replace('"', '').strip()
    if " " in t or "/" in t or "," in t:
        return f'"{t}"[Title/Abstract]'
    return f'{t}[Title/Abstract]'


def _or_block(terms: Iterable[str]) -> str:
    parts = [_quote(t) for t in terms if t.strip()]
    if not parts:
        return ""
    return "(" + " OR ".join(parts) + ")"


def _and_blocks(blocks: Iterable[str]) -> str:
    parts = [b for b in blocks if b]
    if not parts:
        return ""
    return " AND ".join(parts)


def _year_clause(year: Optional[int], year_range: Optional[tuple[int, int]] = None) -> str:
    if year_range is not None:
        return f"{year_range[0]}:{year_range[1]}[pdat]"
    if year is not None:
        return f"{year}:{year}[pdat]"
    return ""


def build_chain_query(
    atom_terms: dict[Atom, list[str]],
    chain: tuple[Atom, ...],
    year: Optional[int] = None,
    year_range: Optional[tuple[int, int]] = None,
) -> str:
    """Build a single compound PubMed query forcing each chain atom present.

    Example with chain (G, IM, D)::

        (APOE OR BDNF OR ...)[T/A]
        AND (hippocampus OR amygdala OR ...)[T/A]
        AND (Alzheimer OR depression OR ...)[T/A]
        AND 2024:2024[pdat]

    Returns "" when any atom has no terms (caller should skip such chains).
    """
    blocks: list[str] = []
    for atom in chain:
        terms = atom_terms.get(atom, [])
        block = _or_block(terms)
        if not block:
            return ""
        blocks.append(block)
    yc = _year_clause(year, year_range)
    if yc:
        blocks.append(yc)
    return _and_blocks(blocks)


def split_terms(terms: list[str], n_groups: int) -> list[list[str]]:
    """Round-robin split a term list into n groups for query diversification."""
    if n_groups <= 1 or len(terms) <= n_groups:
        return [terms]
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, t in enumerate(terms):
        groups[i % n_groups].append(t)
    return groups


def build_chain_queries(
    chain: TaskChain,
    kg: KnowledgeGraph,
    year: Optional[int] = None,
    year_range: Optional[tuple[int, int]] = None,
    terms_per_atom: int = 12,
    n_subqueries: int = 1,
) -> list[str]:
    """Build N PubMed queries for a TaskChain by partitioning the term pool.

    Diversification: by default a single query unions the top terms for
    each atom. Higher ``n_subqueries`` round-robin-splits the seed atom's
    term pool into N buckets, producing N narrower queries that together
    surface a wider set of papers without one mega-query.

    Returns an empty list if any atom in the chain has no KG terms
    (cannot form a meaningful query).
    """
    atom_terms = {
        a: kg_top_terms_for_atom(kg, a, k=terms_per_atom)
        for a in set(chain.chain)
    }
    for a, terms in atom_terms.items():
        logger.debug(f"chain {chain.name}: atom {a.value} → {len(terms)} terms")
        if not terms:
            logger.warning(
                f"chain {chain.name}: atom {a.value} has 0 KG terms — skipping"
            )
            return []

    # Pick the atom with the largest pool to split for diversification.
    seed_atom = max(atom_terms, key=lambda a: len(atom_terms[a]))
    seed_groups = split_terms(atom_terms[seed_atom], n_subqueries)

    queries: list[str] = []
    for group in seed_groups:
        per_atom = dict(atom_terms)
        per_atom[seed_atom] = group
        q = build_chain_query(per_atom, chain.chain, year=year, year_range=year_range)
        if q:
            queries.append(q)
    return queries


def build_task_queries(
    task: Task,
    kg: KnowledgeGraph,
    year: Optional[int] = None,
    year_range: Optional[tuple[int, int]] = None,
    terms_per_atom: int = 12,
    n_subqueries: int = 1,
) -> list[str]:
    """Build N PubMed queries for a flat Task. Inputs are AND'd unordered."""
    chain_atoms = tuple(task.inputs) + (task.output,)
    atom_terms = {
        a: kg_top_terms_for_atom(kg, a, k=terms_per_atom)
        for a in set(chain_atoms)
    }
    for a, terms in atom_terms.items():
        if not terms:
            logger.warning(f"task {task.name}: atom {a.value} has 0 KG terms — skipping")
            return []
    seed_atom = max(atom_terms, key=lambda a: len(atom_terms[a]))
    seed_groups = split_terms(atom_terms[seed_atom], n_subqueries)
    queries: list[str] = []
    for group in seed_groups:
        per_atom = dict(atom_terms)
        per_atom[seed_atom] = group
        q = build_chain_query(per_atom, chain_atoms, year=year, year_range=year_range)
        if q:
            queries.append(q)
    return queries


__all__ = [
    "kg_top_terms_for_atom",
    "build_chain_query",
    "build_chain_queries",
    "build_task_queries",
    "split_terms",
]
