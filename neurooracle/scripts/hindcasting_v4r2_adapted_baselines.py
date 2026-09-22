"""Leakage-safe method-family adapters for Hindcasting v4r2.

The original v4r1 baselines were deliberately lightweight, but that made the
SciAgents proxy task-blind and reduced the OpenScholar proxy to a single
endpoint-evidence sort.  This module keeps the historical discovery boundary
unchanged while restoring the defining *algorithmic* ideas that can be
implemented without a contemporary language model:

* ``sciagents_adapted`` uses a task-scoped graph, explicit ontologist /
  scientist / critic score components, graph-path grounding, and a seeded
  diversity policy.
* ``openscholar_rag_adapted`` performs query-conditioned retrieval, a
  deterministic pseudo-relevance-feedback pass, re-retrieval, citation-aware
  synthesis, and a fixed hypothesis compiler.

These are method-family adaptations, not claims of byte-for-byte execution of
the upstream systems.  No post-cutoff publication corpus, evaluator symbol,
or contemporary model call is used here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
import re
import sys
import types
from typing import Any, Iterable, Mapping, Sequence

# The legacy generator imports three CLI-only window helpers from
# run_case_study_hindcasting.  Importing that CLI module would also import the
# retrospective evaluator.  Install a minimal, explicit compatibility module
# before importing the generator so discovery cannot cross that boundary.
_WINDOW_MODULE = "neurooracle.scripts.run_case_study_hindcasting"
_FORBIDDEN_EVALUATOR_MODULE = "neurooracle.scripts.case_study_hindcasting_eval"
_EVALUATOR_PRELOADED = _FORBIDDEN_EVALUATOR_MODULE in sys.modules


@dataclass(frozen=True)
class _DiscoveryWindow:
    freeze_year: int
    future_start_year: int
    future_end_year: int

    @property
    def label(self) -> str:
        return (
            f"kg{self.freeze_year}_to_{self.future_start_year}_"
            f"{self.future_end_year}"
        )


def _parse_discovery_window(raw: str) -> _DiscoveryWindow:
    parts = str(raw).replace(",", ":").split(":")
    if len(parts) != 3:
        raise ValueError("window must be freeze:start:end")
    freeze, start, end = (int(part) for part in parts)
    if not (freeze < start <= end):
        raise ValueError("window must satisfy freeze < start <= end")
    return _DiscoveryWindow(freeze, start, end)


if _WINDOW_MODULE not in sys.modules:
    shim = types.ModuleType(_WINDOW_MODULE)
    shim.Window = _DiscoveryWindow
    shim.DEFAULT_WINDOWS = tuple(
        _DiscoveryWindow(year, year + 1, year + 5) for year in range(2016, 2021)
    )
    shim.parse_window = _parse_discovery_window
    shim.__dict__["__hindcasting_discovery_shim__"] = True
    sys.modules[_WINDOW_MODULE] = shim
_NON_ISOLATED_WINDOW_MODULE_PRELOADED = not getattr(
    sys.modules[_WINDOW_MODULE], "__hindcasting_discovery_shim__", False
)

from neurooracle.scripts.generate_case_study_frozen_baselines import (
    FrozenEdge,
    FrozenGraphIndex,
    _merge_adjacencies,
    generate_case_hypotheses,
)
from neurooracle.src.case_studies import CaseStudy


ADAPTER_VERSION = "hindcasting-v4r2-adapted-baselines.v1"
METHODS = ("sciagents_adapted", "openscholar_rag_adapted")
POOL_SIZE = 2500
MAX_RANK = 1000


TASK_PROFILES: dict[str, dict[str, Any]] = {
    "case1_transdiagnostic": {
        "mode": "transdiagnostic_breadth",
        "query_terms": (
            "brain", "imaging", "marker", "transdiagnostic", "cross", "disorder",
            "shared", "psychiatric", "neurological", "disease", "subtype",
            "cluster", "profile", "network", "structural", "functional",
        ),
        "minimum_disease_breadth": 3,
    },
    "biomarker_discovery": {
        "mode": "disease_specific_biomarker",
        "query_terms": (
            "brain", "imaging", "marker", "biomarker", "diagnosis", "diagnostic",
            "prognostic", "predictive", "disease", "mri", "pet", "eeg",
            "structural", "functional", "association",
        ),
        "minimum_disease_breadth": 1,
    },
}


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOP = frozenset(
    {
        "and", "are", "for", "from", "has", "have", "into", "may", "not",
        "our", "that", "the", "their", "this", "through", "using", "was",
        "were", "with", "within", "study", "results", "associated", "effect",
        "patient", "patients", "analysis", "data", "group", "groups",
    }
)


def stable_unit(*values: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_RE.findall(str(value or "").casefold())
        if token not in _STOP
    )


def _paper_key(edge: FrozenEdge) -> str:
    paper = edge.source_paper or {}
    return str(
        paper.get("pmid")
        or paper.get("doi")
        or paper.get("title")
        or edge.claim_id
        or ""
    )


def _edge_key(edge: FrozenEdge) -> tuple[str, ...]:
    return (
        edge.source_id,
        edge.target_id,
        edge.relation,
        edge.claim_id,
        _paper_key(edge),
        str(edge.year or ""),
    )


def unique_edges(adjacency: Mapping[str, Sequence[FrozenEdge]]) -> list[FrozenEdge]:
    observed: dict[tuple[str, ...], FrozenEdge] = {}
    for edges in adjacency.values():
        for edge in edges:
            observed.setdefault(_edge_key(edge), edge)
    return list(observed.values())


def _edge_text(edge: FrozenEdge, index: FrozenGraphIndex) -> str:
    paper = edge.source_paper or {}
    return " ".join(
        (
            index.names.get(edge.source_id, edge.source_id),
            index.names.get(edge.target_id, edge.target_id),
            edge.relation,
            edge.raw_text,
            str(paper.get("title") or ""),
        )
    )


def _recency(edge: FrozenEdge, freeze_year: int) -> float:
    if edge.year is None:
        return 0.25
    age = max(0, freeze_year - int(edge.year))
    return math.exp(-age / 8.0)


def _lexical_score(tokens: frozenset[str], query: frozenset[str]) -> float:
    if not query:
        return 0.0
    return len(tokens & query) / math.sqrt(len(query) * max(1, len(tokens)))


def _retrieval_pass(
    *,
    index: FrozenGraphIndex,
    adjacency: Mapping[str, Sequence[FrozenEdge]],
    query_terms: Sequence[str],
    freeze_year: int,
    retrieval_limit: int,
) -> tuple[dict[str, list[FrozenEdge]], dict[str, Any], dict[str, float], dict[str, set[str]]]:
    """Run query retrieval followed by deterministic pseudo-feedback retrieval."""

    query = frozenset(str(value).casefold() for value in query_terms)
    edge_rows: list[tuple[float, FrozenEdge, frozenset[str]]] = []
    for edge in unique_edges(adjacency):
        tokens = _tokens(_edge_text(edge, index))
        lexical = _lexical_score(tokens, query)
        citation = 1.0 if _paper_key(edge) else 0.0
        initial = (
            0.48 * lexical
            + 0.27 * float(edge.confidence)
            + 0.15 * _recency(edge, freeze_year)
            + 0.10 * citation
        )
        edge_rows.append((initial, edge, tokens))

    if not edge_rows:
        return {}, {
            "query_terms": sorted(query),
            "feedback_expansion_terms": [],
            "corpus_edges": 0,
            "retrieved_edges": 0,
            "cited_retrieved_edges": 0,
        }, {}, {}

    feedback_head = heapq.nlargest(
        min(512, len(edge_rows)), edge_rows, key=lambda row: row[0]
    )
    expansion_counts: Counter[str] = Counter()
    for score, _edge, tokens in feedback_head:
        for token in tokens - query:
            if len(token) >= 4:
                expansion_counts[token] += max(1, int(round(100.0 * score)))
    expansions = tuple(
        token
        for token, _ in sorted(
            expansion_counts.items(), key=lambda item: (-item[1], item[0])
        )[:16]
    )
    expansion_query = frozenset(expansions)

    reranked: list[tuple[float, FrozenEdge, frozenset[str]]] = []
    for initial, edge, tokens in edge_rows:
        expansion = _lexical_score(tokens, expansion_query)
        query_coverage = min(1.0, len(tokens & query) / 4.0)
        final = 0.68 * initial + 0.20 * expansion + 0.12 * query_coverage
        reranked.append((final, edge, tokens))
    selected = heapq.nlargest(
        min(retrieval_limit, len(reranked)), reranked, key=lambda row: row[0]
    )
    selected.sort(key=lambda row: (-row[0], _edge_key(row[1])))

    retrieved: dict[str, list[FrozenEdge]] = defaultdict(list)
    node_score: dict[str, float] = defaultdict(float)
    node_papers: dict[str, set[str]] = defaultdict(set)
    for score, edge, _tokens_value in selected:
        retrieved[edge.source_id].append(edge)
        retrieved[edge.target_id].append(edge)
        for node_id in (edge.source_id, edge.target_id):
            node_score[node_id] = max(node_score[node_id], float(score))
            paper = _paper_key(edge)
            if paper:
                node_papers[node_id].add(paper)

    top_citations = [
        _paper_key(edge)
        for _, edge, _ in selected
        if _paper_key(edge)
    ]
    audit = {
        "query_terms": sorted(query),
        "feedback_expansion_terms": list(expansions),
        "corpus_edges": len(edge_rows),
        "first_pass_feedback_documents": len(feedback_head),
        "retrieved_edges": len(selected),
        "cited_retrieved_edges": sum(bool(_paper_key(edge)) for _, edge, _ in selected),
        "top_retrieval_citations": list(dict.fromkeys(top_citations))[:50],
        "feedback_policy": "deterministic_pseudo_relevance_feedback",
    }
    return dict(retrieved), audit, dict(node_score), dict(node_papers)


def _node_graph_features(
    index: FrozenGraphIndex,
    adjacency: Mapping[str, Sequence[FrozenEdge]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    neighbours: dict[str, set[str]] = defaultdict(set)
    disease_neighbours: dict[str, set[str]] = defaultdict(set)
    degrees: dict[str, int] = {}
    for node_id, edges in adjacency.items():
        for edge in edges:
            other = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbours[node_id].add(other)
            if "disease" in index.atoms.get(other, frozenset()):
                disease_neighbours[node_id].add(other)
        degrees[node_id] = len(neighbours[node_id])
    return dict(neighbours), dict(disease_neighbours), degrees


def _breadth_score(count: int) -> float:
    return min(1.0, math.log1p(max(0, count)) / math.log(9.0))


def _task_fit(task_id: str, disease_count: int) -> float:
    mode = str(TASK_PROFILES[task_id]["mode"])
    breadth = _breadth_score(disease_count)
    if mode == "transdiagnostic_breadth":
        threshold = int(TASK_PROFILES[task_id]["minimum_disease_breadth"])
        gate = min(1.0, disease_count / max(1, threshold))
        return 0.70 * gate + 0.30 * breadth
    return 0.65 * (1.0 - breadth) + 0.35 * min(1.0, disease_count)


def _pair_jaccard(
    left: str,
    right: str,
    neighbours: Mapping[str, set[str]],
) -> float:
    a = neighbours.get(left, set())
    b = neighbours.get(right, set())
    union = a | b
    return 0.0 if not union else len(a & b) / len(union)


def _candidate_citations(
    row: Mapping[str, Any],
    node_papers: Mapping[str, set[str]],
) -> list[str]:
    metadata = row.get("metadata") or {}
    values: list[str] = []
    values.extend(str(value) for value in metadata.get("source_paper_keys") or ())
    for link in row.get("path") or ():
        paper = link.get("source_paper") or {}
        value = paper.get("pmid") or paper.get("doi") or paper.get("title")
        if value:
            values.append(str(value))
        claim_id = link.get("claim_id")
        if claim_id:
            values.append(str(claim_id))
    for node_id in (str(row.get("source_id") or ""), str(row.get("target_id") or "")):
        values.extend(sorted(node_papers.get(node_id, set())))
    values.extend(str(value) for value in row.get("supporting_claims") or ())
    return list(dict.fromkeys(value for value in values if value))


def _diverse_select(
    ranked: Sequence[dict[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    """Seed-stable coverage pass followed by score-preserving fill."""

    selected: list[dict[str, Any]] = []
    chosen: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for source_cap, target_cap in ((8, 20), (20, 50), (50, 1000)):
        for row in ranked:
            source = str(row.get("source_id") or "")
            target = str(row.get("target_id") or "")
            key = (source, target)
            if not source or not target or key in chosen:
                continue
            if source_counts[source] >= source_cap or target_counts[target] >= target_cap:
                continue
            selected.append(row)
            chosen.add(key)
            source_counts[source] += 1
            target_counts[target] += 1
            if len(selected) >= target_count:
                return selected
    for row in ranked:
        key = (str(row.get("source_id") or ""), str(row.get("target_id") or ""))
        if all(key) and key not in chosen:
            selected.append(row)
            chosen.add(key)
            if len(selected) >= target_count:
                break
    return selected


def _rank_sciagents(
    *,
    candidates: Sequence[Mapping[str, Any]],
    task_id: str,
    seed: int,
    index: FrozenGraphIndex,
    adjacency: Mapping[str, Sequence[FrozenEdge]],
    target_count: int,
    graph_features: tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    neighbours, disease_neighbours, degrees = graph_features or _node_graph_features(
        index, adjacency
    )
    ranked: list[dict[str, Any]] = []
    for raw in candidates:
        if raw.get("hypothesis_type") == "generation_failure":
            continue
        row = dict(raw)
        source = str(row.get("source_id") or "")
        target = str(row.get("target_id") or "")
        if not source or not target:
            continue
        path = list(row.get("path") or ())
        mediator = str(path[0].get("to_id") or "") if path else ""
        evidence = min(1.0, max(0.0, float(row.get("evidence_score") or 0.0)))
        grounding = min(1.0, (len(row.get("supporting_claims") or ()) + sum(
            bool((link.get("source_paper") or {}).get("pmid") or
                 (link.get("source_paper") or {}).get("doi"))
            for link in path
        )) / 4.0)
        mediator_degree = degrees.get(mediator, 0)
        specificity = 1.0 / (1.0 + math.log1p(max(0, mediator_degree - 1)))
        coherence = _pair_jaccard(source, target, neighbours)
        task_fit = _task_fit(task_id, len(disease_neighbours.get(source, set())))
        ontologist = 1.0 if path and len(path) >= 2 else 0.55
        scientist = 0.55 * evidence + 0.25 * grounding + 0.20 * coherence
        critic = 0.55 * grounding + 0.45 * specificity
        jitter = stable_unit(ADAPTER_VERSION, "sciagents", task_id, seed, source, target, mediator)
        score = (
            0.18 * ontologist
            + 0.32 * scientist
            + 0.25 * critic
            + 0.20 * task_fit
            + 0.05 * jitter
        )
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "adapter_version": ADAPTER_VERSION,
                "adaptation_label": "SciAgents-adapted",
                "native_upstream_execution": False,
                "task_conditioned": True,
                "deliberation_policy": "deterministic_ontologist_scientist_critic_decomposition",
                "role_scores": {
                    "ontologist": ontologist,
                    "scientist": scientist,
                    "critic": critic,
                    "task_fit": task_fit,
                },
                "source_disease_breadth": len(disease_neighbours.get(source, set())),
                "mediator_degree": mediator_degree,
                "publication_labels_available": False,
            }
        )
        row["metadata"] = metadata
        row["confidence_score"] = score
        row["composite_score"] = score
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            -float(row.get("composite_score") or 0.0),
            str(row.get("source_id") or ""),
            str(row.get("target_id") or ""),
        )
    )
    return _diverse_select(ranked, target_count)


def _rank_openscholar(
    *,
    candidates: Sequence[Mapping[str, Any]],
    task_id: str,
    seed: int,
    index: FrozenGraphIndex,
    scoped_adjacency: Mapping[str, Sequence[FrozenEdge]],
    retrieved_adjacency: Mapping[str, Sequence[FrozenEdge]],
    retrieval_audit: Mapping[str, Any],
    node_score: Mapping[str, float],
    node_papers: Mapping[str, set[str]],
    target_count: int,
    graph_features: tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    neighbours, disease_neighbours, _degrees = graph_features or _node_graph_features(
        index, scoped_adjacency
    )
    max_node_score = max(node_score.values(), default=1.0) or 1.0
    query = frozenset(retrieval_audit.get("query_terms") or ())
    expansions = frozenset(retrieval_audit.get("feedback_expansion_terms") or ())
    ranked: list[dict[str, Any]] = []
    for raw in candidates:
        if raw.get("hypothesis_type") == "generation_failure":
            continue
        row = dict(raw)
        source = str(row.get("source_id") or "")
        target = str(row.get("target_id") or "")
        if not source or not target:
            continue
        citations = _candidate_citations(row, node_papers)
        if not citations:
            continue
        retrieval = (node_score.get(source, 0.0) + node_score.get(target, 0.0)) / (
            2.0 * max_node_score
        )
        citation_diversity = min(1.0, math.log1p(len(citations)) / math.log(6.0))
        evidence = min(1.0, max(0.0, float(row.get("evidence_score") or 0.0)))
        pair_tokens = _tokens(
            f"{index.names.get(source, source)} {index.names.get(target, target)}"
        )
        coverage = min(
            1.0,
            _lexical_score(pair_tokens, query) + 0.5 * _lexical_score(pair_tokens, expansions),
        )
        structural = _pair_jaccard(source, target, neighbours)
        task_fit = _task_fit(task_id, len(disease_neighbours.get(source, set())))
        jitter = stable_unit(ADAPTER_VERSION, "openscholar", task_id, seed, source, target)
        score = (
            0.34 * retrieval
            + 0.20 * citation_diversity
            + 0.15 * evidence
            + 0.13 * coverage
            + 0.13 * task_fit
            + 0.03 * structural
            + 0.02 * jitter
        )
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "adapter_version": ADAPTER_VERSION,
                "adaptation_label": "OpenScholar-RAG-adapted",
                "native_upstream_execution": False,
                "task_conditioned": True,
                "retrieval_policy": "query_retrieve_feedback_reretrieve_fixed_compiler",
                "retrieval_query_terms": sorted(query),
                "feedback_expansion_terms": sorted(expansions),
                "retrieval_citations": citations[:20],
                "retrieval_score": retrieval,
                "citation_diversity_score": citation_diversity,
                "query_coverage_score": coverage,
                "task_fit_score": task_fit,
                "source_disease_breadth": len(disease_neighbours.get(source, set())),
                "publication_labels_available": False,
            }
        )
        row["metadata"] = metadata
        row["confidence_score"] = score
        row["composite_score"] = score
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            -float(row.get("composite_score") or 0.0),
            str(row.get("source_id") or ""),
            str(row.get("target_id") or ""),
        )
    )
    return _diverse_select(ranked, target_count)


def generate_adapted_hypotheses(
    *,
    method: str,
    case: CaseStudy,
    index: FrozenGraphIndex,
    graph_adjacency: Mapping[str, Sequence[FrozenEdge]],
    scoped_claim_adjacency: Mapping[str, Sequence[FrozenEdge]],
    freeze_year: int,
    seed: int,
    target_count: int = MAX_RANK,
    pool_size: int = POOL_SIZE,
) -> dict[str, Any]:
    """Generate one publication-blind adapted-baseline sequence."""

    if method not in METHODS:
        raise ValueError(f"unsupported adapted method: {method}")
    if case.name not in TASK_PROFILES:
        raise ValueError(f"missing preregistered task profile: {case.name}")
    if target_count < 1 or pool_size < target_count:
        raise ValueError("pool_size must be at least target_count")

    task_id = case.name
    task_graph_cache = getattr(index, "_v4r2_task_graph_cache", None)
    if task_graph_cache is None:
        task_graph_cache = {}
        setattr(index, "_v4r2_task_graph_cache", task_graph_cache)
    feature_cache = getattr(index, "_v4r2_graph_feature_cache", None)
    if feature_cache is None:
        feature_cache = {}
        setattr(index, "_v4r2_graph_feature_cache", feature_cache)
    if method == "sciagents_adapted":
        task_graph = task_graph_cache.get(("sciagents", task_id))
        if task_graph is None:
            task_graph = _merge_adjacencies(
                dict(graph_adjacency), dict(scoped_claim_adjacency)
            )
            task_graph_cache[("sciagents", task_id)] = task_graph
        graph_features = feature_cache.get(("sciagents", task_id))
        if graph_features is None:
            graph_features = _node_graph_features(index, task_graph)
            feature_cache[("sciagents", task_id)] = graph_features
        base = generate_case_hypotheses(
            method="sciagents",
            case=case,
            index=index,
            adjacency=task_graph,
            freeze_year=freeze_year,
            target_count=pool_size,
            seed=seed,
        )
        selected = _rank_sciagents(
            candidates=base.get("hypotheses") or (),
            task_id=task_id,
            seed=seed,
            index=index,
            adjacency=task_graph,
            target_count=target_count,
            graph_features=graph_features,
        )
        audit: dict[str, Any] = {
            "method_family": "SciAgents",
            "adapter_label": "SciAgents-adapted",
            "task_graph_nodes": len(task_graph),
            "scoped_claim_nodes": len(scoped_claim_adjacency),
            "candidate_pool_requested": pool_size,
            "candidate_pool_valid": sum(
                row.get("hypothesis_type") != "generation_failure"
                for row in base.get("hypotheses") or ()
            ),
            "role_topology": ["ontologist", "scientist", "critic", "ranker"],
            "contemporary_llm_used": False,
        }
    else:
        retrieval_limit = max(20000, target_count * 20)
        retrieval_cache = getattr(index, "_v4r2_retrieval_cache", None)
        if retrieval_cache is None:
            retrieval_cache = {}
            setattr(index, "_v4r2_retrieval_cache", retrieval_cache)
        retrieval_context = retrieval_cache.get(task_id)
        if retrieval_context is None:
            retrieval_context = _retrieval_pass(
                index=index,
                adjacency=scoped_claim_adjacency,
                query_terms=TASK_PROFILES[task_id]["query_terms"],
                freeze_year=freeze_year,
                retrieval_limit=retrieval_limit,
            )
            retrieval_cache[task_id] = retrieval_context
        retrieved, retrieval_audit, node_score, node_papers = retrieval_context
        graph_features = feature_cache.get(("openscholar", task_id))
        if graph_features is None:
            graph_features = _node_graph_features(index, scoped_claim_adjacency)
            feature_cache[("openscholar", task_id)] = graph_features
        base = generate_case_hypotheses(
            method="openscholar_rag",
            case=case,
            index=index,
            adjacency=retrieved,
            freeze_year=freeze_year,
            target_count=pool_size,
            seed=seed,
        )
        selected = _rank_openscholar(
            candidates=base.get("hypotheses") or (),
            task_id=task_id,
            seed=seed,
            index=index,
            scoped_adjacency=scoped_claim_adjacency,
            retrieved_adjacency=retrieved,
            retrieval_audit=retrieval_audit,
            node_score=node_score,
            node_papers=node_papers,
            target_count=target_count,
            graph_features=graph_features,
        )
        audit = {
            "method_family": "OpenScholar",
            "adapter_label": "OpenScholar-RAG-adapted",
            "candidate_pool_requested": pool_size,
            "candidate_pool_valid": sum(
                row.get("hypothesis_type") != "generation_failure"
                for row in base.get("hypotheses") or ()
            ),
            "retrieval": retrieval_audit,
            "fixed_hypothesis_compiler": True,
            "contemporary_llm_used": False,
        }

    pairs = [
        (str(row.get("source_id") or ""), str(row.get("target_id") or ""))
        for row in selected
    ]
    if len(selected) != target_count or len(set(pairs)) != target_count:
        raise RuntimeError(
            f"{method}/{task_id}/KG{freeze_year}/seed{seed} produced "
            f"{len(selected)} valid unique pairs, expected {target_count}"
        )
    return {
        "metadata": {
            "adapter_version": ADAPTER_VERSION,
            "method": method,
            "case_study_id": task_id,
            "freeze_year": freeze_year,
            "seed": seed,
            "requested": target_count,
            "valid": len(selected),
            "llm_api_enabled": False,
            "publication_labels_available": False,
            "adaptation_audit": audit,
        },
        "hypotheses": selected,
    }


def sequence_pair_overlap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    k: int = 100,
) -> dict[str, Any]:
    def pairs(payload: Mapping[str, Any]) -> set[tuple[str, str]]:
        return {
            (str(row.get("source_id") or ""), str(row.get("target_id") or ""))
            for row in list(payload.get("hypotheses") or ())[:k]
        }

    a, b = pairs(left), pairs(right)
    union = a | b
    return {
        "k": int(k),
        "left_unique": len(a),
        "right_unique": len(b),
        "intersection": len(a & b),
        "jaccard": 0.0 if not union else len(a & b) / len(union),
        "identical_order": [
            (row.get("source_id"), row.get("target_id"))
            for row in list(left.get("hypotheses") or ())[:k]
        ] == [
            (row.get("source_id"), row.get("target_id"))
            for row in list(right.get("hypotheses") or ())[:k]
        ],
    }


__all__ = [
    "ADAPTER_VERSION",
    "MAX_RANK",
    "METHODS",
    "POOL_SIZE",
    "TASK_PROFILES",
    "generate_adapted_hypotheses",
    "sequence_pair_overlap",
]
