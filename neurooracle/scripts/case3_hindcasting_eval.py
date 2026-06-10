from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from neurooracle.src.kge.complex_scorer import ComplExScorer


TREE_RELATIONS = {"is_a", "part_of", "about"}
GENE_DOMAINS = {"gene", "neurotransmitter"}
IMAGING_DOMAINS = {"biomarker", "imaging_feature", "connectivity", "neuroanatomy"}
DISEASE_DOMAINS = {"disease"}
NEGATIVE_RELATION_PREFIXES = (
    "does_not",
    "not_",
    "lacks",
    "fails_to",
)
GENERIC_IMAGING_NAMES = {
    "brain",
    "brains",
    "cerebrum",
    "central nervous system",
    "nervous system",
    "neural structures",
    "neural structure",
}


@dataclass
class PairEvidence:
    score: float = 0.0
    evidence_count: int = 0
    claim_edge_count: int = 0
    pmids: set[str] = field(default_factory=set)
    relations: Counter[str] = field(default_factory=Counter)

    def add(self, confidence: float, relation: str, pmid: str = "", claim_backed: bool = False) -> None:
        self.score = max(self.score, float(confidence or 0.0))
        self.evidence_count += 1
        if claim_backed:
            self.claim_edge_count += 1
        if pmid:
            self.pmids.add(str(pmid))
        self.relations[relation] += 1


@dataclass(order=True)
class RankedCandidate:
    sort_score: float
    gene_id: str = field(compare=False)
    imaging_id: str = field(compare=False)
    disease_id: str = field(compare=False)
    score: float = field(compare=False)
    plausibility: float = field(compare=False)
    mechanism_consistency: float = field(compare=False)
    reproducibility: float = field(compare=False)
    terminal_support: float = field(compare=False)
    kge_path_score: float = field(compare=False)
    gene_imaging_evidence: int = field(compare=False)
    imaging_disease_evidence: int = field(compare=False)
    claim_backed_edges: int = field(compare=False)
    claim_backed_segments: int = field(compare=False)
    gene_imaging_pmids: int = field(compare=False)
    imaging_disease_pmids: int = field(compare=False)
    gene_imaging_relations: tuple[str, ...] = field(compare=False)
    imaging_disease_relations: tuple[str, ...] = field(compare=False)

    def to_dict(self, names: dict[str, str]) -> dict[str, Any]:
        return {
            "gene_id": self.gene_id,
            "gene_name": names.get(self.gene_id, self.gene_id),
            "imaging_id": self.imaging_id,
            "imaging_name": names.get(self.imaging_id, self.imaging_id),
            "disease_id": self.disease_id,
            "disease_name": names.get(self.disease_id, self.disease_id),
            "score": self.score,
            "plausibility": self.plausibility,
            "mechanism_consistency": self.mechanism_consistency,
            "reproducibility": self.reproducibility,
            "terminal_support": self.terminal_support,
            "kge_path_score": self.kge_path_score,
            "gene_imaging_evidence": self.gene_imaging_evidence,
            "imaging_disease_evidence": self.imaging_disease_evidence,
            "claim_backed_edges": self.claim_backed_edges,
            "claim_backed_segments": self.claim_backed_segments,
            "gene_imaging_pmids": self.gene_imaging_pmids,
            "imaging_disease_pmids": self.imaging_disease_pmids,
            "gene_imaging_relations": list(self.gene_imaging_relations),
            "imaging_disease_relations": list(self.imaging_disease_relations),
        }


@dataclass
class NeighborhoodGraph:
    """Lightweight weighted undirected support graph for frozen-KG link scoring."""

    adj: dict[str, dict[str, float]]

    def degree(self, node_id: str) -> int:
        return len(self.adj.get(node_id, {}))

    def common_neighbor_score(self, left: str, right: str, max_common: int = 250) -> float:
        left_adj = self.adj.get(left, {})
        right_adj = self.adj.get(right, {})
        if not left_adj or not right_adj:
            return 0.0
        if len(left_adj) > len(right_adj):
            left_adj, right_adj = right_adj, left_adj
        common = []
        for nbr, lw in left_adj.items():
            rw = right_adj.get(nbr)
            if rw is None:
                continue
            deg = max(1, self.degree(nbr))
            # Weighted resource allocation / Adamic-Adar hybrid. Hub nodes
            # still help, but specific shared neighbors count much more.
            common.append(math.sqrt(lw * rw) / math.log1p(deg + 1))
        if not common:
            return 0.0
        common.sort(reverse=True)
        raw = sum(common[:max_common])
        return 1.0 - math.exp(-raw)

    def endpoint_support_score(self, left: str, right: str) -> float:
        left_deg = self.degree(left)
        right_deg = self.degree(right)
        if not left_deg or not right_deg:
            return 0.0
        # Weak prior: endpoints that both have a rich frozen-KG evidence
        # neighborhood are more likely to become future claim endpoints, but
        # this should not dominate shared-neighbor support.
        raw = math.sqrt(math.log1p(left_deg) * math.log1p(right_deg))
        return min(1.0, raw / math.log(5000))


def _domains(node: dict[str, Any] | None) -> set[str]:
    if not node:
        return set()
    return {str(x) for x in (node.get("domain_tags") or [])}


def _role_for_node(node: dict[str, Any] | None) -> str:
    domains = _domains(node)
    if domains & GENE_DOMAINS:
        return "gene"
    if domains & DISEASE_DOMAINS:
        return "disease"
    if domains & IMAGING_DOMAINS:
        return "imaging"
    return ""


def _is_stable_disease_node(nid: str, node: dict[str, Any] | None) -> bool:
    if _role_for_node(node) != "disease":
        return False
    if nid.startswith(("MSH:", "COGAT_DISORDER:", "CUI:")):
        return True
    return str((node or {}).get("source_vocab") or "") not in {"claim_extraction"}


def _is_specific_imaging_node(node: dict[str, Any] | None) -> bool:
    if _role_for_node(node) != "imaging":
        return False
    name = str((node or {}).get("preferred_name") or "").strip().casefold()
    if name in GENERIC_IMAGING_NAMES:
        return False
    if len(name) <= 3:
        return False
    return True


def _claim_year(claim: dict[str, Any]) -> int | None:
    year = (claim.get("source_paper") or {}).get("year") or claim.get("year")
    try:
        return int(year)
    except (TypeError, ValueError):
        return None


def _claim_pmid_from_node(node: dict[str, Any] | None) -> str:
    if not node:
        return ""
    paper = (node.get("metadata") or {}).get("source_paper") or {}
    return str(paper.get("pmid") or "")


def _top_relations(counter: Counter[str], n: int = 3) -> tuple[str, ...]:
    return tuple(rel for rel, _count in counter.most_common(n))


def _kge_best_pair_score(
    scorer: ComplExScorer | None,
    left: str,
    right: str,
    relations: Iterable[str],
) -> float:
    if scorer is None:
        return 0.0
    rels = [rel for rel in relations if rel]
    if not rels:
        return 0.5
    triples: list[tuple[str, str, str]] = []
    for rel in rels:
        triples.append((left, rel, right))
        triples.append((right, rel, left))
    return max(scorer.score_batch(triples), default=0.5)


def _kge_candidate_path_score(
    scorer: ComplExScorer | None,
    gene_id: str,
    imaging_id: str,
    disease_id: str,
    gi_ev: PairEvidence,
    imd_ev: PairEvidence,
) -> float:
    if scorer is None:
        return 0.0
    gi_score = _kge_best_pair_score(scorer, gene_id, imaging_id, _top_relations(gi_ev.relations))
    imd_score = _kge_best_pair_score(scorer, imaging_id, disease_id, _top_relations(imd_ev.relations))
    return math.sqrt(max(gi_score, 1e-6) * max(imd_score, 1e-6))


def _claim_oriented_triple_for_pair(row: dict[str, Any], pair: tuple[str, str]) -> tuple[str, str, str]:
    left_role, right_role = row["key"].split("_", 1)
    subject_role = str(row.get("subject_role") or "")
    predicate = str(row.get("predicate") or "")
    if subject_role == left_role:
        return pair[0], predicate, pair[1]
    if subject_role == right_role:
        return pair[1], predicate, pair[0]
    return str(row.get("subject_id") or ""), predicate, str(row.get("object_id") or "")


def _mechanism_score(imaging_node: dict[str, Any] | None, gi: PairEvidence, imd: PairEvidence) -> float:
    domains = _domains(imaging_node)
    score = 0.55
    if domains & {"imaging_feature", "connectivity"}:
        score += 0.20
    elif domains & {"neuroanatomy"}:
        score += 0.12
    elif domains & {"biomarker"}:
        score += 0.08
    rels = set(gi.relations) | set(imd.relations)
    if rels & {
        "gene_associated_with_anatomy",
        "gene_enriched_in_region",
        "is_imaging_feature_of",
        "measured_by_modality",
        "is_biomarker_of",
        "correlates_with",
        "predicts",
        "distinguishes",
    }:
        score += 0.15
    if gi.evidence_count > 0 and imd.evidence_count > 0:
        score += 0.10
    return min(score, 1.0)


def _reproducibility_score(gi: PairEvidence, imd: PairEvidence) -> float:
    n_pmids = len(gi.pmids | imd.pmids)
    n_edges = gi.evidence_count + imd.evidence_count
    pmid_score = min(1.0, math.log1p(n_pmids) / math.log(12))
    edge_score = min(1.0, math.log1p(n_edges) / math.log(20))
    return 0.65 * pmid_score + 0.35 * edge_score


def load_kg_index(kg_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    graph = json.load(kg_path.open("r", encoding="utf-8"))
    concepts = graph["concepts"]
    names = {nid: str(node.get("preferred_name") or nid) for nid, node in concepts.items()}
    return concepts, graph["edges"], names


def _is_negative_relation(relation: str) -> bool:
    rel = relation.strip().casefold()
    return any(rel.startswith(prefix) for prefix in NEGATIVE_RELATION_PREFIXES)


def build_neighborhood_graph(
    concepts: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    strict_candidate_anchors: bool = False,
) -> NeighborhoodGraph:
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in edges:
        rel = str(edge.get("relation_type") or "")
        if rel in TREE_RELATIONS or _is_negative_relation(rel):
            continue
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if not source_id or not target_id or source_id == target_id:
            continue
        source_node = concepts.get(source_id)
        target_node = concepts.get(target_id)
        if not source_node or not target_node:
            continue
        source_domains = _domains(source_node)
        target_domains = _domains(target_node)
        if "claim" in source_domains or "claim" in target_domains:
            continue
        if strict_candidate_anchors:
            if _role_for_node(source_node) == "imaging" and not _is_specific_imaging_node(source_node):
                continue
            if _role_for_node(target_node) == "imaging" and not _is_specific_imaging_node(target_node):
                continue
            if _role_for_node(source_node) == "disease" and not _is_stable_disease_node(source_id, source_node):
                continue
            if _role_for_node(target_node) == "disease" and not _is_stable_disease_node(target_id, target_node):
                continue
        confidence = max(0.01, float(edge.get("confidence") or 0.0))
        # Several infrastructure-curated edges carry useful semantics but
        # lower confidence; keep them as weak topology evidence.
        confidence = min(1.0, confidence)
        if confidence > adj[source_id].get(target_id, 0.0):
            adj[source_id][target_id] = confidence
        if confidence > adj[target_id].get(source_id, 0.0):
            adj[target_id][source_id] = confidence
    return NeighborhoodGraph(adj={k: dict(v) for k, v in adj.items()})


def build_pair_evidence(
    concepts: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    strict_candidate_anchors: bool = False,
) -> tuple[
    dict[tuple[str, str], PairEvidence],
    dict[tuple[str, str], PairEvidence],
    dict[tuple[str, str], PairEvidence],
    set[tuple[str, str]],
]:
    gene_imaging: dict[tuple[str, str], PairEvidence] = defaultdict(PairEvidence)
    imaging_disease: dict[tuple[str, str], PairEvidence] = defaultdict(PairEvidence)
    gene_disease: dict[tuple[str, str], PairEvidence] = defaultdict(PairEvidence)
    historical_direct_pairs: set[tuple[str, str]] = set()

    for edge in edges:
        rel = str(edge.get("relation_type") or "")
        if rel in TREE_RELATIONS:
            continue
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        source_node = concepts.get(source_id)
        target_node = concepts.get(target_id)
        source_role = _role_for_node(source_node)
        target_role = _role_for_node(target_node)
        if not source_role or not target_role:
            continue
        historical_direct_pairs.add((source_id, target_id))
        historical_direct_pairs.add((target_id, source_id))

        confidence = float(edge.get("confidence") or 0.0)
        claim_id = str((edge.get("metadata") or {}).get("claim_id") or "")
        claim_backed = bool(claim_id)
        pmid = _claim_pmid_from_node(concepts.get(claim_id))

        roles = {source_role, target_role}
        if roles == {"gene", "imaging"}:
            gene = source_id if source_role == "gene" else target_id
            imaging = source_id if source_role == "imaging" else target_id
            if strict_candidate_anchors and not _is_specific_imaging_node(concepts.get(imaging)):
                continue
            gene_imaging[(gene, imaging)].add(confidence, rel, pmid, claim_backed=claim_backed)
        elif roles == {"imaging", "disease"}:
            imaging = source_id if source_role == "imaging" else target_id
            disease = source_id if source_role == "disease" else target_id
            if strict_candidate_anchors and (
                not _is_specific_imaging_node(concepts.get(imaging))
                or not _is_stable_disease_node(disease, concepts.get(disease))
            ):
                continue
            imaging_disease[(imaging, disease)].add(confidence, rel, pmid, claim_backed=claim_backed)
        elif roles == {"gene", "disease"}:
            gene = source_id if source_role == "gene" else target_id
            disease = source_id if source_role == "disease" else target_id
            if strict_candidate_anchors and not _is_stable_disease_node(disease, concepts.get(disease)):
                continue
            gene_disease[(gene, disease)].add(confidence, rel, pmid, claim_backed=claim_backed)

    return gene_imaging, imaging_disease, gene_disease, historical_direct_pairs


def generate_candidates(
    concepts: dict[str, dict[str, Any]],
    gene_imaging: dict[tuple[str, str], PairEvidence],
    imaging_disease: dict[tuple[str, str], PairEvidence],
    max_candidates: int,
    per_imaging_gene_limit: int,
    per_imaging_disease_limit: int,
    claim_edge_policy: str = "prefer",
    neighborhood: NeighborhoodGraph | None = None,
    terminal_support_weight: float = 0.20,
    kge_scorer: ComplExScorer | None = None,
    kge_path_weight: float = 0.0,
) -> list[RankedCandidate]:
    genes_by_imaging: dict[str, list[tuple[str, PairEvidence]]] = defaultdict(list)
    diseases_by_imaging: dict[str, list[tuple[str, PairEvidence]]] = defaultdict(list)
    for (gene, imaging), ev in gene_imaging.items():
        genes_by_imaging[imaging].append((gene, ev))
    for (imaging, disease), ev in imaging_disease.items():
        diseases_by_imaging[imaging].append((disease, ev))

    heap: list[RankedCandidate] = []
    for imaging_id in sorted(set(genes_by_imaging) & set(diseases_by_imaging)):
        genes = sorted(
            genes_by_imaging[imaging_id],
            key=lambda item: (item[1].score, len(item[1].pmids), item[1].evidence_count),
            reverse=True,
        )[:per_imaging_gene_limit]
        diseases = sorted(
            diseases_by_imaging[imaging_id],
            key=lambda item: (item[1].score, len(item[1].pmids), item[1].evidence_count),
            reverse=True,
        )[:per_imaging_disease_limit]
        for gene_id, gi_ev in genes:
            for disease_id, imd_ev in diseases:
                claim_segments = int(gi_ev.claim_edge_count > 0) + int(imd_ev.claim_edge_count > 0)
                claim_edges = gi_ev.claim_edge_count + imd_ev.claim_edge_count
                if claim_edge_policy == "require-any" and claim_segments == 0:
                    continue
                if claim_edge_policy == "require-all" and claim_segments < 2:
                    continue
                plausibility = math.sqrt(max(gi_ev.score, 1e-6) * max(imd_ev.score, 1e-6))
                mechanism = _mechanism_score(concepts.get(imaging_id), gi_ev, imd_ev)
                reproducibility = _reproducibility_score(gi_ev, imd_ev)
                score = 0.45 * plausibility + 0.35 * mechanism + 0.20 * reproducibility
                terminal_support = (
                    neighborhood.common_neighbor_score(gene_id, disease_id)
                    if neighborhood is not None else 0.0
                )
                if terminal_support_weight > 0:
                    score = (
                        (1.0 - terminal_support_weight) * score
                        + terminal_support_weight * terminal_support
                    )
                kge_path_score = _kge_candidate_path_score(
                    kge_scorer, gene_id, imaging_id, disease_id, gi_ev, imd_ev
                )
                if kge_scorer is not None and kge_path_weight > 0:
                    score = (1.0 - kge_path_weight) * score + kge_path_weight * kge_path_score
                if claim_edge_policy == "prefer":
                    score = 0.88 * score + 0.12 * (claim_segments / 2.0)
                cand = RankedCandidate(
                    sort_score=score,
                    gene_id=gene_id,
                    imaging_id=imaging_id,
                    disease_id=disease_id,
                    score=score,
                    plausibility=plausibility,
                    mechanism_consistency=mechanism,
                    reproducibility=reproducibility,
                    terminal_support=terminal_support,
                    kge_path_score=kge_path_score,
                    gene_imaging_evidence=gi_ev.evidence_count,
                    imaging_disease_evidence=imd_ev.evidence_count,
                    claim_backed_edges=claim_edges,
                    claim_backed_segments=claim_segments,
                    gene_imaging_pmids=len(gi_ev.pmids),
                    imaging_disease_pmids=len(imd_ev.pmids),
                    gene_imaging_relations=_top_relations(gi_ev.relations),
                    imaging_disease_relations=_top_relations(imd_ev.relations),
                )
                if len(heap) < max_candidates:
                    heapq.heappush(heap, cand)
                elif cand.sort_score > heap[0].sort_score:
                    heapq.heapreplace(heap, cand)
    return sorted(heap, key=lambda c: c.score, reverse=True)


def _candidate_key_sets(candidates: list[RankedCandidate]) -> dict[str, dict[tuple[str, str], float]]:
    keys = {
        "gene_disease": {},
        "gene_imaging": {},
        "imaging_disease": {},
    }
    for cand in candidates:
        keys["gene_disease"].setdefault((cand.gene_id, cand.disease_id), cand.score)
        keys["gene_imaging"].setdefault((cand.gene_id, cand.imaging_id), cand.score)
        keys["imaging_disease"].setdefault((cand.imaging_id, cand.disease_id), cand.score)
    return keys


def _role_pair_key(a: str, b: str) -> str:
    roles = {a, b}
    if roles == {"gene", "disease"}:
        return "gene_disease"
    if roles == {"gene", "imaging"}:
        return "gene_imaging"
    if roles == {"imaging", "disease"}:
        return "imaging_disease"
    return ""


def _ordered_pair_for_key(sid: str, srole: str, oid: str, orole: str, key: str) -> tuple[str, str]:
    if key == "gene_disease":
        return (sid, oid) if srole == "gene" else (oid, sid)
    if key == "gene_imaging":
        return (sid, oid) if srole == "gene" else (oid, sid)
    if key == "imaging_disease":
        return (sid, oid) if srole == "imaging" else (oid, sid)
    return ("", "")


def load_future_claim_pairs(
    claims_path: Path,
    concepts: dict[str, dict[str, Any]],
    start_year: int,
    end_year: int,
    historical_direct_pairs: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    future: list[dict[str, Any]] = []
    stats = Counter()
    for line in claims_path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        claim = json.loads(line)
        year = _claim_year(claim)
        if year is None or year < start_year or year > end_year:
            continue
        stats["future_claims_total"] += 1
        sid, oid = str(claim.get("subject_id") or ""), str(claim.get("object_id") or "")
        if not sid or not oid:
            stats["missing_endpoint"] += 1
            continue
        srole, orole = _role_for_node(concepts.get(sid)), _role_for_node(concepts.get(oid))
        key = _role_pair_key(srole, orole)
        if not key:
            stats["wrong_atom_pair"] += 1
            continue
        pair = _ordered_pair_for_key(sid, srole, oid, orole, key)
        if not pair[0] or not pair[1]:
            stats["bad_pair"] += 1
            continue
        if pair in historical_direct_pairs:
            stats["already_direct_in_frozen_kg"] += 1
            continue
        future.append({
            "claim_id": claim.get("id"),
            "pmid": (claim.get("source_paper") or {}).get("pmid"),
            "year": year,
            "key": key,
            "pair": pair,
            "subject_id": sid,
            "subject_role": srole,
            "subject_name": claim.get("subject_name"),
            "predicate": claim.get("predicate"),
            "object_id": oid,
            "object_role": orole,
            "object_name": claim.get("object_name"),
        })
        stats[f"evaluable_{key}"] += 1
    stats["future_evaluable_novel"] = len(future)
    return future, dict(stats)


def recall_at_k(future: list[dict[str, Any]], candidates: list[RankedCandidate], ks: Iterable[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    terminal_future = [row for row in future if row["key"] == "gene_disease"]
    for k in ks:
        keys = _candidate_key_sets(candidates[:k])
        top = candidates[:k]
        claim_backed_any = sum(1 for c in top if c.claim_backed_segments > 0)
        claim_backed_all = sum(1 for c in top if c.claim_backed_segments >= 2)
        hits = 0
        by_key = Counter()
        totals = Counter()
        for row in future:
            totals[row["key"]] += 1
            if tuple(row["pair"]) in keys[row["key"]]:
                hits += 1
                by_key[row["key"]] += 1
        denom = len(future)
        out[f"recall@{k}"] = hits / denom if denom else 0.0
        out[f"hits@{k}"] = hits
        terminal_hits = sum(
            1 for row in terminal_future
            if tuple(row["pair"]) in keys["gene_disease"]
        )
        out[f"terminal_gene_disease_recall@{k}"] = (
            terminal_hits / len(terminal_future) if terminal_future else 0.0
        )
        out[f"terminal_gene_disease_hits@{k}"] = terminal_hits
        out[f"terminal_gene_disease_total@{k}"] = len(terminal_future)
        out[f"claim_backed_any@{k}"] = claim_backed_any
        out[f"claim_backed_all@{k}"] = claim_backed_all
        out[f"by_key@{k}"] = {
            key: {
                "hits": by_key[key],
                "total": totals[key],
                "recall": by_key[key] / totals[key] if totals[key] else 0.0,
            }
            for key in sorted(totals)
        }
    return out


def _score_pair(key: str, pair: tuple[str, str], candidate_keys: dict[str, dict[tuple[str, str], float]]) -> float:
    return float(candidate_keys.get(key, {}).get(pair, 0.0))


def _direct_score(ev: PairEvidence | None) -> float:
    if ev is None:
        return 0.0
    return 0.70 * ev.score + 0.30 * min(1.0, math.log1p(len(ev.pmids) + ev.evidence_count) / math.log(20))


def _chain_score(concepts: dict[str, dict[str, Any]], imaging_id: str, left: PairEvidence, right: PairEvidence) -> float:
    plausibility = math.sqrt(max(left.score, 1e-6) * max(right.score, 1e-6))
    mechanism = _mechanism_score(concepts.get(imaging_id), left, right)
    reproducibility = _reproducibility_score(left, right)
    return 0.45 * plausibility + 0.35 * mechanism + 0.20 * reproducibility


def _build_score_indexes(
    gene_imaging: dict[tuple[str, str], PairEvidence],
    imaging_disease: dict[tuple[str, str], PairEvidence],
    gene_disease: dict[tuple[str, str], PairEvidence],
) -> dict[str, dict[str, list[tuple[str, PairEvidence]]]]:
    idx: dict[str, dict[str, list[tuple[str, PairEvidence]]]] = {
        "gi_by_gene": defaultdict(list),
        "gi_by_imaging": defaultdict(list),
        "imd_by_imaging": defaultdict(list),
        "imd_by_disease": defaultdict(list),
        "gd_by_gene": defaultdict(list),
        "gd_by_disease": defaultdict(list),
    }
    for (gene, imaging), ev in gene_imaging.items():
        idx["gi_by_gene"][gene].append((imaging, ev))
        idx["gi_by_imaging"][imaging].append((gene, ev))
    for (imaging, disease), ev in imaging_disease.items():
        idx["imd_by_imaging"][imaging].append((disease, ev))
        idx["imd_by_disease"][disease].append((imaging, ev))
    for (gene, disease), ev in gene_disease.items():
        idx["gd_by_gene"][gene].append((disease, ev))
        idx["gd_by_disease"][disease].append((gene, ev))
    return idx


def _path_based_pair_score(
    key: str,
    pair: tuple[str, str],
    concepts: dict[str, dict[str, Any]],
    gene_imaging: dict[tuple[str, str], PairEvidence],
    imaging_disease: dict[tuple[str, str], PairEvidence],
    gene_disease: dict[tuple[str, str], PairEvidence],
    indexes: dict[str, dict[str, list[tuple[str, PairEvidence]]]],
    max_neighbors: int = 120,
) -> float:
    if key == "gene_disease":
        gene, disease = pair
        best = _direct_score(gene_disease.get((gene, disease)))
        left = sorted(indexes["gi_by_gene"].get(gene, []), key=lambda x: x[1].score, reverse=True)[:max_neighbors]
        right_map = dict(indexes["imd_by_disease"].get(disease, []))
        for imaging, gi_ev in left:
            imd_ev = right_map.get(imaging)
            if imd_ev:
                best = max(best, _chain_score(concepts, imaging, gi_ev, imd_ev))
        return best
    if key == "gene_imaging":
        gene, imaging = pair
        best = _direct_score(gene_imaging.get((gene, imaging)))
        left = sorted(indexes["gd_by_gene"].get(gene, []), key=lambda x: x[1].score, reverse=True)[:max_neighbors]
        right_map = dict(indexes["imd_by_imaging"].get(imaging, []))
        for disease, gd_ev in left:
            imd_ev = right_map.get(disease)
            if imd_ev:
                # Reuse mechanism scoring with the imaging node as the mediator-like anchor.
                best = max(best, _chain_score(concepts, imaging, gd_ev, imd_ev))
        return best
    if key == "imaging_disease":
        imaging, disease = pair
        best = _direct_score(imaging_disease.get((imaging, disease)))
        left = sorted(indexes["gi_by_imaging"].get(imaging, []), key=lambda x: x[1].score, reverse=True)[:max_neighbors]
        right_map = dict(indexes["gd_by_disease"].get(disease, []))
        for gene, gi_ev in left:
            gd_ev = right_map.get(gene)
            if gd_ev:
                best = max(best, _chain_score(concepts, imaging, gi_ev, gd_ev))
        return best
    return 0.0


def _hybrid_pair_score(
    key: str,
    pair: tuple[str, str],
    row: dict[str, Any],
    concepts: dict[str, dict[str, Any]],
    gene_imaging: dict[tuple[str, str], PairEvidence],
    imaging_disease: dict[tuple[str, str], PairEvidence],
    gene_disease: dict[tuple[str, str], PairEvidence],
    indexes: dict[str, dict[str, list[tuple[str, PairEvidence]]]],
    neighborhood: NeighborhoodGraph,
    shared_weight: float,
    path_weight: float,
    endpoint_weight: float,
    kge_weight: float,
    kge_scorer: ComplExScorer | None,
) -> dict[str, float]:
    path_score = _path_based_pair_score(
        key, pair, concepts, gene_imaging, imaging_disease, gene_disease, indexes
    )
    left, right = pair
    shared = neighborhood.common_neighbor_score(left, right)
    endpoint = neighborhood.endpoint_support_score(left, right)
    kge_claim = 0.0
    if kge_scorer is not None:
        kge_claim = kge_scorer.score_triple(*_claim_oriented_triple_for_pair(row, pair))
    weight_sum = max(shared_weight + path_weight + endpoint_weight + kge_weight, 1e-9)
    shared_w = shared_weight / weight_sum
    path_w = path_weight / weight_sum
    endpoint_w = endpoint_weight / weight_sum
    kge_w = kge_weight / weight_sum
    # Classification is about likelihood of future confirmation, not
    # top-candidate generation. In the current proxy setup, weighted common
    # evidence neighbors are the most discriminative frozen-KG signal; path and
    # endpoint support remain configurable for stricter future negative pools.
    hybrid = shared_w * shared + path_w * path_score + endpoint_w * endpoint + kge_w * kge_claim
    return {
        "hybrid": min(1.0, hybrid),
        "path": path_score,
        "shared_neighbor": shared,
        "endpoint_support": endpoint,
        "kge_claim": kge_claim,
    }


def _auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [(s, i) for i, (label, s) in enumerate(zip(labels, scores)) if label == 1]
    neg = [(s, i) for i, (label, s) in enumerate(zip(labels, scores)) if label == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _auprc(labels: list[int], scores: list[float]) -> float | None:
    if not any(labels):
        return None
    ordered = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    total_pos = sum(labels)
    for _score, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def classification_proxy(
    future: list[dict[str, Any]],
    candidates: list[RankedCandidate],
    concepts: dict[str, dict[str, Any]],
    gene_imaging: dict[tuple[str, str], PairEvidence],
    imaging_disease: dict[tuple[str, str], PairEvidence],
    gene_disease: dict[tuple[str, str], PairEvidence],
    neighborhood: NeighborhoodGraph,
    historical_direct_pairs: set[tuple[str, str]],
    max_examples: int,
    seed: int,
    shared_weight: float,
    path_weight: float,
    endpoint_weight: float,
    kge_weight: float,
    kge_scorer: ComplExScorer | None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    candidate_keys = _candidate_key_sets(candidates)
    indexes = _build_score_indexes(gene_imaging, imaging_disease, gene_disease)
    role_nodes = {
        "gene": [nid for nid, node in concepts.items() if _role_for_node(node) == "gene"],
        "imaging": [nid for nid, node in concepts.items() if _role_for_node(node) == "imaging"],
        "disease": [nid for nid, node in concepts.items() if _role_for_node(node) == "disease"],
    }
    positives = future[:max_examples]
    labels: list[int] = []
    scores: list[float] = []
    path_scores: list[float] = []
    shared_scores: list[float] = []
    endpoint_scores: list[float] = []
    kge_scores: list[float] = []
    negatives_made = 0
    for row in positives:
        key = row["key"]
        pair = tuple(row["pair"])
        labels.append(1)
        pos_score = _hybrid_pair_score(
            key, pair, row, concepts, gene_imaging, imaging_disease, gene_disease, indexes,
            neighborhood, shared_weight, path_weight, endpoint_weight, kge_weight, kge_scorer,
        )
        scores.append(pos_score["hybrid"])
        path_scores.append(pos_score["path"])
        shared_scores.append(pos_score["shared_neighbor"])
        endpoint_scores.append(pos_score["endpoint_support"])
        kge_scores.append(pos_score["kge_claim"])

        left_role, right_role = key.split("_", 1)
        left, right = pair
        replacement_pool = role_nodes[right_role]
        neg_pair = None
        for _ in range(100):
            repl = rng.choice(replacement_pool)
            candidate = (left, repl)
            if candidate == pair:
                continue
            if candidate in historical_direct_pairs:
                continue
            neg_pair = candidate
            break
        if neg_pair is None:
            continue
        negatives_made += 1
        labels.append(0)
        neg_score = _hybrid_pair_score(
            key, neg_pair, row, concepts, gene_imaging, imaging_disease, gene_disease, indexes,
            neighborhood, shared_weight, path_weight, endpoint_weight, kge_weight, kge_scorer,
        )
        scores.append(neg_score["hybrid"])
        path_scores.append(neg_score["path"])
        shared_scores.append(neg_score["shared_neighbor"])
        endpoint_scores.append(neg_score["endpoint_support"])
        kge_scores.append(neg_score["kge_claim"])
    topk_labels: list[int] = []
    topk_scores: list[float] = []
    for row in positives:
        key = row["key"]
        pair = tuple(row["pair"])
        topk_labels.append(1)
        topk_scores.append(_score_pair(key, pair, candidate_keys))
    return {
        "positive_examples": len(positives),
        "negative_examples": negatives_made,
        "auc": _auc(labels, scores),
        "auprc": _auprc(labels, scores),
        "positive_mean_score": sum(s for l, s in zip(labels, scores) if l) / max(1, sum(labels)),
        "negative_mean_score": sum(s for l, s in zip(labels, scores) if not l) / max(1, len(labels) - sum(labels)),
        "positive_mean_path_score": sum(s for l, s in zip(labels, path_scores) if l) / max(1, sum(labels)),
        "negative_mean_path_score": sum(s for l, s in zip(labels, path_scores) if not l) / max(1, len(labels) - sum(labels)),
        "positive_mean_shared_neighbor_score": sum(s for l, s in zip(labels, shared_scores) if l) / max(1, sum(labels)),
        "negative_mean_shared_neighbor_score": sum(s for l, s in zip(labels, shared_scores) if not l) / max(1, len(labels) - sum(labels)),
        "positive_mean_endpoint_support_score": sum(s for l, s in zip(labels, endpoint_scores) if l) / max(1, sum(labels)),
        "negative_mean_endpoint_support_score": sum(s for l, s in zip(labels, endpoint_scores) if not l) / max(1, len(labels) - sum(labels)),
        "positive_mean_kge_score": sum(s for l, s in zip(labels, kge_scores) if l) / max(1, sum(labels)),
        "negative_mean_kge_score": sum(s for l, s in zip(labels, kge_scores) if not l) / max(1, len(labels) - sum(labels)),
        "top_candidate_positive_mean_score": sum(topk_scores) / max(1, len(topk_scores)),
        "classification_weights": {
            "shared_neighbor": shared_weight,
            "path_support": path_weight,
            "endpoint_support": endpoint_weight,
            "kge_claim": kge_weight,
        },
        "scoring_note": (
            "Proxy classification: positives are 2021-2026 held-out future claims; "
            "negatives are type-preserving corrupted pairs. Claims are scored by a configurable weighted mix of frozen-KG path support, shared-neighbor evidence, endpoint support, and optional KGE claim plausibility, "
            "not by novelty. Replace with retracted/zero-citation claims when metadata is available."
        ),
    }


def write_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    freeze_label = f"KG_{manifest['freeze_year']}" if manifest.get("freeze_year") is not None else "the frozen KG"
    lines = [
        "# Case Study 3 Hindcasting Smoke Evaluation",
        "",
        "## Setup",
        f"- KG: `{manifest['kg_path']}`",
        f"- Freeze year: {manifest.get('freeze_year', 'unknown')}",
        f"- Future claims: `{manifest['future_claims_path']}`",
        f"- Future window: {manifest['future_start_year']}-{manifest['future_end_year']}",
        f"- Candidate chain: `GENE_TARGET -> IMAGING_MARKER -> DISEASE`",
        "- Score dimensions: plausibility + mechanism consistency + reproducibility; novelty is not used.",
        "",
        "## Candidate Generation",
        f"- Candidates generated: {manifest['candidate_count']}",
        f"- Gene-imaging historical pairs: {manifest['gene_imaging_pairs']}",
        f"- Imaging-disease historical pairs: {manifest['imaging_disease_pairs']}",
        f"- Gene-disease historical pairs: {manifest.get('gene_disease_pairs', 0)}",
        f"- Strict candidate anchors: {manifest.get('strict_candidate_anchors', False)}",
        f"- Claim-edge policy: {manifest.get('candidate_claim_edge_policy', 'unknown')}",
        f"- Terminal support weight: {manifest.get('candidate_terminal_support_weight', 0.0)}",
        "",
        "## Future Claim Pool",
    ]
    for key, value in sorted(manifest["future_pool_stats"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Forward Recall"])
    for key, value in manifest["forward_recall"].items():
        if key.startswith("recall@"):
            k = key.split("@", 1)[1]
            any_n = manifest["forward_recall"].get("claim_backed_any@" + k, 0)
            all_n = manifest["forward_recall"].get("claim_backed_all@" + k, 0)
            lines.append(
                f"- Recall@{k}: {value:.6f} ({manifest['forward_recall'].get('hits@' + k, 0)} hits; "
                f"claim-backed any/all: {any_n}/{all_n})"
            )
            term = manifest["forward_recall"].get("terminal_gene_disease_recall@" + k)
            if term is not None:
                term_hits = manifest["forward_recall"].get("terminal_gene_disease_hits@" + k, 0)
                term_total = manifest["forward_recall"].get("terminal_gene_disease_total@" + k, 0)
                lines.append(
                    f"  - Terminal gene-disease Recall@{k}: {term:.6f} "
                    f"({term_hits}/{term_total})"
                )
    lines.extend(["", "## Classification Proxy"])
    cls = manifest["classification_proxy"]
    lines.append(f"- AUC: {cls['auc']:.6f}" if cls["auc"] is not None else "- AUC: NA")
    lines.append(f"- AUPRC: {cls['auprc']:.6f}" if cls["auprc"] is not None else "- AUPRC: NA")
    lines.append(f"- Positive examples: {cls['positive_examples']}")
    lines.append(f"- Negative examples: {cls['negative_examples']}")
    lines.append(f"- Positive mean score: {cls['positive_mean_score']:.6f}")
    lines.append(f"- Negative mean score: {cls['negative_mean_score']:.6f}")
    if "positive_mean_shared_neighbor_score" in cls:
        lines.append(f"- Positive shared-neighbor score: {cls['positive_mean_shared_neighbor_score']:.6f}")
        lines.append(f"- Negative shared-neighbor score: {cls['negative_mean_shared_neighbor_score']:.6f}")
    if "positive_mean_kge_score" in cls:
        lines.append(f"- Positive KGE claim score: {cls['positive_mean_kge_score']:.6f}")
        lines.append(f"- Negative KGE claim score: {cls['negative_mean_kge_score']:.6f}")
    lines.extend([
        "",
        "## Limitations",
        "- Local data currently lacks impact factor, citation count, retraction labels, and zero-citation labels.",
        "- This run therefore uses all evaluable future claims for recall and type-preserving corrupted negatives for classification.",
        f"- Future claims whose endpoints did not exist in {freeze_label} are excluded from the evaluable denominator.",
        f"- Claims already directly connected in {freeze_label} are excluded to reduce leakage from already-known facts.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Case Study 3 hindcasting smoke evaluator.")
    parser.add_argument("--kg", type=Path, default=Path("neurooracle/data/snapshots/kg_2020_from_full_snapshot_v1/knowledge_graph.json"))
    parser.add_argument("--future-claims", type=Path, default=Path("neurooracle/data/full_snapshot_v1/extracted_claims.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/kg2020_smoke_eval"))
    parser.add_argument("--freeze-year", type=int, default=None)
    parser.add_argument("--future-start-year", type=int, default=2021)
    parser.add_argument("--future-end-year", type=int, default=2026)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--per-imaging-gene-limit", type=int, default=80)
    parser.add_argument("--per-imaging-disease-limit", type=int, default=80)
    parser.add_argument("--classification-max-examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--classification-shared-weight", type=float, default=1.0)
    parser.add_argument("--classification-path-weight", type=float, default=0.0)
    parser.add_argument("--classification-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--classification-kge-weight", type=float, default=0.0)
    parser.add_argument("--kge-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--candidate-terminal-support-weight",
        type=float,
        default=0.20,
        help="Weight for frozen-KG shared-neighbor support between terminal gene and disease in candidate ranking.",
    )
    parser.add_argument(
        "--candidate-kge-path-weight",
        type=float,
        default=0.0,
        help="Weight for ComplEx path support on candidate GENE->IMAGING->DISEASE segments.",
    )
    parser.add_argument(
        "--candidate-claim-edge-policy",
        choices=("prefer", "require-any", "require-all", "off"),
        default="require-any",
        help=(
            "How candidate generation handles historical claim-backed edges on "
            "GENE->IMAGING and IMAGING->DISEASE segments."
        ),
    )
    parser.add_argument(
        "--strict-candidate-anchors",
        action="store_true",
        help="Filter generic imaging anchors and claim-extraction disease targets in the frozen-KG support graph.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kge_scorer = ComplExScorer.load(args.kge_checkpoint) if args.kge_checkpoint else None
    concepts, edges, names = load_kg_index(args.kg)
    gene_imaging, imaging_disease, gene_disease, historical_direct_pairs = build_pair_evidence(
        concepts,
        edges,
        strict_candidate_anchors=args.strict_candidate_anchors,
    )
    neighborhood = build_neighborhood_graph(
        concepts,
        edges,
        strict_candidate_anchors=args.strict_candidate_anchors,
    )
    future, future_stats = load_future_claim_pairs(
        claims_path=args.future_claims,
        concepts=concepts,
        start_year=args.future_start_year,
        end_year=args.future_end_year,
        historical_direct_pairs=historical_direct_pairs,
    )
    candidates = generate_candidates(
        concepts=concepts,
        gene_imaging=gene_imaging,
        imaging_disease=imaging_disease,
        max_candidates=args.max_candidates,
        per_imaging_gene_limit=args.per_imaging_gene_limit,
        per_imaging_disease_limit=args.per_imaging_disease_limit,
        claim_edge_policy=args.candidate_claim_edge_policy,
        neighborhood=neighborhood,
        terminal_support_weight=args.candidate_terminal_support_weight,
        kge_scorer=kge_scorer,
        kge_path_weight=args.candidate_kge_path_weight,
    )
    recall = recall_at_k(future, candidates, ks=(10, 100, 1000))
    cls = classification_proxy(
        future=future,
        candidates=candidates,
        concepts=concepts,
        gene_imaging=gene_imaging,
        imaging_disease=imaging_disease,
        gene_disease=gene_disease,
        neighborhood=neighborhood,
        historical_direct_pairs=historical_direct_pairs,
        max_examples=args.classification_max_examples,
        seed=args.seed,
        shared_weight=args.classification_shared_weight,
        path_weight=args.classification_path_weight,
        endpoint_weight=args.classification_endpoint_weight,
        kge_weight=args.classification_kge_weight,
        kge_scorer=kge_scorer,
    )

    with (args.output_dir / "candidates_top.json").open("w", encoding="utf-8") as f:
        json.dump([c.to_dict(names) for c in candidates[:1000]], f, indent=2, ensure_ascii=False)
    with (args.output_dir / "future_evaluable_claims_sample.json").open("w", encoding="utf-8") as f:
        json.dump(future[:1000], f, indent=2, ensure_ascii=False)

    manifest = {
        "kg_path": str(args.kg),
        "future_claims_path": str(args.future_claims),
        "output_dir": str(args.output_dir),
        "future_start_year": args.future_start_year,
        "future_end_year": args.future_end_year,
        "freeze_year": args.freeze_year,
        "candidate_count": len(candidates),
        "candidate_claim_edge_policy": args.candidate_claim_edge_policy,
        "candidate_terminal_support_weight": args.candidate_terminal_support_weight,
        "candidate_kge_path_weight": args.candidate_kge_path_weight,
        "kge_checkpoint": str(args.kge_checkpoint) if args.kge_checkpoint else None,
        "gene_imaging_pairs": len(gene_imaging),
        "imaging_disease_pairs": len(imaging_disease),
        "gene_disease_pairs": len(gene_disease),
        "neighborhood_nodes": len(neighborhood.adj),
        "neighborhood_edges_undirected": sum(len(v) for v in neighborhood.adj.values()) // 2,
        "future_pool_stats": future_stats,
        "forward_recall": recall,
        "classification_proxy": cls,
        "score_formula": (
            "classification: normalized weighted sum of shared_neighbor, path_support, endpoint_support, kge_claim; "
            "candidates: 0.45*plausibility + 0.35*mechanism_consistency + 0.20*reproducibility, optionally mixed with KGE path support"
        ),
        "strict_candidate_anchors": args.strict_candidate_anchors,
        "limitations": [
            "No local impact-factor or citation-count metadata was available.",
            "No local retracted or zero-citation negative labels were available.",
            "Classification uses type-preserving corrupted negatives as a proxy.",
            "Novelty is intentionally excluded from scoring.",
        ],
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_report(args.output_dir, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
