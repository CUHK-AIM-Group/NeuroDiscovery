from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from case3_hindcasting_eval import (
    NeighborhoodGraph,
    RankedCandidate,
    _auc,
    _auprc,
    _hybrid_pair_score,
    _kge_best_pair_score,
    _role_for_node,
    _role_pair_key,
    _ordered_pair_for_key,
    _build_score_indexes,
    build_neighborhood_graph,
    build_pair_evidence,
    generate_candidates,
    load_future_claim_pairs,
    load_kg_index,
    recall_at_k,
)
from neurooracle.src.kge.complex_scorer import ComplExScorer, TrainConfig
from neurooracle.src.kge.triple_loader import load_triples_from_kg, split_triples


WINDOWS = (
    (2016, 2017, 2021),
    (2018, 2019, 2023),
    (2020, 2021, 2025),
    (2022, 2023, 2025),
)
METHODS = (
    "random_walk",
    "llm_brainstorm",
    "graph_degree",
    "neurodiscovery",
    "neurodiscovery_kge",
    "neurodiscovery_debate",
)
METHOD_LABELS = {
    "random_walk": "Random walk",
    "llm_brainstorm": "LLM BrainSTORM",
    "graph_degree": "Graph degree",
    "neurodiscovery": "ND-lite",
    "neurodiscovery_kge": "ND-KGE",
    "neurodiscovery_debate": "ND-KGE + local debate",
}
TOP_KS = (10, 20, 50, 100, 200, 300, 500, 750, 1000)


def parse_window(raw: str) -> tuple[int, int, int]:
    parts = raw.replace(",", ":").split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"window must be freeze:start:end, got {raw!r}")
    try:
        freeze, start, end = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"window years must be integers, got {raw!r}") from exc
    if not (freeze < start <= end):
        raise argparse.ArgumentTypeError(f"window must satisfy freeze < start <= end, got {raw!r}")
    return freeze, start, end


LLM_PRIOR_TERMS = {
    "gene": {
        "apoe": 0.22,
        "bdnf": 0.18,
        "drd": 0.16,
        "comt": 0.15,
        "slc6a": 0.14,
        "serotonin": 0.12,
        "dopamine": 0.12,
        "inflammation": 0.10,
        "immune": 0.10,
    },
    "imaging": {
        "hippocamp": 0.22,
        "amygdala": 0.18,
        "prefrontal": 0.18,
        "cingulate": 0.16,
        "cortical thickness": 0.16,
        "connectivity": 0.15,
        "default mode": 0.13,
        "striatal": 0.12,
        "white matter": 0.10,
        "amyloid": 0.16,
        "tau": 0.15,
        "fdg": 0.12,
    },
    "disease": {
        "alzheimer": 0.22,
        "cognitive": 0.18,
        "schizophrenia": 0.18,
        "psychosis": 0.16,
        "depression": 0.16,
        "bipolar": 0.13,
        "adhd": 0.12,
        "autism": 0.12,
        "parkinson": 0.12,
    },
    "predicate": {
        "predict": 0.15,
        "biomarker": 0.12,
        "associated": 0.08,
        "distinguish": 0.10,
        "correlat": 0.08,
    },
}


def _norm_log(value: float, denom: float) -> float:
    return min(1.0, math.log1p(max(value, 0.0)) / math.log(denom))


def _name(concepts: dict[str, dict[str, Any]], node_id: str) -> str:
    node = concepts.get(node_id) or {}
    return str(node.get("preferred_name") or node_id)


def _keyword_score(text: str, group: str) -> float:
    low = text.casefold()
    return sum(weight for term, weight in LLM_PRIOR_TERMS[group].items() if term in low)


def llm_style_pair_score(
    concepts: dict[str, dict[str, Any]],
    key: str,
    pair: tuple[str, str],
    predicate: str = "",
) -> float:
    if key == "gene_disease":
        gene, disease = pair
        score = 0.42
        score += _keyword_score(_name(concepts, gene), "gene")
        score += _keyword_score(_name(concepts, disease), "disease")
    elif key == "gene_imaging":
        gene, imaging = pair
        score = 0.40
        score += _keyword_score(_name(concepts, gene), "gene")
        score += _keyword_score(_name(concepts, imaging), "imaging")
    elif key == "imaging_disease":
        imaging, disease = pair
        score = 0.40
        score += _keyword_score(_name(concepts, imaging), "imaging")
        score += _keyword_score(_name(concepts, disease), "disease")
    else:
        score = 0.0
    score += _keyword_score(predicate, "predicate")
    return min(1.0, score)


def graph_degree_pair_score(neighborhood: NeighborhoodGraph, pair: tuple[str, str]) -> float:
    left, right = pair
    return min(1.0, math.sqrt(math.log1p(neighborhood.degree(left)) * math.log1p(neighborhood.degree(right))) / math.log(5000))


def neurodiscovery_lite_score(cand: RankedCandidate) -> float:
    base = (
        0.45 * cand.plausibility
        + 0.35 * cand.mechanism_consistency
        + 0.20 * cand.reproducibility
    )
    return min(1.0, 0.80 * base + 0.20 * cand.terminal_support)


def neurodiscovery_kge_score(
    cand: RankedCandidate,
    kge_path_weight: float,
    kge_norm_score: float | None = None,
) -> float:
    lite = neurodiscovery_lite_score(cand)
    kge_score = cand.kge_path_score if kge_norm_score is None else kge_norm_score
    if kge_score <= 0:
        return lite
    return min(1.0, (1.0 - kge_path_weight) * lite + kge_path_weight * kge_score)


def local_debate_score(
    cand: RankedCandidate,
    concepts: dict[str, dict[str, Any]],
    neighborhood: NeighborhoodGraph,
    kge_path_weight: float,
    kge_norm_score: float | None = None,
) -> float:
    """Deterministic host-agent proxy for three critic perspectives.

    This is intentionally transparent: it does not call an LLM API. It mimics
    three reviewers whose comments are grounded in fields already computed for
    each candidate.
    """
    nd_kge = neurodiscovery_kge_score(cand, kge_path_weight, kge_norm_score)
    stat_agent = min(
        1.0,
        0.40 * cand.reproducibility
        + 0.25 * min(1.0, math.log1p(cand.gene_imaging_pmids) / math.log(8))
        + 0.25 * min(1.0, math.log1p(cand.imaging_disease_pmids) / math.log(8))
        + 0.10 * (cand.claim_backed_segments / 2.0),
    )
    bio_agent = min(
        1.0,
        0.35 * cand.mechanism_consistency
        + 0.25 * cand.plausibility
        + 0.20 * cand.terminal_support
        + 0.20 * max(cand.kge_path_score if kge_norm_score is None else kge_norm_score, 0.0),
    )
    names = " ".join(
        _name(concepts, node_id).casefold()
        for node_id in (cand.gene_id, cand.imaging_id, cand.disease_id)
    )
    generic_penalty = 0.0
    for term in ("disorder", "disease", "symptom", "brain", "imaging", "marker"):
        generic_penalty += 0.025 if term in names else 0.0
    hub_raw = max(
        neighborhood.degree(cand.gene_id),
        neighborhood.degree(cand.imaging_id),
        neighborhood.degree(cand.disease_id),
    )
    hub_penalty = 0.18 * _norm_log(hub_raw, 10000)
    method_agent = max(
        0.0,
        min(
            1.0,
            0.35 * cand.mechanism_consistency
            + 0.25 * (cand.claim_backed_segments / 2.0)
            + 0.20 * cand.reproducibility
            + 0.20 * (1.0 - hub_penalty)
            - generic_penalty,
        ),
    )
    critic_consensus = 0.34 * stat_agent + 0.38 * bio_agent + 0.28 * method_agent
    return min(1.0, 0.62 * nd_kge + 0.38 * critic_consensus)


def candidate_score(
    method: str,
    cand: RankedCandidate,
    concepts: dict[str, dict[str, Any]],
    neighborhood: NeighborhoodGraph,
    kge_path_weight: float = 0.25,
    kge_norm_score: float | None = None,
) -> float:
    if method == "neurodiscovery":
        return neurodiscovery_lite_score(cand)
    if method == "neurodiscovery_kge":
        return neurodiscovery_kge_score(cand, kge_path_weight, kge_norm_score)
    if method == "neurodiscovery_debate":
        return local_debate_score(cand, concepts, neighborhood, kge_path_weight, kge_norm_score)
    if method == "graph_degree":
        gene = _norm_log(neighborhood.degree(cand.gene_id), 5000)
        imaging = _norm_log(neighborhood.degree(cand.imaging_id), 5000)
        disease = _norm_log(neighborhood.degree(cand.disease_id), 5000)
        terminal = graph_degree_pair_score(neighborhood, (cand.gene_id, cand.disease_id))
        return min(1.0, 0.25 * gene + 0.25 * imaging + 0.25 * disease + 0.25 * terminal)
    if method == "llm_brainstorm":
        gi = llm_style_pair_score(concepts, "gene_imaging", (cand.gene_id, cand.imaging_id))
        imd = llm_style_pair_score(concepts, "imaging_disease", (cand.imaging_id, cand.disease_id))
        gd = llm_style_pair_score(concepts, "gene_disease", (cand.gene_id, cand.disease_id))
        return min(1.0, 0.38 * gi + 0.38 * imd + 0.24 * gd)
    raise KeyError(method)


def rerank_candidates(
    method: str,
    candidates: list[RankedCandidate],
    concepts: dict[str, dict[str, Any]],
    neighborhood: NeighborhoodGraph,
    rng: random.Random,
    kge_path_weight: float,
    kge_rank_mode: str,
) -> list[RankedCandidate]:
    if method == "random_walk":
        scored = [(rng.random(), i, c) for i, c in enumerate(candidates)]
        scored.sort(reverse=True)
        return [replace(c, score=s, sort_score=s) for s, _i, c in scored]
    kge_norm: dict[int, float] = {}
    if method in {"neurodiscovery_kge", "neurodiscovery_debate"} and kge_rank_mode != "none":
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (item[1].kge_path_score, item[1].plausibility, item[1].reproducibility),
        )
        denom = max(1, len(ordered) - 1)
        for rank, (idx, _cand) in enumerate(ordered):
            pct = rank / denom
            kge_norm[idx] = 1.0 - pct if kge_rank_mode == "low-rank" else pct
    rescored = []
    for i, cand in enumerate(candidates):
        score = candidate_score(
            method,
            cand,
            concepts,
            neighborhood,
            kge_path_weight,
            kge_norm_score=kge_norm.get(i),
        )
        rescored.append((score, -i, replace(cand, score=score, sort_score=score)))
    rescored.sort(reverse=True)
    return [c for _score, _i, c in rescored]


def make_negatives(
    positives: list[dict[str, Any]],
    concepts: dict[str, dict[str, Any]],
    historical_direct_pairs: set[tuple[str, str]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    role_nodes: dict[str, list[str]] = defaultdict(list)
    for nid, node in concepts.items():
        role = _role_for_node(node)
        if role in {"gene", "imaging", "disease"}:
            role_nodes[role].append(nid)
    negatives: list[dict[str, Any]] = []
    for row in positives:
        left, _right = tuple(row["pair"])
        left_role, right_role = row["key"].split("_", 1)
        pool = role_nodes[right_role]
        for _ in range(100):
            repl = rng.choice(pool)
            pair = (left, repl)
            if pair == tuple(row["pair"]) or pair in historical_direct_pairs:
                continue
            neg = dict(row)
            neg["pair"] = pair
            neg["claim_id"] = f"DECOY:{row.get('claim_id')}"
            neg["object_id"] = repl
            neg["object_name"] = _name(concepts, repl)
            neg["object_role"] = right_role
            negatives.append(neg)
            break
    return negatives


def pair_scores(
    rows: list[dict[str, Any]],
    method: str,
    concepts: dict[str, dict[str, Any]],
    gene_imaging,
    imaging_disease,
    gene_disease,
    neighborhood: NeighborhoodGraph,
    rng: random.Random,
    kge_scorer: ComplExScorer | None,
    kge_path_weight: float,
) -> list[float]:
    indexes = _build_score_indexes(gene_imaging, imaging_disease, gene_disease)
    scores: list[float] = []
    for row in rows:
        pair = tuple(row["pair"])
        if method == "random_walk":
            scores.append(rng.random())
        elif method == "llm_brainstorm":
            scores.append(llm_style_pair_score(concepts, row["key"], pair, str(row.get("predicate") or "")))
        elif method == "graph_degree":
            scores.append(graph_degree_pair_score(neighborhood, pair))
        elif method in {"neurodiscovery", "neurodiscovery_kge", "neurodiscovery_debate"}:
            if method == "neurodiscovery":
                shared_weight, path_weight, endpoint_weight, kge_weight = 1.0, 0.0, 0.0, 0.0
            elif method == "neurodiscovery_kge":
                shared_weight, path_weight, endpoint_weight, kge_weight = 0.55, 0.15, 0.05, 0.25
            else:
                shared_weight, path_weight, endpoint_weight, kge_weight = 0.45, 0.20, 0.10, 0.25
            score = _hybrid_pair_score(
                row["key"],
                pair,
                row,
                concepts,
                gene_imaging,
                imaging_disease,
                gene_disease,
                indexes,
                neighborhood,
                shared_weight=shared_weight,
                path_weight=path_weight,
                endpoint_weight=endpoint_weight,
                kge_weight=kge_weight if kge_scorer is not None else 0.0,
                kge_scorer=kge_scorer,
            )
            if method == "neurodiscovery_debate":
                endpoint = graph_degree_pair_score(neighborhood, pair)
                scores.append(min(1.0, 0.75 * score["hybrid"] + 0.25 * endpoint))
            else:
                scores.append(score["hybrid"])
        else:
            raise KeyError(method)
    return scores


def summarize(values: list[float]) -> tuple[float, float, float]:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return vals[idx]
    return sum(vals) / len(vals), q(0.025), q(0.975)


def ensure_kge_scorer(
    kg_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    negatives_per_pos: int,
    min_confidence: float,
    seed: int,
    device: str | None,
) -> ComplExScorer:
    if checkpoint_path.exists():
        return ComplExScorer.load(checkpoint_path, device=device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    triples, node_domain = load_triples_from_kg(kg_path, min_confidence=min_confidence)
    train, val, test = split_triples(triples, node_domain, seed=seed)
    cfg = TrainConfig(
        embedding_dim=dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        negatives_per_pos=negatives_per_pos,
        eval_every=max(1, min(epochs, 5)),
        early_stop_patience=2 if epochs >= 8 else 0,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    scorer = ComplExScorer(dim=dim, device=cfg.device, checkpoint_name=checkpoint_path.stem)
    history = scorer.fit(train, val=val, cfg=cfg)
    scorer.save(checkpoint_path)
    test_auroc = scorer.auroc(test) if test else 0.0
    report_path.write_text(
        json.dumps(
            {
                "kg_path": str(kg_path),
                "checkpoint": str(checkpoint_path),
                "n_triples": len(triples),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "dim": dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "negatives_per_pos": negatives_per_pos,
                "min_confidence": min_confidence,
                "seed": seed,
                "device": cfg.device,
                "loss_curve": history.get("loss", []),
                "val_auroc_curve": history.get("val_auroc", []),
                "test_auroc": test_auroc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return scorer


def evaluate_window(
    kg_path: Path,
    claims_path: Path,
    freeze_year: int,
    future_start: int,
    future_end: int,
    max_candidates: int,
    random_trials: int,
    seed: int,
    candidate_universe: str,
    kge_scorer: ComplExScorer | None,
    kge_path_weight: float,
    kge_rank_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    concepts, edges, names = load_kg_index(kg_path)
    gene_imaging, imaging_disease, gene_disease, historical_direct_pairs = build_pair_evidence(
        concepts, edges, strict_candidate_anchors=True
    )
    neighborhood = build_neighborhood_graph(concepts, edges, strict_candidate_anchors=True)
    future, future_stats = load_future_claim_pairs(
        claims_path=claims_path,
        concepts=concepts,
        start_year=future_start,
        end_year=future_end,
        historical_direct_pairs=historical_direct_pairs,
    )
    if candidate_universe == "full-chain":
        candidate_limit = 1_000_000_000 if max_candidates <= 0 else max_candidates
        per_imaging_gene_limit = 1_000_000
        per_imaging_disease_limit = 1_000_000
    else:
        candidate_limit = max_candidates
        per_imaging_gene_limit = 80
        per_imaging_disease_limit = 80

    candidates = generate_candidates(
        concepts=concepts,
        gene_imaging=gene_imaging,
        imaging_disease=imaging_disease,
        max_candidates=candidate_limit,
        per_imaging_gene_limit=per_imaging_gene_limit,
        per_imaging_disease_limit=per_imaging_disease_limit,
        claim_edge_policy="require-any",
        neighborhood=neighborhood,
        terminal_support_weight=0.20,
        kge_scorer=kge_scorer,
        kge_path_weight=0.0,
    )
    negatives = make_negatives(future, concepts, historical_direct_pairs, seed + freeze_year)
    paired_rows = future + negatives
    labels = [1] * len(future) + [0] * len(negatives)

    baseline_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    for method in METHODS:
        metric_trials: list[dict[str, float]] = []
        topk_trials: list[dict[str, float]] = []
        n_trials = random_trials if method == "random_walk" else 1
        for trial in range(n_trials):
            rng = random.Random(seed + freeze_year * 1009 + trial)
            scores = pair_scores(
                paired_rows,
                method,
                concepts,
                gene_imaging,
                imaging_disease,
                gene_disease,
                neighborhood,
                rng,
                kge_scorer,
                kge_path_weight,
            )
            metric_trials.append({
                "auc": _auc(labels, scores) or 0.0,
                "auprc": _auprc(labels, scores) or 0.0,
            })
            ranked = rerank_candidates(
                method,
                candidates,
                concepts,
                neighborhood,
                rng,
                kge_path_weight,
                kge_rank_mode,
            )
            recall = recall_at_k(future, ranked, TOP_KS)
            topk_trials.append({f"hits@{k}": float(recall[f"hits@{k}"]) for k in TOP_KS})
            topk_trials[-1].update({f"recall@{k}": float(recall[f"recall@{k}"]) for k in TOP_KS})
            if method in {"neurodiscovery", "neurodiscovery_kge", "neurodiscovery_debate"} and trial == 0:
                candidate_keys = {}
                for rank, cand in enumerate(ranked, start=1):
                    for key, pair in (
                        ("gene_disease", (cand.gene_id, cand.disease_id)),
                        ("gene_imaging", (cand.gene_id, cand.imaging_id)),
                        ("imaging_disease", (cand.imaging_id, cand.disease_id)),
                    ):
                        candidate_keys.setdefault((key, pair), (rank, cand))
                for row in future:
                    match = candidate_keys.get((row["key"], tuple(row["pair"])))
                    if not match:
                        continue
                    rank, cand = match
                    if rank > 1000:
                        continue
                    hit_rows.append({
                        "freeze_year": freeze_year,
                        "future_year": row["year"],
                        "lead_time": int(row["year"]) - freeze_year,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "rank": rank,
                        "score": cand.score,
                        "kge_path_score": cand.kge_path_score,
                        "plausibility": cand.plausibility,
                        "mechanism_consistency": cand.mechanism_consistency,
                        "reproducibility": cand.reproducibility,
                        "terminal_support": cand.terminal_support,
                        "claim_id": row.get("claim_id"),
                        "pmid": row.get("pmid"),
                        "hit_key": row["key"],
                        "gene_name": names.get(cand.gene_id, cand.gene_id),
                        "imaging_name": names.get(cand.imaging_id, cand.imaging_id),
                        "disease_name": names.get(cand.disease_id, cand.disease_id),
                        "future_subject": row.get("subject_name"),
                        "future_object": row.get("object_name"),
                        "predicate": row.get("predicate"),
                    })

        for metric in ("auc", "auprc"):
            mean, lo, hi = summarize([r[metric] for r in metric_trials])
            baseline_rows.append({
                "freeze_year": freeze_year,
                "future_window": f"{future_start}-{future_end}",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "metric": metric,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n_trials": n_trials,
                "candidate_universe": candidate_universe,
                "candidate_count": len(candidates),
                "kge_rank_mode": kge_rank_mode,
                "future_evaluable_novel": len(future),
                **future_stats,
            })
        for k in TOP_KS:
            mean, lo, hi = summarize([r[f"hits@{k}"] for r in topk_trials])
            topk_rows.append({
                "freeze_year": freeze_year,
                "future_window": f"{future_start}-{future_end}",
                "method": method,
                "method_label": METHOD_LABELS[method],
                "k": k,
                "hits_mean": mean,
                "hits_lo": lo,
                "hits_hi": hi,
                "n_trials": n_trials,
                "candidate_universe": candidate_universe,
                "candidate_count": len(candidates),
                "kge_rank_mode": kge_rank_mode,
            })
    return baseline_rows, topk_rows, hit_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Case Study 3 GENE-IMAGING-DISEASE baseline comparison.")
    parser.add_argument("--snapshot-root", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/snapshots_full_v2_case1_case2_v1"))
    parser.add_argument("--snapshot-template", type=str, default="kg_{freeze}_from_full_snapshot_v1")
    parser.add_argument("--future-claims", type=Path, default=Path("neurooracle/data/full_v2/extracted_claims.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/gid_baselines_full_v2_case1_case2_v1"))
    parser.add_argument("--candidate-universe", choices=("full-chain", "neurodiscovery-top"), default="full-chain")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means no cap for --candidate-universe full-chain.")
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260617)
    parser.add_argument("--kge-dir", type=Path, default=Path("neurooracle/data/cs_runs/case3_hindcasting/kge_cache_full_v2_case1_case2"))
    parser.add_argument("--kge-dim", type=int, default=64)
    parser.add_argument("--kge-epochs", type=int, default=12)
    parser.add_argument("--kge-batch-size", type=int, default=4096)
    parser.add_argument("--kge-lr", type=float, default=1e-3)
    parser.add_argument("--kge-negatives-per-pos", type=int, default=5)
    parser.add_argument("--kge-min-confidence", type=float, default=0.2)
    parser.add_argument("--kge-path-weight", type=float, default=0.25)
    parser.add_argument("--kge-rank-mode", choices=("none", "high-rank", "low-rank"), default="none")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--windows",
        nargs="*",
        type=parse_window,
        default=list(WINDOWS),
        help="Hindcasting windows as freeze:start:end, e.g. 2020:2021:2025.",
    )
    args = parser.parse_args()

    all_baselines: list[dict[str, Any]] = []
    all_topk: list[dict[str, Any]] = []
    all_hits: list[dict[str, Any]] = []
    for freeze, start, end in args.windows:
        kg = args.snapshot_root / args.snapshot_template.format(freeze=freeze) / "knowledge_graph.json"
        print(f"[window] {freeze}->{start}-{end}", flush=True)
        kge_checkpoint = args.kge_dir / f"kge_{freeze}_complex_dim{args.kge_dim}_ep{args.kge_epochs}.pt"
        kge_report = args.kge_dir / f"kge_{freeze}_complex_dim{args.kge_dim}_ep{args.kge_epochs}.json"
        kge_scorer = ensure_kge_scorer(
            kg_path=kg,
            checkpoint_path=kge_checkpoint,
            report_path=kge_report,
            dim=args.kge_dim,
            epochs=args.kge_epochs,
            batch_size=args.kge_batch_size,
            lr=args.kge_lr,
            negatives_per_pos=args.kge_negatives_per_pos,
            min_confidence=args.kge_min_confidence,
            seed=args.seed + freeze,
            device=args.device,
        )
        baseline_rows, topk_rows, hit_rows = evaluate_window(
            kg_path=kg,
            claims_path=args.future_claims,
            freeze_year=freeze,
            future_start=start,
            future_end=end,
            max_candidates=args.max_candidates,
            random_trials=args.random_trials,
            seed=args.seed,
            candidate_universe=args.candidate_universe,
            kge_scorer=kge_scorer,
            kge_path_weight=args.kge_path_weight,
            kge_rank_mode=args.kge_rank_mode,
        )
        all_baselines.extend(baseline_rows)
        all_topk.extend(topk_rows)
        all_hits.extend(hit_rows)

    write_csv(args.output_dir / "baseline_metrics_by_window.csv", all_baselines)
    write_csv(args.output_dir / "topk_hits_by_window.csv", all_topk)
    write_csv(args.output_dir / "neurodiscovery_recovered_top1000_hits.csv", sorted(all_hits, key=lambda r: (r["rank"], -r["score"])))
    manifest = {
        "methods": METHOD_LABELS,
        "windows": args.windows,
        "future_claims": str(args.future_claims),
        "snapshot_root": str(args.snapshot_root),
        "snapshot_template": args.snapshot_template,
        "candidate_universe": args.candidate_universe,
        "max_candidates": args.max_candidates,
        "random_trials": args.random_trials,
        "kge_dir": str(args.kge_dir),
        "kge_dim": args.kge_dim,
        "kge_epochs": args.kge_epochs,
        "kge_path_weight": args.kge_path_weight,
        "kge_rank_mode": args.kge_rank_mode,
        "llm_brainstorm_note": (
            "Deterministic LLM-style brainstorm surrogate based on broad biomedical "
            "priors over gene, imaging, disease, and predicate names. Replace with "
            "true LLM-generated/ranked candidates when available."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "baseline_rows": len(all_baselines), "topk_rows": len(all_topk), "hit_rows": len(all_hits)}, indent=2))


if __name__ == "__main__":
    main()
