"""CLI for KG-based plausibility scoring (Phase A of plan
plausibility-scorer-c.md).

Two subcommands, registered as ``phase4 kge train`` and ``phase4 plausibility``:

    # 1. Train ComplEx on the KG (≈ 5–20 min on CPU, < 1 min on GPU)
    python -m neurooracle.phase4 kge-train \
        --kg neurooracle/data/quick/knowledge_graph.json \
        --output neurooracle/data/quick/kge_complex.pt \
        --report neurooracle/data/quick/kge_eval_report.json

    # 2. Score a hypothesis file (writes kge_score / surprise_gap into metadata)
    python -m neurooracle.phase4 plausibility \
        --input neurooracle/data/quick/hypotheses_baseline.json \
        --kge neurooracle/data/quick/kge_complex.pt \
        --novelty-cache neurooracle/data/quick/novelty_cache.json \
        --output neurooracle/data/quick/hypotheses_plausibility.json \
        --skip-existing
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

from ..hypothesis_engine import Hypothesis
from .complex_scorer import ComplExScorer, TrainConfig
from .plausibility import score_hypothesis
from .triple_loader import load_triples_from_kg, split_triples

logger = logging.getLogger(__name__)


# ── train command ──────────────────────────────────────────────────────

def cmd_kge_train(
    kg_path: str,
    output: str,
    report: Optional[str] = None,
    dim: int = 64,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    negatives_per_pos: int = 10,
    weight_decay: float = 1e-6,
    eval_every: int = 5,
    early_stop_patience: int = 0,
    min_confidence: float = 0.2,
    seed: int = 42,
    device: Optional[str] = None,
) -> None:
    triples, node_domain = load_triples_from_kg(kg_path, min_confidence=min_confidence)
    train, val, test = split_triples(triples, node_domain, seed=seed)

    cfg = TrainConfig(
        embedding_dim=dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        negatives_per_pos=negatives_per_pos,
        weight_decay=weight_decay,
        eval_every=eval_every,
        early_stop_patience=early_stop_patience,
        device=device or ("cuda" if _has_cuda() else "cpu"),
    )
    ckpt_name = Path(output).stem
    scorer = ComplExScorer(dim=dim, device=cfg.device, checkpoint_name=ckpt_name)
    history = scorer.fit(train, val=val, cfg=cfg)
    scorer.save(output)

    test_auroc = scorer.auroc(test) if test else 0.0
    logger.info("test AUROC = %.3f", test_auroc)

    if report:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": output,
                "n_triples": len(triples),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "n_entities": len(scorer.ent2idx),
                "n_relations": len(scorer.rel2idx),
                "config": {
                    "dim": dim, "epochs": epochs, "batch_size": batch_size,
                    "lr": lr, "negatives_per_pos": negatives_per_pos,
                    "min_confidence": min_confidence, "seed": seed,
                    "device": cfg.device,
                },
                "loss_curve": history["loss"],
                "val_auroc_curve": history["val_auroc"],
                "test_auroc": test_auroc,
            }, f, indent=2)
        logger.info("wrote training report → %s", report_path)


# ── plausibility command ──────────────────────────────────────────────

def cmd_plausibility(
    input_path: str,
    kge_checkpoint: str,
    output: Optional[str] = None,
    novelty_cache: Optional[str] = None,
    skip_existing: bool = True,
    no_pubmed: bool = False,
    top: int = 0,
    device: Optional[str] = None,
    kg_path: Optional[str] = None,
) -> None:
    in_path = Path(input_path)
    out_path = Path(output) if output else in_path

    with in_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Schema flexibility: a list or a dict with "hypotheses" key.
    hyp_list_raw = raw if isinstance(raw, list) else raw.get("hypotheses", [])
    hypotheses = [Hypothesis.from_dict(d) for d in hyp_list_raw]
    logger.info("loaded %d hypotheses from %s", len(hypotheses), in_path)

    if top > 0:
        hypotheses_sorted = sorted(
            hypotheses, key=lambda h: h.composite_score, reverse=True
        )
        hypotheses = hypotheses_sorted[:top]
        logger.info("scoring top-%d by composite_score", top)

    scorer = ComplExScorer.load(kge_checkpoint, device=device)
    pubmed_fn = None if no_pubmed else _make_pubmed_count_fn(novelty_cache)

    # Load KG concepts + degrees once for the path-specificity filter. This
    # lets the surprise_gap output flag hub-umbrella / CLM-noise contamination
    # without re-traversing the graph for every hypothesis.
    concepts: dict = {}
    degrees: dict[str, int] = {}
    if kg_path:
        from .specificity import build_degree_map
        kg_p = Path(kg_path)
        if kg_p.exists():
            logger.info("loading KG concepts + degrees for specificity filter")
            with kg_p.open(encoding="utf-8") as f:
                kg = json.load(f)
            concepts = kg.get("concepts") or {}
            degrees = build_degree_map(kg.get("edges") or [])
            logger.info("specificity filter ready: %d concepts, %d degree entries",
                        len(concepts), len(degrees))
        else:
            logger.warning("KG path %s not found; specificity filter will run "
                           "with id-only rules", kg_p)

    n_scored, n_skipped = 0, 0
    gaps: list[float] = []
    locals_: list[float] = []
    spec_scores: list[float] = []
    for i, h in enumerate(hypotheses):
        result = score_hypothesis(h, scorer, pubmed_fn, skip_existing=skip_existing,
                                  concepts=concepts, degrees=degrees)
        if result["skipped"]:
            n_skipped += 1
        else:
            n_scored += 1
            if result["kge_score"] is not None:
                locals_.append(result["kge_score"])
            if result["surprise_gap"] is not None:
                gaps.append(result["surprise_gap"])
            if result.get("specificity_score") is not None:
                spec_scores.append(result["specificity_score"])
        if (i + 1) % 25 == 0:
            logger.info("  ...%d/%d", i + 1, len(hypotheses))

    logger.info("scored %d new, skipped %d", n_scored, n_skipped)

    # Persist back into the original schema shape
    if isinstance(raw, list):
        out_raw = [h.to_dict() for h in hypotheses]
    else:
        out_raw = dict(raw)
        out_raw["hypotheses"] = [h.to_dict() for h in hypotheses]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_raw, f, indent=2, ensure_ascii=False)
    logger.info("wrote → %s", out_path)

    if locals_:
        _print_distribution("kge_score", locals_)
    if gaps:
        _print_distribution("surprise_gap", gaps)
    if spec_scores:
        _print_distribution("specificity_score", spec_scores)
        n_clean = sum(1 for s in spec_scores if s == 1.0)
        logger.info("clean paths (specificity=1.0): %d / %d (%.1f%%)",
                    n_clean, len(spec_scores), 100 * n_clean / len(spec_scores))


# ── helpers ────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _make_pubmed_count_fn(novelty_cache: Optional[str]):
    """Build a cached pubmed-count function that piggy-backs on the cache
    used by NoveltyChecker so we don't re-issue identical queries."""
    import requests

    PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    cache: dict[str, dict] = {}
    cache_path = Path(novelty_cache) if novelty_cache else None
    if cache_path and cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as f:
                cache = json.load(f) or {}
        except Exception as exc:
            logger.warning("could not load novelty cache: %s", exc)
            cache = {}

    def _persist():
        if not cache_path:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as exc:
            logger.warning("could not persist novelty cache: %s", exc)

    def pubmed_count(query: str) -> int:
        key = query.lower().strip()
        if key in cache and "pubmed_hits" in cache[key]:
            return int(cache[key]["pubmed_hits"])
        try:
            resp = requests.get(
                PUBMED_ESEARCH,
                params={
                    "db": "pubmed",
                    "term": query,
                    "rettype": "json",
                    "retmode": "json",
                    "retmax": 0,
                },
                timeout=10,
            )
            resp.raise_for_status()
            count = int(resp.json().get("esearchresult", {}).get("count", 0))
        except Exception as exc:
            logger.debug("pubmed query failed (%s): %s", query[:60], exc)
            count = 0
        cache.setdefault(key, {})["pubmed_hits"] = count
        # Persist every 20 new queries to avoid losing the cache on Ctrl+C
        if len(cache) % 20 == 0:
            _persist()
        return count

    return pubmed_count


def _print_distribution(label: str, xs: list[float]) -> None:
    if not xs:
        return
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(p * n)))
        return xs_sorted[idx]

    logger.info(
        "%s distribution n=%d  min=%.3f  p25=%.3f  p50=%.3f  p75=%.3f  p95=%.3f  max=%.3f  mean=%.3f",
        label, n, xs_sorted[0], pct(0.25), pct(0.50), pct(0.75), pct(0.95),
        xs_sorted[-1], statistics.fmean(xs_sorted),
    )
