"""ComplEx KG embedding model + scorer.

A small, dependency-light implementation in plain PyTorch. Trains in 5–20
minutes on a CPU for a 80k-edge graph at dim=64. Used as the Phase A baseline
in plans/plausibility-scorer-c.md; can be swapped for ULTRA / Gamma later.

Reference: Trouillon et al., 'Complex Embeddings for Simple Link Prediction',
ICML 2016. We use the standard score:

    f(s, p, o) = Re(<e_s, w_p, conj(e_o)>)

trained with binary cross-entropy and 1-vs-N negative sampling.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .base import Scorer
from .triple_loader import Triple

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    embedding_dim: int = 64
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    negatives_per_pos: int = 10
    weight_decay: float = 1e-6
    eval_every: int = 5
    early_stop_patience: int = 0  # 0 = disabled; otherwise # of evals without val improvement
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _ComplEx(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int):
        super().__init__()
        self.dim = dim
        self.ent_re = nn.Embedding(n_entities, dim)
        self.ent_im = nn.Embedding(n_entities, dim)
        self.rel_re = nn.Embedding(n_relations, dim)
        self.rel_im = nn.Embedding(n_relations, dim)
        for emb in (self.ent_re, self.ent_im, self.rel_re, self.rel_im):
            nn.init.xavier_uniform_(emb.weight)

    def score(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        s_re, s_im = self.ent_re(s), self.ent_im(s)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        o_re, o_im = self.ent_re(o), self.ent_im(o)
        # Re(<s, r, conj(o)>)  for complex vectors
        score = (
            (s_re * r_re * o_re).sum(dim=-1)
            + (s_im * r_re * o_im).sum(dim=-1)
            + (s_re * r_im * o_im).sum(dim=-1)
            - (s_im * r_im * o_re).sum(dim=-1)
        )
        return score


class ComplExScorer(Scorer):
    """Train + score with a ComplEx model."""

    def __init__(self, dim: int = 64, device: Optional[str] = None,
                 checkpoint_name: str = "complex-v1"):
        self.dim = dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_name = checkpoint_name
        self.ent2idx: dict[str, int] = {}
        self.rel2idx: dict[str, int] = {}
        self.model: Optional[_ComplEx] = None
        # Cache of triples seen in train, used for filtered ranking
        self._train_set: set[tuple[int, int, int]] = set()

    @property
    def name(self) -> str:
        return self.checkpoint_name

    # ── training ────────────────────────────────────────────────────────

    def fit(self, train: list[Triple], val: list[Triple] | None = None,
            cfg: TrainConfig | None = None) -> dict:
        cfg = cfg or TrainConfig(embedding_dim=self.dim)
        self.dim = cfg.embedding_dim

        ents, rels = set(), set()
        for t in train:
            ents.add(t.source_id); ents.add(t.target_id); rels.add(t.relation_type)
        if val:
            for t in val:
                ents.add(t.source_id); ents.add(t.target_id); rels.add(t.relation_type)
        self.ent2idx = {e: i for i, e in enumerate(sorted(ents))}
        self.rel2idx = {r: i for i, r in enumerate(sorted(rels))}

        n_ent, n_rel = len(self.ent2idx), len(self.rel2idx)
        logger.info("ComplEx training: %d entities, %d relations, dim=%d, device=%s",
                    n_ent, n_rel, cfg.embedding_dim, cfg.device)

        self.model = _ComplEx(n_ent, n_rel, cfg.embedding_dim).to(cfg.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)

        train_ids = torch.tensor(
            [[self.ent2idx[t.source_id], self.rel2idx[t.relation_type], self.ent2idx[t.target_id]]
             for t in train],
            dtype=torch.long, device=cfg.device,
        )
        self._train_set = {tuple(row.tolist()) for row in train_ids}

        history = {"loss": [], "val_auroc": []}
        n = train_ids.shape[0]

        best_auroc = -1.0
        best_state = None
        best_epoch = 0
        evals_since_best = 0

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            perm = torch.randperm(n, device=cfg.device)
            total_loss = 0.0
            t0 = time.time()
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                pos = train_ids[idx]  # (B, 3)
                B = pos.shape[0]
                # Corrupt either head or tail uniformly at random
                neg = pos.repeat_interleave(cfg.negatives_per_pos, dim=0).clone()
                rand_ents = torch.randint(0, n_ent, (neg.shape[0],), device=cfg.device)
                corrupt_head = torch.rand(neg.shape[0], device=cfg.device) < 0.5
                neg[corrupt_head, 0] = rand_ents[corrupt_head]
                neg[~corrupt_head, 2] = rand_ents[~corrupt_head]

                pos_score = self.model.score(pos[:, 0], pos[:, 1], pos[:, 2])
                neg_score = self.model.score(neg[:, 0], neg[:, 1], neg[:, 2])

                pos_target = torch.ones_like(pos_score)
                neg_target = torch.zeros_like(neg_score)

                loss = (
                    nn.functional.binary_cross_entropy_with_logits(pos_score, pos_target)
                    + nn.functional.binary_cross_entropy_with_logits(neg_score, neg_target)
                )

                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item() * B
            avg_loss = total_loss / max(1, n)
            history["loss"].append(avg_loss)
            elapsed = time.time() - t0

            log_line = f"epoch {epoch:03d}  loss={avg_loss:.4f}  ({elapsed:.1f}s)"
            if val and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
                auroc = self.auroc(val)
                history["val_auroc"].append((epoch, auroc))
                log_line += f"  val_auroc={auroc:.3f}"
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    evals_since_best = 0
                    log_line += "  (best)"
                else:
                    evals_since_best += 1
                    if cfg.early_stop_patience > 0 and evals_since_best >= cfg.early_stop_patience:
                        logger.info(log_line)
                        logger.info(
                            "early stop: %d evals without val improvement; best=%.3f at epoch %d",
                            evals_since_best, best_auroc, best_epoch,
                        )
                        break
            logger.info(log_line)

        # Restore best-on-val checkpoint so test/inference use the un-overfit model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("restored best val checkpoint from epoch %d (val_auroc=%.3f)",
                        best_epoch, best_auroc)

        return history

    # ── inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def score_triple(self, source_id: str, relation_type: str, target_id: str) -> float:
        if self.model is None:
            raise RuntimeError("Scorer is not trained or loaded.")
        s = self.ent2idx.get(source_id)
        r = self.rel2idx.get(relation_type)
        o = self.ent2idx.get(target_id)
        if s is None or r is None or o is None:
            # Out-of-vocab → neutral prior
            return 0.5
        device = next(self.model.parameters()).device
        s_t = torch.tensor([s], device=device)
        r_t = torch.tensor([r], device=device)
        o_t = torch.tensor([o], device=device)
        logit = self.model.score(s_t, r_t, o_t).item()
        return float(torch.sigmoid(torch.tensor(logit)).item())

    @torch.no_grad()
    def score_batch(self, triples: list[tuple[str, str, str]]) -> list[float]:
        if self.model is None:
            raise RuntimeError("Scorer is not trained or loaded.")
        if not triples:
            return []
        device = next(self.model.parameters()).device
        s_idx, r_idx, o_idx, oov_mask = [], [], [], []
        for s, r, o in triples:
            si, ri, oi = self.ent2idx.get(s), self.rel2idx.get(r), self.ent2idx.get(o)
            if si is None or ri is None or oi is None:
                s_idx.append(0); r_idx.append(0); o_idx.append(0)
                oov_mask.append(True)
            else:
                s_idx.append(si); r_idx.append(ri); o_idx.append(oi)
                oov_mask.append(False)
        s_t = torch.tensor(s_idx, device=device)
        r_t = torch.tensor(r_idx, device=device)
        o_t = torch.tensor(o_idx, device=device)
        logits = self.model.score(s_t, r_t, o_t)
        probs = torch.sigmoid(logits).tolist()
        return [0.5 if oov else p for p, oov in zip(probs, oov_mask)]

    @torch.no_grad()
    def auroc(self, eval_triples: list[Triple], n_negatives: int = 5) -> float:
        """Quick AUROC on positive vs random-corruption negatives."""
        if not eval_triples or self.model is None:
            return 0.0
        device = next(self.model.parameters()).device
        n_ent = len(self.ent2idx)
        pos_scores, neg_scores = [], []
        for t in eval_triples:
            s = self.ent2idx.get(t.source_id)
            r = self.rel2idx.get(t.relation_type)
            o = self.ent2idx.get(t.target_id)
            if s is None or r is None or o is None:
                continue
            s_t = torch.tensor([s], device=device)
            r_t = torch.tensor([r], device=device)
            o_t = torch.tensor([o], device=device)
            pos_scores.append(self.model.score(s_t, r_t, o_t).item())
            for _ in range(n_negatives):
                # Corrupt tail
                o_neg = int(torch.randint(0, n_ent, (1,), device=device).item())
                if (s, r, o_neg) in self._train_set:
                    continue
                o_neg_t = torch.tensor([o_neg], device=device)
                neg_scores.append(self.model.score(s_t, r_t, o_neg_t).item())
        if not pos_scores or not neg_scores:
            return 0.0
        # Manual AUROC: P(pos > neg)
        wins, total = 0, 0
        # Sample pairs to keep this O(N) instead of O(N²)
        max_pairs = 50_000
        import random as _r
        rng = _r.Random(0)
        for _ in range(max_pairs):
            p = rng.choice(pos_scores)
            n = rng.choice(neg_scores)
            wins += (p > n) + 0.5 * (p == n)
            total += 1
        return wins / max(1, total)

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "ent2idx": self.ent2idx,
            "rel2idx": self.rel2idx,
            "dim": self.dim,
            "checkpoint_name": self.checkpoint_name,
        }, path)
        logger.info("saved ComplEx checkpoint → %s", path)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "ComplExScorer":
        path = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        scorer = cls(dim=ckpt["dim"], device=device,
                     checkpoint_name=ckpt.get("checkpoint_name", path.stem))
        scorer.ent2idx = ckpt["ent2idx"]
        scorer.rel2idx = ckpt["rel2idx"]
        scorer.model = _ComplEx(len(scorer.ent2idx), len(scorer.rel2idx), scorer.dim).to(device)
        scorer.model.load_state_dict(ckpt["state_dict"])
        scorer.model.eval()
        logger.info("loaded ComplEx checkpoint ← %s", path)
        return scorer
