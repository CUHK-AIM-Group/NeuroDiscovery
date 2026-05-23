"""Load (s, p, o) triples from a NeuroOracle KG and produce 80/10/10 splits.

Filtering rules (matching the spirit of EDGE_TIER in schema.py):
  - Drop infrastructure edges (atlas / modality / dataset / ml_model nodes)
  - Drop 'about' / 'supported_by' provenance edges
  - Drop self-loops
  - Drop edges whose confidence < min_confidence (default 0.2)

Stratified split by (source_domain, target_domain) pair so that the test set
isn't dominated by a single domain pair (e.g. disease-disease) — see
plans/plausibility-scorer-c.md risks section.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Edges we never train on (provenance / infrastructure scaffolding).
# These don't carry biological signal and would dominate the train set.
_DROP_RELATIONS = frozenset({
    "about",
    "supported_by",
    "supports_modality",
    "provides_modality",
    "modality_provides",
    "measured_in_modality",
    "provides_signal_for",
    "assessed_in",
    "affects_system",
})

# Domain tags that mark "infrastructure" nodes. Edges incident to these are
# skipped because they encode experimental setup rather than scientific claims.
_INFRA_DOMAINS = frozenset({"atlas", "modality", "dataset", "ml_model", "recipe"})


@dataclass(frozen=True)
class Triple:
    source_id: str
    relation_type: str
    target_id: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.source_id, self.relation_type, self.target_id)


def _node_is_infra(node: dict) -> bool:
    tags = node.get("domain_tags") or []
    return any(t in _INFRA_DOMAINS for t in tags)


def _primary_domain(node: dict) -> str:
    tags = node.get("domain_tags") or []
    return tags[0] if tags else "unknown"


def load_triples_from_kg(
    kg_path: str | Path,
    min_confidence: float = 0.2,
) -> tuple[list[Triple], dict[str, str]]:
    """Read ``knowledge_graph.json`` and return (triples, node_id → domain).

    The second return value is used downstream for stratified splitting and
    for negative sampling restricted to plausible domains.
    """
    kg_path = Path(kg_path)
    with kg_path.open(encoding="utf-8") as f:
        kg = json.load(f)

    concepts = kg.get("concepts") or {}
    edges = kg.get("edges") or []

    node_domain = {cid: _primary_domain(node) for cid, node in concepts.items()}
    infra_nodes = {cid for cid, node in concepts.items() if _node_is_infra(node)}

    triples: list[Triple] = []
    drop_reasons: dict[str, int] = defaultdict(int)

    for e in edges:
        s = e.get("source_id")
        t = e.get("target_id")
        r = e.get("relation_type")
        if not (s and t and r):
            drop_reasons["incomplete"] += 1
            continue
        if s == t:
            drop_reasons["self_loop"] += 1
            continue
        if r in _DROP_RELATIONS:
            drop_reasons["dropped_relation"] += 1
            continue
        if s in infra_nodes or t in infra_nodes:
            drop_reasons["infra"] += 1
            continue
        if (e.get("confidence") or 0.0) < min_confidence:
            drop_reasons["low_confidence"] += 1
            continue
        if s not in concepts or t not in concepts:
            drop_reasons["dangling"] += 1
            continue
        triples.append(Triple(s, r, t))

    logger.info(
        "loaded %d triples from %s (dropped: %s)",
        len(triples), kg_path.name, dict(drop_reasons),
    )
    return triples, node_domain


def split_triples(
    triples: list[Triple],
    node_domain: dict[str, str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Triple], list[Triple], list[Triple]]:
    """Stratified 80/10/10 by (source_domain, target_domain, relation_type).

    Pairs with <3 triples go entirely into train (no val/test slot).
    """
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[Triple]] = defaultdict(list)
    for t in triples:
        key = (
            node_domain.get(t.source_id, "unknown"),
            node_domain.get(t.target_id, "unknown"),
            t.relation_type,
        )
        buckets[key].append(t)

    train, val, test = [], [], []
    for key, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            train.extend(group)
            continue
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        # Ensure at least 1 test triple
        n_test = max(1, n - n_train - n_val)
        if n_train + n_val + n_test > n:
            # Trim train if rounding overshot
            n_train = n - n_val - n_test
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val : n_train + n_val + n_test])

    logger.info(
        "split %d triples → train=%d val=%d test=%d (%d strata)",
        len(triples), len(train), len(val), len(test), len(buckets),
    )
    return train, val, test
