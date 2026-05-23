"""Chain coverage analysis: how much of each task / chain is supported by claims.

Phase 2 currently fetches papers under a disease+imaging query that biases
the KG toward chains anchored on (DISEASE × IMAGING_MARKER). Other chains
(genetic_imaging_disease, drug_imaging_outcome, task_brain_behavior) are
incidentally populated and often appear as "structural skeletons" without
claim or hypothesis support.

This module reads a KG and counts, per ``Task`` / ``TaskChain``::

    n_concepts_by_atom    nodes that can play each atom
    n_directed_edges      curated edges aligning with consecutive atom pairs
    n_paths               simple paths visiting atoms in chain order
                          (capped to avoid blow-up; just a coverage proxy)
    n_claims_supporting   claim records whose subject/object resolve onto
                          adjacent atoms in the chain (under-counts: a claim
                          with same atom on both sides only matches one hop)

Surfaces "sparse" chains/tasks under user-defined thresholds so a backfill
run can target them with chain-aware PubMed queries.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional

from .atoms import (
    Atom, Task, TaskChain,
    CANONICAL_TASKS, CANONICAL_CHAINS,
    ATOM_TO_DOMAINS, atoms_for_domain,
)
from .graph_manager import KnowledgeGraph
from .schema import ConceptNode
from .storage import load_graph

logger = logging.getLogger(__name__)


@dataclass
class ChainCoverage:
    name: str
    kind: str                            # "chain" | "task"
    signature: str
    atoms: list[str]
    n_concepts_by_atom: dict[str, int] = field(default_factory=dict)
    n_directed_edges: int = 0
    n_claims_supporting: int = 0
    n_paths_sample: int = 0              # simple paths sampled, capped

    def to_dict(self) -> dict:
        return asdict(self)


def _atom_to_concept_ids(kg: KnowledgeGraph) -> dict[Atom, set[str]]:
    """For each atom, the set of node ids in the KG that can play it."""
    bucket: dict[Atom, set[str]] = defaultdict(set)
    for cid, node in kg._index.items():
        if not isinstance(node, ConceptNode):
            continue
        for d in node.domain_tags:
            for a in atoms_for_domain(d):
                bucket[a].add(cid)
    return bucket


def _count_directed_edges(
    kg: KnowledgeGraph,
    src_ids: set[str],
    dst_ids: set[str],
) -> int:
    """Edges from src bucket → dst bucket (excludes 'about' / claim-attach edges)."""
    n = 0
    for u, v, data in kg.G.edges(data=True):
        if u in src_ids and v in dst_ids:
            rt = data.get("relation_type", "")
            if rt in ("about", "supported_by"):
                continue
            n += 1
    return n


def _count_claims_supporting(
    kg: KnowledgeGraph,
    chain_atoms: tuple[Atom, ...],
    atom_ids: dict[Atom, set[str]],
) -> int:
    """Claims whose subject/object hits *any* adjacent atom pair in the chain.

    Implementation: a CLM:* node has an 'about' edge to its subject and
    object. We mark a claim as "supporting" the chain if (subject_atom,
    object_atom) is one of the consecutive pairs.
    """
    pairs: set[tuple[Atom, Atom]] = set()
    for i in range(len(chain_atoms) - 1):
        pairs.add((chain_atoms[i], chain_atoms[i + 1]))
        # claims are unordered — accept reverse too (over-counts slightly,
        # but in practice the ingestion canonicalises direction so most
        # match only one way).
        pairs.add((chain_atoms[i + 1], chain_atoms[i]))

    # Build a quick reverse lookup: cid → set of atoms it can play.
    cid_to_atoms: dict[str, set[Atom]] = defaultdict(set)
    for atom, ids in atom_ids.items():
        for cid in ids:
            cid_to_atoms[cid].add(atom)

    n = 0
    for cid in kg._index:
        if not cid.startswith("CLM:"):
            continue
        # CLM nodes have 'about' edges to their subject + object
        targets = [
            v for _, v, data in kg.G.out_edges(cid, data=True)
            if data.get("relation_type") == "about"
        ]
        if len(targets) < 2:
            continue
        # Try pairs of (subject, object) anchors
        hit = False
        for s in targets:
            for o in targets:
                if s == o:
                    continue
                s_atoms = cid_to_atoms.get(s, set())
                o_atoms = cid_to_atoms.get(o, set())
                for sa in s_atoms:
                    for oa in o_atoms:
                        if (sa, oa) in pairs:
                            hit = True
                            break
                    if hit: break
                if hit: break
            if hit: break
        if hit:
            n += 1
    return n


def _sample_simple_paths(
    kg: KnowledgeGraph,
    chain_atoms: tuple[Atom, ...],
    atom_ids: dict[Atom, set[str]],
    cap: int = 200,
) -> int:
    """Count simple paths visiting concepts in chain-atom order, capped."""
    if len(chain_atoms) < 2:
        return 0
    # Precompute atom membership for fast lookup.
    cid_atoms: dict[str, set[Atom]] = defaultdict(set)
    for a, ids in atom_ids.items():
        for cid in ids:
            cid_atoms[cid].add(a)

    # Seed nodes for each position.
    pos_nodes: list[set[str]] = [atom_ids[a] for a in chain_atoms]

    # BFS from each seed in pos_nodes[0], following structural edges only.
    found = 0
    for seed in islice_set(pos_nodes[0], 200):  # cap fan-out per chain
        if found >= cap:
            break
        # current frontier: list of paths (each path is a tuple of cids).
        frontier = [(seed,)]
        for pos_idx in range(1, len(chain_atoms)):
            need_atom = chain_atoms[pos_idx]
            next_frontier: list[tuple[str, ...]] = []
            for path in frontier:
                tail = path[-1]
                for _, v, data in kg.G.out_edges(tail, data=True):
                    if data.get("relation_type") in ("about", "supported_by"):
                        continue
                    if v in path:
                        continue  # simple path: no revisits
                    if need_atom not in cid_atoms.get(v, ()):
                        continue
                    next_frontier.append(path + (v,))
                    if len(next_frontier) >= cap * 4:
                        break
                if len(next_frontier) >= cap * 4:
                    break
            frontier = next_frontier
            if not frontier:
                break
        found += len(frontier)
        if found >= cap:
            return cap
    return found


def islice_set(s: set, n: int):
    out = []
    for x in s:
        if len(out) >= n:
            break
        out.append(x)
    return out


# ── Top-level analysis ─────────────────────────────────────────────────────────


def analyse(kg: KnowledgeGraph, path_cap: int = 200) -> list[ChainCoverage]:
    """Run coverage analysis for every CANONICAL_CHAIN and CANONICAL_TASK."""
    atom_ids = _atom_to_concept_ids(kg)
    out: list[ChainCoverage] = []

    for chain in CANONICAL_CHAINS:
        cov = ChainCoverage(
            name=chain.name,
            kind="chain",
            signature=chain.signature,
            atoms=[a.value for a in chain.chain],
        )
        cov.n_concepts_by_atom = {
            a.value: len(atom_ids.get(a, set())) for a in set(chain.chain)
        }
        # Sum directed edges across consecutive atom buckets
        e = 0
        for i in range(len(chain.chain) - 1):
            e += _count_directed_edges(
                kg, atom_ids[chain.chain[i]], atom_ids[chain.chain[i + 1]],
            )
        cov.n_directed_edges = e
        cov.n_claims_supporting = _count_claims_supporting(kg, chain.chain, atom_ids)
        cov.n_paths_sample = _sample_simple_paths(kg, chain.chain, atom_ids, cap=path_cap)
        out.append(cov)

    for task in CANONICAL_TASKS:
        chain_atoms = tuple(task.inputs) + (task.output,)
        cov = ChainCoverage(
            name=task.name,
            kind="task",
            signature=task.signature,
            atoms=[a.value for a in chain_atoms],
        )
        cov.n_concepts_by_atom = {
            a.value: len(atom_ids.get(a, set())) for a in set(chain_atoms)
        }
        # For tasks the inputs are unordered; count any directed edge
        # within the input atom set ∪ to-output as "supporting". This
        # over-counts slightly compared to chains but is fine as a
        # density proxy.
        e = 0
        for a_in in task.inputs:
            e += _count_directed_edges(kg, atom_ids[a_in], atom_ids[task.output])
        cov.n_directed_edges = e
        cov.n_claims_supporting = _count_claims_supporting(kg, chain_atoms, atom_ids)
        cov.n_paths_sample = _sample_simple_paths(kg, chain_atoms, atom_ids, cap=path_cap)
        out.append(cov)

    return out


def find_sparse(
    coverage: list[ChainCoverage],
    min_claims: int = 50,
    min_edges: int = 100,
) -> list[ChainCoverage]:
    """Filter to the chains/tasks below either threshold."""
    return [
        c for c in coverage
        if c.n_claims_supporting < min_claims or c.n_directed_edges < min_edges
    ]


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analyse atom-chain / task coverage in the KG."
    )
    parser.add_argument("--graph", type=str, required=True,
                        help="Path to knowledge_graph.json")
    parser.add_argument("--out", type=str, default=None,
                        help="Write coverage report as JSON here")
    parser.add_argument("--path-cap", type=int, default=200,
                        help="Max sampled simple paths per chain (default 200)")
    parser.add_argument("--min-claims", type=int, default=50)
    parser.add_argument("--min-edges", type=int, default=100)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    kg = load_graph(Path(args.graph))
    logger.info(f"loaded graph: {kg.stats()['n_concepts']} concepts, "
                f"{kg.stats()['n_edges']} edges")

    coverage = analyse(kg, path_cap=args.path_cap)
    sparse = find_sparse(coverage, min_claims=args.min_claims, min_edges=args.min_edges)

    print(f"\n{'='*70}")
    print(f"CHAIN / TASK COVERAGE")
    print(f"{'='*70}")
    print(f"{'name':<32} {'kind':<6} {'signature':<26} {'edges':>7} {'claims':>7} {'paths':>7}")
    for c in coverage:
        print(f"{c.name[:32]:<32} {c.kind:<6} {c.signature[:26]:<26} "
              f"{c.n_directed_edges:>7} {c.n_claims_supporting:>7} {c.n_paths_sample:>7}")

    print(f"\n{'='*70}")
    print(f"SPARSE (< {args.min_claims} claims OR < {args.min_edges} edges):")
    print(f"{'='*70}")
    for c in sparse:
        print(f"  {c.kind:<6} {c.name:<32} signature={c.signature}  "
              f"edges={c.n_directed_edges} claims={c.n_claims_supporting}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([c.to_dict() for c in coverage], indent=2),
            encoding="utf-8",
        )
        logger.info(f"wrote coverage to {out_path}")


if __name__ == "__main__":
    main()
