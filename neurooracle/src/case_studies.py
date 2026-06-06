"""Case-study registry for the NeuroClaw Nature paper.

The autoresearch CLI runs a four-stage cycle (batch -> novelty -> critic ->
plausibility) over the canonical task / chain registry in :mod:`atoms`. The
three case studies that anchor the Nature paper do not all map cleanly onto a
single canonical task, so this module records the extra routing metadata the
generic engine needs:

- CS2 uses a cluster-mining generator over ROI x modality x sign buckets.
- CS3 uses a canonical chain plus case-specific atom-pool restrictions.
- CS-gamma is reserved for hindcasting over a frozen historical KG snapshot.

This module declares each case study as a :class:`CaseStudy` config: the
generator family it dispatches to, the underlying canonical task / chain
(when any), per-stage parameter overrides, and pre/post hooks that adapt
the generic engine to case-specific constraints. The CLI's ``case-study``
subcommand reads this registry and routes execution accordingly.

At the current rollout state, CS2 and CS3 are wired into the case-study
orchestrator. CS-gamma remains a reserved generator family until the
historical claim / snapshot pipeline is finalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .atoms import (
    Atom,
    Task,
    TaskChain,
    chain_by_name,
    task_by_name,
)


# ── Generator families ────────────────────────────────────────────────────────
# Identifies which code path inside the case-study orchestrator handles
# stage [1/4] (raw hypothesis generation) for a given case study.
GENERATOR_TASK              = "task"               # CANONICAL_TASKS via batch_generate_for_task
GENERATOR_CHAIN             = "chain"              # CANONICAL_CHAINS via batch_generate_for_chain
GENERATOR_CLUSTER_MINING    = "cluster_mining"    # CS2 via find_transdiagnostic_clusters
GENERATOR_ATOM_SUBSTITUTION = "atom_substitution"  # CS-gamma hindcasting (reserved)

_KNOWN_GENERATORS = frozenset({
    GENERATOR_TASK,
    GENERATOR_CHAIN,
    GENERATOR_CLUSTER_MINING,
    GENERATOR_ATOM_SUBSTITUTION,
})


# ── Per-stage parameter blocks ───────────────────────────────────────────────
# Each stage's params mirror the kwargs of the corresponding cmd_* in
# hypothesis_cli.py / kge.cli — the orchestrator forwards them verbatim.

@dataclass(frozen=True)
class BatchParams:
    """Parameters for stage [1/4] — raw hypothesis generation."""
    max_hops: int = 3
    min_hops: int = 2
    metapath_min_domains: int = 2
    max_paths: int = 4
    max_seeds: int = 30
    target_per_task: int = 100
    max_retries: int = 4
    retry_scale: float = 2.0
    prefer_longer_paths: bool = True


@dataclass(frozen=True)
class NoveltyParams:
    """Parameters for stage [2/4] — PubMed + Semantic Scholar novelty check."""
    top: int = 200
    alpha: float = 0.5
    skip_pubmed: bool = False
    skip_semantic: bool = False


@dataclass(frozen=True)
class CriticParams:
    """Parameters for stage [3/4] — three-perspective Critic Agent."""
    top: int = 100
    max_rounds: int = 2
    threshold: float = 0.55
    max_workers: int = 12


@dataclass(frozen=True)
class PlausibilityParams:
    """Parameters for stage [4/4] — KGE ComplEx + PubMed attestation."""
    top: int = 100
    no_pubmed: bool = False
    skip_existing: bool = True
    enable_surprise: bool = False
    surprise_alpha: float = 0.1
    evo_surprise_min: Optional[float] = None
    device: Optional[str] = None


@dataclass(frozen=True)
class StageParams:
    """Bundle of stage configs — one block per pipeline stage."""
    batch: BatchParams = field(default_factory=BatchParams)
    novelty: NoveltyParams = field(default_factory=NoveltyParams)
    critic: CriticParams = field(default_factory=CriticParams)
    plausibility: PlausibilityParams = field(default_factory=PlausibilityParams)


# ── Case-study record ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CaseStudy:
    """A single Nature-paper case study and its autoresearch config.

    Fields
    ------
    name:
        Stable slug used as CLI argument and output-directory key.
    chinese_name / english_name:
        Frozen copy of the names finalized in the project memory; the CLI
        echoes both in the run log so figure / paper text stays in sync
        with whatever the orchestrator actually generated.
    generator:
        One of GENERATOR_TASK / GENERATOR_CHAIN / GENERATOR_CLUSTER_MINING
        / GENERATOR_ATOM_SUBSTITUTION.
    task / chain:
        The canonical Task or TaskChain backing the generator. Exactly one
        is set for GENERATOR_TASK / GENERATOR_CHAIN. CS2 still pins a Task
        for downstream tagging; CS-γ pins a chain as its schema anchor.
    stage_params:
        Per-stage parameter overrides. Defaults match run_cycle.sh.
    pre_hooks / post_hooks:
        Optional callables invoked around stage [1/4]. Each receives the
        active :class:`HypothesisEngine` and the case study itself; they
        may mutate engine state (e.g. _path_ignore_ids for CS3's
        pathway-only GENE pool) or rewrap generated hypotheses.
    extras:
        Generator-specific config dict that doesn't fit anywhere else
        (snapshot paths for CS-γ, cluster-mining knobs for CS2, ...).
    """
    name: str
    chinese_name: str
    english_name: str
    generator: str
    task: Optional[Task] = None
    chain: Optional[TaskChain] = None
    stage_params: StageParams = field(default_factory=StageParams)
    pre_hooks: tuple[Callable[..., None], ...] = ()
    post_hooks: tuple[Callable[..., None], ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CaseStudy.name must be non-empty")
        if self.generator not in _KNOWN_GENERATORS:
            raise ValueError(
                f"CaseStudy '{self.name}': unknown generator {self.generator!r} "
                f"(valid: {sorted(_KNOWN_GENERATORS)})"
            )
        if self.generator == GENERATOR_TASK and self.task is None:
            raise ValueError(f"CaseStudy '{self.name}': generator='task' requires .task")
        if self.generator == GENERATOR_CHAIN and self.chain is None:
            raise ValueError(f"CaseStudy '{self.name}': generator='chain' requires .chain")


# ── Pre_hooks ────────────────────────────────────────────────────────────────

def _cs3_pin_atom_pools(engine, case) -> None:
    """CS3: force the chain to actually traverse IM:*/OUTCOME:* anchors.

    Without this, ``batch_generate_for_chain`` accepts any node whose
    ``domain_tags`` overlap the atom's domain set — for IMAGING_MARKER that
    includes ``neuroanatomy``, so raw region CUIs (Hippocampus, etc.) end up
    serving as the marker anchor and the new IM:* atom layer is never
    reached. CS3 also wants gene anchors restricted to GENESET-aware genes
    (the ``gene_pool_filter='pathway_aggregated'`` flag) and outcome anchors
    pinned to the OUTCOME:* rating-scale atoms (not the broader
    dataset_variable hubs).
    """
    # Genes that participate in any GENESET (the "pathway_aggregated" pool).
    pathway_genes: set[str] = set()
    for u, v, d in engine.G.edges(data=True):
        if d.get("relation_type") == "part_of" and v.startswith("GENESET:"):
            pathway_genes.add(u)

    def _im_filter(nid, node):
        return nid.startswith("IM:")

    def _gene_filter(nid, node):
        if node is None:
            return False
        if "gene" not in (node.domain_tags or []):
            return False
        # only keep genes that show up in a pathway / GENESET; falls back to
        # all gene CUIs if the GENESET layer somehow didn't load.
        return (nid in pathway_genes) if pathway_genes else True

    def _outcome_filter(nid, node):
        return nid.startswith("OUTCOME:")

    engine._chain_atom_filters = {
        Atom.IMAGING_MARKER: _im_filter,
        Atom.GENE_TARGET:    _gene_filter,
        Atom.OUTCOME:        _outcome_filter,
    }


# ── Concrete case studies (frozen for the Nature paper) ──────────────────────

CS2 = CaseStudy(
    name="cs2_transdiagnostic",
    chinese_name="跨诊断精神疾病脑影像图谱",
    english_name="Transdiagnostic Brain Atlas of Psychiatric Disorders",
    generator=GENERATOR_CLUSTER_MINING,
    task=task_by_name("transdiagnostic_clustering"),
    stage_params=StageParams(
        batch=BatchParams(max_paths=6, max_seeds=60, target_per_task=80),
        novelty=NoveltyParams(top=120, alpha=0.5),
        critic=CriticParams(top=60, max_rounds=2, threshold=0.55),
        plausibility=PlausibilityParams(top=60),
    ),
    extras={
        "min_diseases_per_cluster": 3,
        "max_clusters": 20,
        "modality_partitioned": True,
    },
)

CS3 = CaseStudy(
    name="cs3_pathway_mediation",
    chinese_name="多基因通路影像中介",
    english_name="Pathway-Level Polygenic Mediation through Brain Imaging",
    generator=GENERATOR_CHAIN,
    chain=chain_by_name("pathway_polygenic_mediation"),
    stage_params=StageParams(
        batch=BatchParams(
            max_hops=3, min_hops=2, max_paths=4, max_seeds=30,
            target_per_task=100, max_retries=4, retry_scale=2.0,
        ),
        novelty=NoveltyParams(top=200, alpha=0.5),
        critic=CriticParams(top=100, max_rounds=2, threshold=0.55),
        plausibility=PlausibilityParams(top=100),
    ),
    pre_hooks=(_cs3_pin_atom_pools,),
    extras={
        "gene_pool_filter": "pathway_aggregated",
    },
)

CS_GAMMA = CaseStudy(
    name="cs_gamma_hindcasting",
    chinese_name="假设回溯预测",
    english_name="Hypothesis Hindcasting",
    generator=GENERATOR_ATOM_SUBSTITUTION,
    chain=chain_by_name("genetic_imaging_disease"),
    stage_params=StageParams(
        batch=BatchParams(max_paths=4, max_seeds=30, target_per_task=100),
        novelty=NoveltyParams(top=200, alpha=0.5),
        critic=CriticParams(top=100, max_rounds=2, threshold=0.55),
        plausibility=PlausibilityParams(top=100),
    ),
    extras={
        "snapshot_2022_kg":         "neurooracle/data/snapshots/kg_2022.json",
        "snapshot_2022_kge":        "neurooracle/data/snapshots/kge_2022.pt",
        "substitution_axes":        ("GENE_TARGET", "IMAGING_MARKER", "DISEASE"),
        "max_substitutions_per_seed": 5,
    },
)


CASE_STUDIES: tuple[CaseStudy, ...] = (CS2, CS3, CS_GAMMA)


def case_study_by_name(name: str) -> CaseStudy:
    """Look up a case study by its registry slug. Raises KeyError."""
    for cs in CASE_STUDIES:
        if cs.name == name:
            return cs
    valid = ", ".join(c.name for c in CASE_STUDIES)
    raise KeyError(f"unknown case study: {name!r} (valid: {valid})")


def list_case_study_names() -> tuple[str, ...]:
    return tuple(cs.name for cs in CASE_STUDIES)


__all__ = [
    "BatchParams",
    "NoveltyParams",
    "CriticParams",
    "PlausibilityParams",
    "StageParams",
    "CaseStudy",
    "CS2",
    "CS3",
    "CS_GAMMA",
    "CASE_STUDIES",
    "case_study_by_name",
    "list_case_study_names",
    "GENERATOR_TASK",
    "GENERATOR_CHAIN",
    "GENERATOR_CLUSTER_MINING",
    "GENERATOR_ATOM_SUBSTITUTION",
]
