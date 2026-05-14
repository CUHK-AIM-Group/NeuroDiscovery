"""Phase 4.2: Evolutionary Hypothesis Mutation.

Uses genetic operators (mutation, crossover, selection) to expand the
hypothesis space beyond single-path enumeration. Discovers multi-biomarker
joint predictions and cross-disease mechanism transfers.

Usage:
    python -m core.knowledge_graph.phase4 evolve \
        --input data/hypotheses_final.json \
        --output data/hypotheses_evolved.json \
        --population 50 --generations 10
"""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Optional

import networkx as nx

from .claim_ingestion import _PREDICATE_KEYWORDS
from .hypothesis_engine import (
    Hypothesis, HypothesisEngine, HypothesisLink,
    _AAL_REGION_KEYWORDS, NON_MEASURABLE_BIOMARKER_TYPES,
    _NON_MEASURABLE_PATTERNS,
)
from .schema import ConceptNode

logger = logging.getLogger(__name__)

# ── biological plausibility constraints ────────────────────────────────

DIRECTIONALITY_RULES: dict[tuple[str, str], bool] = {
    ("gene", "protein"): True,           # gene -> protein (transcription)
    ("protein", "gene"): False,          # protein -> gene (forbidden)
    ("brain_region", "disease"): True,   # brain region -> disease (valid)
    ("disease", "brain_region"): False,  # disease -> brain region (forbidden)
}

PREDICATE_TYPE_COMPAT: dict[str, dict] = {
    "is_biomarker_of": {
        "source_types": {"biomarker", "gene", "protein"},
        "target_types": {"disease", "disorder"},
        "forbidden": {"neuroanatomy"},   # brain regions are not biomarkers
    },
    "causes": {
        "source_types": {"gene", "drug", "pathogen", "environmental_factor"},
        "target_types": {"disease", "symptom", "disorder"},
    },
    "treats": {
        "source_types": {"drug", "intervention"},
        "target_types": {"disease", "symptom"},
    },
}

MODALITY_COMPAT: dict[str, set[str]] = {
    "sMRI": {"cortical_thickness", "gray_matter_volume", "subcortical_volume"},
    "fMRI": {"functional_connectivity", "activation"},
    "PET":  {"amyloid_SUVR", "tau_SUVR", "FDG_uptake"},
    "dMRI": {"structural_connectivity", "FA", "MD"},
}

# Vague predicates that should be refined
_VAGUE_PREDICATES = {"is_associated_with", "associated_with", "correlates_with", "about"}


class EvolutionMemory:
    """Tracks operator performance across generations for self-evolutionary learning.

    Records trials/successes/fitness per operator, and per-generation stats.
    Used by PES directed mutation to weight operator selection.
    """

    def __init__(self):
        self.operator_stats: dict[str, dict] = {}  # op_name -> {trials, successes, total_fitness}
        self.generation_log: list[dict] = []
        self.failed_mutations: list[dict] = []

    def record_trial(self, op_name: str):
        if op_name not in self.operator_stats:
            self.operator_stats[op_name] = {"trials": 0, "successes": 0, "total_fitness": 0.0}
        self.operator_stats[op_name]["trials"] += 1

    def record_success(self, op_name: str, fitness: float):
        if op_name not in self.operator_stats:
            self.operator_stats[op_name] = {"trials": 0, "successes": 0, "total_fitness": 0.0}
        self.operator_stats[op_name]["successes"] += 1
        self.operator_stats[op_name]["total_fitness"] += fitness

    def record_failure(self, op_name: str, reason: str = ""):
        self.failed_mutations.append({"operator": op_name, "reason": reason})
        if len(self.failed_mutations) > 500:
            self.failed_mutations = self.failed_mutations[-300:]

    def record_generation(self, gen: int, population: list):
        fitnesses = [h.metadata.get("fitness", 0) for h in population]
        self.generation_log.append({
            "generation": gen,
            "best": max(fitnesses) if fitnesses else 0,
            "avg": sum(fitnesses) / max(len(fitnesses), 1),
            "pop_size": len(population),
        })

    def get_operator_weights(self) -> dict[str, float]:
        """Compute operator weights based on historical success rate × avg fitness.

        Returns dict of op_name -> weight (for weighted random selection).
        Untried operators get a default weight of 1.0.
        """
        weights = {}
        for op_name, stats in self.operator_stats.items():
            trials = stats["trials"]
            if trials == 0:
                weights[op_name] = 1.0
                continue
            success_rate = stats["successes"] / trials
            avg_fitness = stats["total_fitness"] / max(stats["successes"], 1)
            # weight = success_rate * avg_fitness, with floor of 0.1
            weights[op_name] = max(0.1, success_rate * avg_fitness)
        return weights

    def summary(self) -> dict:
        return {
            "operators": dict(self.operator_stats),
            "generations": len(self.generation_log),
            "total_failures": len(self.failed_mutations),
        }


class EvolutionEngine:
    """Evolve hypotheses via mutation, crossover, and selection."""

    def __init__(
        self,
        engine: HypothesisEngine,
        population_size: int = 50,
        n_generations: int = 10,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.3,
        tournament_size: int = 3,
        elitism_n: int = 5,
    ):
        self.engine = engine
        self.G = engine.G
        self._index = engine._index
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_n = elitism_n
        self._current_population: list[Hypothesis] = []
        self.memory = EvolutionMemory()

        self.mutators = [
            self._hop_extension,
            self._hop_contraction,
            self._biomarker_swap,
            self._outcome_pivot,
            self._mediator_injection,
            self._convergence_fusion,
        ]
        self.crossovers = [
            self._path_crossover,
            self._mechanism_transfer,
        ]

    # ── main loop ──────────────────────────────────────────────────────

    def evolve(self, seed_hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Run the full evolutionary loop."""
        population = self._initialize_population(seed_hypotheses)
        logger.info(f"initial population: {len(population)} individuals")

        for gen in range(1, self.n_generations + 1):
            self._current_population = population
            # update target/source counts for diversity penalty
            from collections import Counter
            self._target_counts = Counter(h.target_name for h in population)
            self._source_counts = Counter(h.source_name for h in population)
            # score fitness
            for h in population:
                h.metadata["fitness"] = self._score_fitness(h)

            # sort by fitness
            population.sort(key=lambda h: h.metadata.get("fitness", 0), reverse=True)
            best = population[0].metadata.get("fitness", 0)
            avg = sum(h.metadata.get("fitness", 0) for h in population) / max(len(population), 1)
            logger.info(f"gen {gen}/{self.n_generations}: best={best:.4f} avg={avg:.4f} pop={len(population)}")
            self.memory.record_generation(gen, population)

            # elitism: keep top N (deduplicate by id)
            seen_ids = set()
            survivors = []
            for h in population:
                if h.id not in seen_ids:
                    survivors.append(h)
                    seen_ids.add(h.id)
                if len(survivors) >= self.elitism_n:
                    break

            # fill rest via tournament → crossover → mutation
            max_fill_attempts = self.population_size * 10
            fill_attempts = 0
            while len(survivors) < self.population_size and fill_attempts < max_fill_attempts:
                fill_attempts += 1
                # tournament select parent(s)
                if random.random() < self.crossover_rate and len(population) >= 2:
                    p1 = self._tournament_select_one(population)
                    p2 = self._tournament_select_one(population)
                    if p1.id != p2.id:
                        child = self._apply_crossover(p1, p2)
                        if child and self._validate(child) and child.id not in seen_ids:
                            survivors.append(child)
                            seen_ids.add(child.id)
                            continue
                parent = self._tournament_select_one(population)
                child = self._apply_mutation(parent)
                if child and self._validate(child) and child.id not in seen_ids:
                    survivors.append(child)
                    seen_ids.add(child.id)

            # if still short, fill with tournament-selected originals
            while len(survivors) < self.population_size:
                parent = self._tournament_select_one(population)
                survivors.append(parent)

            population = survivors[: self.population_size]

        # final scoring
        for h in population:
            h.metadata["fitness"] = self._score_fitness(h)
            h.evolve_score = h.metadata["fitness"]
            # recompute composite using evolve_score as novelty boost
            h.composite_score = self.engine._composite_score(h)

        # Post-filter evolved variants: drop EVO:* that don't meet quality bar.
        # Originals (HYP:*) are always kept since they already passed Critic.
        # This addresses the "Evolution produces 0 winners in top-10" failure mode
        # by rejecting variants that mutation pushed below their parent's quality.
        before_filter = len(population)
        # Precompute hub IDs once (top-50 by non-'about' degree in the KG)
        from collections import Counter
        _deg = Counter()
        for u, v, data in self.G.edges(data=True):
            if data.get("relation_type") != "about":
                _deg[u] += 1
                _deg[v] += 1
        hub_ids = {cid for cid, _ in _deg.most_common(50)}

        filtered = []
        for h in population:
            if not h.id.startswith("EVO:"):
                filtered.append(h)
                continue
            # Quality gates for evolved variants:
            if h.composite_score < 0.40:
                continue
            if any(s.confidence < 0.05 for s in h.path):
                continue
            vague_count = sum(1 for s in h.path
                              if s.relation_type in {"is_associated_with", "correlates_with"})
            if vague_count >= 2:
                continue
            if len(h.path) > 7:
                continue
            # New: reject if any step is hub-to-hub (generic category→category edge)
            if any(s.from_id in hub_ids and s.to_id in hub_ids for s in h.path):
                continue
            # Also reject if any node in the path is in the explicit blacklist
            # of vague COGAT/MeSH umbrella hubs (memory/logic/loss/Brain/etc.).
            # Mirrors hypothesis_engine.post_process so evolved variants
            # cannot smuggle these in by direct graph sampling.
            from .hypothesis_engine import (
                PATH_IGNORE_NODE_IDS, INTERMEDIATE_ONLY_IGNORE_IDS,
                HypothesisEngine, DIRECTIONAL_RELATIONS,
            )
            if h.source_id in PATH_IGNORE_NODE_IDS or h.target_id in PATH_IGNORE_NODE_IDS:
                continue
            if any(s.from_id in PATH_IGNORE_NODE_IDS or s.to_id in PATH_IGNORE_NODE_IDS
                   for s in h.path):
                continue
            # Reject paths transiting through disease mega-hubs
            if HypothesisEngine._transits_intermediate_only_hub(h):
                continue
            # (C-1/2/3) mirror the phrase-level, directional-density, and
            # too-broad-target gates from hypothesis_engine.post_process so
            # evolved variants cannot slip generic phrases past us by
            # construction (e.g. mutating into "neural activity" as hop).
            intermediate_names: list[str] = []
            for i, link in enumerate(h.path):
                if i >= 1:
                    intermediate_names.append(link.from_name or "")
                if i < len(h.path) - 1:
                    intermediate_names.append(link.to_name or "")
            if any(HypothesisEngine._is_generic_intermediate(n) for n in intermediate_names):
                continue
            if len(h.path) >= 3:
                directional = sum(1 for s in h.path if s.relation_type in DIRECTIONAL_RELATIONS)
                if directional * 2 < len(h.path):
                    continue
            if HypothesisEngine._is_too_broad_target(h.target_name):
                continue
            filtered.append(h)
        n_dropped = before_filter - len(filtered)
        if n_dropped > 0:
            logger.info(
                f"post-filter dropped {n_dropped} evolved variants "
                f"(quality gates: composite<0.4, weak edge, dual-vague, len>7, "
                f"hub-to-hub, vague-umbrella-hub)"
            )
        population = filtered

        population.sort(key=lambda h: h.metadata.get("fitness", 0), reverse=True)

        # log evolution memory summary
        mem_summary = self.memory.summary()
        logger.info(
            f"evolution complete: {mem_summary['generations']} generations, "
            f"{mem_summary['total_failures']} failures"
        )
        for op_name, stats in mem_summary.get("operators", {}).items():
            if stats["trials"] > 0:
                sr = stats["successes"] / stats["trials"]
                logger.info(f"  {op_name}: {stats['trials']} trials, {sr:.1%} success")

        return population

    # ── initialization ─────────────────────────────────────────────────

    def _initialize_population(self, seeds: list[Hypothesis]) -> list[Hypothesis]:
        """Seeds + random mutations to fill population."""
        seen_ids: set[str] = set()
        population: list[Hypothesis] = []
        self._current_population = seeds
        for h in seeds:
            if h.id not in seen_ids:
                population.append(h)
                seen_ids.add(h.id)

        self._current_population = population
        attempts = 0
        max_attempts = self.population_size * 5

        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            parent = random.choice(seeds)
            child = self._apply_mutation(parent)
            if child and self._validate(child) and child.id not in seen_ids:
                population.append(child)
                seen_ids.add(child.id)
            # also try crossover between seeds
            if len(seeds) >= 2 and random.random() < 0.3 and len(population) < self.population_size:
                p1, p2 = random.sample(seeds, 2)
                child = self._apply_crossover(p1, p2)
                if child and self._validate(child) and child.id not in seen_ids:
                    population.append(child)
                    seen_ids.add(child.id)

        if len(population) < self.population_size:
            logger.warning(f"could only fill {len(population)}/{self.population_size} after {max_attempts} attempts")

        return population

    # ── selection ──────────────────────────────────────────────────────

    def _tournament_select_one(self, population: list[Hypothesis]) -> Hypothesis:
        """Tournament selection: pick k random, return the fittest."""
        k = min(self.tournament_size, len(population))
        candidates = random.sample(population, k)
        return max(candidates, key=lambda h: h.metadata.get("fitness", 0))

    # ── fitness ────────────────────────────────────────────────────────

    def _score_fitness(self, h: Hypothesis) -> float:
        """fitness = α·confidence + β·evidence + γ·novelty + δ·testability + ε·convergence_bonus - diversity_penalty."""
        c = h.confidence_score
        e = h.evidence_score
        n = h.novelty_score
        t = h.testability_score

        fitness = 0.20 * c + 0.20 * e + 0.25 * n + 0.35 * t

        # convergence bonus: if hypothesis has multiple independent paths
        n_paths = h.metadata.get("n_independent_paths", 1)
        if n_paths > 1:
            max_paths = 10
            bonus = math.log(1 + n_paths) / math.log(1 + max_paths)
            fitness += 0.15 * bonus

        # diversity penalty: penalize over-represented targets AND sources
        if hasattr(self, '_target_counts') and self._target_counts:
            count = self._target_counts.get(h.target_name, 0)
            if count > 5:
                fitness *= max(0.7, 1.0 - 0.03 * (count - 5))
        if hasattr(self, '_source_counts') and self._source_counts:
            count = self._source_counts.get(h.source_name, 0)
            if count > 5:
                fitness *= max(0.7, 1.0 - 0.03 * (count - 5))

        # length penalty: exponential penalty for paths > 5 hops
        path_len = len(h.path)
        if path_len > 5:
            # 6 hops: 0.95x, 7 hops: 0.90x, 8 hops: 0.86x
            fitness *= 0.95 ** (path_len - 5)

        return fitness

    # ── mutation operators ─────────────────────────────────────────────

    def _hop_extension(self, h: Hypothesis) -> Optional[Hypothesis]:
        """A→B→C → A→B→X→C: insert intermediate node X between B and C."""
        if len(h.path) < 2:
            return None

        # pick a random hop to extend
        idx = random.randint(0, len(h.path) - 2)
        link_a = h.path[idx]  # A→B
        link_b = h.path[idx + 1]  # B→C

        mid_id = link_a.to_id  # B
        target_id = link_b.to_id  # C

        if mid_id not in self.G or target_id not in self.G:
            return None

        # find X: successor of B that also has edge to C
        successors = list(self.G.successors(mid_id))
        random.shuffle(successors)

        for x_id in successors[:20]:
            if x_id == target_id or x_id == link_a.from_id:
                continue
            if not self.G.has_edge(x_id, target_id):
                continue
            # check X is not a claim node
            x_node = self._index.get(x_id)
            if x_node and "claim" in x_node.domain_tags:
                continue
            # build new path
            new_path = list(h.path[:idx])
            new_link_a = HypothesisLink(
                from_id=link_a.from_id, from_name=link_a.from_name,
                to_id=mid_id, to_name=link_a.to_name,
                relation_type=link_a.relation_type, confidence=link_a.confidence,
                claim_id=link_a.claim_id, raw_text=link_a.raw_text,
                evidence=link_a.evidence, source_paper=link_a.source_paper,
            )
            new_link_mid = self._make_link(mid_id, x_id)
            new_link_c = self._make_link(x_id, target_id)
            if not new_link_mid or not new_link_c:
                continue
            new_path.extend([new_link_a, new_link_mid, new_link_c])
            new_path.extend(h.path[idx + 2:])
            return self._build_child(h, new_path, "hop_extension")

        return None

    def _hop_contraction(self, h: Hypothesis) -> Optional[Hypothesis]:
        """A→B→C→D → A→C→D: remove intermediate node B."""
        if len(h.path) < 3:
            return None

        # pick a random intermediate hop to remove
        idx = random.randint(0, len(h.path) - 3)
        link_a = h.path[idx]    # A→B
        link_b = h.path[idx + 1]  # B→C

        src_id = link_a.from_id  # A
        mid_id = link_a.to_id    # B (to remove)
        tgt_id = link_b.to_id    # C

        if not self.G.has_edge(src_id, tgt_id):
            return None

        new_link = self._make_link(src_id, tgt_id)
        if not new_link:
            return None

        new_path = list(h.path[:idx]) + [new_link] + list(h.path[idx + 2:])
        return self._build_child(h, new_path, "hop_contraction")

    def _biomarker_swap(self, h: Hypothesis) -> Optional[Hypothesis]:
        """A→...→outcome → A'→...→outcome: swap source with same-domain node."""
        if len(h.path) < 2:
            return None

        source_node = self._index.get(h.source_id)
        if not source_node:
            return None

        domains = set(source_node.domain_tags) - {"claim"}
        if not domains:
            return None

        # find same-domain node with edge to next hop
        next_id = h.path[0].to_id
        candidates = []
        for nid, data in self.G.nodes(data=True):
            if nid == h.source_id or nid == h.target_id:
                continue
            node_domains = set(data.get("domain_tags", [])) - {"claim"}
            if node_domains & domains and self.G.has_edge(nid, next_id):
                # For imaging hypotheses, prefer AAL regions
                if h.metadata.get("dataset") and "neuroanatomy" in node_domains:
                    name = data.get("preferred_name", "")
                    if any(kw.lower() in name.lower() for kw in _AAL_REGION_KEYWORDS):
                        candidates.append(nid)
                        continue
                candidates.append(nid)

        if not candidates:
            return None

        new_src_id = random.choice(candidates)
        new_src_node = self._index.get(new_src_id)
        if not new_src_node:
            return None

        new_first_link = self._make_link(new_src_id, next_id)
        if not new_first_link:
            return None

        new_path = [new_first_link] + list(h.path[1:])
        child = self._build_child(h, new_path, "biomarker_swap",
                                  source_id=new_src_id, source_name=new_src_node.preferred_name)
        # Update imaging feature name if this is an imaging hypothesis
        if child and h.metadata.get("dataset") and h.metadata.get("input_feature"):
            old_region = h.metadata.get("input_region", "")
            new_region = new_src_node.preferred_name
            feat = h.metadata.get("input_feature", "").replace(old_region, new_region)
            child.metadata["input_feature"] = feat
            child.metadata["input_region"] = new_region
            child.source_name = feat
        return child

    def _outcome_pivot(self, h: Hypothesis) -> Optional[Hypothesis]:
        """...→X→outcome → ...→X→outcome': swap outcome with same-domain node."""
        if len(h.path) < 2:
            return None

        target_node = self._index.get(h.target_id)
        if not target_node:
            return None

        domains = set(target_node.domain_tags) - {"claim"}
        if not domains:
            return None

        prev_id = h.path[-1].from_id  # X
        candidates = []
        for nid, data in self.G.nodes(data=True):
            if nid == h.source_id or nid == h.target_id:
                continue
            node_domains = set(data.get("domain_tags", [])) - {"claim"}
            if node_domains & domains and self.G.has_edge(prev_id, nid):
                candidates.append(nid)

        if not candidates:
            return None

        # For imaging hypotheses, prefer disease/cognitive_function outcomes
        is_imaging = bool(h.metadata.get("dataset"))
        if is_imaging:
            preferred = [nid for nid in candidates
                         if {"disease", "cognitive_function"} & set(
                             self._index.get(nid, ConceptNode(id="", preferred_name="")).domain_tags)]
            if preferred:
                candidates = preferred

        new_tgt_id = random.choice(candidates)
        new_tgt_node = self._index.get(new_tgt_id)
        if not new_tgt_node:
            return None

        new_last_link = self._make_link(prev_id, new_tgt_id)
        if not new_last_link:
            return None

        new_path = list(h.path[:-1]) + [new_last_link]
        child = self._build_child(h, new_path, "outcome_pivot",
                                  target_id=new_tgt_id, target_name=new_tgt_node.preferred_name)
        # Update outcome_type for imaging hypotheses
        if child and is_imaging:
            child.metadata["outcome_type"] = self.engine._classify_outcome(new_tgt_node)
        return child

    def _mediator_injection(self, h: Hypothesis) -> Optional[Hypothesis]:
        """A→C → A→M→C: find mediator M between A and C."""
        if len(h.path) < 1:
            return None

        # pick a random direct hop
        idx = random.randint(0, len(h.path) - 1)
        link = h.path[idx]
        src_id = link.from_id
        tgt_id = link.to_id

        if src_id not in self.G or tgt_id not in self.G:
            return None

        # find M: successor of src that also has edge to tgt
        successors = list(self.G.successors(src_id))
        random.shuffle(successors)

        for m_id in successors[:20]:
            if m_id == tgt_id or m_id == src_id:
                continue
            if not self.G.has_edge(m_id, tgt_id):
                continue
            m_node = self._index.get(m_id)
            if m_node and "claim" in m_node.domain_tags:
                continue

            link_1 = self._make_link(src_id, m_id)
            link_2 = self._make_link(m_id, tgt_id)
            if not link_1 or not link_2:
                continue

            new_path = list(h.path[:idx]) + [link_1, link_2] + list(h.path[idx + 1:])
            return self._build_child(h, new_path, "mediator_injection")

        return None

    def _convergence_fusion(self, h: Hypothesis) -> Optional[Hypothesis]:
        """Merge two independent hypotheses sharing the same target into a
        multi-biomarker joint prediction.

        Input:  h points to some target T, population has another h2 → T
        Output: h's path + h2's source as co-biomarker, n_independent_paths=2

        Constraint: Only fuse original hypotheses (not already-fused ones) to avoid
        long concatenated source names like "A + B + C + D".
        """
        if len(h.path) < 2:
            return None

        # Reject if h is already a fusion result
        if h.metadata.get("n_independent_paths", 1) > 1:
            return None
        if "+" in h.source_name:
            return None  # defensive: even if metadata was stripped

        # 1. Find candidate partners with same target, different source
        # Also reject partners that are already fusion results
        candidates = [
            h2 for h2 in self._current_population
            if h2.target_id == h.target_id
            and h2.id != h.id
            and h2.source_id != h.source_id
            and len(h2.path) >= 2
            and h2.metadata.get("n_independent_paths", 1) == 1  # only original
            # Belt-and-suspenders: also reject if the source_name already looks fused.
            # (Should be covered by n_independent_paths check, but _build_child
            # previously stripped that metadata, letting fused lineages slip through.)
            and "+" not in h2.source_name
        ]
        if not candidates:
            return None

        # 2. Select partner: prefer higher fitness
        candidates.sort(
            key=lambda x: x.metadata.get("fitness", 0), reverse=True
        )
        h2 = candidates[0]

        # 3. Pick the longer/better path as the structural base
        if len(h.path) >= len(h2.path):
            base, partner = h, h2
        else:
            base, partner = h2, h

        # 4. Build composite source name (only 2 sources max now)
        new_source_name = f"{base.source_name} + {partner.source_name}"

        # 5. Build child using base's path (structural base preserved)
        child = self._build_child(
            base,
            list(base.path),
            "convergence_fusion",
            source_id=base.source_id,
            source_name=new_source_name,
            target_id=base.target_id,
            target_name=base.target_name,
        )
        if not child:
            return None

        # 6. Set convergence metadata — triggers bonus in _score_fitness
        child.metadata["n_independent_paths"] = 2
        child.metadata["co_biomarker_id"] = partner.source_id
        child.metadata["co_biomarker_name"] = partner.source_name
        child.metadata["fusion_partner_id"] = partner.id
        return child

    # ── crossover operators ────────────────────────────────────────────

    def _path_crossover(self, h1: Hypothesis, h2: Hypothesis) -> Optional[Hypothesis]:
        """A→X₁→C + B→X₂→D → A→X₂→C: swap middle nodes."""
        if len(h1.path) < 2 or len(h2.path) < 2:
            return None

        # take source from h1, middle from h2, target from h1
        src_id = h1.source_id
        src_name = h1.source_name
        tgt_id = h1.target_id
        tgt_name = h1.target_name

        # pick middle node from h2 (not source/target of h1)
        mid_candidates = [
            l.to_id for l in h2.path[:-1]
            if l.to_id != src_id and l.to_id != tgt_id
        ]
        if not mid_candidates:
            return None

        mid_id = random.choice(mid_candidates)
        mid_node = self._index.get(mid_id)
        if not mid_node:
            return None

        # check edges: src→mid and mid→tgt
        if not self.G.has_edge(src_id, mid_id) or not self.G.has_edge(mid_id, tgt_id):
            return None

        link_1 = self._make_link(src_id, mid_id)
        link_2 = self._make_link(mid_id, tgt_id)
        if not link_1 or not link_2:
            return None

        new_path = [link_1, link_2]
        return self._build_child(h1, new_path, "path_crossover")

    def _mechanism_transfer(self, h1: Hypothesis, h2: Hypothesis) -> Optional[Hypothesis]:
        """A→X→Y→C + B→X→D → B→X→Y→D: borrow mechanism chain from h1 to h2."""
        if len(h1.path) < 3 or len(h2.path) < 2:
            return None

        # find shared intermediate node between h1 and h2
        h1_nodes = {l.from_id for l in h1.path} | {h1.path[-1].to_id}
        h2_nodes = {l.from_id for l in h2.path} | {h2.path[-1].to_id}
        shared = (h1_nodes & h2_nodes) - {h1.source_id, h1.target_id, h2.source_id, h2.target_id}

        if not shared:
            return None

        shared_id = random.choice(list(shared))

        # find h1 subpath starting from shared_id (X→Y→...→C)
        h1_subpath = []
        found = False
        for link in h1.path:
            if link.from_id == shared_id:
                found = True
            if found:
                h1_subpath.append(link)

        if len(h1_subpath) < 2:
            return None

        # find h2 path to shared_id (B→...→X)
        h2_prefix = []
        for link in h2.path:
            h2_prefix.append(link)
            if link.to_id == shared_id:
                break
        else:
            return None

        # check connection: last node of h2_prefix → first target of h1_subpath
        if not self.G.has_edge(shared_id, h1_subpath[0].to_id):
            return None

        # build: h2_prefix + h1_subpath
        new_path = list(h2_prefix) + list(h1_subpath[1:])  # skip shared_id duplicate

        src_id = h2.source_id
        src_name = h2.source_name
        tgt_id = h1_subpath[-1].to_id
        tgt_name = h1_subpath[-1].to_name

        return self._build_child(h2, new_path, "mechanism_transfer",
                                 source_id=src_id, source_name=src_name,
                                 target_id=tgt_id, target_name=tgt_name)

    # ── PES: Plan (weakness analysis) ──────────────────────────────────

    def _analyze_weakness(self, h: Hypothesis) -> dict:
        """Rule-based weakness analysis: identify the weakest aspect of a hypothesis
        and recommend a mutation operator.

        Returns dict with keys: suggested_operator, reason, weak_hop (optional).
        """
        if len(h.path) < 2:
            return {"suggested_operator": "hop_extension", "reason": "path too short"}

        # Check 1: vague predicates → hop_extension to find more specific path
        vague_hops = [
            i for i, link in enumerate(h.path)
            if link.relation_type in _VAGUE_PREDICATES
        ]
        if vague_hops:
            return {
                "suggested_operator": "hop_extension",
                "reason": f"vague predicate at hop {vague_hops[0]}: {h.path[vague_hops[0]].relation_type}",
                "weak_hop": vague_hops[0],
            }

        # Check 2: low-confidence link → hop_contraction to remove weak link
        confidences = [
            (i, link.confidence * (self.memory.operator_stats.get("_hop_contraction", {}).get("successes", 1) / max(self.memory.operator_stats.get("_hop_contraction", {}).get("trials", 1), 1)))
            for i, link in enumerate(h.path)
        ]
        if confidences:
            weakest_hop = min(confidences, key=lambda x: x[1])
            if weakest_hop[1] < 0.3 and len(h.path) >= 3:
                return {
                    "suggested_operator": "hop_contraction",
                    "reason": f"low confidence at hop {weakest_hop[0]}: {h.path[weakest_hop[0]].from_name} → {h.path[weakest_hop[0]].to_name}",
                    "weak_hop": weakest_hop[0],
                }

        # Check 3: convergence opportunity → convergence_fusion
        same_target = [
            h2 for h2 in self._current_population
            if h2.target_id == h.target_id and h2.id != h.id and h2.source_id != h.source_id
        ]
        if same_target and h.metadata.get("n_independent_paths", 1) < 2:
            return {
                "suggested_operator": "convergence_fusion",
                "reason": f"convergence opportunity: {len(same_target)} other paths to {h.target_name}",
            }

        # Check 4: short path → mediator_injection
        if len(h.path) == 2:
            return {"suggested_operator": "mediator_injection", "reason": "short path, try adding mediator"}

        # Default: weighted random selection
        return {"suggested_operator": "", "reason": "no specific weakness found"}

    # ── helpers ────────────────────────────────────────────────────────

    def _apply_mutation(self, h: Hypothesis) -> Optional[Hypothesis]:
        """PES directed mutation: Plan → Execute → Summarize.

        Phase 1 (Plan): analyze weakness, recommend operator
        Phase 2 (Execute): try recommended operator first, then weighted fallback
        Phase 3 (Summarize): record success/failure in EvolutionMemory
        """
        if random.random() > self.mutation_rate:
            return None

        # Phase 1: Plan — analyze weakness
        weakness = self._analyze_weakness(h)
        recommended = weakness.get("suggested_operator", "")

        # Phase 2: Execute — build operator order (recommended first, then weighted random)
        op_by_name = {op.__name__.lstrip("_"): op for op in self.mutators}
        weights = self.memory.get_operator_weights()

        # ordered list: recommended first, then by weight descending
        remaining = [op for op in self.mutators if op.__name__.lstrip("_") != recommended]
        remaining.sort(key=lambda op: weights.get(op.__name__.lstrip("_"), 1.0), reverse=True)

        if recommended and recommended in op_by_name:
            ordered = [op_by_name[recommended]] + remaining
        else:
            ordered = remaining

        for op in ordered:
            op_name = op.__name__.lstrip("_")
            self.memory.record_trial(op_name)
            try:
                child = op(h)
                if child and self._validate(child) and child.id != h.id:
                    fitness = self._score_fitness(child)
                    self.memory.record_success(op_name, fitness)
                    return child
                else:
                    self.memory.record_failure(op_name, "invalid or identical child")
            except Exception as e:
                logger.debug(f"mutation {op_name} failed: {e}")
                self.memory.record_failure(op_name, str(e))

        return None

    def _apply_crossover(self, h1: Hypothesis, h2: Hypothesis) -> Optional[Hypothesis]:
        """Apply a random crossover operator."""
        op = random.choice(self.crossovers)
        try:
            child = op(h1, h2)
            if child and self._validate(child):
                return child
        except Exception as e:
            logger.debug(f"crossover {op.__name__} failed: {e}")
        return None

    def _make_link(self, src_id: str, tgt_id: str) -> Optional[HypothesisLink]:
        """Create a HypothesisLink from graph edge data.

        Refines vague predicates (`is_associated_with`, `correlates_with`)
        using the rule-based keyword matcher from claim_ingestion. Critic
        already filters these out; without re-refinement, mutation operators
        re-introduce them (~30% of evolved path edges in audits).
        """
        if not self.G.has_edge(src_id, tgt_id):
            return None
        edge_data = self.G.edges[src_id, tgt_id]
        src_node = self._index.get(src_id)
        tgt_node = self._index.get(tgt_id)

        claim_id = edge_data.get("metadata", {}).get("claim_id", "")
        claim_node = self._index.get(claim_id) if claim_id else None

        evidence = {}
        paper = {}
        raw_text = ""
        if claim_node and claim_node.metadata:
            meta = claim_node.metadata
            evidence = meta.get("evidence", {})
            paper = meta.get("source_paper", {})
            raw_text = meta.get("raw_text", "")

        relation_type = edge_data.get("relation_type", "unknown")

        # Rule-based predicate refinement on vague edges
        if relation_type in _VAGUE_PREDICATES and raw_text:
            for refined_pred, patterns in _PREDICATE_KEYWORDS.items():
                if any(p.search(raw_text) for p in patterns):
                    relation_type = refined_pred
                    break

        return HypothesisLink(
            from_id=src_id,
            from_name=src_node.preferred_name if src_node else src_id,
            to_id=tgt_id,
            to_name=tgt_node.preferred_name if tgt_node else tgt_id,
            relation_type=relation_type,
            confidence=edge_data.get("confidence", 0.5),
            claim_id=claim_id,
            raw_text=raw_text,
            evidence=evidence,
            source_paper=paper,
        )

    def _build_child(
        self,
        parent: Hypothesis,
        new_path: list[HypothesisLink],
        operator: str,
        source_id: str = "",
        source_name: str = "",
        target_id: str = "",
        target_name: str = "",
    ) -> Hypothesis:
        """Build a child hypothesis from a parent with a new path."""
        sid = source_id or parent.source_id
        sname = source_name or parent.source_name
        tid = target_id or parent.target_id
        tname = target_name or parent.target_name

        conf = self.engine._compute_confidence_score(new_path)
        nov = self.engine._compute_novelty_score(new_path)
        evi = self.engine._compute_evidence_score(new_path)
        test, test_reason = self.engine._compute_testability_score(new_path)
        claim_ids = [l.claim_id for l in new_path if l.claim_id]

        # Preserve imaging metadata from parent if present
        meta = {
            "parent_id": parent.id,
            "operator": operator,
            "generation": parent.metadata.get("generation", 0) + 1,
        }
        for key in ("dataset", "input_modality", "input_feature", "input_level",
                     "input_tool", "input_region", "input_region_a", "input_region_b",
                     "outcome_type",
                     # Preserve fusion lineage so downstream operators don't re-fuse
                     # already-fused ancestors into multi-plus names ("A + B + C + D")
                     "n_independent_paths", "co_biomarker_id", "co_biomarker_name",
                     "fusion_partner_id"):
            if key in parent.metadata:
                meta[key] = parent.metadata[key]

        child = Hypothesis(
            id=f"EVO:{random.randint(100000, 999999)}",
            hypothesis_type=parent.hypothesis_type,
            source_id=sid,
            source_name=sname,
            target_id=tid,
            target_name=tname,
            path=new_path,
            confidence_score=conf,
            novelty_score=nov,
            evidence_score=evi,
            testability_score=test,
            supporting_claims=claim_ids,
            testability_reason=test_reason,
            metadata=meta,
        )
        child.explanation = self.engine._generate_explanation(child)
        child.composite_score = self.engine._composite_score(child)
        return child

    # ── biological plausibility ────────────────────────────────────────

    def _check_biological_plausibility(self, h: Hypothesis) -> list[str]:
        """Check biological plausibility of hypothesis path.

        Returns a list of violation descriptions. Empty list = plausible.
        """
        violations: list[str] = []

        for i, link in enumerate(h.path):
            src_node = self._index.get(link.from_id)
            tgt_node = self._index.get(link.to_id)
            if not src_node or not tgt_node:
                continue

            src_domains = set(src_node.domain_tags) - {"claim"}
            tgt_domains = set(tgt_node.domain_tags) - {"claim"}

            # 1. Directionality check
            for (s_type, t_type), allowed in DIRECTIONALITY_RULES.items():
                if s_type in src_domains and t_type in tgt_domains:
                    if not allowed:
                        violations.append(
                            f"Step {i+1}: directionality violation — "
                            f"{s_type} -> {t_type} is forbidden "
                            f"({link.from_name} --[{link.relation_type}]--> {link.to_name})"
                        )

            # 2. Predicate-type compatibility check
            compat = PREDICATE_TYPE_COMPAT.get(link.relation_type)
            if compat:
                allowed_src = compat.get("source_types")
                forbidden_src = compat.get("forbidden", set())
                if allowed_src and src_domains:
                    if src_domains & forbidden_src:
                        violations.append(
                            f"Step {i+1}: {link.relation_type} source type "
                            f"{src_domains & forbidden_src} is forbidden "
                            f"({link.from_name})"
                        )
                    elif not (src_domains & allowed_src):
                        violations.append(
                            f"Step {i+1}: {link.relation_type} source type "
                            f"{src_domains} not in allowed types {allowed_src} "
                            f"({link.from_name})"
                        )

                forbidden_tgt = compat.get("forbidden", set())
                if forbidden_tgt and tgt_domains & forbidden_tgt:
                    violations.append(
                        f"Step {i+1}: {link.relation_type} target type "
                        f"{tgt_domains & forbidden_tgt} is forbidden "
                        f"({link.to_name})"
                    )

                allowed_tgt = compat.get("target_types")
                if allowed_tgt and tgt_domains and not (tgt_domains & allowed_tgt):
                    violations.append(
                        f"Step {i+1}: {link.relation_type} target type "
                        f"{tgt_domains} not in allowed types {allowed_tgt} "
                        f"({link.to_name})"
                    )

            # 3. Modality compatibility (soft check — infrastructure only)
            if h.metadata.get("input_modality"):
                modality = h.metadata["input_modality"]
                compatible_features = MODALITY_COMPAT.get(modality)
                if compatible_features:
                    input_feat = h.metadata.get("input_feature", "").lower()
                    feat_normalized = input_feat.replace(" ", "_")
                    if input_feat and not any(
                        cf.lower() in feat_normalized for cf in compatible_features
                    ):
                        pass  # reserved for future strict mode

        return violations

    def _validate(self, h: Hypothesis) -> bool:
        """Validate a hypothesis: all edges exist, no cycles, different domains, no noisy/same-PMID."""
        if not h.path or len(h.path) < 1:
            return False

        # check all edges exist in graph
        for link in h.path:
            if not self.G.has_edge(link.from_id, link.to_id):
                return False

        # check path continuity
        for i in range(len(h.path) - 1):
            if h.path[i].to_id != h.path[i + 1].from_id:
                return False

        # check no cycles (no repeated node IDs except possible start=end for cycles)
        node_ids = [h.path[0].from_id] + [l.to_id for l in h.path]
        if len(node_ids) != len(set(node_ids)):
            return False

        # check source and target are different
        if h.source_id == h.target_id:
            return False

        # filter noisy entities (short tokens like "Id"/"Ca"/"RN" AND
        # nominalized words like "loss"/"Family"/"tissue volumes"). Shares
        # the full noise-word set from hypothesis_engine._is_noisy_entity.
        all_names = {h.source_name, h.target_name}
        for link in h.path:
            all_names.add(link.from_name)
            all_names.add(link.to_name)
        for name in all_names:
            if self.engine._is_noisy_entity(name):
                return False

        # reject single-PMID bridges (both hops cite the same paper)
        if len(h.path) >= 2:
            pmids = set()
            for link in h.path:
                pmid = link.source_paper.get("pmid", "") if isinstance(link.source_paper, dict) else ""
                if pmid:
                    pmids.add(pmid)
            if len(pmids) == 1:
                return False

        # target must be a dataset outcome (diagnosis/cognition/behavior),
        # not an anatomical structure or molecular entity
        if not self.engine._is_dataset_outcome(h):
            return False

        # filter non-measurable entities (CSF, blood, saliva biomarkers)
        for link in h.path:
            for name, nid in [(link.from_name, link.from_id), (link.to_name, link.to_id)]:
                node = self._index.get(nid)
                if node:
                    domains = set(node.domain_tags) - {"claim"}
                    if domains & NON_MEASURABLE_BIOMARKER_TYPES:
                        return False
                for pattern in _NON_MEASURABLE_PATTERNS:
                    if pattern.search(name):
                        return False

        # biological plausibility checks
        violations = self._check_biological_plausibility(h)
        if violations:
            logger.debug(
                f"biological plausibility rejected {h.id}: {violations[0]}"
            )
            return False

        return True
