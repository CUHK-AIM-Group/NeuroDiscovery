"""Knowledge graph manager built on NetworkX."""

from __future__ import annotations

import logging
import json
from collections import Counter
from copy import deepcopy
from typing import Optional

import networkx as nx

from .schema import (
    Claim,
    ConceptNode,
    Edge,
    Evidence,
    EDGE_TIER,
    DISPLAY_TIERS_DEFAULT,
    RELATION_TYPES,
    normalize_domain_tag,
    normalize_domain_tags,
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Directed knowledge graph for neuroscience concepts and relationships."""

    # Relation types that carry no semantic content for hypothesis traversal.
    # `about` is provenance only (claim -> subject/object); traversal must skip it.
    PROVENANCE_RELATIONS: set[str] = {"about"}

    def __init__(self):
        self.G = nx.DiGraph()
        self._index: dict[str, ConceptNode] = {}  # id -> ConceptNode
        self._semantic_view = None  # lazy: built on first access
        self.serialization_metadata: dict = {}  # opt-in storage layout and source declarations
        self.claim_revision = 0  # supported claim mutations invalidate ingestion caches
        self.relation_identities = None  # optional, independently validated exact-term registry
        self.paper_identities = None  # immutable bibliography witnesses; claims are checked on every query
        # G remains the legacy traversal projection, NOT the evidence store.
        # Preserve distinct non-winning records and self loops for lossless IO.
        self._parallel_edges: dict[tuple[str, str], list[dict]] = {}
        self._self_loop_edges: dict[tuple[str, str], list[dict]] = {}

    @property
    def semantic_view(self) -> nx.DiGraph:
        """Read-only view of G with provenance edges (`about`) filtered out.

        Hypothesis traversal MUST use this view, not `self.G`. The full graph
        retains `about` edges for confidence propagation and claim lookup.

        Materialized as a real DiGraph (not subgraph_view) because lazy edge
        filtering destroys all_simple_paths performance on multi-million-edge
        graphs. Memory cost: ~22% of full graph (`about` is 78% of edges).
        Built once on first access.
        """
        if self._semantic_view is None:
            sv = nx.DiGraph()
            sv.add_nodes_from(self.G.nodes(data=True))
            for u, v, d in self.G.edges(data=True):
                if d.get("relation_type", "") not in self.PROVENANCE_RELATIONS:
                    sv.add_edge(u, v, **d)
            logger.info(
                f"semantic_view built: {sv.number_of_nodes()} nodes, "
                f"{sv.number_of_edges()} edges (filtered "
                f"{self.G.number_of_edges() - sv.number_of_edges()} `about` edges)"
            )
            self._semantic_view = sv
        return self._semantic_view

    def invalidate_semantic_view(self) -> None:
        """Drop the cached semantic view. Call after large edge additions/removals."""
        self._semantic_view = None

    # ── node operations ──────────────────────────────────────────────

    def add_concept(self, node: ConceptNode) -> None:
        if self.relation_identities and (node.id in self.relation_identities.node_ids or node.id.startswith("CLM_ATOM:")):
            self.relation_identities = None
        if node.id in self._index:
            # merge: update existing node with new info
            existing = self._index[node.id]
            existing.aliases = list(set(existing.aliases + node.aliases))
            existing.external_ids.update(node.external_ids)
            if not existing.definition and node.definition:
                existing.definition = node.definition
            if not existing.spatial_mapping and node.spatial_mapping:
                existing.spatial_mapping = node.spatial_mapping
            for tag in node.domain_tags:
                if tag not in existing.domain_tags:
                    existing.domain_tags.append(tag)
            for st in node.semantic_types:
                if st not in existing.semantic_types:
                    existing.semantic_types.append(st)
            return

        self._index[node.id] = node
        self.G.add_node(node.id, **node.to_dict())
        if node.id.startswith("CLM:"):
            self.claim_revision += 1

    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        return self._index.get(concept_id)

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._index

    # ── edge operations ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        # Uniqueness was checked across ALL candidate atoms, not just the
        # selected proof atom. New mappings may introduce an alternate CUI.
        if self.relation_identities and edge.relation_type == "maps_to":
            self.relation_identities = None
        if edge.source_id not in self._index:
            logger.warning(f"source node {edge.source_id} not in graph, skipping edge")
            return
        if edge.target_id not in self._index:
            logger.warning(f"target node {edge.target_id} not in graph, skipping edge")
            return
        if edge.relation_type not in RELATION_TYPES:
            logger.debug(f"unknown relation type: {edge.relation_type}")

        pair = (edge.source_id, edge.target_id)
        incoming = edge.to_dict()
        if pair[0] == pair[1]:
            self._remember_edge(self._self_loop_edges, pair, incoming)
            return  # self loops stay out of traversal, but survive saving
        if self.G.has_edge(edge.source_id, edge.target_id):
            existing = self.G.edges[edge.source_id, edge.target_id]
            # Only the traversal representative competes on confidence. Every
            # distinct paper/relation/negation payload remains accessible.
            if edge.confidence > existing.get("confidence", 0):
                self._remember_edge(self._parallel_edges, pair, dict(existing))
                existing.clear()
                existing.update(deepcopy(incoming))
                self._parallel_edges[pair] = [record for record in self._parallel_edges[pair]
                    if self._edge_signature(record) != self._edge_signature(incoming)]
                self.invalidate_semantic_view()
            elif self._edge_signature(existing) != self._edge_signature(incoming):
                self._remember_edge(self._parallel_edges, pair, incoming)
            return

        self._parallel_edges.pop(pair, None)  # stale records after direct G removal
        self.G.add_edge(edge.source_id, edge.target_id, **deepcopy(incoming))
        self.invalidate_semantic_view()

    @staticmethod
    def _edge_signature(record: dict) -> str:
        # JSON equality preserves distinctions such as False versus 0.
        return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    def _remember_edge(self, store: dict, pair: tuple[str, str], record: dict) -> None:
        values = store.setdefault(pair, [])
        signature = self._edge_signature(record)
        if all(self._edge_signature(value) != signature for value in values):
            values.append(deepcopy(record))

    def iter_edge_records(self):
        """Yield detached evidence records; G/semantic_view are lossy projections.

        Repeated identical API insertion is idempotent. Distinct claim owners,
        predicates, negations and attributes are never confidence-deduplicated.
        """
        for sid, tid, primary in self.G.edges(data=True):
            record = dict(primary, source_id=sid, target_id=tid)
            signature = self._edge_signature(record)
            yield deepcopy(record)
            for other in self._parallel_edges.get((sid, tid), ()):
                if self._edge_signature(other) != signature:
                    yield deepcopy(other)
        for (sid, tid), records in self._self_loop_edges.items():
            if sid in self.G and tid in self.G:
                yield from deepcopy(records)

    def iter_relation_evidence(self, *, minimum_claims: int = 1):
        """Group claim evidence without asserting consensus or losing context."""
        from .relation_evidence import group_claim_evidence
        from .correlation_grouping import enabled, IndexTerms
        identities = IndexTerms(self.relation_identities) if enabled(self.serialization_metadata) else self.relation_identities
        return group_claim_evidence((node.metadata for nid, node in self._index.items()
            if nid.startswith("CLM:")), minimum_claims=minimum_claims, identities=identities,
            papers=self.paper_identities)

    def add_edges(self, edges: list[Edge]) -> int:
        count = 0
        for e in edges:
            before = self.G.number_of_edges()
            self.add_edge(e)
            if self.G.number_of_edges() > before:
                count += 1
        return count

    # ── claim operations ───────────────────────────────────────────────

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a Claim by its ID from the graph."""
        node = self._index.get(claim_id)
        if node is None:
            return None
        meta = node.metadata
        if not meta or "subject_name" not in meta:
            return None
        return Claim.from_dict(meta)

    def update_claim(
        self,
        claim_id: str,
        new_evidence: Optional[Evidence] = None,
        new_confidence: Optional[float] = None,
        extra_metadata: Optional[dict] = None,
    ) -> bool:
        """Update a claim's evidence, confidence, and/or metadata in-place.

        Updates:
        1. The claim node's metadata (serialized claim data)
        2. The simplified edge's confidence
        3. The 'about' edges' confidence

        Returns True if the claim was found and updated.
        """
        node = self._index.get(claim_id)
        if node is None:
            logger.warning(f"claim {claim_id} not found in graph")
            return False

        meta = node.metadata
        if not meta or "subject_name" not in meta:
            logger.warning(f"node {claim_id} is not a claim node")
            return False

        # update evidence in metadata
        if new_evidence is not None:
            meta["evidence"] = new_evidence.to_dict()

        # update confidence
        if new_confidence is not None:
            meta["confidence"] = new_confidence

        # merge extra metadata
        if extra_metadata:
            meta.update(extra_metadata)

        # refresh display name
        subject = meta.get("subject_name", "")
        predicate = meta.get("predicate", "")
        obj = meta.get("object_name", "")
        node.preferred_name = f"{subject} {predicate} {obj}"

        # also update the serialized claim in node.metadata so it round-trips
        node.metadata = meta
        self.claim_revision += 1

        # update simplified edge (subject → object)
        conf = new_confidence if new_confidence is not None else meta.get("confidence", 0.5)
        subj_id = meta.get("subject_id", "")
        obj_id = meta.get("object_id", "")
        if subj_id and obj_id and self.G.has_edge(subj_id, obj_id):
            edge_data = self.G.edges[subj_id, obj_id]
            if edge_data.get("metadata", {}).get("claim_id") == claim_id:
                edge_data["confidence"] = conf
            for record in self._parallel_edges.get((subj_id, obj_id), ()):
                if record.get("metadata", {}).get("claim_id") == claim_id:
                    record["confidence"] = conf
            candidates = [dict(edge_data), *self._parallel_edges.get((subj_id, obj_id), ())]
            winner = max(candidates, key=lambda item: item.get("confidence", 0))
            if winner is not candidates[0]:
                edge_data.clear()
                edge_data.update(deepcopy(winner))
            self._parallel_edges[(subj_id, obj_id)] = [deepcopy(item) for item in candidates
                if self._edge_signature(item) != self._edge_signature(winner)]
        for record in self._self_loop_edges.get((subj_id, obj_id), ()):
            if record.get("metadata", {}).get("claim_id") == claim_id:
                record["confidence"] = conf

        # update 'about' edges (claim → subject, claim → object)
        for _, tgt, data in self.G.out_edges(claim_id, data=True):
            if data.get("relation_type") == "about":
                data["confidence"] = conf

        logger.debug(f"updated claim {claim_id}, confidence={conf}")
        self.invalidate_semantic_view()
        return True

    # ── query ────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        concept_id: str,
        relation_type: Optional[str] = None,
        direction: str = "out",  # 'out', 'in', 'both'
    ) -> list[tuple[str, Edge]]:
        """Get neighboring concepts with optional relation filter."""
        results = []
        if direction in ("out", "both"):
            for _, tgt, data in self.G.out_edges(concept_id, data=True):
                if relation_type and data.get("relation_type") != relation_type:
                    continue
                edge = Edge.from_dict(data)
                results.append((tgt, edge))
        if direction in ("in", "both"):
            for src, _, data in self.G.in_edges(concept_id, data=True):
                if relation_type and data.get("relation_type") != relation_type:
                    continue
                edge = Edge.from_dict(data)
                results.append((src, edge))
        return results

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        relation_filter: Optional[set[str]] = None,
    ) -> list[list[tuple[str, str]]]:
        """Find all simple paths between two concepts up to max_hops.

        Returns list of paths, each path is a list of (node_id, relation_type) tuples.
        """
        if source_id not in self.G or target_id not in self.G:
            return []

        subgraph = self.G
        if relation_filter:
            edges_to_keep = [
                (u, v) for u, v, d in self.G.edges(data=True)
                if d.get("relation_type") in relation_filter
            ]
            subgraph = self.G.edge_subgraph(edges_to_keep).copy()

        raw_paths = list(nx.all_simple_paths(
            subgraph, source_id, target_id, cutoff=max_hops
        ))

        # annotate paths with relation types
        annotated = []
        for path in raw_paths:
            annotated_path = []
            for i in range(len(path) - 1):
                edge_data = subgraph.edges[path[i], path[i + 1]]
                annotated_path.append((path[i], edge_data.get("relation_type", "unknown")))
            annotated_path.append((path[-1], ""))
            annotated.append(annotated_path)

        return annotated

    def multi_hop_traverse(
        self,
        start_ids: list[str],
        max_hops: int = 3,
        relation_filter: Optional[set[str]] = None,
    ) -> dict[str, list[list[str]]]:
        """Traverse from multiple starting points, collecting reachable nodes.

        Returns: {start_id: [[path_nodes], ...]}
        """
        results = {}
        for sid in start_ids:
            if sid not in self.G:
                continue
            paths = []
            for target in self.G.nodes():
                if target == sid:
                    continue
                for path in self.find_paths(sid, target, max_hops, relation_filter):
                    paths.append([n for n, _ in path])
            results[sid] = paths
        return results

    def get_subgraph_by_domain(self, domain_tag: str) -> nx.DiGraph:
        """Extract subgraph containing only concepts with a given domain tag."""
        domain_tag = normalize_domain_tag(domain_tag)
        nodes = [
            nid for nid, data in self.G.nodes(data=True)
            if domain_tag in normalize_domain_tags(data.get("domain_tags", []))
        ]
        return self.G.subgraph(nodes).copy()

    def get_subgraph_by_relation(self, relation_type: str) -> nx.DiGraph:
        """Extract subgraph with only edges of a given relation type."""
        edges = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if d.get("relation_type") == relation_type
        ]
        return self.G.edge_subgraph(edges).copy()

    def export_display_subgraph(
        self,
        tiers: Optional[set[str]] = None,
        drop_isolated_nodes: bool = True,
    ) -> nx.DiGraph:
        """Return a subgraph for human display, dropping admin/inverse/provenance edges.

        The full graph (`self.G`) is unchanged — the hypothesis engine still
        traverses every edge. This method produces a *view* with only edges
        whose tier (per `schema.EDGE_TIER`) is in `tiers`.

        Args:
            tiers: which tiers to keep. Defaults to `{"discovery"}`. Pass
                `{"discovery", "skeleton"}` to also surface is_a / part_of.
                Unknown relation types default to "discovery" so the
                long-tail of claim-extracted predicates is still surfaced.
            drop_isolated_nodes: after edge filtering, drop nodes with no
                remaining edges. Useful for visualization; turn off if you
                need to preserve the node universe.

        Returns:
            A NetworkX DiGraph view (deep-copied so callers can mutate it
            without touching the source graph).
        """
        keep = tiers if tiers is not None else set(DISPLAY_TIERS_DEFAULT)
        edges_to_keep = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if EDGE_TIER.get(d.get("relation_type", ""), "discovery") in keep
        ]
        sub = self.G.edge_subgraph(edges_to_keep).copy()
        if drop_isolated_nodes:
            isolated = [n for n in sub.nodes() if sub.degree(n) == 0]
            sub.remove_nodes_from(isolated)
        return sub

    # ── search ───────────────────────────────────────────────────────

    def find_by_name_exact(
        self,
        name: str,
        exclude_source_vocab: Optional[str] = None,
        exclude_id_prefixes: Optional[tuple[str, ...]] = None,
    ) -> list[ConceptNode]:
        """Return concepts whose preferred_name matches `name` case-insensitively.

        Used by seed importers to detect cross-source name collisions: when a
        seed (e.g. VROI:FFA "Fusiform Face Area") would collide with an
        already-extracted claim concept of the same name, callers can merge
        aliases / metadata into the existing node rather than skipping or
        creating a duplicate.

        Args:
            name: target preferred_name (case-insensitive match).
            exclude_source_vocab: skip nodes from this vocab (typically the
                caller's own vocab, so importers don't match themselves).
            exclude_id_prefixes: skip ids starting with any of these prefixes.
        """
        target = name.strip().lower()
        out: list[ConceptNode] = []
        for node in self._index.values():
            if node.preferred_name.strip().lower() != target:
                continue
            if exclude_source_vocab and node.source_vocab == exclude_source_vocab:
                continue
            if exclude_id_prefixes and node.id.startswith(exclude_id_prefixes):
                continue
            out.append(node)
        return out

    def merge_seed_into_existing(
        self,
        target_id: str,
        seed_aliases: list[str],
        seed_metadata: Optional[dict] = None,
        seed_domain_tags: Optional[list[str]] = None,
    ) -> bool:
        """Augment an existing concept with seed-side aliases/metadata/tags.

        Returns True if at least one new alias / tag / metadata key was added.
        Does not change `preferred_name` or `source_vocab` of the target
        node — the existing node remains the canonical entry; the seed
        information enriches it.
        """
        node = self._index.get(target_id)
        if node is None:
            return False
        changed = False
        for a in seed_aliases or []:
            if a and a not in node.aliases:
                node.aliases.append(a)
                changed = True
        for tag in seed_domain_tags or []:
            tag = normalize_domain_tag(tag)
            if tag and tag not in node.domain_tags:
                node.domain_tags.append(tag)
                changed = True
        if seed_metadata:
            md = dict(node.metadata) if node.metadata else {}
            for k, v in seed_metadata.items():
                if k not in md:
                    md[k] = v
                    changed = True
            node.metadata = md
        if changed:
            data = self.G.nodes[target_id]
            data["aliases"] = list(node.aliases)
            data["domain_tags"] = list(node.domain_tags)
            data["metadata"] = dict(node.metadata) if node.metadata else {}
        return changed

    def search_by_name(self, query: str, limit: int = 20) -> list[ConceptNode]:
        """Fuzzy search concepts by preferred_name or aliases."""
        query_lower = query.lower()
        results = []
        for node in self._index.values():
            if query_lower in node.preferred_name.lower():
                results.append(node)
                continue
            for alias in node.aliases:
                if query_lower in alias.lower():
                    results.append(node)
                    break
            if len(results) >= limit:
                break
        return results

    def search_by_domain(self, domain_tag: str) -> list[ConceptNode]:
        domain_tag = normalize_domain_tag(domain_tag)
        return [
            n for n in self._index.values()
            if domain_tag in normalize_domain_tags(n.domain_tags)
        ]

    # ── statistics ───────────────────────────────────────────────────

    def stats(self) -> dict:
        domain_counts = Counter()
        source_counts = Counter()
        relation_counts = Counter()

        for node in self._index.values():
            for tag in node.domain_tags:
                domain_counts[tag] += 1
            source_counts[node.source_vocab] += 1

        for _, _, data in self.G.edges(data=True):
            relation_counts[data.get("relation_type", "unknown")] += 1

        return {
            "n_concepts": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "domains": dict(domain_counts),
            "sources": dict(source_counts),
            "relations": dict(relation_counts),
            "connected_components": nx.number_weakly_connected_components(self.G),
        }

    def __len__(self) -> int:
        return self.G.number_of_nodes()
