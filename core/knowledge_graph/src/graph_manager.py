"""Knowledge graph manager built on NetworkX."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import networkx as nx

from .schema import Claim, ConceptNode, Edge, Evidence, RELATION_TYPES

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Directed knowledge graph for neuroscience concepts and relationships."""

    def __init__(self):
        self.G = nx.DiGraph()
        self._index: dict[str, ConceptNode] = {}  # id -> ConceptNode

    # ── node operations ──────────────────────────────────────────────

    def add_concept(self, node: ConceptNode) -> None:
        if node.id in self._index:
            # merge: update existing node with new info
            existing = self._index[node.id]
            existing.aliases = list(set(existing.aliases + node.aliases))
            existing.external_ids.update(node.external_ids)
            if not existing.definition and node.definition:
                existing.definition = node.definition
            if not existing.atlas_mapping and node.atlas_mapping:
                existing.atlas_mapping = node.atlas_mapping
            for tag in node.domain_tags:
                if tag not in existing.domain_tags:
                    existing.domain_tags.append(tag)
            for st in node.semantic_types:
                if st not in existing.semantic_types:
                    existing.semantic_types.append(st)
            return

        self._index[node.id] = node
        self.G.add_node(node.id, **node.to_dict())

    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        return self._index.get(concept_id)

    def has_concept(self, concept_id: str) -> bool:
        return concept_id in self._index

    # ── edge operations ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        if edge.source_id not in self._index:
            logger.warning(f"source node {edge.source_id} not in graph, skipping edge")
            return
        if edge.target_id not in self._index:
            logger.warning(f"target node {edge.target_id} not in graph, skipping edge")
            return
        if edge.relation_type not in RELATION_TYPES:
            logger.debug(f"unknown relation type: {edge.relation_type}")

        # for DiGraph: use relation_type as key to allow multiple relation types
        # between the same pair of nodes
        key = edge.relation_type
        if self.G.has_edge(edge.source_id, edge.target_id):
            existing = self.G.edges[edge.source_id, edge.target_id]
            if existing.get("relation_type") == edge.relation_type:
                # same relation type: keep higher confidence
                if edge.confidence > existing.get("confidence", 0):
                    self.G.edges[edge.source_id, edge.target_id].update(edge.to_dict())
                return
            # different relation type: store as metadata on the edge
            # since DiGraph only supports one edge per pair, we keep the higher-confidence one
            if edge.confidence > existing.get("confidence", 0):
                self.G.edges[edge.source_id, edge.target_id].update(edge.to_dict())
            return

        self.G.add_edge(edge.source_id, edge.target_id, **edge.to_dict())

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

        # update simplified edge (subject → object)
        conf = new_confidence if new_confidence is not None else meta.get("confidence", 0.5)
        subj_id = meta.get("subject_id", "")
        obj_id = meta.get("object_id", "")
        if subj_id and obj_id and self.G.has_edge(subj_id, obj_id):
            edge_data = self.G.edges[subj_id, obj_id]
            if edge_data.get("metadata", {}).get("claim_id") == claim_id:
                edge_data["confidence"] = conf

        # update 'about' edges (claim → subject, claim → object)
        for _, tgt, data in self.G.out_edges(claim_id, data=True):
            if data.get("relation_type") == "about":
                data["confidence"] = conf

        logger.debug(f"updated claim {claim_id}, confidence={conf}")
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
        nodes = [
            nid for nid, data in self.G.nodes(data=True)
            if domain_tag in data.get("domain_tags", [])
        ]
        return self.G.subgraph(nodes).copy()

    def get_subgraph_by_relation(self, relation_type: str) -> nx.DiGraph:
        """Extract subgraph with only edges of a given relation type."""
        edges = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if d.get("relation_type") == relation_type
        ]
        return self.G.edge_subgraph(edges).copy()

    # ── search ───────────────────────────────────────────────────────

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
        return [n for n in self._index.values() if domain_tag in n.domain_tags]

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
