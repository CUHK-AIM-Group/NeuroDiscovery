"""Ingestion module for MeSH (Medical Subject Headings) neuroscience subset.

Data source: https://www.nlm.nih.gov/mesh/filelist.html
MeSH provides a controlled vocabulary with tree hierarchy.

Expected input: MeSH XML descriptor files (desc*.xml).
Download from: https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/

Neuroscience-relevant MeSH tree branches:
  - C10    Nervous System Diseases
  - C10.114 Central Nervous System Diseases
  - C10.228 Brain Diseases
  - F03    Mental Disorders
  - A08    Nervous System (anatomy)
  - D14    Neurotransmitters and Neurotransmitter Agents
  - D27    Chemical Actions and Uses (pharmacology subset)
  - G11    Nervous System Physiology
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# MeSH tree number prefixes to include
NEURO_TREE_PREFIXES = [
    "C10",      # Nervous System Diseases
    "C10.114",  # Central Nervous System Diseases
    "C10.228",  # Brain Diseases
    "C10.500",  # Neurodegenerative Diseases
    "C10.574",  # Neuroimmunologic Diseases
    "F01",      # Behavior and Behavior Mechanisms
    "F03",      # Mental Disorders
    "A08",      # Nervous System
    "A08.186",  # Central Nervous System
    "A08.561",  # Peripheral Nervous System
    "D14",      # Neurotransmitters and Neurotransmitter Agents
    "D14.600",  # Neurotransmitter Agents
    "G11",      # Nervous System Physiology
    "G11.561",  # Neurophysiology
]

# Map MeSH tree prefixes to domain tags
PREFIX_TO_DOMAIN = {
    "C10": DomainTag.DISEASE,
    "C10.114": DomainTag.DISEASE,
    "C10.228": DomainTag.DISEASE,
    "C10.500": DomainTag.DISEASE,
    "C10.574": DomainTag.DISEASE,
    "F01": DomainTag.COGNITIVE_FUNCTION,
    "F03": DomainTag.DISEASE,
    "A08": DomainTag.NEUROANATOMY,
    "A08.186": DomainTag.NEUROANATOMY,
    "A08.561": DomainTag.NEUROANATOMY,
    "D14": DomainTag.NEUROTRANSMITTER,
    "D14.600": DomainTag.NEUROTRANSMITTER,
    "G11": DomainTag.COGNITIVE_FUNCTION,
    "G11.561": DomainTag.COGNITIVE_FUNCTION,
}


def _is_neuro_relevant(tree_numbers: list[str]) -> bool:
    """Check if any tree number falls under neuroscience branches."""
    for tn in tree_numbers:
        for prefix in NEURO_TREE_PREFIXES:
            if tn.startswith(prefix):
                return True
    return False


def _get_domain_tags(tree_numbers: list[str]) -> list[str]:
    """Determine domain tags from tree numbers."""
    tags = set()
    for tn in tree_numbers:
        # match longest prefix first
        best_match = ""
        for prefix in PREFIX_TO_DOMAIN:
            if tn.startswith(prefix) and len(prefix) > len(best_match):
                best_match = prefix
        if best_match:
            tags.add(PREFIX_TO_DOMAIN[best_match].value)
    return list(tags)


def _parse_descriptor(elem: ET.Element) -> Optional[tuple[ConceptNode, list[str]]]:
    """Parse a MeSH DescriptorUI element into a ConceptNode.

    Returns (node, tree_numbers) or None.
    """
    ui = ""
    name = ""
    tree_numbers = []
    synonyms = []
    definition = ""

    for child in elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "DescriptorUI":
            ui = child.text.strip() if child.text else ""
        elif tag == "DescriptorName":
            for sub in child:
                subtag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if subtag == "String" and sub.text:
                    name = sub.text.strip()
        elif tag == "TreeNumberList":
            for sub in child:
                subtag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if subtag == "TreeNumber" and sub.text:
                    tree_numbers.append(sub.text.strip())
        elif tag == "ConceptList":
            for concept in child:
                for sub in concept:
                    subtag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                    if subtag == "TermList":
                        for term in sub:
                            for t in term:
                                ttag = t.tag.split("}")[-1] if "}" in t.tag else t.tag
                                if ttag == "String" and t.text:
                                    s = t.text.strip()
                                    if s.lower() != name.lower():
                                        synonyms.append(s)
        elif tag == "ScopeNote" and child.text:
            definition = child.text.strip()

    if not ui or not name or not _is_neuro_relevant(tree_numbers):
        return None

    node = ConceptNode(
        id=f"MSH:{ui}",
        preferred_name=name,
        semantic_types=["T047"],  # will be refined by domain
        domain_tags=_get_domain_tags(tree_numbers),
        source_vocab="MeSH",
        definition=definition,
        aliases=synonyms,
        external_ids={"MeSH_UI": ui},
        metadata={"tree_numbers": tree_numbers},
    )
    return node, tree_numbers


def _build_mesh_hierarchy(nodes_with_trees: list[tuple[ConceptNode, list[str]]]) -> list[Edge]:
    """Build is_a edges from MeSH tree number hierarchy.

    If node A has tree C10.228.140 and node B has tree C10.228, then A is_a B.
    """
    # build lookup: tree_number -> node_id
    tree_to_node: dict[str, str] = {}
    for node, trees in nodes_with_trees:
        for t in trees:
            tree_to_node[t] = node.id

    edges = []
    seen = set()
    for node, trees in nodes_with_trees:
        for t in trees:
            # parent is the tree number with last segment removed
            parts = t.rsplit(".", 1)
            if len(parts) == 2:
                parent_tree = parts[0]
                if parent_tree in tree_to_node and tree_to_node[parent_tree] != node.id:
                    key = (node.id, tree_to_node[parent_tree])
                    if key not in seen:
                        seen.add(key)
                        edges.append(Edge(
                            source_id=node.id,
                            target_id=tree_to_node[parent_tree],
                            relation_type="is_a",
                            source="MeSH_hierarchy",
                            confidence=1.0,
                        ))
    return edges


def ingest_mesh(
    kg: KnowledgeGraph,
    data_dir: Optional[Path] = None,
) -> dict:
    """Ingest MeSH neuroscience subset into the knowledge graph.

    Args:
        kg: The knowledge graph to populate.
        data_dir: Directory containing MeSH XML files (desc*.xml).
                  Defaults to data/raw/.

    Returns:
        Summary dict with counts of concepts and edges added.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    xml_files = sorted(data_dir.glob("desc*.xml"))
    if not xml_files:
        logger.warning(
            f"No MeSH XML files found in {data_dir}. "
            "Download from https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/ "
            "and place desc*.xml files in the data/raw/ directory."
        )
        return {"concepts_added": 0, "edges_added": 0, "error": "no XML files found"}

    nodes_with_trees: list[tuple[ConceptNode, list[str]]] = []
    concepts_added = 0

    for xml_file in xml_files:
        logger.info(f"parsing {xml_file.name}...")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for descriptor in root.iter(f"{ns}DescriptorRecord"):
            result = _parse_descriptor(descriptor)
            if result:
                node, tree_nums = result
                kg.add_concept(node)
                nodes_with_trees.append((node, tree_nums))
                concepts_added += 1

    edges = _build_mesh_hierarchy(nodes_with_trees)
    edges_added = kg.add_edges(edges)

    summary = {"concepts_added": concepts_added, "edges_added": edges_added}
    logger.info(f"MeSH ingestion complete: {summary}")
    return summary
