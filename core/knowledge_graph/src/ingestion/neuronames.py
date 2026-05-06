"""Ingestion module for NeuroNames (BrainInfo) brain hierarchy data.

Data source: https://braininfo.rprc.washington.edu/
NeuroNames provides a canonical hierarchy of ~2,500 brain structures.

Expected input format: TSV file with columns:
  NN_ID, Name, Latin_Name, Synonyms, Parent_ID, Brodmann_area, ...

If no local file is provided, attempts to download from BrainInfo.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

# BrainInfo download URL for the NeuroNames hierarchy
# Users should download from https://braininfo.rprc.washington.edu/central.aspx
# and place the TSV at the expected path, or provide a custom path.
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_FILE = DEFAULT_DATA_DIR / "neuronames.tsv"


def _normalize_id(raw_id: str) -> str:
    """Normalize an ID, adding NN: prefix only if not already present."""
    if raw_id.startswith("NN:"):
        return raw_id
    return f"NN:{raw_id}"


def _parse_row(row: dict) -> Optional[ConceptNode]:
    """Parse a single TSV row into a ConceptNode."""
    nn_id = row.get("NN_ID", "").strip()
    name = row.get("Name", "").strip()
    if not nn_id or not name:
        return None

    latin_name = row.get("Latin_Name", "").strip()
    synonyms_raw = row.get("Synonyms", "").strip()
    synonyms = [s.strip() for s in synonyms_raw.split(";") if s.strip()] if synonyms_raw else []

    aliases = synonyms.copy()
    if latin_name and latin_name.lower() != name.lower():
        aliases.append(latin_name)

    # determine if Brodmann area
    ba = row.get("Brodmann_area", "").strip()

    return ConceptNode(
        id=_normalize_id(nn_id),
        preferred_name=name,
        semantic_types=["T023"],  # Body Part, Organ, or Organ Component
        domain_tags=[DomainTag.NEUROANATOMY.value],
        source_vocab="NeuroNames",
        aliases=aliases,
        external_ids={"NN_ID": nn_id},
        metadata={"latin_name": latin_name, "brodmann_area": ba} if ba else {"latin_name": latin_name},
    )


def _build_hierarchy(rows: list[dict]) -> list[Edge]:
    """Build part_of edges from parent_id relationships."""
    edges = []
    for row in rows:
        nn_id = row.get("NN_ID", "").strip()
        parent_id = row.get("Parent_ID", "").strip()
        if nn_id and parent_id:
            edges.append(Edge(
                source_id=_normalize_id(nn_id),
                target_id=_normalize_id(parent_id),
                relation_type="part_of",
                source="NeuroNames",
                confidence=1.0,
            ))
    return edges


def ingest_neuronames(
    kg: KnowledgeGraph,
    filepath: Optional[Path] = None,
) -> dict:
    """Ingest NeuroNames data into the knowledge graph.

    Args:
        kg: The knowledge graph to populate.
        filepath: Path to NeuroNames TSV file. Defaults to data/raw/neuronames.tsv.

    Returns:
        Summary dict with counts of concepts and edges added.
    """
    filepath = Path(filepath) if filepath else DEFAULT_FILE
    if not filepath.exists():
        logger.warning(
            f"NeuroNames file not found at {filepath}. "
            "Download from https://braininfo.rprc.washington.edu/central.aspx "
            "and save as TSV with columns: NN_ID, Name, Latin_Name, Synonyms, Parent_ID, Brodmann_area"
        )
        return {"concepts_added": 0, "edges_added": 0, "error": "file not found"}

    rows = []
    concepts_added = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            node = _parse_row(row)
            if node:
                kg.add_concept(node)
                concepts_added += 1

    edges = _build_hierarchy(rows)
    edges_added = kg.add_edges(edges)

    summary = {"concepts_added": concepts_added, "edges_added": edges_added}
    logger.info(f"NeuroNames ingestion complete: {summary}")
    return summary
