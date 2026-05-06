"""Ingestion module for Cognitive Atlas cognitive tasks and concepts.

Data source: https://www.cognitiveatlas.org/
Provides a structured ontology of cognitive tasks, concepts, and disorders.

API endpoints:
  /api/v-alpha/task    — all tasks (JSON array)
  /api/v-alpha/concept — all concepts (JSON array)
  /api/v-alpha/disorder — all disorders (JSON array)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

API_BASE = "https://www.cognitiveatlas.org/api/v-alpha"


def download_cognitive_atlas(output_dir: Optional[Path] = None) -> dict[str, Path]:
    """Download Cognitive Atlas data via API.

    Returns dict mapping resource name to local file path.
    """
    import requests

    output_dir = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for resource in ["task", "concept", "disorder"]:
        url = f"{API_BASE}/{resource}"
        logger.info(f"downloading {url}...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            outpath = output_dir / f"cogat_{resource}.json"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(resp.json(), f, ensure_ascii=False, indent=2)
            saved[resource] = outpath
            logger.info(f"saved {outpath}")
        except Exception as e:
            logger.warning(f"failed to download {resource}: {e}")

    return saved


def _parse_task(task: dict) -> Optional[ConceptNode]:
    """Parse a Cognitive Atlas task JSON object into a ConceptNode."""
    task_id = task.get("id", "").strip()
    name = task.get("name", "").strip()
    if not task_id or not name:
        return None

    definition = task.get("definition_text", "") or task.get("description", "") or ""
    aliases = []
    if task.get("alias"):
        aliases.append(task["alias"])

    # extract associated concepts from relationships
    concepts = []
    for rel in task.get("contrasts", []):
        concepts.append(rel.get("id", ""))

    return ConceptNode(
        id=f"COGAT_TASK:{task_id}",
        preferred_name=name,
        domain_tags=[DomainTag.PARADIGM.value],
        source_vocab="CognitiveAtlas",
        definition=definition.strip(),
        aliases=aliases,
        external_ids={"cogat_id": task_id},
        metadata={"type": "task"},
    )


def _parse_concept(concept: dict) -> Optional[ConceptNode]:
    """Parse a Cognitive Atlas concept JSON object into a ConceptNode."""
    concept_id = concept.get("id", "").strip()
    name = concept.get("name", "").strip()
    if not concept_id or not name:
        return None

    definition = concept.get("definition_text", "") or ""

    # determine domain from concept type
    domain_tags = [DomainTag.COGNITIVE_FUNCTION.value]

    return ConceptNode(
        id=f"COGAT_CONCEPT:{concept_id}",
        preferred_name=name,
        domain_tags=domain_tags,
        source_vocab="CognitiveAtlas",
        definition=definition.strip(),
        external_ids={"cogat_id": concept_id},
        metadata={"type": "concept"},
    )


def _parse_disorder(disorder: dict) -> Optional[ConceptNode]:
    """Parse a Cognitive Atlas disorder JSON object into a ConceptNode."""
    dis_id = disorder.get("id", "").strip()
    name = disorder.get("name", "").strip()
    if not dis_id or not name:
        return None

    definition = disorder.get("definition_text", "") or ""

    return ConceptNode(
        id=f"COGAT_DISORDER:{dis_id}",
        preferred_name=name,
        domain_tags=[DomainTag.DISEASE.value],
        source_vocab="CognitiveAtlas",
        definition=definition.strip(),
        external_ids={"cogat_id": dis_id},
        metadata={"type": "disorder"},
    )


def _build_task_concept_edges(tasks: list[dict], concepts: list[dict]) -> list[Edge]:
    """Build edges from tasks to their associated concepts."""
    edges = []
    for task in tasks:
        task_id = task.get("id", "")
        if not task_id:
            continue
        # Cognitive Atlas API may include concept associations
        for rel in task.get("concepts", []):
            concept_id = rel.get("id", "")
            if concept_id:
                edges.append(Edge(
                    source_id=f"COGAT_TASK:{task_id}",
                    target_id=f"COGAT_CONCEPT:{concept_id}",
                    relation_type="associated_with",
                    source="CognitiveAtlas",
                    confidence=0.9,
                ))
    return edges


def ingest_cognitive_atlas(
    kg: KnowledgeGraph,
    data_dir: Optional[Path] = None,
    download: bool = False,
) -> dict:
    """Ingest Cognitive Atlas data into the knowledge graph.

    Args:
        kg: The knowledge graph to populate.
        data_dir: Directory containing cogat_*.json files.
        download: If True, download data from API first.

    Returns:
        Summary dict.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    if download:
        download_cognitive_atlas(data_dir)

    tasks_added = 0
    concepts_added = 0
    disorders_added = 0
    edges_added = 0

    # parse tasks
    task_file = data_dir / "cogat_task.json"
    tasks = []
    if task_file.exists():
        with open(task_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        for task in tasks:
            node = _parse_task(task)
            if node:
                kg.add_concept(node)
                tasks_added += 1
        logger.info(f"parsed {tasks_added} tasks from Cognitive Atlas")
    else:
        logger.info(f"task file not found at {task_file}, skipping")

    # parse concepts
    concept_file = data_dir / "cogat_concept.json"
    concepts = []
    if concept_file.exists():
        with open(concept_file, "r", encoding="utf-8") as f:
            concepts = json.load(f)
        for concept in concepts:
            node = _parse_concept(concept)
            if node:
                kg.add_concept(node)
                concepts_added += 1
        logger.info(f"parsed {concepts_added} concepts from Cognitive Atlas")
    else:
        logger.info(f"concept file not found at {concept_file}, skipping")

    # parse disorders
    disorder_file = data_dir / "cogat_disorder.json"
    if disorder_file.exists():
        with open(disorder_file, "r", encoding="utf-8") as f:
            disorders = json.load(f)
        for disorder in disorders:
            node = _parse_disorder(disorder)
            if node:
                kg.add_concept(node)
                disorders_added += 1
        logger.info(f"parsed {disorders_added} disorders from Cognitive Atlas")
    else:
        logger.info(f"disorder file not found at {disorder_file}, skipping")

    # build edges
    edges = _build_task_concept_edges(tasks, concepts)
    edges_added = kg.add_edges(edges)

    summary = {
        "tasks_added": tasks_added,
        "concepts_added": concepts_added,
        "disorders_added": disorders_added,
        "edges_added": edges_added,
    }
    logger.info(f"Cognitive Atlas ingestion complete: {summary}")
    return summary
