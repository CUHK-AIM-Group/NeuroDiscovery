"""Ingestion module for DisGeNET gene-disease associations.

Data source: https://www.disgenet.org/
DisGeNET provides gene-disease associations with evidence scores.

Expected input: TSV file from DisGeNET download (curated or all associations).
Columns: geneId, geneSymbol, geneName, diseaseId, diseaseName, score, ...
Download from: https://www.disgenet.org/downloads (free registration required)

For the prototype, we filter for nervous system / mental disorder diseases
by matching diseaseName or using associated MeSH/OMIM IDs.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_FILE = DEFAULT_DATA_DIR / "disgenet_gene_disease.tsv"

# Keywords to identify nervous-system-related diseases
NEURO_DISEASE_KEYWORDS = [
    "alzheimer", "parkinson", "huntington", "epilepsy", "schizophrenia",
    "bipolar", "depression", "autism", "adhd", "anxiety", "dementia",
    "migraine", "stroke", "multiple sclerosis", "amyotrophic",
    "neurodegenerat", "neuropath", "brain", "cerebral", "neurolog",
    "psychiatr", "mental", "cognitive", "seizure", "ataxia",
    "lewy body", "frontotemporal", "prion", "insomnia", "narcolepsy",
    "tourette", "ocd", "ptsd", "schizoaffect", "psychos",
    "glioblastoma", "meningioma", "neuroblastoma", "astrocytoma",
    "neurofibromatosis", "tuberous sclerosis", "rett",
]


def _is_neuro_disease(disease_name: str) -> bool:
    """Check if disease name matches neuroscience keywords."""
    name_lower = disease_name.lower()
    return any(kw in name_lower for kw in NEURO_DISEASE_KEYWORDS)


def ingest_disgenet(
    kg: KnowledgeGraph,
    filepath: Optional[Path] = None,
    min_score: float = 0.1,
) -> dict:
    """Ingest DisGeNET gene-disease associations into the knowledge graph.

    Args:
        kg: The knowledge graph to populate.
        filepath: Path to DisGeNET TSV file.
        min_score: Minimum association score to include (0.0-1.0).

    Returns:
        Summary dict with counts of concepts and edges added.
    """
    filepath = Path(filepath) if filepath else DEFAULT_FILE
    if not filepath.exists():
        logger.warning(
            f"DisGeNET file not found at {filepath}. "
            "Download from https://www.disgenet.org/downloads "
            "and save as TSV in data/raw/ directory."
        )
        return {"concepts_added": 0, "edges_added": 0, "error": "file not found"}

    genes_added = 0
    diseases_added = 0
    edges_added = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_id = row.get("geneId", "").strip()
            gene_symbol = row.get("geneSymbol", "").strip()
            gene_name = row.get("geneName", "").strip()
            disease_id = row.get("diseaseId", "").strip()
            disease_name = row.get("diseaseName", "").strip()
            score_str = row.get("score", "0").strip()

            if not gene_id or not disease_id:
                continue

            try:
                score = float(score_str)
            except ValueError:
                continue

            if score < min_score:
                continue

            # filter for neuro-related diseases
            if not _is_neuro_disease(disease_name):
                continue

            # add gene node
            gene_node_id = f"GENE:{gene_symbol}"
            if not kg.has_concept(gene_node_id):
                kg.add_concept(ConceptNode(
                    id=gene_node_id,
                    preferred_name=gene_symbol,
                    semantic_types=["T028"],  # Gene or Genome
                    domain_tags=[DomainTag.GENE.value],
                    source_vocab="DisGeNET",
                    definition=gene_name,
                    external_ids={"NCBI_Gene": gene_id},
                ))
                genes_added += 1

            # add disease node
            # Use DisGeNET disease ID, but also check if MeSH equivalent exists
            disease_node_id = f"DISGENET:{disease_id}"
            if not kg.has_concept(disease_node_id):
                kg.add_concept(ConceptNode(
                    id=disease_node_id,
                    preferred_name=disease_name,
                    semantic_types=["T047"],  # Disease or Syndrome
                    domain_tags=[DomainTag.DISEASE.value],
                    source_vocab="DisGeNET",
                    external_ids={"DisGeNET_ID": disease_id},
                ))
                diseases_added += 1

            # add gene-disease association edge
            edge_before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=gene_node_id,
                target_id=disease_node_id,
                relation_type="gene_associated_with_disease",
                source="DisGeNET",
                confidence=score,
                evidence_ref=f"DisGeNET score: {score:.3f}",
            ))
            if kg.G.number_of_edges() > edge_before:
                edges_added += 1

    summary = {
        "genes_added": genes_added,
        "diseases_added": diseases_added,
        "edges_added": edges_added,
    }
    logger.info(f"DisGeNET ingestion complete: {summary}")
    return summary
