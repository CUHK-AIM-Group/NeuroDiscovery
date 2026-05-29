"""Ingestion module for DisGeNET gene-disease associations.

Data source: https://www.disgenet.org/
DisGeNET provides gene-disease associations with evidence scores.

Expected input: TSV file from DisGeNET download (curated or all associations).
Columns: geneId, geneSymbol, geneName, diseaseId, diseaseName, score, ...
Download from: https://www.disgenet.org/downloads (free registration required)

DisGeNET disease IDs are UMLS CUIs (e.g. C0036341 for Schizophrenia),
disjoint from the MSH:Dxxx ids the MeSH ingest produced. To prevent the
disease layer from splitting into two unconnected halves (one with MSH
ids carrying ENIGMA structural-imaging edges, the other with DISGENET
ids carrying gene edges), we:

  1. Build a CUI -> MSH-UI bridge via MRCONSO.RRF
     (`_disease_canonicalize.build_cui_to_mesh`).
  2. For each DisGeNET row, if the CUI already maps to an MSH node we
     have, redirect the gene-disease edge to that MSH node and skip
     creating a DISGENET:CUI duplicate.
  3. Otherwise apply a word-boundary neuro filter (with explicit
     non-neuro blacklist - "experimental", "neoplasm", "carcinoma" etc.
     - to avoid the substring trap where 'mental' matches
     'experimental') and create a standalone DISGENET:CUI node only if
     the disease is genuinely neuro and lacks an MSH twin.

Net effect: schizophrenia, MDD, BD, ASD etc. become single nodes that
carry both ENIGMA structural edges and DisGeNET genetic edges, which is
the prerequisite for any GENE -> IM -> DISEASE chain to be traversable.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge
from ._disease_canonicalize import build_cui_to_mesh, canonical_disease_id

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_FILE = DEFAULT_DATA_DIR / "disgenet_gene_disease.tsv"

NEURO_DISEASE_KEYWORDS: tuple[str, ...] = (
    "alzheimer", "parkinson", "huntington", "epilepsy", "schizophrenia",
    "bipolar", "depression", "depressive", "autism", "adhd", "anxiety",
    "dementia", "migraine", "stroke", "multiple sclerosis", "amyotrophic",
    "neurodegenerative", "neurodegeneration", "neuropathy", "neuropathic",
    "brain", "cerebral", "cerebellar", "neurologic", "neurological",
    "psychiatric", "psychiatry", "cognitive", "seizure", "ataxia",
    "lewy body", "frontotemporal", "prion", "insomnia", "narcolepsy",
    "tourette", "obsessive", "compulsive", "post-traumatic stress", "ptsd",
    "schizoaffective", "psychosis", "psychotic",
    "glioblastoma", "glioma", "meningioma", "neuroblastoma", "astrocytoma",
    "neurofibromatosis", "tuberous sclerosis", "rett syndrome",
    "spinal cord", "myelopathy", "encephalopathy", "encephalitis",
    "meningitis", "myasthenia", "muscular dystrophy", "polyneuropathy",
    "essential tremor", "dystonia", "myoclonus", "chorea",
    "intellectual disability", "developmental delay", "autistic",
    "panic disorder", "phobia", "addiction", "substance use",
    "eating disorder", "anorexia nervosa", "bulimia",
    "sleep disorder", "restless legs",
)

# Compile word-boundary regex once. \bmental\b will not match
# 'experimental' (the bug that let liver/breast/diabetes through).
_NEURO_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in NEURO_DISEASE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Hard exclusion: even if a neuro keyword matched, drop diseases whose
# name contains one of these tokens. Catches 'experimental',
# 'mammary neoplasm', drug-induced cardiac etc.
NON_NEURO_BLACKLIST: tuple[str, ...] = (
    "experimental", "neoplasm of", "neoplasms,", "neoplasms of",
    "breast", "mammary", "lung", "liver", "hepatic", "renal", "kidney",
    "bladder", "prostate", "colorectal", "colon", "pancreatic",
    "ovarian", "cervical", "gastric", "esophageal", "thyroid",
    "leukemia", "lymphoma", "myeloma", "carcinoma", "sarcoma",
    "diabetes mellitus, experimental",
    "drug allergy", "leukopenia", "thrombocytopenia",
)
_BLACKLIST_RE = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in NON_NEURO_BLACKLIST) + r")\b",
    re.IGNORECASE,
)


def _is_neuro_disease(disease_name: str) -> bool:
    """Word-boundary neuro check with non-neuro blacklist override.

    A name passes only if a neuro keyword matches at word boundary AND
    the name does not contain a blacklisted non-neuro phrase. Catches
    cases like 'Liver Cirrhosis, Experimental' (matched via 'mental'
    substring under the old logic, now killed by both word-boundary
    and the explicit 'experimental' blacklist).
    """
    if not disease_name:
        return False
    if _BLACKLIST_RE.search(disease_name):
        return False
    return _NEURO_RE.search(disease_name) is not None


def ingest_disgenet(
    kg: KnowledgeGraph,
    filepath: Optional[Path] = None,
    min_score: float = 0.1,
) -> dict:
    """Ingest DisGeNET gene-disease associations into the knowledge graph.

    Args:
        kg: The knowledge graph to populate. MeSH should be ingested
            first so we can canonicalize CUIs onto existing MSH nodes.
        filepath: Path to DisGeNET TSV file.
        min_score: Minimum association score to include (0.0-1.0).

    Returns:
        Summary dict with counts of concepts, edges, and the count of
        rows redirected from a DISGENET:CUI to an existing MSH node.
    """
    filepath = Path(filepath) if filepath else DEFAULT_FILE
    if not filepath.exists():
        logger.warning(
            f"DisGeNET file not found at {filepath}. "
            "Download from https://www.disgenet.org/downloads "
            "and save as TSV in data/raw/ directory."
        )
        return {"concepts_added": 0, "edges_added": 0, "error": "file not found"}

    # Build CUI -> MSH UI bridge (cached after first run).
    cui_to_msh = build_cui_to_mesh()
    kg_msh_uis: set[str] = {
        nid.split(":", 1)[1] for nid in kg.G.nodes() if nid.startswith("MSH:")
    }

    genes_added = 0
    diseases_added = 0
    edges_added = 0
    rows_canonicalized_to_mesh = 0
    rows_kept_as_disgenet = 0
    rows_filtered_non_neuro = 0

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

            # Canonicalize disease ID. If the CUI maps to an MSH node we
            # already have, use that. Otherwise apply neuro-keyword
            # filter and create a standalone DISGENET node.
            canonical = canonical_disease_id(disease_id, cui_to_msh, kg_msh_uis)
            if canonical is not None:
                disease_node_id = canonical
                rows_canonicalized_to_mesh += 1
            else:
                if not _is_neuro_disease(disease_name):
                    rows_filtered_non_neuro += 1
                    continue
                disease_node_id = f"DISGENET:{disease_id}"
                if not kg.has_concept(disease_node_id):
                    kg.add_concept(ConceptNode(
                        id=disease_node_id,
                        preferred_name=disease_name,
                        semantic_types=["T047"],
                        domain_tags=[DomainTag.DISEASE.value],
                        source_vocab="DisGeNET",
                        external_ids={"DisGeNET_ID": disease_id,
                                      "UMLS_CUI": disease_id},
                    ))
                    diseases_added += 1
                rows_kept_as_disgenet += 1

            # add gene node
            gene_node_id = f"GENE:{gene_symbol}"
            if not kg.has_concept(gene_node_id):
                kg.add_concept(ConceptNode(
                    id=gene_node_id,
                    preferred_name=gene_symbol,
                    semantic_types=["T028"],
                    domain_tags=[DomainTag.GENE.value],
                    source_vocab="DisGeNET",
                    definition=gene_name,
                    external_ids={"NCBI_Gene": gene_id},
                ))
                genes_added += 1

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
        "rows_canonicalized_to_mesh": rows_canonicalized_to_mesh,
        "rows_kept_as_disgenet_cui": rows_kept_as_disgenet,
        "rows_filtered_non_neuro": rows_filtered_non_neuro,
    }
    logger.info(f"DisGeNET ingestion complete: {summary}")
    return summary

