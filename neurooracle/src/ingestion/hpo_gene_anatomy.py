"""HPO gene -> neuroanatomy importer (GENE -> IM layer 1).

Builds `gene_associated_with_anatomy` edges from HPO's gene-to-phenotype
table by:

  1. Loading the curated HP -> region-label mapping (`HP_TO_REGION_LABEL`)
     from .hpo_term_to_nn.
  2. Resolving each region label to an NN: node already present in the KG
     (NeuroNames must be ingested first).
  3. Streaming hpo_genes_to_phenotype.txt; for every (gene_symbol, hp_id)
     pair where hp_id is in the curated map, emit one edge per
     gene -> NN: target. Edges are deduplicated by (source, target, source).
  4. Adding any HPO genes not already present in the KG as new GENE: nodes
     so the gene -> region edges have valid endpoints. The gene id format
     `GENE:<symbol>` matches DisGeNET's convention; nodes are merged.

Edge confidence comes from HPO frequency annotations (HP:0040280 series):
  HP:0040280 Obligate     -> 1.00   (all carriers manifest)
  HP:0040281 Very frequent-> 0.95
  HP:0040282 Frequent     -> 0.75
  HP:0040283 Occasional   -> 0.30
  HP:0040284 Very rare    -> 0.10
  HP:0040285 Excluded     -> skipped (annotation says phenotype is absent)
  '' / '-' / unknown      -> 0.50   (HPO recommended default for
                                     'frequency unknown')

When the same (gene, hp_id) pair has multiple frequency annotations across
diseases, the maximum confidence is used. Edges from multiple HP terms to
the same region are also collapsed by max-confidence.

Data file expected at <data_dir>/hpo_genes_to_phenotype.txt with columns:
  ncbi_gene_id, gene_symbol, hpo_id, hpo_name, frequency, disease_id

Source: https://hpo.jax.org/data/annotations (CC0).
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge
from .hpo_term_to_nn import HP_TO_REGION_LABEL, REGION_LABEL_TO_CANDIDATES

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_FILE = DEFAULT_DATA_DIR / "hpo_genes_to_phenotype.txt"

# HPO frequency annotation IDs -> confidence floats.
FREQUENCY_TO_CONFIDENCE: dict[str, float] = {
    "HP:0040280": 1.00,   # Obligate
    "HP:0040281": 0.95,   # Very frequent (99-80%)
    "HP:0040282": 0.75,   # Frequent (79-30%)
    "HP:0040283": 0.30,   # Occasional (29-5%)
    "HP:0040284": 0.10,   # Very rare (<5%)
    # HP:0040285 (Excluded) is not present here; rows with it are skipped.
}
EXCLUDED_FREQUENCY = "HP:0040285"
DEFAULT_CONFIDENCE_UNKNOWN = 0.50


def _frequency_to_confidence(freq: str) -> Optional[float]:
    """Map an HPO frequency annotation to a confidence in [0, 1].

    Returns None if the frequency says the phenotype is excluded; the row
    should then be skipped entirely.
    """
    f = (freq or "").strip()
    if not f or f == "-":
        return DEFAULT_CONFIDENCE_UNKNOWN
    if f == EXCLUDED_FREQUENCY:
        return None
    if f in FREQUENCY_TO_CONFIDENCE:
        return FREQUENCY_TO_CONFIDENCE[f]
    # HPO also sometimes embeds raw percentages or "n/m" fractions. Fall
    # back to the unknown default rather than dropping the row, since
    # those are still real annotations.
    return DEFAULT_CONFIDENCE_UNKNOWN


def _resolve_region_targets(
    kg: KnowledgeGraph,
    label_to_candidates: dict[str, list[str]],
) -> dict[str, str]:
    """For each region label, find the NN: node whose preferred_name or
    alias matches a candidate. Returns label -> NN node id.

    Labels that fail to resolve are logged and dropped.
    """
    resolved: dict[str, str] = {}
    for label, candidates in label_to_candidates.items():
        needles = {c.lower().strip() for c in candidates}
        # also accept underscore variant of the label itself
        needles.add(label.lower().replace("_", " "))
        found_id: Optional[str] = None
        for nid in kg.G.nodes():
            if not nid.startswith("NN:"):
                continue
            node = kg._index.get(nid)
            if node is None:
                continue
            names = [(node.preferred_name or "").lower()]
            names += [(a or "").lower() for a in (node.aliases or [])]
            for name in names:
                norm = name.replace("_", " ").strip()
                if norm in needles:
                    found_id = nid
                    break
            if found_id:
                break
        if found_id:
            resolved[label] = found_id
        else:
            logger.warning(
                f"hpo_gene_anatomy: region label {label!r} did not resolve "
                f"to any NN: node (candidates: {candidates}); "
                f"its HP terms will be skipped."
            )
    return resolved


def ingest_hpo_gene_anatomy(
    kg: KnowledgeGraph,
    filepath: Optional[Path] = None,
) -> dict:
    """Ingest HPO gene -> brain-region edges into the knowledge graph.

    Args:
        kg: The knowledge graph to populate. NeuroNames must be ingested
            first so that region labels resolve to NN: nodes.
        filepath: Path to genes_to_phenotype.txt.

    Returns:
        Summary dict: regions_resolved, hp_terms_active, genes_added,
        edges_added, rows_scanned, rows_excluded.
    """
    filepath = Path(filepath) if filepath else DEFAULT_FILE
    if not filepath.exists():
        logger.warning(
            f"HPO gene-anatomy file not found at {filepath}; skipping."
        )
        return {"regions_resolved": 0, "edges_added": 0,
                "error": "file not found"}

    label_to_nn = _resolve_region_targets(kg, REGION_LABEL_TO_CANDIDATES)
    logger.info(
        f"hpo_gene_anatomy: resolved {len(label_to_nn)}/"
        f"{len(REGION_LABEL_TO_CANDIDATES)} region labels to NN: nodes"
    )

    # Restrict the curation map to HP terms whose region resolved.
    active_map = {
        hp: label for hp, label in HP_TO_REGION_LABEL.items()
        if label in label_to_nn
    }

    # First pass: collect (gene_symbol, ncbi_id, nn_id) -> max_confidence
    pair_conf: dict[tuple[str, str, str], float] = {}
    gene_to_ncbi: dict[str, str] = {}
    rows_scanned = 0
    rows_excluded = 0
    with open(filepath, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            rows_scanned += 1
            hp_id = (row.get("hpo_id") or "").strip()
            if hp_id not in active_map:
                continue
            sym = (row.get("gene_symbol") or "").strip()
            ncbi = (row.get("ncbi_gene_id") or "").strip()
            if not sym:
                continue
            conf = _frequency_to_confidence(row.get("frequency", ""))
            if conf is None:
                rows_excluded += 1
                continue
            nn_id = label_to_nn[active_map[hp_id]]
            key = (sym, ncbi, nn_id)
            prev = pair_conf.get(key)
            if prev is None or conf > prev:
                pair_conf[key] = conf
            if ncbi and sym not in gene_to_ncbi:
                gene_to_ncbi[sym] = ncbi

    # Second pass: collapse by (gene, region) keeping max confidence
    # (gene may map to multiple HP terms each landing on same region).
    gene_region_conf: dict[tuple[str, str], float] = {}
    for (sym, _ncbi, nn_id), conf in pair_conf.items():
        key = (sym, nn_id)
        prev = gene_region_conf.get(key)
        if prev is None or conf > prev:
            gene_region_conf[key] = conf

    # Add any missing GENE: nodes so edges have valid endpoints.
    genes_added = 0
    edges_added = 0
    for (sym, nn_id), conf in gene_region_conf.items():
        gene_node_id = f"GENE:{sym}"
        if not kg.has_concept(gene_node_id):
            kg.add_concept(ConceptNode(
                id=gene_node_id,
                preferred_name=sym,
                semantic_types=["T028"],
                domain_tags=[DomainTag.GENE.value],
                source_vocab="HPO",
                external_ids=(
                    {"NCBI_Gene": gene_to_ncbi[sym]}
                    if sym in gene_to_ncbi else {}
                ),
            ))
            genes_added += 1
        elif sym in gene_to_ncbi:
            existing = kg._index.get(gene_node_id)
            if existing and "NCBI_Gene" not in existing.external_ids:
                existing.external_ids["NCBI_Gene"] = gene_to_ncbi[sym]

        edge_before = kg.G.number_of_edges()
        kg.add_edge(Edge(
            source_id=gene_node_id,
            target_id=nn_id,
            relation_type="gene_associated_with_anatomy",
            source="HPO",
            confidence=conf,
            evidence_ref="HPO genes_to_phenotype",
        ))
        if kg.G.number_of_edges() > edge_before:
            edges_added += 1

    summary = {
        "regions_resolved": len(label_to_nn),
        "hp_terms_active": len(active_map),
        "genes_added": genes_added,
        "edges_added": edges_added,
        "rows_scanned": rows_scanned,
        "rows_excluded_freq": rows_excluded,
    }
    logger.info(f"HPO gene-anatomy ingestion complete: {summary}")
    return summary
