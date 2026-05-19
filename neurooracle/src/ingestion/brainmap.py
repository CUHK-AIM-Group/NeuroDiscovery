"""Ingestion module for BrainMap functional activation data.

Data source: https://brainmap.org/
BrainMap provides coordinate-based functional neuroimaging meta-analyses.

Expected input options:
  1. SLEUTH export (.txt) — coordinate tables from BrainMap queries
  2. GingerALE ALE maps — statistical activation maps
  3. BrainMap taxonomy CSV — experimental paradigm definitions

For the prototype, we focus on the taxonomy (paradigms) and coordinate data.
Each experiment in BrainMap links: paradigm → activation coordinates → brain regions.

Coordinate format (SLEUTH export):
  Each experiment block has:
    Reference: <citation>
    Experiment: <paradigm_name>
    Subjects: <n>
    x y z  (one per line, Talairach or MNI coordinates)
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def _parse_sleuth_export(filepath: Path) -> list[dict]:
    """Parse a SLEUTH-format export file from BrainMap.

    Returns list of experiments, each with:
      {'reference': str, 'experiment': str, 'subjects': int, 'coordinates': [(x,y,z), ...]}
    """
    experiments = []
    current = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Reference:"):
                if current:
                    experiments.append(current)
                current = {
                    "reference": line[len("Reference:"):].strip(),
                    "experiment": "",
                    "subjects": 0,
                    "coordinates": [],
                }
            elif line.startswith("Experiment:") and current:
                current["experiment"] = line[len("Experiment:"):].strip()
            elif line.startswith("Subjects:") and current:
                try:
                    current["subjects"] = int(line[len("Subjects:"):].strip())
                except ValueError:
                    pass
            elif current and re.match(r"^-?\d+\.?\d*\s+-?\d+\.?\d*\s+-?\d+\.?\d*$", line):
                parts = line.split()
                current["coordinates"].append((
                    float(parts[0]), float(parts[1]), float(parts[2])
                ))

    if current:
        experiments.append(current)

    return experiments


def _parse_taxonomy_csv(filepath: Path) -> list[dict]:
    """Parse BrainMap taxonomy CSV.

    Expected columns: paradigm_name, category, description, ...
    """
    paradigms = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("paradigm_name", "").strip()
            if name:
                paradigms.append(row)
    return paradigms


def _coordinate_to_region_label(x: float, y: float, z: float) -> str:
    """Approximate mapping of MNI coordinates to brain region labels.

    This is a simplified lookup. For production, use nilearn's
    atlas lookup or the actual BrainMap region assignment.
    """
    # simplified Talairach/MNI quadrant labeling
    hemi = "L" if x < 0 else "R" if x > 0 else "M"
    # major lobes by y-coordinate (very approximate)
    if y > 30:
        lobe = "Frontal"
    elif y > -10:
        lobe = "Parietal"
    elif y > -40:
        lobe = "Temporal"
    else:
        lobe = "Occipital"
    return f"{hemi}_{lobe}_approx"


def ingest_brainmap(
    kg: KnowledgeGraph,
    sleuth_file: Optional[Path] = None,
    taxonomy_file: Optional[Path] = None,
) -> dict:
    """Ingest BrainMap data into the knowledge graph.

    Args:
        kg: The knowledge graph to populate.
        sleuth_file: Path to SLEUTH export file.
        taxonomy_file: Path to BrainMap taxonomy CSV.

    Returns:
        Summary dict.
    """
    data_dir = DEFAULT_DATA_DIR
    paradigms_added = 0
    coactivations_added = 0

    # ingest taxonomy (experimental paradigms)
    tf = Path(taxonomy_file) if taxonomy_file else data_dir / "brainmap_taxonomy.csv"
    if tf.exists():
        paradigms = _parse_taxonomy_csv(tf)
        for p in paradigms:
            name = p.get("paradigm_name", "").strip()
            category = p.get("category", "").strip()
            node_id = f"BM_PARADIGM:{name.replace(' ', '_')}"
            if not kg.has_concept(node_id):
                kg.add_concept(ConceptNode(
                    id=node_id,
                    preferred_name=name,
                    domain_tags=[DomainTag.PARADIGM.value],
                    source_vocab="BrainMap",
                    metadata={"category": category, **{k: v for k, v in p.items() if k not in ("paradigm_name",)}},
                ))
                paradigms_added += 1
    else:
        logger.info(f"BrainMap taxonomy not found at {tf}, skipping paradigm ingestion")

    # ingest coordinate data (SLEUTH export)
    sf = Path(sleuth_file) if sleuth_file else data_dir / "brainmap_sleuth.txt"
    if sf.exists():
        experiments = _parse_sleuth_export(sf)

        # collect all coordinates to find co-activation patterns
        region_experiments: dict[str, list[str]] = {}  # region -> [experiment_ids]

        for i, exp in enumerate(experiments):
            exp_id = f"BM_EXP:{i}"
            exp_name = exp["experiment"] or f"Experiment_{i}"

            # add experiment as a concept
            if not kg.has_concept(exp_id):
                kg.add_concept(ConceptNode(
                    id=exp_id,
                    preferred_name=exp_name,
                    domain_tags=[DomainTag.PARADIGM.value],
                    source_vocab="BrainMap",
                    metadata={
                        "reference": exp["reference"],
                        "n_subjects": exp["subjects"],
                        "n_coordinates": len(exp["coordinates"]),
                    },
                ))

            # map coordinates to approximate regions
            regions = set()
            for x, y, z in exp["coordinates"]:
                region_label = _coordinate_to_region_label(x, y, z)
                regions.add(region_label)

                # ensure region node exists
                region_id = f"BM_REGION:{region_label}"
                if not kg.has_concept(region_id):
                    kg.add_concept(ConceptNode(
                        id=region_id,
                        preferred_name=region_label.replace("_", " "),
                        domain_tags=[DomainTag.NEUROANATOMY.value],
                        source_vocab="BrainMap_approx",
                    ))

                region_experiments.setdefault(region_label, []).append(exp_id)

            # link experiment to its activated regions
            for region_label in regions:
                region_id = f"BM_REGION:{region_label}"
                edge_before = kg.G.number_of_edges()
                kg.add_edge(Edge(
                    source_id=exp_id,
                    target_id=region_id,
                    relation_type="activates",
                    source="BrainMap",
                    confidence=0.8,  # approximate mapping
                ))
                if kg.G.number_of_edges() > edge_before:
                    coactivations_added += 1

        # build co-activation edges between regions that appear together
        region_list = list(region_experiments.keys())
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                r1, r2 = region_list[i], region_list[j]
                shared = set(region_experiments[r1]) & set(region_experiments[r2])
                if len(shared) >= 2:  # at least 2 shared experiments
                    coactivation_score = len(shared) / max(
                        len(region_experiments[r1]), len(region_experiments[r2])
                    )
                    r1_id = f"BM_REGION:{r1}"
                    r2_id = f"BM_REGION:{r2}"
                    edge_before = kg.G.number_of_edges()
                    kg.add_edge(Edge(
                        source_id=r1_id,
                        target_id=r2_id,
                        relation_type="coactivates",
                        source="BrainMap",
                        confidence=min(coactivation_score, 1.0),
                        metadata={"n_shared_experiments": len(shared)},
                    ))
                    if kg.G.number_of_edges() > edge_before:
                        coactivations_added += 1
    else:
        logger.info(f"BrainMap SLEUTH file not found at {sf}, skipping coordinate ingestion")

    summary = {
        "paradigms_added": paradigms_added,
        "edges_added": coactivations_added,
    }
    logger.info(f"BrainMap ingestion complete: {summary}")
    return summary
