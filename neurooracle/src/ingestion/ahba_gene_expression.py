"""AHBA gene expression -> neuroanatomy importer (GENE -> IM layer 2).

Builds `gene_enriched_in_region` edges from the Allen Human Brain Atlas
(AHBA) microarray expression data. Where HPO layer 1 covers genes whose
mutations cause anatomy-restricted phenotypes, this layer covers genes
whose normal expression is enriched in a specific region across six
post-mortem human brains. The two layers are complementary: HPO is
strong on cortex (broad developmental endpoints) and weak on
subcortical nuclei; AHBA is strong on subcortex (Hypothalamus, Thalamus,
Amygdala, Caudate, Putamen, Pons, Cerebellar Cortex have specialised
expression signatures) and weak on cortex (cortical areas are
expressionally homogeneous in adult human).

Pipeline per donor:
  1. Sample -> NN node mapping via ontology parent-chain rollup. The
     Allen ontology provides a slash-delimited structure_id_path; we
     walk it bottom-up, normalise each ancestor's name + acronym, and
     return the first match against an NN: node's preferred_name or
     alias. Coverage on the six donors is 100% (3702 / 3702 samples).
  2. Probe -> gene_symbol mapping via Probes.csv. Multiple probes may
     map to one gene; we average their expression per sample.
  3. Stream MicroarrayExpression.csv row by row (probe x sample),
     accumulate (gene, sample-region) sums + counts.
  4. For each region with >= MIN_DONOR_SAMPLES samples in this donor,
     compute mean expression per gene, then z-score across the final
     region set within this donor.
  5. Across donors, average per-donor z for genes appearing in
     >= MIN_DONOR_PRESENCE donors. Filter against HGNC approved
     symbols. Threshold at z >= Z_THRESHOLD (default 2.5).
  6. Emit one `gene_enriched_in_region` edge per (gene, region) above
     threshold, with raw z in metadata and confidence linearly mapped
     from z.

Confidence mapping:
  conf = clip(z / 5.0, 0.5, 1.0)
    z = 2.5 -> 0.50  (matches HPO 'Frequent' floor)
    z = 3.0 -> 0.60
    z = 4.0 -> 0.80
    z >= 5.0 -> 1.00 (saturated)

Excluded NN ids (root anatomy with no region-specific signal):
  NN:NN_TAL:10076 Gray Matter, NN:NN_TAL:10077 White Matter,
  NN:10 Telencephalon, NN:800 Diencephalon.

Source: Allen Human Brain Atlas (Hawrylycz 2012, Nature 489), via the
abagen-data mirror at neurooracle/data/raw/abagen-data/microarray.
"""
from __future__ import annotations

import csv
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_AHBA_DIR = DEFAULT_DATA_DIR / "abagen-data" / "microarray"
DEFAULT_HGNC_FILE = DEFAULT_DATA_DIR / "hgnc_complete_set.txt"

DONORS = ("9861", "10021", "12876", "14380", "15496", "15697")

EXCLUDE_NN_AS_TARGET: frozenset[str] = frozenset({
    "NN:NN_TAL:10076",
    "NN:NN_TAL:10077",
    "NN:10",
    "NN:800",
})

MIN_SAMPLES_PER_REGION = 5
MIN_DONOR_SAMPLES = 2
MIN_DONOR_PRESENCE = 3
Z_THRESHOLD = 2.5


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("_", " ").replace("-", " ").strip()
    return " ".join(s.split())


def _name_synonyms(name: str, acronym: str) -> list[str]:
    out: list[str] = []
    n = _normalize(name)
    if n:
        out.append(n)
        if "(" in n:
            out.append(n.split("(")[0].strip())
        for token in (", left", ", right", " left", " right"):
            if n.endswith(token):
                out.append(n[: -len(token)].strip())
    a = _normalize(acronym)
    if a:
        out.append(a)
    return out


def _build_nn_lookup(kg: KnowledgeGraph) -> dict[str, str]:
    """Return normalized-name -> NN node id for all NN: concepts."""
    lookup: dict[str, str] = {}
    for nid in kg.G.nodes():
        if not nid.startswith("NN:"):
            continue
        node = kg._index.get(nid)
        if node is None:
            continue
        names: list[str] = []
        if node.preferred_name:
            names.append(node.preferred_name)
        names.extend(node.aliases or [])
        for nm in names:
            norm = _normalize(nm)
            if norm:
                lookup.setdefault(norm, nid)
            for token in (", left", ", right", " left", " right"):
                if norm.endswith(token):
                    lookup.setdefault(norm[: -len(token)].strip(), nid)
    return lookup


def _load_hgnc_symbols(filepath: Path) -> set[str]:
    """Return the set of HGNC-approved gene symbols (status=Approved).

    Accepts only canonical 'symbol' column entries; previous and alias
    symbols are ignored to avoid resurrecting deprecated names.
    """
    syms: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            if (row.get("status") or "").strip() != "Approved":
                continue
            sym = (row.get("symbol") or "").strip()
            if sym:
                syms.add(sym)
    return syms


def _load_donor_sample_to_nn(
    donor: str,
    nn_lookup: dict[str, str],
    ahba_dir: Path,
) -> list[str]:
    ddir = ahba_dir / f"normalized_microarray_donor{donor}"
    ontology: dict[str, dict] = {}
    with open(ddir / "Ontology.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ontology[row["id"]] = {
                "acronym": row.get("acronym", ""),
                "name": row.get("name", ""),
                "path": row.get("structure_id_path", "") or "",
            }

    def find_nn(sid: str) -> str:
        info = ontology.get(sid)
        if not info:
            return ""
        chain = [s for s in info["path"].strip("/").split("/") if s]
        if not chain or chain[-1] != sid:
            chain.append(sid)
        for cur_id in reversed(chain):
            cur_info = ontology.get(cur_id)
            if not cur_info:
                continue
            for cand in _name_synonyms(cur_info["name"], cur_info["acronym"]):
                if cand in nn_lookup:
                    return nn_lookup[cand]
        return ""

    nn_per_sample: list[str] = []
    with open(ddir / "SampleAnnot.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["structure_id"]
            nn = find_nn(sid)
            if nn in EXCLUDE_NN_AS_TARGET:
                nn = ""
            nn_per_sample.append(nn)
    return nn_per_sample


def _load_donor_probe_to_gene(
    donor: str,
    ahba_dir: Path,
    hgnc_symbols: set[str],
) -> tuple[list[str], list[str]]:
    """Return (probe_ids, gene_symbols), HGNC-filtered.

    Probes whose gene_symbol is empty, missing, or not in HGNC's
    Approved symbol set are dropped. This excludes Agilent probe IDs
    (A_xx_Pxx), LOC/AC tentative genbank symbols, and pseudogenes
    that were never approved.
    """
    ddir = ahba_dir / f"normalized_microarray_donor{donor}"
    probe_ids: list[str] = []
    gene_syms: list[str] = []
    with open(ddir / "Probes.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = (row.get("gene_symbol") or "").strip()
            if not sym or sym in {"na", "NA", "-"}:
                continue
            if sym not in hgnc_symbols:
                continue
            probe_ids.append(row["probe_id"])
            gene_syms.append(sym)
    return probe_ids, gene_syms


def _build_donor_region_gene(
    donor: str,
    nn_lookup: dict[str, str],
    final_regions: list[str],
    ahba_dir: Path,
    hgnc_symbols: set[str],
) -> dict[str, np.ndarray]:
    """Per donor, return gene_symbol -> z-scored expression vector
    aligned to `final_regions` (NaN where this donor has insufficient
    samples). Returns empty dict on any IO failure.
    """
    sample_nn = _load_donor_sample_to_nn(donor, nn_lookup, ahba_dir)
    probe_ids, gene_syms = _load_donor_probe_to_gene(
        donor, ahba_dir, hgnc_symbols,
    )
    if not probe_ids:
        return {}
    probe_set = set(probe_ids)
    probe_to_gene = dict(zip(probe_ids, gene_syms))

    region_to_cols: dict[str, list[int]] = defaultdict(list)
    for col_idx, nn in enumerate(sample_nn):
        if nn:
            region_to_cols[nn].append(col_idx)
    kept_regions = [
        nn for nn, cols in region_to_cols.items()
        if len(cols) >= MIN_DONOR_SAMPLES
    ]
    region_index = {nn: i for i, nn in enumerate(kept_regions)}
    n_total_samples = len(sample_nn)

    gene_sums: dict[str, np.ndarray] = {}
    gene_counts: dict[str, np.ndarray] = {}

    expr_path = (
        ahba_dir / f"normalized_microarray_donor{donor}"
        / "MicroarrayExpression.csv"
    )
    with open(expr_path, "r", encoding="utf-8") as f:
        for line in f:
            comma1 = line.find(",")
            if comma1 < 0:
                continue
            pid = line[:comma1]
            if pid not in probe_set:
                continue
            sym = probe_to_gene.get(pid)
            if not sym:
                continue
            vals = np.fromstring(line[comma1 + 1:], sep=",")
            if vals.shape[0] != n_total_samples:
                continue
            row_sum = np.zeros(len(kept_regions), dtype=np.float64)
            row_cnt = np.zeros(len(kept_regions), dtype=np.int32)
            for nn, cols in region_to_cols.items():
                if nn not in region_index:
                    continue
                ri = region_index[nn]
                row_sum[ri] = vals[cols].sum()
                row_cnt[ri] = len(cols)
            if sym not in gene_sums:
                gene_sums[sym] = row_sum
                gene_counts[sym] = row_cnt
            else:
                gene_sums[sym] += row_sum
                gene_counts[sym] += row_cnt

    n_final = len(final_regions)
    final_idx = {nn: i for i, nn in enumerate(final_regions)}
    z_dict: dict[str, np.ndarray] = {}
    for sym, sums in gene_sums.items():
        cnt = gene_counts[sym]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        full = np.full(n_final, np.nan, dtype=np.float64)
        for src_i, nn in enumerate(kept_regions):
            if nn in final_idx:
                full[final_idx[nn]] = mean[src_i]
        mu = np.nanmean(full)
        sd = np.nanstd(full)
        if not np.isfinite(sd) or sd <= 1e-9:
            continue
        z_dict[sym] = (full - mu) / sd
    return z_dict


def _z_to_confidence(z: float) -> float:
    return float(min(1.0, max(0.5, z / 5.0)))


def ingest_ahba_gene_expression(
    kg: KnowledgeGraph,
    ahba_dir: Optional[Path] = None,
    hgnc_file: Optional[Path] = None,
    z_threshold: float = Z_THRESHOLD,
) -> dict:
    """Ingest AHBA gene-expression -> region edges into the KG.

    Args:
        kg: graph to populate; NeuroNames must be ingested first so
            sample annotations resolve to NN: nodes.
        ahba_dir: directory containing
            normalized_microarray_donor<id>/{Ontology,Probes,SampleAnnot,
            MicroarrayExpression}.csv for each of the six donors.
        hgnc_file: path to HGNC complete_set.txt (TSV, columns include
            symbol + status).
        z_threshold: minimum cross-donor average z to emit an edge.

    Returns:
        Summary dict with counts of regions resolved, donors used,
        genes considered, edges added, and the chosen threshold.
    """
    ahba_dir = Path(ahba_dir) if ahba_dir else DEFAULT_AHBA_DIR
    hgnc_file = Path(hgnc_file) if hgnc_file else DEFAULT_HGNC_FILE
    if not ahba_dir.exists():
        logger.warning(f"AHBA dir not found at {ahba_dir}; skipping.")
        return {"edges_added": 0, "error": "ahba dir missing"}
    if not hgnc_file.exists():
        logger.warning(f"HGNC file not found at {hgnc_file}; skipping.")
        return {"edges_added": 0, "error": "hgnc file missing"}

    nn_lookup = _build_nn_lookup(kg)
    hgnc_symbols = _load_hgnc_symbols(hgnc_file)
    logger.info(
        f"AHBA: NN lookup keys={len(nn_lookup)}, "
        f"HGNC approved symbols={len(hgnc_symbols)}"
    )

    region_total_samples: Counter = Counter()
    donors_present: list[str] = []
    for d in DONORS:
        donor_path = ahba_dir / f"normalized_microarray_donor{d}"
        if not donor_path.exists():
            logger.warning(f"AHBA: donor {d} dir missing at {donor_path}")
            continue
        donors_present.append(d)
        per = _load_donor_sample_to_nn(d, nn_lookup, ahba_dir)
        for nn in per:
            if nn:
                region_total_samples[nn] += 1
    final_regions = sorted(
        nn for nn, n in region_total_samples.items()
        if n >= MIN_SAMPLES_PER_REGION
    )
    logger.info(
        f"AHBA: {len(donors_present)} donors, "
        f"{len(final_regions)} regions >= {MIN_SAMPLES_PER_REGION} samples"
    )
    if not final_regions:
        return {"edges_added": 0, "error": "no regions met sample threshold"}

    donor_z: dict[str, dict[str, np.ndarray]] = {}
    for d in donors_present:
        z_dict = _build_donor_region_gene(
            d, nn_lookup, final_regions, ahba_dir, hgnc_symbols,
        )
        donor_z[d] = z_dict
        logger.info(f"AHBA donor {d}: {len(z_dict)} HGNC genes z-scored")

    all_genes: set[str] = set()
    for d in donor_z.values():
        all_genes.update(d.keys())

    final_z: dict[str, np.ndarray] = {}
    for sym in all_genes:
        stack = [donor_z[d][sym] for d in donors_present
                 if sym in donor_z[d]]
        if len(stack) < MIN_DONOR_PRESENCE:
            continue
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(np.vstack(stack), axis=0)
        final_z[sym] = avg

    edges_added = 0
    genes_added = 0
    edges_per_region: Counter = Counter()
    for sym in sorted(final_z.keys()):
        z_vec = final_z[sym]
        gene_node_id = f"GENE:{sym}"
        gene_added_this_iter = False
        for ri, nn_id in enumerate(final_regions):
            z = z_vec[ri]
            if not np.isfinite(z) or z < z_threshold:
                continue
            if not kg.has_concept(gene_node_id):
                kg.add_concept(ConceptNode(
                    id=gene_node_id,
                    preferred_name=sym,
                    semantic_types=["T028"],
                    domain_tags=[DomainTag.GENE.value],
                    source_vocab="HGNC",
                ))
                if not gene_added_this_iter:
                    genes_added += 1
                    gene_added_this_iter = True
            edge_before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=gene_node_id,
                target_id=nn_id,
                relation_type="gene_enriched_in_region",
                source="AHBA",
                confidence=_z_to_confidence(float(z)),
                evidence_ref="Allen Human Brain Atlas microarray",
                metadata={
                    "z_score": float(z),
                    "n_donors": int(sum(
                        1 for d in donors_present
                        if sym in donor_z[d]
                        and np.isfinite(donor_z[d][sym][ri])
                    )),
                },
            ))
            if kg.G.number_of_edges() > edge_before:
                edges_added += 1
                edges_per_region[nn_id] += 1

    summary = {
        "donors_used": len(donors_present),
        "regions_used": len(final_regions),
        "genes_considered": len(final_z),
        "z_threshold": z_threshold,
        "genes_added": genes_added,
        "edges_added": edges_added,
        "top_regions": edges_per_region.most_common(10),
    }
    logger.info(f"AHBA gene-expression ingestion complete: {summary}")
    return summary
