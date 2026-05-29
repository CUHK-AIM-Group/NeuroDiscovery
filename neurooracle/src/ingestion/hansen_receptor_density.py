"""Phase 1 ingester: Hansen 2022 receptor / transporter PET density per ROI.

Adds GENE -> NN_region edges with predicate `receptor_density_in`, where
GENE is the receptor / transporter gene (e.g. GENE:HTR2A) and the region is
a Desikan-Killiany cortical ROI or Aseg subcortical ROI canonicalised to NN.

Why a new predicate (vs. AHBA's `gene_enriched_in_region`):
  - AHBA measures mRNA expression; Hansen maps measure receptor protein
    density via PET tracers. They diverge - e.g. SLC6A4 mRNA correlates
    weakly with DASB SERT density (Hansen 2022 Fig 4).
  - Downstream hypothesis paths can ask "drugs binding receptor X act on
    regions where X is densely expressed at the protein level", which is a
    different (and pharmacologically more actionable) claim than mRNA.

Pipeline:
  1. Read `data/raw/hansen_receptors/` for pre-downloaded MNI152 NIfTI maps.
     If a map is missing, download directly from OSF using neuromaps' osf.json
     index. We bypass `neuromaps.fetch_annotation` because it is incompatible
     with nilearn >= 0.13 (passes str where Path is expected).
  2. For each map, compute mean receptor density per Desikan-Killiany ROI
     (34 cortical L+R merged, 7 Aseg subcortical L+R merged) using the
     atlas shipped with abagen.
  3. Z-score across regions per receptor; emit edges where |z| >= z_threshold
     (default 1.5; receptor maps have far fewer regions than AHBA so we
     loosen the threshold from 2.5).

Receptor -> tracer -> gene table: Hansen et al. 2022 Nat Neurosci, Table 1.
Each receptor has one canonical tracer; for receptors with multiple available
tracers we pick the one Hansen used in the released atlas.
"""

from __future__ import annotations

import logging
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from ..graph_manager import KnowledgeGraph
from ..schema import Edge
from .enigma_disease_im import ASEG_ROI_TO_NN, DK_ROI_TO_NN

logger = logging.getLogger(__name__)


# Receptor / transporter -> (canonical tracer, HGNC gene symbol, neurotransmitter system)
# Tracer keys match neuromaps OSF `desc` field for source=beliveau2017/...
# Gene symbol is the HGNC primary that codes the receptor / transporter
# protein measured by the tracer.
RECEPTOR_TRACERS: list[dict] = [
    # Serotonergic
    {"receptor": "5-HT1A",  "gene": "HTR1A",   "tracer": "way100635",       "source": "savli2012",     "system": "serotonergic"},
    {"receptor": "5-HT1B",  "gene": "HTR1B",   "tracer": "p943",            "source": "gallezot2010",  "system": "serotonergic"},
    {"receptor": "5-HT2A",  "gene": "HTR2A",   "tracer": "cimbi36",         "source": "beliveau2017",  "system": "serotonergic"},
    {"receptor": "5-HT4",   "gene": "HTR4",    "tracer": "sb207145",        "source": "beliveau2017",  "system": "serotonergic"},
    {"receptor": "5-HT6",   "gene": "HTR6",    "tracer": "gsk215083",       "source": "radnakrishnan2018", "system": "serotonergic"},
    {"receptor": "5-HTT",   "gene": "SLC6A4",  "tracer": "dasb",            "source": "beliveau2017",  "system": "serotonergic"},
    # Dopaminergic
    {"receptor": "D1",      "gene": "DRD1",    "tracer": "sch23390",        "source": "kaller2017",    "system": "dopaminergic"},
    {"receptor": "D2",      "gene": "DRD2",    "tracer": "fallypride",      "source": "jaworska2020",  "system": "dopaminergic"},
    {"receptor": "DAT",     "gene": "SLC6A3",  "tracer": "fpcit",           "source": "dukart2018",    "system": "dopaminergic"},
    # Noradrenergic
    {"receptor": "NET",     "gene": "SLC6A2",  "tracer": "methylreboxetine","source": "hesse2017",     "system": "noradrenergic"},
    # Cholinergic
    {"receptor": "VAChT",   "gene": "SLC18A3", "tracer": "feobv",           "source": "aghourian2017", "system": "cholinergic"},
    {"receptor": "M1",      "gene": "CHRM1",   "tracer": "lsn3172176",      "source": "naganawa2020",  "system": "cholinergic"},
    {"receptor": "alpha4beta2", "gene": "CHRNA4", "tracer": "flubatine",    "source": "hillmer2016",   "system": "cholinergic"},
    # GABAergic
    {"receptor": "GABA-A",  "gene": "GABRA1",  "tracer": "flumazenil",      "source": "norgaard2021",  "system": "gabaergic"},
    # Glutamatergic
    {"receptor": "mGluR5",  "gene": "GRM5",    "tracer": "abp688",          "source": "smart2019",     "system": "glutamatergic"},
    # Opioid
    {"receptor": "MOR",     "gene": "OPRM1",   "tracer": "carfentanil",     "source": "kantonen2020",  "system": "opioid"},
    # Cannabinoid
    {"receptor": "CB1",     "gene": "CNR1",    "tracer": "omar",            "source": "normandin2015", "system": "cannabinoid"},
    # Histaminergic
    {"receptor": "H3",      "gene": "HRH3",    "tracer": "gsk189254",       "source": "gallezot2017",  "system": "histaminergic"},
    # Synaptic density (not a neurotransmitter receptor but Hansen includes it)
    {"receptor": "SV2A",    "gene": "SV2A",    "tracer": "ucbj",            "source": "finnema2016",   "system": "synaptic_density"},
]


def _osf_index() -> list[dict]:
    """Read neuromaps' osf.json so we can resolve OSF file ids without using
    neuromaps' broken fetch_annotation."""
    import json
    from pkg_resources import resource_filename
    p = resource_filename("neuromaps", os.path.join("datasets", "data", "osf.json"))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["annotations"]


def _resolve_tracer_url(index: list[dict], source: str, tracer: str) -> Optional[str]:
    """Find the MNI152 volumetric file id for (source, tracer).

    OSF API URL format: files.osf.io/v1/resources/<project>/providers/osfstorage/<file>
    (the bare https://osf.io/<id>/download form returns the project landing page,
    not the file).
    """
    for e in index:
        if e.get("source") == source and e.get("desc") == tracer and e.get("space") == "MNI152":
            url = e.get("url")
            if isinstance(url, list) and len(url) >= 2:
                return f"https://files.osf.io/v1/resources/{url[0]}/providers/osfstorage/{url[1]}"
    return None


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=120) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        return dest.stat().st_size > 0
    except Exception as e:
        logger.warning(f"  download failed for {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _build_dk_label_volume() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (label_data, affine, info_df) for abagen's Desikan-Killiany atlas."""
    import abagen
    a = abagen.fetch_desikan_killiany()
    img = nib.load(a["image"])
    info = pd.read_csv(a["info"])
    return img.get_fdata().astype(int), img.affine, img, info


def _resample_to_atlas(src_img, atlas_img):
    from nilearn.image import resample_to_img
    return resample_to_img(src_img, atlas_img, interpolation="linear",
                           force_resample=True, copy_header=True)


def _roi_means(receptor_data: np.ndarray, label_data: np.ndarray,
               info: pd.DataFrame) -> dict[str, float]:
    """Aggregate L+R per ROI label name."""
    out_per_label: dict[str, list[float]] = {}
    flat_recv = receptor_data.ravel()
    flat_lab = label_data.ravel()
    for _, row in info.iterrows():
        lab_id = int(row["id"])
        mask = flat_lab == lab_id
        if not mask.any():
            continue
        vals = flat_recv[mask]
        vals = vals[np.isfinite(vals) & (vals != 0)]
        if vals.size < 5:
            continue
        out_per_label.setdefault(row["label"], []).append(float(vals.mean()))
    return {k: float(np.mean(v)) for k, v in out_per_label.items()}


def _label_to_nn(label: str, structure: str) -> Optional[str]:
    """Map a DK / Aseg label name to a NN node id using the same lookups
    enigma_disease_im uses, so atlas/disease/receptor edges all share targets."""
    if structure == "cortex":
        return DK_ROI_TO_NN.get(label)
    aseg_alias = {
        "thalamusproper": "thal",
        "caudate": "caud",
        "putamen": "put",
        "pallidum": "pal",
        "accumbensarea": "accumb",
        "hippocampus": "hippo",
        "amygdala": "amyg",
    }
    key = aseg_alias.get(label)
    return ASEG_ROI_TO_NN.get(key) if key else None


def ingest_hansen_receptor_density(
    kg: KnowledgeGraph,
    raw_dir: Path,
    z_threshold: float = 1.0,
) -> dict:
    """Compute receptor density per ROI, emit GENE -> NN edges. Idempotent.

    Args:
        kg: KnowledgeGraph (must already have GENE:* and NN:* nodes loaded).
        raw_dir: Directory under data/raw to cache tracer NIfTI files.
        z_threshold: |z| cutoff for emitting an edge. Default 1.0 (DK has
            ~41 regions across 19 receptor maps, vs. AHBA's 55 regions across
            16574 genes - we loosen vs. AHBA's 2.5 since each map yields
            far fewer edges).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    osf_index = _osf_index()

    # Step 1: download all tracer maps that resolve.
    available: list[dict] = []
    missing: list[str] = []
    for spec in RECEPTOR_TRACERS:
        tracer, source = spec["tracer"], spec["source"]
        fname = f"source-{source}_desc-{tracer}_space-MNI152.nii.gz"
        dest = raw_dir / fname
        url = _resolve_tracer_url(osf_index, source, tracer)
        if url is None:
            logger.warning(f"  no OSF entry for {source}/{tracer}, skipping")
            missing.append(spec["receptor"])
            continue
        if not _download(url, dest):
            missing.append(spec["receptor"])
            continue
        spec_with_path = dict(spec)
        spec_with_path["path"] = dest
        available.append(spec_with_path)

    logger.info(f"Hansen: {len(available)}/{len(RECEPTOR_TRACERS)} tracer maps available; missing: {missing}")
    if not available:
        return {"receptors_used": 0, "regions_used": 0, "edges_added": 0,
                "missing_receptors": missing}

    # Step 2: build DK label volume.
    import abagen
    a = abagen.fetch_desikan_killiany()
    atlas_img = nib.load(a["image"])
    info = pd.read_csv(a["info"])
    label_data = atlas_img.get_fdata().astype(int)

    # Resolve label -> NN id once
    label_nn: dict[str, str] = {}
    for _, row in info.iterrows():
        nn = _label_to_nn(row["label"], "cortex" if row["structure"] == "cortex"
                          else "subcortex")
        if nn and kg.has_concept(nn):
            label_nn[row["label"]] = nn
    logger.info(f"Hansen: {len(label_nn)} ROI labels resolve to NN nodes in KG")

    # Step 3: per-tracer ROI means + z-scores + edges
    edges_added = 0
    receptors_used = 0
    per_receptor_edges: dict[str, int] = {}
    for spec in available:
        gene_id = f"GENE:{spec['gene']}"
        if not kg.has_concept(gene_id):
            logger.info(f"  GENE:{spec['gene']} not in KG, skipping {spec['receptor']}")
            continue
        try:
            r_img = nib.load(spec["path"])
            r_img = _resample_to_atlas(r_img, atlas_img)
            r_data = r_img.get_fdata()
        except Exception as e:
            logger.warning(f"  failed to load/resample {spec['tracer']}: {e}")
            continue
        roi_vals = _roi_means(r_data, label_data, info)
        # restrict to labels with a NN target
        roi_vals = {k: v for k, v in roi_vals.items() if k in label_nn}
        if len(roi_vals) < 8:
            logger.warning(f"  {spec['receptor']}: only {len(roi_vals)} usable ROIs, skip")
            continue
        vals = np.array(list(roi_vals.values()), dtype=float)
        mu, sd = float(vals.mean()), float(vals.std(ddof=1) or 1.0)
        receptors_used += 1
        n_emit = 0
        for label, val in roi_vals.items():
            z = (val - mu) / sd
            if abs(z) < z_threshold:
                continue
            nn_id = label_nn[label]
            confidence = float(min(1.0, max(0.5, abs(z) / 4.0)))
            kg.add_edge(Edge(
                source_id=gene_id,
                target_id=nn_id,
                relation_type="receptor_density_in",
                source="HansenReceptor2022",
                confidence=confidence,
                evidence_ref=f"Hansen 2022 PET: {spec['tracer']} ({spec['source']}); z={z:.2f}",
                metadata={
                    "receptor": spec["receptor"],
                    "tracer": spec["tracer"],
                    "tracer_source": spec["source"],
                    "neurotransmitter_system": spec["system"],
                    "z_score": float(z),
                    "roi_label": label,
                },
            ))
            edges_added += 1
            n_emit += 1
        per_receptor_edges[spec["receptor"]] = n_emit
        logger.info(f"  {spec['receptor']:14s} ({spec['tracer']:18s}): "
                    f"{len(roi_vals)} ROIs, {n_emit} edges (|z|>={z_threshold})")

    summary = {
        "receptors_used": receptors_used,
        "regions_used": len(label_nn),
        "edges_added": edges_added,
        "missing_receptors": missing,
        "per_receptor_edges": per_receptor_edges,
    }
    logger.info(f"Hansen receptor-density ingestion complete: {summary}")
    return summary


__all__ = ["ingest_hansen_receptor_density", "RECEPTOR_TRACERS"]
