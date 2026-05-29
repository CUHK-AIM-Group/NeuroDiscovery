"""ENIGMA Toolbox case-control summary stats -> DISEASE -> IM edges.

Builds `is_associated_with` and `correlates_with` edges from ENIGMA's
published meta/mega-analysis effect sizes, using the
`enigmatoolbox.datasets` summary_statistics CSVs as the primary source
(Lariviere 2021 Nat Methods, MICA-MNI/ENIGMA on GitHub). Each CSV is a
single working group's case-vs-control comparison reporting Cohen's d
and FDR-corrected p per Desikan-Killiany cortical ROI (CortThick or
CortSurf, 68 rows = 34 ROI x 2 hemispheres) or Aseg subcortical ROI
(SubVol, 16 rows = 8 ROI x 2 hemispheres).

Edge direction: disease -> NN region.
Predicate:
  - 'reduces' if d_icv < 0 and FDR-significant (q < 0.05): disease
    reduces this region's volume/thickness/area.
  - 'increases' if d_icv > 0 and FDR-significant.
  - 'correlates_with' otherwise (effect estimated but not significant
    at q < 0.05); kept because absence of significance is informative
    for a KG.
Confidence: |d| -> [0.50, 1.00] linearly clipped at |d| = 0.30 (large
effect by Cohen convention is 0.5; meta-analysis effects in
psychiatry are typically 0.05-0.30, so we cap higher).
  conf = clip(0.50 + 1.5 * |d|, 0.5, 1.0)
For non-significant rows (q >= 0.05): conf = max(0.30, conf - 0.20).

The ENIGMA Toolbox includes ~13 disorder umbrellas with multiple
case-vs-control variants (e.g. parkinsons has HY1/2/3/4-5 stage
subgroups, MDD has adolescent/adult x firstepisode/recurrent, OCD has
medicated/unmedicated). To avoid an explosion of redundant edges, we
restrict ingestion to a curated whitelist of ONE primary case-vs-CN
comparison per disorder (e.g. mddadult_case-controls_*, ocdadult_case-
controls_SubVol, parkinsons_case-controls_*_PDvsCN) and aggregate
hemispheres so each (disease, NN-region, modality) keeps the larger
|d| effect. The metadata of each edge records the source CSV and the
exact n_controls / n_patients so provenance is preserved.

Modality -> imaging marker:
  - CortThick -> cortical thickness
  - CortSurf  -> cortical surface area
  - SubVol    -> regional volume
These are stored in edge metadata as 'modality' so downstream
analysis can filter by IM type.
"""
from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..graph_manager import KnowledgeGraph
from ..schema import ConceptNode, DomainTag, Edge

logger = logging.getLogger(__name__)


ASEG_ROI_TO_NN: dict[str, str] = {
    "accumb":  "NN:704",            # Nucleus Accumbens
    "amyg":    "NN:902",            # Amygdala
    "caud":    "NN:701",            # Caudate Nucleus
    "hippo":   "NN:901",            # Hippocampus
    "pal":     "NN:703",            # Globus Pallidus
    "put":     "NN:702",            # Putamen
    "thal":    "NN:801",            # Thalamus
    "LatVent": "NN:3001",           # Lateral Ventricle
}

DK_ROI_TO_NN: dict[str, str] = {
    "bankssts":                 "NN:302",            # STG (bank of STS folds into STG)
    "caudalanteriorcingulate":  "NN:501",            # Anterior Cingulate Cortex
    "caudalmiddlefrontal":      "NN:113",            # Middle Frontal Gyrus
    "cuneus":                   "NN:NN_TAL:10029",   # Cuneus
    "entorhinal":               "NN:307",            # Entorhinal Cortex
    "fusiform":                 "NN:305",            # Fusiform Gyrus
    "inferiorparietal":         "NN:207",            # Inferior Parietal Lobule
    "inferiortemporal":         "NN:304",            # Inferior Temporal Gyrus
    "isthmuscingulate":         "NN:502",            # Posterior Cingulate Cortex
    "lateraloccipital":         "NN:NN_HO:20022",    # Lateral Occipital Cortex (sup div)
    "lateralorbitofrontal":     "NN:102",            # Orbitofrontal Cortex
    "lingual":                  "NN:NN_TAL:10044",   # Lingual Gyrus
    "medialorbitofrontal":      "NN:102",            # Orbitofrontal Cortex
    "middletemporal":           "NN:303",            # Middle Temporal Gyrus
    "parahippocampal":          "NN:308",            # Parahippocampal Gyrus
    "paracentral":              "NN:NN_TAL:10051",   # Paracentral Lobule
    "parsopercularis":          "NN:NN_HO:20006",    # IFG pars opercularis
    "parsorbitalis":            "NN:114",            # Inferior Frontal Gyrus
    "parstriangularis":         "NN:NN_HO:20005",    # IFG pars triangularis
    "pericalcarine":            "NN:NN_HO:20024",    # Intracalcarine Cortex
    "postcentral":              "NN:NN_TAL:10053",   # Postcentral Gyrus
    "posteriorcingulate":       "NN:NN_TAL:10054",   # Posterior Cingulate
    "precentral":               "NN:NN_TAL:10055",   # Precentral Gyrus
    "precuneus":                "NN:205",            # Precuneus
    "rostralanteriorcingulate": "NN:501",            # Anterior Cingulate Cortex
    "rostralmiddlefrontal":     "NN:113",            # Middle Frontal Gyrus
    "superiorfrontal":          "NN:112",            # Superior Frontal Gyrus
    "superiorparietal":         "NN:206",            # Superior Parietal Lobule
    "superiortemporal":         "NN:302",            # Superior Temporal Gyrus
    "supramarginal":            "NN:203",            # Supramarginal Gyrus
    "frontalpole":              "NN:NN_HO:20001",    # Frontal Pole
    "temporalpole":             "NN:306",            # Temporal Pole
    "transversetemporal":       "NN:NN_TAL:10069",   # Transverse Temporal Gyrus
    "insula":                   "NN:600",            # Insular Cortex
}

# Curated whitelist: file_basename -> (disease_node_id, label, modality).
# One primary case-vs-CN comparison per disorder, kept across all three
# modalities (CortThick, CortSurf, SubVol) where available.
DISEASE_FILE_MAP: dict[str, tuple[str, str, str]] = {
    # Major depression: adult primary, plus pooled-age SubVol
    "mddadult_case-controls_CortThick.csv":   ("MSH:D003865", "MDD adult", "CortThick"),
    "mddadult_case-controls_CortSurf.csv":    ("MSH:D003865", "MDD adult", "CortSurf"),
    "mdd_case-controls_SubVol.csv":           ("MSH:D003865", "MDD pooled", "SubVol"),
    # Schizophrenia
    "Schizophrenia_case-controls_CortThick.csv": ("MSH:D012559", "SCZ", "CortThick"),
    "scz_case-controls_CortSurf.csv":            ("MSH:D012559", "SCZ", "CortSurf"),
    "scz_case-controls_SubVol.csv":              ("MSH:D012559", "SCZ", "SubVol"),
    # Bipolar (adult arms have most n; SubVol typeI is the canonical)
    "bd_case-controls_CortThick_adult.csv":   ("MSH:D001714", "BD adult", "CortThick"),
    "bd_case-controls_CortSurf_adult.csv":    ("MSH:D001714", "BD adult", "CortSurf"),
    "bd_case-controls_SubVol_typeI.csv":      ("MSH:D001714", "BD-I", "SubVol"),
    # Autism (mega-analysis preferred where dual)
    "asd_mega-analysis_case-controls_CortThick.csv": ("MSH:D000067877", "ASD mega", "CortThick"),
    "asd_meta-analysis_case-controls_SubVol.csv":    ("MSH:D000067877", "ASD meta", "SubVol"),
    # ADHD: allages preferred (pooled across age groups)
    "adhdallages_case-controls_CortThick.csv": ("MSH:D001289", "ADHD allages", "CortThick"),
    "adhdallages_case-controls_CortSurf.csv":  ("MSH:D001289", "ADHD allages", "CortSurf"),
    "adhdallages_case-controls_SubVol.csv":    ("MSH:D001289", "ADHD allages", "SubVol"),
    # OCD: adult primary
    "ocdadults_case-controls_CortThick.csv":   ("MSH:D009771", "OCD adult", "CortThick"),
    "ocdadults_case-controls_CortSurf.csv":    ("MSH:D009771", "OCD adult", "CortSurf"),
    "ocdadult_case-controls_SubVol.csv":       ("MSH:D009771", "OCD adult", "SubVol"),
    # Anorexia nervosa: case-controls (no global covariate)
    "anorexia_case-controls_CortThick.csv":    ("MSH:D000856", "AN", "CortThick"),
    "anorexia_case-controls_CortSurf.csv":     ("MSH:D000856", "AN", "CortSurf"),
    "anorexia_case-controls_SubVol.csv":       ("MSH:D000856", "AN", "SubVol"),
    # Parkinsons: pooled PDvsCN canonical
    "parkinsons_case-controls_CortThick_PDvsCN.csv": ("MSH:D010300", "PD vs CN", "CortThick"),
    "parkinsons_case-controls_CortSurf_PDvsCN.csv":  ("MSH:D010300", "PD vs CN", "CortSurf"),
    "parkinsons_case-controls_Subvol_PDvsCN.csv":    ("MSH:D010300", "PD vs CN", "SubVol"),
    # Epilepsy: pooled allepi
    "allepi_case-controls_CortThick.csv":      ("MSH:D004827", "All epilepsy", "CortThick"),
    "allepi_case-controls_SubVol.csv":         ("MSH:D004827", "All epilepsy", "SubVol"),
    # Psychosis (CHR-PS+ vs CN as canonical clinical-high-risk transition arm)
    "psychosis_case-controls_CortThick_CHR-PS+vsCN_postComBatmegaanalysis.csv":
        ("MSH:D011618", "CHR-PS+ vs CN", "CortThick"),
    "psychosis_case-controls_CortSurf_CHR-PS+vsCN_postComBatmegaanalysis.csv":
        ("MSH:D011618", "CHR-PS+ vs CN", "CortSurf"),
    "psychosis_case-controls_SubVol_CHR-PS+vsCN_postComBatmegaanalysis.csv":
        ("MSH:D011618", "CHR-PS+ vs CN", "SubVol"),
    # 22q11.2 deletion syndrome: no disease node yet; importer creates it.
    "22q_case-controls_CortThick.csv":         ("MSH:D004062", "22q vs CN", "CortThick"),
    "22q_case-controls_CortSurf.csv":          ("MSH:D004062", "22q vs CN", "CortSurf"),
    "22q_case-controls_SubVol.csv":            ("MSH:D004062", "22q vs CN", "SubVol"),
}

# 22q11.2 needs a brand-new node: MSH:D004062 is the actual MeSH ID for
# "DiGeorge Syndrome" (=22q11.2 deletion syndrome).
EXTRA_DISEASE_NODES: dict[str, ConceptNode] = {
    "MSH:D004062": ConceptNode(
        id="MSH:D004062",
        preferred_name="DiGeorge Syndrome",
        semantic_types=["T047"],
        domain_tags=[DomainTag.DISEASE.value],
        source_vocab="MeSH",
        aliases=["22q11.2 deletion syndrome",
                 "Velocardiofacial syndrome",
                 "22q deletion syndrome"],
        external_ids={"MeSH": "D004062"},
    ),
}

FDR_SIG = 0.05


def _strip_hemisphere_prefix(structure: str) -> tuple[str, str]:
    """Return (roi_code, hemi). Handles both DK 'L_xxx'/'R_xxx' and Aseg
    'Lxxx'/'Rxxx' / 'LLatVent'/'RLatVent'.
    """
    s = structure.strip()
    if s.startswith("L_") or s.startswith("R_"):
        return s[2:], s[0]
    if s.startswith("LLatVent"):
        return "LatVent", "L"
    if s.startswith("RLatVent"):
        return "LatVent", "R"
    if s.startswith("L") and len(s) > 1 and s[1].islower():
        return s[1:], "L"
    if s.startswith("R") and len(s) > 1 and s[1].islower():
        return s[1:], "R"
    return s, ""


def _d_to_predicate_and_confidence(
    d: float, fdr_p: Optional[float],
) -> tuple[str, float]:
    sig = fdr_p is not None and fdr_p < FDR_SIG
    abs_d = abs(d)
    conf = max(0.5, min(1.0, 0.5 + 1.5 * abs_d))
    if not sig:
        conf = max(0.30, conf - 0.20)
    if sig and d < 0:
        return "reduces", conf
    if sig and d > 0:
        return "increases", conf
    return "correlates_with", conf


def _aggregate_by_region(
    rows: list[dict],
) -> dict[str, dict]:
    """Collapse hemispheres: keep the larger |d| effect per ROI base.

    Returns roi_code -> {d, fdr_p, hemi, n_controls, n_patients,
                         abs_d}.
    """
    out: dict[str, dict] = {}
    for r in rows:
        struct = r.get("Structure")
        if not struct:
            continue
        roi, hemi = _strip_hemisphere_prefix(struct)
        try:
            d = float(r.get("d_icv", "")) if r.get("d_icv") not in ("", None) else None
        except (ValueError, TypeError):
            d = None
        if d is None:
            continue
        try:
            p_raw = r.get("fdr_p", "")
            if p_raw in ("", None) or (isinstance(p_raw, float)):
                fdr_p = float(p_raw) if p_raw not in ("", None) else None
            else:
                p_str = str(p_raw).strip()
                fdr_p = float(p_str.lstrip("<"))  # handles '<0.001'
        except (ValueError, TypeError):
            fdr_p = None
        try:
            n_c = int(float(r.get("n_controls", 0) or 0))
        except (ValueError, TypeError):
            n_c = 0
        try:
            n_p = int(float(r.get("n_patients", 0) or 0))
        except (ValueError, TypeError):
            n_p = 0
        prev = out.get(roi)
        if prev is None or abs(d) > prev["abs_d"]:
            out[roi] = {
                "d": d, "fdr_p": fdr_p, "hemi": hemi,
                "n_controls": n_c, "n_patients": n_p,
                "abs_d": abs(d),
            }
    return out


def _read_csv_normalized(fpath: Path) -> list[dict]:
    """Read an ENIGMA CSV with auto-detected delimiter and stripped
    column names. Some files (notably Schizophrenia_case-controls_
    CortThick.csv) use ';' instead of ',', and some (notably
    anorexia_case-controls_SubVol.csv) leave trailing whitespace on
    'pobs'/etc. Field names are normalised with .strip(), and we keep
    the canonical key 'Structure' addressable.
    """
    with open(fpath, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    delim = ";" if sample.count(";") > sample.count(",") else ","
    rows_out: list[dict] = []
    with open(fpath, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter=delim)
        if rdr.fieldnames:
            cleaned_names = [(c or "").strip() for c in rdr.fieldnames]
            rdr.fieldnames = cleaned_names  # type: ignore[assignment]
        for raw in rdr:
            row = {(k or "").strip(): v for k, v in raw.items()}
            rows_out.append(row)
    return rows_out


def ingest_enigma_disease_im(
    kg: KnowledgeGraph,
    enigma_summary_dir: Optional[Path] = None,
) -> dict:
    """Ingest ENIGMA Toolbox summary stats as DISEASE -> NN edges.

    Args:
        kg: graph; NeuroNames + MeSH must be ingested first.
        enigma_summary_dir: directory containing ENIGMA Toolbox's
            summary_statistics CSVs. Defaults to the installed
            enigmatoolbox package path.
    """
    if enigma_summary_dir is None:
        try:
            import enigmatoolbox.datasets as _ds
            enigma_summary_dir = (
                Path(os.path.dirname(_ds.__file__)) / "summary_statistics"
            )
        except Exception as e:
            logger.warning(f"enigmatoolbox not importable: {e}; skipping.")
            return {"edges_added": 0, "error": "enigmatoolbox missing"}
    enigma_summary_dir = Path(enigma_summary_dir)
    if not enigma_summary_dir.exists():
        logger.warning(
            f"ENIGMA summary_statistics dir not found at {enigma_summary_dir}"
        )
        return {"edges_added": 0, "error": "summary dir missing"}

    # Ensure 22q disease node exists before edges land on it.
    diseases_added = 0
    for did, node in EXTRA_DISEASE_NODES.items():
        if not kg.has_concept(did):
            kg.add_concept(node)
            diseases_added += 1

    # Verify all targeted disease nodes exist.
    missing_disease: set[str] = set()
    for fname, (did, _, _) in DISEASE_FILE_MAP.items():
        if not kg.has_concept(did):
            missing_disease.add(did)
    if missing_disease:
        logger.warning(
            f"ENIGMA: missing disease nodes {sorted(missing_disease)}; "
            "edges to those will be skipped."
        )

    # Verify NN region targets exist.
    all_target_nn = set(ASEG_ROI_TO_NN.values()) | set(DK_ROI_TO_NN.values())
    missing_nn = {n for n in all_target_nn if not kg.has_concept(n)}
    if missing_nn:
        logger.warning(
            f"ENIGMA: NN target nodes missing: {sorted(missing_nn)}"
        )

    files_used = 0
    files_missing: list[str] = []
    edges_added = 0
    edges_significant = 0
    edges_per_disease: defaultdict[str, int] = defaultdict(int)

    for fname, (did, label, modality) in DISEASE_FILE_MAP.items():
        if did in missing_disease:
            continue
        fpath = enigma_summary_dir / fname
        if not fpath.exists():
            files_missing.append(fname)
            continue
        files_used += 1
        rows = _read_csv_normalized(fpath)
        agg = _aggregate_by_region(rows)
        is_subvol = modality == "SubVol"
        roi_map = ASEG_ROI_TO_NN if is_subvol else DK_ROI_TO_NN
        for roi_code, info in agg.items():
            nn_id = roi_map.get(roi_code)
            if nn_id is None:
                continue
            if nn_id in missing_nn:
                continue
            d = info["d"]
            fdr_p = info["fdr_p"]
            predicate, conf = _d_to_predicate_and_confidence(d, fdr_p)
            sig = fdr_p is not None and fdr_p < FDR_SIG
            edge_before = kg.G.number_of_edges()
            kg.add_edge(Edge(
                source_id=did,
                target_id=nn_id,
                relation_type=predicate,
                source="ENIGMA",
                confidence=conf,
                evidence_ref=f"ENIGMA Toolbox: {fname}",
                metadata={
                    "cohens_d": float(d),
                    "fdr_p": float(fdr_p) if fdr_p is not None else None,
                    "modality": modality,  # CortThick | CortSurf | SubVol
                    "n_controls": info["n_controls"],
                    "n_patients": info["n_patients"],
                    "comparison": label,
                    "hemisphere_kept": info["hemi"],
                },
            ))
            if kg.G.number_of_edges() > edge_before:
                edges_added += 1
                edges_per_disease[did] += 1
                if sig:
                    edges_significant += 1

    summary = {
        "files_used": files_used,
        "files_missing": len(files_missing),
        "diseases_added": diseases_added,
        "edges_added": edges_added,
        "edges_significant_fdr": edges_significant,
        "edges_per_disease": dict(edges_per_disease),
    }
    logger.info(f"ENIGMA disease-IM ingestion complete: {summary}")
    if files_missing:
        logger.info(
            f"ENIGMA: {len(files_missing)} expected files not found: "
            f"{files_missing[:5]}{'...' if len(files_missing) > 5 else ''}"
        )
    return summary
