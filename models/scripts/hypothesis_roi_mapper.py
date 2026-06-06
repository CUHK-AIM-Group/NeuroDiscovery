"""Map KG hypothesis brain region names to atlas ROI indices.

Parses hypotheses_imaging_hcp.json, extracts brain region names from
fMRI hypotheses, and maps them to AAL-116 ROI indices via fuzzy matching.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
FMRI_ROOT = ROOT / "data" / "braingnn_input"

REGION_TO_AAL = {
    "anterior cingulate": ["Cingulum_Ant_L", "Cingulum_Ant_R"],
    "cingulate cortex": ["Cingulum_Ant_L", "Cingulum_Ant_R", "Cingulum_Mid_L", "Cingulum_Mid_R", "Cingulum_Post_L", "Cingulum_Post_R"],
    "posterior cingulate": ["Cingulum_Post_L", "Cingulum_Post_R"],
    "insula": ["Insula_L", "Insula_R"],
    "insular cortex": ["Insula_L", "Insula_R"],
    "anterior insula": ["Insula_L", "Insula_R"],
    "posterior insula": ["Insula_L", "Insula_R"],
    "caudate": ["Caudate_L", "Caudate_R"],
    "caudate nucleus": ["Caudate_L", "Caudate_R"],
    "angular gyrus": ["Angular_L", "Angular_R"],
    "cuneus": ["Cuneus_L", "Cuneus_R"],
    "precuneus": ["Precuneus_L", "Precuneus_R"],
    "temporal pole": ["Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R", "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R"],
    "occipital lobe": ["Occipital_Sup_L", "Occipital_Sup_R", "Occipital_Mid_L", "Occipital_Mid_R", "Occipital_Inf_L", "Occipital_Inf_R"],
    "inferior occipital gyrus": ["Occipital_Inf_L", "Occipital_Inf_R"],
    "hypothalamus": ["Thalamus_L", "Thalamus_R"],
    "hypothalamus, anterior": ["Thalamus_L", "Thalamus_R"],
    "hypothalamus, middle": ["Thalamus_L", "Thalamus_R"],
    "epithalamus": ["Thalamus_L", "Thalamus_R"],
    "posterior parietal cortex": ["Parietal_Sup_L", "Parietal_Sup_R", "Parietal_Inf_L", "Parietal_Inf_R"],
    "medial temporal structures": ["Hippocampus_L", "Hippocampus_R", "ParaHippocampal_L", "ParaHippocampal_R"],
    "fronto-parietal network": ["Frontal_Mid_L", "Frontal_Mid_R", "Parietal_Inf_L", "Parietal_Inf_R"],
    "frontoparietal network (seitzman)": ["Frontal_Mid_L", "Frontal_Mid_R", "Parietal_Inf_L", "Parietal_Inf_R"],
    "planum temporale": ["Temporal_Sup_L", "Temporal_Sup_R"],
    "olfactory bulb": ["Olfactory_L", "Olfactory_R"],
    "medial parietal areas": ["Precuneus_L", "Precuneus_R", "Parietal_Sup_L", "Parietal_Sup_R"],
    "temporal areas": ["Temporal_Sup_L", "Temporal_Sup_R", "Temporal_Mid_L", "Temporal_Mid_R", "Temporal_Inf_L", "Temporal_Inf_R"],
    "visual perceptual areas of the ventral temporal pathway": ["Fusiform_L", "Fusiform_R", "Temporal_Inf_L", "Temporal_Inf_R"],
}


def get_roi_names(atlas: str) -> list[str]:
    atlas_dir = FMRI_ROOT / atlas
    sample = next(atlas_dir.glob("sub-*.pt"), None)
    if sample is None:
        raise FileNotFoundError(f"No .pt files in {atlas_dir}")
    data = torch.load(sample, weights_only=False)
    return data["roi_names"]


def parse_hypothesis_regions(hypotheses_path: str | Path) -> list[set[str]]:
    """Extract brain region sets from each fMRI hypothesis."""
    with open(hypotheses_path) as f:
        data = json.load(f)

    results = []
    for h in data["hypotheses"]:
        meta = h.get("metadata", {})
        if meta.get("input_modality") != "fMRI":
            continue
        feat = meta.get("input_feature", "")
        regions = set()
        match = re.match(r"functional connectivity between (.+) and (.+)", feat)
        if match:
            regions.add(match.group(1).strip().lower())
            regions.add(match.group(2).strip().lower())
        else:
            region = meta.get("input_region", "")
            if region:
                regions.add(region.lower())
        results.append(regions)
    return results


def regions_to_roi_indices(regions: set[str], roi_names: list[str]) -> list[int]:
    """Map a set of region names to ROI indices in the atlas."""
    matched_names = set()
    for region in regions:
        region_lower = region.lower().strip()
        if region_lower in REGION_TO_AAL:
            matched_names.update(REGION_TO_AAL[region_lower])
        else:
            for key, rois in REGION_TO_AAL.items():
                if key in region_lower or region_lower in key:
                    matched_names.update(rois)
                    break

    name_to_idx = {name: i for i, name in enumerate(roi_names)}
    indices = sorted(name_to_idx[n] for n in matched_names if n in name_to_idx)
    return indices


def build_kg_roi_mask(hypotheses_path: str | Path, atlas: str = "aal_116") -> list[bool]:
    """Build a boolean ROI mask from all fMRI hypotheses.

    Returns a list of length n_roi where True means the ROI is referenced
    by at least one KG hypothesis.
    """
    roi_names = get_roi_names(atlas)
    n_roi = len(roi_names)
    region_sets = parse_hypothesis_regions(hypotheses_path)

    mask = [False] * n_roi
    for regions in region_sets:
        indices = regions_to_roi_indices(regions, roi_names)
        for idx in indices:
            mask[idx] = True

    return mask


def build_hypothesis_roi_pairs(hypotheses_path: str | Path, atlas: str = "aal_116"
                               ) -> list[dict]:
    """Build per-hypothesis ROI pair info for targeted prediction tasks.

    Returns list of dicts with hypothesis metadata + mapped ROI indices.
    """
    roi_names = get_roi_names(atlas)
    with open(hypotheses_path) as f:
        data = json.load(f)

    results = []
    for h in data["hypotheses"]:
        meta = h.get("metadata", {})
        if meta.get("input_modality") != "fMRI":
            continue
        feat = meta.get("input_feature", "")
        match = re.match(r"functional connectivity between (.+) and (.+)", feat)
        if not match:
            continue

        region_a = match.group(1).strip().lower()
        region_b = match.group(2).strip().lower()
        idx_a = regions_to_roi_indices({region_a}, roi_names)
        idx_b = regions_to_roi_indices({region_b}, roi_names)

        if idx_a and idx_b:
            results.append({
                "hypothesis_id": h["id"],
                "target_name": h["target_name"],
                "source_name": h["source_name"],
                "confidence_score": h.get("confidence_score", 0),
                "composite_score": h.get("composite_score", 0),
                "region_a": region_a,
                "region_b": region_b,
                "roi_indices_a": idx_a,
                "roi_indices_b": idx_b,
            })
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hypotheses", default="neurooracle/data/quick/hypotheses_imaging_hcp.json")
    p.add_argument("--atlas", default="aal_116")
    args = p.parse_args()

    hyp_path = ROOT / args.hypotheses
    mask = build_kg_roi_mask(hyp_path, args.atlas)
    roi_names = get_roi_names(args.atlas)

    selected = [roi_names[i] for i, m in enumerate(mask) if m]
    print(f"KG-guided ROI mask: {sum(mask)}/{len(mask)} ROIs selected")
    print(f"Selected ROIs: {selected}")

    pairs = build_hypothesis_roi_pairs(hyp_path, args.atlas)
    print(f"\nHypothesis ROI pairs: {len(pairs)}")
    for pair in pairs[:5]:
        print(f"  {pair['region_a']} (idx {pair['roi_indices_a']}) <-> "
              f"{pair['region_b']} (idx {pair['roi_indices_b']}) -> {pair['target_name']}")
