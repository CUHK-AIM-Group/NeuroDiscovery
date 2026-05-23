"""Map hypothesis input_region to AAL-116 ROI indices.

Provides two KG-guided strategies:
  A) ROI mask: boolean mask selecting relevant ROIs for subgraph training
  B) Edge boost: matrix of boost factors for FC edges between relevant ROIs
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

AAL_116_NAMES = [
    "Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R", "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R", "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R", "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R", "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R", "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R", "Rectus_L", "Rectus_R",
    "Insula_L", "Insula_R", "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R", "Cingulum_Post_L", "Cingulum_Post_R",
    "Hippocampus_L", "Hippocampus_R", "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R", "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R", "Lingual_L", "Lingual_R",
    "Occipital_Sup_L", "Occipital_Sup_R", "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R", "Fusiform_L", "Fusiform_R",
    "Postcentral_L", "Postcentral_R", "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R", "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R", "Precuneus_L", "Precuneus_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R", "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R", "Pallidum_L", "Pallidum_R",
    "Thalamus_L", "Thalamus_R", "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R", "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R", "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R", "Cerebelum_Crus1_L", "Cerebelum_Crus1_R",
    "Cerebelum_Crus2_L", "Cerebelum_Crus2_R", "Cerebelum_3_L", "Cerebelum_3_R",
    "Cerebelum_4_5_L", "Cerebelum_4_5_R", "Cerebelum_6_L", "Cerebelum_6_R",
    "Cerebelum_7b_L", "Cerebelum_7b_R", "Cerebelum_8_L", "Cerebelum_8_R",
    "Cerebelum_9_L", "Cerebelum_9_R", "Cerebelum_10_L", "Cerebelum_10_R",
    "Vermis_1_2", "Vermis_3", "Vermis_4_5", "Vermis_6",
    "Vermis_7", "Vermis_8", "Vermis_9", "Vermis_10",
]

REGION_TO_AAL: dict[str, list[str]] = {
    "Hypothalamus": ["Thalamus_L", "Thalamus_R"],
    "Hypothalamus, Middle": ["Thalamus_L", "Thalamus_R"],
    "Hypothalamus, Anterior": ["Thalamus_L", "Thalamus_R"],
    "Insular Cortex": ["Insula_L", "Insula_R"],
    "Anterior Insula": ["Insula_L", "Insula_R"],
    "Posterior Insula": ["Insula_L", "Insula_R"],
    "Anterior Cingulate": ["Cingulum_Ant_L", "Cingulum_Ant_R"],
    "Angular Gyrus": ["Angular_L", "Angular_R"],
    "Caudate": ["Caudate_L", "Caudate_R"],
    "Caudate Nucleus": ["Caudate_L", "Caudate_R"],
    "Temporal Pole": ["Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
                      "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R"],
    "Occipital Lobe": ["Occipital_Sup_L", "Occipital_Sup_R",
                       "Occipital_Mid_L", "Occipital_Mid_R",
                       "Occipital_Inf_L", "Occipital_Inf_R"],
    "Precuneus": ["Precuneus_L", "Precuneus_R"],
    "Postcentral Gyrus": ["Postcentral_L", "Postcentral_R"],
    "Middle Temporal Gyrus": ["Temporal_Mid_L", "Temporal_Mid_R"],
    "Planum Temporale": ["Heschl_L", "Heschl_R",
                         "Temporal_Sup_L", "Temporal_Sup_R"],
    "Medial temporal structures": ["Hippocampus_L", "Hippocampus_R",
                                   "ParaHippocampal_L", "ParaHippocampal_R",
                                   "Amygdala_L", "Amygdala_R"],
    "fronto-parietal network": ["Frontal_Sup_L", "Frontal_Sup_R",
                                "Frontal_Mid_L", "Frontal_Mid_R",
                                "Parietal_Sup_L", "Parietal_Sup_R",
                                "Parietal_Inf_L", "Parietal_Inf_R"],
    "FrontoParietal network (Seitzman)": ["Frontal_Sup_L", "Frontal_Sup_R",
                                          "Frontal_Mid_L", "Frontal_Mid_R",
                                          "Parietal_Sup_L", "Parietal_Sup_R",
                                          "Parietal_Inf_L", "Parietal_Inf_R"],
    "visual perceptual areas of the ventral temporal pathway": [
        "Fusiform_L", "Fusiform_R", "Temporal_Inf_L", "Temporal_Inf_R",
        "Lingual_L", "Lingual_R"],
    "Cingulate Cortex": ["Cingulum_Ant_L", "Cingulum_Ant_R",
                         "Cingulum_Mid_L", "Cingulum_Mid_R",
                         "Cingulum_Post_L", "Cingulum_Post_R"],
    "Epithalamus": ["Thalamus_L", "Thalamus_R"],
    "Posterior Parietal Cortex": ["Parietal_Sup_L", "Parietal_Sup_R",
                                  "Parietal_Inf_L", "Parietal_Inf_R"],
    "Inferior Occipital Gyrus": ["Occipital_Inf_L", "Occipital_Inf_R"],
    "Olfactory Bulb": ["Olfactory_L", "Olfactory_R"],
    "Posterior Cingulate": ["Cingulum_Post_L", "Cingulum_Post_R"],
    "Mid Cingulate": ["Cingulum_Mid_L", "Cingulum_Mid_R"],
    "Pallidum": ["Pallidum_L", "Pallidum_R"],
    "Right Pallidum": ["Pallidum_R"],
    "Left Pallidum": ["Pallidum_L"],
    "Putamen": ["Putamen_L", "Putamen_R"],
    "Thalamus": ["Thalamus_L", "Thalamus_R"],
    "Amygdala": ["Amygdala_L", "Amygdala_R"],
    "Hippocampus": ["Hippocampus_L", "Hippocampus_R"],
    "bilateral parietal and prefrontal regions": [
        "Frontal_Sup_L", "Frontal_Sup_R", "Frontal_Mid_L", "Frontal_Mid_R",
        "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
        "Parietal_Sup_L", "Parietal_Sup_R", "Parietal_Inf_L", "Parietal_Inf_R",
    ],
    "prefrontal cortex": ["Frontal_Sup_L", "Frontal_Sup_R",
                          "Frontal_Mid_L", "Frontal_Mid_R",
                          "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
                          "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R"],
    "parietal cortex": ["Parietal_Sup_L", "Parietal_Sup_R",
                        "Parietal_Inf_L", "Parietal_Inf_R"],
    "Parietal Lobe": ["Parietal_Sup_L", "Parietal_Sup_R",
                      "Parietal_Inf_L", "Parietal_Inf_R",
                      "Postcentral_L", "Postcentral_R",
                      "SupraMarginal_L", "SupraMarginal_R",
                      "Angular_L", "Angular_R", "Precuneus_L", "Precuneus_R"],
    "Superior Temporal Gyrus": ["Temporal_Sup_L", "Temporal_Sup_R"],
    "Cingulate Gyrus": ["Cingulum_Ant_L", "Cingulum_Ant_R",
                        "Cingulum_Mid_L", "Cingulum_Mid_R",
                        "Cingulum_Post_L", "Cingulum_Post_R"],
    "Insula": ["Insula_L", "Insula_R"],
    "occipital network": ["Calcarine_L", "Calcarine_R",
                          "Cuneus_L", "Cuneus_R", "Lingual_L", "Lingual_R",
                          "Occipital_Sup_L", "Occipital_Sup_R",
                          "Occipital_Mid_L", "Occipital_Mid_R",
                          "Occipital_Inf_L", "Occipital_Inf_R"],
}


def _name_to_idx(name: str) -> Optional[int]:
    try:
        return AAL_116_NAMES.index(name)
    except ValueError:
        return None


def resolve_region(region_str: str) -> list[int]:
    """Resolve a region string to AAL-116 indices.

    For connectivity regions like 'A - B', resolves both endpoints.
    """
    indices = []
    if " - " in region_str:
        parts = region_str.split(" - ")
        for part in parts:
            part = part.strip()
            indices.extend(_resolve_single_region(part))
    else:
        indices.extend(_resolve_single_region(region_str))
    return sorted(set(indices))


def _resolve_single_region(region: str) -> list[int]:
    """Resolve a single region name to AAL indices."""
    if region in REGION_TO_AAL:
        aal_names = REGION_TO_AAL[region]
        return [i for name in aal_names for i in [_name_to_idx(name)] if i is not None]

    # Fuzzy match: check if region is a substring of any AAL name
    matches = []
    region_lower = region.lower().replace(" ", "_")
    for idx, name in enumerate(AAL_116_NAMES):
        if region_lower in name.lower():
            matches.append(idx)
    return matches


def build_roi_mask(hypothesis: dict, n_roi: int = 116) -> list[bool]:
    """Build a boolean ROI mask from a hypothesis's input_region.

    For connectivity hypotheses, includes both endpoint regions.
    Minimum mask size is 10 ROIs (pads with neighboring regions if needed).
    """
    region = hypothesis.get("metadata", {}).get("input_region", "")
    indices = resolve_region(region)

    if not indices:
        return [True] * n_roi

    # Ensure minimum mask size of 10 for meaningful training
    if len(indices) < 10:
        expanded = set(indices)
        for radius in range(1, 10):
            for idx in list(indices):
                for offset in [-radius, radius]:
                    neighbor = idx + offset
                    if 0 <= neighbor < n_roi:
                        expanded.add(neighbor)
            if len(expanded) >= 10:
                break
        indices = sorted(expanded)

    mask = [False] * n_roi
    for i in indices:
        if 0 <= i < n_roi:
            mask[i] = True
    return mask


def build_edge_boost_matrix(hypothesis: dict, n_roi: int = 116) -> np.ndarray:
    """Build an edge boost matrix for KG-guided edge weighting.

    Returns an [n_roi, n_roi] matrix where entries between hypothesis-relevant
    ROIs are boosted by (1 + confidence_score).
    """
    boost = np.ones((n_roi, n_roi), dtype=np.float32)
    region = hypothesis.get("metadata", {}).get("input_region", "")
    confidence = hypothesis.get("confidence_score", 0.3)
    boost_factor = 1.0 + confidence

    indices = resolve_region(region)
    if not indices:
        return boost

    # Boost edges between all relevant ROIs
    for i in indices:
        for j in indices:
            if i != j and 0 <= i < n_roi and 0 <= j < n_roi:
                boost[i, j] = boost_factor

    return boost


if __name__ == "__main__":
    import json

    hyp_path = ROOT / "neurooracle" / "data" / "quick" / "hypotheses_imaging_hcp.json"
    with open(hyp_path) as f:
        data = json.load(f)

    print(f"AAL-116 ROI count: {len(AAL_116_NAMES)}")
    print()
    for h in data["hypotheses"][:5]:
        region = h.get("metadata", {}).get("input_region", "")
        indices = resolve_region(region)
        mask = build_roi_mask(h)
        print(f"  {h['id']} | region: {region}")
        print(f"    -> indices: {indices}")
        print(f"    -> mask size: {sum(mask)}")
        print(f"    -> ROIs: {[AAL_116_NAMES[i] for i in indices[:6]]}")
        print()
