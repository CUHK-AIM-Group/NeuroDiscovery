"""Atlas registry for ABIDE I/II ROI extraction.

Defines the 17-atlas set used across the pipeline:
  - 7 ABIDE PCP official atlases (cached on Z:\\Public Dataset\\_atlas_cache)
  - 14 local atlases in NeuroClaw/data/atlas/ (3 of which overlap PCP)
We dedupe to 17 unique atlases and tag each with masker kind:
  'labels'  -> hard-parcellation NiftiLabelsMasker
  'maps'    -> probabilistic NiftiMapsMasker (msdl)
  'spheres' -> coordinate-based NiftiSpheresMasker (power_264)

Each atlas exposes:
  name, kind, n_rois, image_path (or coords), labels_csv (optional).

Resampling to a target affine/shape happens lazily once per (atlas, dataset).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


REPO = Path(__file__).resolve().parents[4]
LOCAL_ATLAS_DIR = REPO / "data" / "atlas"


@dataclass
class AtlasSpec:
    name: str
    kind: Literal["labels", "maps", "spheres"]
    n_rois: int
    image_path: Optional[Path] = None
    labels_csv: Optional[Path] = None
    coords_npy: Optional[Path] = None
    radius_mm: float = 5.0  # only for spheres
    note: str = ""

    def exists(self) -> bool:
        if self.kind in ("labels", "maps"):
            return self.image_path is not None and self.image_path.exists()
        if self.kind == "spheres":
            return self.coords_npy is not None and self.coords_npy.exists()
        return False


def _local(name: str) -> Path:
    return LOCAL_ATLAS_DIR / name / "atlas.nii.gz"


def _local_labels(name: str) -> Path:
    return LOCAL_ATLAS_DIR / name / "labels.csv"


# 17-atlas spec.
# Conventions:
#   - When PCP and local provide the same atlas (aal, cc200, ho), prefer PCP
#     (it is the exact NIfTI used to derive ABIDE I .1D).
#   - cc400 / dosenbach160 / ez / tt are PCP-only.
#   - power_264 is coords-only -> NiftiSpheresMasker; we pick 5 mm sphere radius
#     (the value used in Power 2011).
ATLASES: list[AtlasSpec] = [
    # === ABIDE PCP official 7 ===
    AtlasSpec("aal_116", "labels", 116,
              image_path=_local("aal_116"),
              labels_csv=_local_labels("aal_116"),
              note="ABIDE PCP aal_roi_atlas (official rois_aal source)"),
    AtlasSpec("cc200", "labels", 200,
              image_path=_local("cc200"),
              labels_csv=_local_labels("cc200"),
              note="Craddock 200 (ABIDE PCP rois_cc200)"),
    AtlasSpec("cc400", "labels", 392,
              image_path=_local("cc400"),
              labels_csv=_local_labels("cc400"),
              note="Craddock 400 (ABIDE PCP rois_cc400)"),
    AtlasSpec("dosenbach160", "labels", 161,
              image_path=_local("dosenbach160"),
              labels_csv=_local_labels("dosenbach160"),
              note="Dosenbach 160 (ABIDE PCP rois_dosenbach160)"),
    AtlasSpec("ez", "labels", 116,
              image_path=_local("ez"),
              labels_csv=_local_labels("ez"),
              note="Eickhoff-Zilles (ABIDE PCP rois_ez)"),
    AtlasSpec("ho", "labels", 111,
              image_path=_local("ho"),
              labels_csv=_local_labels("ho"),
              note="Harvard-Oxford merged cort+sub (ABIDE PCP rois_ho)"),
    AtlasSpec("tt", "labels", 97,
              image_path=_local("tt"),
              labels_csv=_local_labels("tt"),
              note="Talairach-Tournoux (ABIDE PCP rois_tt)"),

    # === local extras (no PCP overlap) ===
    AtlasSpec("aal3_166", "labels", 166,
              image_path=_local("aal3_166"),
              labels_csv=_local_labels("aal3_166")),
    AtlasSpec("basc_122", "labels", 122,
              image_path=_local("basc_122"),
              labels_csv=_local_labels("basc_122")),
    AtlasSpec("dk_112", "labels", 112,
              image_path=_local("dk_112"),
              labels_csv=_local_labels("dk_112")),
    AtlasSpec("destrieux_148", "labels", 148,
              image_path=_local("destrieux_148"),
              labels_csv=_local_labels("destrieux_148")),
    AtlasSpec("glasser_360", "labels", 360,
              image_path=_local("glasser_360"),
              labels_csv=_local_labels("glasser_360")),
    AtlasSpec("msdl_39", "maps", 39,
              image_path=_local("msdl_39"),
              labels_csv=_local_labels("msdl_39"),
              note="probabilistic atlas, NiftiMapsMasker"),
    AtlasSpec("power_264", "spheres", 264,
              coords_npy=LOCAL_ATLAS_DIR / "power_264" / "coords.csv",
              radius_mm=5.0,
              note="Power 2011 coords (MNI mm) + 5mm sphere"),
    AtlasSpec("schaefer_100_7net", "labels", 100,
              image_path=_local("schaefer_100_7net"),
              labels_csv=_local_labels("schaefer_100_7net")),
    AtlasSpec("schaefer_200_7net", "labels", 200,
              image_path=_local("schaefer_200_7net"),
              labels_csv=_local_labels("schaefer_200_7net")),
    AtlasSpec("schaefer_400_7net", "labels", 400,
              image_path=_local("schaefer_400_7net"),
              labels_csv=_local_labels("schaefer_400_7net")),
]


def get_registry() -> list[AtlasSpec]:
    return list(ATLASES)


def by_name(name: str) -> AtlasSpec:
    for a in ATLASES:
        if a.name == name:
            return a
    raise KeyError(f"unknown atlas: {name}")


if __name__ == "__main__":
    import sys
    print(f"# atlas registry: {len(ATLASES)} atlases\n")
    print(f"{'name':<22} {'kind':<8} {'n_rois':>7}  {'exists':<7} {'path'}")
    missing = 0
    for a in ATLASES:
        ok = a.exists()
        missing += 0 if ok else 1
        target = a.image_path if a.kind != "spheres" else a.coords_npy
        print(f"{a.name:<22} {a.kind:<8} {a.n_rois:>7}  {'OK' if ok else 'MISS':<7} {target}")
    print(f"\nmissing: {missing}/{len(ATLASES)}")
    sys.exit(0 if missing == 0 else 1)
