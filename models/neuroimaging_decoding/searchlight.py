"""Nilearn SearchLight with explicit CV groups and output map."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_searchlight(
    images: list[str | Path],
    labels: np.ndarray,
    mask_img: str | Path,
    output_map: str | Path,
    process_mask_img: str | Path | None = None,
    radius_mm: float = 5.0,
    cv: int | object = 5,
    scoring: str = "balanced_accuracy",
    n_jobs: int = 1,
):
    try:
        import nibabel as nib
        from nilearn.decoding import SearchLight
    except ImportError as exc:
        raise RuntimeError("nilearn and nibabel are required for SearchLight") from exc
    image_objects = [nib.load(str(path)) for path in images]
    mask = nib.load(str(mask_img))
    process_mask = nib.load(str(process_mask_img)) if process_mask_img else mask
    searchlight = SearchLight(
        mask_img=mask,
        process_mask_img=process_mask,
        radius=radius_mm,
        estimator="svc",
        n_jobs=n_jobs,
        scoring=scoring,
        cv=cv,
        verbose=0,
    )
    searchlight.fit(image_objects, np.asarray(labels))
    score_img = nib.Nifti1Image(
        np.asarray(searchlight.scores_, dtype=np.float32),
        affine=process_mask.affine,
        header=process_mask.header,
    )
    output = Path(output_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(score_img, output)
    return searchlight
