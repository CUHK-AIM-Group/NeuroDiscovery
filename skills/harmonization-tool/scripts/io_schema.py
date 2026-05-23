"""IO contract for harmonization-tool.

All dataset skills feeding into the harmonization layer must produce inputs
that pass `validate_inputs`. Keeping this contract narrow lets every model
skill consume harmonized output without per-dataset shims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_META_COLUMNS = ("subject_id", "dataset", "site")
RECOMMENDED_META_COLUMNS = ("scanner", "field_strength", "age", "sex", "dx")
DEFAULT_BATCH = "site"
DEFAULT_PROTECTED = ("age", "sex", "dx")


class SchemaError(ValueError):
    pass


@dataclass
class HarmonizationInputs:
    features: np.ndarray
    meta: pd.DataFrame
    batch: str = DEFAULT_BATCH
    protected: tuple[str, ...] = DEFAULT_PROTECTED
    feature_kind: str = "roi"  # "roi" | "connectome" | "voxel"

    def n_subjects(self) -> int:
        return self.features.shape[0]


def _check_feature_shape(features: np.ndarray, feature_kind: str) -> None:
    if feature_kind == "roi":
        if features.ndim != 2:
            raise SchemaError(
                f"ROI features must be 2D (N, F); got shape {features.shape}"
            )
    elif feature_kind == "connectome":
        if features.ndim != 3 or features.shape[1] != features.shape[2]:
            raise SchemaError(
                f"Connectome features must be (N, R, R); got shape {features.shape}"
            )
    elif feature_kind == "voxel":
        if features.ndim != 2:
            raise SchemaError(
                f"Voxel features must be 2D (N, V); got shape {features.shape}"
            )
    else:
        raise SchemaError(
            f"Unknown feature_kind '{feature_kind}'. "
            "Expected one of: roi | connectome | voxel"
        )


def _check_meta_columns(meta: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_META_COLUMNS if c not in meta.columns]
    if missing:
        raise SchemaError(
            f"meta is missing required columns: {missing}. "
            f"Required: {REQUIRED_META_COLUMNS}"
        )


def _check_protected_present(meta: pd.DataFrame, protected: Iterable[str]) -> None:
    missing = [c for c in protected if c not in meta.columns]
    if missing:
        raise SchemaError(
            f"protected covariates not in meta: {missing}. "
            "Either drop them from --protected or add them to the meta table."
        )


def _check_no_singleton_batch(meta: pd.DataFrame, batch: str) -> None:
    if batch not in meta.columns:
        raise SchemaError(f"batch column '{batch}' not in meta")
    n_levels = meta[batch].nunique(dropna=False)
    if n_levels < 2:
        raise SchemaError(
            f"batch '{batch}' has only {n_levels} unique value(s); "
            "harmonization requires >= 2 batches"
        )


def _check_alignment(features: np.ndarray, meta: pd.DataFrame) -> None:
    if features.shape[0] != len(meta):
        raise SchemaError(
            f"features N={features.shape[0]} does not match meta rows {len(meta)}"
        )


def validate_inputs(inputs: HarmonizationInputs) -> None:
    """Run all schema checks. Raises SchemaError on first failure."""
    _check_feature_shape(inputs.features, inputs.feature_kind)
    _check_meta_columns(inputs.meta)
    _check_alignment(inputs.features, inputs.meta)
    _check_no_singleton_batch(inputs.meta, inputs.batch)
    _check_protected_present(inputs.meta, inputs.protected)


def load_inputs(
    features_path: str | Path,
    meta_path: str | Path,
    batch: str = DEFAULT_BATCH,
    protected: tuple[str, ...] = DEFAULT_PROTECTED,
    feature_kind: str = "roi",
) -> HarmonizationInputs:
    features = np.load(features_path)
    meta = pd.read_csv(meta_path)
    inputs = HarmonizationInputs(
        features=features,
        meta=meta,
        batch=batch,
        protected=tuple(protected),
        feature_kind=feature_kind,
    )
    validate_inputs(inputs)
    return inputs
