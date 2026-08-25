"""Shared tabular loading and leakage-safe split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


@dataclass
class TabularData:
    subject_ids: np.ndarray
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    groups: np.ndarray | None = None


def load_tabular_csv(
    path: str | Path,
    target: str,
    subject_col: str = "subject_id",
    group_col: str | None = None,
    feature_cols: Iterable[str] | None = None,
) -> TabularData:
    frame = pd.read_csv(path)
    required = {target, subject_col}
    if group_col:
        required.add(group_col)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if feature_cols is None:
        excluded = required
        feature_cols = [
            c for c in frame.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
        ]
    feature_names = list(feature_cols)
    if not feature_names:
        raise ValueError("No numeric feature columns were selected")
    clean = frame.dropna(subset=[target, *feature_names]).reset_index(drop=True)
    return TabularData(
        subject_ids=clean[subject_col].astype(str).to_numpy(),
        X=clean[feature_names].to_numpy(dtype=np.float32),
        y=clean[target].to_numpy(),
        feature_names=feature_names,
        groups=clean[group_col].to_numpy() if group_col else None,
    )


def make_splits(
    y: np.ndarray,
    task: str,
    n_splits: int = 5,
    seed: int = 123,
    groups: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    if len(y) < 2:
        raise ValueError("At least two samples are required")
    n_splits = max(2, min(int(n_splits), len(y)))
    indices = np.arange(len(y))
    if groups is not None:
        unique_groups = np.unique(groups)
        n_splits = min(n_splits, len(unique_groups))
        if n_splits < 2:
            raise ValueError("Group-aware CV requires at least two groups")
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(indices, y, groups))
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        n_splits = min(n_splits, int(counts.min()))
        if n_splits < 2:
            raise ValueError("Each class needs at least two samples for CV")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices, y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(indices))
