"""Data adapter for LG-GNN — reuses the BrainGNN PyG Data format.

LG-GNN's Local_GNN backbone takes the same per-subject brain graph as
BrainGNN (PyG Data with x, edge_index, edge_attr, batch). This module simply
re-exports the BrainGNN dataset/loader so the training scripts can stay
symmetric with the other PyG-based models.
"""
from __future__ import annotations

from models.braingnn.scripts.data_adapter import (
    NeuroClawFCDataset,
    build_labels_from_csv,
)

__all__ = ["NeuroClawFCDataset", "build_labels_from_csv"]
