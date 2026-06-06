"""Data adapter for IBGNN — reuses the BrainGNN PyG Data format."""
from __future__ import annotations

from models.braingnn.scripts.data_adapter import (
    NeuroClawFCDataset,
    build_labels_from_csv,
)

__all__ = ["NeuroClawFCDataset", "build_labels_from_csv"]
