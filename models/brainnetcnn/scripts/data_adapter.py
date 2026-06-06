"""Data adapter for BrainNetCNN.

BrainNetCNN takes a dense FC matrix [N, N] per subject — same as BNT.
This module re-exports BNT's data adapter since the input format is identical.
"""
from models.bnt.scripts.data_adapter import (
    BNTDataset as BrainNetCNNDataset,
    bnt_collate as brainnetcnn_collate,
    build_labels_from_csv,
)

__all__ = ["BrainNetCNNDataset", "brainnetcnn_collate", "build_labels_from_csv"]
