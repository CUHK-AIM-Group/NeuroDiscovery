"""BrainNetCNN (Kawahara et al., 2017, NeuroImage).

Specialized CNN for brain connectivity matrices with three novel layers:
  - E2E (Edge-to-Edge): cross-shaped convolution capturing row+column patterns
  - E2N (Edge-to-Node): reduces edge features to node-level (1×N conv)
  - N2G (Node-to-Graph): reduces node features to graph-level (N×1 conv)

Input: [B, 1, N, N] — connectivity matrix (Pearson correlation)
Output: [B, nclass] — logits

Reference: https://github.com/nicofarr/brainnetcnn
Paper: Kawahara et al. "BrainNetCNN: Convolutional neural networks for brain
       networks; towards predicting neurodevelopment." NeuroImage 146, 2017.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EBlock(nn.Module):
    """Edge-to-Edge layer: cross-shaped convolution on connectivity matrix.

    For each output channel, applies a row filter (1×N) and a column filter (N×1)
    and sums them. This captures patterns along rows and columns of the FC matrix
    (i.e., all connections of a given ROI).
    """

    def __init__(self, in_channels: int, out_channels: int, n_roi: int, bias: bool = True):
        super().__init__()
        self.row_conv = nn.Conv2d(in_channels, out_channels, (1, n_roi), bias=False)
        self.col_conv = nn.Conv2d(in_channels, out_channels, (n_roi, 1), bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, N, N]
        row_out = self.row_conv(x)   # [B, C_out, N, 1]
        col_out = self.col_conv(x)   # [B, C_out, 1, N]
        out = row_out + col_out      # broadcast -> [B, C_out, N, N]
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class E2NBlock(nn.Module):
    """Edge-to-Node layer: aggregates edge features into node representations.

    Applies a 1×N convolution to reduce each row to a scalar per channel,
    producing [B, C_out, N, 1] node features.
    """

    def __init__(self, in_channels: int, out_channels: int, n_roi: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, n_roi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B, C_out, N, 1]


class N2GBlock(nn.Module):
    """Node-to-Graph layer: aggregates node features into graph-level representation.

    Applies an N×1 convolution to reduce N nodes to a single scalar per channel,
    producing [B, C_out, 1, 1].
    """

    def __init__(self, in_channels: int, out_channels: int, n_roi: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (n_roi, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B, C_out, 1, 1]


class BrainNetCNN(nn.Module):
    """BrainNetCNN model.

    Architecture:
        E2E(1, e2e_channels) -> LeakyReLU
        E2E(e2e_channels, e2e_channels) -> LeakyReLU
        E2N(e2e_channels, e2n_channels) -> LeakyReLU
        N2G(e2n_channels, n2g_channels) -> LeakyReLU
        Dropout -> FC -> LeakyReLU -> Dropout -> FC(nclass)

    Args:
        n_roi: number of ROIs (nodes), determines conv kernel sizes
        nclass: output classes (2 for binary classification, 1 for regression)
        e2e_channels: channels in E2E layers (default 32)
        e2n_channels: channels in E2N layer (default 64)
        n2g_channels: channels in N2G layer (default 256)
        dropout: dropout rate (default 0.5)
        task: "classification" or "regression"
    """

    def __init__(self, n_roi: int, nclass: int = 2,
                 e2e_channels: int = 32, e2n_channels: int = 64,
                 n2g_channels: int = 256, dropout: float = 0.5,
                 task: str = "classification"):
        super().__init__()
        assert task in ("classification", "regression")
        self.task = task
        self.n_roi = n_roi

        self.e2e1 = E2EBlock(1, e2e_channels, n_roi)
        self.e2e2 = E2EBlock(e2e_channels, e2e_channels, n_roi)
        self.e2n = E2NBlock(e2e_channels, e2n_channels, n_roi)
        self.n2g = N2GBlock(e2n_channels, n2g_channels, n_roi)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n2g_channels, 128),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(128, 30),
            nn.LeakyReLU(0.33),
            nn.Linear(30, nclass),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, N] connectivity matrix (or [B, 1, N, N])
        Returns:
            logits: [B, nclass]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, N, N]

        x = F.leaky_relu(self.e2e1(x), 0.33)
        x = F.leaky_relu(self.e2e2(x), 0.33)
        x = F.leaky_relu(self.e2n(x), 0.33)   # [B, e2n, N, 1]
        x = F.leaky_relu(self.n2g(x), 0.33)   # [B, n2g, 1, 1]

        x = x.view(x.size(0), -1)  # [B, n2g]
        logits = self.fc(x)         # [B, nclass]
        return logits
