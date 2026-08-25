"""Compact residual 3D CNN for structural or functional volumes."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class VoxelCNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 2,
        base_channels: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4]
        stages = []
        previous = in_channels
        for width in widths:
            stages.extend(
                [
                    nn.Conv3d(previous, width, 3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(width, affine=True),
                    nn.GELU(),
                    ResidualBlock3D(width, dropout),
                ]
            )
            previous = width
        self.encoder = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], output_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))
