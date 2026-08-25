"""LSTM, GRU, temporal CNN, and Transformer encoders."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * denominator)
        encoding[:, 1::2] = torch.cos(position * denominator[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]


class TemporalPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: str = "gru",
        hidden_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
    ):
        super().__init__()
        self.model_name = model.lower().replace("-", "_")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        if self.model_name in {"lstm", "gru"}:
            recurrent = nn.LSTM if self.model_name == "lstm" else nn.GRU
            self.encoder = recurrent(
                hidden_dim,
                hidden_dim,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.model_name in {"tcn", "temporal_cnn", "cnn1d"}:
            blocks = []
            for index in range(layers):
                dilation = 2**index
                blocks.extend(
                    [
                        nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
            self.encoder = nn.Sequential(*blocks)
        elif self.model_name in {"transformer", "temporal_transformer"}:
            compatible_heads = max(
                head for head in range(1, nhead + 1) if hidden_dim % head == 0
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=compatible_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.position = PositionalEncoding(hidden_dim)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        else:
            raise ValueError(f"Unknown temporal model: {model}")
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, lengths=None):
        x = self.input_projection(x)
        if self.model_name in {"lstm", "gru"}:
            encoded, _ = self.encoder(x)
            if lengths is None:
                pooled = encoded[:, -1]
            else:
                index = (lengths - 1).clamp_min(0)
                pooled = encoded[torch.arange(len(encoded), device=x.device), index]
        elif self.model_name in {"tcn", "temporal_cnn", "cnn1d"}:
            pooled = self.encoder(x.transpose(1, 2)).mean(dim=-1)
        else:
            padding_mask = None
            if lengths is not None:
                positions = torch.arange(x.size(1), device=x.device)[None, :]
                padding_mask = positions >= lengths[:, None]
            encoded = self.encoder(
                self.position(x), src_key_padding_mask=padding_mask
            )
            if padding_mask is None:
                pooled = encoded.mean(dim=1)
            else:
                valid = (~padding_mask).unsqueeze(-1)
                pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)
