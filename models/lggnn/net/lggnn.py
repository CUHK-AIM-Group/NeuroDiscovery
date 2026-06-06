"""LG-GNN (Local-to-Global GNN) — single-subject adaptation.

Source: cnuzh/LG-GNN, "Local to Global Hierarchical Graph Neural Network for
Brain Disorder Diagnosis" (Zhang et al. 2022, MICCAI).

The original combines a per-subject Local_GNN (GCN + SABP mutual-info pooling)
with a Global_GNN over a population graph built from non-imaging phenotypic
data. Since NeuroClaw works at the per-subject brain-graph level only, we keep
the Local_GNN backbone (the novel contribution: SABP pooling with MI loss) and
replace the population Global_GNN with a graph-level MLP head.

Paper hyperparameters (from opt.py):
  lr=0.01, wd=5e-5
  hgc=16 (GCN hidden), lg=4
  dropout=0.2, edropout=0.3
  num_iter=400 (we use 50 by default)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.pool.select.topk import topk


def _filter_adj(edge_index, edge_attr, perm, num_nodes):
    """Inline replacement for filter_adj (removed in PyG 2.7).

    Keep only edges where both endpoints survive the topk selection, and
    remap node ids to the compact range [0, len(perm)).
    """
    mask = perm.new_full((num_nodes,), -1)
    mask[perm] = torch.arange(perm.size(0), device=perm.device)
    row, col = edge_index[0], edge_index[1]
    keep = (mask[row] >= 0) & (mask[col] >= 0)
    edge_index = torch.stack([mask[row[keep]], mask[col[keep]]], dim=0)
    if edge_attr is not None:
        edge_attr = edge_attr[keep]
    return edge_index, edge_attr


class SABP(nn.Module):
    """Self-Attention Brain Pooling with mutual-information regularization.

    Adapted from layer.py of cnuzh/LG-GNN. Selects top-k ROIs by score,
    multiplies kept features by tanh(score), and returns an MI estimate
    used as auxiliary loss to push embeddings away from random shuffles.
    """

    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = GCNConv(in_channels, 1)
        self.gcn = GCNConv(in_channels, in_channels)
        self.fc = nn.Linear(in_channels * 2, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score_neg = x[torch.randperm(x.size(0))]
        embed = self.gcn(x, edge_index, edge_attr)
        joint = self.fc(torch.cat([embed, x], dim=-1))
        margin = self.fc(torch.cat([embed, score_neg], dim=-1))
        joint = F.normalize(joint, dim=1)
        margin = F.normalize(margin, dim=1)
        mi_est = joint.mean() - torch.log(torch.exp(margin).mean().clamp(min=1e-8))

        score = self.score_layer(x, edge_index, edge_attr).squeeze(-1)
        perm = topk(score, self.ratio, batch)
        x = x[perm] * torch.tanh(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = _filter_adj(edge_index, edge_attr, perm,
                                             num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm, mi_est


class LocalGNN(nn.Module):
    """Per-subject brain-graph encoder: GCN -> GCN -> SABP -> GCN -> mean pool."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 20,
                 ratio: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.pool = SABP(embed_dim, ratio=ratio)
        self.conv3 = GCNConv(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h1 = F.relu(self.conv1(x, edge_index, edge_attr))
        h2 = F.relu(self.conv2(h1, edge_index, edge_attr))
        h2_p, ei_p, ea_p, batch_p, perm, mi = self.pool(h2, edge_index, edge_attr, batch)
        h3 = F.relu(self.conv3(h2_p, ei_p, ea_p))
        node_emb = h2_p + h3
        graph_emb = global_mean_pool(node_emb, batch_p)
        return graph_emb, perm, mi


class LGGNN(nn.Module):
    """LG-GNN single-subject classifier/regressor.

    Args:
        n_roi: number of ROIs (input feature dim, since x = FC rows).
        nclass: number of classes (or 1 for regression).
        hidden_dim: GCN hidden width (paper: 64).
        embed_dim: post-SABP embedding width (paper: 20).
        ratio: SABP keep ratio (paper: 0.9 with hgc=16; we default 0.5 to match
               BrainGNN's pooling rate at this scale).
        dropout: head dropout.
        task: "classification" or "regression".
    """

    def __init__(self, n_roi: int, nclass: int = 2, hidden_dim: int = 64,
                 embed_dim: int = 20, ratio: float = 0.5, dropout: float = 0.3,
                 task: str = "classification"):
        super().__init__()
        self.task = task
        self.local = LocalGNN(in_dim=n_roi, hidden_dim=hidden_dim,
                              embed_dim=embed_dim, ratio=ratio)
        out_dim = nclass if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
        graph_emb, _, mi = self.local(x, edge_index, edge_attr, batch)
        out = self.head(graph_emb)
        return out, mi
