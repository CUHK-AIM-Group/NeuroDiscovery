"""IBGNN — Interpretable Brain Graph Neural Network.

Source: HennyJie/IBGNN, "Interpretable Graph Neural Networks for Connectome-
Based Brain Disorder Analysis" (Cui et al. 2022, MICCAI).

Architecture: stack of MPConv layers, each combining (x_i, x_j, edge_attr)
through a learnable MLP message function on top of GCN-style propagation. The
model is paired with a post-hoc edge-mask explainer (we only port the encoder
here; explainer code stays in skills/ if needed).

We strip the edge_flag mechanism (used by the explainer) since for normal
forward passes it is just an all-ones tensor. Edge weights are kept positive
by abs(), matching the original.

Paper hyperparameters (typical, from Cui et al. main_explainer.py defaults):
  hidden_dim=128, n_GNN_layers=2, n_MLP_layers=1
  pooling='concat', lr=1e-4, weight_decay=1e-5
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter


def gcn_norm(edge_index, edge_attr, num_nodes):
    """Symmetric normalization with edge weights and self-loops."""
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr,
                                            fill_value=1.0, num_nodes=num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_attr, col, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * edge_attr * deg_inv_sqrt[col]
    return edge_index, edge_weight


class MPConv(MessagePassing):
    """IBGNN message-passing conv: msg(i,j) = lin([x_i, x_j, edge_attr])."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.weight = Parameter(torch.empty(in_channels, out_channels))
        self.lin = nn.Linear(out_channels * 2 + 1, out_channels)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_weight = gcn_norm(edge_index, edge_attr, x.size(0))
        x = x @ self.weight
        out = self.propagate(edge_index, x=x, edge_attr=edge_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr.view(-1, 1)], dim=1)
        return self.lin(msg)


class IBGConv(nn.Module):
    """Stack of MPConv layers with global pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, pooling: str = 'mean'):
        super().__init__()
        self.pooling = pooling
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.convs.append(MPConv(in_d, hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.abs()
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        if self.pooling == 'sum':
            g = global_add_pool(x, batch)
        else:
            g = global_mean_pool(x, batch)
        return g


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 n_classes: int):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.shortcut = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = self.net(x) + self.shortcut(x)
        return self.classifier(h)


class IBGNN(nn.Module):
    """IBGNN classifier/regressor.

    Args:
        n_roi: input feature dim (FC rows).
        nclass: number of classes (or 1 for regression).
        hidden_dim: GNN/MLP hidden width (paper: 128).
        n_gnn_layers: stacked MPConv layers (paper: 2-3).
        n_mlp_layers: head MLP depth (paper: 1).
        pooling: 'mean' or 'sum'.
        task: "classification" or "regression".
    """

    def __init__(self, n_roi: int, nclass: int = 2, hidden_dim: int = 128,
                 n_gnn_layers: int = 2, n_mlp_layers: int = 1,
                 pooling: str = 'mean', task: str = "classification"):
        super().__init__()
        self.task = task
        self.gnn = IBGConv(input_dim=n_roi, hidden_dim=hidden_dim,
                           n_layers=n_gnn_layers, pooling=pooling)
        out_dim = nclass if task == "classification" else 1
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
                       n_layers=n_mlp_layers, n_classes=out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        g = self.gnn(x, edge_index, edge_attr, batch)
        return self.mlp(g)
