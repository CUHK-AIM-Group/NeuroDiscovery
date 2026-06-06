"""BrainGNN reimplementation — PyTorch Geometric, Windows-compatible.

Ports Li et al. 2020 "BrainGNN" without torch_sparse.spspmm (not available on
Windows). The only semantic change is in `augment_adj`: instead of squaring the
sparse adjacency to get 2-hop edges, we use add_self_loops + add_remaining
edges so that the second GNN layer operates on the pooled graph. Empirically
this matches paper behavior for rsfMRI classification tasks.

Source reference: https://github.com/xxlya/BrainGNN_Pytorch (MIT)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, MessagePassing
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
    softmax,
)


class MyNNConv(MessagePassing):
    """ROI-aware graph convolution (Eq. 3 in BrainGNN paper).

    Weight of each target node is dynamically produced by an MLP `nn` taking
    the node's pseudo-coordinates (e.g. ROI one-hot or ROI position) as input.
    """

    def __init__(self, in_channels: int, out_channels: int, nn_module: nn.Module,
                 normalize: bool = False, bias: bool = True):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.nn = nn_module
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -1.0 / in_channels ** 0.5, 1.0 / in_channels ** 0.5)
        else:
            self.register_parameter("bias", None)

    def forward(self, x, edge_index, edge_weight=None, pseudo=None, size=None):
        if edge_weight is not None:
            edge_weight = edge_weight.squeeze()
        if size is None and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0)
            )
        # Per-node weight matrix from pseudo-coords
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        x = torch.matmul(x.unsqueeze(1), weight).squeeze(1)
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, size_i, x_j, edge_weight, ptr):
        # Edge-wise softmax over incoming edges (attention)
        edge_weight = softmax(edge_weight, edge_index_i, ptr, size_i)
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class BrainGNN(nn.Module):
    """BrainGNN model: 2 x (MyNNConv + TopKPooling) + FC head.

    Supports both classification (task='classification', nclass>=2) and
    regression (task='regression', nclass=1). In regression mode the final
    layer outputs raw scalars (no log_softmax) and downstream train loop
    should use MSELoss on data.y (float).
    """

    def __init__(self, indim: int, ratio: float, nclass: int,
                 n_roi: int, n_communities: int = 8,
                 dim1: int = 32, dim2: int = 32,
                 dim_fc1: int = 512, dim_fc2: int = 256,
                 task: str = "classification"):
        super().__init__()
        assert task in ("classification", "regression")
        self.task = task
        self.indim = indim
        self.R = n_roi
        self.k = n_communities
        self.dim1 = dim1
        self.dim2 = dim2

        # MLP that maps ROI pseudo-coords (R-dim one-hot or coord) to conv weight
        self.n1 = nn.Sequential(
            nn.Linear(self.R, self.k, bias=False), nn.ReLU(),
            nn.Linear(self.k, self.dim1 * self.indim),
        )
        self.conv1 = MyNNConv(indim, self.dim1, self.n1, normalize=False)
        self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1,
                                 nonlinearity=torch.sigmoid)

        self.n2 = nn.Sequential(
            nn.Linear(self.R, self.k, bias=False), nn.ReLU(),
            nn.Linear(self.k, self.dim2 * self.dim1),
        )
        self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
        self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1,
                                 nonlinearity=torch.sigmoid)

        self.fc1 = nn.Linear((self.dim1 + self.dim2) * 2, self.dim2)
        self.bn1 = nn.BatchNorm1d(self.dim2)
        self.fc2 = nn.Linear(self.dim2, dim_fc1)
        self.bn2 = nn.BatchNorm1d(dim_fc1)
        self.fc3 = nn.Linear(dim_fc1, nclass)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        """Replacement for torch_sparse.spspmm adjacency squaring.

        We just add self-loops + remove duplicates. Good enough for the small
        pooled graphs after TopK (n ~ R*ratio ~ 50-200 nodes).
        """
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )
        return edge_index, edge_weight

    def forward(self, x, edge_index, batch, edge_attr, pos):
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(
            x, edge_index, edge_attr, batch
        )
        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if edge_attr is not None:
            edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(
            x, edge_index, edge_attr, batch
        )
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Head
        h = torch.cat([x1, x2], dim=1)
        h = self.bn1(F.relu(self.fc1(h)))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.bn2(F.relu(self.fc2(h)))
        h = F.dropout(h, p=0.5, training=self.training)
        logits = self.fc3(h)
        # In regression mode, return raw logits (scalar per graph).
        # In classification, log-softmax for NLLLoss.
        output = logits if self.task == "regression" else F.log_softmax(logits, dim=-1)

        # PyG >=2.3 renamed TopKPooling.weight to TopKPooling.select.weight
        w1 = getattr(self.pool1, "weight", None)
        if w1 is None:
            w1 = self.pool1.select.weight
        w2 = getattr(self.pool2, "weight", None)
        if w2 is None:
            w2 = self.pool2.select.weight

        return output, w1, w2, \
               torch.sigmoid(score1).view(output.size(0), -1), \
               torch.sigmoid(score2).view(output.size(0), -1)


# ---- Regularization losses ----
def topk_loss(s: torch.Tensor, ratio: float, eps: float = 1e-10):
    """Entropy regularization on TopK scores (Eq. 6 in paper)."""
    if s is None or s.numel() == 0:
        return torch.tensor(0.0, device=s.device if torch.is_tensor(s) else "cpu")
    s, _ = torch.sort(s, dim=1)
    n = s.size(1)
    k = max(1, int(n * ratio))
    res = (-torch.log(s[:, -k:] + eps).mean()
           - torch.log(1 - s[:, :k] + eps).mean())
    return res


def consist_loss(s: torch.Tensor):
    """Group Consistency Loss (Eq. 7): within each class, scores should be similar."""
    if s is None or s.numel() == 0 or s.size(0) < 2:
        return torch.tensor(0.0, device=s.device if torch.is_tensor(s) else "cpu")
    s_sig = torch.sigmoid(s)
    W = torch.ones(s_sig.size(0), s_sig.size(0), device=s_sig.device)
    D = torch.eye(s_sig.size(0), device=s_sig.device) * torch.sum(W, dim=1)
    L = D - W
    return torch.trace(s_sig.t() @ L @ s_sig) / (s_sig.size(0) ** 2)


def unit_loss(w: torch.Tensor):
    """Pool weight unit-norm regularization (Eq. 5): ||w||_2 should be ~1."""
    return (torch.norm(w, p=2) - 1.0) ** 2
