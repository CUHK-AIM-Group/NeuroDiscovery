"""Com-BrainTF — Community-aware Transformer for fMRI Connectome.

Source: ubc-tea/Com-BrainTF, "Community-Aware Transformer for Autism Prediction
in fMRI Connectome" (Bannadabhavi et al. 2023, MICCAI).

Two-level architecture:
  1. Local transformer per community (Yeo 7-network or atlas-derived groups):
     each community gets its own learnable [CLS] token, processed independently
     by a shared local transformer.
  2. Global transformer on community [CLS] tokens + remaining node features,
     followed by DEC pooling (cluster N nodes -> output_node_num clusters).
  3. Dim reduction Linear(d_model -> 8) -> flatten -> MLP head.

The original loads `node_clus_map.pickle` for Schaefer-400 (8 communities). We
make it general: caller passes `community_ids: list[int]` of length n_roi at
construction time. Helpers in data_adapter build common community maps for AAL,
Schaefer (Yeo 7-net), HO.

Paper hyperparameters (conf/model/comtf.yaml):
  nhead=8, num_MHSA=1, hidden_size=1024
  sizes=[node_sz, 8] (8 output clusters)
  pooling=[False, True], orthogonal=True, freeze_center=True, project_assignment=True
  pos_encoding='none' (default), pos_embed_dim=360
  lr=1e-4 typical
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class InterpretableTransformerEncoder(TransformerEncoderLayer):
    """TransformerEncoderLayer that exposes self-attention weights."""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 batch_first=True):
        super().__init__(d_model=d_model, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout,
                         batch_first=batch_first)
        self.attention_weights: Optional[torch.Tensor] = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x_out, w = self.self_attn(x, x, x, attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=True, is_causal=is_causal)
        self.attention_weights = w
        return self.dropout1(x_out)

    def get_attention_weights(self):
        return self.attention_weights


class ClusterAssignment(nn.Module):
    """DEC soft cluster assignment with optional orthogonal init."""

    def __init__(self, n_clusters: int, embed_dim: int, alpha: float = 1.0,
                 orthogonal: bool = True, freeze_center: bool = True,
                 project_assignment: bool = True):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.project_assignment = project_assignment

        centers = torch.zeros(n_clusters, embed_dim)
        nn.init.xavier_uniform_(centers)
        if orthogonal:
            ortho = torch.zeros_like(centers)
            ortho[0] = centers[0]
            for i in range(1, n_clusters):
                proj = torch.zeros_like(centers[0])
                for j in range(i):
                    u, v = centers[j], centers[i]
                    proj = proj + (torch.dot(u, v) / torch.dot(u, u).clamp(min=1e-8)) * u
                centers[i] = centers[i] - proj
                ortho[i] = centers[i] / centers[i].norm(p=2).clamp(min=1e-8)
            centers = ortho
        self.cluster_centers = nn.Parameter(centers, requires_grad=not freeze_center)

    def forward(self, x):  # x: [B*N, F]
        if self.project_assignment:
            assign = (x @ self.cluster_centers.T) ** 2
            norm = self.cluster_centers.norm(p=2, dim=-1).clamp(min=1e-8)
            return F.softmax(assign / norm, dim=-1)
        d = ((x.unsqueeze(1) - self.cluster_centers) ** 2).sum(-1)
        num = (1.0 + d / self.alpha) ** (-(self.alpha + 1) / 2)
        return num / num.sum(dim=1, keepdim=True)


class DECPool(nn.Module):
    """Encoder + cluster assignment producing [B, n_clusters, F] from [B, N, F]."""

    def __init__(self, n_clusters: int, embed_dim: int, n_in_nodes: int,
                 enc_hidden: int = 32, orthogonal: bool = True,
                 freeze_center: bool = True, project_assignment: bool = True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * n_in_nodes, enc_hidden), nn.LeakyReLU(),
            nn.Linear(enc_hidden, enc_hidden), nn.LeakyReLU(),
            nn.Linear(enc_hidden, embed_dim * n_in_nodes),
        )
        self.assign = ClusterAssignment(n_clusters, embed_dim,
                                         orthogonal=orthogonal,
                                         freeze_center=freeze_center,
                                         project_assignment=project_assignment)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):  # [B, N, F]
        bz, n, f = x.shape
        flat = x.reshape(bz, -1)
        encoded = self.encoder(flat).reshape(bz * n, f)
        assignment = self.assign(encoded).reshape(bz, n, -1)  # [B, N, K]
        encoded = encoded.reshape(bz, n, f)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)  # [B, K, F]
        return node_repr, assignment

    def dec_loss(self, assignment):
        flat = assignment.reshape(-1, assignment.size(-1)).clamp(min=1e-8)
        weight = (flat ** 2) / flat.sum(0).clamp(min=1e-8)
        target = (weight.t() / weight.sum(1).clamp(min=1e-8)).t().detach()
        return self.loss_fn(flat.log(), target)


class TransPoolingEncoder(nn.Module):
    """Transformer + optional DEC pooling. Adds a [CLS] token, returns it."""

    def __init__(self, d_model: int, n_in_nodes: int, n_out_nodes: int,
                 hidden_size: int = 1024, nhead: int = 8, pooling: bool = True,
                 local_transformer: bool = False, **dec_kwargs):
        super().__init__()
        self.pooling = pooling and not local_transformer
        self.local_transformer = local_transformer
        self.transformer = InterpretableTransformerEncoder(
            d_model=d_model, nhead=nhead, dim_feedforward=hidden_size,
            batch_first=True)
        if local_transformer:
            self.cls_token = nn.Parameter(torch.empty(1, d_model))
            nn.init.xavier_normal_(self.cls_token)
        elif self.pooling:
            self.dec = DECPool(n_clusters=n_out_nodes, embed_dim=d_model,
                               n_in_nodes=n_in_nodes, **dec_kwargs)

    def forward(self, x):  # [B, N, F]
        bz = x.size(0)
        if self.local_transformer:
            cls = self.cls_token.unsqueeze(0).expand(bz, -1, -1)  # [B, 1, F]
            x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        if self.local_transformer:
            cls_out, x = x[:, 0:1, :], x[:, 1:, :]
            return x, None, cls_out
        if self.pooling:
            x_out, assignment = self.dec(x)
            return x_out, assignment, None
        return x, None, None

    def dec_loss(self, assignment):
        if not self.pooling or assignment is None:
            return torch.zeros((), device=assignment.device if assignment is not None else 'cpu')
        return self.dec.dec_loss(assignment)


class ComBrainTF(nn.Module):
    """Community-aware brain transformer.

    Args:
        n_roi: number of ROIs.
        nclass: classes (or 1 for regression).
        community_ids: list[int] of length n_roi mapping ROI -> community id
                       (must be contiguous 0..n_communities-1, sorted by id
                       so ROIs in the same community are adjacent).
        n_communities: K. If None, inferred from community_ids.
        n_clusters: DEC output cluster count (paper: 8 == n_communities).
        hidden_size: transformer feedforward width (paper: 1024).
        nhead: attention heads (paper: 8).
        task: 'classification' or 'regression'.
    """

    def __init__(self, n_roi: int, nclass: int, community_ids: List[int],
                 n_communities: Optional[int] = None, n_clusters: int = 8,
                 hidden_size: int = 1024, nhead: int = 8,
                 task: str = "classification"):
        super().__init__()
        self.task = task
        self.n_roi = n_roi
        if n_communities is None:
            n_communities = len(set(community_ids))
        self.n_communities = n_communities

        # ROI permutation: group ROIs by community, then count community sizes
        order = sorted(range(n_roi), key=lambda i: community_ids[i])
        self.register_buffer('rearranged_idx',
                             torch.as_tensor(order, dtype=torch.long))
        sizes = [0] * n_communities
        for cid in community_ids:
            sizes[cid] += 1
        cum = []
        s = 0
        for sz in sizes:
            s += sz
            cum.append(s)
        self.community_sizes = sizes
        self.community_cum = cum  # community k spans [cum[k-1], cum[k])

        d_model = n_roi  # node features = FC row
        # One shared local transformer per community (matches original: same
        # weights, but a CLS token per community). We use a ModuleList of
        # n_communities small encoders to keep per-community CLS distinct.
        self.local_transformers = nn.ModuleList([
            TransPoolingEncoder(d_model=d_model, n_in_nodes=sz, n_out_nodes=sz,
                                hidden_size=hidden_size, nhead=nhead,
                                local_transformer=True)
            for sz in sizes
        ])
        # CLS token aggregator: [B, K*F] -> [B, F]
        self.cls_mlp = nn.Sequential(
            nn.Linear(n_communities * d_model, max(hidden_size // 2, 256)),
            nn.LeakyReLU(),
            nn.Linear(max(hidden_size // 2, 256), d_model),
            nn.LeakyReLU(),
        )
        # Global transformer with DEC pool (n_roi+1 -> n_clusters)
        self.global_transformer = TransPoolingEncoder(
            d_model=d_model, n_in_nodes=n_roi + 1, n_out_nodes=n_clusters,
            hidden_size=hidden_size, nhead=nhead, pooling=True,
            orthogonal=True, freeze_center=True, project_assignment=True)
        self.dim_reduce = nn.Sequential(nn.Linear(d_model, 8), nn.LeakyReLU())
        out_dim = nclass if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(8 * n_clusters, 256), nn.LeakyReLU(),
            nn.Linear(256, 32), nn.LeakyReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):  # x: [B, N, N]
        bz = x.size(0)
        # Rearrange rows AND columns by community (FC is symmetric)
        idx = self.rearranged_idx
        x = x[:, idx, :][:, :, idx]

        # Per-community local transformer + collect CLS tokens
        cls_tokens = []
        out_pieces = []
        prev = 0
        for k, end in enumerate(self.community_cum):
            sub = x[:, prev:end, :]
            sub_out, _, cls = self.local_transformers[k](sub)
            cls_tokens.append(cls)
            out_pieces.append(sub_out)
            prev = end
        x = torch.cat(out_pieces, dim=1)  # [B, N, F]
        cls_cat = torch.cat(cls_tokens, dim=1).reshape(bz, -1)  # [B, K*F]
        global_cls = self.cls_mlp(cls_cat).reshape(bz, 1, -1)
        x = torch.cat([global_cls, x], dim=1)  # [B, N+1, F]

        x_out, assignment, _ = self.global_transformer(x)  # [B, K_clusters, F]
        x_out = self.dim_reduce(x_out).reshape(bz, -1)
        return self.head(x_out), assignment

    def dec_loss(self, assignment):
        return self.global_transformer.dec_loss(assignment)
