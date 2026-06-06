"""BrainNetworkTransformer reimplementation (Wayfear 2022).

Ports source/models/BNT/ with DEC pooling + interpretable transformer encoder,
but simplified to work without hydra/omegaconf — just pass in plain args.

Architecture (matches the paper):
  Input: FC matrix  [B, N, N]  (Pearson rows as node features)
       + optional learnable positional embedding [B, N, P]  -> [B, N, F]
       where F = N (+ P if pos_embed enabled)
  Stack of TransPoolingEncoder:
       - Transformer encoder layer (self-attention with nhead=4)
       - Optional DEC pooling: cluster N_in nodes into N_out clusters
  Dim reduction: Linear(F -> 8) + LeakyReLU
  Head: Flatten -> Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear(->nclass or 1)

Source: https://github.com/Wayfear/BrainNetworkTransformer (MIT)
"""
from __future__ import annotations

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


# -------------------- Interpretable transformer --------------------
class InterpretableTransformerEncoder(TransformerEncoderLayer):
    """Same as TransformerEncoderLayer but stores attention weights."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=True,
                 norm_first=False, device=None, dtype=None):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[torch.Tensor] = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x_out, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )
        self.attention_weights = weights
        return self.dropout1(x_out)

    def get_attention_weights(self):
        return self.attention_weights


# -------------------- DEC pooling --------------------
class ClusterAssignment(nn.Module):
    """Soft cluster assignment with optional orthogonal init + projection mode."""

    def __init__(self, cluster_number: int, embedding_dim: int, alpha: float = 1.0,
                 orthogonal: bool = True, freeze_center: bool = False,
                 project_assignment: bool = True):
        super().__init__()
        self.cluster_number = cluster_number
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.project_assignment = project_assignment

        centers = torch.zeros(cluster_number, embedding_dim, dtype=torch.float)
        nn.init.xavier_uniform_(centers)
        if orthogonal:
            ortho = torch.zeros_like(centers)
            ortho[0] = centers[0]
            for i in range(1, cluster_number):
                proj = torch.zeros_like(centers[0])
                for j in range(i):
                    proj = proj + self._project(centers[j], centers[i])
                centers[i] = centers[i] - proj
                ortho[i] = centers[i] / centers[i].norm(p=2).clamp(min=1e-8)
            centers = ortho
        self.cluster_centers = nn.Parameter(centers, requires_grad=not freeze_center)

    @staticmethod
    def _project(u, v):
        return (torch.dot(u, v) / torch.dot(u, u).clamp(min=1e-8)) * u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: [B*N, D]
        if self.project_assignment:
            assign = batch @ self.cluster_centers.T          # [B*N, K]
            assign = assign.pow(2)
            norm = self.cluster_centers.norm(p=2, dim=-1)    # [K]
            assign = assign / norm.clamp(min=1e-8)
            return F.softmax(assign, dim=-1)
        # Student's t-distribution (original DEC)
        norm_sq = torch.sum(
            (batch.unsqueeze(1) - self.cluster_centers) ** 2, dim=2
        )
        numer = 1.0 / (1.0 + norm_sq / self.alpha)
        numer = numer ** ((self.alpha + 1) / 2)
        return numer / numer.sum(dim=1, keepdim=True).clamp(min=1e-8)


class DEC(nn.Module):
    """Deep Embedded Clustering pooling module."""

    def __init__(self, cluster_number: int, hidden_dim: int, encoder: nn.Module,
                 orthogonal: bool = True, freeze_center: bool = False,
                 project_assignment: bool = True):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.cluster_number = cluster_number
        self.assignment = ClusterAssignment(
            cluster_number, hidden_dim,
            orthogonal=orthogonal, freeze_center=freeze_center,
            project_assignment=project_assignment,
        )

    def forward(self, batch: torch.Tensor):
        # batch: [B, N, D]
        B, N, _ = batch.shape
        flat = batch.reshape(B, -1)               # [B, N*D]
        encoded = self.encoder(flat)              # [B, N*D_hidden] actually same shape
        encoded = encoded.view(B * N, -1)         # [B*N, D]
        assign = self.assignment(encoded)         # [B*N, K]
        assign = assign.view(B, N, -1)            # [B, N, K]
        encoded = encoded.view(B, N, -1)          # [B, N, D]
        # Soft pooling: cluster membership weighted sum
        node_repr = torch.bmm(assign.transpose(1, 2), encoded)  # [B, K, D]
        return node_repr, assign

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        w = q.pow(2) / q.sum(dim=0).clamp(min=1e-8)
        return (w.t() / w.sum(dim=1).clamp(min=1e-8)).t()

    def loss(self, assignment: torch.Tensor) -> torch.Tensor:
        # KL(P || Q) with P = target, Q = assignment
        flat = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flat).detach()
        return F.kl_div(flat.clamp(min=1e-10).log(), target, reduction="batchmean")


# -------------------- TransPoolingEncoder --------------------
class TransPoolingEncoder(nn.Module):
    """Transformer encoder with optional DEC pooling."""

    def __init__(self, input_feature_size: int, input_node_num: int,
                 hidden_size: int, output_node_num: int,
                 pooling: bool = True, orthogonal: bool = True,
                 freeze_center: bool = False, project_assignment: bool = True,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=nhead,
            dim_feedforward=hidden_size, dropout=dropout,
            batch_first=True,
        )
        self.pooling = pooling
        if pooling:
            encoder_hidden = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden, encoder_hidden),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden, input_feature_size * input_node_num),
            )
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dim=input_feature_size,
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
            )

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


# -------------------- Main model --------------------
class BrainNetworkTransformer(nn.Module):
    """BNT head.

    Args:
        n_roi: number of ROIs (nodes)
        sizes: cluster counts for successive TransPoolingEncoder layers.
            Typical: [100, 20] for a 100-ROI atlas (2-layer with pooling).
            If pooling=False for all layers, sizes are ignored (shape unchanged).
        do_pooling: bool per layer; must have same length as `sizes`.
        pos_encoding: None | "identity" (learned positional embedding)
        pos_embed_dim: dim of identity positional embedding (if used)
        nclass: output dim. Use 1 for regression, >=2 for classification.
        task: "classification" | "regression"
    """

    def __init__(self, n_roi: int, sizes: List[int] = None,
                 do_pooling: List[bool] = None,
                 pos_encoding: Optional[str] = "identity",
                 pos_embed_dim: int = 8,
                 orthogonal: bool = True, freeze_center: bool = False,
                 project_assignment: bool = True,
                 hidden_size: int = 1024, nhead: int = 4, dropout: float = 0.1,
                 nclass: int = 2, task: str = "classification"):
        super().__init__()
        if sizes is None:
            # Paper uses [100, 20] for ABIDE (~200 ROI). Scale by ROI count:
            sizes = [max(32, n_roi // 2), max(8, n_roi // 10)]
        if do_pooling is None:
            do_pooling = [True] * len(sizes)
        assert len(sizes) == len(do_pooling)
        assert task in ("classification", "regression")

        self.task = task
        self.n_roi = n_roi
        self.pos_encoding = pos_encoding
        forward_dim = n_roi
        if pos_encoding == "identity":
            self.node_identity = nn.Parameter(
                torch.zeros(n_roi, pos_embed_dim), requires_grad=True
            )
            nn.init.kaiming_normal_(self.node_identity)
            forward_dim = n_roi + pos_embed_dim
        else:
            self.node_identity = None

        # Auto-adjust nhead so that forward_dim is divisible.
        # Prefer the requested nhead if compatible, otherwise fall back to
        # largest divisor of forward_dim that is <= requested nhead.
        if forward_dim % nhead != 0:
            for h in range(nhead, 0, -1):
                if forward_dim % h == 0:
                    nhead = h
                    break
        self.nhead = nhead

        # The first encoder always takes n_roi nodes; sizes[0] is the first pool target.
        in_sizes = [n_roi] + sizes[:-1]
        self.attention_list = nn.ModuleList([
            TransPoolingEncoder(
                input_feature_size=forward_dim,
                input_node_num=in_sizes[idx],
                hidden_size=hidden_size,
                output_node_num=sizes[idx],
                pooling=do_pooling[idx],
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
                nhead=nhead, dropout=dropout,
            )
            for idx in range(len(sizes))
        ])

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU(),
        )

        final_nodes = sizes[-1] if do_pooling[-1] else in_sizes[-1]
        self.fc = nn.Sequential(
            nn.Linear(8 * final_nodes, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, nclass),
        )

    def forward(self, node_feature: torch.Tensor):
        """
        node_feature: [B, N, N]  -- Pearson correlation matrix per subject
        Returns:
            logits: [B, nclass] (raw logits for regression, logits for cls)
            assignments: list of [B, N, K] or None per layer (for DEC loss)
        """
        B = node_feature.size(0)
        x = node_feature
        if self.pos_encoding == "identity":
            pos_emb = self.node_identity.expand(B, *self.node_identity.shape)
            x = torch.cat([x, pos_emb], dim=-1)  # [B, N, N+P]

        assignments = []
        for layer in self.attention_list:
            x, assignment = layer(x)
            assignments.append(assignment)

        x = self.dim_reduction(x)            # [B, K_last, 8]
        x = x.reshape(B, -1)                 # [B, 8 * K_last]
        logits = self.fc(x)
        return logits, assignments

    def dec_loss(self, assignments) -> torch.Tensor:
        """Sum of DEC KL losses across pooling layers. 0 if no pooling."""
        total = None
        for layer, assign in zip(self.attention_list, assignments):
            if not layer.is_pooling_enabled() or assign is None:
                continue
            term = layer.loss(assign)
            total = term if total is None else total + term
        if total is None:
            return torch.tensor(0.0,
                                device=next(self.parameters()).device)
        return total
