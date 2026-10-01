"""Relation-aware GNN encoders with a DistMult decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class TripleIndex:
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]

    @classmethod
    def from_triples(cls, triples):
        entities = sorted(
            {value for triple in triples for value in (triple.source_id, triple.target_id)}
        )
        relations = sorted({triple.relation_type for triple in triples})
        return cls(
            {value: index for index, value in enumerate(entities)},
            {value: index for index, value in enumerate(relations)},
        )

    def encode(self, triples) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    self.entity_to_id[triple.source_id],
                    self.relation_to_id[triple.relation_type],
                    self.entity_to_id[triple.target_id],
                ]
                for triple in triples
            ],
            dtype=torch.long,
        )


class GNNLinkPredictor(nn.Module):
    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        model: str = "rgcn",
        embedding_dim: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        decoder: str = "distmult",
        reverse_relations: bool = False,
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, RGCNConv, SAGEConv
        except ImportError as exc:
            raise RuntimeError("torch-geometric is required for GNN link prediction") from exc
        self.model_name = model.lower().replace("-", "_")
        # Message-passing relation slots. With `reverse_relations` the reversed
        # edges of `directed_message_graph` get their own type ids (offset by
        # n_relations) so a relation and its reverse are not the same message.
        self.relation_types = n_relations * 2 if reverse_relations else n_relations
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            if self.model_name == "rgcn":
                self.layers.append(RGCNConv(embedding_dim, embedding_dim, self.relation_types))
            elif self.model_name in {"graphsage", "sage"}:
                self.layers.append(SAGEConv(embedding_dim, embedding_dim))
            elif self.model_name == "gat":
                self.layers.append(
                    GATConv(embedding_dim, embedding_dim, heads=1, concat=False)
                )
            else:
                raise ValueError(f"Unknown GNN link model: {model}")
        self.dropout = nn.Dropout(dropout)
        self.decoder_name = decoder.lower().replace("-", "_")
        if self.decoder_name in {"distmult", "default"}:
            self.decoder = None
        elif self.decoder_name in {"directional", "bilinear", "rescal"}:
            self.decoder = DirectionalDecoder(n_relations, embedding_dim)
        else:
            raise ValueError(f"Unknown decoder: {decoder}")
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        x = self.entity_embedding.weight
        for layer in self.layers:
            if self.model_name == "rgcn":
                x = layer(x, edge_index, edge_type)
            else:
                x = layer(x, edge_index)
            x = self.dropout(torch.relu(x))
        return x

    def score(self, embeddings, triples: torch.Tensor):
        source = embeddings[triples[:, 0]]
        target = embeddings[triples[:, 2]]
        if self.decoder is not None:
            return self.decoder(source, triples[:, 1], target)
        relation = self.relation_embedding(triples[:, 1])
        return (source * relation * target).sum(dim=-1)

    def forward(self, edge_index, edge_type, triples):
        return self.score(self.encode(edge_index, edge_type), triples)


class DirectionalDecoder(nn.Module):
    """Asymmetric relation decoder: `score = source^T M_relation target + bias`.

    DistMult is symmetric under endpoint reversal (`s·(r·t) == t·(r·s)`), so it
    cannot represent a directed scientific relation: it scores `A causes B` and
    `B causes A` identically. A relation-specific, unconstrained bilinear matrix
    (RESCAL-style) is not symmetric, so the direction is representable. The
    matrices are initialised near zero with asymmetric random values, so an
    untrained model already separates the two directions instead of starting
    from a symmetric point.
    """

    def __init__(self, n_relations: int, embedding_dim: int):
        super().__init__()
        self.bilinear = nn.Parameter(torch.empty(n_relations, embedding_dim, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(n_relations))
        nn.init.xavier_uniform_(self.bilinear)
        # Break the symmetric initial point; xavier alone can be near symmetric
        # for a square matrix without biasing magnitude.
        with torch.no_grad():
            self.bilinear.add_(0.01 * torch.randn_like(self.bilinear))

    def forward(self, source: torch.Tensor, relation_index: torch.Tensor, target: torch.Tensor):
        scores = source.new_zeros(source.shape[0])
        for relation_id in torch.unique(relation_index):
            positions = torch.where(relation_index == relation_id)[0]
            transformed = source[positions] @ self.bilinear[relation_id]
            values = (transformed * target[positions]).sum(dim=-1) + self.bias[relation_id]
            scores = scores.index_copy(0, positions, values)
        return scores


def message_graph(train_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward = train_ids[:, [0, 2]].T
    reverse = train_ids[:, [2, 0]].T
    edge_index = torch.cat([forward, reverse], dim=1)
    edge_type = torch.cat([train_ids[:, 1], train_ids[:, 1]], dim=0)
    return edge_index, edge_type


def directed_message_graph(
    train_ids: torch.Tensor, n_relations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Message graph whose reversed edges carry distinct relation type ids.

    `message_graph` reuses the forward relation type for the reversed edge, so
    an R-GCN cannot tell `A → B` from `B → A`. Here the reverse edge is offset by
    `n_relations` (the number of forward relation types), which requires the
    model to be built with `reverse_relations=True` so it has 2×n_relations slots.
    """

    forward = train_ids[:, [0, 2]].T
    reverse = train_ids[:, [2, 0]].T
    edge_index = torch.cat([forward, reverse], dim=1)
    edge_type = torch.cat([train_ids[:, 1], train_ids[:, 1] + n_relations], dim=0)
    return edge_index, edge_type


def sample_negatives(
    positive: torch.Tensor,
    n_entities: int,
    known: set[tuple[int, int, int]],
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    rows = []
    for source, relation, target in positive.tolist():
        for _ in range(negatives_per_positive):
            for _attempt in range(100):
                if rng.random() < 0.5:
                    candidate = (int(rng.integers(n_entities)), relation, target)
                else:
                    candidate = (source, relation, int(rng.integers(n_entities)))
                if candidate not in known:
                    rows.append(candidate)
                    break
    return torch.tensor(rows, dtype=torch.long)
