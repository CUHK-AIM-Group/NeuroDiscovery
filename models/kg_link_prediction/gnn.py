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
    ):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, RGCNConv, SAGEConv
        except ImportError as exc:
            raise RuntimeError("torch-geometric is required for GNN link prediction") from exc
        self.model_name = model.lower().replace("-", "_")
        self.entity_embedding = nn.Embedding(n_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(n_relations, embedding_dim)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            if self.model_name == "rgcn":
                self.layers.append(RGCNConv(embedding_dim, embedding_dim, n_relations))
            elif self.model_name in {"graphsage", "sage"}:
                self.layers.append(SAGEConv(embedding_dim, embedding_dim))
            elif self.model_name == "gat":
                self.layers.append(
                    GATConv(embedding_dim, embedding_dim, heads=1, concat=False)
                )
            else:
                raise ValueError(f"Unknown GNN link model: {model}")
        self.dropout = nn.Dropout(dropout)
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
        relation = self.relation_embedding(triples[:, 1])
        target = embeddings[triples[:, 2]]
        return (source * relation * target).sum(dim=-1)

    def forward(self, edge_index, edge_type, triples):
        return self.score(self.encode(edge_index, edge_type), triples)


def message_graph(train_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forward = train_ids[:, [0, 2]].T
    reverse = train_ids[:, [2, 0]].T
    edge_index = torch.cat([forward, reverse], dim=1)
    edge_type = torch.cat([train_ids[:, 1], train_ids[:, 1]], dim=0)
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
