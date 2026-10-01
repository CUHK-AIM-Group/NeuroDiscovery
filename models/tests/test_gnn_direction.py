"""Directional link-prediction behaviour for the relation-aware GNN.

These tests use small synthetic graphs and deterministic weights. They assert
representational facts about the decoder, not scientific validity, novelty or
the quality of a trained model.
"""

from __future__ import annotations

import pytest
import torch

from models.kg_link_prediction.gnn import (
    GNNLinkPredictor,
    directed_message_graph,
    message_graph,
)


def _model(**kwargs) -> GNNLinkPredictor:
    torch.manual_seed(0)
    return GNNLinkPredictor(4, 2, model="rgcn", embedding_dim=8, layers=1, **kwargs)


def test_distmult_cannot_separate_a_directed_relation():
    """Documents the defect the directional decoder exists to fix."""
    model = _model()
    embeddings = model.entity_embedding.weight.detach()
    forward = torch.tensor([[0, 1, 2]])
    reverse = torch.tensor([[2, 1, 0]])
    with torch.no_grad():
        model.eval()
        assert torch.allclose(model.score(embeddings, forward), model.score(embeddings, reverse))


def test_directional_decoder_separates_the_two_endpoint_orders():
    model = _model(decoder="directional")
    embeddings = model.entity_embedding.weight.detach()
    forward = torch.tensor([[0, 1, 2]])
    reverse = torch.tensor([[2, 1, 0]])
    with torch.no_grad():
        model.eval()
        assert not torch.allclose(model.score(embeddings, forward), model.score(embeddings, reverse))


def test_unknown_decoder_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError):
        _model(decoder="transformer")


def test_reverse_relations_allocate_separate_message_slots():
    triples = torch.tensor([[0, 0, 1], [1, 1, 2]])
    edge_index, edge_type = directed_message_graph(triples, n_relations=2)
    # Forward edges first, then reversed edges offset by n_relations.
    assert edge_type.tolist() == [0, 1, 2, 3]
    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]


def test_directed_message_graph_forward_pass_uses_all_relation_slots():
    triples = torch.tensor([[0, 0, 1], [1, 1, 2]])
    edge_index, edge_type = directed_message_graph(triples, n_relations=2)
    model = _model(reverse_relations=True).eval()
    with torch.no_grad():
        output = model(edge_index, edge_type, triples)
    assert output.shape == (2,)
    assert bool(torch.isfinite(output).all())


def test_defaults_preserve_the_historical_symmetric_setup():
    triples = torch.tensor([[0, 0, 1], [1, 1, 2]])
    edge_index, edge_type = message_graph(triples)
    assert edge_type.tolist() == [0, 1, 0, 1]
    model = _model().eval()
    assert model.decoder is None
    with torch.no_grad():
        assert model(edge_index, edge_type, triples).shape == (2,)


def test_directional_decoder_is_trainable_on_an_asymmetric_task():
    """Bounded trainability check: the asymmetric term can actually be learned."""
    torch.manual_seed(0)
    model = GNNLinkPredictor(3, 1, model="rgcn", embedding_dim=8, layers=1, decoder="directional")
    positives = torch.tensor([[0, 0, 1], [1, 0, 2]])
    negatives = torch.tensor([[1, 0, 0], [0, 0, 2]])

    def forward_loss(model: GNNLinkPredictor) -> torch.Tensor:
        embeddings = model.encode(*message_graph(torch.cat([positives, negatives])))
        positive_score = model.score(embeddings, positives)
        negative_score = model.score(embeddings, negatives)
        return (
            torch.nn.functional.binary_cross_entropy_with_logits(positive_score, torch.ones(len(positives)))
            + torch.nn.functional.binary_cross_entropy_with_logits(negative_score, torch.zeros(len(negatives)))
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = float(forward_loss(model))
    for _ in range(300):
        optimizer.zero_grad()
        batch_loss = forward_loss(model)
        batch_loss.backward()
        optimizer.step()

    assert float(forward_loss(model)) < initial
