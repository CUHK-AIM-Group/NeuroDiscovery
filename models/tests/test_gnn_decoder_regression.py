import pytest
import torch

from models.kg_link_prediction.gnn import DirectionalDecoder, GNNLinkPredictor
from models.kg_link_prediction.train import _filtered_ranking_metrics


def test_grouped_decoder_matches_reference_values_and_gradients():
    torch.manual_seed(4)
    decoder = DirectionalDecoder(3, 4).double()
    source = torch.randn(9, 4, dtype=torch.double, requires_grad=True)
    target = torch.randn(9, 4, dtype=torch.double, requires_grad=True)
    relations = torch.tensor([0, 2, 0, 1, 2, 2, 1, 0, 1])
    result = decoder(source, relations, target)
    reference = torch.einsum("bd,bde,be->b", source, decoder.bilinear[relations], target) + decoder.bias[relations]
    assert torch.allclose(result, reference)
    parameters = (source, target, decoder.bilinear, decoder.bias)
    actual_gradients = torch.autograd.grad(result.sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(reference.sum(), parameters)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual, expected)


def test_grouped_decoder_accepts_empty_batch():
    decoder = DirectionalDecoder(2, 4)
    assert decoder(torch.empty(0, 4), torch.empty(0, dtype=torch.long), torch.empty(0, 4)).shape == (0,)


def test_ranking_uses_directional_decoder_not_unused_distmult_embedding():
    model = GNNLinkPredictor(3, 1, embedding_dim=2, layers=1, decoder="directional")
    embeddings = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    with torch.no_grad():
        model.decoder.bilinear.zero_()
        model.decoder.bilinear[0, 0, 1] = 5
        model.relation_embedding.weight.zero_()
    result = _filtered_ranking_metrics(model, embeddings, torch.tensor([[0, 0, 1]]), set())
    assert result["mrr"] == 1
    with torch.no_grad():
        model.decoder.bilinear.zero_()
    result = _filtered_ranking_metrics(model, embeddings, torch.tensor([[0, 0, 1]]), set())
    assert result["mrr"] == pytest.approx(0.5)


def test_ranking_no_test_rows_is_not_zero_accuracy():
    model = GNNLinkPredictor(3, 1, embedding_dim=2, layers=1)
    result = _filtered_ranking_metrics(model, torch.zeros(3, 2), torch.empty(0, 3, dtype=torch.long), set())
    assert result["mrr"] is None
