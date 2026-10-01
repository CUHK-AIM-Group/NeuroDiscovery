import numpy as np
import pytest
import torch

from models.kg_link_prediction.full_graph_data import Union, split_records, join_version_groups
from models.kg_link_prediction.gnn import GNNLinkPredictor, directed_message_graph
from models.kg_link_prediction.sparse_rgcn import (encode_sparse, relation_blocks,
                                                  sample_typed_negatives, triple_keys)


def test_source_pair_groups_protected_proxies_and_no_fold_moving():
    records = [(0,0,1,'p1'), (1,1,0,'p2'), (2,0,3,'fixed'), (4,0,5,'proxy')]
    triples, fold, groups, excluded, counts = split_records(records, {'fixed'}, {(4,5)})
    assert len(triples) == 3
    assert groups['p1'] == groups['p2']
    assert groups['fixed'] == 0
    assert excluded == {'protected_endpoint_proxy': 1}
    # Adding a duplicate edge does not increase the retained training examples.
    duplicate = split_records(records + [records[0]], {'fixed'}, {(4,5)})
    np.testing.assert_array_equal(duplicate[0], triples)


def test_known_potential_versions_join_before_source_and_pair_splits():
    groups=Union()
    join_version_groups(groups,[('abstract-a','paper1'),('abstract-a','paper2'),('abstract-b','paper2'),('abstract-b','paper3')])
    assert groups.find('paper1')==groups.find('paper2')==groups.find('paper3')
    records=[(0,0,1,groups.find('paper1')),(2,0,3,groups.find('paper3'))]
    _,fold,_,_,_=split_records(records,set(),set())
    assert fold[0]==fold[1]


@pytest.mark.parametrize('checkpoints', [False, True])
@pytest.mark.parametrize('dropout', [0., .1])
def test_sparse_rgcn_matches_pyg_values_and_all_gradients(checkpoints,dropout):
    torch.manual_seed(27)
    model = GNNLinkPredictor(8,3,'rgcn',8,2,dropout,'directional',True)
    edges = torch.tensor([[0,0,1],[2,0,1],[1,1,3],[2,2,4],[3,2,4],[4,1,7]])
    graph, kinds = directed_message_graph(edges, 3)
    blocks = relation_blocks(edges.numpy(), 8, 3, 'cpu')
    torch.manual_seed(91)
    expected = model.encode(graph, kinds)
    loss = model.score(expected, edges).sum()
    loss.backward()
    gradients = {name: parameter.grad.clone() for name,parameter in model.named_parameters() if parameter.grad is not None}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(91)
    actual = encode_sparse(model, blocks, checkpoint_layers=checkpoints)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    model.score(actual, edges).sum().backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-6, rtol=1e-5)


def test_typed_sampling_avoids_all_recorded_edges_and_untrained_nodes():
    edges = np.array([[0,0,2],[1,0,3],[0,1,3],[1,1,2]])
    domains = np.array([0,0,1,1,1,0])
    trained = np.array([True,True,True,True,True,False])
    keys = np.sort(triple_keys(edges, len(domains), 2))
    out = sample_typed_negatives(np.tile(edges,(15,1)), domains, trained, keys, 2, np.random.default_rng(9))
    assert not np.isin(triple_keys(out,len(domains),2), keys).any()
    assert trained[out[:,[0,2]]].all()
    np.testing.assert_array_equal(domains[out[:,0]], np.tile(domains[edges[:,0]],15))
    np.testing.assert_array_equal(domains[out[:,2]], np.tile(domains[edges[:,2]],15))


def test_decoder_chunking_preserves_full_loss_gradients():
    torch.manual_seed(22)
    model = GNNLinkPredictor(8,3,'rgcn',8,2,0.,'directional',True)
    positive = torch.tensor([[0,0,1],[2,0,1],[1,1,3],[2,2,4],[3,2,4],[4,1,7]])
    negative = positive.clone()
    negative[:,2] = (negative[:,2]+2)%8
    blocks = relation_blocks(positive.numpy(),8,3,'cpu')
    z = encode_sparse(model,blocks)
    loss = (torch.nn.functional.softplus(-model.score(z,positive)).sum()
            +torch.nn.functional.softplus(model.score(z,negative)).sum())/len(positive)
    loss.backward()
    expected = {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    z = encode_sparse(model,blocks,checkpoint_layers=True)
    leaf = z.detach().requires_grad_(True)
    for start in range(0,len(positive),2):
        loss = (torch.nn.functional.softplus(-model.score(leaf,positive[start:start+2])).sum()
                +torch.nn.functional.softplus(model.score(leaf,negative[start:start+2])).sum())/len(positive)
        loss.backward()
    z.backward(leaf.grad)
    for name,p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad,expected[name],atol=2e-6,rtol=1e-5)


def test_saturated_type_keeps_positive_without_fake_contrast():
    positive=np.array([[0,0,1]])
    domains=np.array([0,1])
    mask=np.ones(2,dtype=bool)
    keys=np.sort(triple_keys(positive,2,1))
    negative,valid=sample_typed_negatives(positive,domains,mask,keys,1,np.random.default_rng(9),allow_missing=True)
    assert len(negative)==len(positive) and not valid.any()
    with pytest.raises(ValueError,match='No type-matched'):
        sample_typed_negatives(positive,domains,mask,keys,1,np.random.default_rng(9))
